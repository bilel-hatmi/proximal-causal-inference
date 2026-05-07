"""
DGP2 Trade-off Exploration (Screening) -- Phase 9B.3

Context
-------
The DGP2 tuning confirmation (Phase 9B.2bis, M=200) delivered:
  - Config A (lambda_h=1e-4, ell_scale=2.0): -55% bias_REG, coverage 0.26 -> 0.76
  - Plateau at cov ~ 0.76 = RKHS-rate bound (rho=0.246)
  - q-tuning alone: no effect
  - policy-smoothing alone: worsens coverage (tighter CI without bias reduction)

This script maps the trade-off surface around the successful h-tuning region using
small M=50 Monte Carlo screening (not confirmation). Four sequential blocks:

    Block H  h-bridge local refinement: lambda_h x ell_scale grid around config A
    Block Q  q-tuning conditional on best h from Block H: lambda_Q x clip grid
    Block P  policy bandwidth trade-off: h_policy_scale sweep on best h config
    Block N  n-trend confirmation: n in {500,1000,2000}, M=75, selected configs

No block auto-promotes to the next. The user reads each report section and selects
configs manually via CLI before running the next block.

Modes
-----
    quick    n=500, M=5  : smoke-test pipeline (Block H: 2 configs; Block P: hpol {1.0,1.5})
    screen   n=1000, M=50: main screening per block
    confirm  n in {500,1000,2000}, M=75: Block N only

CLI
---
    --block {H, Q, P, N, all}           required
    --mode {quick, screen, confirm}      default: screen
    --h-configs "id1,id2"               for Block Q (comma-separated BLOCK_H IDs)
    --h-config "id"                      for Block P
    --n-configs "id1,id2,..."           for Block N
    --force                             redo blocks even if pkl exists

Output
------
    simulations/results/raw/dgp2_tradeoff/dgp2tradeoff_{block}_{config_id}_n{n}_M{m}.pkl
    simulations/results/summaries/dgp2_tradeoff_exploration.md

Reuse
-----
Imports helpers from:
    simulations/experiments/dgp2_bias_diagnostics.py   (core MC helpers + constants)
    simulations/experiments/dgp2_tuning_confirmation.py  (make_factory + analytical helpers)

No modification to any existing estimator or DGP code.
"""
from __future__ import annotations

import argparse
import datetime
import pickle
import re
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# ── Reuse helpers + constants from dgp2_bias_diagnostics ──────────────────────
from simulations.experiments.dgp2_bias_diagnostics import (
    _silverman_h,
    _compute_J_policy_true,
    _median_bandwidth,
    _run_block,
    _build_record,
    _decompose,
    _pred_cov,
    _fmt,
    A_GRID,
    K,
    REF_IDX,
    SNR,
    CHECKPOINT_FREQ,
    LAMBDA_Q_DEFAULT,
    CLIP_DEFAULT,
    N_FOLDS,
    RANDOM_STATE,
)

# ── Reuse factory + analytical helpers from dgp2_tuning_confirmation ──────────
from simulations.archive.utilities.dgp2_tuning_confirmation import (
    make_factory,
    _compute_ser,
    _per_dose_empirical_coverage,
    _per_dose_se,
    _max_weight_max,
    _q_neg_share_mean,
    _M_ok_ratio,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR    = Path(__file__).resolve().parents[2]
_RAW_DIR     = _BASE_DIR / "simulations" / "results" / "raw" / "dgp2_tradeoff"
_SUMM_DIR    = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "dgp2_tradeoff_exploration.md"

# ── Mode definitions (n_list, M) ──────────────────────────────────────────────
_MODES: dict[str, tuple[list[int], int]] = {
    "quick":   ([500],            5),
    "screen":  ([1000],          50),
    "confirm": ([500, 1000, 2000], 75),
}

# ── Pareto classification thresholds (centralized) ────────────────────────────
_BIAS_RED_MIN    = 0.30   # |bias_DR| reduced >= 30% vs H_baseline
_W_P99_MAX_RATIO = 2.0   # w_p99 < 2x H_baseline w_p99
_ESS_MIN         = 20.0  # ESS_min_mean >= 20
_QCLIP_MAX       = 0.05  # q_clip_fraction < 0.05
_SER_LOW         = 0.75
_SER_HIGH        = 1.25
_BSE_STRONG      = 1.0   # |bias|/SE < 1.0 = "strong"

# q-correction activity threshold for Block Q
_CORR_RATIO_ACTIVE = 0.10  # if max |correction_ratio| < this, q is inactive


# ══════════════════════════════════════════════════════════════════════════════
#  Block H config table -- single source of truth
#  Grid: lambda_h in {3e-5, 1e-4, 3e-4} x ell_scale in {1.5, 2.0, 2.5} + baseline
# ══════════════════════════════════════════════════════════════════════════════

BLOCK_H_CONFIGS: dict[str, dict] = {
    "H_baseline": dict(
        lambda_h=None, ell_scale=1.0,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT,
        h_policy_scale=1.0,
        description="Reference: KPV defaults, no h-bridge tuning.",
    ),
    "H_lam3e-5_ell1.5": dict(
        lambda_h=3e-5, ell_scale=1.5,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT,
        h_policy_scale=1.0,
        description="lambda_h=3e-5, ell_scale=1.5.",
    ),
    "H_lam1e-4_ell1.5": dict(
        lambda_h=1e-4, ell_scale=1.5,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT,
        h_policy_scale=1.0,
        description="lambda_h=1e-4, ell_scale=1.5.",
    ),
    "H_lam3e-4_ell1.5": dict(
        lambda_h=3e-4, ell_scale=1.5,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT,
        h_policy_scale=1.0,
        description="lambda_h=3e-4, ell_scale=1.5.",
    ),
    "H_lam3e-5_ell2.0": dict(
        lambda_h=3e-5, ell_scale=2.0,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT,
        h_policy_scale=1.0,
        description="lambda_h=3e-5, ell_scale=2.0.",
    ),
    "H_lam1e-4_ell2.0": dict(  # = tuning_confirmation config A
        lambda_h=1e-4, ell_scale=2.0,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT,
        h_policy_scale=1.0,
        description="lambda_h=1e-4, ell_scale=2.0 (= confirmation config A).",
    ),
    "H_lam3e-4_ell2.0": dict(
        lambda_h=3e-4, ell_scale=2.0,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT,
        h_policy_scale=1.0,
        description="lambda_h=3e-4, ell_scale=2.0.",
    ),
    "H_lam3e-5_ell2.5": dict(
        lambda_h=3e-5, ell_scale=2.5,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT,
        h_policy_scale=1.0,
        description="lambda_h=3e-5, ell_scale=2.5.",
    ),
    "H_lam1e-4_ell2.5": dict(
        lambda_h=1e-4, ell_scale=2.5,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT,
        h_policy_scale=1.0,
        description="lambda_h=1e-4, ell_scale=2.5.",
    ),
    "H_lam3e-4_ell2.5": dict(
        lambda_h=3e-4, ell_scale=2.5,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT,
        h_policy_scale=1.0,
        description="lambda_h=3e-4, ell_scale=2.5.",
    ),
}

BLOCK_H_ORDER: list[str] = [
    "H_baseline",
    "H_lam3e-5_ell1.5", "H_lam1e-4_ell1.5", "H_lam3e-4_ell1.5",
    "H_lam3e-5_ell2.0", "H_lam1e-4_ell2.0", "H_lam3e-4_ell2.0",
    "H_lam3e-5_ell2.5", "H_lam1e-4_ell2.5", "H_lam3e-4_ell2.5",
]

# Quick mode: only baseline + config A (2 configs)
_BLOCK_H_QUICK: list[str] = ["H_baseline", "H_lam1e-4_ell2.0"]

# Block Q grids
_LQ_GRID: list[float]        = [3e-4, 1e-3, 3e-3, 1e-2]
_CLIP_GRID: list[Optional[float]] = [None, 20.0]

# Block P grid
_HPOL_GRID: list[float] = [0.75, 1.0, 1.25, 1.5]


# ══════════════════════════════════════════════════════════════════════════════
#  Config spec builders (Blocks Q and P)
# ══════════════════════════════════════════════════════════════════════════════

def make_block_q_spec(h_config_id: str, lambda_Q: float, clip: Optional[float]) -> dict:
    """Full spec dict for Block Q, inheriting h params from BLOCK_H_CONFIGS."""
    base = dict(BLOCK_H_CONFIGS[h_config_id])
    base["lambda_Q"] = float(lambda_Q)
    base["clip"] = clip
    clip_str = "None" if clip is None else f"{int(clip)}"
    base["description"] = (
        f"Block Q: {h_config_id}, lambda_Q={lambda_Q:.0e}, clip={clip_str}"
    )
    return base


def make_block_p_spec(h_config_id: str, h_policy_scale: float) -> dict:
    """Full spec dict for Block P, overriding h_policy_scale."""
    base = dict(BLOCK_H_CONFIGS[h_config_id])
    base["h_policy_scale"] = float(h_policy_scale)
    base["description"] = (
        f"Block P: {h_config_id}, h_policy_scale={h_policy_scale:.2f}"
    )
    return base


def _q_config_id(h_config_id: str, lambda_Q: float, clip: Optional[float]) -> str:
    clip_str = "None" if clip is None else f"{int(clip)}"
    return f"Q_{h_config_id}_lamQ{lambda_Q:.0e}_clip{clip_str}"


def _p_config_id(h_config_id: str, h_policy_scale: float) -> str:
    scale_str = f"{h_policy_scale:.2f}".replace(".", "p")
    return f"P_{h_config_id}_hpol{scale_str}"


# ══════════════════════════════════════════════════════════════════════════════
#  Seed scheme (disjoint from all prior runs)
# ══════════════════════════════════════════════════════════════════════════════
#  Prior scripts: Exp1A (5M-5.9M), bias_diag (6M-8.9M), kallus (7M range),
#  tuning_confirm (9M-9.9M). This script: 10M-13.9M.

_BLOCK_SEED_BASE: dict[str, int] = {
    "H": 10_000_000,
    "Q": 11_000_000,
    "P": 12_000_000,
    "N": 13_000_000,
}


def _seed(block_char: str, cfg_idx: int, n: int) -> int:
    """Seed = base + cfg_idx*10_000 + n. Disjoint per block and config."""
    return _BLOCK_SEED_BASE[block_char] + cfg_idx * 10_000 + int(n)


# ══════════════════════════════════════════════════════════════════════════════
#  PKL label + loader
# ══════════════════════════════════════════════════════════════════════════════

def _pkl_label(block_char: str, config_id: str, n: int, M: int) -> str:
    return f"dgp2tradeoff_{block_char}_{config_id}_n{n}_M{M}"


def _try_load_pkl(label: str) -> Optional[dict]:
    """Load pkl from _RAW_DIR if it exists; return None otherwise."""
    path = _RAW_DIR / f"{label}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Per-block runner (unified for all blocks)
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_block_tradeoff(
    dgp,
    config_id: str,
    spec: dict,
    block_char: str,
    cfg_idx: int,
    n: int,
    M: int,
    force: bool = False,
) -> tuple[Optional[dict], float, Optional[dict]]:
    """
    Run one (config x n) block.

    Returns (decomp_dict, wall_seconds, raw_data).
    raw_data is the pkl-loaded dict used for per-dose helpers in the report.

    h_ref and ell_* are anchored at seed=0 for reproducibility.
    J_policy_true is recomputed per h_policy_scale (Block P changes estimand).
    """
    ref_sample = dgp.generate(n=n, seed=0)
    h_ref  = _silverman_h(ref_sample.A)
    ell_W0 = _median_bandwidth(ref_sample.W)
    ell_A0 = _median_bandwidth(ref_sample.A)
    ell_Z0 = _median_bandwidth(ref_sample.Z)

    h_policy = h_ref * float(spec["h_policy_scale"])
    J_pt     = _compute_J_policy_true(dgp, A_GRID, h_policy)
    factory  = make_factory(spec, A_GRID, h_ref, ell_W0, ell_A0, ell_Z0)

    label     = _pkl_label(block_char, config_id, n, M)
    seed_base = _seed(block_char, cfg_idx, n)

    if force:
        for sfx in (".pkl", ".partial.pkl"):
            p = _RAW_DIR / f"{label}{sfx}"
            if p.exists():
                p.unlink()
                print(f"  [FORCE] removed {p.name}")

    t0   = time.perf_counter()
    data = _run_block(
        dgp, factory, A_GRID, J_pt,
        n=n, M=M, seed_base=seed_base, label=label, raw_dir=_RAW_DIR,
    )
    wall = time.perf_counter() - t0

    return _decompose(data), wall, data


def _print_brief(config_id: str, n: int, decomp: Optional[dict], wall: float) -> None:
    if decomp is not None:
        print(
            f"    bias_DR(ref)={decomp['bias_dr'][REF_IDX]:+.4f}  "
            f"|b|/SE={decomp['bse_grid'][REF_IDX]:.2f}  "
            f"cov={decomp['coverage_ref']:.3f}  "
            f"wall={wall/60:.1f}min"
        )
    else:
        print(
            f"    [WARN] No successful reps for {config_id} n={n} "
            f"(wall={wall/60:.1f}min)"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Block runners
# ══════════════════════════════════════════════════════════════════════════════

# Type alias for block results: config_id -> {n: (decomp, wall, raw_data)}
BlockResults = dict[str, dict[int, tuple[Optional[dict], float, Optional[dict]]]]


def run_block_H(
    dgp,
    n_list: list[int],
    M: int,
    config_ids: list[str],
    force: bool = False,
) -> BlockResults:
    """Run Block H for given config_ids x n_list."""
    results: BlockResults = {}
    for config_id in config_ids:
        results[config_id] = {}
        cfg_idx = BLOCK_H_ORDER.index(config_id)
        for n in n_list:
            print(f"\n--- Block H | {config_id} | n={n} | M={M} ---")
            spec = BLOCK_H_CONFIGS[config_id]
            decomp, wall, data = _run_one_block_tradeoff(
                dgp, config_id, spec, "H", cfg_idx, n, M, force=force,
            )
            results[config_id][n] = (decomp, wall, data)
            _print_brief(config_id, n, decomp, wall)
    return results


def run_block_Q(
    dgp,
    n_list: list[int],
    M: int,
    h_config_ids: list[str],
    force: bool = False,
) -> BlockResults:
    """Run Block Q: lambda_Q x clip grid for each parent h config."""
    results: BlockResults = {}
    q_configs: list[tuple[str, dict]] = []
    for h_id in h_config_ids:
        for lq in _LQ_GRID:
            for clip in _CLIP_GRID:
                cid  = _q_config_id(h_id, lq, clip)
                spec = make_block_q_spec(h_id, lq, clip)
                q_configs.append((cid, spec))

    for cfg_idx, (config_id, spec) in enumerate(q_configs):
        results[config_id] = {}
        for n in n_list:
            print(f"\n--- Block Q | {config_id} | n={n} | M={M} ---")
            decomp, wall, data = _run_one_block_tradeoff(
                dgp, config_id, spec, "Q", cfg_idx, n, M, force=force,
            )
            results[config_id][n] = (decomp, wall, data)
            _print_brief(config_id, n, decomp, wall)
    return results


def run_block_P(
    dgp,
    n_list: list[int],
    M: int,
    h_config_id: str,
    force: bool = False,
) -> BlockResults:
    """Run Block P: h_policy_scale sweep on the chosen h config."""
    results: BlockResults = {}
    p_configs: list[tuple[str, dict]] = []
    for hpol in _HPOL_GRID:
        cid  = _p_config_id(h_config_id, hpol)
        spec = make_block_p_spec(h_config_id, hpol)
        p_configs.append((cid, spec))

    for cfg_idx, (config_id, spec) in enumerate(p_configs):
        results[config_id] = {}
        for n in n_list:
            print(f"\n--- Block P | {config_id} | n={n} | M={M} ---")
            decomp, wall, data = _run_one_block_tradeoff(
                dgp, config_id, spec, "P", cfg_idx, n, M, force=force,
            )
            results[config_id][n] = (decomp, wall, data)
            _print_brief(config_id, n, decomp, wall)
    return results


def run_block_N(
    dgp,
    M: int,
    n_config_ids: list[str],
    force: bool = False,
) -> BlockResults:
    """Run Block N: n in {500,1000,2000} for selected configs (H IDs only)."""
    n_list = [500, 1000, 2000]
    results: BlockResults = {}
    for cfg_idx, config_id in enumerate(n_config_ids):
        spec = BLOCK_H_CONFIGS.get(config_id)
        if spec is None:
            print(f"[WARN] Block N: unknown config_id {config_id!r} -- skipping.")
            print(f"       Block N currently only accepts Block H config IDs.")
            continue
        results[config_id] = {}
        for n in n_list:
            print(f"\n--- Block N | {config_id} | n={n} | M={M} ---")
            decomp, wall, data = _run_one_block_tradeoff(
                dgp, config_id, spec, "N", cfg_idx, n, M, force=force,
            )
            results[config_id][n] = (decomp, wall, data)
            _print_brief(config_id, n, decomp, wall)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Report-only: load existing PKLs for all blocks
# ══════════════════════════════════════════════════════════════════════════════

_PKL_PATTERN = re.compile(r"dgp2tradeoff_([HQPN])_(.+)_n(\d+)_M(\d+)\.pkl$")


def _load_all_blocks() -> tuple[BlockResults, BlockResults, BlockResults, BlockResults]:
    """
    Scan _RAW_DIR for all dgp2tradeoff_*.pkl files and load them.
    Returns (results_H, results_Q, results_P, results_N).
    Wall-clock is set to 0.0 for all loaded entries.
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, BlockResults] = {"H": {}, "Q": {}, "P": {}, "N": {}}

    for pkl_path in sorted(_RAW_DIR.glob("dgp2tradeoff_*.pkl")):
        m = _PKL_PATTERN.match(pkl_path.name)
        if not m:
            continue
        block_char = m.group(1)
        config_id  = m.group(2)
        n          = int(m.group(3))
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
        decomp = _decompose(data)
        br = results[block_char]
        if config_id not in br:
            br[config_id] = {}
        br[config_id][n] = (decomp, 0.0, data)

    return results["H"], results["Q"], results["P"], results["N"]


def _merge_block_results(new: BlockResults, loaded: BlockResults) -> BlockResults:
    """Newly-computed takes priority over loaded."""
    merged = dict(loaded)
    for cid, n_dict in new.items():
        if cid not in merged:
            merged[cid] = {}
        merged[cid].update(n_dict)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  Report builder — 10 sections
# ══════════════════════════════════════════════════════════════════════════════

def _build_report(
    results_H: BlockResults,
    results_Q: Optional[BlockResults],
    results_P: Optional[BlockResults],
    results_N: Optional[BlockResults],
    dgp=None,
) -> list[str]:
    lines: list[str] = []
    lines.append("# DGP2 Trade-off Exploration")
    lines.append(f"# Generated : {datetime.date.today()}")
    lines.append(
        f"# SNR={SNR}, Michaelis-Menten, "
        f"ref dose a={A_GRID[REF_IDX]:.1f} (REF_IDX={REF_IDX})"
    )
    lines.append("")
    lines.append(
        "> **Estimand note** -- Block P (h_policy_scale != 1.0) changes the estimand"
    )
    lines.append(
        "> to J(pi_{a, h_policy}). Coverage_ref for Block P is NOT comparable to"
    )
    lines.append(
        "> baseline. Jensen gap (Section 5) quantifies the estimand shift."
    )
    lines.append("")

    _sec1_objective(lines)
    _sec2_configs(lines, results_H, results_Q, results_P, results_N)
    _sec3_block_h(lines, results_H)
    _sec4_block_q(lines, results_Q, results_H)
    _sec5_block_p(lines, results_P, dgp)
    _sec6_block_n(lines, results_N)
    _sec7_pareto(lines, results_H)
    _sec8_dominated(lines, results_H)
    _sec9_interpretation(lines)
    _sec10_recommendation(lines, results_H)

    return lines


# ── Section 1 ─────────────────────────────────────────────────────────────────

def _sec1_objective(add: list[str]) -> None:
    add.append("## 1. Objective and design")
    add.append("")
    add.append(
        "Map the empirical trade-off surface around the successful h-tuning region"
    )
    add.append(
        "(config A, Phase 9B.2bis). Four sequential blocks, user-mediated:"
    )
    add.append("")
    add.append(
        "| Block | Parameter varied | n | M | Decision gate |"
    )
    add.append("|-------|-----------------|---|---|---------------|")
    add.append(
        "| H | lambda_h x ell_scale (3x3 + baseline) | 1000 | 50 | Pick top 2 h configs |"
    )
    add.append(
        "| Q | lambda_Q x clip (4x2), conditional on best h | 1000 | 50 | q active? |"
    )
    add.append(
        "| P | h_policy_scale in {0.75,1.0,1.25,1.5} | 1000 | 50 | Locality/inference trade-off |"
    )
    add.append(
        "| N | n in {500,1000,2000}, selected configs, M=75 | 500-2000 | 75 | Trend in n |"
    )
    add.append("")
    add.append(
        "This is a screening experiment. No automatic promotion between blocks."
    )
    add.append("")


# ── Section 2 ─────────────────────────────────────────────────────────────────

def _sec2_configs(
    add: list[str],
    results_H: BlockResults,
    results_Q: Optional[BlockResults],
    results_P: Optional[BlockResults],
    results_N: Optional[BlockResults],
) -> None:
    add.append("## 2. Configs run (from PKLs in dgp2_tradeoff/)")
    add.append("")

    def _list_block(name: str, res: Optional[BlockResults]) -> None:
        if not res:
            add.append(f"**Block {name}**: not run yet.")
            return
        add.append(f"**Block {name}**: {len(res)} config(s).")
        for cid in sorted(res):
            ns = sorted(res[cid].keys())
            add.append(f"  - `{cid}` at n={ns}")

    _list_block("H", results_H)
    _list_block("Q", results_Q)
    _list_block("P", results_P)
    _list_block("N", results_N)
    add.append("")


# ── Section 3 — Block H ───────────────────────────────────────────────────────

def _sec3_block_h(add: list[str], results_H: BlockResults) -> None:
    add.append("## 3. Block H -- h-bridge local refinement (lambda_h x ell_scale)")
    add.append("")

    if not results_H:
        add.append("*Block H not run yet.*")
        add.append("")
        add.append("```bash")
        add.append(
            "python -m simulations.archive.utilities.dgp2_tradeoff_exploration "
            "--block H --mode screen"
        )
        add.append("```")
        add.append("")
        return

    ns_available = sorted({n for v in results_H.values() for n in v.keys()})
    n_show = max(ns_available)

    add.append(f"### 3a. Main metrics at n={n_show} (dose ref a={A_GRID[REF_IDX]:.1f})")
    add.append("")
    add.append(
        "| config | lambda_h | ell_scale | bias_REG | bias_DR | "
        "|b|/SE | cov_ref | SER | ESS_min | w_p99 |"
    )
    add.append("|--------|----------|-----------|----------|---------|"
               "--------|---------|-----|---------|-------|")

    for cid in BLOCK_H_ORDER:
        if cid not in results_H:
            continue
        entry = results_H[cid].get(n_show, (None, 0.0, None))
        d, _, data = entry
        spec = BLOCK_H_CONFIGS[cid]
        lam_h_str = "default" if spec["lambda_h"] is None else f"{spec['lambda_h']:.0e}"
        ser = _compute_ser(data)
        if d is None:
            add.append(
                f"| {cid} | {lam_h_str} | {spec['ell_scale']} "
                f"| --- | --- | --- | --- | --- | --- | --- |"
            )
            continue
        add.append(
            f"| {cid} | {lam_h_str} | {spec['ell_scale']} | "
            f"{_fmt(d['bias_reg'][REF_IDX], '+.4f')} | "
            f"{_fmt(d['bias_dr'][REF_IDX], '+.4f')} | "
            f"{_fmt(d['bse_grid'][REF_IDX], '.2f')} | "
            f"{_fmt(d['coverage_ref'], '.3f')} | "
            f"{_fmt(ser, '.2f')} | "
            f"{_fmt(d['ess_min_mean'], '.0f')} | "
            f"{_fmt(d['w_p99_mean'], '.2f')} |"
        )
    add.append("")

    # Heatmap: bias_REG by (lambda_h rows x ell_scale cols)
    add.append("### 3b. bias_REG heatmap (rows=lambda_h, cols=ell_scale)")
    add.append("")
    _lam_vals = [3e-5, 1e-4, 3e-4]
    _ell_vals = [1.5, 2.0, 2.5]
    add.append(
        "| lambda_h \\ ell_scale | "
        + " | ".join(f"ell={e}" for e in _ell_vals)
        + " |"
    )
    add.append("|" + "---|" * (len(_ell_vals) + 1))
    # Build reverse lookup (lambda_h, ell_scale) -> config_id to avoid
    # platform-dependent .0e formatting differences (3e-05 vs 3e-5).
    _hmap: dict[tuple, str] = {
        (v["lambda_h"], v["ell_scale"]): k
        for k, v in BLOCK_H_CONFIGS.items()
        if v["lambda_h"] is not None
    }
    baseline_d = results_H.get("H_baseline", {}).get(n_show, (None,))[0]
    for lam in _lam_vals:
        row = [f"{lam:.0e}"]
        for ell in _ell_vals:
            cid = _hmap.get((lam, ell))
            if cid is None:
                row.append("---")
                continue
            entry = results_H.get(cid, {}).get(n_show, (None, 0.0, None))
            d_h = entry[0]
            if d_h is None:
                row.append("---")
            else:
                row.append(f"{float(d_h['bias_reg'][REF_IDX]):+.4f}")
        add.append("| " + " | ".join(row) + " |")
    add.append("")
    bl_bias_str = (
        _fmt(baseline_d["bias_reg"][REF_IDX], "+.4f") if baseline_d else "---"
    )
    add.append(f"Baseline bias_REG = {bl_bias_str}")
    add.append("")


# ── Section 4 — Block Q ───────────────────────────────────────────────────────

def _sec4_block_q(
    add: list[str],
    results_Q: Optional[BlockResults],
    results_H: BlockResults,
) -> None:
    add.append("## 4. Block Q -- q-tuning conditional on best h")
    add.append("")

    if not results_Q:
        add.append("*Block Q not run yet.*")
        add.append("")
        add.append(
            "After reading Section 3, pick the top 2 h configs and run:"
        )
        add.append("```bash")
        add.append(
            'python -m simulations.archive.utilities.dgp2_tradeoff_exploration '
            '--block Q --mode screen --h-configs "H_best1,H_best2"'
        )
        add.append("```")
        add.append("")
        return

    ns_available = sorted({n for v in results_Q.values() for n in v.keys()})
    n_show = max(ns_available) if ns_available else 1000

    add.append(f"Results at n={n_show}.")
    add.append("")
    add.append(
        "| config_id | lambda_Q | clip | bias_DR | |b|/SE | cov_ref | "
        "corr_ratio | ESS |"
    )
    add.append("|-----------|----------|------|---------|--------|---------|"
               "------------|-----|")

    all_corr_ratios: list[float] = []
    for cid in sorted(results_Q):
        entry = results_Q[cid].get(n_show, (None, 0.0, None))
        d = entry[0]
        # Parse lambda_Q and clip from config ID
        lq_m   = re.search(r"lamQ([\de+\-\.]+)", cid)
        cl_m   = re.search(r"clip(\w+)$", cid)
        lq_str = lq_m.group(1) if lq_m else "?"
        cl_str = cl_m.group(1) if cl_m else "?"
        if d is None:
            add.append(
                f"| {cid} | {lq_str} | {cl_str} "
                f"| --- | --- | --- | --- | --- |"
            )
            continue
        cr_ref = float(d["correction_ratio"][REF_IDX])
        all_corr_ratios.append(abs(cr_ref))
        add.append(
            f"| {cid} | {lq_str} | {cl_str} | "
            f"{_fmt(d['bias_dr'][REF_IDX], '+.4f')} | "
            f"{_fmt(d['bse_grid'][REF_IDX], '.2f')} | "
            f"{_fmt(d['coverage_ref'], '.3f')} | "
            f"{_fmt(cr_ref, '+.3f')} | "
            f"{_fmt(d['ess_min_mean'], '.0f')} |"
        )
    add.append("")

    # Decision rule
    if all_corr_ratios:
        max_cr = max(all_corr_ratios)
        if max_cr < _CORR_RATIO_ACTIVE:
            add.append(
                f"> **Decision**: q is **INACTIVE** conditional on tuned h. "
                f"max|correction_ratio| = {max_cr:.4f} < {_CORR_RATIO_ACTIVE}."
            )
            add.append(
                "> q-bridge is not the bottleneck. Do not expand q grid further."
            )
        else:
            add.append(
                f"> **Decision**: q shows some activity. "
                f"max|correction_ratio| = {max_cr:.4f} >= {_CORR_RATIO_ACTIVE}."
            )
            add.append(
                "> Check whether bias_DR improvement is meaningful before expanding."
            )
    add.append("")


# ── Section 5 — Block P ───────────────────────────────────────────────────────

def _sec5_block_p(
    add: list[str],
    results_P: Optional[BlockResults],
    dgp=None,
) -> None:
    add.append("## 5. Block P -- policy bandwidth trade-off (h_policy_scale)")
    add.append("")

    if not results_P:
        add.append("*Block P not run yet.*")
        add.append("")
        add.append("After reading Section 3, run:")
        add.append("```bash")
        add.append(
            'python -m simulations.archive.utilities.dgp2_tradeoff_exploration '
            '--block P --mode screen --h-config "H_best"'
        )
        add.append("```")
        add.append("")
        return

    ns_available = sorted({n for v in results_P.values() for n in v.keys()})
    n_show = max(ns_available) if ns_available else 1000

    add.append(f"Results at n={n_show}.")
    add.append("")
    add.append(
        "| config_id | h_policy_scale | bias_DR | |b|/SE | cov_ref | "
        "Jensen_gap_ref | MISE |"
    )
    add.append("|-----------|---------------|---------|--------|---------|"
               "----------------|------|")

    for cid in sorted(results_P):
        entry = results_P[cid].get(n_show, (None, 0.0, None))
        d = entry[0]
        # Parse h_policy_scale from config ID: hpol{X}p{Y}
        hm = re.search(r"hpol([\dp]+)$", cid)
        if hm:
            hpol_str = hm.group(1).replace("p", ".")
        else:
            hpol_str = "?"
        if d is None:
            add.append(
                f"| {cid} | {hpol_str} | --- | --- | --- | --- | --- |"
            )
            continue
        jg_ref = float(np.nanmean(d["jensen_gap_mean"])) if d["jensen_gap_mean"] is not None else float("nan")
        add.append(
            f"| {cid} | {hpol_str} | "
            f"{_fmt(d['bias_dr'][REF_IDX], '+.4f')} | "
            f"{_fmt(d['bse_grid'][REF_IDX], '.2f')} | "
            f"{_fmt(d['coverage_ref'], '.3f')} | "
            f"{_fmt(jg_ref, '+.5f')} | "
            f"{_fmt(d['mise_mean'], '.5f')} |"
        )
    add.append("")
    add.append(
        "> **Note**: h_policy_scale != 1.0 changes the estimand to J(pi_{a, h_policy})."
    )
    add.append(
        "> Do not interpret Block P coverage improvements as bias 'fixes'."
    )
    add.append(
        "> This block documents the locality/inference trade-off for S6."
    )
    add.append("")


# ── Section 6 — Block N ───────────────────────────────────────────────────────

def _sec6_block_n(add: list[str], results_N: Optional[BlockResults]) -> None:
    add.append("## 6. Block N -- n-trend confirmation")
    add.append("")

    if not results_N:
        add.append("*Block N not run yet.*")
        add.append("")
        add.append("After reading Sections 3-5, run:")
        add.append("```bash")
        add.append(
            'python -m simulations.archive.utilities.dgp2_tradeoff_exploration '
            '--block N --mode confirm --n-configs "H_baseline,H_best1,..."'
        )
        add.append("```")
        add.append("")
        return

    ns_all     = sorted({n for v in results_N.values() for n in v.keys()})
    config_ids = [c for c in BLOCK_H_ORDER if c in results_N] + \
                 [c for c in sorted(results_N) if c not in BLOCK_H_ORDER]

    def _get_d(cid: str, n: int) -> Optional[dict]:
        entry = results_N.get(cid, {}).get(n, (None,))
        return entry[0]

    n_hdr = " | ".join(f"n={n}" for n in ns_all)
    sep   = "|" + "---|" * (len(ns_all) + 1)

    add.append("### 6a. |bias|/SE by n")
    add.append("")
    add.append(f"| config | {n_hdr} |")
    add.append(sep)
    for cid in config_ids:
        vals = [
            _fmt(_get_d(cid, n)["bse_grid"][REF_IDX] if _get_d(cid, n) else float("nan"), ".2f")
            for n in ns_all
        ]
        add.append(f"| {cid} | " + " | ".join(vals) + " |")
    add.append("")

    add.append("### 6b. coverage_ref by n")
    add.append("")
    add.append(f"| config | {n_hdr} |")
    add.append(sep)
    for cid in config_ids:
        vals = [
            _fmt(_get_d(cid, n)["coverage_ref"] if _get_d(cid, n) else float("nan"), ".3f")
            for n in ns_all
        ]
        add.append(f"| {cid} | " + " | ".join(vals) + " |")
    add.append("")

    add.append("### 6c. bias_REG and bias_DR by n")
    add.append("")
    add.append(f"| config | n | bias_REG | bias_DR |")
    add.append("|--------|---|----------|---------|")
    for cid in config_ids:
        for n in ns_all:
            d = _get_d(cid, n)
            if d is None:
                add.append(f"| {cid} | {n} | --- | --- |")
            else:
                add.append(
                    f"| {cid} | {n} | "
                    f"{_fmt(d['bias_reg'][REF_IDX], '+.4f')} | "
                    f"{_fmt(d['bias_dr'][REF_IDX], '+.4f')} |"
                )
    add.append("")


# ── Section 7 — Pareto frontier ───────────────────────────────────────────────

def _sec7_pareto(add: list[str], results_H: BlockResults) -> None:
    add.append("## 7. Pareto frontier (Block H configs)")
    add.append("")

    if not results_H:
        add.append("*Awaiting Block H results.*")
        add.append("")
        return

    ns_available = sorted({n for v in results_H.values() for n in v.keys()})
    n_eval = 1000 if 1000 in ns_available else (max(ns_available) if ns_available else None)
    if n_eval is None:
        add.append("*No usable n found.*")
        add.append("")
        return

    baseline_d = results_H.get("H_baseline", {}).get(n_eval, (None,))[0]
    if baseline_d is None:
        add.append(f"*H_baseline missing at n={n_eval} -- Pareto deferred.*")
        add.append("")
        return

    bl_bias = abs(float(baseline_d["bias_dr"][REF_IDX]))
    bl_w    = float(baseline_d["w_p99_mean"])

    add.append(
        f"Classification at n={n_eval}. "
        f"Baseline |bias_DR(ref)| = {bl_bias:.4f}, w_p99 = {bl_w:.2f}."
    )
    add.append("")
    add.append(
        "Thresholds: bias_red >= 30%, w_p99 < 2x baseline, ESS >= 20, "
        "q_clip < 5%, SER in [0.75, 1.25]."
    )
    add.append(
        "**strong** = also satisfies |b|/SE < 1.0."
    )
    add.append("")
    add.append(
        "| config | bias_DR | bias_red% | |b|/SE | cov_ref | SER | "
        "w_p99 | ESS | q_clip | Classification |"
    )
    add.append(
        "|--------|---------|-----------|--------|---------|-----|"
        "-------|-----|--------|----------------|"
    )

    for cid in BLOCK_H_ORDER:
        if cid not in results_H:
            continue
        entry = results_H[cid].get(n_eval, (None, 0.0, None))
        d, _, data = entry
        if d is None:
            continue
        spec     = BLOCK_H_CONFIGS[cid]
        bias_dr  = abs(float(d["bias_dr"][REF_IDX]))
        bias_red = (bl_bias - bias_dr) / max(bl_bias, 1e-12) * 100.0
        bse      = float(d["bse_grid"][REF_IDX])
        cov      = float(d["coverage_ref"])
        ser      = _compute_ser(data)
        w_p99    = float(d["w_p99_mean"])
        ess      = float(d["ess_min_mean"])
        q_clip   = float(d["q_clip_mean"])

        promising = (
            bias_red >= _BIAS_RED_MIN * 100.0
            and w_p99 < _W_P99_MAX_RATIO * bl_w
            and ess >= _ESS_MIN
            and q_clip < _QCLIP_MAX
            and (np.isfinite(ser) and _SER_LOW <= ser <= _SER_HIGH)
        )
        if promising and bse < _BSE_STRONG:
            cls = "**strong**"
        elif promising:
            cls = "promising"
        else:
            cls = "dominated"

        add.append(
            f"| {cid} | "
            f"{_fmt(d['bias_dr'][REF_IDX], '+.4f')} | "
            f"{_fmt(bias_red, '.1f')}% | "
            f"{_fmt(bse, '.2f')} | "
            f"{_fmt(cov, '.3f')} | "
            f"{_fmt(ser, '.2f')} | "
            f"{_fmt(w_p99, '.2f')} | "
            f"{_fmt(ess, '.0f')} | "
            f"{_fmt(q_clip, '.4f')} | "
            f"{cls} |"
        )
    add.append("")


# ── Section 8 — Dominated configs ─────────────────────────────────────────────

def _sec8_dominated(add: list[str], results_H: BlockResults) -> None:
    add.append("## 8. Dominated configs (not recommended for Block N)")
    add.append("")

    if not results_H:
        add.append("*Awaiting Block H results.*")
        add.append("")
        return

    ns_available = sorted({n for v in results_H.values() for n in v.keys()})
    n_eval = 1000 if 1000 in ns_available else (max(ns_available) if ns_available else None)
    if n_eval is None:
        add.append("*No usable n.*")
        add.append("")
        return

    baseline_d = results_H.get("H_baseline", {}).get(n_eval, (None,))[0]
    bl_bias = abs(float(baseline_d["bias_dr"][REF_IDX])) if baseline_d else 1.0

    dominated_found = False
    for cid in BLOCK_H_ORDER:
        if cid not in results_H:
            continue
        entry = results_H[cid].get(n_eval, (None,))
        d = entry[0]
        if d is None:
            continue
        bias_dr  = abs(float(d["bias_dr"][REF_IDX]))
        bias_red = (bl_bias - bias_dr) / max(bl_bias, 1e-12) * 100.0
        if bias_red < _BIAS_RED_MIN * 100.0:
            add.append(
                f"- `{cid}`: bias reduction {bias_red:.1f}% < {_BIAS_RED_MIN*100:.0f}% threshold."
            )
            dominated_found = True
    if not dominated_found:
        add.append("All tested configs meet the 30% bias reduction threshold.")
    add.append("")


# ── Section 9 — Interpretation ────────────────────────────────────────────────

def _sec9_interpretation(add: list[str]) -> None:
    add.append("## 9. Interpretation for S6/S7")
    add.append("")
    add.append(
        "The trade-off surface characterizes three independent levers:"
    )
    add.append("")
    add.append("**Lever 1 -- h-bridge regularization (Block H):**")
    add.append(
        "  Lower lambda_h + wider ell reduces RKHS approximation error in KPVBridgeH."
    )
    add.append(
        "  The bias reduction quantifies how much of |bias| ~ n^{-rho} can be recovered."
    )
    add.append("")
    add.append("**Lever 2 -- q-correction activity (Block Q):**")
    add.append(
        "  The DR correction J_dr - J_reg = (1/n) sum w_i(Y_i - J_reg(A_i))."
    )
    add.append(
        "  If correction_ratio stays near 0 after h-tuning, q is not the bottleneck."
    )
    add.append("")
    add.append("**Lever 3 -- policy bandwidth (Block P):**")
    add.append(
        "  Increasing h_policy_scale smooths the estimand J(pi_{a,h}) toward the marginal mean."
    )
    add.append(
        "  Jensen gap quantifies how far the estimand moves from m_true(a)."
    )
    add.append("")
    add.append(
        "Essay phrase (S6): 'We find that the main lever is the outcome bridge h;"
    )
    add.append(
        " q-correction and policy smoothing have secondary roles that the RKHS rate"
    )
    add.append(
        " constraint ultimately limits.'"
    )
    add.append("")


# ── Section 10 — Recommendation ───────────────────────────────────────────────

def _sec10_recommendation(
    add: list[str],
    results_H: BlockResults,
) -> None:
    add.append("## 10. Recommendation")
    add.append("")
    if not results_H:
        add.append("*Awaiting Block H results before any recommendation.*")
        add.append("")
        return
    add.append("**For S6 main sensitivity analysis:**")
    add.append(
        "  Use the Block H config with lowest |b|/SE (classified 'strong' or 'promising'),"
    )
    add.append(
        "  verified by Block N trend. Present the heatmap (Section 3b) as the"
    )
    add.append("  primary empirical contribution.")
    add.append("")
    add.append("**For S7 stress-test narrative:**")
    add.append(
        "  H_baseline is the canonical 'untuned' comparison. If no config is 'strong',"
    )
    add.append(
        "  DGP2 remains a stress-test illustrating RKHS-rate limitations."
    )
    add.append("")
    add.append("**Bennett-lite priority (Phase 11):**")
    add.append(
        "  If best Block H config still has |b|/SE > 1 at n=2000 in Block N,"
    )
    add.append(
        "  Bennett-lite (alternative nuisance estimation) becomes higher priority."
    )
    add.append("")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI + main()
# ══════════════════════════════════════════════════════════════════════════════

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DGP2 Trade-off Exploration -- Phase 9B.3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--block", required=True,
        choices=["H", "Q", "P", "N", "all"],
        help="Which block to run. 'all' = report only from existing PKLs.",
    )
    p.add_argument(
        "--mode", default="screen",
        choices=["quick", "screen", "confirm"],
        help="Computation mode (default: screen).",
    )
    p.add_argument(
        "--h-configs", default=None,
        help=(
            "Comma-separated Block H config IDs for Block Q "
            "(e.g. 'H_lam1e-4_ell2.0,H_lam3e-5_ell2.0')."
        ),
    )
    p.add_argument(
        "--h-config", default=None,
        help="Single Block H config ID for Block P (e.g. 'H_lam1e-4_ell2.0').",
    )
    p.add_argument(
        "--n-configs", default=None,
        help="Comma-separated config IDs for Block N (Block H IDs only).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Redo blocks even if pkl exists.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_cli()

    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)

    if "OneDrive" in str(_RAW_DIR):
        print(
            "[WARN] Output is under OneDrive -- risk of .partial.pkl corruption. "
            "Monitor file sizes during long runs."
        )

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)

    results_H: BlockResults = {}
    results_Q: Optional[BlockResults] = None
    results_P: Optional[BlockResults] = None
    results_N: Optional[BlockResults] = None

    if args.block == "all":
        results_H, results_Q, results_P, results_N = _load_all_blocks()

    elif args.block == "H":
        if args.mode == "quick":
            n_list, M = _MODES["quick"]
            config_ids = _BLOCK_H_QUICK
        elif args.mode == "screen":
            n_list, M = _MODES["screen"]
            config_ids = BLOCK_H_ORDER
        else:
            raise ValueError(
                "Block H does not support --mode confirm. Use 'quick' or 'screen'."
            )
        print(
            f"\n=== Block H | mode={args.mode} | n={n_list} | M={M} | "
            f"{len(config_ids)} configs ==="
        )
        results_H = run_block_H(dgp, n_list, M, config_ids, force=args.force)

    elif args.block == "Q":
        if args.mode != "screen":
            raise ValueError("Block Q only supports --mode screen.")
        if not args.h_configs:
            raise ValueError("Block Q requires --h-configs 'id1,id2,...'")
        h_config_ids = [x.strip() for x in args.h_configs.split(",")]
        unknown = [c for c in h_config_ids if c not in BLOCK_H_CONFIGS]
        if unknown:
            raise ValueError(
                f"Unknown Block H config IDs: {unknown}. "
                f"Valid IDs: {BLOCK_H_ORDER}"
            )
        n_list, M = _MODES["screen"]
        print(
            f"\n=== Block Q | n={n_list} | M={M} | "
            f"parent h configs: {h_config_ids} ==="
        )
        results_Q = run_block_Q(dgp, n_list, M, h_config_ids, force=args.force)

    elif args.block == "P":
        if args.mode not in ("quick", "screen"):
            raise ValueError("Block P supports --mode quick or screen only.")
        if args.mode == "quick":
            n_list, M = _MODES["quick"]
            h_id = args.h_config or "H_lam1e-4_ell2.0"
        else:
            n_list, M = _MODES["screen"]
            if not args.h_config:
                raise ValueError(
                    "Block P requires --h-config 'id' in screen mode."
                )
            h_id = args.h_config
        if h_id not in BLOCK_H_CONFIGS:
            raise ValueError(
                f"Unknown h config {h_id!r}. Valid: {BLOCK_H_ORDER}"
            )
        print(
            f"\n=== Block P | mode={args.mode} | n={n_list} | M={M} | "
            f"parent: {h_id} ==="
        )
        results_P = run_block_P(dgp, n_list, M, h_id, force=args.force)

    elif args.block == "N":
        if args.mode != "confirm":
            raise ValueError("Block N only supports --mode confirm.")
        if not args.n_configs:
            raise ValueError("Block N requires --n-configs 'id1,id2,...'")
        n_config_ids = [x.strip() for x in args.n_configs.split(",")]
        _, M = _MODES["confirm"]
        print(f"\n=== Block N | M={M} | configs: {n_config_ids} ===")
        results_N = run_block_N(dgp, M, n_config_ids, force=args.force)

    # Merge with existing PKLs so the report reflects the full picture
    loaded_H, loaded_Q, loaded_P, loaded_N = _load_all_blocks()
    results_H = _merge_block_results(results_H, loaded_H)
    if results_Q is None:
        results_Q = loaded_Q or None
    if results_P is None:
        results_P = loaded_P or None
    if results_N is None:
        results_N = loaded_N or None

    print(f"\n[Report] Building {_REPORT_PATH} ...")
    report_lines = _build_report(results_H, results_Q, results_P, results_N, dgp=dgp)
    _REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[Report] Done -> {_REPORT_PATH}")


if __name__ == "__main__":
    main()
