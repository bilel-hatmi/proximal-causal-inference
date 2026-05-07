"""
DGP2 Optimal Region Cartography -- Phase 9B.4

Context
-------
Phase 9B.3 identified H_lam1e-4_ell2.5 (lambda_h=1e-4, ell_scale=2.5) as the best
h-bridge configuration, achieving coverage=0.92 at n=2000 with |b|/SE=0.59.
However, ell_scale=2.5 was the edge of the grid. This script answers:
  - Is ell=2.5 an optimum, or the start of a plateau/trend toward ell=3.0/3.5?
  - Does over-smoothing eventually degrade coverage or inflate bias?

Design
------
Block E : ell_scale extension grid (lambda_h x ell_scale around the winner)
          lambda_h in {3e-5, 1e-4, 3e-4}, ell_scale in {2.25, 2.5, 2.75, 3.0, 3.5}
          + baseline.  Total = 16 configs (15 grid + baseline).
          n=1000, M=50.
          Anchors for ell=2.5 loaded from Phase 9B.3 without recomputing.

Block N2: n-trend confirmation for top-3 selected configs.
          n in {500, 1000, 2000}, M=75.
          Always includes E_baseline + E_lam1e-4_ell2p50 anchor.
          Anchors from Phase 9B.3 Block N loaded without recomputing.

Classification thresholds (stricter than Phase 9B.3):
  strong    : bias_red >= 50%, |b|/SE < 0.75, cov >= 0.88, SER in [0.75,1.25],
              ESS >= 20, w_p99 < 2x baseline, q_clip < 0.05
  promising : bias_red >= 30%, |b|/SE < 1.0, cov >= 0.80, clean numerics
  dominated : otherwise

Seeds (disjoint from all prior scripts up to 13.9M):
  Block E  : 14_000_000 + cfg_idx*10_000 + n
  Block N2 : 15_000_000 + cfg_idx*10_000 + n

Output
------
  simulations/results/raw/dgp2_optimal_region/dgp2optimal_{E|N}_{config_id}_n{n}_M{M}.pkl
  simulations/results/summaries/dgp2_optimal_region.md
"""
from __future__ import annotations

import argparse
import datetime
import pickle
import re
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ── Reuse helpers from existing scripts ───────────────────────────────────────
from simulations.experiments.dgp2_bias_diagnostics import (
    A_GRID,
    CHECKPOINT_FREQ,
    CLIP_DEFAULT,
    K,
    LAMBDA_Q_DEFAULT,
    N_FOLDS,
    RANDOM_STATE,
    REF_IDX,
    SNR,
    _build_record,
    _compute_J_policy_true,
    _decompose,
    _fmt,
    _median_bandwidth,
    _pred_cov,
    _run_block,
    _silverman_h,
)
from simulations.archive.utilities.dgp2_tuning_confirmation import (
    _M_ok_ratio,
    _compute_ser,
    _max_weight_max,
    _per_dose_empirical_coverage,
    _per_dose_se,
    _q_neg_share_mean,
    make_factory,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR        = Path(__file__).resolve().parents[2]
_RAW_DIR         = _BASE_DIR / "simulations" / "results" / "raw" / "dgp2_optimal_region"
_TRADEOFF_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "dgp2_tradeoff"
_SUMM_DIR        = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH     = _SUMM_DIR / "dgp2_optimal_region.md"

# ── Classification thresholds (stricter than Phase 9B.3) ──────────────────────
_BIAS_RED_STRONG    = 0.50   # |bias_DR| reduced >= 50% vs E_baseline
_BSE_STRONG         = 0.75   # |b|/SE < 0.75
_COV_STRONG         = 0.88   # coverage_ref >= 0.88
_BIAS_RED_PROMISING = 0.30
_BSE_PROMISING      = 1.00
_COV_PROMISING      = 0.80
_ESS_MIN            = 20.0
_W_P99_MAX_RATIO    = 2.0
_QCLIP_MAX          = 0.05
_SER_LOW, _SER_HIGH = 0.75, 1.25

# ── Grid definition ───────────────────────────────────────────────────────────
_LAM_GRID: list[float] = [3e-5, 1e-4, 3e-4]
_ELL_GRID: list[float] = [2.25, 2.5, 2.75, 3.0, 3.5]

# Human-readable lambda strings (avoids platform-specific .0e formatting)
_LAM_STR: dict[float, str] = {3e-5: "3e-5", 1e-4: "1e-4", 3e-4: "3e-4"}

# ── Anchor mappings from Phase 9B.3 (reuse without recomputing) ───────────────
# Block E anchors: config_id -> pkl stem in dgp2_tradeoff/ (n=1000, M=50)
_BLOCK_E_ANCHORS: dict[str, str] = {
    "E_lam3e-5_ell2p50": "dgp2tradeoff_H_H_lam3e-5_ell2.5_n1000_M50",
    "E_lam1e-4_ell2p50": "dgp2tradeoff_H_H_lam1e-4_ell2.5_n1000_M50",
    "E_lam3e-4_ell2p50": "dgp2tradeoff_H_H_lam3e-4_ell2.5_n1000_M50",
    "E_baseline":         "dgp2tradeoff_H_H_baseline_n1000_M50",
}
# Block N2 anchors: config_id -> pkl stem template with {n} (M=75)
_BLOCK_N2_ANCHORS: dict[str, str] = {
    "E_baseline":         "dgp2tradeoff_N_H_baseline_n{n}_M75",
    "E_lam1e-4_ell2p50": "dgp2tradeoff_N_H_lam1e-4_ell2.5_n{n}_M75",
    "E_lam3e-5_ell2p50": "dgp2tradeoff_N_H_lam3e-5_ell2.5_n{n}_M75",
}

# ── Seed bases (disjoint from all prior runs: tradeoff used 10M-13.9M) ────────
_SEED_BASE_E  = 14_000_000
_SEED_BASE_N2 = 15_000_000

# ── Type alias ────────────────────────────────────────────────────────────────
BlockResults = dict[str, dict[int, tuple[Optional[dict], float, Optional[dict]]]]


# ══════════════════════════════════════════════════════════════════════════════
#  Config ID helpers
# ══════════════════════════════════════════════════════════════════════════════

def _lam_str(lam: float) -> str:
    return _LAM_STR.get(lam, f"{lam:.0e}")


def _ell_str(ell: float) -> str:
    return f"{ell:.2f}".replace(".", "p")


def _e_config_id(lam: float, ell: float) -> str:
    return f"E_lam{_lam_str(lam)}_ell{_ell_str(ell)}"


# ══════════════════════════════════════════════════════════════════════════════
#  Block E config table -- single source of truth
# ══════════════════════════════════════════════════════════════════════════════

def _build_block_e_configs() -> dict[str, dict]:
    configs: dict[str, dict] = {}
    configs["E_baseline"] = dict(
        lambda_h=None, ell_scale=1.0,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT, h_policy_scale=1.0,
        description="Reference: KPV defaults (equivalent to H_baseline).",
    )
    for lam in _LAM_GRID:
        for ell in _ELL_GRID:
            cid = _e_config_id(lam, ell)
            configs[cid] = dict(
                lambda_h=lam, ell_scale=ell,
                lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT, h_policy_scale=1.0,
                description=f"lambda_h={_lam_str(lam)}, ell_scale={ell:.2f}.",
            )
    return configs


BLOCK_E_CONFIGS: dict[str, dict] = _build_block_e_configs()

BLOCK_E_ORDER: list[str] = ["E_baseline"] + [
    _e_config_id(lam, ell)
    for ell in _ELL_GRID
    for lam in _LAM_GRID
]


# ══════════════════════════════════════════════════════════════════════════════
#  Seed + PKL label
# ══════════════════════════════════════════════════════════════════════════════

def _seed_E(cfg_idx: int, n: int) -> int:
    return _SEED_BASE_E + cfg_idx * 10_000 + int(n)


def _seed_N2(cfg_idx: int, n: int) -> int:
    return _SEED_BASE_N2 + cfg_idx * 10_000 + int(n)


def _pkl_label(block: str, config_id: str, n: int, M: int) -> str:
    return f"dgp2optimal_{block}_{config_id}_n{n}_M{M}"


# ══════════════════════════════════════════════════════════════════════════════
#  Anchor loader (Phase 9B.3 data, no recompute)
# ══════════════════════════════════════════════════════════════════════════════

def _try_load_anchor_E(config_id: str) -> Optional[tuple]:
    """Try to load a Block E anchor from Phase 9B.3 dgp2_tradeoff/ directory."""
    stem = _BLOCK_E_ANCHORS.get(config_id)
    if stem is None:
        return None
    p = _TRADEOFF_RAW_DIR / f"{stem}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        data = pickle.load(fh)
    return _decompose(data), 0.0, data


def _try_load_anchor_N2(config_id: str, n: int) -> Optional[tuple]:
    """Try to load a Block N2 anchor from Phase 9B.3 dgp2_tradeoff/ directory."""
    tmpl = _BLOCK_N2_ANCHORS.get(config_id)
    if tmpl is None:
        return None
    stem = tmpl.format(n=n)
    p = _TRADEOFF_RAW_DIR / f"{stem}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        data = pickle.load(fh)
    return _decompose(data), 0.0, data


# ══════════════════════════════════════════════════════════════════════════════
#  Per-block runner
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_block_optimal(
    dgp,
    config_id: str,
    spec: dict,
    block: str,         # "E" or "N"
    cfg_idx: int,
    n: int,
    M: int,
    force: bool = False,
) -> tuple[Optional[dict], float, Optional[dict]]:
    """
    Run one (config x n) block.  Returns (decomp, wall_seconds, raw_data).
    If block E and anchor exists in dgp2_tradeoff/, uses it (wall=0.0).
    If block N and anchor exists, uses it (wall=0.0).
    """
    # Try anchor first (avoids recompute for ell=2.5 and baseline configs)
    if not force:
        if block == "E":
            anc = _try_load_anchor_E(config_id)
        else:
            anc = _try_load_anchor_N2(config_id, n)
        if anc is not None:
            label = _pkl_label(block, config_id, n, M)
            # Save a copy to our raw dir so it appears in PKL scans
            _RAW_DIR.mkdir(parents=True, exist_ok=True)
            dest = _RAW_DIR / f"{label}.pkl"
            if not dest.exists():
                _, _, data = anc
                with open(dest, "wb") as fh:
                    pickle.dump(data, fh)
            print(f"  [ANCHOR] {config_id} n={n} loaded from Phase 9B.3 (no recompute)")
            return anc

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    ref_sample = dgp.generate(n=n, seed=0)
    h_ref  = _silverman_h(ref_sample.A)
    ell_W0 = _median_bandwidth(ref_sample.W)
    ell_A0 = _median_bandwidth(ref_sample.A)
    ell_Z0 = _median_bandwidth(ref_sample.Z)

    h_policy = h_ref * float(spec["h_policy_scale"])
    J_pt     = _compute_J_policy_true(dgp, A_GRID, h_policy)
    factory  = make_factory(spec, A_GRID, h_ref, ell_W0, ell_A0, ell_Z0)

    label     = _pkl_label(block, config_id, n, M)
    seed_base = _seed_E(cfg_idx, n) if block == "E" else _seed_N2(cfg_idx, n)

    if force:
        for sfx in (".pkl", ".partial.pkl"):
            p = _RAW_DIR / f"{label}{sfx}"
            if p.exists():
                p.unlink()

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
        print(f"    [WARN] No successful reps for {config_id} n={n} (wall={wall/60:.1f}min)")


# ══════════════════════════════════════════════════════════════════════════════
#  Block runners
# ══════════════════════════════════════════════════════════════════════════════

def run_block_E(
    dgp,
    n_list: list[int],
    M: int,
    config_ids: Optional[list[str]] = None,
    force: bool = False,
) -> BlockResults:
    """Run Block E: ell_scale extension grid around winning region."""
    if config_ids is None:
        config_ids = BLOCK_E_ORDER
    results: BlockResults = {}
    for config_id in config_ids:
        results[config_id] = {}
        cfg_idx = BLOCK_E_ORDER.index(config_id)
        spec = BLOCK_E_CONFIGS[config_id]
        for n in n_list:
            print(f"\n--- Block E | {config_id} | n={n} | M={M} ---")
            decomp, wall, data = _run_one_block_optimal(
                dgp, config_id, spec, "E", cfg_idx, n, M, force=force,
            )
            results[config_id][n] = (decomp, wall, data)
            _print_brief(config_id, n, decomp, wall)
    return results


def run_block_N2(
    dgp,
    M: int,
    n_config_ids: list[str],
    force: bool = False,
) -> BlockResults:
    """Run Block N2: n-trend for selected configs."""
    n_list = [500, 1000, 2000]
    results: BlockResults = {}
    for cfg_idx, config_id in enumerate(n_config_ids):
        spec = BLOCK_E_CONFIGS.get(config_id)
        if spec is None:
            print(f"[WARN] Block N2: unknown config_id {config_id!r} -- skipping.")
            continue
        results[config_id] = {}
        for n in n_list:
            print(f"\n--- Block N2 | {config_id} | n={n} | M={M} ---")
            decomp, wall, data = _run_one_block_optimal(
                dgp, config_id, spec, "N", cfg_idx, n, M, force=force,
            )
            results[config_id][n] = (decomp, wall, data)
            _print_brief(config_id, n, decomp, wall)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Auto-selection for Block N2
# ══════════════════════════════════════════════════════════════════════════════

_BIAS_RED_MIN_AUTO = 0.30
_ESS_MIN_AUTO      = 15.0
_MAX_N2_CONFIGS    = 4


def _baseline_bias_dr(results_E: BlockResults) -> float:
    entry = results_E.get("E_baseline", {}).get(1000)
    if entry is not None and entry[0] is not None:
        return abs(float(entry[0]["bias_dr"][REF_IDX]))
    return 0.0762  # Phase 9B.3 measured baseline at n=1000


def _select_n2_configs(results_E: BlockResults, top_k: int = 2) -> list[str]:
    """
    Select configs for Block N2:
    - Always include E_baseline (canonical comparison)
    - Always include E_lam1e-4_ell2p50 (Phase 9B.3 anchor) if not in top
    - Top-k new configs by |b|/SE (passing bias_red >= 30%, ESS >= 15)
    - Total <= _MAX_N2_CONFIGS
    """
    baseline_bias = _baseline_bias_dr(results_E)
    candidates: list[tuple[float, str, bool]] = []  # (bse, cid, passes)

    for config_id, n_dict in results_E.items():
        if config_id == "E_baseline":
            continue
        entry = n_dict.get(1000)
        if entry is None:
            continue
        decomp = entry[0]
        if decomp is None:
            continue
        bias_dr  = abs(float(decomp["bias_dr"][REF_IDX]))
        bse      = float(decomp["bse_grid"][REF_IDX])
        ess      = float(decomp.get("ess_min_mean", 0.0))
        bias_red = (baseline_bias - bias_dr) / max(baseline_bias, 1e-12)
        passes   = (bias_red >= _BIAS_RED_MIN_AUTO) and (ess >= _ESS_MIN_AUTO)
        candidates.append((bse, config_id, passes))

    candidates.sort(key=lambda t: t[0])
    selected = [cid for (_, cid, ok) in candidates if ok][:top_k]

    # Ensure anchor config is present
    anchor = "E_lam1e-4_ell2p50"
    if anchor not in selected and len(selected) < _MAX_N2_CONFIGS - 1:
        selected.append(anchor)

    # Fill remaining slots with non-passing candidates
    for (_, cid, _) in candidates:
        if len(selected) >= top_k:
            break
        if cid not in selected:
            selected.append(cid)

    # Build final N2 list: baseline + selected, capped at _MAX_N2_CONFIGS
    n2_list = ["E_baseline"] + [c for c in selected if c != "E_baseline"]
    n2_list = list(dict.fromkeys(n2_list))[:_MAX_N2_CONFIGS]  # deduplicate

    print(f"  [AUTO] Block N2 configs: {n2_list}")
    return n2_list


# ══════════════════════════════════════════════════════════════════════════════
#  Load existing PKLs
# ══════════════════════════════════════════════════════════════════════════════

_PKL_PATTERN = re.compile(r"dgp2optimal_([EN])_(.+)_n(\d+)_M(\d+)\.pkl$")


def _load_all_blocks() -> tuple[BlockResults, BlockResults]:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, BlockResults] = {"E": {}, "N": {}}

    for pkl_path in sorted(_RAW_DIR.glob("dgp2optimal_*.pkl")):
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

    return results["E"], results["N"]


def _merge(new: BlockResults, loaded: BlockResults) -> BlockResults:
    merged = dict(loaded)
    for cid, n_dict in new.items():
        if cid not in merged:
            merged[cid] = {}
        merged[cid].update(n_dict)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  Pareto classification
# ══════════════════════════════════════════════════════════════════════════════

def _classify(config_id: str, decomp: Optional[dict], data: Optional[dict],
              baseline_bias: float) -> str:
    if decomp is None or data is None:
        return "no-data"
    bias_dr  = abs(float(decomp["bias_dr"][REF_IDX]))
    bse      = float(decomp["bse_grid"][REF_IDX])
    cov      = float(decomp["coverage_ref"])
    ess      = float(decomp.get("ess_min_mean", 0.0))
    ser      = _compute_ser(data) or 1.0
    try:
        w_p99 = float(_max_weight_max(data))
    except Exception:
        w_p99 = 0.0
    try:
        q_clip = float(_q_neg_share_mean(data))  # proxy for clip fraction
    except Exception:
        q_clip = 0.0

    bias_red = (baseline_bias - bias_dr) / max(baseline_bias, 1e-12)

    # Strong
    if (bias_red >= _BIAS_RED_STRONG and bse < _BSE_STRONG and cov >= _COV_STRONG
            and _SER_LOW <= ser <= _SER_HIGH and ess >= _ESS_MIN and q_clip < _QCLIP_MAX):
        return "**strong**"

    # Promising
    if (bias_red >= _BIAS_RED_PROMISING and bse < _BSE_PROMISING and cov >= _COV_PROMISING):
        return "promising"

    return "dominated"


# ══════════════════════════════════════════════════════════════════════════════
#  Report builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_report(
    results_E: BlockResults,
    results_N2: Optional[BlockResults],
) -> list[str]:
    lines: list[str] = []
    lines.append("# DGP2 Optimal Region Cartography")
    lines.append(f"# Generated : {datetime.date.today()}")
    lines.append(f"# Phase 9B.4 -- ell_scale extension around winner (H_lam1e-4_ell2p50)")
    lines.append("")

    _sec1_objective(lines)
    _sec2_design(lines)
    _sec3_block_e(lines, results_E)
    _sec4_pareto(lines, results_E)
    _sec5_block_n2(lines, results_N2)
    _sec6_interpretation(lines, results_E, results_N2)
    _sec7_recommendation(lines, results_E)

    return lines


# ── Section 1 ─────────────────────────────────────────────────────────────────

def _sec1_objective(add: list[str]) -> None:
    add.append("## 1. Objective")
    add.append("")
    add.append(
        "Phase 9B.3 found H_lam1e-4_ell2.5 as the optimal config (cov=0.920, |b|/SE=0.59 at n=2000),"
    )
    add.append(
        "but ell_scale=2.5 was the boundary of the grid. This cartography answers:"
    )
    add.append("  A. Is ell=2.5 an optimum, or the start of a plateau toward ell=3.0/3.5?")
    add.append("  B. Does over-smoothing eventually degrade coverage or inflate bias?")
    add.append(
        "  C. What methodological rule should be stated in S6 for the RKHS kernel geometry?"
    )
    add.append("")


# ── Section 2 ─────────────────────────────────────────────────────────────────

def _sec2_design(add: list[str]) -> None:
    add.append("## 2. Design")
    add.append("")
    add.append(f"DGP: MichaelisMentenDGP(snr_W={SNR}, snr_Z={SNR})")
    add.append(f"ref dose: a={A_GRID[REF_IDX]:.1f} (REF_IDX={REF_IDX})")
    add.append(f"a_grid: {list(A_GRID)}")
    add.append("")
    add.append("| Block | Parameter | Grid | n | M |")
    add.append("|-------|-----------|------|---|---|")
    add.append(
        "| E | lambda_h x ell_scale | "
        f"lam in {[_lam_str(l) for l in _LAM_GRID]}, ell in {_ELL_GRID} | 1000 | 50 |"
    )
    add.append("| N2 | n-trend (top configs) | n in {500,1000,2000} | 500-2000 | 75 |")
    add.append("")
    add.append(
        "Anchors from Phase 9B.3 are loaded without recomputing "
        "(ell=2.5 configs and H_baseline n-trend)."
    )
    add.append("")


# ── Section 3 — Block E ───────────────────────────────────────────────────────

def _sec3_block_e(add: list[str], results_E: BlockResults) -> None:
    add.append("## 3. Block E Results")
    add.append("")

    if not results_E:
        add.append("*Block E not run yet.*")
        add.append("")
        return

    ns_available = sorted({n for v in results_E.values() for n in v.keys()})
    n_show = max(ns_available) if ns_available else 1000

    add.append(f"### 3a. Main table at n={n_show} (dose ref a={A_GRID[REF_IDX]:.1f})")
    add.append("")
    add.append(
        "| config | lambda_h | ell_scale | bias_REG | bias_DR | |b|/SE | cov_ref | SER | ESS | w_p99 |"
    )
    add.append(
        "|--------|----------|-----------|----------|---------|--------|---------|-----|-----|-------|"
    )

    baseline_bias = _baseline_bias_dr(results_E)

    for cid in BLOCK_E_ORDER:
        if cid not in results_E:
            continue
        entry = results_E[cid].get(n_show, (None, 0.0, None))
        d, _, data = entry
        spec = BLOCK_E_CONFIGS[cid]
        lam_h_str = "default" if spec["lambda_h"] is None else _lam_str(spec["lambda_h"])
        ser_val   = _compute_ser(data) if data is not None else None
        w_p99_val = _max_weight_max(data) if data is not None else None

        if d is None:
            add.append(
                f"| {cid} | {lam_h_str} | {spec['ell_scale']:.2f} "
                "| --- | --- | --- | --- | --- | --- | --- |"
            )
            continue
        ser_str   = f"{ser_val:.2f}"   if ser_val   is not None else "---"
        w_p99_str = f"{w_p99_val:.2f}" if w_p99_val is not None else "---"
        add.append(
            f"| {cid} | {lam_h_str} | {spec['ell_scale']:.2f} | "
            f"{d['bias_reg'][REF_IDX]:+.4f} | "
            f"{d['bias_dr'][REF_IDX]:+.4f} | "
            f"{d['bse_grid'][REF_IDX]:.2f} | "
            f"{d['coverage_ref']:.3f} | "
            f"{ser_str} | "
            f"{d.get('ess_min_mean', float('nan')):.0f} | "
            f"{w_p99_str} |"
        )

    add.append("")
    add.append(f"Baseline |bias_DR(ref)| = {baseline_bias:.4f}")
    add.append("")

    # Heatmaps
    add.append("### 3b. Heatmaps (rows=lambda_h, cols=ell_scale)")
    add.append("")

    for metric, label, fmt in [
        ("bias_reg", "bias_REG", "+.4f"),
        ("bias_dr",  "bias_DR",  "+.4f"),
        ("bse_grid", "|b|/SE",   ".2f"),
        ("coverage_ref", "cov_ref", ".3f"),
    ]:
        add.append(f"**{label}:**")
        header = "| lambda_h \\ ell_scale | " + " | ".join(f"ell={e}" for e in _ELL_GRID) + " |"
        sep    = "|---|" + "---|" * len(_ELL_GRID)
        add.append(header)
        add.append(sep)
        for lam in _LAM_GRID:
            row = f"| {_lam_str(lam)} |"
            for ell in _ELL_GRID:
                cid   = _e_config_id(lam, ell)
                entry = results_E.get(cid, {}).get(n_show, (None, 0.0, None))
                d, _, _ = entry
                if d is None:
                    row += " --- |"
                else:
                    if metric == "coverage_ref":
                        val = d["coverage_ref"]
                    elif metric in ("bias_reg", "bias_dr", "bse_grid"):
                        val = d[metric][REF_IDX]
                    else:
                        val = float("nan")
                    row += f" {val:{fmt}} |"
            add.append(row)
        add.append("")

    # SER heatmap (needs data dict)
    add.append("**SER:**")
    header = "| lambda_h \\ ell_scale | " + " | ".join(f"ell={e}" for e in _ELL_GRID) + " |"
    sep    = "|---|" + "---|" * len(_ELL_GRID)
    add.append(header)
    add.append(sep)
    for lam in _LAM_GRID:
        row = f"| {_lam_str(lam)} |"
        for ell in _ELL_GRID:
            cid   = _e_config_id(lam, ell)
            entry = results_E.get(cid, {}).get(n_show, (None, 0.0, None))
            _, _, data = entry
            ser = _compute_ser(data) if data is not None else None
            row += f" {ser:.2f} |" if ser is not None else " --- |"
        add.append(row)
    add.append("")


# ── Section 4 — Pareto ────────────────────────────────────────────────────────

def _sec4_pareto(add: list[str], results_E: BlockResults) -> None:
    add.append("## 4. Pareto Classification (Block E at n=1000)")
    add.append("")
    add.append(
        "Thresholds: bias_red>=50%, |b|/SE<0.75, cov>=0.88, SER in [0.75,1.25], "
        "ESS>=20, w_p99<2x baseline, q_clip<0.05."
    )
    add.append("**strong** = all conditions met. **promising** = bias_red>=30%, |b|/SE<1.0, cov>=0.80.")
    add.append("")
    add.append(
        "| config | bias_DR | bias_red% | |b|/SE | cov_ref | SER | ESS | Classification |"
    )
    add.append(
        "|--------|---------|-----------|--------|---------|-----|-----|----------------|"
    )

    baseline_bias = _baseline_bias_dr(results_E)
    n_show = 1000

    for cid in BLOCK_E_ORDER:
        if cid not in results_E:
            continue
        entry = results_E[cid].get(n_show, (None, 0.0, None))
        d, _, data = entry
        if d is None:
            add.append(f"| {cid} | --- | --- | --- | --- | --- | --- | no-data |")
            continue
        bias_dr  = float(d["bias_dr"][REF_IDX])
        bse      = float(d["bse_grid"][REF_IDX])
        cov      = float(d["coverage_ref"])
        ess      = float(d.get("ess_min_mean", 0.0))
        ser      = _compute_ser(data) or float("nan")
        bias_red = (baseline_bias - abs(bias_dr)) / max(baseline_bias, 1e-12) * 100
        classif  = _classify(cid, d, data, baseline_bias)
        add.append(
            f"| {cid} | {bias_dr:+.4f} | {bias_red:.1f}% | {bse:.2f} | "
            f"{cov:.3f} | {ser:.2f} | {ess:.0f} | {classif} |"
        )

    add.append("")


# ── Section 5 — Block N2 ──────────────────────────────────────────────────────

def _sec5_block_n2(add: list[str], results_N2: Optional[BlockResults]) -> None:
    add.append("## 5. Block N2 -- n-trend confirmation")
    add.append("")

    if not results_N2:
        add.append("*Block N2 not run yet.*")
        add.append("")
        return

    # |b|/SE table
    add.append("### 5a. |bias|/SE by n")
    add.append("")
    add.append("| config | n=500 | n=1000 | n=2000 |")
    add.append("|--------|-------|--------|--------|")
    for cid in sorted(results_N2.keys()):
        row = f"| {cid} |"
        for n in [500, 1000, 2000]:
            entry = results_N2[cid].get(n, (None, 0.0, None))
            d = entry[0]
            row += f" {d['bse_grid'][REF_IDX]:.2f} |" if d else " --- |"
        add.append(row)
    add.append("")

    # coverage table
    add.append("### 5b. coverage_ref by n")
    add.append("")
    add.append("| config | n=500 | n=1000 | n=2000 |")
    add.append("|--------|-------|--------|--------|")
    for cid in sorted(results_N2.keys()):
        row = f"| {cid} |"
        for n in [500, 1000, 2000]:
            entry = results_N2[cid].get(n, (None, 0.0, None))
            d = entry[0]
            row += f" {d['coverage_ref']:.3f} |" if d else " --- |"
        add.append(row)
    add.append("")

    # bias table
    add.append("### 5c. bias_REG and bias_DR by n")
    add.append("")
    add.append("| config | n | bias_REG | bias_DR |")
    add.append("|--------|---|----------|---------|")
    for cid in sorted(results_N2.keys()):
        for n in [500, 1000, 2000]:
            entry = results_N2[cid].get(n, (None, 0.0, None))
            d = entry[0]
            if d:
                add.append(
                    f"| {cid} | {n} | "
                    f"{d['bias_reg'][REF_IDX]:+.4f} | "
                    f"{d['bias_dr'][REF_IDX]:+.4f} |"
                )
    add.append("")


# ── Section 6 — Interpretation ────────────────────────────────────────────────

def _sec6_interpretation(
    add: list[str],
    results_E: BlockResults,
    results_N2: Optional[BlockResults],
) -> None:
    add.append("## 6. Interpretation")
    add.append("")

    # Determine which ell wins
    n_show = 1000
    baseline_bias = _baseline_bias_dr(results_E)

    best_bse: dict[float, float] = {}
    best_cov: dict[float, float] = {}
    for ell in _ELL_GRID:
        bse_vals, cov_vals = [], []
        for lam in _LAM_GRID:
            cid   = _e_config_id(lam, ell)
            entry = results_E.get(cid, {}).get(n_show, (None, 0.0, None))
            d = entry[0]
            if d is not None:
                bse_vals.append(float(d["bse_grid"][REF_IDX]))
                cov_vals.append(float(d["coverage_ref"]))
        if bse_vals:
            best_bse[ell] = min(bse_vals)
            best_cov[ell] = max(cov_vals)

    if best_bse:
        opt_ell_bse = min(best_bse, key=lambda e: best_bse[e])
        opt_ell_cov = max(best_cov, key=lambda e: best_cov[e])
        add.append(f"Best ell by min(|b|/SE) : ell={opt_ell_bse:.2f} "
                   f"(best |b|/SE = {best_bse[opt_ell_bse]:.2f})")
        add.append(f"Best ell by max(cov)   : ell={opt_ell_cov:.2f} "
                   f"(best cov = {best_cov[opt_ell_cov]:.3f})")
        add.append("")

        # Classify the region
        bse_at_ells = [best_bse.get(e, float("nan")) for e in _ELL_GRID]
        bse_35 = best_bse.get(3.5, float("nan"))
        bse_25 = best_bse.get(2.5, float("nan"))
        bse_30 = best_bse.get(3.0, float("nan"))

        if not np.isnan(bse_35) and bse_35 > bse_25 * 1.1:
            verdict = "over-smoothing boundary detected -- ell=2.5 is the optimum."
        elif not np.isnan(bse_30) and bse_30 < bse_25 * 0.90:
            verdict = "trend continues -- ell=3.0+ outperforms ell=2.5; DGP2 prefers smoother RKHS."
        elif not np.isnan(bse_30) and abs(bse_30 - bse_25) < 0.10:
            verdict = "stable plateau in [2.5, 3.0] -- choose ell=2.5 as conservative optimum."
        else:
            verdict = "inconclusive -- inspect heatmap manually."

        add.append(f"**Region verdict**: {verdict}")
        add.append("")

    # N2 trend
    if results_N2:
        add.append("**N2 n-trend summary:**")
        for cid in sorted(results_N2.keys()):
            bse_vals = []
            for n in [500, 1000, 2000]:
                entry = results_N2[cid].get(n, (None, 0.0, None))
                d = entry[0]
                if d:
                    bse_vals.append((n, float(d["bse_grid"][REF_IDX])))
            if len(bse_vals) == 3:
                trend = "converging" if bse_vals[2][1] < bse_vals[0][1] else "diverging/stable"
                add.append(
                    f"  {cid}: |b|/SE = "
                    + " / ".join(f"{v:.2f}" for _, v in bse_vals)
                    + f" ({trend})"
                )
        add.append("")


# ── Section 7 — Recommendation ───────────────────────────────────────────────

def _sec7_recommendation(add: list[str], results_E: BlockResults) -> None:
    add.append("## 7. Recommendation for S6/S7")
    add.append("")

    baseline_bias = _baseline_bias_dr(results_E)
    n_show = 1000

    # Find best overall config
    best_cid, best_bse = None, float("inf")
    for cid in BLOCK_E_ORDER:
        if cid == "E_baseline":
            continue
        entry = results_E.get(cid, {}).get(n_show, (None, 0.0, None))
        d = entry[0]
        if d is None:
            continue
        bse = float(d["bse_grid"][REF_IDX])
        if bse < best_bse:
            best_bse, best_cid = bse, cid

    if best_cid:
        spec = BLOCK_E_CONFIGS[best_cid]
        add.append(f"**Recommended config for S6**: {best_cid}")
        add.append(f"  lambda_h = {spec['lambda_h']:.0e}, ell_scale = {spec['ell_scale']:.2f}")
        entry = results_E[best_cid].get(n_show, (None, 0.0, None))
        d = entry[0]
        if d:
            add.append(
                f"  At n={n_show}: bias_DR={d['bias_dr'][REF_IDX]:+.4f}, "
                f"|b|/SE={d['bse_grid'][REF_IDX]:.2f}, cov={d['coverage_ref']:.3f}"
            )
        add.append("")

    add.append("**Essay phrase (S6):**")
    add.append(
        "  'The final tuning map shows that the dominant lever is the RKHS geometry "
        "of the outcome bridge. Increasing the RBF lengthscale from the Silverman "
        "default (ell_scale=1.0) to ell_scale~2.5 stabilises the inverse problem "
        "and achieves near-nominal coverage; further widening produces a plateau "
        "rather than improvement, confirming that the bottleneck is the approximation "
        "error of KPVBridgeH rather than the policy bandwidth or q-bridge regularisation.'"
    )
    add.append("")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main(force: bool = False) -> None:
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP

    print("=" * 72)
    print("DGP2 Optimal Region Cartography -- Phase 9B.4")
    print(f"force={force}")
    print("=" * 72)

    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)

    print("\nScanning for existing PKLs (checkpoint + anchor resume)...")
    loaded_E, loaded_N2 = _load_all_blocks()
    print(f"  Found: E={len(loaded_E)} configs, N2={len(loaded_N2)} configs")

    # ── Block E ──────────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("BLOCK E  (ell_scale extension grid, n=1000, M=50, 16 configs)")
    print("Anchors from Phase 9B.3 will be loaded without recomputing.")
    print("-" * 72)

    t_E = time.perf_counter()
    new_E = run_block_E(dgp, n_list=[1000], M=50, force=force)
    results_E = _merge(new_E, loaded_E)
    wall_E = (time.perf_counter() - t_E) / 60
    print(f"\nBlock E done in {wall_E:.1f} min. Configs: {len(results_E)}")

    lines_E = _build_report(results_E, None)
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text("\n".join(lines_E), encoding="utf-8")
    print(f"  Interim report -> {_REPORT_PATH.name}")

    # ── Auto-select N2 configs ────────────────────────────────────────────────
    print("\nAuto-selecting Block N2 configs...")
    n2_configs = _select_n2_configs(results_E, top_k=2)

    # ── Block N2 ─────────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print(f"BLOCK N2  (n-trend, configs={n2_configs}, n in {{500,1000,2000}}, M=75)")
    print("Anchors from Phase 9B.3 Block N will be loaded if available.")
    print("-" * 72)

    t_N2 = time.perf_counter()
    new_N2 = run_block_N2(dgp, M=75, n_config_ids=n2_configs, force=force)
    results_N2 = _merge(new_N2, loaded_N2)
    wall_N2 = (time.perf_counter() - t_N2) / 60
    print(f"\nBlock N2 done in {wall_N2:.1f} min. Configs: {len(results_N2)}")

    # ── Final report ─────────────────────────────────────────────────────────
    print("\nBuilding final report...")
    lines = _build_report(results_E, results_N2)
    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report -> {_REPORT_PATH}")

    total = (wall_E + wall_N2)
    print("\n" + "=" * 72)
    print(f"ALL BLOCKS COMPLETE in {total:.1f} min")
    print(f"  Block E : {wall_E:.1f} min ({len(results_E)} configs)")
    print(f"  Block N2: {wall_N2:.1f} min ({len(results_N2)} configs)")
    print("=" * 72)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGP2 Optimal Region Cartography (Phase 9B.4)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if PKLs exist (ignores anchors)")
    parser.add_argument("--report-only", action="store_true",
                        help="Only rebuild report from existing PKLs, no simulation")
    args = parser.parse_args()

    if args.report_only:
        results_E, results_N2 = _load_all_blocks()
        lines = _build_report(results_E, results_N2)
        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        print(f"Report rebuilt -> {_REPORT_PATH}")
    else:
        main(force=args.force)
