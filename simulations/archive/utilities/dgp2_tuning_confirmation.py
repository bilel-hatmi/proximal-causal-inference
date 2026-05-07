"""
DGP2 Tuning Confirmation Pilot (M=200) — Phase 9B.2bis

Context
-------
Prior diagnostics in `dgp2_bias_diagnostics.py` (Diag A/B/C, M=5..30) suggested
that several hyperparameter levers reduce the DGP2 DRKernel xfit-q bias:
    Config A  (lambda_h=1e-4, ell_scale=2.0)         -57% |bias_REG|
    Config C  (h_policy_scale=1.5)                   coverage ~0.97 (smoothed estimand)
    Config B  (lambda_Q=1e-2, clip=20)               q-correction more active

This script CONFIRMS these effects at M=200 — the noise floor at M=30 is too
high to commit to a "DGP2 main experiment" decision. After this run, we trigger
a binary GO/NO-GO:
    GO tuned     : DGP2 promoted to S6 main experiment with the winning config
    GO smoothed  : DGP2 included via h_policy=1.5*Silverman (estimand change documented)
    NO-GO        : DGP2 relegated to S7 stress-test section

Reuse
-----
This script imports helpers from `dgp2_bias_diagnostics.py` (SEPARATE script —
do not modify). The previous diag pkls in `simulations/results/raw/dgp2_bias_diag/`
are NOT touched (different label scheme + different output dir).

Modes
-----
    quick   : n in {500, 1000},  M=10  (~12 min total) — pipeline smoke
    main    : n in {500, 1000, 2000}, M=200 (~3.5h with 5 parallel terminals)
    single  : one (config, n) at user-supplied M (advanced)

Parallel strategy
-----------------
For main mode, launch 5 PowerShell terminals (one per config), each:

    python -m simulations.archive.utilities.dgp2_tuning_confirmation --mode main --config baseline
    python -m simulations.archive.utilities.dgp2_tuning_confirmation --mode main --config A
    python -m simulations.archive.utilities.dgp2_tuning_confirmation --mode main --config B
    python -m simulations.archive.utilities.dgp2_tuning_confirmation --mode main --config C
    python -m simulations.archive.utilities.dgp2_tuning_confirmation --mode main --config AC

Each terminal runs sequentially across n in {500, 1000, 2000} for its config.
After all 15 pkls are produced, run once more without --config to write the report:

    python -m simulations.archive.utilities.dgp2_tuning_confirmation --mode main

The main() detects existing pkls and skips them, only building the report.

Output
------
    simulations/results/raw/dgp2_tuning_confirm/dgp2_tunconf_<config>_n<N>_M<M>.pkl
    simulations/results/summaries/dgp2_tuning_confirmation.md

See `skills/pci-simulation/references/current/compute_management.md` §5 (pre-run
checklist) before launching main mode.
"""
from __future__ import annotations

import argparse
import datetime
import pickle
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

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parents[2]
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "dgp2_tuning_confirm"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "dgp2_tuning_confirmation.md"

# ── Modes ─────────────────────────────────────────────────────────────────────
_MODES: dict[str, tuple[list[int], int]] = {
    "quick": ([500, 1000], 10),
    "main": ([500, 1000, 2000], 200),
}

# ── GO/NO-GO thresholds (centralized → easy to retune) ────────────────────────
_GO_BIAS_REG_REDUCTION_PCT = 0.30   # |bias_REG| must shrink ≥ 30% vs baseline
_GO_BSE_MAX = 1.0                    # |bias_DR|/SE ≤ 1.0
_GO_COVERAGE_TUNED = 0.85            # coverage_ref ≥ 0.85 (configs A/AC)
_GO_COVERAGE_SMOOTHED = 0.90         # coverage_ref ≥ 0.90 (configs C/AC, smoothed estimand)
_GO_ESS_MIN = 20.0                   # ESS_min_mean ≥ 20
_GO_W_P99_MAX_RATIO = 5.0            # w_p99 < 5× baseline w_p99
_GO_QCLIP_MAX = 0.10                 # q_clip_fraction < 0.10
_GO_M_OK_RATIO = 0.80                # M_ok / M ≥ 0.80 (else flag config NO-GO)


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration table — single source of truth (5 configs, NOT extensible)
# ══════════════════════════════════════════════════════════════════════════════

CONFIGS: dict[str, dict] = {
    "baseline": dict(
        lambda_h=None, ell_scale=1.0,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT,
        h_policy_scale=1.0,
        description="Reference run: KPV defaults, no tuning.",
    ),
    "A": dict(
        lambda_h=1e-4, ell_scale=2.0,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT,
        h_policy_scale=1.0,
        description="h-bridge tuning: smaller lambda_h + wider RBF lengthscale.",
    ),
    "B": dict(
        lambda_h=None, ell_scale=1.0,
        lambda_Q=1e-2, clip=20.0,
        h_policy_scale=1.0,
        description="q-bridge tuning: larger lambda_Q + tighter clip.",
    ),
    "C": dict(
        lambda_h=None, ell_scale=1.0,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT,
        h_policy_scale=1.5,
        description="Policy-smoothed: 1.5x Silverman bandwidth (changes estimand).",
    ),
    "AC": dict(
        lambda_h=1e-4, ell_scale=2.0,
        lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT,
        h_policy_scale=1.5,
        description="A + C combined: h-tuned + policy-smoothed.",
    ),
}
_CONFIG_ORDER: list[str] = ["baseline", "A", "B", "C", "AC"]


# ══════════════════════════════════════════════════════════════════════════════
#  Factory builder — one function for all configs
# ══════════════════════════════════════════════════════════════════════════════

def make_factory(
    config_spec: dict,
    a_grid: np.ndarray,
    h_ref: float,
    ell_W0: float,
    ell_A0: float,
    ell_Z0: float,
) -> Callable:
    """
    Build a zero-arg factory closing over a frozen config and per-n bandwidths.

    h_ref  : Silverman bandwidth at this n (computed by caller from a fresh sample).
    ell_*0 : median-bandwidth heuristic at this n (computed by caller).
    h_policy = h_ref * config_spec["h_policy_scale"] is applied SIMULTANEOUSLY to
        - KPVPolicyBridgeQ.h_KDE
        - DRKernel.bandwidth
    so the K_h convolution kernel is identical on both sides of the DR score.
    For h_policy_scale != 1.0, this changes the target estimand to J(pi_{a, h_policy}).
    The Jensen gap section in the report makes this explicit.
    """
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ

    ell_scale = float(config_spec["ell_scale"])
    h_policy = float(h_ref) * float(config_spec["h_policy_scale"])
    lam_h = config_spec["lambda_h"]
    lam_q = float(config_spec["lambda_Q"])
    clip = config_spec["clip"]

    bridge_kwargs: dict = {}
    if lam_h is not None:
        bridge_kwargs["lambda_1"] = float(lam_h)
        bridge_kwargs["lambda_2"] = float(lam_h)
    if (lam_h is not None) or (ell_scale != 1.0):
        bridge_kwargs["ell_W"] = float(ell_W0) * ell_scale
        bridge_kwargs["ell_A"] = float(ell_A0) * ell_scale
        bridge_kwargs["ell_Z"] = float(ell_Z0) * ell_scale

    def factory():
        q_model = KPVPolicyBridgeQ(
            a_grid=a_grid,
            h_KDE=h_policy,
            lambda_Q=lam_q,
            clip=clip,
            # compute_cond defaults to False (KPVPolicyBridgeQ since 2026-04-26)
        )
        return DRKernel(
            a_grid=a_grid,
            q_model=q_model,
            bandwidth=h_policy,
            n_folds=N_FOLDS,
            random_state=RANDOM_STATE,
            ref_dose_index=REF_IDX,
            cross_fit_q=True,
            bridge_kwargs=(bridge_kwargs if bridge_kwargs else None),
        )

    return factory


# ══════════════════════════════════════════════════════════════════════════════
#  Per-block runner
# ══════════════════════════════════════════════════════════════════════════════

def _seed_base(config_name: str, n: int) -> int:
    """Disjoint from prior runs (5M / 5.9M / 6M / 6.9M / 7M / 8M reserved)."""
    cfg_idx = _CONFIG_ORDER.index(config_name)
    return 9_000_000 + cfg_idx * 100_000 + int(n)


def _label(config_name: str, n: int, M: int) -> str:
    return f"dgp2_tunconf_{config_name}_n{n}_M{M}"


def _run_one_block(
    dgp,
    config_name: str,
    n: int,
    M: int,
    force: bool = False,
) -> tuple[Optional[dict], float, dict | None]:
    """
    Run one (config × n) block. Returns (decomp_dict, wall_seconds, raw_data).

    raw_data is the pkl-loaded dict with records + meta — useful for SER and
    per-dose coverage computation in the report builder.
    """
    spec = CONFIGS[config_name]

    # Reference sample at this n for bandwidth + lengthscales
    ref_sample = dgp.generate(n=n, seed=0)
    h_ref = _silverman_h(ref_sample.A)
    ell_W0 = _median_bandwidth(ref_sample.W)
    ell_A0 = _median_bandwidth(ref_sample.A)
    ell_Z0 = _median_bandwidth(ref_sample.Z)

    # Per-config J_policy_true (depends on h_policy_scale)
    h_policy = h_ref * float(spec["h_policy_scale"])
    J_pt = _compute_J_policy_true(dgp, A_GRID, h_policy)

    factory = make_factory(spec, A_GRID, h_ref, ell_W0, ell_A0, ell_Z0)

    label = _label(config_name, n, M)
    seed_base = _seed_base(config_name, n)

    if force:
        for sfx in (".pkl", ".partial.pkl"):
            p = _RAW_DIR / f"{label}{sfx}"
            if p.exists():
                p.unlink()
                print(f"  [FORCE] removed {p.name}")

    t0 = time.perf_counter()
    data = _run_block(
        dgp, factory, A_GRID, J_pt,
        n=n, M=M, seed_base=seed_base, label=label, raw_dir=_RAW_DIR,
    )
    wall = time.perf_counter() - t0

    return _decompose(data), wall, data


# ══════════════════════════════════════════════════════════════════════════════
#  Per-block analytical helpers (operate on raw data / records)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_ser(data: dict) -> float:
    """SER = mean(SE_hat_ref) / sd(J_dr_ref). 1.0 = calibrated."""
    if data is None:
        return float("nan")
    records = [r for r in data["records"] if r.get("error") is None]
    if len(records) < 2:
        return float("nan")
    n = int(data["meta"]["n"])
    J_dr_ref = np.array([r["J_dr"][REF_IDX] for r in records])
    V_ref = np.array([r["V_hat_grid"][REF_IDX] for r in records])
    SE_ref = np.sqrt(np.abs(V_ref) / n)
    sd_psi = float(np.std(J_dr_ref, ddof=1))
    mean_se = float(np.nanmean(SE_ref))
    if sd_psi <= 0:
        return float("nan")
    return mean_se / sd_psi


def _per_dose_empirical_coverage(data: dict) -> np.ndarray:
    """Empirical coverage per dose using V_hat_grid for SE per rep."""
    if data is None:
        return np.full(K, np.nan)
    records = [r for r in data["records"] if r.get("error") is None]
    if not records:
        return np.full(K, np.nan)
    n = int(data["meta"]["n"])
    J_pt = np.array(data["meta"]["J_policy_true"])
    J_dr_all = np.array([r["J_dr"] for r in records])           # (M, K)
    V_hg_all = np.array([r["V_hat_grid"] for r in records])
    SE_all = np.sqrt(np.abs(V_hg_all) / n)
    cov = np.zeros(K)
    for k in range(K):
        cov[k] = float(np.mean(np.abs(J_dr_all[:, k] - J_pt[k]) <= 1.96 * SE_all[:, k]))
    return cov


def _per_dose_se(data: dict) -> np.ndarray:
    """Mean per-dose SE = sqrt(mean(V_hat_grid)/n)."""
    if data is None:
        return np.full(K, np.nan)
    records = [r for r in data["records"] if r.get("error") is None]
    if not records:
        return np.full(K, np.nan)
    n = int(data["meta"]["n"])
    V_hg_all = np.array([r["V_hat_grid"] for r in records])
    return np.sqrt(np.abs(np.nanmean(V_hg_all, axis=0)) / n)


def _max_weight_max(data: dict) -> float:
    """Max over reps of max over doses of weight_max_grid."""
    if data is None:
        return float("nan")
    records = [r for r in data["records"] if r.get("error") is None]
    if not records:
        return float("nan")
    arr = np.array([r["weight_max_grid"] for r in records])  # (M, K)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.nanmax(arr))


def _q_neg_share_mean(data: dict) -> float:
    """Mean q_negative_share across reps and doses."""
    if data is None:
        return float("nan")
    records = [r for r in data["records"] if r.get("error") is None]
    if not records:
        return float("nan")
    arr = np.array([r["q_negative_share_grid"] for r in records])  # (M, K)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.nanmean(arr))


def _M_ok_ratio(data: dict, M_target: int) -> float:
    if data is None:
        return float("nan")
    M_ok = sum(1 for r in data["records"] if r.get("error") is None)
    return M_ok / max(M_target, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  Main orchestration
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_cli()

    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)

    # OneDrive sync warning
    if "OneDrive" in str(_RAW_DIR):
        print(
            "[WARN] Output directory is under OneDrive — risk of .partial.pkl "
            "corruption. Monitor file sizes; pause OneDrive for long runs."
        )

    # Determine configs and n list
    if args.mode == "single":
        cfgs = [args.config]
        ns = [args.n]
        M = args.M
    else:
        cfgs = [args.config] if args.config else _CONFIG_ORDER
        ns, M = _MODES[args.mode]

    print(f"\n=== DGP2 Tuning Confirmation ===")
    print(f"Mode    : {args.mode}")
    print(f"Configs : {cfgs}")
    print(f"n grid  : {ns}")
    print(f"M       : {M}")
    print(f"Output  : {_RAW_DIR}")
    print(f"Report  : {_REPORT_PATH}")

    # Storage: results[cfg][n] = decomp_dict | None ; data_cache[cfg][n] = raw data
    results: dict[str, dict[int, Optional[dict]]] = {c: {} for c in cfgs}
    walls: dict[str, dict[int, float]] = {c: {} for c in cfgs}
    data_cache: dict[str, dict[int, Optional[dict]]] = {c: {} for c in cfgs}

    for cfg in cfgs:
        for n in ns:
            print(f"\n--- Config {cfg} | n={n} | M={M} ---")
            decomp, wall, raw = _run_one_block(dgp, cfg, n, M, force=args.force)
            results[cfg][n] = decomp
            walls[cfg][n] = wall
            data_cache[cfg][n] = raw
            if decomp is not None:
                print(f"    bias_DR(ref)={decomp['bias_dr'][REF_IDX]:+.4f}  "
                      f"|b|/SE={decomp['bse_grid'][REF_IDX]:.2f}  "
                      f"cov={decomp['coverage_ref']:.3f}  "
                      f"wall={wall/60:.1f} min")
            else:
                print(f"    [WARN] block {cfg}/n={n} returned no successful reps "
                      f"(wall={wall/60:.1f} min)")

    # Build report only if running multi-config sweep
    if args.mode != "single":
        # Reload pkls for any configs we DID NOT run this terminal (skipped if already done)
        if args.config is not None:
            print(f"\n[Note] Single-config terminal — report build skipped.")
            print(f"       Run without --config once all 15 pkls exist to build report.")
            return

        print(f"\n[Report] Building markdown ...")
        lines = _build_report(results, walls, data_cache, dgp, ns, M)
        _SUMM_DIR.mkdir(parents=True, exist_ok=True)
        _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        print(f"[Report] -> {_REPORT_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
#  Report builder — 10 sections per the user brief
# ══════════════════════════════════════════════════════════════════════════════

def _build_report(
    results: dict[str, dict[int, Optional[dict]]],
    walls: dict[str, dict[int, float]],
    data_cache: dict[str, dict[int, Optional[dict]]],
    dgp,
    ns: list[int],
    M: int,
) -> list[str]:
    lines: list[str] = []

    lines.append("# DGP2 Tuning Confirmation — Pilot M={}".format(M))
    lines.append(f"# Generated : {datetime.date.today()}")
    lines.append(f"# SNR={SNR}, Michaelis-Menten, ref dose a={A_GRID[REF_IDX]} (REF_IDX={REF_IDX})")
    lines.append(f"# n grid : {ns}, configs : {list(results)}")
    lines.append("")
    lines.append("> NOTE — Configs C and AC use h_policy=1.5*Silverman, which CHANGES the estimand to")
    lines.append("> J(pi_{a, 1.5h}). Their coverage_ref measures coverage of the smoothed target,")
    lines.append("> not of J(pi_{a, h}). The Jensen gap (Section 5) makes this trade-off explicit.")
    lines.append("")

    _section_1_objectif(lines)
    _section_2_configs(lines)
    _section_3_main_table(lines, results, walls, data_cache, ns, M)
    _section_4_per_dose(lines, results, data_cache, ns)
    _section_5_jensen_gap(lines, data_cache, dgp, ns)
    _section_6_variance(lines, results, ns)
    _section_7_q_diagnostics(lines, results, data_cache, ns)
    _section_8_compute(lines, walls, ns)
    decision = _section_9_go_nogo(lines, results, data_cache, ns, M)
    _section_10_implication(lines, decision)
    return lines


# ── Section 1 ────────────────────────────────────────────────────────────────

def _section_1_objectif(add: list[str]) -> None:
    add.append("## 1. Objectif")
    add.append("")
    add.append("Confirmer a M=200 si les leviers de tuning identifies en M=5..30 (Diag A/B/C dans")
    add.append("dgp2_bias_diagnostics) survivent au passage a un MC plus large. Decision binaire :")
    add.append("DGP2 promu en experience principale (config tuned ou smoothed) ou relegue en stress-test.")
    add.append("")


# ── Section 2 ────────────────────────────────────────────────────────────────

def _section_2_configs(add: list[str]) -> None:
    add.append("## 2. Configurations testees")
    add.append("")
    add.append("| Config | lambda_h | ell_scale | lambda_Q | clip | h_policy_scale | Description |")
    add.append("|--------|----------|-----------|----------|------|----------------|-------------|")
    for name in _CONFIG_ORDER:
        s = CONFIGS[name]
        lam_h = "default" if s["lambda_h"] is None else f"{s['lambda_h']:.0e}"
        clip = "default" if s["clip"] is None else f"{s['clip']:g}"
        add.append(
            f"| {name} | {lam_h} | {s['ell_scale']} | {s['lambda_Q']:.0e} | {clip} | "
            f"{s['h_policy_scale']} | {s['description']} |"
        )
    add.append("")


# ── Section 3 ────────────────────────────────────────────────────────────────

def _section_3_main_table(
    add: list[str],
    results: dict[str, dict[int, Optional[dict]]],
    walls: dict[str, dict[int, float]],
    data_cache: dict[str, dict[int, Optional[dict]]],
    ns: list[int],
    M: int,
) -> None:
    add.append("## 3. Table principale (config x n, dose ref a={:.1f})".format(A_GRID[REF_IDX]))
    add.append("")
    add.append("| config | n | bias_REG | bias_DR | |b|/SE | cov_ref | pred_cov | SER | ESS_min | q_clip | M_ok/M | wall (min) |")
    add.append("|--------|---|----------|---------|--------|---------|----------|-----|---------|--------|--------|------------|")
    for cfg in _CONFIG_ORDER:
        if cfg not in results:
            continue
        for n in ns:
            d = results[cfg].get(n)
            data = data_cache.get(cfg, {}).get(n)
            wall = walls.get(cfg, {}).get(n, float("nan"))
            if d is None:
                add.append(f"| {cfg} | {n} | (missing) | | | | | | | | | {_fmt(wall/60.0,'.1f')} |")
                continue
            ser = _compute_ser(data)
            m_ok = _M_ok_ratio(data, M)
            add.append(
                f"| {cfg} | {n} | "
                f"{_fmt(d['bias_reg'][REF_IDX], '+.4f')} | "
                f"{_fmt(d['bias_dr'][REF_IDX], '+.4f')} | "
                f"{_fmt(d['bse_grid'][REF_IDX], '.2f')} | "
                f"{_fmt(d['coverage_ref'], '.3f')} | "
                f"{_fmt(d['pred_cov_grid'][REF_IDX], '.3f')} | "
                f"{_fmt(ser, '.2f')} | "
                f"{_fmt(d['ess_min_mean'], '.0f')} | "
                f"{_fmt(d['q_clip_mean'], '.4f')} | "
                f"{_fmt(m_ok, '.2f')} | "
                f"{_fmt(wall/60.0, '.1f')} |"
            )
    add.append("")
    add.append("Legende : SER = mean(SE_hat) / sd(J_dr) (1.0 = calibre, <1 sous-estime, >1 sur-estime).")
    add.append("           pred_cov = couverture predite formule normale-biaisee.")
    add.append("           |b|/SE = abs(bias_DR_ref) / SE_ref (ratio biais/erreur-type).")
    add.append("")


# ── Section 4 ────────────────────────────────────────────────────────────────

def _section_4_per_dose(
    add: list[str],
    results: dict[str, dict[int, Optional[dict]]],
    data_cache: dict[str, dict[int, Optional[dict]]],
    ns: list[int],
) -> None:
    """Per-dose tables at the LARGEST n available (most informative)."""
    n_show = max(ns)
    add.append(f"## 4. Tables par dose (n={n_show})")
    add.append("")
    dose_hdr = " | ".join(f"a={a:.1f}" for a in A_GRID)
    sep = "|" + "---|" * (K + 1)

    add.append("### 4a. bias_DR par dose")
    add.append("")
    add.append(f"| config | {dose_hdr} |")
    add.append(sep)
    for cfg in _CONFIG_ORDER:
        if cfg not in results:
            continue
        d = results[cfg].get(n_show)
        if d is None:
            add.append(f"| {cfg} | " + " | ".join(["---"] * K) + " |")
            continue
        cells = " | ".join(_fmt(v, "+.4f") for v in d["bias_dr"])
        add.append(f"| {cfg} | {cells} |")
    add.append("")

    add.append("### 4b. SE par dose")
    add.append("")
    add.append(f"| config | {dose_hdr} |")
    add.append(sep)
    for cfg in _CONFIG_ORDER:
        if cfg not in results:
            continue
        d = results[cfg].get(n_show)
        if d is None:
            add.append(f"| {cfg} | " + " | ".join(["---"] * K) + " |")
            continue
        cells = " | ".join(_fmt(v, ".4f") for v in d["SE_grid"])
        add.append(f"| {cfg} | {cells} |")
    add.append("")

    add.append("### 4c. Coverage empirique par dose")
    add.append("")
    add.append(f"| config | {dose_hdr} |")
    add.append(sep)
    for cfg in _CONFIG_ORDER:
        if cfg not in results:
            continue
        data = data_cache.get(cfg, {}).get(n_show)
        cov = _per_dose_empirical_coverage(data)
        cells = " | ".join(_fmt(v, ".3f") for v in cov)
        add.append(f"| {cfg} | {cells} |")
    add.append("")


# ── Section 5 ────────────────────────────────────────────────────────────────

def _section_5_jensen_gap(
    add: list[str],
    data_cache: dict[str, dict[int, Optional[dict]]],
    dgp,
    ns: list[int],
) -> None:
    add.append("## 5. Jensen gap (J_true(a) - J_policy_true(a, h_policy))")
    add.append("")
    add.append("Jensen gap = ecart entre m_true(a) et la cible lissee J(pi_{a,h_policy}).")
    add.append("Pour configs C/AC, h_policy=1.5h_ref => Jensen gap plus grand : la couverture")
    add.append("vise une cible explicitement modifiee.")
    add.append("")
    n_show = max(ns)
    dose_hdr = " | ".join(f"a={a:.1f}" for a in A_GRID)
    sep = "|" + "---|" * (K + 1)
    add.append(f"| config (n={n_show}) | {dose_hdr} |")
    add.append(sep)
    J_true = np.array([float(dgp.m_true(float(a))) for a in A_GRID])
    for cfg in _CONFIG_ORDER:
        if cfg not in data_cache:
            continue
        data = data_cache[cfg].get(n_show)
        if data is None:
            add.append(f"| {cfg} | " + " | ".join(["---"] * K) + " |")
            continue
        J_pt = np.array(data["meta"]["J_policy_true"])
        gap = J_true - J_pt
        cells = " | ".join(_fmt(v, "+.5f") for v in gap)
        add.append(f"| {cfg} | {cells} |")
    add.append("")


# ── Section 6 ────────────────────────────────────────────────────────────────

def _section_6_variance(
    add: list[str],
    results: dict[str, dict[int, Optional[dict]]],
    ns: list[int],
) -> None:
    add.append("## 6. Variance decomposition (dose ref)")
    add.append("")
    add.append("V_total = V_reg + V_correction + 2*V_cov. % V_corr / V_total flagge si la correction")
    add.append("q est inerte (~0%) ou amplifie (>50%).")
    add.append("")
    add.append("| config | n | V_total | V_reg | V_corr | V_cov | %V_corr/V_total |")
    add.append("|--------|---|---------|-------|--------|-------|------------------|")
    for cfg in _CONFIG_ORDER:
        if cfg not in results:
            continue
        for n in ns:
            d = results[cfg].get(n)
            if d is None:
                add.append(f"| {cfg} | {n} | --- | --- | --- | --- | --- |")
                continue
            v_tot = float(d["V_total"][REF_IDX])
            v_reg = float(d["V_reg"][REF_IDX])
            v_corr = float(d["V_corr"][REF_IDX])
            v_cov = float(d["V_cov"][REF_IDX])
            pct = (v_corr / v_tot * 100.0) if abs(v_tot) > 1e-15 else float("nan")
            add.append(
                f"| {cfg} | {n} | "
                f"{_fmt(v_tot, '.4f')} | {_fmt(v_reg, '.4f')} | "
                f"{_fmt(v_corr, '.4f')} | {_fmt(v_cov, '.4f')} | "
                f"{_fmt(pct, '.1f')}% |"
            )
    add.append("")


# ── Section 7 ────────────────────────────────────────────────────────────────

def _section_7_q_diagnostics(
    add: list[str],
    results: dict[str, dict[int, Optional[dict]]],
    data_cache: dict[str, dict[int, Optional[dict]]],
    ns: list[int],
) -> None:
    add.append("## 7. q diagnostics (poids IPW, clipping, Riesz residual)")
    add.append("")
    add.append("| config | n | w_p99_mean | w_max_overall | q_clip_frac | q_neg_share | riesz_res |")
    add.append("|--------|---|------------|---------------|-------------|-------------|-----------|")
    for cfg in _CONFIG_ORDER:
        if cfg not in results:
            continue
        for n in ns:
            d = results[cfg].get(n)
            data = data_cache.get(cfg, {}).get(n)
            if d is None:
                add.append(f"| {cfg} | {n} | --- | --- | --- | --- | --- |")
                continue
            w_max = _max_weight_max(data)
            q_neg = _q_neg_share_mean(data)
            add.append(
                f"| {cfg} | {n} | "
                f"{_fmt(d['w_p99_mean'], '.2f')} | "
                f"{_fmt(w_max, '.2f')} | "
                f"{_fmt(d['q_clip_mean'], '.4f')} | "
                f"{_fmt(q_neg, '.3f')} | "
                f"{_fmt(d['riesz_res_mean'], '.4f')} |"
            )
    add.append("")


# ── Section 8 ────────────────────────────────────────────────────────────────

def _section_8_compute(
    add: list[str],
    walls: dict[str, dict[int, float]],
    ns: list[int],
) -> None:
    add.append("## 8. Wall-clock summary (minutes)")
    add.append("")
    n_hdr = " | ".join(f"n={n}" for n in ns)
    sep = "|" + "---|" * (len(ns) + 2)
    add.append(f"| config | {n_hdr} | total |")
    add.append(sep)
    grand_total = 0.0
    for cfg in _CONFIG_ORDER:
        if cfg not in walls:
            continue
        cells = []
        cfg_total = 0.0
        for n in ns:
            w = walls[cfg].get(n, 0.0)
            cfg_total += w
            cells.append(_fmt(w / 60.0, ".1f"))
        grand_total += cfg_total
        add.append(f"| {cfg} | " + " | ".join(cells) + f" | {cfg_total/60.0:.1f} |")
    add.append(f"| **total** | " + " | ".join(["---"] * len(ns)) + f" | **{grand_total/60.0:.1f}** |")
    add.append("")


# ── Section 9 — GO/NO-GO decision logic ──────────────────────────────────────

def _no_weight_blowup(d: dict, baseline: dict) -> bool:
    """True if w_p99 within 5x baseline AND q_clip < 0.10."""
    if d is None or baseline is None:
        return False
    bl_w99 = float(baseline["w_p99_mean"])
    if not np.isfinite(bl_w99) or bl_w99 <= 0:
        return False
    return (
        float(d["w_p99_mean"]) < _GO_W_P99_MAX_RATIO * bl_w99
        and float(d["q_clip_mean"]) < _GO_QCLIP_MAX
    )


def _section_9_go_nogo(
    add: list[str],
    results: dict[str, dict[int, Optional[dict]]],
    data_cache: dict[str, dict[int, Optional[dict]]],
    ns: list[int],
    M: int,
) -> dict:
    add.append("## 9. Decision GO / NO-GO")
    add.append("")
    add.append("### Criteres")
    add.append("")
    add.append("**GO DGP2-tuned** (configs A ou AC) : toutes ces conditions a n=1000 ET n=2000 :")
    add.append(f"- |bias_REG| reduit >= {int(_GO_BIAS_REG_REDUCTION_PCT*100)}% vs baseline")
    add.append(f"- |bias_DR|/SE <= {_GO_BSE_MAX}")
    add.append(f"- coverage_ref >= {_GO_COVERAGE_TUNED}")
    add.append(f"- ESS_min_mean >= {_GO_ESS_MIN}")
    add.append(f"- w_p99 < {_GO_W_P99_MAX_RATIO}x baseline_w_p99 ET q_clip < {_GO_QCLIP_MAX}")
    add.append("")
    add.append("**GO policy-smoothed** (configs C ou AC) :")
    add.append(f"- coverage_ref >= {_GO_COVERAGE_SMOOTHED} a n=1000 ou n=2000")
    add.append("- (estimand modifie, Jensen gap accepte, voir Section 5)")
    add.append("")
    add.append("**NO-GO** : aucune des deux branches passe.")
    add.append("")

    decision = {
        "go_tuned": False,
        "go_smoothed": False,
        "deferred": False,
        "best_tuned_cfg": None,
        "best_smoothed_cfg": None,
        "reasons": [],
    }

    # Defer if n=2000 not run
    if 2000 not in ns:
        decision["deferred"] = True
        add.append("### Verdict")
        add.append("")
        add.append("**DECISION DEFEREE** — n=2000 non couru (mode quick).")
        add.append("Lancer `--mode main` pour evaluer les criteres complets.")
        add.append("")
        return decision

    bl_1k = results.get("baseline", {}).get(1000)
    bl_2k = results.get("baseline", {}).get(2000)

    if bl_1k is None or bl_2k is None:
        add.append("### Verdict")
        add.append("")
        add.append("**DECISION IMPOSSIBLE** — baseline n=1000 ou n=2000 manquant.")
        add.append("Lancer `python -m simulations.archive.utilities.dgp2_tuning_confirmation --mode main --config baseline`.")
        add.append("")
        return decision

    bl_breg_1k = abs(float(bl_1k["bias_reg"][REF_IDX]))
    bl_breg_2k = abs(float(bl_2k["bias_reg"][REF_IDX]))

    add.append("### Evaluation par config")
    add.append("")
    add.append("| config | check | n=1000 | n=2000 | pass |")
    add.append("|--------|-------|--------|--------|------|")

    # Tuned candidates : A and AC
    for cfg_name in ("A", "AC"):
        d_1k = results.get(cfg_name, {}).get(1000)
        d_2k = results.get(cfg_name, {}).get(2000)
        if d_1k is None or d_2k is None:
            decision["reasons"].append(f"Config {cfg_name}: n=1000 ou n=2000 manquant.")
            continue
        data_1k = data_cache.get(cfg_name, {}).get(1000)
        data_2k = data_cache.get(cfg_name, {}).get(2000)

        # Stability check
        if _M_ok_ratio(data_1k, M) < _GO_M_OK_RATIO or _M_ok_ratio(data_2k, M) < _GO_M_OK_RATIO:
            decision["reasons"].append(f"Config {cfg_name}: M_ok/M < {_GO_M_OK_RATIO} (instable).")
            continue

        breg_1k = abs(float(d_1k["bias_reg"][REF_IDX]))
        breg_2k = abs(float(d_2k["bias_reg"][REF_IDX]))
        bse_1k = float(d_1k["bse_grid"][REF_IDX])
        bse_2k = float(d_2k["bse_grid"][REF_IDX])
        cov_1k = float(d_1k["coverage_ref"])
        cov_2k = float(d_2k["coverage_ref"])
        ess_1k = float(d_1k["ess_min_mean"])
        ess_2k = float(d_2k["ess_min_mean"])

        c_breg_1k = breg_1k <= (1 - _GO_BIAS_REG_REDUCTION_PCT) * bl_breg_1k
        c_breg_2k = breg_2k <= (1 - _GO_BIAS_REG_REDUCTION_PCT) * bl_breg_2k
        c_bse_1k = bse_1k <= _GO_BSE_MAX
        c_bse_2k = bse_2k <= _GO_BSE_MAX
        c_cov_1k = cov_1k >= _GO_COVERAGE_TUNED
        c_cov_2k = cov_2k >= _GO_COVERAGE_TUNED
        c_ess_1k = ess_1k >= _GO_ESS_MIN
        c_ess_2k = ess_2k >= _GO_ESS_MIN
        c_w_1k = _no_weight_blowup(d_1k, bl_1k)
        c_w_2k = _no_weight_blowup(d_2k, bl_2k)

        rows = [
            ("|b_REG| -30%", c_breg_1k, c_breg_2k, f"{breg_1k:.4f} vs {bl_breg_1k:.4f}",
             f"{breg_2k:.4f} vs {bl_breg_2k:.4f}"),
            (f"|b_DR|/SE<={_GO_BSE_MAX}", c_bse_1k, c_bse_2k, f"{bse_1k:.2f}", f"{bse_2k:.2f}"),
            (f"cov>={_GO_COVERAGE_TUNED}", c_cov_1k, c_cov_2k, f"{cov_1k:.3f}", f"{cov_2k:.3f}"),
            (f"ESS>={_GO_ESS_MIN}", c_ess_1k, c_ess_2k, f"{ess_1k:.0f}", f"{ess_2k:.0f}"),
            ("no w blowup", c_w_1k, c_w_2k, "ok" if c_w_1k else "fail",
             "ok" if c_w_2k else "fail"),
        ]
        for label, p1, p2, v1, v2 in rows:
            mark = "OK" if (p1 and p2) else "FAIL"
            add.append(f"| {cfg_name} | {label} | {v1} | {v2} | {mark} |")

        all_pass = all([c_breg_1k, c_breg_2k, c_bse_1k, c_bse_2k,
                        c_cov_1k, c_cov_2k, c_ess_1k, c_ess_2k, c_w_1k, c_w_2k])
        if all_pass:
            decision["go_tuned"] = True
            decision["best_tuned_cfg"] = cfg_name
            decision["reasons"].append(f"Config {cfg_name}: 10/10 checks pass.")
            break  # First passing config wins
        else:
            failed = [label for label, p1, p2, _, _ in rows if not (p1 and p2)]
            decision["reasons"].append(f"Config {cfg_name} ne passe pas: {failed}")

    # Smoothed candidates : C and AC (highest coverage_ref)
    best_cov = -1.0
    for cfg_name in ("C", "AC"):
        d_1k = results.get(cfg_name, {}).get(1000)
        d_2k = results.get(cfg_name, {}).get(2000)
        for d in (d_1k, d_2k):
            if d is None:
                continue
            cov = float(d["coverage_ref"])
            if cov >= _GO_COVERAGE_SMOOTHED and cov > best_cov:
                best_cov = cov
                decision["go_smoothed"] = True
                decision["best_smoothed_cfg"] = cfg_name

    add.append("")
    add.append("### Verdict")
    add.append("")
    if decision["go_tuned"]:
        add.append(f"**GO DGP2-tuned** avec config **{decision['best_tuned_cfg']}**.")
        if decision["go_smoothed"]:
            add.append(f"_(GO smoothed cumule via config {decision['best_smoothed_cfg']}, coverage>={_GO_COVERAGE_SMOOTHED}.)_")
    elif decision["go_smoothed"]:
        add.append(f"**GO policy-smoothed** avec config **{decision['best_smoothed_cfg']}** "
                  f"(coverage>={_GO_COVERAGE_SMOOTHED}, estimand modifie).")
    else:
        add.append("**NO-GO** — aucune configuration ne satisfait les criteres.")
    add.append("")
    add.append("### Details")
    for r in decision["reasons"]:
        add.append(f"- {r}")
    add.append("")
    return decision


# ── Section 10 ───────────────────────────────────────────────────────────────

def _section_10_implication(add: list[str], decision: dict) -> None:
    add.append("## 10. Implication pour S6 (Simulations)")
    add.append("")
    if decision.get("deferred", False):
        add.append("Decision deferee : aucune implication arretee tant que `--mode main` n'a")
        add.append("pas tourne (n=2000 requis pour evaluer les criteres complets).")
        add.append("")
        add.append("Sanity quick mode : verifier que le pipeline tourne sans erreur, que le")
        add.append("rapport contient toutes les sections, que bias_REG (config A/AC) est plus")
        add.append("petit que bias_REG (baseline) meme a M=10. Si oui, lancer main mode.")
        add.append("")
        return
    if decision["go_tuned"]:
        cfg = decision["best_tuned_cfg"]
        s = CONFIGS[cfg]
        add.append(f"DGP2 promu en experience PRINCIPALE de S6 avec config **{cfg}** :")
        add.append(f"- lambda_h = {s['lambda_h'] if s['lambda_h'] is not None else 'default'}")
        add.append(f"- ell_scale = {s['ell_scale']}")
        add.append(f"- lambda_Q = {s['lambda_Q']:.0e}")
        add.append(f"- clip = {s['clip'] if s['clip'] is not None else 'default'}")
        add.append(f"- h_policy_scale = {s['h_policy_scale']}")
        add.append("")
        add.append("Action :")
        add.append("1. Lancer grande grille DGP2 (3 SNR x 3 n x M=500/1000) avec ces hyperparametres.")
        add.append("2. Noter dans methods.md les hyperparametres canoniques pour DGP2.")
        add.append("3. Citer baseline en annexe (appendix bias diagnostics).")
        add.append("4. Mettre a jour CLAUDE.md tableau Exp1 : DGP2 GO tuned.")
    elif decision["go_smoothed"]:
        cfg = decision["best_smoothed_cfg"]
        add.append(f"DGP2 inclus en S6 avec config **{cfg}** (h_policy=1.5xSilverman explicite) :")
        add.append("- Documenter le trade-off bias/Jensen-gap dans la section S6 DGP2.")
        add.append("- L'estimand cible J(pi_{a, 1.5h}) est explicitement defini comme cible lissee.")
        add.append("- Citer Section 5 du present rapport pour quantifier le shift d'estimand.")
        add.append("- Tuned-config A reste a documenter en appendix si bias_REG reduit.")
        add.append("- Mettre a jour CLAUDE.md tableau Exp1 : DGP2 GO smoothed.")
    else:
        add.append("DGP2 RELEGUE en S7 stress-test :")
        add.append("- Cobb-Douglas (DGP1) reste le cas principal de S6.")
        add.append("- DGP2 fournit une 'box failure mode' : 'Bridge-bias under structural curvature'.")
        add.append("- Documenter rho ~ 0.246 < 0.5 comme limite RKHS theorique (KPV rate).")
        add.append("- Pas de grande grille DGP2 prevue.")
        add.append("- Mettre a jour CLAUDE.md tableau Exp1 : DGP2 NO-GO (stress-test only).")
        add.append("- Phase 11 (Bennett-lite) prend la priorite sur extensions DGP2.")
    add.append("")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DGP2 tuning confirmation pilot (M=200) — Phase 9B.2bis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # quick smoke (~12 min)\n"
            "  python -m simulations.archive.utilities.dgp2_tuning_confirmation --mode quick\n"
            "\n"
            "  # main (5 terminals in parallel, one per config; ~3.5h):\n"
            "  python -m simulations.archive.utilities.dgp2_tuning_confirmation --mode main --config baseline\n"
            "  python -m simulations.archive.utilities.dgp2_tuning_confirmation --mode main --config A\n"
            "  ... (B, C, AC) ...\n"
            "  # then build report:\n"
            "  python -m simulations.archive.utilities.dgp2_tuning_confirmation --mode main\n"
        ),
    )
    p.add_argument("--mode", choices=["quick", "main", "single"], default="main")
    p.add_argument("--config", choices=_CONFIG_ORDER, default=None,
                   help="Run only this config (other configs skipped).")
    p.add_argument("--n", type=int, default=None,
                   help="Required when --mode=single.")
    p.add_argument("--M", type=int, default=None,
                   help="Required when --mode=single.")
    p.add_argument("--force", action="store_true",
                   help="Re-run blocks even if pkl exists (deletes .pkl + .partial.pkl).")
    args = p.parse_args()

    if args.mode == "single":
        if args.config is None or args.n is None or args.M is None:
            p.error("--mode=single requires --config, --n, and --M.")

    return args


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
