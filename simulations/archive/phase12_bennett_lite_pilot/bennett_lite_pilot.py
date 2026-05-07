"""
Phase 12A -- Bennett-lite Pilot Screening.

Compares BennettFunctionalDR (direct Riesz representer estimation) against
the KPV-super DRKernel benchmark on DGP2 (Michaelis-Menten, SNR=0.95).

Motivation: Phases 11-11C+ showed KPV-super + Q_KPV is the robust finite-sample
benchmark, with Kallus diagnostic-only. Phase 12 tests whether bypassing the
q-bridge factorisation via Bennett's Riesz representer improves over KPV.

Configurations (8 total)
------------------------
  REG_Hsuper    : BennettFunctionalDR(lambda_r=1e10) = plug-in only (no correction)
  DRK_Hsuper    : DRKernel(KPV-super, Q_KPV)  ** BENCHMARK **
  DRK_Htuned    : DRKernel(KPV-tuned, Q_KPV)  (lambda_h=1e-4, ell_scale=2.0)
  BEN2_L1e4_Hs  : BennettFunctionalDR(degree=2, lambda_r=1e-4, KPV-super)
  BEN2_L1e3_Hs  : BennettFunctionalDR(degree=2, lambda_r=1e-3, KPV-super)
  BEN2_L1e2_Hs  : BennettFunctionalDR(degree=2, lambda_r=1e-2, KPV-super)
  BEN3_L1e3_Hs  : BennettFunctionalDR(degree=3, lambda_r=1e-3, KPV-super)
  BEN2_L1e3_Ht  : BennettFunctionalDR(degree=2, lambda_r=1e-3, KPV-tuned)

Modes
-----
  quick    : n=300, M=3, 3 configs (REG/DRK/BEN2_L1e3) -- plumbing test, ~3-5 min
  screen   : n=1000, M=50, all 8 configs               -- verdict preliminary, ~40 min
  validate : n=2000, M=100, all 8 configs              -- (if screen shows Case A/B1)
  final    : n=2000, M=200, all 8 configs              -- anti small-M validation

Anti small-M rule: NEVER promote a winner at M<=100 without M=200 validation.

seed_base = 23_000_000 (disjoint from all prior phases: 16M/17M/19M/20M/21M/22M)

Usage
-----
  python -m simulations.experiments.bennett_lite_pilot --mode quick
  python -m simulations.experiments.bennett_lite_pilot --mode screen
  python -m simulations.experiments.bennett_lite_pilot --mode validate
  python -m simulations.experiments.bennett_lite_pilot --mode final --force

Output
------
  simulations/results/raw/bennett_lite_pilot/<label>.pkl
  simulations/results/summaries/bennett_lite_pilot.md
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Thread control (before any BLAS import)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR    = Path(__file__).resolve().parents[2]
_RAW_DIR     = _BASE_DIR / "simulations" / "results" / "raw" / "bennett_lite_pilot"
_SUMM_DIR    = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "bennett_lite_pilot.md"

# ── Constants (same as DGP2 benchmark) ────────────────────────────────────────
A_GRID     = np.array([1.8, 2.2, 2.6, 3.0])
K          = len(A_GRID)
REF_IDX    = 2           # a = 2.6
SNR        = 0.95
N_FOLDS    = 5
RANDOM_STATE = 42
SEED_BASE  = 23_000_000  # disjoint from all prior phases

# KPV-super anchor (benchmark)
SUPER_HP   = dict(lambda_h=3e-5, ell_scale=3.5)
# KPV-tuned alternative
TUNED_HP   = dict(lambda_h=1e-4, ell_scale=2.0)
# Bennett lambda_r grid
LAMBDA_R_GRID = [1e-4, 1e-3, 1e-2]


# ── Import utilities from dgp2_bias_diagnostics ────────────────────────────────
# Reuse _run_block, _decompose, _silverman_h, _median_bandwidth, _build_record.
from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
    _compute_J_policy_true, _build_record,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Factory functions
# ══════════════════════════════════════════════════════════════════════════════

def _make_reg_hsuper(h_KDE: float, ell_W0: float, ell_A0: float, ell_Z0: float):
    """REG-only: BennettFunctionalDR with lambda_r=1e10 (no correction)."""
    from simulations.methods.bennett_functional_dr import BennettFunctionalDR
    def factory():
        return BennettFunctionalDR(
            lambda_h=SUPER_HP["lambda_h"],
            ell_scale=SUPER_HP["ell_scale"],
            lambda_r=1e10, degree=2,
            n_folds=N_FOLDS, a_grid=A_GRID.tolist(),
            bandwidth=h_KDE, seed=RANDOM_STATE,
            ref_dose_index=REF_IDX,
        )
    return factory


def _make_drk_hsuper(h_KDE: float, ell_W0: float, ell_A0: float, ell_Z0: float):
    """DRKernel with KPV-super h + Q_KPV  (** BENCHMARK **)."""
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
    def factory():
        q_model = KPVPolicyBridgeQ(
            a_grid=A_GRID, h_KDE=h_KDE, lambda_Q=1e-3, clip=None,
        )
        return DRKernel(
            a_grid=A_GRID, q_model=q_model, bandwidth=h_KDE,
            n_folds=N_FOLDS, random_state=RANDOM_STATE,
            ref_dose_index=REF_IDX, cross_fit_q=True,
            bridge_kwargs=dict(
                lambda_1=SUPER_HP["lambda_h"], lambda_2=SUPER_HP["lambda_h"],
                ell_W=ell_W0 * SUPER_HP["ell_scale"],
                ell_A=ell_A0 * SUPER_HP["ell_scale"],
                ell_Z=ell_Z0 * SUPER_HP["ell_scale"],
            ),
        )
    return factory


def _make_drk_htuned(h_KDE: float, ell_W0: float, ell_A0: float, ell_Z0: float):
    """DRKernel with KPV-tuned h + Q_KPV."""
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
    def factory():
        q_model = KPVPolicyBridgeQ(
            a_grid=A_GRID, h_KDE=h_KDE, lambda_Q=1e-3, clip=None,
        )
        return DRKernel(
            a_grid=A_GRID, q_model=q_model, bandwidth=h_KDE,
            n_folds=N_FOLDS, random_state=RANDOM_STATE,
            ref_dose_index=REF_IDX, cross_fit_q=True,
            bridge_kwargs=dict(
                lambda_1=TUNED_HP["lambda_h"], lambda_2=TUNED_HP["lambda_h"],
                ell_W=ell_W0 * TUNED_HP["ell_scale"],
                ell_A=ell_A0 * TUNED_HP["ell_scale"],
                ell_Z=ell_Z0 * TUNED_HP["ell_scale"],
            ),
        )
    return factory


def _make_bennett(
    lambda_r: float, degree: int, h_KDE: float,
    ell_scale: float = 3.5, lambda_h: float = 3e-5
):
    """BennettFunctionalDR with given hyperparameters."""
    from simulations.methods.bennett_functional_dr import BennettFunctionalDR
    def factory():
        return BennettFunctionalDR(
            lambda_h=lambda_h, ell_scale=ell_scale,
            lambda_r=lambda_r, degree=degree,
            n_folds=N_FOLDS, a_grid=A_GRID.tolist(),
            bandwidth=h_KDE, seed=RANDOM_STATE,
            ref_dose_index=REF_IDX,
        )
    return factory


def _build_all_configs(h_KDE, ell_W0, ell_A0, ell_Z0, mode="screen"):
    """
    Returns dict of {config_id: factory_fn}.
    For quick mode, returns only 3 core configs.
    """
    configs = {}

    if mode == "quick":
        configs["REG_Hsuper"]   = _make_reg_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0)
        configs["DRK_Hsuper"]   = _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0)
        configs["BEN2_L1e3_Hs"] = _make_bennett(1e-3, 2, h_KDE, ell_scale=3.5, lambda_h=3e-5)
        return configs

    # Full config set for screen / validate / final
    configs["REG_Hsuper"]   = _make_reg_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0)
    configs["DRK_Hsuper"]   = _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0)
    configs["DRK_Htuned"]   = _make_drk_htuned(h_KDE, ell_W0, ell_A0, ell_Z0)
    configs["BEN2_L1e4_Hs"] = _make_bennett(1e-4, 2, h_KDE, ell_scale=3.5, lambda_h=3e-5)
    configs["BEN2_L1e3_Hs"] = _make_bennett(1e-3, 2, h_KDE, ell_scale=3.5, lambda_h=3e-5)
    configs["BEN2_L1e2_Hs"] = _make_bennett(1e-2, 2, h_KDE, ell_scale=3.5, lambda_h=3e-5)
    configs["BEN3_L1e3_Hs"] = _make_bennett(1e-3, 3, h_KDE, ell_scale=3.5, lambda_h=3e-5)
    configs["BEN2_L1e3_Ht"] = _make_bennett(1e-3, 2, h_KDE, ell_scale=TUNED_HP["ell_scale"],
                                              lambda_h=TUNED_HP["lambda_h"])
    return configs


# ══════════════════════════════════════════════════════════════════════════════
#  Summary helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v, fmt=".4f"):
    try:
        f = float(v)
        if not np.isfinite(f):
            return "---"
        return f"{f:.2e}" if abs(f) >= 1e5 else f"{f:{fmt}}"
    except Exception:
        return str(v)


def _row(cid: str, decomp: Optional[dict], J_pt: np.ndarray, n: int) -> str:
    """Build one markdown table row."""
    if decomp is None:
        return f"| {cid:<22} | --- | --- | --- | --- | --- | --- | --- | --- |"
    M_ok = int(decomp["M_ok"])
    bias_dr  = float(decomp["bias_dr"][REF_IDX])
    bse      = float(decomp["bse_grid"][REF_IDX])
    cov      = float(decomp["coverage_ref"])
    ess_min  = float(decomp.get("ess_min_mean", np.nan))
    w_p99    = float(decomp.get("w_p99_mean", np.nan))
    rr       = float(decomp.get("riesz_res_mean", np.nan))
    return (
        f"| {cid:<22} | {M_ok:>4} | {_fmt(bias_dr)} | {_fmt(bse)} | "
        f"{_fmt(cov)} | {_fmt(ess_min, '.1f')} | "
        f"{_fmt(w_p99, '.2f')} | {_fmt(rr, '.2e')} |"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Run one n/M block for all configs
# ══════════════════════════════════════════════════════════════════════════════

def _run_n_block(dgp, n: int, M: int, configs: dict, J_pt: np.ndarray,
                 raw_dir: Path, force: bool = False) -> dict:
    """Run all configs at a given n and return {cid: decomp}."""
    if force:
        # Remove existing PKLs to force recompute
        for label_patt in raw_dir.glob(f"bennett_*_n{n}_M{M}.pkl"):
            label_patt.unlink()

    results = {}
    for cfg_idx, (cid, factory_fn) in enumerate(configs.items()):
        label = f"bennett_{cid}_n{n}_M{M}"
        # seed offset per config to ensure independence
        seed_b = SEED_BASE + cfg_idx * 1_000_000 + n
        print(f"  [{cfg_idx+1}/{len(configs)}] {cid}  n={n}  M={M}")
        t0 = time.time()
        data = _run_block(
            dgp, factory_fn, A_GRID, J_pt,
            n=n, M=M, seed_base=seed_b,
            label=label, raw_dir=raw_dir,
        )
        elapsed = time.time() - t0
        decomp = _decompose(data)
        results[cid] = decomp
        if decomp is not None:
            bse = float(decomp["bse_grid"][REF_IDX])
            cov = float(decomp["coverage_ref"])
            print(f"    |b|/SE={_fmt(bse)}  cov={_fmt(cov)}  [{elapsed:.0f}s]")
        else:
            print(f"    [decompose returned None — all reps failed]  [{elapsed:.0f}s]")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Report builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_report(
    results_by_n: dict,   # {n: {cid: decomp}}
    mode: str,
    J_pt_by_n: dict,
) -> str:
    """Build a markdown report string from the results."""
    import datetime
    lines = [
        "# Phase 12A Bennett-lite Pilot Screening",
        f"Date: {datetime.date.today().isoformat()}",
        f"Mode: {mode}  |  DGP: MichaelisMentenDGP(SNR=0.95)  |  seed_base={SEED_BASE}",
        f"a_grid: {A_GRID.tolist()}  |  REF_IDX={REF_IDX} (a={A_GRID[REF_IDX]})",
        "",
        "## Benchmark",
        "DRK_Hsuper = DRKernel(KPV-super h, Q_KPV q) -- same as Phase 9B+ benchmark.",
        "Anti small-M rule: NEVER promote winner at M<=100 without M=200 validation.",
        "",
        "## Decision criteria (Cas A)",
        "Bennett wins ssi (at n=2000, M>=100):",
        "  * |b|/SE of Bennett <= 0.85 * DRK_Hsuper  (-15%)",
        "  * coverage >= DRK_cov - 0.03",
        "  * ESS_abs > 50",
        "  * riesz_residual_mean < 1e-3",
        "",
    ]

    for n, results in sorted(results_by_n.items()):
        J_pt = J_pt_by_n[n]
        lines.append(f"## Results  n={n}")
        lines.append("")
        header = (
            "| Config                 | M_ok | bias_DR  | |b|/SE  | "
            "cov    | ESS_abs | w_p99 | riesz_res |"
        )
        sep = "|" + "|".join(["-" * w for w in [24, 6, 10, 8, 7, 9, 7, 11]]) + "|"
        lines.append(header)
        lines.append(sep)
        for cid, decomp in results.items():
            lines.append(_row(cid, decomp, J_pt, n))
        lines.append("")

        # Decision for this n
        if "DRK_Hsuper" in results and results["DRK_Hsuper"] is not None:
            bench = results["DRK_Hsuper"]
            bench_bse = float(bench["bse_grid"][REF_IDX])
            bench_cov = float(bench["coverage_ref"])
            lines.append(f"**Benchmark DRK_Hsuper**: |b|/SE={_fmt(bench_bse)}, cov={_fmt(bench_cov)}")
            lines.append("")

            # Find best Bennett config
            ben_configs = {k: v for k, v in results.items()
                          if k.startswith("BEN") and v is not None}
            if ben_configs:
                best_cid = min(ben_configs, key=lambda c: float(ben_configs[c]["bse_grid"][REF_IDX]))
                best_bse = float(ben_configs[best_cid]["bse_grid"][REF_IDX])
                best_cov = float(ben_configs[best_cid]["coverage_ref"])
                ratio = best_bse / bench_bse if bench_bse > 0 else np.nan
                lines.append(f"**Best Bennett**: {best_cid}  |b|/SE={_fmt(best_bse)} "
                             f"({_fmt(ratio, '.2f')}x benchmark), cov={_fmt(best_cov)}")
                lines.append("")
                if ratio <= 0.85:
                    lines.append("=> Cas A signal detected at this n. Validate at larger M.")
                else:
                    lines.append("=> Cas B/D: no clear gain. Bennett ~= benchmark or worse.")
                lines.append("")

    lines.append("## Configs")
    lines.append("| ID | h-bridge | lambda_r | degree | Role |")
    lines.append("|---|---|---|---|---|")
    lines.append("| REG_Hsuper   | KPV-super (3e-5, 3.5) | 1e10 (off) | 2 | Plug-in only (no correction) |")
    lines.append("| DRK_Hsuper   | KPV-super             | Q_KPV      | - | ** BENCHMARK ** |")
    lines.append("| DRK_Htuned   | KPV-tuned (1e-4, 2.0) | Q_KPV      | - | DRK alternative |")
    lines.append("| BEN2_L1e4_Hs | KPV-super             | 1e-4       | 2 | Bennett strong reg |")
    lines.append("| BEN2_L1e3_Hs | KPV-super             | 1e-3       | 2 | Bennett central |")
    lines.append("| BEN2_L1e2_Hs | KPV-super             | 1e-2       | 2 | Bennett light reg |")
    lines.append("| BEN3_L1e3_Hs | KPV-super             | 1e-3       | 3 | Bennett degree-3 |")
    lines.append("| BEN2_L1e3_Ht | KPV-tuned             | 1e-3       | 2 | Bennett on tuned h |")
    lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 12A Bennett-lite pilot")
    parser.add_argument("--mode", choices=["quick", "screen", "validate", "final"],
                        default="quick", help="Run mode")
    parser.add_argument("--force", action="store_true",
                        help="Force recompute (ignore cached PKLs)")
    args = parser.parse_args()
    mode = args.mode

    # Mode-specific parameters
    mode_params = {
        "quick":    [(300,  3)],
        "screen":   [(1000, 50)],
        "validate": [(2000, 100)],
        "final":    [(2000, 200)],
    }
    n_M_list = mode_params[mode]

    print("=" * 72)
    print(f"Phase 12A Bennett-lite Pilot  --mode {mode}")
    print("=" * 72)
    print(f"Configurations: {'3 (quick)' if mode == 'quick' else '8 (full)'}")
    print(f"seed_base: {SEED_BASE}")
    print()

    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)

    # Reference sample for bandwidth computation
    ref_sample = dgp.generate(n=2000, seed=1)
    h_KDE  = _silverman_h(ref_sample.A)
    ell_W0 = _median_bandwidth(ref_sample.W)
    ell_A0 = _median_bandwidth(ref_sample.A)
    ell_Z0 = _median_bandwidth(ref_sample.Z)

    print(f"h_KDE={h_KDE:.4f}  ell_W0={ell_W0:.4f}  ell_A0={ell_A0:.4f}  ell_Z0={ell_Z0:.4f}")
    print()

    configs = _build_all_configs(h_KDE, ell_W0, ell_A0, ell_Z0, mode=mode)
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)

    results_by_n = {}
    J_pt_by_n = {}
    total_t0 = time.time()

    for n, M in n_M_list:
        J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)
        J_pt_by_n[n] = J_pt
        print(f"\n--- n={n}  M={M}  ({len(configs)} configs) ---")
        print(f"J_policy_true: {[f'{v:.4f}' for v in J_pt]}")
        results_by_n[n] = _run_n_block(
            dgp=dgp, n=n, M=M, configs=configs,
            J_pt=J_pt, raw_dir=_RAW_DIR, force=args.force,
        )

    total_elapsed = time.time() - total_t0
    print(f"\nTotal elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Build and save report
    report = _build_report(results_by_n, mode=mode, J_pt_by_n=J_pt_by_n)
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\nReport saved: {_REPORT_PATH}")
    print()
    # Print summary to stdout
    for line in report.split("\n")[:40]:
        print(line)


if __name__ == "__main__":
    main()
