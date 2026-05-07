"""
Phase 16 EXP-C -- Spectral diagnostics on (n x SNR) grid.

Design
------
DGP        : MichaelisMentenDGP (DGP2)
Methods    : [linearDR, KPVREG, DRKPV, best_bennett]
Cells      : n in {1000, 2000} x SNR in {0.85, 0.90, 0.95}  -> 6 cells
             SAME GRID as EXP-A for direct cross-correlation analysis.
M          : 50 paired seeds (light, just to estimate kappa means)
seed_base  : 42_000_000  (disjoint from EXP-A 40M and EXP-B 41M)
a_grid     : [1.8, 2.2, 2.6, 3.0]
REF_IDX    : 2

NEW
---
For DRKPV: enable compute_cond=True on KPVPolicyBridgeQ (q-side cond_M).
Bennett methods already compute spectral by default.

Output
------
  raw : simulations/results/raw/phase16_EXPC/n{n}_snr{snr}/{method}_M50.pkl
  report : simulations/results/summaries/phase16_EXPC_spectral.md
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[3]  # +1: moved to experiments/simulation/
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "s7" / "spectral_diagnostics"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "spectral_diagnostics_FINAL.md"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
K = len(A_GRID)
REF_IDX = 2
N_FOLDS = 5
RANDOM_STATE = 42
SEED_BASE = 42_000_000

METHODS_EXPC = ["linearDR", "KPVREG", "DRKPV", "best_bennett"]
N_LIST = [1000, 2000]
SNR_LIST = [0.85, 0.90, 0.95]

from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
    _compute_J_policy_true,
)
from pci.runner.diagnostics import _augment_decomp_with_ser
from simulations.methods.method_registry import METHOD_REGISTRY


# ── Custom factory: DRKPV with compute_cond=True on q-side ───────────────────

def make_DRKPV_with_cond(dgp, a_grid, h_KDE, ell_W0, ell_A0, ell_Z0, **kwargs):
    """Same as METHOD_REGISTRY['DRKPV'] but with compute_cond=True for spectral."""
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
    q_model = KPVPolicyBridgeQ(
        a_grid=a_grid, h_KDE=h_KDE, lambda_Q=1e-3, clip=None,
        compute_cond=True,  # ENABLED for spectral diagnostics
    )
    return DRKernel(
        a_grid=a_grid, q_model=q_model, bandwidth=h_KDE,
        n_folds=5, random_state=42,
        ref_dose_index=kwargs.get("ref_dose_index", 2), cross_fit_q=True,
        bridge_kwargs=dict(
            lambda_1=3e-5, lambda_2=3e-5,
            ell_W=ell_W0 * 3.5, ell_A=ell_A0 * 3.5, ell_Z=ell_Z0 * 3.5,
        ),
    )


# Override registry locally for EXP-C only
METHOD_REGISTRY_EXPC = dict(METHOD_REGISTRY)
METHOD_REGISTRY_EXPC["DRKPV"] = make_DRKPV_with_cond


def _run_method(method_name: str, *, dgp, a_grid, h_KDE, ell_W0, ell_A0, ell_Z0,
                 J_pt, n: int, M: int, seed_base: int, raw_dir: Path,
                 ref_idx: int, force=False):
    factory_factory = METHOD_REGISTRY_EXPC[method_name]

    def factory():
        kwargs = dict(dgp=dgp, a_grid=a_grid, h_KDE=h_KDE,
                       ell_W0=ell_W0, ell_A0=ell_A0, ell_Z0=ell_Z0,
                       ref_dose_index=ref_idx)
        if method_name == "oracle":
            kwargs["is_linear_in_A"] = False
        return factory_factory(**kwargs)

    label = f"{method_name}_M{M}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for f in raw_dir.glob(f"{label}.pkl"):
            f.unlink()

    return _run_block(
        dgp, factory, a_grid, J_pt,
        n=n, M=M, seed_base=seed_base,
        label=label, raw_dir=raw_dir,
    )


def _run_cell(*, n: int, snr: float, M: int, force=False) -> Dict[str, dict]:
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=snr, snr_Z=snr)
    ref = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref.A)
    ell_W0 = _median_bandwidth(ref.W)
    ell_A0 = _median_bandwidth(ref.A)
    ell_Z0 = _median_bandwidth(ref.Z)
    J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)
    print(f"\n=== Cell n={n}  SNR={snr}  J_pt={J_pt}  h_KDE={h_KDE:.3f} ===")

    cell_tag = f"n{n}_snr{int(snr * 100):02d}"
    raw_dir = _RAW_DIR / cell_tag
    out: Dict[str, dict] = {}
    seed_base_cell = SEED_BASE + N_LIST.index(n) * 1000 + SNR_LIST.index(snr) * 100
    for method_name in METHODS_EXPC:
        print(f"  [{method_name}] n={n} SNR={snr}")
        t0 = time.time()
        try:
            data = _run_method(
                method_name, dgp=dgp, a_grid=A_GRID, h_KDE=h_KDE,
                ell_W0=ell_W0, ell_A0=ell_A0, ell_Z0=ell_Z0,
                J_pt=J_pt, n=n, M=M, seed_base=seed_base_cell,
                raw_dir=raw_dir, ref_idx=REF_IDX, force=force,
            )
            decomp = _decompose(data)
            if decomp is not None:
                decomp = _augment_decomp_with_ser(decomp, data)
                decomp["_data"] = data
                # Aggregate spectral fields across reps
                records = [r for r in data["records"] if r.get("error") is None]
                kappa_h_arr = np.array([r.get("kappa_h", np.nan) for r in records])
                eff_rank_h_arr = np.array([r.get("eff_rank_h", np.nan) for r in records])
                resid_h_arr = np.array([r.get("residual_norm_h", np.nan) for r in records])
                kappa_r_arr = np.array([r.get("kappa_r", np.nan) for r in records])
                eff_rank_r_arr = np.array([r.get("eff_rank_r", np.nan) for r in records])
                resid_r_arr = np.array([r.get("residual_norm_r", np.nan) for r in records])
                decomp["spectral"] = {
                    "kappa_h_mean":      float(np.nanmean(kappa_h_arr)),
                    "kappa_h_p50":       float(np.nanmedian(kappa_h_arr)),
                    "eff_rank_h_mean":   float(np.nanmean(eff_rank_h_arr)),
                    "residual_norm_h_mean": float(np.nanmean(resid_h_arr)),
                    "kappa_r_mean":      float(np.nanmean(kappa_r_arr)),
                    "kappa_r_p50":       float(np.nanmedian(kappa_r_arr)),
                    "eff_rank_r_mean":   float(np.nanmean(eff_rank_r_arr)),
                    "residual_norm_r_mean": float(np.nanmean(resid_r_arr)),
                }
            out[method_name] = decomp
            elapsed = time.time() - t0
            if decomp is not None:
                bse = float(decomp["bse_grid"][REF_IDX])
                cov = float(decomp["coverage_ref"])
                spec = decomp.get("spectral", {})
                kh = spec.get("kappa_h_mean", float("nan"))
                kr = spec.get("kappa_r_mean", float("nan"))
                print(f"    bse={bse:.4f}  cov={cov:.3f}  kappa_h={kh:.3e}  kappa_r={kr:.3e}  [{elapsed:.0f}s]")
            else:
                print(f"    [no valid reps]  [{elapsed:.0f}s]")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            out[method_name] = None
    return out


def _write_report(per_cell: Dict[tuple, Dict[str, dict]]):
    import datetime
    lines = []
    lines.append("# Phase 16 EXP-C -- Spectral diagnostics (DGP2, 4 methods, n x SNR grid)")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append("")
    lines.append("DGP2 = MichaelisMentenDGP, symmetric SNR (snr_W = snr_Z)")
    lines.append(f"M=50, seed_base={SEED_BASE}, compute_cond=True for KPV q-side")
    lines.append(f"Grid: {N_LIST} x {SNR_LIST} = 6 cells (matches EXP-A grid)")
    lines.append("")
    lines.append("Methods: " + ", ".join(METHODS_EXPC))
    lines.append("")
    lines.append("Goal: collect spectral fields kappa_h, eff_rank_h, residual_norm_h,")
    lines.append("kappa_r, eff_rank_r, residual_norm_r per (method x cell) for the")
    lines.append("Spearman correlation analysis (spectral health vs bse_observed).")
    lines.append("")

    def _fmt(v):
        if v is None or not np.isfinite(v):
            return "---"
        if abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
            return f"{v:.2e}"
        return f"{v:.4f}"

    cols = ["bse_ref", "cov_ref", "kappa_h", "eff_rank_h", "resid_h",
            "kappa_r", "eff_rank_r", "resid_r"]

    for (n, snr), results in sorted(per_cell.items()):
        lines.append(f"## n={n}, SNR={snr}")
        lines.append("")
        lines.append("| method | " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
        for method_name in METHODS_EXPC:
            d = results.get(method_name)
            if d is None:
                lines.append(f"| {method_name} | (failed) |" +
                              "|".join(["" for _ in cols[1:]]) + " |")
                continue
            spec = d.get("spectral", {})
            row = [
                _fmt(d["bse_grid"][REF_IDX]),
                _fmt(d["coverage_ref"]),
                _fmt(spec.get("kappa_h_mean", float("nan"))),
                _fmt(spec.get("eff_rank_h_mean", float("nan"))),
                _fmt(spec.get("residual_norm_h_mean", float("nan"))),
                _fmt(spec.get("kappa_r_mean", float("nan"))),
                _fmt(spec.get("eff_rank_r_mean", float("nan"))),
                _fmt(spec.get("residual_norm_r_mean", float("nan"))),
            ]
            lines.append(f"| {method_name} | " + " | ".join(row) + " |")
        lines.append("")

    # Spectral correlation analysis (Spearman)
    lines.append("## Cross-cell Spearman correlations (spectral health vs bse_ref)")
    lines.append("")
    lines.append("For each method, correlate the spectral diagnostic across (n, SNR) cells")
    lines.append("with the observed bse_ref. Strong correlation = actionable diagnostic.")
    lines.append("")

    try:
        from scipy.stats import spearmanr
        diag_keys = [
            ("kappa_h_mean", "kappa_h"),
            ("eff_rank_h_mean", "eff_rank_h"),
            ("residual_norm_h_mean", "resid_h"),
            ("kappa_r_mean", "kappa_r"),
            ("eff_rank_r_mean", "eff_rank_r"),
            ("residual_norm_r_mean", "resid_r"),
        ]
        for method_name in METHODS_EXPC:
            lines.append(f"### {method_name}")
            lines.append("")
            lines.append("| Diagnostic | Spearman rho | p-value | n_cells |")
            lines.append("|---|---|---|---|")
            for spec_key, label in diag_keys:
                bses = []
                diags = []
                for (n, snr), results in per_cell.items():
                    d = results.get(method_name)
                    if d is None:
                        continue
                    spec = d.get("spectral", {})
                    val = spec.get(spec_key, float("nan"))
                    bse = d["bse_grid"][REF_IDX]
                    if np.isfinite(val) and np.isfinite(bse):
                        diags.append(val)
                        bses.append(bse)
                if len(diags) >= 3:
                    rho, p = spearmanr(diags, bses)
                    lines.append(f"| {label} | {rho:.3f} | {p:.3f} | {len(diags)} |")
                else:
                    lines.append(f"| {label} | --- | --- | {len(diags)} |")
            lines.append("")
    except ImportError:
        lines.append("(scipy.stats.spearmanr not available; skipping correlation analysis)")
        lines.append("")

    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Report: {_REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.mode == "smoke":
        n_list_local = [1000]
        snr_list_local = [0.95]
        M = 3
    else:
        n_list_local = N_LIST
        snr_list_local = SNR_LIST
        M = args.M

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    per_cell: Dict[tuple, Dict[str, dict]] = {}
    for n in n_list_local:
        for snr in snr_list_local:
            per_cell[(n, snr)] = _run_cell(n=n, snr=snr, M=M, force=args.force)

    _write_report(per_cell)

    elapsed = time.time() - t_total
    print(f"\n=== Phase 16 EXP-C total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")


if __name__ == "__main__":
    main()
