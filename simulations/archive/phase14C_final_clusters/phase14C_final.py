"""
Phase 14C.2 — Final 6-method comparison on DGP2 at SNR>=0.90.

Methods (in order):
  1. REG_KPV_super       (Phase 9B+ baseline, plug-in only)
  2. DRK_KPV_super       (DR baseline, KPV q-bridge)
  3. Bennett_lite        (Phase 12B winner: RFF500_e35_L1e2_Hs)
  4. Bennett_indep_super (Phase 14B winner: BIN_mh2000)
  5. Kallus_best         (Phase 14C.1 winner: K_gH5_ls001)
  6. SAFE_R3             (Phase 13B post-hoc, computed from REG + DRK records)

4 cells: (n in {1000, 2000}) x (SNR in {0.90, 0.95})
M=100, seed_base = 30_000_000 (DISJOINT from all prior phases).

Outputs:
  simulations/results/raw/phase14C_final/snr<S>_n<n>/<cid>.pkl
  simulations/results/summaries/phase14C_final.md
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[2]
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "phase14C_final"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "phase14C_final.md"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
K = len(A_GRID)
REF_IDX = 2
N_FOLDS = 5
RANDOM_STATE = 42
SEED_BASE = 30_000_000


from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
    _compute_J_policy_true,
)
from simulations.experiments.bennett_rff_overnight import (
    _augment_decomp_with_ser, _make_reg_hsuper, _make_drk_hsuper, _make_bennett_rff,
)
from simulations.experiments.phase14B_bennett_indep import _make_bennett_indep
from simulations.experiments.phase14C_kallus_best import _make_drk_kallus_h


def _safe_dr_post_hoc(reg_records: list, drk_records: list, J_pt: np.ndarray, n: int) -> dict:
    """SAFE_R3 from Phase 13B: eta=0 if weight_p99(a)/median(weight_p99) > 1.5, else 1.

    J_SAFE = J_REG + eta * (J_DRK - J_REG)
    V_SAFE = V_REG + eta^2 * V_corr  (approximation)
    """
    M_actual = len(reg_records)
    J_reg = np.array([r["J_reg"] for r in reg_records])
    J_drk = np.array([r["J_dr"] for r in drk_records])
    V_reg = np.array([r["V_reg_grid"] for r in drk_records])
    V_corr = np.array([r["V_correction_grid"] for r in drk_records])
    weight_p99 = np.array([r["weight_p99_grid"] for r in drk_records])

    median_p99 = np.median(weight_p99, axis=1, keepdims=True)
    eta = (weight_p99 <= 1.5 * median_p99).astype(float)
    J_safe = J_reg + eta * (J_drk - J_reg)
    V_safe = V_reg + (eta ** 2) * V_corr

    bias = J_safe.mean(axis=0) - J_pt
    emp_sd = np.std(J_safe, axis=0, ddof=1)
    pred_se = np.sqrt(np.maximum(V_safe.mean(axis=0), 0.0) / n)
    bse = np.where(pred_se > 1e-12, np.abs(bias) / pred_se, np.nan)
    ser = np.where(emp_sd > 1e-12, pred_se / emp_sd, np.nan)
    cov_per_dose = np.array([
        float(np.mean(np.abs(J_safe[:, k] - J_pt[k]) <=
                      1.96 * np.sqrt(np.maximum(V_safe[:, k], 0.0) / n)))
        for k in range(K)
    ])

    return {
        "bias_dr": bias,
        "bse_grid": bse,
        "coverage_ref": float(cov_per_dose[REF_IDX]),
        "ser_ref": float(ser[REF_IDX]),
        "mean_eta": eta.mean(axis=0).tolist(),
        "_J_safe": J_safe,  # for downstream
    }


def _run_cell(snr: float, n: int, M: int = 100, force: bool = False) -> Dict:
    """Run all 6 methods at a single (snr, n) cell with paired seeds."""
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=snr, snr_Z=snr)
    ref = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref.A)
    ell_W0 = _median_bandwidth(ref.W)
    ell_A0 = _median_bandwidth(ref.A)
    ell_Z0 = _median_bandwidth(ref.Z)
    J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)
    print(f"  cell SNR={snr}, n={n}: h_KDE={h_KDE:.4f}, J_pt={[f'{v:.4f}' for v in J_pt]}")

    seed_base_cell = SEED_BASE + int(snr * 100) * 1_000 + n

    factories = {
        "REG_KPV_super": _make_reg_hsuper(h_KDE),
        "DRK_KPV_super": _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0),
        "Bennett_lite": _make_bennett_rff(
            n_features=500, ell_scale_rff=3.5, lambda_r=1e-2,
            h_KDE=h_KDE, rff_seed_base=seed_base_cell + 100,
        ),
        "BIN_mh2000": _make_bennett_indep(
            m_h=2000, m_c=500, ell_h=2.75, ell_c=2.0,
            lambda_h=1e-5, gamma_critic=1e-4,
            lambda_r=1e-2, n_features_r=500,
            h_KDE=h_KDE,
            h_seed_base=seed_base_cell + 200,
            rff_seed_base_r=seed_base_cell + 300,
        ),
        "Kallus_best": _make_drk_kallus_h(
            h_KDE, ell_W0, ell_A0, ell_Z0,
            gamma_H=1e-5, lambda_stab_h=0.1, gamma_critic_h=1e-3,
            n_features_h=400, n_features_critic=400, ell_scale=3.5,
            feature_seed=seed_base_cell + 400,
        ),
    }

    cell_dir = _RAW_DIR / f"snr{int(snr*100)}_n{n}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    out = {}
    for cid, factory_fn in factories.items():
        label = f"{cid}_M{M}"
        if force:
            for f in cell_dir.glob(f"{label}.pkl"):
                f.unlink()
        print(f"    [{cid}]")
        t0 = time.time()
        try:
            data = _run_block(
                dgp, factory_fn, A_GRID, J_pt,
                n=n, M=M, seed_base=seed_base_cell,
                label=label, raw_dir=cell_dir,
            )
            elapsed = time.time() - t0
            decomp = _decompose(data)
            if decomp is not None:
                decomp = _augment_decomp_with_ser(decomp, data)
            out[cid] = (decomp, data)
            if decomp is not None:
                bse = float(decomp["bse_grid"][REF_IDX])
                cov = float(decomp["coverage_ref"])
                ser = float(decomp.get("ser_ref", float("nan")))
                print(f"      bse_ref={bse:.4f}  cov={cov:.3f}  SER={ser:.3f}  [{elapsed:.0f}s]")
        except Exception as e:
            print(f"    [{cid}] FAILED: {type(e).__name__}: {e}")
            out[cid] = (None, None)

    # SAFE_R3 post-hoc from REG and DRK records
    reg_data = out.get("REG_KPV_super", (None, None))[1]
    drk_data = out.get("DRK_KPV_super", (None, None))[1]
    if reg_data is not None and drk_data is not None:
        reg_recs = [r for r in reg_data["records"] if r.get("error") is None]
        drk_recs = [r for r in drk_data["records"] if r.get("error") is None]
        # Pair by seed
        reg_by_seed = {r["seed"]: r for r in reg_recs}
        drk_by_seed = {r["seed"]: r for r in drk_recs}
        common = sorted(set(reg_by_seed.keys()) & set(drk_by_seed.keys()))
        if common:
            reg_paired = [reg_by_seed[s] for s in common]
            drk_paired = [drk_by_seed[s] for s in common]
            safe = _safe_dr_post_hoc(reg_paired, drk_paired, J_pt, n)
            out["SAFE_R3"] = (safe, None)
            bse = float(safe["bse_grid"][REF_IDX])
            cov = float(safe["coverage_ref"])
            ser = float(safe["ser_ref"])
            print(f"    [SAFE_R3] bse_ref={bse:.4f}  cov={cov:.3f}  SER={ser:.3f}  [post-hoc]")

    return out, J_pt


def _build_report(all_results: Dict[Tuple[float, int], Dict],
                  J_pt_per_cell: Dict[Tuple[float, int], np.ndarray]) -> str:
    import datetime
    lines = []
    lines.append(f"# Phase 14C — Final 6-method comparison")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append(f"DGP: MichaelisMentenDGP, SNR x n grid")
    lines.append(f"M=100, seed_base={SEED_BASE} (DISJOINT from all prior phases)")
    lines.append(f"a_grid={A_GRID.tolist()}, REF_IDX={REF_IDX}")
    lines.append("")

    method_order = ["REG_KPV_super", "DRK_KPV_super", "Bennett_lite",
                    "BIN_mh2000", "Kallus_best", "SAFE_R3"]

    # Per-cell tables
    for (snr, n) in sorted(all_results.keys()):
        cell = all_results[(snr, n)]
        lines.append(f"## SNR={snr}, n={n}")
        lines.append("")
        lines.append("| method | bias_ref | bse_ref | bse_a18 | bse_a22 | bse_a30 | avg_bse | cov | SER |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for cid in method_order:
            res = cell.get(cid)
            if res is None or res[0] is None:
                lines.append(f"| {cid} | --- | --- | --- | --- | --- | --- | --- | --- |")
                continue
            decomp, _ = res
            bse = decomp["bse_grid"]
            bias = decomp.get("bias_dr", np.full(K, np.nan))
            cov = float(decomp.get("coverage_ref", float("nan")))
            ser = float(decomp.get("ser_ref", float("nan")))
            avg_bse = float(np.mean(bse))
            lines.append(
                f"| {cid} | {float(bias[REF_IDX]):+.4f} | {float(bse[REF_IDX]):.4f} | "
                f"{float(bse[0]):.4f} | {float(bse[1]):.4f} | {float(bse[3]):.4f} | "
                f"{avg_bse:.4f} | {cov:.3f} | {ser:.3f} |"
            )
        lines.append("")

    # Aggregate ranking across all 4 cells
    lines.append("## AGGREGATE (mean across 4 cells)")
    lines.append("")
    lines.append("| method | avg_bse | avg_bse_ref | avg_bse_a18 | avg_cov | avg_SER |")
    lines.append("|---|---|---|---|---|---|")
    agg = {}
    for cid in method_order:
        rows = []
        for (snr, n), cell in all_results.items():
            res = cell.get(cid)
            if res is None or res[0] is None:
                continue
            decomp = res[0]
            bse = decomp["bse_grid"]
            cov = float(decomp.get("coverage_ref", float("nan")))
            ser = float(decomp.get("ser_ref", float("nan")))
            rows.append((float(np.mean(bse)), float(bse[REF_IDX]), float(bse[0]), cov, ser))
        if not rows:
            lines.append(f"| {cid} | --- | --- | --- | --- | --- |")
            continue
        arr = np.array(rows)
        agg[cid] = arr.mean(axis=0)
        lines.append(
            f"| {cid} | {agg[cid][0]:.4f} | {agg[cid][1]:.4f} | "
            f"{agg[cid][2]:.4f} | {agg[cid][3]:.3f} | {agg[cid][4]:.3f} |"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cells", type=str, default="all",
                        help="Comma-separated cell list e.g. '90_1000,95_2000', or 'all'")
    args = parser.parse_args()

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    if args.cells == "all":
        cells = [(0.90, 1000), (0.90, 2000), (0.95, 1000), (0.95, 2000)]
    else:
        cells = []
        for c in args.cells.split(","):
            snr_str, n_str = c.split("_")
            cells.append((int(snr_str)/100, int(n_str)))

    all_results = {}
    J_pt_per_cell = {}
    for snr, n in cells:
        print(f"\n=== Cell SNR={snr}, n={n} ===")
        cell_out, J_pt = _run_cell(snr, n, M=100, force=args.force)
        all_results[(snr, n)] = cell_out
        J_pt_per_cell[(snr, n)] = J_pt

    # Build final report
    report = _build_report(all_results, J_pt_per_cell)
    with open(_REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\nFinal report: {_REPORT_PATH}")
    elapsed = time.time() - t_total
    print(f"Phase 14C.2 total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print()
    for line in report.split("\n")[:60]:
        print(line)


if __name__ == "__main__":
    main()
