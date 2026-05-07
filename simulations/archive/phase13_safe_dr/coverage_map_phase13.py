"""
Phase 13A -- Coverage Map for DGP2 (Michaelis-Menten).

Maps coverage / bias / SER / boundary advantage for three estimators across
(SNR, n, dose) on DGP2:

  REG_Hsuper   = plug-in T_pi h_hat  (KPV-super)
  DRK_Hsuper   = DRKernel(KPV-super h, Q_KPV)         (Phase 9B+ benchmark)
  RFF500       = BennettFunctionalDR(RFF500_e35_L1e2) (Phase 12B Case B winner)

Goal: characterise the *regime of existence* of the Phase 12B Case B finding
(boundary advantage of Bennett-RFF over DRKernel at low-density boundary doses).

Grid (default "full"):
  SNR_W = SNR_Z in {0.80, 0.90, 0.95}
  n in {500, 1000, 2000}
  M = 100
  doses = [1.8, 2.2, 2.6, 3.0]
  seed_base = 26_000_000  (disjoint from all prior phases)

Same-seed-within-cell discipline:
  All 3 methods within a (SNR, n) cell use the SAME data (same seed_base for
  the cell), so cross-method comparisons are paired by-rep. Different cells
  use different seed_bases (snr_idx + n_idx offsets).

Stages
------
  smoke       : n=300, M=3, SNR=0.95 only, 3 methods (~3 min)
  reduced     : full SNR x n grid but M=50 (~75 min, half cost)
  full        : full SNR x n grid, M=100 (~3-4 h)
  boundary    : run only n in {1000, 2000}, SNR in {0.90, 0.95} (focus mode)

Usage
-----
  python -m simulations.experiments.coverage_map_phase13 --mode smoke
  python -m simulations.experiments.coverage_map_phase13 --mode reduced
  python -m simulations.experiments.coverage_map_phase13 --mode full
  python -m simulations.experiments.coverage_map_phase13 --mode boundary

Output
------
  simulations/results/raw/coverage_map_phase13/snr<SNR>_n<n>/<method>_M<M>.pkl
  simulations/results/summaries/phase13_coverage_map.md
  simulations/results/summaries/phase13_boundary_map.md
  simulations/results/summaries/phase13_snr_transition.md
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Thread control before BLAS imports
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[2]
_RAW_BASE = _BASE_DIR / "simulations" / "results" / "raw" / "coverage_map_phase13"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
K = len(A_GRID)
REF_IDX = 2
N_FOLDS = 5
RANDOM_STATE = 42
SEED_BASE = 26_000_000

GRID_FULL = {
    "snr": [0.80, 0.90, 0.95],
    "n":   [500, 1000, 2000],
}
GRID_BOUNDARY = {
    "snr": [0.90, 0.95],
    "n":   [1000, 2000],
}
GRID_REDUCED = GRID_FULL  # same grid, M=50 instead of M=100
GRID_SMOKE = {
    "snr": [0.95],
    "n":   [300],
}


# ── Reuse utilities ────────────────────────────────────────────────────────────

from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
    _compute_J_policy_true,
)
from simulations.experiments.bennett_rff_overnight import (
    _augment_decomp_with_ser,
)


# ── Factories (parameterised by per-cell h_KDE / median ells) ─────────────────

def _make_reg(h_KDE, lambda_h=3e-5, ell_scale=3.5):
    """REG_Hsuper: plug-in only (BennettFunctionalDR with lambda_r=1e10)."""
    from simulations.methods.bennett_functional_dr import BennettFunctionalDR
    def factory():
        return BennettFunctionalDR(
            lambda_h=lambda_h, ell_scale=ell_scale,
            lambda_r=1e10, degree=2, feature_map_type="polynomial",
            n_folds=N_FOLDS, a_grid=A_GRID.tolist(),
            bandwidth=h_KDE, seed=RANDOM_STATE, ref_dose_index=REF_IDX,
        )
    return factory


def _make_drk(h_KDE, ell_W0, ell_A0, ell_Z0,
              lambda_h=3e-5, ell_scale=3.5):
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
                lambda_1=lambda_h, lambda_2=lambda_h,
                ell_W=ell_W0 * ell_scale,
                ell_A=ell_A0 * ell_scale,
                ell_Z=ell_Z0 * ell_scale,
            ),
        )
    return factory


def _make_rff500(h_KDE, lambda_h=3e-5, ell_scale=3.5,
                  rff_seed_base=2025500):
    from simulations.methods.bennett_functional_dr import BennettFunctionalDR
    def factory():
        return BennettFunctionalDR(
            lambda_h=lambda_h, ell_scale=ell_scale,
            lambda_r=1e-2,
            feature_map_type="rff",
            n_features=500, ell_scale_rff=3.5,
            rff_seed_base=rff_seed_base,
            n_folds=N_FOLDS, a_grid=A_GRID.tolist(),
            bandwidth=h_KDE, seed=RANDOM_STATE, ref_dose_index=REF_IDX,
        )
    return factory


# ── Cell runner (same seeds within cell) ──────────────────────────────────────

def _run_cell(
    *,
    snr: float,
    n: int,
    M: int,
    seed_base_cell: int,
    raw_dir_cell: Path,
    force: bool = False,
):
    """Run all 3 methods at a single (SNR, n) cell with paired seeds."""
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=snr, snr_Z=snr)

    ref_sample = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref_sample.A)
    ell_W0 = _median_bandwidth(ref_sample.W)
    ell_A0 = _median_bandwidth(ref_sample.A)
    ell_Z0 = _median_bandwidth(ref_sample.Z)
    J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)

    print(f"  cell SNR={snr}, n={n} : h_KDE={h_KDE:.4f}, ell_W={ell_W0:.4f}, "
          f"ell_A={ell_A0:.4f}, ell_Z={ell_Z0:.4f}")
    print(f"  J_policy_true: {[f'{v:.4f}' for v in J_pt]}")

    factories = {
        "REG_Hsuper": _make_reg(h_KDE),
        "DRK_Hsuper": _make_drk(h_KDE, ell_W0, ell_A0, ell_Z0),
        "RFF500":     _make_rff500(h_KDE, rff_seed_base=int(seed_base_cell + 500)),
    }

    raw_dir_cell.mkdir(parents=True, exist_ok=True)
    if force:
        for f in raw_dir_cell.glob(f"*_M{M}.pkl"):
            f.unlink()

    out = {}
    for cid, factory_fn in factories.items():
        label = f"{cid}_M{M}"
        # SAME seed_base_cell for all 3 methods -> SAME data per rep
        print(f"    [{cid}] n={n} M={M}")
        t0 = time.time()
        data = _run_block(
            dgp, factory_fn, A_GRID, J_pt,
            n=n, M=M, seed_base=seed_base_cell,
            label=label, raw_dir=raw_dir_cell,
        )
        elapsed = time.time() - t0
        decomp = _decompose(data)
        if decomp is not None:
            decomp = _augment_decomp_with_ser(decomp, data)
        out[cid] = decomp
        if decomp is not None:
            bse = float(decomp["bse_grid"][REF_IDX])
            cov = float(decomp["coverage_ref"])
            ser = float(decomp.get("ser_ref", float("nan")))
            print(f"      bse_ref={bse:.4f}  cov={cov:.3f}  SER={ser:.3f}  [{elapsed:.0f}s]")
    return out, J_pt


# ── Main run ──────────────────────────────────────────────────────────────────

def _run_full_grid(grid: dict, M: int, force: bool = False) -> dict:
    """Run the full SNR x n grid. Returns dict[(snr, n)] -> {method: decomp}."""
    results: Dict[Tuple[float, int], Dict[str, Optional[dict]]] = {}
    J_pt_per_cell: Dict[Tuple[float, int], np.ndarray] = {}

    for snr_idx, snr in enumerate(grid["snr"]):
        for n_idx, n in enumerate(grid["n"]):
            seed_base_cell = SEED_BASE + snr_idx * 100_000 + n_idx * 1_000 + n
            raw_dir_cell = _RAW_BASE / f"snr{int(snr*100):d}_n{n}"
            print()
            print(f"{'-'*72}")
            print(f"CELL: SNR={snr}, n={n}, M={M}, seed_base_cell={seed_base_cell}")
            print(f"{'-'*72}")
            cell_results, J_pt = _run_cell(
                snr=snr, n=n, M=M,
                seed_base_cell=seed_base_cell,
                raw_dir_cell=raw_dir_cell,
                force=force,
            )
            results[(snr, n)] = cell_results
            J_pt_per_cell[(snr, n)] = J_pt
    return results, J_pt_per_cell


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt(v, fmt=".4f") -> str:
    try:
        f = float(v)
        if not np.isfinite(f):
            return "---"
        return f"{f:.2e}" if (abs(f) >= 1e5 or (0 < abs(f) < 1e-3)) else f"{f:{fmt}}"
    except Exception:
        return str(v)


def _build_main_report(results: dict, J_pt_per_cell: dict, M: int) -> str:
    import datetime
    lines = []
    lines.append(f"# Phase 13A Coverage Map (DGP2 Michaelis-Menten)")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append(f"M = {M} (paired seeds within each cell)")
    lines.append(f"a_grid = {A_GRID.tolist()}")
    lines.append("")
    lines.append("## Master table (REF dose a=2.6)")
    lines.append("")
    lines.append("| SNR | n | method | bias_ref | |b|/SE_ref | cov_ref | SER_ref | bse_a18 | bse_a22 | bse_a26 | bse_a30 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (snr, n), cell in sorted(results.items()):
        for cid in ["REG_Hsuper", "DRK_Hsuper", "RFF500"]:
            dec = cell.get(cid)
            if dec is None:
                lines.append(f"| {snr} | {n} | {cid} | --- |")
                continue
            bse = dec["bse_grid"]
            bias_dr = float(dec["bias_dr"][REF_IDX])
            ser = float(dec.get("ser_ref", float("nan")))
            cov = float(dec["coverage_ref"])
            lines.append(
                f"| {snr} | {n} | {cid} | {bias_dr:+.4f} | {float(bse[REF_IDX]):.3f} | "
                f"{cov:.3f} | {ser:.3f} | "
                f"{float(bse[0]):.3f} | {float(bse[1]):.3f} | "
                f"{float(bse[2]):.3f} | {float(bse[3]):.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _build_boundary_report(results: dict, M: int) -> str:
    import datetime
    lines = []
    lines.append(f"# Phase 13A Boundary-dose map (DGP2)")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append(f"M = {M}")
    lines.append("")
    lines.append("Focus on a=1.8 (low-density lower boundary) and a=3.0 (upper boundary).")
    lines.append("")

    for boundary_idx, dose_label in [(0, "a=1.8 (lower boundary)"), (3, "a=3.0 (upper boundary)")]:
        lines.append(f"## {dose_label}")
        lines.append("")
        lines.append("| SNR | n | REG bse | DRK bse | RFF500 bse | RFF/DRK | Winner |")
        lines.append("|---|---|---|---|---|---|---|")
        for (snr, n), cell in sorted(results.items()):
            reg = cell.get("REG_Hsuper")
            drk = cell.get("DRK_Hsuper")
            rff = cell.get("RFF500")
            if any(x is None for x in [reg, drk, rff]):
                continue
            reg_b = float(reg["bse_grid"][boundary_idx])
            drk_b = float(drk["bse_grid"][boundary_idx])
            rff_b = float(rff["bse_grid"][boundary_idx])
            ratio = rff_b / drk_b if drk_b > 0 else float("inf")
            # winner = min of the three
            best_b = min(reg_b, drk_b, rff_b)
            if best_b == rff_b:
                winner = "RFF500"
            elif best_b == drk_b:
                winner = "DRK"
            else:
                winner = "REG"
            lines.append(
                f"| {snr} | {n} | {reg_b:.3f} | {drk_b:.3f} | {rff_b:.3f} | "
                f"{ratio:.2f}x | **{winner}** |"
            )
        lines.append("")
    return "\n".join(lines)


def _build_transition_report(results: dict, M: int) -> str:
    import datetime
    lines = []
    lines.append(f"# Phase 13A SNR transition map (boundary a=1.8)")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append(f"M = {M}")
    lines.append("")
    lines.append("Hypothesis: RFF500 boundary advantage strengthens monotonely with SNR.")
    lines.append("")

    # Group by n, then sweep SNR
    n_seen = sorted({n for (_, n) in results.keys()})
    for n in n_seen:
        lines.append(f"## n = {n}")
        lines.append("")
        lines.append("| SNR | DRK bse(a=1.8) | RFF500 bse(a=1.8) | RFF/DRK | reduction | winner |")
        lines.append("|---|---|---|---|---|---|")
        for (snr, n_), cell in sorted(results.items()):
            if n_ != n:
                continue
            drk = cell.get("DRK_Hsuper")
            rff = cell.get("RFF500")
            if drk is None or rff is None:
                continue
            drk_b = float(drk["bse_grid"][0])
            rff_b = float(rff["bse_grid"][0])
            ratio = rff_b / drk_b if drk_b > 0 else float("inf")
            reduction = (1 - ratio) * 100
            winner = "RFF500" if rff_b < drk_b else "DRK"
            lines.append(
                f"| {snr} | {drk_b:.3f} | {rff_b:.3f} | {ratio:.2f}x | "
                f"{reduction:+.1f}% | **{winner}** |"
            )
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    # Determine SNR threshold: smallest SNR where RFF beats DRK at a=1.8
    snr_seen = sorted({s for (s, _) in results.keys()})
    threshold = None
    for snr in snr_seen:
        # average across n at this SNR
        ratios = []
        for (s, n_), cell in results.items():
            if s != snr:
                continue
            drk = cell.get("DRK_Hsuper")
            rff = cell.get("RFF500")
            if drk is None or rff is None:
                continue
            drk_b = float(drk["bse_grid"][0])
            rff_b = float(rff["bse_grid"][0])
            if drk_b > 0:
                ratios.append(rff_b / drk_b)
        if ratios and np.mean(ratios) < 1.0:
            threshold = snr
            break
    if threshold is not None:
        lines.append(f"- RFF500 starts beating DRK at the lower boundary at SNR ~ {threshold:.2f}.")
    else:
        lines.append("- RFF500 does not consistently beat DRK at the lower boundary in this grid.")
    lines.append("")
    return "\n".join(lines)


# ── Main entry ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 13A Coverage Map")
    parser.add_argument("--mode", choices=["smoke", "reduced", "full", "boundary"],
                        default="smoke")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--M", type=int, default=None,
                        help="Override default M for the chosen mode")
    args = parser.parse_args()
    mode = args.mode

    if mode == "smoke":
        grid = GRID_SMOKE
        M = args.M or 3
    elif mode == "reduced":
        grid = GRID_REDUCED
        M = args.M or 50
    elif mode == "full":
        grid = GRID_FULL
        M = args.M or 100
    elif mode == "boundary":
        grid = GRID_BOUNDARY
        M = args.M or 100
    else:
        raise ValueError(f"unknown mode {mode}")

    print("=" * 72)
    print(f"Phase 13A Coverage Map  --mode {mode}, M={M}")
    print(f"  SNR in {grid['snr']}")
    print(f"  n   in {grid['n']}")
    print(f"  cells = {len(grid['snr']) * len(grid['n'])}")
    print(f"  methods = REG_Hsuper / DRK_Hsuper / RFF500")
    print(f"  seed_base = {SEED_BASE}")
    print("=" * 72)

    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    _RAW_BASE.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results, J_pt_per_cell = _run_full_grid(grid, M=M, force=args.force)
    elapsed = time.time() - t0
    print(f"\nTotal coverage map: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Build reports
    main_report = _build_main_report(results, J_pt_per_cell, M)
    main_path = _SUMM_DIR / f"phase13_coverage_map_{mode}.md"
    with open(main_path, "w", encoding="utf-8") as fh:
        fh.write(main_report)
    print(f"Main report: {main_path}")

    boundary_report = _build_boundary_report(results, M)
    boundary_path = _SUMM_DIR / f"phase13_boundary_map_{mode}.md"
    with open(boundary_path, "w", encoding="utf-8") as fh:
        fh.write(boundary_report)
    print(f"Boundary report: {boundary_path}")

    transition_report = _build_transition_report(results, M)
    transition_path = _SUMM_DIR / f"phase13_snr_transition_{mode}.md"
    with open(transition_path, "w", encoding="utf-8") as fh:
        fh.write(transition_report)
    print(f"SNR transition report: {transition_path}")

    # Echo summary
    print()
    print("=" * 72)
    print("MASTER TABLE PREVIEW (first 20 rows):")
    print("=" * 72)
    for line in main_report.split("\n")[:25]:
        print(line)


if __name__ == "__main__":
    main()
