"""
Phase 16 EXP-B -- Proxy feasibility map: 4x4 SNR grid.

Design
------
DGP        : MichaelisMentenDGP (DGP2)
Methods    : [linearDR, KPVREG, DRKPV, best_bennett]
Cells      : SNR_W in {0.50, 0.70, 0.85, 0.95} x SNR_Z in idem  -> 16 cells
M          : 80 paired seeds, n = 1000
seed_base  : 41_000_000
a_grid     : [1.8, 2.2, 2.6, 3.0]
REF_IDX    : 2

Goal: extend Phase 15C E3 (4 cells) to a full 4x4 feasibility map.
Identifies the no-go zone where no method achieves cov >= 0.85.
Output feeds figure F3 (4-panel feasibility heatmap).

Output
------
  raw : simulations/results/raw/phase16_EXPB/snrW{XX}_snrZ{YY}/{method}_M80.pkl
  report : simulations/results/summaries/phase16_EXPB_proxy_feasibility.md
"""
from __future__ import annotations

import argparse
import os
import time
from itertools import product
from pathlib import Path
from typing import Dict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[3]  # +1: moved to experiments/simulation/
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "s7" / "proxy_quality_grid"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "proxy_feasibility_FINAL.md"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
K = len(A_GRID)
REF_IDX = 2
N = 1000
SEED_BASE = 41_000_000

METHODS_EXPB = ["linearDR", "KPVREG", "DRKPV", "best_bennett"]
SNR_LEVELS = [0.50, 0.70, 0.85, 0.95]
SNR_CELLS = list(product(SNR_LEVELS, SNR_LEVELS))  # 16 cells

from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
    _compute_J_policy_true,
)
from pci.runner.diagnostics import _augment_decomp_with_ser
from simulations.methods.method_registry import METHOD_REGISTRY


def _run_method(method_name: str, *, dgp, a_grid, h_KDE, ell_W0, ell_A0, ell_Z0,
                 J_pt, n: int, M: int, seed_base: int, raw_dir: Path,
                 ref_idx: int, force=False):
    factory_factory = METHOD_REGISTRY[method_name]

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


def _run_cell(*, snr_W: float, snr_Z: float, M: int, force=False) -> Dict[str, dict]:
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=snr_W, snr_Z=snr_Z)
    ref = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref.A)
    ell_W0 = _median_bandwidth(ref.W)
    ell_A0 = _median_bandwidth(ref.A)
    ell_Z0 = _median_bandwidth(ref.Z)
    J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)
    print(f"\n=== Cell SNR_W={snr_W}  SNR_Z={snr_Z}  J_pt={J_pt}  h_KDE={h_KDE:.3f} ===")

    cell_tag = f"snrW{int(snr_W * 100):02d}_snrZ{int(snr_Z * 100):02d}"
    raw_dir = _RAW_DIR / cell_tag
    out: Dict[str, dict] = {}
    seed_base_cell = SEED_BASE + int(snr_W * 100) * 10_000 + int(snr_Z * 100)
    for method_name in METHODS_EXPB:
        print(f"  [{method_name}] {cell_tag}")
        t0 = time.time()
        try:
            data = _run_method(
                method_name, dgp=dgp, a_grid=A_GRID, h_KDE=h_KDE,
                ell_W0=ell_W0, ell_A0=ell_A0, ell_Z0=ell_Z0,
                J_pt=J_pt, n=N, M=M, seed_base=seed_base_cell,
                raw_dir=raw_dir, ref_idx=REF_IDX, force=force,
            )
            decomp = _decompose(data)
            if decomp is not None:
                decomp = _augment_decomp_with_ser(decomp, data)
                decomp["_data"] = data
            out[method_name] = decomp
            elapsed = time.time() - t0
            if decomp is not None:
                bse = float(decomp["bse_grid"][REF_IDX])
                cov = float(decomp["coverage_ref"])
                ser = float(decomp.get("ser_ref", float("nan")))
                print(f"    bse_ref={bse:.4f}  cov={cov:.3f}  SER={ser:.3f}  [{elapsed:.0f}s]")
            else:
                print(f"    [no valid reps]  [{elapsed:.0f}s]")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            out[method_name] = None
    return out


def _write_report(per_cell: Dict[tuple, Dict[str, dict]]):
    import datetime
    lines = []
    lines.append("# Phase 16 EXP-B -- Proxy feasibility map (4x4 SNR grid)")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append("")
    lines.append("DGP2 = MichaelisMentenDGP, asymmetric (snr_W, snr_Z) grid")
    lines.append(f"M=80, n={N}, seed_base={SEED_BASE}")
    lines.append(f"a_grid={A_GRID.tolist()}, REF_IDX={REF_IDX} (a={A_GRID[REF_IDX]})")
    lines.append(f"SNR_W x SNR_Z grid: {SNR_LEVELS} x {SNR_LEVELS} = 16 cells")
    lines.append("")
    lines.append("Methods: " + ", ".join(METHODS_EXPB))
    lines.append("Goal: identify no-go zone (where no method achieves cov >= 0.85).")
    lines.append("")

    def _fmt(v):
        if v is None or not np.isfinite(v):
            return "---"
        if abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
            return f"{v:.2e}"
        return f"{v:.4f}"

    # Per-method bse_ref heatmap (4x4 each)
    for method_name in METHODS_EXPB:
        lines.append(f"## {method_name} -- bse_ref by (SNR_W, SNR_Z)")
        lines.append("")
        header = "| SNR_W \\ SNR_Z | " + " | ".join([str(s) for s in SNR_LEVELS]) + " |"
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (1 + len(SNR_LEVELS))) + "|")
        for snr_W in SNR_LEVELS:
            cells = [str(snr_W)]
            for snr_Z in SNR_LEVELS:
                d = per_cell.get((snr_W, snr_Z), {}).get(method_name)
                if d is None:
                    cells.append("---")
                else:
                    cells.append(_fmt(d["bse_grid"][REF_IDX]))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Per-method coverage heatmap (4x4 each)
    for method_name in METHODS_EXPB:
        lines.append(f"## {method_name} -- cov_ref by (SNR_W, SNR_Z)")
        lines.append("")
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (1 + len(SNR_LEVELS))) + "|")
        for snr_W in SNR_LEVELS:
            cells = [str(snr_W)]
            for snr_Z in SNR_LEVELS:
                d = per_cell.get((snr_W, snr_Z), {}).get(method_name)
                if d is None:
                    cells.append("---")
                else:
                    cells.append(f"{d['coverage_ref']:.3f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Detailed table
    lines.append("## Detailed per-cell metrics")
    lines.append("")
    cols = ["bias_ref", "bse_ref", "cov_ref", "ser_ref", "ESS_min_mean", "w_p99_mean"]
    for (snr_W, snr_Z), results in sorted(per_cell.items()):
        cell_tag = f"SNR_W={snr_W}, SNR_Z={snr_Z}"
        lines.append(f"### {cell_tag}")
        lines.append("")
        lines.append("| method | " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
        for method_name in METHODS_EXPB:
            d = results.get(method_name)
            if d is None:
                lines.append(f"| {method_name} | (failed) |" +
                              "|".join(["" for _ in cols[1:]]) + " |")
                continue
            row = [
                _fmt(d["bias_dr"][REF_IDX]),
                _fmt(d["bse_grid"][REF_IDX]),
                _fmt(d["coverage_ref"]),
                _fmt(d.get("ser_ref", float("nan"))),
                _fmt(d.get("ess_min_mean", float("nan"))),
                _fmt(d.get("w_p99_mean", float("nan"))),
            ]
            lines.append(f"| {method_name} | " + " | ".join(row) + " |")
        lines.append("")

    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Report: {_REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--M", type=int, default=80)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.mode == "smoke":
        # 1 cell (.95, .95), 4 methods, M=3
        cells = [(0.95, 0.95)]
        M = 3
    else:
        cells = SNR_CELLS
        M = args.M

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    per_cell: Dict[tuple, Dict[str, dict]] = {}
    for snr_W, snr_Z in cells:
        per_cell[(snr_W, snr_Z)] = _run_cell(
            snr_W=snr_W, snr_Z=snr_Z, M=M, force=args.force
        )

    _write_report(per_cell)

    elapsed = time.time() - t_total
    print(f"\n=== Phase 16 EXP-B total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")


if __name__ == "__main__":
    main()
