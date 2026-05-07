"""
Phase 15C E3 -- W/Z asymmetry map on DGP2.

Questions covered: Q2 (PCI feasibility under proxy degradation),
Q3 (W/Z asymmetry, is Z more critical than W?).

Design:
  DGP : MichaelisMentenDGP (DGP2)
  4 cells (SNR_W, SNR_Z) :
    (0.95, 0.95)   baseline
    (0.95, 0.50)   Z weak: tests q-bridge via Z
    (0.50, 0.95)   W weak: tests h-bridge via W
    (0.50, 0.50)   PCI no-go zone
  n = 1000, M = 50, seed_base = 33_000_000
  Methods : 4 = {linearREG, KPVREG, DRKPV, best_bennett}
            (drop oracle for compute, drop Kallus -- saved by E2 anyway)

Expected (hypothesis):
  (0.95, 0.95) baseline strong
  (0.95, 0.50) Z weak  : DRKPV/best_bennett correction degrades
  (0.50, 0.95) W weak  : linearREG/KPVREG (h-bridge) degrades
  (0.50, 0.50) overall failure (PCI no-go)

If Z more critical than W -> methodological contribution for S6.

Outputs:
  simulations/results/raw/phase15C_E3/<snrW>_<snrZ>/<method>_M50.pkl
  simulations/results/summaries/phase15C_E3_asymmetry.md
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[3]  # +1: moved to experiments/simulation/
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "s7" / "proxy_asymmetry"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "proxy_asymmetry_FINAL.md"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
K = len(A_GRID)
REF_IDX = 2
N_FOLDS = 5
RANDOM_STATE = 42
SEED_BASE = 33_000_000

METHODS_E3 = ["linearREG", "KPVREG", "DRKPV", "best_bennett"]
SNR_CELLS: Tuple[Tuple[float, float], ...] = (
    (0.95, 0.95),
    (0.95, 0.50),
    (0.50, 0.95),
    (0.50, 0.50),
)

from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
    _compute_J_policy_true,
)
from pci.runner.diagnostics import _augment_decomp_with_ser
from pci.runner.reporting import (
    _aggregate_metrics, _print_table, _build_stage_report, _save_stage_report,
)
from simulations.methods.method_registry import METHOD_REGISTRY


def _run_method(method_name: str, *, dgp, a_grid, h_KDE, ell_W0, ell_A0, ell_Z0,
                 J_pt, n: int, M: int, seed_base: int, raw_dir: Path,
                 ref_idx: int, force=False):
    factory_factory = METHOD_REGISTRY[method_name]

    def factory():
        kwargs = dict(dgp=dgp, a_grid=a_grid, h_KDE=h_KDE,
                       ell_W0=ell_W0, ell_A0=ell_A0, ell_Z0=ell_Z0,
                       ref_dose_index=ref_idx)
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


def _run_cell(*, snr_W: float, snr_Z: float, n: int, M: int, force=False) -> Dict[str, dict]:
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=snr_W, snr_Z=snr_Z)
    ref = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref.A)
    ell_W0 = _median_bandwidth(ref.W)
    ell_A0 = _median_bandwidth(ref.A)
    ell_Z0 = _median_bandwidth(ref.Z)
    J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)
    print(f"\n=== Cell SNR_W={snr_W}, SNR_Z={snr_Z}, n={n}  J_policy_true={J_pt} ===")

    cell_tag = f"snrW{int(snr_W*100):02d}_snrZ{int(snr_Z*100):02d}"
    raw_dir = _RAW_DIR / cell_tag
    out: Dict[str, dict] = {}
    # Cell-specific seed offset for paired but distinct seeds
    seed_base_cell = SEED_BASE + int(snr_W*100)*100 + int(snr_Z*100)
    for method_name in METHODS_E3:
        print(f"  [{method_name}] n={n}")
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


def _write_e3_report(per_cell: Dict[Tuple[float, float], Dict[str, dict]]):
    import datetime
    lines = []
    lines.append("# Phase 15C E3 -- W/Z asymmetry map (DGP2)")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append("")
    lines.append(f"DGP2, n=1000, M=50, seed_base={SEED_BASE}")
    lines.append(f"a_grid={A_GRID.tolist()}, REF_IDX={REF_IDX}")
    lines.append("")
    lines.append("Hypothesis: Z weak hurts DR correction (q-bridge via Z); "
                 "W weak hurts h-bridge (REG-only methods).")
    lines.append("")
    cols = ["bias_ref", "bse_ref", "bse_a18", "bse_a30", "cov_ref", "ser_ref",
            "ESS_min_mean"]
    # Per-cell table
    for (snr_W, snr_Z), results in per_cell.items():
        lines.append(f"## SNR_W={snr_W}, SNR_Z={snr_Z}")
        lines.append("")
        rows = _aggregate_metrics({k: v for k, v in results.items() if v is not None})
        lines.append("| method | " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
        for method_name in METHODS_E3:
            m = rows.get(method_name)
            if m is None:
                lines.append(f"| {method_name} | (failed) |" + " |".join(["" for _ in cols[1:]]) + " |")
                continue
            cells = [method_name]
            for c in cols:
                v = m.get(c, float("nan"))
                if not np.isfinite(v):
                    cells.append("---")
                elif abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
                    cells.append(f"{v:.2e}")
                else:
                    cells.append(f"{v:.4f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Asymmetry comparison: bse_ref summary across cells
    lines.append("## Cross-cell summary (bse_ref)")
    lines.append("")
    lines.append("| method | (.95,.95) | (.95,.50) | (.50,.95) | (.50,.50) |")
    lines.append("|---|---|---|---|---|")
    for method_name in METHODS_E3:
        cells = [method_name]
        for snrs in SNR_CELLS:
            decomp = per_cell.get(snrs, {}).get(method_name)
            if decomp is None:
                cells.append("---")
            else:
                bse = float(decomp["bse_grid"][REF_IDX])
                cells.append(f"{bse:.4f}" if np.isfinite(bse) else "---")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Z-vs-W diagnostic: compare degradation (.95,.50) - (.95,.95) "
                 "vs (.50,.95) - (.95,.95). Larger Z-side hit indicates Z is the "
                 "more critical proxy.")
    lines.append("")

    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Report: {_REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=str, default="all",
                         help="comma-separated SNR pairs, e.g. '0.95,0.95;0.95,0.50' or 'all'")
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    if args.cells == "all":
        cells = list(SNR_CELLS)
    else:
        cells = []
        for chunk in args.cells.split(";"):
            sw, sz = chunk.strip().split(",")
            cells.append((float(sw), float(sz)))

    per_cell: Dict[Tuple[float, float], Dict[str, dict]] = {}
    for snr_W, snr_Z in cells:
        per_cell[(snr_W, snr_Z)] = _run_cell(
            snr_W=snr_W, snr_Z=snr_Z, n=args.n, M=args.M, force=args.force,
        )

    _write_e3_report(per_cell)

    elapsed = time.time() - t_total
    print(f"\n=== Phase 15C E3 total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")


if __name__ == "__main__":
    main()
