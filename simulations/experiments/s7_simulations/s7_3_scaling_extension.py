"""
Phase 19A -- Scaling map: extends Phase 16 EXP-A to n in {250, 500}.

Design
------
DGP        : MichaelisMentenDGP (DGP2)
Methods    : [linearDR, KPVREG, DRKPV, best_bennett]
Cells      : n in {250, 500} x SNR=0.95 (symmetric W=Z) -> 2 cells
M          : 100 paired seeds (smaller than EXP-A's 200 for compute budget)
seed_base  : 90_000_000  (disjoint from Phase 16 EXP-A's 40M)
a_grid     : [1.8, 2.2, 2.6, 3.0]
REF_IDX    : 2 (a=2.6 centre)

Goal: complete the n-axis (n=1000, 2000 already in Phase 16 EXP-A) for
the scaling figure F-NEW-C.

Output
------
  raw : simulations/results/raw/phase19A/n{n}_snr95/{method}_M100.pkl
  report : simulations/results/summaries/phase19A_scaling.md
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
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "s7" / "scaling_extension"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "scaling_map_FINAL.md"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
K = len(A_GRID)
REF_IDX = 2
SEED_BASE = 90_000_000

METHODS_19A = ["linearDR", "KPVREG", "DRKPV", "best_bennett"]
N_LIST = [250, 500]
SNR_LIST = [0.95]

from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
    _compute_J_policy_true,
)
from pci.runner.diagnostics import _augment_decomp_with_ser
from simulations.methods.method_registry import METHOD_REGISTRY


def _run_method(method_name, *, dgp, a_grid, h_KDE, ell_W0, ell_A0, ell_Z0,
                J_pt, n, M, seed_base, raw_dir, ref_idx, force=False):
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


def _run_cell(*, n, snr, M, force=False):
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
    seed_base_cell = SEED_BASE + N_LIST.index(n) * 1000
    for method_name in METHODS_19A:
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
            import traceback
            print(f"    ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            out[method_name] = None
    return out


def _write_report(per_cell):
    import datetime
    lines = []
    lines.append("# Phase 19A -- Scaling map (DGP2, n in {250, 500}, SNR=0.95)")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append("")
    lines.append("Goal: complete the n-axis (n=1000, 2000 already in Phase 16 EXP-A)")
    lines.append(f"Methods: {METHODS_19A}")
    lines.append(f"M=100, seed_base={SEED_BASE}")
    lines.append("")
    cols = ["bias_ref", "bse_ref", "bse_a18", "bse_a30", "cov_ref", "ser_ref"]

    def _fmt(v):
        if v is None or not np.isfinite(v):
            return "---"
        if abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
            return f"{v:.2e}"
        return f"{v:.4f}"

    for (n, snr), results in sorted(per_cell.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        lines.append(f"## n={n}, SNR={snr}")
        lines.append("")
        lines.append("| method | " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
        for method_name in METHODS_19A:
            decomp = results.get(method_name)
            if decomp is None:
                lines.append(f"| {method_name} | (failed) |" + "|".join(["" for _ in cols[1:]]) + " |")
                continue
            bias_ref = decomp["bias_dr"][REF_IDX]
            bse_ref = decomp["bse_grid"][REF_IDX]
            bse_a18 = decomp["bse_grid"][0]
            bse_a30 = decomp["bse_grid"][3]
            cov_ref = decomp["coverage_ref"]
            ser_ref = decomp.get("ser_ref", float("nan"))
            row = [_fmt(v) for v in [bias_ref, bse_ref, bse_a18, bse_a30, cov_ref, ser_ref]]
            lines.append(f"| {method_name} | " + " | ".join(row) + " |")
        lines.append("")

    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Report: {_REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.mode == "smoke":
        n_list_local = [250]
        M = 3
    else:
        n_list_local = N_LIST
        M = args.M

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    per_cell = {}
    for n in n_list_local:
        for snr in SNR_LIST:
            per_cell[(n, snr)] = _run_cell(n=n, snr=snr, M=M, force=args.force)

    _write_report(per_cell)
    elapsed = time.time() - t_total
    print(f"\n=== Phase 19A total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")


if __name__ == "__main__":
    main()
