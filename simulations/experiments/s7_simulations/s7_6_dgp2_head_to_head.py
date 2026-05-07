"""
Phase 15C E2 -- DGP2 nonlinear centre.

Questions covered: Q4 (linearity sufficiency), Q5 (h-only), Q6 (correction
utility), Q8 (KPV safe).

Design:
  DGP : MichaelisMentenDGP (DGP2)
  Cells : 2 = (n in {1000, 2000}) x SNR=0.95
  Methods : ALL 8 from METHOD_REGISTRY
  M = 100 paired seeds, seed_base = 32_500_000

Expected:
  - linearREG/DR fail (nonlinear bridge)
  - KPVREG/DRKPV moderate
  - best_bennett dominates (Phase 14C.2 confirmed)
  - best_kallus competitive at n=1000, fails at n=2000 (Phase 14C.2 pattern)

Outputs:
  simulations/results/raw/phase15C_E2/<n>/<method>_M100.pkl
  simulations/results/summaries/phase15C_E2_dgp2.md
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Dict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[3]  # +1: moved to experiments/simulation/
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "s7" / "dgp2_head_to_head"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "dgp2_head_to_head_FINAL.md"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
K = len(A_GRID)
REF_IDX = 2  # a=2.6 centre per Phase 14B convention
N_FOLDS = 5
RANDOM_STATE = 42
SEED_BASE = 32_500_000

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
        if method_name == "oracle":
            kwargs["is_linear_in_A"] = False  # DGP2 nonlinear
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


def _run_cell(*, n: int, M: int, force=False) -> Dict[str, dict]:
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)
    ref = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref.A)
    ell_W0 = _median_bandwidth(ref.W)
    ell_A0 = _median_bandwidth(ref.A)
    ell_Z0 = _median_bandwidth(ref.Z)
    J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)
    print(f"\n=== Cell n={n}  SNR_W=SNR_Z=0.95  J_policy_true={J_pt}  h_KDE={h_KDE:.3f} ===")

    raw_dir = _RAW_DIR / f"n{n}"
    out: Dict[str, dict] = {}
    seed_base_cell = SEED_BASE + (0 if n == 1000 else 1000)
    for method_name in METHOD_REGISTRY.keys():
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


def _write_e2_report(per_n: Dict[int, Dict[str, dict]]):
    import datetime
    lines = []
    lines.append("# Phase 15C E2 -- DGP2 nonlinear centre")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append("")
    lines.append("DGP2 = MichaelisMentenDGP, SNR_W=SNR_Z=0.95, "
                 "M=100, seed_base=32_500_000")
    lines.append(f"a_grid={A_GRID.tolist()}, REF_IDX={REF_IDX} (a={A_GRID[REF_IDX]})")
    lines.append("")
    lines.append("Expected: linearREG/linearDR fail; KPVREG/DRKPV moderate; "
                 "best_bennett dominates; best_kallus competitive at n=1000.")
    lines.append("")
    cols = ["bias_ref", "bse_ref", "bse_a18", "bse_a30", "cov_ref", "ser_ref",
            "ESS_min_mean", "w_p99_mean"]
    for n, results in per_n.items():
        lines.append(f"## n={n}")
        lines.append("")
        rows = _aggregate_metrics({k: v for k, v in results.items() if v is not None})
        lines.append("| method | " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
        for method_name in METHOD_REGISTRY.keys():
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
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Report: {_REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    n_list = [args.n] if args.n is not None else [1000, 2000]
    per_n: Dict[int, Dict[str, dict]] = {}
    for n in n_list:
        per_n[n] = _run_cell(n=n, M=args.M, force=args.force)

    _write_e2_report(per_n)

    elapsed = time.time() - t_total
    print(f"\n=== Phase 15C E2 total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")


if __name__ == "__main__":
    main()
