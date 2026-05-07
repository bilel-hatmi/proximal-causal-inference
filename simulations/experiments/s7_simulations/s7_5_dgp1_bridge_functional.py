"""
Phase 15C E1 -- DGP1 sanity (parametric benchmark).

Question covered: Q1 (target J(pi)), Q4 (linearity sufficiency), Q5 (h-only).

Design:
  DGP : CobbDouglasLinearDGP (DGP1, log-linear h_0)
  Cells : 2 = (n in {1000, 2000}) x SNR=0.95
  Methods : ALL 8 from METHOD_REGISTRY (oracle, naiveREG, linearREG,
            linearDR, KPVREG, DRKPV, best_bennett, best_kallus)
  M = 100 paired seeds, seed_base = 32_000_000

Expected (sanity): oracle ~ linearREG ~ linearDR (DGP1 linear).
KPVREG/DRKPV match. naiveREG biased. best_kallus/best_bennett match.

Outputs:
  simulations/results/raw/phase15C_E1/<n>/<method>_M100.pkl
  simulations/results/summaries/phase15C_E1_dgp1.md
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

_BASE_DIR = Path(__file__).resolve().parents[3]  # +1 because moved from experiments/ to experiments/simulation/
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "s7" / "dgp1_bridge_functional"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "dgp1_sanity_FINAL.md"

A_GRID = np.array([-0.5, 0.0, 0.5])
K = len(A_GRID)
REF_IDX = 1   # a=0.0 centre
N_FOLDS = 5
RANDOM_STATE = 42
SEED_BASE = 32_000_000

from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
)
from pci.runner.diagnostics import _augment_decomp_with_ser
from pci.runner.reporting import (
    _print_table, _build_stage_report, _save_stage_report,
)
from simulations.methods.method_registry import METHOD_REGISTRY


def _aggregate_metrics(stage_results):
    """Grid-agnostic aggregator for DGP1 (3-element a_grid)."""
    rows = {}
    for cid, decomp in stage_results.items():
        if decomp is None:
            continue
        bse = decomp["bse_grid"]
        bias = decomp["bias_dr"]
        K_local = len(bse)
        rows[cid] = {
            "bias_ref": float(bias[REF_IDX]),
            "bse_ref": float(bse[REF_IDX]),
            "bse_a18": float(bse[0]),                        # boundary low
            "bse_a30": float(bse[K_local - 1]),              # boundary high
            "cov_ref": float(decomp["coverage_ref"]),
            "ser_ref": float(decomp.get("ser_ref", float("nan"))),
            "ESS_min_mean": float(decomp.get("ess_min_mean", float("nan"))),
            "w_p99_mean": float(decomp.get("w_p99_mean", float("nan"))),
            "riesz_res_mean": float(decomp.get("riesz_res_mean", float("nan"))),
        }
    return rows


def _compute_J_policy_true_dgp1(dgp, a_grid: np.ndarray, h_KDE: float, n_quad: int = 25):
    """For DGP1 m_true(t) = alpha*t (linear). J(pi_a,h) = alpha*a (since
    Gaussian kernel symmetric)."""
    return np.array([dgp.m_true(a) for a in a_grid], dtype=float)


def _run_method(method_name: str, *, dgp, a_grid, h_KDE, ell_W0, ell_A0, ell_Z0,
                 J_pt, n: int, M: int, seed_base: int, raw_dir: Path,
                 ref_idx: int, force=False):
    """Build factory_fn returning a fresh estimator each rep, then call _run_block."""
    factory_factory = METHOD_REGISTRY[method_name]

    def factory():
        kwargs = dict(dgp=dgp, a_grid=a_grid, h_KDE=h_KDE,
                       ell_W0=ell_W0, ell_A0=ell_A0, ell_Z0=ell_Z0,
                       ref_dose_index=ref_idx)
        # Indicate to oracle that DGP is linear-in-A
        if method_name == "oracle":
            kwargs["is_linear_in_A"] = True
        return factory_factory(**kwargs)

    label = f"{method_name}_M{M}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for f in raw_dir.glob(f"{label}.pkl"):
            f.unlink()

    # Set globals K and ref index for _build_record (it uses module-level K)
    import simulations.experiments.dgp2_bias_diagnostics as bd
    saved_K = bd.K
    saved_REF = getattr(bd, "REF_DOSE_INDEX", None)
    bd.K = K
    if hasattr(bd, "REF_DOSE_INDEX"):
        bd.REF_DOSE_INDEX = ref_idx

    try:
        data = _run_block(
            dgp, factory, a_grid, J_pt,
            n=n, M=M, seed_base=seed_base,
            label=label, raw_dir=raw_dir,
        )
    finally:
        bd.K = saved_K
        if saved_REF is not None and hasattr(bd, "REF_DOSE_INDEX"):
            bd.REF_DOSE_INDEX = saved_REF
    return data


def _run_cell(*, n: int, M: int, force=False) -> Dict[str, dict]:
    from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP
    dgp = CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)
    ref = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref.A)
    ell_W0 = _median_bandwidth(ref.W)
    ell_A0 = _median_bandwidth(ref.A)
    ell_Z0 = _median_bandwidth(ref.Z)
    J_pt = _compute_J_policy_true_dgp1(dgp, A_GRID, h_KDE)
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


def _write_e1_report(per_n: Dict[int, Dict[str, dict]]):
    import datetime
    lines = []
    lines.append("# Phase 15C E1 -- DGP1 sanity (parametric benchmark)")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append("")
    lines.append("DGP1 = CobbDouglasLinearDGP, SNR_W=SNR_Z=0.95, "
                 "M=100, seed_base=32_000_000")
    lines.append(f"a_grid={A_GRID.tolist()}, REF_IDX={REF_IDX} (a=0)")
    lines.append("")
    lines.append("Expected: oracle ~ linearREG ~ linearDR (DGP1 linear); "
                 "KPVREG/DRKPV match; naiveREG biased; best_kallus/best_bennett match.")
    lines.append("")
    cols = ["bias_ref", "bse_ref", "bse_a18", "bse_a30", "cov_ref", "ser_ref",
            "ESS_min_mean", "w_p99_mean"]
    for n, results in per_n.items():
        lines.append(f"## n={n}")
        lines.append("")
        rows = _aggregate_metrics({k: v for k, v in results.items() if v is not None})
        # Note bse_a18/bse_a30 are placeholders pointing to a_grid[0] / a_grid[-1]
        # which are -0.5 and 0.5 here, not 1.8/3.0. Keep the labels for layout
        # consistency but document.
        lines.append("Note: 'bse_a18' = bse at a=-0.5, 'bse_a30' = bse at a=0.5.")
        lines.append("")
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
    parser.add_argument("--n", type=int, default=None,
                         help="If set, run only that n (1000 or 2000)")
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

    _write_e1_report(per_n)

    elapsed = time.time() - t_total
    print(f"\n=== Phase 15C E1 total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")


if __name__ == "__main__":
    main()
