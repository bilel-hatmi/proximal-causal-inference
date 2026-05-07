"""
Phase 19C -- Identifiability cross lambda_h x lambda_r on DGP2 (Bennett indep DR).

Mirror of Phase 19B but for Bennett: instead of cross h x q, we cross
the two regularisations of the Bennett-indep functional DR estimator.

Design
------
DGP        : MichaelisMentenDGP, n=2000, SNR_W=SNR_Z=0.95
Method     : BennettIndepFunctionalDR (h-bridge RFF + Riesz r RFF)
Cross lambda_h x lambda_r grid (3x3 = 9 configs):
  lambda_h: {1e-7 (under), 1e-5 (super), 1e-1 (over)}
  lambda_r: {1e-5 (under), 1e-2 (super), 1e0  (over)}
M          : 50 paired seeds
seed_base  : 92_000_000  (disjoint from Phase 19A 90M, Phase 19B 91M, Phase 16 EXP-A 40M)
a_grid     : [1.8, 2.2, 2.6, 3.0]
REF_IDX    : 2
Bridge config (oracle BIN_mh2000 winner):
  m_h=2000, ell_h=2.75, m_c=500, gamma_critic=1e-3
  m_r=500, ell_scale_r=3.5

Goal: empirical evidence that the Bennett functional is identifiable in
the (lambda_h, lambda_r) space -- counterpart to Phase 19B for DRKernel.

Output
------
  raw : simulations/results/raw/phase19C/h{lvl}_r{lvl}_M50.pkl
  report : simulations/results/summaries/phase19C_bennett_identifiability.md
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
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "s7" / "identifiability_bennett"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "identifiability_bennett_FINAL.md"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
REF_IDX = 2
N = 2000
SNR = 0.95
SEED_BASE = 92_000_000

# Cross lambda_h x lambda_r regularisation grid
H_LEVELS = {  # lambda_h for BennettIndepBridgeH
    "under": 1e-7,
    "super": 1e-5,   # BIN_mh2000 winner
    "over":  1e-1,
}
R_LEVELS = {  # lambda_r for BennettPolicyRiesz
    "under": 1e-5,
    "super": 1e-2,   # BIN_mh2000 winner
    "over":  1e0,
}

# BIN_mh2000 winner config (from phase14B)
M_H = 2000
ELL_H = 2.75
M_C = 500
GAMMA_CRITIC = 1e-3
M_R = 500
ELL_SCALE_R = 3.5

from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
    _compute_J_policy_true,
)
from pci.runner.diagnostics import _augment_decomp_with_ser


def _make_factory(lambda_h: float, lambda_r: float, *, a_grid, h_KDE, ref_idx):
    def factory():
        from simulations.methods.bennett_indep_dr import BennettIndepFunctionalDR
        return BennettIndepFunctionalDR(
            m_h=M_H,
            m_c=M_C,
            ell_h=ELL_H,
            ell_c=ELL_H,           # critic uses same ell_h
            lambda_h=lambda_h,
            gamma_critic=GAMMA_CRITIC,
            lambda_r=lambda_r,
            n_features_r=M_R,
            ell_scale_r=ELL_SCALE_R,
            stabilized_r=False,
            n_folds=5,
            a_grid=list(a_grid),
            bandwidth=h_KDE,
            ref_dose_index=ref_idx,
            seed=42,
        )
    return factory


def _run_config(*, h_lvl, r_lvl, dgp, a_grid, h_KDE, J_pt, M, seed_base,
                 raw_dir, force=False):
    lambda_h = H_LEVELS[h_lvl]
    lambda_r = R_LEVELS[r_lvl]
    factory = _make_factory(lambda_h, lambda_r,
                              a_grid=a_grid, h_KDE=h_KDE, ref_idx=REF_IDX)
    label = f"h{h_lvl}_r{r_lvl}_M{M}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for f in raw_dir.glob(f"{label}.pkl"):
            f.unlink()

    return _run_block(
        dgp, factory, a_grid, J_pt,
        n=N, M=M, seed_base=seed_base,
        label=label, raw_dir=raw_dir,
    )


def _write_report(per_config):
    import datetime
    lines = []
    lines.append("# Phase 19C -- Identifiability cross lambda_h x lambda_r "
                 "(Bennett indep DR, DGP2)")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append("")
    lines.append(f"Cell: n={N}, SNR_W=SNR_Z={SNR}")
    lines.append(f"M=50, seed_base={SEED_BASE}")
    lines.append(f"a_grid={A_GRID.tolist()}, REF_IDX={REF_IDX} "
                 f"(a={A_GRID[REF_IDX]})")
    lines.append("")
    lines.append(f"Bridge config (BIN_mh2000): m_h={M_H}, ell_h={ELL_H}, "
                 f"m_c={M_C}, m_r={M_R}, ell_scale_r={ELL_SCALE_R}")
    lines.append(f"H levels (lambda_h): {H_LEVELS}")
    lines.append(f"R levels (lambda_r): {R_LEVELS}")
    lines.append("")
    lines.append("## Heatmap bse_ref by (h_level, r_level)")
    lines.append("")
    h_levels = list(H_LEVELS.keys())
    r_levels = list(R_LEVELS.keys())
    lines.append("| h \\ r | " + " | ".join(r_levels) + " |")
    lines.append("|" + "|".join(["---"] * (len(r_levels) + 1)) + "|")

    def _fmt(v):
        if v is None or not np.isfinite(v):
            return "---"
        if abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
            return f"{v:.2e}"
        return f"{v:.4f}"

    for h_lvl in h_levels:
        cells = [h_lvl]
        for r_lvl in r_levels:
            d = per_config.get((h_lvl, r_lvl))
            if d is None:
                cells.append("---")
            else:
                cells.append(_fmt(d["bse_grid"][REF_IDX]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Heatmap cov_ref by (h_level, r_level)")
    lines.append("")
    lines.append("| h \\ r | " + " | ".join(r_levels) + " |")
    lines.append("|" + "|".join(["---"] * (len(r_levels) + 1)) + "|")
    for h_lvl in h_levels:
        cells = [h_lvl]
        for r_lvl in r_levels:
            d = per_config.get((h_lvl, r_lvl))
            if d is None:
                cells.append("---")
            else:
                cells.append(f"{d['coverage_ref']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Detailed per-config")
    cols = ["bias_ref", "bse_ref", "cov_ref", "ser_ref"]
    lines.append("")
    lines.append("| h | r | " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * (2 + len(cols))) + "|")
    for h_lvl in h_levels:
        for r_lvl in r_levels:
            d = per_config.get((h_lvl, r_lvl))
            if d is None:
                lines.append(f"| {h_lvl} | {r_lvl} | (failed) | | | |")
                continue
            row = [
                _fmt(d["bias_dr"][REF_IDX]),
                _fmt(d["bse_grid"][REF_IDX]),
                _fmt(d["coverage_ref"]),
                _fmt(d.get("ser_ref", float("nan"))),
            ]
            lines.append(f"| {h_lvl} | {r_lvl} | " + " | ".join(row) + " |")
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
        configs = [("super", "super")]
        M = 3
    else:
        configs = list(product(H_LEVELS.keys(), R_LEVELS.keys()))
        M = args.M

    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)
    ref = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref.A)
    J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)
    print(f"\n=== Phase 19C  n={N}  SNR={SNR}  J_pt={J_pt}  h_KDE={h_KDE:.3f} ===")

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()
    per_config: Dict[tuple, dict] = {}
    for (h_lvl, r_lvl) in configs:
        print(f"\n  Config h={h_lvl} (lambda_h={H_LEVELS[h_lvl]:.0e}) "
              f"x r={r_lvl} (lambda_r={R_LEVELS[r_lvl]:.0e})")
        t0 = time.time()
        try:
            data = _run_config(
                h_lvl=h_lvl, r_lvl=r_lvl,
                dgp=dgp, a_grid=A_GRID, h_KDE=h_KDE, J_pt=J_pt,
                M=M, seed_base=SEED_BASE, raw_dir=_RAW_DIR,
                force=args.force,
            )
            decomp = _decompose(data)
            if decomp is not None:
                decomp = _augment_decomp_with_ser(decomp, data)
                decomp["_data"] = data
            per_config[(h_lvl, r_lvl)] = decomp
            elapsed = time.time() - t0
            if decomp is not None:
                bse = float(decomp["bse_grid"][REF_IDX])
                cov = float(decomp["coverage_ref"])
                ser = float(decomp.get("ser_ref", float("nan")))
                print(f"    bse_ref={bse:.4f}  cov={cov:.3f}  SER={ser:.3f}  "
                      f"[{elapsed:.0f}s]")
            else:
                print(f"    [no valid reps]  [{elapsed:.0f}s]")
        except Exception as e:
            import traceback
            print(f"    ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            per_config[(h_lvl, r_lvl)] = None

    _write_report(per_config)
    elapsed = time.time() - t_total
    print(f"\n=== Phase 19C total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")


if __name__ == "__main__":
    main()
