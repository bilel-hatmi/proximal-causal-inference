"""
Phase 12B Stage C -- Damping diagnostic (post-hoc, no new fits).

For top configs from Stage A or Stage B, re-process the per-rep records and
compute J_eta = J_reg + eta * (J_dr - J_reg) for eta in {0.25, 0.5, 0.75, 1.0}.

Diagnostic interpretation:
  - eta low improves while eta=1.0 explodes -> Riesz direction useful but amplitude unstable
  - eta has no effect           -> Riesz correction is decorative
  - eta worsens                 -> Riesz direction wrong

Usage
-----
  python -m simulations.archive.utilities.bennett_rff_damping_stageC --stage A
  python -m simulations.archive.utilities.bennett_rff_damping_stageC --stage A --top 3

Reads PKLs from simulations/results/raw/bennett_rff/stage{A,B}/
Writes report to simulations/results/summaries/phase12B_bennett_rff_stageC.md
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[2]
_RAW_BASE = _BASE_DIR / "simulations" / "results" / "raw" / "bennett_rff"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
REF_IDX = 2

ETAS = [0.25, 0.5, 0.75, 1.0]


def _load_records(pkl_path: Path) -> Optional[dict]:
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as fh:
        return pickle.load(fh)


def _eta_diagnostic(records: list, J_pt: np.ndarray, n: int) -> dict:
    """For one config's records, compute (bias, |b|/SE, cov, SER, V_total) per eta."""
    out = {}
    valid = [r for r in records if r.get("error") is None]
    if not valid:
        return {eta: None for eta in ETAS}
    J_dr_all = np.array([r["J_dr"] for r in valid])    # (M, K)
    J_rg_all = np.array([r["J_reg"] for r in valid])
    V_dr = np.array([r["V_hat_grid"] for r in valid])
    V_rg = np.array([r["V_reg_grid"] for r in valid])

    for eta in ETAS:
        J_eta = J_rg_all + eta * (J_dr_all - J_rg_all)   # (M, K)
        # Approximate variance as var(J_eta) across reps + per-rep V_hat for SER:
        # For SER, we need a proxy V_hat for J_eta. Approx: V_eta = var across (eta scaled)
        # contribution. Use empirical SD of J_eta across reps as truth, and approximate
        # predicted SE via interpolation: V_eta ~ V_reg + eta^2 (V_dr - V_reg).
        V_eta = V_rg + (eta ** 2) * np.maximum(V_dr - V_rg, 0.0)
        SE_eta = np.sqrt(np.mean(V_eta, axis=0) / n)
        bias_eta = J_eta.mean(axis=0) - J_pt
        bse_eta = np.abs(bias_eta) / np.where(SE_eta > 0, SE_eta, np.nan)
        cov_eta = float(np.mean(
            np.abs(J_eta[:, REF_IDX] - J_pt[REF_IDX]) <= 1.96 * SE_eta[REF_IDX]
        ))
        emp_sd = float(np.std(J_eta[:, REF_IDX], ddof=1))
        ser = float(SE_eta[REF_IDX] / emp_sd) if emp_sd > 1e-12 else np.nan
        out[eta] = {
            "bias_ref": float(bias_eta[REF_IDX]),
            "bse_ref": float(bse_eta[REF_IDX]),
            "cov_ref": cov_eta,
            "ser_ref": ser,
            "se_ref": float(SE_eta[REF_IDX]),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B"], default="A")
    parser.add_argument("--top", type=int, default=3,
                        help="number of top configs from stage to analyze")
    args = parser.parse_args()

    raw_dir = _RAW_BASE / f"stage{args.stage}"
    if not raw_dir.exists():
        print(f"Stage {args.stage} raw dir not found: {raw_dir}")
        return

    # Find PKLs at n=1000, M=50 for the stage
    pkls = sorted(raw_dir.glob("*_n1000_M50.pkl"))
    print(f"Found {len(pkls)} PKLs in stage {args.stage}")

    # First pass: rank by |b|/SE at REF_IDX
    rankings = []
    for pkl in pkls:
        cid = pkl.stem.replace("_n1000_M50", "")
        if cid in ("REG_Hsuper", "DRK_Hsuper"):
            continue
        if not (cid.startswith("RFF") or cid.startswith("STAB")):
            continue
        data = _load_records(pkl)
        if data is None:
            continue
        records = [r for r in data["records"] if r.get("error") is None]
        if not records:
            continue
        J_pt = np.array(data["meta"]["J_policy_true"])
        n = int(data["meta"]["n"])
        J_dr = np.array([r["J_dr"] for r in records])
        V_dr = np.array([r["V_hat_grid"] for r in records])
        bias = float(np.mean(J_dr[:, REF_IDX])) - J_pt[REF_IDX]
        SE = float(np.sqrt(np.mean(V_dr[:, REF_IDX]) / n))
        bse = abs(bias) / SE if SE > 0 else np.inf
        rankings.append((cid, bse, pkl, J_pt, n, records))
    rankings.sort(key=lambda t: t[1])
    top_configs = rankings[:args.top]
    print(f"Top {len(top_configs)} configs by |b|/SE:")
    for cid, bse, *_ in top_configs:
        print(f"  {cid}: |b|/SE = {bse:.4f}")
    print()

    # Damping analysis
    lines = [
        "# Phase 12B Bennett-lite RFF -- Stage C (damping diagnostic, post-hoc)",
        "",
        f"Source stage: {args.stage}, top {len(top_configs)} configs",
        "",
        ("Damping: J_eta = J_reg + eta * (J_dr - J_reg). "
         "Pure diagnostic -- no new fits."),
        "",
    ]
    for cid, bse, pkl, J_pt, n, records in top_configs:
        diag = _eta_diagnostic(records, J_pt, n)
        lines.append(f"## {cid}")
        lines.append("")
        lines.append("| eta | bias_ref | |b|/SE | cov | SER |")
        lines.append("|---|---|---|---|---|")
        for eta in ETAS:
            d = diag[eta]
            if d is None:
                lines.append(f"| {eta} | --- | --- | --- | --- |")
            else:
                lines.append(
                    f"| {eta} | {d['bias_ref']:+.4f} | {d['bse_ref']:.4f} | "
                    f"{d['cov_ref']:.3f} | {d['ser_ref']:.3f} |"
                )
        lines.append("")
    out = "\n".join(lines)
    out_path = _SUMM_DIR / "phase12B_bennett_rff_stageC.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"Stage C report: {out_path}")
    print()
    for line in out.split("\n")[:60]:
        print(line)


if __name__ == "__main__":
    main()
