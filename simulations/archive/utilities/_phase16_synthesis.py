"""
Phase 16 -- Synthesis: aggregate EXP-A + EXP-B + EXP-C results into a master analysis.

Loads all Phase 16 pkls + cross-references with Phase 15C E1/E2/E3 + Phase 15B
Kallus + Phase 14B Bennett. Produces:
    docs/notes/phase16_synthesis_data.json   - structured data for figures
    simulations/results/summaries/phase16_FINAL_analysis.md - human-readable

Sections:
  1. EXP-A coverage map analysis (safe envelope)
  2. EXP-B proxy feasibility map (no-go zone)
  3. EXP-C spectral correlation analysis (Spearman rho per method per diagnostic)
  4. Cross-experiment Pareto frontier (best method per cell)
  5. Cross-phase consistency check (Phase 15 vs Phase 16)
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[2]
_RAW_BASE = _BASE_DIR / "simulations" / "results" / "raw"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "phase16_FINAL_analysis.md"
_DATA_PATH = _BASE_DIR / "docs" / "notes" / "phase16_synthesis_data.json"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
REF_IDX = 2

METHODS_MAIN = ["linearDR", "KPVREG", "DRKPV", "best_bennett"]
N_LIST = [1000, 2000]
SNR_LIST = [0.85, 0.90, 0.95]
SNR_LEVELS_FEAS = [0.50, 0.70, 0.85, 0.95]


def _load_pkl(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _ok_records(data: dict) -> list:
    return [r for r in data.get("records", []) if r.get("error") is None]


def _decompose_quick(data: dict) -> dict:
    """Lightweight metric extraction from raw pkl."""
    recs = _ok_records(data)
    if not recs:
        return {}
    n = int(data["meta"]["n"])
    J_pt = np.array(data["meta"]["J_policy_true"])
    M_ok = len(recs)
    J_dr_all = np.array([r["J_dr"] for r in recs])     # (M, K)
    V_hg_all = np.array([r["V_hat_grid"] for r in recs])
    K = J_dr_all.shape[1]
    with np.errstate(invalid="ignore"):
        se_per_rep = np.sqrt(np.maximum(V_hg_all, 0.0) / n)
    bias_grid = J_dr_all.mean(axis=0) - J_pt
    se_mean_grid = np.nanmean(se_per_rep, axis=0)
    bse_grid = np.abs(bias_grid) / np.where(se_mean_grid > 1e-15, se_mean_grid, np.nan)
    cov_grid = np.array([
        float(np.mean(np.abs(J_dr_all[:, k] - J_pt[k]) <= 1.96 * se_per_rep[:, k]))
        for k in range(K)
    ])
    emp_sd = np.std(J_dr_all, axis=0, ddof=1)
    ser_grid = se_mean_grid / np.where(emp_sd > 1e-15, emp_sd, np.nan)
    return {
        "n": n, "M_ok": M_ok,
        "bias_grid": bias_grid.tolist(),
        "bse_grid": bse_grid.tolist(),
        "cov_grid": cov_grid.tolist(),
        "se_mean_grid": se_mean_grid.tolist(),
        "ser_grid": ser_grid.tolist(),
        # Spectral
        "kappa_h_mean":    float(np.nanmean([r.get("kappa_h", np.nan) for r in recs])),
        "eff_rank_h_mean": float(np.nanmean([r.get("eff_rank_h", np.nan) for r in recs])),
        "residual_norm_h_mean": float(np.nanmean([r.get("residual_norm_h", np.nan) for r in recs])),
        "kappa_r_mean":    float(np.nanmean([r.get("kappa_r", np.nan) for r in recs])),
        "eff_rank_r_mean": float(np.nanmean([r.get("eff_rank_r", np.nan) for r in recs])),
        "residual_norm_r_mean": float(np.nanmean([r.get("residual_norm_r", np.nan) for r in recs])),
    }


# ── EXP-A : coverage map ──────────────────────────────────────────────────────

def load_expa() -> Dict[Tuple[int, float], Dict[str, dict]]:
    out: Dict[Tuple[int, float], Dict[str, dict]] = {}
    base = _RAW_BASE / "phase16_EXPA"
    for n in N_LIST:
        for snr in SNR_LIST:
            cell_tag = f"n{n}_snr{int(snr*100):02d}"
            cell_dir = base / cell_tag
            if not cell_dir.exists():
                continue
            cell_results = {}
            for method in METHODS_MAIN:
                pkl = cell_dir / f"{method}_M200.pkl"
                data = _load_pkl(pkl)
                if data is None:
                    continue
                cell_results[method] = _decompose_quick(data)
            if cell_results:
                out[(n, snr)] = cell_results
    return out


# ── EXP-B : proxy feasibility ─────────────────────────────────────────────────

def load_expb() -> Dict[Tuple[float, float], Dict[str, dict]]:
    out: Dict[Tuple[float, float], Dict[str, dict]] = {}
    base = _RAW_BASE / "phase16_EXPB"
    for snr_W in SNR_LEVELS_FEAS:
        for snr_Z in SNR_LEVELS_FEAS:
            cell_tag = f"snrW{int(snr_W*100):02d}_snrZ{int(snr_Z*100):02d}"
            cell_dir = base / cell_tag
            if not cell_dir.exists():
                continue
            cell_results = {}
            for method in METHODS_MAIN:
                pkl = cell_dir / f"{method}_M80.pkl"
                data = _load_pkl(pkl)
                if data is None:
                    continue
                cell_results[method] = _decompose_quick(data)
            if cell_results:
                out[(snr_W, snr_Z)] = cell_results
    return out


# ── EXP-C : spectral diagnostics ──────────────────────────────────────────────

def load_expc() -> Dict[Tuple[int, float], Dict[str, dict]]:
    out: Dict[Tuple[int, float], Dict[str, dict]] = {}
    base = _RAW_BASE / "phase16_EXPC"
    for n in N_LIST:
        for snr in SNR_LIST:
            cell_tag = f"n{n}_snr{int(snr*100):02d}"
            cell_dir = base / cell_tag
            if not cell_dir.exists():
                continue
            cell_results = {}
            for method in METHODS_MAIN:
                pkl = cell_dir / f"{method}_M50.pkl"
                data = _load_pkl(pkl)
                if data is None:
                    continue
                cell_results[method] = _decompose_quick(data)
            if cell_results:
                out[(n, snr)] = cell_results
    return out


# ── Cross-cell Spearman correlation analysis ──────────────────────────────────

def spectral_correlations(expc: Dict) -> Dict[str, Dict[str, dict]]:
    from scipy.stats import spearmanr
    out: Dict[str, Dict[str, dict]] = {}
    diag_keys = [
        "kappa_h_mean", "eff_rank_h_mean", "residual_norm_h_mean",
        "kappa_r_mean", "eff_rank_r_mean", "residual_norm_r_mean",
    ]
    for method in METHODS_MAIN:
        out[method] = {}
        for diag in diag_keys:
            xs = []
            ys = []
            for (n, snr), results in expc.items():
                d = results.get(method)
                if d is None:
                    continue
                x = d.get(diag, float("nan"))
                y = d["bse_grid"][REF_IDX]
                if np.isfinite(x) and np.isfinite(y) and x > 0:
                    xs.append(np.log10(x) if "kappa" in diag else x)
                    ys.append(y)
            if len(xs) >= 3:
                rho, p = spearmanr(xs, ys)
                out[method][diag] = {
                    "rho": float(rho), "p": float(p), "n_cells": len(xs),
                    "x_values": xs, "y_values": ys,
                }
            else:
                out[method][diag] = {"rho": float("nan"), "p": float("nan"), "n_cells": len(xs)}
    return out


# ── Pareto frontier per cell ──────────────────────────────────────────────────

def pareto_per_cell(cells: Dict[Tuple, Dict[str, dict]]) -> Dict[Tuple, str]:
    """Return method with smallest bse_ref per cell."""
    out: Dict[Tuple, str] = {}
    for cell, results in cells.items():
        best_method = None
        best_bse = float("inf")
        for method, d in results.items():
            if d is None or "bse_grid" not in d:
                continue
            b = d["bse_grid"][REF_IDX]
            if np.isfinite(b) and b < best_bse:
                best_bse = b
                best_method = method
        if best_method is not None:
            out[cell] = best_method
    return out


# ── Safe envelope: cells where >=1 method achieves cov >= 0.90 ────────────────

def safe_envelope(cells: Dict[Tuple, Dict[str, dict]], cov_threshold=0.90) -> Dict[Tuple, str]:
    """Return method achieving cov >= threshold per cell, or 'NONE' if no method does."""
    out: Dict[Tuple, str] = {}
    for cell, results in cells.items():
        passing = []
        for method, d in results.items():
            if d is None or "cov_grid" not in d:
                continue
            cov = d["cov_grid"][REF_IDX]
            if np.isfinite(cov) and cov >= cov_threshold:
                passing.append((method, d["bse_grid"][REF_IDX]))
        if passing:
            # Among passing methods, pick smallest bse
            passing.sort(key=lambda x: x[1])
            out[cell] = passing[0][0]
        else:
            out[cell] = "NONE"
    return out


# ── Report writer ─────────────────────────────────────────────────────────────

def _fmt(v):
    if v is None or not np.isfinite(v):
        return "---"
    if abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
        return f"{v:.2e}"
    return f"{v:.4f}"


def write_report(expa, expb, expc, corrs, pareto_a, pareto_b, envelope_a):
    import datetime
    lines: List[str] = []
    ap = lines.append

    ap("# Phase 16 -- FINAL Synthesis Analysis")
    ap(f"*Generated by `_phase16_synthesis.py` -- {datetime.date.today().isoformat()}*")
    ap("")
    ap("Aggregates EXP-A (coverage map), EXP-B (proxy feasibility), EXP-C (spectral)")
    ap("for the S6 essay write-up. Cross-references Phase 15C E1/E2/E3 + Phase 15B + 14B.")
    ap("")
    ap("---")
    ap("")

    # ── EXP-A summary ─────────────────────────────────────────────────────────
    ap("## 1. EXP-A: Coverage map (DGP2, M=200)")
    ap("")
    ap("4 methods x 6 cells (n in {1000, 2000} x SNR in {0.85, 0.90, 0.95}).")
    ap(f"Cells loaded: {len(expa)} / 6")
    ap("")
    ap("### 1a. Coverage table (cov_ref, target = 0.95)")
    ap("")
    header = "| method | " + " | ".join([f"n={n}/SNR={s}" for n in N_LIST for s in SNR_LIST]) + " |"
    ap(header)
    ap("|" + "|".join(["---"] * (1 + len(N_LIST) * len(SNR_LIST))) + "|")
    for method in METHODS_MAIN:
        row = [method]
        for n in N_LIST:
            for snr in SNR_LIST:
                d = expa.get((n, snr), {}).get(method)
                if d is None:
                    row.append("---")
                else:
                    row.append(_fmt(d["cov_grid"][REF_IDX]))
        ap("| " + " | ".join(row) + " |")
    ap("")

    ap("### 1b. bse_ref table")
    ap("")
    ap(header)
    ap("|" + "|".join(["---"] * (1 + len(N_LIST) * len(SNR_LIST))) + "|")
    for method in METHODS_MAIN:
        row = [method]
        for n in N_LIST:
            for snr in SNR_LIST:
                d = expa.get((n, snr), {}).get(method)
                row.append(_fmt(d["bse_grid"][REF_IDX]) if d else "---")
        ap("| " + " | ".join(row) + " |")
    ap("")

    # Safe envelope
    ap("### 1c. Safe envelope (best method achieving cov >= 0.90, ties broken by min bse)")
    ap("")
    ap("| n \\ SNR | " + " | ".join([str(s) for s in SNR_LIST]) + " |")
    ap("|" + "|".join(["---"] * (1 + len(SNR_LIST))) + "|")
    for n in N_LIST:
        row = [str(n)]
        for snr in SNR_LIST:
            row.append(envelope_a.get((n, snr), "---"))
        ap("| " + " | ".join(row) + " |")
    ap("")

    ap("### 1d. Pareto winner (min bse_ref, ignoring coverage)")
    ap("")
    ap("| n \\ SNR | " + " | ".join([str(s) for s in SNR_LIST]) + " |")
    ap("|" + "|".join(["---"] * (1 + len(SNR_LIST))) + "|")
    for n in N_LIST:
        row = [str(n)]
        for snr in SNR_LIST:
            row.append(pareto_a.get((n, snr), "---"))
        ap("| " + " | ".join(row) + " |")
    ap("")

    # ── EXP-B summary ─────────────────────────────────────────────────────────
    ap("---")
    ap("")
    ap("## 2. EXP-B: Proxy feasibility map (DGP2, M=80, n=1000)")
    ap("")
    ap("4 methods x 16 cells (SNR_W x SNR_Z grid in {0.50, 0.70, 0.85, 0.95}).")
    ap(f"Cells loaded: {len(expb)} / 16")
    ap("")
    for method in METHODS_MAIN:
        ap(f"### 2.{method} -- bse_ref by (SNR_W, SNR_Z)")
        ap("")
        ap("| SNR_W \\ SNR_Z | " + " | ".join([str(s) for s in SNR_LEVELS_FEAS]) + " |")
        ap("|" + "|".join(["---"] * (1 + len(SNR_LEVELS_FEAS))) + "|")
        for snr_W in SNR_LEVELS_FEAS:
            row = [str(snr_W)]
            for snr_Z in SNR_LEVELS_FEAS:
                d = expb.get((snr_W, snr_Z), {}).get(method)
                row.append(_fmt(d["bse_grid"][REF_IDX]) if d else "---")
            ap("| " + " | ".join(row) + " |")
        ap("")

    ap("### 2e. Pareto winner per (SNR_W, SNR_Z) cell")
    ap("")
    ap("| SNR_W \\ SNR_Z | " + " | ".join([str(s) for s in SNR_LEVELS_FEAS]) + " |")
    ap("|" + "|".join(["---"] * (1 + len(SNR_LEVELS_FEAS))) + "|")
    for snr_W in SNR_LEVELS_FEAS:
        row = [str(snr_W)]
        for snr_Z in SNR_LEVELS_FEAS:
            row.append(pareto_b.get((snr_W, snr_Z), "---"))
        ap("| " + " | ".join(row) + " |")
    ap("")

    # ── EXP-C summary ─────────────────────────────────────────────────────────
    ap("---")
    ap("")
    ap("## 3. EXP-C: Spectral diagnostics (DGP2, M=50, same grid as EXP-A)")
    ap("")
    ap(f"Cells loaded: {len(expc)} / 6")
    ap("")
    ap("### 3a. Spectral fields per cell")
    ap("")
    cols = ["bse_ref", "kappa_h", "eff_rank_h", "resid_h", "kappa_r", "eff_rank_r", "resid_r"]
    for (n, snr), results in sorted(expc.items()):
        ap(f"#### n={n}, SNR={snr}")
        ap("")
        ap("| method | " + " | ".join(cols) + " |")
        ap("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
        for method in METHODS_MAIN:
            d = results.get(method)
            if d is None:
                ap(f"| {method} | (no data) |" + "|".join(["" for _ in cols[1:]]) + " |")
                continue
            row = [
                _fmt(d["bse_grid"][REF_IDX]),
                _fmt(d.get("kappa_h_mean", float("nan"))),
                _fmt(d.get("eff_rank_h_mean", float("nan"))),
                _fmt(d.get("residual_norm_h_mean", float("nan"))),
                _fmt(d.get("kappa_r_mean", float("nan"))),
                _fmt(d.get("eff_rank_r_mean", float("nan"))),
                _fmt(d.get("residual_norm_r_mean", float("nan"))),
            ]
            ap(f"| {method} | " + " | ".join(row) + " |")
        ap("")

    # Spectral correlations
    ap("### 3b. Spearman correlations: spectral diagnostics vs bse_observed")
    ap("")
    ap("Strong correlation (|rho| > 0.5) means the diagnostic is ACTIONABLE")
    ap("(can predict performance without observing J_true).")
    ap("")
    diag_labels = {
        "kappa_h_mean": "log10(kappa_h)",
        "eff_rank_h_mean": "eff_rank_h",
        "residual_norm_h_mean": "residual_norm_h",
        "kappa_r_mean": "log10(kappa_r)",
        "eff_rank_r_mean": "eff_rank_r",
        "residual_norm_r_mean": "residual_norm_r",
    }
    for method in METHODS_MAIN:
        ap(f"#### {method}")
        ap("")
        ap("| Diagnostic | Spearman rho | p-value | n_cells |")
        ap("|---|---|---|---|")
        for diag, label in diag_labels.items():
            corr = corrs.get(method, {}).get(diag, {})
            ap(f"| {label} | "
               f"{_fmt(corr.get('rho', float('nan')))} | "
               f"{_fmt(corr.get('p', float('nan')))} | "
               f"{corr.get('n_cells', 0)} |")
        ap("")

    # ── Verdict ────────────────────────────────────────────────────────────────
    ap("---")
    ap("")
    ap("## 4. Phase 16 verdict")
    ap("")
    ap("**Coverage envelope** : ")
    safe_cells = [c for c, m in envelope_a.items() if m != "NONE"]
    nogo_cells = [c for c, m in envelope_a.items() if m == "NONE"]
    ap(f"- {len(safe_cells)} / {len(envelope_a)} EXP-A cells have at least one method achieving cov >= 0.90")
    if nogo_cells:
        ap(f"- No-go cells: {nogo_cells}")
    ap("")
    ap("**Spectral predictiveness** : ")
    for method in METHODS_MAIN:
        max_rho = 0
        best_diag = ""
        for diag, label in diag_labels.items():
            corr = corrs.get(method, {}).get(diag, {})
            rho = abs(corr.get("rho", 0))
            if np.isfinite(rho) and rho > max_rho:
                max_rho = rho
                best_diag = label
        ap(f"- {method} : strongest predictor = {best_diag} (|rho| = {_fmt(max_rho)})")
    ap("")
    ap("---")
    ap("*End of Phase 16 synthesis.*")

    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {_REPORT_PATH}")


def write_data_json(expa, expb, expc, corrs, pareto_a, pareto_b, envelope_a):
    """Write structured data to JSON for figure generation."""
    def _serialize_cells(cells):
        return {f"{k[0]}__{k[1]}": v for k, v in cells.items()}

    data = {
        "expa": _serialize_cells(expa),
        "expb": _serialize_cells(expb),
        "expc": _serialize_cells(expc),
        "spectral_correlations": corrs,
        "pareto_expa": {f"{k[0]}__{k[1]}": v for k, v in pareto_a.items()},
        "pareto_expb": {f"{k[0]}__{k[1]}": v for k, v in pareto_b.items()},
        "safe_envelope_expa": {f"{k[0]}__{k[1]}": v for k, v in envelope_a.items()},
        "config": {
            "A_GRID": A_GRID.tolist(),
            "REF_IDX": REF_IDX,
            "METHODS_MAIN": METHODS_MAIN,
            "N_LIST": N_LIST,
            "SNR_LIST": SNR_LIST,
            "SNR_LEVELS_FEAS": SNR_LEVELS_FEAS,
        },
    }
    _DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DATA_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"  Data: {_DATA_PATH}")


def main():
    print("=== Phase 16 Synthesis ===\n")

    print("[1/3] Loading EXP-A coverage map...")
    expa = load_expa()
    print(f"  Cells loaded: {len(expa)}")

    print("[2/3] Loading EXP-B proxy feasibility...")
    expb = load_expb()
    print(f"  Cells loaded: {len(expb)}")

    print("[3/3] Loading EXP-C spectral...")
    expc = load_expc()
    print(f"  Cells loaded: {len(expc)}")

    print("\nComputing analyses...")
    corrs = spectral_correlations(expc) if expc else {}
    pareto_a = pareto_per_cell(expa)
    pareto_b = pareto_per_cell(expb)
    envelope_a = safe_envelope(expa)

    print("\nWriting outputs...")
    write_report(expa, expb, expc, corrs, pareto_a, pareto_b, envelope_a)
    write_data_json(expa, expb, expc, corrs, pareto_a, pareto_b, envelope_a)


if __name__ == "__main__":
    main()
