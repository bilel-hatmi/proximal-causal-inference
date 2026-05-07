"""
Audit empirical des métriques pour le blind tuning.

Pour chaque méthode (linearDR, KPVREG, DRKPV, best_bennett) et chaque cellule
de Phase 16 EXP-A (M=200) + EXP-C (spectral M=50), on extrait :
  - bse_ref (= |bias|/SE) -- le critère "vraie qualité" (J_true requis)
  - coverage_ref -- aussi non-aveugle
  - SER (= SE_pred / SD_emp)
  - RR (Riesz residual)
  - ESS_min, weight_p99
  - kappa_h, eff_rank_h, residual_norm_h
  - kappa_r, eff_rank_r, residual_norm_r
  - cond_M_grid_mean (DRKPV)
  - h_weighted_residual_mean (Bennett)

Puis on calcule la corrélation Spearman entre chaque métrique aveugle et
bse_ref / coverage à travers les cellules, par méthode.

Output : docs/notes/phase16_metric_audit.md (une analyse par méthode)
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

_BASE_DIR = Path(__file__).resolve().parents[2]
_RAW_BASE = _BASE_DIR / "simulations" / "results" / "raw"
_OUT_PATH = _BASE_DIR / "docs" / "notes" / "phase16_metric_audit.md"

METHODS = ["linearDR", "KPVREG", "DRKPV", "best_bennett"]
N_LIST = [1000, 2000]
SNR_LIST = [0.85, 0.90, 0.95]
SNR_LEVELS_FEAS = [0.50, 0.70, 0.85, 0.95]  # EXP-B grid
REF_IDX = 2


def _load_pkl(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _ok(data: dict) -> list:
    return [r for r in data.get("records", []) if r.get("error") is None]


def _safe(v, default=float("nan")):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _array_safe(arr, idx):
    try:
        return float(arr[idx])
    except Exception:
        return float("nan")


def _extract_metrics(data: dict, method: str) -> dict:
    """Extract per-cell metrics from a pkl: aggregates over reps."""
    recs = _ok(data)
    if not recs:
        return {}
    M_ok = len(recs)
    n = int(data["meta"]["n"])
    J_pt = np.array(data["meta"]["J_policy_true"])

    # bse_ref + coverage_ref + SER
    J_dr = np.array([r["J_dr"] for r in recs])  # (M, K)
    V_hg = np.array([r["V_hat_grid"] for r in recs])  # (M, K)
    with np.errstate(invalid="ignore"):
        se_per = np.sqrt(np.maximum(V_hg, 0.0) / n)
    bias_ref = float(np.mean(J_dr[:, REF_IDX]) - J_pt[REF_IDX])
    SE_pred = float(np.nanmean(se_per[:, REF_IDX]))
    SD_emp = float(np.std(J_dr[:, REF_IDX], ddof=1))
    bse_ref = abs(bias_ref) / SE_pred if SE_pred > 1e-12 else float("nan")
    cov_ref = float(np.mean(np.abs(J_dr[:, REF_IDX] - J_pt[REF_IDX]) <= 1.96 * se_per[:, REF_IDX]))
    SER = SE_pred / SD_emp if SD_emp > 1e-12 else float("nan")

    # ESS, weight diagnostics
    ESS_min = float(np.nanmean([_safe(r.get("ESS_min")) for r in recs]))
    w_p99_ref = float(np.nanmean([_array_safe(r.get("weight_p99_grid", []), REF_IDX) for r in recs]))
    w_max_ref = float(np.nanmean([_array_safe(r.get("weight_max_grid", []), REF_IDX) for r in recs]))

    # Riesz residual
    RR_ref = float(np.nanmean([_array_safe(r.get("riesz_residual_grid_mean", []), REF_IDX) for r in recs]))

    # Spectral fields (Phase 16 new)
    kappa_h = float(np.nanmean([_safe(r.get("kappa_h")) for r in recs]))
    kappa_r = float(np.nanmean([_safe(r.get("kappa_r")) for r in recs]))
    eff_rank_h = float(np.nanmean([_safe(r.get("eff_rank_h")) for r in recs]))
    eff_rank_r = float(np.nanmean([_safe(r.get("eff_rank_r")) for r in recs]))
    residual_norm_h = float(np.nanmean([_safe(r.get("residual_norm_h")) for r in recs]))
    residual_norm_r = float(np.nanmean([_safe(r.get("residual_norm_r")) for r in recs]))

    # Method-specific
    if method == "best_bennett":
        # h_cond_G_mean is from Bennett-indep
        h_weighted = float(np.nanmean([_safe(r.get("residual_norm_h"), default=float("nan")) for r in recs]))
    cond_M = float(np.nanmean([_array_safe(r.get("cond_M_grid_mean", []), REF_IDX) for r in recs]))

    return {
        "M_ok": M_ok, "n": n,
        "bse_ref": bse_ref, "cov_ref": cov_ref, "SER": SER,
        "ESS_min": ESS_min, "ESS_min_ratio": ESS_min / n,
        "w_p99_ref": w_p99_ref, "w_max_ref": w_max_ref,
        "RR_ref": RR_ref,
        "kappa_h": kappa_h, "kappa_r": kappa_r,
        "eff_rank_h": eff_rank_h, "eff_rank_r": eff_rank_r,
        "residual_norm_h": residual_norm_h, "residual_norm_r": residual_norm_r,
        "cond_M": cond_M,
    }


def load_all_metrics() -> Dict[str, Dict[Any, Dict]]:
    """Load Phase 16 EXP-A (M=200) + EXP-B (M=80) + EXP-C (M=50, spectral)."""
    out: Dict[str, Dict[Any, Dict]] = {m: {} for m in METHODS}

    # EXP-A: 6 cells (n × SNR), symmetric SNR_W = SNR_Z
    for method in METHODS:
        for n in N_LIST:
            for snr in SNR_LIST:
                cell_tag = f"n{n}_snr{int(snr * 100):02d}"
                pkl_a = _RAW_BASE / "phase16_EXPA" / cell_tag / f"{method}_M200.pkl"
                data_a = _load_pkl(pkl_a)
                if data_a is None:
                    continue
                metrics_a = _extract_metrics(data_a, method)

                # EXP-C spectral
                pkl_c = _RAW_BASE / "phase16_EXPC" / cell_tag / f"{method}_M50.pkl"
                data_c = _load_pkl(pkl_c)
                metrics_c = _extract_metrics(data_c, method) if data_c else {}

                merged = dict(metrics_a)
                for k in ["kappa_h", "kappa_r", "eff_rank_h", "eff_rank_r",
                           "residual_norm_h", "residual_norm_r", "cond_M"]:
                    if k in metrics_c and np.isfinite(metrics_c[k]):
                        merged[k] = metrics_c[k]
                    elif np.isnan(merged.get(k, np.nan)):
                        merged[k] = float("nan")

                merged["source"] = "EXP-A"
                merged["snr_W"] = snr
                merged["snr_Z"] = snr
                out[method][("EXP-A", n, snr, snr)] = merged

    # EXP-B: 16 cells (SNR_W × SNR_Z), n=1000 fixed
    n = 1000
    for method in METHODS:
        for snr_W in SNR_LEVELS_FEAS:
            for snr_Z in SNR_LEVELS_FEAS:
                cell_tag = f"snrW{int(snr_W * 100):02d}_snrZ{int(snr_Z * 100):02d}"
                pkl_b = _RAW_BASE / "phase16_EXPB" / cell_tag / f"{method}_M80.pkl"
                data_b = _load_pkl(pkl_b)
                if data_b is None:
                    continue
                merged = _extract_metrics(data_b, method)
                merged["source"] = "EXP-B"
                merged["snr_W"] = snr_W
                merged["snr_Z"] = snr_Z
                out[method][("EXP-B", n, snr_W, snr_Z)] = merged

    return out


def correlation_analysis(metrics_per_method: Dict) -> Dict[str, Dict[str, dict]]:
    """For each method, compute Spearman rho(metric, bse_ref) across cells."""
    targets = ["bse_ref", "cov_ref"]
    blind_metrics = ["RR_ref", "ESS_min_ratio", "w_p99_ref", "w_max_ref",
                      "kappa_h", "kappa_r", "eff_rank_h", "eff_rank_r",
                      "residual_norm_h", "residual_norm_r", "cond_M"]
    pseudo_blind = ["SER"]

    out = {}
    for method, cells in metrics_per_method.items():
        if not cells:
            continue
        cell_data = list(cells.values())
        out[method] = {}
        for target in targets:
            ys = [d.get(target, float("nan")) for d in cell_data]
            for metric in blind_metrics + pseudo_blind:
                xs = [d.get(metric, float("nan")) for d in cell_data]
                pairs = [(x, y) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
                if len(pairs) < 3:
                    out[method][f"{metric}__{target}"] = {"rho": float("nan"), "p": float("nan"), "n": len(pairs)}
                    continue
                xv, yv = zip(*pairs)
                # Use log10 for highly skewed metrics (kappa, ESS, weight)
                if metric in ["kappa_h", "kappa_r", "ESS_min_ratio", "w_p99_ref", "w_max_ref",
                                "cond_M", "RR_ref"]:
                    xv = [np.log10(abs(x) + 1e-30) for x in xv]
                rho, p = spearmanr(xv, yv)
                out[method][f"{metric}__{target}"] = {
                    "rho": float(rho), "p": float(p), "n": len(pairs),
                    "metric_range": [float(min(xs)), float(max(xs))] if xs else [None, None],
                }
    return out


def write_report(metrics_per_method, correlations) -> None:
    lines = []
    ap = lines.append
    import datetime

    ap("# Phase 16 — Empirical Audit of Metrics for Blind Tuning")
    ap(f"*Generated by `_phase16_metric_audit.py` — {datetime.date.today().isoformat()}*")
    ap("")
    ap("Cross-cell Spearman correlations between candidate blind metrics and")
    ap("the gold-standard non-blind metric `bse_ref` (= |bias|/SE) and `cov_ref`.")
    ap("Computed from Phase 16 EXP-A (M=200, 6 cells) + EXP-C (M=50, spectral, 6 cells).")
    ap("")
    ap("**Interpretation key**:")
    ap("- |rho| > 0.6 : metric is **actionable** (strongly predicts bse_ref)")
    ap("- 0.3 < |rho| < 0.6 : metric is **suggestive** (use with caution)")
    ap("- |rho| < 0.3 : metric is **not predictive** for this method")
    ap("- Sign of rho with bse_ref: positive = larger metric → larger bias problem")
    ap("- Sign of rho with cov_ref: positive = larger metric → better coverage (probably wrong direction!)")
    ap("")
    ap("---")
    ap("")

    # Per-method analysis
    for method, cells in metrics_per_method.items():
        if not cells:
            continue
        ap(f"## {method}")
        ap("")
        ap(f"Cells loaded: {len(cells)} / {len(N_LIST) * len(SNR_LIST)}")
        ap("")
        ap("### Cell-level metric snapshot (EXP-A only — symmetric SNR)")
        ap("")
        cols = ["bse_ref", "cov_ref", "SER", "RR_ref", "ESS_min_ratio",
                 "kappa_h", "kappa_r", "residual_norm_h"]
        ap("| (src, n, SNR_W, SNR_Z) | " + " | ".join(cols) + " |")
        ap("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
        for key, d in sorted(cells.items()):
            if d.get("source") != "EXP-A":
                continue
            src, n, snr_W, snr_Z = key
            row = [f"({src}, {n}, {snr_W}, {snr_Z})"]
            for c in cols:
                v = d.get(c, float("nan"))
                if not np.isfinite(v):
                    row.append("---")
                elif abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
                    row.append(f"{v:.2e}")
                else:
                    row.append(f"{v:.4f}")
            ap("| " + " | ".join(row) + " |")
        ap("")

        ap("### Spearman correlations (across cells, n=6)")
        ap("")
        ap("Strongly significant (|rho| > 0.6) shown in bold.")
        ap("")
        ap("| Metric | rho(metric, bse_ref) | p | rho(metric, cov_ref) | p |")
        ap("|---|---|---|---|---|")
        corrs = correlations.get(method, {})
        metric_list = ["RR_ref", "SER", "ESS_min_ratio", "w_p99_ref", "w_max_ref",
                        "kappa_h", "kappa_r", "eff_rank_h", "eff_rank_r",
                        "residual_norm_h", "residual_norm_r", "cond_M"]
        for metric in metric_list:
            rb = corrs.get(f"{metric}__bse_ref", {})
            rc = corrs.get(f"{metric}__cov_ref", {})
            rho_b = rb.get("rho", float("nan"))
            p_b = rb.get("p", float("nan"))
            rho_c = rc.get("rho", float("nan"))
            p_c = rc.get("p", float("nan"))
            label = metric
            if np.isfinite(rho_b) and abs(rho_b) > 0.6:
                rho_b_str = f"**{rho_b:.3f}**"
            elif np.isfinite(rho_b):
                rho_b_str = f"{rho_b:.3f}"
            else:
                rho_b_str = "---"
            if np.isfinite(rho_c) and abs(rho_c) > 0.6:
                rho_c_str = f"**{rho_c:.3f}**"
            elif np.isfinite(rho_c):
                rho_c_str = f"{rho_c:.3f}"
            else:
                rho_c_str = "---"
            p_b_str = f"{p_b:.3f}" if np.isfinite(p_b) else "---"
            p_c_str = f"{p_c:.3f}" if np.isfinite(p_c) else "---"
            ap(f"| {label} | {rho_b_str} | {p_b_str} | {rho_c_str} | {p_c_str} |")
        ap("")

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {_OUT_PATH}")


def main():
    print("=== Phase 16 Metric Audit ===")
    print("[1/3] Loading metrics from EXP-A + EXP-C...")
    metrics = load_all_metrics()
    for m, cells in metrics.items():
        print(f"  {m}: {len(cells)} cells")

    print("[2/3] Computing Spearman correlations...")
    corrs = correlation_analysis(metrics)

    print("[3/3] Writing report...")
    write_report(metrics, corrs)


if __name__ == "__main__":
    main()
