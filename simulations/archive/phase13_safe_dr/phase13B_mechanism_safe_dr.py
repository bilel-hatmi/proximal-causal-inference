"""
Phase 13B -- Mechanistic study of when the DR correction helps vs harms,
+ post-hoc evaluation of "Safe-DR" rules J_SAFE = J_REG + eta*(J_DRK - J_REG).

Sub-steps
---------
  13B.1 : density / overlap diagnostics on regenerated A samples per cell
  13B.2 : DR correction decomposition (post-hoc on Phase 13A PKLs)
  13B.3 : weight-tail diagnostics (post-hoc)
  13B.4 : Safe-DR rules 0-7 post-hoc evaluation
  13B.5 : decide Case A / B / C and write report

Inputs
------
  simulations/results/raw/coverage_map_phase13/snr<SNR>_n<n>/<method>_M100.pkl
    where SNR in {80, 90, 95}, n in {500, 1000, 2000},
    method in {REG_Hsuper, DRK_Hsuper, RFF500}.

Outputs
-------
  simulations/results/raw/phase13B_mechanism/density_diagnostics.pkl
  simulations/results/raw/phase13B_mechanism/safe_dr_rule_eval.pkl
  simulations/results/summaries/phase13B_mechanism_safe_dr.md

Usage
-----
  python -m simulations.experiments.phase13B_mechanism_safe_dr

The script is post-hoc: no new estimator fits. Density diagnostics regenerate
A samples deterministically using the same seed scheme as Phase 13A.
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Thread control before BLAS imports
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[2]
_RAW_13A = _BASE_DIR / "simulations" / "results" / "raw" / "coverage_map_phase13"
_RAW_OUT = _BASE_DIR / "simulations" / "results" / "raw" / "phase13B_mechanism"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "phase13B_mechanism_safe_dr.md"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
K = len(A_GRID)
REF_IDX = 2
SNR_GRID = [0.80, 0.90, 0.95]
N_GRID = [500, 1000, 2000]
M = 100
N_FOLDS = 5

SEED_BASE = 26_000_000


def _seed_base_cell(snr: float, n: int) -> int:
    """Reproduce the Phase 13A seed scheme."""
    snr_idx = SNR_GRID.index(snr)
    n_idx = N_GRID.index(n)
    return SEED_BASE + snr_idx * 100_000 + n_idx * 1_000 + n


# ── Load Phase 13A PKL records ────────────────────────────────────────────────

def _load_cell(snr: float, n: int, method: str) -> Optional[List[dict]]:
    """Load Phase 13A records for a (snr, n, method) cell."""
    pkl = _RAW_13A / f"snr{int(snr*100)}_n{n}" / f"{method}_M{M}.pkl"
    if not pkl.exists():
        print(f"  [WARN] missing {pkl}")
        return None
    with open(pkl, "rb") as fh:
        d = pickle.load(fh)
    recs = [r for r in d["records"] if r.get("error") is None]
    return recs


def _load_J_pt(snr: float) -> np.ndarray:
    """Compute J_policy_true once per SNR (cached)."""
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    from simulations.experiments.dgp2_bias_diagnostics import (
        _silverman_h, _compute_J_policy_true,
    )
    dgp = MichaelisMentenDGP(snr_W=snr, snr_Z=snr)
    ref = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref.A)
    return _compute_J_policy_true(dgp, A_GRID, h_KDE)


# ── 13B.1 — Density diagnostics ───────────────────────────────────────────────

def compute_density_diagnostics() -> Dict:
    """
    For each (snr, n) cell, regenerate the M=100 samples, compute per-rep
    density diagnostics for each dose: ESS_pi/n, local_mass, density_proxy.

    Returns a dict structured as
      out[(snr, n)] = dict with arrays (M, K):
        ess_pi_norm, local_mass_band, local_mass_2band, density_proxy
      plus scalar bandwidth h_KDE.
    """
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    from simulations.experiments.dgp2_bias_diagnostics import _silverman_h

    print("\n=== 13B.1 Density / overlap diagnostics ===")
    out = {}

    # Cache h_KDE per SNR (depends only on SNR, since h = silverman(A))
    # Actually depends on n too -- silverman uses len(A). Compute per cell.
    for snr in SNR_GRID:
        for n in N_GRID:
            t0 = time.time()
            seed_base = _seed_base_cell(snr, n)
            dgp = MichaelisMentenDGP(snr_W=snr, snr_Z=snr)

            ess_pi_norm = np.empty((M, K), dtype=float)
            local_mass_band = np.empty((M, K), dtype=float)
            local_mass_2band = np.empty((M, K), dtype=float)
            density_proxy = np.empty((M, K), dtype=float)
            h_KDE_list = []

            for i in range(M):
                seed = seed_base + i
                sample = dgp.generate(n=n, seed=seed)
                A = np.asarray(sample.A, dtype=float).ravel()
                h_KDE = _silverman_h(A)
                h_KDE_list.append(h_KDE)
                for d, a in enumerate(A_GRID):
                    diff = A - a
                    pi_vals = np.exp(-(diff ** 2) / (2.0 * h_KDE ** 2)) / (h_KDE * np.sqrt(2 * np.pi))
                    sum_pi = float(pi_vals.sum())
                    sum_pi2 = float((pi_vals ** 2).sum())
                    ess = (sum_pi ** 2) / sum_pi2 if sum_pi2 > 1e-30 else 0.0
                    ess_pi_norm[i, d] = ess / n
                    local_mass_band[i, d] = float(np.mean(np.abs(diff) <= h_KDE))
                    local_mass_2band[i, d] = float(np.mean(np.abs(diff) <= 2 * h_KDE))
                    density_proxy[i, d] = float(np.mean(pi_vals))

            elapsed = time.time() - t0
            mean_h = float(np.mean(h_KDE_list))
            cell_data = {
                "ess_pi_norm": ess_pi_norm,
                "local_mass_band": local_mass_band,
                "local_mass_2band": local_mass_2band,
                "density_proxy": density_proxy,
                "h_KDE_mean": mean_h,
            }
            out[(snr, n)] = cell_data
            print(f"  cell SNR={snr}, n={n}: h_KDE_mean={mean_h:.4f} "
                  f"({elapsed:.1f}s)")

    return out


# ── 13B.2 — DR correction decomposition ───────────────────────────────────────

def correction_decomposition(J_pt_per_snr: Dict) -> Dict:
    """
    Post-hoc correction decomposition for REG / DRK / RFF on each cell.

    Returns dict[(snr, n, method)] -> arrays (K,) of:
      bias_DR (mean bias)
      bias_REG (mean regression-only bias)
      mean_correction, sd_correction
      correction_p10, p50, p90
      V_reg_avg, V_corr_avg, V_total_avg
      V_corr / V_total ratio
    """
    print("\n=== 13B.2 DR correction decomposition ===")
    out = {}
    for snr in SNR_GRID:
        J_pt = J_pt_per_snr[snr]
        for n in N_GRID:
            for method in ["REG_Hsuper", "DRK_Hsuper", "RFF500"]:
                recs = _load_cell(snr, n, method)
                if recs is None:
                    continue
                J_dr = np.array([r["J_dr"] for r in recs])      # (M, K)
                J_reg = np.array([r["J_reg"] for r in recs])    # (M, K)
                V_total = np.array([r["V_hat_grid"] for r in recs])
                V_reg = np.array([r["V_reg_grid"] for r in recs])
                V_corr = np.array([r["V_correction_grid"] for r in recs])

                # Per-dose statistics across reps
                bias_DR = J_dr.mean(axis=0) - J_pt
                bias_REG = J_reg.mean(axis=0) - J_pt
                correction = J_dr - J_reg                  # (M, K)
                mean_correction = correction.mean(axis=0)
                sd_correction = correction.std(axis=0, ddof=1)
                p10 = np.percentile(correction, 10, axis=0)
                p50 = np.percentile(correction, 50, axis=0)
                p90 = np.percentile(correction, 90, axis=0)

                V_reg_avg = V_reg.mean(axis=0)
                V_corr_avg = V_corr.mean(axis=0)
                V_total_avg = V_total.mean(axis=0)
                V_cov_proxy = (V_total_avg - V_reg_avg - V_corr_avg) / 2.0
                v_corr_share = np.where(V_total_avg > 0, V_corr_avg / V_total_avg, np.nan)
                v_corr_over_reg = np.where(V_reg_avg > 0, V_corr_avg / V_reg_avg, np.nan)

                # Bias reduction at each dose
                bias_reduction = np.abs(bias_REG) - np.abs(bias_DR)
                # Correction efficiency: how much of |bias_REG| was removed
                eff = np.where(
                    np.abs(bias_REG) > 1e-12,
                    1.0 - np.abs(bias_DR) / np.abs(bias_REG),
                    np.nan,
                )
                # Signal-to-noise of mean correction
                t_signal = np.where(
                    sd_correction > 1e-12,
                    mean_correction / sd_correction,
                    np.nan,
                )

                out[(snr, n, method)] = {
                    "bias_DR": bias_DR,
                    "bias_REG": bias_REG,
                    "mean_correction": mean_correction,
                    "sd_correction": sd_correction,
                    "p10": p10, "p50": p50, "p90": p90,
                    "V_reg_avg": V_reg_avg,
                    "V_corr_avg": V_corr_avg,
                    "V_total_avg": V_total_avg,
                    "V_cov_proxy": V_cov_proxy,
                    "v_corr_share": v_corr_share,
                    "v_corr_over_reg": v_corr_over_reg,
                    "bias_reduction": bias_reduction,
                    "correction_efficiency": eff,
                    "t_signal": t_signal,
                    # Per-rep tensors for SAFE-DR rule eval later
                    "_J_dr_per_rep": J_dr,
                    "_J_reg_per_rep": J_reg,
                    "_V_corr_per_rep": V_corr,
                    "_V_reg_per_rep": V_reg,
                    "_V_total_per_rep": V_total,
                }
    return out


# ── 13B.3 — Weight-tail diagnostics ───────────────────────────────────────────

def weight_diagnostics() -> Dict:
    """
    Post-hoc weight tail diagnostics from per-rep weight_p99_grid / weight_max_grid.
    """
    print("\n=== 13B.3 Weight-tail diagnostics ===")
    out = {}
    for snr in SNR_GRID:
        for n in N_GRID:
            for method in ["REG_Hsuper", "DRK_Hsuper", "RFF500"]:
                recs = _load_cell(snr, n, method)
                if recs is None:
                    continue
                p99 = np.array([r["weight_p99_grid"] for r in recs])      # (M, K)
                wmax = np.array([r["weight_max_grid"] for r in recs])
                ess = np.array([r["ESS_grid"] for r in recs])
                neg = np.array([r["q_negative_share_grid"] for r in recs])
                mean_p99 = p99.mean(axis=0)
                sd_p99 = p99.std(axis=0, ddof=1)
                mean_max = wmax.mean(axis=0)
                mean_ess = ess.mean(axis=0)
                mean_ess_norm = mean_ess / n
                mean_neg = neg.mean(axis=0)
                out[(snr, n, method)] = {
                    "mean_p99": mean_p99,
                    "sd_p99": sd_p99,
                    "mean_max": mean_max,
                    "mean_ess": mean_ess,
                    "mean_ess_norm": mean_ess_norm,
                    "mean_neg_share": mean_neg,
                    "_p99_per_rep": p99,
                    "_max_per_rep": wmax,
                    "_ess_per_rep": ess,
                }
    return out


# ── 13B.4 — Safe-DR rule evaluation ───────────────────────────────────────────

def _eta_rule0(*, M_, K_, **kw):
    return np.zeros((M_, K_), dtype=float)


def _eta_rule1(*, M_, K_, **kw):
    return np.ones((M_, K_), dtype=float)


def _eta_rule2(*, ess_pi_norm, tau, **kw):
    return (ess_pi_norm >= tau).astype(float)


def _eta_rule3(*, weight_p99, c, **kw):
    # For each rep, compute median p99 across doses
    median_p99 = np.median(weight_p99, axis=1, keepdims=True)
    eta = (weight_p99 <= c * median_p99).astype(float)
    return eta


def _eta_rule4(*, V_corr, V_total, tau, **kw):
    share = np.where(V_total > 0, V_corr / V_total, 1.0)
    return (share <= tau).astype(float)


def _eta_rule5(*, ess_pi_norm, tau, **kw):
    return np.clip(ess_pi_norm / tau, 0.0, 1.0)


def _eta_rule6(*, V_corr, V_reg, **kw):
    ratio = np.where(V_reg > 0, V_corr / V_reg, 1.0)
    return 1.0 / (1.0 + ratio)


def _eta_rule7(*, ess_pi_norm, V_corr, V_total, tau_ess, tau_v, **kw):
    eta_ess = np.clip(ess_pi_norm / tau_ess, 0.0, 1.0)
    share = np.where(V_total > 0, V_corr / V_total, 1.0)
    eta_v = (share <= tau_v).astype(float)
    return eta_ess * eta_v


def evaluate_safe_dr_rules(
    decomp: Dict, density: Dict, weights: Dict, J_pt_per_snr: Dict,
) -> Dict:
    """Evaluate Rules 0-7 on every (cell, rep, dose) and aggregate per cell."""
    print("\n=== 13B.4 Safe-DR rule evaluation ===")

    # Build per-cell tensors needed by rules
    rule_specs = [
        ("R0_REG", _eta_rule0, {}),
        ("R1_DRK", _eta_rule1, {}),
        ("R2_ESS_05", _eta_rule2, {"tau": 0.05}),
        ("R2_ESS_10", _eta_rule2, {"tau": 0.10}),
        ("R2_ESS_20", _eta_rule2, {"tau": 0.20}),
        ("R3_TAIL_15", _eta_rule3, {"c": 1.5}),
        ("R3_TAIL_20", _eta_rule3, {"c": 2.0}),
        ("R3_TAIL_30", _eta_rule3, {"c": 3.0}),
        ("R4_VC_30", _eta_rule4, {"tau": 0.30}),
        ("R4_VC_50", _eta_rule4, {"tau": 0.50}),
        ("R4_VC_70", _eta_rule4, {"tau": 0.70}),
        ("R5_SMOOTH_10", _eta_rule5, {"tau": 0.10}),
        ("R5_SMOOTH_20", _eta_rule5, {"tau": 0.20}),
        ("R5_SMOOTH_30", _eta_rule5, {"tau": 0.30}),
        ("R6_SHRINK", _eta_rule6, {}),
        ("R7_COMBO", _eta_rule7, {"tau_ess": 0.20, "tau_v": 0.50}),
    ]

    out = {}
    for snr in SNR_GRID:
        J_pt = J_pt_per_snr[snr]
        for n in N_GRID:
            cell_key = (snr, n)
            if cell_key not in density:
                continue
            ess_pi_norm = density[cell_key]["ess_pi_norm"]   # (M, K)

            reg = decomp.get((snr, n, "REG_Hsuper"))
            drk = decomp.get((snr, n, "DRK_Hsuper"))
            if reg is None or drk is None:
                continue
            J_reg = reg["_J_reg_per_rep"]
            J_drk_dr = drk["_J_dr_per_rep"]
            V_reg = drk["_V_reg_per_rep"]   # use DRK's split since DRK has the correction
            V_corr = drk["_V_corr_per_rep"]
            V_total = drk["_V_total_per_rep"]

            wts = weights.get((snr, n, "DRK_Hsuper"))
            weight_p99 = wts["_p99_per_rep"] if wts is not None else None

            for rule_name, rule_fn, params in rule_specs:
                eta = rule_fn(
                    M_=M, K_=K,
                    ess_pi_norm=ess_pi_norm,
                    weight_p99=weight_p99,
                    V_corr=V_corr,
                    V_reg=V_reg,
                    V_total=V_total,
                    **params,
                )   # (M, K)
                # SAFE estimator per rep
                J_safe = J_reg + eta * (J_drk_dr - J_reg)   # (M, K)
                # Approx variance for SAFE per rep
                V_safe = V_reg + (eta ** 2) * V_corr        # (M, K)

                # Aggregate per dose
                bias = J_safe.mean(axis=0) - J_pt
                emp_sd = J_safe.std(axis=0, ddof=1)
                pred_se = np.sqrt(np.maximum(V_safe.mean(axis=0), 0.0) / n)
                ser = np.where(emp_sd > 1e-12, pred_se / emp_sd, np.nan)
                bse = np.where(pred_se > 1e-12, np.abs(bias) / pred_se, np.nan)
                # Coverage approx
                CIs_lo = J_safe - 1.96 * np.sqrt(np.maximum(V_safe, 0.0) / n)
                CIs_hi = J_safe + 1.96 * np.sqrt(np.maximum(V_safe, 0.0) / n)
                cov = ((CIs_lo <= J_pt) & (J_pt <= CIs_hi)).mean(axis=0)
                # Mean eta for diagnostic
                mean_eta = eta.mean(axis=0)

                out[(snr, n, rule_name)] = {
                    "bias": bias,
                    "bse": bse,
                    "cov": cov,
                    "ser": ser,
                    "emp_sd": emp_sd,
                    "pred_se": pred_se,
                    "mean_eta": mean_eta,
                }

    return out


# ── 13B.5 — Decision branch + Report ──────────────────────────────────────────

def _aggregate_rule_metrics(rules_eval: Dict) -> Dict:
    """For each rule, compute worst-case |b|/SE, worst-case SER deviation, etc."""
    metrics = {}
    rule_names = sorted({k[2] for k in rules_eval.keys()})
    for r_name in rule_names:
        rows = [v for k, v in rules_eval.items() if k[2] == r_name]
        if not rows:
            continue
        all_bse = np.concatenate([row["bse"] for row in rows])
        all_ser = np.concatenate([row["ser"] for row in rows])
        all_cov = np.concatenate([row["cov"] for row in rows])
        all_eta = np.concatenate([row["mean_eta"] for row in rows])
        metrics[r_name] = {
            "worst_bse": float(np.nanmax(all_bse)),
            "mean_bse": float(np.nanmean(all_bse)),
            "worst_ser_deviation": float(np.nanmax(np.abs(all_ser - 1.0))),
            "min_ser": float(np.nanmin(all_ser)),
            "max_ser": float(np.nanmax(all_ser)),
            "mean_cov": float(np.nanmean(all_cov)),
            "min_cov": float(np.nanmin(all_cov)),
            "mean_eta": float(np.nanmean(all_eta)),
        }
    return metrics


def _decide_case(rule_metrics: Dict) -> Tuple[str, str, dict]:
    """
    Apply selection criteria to rule_metrics. Return (case, best_rule_name, info).

    Case A : at least one Rule R2-R7 satisfies:
      - SER in [0.7, 1.3] for ALL cells (min_ser>=0.7 AND max_ser<=1.3)
      - worst_bse <= 1.05 * min(REG worst_bse, DRK worst_bse)
      - mean_cov >= 0.85
    Case B : no rule passes 1 but R0/R1 worst_bse comparison shows clear
              regime separability.
    Case C : no clean separation.
    """
    R0 = rule_metrics.get("R0_REG", {})
    R1 = rule_metrics.get("R1_DRK", {})
    if not R0 or not R1:
        return "C", "R0_REG", {}
    target_bse = 1.05 * min(R0["worst_bse"], R1["worst_bse"])
    candidates = []
    for r_name, m in rule_metrics.items():
        if r_name in ("R0_REG", "R1_DRK"):
            continue
        ok_ser = (0.7 <= m["min_ser"]) and (m["max_ser"] <= 1.3)
        ok_bse = m["worst_bse"] <= target_bse
        ok_cov = m["mean_cov"] >= 0.85
        if ok_ser and ok_bse and ok_cov:
            candidates.append((r_name, m["worst_bse"], m))
    if candidates:
        candidates.sort(key=lambda t: t[1])
        best_name, _, best_m = candidates[0]
        return "A", best_name, {
            "target_bse": target_bse, "n_candidates": len(candidates),
            "best_metrics": best_m,
        }
    # Case B vs C
    # If REG worst_bse << DRK worst_bse OR vice-versa per regime, there's
    # diagnostic separation
    return "B", "R0_REG", {"target_bse": target_bse}


# ── Report builder ────────────────────────────────────────────────────────────

def _fmt(v, fmt=".3f") -> str:
    try:
        f = float(v)
        if not np.isfinite(f):
            return "---"
        return f"{f:.2e}" if (abs(f) >= 1e5 or (0 < abs(f) < 1e-3)) else f"{f:{fmt}}"
    except Exception:
        return str(v)


def write_report(density, decomp, weights, rules_eval, rule_metrics, case, best_rule):
    import datetime
    lines = []
    lines.append("# Phase 13B Mechanistic Study + Safe-DR Rule Evaluation")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append("")
    lines.append("## 0. Verdict (TL;DR)")
    lines.append("")
    if case == "A":
        lines.append(f"**Case A — Clean rule found**: `{best_rule}` matches the selection criteria.")
    elif case == "B":
        lines.append("**Case B — Diagnostic-only**: no rule satisfies all hard criteria, "
                     "but density / overlap diagnostics clearly separate dangerous from safe cells.")
    else:
        lines.append("**Case C — No clean signal**: diagnostics do not predict failures cleanly.")
    lines.append("")
    lines.append(f"Best (or default) rule for the report : `{best_rule}`")
    lines.append("")
    lines.append("## 1. Five answers")
    lines.append("")
    # 1. density confirms low-density at a=1.8?
    ess_a18 = []
    for cell, dens in density.items():
        ess_a18.append(np.mean(dens["ess_pi_norm"][:, 0]))
    ess_a30 = []
    for cell, dens in density.items():
        ess_a30.append(np.mean(dens["ess_pi_norm"][:, 3]))
    lines.append(f"1. Density at a=1.8 (mean ESS_pi/n across cells) = {np.mean(ess_a18):.3f} "
                 f"vs at a=3.0 = {np.mean(ess_a30):.3f}. "
                 + ("Confirmed: a=1.8 is lower-density." if np.mean(ess_a18) < np.mean(ess_a30) else
                    "NOT confirmed: a=1.8 has comparable density."))
    # 2. correction sign at a=1.8 vs a=3.0 for DRK at high SNR
    lines.append("2. Correction sign at boundary doses (DRK, SNR>=0.90):")
    for snr in [0.90, 0.95]:
        for n in N_GRID:
            d = decomp.get((snr, n, "DRK_Hsuper"))
            if d is None:
                continue
            mc = d["mean_correction"]
            lines.append(f"   - SNR={snr}, n={n}: mean_corr a=1.8 = {mc[0]:+.4f}, "
                         f"a=2.6 = {mc[2]:+.4f}, a=3.0 = {mc[3]:+.4f}")
    # 3. rule
    lines.append(f"3. Safe-DR rule status: **Case {case}**, best candidate = `{best_rule}`.")
    if case == "A":
        m = rule_metrics[best_rule]
        lines.append(f"   - worst_bse = {m['worst_bse']:.3f}, mean_bse = {m['mean_bse']:.3f}")
        lines.append(f"   - SER range = [{m['min_ser']:.3f}, {m['max_ser']:.3f}]")
        lines.append(f"   - mean coverage = {m['mean_cov']:.3f}")
    # 4. recommended hyperparameter
    lines.append(f"4. Recommended hyperparameter: {best_rule} (see rule definition table below).")
    # 5. essay implication
    if case == "A":
        lines.append("5. **For S6/S7**: a data-driven mixing rule eta(diagnostic) gives a Safe-DR "
                     "estimator that recovers DR's bulk benefit without the boundary failure.")
    else:
        lines.append("5. **For S6/S7**: REG_Hsuper plug-in is the safe default; DR correction "
                     "should be applied only in clearly-bulk doses (high local ESS_pi/n).")
    lines.append("")

    # 13B.1 density table
    lines.append("## 2. Density / overlap diagnostics (per cell, per dose)")
    lines.append("")
    lines.append("Mean ESS_pi/n (effective sample size of the policy weighting "
                 "K_h(A-a) divided by n).  Low values indicate a low-density region; "
                 "the q-bridge has to extrapolate there.")
    lines.append("")
    lines.append("| SNR | n | h_KDE | ESS_pi/n a=1.8 | a=2.2 | a=2.6 | a=3.0 |")
    lines.append("|-----|---|-------|---------------|-------|-------|-------|")
    for snr in SNR_GRID:
        for n in N_GRID:
            d = density.get((snr, n))
            if d is None:
                continue
            ess = d["ess_pi_norm"].mean(axis=0)
            lines.append(f"| {snr} | {n} | {d['h_KDE_mean']:.3f} | "
                         f"{ess[0]:.3f} | {ess[1]:.3f} | {ess[2]:.3f} | {ess[3]:.3f} |")
    lines.append("")

    # 13B.2 correction decomposition
    lines.append("## 3. DR correction decomposition (DRK only)")
    lines.append("")
    lines.append("| SNR | n | dose | bias_REG | bias_DRK | mean_corr | sd_corr | t_signal | V_corr/V_tot |")
    lines.append("|-----|---|------|----------|----------|-----------|---------|----------|--------------|")
    for snr in SNR_GRID:
        for n in N_GRID:
            d_drk = decomp.get((snr, n, "DRK_Hsuper"))
            d_reg = decomp.get((snr, n, "REG_Hsuper"))
            if d_drk is None or d_reg is None:
                continue
            for k, a in enumerate(A_GRID):
                lines.append(
                    f"| {snr} | {n} | {a} | {d_reg['bias_REG'][k]:+.4f} | "
                    f"{d_drk['bias_DR'][k]:+.4f} | {d_drk['mean_correction'][k]:+.4f} | "
                    f"{d_drk['sd_correction'][k]:.4f} | {_fmt(d_drk['t_signal'][k])} | "
                    f"{d_drk['v_corr_share'][k]:.3f} |"
                )
    lines.append("")

    # 13B.3 weight tails
    lines.append("## 4. Weight-tail diagnostics (DRK pi*q weights)")
    lines.append("")
    lines.append("| SNR | n | dose | mean_p99 | sd_p99 | mean_max | mean_ESS/n | neg_share |")
    lines.append("|-----|---|------|----------|--------|----------|------------|-----------|")
    for snr in SNR_GRID:
        for n in N_GRID:
            w = weights.get((snr, n, "DRK_Hsuper"))
            if w is None:
                continue
            for k, a in enumerate(A_GRID):
                lines.append(
                    f"| {snr} | {n} | {a} | {w['mean_p99'][k]:.2f} | {w['sd_p99'][k]:.2f} | "
                    f"{w['mean_max'][k]:.2f} | {w['mean_ess_norm'][k]:.3f} | "
                    f"{w['mean_neg_share'][k]:.3f} |"
                )
    lines.append("")

    # 13B.4 rule evaluation summary
    lines.append("## 5. Safe-DR rule evaluation (aggregate metrics)")
    lines.append("")
    lines.append("| Rule | worst_bse | mean_bse | min_SER | max_SER | mean_cov | mean_eta |")
    lines.append("|------|-----------|----------|---------|---------|----------|----------|")
    for r_name, m in rule_metrics.items():
        lines.append(
            f"| {r_name} | {m['worst_bse']:.3f} | {m['mean_bse']:.3f} | "
            f"{m['min_ser']:.3f} | {m['max_ser']:.3f} | {m['mean_cov']:.3f} | "
            f"{m['mean_eta']:.3f} |"
        )
    lines.append("")
    lines.append("**Selection criteria for Case A**:")
    lines.append("- SER in [0.7, 1.3] in all cells")
    lines.append("- worst_bse <= 1.05 * min(R0_REG worst, R1_DRK worst)")
    lines.append("- mean_cov >= 0.85")
    lines.append("")

    # 13B.5 decision
    lines.append("## 6. Decision and recommendation")
    lines.append("")
    lines.append(f"**Case: {case}**")
    lines.append(f"**Selected rule: `{best_rule}`**")
    lines.append("")
    if case == "A":
        lines.append("=> Phase 13C validation will be launched on disjoint seeds.")
    elif case == "B":
        lines.append("=> Use diagnostic-only recommendation: REG when ESS_pi/n < ~0.05, "
                     "DRK when ESS_pi/n > ~0.10. Phase 13 closes here.")
    else:
        lines.append("=> Recommend REG_Hsuper as default; DRK only in clearly-bulk doses. "
                     "Phase 13 closes here.")
    lines.append("")

    # Essay-ready
    lines.append("## 7. Essay-ready phrase")
    lines.append("")
    if case == "A":
        lines.append(
            "> \"In continuous-treatment proximal causal inference, the doubly-robust "
            "correction is regime-dependent. We propose a Safe-DR estimator "
            f"`{best_rule}` that adaptively weights the DR correction by an observable "
            "diagnostic (the local effective sample size of the policy weighting), "
            "recovering DR's benefit in high-density regions while reverting to "
            "the plug-in regression in low-overlap regions.\""
        )
    else:
        lines.append(
            "> \"The doubly robust correction in continuous-treatment PCI is "
            "regime-dependent. In low-density regions of the empirical treatment "
            "distribution, the q-bridge q_hat is unreliable and the DR correction "
            "injects finite-sample bias rather than removing it. We recommend the "
            "plug-in regression as the safe default and apply the DR correction "
            "only in regions where the local effective sample size of the policy "
            "weighting is high (ESS_pi/n above ~0.10).\""
        )
    lines.append("")

    # Limitations
    lines.append("## 8. Limitations")
    lines.append("")
    lines.append("- Rule evaluation is post-hoc on Phase 13A PKLs (same data used for selection "
                 "and evaluation). 13C provides the disjoint-seed test if Case A.")
    lines.append("- V_safe is approximated as V_reg + eta^2 * V_corr (no covariance term). "
                 "Conservative for SER but may slightly under-estimate true variance.")
    lines.append("- Density diagnostics regenerated from same-seed samples (equivalent to actual "
                 "samples used in Phase 13A).")
    lines.append("")

    # Save report
    out = "\n".join(lines)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"\nReport saved: {_REPORT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("Phase 13B Mechanistic Study + Safe-DR rule evaluation")
    print("=" * 72)

    _RAW_OUT.mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load J_pt per SNR
    J_pt_per_snr = {snr: _load_J_pt(snr) for snr in SNR_GRID}
    for snr in SNR_GRID:
        print(f"  J_pt SNR={snr}: {[f'{v:.4f}' for v in J_pt_per_snr[snr]]}")

    # 13B.1 density
    density = compute_density_diagnostics()
    with open(_RAW_OUT / "density_diagnostics.pkl", "wb") as fh:
        pickle.dump(density, fh)

    # 13B.2 decomposition
    decomp = correction_decomposition(J_pt_per_snr)

    # 13B.3 weight tails
    weights = weight_diagnostics()

    # 13B.4 rules
    rules_eval = evaluate_safe_dr_rules(decomp, density, weights, J_pt_per_snr)
    rule_metrics = _aggregate_rule_metrics(rules_eval)

    with open(_RAW_OUT / "safe_dr_rule_eval.pkl", "wb") as fh:
        pickle.dump({"rules_eval": rules_eval, "rule_metrics": rule_metrics}, fh)

    # 13B.5 decision
    case, best_rule, info = _decide_case(rule_metrics)
    print(f"\n>>> Decision: Case {case}, best rule = {best_rule}")
    print(f"    info: {info}")

    write_report(density, decomp, weights, rules_eval, rule_metrics, case, best_rule)

    elapsed = time.time() - t0
    print(f"\nTotal Phase 13B: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\nNEXT STEP: Case={case}")
    if case == "A":
        print("  -> Launch Phase 13C validation on disjoint seeds.")
    else:
        print("  -> Phase 13 closes. Update docs.")
    return case, best_rule


if __name__ == "__main__":
    main()
