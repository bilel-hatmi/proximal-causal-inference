"""
pci.runner.diagnostics
======================

Post-block aggregation utilities used by every report-generation pass:

* :func:`_decompose`              Aggregate M reps -> bias / variance / coverage / SER
* :func:`_augment_decomp_with_ser`  Inject SER (predicted SE / empirical SE) at ref dose
* :func:`_pred_cov`                Biased-normal predicted coverage
* :func:`_fmt`                    Numeric formatter (sci-notation auto-switch)

Extracted from
``simulations/experiments/dgp2_bias_diagnostics.py`` and
``simulations/experiments/bennett_rff_overnight.py``.

Note on REF_IDX
---------------
``_decompose`` and ``_augment_decomp_with_ser`` accept ``ref_idx`` as a
parameter (default 2 — the historical value used in DGP2 a_grid =
[1.8, 2.2, 2.6, 3.0]). Callers should pass an explicit ``ref_idx`` when their
``a_grid`` doesn't have the reference dose at index 2 (e.g., DGP1 with K=3
uses ref_idx=1). The shims in the original modules pass their module-level
``REF_IDX``, preserving prior behaviour.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import stats


def _fmt(v: Any, fmt: str = ".4f", sci_thresh: float = 1e5) -> str:
    """Numeric formatter with auto sci-notation for |v| >= sci_thresh.

    Returns "---" for None/NaN, "inf" for +/-inf, sci-notation for big |v|,
    else fixed-point with the requested format spec.
    """
    try:
        if v is None:
            return "---"
        f = float(v)
        if not np.isfinite(f):
            return "---" if np.isnan(f) else "inf"
        return f"{f:.2e}" if abs(f) >= sci_thresh else f"{f:{fmt}}"
    except Exception:
        return str(v)


def _pred_cov(bias: float, se: float) -> float:
    """Predicted 95% coverage given a bias-affected normal estimator.

    P(|N(bias, se^2)| <= 1.96 * se) = Phi(1.96 - z) - Phi(-1.96 - z)
    with z = |bias| / se.
    """
    if se <= 0 or not np.isfinite(se) or not np.isfinite(bias):
        return np.nan
    z = abs(bias) / se
    return float(stats.norm.cdf(1.96 - z) - stats.norm.cdf(-1.96 - z))


def _decompose(data: dict, ref_idx: int = 2) -> dict | None:
    """Aggregate the M reps in ``data["records"]`` into bias / variance / coverage.

    Returns a dict (or None if no successful reps) with:

    Per-grid (length K) :
      bias_reg, bias_dr, bias_corr, required_corr, correction_ratio,
      correction_efficiency, V_total, V_reg, V_corr, V_cov,
      SE_grid, bse_grid (=|bias_dr|/SE), pred_cov_grid

    Scalar:
      n, M_ok, J_pt, coverage_ref (empirical 95% coverage at ref_idx),
      ess_min_mean, w_p99_mean, q_clip_mean, riesz_res_mean,
      neg_share_mean, jensen_gap_mean, mise_mean
    """
    records = [r for r in data["records"] if r.get("error") is None]
    if not records:
        return None
    n      = int(data["meta"]["n"])
    J_pt   = np.array(data["meta"]["J_policy_true"])
    M_ok   = len(records)

    J_rg_all = np.array([r["J_reg"] for r in records])
    J_dr_all = np.array([r["J_dr"]  for r in records])
    V_hg_all = np.array([r["V_hat_grid"]        for r in records])
    V_rg_all = np.array([r["V_reg_grid"]        for r in records])
    V_co_all = np.array([r["V_correction_grid"] for r in records])

    mean_Jrg = np.mean(J_rg_all, axis=0)
    mean_Jdr = np.mean(J_dr_all, axis=0)
    bias_reg  = mean_Jrg - J_pt
    bias_dr   = mean_Jdr - J_pt
    bias_corr = mean_Jdr - mean_Jrg

    required_corr = -bias_reg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        correction_ratio = np.where(
            np.abs(required_corr) > 1e-12,
            bias_corr / required_corr,
            np.nan,
        )
        eff = np.where(
            np.abs(bias_reg) > 1e-12,
            1.0 - np.abs(bias_dr) / np.abs(bias_reg),
            np.nan,
        )
        V_total = np.nanmean(V_hg_all, axis=0)
        V_reg   = np.nanmean(V_rg_all, axis=0)
        V_corr  = np.nanmean(V_co_all, axis=0)
    V_cov = (V_total - V_reg - V_corr) / 2.0

    SE_grid  = np.sqrt(np.abs(V_total) / n)
    bse_grid = np.abs(bias_dr) / np.where(SE_grid > 1e-15, SE_grid, np.nan)
    K_actual = int(bias_dr.shape[0])
    pred_cov_grid = np.array([
        _pred_cov(bias_dr[k], SE_grid[k]) for k in range(K_actual)
    ])

    coverage_ref = float(np.mean(
        np.abs(J_dr_all[:, ref_idx] - J_pt[ref_idx]) <= 1.96 * SE_grid[ref_idx]
    ))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ess_arr     = np.array([r["ESS_min"]       for r in records])
        w_p99_arr   = np.nanmax(np.array([r["weight_p99_grid"] for r in records]), axis=1)
        q_clip_arr  = np.array([r["q_clip_fraction"] for r in records])
        rr_arr      = np.nanmean(np.array([r["riesz_residual_grid_mean"] for r in records]), axis=1)
        ns_arr      = np.nanmean(np.array([r["neg_share_grid_mean"]      for r in records]), axis=1)
        jg_arr      = np.nanmean(np.array([r["jensen_gap"] for r in records]), axis=0)
        mise_arr    = np.array([r["mise"] for r in records])

    return dict(
        n=n, M_ok=M_ok,
        J_pt=J_pt,
        bias_reg=bias_reg, bias_dr=bias_dr, bias_corr=bias_corr,
        required_corr=required_corr, correction_ratio=correction_ratio,
        correction_efficiency=eff,
        V_total=V_total, V_reg=V_reg, V_corr=V_corr, V_cov=V_cov,
        SE_grid=SE_grid, bse_grid=bse_grid,
        pred_cov_grid=pred_cov_grid,
        coverage_ref=coverage_ref,
        ess_min_mean    = float(np.nanmean(ess_arr)),
        w_p99_mean      = float(np.nanmean(w_p99_arr)),
        q_clip_mean     = float(np.nanmean(q_clip_arr)),
        riesz_res_mean  = float(np.nanmean(rr_arr)),
        neg_share_mean  = float(np.nanmean(ns_arr)),
        jensen_gap_mean = jg_arr,
        mise_mean       = float(np.nanmean(mise_arr)),
    )


def _augment_decomp_with_ser(decomp: dict, data: dict, ref_idx: int = 2) -> dict:
    """Compute SER (predicted SE / empirical SE) at ref_idx and inject into decomp.

    SER ~ 1.0 = well-calibrated bootstrap variance.
    SER > 1.0 = over-conservative (predicted SE too big).
    SER < 1.0 = under-coverage hazard (predicted SE too small).
    """
    if decomp is None:
        return decomp
    try:
        records = [r for r in data["records"] if r.get("error") is None]
        if not records:
            return decomp
        n = int(data["meta"]["n"])
        J_dr_all = np.array([r["J_dr"] for r in records])
        V_hg_all = np.array([r["V_hat_grid"] for r in records])
        emp_sd = float(np.std(J_dr_all[:, ref_idx], ddof=1))
        with np.errstate(invalid="ignore"):
            se_per_rep = np.sqrt(np.maximum(V_hg_all[:, ref_idx], 0.0) / n)
            pred_se = float(np.nanmean(se_per_rep))
        ser = pred_se / emp_sd if emp_sd > 1e-12 else np.nan
        decomp["ser_ref"] = ser
        decomp["emp_sd_ref"] = emp_sd
        decomp["pred_se_ref"] = pred_se
    except Exception:
        decomp["ser_ref"] = np.nan
    return decomp
