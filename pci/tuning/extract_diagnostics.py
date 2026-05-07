"""
pci.tuning.extract_diagnostics
==============================

Aggregate per-rep records (the canonical PKL schema produced by
:func:`pci.runner.block_runner._build_record`) into a flat diagnostics dict
ready to feed into the blind score functions.

The output dict contains:

* M_ok, n
* kappa_h, kappa_r, eff_rank_h, eff_rank_r, residual_norm_h, residual_norm_r
* RR_ref (Riesz residual at the reference dose)
* ESS_min_ratio, w_p99_ref
* SER (predicted SE / empirical SE at ref dose) — pseudo-blind quality probe
* _bse_ref_NONBLIND, _cov_ref_NONBLIND, _bias_ref_NONBLIND — POST-HOC ONLY
  (these touch J_policy_true and MUST NOT be used during model selection;
  the leading underscore + uppercase suffix flag them as off-limits to the
  blind score functions)

Extracted from
``simulations/experiments/simulation/blind_tuning_dgp2_bennett.py``. The original module continues to re-export this as a
backward-compatibility shim.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


# Default reference dose index for DGP2 (a_grid = [1.8, 2.2, 2.6, 3.0],
# ref = a=2.6 = index 2). Override via the ref_idx parameter for other
# a_grids (e.g. DGP1 with K=3 uses ref_idx=1).
_DEFAULT_REF_IDX = 2


def _safe(v, default=float("nan")):
    """Convert v to float; return default on TypeError/ValueError or non-finite."""
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _array_safe(arr, idx, default=float("nan")):
    """Index arr[idx] and convert to float; return default on any error."""
    try:
        return float(arr[idx])
    except Exception:
        return default


def extract_diagnostics(data: dict, ref_idx: int = _DEFAULT_REF_IDX) -> Dict[str, float]:
    """Aggregate per-rep records into mean diagnostics for the blind score.

    Returns dict with keys used by ``bennett_blind_score_v2`` /
    ``drkpv_blind_score``. See module docstring for the full key list.
    """
    recs = [r for r in data.get("records", []) if r.get("error") is None]
    if not recs:
        return {"M_ok": 0}
    n = int(data["meta"]["n"])

    out = {
        "M_ok": len(recs),
        "n": n,
        "kappa_h": float(np.nanmean([_safe(r.get("kappa_h")) for r in recs])),
        "kappa_r": float(np.nanmean([_safe(r.get("kappa_r")) for r in recs])),
        "eff_rank_h": float(np.nanmean([_safe(r.get("eff_rank_h")) for r in recs])),
        "eff_rank_r": float(np.nanmean([_safe(r.get("eff_rank_r")) for r in recs])),
        "residual_norm_h": float(np.nanmean([_safe(r.get("residual_norm_h")) for r in recs])),
        "residual_norm_r": float(np.nanmean([_safe(r.get("residual_norm_r")) for r in recs])),
        "RR_ref": float(np.nanmean([_array_safe(r.get("riesz_residual_grid_mean", []), ref_idx) for r in recs])),
        "ESS_min_ratio": float(np.nanmean([_safe(r.get("ESS_min")) for r in recs])) / n,
        "w_p99_ref": float(np.nanmean([_array_safe(r.get("weight_p99_grid", []), ref_idx) for r in recs])),
    }

    # SER (pseudo-blind): SE_predicted_mean / SD_empirical
    try:
        J_dr = np.array([r["J_dr"] for r in recs])
        V_hg = np.array([r["V_hat_grid"] for r in recs])
        with np.errstate(invalid="ignore"):
            se_per = np.sqrt(np.maximum(V_hg, 0.0) / n)
        SE_pred = float(np.nanmean(se_per[:, ref_idx]))
        SD_emp = float(np.std(J_dr[:, ref_idx], ddof=1))
        out["SER"] = SE_pred / SD_emp if SD_emp > 1e-12 else float("nan")
        # bse_ref kept for FINAL VALIDATION only (post-hoc), NEVER for selection
        if "J_policy_true" in data["meta"]:
            J_pt = np.array(data["meta"]["J_policy_true"])
            bias_ref = float(np.mean(J_dr[:, ref_idx]) - J_pt[ref_idx])
            out["_bse_ref_NONBLIND"] = abs(bias_ref) / SE_pred if SE_pred > 1e-12 else float("nan")
            out["_cov_ref_NONBLIND"] = float(np.mean(np.abs(J_dr[:, ref_idx] - J_pt[ref_idx]) <= 1.96 * se_per[:, ref_idx]))
            out["_bias_ref_NONBLIND"] = bias_ref
    except Exception:
        out["SER"] = float("nan")

    return out
