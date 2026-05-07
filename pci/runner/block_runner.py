"""
pci.runner.block_runner
=======================

Two-level checkpointed Monte-Carlo runner used by virtually every active
experiment script. Splits a job ``(dgp, factory_fn, n, M, seed_base)`` into
M reps, generating one DGP sample per rep, fitting the estimator, and
recording the result.

Two checkpoint levels:

* **Per-block PKL** at ``raw_dir/<label>.pkl`` — final output, present once
  the block of M reps is complete; if present, the block is skipped.
* **Partial PKL** at ``raw_dir/<label>.partial.pkl`` — flushed every
  ``CHECKPOINT_FREQ`` reps; if present at startup, the run resumes from
  ``len(records)``.

The :func:`_build_record` helper extracts the canonical 30+ keys from an
``EstimationResult`` into a JSON-friendly dict; this format is the de-facto
PKL schema across the project. New keys can be added without breaking older
PKLs (they default to NaN).

These were extracted from
``simulations/experiments/dgp2_bias_diagnostics.py`` where
they had accumulated for historical reasons. The original module continues
to re-export them as a backward-compatibility shim.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np


# Number of reps between partial checkpoint writes. Conservative default —
# users can raise it for fast estimators.
CHECKPOINT_FREQ = 10


def _run_block(
    dgp,
    factory_fn,    # callable () -> estimator
    a_grid: np.ndarray,
    J_pt: np.ndarray,
    n: int,
    M: int,
    seed_base: int,
    label: str,
    raw_dir: Path,
) -> dict:
    """Run M reps with two-level checkpointing. Returns data dict.

    Behaviour
    ---------
    1. If ``raw_dir/<label>.pkl`` exists: skip, return loaded data (cached
       result of a previous full run).
    2. Else if ``raw_dir/<label>.partial.pkl`` exists: load, resume from
       ``len(records)``.
    3. Else: start at rep 0.

    For each rep i in range(start_i, M):
        seed = seed_base + i
        sample = dgp.generate(n=n, seed=seed)
        try:
            est = factory_fn()
            est.fit(sample, m_true_fn=dgp.m_true)
            res = est.estimate()
            records.append(_build_record(res, K, seed, J_pt))
        except Exception as exc:
            records.append({"error": str(exc), "seed": seed})

        if (i+1) % CHECKPOINT_FREQ == 0: write partial.

    On full completion, writes the final PKL and removes the partial.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_path    = raw_dir / f"{label}.pkl"
    partial_path = raw_dir / f"{label}.partial.pkl"

    if save_path.exists():
        print(f"  [SKIP] {label}")
        with open(save_path, "rb") as fh:
            return pickle.load(fh)

    if partial_path.exists():
        with open(partial_path, "rb") as fh:
            records = pickle.load(fh)
        start_i = len(records)
        print(f"  [RESUME] {label} from rep {start_i}/{M}")
    else:
        records = []
        start_i = 0

    K = len(a_grid)

    for i in range(start_i, M):
        seed = seed_base + i
        sample = dgp.generate(n=n, seed=seed)
        try:
            est = factory_fn()
            est.fit(sample, m_true_fn=dgp.m_true)
            res = est.estimate()
            records.append(_build_record(res, K, seed, J_pt))
        except Exception as exc:
            warnings.warn(f"[{label} seed={seed}] {exc}", RuntimeWarning)
            records.append({"error": str(exc), "seed": seed})

        if (i + 1) % CHECKPOINT_FREQ == 0:
            with open(partial_path, "wb") as fh:
                pickle.dump(records, fh)

    data = {
        "records": records,
        "meta": {
            "label": label, "n": n, "M": M, "J_policy_true": J_pt.tolist(),
            "a_grid": a_grid.tolist(), "seed_base": seed_base,
        },
    }
    with open(save_path, "wb") as fh:
        pickle.dump(data, fh)
    if partial_path.exists():
        partial_path.unlink()
    print(f"  [DONE] {label}")
    return data


def _build_record(res, K: int, seed: int, J_pt: np.ndarray) -> dict:
    """Extract the canonical PKL record schema from an EstimationResult.

    The schema has 30+ keys covering point estimates, variance decomposition,
    weight diagnostics, spectral diagnostics (Phase 16), and Phase 11 Kallus
    passthrough fields (only present when a Kallus estimator was used).

    Missing keys default to NaN (or a NaN-array of length K) — adding a new
    key in a future version does not break older PKLs.
    """
    ex = res.extra

    def _arr(key):
        v = ex.get(key, np.full(K, np.nan))
        return v.tolist() if hasattr(v, "tolist") else list(v)

    def _scl(key, default=np.nan):
        return float(ex.get(key, default))

    rec = {
        "seed": seed, "error": None,
        "psi_hat": float(res.psi_hat),
        "V_hat":   float(res.V_hat),
        "J_reg":              _arr("J_reg"),
        "J_dr":               _arr("J_dr"),
        "V_hat_grid":         _arr("V_hat_grid"),
        "V_reg_grid":         _arr("V_reg_grid"),
        "V_correction_grid":  _arr("V_correction_grid"),
        "ESS_grid":           _arr("ESS_grid"),
        "ESS_min":            _scl("ESS_min"),
        "weight_p99_grid":    _arr("weight_p99_grid"),
        "weight_max_grid":    _arr("weight_max_grid"),
        "q_clip_fraction":    _scl("q_clip_fraction"),
        "q_negative_share_grid": _arr("q_negative_share_grid"),
        "riesz_residual_grid_mean": _arr("riesz_residual_grid_mean"),
        "cond_M_grid_mean":   _arr("cond_M_grid_mean"),
        "neg_share_grid_mean":_arr("neg_share_grid_mean"),
        "bandwidth":          _scl("bandwidth"),
        "jensen_gap":         _arr("jensen_gap"),
        "mise":               _scl("mise"),
        # Phase 16 canonical spectral diagnostics
        "kappa_h":          _scl("kappa_h"),
        "eff_rank_h":       _scl("eff_rank_h"),
        "residual_norm_h":  _scl("residual_norm_h"),
        "kappa_r":          _scl("kappa_r"),
        "eff_rank_r":       _scl("eff_rank_r"),
        "residual_norm_r":  _scl("residual_norm_r"),
        "residual_norm_r_grid": _arr("residual_norm_r_grid"),
    }
    # Phase 11 passthrough (only if KallusStabilizedPolicyQ was used)
    if ex.get("kallus_present", False):
        rec["kallus_present"]                  = True
        rec["kallus_weighted_residual_grid"]   = _arr("kallus_weighted_residual_grid")
        rec["kallus_inner_max_grid"]           = _arr("kallus_inner_max_grid")
        rec["kallus_alpha_norm_grid"]          = _arr("kallus_alpha_norm_grid")
        rec["kallus_ESS_grid"]                 = _arr("kallus_ESS_grid")
        rec["kallus_weight_p99_grid"]          = _arr("kallus_weight_p99_grid")
    # Phase 11B passthrough (only if KallusMinimaxBridgeH was used)
    if ex.get("h_kallus_present", False):
        rec["h_method"]                  = ex.get("h_method", "?")
        rec["h_kallus_present"]          = True
        rec["h_raw_residual_mean"]       = _scl("h_raw_residual_mean")
        rec["h_weighted_residual_mean"]  = _scl("h_weighted_residual_mean")
        rec["h_inner_max_value_mean"]    = _scl("h_inner_max_value_mean")
        rec["h_beta_norm_mean"]          = _scl("h_beta_norm_mean")
    return rec
