"""
pci.inference.variance
======================

Variance decomposition utilities for the doubly-robust score
::

    psi_hat_i = h_policy_i + r_correction_i * (Y_i - h_obs_i)

Used by the three DR estimators (``BennettIndepFunctionalDR``,
``BennettFunctionalDR``, ``DRKernel``) to populate the ``V_total / V_reg
/ V_corr / V_cov`` scalars and the ``V_hat_grid / V_reg_grid /
V_correction_grid`` per-dose arrays in their ``EstimationResult.extra``.

Two conventions co-exist in the existing code, deliberately preserved
here as ``ddof`` and ``with_scalars`` keyword arguments:

* **Bennett family** uses ``ddof=1`` (sample variance) and reports
  scalar summaries ``V_total = mean(V_hat_grid)``, etc., plus the
  cross-product term ``V_cov = (V_total - V_reg - V_corr) / 2`` (an
  identity from the polarisation rule for the variance of a sum).

* **DRKernel** uses ``ddof=0`` (biased estimator) and only reports
  per-dose grids.

The functions below are pure: ``score``, ``h_policy``, ``correction``
arrays in, dictionary of variance summaries out. **No ``V_hat`` is
populated** here — callers pick ``V_hat_grid[ref_dose_index]`` after the
fact (see :mod:`pci.estimators.bennett_indep_dr`).

extracted from inline code in the three DR estimator
modules. Existing estimators continue to compute these inline; the
extracted helpers are validated against the inline code by bit-identity
tests in ``pci/tests/test_inference.py``. the estimators estimators to call these helpers in their ``estimate()`` methods.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def variance_decomposition(
    score: np.ndarray,
    h_policy: np.ndarray,
    correction: np.ndarray,
    *,
    ddof: int = 1,
    with_scalars: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute the per-dose variance decomposition of the DR score.

    Parameters
    ----------
    score : ndarray, shape (n, K)
        DR score per observation per dose: ``h_policy + correction``.
    h_policy : ndarray, shape (n, K)
        Plug-in (regression) component per observation per dose.
    correction : ndarray, shape (n, K)
        IPW (Riesz) correction component per observation per dose.
    ddof : {0, 1}, default 1
        Delta degrees of freedom for ``np.var``. Bennett family uses 1
        (sample variance); DRKernel uses 0.
    with_scalars : bool, default True
        If True, also return scalar summaries ``V_total``, ``V_reg``,
        ``V_corr``, ``V_cov`` (Bennett convention). DRKernel passes
        ``with_scalars=False``.

    Returns
    -------
    dict
        Always contains:
            * ``V_hat_grid``    (K,) ``np.var(score, axis=0, ddof=ddof)``
            * ``V_reg_grid``    (K,) ``np.var(h_policy, axis=0, ddof=ddof)``
            * ``V_correction_grid``  (K,) ``np.var(correction, axis=0, ddof=ddof)``
        If ``with_scalars`` is True, also:
            * ``V_total``   mean over doses of V_hat_grid
            * ``V_reg``     mean over doses of V_reg_grid
            * ``V_corr``    mean over doses of V_correction_grid
            * ``V_cov``     ``(V_total - V_reg - V_corr) / 2``

    Notes
    -----
    The cross-product identity ``V_cov = (V_total - V_reg - V_corr)/2``
    holds because ``Var(A + B) = Var(A) + Var(B) + 2 Cov(A, B)``.
    """
    score = np.asarray(score)
    h_policy = np.asarray(h_policy)
    correction = np.asarray(correction)
    if score.shape != h_policy.shape or score.shape != correction.shape:
        raise ValueError(
            f"shape mismatch: score={score.shape}, h_policy={h_policy.shape}, "
            f"correction={correction.shape}"
        )
    if score.ndim != 2:
        raise ValueError(f"score must be 2-D (n, K); got shape {score.shape}")

    V_hat_grid = np.var(score, axis=0, ddof=ddof)
    V_reg_grid = np.var(h_policy, axis=0, ddof=ddof)
    V_correction_grid = np.var(correction, axis=0, ddof=ddof)

    out: Dict[str, np.ndarray] = {
        "V_hat_grid": V_hat_grid,
        "V_reg_grid": V_reg_grid,
        "V_correction_grid": V_correction_grid,
    }
    if with_scalars:
        V_total = float(np.mean(V_hat_grid))
        V_reg = float(np.mean(V_reg_grid))
        V_corr = float(np.mean(V_correction_grid))
        V_cov = (V_total - V_reg - V_corr) / 2.0
        out["V_total"] = V_total
        out["V_reg"] = V_reg
        out["V_corr"] = V_corr
        out["V_cov"] = V_cov
    return out


def variance_decomposition_per_dose(
    scores_per_dose: np.ndarray,
    plug_in_per_dose: np.ndarray,
    correction_per_dose: np.ndarray,
    *,
    ddof: int = 0,
) -> Dict[str, np.ndarray]:
    """Per-dose variant matching the DRKernel inner loop.

    DRKernel computes variances inside a ``for d, a_d in enumerate(a_grid)``
    loop, with ``V_hat_grid[d] = mean((scores_d - J_dr[d])**2)``. That
    expression equals ``np.var(scores_d, ddof=0)`` because ``J_dr[d] =
    mean(scores_d)``. This helper mirrors that convention exactly.

    Parameters
    ----------
    scores_per_dose, plug_in_per_dose, correction_per_dose
        Arrays of shape ``(n, K)`` -- the per-observation, per-dose
        components of the DR score, plug-in, and correction.
    ddof : int, default 0 (DRKernel convention)
    """
    return variance_decomposition(
        scores_per_dose, plug_in_per_dose, correction_per_dose,
        ddof=ddof, with_scalars=False,
    )
