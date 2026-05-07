"""
pci.runner.bandwidth_helpers
============================

Bandwidth-selection helpers used by the dose-response estimators (KDE policy
weights for J(pi_a) integration) and by the kernel-based bridges (RBF
length-scale defaults).

Three small utilities, all stateless and dependency-free:

* :func:`_silverman_h`         Silverman's rule of thumb for univariate KDE
* :func:`_median_bandwidth`    Median heuristic for RBF length-scale
* :func:`_compute_J_policy_true`  Gauss-Hermite oracle for J(pi_a) given a DGP

These were extracted from
``simulations/experiments/dgp2_bias_diagnostics.py`` where
they had accumulated for historical reasons. The original module continues to
re-export them as a backward-compatibility shim.

Naming preserves the underscore prefix (``_silverman_h`` etc.) since these
were already used by tests and other internal scripts under that name; a
follow-up PR may drop the underscore.
"""
from __future__ import annotations

import numpy as np


def _silverman_h(A: np.ndarray) -> float:
    """Silverman's rule of thumb bandwidth for univariate KDE.

    h = 1.06 * sigma_A * n^{-1/5}
    """
    return 1.06 * float(np.std(A)) * len(A) ** (-0.2)


def _compute_J_policy_true(
    dgp, a_grid: np.ndarray, h: float, n_quad: int = 25
) -> np.ndarray:
    """J_policy_true[k] = integral of m(t) K_h(t-a_k) dt via Gauss-Hermite.

    Used as the oracle target for evaluating policy-functional estimators on
    a DGP whose ``m_true(a)`` is available analytically.

    Parameters
    ----------
    dgp : object
        Must expose a callable ``m_true(a) -> float`` returning E[Y(a)] on
        the *same scale* as the DGP outcome.
    a_grid : np.ndarray
        Dose grid at which to evaluate J_pi_a.
    h : float
        KDE bandwidth (typically Silverman ROT or a Stage-tuned value).
    n_quad : int
        Number of Gauss-Hermite nodes (default 25 — abscissae are accurate
        to machine precision for smooth integrands).
    """
    from numpy.polynomial.hermite_e import hermegauss
    x_q, w_q = hermegauss(n_quad)
    J_pt = np.zeros(len(a_grid))
    for d, a_d in enumerate(a_grid):
        t_vals = float(a_d) + h * x_q
        vals = np.array([float(dgp.m_true(float(t))) for t in t_vals])
        J_pt[d] = float(np.dot(w_q, vals) / np.sqrt(2.0 * np.pi))
    return J_pt


def _median_bandwidth(x: np.ndarray) -> float:
    """Median absolute deviation heuristic for RBF length-scale.

    Returns the median pairwise distance between values of ``x``, with a
    sub-sample to 500 points when the input is larger (to keep the cost
    quadratic-but-bounded). Falls back to 1.0 when all values coincide
    (degenerate; see DGP1 with sigma_A -> 0).
    """
    if len(x) > 500:
        rng = np.random.default_rng(0)
        x = rng.choice(x, 500, replace=False)
    diffs = np.abs(x[:, None] - x[None, :])
    med = float(np.median(diffs[diffs > 0]))
    return med if med > 1e-12 else 1.0
