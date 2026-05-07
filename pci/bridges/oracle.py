"""
pci.bridges.oracle
==================

Oracle and intentionally-wrong bridge wrappers for ``DRDoseResponse``.

These are NOT estimators in their own right. They expose a ``predict`` method
that ``DRDoseResponse`` can call without estimating bridges from data — used to
test the DR formula in isolation (oracle mode) and to verify the double
robustness property (one bridge correct, the other wrong).

Four classes:

* :class:`OracleBridgeH`        h(W,A,X) = analytic h_0 from DGP
* :class:`OracleBridgeQ`        q(Z,A,X) = analytic q_0 from DGP
* :class:`WrongBridgeH_DropsW`  intentionally drops W (violates Fredholm)
* :class:`WrongBridgeQ_Constant` constant 1 everywhere

Extracted from ``pci/estimators/_oracle_bridges.py``.
The original module continues to re-export these as a backward-compatibility
shim with class identity preserved.
"""

from __future__ import annotations

import numpy as np

from pci.dgps.base import PCIDgp


# ── Oracle bridges (use the analytic h₀ / q₀ from the DGP) ───────────────────

class OracleBridgeH:
    """
    h(W, A, X) = analytic h₀ from the DGP — for testing only.

    Parameters
    ----------
    dgp : PCIDgp
    is_linear_in_A : bool
        Set to True when dgp.h0 is linear in A (e.g., CobbDouglasLinearDGP).
        DRDoseResponse uses this flag to apply the exact shortcut
            ∫ h(W,t,X) K_h(t-a) dt = h(W, a, X)
        instead of numerical quadrature. For nonlinear h (e.g., KRR in
        MichaelisMentenDGP), leave False — quadrature is used automatically.
    has_rkhs_plug_in : bool
        Always False for the analytic oracle. Reserved here so that
        DRDoseResponse._policy_integral can dispatch on a single attribute
        regardless of whether the bridge is oracle, KPV, or other.
    """

    def __init__(
        self,
        dgp: PCIDgp,
        is_linear_in_A: bool = False,
        has_rkhs_plug_in: bool = False,
    ) -> None:
        self.dgp = dgp
        self.is_linear_in_A = is_linear_in_A
        self.has_rkhs_plug_in = has_rkhs_plug_in

    def predict(self, W: np.ndarray, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self.dgp.h0(W, A, X)


class OracleBridgeQ:
    """q(Z, A, X) = analytic q₀ from the DGP — for testing only."""

    def __init__(self, dgp: PCIDgp) -> None:
        self.dgp = dgp

    def predict(self, Z: np.ndarray, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self.dgp.q0(Z, A, X)


# ── Wrong bridges (intentionally misspecified — for DR robustness tests) ─────

class WrongBridgeH_DropsW:
    """
    Misspecified h-bridge: drops the proxy W entirely.

        h_wrong(W, A, X) = alpha * A + (1 - alpha) * X

    Equivalent to ignoring the unmeasured-confounding correction.
    The Fredholm condition is violated: E[Y - h_wrong | Z, A, X] = E[U | Z, A] ≠ 0.
    """

    def __init__(self, dgp: PCIDgp) -> None:
        self.dgp = dgp

    def predict(self, W: np.ndarray, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self.dgp.alpha * A + (1.0 - self.dgp.alpha) * X


class WrongBridgeQ_Constant:
    """
    Misspecified q-bridge: returns the constant 1 everywhere.

    Removes the proxy-treatment correction entirely. Under Version B
    of the DR formula, this means the IPW-style correction term reduces to
    a kernel-weighted average of (Y - h), which only cancels the bias
    when h is correctly specified.
    """

    def predict(self, Z: np.ndarray, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.ones_like(A, dtype=float)
