"""
DRDoseResponse — Doubly-Robust estimator for kernel-localised policy values.

Estimand
--------
For a kernel-localised stochastic intervention pi_{a,h}(t) = K_h(t - a),
the policy value is

    J(pi_{a,h}) = ∫ m(t) K_h(t - a) dt

where m(a) = E[Y(a)] is the dose-response function.

Identification (Kallus 2021, Lemma 2):
    J(pi_{a,h}) = E[ ∫ h₀(W, t, X) K_h(t - a) dt ]

For DGP1 (linear h, centered kernel): J(pi_{a,h}) = m(a) exactly.
For DGP2 (concave m): J(pi_{a,h}) <= m(a) by Jensen.

Formula (Version B, NON-NEGOTIABLE)
-----------------------------------
    J_DR(pi_{a,h}) = (1/n) Σᵢ [
        K_h(Aᵢ - a) · q̂(Zᵢ, Aᵢ, Xᵢ) · (Yᵢ - ĥ(Wᵢ, Aᵢ, Xᵢ))
        + ∫ ĥ(Wᵢ, t, Xᵢ) · K_h(t - a) dt
    ]

Three things that MUST be right:
    1. q̂ is evaluated at the OBSERVED treatment Aᵢ (NOT at target dose a).
       Using q̂(Zᵢ, a, Xᵢ) is "Version A" — DR property breaks at fixed h.
    2. Residual is (Yᵢ - ĥ(Wᵢ, Aᵢ, Xᵢ)) — observed Aᵢ in the bridge.
    3. Plug-in is ∫ ĥ(Wᵢ, t, Xᵢ) K_h(t - a) dt — the policy integral.
       For DGP1 (linear h), this equals ĥ(Wᵢ, a, Xᵢ).
       For DGP2 (nonlinear h), numerical quadrature is required.

Double robustness at fixed h
----------------------------
    Bias = -E[ K_h(A - a) · (q̂ - q₀)(Z, A, X) · (ĥ - h₀)(W, A, X) ]

If ĥ = h₀ (regardless of q̂) → bias = 0.
If q̂ = q₀ (regardless of ĥ) → bias = 0.

Reference
---------
Kallus, Mao, Uehara (2021), "Causal inference under unmeasured confounding
with negative controls", Lemma 2 + Section 4.
Cui, Pu, Shi, Miao, Tchetgen (2023), JASA — EIF (binary case),
adapted for continuous A via Version B.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from pci.dgps.base import DGPSample
from pci.estimators.base import EstimationResult, PCIEstimator


class DRDoseResponse(PCIEstimator):
    """
    DR estimator for the policy value J(pi_{a,h}) on a grid of dose values.

    Constructor
    -----------
    a_grid    : 1D array of dose values to evaluate
    h_model   : object with predict(W, A, X) -> array — outcome bridge
    q_model   : object with predict(Z, A, X) -> array — treatment bridge
    bandwidth : optional float; default = Silverman 1.06 * std(A) * n^(-1/5)
    kernel    : 'gaussian' or 'epanechnikov'

    Output
    ------
    EstimationResult with psi_hat=NaN (no scalar estimand) and a rich `extra`:
        J_reg         : (K,) plug-in regression curve E_n[h(W, a_k, X)]
        J_dr          : (K,) DR estimate of J(pi_{a_k, h})
        J_true        : (K,) ground truth m(a_k) if m_true_fn provided, else NaN
        V_hat_grid    : (K,) per-grid empirical influence-function variance
        a_grid        : (K,) the grid passed in
        bandwidth     : float, the bandwidth actually used
        alpha_derived : float, OLS slope of J_dr on a_grid (DGP1 diagnostic)
        mise          : float, mean integrated squared error vs J_true
    """

    def __init__(
        self,
        a_grid: np.ndarray,
        h_model,
        q_model,
        bandwidth: Optional[float] = None,
        kernel: str = "gaussian",
        n_quad: int = 25,
    ) -> None:
        self.a_grid = np.asarray(a_grid, dtype=float)
        self.h_model = h_model
        self.q_model = q_model
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.n_quad = n_quad
        self._result: Optional[EstimationResult] = None

    @property
    def name(self) -> str:
        return "DRDoseResponse"

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _kernel(self, u: np.ndarray, h: float) -> np.ndarray:
        """K_h(u) = K(u/h) / h."""
        if self.kernel == "gaussian":
            return np.exp(-0.5 * (u / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
        if self.kernel == "epanechnikov":
            v = u / h
            return np.where(np.abs(v) <= 1.0, 0.75 * (1.0 - v ** 2) / h, 0.0)
        raise ValueError(f"Unknown kernel: {self.kernel!r}")

    def _policy_integral(
        self,
        W: np.ndarray,
        X: np.ndarray,
        a_k: float,
        h: float,
    ) -> np.ndarray:
        """
        Compute ∫ ĥ(W_i, t, X_i) K_h(t - a_k) dt for each i.

        DGP1 shortcut (linear h, centered kernel):
            When h_model.is_linear_in_A is True, the integral reduces exactly to
            ĥ(W_i, a_k, X_i) by the moment property of centered kernels.

        DGP2 / nonlinear h — trapezoidal quadrature:
            Integration over t ∈ [a_k − 4h, a_k + 4h] with self.n_quad points.
            K_h is a gaussian kernel; integrand is near-zero outside ±4h.
            Accuracy: O((h/n_quad)²) for smooth integrands — n_quad=25 gives
            ~1e-4 relative error for the smooth MM dose-response function.
        """
        n = len(W)
        a_vec = np.full(n, a_k, dtype=float)

        # ── Priority 1: RKHS plug-in (KPVBridgeH) ─────────────────────────────
        # KPVBridgeH exposes plug_in_policy(W, a, h) using closed-form Gauss-Gauss
        # convolution. It does NOT take X (Phase 8 DGPs don't use X in bridges).
        if getattr(self.h_model, "has_rkhs_plug_in", False):
            return self.h_model.plug_in_policy(W, a_k, h)

        # ── Priority 2: Shortcut for linear h (DGP1, OracleBridgeH linear) ────
        if getattr(self.h_model, "is_linear_in_A", False):
            return self.h_model.predict(W, a_vec, X)

        # ── Priority 3: Trapezoidal quadrature (generic nonlinear h) ──────────
        t_grid = np.linspace(a_k - 4.0 * h, a_k + 4.0 * h, self.n_quad)
        k_weights = self._kernel(t_grid - a_k, h)
        dt = (t_grid[-1] - t_grid[0]) / (self.n_quad - 1)

        result = np.zeros(n)
        for t_j, k_j in zip(t_grid, k_weights):
            t_vec = np.full(n, t_j, dtype=float)
            result += k_j * self.h_model.predict(W, t_vec, X)
        return result * dt

    # ── Main entry point ─────────────────────────────────────────────────────

    def fit(
        self,
        sample: DGPSample,
        m_true_fn: Optional[Callable[[float], float]] = None,
    ) -> "DRDoseResponse":
        """
        Fit the DR estimator on `sample`.

        Parameters
        ----------
        sample    : DGPSample
        m_true_fn : optional callable a -> float for ground-truth dose response.
                    If provided, J_true and MISE are populated; otherwise NaN.
        """
        n = len(sample.Y)
        Y, A, W, Z = sample.Y, sample.A, sample.W, sample.Z
        X = sample.X if sample.X is not None else np.zeros(n)

        # Bandwidth — Silverman default
        h = self.bandwidth if self.bandwidth is not None \
            else 1.06 * float(np.std(A)) * n ** (-1.0 / 5.0)

        # ── Evaluate nuisances ONCE at the OBSERVED treatment Aᵢ ─────────────
        # ⚠ NON-NEGOTIABLE: q is evaluated at Aᵢ (Version B), NOT at a_k.
        h_obs = self.h_model.predict(W, A, X)   # ĥ(Wᵢ, Aᵢ, Xᵢ)
        q_obs = self.q_model.predict(Z, A, X)   # q̂(Zᵢ, Aᵢ, Xᵢ)

        K = len(self.a_grid)
        J_reg = np.zeros(K)
        J_dr = np.zeros(K)
        V_hat_grid = np.zeros(K)

        for k, a_k in enumerate(self.a_grid):
            # Kernel weights at target dose
            w_k = self._kernel(A - a_k, h)

            # Plug-in: ∫ ĥ(W, t, X) K_h(t - a_k) dt
            # For DGP1 linear h: equals ĥ(W, a_k, X)
            h_policy = self._policy_integral(W, X, a_k, h)

            # REG (plug-in only, no correction)
            J_reg[k] = float(np.mean(h_policy))

            # DR correction: K_h(A - a_k) * q(Z, A, X) * (Y - h(W, A, X))
            correction = w_k * q_obs * (Y - h_obs)

            # Per-observation score
            scores = correction + h_policy
            J_dr[k] = float(np.mean(scores))

            # Empirical variance of the influence function
            phi = scores - J_dr[k]
            V_hat_grid[k] = float(np.mean(phi ** 2))

        # Ground truth and derived diagnostics
        if m_true_fn is not None:
            J_true = np.array([float(m_true_fn(float(a))) for a in self.a_grid])
            mise = float(np.mean((J_dr - J_true) ** 2))
        else:
            J_true = np.full(K, np.nan)
            mise = float("nan")

        # alpha_derived = OLS slope of J_dr on a_grid (DGP1 diagnostic)
        alpha_derived = float(np.polyfit(self.a_grid, J_dr, 1)[0])

        self._result = EstimationResult(
            psi_hat=float("nan"),                  # no scalar estimand
            V_hat=float(np.mean(V_hat_grid)),      # average IF variance
            method_name=self.name,
            n=n,
            extra={
                "J_reg": J_reg,
                "J_dr": J_dr,
                "J_true": J_true,
                "V_hat_grid": V_hat_grid,
                "a_grid": self.a_grid,
                "bandwidth": h,
                "alpha_derived": alpha_derived,
                "mise": mise,
            },
        )
        return self

    def estimate(self) -> EstimationResult:
        if self._result is None:
            raise RuntimeError("Call fit() before estimate().")
        return self._result
