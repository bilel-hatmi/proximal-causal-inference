"""
Two-Stage Least Squares estimator for Proximal Causal Inference.

Stage 1 : OLS  W ~ [1, A, X, Z]  →  Ŵ  (bridge outcome approximation)
Stage 2 : OLS  Y ~ [1, A, X, Ŵ]  →  β̂  (ATE = β̂[A])

The standard-error formula propagates first-stage uncertainty through the
"generated regressor" correction term in the Jacobian cross-block J₂₁.
Ignoring J₂₁ (treating Ŵ as fixed) gives SEs that are systematically
too small — this is the Pagan (1984) generated-regressor bias in SEs.

Reference implementation : KenLi93/pci2s (R), function p2sls.lm / pcilm.R.
The sandwich formula here is a direct Python port of that logic.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pci.dgps.base import DGPSample
from pci.estimators.base import EstimationResult, PCIEstimator


# ── Sandwich helper (module-level → unit-testable independently) ──────────────

def _sandwich_2stage(
    Xw: np.ndarray,      # (n, p1)  stage-1 design matrix
    S2: np.ndarray,      # (n, p2)  stage-2 design matrix  [last col = Ŵ]
    beta: np.ndarray,    # (p2,)    stage-2 OLS coefficients
    resid1: np.ndarray,  # (n,)     W − Ŵ
    resid2: np.ndarray,  # (n,)     Y − Ŷ
) -> np.ndarray:
    """
    Full two-stage sandwich: returns Var(θ̂) as a (p1+p2, p1+p2) matrix.

    Parameter layout: θ = (γ₀..γ_{p1-1}, β₀..β_{p2-1})
    Accounts for the generated regressor Ŵ = Xw @ γ entering S2.

    Key identity
    ------------
    J₂₁ ≠ 0 : the second-stage score depends on γ through Ŵ.
    Specifically (derivation via chain rule):
        J₂₁ = −β_W · (S2ᵀ Xw)
        J₂₁[-1, :] += Xwᵀ ε₂          (correction for the Ŵ row)
    where β_W = beta[-1] = coefficient of Ŵ in stage 2.
    """
    p1 = Xw.shape[1]
    p2 = S2.shape[1]
    p  = p1 + p2

    # ── Jacobian blocks (all are sums, not means) ─────────────────────────────
    J11 = -(Xw.T @ Xw)                 # (p1, p1)
    J12 = np.zeros((p1, p2))           # (p1, p2) — stage-1 score ⟂ β
    J22 = -(S2.T @ S2)                 # (p2, p2)

    beta_W = float(beta[-1])           # coeff of Ŵ (last column of S2)
    J21 = -beta_W * (S2.T @ Xw)       # (p2, p1) — base term
    J21[-1, :] += Xw.T @ resid2       # (p1,) added to last row of J21

    J = np.block([[J11, J12],
                  [J21, J22]])          # (p, p)

    # ── Meat (sum of outer products of stacked scores) ────────────────────────
    U1 = Xw * resid1[:, None]          # (n, p1)
    U2 = S2 * resid2[:, None]          # (n, p2)
    U  = np.hstack([U1, U2])           # (n, p)
    M  = U.T @ U                       # (p, p)

    # ── Sandwich: Var(θ̂) = J⁻¹ M (J⁻¹)ᵀ ────────────────────────────────────
    J_inv = np.linalg.solve(J, np.eye(p))
    return J_inv @ M @ J_inv.T


# ── Estimator class ───────────────────────────────────────────────────────────

class TwoStageLeastSquares(PCIEstimator):
    """
    Two-stage least squares PCI estimator.

    Stage 1 : OLS  W ~ [1, A, X, Z]         →  Ŵ
    Stage 2 : OLS  Y ~ [1, A, X, Ŵ]         →  β̂
    ATE     : β̂[1]  (coefficient of A in stage 2)
    SE      : full two-stage sandwich (generated-regressor correction)

    V_hat convention (matches Oracle/Naive):
        V_hat = se_A² · n    so that    SE = sqrt(V_hat / n).

    Extras reported in EstimationResult.extra
    -----------------------------------------
    beta_W    : coefficient of Ŵ in stage 2  (should be ≈ 1 for DGP1)
    gamma_Z   : coefficient of Z in stage 1  (instrument strength)
    t_Z       : first-stage t-statistic for Z  (weak-instrument check)
    stage1_F  : t_Z² (partial F-statistic for the single instrument Z)
    """

    def __init__(self) -> None:
        self._result: Optional[EstimationResult] = None

    @property
    def name(self) -> str:
        return "TwoStageLeastSquares"

    def fit(self, sample: DGPSample) -> "TwoStageLeastSquares":
        n = len(sample.Y)
        Y = sample.Y
        A = sample.A
        W = sample.W
        Z = sample.Z
        X = sample.X if sample.X is not None else np.zeros(n)

        ones = np.ones(n)

        # ── Stage 1 : W ~ [1, A, X, Z] ───────────────────────────────────────
        Xw = np.column_stack([ones, A, X, Z])                    # (n, 4)
        gamma, _, _, _ = np.linalg.lstsq(Xw, W, rcond=None)
        W_hat  = Xw @ gamma
        resid1 = W - W_hat

        # ── Stage 2 : Y ~ [1, A, X, Ŵ] ──────────────────────────────────────
        S2 = np.column_stack([ones, A, X, W_hat])                # (n, 4)
        beta, _, _, _ = np.linalg.lstsq(S2, Y, rcond=None)
        resid2 = Y - S2 @ beta

        psi_hat = float(beta[1])   # ATE = coefficient of A

        # ── Two-stage sandwich SE ─────────────────────────────────────────────
        # theta = [gamma(4), beta(4)]
        # idx of beta_A in theta = p1 + 1 = 4 + 1 = 5
        V_full = _sandwich_2stage(Xw, S2, beta, resid1, resid2)
        idx_A  = Xw.shape[1] + 1
        V_hat  = float(V_full[idx_A, idx_A] * n)

        # ── First-stage diagnostics ───────────────────────────────────────────
        dof1     = n - Xw.shape[1]
        sigma2_1 = float(np.dot(resid1, resid1) / dof1)
        XwTXw_inv = np.linalg.inv(Xw.T @ Xw)
        se_gZ = float(np.sqrt(max(sigma2_1 * XwTXw_inv[3, 3], 0.0)))
        t_Z   = float(gamma[3] / se_gZ) if se_gZ > 1e-15 else np.nan
        F_stat = t_Z ** 2 if np.isfinite(t_Z) else np.nan

        self._result = EstimationResult(
            psi_hat=psi_hat,
            V_hat=V_hat,
            method_name=self.name,
            n=n,
            extra={
                "beta_W":   float(beta[3]),
                "gamma_Z":  float(gamma[3]),
                "t_Z":      t_Z,
                "stage1_F": F_stat,
            },
        )
        return self

    def estimate(self) -> EstimationResult:
        if self._result is None:
            raise RuntimeError("Call fit() before estimate().")
        return self._result
