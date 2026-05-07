"""
DGP 1 — Cobb-Douglas (log-linear)

Structural equations (all variables in log-space):

    log U  ~  N(0, sigma_U²)
    log Z  =  log U  +  eps_Z,   eps_Z ~ N(0, sigma_Z²)
    log W  =  log U  +  eps_W,   eps_W ~ N(0, sigma_W²)
    log A  =  gamma_K * log U  +  eta_A,  eta_A ~ N(0, sigma_K²)
    log X  ~  N(mu_L, sigma_L²)   [exogenous — no U in this equation]
    log Y  =  log U  +  alpha * log A  +  (1-alpha) * log X  +  eps_Y

Bridges (analytically derived from Fredholm integral equation):
    h₀(W, A, X) = W + alpha * A + (1-alpha) * X
    q₀(Z, A)    = sqrt(2pi * sigma_K^4 / D_Z) * exp[(A - gamma_K*Z)^2 / (2*D_Z)]
    where D_Z   = sigma_K^2 + gamma_K^2 * sigma_Z^2

True ATE:
    psi_0 = alpha = 0.30  (by construction, Cobb-Douglas elasticity of capital)
"""

from typing import Optional

import numpy as np

from pci.dgps.base import DGPSample, PCIDgp
from pci.dgps.utils import snr_to_sigma


# Default parameters (decision D2: parametrize by SNR, never by sigma directly)
DEFAULTS_CD = {
    "sigma_U": 1.0,    # reference scale, do NOT change (D2)
    "alpha": 0.3,      # capital elasticity = true ATE
    "gamma_K": 0.5,    # confounding strength (U -> A)
    "sigma_K": 0.5,    # noise in capital decision
    "mu_L": 1.0,       # mean of log labour
    "sigma_L": 0.3,    # std of log labour
    "sigma_Y": 0.1,    # outcome noise
    "snr_W": 0.80,     # SNR of outcome-inducing proxy W
    "snr_Z": 0.80,     # SNR of treatment-inducing proxy Z
}


class CobbDouglasLinearDGP(PCIDgp):
    """
    Cobb-Douglas log-linear DGP for proximal causal inference simulations.

    All DGPSample arrays are in log-space. The ATE is an elasticity:
    a unit increase in log A (log capital) increases log Y by alpha.

    Parameters match DEFAULTS_CD; SNR values determine proxy quality.
    """

    def __init__(
        self,
        sigma_U: float = DEFAULTS_CD["sigma_U"],
        alpha: float = DEFAULTS_CD["alpha"],
        gamma_K: float = DEFAULTS_CD["gamma_K"],
        sigma_K: float = DEFAULTS_CD["sigma_K"],
        mu_L: float = DEFAULTS_CD["mu_L"],
        sigma_L: float = DEFAULTS_CD["sigma_L"],
        sigma_Y: float = DEFAULTS_CD["sigma_Y"],
        snr_W: float = DEFAULTS_CD["snr_W"],
        snr_Z: float = DEFAULTS_CD["snr_Z"],
    ):
        self.sigma_U = sigma_U
        self.alpha = alpha
        self.gamma_K = gamma_K
        self.sigma_K = sigma_K
        self.mu_L = mu_L
        self.sigma_L = sigma_L
        self.sigma_Y = sigma_Y
        self._snr_W = snr_W
        self._snr_Z = snr_Z
        # Pre-compute proxy sigmas once — not at every generate() call
        self.sigma_W = snr_to_sigma(snr_W, sigma_U)
        self.sigma_Z = snr_to_sigma(snr_Z, sigma_U)

    # ── Core interface ──────────────────────────────────────────────────────

    def generate(self, n: int, seed: int) -> DGPSample:
        """
        Generate n observations. Fully vectorized, reproducible given seed.

        Generation order (structural, respects causal graph):
            1. log U  (latent confounder)
            2. log Z, log W  (proxies — depend on U only)
            3. log A  (treatment — depends on U)
            4. log X  (exogenous covariate — independent of U)
            5. log Y  (outcome — depends on U, A, X)
        """
        rng = np.random.default_rng(seed)

        U = rng.normal(0.0, self.sigma_U, n)                        # log U
        Z = U + rng.normal(0.0, self.sigma_Z, n)                    # log Z
        W = U + rng.normal(0.0, self.sigma_W, n)                    # log W
        A = self.gamma_K * U + rng.normal(0.0, self.sigma_K, n)     # log A
        X = rng.normal(self.mu_L, self.sigma_L, n)                  # log X (exog.)
        Y = (U
             + self.alpha * A
             + (1 - self.alpha) * X
             + rng.normal(0.0, self.sigma_Y, n))                     # log Y

        return DGPSample(
            Y=Y, A=A, X=X, Z=Z, W=W, U=U,
            psi_0=self.psi_true(a=1.0)
        )

    def h0(self, W: np.ndarray, A: np.ndarray,
           X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Outcome bridge — analytically derived from the Fredholm condition:
            E[Y - h₀(W, A, X) | Z, A, X] = 0

        Formula:  h₀(W, A, X) = W + alpha * A + (1-alpha) * X

        The coefficient of W is 1, NOT snr_W.
        Proof: W = U + eps_W with eps_W ⊥ (Z, A).
               Therefore E[W | Z, A] = E[U | Z, A].
               Substituting into the Fredholm equation, the attenuation factors
               on both sides cancel exactly, leaving coefficient = 1.
        """
        if X is None:
            return W + self.alpha * A
        return W + self.alpha * A + (1.0 - self.alpha) * X

    def q0(self, Z: np.ndarray, A: np.ndarray,
           X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Treatment bridge — analytically derived.

        Formula:  q₀(Z, A) = sqrt(2pi * sigma_K^4 / D_Z) * exp[(A - gamma_K*Z)^2 / (2*D_Z)]
        where     D_Z = sigma_K^2 + gamma_K^2 * sigma_Z^2

        Derivation: Solve E[q₀(Z, a) | U=u] = 1 / f(A=a | U=u) where
            f(A | U) = N(gamma_K * U, sigma_K^2).
        Then use the Bayesian decomposition
            E[q₀(Z, a) | W, A=a] = E[1/f(A=a|U) | W, A=a]
        to verify the treatment bridge condition.

        Does NOT depend on X (X is exogenous to both U and Z).
        """
        D_Z = self.sigma_K ** 2 + self.gamma_K ** 2 * self.sigma_Z ** 2
        prefactor = np.sqrt(2.0 * np.pi * self.sigma_K ** 4 / D_Z)
        return prefactor * np.exp((A - self.gamma_K * Z) ** 2 / (2.0 * D_Z))

    def psi_true(self, a: float) -> float:
        """
        True ATE = alpha = 0.30 (Cobb-Douglas capital elasticity).
        By construction of the log-linear model, a unit increase in log A
        increases E[log Y] by exactly alpha, regardless of U or X.

        NOTE: This is the SLOPE dm/da, not the dose-response level m(a).
        Use m_true(a) for the level.
        """
        return float(self.alpha)

    def m_true(self, a: float) -> float:
        """
        True dose-response m(a) = E[Y(a)] at dose a.

        Derivation:
            E[Y(a)] = E[U + alpha*a + (1-alpha)*X + eps_Y]
                   = 0 + alpha*a + (1-alpha)*mu_L + 0
                   = alpha*a + (1-alpha)*mu_L

        For linear DGP1 with centered kernel, J(pi_{a,h}) = m(a) exactly.
        """
        return float(self.alpha * a + (1.0 - self.alpha) * self.mu_L)

    # ── SNR properties ──────────────────────────────────────────────────────

    def snr_W(self) -> float:
        """Declared SNR of the outcome proxy W."""
        return self._snr_W

    def snr_Z(self) -> float:
        """Declared SNR of the treatment proxy Z."""
        return self._snr_Z
