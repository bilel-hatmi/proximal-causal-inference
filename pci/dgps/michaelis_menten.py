"""
DGP 2 — Michaelis-Menten (non-linear)

Structural equations (raw space — not log-space like DGP1):

    U     ~  LogNormal(0, sigma_U²)            [latent confounder, strictly positive]
    Z     =  U + eps_Z,     eps_Z ~ N(0, sigma_Z²)   [treatment-inducing proxy / NCE]
    W     =  U + eps_W,     eps_W ~ N(0, sigma_W²)   [outcome-inducing proxy  / NCO]
    A     =  base_dose + sensitivity * U + eta_A,     eta_A ~ N(0, sigma_A²)
    Y     =  (Vmax * U * A) / (Km * U + A) + eps_Y,  eps_Y ~ N(0, sigma_Y²)

Key properties:
- A depends on U (confounding), NOT on Z directly. Z is a proxy of U (NCE).
  This maintains the NCE/NCO symmetry of the PCI framework.
- Y is non-linear in (U, A). m_true(a) = E[Y(a)] has no closed form;
  computed via Gauss-Hermite quadrature.
- m_true is strictly concave in a → J(pi_{a,h}) <= m_true(a) by Jensen.

Bridge functions:
    h0(W, A): regression oracle E[mu(U,A)|W,A], approximated by kernel ridge.
              NOTE: this is NOT the Fredholm bridge (differs from h0_fredholm by
              the attenuation factor, same issue as snr_W*W vs W in DGP1).
              Exact Fredholm bridge deferred to Phase 8 (DRKernel).
              Consequence: T5b (Fredholm condition) is marked xfail for DGP2.

    q0(Z, A): same Gaussian form as DGP1, with sigma_A <-> sigma_K, sensitivity <-> gamma_K.
              Formula: C * exp[(A - base_dose - sensitivity*Z)^2 / (2*D_Z)]
              where D_Z = sigma_A^2 + sensitivity^2 * sigma_Z^2
              Convergence: D_Z > sensitivity^2 * sigma_Z^2 always holds. ✓

Proxy SNR calibration:
    For LogNormal U, Var(U) = exp(sigma_U^2) * (exp(sigma_U^2) - 1).
    sigma_W = sqrt(Var(U) * (1/snr_W - 1)) — ensures snr_empirical matches declared.
    (Differs from snr_to_sigma which assumes Gaussian U.)
"""

from typing import Optional

import numpy as np

from pci.dgps.base import DGPSample, PCIDgp


# Default parameters (decision D2: parametrize by SNR, never by sigma directly)
DEFAULTS_MM = {
    "sigma_U":     1.0,   # reference scale for LogNormal confounder
    "Vmax":        2.0,   # maximum reaction velocity
    "Km":          1.0,   # Michaelis constant (half-saturation)
    "sensitivity": 0.3,   # confounding strength U -> A
    "sigma_A":     0.6,   # noise in A (chosen so Cor(U,A) ≈ 0.73 with LogNormal U)
    "base_dose":   2.0,   # intercept in A equation (ensures A > 0 with high prob)
    "sigma_Y":     0.05,  # outcome noise
    "snr_W":       0.80,  # SNR of outcome-inducing proxy W
    "snr_Z":       0.80,  # SNR of treatment-inducing proxy Z
}


def _lognormal_var(sigma_U: float) -> float:
    """Var(U) for U ~ LogNormal(0, sigma_U^2)."""
    s2 = sigma_U ** 2
    return float(np.exp(s2) * (np.exp(s2) - 1.0))


def _snr_to_sigma_lognormal(snr: float, sigma_U: float) -> float:
    """
    Proxy noise sigma such that Var(U) / (Var(U) + sigma^2) = snr,
    for U ~ LogNormal(0, sigma_U^2).

    Unlike snr_to_sigma() (which assumes Gaussian U), this accounts for the
    true variance of the LogNormal distribution.
    """
    if not (0.0 < snr < 1.0):
        raise ValueError(f"SNR must be in (0, 1), got {snr}")
    var_U = _lognormal_var(sigma_U)
    return float(np.sqrt(var_U * (1.0 / snr - 1.0)))


class MichaelisMentenDGP(PCIDgp):
    """
    Michaelis-Menten non-linear DGP for PCI simulations.

    U is LogNormal (strictly positive), A depends linearly on U (confounding),
    Y is the MM functional form — strictly concave in A for fixed U > 0.
    m_true(a) = E[Y(a)] computed via Gauss-Hermite quadrature over U.
    h0 is a regression oracle (E[mu(U,A)|W,A] via kernel ridge on N=5k).
    """

    def __init__(
        self,
        sigma_U:     float = DEFAULTS_MM["sigma_U"],
        Vmax:        float = DEFAULTS_MM["Vmax"],
        Km:          float = DEFAULTS_MM["Km"],
        sensitivity: float = DEFAULTS_MM["sensitivity"],
        sigma_A:     float = DEFAULTS_MM["sigma_A"],
        base_dose:   float = DEFAULTS_MM["base_dose"],
        sigma_Y:     float = DEFAULTS_MM["sigma_Y"],
        snr_W:       float = DEFAULTS_MM["snr_W"],
        snr_Z:       float = DEFAULTS_MM["snr_Z"],
    ):
        self.sigma_U     = sigma_U
        self.Vmax        = Vmax
        self.Km          = Km
        self.sensitivity = sensitivity
        self.sigma_A     = sigma_A
        self.base_dose   = base_dose
        self.sigma_Y     = sigma_Y
        self._snr_W      = snr_W
        self._snr_Z      = snr_Z

        # Pre-compute proxy sigmas from LogNormal SNR formula
        self.sigma_W = _snr_to_sigma_lognormal(snr_W, sigma_U)
        self.sigma_Z = _snr_to_sigma_lognormal(snr_Z, sigma_U)

        # Sanity check: P(A < 0) is negligible only if base_dose >> sigma_A
        assert self.base_dose > 3.0 * self.sigma_A, (
            f"Risk of A<0: base_dose={self.base_dose:.2f} must be > "
            f"3*sigma_A={3*self.sigma_A:.2f}. Increase base_dose or reduce sigma_A."
        )

        # D_Z = sigma_A^2 + sensitivity^2 * sigma_Z^2 — used in q0
        # Convergence of bridge integral requires D_Z > sensitivity^2 * sigma_Z^2
        # (always satisfied by construction since sigma_A^2 > 0)
        self._D_Z = self.sigma_A ** 2 + self.sensitivity ** 2 * self.sigma_Z ** 2

        # Lazy-loaded kernel ridge model for h0
        self._h0_model = None

    # ── Core interface ──────────────────────────────────────────────────────

    def generate(self, n: int, seed: int) -> DGPSample:
        """
        Generate n observations. Fully vectorized, reproducible given seed.

        Generation order (causal):
            1. U  ~ LogNormal(0, sigma_U^2)
            2. Z, W  (proxies of U)
            3. A  = base_dose + sensitivity*U + eta_A   (confounded by U)
            4. Y  = MM(U, A) + eps_Y
        No exogenous covariates X (None).
        """
        rng = np.random.default_rng(seed)

        U   = rng.lognormal(mean=0.0, sigma=self.sigma_U, size=n)
        Z   = U + rng.normal(0.0, self.sigma_Z, n)
        W   = U + rng.normal(0.0, self.sigma_W, n)
        A   = (self.base_dose
               + self.sensitivity * U
               + rng.normal(0.0, self.sigma_A, n))
        Y   = ((self.Vmax * U * A) / (self.Km * U + A)
               + rng.normal(0.0, self.sigma_Y, n))

        return DGPSample(
            Y=Y, A=A, X=None, Z=Z, W=W, U=U,
            psi_0=self.psi_true(a=1.0),
        )

    def m_true(self, a: float) -> float:
        """
        True dose-response m(a) = E[Y(a)] = E[(Vmax * U * a) / (Km * U + a)]
        where U ~ LogNormal(0, sigma_U^2).

        Computed via Gauss-Hermite quadrature with the lognormal transform:
            U = exp(sigma_U * z),  z ~ N(0,1)
            m(a) = (1/sqrt(2pi)) * sum_j w_j * f(exp(sigma_U * x_j))

        ⚠ CRITICAL: hermite_e.hermegauss gives weights for exp(-x^2/2) dx,
        NOT for the standard normal density. The factor 1/sqrt(2pi) is mandatory.

        Degree 25 gives ~1e-8 precision for smooth integrands.
        """
        from numpy.polynomial.hermite_e import hermegauss  # probabilist's Hermite
        a = float(a)
        if a <= 0.0:
            return 0.0   # mu(U, 0) = 0 exactly for all U
        x, w = hermegauss(25)
        u_vals = np.exp(self.sigma_U * x)       # U = exp(sigma_U * z)
        integrand = (self.Vmax * u_vals * a) / (self.Km * u_vals + a)
        return float(np.dot(w, integrand) / np.sqrt(2.0 * np.pi))   # 1/sqrt(2pi) ← mandatory

    def psi_true(self, a: float) -> float:
        """
        Causal effect of dose a vs zero-dose baseline = m_true(a) - m_true(0).

        For Michaelis-Menten: m_true(0) = 0 exactly (mu(U, 0) = 0 for all U > 0),
        so psi_true(a) = m_true(a). This is the ABSOLUTE dose-response level,
        not a marginal effect dm/da.

        Conceptual note:
        - This is the CONTRAST against zero dose (not dm/da the slope).
        - For DGP1: psi_true(a) = alpha (slope, independent of a).
        - For DGP2: psi_true(a) = m_true(a) (varies with a — dose-response level).
        - DRDoseResponse estimates J(π_{a,h}) ≤ m_true(a) by Jensen (m concave).
          The Jensen gap is ≤ 0.02 with Silverman bandwidth at n=2000.

        Marginal effect (slope) if needed: finite difference of m_true.
            dm_da(a) ≈ (m_true(a + 1e-5) - m_true(a - 1e-5)) / (2e-5)
        """
        return self.m_true(float(a))   # m_true(0) = 0 exactly

    def h0(self, W: np.ndarray, A: np.ndarray,
           X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Regression oracle for the outcome bridge.

        Returns E[mu(U, A) | W, A] approximated by kernel ridge regression
        on a fresh sample of n_train=5000 observations (lazy-loaded on first call).

        ⚠ NOTE: This is the REGRESSION oracle (E[mu|W,A]), NOT the Fredholm bridge.
        For DGP1 (linear), h0_Fredholm = W + alpha*A (coefficient 1 on W),
        while E[Y|W,A] = snr_W*W + alpha*A (attenuated). DGP2 has the same issue.
        Consequence: T5b (Fredholm condition E[resid*Z]=0) is expected to fail.
        Deferred to Phase 8 (DRKernel) for the exact Fredholm bridge.
        """
        if self._h0_model is None:
            self._fit_h0_master()
        W_arr = np.asarray(W, dtype=float)
        A_arr = np.asarray(A, dtype=float)
        feats = np.column_stack([W_arr, A_arr])
        return self._h0_model.predict(feats)

    def q0(self, Z: np.ndarray, A: np.ndarray,
           X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Treatment bridge — same Gaussian form as CobbDouglasLinearDGP.q0.

        Structural analogy (sensitivity <-> gamma_K, sigma_A <-> sigma_K):
            D_Z = sigma_A^2 + sensitivity^2 * sigma_Z^2
            q0(Z, A) = sqrt(2pi * sigma_A^4 / D_Z)
                       * exp[(A - base_dose - sensitivity*Z)^2 / (2*D_Z)]

        Derivation: A | U ~ N(base_dose + sensitivity*U, sigma_A^2).
        Solve E[q0(Z,a) | U=u] = 1/f(a|U=u) in the Gaussian family.
        Convergence holds since D_Z > sensitivity^2 * sigma_Z^2. ✓
        """
        Z_arr = np.asarray(Z, dtype=float)
        A_arr = np.asarray(A, dtype=float)
        prefactor = np.sqrt(2.0 * np.pi * self.sigma_A ** 4 / self._D_Z)
        exponent = (A_arr - self.base_dose - self.sensitivity * Z_arr) ** 2 / (2.0 * self._D_Z)
        return prefactor * np.exp(exponent)

    # ── SNR properties ──────────────────────────────────────────────────────

    def snr_W(self) -> float:
        """Declared SNR of the outcome proxy W (calibrated for LogNormal U)."""
        return self._snr_W

    def snr_Z(self) -> float:
        """Declared SNR of the treatment proxy Z (calibrated for LogNormal U)."""
        return self._snr_Z

    # ── Private helper ──────────────────────────────────────────────────────

    def _fit_h0_master(self, n_train: int = 5_000, seed: int = 0) -> None:
        """
        Fit the regression oracle h0(W,A) ≈ E[mu(U,A) | W, A].

        Method: kernel ridge regression on mu(U,A) = Vmax*U*A/(Km*U+A) — the
        structural outcome (U is known in simulation). A StandardScaler normalises
        the (W, A) features before fitting.

        This is the 'master' method for the regression oracle. Lazy-loaded on the
        first call to h0(). Uses n_train=5000 by default (~1-2 seconds).
        """
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        sample = self.generate(n=n_train, seed=seed)
        mu_vals = ((self.Vmax * sample.U * sample.A)
                   / (self.Km * sample.U + sample.A))
        feats = np.column_stack([sample.W, sample.A])

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("kr", KernelRidge(kernel="rbf", alpha=1e-3, gamma=0.5)),
        ])
        pipe.fit(feats, mu_vals)
        self._h0_model = pipe
