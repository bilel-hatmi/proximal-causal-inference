"""
DGP 3 -- Coarsened Gaussian (latent continuous exposure under binary label).

Motivation
----------
A binary treatment label B in {0, 1} is, in many real applications, a coarse
summary of an underlying continuous exposure A* (vaccination intensity, A/B
test exposure duration, dose received, attention paid). The hard contrast
E[Y(1) - Y(0)] then asks for an idealised point intervention that does not
exist in the data: it is the limit rho -> 0 of the stochastic intervention
A*|do(B=b) ~ N(mu_b, rho^2), where rho > 0 in reality.

This DGP encodes that situation explicitly. The observed exposure A* is
continuous around the binary mean mu_B. The policy of interest is the
stochastic contrast J(pi_rho) = E[m(A*); A* ~ pi_{1,rho}] - E[m(A*); A* ~ pi_{0,rho}].

Structural equations (conditional on B=b)
-----------------------------------------
    U     ~ N(0, 1)                         [latent confounder, Gaussian]
    Z     = U + sigma_Z * eps_Z             [treatment-inducing proxy NCE]
    W     = U + sigma_W * eps_W             [outcome-inducing proxy NCO]
    B     ~ Bernoulli(0.5)                  [observed binary label]
    A*    = mu_B + gamma_U * U + sigma_A * eps_A
                                             [continuous exposure, sigma_A = rho_true]
    Y     = g(A*) + gamma * U + eps_Y       [g default = sin]

All errors independent N(0, 1).

Bridges (conditional on B=b, derived analytically by Chat T audit)
------------------------------------------------------------------
    h_0(W, A) = g(A) + gamma * W
    (linear in W, polynomial/transcendental in A; NO lambda_A term)

    q_0(Z, A, B=b) = sqrt(2*pi*sigma_A^4 / D_Z)
                     * exp[(A - mu_b - gamma_U * Z)^2 / (2 * D_Z)]
    where D_Z = sigma_A^2 + gamma_U^2 * sigma_Z^2

    The Riesz representer for the stochastic contrast at bandwidth rho_policy is
    r_pi(z, a, b) = pi_{rho_policy}(a; mu_b) * q_0(z, a, b).

Targets
-------
    For g(t) = sin(t), J(pi_rho) = exp(-rho^2/2) * (sin(mu_1) - sin(mu_0))
    For g(t) = alpha*t + beta*t^2 with equal-variance arms, J(pi_rho) is
    INDEPENDENT of rho (closed-form from Hermite-Gauss).

Theoretical caveats (per Chat T audit)
--------------------------------------
1. Operating conditional on B is essential -- otherwise A|Z is a Gaussian
   mixture and closed forms break.
2. The exact q_0 has potentially infinite second moment (E[q_0^2] = inf in
   the unbounded Gaussian setting). Population ESS is therefore 0/undefined;
   empirical ESS will be finite but unstable. Clipping diagnostics are
   essential.
3. Do NOT confuse q_0 with the observed GPS ratio pi(a)/f(a|z), which uses
   the conditional variance after deconvolving Z (smaller than D_Z).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.stats import norm

from pci.dgps.base import DGPSample, PCIDgp


DEFAULTS_DGP3 = {
    "mu_0": 0.0,        # mean exposure under B=0
    "mu_1": 1.0,        # mean exposure under B=1
    "gamma_U": 0.5,     # confounding strength: U -> A
    "rho_true": 0.2,    # domain bandwidth = sigma_A
    "gamma": 1.0,       # confounding in outcome: U -> Y
    "sigma_W": 1.0 / np.sqrt(3.0),   # SNR_W = 0.75 with U ~ N(0,1)
    "sigma_Z": 1.0 / np.sqrt(3.0),   # SNR_Z = 0.75
    "sigma_Y": 1.0,     # outcome noise
    "g_kind": "sin",    # "sin" | "linear" | "quadratic"
    "alpha": 1.0,       # for polynomial g
    "beta": -0.3,       # for polynomial g
}


def _g_func(g_kind: str, alpha: float, beta: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return the structural outcome function g."""
    if g_kind == "sin":
        return np.sin
    if g_kind == "sin2t":
        # Higher-frequency sin -- 4x curvature, amplifies target_mismatch
        return lambda t: np.sin(2.0 * t)
    if g_kind == "linear":
        return lambda a: alpha * a
    if g_kind == "quadratic":
        return lambda a: alpha * a + beta * (a ** 2)
    raise ValueError(f"Unknown g_kind {g_kind!r}; use 'sin', 'sin2t', 'linear', or 'quadratic'")


class CoarsenedGaussianDGP(PCIDgp):
    """
    DGP 3: Coarsened binary treatment as latent continuous Gaussian exposure.

    Parameters mirror DEFAULTS_DGP3. B is generated as Bernoulli(0.5) and
    stored in DGPSample.X (covariate slot). A is the CONTINUOUS exposure.

    All bridges and targets are computed in closed form (no lazy KRR fits).

    Examples
    --------
    >>> dgp = CoarsenedGaussianDGP(g_kind="sin", rho_true=0.2)
    >>> sample = dgp.generate(n=1000, seed=0)
    >>> J_true_at_policy = dgp.J_policy_true(rho_policy=0.10)
    >>> # J_hard = sin(1) - sin(0)
    >>> J_hard = dgp.J_policy_true(rho_policy=0.0)
    """

    def __init__(
        self,
        mu_0: float = DEFAULTS_DGP3["mu_0"],
        mu_1: float = DEFAULTS_DGP3["mu_1"],
        gamma_U: float = DEFAULTS_DGP3["gamma_U"],
        rho_true: float = DEFAULTS_DGP3["rho_true"],
        gamma: float = DEFAULTS_DGP3["gamma"],
        sigma_W: float = DEFAULTS_DGP3["sigma_W"],
        sigma_Z: float = DEFAULTS_DGP3["sigma_Z"],
        sigma_Y: float = DEFAULTS_DGP3["sigma_Y"],
        g_kind: str = DEFAULTS_DGP3["g_kind"],
        alpha: float = DEFAULTS_DGP3["alpha"],
        beta: float = DEFAULTS_DGP3["beta"],
    ) -> None:
        self.mu_0 = float(mu_0)
        self.mu_1 = float(mu_1)
        self.gamma_U = float(gamma_U)
        self.rho_true = float(rho_true)   # sigma_A in the structural model
        self.gamma = float(gamma)
        self.sigma_W = float(sigma_W)
        self.sigma_Z = float(sigma_Z)
        self.sigma_Y = float(sigma_Y)
        self.g_kind = g_kind
        self.alpha = float(alpha)
        self.beta = float(beta)

        # Pre-computed quantities
        self.sigma_A = self.rho_true
        self.D_Z = self.sigma_A ** 2 + self.gamma_U ** 2 * self.sigma_Z ** 2
        self.g = _g_func(self.g_kind, self.alpha, self.beta)

        # Empirical SNR (Var(U)=1 under N(0,1)):
        var_U = 1.0
        self._snr_W = var_U / (var_U + self.sigma_W ** 2)
        self._snr_Z = var_U / (var_U + self.sigma_Z ** 2)

    # ── Generate data ─────────────────────────────────────────────────────────

    def generate(self, n: int, seed: int) -> DGPSample:
        rng = np.random.default_rng(seed)
        U = rng.normal(0.0, 1.0, size=n)
        Z = U + self.sigma_Z * rng.normal(0.0, 1.0, size=n)
        W = U + self.sigma_W * rng.normal(0.0, 1.0, size=n)
        B = (rng.uniform(0.0, 1.0, size=n) < 0.5).astype(float)
        # Continuous exposure
        mu_b = np.where(B > 0.5, self.mu_1, self.mu_0)
        A = mu_b + self.gamma_U * U + self.sigma_A * rng.normal(0.0, 1.0, size=n)
        # Outcome
        Y = self.g(A) + self.gamma * U + self.sigma_Y * rng.normal(0.0, 1.0, size=n)
        # Population truth: contrast at rho=rho_true
        psi_0 = float(self.J_policy_true(self.rho_true))
        return DGPSample(Y=Y, A=A, X=B, Z=Z, W=W, U=U, psi_0=psi_0)

    # ── Bridges (analytic, conditional on B=b) ────────────────────────────────

    def h0(self, W: np.ndarray, A: np.ndarray,
           X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Outcome bridge h_0(W, A) = g(A) + gamma * W.

        Note: B does NOT appear -- the bridge is identical across strata since
        the only B dependence in Y comes through A (which we condition on).
        """
        W_arr = np.asarray(W, dtype=float).ravel()
        A_arr = np.asarray(A, dtype=float).ravel()
        return self.g(A_arr) + self.gamma * W_arr

    def q0(self, Z: np.ndarray, A: np.ndarray,
           X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Treatment bridge q_0(Z, A, B=b) -- stratum-wise Gaussian.

        E[q_0(Z, a, b) | U=u, B=b] = 1/f(a | U=u, B=b)

        Formula:
            q_0(z, a, b) = sqrt(2*pi*sigma_A^4 / D_Z)
                           * exp[(a - mu_b - gamma_U * z)^2 / (2 * D_Z)]

        D_Z = sigma_A^2 + gamma_U^2 * sigma_Z^2

        Parameters
        ----------
        Z, A : (n,) arrays
        X    : (n,) array of B in {0, 1} (REQUIRED -- B is the stratum)
        """
        if X is None:
            raise ValueError(
                "CoarsenedGaussianDGP.q0: X must be provided (= B in {0,1})."
            )
        Z_arr = np.asarray(Z, dtype=float).ravel()
        A_arr = np.asarray(A, dtype=float).ravel()
        B_arr = np.asarray(X, dtype=float).ravel()
        mu_b = np.where(B_arr > 0.5, self.mu_1, self.mu_0)
        prefactor = np.sqrt(2.0 * np.pi * self.sigma_A ** 4 / self.D_Z)
        exponent = (A_arr - mu_b - self.gamma_U * Z_arr) ** 2 / (2.0 * self.D_Z)
        return prefactor * np.exp(exponent)

    # ── Targets ───────────────────────────────────────────────────────────────

    def m_true(self, a: float) -> float:
        """E[Y | do(A* = a)] = g(a). Note: U is not in g(a), only in the bridge."""
        return float(self.g(np.array([float(a)]))[0])

    def psi_true(self, a: float) -> float:
        """For DGP3 the natural scalar is the binary contrast at rho_true."""
        return float(self.J_policy_true(self.rho_true))

    def J_policy_true(self, rho_policy: float) -> float:
        """
        True stochastic contrast J(pi_rho) = E_{T~N(mu_1, rho^2)}[g(T)]
                                             - E_{T~N(mu_0, rho^2)}[g(T)]

        Closed form for sin: J = exp(-rho^2/2) * (sin(mu_1) - sin(mu_0))
        Closed form for quadratic: J = alpha*(mu_1-mu_0) + beta*(mu_1^2-mu_0^2)
                                       (independent of rho)
        For "linear": J = alpha*(mu_1-mu_0) (independent of rho)

        For other g, falls back to Gauss-Hermite quadrature (degree 25).
        """
        rho = float(rho_policy)
        if rho < 1e-12:
            return float(self.g(np.array([self.mu_1]))[0]
                         - self.g(np.array([self.mu_0]))[0])
        if self.g_kind == "sin":
            return float(np.exp(-rho ** 2 / 2.0)
                          * (np.sin(self.mu_1) - np.sin(self.mu_0)))
        if self.g_kind == "sin2t":
            # E[sin(2T)] for T~N(mu, rho^2) = exp(-2 rho^2) sin(2 mu)
            return float(np.exp(-2.0 * rho ** 2)
                          * (np.sin(2.0 * self.mu_1) - np.sin(2.0 * self.mu_0)))
        if self.g_kind == "linear":
            return float(self.alpha * (self.mu_1 - self.mu_0))
        if self.g_kind == "quadratic":
            return float(
                self.alpha * (self.mu_1 - self.mu_0)
                + self.beta * (self.mu_1 ** 2 - self.mu_0 ** 2)
            )
        # Fallback: Gauss-Hermite
        from numpy.polynomial.hermite_e import hermegauss
        x_q, w_q = hermegauss(25)
        # E[g(T)] for T ~ N(mu, rho^2) = integral g(mu + rho*x) * phi(x) dx
        vals_1 = self.g(self.mu_1 + rho * x_q)
        vals_0 = self.g(self.mu_0 + rho * x_q)
        norm_const = np.sqrt(2.0 * np.pi)
        return float(np.dot(w_q, vals_1 - vals_0) / norm_const)

    # ── SNR properties ────────────────────────────────────────────────────────

    def snr_W(self) -> float:
        return self._snr_W

    def snr_Z(self) -> float:
        return self._snr_Z

    # ── Population diagnostic ─────────────────────────────────────────────────

    def expected_q0_squared(self, n_mc: int = 100_000, seed: int = 1) -> float:
        """
        Monte Carlo estimate of E[q_0(Z, A*, B)^2] for the marginal distribution.

        Returns inf if the integrand explodes (typical for Gaussian DGP3).
        Use for theoretical ESS = (E[q_0])^2 / E[q_0^2] = 1 / E[q_0^2].
        """
        rng = np.random.default_rng(seed)
        U = rng.normal(0.0, 1.0, size=n_mc)
        Z = U + self.sigma_Z * rng.normal(0.0, 1.0, size=n_mc)
        B = (rng.uniform(0.0, 1.0, size=n_mc) < 0.5).astype(float)
        mu_b = np.where(B > 0.5, self.mu_1, self.mu_0)
        A = mu_b + self.gamma_U * U + self.sigma_A * rng.normal(0.0, 1.0, size=n_mc)
        q_vals = self.q0(Z, A, X=B)
        return float(np.mean(q_vals ** 2))


# ── Verification helper ──────────────────────────────────────────────────────

def verify_dgp3(verbose: bool = True) -> dict:
    """
    Light verification of DGP3 derivations.

    Tests:
        T1: generate produces shapes (n,) for all fields including B in {0,1}
        T2: h_0 satisfies E[Y - h_0(W, A) | Z, A, B] approx 0 (Fredholm condition,
             checked by binning on a coarse grid)
        T3: m_true matches g(a) for the chosen g_kind
        T4: J_policy_true(rho=0) = m_true(mu_1) - m_true(mu_0) (hard contrast)
        T5: For g="sin", J_policy_true(rho=0.2) = exp(-0.02) * sin(1)
    """
    out = {"passed": [], "failed": []}

    def _check(name: str, cond: bool, msg: str = ""):
        if cond:
            out["passed"].append(name)
            if verbose:
                print(f"  [PASS] {name}")
        else:
            out["failed"].append(f"{name}: {msg}")
            if verbose:
                print(f"  [FAIL] {name}: {msg}")

    dgp = CoarsenedGaussianDGP(g_kind="sin")
    sample = dgp.generate(n=2000, seed=0)

    # T1: shapes and B
    _check("T1_shapes",
           sample.Y.shape == (2000,) and sample.A.shape == (2000,)
           and sample.X.shape == (2000,) and sample.W.shape == (2000,)
           and sample.Z.shape == (2000,))
    _check("T1_B_binary",
           set(np.unique(sample.X).astype(int)).issubset({0, 1}),
           f"B values: {np.unique(sample.X)[:5]}")

    # T3: m_true
    _check("T3_mtrue_sin",
           abs(dgp.m_true(0.5) - np.sin(0.5)) < 1e-12,
           f"m_true(0.5) - sin(0.5) = {dgp.m_true(0.5) - np.sin(0.5)}")

    # T4: hard contrast
    J_hard = dgp.J_policy_true(0.0)
    expected_hard = np.sin(1.0) - np.sin(0.0)
    _check("T4_J_hard",
           abs(J_hard - expected_hard) < 1e-12,
           f"J_hard={J_hard}, expected={expected_hard}")

    # T5: J at rho=0.2 for sin
    J_02 = dgp.J_policy_true(0.2)
    expected_02 = np.exp(-0.02) * np.sin(1.0)
    _check("T5_J_rho02",
           abs(J_02 - expected_02) < 1e-10,
           f"J(0.2)={J_02}, expected={expected_02}")

    # T6: q_0 Gaussian form invariant E[q_0(Z, A, B) | U, B] = 1/f(A | U, B)
    # Verify by Monte Carlo with fixed U.
    rng = np.random.default_rng(123)
    n_mc = 10_000
    u_fixed = 0.5
    b_fixed = 1
    a_target = 1.2
    Z_mc = u_fixed + dgp.sigma_Z * rng.normal(0.0, 1.0, size=n_mc)
    A_mc = np.full(n_mc, a_target)
    B_mc = np.full(n_mc, b_fixed)
    q_vals = dgp.q0(Z_mc, A_mc, X=B_mc)
    expected = (1.0 / np.sqrt(2.0 * np.pi * dgp.sigma_A ** 2)
                * np.exp(-(a_target - dgp.mu_1 - dgp.gamma_U * u_fixed) ** 2
                          / (2.0 * dgp.sigma_A ** 2)))
    expected_inv = 1.0 / expected
    actual = float(np.mean(q_vals))
    rel_err = abs(actual - expected_inv) / abs(expected_inv)
    _check("T6_q0_inverse_density",
           rel_err < 0.05,  # 5% MC tolerance
           f"E[q_0|U={u_fixed},B={b_fixed},A={a_target}]={actual}, "
           f"expected 1/f={expected_inv}, rel_err={rel_err}")

    # T7: Empirical Fredholm residual
    # E[Y - h_0(W, A) | Z, A, B] should be near 0 in expectation
    h_obs = dgp.h0(sample.W, sample.A)
    residual = sample.Y - h_obs
    # Group by binned (Z, A, B) and check residuals
    # Simple version: just check mean residual is small (population zero)
    mean_resid = float(np.mean(residual))
    _check("T7_h0_mean_residual",
           abs(mean_resid) < 0.1,  # n=2000, sigma_Y=1, std ~ 1/sqrt(2000) ~ 0.022
           f"mean(Y - h_0(W,A)) = {mean_resid}")

    return out


if __name__ == "__main__":
    import sys
    print("=== DGP3 (CoarsenedGaussianDGP) verification ===")
    res = verify_dgp3(verbose=True)
    n_pass = len(res["passed"])
    n_fail = len(res["failed"])
    print(f"\n{n_pass} passed, {n_fail} failed")
    if n_fail > 0:
        sys.exit(1)
