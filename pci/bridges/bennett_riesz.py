"""
BennettPolicyRiesz — Finite-feature Riesz representer for policy value J(pi).

Phase 12A: direct estimation of r_pi(Z,A) — the Riesz representer of the functional
J(pi_{a,h}) — via a finite polynomial feature approximation to the moment condition:

    E[r_pi(Z,A) * g(W,A)] = E[T_pi g(W)]    for all test functions g.

This bypasses the q-bridge factorisation q(Z,A) * pi(A) used in DRKernel.

Phase 12B (2026-04-28): adds RFF (random Fourier feature) mode for richer feature
classes capable of approximating the Riesz representer for nonlinear DGPs.

Architecture
-----------
  gaussian_raw_moment(order, mu, sigma)  -> float    (analytic E[X^k] for X~N(mu,sigma^2))
  PolynomialPairFeatureMap               -> fits/transforms (x1,x2) to monomials
  PolicyPolynomialIntegrator             -> analytic E_{A~pi_a}[Psi(W,A)] per row
  RandomFourierPairFeatureMap            -> fits/transforms (x1,x2) to cos features
  RandomFourierPolicyIntegrator          -> closed-form E_{A~pi_a}[Psi(W,A)] per row
  BennettPolicyRiesz                     -> ridge solve for gamma: r_pi = Phi(Z,A) @ gamma
  StabilizedBennettPolicyRiesz           -> Phase 12B Stage B optional minimax-stable variant

KEY RULES
---------
* BennettPolicyRiesz.fit() NEVER takes Y as argument.
  Y is identified only in BennettFunctionalDR.estimate().
* No np.linalg.inv — Cholesky -> solve -> lstsq fallback chain.
* V_hat convention: O(1), NOT divided by n. SE = sqrt(V_hat / n).
* Uppercase variance keys: V_total, V_reg, V_corr, V_cov.

References
----------
Bennett (2022) "Inference on Functionals under First Order Exit Conditions",
Journal of Econometrics. Eq. (2.2) — Riesz representer definition.
docs/notes/bennett_implementation_mapping.md — notation cross-reference.
docs/notes/bennett_rff_integration_math.md — closed-form RFF policy integration.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import scipy.linalg as sla


# ── Module-level utilities ────────────────────────────────────────────────────

def gaussian_raw_moment(order: int, mu: float, sigma: float) -> float:
    """
    E[X^order] where X ~ N(mu, sigma^2). Analytic formula via Hermite polynomials.

    Valid for orders 0, 1, 2, 3, 4. Raises ValueError for order > 4.

    Parameters
    ----------
    order : int, 0 <= order <= 4
    mu    : float, mean of the normal distribution
    sigma : float, standard deviation (may be 0 for a degenerate distribution)

    Returns
    -------
    float
        The raw moment E[X^order].
    """
    if order == 0:
        return 1.0
    elif order == 1:
        return float(mu)
    elif order == 2:
        return float(mu) ** 2 + float(sigma) ** 2
    elif order == 3:
        m, s2 = float(mu), float(sigma) ** 2
        return m ** 3 + 3.0 * m * s2
    elif order == 4:
        m, s2 = float(mu), float(sigma) ** 2
        return m ** 4 + 6.0 * m ** 2 * s2 + 3.0 * s2 ** 2
    else:
        raise ValueError(
            f"gaussian_raw_moment: order must be in {{0,1,2,3,4}}, got {order}."
        )


def _spd_solve_bennett(G: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve G x = b where G is symmetric positive (semi-)definite.

    Tries Cholesky factorisation, falls back to solve(), then lstsq.
    Adds a small jitter (1e-8 * ||G||) if the first Cholesky attempt fails.
    NEVER calls np.linalg.inv.

    Parameters
    ----------
    G : (p, p) SPD matrix
    b : (p,) right-hand side

    Returns
    -------
    (p,) solution vector
    """
    try:
        c, low = sla.cho_factor(G, lower=True, check_finite=False)
        return sla.cho_solve((c, low), b, check_finite=False)
    except (sla.LinAlgError, np.linalg.LinAlgError):
        pass

    # Jitter retry
    jitter_scale = 1e-8 * float(np.linalg.norm(G))
    G_jitter = G + jitter_scale * np.eye(G.shape[0])
    try:
        c, low = sla.cho_factor(G_jitter, lower=True, check_finite=False)
        return sla.cho_solve((c, low), b, check_finite=False)
    except (sla.LinAlgError, np.linalg.LinAlgError):
        pass

    try:
        return np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        warnings.warn(
            "BennettPolicyRiesz: ill-conditioned Gram matrix, falling back to lstsq.",
            RuntimeWarning,
        )
        sol, *_ = np.linalg.lstsq(G, b, rcond=1e-10)
        return sol


# ── PolynomialPairFeatureMap ──────────────────────────────────────────────────

class PolynomialPairFeatureMap:
    """
    Maps pairs (x1, x2) to polynomial monomial features x1^a * x2^b with a+b <= degree.

    Enumeration (degree=2, 6 features):
        (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)

    Enumeration (degree=3, 10 features):
        adds (3,0), (2,1), (1,2), (0,3)

    Normalisation: a single StandardScaler is fitted on the concatenated data
    [x1.ravel(), x2.ravel()] to get a pooled mean and std. This prevents polynomial
    terms from blowing up numerically. The same scaler is applied in transform()
    before computing the monomials.

    Parameters
    ----------
    degree : int, 2 or 3
        Maximum total degree of monomials.

    API
    ---
    fit(x1, x2)        -> PolynomialPairFeatureMap  (fits scaler on train data)
    transform(x1, x2)  -> (n, p) float64 array
    n_features()       -> int   (= (degree+1)(degree+2)//2)
    """

    def __init__(self, degree: int = 2) -> None:
        if degree not in (2, 3):
            raise ValueError(f"PolynomialPairFeatureMap: degree must be 2 or 3, got {degree}.")
        self.degree = int(degree)
        # Monomial exponents: list of (a_exp, b_exp) with a_exp + b_exp <= degree
        self._exponents: List[tuple] = [
            (a_exp, d - a_exp)
            for d in range(degree + 1)
            for a_exp in range(d + 1)
        ]
        # Fitted scaler parameters
        self._mean: Optional[float] = None
        self._scale: Optional[float] = None
        self._fitted: bool = False

    def n_features(self) -> int:
        """Number of monomial features = (degree+1)(degree+2)//2."""
        return len(self._exponents)

    def fit(self, x1: np.ndarray, x2: np.ndarray) -> "PolynomialPairFeatureMap":
        """
        Fit the pooled StandardScaler on [x1, x2] concatenated.

        Parameters
        ----------
        x1, x2 : (n,) arrays (training data only — no leakage)

        Returns
        -------
        self
        """
        x1 = np.asarray(x1, dtype=float).ravel()
        x2 = np.asarray(x2, dtype=float).ravel()
        all_data = np.concatenate([x1, x2])
        self._mean = float(np.mean(all_data))
        self._scale = float(np.std(all_data))
        if self._scale < 1e-10:
            self._scale = 1.0  # degenerate: constant data
        self._fitted = True
        return self

    def transform(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Map (x1, x2) to polynomial features of shape (n, p).

        Applies scaler BEFORE computing monomials to prevent numerical blow-up.

        Parameters
        ----------
        x1, x2 : (n,) arrays (may be test data)

        Returns
        -------
        (n, p) float64 array
        """
        if not self._fitted:
            raise RuntimeError("PolynomialPairFeatureMap: call fit() before transform().")
        x1 = np.asarray(x1, dtype=float).ravel()
        x2 = np.asarray(x2, dtype=float).ravel()
        n = len(x1)
        if len(x2) != n:
            raise ValueError(
                f"PolynomialPairFeatureMap.transform: x1, x2 must have the same length, "
                f"got {len(x1)}, {len(x2)}."
            )
        # Normalise
        s1 = (x1 - self._mean) / self._scale   # scaled x1
        s2 = (x2 - self._mean) / self._scale   # scaled x2

        p = self.n_features()
        out = np.empty((n, p), dtype=float)
        for k, (a_exp, b_exp) in enumerate(self._exponents):
            col = np.ones(n, dtype=float)
            if a_exp > 0:
                col *= s1 ** a_exp
            if b_exp > 0:
                col *= s2 ** b_exp
            out[:, k] = col
        return out


# ── PolicyPolynomialIntegrator ────────────────────────────────────────────────

class PolicyPolynomialIntegrator:
    """
    Computes E_{A~pi_{a,h}}[Psi(W, A)] analytically for each observation.

    The Gaussian policy is pi_a(t) = N(a, h^2) = exp(-(t-a)^2/(2h^2)) / (h * sqrt(2pi)),
    which is the same convention as DRKernel._kernel().

    For the monomial feature (a_exp, b_exp):
        E[scaled_W^a_exp * scaled_A^b_exp] = scaled_W^a_exp * E[scaled_A^b_exp]
    where scaled_A ~ N((a - mean) / scale, (bandwidth / scale)^2)
    and the moments of scaled_A are computed via gaussian_raw_moment().

    Parameters
    ----------
    feature_map : PolynomialPairFeatureMap
        A FITTED feature map for Psi (fitted on (W_train, A_train)).
        Carries the scaler parameters needed for moment transformation.
    """

    def __init__(self, feature_map: PolynomialPairFeatureMap) -> None:
        if not feature_map._fitted:
            raise RuntimeError(
                "PolicyPolynomialIntegrator: feature_map must be fitted before use."
            )
        self.feature_map = feature_map

    def integrate(
        self,
        W: np.ndarray,
        a_target: float,
        bandwidth: float,
    ) -> np.ndarray:
        """
        Compute E_{A~pi_{a,h}}[Psi(W_i, A)] for each observation i.

        Returns
        -------
        (n, p) array where row i = E_{A~pi_{a,h}}[Psi(W_i, A)]

        Notes
        -----
        Under the Gaussian policy A ~ N(a_target, bandwidth^2), A is independent
        of W (it's drawn from the policy), so:
            E[scaled_W_i^a * scaled_A^b] = scaled_W_i^a * E[scaled_A^b]
        The scaler is from the feature_map fitted on training data.
        """
        fm = self.feature_map
        W = np.asarray(W, dtype=float).ravel()
        n = len(W)
        mean_ = fm._mean
        scale_ = fm._scale

        # Scale the W observations (deterministic)
        scaled_W = (W - mean_) / scale_

        # Scale the policy parameters so moments are in the normalised space
        a_scaled = (float(a_target) - mean_) / scale_
        bw_scaled = float(bandwidth) / scale_

        p = fm.n_features()
        out = np.empty((n, p), dtype=float)
        for k, (a_exp, b_exp) in enumerate(fm._exponents):
            # E[scaled_A^b_exp] under scaled_A ~ N(a_scaled, bw_scaled^2)
            E_A_b = gaussian_raw_moment(b_exp, a_scaled, bw_scaled)

            if a_exp == 0:
                # Feature = 1 * scaled_A^b_exp → constant across i (except b_exp>0)
                out[:, k] = E_A_b  # broadcasts to (n,)
            else:
                out[:, k] = (scaled_W ** a_exp) * E_A_b

        return out


# ── BennettPolicyRiesz ────────────────────────────────────────────────────────

class BennettPolicyRiesz:
    """
    Finite-feature Riesz representer for the policy value J(pi_{a,h}).

    Estimates r_pi_a(Z,A) satisfying the discretised moment condition:
        C @ gamma_a ≈ b_a
    where:
        C[j,k] = (1/n) sum_i Psi_j(W_i,A_i) * Phi_k(Z_i,A_i)  -- cross-moment matrix
        b_a[j] = (1/n) sum_i E_{A~pi_a}[Psi_j(W_i,A)]          -- integrated policy
        r_hat_a(Z,A) = Phi(Z,A) @ gamma_a

    Ridge regression (primal form, p_phi x p_phi system):
        (C.T @ C + lambda_r * I) @ gamma_a = C.T @ b_a

    KEY: Y is NEVER passed to fit(). Passing Y as keyword raises TypeError.

    Parameters
    ----------
    lambda_r : float
        Ridge regularisation for the Riesz equation. Default 1e-3.
    degree : int (2 or 3)
        Polynomial degree for both Phi (Z,A) and Psi (W,A) feature maps.
    feature_map_type : str
        Only "polynomial" is supported (extensible placeholder).

    API
    ---
    fit(W, A, Z, a_grid, bandwidth)  -> self
    predict(Z, A, a_target)          -> (n_test,) array r_hat
    riesz_residual(W, A, Z, a_target) -> float ||C gamma - b||^2

    References
    ----------
    Bennett (2022) Eq. (2.2).
    docs/notes/bennett_implementation_mapping.md.
    """

    def __init__(
        self,
        lambda_r: float = 1e-3,
        degree: int = 2,
        feature_map_type: str = "polynomial",
        # Phase 12B RFF parameters
        n_features: int = 250,
        ell_scale: float = 2.5,
        rff_seed: Optional[int] = None,
    ) -> None:
        self.lambda_r = float(lambda_r)
        self.degree = int(degree)
        self.feature_map_type = str(feature_map_type)
        if self.feature_map_type not in ("polynomial", "rff"):
            raise ValueError(
                f"BennettPolicyRiesz: feature_map_type must be 'polynomial' or 'rff', "
                f"got {feature_map_type!r}."
            )
        # RFF-specific params (ignored in polynomial mode)
        self.n_features = int(n_features)
        self.ell_scale = float(ell_scale)
        self.rff_seed = rff_seed  # may be None

        # Fitted state (type widened to support both feature map types)
        self._fitted: bool = False
        self.phi_map_ = None  # (Z, A) -> r_pi basis
        self.psi_map_ = None  # (W, A) -> test functions basis
        self.gamma_dict_: Dict[float, np.ndarray] = {}
        # Cache cross-moment matrix C and rhs b per dose for riesz_residual()
        self._C_train: Optional[np.ndarray] = None   # (p_psi, p_phi), shared across doses
        self._b_dict_: Dict[float, np.ndarray] = {}  # b_a per dose
        self.a_grid_: Optional[np.ndarray] = None
        self.bandwidth_: Optional[float] = None

        # Phase 16 spectral diagnostics for r-bridge (computed once at fit)
        self.kappa_r_: Optional[float] = None              # cond(G_r) where G_r = C^T C + λ_r I
        self.eff_rank_r_: Optional[float] = None           # tr(G_r^-1 (G_r - λ_r I))
        self.residual_norm_r_per_dose_: Optional[Dict[float, float]] = None  # ||G_r γ - C^T b_a|| / ||C^T b_a|| per dose

    def fit(
        self,
        W: np.ndarray,
        A: np.ndarray,
        Z: np.ndarray,
        a_grid,
        bandwidth: float,
        X: np.ndarray = None,
    ) -> "BennettPolicyRiesz":
        """
        Fit the Riesz representer on training data.

        Parameters
        ----------
        W         : (n,) outcome proxy (NCO)
        A         : (n,) observed treatment
        Z         : (n,) treatment proxy (NCE)
        a_grid    : sequence of dose targets (K values)
        bandwidth : policy bandwidth h (Gaussian N(a, h^2))
        X         : covariates — reserved for future use, currently ignored

        Returns
        -------
        self (for method chaining)

        Note
        ----
        Y must NOT be passed. Passing Y=... raises TypeError since Y is not in
        the function signature. This is by design: the Riesz equation is
        identified from (W, A, Z) alone.
        """
        W = np.asarray(W, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        Z = np.asarray(Z, dtype=float).ravel()
        n = len(W)
        if not (len(A) == len(Z) == n):
            raise ValueError(
                f"BennettPolicyRiesz.fit: W, A, Z must have the same length. "
                f"Got len(W)={len(W)}, len(A)={len(A)}, len(Z)={len(Z)}."
            )
        a_grid_arr = [float(a) for a in a_grid]
        bandwidth = float(bandwidth)
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be > 0, got {bandwidth}.")

        # ── Feature maps (no leakage: fitted on THIS training slice) ─────────
        if self.feature_map_type == "polynomial":
            phi_map = PolynomialPairFeatureMap(degree=self.degree).fit(Z, A)
            psi_map = PolynomialPairFeatureMap(degree=self.degree).fit(W, A)
            integrator = PolicyPolynomialIntegrator(psi_map)
        elif self.feature_map_type == "rff":
            # Two distinct seeds for phi and psi: rff_seed and rff_seed + 100003
            seed_phi = self.rff_seed
            seed_psi = (
                self.rff_seed + 100003 if self.rff_seed is not None else None
            )
            phi_map = RandomFourierPairFeatureMap(
                n_features=self.n_features,
                ell_scale=self.ell_scale,
                seed=seed_phi,
            ).fit(Z, A)
            psi_map = RandomFourierPairFeatureMap(
                n_features=self.n_features,
                ell_scale=self.ell_scale,
                seed=seed_psi,
            ).fit(W, A)
            integrator = RandomFourierPolicyIntegrator(psi_map)
        else:
            raise ValueError(f"unknown feature_map_type {self.feature_map_type!r}")

        # ── Feature matrices on training data ─────────────────────────────────
        phi_mat = phi_map.transform(Z, A)   # (n, p_phi)
        psi_mat = psi_map.transform(W, A)   # (n, p_psi)

        # ── Cross-moment matrix C = Psi.T @ Phi / n : (p_psi, p_phi) ─────────
        # Riesz equation:  C @ gamma ≈ b_a  (p_psi equations, p_phi unknowns)
        C = psi_mat.T @ phi_mat / n   # (p_psi, p_phi)

        p_phi = phi_mat.shape[1]

        # ── Solve per dose (subclass hook for stabilized variant) ────────────
        gamma_dict, b_dict = self._solve_per_dose(
            phi_mat=phi_mat,
            psi_mat=psi_mat,
            C=C,
            integrator=integrator,
            W=W,
            a_grid_arr=a_grid_arr,
            bandwidth=bandwidth,
            n=n,
            p_phi=p_phi,
        )

        # ── Cache fitted state ─────────────────────────────────────────────────
        self.phi_map_ = phi_map
        self.psi_map_ = psi_map
        self.gamma_dict_ = gamma_dict
        self._C_train = C.copy()
        self._b_dict_ = b_dict
        self.a_grid_ = np.array(a_grid_arr)
        self.bandwidth_ = bandwidth
        self._fitted = True
        return self

    # ── Hook for subclass override (StabilizedBennettPolicyRiesz) ─────────────
    def _solve_per_dose(
        self,
        phi_mat: np.ndarray,
        psi_mat: np.ndarray,
        C: np.ndarray,
        integrator,
        W: np.ndarray,
        a_grid_arr: list,
        bandwidth: float,
        n: int,
        p_phi: int,
    ):
        """
        Solve the Riesz equation for each dose. Default: primal ridge form.
            (C.T @ C + lambda_r * I) gamma = C.T @ b_a

        Subclasses (StabilizedBennettPolicyRiesz) override for minimax-stable variants.

        Returns
        -------
        gamma_dict : Dict[float, np.ndarray]   gamma per dose
        b_dict     : Dict[float, np.ndarray]   b_a per dose (cached for diagnostics)
        """
        # ── Normal matrix for ridge: G = C.T @ C + lambda_r * I ──────────────
        G = C.T @ C + self.lambda_r * np.eye(p_phi)
        G = 0.5 * (G + G.T)

        # Phase 16 spectral diagnostics for r-bridge (computed once on G)
        try:
            self.kappa_r_ = float(np.linalg.cond(G))
        except Exception:
            self.kappa_r_ = float("nan")
        try:
            G_data = G - self.lambda_r * np.eye(p_phi)
            G_inv_data = np.linalg.solve(G, G_data)
            self.eff_rank_r_ = float(np.trace(G_inv_data))
        except Exception:
            self.eff_rank_r_ = float("nan")

        gamma_dict: Dict[float, np.ndarray] = {}
        b_dict: Dict[float, np.ndarray] = {}
        residual_norm_per_dose: Dict[float, float] = {}

        for a in a_grid_arr:
            # b_a[j] = (1/n) sum_i E_{A~pi_a}[Psi_j(W_i, A)]
            BarPsi = integrator.integrate(W, a, bandwidth)
            b_a = BarPsi.mean(axis=0)
            rhs = C.T @ b_a
            gamma = _spd_solve_bennett(G, rhs)
            gamma_dict[a] = gamma
            b_dict[a] = b_a

            # Phase 16: per-dose residual_norm_r = ||G gamma - rhs|| / ||rhs||
            try:
                residual = G @ gamma - rhs
                rhs_norm = float(np.linalg.norm(rhs))
                if rhs_norm > 1e-15:
                    residual_norm_per_dose[a] = float(np.linalg.norm(residual)) / rhs_norm
                else:
                    residual_norm_per_dose[a] = float("nan")
            except Exception:
                residual_norm_per_dose[a] = float("nan")

        self.residual_norm_r_per_dose_ = residual_norm_per_dose
        return gamma_dict, b_dict

    def predict(
        self,
        Z: np.ndarray,
        A: np.ndarray,
        a_target: float,
    ) -> np.ndarray:
        """
        Evaluate r_hat(Z_i, A_i) for dose a_target. Shape (n_test,).

        r_hat = Phi(Z, A) @ gamma_{a_target}

        Parameters
        ----------
        Z, A     : (n_test,) arrays
        a_target : float, must be in the fitted a_grid

        Returns
        -------
        (n_test,) float64 array
        """
        if not self._fitted:
            raise RuntimeError("BennettPolicyRiesz.predict: call fit() first.")
        a_target = float(a_target)
        if a_target not in self.gamma_dict_:
            raise KeyError(
                f"BennettPolicyRiesz.predict: a_target={a_target} not in fitted "
                f"a_grid. Available: {sorted(self.gamma_dict_.keys())}."
            )
        Z = np.asarray(Z, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        if len(Z) != len(A):
            raise ValueError(
                f"BennettPolicyRiesz.predict: Z and A must have the same length, "
                f"got {len(Z)}, {len(A)}."
            )
        phi_test = self.phi_map_.transform(Z, A)   # (n_test, p_phi)
        gamma = self.gamma_dict_[a_target]
        return phi_test @ gamma                     # (n_test,)

    def riesz_residual(
        self,
        W: np.ndarray,
        A: np.ndarray,
        Z: np.ndarray,
        a_target: float,
    ) -> float:
        """
        Compute ||C_test @ gamma_{a_target} - b_test||^2 on the given data.

        This quantifies how well the Riesz moment condition is satisfied:
            C @ gamma ≈ b_a
        where C and b_a are computed from (W, A, Z) with the fitted scalers.

        A smaller value indicates better fit of the Riesz equation.
        Used in T_BEN6 to verify residual decreases as lambda_r decreases.

        Parameters
        ----------
        W, A, Z  : (n,) arrays (typically training data)
        a_target : float

        Returns
        -------
        float : ||C @ gamma - b||^2
        """
        if not self._fitted:
            raise RuntimeError("BennettPolicyRiesz.riesz_residual: call fit() first.")
        a_target = float(a_target)
        if a_target not in self.gamma_dict_:
            raise KeyError(
                f"BennettPolicyRiesz.riesz_residual: a_target={a_target} not in a_grid."
            )
        W = np.asarray(W, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        Z = np.asarray(Z, dtype=float).ravel()
        n = len(W)

        phi_test = self.phi_map_.transform(Z, A)   # (n, p_phi)
        psi_test = self.psi_map_.transform(W, A)   # (n, p_psi)
        C_test = psi_test.T @ phi_test / n         # (p_psi, p_phi)

        if self.feature_map_type == "polynomial":
            integrator = PolicyPolynomialIntegrator(self.psi_map_)
        elif self.feature_map_type == "rff":
            integrator = RandomFourierPolicyIntegrator(self.psi_map_)
        else:
            raise ValueError(f"unknown feature_map_type {self.feature_map_type!r}")
        BarPsi = integrator.integrate(W, a_target, self.bandwidth_)
        b_test = BarPsi.mean(axis=0)               # (p_psi,)

        gamma = self.gamma_dict_[a_target]
        residual = C_test @ gamma - b_test         # (p_psi,)
        return float(np.dot(residual, residual))


# ── RandomFourierPairFeatureMap (Phase 12B) ───────────────────────────────────

class RandomFourierPairFeatureMap:
    """
    Random Fourier feature map (RBF kernel approximation) on pairs (x1, x2).

    For pair (x1_i, x2_i), the feature vector is:
        psi(x1_i, x2_i) = sqrt(2/m) * cos(Omega @ [x1_scaled, x2_scaled].T + b)
    where:
        Omega ~ N(0, 1/ell_scale^2 * I_2)   shape (m, 2)
        b     ~ Uniform(0, 2*pi)             shape (m,)
        scaler is pooled StandardScaler on concat([x1, x2]) (no leakage)

    Approximates k(x, x') = exp(-||x - x'||^2 / (2 * ell_scale^2)) (RBF / Gaussian
    kernel) with m output features in expectation.

    Parameters
    ----------
    n_features : int
        Number of cosine features m. Default 250.
    ell_scale : float
        Bandwidth for the underlying RBF kernel. Default 2.5.
    seed : int or None
        Seed for the RNG drawing Omega and b. Different seeds for phi vs psi maps.

    API
    ---
    fit(x1, x2)        -> self  (fits scaler + draws Omega, b)
    transform(x1, x2)  -> (n, m) array
    n_features()       -> int
    """

    def __init__(
        self,
        n_features: int = 250,
        ell_scale: float = 2.5,
        seed: Optional[int] = None,
    ) -> None:
        if n_features <= 0:
            raise ValueError(
                f"RandomFourierPairFeatureMap: n_features must be > 0, got {n_features}."
            )
        if ell_scale <= 0:
            raise ValueError(
                f"RandomFourierPairFeatureMap: ell_scale must be > 0, got {ell_scale}."
            )
        self.n_features_ = int(n_features)
        self.ell_scale = float(ell_scale)
        self.seed = seed

        # Fitted state
        self._mean: Optional[float] = None
        self._scale: Optional[float] = None
        self._Omega: Optional[np.ndarray] = None  # (m, 2)
        self._b: Optional[np.ndarray] = None      # (m,)
        self._fitted: bool = False

    def n_features(self) -> int:
        return self.n_features_

    def fit(self, x1: np.ndarray, x2: np.ndarray) -> "RandomFourierPairFeatureMap":
        """
        Fit pooled scaler on concat([x1, x2]); draw Omega and b.
        """
        x1 = np.asarray(x1, dtype=float).ravel()
        x2 = np.asarray(x2, dtype=float).ravel()
        all_data = np.concatenate([x1, x2])
        self._mean = float(np.mean(all_data))
        self._scale = float(np.std(all_data))
        if self._scale < 1e-10:
            self._scale = 1.0

        # Draw Omega and b with the seed (reproducible per fold)
        rng = np.random.default_rng(self.seed)
        # Omega entries ~ N(0, 1/ell_scale^2)
        sigma_omega = 1.0 / self.ell_scale
        self._Omega = rng.normal(0.0, sigma_omega, size=(self.n_features_, 2))
        self._b = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features_)
        self._fitted = True
        return self

    def transform(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Apply pooled scaler then compute sqrt(2/m) * cos(pairs @ Omega.T + b).

        Returns
        -------
        (n, m) float64 array
        """
        if not self._fitted:
            raise RuntimeError("RandomFourierPairFeatureMap: call fit() before transform().")
        x1 = np.asarray(x1, dtype=float).ravel()
        x2 = np.asarray(x2, dtype=float).ravel()
        n = len(x1)
        if len(x2) != n:
            raise ValueError(
                f"RandomFourierPairFeatureMap.transform: length mismatch "
                f"len(x1)={len(x1)}, len(x2)={len(x2)}."
            )
        s1 = (x1 - self._mean) / self._scale
        s2 = (x2 - self._mean) / self._scale
        pairs = np.column_stack([s1, s2])           # (n, 2)
        arg = pairs @ self._Omega.T + self._b        # (n, m)
        return np.sqrt(2.0 / self.n_features_) * np.cos(arg)

    def get_omega(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() first")
        return self._Omega

    def get_b(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() first")
        return self._b

    def get_scaler_params(self) -> tuple:
        if not self._fitted:
            raise RuntimeError("call fit() first")
        return (self._mean, self._scale)


# ── RandomFourierPolicyIntegrator (Phase 12B) ─────────────────────────────────

class RandomFourierPolicyIntegrator:
    """
    Compute E_{A ~ pi_a}[psi(W, A)] in CLOSED FORM for RFF features.

    Derivation (see docs/notes/bennett_rff_integration_math.md):

        psi_j(W, A) = sqrt(2/m) * cos(omega_j_W * W_scaled + omega_j_A * A_scaled + b_j)

    Under A ~ N(a_target, bandwidth^2):
        E[psi_j(W_i, A)]
          = sqrt(2/m) * cos(omega_j_W * W_scaled_i
                            + omega_j_A * a_target_scaled + b_j)
            * exp(-omega_j_A^2 * bandwidth_scaled^2 / 2)

    where bandwidth_scaled = bandwidth / scale.

    Parameters
    ----------
    feature_map : RandomFourierPairFeatureMap
        A FITTED RFF map (carries Omega, b, mean, scale).

    API
    ---
    integrate(W, a_target, bandwidth) -> (n, m) array
    """

    def __init__(self, feature_map: "RandomFourierPairFeatureMap") -> None:
        if not feature_map._fitted:
            raise RuntimeError(
                "RandomFourierPolicyIntegrator: feature_map must be fitted before use."
            )
        self.feature_map = feature_map

    def integrate(
        self,
        W: np.ndarray,
        a_target: float,
        bandwidth: float,
    ) -> np.ndarray:
        fm = self.feature_map
        W = np.asarray(W, dtype=float).ravel()
        n = len(W)
        mean_ = fm._mean
        scale_ = fm._scale
        m = fm.n_features_
        Omega = fm._Omega   # (m, 2)
        b_off = fm._b       # (m,)

        # Scale W (deterministic) and a_target (centre of policy)
        W_scaled = (W - mean_) / scale_                    # (n,)
        a_scaled = (float(a_target) - mean_) / scale_      # scalar

        # Pairs with A replaced by a_scaled (scalar broadcast)
        pairs = np.column_stack([W_scaled, np.full(n, a_scaled)])   # (n, 2)
        arg = pairs @ Omega.T + b_off                                # (n, m)
        cos_term = np.cos(arg)                                       # (n, m)

        # Dampening factor exp(-omega_A^2 * bandwidth_scaled^2 / 2) per feature
        bw_scaled = float(bandwidth) / scale_
        damp = np.exp(-(Omega[:, 1] ** 2) * (bw_scaled ** 2) / 2.0)  # (m,)

        return (np.sqrt(2.0 / m) * cos_term) * damp                  # (n, m)


# ── StabilizedBennettPolicyRiesz (Phase 12B Stage B optional) ─────────────────

class StabilizedBennettPolicyRiesz(BennettPolicyRiesz):
    """
    Minimax-stable variant of the Riesz solver. Same feature pipeline as
    BennettPolicyRiesz, but the Riesz equation is solved through a Tikhonov-
    regularised inverse covariance on the test (psi) side:

        S = lambda_stab * (psi_mat.T @ psi_mat) / n + gamma_critic * I_p_psi
        gamma = solve(M.T @ S^{-1} @ M + lambda_r * I_p_phi, M.T @ S^{-1} @ b_a)

    where M = psi_mat.T @ phi_mat / n = C is the cross-moment matrix.

    Intended for Phase 12B Stage B if direct RFF shows partial signal but variance
    or SER is uncalibrated.

    Parameters
    ----------
    lambda_stab : float
        Weight on the empirical test covariance term. Default 1.0.
    gamma_critic : float
        Diagonal regularisation on the critic's covariance. Default 1e-3.

    All other kwargs forwarded to BennettPolicyRiesz.
    """

    def __init__(
        self,
        lambda_stab: float = 1.0,
        gamma_critic: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.lambda_stab = float(lambda_stab)
        self.gamma_critic = float(gamma_critic)

    def _solve_per_dose(
        self,
        phi_mat: np.ndarray,
        psi_mat: np.ndarray,
        C: np.ndarray,
        integrator,
        W: np.ndarray,
        a_grid_arr: list,
        bandwidth: float,
        n: int,
        p_phi: int,
    ):
        """
        Stabilized solve:
            S = lambda_stab * Psi.T @ Psi / n + gamma_critic * I_p_psi
            For each a:
                M_tilde = S^{-1} @ M_hat   where M_hat = C
                b_tilde = S^{-1} @ b_a
                G = M.T @ M_tilde + lambda_r * I
                gamma = solve(G, M.T @ b_tilde)
        """
        p_psi = psi_mat.shape[1]
        # S is SPD by construction
        S = self.lambda_stab * (psi_mat.T @ psi_mat) / n + \
            self.gamma_critic * np.eye(p_psi)
        S = 0.5 * (S + S.T)

        # Cholesky factor S once; re-use across doses
        try:
            L_S = sla.cho_factor(S, lower=True, check_finite=False)
            cho_ok = True
        except (sla.LinAlgError, np.linalg.LinAlgError):
            cho_ok = False

        if cho_ok:
            # M_tilde = S^{-1} @ M_hat (= C)  —> shape (p_psi, p_phi)
            M_tilde = sla.cho_solve(L_S, C, check_finite=False)
        else:
            # Fallback: solve directly per column
            M_tilde = np.linalg.solve(S, C)

        # G_phi = C.T @ M_tilde + lambda_r * I_p_phi  -> (p_phi, p_phi)
        G_phi = C.T @ M_tilde + self.lambda_r * np.eye(p_phi)
        G_phi = 0.5 * (G_phi + G_phi.T)

        gamma_dict: Dict[float, np.ndarray] = {}
        b_dict: Dict[float, np.ndarray] = {}
        for a in a_grid_arr:
            BarPsi = integrator.integrate(W, a, bandwidth)
            b_a = BarPsi.mean(axis=0)
            if cho_ok:
                b_tilde = sla.cho_solve(L_S, b_a, check_finite=False)
            else:
                b_tilde = np.linalg.solve(S, b_a)
            rhs = C.T @ b_tilde
            gamma = _spd_solve_bennett(G_phi, rhs)
            gamma_dict[a] = gamma
            b_dict[a] = b_a
        return gamma_dict, b_dict
