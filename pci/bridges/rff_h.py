"""
pci.bridges.rff_h
==================

RFFBridgeH -- Drop-in replacement for KPVBridgeH with Random Fourier Feature
approximation of the RBF Grams.

Extracted from ``pci/estimators/rff_bridge_h.py``. The
original module continues to re-export ``RFFBridgeH``, ``RFFFeatureMap1D``,
and ``_rff_gram`` as a backward-compatibility shim with class identity
preserved.


Same 2-stage Mastouri (2021) bridge math as KPVBridgeH, but each RBF Gram
K_X(X1, X2) is replaced by Phi_X(X1) Phi_X(X2)^T where Phi_X are RFF features
approximating the Gaussian RBF kernel:

    k_X(x, y) = exp(-(x-y)^2 / (2 ell_X^2))   (RBF)
    Phi_X(x) = sqrt(2/m) * cos(omega^T x + b)
    omega ~ N(0, 1/ell_X^2 I),  b ~ Uniform(0, 2*pi)
    => E[Phi_X(x)^T Phi_X(y)] = k_X(x, y)

The 2-stage Mastouri/KPV solver (Stage 1: Gamma, Stage 2: alpha) is unchanged.
For plug_in_policy, the Gauss-Gauss convolution is replaced by the closed-form
expectation of cosine features under a Gaussian policy:

    E_{A ~ N(a, h^2)}[cos(omega A + b)] = cos(omega a + b) * exp(-omega^2 h^2 / 2)

so the integral of h_hat against the policy kernel reduces to a feature-space
inner product against an analytically-computed expected feature vector.

Purpose
-------
Phase 14A (RFF-fairness gating) needs a faithful RFF substitute for KPVBridgeH
to test whether RFF approximation of the RBF kernel is methodologically fair
in the Bennett vs KPV comparison. Same API, same fold structure, only the
Gram evaluation differs.

API contract
------------
RFFBridgeH(lambda_1, lambda_2, ell_W, ell_A, ell_Z,
           n_features_W, n_features_A, n_features_Z, seed)
.fit(W, A, Z, Y) -> self
.predict(W, A) -> (n_test,) array
.plug_in_policy(W, a, h) -> (n_test,) array (closed form)
"""
from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import scipy.linalg as sla

from pci.bridges.kpv import _median_bandwidth, _spd_solve


class RFFFeatureMap1D:
    """
    1D Random Fourier Feature map for the Gaussian RBF kernel.

    For ell > 0 and m features:
        Phi(x)[k] = sqrt(2/m) * cos(omega_k * x + b_k)
        omega_k ~ N(0, 1/ell^2),  b_k ~ Uniform(0, 2*pi)

    Then E[Phi(x)^T Phi(y)] = exp(-(x-y)^2 / (2 ell^2)) (RBF kernel).
    """

    def __init__(self, n_features: int, ell: float, seed: Optional[int] = None) -> None:
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}.")
        if ell <= 0:
            raise ValueError(f"ell must be > 0, got {ell}.")
        self.m = int(n_features)
        self.ell = float(ell)
        self.seed = seed
        rng = np.random.default_rng(seed)
        # omega ~ N(0, 1/ell^2)  -> sigma = 1/ell
        self.omega = rng.normal(0.0, 1.0 / self.ell, size=self.m)   # (m,)
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=self.m)         # (m,)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Map (n,) array to (n, m) RFF feature matrix."""
        x = np.asarray(x, dtype=float).ravel()
        # arg[i, k] = omega[k] * x[i] + b[k]
        arg = np.outer(x, self.omega) + self.b   # (n, m)
        return np.sqrt(2.0 / self.m) * np.cos(arg)

    def expected_feature_under_gaussian(
        self, mu: float, sigma: float
    ) -> np.ndarray:
        """
        Closed-form E_{X ~ N(mu, sigma^2)}[Phi(X)] for cosine features.

            E[cos(omega*X + b)] = cos(omega*mu + b) * exp(-omega^2 * sigma^2 / 2)

        Returns (m,) array.
        """
        return np.sqrt(2.0 / self.m) * np.cos(self.omega * mu + self.b) * \
               np.exp(-(self.omega ** 2) * (sigma ** 2) / 2.0)


def _rff_gram(phi_X: np.ndarray, phi_Y: np.ndarray) -> np.ndarray:
    """
    RFF-approximated RBF Gram K[i, j] = Phi(X_i)^T Phi(Y_j).

    phi_X : (n_X, m), phi_Y : (n_Y, m)  =>  K : (n_X, n_Y)
    """
    return phi_X @ phi_Y.T


class RFFBridgeH:
    """
    Drop-in replacement for KPVBridgeH using RFF Gram approximation.

    Same 2-stage Mastouri (2021) ridge bridge:
      Stage 1: Gamma = (K_ZA + n*lambda_1*I)^{-1} K_ZA
      Stage 2: alpha = (G + n*lambda_2*I)^{-1} Y, where G = (Gamma^T K_W Gamma) (.) K_A

    K_W, K_A, K_Z are RFF approximations of the RBF Grams.

    Parameters
    ----------
    lambda_1, lambda_2 : float
        Tikhonov reg for Stage 1 / Stage 2. Default 1e-2 * n^(-0.7).
    ell_W, ell_A, ell_Z : float, optional
        RBF length-scales. Default median heuristic on training slice.
    n_features_W, n_features_A, n_features_Z : int
        RFF feature counts per variable. Default 500.
    seed : int
        Seed for RFF random draw (omega, b). Different seeds for W/A/Z to
        avoid spurious correlations.
    """

    has_rkhs_plug_in: bool = True
    is_linear_in_A:   bool = False

    def __init__(
        self,
        lambda_1: Optional[float] = None,
        lambda_2: Optional[float] = None,
        ell_W: Optional[float] = None,
        ell_A: Optional[float] = None,
        ell_Z: Optional[float] = None,
        n_features_W: int = 500,
        n_features_A: int = 500,
        n_features_Z: int = 500,
        seed: int = 2026,
    ) -> None:
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.ell_W = ell_W
        self.ell_A = ell_A
        self.ell_Z = ell_Z
        self.n_features_W = int(n_features_W)
        self.n_features_A = int(n_features_A)
        self.n_features_Z = int(n_features_Z)
        self.seed = int(seed)

        # Trained state
        self._fitted: bool = False
        self._W_train: Optional[np.ndarray] = None
        self._A_train: Optional[np.ndarray] = None
        # Cached RFF maps (need them at predict time)
        self._phi_W_map: Optional[RFFFeatureMap1D] = None
        self._phi_A_map: Optional[RFFFeatureMap1D] = None
        self._phi_Z_map: Optional[RFFFeatureMap1D] = None
        # Cached training feature matrices
        self._Phi_W_train: Optional[np.ndarray] = None
        self._Phi_A_train: Optional[np.ndarray] = None
        # Stage 1 / Stage 2 caches
        self._Gamma: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._n_train: int = 0

    def fit(
        self,
        W: np.ndarray,
        A: np.ndarray,
        Z: np.ndarray,
        Y: np.ndarray,
    ) -> "RFFBridgeH":
        W = np.asarray(W, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        Z = np.asarray(Z, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        n = len(Y)
        if not (len(W) == len(A) == len(Z) == n):
            raise ValueError("W, A, Z, Y must all have the same length.")

        # Resolve bandwidths
        ell_W = self.ell_W if self.ell_W is not None else _median_bandwidth(W)
        ell_A = self.ell_A if self.ell_A is not None else _median_bandwidth(A)
        ell_Z = self.ell_Z if self.ell_Z is not None else _median_bandwidth(Z)

        # Resolve regularisation (Ying scaling, same as KPVBridgeH)
        lam1 = self.lambda_1 if self.lambda_1 is not None else 1e-2 * n ** (-0.7)
        lam2 = self.lambda_2 if self.lambda_2 is not None else 1e-2 * n ** (-0.7)

        # ── RFF feature maps (offset seeds for W, A, Z) ──────────────────────
        phi_W_map = RFFFeatureMap1D(self.n_features_W, ell_W, seed=self.seed)
        phi_A_map = RFFFeatureMap1D(self.n_features_A, ell_A, seed=self.seed + 100003)
        phi_Z_map = RFFFeatureMap1D(self.n_features_Z, ell_Z, seed=self.seed + 200017)

        # ── Build training feature matrices ──────────────────────────────────
        Phi_W = phi_W_map.transform(W)   # (n, m_W)
        Phi_A = phi_A_map.transform(A)   # (n, m_A)
        Phi_Z = phi_Z_map.transform(Z)   # (n, m_Z)

        # ── Build RFF-approximated Grams ─────────────────────────────────────
        K_W = Phi_W @ Phi_W.T    # (n, n) approx of exp(-(W_i-W_j)^2/(2 ell_W^2))
        K_A = Phi_A @ Phi_A.T
        K_Z = Phi_Z @ Phi_Z.T
        K_ZA = K_Z * K_A         # tensor product RKHS Gram

        # ── Stage 1: Gamma = (K_ZA + n*lam1*I)^{-1} K_ZA ─────────────────────
        Gamma = _spd_solve(K_ZA + n * lam1 * np.eye(n), K_ZA)

        # ── Stage 2: G = (Gamma^T K_W Gamma) ⊙ K_A,  alpha = (G+n*lam2*I)^-1 Y ─
        G = (Gamma.T @ K_W @ Gamma) * K_A
        G = 0.5 * (G + G.T)
        alpha = _spd_solve(G + n * lam2 * np.eye(n), Y)

        # ── Cache ────────────────────────────────────────────────────────────
        self._W_train = W
        self._A_train = A
        self._phi_W_map = phi_W_map
        self._phi_A_map = phi_A_map
        self._phi_Z_map = phi_Z_map
        self._Phi_W_train = Phi_W
        self._Phi_A_train = Phi_A
        self._Gamma = Gamma
        self._alpha = alpha
        self._n_train = n
        self.ell_W = ell_W
        self.ell_A = ell_A
        self.ell_Z = ell_Z
        self.lambda_1 = lam1
        self.lambda_2 = lam2
        self._fitted = True
        return self

    def predict(self, W: np.ndarray, A: np.ndarray) -> np.ndarray:
        """h_hat(W_i, A_i) for each test point."""
        if not self._fitted:
            raise RuntimeError("RFFBridgeH.predict called before fit().")
        W = np.asarray(W, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        if len(W) != len(A):
            raise ValueError("W and A must have the same length.")
        Phi_W_test = self._phi_W_map.transform(W)               # (n_test, m_W)
        Phi_A_test = self._phi_A_map.transform(A)               # (n_test, m_A)
        K_W_test = Phi_W_test @ self._Phi_W_train.T             # (n_test, n)
        K_A_test = Phi_A_test @ self._Phi_A_train.T
        Ew_test = K_W_test @ self._Gamma                         # (n_test, n)
        return np.sum(Ew_test * K_A_test * self._alpha[None, :], axis=1)

    def plug_in_policy(
        self,
        W: np.ndarray,
        a: float,
        h: float,
    ) -> np.ndarray:
        """
        Compute integral h_hat(W_i, t) K_h(t-a) dt for each test point.

        Closed form via RFF:
            E_{A ~ N(a, h^2)}[Phi_A(A)] = phi_a (analytic, m_A vector)
            k_tilde(A_train_j, a, h) = Phi_A_train[j] @ phi_a (n,)
        Then:
            integral = (K_W_test @ Gamma) @ (alpha (.) k_tilde)
        """
        if not self._fitted:
            raise RuntimeError("RFFBridgeH.plug_in_policy called before fit().")
        W = np.asarray(W, dtype=float).ravel()

        # Expected RFF feature for A under N(a, h^2)
        phi_a = self._phi_A_map.expected_feature_under_gaussian(float(a), float(h))   # (m_A,)
        # k_tilde[j] = Phi_A_train[j] @ phi_a  (closed-form RBF Gauss convolution under RFF)
        k_tilde = self._Phi_A_train @ phi_a                          # (n,)

        Phi_W_test = self._phi_W_map.transform(W)                    # (n_test, m_W)
        K_W_test = Phi_W_test @ self._Phi_W_train.T                  # (n_test, n)
        Ew_test = K_W_test @ self._Gamma                              # (n_test, n)
        return Ew_test @ (self._alpha * k_tilde)
