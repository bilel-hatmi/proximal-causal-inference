"""
BennettIndepBridgeH -- True Bennett moment-solve h-bridge with joint RFF features.

Method
------
Solves the conditional moment condition

    E[ g(Z, A) (Y - h(W, A)) ] = 0   for all g in the critic class C

with finite-feature ansatz:
    h(W, A) = phi_H(W, A)^T beta        phi_H : (W, A) -> R^{m_H}  (joint RFF)
    g(Z, A) = phi_C(Z, A)^T theta       phi_C : (Z, A) -> R^{m_C}  (joint RFF, critic)

Empirical moment matrices:
    A_h    = (1/n) Phi_C^T Phi_H        (m_C, m_H)  cross-moment
    d_h    = (1/n) Phi_C^T Y            (m_C,)
    S_C    = (1/n) Phi_C^T Phi_C + gamma_critic * I    (m_C, m_C, SPD)

Critic-stabilised moment solve (from the dual minimax problem with critic norm
penalised by S_C^{-1}):

    (A_h^T S_C^{-1} A_h + lambda_H * I_{m_H}) beta = A_h^T S_C^{-1} d_h

This is the finite-feature Bennett (2022) h-estimator. INDEPENDENT of KPV
two-stage structure: only one moment solve on joint features.

Distinction from RFFBridgeH
---------------------------
- RFFBridgeH replaces each RBF Gram (W, A, Z marginals) by RFF approximation
  but keeps the 2-stage Mastouri/KPV structure.
- BennettIndepBridgeH uses joint (W, A) features for h and joint (Z, A) features
  for the critic, solving ONE moment system. No "Stage 1 / Stage 2" split.

API contract
------------
fit(W, A, Z, Y) -> self
predict(W, A) -> (n_test,) array
plug_in_policy(W, a, h) -> (n_test,) array

plug_in_policy uses closed-form Gaussian-policy integration of joint cosine
features (analogous to bennett_riesz.RandomFourierPolicyIntegrator).

Diagnostics (post-fit)
----------------------
- bridge_residual = ||A_h beta - d_h||^2  (raw moment residual)
- weighted_residual = (A_h beta - d_h)^T S_C^{-1} (A_h beta - d_h)
                                         (Bennett-style weighted residual)
- coefficient_norm = ||beta||^2
- condition_number of (A_h^T S_C^{-1} A_h + lambda_H I)
"""
from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import scipy.linalg as sla

from pci.bridges.kpv import _median_bandwidth
from pci.bridges.bennett_riesz import RandomFourierPairFeatureMap


def _spd_solve(G: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cholesky-first solve with fallback. Same as kpv_bridge._spd_solve."""
    try:
        c, low = sla.cho_factor(G, lower=True, check_finite=False)
        return sla.cho_solve((c, low), b, check_finite=False)
    except (sla.LinAlgError, np.linalg.LinAlgError):
        try:
            return np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            warnings.warn(
                "BennettIndepBridgeH: ill-conditioned system, lstsq fallback.",
                RuntimeWarning,
            )
            sol, *_ = np.linalg.lstsq(G, b, rcond=1e-10)
            return sol


class BennettIndepBridgeH:
    """
    Bennett-style independent h-bridge using joint RFF features for h and critic.

    Parameters
    ----------
    m_h : int
        Number of RFF features for phi_H(W, A).
    m_c : int
        Number of RFF features for phi_C(Z, A).
    ell_h : float
        Bandwidth for phi_H. Default: median heuristic on concat([W, A]).
    ell_c : float
        Bandwidth for phi_C. Default: median heuristic on concat([Z, A]).
    lambda_h : float
        Tikhonov regularisation on beta. Default 1e-3.
    gamma_critic : float
        Diagonal regularisation on critic covariance S_C. Default 1e-3.
    seed : int
        Seed for RFF maps (offset for h vs c).
    """

    has_rkhs_plug_in: bool = True
    is_linear_in_A:   bool = False

    def __init__(
        self,
        m_h: int = 1000,
        m_c: int = 1000,
        ell_h: Optional[float] = None,
        ell_c: Optional[float] = None,
        lambda_h: float = 1e-3,
        gamma_critic: float = 1e-3,
        seed: int = 2026,
    ) -> None:
        self.m_h = int(m_h)
        self.m_c = int(m_c)
        self.ell_h = ell_h
        self.ell_c = ell_c
        self.lambda_h = float(lambda_h)
        self.gamma_critic = float(gamma_critic)
        self.seed = int(seed)

        # Trained state
        self._fitted: bool = False
        self._phi_H_map: Optional[RandomFourierPairFeatureMap] = None
        self._phi_C_map: Optional[RandomFourierPairFeatureMap] = None
        self._beta: Optional[np.ndarray] = None
        self._n_train: int = 0

        # Diagnostics
        self.bridge_residual_: Optional[float] = None
        self.weighted_residual_: Optional[float] = None
        self.coefficient_norm_: Optional[float] = None
        self.cond_G_: Optional[float] = None
        # Phase 16 spectral diagnostics (canonical names)
        self.kappa_h_: Optional[float] = None              # alias of cond_G_
        self.eff_rank_h_: Optional[float] = None           # tr(G^-1 (G - λI))
        self.residual_norm_h_: Optional[float] = None      # ||A_h beta - d_h|| / ||d_h||

    def fit(
        self,
        W: np.ndarray,
        A: np.ndarray,
        Z: np.ndarray,
        Y: np.ndarray,
    ) -> "BennettIndepBridgeH":
        W = np.asarray(W, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        Z = np.asarray(Z, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        n = len(Y)
        if not (len(W) == len(A) == len(Z) == n):
            raise ValueError("W, A, Z, Y must have same length.")

        # Resolve bandwidths
        if self.ell_h is None:
            ell_h = _median_bandwidth(np.concatenate([W, A]))
        else:
            ell_h = float(self.ell_h)
        if self.ell_c is None:
            ell_c = _median_bandwidth(np.concatenate([Z, A]))
        else:
            ell_c = float(self.ell_c)

        # Build RFF feature maps
        phi_H_map = RandomFourierPairFeatureMap(
            n_features=self.m_h, ell_scale=ell_h, seed=self.seed,
        ).fit(W, A)
        phi_C_map = RandomFourierPairFeatureMap(
            n_features=self.m_c, ell_scale=ell_c, seed=self.seed + 100003,
        ).fit(Z, A)

        # Feature matrices
        Phi_H = phi_H_map.transform(W, A)   # (n, m_H)
        Phi_C = phi_C_map.transform(Z, A)   # (n, m_C)

        # Empirical moment matrices
        A_h = Phi_C.T @ Phi_H / n            # (m_C, m_H)
        d_h = Phi_C.T @ Y / n                # (m_C,)
        S_C = Phi_C.T @ Phi_C / n + self.gamma_critic * np.eye(self.m_c)
        S_C = 0.5 * (S_C + S_C.T)

        # Solve S_C^{-1} A_h and S_C^{-1} d_h via Cholesky
        try:
            L_S = sla.cho_factor(S_C, lower=True, check_finite=False)
            cho_ok = True
        except (sla.LinAlgError, np.linalg.LinAlgError):
            cho_ok = False
        if cho_ok:
            A_tilde = sla.cho_solve(L_S, A_h, check_finite=False)   # (m_C, m_H)
            d_tilde = sla.cho_solve(L_S, d_h, check_finite=False)   # (m_C,)
        else:
            A_tilde = np.linalg.solve(S_C, A_h)
            d_tilde = np.linalg.solve(S_C, d_h)

        # Final SPD system: G = A_h^T S_C^{-1} A_h + lambda_h I
        G = A_h.T @ A_tilde + self.lambda_h * np.eye(self.m_h)
        G = 0.5 * (G + G.T)
        rhs = A_h.T @ d_tilde
        beta = _spd_solve(G, rhs)

        # Diagnostics
        residual_raw = A_h @ beta - d_h
        bridge_residual = float(np.dot(residual_raw, residual_raw))
        # Weighted residual (Bennett-style): r^T S_C^{-1} r
        if cho_ok:
            r_tilde = sla.cho_solve(L_S, residual_raw, check_finite=False)
        else:
            r_tilde = np.linalg.solve(S_C, residual_raw)
        weighted_residual = float(np.dot(residual_raw, r_tilde))
        coefficient_norm = float(np.dot(beta, beta))
        try:
            cond_G = float(np.linalg.cond(G))
        except np.linalg.LinAlgError:
            cond_G = float("inf")

        # Cache
        self._phi_H_map = phi_H_map
        self._phi_C_map = phi_C_map
        self._beta = beta
        self._n_train = n
        self.ell_h = ell_h
        self.ell_c = ell_c

        self.bridge_residual_ = bridge_residual
        self.weighted_residual_ = weighted_residual
        self.coefficient_norm_ = coefficient_norm
        self.cond_G_ = cond_G

        # ── Phase 16 canonical spectral diagnostics ──────────────────────────
        self.kappa_h_ = cond_G  # alias under canonical name

        # Effective rank: tr(G^-1 (G - λI)) where G - λI = A_h^T S_C^-1 A_h
        # This measures the "effective dimension" of the bridge.
        try:
            G_data = G - self.lambda_h * np.eye(self.m_h)
            G_inv_data = np.linalg.solve(G, G_data)
            self.eff_rank_h_ = float(np.trace(G_inv_data))
        except Exception:
            self.eff_rank_h_ = float("nan")

        # Normalised residual: ||A_h beta - d_h|| / ||d_h||
        try:
            d_norm = float(np.linalg.norm(d_h))
            if d_norm > 1e-15:
                self.residual_norm_h_ = float(np.linalg.norm(residual_raw)) / d_norm
            else:
                self.residual_norm_h_ = float("nan")
        except Exception:
            self.residual_norm_h_ = float("nan")

        self._fitted = True
        return self

    def predict(self, W: np.ndarray, A: np.ndarray) -> np.ndarray:
        """h_hat(W_i, A_i) for each test point."""
        if not self._fitted:
            raise RuntimeError("BennettIndepBridgeH.predict before fit().")
        W = np.asarray(W, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        if len(W) != len(A):
            raise ValueError("W, A must have same length.")
        Phi_H_test = self._phi_H_map.transform(W, A)   # (n_test, m_H)
        return Phi_H_test @ self._beta

    def plug_in_policy(
        self,
        W: np.ndarray,
        a: float,
        h: float,
    ) -> np.ndarray:
        """
        Compute integral h_hat(W_i, t) K_h(t-a) dt for each test point.

        Closed form via RFF features:
            psi_a[k] = E_{A ~ N(a, h^2)}[phi_H(W_i, A)[k]]
                    (depends on W_i due to joint feature; computed analytically)
            integral = psi_a(W_i, a, h) @ beta

        Using RandomFourierPolicyIntegrator-style closed form:
            E[cos(omega_W * W_scaled + omega_A * A_scaled + b_phase)]
              under A ~ N(a, h^2)
              = cos(omega_W * W_scaled + omega_A * a_scaled + b_phase)
                * exp(-omega_A^2 * (h/scale)^2 / 2)
        """
        if not self._fitted:
            raise RuntimeError("BennettIndepBridgeH.plug_in_policy before fit().")
        W = np.asarray(W, dtype=float).ravel()
        n = len(W)

        fm = self._phi_H_map
        mean_, scale_ = fm._mean, fm._scale
        Omega = fm._Omega   # (m_H, 2)
        b_off = fm._b       # (m_H,)
        m_H = fm.n_features_

        # Scale W and a_target via the pooled scaler
        W_scaled = (W - mean_) / scale_
        a_scaled = (float(a) - mean_) / scale_

        # Argument: omega_W * W_scaled[i] + omega_A * a_scaled + b
        pairs = np.column_stack([W_scaled, np.full(n, a_scaled)])     # (n, 2)
        arg = pairs @ Omega.T + b_off                                  # (n, m_H)
        cos_term = np.cos(arg)                                         # (n, m_H)

        # Dampening factor exp(-omega_A^2 * (h_bandwidth/scale)^2 / 2) per feature
        bw_scaled = float(h) / scale_
        damp = np.exp(-(Omega[:, 1] ** 2) * (bw_scaled ** 2) / 2.0)    # (m_H,)

        psi_test = (np.sqrt(2.0 / m_H) * cos_term) * damp              # (n, m_H)
        return psi_test @ self._beta
