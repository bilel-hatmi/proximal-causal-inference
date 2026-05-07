"""
KallusStabilizedPolicyQ -- Phase 11 finite-feature q-bridge solver.

Drop-in replacement for KPVPolicyBridgeQ inside DRKernel. Implements two modes
that share the same Nystroem features (so any difference between them comes
from the solve, not from the function class):

    mode = "ridge_gmm"
        (M^T M + gamma_Q I) alpha = M^T b

    mode = "kallus_stabilized"
        (M^T S^{-1} M + gamma_Q I) alpha = M^T S^{-1} b
        where  S = lambda_stab * Phi_H^T Phi_H / n + gamma_critic * I
        is the empirical critic-side stabilisation (Kallus 2021, Eq. 13).

Notation
--------
- Phi_Q : finite-feature map for q_a(Z, A)         shape (n, m_Q)
- Phi_H : finite-feature map for the critic g(W, A) shape (n, m_H)
  (PairFeatureMap on (W, A) optionally augmented with [1, W, A, W*A])
- For dose a_k:
    Lambda_k[i] = pi_{a_k}(A_i) = K_h(A_i - a_k) (normalised Gaussian density)
    M_k = (Phi_H * Lambda_k[:,None]).T @ Phi_Q / n        shape (m_H, m_Q)
    b_k = mean_i (T_pi Phi_H)(W_i, a_k)                   shape (m_H,)
- The estimated q is then
    q_a(Z, A) = Phi_Q(Z, A) @ alpha_k                     shape (n_test,)

Numerical discipline
--------------------
- Never form inv(.). All linear systems use scipy.linalg.cho_factor + cho_solve
  with a fallback to scipy.linalg.solve(assume_a="sym") and finally
  np.linalg.lstsq.
- All SPD matrices get a small jitter * I before factorisation.
- Diagnostics expensive in n^3 (cond_S, cond_lhs, effective_rank) only when
  compute_diagnostics="full". The default "light" path is safe in MC loops.

Cross-fitting
-------------
fit(W, A, Z) is called per fold by DRKernel. Each call rebuilds its own
landmarks, scalers, integrator grid, and per-dose alphas -- nothing leaks
across folds.

References
----------
- Kallus (2021), "Optimal Estimation of Generalised Average Treatment Effects
  using Kernel Optimal Matching", JASA 116(534).
- Mastouri et al. (2021), "Proximal Causal Learning with Kernels", ICML 2021.
"""
from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import scipy.linalg as sla

from pci.estimators.features import (
    NystromFeatureMap,
    PairFeatureMap,
    PolicyFeatureIntegrator,
)


# ── Module-level utility ─────────────────────────────────────────────────────

def _median_bandwidth_safe(x: np.ndarray) -> float:
    from pci.estimators.kpv_bridge import _median_bandwidth
    return _median_bandwidth(x)


def _spd_solve(A: np.ndarray, B: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """
    Solve A X = B with A SPD (or near-SPD). Returns X.

    Tries Cholesky first; falls back to symmetric solve, then to lstsq.
    Always adds `jitter * I` before factorisation.
    """
    n = A.shape[0]
    A_jit = A + jitter * np.eye(n)
    try:
        c, low = sla.cho_factor(A_jit, lower=True, check_finite=False)
        return sla.cho_solve((c, low), B, check_finite=False)
    except (sla.LinAlgError, np.linalg.LinAlgError):
        try:
            return sla.solve(A_jit, B, assume_a="sym", check_finite=False)
        except (sla.LinAlgError, np.linalg.LinAlgError):
            warnings.warn(
                "KallusStabilizedPolicyQ: SPD solve failed, falling back to lstsq.",
                RuntimeWarning,
            )
            X, *_ = np.linalg.lstsq(A_jit, B, rcond=1e-10)
            return X


# ══════════════════════════════════════════════════════════════════════════════
#  KallusStabilizedPolicyQ
# ══════════════════════════════════════════════════════════════════════════════

class KallusStabilizedPolicyQ:
    """
    Finite-feature q-bridge solver. See module docstring for math.

    Drop-in for DRKernel: implements fit(W, A, Z), predict(Z, A, dose_index),
    predict_all(Z, A), predict_raw_all(Z, A), and exposes the diagnostic lists
    that DRKernel harvests at dr_kernel.py:278-293.

    Parameters
    ----------
    a_grid       : 1D array of dose values, one alpha per dose.
    h_KDE        : Gaussian bandwidth for the policy K_h.
    mode         : "ridge_gmm" or "kallus_stabilized".
    gamma_Q      : ridge on alpha (regularises q in the Phi_Q feature space).
    lambda_stab  : weight of empirical Phi_H^T Phi_H / n inside S
                   (only used when mode="kallus_stabilized").
    gamma_critic : ridge on S (only used when mode="kallus_stabilized").
    jitter       : added to all SPD matrices before Cholesky.
    n_features_q,
    n_features_h : Nystroem dimensions for Phi_Q and Phi_H.
    feature_seed : reproducibility for landmark sub-sampling.
    ell_W, ell_A, ell_Z :
                   RBF length-scales (median heuristic if None).
    augment_critic_with_simple :
                   if True, append [1, W, A, W*A] to Phi_H.
    integration_grid_size :
                   number of trapezoidal nodes for T_pi.
    integration_range :
                   (a_min, a_max) or "auto" (per coordinate).
    clip         : upper bound for clipping q in predict()/predict_all().
                   Default: 5 * n^(1/4).
    compute_diagnostics : "light" or "full".
    """

    def __init__(
        self,
        a_grid: np.ndarray,
        h_KDE: float,
        # Mode
        mode: str = "kallus_stabilized",
        # Regularisations
        gamma_Q: float = 1e-3,
        lambda_stab: float = 1.0,
        gamma_critic: float = 1e-3,
        jitter: float = 1e-8,
        # Features
        n_features_q: int = 150,
        n_features_h: int = 150,
        feature_seed: int = 20260427,
        ell_W: Optional[float] = None,
        ell_A: Optional[float] = None,
        ell_Z: Optional[float] = None,
        augment_critic_with_simple: bool = True,
        # Policy integration
        integration_grid_size: int = 400,
        integration_range=("auto", "auto"),
        # Stability
        clip: Optional[float] = None,
        # Diagnostics
        compute_diagnostics: str = "light",
    ) -> None:
        if mode not in ("ridge_gmm", "kallus_stabilized"):
            raise ValueError(
                f"mode must be 'ridge_gmm' or 'kallus_stabilized', got {mode!r}."
            )
        if compute_diagnostics not in ("light", "full"):
            raise ValueError(
                f"compute_diagnostics must be 'light' or 'full', got "
                f"{compute_diagnostics!r}."
            )

        self.a_grid       = np.asarray(a_grid, dtype=float)
        self.h_KDE        = float(h_KDE)
        self.mode         = mode
        self.gamma_Q      = float(gamma_Q)
        self.lambda_stab  = float(lambda_stab)
        self.gamma_critic = float(gamma_critic)
        self.jitter       = float(jitter)

        self.n_features_q  = int(n_features_q)
        self.n_features_h  = int(n_features_h)
        self.feature_seed  = int(feature_seed)
        self.ell_W         = ell_W
        self.ell_A         = ell_A
        self.ell_Z         = ell_Z
        self.augment_critic_with_simple = bool(augment_critic_with_simple)

        self.integration_grid_size = int(integration_grid_size)
        self.integration_range     = integration_range

        self.clip                = clip
        self.compute_diagnostics = compute_diagnostics

        # ── Trained state (populated by fit) ──────────────────────────────────
        self._fitted: bool = False
        self._n_train: int = 0
        self._phi_Q_map: Optional[PairFeatureMap] = None     # for q on (Z, A)
        self._phi_H_map: Optional[PairFeatureMap] = None     # for critic on (W, A)
        self._integrator: Optional[PolicyFeatureIntegrator] = None
        self._W_train: Optional[np.ndarray] = None
        self._alpha_list: list = []                          # [alpha_k (m_Q,)]
        # Diagnostic lists harvested by DRKernel (parallel to KPVPolicyBridgeQ)
        self._cond_M_list: list = []
        self._cond_Areg_list: list = []
        self._residual_norm_rel: list = []
        self._negative_share_list: list = []
        self._q_raw_p99_list: list = []
        # Kallus-specific extra dict (one per dose)
        self._kallus_diag: list = []

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, W: np.ndarray, A: np.ndarray, Z: np.ndarray) -> "KallusStabilizedPolicyQ":
        W = np.asarray(W, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        Z = np.asarray(Z, dtype=float).ravel()
        n = len(W)
        if not (len(A) == len(Z) == n):
            raise ValueError("W, A, Z must all have the same length.")

        # Resolve bandwidths (median heuristic on the train slice)
        ell_W = self.ell_W if self.ell_W is not None else _median_bandwidth_safe(W)
        ell_A = self.ell_A if self.ell_A is not None else _median_bandwidth_safe(A)
        ell_Z = self.ell_Z if self.ell_Z is not None else _median_bandwidth_safe(Z)
        self.ell_W, self.ell_A, self.ell_Z = ell_W, ell_A, ell_Z

        # Build feature maps on the train fold only (no leakage across folds)
        n_q = min(self.n_features_q, n)
        n_h = min(self.n_features_h, n)

        self._phi_Q_map = PairFeatureMap(
            ell_1=ell_Z, ell_2=ell_A,
            n_features_1=n_q, n_features_2=n_q,
            n_cross=min(n_q, 32),
            augment_simple=False,                      # q-side is "expansion" only
            seed=self.feature_seed,
            scaler=False,
            jitter=self.jitter,
        ).fit(Z, A)

        self._phi_H_map = PairFeatureMap(
            ell_1=ell_W, ell_2=ell_A,
            n_features_1=n_h, n_features_2=n_h,
            n_cross=min(n_h, 32),
            augment_simple=self.augment_critic_with_simple,
            seed=self.feature_seed + 1000,
            scaler=False,
            jitter=self.jitter,
        ).fit(W, A)

        # Materialise train features
        Phi_Q = self._phi_Q_map.transform(Z, A)         # (n, m_Q_total)
        Phi_H = self._phi_H_map.transform(W, A)         # (n, m_H_total)
        m_Q = Phi_Q.shape[1]
        m_H = Phi_H.shape[1]

        # Policy integrator on the train A grid
        self._integrator = PolicyFeatureIntegrator(
            h_policy=self.h_KDE,
            grid_size=self.integration_grid_size,
            grid_range=self.integration_range,
        ).fit_grid(A)

        # Build the Kallus stabilisation matrix S once if needed
        S_inv_factor = None
        cond_S = np.nan
        eff_rank_S = np.nan
        if self.mode == "kallus_stabilized":
            S = (self.lambda_stab / n) * (Phi_H.T @ Phi_H) \
                + self.gamma_critic * np.eye(m_H)
            # Pre-factor S once; reuse for every dose.
            S_jit = S + self.jitter * np.eye(m_H)
            try:
                S_chol = sla.cho_factor(S_jit, lower=True, check_finite=False)
                S_solve = lambda B: sla.cho_solve(S_chol, B, check_finite=False)
            except (sla.LinAlgError, np.linalg.LinAlgError):
                warnings.warn(
                    "KallusStabilizedPolicyQ: S Cholesky failed, using lstsq.",
                    RuntimeWarning,
                )
                S_solve = lambda B: np.linalg.lstsq(S_jit, B, rcond=1e-10)[0]
            S_inv_factor = S_solve
            if self.compute_diagnostics == "full":
                cond_S = float(np.linalg.cond(S))
                S_eig = sla.eigvalsh(S, check_finite=False)
                eff_rank_S = float((S_eig > 1e-8 * S_eig.max()).sum())

        # Cache cheap quantities used in T_pi b
        self._n_train  = n
        self._W_train  = W.copy()

        # ── Per-dose solve ────────────────────────────────────────────────────
        alpha_list = []
        cond_M_list = []
        cond_Areg_list = []
        residual_norm_rel_list = []
        negative_share_list = []
        q_raw_p99_list = []
        kallus_diag_list = []

        # K_h(A_i - a_k) for every dose at once (n, K)
        h = self.h_KDE
        a_grid = self.a_grid
        diff = A[:, None] - a_grid[None, :]                         # (n, K)
        Lambda = np.exp(-0.5 * (diff / h) ** 2) / (h * np.sqrt(2.0 * np.pi))

        for k, a_k in enumerate(a_grid):
            a_k = float(a_k)
            Lambda_k = Lambda[:, k]                                 # (n,)

            # M_k = (Phi_H * Lambda_k[:,None]).T @ Phi_Q / n         (m_H, m_Q)
            M_k = (Phi_H * Lambda_k[:, None]).T @ Phi_Q / n

            # b_k = mean_i T_pi_{a_k} Phi_H(W_i)                     (m_H,)
            # Phase 11 optimisation: closed-form via separable block decomposition
            # of PairFeatureMap (eliminates ~77% of fit wall-clock vs the generic
            # quadrature `integrator.integrate`). Mathematically identical when
            # Phi_H_map.scaler == False (the default in this class).
            b_k = self._phi_H_map.compute_TPI_mean(
                W, a_k, self._integrator,
            )                                                        # (m_H,)

            # Solve for alpha_k
            if self.mode == "ridge_gmm":
                lhs = M_k.T @ M_k + self.gamma_Q * np.eye(m_Q)       # (m_Q, m_Q)
                rhs = M_k.T @ b_k                                    # (m_Q,)
                S_inv_M = None  # not used
                S_inv_b = None
            else:
                # kallus_stabilized
                S_inv_M = S_inv_factor(M_k)                          # (m_H, m_Q)
                S_inv_b = S_inv_factor(b_k)                          # (m_H,)
                lhs = M_k.T @ S_inv_M + self.gamma_Q * np.eye(m_Q)
                rhs = M_k.T @ S_inv_b

            alpha_k = _spd_solve(lhs, rhs, jitter=self.jitter)
            if not np.isfinite(alpha_k).all():
                warnings.warn(
                    f"KallusStabilizedPolicyQ: non-finite alpha at dose index "
                    f"{k}; falling back to zero.",
                    RuntimeWarning,
                )
                alpha_k = np.zeros(m_Q)

            alpha_list.append(alpha_k)

            # ── Diagnostics ──────────────────────────────────────────────────
            q_train = Phi_Q @ alpha_k                                # (n,)
            residual = M_k @ alpha_k - b_k                           # (m_H,)
            b_norm = float(np.linalg.norm(b_k))
            res_rel = float(np.linalg.norm(residual)) / (b_norm + 1e-15)

            residual_norm_rel_list.append(res_rel)
            negative_share_list.append(float(np.mean(q_train < 0.0)))
            q_raw_p99_list.append(float(np.percentile(np.abs(q_train), 99)))

            # cond_M, cond_Areg only in "full" mode (O(n^3) SVD otherwise)
            if self.compute_diagnostics == "full":
                cond_M_list.append(float(np.linalg.cond(M_k)))
                cond_Areg_list.append(float(np.linalg.cond(lhs)))
            else:
                cond_M_list.append(np.nan)
                cond_Areg_list.append(np.nan)

            # Kallus-specific diag (always recorded; cheap)
            if self.mode == "kallus_stabilized":
                # weighted residual = sqrt(r^T S^{-1} r)
                r_in_Sinv = S_inv_factor(residual)
                weighted_res2 = float(residual @ r_in_Sinv)
                weighted_res2 = max(weighted_res2, 0.0)             # guard noise
                weighted_res = float(np.sqrt(weighted_res2))
            else:
                # In ridge_gmm mode, S=I implicitly; weighted_res == residual norm
                weighted_res2 = float(residual @ residual)
                weighted_res = float(np.sqrt(weighted_res2))

            # K_h-weighted q on training data: w_k = K_h(A - a_k) * q_train
            ipw_w = Lambda_k * q_train                               # (n,)
            abs_w = np.abs(ipw_w)
            w_p99 = float(np.percentile(abs_w, 99))
            w_max = float(abs_w.max())
            sum_w  = float(abs_w.sum())
            sum_w2 = float((abs_w ** 2).sum())
            ess = (sum_w ** 2) / (sum_w2 + 1e-15)
            ess_ratio = ess / max(n, 1)

            kd = dict(
                weighted_residual = weighted_res,
                inner_max_value   = weighted_res2,
                alpha_norm        = float(np.linalg.norm(alpha_k)),
                q_raw_min  = float(q_train.min()),
                q_raw_max  = float(q_train.max()),
                q_raw_p95  = float(np.percentile(q_train, 95)),
                q_raw_p99  = float(np.percentile(q_train, 99)),
                weight_p99 = w_p99,
                weight_max = w_max,
                ESS        = float(ess),
                ESS_ratio  = float(ess_ratio),
                cond_S        = cond_S,
                eff_rank_S    = eff_rank_S,
                cond_lhs      = float(np.linalg.cond(lhs))
                                if self.compute_diagnostics == "full" else np.nan,
            )
            kallus_diag_list.append(kd)

        # Cache fitted state
        self._alpha_list           = alpha_list
        self._cond_M_list          = cond_M_list
        self._cond_Areg_list       = cond_Areg_list
        self._residual_norm_rel    = residual_norm_rel_list
        self._negative_share_list  = negative_share_list
        self._q_raw_p99_list       = q_raw_p99_list
        self._kallus_diag          = kallus_diag_list
        self._fitted               = True
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def _q_raw_one(self, Z: np.ndarray, A: np.ndarray, dose_index: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "KallusStabilizedPolicyQ.predict called before fit()."
            )
        Z = np.asarray(Z, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        Phi_Q_test = self._phi_Q_map.transform(Z, A)          # (n_test, m_Q)
        return Phi_Q_test @ self._alpha_list[dose_index]      # (n_test,)

    def _apply_clip(self, q_raw: np.ndarray) -> np.ndarray:
        M_n = self.clip if self.clip is not None else 5.0 * self._n_train ** 0.25
        q_clipped = np.clip(q_raw, -np.inf, M_n)
        frac = float(np.mean(q_raw > M_n))
        if frac > 0.10:
            warnings.warn(
                f"KallusStabilizedPolicyQ: {100 * frac:.1f}% of q clipped at "
                f"M_n={M_n:.2f}. Possible overlap issue.",
                RuntimeWarning,
            )
        return q_clipped

    def predict(self, Z: np.ndarray, A: np.ndarray, dose_index: int) -> np.ndarray:
        return self._apply_clip(self._q_raw_one(Z, A, dose_index))

    def predict_raw(self, Z: np.ndarray, A: np.ndarray, dose_index: int) -> np.ndarray:
        return self._q_raw_one(Z, A, dose_index)

    def predict_all(self, Z: np.ndarray, A: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("predict_all called before fit().")
        Z = np.asarray(Z, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        Phi_Q_test = self._phi_Q_map.transform(Z, A)              # (n_test, m_Q)
        K = len(self.a_grid)
        out = np.zeros((len(Z), K))
        for d in range(K):
            q_raw = Phi_Q_test @ self._alpha_list[d]
            out[:, d] = self._apply_clip(q_raw)
        return out

    def predict_raw_all(self, Z: np.ndarray, A: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("predict_raw_all called before fit().")
        Z = np.asarray(Z, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        Phi_Q_test = self._phi_Q_map.transform(Z, A)
        K = len(self.a_grid)
        out = np.zeros((len(Z), K))
        for d in range(K):
            out[:, d] = Phi_Q_test @ self._alpha_list[d]
        return out

    def get_diagnostics(self) -> dict:
        return dict(
            mode               = self.mode,
            n_train            = self._n_train,
            cond_M_list        = list(self._cond_M_list),
            cond_Areg_list     = list(self._cond_Areg_list),
            residual_norm_rel  = list(self._residual_norm_rel),
            negative_share     = list(self._negative_share_list),
            q_raw_p99          = list(self._q_raw_p99_list),
            kallus_diag        = list(self._kallus_diag),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  KallusMinimaxBridgeH (Phase 11B) -- h-side counterpart to KallusStabilizedPolicyQ
# ══════════════════════════════════════════════════════════════════════════════

class KallusMinimaxBridgeH:
    """
    Finite-feature minimax/stabilized outcome-bridge solver for h0(W, A)
    satisfying the Fredholm condition E[h0(W, A) | Z, A] = E[Y | Z, A].

    Drop-in replacement for KPVBridgeH inside DRKernel. Implements two modes
    that share the same Nystroem features (so any difference between them comes
    from the solve, not from the function class):

        mode = "ridge_gmm"
            (A_mat^T A_mat + gamma_H I) beta = A_mat^T d

        mode = "kallus_stabilized"
            (A_mat^T S_C^{-1} A_mat + gamma_H I) beta = A_mat^T S_C^{-1} d
            where S_C = lambda_stab_h * Phi_C^T Phi_C / n + gamma_critic_h * I.

    Notation
    --------
    - Phi_H : finite-feature map for h(W, A)          shape (n, m_H)
    - Phi_C : finite-feature map for the critic g(Z, A) shape (n, m_C)
    - A_mat = Phi_C.T @ Phi_H / n                      shape (m_C, m_H)
    - d     = Phi_C.T @ Y / n                          shape (m_C,)
    - h_hat(W, A) = Phi_H(W, A) @ beta                 shape (n_test,)
    - plug_in_policy(W, a, h) = compute_TPI(W, a) @ beta

    Interface contract
    ------------------
    DRKernel calls in this exact order (see dr_kernel.py:262-269):
        bridge.fit(W, A, Z, Y)
        bridge.predict(W_test, A_test)
        bridge.plug_in_policy(W_test, a_dose, h_KDE)

    Numerical discipline
    --------------------
    - Never form inv(.). All linear systems use scipy.linalg.cho_factor +
      cho_solve with a fallback chain (sym solve -> lstsq).
    - Jitter ~1e-8 added to all SPD matrices before factorisation.
    - Diagnostics expensive in n^3 (cond_S_C, cond_lhs, eff_rank) only when
      compute_cond=True.

    References
    ----------
    Kallus (2021), JASA 116(534), Eq. 13 stabilized minimax weighting.
    Mastouri et al. (2021), ICML, KPV ridge bridge as comparison baseline.
    """

    has_rkhs_plug_in: bool = True
    is_linear_in_A:   bool = False

    def __init__(
        self,
        # Mode
        mode: str = "kallus_stabilized",
        # Regularisations
        gamma_H: float = 1e-3,
        lambda_stab_h: float = 1.0,
        gamma_critic_h: float = 1e-3,
        jitter: float = 1e-8,
        # Features
        n_features_h: int = 150,
        n_features_critic: int = 150,
        feature_seed: int = 20260428,
        ell_W: Optional[float] = None,
        ell_A: Optional[float] = None,
        ell_Z: Optional[float] = None,
        ell_scale: float = 1.0,
        augment_with_simple: bool = True,
        # Policy integration (cached per h_policy in plug_in_policy)
        integration_grid_size: int = 400,
        integration_range=("auto", "auto"),
        # Diagnostics
        compute_diagnostics: str = "light",
        compute_cond: bool = False,
    ) -> None:
        if mode not in ("ridge_gmm", "kallus_stabilized"):
            raise ValueError(
                f"mode must be 'ridge_gmm' or 'kallus_stabilized', got {mode!r}."
            )
        if compute_diagnostics not in ("light", "full"):
            raise ValueError(
                f"compute_diagnostics must be 'light' or 'full', got "
                f"{compute_diagnostics!r}."
            )

        self.mode             = mode
        self.gamma_H          = float(gamma_H)
        self.lambda_stab_h    = float(lambda_stab_h)
        self.gamma_critic_h   = float(gamma_critic_h)
        self.jitter           = float(jitter)

        self.n_features_h        = int(n_features_h)
        self.n_features_critic   = int(n_features_critic)
        self.feature_seed        = int(feature_seed)
        self.ell_W               = ell_W
        self.ell_A               = ell_A
        self.ell_Z               = ell_Z
        self.ell_scale           = float(ell_scale)
        self.augment_with_simple = bool(augment_with_simple)

        self.integration_grid_size = int(integration_grid_size)
        self.integration_range     = integration_range

        self.compute_diagnostics = compute_diagnostics
        self.compute_cond        = bool(compute_cond)

        # ── Trained state (populated by fit) ──────────────────────────────────
        self._fitted: bool = False
        self._n_train: int = 0
        self._A_train: Optional[np.ndarray] = None
        self._phi_H_map: Optional[PairFeatureMap] = None      # (W, A)
        self._phi_C_map: Optional[PairFeatureMap] = None      # (Z, A)
        self._beta: Optional[np.ndarray] = None
        # Cached integrator (one per h_policy used in plug_in_policy)
        self._integrator_cache: dict = {}                      # {h_policy: PFI}

        # Diagnostics (populated by fit)
        self._h_raw_residual: float = float("nan")
        self._h_weighted_residual: float = float("nan")
        self._h_inner_max_value: float = float("nan")
        self._beta_norm: float = float("nan")
        self._h_n_features: int = 0
        self._h_n_features_critic: int = 0
        self._jitter_used: float = 0.0
        self._fallback_count: int = 0
        self._cond_S_C: float = float("nan")
        self._cond_lhs: float = float("nan")
        self._eff_rank_S_C: float = float("nan")

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        W: np.ndarray,
        A: np.ndarray,
        Z: np.ndarray,
        Y: np.ndarray,
    ) -> "KallusMinimaxBridgeH":
        W = np.asarray(W, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        Z = np.asarray(Z, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        n = len(Y)
        if not (len(W) == len(A) == len(Z) == n):
            raise ValueError("W, A, Z, Y must all have the same length.")

        # Resolve bandwidths (median heuristic on train slice, scaled)
        ell_W = self.ell_W if self.ell_W is not None else _median_bandwidth_safe(W)
        ell_A = self.ell_A if self.ell_A is not None else _median_bandwidth_safe(A)
        ell_Z = self.ell_Z if self.ell_Z is not None else _median_bandwidth_safe(Z)
        ell_W *= self.ell_scale
        ell_A *= self.ell_scale
        ell_Z *= self.ell_scale
        self.ell_W, self.ell_A, self.ell_Z = ell_W, ell_A, ell_Z

        # Build features on train fold only (no leakage across folds)
        n_h = min(self.n_features_h, n)
        n_c = min(self.n_features_critic, n)

        self._phi_H_map = PairFeatureMap(
            ell_1=ell_W, ell_2=ell_A,
            n_features_1=n_h, n_features_2=n_h,
            n_cross=min(n_h, 32),
            augment_simple=self.augment_with_simple,
            seed=self.feature_seed,
            scaler=False,
            jitter=self.jitter,
        ).fit(W, A)

        self._phi_C_map = PairFeatureMap(
            ell_1=ell_Z, ell_2=ell_A,
            n_features_1=n_c, n_features_2=n_c,
            n_cross=min(n_c, 32),
            augment_simple=self.augment_with_simple,
            seed=self.feature_seed + 2000,
            scaler=False,
            jitter=self.jitter,
        ).fit(Z, A)

        # Materialise train features
        Phi_H = self._phi_H_map.transform(W, A)              # (n, m_H)
        Phi_C = self._phi_C_map.transform(Z, A)              # (n, m_C)
        m_H = Phi_H.shape[1]
        m_C = Phi_C.shape[1]
        self._h_n_features = m_H
        self._h_n_features_critic = m_C

        # Build A_mat and d (the moment system)
        A_mat = Phi_C.T @ Phi_H / n                           # (m_C, m_H)
        d     = Phi_C.T @ Y     / n                           # (m_C,)

        # Solve for beta
        if self.mode == "ridge_gmm":
            lhs = A_mat.T @ A_mat + self.gamma_H * np.eye(m_H)   # (m_H, m_H)
            rhs = A_mat.T @ d                                     # (m_H,)
            S_solve = None
        else:
            # kallus_stabilized: build S_C and pre-factor once
            S_C = (self.lambda_stab_h / n) * (Phi_C.T @ Phi_C) \
                  + self.gamma_critic_h * np.eye(m_C)
            S_C_jit = S_C + self.jitter * np.eye(m_C)
            try:
                S_chol = sla.cho_factor(S_C_jit, lower=True, check_finite=False)
                S_solve = lambda B: sla.cho_solve(S_chol, B, check_finite=False)
            except (sla.LinAlgError, np.linalg.LinAlgError):
                warnings.warn(
                    "KallusMinimaxBridgeH: S_C Cholesky failed, using lstsq.",
                    RuntimeWarning,
                )
                self._fallback_count += 1
                S_solve = lambda B: np.linalg.lstsq(S_C_jit, B, rcond=1e-10)[0]

            S_inv_A = S_solve(A_mat)                              # (m_C, m_H)
            S_inv_d = S_solve(d)                                  # (m_C,)
            lhs = A_mat.T @ S_inv_A + self.gamma_H * np.eye(m_H)
            rhs = A_mat.T @ S_inv_d

            if self.compute_cond:
                self._cond_S_C = float(np.linalg.cond(S_C))
                eig_S = sla.eigvalsh(S_C, check_finite=False)
                self._eff_rank_S_C = float((eig_S > 1e-8 * eig_S.max()).sum())

        beta = _spd_solve(lhs, rhs, jitter=self.jitter)
        if not np.isfinite(beta).all():
            warnings.warn(
                "KallusMinimaxBridgeH: non-finite beta; falling back to zero.",
                RuntimeWarning,
            )
            self._fallback_count += 1
            beta = np.zeros(m_H)

        # ── Diagnostics ──────────────────────────────────────────────────────
        residual = A_mat @ beta - d                              # (m_C,)
        d_norm   = float(np.linalg.norm(d))
        self._h_raw_residual  = float(np.linalg.norm(residual)) / (d_norm + 1e-15)
        self._beta_norm       = float(np.linalg.norm(beta))
        self._jitter_used     = self.jitter

        if self.mode == "kallus_stabilized":
            r_in_Sinv = S_solve(residual)
            wres2 = float(residual @ r_in_Sinv)
            wres2 = max(wres2, 0.0)                              # guard noise
            self._h_weighted_residual = float(np.sqrt(wres2))
            self._h_inner_max_value   = wres2
        else:
            # ridge_gmm: S=I implicitly
            self._h_weighted_residual = float(np.linalg.norm(residual))
            self._h_inner_max_value   = float(residual @ residual)

        if self.compute_cond:
            self._cond_lhs = float(np.linalg.cond(lhs))

        # Cache state
        self._A_train          = A.copy()
        self._beta             = beta
        self._n_train          = n
        self._integrator_cache = {}                              # reset per fit
        self._fitted           = True
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, W: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Return h_hat(W_i, A_i) at each test point. Shape (n_test,)."""
        if not self._fitted:
            raise RuntimeError("KallusMinimaxBridgeH.predict before fit().")
        W = np.asarray(W, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        if len(W) != len(A):
            raise ValueError("W and A must have the same length for prediction.")
        Phi_H_test = self._phi_H_map.transform(W, A)             # (n_test, m_H)
        return Phi_H_test @ self._beta                            # (n_test,)

    def _get_integrator(self, h_policy: float) -> "PolicyFeatureIntegrator":
        h_key = float(h_policy)
        if h_key not in self._integrator_cache:
            integ = PolicyFeatureIntegrator(
                h_policy=h_key,
                grid_size=self.integration_grid_size,
                grid_range=self.integration_range,
            ).fit_grid(self._A_train)
            self._integrator_cache[h_key] = integ
        return self._integrator_cache[h_key]

    def plug_in_policy(
        self,
        W: np.ndarray,
        a: float,
        h: float,
    ) -> np.ndarray:
        """
        Compute integral_t h(W_i, t) K_h(t-a) dt for each test W_i.

        Closed-form via PairFeatureMap.compute_TPI (separable):
            T_pi h_hat(W_i) = (T_pi Phi_H(W_i, .)) @ beta
        """
        if not self._fitted:
            raise RuntimeError("KallusMinimaxBridgeH.plug_in_policy before fit().")
        W = np.asarray(W, dtype=float).ravel()
        integrator = self._get_integrator(h)
        Phi_H_TPI = self._phi_H_map.compute_TPI(W, float(a), integrator)  # (n_test, m_H)
        return Phi_H_TPI @ self._beta                                      # (n_test,)

    def get_diagnostics(self) -> dict:
        return dict(
            mode                  = self.mode,
            n_train               = self._n_train,
            h_raw_residual        = self._h_raw_residual,
            h_weighted_residual   = self._h_weighted_residual,
            h_inner_max_value     = self._h_inner_max_value,
            beta_norm             = self._beta_norm,
            h_n_features          = self._h_n_features,
            h_n_features_critic   = self._h_n_features_critic,
            jitter_used           = self._jitter_used,
            fallback_count        = self._fallback_count,
            cond_S_C              = self._cond_S_C,
            cond_lhs              = self._cond_lhs,
            eff_rank_S_C          = self._eff_rank_S_C,
        )
