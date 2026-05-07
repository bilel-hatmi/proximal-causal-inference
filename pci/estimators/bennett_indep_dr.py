"""
BennettIndepFunctionalDR -- Doubly-robust estimator combining BennettIndepBridgeH
(true Bennett moment-solve h-bridge) with BennettPolicyRiesz (RFF direct Riesz).

Score (Bennett 2022 Eq. 2.2):
    J_hat_B(a) = mean_i [ T_pi_a h_hat(W_i)
                          + r_hat_pi_a(Z_i, A_i) * (Y_i - h_hat(W_i, A_i)) ]

Where:
  - h_hat fitted with BennettIndepBridgeH (joint RFF + critic-stabilised moment)
  - r_hat fitted with BennettPolicyRiesz (RFF mode, Phase 12B winner)

Cross-fit K-fold structure same as BennettFunctionalDR.

Distinction from BennettFunctionalDR
------------------------------------
- BennettFunctionalDR uses KPVBridgeH for h (RKHS 2-stage Mastouri).
- BennettIndepFunctionalDR uses BennettIndepBridgeH for h (joint RFF moment).
- Both use BennettPolicyRiesz for r (RFF mode).

Hence "indep" = independent of KPV.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from sklearn.model_selection import KFold

from pci.dgps.base import DGPSample
from pci.estimators.base import EstimationResult, PCIEstimator
from pci.bridges.bennett_h import BennettIndepBridgeH
from pci.bridges.bennett_riesz import (
    BennettPolicyRiesz,
    StabilizedBennettPolicyRiesz,
)


class BennettIndepFunctionalDR(PCIEstimator):
    """
    Bennett-independent DR estimator: BennettIndepBridgeH (h) + BennettPolicyRiesz (r).

    Parameters
    ----------
    # h_B (BennettIndepBridgeH) hyperparameters
    m_h : int             -- RFF features for phi_H(W, A). Default 1000.
    m_c : int             -- RFF features for phi_C(Z, A) (critic). Default 500.
    ell_h : float or None -- bandwidth for phi_H. None => median heuristic.
    ell_c : float or None -- bandwidth for phi_C. None => median heuristic.
    lambda_h : float      -- Tikhonov on beta. Default 1e-3.
    gamma_critic : float  -- Critic covariance regularisation. Default 1e-3.
    h_seed_base : int     -- Seed base for RFF h (per-fold offset).

    # r_B (BennettPolicyRiesz RFF) hyperparameters
    lambda_r : float      -- Tikhonov on Riesz gamma. Default 1e-2.
    n_features_r : int    -- RFF features for r. Default 500.
    ell_scale_r : float   -- bandwidth for r-side RFF. Default 3.5.
    rff_seed_base_r : int

    # Optional Stabilized Riesz (Phase 12B Stage B)
    stabilized_r : bool   -- Use StabilizedBennettPolicyRiesz. Default False.
    lambda_stab : float
    gamma_critic_r : float

    # Cross-fit
    n_folds : int         -- Default 5.
    a_grid : list         -- Default [1.8, 2.2, 2.6, 3.0].
    bandwidth : float     -- Policy bandwidth. None => Silverman.
    ref_dose_index : int  -- Default 2 (a=2.6).
    seed : int            -- KFold random_state. Default 42.
    """

    def __init__(
        self,
        m_h: int = 1000,
        m_c: int = 500,
        ell_h: Optional[float] = None,
        ell_c: Optional[float] = None,
        lambda_h: float = 1e-3,
        gamma_critic: float = 1e-3,
        h_seed_base: int = 2026000,
        lambda_r: float = 1e-2,
        n_features_r: int = 500,
        ell_scale_r: float = 3.5,
        rff_seed_base_r: int = 2025500,
        stabilized_r: bool = False,
        lambda_stab: float = 1.0,
        gamma_critic_r: float = 1e-3,
        n_folds: int = 5,
        a_grid: Optional[List[float]] = None,
        bandwidth: Optional[float] = None,
        ref_dose_index: int = 2,
        seed: int = 42,
    ) -> None:
        self.m_h = int(m_h)
        self.m_c = int(m_c)
        self.ell_h = ell_h
        self.ell_c = ell_c
        self.lambda_h = float(lambda_h)
        self.gamma_critic = float(gamma_critic)
        self.h_seed_base = int(h_seed_base)
        self.lambda_r = float(lambda_r)
        self.n_features_r = int(n_features_r)
        self.ell_scale_r = float(ell_scale_r)
        self.rff_seed_base_r = int(rff_seed_base_r)
        self.stabilized_r = bool(stabilized_r)
        self.lambda_stab = float(lambda_stab)
        self.gamma_critic_r = float(gamma_critic_r)
        self.n_folds = int(n_folds)
        self.a_grid = list(a_grid) if a_grid is not None else [1.8, 2.2, 2.6, 3.0]
        self.bandwidth = bandwidth
        self.ref_dose_index = int(ref_dose_index)
        self.seed = int(seed)
        self._result: Optional[EstimationResult] = None

    @property
    def name(self) -> str:
        return (
            f"BennettIndep(m_h={self.m_h},m_c={self.m_c},"
            f"lam_h={self.lambda_h:.0e},gam_c={self.gamma_critic:.0e},"
            f"lam_r={self.lambda_r:.0e},m_r={self.n_features_r})"
        )

    @staticmethod
    def _silverman_h(A: np.ndarray) -> float:
        return 1.06 * float(np.std(A)) * len(A) ** (-1.0 / 5.0)

    @staticmethod
    def _ess_abs(r: np.ndarray) -> float:
        sum_abs = float(np.sum(np.abs(r)))
        sum_sq = float(np.sum(r ** 2))
        if sum_sq < 1e-20:
            return 0.0
        return sum_abs ** 2 / sum_sq

    def fit(
        self,
        sample: DGPSample,
        m_true_fn: Optional[Callable[[float], float]] = None,
    ) -> "BennettIndepFunctionalDR":
        n = len(sample.Y)
        Y = np.asarray(sample.Y, dtype=float)
        A = np.asarray(sample.A, dtype=float)
        W = np.asarray(sample.W, dtype=float)
        Z = np.asarray(sample.Z, dtype=float)
        a_grid = np.array(self.a_grid, dtype=float)
        K_doses = len(a_grid)

        h_KDE = (
            float(self.bandwidth) if self.bandwidth is not None
            else self._silverman_h(A)
        )

        # J_policy_true (optional)
        J_policy_true = np.full(K_doses, np.nan)
        if m_true_fn is not None:
            try:
                from numpy.polynomial.hermite_e import hermegauss
                x_q, w_q = hermegauss(25)
                for d, a_d in enumerate(a_grid):
                    t_vals = float(a_d) + h_KDE * x_q
                    vals = np.array([float(m_true_fn(float(t))) for t in t_vals])
                    J_policy_true[d] = float(np.dot(w_q, vals) / np.sqrt(2.0 * np.pi))
            except Exception:
                pass

        # Per-obs accumulators
        h_obs = np.full(n, np.nan)
        h_policy = np.full((n, K_doses), np.nan)
        r_hat = np.full((n, K_doses), np.nan)

        # Per-fold diagnostics
        bridge_residuals = np.full(self.n_folds, np.nan)
        weighted_residuals = np.full(self.n_folds, np.nan)
        coef_norms = np.full(self.n_folds, np.nan)
        cond_Gs = np.full(self.n_folds, np.nan)
        riesz_resid_folds = np.full((self.n_folds, K_doses), np.nan)
        # Phase 16 spectral diagnostics per fold
        kappa_h_folds = np.full(self.n_folds, np.nan)
        eff_rank_h_folds = np.full(self.n_folds, np.nan)
        residual_norm_h_folds = np.full(self.n_folds, np.nan)
        kappa_r_folds = np.full(self.n_folds, np.nan)
        eff_rank_r_folds = np.full(self.n_folds, np.nan)
        residual_norm_r_folds = np.full((self.n_folds, K_doses), np.nan)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        indices = np.arange(n)

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(indices)):
            W_tr, A_tr, Z_tr, Y_tr = W[train_idx], A[train_idx], Z[train_idx], Y[train_idx]
            W_te, A_te, Z_te = W[test_idx], A[test_idx], Z[test_idx]

            # ── Fit Bennett-indep h-bridge ────────────────────────────────────
            h_bridge = BennettIndepBridgeH(
                m_h=self.m_h, m_c=self.m_c,
                ell_h=self.ell_h, ell_c=self.ell_c,
                lambda_h=self.lambda_h,
                gamma_critic=self.gamma_critic,
                seed=self.h_seed_base + fold_idx * 1009,
            )
            h_bridge.fit(W_tr, A_tr, Z_tr, Y_tr)
            h_obs[test_idx] = h_bridge.predict(W_te, A_te)
            for d, a_d in enumerate(a_grid):
                h_policy[test_idx, d] = h_bridge.plug_in_policy(W_te, float(a_d), h_KDE)

            bridge_residuals[fold_idx] = h_bridge.bridge_residual_
            weighted_residuals[fold_idx] = h_bridge.weighted_residual_
            coef_norms[fold_idx] = h_bridge.coefficient_norm_
            cond_Gs[fold_idx] = h_bridge.cond_G_
            # Phase 16 spectral
            kappa_h_folds[fold_idx] = h_bridge.kappa_h_ if h_bridge.kappa_h_ is not None else np.nan
            eff_rank_h_folds[fold_idx] = h_bridge.eff_rank_h_ if h_bridge.eff_rank_h_ is not None else np.nan
            residual_norm_h_folds[fold_idx] = h_bridge.residual_norm_h_ if h_bridge.residual_norm_h_ is not None else np.nan

            # ── Fit RFF Riesz r ───────────────────────────────────────────────
            riesz_seed = self.rff_seed_base_r + 1009 * fold_idx
            common_kwargs = dict(
                lambda_r=self.lambda_r,
                feature_map_type="rff",
                n_features=self.n_features_r,
                ell_scale=self.ell_scale_r,
                rff_seed=riesz_seed,
            )
            if self.stabilized_r:
                riesz = StabilizedBennettPolicyRiesz(
                    lambda_stab=self.lambda_stab,
                    gamma_critic=self.gamma_critic_r,
                    **common_kwargs,
                )
            else:
                riesz = BennettPolicyRiesz(**common_kwargs)
            riesz.fit(W_tr, A_tr, Z_tr, a_grid=a_grid.tolist(), bandwidth=h_KDE)
            for d, a_d in enumerate(a_grid):
                r_hat[test_idx, d] = riesz.predict(Z_te, A_te, a_target=float(a_d))
            for d, a_d in enumerate(a_grid):
                try:
                    riesz_resid_folds[fold_idx, d] = riesz.riesz_residual(
                        W_te, A_te, Z_te, a_target=float(a_d)
                    )
                except Exception:
                    pass
            # Phase 16 r-bridge spectral
            kappa_r_folds[fold_idx] = riesz.kappa_r_ if riesz.kappa_r_ is not None else np.nan
            eff_rank_r_folds[fold_idx] = riesz.eff_rank_r_ if riesz.eff_rank_r_ is not None else np.nan
            if riesz.residual_norm_r_per_dose_ is not None:
                for d, a_d in enumerate(a_grid):
                    residual_norm_r_folds[fold_idx, d] = riesz.residual_norm_r_per_dose_.get(
                        float(a_d), float("nan")
                    )

        # ── Compute DR scores ─────────────────────────────────────────────────
        residual_Y = (Y - h_obs)[:, None]
        correction = r_hat * residual_Y
        score = h_policy + correction

        J_reg = h_policy.mean(axis=0)
        J_bennett = score.mean(axis=0)

        # Variance decomposition (canonical helper, C8.3 May 2026).
        # Bit-identical to the previous inline code: same np.var(..., ddof=1)
        # calls and same V_cov polarisation identity.
        from pci.inference.variance import variance_decomposition
        _vd = variance_decomposition(score, h_policy, correction,
                                      ddof=1, with_scalars=True)
        V_hat_grid = _vd["V_hat_grid"]
        V_reg_grid = _vd["V_reg_grid"]
        V_corr_grid = _vd["V_correction_grid"]
        V_total = _vd["V_total"]
        V_reg = _vd["V_reg"]
        V_corr = _vd["V_corr"]
        V_cov = _vd["V_cov"]

        ESS_abs_grid = np.array([self._ess_abs(r_hat[:, d]) for d in range(K_doses)])
        ESS_min = float(np.min(ESS_abs_grid))
        r_abs = np.abs(r_hat)
        weight_p99_grid = np.array([
            float(np.percentile(r_abs[:, d], 99)) for d in range(K_doses)
        ])
        weight_max_grid = np.array([
            float(np.max(r_abs[:, d])) for d in range(K_doses)
        ])
        riesz_resid_mean_per_dose = np.nanmean(riesz_resid_folds, axis=0)
        riesz_resid_overall = float(np.nanmean(riesz_resid_folds))

        ref_idx = min(self.ref_dose_index, K_doses - 1)
        extra = {
            "J_reg": J_reg, "J_dr": J_bennett,
            "J_true": np.full(K_doses, np.nan),
            "J_policy_true": J_policy_true,
            "V_hat_grid": V_hat_grid,
            "V_reg_grid": V_reg_grid,
            "V_correction_grid": V_corr_grid,
            "ESS_grid": ESS_abs_grid,
            "ESS_ratio_grid": ESS_abs_grid / n,
            "ESS_min": ESS_min,
            "ESS_ratio_min": ESS_min / n,
            "weight_p99_grid": weight_p99_grid,
            "weight_max_grid": weight_max_grid,
            "q_clip_fraction": 0.0,
            "q_negative_share_grid": np.zeros(K_doses),
            "riesz_residual_grid_mean": riesz_resid_mean_per_dose,
            "cond_M_grid_mean": np.full(K_doses, np.nan),
            "neg_share_grid_mean": np.zeros(K_doses),
            "bandwidth": h_KDE,
            "jensen_gap": np.full(K_doses, np.nan),
            "mise": np.nan,
            "a_grid": a_grid,
            "ref_dose_index": ref_idx,
            "ref_dose": float(a_grid[ref_idx]),
            "n_folds": self.n_folds,
            # Bennett-indep h-bridge diagnostics
            "h_bridge_residual_mean": float(np.mean(bridge_residuals)),
            "h_weighted_residual_mean": float(np.mean(weighted_residuals)),
            "h_coefficient_norm_mean": float(np.mean(coef_norms)),
            "h_cond_G_mean": float(np.mean(cond_Gs)),
            "h_bridge_residual_std": float(np.std(bridge_residuals, ddof=1)),
            # Phase 16 canonical spectral diagnostics (mean over folds)
            "kappa_h": float(np.nanmean(kappa_h_folds)),
            "eff_rank_h": float(np.nanmean(eff_rank_h_folds)),
            "residual_norm_h": float(np.nanmean(residual_norm_h_folds)),
            "kappa_r": float(np.nanmean(kappa_r_folds)),
            "eff_rank_r": float(np.nanmean(eff_rank_r_folds)),
            "residual_norm_r": float(np.nanmean(residual_norm_r_folds)),
            "residual_norm_r_grid": np.nanmean(residual_norm_r_folds, axis=0),
            # Hyperparams
            "m_h": self.m_h,
            "m_c": self.m_c,
            "lambda_h": self.lambda_h,
            "gamma_critic": self.gamma_critic,
            "lambda_r": self.lambda_r,
            "n_features_r": self.n_features_r,
            "ell_scale_r": self.ell_scale_r,
            "stabilized_r": self.stabilized_r,
            # Variance scalars (UPPERCASE — Phase 11C lesson)
            "V_total": V_total,
            "V_reg": V_reg,
            "V_corr": V_corr,
            "V_cov": V_cov,
            # Bennett-specific Riesz
            "riesz_residual_grid": riesz_resid_mean_per_dose,
            "riesz_residual_mean": riesz_resid_overall,
            "alignment_ratio_grid": np.full(K_doses, np.nan),
            "alignment_mean": np.nan,
        }

        psi_hat = float(J_bennett[ref_idx])
        V_hat = float(V_hat_grid[ref_idx])
        self._result = EstimationResult(
            psi_hat=psi_hat, V_hat=V_hat,
            method_name=self.name, n=n, extra=extra,
        )
        return self

    def estimate(self) -> EstimationResult:
        if self._result is None:
            raise RuntimeError("BennettIndepFunctionalDR.estimate before fit().")
        return self._result
