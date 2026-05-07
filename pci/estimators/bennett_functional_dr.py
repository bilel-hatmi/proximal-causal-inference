"""
BennettFunctionalDR — Doubly-Robust estimator of J(pi_{a,h}) via direct Riesz
representer estimation (Bennett 2022), with K-fold cross-fitting.

Score formula (Bennett Eq. 2.2):
    J_hat_B(a) = mean_i [ T_pi_a h_hat(W_i) + r_hat_pi_a(Z_i,A_i) * (Y_i - h_hat(W_i,A_i)) ]

Cross-fit loop (K folds, same structure as DRKernel):
    Fold k:
      1. Fit h  on train_k  ->  predict h_obs[test_k], h_policy[test_k, :]
      2. Fit r  on train_k  ->  predict r_hat[test_k, :]   (NO Y in r fit)
      3. score[test_k, d] = h_policy[test_k, d] + r_hat[test_k, d] * (Y[test_k] - h_obs[test_k])

Key differences from DRKernel:
  - BennettFunctionalDR does NOT import DRKernel (zero coupling).
  - r_pi estimated via BennettPolicyRiesz (polynomial features) not RKHS q-bridge.
  - No policy weighting K_h(A-a): r_hat directly absorbs the policy.
  - ESS_abs = (sum|r_i|)^2 / sum(r_i^2)  (classical ESS misleading for signed r_pi).

Output (EstimationResult.extra) is backward-compatible with _build_record in
dgp2_bias_diagnostics.py via the alias extra["J_dr"] = J_bennett_grid.

CRITICAL RULES
--------------
* V_hat is O(1) — NOT divided by n. SE = sqrt(V_hat / n).
* Uppercase variance keys: V_total, V_reg, V_corr, V_cov (lesson from Phase 11C bug).
* ASCII-only stdout (Windows cp1252 — no Unicode special chars).
* No np.linalg.inv anywhere.

References
----------
Bennett (2022) — Riesz representer definition, Eq. (2.2).
Mastouri et al. (2021) — KPVBridgeH for h estimation.
docs/notes/bennett_implementation_mapping.md — design decisions.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from sklearn.model_selection import KFold

from pci.dgps.base import DGPSample
from pci.estimators.base import EstimationResult, PCIEstimator
from pci.bridges.kpv import KPVBridgeH
from pci.bridges.bennett_riesz import (
    BennettPolicyRiesz,
    StabilizedBennettPolicyRiesz,
)


class BennettFunctionalDR(PCIEstimator):
    """
    DR estimator for J(pi_{a,h}) using direct Riesz representer estimation.

    h-bridge: KPVBridgeH (same as DRKernel benchmark)
    r-bridge: BennettPolicyRiesz (polynomial features, NO Y in fit)
    Cross-fitting: K-fold (default K=5)

    Parameters
    ----------
    lambda_h : float
        KPVBridgeH ridge regularisation (lambda_1 = lambda_2 = lambda_h).
        Default 3e-5 = KPV-super benchmark value.
    ell_scale : float
        Multiply median-heuristic bandwidths for KPVBridgeH by this factor.
        Default 3.5 = KPV-super benchmark value.
    lambda_r : float
        Riesz ridge regularisation for BennettPolicyRiesz. Default 1e-3.
    degree : int (2 or 3)
        Polynomial degree for the Riesz feature maps. Default 2.
    n_folds : int
        Number of cross-fitting folds. Default 5.
    a_grid : list or array of float, optional
        Dose targets. Default: [1.8, 2.2, 2.6, 3.0] (DGP2 standard grid).
    bandwidth : float, optional
        Policy bandwidth h. Default: Silverman 1.06 * std(A) * n^(-1/5).
    ref_dose_index : int
        Which a_grid entry is reported as psi_hat. Default 2 (a=2.6).
    seed : int
        KFold random state. Default 42.

    API
    ---
    fit(sample, m_true_fn=None) -> self
    estimate()                  -> EstimationResult
    name                        -> str
    """

    def __init__(
        self,
        lambda_h: float = 3e-5,
        ell_scale: float = 3.5,
        lambda_r: float = 1e-3,
        degree: int = 2,
        n_folds: int = 5,
        a_grid: Optional[List[float]] = None,
        bandwidth: Optional[float] = None,
        ref_dose_index: int = 2,
        seed: int = 42,
        # Phase 12B passthrough to BennettPolicyRiesz
        feature_map_type: str = "polynomial",
        n_features: int = 250,
        ell_scale_rff: float = 2.5,
        rff_seed_base: Optional[int] = None,
        # Phase 12B Stage B optional stabilized variant
        stabilized: bool = False,
        lambda_stab: float = 1.0,
        gamma_critic: float = 1e-3,
    ) -> None:
        self.lambda_h = float(lambda_h)
        self.ell_scale = float(ell_scale)
        self.lambda_r = float(lambda_r)
        self.degree = int(degree)
        self.n_folds = int(n_folds)
        self.a_grid = (
            list(a_grid) if a_grid is not None else [1.8, 2.2, 2.6, 3.0]
        )
        self.bandwidth = bandwidth
        self.ref_dose_index = int(ref_dose_index)
        self.seed = int(seed)
        # Phase 12B fields
        self.feature_map_type = str(feature_map_type)
        if self.feature_map_type not in ("polynomial", "rff"):
            raise ValueError(
                f"BennettFunctionalDR: feature_map_type must be 'polynomial' or 'rff', "
                f"got {feature_map_type!r}."
            )
        self.n_features = int(n_features)
        self.ell_scale_rff = float(ell_scale_rff)
        self.rff_seed_base = rff_seed_base
        self.stabilized = bool(stabilized)
        self.lambda_stab = float(lambda_stab)
        self.gamma_critic = float(gamma_critic)
        self._result: Optional[EstimationResult] = None

    @property
    def name(self) -> str:
        if self.feature_map_type == "rff":
            stab_tag = (
                f",stab(ls={self.lambda_stab:.0e},gc={self.gamma_critic:.0e})"
                if self.stabilized else ""
            )
            return (
                f"BennettFunctionalDR(lam_h={self.lambda_h:.0e},"
                f"lam_r={self.lambda_r:.0e},rff={self.n_features},"
                f"ell={self.ell_scale_rff:.2f}{stab_tag})"
            )
        # polynomial (Phase 12A backward compat)
        return (
            f"BennettFunctionalDR(lam_h={self.lambda_h:.0e},"
            f"lam_r={self.lambda_r:.0e},deg={self.degree})"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _silverman_h(A: np.ndarray) -> float:
        return 1.06 * float(np.std(A)) * len(A) ** (-1.0 / 5.0)

    @staticmethod
    def _median_bandwidth(x: np.ndarray) -> float:
        """Median absolute deviation heuristic for RBF lengthscale."""
        x = np.asarray(x, dtype=float).ravel()
        if len(x) > 500:
            rng = np.random.default_rng(0)
            x = rng.choice(x, 500, replace=False)
        diffs = np.abs(x[:, None] - x[None, :])
        med = float(np.median(diffs[diffs > 0]))
        return med if med > 1e-12 else 1.0

    @staticmethod
    def _ess_abs(r: np.ndarray) -> float:
        """ESS_abs = (sum|r|)^2 / sum(r^2). Classical ESS misleading for signed r_pi."""
        sum_abs = float(np.sum(np.abs(r)))
        sum_sq = float(np.sum(r ** 2))
        if sum_sq < 1e-20:
            return 0.0
        return sum_abs ** 2 / sum_sq

    # ── Main entry point ─────────────────────────────────────────────────────

    def fit(
        self,
        sample: DGPSample,
        m_true_fn: Optional[Callable[[float], float]] = None,
    ) -> "BennettFunctionalDR":
        """
        Fit BennettFunctionalDR with K-fold cross-fitting.

        Parameters
        ----------
        sample    : DGPSample
        m_true_fn : optional callable a -> float, for J_policy_true diagnostics.

        Returns
        -------
        self (for chaining)
        """
        n = len(sample.Y)
        Y = np.asarray(sample.Y, dtype=float)
        A = np.asarray(sample.A, dtype=float)
        W = np.asarray(sample.W, dtype=float)
        Z = np.asarray(sample.Z, dtype=float)

        a_grid = np.array(self.a_grid, dtype=float)
        K_doses = len(a_grid)

        # ── Bandwidth ─────────────────────────────────────────────────────────
        h_KDE = (
            float(self.bandwidth) if self.bandwidth is not None
            else self._silverman_h(A)
        )

        # ── Compute J_policy_true if m_true_fn provided ───────────────────────
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
                pass  # J_policy_true stays NaN; not critical

        # ── Median bandwidths for KPVBridgeH (computed once on full data) ────
        ell_W0 = self._median_bandwidth(W) * self.ell_scale
        ell_A0 = self._median_bandwidth(A) * self.ell_scale
        ell_Z0 = self._median_bandwidth(Z) * self.ell_scale

        # ── Per-observation accumulators (filled fold by fold) ────────────────
        h_obs = np.full(n, np.nan)                     # h_hat(W_i, A_i)
        h_policy = np.full((n, K_doses), np.nan)       # T_pi_a h_hat(W_i)
        r_hat = np.full((n, K_doses), np.nan)          # r_hat_a(Z_i, A_i)

        # Per-fold Riesz residual diagnostics
        riesz_resid_folds = np.full((self.n_folds, K_doses), np.nan)
        # Phase 16 spectral diagnostics per fold
        kappa_r_folds = np.full(self.n_folds, np.nan)
        eff_rank_r_folds = np.full(self.n_folds, np.nan)
        residual_norm_r_folds = np.full((self.n_folds, K_doses), np.nan)
        # KPVBridgeH cond diagnostics (only populated when compute_cond=True)
        kpv_h_cond_M_folds = np.full(self.n_folds, np.nan)
        kpv_h_cond_Areg_folds = np.full(self.n_folds, np.nan)
        kpv_h_residual_folds = np.full(self.n_folds, np.nan)

        # ── K-fold cross-fit loop ─────────────────────────────────────────────
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        indices = np.arange(n)

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(indices)):
            W_tr, A_tr, Z_tr, Y_tr = W[train_idx], A[train_idx], Z[train_idx], Y[train_idx]
            W_te, A_te, Z_te       = W[test_idx],  A[test_idx],  Z[test_idx]

            n_tr = len(train_idx)

            # ── 1. Fit h-bridge on training fold ─────────────────────────────
            h_bridge = KPVBridgeH(
                lambda_1=self.lambda_h,
                lambda_2=self.lambda_h,
                ell_W=ell_W0,
                ell_A=ell_A0,
                ell_Z=ell_Z0,
            )
            h_bridge.fit(W_tr, A_tr, Z_tr, Y_tr)

            # ── 2. Predict h_obs and h_policy on test fold ────────────────────
            h_obs[test_idx] = h_bridge.predict(W_te, A_te)
            for d, a_d in enumerate(a_grid):
                h_policy[test_idx, d] = h_bridge.plug_in_policy(W_te, float(a_d), h_KDE)

            # ── 3. Fit r-bridge on training fold (NO Y) ───────────────────────
            # Per-fold rff seed for reproducibility (deterministic given base)
            rff_seed_fold = (
                self.rff_seed_base + 1009 * fold_idx
                if self.rff_seed_base is not None else None
            )
            common_kwargs = dict(
                lambda_r=self.lambda_r,
                degree=self.degree,
                feature_map_type=self.feature_map_type,
                n_features=self.n_features,
                ell_scale=self.ell_scale_rff,
                rff_seed=rff_seed_fold,
            )
            if self.stabilized:
                riesz = StabilizedBennettPolicyRiesz(
                    lambda_stab=self.lambda_stab,
                    gamma_critic=self.gamma_critic,
                    **common_kwargs,
                )
            else:
                riesz = BennettPolicyRiesz(**common_kwargs)
            riesz.fit(W_tr, A_tr, Z_tr, a_grid=a_grid.tolist(), bandwidth=h_KDE)

            # ── 4. Predict r_hat on test fold ──────────────────────────────────
            for d, a_d in enumerate(a_grid):
                r_hat[test_idx, d] = riesz.predict(Z_te, A_te, a_target=float(a_d))

            # ── 5. Riesz residual on test fold (diagnostic) ───────────────────
            for d, a_d in enumerate(a_grid):
                try:
                    res = riesz.riesz_residual(W_te, A_te, Z_te, a_target=float(a_d))
                    riesz_resid_folds[fold_idx, d] = res
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

            # Phase 16 h-bridge spectral (KPVBridgeH; populated when compute_cond=True)
            if hasattr(h_bridge, "_cond_M_list") and len(h_bridge._cond_M_list) > 0:
                vals_M = [v for v in h_bridge._cond_M_list if v is not None and np.isfinite(v)]
                if vals_M:
                    kpv_h_cond_M_folds[fold_idx] = float(np.mean(vals_M))
            if hasattr(h_bridge, "_cond_Areg_list") and len(h_bridge._cond_Areg_list) > 0:
                vals_A = [v for v in h_bridge._cond_Areg_list if v is not None and np.isfinite(v)]
                if vals_A:
                    kpv_h_cond_Areg_folds[fold_idx] = float(np.mean(vals_A))
            if hasattr(h_bridge, "_residual_norm_rel") and len(h_bridge._residual_norm_rel) > 0:
                vals_R = [v for v in h_bridge._residual_norm_rel if v is not None and np.isfinite(v)]
                if vals_R:
                    kpv_h_residual_folds[fold_idx] = float(np.mean(vals_R))

        # ── Compute DR scores ─────────────────────────────────────────────────
        # score_i(a_d) = h_policy_i(a_d) + r_hat_i(a_d) * (Y_i - h_obs_i)
        residual_Y = (Y - h_obs)[:, None]              # (n, 1) broadcast
        correction = r_hat * residual_Y                # (n, K_doses)
        score = h_policy + correction                  # (n, K_doses)

        J_reg = h_policy.mean(axis=0)                  # (K_doses,)
        J_bennett = score.mean(axis=0)                 # (K_doses,)

        # ── Variance decomposition ────────────────────────────────────────────
        # V_total = var(score) per dose, then mean over doses
        # O(1) — NOT divided by n. SE = sqrt(V_hat / n).
        # Canonical helper. Bit-identical to the prior
        # inline np.var(..., ddof=1) + V_cov polarisation identity.
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

        # ── ESS_abs per dose ──────────────────────────────────────────────────
        # r_pi is signed -> classical IPW ESS misleading. Use ESS_abs.
        ESS_abs_grid = np.array([
            self._ess_abs(r_hat[:, d]) for d in range(K_doses)
        ])
        ESS_min = float(np.min(ESS_abs_grid))

        # ── Weight diagnostics (|r_hat| proxies for importance weights) ──────
        r_abs = np.abs(r_hat)
        weight_p99_grid = np.array([
            float(np.percentile(r_abs[:, d], 99)) for d in range(K_doses)
        ])
        weight_max_grid = np.array([
            float(np.max(r_abs[:, d])) for d in range(K_doses)
        ])

        # ── Riesz residual diagnostics ────────────────────────────────────────
        riesz_resid_mean_per_dose = np.nanmean(riesz_resid_folds, axis=0)  # (K_doses,)
        riesz_resid_overall = float(np.nanmean(riesz_resid_folds))

        # ── Jensen gap (placeholder — not computed for Bennett) ──────────────
        # J_true and J_policy_true are not the same for a non-degenerate policy.
        J_true = np.full(K_doses, np.nan)
        jensen_gap = np.full(K_doses, np.nan)

        # ── ref_dose_index validation ─────────────────────────────────────────
        ref_idx = min(self.ref_dose_index, K_doses - 1)

        # ── Build EstimationResult ────────────────────────────────────────────
        extra = {
            # ── Backward compat with _build_record (dgp2_bias_diagnostics) ───
            "J_reg":               J_reg,
            "J_dr":                J_bennett,          # alias for _build_record
            "J_true":              J_true,
            "J_policy_true":       J_policy_true,
            "V_hat_grid":          V_hat_grid,
            "V_reg_grid":          V_reg_grid,
            "V_correction_grid":   V_corr_grid,
            "ESS_grid":            ESS_abs_grid,
            "ESS_ratio_grid":      ESS_abs_grid / n,
            "ESS_min":             ESS_min,
            "ESS_ratio_min":       ESS_min / n,
            "weight_p99_grid":     weight_p99_grid,
            "weight_max_grid":     weight_max_grid,
            "q_clip_fraction":     0.0,                 # not applicable for Bennett
            "q_negative_share_grid": np.zeros(K_doses),# not applicable
            "riesz_residual_grid_mean": riesz_resid_mean_per_dose,
            "cond_M_grid_mean":    np.full(K_doses, np.nan),  # not computed
            "neg_share_grid_mean": np.zeros(K_doses),
            "bandwidth":           h_KDE,
            "jensen_gap":          jensen_gap,
            "mise":                np.nan,
            "a_grid":              a_grid,
            "ref_dose_index":      ref_idx,
            "ref_dose":            float(a_grid[ref_idx]),
            "n_folds":             self.n_folds,
            "lambda_h":            self.lambda_h,
            "lambda_r":            self.lambda_r,
            "degree":              self.degree,
            # Phase 12B record-keeping
            "feature_map_type":    self.feature_map_type,
            "n_features":          self.n_features if self.feature_map_type == "rff" else None,
            "ell_scale_rff":       self.ell_scale_rff if self.feature_map_type == "rff" else None,
            "stabilized":          self.stabilized,
            "lambda_stab":         self.lambda_stab if self.stabilized else None,
            "gamma_critic":        self.gamma_critic if self.stabilized else None,

            # ── UPPERCASE variance scalars (Phase 11C bug lesson) ─────────────
            "V_total":  V_total,
            "V_reg":    V_reg,
            "V_corr":   V_corr,
            "V_cov":    V_cov,

            # ── Bennett-specific diagnostics ──────────────────────────────────
            "riesz_residual_grid": riesz_resid_mean_per_dose,
            "riesz_residual_mean": riesz_resid_overall,
            # Alignment ratio (correction vs plug-in error) — computed as NaN
            # here; add in pilot script where J_true is available.
            "alignment_ratio_grid": np.full(K_doses, np.nan),
            "alignment_mean":       np.nan,

            # Phase 16 canonical spectral diagnostics (mean over folds)
            "kappa_h":         float(np.nanmean(kpv_h_cond_M_folds)),    # KPV M cond
            "eff_rank_h":      float("nan"),                              # not directly available for KPV
            "residual_norm_h": float(np.nanmean(kpv_h_residual_folds)),
            "kappa_r":         float(np.nanmean(kappa_r_folds)),
            "eff_rank_r":      float(np.nanmean(eff_rank_r_folds)),
            "residual_norm_r": float(np.nanmean(residual_norm_r_folds)),
            "residual_norm_r_grid": np.nanmean(residual_norm_r_folds, axis=0),
            # Auxiliary KPV cond
            "kpv_h_cond_M":    float(np.nanmean(kpv_h_cond_M_folds)),
            "kpv_h_cond_Areg": float(np.nanmean(kpv_h_cond_Areg_folds)),
        }

        psi_hat = float(J_bennett[ref_idx])
        V_hat = float(V_hat_grid[ref_idx])

        self._result = EstimationResult(
            psi_hat=psi_hat,
            V_hat=V_hat,
            method_name=self.name,
            n=n,
            extra=extra,
        )
        return self

    def estimate(self) -> EstimationResult:
        """Return the EstimationResult (must call fit() first)."""
        if self._result is None:
            raise RuntimeError("BennettFunctionalDR.estimate: call fit() first.")
        return self._result
