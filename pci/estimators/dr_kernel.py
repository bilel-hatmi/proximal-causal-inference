"""
DRKernel — Doubly-Robust kernel-based dose-response estimator with cross-fitting.

Estimand
--------
For a kernel-localised intervention π_{a,h}(t) = K_h(t - a), the policy value is
    J(π_{a,h}) = ∫ m(t) K_h(t - a) dt
where m(a) = E[Y(a)] is the dose-response function. For a grid {a_k}, returns
the curve k → Ĵ(a_k).

Method
------
Same Version B DR formula as DRDoseResponse:
    Ĵ(a_k) = (1/n) Σᵢ [
        K_h(Aᵢ - a_k) · q̂(Zᵢ, Aᵢ, Xᵢ) · (Yᵢ - ĥ(Wᵢ, Aᵢ, Xᵢ))
        + ∫ ĥ(Wᵢ, t, Xᵢ) K_h(t - a_k) dt
    ]

Differences from DRDoseResponse:
    1. h-bridge is estimated INTERNALLY via KPVBridgeH (true KPV ridge, Mastouri
       2021) — no h_model argument. q-bridge is still passed externally
       (typically OracleBridgeQ for Phase 8 — mode "oracle-q").
    2. K-fold CROSS-FITTING (default K=5): per-fold ĥ^(-k) trained on K-1 folds
       and evaluated on the held-out fold, eliminating own-observation bias.
    3. Plug-in integral uses KPV's closed-form Gauss-Gauss convolution (not
       trapezoidal quadrature).
    4. Reports decomposed diagnostics: ESS_grid + ESS_ratio_grid + V_reg_grid
       + V_correction_grid — separates IPW variance from plug-in noise.
    5. Scalar `psi_hat = J_dr[ref_dose_index]` for harness compatibility (the
       full curve lives in `extra["J_dr"]`).

Validation regime (Phase 8)
---------------------------
DRKernel is validated in the IN-SUPPORT regime with oracle q. Specifically:
    - DGP1: a_grid in [-0.5, 0.5] (A bulk in [-1, 1]) → err ≤ 0.10, ESS > 100
    - DGP2: a_grid in [1.8, 3.0] (A IQR ≈ [1.5, 3.5]) → err ≤ 0.10, ESS > 100

At BOUNDARY doses (e.g. DGP2 a=0.5 or a=4.0), the policy weights collapse:
ESS drops to single digits, IPW variance dominates the DR correction, and the
total error rises to ~0.40 regardless of bridge quality. This is a problem of
local overlap (q is heavy-tailed there), NOT a KPV bridge failure: the plug-in
component J_reg remains accurate at the boundaries (~0.04 error at a=4.0
on DGP2). Phase 9 addresses this via estimated q with weight clipping.

Output
------
EstimationResult with:
    psi_hat       : J_dr at a_grid[ref_dose_index] (default first grid point)
    V_hat         : variance of DR scores at ref dose
    extra         : dict with the full diagnostics:
        J_reg              (K,) plug-in regression curve
        J_dr               (K,) DR estimate
        J_true             (K,) m_true(a_k) pointwise (if m_true_fn given)
        J_policy_true      (K,) ∫ m_true(t) K_h(t-a_k) dt (Gauss-Hermite)
        jensen_gap         (K,) J_true - J_policy_true (≥ 0 if m_true concave)
        V_hat_grid         (K,) total DR-score variance per dose
        V_reg_grid         (K,) plug-in variance per dose
        V_correction_grid  (K,) IPW correction variance per dose
        ESS_grid           (K,) effective sample size per dose
        ESS_ratio_grid     (K,) ESS / n per dose
        ESS_min            scalar minimum ESS across grid (overlap diagnostic)
        ESS_ratio_min      scalar minimum ESS/n
        a_grid             (K,) dose grid passed in
        ref_dose_index     int — which entry is reported as psi_hat
        ref_dose           float — a_grid[ref_dose_index]
        bandwidth          h_KDE actually used
        policy_kernel      "gaussian_density" (normalisation convention)
        alpha_derived      OLS slope of J_dr ~ a_grid (linearity diagnostic)
        mise               mean integrated squared error vs J_true
        n_folds            K
        lambda_1, lambda_2 KPV ridge parameters

Reference
---------
KPV bridge: Mastouri, Zhu, Gretton et al. (2021) ICML.
Cross-fitting pattern: Ying, Tchetgen-Tchetgen (2024+) "Proximal causal
inference for complex longitudinal studies", CF_eval procedure.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.model_selection import KFold

import copy

from pci.dgps.base import DGPSample
from pci.estimators.base import EstimationResult, PCIEstimator
from pci.estimators.kpv_bridge import KPVBridgeH


class DRKernel(PCIEstimator):
    """
    DR kernel-based dose-response estimator with K-fold cross-fitting.

    h-bridge: KPVBridgeH (true KPV ridge, refit per fold).
    q-bridge: passed externally (typically OracleBridgeQ — "oracle-q" mode).

    Constructor
    -----------
    a_grid        : 1D array of dose values to evaluate
    q_model       : object with predict(Z, A, X) -> array (oracle mode),
                    OR object with fit(W,A,Z) + predict_all(Z,A) -> (n,K) (estimated mode).
                    Detection: hasattr(q_model, 'fit') → estimated mode.
    bandwidth     : optional float; default = Silverman 1.06 * std(A) * n^(-1/5)
    kernel        : 'gaussian' (only option for now — matches plug-in convolution)
    n_folds       : int, default 5
    bridge_kwargs : dict of kwargs for KPVBridgeH (e.g. lambda_1, lambda_2)
    random_state  : int, KFold seed (independent of DGP seed)

    Estimated q mode (Phase 9B)
    ---------------------------
    cross_fit_q=True  (default, Phase 9B.2): q re-fitted per fold via copy.deepcopy →
                 valid for inference. q_mode = "<class>_xfit".
    cross_fit_q=False (Phase 9B.1 smoke-test): q fitted once globally →
                 smoke-test only, NOT valid for coverage. q_mode = "<class>_global".

    Output extras
    -------------
    Same as DRDoseResponse + ESS_grid + policy_kernel + n_folds + lambda_1, lambda_2.
    Phase 9B adds: q_estimated, q_mode, ESS_raw_grid, q_raw_* stats,
                   riesz_moment_g1_grid, q_clip_level.
    """

    def __init__(
        self,
        a_grid: np.ndarray,
        q_model,
        bandwidth: Optional[float] = None,
        kernel: str = "gaussian",
        n_folds: int = 5,
        bridge_kwargs: Optional[dict] = None,
        random_state: int = 0,
        ref_dose_index: int = 0,
        cross_fit_q: bool = True,
        h_bridge_factory: Optional[Callable] = None,
    ) -> None:
        """
        Parameters
        ----------
        ref_dose_index : int
            Which entry of `a_grid` is reported as the scalar `psi_hat` in
            EstimationResult (the rest of the curve lives in `extra["J_dr"]`).
            Default 0 (first grid point). Use a_grid.searchsorted(ref) to pick
            the index near a representative dose. Required for compatibility
            with the harness Monte-Carlo loop which expects a scalar estimand.
        """
        self.a_grid = np.asarray(a_grid, dtype=float)
        self.q_model = q_model
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.n_folds = n_folds
        self.bridge_kwargs = dict(bridge_kwargs) if bridge_kwargs else {}
        self.random_state = random_state
        if not (0 <= ref_dose_index < len(self.a_grid)):
            raise ValueError(
                f"ref_dose_index={ref_dose_index} out of range [0, {len(self.a_grid)})"
            )
        self.ref_dose_index = ref_dose_index
        # cross_fit_q=True (default): q is cross-fitted per fold → valid for inference.
        # cross_fit_q=False: q is fitted once globally (Phase 9B.1 smoke-test mode).
        # Only applies when q_model has .fit() (estimated mode); oracle path ignores this.
        self.cross_fit_q = cross_fit_q
        # Phase 11B: optional zero-arg factory for an alternative h-bridge class
        # (must implement fit(W,A,Z,Y), predict(W,A), plug_in_policy(W,a,h)).
        # Default None -> uses KPVBridgeH(**bridge_kwargs) as before.
        self.h_bridge_factory = h_bridge_factory
        self._result: Optional[EstimationResult] = None

    @property
    def name(self) -> str:
        return "DRKernel"

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _kernel(self, u: np.ndarray, h: float) -> np.ndarray:
        """K_h(u) = K(u/h)/h, normalised Gaussian density (matches DRDoseResponse)."""
        if self.kernel == "gaussian":
            return np.exp(-0.5 * (u / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
        raise ValueError(
            f"DRKernel currently supports only 'gaussian' kernel "
            f"(plug-in convolution formula assumes it), got {self.kernel!r}"
        )

    # ── Main entry point ─────────────────────────────────────────────────────

    def fit(
        self,
        sample: DGPSample,
        m_true_fn: Optional[Callable[[float], float]] = None,
    ) -> "DRKernel":
        """
        Fit DRKernel on `sample` with K-fold cross-fitting.

        Parameters
        ----------
        sample    : DGPSample
        m_true_fn : optional callable a -> float for ground-truth dose response.
        """
        n = len(sample.Y)
        Y, A, W, Z = sample.Y, sample.A, sample.W, sample.Z
        X = sample.X if sample.X is not None else np.zeros(n)

        # Bandwidth (Silverman default)
        h_KDE = self.bandwidth if self.bandwidth is not None \
            else 1.06 * float(np.std(A)) * n ** (-1.0 / 5.0)

        K_doses = len(self.a_grid)

        # Per-observation arrays — filled fold by fold
        h_obs = np.full(n, np.nan)                      # ĥ^(-k)(Wᵢ, Aᵢ)
        h_policy = np.full((n, K_doses), np.nan)        # plug-in for each a_k

        # ── q-bridge: oracle (Phase 8) vs estimated (Phase 9B) ───────────────
        # Detection: KPVPolicyBridgeQ has .fit() — OracleBridgeQ does not.
        _q_is_estimated = hasattr(self.q_model, 'fit')

        if not _q_is_estimated:
            # ── Oracle path (backward-compatible) ─────────────────────────────
            # q is global and 1D — same for every dose
            q_obs_1d = self.q_model.predict(Z, A, X)             # (n,)
            q_obs_2d = np.tile(q_obs_1d[:, None], (1, K_doses))  # (n, K)
            q_raw_2d = q_obs_2d.copy()                            # no clipping distinction
            _q_mode = "oracle"

        elif not self.cross_fit_q:
            # ── Phase 9B.1: global q (smoke-test mode, NOT valid for inference) ──
            # q_model.fit(W, A, Z) then predict_all(Z, A) → (n, K)
            self.q_model.fit(W, A, Z)
            q_obs_2d = self.q_model.predict_all(Z, A)             # (n, K) clipped
            if hasattr(self.q_model, 'predict_raw_all'):
                q_raw_2d = self.q_model.predict_raw_all(Z, A)     # (n, K) unclipped
            else:
                q_raw_2d = q_obs_2d.copy()
            _q_mode = type(self.q_model).__name__ + "_global"

        else:
            # ── Phase 9B.2: cross-fitted q (valid for inference) ──────────────
            # q placeholders filled fold-by-fold in the KFold loop below
            q_obs_2d = np.full((n, K_doses), np.nan)
            q_raw_2d = np.full((n, K_doses), np.nan)
            _q_mode = type(self.q_model).__name__ + "_xfit"

        # Accumulators for per-fold per-dose q diagnostics (Correction 2)
        # Only used when _q_is_estimated and self.cross_fit_q.
        cond_M_acc    = np.zeros((K_doses, 2))   # [:,0]=sum, [:,1]=running_max
        cond_Areg_acc = np.zeros((K_doses, 2))
        riesz_res_acc = np.zeros((K_doses, 2))
        neg_share_acc = np.zeros((K_doses, 2))
        # Phase 11: Kallus-specific diagnostics (harvested only if q_k exposes
        # a `_kallus_diag` list -- KallusStabilizedPolicyQ does, KPVPolicyBridgeQ
        # does not, so this is fully backward-compatible).
        kallus_present_any = False
        kallus_wres_acc    = np.zeros((K_doses, 2))   # weighted residual
        kallus_inner_acc   = np.zeros((K_doses, 2))   # inner_max_value
        kallus_anorm_acc   = np.zeros((K_doses, 2))   # alpha_norm
        kallus_ess_acc     = np.zeros((K_doses, 2))   # train-side ESS
        kallus_wp99_acc    = np.zeros((K_doses, 2))   # train-side weight p99

        # ── K-fold cross-fitting ──────────────────────────────────────────────
        kf = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=self.random_state)
        # Phase 11B: harvest h-bridge diagnostics (only populated if a Kallus-h
        # bridge is used; KPVBridgeH does not expose these attributes, so this
        # is fully backward-compatible).
        h_kallus_present = False
        h_kallus_acc = {
            "h_raw_residual":      [],
            "h_weighted_residual": [],
            "h_inner_max_value":   [],
            "beta_norm":           [],
            "h_n_features":        [],
            "h_n_features_critic": [],
        }
        h_method_name = "KPVBridgeH"

        # Phase 16 spectral diagnostics for h-bridge (KPVBridgeH cond_M, cond_Areg)
        kpv_h_cond_M_folds: list = []
        kpv_h_cond_Areg_folds: list = []
        kpv_h_residual_folds: list = []

        for train_idx, test_idx in kf.split(np.arange(n)):
            # h-bridge: always cross-fitted (Phase 8 behaviour unchanged)
            # Phase 11B: factory injection for alternative h-bridges
            if self.h_bridge_factory is not None:
                bridge_k = self.h_bridge_factory().fit(
                    W[train_idx], A[train_idx], Z[train_idx], Y[train_idx]
                )
            else:
                bridge_k = KPVBridgeH(**self.bridge_kwargs).fit(
                    W[train_idx], A[train_idx], Z[train_idx], Y[train_idx]
                )

            # Phase 16: harvest h-bridge spectral (only when compute_cond=True)
            if hasattr(bridge_k, "_cond_M_list") and len(bridge_k._cond_M_list) > 0:
                vals_M = [v for v in bridge_k._cond_M_list if v is not None and np.isfinite(v)]
                if vals_M:
                    kpv_h_cond_M_folds.append(float(np.mean(vals_M)))
            if hasattr(bridge_k, "_cond_Areg_list") and len(bridge_k._cond_Areg_list) > 0:
                vals_A = [v for v in bridge_k._cond_Areg_list if v is not None and np.isfinite(v)]
                if vals_A:
                    kpv_h_cond_Areg_folds.append(float(np.mean(vals_A)))
            if hasattr(bridge_k, "_residual_norm_rel") and len(bridge_k._residual_norm_rel) > 0:
                vals_R = [v for v in bridge_k._residual_norm_rel if v is not None and np.isfinite(v)]
                if vals_R:
                    kpv_h_residual_folds.append(float(np.mean(vals_R)))
            h_obs[test_idx] = bridge_k.predict(W[test_idx], A[test_idx])
            for d, a_d in enumerate(self.a_grid):
                h_policy[test_idx, d] = bridge_k.plug_in_policy(
                    W[test_idx], float(a_d), h_KDE
                )

            # Harvest h-side diagnostics if Kallus
            if hasattr(bridge_k, "_h_weighted_residual"):
                h_kallus_present = True
                h_method_name = type(bridge_k).__name__
                for k in h_kallus_acc:
                    v = getattr(bridge_k, "_" + k, None)
                    if v is not None and np.isfinite(v):
                        h_kallus_acc[k].append(float(v))

            # q-bridge: cross-fitted per fold when cross_fit_q=True
            if _q_is_estimated and self.cross_fit_q:
                q_k = copy.deepcopy(self.q_model)
                q_k.fit(W[train_idx], A[train_idx], Z[train_idx])
                q_obs_2d[test_idx, :] = q_k.predict_all(
                    Z[test_idx], A[test_idx]
                )                                                  # (n_test, K) clipped
                if hasattr(q_k, 'predict_raw_all'):
                    q_raw_2d[test_idx, :] = q_k.predict_raw_all(
                        Z[test_idx], A[test_idx]
                    )                                              # (n_test, K) unclipped
                else:
                    q_raw_2d[test_idx, :] = q_obs_2d[test_idx, :]

                # Accumulate per-dose fold diagnostics (Correction 2)
                for d in range(K_doses):
                    if hasattr(q_k, '_cond_M_list') and d < len(q_k._cond_M_list):
                        v = q_k._cond_M_list[d]
                        cond_M_acc[d, 0] += v
                        cond_M_acc[d, 1] = max(cond_M_acc[d, 1], v)
                    if hasattr(q_k, '_cond_Areg_list') and d < len(q_k._cond_Areg_list):
                        v = q_k._cond_Areg_list[d]
                        cond_Areg_acc[d, 0] += v
                        cond_Areg_acc[d, 1] = max(cond_Areg_acc[d, 1], v)
                    if hasattr(q_k, '_residual_norm_rel') and d < len(q_k._residual_norm_rel):
                        v = q_k._residual_norm_rel[d]
                        riesz_res_acc[d, 0] += v
                        riesz_res_acc[d, 1] = max(riesz_res_acc[d, 1], v)
                    if hasattr(q_k, '_negative_share_list') and d < len(q_k._negative_share_list):
                        v = q_k._negative_share_list[d]
                        neg_share_acc[d, 0] += v
                        neg_share_acc[d, 1] = max(neg_share_acc[d, 1], v)

                # Phase 11: harvest Kallus-specific extras if available
                if hasattr(q_k, '_kallus_diag') and q_k._kallus_diag:
                    kallus_present_any = True
                    for d in range(K_doses):
                        if d >= len(q_k._kallus_diag):
                            continue
                        kd = q_k._kallus_diag[d]
                        for v, acc in [
                            (kd.get("weighted_residual", np.nan), kallus_wres_acc),
                            (kd.get("inner_max_value",   np.nan), kallus_inner_acc),
                            (kd.get("alpha_norm",        np.nan), kallus_anorm_acc),
                            (kd.get("ESS",               np.nan), kallus_ess_acc),
                            (kd.get("weight_p99",        np.nan), kallus_wp99_acc),
                        ]:
                            if np.isfinite(v):
                                acc[d, 0] += v
                                acc[d, 1] = max(acc[d, 1], v)

        # Sanity checks
        if np.any(np.isnan(h_obs)) or np.any(np.isnan(h_policy)):
            raise RuntimeError(
                "DRKernel: h cross-fitting left NaN entries — KFold did not "
                "cover all indices. Check n_folds vs n."
            )
        if _q_is_estimated and self.cross_fit_q and np.any(np.isnan(q_obs_2d)):
            raise RuntimeError(
                "DRKernel: q cross-fitting left NaN entries — some test indices "
                "were not covered. Check n_folds vs n."
            )

        # ── Finalise per-dose q diagnostics ───────────────────────────────────
        if _q_is_estimated and self.cross_fit_q:
            cond_M_grid_mean    = cond_M_acc[:, 0] / self.n_folds
            cond_M_grid_max     = cond_M_acc[:, 1]
            cond_Areg_grid_mean = cond_Areg_acc[:, 0] / self.n_folds
            cond_Areg_grid_max  = cond_Areg_acc[:, 1]
            riesz_res_grid_mean = riesz_res_acc[:, 0] / self.n_folds
            riesz_res_grid_max  = riesz_res_acc[:, 1]
            neg_share_grid_mean = neg_share_acc[:, 0] / self.n_folds
            neg_share_grid_max  = neg_share_acc[:, 1]
        elif _q_is_estimated and not self.cross_fit_q:
            # Global q: read directly from q_model (single fit)
            def _arr(attr):
                return (np.array(getattr(self.q_model, attr))
                        if hasattr(self.q_model, attr)
                        else np.full(K_doses, np.nan))
            cond_M_grid_mean    = _arr('_cond_M_list')
            cond_M_grid_max     = cond_M_grid_mean.copy()
            cond_Areg_grid_mean = _arr('_cond_Areg_list')
            cond_Areg_grid_max  = cond_Areg_grid_mean.copy()
            riesz_res_grid_mean = _arr('_residual_norm_rel')
            riesz_res_grid_max  = riesz_res_grid_mean.copy()
            neg_share_grid_mean = _arr('_negative_share_list')
            neg_share_grid_max  = neg_share_grid_mean.copy()
        else:
            # Oracle: all NaN
            _nan = np.full(K_doses, np.nan)
            (cond_M_grid_mean, cond_M_grid_max,
             cond_Areg_grid_mean, cond_Areg_grid_max,
             riesz_res_grid_mean, riesz_res_grid_max,
             neg_share_grid_mean, neg_share_grid_max) = [_nan.copy() for _ in range(8)]

        # Phase 11: finalise Kallus-specific diagnostics (mean across folds).
        # Only meaningful when q_model exposes _kallus_diag.
        if _q_is_estimated and self.cross_fit_q and kallus_present_any:
            kallus_wres_grid  = kallus_wres_acc[:, 0]  / self.n_folds
            kallus_inner_grid = kallus_inner_acc[:, 0] / self.n_folds
            kallus_anorm_grid = kallus_anorm_acc[:, 0] / self.n_folds
            kallus_ess_grid   = kallus_ess_acc[:, 0]   / self.n_folds
            kallus_wp99_grid  = kallus_wp99_acc[:, 0]  / self.n_folds
        elif _q_is_estimated and not self.cross_fit_q and hasattr(
            self.q_model, '_kallus_diag'
        ) and self.q_model._kallus_diag:
            kallus_wres_grid  = np.array([d.get("weighted_residual", np.nan)
                                          for d in self.q_model._kallus_diag])
            kallus_inner_grid = np.array([d.get("inner_max_value", np.nan)
                                          for d in self.q_model._kallus_diag])
            kallus_anorm_grid = np.array([d.get("alpha_norm", np.nan)
                                          for d in self.q_model._kallus_diag])
            kallus_ess_grid   = np.array([d.get("ESS", np.nan)
                                          for d in self.q_model._kallus_diag])
            kallus_wp99_grid  = np.array([d.get("weight_p99", np.nan)
                                          for d in self.q_model._kallus_diag])
            kallus_present_any = True
        else:
            kallus_wres_grid  = np.full(K_doses, np.nan)
            kallus_inner_grid = np.full(K_doses, np.nan)
            kallus_anorm_grid = np.full(K_doses, np.nan)
            kallus_ess_grid   = np.full(K_doses, np.nan)
            kallus_wp99_grid  = np.full(K_doses, np.nan)

        # Per-dose aggregation
        J_reg = np.zeros(K_doses)
        J_dr = np.zeros(K_doses)
        V_hat_grid = np.zeros(K_doses)           # variance of full DR score
        V_reg_grid = np.zeros(K_doses)            # variance of plug-in component
        V_correction_grid = np.zeros(K_doses)     # variance of IPW correction component
        ESS_grid = np.zeros(K_doses)
        ESS_raw_grid = np.zeros(K_doses)          # ESS using unclipped q
        q_raw_p99_grid = np.zeros(K_doses)        # p99 of |q̂| per dose
        q_negative_share_grid = np.zeros(K_doses) # fraction q < 0 per dose
        weight_p99_grid = np.zeros(K_doses)       # p99 of |K_h * q̂| per dose
        weight_max_grid = np.zeros(K_doses)       # max of |K_h * q̂| per dose

        for d, a_d in enumerate(self.a_grid):
            w_k = self._kernel(A - a_d, h_KDE)           # K_h(Aᵢ - a_d)  (n,)

            # q̂_a for this dose (clipped)
            q_obs_d = q_obs_2d[:, d]                      # (n,)
            q_raw_d = q_raw_2d[:, d]                      # (n,) unclipped

            # Per-dose q diagnostics
            q_raw_p99_grid[d] = float(np.percentile(np.abs(q_raw_d), 99))
            q_negative_share_grid[d] = float(np.mean(q_raw_d < 0.0))

            # Importance weights:  K_h(A-a) * q̂_a(Z,A)
            # (DRKernel multiplies by K_h here — q̂ from KPVPolicyBridgeQ is q only)
            ipw_weights = w_k * q_obs_d                   # (n,)
            ipw_raw = w_k * q_raw_d                       # (n,) for ESS_raw

            # Weight magnitude diagnostics
            abs_weights = np.abs(ipw_weights)
            weight_p99_grid[d] = float(np.percentile(abs_weights, 99))
            weight_max_grid[d] = float(np.max(abs_weights))

            # Effective sample size (clipped weights)
            denom = float(np.sum(ipw_weights ** 2))
            ESS_grid[d] = (
                (float(np.sum(ipw_weights)) ** 2) / denom if denom > 0 else 0.0
            )
            # ESS (raw weights)
            denom_raw = float(np.sum(ipw_raw ** 2))
            ESS_raw_grid[d] = (
                (float(np.sum(ipw_raw)) ** 2) / denom_raw if denom_raw > 0 else 0.0
            )

            correction = ipw_weights * (Y - h_obs)
            plug_in_i = h_policy[:, d]
            scores = correction + plug_in_i
            J_reg[d] = float(np.mean(plug_in_i))
            J_dr[d]  = float(np.mean(scores))
            V_hat_grid[d]        = float(np.mean((scores - J_dr[d]) ** 2))
            V_reg_grid[d]        = float(np.var(plug_in_i, ddof=0))
            V_correction_grid[d] = float(np.var(correction, ddof=0))

        # ESS as a fraction of n (interpretable diagnostic)
        ESS_ratio_grid = ESS_grid / n
        ESS_min = float(np.min(ESS_grid))
        ESS_ratio_min = float(np.min(ESS_ratio_grid))

        # ── Phase 9B extras — Riesz g=1 per dose ─────────────────────────────
        # For global q: read from q_model._riesz_g1_list (training-data diagnostic).
        # For cross-fit q: recompute from final q_obs_2d (test-fold predictions).
        riesz_g1_grid = np.zeros(K_doses)
        for d, a_d in enumerate(self.a_grid):
            w_k_d = self._kernel(A - a_d, h_KDE)          # K_h(A-a_d)
            riesz_g1_grid[d] = float(np.mean(w_k_d * q_obs_2d[:, d]))

        M_n = (
            self.q_model.clip
            if (_q_is_estimated and hasattr(self.q_model, 'clip')
                and self.q_model.clip is not None)
            else (5.0 * n ** 0.25 if _q_is_estimated else float('nan'))
        )
        q_clip_fraction = (
            float(np.mean(q_raw_2d > M_n)) if not np.isnan(M_n) else float('nan')
        )

        # Ground truth (point-wise m_true) AND policy-convolved J_policy_true
        if m_true_fn is not None:
            J_true = np.array([float(m_true_fn(float(a))) for a in self.a_grid])
            mise = float(np.mean((J_dr - J_true) ** 2))
            # J_policy_true(a, h) = ∫ m_true(t) K_h(t - a) dt via Gauss-Hermite
            # (degree 25 → ~1e-8 precision for smooth m_true)
            from numpy.polynomial.hermite_e import hermegauss
            x_q, w_q = hermegauss(25)
            J_policy_true = np.zeros(K_doses)
            for d, a_d in enumerate(self.a_grid):
                t_vals = float(a_d) + h_KDE * x_q
                # Let m_true_fn handle its own domain (DGP2's m_true returns 0
                # for t <= 0; DGP1 is defined on all of R).
                vals = np.array([float(m_true_fn(float(t))) for t in t_vals])
                J_policy_true[d] = float(np.dot(w_q, vals) / np.sqrt(2.0 * np.pi))
            jensen_gap = J_true - J_policy_true   # >= 0 if m_true concave
        else:
            J_true = np.full(K_doses, np.nan)
            J_policy_true = np.full(K_doses, np.nan)
            jensen_gap = np.full(K_doses, np.nan)
            mise = float("nan")
        alpha_derived = float(np.polyfit(self.a_grid, J_dr, 1)[0])

        # Scalar psi_hat at reference dose (compatibility with MC harness)
        psi_hat = float(J_dr[self.ref_dose_index])
        V_hat_scalar = float(V_hat_grid[self.ref_dose_index])

        self._result = EstimationResult(
            psi_hat=psi_hat,                            # at a_grid[ref_dose_index]
            V_hat=V_hat_scalar,                         # variance at ref dose
            method_name=self.name,
            n=n,
            extra={
                "J_reg":             J_reg,
                "J_dr":              J_dr,
                "J_true":            J_true,             # m_true(a_d) point-wise
                "J_policy_true":     J_policy_true,      # ∫ m(t) K_h(t-a_d) dt
                "jensen_gap":        jensen_gap,         # J_true - J_policy_true
                "V_hat_grid":        V_hat_grid,
                "V_reg_grid":        V_reg_grid,
                "V_correction_grid": V_correction_grid,
                "ESS_grid":          ESS_grid,
                "ESS_ratio_grid":    ESS_ratio_grid,
                "ESS_min":           ESS_min,
                "ESS_ratio_min":     ESS_ratio_min,
                "a_grid":            self.a_grid,
                "ref_dose_index":    self.ref_dose_index,
                "ref_dose":          float(self.a_grid[self.ref_dose_index]),
                "bandwidth":         h_KDE,
                "policy_kernel":     "gaussian_density",
                "alpha_derived":     alpha_derived,
                "mise":              mise,
                "n_folds":           self.n_folds,
                "lambda_1":          self.bridge_kwargs.get("lambda_1"),
                "lambda_2":          self.bridge_kwargs.get("lambda_2"),
                # ── Phase 9B extras (q-bridge diagnostics) ───────────────────
                "q_estimated":              _q_is_estimated,
                "q_mode":                   _q_mode,
                "cross_fit_q":              (self.cross_fit_q if _q_is_estimated else False),
                "ESS_raw_grid":             ESS_raw_grid,
                "q_raw_p99_grid":           q_raw_p99_grid,
                "q_negative_share_grid":    q_negative_share_grid,
                "weight_p99_grid":          weight_p99_grid,
                "weight_max_grid":          weight_max_grid,
                "riesz_g1_grid":            riesz_g1_grid,
                "q_clip_level":             M_n,
                "q_clip_fraction":          q_clip_fraction,
                # Per-dose q diagnostics aggregated across folds (Correction 2)
                "cond_M_grid_mean":         cond_M_grid_mean,
                "cond_M_grid_max":          cond_M_grid_max,
                "cond_Areg_grid_mean":      cond_Areg_grid_mean,
                "cond_Areg_grid_max":       cond_Areg_grid_max,
                "riesz_residual_grid_mean": riesz_res_grid_mean,
                "riesz_residual_grid_max":  riesz_res_grid_max,
                "neg_share_grid_mean":      neg_share_grid_mean,
                "neg_share_grid_max":       neg_share_grid_max,
                # ── Phase 11: Kallus-specific extras (NaN if q_model is not
                # KallusStabilizedPolicyQ; otherwise mean across folds) ──────
                "kallus_present":              bool(kallus_present_any),
                "kallus_weighted_residual_grid": kallus_wres_grid,
                "kallus_inner_max_grid":         kallus_inner_grid,
                "kallus_alpha_norm_grid":        kallus_anorm_grid,
                "kallus_ESS_grid":               kallus_ess_grid,
                "kallus_weight_p99_grid":        kallus_wp99_grid,
                # ── Phase 11B: h-side Kallus extras (NaN unless KallusMinimaxBridgeH) ──
                "h_method": h_method_name,
                "h_kallus_present": bool(h_kallus_present),
                "h_raw_residual_mean": (
                    float(np.mean(h_kallus_acc["h_raw_residual"]))
                    if h_kallus_acc["h_raw_residual"] else float("nan")
                ),
                "h_weighted_residual_mean": (
                    float(np.mean(h_kallus_acc["h_weighted_residual"]))
                    if h_kallus_acc["h_weighted_residual"] else float("nan")
                ),
                "h_inner_max_value_mean": (
                    float(np.mean(h_kallus_acc["h_inner_max_value"]))
                    if h_kallus_acc["h_inner_max_value"] else float("nan")
                ),
                "h_beta_norm_mean": (
                    float(np.mean(h_kallus_acc["beta_norm"]))
                    if h_kallus_acc["beta_norm"] else float("nan")
                ),
                # ── Phase 16 canonical spectral diagnostics ──────────────────
                # h-bridge (KPVBridgeH; NaN unless compute_cond=True)
                "kappa_h": (
                    float(np.mean(kpv_h_cond_M_folds))
                    if kpv_h_cond_M_folds else float("nan")
                ),
                "eff_rank_h": float("nan"),  # not directly available for KPV
                "residual_norm_h": (
                    float(np.mean(kpv_h_residual_folds))
                    if kpv_h_residual_folds else float("nan")
                ),
                # r/q-bridge: use cond_M_grid_mean (q-bridge cond), avg across doses
                "kappa_r": (
                    float(np.nanmean(cond_M_grid_mean))
                    if cond_M_grid_mean is not None else float("nan")
                ),
                "eff_rank_r": float("nan"),
                "residual_norm_r": (
                    float(np.nanmean(riesz_res_grid_mean))
                    if riesz_res_grid_mean is not None else float("nan")
                ),
                "residual_norm_r_grid": riesz_res_grid_mean,
            },
        )
        return self

    def estimate(self) -> EstimationResult:
        if self._result is None:
            raise RuntimeError("Call fit() before estimate().")
        return self._result
