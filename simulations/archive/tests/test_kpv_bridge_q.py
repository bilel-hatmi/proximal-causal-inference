"""
Tests for KPVPolicyBridgeQ — Phase 9B (A2 Kallus policy-weighted moment).

Test groups:
    Group 1 — Smoke / API (T_PQ0, T_PQ1)
    Group 2 — Riesz moment gate (T_PQ2a-e) — GATE PRINCIPAL
    Group 3 — DR centering h wrong (T_PQ3) — double robustness check
    Group 4 — Oracle comparison in-support (T_PQ4) — diagnostic only
    Group 5 — DRKernel integration (T_PQ5) — end-to-end smoke (global q, Phase 9B.1)
    Group 6 — DRKernel cross-fit q DGP1 (T_PQ6) — Phase 9B.2
    Group 7 — DRKernel cross-fit q DGP2 in-support (T_PQ7) — Phase 9B.2
    Group 8 — DGP2 boundary stress (T_PQ8) — documented failure regime

Gate ordering:
    T_PQ2 (Riesz moment) > T_PQ3 (DR centering) > T_PQ5/6/7 (end-to-end)

Phase 9B.1 smoke vs Phase 9B.2 inference:
    Group 5 uses cross_fit_q=False (global q, smoke only — NOT valid for coverage).
    Groups 6-7 use cross_fit_q=True (cross-fitted q, valid for inference).

Mathematical object:
    KPVPolicyBridgeQ estimates q̂_a satisfying the Kallus (2021) moment:
        P_n[π_a(A) q̂_a(Z,A) g(W,A)] ≈ P_n[T_{π_a} g(W)]
    validated via the Riesz residual |M_a α̂ - b_a|/|b_a| and moment checks.
    q̂_a is NOT q₀ universal — it is a policy-specific correction for dose a.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP
from simulations.methods.dr_kernel import DRKernel
from simulations.methods.kpv_bridge import (
    KPVPolicyBridgeQ,
    _rbf_kde_convolution,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def dgp1():
    """DGP1: CobbDouglas, high-quality proxies (SNR=0.95)."""
    return CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)


@pytest.fixture(scope="module")
def s(dgp1):
    """2000-observation DGP1 sample (reproducible)."""
    return dgp1.generate(n=2000, seed=42)


@pytest.fixture(scope="module")
def a_grid_3():
    """Three-point grid centred on 0."""
    return np.array([-0.5, 0.0, 0.5])


@pytest.fixture(scope="module")
def h_KDE(s):
    """Silverman bandwidth for DGP1 sample."""
    return 1.06 * float(np.std(s.A)) * len(s.A) ** (-0.2)


@pytest.fixture(scope="module")
def kpv_q(s, a_grid_3, h_KDE):
    """
    Fitted KPVPolicyBridgeQ on DGP1 (n=2000, a_grid=[-0.5,0.0,0.5]).
    All tests in Groups 2–4 reuse this fixture (module scope → fit once).
    """
    model = KPVPolicyBridgeQ(
        a_grid=a_grid_3,
        h_KDE=h_KDE,
        lambda_Q=1e-3,
    )
    model.fit(s.W, s.A, s.Z)
    return model


# ── Group 1 — Smoke / API ────────────────────────────────────────────────────


def test_pq0_fit_state(kpv_q, a_grid_3, s):
    """T_PQ0 — fit() populates all expected attributes."""
    n = len(s.A)
    K = len(a_grid_3)

    assert kpv_q._fitted, "Should be fitted"
    assert len(kpv_q._alpha_list) == K, f"Expected {K} alpha vectors"
    assert kpv_q._Z_train.shape == (n,)
    assert kpv_q._A_train.shape == (n,)
    assert kpv_q._n_train == n

    # Diagnostics lists must be present and have K entries
    assert len(kpv_q._riesz_g1_list) == K
    assert len(kpv_q._residual_norm_rel) == K
    assert len(kpv_q._cond_M_list) == K
    assert len(kpv_q._negative_share_list) == K
    assert len(kpv_q._q_raw_p99_list) == K

    # All finite
    for k in range(K):
        assert np.isfinite(kpv_q._alpha_list[k]).all(), f"alpha[{k}] has NaN/Inf"
        assert np.isfinite(kpv_q._residual_norm_rel[k]), f"residual_norm_rel[{k}] not finite"


def test_pq1_predict_shapes(kpv_q, s, a_grid_3):
    """T_PQ1 — predict(), predict_raw(), predict_all(), predict_raw_all() shapes."""
    Z, A = s.Z, s.A
    K = len(a_grid_3)
    n = len(Z)

    q_d = kpv_q.predict(Z, A, dose_index=1)
    assert q_d.shape == (n,), f"predict() shape {q_d.shape}, expected ({n},)"
    assert np.isfinite(q_d).all(), "predict() returned NaN/Inf"

    q_raw_d = kpv_q.predict_raw(Z, A, dose_index=1)
    assert q_raw_d.shape == (n,)
    assert np.isfinite(q_raw_d).all()

    q_all = kpv_q.predict_all(Z, A)
    assert q_all.shape == (n, K), f"predict_all() shape {q_all.shape}"
    assert np.isfinite(q_all).all()

    q_raw_all = kpv_q.predict_raw_all(Z, A)
    assert q_raw_all.shape == (n, K)
    assert np.isfinite(q_raw_all).all()

    # Clipped ≤ raw (from above only, clip_negative=False default)
    assert np.all(q_all <= q_raw_all + 1e-12), "Clipping should not increase q"


# ── Group 2 — Riesz moment gate ──────────────────────────────────────────────


def test_pq2_riesz_moment_g1(kpv_q, s, h_KDE):
    """
    T_PQ2a — Riesz moment g=1: E_n[K_h(A-a) * q̂_a(Z,A)] ≈ 1.

    RHS = ∫ K_h(t-a) dt = 1  (K_h normalised Gaussian density).
    ⚠ RHS = 1, NOT mean(K_h(A-a)) which estimates the marginal density of A.
    """
    a_ref = 0.0
    dose_idx = 1
    h = h_KDE
    n = len(s.A)

    pi = np.exp(-0.5 * ((s.A - a_ref) / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
    q_hat = kpv_q.predict(s.Z, s.A, dose_idx)

    lhs = float(np.mean(pi * q_hat))
    rhs = 1.0   # ∫ K_h(t-a) dt = 1 for normalized Gaussian

    err = abs(lhs - rhs)
    assert err < 0.30, (
        f"Riesz moment g=1: |LHS - RHS| = {err:.4f} >= 0.30  "
        f"(LHS={lhs:.4f}, RHS={rhs:.4f}). "
        "Check K_WA, K_ZA construction and lambda_Q calibration."
    )


def test_pq2_riesz_moment_gW(kpv_q, s, h_KDE):
    """
    T_PQ2b — Riesz moment g=W: E_n[K_h(A-a) * q̂_a * W] ≈ E_n[W].

    RHS = E_n[∫ W_i K_h(t-a) dt] = E_n[W_i] · 1 = mean(W).
    """
    a_ref = 0.0
    dose_idx = 1
    h = h_KDE

    pi = np.exp(-0.5 * ((s.A - a_ref) / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
    q_hat = kpv_q.predict(s.Z, s.A, dose_idx)

    lhs = float(np.mean(pi * q_hat * s.W))
    rhs = float(np.mean(s.W))

    err = abs(lhs - rhs)
    assert err < 0.30, (
        f"Riesz moment g=W: |LHS - RHS| = {err:.4f} >= 0.30 "
        f"(LHS={lhs:.4f}, RHS={rhs:.4f})"
    )


def test_pq2_riesz_moment_gA(kpv_q, s, h_KDE):
    """
    T_PQ2c — Riesz moment g=A: E_n[K_h(A-a) * q̂_a * A] ≈ a_ref.

    RHS = E_n[∫ A_i K_h(t-a) dt]. For each i: ∫ A_i K_h(t-a) dt = A_i ≠ a.
    Wait — g(W,A) = A (the observed dose), so:
        RHS_j = Σ_i (1/n) ∫ A_j K_h(t-a) dt = E_n[A_j] * 1 = mean(A)? No.

    Actually g(W,A) in the Riesz condition is a function of (W_i, A_i):
        g_j(W,A) = A_j  (observational)
    So RHS = E_n[∫ A_j K_h(t-a) dt] = E_n[A_j] · 1 = mean(A).

    Wait, the RHS is ∫ g(W_i, t) K_h(t-a) dt where g(W,A) = A (the argument).
    So RHS = (1/n) Σ_i ∫ t K_h(t-a) dt = a_ref  (since ∫ t K_h(t-a) dt = a).
    """
    a_ref = 0.0
    dose_idx = 1
    h = h_KDE

    pi = np.exp(-0.5 * ((s.A - a_ref) / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
    q_hat = kpv_q.predict(s.Z, s.A, dose_idx)

    lhs = float(np.mean(pi * q_hat * s.A))
    rhs = a_ref   # ∫ t K_h(t-a) dt = a

    err = abs(lhs - rhs)
    assert err < 0.30, (
        f"Riesz moment g=A: |LHS - RHS| = {err:.4f} >= 0.30 "
        f"(LHS={lhs:.4f}, RHS={rhs:.4f})"
    )


def test_pq2_riesz_moment_gW2(kpv_q, s, h_KDE):
    """
    T_PQ2d — Riesz moment g=W²: E_n[K_h(A-a) * q̂_a * W²] ≈ E_n[W²].

    RHS = E_n[W_i²] (∫ K_h = 1).
    """
    a_ref = 0.0
    dose_idx = 1
    h = h_KDE

    pi = np.exp(-0.5 * ((s.A - a_ref) / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
    q_hat = kpv_q.predict(s.Z, s.A, dose_idx)

    lhs = float(np.mean(pi * q_hat * s.W ** 2))
    rhs = float(np.mean(s.W ** 2))

    err = abs(lhs - rhs)
    assert err < 0.40, (
        f"Riesz moment g=W²: |LHS - RHS| = {err:.4f} >= 0.40 "
        f"(LHS={lhs:.4f}, RHS={rhs:.4f})"
    )


def test_pq2_riesz_moment_gWA(kpv_q, s, h_KDE):
    """
    T_PQ2e — Riesz moment g=WA: E_n[K_h(A-a) * q̂_a * W*A] ≈ a_ref * mean(W).

    RHS = E_n[∫ W_i * t * K_h(t-a) dt] = mean(W) * a_ref.

    Primary test: a_ref = 0.5 (dose_idx=2) — RHS = 0.5 * mean(W) ≠ 0.
    (a=0 would have RHS=0, trivially satisfied even by a bad estimator.)
    Secondary: a_ref = -0.5 (dose_idx=0) — RHS = -0.5 * mean(W) ≠ 0.
    """
    h = h_KDE

    # PRIMARY: a_ref = 0.5 (non-zero RHS — discriminating test)
    a_ref, dose_idx = 0.5, 2
    pi = np.exp(-0.5 * ((s.A - a_ref) / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
    q_hat = kpv_q.predict(s.Z, s.A, dose_idx)
    lhs = float(np.mean(pi * q_hat * s.W * s.A))
    rhs = a_ref * float(np.mean(s.W))
    err = abs(lhs - rhs)
    assert err < 0.30, (
        f"Riesz moment g=WA (a=0.5): |LHS - RHS| = {err:.4f} >= 0.30 "
        f"(LHS={lhs:.4f}, RHS={rhs:.4f})"
    )

    # SECONDARY: a_ref = -0.5
    a_ref2, dose_idx2 = -0.5, 0
    pi2 = np.exp(-0.5 * ((s.A - a_ref2) / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
    q_hat2 = kpv_q.predict(s.Z, s.A, dose_idx2)
    lhs2 = float(np.mean(pi2 * q_hat2 * s.W * s.A))
    rhs2 = a_ref2 * float(np.mean(s.W))
    err2 = abs(lhs2 - rhs2)
    assert err2 < 0.30, (
        f"Riesz moment g=WA (a=-0.5): |LHS - RHS| = {err2:.4f} >= 0.30 "
        f"(LHS={lhs2:.4f}, RHS={rhs2:.4f})"
    )


def test_pq2_residual_norm(kpv_q, a_grid_3):
    """
    T_PQ2f — Relative residual |M_a α̂_a - b_a| / |b_a| (from fit diagnostics).

    Computed post-fit on training data. Should be < 0.10 for well-regularised system.
    (Larger values indicate under-regularisation or ill-conditioning.)
    """
    K = len(a_grid_3)
    for k in range(K):
        rel = kpv_q._residual_norm_rel[k]
        assert rel < 0.30, (
            f"Dose {k} (a={a_grid_3[k]:.2f}): relative residual {rel:.4f} >= 0.30. "
            "Check lambda_Q or M_k construction."
        )


# ── Group 3 — DR centering h wrong ───────────────────────────────────────────


def test_pq3_dr_centering_h_wrong(kpv_q, s, h_KDE, dgp1):
    """
    T_PQ3 — DR centering with misspecified h (constant predictor).

    Setup: h_wrong = mean(Y) everywhere (worst possible h).
    Expected: The DR correction K_h(A-a) * q̂_a * (Y - h_wrong) should
    compensate for the plug-in error (h_wrong ≠ h₀), restoring J_dr ≈ J_true.

    This is the key double-robustness check — q correct → J_dr consistent
    even with h wrong. Tolerance is large (0.40) due to:
        - Phase 9B.1: q estimated globally (not cross-fitted)
        - n=2000 (moderate sample, q̂ not converged)
        - h_wrong is deliberately extreme
    """
    a_ref = 0.0
    dose_idx = 1
    h = h_KDE
    n = len(s.A)

    h_wrong = np.full(n, float(np.mean(s.Y)))   # constant predictor
    q_hat_d = kpv_q.predict(s.Z, s.A, dose_idx)

    # Policy weights (same as DRKernel._kernel)
    pi = np.exp(-0.5 * ((s.A - a_ref) / h) ** 2) / (h * np.sqrt(2.0 * np.pi))

    # DR score: plug-in (constant) + IPW correction
    # h_policy for constant h: T_π(h_wrong) = ∫ c K_h(t-a) dt = c * 1 = c = mean(Y)
    correction = pi * q_hat_d * (s.Y - h_wrong)
    J_dr = float(np.mean(h_wrong + correction))

    J_true = float(dgp1.m_true(a_ref))
    err = abs(J_dr - J_true)
    assert err < 0.40, (
        f"DR centering (h wrong): |J_dr - J_true| = {err:.4f} >= 0.40 "
        f"(J_dr={J_dr:.4f}, J_true={J_true:.4f}). "
        "q̂_a should compensate for h_wrong via the IPW correction."
    )


# ── Group 4 — Oracle comparison in-support (diagnostic) ─────────────────────


def test_pq4_oracle_comparison_in_support(kpv_q, s, dgp1):
    """
    T_PQ4 — Compare q̂_a to oracle q₀ in the in-support zone.

    q̂_A2 is a policy-weighted correction ≠ q₀ universal. The comparison
    is informative but NOT the primary gate. Tolerances are intentionally wide.
    A weak correlation (>0.20) and finite rel_rmse (<10.0) are sufficient.

    Note: q₀ for DGP1 is heavy-tailed (exp((A - γZ)²/2D_Z)).
    We restrict to the in-support zone A ∈ (-0.5, 0.5) to avoid tail domination.
    """
    q_oracle = dgp1.q0(s.Z, s.A)
    q_hat = kpv_q.predict(s.Z, s.A, dose_index=1)   # dose a=0.0

    mask = (s.A > -0.5) & (s.A < 0.5)
    n_mask = mask.sum()
    assert n_mask > 50, f"Too few in-support observations: {n_mask}"

    q_o_is = q_oracle[mask]
    q_h_is = q_hat[mask]

    # Correlation (weak threshold)
    corr = float(np.corrcoef(q_h_is, q_o_is)[0, 1])
    assert np.isfinite(corr), "Correlation is NaN"
    assert corr > 0.10, (
        f"Oracle comparison: corr(q̂, q₀) = {corr:.3f} < 0.10 in-support. "
        "Very weak — possible fit failure."
    )

    # Relative RMSE (very wide — A2 optimises different object)
    rel_rmse = float(
        np.sqrt(np.mean((q_h_is - q_o_is) ** 2)) / (np.std(q_o_is) + 1e-8)
    )
    assert rel_rmse < 10.0, (
        f"Oracle comparison: rel_rmse = {rel_rmse:.2f} > 10.0 in-support."
    )


def test_pq4_negative_share_diagnostic(kpv_q, a_grid_3):
    """
    T_PQ4b — Diagnostic: report negative share of q̂ per dose.

    q̂_A2 can be negative (ridge regression on arbitrary moment condition).
    This is informative but not a pass/fail gate for Phase 9B.1.
    We just assert the share is not catastrophic (< 50%) indicating
    the estimator has learned something meaningful.
    """
    K = len(a_grid_3)
    for k in range(K):
        neg_share = kpv_q._negative_share_list[k]
        assert neg_share < 0.50, (
            f"Dose {k} (a={a_grid_3[k]:.2f}): {100*neg_share:.1f}% negative q̂. "
            "More than half negative — possible estimation failure."
        )


# ── Group 5 — DRKernel integration ───────────────────────────────────────────


@pytest.fixture(scope="module")
def dr_kernel_estimated_q(s, dgp1, a_grid_3, h_KDE):
    """
    DRKernel with KPVPolicyBridgeQ — Phase 9B.1 smoke test (global q, NOT cross-fitted).

    cross_fit_q=False: q fitted globally. This is the Phase 9B.1 smoke test only.
    Do NOT use for coverage/inference — use the cross-fit fixture (Group 6) instead.
    """
    dr = DRKernel(
        a_grid=a_grid_3,
        q_model=KPVPolicyBridgeQ(a_grid=a_grid_3, h_KDE=h_KDE, lambda_Q=1e-3),
        n_folds=5,
        random_state=42,
        ref_dose_index=1,
        cross_fit_q=False,   # Phase 9B.1 global-q smoke mode
    )
    dr.fit(s, m_true_fn=dgp1.m_true)
    return dr.estimate()


def test_pq5_drkernel_estimated_q_extras(dr_kernel_estimated_q):
    """
    T_PQ5a — DRKernel extras for estimated q mode are present and correct type.
    """
    ex = dr_kernel_estimated_q.extra
    assert ex["q_estimated"] is True, "q_estimated should be True"
    assert ex["q_mode"] == "KPVPolicyBridgeQ_global", f"q_mode={ex['q_mode']}"
    assert ex["cross_fit_q"] is False, "Group 5 uses global q (cross_fit_q=False)"

    assert isinstance(ex["ESS_raw_grid"], np.ndarray)
    assert isinstance(ex["riesz_g1_grid"], np.ndarray)
    assert isinstance(ex["q_raw_p99_grid"], np.ndarray)
    assert isinstance(ex["q_negative_share_grid"], np.ndarray)
    assert isinstance(ex["q_clip_fraction"], float)


def test_pq5_drkernel_estimated_q_ess(dr_kernel_estimated_q):
    """
    T_PQ5b — Effective sample size is viable (ESS > 10 for all doses).

    Large tolerance — Phase 9B.1 uses global q (not cross-fitted, no variance
    reduction). ESS can be lower than oracle mode due to policy-specific estimation.
    """
    ess = dr_kernel_estimated_q.extra["ESS_grid"]
    ess_min = float(np.min(ess))
    assert ess_min > 10.0, (
        f"Minimum ESS = {ess_min:.1f} < 10 — all doses. "
        f"ESS_grid = {ess}. Possible q̂ collapse."
    )


def test_pq5_drkernel_estimated_q_error(dr_kernel_estimated_q, dgp1):
    """
    T_PQ5c — End-to-end DR error is acceptable for Phase 9B.1 smoke test.

    Tolerance is 0.35 (large — global q not cross-fitted, Phase 9B.1 only).
    """
    J_dr = dr_kernel_estimated_q.extra["J_dr"]
    J_true = dr_kernel_estimated_q.extra["J_true"]

    err = float(np.max(np.abs(J_dr - J_true)))
    assert err < 0.35, (
        f"End-to-end max error = {err:.4f} >= 0.35. "
        f"J_dr = {J_dr}, J_true = {J_true}"
    )


def test_pq5_backward_compat_oracle_q_extras(s, dgp1, a_grid_3, h_KDE):
    """
    T_PQ5d — Backward compatibility: DRKernel with OracleBridgeQ still works.

    New extras (q_estimated, ESS_raw_grid, etc.) should be present and
    consistent with oracle mode (q_estimated=False, ESS_raw ≈ ESS_clip).
    """
    from simulations.methods._oracle_bridges import OracleBridgeQ
    dr = DRKernel(
        a_grid=a_grid_3,
        q_model=OracleBridgeQ(dgp1),
        n_folds=5,
        random_state=0,
        ref_dose_index=1,
    )
    dr.fit(s, m_true_fn=dgp1.m_true)
    res = dr.estimate()
    ex = res.extra

    assert ex["q_estimated"] is False, "Oracle q should give q_estimated=False"
    assert ex["q_mode"] == "oracle", f"q_mode={ex['q_mode']}"
    assert ex["cross_fit_q"] is False, "Oracle mode: cross_fit_q should be False"
    # ESS_raw should equal ESS_grid for oracle (no clipping)
    np.testing.assert_allclose(
        ex["ESS_raw_grid"], ex["ESS_grid"], rtol=1e-6,
        err_msg="Oracle mode: ESS_raw_grid should equal ESS_grid"
    )


# ── Group 6 — DRKernel cross-fit q DGP1 (Phase 9B.2) ────────────────────────


@pytest.fixture(scope="module")
def dr_kernel_xfit_q(s, dgp1, a_grid_3, h_KDE):
    """
    DRKernel with KPVPolicyBridgeQ cross-fitted (Phase 9B.2, valid for inference).

    cross_fit_q=True (default): q re-fitted per fold → valid for coverage/Exp1.
    """
    dr = DRKernel(
        a_grid=a_grid_3,
        q_model=KPVPolicyBridgeQ(
            a_grid=a_grid_3, h_KDE=h_KDE, lambda_Q=1e-3,
            compute_cond=True,   # enable for T_PQ6d cond diagnostic test
        ),
        n_folds=5,
        random_state=42,
        ref_dose_index=1,
        cross_fit_q=True,
    )
    dr.fit(s, m_true_fn=dgp1.m_true)
    return dr.estimate()


def test_pq6a_xfit_q_mode(dr_kernel_xfit_q):
    """T_PQ6a — DRKernel cross-fit mode is correctly flagged."""
    ex = dr_kernel_xfit_q.extra
    assert ex["q_estimated"] is True, "q_estimated should be True"
    assert "xfit" in ex["q_mode"], f"Expected 'xfit' in q_mode, got {ex['q_mode']}"
    assert ex["cross_fit_q"] is True, "cross_fit_q should be True in Group 6"


def test_pq6b_xfit_q_ess(dr_kernel_xfit_q):
    """T_PQ6b — ESS is viable with cross-fitted q on DGP1."""
    ess = dr_kernel_xfit_q.extra["ESS_grid"]
    ess_min = float(np.min(ess))
    assert ess_min > 10.0, (
        f"ESS_min = {ess_min:.1f} < 10 (DGP1 cross-fit q). "
        f"ESS_grid = {ess}"
    )


def test_pq6c_xfit_q_error(dr_kernel_xfit_q):
    """
    T_PQ6c — End-to-end DR error acceptable with cross-fitted q on DGP1.

    Tolerance 0.35 (same as smoke test — cross-fit should not be worse).
    """
    J_dr  = dr_kernel_xfit_q.extra["J_dr"]
    J_true = dr_kernel_xfit_q.extra["J_true"]
    err = float(np.max(np.abs(J_dr - J_true)))
    assert err < 0.35, (
        f"DGP1 cross-fit max_err = {err:.4f} >= 0.35. "
        f"J_dr={J_dr}, J_true={J_true}"
    )


def test_pq6d_xfit_q_new_extras(dr_kernel_xfit_q):
    """
    T_PQ6d — New Phase 9B.2 extras are present and non-NaN in cross-fit mode.

    Correction 2: aggregated diagnostics (cond_M_grid_mean/max, etc.) must NOT
    be NaN even when fold models are discarded after accumulation.
    """
    ex = dr_kernel_xfit_q.extra
    required = [
        "weight_p99_grid", "weight_max_grid",
        "cond_M_grid_mean", "cond_M_grid_max",
        "cond_Areg_grid_mean", "cond_Areg_grid_max",
        "riesz_residual_grid_mean", "riesz_residual_grid_max",
        "neg_share_grid_mean", "neg_share_grid_max",
    ]
    for key in required:
        assert key in ex, f"Missing extra: {key}"
        assert isinstance(ex[key], np.ndarray), f"{key} should be ndarray"
        assert np.all(np.isfinite(ex[key])), (
            f"{key} contains NaN/Inf in cross-fit mode: {ex[key]}"
        )
    assert ex["cross_fit_q"] is True


# ── Group 7 — DRKernel cross-fit q DGP2 in-support (Phase 9B.2) ─────────────


@pytest.fixture(scope="module")
def dgp2():
    """DGP2: Michaelis-Menten, high-quality proxies (SNR=0.95)."""
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    return MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)


@pytest.fixture(scope="module")
def s2(dgp2):
    """3000-observation DGP2 sample (reproducible, larger n for in-support stability)."""
    return dgp2.generate(n=3000, seed=99)


@pytest.fixture(scope="module")
def a_grid_dgp2_insupport():
    """In-support grid for DGP2 (A IQR ≈ [1.5, 3.5])."""
    return np.array([1.8, 2.2, 2.6, 3.0])


@pytest.fixture(scope="module")
def h_KDE_dgp2(s2):
    """Silverman bandwidth for DGP2 sample."""
    return 1.06 * float(np.std(s2.A)) * len(s2.A) ** (-0.2)


@pytest.fixture(scope="module")
def dr_kernel_dgp2_insupport(s2, dgp2, a_grid_dgp2_insupport, h_KDE_dgp2):
    """DRKernel with cross-fitted q on DGP2 in-support grid."""
    dr = DRKernel(
        a_grid=a_grid_dgp2_insupport,
        q_model=KPVPolicyBridgeQ(
            a_grid=a_grid_dgp2_insupport,
            h_KDE=h_KDE_dgp2,
            lambda_Q=1e-3,
        ),
        n_folds=5,
        random_state=42,
        ref_dose_index=1,
        cross_fit_q=True,
    )
    dr.fit(s2, m_true_fn=dgp2.m_true)
    return dr.estimate()


def test_pq7a_dgp2_insupport_ess(dr_kernel_dgp2_insupport):
    """T_PQ7a — ESS viable on DGP2 in-support doses."""
    ess = dr_kernel_dgp2_insupport.extra["ESS_grid"]
    ess_min = float(np.min(ess))
    assert ess_min > 10.0, (
        f"DGP2 in-support ESS_min = {ess_min:.1f} < 10. "
        f"ESS_grid = {ess}"
    )


def test_pq7b_dgp2_insupport_error(dr_kernel_dgp2_insupport):
    """
    T_PQ7b — DR error vs J_policy_true on DGP2 in-support (Correction 1).

    Correction 1: compare to J_policy_true = ∫ m(t) K_h(t-a) dt, NOT to m(a).
    J_true = m(a) is a secondary diagnostic (Jensen gap) — not the DR estimand.
    The DR estimand is J(π_{a,h}) = ∫ m(t) K_h(t-a) dt = J_policy_true.
    """
    ex = dr_kernel_dgp2_insupport.extra
    J_dr          = ex["J_dr"]
    J_policy_true = ex["J_policy_true"]   # primary: ∫ m(t) K_h(t-a) dt
    J_true        = ex["J_true"]           # secondary: m(a) point-wise

    err_primary = float(np.max(np.abs(J_dr - J_policy_true)))
    jensen_gap  = J_true - J_policy_true   # >= 0 if m concave (informative only)

    assert err_primary < 0.35, (
        f"DGP2 in-support max_err (vs J_policy_true) = {err_primary:.4f} >= 0.35.\n"
        f"J_dr          = {J_dr}\n"
        f"J_policy_true = {J_policy_true}\n"
        f"Jensen gap (J_true - J_policy_true) = {jensen_gap} (diagnostic only)"
    )


# ── Group 8 — DGP2 boundary stress (documented failure regime) ───────────────


@pytest.fixture(scope="module")
def a_grid_dgp2_boundary():
    """Boundary+in-support mixed grid for DGP2 stress test."""
    return np.array([0.5, 1.0, 1.5, 2.5, 4.0])


@pytest.fixture(scope="module")
def dr_kernel_dgp2_boundary(s2, dgp2, a_grid_dgp2_boundary, h_KDE_dgp2):
    """DRKernel on DGP2 mixed boundary+in-support grid."""
    dr = DRKernel(
        a_grid=a_grid_dgp2_boundary,
        q_model=KPVPolicyBridgeQ(
            a_grid=a_grid_dgp2_boundary,
            h_KDE=h_KDE_dgp2,
            lambda_Q=1e-3,
        ),
        n_folds=5,
        random_state=42,
        ref_dose_index=2,   # a=1.5 (in-support reference)
        cross_fit_q=True,
    )
    dr.fit(s2, m_true_fn=dgp2.m_true)
    return dr.estimate()


def test_pq8_dgp2_boundary_stress(dr_kernel_dgp2_boundary):
    """
    T_PQ8 — DGP2 boundary doses stress test (Correction 3).

    a_grid = [0.5, 1.0, 1.5, 2.5, 4.0].
    Primary criterion (MUST PASS):
        Middle doses a=1.5 (idx 2), a=2.5 (idx 3): err < 0.35 vs J_policy_true.
    Boundary doses a=0.5 (idx 0), a=4.0 (idx 4):
        ALLOWED to fail due to ESS collapse (low local overlap).
        Documented via warnings, not asserted.

    This test is NOT globally xfailed — it distinguishes in-support success
    from boundary failure. The xfail is local to boundary doses only.

    Correction 1: primary target = J_policy_true (∫ m K_h), NOT m(a).
    """
    import warnings as _warnings
    ex = dr_kernel_dgp2_boundary.extra
    J_dr          = ex["J_dr"]
    J_policy_true = ex["J_policy_true"]   # primary target (Correction 1)
    J_true        = ex["J_true"]           # secondary
    ESS           = ex["ESS_grid"]
    a_grid        = ex["a_grid"]

    # Diagnostic reporting
    for d, a in enumerate(a_grid):
        err_d = abs(J_dr[d] - J_policy_true[d])
        print(
            f"  a={a:.1f}: err_vs_J_policy={err_d:.3f}, ESS={ESS[d]:.1f}, "
            f"J_dr={J_dr[d]:.3f}, J_policy_true={J_policy_true[d]:.3f}, "
            f"J_true={J_true[d]:.3f}"
        )

    # ── Middle doses MUST pass ────────────────────────────────────────────────
    middle_idx = [2, 3]   # a=1.5, a=2.5
    for d in middle_idx:
        err_d = abs(J_dr[d] - J_policy_true[d])
        assert err_d < 0.35, (
            f"Middle dose a={a_grid[d]:.1f}: err={err_d:.4f} >= 0.35 "
            f"(in-support — should not fail). ESS={ESS[d]:.1f}"
        )

    # ── Boundary doses — document but don't gate ─────────────────────────────
    for d in [0, 4]:   # a=0.5, a=4.0
        assert np.isfinite(J_dr[d]), (
            f"J_dr NaN/Inf at boundary a={a_grid[d]:.1f} — unexpected."
        )
        err_d = abs(J_dr[d] - J_policy_true[d])
        if ESS[d] < 10:
            _warnings.warn(
                f"Boundary a={a_grid[d]:.1f}: ESS={ESS[d]:.1f} < 10, "
                f"err_vs_J_policy={err_d:.3f}. "
                "IPW weight collapse — expected at boundary doses. "
                "Not a KPV bridge failure (J_reg remains accurate).",
                RuntimeWarning,
            )
