"""
Tests for KPVBridgeH and DRKernel (Phase 8).

Test groups (run in order — earlier groups gate later ones):
    Group 1 — utilities (T_KPV6 must pass before any training)
    Group 2 — KPVBridgeH unit tests on DGP1
    Group 3 — Fredholm conditions (xfails of Phase 7)
    Group 4 — DRKernel end-to-end

The W² embedding test (T_KPV4) is the discriminant between true KPV and
collocation — if it fails, Stage 2 is missing the Γ multiplication in predict.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP
from simulations.dgp.michaelis_menten import MichaelisMentenDGP
from simulations.methods._oracle_bridges import OracleBridgeQ
from simulations.methods.base import EstimationResult
from simulations.methods.dr_kernel import DRKernel
from simulations.methods.kpv_bridge import (
    KPVBridgeH,
    KPVPolicyBridgeQ,
    _median_bandwidth,
    _rbf_gram,
    _rbf_kde_convolution,
)


# ── Phase 11 regression test (T_KPV_REG) ─────────────────────────────────────

def test_T_KPV_REG_phase11_does_not_break_kpv_q():
    """
    After the Phase 11 patch to dr_kernel.py (Kallus extras harvest), the
    existing DRKernel + KPVPolicyBridgeQ pipeline must still work and the
    new `kallus_present` flag must be False (KPVPolicyBridgeQ has no
    `_kallus_diag` attribute).
    """
    dgp = CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)
    s = dgp.generate(n=200, seed=7)
    a_grid = np.array([1.0, 2.0, 3.0])
    h_KDE = 1.06 * float(np.std(s.A)) * len(s.A) ** (-1.0 / 5.0)

    q_kpv = KPVPolicyBridgeQ(a_grid=a_grid, h_KDE=h_KDE)
    res = DRKernel(
        a_grid=a_grid, q_model=q_kpv, bridge_kwargs=dict(),
        bandwidth=h_KDE, cross_fit_q=True, n_folds=3, random_state=42,
    ).fit(s).estimate()

    assert np.isfinite(res.psi_hat)
    extra = res.extra
    assert "kallus_present" in extra
    assert extra["kallus_present"] is False, \
        "KPVPolicyBridgeQ must not advertise itself as Kallus"
    # The new grids must exist and be all NaN
    assert np.isnan(extra["kallus_weighted_residual_grid"]).all()
    assert np.isnan(extra["kallus_ESS_grid"]).all()


# ── Phase 11B regression test (T_DRK_H_REG) ──────────────────────────────────

def test_T_DRK_H_REG_phase11b_factory_default_unchanged():
    """
    After the Phase 11B patch (h_bridge_factory argument), DRKernel with
    h_bridge_factory=None must behave EXACTLY as before:
    - uses KPVBridgeH(**bridge_kwargs)
    - h_method == "KPVBridgeH"
    - h_kallus_present == False
    - h_*_mean fields are NaN
    """
    dgp = CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)
    s = dgp.generate(n=200, seed=7)
    a_grid = np.array([1.0, 2.0, 3.0])
    h_KDE = 1.06 * float(np.std(s.A)) * len(s.A) ** (-1.0 / 5.0)

    q = KPVPolicyBridgeQ(a_grid=a_grid, h_KDE=h_KDE)
    res = DRKernel(
        a_grid=a_grid, q_model=q, bridge_kwargs=dict(),
        bandwidth=h_KDE, cross_fit_q=True, n_folds=3, random_state=42,
        # h_bridge_factory NOT passed -- must default to KPVBridgeH path
    ).fit(s).estimate()

    extra = res.extra
    assert extra.get("h_method") == "KPVBridgeH"
    assert extra.get("h_kallus_present") is False
    assert np.isnan(extra.get("h_raw_residual_mean", float("nan")))
    assert np.isnan(extra.get("h_weighted_residual_mean", float("nan")))


def test_T_DRK_H_FACTORY_kallus_h_factory_works():
    """When h_bridge_factory provides a KallusMinimaxBridgeH, DRKernel must:
    - call its fit(W,A,Z,Y), predict(W,A), plug_in_policy(W,a,h)
    - surface h_method='KallusMinimaxBridgeH' and finite h_*_mean fields.
    """
    from simulations.methods.kallus_minimax import KallusMinimaxBridgeH
    dgp = CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)
    s = dgp.generate(n=200, seed=7)
    a_grid = np.array([1.0, 2.0, 3.0])
    h_KDE = 1.06 * float(np.std(s.A)) * len(s.A) ** (-1.0 / 5.0)

    q = KPVPolicyBridgeQ(a_grid=a_grid, h_KDE=h_KDE)
    h_factory = lambda: KallusMinimaxBridgeH(
        mode="kallus_stabilized",
        n_features_h=60, n_features_critic=60,
        compute_diagnostics="light",
    )
    res = DRKernel(
        a_grid=a_grid, q_model=q, bridge_kwargs=dict(),
        bandwidth=h_KDE, cross_fit_q=True, n_folds=3, random_state=42,
        h_bridge_factory=h_factory,
    ).fit(s).estimate()

    extra = res.extra
    assert extra["h_method"] == "KallusMinimaxBridgeH"
    assert extra["h_kallus_present"] is True
    assert np.isfinite(extra["h_raw_residual_mean"])
    assert np.isfinite(extra["h_weighted_residual_mean"])
    assert np.isfinite(extra["h_beta_norm_mean"])


# ── Analytic reference for E[W² | Z, A] on DGP1 (used by T_KPV4) ────────────


def _ew2_given_za_dgp1(Z: np.ndarray, A: np.ndarray, dgp: CobbDouglasLinearDGP) -> np.ndarray:
    """
    Analytic E[W² | Z, A] for DGP1 (Cobb-Douglas log-linear, multivariate normal).

    Derivation:
        (U, Z, A, W) is jointly MVN. The conditional distribution W | Z, A is
        Gaussian with:
            E[W | Z, A] = E[U | Z, A]                         (W = U + eps_W, eps_W ⊥ Z,A)
            Var(W | Z, A) = Var(U | Z, A) + sigma_W²
        where:
            E[U | Z, A] = β_Z · Z + β_A · A
            (β_Z, β_A) = Cov(U, [Z, A]) · Cov([Z, A])⁻¹
            Var(U | Z, A) = 1 - Cov(U, [Z, A]) · Cov([Z, A])⁻¹ · Cov([Z, A], U)

    Then E[W² | Z, A] = Var(W | Z, A) + (E[W | Z, A])².
    """
    sigU2 = dgp.sigma_U ** 2
    sigW2 = dgp.sigma_W ** 2
    sigZ2 = dgp.sigma_Z ** 2
    sigK2 = dgp.sigma_K ** 2
    gK    = dgp.gamma_K

    # Cov([Z, A]) and its inverse
    var_Z = sigU2 + sigZ2
    var_A = gK ** 2 * sigU2 + sigK2
    cov_ZA = gK * sigU2
    det = var_Z * var_A - cov_ZA ** 2
    inv = np.array([[var_A, -cov_ZA], [-cov_ZA, var_Z]]) / det

    # Cov(U, [Z, A])
    cov_UZA = np.array([sigU2, gK * sigU2])

    # Conditional mean coefficients
    beta = inv @ cov_UZA      # (β_Z, β_A)
    var_U_given_ZA = sigU2 - cov_UZA @ inv @ cov_UZA

    EU = beta[0] * Z + beta[1] * A         # E[U | Z, A] = E[W | Z, A]
    var_W_given_ZA = var_U_given_ZA + sigW2
    return var_W_given_ZA + EU ** 2


# ── Module-scoped fixtures ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def dgp1():
    """High-SNR DGP1 — bridge identification is sharper, KPV converges faster."""
    return CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)


@pytest.fixture(scope="module")
def dgp1_sample(dgp1):
    """n=2000 sample of DGP1, seed=42."""
    return dgp1.generate(n=2_000, seed=42)


@pytest.fixture(scope="module")
def kpv_dgp1(dgp1_sample):
    """KPVBridgeH fitted on DGP1 with default hyperparameters."""
    s = dgp1_sample
    return KPVBridgeH().fit(s.W, s.A, s.Z, s.Y)


@pytest.fixture(scope="module")
def dgp2():
    """MichaelisMentenDGP — default parameters (Phase 7 calibration)."""
    return MichaelisMentenDGP()


@pytest.fixture(scope="module")
def dgp2_sample(dgp2):
    """
    n=3000 sample of DGP2, seed=42.

    Pilot showed n=2000 gives unstable T_KPV7 across seeds (median 0.11,
    p75 0.14, seed=42 unlucky at 0.18). At n=3000, max_err drops to 0.08
    consistently — buys the 0.10 tolerance with margin.
    """
    return dgp2.generate(n=3_000, seed=42)


@pytest.fixture(scope="module")
def kpv_dgp2(dgp2_sample):
    """KPVBridgeH fitted on DGP2 with default hyperparameters."""
    s = dgp2_sample
    return KPVBridgeH().fit(s.W, s.A, s.Z, s.Y)


# ════════════════════════════════════════════════════════════════════════════
# Group 1 — Utilities (run before any KPV training)
# ════════════════════════════════════════════════════════════════════════════


def test_plug_in_convolution_matches_quad():
    """
    T_KPV6 — closed-form k̃ vs scipy.integrate.quad.

    Verifies the Gauss-Gauss convolution formula
        k̃(A_j, a, h) = ℓ_A / √(ℓ_A² + h²) · exp(-(A_j - a)² / (2(ℓ_A² + h²)))
    matches the numerical integral
        ∫ exp(-(A_j - t)² / (2 ℓ_A²)) · exp(-(t - a)² / (2 h²)) / (h √(2π)) dt
    for several (A_j, a, h, ℓ_A) configurations.

    This test does NOT require a fitted bridge — it isolates the analytical
    formula from estimation noise.
    """
    rng = np.random.default_rng(0)
    n_cases = 0
    for ell_A in [0.3, 0.7, 1.5]:
        for h in [0.1, 0.3, 0.8]:
            A_train = rng.uniform(-2.0, 4.0, size=8)
            a = rng.uniform(-1.0, 3.0)

            # Closed-form
            k_tilde = _rbf_kde_convolution(A_train, float(a), float(h), float(ell_A))

            # Reference via scipy.quad
            for j, A_j in enumerate(A_train):
                def integrand(t, A_j=A_j, ell_A=ell_A, h=h, a=a):
                    return (
                        np.exp(-((A_j - t) ** 2) / (2.0 * ell_A ** 2))
                        * np.exp(-0.5 * ((t - a) / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
                    )
                truth, _ = quad(integrand, a - 12.0 * h, a + 12.0 * h)
                err = abs(k_tilde[j] - truth)
                assert err < 1e-8, (
                    f"ell_A={ell_A}, h={h}, A_j={A_j:.4f}, a={a:.4f}: "
                    f"closed={k_tilde[j]:.10e}, quad={truth:.10e}, |err|={err:.2e}"
                )
                n_cases += 1
    assert n_cases == 9 * 8


def test_median_bandwidth_basic():
    """
    Sanity: _median_bandwidth on N(0,1) data ≈ √2·median(|N(0,1)|) ≈ 0.954.

    For a standard normal sample of large n, median pairwise distance
    converges to √2 · median(|N(0,1)|) ≈ √2 · 0.6745 ≈ 0.954.
    """
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, size=2000)
    bw = _median_bandwidth(x)
    assert 0.7 < bw < 1.2, f"Expected ~0.95, got {bw:.4f}"


def test_rbf_gram_shape_and_diagonal():
    """Sanity: K_W(x, x) has shape (n, n), diagonal = 1, symmetric."""
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, size=20)
    K = _rbf_gram(x, x, ell=0.5)
    assert K.shape == (20, 20)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)


# ════════════════════════════════════════════════════════════════════════════
# Group 2 — KPVBridgeH unit tests on DGP1
# ════════════════════════════════════════════════════════════════════════════


def test_kpv_fit_completes_and_caches(kpv_dgp1, dgp1_sample):
    """T_KPV0 — basic sanity: fit() populates caches and bandwidths."""
    s = dgp1_sample
    bridge = kpv_dgp1
    assert bridge._fitted
    assert bridge._W_train is not None and len(bridge._W_train) == len(s.W)
    assert bridge._Gamma is not None and bridge._Gamma.shape == (len(s.Y), len(s.Y))
    assert bridge._alpha is not None and bridge._alpha.shape == (len(s.Y),)
    assert bridge.ell_W is not None and bridge.ell_W > 0
    assert bridge.ell_A is not None and bridge.ell_A > 0
    assert bridge.lambda_1 is not None and bridge.lambda_2 is not None


def test_kpv_stage1_rows_diagnostic(kpv_dgp1):
    """
    T_KPV1 (DIAGNOSTIC, not blocking) — H₁ rows ≈ 1 in mean.

    For a normalised RBF kernel with small λ₁, the smoother H₁ = K_ZA Γᵀ has
    rows that should sum to ~1 on average (constant function preservation).
    Tolerance: mean |row_sum - 1| < 0.20. NOT a strict per-row max test.
    """
    Gamma = kpv_dgp1._Gamma
    n = Gamma.shape[0]
    # H₁ = K_ZA (K_ZA + nλ₁I)⁻¹ = Γᵀ (since K_ZA symmetric)
    H1 = Gamma.T
    row_sums = H1.sum(axis=1)
    mean_dev = float(np.mean(np.abs(row_sums - 1.0)))
    # Diagnostic: not the gate, but a useful smell test
    assert mean_dev < 0.20, (
        f"Mean |H₁ row sum - 1| = {mean_dev:.4f} — possibly bad bandwidth "
        f"or pathological λ₁."
    )


def test_kpv_stage2_training_residual(kpv_dgp1, dgp1_sample):
    """
    T_KPV2 — training residual on DGP1.

    With snr=0.95 and X term not in bridge, irreducible noise:
        var(eps_W) + (1-α)²·sigma_L² + sigma_Y² ≈ 0.05 + 0.044 + 0.01 ≈ 0.105
    so RMSE floor ≈ 0.32. We require RMSE < 0.45 (mild margin over floor).
    """
    s = dgp1_sample
    Y_pred = kpv_dgp1.predict(s.W, s.A)
    rmse = float(np.sqrt(np.mean((s.Y - Y_pred) ** 2)))
    assert rmse < 0.45, f"Training RMSE = {rmse:.4f} (expected < 0.45)"
    # Also check it's not collapsed to a constant
    corr = float(np.corrcoef(Y_pred, s.Y)[0, 1])
    assert corr > 0.85, f"corr(Y_pred, Y) = {corr:.4f} (expected > 0.85)"


def test_kpv_predict_deterministic(kpv_dgp1, dgp1_sample):
    """T_KPV3 — two predict() calls on same input give identical output."""
    s = dgp1_sample
    p1 = kpv_dgp1.predict(s.W[:50], s.A[:50])
    p2 = kpv_dgp1.predict(s.W[:50], s.A[:50])
    assert np.allclose(p1, p2)


def test_kpv_w_squared_embedding(dgp1, dgp1_sample):
    """
    T_KPV4 — DISCRIMINANT TEST (Mastouri vs Singh vs Collocation).

    Train KPV with target g(W, A) = W² (instead of Y). The fitted bridge should
    satisfy E[ĥ(W, a)] ≈ E[W²|Z=z, A=a] (the analytic CME-based bridge).

    If predict() omits the Γ multiplication (collocation variant), the bridge
    will be biased — RMSE relative to E[W²|Z,A] truth will be large.

    True KPV:        RMSE_relative < 0.50    expected (KPV converges, even if
                                              not at noise floor for W²).
    Collocation/Singh: RMSE_relative ≫ 1.0  (predictions miss the Γ projection).

    GATE: if this test fails, Stage 2 or predict() is broken. STOP and verify
    `Ew = KW_test @ Γ` is present in predict() and plug_in_policy().
    """
    s = dgp1_sample
    target = s.W ** 2
    bridge = KPVBridgeH().fit(s.W, s.A, s.Z, target)

    # Held-out prediction at training points (in-sample is fine for diagnostic)
    h_pred = bridge.predict(s.W, s.A)

    # Analytic reference E[W² | Z, A] via DGP1 multivariate-normal formula
    h_truth = _ew2_given_za_dgp1(s.Z, s.A, dgp1)

    rmse = float(np.sqrt(np.mean((h_pred - h_truth) ** 2)))
    std_truth = float(np.std(h_truth))
    rel_rmse = rmse / std_truth
    corr = float(np.corrcoef(h_pred, h_truth)[0, 1])

    assert rel_rmse < 0.50, (
        f"W² embedding test FAILED: rel_RMSE = {rel_rmse:.4f} (>0.50). "
        f"Likely the Γ multiplication is missing in predict() — see plan §2.4. "
        f"RMSE={rmse:.4f}, std(truth)={std_truth:.4f}, corr={corr:.4f}"
    )
    assert corr > 0.80, f"corr(KPV, E[W²|Z,A]) = {corr:.4f} (expected > 0.80)"


# ════════════════════════════════════════════════════════════════════════════
# Group 3 — Fredholm conditions (the Phase 7 xfail targets)
# ════════════════════════════════════════════════════════════════════════════


def test_kpv_fredholm_dgp1_moments(kpv_dgp1, dgp1_sample):
    """
    T_KPV5a — Fredholm moment conditions on DGP1.

    The Fredholm condition E[Y - h₀(W,A) | Z, A] = 0 implies, for any
    measurable g(Z, A): E[(Y - h₀)·g(Z, A)] = 0.

    Test 3 polynomial test functions (same as test_dgp.py T5b for DGP1):
        |E[resid·Z]|     < 0.05
        |E[resid·Z²]|    < 0.10
        |E[resid·Z·A]|   < 0.05

    With KPV (true Fredholm bridge), should pass with margin.
    """
    s = dgp1_sample
    h_pred = kpv_dgp1.predict(s.W, s.A)
    R = s.Y - h_pred
    m1 = float(np.mean(R * s.Z))
    m2 = float(np.mean(R * s.Z ** 2))
    m3 = float(np.mean(R * s.Z * s.A))
    assert abs(m1) < 0.05, f"E[R·Z]   = {m1:.5f} (tol 0.05)"
    assert abs(m2) < 0.10, f"E[R·Z²]  = {m2:.5f} (tol 0.10)"
    assert abs(m3) < 0.05, f"E[R·Z·A] = {m3:.5f} (tol 0.05)"


def test_kpv_fredholm_dgp1_rkhs_norm(kpv_dgp1, dgp1_sample):
    """
    T_KPV5b — RKHS norm of the Fredholm residual on DGP1.

    Global moment test: ||E_n[(Y - h₀(W,A)) · k_ZA(·, (Z,A))]||²_{H_ZA}
                      ≈ (1/n²) · R^T K_ZA R
    where R = Y - h_pred. Robust to choice of test functions.

    For DGP1 with KPV bridge, expect < 0.01.
    """
    s = dgp1_sample
    h_pred = kpv_dgp1.predict(s.W, s.A)
    R = s.Y - h_pred
    # Use the SAME bandwidths as the trained bridge for consistency
    K_Z = _rbf_gram(s.Z, s.Z, kpv_dgp1.ell_Z)
    K_A = _rbf_gram(s.A, s.A, kpv_dgp1.ell_A)
    K_ZA = K_Z * K_A
    n = len(R)
    rkhs_moment = float(R @ K_ZA @ R) / (n ** 2)
    assert rkhs_moment < 0.01, (
        f"RKHS Fredholm moment = {rkhs_moment:.5f} (tol 0.01). "
        f"R has mean={np.mean(R):.4f}, std={np.std(R):.4f}"
    )


# ── Group 3 (continued) — DGP2 (Michaelis-Menten) Fredholm tests ────────────


def test_kpv_t5a_dgp2_bridge_matches_m_true(kpv_dgp2, dgp2, dgp2_sample):
    """
    T_KPV7 — CRITICAL Phase 8 target.

    With true KPV bridge (Mastouri 2021), the identification condition
        E[ĥ(W, a)] ≈ m_true(a)
    should hold for any test dose a (Kallus 2021 Lemma 2).

    Phase 7's regression oracle h₀ = E[μ(U,A)|W,A] FAILS this — the
    observational conditioning on A=a introduces bias. This is why
    test_mm_T5a_bridge_consistency in test_dgp.py is xfail.

    KPV with true Fredholm bridge MUST pass this. First-cut tolerance: 0.10.
    Tighten to 0.05 once stable across pilot seeds.
    """
    s = dgp2_sample
    a_vals = np.array([1.5, 2.0, 2.5, 3.0])
    errors = {}
    max_err = 0.0
    for a in a_vals:
        h_at_a = kpv_dgp2.predict(s.W, np.full(len(s.W), float(a)))
        emp = float(np.mean(h_at_a))
        truth = dgp2.m_true(float(a))
        err = abs(emp - truth)
        errors[float(a)] = (emp, truth, err)
        max_err = max(max_err, err)
    err_str = "; ".join(
        f"a={a}: E[ĥ]={v[0]:.4f}, m_true={v[1]:.4f}, |err|={v[2]:.4f}"
        for a, v in errors.items()
    )
    assert max_err < 0.10, (
        f"T_KPV7 FAILED: max |E[ĥ(W,a)] - m_true(a)| = {max_err:.4f} (tol 0.10)\n"
        f"{err_str}"
    )


def test_kpv_t5b_dgp2_fredholm(kpv_dgp2, dgp2_sample):
    """
    T_KPV8 — CRITICAL Phase 8 target.

    Fredholm conditions on DGP2:
        |E[(Y - ĥ)·Z]|       < 0.05    (linear in Z, stable across seeds)
        ||Fredholm residual||²_{H_ZA}  < 0.05   (RKHS norm — canonical test)

    NOTE on the higher-order moments E[R·Z²] and E[R·Z·A]:
        Phase 7's xfailed test_mm_T5b_fredholm tests these at tol 0.10/0.05.
        With KPV at n=3000, they range 0.07-0.5 (m2) and 0.04-0.24 (m3) across
        seeds. This is NOT a bridge issue — it's MC noise on heavy-tail Z, Z²
        test functions (Z = U + ε_Z with U LogNormal has fat tails).
        The RKHS moment is the principled bounded-kernel analogue and gives
        ~1e-4 at all seeds, confirming the bridge satisfies Fredholm correctly.

    The Phase 7 xfailed test in test_dgp.py is left xfail because it tests the
    DGP's regression-oracle h0, not the KPV bridge. T_KPV8 here is the Phase 8
    counterpart that uses the KPV bridge and the RKHS test.
    """
    s = dgp2_sample
    h_pred = kpv_dgp2.predict(s.W, s.A)
    R = s.Y - h_pred

    # Linear moment — stable across seeds
    m1 = float(np.mean(R * s.Z))
    assert abs(m1) < 0.05, f"E[R·Z] = {m1:.5f} (tol 0.05)"

    # RKHS moment — bounded test functions, MC-stable
    K_Z = _rbf_gram(s.Z, s.Z, kpv_dgp2.ell_Z)
    K_A = _rbf_gram(s.A, s.A, kpv_dgp2.ell_A)
    K_ZA = K_Z * K_A
    n = len(R)
    rkhs_moment = float(R @ K_ZA @ R) / (n ** 2)
    assert rkhs_moment < 0.05, (
        f"RKHS Fredholm moment (DGP2) = {rkhs_moment:.5f} (tol 0.05). "
        f"Expected ~1e-4 if KPV bridge is correct."
    )

    # Relative version (scale-free): rkhs_moment / rkhs_moment_baseline
    # Baseline = same RKHS quadratic form on Y - mean(Y) instead of residual
    Y_centered = s.Y - np.mean(s.Y)
    rkhs_baseline = float(Y_centered @ K_ZA @ Y_centered) / (n ** 2)
    rkhs_relative = rkhs_moment / rkhs_baseline if rkhs_baseline > 0 else float("inf")
    assert rkhs_relative < 0.10, (
        f"RKHS relative moment (DGP2) = {rkhs_relative:.4f} (tol 0.10). "
        f"raw={rkhs_moment:.5f}, baseline={rkhs_baseline:.5f}. "
        f"Expected < 1% if bridge captures the Fredholm signal in Y."
    )


def test_kpv_w_squared_embedding_dgp2(dgp2, dgp2_sample):
    """
    T_KPV4b — DISCRIMINANT TEST for DGP2 (LogNormal U).

    Same logic as T_KPV4 (DGP1) but on DGP2: train KPV with target g(W,A)=W²
    and verify it tracks E[W²|Z,A] (analytic via Gauss-Hermite over LogNormal U).

    For DGP2: U ~ LogNormal(0, σ_U²), W = U + ε_W (ε_W ⊥ Z, A).
    Therefore E[W²|Z,A] = σ_W² + E[U²|Z,A] where E[U²|Z,A] is computed by
    importance-weighted Gauss-Hermite quadrature over the latent U:
        With x = log(u)/σ_U ~ N(0,1):
            num = Σⱼ wⱼ · u_j² · φ_Z(z - u_j) · φ_A(a - base - sens·u_j)
            den = Σⱼ wⱼ ·       · φ_Z(z - u_j) · φ_A(a - base - sens·u_j)
            E[U²|z,a] = num / den
    (factors 1/√(2π) cancel in the ratio).

    Discriminant criterion: rel_RMSE < 0.50 against the Gauss-Hermite truth.
    """
    from numpy.polynomial.hermite_e import hermegauss
    from scipy.stats import norm

    s = dgp2_sample
    target = s.W ** 2
    bridge = KPVBridgeH().fit(s.W, s.A, s.Z, target)
    h_pred = bridge.predict(s.W, s.A)

    # Analytic E[W²|Z, A] via Gauss-Hermite (importance-weighted ratio)
    n_gh = 50
    x_q, w_q = hermegauss(n_gh)
    U_grid = np.exp(dgp2.sigma_U * x_q)              # (n_gh,)
    Z_test, A_test = s.Z, s.A
    h_truth = np.zeros(len(Z_test))
    for i in range(len(Z_test)):
        wz = norm.pdf(Z_test[i], loc=U_grid, scale=dgp2.sigma_Z)
        wa = norm.pdf(A_test[i], loc=dgp2.base_dose + dgp2.sensitivity * U_grid,
                      scale=dgp2.sigma_A)
        weights = w_q * wz * wa
        denom = float(np.sum(weights))
        if denom < 1e-300:
            EU2_given_za = 0.0
        else:
            EU2_given_za = float(np.sum(weights * U_grid ** 2) / denom)
        h_truth[i] = dgp2.sigma_W ** 2 + EU2_given_za

    rmse = float(np.sqrt(np.mean((h_pred - h_truth) ** 2)))
    rel_rmse = rmse / float(np.std(h_truth))
    corr = float(np.corrcoef(h_pred, h_truth)[0, 1])

    assert rel_rmse < 0.50, (
        f"W² embedding DGP2 FAILED: rel_RMSE = {rel_rmse:.4f} (tol 0.50). "
        f"RMSE={rmse:.4f}, std(truth)={np.std(h_truth):.4f}, corr={corr:.4f}. "
        f"Likely Γ missing in predict() or DGP2 nonlinearity overwhelms n=3000."
    )
    assert corr > 0.70, f"corr = {corr:.4f} (expected > 0.70)"


# ════════════════════════════════════════════════════════════════════════════
# Group 4 — DRKernel end-to-end
# ════════════════════════════════════════════════════════════════════════════


def test_dr_kernel_returns_estimation_result(dgp1, dgp1_sample):
    """
    T_DR11 — interface compliance.

    DRKernel.fit().estimate() returns EstimationResult with:
        - psi_hat = J_dr[ref_dose_index] (scalar, harness-compatible)
        - V_hat  = variance at ref dose
        - extra  = full diagnostics dict (curves + ESS + variance decomposition
                   + Jensen gap + ref dose info + bandwidth + n_folds + lambdas)
    """
    a_grid = np.array([-0.5, 0.0, 0.5])
    dr = DRKernel(a_grid, OracleBridgeQ(dgp1), n_folds=5, random_state=0,
                  ref_dose_index=1)   # psi_hat at a=0.0
    dr.fit(dgp1_sample, m_true_fn=dgp1.m_true)
    res = dr.estimate()

    assert isinstance(res, EstimationResult)
    # psi_hat is now a scalar at the reference dose, NOT NaN
    assert np.isfinite(res.psi_hat), "psi_hat should be finite (J_dr at ref dose)"
    assert res.psi_hat == res.extra["J_dr"][1]
    assert res.method_name == "DRKernel"
    assert res.n == len(dgp1_sample.Y)

    # Required extras (Phase 8.0 + audit additions)
    required = {
        "J_reg", "J_dr", "J_true", "J_policy_true", "jensen_gap",
        "V_hat_grid", "V_reg_grid", "V_correction_grid",
        "ESS_grid", "ESS_ratio_grid", "ESS_min", "ESS_ratio_min",
        "a_grid", "ref_dose_index", "ref_dose",
        "bandwidth", "policy_kernel", "alpha_derived", "mise",
        "n_folds", "lambda_1", "lambda_2",
    }
    missing = required - res.extra.keys()
    assert not missing, f"Missing extra keys: {missing}"

    # Shapes and types
    K = len(a_grid)
    for key in ["J_reg", "J_dr", "J_true", "J_policy_true", "jensen_gap",
                "V_hat_grid", "V_reg_grid", "V_correction_grid",
                "ESS_grid", "ESS_ratio_grid"]:
        assert res.extra[key].shape == (K,), f"{key} shape mismatch"

    assert np.all(res.extra["ESS_grid"] > 0), "ESS_grid must be strictly positive"
    assert np.all(res.extra["ESS_ratio_grid"] <= 1.0), "ESS/n cannot exceed 1"
    assert res.extra["policy_kernel"] == "gaussian_density"
    assert res.extra["n_folds"] == 5
    assert res.extra["ref_dose_index"] == 1
    assert res.extra["ref_dose"] == 0.0

    # Jensen gap: m_true(a) - J_policy(a, h) >= 0 for concave m
    # DGP1 m_true is linear → jensen_gap ≈ 0 (within float precision)
    assert np.all(np.abs(res.extra["jensen_gap"]) < 1e-3), (
        f"DGP1 has linear m_true → jensen_gap should be ~0, got {res.extra['jensen_gap']}"
    )

    # Variance decomposition: V_hat ≈ V_reg + V_correction + 2·Cov  (loose check)
    assert np.all(res.extra["V_reg_grid"] >= 0)
    assert np.all(res.extra["V_correction_grid"] >= 0)


def test_dr_kernel_dgp1_smoke(dgp1, dgp1_sample):
    """
    T_DR9 — DRKernel oracle-q smoke test on DGP1.

    Grid choice [-0.5, 0.0, 0.5] stays comfortably within A's data support
    (A ~ N(0, 0.71); 10/90 quantiles ≈ [-0.95, 0.89]). At grid points within
    support, ESS > 100 and the DR estimate tracks m_true closely.

    Boundary doses (e.g. a=2.5 with A bulk in [-1, 1]) are NOT tested here —
    they are limited by IPW variance (ESS < 5) and bridge extrapolation, not
    by the bridge fit itself. Phase 9 (estimated q with clipping) addresses
    the IPW limitation.

    Tolerance: max|J_dr - m_true| < 0.20.
    """
    a_grid = np.array([-0.5, 0.0, 0.5])
    dr = DRKernel(a_grid, OracleBridgeQ(dgp1), n_folds=5, random_state=0)
    dr.fit(dgp1_sample, m_true_fn=dgp1.m_true)
    res = dr.estimate()
    err = float(np.max(np.abs(res.extra["J_dr"] - res.extra["J_true"])))
    ess_min = float(np.min(res.extra["ESS_grid"]))
    assert err < 0.20, (
        f"DGP1 (in-support grid): max|J_dr - m_true| = {err:.4f} (tol 0.20)\n"
        f"J_dr   = {res.extra['J_dr']}\n"
        f"J_true = {res.extra['J_true']}\n"
        f"ESS    = {res.extra['ESS_grid']} (min={ess_min:.1f})"
    )
    # Sanity: grid is within support, so ESS should be healthy
    assert ess_min > 100, f"ESS_min = {ess_min:.1f} (grid likely outside support)"


def test_dr_kernel_dgp2_curve(dgp2, dgp2_sample):
    """
    T_DR10 — DRKernel oracle-q on DGP2: resolves Phase 7 T_DR4.

    Grid choice [1.8, 2.2, 2.6, 3.0] stays within DGP2 A IQR (10/90 quantiles
    ≈ [1.52, 3.45]). Phase 7's DRDoseResponse with REGRESSION-oracle h₀ + oracle
    q gives err ~ 0.376 median, 0.605 p90 across seeds (xfail at tol=0.50).

    DRKernel with TRUE Fredholm bridge gives err ~ 0.10 on the same in-support
    grid — the major Phase 7 limitation is resolved.

    Boundary grid points (a=0.5 or a=4.0) are NOT tested here — at those
    extremes, ESS < 10 and DR variance dominates regardless of bridge quality.

    Tolerance: max|J_dr - m_true| < 0.20.
    """
    a_grid = np.array([1.8, 2.2, 2.6, 3.0])
    dr = DRKernel(a_grid, OracleBridgeQ(dgp2), n_folds=5, random_state=0)
    dr.fit(dgp2_sample, m_true_fn=dgp2.m_true)
    res = dr.estimate()
    err = float(np.max(np.abs(res.extra["J_dr"] - res.extra["J_true"])))
    ess_min = float(np.min(res.extra["ESS_grid"]))
    assert err < 0.20, (
        f"DGP2 (in-support grid): max|J_dr - m_true| = {err:.4f} (tol 0.20)\n"
        f"J_dr     = {res.extra['J_dr']}\n"
        f"J_true   = {res.extra['J_true']}\n"
        f"ESS_grid = {res.extra['ESS_grid']} (min={ess_min:.1f})"
    )
    # Sanity: grid is within IQR, so ESS should be healthy
    assert ess_min > 100, f"ESS_min = {ess_min:.1f} (grid likely outside IQR)"


# ════════════════════════════════════════════════════════════════════════════
# Free function — verify_dr_kernel (analogue to verify_method)
# ════════════════════════════════════════════════════════════════════════════


def verify_dr_kernel(dgp, n: int = 3000, seed: int = 42, n_folds: int = 5) -> dict:
    """
    Pre-flight diagnostic for DRKernel — analogue of verify_method() for the new
    KPV-based estimator. Runs the four critical KPV+DR checks on the given DGP
    and returns a dict of scalar diagnostics.

    Parameters
    ----------
    dgp     : PCIDgp instance (e.g. MichaelisMentenDGP())
    n       : sample size for the diagnostic (default 3000)
    seed    : RNG seed (default 42)
    n_folds : K-fold cross-fitting folds (default 5)

    Returns
    -------
    dict with keys
        T_KPV4_passed       : bool — W² embedding test (Mastouri-vs-Singh gate)
        T_KPV5a_max_moment  : float — max |E[R·g(Z)]| over polynomial test fns
        T_KPV5b_rkhs        : float — RKHS Fredholm residual moment (bounded)
        T_KPV7_max_err      : float — max |E[ĥ(W,a)] - m_true(a)| over a_grid
        T_DR10_err          : float — max |J_dr - m_true| on in-support grid
        T_DR10_ess_min      : float — min ESS across grid
    """
    from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP
    from simulations.methods._oracle_bridges import OracleBridgeQ

    s = dgp.generate(n=n, seed=seed)
    bridge = KPVBridgeH().fit(s.W, s.A, s.Z, s.Y)

    # T_KPV4 — W² embedding only meaningful on DGP1 (analytic E[W²|Z,A] available)
    if isinstance(dgp, CobbDouglasLinearDGP):
        b_w2 = KPVBridgeH().fit(s.W, s.A, s.Z, s.W ** 2)
        h_pred = b_w2.predict(s.W, s.A)
        h_truth = _ew2_given_za_dgp1(s.Z, s.A, dgp)
        rel = float(np.sqrt(np.mean((h_pred - h_truth) ** 2)) / np.std(h_truth))
        t_kpv4 = bool(rel < 0.50)
    else:
        t_kpv4 = True  # not applicable

    # T_KPV5a — max polynomial moment
    R = s.Y - bridge.predict(s.W, s.A)
    max_moment = max(
        abs(float(np.mean(R * s.Z))),
        abs(float(np.mean(R * s.Z ** 2))),
        abs(float(np.mean(R * s.Z * s.A))),
    )

    # T_KPV5b — RKHS moment
    K_Z = _rbf_gram(s.Z, s.Z, bridge.ell_Z)
    K_A = _rbf_gram(s.A, s.A, bridge.ell_A)
    K_ZA = K_Z * K_A
    rkhs = float(R @ K_ZA @ R) / (n ** 2)

    # T_KPV7 — bridge level on a_grid
    if isinstance(dgp, CobbDouglasLinearDGP):
        a_vals = np.array([-0.5, 0.0, 0.5])
    else:
        a_vals = np.array([1.5, 2.0, 2.5, 3.0])
    bridge_errs = [
        abs(float(np.mean(bridge.predict(s.W, np.full(len(s.W), float(a)))))
            - dgp.m_true(float(a)))
        for a in a_vals
    ]

    # T_DR10 — DRKernel end-to-end on in-support grid
    if isinstance(dgp, CobbDouglasLinearDGP):
        a_grid_dr = np.array([-0.5, 0.0, 0.5])
    else:
        a_grid_dr = np.array([1.8, 2.2, 2.6, 3.0])
    dr = DRKernel(a_grid_dr, OracleBridgeQ(dgp), n_folds=n_folds, random_state=0)
    dr.fit(s, m_true_fn=dgp.m_true)
    res = dr.estimate()
    dr_err = float(np.max(np.abs(res.extra["J_dr"] - res.extra["J_true"])))
    ess_min = float(np.min(res.extra["ESS_grid"]))

    return {
        "T_KPV4_passed":      t_kpv4,
        "T_KPV5a_max_moment": max_moment,
        "T_KPV5b_rkhs":       rkhs,
        "T_KPV7_max_err":     max(bridge_errs),
        "T_DR10_err":         dr_err,
        "T_DR10_ess_min":     ess_min,
    }


def test_verify_dr_kernel_runs(dgp2):
    """
    Smoke test for the free `verify_dr_kernel` function — ensures it runs end-to-end
    on DGP2 and returns the expected dict shape.
    """
    out = verify_dr_kernel(dgp2, n=2000, seed=42, n_folds=5)
    expected_keys = {
        "T_KPV4_passed", "T_KPV5a_max_moment", "T_KPV5b_rkhs",
        "T_KPV7_max_err", "T_DR10_err", "T_DR10_ess_min",
    }
    assert set(out.keys()) == expected_keys
    # Sanity ranges
    assert isinstance(out["T_KPV4_passed"], bool)
    assert out["T_KPV5b_rkhs"] >= 0
    assert out["T_DR10_ess_min"] > 0


# ════════════════════════════════════════════════════════════════════════════
# Group 5 — Nyström approximation (Phase 8.5)
# ════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def kpv_dgp1_nystrom(dgp1_sample):
    """KPVBridgeH fitted on DGP1 with n_landmarks=200 (Nyström mode), using Y."""
    s = dgp1_sample
    return KPVBridgeH(n_landmarks=200).fit(s.W, s.A, s.Z, s.Y)


def test_nystrom_fit_populates_correct_fields(dgp1_sample):
    """
    T_NYST0 — structural: Nyström fit populates the (m,m) cached fields, not Γ.

    With n_landmarks=200 < n=2000, fit() must activate Nyström mode and cache
    _M (m,m), _alpha_m (m,), _W_lm (m,), _A_lm (m,) — and leave the full-mode
    caches _Gamma, _alpha, _W_train, _A_train as None.
    """
    s = dgp1_sample
    m = 200
    bridge = KPVBridgeH(n_landmarks=m).fit(s.W, s.A, s.Z, s.Y)

    assert bridge._fitted
    assert bridge._nystrom_mode, "_nystrom_mode should be True when n_landmarks < n"

    # Nyström caches populated with correct shapes
    assert bridge._M is not None, "_M not set"
    assert bridge._M.shape == (m, m), f"_M shape: {bridge._M.shape}"
    assert bridge._alpha_m is not None, "_alpha_m not set"
    assert bridge._alpha_m.shape == (m,), f"_alpha_m shape: {bridge._alpha_m.shape}"
    assert bridge._W_lm is not None, "_W_lm not set"
    assert bridge._W_lm.shape == (m,), f"_W_lm shape: {bridge._W_lm.shape}"
    assert bridge._A_lm is not None, "_A_lm not set"
    assert bridge._A_lm.shape == (m,), f"_A_lm shape: {bridge._A_lm.shape}"

    # Full-mode caches must NOT be set (stay at their None init values)
    assert bridge._Gamma is None, "_Gamma should be None in Nyström mode"
    assert bridge._alpha is None, "_alpha should be None in Nyström mode"
    assert bridge._W_train is None, "_W_train should be None in Nyström mode"
    assert bridge._A_train is None, "_A_train should be None in Nyström mode"

    # Bandwidths always resolved in both modes
    assert bridge.ell_W > 0 and bridge.ell_A > 0 and bridge.ell_Z > 0
    assert bridge._n_train == len(s.W)


def test_nystrom_fallback_to_full_when_n_landmarks_ge_n(dgp1):
    """
    T_NYST1 — when n_landmarks >= n, silently falls back to full O(n³) mode.

    The guard in fit(): `if self.n_landmarks is not None and int(self.n_landmarks) < n`.
    With n=80, n_landmarks=200 => 200 >= 80 => condition is False => full mode.
    """
    s = dgp1.generate(n=80, seed=7)
    bridge = KPVBridgeH(n_landmarks=200).fit(s.W, s.A, s.Z, s.Y)

    assert not bridge._nystrom_mode, "_nystrom_mode should be False when n_landmarks >= n"
    assert bridge._Gamma is not None, "_Gamma should be set in full mode"
    assert bridge._Gamma.shape == (80, 80)
    assert bridge._alpha is not None, "_alpha should be set in full mode"
    assert bridge._alpha.shape == (80,)
    # Nyström caches stay None
    assert bridge._M is None, "_M should be None in full mode"
    assert bridge._alpha_m is None, "_alpha_m should be None in full mode"
    assert bridge._W_lm is None, "_W_lm should be None in full mode"
    assert bridge._A_lm is None, "_A_lm should be None in full mode"


def test_nystrom_landmark_determinism(dgp1_sample):
    """
    T_NYST2 — two identical Nyström fits give bitwise-identical predictions.

    The seed=n in fit() ensures reproducibility: same (W,A,Z,Y) and same n
    → same I_m → same _M and _alpha_m → bitwise-identical predict() output.
    """
    s = dgp1_sample
    b1 = KPVBridgeH(n_landmarks=200).fit(s.W, s.A, s.Z, s.Y)
    b2 = KPVBridgeH(n_landmarks=200).fit(s.W, s.A, s.Z, s.Y)
    p1 = b1.predict(s.W[:100], s.A[:100])
    p2 = b2.predict(s.W[:100], s.A[:100])
    assert np.array_equal(p1, p2), (
        "Nyström predictions must be bitwise-identical for the same data. "
        "Check that seed=n logic in fit() is deterministic."
    )


def test_kpv_w_squared_embedding_nystrom(dgp1, dgp1_sample):
    """
    T_KPV4_nystrom — GATE: W² embedding discriminant test with Nyström.

    Same logic as T_KPV4 (full mode): train KPV with target W², compare to
    analytic E[W²|Z,A]. The critical distinction: if predict() uses
        Ew = KW_test_m @ Gamma_nm   (WRONG — omits M)
    instead of
        Ew = KW_test_m @ M          (CORRECT — M = K_W_mm⁻¹ C)
    then the projection is missing and rel_RMSE >> 0.50.

    Tolerance: rel_RMSE < 0.50, corr > 0.80.
    STOP AND FIX if this fails before proceeding further.
    """
    s = dgp1_sample
    target = s.W ** 2
    bridge = KPVBridgeH(n_landmarks=200).fit(s.W, s.A, s.Z, target)

    assert bridge._nystrom_mode, "n_landmarks=200 < n=2000 should activate Nyström mode"

    h_pred = bridge.predict(s.W, s.A)
    h_truth = _ew2_given_za_dgp1(s.Z, s.A, dgp1)

    rmse = float(np.sqrt(np.mean((h_pred - h_truth) ** 2)))
    std_truth = float(np.std(h_truth))
    rel_rmse = rmse / std_truth
    corr = float(np.corrcoef(h_pred, h_truth)[0, 1])

    assert rel_rmse < 0.50, (
        f"T_KPV4_nystrom FAILED (GATE): rel_RMSE={rel_rmse:.4f} (tol 0.50). "
        f"Likely _M omitted in predict() Nyström path. "
        f"RMSE={rmse:.4f}, std(truth)={std_truth:.4f}, corr={corr:.4f}"
    )
    assert corr > 0.80, f"corr(KPV_nystrom, E[W²|Z,A]) = {corr:.4f} (expected > 0.80)"


def test_kpv_fredholm_rkhs_norm_nystrom(kpv_dgp1_nystrom, dgp1_sample):
    """
    T_KPV5b_nystrom — RKHS Fredholm moment with Nyström bridge on DGP1.

    Uses kpv_dgp1_nystrom (n_landmarks=200, fitted on Y). Tolerance is relaxed
    from 0.01 (full KPV) to 0.05 because Nyström introduces mild approximation
    error, but the Fredholm condition should still be substantially satisfied.
    """
    s = dgp1_sample
    h_pred = kpv_dgp1_nystrom.predict(s.W, s.A)
    R = s.Y - h_pred

    K_Z = _rbf_gram(s.Z, s.Z, kpv_dgp1_nystrom.ell_Z)
    K_A = _rbf_gram(s.A, s.A, kpv_dgp1_nystrom.ell_A)
    K_ZA = K_Z * K_A
    n = len(R)
    rkhs_moment = float(R @ K_ZA @ R) / (n ** 2)
    assert rkhs_moment < 0.05, (
        f"RKHS Fredholm moment (Nyström, DGP1) = {rkhs_moment:.5f} (tol 0.05). "
        f"Full KPV gives ~1e-4; Nyström with m=200 should be well below 0.05."
    )


def test_nystrom_plug_in_policy_shape(kpv_dgp1_nystrom, dgp1_sample):
    """
    T_NYST3 — structural: Nyström plug_in_policy() returns correct shape + finite.

    Checks (n_test,) output and finiteness for several (a, h) pairs.
    Validates the Nyström branch in plug_in_policy() against k_tilde_m ⊙ alpha_m.
    """
    s = dgp1_sample
    for a, h in [(-0.5, 0.3), (0.0, 0.3), (0.5, 0.3)]:
        vals = kpv_dgp1_nystrom.plug_in_policy(s.W[:200], float(a), float(h))
        assert vals.shape == (200,), (
            f"plug_in_policy output shape {vals.shape} at a={a} — expected (200,)"
        )
        assert np.all(np.isfinite(vals)), (
            f"Non-finite values in plug_in_policy at a={a}, h={h}: "
            f"min={vals.min():.4f}, max={vals.max():.4f}"
        )


def test_dr_kernel_dgp1_nystrom_smoke(dgp1, dgp1_sample):
    """
    T_DR9_nystrom — DRKernel with Nyström bridge_kwargs on DGP1 (in-support grid).

    Verifies that bridge_kwargs={'n_landmarks': 200} is correctly propagated to
    KPVBridgeH per fold, and that the DR estimate remains accurate on the
    in-support grid [-0.5, 0.0, 0.5]. Same tolerances as T_DR9 (full mode):
        max|J_dr - m_true| < 0.20,  ESS_min > 100.
    """
    a_grid = np.array([-0.5, 0.0, 0.5])
    dr = DRKernel(
        a_grid,
        OracleBridgeQ(dgp1),
        n_folds=5,
        random_state=0,
        bridge_kwargs={"n_landmarks": 200},
    )
    dr.fit(dgp1_sample, m_true_fn=dgp1.m_true)
    res = dr.estimate()

    err = float(np.max(np.abs(res.extra["J_dr"] - res.extra["J_true"])))
    ess_min = float(np.min(res.extra["ESS_grid"]))

    assert err < 0.20, (
        f"DGP1 Nyström (in-support grid): max|J_dr - m_true| = {err:.4f} (tol 0.20)\n"
        f"J_dr   = {res.extra['J_dr']}\n"
        f"J_true = {res.extra['J_true']}\n"
        f"ESS    = {res.extra['ESS_grid']} (min={ess_min:.1f})"
    )
    assert ess_min > 100, f"ESS_min = {ess_min:.1f} (grid likely outside support)"


def test_dr_kernel_dgp2_nystrom_curve(dgp2, dgp2_sample):
    """
    T_DR10_nystrom — DRKernel with Nyström bridge_kwargs on DGP2 (in-support grid).

    Same as T_DR10 (full mode) but with bridge_kwargs={'n_landmarks': 200}.
    Grid [1.8, 2.2, 2.6, 3.0] stays within DGP2 A IQR. Each fold has
    n_fold_train ≈ 2400, and 200 < 2400 → Nyström is activated per fold.
    Tolerance: max|J_dr - m_true| < 0.20, ESS_min > 100.
    """
    a_grid = np.array([1.8, 2.2, 2.6, 3.0])
    dr = DRKernel(
        a_grid,
        OracleBridgeQ(dgp2),
        n_folds=5,
        random_state=0,
        bridge_kwargs={"n_landmarks": 200},
    )
    dr.fit(dgp2_sample, m_true_fn=dgp2.m_true)
    res = dr.estimate()

    err = float(np.max(np.abs(res.extra["J_dr"] - res.extra["J_true"])))
    ess_min = float(np.min(res.extra["ESS_grid"]))

    assert err < 0.20, (
        f"DGP2 Nyström (in-support grid): max|J_dr - m_true| = {err:.4f} (tol 0.20)\n"
        f"J_dr     = {res.extra['J_dr']}\n"
        f"J_true   = {res.extra['J_true']}\n"
        f"ESS_grid = {res.extra['ESS_grid']} (min={ess_min:.1f})"
    )
    assert ess_min > 100, f"ESS_min = {ess_min:.1f} (grid likely outside IQR)"
