"""
Tests for simulations/methods/bennett_riesz.py (Phase 12A).

Tests
-----
T_BEN1 -- PolynomialPairFeatureMap shapes: degree=2 -> 6 features, degree=3 -> 10.
T_BEN2 -- gaussian_raw_moment analytic vs Gauss-Hermite quadrature (orders 0-4).
T_BEN3 -- PolicyPolynomialIntegrator.integrate() shape and no NaN.
T_BEN4 -- BennettPolicyRiesz.fit() never accepts Y: TypeError raised.
T_BEN5 -- BennettPolicyRiesz: fundamental moment condition mean(r_hat) approx 1.
T_BEN6 -- BennettPolicyRiesz.riesz_residual() decreases as lambda_r decreases.
T_BEN7 -- BennettFunctionalDR smoke: n=200, no NaN, extra keys present, V_hat > 0.
T_BEN8 -- BennettFunctionalDR backward compat: extra["J_dr"] == J_bennett_grid.
T_BEN9 -- BennettFunctionalDR REG-only mode (lambda_r=1e10): correction approx 0.
"""
from __future__ import annotations

import numpy as np
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_simple_data(n: int = 300, seed: int = 42):
    """Generate simple iid N(0,1) data for unit tests."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal(n)
    A = rng.standard_normal(n)
    Z = rng.standard_normal(n)
    return W, A, Z


def _dgp2_sample(n: int = 200, seed: int = 42):
    """Return a small DGP2 sample for BennettFunctionalDR smoke tests."""
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)
    return dgp, dgp.generate(n=n, seed=seed)


A_GRID_TEST = np.array([1.8, 2.2, 2.6, 3.0])


# ══════════════════════════════════════════════════════════════════════════════
#  T_BEN1 -- PolynomialPairFeatureMap shapes
# ══════════════════════════════════════════════════════════════════════════════

def test_T_BEN1_feature_map_shapes():
    """
    T_BEN1: degree=2 -> 6 features, degree=3 -> 10 features.
    Transform output shape is (n, p).
    """
    from simulations.methods.bennett_riesz import PolynomialPairFeatureMap

    rng = np.random.default_rng(0)
    x1 = rng.standard_normal(50)
    x2 = rng.standard_normal(50)

    # degree=2
    fm2 = PolynomialPairFeatureMap(degree=2).fit(x1, x2)
    assert fm2.n_features() == 6, f"Expected 6 features for degree=2, got {fm2.n_features()}"
    out2 = fm2.transform(x1, x2)
    assert out2.shape == (50, 6), f"Expected shape (50, 6), got {out2.shape}"
    assert np.all(np.isfinite(out2)), "degree=2 transform has non-finite values"

    # degree=3
    fm3 = PolynomialPairFeatureMap(degree=3).fit(x1, x2)
    assert fm3.n_features() == 10, f"Expected 10 features for degree=3, got {fm3.n_features()}"
    out3 = fm3.transform(x1, x2)
    assert out3.shape == (50, 10), f"Expected shape (50, 10), got {out3.shape}"
    assert np.all(np.isfinite(out3)), "degree=3 transform has non-finite values"

    # Verify (0,0) monomial = 1 for all rows
    assert np.allclose(out2[:, 0], 1.0), "First column (0,0 monomial) should be all 1s"
    assert np.allclose(out3[:, 0], 1.0), "First column (0,0 monomial) should be all 1s"

    # degree=1 should raise ValueError
    with pytest.raises(ValueError, match="degree must be 2 or 3"):
        PolynomialPairFeatureMap(degree=1)


# ══════════════════════════════════════════════════════════════════════════════
#  T_BEN2 -- gaussian_raw_moment analytic vs Gauss-Hermite
# ══════════════════════════════════════════════════════════════════════════════

def test_T_BEN2_gaussian_raw_moment_vs_quadrature():
    """
    T_BEN2: Verify gaussian_raw_moment(k, mu, sigma) matches Gauss-Hermite
    quadrature for orders 0-4. Max absolute error < 1e-6.
    """
    from simulations.methods.bennett_riesz import gaussian_raw_moment
    from numpy.polynomial.hermite_e import hermegauss

    # Gauss-Hermite quadrature for E[X^k] where X ~ N(mu, sigma^2)
    # hermegauss uses HermiteE (probabilist) with weight exp(-x^2/2)
    # Conversion: E[X^k] = (1/sqrt(2pi)) int (mu + sigma*x)^k exp(-x^2/2) dx
    #           = sum_i w_i * (mu + sigma * x_i)^k  [with hermegauss weights]
    nodes, weights = hermegauss(25)   # probabilist Hermite

    test_cases = [
        (2.0, 0.5),   # mu=2, sigma=0.5
        (0.0, 1.0),   # standard normal
        (-1.5, 2.0),  # negative mu, large sigma
        (3.0, 0.1),   # large mu, small sigma
    ]

    for mu, sigma in test_cases:
        # hermegauss: integral ∫ f(x) exp(-x^2/2) dx ≈ Σ w_i f(x_i), weights sum to √(2π).
        # For X ~ N(mu, sigma^2): E[X^k] = (1/√(2π)) ∫ (mu+sigma*t)^k exp(-t^2/2) dt
        #                               ≈ (1/√(2π)) Σ w_i (mu + sigma * x_i)^k
        # Same normalization as dgp2_bias_diagnostics._compute_J_policy_true.
        X_vals = mu + sigma * nodes
        for order in range(5):
            analytical = gaussian_raw_moment(order, mu, sigma)
            numerical = float(np.dot(weights, X_vals ** order)) / np.sqrt(2.0 * np.pi)
            err = abs(analytical - numerical)
            assert err < 1e-6, (
                f"gaussian_raw_moment({order}, mu={mu}, sigma={sigma}): "
                f"analytical={analytical:.8f}, numerical={numerical:.8f}, err={err:.2e}"
            )

    # order 0: always 1 regardless of params
    assert gaussian_raw_moment(0, 5.0, 100.0) == 1.0

    # order > 4: should raise ValueError
    with pytest.raises(ValueError, match="order must be in"):
        gaussian_raw_moment(5, 1.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  T_BEN3 -- PolicyPolynomialIntegrator shapes and no NaN
# ══════════════════════════════════════════════════════════════════════════════

def test_T_BEN3_integrator_shape_and_no_nan():
    """
    T_BEN3: integrate() returns (n, p) with no NaN, for both degree=2 and degree=3.
    """
    from simulations.methods.bennett_riesz import (
        PolynomialPairFeatureMap, PolicyPolynomialIntegrator
    )

    rng = np.random.default_rng(7)
    W_train = rng.standard_normal(100)
    A_train = rng.standard_normal(100)
    W_test  = rng.standard_normal(50)

    for degree in (2, 3):
        fm = PolynomialPairFeatureMap(degree=degree).fit(W_train, A_train)
        integrator = PolicyPolynomialIntegrator(fm)

        p = fm.n_features()
        n_test = len(W_test)

        # Typical dose and bandwidth
        out = integrator.integrate(W_test, a_target=2.0, bandwidth=0.5)
        assert out.shape == (n_test, p), (
            f"integrate() shape={out.shape}, expected ({n_test}, {p})"
        )
        assert np.all(np.isfinite(out)), f"degree={degree}: integrate() has non-finite values"

        # First column (0,0 monomial) = 1 * E[1] = 1.0 for all rows
        assert np.allclose(out[:, 0], 1.0), (
            f"degree={degree}: column 0 (constant monomial) should be all 1.0"
        )

    # Unfitted feature_map should raise RuntimeError
    fm_unfitted = PolynomialPairFeatureMap(degree=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        PolicyPolynomialIntegrator(fm_unfitted)


# ══════════════════════════════════════════════════════════════════════════════
#  T_BEN4 -- BennettPolicyRiesz: Y must NOT be accepted
# ══════════════════════════════════════════════════════════════════════════════

def test_T_BEN4_no_Y_in_fit():
    """
    T_BEN4: BennettPolicyRiesz.fit() does NOT accept Y as a keyword argument.
    Passing Y=... must raise TypeError.
    """
    from simulations.methods.bennett_riesz import BennettPolicyRiesz

    W, A, Z = _make_simple_data(n=50)
    rng = np.random.default_rng(99)
    Y = rng.standard_normal(50)   # should never be accepted

    riesz = BennettPolicyRiesz(lambda_r=1e-3, degree=2)

    # Passing Y as keyword arg must raise TypeError
    with pytest.raises(TypeError):
        riesz.fit(W, A, Z, a_grid=[0.0], bandwidth=0.5, Y=Y)

    # Passing without Y must succeed
    riesz.fit(W, A, Z, a_grid=[0.0], bandwidth=0.5)
    assert riesz._fitted, "fit() without Y should succeed"

    # predict should work after fit
    r_hat = riesz.predict(Z, A, a_target=0.0)
    assert r_hat.shape == (len(Z),), f"predict() shape {r_hat.shape}"
    assert np.all(np.isfinite(r_hat)), "predict() has non-finite values"


# ══════════════════════════════════════════════════════════════════════════════
#  T_BEN5 -- BennettPolicyRiesz: fundamental moment condition mean(r_hat) ≈ 1
# ══════════════════════════════════════════════════════════════════════════════

def test_T_BEN5_fundamental_moment_condition():
    """
    T_BEN5: For any distribution (W,A,Z) and any Gaussian policy, the Riesz
    representer must satisfy E[r_pi(Z,A)] = 1 (from the g=1 test function).

    This follows from the Riesz equation: C @ gamma ≈ b_a.
    The first row of C (constant test function) gives:
        mean(Phi, axis=0) @ gamma ≈ b_a[0] = E[T_pi 1] = 1
        mean(r_hat) = mean(Phi) @ gamma ≈ 1.

    Test with n=500, small lambda_r, normal data. Tolerance 0.15 (small sample).
    """
    from simulations.methods.bennett_riesz import BennettPolicyRiesz

    rng = np.random.default_rng(123)
    n = 500
    W = rng.standard_normal(n)
    A = rng.standard_normal(n)
    Z = rng.standard_normal(n)

    a_grid = [0.0, 0.5]
    bandwidth = 0.5

    riesz = BennettPolicyRiesz(lambda_r=1e-4, degree=2)
    riesz.fit(W, A, Z, a_grid=a_grid, bandwidth=bandwidth)

    for a in a_grid:
        r_hat = riesz.predict(Z, A, a_target=a)
        assert np.all(np.isfinite(r_hat)), f"r_hat has non-finite values at a={a}"

        mean_r = float(np.mean(r_hat))
        assert abs(mean_r - 1.0) < 0.15, (
            f"T_BEN5: mean(r_hat) = {mean_r:.4f} at a={a}, expected close to 1.0 "
            f"(fundamental Riesz moment condition E[r_pi] = 1). "
            f"Deviation = {abs(mean_r-1.0):.4f} > 0.15"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  T_BEN6 -- riesz_residual() decreases as lambda_r decreases
# ══════════════════════════════════════════════════════════════════════════════

def test_T_BEN6_residual_decreases_with_lambda():
    """
    T_BEN6: ||C @ gamma - b||^2 should decrease as lambda_r decreases from 1e-1
    to 1e-5, since smaller regularisation allows better fit to the moment condition.
    """
    from simulations.methods.bennett_riesz import BennettPolicyRiesz

    W, A, Z = _make_simple_data(n=200, seed=17)
    a_target = 0.0
    bandwidth = 0.5
    lambda_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    residuals = []
    for lam in lambda_values:
        riesz = BennettPolicyRiesz(lambda_r=lam, degree=2)
        riesz.fit(W, A, Z, a_grid=[a_target], bandwidth=bandwidth)
        res = riesz.riesz_residual(W, A, Z, a_target=a_target)
        assert np.isfinite(res), f"riesz_residual is non-finite at lambda_r={lam}"
        residuals.append(res)

    # Verify monotone non-increasing trend (allow small non-monotonicity)
    # At least 3 out of 4 consecutive pairs should decrease
    decreases = sum(1 for i in range(len(residuals)-1) if residuals[i+1] < residuals[i])
    assert decreases >= 3, (
        f"T_BEN6: riesz_residual should decrease as lambda_r decreases. "
        f"Residuals: {[f'{r:.4e}' for r in residuals]}. "
        f"Only {decreases} / {len(residuals)-1} consecutive pairs decreased."
    )

    # Overall: smallest lambda should give smaller residual than largest
    assert residuals[-1] < residuals[0], (
        f"T_BEN6: residual at lambda_r=1e-5 ({residuals[-1]:.4e}) should be "
        f"smaller than at lambda_r=1e-1 ({residuals[0]:.4e})"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T_BEN7 -- BennettFunctionalDR smoke (n=200, M=1, DGP2, no NaN)
# ══════════════════════════════════════════════════════════════════════════════

def test_T_BEN7_functional_dr_smoke():
    """
    T_BEN7: BennettFunctionalDR smoke test on DGP2 (n=200, one rep).
    Checks: no NaN/Inf, required extra keys present, V_hat > 0, psi_hat finite.
    """
    from simulations.methods.bennett_functional_dr import BennettFunctionalDR

    dgp, sample = _dgp2_sample(n=200, seed=42)

    est = BennettFunctionalDR(
        lambda_h=3e-5,
        ell_scale=3.5,
        lambda_r=1e-3,
        degree=2,
        n_folds=2,               # 2 folds for speed
        a_grid=A_GRID_TEST,
        bandwidth=None,          # Silverman default
        seed=42,
    )
    est.fit(sample)
    result = est.estimate()

    assert np.isfinite(result.psi_hat), f"psi_hat={result.psi_hat} is not finite"
    assert result.V_hat > 0, f"V_hat={result.V_hat} must be > 0"
    assert result.n == 200, f"n={result.n}"

    ex = result.extra

    # Required keys for _build_record backward compat
    required_keys = [
        "J_reg", "J_dr", "V_hat_grid", "V_reg_grid", "V_correction_grid",
        "ESS_grid", "ESS_min", "weight_p99_grid", "weight_max_grid",
        "q_clip_fraction", "q_negative_share_grid",
        "bandwidth", "jensen_gap", "mise",
        # Bennett-specific
        "riesz_residual_grid", "riesz_residual_mean",
        # Uppercase variance
        "V_total", "V_reg", "V_corr", "V_cov",
    ]
    for key in required_keys:
        assert key in ex, f"T_BEN7: missing extra key '{key}'"

    # Check shapes
    K = len(A_GRID_TEST)
    assert np.array(ex["J_reg"]).shape == (K,), f"J_reg shape wrong: {np.array(ex['J_reg']).shape}"
    assert np.array(ex["J_dr"]).shape == (K,), f"J_dr shape wrong"

    # No NaN in main arrays
    J_dr = np.array(ex["J_dr"])
    assert np.all(np.isfinite(J_dr)), f"J_dr has non-finite values: {J_dr}"

    V_total = float(ex["V_total"])
    assert np.isfinite(V_total) and V_total > 0, f"V_total={V_total} must be finite and > 0"


# ══════════════════════════════════════════════════════════════════════════════
#  T_BEN8 -- BennettFunctionalDR: backward compat key extra["J_dr"]
# ══════════════════════════════════════════════════════════════════════════════

def test_T_BEN8_j_dr_backward_compat():
    """
    T_BEN8: extra["J_dr"] must equal the Bennett DR estimate array.
    Required for _build_record in dgp2_bias_diagnostics.py compatibility.
    """
    from simulations.methods.bennett_functional_dr import BennettFunctionalDR

    dgp, sample = _dgp2_sample(n=200, seed=43)

    est = BennettFunctionalDR(
        lambda_r=1e-3, degree=2, n_folds=2, a_grid=A_GRID_TEST, seed=43
    )
    est.fit(sample)
    result = est.estimate()

    ex = result.extra
    assert "J_dr" in ex, "extra['J_dr'] key missing"

    J_dr = np.array(ex["J_dr"])
    K = len(A_GRID_TEST)
    assert J_dr.shape == (K,), f"J_dr shape {J_dr.shape} != ({K},)"

    # psi_hat should equal J_dr[ref_dose_index]
    ref_idx = est.ref_dose_index
    assert abs(result.psi_hat - float(J_dr[ref_idx])) < 1e-10, (
        f"psi_hat={result.psi_hat} != J_dr[{ref_idx}]={J_dr[ref_idx]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T_BEN9 -- REG-only mode (lambda_r=1e10): correction ≈ 0
# ══════════════════════════════════════════════════════════════════════════════

def test_T_BEN9_reg_only_mode():
    """
    T_BEN9: With lambda_r=1e10, the Riesz representer gamma is effectively 0
    (over-regularised to zero), so the correction term r_hat * (Y - h_hat) ≈ 0
    and J_dr ≈ J_reg.
    """
    from simulations.methods.bennett_functional_dr import BennettFunctionalDR

    dgp, sample = _dgp2_sample(n=200, seed=44)

    # Normal Bennett (lambda_r=1e-3)
    est_dr = BennettFunctionalDR(
        lambda_r=1e-3, degree=2, n_folds=2, a_grid=A_GRID_TEST, seed=44
    )
    est_dr.fit(sample)
    res_dr = est_dr.estimate()

    # REG-only mode (lambda_r=1e10)
    est_reg = BennettFunctionalDR(
        lambda_r=1e10, degree=2, n_folds=2, a_grid=A_GRID_TEST, seed=44
    )
    est_reg.fit(sample)
    res_reg = est_reg.estimate()

    J_dr_reg = np.array(res_reg.extra["J_dr"])
    J_reg_reg = np.array(res_reg.extra["J_reg"])

    # J_dr should be close to J_reg when lambda_r is huge
    diff = np.abs(J_dr_reg - J_reg_reg)
    max_J_reg = np.max(np.abs(J_reg_reg)) + 1e-10
    rel_diff = diff / max_J_reg

    assert np.all(rel_diff < 0.05), (
        f"T_BEN9: With lambda_r=1e10, J_dr should ≈ J_reg. "
        f"Max relative diff={float(np.max(rel_diff)):.4f} (expected < 0.05). "
        f"J_dr={J_dr_reg}, J_reg={J_reg_reg}"
    )
