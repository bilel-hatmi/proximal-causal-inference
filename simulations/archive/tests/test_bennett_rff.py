"""
Phase 12B Bennett-lite RFF — Test suite T_RFF1 .. T_RFF9.

Coverage
--------
T_RFF1 : RandomFourierPairFeatureMap shape + reproducibility
T_RFF2 : RandomFourierPolicyIntegrator closed-form vs Monte Carlo
T_RFF3 : BennettPolicyRiesz feature_map_type="rff" mean(r_hat) ~= 1
T_RFF4 : Y still forbidden in fit() with rff mode
T_RFF5 : riesz_residual decreases with n_features
T_RFF6 : RFF stress (large n_features, no NaN)
T_RFF7 : BennettFunctionalDR smoke RFF mode (DGP2)
T_RFF8 : feature_map_type="polynomial" backward-compat (Phase 12A reproducibility)
T_RFF9 : StabilizedBennettPolicyRiesz smoke

These tests verify that Phase 12A code paths remain bit-identical when
feature_map_type defaults to "polynomial".
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from simulations.methods.bennett_riesz import (
    BennettPolicyRiesz,
    PolynomialPairFeatureMap,
    PolicyPolynomialIntegrator,
    RandomFourierPairFeatureMap,
    RandomFourierPolicyIntegrator,
    StabilizedBennettPolicyRiesz,
)
from simulations.methods.bennett_functional_dr import BennettFunctionalDR
from simulations.dgp.michaelis_menten import MichaelisMentenDGP


# ---------------------------------------------------------------------------
# T_RFF1 — RandomFourierPairFeatureMap shapes + reproducibility
# ---------------------------------------------------------------------------

def test_T_RFF1_rff_feature_map_shapes_and_reproducibility() -> None:
    rng = np.random.default_rng(2026)
    x1 = rng.normal(size=120)
    x2 = rng.normal(size=120)

    m = 100
    fm_a = RandomFourierPairFeatureMap(n_features=m, ell_scale=2.0, seed=7).fit(x1, x2)
    fm_b = RandomFourierPairFeatureMap(n_features=m, ell_scale=2.0, seed=7).fit(x1, x2)
    fm_c = RandomFourierPairFeatureMap(n_features=m, ell_scale=2.0, seed=99).fit(x1, x2)

    assert fm_a.n_features() == m
    out_a = fm_a.transform(x1, x2)
    out_b = fm_b.transform(x1, x2)
    out_c = fm_c.transform(x1, x2)
    assert out_a.shape == (120, m)
    np.testing.assert_allclose(out_a, out_b, rtol=0, atol=1e-12,
                               err_msg="same seed must give bit-identical features")
    assert not np.allclose(out_a, out_c), "different seed must give different features"
    assert np.isfinite(out_a).all()


# ---------------------------------------------------------------------------
# T_RFF2 — RandomFourierPolicyIntegrator vs Monte Carlo
# ---------------------------------------------------------------------------

def test_T_RFF2_rff_integrator_closed_form_vs_mc() -> None:
    rng = np.random.default_rng(42)
    n = 30
    n_mc = 50_000
    x1_train = rng.normal(0, 1, size=120)
    x2_train = rng.normal(0, 1, size=120)

    fm = RandomFourierPairFeatureMap(n_features=64, ell_scale=2.0, seed=11).fit(
        x1_train, x2_train
    )
    integrator = RandomFourierPolicyIntegrator(fm)

    W = rng.normal(0, 1, size=n)
    a_target = 0.7
    bandwidth = 0.4

    # Closed form
    out_cf = integrator.integrate(W, a_target, bandwidth)   # (n, m)

    # Monte Carlo: draw A_k ~ N(a_target, bw^2), compute mean over k
    A_mc = rng.normal(a_target, bandwidth, size=n_mc)        # (n_mc,)
    # For each W_i, compute psi(W_i, A_k) for all k and average over k.
    # psi shape if we tile: (n, n_mc, m). Use matmul trick.
    # Better: compute features for (W_i, A_k) batch (n*n_mc, 2)
    # Memory check: 30 * 50_000 = 1.5M rows × 64 features = 96MB float64. Too much.
    # Instead: iterate over W_i (only n=30).
    out_mc = np.empty_like(out_cf)
    for i in range(n):
        W_rep = np.full(n_mc, W[i])
        psi_mc = fm.transform(W_rep, A_mc)   # (n_mc, m)
        out_mc[i] = psi_mc.mean(axis=0)

    err = np.abs(out_cf - out_mc).max()
    # MC error scales like 1/sqrt(n_mc); 50_000 samples should give error < 1e-2.
    # Closed form is exact, so this tests both implementation correctness and
    # the math note derivation.
    assert err < 1e-2, f"closed form vs MC max err = {err:.4e} (expected < 1e-2)"


# ---------------------------------------------------------------------------
# T_RFF3 — BennettPolicyRiesz mode='rff' fundamental moment condition
# ---------------------------------------------------------------------------

def test_T_RFF3_rff_riesz_mean_one_condition() -> None:
    """
    For g(W,A)=1 (constant), Riesz equation reduces to E[r_pi(Z,A)] = 1.
    Tests that mode='rff' gives mean(r_hat) ~ 1 on training data.
    """
    rng = np.random.default_rng(2025)
    n = 800
    U = rng.normal(0, 1, n)
    Z = U + 0.3 * rng.normal(0, 1, n)
    W = U + 0.3 * rng.normal(0, 1, n)
    A = U + 0.3 * rng.normal(0, 1, n)

    riesz = BennettPolicyRiesz(
        lambda_r=1e-2,
        feature_map_type="rff",
        n_features=200,
        ell_scale=2.5,
        rff_seed=2025,
    )
    riesz.fit(W=W, A=A, Z=Z, a_grid=[0.0], bandwidth=0.4)
    r_hat = riesz.predict(Z, A, a_target=0.0)
    mean_r = float(np.mean(r_hat))
    # RFF gives more flexibility than polynomial -> tolerance 0.30 (vs 0.15 for poly)
    assert abs(mean_r - 1.0) < 0.30, f"mean(r_hat) = {mean_r:.4f}, expected ~1.0"
    assert np.isfinite(r_hat).all()


# ---------------------------------------------------------------------------
# T_RFF4 — Y still forbidden in fit() with RFF mode
# ---------------------------------------------------------------------------

def test_T_RFF4_no_Y_in_fit_rff_mode() -> None:
    rng = np.random.default_rng(0)
    W = rng.normal(size=50)
    A = rng.normal(size=50)
    Z = rng.normal(size=50)
    Y = rng.normal(size=50)
    riesz = BennettPolicyRiesz(
        lambda_r=1e-3,
        feature_map_type="rff",
        n_features=64,
        ell_scale=2.0,
        rff_seed=1,
    )
    with pytest.raises(TypeError):
        riesz.fit(W, A, Z, a_grid=[0.0], bandwidth=0.5, Y=Y)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# T_RFF5 — riesz_residual decreases as n_features increases
# ---------------------------------------------------------------------------

def test_T_RFF5_riesz_residual_decreases_with_n_features() -> None:
    rng = np.random.default_rng(123)
    n = 600
    U = rng.normal(0, 1, n)
    Z = U + 0.3 * rng.normal(0, 1, n)
    W = U + 0.3 * rng.normal(0, 1, n)
    A = 0.5 * U + 0.5 * rng.normal(0, 1, n)
    bandwidth = 0.4

    n_feats_list = [50, 100, 250, 500]
    residuals = []
    for nf in n_feats_list:
        riesz = BennettPolicyRiesz(
            lambda_r=1e-3,
            feature_map_type="rff",
            n_features=nf,
            ell_scale=2.5,
            rff_seed=7,
        )
        riesz.fit(W=W, A=A, Z=Z, a_grid=[0.0], bandwidth=bandwidth)
        res = riesz.riesz_residual(W, A, Z, a_target=0.0)
        residuals.append(res)

    # Trend: at least 2/3 consecutive pairs should decrease
    decreases = sum(
        1 for i in range(len(residuals) - 1) if residuals[i + 1] <= residuals[i] * 1.05
    )
    assert decreases >= 2, (
        f"expected residual to decrease with n_features, got: {residuals}"
    )


# ---------------------------------------------------------------------------
# T_RFF6 — Stress: large n_features, no NaN, fast
# ---------------------------------------------------------------------------

def test_T_RFF6_large_n_features_finite() -> None:
    rng = np.random.default_rng(0)
    n = 200
    W = rng.normal(0, 1, n)
    A = rng.normal(0, 1, n)
    Z = rng.normal(0, 1, n)

    riesz = BennettPolicyRiesz(
        lambda_r=1e-2,
        feature_map_type="rff",
        n_features=750,
        ell_scale=2.0,
        rff_seed=0,
    )
    t0 = time.time()
    riesz.fit(W=W, A=A, Z=Z, a_grid=[0.0, 0.5], bandwidth=0.5)
    elapsed = time.time() - t0
    r_hat = riesz.predict(Z, A, a_target=0.0)
    assert np.isfinite(r_hat).all()
    assert elapsed < 10.0, f"fit too slow at n=200, m=750: {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# T_RFF7 — BennettFunctionalDR smoke RFF mode (DGP2)
# ---------------------------------------------------------------------------

def test_T_RFF7_functional_dr_rff_smoke() -> None:
    dgp = MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)
    sample = dgp.generate(n=300, seed=42)

    est = BennettFunctionalDR(
        lambda_h=3e-5,
        ell_scale=3.5,
        lambda_r=1e-2,
        feature_map_type="rff",
        n_features=128,
        ell_scale_rff=2.5,
        rff_seed_base=2024,
        n_folds=3,
        a_grid=[1.8, 2.2, 2.6, 3.0],
        seed=42,
    )
    est.fit(sample, m_true_fn=dgp.m_true)
    res = est.estimate()

    assert np.isfinite(res.psi_hat)
    assert res.V_hat > 0.0
    extra = res.extra
    for k in [
        "J_dr", "J_reg", "V_total", "V_reg", "V_corr", "V_cov",
        "ESS_grid", "weight_p99_grid", "riesz_residual_grid",
        "riesz_residual_mean", "feature_map_type", "n_features",
        "ell_scale_rff",
    ]:
        assert k in extra, f"missing extra key {k}"
    assert extra["feature_map_type"] == "rff"
    assert extra["n_features"] == 128
    # All Bennett scores finite
    assert np.isfinite(np.array(extra["J_dr"])).all()
    assert np.isfinite(np.array(extra["V_hat_grid"])).all()


# ---------------------------------------------------------------------------
# T_RFF8 — Polynomial mode backward-compat (Phase 12A reproducibility)
# ---------------------------------------------------------------------------

def test_T_RFF8_polynomial_mode_backward_compat() -> None:
    """
    BennettPolicyRiesz with feature_map_type='polynomial' must give bit-identical
    output to the pre-12B default behaviour (just `degree=2`).
    """
    rng = np.random.default_rng(7)
    n = 200
    W = rng.normal(size=n)
    A = rng.normal(size=n)
    Z = rng.normal(size=n)

    # New API explicit
    r_new = BennettPolicyRiesz(
        lambda_r=1e-2, degree=2, feature_map_type="polynomial"
    ).fit(W=W, A=A, Z=Z, a_grid=[0.0, 0.5], bandwidth=0.5)
    # Default API (relies on default = "polynomial")
    r_default = BennettPolicyRiesz(lambda_r=1e-2, degree=2).fit(
        W=W, A=A, Z=Z, a_grid=[0.0, 0.5], bandwidth=0.5
    )

    p_new = r_new.predict(Z, A, 0.0)
    p_def = r_default.predict(Z, A, 0.0)
    np.testing.assert_allclose(p_new, p_def, rtol=0, atol=1e-14,
                               err_msg="polynomial mode default vs explicit must be bit-equal")


# ---------------------------------------------------------------------------
# T_RFF9 — StabilizedBennettPolicyRiesz smoke
# ---------------------------------------------------------------------------

def test_T_RFF9_stabilized_riesz_smoke() -> None:
    rng = np.random.default_rng(2025)
    n = 500
    U = rng.normal(0, 1, n)
    Z = U + 0.3 * rng.normal(0, 1, n)
    W = U + 0.3 * rng.normal(0, 1, n)
    A = 0.5 * U + 0.5 * rng.normal(0, 1, n)

    riesz = StabilizedBennettPolicyRiesz(
        lambda_stab=1.0,
        gamma_critic=1e-3,
        lambda_r=1e-2,
        feature_map_type="rff",
        n_features=128,
        ell_scale=2.5,
        rff_seed=2025,
    )
    riesz.fit(W=W, A=A, Z=Z, a_grid=[0.0], bandwidth=0.4)
    r_hat = riesz.predict(Z, A, a_target=0.0)
    assert np.isfinite(r_hat).all()
    res = riesz.riesz_residual(W, A, Z, a_target=0.0)
    assert np.isfinite(res)
    # Stabilized solver should produce well-behaved correction (not near-zero
    # but not exploding either).
    mean_r = float(np.mean(r_hat))
    assert abs(mean_r) < 50.0, f"mean(r_hat) explodes: {mean_r}"
