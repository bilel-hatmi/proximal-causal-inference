"""
Tests for simulations/methods/features.py (Phase 11).

Tests
-----
T_F1 -- NystromFeatureMap dimensions and whitening sanity.
T_F2 -- NystromFeatureMap no leakage (fit on train, transform on test).
T_F3 -- PairFeatureMap dimensions and basic shape contract.
T_F4 -- PolicyIntegrator: T_pi 1 ~ 1.
T_F5 -- PolicyIntegrator: T_pi A ~ a_target.
T_F6 -- PolicyIntegrator: T_pi W ~ W (constant in t).

Each test runs in well under 30 seconds.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulations.methods.features import (
    NystromFeatureMap,
    PairFeatureMap,
    PolicyFeatureIntegrator,
)


# ══════════════════════════════════════════════════════════════════════════════
#  T_F1 -- NystromFeatureMap dimensions + whitening sanity
# ══════════════════════════════════════════════════════════════════════════════

def test_T_F1_nystrom_dimensions_and_kernel_recovery():
    """
    Check shape contract and that Phi @ Phi^T recovers the RBF Gram matrix
    (the actual property guaranteed by Nystroem whitening with our W = U @
    diag(1/sqrt(S))).

    Note: Phi^T @ Phi / n is NOT supposed to be the identity -- that is a
    misreading of Nystroem feature whitening. The correct property is
    Phi @ Phi^T = K_NN approximated by rank n_features.
    """
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal(n)

    fm = NystromFeatureMap(ell=1.0, n_features=50, seed=1)
    Phi = fm.fit_transform(X)

    # Shape contract
    assert Phi.shape == (n, 50), f"Expected (200, 50), got {Phi.shape}"
    assert np.isfinite(Phi).all(), "Phi contains NaN/Inf"

    # Kernel recovery: Phi @ Phi^T should approximate the RBF Gram K_NN.
    # Diagonal of K_NN is 1 (exp(0)) for RBF; check the diagonal of Phi @ Phi^T.
    diag_pp = np.einsum("ij,ij->i", Phi, Phi)   # (n,)
    diag_err = float(np.abs(diag_pp - 1.0).mean())
    assert diag_err < 0.10, \
        f"Phi @ Phi^T diagonal off (should be ~1): mean error = {diag_err}"

    # Compare a small random patch of Phi @ Phi^T against the true RBF gram
    from simulations.methods.features import _rbf_gram_1d
    sub_idx = np.arange(0, n, 5)                # 40 indices
    K_true  = _rbf_gram_1d(X[sub_idx], X[sub_idx], 1.0)
    K_appr  = Phi[sub_idx] @ Phi[sub_idx].T
    rel_err = float(np.linalg.norm(K_true - K_appr) / np.linalg.norm(K_true))
    assert rel_err < 0.20, \
        f"Nystroem reconstruction relative error too large: {rel_err}"


# ══════════════════════════════════════════════════════════════════════════════
#  T_F2 -- NystromFeatureMap no leakage
# ══════════════════════════════════════════════════════════════════════════════

def test_T_F2_nystrom_no_leakage():
    rng = np.random.default_rng(1)
    X_train = rng.standard_normal(150)
    X_test  = rng.standard_normal(50)

    fm = NystromFeatureMap(ell=1.0, n_features=40, seed=2)
    fm.fit(X_train)
    Phi_test  = fm.transform(X_test)
    Phi_train = fm.transform(X_train)

    assert Phi_test.shape  == (50, 40)
    assert Phi_train.shape == (150, 40)
    assert np.isfinite(Phi_test).all()

    # Refitting on a different dataset must NOT change the original transform
    # of X_train if we reuse the same NystromFeatureMap object on a different
    # dataset (the user is responsible for not refitting in cross-fit code).
    # Here we instead test: a *fresh* fit on (X_train + X_test) gives DIFFERENT
    # landmarks (almost surely), so Phi(X_train) differs.
    fm2 = NystromFeatureMap(ell=1.0, n_features=40, seed=2)
    fm2.fit(np.concatenate([X_train, X_test]))
    Phi_train2 = fm2.transform(X_train)

    diff = np.abs(Phi_train - Phi_train2).mean()
    # They should be different because the landmarks are sampled from a
    # different pool. (If they were identical, that would prove leakage was
    # not isolated.)
    assert diff > 1e-6, "Refitting did not change features -- leakage suspect."


# ══════════════════════════════════════════════════════════════════════════════
#  T_F3 -- PairFeatureMap dimensions
# ══════════════════════════════════════════════════════════════════════════════

def test_T_F3_pair_feature_map_dimensions():
    rng = np.random.default_rng(2)
    n = 300
    Z = rng.standard_normal(n)
    A = rng.uniform(-2, 2, n)

    pm = PairFeatureMap(
        ell_1=1.0, ell_2=1.0,
        n_features_1=40, n_features_2=40,
        n_cross=10, augment_simple=True,
        seed=3,
    )
    Phi = pm.fit_transform(Z, A)

    expected_m = 40 + 40 + 10 + 4   # nystrom(Z) + nystrom(A) + cross + [1, Z, A, Z*A]
    assert Phi.shape == (n, expected_m), \
        f"Expected (n, {expected_m}), got {Phi.shape}"
    assert np.isfinite(Phi).all()

    # n_features_total accessor must agree
    assert pm.n_features_total == expected_m


# ══════════════════════════════════════════════════════════════════════════════
#  T_F4 -- PolicyIntegrator: T_pi 1 ~ 1
# ══════════════════════════════════════════════════════════════════════════════

def test_T_F4_policy_integrator_constant():
    rng = np.random.default_rng(3)
    A_train = rng.uniform(0.0, 5.0, 200)
    h_policy = 0.3
    a_target = 2.5
    n_test = 5
    W = rng.standard_normal(n_test)

    integ = PolicyFeatureIntegrator(h_policy=h_policy, grid_size=400)
    integ.fit_grid(A_train)

    # Feature: constant 1
    def feat(W_in, t):
        return np.ones((len(W_in), 1))

    T = integ.integrate(feat, W, a_target)        # (n_test, 1)
    err = np.abs(T - 1.0).max()
    assert err < 1e-2, f"T_pi 1 should be ~1, got max error {err}"


# ══════════════════════════════════════════════════════════════════════════════
#  T_F5 -- PolicyIntegrator: T_pi A ~ a_target
# ══════════════════════════════════════════════════════════════════════════════

def test_T_F5_policy_integrator_identity_on_A():
    rng = np.random.default_rng(4)
    A_train = rng.uniform(0.0, 5.0, 200)
    h_policy = 0.3
    a_target = 2.5
    W = rng.standard_normal(5)

    integ = PolicyFeatureIntegrator(h_policy=h_policy, grid_size=400)
    integ.fit_grid(A_train)

    # Feature: identity on A. We use a closure that returns t in column 0 for
    # each W_i.
    def feat(W_in, t):
        out = np.empty((len(W_in), 1))
        out[:, 0] = t
        return out

    T = integ.integrate(feat, W, a_target)
    err = np.abs(T - a_target).max()
    # T_pi A should equal a_target up to bandwidth/quadrature noise
    assert err < 2.0 * h_policy, \
        f"T_pi A should be ~{a_target}, got values with max error {err}"


# ══════════════════════════════════════════════════════════════════════════════
#  T_F6 -- PolicyIntegrator: T_pi W ~ W (constant in t)
# ══════════════════════════════════════════════════════════════════════════════

def test_T_F6_policy_integrator_constant_in_t():
    rng = np.random.default_rng(5)
    A_train = rng.uniform(0.0, 5.0, 200)
    h_policy = 0.3
    a_target = 2.5
    W = rng.standard_normal(7)

    integ = PolicyFeatureIntegrator(h_policy=h_policy, grid_size=400)
    integ.fit_grid(A_train)

    # Feature: W (constant in t)
    def feat(W_in, t):
        return W_in.reshape(-1, 1)

    T = integ.integrate(feat, W, a_target)             # (7, 1)
    err = np.abs(T[:, 0] - W).max()
    # Should equal W exactly up to T_pi 1 ~ 1 quadrature error
    assert err < 1e-2, f"T_pi W should be ~W, got max error {err}"


# ══════════════════════════════════════════════════════════════════════════════
#  T_F7 -- compute_TPI per-point ↔ compute_TPI_mean coherence (Phase 11B)
# ══════════════════════════════════════════════════════════════════════════════

def test_T_F7_compute_TPI_mean_equals_compute_TPI_columnwise_mean():
    """
    The per-point T_pi (compute_TPI, Phase 11B) and the column-mean T_pi
    (compute_TPI_mean, Phase 11) must agree by construction:

        compute_TPI(X1, a, integ).mean(axis=0) == compute_TPI_mean(X1, a, integ)

    This pinpoints any discrepancy between the two separable decompositions.
    """
    rng = np.random.default_rng(11)
    n = 300
    Z = rng.standard_normal(n)
    A = rng.uniform(0.0, 5.0, n)
    pm = PairFeatureMap(
        ell_1=1.0, ell_2=1.0,
        n_features_1=40, n_features_2=40,
        n_cross=12, augment_simple=True,
        seed=7,
    ).fit(Z, A)

    integ = PolicyFeatureIntegrator(h_policy=0.4, grid_size=400)
    integ.fit_grid(A)

    a_target = 2.7
    T_per_point = pm.compute_TPI(Z, a_target, integ)          # (n, m_total)
    T_mean      = pm.compute_TPI_mean(Z, a_target, integ)     # (m_total,)

    # Shape sanity
    assert T_per_point.shape == (n, pm.n_features_total)
    assert T_mean.shape == (pm.n_features_total,)

    diff = np.abs(T_per_point.mean(axis=0) - T_mean)
    assert diff.max() < 1e-10, \
        f"compute_TPI(.).mean differs from compute_TPI_mean: max diff = {diff.max()}"
