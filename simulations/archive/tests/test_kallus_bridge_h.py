"""
Tests for KallusMinimaxBridgeH (Phase 11B).

Tests
-----
T_H1 -- Dimensions: Phi_H, Phi_C, A_mat, d, beta, predict, plug_in_policy shapes.
T_H2 -- No np.linalg.inv in source.
T_H3 -- ridge_gmm == kallus_stabilized when S_C = I (lambda_stab_h=0, gamma_critic_h=1).
T_H4 -- Residuals finite and >= 0.
T_H5 -- DGP1 sanity end-to-end (n=300, finite metrics, no catastrophic blow-up).
T_H6 -- plug_in_policy contract: shape (n_test,), finite, propagates h_KDE.
T_H7 -- Cross-fit no leakage (landmarks Phi_H and Phi_C drawn from train indices only).
T_H8 -- Regression: existing KPV bridge tests still green (covered by running their suites).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from simulations.methods.kallus_minimax import KallusMinimaxBridgeH


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dgp1_sample():
    from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP
    dgp = CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)
    return dgp.generate(n=300, seed=42)


@pytest.fixture(scope="module")
def dgp2_sample():
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)
    return dgp.generate(n=300, seed=42)


def _silverman_h(A: np.ndarray) -> float:
    return 1.06 * float(np.std(A)) * len(A) ** (-1.0 / 5.0)


# ══════════════════════════════════════════════════════════════════════════════
#  T_H1 -- Dimensions
# ══════════════════════════════════════════════════════════════════════════════

def test_T_H1_dimensions(dgp1_sample):
    h = KallusMinimaxBridgeH(
        mode="ridge_gmm",
        n_features_h=80, n_features_critic=80,
        compute_diagnostics="light",
    ).fit(dgp1_sample.W, dgp1_sample.A, dgp1_sample.Z, dgp1_sample.Y)

    # beta dimension
    m_H = h._h_n_features
    assert h._beta.shape == (m_H,), f"beta shape {h._beta.shape} != ({m_H},)"
    assert np.isfinite(h._beta).all(), "beta has NaN/Inf"

    # predict shape
    pred = h.predict(dgp1_sample.W, dgp1_sample.A)
    assert pred.shape == (300,), f"predict shape {pred.shape} != (300,)"
    assert np.isfinite(pred).all(), "predict has NaN/Inf"

    # plug_in_policy shape
    plug = h.plug_in_policy(dgp1_sample.W[:50], a=2.0, h=_silverman_h(dgp1_sample.A))
    assert plug.shape == (50,), f"plug_in_policy shape {plug.shape} != (50,)"
    assert np.isfinite(plug).all(), "plug_in_policy has NaN/Inf"


# ══════════════════════════════════════════════════════════════════════════════
#  T_H2 -- No np.linalg.inv
# ══════════════════════════════════════════════════════════════════════════════

def test_T_H2_no_inv_in_source():
    """Audit static: kallus_minimax.py must contain 0 np.linalg.inv() invocations."""
    here = Path(__file__).resolve().parents[2] / "methods" / "kallus_minimax.py"
    src = here.read_text(encoding="utf-8")
    code_lines = [l for l in src.splitlines() if not l.lstrip().startswith("#")]
    code = "\n".join(code_lines)
    matches = re.findall(r"np\.linalg\.inv\s*\(", code)
    assert len(matches) == 0, \
        f"kallus_minimax.py uses np.linalg.inv -- forbidden ({len(matches)} occurrences)"


# ══════════════════════════════════════════════════════════════════════════════
#  T_H3 -- S_C = I equivalence
# ══════════════════════════════════════════════════════════════════════════════

def test_T_H3_ridge_gmm_equiv_kallus_when_S_C_identity(dgp1_sample):
    """
    When lambda_stab_h=0 and gamma_critic_h=1, S_C = I. Then
        (A_mat^T S_C^{-1} A_mat + gamma_H I) beta = A_mat^T S_C^{-1} d
    becomes
        (A_mat^T A_mat + gamma_H I) beta = A_mat^T d
    which is exactly the ridge_gmm system. Both modes must agree.
    """
    common = dict(
        gamma_H=1e-3,
        n_features_h=60, n_features_critic=60,
        feature_seed=999,
        compute_diagnostics="light",
        jitter=1e-10,
    )
    h_ridge = KallusMinimaxBridgeH(mode="ridge_gmm", **common).fit(
        dgp1_sample.W, dgp1_sample.A, dgp1_sample.Z, dgp1_sample.Y
    )
    h_kallus = KallusMinimaxBridgeH(
        mode="kallus_stabilized",
        lambda_stab_h=0.0, gamma_critic_h=1.0,
        **common,
    ).fit(dgp1_sample.W, dgp1_sample.A, dgp1_sample.Z, dgp1_sample.Y)

    rel = np.linalg.norm(h_ridge._beta - h_kallus._beta) / (
        np.linalg.norm(h_ridge._beta) + 1e-15
    )
    assert rel < 1e-5, f"||beta_ridge - beta_kallus|| / ||beta_ridge|| = {rel}"


# ══════════════════════════════════════════════════════════════════════════════
#  T_H4 -- Residuals finite and >= 0
# ══════════════════════════════════════════════════════════════════════════════

def test_T_H4_residuals_finite(dgp1_sample):
    for mode in ("ridge_gmm", "kallus_stabilized"):
        h = KallusMinimaxBridgeH(
            mode=mode,
            n_features_h=50, n_features_critic=50,
            compute_diagnostics="light",
        ).fit(dgp1_sample.W, dgp1_sample.A, dgp1_sample.Z, dgp1_sample.Y)
        assert np.isfinite(h._h_raw_residual)
        assert h._h_raw_residual >= 0.0
        assert np.isfinite(h._h_weighted_residual)
        assert h._h_weighted_residual >= 0.0
        assert np.isfinite(h._beta_norm)
        assert h._beta_norm >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  T_H5 -- DGP1 sanity end-to-end (no catastrophic blow-up)
# ══════════════════════════════════════════════════════════════════════════════

def test_T_H5_dgp1_sanity_end_to_end(dgp1_sample):
    h = KallusMinimaxBridgeH(
        mode="kallus_stabilized",
        n_features_h=80, n_features_critic=80,
        gamma_H=1e-3, lambda_stab_h=1.0, gamma_critic_h=1e-3,
        compute_diagnostics="light",
    ).fit(dgp1_sample.W, dgp1_sample.A, dgp1_sample.Z, dgp1_sample.Y)

    # Predict on a few points
    pred = h.predict(dgp1_sample.W[:30], dgp1_sample.A[:30])
    assert np.isfinite(pred).all()
    # Plug-in policy at a few doses
    h_KDE = _silverman_h(dgp1_sample.A)
    for a in [1.0, 2.0, 3.0]:
        plug = h.plug_in_policy(dgp1_sample.W[:30], a=a, h=h_KDE)
        assert np.isfinite(plug).all(), f"plug_in_policy at a={a} has NaN/Inf"


# ══════════════════════════════════════════════════════════════════════════════
#  T_H6 -- plug_in_policy contract + h_KDE caching
# ══════════════════════════════════════════════════════════════════════════════

def test_T_H6_plug_in_policy_contract(dgp2_sample):
    h_KDE = _silverman_h(dgp2_sample.A)
    h = KallusMinimaxBridgeH(
        mode="kallus_stabilized",
        n_features_h=60, n_features_critic=60,
        compute_diagnostics="light",
    ).fit(dgp2_sample.W, dgp2_sample.A, dgp2_sample.Z, dgp2_sample.Y)

    # Same h reused across doses must use the cached integrator
    for a in [1.8, 2.2, 2.6, 3.0]:
        out = h.plug_in_policy(dgp2_sample.W[:50], a=a, h=h_KDE)
        assert out.shape == (50,)
        assert np.isfinite(out).all()
    # The integrator cache should have exactly 1 entry (same h_KDE used)
    assert len(h._integrator_cache) == 1, \
        f"Expected 1 cached integrator, got {len(h._integrator_cache)}"

    # Different h => different cache entry
    out2 = h.plug_in_policy(dgp2_sample.W[:50], a=2.6, h=h_KDE * 1.5)
    assert out2.shape == (50,)
    assert len(h._integrator_cache) == 2


# ══════════════════════════════════════════════════════════════════════════════
#  T_H7 -- Cross-fit no leakage (landmarks come from train fold only)
# ══════════════════════════════════════════════════════════════════════════════

def test_T_H7_cross_fit_no_leakage(dgp1_sample):
    """
    Two fits on disjoint halves of the data must NOT share landmarks
    (almost-sure property when training data are disjoint).
    """
    n = len(dgp1_sample.W)
    half = n // 2
    common = dict(
        mode="ridge_gmm",
        n_features_h=40, n_features_critic=40,
        feature_seed=11,
        compute_diagnostics="light",
    )
    import copy
    template = KallusMinimaxBridgeH(**common)
    h_a = copy.deepcopy(template).fit(
        dgp1_sample.W[:half], dgp1_sample.A[:half],
        dgp1_sample.Z[:half], dgp1_sample.Y[:half],
    )
    h_b = copy.deepcopy(template).fit(
        dgp1_sample.W[half:], dgp1_sample.A[half:],
        dgp1_sample.Z[half:], dgp1_sample.Y[half:],
    )

    # Phi_H landmarks (first sub-map = W axis)
    W_lm_a = h_a._phi_H_map._map_1._landmarks
    W_lm_b = h_b._phi_H_map._map_1._landmarks
    train_W_a = dgp1_sample.W[:half]
    train_W_b = dgp1_sample.W[half:]
    assert all(w in train_W_a for w in W_lm_a), \
        "Phi_H W-landmarks fold A include points outside fold A (leakage)"
    assert all(w in train_W_b for w in W_lm_b), \
        "Phi_H W-landmarks fold B include points outside fold B (leakage)"
    overlap_W = len(set(W_lm_a.tolist()) & set(W_lm_b.tolist()))
    assert overlap_W == 0, \
        f"Phi_H W-landmark overlap across folds = {overlap_W} (training disjoint)"

    # Phi_C landmarks (first sub-map = Z axis)
    Z_lm_a = h_a._phi_C_map._map_1._landmarks
    Z_lm_b = h_b._phi_C_map._map_1._landmarks
    train_Z_a = dgp1_sample.Z[:half]
    train_Z_b = dgp1_sample.Z[half:]
    assert all(z in train_Z_a for z in Z_lm_a), \
        "Phi_C Z-landmarks fold A include points outside fold A"
    assert all(z in train_Z_b for z in Z_lm_b), \
        "Phi_C Z-landmarks fold B include points outside fold B"
    overlap_Z = len(set(Z_lm_a.tolist()) & set(Z_lm_b.tolist()))
    assert overlap_Z == 0, \
        f"Phi_C Z-landmark overlap across folds = {overlap_Z}"
