"""
pci/tests/test_bridges.py
=========================

Smoke-and-identity tests for the canonical bridges in ``pci.bridges``.
These guarantee that:

* Each bridge fits + predicts deterministically for fixed seed (S6/S7
  reproducibility).
* The shim chain ``pci.bridges -> pci.estimators -> simulations.methods``
  preserves Python identity (pickle compatibility for the cached PKLs in
  ``simulations/results/raw/``).
* Oracle bridges produce the analytic identity expected by the DR
  estimators (S4 baseline).
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def dgp1_sample():
    """Reusable DGP1 sample (n=200) for bridge fits."""
    from pci.dgps.cobb_douglas import CobbDouglasLinearDGP
    return CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95).generate(n=200, seed=11)


@pytest.fixture(scope="module")
def dgp2_sample():
    """Reusable DGP2 sample (n=300) for bridge fits."""
    from pci.dgps.michaelis_menten import MichaelisMentenDGP
    return MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95).generate(n=300, seed=11)


# ════════════════════════════════════════════════════════════════════════════
#  Oracle bridges (pci.bridges.oracle)
# ════════════════════════════════════════════════════════════════════════════

def test_oracle_bridge_h_identity(dgp1_sample):
    """OracleBridgeH.predict(W, A, X) should equal dgp.h0(W, A, X)."""
    from pci.bridges.oracle import OracleBridgeH
    from pci.dgps.cobb_douglas import CobbDouglasLinearDGP
    dgp = CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)
    bridge = OracleBridgeH(dgp)
    pred = bridge.predict(dgp1_sample.W, dgp1_sample.A, None)
    truth = dgp.h0(dgp1_sample.W, dgp1_sample.A, None)
    np.testing.assert_allclose(np.ravel(pred), np.ravel(truth), rtol=1e-10)


def test_oracle_bridge_q_callable(dgp1_sample):
    """OracleBridgeQ.predict(Z, A, X) returns finite values of correct shape."""
    from pci.bridges.oracle import OracleBridgeQ
    from pci.dgps.cobb_douglas import CobbDouglasLinearDGP
    dgp = CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)
    bridge = OracleBridgeQ(dgp)
    val = bridge.predict(dgp1_sample.Z, dgp1_sample.A, None)
    assert val.shape[0] == dgp1_sample.A.shape[0]
    assert np.all(np.isfinite(val))


def test_oracle_bridges_shim_identity():
    """pci.bridges.oracle -> pci.estimators._oracle_bridges identity."""
    from pci.bridges.oracle import OracleBridgeH as A
    from pci.estimators._oracle_bridges import OracleBridgeH as B
    assert A is B, "shim must preserve Python identity for pickle safety"


# ════════════════════════════════════════════════════════════════════════════
#  RFFBridgeH (pci.bridges.rff_h)
# ════════════════════════════════════════════════════════════════════════════

def test_rff_bridge_h_fit_deterministic(dgp2_sample):
    """RFFBridgeH fit gives bit-identical predictions for fixed seed."""
    from pci.bridges.rff_h import RFFBridgeH
    s = dgp2_sample
    preds = []
    for _ in range(2):
        b = RFFBridgeH(
            lambda_1=1e-3, lambda_2=1e-3,
            ell_W=1.0, ell_A=1.0, ell_Z=1.0,
            n_features_W=200, n_features_A=200, n_features_Z=200,
            seed=42,
        )
        b.fit(s.W, s.A, s.Z, s.Y)
        preds.append(b.predict(s.W[:50], s.A[:50]))
    np.testing.assert_allclose(preds[0], preds[1], rtol=1e-12)


def test_rff_bridge_h_shim_identity():
    """pci.bridges.rff_h -> pci.estimators.rff_bridge_h identity."""
    from pci.bridges.rff_h import RFFBridgeH as A
    from pci.estimators.rff_bridge_h import RFFBridgeH as B
    assert A is B


# ════════════════════════════════════════════════════════════════════════════
#  KPV bridges (pci.bridges.kpv)
# ════════════════════════════════════════════════════════════════════════════

def test_kpv_bridge_h_fit_deterministic(dgp2_sample):
    """KPVBridgeH gives bit-identical predictions for fixed seed."""
    from pci.bridges.kpv import KPVBridgeH
    s = dgp2_sample
    preds = []
    for _ in range(2):
        b = KPVBridgeH(lambda_1=1e-3, lambda_2=1e-3,
                       ell_W=1.0, ell_A=1.0, ell_Z=1.0)
        b.fit(s.W, s.A, s.Z, s.Y)
        preds.append(b.predict(s.W[:50], s.A[:50]))
    np.testing.assert_allclose(preds[0], preds[1], rtol=1e-10)


def test_kpv_policy_bridge_q_fit_deterministic(dgp2_sample):
    """KPVPolicyBridgeQ gives bit-identical q for fixed seed."""
    from pci.bridges.kpv import KPVPolicyBridgeQ
    s = dgp2_sample
    a_grid = np.array([1.8, 2.2, 2.6, 3.0])
    qs = []
    for _ in range(2):
        b = KPVPolicyBridgeQ(
            a_grid=a_grid, h_KDE=0.3, lambda_Q=1e-3,
            ell_W=1.0, ell_A=1.0, ell_Z=1.0,
        )
        b.fit(s.W, s.A, s.Z)
        # predict(Z, A, dose_index=2) gives weights for the reference dose
        qs.append(b.predict(s.Z, s.A, 2))
    np.testing.assert_allclose(qs[0], qs[1], rtol=1e-10)


# ════════════════════════════════════════════════════════════════════════════
#  Bennett bridges (pci.bridges.bennett_h, pci.bridges.bennett_riesz) -- C9
# ════════════════════════════════════════════════════════════════════════════

def test_bennett_indep_bridge_h_fit_deterministic(dgp2_sample):
    """BennettIndepBridgeH fits + predicts bit-identically for fixed seed."""
    from pci.bridges.bennett_h import BennettIndepBridgeH
    s = dgp2_sample
    preds = []
    plug_ins = []
    for _ in range(2):
        b = BennettIndepBridgeH(
            m_h=200, m_c=100, ell_h=2.75, ell_c=2.0,
            lambda_h=1e-3, gamma_critic=1e-4, seed=42,
        )
        b.fit(s.W, s.A, s.Z, s.Y)
        preds.append(b.predict(s.W[:50], s.A[:50]))
        plug_ins.append(b.plug_in_policy(s.W[:50], 2.6, 0.3))
    np.testing.assert_array_equal(preds[0], preds[1])
    np.testing.assert_array_equal(plug_ins[0], plug_ins[1])


def test_bennett_indep_bridge_h_shim_identity():
    """pci.bridges.bennett_h -> pci.estimators.bennett_indep_h identity."""
    from pci.bridges.bennett_h import BennettIndepBridgeH as A
    from pci.estimators.bennett_indep_h import BennettIndepBridgeH as B
    assert A is B


def test_bennett_indep_bridge_h_legacy_shim_identity():
    """The simulations.methods shim chain still resolves to the canonical class."""
    from pci.bridges.bennett_h import BennettIndepBridgeH as A
    from simulations.methods.bennett_indep_h import BennettIndepBridgeH as B
    assert A is B


def test_bennett_policy_riesz_polynomial_deterministic(dgp2_sample):
    """BennettPolicyRiesz polynomial mode is deterministic."""
    from pci.bridges.bennett_riesz import BennettPolicyRiesz
    s = dgp2_sample
    preds = []
    for _ in range(2):
        b = BennettPolicyRiesz(lambda_r=1e-3, degree=2, feature_map_type="polynomial")
        b.fit(s.W, s.A, s.Z, a_grid=np.array([1.8, 2.2, 2.6, 3.0]), bandwidth=0.3)
        preds.append(b.predict(s.Z[:30], s.A[:30], a_target=2.6))
    np.testing.assert_array_equal(preds[0], preds[1])


def test_bennett_policy_riesz_rff_deterministic(dgp2_sample):
    """BennettPolicyRiesz RFF mode (Phase 12B winner) is deterministic."""
    from pci.bridges.bennett_riesz import BennettPolicyRiesz
    s = dgp2_sample
    preds = []
    for _ in range(2):
        b = BennettPolicyRiesz(
            lambda_r=1e-2, feature_map_type="rff",
            n_features=200, ell_scale=3.5, rff_seed=42,
        )
        b.fit(s.W, s.A, s.Z, a_grid=np.array([1.8, 2.2, 2.6, 3.0]), bandwidth=0.3)
        preds.append(b.predict(s.Z[:30], s.A[:30], a_target=2.6))
    np.testing.assert_array_equal(preds[0], preds[1])


def test_bennett_riesz_shim_identity():
    """All Bennett-Riesz public + private symbols preserve identity."""
    from pci.bridges.bennett_riesz import (
        BennettPolicyRiesz, StabilizedBennettPolicyRiesz,
        PolynomialPairFeatureMap, PolicyPolynomialIntegrator,
        RandomFourierPairFeatureMap, RandomFourierPolicyIntegrator,
        gaussian_raw_moment, _spd_solve_bennett,
    )
    from pci.estimators.bennett_riesz import (
        BennettPolicyRiesz as B_BPR,
        StabilizedBennettPolicyRiesz as B_SBR,
        PolynomialPairFeatureMap as B_PPFM,
        PolicyPolynomialIntegrator as B_PPI,
        RandomFourierPairFeatureMap as B_RFPFM,
        RandomFourierPolicyIntegrator as B_RFPI,
        gaussian_raw_moment as B_grm,
        _spd_solve_bennett as B_sdb,
    )
    assert BennettPolicyRiesz is B_BPR
    assert StabilizedBennettPolicyRiesz is B_SBR
    assert PolynomialPairFeatureMap is B_PPFM
    assert PolicyPolynomialIntegrator is B_PPI
    assert RandomFourierPairFeatureMap is B_RFPFM
    assert RandomFourierPolicyIntegrator is B_RFPI
    assert gaussian_raw_moment is B_grm
    assert _spd_solve_bennett is B_sdb


def test_bennett_riesz_legacy_shim_identity():
    """simulations.methods.bennett_riesz -> canonical identity."""
    from pci.bridges.bennett_riesz import BennettPolicyRiesz as A
    from simulations.methods.bennett_riesz import BennettPolicyRiesz as B
    assert A is B


def test_kpv_shim_identity():
    """pci.bridges.kpv -> pci.estimators.kpv_bridge identity (private helpers too)."""
    from pci.bridges.kpv import KPVBridgeH as A1, KPVPolicyBridgeQ as A2
    from pci.estimators.kpv_bridge import KPVBridgeH as B1, KPVPolicyBridgeQ as B2
    assert A1 is B1
    assert A2 is B2

    # private helpers used by older tests
    from pci.bridges.kpv import _rbf_gram, _median_bandwidth, _spd_solve, _rbf_kde_convolution
    from pci.estimators.kpv_bridge import (
        _rbf_gram as _rbf_gram_legacy,
        _median_bandwidth as _mb_legacy,
        _spd_solve as _spd_legacy,
        _rbf_kde_convolution as _rbf_kde_legacy,
    )
    assert _rbf_gram is _rbf_gram_legacy
    assert _median_bandwidth is _mb_legacy
    assert _spd_solve is _spd_legacy
    assert _rbf_kde_convolution is _rbf_kde_legacy
