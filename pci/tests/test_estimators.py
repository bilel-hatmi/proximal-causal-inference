"""
pci/tests/test_estimators.py
============================

Smoke tests for the canonical DR estimators in ``pci.estimators``.
Each estimator is fitted on a tiny DGP1/DGP2 sample and we check:

* ``fit(sample).estimate()`` returns an EstimationResult with finite
  ``psi_hat`` / ``V_hat``.
* The output is deterministic for a fixed seed (essential for S6 figure
  reproducibility).
* The ``extra["J_dr"]`` array length matches the requested ``a_grid``.

Estimators covered: TwoStageLeastSquares, DRDoseResponse (oracle),
DRKernel (KPV-h + cross-fit q), BennettIndepFunctionalDR (winner of S7).
"""
from __future__ import annotations

import numpy as np
import pytest


_AGRID = np.array([1.8, 2.2, 2.6, 3.0])


@pytest.fixture(scope="module")
def small_dgp1():
    """DGP1 (Cobb-Douglas) sample with n=200 -- fast for CI."""
    from pci.dgps.cobb_douglas import CobbDouglasLinearDGP
    dgp = CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)
    return dgp, dgp.generate(n=200, seed=0)


@pytest.fixture(scope="module")
def small_dgp2():
    from pci.dgps.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)
    return dgp, dgp.generate(n=300, seed=0)


def _extract_J(res):
    """Helper: pull J_dr (or J_policy) from the EstimationResult.extra dict."""
    extra = res.extra
    for key in ("J_dr", "J_policy", "ate"):
        if key in extra:
            return np.asarray(extra[key])
    raise AssertionError(
        f"EstimationResult.extra has no J_dr/J_policy/ate; got keys {list(extra.keys())[:5]}"
    )


# ════════════════════════════════════════════════════════════════════════════
#  TwoStageLeastSquares
# ════════════════════════════════════════════════════════════════════════════

def test_two_stage_least_squares_smoke(small_dgp1):
    from pci.estimators.two_stage_linear import TwoStageLeastSquares
    dgp, s = small_dgp1
    est = TwoStageLeastSquares()
    res = est.fit(s).estimate()
    assert np.isfinite(res.psi_hat)
    assert np.isfinite(res.V_hat)


# ════════════════════════════════════════════════════════════════════════════
#  Oracle DR (S4 baseline)
# ════════════════════════════════════════════════════════════════════════════

def test_dr_dose_response_oracle_deterministic(small_dgp1):
    """DRDoseResponse with oracle h, q is bit-identical for the same seed."""
    from pci.estimators.dr_dose_response import DRDoseResponse
    from pci.bridges.oracle import OracleBridgeH, OracleBridgeQ
    dgp, s = small_dgp1
    a_grid = np.array([1.8, 2.2, 2.6])
    Js = []
    for _ in range(2):
        est = DRDoseResponse(
            a_grid=a_grid,
            h_model=OracleBridgeH(dgp),
            q_model=OracleBridgeQ(dgp),
            bandwidth=0.4,
        )
        res = est.fit(s).estimate()
        Js.append(_extract_J(res))
    np.testing.assert_allclose(Js[0], Js[1], rtol=1e-10)


# ════════════════════════════════════════════════════════════════════════════
#  DRKernel (KPV-h + cross-fit q)
# ════════════════════════════════════════════════════════════════════════════

def test_dr_kernel_deterministic(small_dgp2):
    """DRKernel produces identical J_dr on two identical fits (DGP2 simulation path)."""
    from pci.estimators.dr_kernel import DRKernel
    from pci.bridges.kpv import KPVPolicyBridgeQ
    dgp, s = small_dgp2
    h_KDE = 0.3
    q_model = KPVPolicyBridgeQ(
        a_grid=_AGRID, h_KDE=h_KDE, lambda_Q=1e-3,
        ell_W=2.0, ell_A=2.0, ell_Z=2.0,
    )
    Js = []
    for _ in range(2):
        est = DRKernel(
            a_grid=_AGRID, q_model=q_model, bandwidth=h_KDE,
            n_folds=2, random_state=42, ref_dose_index=2, cross_fit_q=True,
            bridge_kwargs=dict(lambda_1=1e-3, lambda_2=1e-3,
                                ell_W=2.0, ell_A=2.0, ell_Z=2.0),
        )
        res = est.fit(s).estimate()
        Js.append(_extract_J(res))
    np.testing.assert_allclose(Js[0], Js[1], rtol=1e-9)


# ════════════════════════════════════════════════════════════════════════════
#  Bennett (S7 winner)
# ════════════════════════════════════════════════════════════════════════════

def test_bennett_indep_functional_dr_deterministic(small_dgp2):
    """BennettIndepFunctionalDR (S7 winner) is bit-identical for same seed."""
    from pci.estimators.bennett_indep_dr import BennettIndepFunctionalDR
    dgp, s = small_dgp2
    h_KDE = 0.3
    Js = []
    for _ in range(2):
        est = BennettIndepFunctionalDR(
            m_h=200, m_c=100,
            ell_h=2.75, ell_c=2.0,
            lambda_h=1e-3, gamma_critic=1e-4,
            lambda_r=1e-2, n_features_r=200, ell_scale_r=3.5,
            n_folds=2, a_grid=_AGRID.tolist(),
            bandwidth=h_KDE, ref_dose_index=2, seed=42,
        )
        res = est.fit(s).estimate()
        Js.append(_extract_J(res))
    # Bennett solves SPD systems in float64; bit-identity expected.
    np.testing.assert_allclose(Js[0], Js[1], rtol=1e-9)


def test_bennett_J_dr_shape_matches_a_grid(small_dgp2):
    """J_dr length must match a_grid (S7 dose-response figures)."""
    from pci.estimators.bennett_indep_dr import BennettIndepFunctionalDR
    dgp, s = small_dgp2
    a_grid = [1.8, 2.2, 2.6, 3.0]
    est = BennettIndepFunctionalDR(
        m_h=200, m_c=100, ell_h=2.75, ell_c=2.0,
        lambda_h=1e-3, gamma_critic=1e-4,
        lambda_r=1e-2, n_features_r=200, ell_scale_r=3.5,
        n_folds=2, a_grid=a_grid,
        bandwidth=0.3, ref_dose_index=2, seed=42,
    )
    res = est.fit(s).estimate()
    J = _extract_J(res)
    assert J.shape == (len(a_grid),)


# ════════════════════════════════════════════════════════════════════════════
#  Estimator shim identity (S7 cached PKLs depend on this)
# ════════════════════════════════════════════════════════════════════════════

def test_estimator_shim_identity():
    """All canonical pci.estimators.* classes equal their simulations.methods.* shim."""
    from pci.estimators.dr_kernel import DRKernel as A1
    from simulations.methods.dr_kernel import DRKernel as B1
    assert A1 is B1

    from pci.estimators.bennett_indep_dr import BennettIndepFunctionalDR as A2
    from simulations.methods.bennett_indep_dr import BennettIndepFunctionalDR as B2
    assert A2 is B2

    from pci.estimators.two_stage_linear import TwoStageLeastSquares as A3
    from simulations.methods.two_stage_linear import TwoStageLeastSquares as B3
    assert A3 is B3
