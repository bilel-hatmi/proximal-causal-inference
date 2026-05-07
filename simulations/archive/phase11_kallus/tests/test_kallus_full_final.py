"""
Smoke tests for kallus_full_final.py (Phase 11C).

Tests
-----
T1 -- Factories: each cell ID instantiates with the correct h-bridge and q-bridge classes.
T2 -- Cross-fit propagation: cross_fit_q=True for estimated q, False for oracle.
T3 -- Quick smoke run: 1 cell per type at n=200, M=2 -> finite metrics, PKL written.
T4 -- Report builder accepts empty / partial results dict without crash.
"""
from __future__ import annotations

import numpy as np
import pytest

import simulations.experiments.kallus_full_final as mod


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dgp2():
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    return MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)


@pytest.fixture
def tmp_raw(tmp_path, monkeypatch):
    """Redirect _RAW_DIR to a tmp dir so smoke tests don't pollute real PKLs."""
    tmp = tmp_path / "raw"
    tmp.mkdir()
    monkeypatch.setattr(mod, "_RAW_DIR", tmp)
    return tmp


# ══════════════════════════════════════════════════════════════════════════════
#  T1 -- Factories instantiate the right classes
# ══════════════════════════════════════════════════════════════════════════════

def test_T1_factories_instantiate_correctly(dgp2):
    """Each cell ID produces a DRKernel with the correct h-bridge type and q model."""
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVBridgeH, KPVPolicyBridgeQ
    from simulations.methods.kallus_minimax import (
        KallusStabilizedPolicyQ, KallusMinimaxBridgeH,
    )
    from simulations.methods._oracle_bridges import OracleBridgeQ

    h_KDE, ell_W0, ell_A0, ell_Z0 = 0.5, 1.0, 1.0, 1.0

    expected_q = {
        "A":  KPVPolicyBridgeQ,
        "B1": KallusStabilizedPolicyQ,
        "B2": KallusStabilizedPolicyQ,
        "C":  KPVPolicyBridgeQ,
        "D1": KallusStabilizedPolicyQ,
        "D2": KallusStabilizedPolicyQ,
        "E":  OracleBridgeQ,
        "F":  OracleBridgeQ,
    }
    expected_h_factory_present = {
        "A": False, "B1": False, "B2": False, "E": False,
        "C": True, "D1": True, "D2": True, "F": True,
    }

    for cid in ["A", "B1", "B2", "C", "D1", "D2", "E", "F"]:
        factory = mod.make_factory(cid, h_KDE, ell_W0, ell_A0, ell_Z0, dgp2)
        drk = factory()
        assert isinstance(drk, DRKernel), f"{cid}: not a DRKernel"
        # q_model class
        assert isinstance(drk.q_model, expected_q[cid]), \
            f"{cid}: q_model is {type(drk.q_model).__name__}, expected {expected_q[cid].__name__}"
        # h_bridge_factory presence
        if expected_h_factory_present[cid]:
            assert drk.h_bridge_factory is not None, f"{cid}: h_bridge_factory should be set"
            inst = drk.h_bridge_factory()
            assert isinstance(inst, KallusMinimaxBridgeH), \
                f"{cid}: h_bridge_factory returns {type(inst).__name__}"
        else:
            assert drk.h_bridge_factory is None, f"{cid}: h_bridge_factory should be None (use KPV)"


# ══════════════════════════════════════════════════════════════════════════════
#  T2 -- cross_fit_q propagation
# ══════════════════════════════════════════════════════════════════════════════

def test_T2_cross_fit_q_propagation(dgp2):
    """cross_fit_q=True for estimated q (KPV, Kallus); False for oracle q."""
    h_KDE, ell_W0, ell_A0, ell_Z0 = 0.5, 1.0, 1.0, 1.0

    estimated_q_cells = ["A", "B1", "B2", "C", "D1", "D2"]
    oracle_q_cells    = ["E", "F"]

    for cid in estimated_q_cells:
        drk = mod.make_factory(cid, h_KDE, ell_W0, ell_A0, ell_Z0, dgp2)()
        assert drk.cross_fit_q is True, f"{cid}: cross_fit_q should be True (estimated q)"

    for cid in oracle_q_cells:
        drk = mod.make_factory(cid, h_KDE, ell_W0, ell_A0, ell_Z0, dgp2)()
        assert drk.cross_fit_q is False, f"{cid}: cross_fit_q should be False (oracle q)"


# ══════════════════════════════════════════════════════════════════════════════
#  T3 -- Quick smoke run (real fit, n=200, M=2)
# ══════════════════════════════════════════════════════════════════════════════

def test_T3_smoke_run_no_nan(dgp2, tmp_raw):
    """Run 3 cells (A, D1, E) at n=200, M=2 -> PKLs written, decomp finite."""
    n, M = 200, 2
    for cid in ["A", "D1", "E"]:
        decomp, wall, data = mod._run_one_cell(
            dgp2, cid, n=n, M=M, cell_idx=0, force=True,
        )
        # PKL exists
        pkl = tmp_raw / f"kallusfull_{cid}_n{n}_M{M}.pkl"
        assert pkl.exists(), f"{cid}: PKL not written"
        # decomp has finite ref-dose metrics
        assert decomp is not None, f"{cid}: decomp is None"
        bias = float(decomp["bias_dr"][mod.REF_IDX])
        assert np.isfinite(bias), f"{cid}: bias_DR not finite"
        cov  = float(decomp["coverage_ref"])
        assert 0.0 <= cov <= 1.0, f"{cid}: cov out of [0,1]: {cov}"


# ══════════════════════════════════════════════════════════════════════════════
#  T4 -- Report builder handles partial / empty
# ══════════════════════════════════════════════════════════════════════════════

def test_T4_report_builder_robust():
    """Report builder must work on empty dict, partial dict (some cells missing)."""
    # Empty
    lines_empty = mod._build_report({})
    assert isinstance(lines_empty, list)
    assert any("Phase 11C" in l for l in lines_empty)
    assert any("Configurations" in l for l in lines_empty)
    assert any("Decision" in l for l in lines_empty)

    # Partial: only A at n=2000
    fake_decomp = dict(
        bias_reg=np.array([0.0, 0.0, -0.02, 0.0]),
        bias_dr=np.array([0.0, 0.0, -0.025, 0.0]),
        bse_grid=np.array([0.0, 0.0, 0.65, 0.0]),
        coverage_ref=0.93,
        ess_min_mean=180.0,
        v_total=np.array([0.0, 0.0, 0.005, 0.0]),
        v_reg=np.array([0.0, 0.0, 0.004, 0.0]),
        v_corr=np.array([0.0, 0.0, 0.001, 0.0]),
        w_p99_mean=2.5,
    )
    partial = {("A", 2000): (fake_decomp, 1.0, None)}
    lines_partial = mod._build_report(partial)
    text = "\n".join(lines_partial)
    assert "0.65" in text or "+0.65" in text or "-0.65" in text
    # No crash on missing cells D1/D2 (Decision section returns "(no full-Kallus results)")
    assert "no full-Kallus results" in text or "insufficient" in text or "Case" in text
