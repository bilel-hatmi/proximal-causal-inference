"""
pci/tests/test_tuning.py
========================

Reproducibility tests for the canonical tuning protocol modules in
``pci.tuning``. These ensure that:

* Blind score gates trip exactly as documented (S6.7 lambda_h x m_h
  interpolation gate caught the Phase 17 SER collapse).
* ``extract_diagnostics`` aggregates per-rep records into the dict shape
  expected by the score functions.
* The factory functions build estimators with the BIN_mh2000 (S7
  winner) defaults.
* The ``BennettAdaptiveTuner`` and ``DRKPVAdaptiveTuner`` constructors
  accept all documented kwargs and expose stable Stage 1 grids.

A separate end-to-end smoke (Stage 1, M=2, n=100) lives in
``test_reproducibility.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ════════════════════════════════════════════════════════════════════════════
#  bennett_blind_score_v2 -- Phase 18 gates
# ════════════════════════════════════════════════════════════════════════════

def test_bennett_blind_score_v2_passes_BIN_mh2000():
    """The S7 winner (BIN_mh2000) passes all gates with a sensible score."""
    from pci.tuning.blind_score import bennett_blind_score_v2
    diags = {
        "lambda_h": 1e-5, "m_h": 2000,        # gate: 1e-5 * 2000 = 0.02 > 0.01 PASS
        "kappa_h": 87000, "kappa_r": 1e3,     # gate: kappa_h < 1e8 PASS
        "RR_ref": 1e-3,                        # gate: < 1e-1 PASS
        "residual_norm_h": 1e-3,
        "eff_rank_h": 100, "eff_rank_r": 100,
    }
    s = bennett_blind_score_v2(diags)
    assert np.isfinite(s)
    assert 0.0 < s < 1.0  # composite weights bounded by [0, 1] in normal regime


def test_bennett_blind_score_v2_interpolation_gate():
    """Gate 1: lambda_h * m_h < 0.01 -> inf (Phase 17 SER collapse fix)."""
    from pci.tuning.blind_score import bennett_blind_score_v2
    diags = {
        "lambda_h": 1e-6, "m_h": 2000,        # 1e-6 * 2000 = 0.002 < 0.01 -> FAIL
        "kappa_h": 1e5, "kappa_r": 1e3,
        "RR_ref": 1e-3, "residual_norm_h": 1e-5,
        "eff_rank_h": 100, "eff_rank_r": 100,
    }
    assert bennett_blind_score_v2(diags) == float("inf")


def test_bennett_blind_score_v2_kappa_gate():
    """Gate 2: kappa_h > 1e8 -> inf."""
    from pci.tuning.blind_score import bennett_blind_score_v2
    diags = {
        "lambda_h": 1e-3, "m_h": 2000,
        "kappa_h": 1e9, "kappa_r": 1e3,       # FAIL
        "RR_ref": 1e-3, "residual_norm_h": 1e-3,
        "eff_rank_h": 100, "eff_rank_r": 100,
    }
    assert bennett_blind_score_v2(diags) == float("inf")


def test_bennett_blind_score_v2_RR_gate():
    """Gate 3: RR_ref > 1e-1 -> inf."""
    from pci.tuning.blind_score import bennett_blind_score_v2
    diags = {
        "lambda_h": 1e-3, "m_h": 2000,
        "kappa_h": 1e5, "kappa_r": 1e3,
        "RR_ref": 0.5, "residual_norm_h": 1e-3,  # RR FAIL
        "eff_rank_h": 100, "eff_rank_r": 100,
    }
    assert bennett_blind_score_v2(diags) == float("inf")


# ════════════════════════════════════════════════════════════════════════════
#  drkpv_blind_score
# ════════════════════════════════════════════════════════════════════════════

def test_drkpv_blind_score_passes_phase17_winner():
    """DRKPV cfg2 winner (Phase 17) gives a finite score."""
    from pci.tuning.blind_score import drkpv_blind_score
    diags = {
        "kappa_r": 1e50,         # DGP2 inherent
        "RR_ref": 1e-3,
        "residual_norm_r": 0.05,
        "SER": 1.0,
    }
    s = drkpv_blind_score(diags)
    assert np.isfinite(s)


def test_drkpv_blind_score_RR_gate():
    from pci.tuning.blind_score import drkpv_blind_score
    diags = {"kappa_r": 1e50, "RR_ref": 0.5, "residual_norm_r": 0.05, "SER": 1.0}
    assert drkpv_blind_score(diags) == float("inf")


# ════════════════════════════════════════════════════════════════════════════
#  extract_diagnostics
# ════════════════════════════════════════════════════════════════════════════

def test_extract_diagnostics_empty_records():
    """No good records -> M_ok=0 dict, no crash."""
    from pci.tuning.extract_diagnostics import extract_diagnostics
    data = {"records": [{"error": "nan"}, {"error": "blew up"}], "meta": {"n": 1000}}
    out = extract_diagnostics(data)
    assert out["M_ok"] == 0


def test_extract_diagnostics_aggregates_correctly():
    """Mean-aggregated output matches the obvious arithmetic mean."""
    from pci.tuning.extract_diagnostics import extract_diagnostics
    K = 4
    recs = [
        {"error": None, "kappa_h": 1e3, "kappa_r": 1e2,
         "eff_rank_h": 50, "eff_rank_r": 50,
         "residual_norm_h": 0.001, "residual_norm_r": 0.01,
         "riesz_residual_grid_mean": [0.001] * K, "ESS_min": 100.0,
         "weight_p99_grid": [1.0] * K,
         "J_dr": [0.5] * K, "V_hat_grid": [0.01] * K},
        {"error": None, "kappa_h": 3e3, "kappa_r": 3e2,
         "eff_rank_h": 70, "eff_rank_r": 70,
         "residual_norm_h": 0.003, "residual_norm_r": 0.03,
         "riesz_residual_grid_mean": [0.003] * K, "ESS_min": 200.0,
         "weight_p99_grid": [3.0] * K,
         "J_dr": [0.7] * K, "V_hat_grid": [0.04] * K},
    ]
    data = {"records": recs, "meta": {"n": 1000}}
    out = extract_diagnostics(data)
    assert out["M_ok"] == 2
    assert out["kappa_h"] == pytest.approx(2e3)         # mean(1e3, 3e3)
    assert out["RR_ref"] == pytest.approx(0.002)        # mean(0.001, 0.003)
    assert out["ESS_min_ratio"] == pytest.approx(0.15)  # mean(100,200)/1000


def test_extract_diagnostics_ref_idx_param():
    """ref_idx=1 (DGP1 K=3) vs ref_idx=2 (DGP2 K=4) gives different RR_ref."""
    from pci.tuning.extract_diagnostics import extract_diagnostics
    rec = {
        "error": None, "kappa_h": 1e3, "kappa_r": 1e2,
        "eff_rank_h": 50, "eff_rank_r": 50,
        "residual_norm_h": 0.001, "residual_norm_r": 0.01,
        "riesz_residual_grid_mean": [0.1, 0.2, 0.3, 0.4],
        "ESS_min": 100.0, "weight_p99_grid": [1.0, 2.0, 3.0, 4.0],
        "J_dr": [0.5] * 4, "V_hat_grid": [0.01] * 4,
    }
    data = {"records": [rec], "meta": {"n": 1000}}
    out_default = extract_diagnostics(data, ref_idx=2)
    out_dgp1 = extract_diagnostics(data, ref_idx=1)
    assert out_default["RR_ref"] == pytest.approx(0.3)
    assert out_dgp1["RR_ref"] == pytest.approx(0.2)


# ════════════════════════════════════════════════════════════════════════════
#  Factories (BIN_mh2000 defaults are the S7 winner)
# ════════════════════════════════════════════════════════════════════════════

def test_bennett_defaults_match_BIN_mh2000():
    """BENNETT_DEFAULTS in pci.tuning.factories must equal the S7 winner spec."""
    from pci.tuning.factories import BENNETT_DEFAULTS
    assert BENNETT_DEFAULTS["m_h"] == 2000
    assert BENNETT_DEFAULTS["ell_h"] == 2.75
    assert BENNETT_DEFAULTS["lambda_h"] == 1e-5
    assert BENNETT_DEFAULTS["gamma_critic"] == 1e-4
    assert BENNETT_DEFAULTS["lambda_r"] == 1e-2
    assert BENNETT_DEFAULTS["n_features_r"] == 500
    assert BENNETT_DEFAULTS["ell_scale_r"] == 3.5


def test_drkpv_defaults_match_phase17_cfg2_winner():
    from pci.tuning.factories import DRKPV_DEFAULTS
    assert DRKPV_DEFAULTS["lambda_h"] == 3e-5
    assert DRKPV_DEFAULTS["ell_scale"] == 3.5


def test_make_bennett_estimator_smoke():
    """Factory returns a fittable estimator with the right a_grid."""
    from pci.tuning.factories import make_bennett_estimator, BENNETT_DEFAULTS
    a_grid = [1.8, 2.2, 2.6, 3.0]
    est = make_bennett_estimator(
        BENNETT_DEFAULTS, dgp=None, h_KDE=0.3,
        ell_W=1.0, ell_A=1.0, ell_Z=1.0,
        a_grid=a_grid, ref_dose_index=2,
    )
    assert hasattr(est, "fit")


# ════════════════════════════════════════════════════════════════════════════
#  AdaptiveTuner constructors and grids
# ════════════════════════════════════════════════════════════════════════════

def test_bennett_adaptive_tuner_construct():
    from pci.tuning.adaptive_tuner_bennett import BennettAdaptiveTuner
    t = BennettAdaptiveTuner(mode="smoke", start_stage=99, stop_stage=99)
    assert t.M_explore == 3
    assert t.M_validate == 5
    assert t.seed_tune == 62_000_000
    grids = t.stage_1_grids()
    assert "lambda_h" in grids
    assert 1e-6 not in grids["lambda_h"], (
        "Phase 18: lambda_h=1e-6 was removed from grid (always fails interpolation gate)"
    )
    assert 1e-5 in grids["lambda_h"]


def test_drkpv_adaptive_tuner_construct():
    from pci.tuning.adaptive_tuner_drkpv import DRKPVAdaptiveTuner
    t = DRKPVAdaptiveTuner(mode="smoke", start_stage=99, stop_stage=99)
    assert t.M_explore == 3
    assert t.M_validate == 5
    assert t.seed_tune == 70_000_000
    grids = t.stage_1_grids()
    assert "lambda_h" in grids
    # Phase 17 keeps 1e-6 in DRKPV grid (no h-bridge interpolation issue)
    assert 1e-6 in grids["lambda_h"]


def test_bennett_tuner_stage1_cell_count():
    """Phase 18: Bennett Stage 1 sweeps TWO cells (n=1000 + n=2000)."""
    from pci.tuning.adaptive_tuner_bennett import BennettAdaptiveTuner
    assert len(BennettAdaptiveTuner.STAGE1_CELLS) == 2
    cells = BennettAdaptiveTuner.STAGE1_CELLS
    n_values = sorted(c[0] for c in cells)
    assert n_values == [1000, 2000]


def test_drkpv_tuner_stage1_cell_count():
    """Phase 17: DRKPV Stage 1 sweeps a SINGLE cell."""
    from pci.tuning.adaptive_tuner_drkpv import DRKPVAdaptiveTuner
    assert len(DRKPVAdaptiveTuner.STAGE1_CELLS) == 1


# ════════════════════════════════════════════════════════════════════════════
#  Tuning shim identity
# ════════════════════════════════════════════════════════════════════════════

def test_tuning_shim_identity_blind_score():
    """Original Phase 18 script re-exports the canonical bennett_blind_score_v2."""
    from pci.tuning.blind_score import bennett_blind_score_v2 as A
    from simulations.experiments.s7_simulations.s7_8_blind_tuning_bennett import (
        bennett_blind_score_v2 as B,
    )
    assert A is B


def test_tuning_shim_identity_factories():
    """make_bennett_estimator: Phase 18 has a wrapper preserving local A_GRID."""
    from pci.tuning.factories import make_bennett_estimator as A
    from simulations.experiments.s7_simulations.s7_8_blind_tuning_bennett import (
        make_bennett_estimator as B,
    )
    # B is a wrapper (not identity), but both must be callable
    assert callable(A) and callable(B)
