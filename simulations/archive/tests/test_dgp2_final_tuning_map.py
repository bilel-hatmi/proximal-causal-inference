"""
Smoke tests for dgp2_final_tuning_map.py (Phase 9B.5).

Tests
-----
T1 -- Config parsing : F config IDs and parsing round-trip.
T2 -- Smoke run (n=100, M=2) : new config produces a pkl with finite metrics.
T3 -- Blind score does NOT use forbidden truth-based fields.
T4 -- Report builders run without errors on minimal input.
"""
from __future__ import annotations

import inspect
import pickle

import numpy as np
import pytest

import simulations.archive.utilities.dgp2_final_tuning_map as mod


@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    raw  = tmp_path / "raw"
    summ = tmp_path / "summ"
    raw.mkdir()
    summ.mkdir()
    monkeypatch.setattr(mod, "_RAW_DIR",       raw)
    monkeypatch.setattr(mod, "_SUMM_DIR",      summ)
    monkeypatch.setattr(mod, "_REPORT_ELL",    summ / "dgp2_final_ell_extension.md")
    monkeypatch.setattr(mod, "_REPORT_BLIND",  summ / "dgp2_blind_tuning_protocol.md")
    # Patch sources to empty for blind protocol smoke
    empty_src = tmp_path / "empty"
    empty_src.mkdir()
    monkeypatch.setattr(mod, "_TRADEOFF_RAW",   empty_src)
    monkeypatch.setattr(mod, "_OPTREGION_RAW",  empty_src)
    return raw, summ


def _make_dgp():
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    return MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)


# ══════════════════════════════════════════════════════════════════════════════
#  T1 -- Config parsing
# ══════════════════════════════════════════════════════════════════════════════

def test_T1_config_parsing():
    """F_lam<...>_ell<...> ids round-trip through _parse_config_id."""
    cid = mod._f_config_id(3e-5, 4.5)
    assert cid == "F_lam3e-5_ell4p50"
    lam, ell = mod._parse_config_id(cid)
    assert lam == pytest.approx(3e-5)
    assert ell == pytest.approx(4.5)

    # Phase 9B.4 E_lam<...>_ell<...> should also parse
    lam2, ell2 = mod._parse_config_id("E_lam1e-4_ell2p50")
    assert lam2 == pytest.approx(1e-4)
    assert ell2 == pytest.approx(2.5)

    # Phase 9B.3 H_lam<...>_ell<...> with dot
    lam3, ell3 = mod._parse_config_id("H_lam1e-4_ell2.5")
    assert lam3 == pytest.approx(1e-4)
    assert ell3 == pytest.approx(2.5)

    # Baseline returns (None, None)
    lam4, ell4 = mod._parse_config_id("E_baseline")
    assert lam4 is None and ell4 is None

    # F_NEW_CONFIGS contains exactly 3 truly-new configs (ell in {4.0, 4.5, 5.0})
    assert len(mod.F_NEW_CONFIGS) == 3
    for cid in mod.F_NEW_CONFIGS:
        lam, ell = mod._parse_config_id(cid)
        assert lam == pytest.approx(3e-5)
        assert ell in {4.0, 4.5, 5.0}


# ══════════════════════════════════════════════════════════════════════════════
#  T2 -- Smoke run end-to-end
# ══════════════════════════════════════════════════════════════════════════════

def test_T2_smoke_run(tmp_dirs):
    """run_ell_extension(n=100, M=3, configs=[one new]) produces a finite pkl."""
    raw, summ = tmp_dirs
    dgp = _make_dgp()
    cid = mod._f_config_id(3e-5, 4.5)
    results = mod.run_ell_extension(
        dgp, n=100, M=3, force=False, configs=[cid],
    )
    assert cid in results
    decomp, wall, data = results[cid]
    if decomp is not None:
        assert np.isfinite(decomp["bias_dr"][mod.REF_IDX])
        assert 0.0 <= decomp["coverage_ref"] <= 1.0
    pkls = list(raw.glob("dgp2final_*.pkl"))
    assert len(pkls) >= 1, f"No PKL written, found: {pkls}"


# ══════════════════════════════════════════════════════════════════════════════
#  T3 -- Blind score does NOT use forbidden fields
# ══════════════════════════════════════════════════════════════════════════════

def test_T3_blind_uses_no_forbidden_fields():
    """
    Inspect _blind_diagnostics source: it must NOT reference any forbidden
    truth-based field name.
    """
    src = inspect.getsource(mod._blind_diagnostics)
    for forbidden in mod._BLIND_FORBIDDEN_FIELDS:
        assert forbidden not in src, (
            f"_blind_diagnostics references forbidden field {forbidden!r}"
        )

    # Same check on _compute_blind_scores
    src2 = inspect.getsource(mod._compute_blind_scores)
    for forbidden in mod._BLIND_FORBIDDEN_FIELDS:
        assert forbidden not in src2, (
            f"_compute_blind_scores references forbidden field {forbidden!r}"
        )

    # Sanity: returned dict from _blind_diagnostics on a synthetic record set
    fake_records = [
        dict(
            psi_hat=1.0, V_hat=0.01, ESS_min=100.0, q_clip_fraction=0.0,
            bandwidth=0.5,
            weight_p99_grid=[2.0, 2.5, 3.0, 2.0],
            weight_max_grid=[5.0, 5.5, 6.0, 5.0],
            riesz_residual_grid_mean=[0.001, 0.002, 0.003, 0.001],
            cond_M_grid_mean=[1e3, 1e3, 1e3, 1e3],
            jensen_gap=[0.005, 0.006, 0.007, 0.005],
            neg_share_grid_mean=[0.0, 0.0, 0.0, 0.0],
            V_correction_grid=[0.001, 0.001, 0.001, 0.001],
            V_reg_grid=[0.009, 0.009, 0.009, 0.009],
            J_dr=[1.0, 1.05, 1.1, 1.0],
            error=None,
        )
        for _ in range(3)
    ]
    diag = mod._blind_diagnostics(dict(records=fake_records))
    expected_keys = {
        "n_reps", "psi_std", "psi_cv", "V_hat_mean", "V_corr_mean",
        "V_reg_mean", "V_corr_share", "ess_min_mean", "w_p99_mean",
        "w_max_mean", "q_clip_mean", "neg_share_mean",
        "riesz_residual_mean", "cond_M_mean", "jensen_mean", "bandwidth",
    }
    assert expected_keys.issubset(diag.keys())
    # NO forbidden key in output
    for forbidden in mod._BLIND_FORBIDDEN_FIELDS:
        assert forbidden not in diag


# ══════════════════════════════════════════════════════════════════════════════
#  T4 -- Report builders smoke-run on minimal input
# ══════════════════════════════════════════════════════════════════════════════

def test_T4_report_builders(tmp_dirs):
    """_build_ell_report and _build_blind_report run on minimal/empty input."""
    raw, summ = tmp_dirs
    # Empty results -> ell report still builds (tables empty)
    txt_ell = mod._build_ell_report({})
    assert "## 1. Objective" in txt_ell
    assert "## 4. Verdict" in txt_ell

    # Empty scan -> blind report builds with placeholder
    txt_blind = mod._build_blind_report({})
    assert "## 1. Protocol definition" in txt_blind
    assert "## 6. Verdict" in txt_blind
