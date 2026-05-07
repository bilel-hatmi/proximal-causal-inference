"""
Smoke tests for dgp2_tradeoff_exploration.py -- Phase 9B.3

Tests
-----
T1 -- make_block_h_factory introspection : H_baseline and H_lam1e-4_ell2.0 produce
      DRKernel with correct hyperparameters.
T2 -- J_policy_true changes with h_policy_scale : rel-diff at REF_IDX > 0.1%.
T3 -- Quick mode end-to-end (Block H, n=200, M=2) : pkl saved, _decompose works,
      report file exists and contains "Block H".
T4 -- Report builder handles missing blocks : _build_report with empty results does
      not raise; output contains "Block Q" placeholder text.
T5 -- Checkpoint skip/resume : write a minimal .partial.pkl; verify
      _run_one_block_tradeoff resumes and produces the final pkl.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

import simulations.archive.utilities.dgp2_tradeoff_exploration as mod


# ── Fixture : redirect output dirs to tmp_path ────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    raw  = tmp_path / "raw"
    summ = tmp_path / "summ"
    raw.mkdir()
    summ.mkdir()
    monkeypatch.setattr(mod, "_RAW_DIR",     raw)
    monkeypatch.setattr(mod, "_SUMM_DIR",    summ)
    monkeypatch.setattr(mod, "_REPORT_PATH", summ / "dgp2_tradeoff_exploration.md")
    return raw, summ


def _make_dgp():
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    return MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)


# ══════════════════════════════════════════════════════════════════════════════
#  T1 — make_block_h_factory introspection
# ══════════════════════════════════════════════════════════════════════════════

def test_T1_make_block_h_factory_introspection():
    """
    H_baseline -> DRKernel with default bridge_kwargs (empty), lambda_Q=1e-3,
                  h_KDE == bandwidth, compute_cond=False.
    H_lam1e-4_ell2.0 -> bridge_kwargs["lambda_1"]=1e-4, ell_W=ell_W0*2.0.
    """
    a_grid = np.array([1.8, 2.2, 2.6, 3.0])
    h_ref  = 0.5
    ell_W0 = 1.0
    ell_A0 = 1.0
    ell_Z0 = 1.0

    # H_baseline
    spec_bl = mod.BLOCK_H_CONFIGS["H_baseline"]
    factory_bl = mod.make_factory(spec_bl, a_grid, h_ref, ell_W0, ell_A0, ell_Z0)
    est_bl = factory_bl()
    assert est_bl.q_model.lambda_Q == pytest.approx(1e-3)
    assert est_bl.q_model.h_KDE   == pytest.approx(h_ref)   # h_policy_scale=1.0
    assert est_bl.bandwidth        == pytest.approx(h_ref)
    # No custom bridge_kwargs for baseline -> None or empty dict
    bk = est_bl.bridge_kwargs
    has_lambda = bk is not None and "lambda_1" in bk
    assert not has_lambda, "H_baseline should not set lambda_1"
    assert getattr(est_bl.q_model, "compute_cond", False) is False

    # H_lam1e-4_ell2.0
    spec_A = mod.BLOCK_H_CONFIGS["H_lam1e-4_ell2.0"]
    factory_A = mod.make_factory(spec_A, a_grid, h_ref, ell_W0, ell_A0, ell_Z0)
    est_A = factory_A()
    bk_A  = est_A.bridge_kwargs
    assert bk_A is not None, "H_lam1e-4_ell2.0 should have bridge_kwargs"
    assert bk_A["lambda_1"] == pytest.approx(1e-4)
    assert bk_A["lambda_2"] == pytest.approx(1e-4)
    assert bk_A["ell_W"]    == pytest.approx(ell_W0 * 2.0)
    assert bk_A["ell_A"]    == pytest.approx(ell_A0 * 2.0)
    assert bk_A["ell_Z"]    == pytest.approx(ell_Z0 * 2.0)
    assert est_A.q_model.h_KDE == pytest.approx(h_ref)   # h_policy_scale=1.0


# ══════════════════════════════════════════════════════════════════════════════
#  T2 — J_policy_true changes with h_policy_scale
# ══════════════════════════════════════════════════════════════════════════════

def test_T2_J_policy_true_differs_with_scale():
    """J_policy_true(h_ref) != J_policy_true(1.5*h_ref) by more than 0.1% at REF_IDX."""
    from simulations.experiments.dgp2_bias_diagnostics import (
        _silverman_h, _compute_J_policy_true, A_GRID, REF_IDX,
    )
    dgp    = _make_dgp()
    sample = dgp.generate(n=500, seed=0)
    h_ref  = _silverman_h(sample.A)

    J_ref     = _compute_J_policy_true(dgp, A_GRID, h_ref)
    J_smooth  = _compute_J_policy_true(dgp, A_GRID, 1.5 * h_ref)

    rel_diff = abs(float(J_ref[REF_IDX]) - float(J_smooth[REF_IDX])) / max(abs(float(J_ref[REF_IDX])), 1e-12)
    assert rel_diff > 1e-3, (
        f"J_policy_true should differ by >0.1% when h_policy_scale changes; "
        f"got rel_diff={rel_diff:.6f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T3 — Quick mode end-to-end (Block H, n=200, M=2)
# ══════════════════════════════════════════════════════════════════════════════

def test_T3_quick_mode_end_to_end(tmp_dirs):
    """
    Run Block H quick with n=200, M=2 (overriding _BLOCK_H_QUICK to 1 config
    for speed). Verify pkl is created, _decompose returns finite values, report exists.
    """
    raw_dir, summ_dir = tmp_dirs
    dgp = _make_dgp()

    # Override quick config list for test speed (just H_baseline)
    results_H = mod.run_block_H(
        dgp,
        n_list=[200],
        M=2,
        config_ids=["H_baseline"],
        force=False,
    )

    # PKL should exist
    pkls = list(raw_dir.glob("dgp2tradeoff_H_H_baseline_n200_M2.pkl"))
    assert len(pkls) == 1, f"Expected 1 pkl, found: {[p.name for p in pkls]}"

    # _decompose should return finite values
    decomp = results_H["H_baseline"][200][0]
    assert decomp is not None, "_decompose returned None (all reps failed)"
    assert np.isfinite(decomp["bias_reg"][mod.REF_IDX]), "bias_reg(ref) not finite"
    assert np.isfinite(decomp["bias_dr"][mod.REF_IDX]),  "bias_dr(ref) not finite"
    assert np.isfinite(decomp["coverage_ref"]),           "coverage_ref not finite"

    # Report should be generated
    report_lines = mod._build_report(results_H, None, None, None)
    report_text  = "\n".join(report_lines)
    assert "Block H" in report_text, "Report missing 'Block H'"

    mod._REPORT_PATH.write_text(report_text, encoding="utf-8")
    assert mod._REPORT_PATH.exists(), "Report file not created"


# ══════════════════════════════════════════════════════════════════════════════
#  T4 — Report builder handles missing blocks gracefully
# ══════════════════════════════════════════════════════════════════════════════

def test_T4_report_builder_missing_blocks():
    """
    _build_report with empty results_H and None for all other blocks must not raise
    and must contain placeholder text for Block Q.
    """
    report_lines = mod._build_report(
        results_H={},
        results_Q=None,
        results_P=None,
        results_N=None,
        dgp=None,
    )
    report_text = "\n".join(report_lines)
    assert "Block Q" in report_text, "Missing 'Block Q' in report"
    assert "Block H" in report_text, "Missing 'Block H' in report"
    assert "Block P" in report_text, "Missing 'Block P' in report"
    assert "Block N" in report_text, "Missing 'Block N' in report"
    # No exception raised = test passes


# ══════════════════════════════════════════════════════════════════════════════
#  T5 — Checkpoint skip / resume
# ══════════════════════════════════════════════════════════════════════════════

def test_T5_checkpoint_skip_resume(tmp_dirs):
    """
    Pre-write a .partial.pkl with 1 record; verify _run_one_block_tradeoff
    resumes from it (final pkl appears, wall is short).
    """
    raw_dir, _ = tmp_dirs
    dgp = _make_dgp()

    n, M = 200, 3
    config_id = "H_baseline"
    spec      = mod.BLOCK_H_CONFIGS[config_id]
    block_char = "H"
    cfg_idx    = mod.BLOCK_H_ORDER.index(config_id)

    # Compute a valid J_pt and seed for the partial record
    from simulations.experiments.dgp2_bias_diagnostics import (
        _silverman_h, _compute_J_policy_true, A_GRID, _build_record, K,
    )
    ref_sample = dgp.generate(n=n, seed=0)
    h_ref = _silverman_h(ref_sample.A)
    J_pt  = _compute_J_policy_true(dgp, A_GRID, h_ref)

    # Generate 1 valid record using the full estimator (seed=seed_base+0)
    seed_base = mod._seed(block_char, cfg_idx, n)
    sample    = dgp.generate(n=n, seed=seed_base)
    factory   = mod.make_factory(spec, A_GRID, h_ref,
                                  1.0, 1.0, 1.0)  # ell values don't matter for smoke
    est = factory()
    est.fit(sample, m_true_fn=dgp.m_true)
    res = est.estimate()
    rec = _build_record(res, K, seed_base, J_pt)

    # Write partial pkl with 1 record
    label = mod._pkl_label(block_char, config_id, n, M)
    partial_path = raw_dir / f"{label}.partial.pkl"
    with open(partial_path, "wb") as fh:
        pickle.dump([rec], fh)

    assert partial_path.exists(), "Partial pkl not written"

    # Run _run_one_block_tradeoff — should resume from the 1 existing record
    decomp, wall, data = mod._run_one_block_tradeoff(
        dgp, config_id, spec, block_char, cfg_idx, n, M, force=False,
    )

    final_path = raw_dir / f"{label}.pkl"
    assert final_path.exists(), "Final pkl not created after resume"
    assert not partial_path.exists(), "Partial pkl not cleaned up"
    assert decomp is not None, "_decompose returned None after resume"
