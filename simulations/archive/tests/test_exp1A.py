"""
Smoke tests for Exp1A coverage runner.

Five tests, n=100, M=5, SNR=0.95 — should all pass in < 90s.

Test IDs
--------
T_E1a : scalar Oracle DGP1 — no NaN in metrics
T_E1b : DRKernel oracle-q DGP1 — M_ok ≥ 1, coverage_per_dose populated
T_E1c : DRKernel xfit DGP1 — M_ok ≥ 1, q_clip_fraction_mean finite
T_E1d : checkpoint resume — partial pkl resumption correct
T_E1e : variance normalization — SE uses V_hat_grid / n, not sqrt(V_hat_grid)
"""
from __future__ import annotations

import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dgp1_095():
    from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP
    return CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)


@pytest.fixture(scope="module")
def a_grid_dgp1():
    return np.array([-0.5, 0.0, 0.5])


# ══════════════════════════════════════════════════════════════════════════════
#  T_E1a — scalar Oracle smoke
# ══════════════════════════════════════════════════════════════════════════════

def test_T_E1a_scalar_oracle_smoke(dgp1_095, tmp_path):
    """Oracle DGP1 n=100, M=5 — metrics dict has no NaN in core fields."""
    from simulations.archive.utilities.exp1A_coverage import _RAW_DIR, run_scalar_block

    # Redirect raw dir to tmp_path for isolation
    import simulations.archive.utilities.exp1A_coverage as mod
    orig_raw = mod._RAW_DIR
    mod._RAW_DIR = tmp_path
    try:
        data = run_scalar_block(
            dgp1_095, "oracle", n=100, M=5,
            seed_base=99_000, label="smoke_oracle_dgp1",
        )
    finally:
        mod._RAW_DIR = orig_raw

    assert data["M"] == 5
    m = data["metrics"]
    assert np.isfinite(m["bias"]),     "bias should be finite"
    assert np.isfinite(m["rmse"]),     "rmse should be finite"
    assert np.isfinite(m["coverage"]), "coverage should be finite"
    assert m["rmse"] < 10.0,           "rmse sanity check (not wildly large)"


# ══════════════════════════════════════════════════════════════════════════════
#  T_E1b — DRKernel oracle-q smoke
# ══════════════════════════════════════════════════════════════════════════════

def test_T_E1b_curve_oracle_smoke(dgp1_095, a_grid_dgp1, tmp_path):
    """DRKernel oracle-q DGP1 n=100, M=5 — M_ok ≥ 1, coverage_per_dose OK."""
    from simulations.archive.utilities.exp1A_coverage import (
        _silverman_h, _compute_J_policy_true,
        _factory_drk_oracle, run_curve_block, summarize_exp1A,
    )
    import simulations.archive.utilities.exp1A_coverage as mod
    orig_raw = mod._RAW_DIR
    mod._RAW_DIR = tmp_path
    try:
        h_ref  = _silverman_h(dgp1_095.generate(n=100, seed=0).A)
        J_pt   = _compute_J_policy_true(dgp1_095, a_grid_dgp1, h_ref)
        factory = _factory_drk_oracle(dgp1_095, a_grid_dgp1, h_ref)
        data   = run_curve_block(
            dgp1_095, factory, a_grid_dgp1, J_pt,
            n=100, M=5, seed_base=99_100,
            label="smoke_drk_oracle_dgp1", h_ref=h_ref,
        )
    finally:
        mod._RAW_DIR = orig_raw

    summ = summarize_exp1A(data)
    assert summ["M_ok"] >= 1, f"All reps failed: {data['records'][0]['error']}"
    cpd = summ["coverage_per_dose"]
    assert len(cpd) == len(a_grid_dgp1), "coverage_per_dose should have K entries"
    assert all(np.isfinite(v) for v in cpd), "coverage_per_dose should be finite"


# ══════════════════════════════════════════════════════════════════════════════
#  T_E1c — DRKernel xfit smoke
# ══════════════════════════════════════════════════════════════════════════════

def test_T_E1c_curve_xfit_smoke(dgp1_095, a_grid_dgp1, tmp_path):
    """DRKernel xfit DGP1 n=100, M=5 — M_ok ≥ 1, q_clip_fraction_mean finite."""
    from simulations.archive.utilities.exp1A_coverage import (
        _silverman_h, _compute_J_policy_true,
        _factory_drk_xfit, run_curve_block, summarize_exp1A,
    )
    import simulations.archive.utilities.exp1A_coverage as mod
    orig_raw = mod._RAW_DIR
    mod._RAW_DIR = tmp_path
    try:
        h_ref   = _silverman_h(dgp1_095.generate(n=100, seed=0).A)
        J_pt    = _compute_J_policy_true(dgp1_095, a_grid_dgp1, h_ref)
        factory = _factory_drk_xfit(dgp1_095, a_grid_dgp1, h_ref)
        data    = run_curve_block(
            dgp1_095, factory, a_grid_dgp1, J_pt,
            n=100, M=5, seed_base=99_200,
            label="smoke_drk_xfit_dgp1", h_ref=h_ref,
        )
    finally:
        mod._RAW_DIR = orig_raw

    summ = summarize_exp1A(data)
    assert summ["M_ok"] >= 1, f"All reps failed: {data['records'][0]['error']}"
    qcf = summ.get("q_clip_fraction_mean")
    assert np.isfinite(qcf), f"q_clip_fraction_mean should be finite, got {qcf}"
    # ESS diagnostics should be populated
    assert np.isfinite(summ.get("ess_min_mean", np.nan)), "ESS_min_mean should be finite"


# ══════════════════════════════════════════════════════════════════════════════
#  T_E1d — checkpoint / resume
# ══════════════════════════════════════════════════════════════════════════════

def test_T_E1d_checkpoint_resume(dgp1_095, a_grid_dgp1, tmp_path):
    """
    Partial checkpoint resumption test.

    1. Run 3 reps with CHECKPOINT_FREQ=3 → .partial.pkl written.
    2. Do NOT save final .pkl (simulate crash after 3rd rep).
    3. Re-run requesting 6 reps total → should resume from rep 3.
    4. Verify final data has exactly 6 records with correct seeds.
    """
    import simulations.archive.utilities.exp1A_coverage as mod
    from simulations.archive.utilities.exp1A_coverage import (
        _silverman_h, _compute_J_policy_true,
        _factory_drk_oracle, CHECKPOINT_FREQ,
    )

    orig_raw  = mod._RAW_DIR
    orig_freq = mod.CHECKPOINT_FREQ
    mod._RAW_DIR       = tmp_path
    mod.CHECKPOINT_FREQ = 3    # trigger partial save after rep 3

    label     = "chkpt_test"
    save_path = tmp_path / f"{label}.pkl"
    partial   = tmp_path / f"{label}.partial.pkl"

    try:
        h_ref   = _silverman_h(dgp1_095.generate(n=50, seed=0).A)
        J_pt    = _compute_J_policy_true(dgp1_095, a_grid_dgp1, h_ref)
        factory = _factory_drk_oracle(dgp1_095, a_grid_dgp1, h_ref)

        # Phase 1 — run 3 reps (CHECKPOINT_FREQ=3 → writes .partial.pkl at rep 3)
        data_first = mod.run_curve_block(
            dgp1_095, factory, a_grid_dgp1, J_pt,
            n=50, M=3, seed_base=88_000,
            label=label, h_ref=h_ref,
        )
        assert save_path.exists(), ".pkl should be written after 3 reps"
        # Simulate crash: remove final pkl but keep partial if it exists
        # (since M=CHECKPOINT_FREQ=3, partial is written then unlinked on success;
        #  we re-create it manually to simulate an interrupted run)
        save_path.unlink()
        # Write a fake partial with 3 records
        with open(partial, "wb") as f:
            pickle.dump(data_first["records"], f)

        # Phase 2 — re-run requesting 6 reps total: should resume from rep 3
        data_resumed = mod.run_curve_block(
            dgp1_095, factory, a_grid_dgp1, J_pt,
            n=50, M=6, seed_base=88_000,
            label=label, h_ref=h_ref,
        )

    finally:
        mod._RAW_DIR       = orig_raw
        mod.CHECKPOINT_FREQ = orig_freq

    records = data_resumed["records"]
    assert len(records) == 6, f"Expected 6 records, got {len(records)}"
    assert not partial.exists(), ".partial.pkl should be deleted after success"
    # Seeds should be 88_000 .. 88_005 in order
    seeds = [r["seed"] for r in records]
    assert seeds == list(range(88_000, 88_006)), f"Seeds mismatch: {seeds}"


# ══════════════════════════════════════════════════════════════════════════════
#  T_E1e — variance normalization: SE = sqrt(V_hat_grid / n), NOT sqrt(V_hat_grid)
# ══════════════════════════════════════════════════════════════════════════════

def test_T_E1e_variance_normalisation():
    """
    Critical normalization check.

    V_hat_grid[k] = mean((score_i - J_dr[k])²) is O(1) (sample variance of scores).
    The SE of the DR estimator J_dr[k] = (1/n) Σ score_i is:
        SE_k = sqrt(V_hat_grid[k] / n)
    not sqrt(V_hat_grid[k]).

    Build synthetic data with V_hat_grid = ones(K) and n = 100.
    summarize_exp1A must produce SE_grid ≈ 0.10 per dose (not 1.0).
    """
    from simulations.archive.utilities.exp1A_coverage import summarize_exp1A

    K = 3
    n = 100
    M_ok = 200
    rng = np.random.default_rng(0)

    # True policy values at 3 doses
    J_pt = np.array([0.3, 0.5, 0.7])

    # Synthetic scores: J_dr ~ N(J_pt, SE²) with SE = sqrt(1.0 / n) = 0.1
    # V_hat_grid = 1.0 for all doses (O(1) score variance)
    true_se = np.sqrt(1.0 / n)           # = 0.10
    J_dr_all = J_pt[None] + rng.normal(0, true_se, size=(M_ok, K))

    # psi_hat = J_dr at ref_idx = K//2 = 1
    psi_all = J_dr_all[:, K // 2]
    V_hat   = np.ones(M_ok)              # V_hat = 1.0 → SE = sqrt(1/100) = 0.10
    V_hat_grid_all = np.ones((M_ok, K)) # same for each dose

    # Build synthetic records matching the expected dict structure
    records = []
    for i in range(M_ok):
        records.append({
            "seed": i, "error": None,
            "psi_hat": float(psi_all[i]),
            "V_hat":   float(V_hat[i]),
            "J_dr":    J_dr_all[i].tolist(),
            "J_reg":   J_dr_all[i].tolist(),
            "J_policy_true_rep": J_pt.tolist(),
            "ESS_grid": [50.0] * K,
            "ESS_min": 50.0, "ESS_ratio_min": 0.5,
            "V_hat_grid":        V_hat_grid_all[i].tolist(),
            "V_reg_grid":        [np.nan] * K,
            "V_correction_grid": [np.nan] * K,
            "weight_p99_grid":   [np.nan] * K,
            "weight_max_grid":   [np.nan] * K,
            "q_clip_fraction":   np.nan,
            "riesz_residual_grid_mean": [np.nan] * K,
            "riesz_residual_grid_max":  [np.nan] * K,
            "cond_M_grid_mean":  [np.nan] * K,
            "cond_M_grid_max":   [np.nan] * K,
            "neg_share_grid_mean": [np.nan] * K,
            "neg_share_grid_max":  [np.nan] * K,
        })

    data = {
        "records": records,
        "meta": {
            "label": "test", "dgp": "Synthetic", "n": n, "M": M_ok,
            "seed_base": 0, "h_ref": 0.1,
            "a_grid": [-0.5, 0.0, 0.5],
            "J_policy_true": J_pt.tolist(),
            "J_true": J_pt.tolist(),
        },
    }

    summ = summarize_exp1A(data)

    # 1. Coverage should be close to 0.95 (±0.05 tolerance, M=200)
    cpd = summ["coverage_per_dose"]
    for d, c in enumerate(cpd):
        assert abs(c - 0.95) < 0.08, (
            f"Dose {d}: coverage_per_dose = {c:.3f}, expected ~0.95. "
            f"Likely /n missing in SE computation."
        )

    # 2. SER should be close to 1.0
    ser = summ["ser"]
    assert 0.80 < ser < 1.25, (
        f"SER = {ser:.3f}, expected ~1.0. "
        f"Check V_hat/n normalisation and SER = mean(se)/std(psi_hat) convention."
    )

    # 3. Explicit SE magnitude check: mean(se) ≈ 0.10 ± 0.02
    # SE = sqrt(V_hat / n) = sqrt(1.0 / 100) = 0.10
    # We compute se_all from scalar V_hat in summarize_exp1A
    # and per-dose SEs from V_hat_grid. Check via SER × std(psi):
    std_psi = float(np.std(psi_all))
    se_implied = ser * std_psi          # mean(se_hat) = SER × std(psi_hat)
    assert abs(se_implied - true_se) < 0.025, (
        f"Implied SE = {se_implied:.4f}, expected {true_se:.4f}. "
        f"V_hat_grid normalization is wrong."
    )
