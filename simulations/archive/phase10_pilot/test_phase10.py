"""
T_P10 — Smoke tests for Phase 10 pilot (phase10_pilot.py).

Tests use tiny settings (n=50, M=3) to verify the pipeline runs end-to-end
without errors. Correctness is validated by the full pilot (M=50), not here.

Run:
    pytest simulations/tests/test_phase10.py -v
"""
from __future__ import annotations

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dgp1():
    from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP
    return CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)


@pytest.fixture(scope="module")
def dgp2():
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    return MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)


@pytest.fixture(scope="module")
def a_grid_dgp1():
    return np.array([-0.5, 0.0, 0.5])


@pytest.fixture(scope="module")
def a_grid_dgp2_in():
    return np.array([1.8, 2.2, 2.6, 3.0])


# ── T_P10a — Scalar runner smoke test ─────────────────────────────────────────

def test_p10a_scalar_smoke(dgp1):
    """T_P10a — Oracle n=50, M=3 terminates without error, metrics non-NaN."""
    from simulations.experiments.phase10_pilot import run_scalar_mc
    from simulations.methods.oracle import OracleDirect

    res = run_scalar_mc(
        dgp1, OracleDirect,
        n=50, M=3, seed_base=9990,
        label="smoke_oracle_dgp1",
    )
    assert "metrics" in res
    assert np.isfinite(res["metrics"]["rmse"]), "RMSE is NaN"
    assert np.isfinite(res["metrics"]["bias"]), "bias is NaN"
    assert res["M"] == 3


# ── T_P10b — Curve runner: DRKernel oracle-q smoke ───────────────────────────

def test_p10b_curve_oracle_smoke(dgp1, a_grid_dgp1):
    """T_P10b — DRKernel oracle-q n=50, M=3 runs, at least 1 rep succeeds."""
    from simulations.experiments.phase10_pilot import (
        _compute_J_policy_true,
        _silverman_h,
        make_dr_kernel_oracle,
        run_curve_mc,
        summarize_curve,
    )

    ref = dgp1.generate(n=50, seed=0)
    h_ref = _silverman_h(ref.A)
    J_pt  = _compute_J_policy_true(dgp1, a_grid_dgp1, h_ref)

    data = run_curve_mc(
        dgp1,
        make_dr_kernel_oracle(dgp1, a_grid_dgp1, h_ref),
        a_grid_dgp1, J_pt,
        n=50, M=3, seed_base=9991,
        label="smoke_drk_oracle_dgp1", h_ref=h_ref,
    )
    summ = summarize_curve(data)
    assert summ["M_ok"] >= 1, f"All 3 reps failed: {[r['error'] for r in data['records']]}"


# ── T_P10c — Curve runner: DRKernel estimated-q smoke ────────────────────────

def test_p10c_curve_estimated_smoke(dgp1, a_grid_dgp1):
    """T_P10c — DRKernel estimated-q (cross-fit) n=50, M=3 runs, at least 1 rep succeeds."""
    from simulations.experiments.phase10_pilot import (
        _compute_J_policy_true,
        _silverman_h,
        make_dr_kernel_estimated,
        run_curve_mc,
        summarize_curve,
    )

    ref = dgp1.generate(n=50, seed=0)
    h_ref = _silverman_h(ref.A)
    J_pt  = _compute_J_policy_true(dgp1, a_grid_dgp1, h_ref)

    data = run_curve_mc(
        dgp1,
        make_dr_kernel_estimated(dgp1, a_grid_dgp1, h_ref),
        a_grid_dgp1, J_pt,
        n=50, M=3, seed_base=9992,
        label="smoke_drk_xfit_dgp1", h_ref=h_ref,
    )
    summ = summarize_curve(data)
    assert summ["M_ok"] >= 1, f"All 3 reps failed: {[r['error'] for r in data['records']]}"


# ── T_P10d — Curve runner: DRDoseResponse oracle smoke ───────────────────────

def test_p10d_curve_dr_dose_smoke(dgp1, a_grid_dgp1):
    """T_P10d — DRDoseResponse oracle n=50, M=3, at least 1 rep succeeds."""
    from simulations.experiments.phase10_pilot import (
        _compute_J_policy_true,
        _silverman_h,
        make_dr_dose_response,
        run_curve_mc,
        summarize_curve,
    )

    ref = dgp1.generate(n=50, seed=0)
    h_ref = _silverman_h(ref.A)
    J_pt  = _compute_J_policy_true(dgp1, a_grid_dgp1, h_ref)

    data = run_curve_mc(
        dgp1,
        make_dr_dose_response(dgp1, a_grid_dgp1, h_ref, is_linear_in_A=True),
        a_grid_dgp1, J_pt,
        n=50, M=3, seed_base=9993,
        label="smoke_dr_dose_dgp1", h_ref=h_ref,
    )
    summ = summarize_curve(data)
    assert summ["M_ok"] >= 1, f"All 3 reps failed: {[r['error'] for r in data['records']]}"


# ── T_P10e — J_policy_true cross-check (DGP1, linear m → J_pt ≈ J_true) ─────

def test_p10e_J_policy_true_dgp1(dgp1, a_grid_dgp1):
    """T_P10e — For DGP1 (linear m), J_policy_true ≈ m(a) (Jensen gap ≈ 0)."""
    from simulations.experiments.phase10_pilot import (
        _compute_J_policy_true,
        _silverman_h,
    )

    ref = dgp1.generate(n=1000, seed=0)
    h_ref = _silverman_h(ref.A)
    J_pt  = _compute_J_policy_true(dgp1, a_grid_dgp1, h_ref)
    J_true = np.array([dgp1.m_true(float(a)) for a in a_grid_dgp1])

    err = np.max(np.abs(J_pt - J_true))
    assert err < 1e-6, (
        f"DGP1 (linear m): J_policy_true should = J_true, got max_err={err:.2e}\n"
        f"J_pt={J_pt}, J_true={J_true}"
    )


# ── T_P10f — J_policy_true cross-check (DGP2, concave m → J_pt < J_true) ────

def test_p10f_J_policy_true_dgp2(dgp2, a_grid_dgp2_in):
    """T_P10f — For DGP2 (concave m), J_policy_true < m(a) (Jensen gap > 0)."""
    from simulations.experiments.phase10_pilot import (
        _compute_J_policy_true,
        _silverman_h,
    )

    ref = dgp2.generate(n=1000, seed=0)
    h_ref = _silverman_h(ref.A)
    J_pt  = _compute_J_policy_true(dgp2, a_grid_dgp2_in, h_ref)
    J_true = np.array([dgp2.m_true(float(a)) for a in a_grid_dgp2_in])

    jensen_gap = J_true - J_pt   # should be > 0 for concave m
    assert np.all(jensen_gap >= -1e-9), (
        f"DGP2 (concave m): J_policy_true should be ≤ J_true everywhere.\n"
        f"Jensen gap: {jensen_gap}"
    )
    # At least one dose should have a non-trivial gap (concavity effect)
    assert np.max(jensen_gap) > 1e-4, (
        f"DGP2 Jensen gap is near-zero everywhere ({np.max(jensen_gap):.2e}), "
        "expected positive for concave Michaelis-Menten"
    )


# ── T_P10g — Summarize curve handles zero-success reps gracefully ─────────────

def test_p10g_summarize_zero_ok():
    """T_P10g — summarize_curve with M_ok=0 returns M_ok=0 without crashing."""
    from simulations.experiments.phase10_pilot import summarize_curve

    data = {
        "records": [{"error": "oops"}, {"error": "oops"}],
        "meta": {"J_policy_true": [0.5, 0.6, 0.7], "J_true": [0.5, 0.6, 0.7]},
    }
    summ = summarize_curve(data)
    assert summ["M_ok"] == 0
    assert summ["n_errors"] == 2
