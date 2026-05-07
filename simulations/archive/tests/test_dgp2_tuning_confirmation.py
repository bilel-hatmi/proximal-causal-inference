"""
Smoke tests for dgp2_tuning_confirmation.py.

Tests
-----
T1 — make_factory introspection : 5 configs map to correct DRKernel hyperparameters
T2 — End-to-end smoke (config=baseline, n=200, M=2) : pkl saved, _decompose works
T3 — J_policy_true differs between Baseline (h_ref) and Config C (1.5*h_ref)
"""
from __future__ import annotations

import numpy as np
import pytest

import simulations.archive.utilities.dgp2_tuning_confirmation as mod


# ── Fixture : redirect output dirs to tmp_path ────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    summ = tmp_path / "summ"
    raw.mkdir()
    summ.mkdir()
    monkeypatch.setattr(mod, "_RAW_DIR", raw)
    monkeypatch.setattr(mod, "_SUMM_DIR", summ)
    monkeypatch.setattr(mod, "_REPORT_PATH", summ / "dgp2_tuning_confirmation.md")
    return raw, summ


def _make_dgp():
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    return MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)


# ══════════════════════════════════════════════════════════════════════════════
#  T1 — make_factory introspection
# ══════════════════════════════════════════════════════════════════════════════

def test_T1_make_factory_per_config_hyperparameters():
    """Each config must produce a DRKernel with the right q_model and bridge_kwargs."""
    a_grid = np.array([1.8, 2.2, 2.6, 3.0])
    h_ref = 0.5
    ell_W0 = 1.0
    ell_A0 = 1.0
    ell_Z0 = 1.0

    expected = {
        "baseline": dict(
            lambda_Q=1e-3, clip=None, h_KDE=0.5, bandwidth=0.5,
            has_bridge_lambda=False, has_bridge_ell=False,
        ),
        "A": dict(
            lambda_Q=1e-3, clip=None, h_KDE=0.5, bandwidth=0.5,
            has_bridge_lambda=True, lambda_1=1e-4, lambda_2=1e-4,
            has_bridge_ell=True, ell_W=2.0,
        ),
        "B": dict(
            lambda_Q=1e-2, clip=20.0, h_KDE=0.5, bandwidth=0.5,
            has_bridge_lambda=False, has_bridge_ell=False,
        ),
        "C": dict(
            lambda_Q=1e-3, clip=None, h_KDE=0.75, bandwidth=0.75,
            has_bridge_lambda=False, has_bridge_ell=False,
        ),
        "AC": dict(
            lambda_Q=1e-3, clip=None, h_KDE=0.75, bandwidth=0.75,
            has_bridge_lambda=True, lambda_1=1e-4, lambda_2=1e-4,
            has_bridge_ell=True, ell_W=2.0,
        ),
    }

    for name, exp in expected.items():
        spec = mod.CONFIGS[name]
        factory = mod.make_factory(spec, a_grid, h_ref, ell_W0, ell_A0, ell_Z0)
        est = factory()

        # q_model checks
        assert est.q_model.lambda_Q == pytest.approx(exp["lambda_Q"]), \
            f"{name}: lambda_Q mismatch"
        assert est.q_model.clip == exp["clip"], f"{name}: clip mismatch"
        assert est.q_model.h_KDE == pytest.approx(exp["h_KDE"]), \
            f"{name}: h_KDE mismatch"
        # compute_cond default = False (production-safe)
        assert est.q_model.compute_cond is False, f"{name}: compute_cond must be False"

        # DRKernel.bandwidth (must equal h_KDE)
        assert est.bandwidth == pytest.approx(exp["bandwidth"]), \
            f"{name}: bandwidth mismatch"

        # Cross-fit q must be enabled
        assert est.cross_fit_q is True, f"{name}: cross_fit_q must be True"

        # bridge_kwargs checks
        bk = est.bridge_kwargs or {}
        if exp["has_bridge_lambda"]:
            assert "lambda_1" in bk, f"{name}: bridge_kwargs missing lambda_1"
            assert bk["lambda_1"] == pytest.approx(exp["lambda_1"])
            assert bk["lambda_2"] == pytest.approx(exp["lambda_2"])
        else:
            assert "lambda_1" not in bk, f"{name}: bridge_kwargs should not have lambda_1"

        if exp["has_bridge_ell"]:
            assert "ell_W" in bk, f"{name}: bridge_kwargs missing ell_W"
            assert bk["ell_W"] == pytest.approx(exp["ell_W"])
        else:
            assert "ell_W" not in bk, f"{name}: bridge_kwargs should not have ell_W"


# ══════════════════════════════════════════════════════════════════════════════
#  T2 — End-to-end smoke
# ══════════════════════════════════════════════════════════════════════════════

def test_T2_run_one_block_smoke(tmp_dirs):
    """_run_one_block at n=200, M=2 should produce a valid pkl + decomp."""
    raw, _ = tmp_dirs
    dgp = _make_dgp()

    decomp, wall, data = mod._run_one_block(
        dgp, config_name="baseline", n=200, M=2, force=False,
    )

    # Pkl saved
    label = mod._label("baseline", 200, 2)
    pkl_path = raw / f"{label}.pkl"
    assert pkl_path.exists(), f"pkl not created at {pkl_path}"

    # decomp returned with required keys
    assert decomp is not None, "decomp should not be None"
    for key in ("bias_reg", "bias_dr", "coverage_ref", "ess_min_mean", "SE_grid"):
        assert key in decomp, f"missing key {key}"

    # Coverage in [0, 1]
    assert 0.0 <= decomp["coverage_ref"] <= 1.0

    # bias arrays are length K=4
    assert len(decomp["bias_reg"]) == mod.K
    assert len(decomp["bias_dr"]) == mod.K

    # Wall-clock is positive
    assert wall > 0

    # data has expected structure
    assert "records" in data and "meta" in data
    assert data["meta"]["n"] == 200
    assert data["meta"]["M"] == 2


# ══════════════════════════════════════════════════════════════════════════════
#  T3 — J_policy_true differs between Baseline and Config C
# ══════════════════════════════════════════════════════════════════════════════

def test_T3_J_policy_true_differs_baseline_vs_C():
    """Config C uses h_policy=1.5*h_ref → J_policy_true must differ from baseline."""
    dgp = _make_dgp()
    sample = dgp.generate(n=500, seed=0)
    h_ref = mod._silverman_h(sample.A)

    J_pt_baseline = mod._compute_J_policy_true(dgp, mod.A_GRID, h_ref)
    J_pt_C = mod._compute_J_policy_true(dgp, mod.A_GRID, h_ref * 1.5)

    # Must differ (Michaelis-Menten is concave → larger h shifts J downward)
    assert not np.allclose(J_pt_baseline, J_pt_C, atol=1e-6), \
        "J_policy_true should differ with different h_policy"

    # Quantitative sanity: relative difference at REF dose > 0.1%
    rel = abs(J_pt_C[mod.REF_IDX] - J_pt_baseline[mod.REF_IDX]) / abs(J_pt_baseline[mod.REF_IDX])
    assert rel > 1e-3, \
        f"Relative J_pt difference at REF too small: {rel:.2e} (expected > 0.001)"

    # h_policy_scale = 1.0 reproduces baseline
    spec_baseline = mod.CONFIGS["baseline"]
    spec_C = mod.CONFIGS["C"]
    assert spec_baseline["h_policy_scale"] == 1.0
    assert spec_C["h_policy_scale"] == 1.5
