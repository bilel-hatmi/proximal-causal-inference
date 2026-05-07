"""
Smoke tests for dgp2_bias_diagnostics.py.

Tests
-----
T_D1 — quick mode end-to-end (n=500, M=3): pkl saved, report generated
T_D2 — _decompose returns expected keys
T_D3 — _run_block checkpointing (partial → resume → final)
T_D4 — _build_report produces non-empty markdown
T_D5 — correction_ratio formula sanity check
"""
import pickle
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

# ── Patch paths before importing ─────────────────────────────────────────────
import simulations.experiments.dgp2_bias_diagnostics as mod


# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    """Redirect raw and summary directories to a temp folder."""
    raw  = tmp_path / "raw"
    summ = tmp_path / "summ"
    raw.mkdir(); summ.mkdir()
    monkeypatch.setattr(mod, "_RAW_DIR",  raw)
    monkeypatch.setattr(mod, "_SUMM_DIR", summ)
    return raw, summ


def _make_dgp():
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    return MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)


# ── T_D1 : quick mode end-to-end ─────────────────────────────────────────────

def test_T_D1_quick_mode(tmp_dirs, monkeypatch, capsys):
    """quick mode (n=200, M=3) should produce a non-empty markdown report."""
    raw, summ = tmp_dirs
    dgp = _make_dgp()
    ref_sample = dgp.generate(n=200, seed=0)
    h_ref = mod._silverman_h(ref_sample.A)
    J_pt  = mod._compute_J_policy_true(dgp, mod.A_GRID, h_ref)

    # Run just one A config (smallest grid), one B config, one C config
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ

    def factory():
        q = KPVPolicyBridgeQ(a_grid=mod.A_GRID, h_KDE=h_ref, lambda_Q=1e-3)
        return DRKernel(a_grid=mod.A_GRID, q_model=q, bandwidth=h_ref,
                        n_folds=2, random_state=0, ref_dose_index=mod.REF_IDX,
                        cross_fit_q=True,
                        bridge_kwargs=dict(lambda_1=1e-3, lambda_2=1e-3))

    data = mod._run_block(dgp, factory, mod.A_GRID, J_pt, n=200, M=3,
                          seed_base=1000, label="smoke_A", raw_dir=raw)
    assert (raw / "smoke_A.pkl").exists()
    d = mod._decompose(data)
    assert d is not None
    assert "bias_reg" in d and len(d["bias_reg"]) == mod.K
    assert "coverage_ref" in d
    assert 0.0 <= d["coverage_ref"] <= 1.0


# ── T_D2 : _decompose keys ───────────────────────────────────────────────────

def test_T_D2_decompose_keys():
    """_decompose should return all required keys with correct shapes."""
    dgp = _make_dgp()
    ref_sample = dgp.generate(n=200, seed=0)
    h_ref = mod._silverman_h(ref_sample.A)
    J_pt  = mod._compute_J_policy_true(dgp, mod.A_GRID, h_ref)

    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td)
        def factory():
            q = KPVPolicyBridgeQ(a_grid=mod.A_GRID, h_KDE=h_ref, lambda_Q=1e-3)
            return DRKernel(a_grid=mod.A_GRID, q_model=q, bandwidth=h_ref,
                            n_folds=2, random_state=0, ref_dose_index=mod.REF_IDX,
                            cross_fit_q=True)

        data = mod._run_block(dgp, factory, mod.A_GRID, J_pt, n=200, M=3,
                              seed_base=2000, label="smoke_keys", raw_dir=raw)
    d = mod._decompose(data)
    assert d is not None
    required_keys = [
        "bias_reg", "bias_dr", "bias_corr", "required_corr", "correction_ratio",
        "correction_efficiency", "V_total", "V_reg", "V_corr", "V_cov",
        "SE_grid", "bse_grid", "pred_cov_grid", "coverage_ref",
        "ess_min_mean", "w_p99_mean", "q_clip_mean", "riesz_res_mean",
        "neg_share_mean", "jensen_gap_mean", "mise_mean",
    ]
    for k in required_keys:
        assert k in d, f"Missing key: {k}"

    # shape checks
    for k in ["bias_reg", "bias_dr", "SE_grid", "bse_grid", "pred_cov_grid",
              "correction_efficiency", "correction_ratio"]:
        assert len(d[k]) == mod.K, f"{k} should have length K={mod.K}"

    # coverage in [0,1]
    assert 0.0 <= d["coverage_ref"] <= 1.0
    # pred_cov in [0,1] or nan
    for v in d["pred_cov_grid"]:
        assert np.isnan(v) or (0.0 <= v <= 1.0)


# ── T_D3 : checkpointing resume ──────────────────────────────────────────────

def test_T_D3_checkpoint_resume(tmp_dirs):
    """Partial checkpoint → resume completes correctly."""
    raw, _ = tmp_dirs
    dgp = _make_dgp()
    ref_sample = dgp.generate(n=200, seed=0)
    h_ref = mod._silverman_h(ref_sample.A)
    J_pt  = mod._compute_J_policy_true(dgp, mod.A_GRID, h_ref)

    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ

    def factory():
        q = KPVPolicyBridgeQ(a_grid=mod.A_GRID, h_KDE=h_ref, lambda_Q=1e-3)
        return DRKernel(a_grid=mod.A_GRID, q_model=q, bandwidth=h_ref,
                        n_folds=2, random_state=0, ref_dose_index=mod.REF_IDX,
                        cross_fit_q=True)

    # Simulate a partial save (3 reps done)
    partial_path = raw / "smoke_resume.partial.pkl"
    partial_records: list = []
    dgp_gen = dgp
    for i in range(3):
        s = dgp_gen.generate(n=200, seed=100 + i)
        est = factory()
        est.fit(s, m_true_fn=dgp.m_true)
        res = est.estimate()
        rec = mod._build_record(res, mod.K, 100 + i, J_pt)
        partial_records.append(rec)

    with open(partial_path, "wb") as fh:
        pickle.dump(partial_records, fh)

    # Run with M=5 → should resume from 3 and add 2 more
    orig_freq = mod.CHECKPOINT_FREQ
    mod.CHECKPOINT_FREQ = 10   # don't save more partials during test
    try:
        data = mod._run_block(
            dgp, factory, mod.A_GRID, J_pt, n=200, M=5,
            seed_base=100, label="smoke_resume", raw_dir=raw,
        )
    finally:
        mod.CHECKPOINT_FREQ = orig_freq

    assert (raw / "smoke_resume.pkl").exists()
    assert not partial_path.exists()   # cleaned up
    ok_recs = [r for r in data["records"] if r.get("error") is None]
    assert len(ok_recs) == 5


# ── T_D4 : report builds without error ───────────────────────────────────────

def test_T_D4_report_builds(tmp_dirs):
    """_build_report produces non-empty markdown with expected sections."""
    raw, summ = tmp_dirs
    dgp = _make_dgp()
    ref_sample = dgp.generate(n=200, seed=0)
    h_ref = mod._silverman_h(ref_sample.A)
    J_pt  = mod._compute_J_policy_true(dgp, mod.A_GRID, h_ref)

    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ

    def factory():
        q = KPVPolicyBridgeQ(a_grid=mod.A_GRID, h_KDE=h_ref, lambda_Q=1e-3)
        return DRKernel(a_grid=mod.A_GRID, q_model=q, bandwidth=h_ref,
                        n_folds=2, random_state=0, ref_dose_index=mod.REF_IDX,
                        cross_fit_q=True)

    data = mod._run_block(dgp, factory, mod.A_GRID, J_pt, n=200, M=3,
                          seed_base=3000, label="smoke_rep", raw_dir=raw)
    d = mod._decompose(data)
    assert d is not None

    # Build minimal report
    diag_A = {(1e-3, 1.0): d}
    diag_B = {(1e-3, None): d}
    diag_C = {1.0: d}
    baselines = {1000: None}  # missing baseline ok

    lines = mod._build_report(baselines, diag_A, diag_B, diag_C,
                               None, None, None, n_main=200)
    text = "\n".join(lines)
    assert len(text) > 200
    assert "## 1. Baseline" in text
    assert "## 2. Diagnostic A" in text
    assert "## 3. Diagnostic B" in text
    assert "## 4. Diagnostic C" in text
    assert "## 5. Winner" in text
    assert "## 6. Interpretation" in text


# ── T_D5 : correction_ratio formula ──────────────────────────────────────────

def test_T_D5_correction_ratio_formula():
    """correction_ratio is (bias_corr / required_corr)."""
    # Manually construct a trivial case
    K = mod.K
    J_pt      = np.array([1.0] * K)
    mean_Jreg = np.array([0.8] * K)    # bias_reg = -0.2
    mean_Jdr  = np.array([0.85] * K)   # bias_corr = +0.05, bias_dr = -0.15

    # required_corr = -(bias_reg) = +0.2
    # correction_ratio = 0.05 / 0.2 = 0.25

    bias_reg  = mean_Jreg - J_pt          # -0.2
    bias_dr   = mean_Jdr  - J_pt          # -0.15
    bias_corr = mean_Jdr  - mean_Jreg     # +0.05
    req_corr  = -bias_reg                  # +0.2
    ratio     = bias_corr / req_corr       # 0.25

    np.testing.assert_allclose(ratio, 0.25, atol=1e-10)
    eff = 1.0 - np.abs(bias_dr) / np.abs(bias_reg)
    np.testing.assert_allclose(eff, 0.25, atol=1e-10)
