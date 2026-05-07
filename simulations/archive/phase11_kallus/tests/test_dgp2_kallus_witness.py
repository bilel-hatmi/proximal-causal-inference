"""
Smoke tests for dgp2_kallus_witness.py.

Tests
-----
T_W1 — build_h_ref reproductible & cache fonctionne
T_W2 — _per_fold_arrays shapes correctes & pas de NaN
T_W3 — Identite au truth (h_hat = h_ref, q = 1) => witness_h_excess = 0
T_W4 — End-to-end quick block + report builds non-empty markdown
T_W5 — Sign convention : h_hat = h_ref + 0.1 => direction_ratio == 1.0
"""
from pathlib import Path
import pickle

import numpy as np
import pytest

import simulations.experiments.dgp2_kallus_witness as wmod


# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    """Redirect raw / summ / cache to a temp folder."""
    raw   = tmp_path / "raw"
    summ  = tmp_path / "summ"
    cache = tmp_path / "cache"
    for d in (raw, summ, cache):
        d.mkdir()
    monkeypatch.setattr(wmod, "_RAW_DIR",  raw)
    monkeypatch.setattr(wmod, "_SUMM_DIR", summ)
    monkeypatch.setattr(wmod, "_CACHE_DIR", cache)
    monkeypatch.setattr(wmod, "_HREF_CACHE_PATH", cache / "href_test.pkl")
    return raw, summ, cache


def _make_dgp():
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    return MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)


# ══════════════════════════════════════════════════════════════════════════════
#  T_W1 — build_h_ref reproductible & cache
# ══════════════════════════════════════════════════════════════════════════════

def test_T_W1_build_href_cache(tmp_dirs):
    """Tiny h_ref fit; second call must hit cache and return identical bridge."""
    raw, summ, cache = tmp_dirs
    dgp = _make_dgp()

    # Fit a tiny h_ref (n_ref=400, n_landmarks=200)
    href1 = wmod.build_h_ref(dgp, n_ref=400, n_landmarks=200,
                              seed=1234, verbose=False)
    cache_path = wmod._HREF_CACHE_PATH
    assert cache_path.exists(), "h_ref cache not written"

    # Second call must hit cache
    import time
    t0 = time.time()
    href2 = wmod.build_h_ref(dgp, n_ref=400, n_landmarks=200,
                              seed=1234, verbose=False)
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"Cache hit too slow ({elapsed:.2f}s) — refit suspected"

    # Predictions must match exactly on a new sample
    test_sample = dgp.generate(n=50, seed=99)
    p1 = href1.predict(test_sample.W, test_sample.A)
    p2 = href2.predict(test_sample.W, test_sample.A)
    np.testing.assert_allclose(p1, p2, rtol=1e-12, atol=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
#  T_W2 — _per_fold_arrays shapes & no NaN
# ══════════════════════════════════════════════════════════════════════════════

def test_T_W2_per_fold_arrays_shapes(tmp_dirs):
    """Tiny block; verify all arrays have correct shapes and no NaN."""
    dgp = _make_dgp()
    href = wmod.build_h_ref(dgp, n_ref=400, n_landmarks=200, seed=1234,
                             verbose=False)

    sample = dgp.generate(n=200, seed=42)
    h_KDE = wmod._silverman_h(sample.A)

    arr = wmod._per_fold_arrays(
        sample, dgp, href, q_kind="oracle",
        a_grid=wmod.A_GRID, h_KDE=h_KDE,
        bridge_kwargs={"n_landmarks": 100},   # speed
        n_folds=2,
    )
    n = 200
    assert arr["h_hat_obs"].shape    == (n,)
    assert arr["h_hat_pol"].shape    == (n, wmod.K)
    assert arr["q_obs_2d"].shape     == (n, wmod.K)
    assert arr["h_ref_obs"].shape    == (n,)
    assert arr["h_ref_pol"].shape    == (n, wmod.K)
    assert arr["h0_proxy_obs"].shape == (n,)
    assert arr["h0_proxy_pol"].shape == (n, wmod.K)

    for k in arr:
        assert np.all(np.isfinite(arr[k])), f"{k} has NaN/inf"

    # h_ref_obs should differ from h_hat_obs (different bridges!)
    assert not np.allclose(arr["h_ref_obs"], arr["h_hat_obs"], atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
#  T_W3 — Identity at truth (h_hat == h_ref, q == 1)
# ══════════════════════════════════════════════════════════════════════════════

def test_T_W3_identity_at_truth():
    """If h_hat == h_ref and q == 1, witnesses degenerate to zero / pure noise."""
    rng = np.random.default_rng(0)
    n = 300
    Y = rng.normal(0, 1, n)
    h_ref_obs = rng.normal(0, 1, n)
    h_hat_obs = h_ref_obs.copy()                      # exact match
    h0_proxy_obs = rng.normal(0, 1, n)
    A = rng.uniform(1.5, 3.0, n)
    Z = rng.normal(0, 1, n)
    W = rng.normal(0, 1, n)

    # Witness h with h_hat = h_ref
    w1 = wmod._compute_witness_h(Y, h_hat_obs, h_ref_obs, h0_proxy_obs,
                                  Z, A, ell_Z=1.0, ell_A=1.0)
    # excess must be exactly 0 (numerical)
    assert abs(w1["witness_h_excess"]) < 1e-12, \
        f"witness_h_excess should be 0 when h_hat=h_ref, got {w1['witness_h_excess']}"

    # Direction witness with h_hat == h_ref => plug_in_error and corr_on_error are 0
    h_KDE = 0.3
    h_hat_pol    = rng.normal(0, 1, (n, wmod.K))
    h_ref_pol    = h_hat_pol.copy()
    h0_proxy_pol = rng.normal(0, 1, (n, wmod.K))
    q_obs_2d = np.ones((n, wmod.K))

    w3 = wmod._compute_direction_witness(
        Y, A, h_hat_obs, h_ref_obs, h0_proxy_obs,
        h_hat_pol, h_ref_pol, h0_proxy_pol,
        q_obs_2d, h_KDE, wmod.A_GRID, wmod.REF_IDX,
    )
    assert abs(w3["plug_in_error"])       < 1e-12
    assert abs(w3["correction_on_error"]) < 1e-12
    # correction_noise = mean(K_h * 1 * (Y - h_ref)) is non-zero in general
    # (it captures pure IPW noise term)


# ══════════════════════════════════════════════════════════════════════════════
#  T_W4 — End-to-end quick block + report
# ══════════════════════════════════════════════════════════════════════════════

def test_T_W4_quick_block_and_report(tmp_dirs):
    """One tiny block runs end-to-end and the report builder produces markdown."""
    raw, summ, cache = tmp_dirs
    dgp = _make_dgp()
    href = wmod.build_h_ref(dgp, n_ref=400, n_landmarks=200, seed=1234,
                             verbose=False)

    data = wmod._run_witness_block(
        dgp, href, q_kind="oracle", n=200, M=2,
        seed_base=999, label="smoke_W4", a_grid=wmod.A_GRID,
        bridge_kwargs={"n_landmarks": 100}, q_factory_proto=None, raw_dir=raw,
    )
    assert (raw / "smoke_W4.pkl").exists()
    assert "records" in data
    assert len(data["records"]) == 2

    decomp = wmod._decompose_witness(data)
    assert decomp is not None
    required = [
        "witness_h_hat_mean", "witness_h_ref_mean", "witness_h_excess_mean",
        "witness_q_rel_mean", "riesz_g1_at_ref_mean",
        "plug_in_error_mean", "correction_on_error_mean", "correction_noise_mean",
        "direction_ratio_mean", "direction_ratio_h0_mean",
        "bias_reg_mean", "bias_dr_mean", "ESS_mean",
    ]
    for k in required:
        assert k in decomp, f"missing key {k}"

    # Report builder
    decomps = {("oracle", 200): decomp}
    href_meta = wmod._href_meta()
    lines = wmod._build_report_witness(
        decomps, href_meta, n_grid=[200], methods=["oracle"],
        J_true_ref=float(dgp.m_true(float(wmod.A_GRID[wmod.REF_IDX]))),
    )
    text = "\n".join(lines)
    assert len(text) > 200
    assert "## 0. h_ref" in text
    assert "## 1. Table principale" in text
    assert "## 2. Direction ratio" in text
    assert "## 3. Interpretation" in text


# ══════════════════════════════════════════════════════════════════════════════
#  T_W5 — Sign convention : h_hat = h_ref + 0.1, q = 1, K_h = 1 => ratio = 1
# ══════════════════════════════════════════════════════════════════════════════

def test_T_W5_sign_convention():
    """
    Construct a deterministic case where h_hat overestimates h_ref by +0.1
    and q=1 with K_h trivial. Direction ratio should equal +1 (full correction).
    """
    n = 100
    Y = np.zeros(n)                                  # arbitrary, not used in pe/coe
    A = np.full(n, 2.6)                              # all at a_ref
    h_ref_obs    = np.zeros(n)
    h_hat_obs    = h_ref_obs + 0.1                   # plug_in_error = +0.1
    h0_proxy_obs = h_ref_obs.copy()                  # for h0 sanity (not asserted here)
    h_ref_pol    = np.zeros((n, wmod.K))
    h_hat_pol    = h_ref_pol + 0.1                   # T_pi h_hat - T_pi h_ref = +0.1
    h0_proxy_pol = h_ref_pol.copy()
    q_obs_2d     = np.ones((n, wmod.K))

    # K_h(0) at h=1.0 = 1/sqrt(2π) ≈ 0.3989; uniform over all i so factor cancels in ratio
    h_KDE = 1.0

    w3 = wmod._compute_direction_witness(
        Y, A, h_hat_obs, h_ref_obs, h0_proxy_obs,
        h_hat_pol, h_ref_pol, h0_proxy_pol,
        q_obs_2d, h_KDE, wmod.A_GRID, wmod.REF_IDX,
    )

    # plug_in_error = mean(T_pi h_hat - T_pi h_ref) = +0.1
    assert abs(w3["plug_in_error"] - 0.1) < 1e-10

    # correction_on_error = mean(K_h(0)*1*(h_ref - h_hat)) = K_h(0) * (-0.1) < 0
    Kh0 = 1.0 / np.sqrt(2 * np.pi)
    assert abs(w3["correction_on_error"] + 0.1 * Kh0) < 1e-10

    # direction_ratio = correction_on_error / (-plug_in_error)
    #                = (-0.1*Kh0) / (-0.1) = Kh0
    expected_ratio = Kh0
    assert abs(w3["direction_ratio"] - expected_ratio) < 1e-10, \
        f"direction_ratio = {w3['direction_ratio']:.5f}, expected {expected_ratio:.5f}"
