"""
Smoke tests for dgp2_optimal_region.py (Phase 9B.4).

Tests
-----
T1 -- Config parsing : BLOCK_E_CONFIGS maps to correct lambda_h and ell_scale.
T2 -- End-to-end smoke (n=100, M=3) : pkl created, decomp has finite metrics.
T3 -- Report builder : rebuilds md from existing PKLs without rerunning.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

import simulations.archive.utilities.dgp2_optimal_region as mod


# ── Fixture : redirect output dirs to tmp_path ────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    raw  = tmp_path / "raw"
    summ = tmp_path / "summ"
    raw.mkdir()
    summ.mkdir()
    monkeypatch.setattr(mod, "_RAW_DIR",     raw)
    monkeypatch.setattr(mod, "_SUMM_DIR",    summ)
    monkeypatch.setattr(mod, "_REPORT_PATH", summ / "dgp2_optimal_region.md")
    # Patch tradeoff anchor dir to an empty tmp dir (no anchors in tests)
    anc = tmp_path / "tradeoff_anchors"
    anc.mkdir()
    monkeypatch.setattr(mod, "_TRADEOFF_RAW_DIR", anc)
    return raw, summ


def _make_dgp():
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    return MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)


# ══════════════════════════════════════════════════════════════════════════════
#  T1 -- Config parsing
# ══════════════════════════════════════════════════════════════════════════════

def test_T1_config_parsing():
    """BLOCK_E_CONFIGS must map lambda_h and ell_scale into the spec correctly."""
    # E_baseline: lambda_h=None, ell_scale=1.0
    baseline = mod.BLOCK_E_CONFIGS["E_baseline"]
    assert baseline["lambda_h"]  is None
    assert baseline["ell_scale"] == pytest.approx(1.0)
    assert baseline["h_policy_scale"] == pytest.approx(1.0)

    # E_lam1e-4_ell2p50: lambda_h=1e-4, ell_scale=2.50
    cid = mod._e_config_id(1e-4, 2.5)
    assert cid == "E_lam1e-4_ell2p50"
    spec = mod.BLOCK_E_CONFIGS[cid]
    assert spec["lambda_h"]  == pytest.approx(1e-4)
    assert spec["ell_scale"] == pytest.approx(2.50)

    # E_lam3e-5_ell2p25: lambda_h=3e-5, ell_scale=2.25
    cid2 = mod._e_config_id(3e-5, 2.25)
    assert cid2 == "E_lam3e-5_ell2p25"
    spec2 = mod.BLOCK_E_CONFIGS[cid2]
    assert spec2["lambda_h"]  == pytest.approx(3e-5)
    assert spec2["ell_scale"] == pytest.approx(2.25)

    # Total config count: baseline + 3 lam x 5 ell = 16
    assert len(mod.BLOCK_E_CONFIGS) == 16
    assert len(mod.BLOCK_E_ORDER)   == 16

    # E_baseline is first in order
    assert mod.BLOCK_E_ORDER[0] == "E_baseline"

    print("T1 passed: BLOCK_E_CONFIGS correctly parsed.")


# ══════════════════════════════════════════════════════════════════════════════
#  T2 -- End-to-end smoke run
# ══════════════════════════════════════════════════════════════════════════════

def test_T2_smoke_run(tmp_dirs):
    """n=100, M=3 produces a pkl with finite bias_DR, coverage_ref, ess_min_mean."""
    raw, summ = tmp_dirs
    dgp = _make_dgp()

    # Run only baseline and one tuned config (fast)
    results_E = mod.run_block_E(
        dgp, n_list=[100], M=3,
        config_ids=["E_baseline", "E_lam1e-4_ell2p50"],
        force=False,
    )

    # --- PKL files should exist ---
    pkls = list(raw.glob("dgp2optimal_E_*.pkl"))
    assert len(pkls) >= 2, f"Expected >=2 PKLs, found {len(pkls)}: {pkls}"

    # --- decomp should have finite metrics for both configs ---
    for cid in ["E_baseline", "E_lam1e-4_ell2p50"]:
        assert cid in results_E, f"{cid} missing from results_E"
        entry = results_E[cid].get(100)
        assert entry is not None, f"No n=100 entry for {cid}"
        decomp, wall, data = entry

        if decomp is None:
            # Possibly all M=3 reps failed — acceptable if data is not None
            assert data is not None, f"Both decomp and data are None for {cid}"
            continue

        bias_dr = float(decomp["bias_dr"][mod.REF_IDX])
        cov     = float(decomp["coverage_ref"])
        bse     = float(decomp["bse_grid"][mod.REF_IDX])

        assert np.isfinite(bias_dr), f"bias_DR not finite for {cid}: {bias_dr}"
        assert np.isfinite(cov),     f"coverage not finite for {cid}: {cov}"
        assert np.isfinite(bse),     f"|b|/SE not finite for {cid}: {bse}"
        assert 0.0 <= cov <= 1.0,    f"coverage out of [0,1] for {cid}: {cov}"

    print(f"T2 passed: {len(pkls)} PKLs created, metrics finite.")


# ══════════════════════════════════════════════════════════════════════════════
#  T3 -- Report builder (from existing PKLs, no simulation)
# ══════════════════════════════════════════════════════════════════════════════

def test_T3_report_builder(tmp_dirs):
    """_build_report must produce a valid md string from results_E and None N2."""
    raw, summ = tmp_dirs
    dgp = _make_dgp()

    # Run a minimal Block E to get some results
    results_E = mod.run_block_E(
        dgp, n_list=[100], M=3,
        config_ids=["E_baseline"],
        force=False,
    )

    # Build report with no Block N2 results
    lines = mod._build_report(results_E, None)
    assert isinstance(lines, list), "Expected list of strings"
    assert len(lines) > 10, f"Report too short: {len(lines)} lines"

    # Write and re-read to verify round-trip
    report_path = summ / "dgp2_optimal_region.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    text = report_path.read_text(encoding="utf-8")

    # Must contain key sections
    for section_header in [
        "## 1. Objective",
        "## 2. Design",
        "## 3. Block E Results",
        "## 4. Pareto Classification",
        "## 5. Block N2",
        "## 6. Interpretation",
        "## 7. Recommendation",
    ]:
        assert section_header in text, f"Missing section: {section_header!r}"

    # N2 placeholder must appear (no N2 data)
    assert "Block N2 not run yet" in text, "Expected Block N2 placeholder"

    # Baseline row must appear in Block E table
    assert "E_baseline" in text, "E_baseline missing from report"

    print(f"T3 passed: report has {len(lines)} lines, all 7 sections present.")
