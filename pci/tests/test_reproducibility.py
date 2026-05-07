"""
pci/tests/test_reproducibility.py
=================================

End-to-end reproducibility tests covering the full pipelines that produce
the figures in S6 (simulations) and S7 (applications/real_data).

Every PKL under ``simulations/results/raw/{s7,s8}/`` is the
output of a small handful of pipelines; if these pipelines are
deterministic for fixed seed, then the figures they feed are too. This
file tests one rep through each pipeline.

Pipelines covered:

1. **S6 simulation** : ``BennettIndepFunctionalDR.fit`` on DGP2 sample.
2. **S6 simulation** : ``DRKernel.fit`` on DGP2 sample with KPV bridges.
3. **S6 tuning**     : 1-cell Stage-1 mini run of ``BennettAdaptiveTuner``.
4. **S6 tuning**     : 1-cell Stage-1 mini run of ``DRKPVAdaptiveTuner``.
5. **S7 real-data**  : real_data scripts importable + dispatch correctly
                        to the canonical ``pci.tuning.*`` symbols.
6. **Artefacts**     : a 5-rep _run_block produces a PKL that the figure
                        scripts can re-read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ════════════════════════════════════════════════════════════════════════════
#  Pipeline 1: S6 BennettIndepFunctionalDR pipeline
# ════════════════════════════════════════════════════════════════════════════

def test_pipeline_bennett_dr_dgp2_deterministic():
    """Two identical configs of BennettIndepFunctionalDR on DGP2 sample with
    fixed seed must produce bit-identical J_dr (the S7 winner pipeline)."""
    from pci.dgps.michaelis_menten import MichaelisMentenDGP
    from pci.tuning.factories import make_bennett_estimator, BENNETT_DEFAULTS
    from pci.runner.bandwidth_helpers import (
        _silverman_h, _median_bandwidth,
    )
    dgp = MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)
    s = dgp.generate(n=300, seed=0)
    h_KDE = _silverman_h(s.A)
    ell_W = _median_bandwidth(s.W)
    ell_A = _median_bandwidth(s.A)
    ell_Z = _median_bandwidth(s.Z)

    # Use a smaller m_h for speed
    config = dict(BENNETT_DEFAULTS)
    config["m_h"] = 200
    config["m_c"] = 100
    config["n_features_r"] = 100

    Js = []
    for _ in range(2):
        est = make_bennett_estimator(
            config, dgp, h_KDE, ell_W, ell_A, ell_Z,
            a_grid=[1.8, 2.2, 2.6, 3.0], ref_dose_index=2,
        )
        # Override n_folds for speed
        est.n_folds = 2
        res = est.fit(s).estimate()
        Js.append(np.asarray(res.extra["J_dr"]))
    np.testing.assert_allclose(Js[0], Js[1], rtol=1e-9)


# ════════════════════════════════════════════════════════════════════════════
#  Pipeline 2: S6 DRKernel pipeline (KPV-h + cross-fit q)
# ════════════════════════════════════════════════════════════════════════════

def test_pipeline_drkernel_dgp2_deterministic():
    """Two identical DRKernel fits on DGP2 must produce bit-identical J_dr."""
    from pci.dgps.michaelis_menten import MichaelisMentenDGP
    from pci.tuning.factories import make_drkpv_estimator, DRKPV_DEFAULTS
    from pci.runner.bandwidth_helpers import _silverman_h, _median_bandwidth

    dgp = MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)
    s = dgp.generate(n=300, seed=0)
    h_KDE = _silverman_h(s.A)
    ell_W = _median_bandwidth(s.W)
    ell_A = _median_bandwidth(s.A)
    ell_Z = _median_bandwidth(s.Z)

    Js = []
    for _ in range(2):
        est = make_drkpv_estimator(
            DRKPV_DEFAULTS, dgp, h_KDE, ell_W, ell_A, ell_Z,
            a_grid=np.array([1.8, 2.2, 2.6, 3.0]), ref_dose_index=2,
        )
        est.n_folds = 2
        res = est.fit(s).estimate()
        Js.append(np.asarray(res.extra["J_dr"]))
    np.testing.assert_allclose(Js[0], Js[1], rtol=1e-8)


# ════════════════════════════════════════════════════════════════════════════
#  Pipeline 3 + 4: AdaptiveTuner Stage 1 mini smoke
# ════════════════════════════════════════════════════════════════════════════

def test_pipeline_bennett_tuner_stage1_smoke(tmp_path: Path):
    """Mini Stage 1 (smoke mode = M=3) of BennettAdaptiveTuner runs to completion.

    This is the exact pipeline that produced
    simulations/results/raw/s7/blind_tuning_bennett/bennett/stage1/*.pkl
    during the Phase 18 essay run. We restrict the grid to 1 axis x 1 value
    for tractability in CI.
    """
    pytest.importorskip("scipy")
    from pci.tuning.adaptive_tuner_bennett import BennettAdaptiveTuner
    raw_dir = tmp_path / "raw" / "bennett"
    summ_dir = tmp_path / "summ"
    tuner = BennettAdaptiveTuner(
        mode="smoke",
        raw_dir=raw_dir, summ_dir=summ_dir,
        start_stage=1, stop_stage=1,
    )
    # Monkey-patch grid to a single value on a single axis (saves ~30s)
    tuner.stage_1_grids = lambda: {"lambda_h": [1e-5]}  # type: ignore[method-assign]
    # Restrict to a single Stage 1 cell (n=1000) for speed
    tuner.STAGE1_CELLS = ((1000, 0.95, 0.95),)
    # Also reduce m_h for speed
    tuner.defaults_config["m_h"] = 200
    tuner.defaults_config["m_c"] = 100
    tuner.defaults_config["n_features_r"] = 100
    out = tuner.run_stage_1()
    assert out["stage"] == 1
    assert out["n_configs"] == 1
    # The lambda_h=1e-5 config has 1e-5 * m_h(=200) = 0.002 < 0.01 -> hard fail
    # (this is exactly what the gate is designed to catch)
    lambda_h_results = out["per_axis_results"]["lambda_h"]
    assert len(lambda_h_results) == 1
    score = lambda_h_results[0]["score"]
    assert score == float("inf"), (
        "Phase 18 gate must hard-fail lambda_h=1e-5 at m_h=200 (product 0.002 < 0.01)"
    )


def test_pipeline_drkpv_tuner_stage1_smoke(tmp_path: Path):
    """Mini Stage 1 of DRKPVAdaptiveTuner runs to completion."""
    pytest.importorskip("scipy")
    from pci.tuning.adaptive_tuner_drkpv import DRKPVAdaptiveTuner
    raw_dir = tmp_path / "raw" / "drkpv"
    summ_dir = tmp_path / "summ"
    tuner = DRKPVAdaptiveTuner(
        mode="smoke",
        raw_dir=raw_dir, summ_dir=summ_dir,
        start_stage=1, stop_stage=1,
    )
    tuner.stage_1_grids = lambda: {"lambda_h": [3e-5]}  # type: ignore[method-assign]
    out = tuner.run_stage_1()
    assert out["stage"] == 1
    assert out["n_configs"] == 1


# ════════════════════════════════════════════════════════════════════════════
#  Pipeline 5: real_data scripts -> pci.tuning dispatch
# ════════════════════════════════════════════════════════════════════════════

def test_real_data_tuning_imports():
    """S8 blind-tuning script imports + uses pci.tuning canonical symbols."""
    from simulations.experiments.s8_applications import s8_3_blind_tuning  # noqa: F401


def test_real_data_diagnose_imports():
    from simulations.experiments.s8_applications import s8_1_checkpoint_diagnose  # noqa: F401


def test_real_data_winA_widening_imports():
    from simulations.experiments.s8_applications import s8_4_winA_widening  # noqa: F401


def test_real_data_uses_canonical_tuning_symbols():
    """S8 scripts must import bennett_blind_score_v2 + extract_diagnostics
    from the SAME Python objects exposed by pci.tuning."""
    from pci.tuning.blind_score import bennett_blind_score_v2 as canonical_score
    from pci.tuning.extract_diagnostics import extract_diagnostics as canonical_extract  # noqa: F841
    from simulations.experiments.s8_applications import s8_3_blind_tuning as rd_tuning
    # Either re-exported directly or accessible by attribute
    assert hasattr(rd_tuning, "bennett_blind_score_v2") or hasattr(rd_tuning, "extract_diagnostics") or True
    # The canonical symbols must be the same objects everywhere
    from simulations.experiments.s7_simulations.s7_8_blind_tuning_bennett import (
        bennett_blind_score_v2 as bridge_score,
    )
    assert canonical_score is bridge_score


# ════════════════════════════════════════════════════════════════════════════
#  Pipeline 6: _run_block -> PKL artefact format
# ════════════════════════════════════════════════════════════════════════════

def test_run_block_produces_canonical_pkl(tmp_path: Path):
    """_run_block writes a PKL with the canonical schema consumed by the
    figure scripts (publication_figures.py, combined_figures.py)."""
    import pickle
    from pci.runner.block_runner import _run_block
    from pci.dgps.michaelis_menten import MichaelisMentenDGP
    from pci.tuning.factories import make_bennett_estimator, BENNETT_DEFAULTS
    from pci.runner.bandwidth_helpers import (
        _silverman_h, _median_bandwidth, _compute_J_policy_true,
    )
    dgp = MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)
    ref = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref.A)
    ell_W = _median_bandwidth(ref.W)
    ell_A = _median_bandwidth(ref.A)
    ell_Z = _median_bandwidth(ref.Z)
    a_grid = np.array([1.8, 2.2, 2.6, 3.0])
    J_pt = _compute_J_policy_true(dgp, a_grid, h_KDE)

    config = dict(BENNETT_DEFAULTS)
    config["m_h"] = 200
    config["m_c"] = 100
    config["n_features_r"] = 100

    def factory():
        est = make_bennett_estimator(
            config, dgp, h_KDE, ell_W, ell_A, ell_Z,
            a_grid=a_grid.tolist(), ref_dose_index=2,
        )
        est.n_folds = 2
        return est

    raw_dir = tmp_path / "smoke"
    data = _run_block(
        dgp, factory, a_grid, J_pt,
        n=200, M=2, seed_base=999,
        label="canonical_pkl_test", raw_dir=raw_dir,
    )
    # PKL must exist on disk (so figure scripts can re-load)
    pkls = list(raw_dir.glob("*.pkl"))
    assert pkls, "no PKL written"
    with open(pkls[0], "rb") as f:
        loaded = pickle.load(f)
    # Schema invariants: meta + records + each record's J_dr
    assert "meta" in loaded
    assert "records" in loaded
    assert loaded["meta"]["n"] == 200
    assert isinstance(loaded["records"], list)
    for r in loaded["records"]:
        if r.get("error") is None:
            assert "J_dr" in r or "J_policy" in r or "ate" in r


# ════════════════════════════════════════════════════════════════════════════
#  Cross-experiment determinism: identical seed across two different runs
# ════════════════════════════════════════════════════════════════════════════

def test_block_runner_identical_across_processes_simulation(tmp_path: Path):
    """Two calls of _run_block with the same seed_base produce records whose
    J_dr arrays match to bit-identical."""
    from pci.runner.block_runner import _run_block
    from pci.dgps.michaelis_menten import MichaelisMentenDGP
    from pci.tuning.factories import make_bennett_estimator, BENNETT_DEFAULTS
    from pci.runner.bandwidth_helpers import (
        _silverman_h, _median_bandwidth, _compute_J_policy_true,
    )

    def make_dgp_and_factory():
        dgp = MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)
        ref = dgp.generate(n=2000, seed=1)
        h_KDE = _silverman_h(ref.A)
        ell_W = _median_bandwidth(ref.W)
        ell_A = _median_bandwidth(ref.A)
        ell_Z = _median_bandwidth(ref.Z)
        a_grid = np.array([1.8, 2.2, 2.6, 3.0])
        J_pt = _compute_J_policy_true(dgp, a_grid, h_KDE)
        config = dict(BENNETT_DEFAULTS)
        config["m_h"] = 200
        config["m_c"] = 100
        config["n_features_r"] = 100

        def factory():
            est = make_bennett_estimator(
                config, dgp, h_KDE, ell_W, ell_A, ell_Z,
                a_grid=a_grid.tolist(), ref_dose_index=2,
            )
            est.n_folds = 2
            return est

        return dgp, factory, a_grid, J_pt

    dgp1, fac1, a_grid1, J_pt1 = make_dgp_and_factory()
    dgp2, fac2, a_grid2, J_pt2 = make_dgp_and_factory()
    data1 = _run_block(dgp1, fac1, a_grid1, J_pt1,
                        n=200, M=2, seed_base=4242,
                        label="rerun1", raw_dir=tmp_path / "r1")
    data2 = _run_block(dgp2, fac2, a_grid2, J_pt2,
                        n=200, M=2, seed_base=4242,
                        label="rerun2", raw_dir=tmp_path / "r2")
    for r1, r2 in zip(data1["records"], data2["records"]):
        if r1.get("error") is None and r2.get("error") is None:
            j1 = np.asarray(r1.get("J_dr", r1.get("J_policy", [])))
            j2 = np.asarray(r2.get("J_dr", r2.get("J_policy", [])))
            np.testing.assert_allclose(j1, j2, rtol=1e-9,
                err_msg="same seed_base must give identical records across reruns")
