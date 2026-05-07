"""
pci/tests/test_simulations_organization.py
==========================================

C12 (May 2026) coherence tests for the simulations/ tree.

These tests guarantee that the essay-mapped naming scheme is
preserved as the codebase evolves:

* All 12 S7 simulation scripts and 6 S8 application scripts live under
  the canonical ``simulations/experiments/{s7_simulations,s8_applications}/``
  paths and import cleanly.
* Both figure modules ``simulations/analysis/{s7_figures.py, s8_figures.py}``
  exist and expose their canonical entry points.
* Raw-PKL trees ``simulations/results/raw/{s7,s8}/`` exist with the
  expected leaf directories.
* No legacy ``simulation/`` or ``real_data/`` folder leaks into the
  active tree (they should only exist under ``simulations/archive/``).
* Cross-imports inside the active tree don't reach back into the
  archive (would create a hidden coupling).

These tests run fast (no fits, no I/O on PKLs); they're pure
file-system + import sanity checks.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[2]
_EXP = _REPO / "simulations" / "experiments"
_ANA = _REPO / "simulations" / "analysis"
_RAW = _REPO / "simulations" / "results" / "raw"


# ════════════════════════════════════════════════════════════════════════════
#  Active S7 / S8 scripts: path + module-import contract
# ════════════════════════════════════════════════════════════════════════════

EXPECTED_S7_SCRIPTS = [
    "s7_1_dgp3_stochastic_policy",
    "s7_3_coverage_map",
    "s7_3_scaling_extension",
    "s7_4_identifiability_bennett",
    "s7_4_identifiability_drkpv",
    "s7_5_dgp1_bridge_functional",
    "s7_6_dgp2_head_to_head",
    "s7_6_proxy_asymmetry",
    "s7_6_proxy_quality_grid",
    "s7_7_spectral_diagnostics",
    "s7_8_blind_tuning_bennett",
    "s7_8_blind_tuning_drkpv",
]

EXPECTED_S8_SCRIPTS = [
    "s8_1_checkpoint_diagnose",
    "s8_3_baselines",
    "s8_3_blind_tuning",
    "s8_4_winA_widening",
    "s8_4_winB_validation",
    "s8_4_present_results",
]


@pytest.mark.parametrize("name", EXPECTED_S7_SCRIPTS)
def test_s7_script_present_and_importable(name):
    """Every S7 script exists at its canonical path and imports cleanly."""
    path = _EXP / "s7_simulations" / f"{name}.py"
    assert path.exists(), f"missing S7 script: {path}"
    importlib.import_module(f"simulations.experiments.s7_simulations.{name}")


@pytest.mark.parametrize("name", EXPECTED_S8_SCRIPTS)
def test_s8_script_present_and_importable(name):
    """Every S8 script exists at its canonical path and imports cleanly."""
    path = _EXP / "s8_applications" / f"{name}.py"
    assert path.exists(), f"missing S8 script: {path}"
    importlib.import_module(f"simulations.experiments.s8_applications.{name}")


# ════════════════════════════════════════════════════════════════════════════
#  Figure generator modules
# ════════════════════════════════════════════════════════════════════════════

def test_s7_figures_module_present():
    """simulations/analysis/s7_figures.py is the canonical S7 figure generator."""
    path = _ANA / "s7_figures.py"
    assert path.exists()
    mod = importlib.import_module("simulations.analysis.s7_figures")
    assert hasattr(mod, "_FIGS_S7")
    assert mod._FIGS_S7.name == "S7"


def test_s8_figures_module_present():
    """simulations/analysis/s8_figures.py is the canonical S8 figure generator
    (merged from publication_figures.py + combined_figures.py)."""
    path = _ANA / "s8_figures.py"
    assert path.exists()
    mod = importlib.import_module("simulations.analysis.s8_figures")
    # All 11 expected functions present after the merge
    expected_fns = [
        "load_records", "load_baseline",
        "compute_slope_dist", "compute_contrast_dist",
        "make_triangulation_figure", "make_doseresponse_figure",
        "make_winA_sensitivity_figure",
        "make_combined_doseresponse_figure", "make_combined_triangulation_figure",
        "_build_methods_for_dataset", "main",
    ]
    for fn in expected_fns:
        assert hasattr(mod, fn), f"s8_figures missing {fn} after merge"
    assert mod._FIG.name == "S8"


# ════════════════════════════════════════════════════════════════════════════
#  Raw-PKL tree layout
# ════════════════════════════════════════════════════════════════════════════

EXPECTED_S7_PKL_LEAVES = [
    "blind_tuning_bennett",
    "blind_tuning_drkpv",
    "coverage_map",
    "dgp1_bridge_functional",
    "dgp2_head_to_head",
    "dgp3_stochastic_policy",
    "identifiability_bennett",
    "identifiability_drkpv",
    "proxy_asymmetry",
    "proxy_quality_grid",
    "scaling_extension",
    "spectral_diagnostics",
]

EXPECTED_S8_DATASETS = ["nlsy79_father", "rhc", "nhanes_cadmium", "eth_ess"]


def test_raw_s7_tree_exists():
    s7 = _RAW / "s7"
    assert s7.is_dir(), f"results/raw/s7/ missing"
    for leaf in EXPECTED_S7_PKL_LEAVES:
        assert (s7 / leaf).is_dir(), f"missing s7 leaf: {leaf}"


def test_raw_s8_tree_exists():
    s8 = _RAW / "s8"
    assert s8.is_dir(), f"results/raw/s8/ missing"
    for ds in EXPECTED_S8_DATASETS:
        assert (s8 / ds).is_dir(), f"missing s8 dataset: {ds}"


def test_no_legacy_raw_subdirs():
    """results/raw/ must NOT contain `simulation/` or `real_data/` after C12."""
    assert not (_RAW / "simulation").exists(), "legacy results/raw/simulation/ still present"
    assert not (_RAW / "real_data").exists(), "legacy results/raw/real_data/ still present"


# ════════════════════════════════════════════════════════════════════════════
#  Active-tree purity: no leak from archive/
# ════════════════════════════════════════════════════════════════════════════

def test_no_legacy_subdirs_in_experiments():
    """simulations/experiments/ must contain ONLY s7_simulations/, s8_applications/,
    plus the deprecated dgp2_bias_diagnostics.py shim and the harness.py shim."""
    actual = {p.name for p in _EXP.iterdir()
              if p.is_dir() and not p.name.startswith("__")}
    expected = {"s7_simulations", "s8_applications"}
    unexpected = actual - expected
    assert not unexpected, (
        f"unexpected dirs in simulations/experiments/: {sorted(unexpected)}"
    )


_LEGACY_IMPORT_PATTERNS = [
    re.compile(r"from\s+simulations\.archive\b"),
    re.compile(r"import\s+simulations\.archive\b"),
    re.compile(r"from\s+simulations\.experiments\.simulation\.\w+"),
    re.compile(r"from\s+simulations\.experiments\.real_data\.\w+"),
    re.compile(r"from\s+simulations\.experiments\.dgp3_real_estimation\b"),
]


def test_active_tree_does_not_import_archive():
    """No file under the active tree (pci/, simulations/{experiments,analysis})/
    may import from simulations.archive.* or from legacy paths
    (simulation/, real_data/, *.py at experiments/ root).

    Excluded: this test file itself, since it embeds the very regex
    patterns it scans for (otherwise self-detection)."""
    active_roots = [
        _REPO / "pci",
        _EXP / "s7_simulations",
        _EXP / "s8_applications",
        _ANA / "s7_figures.py",
        _ANA / "s8_figures.py",
    ]
    self_path = Path(__file__).resolve()
    bad = []
    for root in active_roots:
        if root.is_file():
            files = [root]
        else:
            files = list(root.rglob("*.py"))
        for f in files:
            if f.resolve() == self_path:
                continue   # exclude self-reference
            text = f.read_text(encoding="utf-8")
            for pattern in _LEGACY_IMPORT_PATTERNS:
                if pattern.search(text):
                    bad.append(f"{f.relative_to(_REPO)}: {pattern.pattern}")
                    break
    if bad:
        pytest.fail(
            "Active tree imports legacy/archive paths:\n  " + "\n  ".join(bad)
        )


# ════════════════════════════════════════════════════════════════════════════
#  Archive structure
# ════════════════════════════════════════════════════════════════════════════

def test_archive_structure_present():
    """simulations/archive/ has utilities/, analysis/, tests/ subdirs after C12."""
    archive = _REPO / "simulations" / "archive"
    assert archive.is_dir()
    for sub in ("utilities", "analysis", "tests"):
        assert (archive / sub).is_dir(), f"missing simulations/archive/{sub}"
        py_count = len(list((archive / sub).glob("*.py")))
        assert py_count > 0, f"simulations/archive/{sub} is empty"


# ════════════════════════════════════════════════════════════════════════════
#  Pytest config: archive must be excluded from default collection
# ════════════════════════════════════════════════════════════════════════════

def test_pytest_excludes_archive_by_default():
    """pytest.ini norecursedirs must include simulations/archive."""
    cfg = _REPO / "pytest.ini"
    assert cfg.exists(), "pytest.ini missing (created in C12)"
    text = cfg.read_text(encoding="utf-8")
    assert "simulations/archive" in text
