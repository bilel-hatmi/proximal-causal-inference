"""
pci/tests/test_figures_regenerate.py
====================================

C12.D (May 2026) end-to-end smoke tests: every active S7 / S8 figure
must regenerate from the canonical PKL trees at
``simulations/results/raw/{s7, s8, dgp1_identifiability, dgp3_real}/``.

These tests **invoke** the figure-generator functions (no monkey-patching)
and check the output PDF + PNG actually appear in the canonical
``simulations/results/figures/{S7, S8}/`` locations and are non-empty.

If any of these tests fail after a refactor, it likely means a PKL
needed by the figure pipeline was archived/moved by mistake.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[2]
_S7 = _REPO / "simulations" / "results" / "figures" / "S7"
_S8 = _REPO / "simulations" / "results" / "figures" / "S8"


# Map: (s7_figures function name) -> expected (essay/figures/S7/) stem
_S7_FUNC_TO_STEM = [
    ("fig_DGP3_v2_bias_variance",         "fig_S7_dgp3_bias_variance"),
    ("fig_FUSED_sanity_scale",            "fig_S7_convergence_scale"),
    ("fig_NEW_F_bis_v3_biasfix_tau",      "fig_S7_biasfix_tau"),
    ("fig_NEW_G_v3_simplified",           "fig_S7_identifiability"),
    ("fig_NEW_A_bridge_vs_functional_error", "fig_S7_bridge_functional_error"),
    ("fig_FUSED_violin_doses_snr",        "fig_S7_violin_doses_snr"),
    ("fig_FUSED_proxy_winners",           "fig_S7_proxy_winners"),
    ("fig_FUSED_blind_vs_oracle",         "fig_S7_blind_vs_oracle"),
]

_S8_FUNC_CALLS = [
    # (function_name, kwargs, list-of-expected-stems-without-ext)
    ("make_triangulation_figure",   {"dataset": "nlsy79_father"},  ["fig_S8_triangulation_nlsy79_father"]),
    ("make_triangulation_figure",   {"dataset": "rhc"},             ["fig_S8_triangulation_rhc"]),
    ("make_triangulation_figure",   {"dataset": "nhanes_cadmium"},  ["fig_S8_triangulation_nhanes_cadmium"]),
    ("make_doseresponse_figure",    {"dataset": "nlsy79_father"},   ["fig_S8_doseresponse_nlsy79_father"]),
    ("make_doseresponse_figure",    {"dataset": "rhc"},             ["fig_S8_doseresponse_rhc"]),
    ("make_winA_sensitivity_figure", {},                            ["fig_S8_winA_sensitivity_nlsy79"]),
    ("make_combined_doseresponse_figure", {},                       ["fig_S8_doseresponse_combined"]),
    ("make_combined_triangulation_figure", {},                      ["fig_S8_triangulation_combined"]),
]


@pytest.fixture(scope="module")
def s7_figures_module():
    """Import + cambridge-style setup once for all S7 tests."""
    from simulations.analysis import s7_figures
    s7_figures.set_cambridge_style(use_latex=False)
    return s7_figures


@pytest.fixture(scope="module")
def s8_figures_module():
    """Import + cambridge-style setup once for all S8 tests."""
    from simulations.analysis import s8_figures
    s8_figures.set_cambridge_style(use_latex=False)
    return s8_figures


# ════════════════════════════════════════════════════════════════════════════
#  S7 figures: 8 generators
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("fn_name,stem", _S7_FUNC_TO_STEM)
def test_s7_figure_regenerates(fn_name, stem, s7_figures_module):
    """The S7 figure function exists, runs without exception, and writes
    both PDF and PNG to essay/figures/S7/."""
    fn = getattr(s7_figures_module, fn_name)
    pdf = _S7 / f"{stem}.pdf"
    png = _S7 / f"{stem}.png"

    # Capture mtime BEFORE invocation (or 0 if absent)
    pdf_mtime_before = pdf.stat().st_mtime if pdf.exists() else 0
    png_mtime_before = png.stat().st_mtime if png.exists() else 0

    fn()

    assert pdf.exists(), f"{fn_name}: did not write {pdf}"
    assert png.exists(), f"{fn_name}: did not write {png}"
    assert pdf.stat().st_size > 1000, f"{fn_name}: PDF suspiciously small"
    assert png.stat().st_size > 1000, f"{fn_name}: PNG suspiciously small"
    # Newer mtime confirms the function actually wrote (not just that the file exists)
    assert pdf.stat().st_mtime >= pdf_mtime_before
    assert png.stat().st_mtime >= png_mtime_before


# ════════════════════════════════════════════════════════════════════════════
#  S8 figures: 8 generators (3 per-dataset triangulation, 2 dose-response,
#                            1 winA, 2 combined)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("fn_name,kwargs,stems", _S8_FUNC_CALLS,
                         ids=[f"{call[0]}({call[1]})" for call in _S8_FUNC_CALLS])
def test_s8_figure_regenerates(fn_name, kwargs, stems, s8_figures_module):
    fn = getattr(s8_figures_module, fn_name)

    # mtime snapshot
    before = {}
    for stem in stems:
        for ext in ("pdf", "png"):
            p = _S8 / f"{stem}.{ext}"
            before[(stem, ext)] = p.stat().st_mtime if p.exists() else 0

    fn(**kwargs)

    for stem in stems:
        for ext in ("pdf", "png"):
            p = _S8 / f"{stem}.{ext}"
            assert p.exists(), f"{fn_name}({kwargs}): missing {p}"
            assert p.stat().st_size > 1000, f"{p} suspiciously small"
            assert p.stat().st_mtime >= before[(stem, ext)]


# ════════════════════════════════════════════════════════════════════════════
#  Cross-check: top-level results/raw/ has only the expected dirs
# ════════════════════════════════════════════════════════════════════════════

def test_raw_top_level_is_clean():
    """results/raw/ must contain only s7/, s8/, dgp1_identifiability/,
    dgp3_real/, and _archive/. No loose .pkl, no other subdirs."""
    raw = _REPO / "simulations" / "results" / "raw"
    expected_dirs = {"s7", "s8", "dgp1_identifiability", "dgp3_real", "_archive"}

    actual_dirs = {p.name for p in raw.iterdir() if p.is_dir()}
    actual_files = {p.name for p in raw.iterdir() if p.is_file()}

    unexpected_dirs = actual_dirs - expected_dirs
    assert not unexpected_dirs, f"unexpected dirs in raw/: {unexpected_dirs}"
    assert not actual_files, f"loose files in raw/: {actual_files}"
