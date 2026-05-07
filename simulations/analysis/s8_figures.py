"""
s8_figures -- Section S8 (Applications) figure generators.

Publication-style figures for the applications chapter (NLSY79
return-to-schooling + RHC mortality).

Merged from two earlier modules into a single canonical analysis module,
parallel to ``simulations/analysis/s7_figures.py`` for the simulation chapter.

Eight figures generated (PDF + PNG), all written to ``simulations/results/figures/S8/``:

Per-dataset (from publication_figures.py)
  - fig_S8_triangulation_nlsy79_father.{pdf,png}    Bennett vs OLS / OLS+X / 2SLS
  - fig_S8_triangulation_rhc.{pdf,png}              same on RHC
  - fig_S8_triangulation_nhanes_cadmium.{pdf,png}   same on NHANES (supplement)
  - fig_S8_doseresponse_nlsy79_father.{pdf,png}     monotone increasing curve
  - fig_S8_doseresponse_rhc.{pdf,png}               monotone decreasing curve
  - fig_S8_winA_sensitivity_nlsy79.{pdf,png}        slope vs ell_h widened-grid sweep

Combined (from combined_figures.py)
  - fig_S8_doseresponse_combined.{pdf,png}    NLSY79 (left) + RHC (right)
  - fig_S8_triangulation_combined.{pdf,png}   NLSY79 (left) + RHC (right)

Run as: ``python -m simulations.analysis.s8_figures``
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cambridge_figures import set_cambridge_style, save_figure
from cambridge_figures.style import figsize as preset_size


# ════════════════════════════════════════════════════════════════════════════
#  Paths and dataset config
# ════════════════════════════════════════════════════════════════════════════

_BASE = Path(__file__).resolve().parents[2]   # repo root from simulations/analysis/
_RAW = _BASE / "simulations" / "results" / "raw" / "s8"
_FIG = _BASE / "simulations" / "results" / "figures" / "S8"
_FIG.mkdir(parents=True, exist_ok=True)


# Per-dataset config: literature ranges (point estimate or [low, high])
DATASET_INFO = {
    "nlsy79_father": {
        "title_short": "NLSY79 — return to schooling",
        "y_label": r"Slope $\partial\log(\text{wage})/\partial(\text{years})$",
        "y_units": "% per year",
        "scale_factor": 100.0,   # log -> percent
        "lit_low": 0.08, "lit_high": 0.12,
        "lit_label": r"Literature [+8\%, +12\%]",
        "stage4_pkl": "stage4/S4_c0_validation.pkl",
        "winA_pkl": "winA/stage3/winA_S3_validation.pkl",  # if exists
        "use_winA": True,
    },
    "rhc": {
        "title_short": "RHC — mortality contrast",
        "y_label": r"Contrast $E[Y(1)] - E[Y(0)]$",
        "y_units": "probability",
        "scale_factor": 1.0,
        "lit_low": -0.030, "lit_high": +0.020,
        "lit_label": r"Cochrane RCT consensus [$-0.03$, $+0.02$]",
        "stage4_pkl": "stage4/S4_c0_validation.pkl",
        "winA_pkl": None,
        "use_winA": False,
    },
    "nhanes_cadmium": {
        "title_short": "NHANES Cadmium — SBP per log-Cd",
        "y_label": r"Slope mmHg / log-Cd unit",
        "y_units": "mmHg/log-unit",
        "scale_factor": 1.0,
        "lit_low": 1.0, "lit_high": 3.0,
        "lit_label": r"Epi reference [+1, +3] (urinary, loose)",
        "stage4_pkl": "stage4/S4_c0_validation.pkl",
        "winA_pkl": None,
        "use_winA": False,
    },
}


# ════════════════════════════════════════════════════════════════════════════
#  Data loaders + summary statistics
# ════════════════════════════════════════════════════════════════════════════

def load_records(dataset: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load J_dr matrix + a_grid for the chosen winner (winA if available, else stage4)."""
    info = DATASET_INFO[dataset]
    if info["use_winA"]:
        pkl_path = _RAW / dataset / "bennett" / info["winA_pkl"]
        if not pkl_path.exists():
            pkl_path = _RAW / dataset / "bennett" / info["stage4_pkl"]
    else:
        pkl_path = _RAW / dataset / "bennett" / info["stage4_pkl"]
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    records = [r for r in data["records"] if r.get("error") is None]
    a_grid = np.asarray(data["meta"]["a_grid"])
    J_dr = np.array([r["J_dr"] for r in records])
    return a_grid, J_dr, data["meta"]


def load_baseline(dataset: str) -> Dict:
    pkl = _RAW / dataset / "baseline" / "baseline.pkl"
    with open(pkl, "rb") as fh:
        return pickle.load(fh)


def compute_slope_dist(a_grid: np.ndarray, J_dr: np.ndarray, scale_factor: float = 1.0) -> Dict:
    """Endpoint slope distribution (M reps)."""
    span = float(a_grid[-1] - a_grid[0])
    slope = (J_dr[:, -1] - J_dr[:, 0]) / span
    return {
        "mean": float(np.mean(slope)) * scale_factor,
        "se": float(np.std(slope, ddof=1)) * scale_factor,
        "ci_lo": float(np.quantile(slope, 0.025)) * scale_factor,
        "ci_hi": float(np.quantile(slope, 0.975)) * scale_factor,
    }


def compute_contrast_dist(a_grid: np.ndarray, J_dr: np.ndarray, scale_factor: float = 1.0) -> Dict:
    """Contrast J_dr(a_high) - J_dr(a_low), NOT divided by span."""
    contrast = J_dr[:, -1] - J_dr[:, 0]
    return {
        "mean": float(np.mean(contrast)) * scale_factor,
        "se": float(np.std(contrast, ddof=1)) * scale_factor,
        "ci_lo": float(np.quantile(contrast, 0.025)) * scale_factor,
        "ci_hi": float(np.quantile(contrast, 0.975)) * scale_factor,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Per-dataset triangulation
# ════════════════════════════════════════════════════════════════════════════

def make_triangulation_figure(dataset: str) -> None:
    """
    Triangulation plot: Bennett vs OLS / OLS+X / 2SLS as horizontal CI bars,
    with literature reference band shaded.
    """
    info = DATASET_INFO[dataset]
    sf = info["scale_factor"]

    # Load Bennett
    a_grid, J_dr, _ = load_records(dataset)
    if dataset == "rhc":
        # For RHC report contrast E[Y(1)] - E[Y(0)] (binary scale)
        bennett = compute_contrast_dist(a_grid, J_dr, scale_factor=sf)
    else:
        bennett = compute_slope_dist(a_grid, J_dr, scale_factor=sf)

    # Load baselines (from baseline.pkl)
    baseline = load_baseline(dataset)
    ols_n = baseline["bootstrap"]["ols_naive"]
    ols_x = baseline["bootstrap"]["ols_with_X"]
    sls = baseline["bootstrap"]["twosls"]

    methods = [
        ("OLS naive", ols_n["psi_mean"] * sf,
         ols_n["psi_q025"] * sf, ols_n["psi_q975"] * sf, "#878787", "o"),
        ("OLS + X", ols_x["psi_mean"] * sf,
         ols_x["psi_q025"] * sf, ols_x["psi_q975"] * sf, "#878787", "s"),
        ("2SLS proximal", sls["psi_mean"] * sf,
         sls["psi_q025"] * sf, sls["psi_q975"] * sf, "#2166ac", "^"),
        ("Bennett-RFF", bennett["mean"],
         bennett["ci_lo"], bennett["ci_hi"], "#762a83", "D"),
    ]

    fw, fh = preset_size("double_column")
    fw, fh = fw, max(fh, 3.0)
    fig, ax = plt.subplots(figsize=(fw, fh))

    y_pos = np.arange(len(methods))[::-1]
    for i, (label, point, lo, hi, color, marker) in enumerate(methods):
        y = y_pos[i]
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]],
                    fmt=marker, color=color, markersize=8,
                    capsize=4, elinewidth=2, markeredgewidth=1.5,
                    label=label, zorder=3)

    ax.axvspan(info["lit_low"] * sf, info["lit_high"] * sf,
               color="#4dac26", alpha=0.18, zorder=1,
               label=info["lit_label"])
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.5, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[0] for m in methods])
    ax.set_xlabel(f"{info['y_label']} ({info['y_units']})")
    ax.set_title(f"Triangulation — {info['title_short']}")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    save_figure(fig, str(_FIG / f"fig_S8_triangulation_{dataset}"))
    plt.close(fig)
    print(f"  Saved triangulation: {dataset}")


# ════════════════════════════════════════════════════════════════════════════
#  Per-dataset dose-response curve
# ════════════════════════════════════════════════════════════════════════════

def make_doseresponse_figure(dataset: str, x_label: str = None) -> None:
    """Dose-response J_dr(a) with bootstrap 95% CI band."""
    info = DATASET_INFO[dataset]
    a_grid, J_dr, meta = load_records(dataset)

    mean = J_dr.mean(axis=0)
    lo = np.quantile(J_dr, 0.025, axis=0)
    hi = np.quantile(J_dr, 0.975, axis=0)

    fw, fh = preset_size("double_column")
    fig, ax = plt.subplots(figsize=(fw, fh))

    ax.fill_between(a_grid, lo, hi, color="#762a83", alpha=0.20,
                    label="95% bootstrap CI")
    ax.plot(a_grid, mean, "o-", color="#762a83", linewidth=2,
            markersize=8, label=r"Bennett-RFF $J(\pi_{a,h})$")

    if x_label is None:
        x_label = {
            "nlsy79_father": "Years of schooling (HGC)",
            "rhc": "Coarsened RHC dose $A^*$",
            "nhanes_cadmium": r"$\log(\text{blood Cd},\ \mu g/L)$",
        }[dataset]

    y_unit = {
        "nlsy79_father": r"$E[\log(\text{wage})\mid \text{do}(A=a)]$",
        "rhc": r"$E[Y(a)]$ — mortality probability",
        "nhanes_cadmium": "Systolic BP (mmHg)",
    }[dataset]

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_unit)
    ax.set_title(f"Bennett dose-response — {info['title_short']}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    save_figure(fig, str(_FIG / f"fig_S8_doseresponse_{dataset}"))
    plt.close(fig)
    print(f"  Saved dose-response: {dataset}")


# ════════════════════════════════════════════════════════════════════════════
#  Win-A sensitivity (NLSY79 only)
# ════════════════════════════════════════════════════════════════════════════

def make_winA_sensitivity_figure() -> None:
    """ell_h widened-grid sweep on NLSY79 — slope per ell_h with CI."""
    base = _RAW / "nlsy79_father" / "bennett" / "winA" / "stage1"
    info = DATASET_INFO["nlsy79_father"]
    sf = info["scale_factor"]

    grid_vals = [0.5, 0.75, 1.0, 1.25, 1.5]
    points, los, his, ses = [], [], [], []
    for i, val in enumerate(grid_vals):
        pkl = base / f"winA_S1_ell_h_{i}.pkl"
        with open(pkl, "rb") as fh:
            data = pickle.load(fh)
        records = [r for r in data["records"] if r.get("error") is None]
        a_grid = np.asarray(data["meta"]["a_grid"])
        J_dr = np.array([r["J_dr"] for r in records])
        slope = compute_slope_dist(a_grid, J_dr, scale_factor=sf)
        points.append(slope["mean"])
        los.append(slope["ci_lo"])
        his.append(slope["ci_hi"])
        ses.append(slope["se"])

    fw, fh = preset_size("double_column")
    fig, ax = plt.subplots(figsize=(fw, fh))

    ax.axhspan(info["lit_low"] * sf, info["lit_high"] * sf,
               color="#4dac26", alpha=0.18,
               label=r"Literature [+8\%, +12\%]")
    ax.axhline(8.0, color="#2166ac", linestyle=":", linewidth=1.0,
               label="Mincer canonical (+8%)")

    for i, val in enumerate(grid_vals):
        marker = "D" if val == 0.5 else "o"
        size = 11 if val == 0.5 else 8
        edgewidth = 2.5 if val == 0.5 else 1.5
        color = "#762a83"
        alpha = 1.0 if val == 0.5 else 0.65
        ax.errorbar(val, points[i], yerr=[[points[i] - los[i]], [his[i] - points[i]]],
                    fmt=marker, color=color, markersize=size,
                    capsize=5, elinewidth=2, markeredgewidth=edgewidth, alpha=alpha,
                    label=("Win A blind winner ($\\ell_h=0.5$)" if val == 0.5 else None))

    ax.set_xlabel(r"Bridge bandwidth $\ell_h$ (widened grid)")
    ax.set_ylabel(r"Bennett slope (% wage premium / year)")
    ax.set_title("NLSY79 — Win A bandwidth sensitivity")
    ax.set_xticks(grid_vals)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    keep = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend([h for h, l in keep], [l for h, l in keep],
              loc="lower left", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    save_figure(fig, str(_FIG / "fig_S8_winA_sensitivity_nlsy79"))
    plt.close(fig)
    print(f"  Saved winA sensitivity: NLSY79")


# ════════════════════════════════════════════════════════════════════════════
#  Combined dose-response (NLSY79 + RHC, shared legend)
# ════════════════════════════════════════════════════════════════════════════

def make_combined_doseresponse_figure() -> None:
    """Two-panel dose-response with shared legend below."""
    fw_single, fh_single = preset_size("double_column")
    fig, axes = plt.subplots(1, 2, figsize=(fw_single * 1.6, fh_single * 1.05))

    panels = [
        ("nlsy79_father", "Years of schooling (HGC)",
         r"$E[\log(\text{wage})\mid \text{do}(A=a)]$"),
        ("rhc", r"Coarsened RHC dose $A^*$",
         r"$E[Y(a)]$ — mortality probability"),
    ]

    line_handle = None
    band_handle = None

    for ax, (dataset, x_label, y_label) in zip(axes, panels):
        a_grid, J_dr, _ = load_records(dataset)
        mean = J_dr.mean(axis=0)
        lo = np.quantile(J_dr, 0.025, axis=0)
        hi = np.quantile(J_dr, 0.975, axis=0)

        h_band = ax.fill_between(a_grid, lo, hi, color="#762a83", alpha=0.20)
        h_line, = ax.plot(a_grid, mean, "o-", color="#762a83", linewidth=2,
                          markersize=8)

        if line_handle is None:
            line_handle = h_line
            band_handle = h_band

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

    fig.legend(
        [line_handle, band_handle],
        [r"Bennett-RFF $J(\pi_{a,h})$", "95% bootstrap CI"],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        framealpha=0.95,
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, str(_FIG / "fig_S8_doseresponse_combined"))
    plt.close(fig)
    print("  Saved combined dose-response: NLSY79 + RHC")


# ════════════════════════════════════════════════════════════════════════════
#  Combined triangulation (NLSY79 + RHC, shared method legend)
# ════════════════════════════════════════════════════════════════════════════

def _build_methods_for_dataset(dataset: str):
    """Returns list of (label, point, ci_lo, ci_hi, color, marker) for triangulation.

    For RHC (binary treatment), Bennett summary is contrast E[Y(1)]-E[Y(0)];
    for NLSY79 / NHANES (continuous), it is the endpoint slope.
    """
    info = DATASET_INFO[dataset]
    sf = info["scale_factor"]

    a_grid, J_dr, _ = load_records(dataset)
    if dataset == "rhc":
        bennett = compute_contrast_dist(a_grid, J_dr, scale_factor=sf)
    else:
        bennett = compute_slope_dist(a_grid, J_dr, scale_factor=sf)

    baseline = load_baseline(dataset)
    ols_n = baseline["bootstrap"]["ols_naive"]
    ols_x = baseline["bootstrap"]["ols_with_X"]
    sls = baseline["bootstrap"]["twosls"]

    methods = [
        ("OLS naive", ols_n["psi_mean"] * sf,
         ols_n["psi_q025"] * sf, ols_n["psi_q975"] * sf, "#878787", "o"),
        ("OLS + X", ols_x["psi_mean"] * sf,
         ols_x["psi_q025"] * sf, ols_x["psi_q975"] * sf, "#878787", "s"),
        ("2SLS proximal", sls["psi_mean"] * sf,
         sls["psi_q025"] * sf, sls["psi_q975"] * sf, "#2166ac", "^"),
        ("Bennett-RFF", bennett["mean"],
         bennett["ci_lo"], bennett["ci_hi"], "#762a83", "D"),
    ]
    return methods, info


def make_combined_triangulation_figure() -> None:
    """Two-panel triangulation; shared method+band legend below; tight y-spacing."""
    fw_single, fh_single = preset_size("double_column")
    fig, axes = plt.subplots(1, 2, figsize=(fw_single * 1.6, fh_single * 0.85))

    panel_datasets = ["nlsy79_father", "rhc"]

    LIT_LABELS = {
        "nlsy79_father": "Literature [+8%, +12%]",
        "rhc": "Cochrane RCT consensus [-0.03, +0.02]",
    }

    method_handles = {}
    band_handle = None

    for ax, dataset in zip(axes, panel_datasets):
        methods, info = _build_methods_for_dataset(dataset)
        sf = info["scale_factor"]

        y_pos = np.arange(len(methods))[::-1]
        for i, (label, point, lo, hi, color, marker) in enumerate(methods):
            y = y_pos[i]
            h = ax.errorbar(
                point, y, xerr=[[point - lo], [hi - point]],
                fmt=marker, color=color, markersize=8,
                capsize=4, elinewidth=2, markeredgewidth=1.5, zorder=3,
            )
            if label not in method_handles:
                method_handles[label] = h

        h_band = ax.axvspan(
            info["lit_low"] * sf, info["lit_high"] * sf,
            color="#4dac26", alpha=0.18, zorder=1,
        )
        if band_handle is None:
            band_handle = h_band

        ax.axvline(0, color="black", linewidth=0.6, alpha=0.5, zorder=2)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([m[0] for m in methods])
        ax.set_xlabel(f"{info['y_label']} ({info['y_units']})")
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_ylim(-0.6, len(methods) - 0.4)

        x_band_mid = (info["lit_low"] + info["lit_high"]) * sf / 2
        ax.text(
            x_band_mid, len(methods) - 0.55, LIT_LABELS[dataset],
            ha="center", va="bottom", fontsize=7.5,
            color="#2d6a18",
        )

    legend_labels = ["OLS naive", "OLS + X", "2SLS proximal", "Bennett-RFF",
                     "Literature reference (range varies by panel)"]
    legend_handles = [method_handles["OLS naive"],
                      method_handles["OLS + X"],
                      method_handles["2SLS proximal"],
                      method_handles["Bennett-RFF"],
                      band_handle]
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        framealpha=0.95,
        fontsize=8,
    )

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save_figure(fig, str(_FIG / "fig_S8_triangulation_combined"))
    plt.close(fig)
    print("  Saved combined triangulation: NLSY79 + RHC")


# ════════════════════════════════════════════════════════════════════════════
#  Main entry point — generate all 8 figures
# ════════════════════════════════════════════════════════════════════════════

def main():
    set_cambridge_style(use_latex=False)

    print("=== Generating Section S8 figures ===\n")

    print("Per-dataset triangulation figures:")
    for ds in ["nlsy79_father", "rhc", "nhanes_cadmium"]:
        make_triangulation_figure(ds)

    print("\nPer-dataset dose-response figures (NLSY79 + RHC):")
    for ds in ["nlsy79_father", "rhc"]:
        make_doseresponse_figure(ds)

    print("\nWin-A sensitivity (NLSY79):")
    make_winA_sensitivity_figure()

    print("\nCombined dose-response (NLSY79 + RHC):")
    make_combined_doseresponse_figure()

    print("\nCombined triangulation (NLSY79 + RHC):")
    make_combined_triangulation_figure()

    print(f"\nAll figures saved to: {_FIG}")


if __name__ == "__main__":
    main()
