"""
S7 figures generator (simulations section).

Generates all S7 figures from cached PKLs under
``simulations/results/raw/s7/``, using a publication-style helper.

Active S7 figures (written to ``simulations/results/figures/S7/``):
  fig_S7_bridge_functional_error   DGP1 bridge vs functional error
  fig_S7_dgp3_bias_variance        DGP3 stochastic-policy bias/variance
  fig_S7_convergence_scale         M and n convergence on DGP2 ref cell
  fig_S7_biasfix_tau               BiasFix as a function of threshold
  fig_S7_identifiability           3x3 (lambda_h, lambda_r) heatmaps
  fig_S7_violin_doses_snr          Per-dose violin (boundary vs centre)
  fig_S7_proxy_winners             4x4 SNR_W x SNR_Z winners
  fig_S7_blind_vs_oracle           Blind vs oracle calibration

Usage:
  python -m simulations.analysis.s7_figures --figs B,D,E,F,H
  python -m simulations.analysis.s7_figures --figs all
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from cambridge_figures.style import set_cambridge_style, SIZE_PRESETS
from cambridge_figures.io import save_figure
from cambridge_figures.palettes import get_method_style as _gms_raw


def get_method_style(method: str, drop_label: bool = True) -> dict:
    """Wrapper that drops the 'label' kwarg from cambridge_figures default
    style so we can pass our own (METHOD_PRETTY) label without conflict."""
    s = dict(_gms_raw(method))
    if drop_label:
        s.pop("label", None)
    return s

_BASE = Path(__file__).resolve().parents[2]
_RAW = _BASE / "simulations" / "results" / "raw"
_FIGS = _BASE / "simulations" / "results" / "figures"
_TABLES = _BASE / "simulations" / "results" / "tables"
_FIGS.mkdir(parents=True, exist_ok=True)
_TABLES.mkdir(parents=True, exist_ok=True)

# canonical figure layout. The 8 active S7
# figures all write to simulations/results/figures/S7/. The thematic subdirs
# (dgp1/, dgp2/{scaling,asymmetry,...}/, dgp3/, conceptual/) are kept as
# aliases so legacy/intermediate figures (orphans, _v2/_v3 variants)
# continue to write to their historical locations under _archive/.
_FIGS_S7            = _FIGS / "S7"
_FIGS_S8            = _FIGS / "S8"
_FIGS_ARCHIVE       = _FIGS / "_archive"
# Legacy sub-directories — historical names kept for orphan/intermediate
# figures. Active S7 figures (referenced in 07_simulations.tex) MUST use
# _FIGS_S7 with the fig_S7_<descriptive> stem (cf. C11 mapping).
_FIGS_CONCEPTUAL    = _FIGS_ARCHIVE / "conceptual"
_FIGS_DGP1          = _FIGS_ARCHIVE / "dgp1"
_FIGS_DGP2_SCALING  = _FIGS_ARCHIVE / "dgp2" / "scaling"
_FIGS_DGP2_DIST     = _FIGS_ARCHIVE / "dgp2" / "distributions"
_FIGS_DGP2_ASYM     = _FIGS_ARCHIVE / "dgp2" / "asymmetry"
_FIGS_DGP2_DR       = _FIGS_ARCHIVE / "dgp2" / "dr_mechanism"
_FIGS_DGP2_IDENT    = _FIGS_ARCHIVE / "dgp2" / "identifiability"
_FIGS_DGP2_SPEC     = _FIGS_ARCHIVE / "dgp2" / "spectral"
_FIGS_DGP2_BLIND    = _FIGS_ARCHIVE / "dgp2" / "blind_tuning"
_FIGS_DGP3          = _FIGS_ARCHIVE / "dgp3"
for _d in (_FIGS_S7, _FIGS_S8, _FIGS_ARCHIVE,
           _FIGS_CONCEPTUAL, _FIGS_DGP1, _FIGS_DGP2_SCALING, _FIGS_DGP2_DIST,
           _FIGS_DGP2_ASYM, _FIGS_DGP2_DR, _FIGS_DGP2_IDENT, _FIGS_DGP2_SPEC,
           _FIGS_DGP2_BLIND, _FIGS_DGP3):
    _d.mkdir(parents=True, exist_ok=True)

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
REF_IDX = 2

# Methods in plot order (left-to-right in legend)
METHODS_PLOT = ["linearDR", "KPVREG", "DRKPV", "best_bennett"]
METHODS_PLOT_WITH_KALLUS = ["linearDR", "KPVREG", "DRKPV", "best_bennett", "best_kallus"]

# Pretty labels for plots
METHOD_PRETTY = {
    "linearDR":     "linearDR",
    "KPVREG":       "KPV-REG",
    "DRKPV":        "DR-KPV",
    "best_bennett": "Bennett (BIN)",
    "best_kallus":  "Kallus",
    "oracle":       "Oracle",
    "naiveREG":     "naiveREG",
    "linearREG":    "linearREG",
}

# Manual high-contrast palette for the 4 main DGP2 methods.
# Used to override cambridge_figures defaults when methods would clash.
METHOD_OVERRIDE = {
    "linearDR":     {"color": "#D72638", "linestyle": "--", "marker": "s",
                     "linewidth": 1.5, "markersize": 6},
    "KPVREG":       {"color": "#3F88C5", "linestyle": "-",  "marker": "o",
                     "linewidth": 1.5, "markersize": 6},
    "DRKPV":        {"color": "#FFA500", "linestyle": "-.", "marker": "v",
                     "linewidth": 1.5, "markersize": 6},
    "best_bennett": {"color": "#1A535C", "linestyle": "-",  "marker": "D",
                     "linewidth": 1.7, "markersize": 6.5},
    "best_kallus":  {"color": "#7B2CBF", "linestyle": ":",  "marker": "P",
                     "linewidth": 1.5, "markersize": 6},
    "naiveREG":     {"color": "#888888", "linestyle": ":",  "marker": "x",
                     "linewidth": 1.3, "markersize": 5},
    "linearREG":    {"color": "#FFB54C", "linestyle": "--", "marker": "+",
                     "linewidth": 1.3, "markersize": 6},
    "oracle":       {"color": "#000000", "linestyle": "-",  "marker": "*",
                     "linewidth": 1.5, "markersize": 7},
}


def get_method_style_v2(method: str, drop_marker: bool = False) -> dict:
    """High-contrast publication style. Falls back to cambridge_figures default."""
    if method in METHOD_OVERRIDE:
        s = dict(METHOD_OVERRIDE[method])
    else:
        s = get_method_style(method)
    if drop_marker:
        s.pop("marker", None)
        s.pop("markersize", None)
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_pkl(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_J_dr_seeds(data):
    """Return (M, K) array of J_dr per seed (M reps x K doses)."""
    recs = [r for r in data["records"] if r.get("error") is None]
    if not recs:
        return None, None, None
    J_dr = np.array([r["J_dr"] for r in recs])         # (M, K)
    V_hat = np.array([r["V_hat_grid"] for r in recs])  # (M, K)
    J_reg = np.array([r["J_reg"] for r in recs])       # (M, K)
    return J_dr, V_hat, J_reg


def _compute_metrics(data):
    """Compute bse_ref, cov_ref, sd_J_ref from a Phase16-style PKL."""
    J_dr, V_hat, _ = _extract_J_dr_seeds(data)
    if J_dr is None:
        return None
    n = data["meta"]["n"]
    J_pt = np.array(data["meta"]["J_policy_true"])
    SE_pred = np.sqrt(np.maximum(V_hat / n, 0.0))
    bias = J_dr.mean(axis=0) - J_pt
    sd_J = J_dr.std(axis=0)
    mean_SE = SE_pred.mean(axis=0)
    bse = np.abs(bias) / np.where(mean_SE > 1e-15, mean_SE, np.nan)
    lo = J_dr - 1.96 * SE_pred
    hi = J_dr + 1.96 * SE_pred
    cov = ((lo <= J_pt) & (J_pt <= hi)).mean(axis=0)
    return {
        "n": n, "M": len(J_dr), "a_grid": np.array(data["meta"]["a_grid"]),
        "J_pt": J_pt, "bias": bias, "sd_J": sd_J, "mean_SE": mean_SE,
        "bse": bse, "cov": cov, "J_dr": J_dr, "V_hat": V_hat,
    }


def load_phase16_expa(method: str, n: int, snr: float):
    """Load Phase 16 EXP-A PKL for given (method, n, snr)."""
    cell_dir = _RAW / "s7" / "coverage_map" / f"n{n}_snr{int(snr*100):02d}"
    pkl = cell_dir / f"{method}_M200.pkl"
    return _load_pkl(pkl)


def load_phase16_expb(method: str, snrW: float, snrZ: float):
    """Load Phase 16 EXP-B PKL."""
    cell_dir = _RAW / "s7" / "proxy_quality_grid" / f"snrW{int(snrW*100):02d}_snrZ{int(snrZ*100):02d}"
    pkl = cell_dir / f"{method}_M80.pkl"
    return _load_pkl(pkl)


def load_phase15C_E2(method: str, n: int):
    """Load Phase 15C E2 PKL (n=1000 or 2000, SNR=0.95)."""
    cell_dir = _RAW / "s7" / "dgp2_head_to_head" / f"n{n}"
    pkl = cell_dir / f"{method}_M100.pkl"
    return _load_pkl(pkl)


def load_phase19A(method: str, n: int):
    """Load Phase 19A PKL (n=250 or 500, SNR=0.95)."""
    cell_dir = _RAW / "s7" / "scaling_extension" / f"n{n}_snr95"
    pkl = cell_dir / f"{method}_M100.pkl"
    return _load_pkl(pkl)


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-B : Stabilisation MC -- bse_ref(M) on 4 methods
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_B_stabilisation_mc():
    """Sub-sample Phase 16 EXP-A M=200 PKLs at M in {30, 50, 100, 150, 200},
    compute mean bse_ref +/- bootstrap CI, plot for 4 methods."""
    M_grid = [30, 50, 100, 150, 200]
    n_boot = 100
    rng = np.random.default_rng(42)

    fig, (ax_bse, ax_cov) = plt.subplots(1, 2, figsize=SIZE_PRESETS["double_column"],
                                          sharex=True)
    for method in METHODS_PLOT:
        data = load_phase16_expa(method, n=2000, snr=0.95)
        if data is None:
            print(f"  [F-NEW-B] missing PKL for {method} n=2000 snr=0.95")
            continue
        J_dr, V_hat, _ = _extract_J_dr_seeds(data)
        if J_dr is None:
            continue
        n = data["meta"]["n"]
        J_pt_ref = data["meta"]["J_policy_true"][REF_IDX]
        M_total = J_dr.shape[0]

        bse_means, bse_los, bse_his = [], [], []
        cov_means, cov_los, cov_his = [], [], []
        for M in M_grid:
            if M > M_total:
                bse_means.append(np.nan); bse_los.append(np.nan); bse_his.append(np.nan)
                cov_means.append(np.nan); cov_los.append(np.nan); cov_his.append(np.nan)
                continue
            bse_boot = np.empty(n_boot)
            cov_boot = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.choice(M_total, size=M, replace=False)
                Jd = J_dr[idx, REF_IDX]
                Vh = V_hat[idx, REF_IDX]
                SE = np.sqrt(np.maximum(Vh / n, 0.0))
                bias = Jd.mean() - J_pt_ref
                bse_boot[b] = abs(bias) / np.nanmean(SE) if np.nanmean(SE) > 1e-15 else np.nan
                lo = Jd - 1.96 * SE; hi = Jd + 1.96 * SE
                cov_boot[b] = float(np.mean((lo <= J_pt_ref) & (J_pt_ref <= hi)))
            bse_means.append(np.nanmean(bse_boot))
            bse_los.append(np.nanpercentile(bse_boot, 5))
            bse_his.append(np.nanpercentile(bse_boot, 95))
            cov_means.append(np.nanmean(cov_boot))
            cov_los.append(np.nanpercentile(cov_boot, 5))
            cov_his.append(np.nanpercentile(cov_boot, 95))

        style = get_method_style_v2(method)
        label = METHOD_PRETTY.get(method, method)
        ax_bse.plot(M_grid, bse_means, label=label, **style)
        ax_bse.fill_between(M_grid, bse_los, bse_his, alpha=0.18,
                              color=style.get("color", None))
        ax_cov.plot(M_grid, cov_means, label=label, **style)
        ax_cov.fill_between(M_grid, cov_los, cov_his, alpha=0.18,
                              color=style.get("color", None))

    ax_bse.set_xlabel("Monte-Carlo replications $M$")
    ax_bse.set_ylabel(r"$|\mathrm{bias}|/\widehat{SE}$ (bse$_\mathrm{ref}$)")
    ax_bse.set_title("Stabilisation of bse with $M$")
    ax_bse.grid(True, alpha=0.3)
    ax_bse.set_xticks(M_grid)

    ax_cov.set_xlabel("Monte-Carlo replications $M$")
    ax_cov.set_ylabel("Empirical coverage")
    ax_cov.axhline(0.95, color="black", linestyle="--", linewidth=0.8,
                    alpha=0.7, label="Nominal 0.95")
    ax_cov.set_title("Stabilisation of coverage with $M$")
    ax_cov.grid(True, alpha=0.3)
    ax_cov.set_xticks(M_grid)
    ax_cov.set_ylim(0.6, 1.02)

    # Shared legend at bottom
    if ax_cov.get_legend(): ax_cov.get_legend().remove()
    h, l = ax_cov.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.0,
               framealpha=0.92, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.07, 1, 1.0])
    out = _FIGS / "phase19_F_NEW_B_stabilisation_mc"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-B] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-D1/D2/D3 : KDE distributions per dose (cellule reference)
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_D_distributions():
    """For each of 3 doses (a=1.8, 2.6, 3.0), KDE of J_dr for all available
    methods (incl. best_kallus from Phase 15C E2), with vertical line J_true."""
    from scipy.stats import gaussian_kde

    cell_n = 2000
    cell_snr = 0.95
    doses_idx = [0, 2, 3]  # a=1.8, 2.6, 3.0
    out_paths = []

    # Load all methods data
    method_data = {}
    for m in METHODS_PLOT:
        data = load_phase16_expa(m, n=cell_n, snr=cell_snr)
        if data is not None:
            J_dr, _, _ = _extract_J_dr_seeds(data)
            method_data[m] = (J_dr, data["meta"]["J_policy_true"])
    # best_kallus from phase15C_E2 (different seeds, but same DGP/cell)
    data_k = load_phase15C_E2("best_kallus", n=cell_n)
    if data_k is not None:
        J_dr_k, _, _ = _extract_J_dr_seeds(data_k)
        method_data["best_kallus"] = (J_dr_k, data_k["meta"]["J_policy_true"])

    if not method_data:
        print("  [F-NEW-D] no data found, abort")
        return None

    for di in doses_idx:
        a = A_GRID[di]
        fig, ax = plt.subplots(figsize=SIZE_PRESETS["double_column"])
        all_x = []
        # First pass: compute global range using inter-method 5-95 percentile
        all_low, all_hi = [], []
        for m, (J_dr, _) in method_data.items():
            if J_dr is None:
                continue
            x = J_dr[:, di]
            all_low.append(np.percentile(x, 2))
            all_hi.append(np.percentile(x, 98))
        if not all_low:
            continue
        x_lo = min(all_low)
        x_hi = max(all_hi)
        pad = 0.05 * (x_hi - x_lo)

        for m, (J_dr, J_pt) in method_data.items():
            if J_dr is None:
                continue
            x = J_dr[:, di]
            all_x.append(x)
            kde = gaussian_kde(x, bw_method=0.4)
            xs = np.linspace(x_lo - pad, x_hi + pad, 400)
            ys = kde(xs)
            style = get_method_style_v2(m, drop_marker=True)
            color = style.get("color", None)
            label = METHOD_PRETTY.get(m, m)
            ax.plot(xs, ys, label=label, linewidth=1.7,
                     color=color, linestyle=style.get("linestyle", "-"))
            ax.fill_between(xs, 0, ys, alpha=0.15, color=color)
        if all_x:
            J_true = method_data[list(method_data.keys())[0]][1][di]
            ax.axvline(J_true, color="black", linestyle="--", linewidth=1.4,
                        label=f"True $J(\\pi)={J_true:.3f}$")
        ax.set_xlabel(r"Estimated $\widehat{J}(\pi_{a,h})$")
        ax.set_ylabel("Density (KDE)")
        ax.set_title(f"Distribution of estimates at dose $a={a:.1f}$ "
                      f"(DGP2, $n={cell_n}$, SNR$={cell_snr}$, $M=200$)",
                      fontsize=9)
        ax.set_xlim(x_lo - pad, x_hi + pad)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7.5, framealpha=0.92)
        fig.tight_layout()
        idx_label = {0: "D1", 2: "D2", 3: "D3"}[di]
        out = _FIGS / f"phase19_F_NEW_{idx_label}_kde_a{int(a*10):02d}"
        save_figure(fig, str(out))
        plt.close(fig)
        print(f"  [F-NEW-{idx_label}] saved -> {out}.{{pdf,png}}")
        out_paths.append(out)
    return out_paths


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-E : Heatmap 4x4 winner with composite score
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_E_winner_heatmap(lambda_cov: float = 1.0):
    """For each (SNR_W, SNR_Z) in {.5,.7,.85,.95}^2, compute for each of 4
    methods score = bse + lambda * |cov - 0.95|. Winner = argmin.
    Annotation: Method | bse | cov | margin %.
    Color: margin % (deeper = winner more dominant)."""
    snr_levels = [0.50, 0.70, 0.85, 0.95]
    n_levels = len(snr_levels)

    winners = np.empty((n_levels, n_levels), dtype=object)
    bse_w = np.full((n_levels, n_levels), np.nan)
    cov_w = np.full((n_levels, n_levels), np.nan)
    margin_pct = np.full((n_levels, n_levels), np.nan)

    for i, snrW in enumerate(snr_levels):
        for j, snrZ in enumerate(snr_levels):
            scores = {}
            metrics = {}
            for m in METHODS_PLOT:
                data = load_phase16_expb(m, snrW, snrZ)
                if data is None:
                    continue
                M = _compute_metrics(data)
                if M is None or not np.isfinite(M["bse"][REF_IDX]):
                    continue
                bse_ref = M["bse"][REF_IDX]
                cov_ref = M["cov"][REF_IDX]
                score = bse_ref + lambda_cov * abs(cov_ref - 0.95)
                scores[m] = score
                metrics[m] = (bse_ref, cov_ref)
            if not scores:
                continue
            sorted_methods = sorted(scores.items(), key=lambda kv: kv[1])
            winner_m, score_w = sorted_methods[0]
            winners[i, j] = winner_m
            bse_w[i, j] = metrics[winner_m][0]
            cov_w[i, j] = metrics[winner_m][1]
            if len(sorted_methods) >= 2:
                score_2nd = sorted_methods[1][1]
                margin_pct[i, j] = ((score_2nd - score_w) / score_2nd * 100
                                      if score_2nd > 0 else np.nan)

    # Build figure
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap = LinearSegmentedColormap.from_list(
        "winner", ["#F2EFEA", "#FFB54C", "#D72638"], N=256
    )
    vmax = np.nanpercentile(margin_pct, 95) if np.any(np.isfinite(margin_pct)) else 50
    im = ax.imshow(margin_pct, cmap=cmap, vmin=0, vmax=vmax,
                    aspect="equal", origin="lower")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Margin over runner-up (\\%, composite score)", fontsize=8.5)

    # Method colour patches per cell
    method_colors = {
        "linearDR":     "#FF6B6B",
        "KPVREG":       "#4ECDC4",
        "DRKPV":        "#95E1D3",
        "best_bennett": "#1A535C",
    }
    for i in range(n_levels):
        for j in range(n_levels):
            w = winners[i, j]
            if w is None:
                ax.text(j, i, "—", ha="center", va="center",
                          fontsize=14, color="gray")
                continue
            mc = method_colors.get(w, "#888")
            # Method name (top), bse + cov (mid), margin (bot)
            method_short = METHOD_PRETTY.get(w, w)
            txt = (f"{method_short}\n"
                    f"bse{bse_w[i,j]:.2f}\n"
                    f"cov{cov_w[i,j]:.2f}\n"
                    f"+{margin_pct[i,j]:.0f}\\%")
            color = "white" if margin_pct[i, j] > vmax * 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.0,
                      color=color, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.18", facecolor=mc,
                                alpha=0.45, edgecolor="black", linewidth=0.5))

    ax.set_xticks(range(n_levels))
    ax.set_xticklabels([f"{s:.2f}" for s in snr_levels])
    ax.set_yticks(range(n_levels))
    ax.set_yticklabels([f"{s:.2f}" for s in snr_levels])
    ax.set_xlabel("SNR$_Z$ (treatment proxy)")
    ax.set_ylabel("SNR$_W$ (outcome proxy)")
    ax.set_title("Pareto-winner per (SNR$_W$, SNR$_Z$) cell -- composite score "
                  + r"$=\mathrm{bse}+|cov-0.95|$" + "\n"
                    f"DGP2, $n=1000$, $M=80$, dose $a=2.6$",
                    fontsize=9)
    fig.tight_layout()
    out = _FIGS / "phase19_F_NEW_E_winner_heatmap"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-E] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-F : Double robustness -- direction + intensity per seed
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_F_double_robustness():
    """For each DR method (linearDR, DRKPV, best_bennett), violin+strip plot
    of (J_dr - J_reg) by seed, normalised by SE_pred. Shows direction (sign)
    and intensity of DR correction."""
    methods = ["linearDR", "DRKPV", "best_bennett"]
    cells = [(2000, 0.95), (1000, 0.95), (2000, 0.85)]

    fig, axes = plt.subplots(len(methods), len(cells),
                              figsize=(8.5, 6.0), sharey="row")
    for r, m in enumerate(methods):
        for c, (n, snr) in enumerate(cells):
            ax = axes[r, c]
            data = load_phase16_expa(m, n=n, snr=snr)
            if data is None:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                          transform=ax.transAxes, fontsize=8, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            J_dr, V_hat, J_reg = _extract_J_dr_seeds(data)
            if J_dr is None:
                continue
            corr = J_dr - J_reg                              # (M, K)
            SE_pred = np.sqrt(np.maximum(V_hat / n, 0.0))    # (M, K)
            # Normalise correction by SE_pred to get intensity ratio
            with np.errstate(divide="ignore", invalid="ignore"):
                norm_corr = corr / SE_pred                   # (M, K)
            # One panel per dose
            x_positions = np.arange(len(A_GRID))
            data_box = [norm_corr[:, k] for k in range(len(A_GRID))]
            parts = ax.violinplot(data_box, positions=x_positions, widths=0.7,
                                    showmeans=False, showmedians=True,
                                    showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(get_method_style_v2(m).get("color", "#888"))
                pc.set_alpha(0.6)
                pc.set_edgecolor("black")
            # Strip plot of individual seeds
            for k in range(len(A_GRID)):
                jitter = np.random.uniform(-0.1, 0.1, size=len(norm_corr))
                ax.scatter(x_positions[k] + jitter, norm_corr[:, k],
                            s=4, color="black", alpha=0.25)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f"{a:.1f}" for a in A_GRID], fontsize=7.5)
            if r == 0:
                ax.set_title(f"$n{{=}}{n}$, SNR$={snr}$", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{METHOD_PRETTY.get(m, m)}\n"
                                r"$(\widehat{J}_\mathrm{DR} - \widehat{J}_\mathrm{REG})/\widehat{SE}$",
                                fontsize=8.5)
            if r == len(methods) - 1:
                ax.set_xlabel("Dose $a$")
            ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Double-robustness correction by seed -- direction (sign) and intensity (units of $\\widehat{SE}$)\n"
                  "DGP2 (Phase 16 EXP-A, $M=200$)",
                  fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = _FIGS / "phase19_F_NEW_F_double_robustness"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-F] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-H : Staircase oracle gap (Phase 17 -> Phase 18 -> BIN_mh2000)
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_H_staircase_oracle_gap():
    """3 levels x 4 doses bar chart at cellule reference n=2000 SNR=0.95.
    Levels:
      - Phase 17 (lambda_h=1e-6, blind, collapsed)
      - Phase 18 (lambda_h=1e-5, blind, gate-fixed)
      - BIN_mh2000 (oracle-tuned)
    Metric: bse per dose."""
    # Load PKLs
    p17 = _load_pkl(_RAW / "phase17" / "bennett" / "stage5" / "S5_n2000_W95_Z95.pkl")
    p18 = _load_pkl(_RAW / "phase18" / "bennett" / "stage5" / "S5_n2000_W95_Z95.pkl")
    pBIN = _load_pkl(_RAW / "phase14B" / "wave5_validation" / "snr95_n2000" / "BIN_mh2000_M100.pkl")

    levels = []
    if p17 is not None: levels.append(("Phase 17 blind\n($\\lambda_h{=}10^{-6}$, collapsed)", _compute_metrics(p17), "#D72638"))
    if p18 is not None: levels.append(("Phase 18 blind\n(gate-corrected)", _compute_metrics(p18), "#FFB54C"))
    if pBIN is not None: levels.append(("BIN$_{mh2000}$\n(oracle-tuned)", _compute_metrics(pBIN), "#1A535C"))

    if not levels:
        print("  [F-NEW-H] missing PKLs, abort")
        return None

    fig, ax = plt.subplots(figsize=SIZE_PRESETS["double_column"])
    n_doses = len(A_GRID)
    bar_w = 0.25
    x = np.arange(n_doses)
    for i, (label, metrics, color) in enumerate(levels):
        if metrics is None:
            continue
        ax.bar(x + (i - 1) * bar_w, metrics["bse"], width=bar_w,
                label=label, color=color, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$a={a:.1f}$" for a in A_GRID])
    ax.set_xlabel("Dose")
    ax.set_ylabel(r"$|\mathrm{bias}|/\widehat{SE}$ (bse)")
    ax.set_title("Bennett oracle gap across blind tuning protocols\n"
                  f"DGP2 cell $n=2000$, SNR$_W{{=}}$SNR$_Z{{=}}0.95$, $M=100$",
                  fontsize=9)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.7,
                label="Acceptable bse $\\leq 0.5$")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = _FIGS / "phase19_F_NEW_H_staircase_oracle_gap"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-H] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-C : Scaling n -- bse_ref(n) and cov_ref(n) (after Phase 19A)
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_C_scaling_n():
    """bse_ref(n) and cov_ref(n) for n in {250, 500, 1000, 2000} on 4 methods.
    n=1000, 2000 from Phase 16 EXP-A; n=250, 500 from Phase 19A."""
    n_vals = [250, 500, 1000, 2000]
    fig, (ax_bse, ax_cov) = plt.subplots(1, 2, figsize=SIZE_PRESETS["double_column"])

    for method in METHODS_PLOT:
        bses, covs = [], []
        for n in n_vals:
            if n in (1000, 2000):
                data = load_phase16_expa(method, n=n, snr=0.95)
            else:
                data = load_phase19A(method, n=n)
            if data is None:
                bses.append(np.nan); covs.append(np.nan); continue
            M = _compute_metrics(data)
            if M is None:
                bses.append(np.nan); covs.append(np.nan); continue
            bses.append(M["bse"][REF_IDX]); covs.append(M["cov"][REF_IDX])

        style = get_method_style_v2(method)
        label = METHOD_PRETTY.get(method, method)
        ax_bse.plot(n_vals, bses, label=label, **style)
        ax_cov.plot(n_vals, covs, label=label, **style)

    # Linear scale on x to better separate 250 and 500
    ax_bse.set_xticks(n_vals)
    ax_bse.set_xticklabels([str(n) for n in n_vals])
    ax_bse.set_xlabel("Sample size $n$")
    ax_bse.set_ylabel(r"$|\mathrm{bias}|/\widehat{SE}$ (bse$_\mathrm{ref}$)")
    ax_bse.set_title("Scaling of bse with $n$")
    ax_bse.grid(True, alpha=0.3)
    ax_cov.set_xticks(n_vals)
    ax_cov.set_xticklabels([str(n) for n in n_vals])
    ax_cov.set_xlabel("Sample size $n$")
    ax_cov.set_ylabel("Empirical coverage")
    ax_cov.axhline(0.95, color="black", linestyle="--", linewidth=0.8,
                    alpha=0.7, label="Nominal 0.95")
    ax_cov.set_title("Scaling of coverage with $n$")
    ax_cov.grid(True, alpha=0.3)
    ax_cov.set_ylim(0.45, 1.02)

    # Shared legend at bottom (ax_cov has methods + nominal line)
    h, l = ax_cov.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.0,
               framealpha=0.92, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.07, 1, 1.0])
    out = _FIGS / "phase19_F_NEW_C_scaling_n"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-C] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-G : Identifiability heatmap 3x3 (after Phase 19B)
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_G_identifiability_heatmap():
    """3x3 heatmap of bse_ref (J_dr error) by (h_level, q_level)."""
    h_levels = ["under", "super", "over"]
    q_levels = ["under", "super", "over"]
    h_lambda = {"under": 1e-7, "super": 3e-5, "over": 1e-1}
    q_lambda = {"under": 1e-7, "super": 1e-3, "over": 1e-1}

    raw_dir = _RAW / "s7" / "identifiability_drkpv"
    if not raw_dir.exists():
        print("  [F-NEW-G] Phase 19B not done yet, abort")
        return None

    bse_grid = np.full((len(h_levels), len(q_levels)), np.nan)
    cov_grid = np.full((len(h_levels), len(q_levels)), np.nan)
    bias_grid = np.full((len(h_levels), len(q_levels)), np.nan)
    for i, h in enumerate(h_levels):
        for j, q in enumerate(q_levels):
            pkl = raw_dir / f"h{h}_q{q}_M50.pkl"
            data = _load_pkl(pkl)
            if data is None:
                continue
            M = _compute_metrics(data)
            if M is None:
                continue
            bse_grid[i, j] = M["bse"][REF_IDX]
            cov_grid[i, j] = M["cov"][REF_IDX]
            bias_grid[i, j] = M["bias"][REF_IDX]

    fig, (ax_bse, ax_cov) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    cmap_bse = LinearSegmentedColormap.from_list(
        "bse", ["#1A535C", "#FFE66D", "#D72638"], N=256)
    vmax = np.nanpercentile(bse_grid, 95) if np.any(np.isfinite(bse_grid)) else 1.0
    im1 = ax_bse.imshow(bse_grid, cmap=cmap_bse, vmin=0, vmax=vmax,
                          origin="lower", aspect="equal")
    fig.colorbar(im1, ax=ax_bse, fraction=0.045, pad=0.04, label="bse")
    for i in range(len(h_levels)):
        for j in range(len(q_levels)):
            if np.isfinite(bse_grid[i, j]):
                txt_color = "white" if bse_grid[i, j] > vmax * 0.55 else "black"
                ax_bse.text(j, i, f"bse={bse_grid[i,j]:.2f}\n"
                                    f"bias={bias_grid[i,j]:+.3f}",
                              ha="center", va="center", fontsize=8.5,
                              color=txt_color, fontweight="bold")
    ax_bse.set_xticks(range(len(q_levels)))
    ax_bse.set_xticklabels([f"$q$:{ql}\n$\\lambda_q{{=}}{q_lambda[ql]:.0e}$"
                              for ql in q_levels], fontsize=8)
    ax_bse.set_yticks(range(len(h_levels)))
    ax_bse.set_yticklabels([f"$h$:{hl}\n$\\lambda_h{{=}}{h_lambda[hl]:.0e}$"
                              for hl in h_levels], fontsize=8)
    ax_bse.set_xlabel("$q$-bridge regularisation")
    ax_bse.set_ylabel("$h$-bridge regularisation")
    ax_bse.set_title("bse$_\\mathrm{ref}$ across the (h, q) grid", fontsize=9.5)

    cmap_cov = LinearSegmentedColormap.from_list(
        "cov", ["#D72638", "#FFE66D", "#1A535C"], N=256)
    im2 = ax_cov.imshow(cov_grid, cmap=cmap_cov, vmin=0.5, vmax=1.0,
                          origin="lower", aspect="equal")
    fig.colorbar(im2, ax=ax_cov, fraction=0.045, pad=0.04, label="coverage")
    for i in range(len(h_levels)):
        for j in range(len(q_levels)):
            if np.isfinite(cov_grid[i, j]):
                txt_color = "black"
                ax_cov.text(j, i, f"{cov_grid[i,j]:.2f}",
                              ha="center", va="center", fontsize=10,
                              color=txt_color, fontweight="bold")
    ax_cov.set_xticks(range(len(q_levels)))
    ax_cov.set_xticklabels([f"$q$:{ql}" for ql in q_levels], fontsize=8.5)
    ax_cov.set_yticks(range(len(h_levels)))
    ax_cov.set_yticklabels([f"$h$:{hl}" for hl in h_levels], fontsize=8.5)
    ax_cov.set_xlabel("$q$-bridge regularisation")
    ax_cov.set_title("Coverage across the (h, q) grid", fontsize=9.5)

    fig.suptitle("Identifiability of $J(\\pi)$ across mis-tuned bridges -- "
                  "DRKernel on DGP2, $n{=}2000$, SNR$_W{=}$SNR$_Z{=}0.95$, $M{=}50$",
                  fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = _FIGS / "phase19_F_NEW_G_identifiability_heatmap"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-G] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-A : DGP1 bridge error vs functional error
# ──────────────────────────────────────────────────────────────────────────────

def load_phase15C_E1(method: str, n: int):
    """Load Phase 15C E1 DGP1 PKL."""
    cell_dir = _RAW / "s7" / "dgp1_bridge_functional" / f"n{n}"
    pkl = cell_dir / f"{method}_M100.pkl"
    return _load_pkl(pkl)


def fig_NEW_A_bridge_vs_functional_error():
    """v3 (empirical only with concentration ellipses): Functional
    identifiability demonstration on DGP1.

    Empirical scatter of 6 methods x 30 seeds with per-method concentration
    ellipses (1-sigma and 2-sigma) overlaid on individual points. The
    ellipses make the concentration of each method-cluster visible while
    preserving outlier visibility through the underlying scatter.

    Source PKL : simulations/results/raw/dgp1_identifiability/results.pkl
    """
    from matplotlib.patches import Ellipse

    pkl_path = _RAW / "dgp1_identifiability" / "results.pkl"
    if not pkl_path.exists():
        print(f"  [F-NEW-A v3] missing PKL {pkl_path}; run "
              f"`python -m simulations.experiments._dgp1_identifiability` first")
        return None

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    empirical = data["empirical"]
    meta = data["meta"]

    fig, ax = plt.subplots(figsize=SIZE_PRESETS["double_column"])

    methods_emp = ["oracle", "naive", "linearREG", "linearDR", "DRKPV",
                    "best_bennett"]
    eps_floor = 0.0  # no log axis -- linear scale shows clusters fine

    # Helper: covariance ellipse for a 2D point cloud.
    def cov_ellipse_params(xs, ys, n_std=1.0):
        if len(xs) < 2:
            return None
        cov = np.cov(xs, ys)
        evals, evecs = np.linalg.eigh(cov)
        # eigenvalues ascending -> evecs[:,1] is principal axis
        order = evals.argsort()[::-1]
        evals, evecs = evals[order], evecs[:, order]
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(np.maximum(evals, 0))
        return float(np.mean(xs)), float(np.mean(ys)), width, height, angle

    legend_handles = []
    for m in methods_emp:
        sub = [r for r in empirical if r["method"] == m
               and np.isfinite(r["bridge_err"])]
        if not sub:
            continue
        xs = np.array([r["bridge_err"] for r in sub])
        ys = np.array([max(r["J_err"], eps_floor) for r in sub])

        style = get_method_style_v2(m, drop_marker=False)
        color = style.get("color", "#888")
        marker = style.get("marker", "o")
        # Naive label cleanup
        label = METHOD_PRETTY.get(m, m) if m != "naive" else "naive (drop $W$)"

        # 1-sigma and 2-sigma ellipses
        for n_std, alpha_fill in [(2.0, 0.07), (1.0, 0.18)]:
            params = cov_ellipse_params(xs, ys, n_std=n_std)
            if params is None or m == "oracle":  # oracle is degenerate (h_err = 0)
                continue
            cx, cy, w, h, angle = params
            ell = Ellipse((cx, cy), width=w, height=h, angle=angle,
                            facecolor=color, alpha=alpha_fill,
                            edgecolor=color, linewidth=0.8, zorder=2)
            ax.add_patch(ell)

        # Individual points (translucent)
        ax.scatter(xs, ys, color=color, marker=marker, s=18,
                     edgecolor="black", linewidth=0.3, alpha=0.55, zorder=4)

        # Centroid (bigger, opaque, with black edge)
        cx_med, cy_med = float(np.median(xs)), float(np.median(ys))
        ax.scatter([cx_med], [cy_med], color=color, marker=marker, s=110,
                     edgecolor="black", linewidth=1.2, alpha=1.0, zorder=6,
                     label=label)

    # ── Style ──
    # Auto-fit axis with a small margin
    all_x = [r["bridge_err"] for r in empirical
              if np.isfinite(r["bridge_err"])]
    all_y = [r["J_err"] for r in empirical if np.isfinite(r["J_err"])]
    x_max = max(all_x) * 1.10
    y_max = max(all_y) * 1.20
    ax.set_xlim(left=-0.04, right=x_max)
    ax.set_ylim(bottom=-0.005, top=y_max)
    ax.set_xlabel(r"Bridge $L^2$ error  $\|\widehat{h} - h_0\|_{L^2(P)}$")
    ax.set_ylabel(r"Functional error  $|\widehat{J}(\pi_a) - J_\mathrm{true}(\pi_a)|$")
    pass  # title removed: params go to LaTeX caption
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92, ncol=1,
                title="Method (centroid = median)", title_fontsize=7.5)
    fig.tight_layout()

    out = _FIGS_S7 / "fig_S7_bridge_functional_error"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-A v3] saved -> {out}.{{pdf,png}}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# v2 redesigns
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-D v2 : violin horizontal + jitter, 3 doses superposees vertical
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_D_v2_violin():
    """3 vertically stacked panels (a=1.8, 2.6, 3.0). Horizontal violin +
    jitter strip for 5 methods. Vertical line at J_true. PER-PANEL xlim
    (centred around J_true) for optimal zoom. Bias/sd annotations to the
    left of the method name (outside the violin area)."""
    cell_n = 2000
    cell_snr = 0.95
    doses_idx = [0, 2, 3]   # a=1.8, 2.6, 3.0
    methods = METHODS_PLOT_WITH_KALLUS  # 5 methods

    # Load all method data
    method_data = {}
    for m in METHODS_PLOT:  # 4 methods from EXP-A M=200
        data = load_phase16_expa(m, n=cell_n, snr=cell_snr)
        if data is not None:
            J_dr, _, _ = _extract_J_dr_seeds(data)
            method_data[m] = (J_dr, data["meta"]["J_policy_true"])
    # Kallus from Phase 15C E2 (different seeds)
    data_k = load_phase15C_E2("best_kallus", n=cell_n)
    if data_k is not None:
        J_dr_k, _, _ = _extract_J_dr_seeds(data_k)
        method_data["best_kallus"] = (J_dr_k, data_k["meta"]["J_policy_true"])

    if not method_data:
        print("  [F-NEW-D v2] no data, abort")
        return None

    fig, axes = plt.subplots(len(doses_idx), 1, figsize=(7.5, 9.0))
    rng = np.random.default_rng(42)

    for ax_idx, di in enumerate(doses_idx):
        ax = axes[ax_idx]
        a = A_GRID[di]
        # y-positions for each method (top->bottom = first->last)
        y_positions = {m: len(methods) - 1 - i for i, m in enumerate(methods)}

        # Compute per-panel xlim centered on J_true (visual zoom).
        # We use 1-99 percentile across all methods at this dose, then add pad.
        first_m = next(iter(method_data))
        J_true = method_data[first_m][1][di]
        local_lo, local_hi = [], []
        for m, (J_dr, _) in method_data.items():
            if J_dr is None or m not in methods:
                continue
            x = J_dr[:, di]
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue
            local_lo.append(np.percentile(x, 1))
            local_hi.append(np.percentile(x, 99))
        local_lo.append(J_true); local_hi.append(J_true)
        x_lo = min(local_lo); x_hi = max(local_hi)
        pad = 0.15 * (x_hi - x_lo)
        x_lo -= pad; x_hi += pad

        for m in methods:
            if m not in method_data:
                continue
            J_dr, J_pt = method_data[m]
            if J_dr is None:
                continue
            x = J_dr[:, di]
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue
            yp = y_positions[m]
            color = get_method_style_v2(m).get("color", "#888")
            # Violin horizontal -- denser fill for visibility
            parts = ax.violinplot([x], positions=[yp], widths=0.7,
                                    vert=False, showmeans=False,
                                    showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.55)
                pc.set_edgecolor(color)
                pc.set_linewidth(1.0)
            if "cmedians" in parts:
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(1.6)
            # Strip jitter overlay (smaller, more transparent)
            jitter = rng.uniform(-0.14, 0.14, size=len(x))
            ax.scatter(x, np.full_like(x, yp) + jitter,
                        s=5, color="black",
                        alpha=0.30, zorder=3, linewidth=0)
            # bias and sd: stored to display in y-tick label, NOT inside plot area
            mean_x = float(np.mean(x))
            sd_x = float(np.std(x))
            bias = mean_x - J_true
            # We'll set custom y-tick labels below

        # J_true vertical line
        ax.axvline(J_true, color="black", linestyle="--", linewidth=1.6,
                    alpha=0.85, zorder=4,
                    label=rf"$J_\mathrm{{true}}={J_true:.3f}$")

        # Custom y-tick labels: method name + bias/sd inline
        ytick_labels = []
        for m in methods:
            if m not in method_data:
                ytick_labels.append(METHOD_PRETTY.get(m, m))
                continue
            J_dr, J_pt = method_data[m]
            if J_dr is None:
                ytick_labels.append(METHOD_PRETTY.get(m, m))
                continue
            x = J_dr[:, di]; x = x[np.isfinite(x)]
            if len(x) == 0:
                ytick_labels.append(METHOD_PRETTY.get(m, m))
                continue
            bias = float(np.mean(x)) - J_true
            sd = float(np.std(x))
            label = (rf"{METHOD_PRETTY.get(m, m)}"
                     "\n"
                     rf"$\widehat{{\mathrm{{bias}}}}{{=}}{bias:+.3f}$"
                     rf"   $\widehat{{\sigma}}{{=}}{sd:.3f}$")
            ytick_labels.append(label)
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(ytick_labels, fontsize=8.5)
        # Tighten vertical: less empty space
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-0.6, len(methods) - 0.4)
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_title(f"Dose $a={a:.1f}$", fontsize=11, loc="left",
                      fontweight="bold", pad=4)
        # Per-panel legend (J_true value differs) -- upper left to avoid
        # overlapping the right-side bias/sd column in y-tick labels.
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92)

    axes[-1].set_xlabel(r"Estimated $\widehat{J}(\pi_{a,h})$",
                          fontsize=10.5)
    fig.tight_layout()
    out = _FIGS_DGP2_DIST / "phase19_F_NEW_D_v2_violin_3doses"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-D v2] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-F v2 : scatter |J_REG-J_true| vs |J_DR-J_true| par seed, zones diag
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_F_v2_scatter():
    """bias^2 + variance decomposition for REG vs DR, all 3 methods.

    Why this replaces R^2(tau)
    --------------------------
    R^2(tau) on cell SNR=0.95 was negative for DR-KPV everywhere because
    KPV-REG is already nearly unbiased there: filtering on |REG_err|>tau*SE
    selects high-MC-noise seeds, not structurally biased ones. The DR
    correction does not reduce MC noise, it reduces structural bias --
    so R^2(tau) measured the wrong thing.

    The honest decomposition
    ------------------------
    For each method m and estimator e in {REG, DR}, at each dose a:
        bias^2_m(a) = ( mean_{seeds} (J^_{m,e} - J_true(a)) )^2
        variance_m(a) = Var_{seeds} ( J^_{m,e} )
        MSE_m(a)    = bias^2 + variance
    We aggregate over doses by averaging.

    Visualisation
    -------------
    Stacked bar chart (3 methods x 2 estimators = 6 bars).
    - Bottom segment: bias^2 (dark)
    - Top segment: variance (light)
    - Total height: MSE
    Lecture: linearDR-REG dominated by bias^2 -> DR kills it.
    DR-KPV-REG and Bennett-REG are bias^2 ~ 0 -> DR adds variance, no gain.
    This is the real DR insurance trade-off.
    """
    methods = ["linearDR", "DRKPV", "best_bennett"]
    cell_n, cell_snr = 2000, 0.95

    # Compute bias^2 and variance per (method, estimator, dose)
    decomp = {m: None for m in methods}
    for m in methods:
        data = load_phase16_expa(m, n=cell_n, snr=cell_snr)
        if data is None: continue
        J_dr, V_hat, J_reg = _extract_J_dr_seeds(data)
        if J_dr is None or J_reg is None: continue
        J_pt = np.array(data["meta"]["J_policy_true"])  # (K,)
        # Per-dose bias^2 and variance, then average over doses
        bias_reg = J_reg.mean(axis=0) - J_pt   # (K,)
        bias_dr  = J_dr.mean(axis=0)  - J_pt   # (K,)
        var_reg = J_reg.var(axis=0, ddof=1)    # (K,)
        var_dr  = J_dr.var(axis=0, ddof=1)
        # Aggregate MSE = mean over doses of bias^2 + var
        decomp[m] = {
            "bias2_reg": float(np.mean(bias_reg**2)),
            "bias2_dr":  float(np.mean(bias_dr**2)),
            "var_reg":   float(np.mean(var_reg)),
            "var_dr":    float(np.mean(var_dr)),
            "bias2_reg_per_dose": bias_reg**2,
            "bias2_dr_per_dose":  bias_dr**2,
            "var_reg_per_dose":   var_reg,
            "var_dr_per_dose":    var_dr,
            "a_grid": np.array(data["meta"]["a_grid"]),
        }

    if not any(decomp.values()):
        print("  [F-NEW-F v2] no data"); return None

    # ── Stacked bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # Order: linearDR_REG, linearDR_DR, DRKPV_REG, DRKPV_DR, Bennett_REG, Bennett_DR
    bar_meta = []   # (method, estimator, x_pos)
    x_pos = []
    cur_x = 0.0
    for m_i, m in enumerate(methods):
        if decomp.get(m) is None:
            cur_x += 1.0; continue
        for e_i, est in enumerate(["REG", "DR"]):
            bar_meta.append((m, est, cur_x))
            x_pos.append(cur_x)
            cur_x += 0.85
        cur_x += 0.70   # gap between methods

    method_color = {
        "linearDR":     "#D72638",
        "DRKPV":        "#FFA500",
        "best_bennett": "#1A535C",
    }

    for (m, est, xp) in bar_meta:
        if decomp.get(m) is None: continue
        d = decomp[m]
        bias2 = d[f"bias2_{est.lower()}"]
        var = d[f"var_{est.lower()}"]
        color = method_color.get(m, "#888")
        # bias^2 (bottom) -- darker
        ax.bar(xp, bias2, width=0.78, bottom=0.0, color=color, alpha=0.95,
                edgecolor="black", linewidth=0.6,
                hatch="" if est == "REG" else "//")
        # variance (top) -- lighter
        ax.bar(xp, var, width=0.78, bottom=bias2,
                color=color, alpha=0.45, edgecolor="black", linewidth=0.6,
                hatch="" if est == "REG" else "//")
        # Label inside bar showing MSE total
        mse_tot = bias2 + var
        ax.text(xp, mse_tot * 1.04,
                  rf"$\widehat{{\mathrm{{MSE}}}}={mse_tot:.1e}$",
                  ha="center", va="bottom", fontsize=7.5,
                  color="#222", rotation=0)
        # Estimator label below x axis
        ax.text(xp, -0.06, est, transform=ax.get_xaxis_transform(),
                  ha="center", va="top", fontsize=9, fontweight="bold",
                  color=color)

    # Method group labels (centered between REG/DR)
    for m in methods:
        if decomp.get(m) is None: continue
        xs_method = [xp for (mm, ee, xp) in bar_meta if mm == m]
        if not xs_method: continue
        x_center = float(np.mean(xs_method))
        ax.text(x_center, -0.16, METHOD_PRETTY.get(m, m),
                  transform=ax.get_xaxis_transform(),
                  ha="center", va="top", fontsize=10.5, fontweight="bold",
                  color=method_color.get(m, "#888"))

    # Y log scale to compare across methods (linearDR much larger)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\widehat{\mathrm{bias}}^{\,2}$ (solid)  +  $\widehat{\mathrm{Var}}$ (hatched)  "
                    r"(log scale)", fontsize=10.5)
    ax.set_xticks([])
    ax.set_xlim(-0.5, cur_x - 0.4)
    ax.grid(True, alpha=0.3, axis="y", which="both")

    # Legend: 2 entries (bias^2, variance) + method colours
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#888", alpha=0.95, edgecolor="black",
                label=r"$\widehat{\mathrm{bias}}^{\,2}$  (solid block)"),
        Patch(facecolor="#888", alpha=0.45, edgecolor="black",
                hatch="//", label=r"$\widehat{\mathrm{Var}}$  (hatched block)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8.5,
                framealpha=0.92, title="MSE decomposition",
                title_fontsize=8.5)

    fig.suptitle(r"\textbf{DR insurance trade-off: bias$^{2}$ vs variance, REG (regression-only) vs DR (doubly-robust)}"
                  "\n"
                  rf"DGP2 reference cell, $n={cell_n}$, SNR$_W{{=}}$SNR$_Z{{=}}{cell_snr}$, "
                  r"$M=200$ seeds, averaged over 4 doses",
                  fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = _FIGS_DGP2_DR / "phase19_F_NEW_F_v2_scatter"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-F v2] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-F v6 : BiasFix scan over multiple cells
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_F_v6_biasfix_scan():
    """replace bias^2/Var decomposition by a
    BiasFix scan across cells. BiasFix measures ONLY the structural bias
    component (where DR has a theoretical reason to help), excluding the
    variance cost.

    Quantity (cross-seed structural):
        b_REG(a) = mean_seeds(J_REG^{(i)}(a)) - J_true(a)
        b_DR(a)  = mean_seeds(J_DR^{(i)}(a))  - J_true(a)
        BiasFix(a) = (|b_REG(a)| - |b_DR(a)|) / |b_REG(a)|

    Positive -> DR reduces structural bias. Negative -> DR amplifies.
    At dose ref a=2.6, scanned across 10 cells (6 EXP-A symmetric + 4
    EXP-B asymmetric).
    Bootstrap CI 95% on the BiasFix point estimate (resample seeds).
    The exhibit cell SNR_W=SNR_Z=0.50 is highlighted -- it is the only
    cell where ALL three DR methods are simultaneously positive.
    """
    methods = ["linearDR", "DRKPV", "best_bennett"]
    rng = np.random.default_rng(42)

    # Cells to scan -- ordered by increasing |b_REG| avg across methods
    # (most favourable -> most biased)
    cells_expa = [
        (2000, 0.95), (1000, 0.95), (2000, 0.90),
        (1000, 0.90), (2000, 0.85), (1000, 0.85),
    ]
    cells_expb = [
        (0.95, 0.50), (0.85, 0.50), (0.70, 0.50), (0.50, 0.50),
    ]
    EXHIBIT = (0.50, 0.50)   # highlighted cell

    def load_cell(cell_spec):
        """cell_spec = (n, snr) for EXP-A or (snrW, snrZ) for EXP-B"""
        out = {}
        for m in methods:
            if isinstance(cell_spec[0], int):  # EXP-A: (n, snr)
                d = load_phase16_expa(m, n=cell_spec[0], snr=cell_spec[1])
            else:  # EXP-B: (snrW, snrZ)
                d = load_phase16_expb(m, snrW=cell_spec[0], snrZ=cell_spec[1])
            out[m] = d
        return out

    def cell_label(cell_spec, source):
        if source == "expa":
            return rf"$n={cell_spec[0]}$" + "\n" + rf"SNR$={cell_spec[1]:.2f}$"
        else:
            return (rf"SNR$_W{{=}}{cell_spec[0]:.2f}$" + "\n"
                     + rf"SNR$_Z{{=}}{cell_spec[1]:.2f}$")

    # Compute BiasFix + bootstrap CI per (cell, method)
    biasfix_data = []
    for cell_spec in cells_expa:
        loaded = load_cell(cell_spec)
        biasfix_data.append(("expa", cell_spec, loaded))
    for cell_spec in cells_expb:
        loaded = load_cell(cell_spec)
        biasfix_data.append(("expb", cell_spec, loaded))

    method_color = {
        "linearDR":     "#D72638",
        "DRKPV":        "#FFA500",
        "best_bennett": "#1A535C",
    }

    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    n_cells = len(biasfix_data)
    bar_w = 0.26
    x_centers = np.arange(n_cells, dtype=float)

    # Highlight exhibit cell with background patch
    for k, (src, cs, _) in enumerate(biasfix_data):
        if src == "expb" and cs == EXHIBIT:
            ax.axvspan(k - 0.5, k + 0.5, color="#FFF8E1",
                          edgecolor="#FFA500", linewidth=1.2,
                          alpha=0.95, zorder=0)
            ax.text(k, 1.02, "exhibit cell\nall 3 methods $>0$",
                      transform=ax.get_xaxis_transform(),
                      ha="center", va="bottom", fontsize=8.5,
                      fontweight="bold", color="#7F4F00",
                      bbox=dict(boxstyle="round,pad=0.25",
                                facecolor="#FFE8B0", edgecolor="#FFA500",
                                linewidth=0.8))

    for j, m in enumerate(methods):
        offsets = (j - 1) * bar_w
        bf_vals = []; ci_lo_vals = []; ci_hi_vals = []
        for (src, cs, loaded) in biasfix_data:
            d = loaded.get(m)
            if d is None:
                bf_vals.append(np.nan); ci_lo_vals.append(np.nan)
                ci_hi_vals.append(np.nan); continue
            recs = [r for r in d["records"] if r.get("error") is None]
            if not recs:
                bf_vals.append(np.nan); ci_lo_vals.append(np.nan)
                ci_hi_vals.append(np.nan); continue
            J_dr = np.array([r["J_dr"] for r in recs])
            J_reg = np.array([r["J_reg"] for r in recs])
            J_pt = np.array(d["meta"]["J_policy_true"])
            di = REF_IDX
            err_reg = J_reg[:, di] - J_pt[di]
            err_dr  = J_dr[:, di]  - J_pt[di]
            n_seeds = len(err_reg)
            if n_seeds < 5:
                bf_vals.append(np.nan); ci_lo_vals.append(np.nan)
                ci_hi_vals.append(np.nan); continue
            b_reg = float(np.mean(err_reg))
            b_dr  = float(np.mean(err_dr))
            if abs(b_reg) < 1e-12:
                bf = 0.0
            else:
                bf = (abs(b_reg) - abs(b_dr)) / abs(b_reg) * 100
            # Bootstrap CI
            B = 800
            boot = np.empty(B)
            for b in range(B):
                idx = rng.integers(0, n_seeds, n_seeds)
                br = float(np.mean(err_reg[idx]))
                bd = float(np.mean(err_dr[idx]))
                boot[b] = ((abs(br) - abs(bd)) / abs(br) * 100
                            if abs(br) > 1e-12 else 0.0)
            bf_vals.append(bf)
            ci_lo_vals.append(np.percentile(boot, 2.5))
            ci_hi_vals.append(np.percentile(boot, 97.5))

        bf_arr = np.array(bf_vals)
        lo = bf_arr - np.array(ci_lo_vals)
        hi = np.array(ci_hi_vals) - bf_arr
        ax.bar(x_centers + offsets, bf_arr, width=bar_w,
                color=method_color[m], edgecolor="black", linewidth=0.5,
                alpha=0.88, label=METHOD_PRETTY.get(m, m),
                yerr=[lo, hi], capsize=2.5, ecolor="black",
                error_kw={"linewidth": 0.6})

    # Reference: 0% line (DR neutral)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                zorder=2)
    # Clip y axis (DR-KPV bootstrap CIs explode when |b_REG| ~ MC noise floor)
    ax.set_ylim(-220, 110)
    ax.text(0.99, 0.02,
              "Note: y-axis clipped at $[-220, 110]\\%$;\n"
              "extreme negative CIs at favourable cells\n"
              "(small $|b_\\mathrm{REG}|$ regime).",
              transform=ax.transAxes, ha="right", va="bottom",
              fontsize=7.5, color="#555", style="italic",
              bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                        alpha=0.85, edgecolor="none"))
    ax.set_ylabel(r"$\widehat{\mathrm{BiasFix}}(a_\mathrm{ref}) = \dfrac{|\widehat{b}_\mathrm{REG}| - |\widehat{b}_\mathrm{DR}|}{|\widehat{b}_\mathrm{REG}|}$  (\%)",
                    fontsize=10.5)
    ax.set_xticks(x_centers)
    labels = [cell_label(cs, src) for (src, cs, _) in biasfix_data]
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlim(-0.6, n_cells - 0.4)
    # Visual separation between EXP-A and EXP-B blocks
    boundary = len(cells_expa) - 0.5
    ax.axvline(boundary, color="#666", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text((len(cells_expa) - 1) / 2, -0.18, r"\textbf{Phase 16 EXP-A}  (symmetric SNR)",
              transform=ax.get_xaxis_transform(), ha="center", va="top",
              fontsize=9, color="#444")
    ax.text(boundary + len(cells_expb) / 2, -0.18,
              r"\textbf{Phase 16 EXP-B}  (asymmetric SNR$_W$, SNR$_Z$, $n=1000$)",
              transform=ax.get_xaxis_transform(), ha="center", va="top",
              fontsize=9, color="#444")

    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92, ncol=3)
    ax.set_title(r"\textbf{Structural bias correction across cells: where does the DR truly rescue REG?}"
                  "\n"
                  r"$\widehat{\mathrm{BiasFix}}>0 \Rightarrow$ DR reduces structural bias at $a_\mathrm{ref}{=}2.6$;  "
                  r"95\% bootstrap CI on cross-seed bias estimates",
                  fontsize=10.5)
    fig.tight_layout()
    out = _FIGS_DGP2_DR / "phase19_F_NEW_F_v6_biasfix_scan"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-F v6] saved -> {out}.{{pdf,png}}")
    return out


def fig_NEW_F_bis_v2_exhibit():
    """Round 5: exhibit cell (SNR_W=SNR_Z=0.50, EXP-B). For each of the
    4 doses, compute BiasFix per method with bootstrap CI 95%. The
    expectation is that all three DR methods show BiasFix > 0 at the
    reference dose a=2.6, with linearDR > DR-KPV > Bennett (consistent
    with the bridge mis-specification ranking)."""
    methods = ["linearDR", "DRKPV", "best_bennett"]
    snrW, snrZ = 0.50, 0.50
    rng = np.random.default_rng(43)

    a_grid_all = None
    biasfix_per_method = {}
    cilo_per_method = {}
    cihi_per_method = {}
    n_seeds_per_method = {}

    for m in methods:
        d = load_phase16_expb(m, snrW=snrW, snrZ=snrZ)
        if d is None: continue
        recs = [r for r in d["records"] if r.get("error") is None]
        if not recs: continue
        J_dr = np.array([r["J_dr"] for r in recs])
        J_reg = np.array([r["J_reg"] for r in recs])
        J_pt = np.array(d["meta"]["J_policy_true"])
        a_grid = np.array(d["meta"]["a_grid"])
        if a_grid_all is None: a_grid_all = a_grid
        K = len(a_grid)
        bf, cilo, cihi = [], [], []
        for di in range(K):
            err_reg = J_reg[:, di] - J_pt[di]
            err_dr  = J_dr[:, di]  - J_pt[di]
            n_in = len(err_reg)
            b_reg = float(np.mean(err_reg)); b_dr = float(np.mean(err_dr))
            if abs(b_reg) < 1e-12:
                bf.append(0.0); cilo.append(0.0); cihi.append(0.0); continue
            bf_pt = (abs(b_reg) - abs(b_dr)) / abs(b_reg) * 100
            B = 1000
            boot = np.empty(B)
            for b in range(B):
                idx = rng.integers(0, n_in, n_in)
                br = float(np.mean(err_reg[idx]))
                bd = float(np.mean(err_dr[idx]))
                boot[b] = ((abs(br) - abs(bd)) / abs(br) * 100
                           if abs(br) > 1e-12 else 0.0)
            bf.append(bf_pt)
            cilo.append(np.percentile(boot, 2.5))
            cihi.append(np.percentile(boot, 97.5))
        biasfix_per_method[m] = bf
        cilo_per_method[m] = cilo
        cihi_per_method[m] = cihi
        n_seeds_per_method[m] = len(recs)

    if not biasfix_per_method:
        print("  [F-NEW-F-bis exhibit] no data, abort"); return None

    method_color = {
        "linearDR":     "#D72638",
        "DRKPV":        "#FFA500",
        "best_bennett": "#1A535C",
    }
    K = len(a_grid_all)
    bar_w = 0.27
    x_centers = np.arange(K, dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for j, m in enumerate(methods):
        if m not in biasfix_per_method: continue
        offset = (j - 1) * bar_w
        bf = np.array(biasfix_per_method[m])
        cilo = np.array(cilo_per_method[m])
        cihi = np.array(cihi_per_method[m])
        ax.bar(x_centers + offset, bf, width=bar_w,
                color=method_color[m], alpha=0.88, edgecolor="black",
                linewidth=0.5,
                label=rf"{METHOD_PRETTY.get(m, m)}  ($M{{=}}{n_seeds_per_method[m]}$)",
                yerr=[bf - cilo, cihi - bf], capsize=3, ecolor="black",
                error_kw={"linewidth": 0.7})

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    # Highlight the reference dose
    ref_x = REF_IDX
    ax.axvspan(ref_x - 0.5, ref_x + 0.5, facecolor="#FFF8E1", alpha=0.5,
                 edgecolor="#FFA500", linewidth=0.8, zorder=0)
    ax.text(ref_x, 1.05, "reference dose",
              transform=ax.get_xaxis_transform(), ha="center", va="bottom",
              fontsize=9, fontweight="bold", color="#7F4F00",
              bbox=dict(boxstyle="round,pad=0.20", facecolor="#FFE8B0",
                        edgecolor="#FFA500", linewidth=0.7))

    # Y-axis CLIPPED to [-150, 100]% to keep the reference-dose message
    # readable. Bootstrap CIs that exceed this range are visually capped.
    ax.set_ylim(-150, 110)
    ax.text(0.99, 0.02,
              "Note: error bars clipped at $\\pm 150\\%$ for readability;\n"
              "at boundary doses some BiasFix CIs extend further\n"
              "(small $|b_\\mathrm{REG}|$ regime).",
              transform=ax.transAxes, ha="right", va="bottom",
              fontsize=7.5, color="#555", style="italic",
              bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                        alpha=0.85, edgecolor="none"))

    ax.set_xticks(x_centers)
    ax.set_xticklabels([rf"$a={a:.1f}$" for a in a_grid_all], fontsize=10)
    ax.set_xlabel("Dose", fontsize=10.5)
    ax.set_ylabel(r"$\widehat{\mathrm{BiasFix}}(a) = \dfrac{|\widehat{b}_\mathrm{REG}| - |\widehat{b}_\mathrm{DR}|}{|\widehat{b}_\mathrm{REG}|}$  (\%)",
                    fontsize=10.5)
    ax.set_title(r"\textbf{Exhibit cell}: structural bias rescue by all three DR methods"
                  "\n"
                  rf"DGP2 Phase 16 EXP-B, SNR$_W{{=}}$SNR$_Z{{=}}{snrW:.2f}$ "
                  r"(both proxies degraded $\Rightarrow$ REG biased for all methods)",
                  fontsize=10.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    out = _FIGS_DGP2_DR / "phase19_F_NEW_F_bis_exhibit"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-F-bis exhibit] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-F-bis v3 : BiasFix(tau) scan on exhibit cell (round 6 -- recommended)
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_F_bis_v3_biasfix_tau():
    """exhibit cell SNR_W=SNR_Z=0.50.
    Scan over tau, plot BiasFix(tau) for the 3 DR methods with bootstrap
    95% CI bands. The expected reading: as tau grows we filter on seeds
    where REG was significantly wrong, BiasFix should increase and at
    tau >= 1 all three methods should be POSITIVE.

    Quantity:
        S(tau) = { i : |J_REG^{(i)}(a_ref) - J_true(a_ref)| > tau * SE_pred^{(i)} }
        b_REG(tau) = mean_{i in S(tau)} (J_REG^{(i)}(a_ref) - J_true(a_ref))
        b_DR(tau)  = mean_{i in S(tau)} (J_DR^{(i)}(a_ref)  - J_true(a_ref))
        BiasFix(tau) = (|b_REG(tau)| - |b_DR(tau)|) / |b_REG(tau)| * 100
    """
    methods = ["linearDR", "DRKPV", "best_bennett"]
    snrW, snrZ = 0.50, 0.50
    di = REF_IDX  # a = 2.6

    panel_data = {}
    for m in methods:
        d = load_phase16_expb(m, snrW=snrW, snrZ=snrZ)
        if d is None: continue
        recs = [r for r in d["records"] if r.get("error") is None]
        if not recs: continue
        J_dr = np.array([r["J_dr"] for r in recs])
        J_reg = np.array([r["J_reg"] for r in recs])
        V_hat = np.array([r["V_hat_grid"] for r in recs])
        J_pt = np.array(d["meta"]["J_policy_true"])
        n = d["meta"]["n"]
        err_reg = J_reg[:, di] - J_pt[di]
        err_dr  = J_dr[:, di]  - J_pt[di]
        SE_pred = np.sqrt(np.maximum(V_hat[:, di] / n, 0.0))
        finite = (np.isfinite(err_reg) & np.isfinite(err_dr)
                   & (SE_pred > 1e-12))
        panel_data[m] = (err_reg[finite], err_dr[finite], SE_pred[finite])

    if not panel_data:
        print("  [F-NEW-F-bis v3] no data, abort"); return None

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    tau_grid = np.geomspace(0.5, 5.0, 22)
    B = 800
    rng = np.random.default_rng(2026)

    for m in methods:
        if m not in panel_data: continue
        err_reg, err_dr, se = panel_data[m]
        ratio_reg = np.abs(err_reg) / se
        bf_means, bf_lo, bf_hi = [], [], []
        for tau in tau_grid:
            mask = ratio_reg > tau
            n_in = int(mask.sum())
            if n_in < 5:
                bf_means.append(np.nan); bf_lo.append(np.nan)
                bf_hi.append(np.nan); continue
            er = err_reg[mask]; ed = err_dr[mask]
            b_reg = float(np.mean(er)); b_dr = float(np.mean(ed))
            bf = ((abs(b_reg) - abs(b_dr)) / abs(b_reg) * 100
                  if abs(b_reg) > 1e-12 else 0.0)
            # Bootstrap CI on the filtered seeds
            boot = np.empty(B)
            for b in range(B):
                idx = rng.integers(0, n_in, n_in)
                br = float(np.mean(er[idx]))
                bd = float(np.mean(ed[idx]))
                boot[b] = ((abs(br) - abs(bd)) / abs(br) * 100
                            if abs(br) > 1e-12 else 0.0)
            bf_means.append(bf)
            bf_lo.append(np.percentile(boot, 2.5))
            bf_hi.append(np.percentile(boot, 97.5))

        bf_arr = np.array(bf_means)
        lo_arr = np.array(bf_lo); hi_arr = np.array(bf_hi)
        color = get_method_style_v2(m).get("color", "#888")
        marker = get_method_style_v2(m).get("marker", "o")
        ax.plot(tau_grid, bf_arr, color=color, marker=marker,
                  markersize=5.5, linewidth=1.7,
                  label=METHOD_PRETTY.get(m, m), zorder=4)
        ax.fill_between(tau_grid, lo_arr, hi_arr, color=color,
                          alpha=0.12, zorder=2)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                label="DR neutral")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 3, 5])
    ax.set_xticklabels(["0.5", "1", "2", "3", "5"])
    ax.set_xlim(tau_grid[0], tau_grid[-1])
    ax.set_xlabel(r"Filter threshold $\tau$  "
                    r"(keep seeds with $|\widehat{J}_\mathrm{REG}{-}J_\mathrm{true}| > \tau\cdot\widehat{SE}$)",
                    fontsize=10.5)
    ax.set_ylabel(r"$\widehat{\mathrm{BiasFix}}(\tau) = \dfrac{|\widehat{b}_\mathrm{REG}(\tau)| - |\widehat{b}_\mathrm{DR}(\tau)|}{|\widehat{b}_\mathrm{REG}(\tau)|}$  (\%)",
                    fontsize=10.5)
    # Zone markers: divergent blue (noise floor τ<1) / red (REG very wrong τ>2)
    ax.axvspan(tau_grid[0], 1.0, color="#3F88C5", alpha=0.08, zorder=1)
    ax.axvspan(2.0, tau_grid[-1], color="#D72638", alpha=0.07, zorder=1)
    # Y clip (round 6 zoom: focus on the [-50, 100] range)
    ax.set_ylim(-50, 100)
    ax.text(0.7, 96, "noise\nfloor", ha="center", va="top", fontsize=10,
              color="#1F5BA6", style="italic")
    ax.text(3.2, 96, "REG very\nwrong", ha="center", va="top", fontsize=10,
              color="#7F1F1F", style="italic")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92,
               edgecolor="lightgray")

    fig.tight_layout()
    out = _FIGS_S7 / "fig_S7_biasfix_tau"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-F-bis v3] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-G v3 : 2 panels h-stress / q-stress (round 6 -- recommended)
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_G_v3_h_q_stress():
    """replace the 3x3 dual heatmap by 2 horizontal panels
    showing how DR-KPV and Bennett degrade when ONE regularisation is
    mis-tuned (the other being held at its 'super' optimum).

    Panel left (h-stress): x = lambda_h levels {under, super, over},
        y = bse_ref, 2 curves (DR-KPV with lambda_Q=super, Bennett with
        lambda_r=super).
    Panel right (q/r-stress): x = lambda_Q (DR-KPV) or lambda_r (Bennett)
        levels, y = bse_ref, 2 curves with lambda_h=super.

    Question: which method is more robust to mis-tuning of h vs of q/r?
    """
    levels = ["under", "super", "over"]
    h_lambda_drkpv = {"under": 1e-7, "super": 3e-5, "over": 1e-1}
    q_lambda       = {"under": 1e-7, "super": 1e-3, "over": 1e-1}
    h_lambda_ben   = {"under": 1e-7, "super": 1e-5, "over": 1e-1}
    r_lambda       = {"under": 1e-5, "super": 1e-2, "over": 1e0}

    # Load Phase 19B (DR-KPV) per (h_lvl, q_lvl)
    p19B = _RAW / "s7" / "identifiability_drkpv"
    p19C = _RAW / "s7" / "identifiability_bennett"

    def get_bse_cov(folder, prefix_h, h_lvl, prefix_q, q_lvl):
        f = folder / f"{prefix_h}{h_lvl}_{prefix_q}{q_lvl}_M50.pkl"
        d = _load_pkl(f)
        if d is None: return None, None
        M = _compute_metrics(d)
        if M is None: return None, None
        return float(M["bse"][REF_IDX]), float(M["cov"][REF_IDX])

    # Panel A: h-stress with q (or r) fixed at super
    drkpv_h_bse, drkpv_h_cov = [], []
    bennett_h_bse, bennett_h_cov = [], []
    for lvl in levels:
        b, c = get_bse_cov(p19B, "h", lvl, "q", "super")
        drkpv_h_bse.append(b); drkpv_h_cov.append(c)
        b, c = get_bse_cov(p19C, "h", lvl, "r", "super")
        bennett_h_bse.append(b); bennett_h_cov.append(c)

    # Panel B: q/r-stress with h fixed at super
    drkpv_q_bse, drkpv_q_cov = [], []
    bennett_r_bse, bennett_r_cov = [], []
    for lvl in levels:
        b, c = get_bse_cov(p19B, "h", "super", "q", lvl)
        drkpv_q_bse.append(b); drkpv_q_cov.append(c)
        b, c = get_bse_cov(p19C, "h", "super", "r", lvl)
        bennett_r_bse.append(b); bennett_r_cov.append(c)

    fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=False)

    color_drkpv = "#FFA500"
    color_bennett = "#1A535C"

    def _draw_panel(ax_bse, bse_d, bse_b, cov_d, cov_b,
                     xtick_labels, xlabel, title, lambda_legend_d, lambda_legend_b):
        """Single-axis panel: bse (log), coverage shown as text annotations."""
        xs = np.arange(len(levels))
        ax_bse.plot(xs, bse_d, color=color_drkpv, marker="v",
                      markersize=9, linewidth=1.9, linestyle="-",
                      label="DR-KPV", zorder=4)
        ax_bse.plot(xs, bse_b, color=color_bennett, marker="D",
                      markersize=9, linewidth=1.9, linestyle="-",
                      label="Bennett (BIN)", zorder=4)
        # bse value annotations
        for k in range(len(levels)):
            v = bse_d[k]
            if v is not None and np.isfinite(v):
                ax_bse.text(k, v, rf"  {v:.2f}", fontsize=7.5,
                              va="center", ha="left", color=color_drkpv,
                              fontweight="bold", alpha=0.95)
            v = bse_b[k]
            if v is not None and np.isfinite(v):
                ax_bse.text(k, v, rf"  {v:.2f}", fontsize=7.5,
                              va="center", ha="left", color=color_bennett,
                              fontweight="bold", alpha=0.95)
        # Coverage as small text below each data point
        for k in range(len(levels)):
            cd = cov_d[k]; cb = cov_b[k]
            if cd is not None and np.isfinite(cd):
                ax_bse.annotate(rf"cov={cd:.2f}",
                    xy=(k, bse_d[k] if (bse_d[k] and np.isfinite(bse_d[k])) else 1.0),
                    xytext=(0, -18), textcoords="offset points",
                    fontsize=7.0, color=color_drkpv, ha="center",
                    style="italic", alpha=0.85)
            if cb is not None and np.isfinite(cb):
                ax_bse.annotate(rf"cov={cb:.2f}",
                    xy=(k, bse_b[k] if (bse_b[k] and np.isfinite(bse_b[k])) else 1.0),
                    xytext=(0, -30), textcoords="offset points",
                    fontsize=7.0, color=color_bennett, ha="center",
                    style="italic", alpha=0.85)
        ax_bse.set_yscale("log")
        ax_bse.set_xticks(xs)
        ax_bse.set_xticklabels(xtick_labels, fontsize=9.5)
        ax_bse.set_xlabel(xlabel, fontsize=10)
        ax_bse.set_ylabel(r"$\widehat{\mathrm{bse}}_\mathrm{ref}$  (log scale)",
                            fontsize=10)
        ax_bse.grid(True, alpha=0.3, which="both", axis="y")
        ax_bse.set_title(title, fontsize=11)

    # Panel A: h-stress  (x-ticks use scientific notation)
    xtick_h = [
        r"under$\;(\lambda_h{=}10^{-7})$",
        r"optimal$\;(\lambda_h{\approx}3{\times}10^{-5})$",
        r"over$\;(\lambda_h{=}10^{-1})$",
    ]
    _draw_panel(ax_h, drkpv_h_bse, bennett_h_bse, drkpv_h_cov, bennett_h_cov,
                  xtick_h, r"$h$-bridge regularisation $\lambda_h$",
                  r"$h$-bridge stress",
                  r"$\lambda_Q$ fixed at optimal", r"$\lambda_r$ fixed at optimal")

    # Panel B: q/r-stress
    xtick_q = [
        r"under$\;(\lambda{\approx}10^{-7})$",
        r"optimal$\;(\lambda_Q{=}10^{-3},\;\lambda_r{=}10^{-2})$",
        r"over$\;(\lambda{\approx}1)$",
    ]
    _draw_panel(ax_q, drkpv_q_bse, bennett_r_bse, drkpv_q_cov, bennett_r_cov,
                  xtick_q, r"$q/r$-bridge regularisation $\lambda_Q$ / $\lambda_r$",
                  r"$q/r$-bridge stress",
                  r"$\lambda_h$ fixed at optimal", r"$\lambda_h$ fixed at optimal")

    # Shared legend at bottom (2 methods only — cov is annotated on each point)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=color_drkpv, marker="v", markersize=9,
                linewidth=1.9, linestyle="-", label="DR-KPV"),
        Line2D([0], [0], color=color_bennett, marker="D", markersize=9,
                linewidth=1.9, linestyle="-", label="Bennett (BIN)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                fontsize=10.0, framealpha=0.92,
                bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.06, 1, 1.0])
    out = _FIGS_DGP2_IDENT / "phase19_F_NEW_G_v3_hstress_qstress"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-G v3] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-F-bis (legacy R^2(tau) on biased cell -- kept for reference)
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_F_bis_rescue_biased_cell():
    """show DR truly rescues REG when REG is
    structurally biased. Cell n=1000, SNR_W=SNR_Z=0.85: KPVREG bse_ref=0.41,
    so KPV-REG IS biased -- DR-KPV should rescue.

    Quantity: rescue magnitude R^2(tau) (same formula as previous F-NEW-F v4)
    but on a biased-REG cell where the rescue regime should be visible.
    """
    methods = ["linearDR", "DRKPV", "best_bennett"]
    cell_n, cell_snr = 1000, 0.85

    panel_data = {}
    for m in methods:
        data = load_phase16_expa(m, n=cell_n, snr=cell_snr)
        if data is None:
            panel_data[m] = None; continue
        J_dr, V_hat, J_reg = _extract_J_dr_seeds(data)
        if J_dr is None or J_reg is None:
            panel_data[m] = None; continue
        J_pt = np.array(data["meta"]["J_policy_true"])
        reg_signed = (J_reg - J_pt[None, :]).ravel()
        dr_signed  = (J_dr  - J_pt[None, :]).ravel()
        SE_pred = np.sqrt(np.maximum(V_hat / cell_n, 0.0)).ravel()
        finite = (np.isfinite(reg_signed) & np.isfinite(dr_signed)
                   & (SE_pred > 1e-12))
        panel_data[m] = (reg_signed[finite], dr_signed[finite], SE_pred[finite])

    if not any(panel_data.values()):
        print("  [F-NEW-F-bis] no data, abort")
        return None

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    tau_grid = np.geomspace(0.5, 5.0, 22)
    B = 600
    rng = np.random.default_rng(2027)

    for m in methods:
        if panel_data.get(m) is None: continue
        reg, dr, se = panel_data[m]
        ratio_reg = np.abs(reg) / se
        r2_means, r2_lo, r2_hi = [], [], []
        for tau in tau_grid:
            mask = ratio_reg > tau
            n_in = int(mask.sum())
            if n_in < 5:
                r2_means.append(np.nan); r2_lo.append(np.nan)
                r2_hi.append(np.nan); continue
            reg_s = reg[mask]; dr_s = dr[mask]
            num = float(np.sum(reg_s**2 - dr_s**2))
            den = float(np.sum(reg_s**2))
            r2 = num / den if den > 0 else np.nan
            r2_means.append(100.0 * r2)
            boot = np.empty(B)
            for b in range(B):
                idx = rng.integers(0, n_in, n_in)
                rs = reg_s[idx]; ds = dr_s[idx]
                d2 = float(np.sum(rs**2))
                boot[b] = (100.0 * float(np.sum(rs**2 - ds**2)) / d2
                           if d2 > 0 else np.nan)
            r2_lo.append(np.nanpercentile(boot, 2.5))
            r2_hi.append(np.nanpercentile(boot, 97.5))

        r2_arr = np.array(r2_means)
        color = get_method_style_v2(m).get("color", "#888")
        marker = get_method_style_v2(m).get("marker", "o")
        ax.plot(tau_grid, r2_arr, color=color, marker=marker,
                  markersize=5.5, linewidth=1.7, label=METHOD_PRETTY.get(m, m),
                  zorder=4)
        ax.fill_between(tau_grid, np.array(r2_lo), np.array(r2_hi),
                          color=color, alpha=0.18, zorder=2)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.7,
                label="DR neutral")
    ax.set_xscale("log")
    ax.set_xlabel(r"Filter threshold $\tau$  "
                    r"(keep seeds with $|\widehat{J}_\mathrm{REG}{-}J_\mathrm{true}| > \tau\cdot\widehat{SE}$)",
                    fontsize=10)
    ax.set_ylabel(r"$R^{2}(\tau) = \dfrac{\sum (\,\mathrm{err}_\mathrm{REG}^{2} - \mathrm{err}_\mathrm{DR}^{2})}{\sum \mathrm{err}_\mathrm{REG}^{2}}$  (\%)",
                    fontsize=10)
    ax.set_xticks([0.5, 1, 2, 3, 5])
    ax.set_xticklabels(["0.5", "1", "2", "3", "5"])
    ax.set_xlim(tau_grid[0], tau_grid[-1])
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9, framealpha=0.92)
    ax.set_title(r"\textbf{DR rescue magnitude on a structurally biased cell}"
                  "\n"
                  rf"DGP2 $n={cell_n}$, SNR$_W{{=}}$SNR$_Z{{=}}{cell_snr}$ "
                  r"(KPV-REG is biased here $\Rightarrow$ DR has structural bias to correct), "
                  r"$M=200$ seeds $\times 4$ doses",
                  fontsize=10)

    fig.tight_layout()
    out = _FIGS_DGP2_DR / "phase19_F_NEW_F_bis_biased_cell"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-F-bis] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-E v2 : Heatmap winner raffine (Option A: matplotlib)
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_E_v3_minibars(lambda_cov: float = 1.0):
    """replace illegible heatmap by
    a 4x4 grid of mini bar charts. Each cell SNR_W x SNR_Z contains a
    sorted bar chart of the 4 methods (smallest bse_ref on the left =
    winner). Winner bar in solid colour; runners-up in light shade.
    Annotation Delta% above the winner reports the relative gap to the
    runner-up bse: Delta = (bse_2 - bse_1) / bse_1 * 100. Big Delta =
    winner dominates; small Delta = no clear winner (annotation greyed)."""
    snr_levels = [0.50, 0.70, 0.85, 0.95]
    n_levels = len(snr_levels)
    methods_used = METHODS_PLOT  # 4 methods

    # Compute per-cell sorted (method, bse, cov) tuples
    cell_data = {}
    for snrW in snr_levels:
        for snrZ in snr_levels:
            entries = []
            for m in methods_used:
                d = load_phase16_expb(m, snrW, snrZ)
                if d is None: continue
                M = _compute_metrics(d)
                if M is None or not np.isfinite(M["bse"][REF_IDX]):
                    continue
                entries.append((m, float(M["bse"][REF_IDX]),
                                  float(M["cov"][REF_IDX])))
            entries.sort(key=lambda kv: kv[1])  # sort by bse asc
            cell_data[(snrW, snrZ)] = entries

    # Global ymax (log scale): use 95th percentile across all bse
    all_bse = []
    for entries in cell_data.values():
        for (m, bse, cov) in entries:
            if bse > 0: all_bse.append(bse)
    ymax = float(np.percentile(all_bse, 99)) * 1.20 if all_bse else 1.0
    ymin = max(1e-3, float(np.min(all_bse)) * 0.5) if all_bse else 1e-3

    method_color = {
        "linearDR":     "#D72638",
        "KPVREG":       "#3F88C5",
        "DRKPV":        "#FFA500",
        "best_bennett": "#1A535C",
    }
    method_short = {
        "linearDR":     "lin",
        "KPVREG":       "KPV",
        "DRKPV":        "DR",
        "best_bennett": "Ben",
    }

    fig, axes = plt.subplots(n_levels, n_levels, figsize=(10.5, 10.0),
                              sharey=True)
    # Reverse rows so SNR_W=0.95 is at top (high quality top)
    for i, snrW in enumerate(reversed(snr_levels)):
        for j, snrZ in enumerate(snr_levels):
            ax = axes[i, j]
            entries = cell_data.get((snrW, snrZ), [])
            if not entries:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                          transform=ax.transAxes, fontsize=14, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            n_meth = len(entries)
            xs = np.arange(n_meth)
            heights = [e[1] for e in entries]
            colors = [method_color.get(e[0], "#888") for e in entries]
            # Winner solid, runners-up light
            alphas = [0.95] + [0.30] * (n_meth - 1)
            ax.bar(xs, heights, width=0.78, color=colors, alpha=1.0,
                    edgecolor="black", linewidth=0.5)
            for k, bar in enumerate(ax.containers[0]):
                bar.set_alpha(alphas[k])
            # x labels = method short names
            ax.set_xticks(xs)
            ax.set_xticklabels([method_short.get(e[0], e[0][:3])
                                  for e in entries], fontsize=8.5)
            ax.set_yscale("log")
            ax.set_ylim(ymin, ymax)
            # Hide y ticks except left column
            if j > 0:
                ax.tick_params(labelleft=False)
            else:
                ax.tick_params(labelsize=7.5)
            # Compute gap Delta%
            if n_meth >= 2:
                bse_1 = entries[0][1]
                bse_2 = entries[1][1]
                delta_pct = (bse_2 - bse_1) / bse_1 * 100 if bse_1 > 0 else 0.0
                # Annotation colour: bold black if Delta>=20%, grey/italic if <20%
                strong = delta_pct >= 20.0
                txt = (rf"$\Delta{{=}}{{+}}{delta_pct:.0f}\%$"
                       if delta_pct < 999 else r"$\Delta\gg$")
                ax.text(0.5, 0.99, txt,
                          transform=ax.transAxes, ha="center", va="top",
                          fontsize=9.5,
                          color="black" if strong else "#777",
                          fontweight="bold" if strong else "normal",
                          style="normal" if strong else "italic",
                          bbox=dict(boxstyle="round,pad=0.20",
                                    facecolor="white", alpha=0.85,
                                    edgecolor="none"))
            # Winner cell colour bar at top
            winner = entries[0][0]
            wcol = method_color.get(winner, "#888")
            ax.add_patch(plt.Rectangle((0.0, 0.93), 1.0, 0.07,
                                          transform=ax.transAxes,
                                          facecolor=wcol, alpha=0.95,
                                          edgecolor="none", clip_on=False))

            ax.grid(True, alpha=0.25, axis="y", which="both")
            ax.set_axisbelow(True)

    # Outer labels
    for j, snrZ in enumerate(snr_levels):
        axes[0, j].set_title(rf"SNR$_Z={snrZ:.2f}$", fontsize=10,
                                fontweight="bold")
    for i, snrW in enumerate(reversed(snr_levels)):
        axes[i, 0].set_ylabel(rf"SNR$_W={snrW:.2f}$" + "\n" + r"$\widehat{\mathrm{bse}}_\mathrm{ref}$ (log)",
                                 fontsize=9, fontweight="bold")

    # Method colour legend (bottom)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=method_color[m], label=METHOD_PRETTY.get(m, m),
                edgecolor="black", linewidth=0.5)
        for m in methods_used
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                fontsize=9, framealpha=0.92,
                bbox_to_anchor=(0.5, 0.01),
                title=r"Solid bar $=$ cell winner;  light bars $=$ runners-up;  $\Delta\%$ above winner $=$ relative gap to 2nd",
                title_fontsize=8.5)

    fig.tight_layout(rect=[0, 0.06, 1, 1.0])
    out = _FIGS_DGP2_ASYM / "phase19_F_NEW_E_v3_minibars"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-E v3 minibars] saved -> {out}.{{pdf,png}}")
    return out


def fig_NEW_E_v2_winner_heatmap(lambda_cov: float = 1.0):
    """Refinement: drop colorbar border weight, enlarge
    noms methodes, simplifier annotations cellules ('Method | bse'),
    palette ColorBlind-safe douce."""
    snr_levels = [0.50, 0.70, 0.85, 0.95]
    n_levels = len(snr_levels)
    methods_used = METHODS_PLOT  # 4 methods

    winners = np.empty((n_levels, n_levels), dtype=object)
    bse_w = np.full((n_levels, n_levels), np.nan)
    cov_w = np.full((n_levels, n_levels), np.nan)
    margin_pct = np.full((n_levels, n_levels), np.nan)

    for i, snrW in enumerate(snr_levels):
        for j, snrZ in enumerate(snr_levels):
            scores, metrics = {}, {}
            for m in methods_used:
                data = load_phase16_expb(m, snrW, snrZ)
                if data is None:
                    continue
                M = _compute_metrics(data)
                if M is None or not np.isfinite(M["bse"][REF_IDX]):
                    continue
                bse_ref = M["bse"][REF_IDX]
                cov_ref = M["cov"][REF_IDX]
                score = bse_ref + lambda_cov * abs(cov_ref - 0.95)
                scores[m] = score
                metrics[m] = (bse_ref, cov_ref)
            if not scores:
                continue
            sorted_m = sorted(scores.items(), key=lambda kv: kv[1])
            wm, sw = sorted_m[0]
            winners[i, j] = wm
            bse_w[i, j] = metrics[wm][0]
            cov_w[i, j] = metrics[wm][1]
            if len(sorted_m) >= 2:
                s2 = sorted_m[1][1]
                margin_pct[i, j] = ((s2 - sw) / s2 * 100 if s2 > 0 else np.nan)

    # Color: encode bse_w (lower=better, viridis-like)
    fig, ax = plt.subplots(figsize=(7.8, 6.5))
    cmap = LinearSegmentedColormap.from_list(
        "bseW",
        ["#1A535C", "#4ECDC4", "#FFE66D", "#FFA500", "#D72638"], N=256
    )
    vmax = np.nanpercentile(bse_w, 95) if np.any(np.isfinite(bse_w)) else 1.0
    im = ax.imshow(bse_w, cmap=cmap, vmin=0, vmax=vmax,
                    aspect="equal", origin="lower")
    # Colorbar removed: each cell already shows "Method | bse",
    # so the colorbar is redundant. The colour gradient alone gives the
    # fast lecture (deeper colour = worse winner). Title makes this explicit.

    # Method colour stripes
    method_colors = {
        "linearDR":     "#D72638",
        "KPVREG":       "#3F88C5",
        "DRKPV":        "#FFA500",
        "best_bennett": "#1A535C",
    }
    for i in range(n_levels):
        for j in range(n_levels):
            w = winners[i, j]
            if w is None:
                ax.text(j, i, "—", ha="center", va="center",
                          fontsize=18, color="gray")
                continue
            mc = method_colors.get(w, "#888")
            method_short = METHOD_PRETTY.get(w, w)
            txt = (f"{method_short}"
                    "\n"
                    rf"$\widehat{{\mathrm{{bse}}}}{{=}}{bse_w[i,j]:.2f}$")
            txt_color = "white" if bse_w[i, j] > vmax * 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5,
                      color=txt_color, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.30", facecolor=mc,
                                alpha=0.55, edgecolor="black",
                                linewidth=0.6))

    ax.set_xticks(range(n_levels))
    ax.set_xticklabels([f"{s:.2f}" for s in snr_levels], fontsize=10)
    ax.set_yticks(range(n_levels))
    ax.set_yticklabels([f"{s:.2f}" for s in snr_levels], fontsize=10)
    ax.set_xlabel(r"SNR$_Z$ (treatment proxy quality)", fontsize=11)
    ax.set_ylabel(r"SNR$_W$ (outcome proxy quality)", fontsize=11)
    ax.set_title(r"\textbf{Out-of-cell transfer robustness} of best-performing methods"
                  r" (configurations frozen at SNR$=0.95$, no per-cell retuning)"
                  "\n"
                  r"Winner $= \arg\min$ (bse $+ |cov - 0.95|$),  "
                  r"DGP2 $n{=}1000$, $M{=}80$, dose $a{=}2.6$",
                  fontsize=9.5)
    # Spine cleanup
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    # Caveat banner under the figure 
    fig.text(0.5, 0.005,
             r"\textbf{Caveat (compute limitation):} this measures "
             r"\emph{transfer robustness}, not absolute method quality. "
             r"Each method uses the hyperparameters that won at the central "
             r"cell SNR$_W{=}$SNR$_Z{=}0.95$; a Bennett loss at low SNR does "
             r"NOT mean Bennett is intrinsically poor there -- the "
             r"\emph{config does not transfer}. DR-KPV's robustness reflects "
             r"fewer fragile hyperparameters (KPV regularisation auto-adapts "
             r"via Mastouri ridge), whereas Bennett-RFF carries fixed "
             r"$(m_h, \ell_h, \lambda_h)$ which are SNR-dependent.",
             ha="center", va="bottom", fontsize=7.0, color="#333",
             style="italic", wrap=True,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF8E1",
                       edgecolor="#FFA500", linewidth=0.7, alpha=0.95))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    out = _FIGS_DGP2_ASYM / "phase19_F_NEW_E_winner_heatmap"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-E v2] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F6 spectral v2 : bar chart horizontal Bennett + DRKPV cote-a-cote
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_F6_spectral_v2():
    """2 panels (Bennett gauche, DRKPV droite). Bar chart horizontal:
    longueur = |rho Spearman vs bse_ref|, couleur = signe (rouge negatif,
    bleu positif). Ligne verticale |rho|=0.6 (actionable threshold).
    Donnees : Phase 16 metric audit (deja calcule)."""
    # Hard-coded from docs/notes/phase16_metric_audit.md (across cells)
    # Format: (diagnostic_name, rho, p_value)
    bennett_rows = [
        ("residual\\_norm\\_h", 0.762, 0.000),
        ("eff\\_rank\\_h",      -0.833, 0.000),
        ("eff\\_rank\\_r",      -0.814, 0.000),
        ("kappa\\_h",           -0.641, 0.001),
        ("kappa\\_r",           -0.809, 0.000),
        ("residual\\_norm\\_r", 0.623, 0.002),
        ("RR\\_ref",            -0.682, 0.000),
        ("ESS\\_min\\_ratio",   0.842, 0.000),  # counter-intuitive
        ("SER",                 -0.519, 0.013),
        ("w\\_p99\\_ref",       0.608, 0.003),
    ]
    drkpv_rows = [
        ("kappa\\_r (log10)",   -0.943, 0.005),
        ("SER",                 -0.485, 0.022),
        ("RR\\_ref",            -0.461, 0.031),
        ("residual\\_norm\\_r", -0.407, 0.060),
        ("ESS\\_min\\_ratio",    0.411, 0.058),
        ("w\\_p99\\_ref",       -0.315, 0.154),
    ]
    threshold = 0.6

    fig, (ax_b, ax_d) = plt.subplots(1, 2, figsize=(10.5, 4.8))

    def _plot_bars(ax, rows, title, n_cells_label):
        # Sort by absolute rho descending (most actionable on top)
        rows_sorted = sorted(rows, key=lambda r: abs(r[1]), reverse=True)
        ys = np.arange(len(rows_sorted))
        names = [r[0] for r in rows_sorted]
        rhos = np.array([r[1] for r in rows_sorted])
        ps = np.array([r[2] for r in rows_sorted])
        abs_rhos = np.abs(rhos)
        # Color by sign
        colors = ["#3F88C5" if r > 0 else "#D72638" for r in rhos]
        # Mark counter-intuitive (ESS_min_ratio with positive rho)
        # already handled by colour (positive=blue) and annotation
        bars = ax.barh(ys, abs_rhos, color=colors, edgecolor="black",
                        linewidth=0.6, alpha=0.85)
        # Annotate value + sign + significance
        for i, (n, r, p) in enumerate(zip(names, rhos, ps)):
            sig = "***" if p < 0.001 else ("**" if p < 0.01
                  else ("*" if p < 0.05 else ""))
            txt = f" $\\rho={r:+.2f}${sig}"
            ax.text(abs(r) + 0.01, i, txt, va="center", ha="left",
                      fontsize=8.0,
                      color=("#1F5BA6" if r > 0 else "#7F1F1F"))
            # Counter-intuitive flag (ESS_min_ratio in Bennett)
            if "ESS" in n and r > 0:
                ax.text(0.02, i, "  (counter-intuitive)",
                          va="center", ha="left", fontsize=7.0,
                          color="#7F1F1F", style="italic",
                          transform=ax.transData)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"${n}$" for n in names], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2,
                    alpha=0.7, label=f"Actionable: $|\\rho|>{threshold}$")
        ax.set_xlabel(r"$|\rho_\mathrm{Spearman}|$ (vs bse$_\mathrm{ref}$)",
                       fontsize=9.5)
        ax.set_title(title, fontsize=10.5)
        ax.grid(True, alpha=0.3, axis="x")
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)

    _plot_bars(ax_b, bennett_rows, "Bennett (BIN)",
                 r"22 cells, EXP-A + EXP-C")
    _plot_bars(ax_d, drkpv_rows, "DR-KPV",
                 r"22 cells (EXP-A); $\kappa_r$ from EXP-C $n_\mathrm{cells}{=}5$")

    # Custom legend for sign meaning
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#3F88C5", label=r"$\rho > 0$ (metric $\uparrow\Rightarrow$ bse $\uparrow$)"),
        Patch(color="#D72638", label=r"$\rho < 0$ (metric $\uparrow\Rightarrow$ bse $\downarrow$)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
                ncol=2, fontsize=8.5, framealpha=0.92,
                bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = _FIGS_DGP2_SPEC / "phase19_F_NEW_F6_v2_spectral_corr"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F6 v2] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-H scinde : H-Bennett (P18 vs BIN) + H-DRKPV (P17 vs oracle)
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_H_bennett():
    """2 niveaux par dose (cell n=2000 SNR=0.95):
      - Phase 18 blind (gate-corrige, lambda_h=1e-5)
      - BIN_mh2000 (oracle-tune)
    Bar chart par dose, metric bse."""
    p18 = _load_pkl(_RAW / "phase18" / "bennett" / "stage5"
                     / "S5_n2000_W95_Z95.pkl")
    pBIN = _load_pkl(_RAW / "phase14B" / "wave5_validation"
                      / "snr95_n2000" / "BIN_mh2000_M100.pkl")
    levels = []
    if p18 is not None:
        m = _compute_metrics(p18)
        if m: levels.append(("Phase 18 blind\n(gate-corrected, $\\lambda_h{=}10^{-5}$)",
                              m, "#FFA500"))
    if pBIN is not None:
        m = _compute_metrics(pBIN)
        if m: levels.append(("BIN$_{m_h=2000}$\n(oracle-tuned)",
                              m, "#1A535C"))
    if not levels:
        print("  [F-NEW-H Bennett] missing PKLs"); return None

    fig, (ax_bse, ax_cov) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    n_doses = len(A_GRID)
    bar_w = 0.36
    x = np.arange(n_doses)
    for i, (label, met, color) in enumerate(levels):
        offset = (i - 0.5) * bar_w
        ax_bse.bar(x + offset, met["bse"], width=bar_w, label=label,
                    color=color, edgecolor="black", linewidth=0.5,
                    alpha=0.85)
        ax_cov.bar(x + offset, met["cov"], width=bar_w, label=label,
                    color=color, edgecolor="black", linewidth=0.5,
                    alpha=0.85)
    for ax in (ax_bse, ax_cov):
        ax.set_xticks(x)
        ax.set_xticklabels([f"$a={a:.1f}$" for a in A_GRID], fontsize=9.5)
        ax.set_xlabel("Dose", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    ax_bse.set_ylabel(r"$|\mathrm{bias}|/\widehat{SE}$ (bse)", fontsize=10)
    ax_bse.set_title("Bias-to-SE ratio", fontsize=10.5)
    ax_bse.axhline(0.5, color="gray", linestyle=":", linewidth=0.8,
                    label=r"Acceptable bse $\leq 0.5$")
    ax_cov.set_ylabel(r"Empirical coverage $\widehat{\mathrm{cov}}$", fontsize=10)
    ax_cov.set_title(r"Coverage of nominal $95\%$ CI", fontsize=10.5)
    ax_cov.axhline(0.95, color="black", linestyle="--", linewidth=0.8,
                    alpha=0.7, label="Nominal 0.95")
    ax_cov.set_ylim(0.0, 1.05)

    # Shared legend at bottom
    h, l = ax_bse.get_legend_handles_labels()
    h2, l2 = ax_cov.get_legend_handles_labels()
    # merge: methods from ax_bse, nominal line from ax_cov
    combined_h = h + [hh for hh, ll in zip(h2, l2) if "Nominal" in ll]
    combined_l = l + [ll for ll in l2 if "Nominal" in ll]
    fig.legend(combined_h, combined_l, loc="lower center", ncol=3,
               fontsize=8.5, framealpha=0.92, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.08, 1, 1.0])
    out = _FIGS_DGP2_BLIND / "phase19_F_NEW_H_bennett"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-H Bennett] saved -> {out}.{{pdf,png}}")
    return out


def fig_NEW_H_drkpv():
    """2 niveaux par dose (cell n=2000 SNR=0.95):
      - Phase 17 DRKPV blind cfg2 (lambda_Q=1e-4, blind multi-dose corrected)
      - DRKPV oracle-tune (Phase 16 EXP-A best config = winner DRKPV)
    Fallback: utiliser Phase 16 EXP-A DRKPV PKL = "oracle-blind-equivalent"
    si pas de tuned PKL distinct."""
    p17 = _load_pkl(_RAW / "phase17" / "drkpv" / "stage5"
                     / "S5_n2000_W95_Z95.pkl")
    if p17 is None:
        # alternate path
        for cand in ["phase17/drkpv/stage5/S5_n2000_W95_Z95.pkl",
                      "phase17/stage5/S5_n2000_W95_Z95.pkl",
                      "phase17/state.json"]:
            p = _RAW / cand
            if p.exists():
                if p.suffix == ".pkl":
                    p17 = _load_pkl(p); break
                else:
                    print(f"  [F-NEW-H DRKPV] found {p} but not pkl")
    # Oracle DRKPV = Phase 16 EXP-A DRKPV (well-tuned in production)
    pORA = load_phase16_expa("DRKPV", n=2000, snr=0.95)

    levels = []
    if p17 is not None:
        m = _compute_metrics(p17)
        if m: levels.append(("Phase 17 blind\n(DRKPV cfg2, $\\lambda_Q{=}10^{-4}$)",
                              m, "#FFA500"))
    if pORA is not None:
        m = _compute_metrics(pORA)
        if m: levels.append(("DRKPV oracle\n(production tuning)",
                              m, "#1A535C"))
    if not levels:
        print("  [F-NEW-H DRKPV] missing PKLs"); return None

    fig, (ax_bse, ax_cov) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    n_doses = len(A_GRID)
    bar_w = 0.36
    x = np.arange(n_doses)
    for i, (label, met, color) in enumerate(levels):
        offset = (i - 0.5) * bar_w
        ax_bse.bar(x + offset, met["bse"], width=bar_w, label=label,
                    color=color, edgecolor="black", linewidth=0.5,
                    alpha=0.85)
        ax_cov.bar(x + offset, met["cov"], width=bar_w, label=label,
                    color=color, edgecolor="black", linewidth=0.5,
                    alpha=0.85)
    for ax in (ax_bse, ax_cov):
        ax.set_xticks(x)
        ax.set_xticklabels([f"$a={a:.1f}$" for a in A_GRID], fontsize=9.5)
        ax.set_xlabel("Dose", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    ax_bse.set_ylabel(r"$|\mathrm{bias}|/\widehat{SE}$ (bse)", fontsize=10)
    ax_bse.set_title("Bias-to-SE ratio", fontsize=10.5)
    ax_bse.axhline(0.5, color="gray", linestyle=":", linewidth=0.8,
                    label=r"Acceptable bse $\leq 0.5$")
    ax_cov.set_ylabel(r"Empirical coverage $\widehat{\mathrm{cov}}$", fontsize=10)
    ax_cov.set_title(r"Coverage of nominal $95\%$ CI", fontsize=10.5)
    ax_cov.axhline(0.95, color="black", linestyle="--", linewidth=0.8,
                    alpha=0.7, label="Nominal 0.95")
    ax_cov.set_ylim(0.0, 1.05)

    # Shared legend at bottom
    h, l = ax_bse.get_legend_handles_labels()
    h2, l2 = ax_cov.get_legend_handles_labels()
    combined_h = h + [hh for hh, ll in zip(h2, l2) if "Nominal" in ll]
    combined_l = l + [ll for ll in l2 if "Nominal" in ll]
    fig.legend(combined_h, combined_l, loc="lower center", ncol=3,
               fontsize=8.5, framealpha=0.92, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.08, 1, 1.0])
    out = _FIGS_DGP2_BLIND / "phase19_F_NEW_H_drkpv"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-H DRKPV] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# F-NEW-G v2 : 2 heatmaps cote-a-cote (DRKPV + Bennett identifiability)
# ──────────────────────────────────────────────────────────────────────────────

def fig_NEW_G_v2_dual_heatmaps(lambda_cov: float = 1.0):
    """2 heatmaps cote-a-cote:
      Gauche : DRKernel cross (lambda_h x lambda_Q) -- Phase 19B
      Droite : Bennett cross (lambda_h x lambda_r)   -- Phase 19C (NEW)
    Score composite : bse + |cov - 0.95|.
    Annotations cellule : 'bse | cov | bias'."""
    # ── Phase 19B (DRKPV) ──
    p19B_dir = _RAW / "s7" / "identifiability_drkpv"
    h_lvls = ["under", "super", "over"]
    q_lvls = ["under", "super", "over"]
    h_lambda = {"under": 1e-7, "super": 3e-5, "over": 1e-1}
    q_lambda = {"under": 1e-7, "super": 1e-3, "over": 1e-1}

    drkpv_score = np.full((3, 3), np.nan)
    drkpv_bse = np.full((3, 3), np.nan)
    drkpv_cov = np.full((3, 3), np.nan)
    drkpv_bias = np.full((3, 3), np.nan)
    for i, h in enumerate(h_lvls):
        for j, q in enumerate(q_lvls):
            p = p19B_dir / f"h{h}_q{q}_M50.pkl"
            d = _load_pkl(p)
            if d is None:
                continue
            M = _compute_metrics(d)
            if M is None: continue
            b = M["bse"][REF_IDX]; c = M["cov"][REF_IDX]
            drkpv_bse[i, j] = b
            drkpv_cov[i, j] = c
            drkpv_bias[i, j] = M["bias"][REF_IDX]
            drkpv_score[i, j] = b + lambda_cov * abs(c - 0.95)

    # ── Phase 19C (Bennett) ──
    p19C_dir = _RAW / "s7" / "identifiability_bennett"
    h_lvls_b = ["under", "super", "over"]
    r_lvls = ["under", "super", "over"]
    h_lambda_b = {"under": 1e-7, "super": 1e-5, "over": 1e-1}
    r_lambda = {"under": 1e-5, "super": 1e-2, "over": 1e0}

    bennett_score = np.full((3, 3), np.nan)
    bennett_bse = np.full((3, 3), np.nan)
    bennett_cov = np.full((3, 3), np.nan)
    bennett_bias = np.full((3, 3), np.nan)
    bennett_loaded = False
    if p19C_dir.exists():
        for i, h in enumerate(h_lvls_b):
            for j, r in enumerate(r_lvls):
                p = p19C_dir / f"h{h}_r{r}_M50.pkl"
                d = _load_pkl(p)
                if d is None:
                    continue
                M = _compute_metrics(d)
                if M is None: continue
                bennett_loaded = True
                b = M["bse"][REF_IDX]; c = M["cov"][REF_IDX]
                bennett_bse[i, j] = b
                bennett_cov[i, j] = c
                bennett_bias[i, j] = M["bias"][REF_IDX]
                bennett_score[i, j] = b + lambda_cov * abs(c - 0.95)

    fig, (ax_dr, ax_bn) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    cmap = LinearSegmentedColormap.from_list(
        "score", ["#1A535C", "#4ECDC4", "#FFE66D", "#FFA500", "#D72638"], N=256
    )

    # Common vmax across both heatmaps for honest comparison
    finite_scores = np.r_[drkpv_score[np.isfinite(drkpv_score)],
                          bennett_score[np.isfinite(bennett_score)]]
    vmax = float(np.nanpercentile(finite_scores, 95)) if len(finite_scores) > 0 else 1.0

    def _draw_panel(ax, score, bse, cov, bias, h_lbls, q_lbls,
                      h_lam, q_lam, title, q_axis_label):
        if not np.any(np.isfinite(score)):
            ax.text(0.5, 0.5, "(not yet run)", ha="center", va="center",
                      transform=ax.transAxes, fontsize=12, color="gray",
                      style="italic")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
            return None
        im = ax.imshow(score, cmap=cmap, vmin=0, vmax=vmax,
                        origin="lower", aspect="equal")
        for i in range(len(h_lbls)):
            for j in range(len(q_lbls)):
                if not np.isfinite(score[i, j]):
                    ax.text(j, i, "—", ha="center", va="center",
                              fontsize=14, color="gray")
                    continue
                txt = (rf"$\widehat{{\mathrm{{bse}}}}{{=}}{bse[i,j]:.2f}$"
                        "\n"
                        rf"$\widehat{{\mathrm{{cov}}}}{{=}}{cov[i,j]:.2f}$"
                        "\n"
                        rf"$\widehat{{\mathrm{{bias}}}}{{=}}{bias[i,j]:+.2f}$")
                color = "white" if score[i, j] > vmax * 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                          fontsize=8.5, color=color, fontweight="bold")
        ax.set_xticks(range(len(q_lbls)))
        ax.set_xticklabels([f"{ql}\n${q_lam[ql]:.0e}$" for ql in q_lbls],
                            fontsize=8.5)
        ax.set_yticks(range(len(h_lbls)))
        ax.set_yticklabels([f"{hl}\n${h_lam[hl]:.0e}$" for hl in h_lbls],
                            fontsize=8.5)
        ax.set_xlabel(q_axis_label, fontsize=10)
        ax.set_ylabel(r"$h$-bridge regularisation $\lambda_h$", fontsize=10)
        ax.set_title(title, fontsize=10.5)
        return im

    im1 = _draw_panel(ax_dr, drkpv_score, drkpv_bse, drkpv_cov, drkpv_bias,
                        h_lvls, q_lvls, h_lambda, q_lambda,
                        "DR-KPV: cross $(\\lambda_h, \\lambda_Q)$",
                        r"$q$-bridge regularisation $\lambda_Q$")
    im2 = _draw_panel(ax_bn, bennett_score, bennett_bse, bennett_cov, bennett_bias,
                        h_lvls_b, r_lvls, h_lambda_b, r_lambda,
                        "Bennett: cross $(\\lambda_h, \\lambda_r)$",
                        r"Riesz regularisation $\lambda_r$")

    # Colorbar removed: annotations already report bse|cov|bias
    # in each cell, so the colorbar is redundant. The colour gradient alone
    # suffices for fast reading of the structure.

    fig.suptitle(r"Identifiability of $J(\pi_{a,h})$ across mis-tuned bridges -- "
                  r"DGP2 $n{=}2000$, $\mathrm{SNR}_W{=}\mathrm{SNR}_Z{=}0.95$, $M{=}50$"
                  "\n"
                  r"Cell colour $\propto$ composite score $\widehat{\mathrm{bse}} + |\widehat{\mathrm{cov}} - 0.95|$ "
                  r"(deeper $=$ worse); cell text reports $\widehat{\mathrm{bse}}$ / $\widehat{\mathrm{cov}}$ / $\widehat{\mathrm{bias}}$",
                  fontsize=10)
    out = _FIGS_DGP2_IDENT / "phase19_F_NEW_G_v2_dual_heatmaps"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-NEW-G v2] saved -> {out}.{{pdf,png}}  "
          f"(bennett loaded: {bennett_loaded})")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# DGP3 v2 BIAS-VARIANCE (round 4 — recommended replacement of the U-curve)
# ──────────────────────────────────────────────────────────────────────────────

def fig_DGP3_v2_bias_variance():
    """the sole DGP3 figure retained =
    decomposition bias^2/variance/MSE. LaTeX mathtext partout, rho_true label
    place hors zone des courbes, log-y."""
    import json
    data_path = _BASE / "simulations" / "results" / "raw" / "dgp3_real" / "dgp3_real_data.json"
    if not data_path.exists():
        print(f"  [DGP3 bias-var] missing data {data_path}, abort"); return None
    data = json.loads(data_path.read_text(encoding="utf-8"))
    bloc_a = data.get("bloc_a", {})
    rho_grid = sorted(data.get("rho_policy_grid", []))
    if not bloc_a or not rho_grid:
        print("  [DGP3 bias-var] no data, abort"); return None

    parsed = {}
    for key, val in bloc_a.items():
        parts = key.split("__")
        rho_pol = float(parts[0].split("=", 1)[1])
        method = parts[1]
        parsed[(rho_pol, method)] = val

    rho_true = 0.20
    methods = ["DRKPV", "best_bennett"]
    method_pretty = {
        "DRKPV":        r"DR-KPV",
        "best_bennett": r"Bennett (RFF)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)

    for ax_i, method in enumerate(methods):
        ax = axes[ax_i]
        bias_sq = [parsed.get((r, method), {}).get("bias_to_domain_sq",
                                                      float("nan")) for r in rho_grid]
        var = [parsed.get((r, method), {}).get("variance_J_hat",
                                                  float("nan")) for r in rho_grid]
        mse = [(b + v) for b, v in zip(bias_sq, var)]

        # Theoretical target_mismatch^2 (deterministic lower bound from policy
        # bandwidth choice alone — NOT statistical)
        sin2_diff = np.sin(2.0) - np.sin(0.0)
        J_true_dom = np.exp(-2.0 * rho_true**2) * sin2_diff
        target_mismatch_sq = [(np.exp(-2.0 * r**2) * sin2_diff - J_true_dom)**2
                                for r in rho_grid]

        ax.plot(rho_grid, bias_sq, marker="^", color="#E63946", linewidth=1.7,
                  markersize=6,
                  label=r"$\widehat{\mathrm{bias}}^{\,2}(\rho_\mathrm{pol})$")
        ax.plot(rho_grid, var, marker="v", color="#1A8FE3", linewidth=1.7,
                  markersize=6,
                  label=r"$\widehat{\mathrm{Var}}(\rho_\mathrm{pol})$")
        ax.plot(rho_grid, mse, marker="o", color="#1A535C", linewidth=2.0,
                  markersize=7,
                  label=r"$\widehat{\mathrm{MSE}} = \widehat{\mathrm{bias}}^{\,2} + \widehat{\mathrm{Var}}$")
        ax.plot(rho_grid, target_mismatch_sq, color="#E63946", linestyle="--",
                  alpha=0.55, linewidth=1.3,
                  label=r"$|\,J(\pi_{\rho_\mathrm{pol}}) - J(\pi_{\rho_\mathrm{true}})|^{2}$  (target mismatch, theoretical)")

        # Vertical at rho_true with label PLACED OUTSIDE plot area (top axis)
        ax.axvline(rho_true, color="black", linestyle=":", linewidth=1.0,
                    alpha=0.7)
        # Use axis-fraction coords for the label to avoid clash with curves
        ax.annotate(rf"$\rho_\mathrm{{true}}={rho_true}$",
                      xy=(rho_true, 1.0), xycoords=("data", "axes fraction"),
                      xytext=(4, -8), textcoords="offset points",
                      ha="left", va="top", fontsize=9.5, color="black",
                      bbox=dict(boxstyle="round,pad=0.20",
                                facecolor="white", edgecolor="black",
                                linewidth=0.6, alpha=0.92))

        ax.set_yscale("log")
        ax.set_xlabel(r"Policy bandwidth $\rho_\mathrm{pol}$", fontsize=10.5)
        if ax_i == 0:
            ax.set_ylabel(r"MSE components  (log scale)", fontsize=10.5)
        ax.set_title(method_pretty[method], fontsize=11.5, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        # No per-panel legend -- shared figure legend below

    # Build shared legend handles outside the axes ()
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="^", color="#E63946", linewidth=1.7,
                markersize=6,
                label=r"$\widehat{\mathrm{bias}}^{\,2}(\rho_\mathrm{pol})$"),
        Line2D([0], [0], marker="v", color="#1A8FE3", linewidth=1.7,
                markersize=6,
                label=r"$\widehat{\mathrm{Var}}(\rho_\mathrm{pol})$"),
        Line2D([0], [0], marker="o", color="#1A535C", linewidth=2.0,
                markersize=7,
                label=r"$\widehat{\mathrm{MSE}} = \widehat{\mathrm{bias}}^{\,2} + \widehat{\mathrm{Var}}$"),
        Line2D([0], [0], color="#E63946", linestyle="--", linewidth=1.3,
                alpha=0.55,
                label=r"$|J(\pi_{\rho_\mathrm{pol}}) - J(\pi_{\rho_\mathrm{true}})|^{2}$  (target mismatch, theoretical)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                fontsize=9.0, framealpha=0.92,
                bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.07, 1, 1.0])
    out = _FIGS_S7 / "fig_S7_dgp3_bias_variance"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [DGP3 bias-var] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# DGP3 v2 : single log-log U-curve (LaTeX notations)  -- secondary figure
# ──────────────────────────────────────────────────────────────────────────────

def fig_DGP3_v2_loglog():
    """the sole DGP3 figure retained = U-curve log-log.
    Notations LaTeX visuellement mathematiques: $\\rho_\\mathrm{pol}$,
    $J(\\pi_\\rho)$, $\\rho_\\mathrm{true}$.
    Source: simulations/results/raw/dgp3_real/dgp3_real_data.json (Bloc A)."""
    import json
    data_path = _BASE / "simulations" / "results" / "raw" / "dgp3_real" / "dgp3_real_data.json"
    if not data_path.exists():
        print(f"  [DGP3 v2] missing data {data_path}, abort"); return None
    data = json.loads(data_path.read_text(encoding="utf-8"))
    bloc_a = data.get("bloc_a", {})
    rho_grid = sorted(data.get("rho_policy_grid", []))
    if not bloc_a or not rho_grid:
        print("  [DGP3 v2] no Bloc A data, abort"); return None

    parsed = {}
    for key, val in bloc_a.items():
        parts = key.split("__")
        rho_pol = float(parts[0].split("=", 1)[1])
        method = parts[1]
        parsed[(rho_pol, method)] = val

    methods = ["naive_OLS", "linear_TSLS", "DRKPV", "best_bennett"]
    method_label = {
        "naive_OLS":    r"Naive OLS ($Y \sim B$)",
        "linear_TSLS":  r"Linear 2SLS",
        "DRKPV":        r"DR-KPV (KPV bridges)",
        "best_bennett": r"Bennett (RFF)",
    }
    method_style = {
        "naive_OLS":    {"color": "#A8A8A8", "marker": "x", "linestyle": ":"},
        "linear_TSLS":  {"color": "#FFA500", "marker": "s", "linestyle": "--"},
        "DRKPV":        {"color": "#95E1D3", "marker": "v", "linestyle": "-."},
        "best_bennett": {"color": "#1A535C", "marker": "D", "linestyle": "-"},
    }

    rho_true = 0.20
    sin2_diff = np.sin(2.0) - np.sin(0.0)
    J_true_dom = np.exp(-2.0 * rho_true**2) * sin2_diff
    oracle_mismatch = [abs(np.exp(-2.0 * r**2) * sin2_diff - J_true_dom)
                        for r in rho_grid]

    fig, ax = plt.subplots(figsize=SIZE_PRESETS["double_column"])

    eps = 1e-3  # floor to avoid log(0) for rho near 0
    for m in methods:
        ys = [parsed.get((r, m), {}).get("RMSE_to_domain", float("nan"))
              for r in rho_grid]
        ys_floored = [max(y, eps) if np.isfinite(y) else np.nan for y in ys]
        st = method_style[m]
        ax.plot(rho_grid, ys_floored, label=method_label[m],
                  color=st["color"], marker=st["marker"],
                  linestyle=st["linestyle"], linewidth=1.6, markersize=6)
        if m in ["DRKPV", "best_bennett"]:
            cis_lo = [parsed.get((r, m), {}).get("RMSE_to_domain_ci",
                                                    [np.nan, np.nan])[0]
                       for r in rho_grid]
            cis_hi = [parsed.get((r, m), {}).get("RMSE_to_domain_ci",
                                                    [np.nan, np.nan])[1]
                       for r in rho_grid]
            cis_lo_f = [max(v, eps) if np.isfinite(v) else np.nan for v in cis_lo]
            cis_hi_f = [max(v, eps) if np.isfinite(v) else np.nan for v in cis_hi]
            ax.fill_between(rho_grid, cis_lo_f, cis_hi_f,
                              color=st["color"], alpha=0.18)

    # Oracle target_mismatch lower bound
    om_floored = [max(v, eps) for v in oracle_mismatch]
    ax.plot(rho_grid, om_floored, color="#E63946", linestyle="--",
              linewidth=1.4, alpha=0.85,
              label=r"$|J(\pi_{\rho_\mathrm{pol}}) - J(\pi_{\rho_\mathrm{true}})|$  (target mismatch, lower bound)")

    # Vertical line at rho_true
    ax.axvline(rho_true, color="black", linestyle=":", linewidth=1.0,
                alpha=0.65)
    ax.text(rho_true * 1.05, ax.get_ylim()[1] * 0.85,
              rf"$\rho_\mathrm{{true}}={rho_true}$",
              fontsize=9.5, color="black")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Policy bandwidth $\rho_\mathrm{pol}$", fontsize=10.5)
    ax.set_ylabel(r"RMSE to $J(\pi_{\rho_\mathrm{true}})$  (log scale)",
                    fontsize=10.5)
    ax.set_title(r"DGP3 (coarsened Gaussian, $g(t)=\sin(2t)$): "
                  r"U-curve in policy bandwidth $\rho_\mathrm{pol}$"
                  "\n"
                  rf"$n{{=}}1000$, $M{{=}}100$, bootstrap 95\\% CI, $\rho_\mathrm{{true}}{{=}}{rho_true}$",
                  fontsize=10)
    ax.grid(True, alpha=0.35, which="both")
    ax.legend(loc="upper left", fontsize=8.0, framealpha=0.92)

    fig.tight_layout()
    out = _FIGS_DGP3 / "phase19_DGP3_v2_ucurve_loglog"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [DGP3 v2] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# FUSED / DENSIFIED figures  (round 7 redesign, 2026-05-05)
# ──────────────────────────────────────────────────────────────────────────────

def fig_FUSED_sanity_scale():
    """2x2: top row = MC stabilisation (Bennett BIN, single neutral dark color);
    bottom row = scaling with n (4 methods with palette colors).
    Shared legend at bottom shows only the 4 method colors.
    Saves to figures/dgp2/scaling/phase19_FUSED_sanity_scale.{pdf,png}
    """
    M_grid = [30, 50, 100, 150, 200]
    n_vals  = [250, 500, 1000, 2000]
    n_boot  = 100
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0),
                              gridspec_kw={"hspace": 0.42, "wspace": 0.30})
    ax_mc_bse, ax_mc_cov = axes[0, 0], axes[0, 1]
    ax_n_bse,  ax_n_cov  = axes[1, 0], axes[1, 1]

    # ── Top row: all methods, full color palette ──
    for method in METHODS_PLOT:
        data_m = load_phase16_expa(method, n=2000, snr=0.95)
        if data_m is None:
            continue
        J_dr_m, V_hat_m, _ = _extract_J_dr_seeds(data_m)
        n_m        = data_m["meta"]["n"]
        J_pt_ref_m = data_m["meta"]["J_policy_true"][REF_IDX]
        M_total_m  = J_dr_m.shape[0]
        style  = get_method_style_v2(method)
        label  = METHOD_PRETTY.get(method, method)
        bse_means, bse_los, bse_his = [], [], []
        cov_means, cov_los, cov_his = [], [], []
        for M in M_grid:
            if M > M_total_m:
                for lst in (bse_means, bse_los, bse_his,
                             cov_means, cov_los, cov_his):
                    lst.append(np.nan)
                continue
            bse_boot = np.empty(n_boot)
            cov_boot = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.choice(M_total_m, size=M, replace=False)
                Jd  = J_dr_m[idx, REF_IDX]
                Vh  = V_hat_m[idx, REF_IDX]
                SE  = np.sqrt(np.maximum(Vh / n_m, 0.0))
                bias = Jd.mean() - J_pt_ref_m
                bse_boot[b] = (abs(bias) / np.nanmean(SE)
                               if np.nanmean(SE) > 1e-15 else np.nan)
                lo = Jd - 1.96 * SE; hi = Jd + 1.96 * SE
                cov_boot[b] = float(np.mean((lo <= J_pt_ref_m) & (J_pt_ref_m <= hi)))
            bse_means.append(np.nanmean(bse_boot))
            bse_los.append(np.nanpercentile(bse_boot, 5))
            bse_his.append(np.nanpercentile(bse_boot, 95))
            cov_means.append(np.nanmean(cov_boot))
            cov_los.append(np.nanpercentile(cov_boot, 5))
            cov_his.append(np.nanpercentile(cov_boot, 95))
        ax_mc_bse.plot(M_grid, bse_means, label=label, **style)
        ax_mc_bse.fill_between(M_grid, bse_los, bse_his, alpha=0.12,
                                color=style.get("color", "#555555"))
        ax_mc_cov.plot(M_grid, cov_means, label=label, **style)
        ax_mc_cov.fill_between(M_grid, cov_los, cov_his, alpha=0.12,
                                color=style.get("color", "#555555"))

    for ax in (ax_mc_bse, ax_mc_cov):
        ax.set_xticks(M_grid)
        ax.set_xlabel("Monte-Carlo replications $M$", fontsize=9.5)
        ax.grid(True, alpha=0.3)
    ax_mc_bse.set_ylabel(
        r"$|\mathrm{bias}|/\widehat{SE}$ (bse$_\mathrm{ref}$)", fontsize=9.5)
    ax_mc_bse.set_title("Stabilisation of bse with $M$", fontsize=10)
    ax_mc_cov.set_ylabel("Empirical coverage", fontsize=9.5)
    ax_mc_cov.axhline(0.95, color="#777", linestyle="--", linewidth=0.8,
                       alpha=0.7, label="Nominal 0.95")
    ax_mc_cov.set_title("Stabilisation of coverage with $M$", fontsize=10)
    ax_mc_cov.set_ylim(0.60, 1.02)

    # ── Bottom row: 4 methods, full color palette ──
    for method in METHODS_PLOT:
        bses, covs = [], []
        for n in n_vals:
            data = (load_phase16_expa(method, n=n, snr=0.95)
                    if n in (1000, 2000) else load_phase19A(method, n=n))
            if data is None:
                bses.append(np.nan); covs.append(np.nan); continue
            Met = _compute_metrics(data)
            if Met is None:
                bses.append(np.nan); covs.append(np.nan); continue
            bses.append(Met["bse"][REF_IDX]); covs.append(Met["cov"][REF_IDX])
        style = get_method_style_v2(method)
        label = METHOD_PRETTY.get(method, method)
        ax_n_bse.plot(n_vals, bses, **style)
        ax_n_cov.plot(n_vals, covs, **style)

    for ax in (ax_n_bse, ax_n_cov):
        ax.set_xticks(n_vals)
        ax.set_xticklabels([str(n) for n in n_vals])
        ax.set_xlabel("Sample size $n$", fontsize=9.5)
        ax.grid(True, alpha=0.3)
    ax_n_bse.set_ylabel(
        r"$|\mathrm{bias}|/\widehat{SE}$ (bse$_\mathrm{ref}$)", fontsize=9.5)
    ax_n_bse.set_title("Scaling of bse with $n$", fontsize=10)
    ax_n_cov.set_ylabel("Empirical coverage", fontsize=9.5)
    ax_n_cov.axhline(0.95, color="#777", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_n_cov.set_title("Scaling of coverage with $n$", fontsize=10)
    ax_n_cov.set_ylim(0.45, 1.02)

    h, l = ax_mc_cov.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.5,
               framealpha=0.92, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.06, 1, 1.0])
    out = _FIGS_S7 / "fig_S7_convergence_scale"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [FUSED sanity+scale] saved -> {out}.{{pdf,png}}")
    return out


def fig_FUSED_violin_doses_snr():
    """2 panels side-by-side: boundary a=1.8 (left), centre a=2.6 (right).
    5 methods. Horizontal violin + jitter. Per-panel xlim centred on J_true.
    Saves to figures/dgp2/distributions/phase19_FUSED_violin_doses_snr.{pdf,png}
    """
    cell_n, cell_snr = 2000, 0.95
    doses_idx   = [0, 2]       # a=1.8, a=2.6
    dose_labels = [r"Boundary:  $a=1.8$", r"Centre:  $a=2.6$"]
    methods     = METHODS_PLOT

    method_data: dict = {}
    for m in METHODS_PLOT:
        data = load_phase16_expa(m, n=cell_n, snr=cell_snr)
        if data is not None:
            J_dr, _, _ = _extract_J_dr_seeds(data)
            method_data[m] = (J_dr, data["meta"]["J_policy_true"])

    if not method_data:
        print("  [FUSED violin] no data, abort"); return None

    # sharey=False: each panel gets its own y-axis so per-dose labels differ
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), sharey=False)
    rng = np.random.default_rng(42)
    y_positions = {m: len(methods) - 1 - i for i, m in enumerate(methods)}
    ylim_shared = (-0.6, len(methods) - 0.4)

    for ax_idx, di in enumerate(doses_idx):
        ax   = axes[ax_idx]
        a    = A_GRID[di]
        first_m = next(iter(method_data))
        J_true  = method_data[first_m][1][di]

        local_lo, local_hi = [J_true], [J_true]
        for m, (J_dr, _) in method_data.items():
            if J_dr is None or m not in methods: continue
            x = J_dr[:, di]; x = x[np.isfinite(x)]
            if len(x) == 0: continue
            local_lo.append(np.percentile(x, 1))
            local_hi.append(np.percentile(x, 99))
        x_lo = min(local_lo); x_hi = max(local_hi)
        pad  = 0.15 * (x_hi - x_lo)
        x_lo -= pad; x_hi += pad

        for m in methods:
            if m not in method_data: continue
            J_dr, _ = method_data[m]
            if J_dr is None: continue
            x = J_dr[:, di]; x = x[np.isfinite(x)]
            if len(x) == 0: continue
            yp    = y_positions[m]
            color = get_method_style_v2(m).get("color", "#888")
            parts = ax.violinplot([x], positions=[yp], widths=0.7,
                                   vert=False, showmeans=False,
                                   showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.55)
                pc.set_edgecolor(color); pc.set_linewidth(1.0)
            if "cmedians" in parts:
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(1.6)
            jitter = rng.uniform(-0.14, 0.14, size=len(x))
            ax.scatter(x, np.full_like(x, yp) + jitter,
                        s=5, color="black", alpha=0.30, zorder=3, linewidth=0)

        ax.axvline(J_true, color="black", linestyle="--", linewidth=1.6,
                    alpha=0.85, zorder=4,
                    label=rf"$J_\mathrm{{true}}={J_true:.3f}$")

        ytick_labels = []
        for m in methods:
            if m not in method_data:
                ytick_labels.append(METHOD_PRETTY.get(m, m)); continue
            J_dr, _ = method_data[m]
            if J_dr is None:
                ytick_labels.append(METHOD_PRETTY.get(m, m)); continue
            x = J_dr[:, di]; x = x[np.isfinite(x)]
            if len(x) == 0:
                ytick_labels.append(METHOD_PRETTY.get(m, m)); continue
            bias = float(np.mean(x)) - J_true
            sd   = float(np.std(x))
            ytick_labels.append(
                rf"{METHOD_PRETTY.get(m, m)}"
                "\n"
                rf"$\widehat{{\mathrm{{bias}}}}{{=}}{bias:+.3f}$"
                "\n"
                rf"$\widehat{{\sigma}}{{=}}{sd:.3f}$"
            )

        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(ytick_labels, fontsize=8.5)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(*ylim_shared)
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_title(dose_labels[ax_idx], fontsize=11, fontweight="bold",
                      loc="left", pad=4)
        ax.set_xlabel(r"Estimated $\widehat{J}(\pi_{a,h})$", fontsize=10)
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92)

    fig.tight_layout()
    out = _FIGS_S7 / "fig_S7_violin_doses_snr"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [FUSED violin] saved -> {out}.{{pdf,png}}")
    return out


def fig_FUSED_proxy_winners():
    """4x4 mini-bar grid (80% width) + strip winner-count aggregate (20%).
    Saves to figures/dgp2/asymmetry/phase19_FUSED_proxy_winners.{pdf,png}
    """
    snr_levels   = [0.50, 0.70, 0.85, 0.95]
    n_levels     = len(snr_levels)
    methods_used = METHODS_PLOT

    cell_data: dict = {}
    for snrW in snr_levels:
        for snrZ in snr_levels:
            entries = []
            for m in methods_used:
                d = load_phase16_expb(m, snrW, snrZ)
                if d is None: continue
                Met = _compute_metrics(d)
                if Met is None or not np.isfinite(Met["bse"][REF_IDX]): continue
                entries.append((m, float(Met["bse"][REF_IDX]),
                                  float(Met["cov"][REF_IDX])))
            entries.sort(key=lambda kv: kv[1])
            cell_data[(snrW, snrZ)] = entries

    all_bse = [e[1] for entries in cell_data.values() for e in entries if e[1] > 0]
    ymax = float(np.percentile(all_bse, 99)) * 1.20 if all_bse else 1.0
    ymin = max(1e-3, float(np.min(all_bse)) * 0.5) if all_bse else 1e-3

    method_color = {
        "linearDR": "#D72638", "KPVREG": "#3F88C5",
        "DRKPV": "#FFA500",    "best_bennett": "#1A535C",
    }
    method_short = {
        "linearDR": "lin", "KPVREG": "KPV",
        "DRKPV": "DR",     "best_bennett": "Ben",
    }

    fig = plt.figure(figsize=(13.5, 10.5))
    gs  = fig.add_gridspec(n_levels, n_levels + 1,
                            width_ratios=[1] * n_levels + [0.22],
                            hspace=0.35, wspace=0.30)
    axes_grid = [[fig.add_subplot(gs[i, j]) for j in range(n_levels)]
                  for i in range(n_levels)]
    ax_strip  = fig.add_subplot(gs[:, n_levels])

    # 4×4 mini-bar grid
    for i, snrW in enumerate(reversed(snr_levels)):
        for j, snrZ in enumerate(snr_levels):
            ax      = axes_grid[i][j]
            entries = cell_data.get((snrW, snrZ), [])
            if not entries:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                          transform=ax.transAxes, fontsize=14, color="gray")
                ax.set_xticks([]); ax.set_yticks([]); continue
            n_meth = len(entries)
            xs     = np.arange(n_meth)
            ax.bar(xs, [e[1] for e in entries], width=0.78,
                    color=[method_color.get(e[0], "#888") for e in entries],
                    edgecolor="black", linewidth=0.5)
            alphas = [0.95] + [0.30] * (n_meth - 1)
            for k, bar in enumerate(ax.containers[0]):
                bar.set_alpha(alphas[k])
            ax.set_xticks(xs)
            ax.set_xticklabels(
                [method_short.get(e[0], e[0][:3]) for e in entries], fontsize=8.5)
            ax.set_yscale("log"); ax.set_ylim(ymin, ymax)
            if j > 0:
                ax.tick_params(labelleft=False)
            else:
                ax.tick_params(labelsize=7.5)
            if n_meth >= 2:
                bse_1, bse_2 = entries[0][1], entries[1][1]
                delta_pct = (bse_2 - bse_1) / bse_1 * 100 if bse_1 > 0 else 0.0
                strong = delta_pct >= 20.0
                txt = (rf"$\Delta{{=}}{{+}}{delta_pct:.0f}\%$"
                       if delta_pct < 999 else r"$\Delta\gg$")
                ax.text(0.5, 0.99, txt, transform=ax.transAxes,
                          ha="center", va="top", fontsize=9.5,
                          color="black" if strong else "#777",
                          fontweight="bold" if strong else "normal",
                          style="normal" if strong else "italic",
                          bbox=dict(boxstyle="round,pad=0.20",
                                    facecolor="white", alpha=0.85,
                                    edgecolor="none"))
            winner = entries[0][0]
            ax.add_patch(plt.Rectangle(
                (0.0, 0.93), 1.0, 0.07, transform=ax.transAxes,
                facecolor=method_color.get(winner, "#888"),
                alpha=0.95, edgecolor="none", clip_on=False))
            ax.grid(True, alpha=0.25, axis="y", which="both")
            ax.set_axisbelow(True)

    for j, snrZ in enumerate(snr_levels):
        axes_grid[0][j].set_title(
            rf"SNR$_Z={snrZ:.2f}$", fontsize=10, fontweight="bold")
    for i, snrW in enumerate(reversed(snr_levels)):
        axes_grid[i][0].set_ylabel(
            rf"SNR$_W={snrW:.2f}$" + "\n"
            + r"$\widehat{\mathrm{bse}}_\mathrm{ref}$ (log)",
            fontsize=9, fontweight="bold")

    # Strip aggregate: stacked bar by winner count
    winners: dict = {}
    for entries in cell_data.values():
        if entries:
            w = entries[0][0]
            winners[w] = winners.get(w, 0) + 1

    y_offset = 0
    for m in methods_used:
        count = winners.get(m, 0)
        if count == 0: continue
        ax_strip.bar([0], [count], bottom=[y_offset],
                      color=method_color[m], width=0.7,
                      edgecolor="white", linewidth=1.5)
        label_y = y_offset + count / 2.0
        fs = 9.5 if count >= 2 else 8.5
        ax_strip.text(0.0, label_y,
                       f"{count}\ncells" if count >= 2 else str(count),
                       ha="center", va="center", fontsize=fs,
                       color="white", fontweight="bold")
        y_offset += count

    ax_strip.set_xlim(-0.5, 0.5)
    ax_strip.set_ylim(0, 16)
    ax_strip.set_title("Cell\nwins", fontsize=9, pad=4)
    ax_strip.axis("off")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=method_color[m], label=METHOD_PRETTY.get(m, m),
                edgecolor="black", linewidth=0.5)
        for m in methods_used
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                fontsize=9, framealpha=0.92,
                bbox_to_anchor=(0.5, 0.01),
                title=(r"Solid bar $=$ cell winner;  light bars $=$ runners-up;  "
                       r"$\Delta\%$ above winner $=$ relative gap to 2nd"),
                title_fontsize=8.5)

    fig.tight_layout(rect=[0, 0.06, 1, 1.0])
    out = _FIGS_S7 / "fig_S7_proxy_winners"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [FUSED proxy winners] saved -> {out}.{{pdf,png}}")
    return out


def fig_NEW_G_v3_simplified():
    """2 heatmaps side-by-side: DRKPV (left), Bennett (right).
    Color = composite score bse + |cov - 0.95|. Annotation = bse only.
    Shared colorbar. X = second-bridge lambda, Y = lambda_h.
    origin=lower so under-regularisation (small lambda) is at the bottom.
    Saves to figures/dgp2/identifiability/phase19_F_NEW_G_v3_simplified.{pdf,png}
    """
    h_levels = ["under", "super", "over"]
    q_levels = ["under", "super", "over"]
    h_lbl = {"under": r"$10^{-7}$",
              "super": r"$3\times10^{-5}$",
              "over":  r"$10^{-1}$"}
    q_lbl = {"under": r"$10^{-7}$", "super": r"$10^{-3}$", "over": r"$10^{-1}$"}
    r_lbl = {"under": r"$10^{-5}$", "super": r"$10^{-2}$", "over": r"$10^{0}$"}

    p19B = _RAW / "s7" / "identifiability_drkpv"
    p19C = _RAW / "s7" / "identifiability_bennett"
    if not p19B.exists() and not p19C.exists():
        print("  [F-IDENT simplified] Phase 19B/C data not found, skip"); return None

    drkpv_score = np.full((3, 3), np.nan)
    drkpv_bse   = np.full((3, 3), np.nan)
    benn_score  = np.full((3, 3), np.nan)
    benn_bse    = np.full((3, 3), np.nan)

    for i, h in enumerate(h_levels):
        for j, q in enumerate(q_levels):
            d = _load_pkl(p19B / f"h{h}_q{q}_M50.pkl")
            if d is not None:
                Met = _compute_metrics(d)
                if Met is not None:
                    b = float(Met["bse"][REF_IDX])
                    c = float(Met["cov"][REF_IDX])
                    drkpv_bse[i, j]   = b
                    drkpv_score[i, j] = b + abs(c - 0.95)
            d = _load_pkl(p19C / f"h{h}_r{q}_M50.pkl")
            if d is not None:
                Met = _compute_metrics(d)
                if Met is not None:
                    b = float(Met["bse"][REF_IDX])
                    c = float(Met["cov"][REF_IDX])
                    benn_bse[i, j]   = b
                    benn_score[i, j] = b + abs(c - 0.95)

    all_scores = np.concatenate([
        drkpv_score[np.isfinite(drkpv_score)],
        benn_score[np.isfinite(benn_score)],
    ])
    if len(all_scores) == 0:
        print("  [F-IDENT simplified] no finite scores"); return None
    vmin = 0.0
    vmax = float(np.nanpercentile(all_scores, 95))

    cmap = "RdYlGn_r"
    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.5),
                                       gridspec_kw={"wspace": 0.38})

    def _draw_hm(ax, score, bse, x_labels, title):
        im = ax.imshow(score, cmap=cmap, vmin=vmin, vmax=vmax,
                        origin="lower", aspect="equal")
        ax.set_xticks(range(3)); ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_yticks(range(3))
        ax.set_yticklabels([h_lbl[h] for h in h_levels], fontsize=9)
        ax.set_xlabel(r"Second-bridge $\lambda$", fontsize=9.5)
        ax.set_ylabel(r"$\lambda_h$", fontsize=9.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        for i in range(3):
            for j in range(3):
                v = bse[i, j]
                if np.isfinite(v):
                    dark = score[i, j] > vmax * 0.60
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                              fontsize=10.5, fontweight="bold",
                              color="white" if dark else "black")
        return im

    im1 = _draw_hm(ax_d, drkpv_score, drkpv_bse,
                    [q_lbl[q] for q in q_levels], "DR-KPV")
    im2 = _draw_hm(ax_b, benn_score,  benn_bse,
                    [r_lbl[q] for q in q_levels], "Bennett (BIN)")

    cbar = fig.colorbar(im2, ax=[ax_d, ax_b], fraction=0.035, pad=0.04,
                          label=r"composite score  (bse $+$ $|$cov $-$ 0.95$|$)")
    cbar.ax.tick_params(labelsize=8.5)

    fig.tight_layout()
    out = _FIGS_S7 / "fig_S7_identifiability"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [F-IDENT simplified] saved -> {out}.{{pdf,png}}")
    return out


def fig_FUSED_blind_vs_oracle():
    """2x2 fused blind vs oracle comparison.
    Row 0 = Bennett RFF DR; row 1 = Kernel DR (DRKPV).
    Col 0 = bse by dose (line plot); col 1 = coverage by dose.
    blind = dark crimson (#C0392B), oracle = academic blue (#2471A3). Shared legend at bottom.
    Saves to simulations/results/figures/S7/fig_S7_blind_vs_oracle.{pdf,png}
    """
    from matplotlib.lines import Line2D

    doses      = A_GRID
    x          = np.arange(len(doses))
    col_blind  = "#C0392B"   # dark crimson  — imperfect phase tuning
    col_oracle = "#2471A3"   # academic blue — calibrated reference

    # Bennett: blind = Stage 5 reference cell of the blind tuner;
    # oracle = best_bennett baseline from the coverage_map cell.
    p_ben_blind  = _load_pkl(_RAW / "s7" / "blind_tuning_bennett" / "bennett"
                              / "stage5" / "S5_n2000_W95_Z95.pkl")
    p_ben_oracle = load_phase16_expa("best_bennett", n=2000, snr=0.95)
    m_ben_blind  = _compute_metrics(p_ben_blind)  if p_ben_blind  is not None else None
    m_ben_oracle = _compute_metrics(p_ben_oracle) if p_ben_oracle is not None else None

    # DRKPV: blind = Stage 5 reference cell of the DRKPV blind tuner;
    # oracle = DRKPV baseline from the coverage_map cell.
    p_drk_blind  = _load_pkl(_RAW / "s7" / "blind_tuning_drkpv" / "drkpv"
                              / "stage5" / "S5_n2000_W95_Z95.pkl")
    p_drk_oracle = load_phase16_expa("DRKPV", n=2000, snr=0.95)
    m_drk_blind  = _compute_metrics(p_drk_blind)  if p_drk_blind  is not None else None
    m_drk_oracle = _compute_metrics(p_drk_oracle) if p_drk_oracle is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5),
                              gridspec_kw={"hspace": 0.38, "wspace": 0.30})

    def _panel(ax, m_blind, m_oracle, metric,
               ylabel, nominal=False, bse_thresh=False):
        if m_blind is not None:
            ax.plot(x, m_blind[metric], color=col_blind,
                     marker="o", markersize=7, linewidth=1.8, linestyle="-",
                     zorder=4)
        if m_oracle is not None:
            ax.plot(x, m_oracle[metric], color=col_oracle,
                     marker="D", markersize=7, linewidth=1.8, linestyle="-",
                     zorder=4)
        if nominal:
            ax.axhline(0.95, color="black", linestyle="--", linewidth=0.8,
                        alpha=0.7)
            ax.set_ylim(0.0, 1.08)
        if bse_thresh:
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"$a={a:.1f}$" for a in doses], fontsize=9.5)
        ax.set_xlabel("Dose $a$", fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.grid(True, alpha=0.3, axis="y")

    _panel(axes[0, 0], m_ben_blind, m_ben_oracle, "bse",
           r"$|\mathrm{bias}|/\widehat{SE}$ (bse)", bse_thresh=True)
    _panel(axes[0, 1], m_ben_blind, m_ben_oracle, "cov",
           r"Empirical coverage $\widehat{\mathrm{cov}}$", nominal=True)
    _panel(axes[1, 0], m_drk_blind, m_drk_oracle, "bse",
           r"$|\mathrm{bias}|/\widehat{SE}$ (bse)", bse_thresh=True)
    _panel(axes[1, 1], m_drk_blind, m_drk_oracle, "cov",
           r"Empirical coverage $\widehat{\mathrm{cov}}$", nominal=True)

    # Column titles (top row only)
    axes[0, 0].set_title("Bias-to-SE ratio  (bse)", fontsize=10.5)
    axes[0, 1].set_title(r"Coverage of nominal $95\%$ CI", fontsize=10.5)

    # Row annotations (rotated, left side of bse panels)
    for ax, lbl in [(axes[0, 0], "Bennett RFF DR"),
                     (axes[1, 0], "Kernel DR (DRKPV)")]:
        ax.text(-0.22, 0.5, lbl,
                  transform=ax.transAxes, fontsize=9, rotation=90,
                  va="center", ha="center", fontweight="bold",
                  fontfamily="sans-serif")

    # Shared legend
    legend_handles = [
        Line2D([0], [0], color=col_blind,  marker="o", markersize=7,
                linewidth=1.8, linestyle="-",
                label="blind  (phase tuning)"),
        Line2D([0], [0], color=col_oracle, marker="D", markersize=7,
                linewidth=1.8, linestyle="-",
                label="oracle  (production tuning)"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.8,
                alpha=0.7, label="Nominal 0.95"),
        Line2D([0], [0], color="gray",  linestyle=":",  linewidth=0.8,
                label=r"Acceptable bse $\leq 0.5$"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                fontsize=8.5, framealpha=0.92, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.07, 1, 1.0])
    out = _FIGS_S7 / "fig_S7_blind_vs_oracle"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"  [FUSED blind vs oracle] saved -> {out}.{{pdf,png}}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

FIG_FUNCS = {
    "A": fig_NEW_A_bridge_vs_functional_error,
    "B": fig_NEW_B_stabilisation_mc,
    "C": fig_NEW_C_scaling_n,
    "D": fig_NEW_D_distributions,
    "E": fig_NEW_E_winner_heatmap,
    "F": fig_NEW_F_double_robustness,
    "G": fig_NEW_G_identifiability_heatmap,
    "H": fig_NEW_H_staircase_oracle_gap,
    # v2 refontes (2026-05-04)
    "DV2":  fig_NEW_D_v2_violin,
    "FV2":  fig_NEW_F_v2_scatter,
    "EV2":  fig_NEW_E_v2_winner_heatmap,
    "F6V2": fig_NEW_F6_spectral_v2,
    "HBEN": fig_NEW_H_bennett,
    "HDRK": fig_NEW_H_drkpv,
    "GV2":  fig_NEW_G_v2_dual_heatmaps,
    "DGP3": fig_DGP3_v2_loglog,
    "DGP3BV": fig_DGP3_v2_bias_variance,    # round 4 recommended
    "FBIS":   fig_NEW_F_bis_rescue_biased_cell,
    "EV3":    fig_NEW_E_v3_minibars,        # round 5 recommended
    "FV6":    fig_NEW_F_v6_biasfix_scan,    # round 5 recommended
    "FBISV2": fig_NEW_F_bis_v2_exhibit,     # round 5 recommended
    "FBISV3": fig_NEW_F_bis_v3_biasfix_tau, # round 6 recommended
    "GV3":    fig_NEW_G_v3_h_q_stress,      # round 6 recommended
    # round 7 fused / densified
    "FUSED_BC":     fig_FUSED_sanity_scale,
    "FUSED_VIOLIN": fig_FUSED_violin_doses_snr,
    "FUSED_PROXY":  fig_FUSED_proxy_winners,
    "FUSED_IDENT":  fig_NEW_G_v3_simplified,
    "FUSED_BLIND":  fig_FUSED_blind_vs_oracle,
}


# ──────────────────────────────────────────────────────────────────────────────
# Tables (LaTeX)
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(v, prec=3):
    if v is None or not np.isfinite(v):
        return "---"
    if abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
        return f"{v:.2e}"
    return f"{v:.{prec}f}"


def table_T1_dgp1():
    """T1 : DGP1 6 methods x {bse_ref, cov_ref, sd_J_ref} at n=1000 and n=2000.
    Includes naiveREG as low reference."""
    methods = ["oracle", "naiveREG", "linearREG", "linearDR", "KPVREG",
                "DRKPV", "best_bennett", "best_kallus"]
    n_list = [1000, 2000]

    rows = []
    for m in methods:
        row = {"method": METHOD_PRETTY.get(m, m)}
        for n in n_list:
            data = load_phase15C_E1(m, n=n)
            if data is None:
                row[f"bse_n{n}"] = "---"
                row[f"cov_n{n}"] = "---"
                row[f"sd_n{n}"] = "---"
                continue
            M = _compute_metrics(data)
            if M is None:
                row[f"bse_n{n}"] = "---"
                row[f"cov_n{n}"] = "---"
                row[f"sd_n{n}"] = "---"
                continue
            # DGP1 ref dose: pick centre (a=0)
            a_grid_dgp1 = M["a_grid"]
            ref_di = int(np.argmin(np.abs(a_grid_dgp1)))
            row[f"bse_n{n}"] = _fmt(M["bse"][ref_di], prec=3)
            row[f"cov_n{n}"] = _fmt(M["cov"][ref_di], prec=3)
            row[f"sd_n{n}"] = _fmt(M["sd_J"][ref_di], prec=3)
        rows.append(row)

    # Build LaTeX
    lines = [
        r"% Table T1 -- DGP1 sanity check across 8 methods (incl. naiveREG)",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l|ccc|ccc}",
        r"\toprule",
        r"& \multicolumn{3}{c|}{$n=1000$} & \multicolumn{3}{c}{$n=2000$} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Method & bse$_\mathrm{ref}$ & cov$_\mathrm{ref}$ & sd$_J$ & bse$_\mathrm{ref}$ & cov$_\mathrm{ref}$ & sd$_J$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(f"{row['method']} & {row['bse_n1000']} & {row['cov_n1000']} & {row['sd_n1000']} "
                      f"& {row['bse_n2000']} & {row['cov_n2000']} & {row['sd_n2000']} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{DGP1 (Cobb-Douglas log-linear, $h_0$ analytic). Reference cell: dose $a{\approx}0$, $M=100$ paired seeds. "
        r"\textbf{naiveREG} (no PCI adjustment) shows that without conditioning on $U$, point estimates are catastrophically biased "
        r"and 95\% CIs never cover. All correctly-specified methods (linearDR, DRKPV, Bennett, Kallus) achieve bse $<0.05$ and "
        r"coverage $\geq 0.90$.}",
        r"\label{tab:T1_dgp1_sanity}",
        r"\end{table}",
    ])
    out = _TABLES / "T1_dgp1_sanity.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [T1] saved -> {out}")
    return out


def table_T2_diagnostics():
    """T2 : Top spectral diagnostics by method (Spearman rho with bse_ref).
    Adapted from phase16_metric_audit.md."""
    # Hard-coded from phase16_metric_audit.md (already computed)
    rows = [
        ("Bennett (BIN)", "residual_norm_h",  "+0.94", "<0.001", 22),
        ("Bennett (BIN)", "eff_rank_r",        "-0.92", "<0.001", 22),
        ("Bennett (BIN)", "eff_rank_h",        "-0.89", "<0.001", 22),
        ("Bennett (BIN)", "kappa_h",           "+0.89", "<0.001", 22),
        ("Bennett (BIN)", "kappa_r",           "+0.89", "<0.001", 22),
        ("Bennett (BIN)", "ESS\\_min\\_ratio", "+0.84*", "0.0001*", 22),
        ("DR-KPV",         "log10($\\kappa_r$)", "-0.94", "0.005", 5),
        ("DR-KPV",         "SER",                "-0.49", "0.022", 22),
        ("KPV-REG",        "log10($\\kappa_r$)", "-0.49", "0.329", 6),
        ("KPV-REG",        "SER",                "-0.13", "0.580", 22),
    ]
    lines = [
        r"% Table T2 -- Spectral diagnostics audit (Phase 16 EXP-C, 22 cells)",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l l r r r}",
        r"\toprule",
        r"Method & Diagnostic & $\rho$ vs bse$_\mathrm{ref}$ & $p$ & $n_\mathrm{cells}$ \\",
        r"\midrule",
    ]
    cur = None
    for method, diag, rho, p, n in rows:
        if method != cur:
            if cur is not None:
                lines.append(r"\midrule")
            cur = method
        lines.append(f"{method} & {diag} & {rho} & {p} & {n} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Spearman correlations between candidate in-sample diagnostics and the gold-standard bias/SE ratio. "
        r"\textbf{Bennett (BIN)} has multiple actionable diagnostics ($|\rho|>0.85$, all $p<0.001$). "
        r"$^*$\,ESS\_min\_ratio is \textbf{counter-intuitive} (positive correlation with bse): it measures policy localisation, "
        r"not bridge quality, and must NOT be used as a tuning criterion. "
        r"DR-KPV: $\kappa_r$ is suggestive ($\rho{=}{-}0.94$) but $n_\mathrm{cells}{=}5$ is small. "
        r"KPV-REG: no diagnostic robustly predicts bse.}",
        r"\label{tab:T2_diagnostics}",
        r"\end{table}",
    ])
    out = _TABLES / "T2_diagnostics.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [T2] saved -> {out}")
    return out


def table_T3_blind_vs_ideal():
    """T3 : Bennett cas moyen at reference cell n=2000 SNR=0.95 -- bias, SE, bse, cov, SER
    for Phase 17 (collapsed) / Phase 18 (gate-corrected) / BIN_mh2000 (oracle)."""
    # Load PKLs
    p17 = _load_pkl(_RAW / "phase17" / "bennett" / "stage5" / "S5_n2000_W95_Z95.pkl")
    p18 = _load_pkl(_RAW / "phase18" / "bennett" / "stage5" / "S5_n2000_W95_Z95.pkl")
    pBIN = _load_pkl(_RAW / "phase14B" / "wave5_validation" / "snr95_n2000" / "BIN_mh2000_M100.pkl")

    levels = []
    if p17 is not None:
        m = _compute_metrics(p17)
        if m: levels.append(("Phase 17 blind ($\\lambda_h{=}10^{-6}$, collapsed)", m))
    if p18 is not None:
        m = _compute_metrics(p18)
        if m: levels.append(("Phase 18 blind (gate-corrected)", m))
    if pBIN is not None:
        m = _compute_metrics(pBIN)
        if m: levels.append(("BIN$_{mh2000}$ (oracle-tuned)", m))

    lines = [
        r"% Table T3 -- Bennett blind tuning oracle gap at reference cell",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"Tuning level & Dose & bias & sd$_J$ & mean SE & bse & cov \\",
        r"\midrule",
    ]
    for label, m in levels:
        for di, a in enumerate(A_GRID):
            row_label = label if di == 0 else ""
            lines.append(
                f"{row_label} & $a={a:.1f}$ & {_fmt(m['bias'][di], 4)} & {_fmt(m['sd_J'][di], 4)} "
                f"& {_fmt(m['mean_SE'][di], 4)} & {_fmt(m['bse'][di], 3)} & {_fmt(m['cov'][di], 3)} \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Bennett tuning at reference cell DGP2 $n{=}2000$, SNR$_W{=}$SNR$_Z{=}0.95$, $M{=}100$. "
        r"\textbf{Phase 17 blind} (no interpolation gate) collapsed: at the centre dose $a{=}2.6$, "
        r"bse $=1.08$ (CI almost never covers truth). \textbf{Phase 18 blind} (gate $\lambda_h\times m_h>0.01$) "
        r"reduces sd$_J$ by ${\sim}2.7\times$ but $\ell_h$ remains mis-tuned, so bias is shifted into systematic "
        r"under-prediction at high doses. \textbf{BIN$_{mh2000}$} (oracle-tuned) achieves bse $\leq 0.20$ across all doses. "
        r"The remaining gap quantifies what blind tuning cannot recover: $\ell_h$ requires either a hold-out set "
        r"or physical domain knowledge.}",
        r"\label{tab:T3_blind_vs_ideal}",
        r"\end{table}",
    ])
    out = _TABLES / "T3_blind_vs_ideal.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [T3] saved -> {out}")
    return out


TABLE_FUNCS = {
    "T1": table_T1_dgp1,
    "T2": table_T2_diagnostics,
    "T3": table_T3_blind_vs_ideal,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figs", default="",
                          help="Comma-separated list of fig codes (A,B,C,D,E,F,G,H) or 'all'")
    parser.add_argument("--tables", default="",
                          help="Comma-separated list of table codes (T1,T2,T3) or 'all'")
    args = parser.parse_args()
    set_cambridge_style()

    if args.figs:
        if args.figs.lower() == "all":
            fig_codes = list(FIG_FUNCS.keys())
        else:
            fig_codes = [c.strip().upper() for c in args.figs.split(",")]
        print(f"Generating figures: {fig_codes}")
        print(f"Output dir: {_FIGS}\n")
        for code in fig_codes:
            if code not in FIG_FUNCS:
                print(f"  [WARN] unknown fig code: {code}"); continue
            try:
                FIG_FUNCS[code]()
            except Exception as e:
                import traceback
                print(f"  [ERROR] fig {code} failed: {e}")
                traceback.print_exc()

    if args.tables:
        if args.tables.lower() == "all":
            tab_codes = list(TABLE_FUNCS.keys())
        else:
            tab_codes = [c.strip().upper() for c in args.tables.split(",")]
        print(f"\nGenerating tables: {tab_codes}")
        print(f"Output dir: {_TABLES}\n")
        for code in tab_codes:
            if code not in TABLE_FUNCS:
                print(f"  [WARN] unknown table code: {code}"); continue
            try:
                TABLE_FUNCS[code]()
            except Exception as e:
                import traceback
                print(f"  [ERROR] table {code} failed: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
