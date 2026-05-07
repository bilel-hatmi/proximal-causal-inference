"""
DGP 3 v3 -- Figure generation for sin(2t) experiment with bias-variance decomp.

Reads `docs/notes/dgp3_real_data.json` (produced by dgp3_real_estimation.py with
g_kind=sin2t and 4 PCI methods including DRKPV).

Figures:
  F-DGP3v3-1  Bias-variance decomposition (the smoking gun)
              For DRKPV and best_bennett: bias^2(rho_pol), variance(rho_pol),
              MSE(rho_pol). Shows the U emerges from the trade-off.

  F-DGP3v3-2  U-curve with all methods + bootstrap CI
              naive_OLS, linear_TSLS, DRKPV, best_bennett, oracle reference.
              Vertical line at rho_true. Annotation of "binary functional curse"
              zone (small rho_pol).

  F-DGP3v3-3  Heatmap robustness (Bloc B): best_bennett RMSE over (rho_true, rho_pol).
              Diagonal optimality.

PNG only, 200 dpi.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BASE_DIR = Path(__file__).resolve().parents[2]
_DATA_PATH = _BASE_DIR / "docs" / "notes" / "dgp3_real_data.json"
_FIG_DIR = _BASE_DIR / "essay" / "figures"

METHOD_COLORS = {
    "naive_OLS":     "#A8A8A8",   # grey
    "linear_TSLS":   "#FFA500",   # orange
    "DRKPV":         "#95E1D3",   # light teal
    "best_bennett":  "#1A535C",   # dark teal
    "oracle_truth":  "#E63946",   # red
}
METHOD_LABELS = {
    "naive_OLS":    "naive OLS  (Y ~ B)",
    "linear_TSLS":  "linear TSLS  (PCI-aware, linear)",
    "DRKPV":        "DRKPV  (KPV bridges)",
    "best_bennett": "best_bennett  (Bennett RFF)",
    "oracle_truth": "J(pi_rho_pol)  analytic ref.",
}

RHO_TRUE_FOCUS = 0.20


def _load() -> dict:
    if not _DATA_PATH.exists():
        raise RuntimeError(f"Missing {_DATA_PATH}; run dgp3_real_estimation.py first.")
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def _parse_a_key(s: str):
    parts = s.split("__")
    rho_pol = float(parts[0].split("=", 1)[1])
    method = parts[1]
    return rho_pol, method


def _parse_b_key(s: str):
    parts = s.split("__")
    rho_true = float(parts[0].split("=", 1)[1])
    rho_pol = float(parts[1].split("=", 1)[1])
    return rho_true, rho_pol


# ══════════════════════════════════════════════════════════════════════════════
# F1 -- Bias-variance decomposition (the key figure)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_bias_variance(data: dict):
    bloc_a_raw = data.get("bloc_a", {})
    rho_grid = data.get("rho_policy_grid", [])
    if not bloc_a_raw or not rho_grid:
        print("  [skip F1] no Bloc A data")
        return

    parsed: Dict[Tuple[float, str], dict] = {}
    for key, val in bloc_a_raw.items():
        rho_pol, method = _parse_a_key(key)
        parsed[(rho_pol, method)] = val

    rho_grid = sorted(rho_grid)
    methods = ["DRKPV", "best_bennett"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_i, method in enumerate(methods):
        ax = axes[ax_i]
        bias_sq = [parsed.get((r, method), {}).get("bias_to_domain_sq", float("nan")) for r in rho_grid]
        var = [parsed.get((r, method), {}).get("variance_J_hat", float("nan")) for r in rho_grid]
        mse = [(b + v) for b, v in zip(bias_sq, var)]

        ax.plot(rho_grid, bias_sq, marker="^", color="#E63946", linewidth=2,
                 markersize=7, label="bias^2(rho_pol)")
        ax.plot(rho_grid, var, marker="v", color="#1A8FE3", linewidth=2,
                 markersize=7, label="variance(rho_pol)")
        ax.plot(rho_grid, mse, marker="o", color="#1A535C", linewidth=2.5,
                 markersize=8, label="MSE = bias^2 + var")

        # Theoretical reference: target_mismatch^2 = (J(pi_rho) - J(pi_rho_true))^2
        # For sin(2t): J(pi_rho) = exp(-2 rho^2) (sin(2 mu_1) - sin(2 mu_0))
        sin2_diff = np.sin(2.0) - np.sin(0.0)
        J_true_dom_th = np.exp(-2.0 * RHO_TRUE_FOCUS**2) * sin2_diff
        target_mismatch_sq_th = [(np.exp(-2.0 * r**2) * sin2_diff - J_true_dom_th) ** 2
                                  for r in rho_grid]
        ax.plot(rho_grid, target_mismatch_sq_th, color="#E63946", linestyle="--",
                 alpha=0.5, linewidth=1.5,
                 label="theoretical target_mismatch^2")

        ax.axvline(RHO_TRUE_FOCUS, color="black", linestyle=":", alpha=0.5)
        ax.text(RHO_TRUE_FOCUS + 0.005, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 1e-3 else 1e-3,
                f"rho_true={RHO_TRUE_FOCUS}", fontsize=9)
        ax.set_yscale("log")
        ax.set_xlabel("rho_policy", fontsize=11)
        ax.set_ylabel("bias^2 / variance / MSE  (log scale)", fontsize=11)
        ax.set_title(f"{method}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9, loc="upper center", ncol=2)

    fig.suptitle("F-DGP3v3-1 -- Bias-variance trade-off (g = sin(2t), n=1000, M=100)\n"
                  "U-curve in MSE emerges from variance decreasing + bias increasing in rho_pol",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()

    out = _FIG_DIR / "phase_dgp3v3_F1_bias_variance.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# F2 -- U-curve all methods with CI
# ══════════════════════════════════════════════════════════════════════════════

def fig2_ucurve_all(data: dict):
    bloc_a_raw = data.get("bloc_a", {})
    rho_grid = data.get("rho_policy_grid", [])
    if not bloc_a_raw or not rho_grid:
        print("  [skip F2] no Bloc A data")
        return

    parsed = {}
    for key, val in bloc_a_raw.items():
        rho_pol, method = _parse_a_key(key)
        parsed[(rho_pol, method)] = val

    rho_grid = sorted(rho_grid)
    methods_present = ["naive_OLS", "linear_TSLS", "DRKPV", "best_bennett"]

    # Oracle reference target_mismatch (sin2t)
    sin2_diff = np.sin(2.0) - np.sin(0.0)
    J_true_dom = np.exp(-2.0 * RHO_TRUE_FOCUS**2) * sin2_diff
    oracle_mismatch = [abs(np.exp(-2.0 * r**2) * sin2_diff - J_true_dom) for r in rho_grid]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for method in methods_present:
        ys = [parsed.get((r, method), {}).get("RMSE_to_domain", float("nan")) for r in rho_grid]
        ax.plot(rho_grid, ys, marker="o", color=METHOD_COLORS[method],
                 label=METHOD_LABELS[method], linewidth=2, markersize=6)
        # CI bands for PCI methods
        if method in ["DRKPV", "best_bennett"]:
            cis_lo = [parsed.get((r, method), {}).get("RMSE_to_domain_ci", [np.nan, np.nan])[0] for r in rho_grid]
            cis_hi = [parsed.get((r, method), {}).get("RMSE_to_domain_ci", [np.nan, np.nan])[1] for r in rho_grid]
            ax.fill_between(rho_grid, cis_lo, cis_hi,
                              color=METHOD_COLORS[method], alpha=0.15)

    # Oracle reference (target_mismatch only)
    ax.plot(rho_grid, oracle_mismatch, color=METHOD_COLORS["oracle_truth"],
             linestyle="--", alpha=0.7, linewidth=1.5,
             label="|target_mismatch|  (lower bound from rho_pol choice)")

    # Vertical line at rho_true
    ax.axvline(RHO_TRUE_FOCUS, color="black", linestyle=":", alpha=0.6, linewidth=1.5)
    y_top = ax.get_ylim()[1]
    ax.text(RHO_TRUE_FOCUS + 0.01, y_top * 0.7, f"rho_true = {RHO_TRUE_FOCUS}",
             fontsize=10, color="black")

    # Highlight "binary functional curse" zone (small rho_pol)
    ax.axvspan(0, 0.05, color="red", alpha=0.05)
    ax.text(0.025, y_top * 0.55,
             "binary\nfunctional\ncurse",
             fontsize=8, color="darkred", ha="center", style="italic")

    ax.set_xlabel("rho_policy  (estimator's policy bandwidth)", fontsize=11)
    ax.set_ylabel("RMSE to J(pi_rho_true=0.20)", fontsize=11)
    ax.set_title("F-DGP3v3-2 -- U-curve in real PCI estimation\n"
                  "g = sin(2t), n=1000, M=100, bootstrap 95% CI",
                  fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out = _FIG_DIR / "phase_dgp3v3_F2_ucurve_all.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# F3 -- Heatmap (Bloc B)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_heatmap_robustness(data: dict):
    bloc_b_raw = data.get("bloc_b", {})
    rho_grid = data.get("rho_policy_grid", [])
    if not bloc_b_raw or not rho_grid:
        print("  [skip F3] no Bloc B data")
        return

    parsed: Dict[Tuple[float, float], dict] = {}
    for key, val in bloc_b_raw.items():
        rho_true, rho_pol = _parse_b_key(key)
        parsed[(rho_true, rho_pol)] = val

    rho_true_grid = sorted(set(k[0] for k in parsed.keys()))
    rho_pol_grid = sorted(rho_grid)

    mat = np.full((len(rho_true_grid), len(rho_pol_grid)), np.nan)
    for i, rho_t in enumerate(rho_true_grid):
        for j, rho_p in enumerate(rho_pol_grid):
            d = parsed.get((rho_t, rho_p), {})
            mat[i, j] = d.get("RMSE_to_domain", float("nan"))

    fig, ax = plt.subplots(figsize=(13, 5))
    from matplotlib.colors import LogNorm
    valid = mat[np.isfinite(mat)]
    if len(valid) == 0:
        print("  [skip F3] no valid values")
        return
    vmin = max(np.percentile(valid, 5), 1e-3)
    vmax = np.percentile(valid, 95)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                    norm=LogNorm(vmin=vmin, vmax=vmax), origin="lower")
    ax.set_xticks(range(len(rho_pol_grid)))
    ax.set_xticklabels([str(r) for r in rho_pol_grid])
    ax.set_yticks(range(len(rho_true_grid)))
    ax.set_yticklabels([str(r) for r in rho_true_grid])
    ax.set_xlabel("rho_policy")
    ax.set_ylabel("rho_true")
    ax.set_title("F-DGP3v3-3 -- best_bennett RMSE_to_domain heatmap\n"
                  "Cyan dashed = closest grid rho_pol to rho_true (calibrated)",
                  fontsize=11, fontweight="bold")

    # Annotate
    for i, rho_t in enumerate(rho_true_grid):
        closest_idx = int(np.argmin(np.abs(np.array(rho_pol_grid) - rho_t)))
        valid_mask = np.isfinite(mat[i, :])
        if not valid_mask.any():
            continue
        argmin_idx = int(np.where(valid_mask)[0][np.argmin(mat[i, valid_mask])])
        for j, rho_p in enumerate(rho_pol_grid):
            v = mat[i, j]
            if np.isfinite(v):
                txt = f"{v:.3g}" if v < 0.01 else f"{v:.3f}"
                color = "white" if v > vmax * 0.5 else "black"
                weight = "bold" if j == argmin_idx else "normal"
                ax.text(j, i, txt, ha="center", va="center",
                         color=color, fontsize=8, fontweight=weight)
            if j == closest_idx:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor="cyan", linewidth=2, linestyle="--"
                ))

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RMSE (log)")
    fig.tight_layout()

    out = _FIG_DIR / "phase_dgp3v3_F3_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== DGP3 v3 figures ===")
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    print("[1/3] F1 bias-variance decomposition...")
    fig1_bias_variance(data)
    print("[2/3] F2 U-curve all methods...")
    fig2_ucurve_all(data)
    print("[3/3] F3 robustness heatmap...")
    fig3_heatmap_robustness(data)
    print(f"\n[DONE] All DGP3 v3 figures in {_FIG_DIR}")


if __name__ == "__main__":
    main()
