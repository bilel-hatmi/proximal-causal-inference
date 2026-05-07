"""
DGP 3 v2 -- Figure generation for REAL estimation experiment.

Reads `docs/notes/dgp3_real_data.json` produced by `dgp3_real_estimation.py`
and creates 3 PNG figures in `essay/figures/`:

  F-DGP3v2-1  Bloc A: U-curve with bootstrap CI
              Lines for naive_OLS, linear_TSLS, best_bennett (with CI band)
              Vertical line at rho_true = 0.20

  F-DGP3v2-2  Bloc B: Heatmap of best_bennett RMSE by (rho_true, rho_policy)
              Diagonal highlighted; argmin marked

  F-DGP3v2-3  Two-panel summary: (left) RMSE U-curve, (right) coverage curves
              Pour synthese visuelle.

PNG only, 200dpi.
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
    "best_bennett":  "#1A535C",   # dark teal
    "oracle_truth":  "#E63946",   # red dashed
}
METHOD_LABELS = {
    "naive_OLS":    "naive OLS  (Y~B)",
    "linear_TSLS":  "linear TSLS  (Y~A_hat+W via Z)",
    "best_bennett": "best_bennett  (calibrated bandwidth)",
    "oracle_truth": "oracle truth  (analytic ref)",
}

RHO_TRUE_FOCUS = 0.20


def _load() -> dict:
    if not _DATA_PATH.exists():
        raise RuntimeError(f"Missing {_DATA_PATH}; run dgp3_real_estimation.py first.")
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def _parse_a_key(s: str):
    """Parse 'rho_pol=X__method' to (X, method)."""
    parts = s.split("__")
    rho_pol = float(parts[0].split("=", 1)[1])
    method = parts[1]
    return rho_pol, method


def _parse_b_key(s: str):
    """Parse 'rho_true=X__rho_pol=Y' to (X, Y)."""
    parts = s.split("__")
    rho_true = float(parts[0].split("=", 1)[1])
    rho_pol = float(parts[1].split("=", 1)[1])
    return rho_true, rho_pol


# ══════════════════════════════════════════════════════════════════════════════
#  F1 -- Bloc A: U-curve with CI
# ══════════════════════════════════════════════════════════════════════════════

def fig1_ucurve(data: dict):
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
    methods_present = ["naive_OLS", "linear_TSLS", "best_bennett"]

    # Compute oracle truth values
    # J(pi_rho_pol) = exp(-rho^2/2) * sin(1)
    # Total error of oracle_truth = J_true(rho_pol) - J_true(rho_true=0.20)
    # so RMSE_dom of oracle_truth = |J(pi_rho_pol) - J(pi_rho_true)|
    sin_1 = float(np.sin(1.0))
    J_true_pol = [np.exp(-r**2 / 2.0) * sin_1 for r in rho_grid]
    J_true_dom = np.exp(-RHO_TRUE_FOCUS**2 / 2.0) * sin_1
    oracle_target_mismatch = [abs(jp - J_true_dom) for jp in J_true_pol]

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods_present:
        ys = []
        ys_lo = []
        ys_hi = []
        for rho_pol in rho_grid:
            d = parsed.get((rho_pol, method), {})
            ys.append(d.get("RMSE_to_domain", float("nan")))
            ci = d.get("RMSE_to_domain_ci", [float("nan"), float("nan")])
            ys_lo.append(ci[0])
            ys_hi.append(ci[1])
        ys = np.array(ys)
        ys_lo = np.array(ys_lo)
        ys_hi = np.array(ys_hi)
        ax.plot(rho_grid, ys, marker="o",
                 color=METHOD_COLORS[method],
                 label=METHOD_LABELS[method],
                 linewidth=2, markersize=7)
        # CI band only for best_bennett (others are constant)
        if method == "best_bennett":
            ax.fill_between(rho_grid, ys_lo, ys_hi,
                              color=METHOD_COLORS[method], alpha=0.2,
                              label=f"{METHOD_LABELS[method]} 95% bootstrap CI")

    # Oracle truth mismatch line (NOT an estimator's RMSE, just |target_mismatch|)
    ax.plot(rho_grid, oracle_target_mismatch, color=METHOD_COLORS["oracle_truth"],
             linestyle="--", linewidth=1.5, alpha=0.7,
             label="|J(pi_rho_pol) - J(pi_rho_true)|  (target mismatch only)")

    # Vertical line at rho_true
    ax.axvline(RHO_TRUE_FOCUS, color="black", linestyle=":", alpha=0.6, linewidth=1.5)
    ax.text(RHO_TRUE_FOCUS + 0.005, ax.get_ylim()[1] * 0.95,
             f"rho_true = {RHO_TRUE_FOCUS}",
             fontsize=10, color="black")

    ax.set_xlabel("rho_policy  (estimator's policy bandwidth)", fontsize=11)
    ax.set_ylabel("RMSE to J(pi_rho_true=0.20)  (operational target)", fontsize=11)
    ax.set_title("F-DGP3v2-1 -- Calibration U-curve with REAL estimation\n"
                  "n=1000, M=100, bootstrap 95% CI",
                  fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_yscale("linear")

    fig.tight_layout()
    out = _FIG_DIR / "phase_dgp3v2_F1_ucurve.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  F2 -- Bloc B: Heatmap of best_bennett RMSE
# ══════════════════════════════════════════════════════════════════════════════

def fig2_heatmap(data: dict):
    bloc_b_raw = data.get("bloc_b", {})
    rho_grid = data.get("rho_policy_grid", [])
    if not bloc_b_raw or not rho_grid:
        print("  [skip F2] no Bloc B data")
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

    fig, ax = plt.subplots(figsize=(12, 4.5))
    from matplotlib.colors import LogNorm
    valid = mat[np.isfinite(mat)]
    if len(valid) == 0:
        print("  [skip F2] no valid values")
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
    ax.set_title("F-DGP3v2-2 -- best_bennett RMSE_to_domain by (rho_true, rho_policy)\n"
                  "Cyan dashed = closest grid rho_policy to rho_true",
                  fontsize=11, fontweight="bold")

    # Annotate values
    for i, rho_t in enumerate(rho_true_grid):
        # Find closest grid rho_pol to rho_t
        closest_idx = int(np.argmin(np.abs(np.array(rho_pol_grid) - rho_t)))
        # Find argmin in this row
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
            # Mark closest-to-rho_true
            if j == closest_idx:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor="cyan", linewidth=2, linestyle="--"
                ))

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RMSE (log)")

    fig.tight_layout()
    out = _FIG_DIR / "phase_dgp3v2_F2_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  F3 -- Two-panel summary: U-curve + coverage
# ══════════════════════════════════════════════════════════════════════════════

def fig3_summary(data: dict):
    bloc_a_raw = data.get("bloc_a", {})
    rho_grid = data.get("rho_policy_grid", [])
    if not bloc_a_raw or not rho_grid:
        print("  [skip F3] no Bloc A data")
        return

    parsed = {}
    for key, val in bloc_a_raw.items():
        rho_pol, method = _parse_a_key(key)
        parsed[(rho_pol, method)] = val

    rho_grid = sorted(rho_grid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel A: RMSE U-curve (zoom) ---
    ax = axes[0]
    methods = ["naive_OLS", "linear_TSLS", "best_bennett"]
    for method in methods:
        ys = [parsed.get((r, method), {}).get("RMSE_to_domain", float("nan")) for r in rho_grid]
        ax.plot(rho_grid, ys, marker="o",
                 color=METHOD_COLORS[method], label=METHOD_LABELS[method],
                 linewidth=2, markersize=6)
        if method == "best_bennett":
            cis_lo = [parsed.get((r, method), {}).get("RMSE_to_domain_ci", [np.nan, np.nan])[0] for r in rho_grid]
            cis_hi = [parsed.get((r, method), {}).get("RMSE_to_domain_ci", [np.nan, np.nan])[1] for r in rho_grid]
            ax.fill_between(rho_grid, cis_lo, cis_hi, color=METHOD_COLORS[method], alpha=0.2)
    ax.axvline(RHO_TRUE_FOCUS, color="black", linestyle=":", alpha=0.6)
    ax.set_xlabel("rho_policy", fontsize=11)
    ax.set_ylabel("RMSE to J(pi_rho_true=0.20)", fontsize=11)
    ax.set_title("Panel A -- Total error (RMSE) with 95% CI", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # --- Panel B: Coverage to policy ---
    ax = axes[1]
    for method in methods:
        ys = [parsed.get((r, method), {}).get("cov_to_policy", float("nan")) for r in rho_grid]
        ax.plot(rho_grid, ys, marker="o",
                 color=METHOD_COLORS[method], label=METHOD_LABELS[method],
                 linewidth=2, markersize=6)
    ax.axvline(RHO_TRUE_FOCUS, color="black", linestyle=":", alpha=0.6)
    ax.axhline(0.95, color="green", linestyle="--", alpha=0.5, label="nominal 0.95")
    ax.set_xlabel("rho_policy", fontsize=11)
    ax.set_ylabel("Coverage of J(pi_rho_pol) at 95% nominal", fontsize=11)
    ax.set_title("Panel B -- Coverage at the policy target", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.0, 1.05])
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("F-DGP3v2-3 -- Summary: REAL estimation under coarsened binary intervention",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()

    out = _FIG_DIR / "phase_dgp3v2_F3_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== DGP3 v2 figures ===")
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    print("[1/3] F1 U-curve...")
    fig1_ucurve(data)
    print("[2/3] F2 heatmap...")
    fig2_heatmap(data)
    print("[3/3] F3 summary panel...")
    fig3_summary(data)
    print(f"\n[DONE] All DGP3 v2 figures in {_FIG_DIR}")


if __name__ == "__main__":
    main()
