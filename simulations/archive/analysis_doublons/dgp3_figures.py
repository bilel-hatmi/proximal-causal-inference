"""
DGP 3 -- Figure generation.

Reads `docs/notes/dgp3_data.json` produced by `dgp3_run.py` and creates
4 PNG figures in `essay/figures/`:

  F-DGP3-1  Toy spectral I_beta(rho) -- 6 panels (3 r x 2 beta)
  F-DGP3-2  Stability x causal cost (3 panels)
            A: |bias|/SE vs rho_policy (stability)
            B: target_mismatch vs rho_policy (causal cost)
            C: RMSE_total vs rho_policy with U-curve at rho_true (bottom line)
  F-DGP3-3  Calibration heatmap (rho_true x rho_policy -> RMSE_dom) per method
  F-DGP3-4  Proxy SNR diagnostic (RMSE vs rho_policy faceted by SNR)

PNG only, 200dpi.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BASE_DIR = Path(__file__).resolve().parents[2]
_DATA_PATH = _BASE_DIR / "docs" / "notes" / "dgp3_data.json"
_FIG_DIR = _BASE_DIR / "essay" / "figures"

METHOD_COLORS = {
    "naive_OLS":              "#A8A8A8",  # grey
    "oracle_h_only":          "#1A535C",  # dark teal (the ideal upper bound)
    "oracle_DR_unclipped":    "#E63946",  # red (unstable!)
    "oracle_DR_clipped":      "#4ECDC4",  # teal
    "oracle_DR_strong_clip":  "#F4A261",  # orange
}
METHOD_LABELS = {
    "naive_OLS":              "naive OLS  (Y~B+W)",
    "oracle_h_only":          "oracle h plug-in",
    "oracle_DR_unclipped":    "oracle DR (no clip)",
    "oracle_DR_clipped":      "oracle DR (clip 5n^.25)",
    "oracle_DR_strong_clip":  "oracle DR (clip 2n^.25)",
}


def _load() -> dict:
    if not _DATA_PATH.exists():
        raise RuntimeError(f"Missing {_DATA_PATH}; run dgp3_run.py first.")
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def _parse_key(s: str, fields: List[str]) -> Dict:
    """Parse string like 'n=1000__rho_pol=0.1__naive_OLS' to dict."""
    out = {}
    parts = s.split("__")
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        else:
            # Last part is method name
            out["method"] = p
    return out


# ══════════════════════════════════════════════════════════════════════════════
# F-DGP3-1: Toy spectral
# ══════════════════════════════════════════════════════════════════════════════

def fig_dgp3_1_spectral(data: dict):
    bloc1 = data.get("bloc1_spectral", {})
    if not bloc1:
        print("  [skip F-DGP3-1] no bloc1 data")
        return

    parsed: Dict[Tuple[float, float, int], float] = {}
    for key, val in bloc1.items():
        # rho=X__r=Y__beta=Z
        d = {}
        for p in key.split("__"):
            k, v = p.split("=", 1)
            d[k] = float(v)
        parsed[(d["rho"], d["r"], int(d["beta"]))] = float(val)

    rho_grid = sorted(set(k[0] for k in parsed.keys()))
    r_grid = sorted(set(k[1] for k in parsed.keys()))
    beta_grid = sorted(set(k[2] for k in parsed.keys()))

    fig, axes = plt.subplots(len(beta_grid), len(r_grid),
                              figsize=(4.0 * len(r_grid), 3.5 * len(beta_grid)),
                              sharex=True)
    if len(beta_grid) == 1:
        axes = axes.reshape(1, -1)
    if len(r_grid) == 1:
        axes = axes.reshape(-1, 1)

    for i, beta in enumerate(beta_grid):
        for j, r in enumerate(r_grid):
            ax = axes[i, j]
            ys = [parsed.get((rho, r, beta), float("nan")) for rho in rho_grid]
            ax.plot(rho_grid, ys, marker="o", color="#1A535C", linewidth=2)
            ax.set_yscale("log")
            ax.set_title(f"r = {r}, beta = {beta}", fontsize=10)
            if i == len(beta_grid) - 1:
                ax.set_xlabel("rho")
            if j == 0:
                ax.set_ylabel("I_beta(rho)  (log scale)")
            ax.grid(True, alpha=0.3)
            # Annotate hard contrast
            y_hard = parsed.get((0.0, r, beta), float("nan"))
            if np.isfinite(y_hard):
                ax.axhline(y_hard, color="#E63946", linestyle="--", alpha=0.5,
                            label=f"hard (rho=0): {y_hard:.1e}")
                ax.legend(fontsize=8, loc="best")

    fig.suptitle("F-DGP3-1 -- Toy spectral I_beta(rho) = sum_k exp(-rho^2 k^2)|Delta_k|^2 k^(2 r beta)\n"
                  "Hard contrast (rho=0) is severe; smoothing controls source norm",
                  fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = _FIG_DIR / "phase_dgp3_F1_spectral.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# F-DGP3-2: Stability + Causal cost + RMSE total (3 panels)
# ══════════════════════════════════════════════════════════════════════════════

def fig_dgp3_2_three_panel(data: dict):
    bloc2 = data.get("bloc2_stability", {})
    if not bloc2:
        print("  [skip F-DGP3-2] no bloc2 data")
        return

    parsed = {}
    for key, val in bloc2.items():
        d = _parse_key(key, ["n", "rho_pol", "method"])
        n = int(d["n"])
        rho_pol = float(d["rho_pol"])
        method = d["method"]
        parsed[(n, rho_pol, method)] = val

    # Filter to n=1000 for the principal figure
    n_focus = 1000
    rho_pol_grid = sorted(set(k[1] for k in parsed.keys()))
    methods_present = sorted(set(k[2] for k in parsed.keys()))
    rho_true = 0.20  # hard-coded from bloc2 setup

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Panel A: Stability (|bias|/SE vs rho_policy) ---
    ax = axes[0]
    for method in methods_present:
        ys = []
        for rho_pol in rho_pol_grid:
            d = parsed.get((n_focus, rho_pol, method), {})
            ys.append(d.get("bse_to_policy", float("nan")))
        ax.plot(rho_pol_grid, ys, marker="o",
                 color=METHOD_COLORS.get(method, "grey"),
                 label=METHOD_LABELS.get(method, method),
                 linewidth=2, markersize=6)
    ax.axvline(rho_true, color="black", linestyle=":", alpha=0.6, linewidth=1)
    ax.text(rho_true + 0.005, ax.get_ylim()[1] * 0.95, f"rho_true={rho_true}",
            fontsize=8, color="black")
    ax.set_xlabel("rho_policy", fontsize=10)
    ax.set_ylabel("|bias to J(pi_rho_policy)| / SE", fontsize=10)
    ax.set_title("Panel A -- Statistical stability", fontsize=11, fontweight="bold")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.grid(True, alpha=0.3)

    # --- Panel B: Causal cost (target mismatch vs rho_policy) ---
    ax = axes[1]
    for method in methods_present:
        ys = []
        for rho_pol in rho_pol_grid:
            d = parsed.get((n_focus, rho_pol, method), {})
            ys.append(abs(d.get("target_mismatch", float("nan"))))
        ax.plot(rho_pol_grid, ys, marker="o",
                 color=METHOD_COLORS.get(method, "grey"),
                 label=METHOD_LABELS.get(method, method),
                 linewidth=2, markersize=6)
    ax.axvline(rho_true, color="black", linestyle=":", alpha=0.6, linewidth=1)
    ax.set_xlabel("rho_policy", fontsize=10)
    ax.set_ylabel("|J(pi_rho_policy) - J(pi_rho_true)|", fontsize=10)
    ax.set_title("Panel B -- Causal cost (target mismatch)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # --- Panel C: Total RMSE (the bottom line) ---
    ax = axes[2]
    for method in methods_present:
        ys = []
        for rho_pol in rho_pol_grid:
            d = parsed.get((n_focus, rho_pol, method), {})
            ys.append(d.get("RMSE_to_domain", float("nan")))
        ax.plot(rho_pol_grid, ys, marker="o",
                 color=METHOD_COLORS.get(method, "grey"),
                 label=METHOD_LABELS.get(method, method),
                 linewidth=2, markersize=6)
    ax.axvline(rho_true, color="black", linestyle=":", alpha=0.6, linewidth=1)
    ax.set_xlabel("rho_policy", fontsize=10)
    ax.set_ylabel("RMSE to J(pi_rho_true=0.2)", fontsize=10)
    ax.set_title("Panel C -- Total error (bottom line)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    fig.suptitle(f"F-DGP3-2 -- Coarsened binary as stochastic intervention (n={n_focus}, M=100)\n"
                  f"Calibrated stochastic policy (rho_policy = rho_true) achieves the lowest total error",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = _FIG_DIR / "phase_dgp3_F2_three_panel.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# F-DGP3-3: Calibration heatmap (rho_true x rho_policy -> RMSE)
# ══════════════════════════════════════════════════════════════════════════════

def fig_dgp3_3_calibration(data: dict):
    bloc4 = data.get("bloc4_calibration", {})
    if not bloc4:
        print("  [skip F-DGP3-3] no bloc4 data")
        return

    parsed = {}
    for key, val in bloc4.items():
        d = _parse_key(key, ["rho_true", "rho_pol", "method"])
        parsed[(float(d["rho_true"]), float(d["rho_pol"]), d["method"])] = val

    methods = sorted(set(k[2] for k in parsed.keys()))
    rho_true_grid = sorted(set(k[0] for k in parsed.keys()))
    rho_pol_grid = sorted(set(k[1] for k in parsed.keys()))

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4.5),
                              sharey=True)
    if n_methods == 1:
        axes = [axes]

    # Global color scale
    all_rmse = []
    for k, v in parsed.items():
        rmse = v.get("RMSE_to_domain", float("nan"))
        if np.isfinite(rmse):
            all_rmse.append(rmse)
    vmax = np.percentile(all_rmse, 95) if all_rmse else 1.0
    vmin = max(np.percentile(all_rmse, 5), 1e-4) if all_rmse else 1e-3

    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=vmin, vmax=vmax)

    for k, method in enumerate(methods):
        ax = axes[k]
        mat = np.full((len(rho_true_grid), len(rho_pol_grid)), np.nan)
        for i, rho_t in enumerate(rho_true_grid):
            for j, rho_p in enumerate(rho_pol_grid):
                d = parsed.get((rho_t, rho_p, method), {})
                rmse = d.get("RMSE_to_domain", float("nan"))
                mat[i, j] = rmse

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", norm=norm,
                        origin="lower")
        ax.set_xticks(range(len(rho_pol_grid)))
        ax.set_xticklabels([str(r) for r in rho_pol_grid])
        ax.set_yticks(range(len(rho_true_grid)))
        ax.set_yticklabels([str(r) for r in rho_true_grid])
        ax.set_xlabel("rho_policy")
        if k == 0:
            ax.set_ylabel("rho_true")
        ax.set_title(method, fontsize=10, fontweight="bold")

        # Highlight diagonal (calibration optimum)
        for i, rho_t in enumerate(rho_true_grid):
            for j, rho_p in enumerate(rho_pol_grid):
                v = mat[i, j]
                if np.isfinite(v):
                    txt = f"{v:.3g}" if v < 0.01 else f"{v:.3f}"
                    color = "white" if v > vmax * 0.5 else "black"
                    weight = "bold" if rho_t == rho_p else "normal"
                    ax.text(j, i, txt, ha="center", va="center",
                            color=color, fontsize=8, fontweight=weight)
                # Mark diagonal
                if rho_t == rho_p:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False,
                        edgecolor="cyan", linewidth=2, linestyle="--"
                    ))

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RMSE (log)")

    fig.suptitle("F-DGP3-3 -- Calibration heatmap: RMSE_to_domain by (rho_true, rho_policy)\n"
                  "Cyan diagonal = calibrated estimator (rho_policy = rho_true). Bold = optimum.",
                  fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = _FIG_DIR / "phase_dgp3_F3_calibration.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# F-DGP3-4: Proxy SNR diagnostic
# ══════════════════════════════════════════════════════════════════════════════

def fig_dgp3_4_snr_proxy(data: dict):
    bloc3 = data.get("bloc3_snr_proxy", {})
    if not bloc3:
        print("  [skip F-DGP3-4] no bloc3 data")
        return

    parsed = {}
    for key, val in bloc3.items():
        d = _parse_key(key, ["snr", "rho_pol", "method"])
        parsed[(float(d["snr"]), float(d["rho_pol"]), d["method"])] = val

    snr_grid = sorted(set(k[0] for k in parsed.keys()), reverse=True)
    rho_pol_grid = sorted(set(k[1] for k in parsed.keys()))
    methods = sorted(set(k[2] for k in parsed.keys()))

    fig, axes = plt.subplots(1, len(snr_grid), figsize=(5 * len(snr_grid), 4.5),
                              sharey=True)
    if len(snr_grid) == 1:
        axes = [axes]

    for k, snr in enumerate(snr_grid):
        ax = axes[k]
        for method in methods:
            ys = []
            for rho_pol in rho_pol_grid:
                d = parsed.get((snr, rho_pol, method), {})
                ys.append(d.get("RMSE_to_domain", float("nan")))
            ax.plot(rho_pol_grid, ys, marker="o",
                     color=METHOD_COLORS.get(method, "grey"),
                     label=METHOD_LABELS.get(method, method),
                     linewidth=2, markersize=6)
        ax.set_xlabel("rho_policy")
        if k == 0:
            ax.set_ylabel("RMSE to J(pi_rho_true)")
        ax.set_title(f"SNR_proxy = {snr}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods),
                bbox_to_anchor=(0.5, 1.02), fontsize=9)

    fig.suptitle("F-DGP3-4 -- Proxy SNR alignment diagnostic (rho_true=0.2, n=1000)\n"
                  "Smoothing rho_policy helps less when ill-posedness comes from proxy weakness",
                  fontsize=11, fontweight="bold", y=1.06)
    fig.tight_layout()
    out = _FIG_DIR / "phase_dgp3_F4_snr_proxy.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== DGP3 figures ===")
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()

    print("[1/4] F1 spectral...")
    fig_dgp3_1_spectral(data)

    print("[2/4] F2 three-panel...")
    fig_dgp3_2_three_panel(data)

    print("[3/4] F3 calibration heatmap...")
    fig_dgp3_3_calibration(data)

    print("[4/4] F4 SNR proxy...")
    fig_dgp3_4_snr_proxy(data)

    print(f"\n[DONE] All DGP3 figures in {_FIG_DIR}")


if __name__ == "__main__":
    main()
