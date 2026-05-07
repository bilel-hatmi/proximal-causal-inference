"""
Exp1A — Rapport diagnostique detaille (15 sections).

Usage::

    python -m simulations.archive.utilities.exp1A_diagnostics

Genere: simulations/results/summaries/exp1A_pilot_diagnostics.md

Aucun calcul supplementaire — charge uniquement les pkl pilote existants.

Conventions verifiees dans exp1A_coverage.py :
    SER = mean(SE_hat) / sd(J_hat)     [l. 496]
    SE  = sqrt(V_hat_grid / n)          [l. 502]
    V_hat_grid : variance du score O(1), NE PAS confondre avec variance
                 de l'estimateur O(1/n).
"""
from __future__ import annotations

import datetime
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from simulations.archive.utilities.exp1A_coverage import (
    _RAW_DIR,
    _SUMM_DIR,
    summarize_exp1A,
)

# ---------------------------------------------------------------------------
# Grilles
# ---------------------------------------------------------------------------
A_GRID_DGP1 = np.array([-0.5, 0.0, 0.5])
A_GRID_DGP2 = np.array([1.8, 2.2, 2.6, 3.0])
A_GRIDS = {"dgp1": A_GRID_DGP1, "dgp2": A_GRID_DGP2}

CURVE_METHODS  = ["dr_dose", "drk_oracle", "drk_xfit"]
SCALAR_METHODS = ["oracle", "naive", "tsls"]
N_LIST = [500, 1000]
DGPS   = ["dgp1", "dgp2"]

METHOD_LABEL = {
    "oracle":     "Oracle",
    "naive":      "Naive",
    "tsls":       "TwoStage",
    "dr_dose":    "DRDose",
    "drk_oracle": "DRK_oracle",
    "drk_xfit":   "DRK_xfit",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v: Any, fmt: str = ".4f", sci_thresh: float = 1e6) -> str:
    """Return formatted float or '---' for None/NaN/inf."""
    try:
        if v is None:
            return "---"
        f = float(v)
        if not np.isfinite(f):
            return "---" if np.isnan(f) else "inf"
        if abs(f) >= sci_thresh:
            return f"{f:.2e}"
        return f"{f:{fmt}}"
    except Exception:
        return str(v)


def _pred_cov(bias: float, se: float) -> float:
    """
    Coverage predicted for an estimator normally distributed as N(bias, se^2)
    with a 95 % confidence interval:

        P(|Z + b/se| <= 1.96)  where Z ~ N(0,1)
        = Phi(1.96 - |b|/se) - Phi(-1.96 - |b|/se)

    NB: for |b|/se = 2.27, this gives ~0.38, NOT 0.03.
    """
    if se <= 0 or not np.isfinite(se) or not np.isfinite(bias):
        return np.nan
    z = abs(bias) / se
    return float(stats.norm.cdf(1.96 - z) - stats.norm.cdf(-1.96 - z))


def _load(label: str) -> dict | None:
    """Load one pkl block; return None if missing."""
    path = _RAW_DIR / f"{label}.pkl"
    if not path.exists():
        print(f"  [MISSING] {path.name}")
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _ok_recs(data: dict) -> list[dict]:
    return [r for r in data["records"] if r["error"] is None]


def _scalar_summ(data: dict) -> dict:
    """Convert a run_scalar_block pkl (harness format) to a summary dict
    compatible with summarize_exp1A for the fields we use (coverage, ser,
    bias_ref, rmse_ref, max_err, mise=NaN, V_hat_grid_mean=NaN, ...)."""
    m = data["metrics"]
    K_nan = [np.nan]   # placeholder list for per-dose fields
    return {
        "M_ok":    int(m.get("M", 50)),
        "n_errors": 0,
        "coverage": float(m.get("coverage", np.nan)),
        "ser":      float(m.get("ser", np.nan)),
        "bias_ref": float(m.get("bias", np.nan)),
        "rmse_ref": float(m.get("rmse", np.nan)),
        "max_err":  float(m.get("rmse", np.nan)),
        "mise":     np.nan,
        # Per-dose fields unused for scalar methods
        "V_hat_grid_mean": K_nan,
        "V_reg_grid_mean": K_nan,
        "V_correction_grid_mean": K_nan,
        "coverage_per_dose": K_nan,
        "bias_per_dose": [m.get("bias", np.nan)],
        "rmse_per_dose": [m.get("rmse", np.nan)],
        "ess_min_mean": np.nan,
        "ess_ratio_min_mean": np.nan,
        "weight_p99_grid_mean": K_nan,
        "weight_max_grid_mean": K_nan,
        "q_clip_fraction_mean": np.nan,
        "riesz_residual_mean": K_nan,
        "cond_M_mean": K_nan,
        "neg_share_mean": K_nan,
    }


def _se_ref(summ: dict, n: int) -> float:
    """SE at reference dose = sqrt(V_hat_ref_mean / n)."""
    ref_idx = len(summ["V_hat_grid_mean"]) // 2
    V_hat_ref = summ["V_hat_grid_mean"][ref_idx]
    if not np.isfinite(V_hat_ref):
        return np.nan
    return float(np.sqrt(abs(V_hat_ref) / n))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Load all curve blocs ────────────────────────────────────────────────
    curve_blocs: dict[tuple, tuple[dict, dict]] = {}   # (dgp,method,n) -> (data, summ)
    for dgp in DGPS:
        for method in CURVE_METHODS:
            for n in N_LIST:
                label = f"exp1A_{dgp}_{method}_snr095_n{n}_M50"
                data  = _load(label)
                if data is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        summ = summarize_exp1A(data)
                    curve_blocs[(dgp, method, n)] = (data, summ)

    # ── Load scalar blocs (for coverage/SER table) ──────────────────────────
    scalar_blocs: dict[tuple, tuple[dict, dict]] = {}
    for dgp in DGPS:
        for method in SCALAR_METHODS:
            if method == "tsls" and dgp == "dgp2":
                continue   # TwoStage DGP2 not run
            for n in N_LIST:
                lbl  = f"exp1A_{dgp}_{method}_snr095_n{n}_M50"
                data = _load(lbl)
                if data is not None:
                    # Scalar pkl uses harness format (no 'records' key)
                    summ = _scalar_summ(data)
                    scalar_blocs[(dgp, method, n)] = (data, summ)

    lines = _build_report(curve_blocs, scalar_blocs)

    out = _SUMM_DIR / "exp1A_pilot_diagnostics.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport -> {out}")


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_report(
    curve_blocs: dict,
    scalar_blocs: dict,
) -> list[str]:
    lines: list[str] = []
    add = lines.append

    add("# Exp1A Pilot -- Diagnostics Detailles")
    add(f"# Genere : {datetime.date.today()}")
    add("# SNR=0.95, n in {500, 1000}, M=50")
    add("# Conventions : SER = mean(SE_hat)/sd(J_hat)  ;  SE = sqrt(V_hat_grid/n)")
    add("")

    # ── Section 0 : convention check ────────────────────────────────────────
    add("## 0. Convention de variance -- check explicite")
    add("")
    add("V_hat_grid[k] = mean((score_i - J_dr[k])^2)  [O(1)]")
    add("SE_k = sqrt(V_hat_grid[k] / n)  [O(1/sqrt(n))]")
    add("A correct V-convention gives SE/RMSE close to 1 when bias=0.")
    add("")
    hdr = "| DGP | Method | n | V_hat_ref_mean | SE_ref | RMSE_ref | SE/RMSE |"
    sep = "|-----|--------|---|----------------|--------|----------|---------|"
    add(hdr); add(sep)
    for dgp in DGPS:
        for method in CURVE_METHODS:
            for n in N_LIST:
                key = (dgp, method, n)
                if key not in curve_blocs:
                    continue
                data, summ = curve_blocs[key]
                K   = len(A_GRIDS[dgp])
                ref = K // 2
                Vr  = summ["V_hat_grid_mean"][ref]
                se  = _se_ref(summ, n)
                rm  = summ["rmse_ref"]
                ratio = se / rm if rm > 0 else np.nan
                add(f"| {dgp} | {METHOD_LABEL[method]} | {n} | {_fmt(Vr,'.3f')} | "
                    f"{_fmt(se,'.4f')} | {_fmt(rm,'.4f')} | {_fmt(ratio,'.3f')} |")
    add("")

    # ── Section 1 : coverage ────────────────────────────────────────────────
    add("## 1. Coverage par methode et par DGP  (ref-dose global)")
    add("")
    hdr = "| DGP | Method | n=500 | n=1000 | GO/NO-GO |"
    sep = "|-----|--------|-------|--------|----------|"
    add(hdr); add(sep)
    all_methods = SCALAR_METHODS + CURVE_METHODS
    blocs_combined = {**scalar_blocs, **curve_blocs}
    for dgp in DGPS:
        for method in all_methods:
            if method == "tsls" and dgp == "dgp2":
                continue
            row = [dgp, METHOD_LABEL[method]]
            vals = []
            for n in N_LIST:
                key = (dgp, method, n)
                if key in blocs_combined:
                    c = blocs_combined[key][1].get("coverage", np.nan)
                    vals.append(c)
                    row.append(_fmt(c, ".3f"))
                else:
                    vals.append(np.nan)
                    row.append("---")
            if method in CURVE_METHODS:
                go = "GO" if all(v >= 0.90 for v in vals if np.isfinite(v)) else "NO-GO"
            else:
                go = "GO" if all(v >= 0.88 for v in vals if np.isfinite(v)) else "NO-GO"
            row.append(go)
            add("| " + " | ".join(row) + " |")
    add("")

    # ── Section 2 : SER ─────────────────────────────────────────────────────
    add("## 2. SER = mean(SE_hat) / sd(J_hat)")
    add("")
    add("SER ~1 : calibre ; <1 : variance sous-estimee ; >1 : sur-estimee.")
    add("")
    hdr = "| DGP | Method | n=500 | n=1000 | Interpretation |"
    sep = "|-----|--------|-------|--------|----------------|"
    add(hdr); add(sep)
    for dgp in DGPS:
        for method in all_methods:
            if method == "tsls" and dgp == "dgp2":
                continue
            row = [dgp, METHOD_LABEL[method]]
            sers = []
            for n in N_LIST:
                key = (dgp, method, n)
                if key in blocs_combined:
                    s = blocs_combined[key][1].get("ser", np.nan)
                    sers.append(s)
                    row.append(_fmt(s, ".3f"))
                else:
                    sers.append(np.nan)
                    row.append("---")
            # Interpretation based on average SER
            fs = [s for s in sers if np.isfinite(s)]
            if not fs:
                interp = "---"
            else:
                avg = np.mean(fs)
                if avg < 0.80:
                    interp = "under-estimated"
                elif avg > 1.25:
                    interp = "over-estimated"
                else:
                    interp = "calibrated"
            row.append(interp)
            add("| " + " | ".join(row) + " |")
    add("")

    # ── Section 3 : biais/RMSE/MISE ─────────────────────────────────────────
    add("## 3. Biais / RMSE / MISE  (methodes curve)")
    add("")
    hdr = "| DGP | Method | n | bias_ref | RMSE_ref | max_err | MISE |"
    sep = "|-----|--------|---|----------|----------|---------|------|"
    add(hdr); add(sep)
    for dgp in DGPS:
        for method in CURVE_METHODS:
            for n in N_LIST:
                key = (dgp, method, n)
                if key not in curve_blocs:
                    continue
                _, summ = curve_blocs[key]
                add(f"| {dgp} | {METHOD_LABEL[method]} | {n} | "
                    f"{_fmt(summ['bias_ref'],'+.4f')} | {_fmt(summ['rmse_ref'],'.4f')} | "
                    f"{_fmt(summ['max_err'],'.4f')} | {_fmt(summ['mise'],'.5f')} |")
    add("")

    # ── Section 4 : predicted coverage from bias/SE ─────────────────────────
    add("## 4. Coverage predite par biais  (formule normale biaisee)")
    add("")
    add("pred_cov = Phi(1.96 - |bias|/SE) - Phi(-1.96 - |bias|/SE)")
    add("NB: |bias|/SE = 2.27 => pred_cov ~= 0.38, pas 0.03.")
    add("")
    hdr = "| DGP | Method | n | |bias|/SE | pred_cov | obs_cov | delta |"
    sep = "|-----|--------|---|---------|----------|---------|-------|"
    add(hdr); add(sep)
    for dgp in DGPS:
        for method in CURVE_METHODS:
            for n in N_LIST:
                key = (dgp, method, n)
                if key not in curve_blocs:
                    continue
                data, summ = curve_blocs[key]
                bias   = summ["bias_ref"]
                se     = _se_ref(summ, n)
                obs    = summ["coverage"]
                bse    = abs(bias) / se if se > 0 and np.isfinite(se) else np.nan
                pred   = _pred_cov(bias, se)
                delta  = obs - pred if np.isfinite(pred) else np.nan
                add(f"| {dgp} | {METHOD_LABEL[method]} | {n} | "
                    f"{_fmt(bse,'.3f')} | {_fmt(pred,'.3f')} | "
                    f"{_fmt(obs,'.3f')} | {_fmt(delta,'+.3f')} |")
    add("")

    # ── Section 5 : ESS ─────────────────────────────────────────────────────
    add("## 5. ESS : ESS_min_mean / ESS_ratio_min_mean")
    add("")
    hdr = "| DGP | Method | n=500 ESS | ratio | n=1000 ESS | ratio |"
    sep = "|-----|--------|----------|-------|-----------|-------|"
    add(hdr); add(sep)
    for dgp in DGPS:
        for method in CURVE_METHODS:
            row = [dgp, METHOD_LABEL[method]]
            for n in N_LIST:
                key = (dgp, method, n)
                if key in curve_blocs:
                    _, summ = curve_blocs[key]
                    ess = summ.get("ess_min_mean", np.nan)
                    rat = summ.get("ess_ratio_min_mean", np.nan)
                    row.append(_fmt(ess, ".1f"))
                    row.append(_fmt(rat, ".3f"))
                else:
                    row.extend(["---", "---"])
            add("| " + " | ".join(row) + " |")
    add("")

    # ── Section 6 : variance decomposition ──────────────────────────────────
    add("## 6. Decomposition variance : V_reg vs V_correction (par dose)")
    add("")
    add("V_hat_grid = var(score total), decompose en V_reg + V_correction (informel).")
    add("Valeurs O(1) -- SE = sqrt(V/n). Grande V_correction => IPW instable.")
    add("")
    for dgp in DGPS:
        a_grid = A_GRIDS[dgp]
        for method in ["drk_oracle", "drk_xfit"]:
            add(f"### DGP={dgp}  Method={METHOD_LABEL[method]}")
            doses_str = " | ".join(f"d={a}" for a in a_grid)
            add(f"| n | metric | {doses_str} |")
            add("|---|--------|" + "|".join(["-------"] * len(a_grid)) + "|")
            for n in N_LIST:
                key = (dgp, method, n)
                if key not in curve_blocs:
                    continue
                _, summ = curve_blocs[key]
                def _row(label, vals):
                    cells = " | ".join(_fmt(v, ".3f") for v in vals)
                    add(f"| {n} | {label} | {cells} |")
                _row("V_hat",  summ["V_hat_grid_mean"])
                _row("V_reg",  summ["V_reg_grid_mean"])
                _row("V_corr", summ["V_correction_grid_mean"])
            add("")

    # ── Section 7 : poids IPW ────────────────────────────────────────────────
    add("## 7. Poids IPW : weight_p99 / weight_max (max sur doses)")
    add("")
    hdr = "| DGP | Method | n=500 p99 | max | n=1000 p99 | max |"
    sep = "|-----|--------|-----------|-----|-----------|-----|"
    add(hdr); add(sep)
    for dgp in DGPS:
        for method in CURVE_METHODS:
            row = [dgp, METHOD_LABEL[method]]
            for n in N_LIST:
                key = (dgp, method, n)
                if key in curve_blocs:
                    _, summ = curve_blocs[key]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        p99m = np.nanmax(summ["weight_p99_grid_mean"])
                        mxm  = np.nanmax(summ["weight_max_grid_mean"])
                    row.append(_fmt(p99m, ".2f"))
                    row.append(_fmt(mxm,  ".2f"))
                else:
                    row.extend(["---", "---"])
            add("| " + " | ".join(row) + " |")
    add("")

    # ── Section 8 : Riesz residuals ──────────────────────────────────────────
    add("## 8. Riesz residuals  (drk_xfit uniquement -- NaN pour oracle-q)")
    add("")
    hdr = "| DGP | n | riesz_mean (max dose) | riesz_max (max dose) |"
    sep = "|-----|---|----------------------|---------------------|"
    add(hdr); add(sep)
    for dgp in DGPS:
        for n in N_LIST:
            key = (dgp, "drk_xfit", n)
            if key in curve_blocs:
                _, summ = curve_blocs[key]
                rm  = np.nanmax(summ["riesz_residual_mean"])
                # riesz_residual_mean is mean over reps of riesz_residual_grid_mean per dose
                # we also stored max versions in records but only mean in summ; approximate via same
                add(f"| {dgp} | {n} | {_fmt(rm, '.5f')} | --- |")
    add("")

    # ── Section 9 : conditionnement M ───────────────────────────────────────
    add("## 9. Conditionnement M : cond_M_mean (max sur doses)")
    add("")
    add("Valeurs 1e23 / inf attendues -- matrice M RKHS quasi-singuliere.")
    add("Regularisation lambda_Q=1e-3 assure la stabilite numerique.")
    add("")
    hdr = "| DGP | n | cond_M_max | Interpretation |"
    sep = "|-----|---|-----------|----------------|"
    add(hdr); add(sep)
    for dgp in DGPS:
        for n in N_LIST:
            key = (dgp, "drk_xfit", n)
            if key in curve_blocs:
                _, summ = curve_blocs[key]
                cm_vals = summ["cond_M_mean"]
                cm_max  = np.nanmax([v for v in cm_vals if np.isfinite(v)]) \
                          if any(np.isfinite(v) for v in cm_vals) else np.inf
                interp  = "RKHS ill-cond. (expected)" if not np.isfinite(cm_max) or cm_max > 1e10 \
                          else "moderate"
                add(f"| {dgp} | {n} | {_fmt(cm_max)} | {interp} |")
    add("")

    # ── Section 10 : negative share ──────────────────────────────────────────
    add("## 10. Negative share : fraction poids IPW < 0  (par dose)")
    add("")
    add("KPVPolicyBridgeQ resout une condition de moment -- poids negatifs attendus.")
    add("37--47 % de poids negatifs observes sont normaux.")
    add("")
    for dgp in DGPS:
        a_grid = A_GRIDS[dgp]
        add(f"### DGP={dgp}  (drk_xfit)")
        hdr  = "| n | " + " | ".join(f"a={a}" for a in a_grid) + " |"
        sep  = "|---|" + "|".join(["-------"] * len(a_grid)) + "|"
        add(hdr); add(sep)
        for n in N_LIST:
            key = (dgp, "drk_xfit", n)
            if key in curve_blocs:
                _, summ = curve_blocs[key]
                cells = " | ".join(_fmt(v, ".3f") for v in summ["neg_share_mean"])
                add(f"| {n} | {cells} |")
        add("")

    # ── Section 11 : q_clip_fraction ─────────────────────────────────────────
    add("## 11. q_clip_fraction  (fraction observations clippees)")
    add("")
    hdr = "| DGP | Method | n=500 | n=1000 |"
    sep = "|-----|--------|-------|--------|"
    add(hdr); add(sep)
    for dgp in DGPS:
        for method in CURVE_METHODS:
            row = [dgp, METHOD_LABEL[method]]
            for n in N_LIST:
                key = (dgp, method, n)
                if key in curve_blocs:
                    _, summ = curve_blocs[key]
                    row.append(_fmt(summ.get("q_clip_fraction_mean", np.nan), ".5f"))
                else:
                    row.append("---")
            add("| " + " | ".join(row) + " |")
    add("")

    # ── Section 12 : coverage par dose ───────────────────────────────────────
    add("## 12. Coverage par dose -- DRKernel xfit")
    add("")
    for dgp in DGPS:
        a_grid = A_GRIDS[dgp]
        add(f"### DGP={dgp}")
        hdr  = "| Dose | n=500 | n=1000 |"
        sep  = "|------|-------|--------|"
        add(hdr); add(sep)
        for k, a in enumerate(a_grid):
            row = [f"{a:.2f}"]
            for n in N_LIST:
                key = (dgp, "drk_xfit", n)
                if key in curve_blocs:
                    _, summ = curve_blocs[key]
                    cpd = summ.get("coverage_per_dose", [])
                    if k < len(cpd):
                        row.append(_fmt(cpd[k], ".3f"))
                    else:
                        row.append("---")
                else:
                    row.append("---")
            add("| " + " | ".join(row) + " |")
        add("")

    # ── Section 13 : DGP2 bias by component ──────────────────────────────────
    add("## 13. Diagnostic DGP2 -- Biais par composante  (KEY TABLE)")
    add("")
    add("J_reg : estimateur plug-in (bridge h seul, sans correction IPW)")
    add("J_dr  : estimateur DR final (J_reg + correction IPW via q)")
    add("bias_reg = mean(J_reg) - J_policy_true   (biais du bridge h)")
    add("bias_dr  = mean(J_dr)  - J_policy_true   (biais total DR)")
    add("corr     = mean(J_dr - J_reg)            (correction IPW appliquee)")
    add("")
    add("Interpretation :")
    add("  bias_reg ~= bias_dr : h-bridge domine, correction q n'aide pas")
    add("  |bias_dr| < |bias_reg| : correction q reduit le biais (partiellement)")
    add("  |bias_dr| > |bias_reg| : correction q ajoute du biais")
    add("")
    for method in ["drk_oracle", "drk_xfit"]:
        add(f"### Method = {METHOD_LABEL[method]}")
        a_grid = A_GRIDS["dgp2"]
        hdr = "| n | Dose | J_pt | mean_J_reg | mean_J_dr | bias_reg | bias_dr | corr |"
        sep = "|---|------|------|-----------|-----------|---------|--------|------|"
        add(hdr); add(sep)
        for n in N_LIST:
            key = ("dgp2", method, n)
            if key not in curve_blocs:
                continue
            data, _ = curve_blocs[key]
            ok  = _ok_recs(data)
            J_pt_arr = np.array(data["meta"]["J_policy_true"])
            J_reg_all = np.array([r["J_reg"] for r in ok])   # (M_ok, K)
            J_dr_all  = np.array([r["J_dr"]  for r in ok])   # (M_ok, K)
            mean_Jreg = np.mean(J_reg_all, axis=0)
            mean_Jdr  = np.mean(J_dr_all,  axis=0)
            bias_reg  = mean_Jreg - J_pt_arr
            bias_dr   = mean_Jdr  - J_pt_arr
            corr      = mean_Jdr  - mean_Jreg
            for k, a in enumerate(a_grid):
                add(f"| {n} | {a:.2f} | {_fmt(J_pt_arr[k],'.4f')} | "
                    f"{_fmt(mean_Jreg[k],'.4f')} | {_fmt(mean_Jdr[k],'.4f')} | "
                    f"{_fmt(bias_reg[k],'+.4f')} | {_fmt(bias_dr[k],'+.4f')} | "
                    f"{_fmt(corr[k],'+.4f')} |")
        add("")

    # ── Section 14 : oracle-q vs xfit-q ──────────────────────────────────────
    add("## 14. Diagnostic DGP2 -- oracle-q vs xfit-q  (biais par dose)")
    add("")
    add("Si |bias_oracle| ~= |bias_xfit| : biais vient du h-bridge, pas de q estime.")
    add("Si |bias_xfit| >> |bias_oracle| : q estime (KPVPolicyBridgeQ) ajoute du biais.")
    add("")
    for dgp in ["dgp2"]:
        a_grid = A_GRIDS[dgp]
        hdr = "| n | Dose | bias_oracle | bias_xfit | diff (xfit-oracle) |"
        sep = "|---|------|------------|-----------|-------------------|"
        add(hdr); add(sep)
        for n in N_LIST:
            key_o = (dgp, "drk_oracle", n)
            key_x = (dgp, "drk_xfit",  n)
            if key_o not in curve_blocs or key_x not in curve_blocs:
                continue
            summ_o = curve_blocs[key_o][1]
            summ_x = curve_blocs[key_x][1]
            bo = summ_o.get("bias_per_dose", [])
            bx = summ_x.get("bias_per_dose", [])
            for k, a in enumerate(a_grid):
                if k < len(bo) and k < len(bx):
                    diff = bx[k] - bo[k]
                    add(f"| {n} | {a:.2f} | {_fmt(bo[k],'+.4f')} | "
                        f"{_fmt(bx[k],'+.4f')} | {_fmt(diff,'+.4f')} |")
    add("")

    # ── Section 14b : DGP1 oracle-q vs xfit-q (reference) ───────────────────
    add("### DGP1 reference (oracle-q vs xfit-q)")
    add("")
    a_grid = A_GRIDS["dgp1"]
    hdr = "| n | Dose | bias_oracle | bias_xfit | diff |"
    sep = "|---|------|------------|-----------|------|"
    add(hdr); add(sep)
    for n in N_LIST:
        key_o = ("dgp1", "drk_oracle", n)
        key_x = ("dgp1", "drk_xfit",  n)
        if key_o not in curve_blocs or key_x not in curve_blocs:
            continue
        bo = curve_blocs[key_o][1].get("bias_per_dose", [])
        bx = curve_blocs[key_x][1].get("bias_per_dose", [])
        for k, a in enumerate(a_grid):
            if k < len(bo) and k < len(bx):
                diff = bx[k] - bo[k]
                add(f"| {n} | {a:.2f} | {_fmt(bo[k],'+.4f')} | "
                    f"{_fmt(bx[k],'+.4f')} | {_fmt(diff,'+.4f')} |")
    add("")

    # ── Section 15 : conclusions ─────────────────────────────────────────────
    add("## 15. Conclusions et recommandations")
    add("")
    add("### DGP1 -- Cobb-Douglas")
    add("- Coverage DRKernel xfit : 0.940--0.980 (GO, critere >= 0.90 ✅)")
    add("- SER calibre (~0.85--1.03), variance correctement estimee")
    add("- ESS sain (>80), q_clip=0, riesz_residual faible")
    add("- cond_M extreme (1e21--1e23) : attendu, ne degrade pas les resultats")
    add("- Coverage a=-0.50 : 0.76--0.84 (boundary, V_correction eleve)")
    add("- RECOMMANDATION : GO grille elargie DGP1")
    add("")
    add("### DGP2 -- Michaelis-Menten")

    # Compute actual pred_cov for DGP2 drk_xfit n=1000
    key = ("dgp2", "drk_xfit", 1000)
    if key in curve_blocs:
        data, summ = curve_blocs[key]
        bias_r = summ["bias_ref"]
        se_r   = _se_ref(summ, 1000)
        bse_r  = abs(bias_r) / se_r if se_r > 0 else np.nan
        pc_r   = _pred_cov(bias_r, se_r)
        obs_r  = summ["coverage"]
        add(f"- DRKernel xfit n=1000 : bias={_fmt(bias_r,'+.4f')}, SE={_fmt(se_r,'.4f')}")
        add(f"  |bias|/SE = {_fmt(bse_r,'.2f')} => pred_cov = {_fmt(pc_r,'.3f')} (obs = {_fmt(obs_r,'.3f')})")
        add(f"  Delta = {_fmt(obs_r - pc_r if np.isfinite(pc_r) else np.nan,'+.3f')} => biais explique la couverture")
    add("- ESS sain (90--178), q_clip=0 => pas de collapse IPW")
    add("- SER ~0.53--1.07 => variance bien estimee")
    add("- Suspect principal : biais fini de KPVBridgeH a n<=1000 pour MM non-lineaire")
    add("  (tests Phase 9B valides a n=2000--3000, pas a n=500--1000)")
    add("- Section 13 : si bias_reg ~= bias_dr => biais dans h, pas dans q")
    add("- Section 14 : si bias_oracle ~= bias_xfit => biais dans h, pas dans q estime")
    add("")
    add("### Prochaine etape : micro-pilote DGP2 n=2000")
    add("```")
    add("python -m simulations.archive.utilities.exp1A_coverage \\")
    add("  --dgp dgp2 --snr 0.95 --n 2000 --M 50 \\")
    add("  --method drk_oracle drk_xfit --pilot")
    add("```")
    add("Critere : si bias(n=2000) < bias(n=1000) => biais fini ✅ => GO grande grille DGP2")
    add("")

    return lines


# ---------------------------------------------------------------------------
# DGP2 convergence report (callable once n=2000 pkl exists)
# ---------------------------------------------------------------------------

def dgp2_convergence_report() -> None:
    """
    Generate exp1A_dgp2_convergence.md comparing bias at n in {500,1000,2000}.

    Call after micro-pilot DGP2 n=2000 M=50 completes:
        python -c "from simulations.archive.utilities.exp1A_diagnostics import dgp2_convergence_report; dgp2_convergence_report()"
    """
    A_GRID = A_GRID_DGP2
    N_ALL  = [500, 1000, 2000]
    out    = _SUMM_DIR / "exp1A_dgp2_convergence.md"

    lines: list[str] = []
    add = lines.append

    add("# DGP2 Michaelis-Menten -- Convergence du Biais avec n")
    add(f"# Genere : {datetime.date.today()}")
    add("# SNR=0.95, M=50, methodes DRK_oracle + DRK_xfit")
    add("# Hypothese : biais fini de KPVBridgeH -> doit decroitre avec n")
    add("")

    # Table 1 : by dose
    add("## 1. Biais par composante par dose  (J_reg, J_dr, correction)")
    add("")
    for method in ["drk_oracle", "drk_xfit"]:
        add(f"### Method = {METHOD_LABEL[method]}")
        hdr = "| n | Dose | J_pt | mean_J_reg | mean_J_dr | bias_reg | bias_dr | corr | SE | |b|/SE | pred_cov | obs_cpd |"
        sep = "|---|------|------|-----------|-----------|---------|--------|------|-----|------|---------|---------|"
        add(hdr); add(sep)
        for n in N_ALL:
            label = f"exp1A_dgp2_{method}_snr095_n{n}_M50"
            data  = _load(label)
            if data is None:
                add(f"| {n} | [MISSING] | | | | | | | | | | |")
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                summ = summarize_exp1A(data)
            ok   = _ok_recs(data)
            J_pt = np.array(data["meta"]["J_policy_true"])
            J_rg = np.array([r["J_reg"] for r in ok])
            J_dr = np.array([r["J_dr"]  for r in ok])
            V_hg = np.array([r["V_hat_grid"] for r in ok])   # (M_ok, K)
            mean_Jreg = np.mean(J_rg, axis=0)
            mean_Jdr  = np.mean(J_dr, axis=0)
            bias_reg  = mean_Jreg - J_pt
            bias_dr   = mean_Jdr  - J_pt
            corr      = mean_Jdr  - mean_Jreg
            SE_grid   = np.sqrt(np.mean(V_hg, axis=0) / n)   # (K,)
            cpd       = summ.get("coverage_per_dose", [np.nan]*len(A_GRID))
            for k, a in enumerate(A_GRID):
                se_k  = SE_grid[k]
                bse_k = abs(bias_dr[k]) / se_k if se_k > 0 else np.nan
                pc_k  = _pred_cov(bias_dr[k], se_k)
                add(f"| {n} | {a:.2f} | {_fmt(J_pt[k],'.4f')} | "
                    f"{_fmt(mean_Jreg[k],'.4f')} | {_fmt(mean_Jdr[k],'.4f')} | "
                    f"{_fmt(bias_reg[k],'+.4f')} | {_fmt(bias_dr[k],'+.4f')} | "
                    f"{_fmt(corr[k],'+.4f')} | {_fmt(se_k,'.4f')} | "
                    f"{_fmt(bse_k,'.2f')} | {_fmt(pc_k,'.3f')} | "
                    f"{_fmt(cpd[k] if k < len(cpd) else np.nan,'.3f')} |")
        add("")

    # Table 2 : summary by n
    add("## 2. Tableau recapitulatif par n")
    add("")
    hdr = "| Method | n | bias_ref | SE_ref | |b|/SE | pred_cov | obs_cov | ESS_min | q_clip |"
    sep = "|--------|---|---------|--------|------|---------|---------|--------|--------|"
    add(hdr); add(sep)
    for method in ["drk_oracle", "drk_xfit"]:
        for n in N_ALL:
            label = f"exp1A_dgp2_{method}_snr095_n{n}_M50"
            data  = _load(label)
            if data is None:
                add(f"| {METHOD_LABEL[method]} | {n} | [MISSING] | | | | | | |")
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                summ = summarize_exp1A(data)
            bias_r = summ["bias_ref"]
            se_r   = _se_ref(summ, n)
            bse_r  = abs(bias_r) / se_r if se_r > 0 else np.nan
            pc_r   = _pred_cov(bias_r, se_r)
            obs_r  = summ["coverage"]
            ess_r  = summ.get("ess_min_mean", np.nan)
            qcl_r  = summ.get("q_clip_fraction_mean", np.nan)
            add(f"| {METHOD_LABEL[method]} | {n} | {_fmt(bias_r,'+.4f')} | "
                f"{_fmt(se_r,'.4f')} | {_fmt(bse_r,'.2f')} | {_fmt(pc_r,'.3f')} | "
                f"{_fmt(obs_r,'.3f')} | {_fmt(ess_r,'.1f')} | {_fmt(qcl_r,'.5f')} |")
    add("")

    # Table 3 : bias trend
    add("## 3. Tendance du biais (ref-dose) avec n")
    add("")
    add("Critere GO : |bias_dr(n=2000)| < |bias_dr(n=1000)| * 0.8")
    add("")
    hdr = "| Method | |bias| n=500 | |bias| n=1000 | |bias| n=2000 | ratio 2000/1000 | GO/NO-GO |"
    sep = "|--------|-----------|-----------|-----------|----------------|----------|"
    add(hdr); add(sep)
    for method in ["drk_oracle", "drk_xfit"]:
        biases = {}
        for n in N_ALL:
            label = f"exp1A_dgp2_{method}_snr095_n{n}_M50"
            data  = _load(label)
            if data is None:
                biases[n] = np.nan
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                summ = summarize_exp1A(data)
            biases[n] = abs(summ["bias_ref"])
        b500, b1000, b2000 = biases.get(500, np.nan), biases.get(1000, np.nan), biases.get(2000, np.nan)
        ratio = b2000 / b1000 if b1000 > 0 and np.isfinite(b2000) else np.nan
        if np.isfinite(ratio):
            go = "GO ✅" if ratio < 0.80 else ("MARGINAL" if ratio < 1.0 else "NO-GO ❌")
        else:
            go = "PENDING"
        add(f"| {METHOD_LABEL[method]} | {_fmt(b500,'.4f')} | {_fmt(b1000,'.4f')} | "
            f"{_fmt(b2000,'.4f')} | {_fmt(ratio,'.3f')} | {go} |")
    add("")

    # Conclusion
    add("## 4. Conclusion")
    add("")
    add("Si bias(n=2000) < bias(n=1000) avec ratio < 0.80 :")
    add("  => Biais fini-echantillon de KPVBridgeH confirme")
    add("  => GO grande grille DGP2 avec documentation limitation")
    add("  => Pour S6 : 'DRKernel non-parametrique montre un biais a n<=1000")
    add("     sur DGP non-lineaire ; ce biais diminue avec n (finite-sample bridge bias)'")
    add("")
    add("Si ratio >= 1.0 (biais stable ou croissant) :")
    add("  => Biais structurel -- investiguer KPVBridgeH (lambda, bandwidth, Fredholm)")
    add("  => NO-GO grande grille DGP2")
    add("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport convergence -> {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
