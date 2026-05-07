"""
Phase 19 -- Checkpoint 0: pre-tuning diagnostic gate.

Reads existing diagnostic JSONs, runs Bennett at default config (M=5 bootstrap
reps), and emits a GO / CAUTION / ABORT verdict.

Decision rule (5 indicators, 4+ pass = GO, 2-3 = CAUTION, ≤1 = ABORT):

| Indicator                       | Pass     | Caution    | Fail   |
|---------------------------------|----------|------------|--------|
| F-stat Z~A (from diag JSON)     | ≥ 50     | 10–50      | < 10   |
| det([Z,W] gram) (from JSON)     | > 0.5    | 0.1–0.5    | < 0.1  |
| Fredholm residual_norm_h (M=5)  | < 0.01   | 0.01–0.1   | > 0.1  |
| RR_ref (M=5)                    | < 0.01   | 0.01–0.05  | > 0.05 |
| 2SLS sign vs reference          | matches  | n/a        | wrong  |

If a diagnostic JSON is missing (NHANES, NLSY79), the (F-stat, det) checks
fall back to fast on-the-fly computation.

Usage:
    python -u -m simulations.experiments.real_data_diagnose --dataset rhc
    python -u -m simulations.experiments.real_data_diagnose --dataset nhanes_cadmium

Output: simulations/results/summaries/realdata_{dataset}_checkpoint0.md
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd

from simulations.experiments.s8_applications.s8_3_blind_tuning import (
    RealDataDGP, DATASETS, _BASE_DIR, _make_bennett_estimator_real,
)
from simulations.experiments.s7_simulations.s7_8_blind_tuning_bennett import (
    BENNETT_DEFAULTS, extract_diagnostics, bennett_blind_score_v2,
)
from simulations.experiments.dgp2_bias_diagnostics import _run_block

_DIAG_DIR = _BASE_DIR / "simulations" / "datasets" / "diagnostics"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_RAW_BASE = _BASE_DIR / "simulations" / "results" / "raw" / "s8"

# Map dataset name -> diagnostic JSON file (or None if absent)
_DIAG_JSON_MAP = {
    "rhc": "rhc_metrics.json",
    "nhanes_cadmium": None,           # not yet computed
    "nlsy79_father": None,             # not yet computed
    "eth_ess": "eth_ess_fertilizer_metrics.json",
}


def _load_diag_json(name: str) -> Dict[str, Any]:
    """Read pre-existing diagnostics if available, else compute basics on-the-fly."""
    fname = _DIAG_JSON_MAP.get(name)
    if fname:
        path = _DIAG_DIR / fname
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

    # Fall-back: compute F-stat, det(Z,W gram) on-the-fly from raw CSV
    cfg = DATASETS[name]
    df = pd.read_csv(_BASE_DIR / "simulations" / "datasets" / "processed" / cfg["csv"])
    if cfg.get("filter") == "A > 0":
        df = df.loc[df[cfg["a_col"]] > 0].reset_index(drop=True)
    A = df[cfg["a_col"]].to_numpy(float)
    Z = df[cfg["z_col"]].to_numpy(float)
    W = df[cfg["w_col"]].to_numpy(float)
    n = len(A)

    # F-stat for Z ~ a + b*A (single instrument)
    Xmat = np.column_stack([np.ones(n), A])
    beta_z, *_ = np.linalg.lstsq(Xmat, Z, rcond=None)
    resid = Z - Xmat @ beta_z
    sigma2 = float(resid @ resid / (n - 2))
    cov = sigma2 * np.linalg.inv(Xmat.T @ Xmat)
    se_b = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else float("nan")
    t_stat = float(beta_z[1] / se_b) if se_b > 0 else float("nan")
    F_stat = t_stat ** 2 if np.isfinite(t_stat) else float("nan")

    # det([Z,W] gram) — standardize Z, W first
    Zs = (Z - Z.mean()) / (Z.std() + 1e-12)
    Ws = (W - W.mean()) / (W.std() + 1e-12)
    gram = np.array([[Zs @ Zs / n, Zs @ Ws / n],
                     [Ws @ Zs / n, Ws @ Ws / n]])
    det_zw = float(np.linalg.det(gram))

    # Naive OLS A → Y
    Y = df[cfg["y_col"]].to_numpy(float)
    Ym = np.column_stack([np.ones(n), A])
    beta_y, *_ = np.linalg.lstsq(Ym, Y, rcond=None)
    naive_ols = float(beta_y[1])

    return {
        "dataset": name,
        "N": int(n),
        "F_stat_ZA": float(F_stat),
        "det_ZW_gram": float(det_zw),
        "naive_ols_coeff": naive_ols,
        "computed_on_the_fly": True,
    }


def _classify(value: float, pass_thr: float, caution_thr: float,
              direction: str = "lower") -> str:
    """Return 'PASS', 'CAUTION', or 'FAIL' based on threshold direction."""
    if not np.isfinite(value):
        return "FAIL"
    if direction == "lower":  # lower is better (e.g. residual)
        if value < pass_thr: return "PASS"
        if value < caution_thr: return "CAUTION"
        return "FAIL"
    else:                     # higher is better (e.g. F-stat)
        if value > pass_thr: return "PASS"
        if value > caution_thr: return "CAUTION"
        return "FAIL"


def run_diagnose(dataset: str, *, M: int = 5, rho_coarsen: float = 0.30,
                 residualize: bool = True, seed_base: int = 79_000_000) -> Dict[str, Any]:
    print(f"\n=== Phase 19 Checkpoint 0: {dataset} ===\n")

    # ─── Step 1: data-level diagnostics from JSON (or on-the-fly fallback) ──
    diag_data = _load_diag_json(dataset)
    print(f"Data-level diagnostics:")
    print(f"  N             = {diag_data.get('N', '?')}")
    print(f"  F_stat_ZA     = {diag_data.get('F_stat_ZA', float('nan')):.2f}")
    print(f"  det_ZW_gram   = {diag_data.get('det_ZW_gram', float('nan')):.4f}")
    print(f"  naive_ols     = {diag_data.get('naive_ols_coeff', float('nan')):+.4f}")
    if "reference_estimate" in diag_data:
        print(f"  reference     = {diag_data['reference_estimate']}")

    # ─── Step 2: 2SLS triangulation (Tchetgen / Cui anchor) ─────────────────
    print(f"\nRunning 2SLS triangulation...")
    from simulations.experiments.s8_applications.s8_3_baselines import _twosls
    from simulations.dgp.base import DGPSample

    rho = rho_coarsen if DATASETS[dataset].get("binary_a") else None
    dgp = RealDataDGP(dataset, rho_coarsen=rho, residualize=residualize)
    full_sample = DGPSample(
        Y=dgp.Y, A=dgp.A, W=dgp.W, Z=dgp.Z,
        X=dgp.X, U=np.full(dgp.N, np.nan), psi_0=float("nan"),
    )
    twosls_res = _twosls(full_sample)
    print(f"  2SLS psi_hat   = {twosls_res['psi_hat']:+.4f}")
    print(f"  2SLS SE_an     = {twosls_res['SE_analytic']:.4f}")
    print(f"  2SLS stage1_F  = {twosls_res['stage1_F']:.1f}")

    # ─── Step 3: Bennett at defaults, M=5 reps ──────────────────────────────
    print(f"\nRunning Bennett at defaults, M={M} bootstrap reps...")
    bw = dgp.bandwidths()

    def factory():
        return _make_bennett_estimator_real(
            BENNETT_DEFAULTS, dgp, bw["h_KDE"],
            bw["ell_W"], bw["ell_A"], bw["ell_Z"],
            a_grid=dgp.a_grid, ref_idx=dgp.ref_idx,
        )

    raw_dir = _RAW_BASE / dataset / "diagnose"
    raw_dir.mkdir(parents=True, exist_ok=True)
    J_pt = np.full(len(dgp.a_grid), np.nan)
    t0 = time.time()
    data = _run_block(
        dgp, factory, np.asarray(dgp.a_grid), J_pt,
        n=dgp.N, M=M, seed_base=seed_base,
        label="diagnose_defaults", raw_dir=raw_dir,
    )
    elapsed = time.time() - t0
    print(f"  Bennett-defaults M={M} done in {elapsed:.1f}s")

    diags = extract_diagnostics(data, ref_idx=dgp.ref_idx)
    diags["lambda_h"] = float(BENNETT_DEFAULTS["lambda_h"])
    diags["m_h"] = float(BENNETT_DEFAULTS["m_h"])
    blind_score = bennett_blind_score_v2(diags, n=dgp.N)
    print(f"\n  blind_score_v2 = {blind_score:.4f}")
    print(f"  residual_norm_h= {diags.get('residual_norm_h', float('nan')):.4f}")
    print(f"  RR_ref         = {diags.get('RR_ref', float('nan')):.4f}")
    print(f"  kappa_h        = {diags.get('kappa_h', float('nan')):.2e}")
    print(f"  kappa_r        = {diags.get('kappa_r', float('nan')):.2e}")
    print(f"  eff_rank_h     = {diags.get('eff_rank_h', float('nan')):.1f}")
    print(f"  eff_rank_r     = {diags.get('eff_rank_r', float('nan')):.1f}")

    # ─── Step 4: apply decision rule ────────────────────────────────────────
    indicators = {
        "F_stat_ZA":      _classify(diag_data.get("F_stat_ZA", float("nan")),
                                    pass_thr=50, caution_thr=10, direction="higher"),
        "det_ZW_gram":    _classify(diag_data.get("det_ZW_gram", float("nan")),
                                    pass_thr=0.5, caution_thr=0.1, direction="higher"),
        "Fredholm_res":   _classify(diags.get("residual_norm_h", float("nan")),
                                    pass_thr=0.01, caution_thr=0.1, direction="lower"),
        "RR_ref":         _classify(diags.get("RR_ref", float("nan")),
                                    pass_thr=0.01, caution_thr=0.05, direction="lower"),
    }
    # 2SLS sign vs reference (only if reference exists)
    cfg = DATASETS[dataset]
    ref = cfg.get("reference_estimate")
    if ref is not None and np.isfinite(twosls_res["psi_hat"]):
        if np.sign(twosls_res["psi_hat"]) == np.sign(ref):
            indicators["2SLS_sign"] = "PASS"
        else:
            indicators["2SLS_sign"] = "FAIL"
    else:
        indicators["2SLS_sign"] = "n/a"

    pass_count = sum(1 for v in indicators.values() if v == "PASS")
    fail_count = sum(1 for v in indicators.values() if v == "FAIL")
    n_indicators = sum(1 for v in indicators.values() if v != "n/a")

    if pass_count >= 4:
        verdict = "GO"
    elif pass_count >= 2 and fail_count <= 1:
        verdict = "CAUTION"
    else:
        verdict = "ABORT"

    print(f"\nIndicator summary:")
    for k, v in indicators.items():
        print(f"  {k:18s} {v}")
    print(f"\n  Pass count: {pass_count}/{n_indicators}, Fail count: {fail_count}")
    print(f"\n  VERDICT: {verdict}")

    # ─── Step 5: write Checkpoint 0 markdown ────────────────────────────────
    md_path = _SUMM_DIR / f"realdata_{dataset}_checkpoint0.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    import datetime
    lines = [
        f"# Phase 19 — {dataset} Checkpoint 0 (pre-tuning diagnostic)",
        f"*Generated {datetime.datetime.now().isoformat()}*",
        "",
        f"## Verdict: **{verdict}**",
        "",
        f"- GO     → ≥4 indicators pass  → proceed to Stage 1 tuning",
        f"- CAUTION → 2-3 pass, ≤1 fail   → proceed but flag in FINAL.md",
        f"- ABORT  → ≤1 pass OR ≥2 fail   → skip tuning, document failure mode",
        "",
        "## Data-level diagnostics",
        f"- N: {diag_data.get('N', '?')}",
        f"- F_stat (Z~A): **{diag_data.get('F_stat_ZA', float('nan')):.2f}** "
        f"({indicators['F_stat_ZA']}; pass≥50, caution≥10)",
        f"- det([Z,W] gram): **{diag_data.get('det_ZW_gram', float('nan')):.4f}** "
        f"({indicators['det_ZW_gram']}; pass>0.5, caution>0.1)",
        f"- naive OLS A→Y: **{diag_data.get('naive_ols_coeff', float('nan')):+.4f}**",
        "",
        "## 2SLS triangulation (TwoStageLeastSquares on residualized data)",
        f"- psi_hat: **{twosls_res['psi_hat']:+.4f}**",
        f"- SE (analytic, 2-stage sandwich): {twosls_res['SE_analytic']:.4f}",
        f"- 2SLS Stage 1 F-stat: {twosls_res['stage1_F']:.1f}",
        f"- 2SLS sign vs reference: {indicators['2SLS_sign']}",
        f"- Reference: {ref} ({cfg.get('reference_source', 'n/a')})",
        "",
        f"## Bennett at defaults (M={M} bootstrap reps)",
        f"- residual_norm_h: **{diags.get('residual_norm_h', float('nan')):.4f}** "
        f"({indicators['Fredholm_res']}; pass<0.01, caution<0.1)",
        f"- RR_ref: **{diags.get('RR_ref', float('nan')):.4f}** "
        f"({indicators['RR_ref']}; pass<0.01, caution<0.05)",
        f"- kappa_h: {diags.get('kappa_h', float('nan')):.2e}",
        f"- kappa_r: {diags.get('kappa_r', float('nan')):.2e}",
        f"- eff_rank_h: {diags.get('eff_rank_h', float('nan')):.1f}",
        f"- eff_rank_r: {diags.get('eff_rank_r', float('nan')):.1f}",
        f"- blind_score_v2: {blind_score:.4f}",
        "",
        "## Indicator table",
        "",
        "| Indicator | Status |",
        "|---|---|",
    ]
    for k, v in indicators.items():
        lines.append(f"| {k} | {v} |")
    lines.append(f"\nPass: {pass_count}/{n_indicators}, Fail: {fail_count}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Markdown: {md_path}")

    return {
        "dataset": dataset, "verdict": verdict,
        "indicators": indicators,
        "data_diag": diag_data,
        "twosls": twosls_res,
        "bennett_diags": diags,
        "blind_score": blind_score,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    p.add_argument("--M", type=int, default=5)
    p.add_argument("--rho_coarsen", type=float, default=0.30)
    p.add_argument("--no_residualize", action="store_true")
    args = p.parse_args()
    run_diagnose(args.dataset, M=args.M, rho_coarsen=args.rho_coarsen,
                 residualize=(not args.no_residualize))


if __name__ == "__main__":
    main()
