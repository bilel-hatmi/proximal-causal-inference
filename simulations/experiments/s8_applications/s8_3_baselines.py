"""
Phase 19 -- Triangulation baselines on real data.

Computes three reference estimators on the residualized + coarsened real data:
  1. Naive OLS  : Y_resid ~ A
  2. OLS + X    : Y ~ A, X (no residualization, raw)
  3. 2SLS proxy : TwoStageLeastSquares (Tchetgen 2020 / Cui 2023 reference)

Each estimator gets:
  - point estimate (psi_hat)
  - analytical SE (where available)
  - bootstrap SE (B=200 reps via the same RealDataDGP sampler used by tuning)

These triangulate against the Bennett blind-tuning winner (Checkpoint 2 of the
Phase 19 protocol). If Bennett and 2SLS disagree by > 2 sigma we have a
specification anomaly to investigate.

Usage
-----
    python -u -m simulations.experiments.real_data_baseline --dataset rhc
    python -u -m simulations.experiments.real_data_baseline --dataset nhanes_cadmium

Output: simulations/results/raw/s8/{dataset}/baseline/baseline.pkl
        simulations/results/summaries/realdata_{dataset}_baseline.md
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from simulations.experiments.s8_applications.s8_3_blind_tuning import RealDataDGP, DATASETS, _BASE_DIR

_RAW_BASE = _BASE_DIR / "simulations" / "results" / "raw" / "s8"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"


def _ols_naive(Y: np.ndarray, A: np.ndarray) -> Dict[str, float]:
    """Y = a + b*A + eps. Returns slope b + analytical SE."""
    n = len(Y)
    X = np.column_stack([np.ones(n), A])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    sigma2 = float(resid @ resid / (n - 2))
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return {
        "psi_hat": float(beta[1]),
        "SE_analytic": float(np.sqrt(cov[1, 1])),
    }


def _ols_with_X(Y: np.ndarray, A: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    """Y = a + b*A + gamma*X + eps."""
    n = len(Y)
    if X is None or X.shape[0] == 0:
        return {"psi_hat": float("nan"), "SE_analytic": float("nan")}
    Xmat = np.column_stack([np.ones(n), A, X])
    beta, *_ = np.linalg.lstsq(Xmat, Y, rcond=None)
    resid = Y - Xmat @ beta
    sigma2 = float(resid @ resid / (n - Xmat.shape[1]))
    cov = sigma2 * np.linalg.inv(Xmat.T @ Xmat)
    return {
        "psi_hat": float(beta[1]),
        "SE_analytic": float(np.sqrt(cov[1, 1])),
    }


def _twosls(sample) -> Dict[str, float]:
    """TwoStageLeastSquares with sandwich SE."""
    from simulations.methods.two_stage_linear import TwoStageLeastSquares
    est = TwoStageLeastSquares()
    est.fit(sample)
    res = est.estimate()
    return {
        "psi_hat": float(res.psi_hat),
        "SE_analytic": float(np.sqrt(res.V_hat / res.n)) if res.V_hat > 0 else float("nan"),
        "stage1_F": float(res.extra.get("stage1_F", float("nan"))),
        "beta_W": float(res.extra.get("beta_W", float("nan"))),
        "n": int(res.n),
    }


def _bootstrap_estimator(estimator_fn, dgp: RealDataDGP, B: int, seed_base: int,
                         label: str) -> Dict[str, Any]:
    """
    Bootstrap an estimator B times. estimator_fn takes (sample) -> dict with 'psi_hat'.
    Returns mean psi_hat across boot reps + bootstrap SE = std(psi_hat).
    """
    psi_vals = []
    for i in range(B):
        sample = dgp.generate(dgp.N, seed=seed_base + i)
        try:
            d = estimator_fn(sample)
            psi_vals.append(d["psi_hat"])
        except Exception:
            psi_vals.append(float("nan"))
    psi_vals = np.asarray(psi_vals, dtype=float)
    finite = psi_vals[np.isfinite(psi_vals)]
    if len(finite) < 2:
        return {"label": label, "B_ok": int(len(finite)),
                "psi_mean": float("nan"), "SE_boot": float("nan")}
    return {
        "label": label,
        "B_ok": int(len(finite)),
        "psi_mean": float(np.mean(finite)),
        "psi_median": float(np.median(finite)),
        "SE_boot": float(np.std(finite, ddof=1)),
        "psi_q025": float(np.quantile(finite, 0.025)),
        "psi_q975": float(np.quantile(finite, 0.975)),
    }


def run_baseline(dataset: str, *, B: int = 200, rho_coarsen: float = 0.30,
                 residualize: bool = True, seed_base: int = 99_000_000) -> Dict[str, Any]:
    """
    Run all three baselines on a dataset. Saves pkl + markdown.
    """
    print(f"\n=== Phase 19 baseline: {dataset} (B={B} bootstrap reps) ===")
    rho = rho_coarsen if DATASETS[dataset].get("binary_a") else None

    # Build DGP (residualized + coarsened — same as tuner uses)
    dgp = RealDataDGP(dataset, rho_coarsen=rho, residualize=residualize)

    # ─── Point estimates on the FULL residualized + coarsened sample ─────────
    from simulations.dgp.base import DGPSample
    full_sample = DGPSample(
        Y=dgp.Y, A=dgp.A, W=dgp.W, Z=dgp.Z,
        X=dgp.X, U=np.full(dgp.N, np.nan), psi_0=float("nan"),
    )

    point = {
        "ols_naive": _ols_naive(dgp.Y, dgp.A),
        "ols_with_X": _ols_with_X(dgp.Y, dgp.A, dgp.X),
        "twosls": _twosls(full_sample),
    }

    # ─── Bootstrap SE for each estimator ─────────────────────────────────────
    t0 = time.time()
    boot = {
        "ols_naive": _bootstrap_estimator(
            lambda s: _ols_naive(s.Y, s.A), dgp, B, seed_base + 0, "ols_naive"),
        "ols_with_X": _bootstrap_estimator(
            lambda s: _ols_with_X(s.Y, s.A, s.X), dgp, B, seed_base + 1000, "ols_with_X"),
        "twosls": _bootstrap_estimator(
            lambda s: _twosls(s), dgp, B, seed_base + 2000, "twosls"),
    }
    elapsed = time.time() - t0
    print(f"  Bootstrap done in {elapsed:.1f}s")

    # ─── Aggregate + save ────────────────────────────────────────────────────
    cfg = DATASETS[dataset]
    result = {
        "dataset": dataset,
        "N": int(dgp.N),
        "rho_coarsen": dgp.rho_coarsen,
        "residualize": dgp.residualize_info,
        "a_grid": [float(x) for x in dgp.a_grid],
        "ref_idx": int(dgp.ref_idx),
        "reference_estimate": cfg.get("reference_estimate"),
        "reference_source": cfg.get("reference_source"),
        "point": point,
        "bootstrap": boot,
        "B": B,
    }

    raw_dir = _RAW_BASE / dataset / "baseline"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = raw_dir / "baseline.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(result, fh)
    print(f"  Pickled: {pkl_path}")

    # ─── Markdown summary ────────────────────────────────────────────────────
    md_path = _SUMM_DIR / f"realdata_{dataset}_baseline.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    import datetime
    lines = [
        f"# Phase 19 — {dataset} triangulation baselines",
        f"*Generated {datetime.datetime.now().isoformat()}*",
        "",
        f"- N: {dgp.N}",
        f"- rho_coarsen: {dgp.rho_coarsen}",
        f"- a_grid: {[round(float(x), 4) for x in dgp.a_grid]}  ref_idx={dgp.ref_idx}",
        f"- Reference: {cfg.get('reference_estimate')} ({cfg.get('reference_source')})",
        f"- Bootstrap reps: {B}",
        "",
        "## Point estimates + bootstrap SE",
        "",
        "| Estimator | psi_hat | SE_analytic | SE_boot | 95% CI (bootstrap) |",
        "|---|---|---|---|---|",
    ]
    for key in ("ols_naive", "ols_with_X", "twosls"):
        p = point[key]; b = boot[key]
        ci = (f"[{b.get('psi_q025', float('nan')):.4f}, "
              f"{b.get('psi_q975', float('nan')):.4f}]") if b.get("B_ok", 0) >= 2 else "—"
        lines.append(
            f"| {key} | {p['psi_hat']:.4f} | "
            f"{p.get('SE_analytic', float('nan')):.4f} | "
            f"{b.get('SE_boot', float('nan')):.4f} | {ci} |"
        )

    if cfg.get("reference_estimate") is not None:
        ref = cfg["reference_estimate"]
        lines += [
            "",
            "## Comparison to literature reference",
            f"Reference: {ref:.4f} ({cfg.get('reference_source')})",
            "",
        ]
        for key in ("ols_naive", "ols_with_X", "twosls"):
            p = point[key]; b = boot[key]
            se = b.get("SE_boot", float("nan"))
            if np.isfinite(se) and se > 0:
                z = (p["psi_hat"] - ref) / se
                lines.append(f"- {key}: psi={p['psi_hat']:.4f} → distance to ref = {z:+.2f} σ_boot")

    if "twosls" in point:
        f = point["twosls"].get("stage1_F", float("nan"))
        lines += ["", "## 2SLS Stage 1 diagnostic",
                  f"- F-stat (Z~A) = {f:.1f} {'[strong]' if f >= 50 else '[moderate]' if f >= 10 else '[weak]'}"]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Markdown: {md_path}")

    # Console summary
    print("\n  Point estimates:")
    for key in ("ols_naive", "ols_with_X", "twosls"):
        p = point[key]; b = boot[key]
        print(f"    {key:12s}: psi={p['psi_hat']:+.4f}  "
              f"SE_an={p.get('SE_analytic', float('nan')):.4f}  "
              f"SE_boot={b.get('SE_boot', float('nan')):.4f}")

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    p.add_argument("--B", type=int, default=200, help="bootstrap reps")
    p.add_argument("--rho_coarsen", type=float, default=0.30)
    p.add_argument("--no_residualize", action="store_true")
    args = p.parse_args()
    run_baseline(args.dataset, B=args.B, rho_coarsen=args.rho_coarsen,
                 residualize=(not args.no_residualize))


if __name__ == "__main__":
    main()
