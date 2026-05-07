"""
Phase 10 — Lightweight Monte Carlo pilot for pipeline validation.

Validates the full end-to-end pipeline (DGP → method → estimate) across
6 methods × 2 DGPs before launching Exp1 (full coverage map).

Usage
-----
    python -m simulations.experiments.phase10_pilot           # full pilot
    python -m simulations.experiments.phase10_pilot --quick   # n=500, M=5

Output
------
    simulations/results/raw/phase10/            -- per-block .pkl files
    simulations/results/raw/phase10/boundary/   -- DGP2 boundary diagnostic
    simulations/results/summaries/phase10_summary.md

Exit criteria
-------------
    DRKernel estimated-q:   max_err < 0.35 vs J_policy_true, ESS_min_mean > 10
    DRKernel / DRDose oracle: max_err < 0.10
    No more than 10% reps in error per block
"""
from __future__ import annotations

import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ── Output directories ─────────────────────────────────────────────────────────
# phase10_pilot.py lives at pci_essay/simulations/experiments/
# parents[0] = experiments/  parents[1] = simulations/  parents[2] = pci_essay/
_ROOT = Path(__file__).parents[2]
_RAW_DIR = _ROOT / "simulations" / "results" / "raw" / "phase10"
_SUMM_DIR = _ROOT / "simulations" / "results" / "summaries"


# ══════════════════════════════════════════════════════════════════════════════
#  Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def _silverman_h(A: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth: 1.06 * std(A) * n^(-1/5)."""
    return 1.06 * float(np.std(A)) * len(A) ** (-0.2)


def _compute_J_policy_true(
    dgp,
    a_grid: np.ndarray,
    h_ref: float,
    n_quad: int = 25,
) -> np.ndarray:
    """
    Compute J_policy_true[k] = ∫ m(t) K_h(t-a_k) dt via Gauss-Hermite.

    Equivalent to E_{t ~ N(a_k, h_ref²)}[m(t)].
    - DGP1 (linear m):  J_policy_true = m(a)  (Jensen gap = 0)
    - DGP2 (concave m): J_policy_true < m(a)  (Jensen gap > 0)

    Uses probabilists' Hermite polynomials (hermegauss):
        ∫ f(x) exp(-x²/2) dx ≈ Σ wᵢ f(xᵢ)
    so  ∫ m(t) K_h(t-a) dt = (1/√2π) Σ wᵢ m(a + h·xᵢ)
    """
    from numpy.polynomial.hermite_e import hermegauss
    x, w = hermegauss(n_quad)
    J = []
    for a in a_grid:
        t_vals = float(a) + h_ref * x          # quadrature nodes N(a, h²)
        m_vals = np.array([dgp.m_true(float(t)) for t in t_vals])
        J.append(float(np.dot(w, m_vals) / np.sqrt(2.0 * np.pi)))
    return np.array(J)


# ══════════════════════════════════════════════════════════════════════════════
#  Scalar MC runner (Oracle / Naive / TwoStage via existing harness)
# ══════════════════════════════════════════════════════════════════════════════

def run_scalar_mc(
    dgp,
    method_cls,
    n: int,
    M: int,
    seed_base: int,
    label: str,
    save_path: Optional[Path] = None,
) -> dict:
    """
    Run M replications of a scalar estimator (Oracle, Naive, TwoStage).

    Wraps simulations.experiments.harness.run_monte_carlo.
    psi_0 = dgp.psi_true(a=1.0) — constant for DGP1 (= alpha = 0.30).

    Returns
    -------
    dict with keys: df, metrics, psi_0, label, n, M
    """
    from simulations.analysis.metrics import compute_metrics
    from simulations.experiments.harness import run_monte_carlo

    if save_path is None:
        save_path = _RAW_DIR / f"{label}_n{n}_M{M}.pkl"

    df = run_monte_carlo(
        dgp, method_cls, n, M, seed_base, save_path=save_path
    )
    psi_0 = dgp.psi_true(a=1.0)      # 0.30 for DGP1
    metrics = compute_metrics(
        df["psi_hat"].values, df["V_hat"].values, psi_0, n
    )
    return {
        "df": df, "metrics": metrics,
        "psi_0": psi_0, "label": label, "n": n, "M": M,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Curve MC runner (DRDoseResponse / DRKernel — custom loop)
# ══════════════════════════════════════════════════════════════════════════════

def run_curve_mc(
    dgp,
    method_factory: Callable,
    a_grid: np.ndarray,
    J_policy_true: np.ndarray,
    n: int,
    M: int,
    seed_base: int,
    label: str,
    h_ref: float,
    save_path: Optional[Path] = None,
) -> dict:
    """
    Run M replications of a curve estimator (DRDoseResponse / DRKernel).

    method_factory: callable() → fresh PCIEstimator instance (closure over dgp,
                    a_grid, h_ref — avoids no-arg harness constraint).

    Failed reps (exception) → record with "error": str(e); not re-raised.

    Returns
    -------
    dict {"records": list[dict], "meta": {...}}
    Saves pkl to save_path (auto-named if None).
    """
    records = []
    K = len(a_grid)
    _nan_K = [np.nan] * K

    for i in range(M):
        seed = seed_base + i
        sample = dgp.generate(n=n, seed=seed)
        try:
            est = method_factory()
            est.fit(sample, m_true_fn=dgp.m_true)
            res = est.estimate()
            ex = res.extra

            def _arr(key):
                v = ex.get(key, np.full(K, np.nan))
                return v.tolist() if hasattr(v, "tolist") else list(v)

            record = {
                "seed": seed,
                "psi_hat": float(res.psi_hat) if np.isfinite(res.psi_hat) else np.nan,
                "V_hat": float(res.V_hat) if np.isfinite(res.V_hat) else np.nan,
                "J_dr":                    _arr("J_dr"),
                "J_reg":                   _arr("J_reg"),
                "J_policy_true_rep":       _arr("J_policy_true"),
                "ESS_grid":                _arr("ESS_grid"),
                "ESS_min":                 float(ex.get("ESS_min", np.nan)),
                "ESS_ratio_min":           float(ex.get("ESS_ratio_min", np.nan)),
                # Phase 9B.2 diagnostics (present for DRKernel; NaN for DRDose)
                "riesz_residual_grid_mean": _arr("riesz_residual_grid_mean"),
                "cond_M_grid_mean":         _arr("cond_M_grid_mean"),
                "neg_share_grid_mean":      _arr("neg_share_grid_mean"),
                "weight_p99_grid":          _arr("weight_p99_grid"),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            record = {
                "seed": seed,
                "psi_hat": np.nan, "V_hat": np.nan,
                "J_dr": _nan_K, "J_reg": _nan_K,
                "J_policy_true_rep": _nan_K,
                "ESS_grid": _nan_K, "ESS_min": np.nan, "ESS_ratio_min": np.nan,
                "riesz_residual_grid_mean": _nan_K,
                "cond_M_grid_mean": _nan_K,
                "neg_share_grid_mean": _nan_K,
                "weight_p99_grid": _nan_K,
                "error": str(e),
            }
            warnings.warn(
                f"[{label} n={n} seed={seed}] Error: {e}", RuntimeWarning, stacklevel=2
            )
        records.append(record)

    data = {
        "records": records,
        "meta": {
            "label": label,
            "dgp": dgp.__class__.__name__,
            "n": n, "M": M, "seed_base": seed_base, "h_ref": h_ref,
            "a_grid": a_grid.tolist(),
            "J_policy_true": J_policy_true.tolist(),
            # Pre-cache m_true to avoid repeated GH calls in summarize
            "J_true": [dgp.m_true(float(a)) for a in a_grid],
        },
    }

    if save_path is None:
        save_path = _RAW_DIR / f"{label}_n{n}_M{M}.pkl"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved -> {save_path}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  Aggregate metrics from a curve block
# ══════════════════════════════════════════════════════════════════════════════

def summarize_curve(data: dict) -> dict:
    """
    Compute aggregate metrics from run_curve_mc output.

    Primary target: J_policy_true = ∫ m(t) K_h(t-a) dt  (not m(a) point-wise).
    ref_idx = K // 2 (central dose) for scalar summary metrics.

    Returns
    -------
    dict with: M_ok, n_errors, bias_ref, rmse_ref, max_err, coverage, ser,
               mise, ess_min_mean, bias_per_dose, rmse_per_dose.
    """
    records = data["records"]
    J_pt = np.array(data["meta"]["J_policy_true"])  # (K,) ground truth
    K = len(J_pt)

    ok = [r for r in records if r["error"] is None]
    M_ok = len(ok)
    n_errors = len(records) - M_ok

    if M_ok == 0:
        return {"M_ok": 0, "n_errors": n_errors}

    J_dr_all = np.array([r["J_dr"] for r in ok])       # (M_ok, K)
    psi_all  = np.array([r["psi_hat"] for r in ok])    # (M_ok,)
    V_all    = np.array([r["V_hat"] for r in ok])      # (M_ok,)
    ESS_min  = np.array([r["ESS_min"] for r in ok])    # (M_ok,)

    bias_d = np.mean(J_dr_all, axis=0) - J_pt          # (K,)
    rmse_d = np.sqrt(np.mean((J_dr_all - J_pt[None]) ** 2, axis=0))  # (K,)
    mise   = float(np.mean(np.mean((J_dr_all - J_pt[None]) ** 2, axis=1)))

    ref_idx = K // 2
    psi_0_ref = float(J_pt[ref_idx])
    se_all    = np.sqrt(np.abs(V_all))
    coverage  = float(np.mean(np.abs(psi_all - psi_0_ref) <= 1.96 * se_all))
    ser       = float(np.std(psi_all) / (np.mean(se_all) + 1e-12))

    return {
        "M_ok": M_ok,
        "n_errors": n_errors,
        "bias_ref":     float(bias_d[ref_idx]),
        "rmse_ref":     float(rmse_d[ref_idx]),
        "max_err":      float(np.max(np.abs(bias_d))),
        "coverage":     coverage,
        "ser":          ser,
        "mise":         mise,
        # nanmean returns NaN (not warning) when all values are NaN (e.g. DRDoseResponse)
        "ess_min_mean": float(np.nanmean(ESS_min)) if np.any(np.isfinite(ESS_min)) else np.nan,
        "bias_per_dose": bias_d.tolist(),
        "rmse_per_dose": rmse_d.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Method factories (closures over dgp + a_grid + h_ref)
# ══════════════════════════════════════════════════════════════════════════════

def make_dr_dose_response(dgp, a_grid: np.ndarray, h_ref: float,
                          is_linear_in_A: bool = False) -> Callable:
    """DRDoseResponse with oracle h and oracle q."""
    def factory():
        from simulations.methods._oracle_bridges import OracleBridgeH, OracleBridgeQ
        from simulations.methods.dr_dose_response import DRDoseResponse
        return DRDoseResponse(
            a_grid=a_grid,
            h_model=OracleBridgeH(dgp, is_linear_in_A=is_linear_in_A),
            q_model=OracleBridgeQ(dgp),
            bandwidth=h_ref,
        )
    return factory


def make_dr_kernel_oracle(dgp, a_grid: np.ndarray, h_ref: float) -> Callable:
    """DRKernel with KPV h-bridge (estimated cross-fit) + oracle q."""
    ref_idx = len(a_grid) // 2
    def factory():
        from simulations.methods._oracle_bridges import OracleBridgeQ
        from simulations.methods.dr_kernel import DRKernel
        return DRKernel(
            a_grid=a_grid,
            q_model=OracleBridgeQ(dgp),
            bandwidth=h_ref,
            n_folds=5,
            random_state=42,
            ref_dose_index=ref_idx,
            cross_fit_q=False,
        )
    return factory


def make_dr_kernel_estimated(dgp, a_grid: np.ndarray, h_ref: float) -> Callable:
    """DRKernel with KPV h-bridge + KPVPolicyBridgeQ cross-fit (Phase 9B.2)."""
    ref_idx = len(a_grid) // 2
    def factory():
        from simulations.methods.dr_kernel import DRKernel
        from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
        q_model = KPVPolicyBridgeQ(
            a_grid=a_grid,
            h_KDE=h_ref,
            lambda_Q=1e-3,
        )
        return DRKernel(
            a_grid=a_grid,
            q_model=q_model,
            bandwidth=h_ref,
            n_folds=5,
            random_state=42,
            ref_dose_index=ref_idx,
            cross_fit_q=True,
        )
    return factory


# ══════════════════════════════════════════════════════════════════════════════
#  Summary markdown writer
# ══════════════════════════════════════════════════════════════════════════════

def write_summary_md(
    all_results: dict,
    a_grid_dgp1: np.ndarray,
    a_grid_dgp2_in: np.ndarray,
    a_grid_dgp2_bound: np.ndarray,
) -> None:
    """Write phase10_summary.md with per-DGP tables and exit-criteria checklist."""
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    out = _SUMM_DIR / "phase10_summary.md"

    lines = []
    lines.append(f"# Phase 10 — Monte Carlo pilot")
    lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # ── DGP1 scalar table ────────────────────────────────────────────────────
    lines.append("## DGP1 — Cobb-Douglas (snr_W=snr_Z=0.95)")
    lines.append(f"a_grid = {a_grid_dgp1.tolist()},  psi_0 = alpha = 0.30")
    lines.append("")
    lines.append("### Méthodes scalaires (vs psi_0 = 0.30)")
    lines.append("| Méthode | n | M_ok | bias | RMSE | coverage | SER |")
    lines.append("|---------|---|------|------|------|----------|-----|")
    for lbl in ["oracle", "naive", "tsls"]:
        for n in [500, 1000]:
            key = (lbl, "dgp1", n)
            if key not in all_results:
                continue
            r = all_results[key]
            m = r["metrics"]
            lines.append(
                f"| {lbl} | {n} | {r['M']} | {m['bias']:+.4f} | "
                f"{m['rmse']:.4f} | {m['coverage']:.3f} | {m['ser']:.3f} |"
            )
    lines.append("")

    lines.append("### Méthodes curve (vs J_policy_true = ∫ m(t) K_h(t-a) dt)")
    lines.append("| Méthode | n | M_ok | n_err | bias_ref | max_err | RMSE_ref | coverage | SER | MISE | ESS_min_mean |")
    lines.append("|---------|---|------|-------|----------|---------|----------|----------|-----|------|--------------|")
    for lbl in ["dr_dose_oracle", "dr_kernel_oracle", "dr_kernel_xfit"]:
        for n in [500, 1000]:
            key = (lbl, "dgp1", n)
            if key not in all_results:
                continue
            s = all_results[key]
            if s.get("M_ok", 0) == 0:
                lines.append(f"| {lbl} | {n} | 0 | {s.get('n_errors', '?')} | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {lbl} | {n} | {s['M_ok']} | {s['n_errors']} | "
                f"{s['bias_ref']:+.4f} | {s['max_err']:.4f} | {s['rmse_ref']:.4f} | "
                f"{s['coverage']:.3f} | {s['ser']:.3f} | {s['mise']:.4f} | "
                f"{s['ess_min_mean']:.1f} |"
            )
    lines.append("")

    # ── DGP2 in-support table ────────────────────────────────────────────────
    lines.append("## DGP2 — Michaelis-Menten in-support (snr_W=snr_Z=0.95)")
    lines.append(f"a_grid = {a_grid_dgp2_in.tolist()}")
    lines.append("*Note: TwoStage excluded (DGP2 is non-linear).*")
    lines.append("*Note: J_true = m(a) ≠ J_policy_true for concave m — Jensen gap reported.*")
    lines.append("")
    lines.append("| Méthode | n | M_ok | n_err | bias_ref | max_err | RMSE_ref | coverage | SER | MISE | ESS_min_mean |")
    lines.append("|---------|---|------|-------|----------|---------|----------|----------|-----|------|--------------|")
    for lbl in ["dr_dose_oracle", "dr_kernel_oracle", "dr_kernel_xfit"]:
        for n in [500, 1000]:
            key = (lbl, "dgp2_in", n)
            if key not in all_results:
                continue
            s = all_results[key]
            if s.get("M_ok", 0) == 0:
                lines.append(f"| {lbl} | {n} | 0 | {s.get('n_errors', '?')} | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {lbl} | {n} | {s['M_ok']} | {s['n_errors']} | "
                f"{s['bias_ref']:+.4f} | {s['max_err']:.4f} | {s['rmse_ref']:.4f} | "
                f"{s['coverage']:.3f} | {s['ser']:.3f} | {s['mise']:.4f} | "
                f"{s['ess_min_mean']:.1f} |"
            )
    lines.append("")

    # ── DGP2 boundary diagnostic ─────────────────────────────────────────────
    lines.append("## DGP2 — Boundary diagnostic (M=20, n=1000, NOT pass/fail)")
    lines.append(f"a_grid = {a_grid_dgp2_bound.tolist()}")
    lines.append("*Warning: ESS collapse expected at a < 1.0 and a > 3.5.*")
    lines.append("")
    key_b = ("boundary_xfit", "dgp2_bound", 1000)
    s_b = all_results.get(key_b)
    if s_b and s_b.get("M_ok", 0) > 0:
        lines.append(f"M_ok={s_b['M_ok']}, n_errors={s_b['n_errors']}, ESS_min_mean={s_b['ess_min_mean']:.1f}")
        lines.append("")
        lines.append("| dose_idx | a | bias | |err| | ESS_min_mean |")
        lines.append("|----------|---|------|-------|--------------|")
        for d, a in enumerate(a_grid_dgp2_bound):
            b = s_b["bias_per_dose"][d]
            r = s_b["rmse_per_dose"][d]
            lines.append(f"| {d} | {a:.1f} | {b:+.4f} | {r:.4f} | {s_b['ess_min_mean']:.1f} |")
    else:
        lines.append("*No results or all reps failed.*")
    lines.append("")

    # ── Exit criteria checklist ───────────────────────────────────────────────
    lines.append("## Critères de sortie Phase 10")
    lines.append("")

    def _pass(cond):
        return "✅" if cond else "❌"

    def _fmt(val, fmt):
        """Format val with fmt spec; return '—' if val is None/missing."""
        if val is None:
            return "—"
        try:
            return format(val, fmt)
        except (TypeError, ValueError):
            return str(val)

    def _curve(lbl, dgp_tag, n):
        key = (lbl, dgp_tag, n)
        return all_results.get(key, {})

    # DRKernel xfit DGP1 n=1000
    s_xfit_1 = _curve("dr_kernel_xfit", "dgp1", 1000)
    ok_err_1  = s_xfit_1.get("max_err", 9999) < 0.35
    ok_ess_1  = s_xfit_1.get("ess_min_mean", 0) > 10
    lines.append("| Critère | Valeur cible | DGP1 n=1000 | DGP2 in-support n=1000 |")
    lines.append("|---------|-------------|-------------|------------------------|")

    s_xfit_2 = _curve("dr_kernel_xfit", "dgp2_in", 1000)
    ok_err_2  = s_xfit_2.get("max_err", 9999) < 0.35
    ok_ess_2  = s_xfit_2.get("ess_min_mean", 0) > 10
    lines.append(
        f"| DRKernel xfit max_err | < 0.35 | "
        f"{_pass(ok_err_1)} {_fmt(s_xfit_1.get('max_err'), '.4f')} | "
        f"{_pass(ok_err_2)} {_fmt(s_xfit_2.get('max_err'), '.4f')} |"
    )
    lines.append(
        f"| DRKernel xfit ESS_min_mean | > 10 | "
        f"{_pass(ok_ess_1)} {_fmt(s_xfit_1.get('ess_min_mean'), '.1f')} | "
        f"{_pass(ok_ess_2)} {_fmt(s_xfit_2.get('ess_min_mean'), '.1f')} |"
    )

    s_orck_1 = _curve("dr_kernel_oracle", "dgp1", 1000)
    ok_orck_1 = s_orck_1.get("max_err", 9999) < 0.10
    s_orck_2 = _curve("dr_kernel_oracle", "dgp2_in", 1000)
    ok_orck_2 = s_orck_2.get("max_err", 9999) < 0.10
    lines.append(
        f"| DRKernel oracle max_err | < 0.10 | "
        f"{_pass(ok_orck_1)} {_fmt(s_orck_1.get('max_err'), '.4f')} | "
        f"{_pass(ok_orck_2)} {_fmt(s_orck_2.get('max_err'), '.4f')} |"
    )

    s_dose_1 = _curve("dr_dose_oracle", "dgp1", 1000)
    ok_dose_1 = s_dose_1.get("max_err", 9999) < 0.10
    s_dose_2 = _curve("dr_dose_oracle", "dgp2_in", 1000)
    ok_dose_2 = s_dose_2.get("max_err", 9999) < 0.10
    lines.append(
        f"| DRDoseResponse oracle max_err | < 0.10 | "
        f"{_pass(ok_dose_1)} {_fmt(s_dose_1.get('max_err'), '.4f')} | "
        f"{_pass(ok_dose_2)} {_fmt(s_dose_2.get('max_err'), '.4f')} |"
    )
    lines.append("")

    all_go = all([ok_err_1, ok_ess_1, ok_err_2, ok_ess_2])
    lines.append(f"**Phase 10 GO/NO-GO: {'✅ GO — Exp1 can launch' if all_go else '❌ NO-GO — review above'}**")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Summary -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main orchestration
# ══════════════════════════════════════════════════════════════════════════════

def main(
    n_list: tuple = (500, 1000),
    M: int = 50,
    M_boundary: int = 20,
    seed_base_dgp1: int = 1000,
    seed_base_dgp2: int = 2000,
    seed_base_boundary: int = 3000,
) -> dict:
    """
    Run the full Phase 10 pilot.

    Parameters
    ----------
    n_list       : sample sizes to sweep (default (500, 1000))
    M            : replications per block (default 50)
    M_boundary   : reps for DGP2 boundary diagnostic (default 20)
    seed_base_*  : seed offsets per DGP block

    Returns
    -------
    all_results : dict keyed by (label, dgp_tag, n)
    """
    from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    from simulations.methods.naive import NaiveRegression
    from simulations.methods.oracle import OracleDirect
    from simulations.methods.two_stage_linear import TwoStageLeastSquares

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    (_RAW_DIR / "boundary").mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)

    dgp1 = CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)
    dgp2 = MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)

    a_grid_dgp1       = np.array([-0.5, 0.0, 0.5])
    a_grid_dgp2_in    = np.array([1.8, 2.2, 2.6, 3.0])
    a_grid_dgp2_bound = np.array([0.5, 1.0, 1.5, 2.5, 4.0])

    all_results: dict = {}

    # ── Bloc 1 : DGP1 scalar (Oracle / Naive / TwoStage) ─────────────────────
    print("\n" + "=" * 60)
    print("Bloc 1 — DGP1 scalar")
    print("=" * 60)
    for n in n_list:
        for method_cls, lbl in [
            (OracleDirect, "oracle"),
            (NaiveRegression, "naive"),
            (TwoStageLeastSquares, "tsls"),
        ]:
            print(f"\n  [{lbl} n={n} M={M}]")
            all_results[(lbl, "dgp1", n)] = run_scalar_mc(
                dgp1, method_cls, n, M, seed_base_dgp1,
                label=f"{lbl}_dgp1",
            )

    # ── Bloc 2 : DGP1 curve ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Bloc 2 — DGP1 curve (DRDoseResponse / DRKernel oracle / xfit)")
    print("=" * 60)
    for n in n_list:
        ref_sample = dgp1.generate(n=n, seed=0)
        h_ref = _silverman_h(ref_sample.A)
        J_pt  = _compute_J_policy_true(dgp1, a_grid_dgp1, h_ref)
        print(f"\n  [DGP1 n={n}]  h_ref={h_ref:.4f}  J_policy_true={J_pt}")

        for factory_fn, lbl in [
            (make_dr_dose_response(dgp1, a_grid_dgp1, h_ref, is_linear_in_A=True),
             "dr_dose_oracle"),
            (make_dr_kernel_oracle(dgp1, a_grid_dgp1, h_ref),
             "dr_kernel_oracle"),
            (make_dr_kernel_estimated(dgp1, a_grid_dgp1, h_ref),
             "dr_kernel_xfit"),
        ]:
            print(f"\n  [{lbl} n={n} M={M}]")
            data = run_curve_mc(
                dgp1, factory_fn, a_grid_dgp1, J_pt,
                n=n, M=M, seed_base=seed_base_dgp1,
                label=f"{lbl}_dgp1", h_ref=h_ref,
            )
            all_results[(lbl, "dgp1", n)] = summarize_curve(data)

    # ── Bloc 3 : DGP2 in-support ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Bloc 3 — DGP2 in-support (DRDoseResponse / DRKernel oracle / xfit)")
    print("=" * 60)
    for n in n_list:
        ref_sample = dgp2.generate(n=n, seed=0)
        h_ref = _silverman_h(ref_sample.A)
        J_pt  = _compute_J_policy_true(dgp2, a_grid_dgp2_in, h_ref)
        print(f"\n  [DGP2-in n={n}]  h_ref={h_ref:.4f}  J_policy_true={J_pt}")

        for factory_fn, lbl in [
            (make_dr_dose_response(dgp2, a_grid_dgp2_in, h_ref, is_linear_in_A=False),
             "dr_dose_oracle"),
            (make_dr_kernel_oracle(dgp2, a_grid_dgp2_in, h_ref),
             "dr_kernel_oracle"),
            (make_dr_kernel_estimated(dgp2, a_grid_dgp2_in, h_ref),
             "dr_kernel_xfit"),
        ]:
            print(f"\n  [{lbl} n={n} M={M}]")
            data = run_curve_mc(
                dgp2, factory_fn, a_grid_dgp2_in, J_pt,
                n=n, M=M, seed_base=seed_base_dgp2,
                label=f"{lbl}_dgp2_in", h_ref=h_ref,
            )
            all_results[(lbl, "dgp2_in", n)] = summarize_curve(data)

    # ── Bloc 4 : DGP2 boundary diagnostic ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Bloc 4 — DGP2 boundary (diagnostic, M=20)")
    print("=" * 60)
    n_b = 1000
    ref_sample_b = dgp2.generate(n=n_b, seed=0)
    h_ref_b = _silverman_h(ref_sample_b.A)
    J_pt_b  = _compute_J_policy_true(dgp2, a_grid_dgp2_bound, h_ref_b)
    print(f"\n  [DGP2-boundary n={n_b}]  h_ref={h_ref_b:.4f}")
    print(f"  J_policy_true={J_pt_b}")

    data_b = run_curve_mc(
        dgp2,
        make_dr_kernel_estimated(dgp2, a_grid_dgp2_bound, h_ref_b),
        a_grid_dgp2_bound, J_pt_b,
        n=n_b, M=M_boundary, seed_base=seed_base_boundary,
        label="dr_kernel_xfit_dgp2_bound", h_ref=h_ref_b,
        save_path=_RAW_DIR / "boundary" / f"dr_kernel_xfit_dgp2_bound_n{n_b}_M{M_boundary}.pkl",
    )
    all_results[("boundary_xfit", "dgp2_bound", n_b)] = summarize_curve(data_b)

    # ── Bloc 5 : Summary markdown ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Bloc 5 — Writing summary markdown")
    print("=" * 60)
    write_summary_md(all_results, a_grid_dgp1, a_grid_dgp2_in, a_grid_dgp2_bound)

    # ── Final status ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Phase 10 pilot DONE.")
    print(f"Results  -> {_RAW_DIR}")
    print(f"Summary  -> {_SUMM_DIR / 'phase10_summary.md'}")
    s_check = all_results.get(("dr_kernel_xfit", "dgp1", 1000), {})
    if s_check:
        go = s_check.get("max_err", 9999) < 0.35 and s_check.get("ess_min_mean", 0) > 10
        status = "GO" if go else "NO-GO"
        me = s_check.get("max_err")
        ess = s_check.get("ess_min_mean")
        me_s = f"{me:.4f}" if me is not None else "—"
        ess_s = f"{ess:.1f}" if ess is not None else "—"
        print(f"DRKernel xfit DGP1 n=1000: max_err={me_s}, ESS_min_mean={ess_s}  [{status}]")
    print(f"{'=' * 60}\n")

    return all_results


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    if quick:
        print("[QUICK MODE] n=500, M=5, M_boundary=3")
        main(n_list=(500,), M=5, M_boundary=3)
    else:
        main()
