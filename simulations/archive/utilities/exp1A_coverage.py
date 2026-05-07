"""
Exp1A — Controlled Coverage Map.

Runs a full SNR × n × method grid to validate H4 (nominal 95% coverage at
favourable SNR, progressive degradation as SNR decreases).

Grid
----
DGPs    : DGP1 (Cobb-Douglas linear), DGP2 (Michaelis-Menten non-linear)
SNR     : 0.95, 0.80, 0.60  (snr_W = snr_Z in each case)
n       : 500, 1000, 2000
M       : 500 reps per block  (smoke/pilot overrideable via CLI args)
Methods : Oracle, Naive, TwoStage (DGP1 only),
          DRDoseResponse oracle, DRKernel oracle-q, DRKernel xfit

Output
------
    simulations/results/raw/exp1A/      -- one .pkl per block
    simulations/results/summaries/exp1A_summary.md

Checkpoint / resume
-------------------
Level 1 — block:   if {label}.pkl exists → skip (block complete).
Level 2 — intra:   partial checkpoint every CHECKPOINT_FREQ reps to
                   {label}.partial.pkl. Resumes if .partial.pkl found
                   but .pkl absent (interrupted mid-block).

Usage
-----
    python -m simulations.archive.utilities.exp1A_coverage               # full run
    python -m simulations.archive.utilities.exp1A_coverage --snr 0.95 --n 500 --M 10
    python -m simulations.archive.utilities.exp1A_coverage --pilot        # n≤1000, M=50
"""
from __future__ import annotations

import argparse
import copy
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ── Output directories ─────────────────────────────────────────────────────────
# exp1A_coverage.py lives at pci_essay/simulations/experiments/
_ROOT     = Path(__file__).parents[2]
_RAW_DIR  = _ROOT / "simulations" / "results" / "raw" / "exp1A"
_SUMM_DIR = _ROOT / "simulations" / "results" / "summaries"

# ── Grid constants ─────────────────────────────────────────────────────────────
SNR_GRID  = [0.95, 0.80, 0.60]
N_LIST    = [500, 1000, 2000]
M_MAIN    = 500

A_GRID_DGP1    = np.array([-0.5, 0.0,  0.5])
A_GRID_DGP2_IN = np.array([ 1.8, 2.2,  2.6, 3.0])

# DGP configs: (name, a_grid, is_linear_in_A, method_labels)
_DGP_CONFIGS: List[Tuple] = [
    (
        "dgp1", A_GRID_DGP1, True,
        ["oracle", "naive", "tsls", "dr_dose", "drk_oracle", "drk_xfit"],
    ),
    (
        "dgp2", A_GRID_DGP2_IN, False,
        ["oracle", "naive", "dr_dose", "drk_oracle", "drk_xfit"],
    ),
]

# Index maps for deterministic seeds
_DGP_IDX    = {"dgp1": 0, "dgp2": 1}
_SNR_IDX    = {0.95: 0, 0.80: 1, 0.60: 2}
_N_IDX      = {500: 0, 1000: 1, 2000: 2}
_METHOD_IDX = {
    "oracle": 0, "naive": 1, "tsls": 2,
    "dr_dose": 3, "drk_oracle": 4, "drk_xfit": 5,
}

CHECKPOINT_FREQ = 25     # partial save every N reps inside a block


# ══════════════════════════════════════════════════════════════════════════════
#  Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def _silverman_h(A: np.ndarray) -> float:
    """Silverman's rule-of-thumb: 1.06 × std(A) × n^(-1/5)."""
    return 1.06 * float(np.std(A)) * len(A) ** (-0.2)


def _compute_J_policy_true(
    dgp,
    a_grid: np.ndarray,
    h_ref: float,
    n_quad: int = 25,
) -> np.ndarray:
    """J_policy_true[k] = ∫ m(t) K_h(t-a_k) dt via Gauss-Hermite quadrature.

    Equivalent to E_{t ~ N(a_k, h_ref²)}[m(t)].
    For linear m (DGP1): J_policy_true = m(a)  (Jensen gap = 0).
    For concave m (DGP2): J_policy_true < m(a) (Jensen gap > 0).
    """
    from numpy.polynomial.hermite_e import hermegauss
    x, w = hermegauss(n_quad)
    J = []
    for a in a_grid:
        t_vals = float(a) + h_ref * x
        m_vals = np.array([dgp.m_true(float(t)) for t in t_vals])
        J.append(float(np.dot(w, m_vals) / np.sqrt(2.0 * np.pi)))
    return np.array(J)


def _block_seed(dgp_name: str, snr: float, n: int, method_label: str) -> int:
    """Deterministic seed: spacing 1 000 > M=500 → no collision between blocks."""
    return (
        _DGP_IDX[dgp_name]         * 1_000_000
        + _SNR_IDX[snr]            *   100_000
        + _N_IDX[n]                *    10_000
        + _METHOD_IDX[method_label] *    1_000
    )


def _block_label(dgp_name: str, method_label: str, snr: float, n: int, M: int) -> str:
    """Canonical label for a block — used as pkl filename stem."""
    snr_str = f"{int(round(snr * 100)):03d}"
    return f"exp1A_{dgp_name}_{method_label}_snr{snr_str}_n{n}_M{M}"


def _nanmean_2d(arr: np.ndarray, K: int) -> list:
    """nanmean over axis=0 of a (M, K) array; return list of NaN if all values are NaN."""
    if not np.any(np.isfinite(arr)):
        return [np.nan] * K
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(arr, axis=0).tolist()


def _fmt(val, fmt: str) -> str:
    """Format a value; return '—' if None/NaN."""
    if val is None:
        return "—"
    try:
        f = format(float(val), fmt)
        return f
    except (TypeError, ValueError):
        return str(val)


# ══════════════════════════════════════════════════════════════════════════════
#  DGP factory
# ══════════════════════════════════════════════════════════════════════════════

def make_dgp(name: str, snr: float):
    """Create a fresh DGP instance for the given name and SNR level."""
    if name == "dgp1":
        from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP
        return CobbDouglasLinearDGP(snr_W=snr, snr_Z=snr)
    elif name == "dgp2":
        from simulations.dgp.michaelis_menten import MichaelisMentenDGP
        return MichaelisMentenDGP(snr_W=snr, snr_Z=snr)
    else:
        raise ValueError(f"Unknown DGP name: {name!r}")


# ══════════════════════════════════════════════════════════════════════════════
#  Method factories
# ══════════════════════════════════════════════════════════════════════════════

def _make_factory(
    dgp,
    a_grid: np.ndarray,
    h_ref: float,
    is_linear_in_A: bool,
    method_label: str,
) -> Callable:
    """Return a zero-arg factory closure for the requested curve method."""
    if method_label == "dr_dose":
        return _factory_dr_dose(dgp, a_grid, h_ref, is_linear_in_A)
    elif method_label == "drk_oracle":
        return _factory_drk_oracle(dgp, a_grid, h_ref)
    elif method_label == "drk_xfit":
        return _factory_drk_xfit(dgp, a_grid, h_ref)
    else:
        raise ValueError(f"Unknown curve method: {method_label!r}")


def _factory_dr_dose(dgp, a_grid, h_ref, is_linear_in_A):
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


def _factory_drk_oracle(dgp, a_grid, h_ref):
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


def _factory_drk_xfit(dgp, a_grid, h_ref):
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
#  Per-rep record builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_record(res, K: int, seed: int) -> dict:
    """Extract all relevant fields from an EstimationResult.extra dict."""
    ex = res.extra

    def _arr(key):
        v = ex.get(key, np.full(K, np.nan))
        return v.tolist() if hasattr(v, "tolist") else list(v)

    def _scl(key):
        return float(ex.get(key, np.nan))

    return {
        "seed":    seed,
        "psi_hat": float(res.psi_hat) if np.isfinite(res.psi_hat) else np.nan,
        "V_hat":   float(res.V_hat)   if np.isfinite(res.V_hat)   else np.nan,
        # Curves
        "J_dr":              _arr("J_dr"),
        "J_reg":             _arr("J_reg"),
        "J_policy_true_rep": _arr("J_policy_true"),
        # ESS
        "ESS_grid":      _arr("ESS_grid"),
        "ESS_min":       _scl("ESS_min"),
        "ESS_ratio_min": _scl("ESS_ratio_min"),
        # Variance decomposition
        "V_hat_grid":        _arr("V_hat_grid"),
        "V_reg_grid":        _arr("V_reg_grid"),
        "V_correction_grid": _arr("V_correction_grid"),
        # IPW weights
        "weight_p99_grid": _arr("weight_p99_grid"),
        "weight_max_grid": _arr("weight_max_grid"),
        "q_clip_fraction": _scl("q_clip_fraction"),
        # Riesz diagnostics (NaN if oracle-q or global-q)
        "riesz_residual_grid_mean": _arr("riesz_residual_grid_mean"),
        "riesz_residual_grid_max":  _arr("riesz_residual_grid_max"),
        "cond_M_grid_mean":         _arr("cond_M_grid_mean"),
        "cond_M_grid_max":          _arr("cond_M_grid_max"),
        "neg_share_grid_mean":      _arr("neg_share_grid_mean"),
        "neg_share_grid_max":       _arr("neg_share_grid_max"),
        "error": None,
    }


def _build_error_record(e: Exception, K: int, seed: int) -> dict:
    _nan_K = [np.nan] * K
    return {
        "seed":    seed,
        "psi_hat": np.nan, "V_hat": np.nan,
        "J_dr": _nan_K, "J_reg": _nan_K, "J_policy_true_rep": _nan_K,
        "ESS_grid": _nan_K, "ESS_min": np.nan, "ESS_ratio_min": np.nan,
        "V_hat_grid": _nan_K, "V_reg_grid": _nan_K, "V_correction_grid": _nan_K,
        "weight_p99_grid": _nan_K, "weight_max_grid": _nan_K,
        "q_clip_fraction": np.nan,
        "riesz_residual_grid_mean": _nan_K, "riesz_residual_grid_max": _nan_K,
        "cond_M_grid_mean": _nan_K, "cond_M_grid_max": _nan_K,
        "neg_share_grid_mean": _nan_K, "neg_share_grid_max": _nan_K,
        "error": str(e),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Curve block runner — two-level checkpoint/resume
# ══════════════════════════════════════════════════════════════════════════════

def run_curve_block(
    dgp,
    method_factory: Callable,
    a_grid: np.ndarray,
    J_policy_true: np.ndarray,
    n: int,
    M: int,
    seed_base: int,
    label: str,
    h_ref: float,
    raw_dir: Path | None = None,
) -> dict:
    """
    Run M replications of a curve estimator with two-level checkpointing.

    Level 1 — if {label}.pkl exists: skip entirely (block complete).
    Level 2 — partial checkpoint every CHECKPOINT_FREQ reps to {label}.partial.pkl;
               resumed automatically on restart.

    Parameters
    ----------
    raw_dir : Path, optional
        Override directory for pkl files. Defaults to module-level _RAW_DIR.
        Useful for sensitivity grids and test fixtures.

    Returns
    -------
    dict {"records": list[dict], "meta": {...}}
    """
    base_dir = raw_dir if raw_dir is not None else _RAW_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    save_path    = base_dir / f"{label}.pkl"
    partial_path = base_dir / f"{label}.partial.pkl"

    # Level 1: block already complete
    if save_path.exists():
        print(f"  [SKIP] {label}")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    # Level 2: resume from partial checkpoint
    if partial_path.exists():
        with open(partial_path, "rb") as f:
            records = pickle.load(f)
        start_i = len(records)
        print(f"  [RESUME] {label} rep {start_i}/{M}")
    else:
        records = []
        start_i = 0

    K = len(a_grid)
    print(f"  [RUN] {label} (rep {start_i}->{M})")

    for i in range(start_i, M):
        seed   = seed_base + i
        sample = dgp.generate(n=n, seed=seed)
        try:
            est = method_factory()
            est.fit(sample, m_true_fn=dgp.m_true)
            res = est.estimate()
            records.append(_build_record(res, K, seed))
        except Exception as e:           # noqa: BLE001
            records.append(_build_error_record(e, K, seed))
            warnings.warn(
                f"[{label} seed={seed}] {type(e).__name__}: {e}",
                RuntimeWarning, stacklevel=2,
            )

        # Partial checkpoint
        if (i + 1) % CHECKPOINT_FREQ == 0:
            with open(partial_path, "wb") as f:
                pickle.dump(records, f)

    # Finalise block
    data = {
        "records": records,
        "meta": {
            "label": label, "dgp": dgp.__class__.__name__,
            "n": n, "M": M, "seed_base": seed_base, "h_ref": h_ref,
            "a_grid": a_grid.tolist(),
            "J_policy_true": J_policy_true.tolist(),
            "J_true": [dgp.m_true(float(a)) for a in a_grid],
        },
    }
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    if partial_path.exists():
        partial_path.unlink()
    print(f"  [DONE] {label} -> {save_path.name}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  Scalar block runner (Oracle / Naive / TwoStage)
# ══════════════════════════════════════════════════════════════════════════════

def _get_scalar_cls(method_label: str):
    """Return the estimator class for a scalar method label."""
    if method_label == "oracle":
        from simulations.methods.oracle import OracleDirect
        return OracleDirect
    elif method_label == "naive":
        from simulations.methods.naive import NaiveRegression
        return NaiveRegression
    elif method_label == "tsls":
        from simulations.methods.two_stage_linear import TwoStageLeastSquares
        return TwoStageLeastSquares
    else:
        raise ValueError(f"Unknown scalar method: {method_label!r}")


def run_scalar_block(
    dgp,
    method_label: str,
    n: int,
    M: int,
    seed_base: int,
    label: str,
    psi_0: Optional[float] = None,
) -> dict:
    """
    Run M scalar-estimator reps with block-level checkpoint.

    psi_0 : true scalar target (default: dgp.psi_true(a=1.0)).
             For DGP2 use dgp.psi_true(a=1.0) = m_true(1.0) — diagnostic only.
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    save_path = _RAW_DIR / f"{label}.pkl"

    if save_path.exists():
        print(f"  [SKIP] {label}")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    from simulations.analysis.metrics import compute_metrics
    from simulations.experiments.harness import run_monte_carlo

    method_cls = _get_scalar_cls(method_label)
    if psi_0 is None:
        psi_0 = float(dgp.psi_true(a=1.0))

    print(f"  [RUN] {label}")
    df = run_monte_carlo(dgp, method_cls, n, M, seed_base, save_path=None)
    metrics = compute_metrics(df["psi_hat"].values, df["V_hat"].values, psi_0, n)

    data = {
        "df": df, "metrics": metrics,
        "psi_0": psi_0, "label": label, "n": n, "M": M,
    }
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"  [DONE] {label} -> {save_path.name}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  Extended summariser
# ══════════════════════════════════════════════════════════════════════════════

def summarize_exp1A(data: dict) -> dict:
    """
    Compute aggregate metrics from a run_curve_block result.

    Primary target: J_policy_true = ∫ m(t) K_h(t-a) dt.
    Coverage is computed at ref_idx = K//2 (scalar summary) AND per dose
    using V_hat_grid[k] as per-dose variance.

    Returns a dict with all metrics required for the Exp1A summary table.
    """
    records  = data["records"]
    meta     = data["meta"]
    J_pt     = np.array(meta["J_policy_true"])  # (K,) truth
    K        = len(J_pt)
    n        = int(meta["n"])                   # sample size — needed for /n normalisation
    ok       = [r for r in records if r["error"] is None]
    M_ok     = len(ok)
    n_errors = len(records) - M_ok

    if M_ok == 0:
        return {"M_ok": 0, "n_errors": n_errors}

    J_dr_all = np.array([r["J_dr"]    for r in ok])  # (M_ok, K)
    psi_all  = np.array([r["psi_hat"] for r in ok])  # (M_ok,)
    V_all    = np.array([r["V_hat"]   for r in ok])  # (M_ok,) asymptotic variance (NOT /n)

    # -- Global metrics -------------------------------------------------------
    # V_hat convention (same as compute_metrics): V_hat = Σ score² / n  (O(1))
    # SE of J_dr = sqrt(V_hat / n)  — must divide by n.
    ref_idx   = K // 2
    bias_d    = np.mean(J_dr_all, axis=0) - J_pt          # (K,)
    rmse_d    = np.sqrt(np.mean((J_dr_all - J_pt[None]) ** 2, axis=0))
    mise      = float(np.mean(np.mean((J_dr_all - J_pt[None]) ** 2, axis=1)))
    psi_0_ref = float(J_pt[ref_idx])
    se_all    = np.sqrt(np.abs(V_all) / n)          # ← divide by n (was missing!)
    coverage  = float(np.mean(np.abs(psi_all - psi_0_ref) <= 1.96 * se_all))
    # SER convention = mean(se_hat) / std(psi_hat) — same as compute_metrics
    ser       = float(np.mean(se_all) / (np.std(psi_all) + 1e-12))

    # -- Per-dose coverage (uses V_hat_grid[k]) --------------------------------
    # V_hat_grid[k] = mean((score_i − J_dr[k])²) — same O(1) scale as V_hat.
    # Per-dose SE = sqrt(V_hat_grid[k] / n).
    V_hg_all = np.array([r["V_hat_grid"] for r in ok])   # (M_ok, K)
    SE_g_all = np.sqrt(np.abs(V_hg_all) / n)             # ← divide by n
    coverage_per_dose = np.mean(
        np.abs(J_dr_all - J_pt[None]) <= 1.96 * SE_g_all, axis=0
    ).tolist()   # list[K]

    # -- ESS ------------------------------------------------------------------
    ESS_min_arr   = np.array([r["ESS_min"]       for r in ok])
    ESS_ratio_arr = np.array([r["ESS_ratio_min"] for r in ok])
    ess_min_mean   = float(np.nanmean(ESS_min_arr))   if np.any(np.isfinite(ESS_min_arr))   else np.nan
    ess_ratio_mean = float(np.nanmean(ESS_ratio_arr)) if np.any(np.isfinite(ESS_ratio_arr)) else np.nan

    # -- Variance decomposition -----------------------------------------------
    V_reg_all  = np.array([r["V_reg_grid"]        for r in ok])
    V_corr_all = np.array([r["V_correction_grid"] for r in ok])

    # -- IPW weights ----------------------------------------------------------
    w_p99_all  = np.array([r["weight_p99_grid"] for r in ok])
    w_max_all  = np.array([r["weight_max_grid"] for r in ok])
    q_clip_arr = np.array([r["q_clip_fraction"] for r in ok])

    # -- Riesz diagnostics (NaN for oracle-q → nanmean is safe) ---------------
    rr_mean_all = np.array([r["riesz_residual_grid_mean"] for r in ok])
    cm_mean_all = np.array([r["cond_M_grid_mean"]         for r in ok])
    ns_mean_all = np.array([r["neg_share_grid_mean"]      for r in ok])

    return {
        "M_ok": M_ok, "n_errors": n_errors,
        # Global
        "bias_ref":  float(bias_d[ref_idx]),
        "rmse_ref":  float(rmse_d[ref_idx]),
        "max_err":   float(np.max(np.abs(bias_d))),
        "coverage":  coverage,
        "ser":       ser,
        "mise":      mise,
        "bias_per_dose": bias_d.tolist(),
        "rmse_per_dose": rmse_d.tolist(),
        # Per-dose coverage
        "coverage_per_dose":  coverage_per_dose,
        # ESS
        "ess_min_mean":       ess_min_mean,
        "ess_ratio_min_mean": ess_ratio_mean,
        # Variance decomposition (mean across reps, per dose)
        "V_hat_grid_mean":        _nanmean_2d(V_hg_all,   K),
        "V_reg_grid_mean":        _nanmean_2d(V_reg_all,  K),
        "V_correction_grid_mean": _nanmean_2d(V_corr_all, K),
        # IPW weights
        "weight_p99_grid_mean":   _nanmean_2d(w_p99_all, K),
        "weight_max_grid_mean":   _nanmean_2d(w_max_all, K),
        "q_clip_fraction_mean":   float(np.nanmean(q_clip_arr)) if np.any(np.isfinite(q_clip_arr)) else np.nan,
        # Riesz diagnostics (all-NaN for oracle-q — suppress empty-slice warning)
        "riesz_residual_mean": _nanmean_2d(rr_mean_all, K),
        "cond_M_mean":         _nanmean_2d(cm_mean_all, K),
        "neg_share_mean":      _nanmean_2d(ns_mean_all, K),
    }


def _summarize_block(data: dict, method_label: str, J_pt: Optional[np.ndarray] = None) -> dict:
    """Dispatch summarization based on method type."""
    if method_label in ("oracle", "naive", "tsls"):
        m = data["metrics"]
        return {
            "type": "scalar",
            "M_ok":     data["M"],
            "n_errors": 0,
            "bias_ref": m["bias"],
            "rmse_ref": m["rmse"],
            "coverage": m["coverage"],
            "ser":      m["ser"],
            "mise":     np.nan,
            "ess_min_mean":    np.nan,
            "max_err":         abs(m["bias"]),
            "bias_per_dose":   [m["bias"]],
            "coverage_per_dose": [m["coverage"]],
        }
    else:
        s = summarize_exp1A(data)
        s["type"] = "curve"
        return s


# ══════════════════════════════════════════════════════════════════════════════
#  GO/NO-GO logic
# ══════════════════════════════════════════════════════════════════════════════

def _block_go_nogo(summ: dict, method_label: str, snr: float) -> Optional[bool]:
    """Return True/False/None (None = no criterion defined for this combo)."""
    if summ.get("M_ok", 0) == 0:
        return False

    n_errors = summ.get("n_errors", 0)
    M_ok     = summ.get("M_ok", 1)
    if n_errors / (M_ok + n_errors) >= 0.05:
        return False

    coverage = summ.get("coverage")
    ess      = summ.get("ess_min_mean")

    if method_label == "drk_oracle":
        if snr >= 0.94:                   # ≈ 0.95
            return coverage is not None and coverage >= 0.92
        return None                       # diagnostic only for lower SNR

    if method_label == "drk_xfit":
        thresholds = {0.95: 0.90, 0.80: 0.82, 0.60: 0.65}
        thr = thresholds.get(snr)
        cov_ok  = (coverage is not None and coverage >= thr) if thr else True
        ess_ok  = (ess is not None and ess > 10) if ess is not None else True
        return cov_ok and ess_ok

    if method_label in ("dr_dose", "drk_oracle"):
        ess_ok = (ess is not None and ess > 10) if ess is not None else True
        return ess_ok

    # Oracle / Naive / TwoStage — no formal criterion
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Summary markdown writer
# ══════════════════════════════════════════════════════════════════════════════

def write_exp1A_summary(
    all_summaries: dict,
    n_list: List[int],
    snr_list: List[float],
    M: int,
    output_label: str = "exp1A_summary",
) -> None:
    """Write {output_label}.md to _SUMM_DIR."""
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    out = _SUMM_DIR / f"{output_label}.md"

    lines: List[str] = []
    lines += [
        "# Exp1A — Coverage Map Contrôlée",
        f"# Généré : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"# M = {M}, n ∈ {n_list}, SNR ∈ {snr_list}",
        "",
    ]

    # ── GO/NO-GO overview table ───────────────────────────────────────────────
    lines.append("## GO/NO-GO global")
    lines.append("")

    def _go_str(summ, method_label, snr):
        r = _block_go_nogo(summ, method_label, snr)
        if r is True:  return "✅"
        if r is False: return "❌"
        return "—"

    for dgp_name, a_grid, _, method_labels in _DGP_CONFIGS:
        lines.append(f"### {dgp_name.upper()}")
        header = "| SNR | n | " + " | ".join(method_labels) + " |"
        sep    = "|-----|---|" + "|".join(["---"] * len(method_labels)) + "|"
        lines += [header, sep]
        for snr in snr_list:
            for n in n_list:
                cells = []
                for m in method_labels:
                    k = (dgp_name, snr, n, m)
                    s = all_summaries.get(k, {})
                    cells.append(_go_str(s, m, snr))
                lines.append(f"| {snr} | {n} | " + " | ".join(cells) + " |")
        lines.append("")

    # ── Per-DGP tables ────────────────────────────────────────────────────────
    for dgp_name, a_grid, _, method_labels in _DGP_CONFIGS:
        lines.append(f"## {dgp_name.upper()} — {'Cobb-Douglas' if dgp_name == 'dgp1' else 'Michaelis-Menten'}")
        lines.append(f"a_grid = {a_grid.tolist()}")
        if dgp_name == "dgp2":
            lines.append("*Note: TwoStage excluded (DGP2 non-linear).*")
            lines.append("*Note: J_true = m(a) ≠ J_policy_true for concave m — Jensen gap applies.*")
        lines.append("")

        for snr in snr_list:
            lines.append(f"### SNR = {snr}")
            header = (
                "| Méthode | n | M_ok | n_err | bias | RMSE | coverage | "
                "SER | MISE | ESS_min | max_err |"
            )
            sep = "|---------|---|------|-------|------|------|----------|-----|------|---------|---------|"
            lines += [header, sep]
            for method_label in method_labels:
                for n in n_list:
                    k = (dgp_name, snr, n, method_label)
                    s = all_summaries.get(k)
                    if s is None:
                        lines.append(f"| {method_label} | {n} | — | — | — | — | — | — | — | — | — |")
                        continue
                    if s.get("M_ok", 0) == 0:
                        lines.append(
                            f"| {method_label} | {n} | 0 | {s.get('n_errors','?')} "
                            "| — | — | — | — | — | — | — |"
                        )
                        continue
                    lines.append(
                        f"| {method_label} | {n} | {s['M_ok']} | {s['n_errors']} | "
                        f"{_fmt(s.get('bias_ref'), '+.4f')} | "
                        f"{_fmt(s.get('rmse_ref'), '.4f')} | "
                        f"{_fmt(s.get('coverage'), '.3f')} | "
                        f"{_fmt(s.get('ser'), '.3f')} | "
                        f"{_fmt(s.get('mise'), '.4f')} | "
                        f"{_fmt(s.get('ess_min_mean'), '.1f')} | "
                        f"{_fmt(s.get('max_err'), '.4f')} |"
                    )
            lines.append("")

        # Per-dose coverage tables (DRKernel xfit only, best SNR)
        best_snr = max(snr_list)
        lines.append(f"### Coverage par dose — DRKernel xfit (SNR={best_snr})")
        dose_header = "| Dose (a) | " + " | ".join(f"n={n}" for n in n_list) + " |"
        dose_sep    = "|----------|" + "|".join(["---"] * len(n_list)) + "|"
        lines += [dose_header, dose_sep]
        for d, a in enumerate(a_grid):
            row = [f"{a:.2f}"]
            for n in n_list:
                k = (dgp_name, best_snr, n, "drk_xfit")
                s = all_summaries.get(k, {})
                cpd = s.get("coverage_per_dose", [])
                row.append(f"{cpd[d]:.3f}" if d < len(cpd) else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # ── IPW diagnostics table ─────────────────────────────────────────────────
    best_snr = max(snr_list)
    lines += [
        f"## Diagnostics IPW — DRKernel xfit (SNR={best_snr})",
        "| DGP | n | weight_p99_max | q_clip_fraction | riesz_residual_max | ESS_min_mean |",
        "|-----|---|----------------|-----------------|-------------------|--------------|",
    ]
    for dgp_name, _, _, _ in _DGP_CONFIGS:
        for n in n_list:
            k = (dgp_name, best_snr, n, "drk_xfit")
            s = all_summaries.get(k, {})
            if s.get("M_ok", 0) == 0:
                lines.append(f"| {dgp_name} | {n} | — | — | — | — |")
                continue
            w_p99 = s.get("weight_p99_grid_mean", [np.nan])
            rr    = s.get("riesz_residual_mean",  [np.nan])
            lines.append(
                f"| {dgp_name} | {n} | "
                f"{_fmt(max(w_p99), '.2f')} | "
                f"{_fmt(s.get('q_clip_fraction_mean'), '.4f')} | "
                f"{_fmt(max(rr), '.4f')} | "
                f"{_fmt(s.get('ess_min_mean'), '.1f')} |"
            )
    lines.append("")

    # ── GO/NO-GO checklist ────────────────────────────────────────────────────
    n_ref   = max(n_list)
    snr_ref = max(snr_list) if snr_list else 0.95

    def _crit(dgp_name, method_label, snr, n, key, threshold, op=">="):
        k = (dgp_name, snr, n, method_label)
        s = all_summaries.get(k, {})
        val = s.get(key)
        if val is None:
            return "—", "—"
        if op == ">=":
            ok = val >= threshold
        else:
            ok = val < threshold
        icon = "✅" if ok else "❌"
        return f"{icon} {_fmt(val, '.3f')}", val

    lines += [
        "## Critères GO/NO-GO formels",
        "",
        f"| Critère | Valeur cible | DGP1 n={n_ref} SNR={snr_ref} | DGP2 n={n_ref} SNR={snr_ref} |",
        "|---------|-------------|----------------------|----------------------|",
    ]

    row1_dgp1, _ = _crit("dgp1", "drk_xfit",    snr_ref, n_ref, "coverage",    0.90)
    row1_dgp2, _ = _crit("dgp2", "drk_xfit",    snr_ref, n_ref, "coverage",    0.90)
    row2_dgp1, _ = _crit("dgp1", "drk_xfit",    snr_ref, n_ref, "ess_min_mean",  10, op=">=")
    row2_dgp2, _ = _crit("dgp2", "drk_xfit",    snr_ref, n_ref, "ess_min_mean",  10, op=">=")
    row3_dgp1, _ = _crit("dgp1", "drk_oracle",  snr_ref, n_ref, "coverage",    0.92)
    row3_dgp2, _ = _crit("dgp2", "drk_oracle",  snr_ref, n_ref, "coverage",    0.92)
    row4_dgp1, _ = _crit("dgp1", "drk_xfit",    snr_ref, n_ref, "max_err",     0.35, op="<")
    row4_dgp2, _ = _crit("dgp2", "drk_xfit",    snr_ref, n_ref, "max_err",     0.35, op="<")

    lines += [
        f"| DRKernel xfit coverage ≥ 0.90 (SNR=0.95) | ≥ 0.90 | {row1_dgp1} | {row1_dgp2} |",
        f"| DRKernel xfit ESS_min_mean > 10           | > 10   | {row2_dgp1} | {row2_dgp2} |",
        f"| DRKernel oracle coverage ≥ 0.92 (SNR=0.95)| ≥ 0.92 | {row3_dgp1} | {row3_dgp2} |",
        f"| DRKernel xfit max_err < 0.35              | < 0.35 | {row4_dgp1} | {row4_dgp2} |",
        "",
    ]

    # Overall GO/NO-GO
    all_vals = [row1_dgp1, row1_dgp2, row2_dgp1, row2_dgp2,
                row3_dgp1, row3_dgp2, row4_dgp1, row4_dgp2]
    go = all("✅" in str(v) for v in all_vals if v != "—")
    lines.append(f"**Exp1A GO/NO-GO: {'✅ GO' if go else '❌ NO-GO'}**")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main orchestration
# ══════════════════════════════════════════════════════════════════════════════

def main(
    snr_list: Optional[List[float]] = None,
    n_list: Optional[List[int]] = None,
    M: int = M_MAIN,
    dgp_names: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    output_label: str = "exp1A_summary",
) -> None:
    """
    Run Exp1A coverage map.

    All parameters can be restricted to subsets for smoke runs / pilots.
    Already-completed pkl blocks are skipped automatically (checkpoint).
    output_label : stem of the summary markdown file (default exp1A_summary).
                   Use 'exp1A_pilot_summary' for pilot runs.
    """
    if snr_list is None:
        snr_list = SNR_GRID
    if n_list is None:
        n_list = N_LIST
    if dgp_names is None:
        dgp_names = [c[0] for c in _DGP_CONFIGS]

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries: Dict = {}

    for dgp_name, a_grid, is_linear, all_method_labels in _DGP_CONFIGS:
        if dgp_name not in dgp_names:
            continue
        method_labels = [m for m in all_method_labels
                         if methods is None or m in methods]

        for snr in snr_list:
            dgp = make_dgp(dgp_name, snr)

            for n in n_list:
                h_ref = _silverman_h(dgp.generate(n=n, seed=0).A)
                J_pt  = _compute_J_policy_true(dgp, a_grid, h_ref)

                print(f"\n[{dgp_name.upper()} SNR={snr} n={n}]")

                for method_label in method_labels:
                    seed_base = _block_seed(dgp_name, snr, n, method_label)
                    label     = _block_label(dgp_name, method_label, snr, n, M)

                    if method_label in ("oracle", "naive", "tsls"):
                        data = run_scalar_block(
                            dgp, method_label, n, M, seed_base, label,
                            psi_0=float(dgp.psi_true(a=1.0)),
                        )
                    else:
                        factory = _make_factory(dgp, a_grid, h_ref, is_linear, method_label)
                        data = run_curve_block(
                            dgp, factory, a_grid, J_pt,
                            n=n, M=M, seed_base=seed_base,
                            label=label, h_ref=h_ref,
                        )

                    summ = _summarize_block(data, method_label, J_pt)
                    all_summaries[(dgp_name, snr, n, method_label)] = summ

    write_exp1A_summary(
        all_summaries, n_list=n_list, snr_list=snr_list, M=M,
        output_label=output_label,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp1A — Coverage Map")
    parser.add_argument("--snr",   type=float, nargs="+",
                        default=None, help="SNR values (e.g. 0.95 0.80)")
    parser.add_argument("--n",     type=int,   nargs="+",
                        default=None, help="Sample sizes (e.g. 500 1000)")
    parser.add_argument("--M",     type=int,   default=M_MAIN,
                        help=f"Replications per block (default {M_MAIN})")
    parser.add_argument("--dgp",   type=str,   nargs="+",
                        default=None, help="DGP names: dgp1 dgp2")
    parser.add_argument("--method", type=str,  nargs="+",
                        default=None, help="Methods: oracle naive tsls dr_dose drk_oracle drk_xfit")
    parser.add_argument("--pilot", action="store_true",
                        help="Quick pilot: n≤1000, M=50")
    args = parser.parse_args()

    if args.pilot:
        main(
            n_list=args.n or [500, 1000],
            M=args.M if args.M != M_MAIN else 50,
            snr_list=args.snr or SNR_GRID,
            dgp_names=args.dgp,
            methods=args.method,
            output_label="exp1A_pilot_summary",
        )
    else:
        main(
            snr_list=args.snr,
            n_list=args.n,
            M=args.M,
            dgp_names=args.dgp,
            methods=args.method,
        )
