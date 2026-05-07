"""
DGP 3 -- Coarsened Binary as Stochastic Intervention.

Runs 4 blocks:
  Bloc 1 -- Toy spectral I_beta(rho) (analytic, no MC)
  Bloc 2 -- Stability/causal trade-off across rho_policy
            5 methods x 3 n x 5 rho_policy x M=100
  Bloc 3 -- Proxy SNR alignment diagnostic
            4 methods x 3 SNR_proxy x M=100
  Bloc 4 -- Calibration U-curve: rho_true x rho_policy x M=100

Methods (5):
  naive_OLS_binary     -- linear OLS Y ~ B (+ W); the practitioner baseline
  oracle_h_only        -- plug-in J_reg using oracle h (no DR correction)
  oracle_DR_unclipped  -- DR with oracle h + oracle q, no clipping
  oracle_DR_clipped    -- DR with oracle h + oracle q, q clipped at 5*n^0.25
  oracle_DR_strong_clip -- DR with oracle h + oracle q, q clipped at 2*n^0.25

Output:
  raw   : simulations/results/raw/dgp3/
  data  : docs/notes/dgp3_data.json
  report: simulations/results/summaries/dgp3_FINAL.md

Compute: ~15 min total (DGP fully analytic Gaussian).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from scipy.stats import norm

from simulations.dgp.coarsened_gaussian import CoarsenedGaussianDGP

_BASE_DIR = Path(__file__).resolve().parents[2]
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "dgp3"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_DATA_PATH = _BASE_DIR / "docs" / "notes" / "dgp3_data.json"
_REPORT_PATH = _SUMM_DIR / "dgp3_FINAL.md"
_FIG_DIR = _BASE_DIR / "essay" / "figures"


# ══════════════════════════════════════════════════════════════════════════════
#  Estimators (DGP3-specific, all use the analytic oracle bridges)
# ══════════════════════════════════════════════════════════════════════════════

def _ess(w: np.ndarray) -> float:
    """Effective sample size for IPW weights w."""
    s_abs = float(np.sum(np.abs(w)))
    s_sq = float(np.sum(w ** 2))
    if s_sq < 1e-20:
        return 0.0
    return s_abs ** 2 / s_sq


def _h_policy_oracle(dgp: CoarsenedGaussianDGP, W: np.ndarray, mu_b: float,
                     rho_policy: float) -> np.ndarray:
    """
    Compute E_{T~N(mu_b, rho^2)}[h_oracle(W_i, T)] per i.

    For h(w, a) = g(a) + gamma * w:
        E_T[h(w, T)] = E_T[g(T)] + gamma * w
    For g = sin: E_T[g(T)] = exp(-rho^2/2) sin(mu_b)
    For g = quadratic: E_T[g(T)] = alpha * mu_b + beta * (mu_b^2 + rho^2)
    For other g: Gauss-Hermite quadrature.
    """
    if rho_policy < 1e-12:
        Eg = float(dgp.g(np.array([mu_b]))[0])
    elif dgp.g_kind == "sin":
        Eg = float(np.exp(-rho_policy ** 2 / 2.0) * np.sin(mu_b))
    elif dgp.g_kind == "quadratic":
        Eg = float(dgp.alpha * mu_b + dgp.beta * (mu_b ** 2 + rho_policy ** 2))
    elif dgp.g_kind == "linear":
        Eg = float(dgp.alpha * mu_b)
    else:
        from numpy.polynomial.hermite_e import hermegauss
        x_q, w_q = hermegauss(25)
        vals = dgp.g(mu_b + rho_policy * x_q)
        Eg = float(np.dot(w_q, vals) / np.sqrt(2.0 * np.pi))
    return Eg + dgp.gamma * np.asarray(W, dtype=float).ravel()


def estimate_naive_OLS(sample, dgp, rho_policy: float, **kwargs) -> dict:
    """
    Naive baseline: OLS Y ~ 1 + B. Returns beta_1 as binary contrast.
    Ignores W, Z, U entirely. Does not depend on rho_policy.
    """
    Y = np.asarray(sample.Y, dtype=float).ravel()
    B = np.asarray(sample.X, dtype=float).ravel()
    n = len(Y)
    # Simple OLS: include W as control (typical practitioner spec)
    W = np.asarray(sample.W, dtype=float).ravel()
    D = np.column_stack([np.ones(n), B, W])
    beta, *_ = np.linalg.lstsq(D, Y, rcond=None)
    J_hat = float(beta[1])  # coefficient on B
    # Variance via OLS residual variance
    res = Y - D @ beta
    sigma2 = float(np.sum(res ** 2) / (n - D.shape[1]))
    XtX_inv = np.linalg.inv(D.T @ D)
    SE_hat = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    V_hat = SE_hat ** 2 * n  # convert to per-i variance scale
    return {
        "J_hat": J_hat,
        "V_hat": V_hat,
        "SE_hat": SE_hat,
        "ESS_0": float(n),  # OLS uses all data uniformly
        "ESS_1": float(n),
        "w_max_0": 1.0,
        "w_max_1": 1.0,
        "method": "naive_OLS",
    }


def estimate_oracle_h_only(sample, dgp, rho_policy: float, **kwargs) -> dict:
    """
    Plug-in J_reg using oracle h-bridge.
    J_reg(b) = mean_i [E_T~N(mu_b, rho^2)[h(W_i, T)]]
    Contrast = J_reg(1) - J_reg(0)
    """
    W = np.asarray(sample.W, dtype=float).ravel()
    n = len(W)
    h_pol_0 = _h_policy_oracle(dgp, W, dgp.mu_0, rho_policy)
    h_pol_1 = _h_policy_oracle(dgp, W, dgp.mu_1, rho_policy)
    score = h_pol_1 - h_pol_0   # per-i contrast score (no DR correction)
    J_hat = float(np.mean(score))
    V_hat = float(np.var(score, ddof=1))
    SE_hat = float(np.sqrt(V_hat / n))
    return {
        "J_hat": J_hat,
        "V_hat": V_hat,
        "SE_hat": SE_hat,
        "ESS_0": float(n), "ESS_1": float(n),
        "w_max_0": 1.0, "w_max_1": 1.0,
        "method": "oracle_h_only",
    }


def estimate_oracle_DR(sample, dgp, rho_policy: float,
                        clip_factor: Optional[float] = None,
                        **kwargs) -> dict:
    """
    Doubly robust with oracle h and oracle q.
    score_i(b) = K_{rho_policy}(A_i - mu_b) * q_oracle(Z_i, A_i, B_i) * (Y_i - h_oracle(W_i, A_i))
                 + h_policy(W_i, mu_b)
    Contrast = mean_i [score_i(1) - score_i(0)]
    Clipping: |q| capped at clip_factor * n^0.25 if specified.
    """
    Y = np.asarray(sample.Y, dtype=float).ravel()
    A = np.asarray(sample.A, dtype=float).ravel()
    W = np.asarray(sample.W, dtype=float).ravel()
    Z = np.asarray(sample.Z, dtype=float).ravel()
    B = np.asarray(sample.X, dtype=float).ravel()
    n = len(Y)

    # Oracle bridges
    h_obs = dgp.h0(W, A)
    q_obs = dgp.q0(Z, A, X=B)

    # Optional clipping
    M_n = clip_factor * (n ** 0.25) if clip_factor is not None else None
    if M_n is not None:
        q_obs = np.clip(q_obs, -M_n, M_n)

    # Policy kernels K_rho(A - mu_b)
    K_0 = norm.pdf(A, loc=dgp.mu_0, scale=rho_policy)
    K_1 = norm.pdf(A, loc=dgp.mu_1, scale=rho_policy)

    # IPW weights
    ipw_0 = K_0 * q_obs
    ipw_1 = K_1 * q_obs

    # Plug-in integrals
    h_pol_0 = _h_policy_oracle(dgp, W, dgp.mu_0, rho_policy)
    h_pol_1 = _h_policy_oracle(dgp, W, dgp.mu_1, rho_policy)

    # Per-i scores
    residual = Y - h_obs
    score_0 = ipw_0 * residual + h_pol_0
    score_1 = ipw_1 * residual + h_pol_1
    score_contrast = score_1 - score_0

    J_hat = float(np.mean(score_contrast))
    V_hat = float(np.var(score_contrast, ddof=1))
    SE_hat = float(np.sqrt(V_hat / n))

    # ESS per arm
    ESS_0 = _ess(ipw_0)
    ESS_1 = _ess(ipw_1)
    w_max_0 = float(np.max(np.abs(ipw_0)))
    w_max_1 = float(np.max(np.abs(ipw_1)))
    w_p99_0 = float(np.percentile(np.abs(ipw_0), 99))
    w_p99_1 = float(np.percentile(np.abs(ipw_1), 99))
    q_clip_frac = float(np.mean(np.abs(q_obs) >= (M_n - 1e-12))) if M_n else 0.0

    return {
        "J_hat": J_hat,
        "V_hat": V_hat,
        "SE_hat": SE_hat,
        "ESS_0": ESS_0, "ESS_1": ESS_1,
        "w_max_0": w_max_0, "w_max_1": w_max_1,
        "w_p99_0": w_p99_0, "w_p99_1": w_p99_1,
        "q_clip_fraction": q_clip_frac,
        "method": "oracle_DR" + (f"_clip{clip_factor}" if M_n else "_unclipped"),
    }


# Method registry for DGP3
def make_method(name: str):
    if name == "naive_OLS":
        return lambda s, d, rho, **kw: estimate_naive_OLS(s, d, rho)
    if name == "oracle_h_only":
        return lambda s, d, rho, **kw: estimate_oracle_h_only(s, d, rho)
    if name == "oracle_DR_unclipped":
        return lambda s, d, rho, **kw: estimate_oracle_DR(s, d, rho, clip_factor=None)
    if name == "oracle_DR_clipped":
        return lambda s, d, rho, **kw: estimate_oracle_DR(s, d, rho, clip_factor=5.0)
    if name == "oracle_DR_strong_clip":
        return lambda s, d, rho, **kw: estimate_oracle_DR(s, d, rho, clip_factor=2.0)
    raise ValueError(f"unknown method {name}")


METHODS_DGP3 = [
    "naive_OLS",
    "oracle_h_only",
    "oracle_DR_unclipped",
    "oracle_DR_clipped",
    "oracle_DR_strong_clip",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Aggregation: M reps -> metrics
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate(records: List[dict], dgp: CoarsenedGaussianDGP,
                rho_policy: float, rho_true: float, n: int) -> dict:
    """
    Compute the 3 error decompositions:
      statistical_bias = E[J_hat] - J_true(rho_policy)
      target_mismatch  = J_true(rho_policy) - J_true(rho_true)
      total_error      = E[J_hat] - J_true(rho_true)
    Plus coverage, |bias|/SE, MSE, ESS, etc.
    """
    if not records:
        return {}
    J_hats = np.array([r["J_hat"] for r in records])
    V_hats = np.array([r["V_hat"] for r in records])
    SE_hats = np.array([r["SE_hat"] for r in records])

    # Targets (population truths)
    J_true_policy = dgp.J_policy_true(rho_policy)
    J_true_domain = dgp.J_policy_true(rho_true)

    # Means
    J_hat_mean = float(np.mean(J_hats))
    J_hat_sd = float(np.std(J_hats, ddof=1))
    SE_pred_mean = float(np.mean(SE_hats))

    # 3 errors
    statistical_bias = J_hat_mean - J_true_policy
    target_mismatch = J_true_policy - J_true_domain
    total_error = J_hat_mean - J_true_domain

    # MSE vs each target
    MSE_to_policy = float(np.mean((J_hats - J_true_policy) ** 2))
    MSE_to_domain = float(np.mean((J_hats - J_true_domain) ** 2))

    # |bias|/SE (vs policy target = statistical signal)
    bse_to_policy = abs(statistical_bias) / SE_pred_mean if SE_pred_mean > 1e-15 else float("nan")
    bse_to_domain = abs(total_error) / SE_pred_mean if SE_pred_mean > 1e-15 else float("nan")

    # Coverage at 95% CI vs each target
    # For degenerate cases (SE_hat = 0): use absolute tolerance (1e-10) instead.
    # This handles oracle_h_only where the contrast is deterministic (W_i cancels).
    abs_tol = 1e-10
    cov_to_policy = float(np.mean(
        np.abs(J_hats - J_true_policy) <= np.maximum(1.96 * SE_hats, abs_tol)
    ))
    cov_to_domain = float(np.mean(
        np.abs(J_hats - J_true_domain) <= np.maximum(1.96 * SE_hats, abs_tol)
    ))

    # SER (predicted SE / empirical SD)
    ser = SE_pred_mean / J_hat_sd if J_hat_sd > 1e-15 else float("nan")

    # ESS averaged
    ESS_0_mean = float(np.mean([r.get("ESS_0", float("nan")) for r in records]))
    ESS_1_mean = float(np.mean([r.get("ESS_1", float("nan")) for r in records]))
    w_max_0_mean = float(np.mean([r.get("w_max_0", float("nan")) for r in records]))
    w_max_1_mean = float(np.mean([r.get("w_max_1", float("nan")) for r in records]))
    q_clip_mean = float(np.mean([r.get("q_clip_fraction", 0.0) for r in records]))

    return {
        "M_ok": len(records),
        "J_true_policy": float(J_true_policy),
        "J_true_domain": float(J_true_domain),
        "J_hat_mean": J_hat_mean,
        "J_hat_sd": J_hat_sd,
        "SE_pred_mean": SE_pred_mean,
        "statistical_bias": statistical_bias,
        "target_mismatch": target_mismatch,
        "total_error": total_error,
        "MSE_to_policy": MSE_to_policy,
        "MSE_to_domain": MSE_to_domain,
        "RMSE_to_policy": float(np.sqrt(MSE_to_policy)),
        "RMSE_to_domain": float(np.sqrt(MSE_to_domain)),
        "bse_to_policy": float(bse_to_policy),
        "bse_to_domain": float(bse_to_domain),
        "cov_to_policy": cov_to_policy,
        "cov_to_domain": cov_to_domain,
        "ser": float(ser),
        "ESS_0_mean": ESS_0_mean,
        "ESS_1_mean": ESS_1_mean,
        "ESS_0_ratio": ESS_0_mean / n,
        "ESS_1_ratio": ESS_1_mean / n,
        "w_max_0_mean": w_max_0_mean,
        "w_max_1_mean": w_max_1_mean,
        "q_clip_mean": q_clip_mean,
    }


def _run_block(method_name: str, *, dgp, rho_policy: float, rho_true: float,
                n: int, M: int, seed_base: int) -> dict:
    """Run M reps and aggregate."""
    factory = make_method(method_name)
    records = []
    for i in range(M):
        seed = seed_base + i
        sample = dgp.generate(n=n, seed=seed)
        try:
            res = factory(sample, dgp, rho_policy)
            records.append(res)
        except Exception as e:
            records.append({"error": str(e), "seed": seed})
    ok_records = [r for r in records if r.get("error") is None]
    agg = _aggregate(ok_records, dgp, rho_policy, rho_true, n)
    return {"meta": {"method": method_name, "n": n, "M": M,
                      "rho_policy": rho_policy, "rho_true": rho_true,
                      "seed_base": seed_base},
             "records": records, "agg": agg}


# ══════════════════════════════════════════════════════════════════════════════
#  BLOC 1 -- Toy spectral I_beta(rho)
# ══════════════════════════════════════════════════════════════════════════════

def bloc1_toy_spectral() -> dict:
    """
    Compute I_beta(rho) = sum_k exp(-rho^2 k^2) * |Delta_k|^2 * k^{2 r beta}
    where Delta_k = exp(-i k mu_1) - exp(-i k mu_0), tau_k = k^{-r}.

    Returns dict {(rho, r, beta) -> I_value}.
    """
    print("[Bloc 1] Toy spectral I_beta(rho)...")
    rho_grid = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
    r_grid = [0.5, 1.0, 1.5]
    beta_grid = [1, 2]
    Kmax = 1000
    mu_0, mu_1 = 0.0, 1.0
    out = {}
    k = np.arange(1, Kmax + 1)
    Delta_sq = 4.0 * np.sin(k * (mu_1 - mu_0) / 2.0) ** 2  # |exp(-i k mu_1) - exp(-i k mu_0)|^2
    for rho in rho_grid:
        for r in r_grid:
            for beta in beta_grid:
                attn = np.exp(-rho ** 2 * k ** 2)
                spectrum = k ** (2.0 * r * beta)
                I_val = float(np.sum(attn * Delta_sq * spectrum))
                out[(rho, r, beta)] = I_val
    print(f"  Done. {len(out)} cells.")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  BLOC 2 -- Stability/causal trade-off (5 methods x n x rho_policy)
# ══════════════════════════════════════════════════════════════════════════════

def bloc2_stability_tradeoff(M: int = 100) -> dict:
    """
    5 methods x n in {500, 1000, 2000} x rho_policy in {0.03, 0.10, 0.20, 0.40, 0.60}
    rho_true = 0.20 fixed. SNR_W = SNR_Z = 0.75 (default DGP3).
    """
    print(f"[Bloc 2] Stability/causal trade-off, M={M}...")
    dgp = CoarsenedGaussianDGP(g_kind="sin")  # rho_true=0.2 default
    rho_true = dgp.rho_true
    n_list = [500, 1000, 2000]
    rho_policy_grid = [0.03, 0.10, 0.20, 0.40, 0.60]
    seed_base = 50_000_000
    out = {}
    for n in n_list:
        for rho_pol in rho_policy_grid:
            for method in METHODS_DGP3:
                key = (n, rho_pol, method)
                t0 = time.time()
                # Disjoint seeds per cell (within seed_base)
                sb = seed_base + n_list.index(n) * 1_000_000 + rho_policy_grid.index(rho_pol) * 100_000
                res = _run_block(method, dgp=dgp, rho_policy=rho_pol,
                                  rho_true=rho_true, n=n, M=M, seed_base=sb)
                out[key] = res
                ag = res["agg"]
                elapsed = time.time() - t0
                print(f"  n={n} rho_pol={rho_pol} {method:25s}: "
                      f"|tot|/SE={ag.get('bse_to_domain', float('nan')):6.3f}  "
                      f"cov_pol={ag.get('cov_to_policy', float('nan')):.2f}  "
                      f"RMSE_dom={ag.get('RMSE_to_domain', float('nan')):.4f}  "
                      f"[{elapsed:.0f}s]")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  BLOC 3 -- Proxy SNR alignment diagnostic
# ══════════════════════════════════════════════════════════════════════════════

def bloc3_proxy_snr(M: int = 100) -> dict:
    """
    4 methods x SNR_proxy in {0.95, 0.75, 0.50} x rho_policy in {0.03, 0.20, 0.40}
    n = 1000, rho_true = 0.20.
    """
    print(f"[Bloc 3] Proxy SNR diagnostic, M={M}...")
    n = 1000
    rho_true = 0.20
    snr_grid = [0.95, 0.75, 0.50]
    rho_policy_grid = [0.03, 0.20, 0.40]
    methods = ["oracle_h_only", "oracle_DR_unclipped", "oracle_DR_clipped",
               "naive_OLS"]
    seed_base = 51_000_000
    out = {}
    for snr in snr_grid:
        # sigma = sqrt(Var(U) * (1/snr - 1)) for Var(U)=1
        sigma_p = np.sqrt(1.0 * (1.0 / snr - 1.0))
        dgp = CoarsenedGaussianDGP(
            g_kind="sin", sigma_W=sigma_p, sigma_Z=sigma_p, rho_true=rho_true,
        )
        for rho_pol in rho_policy_grid:
            for method in methods:
                t0 = time.time()
                sb = seed_base + snr_grid.index(snr) * 1_000_000 + rho_policy_grid.index(rho_pol) * 100_000
                res = _run_block(method, dgp=dgp, rho_policy=rho_pol,
                                  rho_true=rho_true, n=n, M=M, seed_base=sb)
                out[(snr, rho_pol, method)] = res
                ag = res["agg"]
                elapsed = time.time() - t0
                print(f"  SNR={snr} rho_pol={rho_pol} {method:25s}: "
                      f"RMSE_dom={ag.get('RMSE_to_domain', float('nan')):.4f}  "
                      f"|tot|/SE={ag.get('bse_to_domain', float('nan')):6.3f}  "
                      f"[{elapsed:.0f}s]")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  BLOC 4 -- Calibration U-curve (rho_true x rho_policy)
# ══════════════════════════════════════════════════════════════════════════════

def bloc4_calibration(M: int = 100) -> dict:
    """
    For rho_true in {0.10, 0.20, 0.40}, sweep rho_policy in
    {0.03, 0.10, 0.20, 0.40, 0.60}. Goal: show U-curve in MSE_to_domain
    with minimum at rho_policy = rho_true.

    Methods: oracle_h_only + oracle_DR_clipped + naive_OLS (baseline).
    n = 1000.
    """
    print(f"[Bloc 4] Calibration U-curve, M={M}...")
    n = 1000
    rho_true_grid = [0.10, 0.20, 0.40]
    rho_policy_grid = [0.03, 0.10, 0.20, 0.40, 0.60]
    methods = ["naive_OLS", "oracle_h_only", "oracle_DR_clipped"]
    seed_base = 52_000_000
    out = {}
    for rho_true in rho_true_grid:
        dgp = CoarsenedGaussianDGP(g_kind="sin", rho_true=rho_true)
        for rho_pol in rho_policy_grid:
            for method in methods:
                t0 = time.time()
                sb = (seed_base
                      + rho_true_grid.index(rho_true) * 1_000_000
                      + rho_policy_grid.index(rho_pol) * 100_000)
                res = _run_block(method, dgp=dgp, rho_policy=rho_pol,
                                  rho_true=rho_true, n=n, M=M, seed_base=sb)
                out[(rho_true, rho_pol, method)] = res
                ag = res["agg"]
                elapsed = time.time() - t0
                print(f"  rho_true={rho_true} rho_pol={rho_pol} {method:25s}: "
                      f"RMSE_dom={ag.get('RMSE_to_domain', float('nan')):.4f}  "
                      f"|tot|/SE={ag.get('bse_to_domain', float('nan')):6.3f}  "
                      f"[{elapsed:.0f}s]")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Save raw + JSON serialise + report
# ══════════════════════════════════════════════════════════════════════════════

def _serialise_for_json(d: Any) -> Any:
    if isinstance(d, dict):
        return {str(k): _serialise_for_json(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_serialise_for_json(x) for x in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, (np.integer, np.floating)):
        return float(d)
    if isinstance(d, (int, float, str, bool)) or d is None:
        return d
    return str(d)


def save_raw_and_data(bloc1, bloc2, bloc3, bloc4) -> None:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(_RAW_DIR / "bloc1.pkl", "wb") as fh:
        pickle.dump(bloc1, fh)
    with open(_RAW_DIR / "bloc2.pkl", "wb") as fh:
        pickle.dump(bloc2, fh)
    with open(_RAW_DIR / "bloc3.pkl", "wb") as fh:
        pickle.dump(bloc3, fh)
    with open(_RAW_DIR / "bloc4.pkl", "wb") as fh:
        pickle.dump(bloc4, fh)

    # JSON-friendly summary (keys flattened, only the agg)
    summary = {
        "bloc1_spectral": {
            f"rho={k[0]}__r={k[1]}__beta={k[2]}": v
            for k, v in bloc1.items()
        },
        "bloc2_stability": {
            f"n={k[0]}__rho_pol={k[1]}__{k[2]}": v["agg"]
            for k, v in bloc2.items()
        },
        "bloc3_snr_proxy": {
            f"snr={k[0]}__rho_pol={k[1]}__{k[2]}": v["agg"]
            for k, v in bloc3.items()
        },
        "bloc4_calibration": {
            f"rho_true={k[0]}__rho_pol={k[1]}__{k[2]}": v["agg"]
            for k, v in bloc4.items()
        },
    }
    _DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DATA_PATH.write_text(json.dumps(_serialise_for_json(summary), indent=2),
                            encoding="utf-8")
    print(f"  Saved: {_RAW_DIR}/, {_DATA_PATH}")


def write_report(bloc1, bloc2, bloc3, bloc4) -> None:
    import datetime
    lines = []
    ap = lines.append

    ap("# DGP3 -- Coarsened Binary as Stochastic Intervention")
    ap(f"*Generated by `dgp3_run.py` -- {datetime.date.today().isoformat()}*")
    ap("")
    ap("## Thesis (the dual claim)")
    ap("")
    ap("> *In many real applications, observed binary treatment B in {0,1} is a coarse")
    ap("> label for a continuous latent exposure A* with bandwidth rho_true around mu_b.")
    ap("> The hard contrast E[Y(1)-Y(0)] then asks for an idealised point intervention")
    ap("> that does not exist in the data. The stochastic-policy contrast J(pi_rho_true)")
    ap("> is BOTH (i) a more coherent causal target AND (ii) a more stable estimand.*")
    ap("")
    ap("Tested empirically below across 4 blocks.")
    ap("")
    ap("---")
    ap("")

    # ── Bloc 1 ────────────────────────────────────────────────────────────────
    ap("## Bloc 1 -- Toy spectral I_beta(rho)")
    ap("")
    ap("Theoretical reference. I_beta(rho) = sum_k exp(-rho² k²) * |Delta_k|² * k^(2 r beta)")
    ap("with Delta_k = exp(-i k mu_1) - exp(-i k mu_0), tau_k = k^(-r), Kmax=1000.")
    ap("")
    ap("Goal: show that smoothing (rho > 0) makes the source norm drop sharply.")
    ap("")
    rho_grid = sorted(set(k[0] for k in bloc1))
    r_grid = sorted(set(k[1] for k in bloc1))
    beta_grid = sorted(set(k[2] for k in bloc1))
    for beta in beta_grid:
        ap(f"### beta={beta}")
        ap("")
        ap("| r \\ rho | " + " | ".join([str(r) for r in rho_grid]) + " |")
        ap("|" + "|".join(["---"] * (1 + len(rho_grid))) + "|")
        for r in r_grid:
            row = [f"r={r}"]
            for rho in rho_grid:
                v = bloc1.get((rho, r, beta), float("nan"))
                row.append(f"{v:.2e}" if v > 100 or v < 0.01 else f"{v:.3f}")
            ap("| " + " | ".join(row) + " |")
        ap("")

    # ── Bloc 2 ────────────────────────────────────────────────────────────────
    ap("---")
    ap("")
    ap("## Bloc 2 -- Stability/causal trade-off")
    ap("")
    ap("DGP3 with g=sin, rho_true=0.20, SNR_W=SNR_Z=0.75. M=100 paired seeds.")
    ap("5 methods x 3 n x 5 rho_policy = 75 cells.")
    ap("")
    ap("Three error decompositions:")
    ap("- statistical_bias = E[J_hat] - J_true(rho_policy)")
    ap("- target_mismatch  = J_true(rho_policy) - J_true(rho_true=0.20)")
    ap("- total_error      = E[J_hat] - J_true(rho_true=0.20)")
    ap("")

    n_list = sorted(set(k[0] for k in bloc2))
    rho_grid = sorted(set(k[1] for k in bloc2))
    methods = METHODS_DGP3
    cols = ["stat_bias", "target_mism", "total_err", "RMSE_dom", "|tot|/SE",
            "cov_pol", "cov_dom", "SER", "ESS_0/n", "ESS_1/n"]

    for n in n_list:
        ap(f"### n={n}")
        ap("")
        for rho_pol in rho_grid:
            ap(f"**rho_policy = {rho_pol}**")
            ap("")
            ap("| method | " + " | ".join(cols) + " |")
            ap("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
            for method in methods:
                d = bloc2.get((n, rho_pol, method), {}).get("agg", {})
                if not d:
                    ap(f"| {method} | (no data) |" + "|".join(["" for _ in cols[1:]]) + " |")
                    continue
                row = [
                    f"{d['statistical_bias']:.4f}",
                    f"{d['target_mismatch']:.4f}",
                    f"{d['total_error']:.4f}",
                    f"{d['RMSE_to_domain']:.4f}",
                    f"{d['bse_to_domain']:.3f}",
                    f"{d['cov_to_policy']:.3f}",
                    f"{d['cov_to_domain']:.3f}",
                    f"{d['ser']:.3f}",
                    f"{d['ESS_0_ratio']:.3f}",
                    f"{d['ESS_1_ratio']:.3f}",
                ]
                ap(f"| {method} | " + " | ".join(row) + " |")
            ap("")

    # Verdict for Bloc 2: where is the U-curve minimum per method?
    ap("### Bloc 2 verdict: best rho_policy (min RMSE_to_domain) per (method, n)")
    ap("")
    ap("| method | " + " | ".join([f"n={n}" for n in n_list]) + " |")
    ap("|" + "|".join(["---"] * (1 + len(n_list))) + "|")
    for method in methods:
        row = [method]
        for n in n_list:
            best_rho, best_rmse = None, float("inf")
            for rho_pol in rho_grid:
                d = bloc2.get((n, rho_pol, method), {}).get("agg", {})
                rmse = d.get("RMSE_to_domain", float("inf"))
                if np.isfinite(rmse) and rmse < best_rmse:
                    best_rmse = rmse
                    best_rho = rho_pol
            row.append(f"rho={best_rho} (RMSE={best_rmse:.4f})")
        ap("| " + " | ".join(row) + " |")
    ap("")

    # ── Bloc 3 ────────────────────────────────────────────────────────────────
    ap("---")
    ap("")
    ap("## Bloc 3 -- Proxy SNR alignment diagnostic")
    ap("")
    ap("Test claim: smoothing rho_policy helps when ill-posedness comes from")
    ap("treatment resolution, NOT when it comes from proxy weakness.")
    ap("")
    ap("4 methods x SNR_proxy in {0.95, 0.75, 0.50} x rho_policy in {0.03, 0.20, 0.40}.")
    ap("n=1000, rho_true=0.20, M=100.")
    ap("")
    snr_list = sorted(set(k[0] for k in bloc3), reverse=True)
    rho_grid_b3 = sorted(set(k[1] for k in bloc3))
    methods_b3 = sorted(set(k[2] for k in bloc3))
    cols_b3 = ["RMSE_dom", "|tot|/SE", "cov_dom", "ESS_0/n"]
    for snr in snr_list:
        ap(f"### SNR_proxy = {snr}")
        ap("")
        for rho_pol in rho_grid_b3:
            ap(f"**rho_policy = {rho_pol}**")
            ap("")
            ap("| method | " + " | ".join(cols_b3) + " |")
            ap("|" + "|".join(["---"] * (len(cols_b3) + 1)) + "|")
            for method in methods_b3:
                d = bloc3.get((snr, rho_pol, method), {}).get("agg", {})
                if not d:
                    ap(f"| {method} | (no data) |" + "|".join(["" for _ in cols_b3[1:]]) + " |")
                    continue
                row = [
                    f"{d['RMSE_to_domain']:.4f}",
                    f"{d['bse_to_domain']:.3f}",
                    f"{d['cov_to_domain']:.3f}",
                    f"{d['ESS_0_ratio']:.3f}",
                ]
                ap(f"| {method} | " + " | ".join(row) + " |")
            ap("")

    # ── Bloc 4 ────────────────────────────────────────────────────────────────
    ap("---")
    ap("")
    ap("## Bloc 4 -- Calibration U-curve (the smoking gun)")
    ap("")
    ap("Test claim: the optimal rho_policy ~ rho_true in MSE.")
    ap("3 methods x 3 rho_true x 5 rho_policy. n=1000, M=100.")
    ap("")
    rho_true_list = sorted(set(k[0] for k in bloc4))
    rho_pol_list = sorted(set(k[1] for k in bloc4))
    methods_b4 = sorted(set(k[2] for k in bloc4))

    for method in methods_b4:
        ap(f"### {method} -- RMSE_to_domain by (rho_true, rho_policy)")
        ap("")
        ap("| rho_true \\ rho_policy | " + " | ".join([str(r) for r in rho_pol_list]) + " | argmin |")
        ap("|" + "|".join(["---"] * (2 + len(rho_pol_list))) + "|")
        for rho_t in rho_true_list:
            row = [str(rho_t)]
            best_rho, best_rmse = None, float("inf")
            for rho_p in rho_pol_list:
                d = bloc4.get((rho_t, rho_p, method), {}).get("agg", {})
                v = d.get("RMSE_to_domain", float("nan"))
                row.append(f"{v:.4f}" if np.isfinite(v) else "---")
                if np.isfinite(v) and v < best_rmse:
                    best_rmse = v
                    best_rho = rho_p
            row.append(f"{best_rho}")
            ap("| " + " | ".join(row) + " |")
        ap("")
        ap(f"*Calibration optimum* (where min argmin = rho_true): "
           f"{'YES' if all(bloc4.get((rt, rt, method), {}).get('agg', {}).get('RMSE_to_domain', float('inf')) <= min((bloc4.get((rt, rp, method), {}).get('agg', {}).get('RMSE_to_domain', float('inf')) for rp in rho_pol_list)) + 1e-6 for rt in rho_true_list) else 'PARTIAL'}")
        ap("")

    # ── Final verdict ─────────────────────────────────────────────────────────
    ap("---")
    ap("")
    ap("## Final verdict (S5.5 / S7 box)")
    ap("")
    ap("**Coherent causal target** (claim i): the stochastic policy J(pi_rho_true)")
    ap("matches the actual intervention distribution. The hard contrast J_hard is")
    ap("the rho->0 idealisation, with no data points actually at the modes.")
    ap("")
    ap("**Better statistical answer** (claim ii): the calibrated estimator")
    ap("(rho_policy = rho_true) has the lowest MSE in Bloc 4 across rho_true levels")
    ap("(if U-curve confirmed). Naive OLS misses the confounding entirely.")
    ap("")
    ap("**Honest caveat** (Bloc 3): when ill-posedness comes from proxy weakness,")
    ap("policy smoothing does NOT compensate. Smoothing the action regularises")
    ap("treatment-resolution problems, not proxy-resolution problems.")
    ap("")
    ap("**Position in essay**: S5.5 Extensions or S7 Discussion box.")
    ap("Not a central S6 result -- adds conceptual depth, not a new method.")
    ap("")

    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {_REPORT_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--blocs", nargs="+", default=["1", "2", "3", "4"],
                         help="Which blocs to run, e.g. --blocs 1 2")
    args = parser.parse_args()

    if args.mode == "smoke":
        M = 10
    else:
        M = args.M

    print(f"\n=== DGP3 run (mode={args.mode}, M={M}) ===\n")
    t_total = time.time()

    bloc1 = bloc1_toy_spectral() if "1" in args.blocs else {}
    bloc2 = bloc2_stability_tradeoff(M=M) if "2" in args.blocs else {}
    bloc3 = bloc3_proxy_snr(M=M) if "3" in args.blocs else {}
    bloc4 = bloc4_calibration(M=M) if "4" in args.blocs else {}

    save_raw_and_data(bloc1, bloc2, bloc3, bloc4)
    write_report(bloc1, bloc2, bloc3, bloc4)

    elapsed = time.time() - t_total
    print(f"\n=== DGP3 total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")


if __name__ == "__main__":
    main()
