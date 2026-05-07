"""
DGP1 -- Functional identifiability demonstration (F-NEW-A v2 source).

Goal
----
Demonstrate that on DGP1 (where h_0 is analytic), two bridges with the SAME
L^2 distance to h_0 can give VERY DIFFERENT J(pi) errors, depending on
whether the perturbation is in the null-space of the policy or in the
informative direction.

Structure
---------
Section A (analytic): for two perturbation families
   - eps_W(W) = W^2 - E[W^2]            (null-space: function of W only)
   - eps_A(A) = A^2 - E[A^2]            (informative: function of A only)
   pre-normalised to ||eps_W||_L2 = ||eps_A_scaled||_L2,
   compute (bridge_err, J_err) for lambda in {0, 0.1, 0.3, 1.0, 3.0}.

   J_err is computed *analytically* on DGP1:
   - null-space: J_err = lambda * E[eps_W(W)] = 0 exactly
   - informative: J_err(a) = lambda * scale * (a^2 + h_KDE^2 - Var(A))
     => at a=0: |J_err| = lambda * scale * |h_KDE^2 - Var(A)|

Section B (empirical): for 6 methods (oracle, naive, linearREG, linearDR,
DRKPV, best_bennett) x M=30 seeds DGP1 n=1000,
   - refit method
   - extract h_hat (predictor function)
   - evaluate ||h_hat - h_0||_{L^2(P)} on a fixed test set N=10^5
   - get J_hat at a=0 from method's output
   - compute |J_err| = |J_hat - J_true|
   Save (method, seed, bridge_err, J_err) per (method, seed).

Output
------
PKL: simulations/results/raw/dgp1_identifiability/results.pkl
   {
     "synthetic": {
        "lambdas": [...],
        "bridge_err_grid": [...],            # same for both directions (normalised)
        "null_space_J_err": [0, 0, 0, 0, 0],  # exact analytic
        "informative_J_err": [0, ...],
        "ref_a_grid": [-0.5, 0, 0.5],
        "h_KDE": float,
        "Var_W": float, "Var_A": float,
     },
     "empirical": [
        {"method": "oracle", "seed": ..., "bridge_err": ..., "J_err": ..., "J_hat": ...},
        ...
     ],
     "meta": {"n": 1000, "M": 30, "N_test": 100000, "ref_dose": 0.0,
              "alpha": 0.3, "J_true_at_ref": ...}
   }
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Callable, List, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import scipy.linalg as sla

_BASE = Path(__file__).resolve().parents[2]
_OUT_DIR = _BASE / "simulations" / "results" / "raw" / "dgp1_identifiability"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_OUT_PATH = _OUT_DIR / "results.pkl"

# ── Settings ──────────────────────────────────────────────────────────────
N_TRAIN = 1000
N_TEST = 100_000  # for h_err evaluation -- big enough for accuracy
M_SEEDS = 30
SEED_BASE_TRAIN = 95_000_000
SEED_TEST = 99_999_999  # fixed test set seed
LAMBDAS = np.array([0.0, 0.1, 0.3, 1.0, 3.0])
REF_DOSE = 0.0  # a = 0 (centre)
A_GRID = np.array([REF_DOSE])  # single dose -- avoids overhead

METHODS = ["oracle", "naive", "linearREG", "linearDR", "DRKPV", "best_bennett"]


# ── DGP setup ─────────────────────────────────────────────────────────────

from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP

DGP = CobbDouglasLinearDGP()  # default params
ALPHA = DGP.alpha
MU_L = DGP.mu_L


def h_0(W: np.ndarray, A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Analytic h_0 of DGP1: h_0(W,A,X) = W + alpha*A + (1-alpha)*X."""
    return W + ALPHA * A + (1 - ALPHA) * X


def J_true_at(a: float) -> float:
    """Analytic J(pi_a) for DGP1 with centred Gaussian policy.
    J(pi_a; h_0) = E[W] + alpha * a + (1-alpha) * E[X] = 0 + alpha * a + (1-alpha)*mu_L
    (since E[W]=0 by construction; for centred Gaussian K_h, integral of t = a)."""
    return ALPHA * a + (1 - ALPHA) * MU_L


def silverman_h(A: np.ndarray) -> float:
    """Silverman's rule of thumb bandwidth."""
    return 1.06 * float(np.std(A)) * len(A) ** (-1.0 / 5.0)


# ── Section A: synthetic perturbations ────────────────────────────────────

def synthetic_trajectories(N_ref: int = 200_000, seed: int = 42) -> dict:
    """Compute analytic trajectories for null-space (eps_W) and informative
    (eps_A) directions, normalised to equal L^2 norm."""
    sample = DGP.generate(n=N_ref, seed=seed)
    W = sample.W
    A = sample.A
    n_train_proxy = N_TRAIN  # use Silverman with n_train for typical bandwidth
    h_KDE = 1.06 * float(np.std(A)) * n_train_proxy ** (-1.0 / 5.0)

    # eps_W(W) = W^2 - E[W^2]
    EW2 = float(np.mean(W**2))
    eps_W = W**2 - EW2
    norm_eps_W = float(np.sqrt(np.mean(eps_W**2)))  # L^2(P) norm

    # eps_A(A) = A^2 - E[A^2]
    EA2 = float(np.mean(A**2))
    eps_A_raw = A**2 - EA2
    norm_eps_A_raw = float(np.sqrt(np.mean(eps_A_raw**2)))

    # Rescale eps_A so its L^2 norm equals that of eps_W
    scale_A = norm_eps_W / norm_eps_A_raw
    # eps_A_scaled = scale_A * eps_A_raw
    norm_eps_A = norm_eps_W  # by construction

    # bridge_err(lambda) = lambda * norm_eps (same for both)
    bridge_err = LAMBDAS * norm_eps_W

    # null-space J_err: integrate eps_W against policy.
    # For h_lambda = h_0 + lambda*eps_W(W), J(pi_a; h_lambda) - J_true(a)
    #     = lambda * E[eps_W(W)] = 0 EXACTLY by construction (eps_W centred).
    null_space_J_err = np.zeros_like(LAMBDAS)

    # informative J_err: integrate scale_A * (t^2 - EA2) against policy K_h(t-a)
    # at a=REF_DOSE. For centred Gaussian K_h with bandwidth h_KDE:
    #   integral t^2 K_h(t - a) dt = a^2 + h_KDE^2
    # so J_err = lambda * scale_A * (a^2 + h_KDE^2 - EA2)
    delta_J_per_lambda = scale_A * (REF_DOSE**2 + h_KDE**2 - EA2)
    informative_J_err = LAMBDAS * abs(delta_J_per_lambda)

    return {
        "lambdas": LAMBDAS.tolist(),
        "bridge_err": bridge_err.tolist(),
        "null_space_J_err": null_space_J_err.tolist(),
        "informative_J_err": informative_J_err.tolist(),
        "h_KDE": h_KDE,
        "Var_W": float(np.var(W)),
        "Var_A": float(np.var(A)),
        "EW2": EW2, "EA2": EA2,
        "norm_eps_W": norm_eps_W, "norm_eps_A_raw": norm_eps_A_raw,
        "scale_A": scale_A,
        "delta_J_per_lambda": float(delta_J_per_lambda),
        "ref_dose": REF_DOSE,
        "N_ref": N_ref,
    }


# ── Section B: empirical refit ────────────────────────────────────────────

def fit_oracle(sample, dgp):
    """Oracle: predictor = h_0 analytic."""
    def predict(W, A, X):
        return h_0(W, A, X)
    return predict


def fit_naive(sample, dgp):
    """OLS of Y on (A, X) (drops W)."""
    n = len(sample.Y)
    Z = np.column_stack([np.ones(n), sample.A, sample.X])
    beta, _, _, _ = sla.lstsq(Z, sample.Y)
    def predict(W, A, X):
        return beta[0] + beta[1] * A + beta[2] * X
    return predict


def fit_linear(sample, dgp):
    """OLS of Y on (A, W, X). Same h for linearREG and linearDR."""
    n = len(sample.Y)
    Z = np.column_stack([np.ones(n), sample.W, sample.A, sample.X])
    beta, _, _, _ = sla.lstsq(Z, sample.Y)
    def predict(W, A, X):
        return beta[0] + beta[1] * W + beta[2] * A + beta[3] * X
    return predict


def fit_kpv(sample, dgp, h_KDE):
    """KPVBridgeH via Mastouri 2-stage. Used by KPVREG and DRKPV."""
    from simulations.methods.kpv_bridge import KPVBridgeH
    bridge = KPVBridgeH(lambda_1=3e-5, lambda_2=3e-5)
    bridge.fit(W=sample.W, A=sample.A, Z=sample.Z, Y=sample.Y)
    def predict(W, A, X):
        return bridge.predict(W, A)  # KPVBridgeH ignores X (it's exogenous)
    return predict


def fit_bennett(sample, dgp, h_KDE):
    """BennettIndepBridgeH (RFF). Used by best_bennett."""
    from simulations.methods.bennett_indep_h import BennettIndepBridgeH
    bridge = BennettIndepBridgeH(
        m_h=2000, m_c=500, ell_h=2.75, ell_c=2.0,
        lambda_h=1e-5, gamma_critic=1e-4,
        seed=42,
    )
    bridge.fit(W=sample.W, A=sample.A, Z=sample.Z, Y=sample.Y)
    def predict(W, A, X):
        return bridge.predict(W, A)
    return predict


def compute_J_for_predictor(predict_fn: Callable, dgp, h_KDE: float,
                              a: float, N_eval: int = 50_000,
                              seed: int = 12345) -> float:
    """Plug-in J(pi_a; h) by Monte Carlo on independent sample.
    J(pi_a) = E_{W,X}[ integral h(W, t, X) K_h(t-a) dt ]
    For Gaussian policy K_h centred at a with bandwidth h:
      E_t[h(W, t, X)] under K_h(t-a) = h(W, a + h*Z, X) averaged over Z~N(0,1).
    Approximation: take few quadrature points along t per sample.
    """
    sample = dgp.generate(n=N_eval, seed=seed)
    # Quadrature for policy: t = a + h_KDE * z, z ~ N(0,1)
    # Use 20 Gauss-Hermite nodes for accuracy.
    from numpy.polynomial.hermite_e import hermegauss
    nodes, weights = hermegauss(20)
    # weights are for integral exp(-x^2/2)/sqrt(2*pi), so just re-normalise
    weights = weights / np.sum(weights)
    # For each W_i, X_i: average h(W_i, a + h_KDE * z, X_i) over z
    J = 0.0
    for z, w in zip(nodes, weights):
        t = a + h_KDE * z
        h_vals = predict_fn(sample.W, np.full_like(sample.W, t), sample.X)
        J += w * float(np.mean(h_vals))
    return J


def compute_bridge_err(predict_fn: Callable, dgp,
                        N_test: int = N_TEST, seed: int = SEED_TEST) -> float:
    """||h_hat - h_0||_{L^2(P)} via large MC sample."""
    sample = dgp.generate(n=N_test, seed=seed)
    h_hat = predict_fn(sample.W, sample.A, sample.X)
    h_true = h_0(sample.W, sample.A, sample.X)
    return float(np.sqrt(np.mean((h_hat - h_true) ** 2)))


def empirical_one_seed(seed: int) -> List[dict]:
    """Refit all methods on a fresh DGP1 sample, extract h_hat, compute
    bridge_err and J_err at REF_DOSE."""
    sample = DGP.generate(n=N_TRAIN, seed=seed)
    h_KDE = silverman_h(sample.A)
    J_true = J_true_at(REF_DOSE)
    results = []

    fit_funcs = {
        "oracle":       lambda: fit_oracle(sample, DGP),
        "naive":        lambda: fit_naive(sample, DGP),
        "linearREG":    lambda: fit_linear(sample, DGP),
        "linearDR":     lambda: fit_linear(sample, DGP),  # same h
        "DRKPV":        lambda: fit_kpv(sample, DGP, h_KDE),
        "best_bennett": lambda: fit_bennett(sample, DGP, h_KDE),
    }
    for method, fit_fn in fit_funcs.items():
        try:
            t0 = time.time()
            predict = fit_fn()
            h_err = compute_bridge_err(predict, DGP)
            J_hat = compute_J_for_predictor(predict, DGP, h_KDE, REF_DOSE,
                                              N_eval=20_000, seed=seed + 7777)
            J_err = abs(J_hat - J_true)
            elapsed = time.time() - t0
            results.append({
                "method": method, "seed": seed,
                "bridge_err": h_err, "J_err": J_err,
                "J_hat": J_hat, "h_KDE": h_KDE,
                "elapsed_s": elapsed,
            })
        except Exception as e:
            print(f"    [WARN] seed={seed} method={method}: {type(e).__name__}: {e}")
            results.append({
                "method": method, "seed": seed,
                "bridge_err": np.nan, "J_err": np.nan, "J_hat": np.nan,
                "h_KDE": h_KDE, "elapsed_s": 0.0,
            })
    return results


def main():
    print("=" * 70)
    print("DGP1 Functional Identifiability -- Section A (synthetic) + B (empirical)")
    print("=" * 70)
    print(f"DGP1 params: alpha={ALPHA}, mu_L={MU_L}")
    print(f"REF_DOSE={REF_DOSE}, J_true(ref)={J_true_at(REF_DOSE):.4f}")

    # -- Section A: synthetic trajectories --
    t0 = time.time()
    print("\n[Section A] Computing analytic trajectories...")
    synth = synthetic_trajectories()
    print(f"  Var(W)={synth['Var_W']:.4f}, Var(A)={synth['Var_A']:.4f}")
    print(f"  h_KDE (Silverman, n={N_TRAIN})={synth['h_KDE']:.4f}")
    print(f"  ||eps_W||_L2={synth['norm_eps_W']:.4f}, "
          f"||eps_A_raw||_L2={synth['norm_eps_A_raw']:.4f}, "
          f"scale_A={synth['scale_A']:.4f}")
    print(f"  delta_J_per_lambda (informative direction at a=0): "
          f"{synth['delta_J_per_lambda']:.4f}")
    print(f"  Bridge errors: {synth['bridge_err']}")
    print(f"  Null-space J_err: {synth['null_space_J_err']}")
    print(f"  Informative J_err: {synth['informative_J_err']}")
    print(f"  [Section A done in {time.time()-t0:.1f}s]")

    # -- Section B: empirical refit --
    print(f"\n[Section B] Empirical refit -- {len(METHODS)} methods x {M_SEEDS} seeds...")
    t0 = time.time()
    empirical: List[dict] = []
    for i in range(M_SEEDS):
        seed = SEED_BASE_TRAIN + i
        seed_results = empirical_one_seed(seed)
        empirical.extend(seed_results)
        if (i + 1) % 5 == 0 or i == M_SEEDS - 1:
            print(f"  seed {i+1}/{M_SEEDS} done ({time.time()-t0:.1f}s elapsed)")
    print(f"  [Section B done in {time.time()-t0:.1f}s, {len(empirical)} records]")

    # Summary by method
    print("\n  Empirical summary (mean +/- std over seeds):")
    print(f"  {'method':<15} {'bridge_err':>12} {'J_err':>12} {'J_hat':>10}")
    for m in METHODS:
        sub = [r for r in empirical if r["method"] == m and np.isfinite(r["bridge_err"])]
        if not sub:
            continue
        bes = np.array([r["bridge_err"] for r in sub])
        jes = np.array([r["J_err"] for r in sub])
        jhs = np.array([r["J_hat"] for r in sub])
        print(f"  {m:<15} {bes.mean():>6.4f} ({bes.std():>5.4f})  "
              f"{jes.mean():>6.4f} ({jes.std():>5.4f})  {jhs.mean():>6.4f}")

    # Save
    out = {
        "synthetic": synth,
        "empirical": empirical,
        "meta": {
            "n_train": N_TRAIN, "n_test": N_TEST, "M_seeds": M_SEEDS,
            "ref_dose": REF_DOSE, "alpha": ALPHA,
            "J_true_at_ref": J_true_at(REF_DOSE),
            "lambdas": LAMBDAS.tolist(),
        }
    }
    with open(_OUT_PATH, "wb") as f:
        pickle.dump(out, f)
    print(f"\n  Saved -> {_OUT_PATH}")


if __name__ == "__main__":
    main()
