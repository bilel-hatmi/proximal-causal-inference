"""
DGP2 Kallus-style Minimax Witness Diagnostic.

Context
-------
Existing diagnostics show DRKernel on DGP2 (Michaelis-Menten) undercovers:
    rho = 0.246 < 0.5  (root-n violated)
    correction efficiency xfit-q ~ 1%, oracle-q ~ 25%
    jensen_gap_ref ~ 0.003 << |bias_dr_ref| ~ 0.08

The remaining open question : in *which functional direction* is h_hat wrong, and
does q_hat correct that direction? Kallus (2021) Eqs. 12-13 turn this into a
minimax test where an adversarial RKHS critic finds the worst violation direction.

Three witnesses
---------------
1. h-bridge witness (with noise floor):
       witness_h_hat   = (1/n^2) r_hat^T K_ZA r_hat   with r_hat = Y - h_hat
       witness_h_ref   = (1/n^2) r_ref^T K_ZA r_ref   with r_ref = Y - h_ref
       witness_h_excess = witness_h_hat - witness_h_ref          (noise-floored)

2. q-bridge RKHS witness (Kallus Eq. 13 closed form):
       lhs_vec_j = (1/n) sum_i K_h(A_i-a_ref) q_hat(Z_i,A_i) k_W(W_i,W_j) k_A(A_i,A_j)
       rhs_vec_j = (1/n) sum_i k_W(W_i,W_j) * ktilde(A_j, a_ref, h)   [Gauss-Gauss]
       witness_q_rel = ||lhs - rhs|| / ||rhs||
   Sanity column : riesz_g1 = mean(K_h * q_hat) - 1.

3. Direction witness (Correction 2 sign : Y - h_hat in DR score => h_ref - h_hat):
       plug_in_error       = mean(T_pi h_hat - T_pi h_ref)
       correction_on_error = mean(K_h(A-a) * q_hat * (h_ref - h_hat))
       correction_noise    = mean(K_h(A-a) * q_hat * (Y - h_ref))
       direction_ratio     = correction_on_error / (-plug_in_error)
   Cross-check : direction_ratio_h0 computed against dgp.h0() (KRR proxy oracle).

Construction of h_ref (Correction 1)
------------------------------------
dgp.h0() is the regression oracle E[mu(U,A)|W,A] (KRR), NOT the Fredholm bridge --
attenuated by snr_W (cf. michaelis_menten.py:208-213). We build h_ref via
KPVBridgeH at n_ref=10000 with Nystrom n_landmarks=500 -- 5x larger than the
largest production sample, with ~2x rate improvement.

Modes
-----
quick : n in {500, 1000}, M=5, oracle-q + xfit-q   (~10-15 min)
main  : n in {500, 1000, 2000}, M=20               (~3-5 h)

Usage
-----
    python -m simulations.experiments.dgp2_kallus_witness --mode quick
    python -m simulations.experiments.dgp2_kallus_witness --mode main
    python -m simulations.experiments.dgp2_kallus_witness --mode main --refit-href

Output
------
    simulations/results/raw/dgp2_kallus_witness/_cache/<href cache pkl>
    simulations/results/raw/dgp2_kallus_witness/kallus_witness_<method>_n<N>_M<M>.pkl
    simulations/results/summaries/dgp2_kallus_witness.md
"""
from __future__ import annotations

import argparse
import copy
import datetime
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from sklearn.model_selection import KFold

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE_DIR  = Path(__file__).resolve().parents[2]
_RAW_DIR   = _BASE_DIR / "simulations" / "results" / "raw" / "dgp2_kallus_witness"
_SUMM_DIR  = _BASE_DIR / "simulations" / "results" / "summaries"
_CACHE_DIR = _RAW_DIR / "_cache"
_HREF_CACHE_PATH = _CACHE_DIR / "dgp2_href_n10000_lm500_seed2026.pkl"

# ── Constants ────────────────────────────────────────────────────────────────
A_GRID         = np.array([1.8, 2.2, 2.6, 3.0])
K              = len(A_GRID)
REF_IDX        = 2          # a_ref = 2.6
SNR            = 0.95
N_FOLDS        = 5
RANDOM_STATE   = 42
CHECKPOINT_FREQ = 1         # per-rep (each rep is expensive)
SEED_BASE      = 8_000_000  # disjoint from production seeds (5e6, 6e6, 7e6)

# h_ref hyperparameters (Correction 1)
HREF_N         = 10000
HREF_LANDMARKS = 500
HREF_LAMBDA    = None       # None -> Ying scaling 1e-2 * n^(-0.7) ~ 1.6e-5
HREF_SEED      = 2026

# Quadrature for T_pi h0_proxy (regression oracle, no closed form)
N_QUAD         = 25


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v: Any, fmt: str = ".4f", sci_thresh: float = 1e5) -> str:
    """Format numeric value with NaN/inf handling."""
    try:
        if v is None:
            return "---"
        f = float(v)
        if not np.isfinite(f):
            return "---" if np.isnan(f) else "inf"
        return f"{f:.2e}" if abs(f) >= sci_thresh else f"{f:{fmt}}"
    except Exception:
        return str(v)


def _silverman_h(A: np.ndarray) -> float:
    """Silverman bandwidth (matches DRKernel and dgp2_bias_diagnostics)."""
    return 1.06 * float(np.std(A)) * len(A) ** (-0.2)


def _kernel(u: np.ndarray, h: float) -> np.ndarray:
    """K_h(u) = phi(u/h)/h, normalised Gaussian density (matches dr_kernel.py:173-176)."""
    return np.exp(-0.5 * (u / h) ** 2) / (h * np.sqrt(2.0 * np.pi))


# ══════════════════════════════════════════════════════════════════════════════
#  h_ref construction (Correction 1)
# ══════════════════════════════════════════════════════════════════════════════

def build_h_ref(
    dgp,
    n_ref: int = HREF_N,
    n_landmarks: int = HREF_LANDMARKS,
    lambda_: Optional[float] = HREF_LAMBDA,
    seed: int = HREF_SEED,
    cache_path: Optional[Path] = None,
    force: bool = False,
    verbose: bool = True,
):
    """
    Fit a high-precision KPVBridgeH on a held-out sample of size n_ref.

    Hyperparameters tuned for proxy of the Fredholm bridge:
    - n_ref=10000 : ~5x the largest production fold
    - n_landmarks=500 : Nystrom path O(n*m^2) = 2.5e9 vs full O(n^3) = 1e12
    - lambda=Ying default (None -> 1e-2 * n^(-0.7))
    - seed=2026 : disjoint from production seeds (5e6+, 6e6+, 7e6+, 8e6+)

    Returns the fitted KPVBridgeH; caches to disk for reuse.

    Also prints Fredholm sanity diagnostics:
        residual_W = mean((Y - h_ref(W,A)) * W)
        residual_Z = mean((Y - h_ref(W,A)) * Z)   -- key diagnostic, should be ~0
    """
    from simulations.methods.kpv_bridge import KPVBridgeH

    # Resolve cache_path lazily to honour monkeypatched module-level default
    if cache_path is None:
        cache_path = _HREF_CACHE_PATH

    if cache_path.exists() and not force:
        if verbose:
            print(f"  [HREF cache hit] {cache_path.name}")
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        return cached["h_ref"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  [HREF] Fitting KPVBridgeH n={n_ref}, landmarks={n_landmarks}, "
              f"lambda={lambda_}, seed={seed} ...")
    sample = dgp.generate(n=n_ref, seed=seed)

    bridge = KPVBridgeH(
        lambda_1=lambda_, lambda_2=lambda_, n_landmarks=n_landmarks,
    ).fit(sample.W, sample.A, sample.Z, sample.Y)

    # Fredholm sanity diagnostics
    h_pred = bridge.predict(sample.W, sample.A)
    resid  = sample.Y - h_pred
    res_W  = float(np.mean(resid * sample.W))
    res_Z  = float(np.mean(resid * sample.Z))
    res_one = float(np.mean(resid))
    res_norm_rel = float(np.linalg.norm(resid) / (np.linalg.norm(sample.Y) + 1e-15))
    if verbose:
        print(f"    h_ref Fredholm sanity: <r,1>={res_one:+.5f}  "
              f"<r,W>={res_W:+.5f}  <r,Z>={res_Z:+.5f}  "
              f"||r||/||Y||={res_norm_rel:.4f}")
    meta = dict(
        n_ref=n_ref, n_landmarks=n_landmarks, lambda_=lambda_, seed=seed,
        ell_W=bridge.ell_W, ell_A=bridge.ell_A, ell_Z=bridge.ell_Z,
        lambda_1=bridge.lambda_1, lambda_2=bridge.lambda_2,
        residual_one=res_one, residual_W=res_W, residual_Z=res_Z,
        residual_norm_rel=res_norm_rel,
    )
    with open(cache_path, "wb") as fh:
        pickle.dump({"h_ref": bridge, "meta": meta}, fh)
    if verbose:
        print(f"  [HREF saved] {cache_path}")
    return bridge


def _href_meta() -> dict | None:
    """Read cached h_ref metadata for reporting."""
    if not _HREF_CACHE_PATH.exists():
        return None
    with open(_HREF_CACHE_PATH, "rb") as fh:
        cached = pickle.load(fh)
    return cached.get("meta")


# ══════════════════════════════════════════════════════════════════════════════
#  Per-fold loop with arrays exposed (replicates dr_kernel.py:248-305)
# ══════════════════════════════════════════════════════════════════════════════

def _per_fold_arrays(
    sample,
    dgp,
    h_ref,
    q_kind: str,
    a_grid: np.ndarray,
    h_KDE: float,
    bridge_kwargs: dict | None = None,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
    q_factory: Optional[object] = None,
) -> dict:
    """
    Replicate DRKernel's 5-fold loop, exposing per-fold predictions on test sets.

    Returns dict with keys:
        h_hat_obs    (n,)       -- ĥ^(-k)(W_i, A_i)
        h_hat_pol    (n, K)     -- T_pi ĥ^(-k) at each dose
        q_obs_2d     (n, K)     -- q̂_a^(-k)(Z_i, A_i)  [oracle: tiled]
        h_ref_obs    (n,)       -- h_ref(W_i, A_i)         [external, no leakage]
        h_ref_pol    (n, K)     -- T_pi h_ref at each dose [Gauss-Gauss closed form]
        h0_proxy_obs (n,)       -- dgp.h0(W_i, A_i)        [KRR sanity]
        h0_proxy_pol (n, K)     -- T_pi h0_proxy via Gauss-Hermite quadrature

    q_kind:
        "oracle" : OracleBridgeQ(dgp).predict (analytic q0)
        "xfit"   : copy q_factory per fold, fit on train, predict_all on test
    """
    from simulations.methods.kpv_bridge import KPVBridgeH
    from simulations.methods._oracle_bridges import OracleBridgeQ

    n = len(sample.Y)
    Y, A, W, Z = sample.Y, sample.A, sample.W, sample.Z
    K_doses = len(a_grid)

    h_obs = np.full(n, np.nan)
    h_pol = np.full((n, K_doses), np.nan)
    q_obs_2d = np.full((n, K_doses), np.nan)

    # Oracle q is dose-independent (universal Cui q0 evaluated pointwise)
    if q_kind == "oracle":
        q_oracle = OracleBridgeQ(dgp)
        q_1d = q_oracle.predict(Z, A, np.zeros(n))   # (n,)
        q_obs_2d = np.tile(q_1d[:, None], (1, K_doses))
    elif q_kind == "xfit":
        if q_factory is None:
            raise ValueError("q_factory required for q_kind='xfit'")
    else:
        raise ValueError(f"Unknown q_kind: {q_kind}")

    bkw = bridge_kwargs or {}
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for tr, te in kf.split(np.arange(n)):
        bridge_k = KPVBridgeH(**bkw).fit(W[tr], A[tr], Z[tr], Y[tr])
        h_obs[te] = bridge_k.predict(W[te], A[te])
        for d, a_d in enumerate(a_grid):
            h_pol[te, d] = bridge_k.plug_in_policy(W[te], float(a_d), h_KDE)

        if q_kind == "xfit":
            q_k = copy.deepcopy(q_factory)
            q_k.fit(W[tr], A[tr], Z[tr])
            q_obs_2d[te, :] = q_k.predict_all(Z[te], A[te])

    if np.any(np.isnan(h_obs)) or np.any(np.isnan(h_pol)):
        raise RuntimeError("h cross-fitting left NaN entries (fold coverage issue)")
    if q_kind == "xfit" and np.any(np.isnan(q_obs_2d)):
        raise RuntimeError("xfit q cross-fitting left NaN entries")

    # h_ref evaluated on FULL sample (trained on disjoint n_ref=10000 -- no leakage)
    h_ref_obs = h_ref.predict(W, A)
    h_ref_pol = np.column_stack([
        h_ref.plug_in_policy(W, float(a_d), h_KDE) for a_d in a_grid
    ])

    # h0_proxy_KRR via Gauss-Hermite quadrature (h0 not in RKHS)
    h0_obs = dgp.h0(W, A)
    h0_pol = np.zeros((n, K_doses))
    x_q, w_q = hermegauss(N_QUAD)
    for d, a_d in enumerate(a_grid):
        col = np.zeros(n)
        for j in range(N_QUAD):
            t = float(a_d) + h_KDE * x_q[j]
            col += w_q[j] * dgp.h0(W, np.full(n, t))
        h0_pol[:, d] = col / np.sqrt(2.0 * np.pi)

    return dict(
        h_hat_obs=h_obs, h_hat_pol=h_pol, q_obs_2d=q_obs_2d,
        h_ref_obs=h_ref_obs, h_ref_pol=h_ref_pol,
        h0_proxy_obs=h0_obs, h0_proxy_pol=h0_pol,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Witness functions
# ══════════════════════════════════════════════════════════════════════════════

def _compute_witness_h(
    Y: np.ndarray,
    h_hat_obs: np.ndarray,
    h_ref_obs: np.ndarray,
    h0_proxy_obs: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray,
    ell_Z: float,
    ell_A: float,
) -> dict:
    """
    Témoin 1 — h-bridge RKHS witness with noise floor (Correction 3).

    K_ZA = K_Z * K_A (Hadamard, RBF). Compute r^T K_ZA r / n^2 as r @ (K_ZA @ r) / n^2
    to keep only one (n,n) matrix in memory.
    """
    from simulations.methods.kpv_bridge import _rbf_gram

    n = len(Y)
    K_Z = _rbf_gram(Z, Z, ell_Z)
    K_A = _rbf_gram(A, A, ell_A)
    K_ZA = K_Z * K_A
    del K_Z, K_A

    r_hat = Y - h_hat_obs
    r_ref = Y - h_ref_obs
    r_h0  = Y - h0_proxy_obs

    # quadratic forms
    w_hat = float(r_hat @ (K_ZA @ r_hat)) / (n ** 2)
    w_ref = float(r_ref @ (K_ZA @ r_ref)) / (n ** 2)
    w_h0  = float(r_h0  @ (K_ZA @ r_h0))  / (n ** 2)

    return dict(
        witness_h_hat=w_hat,
        witness_h_ref=w_ref,
        witness_h_excess=w_hat - w_ref,
        witness_h_h0=w_h0,
    )


def _compute_witness_q(
    W: np.ndarray,
    A: np.ndarray,
    Z: np.ndarray,
    q_at_ref: np.ndarray,        # (n,) clipped q̂_a at a_ref
    h_KDE: float,
    a_ref: float,
    ell_W: float,
    ell_A: float,
) -> dict:
    """
    Témoin 2 — q-bridge RKHS witness via Gauss-Gauss closed form (Kallus Eq. 13).

    For each j:
        lhs_vec_j = (1/n) sum_i K_h(A_i-a_ref) q_at_ref_i k_W(W_i,W_j) k_A(A_i,A_j)
        rhs_vec_j = (1/n) (sum_i k_W(W_i,W_j)) * ktilde(A_j, a_ref, h_KDE)

    where ktilde(s, a, h) = (ell_A/sqrt(ell_A^2+h^2)) * exp(-(s-a)^2/(2(ell_A^2+h^2))).

    Returns:
        witness_q_norm     = sqrt(mean((lhs - rhs)^2))
        witness_q_rel      = witness_q_norm / sqrt(mean(rhs^2))
        riesz_g1_at_ref    = mean(K_h(A-a_ref) * q_at_ref) - 1   [sanity]
    """
    from simulations.methods.kpv_bridge import _rbf_gram, _rbf_kde_convolution

    n = len(W)
    K_h_a_ref = _kernel(A - a_ref, h_KDE)            # (n,)

    # K_W and K_A (n,n) — needed for lhs only; we factor smartly
    K_W = _rbf_gram(W, W, ell_W)                     # (n,n)
    K_A = _rbf_gram(A, A, ell_A)                     # (n,n)

    weights_lhs = K_h_a_ref * q_at_ref               # (n,) effective weight per i
    # lhs_vec_j = (1/n) sum_i weights_lhs[i] * K_W[i,j] * K_A[i,j]
    lhs_vec = (K_W * K_A).T @ weights_lhs / n        # (n,)
    del K_A  # free

    # rhs : closed-form Gauss-Gauss for the inner integral, then mean over W kernel
    ktilde = _rbf_kde_convolution(A, a_ref, h_KDE, ell_A)   # (n,) at each A_j
    mean_K_W = K_W.mean(axis=0)                              # (n,) (1/n) sum_i K_W[i,j]
    rhs_vec = mean_K_W * ktilde                              # (n,)
    del K_W

    diff = lhs_vec - rhs_vec
    witness_q_norm = float(np.sqrt(np.mean(diff ** 2)))
    rhs_rms = float(np.sqrt(np.mean(rhs_vec ** 2)))
    witness_q_rel = witness_q_norm / (rhs_rms + 1e-15)

    riesz_g1_at_ref = float(np.mean(K_h_a_ref * q_at_ref) - 1.0)

    return dict(
        witness_q_norm=witness_q_norm,
        witness_q_rel=witness_q_rel,
        riesz_g1_at_ref=riesz_g1_at_ref,
    )


def _compute_direction_witness(
    Y: np.ndarray,
    A: np.ndarray,
    h_hat_obs: np.ndarray,
    h_ref_obs: np.ndarray,
    h0_proxy_obs: np.ndarray,
    h_hat_pol: np.ndarray,
    h_ref_pol: np.ndarray,
    h0_proxy_pol: np.ndarray,
    q_obs_2d: np.ndarray,
    h_KDE: float,
    a_grid: np.ndarray,
    ref_idx: int,
) -> dict:
    """
    Témoin 3 — Direction résiduelle (Corrections 2 + 4).

    Sign convention from DR score (dr_kernel.py:384):
        correction = K_h(A-a) * q * (Y - h_hat)
    => the "right direction" of the h-error is (h_ref - h_hat).

        plug_in_error       = mean(T_pi h_hat - T_pi h_ref)
        correction_on_error = mean(K_h * q * (h_ref - h_hat))     <-- Correction 4
        correction_noise    = mean(K_h * q * (Y - h_ref))          <-- Correction 4
        direction_ratio     = correction_on_error / (-plug_in_error)

    Cross-check using h0_proxy_KRR :
        plug_in_error_h0       = mean(T_pi h_hat - T_pi h0_proxy)
        correction_on_error_h0 = mean(K_h * q * (h0_proxy - h_hat))
        direction_ratio_h0     = correction_on_error_h0 / (-plug_in_error_h0)

    Computed at a_ref AND for full grid (returned as _grid arrays).
    """
    K_doses = len(a_grid)
    pe_grid       = np.zeros(K_doses)
    pe_h0_grid    = np.zeros(K_doses)
    coe_grid      = np.zeros(K_doses)
    coe_h0_grid   = np.zeros(K_doses)
    cnoise_grid   = np.zeros(K_doses)
    dr_grid       = np.zeros(K_doses)
    dr_h0_grid    = np.zeros(K_doses)

    diff_h_ref = h_ref_obs    - h_hat_obs    # (n,) Correction 2 sign
    diff_h_h0  = h0_proxy_obs - h_hat_obs    # (n,)
    noise_term = Y - h_ref_obs               # (n,)

    for d, a_d in enumerate(a_grid):
        K_h_a = _kernel(A - float(a_d), h_KDE)        # (n,)
        q_at  = q_obs_2d[:, d]                         # (n,)

        # Plug-in error : T_pi h_hat - T_pi h_ref
        pe   = float(np.mean(h_hat_pol[:, d] - h_ref_pol[:, d]))
        peh0 = float(np.mean(h_hat_pol[:, d] - h0_proxy_pol[:, d]))

        coe  = float(np.mean(K_h_a * q_at * diff_h_ref))
        coeh0 = float(np.mean(K_h_a * q_at * diff_h_h0))
        cn   = float(np.mean(K_h_a * q_at * noise_term))

        ratio    = coe   / (-pe)   if abs(pe)   > 1e-12 else np.nan
        ratio_h0 = coeh0 / (-peh0) if abs(peh0) > 1e-12 else np.nan

        pe_grid[d]     = pe
        pe_h0_grid[d]  = peh0
        coe_grid[d]    = coe
        coe_h0_grid[d] = coeh0
        cnoise_grid[d] = cn
        dr_grid[d]     = ratio
        dr_h0_grid[d]  = ratio_h0

    return dict(
        plug_in_error          = float(pe_grid[ref_idx]),
        plug_in_error_h0       = float(pe_h0_grid[ref_idx]),
        correction_on_error    = float(coe_grid[ref_idx]),
        correction_on_error_h0 = float(coe_h0_grid[ref_idx]),
        correction_noise       = float(cnoise_grid[ref_idx]),
        direction_ratio        = float(dr_grid[ref_idx]),
        direction_ratio_h0     = float(dr_h0_grid[ref_idx]),
        plug_in_error_grid       = pe_grid.tolist(),
        correction_on_error_grid = coe_grid.tolist(),
        direction_ratio_grid     = dr_grid.tolist(),
        direction_ratio_h0_grid  = dr_h0_grid.tolist(),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  One-rep witness builder
# ══════════════════════════════════════════════════════════════════════════════

def _witness_one_rep(
    sample,
    dgp,
    h_ref,
    q_kind: str,
    a_grid: np.ndarray,
    h_KDE: float,
    J_pt: np.ndarray,
    bridge_kwargs: dict | None = None,
    q_factory: Optional[object] = None,
    seed: int = 0,
) -> dict:
    """
    Compute all three witnesses for one Monte-Carlo replication.

    Returns a record dict with witness columns + production diagnostics.
    """
    from simulations.methods.kpv_bridge import _median_bandwidth

    n = len(sample.Y)
    Y, A, W, Z = sample.Y, sample.A, sample.W, sample.Z
    a_ref = float(a_grid[REF_IDX])

    arr = _per_fold_arrays(
        sample, dgp, h_ref, q_kind, a_grid, h_KDE,
        bridge_kwargs=bridge_kwargs, q_factory=q_factory,
    )

    # Bandwidths for the witness kernels (median heuristic on rep sample)
    ell_W = _median_bandwidth(W)
    ell_A = _median_bandwidth(A)
    ell_Z = _median_bandwidth(Z)

    w1 = _compute_witness_h(Y, arr["h_hat_obs"], arr["h_ref_obs"],
                            arr["h0_proxy_obs"], Z, A, ell_Z, ell_A)
    w2 = _compute_witness_q(W, A, Z, arr["q_obs_2d"][:, REF_IDX],
                            h_KDE, a_ref, ell_W, ell_A)
    w3 = _compute_direction_witness(Y, A,
                                    arr["h_hat_obs"], arr["h_ref_obs"],
                                    arr["h0_proxy_obs"],
                                    arr["h_hat_pol"], arr["h_ref_pol"],
                                    arr["h0_proxy_pol"],
                                    arr["q_obs_2d"], h_KDE, a_grid, REF_IDX)

    # Production diagnostics — compatible with _decompose in dgp2_bias_diagnostics
    K_doses = len(a_grid)
    J_reg_grid = np.array([float(np.mean(arr["h_hat_pol"][:, d])) for d in range(K_doses)])
    J_dr_grid  = np.zeros(K_doses)
    V_hat_ref  = 0.0
    ess_min    = np.inf
    for d in range(K_doses):
        K_h_a = _kernel(A - float(a_grid[d]), h_KDE)
        ipw = K_h_a * arr["q_obs_2d"][:, d]
        scores = ipw * (Y - arr["h_hat_obs"]) + arr["h_hat_pol"][:, d]
        J_dr_grid[d] = float(np.mean(scores))
        if d == REF_IDX:
            V_hat_ref = float(np.var(scores, ddof=0))
        denom = float(np.sum(ipw ** 2))
        ess_d = (float(np.sum(ipw)) ** 2) / denom if denom > 0 else 0.0
        ess_min = min(ess_min, ess_d)

    bias_reg = float(J_reg_grid[REF_IDX] - J_pt[REF_IDX])
    bias_dr  = float(J_dr_grid[REF_IDX]  - J_pt[REF_IDX])
    SE_ref   = float(np.sqrt(abs(V_hat_ref) / n))

    return dict(
        seed=seed, error=None, n=n, method=q_kind, h_KDE=h_KDE,
        # Témoin 1
        witness_h_hat    = w1["witness_h_hat"],
        witness_h_ref    = w1["witness_h_ref"],
        witness_h_excess = w1["witness_h_excess"],
        witness_h_h0     = w1["witness_h_h0"],
        # Témoin 2
        witness_q_norm   = w2["witness_q_norm"],
        witness_q_rel    = w2["witness_q_rel"],
        riesz_g1_at_ref  = w2["riesz_g1_at_ref"],
        # Témoin 3
        plug_in_error          = w3["plug_in_error"],
        plug_in_error_h0       = w3["plug_in_error_h0"],
        correction_on_error    = w3["correction_on_error"],
        correction_on_error_h0 = w3["correction_on_error_h0"],
        correction_noise       = w3["correction_noise"],
        direction_ratio        = w3["direction_ratio"],
        direction_ratio_h0     = w3["direction_ratio_h0"],
        plug_in_error_grid       = w3["plug_in_error_grid"],
        correction_on_error_grid = w3["correction_on_error_grid"],
        direction_ratio_grid     = w3["direction_ratio_grid"],
        direction_ratio_h0_grid  = w3["direction_ratio_h0_grid"],
        # Production diagnostics
        bias_reg = bias_reg, bias_dr = bias_dr,
        SE_ref   = SE_ref, ESS = float(ess_min),
        J_pt_ref = float(J_pt[REF_IDX]),
        # Witness bandwidths
        ell_W=ell_W, ell_A=ell_A, ell_Z=ell_Z,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Block runner with checkpointing
# ══════════════════════════════════════════════════════════════════════════════

def _run_witness_block(
    dgp,
    h_ref,
    q_kind: str,
    n: int,
    M: int,
    seed_base: int,
    label: str,
    a_grid: np.ndarray,
    bridge_kwargs: dict | None = None,
    q_factory_proto: Optional[object] = None,
    raw_dir: Path = _RAW_DIR,
) -> dict:
    """
    Run M reps of witness diagnostic with per-rep checkpointing (CHECKPOINT_FREQ=1).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_path    = raw_dir / f"{label}.pkl"
    partial_path = raw_dir / f"{label}.partial.pkl"

    if save_path.exists():
        print(f"  [SKIP] {label}")
        with open(save_path, "rb") as fh:
            return pickle.load(fh)

    if partial_path.exists():
        with open(partial_path, "rb") as fh:
            records = pickle.load(fh)
        start_i = len(records)
        print(f"  [RESUME] {label} from rep {start_i}/{M}")
    else:
        records = []
        start_i = 0

    # Compute J_pt once per (n, dgp) outside the rep loop
    h_KDE_ref = _silverman_h(dgp.generate(n=n, seed=seed_base + 99999).A)
    J_pt = np.zeros(len(a_grid))
    x_q, w_q = hermegauss(N_QUAD)
    for d, a_d in enumerate(a_grid):
        t_vals = float(a_d) + h_KDE_ref * x_q
        vals = np.array([float(dgp.m_true(float(t))) for t in t_vals])
        J_pt[d] = float(np.dot(w_q, vals) / np.sqrt(2.0 * np.pi))

    for i in range(start_i, M):
        seed = seed_base + i
        sample = dgp.generate(n=n, seed=seed)
        h_KDE = _silverman_h(sample.A)
        try:
            rec = _witness_one_rep(
                sample, dgp, h_ref, q_kind, a_grid, h_KDE, J_pt,
                bridge_kwargs=bridge_kwargs,
                q_factory=q_factory_proto, seed=seed,
            )
            records.append(rec)
        except Exception as exc:
            warnings.warn(f"[{label} seed={seed}] {exc}", RuntimeWarning)
            records.append({"error": str(exc), "seed": seed})

        if (i + 1) % CHECKPOINT_FREQ == 0:
            with open(partial_path, "wb") as fh:
                pickle.dump(records, fh)
            print(f"  [{label}] rep {i+1}/{M} done")

    data = {
        "records": records,
        "meta": {
            "label": label, "n": n, "M": M,
            "q_kind": q_kind, "a_grid": a_grid.tolist(),
            "J_policy_true": J_pt.tolist(), "seed_base": seed_base,
        },
    }
    with open(save_path, "wb") as fh:
        pickle.dump(data, fh)
    if partial_path.exists():
        partial_path.unlink()
    print(f"  [DONE] {label}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  Decomposition (mean + SE per column over reps)
# ══════════════════════════════════════════════════════════════════════════════

def _decompose_witness(data: dict) -> dict | None:
    """Aggregate per-rep records into mean/SE for each witness column."""
    records = [r for r in data["records"] if r.get("error") is None]
    if not records:
        return None
    n = int(data["meta"]["n"])
    M_ok = len(records)

    cols = [
        "witness_h_hat", "witness_h_ref", "witness_h_excess", "witness_h_h0",
        "witness_q_norm", "witness_q_rel", "riesz_g1_at_ref",
        "plug_in_error", "plug_in_error_h0", "correction_on_error",
        "correction_on_error_h0", "correction_noise",
        "direction_ratio", "direction_ratio_h0",
        "bias_reg", "bias_dr", "SE_ref", "ESS",
    ]
    out: dict[str, Any] = dict(n=n, M_ok=M_ok)
    for c in cols:
        vals = np.array([float(r[c]) for r in records if np.isfinite(r.get(c, np.nan))])
        if len(vals) == 0:
            out[c + "_mean"] = np.nan
            out[c + "_se"]   = np.nan
        else:
            out[c + "_mean"] = float(np.mean(vals))
            out[c + "_se"]   = float(np.std(vals, ddof=1) / np.sqrt(len(vals))
                                     if len(vals) > 1 else 0.0)
    # Per-dose grids (mean only)
    for grid_key in ["plug_in_error_grid", "correction_on_error_grid",
                      "direction_ratio_grid", "direction_ratio_h0_grid"]:
        arr = np.array([r[grid_key] for r in records])     # (M_ok, K)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[grid_key + "_mean"] = np.nanmean(arr, axis=0).tolist()
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Report builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_report_witness(
    decomps: dict,            # (method, n) -> decomp dict
    href_meta: dict | None,
    n_grid: list[int],
    methods: list[str],
    J_true_ref: float,
) -> list[str]:
    lines: list[str] = []
    add = lines.append

    add("# DGP2 Kallus Witness Diagnostic")
    add(f"# Genere : {datetime.date.today()}")
    add("# SNR=0.95, Michaelis-Menten, ref dose a=2.6 (REF_IDX=2)")
    add("# 3 temoins : h-RKHS (with noise floor), q-RKHS (Kallus Eq. 13), direction (Cor. 2+4)")
    add("")

    # ── Section 0 : h_ref metadata ──────────────────────────────────────────
    add("## 0. h_ref haute precision (Correction 1)")
    add("")
    if href_meta is None:
        add("(h_ref cache absent — `--mode quick` will rebuild)")
    else:
        # Sanity : flag if cache parameters don't match production values
        if (href_meta.get("n_ref") != HREF_N
            or href_meta.get("n_landmarks") != HREF_LANDMARKS
            or href_meta.get("seed") != HREF_SEED):
            add(f"⚠ ⚠ ⚠ CACHE MISMATCH : expected n_ref={HREF_N}, lm={HREF_LANDMARKS}, "
                f"seed={HREF_SEED} ⚠ ⚠ ⚠")
            add(f"⚠   Got n_ref={href_meta.get('n_ref')}, "
                f"lm={href_meta.get('n_landmarks')}, seed={href_meta.get('seed')}")
            add(f"⚠   Re-run with `--refit-href` to rebuild a clean cache.")
            add("")
        add(f"- KPVBridgeH n_ref={href_meta['n_ref']}, "
            f"n_landmarks={href_meta['n_landmarks']}, seed={href_meta['seed']}")
        add(f"- bandwidths : ell_W={href_meta['ell_W']:.4f}  "
            f"ell_A={href_meta['ell_A']:.4f}  ell_Z={href_meta['ell_Z']:.4f}")
        add(f"- regularisation : lambda_1={href_meta['lambda_1']:.2e}  "
            f"lambda_2={href_meta['lambda_2']:.2e}")
        add("- Fredholm sanity (residuals on training sample) :")
        add(f"    <r,1> = {href_meta['residual_one']:+.5f}     "
            f"(should be ~0 if bridge satisfies E[Y-h(W,A)]=0)")
        add(f"    <r,W> = {href_meta['residual_W']:+.5f}     "
            f"(weak Fredholm test on W)")
        add(f"    <r,Z> = {href_meta['residual_Z']:+.5f}     "
            f"(★ key Fredholm test : E[Y-h(W,A)|Z,A]=0)")
        add(f"    ||r||/||Y|| = {href_meta['residual_norm_rel']:.4f}")
        if abs(href_meta['residual_Z']) > 1e-2:
            add("    ⚠ |<r,Z>| > 1e-2 : h_ref viole encore la condition Fredholm.")
            add("       Reduire lambda ou augmenter n_ref.")
    add("")

    # ── Section 1 : Main table ─────────────────────────────────────────────
    add("## 1. Table principale — temoins par (methode, n)")
    add("")
    add(f"J_true(a_ref={A_GRID[REF_IDX]:.2f}) = {J_true_ref:.5f}")
    add("")
    hdr = (
        "| method | n | wit_h_hat | wit_h_ref | wit_h_excess | wit_q_rel | riesz_g1 | "
        "plug_err | corr_on_err | corr_noise | dir_ratio | dir_ratio_h0 | bias_reg | bias_dr | ESS |"
    )
    sep = "|--------|---|----------|----------|-------------|----------|---------|---------|-----------|----------|----------|------------|---------|---------|-----|"
    add(hdr); add(sep)
    for method in methods:
        for n in n_grid:
            d = decomps.get((method, n))
            if d is None:
                add(f"| {method} | {n} | (missing) | | | | | | | | | | | | |")
                continue
            add(
                f"| {method} | {n} | "
                f"{_fmt(d['witness_h_hat_mean'], '.4f')} | "
                f"{_fmt(d['witness_h_ref_mean'], '.4f')} | "
                f"{_fmt(d['witness_h_excess_mean'], '+.4f')} | "
                f"{_fmt(d['witness_q_rel_mean'], '.4f')} | "
                f"{_fmt(d['riesz_g1_at_ref_mean'], '+.4f')} | "
                f"{_fmt(d['plug_in_error_mean'], '+.4f')} | "
                f"{_fmt(d['correction_on_error_mean'], '+.4f')} | "
                f"{_fmt(d['correction_noise_mean'], '+.4f')} | "
                f"{_fmt(d['direction_ratio_mean'], '+.3f')} | "
                f"{_fmt(d['direction_ratio_h0_mean'], '+.3f')} | "
                f"{_fmt(d['bias_reg_mean'], '+.4f')} | "
                f"{_fmt(d['bias_dr_mean'], '+.4f')} | "
                f"{_fmt(d['ESS_mean'], '.0f')} |"
            )
    add("")
    add("**Lecture des colonnes** :")
    add("- `wit_h_excess` = wit_h_hat - wit_h_ref : croissance avec n => h_hat biaise dans direction RKHS detectable")
    add("- `wit_q_rel` >> |riesz_g1| : direction RKHS pas capturee par moments simples (g=1,W,A,WA)")
    add("- `dir_ratio` ≈ 1 : q corrige la direction d'erreur de h ; ≈ 0 : q est orthogonal ; < 0 : q aggrave")
    add("- `dir_ratio_h0` ≠ `dir_ratio` : valide empiriquement Correction 1 (dgp.h0() != bridge Fredholm)")
    add("")

    # ── Section 2 : Direction ratios per dose ───────────────────────────────
    add("## 2. Direction ratio par dose (full grid)")
    add("")
    for method in methods:
        add(f"### {method}")
        hdr2 = "| n | dose | dir_ratio | dir_ratio_h0 | plug_err | corr_on_err |"
        sep2 = "|---|------|----------|------------|---------|-----------|"
        add(hdr2); add(sep2)
        for n in n_grid:
            d = decomps.get((method, n))
            if d is None: continue
            for k, a_k in enumerate(A_GRID):
                add(
                    f"| {n} | {a_k:.2f} | "
                    f"{_fmt(d['direction_ratio_grid_mean'][k], '+.3f')} | "
                    f"{_fmt(d['direction_ratio_h0_grid_mean'][k], '+.3f')} | "
                    f"{_fmt(d['plug_in_error_grid_mean'][k], '+.4f')} | "
                    f"{_fmt(d['correction_on_error_grid_mean'][k], '+.4f')} |"
                )
        add("")

    # ── Section 3 : Interpretation ───────────────────────────────────────────
    add("## 3. Interpretation")
    add("")
    add("### Decision Kallus (selon les patterns observes)")
    add("")
    add("| Pattern | Interpretation | Action |")
    add("|---------|---------------|--------|")
    add("| dir_ratio oracle-q ≈ 0.25-0.40 ET xfit-q ≈ 0 | h-bottleneck + q xfit ne voit pas direction | DGP2 stress-test → S6 |")
    add("| dir_ratio oracle-q ≈ 0.25 ET xfit-q ≈ 0.15 | KPVPolicyBridgeQ sous-performe (bonne direction) | Tuner q (clip, lambda_Q) |")
    add("| dir_ratio oracle-q ≈ 0 ET wit_h_excess grand | Weak ID : moments OK mais h non unique | Bennett-lite Phase 11 |")
    add("| dir_ratio xfit-q < 0 | q aggrave la direction | Bug ou hyperparametre — investiguer |")
    add("| wit_q_rel >> |riesz_g1| | Direction RKHS hors moments standard | Documenter en S6 |")
    add("")
    return lines


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="DGP2 Kallus Witness Diagnostic")
    parser.add_argument("--mode", choices=["quick", "main"], default="quick")
    parser.add_argument("--refit-href", action="store_true",
                        help="Force refit of h_ref (ignore cache)")
    args = parser.parse_args()

    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ

    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)

    # Trigger lazy h0 fit early (avoids first-fit jitter inside per-rep timing)
    print("[Init] Triggering dgp.h0 lazy fit ...")
    _ = dgp.h0(np.array([1.0]), np.array([2.0]))
    print("[Init] h0 ready.")

    # Build h_ref (with cache)
    print("[h_ref] Building high-precision KPVBridgeH ...")
    h_ref = build_h_ref(dgp, force=args.refit_href)
    href_meta = _href_meta()

    # Mode configuration
    if args.mode == "quick":
        n_grid = [500, 1000]
        M = 5
    else:
        n_grid = [500, 1000, 2000]
        M = 20
    methods = ["oracle", "xfit"]
    print(f"[Mode] {args.mode}  n in {n_grid}  M={M}  methods={methods}")

    # h_KDE for q_factory (built on a single n=1000 reference sample)
    ref_sample = dgp.generate(n=1000, seed=0)
    h_KDE_factory = _silverman_h(ref_sample.A)

    # Run all (method, n) blocks
    decomps: dict[tuple, dict] = {}
    for method in methods:
        # q_factory prototype for xfit (deepcopy per fold inside _per_fold_arrays)
        q_factory_proto = (
            KPVPolicyBridgeQ(a_grid=A_GRID, h_KDE=h_KDE_factory, lambda_Q=1e-3)
            if method == "xfit" else None
        )
        for n in n_grid:
            seed_base = SEED_BASE + (0 if method == "oracle" else 100000) + n
            label = f"kallus_witness_{method}_n{n}_M{M}"
            print(f"\n[{method} n={n}] {label}")
            data = _run_witness_block(
                dgp, h_ref, method, n, M,
                seed_base=seed_base, label=label, a_grid=A_GRID,
                bridge_kwargs=None,
                q_factory_proto=q_factory_proto,
                raw_dir=_RAW_DIR,
            )
            decomps[(method, n)] = _decompose_witness(data)

    # Build report
    print("\n[Report] Building markdown ...")
    J_true_ref = float(dgp.m_true(float(A_GRID[REF_IDX])))
    lines = _build_report_witness(
        decomps, href_meta, n_grid=n_grid, methods=methods,
        J_true_ref=J_true_ref,
    )
    out = _SUMM_DIR / "dgp2_kallus_witness.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport -> {out}")


if __name__ == "__main__":
    main()
