"""
Phase 14A -- RFF fairness gating for Bennett-indep planning.

Question:
  Is the Random Fourier Feature approximation of the Gaussian RBF kernel
  faithful enough to be a methodologically fair substitute for KPVBridgeH
  in the Bennett-indep comparison?

Sub-steps:
  14A.1 -- Geometric fidelity test (Frobenius / spectral norm error of RFF
           Gram vs true RBF Gram, on actual DGP samples).
  14A.2 -- Functional fidelity test: REG_KPV vs REG_RFF (with m adaptive)
           on DGP2 across SNR x n. Adaptive m grid: stops early if
           consecutive m values give similar results.

Decision:
  Branch A (RFF fair): proceed with full RFF Bennett-indep in Phase 14B.
  Branch B (RFF unfair): switch to RKHS-based Bennett (extend
           KallusMinimaxBridgeH) in Phase 14B.

Outputs:
  simulations/results/raw/phase14A/geometry/<m>.pkl
  simulations/results/raw/phase14A/functional/<method>_snr<S>_n<n>_M50.pkl
  simulations/results/summaries/phase14A_rff_fairness.md

seed_base = 28_000_000
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[2]
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "phase14A"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "phase14A_rff_fairness.md"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
K = len(A_GRID)
REF_IDX = 2
N_FOLDS = 5
RANDOM_STATE = 42
SEED_BASE = 28_000_000

SNR_GRID = [0.90, 0.95]
N_GRID = [1000, 2000]

# Adaptive m grid (sequential, stop early if consecutive m yield similar results)
M_RFF_GRID_FULL = [100, 250, 500, 750, 1000, 1500]


# Reuse utilities
from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
    _compute_J_policy_true,
)
from simulations.experiments.bennett_rff_overnight import _augment_decomp_with_ser
from simulations.methods.kpv_bridge import _rbf_gram, _median_bandwidth as _med_bw
from simulations.methods.rff_bridge_h import RFFBridgeH, RFFFeatureMap1D


# ── 14A.1 Geometry test ──────────────────────────────────────────────────────

def _geometry_test() -> dict:
    """
    For each m in M_RFF_GRID_FULL and each SNR in SNR_GRID:
      Generate a fixed reference sample (n=2000, seed=1).
      Compute K_RBF (true Gaussian RBF Gram) on (W, A, Z) marginals.
      Compute K_RFF (RFF approximation, averaged over 5 random seeds).
      Report Frobenius rel err, spectral rel err, max abs err.
    """
    print("\n=== 14A.1 Geometry test ===")
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP

    out = {}
    for snr in SNR_GRID:
        dgp = MichaelisMentenDGP(snr_W=snr, snr_Z=snr)
        sample = dgp.generate(n=2000, seed=1)
        W, A, Z = sample.W, sample.A, sample.Z
        ell_W = _med_bw(W)
        ell_A = _med_bw(A)
        ell_Z = _med_bw(Z)
        # Use n=500 subsample for the Frobenius computation (n^2 cost)
        rng = np.random.default_rng(0)
        sub = rng.choice(2000, size=500, replace=False)
        W_s, A_s, Z_s = W[sub], A[sub], Z[sub]
        K_W_true = _rbf_gram(W_s, W_s, ell_W)
        K_A_true = _rbf_gram(A_s, A_s, ell_A)
        K_Z_true = _rbf_gram(Z_s, Z_s, ell_Z)
        norms_true = {
            "W": (np.linalg.norm(K_W_true), np.linalg.norm(K_W_true, ord=2)),
            "A": (np.linalg.norm(K_A_true), np.linalg.norm(K_A_true, ord=2)),
            "Z": (np.linalg.norm(K_Z_true), np.linalg.norm(K_Z_true, ord=2)),
        }
        for m in M_RFF_GRID_FULL:
            errors_per_seed = {"W": [], "A": [], "Z": []}
            for s in range(5):
                phi_W = RFFFeatureMap1D(n_features=m, ell=ell_W, seed=s).transform(W_s)
                phi_A = RFFFeatureMap1D(n_features=m, ell=ell_A, seed=s + 1).transform(A_s)
                phi_Z = RFFFeatureMap1D(n_features=m, ell=ell_Z, seed=s + 2).transform(Z_s)
                K_W_rff = phi_W @ phi_W.T
                K_A_rff = phi_A @ phi_A.T
                K_Z_rff = phi_Z @ phi_Z.T
                # Errors per variable
                errors_per_seed["W"].append((
                    np.linalg.norm(K_W_rff - K_W_true) / norms_true["W"][0],     # Frobenius rel
                    np.linalg.norm(K_W_rff - K_W_true, ord=2) / norms_true["W"][1],  # spectral rel
                    float(np.max(np.abs(K_W_rff - K_W_true))),
                ))
                errors_per_seed["A"].append((
                    np.linalg.norm(K_A_rff - K_A_true) / norms_true["A"][0],
                    np.linalg.norm(K_A_rff - K_A_true, ord=2) / norms_true["A"][1],
                    float(np.max(np.abs(K_A_rff - K_A_true))),
                ))
                errors_per_seed["Z"].append((
                    np.linalg.norm(K_Z_rff - K_Z_true) / norms_true["Z"][0],
                    np.linalg.norm(K_Z_rff - K_Z_true, ord=2) / norms_true["Z"][1],
                    float(np.max(np.abs(K_Z_rff - K_Z_true))),
                ))
            for var in ("W", "A", "Z"):
                arr = np.array(errors_per_seed[var])
                out[(snr, m, var)] = {
                    "frob_rel_mean": float(arr[:, 0].mean()),
                    "frob_rel_std": float(arr[:, 0].std()),
                    "spec_rel_mean": float(arr[:, 1].mean()),
                    "max_abs_mean": float(arr[:, 2].mean()),
                }
            print(f"  SNR={snr}, m={m}: W frob={out[(snr,m,'W')]['frob_rel_mean']:.4f}, "
                  f"A frob={out[(snr,m,'A')]['frob_rel_mean']:.4f}, "
                  f"Z frob={out[(snr,m,'Z')]['frob_rel_mean']:.4f}")
    return out


# ── 14A.2 Functional test ────────────────────────────────────────────────────

def _make_reg_kpv(h_KDE, ell_W, ell_A, ell_Z):
    """REG with KPV-super h-bridge."""
    from simulations.methods.bennett_functional_dr import BennettFunctionalDR
    def factory():
        # BennettFunctionalDR with lambda_r=1e10 -> plug-in only.
        # h-bridge is KPVBridgeH internally.
        return BennettFunctionalDR(
            lambda_h=3e-5, ell_scale=3.5,
            lambda_r=1e10, degree=2, feature_map_type="polynomial",
            n_folds=N_FOLDS, a_grid=A_GRID.tolist(),
            bandwidth=h_KDE, seed=RANDOM_STATE, ref_dose_index=REF_IDX,
        )
    return factory


def _make_reg_rff(h_KDE, ell_W, ell_A, ell_Z, n_features: int, rff_seed: int):
    """
    REG with RFF h-bridge (replaces KPVBridgeH). Uses a thin wrapper that:
      - in fit: builds RFFBridgeH instead of KPVBridgeH per fold
      - in predict + plug_in_policy: same API
      - everything else identical to BennettFunctionalDR's REG mode.

    To minimise code changes, we monkey-patch BennettFunctionalDR's fit() to
    use RFFBridgeH per fold. The cleanest way is to subclass.
    """
    from simulations.methods.bennett_functional_dr import BennettFunctionalDR
    from simulations.methods.rff_bridge_h import RFFBridgeH
    import numpy as np
    from sklearn.model_selection import KFold

    # Subclass that uses RFFBridgeH instead of KPVBridgeH
    class BennettFunctionalDR_RFFh(BennettFunctionalDR):
        def __init__(self, *, n_features_h: int, rff_seed_h: int, **kwargs):
            super().__init__(**kwargs)
            self.n_features_h = int(n_features_h)
            self.rff_seed_h = int(rff_seed_h)

        @property
        def name(self) -> str:
            return f"REG_RFF_h(m={self.n_features_h})"

        def fit(self, sample, m_true_fn=None):
            # Same logic as BennettFunctionalDR.fit but with RFFBridgeH instead of KPVBridgeH
            from simulations.dgp.base import DGPSample
            from simulations.methods.base import EstimationResult
            from simulations.methods.bennett_riesz import BennettPolicyRiesz, StabilizedBennettPolicyRiesz
            n = len(sample.Y)
            Y = np.asarray(sample.Y, dtype=float)
            A = np.asarray(sample.A, dtype=float)
            W = np.asarray(sample.W, dtype=float)
            Z = np.asarray(sample.Z, dtype=float)
            a_grid = np.array(self.a_grid, dtype=float)
            K_doses = len(a_grid)

            h_KDE = (
                float(self.bandwidth) if self.bandwidth is not None
                else self._silverman_h(A)
            )
            J_policy_true = np.full(K_doses, np.nan)
            if m_true_fn is not None:
                try:
                    from numpy.polynomial.hermite_e import hermegauss
                    x_q, w_q = hermegauss(25)
                    for d, a_d in enumerate(a_grid):
                        t_vals = float(a_d) + h_KDE * x_q
                        vals = np.array([float(m_true_fn(float(t))) for t in t_vals])
                        J_policy_true[d] = float(np.dot(w_q, vals) / np.sqrt(2.0 * np.pi))
                except Exception:
                    pass

            ell_W0 = self._median_bandwidth(W) * self.ell_scale
            ell_A0 = self._median_bandwidth(A) * self.ell_scale
            ell_Z0 = self._median_bandwidth(Z) * self.ell_scale

            h_obs = np.full(n, np.nan)
            h_policy = np.full((n, K_doses), np.nan)
            r_hat = np.full((n, K_doses), np.nan)
            riesz_resid_folds = np.full((self.n_folds, K_doses), np.nan)

            kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            indices = np.arange(n)

            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(indices)):
                W_tr, A_tr, Z_tr, Y_tr = W[train_idx], A[train_idx], Z[train_idx], Y[train_idx]
                W_te, A_te, Z_te = W[test_idx], A[test_idx], Z[test_idx]

                # ── 1. Fit RFF h-bridge on train fold ─────────────────────────
                h_bridge = RFFBridgeH(
                    lambda_1=self.lambda_h,
                    lambda_2=self.lambda_h,
                    ell_W=ell_W0,
                    ell_A=ell_A0,
                    ell_Z=ell_Z0,
                    n_features_W=self.n_features_h,
                    n_features_A=self.n_features_h,
                    n_features_Z=self.n_features_h,
                    seed=self.rff_seed_h + fold_idx * 1009,
                )
                h_bridge.fit(W_tr, A_tr, Z_tr, Y_tr)
                h_obs[test_idx] = h_bridge.predict(W_te, A_te)
                for d, a_d in enumerate(a_grid):
                    h_policy[test_idx, d] = h_bridge.plug_in_policy(W_te, float(a_d), h_KDE)

                # ── 2. Fit RFF Riesz r (same as default Bennett, lambda_r huge -> REG only)
                # Since lambda_r=1e10, correction is essentially zero. r_hat stays small.
                rff_seed_fold = self.rff_seed_base + 1009 * fold_idx if self.rff_seed_base is not None else None
                common_kwargs = dict(
                    lambda_r=self.lambda_r,
                    degree=self.degree,
                    feature_map_type=self.feature_map_type,
                    n_features=self.n_features,
                    ell_scale=self.ell_scale_rff,
                    rff_seed=rff_seed_fold,
                )
                if self.stabilized:
                    riesz = StabilizedBennettPolicyRiesz(
                        lambda_stab=self.lambda_stab,
                        gamma_critic=self.gamma_critic,
                        **common_kwargs,
                    )
                else:
                    riesz = BennettPolicyRiesz(**common_kwargs)
                riesz.fit(W_tr, A_tr, Z_tr, a_grid=a_grid.tolist(), bandwidth=h_KDE)
                for d, a_d in enumerate(a_grid):
                    r_hat[test_idx, d] = riesz.predict(Z_te, A_te, a_target=float(a_d))
                for d, a_d in enumerate(a_grid):
                    try:
                        res = riesz.riesz_residual(W_te, A_te, Z_te, a_target=float(a_d))
                        riesz_resid_folds[fold_idx, d] = res
                    except Exception:
                        pass

            # ── Build EstimationResult (mostly copy from BennettFunctionalDR) ─
            residual_Y = (Y - h_obs)[:, None]
            correction = r_hat * residual_Y
            score = h_policy + correction
            J_reg = h_policy.mean(axis=0)
            J_bennett = score.mean(axis=0)
            V_hat_grid = np.var(score, axis=0, ddof=1)
            V_reg_grid = np.var(h_policy, axis=0, ddof=1)
            V_corr_grid = np.var(correction, axis=0, ddof=1)
            V_total = float(np.mean(V_hat_grid))
            V_reg = float(np.mean(V_reg_grid))
            V_corr = float(np.mean(V_corr_grid))
            V_cov = (V_total - V_reg - V_corr) / 2.0

            ESS_abs_grid = np.array([self._ess_abs(r_hat[:, d]) for d in range(K_doses)])
            ESS_min = float(np.min(ESS_abs_grid))
            r_abs = np.abs(r_hat)
            weight_p99_grid = np.array([
                float(np.percentile(r_abs[:, d], 99)) for d in range(K_doses)
            ])
            weight_max_grid = np.array([
                float(np.max(r_abs[:, d])) for d in range(K_doses)
            ])
            riesz_resid_mean_per_dose = np.nanmean(riesz_resid_folds, axis=0)
            riesz_resid_overall = float(np.nanmean(riesz_resid_folds))

            ref_idx = min(self.ref_dose_index, K_doses - 1)
            extra = {
                "J_reg": J_reg,
                "J_dr": J_bennett,
                "J_true": np.full(K_doses, np.nan),
                "J_policy_true": J_policy_true,
                "V_hat_grid": V_hat_grid,
                "V_reg_grid": V_reg_grid,
                "V_correction_grid": V_corr_grid,
                "ESS_grid": ESS_abs_grid,
                "ESS_ratio_grid": ESS_abs_grid / n,
                "ESS_min": ESS_min,
                "ESS_ratio_min": ESS_min / n,
                "weight_p99_grid": weight_p99_grid,
                "weight_max_grid": weight_max_grid,
                "q_clip_fraction": 0.0,
                "q_negative_share_grid": np.zeros(K_doses),
                "riesz_residual_grid_mean": riesz_resid_mean_per_dose,
                "cond_M_grid_mean": np.full(K_doses, np.nan),
                "neg_share_grid_mean": np.zeros(K_doses),
                "bandwidth": h_KDE,
                "jensen_gap": np.full(K_doses, np.nan),
                "mise": np.nan,
                "a_grid": a_grid,
                "ref_dose_index": ref_idx,
                "ref_dose": float(a_grid[ref_idx]),
                "n_folds": self.n_folds,
                "lambda_h": self.lambda_h,
                "lambda_r": self.lambda_r,
                "degree": self.degree,
                "feature_map_type": self.feature_map_type,
                "n_features": self.n_features if self.feature_map_type == "rff" else None,
                "ell_scale_rff": self.ell_scale_rff if self.feature_map_type == "rff" else None,
                "stabilized": self.stabilized,
                "lambda_stab": self.lambda_stab if self.stabilized else None,
                "gamma_critic": self.gamma_critic if self.stabilized else None,
                "n_features_h_RFF": self.n_features_h,
                "V_total": V_total,
                "V_reg": V_reg,
                "V_corr": V_corr,
                "V_cov": V_cov,
                "riesz_residual_grid": riesz_resid_mean_per_dose,
                "riesz_residual_mean": riesz_resid_overall,
                "alignment_ratio_grid": np.full(K_doses, np.nan),
                "alignment_mean": np.nan,
            }

            psi_hat = float(J_bennett[ref_idx])
            V_hat = float(V_hat_grid[ref_idx])
            self._result = EstimationResult(
                psi_hat=psi_hat,
                V_hat=V_hat,
                method_name=self.name,
                n=n,
                extra=extra,
            )
            return self

    def factory():
        return BennettFunctionalDR_RFFh(
            n_features_h=n_features,
            rff_seed_h=rff_seed,
            lambda_h=3e-5, ell_scale=3.5,
            lambda_r=1e10, degree=2, feature_map_type="polynomial",
            n_folds=N_FOLDS, a_grid=A_GRID.tolist(),
            bandwidth=h_KDE, seed=RANDOM_STATE, ref_dose_index=REF_IDX,
        )
    return factory


def _functional_test(M_reps: int, m_grid: List[int], force: bool = False) -> dict:
    """
    For each (snr, n) cell, run REG_KPV and REG_RFF (multiple m).
    Returns dict[(snr, n, method)] -> decomp.

    Adaptive: stops adding m values when consecutive m yield similar |b|/SE
    (within 5% relative).
    """
    print("\n=== 14A.2 Functional test ===")
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP

    out = {}
    for snr in SNR_GRID:
        dgp = MichaelisMentenDGP(snr_W=snr, snr_Z=snr)
        ref = dgp.generate(n=2000, seed=1)
        h_KDE = _silverman_h(ref.A)
        ell_W0 = _median_bandwidth(ref.W)
        ell_A0 = _median_bandwidth(ref.A)
        ell_Z0 = _median_bandwidth(ref.Z)
        J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)

        for n in N_GRID:
            cell_dir = _RAW_DIR / "functional" / f"snr{int(snr*100)}_n{n}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            seed_base_cell = SEED_BASE + SNR_GRID.index(snr) * 100_000 + N_GRID.index(n) * 1_000 + n
            print(f"\n--- Cell SNR={snr}, n={n}, M={M_reps} ---")

            # 1. REG_KPV reference
            cid = "REG_KPV"
            label = f"{cid}_M{M_reps}"
            print(f"  [{cid}] n={n}")
            t0 = time.time()
            data = _run_block(
                dgp, _make_reg_kpv(h_KDE, ell_W0, ell_A0, ell_Z0),
                A_GRID, J_pt, n=n, M=M_reps, seed_base=seed_base_cell,
                label=label, raw_dir=cell_dir,
            )
            elapsed = time.time() - t0
            decomp = _decompose(data)
            if decomp is not None:
                decomp = _augment_decomp_with_ser(decomp, data)
            out[(snr, n, cid)] = decomp
            if decomp is not None:
                print(f"    bse_ref={decomp['bse_grid'][REF_IDX]:.4f}  "
                      f"cov={decomp['coverage_ref']:.3f}  "
                      f"SER={decomp.get('ser_ref', float('nan')):.3f}  [{elapsed:.0f}s]")

            # 2. REG_RFF (adaptive m loop)
            kpv_bse = float(out[(snr, n, "REG_KPV")]["bse_grid"][REF_IDX])
            prev_bse_rff = None
            for i_m, m_rff in enumerate(m_grid):
                cid = f"REG_RFF_m{m_rff}"
                label = f"{cid}_M{M_reps}"
                print(f"  [{cid}] n={n}")
                t0 = time.time()
                data = _run_block(
                    dgp, _make_reg_rff(h_KDE, ell_W0, ell_A0, ell_Z0, m_rff, rff_seed=2026 + i_m),
                    A_GRID, J_pt, n=n, M=M_reps, seed_base=seed_base_cell,
                    label=label, raw_dir=cell_dir,
                )
                elapsed = time.time() - t0
                decomp = _decompose(data)
                if decomp is not None:
                    decomp = _augment_decomp_with_ser(decomp, data)
                out[(snr, n, cid)] = decomp
                if decomp is not None:
                    bse_rff = float(decomp["bse_grid"][REF_IDX])
                    print(f"    bse_ref={bse_rff:.4f}  "
                          f"cov={decomp['coverage_ref']:.3f}  "
                          f"SER={decomp.get('ser_ref', float('nan')):.3f}  [{elapsed:.0f}s]")
                    # Adaptive early stopping: if bse converges to KPV within 5% AND
                    # consecutive m values are stable within 3%, stop.
                    if prev_bse_rff is not None:
                        rel_to_prev = abs(bse_rff - prev_bse_rff) / max(abs(prev_bse_rff), 1e-6)
                        rel_to_kpv = abs(bse_rff - kpv_bse) / max(abs(kpv_bse), 1e-6)
                        if rel_to_kpv < 0.05 and rel_to_prev < 0.03:
                            print(f"    [Early stop] bse_rff converged to KPV within 5% "
                                  f"AND stable to prev m within 3% (rel_to_kpv={rel_to_kpv:.3f}, "
                                  f"rel_to_prev={rel_to_prev:.3f})")
                            break
                    prev_bse_rff = bse_rff
    return out


# ── Decision builder ──────────────────────────────────────────────────────────

def _decide_branch(geometry: dict, functional: dict) -> Tuple[str, dict]:
    """
    Decide Branch A (RFF fair) or B (RKHS needed).

    Branch A criteria :
      - At m=500 or below: Frobenius rel err <= 5% across all (snr, var)
      - At m=500 or below: REG_RFF |b|/SE within 10% of REG_KPV across cells

    Branch B otherwise.

    Returns (branch_name, info_dict).
    """
    info = {}
    # Geometry: smallest m where ALL Frobenius rel err <= 5%
    m_geometry_pass = None
    for m in sorted({k[1] for k in geometry.keys()}):
        all_low = True
        for snr in SNR_GRID:
            for var in ("W", "A", "Z"):
                if (snr, m, var) in geometry:
                    if geometry[(snr, m, var)]["frob_rel_mean"] > 0.05:
                        all_low = False
                        break
            if not all_low:
                break
        if all_low:
            m_geometry_pass = m
            break
    info["m_geometry_pass"] = m_geometry_pass

    # Functional: pick m where REG_RFF best matches REG_KPV
    func_passes = []
    for (snr, n, cid), dec in functional.items():
        if not cid.startswith("REG_RFF_m"):
            continue
        m_rff = int(cid.replace("REG_RFF_m", ""))
        kpv_dec = functional.get((snr, n, "REG_KPV"))
        if kpv_dec is None or dec is None:
            continue
        kpv_bse = float(kpv_dec["bse_grid"][REF_IDX])
        rff_bse = float(dec["bse_grid"][REF_IDX])
        rel = abs(rff_bse - kpv_bse) / max(abs(kpv_bse), 1e-6)
        if rel < 0.10:
            func_passes.append((m_rff, snr, n, rel))
    info["functional_passes"] = func_passes

    # Decide
    if m_geometry_pass is not None and m_geometry_pass <= 500:
        if len(func_passes) >= len(SNR_GRID) * len(N_GRID):
            return "A", info
    if m_geometry_pass is not None and m_geometry_pass <= 1000:
        return "A_with_min_m", info
    return "B", info


def _build_report(geometry, functional, branch, info, m_grid_used):
    import datetime
    lines = [
        f"# Phase 14A — RFF fairness gating",
        f"Date: {datetime.date.today().isoformat()}",
        f"Branch decision: **{branch}**",
        "",
        "## 0. Verdict",
        "",
    ]
    if branch == "A":
        lines.append("**Branch A (RFF fair)**: RFF reproduces RBF Gram and KPV-h within tolerance.")
        lines.append("Phase 14B can use full RFF Bennett-indep (RFF for both h and r).")
    elif branch == "A_with_min_m":
        lines.append(f"**Branch A conditional (min m={info.get('m_geometry_pass')})**: RFF acceptable but only at high m.")
        lines.append("Phase 14B uses RFF with this minimum m.")
    else:
        lines.append("**Branch B (RFF unfair)**: switch to RKHS-based Bennett-h.")
        lines.append("Phase 14B will extend KallusMinimaxBridgeH instead.")
    lines.append("")

    # 14A.1 geometry
    lines.append("## 1. Geometric fidelity (RFF Gram vs true RBF Gram, n=500 sub-sample)")
    lines.append("")
    lines.append("Frobenius relative error = ||K_RFF - K_RBF||_F / ||K_RBF||_F (averaged over 5 RFF seeds)")
    lines.append("")
    lines.append("| SNR | m | W frob | W spec | A frob | A spec | Z frob | Z spec |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for snr in SNR_GRID:
        for m in sorted({k[1] for k in geometry.keys()}):
            row = [f"{snr}", f"{m}"]
            for var in ("W", "A", "Z"):
                if (snr, m, var) in geometry:
                    row.append(f"{geometry[(snr, m, var)]['frob_rel_mean']:.4f}")
                    row.append(f"{geometry[(snr, m, var)]['spec_rel_mean']:.4f}")
                else:
                    row.append("---")
                    row.append("---")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(f"**Smallest m with all Frobenius rel err <= 5% : m = {info.get('m_geometry_pass')}**")
    lines.append("")

    # 14A.2 functional
    lines.append("## 2. Functional fidelity (REG_KPV vs REG_RFF, M=50 reps)")
    lines.append("")
    lines.append("| SNR | n | method | bias | |b|/SE | cov | SER | rel to KPV |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (snr, n, cid), dec in functional.items():
        if dec is None:
            continue
        bias = float(dec["bias_dr"][REF_IDX])
        bse = float(dec["bse_grid"][REF_IDX])
        cov = float(dec["coverage_ref"])
        ser = float(dec.get("ser_ref", float("nan"))) if "ser_ref" in dec else float("nan")
        if cid.startswith("REG_RFF_"):
            kpv_dec = functional.get((snr, n, "REG_KPV"))
            kpv_bse = float(kpv_dec["bse_grid"][REF_IDX]) if kpv_dec else float("nan")
            rel = abs(bse - kpv_bse) / max(abs(kpv_bse), 1e-6) if not np.isnan(kpv_bse) else float("nan")
            rel_str = f"{rel*100:+.1f}%"
        else:
            rel_str = "(ref)"
        lines.append(
            f"| {snr} | {n} | {cid} | {bias:+.4f} | {bse:.4f} | {cov:.3f} | {ser:.3f} | {rel_str} |"
        )
    lines.append("")

    # 14A.3 decision
    lines.append("## 3. Decision logic")
    lines.append("")
    lines.append(f"- Smallest m with Frobenius < 5%: {info.get('m_geometry_pass')}")
    lines.append(f"- Functional cells where REG_RFF within 10% of REG_KPV: "
                 f"{len(info.get('functional_passes', []))}")
    lines.append(f"- m grid actually used: {m_grid_used}")
    lines.append(f"- Final branch: **{branch}**")
    lines.append("")

    out = "\n".join(lines)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"\nReport saved: {_REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--m-grid", type=str, default=None,
                        help="Comma-separated RFF m values; default adaptive 100..1500")
    args = parser.parse_args()
    M_reps = args.M

    if args.mode == "smoke":
        m_grid = [250, 500]
        M_reps = 5
        print("SMOKE mode: m=[250, 500], M=5")
    else:
        if args.m_grid is not None:
            m_grid = [int(x) for x in args.m_grid.split(",")]
        else:
            m_grid = M_RFF_GRID_FULL

    print("=" * 72)
    print(f"Phase 14A RFF fairness gating  --mode {args.mode}, M={M_reps}")
    print(f"  m_grid (max) = {m_grid}")
    print(f"  SNR_GRID = {SNR_GRID}, N_GRID = {N_GRID}")
    print("=" * 72)

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    geometry = _geometry_test()
    with open(_RAW_DIR / "geometry.pkl", "wb") as fh:
        pickle.dump(geometry, fh)
    print(f"\nGeometry test elapsed: {time.time()-t0:.0f}s")

    t1 = time.time()
    functional = _functional_test(M_reps=M_reps, m_grid=m_grid)
    with open(_RAW_DIR / "functional.pkl", "wb") as fh:
        pickle.dump(functional, fh)
    print(f"\nFunctional test elapsed: {time.time()-t1:.0f}s")

    branch, info = _decide_branch(geometry, functional)
    print(f"\n>>> BRANCH DECISION: {branch}")
    print(f"    info: {info}")

    _build_report(geometry, functional, branch, info, m_grid)

    print(f"\nTotal Phase 14A: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
    return branch, info


if __name__ == "__main__":
    main()
