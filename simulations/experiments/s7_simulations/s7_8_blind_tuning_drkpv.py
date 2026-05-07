"""
Phase 17 -- Adaptive Blind Tuning Controller

Tunes Bennett (BennettIndepFunctionalDR) and DRKPV (DRKernel + KPV bridges)
using ONLY blind metrics (no J_true reference). The controller:

1. Runs adaptive 5-stage protocol (1D scans -> 2D refine -> cross-cell ->
   disjoint validation -> regime robustness)
2. Persists state in JSON for resume safety
3. Adaptive grids based on previous stage diagnostics
4. Per-stage reports for monitoring

Usage:
    python -u -m simulations.experiments.phase17_blind_tuning \
        --method bennett --mode full
    python -u -m simulations.experiments.phase17_blind_tuning \
        --method drkpv --mode full --resume

Anti-faux-positifs discipline:
    - Disjoint seed_base for tuning (60M Bennett, 70M DRKPV) and validation (+500K)
    - Composite blind score (no single metric)
    - M=100 obligatoire for Stage 4-5
    - Quality gates per stage
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import traceback
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[3]  # +1: moved to experiments/simulation/
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "s7" / "blind_tuning_drkpv"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
REF_IDX = 2

# ════════════════════════════════════════════════════════════════════════════
#  Composite blind scoring functions (Phase 16 audit-validated weights)
# ════════════════════════════════════════════════════════════════════════════

def bennett_blind_score(diags: Dict[str, float]) -> float:
    """
    Composite blind score for Bennett. Lower is better.
    Weights from Phase 16 metric audit (rho with cov_ref).
    """
    kappa_h = diags.get("kappa_h", float("nan"))
    kappa_r = diags.get("kappa_r", float("nan"))
    rr_ref = diags.get("RR_ref", float("nan"))
    residual_norm_h = diags.get("residual_norm_h", float("nan"))
    eff_rank_h = diags.get("eff_rank_h", float("nan"))
    eff_rank_r = diags.get("eff_rank_r", float("nan"))

    # Hard fails
    if not np.isfinite(kappa_h) or kappa_h > 1e10:
        return float("inf")
    if not np.isfinite(kappa_r) or kappa_r > 1e10:
        return float("inf")
    if not np.isfinite(rr_ref) or rr_ref > 1e-1:
        return float("inf")
    if not np.isfinite(residual_norm_h):
        return float("inf")

    # Z-score-style normalisations (target levels from Phase 16 audit)
    s_residual_h = residual_norm_h / 0.005           # cible < 0.005
    s_eff_rank_r = max(0, 50 - eff_rank_r) / 50      # cible >= 50
    s_eff_rank_h = max(0, 50 - eff_rank_h) / 50
    s_RR = rr_ref / 0.005                             # cible < 0.005
    s_kappa_h = max(0, np.log10(kappa_h) - 6) / 4     # gate at 1e6
    s_kappa_r = max(0, np.log10(kappa_r) - 4) / 4

    score = (
        0.30 * s_residual_h +
        0.20 * s_eff_rank_r +
        0.15 * s_eff_rank_h +
        0.15 * s_RR +
        0.10 * s_kappa_h +
        0.10 * s_kappa_r
    )
    return float(score)


def drkpv_blind_score(diags: Dict[str, float]) -> float:
    """
    Composite blind score for DRKPV. Lower is better.

    Note: DRKPV on DGP2 has INHERENTLY catastrophic kappa_r (~1e+50 by design,
    Phase 16 EXP-C confirmed). Hard-failing on kappa_r would reject ALL configs.
    So we use it as a PENALTY (ranks differences in log10) but not as a gate.

    Real gates:
      - RR > 1e-1 (true convergence failure)
      - residual_norm_r > 0.5 (very poor fit)
    """
    kappa_r = diags.get("kappa_r", float("nan"))
    rr_ref = diags.get("RR_ref", float("nan"))
    residual_norm_r = diags.get("residual_norm_r", float("nan"))
    SER = diags.get("SER", float("nan"))
    ESS_min_ratio = diags.get("ESS_min_ratio", float("nan"))

    # Hard fails (real convergence failure only)
    if not np.isfinite(rr_ref) or rr_ref > 1e-1:
        return float("inf")
    if np.isfinite(residual_norm_r) and residual_norm_r > 0.5:
        return float("inf")

    # Continuous penalties
    s_RR = rr_ref / 0.005
    s_residual_r = (residual_norm_r if np.isfinite(residual_norm_r) else 1.0) / 0.05

    # kappa_r penalty: relative ordering across configs (since all are huge)
    # log10(kappa_r) typically 40-70 for DGP2; lower is better
    if np.isfinite(kappa_r) and kappa_r > 0:
        s_kappa_r = max(0, (np.log10(kappa_r) - 30) / 40)  # 30-70 → 0-1
    else:
        s_kappa_r = 1.0

    # SER penalty (target ~1.0). Use only if available (M >= ~30 reps).
    if np.isfinite(SER):
        s_SER = abs(SER - 1.0) / 0.5  # 0 if SER=1, 1 if SER=0.5 or 1.5
    else:
        s_SER = 0.0  # no penalty if not available

    return float(0.30 * s_RR + 0.20 * s_residual_r + 0.20 * s_kappa_r + 0.30 * s_SER)


SCORING = {"bennett": bennett_blind_score, "drkpv": drkpv_blind_score}


# ════════════════════════════════════════════════════════════════════════════
#  Diagnostic extraction from per-rep records
# ════════════════════════════════════════════════════════════════════════════

def _safe(v, default=float("nan")):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _array_safe(arr, idx, default=float("nan")):
    try:
        return float(arr[idx])
    except Exception:
        return default


def extract_diagnostics(data: dict, ref_idx: int = REF_IDX) -> Dict[str, float]:
    """
    Aggregate per-rep records into mean diagnostics for the blind score.
    Returns dict with keys used by *_blind_score.
    """
    recs = [r for r in data.get("records", []) if r.get("error") is None]
    if not recs:
        return {"M_ok": 0}
    n = int(data["meta"]["n"])

    # Spectral diagnostics (mean across reps)
    out = {
        "M_ok": len(recs),
        "n": n,
        "kappa_h": float(np.nanmean([_safe(r.get("kappa_h")) for r in recs])),
        "kappa_r": float(np.nanmean([_safe(r.get("kappa_r")) for r in recs])),
        "eff_rank_h": float(np.nanmean([_safe(r.get("eff_rank_h")) for r in recs])),
        "eff_rank_r": float(np.nanmean([_safe(r.get("eff_rank_r")) for r in recs])),
        "residual_norm_h": float(np.nanmean([_safe(r.get("residual_norm_h")) for r in recs])),
        "residual_norm_r": float(np.nanmean([_safe(r.get("residual_norm_r")) for r in recs])),
        "RR_ref": float(np.nanmean([_array_safe(r.get("riesz_residual_grid_mean", []), ref_idx) for r in recs])),
        "ESS_min_ratio": float(np.nanmean([_safe(r.get("ESS_min")) for r in recs])) / n,
        "w_p99_ref": float(np.nanmean([_array_safe(r.get("weight_p99_grid", []), ref_idx) for r in recs])),
    }

    # SER (pseudo-blind): SE_predicted_mean / SD_empirical
    try:
        J_dr = np.array([r["J_dr"] for r in recs])
        V_hg = np.array([r["V_hat_grid"] for r in recs])
        with np.errstate(invalid="ignore"):
            se_per = np.sqrt(np.maximum(V_hg, 0.0) / n)
        SE_pred = float(np.nanmean(se_per[:, ref_idx]))
        SD_emp = float(np.std(J_dr[:, ref_idx], ddof=1))
        out["SER"] = SE_pred / SD_emp if SD_emp > 1e-12 else float("nan")
        # bse_ref kept for FINAL VALIDATION only (post-hoc), NEVER for selection
        if "J_policy_true" in data["meta"]:
            J_pt = np.array(data["meta"]["J_policy_true"])
            bias_ref = float(np.mean(J_dr[:, ref_idx]) - J_pt[ref_idx])
            out["_bse_ref_NONBLIND"] = abs(bias_ref) / SE_pred if SE_pred > 1e-12 else float("nan")
            out["_cov_ref_NONBLIND"] = float(np.mean(np.abs(J_dr[:, ref_idx] - J_pt[ref_idx]) <= 1.96 * se_per[:, ref_idx]))
            out["_bias_ref_NONBLIND"] = bias_ref
    except Exception:
        out["SER"] = float("nan")

    return out


# ════════════════════════════════════════════════════════════════════════════
#  Estimator factories
# ════════════════════════════════════════════════════════════════════════════

# Default config templates (starting points for Stage 1 scans)
BENNETT_DEFAULTS = {
    "m_h": 2000, "m_c": 500,
    "ell_h": 2.75, "ell_c": 2.0,
    "lambda_h": 1e-5, "gamma_critic": 1e-4,
    "lambda_r": 1e-2, "n_features_r": 500, "ell_scale_r": 3.5,
}

DRKPV_DEFAULTS = {
    "lambda_h": 3e-5,        # for KPVBridgeH lambda_1=lambda_2
    "ell_scale": 3.5,        # multiplier for ell_W, ell_A, ell_Z
    "lambda_Q": 1e-3,
    "clip_factor": None,     # None or float (5, 3, 2)
}


def make_bennett_estimator(config: Dict, dgp, h_KDE: float,
                            ell_W: float, ell_A: float, ell_Z: float):
    """Build BennettIndepFunctionalDR from config dict."""
    from simulations.methods.bennett_indep_dr import BennettIndepFunctionalDR
    return BennettIndepFunctionalDR(
        m_h=int(config.get("m_h", 2000)),
        m_c=int(config.get("m_c", 500)),
        ell_h=float(config.get("ell_h", 2.75)),
        ell_c=float(config.get("ell_c", 2.0)),
        lambda_h=float(config.get("lambda_h", 1e-5)),
        gamma_critic=float(config.get("gamma_critic", 1e-4)),
        lambda_r=float(config.get("lambda_r", 1e-2)),
        n_features_r=int(config.get("n_features_r", 500)),
        ell_scale_r=float(config.get("ell_scale_r", 3.5)),
        n_folds=5,
        a_grid=A_GRID.tolist(),
        bandwidth=h_KDE,
        ref_dose_index=REF_IDX,
        seed=42,
    )


def make_drkpv_estimator(config: Dict, dgp, h_KDE: float,
                          ell_W: float, ell_A: float, ell_Z: float):
    """Build DRKernel + KPVPolicyBridgeQ + KPVBridgeH from config dict."""
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
    ell_scale = float(config.get("ell_scale", 3.5))
    n = 1000  # used for clip; will be overridden by sample size in DRKernel
    clip_factor = config.get("clip_factor", None)
    clip = (clip_factor * (n ** 0.25)) if clip_factor is not None else None
    q_model = KPVPolicyBridgeQ(
        a_grid=A_GRID, h_KDE=h_KDE,
        lambda_Q=float(config.get("lambda_Q", 1e-3)),
        ell_W=ell_W * ell_scale,
        ell_A=ell_A * ell_scale,
        ell_Z=ell_Z * ell_scale,
        clip=clip,
        compute_cond=True,
    )
    return DRKernel(
        a_grid=A_GRID, q_model=q_model, bandwidth=h_KDE,
        n_folds=5, random_state=42,
        ref_dose_index=REF_IDX, cross_fit_q=True,
        bridge_kwargs=dict(
            lambda_1=float(config.get("lambda_h", 3e-5)),
            lambda_2=float(config.get("lambda_h", 3e-5)),
            ell_W=ell_W * ell_scale,
            ell_A=ell_A * ell_scale,
            ell_Z=ell_Z * ell_scale,
        ),
    )


FACTORIES = {"bennett": make_bennett_estimator, "drkpv": make_drkpv_estimator}
DEFAULTS = {"bennett": BENNETT_DEFAULTS, "drkpv": DRKPV_DEFAULTS}


# AdaptiveTuner extracted to pci.tuning.adaptive_tuner_drkpv.
# The local AdaptiveTuner class below remains for backward compatibility (it
# also handles method="bennett" via the original Phase 17 score for legacy
# comparisons). New code should import DRKPVAdaptiveTuner directly from
# pci.tuning, which only handles the DRKPV protocol (Phase 17 was sound for
# DRKPV; Bennett needed Phase 18 corrections).
from pci.tuning.adaptive_tuner_drkpv import DRKPVAdaptiveTuner  # noqa: F401


# ════════════════════════════════════════════════════════════════════════════
#  Cell setup helpers
# ════════════════════════════════════════════════════════════════════════════

def _build_dgp_and_meta(snr_W: float, snr_Z: float, n: int):
    """Construct DGP2 + reference sample + bandwidth + ell scales."""
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    from simulations.experiments.dgp2_bias_diagnostics import (
        _silverman_h, _median_bandwidth, _compute_J_policy_true,
    )
    dgp = MichaelisMentenDGP(snr_W=snr_W, snr_Z=snr_Z)
    ref = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref.A)
    ell_W0 = _median_bandwidth(ref.W)
    ell_A0 = _median_bandwidth(ref.A)
    ell_Z0 = _median_bandwidth(ref.Z)
    J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)
    return dgp, h_KDE, ell_W0, ell_A0, ell_Z0, J_pt


def _run_one_config(method: str, config: Dict, *,
                     n: int, snr_W: float, snr_Z: float,
                     M: int, seed_base: int,
                     raw_dir: Path, label: str) -> Dict[str, Any]:
    """Run M reps for a single config and return diagnostics + score."""
    from simulations.experiments.dgp2_bias_diagnostics import _run_block
    dgp, h_KDE, ell_W0, ell_A0, ell_Z0, J_pt = _build_dgp_and_meta(snr_W, snr_Z, n)
    factory_factory = FACTORIES[method]

    def factory():
        return factory_factory(config, dgp, h_KDE, ell_W0, ell_A0, ell_Z0)

    raw_dir.mkdir(parents=True, exist_ok=True)
    data = _run_block(
        dgp, factory, A_GRID, J_pt,
        n=n, M=M, seed_base=seed_base,
        label=label, raw_dir=raw_dir,
    )
    diags = extract_diagnostics(data, ref_idx=REF_IDX)
    score = SCORING[method](diags)
    return {
        "label": label,
        "config": config,
        "diags": diags,
        "score": score,
        "n": n, "snr_W": snr_W, "snr_Z": snr_Z, "M": M,
    }


# ════════════════════════════════════════════════════════════════════════════
#  AdaptiveTuner -- the controller
# ════════════════════════════════════════════════════════════════════════════

class AdaptiveTuner:
    """5-stage adaptive blind tuner for Bennett or DRKPV on DGP2."""

    def __init__(self, method: str, mode: str = "full",
                  start_stage: int = 1, stop_stage: int = 5):
        if method not in ("bennett", "drkpv"):
            raise ValueError(f"Unknown method {method!r}")
        self.method = method
        self.mode = mode
        self.start_stage = start_stage
        self.stop_stage = stop_stage

        # Seed bases
        if method == "bennett":
            self.seed_tune = 60_000_000
            self.seed_valid = 60_500_000
            self.seed_robust = 61_000_000
        else:
            self.seed_tune = 70_000_000
            self.seed_valid = 70_500_000
            self.seed_robust = 71_000_000

        self.raw_dir = _RAW_DIR / method
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.summ_dir = _SUMM_DIR
        self.state_path = self.raw_dir / "state.json"

        # M sizes
        if mode == "smoke":
            self.M_explore = 3
            self.M_validate = 5
        else:
            self.M_explore = 30
            self.M_validate = 100

        self.t_start = time.time()

    def _load_state(self) -> Dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {
            "method": self.method,
            "current_stage": 1,
            "converged": False,
            "stage_history": [],
            "current_top_configs": [],
            "drapeaux": [],
            "compute_used_minutes": 0.0,
            "winner": None,
        }

    @staticmethod
    def _make_json_serializable(obj):
        """Recursively convert tuple dict-keys to strings for JSON serialization."""
        if isinstance(obj, dict):
            return {
                (str(k) if isinstance(k, tuple) else k): AdaptiveTuner._make_json_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [AdaptiveTuner._make_json_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj

    def _save_state(self, state: Dict):
        state["compute_used_minutes"] = (time.time() - self.t_start) / 60.0
        self.state_path.write_text(
            json.dumps(self._make_json_serializable(state), indent=2, default=str),
            encoding="utf-8",
        )

    # ────────────────────────────────────────────────────────────────────────
    #  Stage 1: 1D coarse scans
    # ────────────────────────────────────────────────────────────────────────

    def stage_1_grids(self) -> Dict[str, List]:
        """Define 1D scan grids per axis."""
        if self.method == "bennett":
            return {
                "lambda_h": [1e-6, 1e-5, 3e-5, 1e-4, 1e-3],
                "gamma_critic": [1e-5, 1e-4, 1e-3, 1e-2],
                "ell_h": [1.5, 2.0, 2.75, 3.5, 5.0],
                "lambda_r": [1e-3, 1e-2, 1e-1],
                "ell_scale_r": [2.5, 3.5, 5.0],
            }
        else:
            return {
                "lambda_h": [1e-6, 1e-5, 3e-5, 1e-4, 1e-3],
                "ell_scale": [1.5, 2.5, 3.5, 5.0],
                "lambda_Q": [1e-5, 1e-4, 1e-3, 1e-2],
                "clip_factor": [None, 5.0, 3.0, 2.0],
            }

    def run_stage_1(self) -> Dict[str, Any]:
        """1D scans on each axis (one at a time, others at default)."""
        print(f"\n=== STAGE 1: 1D coarse scans for {self.method} ===")
        grids = self.stage_1_grids()
        defaults = DEFAULTS[self.method]
        per_axis_results: Dict[str, List[Dict]] = {}

        for axis, values in grids.items():
            print(f"\n  [{self.method}] Axis: {axis}")
            axis_results = []
            for i, v in enumerate(values):
                config = dict(defaults)
                config[axis] = v
                label = f"S1_{axis}_{i}"
                seed_base = self.seed_tune + hash(label) % 100_000
                t0 = time.time()
                try:
                    res = _run_one_config(
                        self.method, config,
                        n=1000, snr_W=0.95, snr_Z=0.95,
                        M=self.M_explore, seed_base=seed_base,
                        raw_dir=self.raw_dir / "stage1",
                        label=label,
                    )
                    score = res["score"]
                    elapsed = time.time() - t0
                    print(f"    {axis}={v}  score={score:.3f}  [{elapsed:.0f}s]")
                    axis_results.append(res)
                except Exception as e:
                    print(f"    ERROR {axis}={v}: {e}")
                    axis_results.append({"label": label, "error": str(e), "config": config})
            per_axis_results[axis] = axis_results

        # Find best per axis
        top_per_axis = {}
        for axis, results in per_axis_results.items():
            valid = [r for r in results if "score" in r and np.isfinite(r["score"])]
            valid.sort(key=lambda r: r["score"])
            top_per_axis[axis] = valid[:2] if valid else []

        # Build a "merged best" config from each axis's #1
        merged = dict(DEFAULTS[self.method])
        for axis, top_list in top_per_axis.items():
            if top_list:
                best_v = top_list[0]["config"][axis]
                merged[axis] = best_v

        return {
            "stage": 1,
            "per_axis_results": per_axis_results,
            "top_per_axis": top_per_axis,
            "merged_best": merged,
            "n_configs": sum(len(r) for r in per_axis_results.values()),
        }

    # ────────────────────────────────────────────────────────────────────────
    #  Stage 2: 2D refinement around merged best
    # ────────────────────────────────────────────────────────────────────────

    def stage_2_grids(self, stage1_result: Dict) -> List[Dict]:
        """Build 2D refinement grid around Stage 1 best."""
        merged = stage1_result["merged_best"]
        top = stage1_result["top_per_axis"]

        if self.method == "bennett":
            # Cross top-2 of lambda_h × top-2 of gamma_critic (most impactful axes)
            lambda_h_vals = [r["config"]["lambda_h"] for r in top.get("lambda_h", [])][:2]
            gamma_vals = [r["config"]["gamma_critic"] for r in top.get("gamma_critic", [])][:2]
            ell_vals = [r["config"]["ell_h"] for r in top.get("ell_h", [])][:2]
            if not lambda_h_vals: lambda_h_vals = [merged["lambda_h"]]
            if not gamma_vals: gamma_vals = [merged["gamma_critic"]]
            if not ell_vals: ell_vals = [merged["ell_h"]]
            configs = []
            for lh, gc, eh in product(lambda_h_vals, gamma_vals, ell_vals):
                cfg = dict(merged)
                cfg["lambda_h"] = lh
                cfg["gamma_critic"] = gc
                cfg["ell_h"] = eh
                configs.append(cfg)
            return configs
        else:
            lambda_h_vals = [r["config"]["lambda_h"] for r in top.get("lambda_h", [])][:2]
            ell_vals = [r["config"]["ell_scale"] for r in top.get("ell_scale", [])][:2]
            lQ_vals = [r["config"]["lambda_Q"] for r in top.get("lambda_Q", [])][:2]
            if not lambda_h_vals: lambda_h_vals = [merged["lambda_h"]]
            if not ell_vals: ell_vals = [merged["ell_scale"]]
            if not lQ_vals: lQ_vals = [merged["lambda_Q"]]
            configs = []
            for lh, esc, lQ in product(lambda_h_vals, ell_vals, lQ_vals):
                cfg = dict(merged)
                cfg["lambda_h"] = lh
                cfg["ell_scale"] = esc
                cfg["lambda_Q"] = lQ
                configs.append(cfg)
            return configs

    def run_stage_2(self, stage1_result: Dict) -> Dict[str, Any]:
        print(f"\n=== STAGE 2: 2D refinement for {self.method} ===")
        configs = self.stage_2_grids(stage1_result)
        results = []
        for i, config in enumerate(configs):
            label = f"S2_c{i}"
            seed_base = self.seed_tune + 100_000 + hash(label) % 100_000
            t0 = time.time()
            try:
                res = _run_one_config(
                    self.method, config,
                    n=1000, snr_W=0.95, snr_Z=0.95,
                    M=max(self.M_explore, 50) if self.mode == "full" else self.M_explore,
                    seed_base=seed_base,
                    raw_dir=self.raw_dir / "stage2",
                    label=label,
                )
                elapsed = time.time() - t0
                print(f"  cfg{i} score={res['score']:.3f}  [{elapsed:.0f}s]")
                results.append(res)
            except Exception as e:
                print(f"  cfg{i} ERROR: {e}")
                results.append({"label": label, "error": str(e), "config": config})

        valid = [r for r in results if "score" in r and np.isfinite(r["score"])]
        valid.sort(key=lambda r: r["score"])
        top3 = valid[:3]
        return {
            "stage": 2,
            "all_results": results,
            "top3": top3,
            "n_configs": len(configs),
        }

    # ────────────────────────────────────────────────────────────────────────
    #  Stage 3: Cross-cell consistency
    # ────────────────────────────────────────────────────────────────────────

    def run_stage_3(self, stage2_result: Dict) -> Dict[str, Any]:
        print(f"\n=== STAGE 3: Cross-cell consistency for {self.method} ===")
        top3 = stage2_result["top3"]
        if not top3:
            return {"stage": 3, "error": "No top-3 from Stage 2"}

        cells = [
            (1000, 0.95, 0.95),
            (2000, 0.90, 0.90),
            (2000, 0.85, 0.85),
        ]
        results_per_config = {}
        for ci, cfg_data in enumerate(top3):
            cfg_results = {}
            for n, snr_W, snr_Z in cells:
                label = f"S3_c{ci}_n{n}_s{int(snr_W*100):02d}"
                seed_base = self.seed_tune + 200_000 + ci * 10_000 + n
                t0 = time.time()
                try:
                    res = _run_one_config(
                        self.method, cfg_data["config"],
                        n=n, snr_W=snr_W, snr_Z=snr_Z,
                        M=max(self.M_explore, 50) if self.mode == "full" else self.M_explore,
                        seed_base=seed_base,
                        raw_dir=self.raw_dir / "stage3",
                        label=label,
                    )
                    elapsed = time.time() - t0
                    print(f"  cfg{ci} n={n} SNR={snr_W}: score={res['score']:.3f}  [{elapsed:.0f}s]")
                    cfg_results[(n, snr_W, snr_Z)] = res
                except Exception as e:
                    print(f"  cfg{ci} n={n} ERROR: {e}")
                    cfg_results[(n, snr_W, snr_Z)] = {"error": str(e)}
            results_per_config[ci] = cfg_results

        # Aggregate: mean score across cells per config
        cfg_means = []
        for ci, cell_results in results_per_config.items():
            scores = [r["score"] for r in cell_results.values()
                      if isinstance(r, dict) and "score" in r and np.isfinite(r["score"])]
            mean_score = float(np.mean(scores)) if scores else float("inf")
            cfg_means.append({
                "ci": ci,
                "config": top3[ci]["config"],
                "mean_score": mean_score,
                "per_cell": cell_results,
            })
        cfg_means.sort(key=lambda x: x["mean_score"])
        return {
            "stage": 3,
            "results_per_config": results_per_config,
            "cfg_means": cfg_means,
            "top2": cfg_means[:2],
        }

    # ────────────────────────────────────────────────────────────────────────
    #  Stage 4: Disjoint validation (M=100, anti-winner-evaporation)
    # ────────────────────────────────────────────────────────────────────────

    def run_stage_4(self, stage3_result: Dict) -> Dict[str, Any]:
        print(f"\n=== STAGE 4: Disjoint validation (M={self.M_validate}) for {self.method} ===")
        top2 = stage3_result.get("top2", [])
        if not top2:
            return {"stage": 4, "error": "No top-2 from Stage 3"}

        cells = [
            (1000, 0.95, 0.95),
            (2000, 0.90, 0.90),
            (2000, 0.85, 0.85),
        ]
        results_per_config = {}
        for ci, cfg_data in enumerate(top2):
            cfg_results = {}
            for n, snr_W, snr_Z in cells:
                label = f"S4_c{ci}_n{n}_s{int(snr_W*100):02d}"
                # IMPORTANT: disjoint seed_base from Stage 3 (+500K)
                seed_base = self.seed_valid + ci * 10_000 + n
                t0 = time.time()
                try:
                    res = _run_one_config(
                        self.method, cfg_data["config"],
                        n=n, snr_W=snr_W, snr_Z=snr_Z,
                        M=self.M_validate, seed_base=seed_base,
                        raw_dir=self.raw_dir / "stage4",
                        label=label,
                    )
                    elapsed = time.time() - t0
                    print(f"  cfg{ci} n={n} SNR={snr_W}: score={res['score']:.3f}  "
                          f"SER={res['diags'].get('SER', float('nan')):.3f}  [{elapsed:.0f}s]")
                    cfg_results[(n, snr_W, snr_Z)] = res
                except Exception as e:
                    print(f"  cfg{ci} n={n} ERROR: {e}")
                    cfg_results[(n, snr_W, snr_Z)] = {"error": str(e)}
            results_per_config[ci] = cfg_results

        # Score each on disjoint validation
        cfg_scores = []
        for ci, cell_results in results_per_config.items():
            scores = [r["score"] for r in cell_results.values()
                      if isinstance(r, dict) and "score" in r and np.isfinite(r["score"])]
            mean_score = float(np.mean(scores)) if scores else float("inf")
            sers = [r["diags"].get("SER", float("nan")) for r in cell_results.values()
                    if isinstance(r, dict) and "diags" in r]
            ser_in_range = sum(1 for s in sers if np.isfinite(s) and 0.85 <= s <= 1.15)
            cfg_scores.append({
                "ci": ci,
                "config": top2[ci]["config"],
                "mean_score": mean_score,
                "ser_in_range_count": ser_in_range,
                "per_cell": cell_results,
            })

        # Winner: lowest score on disjoint, with SER preference
        cfg_scores.sort(key=lambda x: (-x["ser_in_range_count"], x["mean_score"]))
        winner = cfg_scores[0] if cfg_scores else None

        # Check for winner-evaporation
        s3_top1 = stage3_result["top2"][0]["config"] if top2 else None
        s4_top1 = winner["config"] if winner else None
        evaporation = (s3_top1 != s4_top1) if (s3_top1 and s4_top1) else None

        return {
            "stage": 4,
            "results_per_config": results_per_config,
            "cfg_scores": cfg_scores,
            "winner": winner,
            "winner_evaporation": evaporation,
        }

    # ────────────────────────────────────────────────────────────────────────
    #  Stage 5: Regime robustness
    # ────────────────────────────────────────────────────────────────────────

    def run_stage_5(self, stage4_result: Dict) -> Dict[str, Any]:
        print(f"\n=== STAGE 5: Regime robustness for {self.method} ===")
        winner = stage4_result.get("winner")
        if not winner:
            return {"stage": 5, "error": "No winner from Stage 4"}

        cells = [
            (1000, 0.80, 0.80),     # low SNR
            (1000, 0.95, 0.50),     # weak Z
            (1000, 0.50, 0.95),     # weak W
            (2000, 0.95, 0.95),     # strong reference
        ]
        results = {}
        for n, snr_W, snr_Z in cells:
            label = f"S5_n{n}_W{int(snr_W*100):02d}_Z{int(snr_Z*100):02d}"
            seed_base = self.seed_robust + n + int(snr_W * 100) * 10
            t0 = time.time()
            try:
                res = _run_one_config(
                    self.method, winner["config"],
                    n=n, snr_W=snr_W, snr_Z=snr_Z,
                    M=self.M_validate, seed_base=seed_base,
                    raw_dir=self.raw_dir / "stage5",
                    label=label,
                )
                elapsed = time.time() - t0
                ser = res["diags"].get("SER", float("nan"))
                cov = res["diags"].get("_cov_ref_NONBLIND", float("nan"))
                print(f"  n={n} W={snr_W} Z={snr_Z}: score={res['score']:.3f}  "
                      f"SER={ser:.3f}  cov={cov:.3f}  [{elapsed:.0f}s]")
                results[(n, snr_W, snr_Z)] = res
            except Exception as e:
                print(f"  n={n} ERROR: {e}")
                results[(n, snr_W, snr_Z)] = {"error": str(e)}

        # Final SER gate
        sers = [r["diags"].get("SER", float("nan")) for r in results.values()
                if isinstance(r, dict) and "diags" in r]
        ser_passed = sum(1 for s in sers if np.isfinite(s) and 0.85 <= s <= 1.15)
        envelope = "ROBUST" if ser_passed >= 3 else "REGIME-CONDITIONAL"

        return {
            "stage": 5,
            "results": results,
            "ser_passed_count": ser_passed,
            "ser_total_count": len(sers),
            "envelope": envelope,
            "winner": winner,
        }

    # ────────────────────────────────────────────────────────────────────────
    #  Per-stage report writing
    # ────────────────────────────────────────────────────────────────────────

    def _write_stage_report(self, stage: int, result: Dict):
        path = self.summ_dir / f"phase17_{self.method}_stage{stage}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        import datetime
        lines = []
        ap = lines.append
        ap(f"# Phase 17 — {self.method} Stage {stage}")
        ap(f"*Generated {datetime.datetime.now().isoformat()}*")
        ap("")

        if stage == 1:
            ap(f"## Stage 1 — 1D coarse scans")
            ap(f"Configs tested: {result.get('n_configs', '?')}")
            ap("")
            ap("### Top per axis")
            for axis, top_list in result["top_per_axis"].items():
                ap(f"\n**{axis}**")
                for r in top_list:
                    ap(f"- value={r['config'][axis]} score={r['score']:.3f}")
            ap("")
            ap("### Merged best (initial Stage 2 starting point)")
            for k, v in result["merged_best"].items():
                ap(f"- {k}: {v}")
        elif stage == 2:
            ap(f"## Stage 2 — 2D refinement")
            ap(f"Configs tested: {result.get('n_configs', '?')}")
            ap("")
            ap("### Top-3 candidates")
            for i, r in enumerate(result.get("top3", [])):
                ap(f"\n**Top {i+1}** (score={r['score']:.4f})")
                for k, v in r["config"].items():
                    ap(f"- {k}: {v}")
        elif stage == 3:
            ap(f"## Stage 3 — Cross-cell consistency")
            ap("\n### Cross-cell mean scores")
            for r in result.get("cfg_means", []):
                ap(f"- ci={r['ci']} mean_score={r['mean_score']:.4f}")
            ap("\n### Top-2 stable")
            for i, r in enumerate(result.get("top2", [])):
                ap(f"**Top {i+1}** mean_score={r['mean_score']:.4f}")
                for k, v in r["config"].items():
                    ap(f"- {k}: {v}")
        elif stage == 4:
            ap(f"## Stage 4 — Disjoint validation (M={self.M_validate})")
            ap(f"Winner-evaporation: {result.get('winner_evaporation')}")
            ap("\n### Final ranking")
            for r in result.get("cfg_scores", []):
                ap(f"- ci={r['ci']} mean_score={r['mean_score']:.4f}  "
                   f"SER_in_range={r['ser_in_range_count']}/3")
            ap("")
            winner = result.get("winner")
            if winner:
                ap("### Winner")
                for k, v in winner["config"].items():
                    ap(f"- {k}: {v}")
        elif stage == 5:
            ap(f"## Stage 5 — Regime robustness")
            ap(f"SER passed: {result.get('ser_passed_count', 0)}/{result.get('ser_total_count', 0)}")
            ap(f"Envelope: **{result.get('envelope', 'UNKNOWN')}**")
            ap("\n### Per-cell results")
            for cell, r in result.get("results", {}).items():
                if isinstance(r, dict) and "diags" in r:
                    d = r["diags"]
                    ap(f"- {cell}: score={r['score']:.3f}  SER={d.get('SER', float('nan')):.3f}  "
                       f"cov={d.get('_cov_ref_NONBLIND', float('nan')):.3f}")
                else:
                    ap(f"- {cell}: ERROR")

        path.write_text("\n".join(lines), encoding="utf-8")

    # ────────────────────────────────────────────────────────────────────────
    #  Main loop
    # ────────────────────────────────────────────────────────────────────────

    def run(self):
        state = self._load_state()
        print(f"\n{'='*70}")
        print(f"Phase 17 Adaptive Tuner: method={self.method} mode={self.mode}")
        print(f"Starting from stage {self.start_stage}, stopping after {self.stop_stage}")
        print(f"{'='*70}")

        cached: Dict[int, Dict] = {}

        # Stage 1
        if self.start_stage <= 1 <= self.stop_stage:
            t0 = time.time()
            stage1 = self.run_stage_1()
            print(f"\nStage 1 done in {(time.time()-t0)/60:.1f} min")
            cached[1] = stage1
            state["stage_history"].append({"stage": 1, "completed_at": time.time()})
            self._write_stage_report(1, stage1)
            self._save_state(state)

        # Stage 2
        if self.start_stage <= 2 <= self.stop_stage:
            t0 = time.time()
            stage2 = self.run_stage_2(cached.get(1, {}))
            print(f"\nStage 2 done in {(time.time()-t0)/60:.1f} min")
            cached[2] = stage2
            state["stage_history"].append({"stage": 2, "completed_at": time.time()})
            self._write_stage_report(2, stage2)
            self._save_state(state)

        # Stage 3
        if self.start_stage <= 3 <= self.stop_stage:
            t0 = time.time()
            stage3 = self.run_stage_3(cached.get(2, {}))
            print(f"\nStage 3 done in {(time.time()-t0)/60:.1f} min")
            cached[3] = stage3
            state["stage_history"].append({"stage": 3, "completed_at": time.time()})
            self._write_stage_report(3, stage3)
            self._save_state(state)

        # Stage 4
        if self.start_stage <= 4 <= self.stop_stage:
            t0 = time.time()
            stage4 = self.run_stage_4(cached.get(3, {}))
            print(f"\nStage 4 done in {(time.time()-t0)/60:.1f} min")
            cached[4] = stage4
            state["stage_history"].append({"stage": 4, "completed_at": time.time()})
            self._write_stage_report(4, stage4)
            self._save_state(state)

        # Stage 5
        if self.start_stage <= 5 <= self.stop_stage:
            t0 = time.time()
            stage5 = self.run_stage_5(cached.get(4, {}))
            print(f"\nStage 5 done in {(time.time()-t0)/60:.1f} min")
            cached[5] = stage5
            state["stage_history"].append({"stage": 5, "completed_at": time.time()})
            self._write_stage_report(5, stage5)

        # Final winner declaration
        if 5 in cached:
            state["winner"] = cached[5].get("winner")
            state["envelope"] = cached[5].get("envelope")
        elif 4 in cached:
            state["winner"] = cached[4].get("winner")
        state["converged"] = True
        self._save_state(state)

        # Final summary
        self._write_final_report(cached, state)

        elapsed_min = (time.time() - self.t_start) / 60
        print(f"\n{'='*70}")
        print(f"Phase 17 {self.method} COMPLETE in {elapsed_min:.1f} min")
        print(f"Winner: {state.get('winner', {}).get('config', 'NONE') if state.get('winner') else 'NONE'}")
        print(f"Envelope: {state.get('envelope', 'unknown')}")
        print(f"{'='*70}")
        return cached

    def _write_final_report(self, cached: Dict, state: Dict):
        path = self.summ_dir / f"phase17_{self.method}_FINAL.md"
        import datetime
        lines = []
        ap = lines.append
        ap(f"# Phase 17 — {self.method} BLIND TUNING FINAL")
        ap(f"*Generated {datetime.datetime.now().isoformat()}*")
        ap(f"*Mode: {self.mode}, Total compute: {state['compute_used_minutes']:.1f} min*")
        ap("")
        ap("## Winner spec")
        winner = state.get("winner")
        if winner:
            ap("```json")
            ap(json.dumps(winner.get("config", {}), indent=2, default=str))
            ap("```")
            ap("")
            if "mean_score" in winner:
                ap(f"Composite blind score (mean across cells): **{winner['mean_score']:.4f}**")
            ap(f"Envelope: **{state.get('envelope', 'N/A')}**")
        else:
            ap("No winner declared.")
        ap("")
        ap("## Stage history")
        for h in state["stage_history"]:
            ap(f"- Stage {h['stage']} completed at {h.get('completed_at', '?')}")
        ap("")
        ap("## Comparison to baseline")
        if self.method == "bennett":
            ap("Baseline: Phase 14B BIN_mh2000 specs (m_h=2000, ell_h=2.75, lambda_h=1e-5,")
            ap("gamma_critic=1e-4, lambda_r=1e-2, n_features_r=500, ell_scale_r=3.5).")
            ap("**OFFICIAL BASELINE NOT REPLACED** -- this is exploratory blind validation.")
        else:
            ap("Baseline: Phase 9B defaults (lambda_1=lambda_2=3e-5, ell_scale=3.5, lambda_Q=1e-3).")
            ap("**This is the first proper tuning of DRKPV.**")
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Final report: {path}")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bennett", "drkpv"], required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--start_stage", type=int, default=1)
    parser.add_argument("--stop_stage", type=int, default=5)
    parser.add_argument("--resume", action="store_true",
                         help="Resume from saved state (use --start_stage to specify)")
    args = parser.parse_args()

    tuner = AdaptiveTuner(
        method=args.method,
        mode=args.mode,
        start_stage=args.start_stage,
        stop_stage=args.stop_stage,
    )
    tuner.run()


if __name__ == "__main__":
    main()
