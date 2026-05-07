"""
Phase 18 -- Adaptive Blind Tuning Controller (Bennett corrected)

Methodologically cleaner version of Phase 17 that fixes the SER collapse
failure caused by the interpolation regime (lambda_h too small → h-bridge
overfits → residual_norm_h → 0 → blind score paradox at n=2000).

Changes vs Phase 17:
  1. bennett_blind_score_v2: NEW gate lambda_h × n_fold < 0.01 → hard fail
  2. bennett_blind_score_v2: kappa_h gate lowered 1e6→1e5 (penalty starts earlier)
  3. Stage 1: each 1D scan run at TWO cells (n=1000 AND n=2000); score = mean
  4. Stage 4: adds (n=2000, SNR=0.95) as 4th cell (the reference cell missed in Phase 17)
  5. Seed bases: 62M/62.5M/63M (disjoint from Phase 17: 60M/60.5M/61M)
  6. Grid: lambda_h starts at 1e-5 (1e-6 removed — always hard-fails gate at any n)

Scientific motivation:
  Phase 17 found lambda_h=1e-6 as blind winner with SER=0.312 at n=2000.
  Root cause: with K=5 folds, n=2000 → n_fold=1600 < m_h=2000.
  lambda_h × n_fold = 0.0016 < 0.01 → underdetermined system → interpolation.
  residual_norm_h → 0 (5.85e-5) → blind score=0.299 (paradoxically best).
  BIN_mh2000 has lambda_h=1e-5 → lambda_h × n_fold = 0.016 > 0.01 ✓.

Usage:
    python -u -m simulations.experiments.phase18_blind_tuning \\
        --method bennett --mode smoke    # ~5 min sanity check
    python -u -m simulations.experiments.phase18_blind_tuning \\
        --method bennett --mode full     # ~8h background run

Note: Phase 17 DRKPV was methodologically sound (no SER collapse). There is
no Phase 18 DRKPV — await Phase 17 DRKPV completion instead.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[3]  # +1: moved to experiments/simulation/
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "s7" / "blind_tuning_bennett"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
REF_IDX = 2

# Scoring functions extracted to pci.tuning.blind_score.
# Re-exported here for backward compatibility.
from pci.tuning.blind_score import (  # noqa: F401
    bennett_blind_score_v2,
    drkpv_blind_score,
    SCORING,
)


# ════════════════════════════════════════════════════════════════════════════
#  Diagnostic extraction from per-rep records
# ════════════════════════════════════════════════════════════════════════════

# extract_diagnostics + helpers extracted to pci.tuning.extract_diagnostics
#. Re-exported with the local REF_IDX as default ref_idx.
from pci.tuning.extract_diagnostics import (  # noqa: F401
    _safe,
    _array_safe,
)
from pci.tuning.extract_diagnostics import extract_diagnostics as _extract_diagnostics_canonical


def extract_diagnostics(data: dict, ref_idx: int = REF_IDX) -> Dict[str, float]:
    """Backward-compat wrapper: passes module-level REF_IDX as default."""
    return _extract_diagnostics_canonical(data, ref_idx=ref_idx)


# ════════════════════════════════════════════════════════════════════════════
#  Estimator factories
# ════════════════════════════════════════════════════════════════════════════

# Estimator factories + DEFAULTS extracted to pci.tuning.factories.
# Re-exported here. Local wrappers preserve the historical script behavior of
# using A_GRID and REF_IDX from this module's globals.
from pci.tuning.factories import (  # noqa: F401
    BENNETT_DEFAULTS,
    DRKPV_DEFAULTS,
    DEFAULTS,
)
from pci.tuning.factories import (
    make_bennett_estimator as _make_bennett_canonical,
    make_drkpv_estimator as _make_drkpv_canonical,
)


def make_bennett_estimator(config: Dict, dgp, h_KDE: float,
                            ell_W: float, ell_A: float, ell_Z: float):
    """Backward-compat wrapper: passes module-level A_GRID and REF_IDX."""
    return _make_bennett_canonical(
        config, dgp, h_KDE, ell_W, ell_A, ell_Z,
        a_grid=A_GRID.tolist(), ref_dose_index=REF_IDX,
    )


def make_drkpv_estimator(config: Dict, dgp, h_KDE: float,
                          ell_W: float, ell_A: float, ell_Z: float):
    """Backward-compat wrapper: passes module-level A_GRID and REF_IDX."""
    return _make_drkpv_canonical(
        config, dgp, h_KDE, ell_W, ell_A, ell_Z,
        a_grid=A_GRID, ref_dose_index=REF_IDX,
    )


FACTORIES = {"bennett": make_bennett_estimator, "drkpv": make_drkpv_estimator}


# AdaptiveTuner extracted to pci.tuning.adaptive_tuner_bennett.
# The local AdaptiveTuner class below remains for backward compatibility (it
# also handles method="drkpv" via Phase 18 score, used in legacy comparisons).
# New code should import BennettAdaptiveTuner directly from pci.tuning.
from pci.tuning.adaptive_tuner_bennett import BennettAdaptiveTuner  # noqa: F401


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
    """
    Run M reps for a single config and return diagnostics + score.

    Phase 18 change: injects config['lambda_h'] into diags before scoring,
    so that bennett_blind_score_v2 can apply the lambda_h × n gate.
    """
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

    # Inject lambda_h and m_h into diags for bennett_blind_score_v2 gate (Phase 18)
    # Gate: lambda_h x m_h < 0.01 -> inf (interpolation protection)
    if method == "bennett":
        if "lambda_h" in config:
            diags["lambda_h"] = float(config["lambda_h"])
        if "m_h" in config:
            diags["m_h"] = float(config["m_h"])

    # Score with n passed for the interpolation gate
    score = SCORING[method](diags, n=n)
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
    """5-stage adaptive blind tuner (Phase 18 corrected for Bennett SER collapse)."""

    def __init__(self, method: str, mode: str = "full",
                  start_stage: int = 1, stop_stage: int = 5):
        if method not in ("bennett", "drkpv"):
            raise ValueError(f"Unknown method {method!r}")
        self.method = method
        self.mode = mode
        self.start_stage = start_stage
        self.stop_stage = stop_stage

        # Seed bases (disjoint from Phase 17: 60M/60.5M/61M Bennett, 70M/70.5M/71M DRKPV)
        if method == "bennett":
            self.seed_tune  = 62_000_000   # Stage 1-3 tuning
            self.seed_valid = 62_500_000   # Stage 4 validation
            self.seed_robust= 63_000_000   # Stage 5 robustness
        else:
            self.seed_tune  = 72_000_000
            self.seed_valid = 72_500_000
            self.seed_robust= 73_000_000

        self.raw_dir = _RAW_DIR / method
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.summ_dir = _SUMM_DIR
        self.state_path = self.raw_dir / "state.json"

        if mode == "smoke":
            self.M_explore  = 3
            self.M_validate = 5
        else:
            self.M_explore  = 30
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
    #  Stage 1: 1D coarse scans at TWO cells (n=1000 AND n=2000)  [PHASE 18 CHANGE]
    # ────────────────────────────────────────────────────────────────────────

    def stage_1_grids(self) -> Dict[str, List]:
        """Define 1D scan grids per axis.

        Phase 18 Bennett change: lambda_h grid starts at 1e-5 (1e-6 removed —
        it hard-fails the lambda_h × n_fold gate at all cells with n >= 1000).
        """
        if self.method == "bennett":
            return {
                "lambda_h":    [1e-5, 3e-5, 1e-4, 1e-3],   # removed 1e-6 (always fails gate)
                "gamma_critic":[1e-5, 1e-4, 1e-3, 1e-2],
                "ell_h":       [1.5, 2.0, 2.75, 3.5, 5.0],
                "lambda_r":    [1e-3, 1e-2, 1e-1],
                "ell_scale_r": [2.5, 3.5, 5.0],
            }
        else:
            return {
                "lambda_h":    [1e-6, 1e-5, 3e-5, 1e-4, 1e-3],
                "ell_scale":   [1.5, 2.5, 3.5, 5.0],
                "lambda_Q":    [1e-5, 1e-4, 1e-3, 1e-2],
                "clip_factor": [None, 5.0, 3.0, 2.0],
            }

    # Stage 1 cells: (n=1000, SNR=0.95) + (n=2000, SNR=0.95) [PHASE 18 NEW]
    STAGE1_CELLS = [(1000, 0.95, 0.95), (2000, 0.95, 0.95)]

    def run_stage_1(self) -> Dict[str, Any]:
        """
        1D scans on each axis, evaluated at BOTH n=1000 and n=2000.

        Phase 18 change: score = mean across both cells. A config that
        hard-fails at n=2000 (lambda_h × n_fold < 0.01) gets score=inf
        and is excluded immediately.
        """
        print(f"\n=== STAGE 1: 1D coarse scans for {self.method} ===")
        print(f"  Cells: {self.STAGE1_CELLS} [Phase 18: 2-cell sweep]")
        grids = self.stage_1_grids()
        defaults = DEFAULTS[self.method]
        per_axis_results: Dict[str, List[Dict]] = {}

        for axis, values in grids.items():
            print(f"\n  [{self.method}] Axis: {axis}")
            axis_results = []
            for i, v in enumerate(values):
                config = dict(defaults)
                config[axis] = v
                label_base = f"S1_{axis}_{i}"
                seed_base = self.seed_tune + hash(label_base) % 100_000
                t0 = time.time()

                # Run at each Stage 1 cell, collect scores
                cell_scores = []
                cell_ok = True
                for n_cell, snr_W_cell, snr_Z_cell in self.STAGE1_CELLS:
                    label_cell = f"{label_base}_n{n_cell}"
                    try:
                        res_cell = _run_one_config(
                            self.method, config,
                            n=n_cell, snr_W=snr_W_cell, snr_Z=snr_Z_cell,
                            M=self.M_explore, seed_base=seed_base,
                            raw_dir=self.raw_dir / "stage1",
                            label=label_cell,
                        )
                        cell_scores.append(res_cell["score"])
                    except Exception as e:
                        print(f"    ERROR {axis}={v} n={n_cell}: {e}")
                        cell_scores.append(float("inf"))
                        cell_ok = False

                # Combined score = mean; inf propagates (hard fail at any cell → excluded)
                if any(not np.isfinite(s) for s in cell_scores):
                    combined_score = float("inf")
                else:
                    combined_score = float(np.mean(cell_scores))

                elapsed = time.time() - t0
                score_str = f"{combined_score:.3f}" if np.isfinite(combined_score) else "inf"
                cell_scores_str = " / ".join(
                    f"n{c[0]}={s:.3f}" if np.isfinite(s) else f"n{c[0]}=inf"
                    for c, s in zip(self.STAGE1_CELLS, cell_scores)
                )
                print(f"    {axis}={v}  score={score_str}  [{cell_scores_str}]  [{elapsed:.0f}s]")

                # Store as a single result with combined score
                axis_results.append({
                    "label": label_base,
                    "config": config,
                    "score": combined_score,
                    "cell_scores": {
                        f"n{c[0]}": s for c, s in zip(self.STAGE1_CELLS, cell_scores)
                    },
                })
            per_axis_results[axis] = axis_results

        # Find best per axis
        top_per_axis = {}
        for axis, results in per_axis_results.items():
            valid = [r for r in results if "score" in r and np.isfinite(r["score"])]
            valid.sort(key=lambda r: r["score"])
            top_per_axis[axis] = valid[:2] if valid else []
            if not valid:
                print(f"  WARNING: no valid config for axis {axis} in Stage 1")

        # Build "merged best" config from each axis's #1
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
            lambda_h_vals  = [r["config"]["lambda_h"]     for r in top.get("lambda_h", [])][:2]
            gamma_vals     = [r["config"]["gamma_critic"]  for r in top.get("gamma_critic", [])][:2]
            ell_vals       = [r["config"]["ell_h"]         for r in top.get("ell_h", [])][:2]
            if not lambda_h_vals: lambda_h_vals = [merged["lambda_h"]]
            if not gamma_vals:    gamma_vals    = [merged["gamma_critic"]]
            if not ell_vals:      ell_vals      = [merged["ell_h"]]
            configs = []
            for lh, gc, eh in product(lambda_h_vals, gamma_vals, ell_vals):
                cfg = dict(merged)
                cfg["lambda_h"]     = lh
                cfg["gamma_critic"] = gc
                cfg["ell_h"]        = eh
                configs.append(cfg)
            return configs
        else:
            lambda_h_vals = [r["config"]["lambda_h"]  for r in top.get("lambda_h", [])][:2]
            ell_vals      = [r["config"]["ell_scale"]  for r in top.get("ell_scale", [])][:2]
            lQ_vals       = [r["config"]["lambda_Q"]   for r in top.get("lambda_Q", [])][:2]
            if not lambda_h_vals: lambda_h_vals = [merged["lambda_h"]]
            if not ell_vals:      ell_vals      = [merged["ell_scale"]]
            if not lQ_vals:       lQ_vals       = [merged["lambda_Q"]]
            configs = []
            for lh, esc, lQ in product(lambda_h_vals, ell_vals, lQ_vals):
                cfg = dict(merged)
                cfg["lambda_h"]  = lh
                cfg["ell_scale"] = esc
                cfg["lambda_Q"]  = lQ
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
    #  Stage 4: Disjoint validation (M=100) [PHASE 18 CHANGE: 4 cells, +n=2000/0.95]
    # ────────────────────────────────────────────────────────────────────────

    def run_stage_4(self, stage3_result: Dict) -> Dict[str, Any]:
        print(f"\n=== STAGE 4: Disjoint validation (M={self.M_validate}) for {self.method} ===")
        top2 = stage3_result.get("top2", [])
        if not top2:
            return {"stage": 4, "error": "No top-2 from Stage 3"}

        # Phase 18 change: added (2000, 0.95, 0.95) — the reference cell that
        # exposed the SER collapse in Phase 17 (was absent from Phase 17 Stage 4)
        cells = [
            (1000, 0.95, 0.95),
            (2000, 0.95, 0.95),   # ← NEW: reference cell (Phase 17 missed this)
            (2000, 0.90, 0.90),
            (2000, 0.85, 0.85),
        ]
        n_cells = len(cells)

        results_per_config = {}
        for ci, cfg_data in enumerate(top2):
            cfg_results = {}
            for n, snr_W, snr_Z in cells:
                label = f"S4_c{ci}_n{n}_s{int(snr_W*100):02d}_W{int(snr_W*100):02d}Z{int(snr_Z*100):02d}"
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
                    ser = res["diags"].get("SER", float("nan"))
                    ser_str = f"{ser:.3f}" if np.isfinite(ser) else "nan"
                    ser_gate = "[OK]" if np.isfinite(ser) and 0.85 <= ser <= 1.15 else "[FAIL]"
                    print(f"  cfg{ci} n={n} W={snr_W} Z={snr_Z}: score={res['score']:.3f}  "
                          f"SER={ser_str} {ser_gate}  [{elapsed:.0f}s]")
                    cfg_results[(n, snr_W, snr_Z)] = res
                except Exception as e:
                    print(f"  cfg{ci} n={n} ERROR: {e}")
                    cfg_results[(n, snr_W, snr_Z)] = {"error": str(e)}
            results_per_config[ci] = cfg_results

        cfg_scores = []
        for ci, cell_results in results_per_config.items():
            scores = [r["score"] for r in cell_results.values()
                      if isinstance(r, dict) and "score" in r and np.isfinite(r["score"])]
            mean_score = float(np.mean(scores)) if scores else float("inf")
            sers = [r["diags"].get("SER", float("nan")) for r in cell_results.values()
                    if isinstance(r, dict) and "diags" in r]
            ser_in_range = sum(1 for s in sers if np.isfinite(s) and 0.85 <= s <= 1.15)

            # Flag if SER fails at n=2000/SNR=0.95 specifically
            ref_cell_key = (2000, 0.95, 0.95)
            ref_res = cell_results.get(ref_cell_key, {})
            ref_ser = ref_res.get("diags", {}).get("SER", float("nan")) if isinstance(ref_res, dict) else float("nan")
            ref_ser_ok = np.isfinite(ref_ser) and 0.85 <= ref_ser <= 1.15

            cfg_scores.append({
                "ci": ci,
                "config": top2[ci]["config"],
                "mean_score": mean_score,
                "ser_in_range_count": ser_in_range,
                "ser_total": n_cells,
                "ref_cell_ser": float(ref_ser),
                "ref_cell_ser_ok": ref_ser_ok,
                "per_cell": cell_results,
            })
            if not ref_ser_ok:
                print(f"  [WARN] cfg{ci}: SER at reference cell (n=2000/SNR=0.95) = {ref_ser:.3f} "
                      f"OUTSIDE [0.85,1.15] -> interpolation failure flag")

        # Winner: prefer SER at reference cell first, then mean score
        cfg_scores.sort(key=lambda x: (
            0 if x["ref_cell_ser_ok"] else 1,
            -x["ser_in_range_count"],
            x["mean_score"],
        ))
        winner = cfg_scores[0] if cfg_scores else None

        # Check winner-evaporation
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
                ser_gate = "[OK]" if np.isfinite(ser) and 0.85 <= ser <= 1.15 else "[FAIL]"
                print(f"  n={n} W={snr_W} Z={snr_Z}: score={res['score']:.3f}  "
                      f"SER={ser:.3f} {ser_gate}  cov={cov:.3f}  [{elapsed:.0f}s]")
                results[(n, snr_W, snr_Z)] = res
            except Exception as e:
                print(f"  n={n} ERROR: {e}")
                results[(n, snr_W, snr_Z)] = {"error": str(e)}

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
        path = self.summ_dir / f"phase18_{self.method}_stage{stage}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        import datetime
        lines = []
        ap = lines.append
        ap(f"# Phase 18 — {self.method} Stage {stage}")
        ap(f"*Generated {datetime.datetime.now().isoformat()}*")
        ap("")

        if stage == 1:
            ap("## Stage 1 — 1D coarse scans (Phase 18: 2 cells per scan)")
            ap(f"Cells: n=1000/SNR=0.95 + n=2000/SNR=0.95 (score = mean)")
            ap(f"Configs tested: {result.get('n_configs', '?')}")
            ap("")
            ap("### Top per axis")
            for axis, top_list in result["top_per_axis"].items():
                ap(f"\n**{axis}**")
                for r in top_list:
                    cs = r.get("cell_scores", {})
                    cells_str = " | ".join(f"{k}={v:.3f}" if isinstance(v, float) and np.isfinite(v) else f"{k}=inf"
                                           for k, v in cs.items())
                    ap(f"- value={r['config'][axis]} combined={r['score']:.3f} [{cells_str}]")
            ap("")
            ap("### Merged best (Stage 2 starting point)")
            for k, v in result["merged_best"].items():
                ap(f"- {k}: {v}")
        elif stage == 2:
            ap("## Stage 2 — 2D refinement")
            ap(f"Configs tested: {result.get('n_configs', '?')}")
            ap("")
            ap("### Top-3 candidates")
            for i, r in enumerate(result.get("top3", [])):
                ap(f"\n**Top {i+1}** (score={r['score']:.4f})")
                for k, v in r["config"].items():
                    ap(f"- {k}: {v}")
        elif stage == 3:
            ap("## Stage 3 — Cross-cell consistency")
            ap("\n### Cross-cell mean scores")
            for r in result.get("cfg_means", []):
                ap(f"- ci={r['ci']} mean_score={r['mean_score']:.4f}")
            ap("\n### Top-2 stable")
            for i, r in enumerate(result.get("top2", [])):
                ap(f"**Top {i+1}** mean_score={r['mean_score']:.4f}")
                for k, v in r["config"].items():
                    ap(f"- {k}: {v}")
        elif stage == 4:
            ap(f"## Stage 4 — Disjoint validation (M={self.M_validate}) [Phase 18: 4 cells]")
            ap(f"Winner-evaporation: {result.get('winner_evaporation')}")
            ap("\n### Final ranking")
            for r in result.get("cfg_scores", []):
                ref_ser = r.get("ref_cell_ser", float("nan"))
                ref_ok = r.get("ref_cell_ser_ok", False)
                ref_str = f"ref_cell_SER={ref_ser:.3f} {'[OK]' if ref_ok else '[FAIL]'}"
                ap(f"- ci={r['ci']} mean_score={r['mean_score']:.4f}  "
                   f"SER_in_range={r['ser_in_range_count']}/{r.get('ser_total', 4)}  {ref_str}")
            ap("")
            winner = result.get("winner")
            if winner:
                ap("### Winner")
                for k, v in winner["config"].items():
                    ap(f"- {k}: {v}")
                ref_ser = winner.get("ref_cell_ser", float("nan"))
                ap(f"\nReference cell SER (n=2000/SNR=0.95): **{ref_ser:.3f}** "
                   f"{'[PASS]' if winner.get('ref_cell_ser_ok') else '[FAIL]'}")
        elif stage == 5:
            ap(f"## Stage 5 — Regime robustness")
            ap(f"SER passed: {result.get('ser_passed_count', 0)}/{result.get('ser_total_count', 0)}")
            ap(f"Envelope: **{result.get('envelope', 'UNKNOWN')}**")
            ap("\n### Per-cell results")
            for cell, r in result.get("results", {}).items():
                if isinstance(r, dict) and "diags" in r:
                    d = r["diags"]
                    ser = d.get("SER", float("nan"))
                    cov = d.get("_cov_ref_NONBLIND", float("nan"))
                    gate = "✓" if np.isfinite(ser) and 0.85 <= ser <= 1.15 else "✗"
                    ap(f"- {cell}: score={r['score']:.3f}  SER={ser:.3f} {gate}  cov={cov:.3f}")
                else:
                    ap(f"- {cell}: ERROR")

        path.write_text("\n".join(lines), encoding="utf-8")

    # ────────────────────────────────────────────────────────────────────────
    #  Main loop
    # ────────────────────────────────────────────────────────────────────────

    def run(self):
        state = self._load_state()
        print(f"\n{'='*70}")
        print(f"Phase 18 Adaptive Tuner: method={self.method} mode={self.mode}")
        print(f"Starting from stage {self.start_stage}, stopping after {self.stop_stage}")
        print(f"Phase 18 changes vs Phase 17:")
        print(f"  1. lambda_h x n_fold gate (hard fail < 0.01)")
        print(f"  2. kappa_h penalty starts at 1e5 (was 1e6)")
        print(f"  3. Stage 1 at 2 cells: n=1000 + n=2000")
        print(f"  4. Stage 4 includes n=2000/SNR=0.95 reference cell")
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

        self._write_final_report(cached, state)

        elapsed_min = (time.time() - self.t_start) / 60
        print(f"\n{'='*70}")
        print(f"Phase 18 {self.method} COMPLETE in {elapsed_min:.1f} min")
        winner = state.get("winner")
        print(f"Winner: {winner.get('config', 'NONE') if winner else 'NONE'}")
        print(f"Envelope: {state.get('envelope', 'unknown')}")
        print(f"{'='*70}")
        return cached

    def _write_final_report(self, cached: Dict, state: Dict):
        path = self.summ_dir / f"phase18_{self.method}_FINAL.md"
        import datetime
        lines = []
        ap = lines.append
        ap(f"# Phase 18 — {self.method} BLIND TUNING FINAL (corrected protocol)")
        ap(f"*Generated {datetime.datetime.now().isoformat()}*")
        ap(f"*Mode: {self.mode}, Total compute: {state['compute_used_minutes']:.1f} min*")
        ap("")
        ap("## Protocol changes vs Phase 17")
        ap("1. **Gate: lambda_h × n_fold < 0.01 → hard fail** (prevents interpolation)")
        ap("2. **kappa_h penalty starts at 1e5** (was 1e6; BIN_mh2000 has kappa_h~87K)")
        ap("3. **Stage 1 at 2 cells** (n=1000 + n=2000; score=mean)")
        ap("4. **Stage 4 includes n=2000/SNR=0.95** (reference cell missed in Phase 17)")
        ap("5. **Seeds 62M/62.5M/63M** (disjoint from Phase 17: 60M/60.5M/61M)")
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
            if "ref_cell_ser" in winner:
                ref_ser = winner["ref_cell_ser"]
                ref_ok = winner.get("ref_cell_ser_ok", False)
                ap(f"SER at reference cell (n=2000/SNR=0.95): **{ref_ser:.3f}** "
                   f"({'✓ PASS' if ref_ok else '✗ FAIL'})")
            ap(f"Envelope: **{state.get('envelope', 'N/A')}**")
        else:
            ap("No winner declared.")
        ap("")
        ap("## Comparison to BIN_mh2000 (Phase 14B oracle-tuned baseline)")
        ap("BIN_mh2000: m_h=2000, ell_h=2.75, lambda_h=1e-5,")
        ap("gamma_critic=1e-4, lambda_r=1e-2, n_features_r=500, ell_scale_r=3.5")
        ap("lambda_h × n_fold (n=2000, K=5): **1e-5 × 1600 = 0.016 > 0.01 ✓**")
        ap("")
        if winner and winner.get("config"):
            cfg = winner["config"]
            ap("| Param | BIN_mh2000 | Phase18_blind | Match? |")
            ap("|-------|-----------|---------------|--------|")
            BIN = BENNETT_DEFAULTS
            for k in BIN:
                v_bin = BIN[k]
                v_p18 = cfg.get(k, "?")
                match = "✓" if v_bin == v_p18 else "**DIFFERS**"
                ap(f"| {k} | {v_bin} | {v_p18} | {match} |")
        ap("")
        ap("## Stage history")
        for h in state["stage_history"]:
            ap(f"- Stage {h['stage']} completed at {h.get('completed_at', '?')}")
        ap("")
        ap("## Scientific interpretation (S6.7 material)")
        ap("Phase 17 found a config with lambda_h=1e-6 that had the best blind score")
        ap("(0.299) at n=2000/SNR=0.95 but worst actual SER (0.312). The blind score")
        ap("paradox: residual_norm_h→0 (5.85e-5) due to interpolation looked excellent.")
        ap("")
        ap("Phase 18 adds an explicit gate lambda_h × n_fold < 0.01 → inf, which")
        ap("prevents this failure mode. The question is whether the corrected procedure")
        ap("converges to BIN_mh2000 — what a well-equipped practitioner would find.")
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Final report: {path}")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 18 adaptive blind tuning (corrected Bennett protocol)"
    )
    parser.add_argument("--method", choices=["bennett", "drkpv"], required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--start_stage", type=int, default=1)
    parser.add_argument("--stop_stage",  type=int, default=5)
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
