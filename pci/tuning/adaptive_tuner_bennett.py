"""
pci.tuning.adaptive_tuner_bennett
=================================

Bennett Adaptive Tuner (Phase 18 corrected protocol). Five-stage blind
selection of Bennett-RFF DR hyperparameters using only the diagnostics
emitted by ``pci.runner.block_runner._build_record`` -- no oracle leakage.

The protocol corrects the SER collapse failure of Phase 17 by adding the
``lambda_h * m_h >= 0.01`` interpolation gate (see :mod:`pci.tuning.blind_score`)
and by including the ``(n=2000, SNR=0.95)`` reference cell in Stage 4.

Stages
------
1. **1D coarse scans** at TWO cells (n=1000 + n=2000), score = mean
2. **2D refinement** around best of Stage 1
3. **Cross-cell consistency** (3 cells, M=50)
4. **Disjoint validation** (M=100, 4 cells incl. reference)
5. **Regime robustness** (low SNR, weak Z, weak W, strong reference)

Extracted from
``simulations/experiments/simulation/blind_tuning_dgp2_bennett.py``. The original module continues to expose
``AdaptiveTuner = BennettAdaptiveTuner`` as a backward-compatibility alias.
"""
from __future__ import annotations

import datetime
import json
import os
import time
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pci.tuning.blind_score import bennett_blind_score_v2, SCORING
from pci.tuning.extract_diagnostics import extract_diagnostics as _extract_diagnostics
from pci.tuning.factories import (
    BENNETT_DEFAULTS,
    A_GRID_DGP2_DEFAULT,
    REF_IDX_DGP2_DEFAULT,
    make_bennett_estimator,
)


# Default base directory: the repo root, four levels up from this file
# (pci/tuning/adaptive_tuner_bennett.py -> pci_essay/).
_PKG_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RAW_DIR = (
    _PKG_ROOT / "simulations" / "results" / "raw" / "s7"
    / "blind_tuning_bennett" / "bennett"
)
_DEFAULT_SUMM_DIR = _PKG_ROOT / "simulations" / "results" / "summaries"


# ════════════════════════════════════════════════════════════════════════════
#  DGP factory (default = MichaelisMentenDGP at requested SNR)
# ════════════════════════════════════════════════════════════════════════════

def _default_dgp_factory(snr_W: float, snr_Z: float):
    """Default DGP factory for Bennett tuning: DGP2 (Michaelis-Menten)."""
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    return MichaelisMentenDGP(snr_W=snr_W, snr_Z=snr_Z)


def _default_meta_builder(dgp, n: int, a_grid: np.ndarray):
    """Build h_KDE, ell_W/A/Z, J_policy_true from a reference sample of n=2000."""
    from pci.runner.bandwidth_helpers import (
        _silverman_h, _median_bandwidth, _compute_J_policy_true,
    )
    ref = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref.A)
    ell_W0 = _median_bandwidth(ref.W)
    ell_A0 = _median_bandwidth(ref.A)
    ell_Z0 = _median_bandwidth(ref.Z)
    J_pt = _compute_J_policy_true(dgp, a_grid, h_KDE)
    return h_KDE, ell_W0, ell_A0, ell_Z0, J_pt


# ════════════════════════════════════════════════════════════════════════════
#  BennettAdaptiveTuner
# ════════════════════════════════════════════════════════════════════════════

class BennettAdaptiveTuner:
    """Five-stage adaptive blind tuner for ``BennettIndepFunctionalDR`` (Phase 18).

    Parameters
    ----------
    mode : {"full", "smoke"}, default "full"
        ``"smoke"`` reduces M_explore to 3 and M_validate to 5 for fast
        sanity checks; ``"full"`` runs M_explore=30 / M_validate=100.
    start_stage, stop_stage : int, default 1, 5
        Inclusive stage range to execute. ``run_stage_k`` methods are
        callable individually too.
    raw_dir : Path, optional
        Directory for per-stage PKL checkpoints. Defaults to
        ``simulations/results/raw/s7/blind_tuning_bennett/bennett``.
    summ_dir : Path, optional
        Directory for per-stage Markdown reports.
    report_prefix : str, default "phase18"
        Prefix for report file names: ``{prefix}_bennett_stage{k}.md``.
    a_grid : Sequence[float], optional
        Dose grid. Defaults to ``[1.8, 2.2, 2.6, 3.0]`` (DGP2).
    ref_idx : int, default 2
        Index of the reference dose in a_grid.
    seed_tune, seed_valid, seed_robust : int, optional
        Random-seed bases for Stage 1-3, Stage 4, Stage 5 respectively.
        Defaults match Phase 18 (62M / 62.5M / 63M).
    dgp_factory : callable, optional
        ``(snr_W, snr_Z) -> dgp`` builder. Defaults to MichaelisMentenDGP.
    defaults_config : dict, optional
        Bennett defaults override. Defaults to ``BENNETT_DEFAULTS``.
    """

    # Stage 1 cells: 2-cell sweep [Phase 18 NEW]
    STAGE1_CELLS: Tuple[Tuple[int, float, float], ...] = (
        (1000, 0.95, 0.95),
        (2000, 0.95, 0.95),
    )

    def __init__(
        self,
        mode: str = "full",
        start_stage: int = 1,
        stop_stage: int = 5,
        raw_dir: Optional[Path] = None,
        summ_dir: Optional[Path] = None,
        report_prefix: str = "phase18",
        a_grid: Optional[Sequence[float]] = None,
        ref_idx: int = REF_IDX_DGP2_DEFAULT,
        seed_tune: int = 62_000_000,
        seed_valid: int = 62_500_000,
        seed_robust: int = 63_000_000,
        dgp_factory: Optional[Callable] = None,
        defaults_config: Optional[Dict[str, Any]] = None,
    ):
        if mode not in ("full", "smoke"):
            raise ValueError(f"Unknown mode {mode!r}")
        self.mode = mode
        self.start_stage = int(start_stage)
        self.stop_stage = int(stop_stage)

        self.raw_dir = Path(raw_dir) if raw_dir is not None else _DEFAULT_RAW_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.summ_dir = Path(summ_dir) if summ_dir is not None else _DEFAULT_SUMM_DIR
        self.summ_dir.mkdir(parents=True, exist_ok=True)
        self.report_prefix = report_prefix
        self.state_path = self.raw_dir / "state.json"

        self.a_grid = (
            np.asarray(a_grid, dtype=float)
            if a_grid is not None else np.asarray(A_GRID_DGP2_DEFAULT, dtype=float)
        )
        self.ref_idx = int(ref_idx)

        self.seed_tune = int(seed_tune)
        self.seed_valid = int(seed_valid)
        self.seed_robust = int(seed_robust)

        self.dgp_factory = dgp_factory if dgp_factory is not None else _default_dgp_factory
        self.defaults_config = (
            dict(defaults_config) if defaults_config is not None else dict(BENNETT_DEFAULTS)
        )

        if mode == "smoke":
            self.M_explore = 3
            self.M_validate = 5
        else:
            self.M_explore = 30
            self.M_validate = 100

        self.t_start = time.time()

    # ───────────────────── Persistent state I/O ─────────────────────────

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {
            "method": "bennett",
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
        """Recursively convert tuple keys to strings for JSON dump."""
        if isinstance(obj, dict):
            return {
                (str(k) if isinstance(k, tuple) else k):
                BennettAdaptiveTuner._make_json_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [BennettAdaptiveTuner._make_json_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj

    def _save_state(self, state: Dict[str, Any]):
        state["compute_used_minutes"] = (time.time() - self.t_start) / 60.0
        self.state_path.write_text(
            json.dumps(self._make_json_serializable(state), indent=2, default=str),
            encoding="utf-8",
        )

    # ───────────────────── Cell construction & 1-config ──────────────────

    def _build_dgp_and_meta(self, snr_W: float, snr_Z: float, n: int):
        """Return (dgp, h_KDE, ell_W0, ell_A0, ell_Z0, J_policy_true)."""
        dgp = self.dgp_factory(snr_W=snr_W, snr_Z=snr_Z)
        h_KDE, ell_W0, ell_A0, ell_Z0, J_pt = _default_meta_builder(dgp, n, self.a_grid)
        return dgp, h_KDE, ell_W0, ell_A0, ell_Z0, J_pt

    def _run_one_config(
        self,
        config: Dict[str, Any],
        *,
        n: int,
        snr_W: float,
        snr_Z: float,
        M: int,
        seed_base: int,
        raw_dir: Path,
        label: str,
    ) -> Dict[str, Any]:
        """Run M reps for one config and compute Phase 18 score."""
        from pci.runner.block_runner import _run_block
        dgp, h_KDE, ell_W0, ell_A0, ell_Z0, J_pt = self._build_dgp_and_meta(snr_W, snr_Z, n)

        def factory():
            return make_bennett_estimator(
                config, dgp, h_KDE, ell_W0, ell_A0, ell_Z0,
                a_grid=self.a_grid.tolist(), ref_dose_index=self.ref_idx,
            )

        raw_dir.mkdir(parents=True, exist_ok=True)
        data = _run_block(
            dgp, factory, self.a_grid, J_pt,
            n=n, M=M, seed_base=seed_base,
            label=label, raw_dir=raw_dir,
        )
        diags = _extract_diagnostics(data, ref_idx=self.ref_idx)

        # Inject lambda_h and m_h for the Phase 18 interpolation gate
        if "lambda_h" in config:
            diags["lambda_h"] = float(config["lambda_h"])
        if "m_h" in config:
            diags["m_h"] = float(config["m_h"])

        score = bennett_blind_score_v2(diags, n=n)
        return {
            "label": label, "config": config,
            "diags": diags, "score": score,
            "n": n, "snr_W": snr_W, "snr_Z": snr_Z, "M": M,
        }

    # ───────────────────── Stage 1: 1D coarse scans ─────────────────────

    def stage_1_grids(self) -> Dict[str, List]:
        """Bennett 1D scan grids (Phase 18: lambda_h starts at 1e-5)."""
        return {
            "lambda_h":     [1e-5, 3e-5, 1e-4, 1e-3],
            "gamma_critic": [1e-5, 1e-4, 1e-3, 1e-2],
            "ell_h":        [1.5, 2.0, 2.75, 3.5, 5.0],
            "lambda_r":     [1e-3, 1e-2, 1e-1],
            "ell_scale_r":  [2.5, 3.5, 5.0],
        }

    def run_stage_1(self) -> Dict[str, Any]:
        print(f"\n=== STAGE 1: 1D coarse scans for bennett ===")
        print(f"  Cells: {self.STAGE1_CELLS} [Phase 18: 2-cell sweep]")
        grids = self.stage_1_grids()
        defaults = self.defaults_config
        per_axis_results: Dict[str, List[Dict]] = {}

        for axis, values in grids.items():
            print(f"\n  [bennett] Axis: {axis}")
            axis_results = []
            for i, v in enumerate(values):
                config = dict(defaults)
                config[axis] = v
                label_base = f"S1_{axis}_{i}"
                seed_base = self.seed_tune + hash(label_base) % 100_000
                t0 = time.time()

                cell_scores = []
                for n_cell, snrW_cell, snrZ_cell in self.STAGE1_CELLS:
                    label_cell = f"{label_base}_n{n_cell}"
                    try:
                        res_cell = self._run_one_config(
                            config, n=n_cell, snr_W=snrW_cell, snr_Z=snrZ_cell,
                            M=self.M_explore, seed_base=seed_base,
                            raw_dir=self.raw_dir / "stage1",
                            label=label_cell,
                        )
                        cell_scores.append(res_cell["score"])
                    except Exception as e:
                        print(f"    ERROR {axis}={v} n={n_cell}: {e}")
                        cell_scores.append(float("inf"))

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

                axis_results.append({
                    "label": label_base,
                    "config": config,
                    "score": combined_score,
                    "cell_scores": {
                        f"n{c[0]}": s for c, s in zip(self.STAGE1_CELLS, cell_scores)
                    },
                })
            per_axis_results[axis] = axis_results

        top_per_axis: Dict[str, List[Dict]] = {}
        for axis, results in per_axis_results.items():
            valid = [r for r in results if "score" in r and np.isfinite(r["score"])]
            valid.sort(key=lambda r: r["score"])
            top_per_axis[axis] = valid[:2] if valid else []
            if not valid:
                print(f"  WARNING: no valid config for axis {axis} in Stage 1")

        merged = dict(self.defaults_config)
        for axis, top_list in top_per_axis.items():
            if top_list:
                merged[axis] = top_list[0]["config"][axis]

        return {
            "stage": 1,
            "per_axis_results": per_axis_results,
            "top_per_axis": top_per_axis,
            "merged_best": merged,
            "n_configs": sum(len(r) for r in per_axis_results.values()),
        }

    # ───────────────────── Stage 2: 2D refinement ───────────────────────

    def stage_2_grids(self, stage1_result: Dict) -> List[Dict]:
        merged = stage1_result["merged_best"]
        top = stage1_result["top_per_axis"]
        lambda_h_vals = [r["config"]["lambda_h"]    for r in top.get("lambda_h", [])][:2]
        gamma_vals    = [r["config"]["gamma_critic"] for r in top.get("gamma_critic", [])][:2]
        ell_vals      = [r["config"]["ell_h"]        for r in top.get("ell_h", [])][:2]
        if not lambda_h_vals: lambda_h_vals = [merged["lambda_h"]]
        if not gamma_vals:    gamma_vals    = [merged["gamma_critic"]]
        if not ell_vals:      ell_vals      = [merged["ell_h"]]
        configs = []
        for lh, gc, eh in product(lambda_h_vals, gamma_vals, ell_vals):
            cfg = dict(merged)
            cfg["lambda_h"] = lh
            cfg["gamma_critic"] = gc
            cfg["ell_h"] = eh
            configs.append(cfg)
        return configs

    def run_stage_2(self, stage1_result: Dict) -> Dict[str, Any]:
        print(f"\n=== STAGE 2: 2D refinement for bennett ===")
        configs = self.stage_2_grids(stage1_result)
        results = []
        for i, config in enumerate(configs):
            label = f"S2_c{i}"
            seed_base = self.seed_tune + 100_000 + hash(label) % 100_000
            t0 = time.time()
            try:
                res = self._run_one_config(
                    config, n=1000, snr_W=0.95, snr_Z=0.95,
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
        return {
            "stage": 2,
            "all_results": results,
            "top3": valid[:3],
            "n_configs": len(configs),
        }

    # ───────────────────── Stage 3: Cross-cell consistency ───────────────

    def run_stage_3(self, stage2_result: Dict) -> Dict[str, Any]:
        print(f"\n=== STAGE 3: Cross-cell consistency for bennett ===")
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
            for n, snrW, snrZ in cells:
                label = f"S3_c{ci}_n{n}_s{int(snrW*100):02d}"
                seed_base = self.seed_tune + 200_000 + ci * 10_000 + n
                t0 = time.time()
                try:
                    res = self._run_one_config(
                        cfg_data["config"], n=n, snr_W=snrW, snr_Z=snrZ,
                        M=max(self.M_explore, 50) if self.mode == "full" else self.M_explore,
                        seed_base=seed_base,
                        raw_dir=self.raw_dir / "stage3",
                        label=label,
                    )
                    elapsed = time.time() - t0
                    print(f"  cfg{ci} n={n} SNR={snrW}: score={res['score']:.3f}  [{elapsed:.0f}s]")
                    cfg_results[(n, snrW, snrZ)] = res
                except Exception as e:
                    print(f"  cfg{ci} n={n} ERROR: {e}")
                    cfg_results[(n, snrW, snrZ)] = {"error": str(e)}
            results_per_config[ci] = cfg_results

        cfg_means = []
        for ci, cell_results in results_per_config.items():
            scores = [r["score"] for r in cell_results.values()
                      if isinstance(r, dict) and "score" in r and np.isfinite(r["score"])]
            mean_score = float(np.mean(scores)) if scores else float("inf")
            cfg_means.append({
                "ci": ci, "config": top3[ci]["config"],
                "mean_score": mean_score, "per_cell": cell_results,
            })
        cfg_means.sort(key=lambda x: x["mean_score"])
        return {
            "stage": 3,
            "results_per_config": results_per_config,
            "cfg_means": cfg_means,
            "top2": cfg_means[:2],
        }

    # ───────────────────── Stage 4: Disjoint validation ─────────────────

    def run_stage_4(self, stage3_result: Dict) -> Dict[str, Any]:
        print(f"\n=== STAGE 4: Disjoint validation (M={self.M_validate}) for bennett ===")
        top2 = stage3_result.get("top2", [])
        if not top2:
            return {"stage": 4, "error": "No top-2 from Stage 3"}

        # Phase 18: 4 cells (added (2000, 0.95, 0.95) reference cell)
        cells = [
            (1000, 0.95, 0.95),
            (2000, 0.95, 0.95),
            (2000, 0.90, 0.90),
            (2000, 0.85, 0.85),
        ]
        n_cells = len(cells)

        results_per_config = {}
        for ci, cfg_data in enumerate(top2):
            cfg_results = {}
            for n, snrW, snrZ in cells:
                label = f"S4_c{ci}_n{n}_s{int(snrW*100):02d}_W{int(snrW*100):02d}Z{int(snrZ*100):02d}"
                seed_base = self.seed_valid + ci * 10_000 + n
                t0 = time.time()
                try:
                    res = self._run_one_config(
                        cfg_data["config"], n=n, snr_W=snrW, snr_Z=snrZ,
                        M=self.M_validate, seed_base=seed_base,
                        raw_dir=self.raw_dir / "stage4",
                        label=label,
                    )
                    elapsed = time.time() - t0
                    ser = res["diags"].get("SER", float("nan"))
                    ser_str = f"{ser:.3f}" if np.isfinite(ser) else "nan"
                    ser_gate = "[OK]" if np.isfinite(ser) and 0.85 <= ser <= 1.15 else "[FAIL]"
                    print(f"  cfg{ci} n={n} W={snrW} Z={snrZ}: score={res['score']:.3f}  "
                          f"SER={ser_str} {ser_gate}  [{elapsed:.0f}s]")
                    cfg_results[(n, snrW, snrZ)] = res
                except Exception as e:
                    print(f"  cfg{ci} n={n} ERROR: {e}")
                    cfg_results[(n, snrW, snrZ)] = {"error": str(e)}
            results_per_config[ci] = cfg_results

        cfg_scores = []
        for ci, cell_results in results_per_config.items():
            scores = [r["score"] for r in cell_results.values()
                      if isinstance(r, dict) and "score" in r and np.isfinite(r["score"])]
            mean_score = float(np.mean(scores)) if scores else float("inf")
            sers = [r["diags"].get("SER", float("nan")) for r in cell_results.values()
                    if isinstance(r, dict) and "diags" in r]
            ser_in_range = sum(1 for s in sers if np.isfinite(s) and 0.85 <= s <= 1.15)

            ref_cell_key = (2000, 0.95, 0.95)
            ref_res = cell_results.get(ref_cell_key, {})
            ref_ser = ref_res.get("diags", {}).get("SER", float("nan")) if isinstance(ref_res, dict) else float("nan")
            ref_ser_ok = np.isfinite(ref_ser) and 0.85 <= ref_ser <= 1.15

            cfg_scores.append({
                "ci": ci, "config": top2[ci]["config"],
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

        cfg_scores.sort(key=lambda x: (
            0 if x["ref_cell_ser_ok"] else 1,
            -x["ser_in_range_count"],
            x["mean_score"],
        ))
        winner = cfg_scores[0] if cfg_scores else None

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

    # ───────────────────── Stage 5: Regime robustness ───────────────────

    def run_stage_5(self, stage4_result: Dict) -> Dict[str, Any]:
        print(f"\n=== STAGE 5: Regime robustness for bennett ===")
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
        for n, snrW, snrZ in cells:
            label = f"S5_n{n}_W{int(snrW*100):02d}_Z{int(snrZ*100):02d}"
            seed_base = self.seed_robust + n + int(snrW * 100) * 10
            t0 = time.time()
            try:
                res = self._run_one_config(
                    winner["config"], n=n, snr_W=snrW, snr_Z=snrZ,
                    M=self.M_validate, seed_base=seed_base,
                    raw_dir=self.raw_dir / "stage5",
                    label=label,
                )
                elapsed = time.time() - t0
                ser = res["diags"].get("SER", float("nan"))
                cov = res["diags"].get("_cov_ref_NONBLIND", float("nan"))
                ser_gate = "[OK]" if np.isfinite(ser) and 0.85 <= ser <= 1.15 else "[FAIL]"
                print(f"  n={n} W={snrW} Z={snrZ}: score={res['score']:.3f}  "
                      f"SER={ser:.3f} {ser_gate}  cov={cov:.3f}  [{elapsed:.0f}s]")
                results[(n, snrW, snrZ)] = res
            except Exception as e:
                print(f"  n={n} ERROR: {e}")
                results[(n, snrW, snrZ)] = {"error": str(e)}

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

    # ───────────────────── Reports & main loop ──────────────────────────

    def _write_stage_report(self, stage: int, result: Dict):
        path = self.summ_dir / f"{self.report_prefix}_bennett_stage{stage}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        ap = lines.append
        ap(f"# {self.report_prefix.title()} -- bennett Stage {stage}")
        ap(f"*Generated {datetime.datetime.now().isoformat()}*")
        ap("")

        if stage == 1:
            ap("## Stage 1 -- 1D coarse scans (Phase 18: 2 cells per scan)")
            ap("Cells: n=1000/SNR=0.95 + n=2000/SNR=0.95 (score = mean)")
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
            ap("## Stage 2 -- 2D refinement")
            ap(f"Configs tested: {result.get('n_configs', '?')}")
            ap("")
            ap("### Top-3 candidates")
            for i, r in enumerate(result.get("top3", [])):
                ap(f"\n**Top {i+1}** (score={r['score']:.4f})")
                for k, v in r["config"].items():
                    ap(f"- {k}: {v}")
        elif stage == 3:
            ap("## Stage 3 -- Cross-cell consistency")
            ap("\n### Cross-cell mean scores")
            for r in result.get("cfg_means", []):
                ap(f"- ci={r['ci']} mean_score={r['mean_score']:.4f}")
            ap("\n### Top-2 stable")
            for i, r in enumerate(result.get("top2", [])):
                ap(f"**Top {i+1}** mean_score={r['mean_score']:.4f}")
                for k, v in r["config"].items():
                    ap(f"- {k}: {v}")
        elif stage == 4:
            ap(f"## Stage 4 -- Disjoint validation (M={self.M_validate}) [Phase 18: 4 cells]")
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
            ap("## Stage 5 -- Regime robustness")
            ap(f"SER passed: {result.get('ser_passed_count', 0)}/{result.get('ser_total_count', 0)}")
            ap(f"Envelope: **{result.get('envelope', 'UNKNOWN')}**")
            ap("\n### Per-cell results")
            for cell, r in result.get("results", {}).items():
                if isinstance(r, dict) and "diags" in r:
                    d = r["diags"]
                    ser = d.get("SER", float("nan"))
                    cov = d.get("_cov_ref_NONBLIND", float("nan"))
                    gate = "OK" if np.isfinite(ser) and 0.85 <= ser <= 1.15 else "FAIL"
                    ap(f"- {cell}: score={r['score']:.3f}  SER={ser:.3f} [{gate}]  cov={cov:.3f}")
                else:
                    ap(f"- {cell}: ERROR")

        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_final_report(self, cached: Dict, state: Dict):
        path = self.summ_dir / f"{self.report_prefix}_bennett_FINAL.md"
        lines = []
        ap = lines.append
        ap(f"# {self.report_prefix.title()} -- bennett BLIND TUNING FINAL (corrected protocol)")
        ap(f"*Generated {datetime.datetime.now().isoformat()}*")
        ap(f"*Mode: {self.mode}, Total compute: {state['compute_used_minutes']:.1f} min*")
        ap("")
        ap("## Protocol changes vs Phase 17")
        ap("1. **Gate: lambda_h x m_h < 0.01 -> hard fail** (prevents interpolation)")
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
                   f"({'PASS' if ref_ok else 'FAIL'})")
            ap(f"Envelope: **{state.get('envelope', 'N/A')}**")
        else:
            ap("No winner declared.")
        ap("")
        ap("## Comparison to BIN_mh2000 (Phase 14B oracle-tuned baseline)")
        ap("BIN_mh2000: m_h=2000, ell_h=2.75, lambda_h=1e-5,")
        ap("gamma_critic=1e-4, lambda_r=1e-2, n_features_r=500, ell_scale_r=3.5")
        ap("lambda_h x n_fold (n=2000, K=5): **1e-5 x 1600 = 0.016 > 0.01 PASS**")
        ap("")
        if winner and winner.get("config"):
            cfg = winner["config"]
            ap("| Param | BIN_mh2000 | Phase18_blind | Match? |")
            ap("|-------|-----------|---------------|--------|")
            for k in self.defaults_config:
                v_bin = self.defaults_config[k]
                v_p18 = cfg.get(k, "?")
                match = "OK" if v_bin == v_p18 else "**DIFFERS**"
                ap(f"| {k} | {v_bin} | {v_p18} | {match} |")
        ap("")
        ap("## Stage history")
        for h in state["stage_history"]:
            ap(f"- Stage {h['stage']} completed at {h.get('completed_at', '?')}")
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Final report: {path}")

    def run(self):
        state = self._load_state()
        print(f"\n{'='*70}")
        print(f"Phase 18 Adaptive Tuner: method=bennett mode={self.mode}")
        print(f"Starting from stage {self.start_stage}, stopping after {self.stop_stage}")
        print(f"{'='*70}")

        cached: Dict[int, Dict] = {}
        runners = [
            (1, self.run_stage_1, lambda: ()),
            (2, self.run_stage_2, lambda: (cached.get(1, {}),)),
            (3, self.run_stage_3, lambda: (cached.get(2, {}),)),
            (4, self.run_stage_4, lambda: (cached.get(3, {}),)),
            (5, self.run_stage_5, lambda: (cached.get(4, {}),)),
        ]
        for stage_idx, runner, args_builder in runners:
            if not (self.start_stage <= stage_idx <= self.stop_stage):
                continue
            t0 = time.time()
            stage_out = runner(*args_builder())
            print(f"\nStage {stage_idx} done in {(time.time() - t0)/60:.1f} min")
            cached[stage_idx] = stage_out
            state["stage_history"].append({"stage": stage_idx, "completed_at": time.time()})
            self._write_stage_report(stage_idx, stage_out)
            self._save_state(state)

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
        print(f"Phase 18 bennett COMPLETE in {elapsed_min:.1f} min")
        winner = state.get("winner")
        print(f"Winner: {winner.get('config', 'NONE') if winner else 'NONE'}")
        print(f"Envelope: {state.get('envelope', 'unknown')}")
        print(f"{'='*70}")
        return cached
