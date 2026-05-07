"""
pci.tuning.adaptive_tuner_drkpv
===============================

DRKPV Adaptive Tuner (Phase 17 protocol). Five-stage blind selection of
``DRKernel + KPVPolicyBridgeQ`` hyperparameters for the kernel proximal
DR estimator on DGP2.

Differences from :class:`BennettAdaptiveTuner` (Phase 18):

* Stage 1 runs at a SINGLE cell (n=1000, SNR=0.95). Phase 18 added a
  second cell for Bennett to expose the lambda_h x m_h interpolation
  trap; DRKPV does not have an h-bridge interpolation issue, so the
  added cell would only inflate compute.
* Stage 4 has 3 cells (Phase 18 added a 4th for Bennett ref-cell
  protection). DRKPV's SER does not collapse at the reference cell.
* Score function is :func:`pci.tuning.blind_score.drkpv_blind_score`
  (no kappa_r hard-fail, since DRKPV inherently has kappa_r ~ 1e+50 by
  design on DGP2).
* Seed bases 70M / 70.5M / 71M (disjoint from Bennett 60M/62M).

Stages
------
1. **1D coarse scans** at one cell (n=1000, SNR=0.95)
2. **2D refinement** around best of Stage 1
3. **Cross-cell consistency** (3 cells, M=50)
4. **Disjoint validation** (M=100, 3 cells)
5. **Regime robustness** (low SNR, weak Z, weak W, strong reference)

Extracted from
``simulations/experiments/simulation/blind_tuning_dgp2_drkpv.py``. The original module continues to expose
``AdaptiveTuner = DRKPVAdaptiveTuner`` as a backward-compatibility alias.
"""
from __future__ import annotations

import datetime
import json
import time
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pci.tuning.blind_score import drkpv_blind_score
from pci.tuning.extract_diagnostics import extract_diagnostics as _extract_diagnostics
from pci.tuning.factories import (
    DRKPV_DEFAULTS,
    A_GRID_DGP2_DEFAULT,
    REF_IDX_DGP2_DEFAULT,
    make_drkpv_estimator,
)


_PKG_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RAW_DIR = (
    _PKG_ROOT / "simulations" / "results" / "raw" / "s7"
    / "blind_tuning_drkpv" / "drkpv"
)
_DEFAULT_SUMM_DIR = _PKG_ROOT / "simulations" / "results" / "summaries"


def _default_dgp_factory(snr_W: float, snr_Z: float):
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    return MichaelisMentenDGP(snr_W=snr_W, snr_Z=snr_Z)


def _default_meta_builder(dgp, n: int, a_grid: np.ndarray):
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


class DRKPVAdaptiveTuner:
    """Five-stage adaptive blind tuner for DRKernel + KPV bridges (Phase 17).

    Constructor parameters mirror :class:`BennettAdaptiveTuner` but with
    DRKPV-appropriate defaults (single Stage 1 cell, 3 Stage 4 cells,
    seed bases 70M+).
    """

    STAGE1_CELLS: Tuple[Tuple[int, float, float], ...] = ((1000, 0.95, 0.95),)

    def __init__(
        self,
        mode: str = "full",
        start_stage: int = 1,
        stop_stage: int = 5,
        raw_dir: Optional[Path] = None,
        summ_dir: Optional[Path] = None,
        report_prefix: str = "phase17",
        a_grid: Optional[Sequence[float]] = None,
        ref_idx: int = REF_IDX_DGP2_DEFAULT,
        seed_tune: int = 70_000_000,
        seed_valid: int = 70_500_000,
        seed_robust: int = 71_000_000,
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
            dict(defaults_config) if defaults_config is not None else dict(DRKPV_DEFAULTS)
        )

        if mode == "smoke":
            self.M_explore = 3
            self.M_validate = 5
        else:
            self.M_explore = 30
            self.M_validate = 100

        self.t_start = time.time()

    # ───────────────────── State I/O ──────────────────────────────────

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {
            "method": "drkpv",
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
        if isinstance(obj, dict):
            return {
                (str(k) if isinstance(k, tuple) else k):
                DRKPVAdaptiveTuner._make_json_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [DRKPVAdaptiveTuner._make_json_serializable(i) for i in obj]
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

    # ───────────────────── Cell construction & 1-config ──────────────

    def _build_dgp_and_meta(self, snr_W: float, snr_Z: float, n: int):
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
        from pci.runner.block_runner import _run_block
        dgp, h_KDE, ell_W0, ell_A0, ell_Z0, J_pt = self._build_dgp_and_meta(snr_W, snr_Z, n)

        def factory():
            return make_drkpv_estimator(
                config, dgp, h_KDE, ell_W0, ell_A0, ell_Z0,
                a_grid=self.a_grid, ref_dose_index=self.ref_idx,
            )

        raw_dir.mkdir(parents=True, exist_ok=True)
        data = _run_block(
            dgp, factory, self.a_grid, J_pt,
            n=n, M=M, seed_base=seed_base,
            label=label, raw_dir=raw_dir,
        )
        diags = _extract_diagnostics(data, ref_idx=self.ref_idx)
        score = drkpv_blind_score(diags, n=n)
        return {
            "label": label, "config": config,
            "diags": diags, "score": score,
            "n": n, "snr_W": snr_W, "snr_Z": snr_Z, "M": M,
        }

    # ───────────────────── Stage 1: 1D coarse scans ────────────────────

    def stage_1_grids(self) -> Dict[str, List]:
        return {
            "lambda_h":    [1e-6, 1e-5, 3e-5, 1e-4, 1e-3],
            "ell_scale":   [1.5, 2.5, 3.5, 5.0],
            "lambda_Q":    [1e-5, 1e-4, 1e-3, 1e-2],
            "clip_factor": [None, 5.0, 3.0, 2.0],
        }

    def run_stage_1(self) -> Dict[str, Any]:
        print("\n=== STAGE 1: 1D coarse scans for drkpv ===")
        grids = self.stage_1_grids()
        defaults = self.defaults_config
        per_axis_results: Dict[str, List[Dict]] = {}

        for axis, values in grids.items():
            print(f"\n  [drkpv] Axis: {axis}")
            axis_results = []
            for i, v in enumerate(values):
                config = dict(defaults)
                config[axis] = v
                label = f"S1_{axis}_{i}"
                seed_base = self.seed_tune + hash(label) % 100_000
                t0 = time.time()
                try:
                    res = self._run_one_config(
                        config, n=1000, snr_W=0.95, snr_Z=0.95,
                        M=self.M_explore, seed_base=seed_base,
                        raw_dir=self.raw_dir / "stage1",
                        label=label,
                    )
                    elapsed = time.time() - t0
                    print(f"    {axis}={v}  score={res['score']:.3f}  [{elapsed:.0f}s]")
                    axis_results.append(res)
                except Exception as e:
                    print(f"    ERROR {axis}={v}: {e}")
                    axis_results.append({"label": label, "error": str(e), "config": config})
            per_axis_results[axis] = axis_results

        top_per_axis: Dict[str, List[Dict]] = {}
        for axis, results in per_axis_results.items():
            valid = [r for r in results if "score" in r and np.isfinite(r["score"])]
            valid.sort(key=lambda r: r["score"])
            top_per_axis[axis] = valid[:2] if valid else []

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

    # ───────────────────── Stage 2: 2D refinement ──────────────────────

    def stage_2_grids(self, stage1_result: Dict) -> List[Dict]:
        merged = stage1_result["merged_best"]
        top = stage1_result["top_per_axis"]
        lambda_h_vals = [r["config"]["lambda_h"] for r in top.get("lambda_h", [])][:2]
        ell_vals      = [r["config"]["ell_scale"] for r in top.get("ell_scale", [])][:2]
        lQ_vals       = [r["config"]["lambda_Q"]  for r in top.get("lambda_Q", [])][:2]
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
        print("\n=== STAGE 2: 2D refinement for drkpv ===")
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
        print("\n=== STAGE 3: Cross-cell consistency for drkpv ===")
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
        print(f"\n=== STAGE 4: Disjoint validation (M={self.M_validate}) for drkpv ===")
        top2 = stage3_result.get("top2", [])
        if not top2:
            return {"stage": 4, "error": "No top-2 from Stage 3"}

        # Phase 17: 3 cells (no n=2000/SNR=0.95 ref cell).
        cells = [
            (1000, 0.95, 0.95),
            (2000, 0.90, 0.90),
            (2000, 0.85, 0.85),
        ]
        results_per_config = {}
        for ci, cfg_data in enumerate(top2):
            cfg_results = {}
            for n, snrW, snrZ in cells:
                label = f"S4_c{ci}_n{n}_s{int(snrW*100):02d}"
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
                    print(f"  cfg{ci} n={n} SNR={snrW}: score={res['score']:.3f}  "
                          f"SER={res['diags'].get('SER', float('nan')):.3f}  [{elapsed:.0f}s]")
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
            cfg_scores.append({
                "ci": ci, "config": top2[ci]["config"],
                "mean_score": mean_score,
                "ser_in_range_count": ser_in_range,
                "per_cell": cell_results,
            })

        cfg_scores.sort(key=lambda x: (-x["ser_in_range_count"], x["mean_score"]))
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
        print("\n=== STAGE 5: Regime robustness for drkpv ===")
        winner = stage4_result.get("winner")
        if not winner:
            return {"stage": 5, "error": "No winner from Stage 4"}

        cells = [
            (1000, 0.80, 0.80),
            (1000, 0.95, 0.50),
            (1000, 0.50, 0.95),
            (2000, 0.95, 0.95),
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
                print(f"  n={n} W={snrW} Z={snrZ}: score={res['score']:.3f}  "
                      f"SER={ser:.3f}  cov={cov:.3f}  [{elapsed:.0f}s]")
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
        path = self.summ_dir / f"{self.report_prefix}_drkpv_stage{stage}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        ap = lines.append
        ap(f"# {self.report_prefix.title()} -- drkpv Stage {stage}")
        ap(f"*Generated {datetime.datetime.now().isoformat()}*")
        ap("")
        if stage == 1:
            ap("## Stage 1 -- 1D coarse scans")
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
            ap(f"## Stage 4 -- Disjoint validation (M={self.M_validate})")
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
            ap("## Stage 5 -- Regime robustness")
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

    def _write_final_report(self, cached: Dict, state: Dict):
        path = self.summ_dir / f"{self.report_prefix}_drkpv_FINAL.md"
        lines = []
        ap = lines.append
        ap(f"# {self.report_prefix.title()} -- drkpv BLIND TUNING FINAL")
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
        ap("Baseline: Phase 9B defaults (lambda_1=lambda_2=3e-5, ell_scale=3.5, lambda_Q=1e-3).")
        ap("**This is the first proper tuning of DRKPV.**")
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Final report: {path}")

    def run(self):
        state = self._load_state()
        print(f"\n{'='*70}")
        print(f"Phase 17 Adaptive Tuner: method=drkpv mode={self.mode}")
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
        print(f"Phase 17 drkpv COMPLETE in {elapsed_min:.1f} min")
        winner = state.get("winner")
        print(f"Winner: {winner.get('config', 'NONE') if winner else 'NONE'}")
        print(f"Envelope: {state.get('envelope', 'unknown')}")
        print(f"{'='*70}")
        return cached
