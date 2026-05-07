"""
Phase 19 -- Real-Data Adaptive Blind Tuning (Bennett-only)

Application of the Phase 18 corrected 5-stage blind tuning protocol to four
real datasets (RHC, NHANES Cadmium, NLSY79 Father, ETH ESS Fertilizer) for
Section 7 of the essay. Architecture:

  - RealDataDGP: bootstrap wrapper around CSV (duck-typed to PCIDgp.generate)
  - Frisch-Waugh-Lovell residualization of Y/W/Z on X (Ridge OLS, A NOT residualized)
  - DGP3-style coarsening for binary A (RHC): A_star = A + rho_coarsen * N(0,1)
  - Single-cell tuning per stage (the lambda_h x m_h gate is n-independent)
  - Stage 5 = proxy substitution sensitivity (replaces Stage 5 SNR variation)
  - Seeds 80M/80.5M/81M (disjoint from Phase 17 60-71M, Phase 18 62-63M)

Usage
-----
    python -u -m simulations.experiments.real_data_tuning --dataset rhc --mode smoke
    python -u -m simulations.experiments.real_data_tuning --dataset nhanes_cadmium --mode full
    python -u -m simulations.experiments.real_data_tuning --dataset rhc --mode full \\
        --start_stage 2 --stop_stage 4 --resume

The 5 stages are intended to be paused at adaptive checkpoints, hence
--start_stage / --stop_stage / --resume flags. See plan file for checkpoint
structure.

Output
------
    simulations/results/raw/s8/{dataset}/bennett/stage{1-5}/*.pkl
    simulations/results/raw/s8/{dataset}/bennett/state.json
    simulations/results/summaries/realdata_{dataset}_bennett_stage{1-5}.md
    simulations/results/summaries/realdata_{dataset}_bennett_FINAL.md
"""
from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd

# Reuse Phase 18 plumbing: scoring fn, diagnostics, base class, _run_block
from simulations.experiments.s7_simulations.s7_8_blind_tuning_bennett import (
    bennett_blind_score_v2,
    extract_diagnostics,
    AdaptiveTuner,
    BENNETT_DEFAULTS,
    DEFAULTS,
)
from simulations.experiments.dgp2_bias_diagnostics import (
    _silverman_h,
    _median_bandwidth,
    _run_block,
)
from simulations.dgp.base import DGPSample

_BASE_DIR = Path(__file__).resolve().parents[3]
_DATA_DIR = _BASE_DIR / "simulations" / "datasets" / "processed"
_RAW_BASE = _BASE_DIR / "simulations" / "results" / "raw" / "s8"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"


# ════════════════════════════════════════════════════════════════════════════
#  Per-dataset configurations
# ════════════════════════════════════════════════════════════════════════════

# Per-dataset default for residualization. Set to False after Checkpoint 0
# diagnostics showed that Ridge(alpha=1) residualization on X destroys proxy
# signal on RHC/NHANES/ETH ESS (Fredholm 0.005 -> 0.4 magnitude jumps). This
# is a methodological finding to document in S7. Override at the CLI with
# --no_residualize / --residualize as needed.
_RESIDUALIZE_DEFAULT = {
    "rhc": False,
    "nhanes_cadmium": False,
    "nlsy79_father": False,
    "eth_ess": False,
}

DATASETS: Dict[str, Dict[str, Any]] = {
    "rhc": {
        "csv": "rhc.csv",
        "y_col": "Y", "a_col": "A", "w_col": "W", "z_col": "Z",
        "x_cols": ["X_age", "X_meanbp"],
        "binary_a": True,
        # Post-coarsening A is N(0.38, 0.30**2 + 0.486**2) ~ continuous-like.
        # a_grid bracketing the two centers (a=0 untreated, a=1 treated)
        "a_grid": np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        "ref_idx": 4,                  # treated arm = a=1
        "w2_col": "W2", "z2_col": "Z2",
        "rho_sens": [0.20, 0.40],      # sensitivity around ρ=0.30
        "reference_estimate": -0.053,
        "reference_source": "Tchetgen 2020 / Cui 2023",
    },
    "nhanes_cadmium": {
        "csv": "nhanes_cadmium.csv",
        "y_col": "Y", "a_col": "A", "w_col": "W", "z_col": "Z",
        "x_cols": ["X_age", "X_sex", "X_ipr", "X_race"],
        "binary_a": False,
        "a_grid_quantiles": [0.10, 0.30, 0.50, 0.70, 0.90],
        "ref_idx": 2,                  # median quantile
        "w2_col": "W_cadmium", "z2_col": "Z_mercury",
        "reference_estimate": None,    # no published PCI estimate
        "reference_source": "Tellez-Plaza 2013 epidemiology (+1-3 mmHg/µg Cd)",
    },
    "nlsy79_father": {
        "csv": "nlsy79_father.csv",
        "y_col": "Y", "a_col": "A", "w_col": "W", "z_col": "Z",
        "x_cols": ["X_race", "X_sex"],
        "binary_a": False,
        "a_grid_quantiles": [0.10, 0.30, 0.50, 0.70, 0.90],
        "ref_idx": 2,
        "w2_col": None, "z2_col": None,
        "reference_estimate": 0.10,    # ~10% return to schooling
        "reference_source": "Egami & Tchetgen 2024 / Card 1995 (8-12% range)",
    },
    "eth_ess": {
        "csv": "eth_ess_fertilizer.csv",
        "y_col": "Y", "a_col": "A", "w_col": "W", "z_col": "Z",
        "x_cols": ["X_crop", "X_ea"],
        "binary_a": False,
        "filter": "A > 0",             # condition on fertilizer users
        "a_grid_quantiles": [0.10, 0.30, 0.50, 0.70, 0.90],
        "ref_idx": 2,
        "w2_col": "W2", "z2_col": None,
        "reference_estimate": 0.010,   # ~+1% log/kg from Duflo 2008 RCT (rough)
        "reference_source": "Duflo, Kremer, Robinson 2008 (Kenya RCT, indicative)",
    },
}


# ════════════════════════════════════════════════════════════════════════════
#  RealDataDGP — bootstrap wrapper duck-typed to PCIDgp.generate / m_true
# ════════════════════════════════════════════════════════════════════════════

class RealDataDGP:
    """
    Wraps a real CSV into a duck-typed DGP for the Phase 18 _run_block harness.

    NOT a subclass of PCIDgp because h0/q0/psi_true are unknown on real data.
    Only `.generate(n, seed)` and `.m_true(a)` are required by `_run_block`.

    Construction steps:
      1. Load CSV, optionally filter (eth_ess: A > 0)
      2. Residualize Y/W/Z on X via Ridge OLS (D6 — Frisch-Waugh-Lovell partial-out)
      3. For binary A (RHC), add Gaussian noise: A_star = A + rho_coarsen * N(0,1)
      4. Build a_grid (fixed for binary, quantile-based for continuous)
      5. Cache fixed bandwidths from full sample (NOT recomputed per bootstrap)
    """

    def __init__(
        self,
        name: str,
        *,
        rho_coarsen: Optional[float] = None,
        use_proxy_alt: Optional[Dict[str, str]] = None,
        residualize: bool = True,
        ridge_alpha: float = 1.0,
        seed_coarsen: int = 0,
    ) -> None:
        if name not in DATASETS:
            raise KeyError(f"Unknown dataset {name!r}. Valid: {list(DATASETS)}")
        cfg = DATASETS[name]
        df = pd.read_csv(_DATA_DIR / cfg["csv"])

        # Filter (e.g. eth_ess fertilizer users only)
        if cfg.get("filter") == "A > 0":
            df = df.loc[df[cfg["a_col"]] > 0].reset_index(drop=True)

        # Drop rows with NaN in any of the columns we will use (keeps Ridge happy)
        cols_used = [cfg["y_col"], cfg["a_col"]]
        proxy_alt = use_proxy_alt or {}
        cols_used.append(proxy_alt.get("w_col") or cfg["w_col"])
        cols_used.append(proxy_alt.get("z_col") or cfg["z_col"])
        if cfg.get("x_cols"):
            cols_used += list(cfg["x_cols"])
        before = len(df)
        df = df.dropna(subset=cols_used).reset_index(drop=True)
        n_dropped = before - len(df)

        # Optional proxy substitution for Stage 5
        proxy_alt = use_proxy_alt or {}
        w_col = proxy_alt.get("w_col") or cfg["w_col"]
        z_col = proxy_alt.get("z_col") or cfg["z_col"]

        Y = df[cfg["y_col"]].to_numpy(float)
        A_obs = df[cfg["a_col"]].to_numpy(float)
        W = df[w_col].to_numpy(float)
        Z = df[z_col].to_numpy(float)
        X = df[cfg["x_cols"]].to_numpy(float) if cfg.get("x_cols") else None

        # D6 — residualize Y/W/Z on X (NOT A)
        residualize_info = {"applied": False}
        if residualize and X is not None and X.shape[0] > 0:
            from sklearn.linear_model import Ridge
            var_before = {"Y": float(np.var(Y)), "W": float(np.var(W)), "Z": float(np.var(Z))}
            Y = Y - Ridge(alpha=ridge_alpha).fit(X, Y).predict(X)
            W = W - Ridge(alpha=ridge_alpha).fit(X, W).predict(X)
            Z = Z - Ridge(alpha=ridge_alpha).fit(X, Z).predict(X)
            var_after = {"Y": float(np.var(Y)), "W": float(np.var(W)), "Z": float(np.var(Z))}
            residualize_info = {
                "applied": True, "alpha": ridge_alpha,
                "var_before": var_before, "var_after": var_after,
                "var_explained_pct": {
                    k: 100.0 * (1.0 - var_after[k] / var_before[k])
                       if var_before[k] > 0 else 0.0
                    for k in var_before
                },
            }

        # D3 — RHC binary coarsening (one-shot, deterministic)
        applied_rho = None
        if cfg.get("binary_a") and rho_coarsen is not None and rho_coarsen > 0:
            rng0 = np.random.default_rng(seed_coarsen)
            A = A_obs + float(rho_coarsen) * rng0.normal(size=A_obs.shape)
            applied_rho = float(rho_coarsen)
        else:
            A = A_obs

        # a_grid construction
        if cfg.get("binary_a"):
            a_grid = np.asarray(cfg["a_grid"], dtype=float)
        else:
            a_grid = np.quantile(A, cfg["a_grid_quantiles"])
            # If A is discrete (e.g. NLSY79 integer years), quantile(0.1) and
            # quantile(0.5) may collide. Replace duplicates by spanning the
            # observed range linearly.
            if len(np.unique(a_grid)) < len(a_grid):
                lo, hi = float(np.min(A)), float(np.max(A))
                a_grid = np.linspace(lo, hi, len(cfg["a_grid_quantiles"]))

        self.name = name
        self.cfg = cfg
        self.Y, self.A, self.W, self.Z = Y, A, W, Z
        self.X = X
        self.A_obs = A_obs
        self.N = len(Y)
        self.n_dropped = int(n_dropped)
        self.rho_coarsen = applied_rho
        self.residualize_info = residualize_info
        self.a_grid = a_grid
        self.ref_idx = int(cfg["ref_idx"])
        self.w_col_used = w_col
        self.z_col_used = z_col

    # Duck-typed interface ----------------------------------------------------

    def generate(self, n: int, seed: int) -> DGPSample:
        """
        Bootstrap resample. The `n` argument is accepted for interface
        compatibility but ignored — we always sample N (full size with
        replacement). Y, A, W, Z residualized + coarsened arrays are reused.
        """
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self.N, size=self.N)
        return DGPSample(
            Y=self.Y[idx], A=self.A[idx], W=self.W[idx], Z=self.Z[idx],
            X=self.X[idx] if self.X is not None else None,
            U=np.full(self.N, np.nan),    # latent — unobserved on real data
            psi_0=float("nan"),            # ATE unknown
        )

    def m_true(self, a: float) -> float:
        return float("nan")

    # Bandwidth helpers (computed once on full sample) ------------------------

    def bandwidths(self) -> Dict[str, float]:
        return {
            "h_KDE":  _silverman_h(self.A),
            "ell_W":  _median_bandwidth(self.W),
            "ell_A":  _median_bandwidth(self.A),
            "ell_Z":  _median_bandwidth(self.Z),
        }


# ════════════════════════════════════════════════════════════════════════════
#  Estimator factory (real-data variant — fixes hardcoded n=1000 bug,
#  takes a_grid + ref_idx as kwargs instead of module-level constants)
# ════════════════════════════════════════════════════════════════════════════

def _make_bennett_estimator_real(
    config: Dict, dgp, h_KDE: float,
    ell_W: float, ell_A: float, ell_Z: float,
    *, a_grid, ref_idx: int,
):
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
        a_grid=list(a_grid),
        bandwidth=h_KDE,
        ref_dose_index=int(ref_idx),
        seed=42,
    )


# ════════════════════════════════════════════════════════════════════════════
#  Real-data adaptive tuner (single-cell, proxy-variant Stage 5)
# ════════════════════════════════════════════════════════════════════════════

class RealDataAdaptiveTuner(AdaptiveTuner):
    """
    Single-cell variant of AdaptiveTuner. All 5 stage methods are overridden
    to bypass the (n, snr_W, snr_Z) cell logic of Phase 18 and instead operate
    on the cached real data (with bootstrap reps inside _run_block).
    """

    def __init__(
        self,
        dataset: str,
        *,
        mode: str = "full",
        start_stage: int = 1,
        stop_stage: int = 5,
        rho_coarsen: float = 0.30,
        residualize: bool = True,
    ) -> None:
        # Initialize base (sets compute timer, t_start). Use method='bennett'
        # to keep DEFAULTS lookup compatible. Override paths and seeds below.
        super().__init__(method="bennett", mode=mode,
                         start_stage=start_stage, stop_stage=stop_stage)

        if dataset not in DATASETS:
            raise KeyError(f"Unknown dataset {dataset!r}")

        self.dataset = dataset
        # Coarsening only relevant for binary A (RHC). Stored even if None for reporting.
        self.rho_coarsen = rho_coarsen if DATASETS[dataset].get("binary_a") else None

        # Override seed bases: 80M/80.5M/81M family
        self.seed_tune   = 80_000_000
        self.seed_valid  = 80_500_000
        self.seed_robust = 81_000_000

        # Override paths (subclass sees self.raw_dir, self.state_path)
        self.raw_dir = _RAW_BASE / dataset / "bennett"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.summ_dir = _SUMM_DIR
        self.state_path = self.raw_dir / "state.json"

        # Build dgp + bandwidths + grid (cached, used by _run_one_real)
        self.dgp = RealDataDGP(
            dataset, rho_coarsen=self.rho_coarsen, residualize=residualize,
        )
        bw = self.dgp.bandwidths()
        self.h_KDE = bw["h_KDE"]
        self.ell_W0 = bw["ell_W"]
        self.ell_A0 = bw["ell_A"]
        self.ell_Z0 = bw["ell_Z"]

        self.A_GRID_LOCAL = self.dgp.a_grid
        self.REF_IDX_LOCAL = self.dgp.ref_idx
        self.J_pt_dummy = np.full(len(self.A_GRID_LOCAL), np.nan)
        self.residualize = residualize     # remembered for Stage 5 variants

        # M overrides (smoke / full)
        if mode == "smoke":
            self.M_explore  = 3
            self.M_refine   = 3
            self.M_validate = 5
        else:
            self.M_explore  = 30
            self.M_refine   = 50
            self.M_validate = 100

    # --- core runner: single cell, bootstrap reps via _run_block ------------

    def _run_one_real(
        self, config: Dict, *, M: int, seed_base: int,
        raw_dir: Path, label: str,
        dgp: Optional[RealDataDGP] = None,
        h_KDE: Optional[float] = None,
        ell_W: Optional[float] = None,
        ell_A: Optional[float] = None,
        ell_Z: Optional[float] = None,
        a_grid: Optional[np.ndarray] = None,
        ref_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """One config × M bootstrap reps. Optional dgp/bandwidth overrides
        are used by Stage 5 proxy variants."""
        dgp = dgp if dgp is not None else self.dgp
        h_KDE = h_KDE if h_KDE is not None else self.h_KDE
        ell_W = ell_W if ell_W is not None else self.ell_W0
        ell_A = ell_A if ell_A is not None else self.ell_A0
        ell_Z = ell_Z if ell_Z is not None else self.ell_Z0
        a_grid = a_grid if a_grid is not None else self.A_GRID_LOCAL
        ref_idx = ref_idx if ref_idx is not None else self.REF_IDX_LOCAL
        J_pt = np.full(len(a_grid), np.nan)

        def factory():
            return _make_bennett_estimator_real(
                config, dgp, h_KDE, ell_W, ell_A, ell_Z,
                a_grid=a_grid, ref_idx=ref_idx,
            )

        raw_dir.mkdir(parents=True, exist_ok=True)
        data = _run_block(
            dgp, factory, np.asarray(a_grid), J_pt,
            n=dgp.N, M=M, seed_base=seed_base,
            label=label, raw_dir=raw_dir,
        )
        diags = extract_diagnostics(data, ref_idx=ref_idx)

        # Inject lambda_h and m_h for the gate
        if "lambda_h" in config:
            diags["lambda_h"] = float(config["lambda_h"])
        if "m_h" in config:
            diags["m_h"] = float(config["m_h"])

        score = bennett_blind_score_v2(diags, n=dgp.N)
        return {
            "label": label,
            "config": config,
            "diags": diags,
            "score": score,
            "n": dgp.N,
            "M": M,
        }

    # --- Stage 1: 1D scans, single cell -------------------------------------

    def run_stage_1(self) -> Dict[str, Any]:
        print(f"\n=== STAGE 1 [real:{self.dataset}/bennett] 1D coarse scans (single cell) ===")
        grids = self.stage_1_grids()
        defaults = DEFAULTS["bennett"]
        per_axis: Dict[str, List[Dict]] = {}

        for axis, values in grids.items():
            print(f"\n  Axis: {axis}")
            results = []
            for i, v in enumerate(values):
                cfg = dict(defaults)
                cfg[axis] = v
                label = f"S1_{axis}_{i}"
                seed_base = self.seed_tune + abs(hash(label)) % 100_000
                t0 = time.time()
                try:
                    res = self._run_one_real(
                        cfg, M=self.M_explore, seed_base=seed_base,
                        raw_dir=self.raw_dir / "stage1", label=label,
                    )
                    elapsed = time.time() - t0
                    sc_str = f"{res['score']:.3f}" if np.isfinite(res['score']) else "inf"
                    print(f"    {axis}={v}  score={sc_str}  [{elapsed:.0f}s]")
                    results.append(res)
                except Exception as e:
                    print(f"    {axis}={v}  ERROR: {e}")
                    results.append({"label": label, "config": cfg,
                                    "score": float("inf"), "error": str(e)})
            per_axis[axis] = results

        top_per_axis = {}
        for axis, results in per_axis.items():
            valid = [r for r in results if np.isfinite(r.get("score", float("inf")))]
            valid.sort(key=lambda r: r["score"])
            top_per_axis[axis] = valid[:2]
            if not valid:
                print(f"  WARNING: no valid config for axis {axis}")

        merged = dict(defaults)
        for axis, top_list in top_per_axis.items():
            if top_list:
                merged[axis] = top_list[0]["config"][axis]

        return {
            "stage": 1,
            "per_axis_results": per_axis,
            "top_per_axis": top_per_axis,
            "merged_best": merged,
            "n_configs": sum(len(r) for r in per_axis.values()),
        }

    # --- Stage 2: 2D refinement (single cell) -------------------------------

    def run_stage_2(self, stage1_result: Dict) -> Dict[str, Any]:
        print(f"\n=== STAGE 2 [real:{self.dataset}/bennett] 2D refinement (single cell) ===")
        configs = self.stage_2_grids(stage1_result)
        results = []
        for i, cfg in enumerate(configs):
            label = f"S2_c{i}"
            seed_base = self.seed_tune + 100_000 + abs(hash(label)) % 100_000
            t0 = time.time()
            try:
                res = self._run_one_real(
                    cfg, M=self.M_refine, seed_base=seed_base,
                    raw_dir=self.raw_dir / "stage2", label=label,
                )
                elapsed = time.time() - t0
                sc_str = f"{res['score']:.3f}" if np.isfinite(res['score']) else "inf"
                print(f"  cfg{i} score={sc_str}  [{elapsed:.0f}s]")
                results.append(res)
            except Exception as e:
                print(f"  cfg{i} ERROR: {e}")
                results.append({"label": label, "error": str(e), "config": cfg})

        valid = [r for r in results if np.isfinite(r.get("score", float("inf")))]
        valid.sort(key=lambda r: r["score"])
        return {"stage": 2, "all_results": results, "top3": valid[:3],
                "n_configs": len(configs)}

    # --- Stage 3: bootstrap stability (top-3 re-run with disjoint seeds) ---

    def run_stage_3(self, stage2_result: Dict) -> Dict[str, Any]:
        print(f"\n=== STAGE 3 [real:{self.dataset}/bennett] bootstrap stability (single cell, disjoint seeds) ===")
        top3 = stage2_result.get("top3", [])
        if not top3:
            return {"stage": 3, "error": "No top-3 from Stage 2"}

        cfg_means = []
        for ci, cfg_data in enumerate(top3):
            label = f"S3_c{ci}_disjoint"
            seed_base = self.seed_tune + 200_000 + ci * 10_000
            t0 = time.time()
            try:
                res = self._run_one_real(
                    cfg_data["config"], M=self.M_refine, seed_base=seed_base,
                    raw_dir=self.raw_dir / "stage3", label=label,
                )
                elapsed = time.time() - t0
                sc_str = f"{res['score']:.3f}" if np.isfinite(res['score']) else "inf"
                print(f"  cfg{ci} score={sc_str}  [{elapsed:.0f}s]")
                cfg_means.append({
                    "ci": ci, "config": cfg_data["config"],
                    "mean_score": float(res["score"]) if np.isfinite(res["score"]) else float("inf"),
                    "per_cell": {("real",): res},
                })
            except Exception as e:
                print(f"  cfg{ci} ERROR: {e}")
                cfg_means.append({"ci": ci, "config": cfg_data["config"],
                                  "mean_score": float("inf"),
                                  "per_cell": {("real",): {"error": str(e)}}})

        cfg_means.sort(key=lambda x: x["mean_score"])
        return {"stage": 3, "cfg_means": cfg_means, "top2": cfg_means[:2]}

    # --- Stage 4: disjoint validation (single cell, M=100) ------------------

    def run_stage_4(self, stage3_result: Dict) -> Dict[str, Any]:
        print(f"\n=== STAGE 4 [real:{self.dataset}/bennett] validation M={self.M_validate} ===")
        top2 = stage3_result.get("top2", [])
        if not top2:
            return {"stage": 4, "error": "No top-2 from Stage 3"}

        cfg_scores = []
        for ci, cfg_data in enumerate(top2):
            label = f"S4_c{ci}_validation"
            seed_base = self.seed_valid + ci * 10_000
            t0 = time.time()
            try:
                res = self._run_one_real(
                    cfg_data["config"], M=self.M_validate, seed_base=seed_base,
                    raw_dir=self.raw_dir / "stage4", label=label,
                )
                elapsed = time.time() - t0
                ser = res["diags"].get("SER", float("nan"))
                ser_str = f"{ser:.3f}" if np.isfinite(ser) else "nan"
                sc_str = f"{res['score']:.3f}" if np.isfinite(res['score']) else "inf"
                print(f"  cfg{ci} score={sc_str}  SER={ser_str} (informational on real data)  [{elapsed:.0f}s]")
                cfg_scores.append({
                    "ci": ci, "config": cfg_data["config"],
                    "mean_score": float(res["score"]) if np.isfinite(res["score"]) else float("inf"),
                    "ser_in_range_count": 1 if (np.isfinite(ser) and 0.85 <= ser <= 1.15) else 0,
                    "ser_total": 1,
                    "ref_cell_ser": float(ser) if np.isfinite(ser) else float("nan"),
                    "ref_cell_ser_ok": bool(np.isfinite(ser) and 0.85 <= ser <= 1.15),
                    "per_cell": {("real",): res},
                })
            except Exception as e:
                print(f"  cfg{ci} ERROR: {e}")
                cfg_scores.append({"ci": ci, "config": cfg_data["config"],
                                   "mean_score": float("inf"),
                                   "ser_in_range_count": 0, "ser_total": 1,
                                   "ref_cell_ser": float("nan"),
                                   "ref_cell_ser_ok": False,
                                   "per_cell": {("real",): {"error": str(e)}}})

        # Rank: mean_score ascending (SER not used as a hard gate on real data
        # because bootstrap SE ≈ plug-in SE → SER ≈ 1 mechanically; we still
        # surface it for diagnostic purposes)
        cfg_scores.sort(key=lambda x: x["mean_score"])
        winner = cfg_scores[0] if cfg_scores else None

        s3_top1 = top2[0]["config"] if top2 else None
        s4_top1 = winner["config"] if winner else None
        return {"stage": 4, "cfg_scores": cfg_scores, "winner": winner,
                "winner_evaporation": (s3_top1 != s4_top1) if (s3_top1 and s4_top1) else None}

    # --- Stage 5: PROXY-SUBSTITUTION SENSITIVITY (replaces Phase 18 SNR variation) ---

    def _stage5_variants(self) -> Dict[str, Dict[str, Any]]:
        """Build a dict {variant_name -> kwargs to RealDataDGP} for Stage 5."""
        cfg = self.dgp.cfg
        variants: Dict[str, Dict[str, Any]] = {
            "baseline": {"rho_coarsen": self.rho_coarsen,
                         "use_proxy_alt": None},
        }
        if self.dataset == "rhc":
            for rho in cfg.get("rho_sens", []):
                variants[f"rho_{rho}"] = {"rho_coarsen": float(rho),
                                          "use_proxy_alt": None}
        if cfg.get("w2_col"):
            variants["W_alt"] = {"rho_coarsen": self.rho_coarsen,
                                 "use_proxy_alt": {"w_col": cfg["w2_col"]}}
        if cfg.get("z2_col"):
            variants["Z_alt"] = {"rho_coarsen": self.rho_coarsen,
                                 "use_proxy_alt": {"z_col": cfg["z2_col"]}}
        return variants

    def run_stage_5(self, stage4_result: Dict) -> Dict[str, Any]:
        print(f"\n=== STAGE 5 [real:{self.dataset}/bennett] proxy substitution sensitivity ===")
        winner = stage4_result.get("winner")
        if not winner:
            return {"stage": 5, "error": "No winner from Stage 4"}

        variants = self._stage5_variants()
        results: Dict[str, Dict[str, Any]] = {}
        for vname, vkwargs in variants.items():
            print(f"\n  Variant: {vname}  kwargs={vkwargs}")
            try:
                dgp_v = RealDataDGP(self.dataset, residualize=self.residualize, **vkwargs)
                bw = dgp_v.bandwidths()
                label = f"S5_{vname}"
                seed_base = self.seed_robust + abs(hash(vname)) % 100_000
                t0 = time.time()
                res = self._run_one_real(
                    winner["config"], M=self.M_validate, seed_base=seed_base,
                    raw_dir=self.raw_dir / "stage5", label=label,
                    dgp=dgp_v, h_KDE=bw["h_KDE"],
                    ell_W=bw["ell_W"], ell_A=bw["ell_A"], ell_Z=bw["ell_Z"],
                    a_grid=dgp_v.a_grid, ref_idx=dgp_v.ref_idx,
                )
                elapsed = time.time() - t0

                # Extract J_dr_ref (mean across reps) for envelope check
                # NB: extract_diagnostics doesn't expose J_dr_ref directly, recompute from records
                j_dr_ref = self._compute_j_dr_ref(self.raw_dir / "stage5" / f"{label}.pkl",
                                                   ref_idx=dgp_v.ref_idx)
                res["J_dr_ref_mean"] = j_dr_ref
                sc_str = f"{res['score']:.3f}" if np.isfinite(res['score']) else "inf"
                print(f"    score={sc_str}  J_dr_ref={j_dr_ref:.4f}  [{elapsed:.0f}s]")
                results[vname] = res
            except Exception as e:
                print(f"    ERROR: {e}")
                results[vname] = {"error": str(e)}

        # Envelope: ±20% of baseline J_dr_ref
        baseline_jdr = results.get("baseline", {}).get("J_dr_ref_mean", float("nan"))
        in_band = []
        if np.isfinite(baseline_jdr) and abs(baseline_jdr) > 1e-12:
            for vname, vres in results.items():
                if vname == "baseline":
                    continue
                jdr = vres.get("J_dr_ref_mean", float("nan")) if isinstance(vres, dict) else float("nan")
                if np.isfinite(jdr):
                    rel = abs(jdr - baseline_jdr) / abs(baseline_jdr)
                    in_band.append(rel <= 0.20)
                else:
                    in_band.append(False)

        if not in_band:
            envelope = "PROXY-SINGLE"
        elif all(in_band):
            envelope = "PROXY-ROBUST"
        elif sum(in_band) >= len(in_band) // 2:
            envelope = "PROXY-CONDITIONAL"
        else:
            envelope = "PROXY-SPEC-ISSUE"

        return {"stage": 5, "results": results, "winner": winner,
                "envelope": envelope, "baseline_jdr": float(baseline_jdr)}

    @staticmethod
    def _compute_j_dr_ref(pkl_path: Path, ref_idx: int) -> float:
        import pickle
        if not pkl_path.exists():
            return float("nan")
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
        recs = [r for r in data.get("records", []) if r.get("error") is None]
        vals = []
        for r in recs:
            jdr = r.get("J_dr")
            if jdr is None:
                continue
            try:
                vals.append(float(jdr[ref_idx]))
            except Exception:
                continue
        return float(np.mean(vals)) if vals else float("nan")

    # --- Reporting ----------------------------------------------------------

    def _write_stage_report(self, stage: int, result: Dict):
        """Override to write under realdata_{dataset}_bennett_stage{k}.md naming."""
        path = self.summ_dir / f"realdata_{self.dataset}_bennett_stage{stage}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        import datetime
        lines = [f"# Phase 19 — {self.dataset} bennett Stage {stage}",
                 f"*Generated {datetime.datetime.now().isoformat()}*", ""]
        ap = lines.append

        if stage == 1:
            ap(f"## Stage 1 — 1D coarse scans (single cell, M={self.M_explore})")
            ap(f"Dataset: {self.dataset} (N={self.dgp.N})")
            ap(f"Bandwidths: h_KDE={self.h_KDE:.4f}  ell_W={self.ell_W0:.4f}  "
               f"ell_A={self.ell_A0:.4f}  ell_Z={self.ell_Z0:.4f}")
            if self.dataset == "rhc":
                ap(f"Coarsening rho={self.rho_coarsen}")
            ap(f"\n### Top per axis")
            for axis, top_list in result.get("top_per_axis", {}).items():
                ap(f"\n**{axis}**")
                for r in top_list:
                    ap(f"- {axis}={r['config'][axis]} score={r['score']:.4f}")
            ap(f"\n### Merged best (Stage 2 starting point)")
            for k, v in result.get("merged_best", {}).items():
                ap(f"- {k}: {v}")
        elif stage == 2:
            ap(f"## Stage 2 — 2D refinement (M={self.M_refine})")
            ap(f"Configs tested: {result.get('n_configs', '?')}")
            ap(f"\n### Top-3")
            for i, r in enumerate(result.get("top3", [])):
                ap(f"\n**Top {i+1}** score={r['score']:.4f}")
                for k, v in r["config"].items():
                    ap(f"- {k}: {v}")
        elif stage == 3:
            ap(f"## Stage 3 — Bootstrap stability (disjoint seeds, M={self.M_refine})")
            ap(f"\n### Top-2 stable")
            for i, r in enumerate(result.get("top2", [])):
                ap(f"\n**Top {i+1}** score={r['mean_score']:.4f}")
                for k, v in r["config"].items():
                    ap(f"- {k}: {v}")
        elif stage == 4:
            ap(f"## Stage 4 — Disjoint validation (M={self.M_validate})")
            ap(f"Winner-evaporation: {result.get('winner_evaporation')}")
            ap(f"\n### Final ranking (by score; SER informational only)")
            for r in result.get("cfg_scores", []):
                ser = r.get("ref_cell_ser", float("nan"))
                ap(f"- ci={r['ci']} score={r['mean_score']:.4f}  SER={ser:.3f}")
            winner = result.get("winner")
            if winner:
                ap(f"\n### Winner")
                for k, v in winner["config"].items():
                    ap(f"- {k}: {v}")
        elif stage == 5:
            ap(f"## Stage 5 — Proxy substitution sensitivity")
            ap(f"Envelope: **{result.get('envelope', 'UNKNOWN')}**")
            ap(f"Baseline J_dr_ref: {result.get('baseline_jdr', float('nan')):.4f}")
            ap(f"\n### Per-variant results")
            for vname, vres in result.get("results", {}).items():
                if isinstance(vres, dict) and "diags" in vres:
                    sc = vres.get("score", float("nan"))
                    jdr = vres.get("J_dr_ref_mean", float("nan"))
                    ap(f"- **{vname}**: score={sc:.4f}  J_dr_ref={jdr:.4f}")
                else:
                    err = vres.get("error", "?") if isinstance(vres, dict) else "?"
                    ap(f"- **{vname}**: ERROR ({err})")

        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_final_report(self, cached: Dict, state: Dict):
        path = self.summ_dir / f"realdata_{self.dataset}_bennett_FINAL.md"
        import datetime
        lines = [
            f"# Phase 19 — {self.dataset} bennett FINAL",
            f"*Generated {datetime.datetime.now().isoformat()}*",
            f"*Mode: {self.mode}, Total compute: {state['compute_used_minutes']:.1f} min*",
            "",
            "## Dataset",
            f"- Name: {self.dataset}",
            f"- N: {self.dgp.N}",
            f"- Binary A: {self.dgp.cfg.get('binary_a', False)}",
            f"- Coarsening rho: {self.rho_coarsen}",
            f"- Reference: {self.dgp.cfg.get('reference_estimate', 'n/a')} ({self.dgp.cfg.get('reference_source', 'n/a')})",
            "",
            "## Residualization (Frisch-Waugh-Lovell)",
        ]
        ap = lines.append
        ri = self.dgp.residualize_info
        if ri.get("applied"):
            ap("Applied (Ridge alpha=1.0). Variance reduction:")
            for k, pct in ri.get("var_explained_pct", {}).items():
                ap(f"- {k}: {pct:.1f}%")
        else:
            ap("Not applied (no X covariates).")

        ap("\n## Bandwidths (computed once on full sample)")
        ap(f"- h_KDE: {self.h_KDE:.4f}")
        ap(f"- ell_W: {self.ell_W0:.4f}")
        ap(f"- ell_A: {self.ell_A0:.4f}")
        ap(f"- ell_Z: {self.ell_Z0:.4f}")
        ap(f"- a_grid: {list(np.round(self.A_GRID_LOCAL, 4))}  ref_idx={self.REF_IDX_LOCAL}")

        ap("\n## Winner spec")
        winner = state.get("winner")
        if winner:
            ap("```json")
            ap(json.dumps(winner.get("config", {}), indent=2, default=str))
            ap("```")
        else:
            ap("(no winner — tuning did not converge)")

        ap("\n## Envelope")
        ap(f"- {state.get('envelope', 'unknown')}")

        ap("\n## Comparison to BIN_mh2000 (Phase 14B oracle baseline)")
        BIN = {"m_h": 2000, "ell_h": 2.75, "lambda_h": 1e-5,
               "gamma_critic": 1e-4, "lambda_r": 1e-2,
               "n_features_r": 500, "ell_scale_r": 3.5}
        if winner:
            wcfg = winner.get("config", {})
            ap("| Param | BIN_mh2000 | Phase 19 winner | Match |")
            ap("|---|---|---|---|")
            for k, v in BIN.items():
                wv = wcfg.get(k, "?")
                match = "✓" if str(wv) == str(v) else "**DIFFERS**"
                ap(f"| {k} | {v} | {wv} | {match} |")

        ap("\n## Stage history")
        for h in state.get("stage_history", []):
            ap(f"- Stage {h.get('stage')} completed at {h.get('completed_at')}")

        path.write_text("\n".join(lines), encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════════
#  Verification helper
# ════════════════════════════════════════════════════════════════════════════

def _verify_dgps() -> None:
    """Quick sanity check across all 4 datasets. Prints a one-line summary."""
    print("\n=== _verify_dgps: sanity check on RealDataDGP for all 4 datasets ===")
    for name in DATASETS:
        rho = 0.30 if DATASETS[name].get("binary_a") else None
        try:
            dgp = RealDataDGP(name, rho_coarsen=rho, residualize=True)
            bw = dgp.bandwidths()
            sample = dgp.generate(dgp.N, seed=0)
            ri = dgp.residualize_info
            ap = sum(np.isnan(np.asarray(sample.U)).astype(int))
            print(f"\n  [{name}] N={dgp.N}  rho_coarsen={dgp.rho_coarsen}")
            print(f"    Y: shape={sample.Y.shape} mean={sample.Y.mean():.4f} sd={sample.Y.std():.4f}")
            print(f"    A: shape={sample.A.shape} mean={sample.A.mean():.4f} sd={sample.A.std():.4f}")
            print(f"    W: shape={sample.W.shape} mean={sample.W.mean():.4f} sd={sample.W.std():.4f}")
            print(f"    Z: shape={sample.Z.shape} mean={sample.Z.mean():.4f} sd={sample.Z.std():.4f}")
            print(f"    bandwidths: h_KDE={bw['h_KDE']:.4f}  ell_W={bw['ell_W']:.4f}  "
                  f"ell_A={bw['ell_A']:.4f}  ell_Z={bw['ell_Z']:.4f}")
            print(f"    a_grid: {list(np.round(dgp.a_grid, 4))}  ref_idx={dgp.ref_idx}")
            print(f"    U all-NaN: {ap == dgp.N}  psi_0 NaN: {np.isnan(sample.psi_0)}")
            if ri.get("applied"):
                pcts = ri.get("var_explained_pct", {})
                print(f"    residualize var-explained: Y={pcts.get('Y',0):.1f}%  "
                      f"W={pcts.get('W',0):.1f}%  Z={pcts.get('Z',0):.1f}%")
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS.keys()),
                   help="dataset name. Omit + use --verify to just sanity-check.")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--rho_coarsen", type=float, default=0.30,
                   help="(rhc only) Gaussian noise SD on binary A. Default 0.30.")
    p.add_argument("--start_stage", type=int, default=1)
    p.add_argument("--stop_stage", type=int, default=5)
    p.add_argument("--no_residualize", action="store_true",
                   help="Skip Y/W/Z residualization on X (overrides per-dataset default).")
    p.add_argument("--residualize", action="store_true",
                   help="Force Y/W/Z residualization on X (overrides per-dataset default).")
    p.add_argument("--verify", action="store_true",
                   help="Run _verify_dgps() and exit. No --dataset needed.")
    args = p.parse_args()

    if args.verify:
        _verify_dgps()
        return

    if not args.dataset:
        p.error("--dataset is required unless --verify is passed.")

    # Resolve residualization: per-dataset default unless CLI overrides
    if args.no_residualize:
        residualize = False
    elif args.residualize:
        residualize = True
    else:
        residualize = _RESIDUALIZE_DEFAULT.get(args.dataset, True)
    print(f"Residualization for {args.dataset}: {'ON' if residualize else 'OFF'} "
          f"(per-dataset default = {_RESIDUALIZE_DEFAULT.get(args.dataset, 'unknown')})")

    tuner = RealDataAdaptiveTuner(
        dataset=args.dataset, mode=args.mode,
        rho_coarsen=args.rho_coarsen,
        start_stage=args.start_stage, stop_stage=args.stop_stage,
        residualize=residualize,
    )
    tuner.run()


if __name__ == "__main__":
    main()
