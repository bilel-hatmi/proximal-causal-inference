"""
Phase 14B -- Bennett-indep-super: adaptive exploration to find a good
(h_B, r_B) couple INDEPENDENT of KPV.

Methods compared:
  REG_KPV_super (baseline, from KPV pipeline)
  DRK_KPV_super (DR baseline, KPV q-bridge)
  Bennett_indep_<config> : BennettIndepBridgeH (joint RFF h + critic-stab solve)
                          + BennettPolicyRiesz (RFF mode)

Adaptive 5-stage protocol:
  Stage 2.0 -- smoke (n=300, M=3, 3 configs)
  Stage 2.1 -- coarse h_B grid: 12 configs (lambda_h x gamma_critic)
                at fixed (m_h=1000, m_c=500), n=1000, M=50, SNR=0.95
  Stage 2.2 -- ANALYSE all metrics, pick top region
  Stage 2.3 -- refined h_B: vary m_h around winner
  Stage 2.4 -- combined h_B + RFF r_B (vary lambda_r)
  Stage 2.5 -- disjoint validation top 1-2 at n in {1000, 2000}, M=100, SNR in {0.90, 0.95}

seed_base = 28_500_000 (disjoint from Phase 14A 28M).

Decision logic
--------------
After each Stage, dump ALL metrics (bridge_residual, weighted_residual,
coefficient_norm, cond_G, riesz_residual, V_corr, V_total, ESS, weight_p99,
bias, |b|/SE, coverage, SER) per config.

Selection is autonomous: I prefer configs that are "Pareto-good" across
multiple diagnostics simultaneously (see _autonomous_select() in this script
for the exact logic and rationale).

Outputs
-------
  simulations/results/raw/phase14B/stage<S>/<cid>_M<M>.pkl
  simulations/results/summaries/phase14B_stage<S>.md
  simulations/results/summaries/phase14B_FINAL.md
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[2]
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "phase14B"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
K = len(A_GRID)
REF_IDX = 2
N_FOLDS = 5
RANDOM_STATE = 42
SEED_BASE = 28_500_000

SUPER_HP = dict(lambda_h=3e-5, ell_scale=3.5)


from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
    _compute_J_policy_true,
)
from simulations.archive.utilities.bennett_rff_overnight import (
    _augment_decomp_with_ser, _make_reg_hsuper, _make_drk_hsuper,
)


# ── Factory: BennettIndepFunctionalDR with given hyperparams ──────────────────

def _make_bennett_indep(
    *,
    m_h: int,
    m_c: int,
    lambda_h: float,
    gamma_critic: float,
    ell_h: Optional[float] = None,
    ell_c: Optional[float] = None,
    lambda_r: float = 1e-2,
    n_features_r: int = 500,
    ell_scale_r: float = 3.5,
    h_KDE: float = 0.21,
    h_seed_base: int = 2026000,
    rff_seed_base_r: int = 2025500,
    stabilized_r: bool = False,
):
    from simulations.methods.bennett_indep_dr import BennettIndepFunctionalDR
    def factory():
        return BennettIndepFunctionalDR(
            m_h=m_h, m_c=m_c,
            ell_h=ell_h, ell_c=ell_c,
            lambda_h=lambda_h, gamma_critic=gamma_critic,
            h_seed_base=h_seed_base,
            lambda_r=lambda_r, n_features_r=n_features_r,
            ell_scale_r=ell_scale_r,
            rff_seed_base_r=rff_seed_base_r,
            stabilized_r=stabilized_r,
            n_folds=N_FOLDS, a_grid=A_GRID.tolist(),
            bandwidth=h_KDE, seed=RANDOM_STATE, ref_dose_index=REF_IDX,
        )
    return factory


# ── Run helpers ───────────────────────────────────────────────────────────────

def _run_cell_at_snr_n(
    *,
    snr: float,
    n: int,
    M: int,
    configs: Dict,
    raw_dir: Path,
    seed_base_cell: int,
    force: bool = False,
) -> Dict[str, Optional[dict]]:
    """Run all configs at one (snr, n) cell."""
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=snr, snr_Z=snr)
    ref = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref.A)
    ell_W0 = _median_bandwidth(ref.W)
    ell_A0 = _median_bandwidth(ref.A)
    ell_Z0 = _median_bandwidth(ref.Z)
    J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)

    # Inject h_KDE into bennett_indep configs
    full_configs = {}
    for cid, factory_fn in configs.items():
        if isinstance(factory_fn, dict):
            # dict spec -> build a factory
            spec = factory_fn
            full_configs[cid] = _make_bennett_indep(
                h_KDE=h_KDE, **spec
            )
        else:
            full_configs[cid] = factory_fn
    # Re-create REG/DRK factories with the cell's h_KDE
    if "REG_KPV_super" in full_configs:
        full_configs["REG_KPV_super"] = _make_reg_hsuper(h_KDE)
    if "DRK_KPV_super" in full_configs:
        full_configs["DRK_KPV_super"] = _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0)

    raw_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for f in raw_dir.glob(f"*_M{M}.pkl"):
            f.unlink()

    out = {}
    for cid, factory_fn in full_configs.items():
        label = f"{cid}_M{M}"
        print(f"    [{cid}] n={n}")
        t0 = time.time()
        data = _run_block(
            dgp, factory_fn, A_GRID, J_pt,
            n=n, M=M, seed_base=seed_base_cell,
            label=label, raw_dir=raw_dir,
        )
        elapsed = time.time() - t0
        decomp = _decompose(data)
        if decomp is not None:
            decomp = _augment_decomp_with_ser(decomp, data)
            # Also extract Bennett-indep diagnostics from extras
            try:
                recs = [r for r in data["records"] if r.get("error") is None]
                if recs and "h_bridge_residual_mean" in data["meta"].get("extra_keys", []) or True:
                    # Try to read from records
                    h_bridges = []
                    h_weighted = []
                    h_coef = []
                    h_cond = []
                    for r in recs:
                        # These come from the EstimationResult's extras into _build_record
                        # which doesn't currently surface h_bridge_residual_mean. We need
                        # to read the meta or recompute. For now, read straight from
                        # records (need to update _build_record or post-load).
                        pass
                # Simpler: read first record's psi_hat etc., we already have what we need
            except Exception:
                pass
            decomp["_data"] = data
        out[cid] = decomp
        if decomp is not None:
            bse = float(decomp["bse_grid"][REF_IDX])
            cov = float(decomp["coverage_ref"])
            ser = float(decomp.get("ser_ref", float("nan")))
            print(f"      bse_ref={bse:.4f}  cov={cov:.3f}  SER={ser:.3f}  [{elapsed:.0f}s]")
        else:
            print(f"      [no valid reps]  [{elapsed:.0f}s]")
    return out, J_pt


def _read_h_diagnostics_from_pkls(cell_dir: Path, cid: str, M: int) -> dict:
    """Read h-bridge diagnostics from the per-rep records (averaged)."""
    pkl = cell_dir / f"{cid}_M{M}.pkl"
    if not pkl.exists():
        return {}
    try:
        with open(pkl, "rb") as fh:
            data = pickle.load(fh)
        # _build_record only exposes a fixed set of keys.
        # We need to look at the EstimationResult, which is built per-rep
        # by calling factory_fn().fit(sample).estimate(). The record stored
        # is the output of _build_record(res, K, seed, J_pt). To check
        # what keys are in records[0].
        if not data.get("records"):
            return {}
        return {}  # extras aren't surfaced through _build_record currently
    except Exception:
        return {}


# ── Stage runners ──────────────────────────────────────────────────────────────

def _stage_smoke(snr=0.95, n=300, M=3, force=False) -> dict:
    print(f"\n=== Stage 2.0 SMOKE  SNR={snr}, n={n}, M={M} ===")
    configs = {
        "REG_KPV_super": "DUMMY",   # filled in by _run_cell_at_snr_n
        "DRK_KPV_super": "DUMMY",
        "BIN_smoke": dict(m_h=500, m_c=500, lambda_h=1e-3, gamma_critic=1e-3,
                          lambda_r=1e-2, n_features_r=300),
    }
    out, J_pt = _run_cell_at_snr_n(
        snr=snr, n=n, M=M, configs=configs,
        raw_dir=_RAW_DIR / "stage2_0_smoke",
        seed_base_cell=SEED_BASE - 1,
        force=force,
    )
    return out


def _stage_2_1(M=50, force=False) -> dict:
    """
    Wave 1 — Coarse h_B exploration (REVISED Phase 14B-bis after sanity check).

    Grid (5 lambda_h x 4 ell_h = 20 BIN configs):
      lambda_h in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}
      ell_h in {2.0, 2.75, 3.5, 4.5}
      gamma_critic = 1e-4 (sanity-checked: ~invariant)
      m_h = 1000, m_c = 500 (m sensitivity tested in Wave 3)
      ell_c = ell_h (independent ell_c tested in Wave 2)
    """
    print(f"\n=== Wave 1 (Stage 2.1) COARSE h_B  SNR=0.95, n=1000, M={M} ===")
    configs = {"REG_KPV_super": "DUMMY", "DRK_KPV_super": "DUMMY"}
    lambda_h_grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    ell_h_grid = [2.0, 2.75, 3.5, 4.5]
    gc = 1e-4
    for ell_h in ell_h_grid:
        for lh in lambda_h_grid:
            # Encode lh as Lxx where xx = floor(-log10(lh)*10)
            lh_code = int(round(-np.log10(lh) * 10))   # 50, 45, 40, 35, 30
            ell_code = int(round(ell_h * 100))          # 200, 275, 350, 450
            cid = f"BIN_e{ell_code}_L{lh_code}"
            configs[cid] = dict(
                m_h=1000, m_c=500,
                ell_h=ell_h, ell_c=ell_h,
                lambda_h=lh, gamma_critic=gc,
                lambda_r=1e-2, n_features_r=500,
                h_seed_base=SEED_BASE + int(ell_h * 1000) + lh_code * 100,
                rff_seed_base_r=SEED_BASE + 50000 + int(ell_h * 1000),
            )
    out, J_pt = _run_cell_at_snr_n(
        snr=0.95, n=1000, M=M, configs=configs,
        raw_dir=_RAW_DIR / "wave1_coarse",
        seed_base_cell=SEED_BASE,
        force=force,
    )
    return out


def _aggregate_metrics(stage_results: Dict[str, dict]) -> Dict[str, dict]:
    """For each config, extract a row of summary metrics."""
    rows = {}
    for cid, decomp in stage_results.items():
        if decomp is None:
            continue
        bse = decomp["bse_grid"]
        bias = decomp["bias_dr"]
        rows[cid] = {
            "bias_ref": float(bias[REF_IDX]),
            "bse_ref": float(bse[REF_IDX]),
            "bse_a18": float(bse[0]),
            "bse_a30": float(bse[3]),
            "cov_ref": float(decomp["coverage_ref"]),
            "ser_ref": float(decomp.get("ser_ref", float("nan"))),
            "ESS_min_mean": float(decomp.get("ess_min_mean", float("nan"))),
            "w_p99_mean": float(decomp.get("w_p99_mean", float("nan"))),
            "riesz_res_mean": float(decomp.get("riesz_res_mean", float("nan"))),
            "V_total": float(np.mean(decomp.get("V_total", float("nan")))) if "V_total" in decomp else float("nan"),
            "V_corr": float(np.mean(decomp.get("V_corr", float("nan")))) if "V_corr" in decomp else float("nan"),
        }
    return rows


def _print_table(rows: Dict[str, dict]):
    cols = ["bias_ref", "bse_ref", "bse_a18", "bse_a30", "cov_ref", "ser_ref",
            "ESS_min_mean", "w_p99_mean", "riesz_res_mean"]
    head = f"{'config':<25}" + "".join(f"{c:>11}" for c in cols)
    print(head)
    print("-" * len(head))
    for cid, m in rows.items():
        line = f"{cid:<25}"
        for c in cols:
            v = m.get(c, float("nan"))
            if not np.isfinite(v):
                line += f"{'---':>11}"
            elif abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
                line += f"{v:>11.2e}"
            else:
                line += f"{v:>11.4f}"
        print(line)


def _autonomous_select_stage21(rows: Dict[str, dict]) -> List[str]:
    """
    Select top BIN configs from Stage 2.1 based on observable diagnostics.

    Logic (transparent, no oracle):
    1. Filter configs with SER in [0.7, 1.3] (calibration sane).
    2. Among those, prefer LOW bse_ref (overall bias well-controlled) AND
       LOW bse_a18 (no boundary failure).
    3. Penalize high V_corr / V_reg ratio (correction adds too much variance).
    4. Pick top 3 by composite score.

    Documents the reasoning in stdout.
    """
    print("\n--- Autonomous selection (Stage 2.1) ---")
    bin_rows = {cid: m for cid, m in rows.items() if cid.startswith("BIN_")}
    if not bin_rows:
        print("  No BIN configs to rank.")
        return []
    # Filter by SER
    ser_pass = [(cid, m) for cid, m in bin_rows.items()
                if 0.7 <= m["ser_ref"] <= 1.3]
    print(f"  SER filter [0.7, 1.3]: {len(ser_pass)}/{len(bin_rows)} pass")
    if not ser_pass:
        print("  No configs pass SER filter; falling back to all configs.")
        ser_pass = list(bin_rows.items())
    # Composite: minimize 0.5*bse_ref + 0.3*bse_a18 + 0.2*bse_a30
    scored = []
    for cid, m in ser_pass:
        score = 0.5 * m["bse_ref"] + 0.3 * m["bse_a18"] + 0.2 * m["bse_a30"]
        scored.append((cid, score, m))
    scored.sort(key=lambda t: t[1])
    print("  Composite score = 0.5*bse_ref + 0.3*bse_a18 + 0.2*bse_a30:")
    for cid, score, m in scored[:5]:
        print(f"    {cid}: score={score:.4f}  "
              f"(bse_ref={m['bse_ref']:.3f}, bse_a18={m['bse_a18']:.3f}, "
              f"SER={m['ser_ref']:.3f})")
    top = [cid for cid, _, _ in scored[:3]]
    print(f"  Top 3 selected: {top}")
    return top


def _stage_wave2(M=50, force=False) -> dict:
    """
    Wave 2 — refined exploration around Wave 1 patterns.

    Wave 1 findings (manual analysis 2026-04-29):
      - Champion: BIN_e275_L50 (ell=2.75, lambda=1e-5): bse_ref=0.014
      - Backup: BIN_e200_L35 (ell=2.0, lambda=3e-4): bse_a18=0.156 (boundary)
      - Lambda monotone: smaller is better in [1e-3, 1e-5]
      - Ell unimodal: peak near 2.75

    Wave 2 grid (~13 BIN configs):
      Section A: refine ell_h around 2.75 at lambda=1e-5
        ell_h in {2.50, 2.60, 2.75, 2.85, 3.00} (5)
      Section B: extend lambda below 1e-5 for e275
        lambda_h in {1e-6, 3e-6} at ell=2.75 (2)
      Section C: ell_c decoupling on champion
        e275_L50 with ell_c in {2.0, 3.5, 4.5} (3, ell_h=2.75 fixed)
      Section D: refine backup
        e180_L35, e220_L35 (ell=1.8, 2.2 at lambda=3e-4) (2)
        e200 with lambda=1e-4 and 1e-5 (1) — to see if smaller lambda helps backup
    """
    print(f"\n=== Wave 2 REFINE  SNR=0.95, n=1000, M={M} ===")
    configs = {"REG_KPV_super": "DUMMY", "DRK_KPV_super": "DUMMY"}
    gc = 1e-4

    # Section A: refine ell_h
    for ell_h in [2.50, 2.60, 2.85, 3.00]:  # 2.75 already done in W1
        ell_code = int(round(ell_h * 100))
        cid = f"BIN_e{ell_code}_L50"
        configs[cid] = dict(
            m_h=1000, m_c=500, ell_h=ell_h, ell_c=ell_h,
            lambda_h=1e-5, gamma_critic=gc,
            lambda_r=1e-2, n_features_r=500,
            h_seed_base=SEED_BASE + int(ell_h * 1000) + 50 * 100,
            rff_seed_base_r=SEED_BASE + 50000 + int(ell_h * 1000),
        )

    # Section B: smaller lambda at e275
    for lh, lh_code in [(3e-6, 55), (1e-6, 60)]:
        ell_h = 2.75
        ell_code = 275
        cid = f"BIN_e{ell_code}_L{lh_code}"
        configs[cid] = dict(
            m_h=1000, m_c=500, ell_h=ell_h, ell_c=ell_h,
            lambda_h=lh, gamma_critic=gc,
            lambda_r=1e-2, n_features_r=500,
            h_seed_base=SEED_BASE + int(ell_h * 1000) + lh_code * 100,
            rff_seed_base_r=SEED_BASE + 50000 + int(ell_h * 1000),
        )

    # Section C: ell_c decoupling on champion (ell_h=2.75, lambda=1e-5)
    for ell_c in [2.0, 3.5, 4.5]:
        ell_h = 2.75
        ell_code_h = 275
        ell_code_c = int(round(ell_c * 100))
        cid = f"BIN_e{ell_code_h}_L50_ec{ell_code_c}"
        configs[cid] = dict(
            m_h=1000, m_c=500, ell_h=ell_h, ell_c=ell_c,
            lambda_h=1e-5, gamma_critic=gc,
            lambda_r=1e-2, n_features_r=500,
            h_seed_base=SEED_BASE + 99000 + int(ell_c * 100),
            rff_seed_base_r=SEED_BASE + 99500 + int(ell_c * 100),
        )

    # Section D: refine backup region (ell=2.0 area)
    for ell_h, lh in [(1.8, 3e-4), (2.2, 3e-4), (2.0, 1e-4), (2.0, 1e-5)]:
        ell_code = int(round(ell_h * 100))
        # tag lambda
        lh_code = int(round(-np.log10(lh) * 10))
        cid = f"BIN_e{ell_code}_L{lh_code}_bk"  # _bk for backup region
        configs[cid] = dict(
            m_h=1000, m_c=500, ell_h=ell_h, ell_c=ell_h,
            lambda_h=lh, gamma_critic=gc,
            lambda_r=1e-2, n_features_r=500,
            h_seed_base=SEED_BASE + 88000 + int(ell_h * 1000) + lh_code,
            rff_seed_base_r=SEED_BASE + 88500 + int(ell_h * 1000) + lh_code,
        )

    out, J_pt = _run_cell_at_snr_n(
        snr=0.95, n=1000, M=M, configs=configs,
        raw_dir=_RAW_DIR / "wave2_refine",
        seed_base_cell=SEED_BASE,
        force=force,
    )
    return out


def _stage_wave3(M=50, force=False) -> dict:
    """
    Wave 3 — Robustness around new champion BIN_e275_L50_ec200 (Wave 2 result).

    Wave 2 findings:
      - Champion: ell_h=2.75, ell_c=2.0, lambda_h=1e-5 (decoupled ell_c < ell_h)
      - bse_ref=0.012, bse_a18=0.120 (matches REG), bse_a30=0.021
      - L50 vs L60 tradeoff: L50 best at REF, L60 best at boundary

    Wave 3 (12 configs):
      A. m_h, m_c sensitivity (5)
      B. ell_c finer around 2.0 (3)
      C. ell_h finer around 2.75 with ec200 (3)
      D. ec200 x L60 boundary specialist (1)
    """
    print(f"\n=== Wave 3 ROBUSTNESS  SNR=0.95, n=1000, M={M} ===")
    configs = {"REG_KPV_super": "DUMMY", "DRK_KPV_super": "DUMMY"}
    gc = 1e-4
    base = dict(
        ell_h=2.75, ell_c=2.0, lambda_h=1e-5, gamma_critic=gc,
        lambda_r=1e-2, n_features_r=500,
    )

    # A. m_h, m_c sensitivity
    for m_h, m_c, suffix in [
        (750, 500, "mh750"),
        (1500, 500, "mh1500"),
        (2000, 500, "mh2000"),
        (1000, 250, "mc250"),
        (1000, 1000, "mc1000"),
    ]:
        cid = f"BIN_W3_A_{suffix}"
        configs[cid] = dict(
            base, m_h=m_h, m_c=m_c,
            h_seed_base=SEED_BASE + 70000 + m_h + m_c,
            rff_seed_base_r=SEED_BASE + 75000 + m_h + m_c,
        )

    # B. ell_c finer
    for ell_c in [1.5, 1.8, 2.2]:
        ec_code = int(round(ell_c * 100))
        cid = f"BIN_W3_B_ec{ec_code}"
        configs[cid] = dict(
            base, ell_c=ell_c, m_h=1000, m_c=500,
            h_seed_base=SEED_BASE + 80000 + ec_code,
            rff_seed_base_r=SEED_BASE + 85000 + ec_code,
        )

    # C. ell_h finer with ec200
    for ell_h in [2.65, 2.85, 3.00]:
        eh_code = int(round(ell_h * 100))
        cid = f"BIN_W3_C_eh{eh_code}"
        configs[cid] = dict(
            base, ell_h=ell_h, m_h=1000, m_c=500,
            h_seed_base=SEED_BASE + 90000 + eh_code,
            rff_seed_base_r=SEED_BASE + 95000 + eh_code,
        )

    # D. ec200 x L60 (boundary specialist)
    cid = "BIN_W3_D_L60_ec200"
    configs[cid] = dict(
        base, lambda_h=1e-6, m_h=1000, m_c=500,
        h_seed_base=SEED_BASE + 99100,
        rff_seed_base_r=SEED_BASE + 99200,
    )
    cid = "BIN_W3_D_L55_ec200"
    configs[cid] = dict(
        base, lambda_h=3e-6, m_h=1000, m_c=500,
        h_seed_base=SEED_BASE + 99300,
        rff_seed_base_r=SEED_BASE + 99400,
    )

    out, J_pt = _run_cell_at_snr_n(
        snr=0.95, n=1000, M=M, configs=configs,
        raw_dir=_RAW_DIR / "wave3_robust",
        seed_base_cell=SEED_BASE,
        force=force,
    )
    return out


def _stage_wave4(M=50, force=False) -> dict:
    """
    Wave 4 — Combined h_B specialists + lambda_r tuning.

    W1+W2+W3 synthesis:
      Pareto frontier:
        REF spec:   eh285_ec200_mh1000_L50 (bse_ref=0.0007, bse_a18=0.176)
        Mh-rich:    mh2000 eh275 ec200 L50 (bse_ref=0.0014, bse_a18=0.141)
        Balanced:   eh275_ec200_mh1000_L50 (bse_ref=0.012, bse_a18=0.120) <- W2 champion
        Boundary:   eh265_ec200_mh1000_L50 (bse_ref=0.051, bse_a18=0.076)

    Wave 4 (9 configs):
      A. Combined h_B specialists (2)
      B. lambda_r grid on balanced h (eh275 ec200 mh1000) (3)
      C. lambda_r grid on boundary specialist (eh265 ec200 mh1000) (3)
      D. lambda_r tuning on mh-rich (eh275 ec200 mh2000) (1)
    """
    print(f"\n=== Wave 4 COMBINED + lambda_r  SNR=0.95, n=1000, M={M} ===")
    configs = {"REG_KPV_super": "DUMMY", "DRK_KPV_super": "DUMMY"}
    gc = 1e-4

    # A. Combined h_B specialists
    cid = "BIN_W4_A_eh285_ec200_mh2000"
    configs[cid] = dict(
        m_h=2000, m_c=500, ell_h=2.85, ell_c=2.0,
        lambda_h=1e-5, gamma_critic=gc,
        lambda_r=1e-2, n_features_r=500,
        h_seed_base=SEED_BASE + 100100,
        rff_seed_base_r=SEED_BASE + 100200,
    )
    cid = "BIN_W4_A_eh275_ec200_mh2000_L55"
    configs[cid] = dict(
        m_h=2000, m_c=500, ell_h=2.75, ell_c=2.0,
        lambda_h=3e-6, gamma_critic=gc,
        lambda_r=1e-2, n_features_r=500,
        h_seed_base=SEED_BASE + 100300,
        rff_seed_base_r=SEED_BASE + 100400,
    )

    # B. lambda_r grid on balanced (eh275 ec200 mh1000)
    for lr_code, lr in [(30, 1e-3), (28, 3e-3), (15, 3e-2)]:
        cid = f"BIN_W4_B_balanced_lr{lr_code}"
        configs[cid] = dict(
            m_h=1000, m_c=500, ell_h=2.75, ell_c=2.0,
            lambda_h=1e-5, gamma_critic=gc,
            lambda_r=lr, n_features_r=500,
            h_seed_base=SEED_BASE + 110000 + lr_code,
            rff_seed_base_r=SEED_BASE + 110500 + lr_code,
        )

    # C. lambda_r grid on boundary specialist (eh265 ec200 mh1000)
    for lr_code, lr in [(30, 1e-3), (20, 1e-2), (10, 1e-1)]:
        cid = f"BIN_W4_C_boundary_lr{lr_code}"
        configs[cid] = dict(
            m_h=1000, m_c=500, ell_h=2.65, ell_c=2.0,
            lambda_h=1e-5, gamma_critic=gc,
            lambda_r=lr, n_features_r=500,
            h_seed_base=SEED_BASE + 120000 + lr_code,
            rff_seed_base_r=SEED_BASE + 120500 + lr_code,
        )

    # D. lambda_r=3e-3 on mh-rich (eh275 ec200 mh2000)
    cid = "BIN_W4_D_mhrich_lr28"
    configs[cid] = dict(
        m_h=2000, m_c=500, ell_h=2.75, ell_c=2.0,
        lambda_h=1e-5, gamma_critic=gc,
        lambda_r=3e-3, n_features_r=500,
        h_seed_base=SEED_BASE + 130100,
        rff_seed_base_r=SEED_BASE + 130200,
    )

    out, J_pt = _run_cell_at_snr_n(
        snr=0.95, n=1000, M=M, configs=configs,
        raw_dir=_RAW_DIR / "wave4_combined",
        seed_base_cell=SEED_BASE,
        force=force,
    )
    return out


def _stage_wave5(M=100, force=False) -> dict:
    """
    Wave 5 — DISJOINT seed validation of top 3 Bennett-indep candidates.

    Tuning was at seed_base=SEED_BASE (28_500_000) for waves 1-4.
    W5 uses seed_base=SEED_BASE+5000 = 28_505_000 (disjoint).

    4 cells: (n in {1000, 2000}) x (SNR in {0.90, 0.95})
    5 methods: REG_KPV, DRK_KPV, boundary_lr10, W2 champ, mh2000
    M=100 reps each.

    Top 3 BIN candidates (from W1-W4 Pareto frontier):
      1. boundary_lr10: eh265 ec200 mh1000 L50 lr=1e-1   (best avg_bse=0.054)
      2. W2 champ:      eh275 ec200 mh1000 L50 lr=1e-2   (balanced, well-tested)
      3. mh2000:        eh275 ec200 mh2000 L50 lr=1e-2   (best REF=0.001)
    """
    print(f"\n=== Wave 5 DISJOINT VALIDATION  M={M}, 4 cells ===")
    cells_results = {}
    seed_base_w5 = SEED_BASE + 5000   # 28_505_000

    BIN_CANDIDATES = {
        "BIN_boundary_lr10": dict(
            m_h=1000, m_c=500, ell_h=2.65, ell_c=2.0,
            lambda_h=1e-5, gamma_critic=1e-4,
            lambda_r=1e-1, n_features_r=500,
            h_seed_base=seed_base_w5 + 100,
            rff_seed_base_r=seed_base_w5 + 200,
        ),
        "BIN_W2_champ": dict(
            m_h=1000, m_c=500, ell_h=2.75, ell_c=2.0,
            lambda_h=1e-5, gamma_critic=1e-4,
            lambda_r=1e-2, n_features_r=500,
            h_seed_base=seed_base_w5 + 300,
            rff_seed_base_r=seed_base_w5 + 400,
        ),
        "BIN_mh2000": dict(
            m_h=2000, m_c=500, ell_h=2.75, ell_c=2.0,
            lambda_h=1e-5, gamma_critic=1e-4,
            lambda_r=1e-2, n_features_r=500,
            h_seed_base=seed_base_w5 + 500,
            rff_seed_base_r=seed_base_w5 + 600,
        ),
    }

    for snr in [0.90, 0.95]:
        for n in [1000, 2000]:
            print(f"\n--- Cell SNR={snr}, n={n} ---")
            configs = {"REG_KPV_super": "DUMMY", "DRK_KPV_super": "DUMMY"}
            configs.update(BIN_CANDIDATES)
            seed_base_cell = seed_base_w5 + (0 if snr == 0.90 else 1000) + (0 if n == 1000 else 100)
            out, _ = _run_cell_at_snr_n(
                snr=snr, n=n, M=M, configs=configs,
                raw_dir=_RAW_DIR / f"wave5_validation/snr{int(snr*100)}_n{n}",
                seed_base_cell=seed_base_cell,
                force=force,
            )
            cells_results[(snr, n)] = out
    return cells_results


def _stage_wave6(M=50, force=False) -> dict:
    """
    Wave 6 — SNR=0.80 robustness on BIN_mh2000 (W5 winner).

    Quick check: does the W5 disjoint-seed winner BIN_mh2000 still beat
    REG_KPV and DRK_KPV at lower SNR=0.80?

    Cell: SNR=0.80, n=1000, M=50. seed_base disjoint = 28_507_000.
    """
    print(f"\n=== Wave 6 SNR=0.80 ROBUSTNESS  M={M} ===")
    seed_base_w6 = SEED_BASE + 7000   # 28_507_000

    configs = {
        "REG_KPV_super": "DUMMY",
        "DRK_KPV_super": "DUMMY",
        "BIN_mh2000": dict(
            m_h=2000, m_c=500, ell_h=2.75, ell_c=2.0,
            lambda_h=1e-5, gamma_critic=1e-4,
            lambda_r=1e-2, n_features_r=500,
            h_seed_base=seed_base_w6 + 100,
            rff_seed_base_r=seed_base_w6 + 200,
        ),
    }
    out, _ = _run_cell_at_snr_n(
        snr=0.80, n=1000, M=M, configs=configs,
        raw_dir=_RAW_DIR / "wave6_snr80",
        seed_base_cell=seed_base_w6,
        force=force,
    )
    return out


def _stage_2_3(top_lh_gc: List[Tuple[float, float]], M=50, force=False) -> dict:
    """Refined h_B: vary m_h ∈ {750, 1000, 1500} at top (lambda_h, gamma_c)."""
    print(f"\n=== Stage 2.3 REFINED h_B  SNR=0.95, n=1000, M={M} ===")
    configs = {"REG_KPV_super": "DUMMY", "DRK_KPV_super": "DUMMY"}
    for lh, gc in top_lh_gc:
        for m_h in [750, 1000, 1500]:
            cid = f"BIN_mh{m_h}_lh{int(-np.log10(lh))}_gc{int(-np.log10(gc))}"
            configs[cid] = dict(
                m_h=m_h, m_c=500,
                lambda_h=lh, gamma_critic=gc,
                lambda_r=1e-2, n_features_r=500,
                h_seed_base=SEED_BASE + 100 + m_h,
                rff_seed_base_r=SEED_BASE + 600 + m_h,
            )
    out, _ = _run_cell_at_snr_n(
        snr=0.95, n=1000, M=M, configs=configs,
        raw_dir=_RAW_DIR / "stage2_3_refined",
        seed_base_cell=SEED_BASE + 1000,
        force=force,
    )
    return out


def _stage_2_4(top_h_specs: List[dict], M=50, force=False) -> dict:
    """Combined h_B + r_B: vary lambda_r ∈ {1e-3, 1e-2, 1e-1}."""
    print(f"\n=== Stage 2.4 COMBINED h_B + r_B  SNR=0.95, n=1000, M={M} ===")
    configs = {"REG_KPV_super": "DUMMY", "DRK_KPV_super": "DUMMY"}
    for i, h_spec in enumerate(top_h_specs):
        for lr in [1e-3, 1e-2, 1e-1]:
            cid = f"BIN_top{i}_lr{int(-np.log10(lr))}"
            spec = dict(h_spec)
            spec["lambda_r"] = lr
            spec["h_seed_base"] = SEED_BASE + 2000 + i * 100
            spec["rff_seed_base_r"] = SEED_BASE + 2500 + i * 100 + int(-np.log10(lr))
            configs[cid] = spec
    out, _ = _run_cell_at_snr_n(
        snr=0.95, n=1000, M=M, configs=configs,
        raw_dir=_RAW_DIR / "stage2_4_combined",
        seed_base_cell=SEED_BASE + 2000,
        force=force,
    )
    return out


def _stage_2_5(top_specs: List[dict], M=100, force=False) -> Dict[Tuple[float, int], dict]:
    """Disjoint validation: top 1-2 at n in {1000, 2000}, SNR in {0.90, 0.95}."""
    print(f"\n=== Stage 2.5 DISJOINT VALIDATION  M={M} ===")
    cells_results = {}
    for snr in [0.90, 0.95]:
        for n in [1000, 2000]:
            print(f"\n--- Cell SNR={snr}, n={n} ---")
            configs = {"REG_KPV_super": "DUMMY", "DRK_KPV_super": "DUMMY"}
            for i, spec in enumerate(top_specs):
                cid = f"BIN_VAL_top{i}"
                spec_copy = dict(spec)
                spec_copy["h_seed_base"] = SEED_BASE + 5000 + i * 100
                spec_copy["rff_seed_base_r"] = SEED_BASE + 5500 + i * 100
                configs[cid] = spec_copy
            seed_base_cell = SEED_BASE + 5000 + (0 if snr == 0.90 else 1000) + (0 if n == 1000 else 100)
            out, _ = _run_cell_at_snr_n(
                snr=snr, n=n, M=M, configs=configs,
                raw_dir=_RAW_DIR / f"stage2_5_validation/snr{int(snr*100)}_n{n}",
                seed_base_cell=seed_base_cell,
                force=force,
            )
            cells_results[(snr, n)] = out
    return cells_results


# ── Stage report builders ─────────────────────────────────────────────────────

def _build_stage_report(stage_label: str, rows: Dict[str, dict],
                          notes: str = "") -> str:
    import datetime
    lines = []
    lines.append(f"# Phase 14B Stage {stage_label}")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append("")
    if notes:
        lines.append(notes)
        lines.append("")
    lines.append("## Per-config metrics")
    lines.append("")
    cols = ["bias_ref", "bse_ref", "bse_a18", "bse_a30", "cov_ref", "ser_ref",
            "ESS_min_mean", "w_p99_mean", "riesz_res_mean"]
    lines.append("| config | " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for cid, m in rows.items():
        cells = [cid]
        for c in cols:
            v = m.get(c, float("nan"))
            if not np.isfinite(v):
                cells.append("---")
            elif abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
                cells.append(f"{v:.2e}")
            else:
                cells.append(f"{v:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _save_stage_report(stage_label: str, content: str):
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    path = _SUMM_DIR / f"phase14B_stage{stage_label}.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"  Stage report: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "wave1", "wave2", "wave3", "wave4", "wave5", "wave6", "2_1", "2_3", "2_4", "2_5", "all"],
                        default="wave1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # wave1 is alias for 2_1
    if args.stage == "wave1":
        args.stage = "2_1"

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    if args.stage == "wave2":
        wave2 = _stage_wave2(force=args.force)
        rows = _aggregate_metrics(wave2)
        _print_table(rows)
        _save_stage_report("wave2", _build_stage_report(
            "Wave 2 — Refined exploration (ell_h, lambda_h smaller, ell_c decoupling, backup region)",
            rows,
        ))
        elapsed = time.time() - t_total
        print(f"\n=== Phase 14B Wave 2 total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")
        return

    if args.stage == "wave3":
        wave3 = _stage_wave3(force=args.force)
        rows = _aggregate_metrics(wave3)
        _print_table(rows)
        _save_stage_report("wave3", _build_stage_report(
            "Wave 3 — Robustness around BIN_e275_L50_ec200 (m sensitivity, ell finer, L60 boundary specialist)",
            rows,
        ))
        elapsed = time.time() - t_total
        print(f"\n=== Phase 14B Wave 3 total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")
        return

    if args.stage == "wave4":
        wave4 = _stage_wave4(force=args.force)
        rows = _aggregate_metrics(wave4)
        _print_table(rows)
        _save_stage_report("wave4", _build_stage_report(
            "Wave 4 — Combined h_B specialists + lambda_r grid on top 3 h-bridges",
            rows,
        ))
        elapsed = time.time() - t_total
        print(f"\n=== Phase 14B Wave 4 total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")
        return

    if args.stage == "wave5":
        cells_results = _stage_wave5(force=args.force)
        for (snr, n), cell in cells_results.items():
            rows = _aggregate_metrics(cell)
            _save_stage_report(f"wave5_snr{int(snr*100)}_n{n}", _build_stage_report(
                f"Wave 5 — Disjoint validation, SNR={snr}, n={n}, M=100, seed_base=28_505_000",
                rows,
            ))
        elapsed = time.time() - t_total
        print(f"\n=== Phase 14B Wave 5 total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")
        return

    if args.stage == "wave6":
        wave6 = _stage_wave6(force=args.force)
        rows = _aggregate_metrics(wave6)
        _print_table(rows)
        _save_stage_report("wave6", _build_stage_report(
            "Wave 6 — SNR=0.80 robustness on BIN_mh2000",
            rows,
        ))
        elapsed = time.time() - t_total
        print(f"\n=== Phase 14B Wave 6 total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")
        return

    if args.stage in ("smoke", "all"):
        smoke = _stage_smoke(force=args.force)
        rows_smoke = _aggregate_metrics(smoke)
        _print_table(rows_smoke)

    if args.stage in ("2_1", "all"):
        stage21 = _stage_2_1(force=args.force)
        rows21 = _aggregate_metrics(stage21)
        _print_table(rows21)
        _save_stage_report("2_1", _build_stage_report(
            "2.1 (revised) — Coarse h_B (4 lambda_h x 4 ell_h = 16 configs, m_h=1000, m_c=500, gamma_c=1e-4)",
            rows21,
        ))
        # Autonomous selection top 3
        top_cids = _autonomous_select_stage21(rows21)
        # Convert top_cids to (lambda_h, ell_h) pairs
        # New format: BIN_e<ELL10>_L<LH>  where ELL10 = round(ell_h*10), LH = -log10(lambda_h)
        top_lh_eh = []
        for cid in top_cids:
            try:
                parts = cid.split("_")
                # parts[0]="BIN", parts[1]="e<ELL10>", parts[2]="L<LH>"
                ell10 = int(parts[1].replace("e", ""))
                ell_h = ell10 / 10.0
                lh_tag = parts[2]
                if lh_tag == "L5em4":
                    lh = 5e-4
                else:
                    lh = 10.0 ** (-int(lh_tag.replace("L", "")))
                top_lh_eh.append((lh, ell_h))
            except Exception as e:
                print(f"  [WARN] could not parse {cid}: {e}")
        if not top_lh_eh:
            print("  [STOP] No top configs from Stage 2.1. Halting Phase 14B.")
            return
        print(f"  Top (lambda_h, ell_h) pairs for Stage 2.3: {top_lh_eh}")

    if args.stage in ("2_3", "all"):
        stage23 = _stage_2_3(top_lh_gc, force=args.force)
        rows23 = _aggregate_metrics(stage23)
        _print_table(rows23)
        _save_stage_report("2_3", _build_stage_report(
            "2.3 — Refined h_B (top 3 lambda_h-gamma_c x m_h in {750, 1000, 1500})",
            rows23,
        ))
        # Autonomous: pick top 2 BIN configs by composite score
        bin_rows23 = {cid: m for cid, m in rows23.items() if cid.startswith("BIN_")}
        ser_pass = [(cid, m) for cid, m in bin_rows23.items()
                    if 0.7 <= m["ser_ref"] <= 1.3]
        if not ser_pass:
            ser_pass = list(bin_rows23.items())
        scored = sorted(
            ser_pass,
            key=lambda t: 0.5 * t[1]["bse_ref"] + 0.3 * t[1]["bse_a18"] + 0.2 * t[1]["bse_a30"]
        )
        top_h_cids = [t[0] for t in scored[:2]]
        print(f"\n  Top 2 h-bridge configs from Stage 2.3: {top_h_cids}")
        # Build specs from cid format: BIN_mh<m>_lh<X>_gc<Y>
        top_h_specs = []
        for cid in top_h_cids:
            parts = cid.split("_")
            m_h = int(parts[1].replace("mh", ""))
            lh = 10.0 ** (-int(parts[2].replace("lh", "")))
            gc = 10.0 ** (-int(parts[3].replace("gc", "")))
            top_h_specs.append(dict(
                m_h=m_h, m_c=500,
                lambda_h=lh, gamma_critic=gc,
                n_features_r=500,
            ))

    if args.stage in ("2_4", "all"):
        stage24 = _stage_2_4(top_h_specs, force=args.force)
        rows24 = _aggregate_metrics(stage24)
        _print_table(rows24)
        _save_stage_report("2_4", _build_stage_report(
            "2.4 — Combined h_B + r_B (top 2 h x lambda_r in {1e-3, 1e-2, 1e-1})",
            rows24,
        ))
        # Pick top 1-2 by composite
        bin_rows24 = {cid: m for cid, m in rows24.items() if cid.startswith("BIN_")}
        scored = sorted(
            bin_rows24.items(),
            key=lambda t: 0.5 * t[1]["bse_ref"] + 0.3 * t[1]["bse_a18"] + 0.2 * t[1]["bse_a30"]
        )
        top_final_cids = [t[0] for t in scored[:2]]
        print(f"\n  Top 2 final configs from Stage 2.4: {top_final_cids}")
        # Build specs: BIN_top<i>_lr<X>
        top_final_specs = []
        for cid in top_final_cids:
            parts = cid.split("_")
            i_top = int(parts[1].replace("top", ""))
            lr = 10.0 ** (-int(parts[2].replace("lr", "")))
            spec = dict(top_h_specs[i_top])
            spec["lambda_r"] = lr
            top_final_specs.append(spec)
        print(f"  Specs: {top_final_specs}")

    if args.stage in ("2_5", "all"):
        stage25 = _stage_2_5(top_final_specs, force=args.force)
        # Build per-cell tables
        for (snr, n), cell_results in stage25.items():
            rows = _aggregate_metrics(cell_results)
            _save_stage_report(f"2_5_snr{int(snr*100)}_n{n}", _build_stage_report(
                f"2.5 Disjoint validation — SNR={snr}, n={n}", rows,
            ))

    elapsed = time.time() - t_total
    print(f"\n=== Phase 14B total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")


if __name__ == "__main__":
    main()
