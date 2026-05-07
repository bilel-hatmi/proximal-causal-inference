"""
Phase 12B -- Bennett-lite RFF Overnight Mission.

Replaces the polynomial Riesz of Phase 12A (failed: Case C) with random Fourier
features (RFF) for richer function approximation of r_pi.

Score formula (same as Phase 12A):
    J_hat_B(a) = mean_i [ T_pi_a h_hat(W_i)
                          + r_hat_pi_a(Z_i, A_i) * (Y_i - h_hat(W_i, A_i)) ]

Phase 12A failure mode (polynomial Riesz):
  - riesz_residual = 9.8e+01 .. 2.3e+10  (criterion < 1e-3 failed)
  - SER = 0.24 .. 0.83                   (criterion [0.8, 1.2] failed)
  - bias_DR worse than REG for ALL configs (correction wrong direction)
  - "Cas A signal" was a false positive (variance inflation x72)

Phase 12B question:
  Does RFF (richer feature class) save the functional-first approach?
  Or does the direct Riesz fini-feature still fail vs KPV pipeline on DGP2?

Stages
------
  quick   : n=300, M=3, ~3 configs RFF + REG + DRK   (smoke test)
  A       : n=1000, M=50, 12 RFF configs + REG + DRK (screening)
  A2      : top 3 Stage A configs x H_tuned          (Case C check)
  B       : stabilized Riesz refinement (top 3)      (Stage B optional)
  D       : validation disjointe top 2, n in {1000, 2000}, M=100
  E       : head-to-head best Bennett vs DRK_Hsuper, n=2000, M=200

Anti small-M rule: NEVER promote winner at M<=100 without M=200 disjoint validation.

seed_base
---------
  Stage A  = 24_000_000
  Stage A2 = 24_200_000
  Stage B  = 24_300_000
  Stage D  = 24_500_000
  Stage E  = 24_900_000
(All disjoint from Phase 12A 23M and from earlier 16M..22M phases.)

Usage
-----
  python -m simulations.archive.utilities.bennett_rff_overnight --stage quick
  python -m simulations.archive.utilities.bennett_rff_overnight --stage A
  python -m simulations.archive.utilities.bennett_rff_overnight --stage A2
  python -m simulations.archive.utilities.bennett_rff_overnight --stage B
  python -m simulations.archive.utilities.bennett_rff_overnight --stage D
  python -m simulations.archive.utilities.bennett_rff_overnight --stage E
  python -m simulations.archive.utilities.bennett_rff_overnight --stage report

Output
------
  simulations/results/raw/bennett_rff/<stage>/<label>.pkl
  simulations/results/summaries/phase12B_bennett_rff_<stage>.md
  simulations/results/summaries/phase12B_bennett_rff_FINAL.md (final integrated)
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Thread control (before BLAS imports)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


# ── Paths ──────────────────────────────────────────────────────────────────────

_BASE_DIR  = Path(__file__).resolve().parents[2]
_RAW_BASE  = _BASE_DIR / "simulations" / "results" / "raw" / "bennett_rff"
_SUMM_DIR  = _BASE_DIR / "simulations" / "results" / "summaries"


# ── Constants ──────────────────────────────────────────────────────────────────

A_GRID       = np.array([1.8, 2.2, 2.6, 3.0])
K            = len(A_GRID)
REF_IDX      = 2           # a = 2.6
SNR          = 0.95
N_FOLDS      = 5
RANDOM_STATE = 42

# h-bridge HP anchors
SUPER_HP = dict(lambda_h=3e-5, ell_scale=3.5)   # KPV-super (benchmark)
TUNED_HP = dict(lambda_h=1e-4, ell_scale=2.5)   # KPV-tuned

# Seed base per stage (disjoint from all prior phases)
SEED_BASES = {
    "quick":  23_500_000,
    "A":      24_000_000,
    "A2":     24_200_000,
    "B":      24_300_000,
    "D":      24_500_000,
    "E":      24_900_000,
}


# ── Reuse utilities from Phase 12A pilot script ────────────────────────────────

from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
    _compute_J_policy_true, _build_record,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Factory functions
# ══════════════════════════════════════════════════════════════════════════════

def _make_reg_hsuper(h_KDE, *_):
    """Plug-in only (lambda_r=1e10 -> correction near 0)."""
    from simulations.methods.bennett_functional_dr import BennettFunctionalDR
    def factory():
        return BennettFunctionalDR(
            lambda_h=SUPER_HP["lambda_h"], ell_scale=SUPER_HP["ell_scale"],
            lambda_r=1e10, degree=2,
            feature_map_type="polynomial",
            n_folds=N_FOLDS, a_grid=A_GRID.tolist(),
            bandwidth=h_KDE, seed=RANDOM_STATE, ref_dose_index=REF_IDX,
        )
    return factory


def _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0):
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
    def factory():
        q_model = KPVPolicyBridgeQ(
            a_grid=A_GRID, h_KDE=h_KDE, lambda_Q=1e-3, clip=None,
        )
        return DRKernel(
            a_grid=A_GRID, q_model=q_model, bandwidth=h_KDE,
            n_folds=N_FOLDS, random_state=RANDOM_STATE,
            ref_dose_index=REF_IDX, cross_fit_q=True,
            bridge_kwargs=dict(
                lambda_1=SUPER_HP["lambda_h"], lambda_2=SUPER_HP["lambda_h"],
                ell_W=ell_W0 * SUPER_HP["ell_scale"],
                ell_A=ell_A0 * SUPER_HP["ell_scale"],
                ell_Z=ell_Z0 * SUPER_HP["ell_scale"],
            ),
        )
    return factory


def _make_bennett_rff(
    *,
    n_features: int,
    ell_scale_rff: float,
    lambda_r: float,
    h_KDE: float,
    h_lambda: float = 3e-5,
    h_ell_scale: float = 3.5,
    rff_seed_base: int = 0,
    stabilized: bool = False,
    lambda_stab: float = 1.0,
    gamma_critic: float = 1e-3,
):
    """RFF Bennett factory."""
    from simulations.methods.bennett_functional_dr import BennettFunctionalDR
    def factory():
        return BennettFunctionalDR(
            lambda_h=h_lambda, ell_scale=h_ell_scale,
            lambda_r=lambda_r,
            feature_map_type="rff",
            n_features=n_features,
            ell_scale_rff=ell_scale_rff,
            rff_seed_base=rff_seed_base,
            stabilized=stabilized,
            lambda_stab=lambda_stab,
            gamma_critic=gamma_critic,
            n_folds=N_FOLDS, a_grid=A_GRID.tolist(),
            bandwidth=h_KDE, seed=RANDOM_STATE, ref_dose_index=REF_IDX,
        )
    return factory


# ══════════════════════════════════════════════════════════════════════════════
#  Config grid builders
# ══════════════════════════════════════════════════════════════════════════════

def _stage_quick_configs(h_KDE, ell_W0, ell_A0, ell_Z0):
    """Smoke: REG, DRK, 1 RFF config."""
    return {
        "REG_Hsuper":    _make_reg_hsuper(h_KDE),
        "DRK_Hsuper":    _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0),
        "RFF128_e25_L1e2_Hs": _make_bennett_rff(
            n_features=128, ell_scale_rff=2.5, lambda_r=1e-2,
            h_KDE=h_KDE, rff_seed_base=20240,
        ),
    }


def _stage_A_configs(h_KDE, ell_W0, ell_A0, ell_Z0):
    """Stage A focused screening: REG + DRK + 12 RFF configs (3 nf × 2 ell × 2 lr)."""
    cfgs = {}
    cfgs["REG_Hsuper"] = _make_reg_hsuper(h_KDE)
    cfgs["DRK_Hsuper"] = _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0)

    # Grid: n_features in {100, 250, 500} x ell_scale_rff in {2.5, 3.5} x lambda_r in {1e-2, 1e-1}
    nf_grid = [100, 250, 500]
    ell_grid = [2.5, 3.5]
    lr_grid = [1e-2, 1e-1]
    for nf in nf_grid:
        for ells in ell_grid:
            for lr in lr_grid:
                lr_tag = f"L{lr:.0e}".replace("e+0", "e").replace("e-0", "em").replace("e+", "e").replace("e-", "em")
                # cleaner tag
                lr_tag = f"L{int(round(np.log10(lr) * -1)):d}" if lr < 1 else f"L{int(round(np.log10(lr))):d}"
                lr_tag = "L" + (f"{int(-np.log10(lr))}" if lr < 1 else f"+{int(np.log10(lr))}")
                # final clean tag
                if lr == 1e-2: lr_str = "L1e2"
                elif lr == 1e-1: lr_str = "L1e1"
                else: lr_str = f"L{lr:.0e}".replace(".", "").replace("+", "").replace("-", "")
                ell_str = f"e{int(round(ells*10))}"
                key = f"RFF{nf}_{ell_str}_{lr_str}_Hs"
                cfgs[key] = _make_bennett_rff(
                    n_features=nf, ell_scale_rff=ells, lambda_r=lr,
                    h_KDE=h_KDE,
                    rff_seed_base=20240 + nf * 17 + int(ells * 10) * 31 + int(-np.log10(lr)) * 53,
                )
    return cfgs


def _stage_A2_configs(h_KDE, ell_W0, ell_A0, ell_Z0, top_keys: List[str]):
    """Stage A2: top configs from A run with H_tuned bridge instead of H_super."""
    # top_keys are Stage A config IDs. Re-create with H_tuned.
    cfgs = {}
    cfgs["DRK_Hsuper"] = _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0)
    # Parse RFF settings from key e.g. "RFF250_e35_L1e2_Hs"
    for k in top_keys:
        # Format: RFF{nf}_e{ell10}_L{lr_tag}_Hs
        try:
            parts = k.split("_")
            nf = int(parts[0].replace("RFF", ""))
            ells = int(parts[1].replace("e", "")) / 10.0
            lr_tag = parts[2]
            if lr_tag == "L1e2": lr = 1e-2
            elif lr_tag == "L1e1": lr = 1e-1
            else: lr = 1e-2  # fallback
        except Exception:
            continue
        new_key = k.replace("_Hs", "_Ht")
        cfgs[new_key] = _make_bennett_rff(
            n_features=nf, ell_scale_rff=ells, lambda_r=lr,
            h_KDE=h_KDE,
            h_lambda=TUNED_HP["lambda_h"], h_ell_scale=TUNED_HP["ell_scale"],
            rff_seed_base=20242 + nf * 17 + int(ells * 10) * 31 + int(-np.log10(lr)) * 53,
        )
    return cfgs


def _stage_B_configs(h_KDE, ell_W0, ell_A0, ell_Z0, top_keys: List[str]):
    """Stage B: stabilized Riesz on top Stage A configs."""
    cfgs = {}
    cfgs["DRK_Hsuper"] = _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0)
    # 3 top × {lambda_stab in {0.1, 1.0, 10.0}} × {gamma_critic in {1e-3, 1e-2}}
    # Reduce: 3 top × 3 lambda_stab × 1 gamma_critic = 9 configs
    for k in top_keys[:3]:
        try:
            parts = k.split("_")
            nf = int(parts[0].replace("RFF", ""))
            ells = int(parts[1].replace("e", "")) / 10.0
            lr_tag = parts[2]
            if lr_tag == "L1e2": lr = 1e-2
            elif lr_tag == "L1e1": lr = 1e-1
            else: lr = 1e-2
        except Exception:
            continue
        for ls in [0.1, 1.0, 10.0]:
            ls_tag = f"S{int(ls*10):d}" if ls < 1 else f"S{int(ls):02d}"
            new_key = f"STAB_{ls_tag}_{k.replace('_Hs', '_Hs')}"
            cfgs[new_key] = _make_bennett_rff(
                n_features=nf, ell_scale_rff=ells, lambda_r=lr,
                h_KDE=h_KDE,
                stabilized=True, lambda_stab=ls, gamma_critic=1e-3,
                rff_seed_base=20243 + int(ls * 100),
            )
    return cfgs


def _stage_D_configs(h_KDE, ell_W0, ell_A0, ell_Z0, top_keys: List[str]):
    """Stage D: validation, top 2 only."""
    cfgs = {}
    cfgs["REG_Hsuper"] = _make_reg_hsuper(h_KDE)
    cfgs["DRK_Hsuper"] = _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0)
    # Re-create top 2 (use SAME hyperparams, NEW seed_base => disjoint random draws)
    for k in top_keys[:2]:
        try:
            parts = k.split("_")
            nf = int(parts[0].replace("RFF", ""))
            ells = int(parts[1].replace("e", "")) / 10.0
            lr_tag = parts[2]
            if lr_tag == "L1e2": lr = 1e-2
            elif lr_tag == "L1e1": lr = 1e-1
            else: lr = 1e-2
        except Exception:
            continue
        cfgs[k] = _make_bennett_rff(
            n_features=nf, ell_scale_rff=ells, lambda_r=lr,
            h_KDE=h_KDE,
            rff_seed_base=20245 + nf * 17,  # disjoint from Stage A
        )
    return cfgs


def _stage_E_configs(h_KDE, ell_W0, ell_A0, ell_Z0, best_key: str):
    """Stage E: head-to-head best Bennett vs DRK_Hsuper at M=200."""
    cfgs = {}
    cfgs["DRK_Hsuper"] = _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0)
    try:
        parts = best_key.split("_")
        nf = int(parts[0].replace("RFF", ""))
        ells = int(parts[1].replace("e", "")) / 10.0
        lr_tag = parts[2]
        if lr_tag == "L1e2": lr = 1e-2
        elif lr_tag == "L1e1": lr = 1e-1
        else: lr = 1e-2
    except Exception:
        return cfgs
    cfgs[best_key] = _make_bennett_rff(
        n_features=nf, ell_scale_rff=ells, lambda_r=lr,
        h_KDE=h_KDE,
        rff_seed_base=20249 + nf * 17,
    )
    return cfgs


# ══════════════════════════════════════════════════════════════════════════════
#  Run helpers
# ══════════════════════════════════════════════════════════════════════════════

# _fmt and _augment_decomp_with_ser canonical home: pci.runner.diagnostics
# (C5.3, May 2026). Local _augment wrapper preserves the historical
# behaviour of using the module-level REF_IDX.
from pci.runner.diagnostics import _fmt  # noqa: F401


def _augment_decomp_with_ser(decomp: dict, data: dict) -> dict:
    """Backward-compat wrapper: passes module-level REF_IDX to canonical fn."""
    from pci.runner.diagnostics import _augment_decomp_with_ser as _augment_canonical
    return _augment_canonical(decomp, data, ref_idx=REF_IDX)


def _run_stage(
    *,
    stage: str,
    n: int,
    M: int,
    configs: Dict,
    J_pt: np.ndarray,
    raw_dir: Path,
    seed_base: int,
    force: bool = False,
) -> Dict[str, Optional[dict]]:
    """Run all configs at given (n, M) and return decompositions per config id."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for f in raw_dir.glob(f"*_n{n}_M{M}.pkl"):
            f.unlink()
    out: Dict[str, Optional[dict]] = {}
    for cfg_idx, (cid, factory_fn) in enumerate(configs.items()):
        label = f"{cid}_n{n}_M{M}"
        seed_b = seed_base + cfg_idx * 1_000_000 + n
        print(f"  [{cfg_idx+1}/{len(configs)}] {cid}  n={n}  M={M}")
        t0 = time.time()
        data = _run_block(
            None,  # dgp injected below via kwargs of _run_block
            factory_fn, A_GRID, J_pt,
            n=n, M=M, seed_base=seed_b,
            label=label, raw_dir=raw_dir,
        )
        elapsed = time.time() - t0
        decomp = _decompose(data)
        decomp = _augment_decomp_with_ser(decomp, data) if decomp is not None else None
        out[cid] = decomp
        if decomp is not None:
            bse = float(decomp["bse_grid"][REF_IDX])
            cov = float(decomp["coverage_ref"])
            print(f"    |b|/SE={_fmt(bse)}  cov={_fmt(cov)}  [{elapsed:.0f}s]")
        else:
            print(f"    [decompose returned None - all reps failed]  [{elapsed:.0f}s]")
    return out


def _run_stage_with_dgp(
    *,
    dgp,
    stage: str,
    n: int,
    M: int,
    configs: Dict,
    J_pt: np.ndarray,
    raw_dir: Path,
    seed_base: int,
    force: bool = False,
) -> Dict[str, Optional[dict]]:
    """Run all configs at given (n, M) and return decompositions per config id."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for f in raw_dir.glob(f"*_n{n}_M{M}.pkl"):
            f.unlink()
    out: Dict[str, Optional[dict]] = {}
    for cfg_idx, (cid, factory_fn) in enumerate(configs.items()):
        label = f"{cid}_n{n}_M{M}"
        seed_b = seed_base + cfg_idx * 1_000_000 + n
        print(f"  [{cfg_idx+1}/{len(configs)}] {cid}  n={n}  M={M}")
        t0 = time.time()
        data = _run_block(
            dgp, factory_fn, A_GRID, J_pt,
            n=n, M=M, seed_base=seed_b,
            label=label, raw_dir=raw_dir,
        )
        elapsed = time.time() - t0
        decomp = _decompose(data)
        decomp = _augment_decomp_with_ser(decomp, data) if decomp is not None else None
        out[cid] = decomp
        if decomp is not None:
            bse = float(decomp["bse_grid"][REF_IDX])
            cov = float(decomp["coverage_ref"])
            ser = float(decomp.get("ser_ref", np.nan))
            print(f"    |b|/SE={_fmt(bse)}  cov={_fmt(cov)}  SER={_fmt(ser, '.3f')}  [{elapsed:.0f}s]")
        else:
            print(f"    [decompose returned None - all reps failed]  [{elapsed:.0f}s]")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════════════

def _row_for_cid(cid: str, decomp: Optional[dict]) -> str:
    if decomp is None:
        return f"| {cid:<28} | --- | --- | --- | --- | --- | --- | --- | --- |"
    M_ok = int(decomp["M_ok"])
    bias_dr = float(decomp["bias_dr"][REF_IDX])
    bse = float(decomp["bse_grid"][REF_IDX])
    cov = float(decomp["coverage_ref"])
    ess_min = float(decomp.get("ess_min_mean", np.nan))
    w_p99 = float(decomp.get("w_p99_mean", np.nan))
    rr = float(decomp.get("riesz_res_mean", np.nan))
    # SER
    ser = float(decomp.get("ser_ref", np.nan)) if "ser_ref" in decomp else np.nan
    return (
        f"| {cid:<28} | {M_ok:>4} | {_fmt(bias_dr)} | {_fmt(bse)} | "
        f"{_fmt(cov)} | {_fmt(ser, '.3f')} | {_fmt(ess_min, '.1f')} | "
        f"{_fmt(w_p99, '.2f')} | {_fmt(rr, '.2e')} |"
    )


def _build_stage_report(stage: str, n: int, M: int, results: Dict[str, Optional[dict]],
                          benchmark_id: str = "DRK_Hsuper") -> str:
    import datetime
    lines = [
        f"# Phase 12B Bennett-lite RFF -- Stage {stage}",
        f"Date: {datetime.date.today().isoformat()}",
        f"DGP: MichaelisMentenDGP(SNR=0.95)  |  n={n}  |  M={M}  |  seed_base={SEED_BASES.get(stage, 'N/A')}",
        f"a_grid: {A_GRID.tolist()}  |  REF_IDX={REF_IDX} (a={A_GRID[REF_IDX]})",
        "",
        "## Results",
        "",
        ("| Config                       | M_ok | bias_DR  | |b|/SE  | "
         "cov    | SER   | ESS_abs | w_p99 | riesz_res |"),
        ("|" + "|".join("-" * w for w in [30, 6, 10, 8, 7, 7, 9, 7, 11]) + "|"),
    ]
    for cid, dec in results.items():
        lines.append(_row_for_cid(cid, dec))
    lines.append("")

    if benchmark_id in results and results[benchmark_id] is not None:
        bench = results[benchmark_id]
        bench_bse = float(bench["bse_grid"][REF_IDX])
        bench_cov = float(bench["coverage_ref"])
        lines.append(
            f"**Benchmark {benchmark_id}**: |b|/SE={_fmt(bench_bse)}, cov={_fmt(bench_cov)}"
        )
        lines.append("")

        # Find best Bennett (RFF/STAB) by |b|/SE
        ben = {k: v for k, v in results.items()
               if (k.startswith("RFF") or k.startswith("STAB")) and v is not None}
        if ben:
            best = min(ben, key=lambda c: float(ben[c]["bse_grid"][REF_IDX]))
            best_bse = float(ben[best]["bse_grid"][REF_IDX])
            best_cov = float(ben[best]["coverage_ref"])
            best_ser = float(ben[best].get("ser_ref", np.nan)) if "ser_ref" in ben[best] else np.nan
            best_rr = float(ben[best].get("riesz_res_mean", np.nan))
            ratio = best_bse / bench_bse if bench_bse > 0 else np.nan
            lines.append(
                f"**Best Bennett by |b|/SE**: {best} -- |b|/SE={_fmt(best_bse)} "
                f"({_fmt(ratio, '.2f')}x bench), cov={_fmt(best_cov)}, "
                f"SER={_fmt(best_ser, '.3f')}, riesz_res={_fmt(best_rr, '.2e')}"
            )
            lines.append("")

            # Filtered top configs (apply hard criteria)
            filtered = []
            for k, v in ben.items():
                bse_k = float(v["bse_grid"][REF_IDX])
                cov_k = float(v["coverage_ref"])
                ser_k = float(v.get("ser_ref", np.nan)) if "ser_ref" in v else np.nan
                rr_k = float(v.get("riesz_res_mean", np.nan))
                ess_k = float(v.get("ess_min_mean", np.nan))
                # Filter:
                #   |b|/SE <= 0.95 * bench AND
                #   cov >= bench_cov - 0.04 AND
                #   SER in [0.7, 1.3] AND
                #   ESS_abs > 50 AND
                #   riesz_res < 1e+1
                ratio_k = bse_k / bench_bse if bench_bse > 0 else np.nan
                if (ratio_k <= 0.95 and
                    cov_k >= bench_cov - 0.04 and
                    0.7 <= ser_k <= 1.3 and
                    ess_k > 50 and
                    np.isfinite(rr_k) and rr_k < 1e+1):
                    filtered.append((k, ratio_k, ser_k, rr_k))
            if filtered:
                filtered.sort(key=lambda t: t[1])
                lines.append(
                    f"**{len(filtered)} configs pass hard filters** (|b|/SE-ratio, SER, ESS, riesz):"
                )
                for (k, r, s, rr) in filtered[:5]:
                    lines.append(f"  - {k}: ratio={r:.3f}, SER={s:.3f}, riesz_res={rr:.2e}")
                lines.append("")
            else:
                lines.append("**No Bennett config passes all hard filters at this stage.**")
                lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Stage selection of top configs (read from Stage A results)
# ══════════════════════════════════════════════════════════════════════════════

def _select_top_from_stage_A(stageA_results: Dict[str, Optional[dict]],
                               benchmark_id: str = "DRK_Hsuper") -> List[str]:
    """Apply hard filters to Stage A results, return ordered top config IDs (best first)."""
    if benchmark_id not in stageA_results or stageA_results[benchmark_id] is None:
        return []
    bench = stageA_results[benchmark_id]
    bench_bse = float(bench["bse_grid"][REF_IDX])
    bench_cov = float(bench["coverage_ref"])
    if bench_bse <= 0:
        return []
    ranked = []
    for k, v in stageA_results.items():
        if not (k.startswith("RFF") or k.startswith("STAB")) or v is None:
            continue
        bse = float(v["bse_grid"][REF_IDX])
        cov = float(v["coverage_ref"])
        ser = float(v.get("ser_ref", np.nan)) if "ser_ref" in v else np.nan
        rr = float(v.get("riesz_res_mean", np.nan))
        ess = float(v.get("ess_min_mean", np.nan))
        ratio = bse / bench_bse
        # Filter
        passes = (
            ratio <= 0.95 and
            cov >= bench_cov - 0.04 and
            0.7 <= ser <= 1.3 and
            ess > 50 and
            np.isfinite(rr) and rr < 1e+1
        )
        if passes:
            ranked.append((k, ratio))
    ranked.sort(key=lambda t: t[1])
    return [k for (k, _) in ranked]


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 12B Bennett-lite RFF overnight")
    parser.add_argument("--stage", choices=["quick", "A", "A2", "B", "D", "E", "report"],
                        default="quick")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    stage = args.stage

    print("=" * 72)
    print(f"Phase 12B Bennett-lite RFF  --stage {stage}")
    print("=" * 72)

    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)

    # Reference sample for bandwidth computation
    ref_sample = dgp.generate(n=2000, seed=1)
    h_KDE  = _silverman_h(ref_sample.A)
    ell_W0 = _median_bandwidth(ref_sample.W)
    ell_A0 = _median_bandwidth(ref_sample.A)
    ell_Z0 = _median_bandwidth(ref_sample.Z)
    print(f"h_KDE={h_KDE:.4f}  ell_W0={ell_W0:.4f}  ell_A0={ell_A0:.4f}  ell_Z0={ell_Z0:.4f}")

    J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)
    print(f"J_policy_true: {[f'{v:.4f}' for v in J_pt]}")
    print()

    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    seed_base = SEED_BASES.get(stage, 24_000_000)

    # Build configs per stage
    if stage == "quick":
        configs = _stage_quick_configs(h_KDE, ell_W0, ell_A0, ell_Z0)
        n, M = 300, 3
        raw_dir = _RAW_BASE / "quick"
    elif stage == "A":
        configs = _stage_A_configs(h_KDE, ell_W0, ell_A0, ell_Z0)
        n, M = 1000, 50
        raw_dir = _RAW_BASE / "stageA"
    elif stage == "A2":
        # Read Stage A results, pick top 3 by hard filters
        prevA = _load_stage_results(_RAW_BASE / "stageA", n=1000, M=50)
        top = _select_top_from_stage_A(prevA)
        if not top:
            print("Stage A2 SKIPPED -- no top configs from Stage A (no signal).")
            return
        configs = _stage_A2_configs(h_KDE, ell_W0, ell_A0, ell_Z0, top[:3])
        n, M = 1000, 50
        raw_dir = _RAW_BASE / "stageA2"
    elif stage == "B":
        prevA = _load_stage_results(_RAW_BASE / "stageA", n=1000, M=50)
        top = _select_top_from_stage_A(prevA)
        if not top:
            print("Stage B SKIPPED -- no top configs from Stage A.")
            return
        configs = _stage_B_configs(h_KDE, ell_W0, ell_A0, ell_Z0, top[:3])
        n, M = 1000, 50
        raw_dir = _RAW_BASE / "stageB"
    elif stage == "D":
        prevA = _load_stage_results(_RAW_BASE / "stageA", n=1000, M=50)
        top = _select_top_from_stage_A(prevA)
        if not top:
            print("Stage D SKIPPED -- no top configs from Stage A.")
            return
        configs = _stage_D_configs(h_KDE, ell_W0, ell_A0, ell_Z0, top[:2])
        # Run at TWO sample sizes
        all_results = {}
        for nn, MM in [(1000, 100), (2000, 100)]:
            print(f"\n--- Stage D  n={nn}  M={MM} ---")
            raw_dir_nn = _RAW_BASE / "stageD"
            r = _run_stage_with_dgp(
                dgp=dgp, stage="D", n=nn, M=MM, configs=configs,
                J_pt=J_pt, raw_dir=raw_dir_nn, seed_base=seed_base,
                force=args.force,
            )
            all_results[nn] = r
            report = _build_stage_report("D", nn, MM, r)
            with open(_SUMM_DIR / f"phase12B_bennett_rff_stageD_n{nn}.md", "w",
                      encoding="utf-8") as fh:
                fh.write(report)
        return
    elif stage == "E":
        # Prefer Stage D survivor (n=2000, M=100) if available
        prevD2k = _load_stage_results(_RAW_BASE / "stageD", n=2000, M=100)
        if prevD2k and "DRK_Hsuper" in prevD2k and prevD2k["DRK_Hsuper"] is not None:
            bench = prevD2k["DRK_Hsuper"]
            bench_bse = float(bench["bse_grid"][REF_IDX])
            survivors = []
            for k, v in prevD2k.items():
                if (k.startswith("RFF") or k.startswith("STAB")) and v is not None:
                    bse_k = float(v["bse_grid"][REF_IDX])
                    survivors.append((k, bse_k / bench_bse if bench_bse > 0 else np.inf))
            survivors.sort(key=lambda t: t[1])
            if not survivors:
                print("Stage E SKIPPED -- no Stage D survivors.")
                return
            best = survivors[0][0]
            print(f"Selecting best from Stage D: {best} (ratio={survivors[0][1]:.3f})")
        else:
            prevA = _load_stage_results(_RAW_BASE / "stageA", n=1000, M=50)
            top = _select_top_from_stage_A(prevA)
            if not top:
                print("Stage E SKIPPED -- no top configs from Stage A.")
                return
            best = top[0]
            print(f"Stage D not available; selecting best from Stage A: {best}")
        configs = _stage_E_configs(h_KDE, ell_W0, ell_A0, ell_Z0, best)
        n, M = 2000, 200
        raw_dir = _RAW_BASE / "stageE"
    elif stage == "report":
        # Build the FINAL integrated report
        _build_final_report(dgp, h_KDE, ell_W0, ell_A0, ell_Z0)
        return
    else:
        raise ValueError(f"unknown stage {stage}")

    print(f"--- Stage {stage}  n={n}  M={M}  ({len(configs)} configs) ---")
    t0 = time.time()
    results = _run_stage_with_dgp(
        dgp=dgp, stage=stage, n=n, M=M, configs=configs,
        J_pt=J_pt, raw_dir=raw_dir, seed_base=seed_base, force=args.force,
    )
    elapsed = time.time() - t0
    print(f"\nTotal elapsed Stage {stage}: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Write stage report
    report = _build_stage_report(stage, n, M, results)
    report_path = _SUMM_DIR / f"phase12B_bennett_rff_stage{stage}.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"Stage report: {report_path}")
    # Echo summary to stdout
    print()
    for line in report.split("\n")[:50]:
        print(line)


def _load_stage_results(raw_dir: Path, n: int, M: int) -> Dict[str, Optional[dict]]:
    """Load all PKLs from a stage and return {cid: decomp}."""
    out = {}
    if not raw_dir.exists():
        return out
    for pkl_path in raw_dir.glob(f"*_n{n}_M{M}.pkl"):
        cid = pkl_path.stem.replace(f"_n{n}_M{M}", "")
        try:
            with open(pkl_path, "rb") as fh:
                data = pickle.load(fh)
            decomp = _decompose(data)
            decomp = _augment_decomp_with_ser(decomp, data) if decomp is not None else None
            out[cid] = decomp
        except Exception as e:
            print(f"  [WARN] failed to load {pkl_path.name}: {e}")
    return out


def _build_final_report(dgp, h_KDE, ell_W0, ell_A0, ell_Z0):
    """Aggregate all stage results into the final report."""
    import datetime
    print("Building final integrated report...")

    stages_data = {}
    # Stage A
    stages_data["A"] = _load_stage_results(_RAW_BASE / "stageA", 1000, 50)
    # Stage A2
    stages_data["A2"] = _load_stage_results(_RAW_BASE / "stageA2", 1000, 50)
    # Stage B
    stages_data["B"] = _load_stage_results(_RAW_BASE / "stageB", 1000, 50)
    # Stage D
    stages_data["D_n1000"] = _load_stage_results(_RAW_BASE / "stageD", 1000, 100)
    stages_data["D_n2000"] = _load_stage_results(_RAW_BASE / "stageD", 2000, 100)
    # Stage E
    stages_data["E"] = _load_stage_results(_RAW_BASE / "stageE", 2000, 200)

    # Determine verdict
    # Case A: Stage D survives at n=2000, M=100 with full hard filters
    verdict = _determine_verdict(stages_data)

    # Compute boundary-dose tables for top configs
    boundary_tables = _build_boundary_tables(stages_data)

    lines = []
    lines.append("# Phase 12B Bennett-lite RFF -- FINAL REPORT")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append("")
    lines.append("## 0. Verdict (TL;DR)")
    lines.append("")
    lines.append(verdict["summary"])
    lines.append("")
    lines.append(f"**Case: {verdict['case']}**")
    lines.append("")
    lines.append("## 1. Six questions answered")
    for q, a in verdict["six_questions"].items():
        lines.append(f"- **{q}**: {a}")
    lines.append("")
    lines.append("## 2. Phase 12A polynomial recap")
    lines.append("Phase 12A tested polynomial degree-2/3 features for the Riesz")
    lines.append("representer. Result: Case C failure -- riesz_residual 9.8e+01..2.3e+10,")
    lines.append("SER 0.24..0.83, all bias worse than REG, low |b|/SE was variance inflation x72.")
    lines.append("Phase 12B replaces polynomials with random Fourier features (RFF) for richer")
    lines.append("approximation. Closed-form policy integration (no quadrature) via Gaussian")
    lines.append("characteristic function (see docs/notes/bennett_rff_integration_math.md).")
    lines.append("")

    # Stage tables
    for stage_label, results in stages_data.items():
        if not results:
            continue
        lines.append(f"## Stage {stage_label} results")
        lines.append("")
        lines.append("| Config | M_ok | bias_DR | |b|/SE | cov | SER | ESS | w_p99 | riesz_res |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for cid, dec in results.items():
            lines.append(_row_for_cid(cid, dec))
        lines.append("")

    if boundary_tables:
        lines.append("## Boundary-dose analysis")
        lines.append("")
        lines.append(boundary_tables)
        lines.append("")

    lines.append("## Recommendation")
    lines.append(verdict["recommendation"])
    lines.append("")
    lines.append("## Essay-ready phrase")
    lines.append("")
    lines.append(verdict["essay_phrase"])
    lines.append("")
    out = "\n".join(lines)
    final_path = _SUMM_DIR / "phase12B_bennett_rff_FINAL.md"
    with open(final_path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"Final report: {final_path}")


def _build_boundary_tables(stages_data) -> str:
    """Build markdown table of bias and |b|/SE per dose for top configs (multi-stage)."""
    out = []
    for stage_key, label in [("A", "Stage A (n=1000, M=50)"),
                             ("D_n1000", "Stage D (n=1000, M=100, disjoint)"),
                             ("D_n2000", "Stage D (n=2000, M=100, disjoint)"),
                             ("E", "Stage E (n=2000, M=200)")]:
        results = stages_data.get(stage_key, {})
        if not results:
            continue
        # Pick relevant configs: REG, DRK, top RFF/STAB
        out.append(f"### {label}")
        out.append("")
        out.append("| Config | a=1.8 |b|/SE | a=2.2 |b|/SE | a=2.6 |b|/SE | a=3.0 |b|/SE |")
        out.append("|---|---|---|---|---|")
        for cid, dec in results.items():
            if dec is None: continue
            if not (cid in ("REG_Hsuper", "DRK_Hsuper") or
                    cid.startswith("RFF") or cid.startswith("STAB")):
                continue
            bse = dec.get("bse_grid")
            if bse is None: continue
            row = f"| {cid} |"
            for k in range(len(bse)):
                row += f" {float(bse[k]):.3f} |"
            out.append(row)
        out.append("")
    return "\n".join(out)


def _determine_verdict(stages_data: Dict[str, Dict[str, Optional[dict]]]) -> Dict[str, str]:
    """Apply decision rules and return verdict dict."""
    # Default: Case D
    out = {
        "case": "D",
        "summary": "Bennett-lite RFF did not improve over the KPV-DRKernel benchmark.",
        "six_questions": {},
        "recommendation": "Phase 12 closed. Move to Phase 13 (Coverage Map).",
        "essay_phrase": (
            "Replacing polynomial features with random Fourier features in the direct "
            "Riesz representation did not yield reproducible bias reduction over the KPV-DRKernel "
            "benchmark on DGP2 (Michaelis-Menten, SNR=0.95). This confirms that the KPV pipeline "
            "is near-optimal for this DGP at finite sample, and motivates the Coverage Map "
            "experiment (Phase 13) as the next simulation priority."
        ),
    }

    # Stage A check
    A = stages_data.get("A", {})
    if not A:
        out["six_questions"]["A. RFF improves anything?"] = "No -- Stage A not run"
        return out

    bench = A.get("DRK_Hsuper")
    if bench is None:
        out["six_questions"]["A. RFF improves anything?"] = "No -- benchmark missing"
        return out

    bench_bse = float(bench["bse_grid"][REF_IDX])
    top_A = _select_top_from_stage_A(A)

    if not top_A:
        out["six_questions"]["A. RFF improves anything?"] = (
            "No -- no Stage A config passed hard filters"
        )
        return out

    # If Stage D ran, check survival
    D2k = stages_data.get("D_n2000", {})
    if D2k and D2k.get("DRK_Hsuper") is not None:
        bench_D = D2k["DRK_Hsuper"]
        bench_D_bse = float(bench_D["bse_grid"][REF_IDX])
        bench_D_cov = float(bench_D["coverage_ref"])
        survivors = []
        for k, v in D2k.items():
            if k.startswith("RFF") and v is not None:
                bse_k = float(v["bse_grid"][REF_IDX])
                cov_k = float(v["coverage_ref"])
                ser_k = float(v.get("ser_ref", np.nan))
                ess_k = float(v.get("ess_min_mean", np.nan))
                ratio_k = bse_k / bench_D_bse
                if (ratio_k <= 0.90 and
                    cov_k >= bench_D_cov - 0.03 and
                    0.7 <= ser_k <= 1.3 and
                    ess_k > 50):
                    survivors.append((k, ratio_k))
        if survivors:
            survivors.sort(key=lambda t: t[1])
            best = survivors[0]
            out["case"] = "A"
            out["summary"] = (
                f"Bennett-lite RFF SURVIVES disjoint validation at n=2000, M=100. "
                f"Best config: {best[0]} with |b|/SE={best[1]:.2f}x DRK_Hsuper."
            )
            out["six_questions"]["A. RFF improves anything?"] = "Yes (validated)"
            out["recommendation"] = (
                "Promote Bennett RFF as a viable estimator for S6/S7. Run Stage E "
                "(M=200 head-to-head) for final confirmation."
            )
            out["essay_phrase"] = (
                "Estimating the Riesz representer r_pi directly via random Fourier features, "
                "bypassing the q-bridge factorisation entirely, reduces finite-sample bias-to-SE "
                f"by {(1 - best[1]) * 100:.1f}% relative to the KPV-DRKernel benchmark on DGP2 "
                "(Michaelis-Menten, SNR=0.95, n=2000, M=100, disjoint seeds). This confirms a "
                "functional-first contribution: J(pi) is more strongly identified than the "
                "factorised bridges used to represent it."
            )
        else:
            out["case"] = "D"
            out["six_questions"]["A. RFF improves anything?"] = (
                "Stage A signal evaporated at validation (n=2000, M=100)"
            )
    else:
        # Only Stage A done
        out["case"] = "E"
        out["summary"] = (
            f"Stage A signal detected ({len(top_A)} configs pass filters). "
            "Validation pending (Stage D not complete)."
        )
        out["six_questions"]["A. RFF improves anything?"] = (
            f"Preliminary signal at n=1000, M=50 ({len(top_A)} configs); not yet validated"
        )
        out["recommendation"] = "Run Stage D (n=2000, M=100, disjoint seeds) before promoting."

    out["six_questions"]["B. Where?"] = (
        f"Top config = {top_A[0]}" if top_A else "n/a"
    )
    out["six_questions"]["C. Disjoint seeds reproducible?"] = (
        "TBD (Stage D not run)" if not stages_data.get("D_n2000") else
        ("Yes" if out["case"] == "A" else "No (evaporated)")
    )
    out["six_questions"]["D. Stable variance/SER/tails?"] = (
        "See Stage A table"
    )
    out["six_questions"]["E. Diagnostics observable?"] = (
        "riesz_residual + SER + ESS_abs flagged failures"
    )
    out["six_questions"]["F. Continue Bennett or close?"] = (
        "Close Phase 12 -> Phase 13 Coverage Map" if out["case"] == "D" else
        "Continue: head-to-head Stage E"
    )
    return out


if __name__ == "__main__":
    main()
