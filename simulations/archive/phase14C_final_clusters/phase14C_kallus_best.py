"""
Phase 14C.1 — Kallus_best focused retune for fair Phase 14 comparison.

Phases 11/11B/11C documented Kallus extensively as "diagnostic only" — no
reproducible win vs KPV pipeline. For the FINAL Phase 14 6-method comparison,
we need ONE stable Kallus config to be a fair representative.

Protocol (2-wave, ~1.5h):
  K1 — Coarse Kallus_h tuning (9 configs, n=1000, M=50, SNR=0.95)
  K2 — Disjoint validation of top 1 (4 cells x M=100, seed_base disjoint)

Method: KallusMinimaxBridgeH (h-side, with KPV q-bridge as in DRKernel).
This is the most promising Kallus path per Phase 11B.

Hyperparam grid (K1):
  gamma_H in {1e-5, 1e-4, 1e-3}
  lambda_stab_h in {0.1, 1.0, 10.0}
  ell_scale = 3.5 (KPV-super match)
  n_features_h = n_features_critic = 400 (rich, Phase 11B "F_rich")
  gamma_critic_h = 1e-3 (default)

Outputs:
  simulations/results/raw/phase14C_kallus/k1_coarse/<cid>.pkl
  simulations/results/raw/phase14C_kallus/k2_validation/<cell>/<cid>.pkl
  simulations/results/summaries/phase14C_kallus_best.md
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
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "phase14C_kallus"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "phase14C_kallus_best.md"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
K = len(A_GRID)
REF_IDX = 2
N_FOLDS = 5
RANDOM_STATE = 42
SEED_BASE = 29_000_000


from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
    _compute_J_policy_true,
)
from simulations.experiments.bennett_rff_overnight import (
    _augment_decomp_with_ser, _make_reg_hsuper, _make_drk_hsuper,
)
from simulations.experiments.phase14B_bennett_indep import _aggregate_metrics, _print_table


def _make_drk_kallus_h(h_KDE, ell_W0, ell_A0, ell_Z0,
                       gamma_H, lambda_stab_h, gamma_critic_h,
                       n_features_h=400, n_features_critic=400,
                       ell_scale=3.5, feature_seed=29_000_000):
    """DRKernel with KallusMinimaxBridgeH for h + KPV q-bridge.

    Uses h_bridge_factory to substitute Kallus h-bridge for KPVBridgeH.
    """
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
    from simulations.methods.kallus_minimax import KallusMinimaxBridgeH

    def h_bridge_factory():
        return KallusMinimaxBridgeH(
            mode="kallus_stabilized",
            gamma_H=gamma_H,
            lambda_stab_h=lambda_stab_h,
            gamma_critic_h=gamma_critic_h,
            n_features_h=n_features_h,
            n_features_critic=n_features_critic,
            ell_scale=ell_scale,
            feature_seed=feature_seed,
        )

    def factory():
        q_model = KPVPolicyBridgeQ(
            a_grid=A_GRID, h_KDE=h_KDE, lambda_Q=1e-3, clip=None,
        )
        return DRKernel(
            a_grid=A_GRID, q_model=q_model,
            bandwidth=h_KDE, n_folds=N_FOLDS, random_state=RANDOM_STATE,
            ref_dose_index=REF_IDX, cross_fit_q=True,
            h_bridge_factory=h_bridge_factory,
        )
    return factory


def _stage_k1(M=50, force=False) -> dict:
    """K1 — Coarse Kallus h-bridge tuning."""
    print(f"\n=== K1 Kallus_h coarse tune  SNR=0.95, n=1000, M={M} ===")
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)
    ref = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref.A)
    ell_W0 = _median_bandwidth(ref.W)
    ell_A0 = _median_bandwidth(ref.A)
    ell_Z0 = _median_bandwidth(ref.Z)
    J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)

    configs = {
        "REG_KPV_super": _make_reg_hsuper(h_KDE),
        "DRK_KPV_super": _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0),
    }

    gamma_H_grid = [1e-5, 1e-4, 1e-3]
    lambda_stab_grid = [0.1, 1.0, 10.0]

    for gH in gamma_H_grid:
        for ls in lambda_stab_grid:
            gH_code = int(round(-np.log10(gH)))
            ls_code = f"{int(ls*10):03d}"
            cid = f"K_gH{gH_code}_ls{ls_code}"
            configs[cid] = _make_drk_kallus_h(
                h_KDE, ell_W0, ell_A0, ell_Z0,
                gamma_H=gH, lambda_stab_h=ls, gamma_critic_h=1e-3,
                n_features_h=400, n_features_critic=400,
                ell_scale=3.5,
                feature_seed=SEED_BASE + 1000 * gH_code + int(ls * 100),
            )

    out = {}
    raw_dir = _RAW_DIR / "k1_coarse"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for cfg_idx, (cid, factory_fn) in enumerate(configs.items()):
        label = f"{cid}_M{M}"
        seed_b = SEED_BASE + cfg_idx * 1000 + 1000  # disjoint per config
        print(f"  [{cid}] n=1000 M={M}")
        if force:
            for f in raw_dir.glob(f"{label}.pkl"):
                f.unlink()
        t0 = time.time()
        try:
            data = _run_block(
                dgp, factory_fn, A_GRID, J_pt,
                n=1000, M=M, seed_base=seed_b,
                label=label, raw_dir=raw_dir,
            )
            elapsed = time.time() - t0
            decomp = _decompose(data)
            if decomp is not None:
                decomp = _augment_decomp_with_ser(decomp, data)
            out[cid] = decomp
            if decomp is not None:
                bse = float(decomp["bse_grid"][REF_IDX])
                cov = float(decomp["coverage_ref"])
                ser = float(decomp.get("ser_ref", float("nan")))
                print(f"    bse_ref={bse:.4f}  cov={cov:.3f}  SER={ser:.3f}  [{elapsed:.0f}s]")
        except Exception as e:
            print(f"  [{cid}] FAILED: {type(e).__name__}: {e}")
            out[cid] = None
    return out


def _stage_k2(top_cid: str, top_kwargs: dict, M=100, force=False) -> Dict:
    """K2 — Disjoint validation of top Kallus config."""
    print(f"\n=== K2 Kallus disjoint validation  M={M} ===")
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    seed_base_k2 = SEED_BASE + 5000  # 29_005_000

    cells_results = {}
    for snr in [0.90, 0.95]:
        for n in [1000, 2000]:
            print(f"\n--- Cell SNR={snr}, n={n} ---")
            dgp = MichaelisMentenDGP(snr_W=snr, snr_Z=snr)
            ref = dgp.generate(n=2000, seed=1)
            h_KDE = _silverman_h(ref.A)
            ell_W0 = _median_bandwidth(ref.W)
            ell_A0 = _median_bandwidth(ref.A)
            ell_Z0 = _median_bandwidth(ref.Z)
            J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)

            configs = {
                "REG_KPV_super": _make_reg_hsuper(h_KDE),
                "DRK_KPV_super": _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0),
                top_cid: _make_drk_kallus_h(h_KDE, ell_W0, ell_A0, ell_Z0,
                                             feature_seed=seed_base_k2,
                                             **top_kwargs),
            }

            seed_base_cell = seed_base_k2 + (0 if snr == 0.90 else 1000) + (0 if n == 1000 else 100)
            cell_dir = _RAW_DIR / f"k2_validation/snr{int(snr*100)}_n{n}"
            cell_dir.mkdir(parents=True, exist_ok=True)

            cell_out = {}
            for cfg_idx, (cid, factory_fn) in enumerate(configs.items()):
                label = f"{cid}_M{M}"
                print(f"    [{cid}] n={n} M={M}")
                if force:
                    for f in cell_dir.glob(f"{label}.pkl"):
                        f.unlink()
                t0 = time.time()
                try:
                    data = _run_block(
                        dgp, factory_fn, A_GRID, J_pt,
                        n=n, M=M, seed_base=seed_base_cell,
                        label=label, raw_dir=cell_dir,
                    )
                    elapsed = time.time() - t0
                    decomp = _decompose(data)
                    if decomp is not None:
                        decomp = _augment_decomp_with_ser(decomp, data)
                    cell_out[cid] = decomp
                    if decomp is not None:
                        bse = float(decomp["bse_grid"][REF_IDX])
                        cov = float(decomp["coverage_ref"])
                        ser = float(decomp.get("ser_ref", float("nan")))
                        print(f"      bse_ref={bse:.4f}  cov={cov:.3f}  SER={ser:.3f}  [{elapsed:.0f}s]")
                except Exception as e:
                    print(f"    [{cid}] FAILED: {type(e).__name__}: {e}")
                    cell_out[cid] = None
            cells_results[(snr, n)] = cell_out
    return cells_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["k1", "k2", "all"], default="k1")
    parser.add_argument("--top-cid", default=None,
                        help="K2 only: top config from K1 to validate")
    parser.add_argument("--gamma-H", type=float, default=1e-4)
    parser.add_argument("--lambda-stab", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.stage in ("k1", "all"):
        k1 = _stage_k1(M=50, force=args.force)
        rows = _aggregate_metrics(k1)
        _print_table(rows)
        # Save report
        import datetime
        lines = [
            "# Phase 14C.1 Kallus_best — K1 coarse retune",
            f"Date: {datetime.date.today().isoformat()}",
            "",
            "| config | bias_ref | bse_ref | bse_a18 | bse_a30 | cov | SER |",
            "|---|---|---|---|---|---|---|",
        ]
        for cid, m in rows.items():
            if m is None: continue
            lines.append(f"| {cid} | {m.get('bias_ref',float('nan')):+.4f} | {m['bse_ref']:.4f} | {m['bse_a18']:.4f} | {m['bse_a30']:.4f} | {m['cov_ref']:.3f} | {m['ser_ref']:.3f} |")
        with open(_SUMM_DIR / "phase14C_k1_coarse.md", "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        print("Stage K1 done.")

    if args.stage in ("k2", "all"):
        # If --top-cid not given, pick best from K1
        if args.top_cid is None:
            print("ERROR: --top-cid required for stage k2")
            return
        # Reconstruct kwargs from cid
        # cid format: K_gHX_lsYYY where X = -log10(gamma_H), YYY = lambda*10
        try:
            parts = args.top_cid.split("_")
            gH = 10.0 ** (-int(parts[1].replace("gH", "")))
            ls = int(parts[2].replace("ls", "")) / 10.0
        except Exception:
            gH = args.gamma_H
            ls = args.lambda_stab
        top_kwargs = dict(
            gamma_H=gH, lambda_stab_h=ls, gamma_critic_h=1e-3,
            n_features_h=400, n_features_critic=400,
            ell_scale=3.5,
        )
        k2 = _stage_k2(args.top_cid, top_kwargs, M=100, force=args.force)
        for (snr, n), cell in k2.items():
            rows = _aggregate_metrics(cell)
            _print_table(rows)

    elapsed = time.time() - t0
    print(f"\nPhase 14C.1 elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
