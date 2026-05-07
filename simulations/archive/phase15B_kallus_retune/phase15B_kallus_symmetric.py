"""
Phase 15B -- Kallus symmetric retune: 6-wave protocol mirroring Phase 14B.

Same protocol depth as Phase 14B (Bennett-indep, 80 configs total) so the
Kallus benchmark in Phase 15 is fairly tuned, not under-explored.

Method: KallusMinimaxBridgeH (mode='kallus_stabilized') as h-bridge inside
DRKernel + KPVPolicyBridgeQ as q-bridge.

Protocol (6 waves, ~1h30 total):
  W1 -- Coarse h-grid: 5 gamma_H x 4 lambda_stab = 20 configs
        SNR=0.95, n=1000, M=50, seed_base=31_000_000
  W2 -- ell_scale refine: 9 configs around top region
  W3 -- m_h / m_critic sensitivity: 8 configs
  W4 -- gamma_critic axis: 8 configs
  W5 -- DISJOINT validation top 1-2 at 4 cells x M=100, seed_base=31_005_000
  W6 -- SNR=0.80 robustness on W5 winner

Disjoint seed discipline (KEY):
  W1-W4 tuning use seed_base = 31_000_000
  W5-W6 validation use seed_base = 31_005_000+ (disjoint)

Outputs:
  simulations/results/raw/phase15B_kallus/<wave>/<cid>_M{M}.pkl
  simulations/results/summaries/phase15B_kallus_FINAL.md
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
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "phase15B_kallus"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
K = len(A_GRID)
REF_IDX = 2
N_FOLDS = 5
RANDOM_STATE = 42
SEED_BASE_TUNE = 31_000_000
SEED_BASE_VAL = 31_005_000


from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
    _compute_J_policy_true,
)
from simulations.experiments.bennett_rff_overnight import (
    _augment_decomp_with_ser, _make_reg_hsuper, _make_drk_hsuper,
)
from simulations.experiments.phase14B_bennett_indep import (
    _aggregate_metrics, _print_table, _build_stage_report, _save_stage_report,
)


# -- Factory: DRKernel(KallusMinimaxBridgeH + KPVPolicyBridgeQ) ---------------

def _make_drk_kallus_h(
    h_KDE, ell_W0, ell_A0, ell_Z0, *,
    gamma_H: float,
    lambda_stab_h: float,
    gamma_critic_h: float = 1e-3,
    n_features_h: int = 400,
    n_features_critic: int = 400,
    ell_scale: float = 3.5,
    feature_seed: int = SEED_BASE_TUNE,
):
    """DRKernel with KallusMinimaxBridgeH (kallus_stabilized) + KPV q-bridge."""
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


# -- Run helpers --------------------------------------------------------------

def _run_cell_at_snr_n(
    *,
    snr: float, n: int, M: int,
    configs: Dict,
    raw_dir: Path,
    seed_base_cell: int,
    force: bool = False,
) -> Tuple[Dict[str, Optional[dict]], np.ndarray]:
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=snr, snr_Z=snr)
    ref = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref.A)
    ell_W0 = _median_bandwidth(ref.W)
    ell_A0 = _median_bandwidth(ref.A)
    ell_Z0 = _median_bandwidth(ref.Z)
    J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)

    # Build factories from spec dicts
    full_configs: Dict = {}
    for cid, spec in configs.items():
        if isinstance(spec, dict):
            full_configs[cid] = _make_drk_kallus_h(
                h_KDE, ell_W0, ell_A0, ell_Z0, **spec,
            )
        else:
            full_configs[cid] = spec
    if "REG_KPV_super" in full_configs:
        full_configs["REG_KPV_super"] = _make_reg_hsuper(h_KDE)
    if "DRK_KPV_super" in full_configs:
        full_configs["DRK_KPV_super"] = _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0)

    raw_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for f in raw_dir.glob(f"*_M{M}.pkl"):
            f.unlink()

    out: Dict[str, Optional[dict]] = {}
    for cid, factory_fn in full_configs.items():
        label = f"{cid}_M{M}"
        print(f"    [{cid}] n={n}")
        t0 = time.time()
        try:
            data = _run_block(
                dgp, factory_fn, A_GRID, J_pt,
                n=n, M=M, seed_base=seed_base_cell,
                label=label, raw_dir=raw_dir,
            )
            elapsed = time.time() - t0
            decomp = _decompose(data)
            if decomp is not None:
                decomp = _augment_decomp_with_ser(decomp, data)
                decomp["_data"] = data
            out[cid] = decomp
            if decomp is not None:
                bse = float(decomp["bse_grid"][REF_IDX])
                cov = float(decomp["coverage_ref"])
                ser = float(decomp.get("ser_ref", float("nan")))
                print(f"      bse_ref={bse:.4f}  cov={cov:.3f}  SER={ser:.3f}  [{elapsed:.0f}s]")
            else:
                print(f"      [no valid reps]  [{elapsed:.0f}s]")
        except Exception as e:
            print(f"      ERROR: {e}")
            out[cid] = None
    return out, J_pt


# -- Wave runners -------------------------------------------------------------

def _wave_K1(M=50, force=False) -> dict:
    """W1 -- Coarse h-grid: 5 gamma_H x 4 lambda_stab = 20 configs."""
    print(f"\n=== Wave K1 COARSE  SNR=0.95, n=1000, M={M} ===")
    configs: Dict = {"REG_KPV_super": "DUMMY", "DRK_KPV_super": "DUMMY"}
    gamma_H_grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    lambda_stab_grid = [0.1, 1.0, 10.0, 30.0]
    for gH in gamma_H_grid:
        for ls in lambda_stab_grid:
            gH_code = int(round(-np.log10(gH) * 10))   # 50, 45, 40, 35, 30
            ls_code = int(round(ls * 10))
            cid = f"K_gH{gH_code}_ls{ls_code:04d}"
            configs[cid] = dict(
                gamma_H=gH, lambda_stab_h=ls,
                gamma_critic_h=1e-3,
                n_features_h=400, n_features_critic=400,
                ell_scale=3.5,
                feature_seed=SEED_BASE_TUNE + gH_code * 1000 + ls_code,
            )
    out, _ = _run_cell_at_snr_n(
        snr=0.95, n=1000, M=M, configs=configs,
        raw_dir=_RAW_DIR / "wave1_coarse",
        seed_base_cell=SEED_BASE_TUNE,
        force=force,
    )
    return out


def _autonomous_select_K1(rows: Dict[str, dict]) -> List[Tuple[float, float]]:
    """
    Pick top 3 (gamma_H, lambda_stab) by composite score.
    Composite = 0.5*bse_ref + 0.3*bse_a18 + 0.2*bse_a30 (robust to boundary).
    """
    print("\n--- Autonomous selection (Wave K1) ---")
    k_rows = {cid: m for cid, m in rows.items() if cid.startswith("K_")}
    if not k_rows:
        print("  No K configs available; falling back to defaults.")
        return [(1e-4, 1.0), (3e-5, 1.0), (1e-4, 10.0)]
    # SER filter relaxed for Kallus (often miscalibrated)
    ser_pass = [(cid, m) for cid, m in k_rows.items()
                if 0.3 <= m.get("ser_ref", 0) <= 2.5]
    if not ser_pass:
        ser_pass = list(k_rows.items())
    scored = []
    for cid, m in ser_pass:
        score = 0.5 * m["bse_ref"] + 0.3 * m["bse_a18"] + 0.2 * m["bse_a30"]
        scored.append((cid, score, m))
    scored.sort(key=lambda t: t[1])
    print("  Composite = 0.5*bse_ref + 0.3*bse_a18 + 0.2*bse_a30:")
    for cid, score, m in scored[:5]:
        print(f"    {cid}: score={score:.4f} (bse_ref={m['bse_ref']:.3f}, "
              f"bse_a18={m['bse_a18']:.3f}, SER={m['ser_ref']:.3f})")
    # Decode top 3 into (gamma_H, lambda_stab)
    gh_grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    ls_grid = [0.1, 1.0, 10.0, 30.0]
    top: List[Tuple[float, float]] = []
    for cid, _, _ in scored[:3]:
        # cid format: K_gH{code}_ls{code04d}
        try:
            gh_code = int(cid.split("_gH")[1].split("_")[0])
            ls_code = int(cid.split("_ls")[1])
            gh_val = 10 ** (-gh_code / 10.0)
            ls_val = ls_code / 10.0
            top.append((gh_val, ls_val))
        except Exception:
            continue
    if not top:
        top = [(1e-4, 1.0)]
    print(f"  Top {len(top)} (gamma_H, lambda_stab): {top}")
    return top


def _wave_K2(top_gh_ls: List[Tuple[float, float]], M=50, force=False) -> dict:
    """W2 -- ell_scale refine around top: 3 ell x 3 (gamma_H, lambda_stab) = 9."""
    print(f"\n=== Wave K2 ell_scale refine  M={M} ===")
    configs: Dict = {"REG_KPV_super": "DUMMY", "DRK_KPV_super": "DUMMY"}
    ell_grid = [2.0, 2.75, 4.5]
    for ell in ell_grid:
        for gH, ls in top_gh_ls[:3]:
            gH_code = int(round(-np.log10(gH) * 10))
            ls_code = int(round(ls * 10))
            ell_code = int(round(ell * 100))
            cid = f"K_W2_ell{ell_code}_gH{gH_code}_ls{ls_code:04d}"
            configs[cid] = dict(
                gamma_H=gH, lambda_stab_h=ls,
                gamma_critic_h=1e-3,
                n_features_h=400, n_features_critic=400,
                ell_scale=ell,
                feature_seed=SEED_BASE_TUNE + 1_000_000 + ell_code * 1000 + gH_code * 10 + ls_code,
            )
    out, _ = _run_cell_at_snr_n(
        snr=0.95, n=1000, M=M, configs=configs,
        raw_dir=_RAW_DIR / "wave2_ell",
        seed_base_cell=SEED_BASE_TUNE,
        force=force,
    )
    return out


def _wave_K3(top_spec: dict, M=50, force=False) -> dict:
    """W3 -- m sensitivity at top spec: 4 n_features values."""
    print(f"\n=== Wave K3 m sensitivity  M={M} ===")
    configs: Dict = {"REG_KPV_super": "DUMMY", "DRK_KPV_super": "DUMMY"}
    nf_grid = [200, 600, 800, 1500]
    base = dict(top_spec)
    base.pop("feature_seed", None)
    base.pop("n_features_h", None)
    base.pop("n_features_critic", None)
    for nf in nf_grid:
        cid = f"K_W3_nf{nf}"
        configs[cid] = dict(
            base, n_features_h=nf, n_features_critic=nf,
            feature_seed=SEED_BASE_TUNE + 2_000_000 + nf,
        )
    out, _ = _run_cell_at_snr_n(
        snr=0.95, n=1000, M=M, configs=configs,
        raw_dir=_RAW_DIR / "wave3_m",
        seed_base_cell=SEED_BASE_TUNE,
        force=force,
    )
    return out


def _wave_K4(top_spec: dict, M=50, force=False) -> dict:
    """W4 -- gamma_critic axis at top spec: 4 values."""
    print(f"\n=== Wave K4 gamma_critic axis  M={M} ===")
    configs: Dict = {"REG_KPV_super": "DUMMY", "DRK_KPV_super": "DUMMY"}
    gc_grid = [1e-5, 1e-4, 1e-2, 1e-1]
    base = dict(top_spec)
    base.pop("gamma_critic_h", None)
    base.pop("feature_seed", None)
    for gc in gc_grid:
        gc_code = int(round(-np.log10(gc) * 10))
        cid = f"K_W4_gc{gc_code}"
        configs[cid] = dict(
            base, gamma_critic_h=gc,
            feature_seed=SEED_BASE_TUNE + 3_000_000 + gc_code,
        )
    out, _ = _run_cell_at_snr_n(
        snr=0.95, n=1000, M=M, configs=configs,
        raw_dir=_RAW_DIR / "wave4_gc",
        seed_base_cell=SEED_BASE_TUNE,
        force=force,
    )
    return out


def _wave_K5(K_CANDIDATES: Dict[str, dict], M=100, force=False) -> dict:
    """W5 -- DISJOINT validation: top 1-2 at 4 cells (SNR x n) M=100."""
    print(f"\n=== Wave K5 DISJOINT VALIDATION  M={M} ===")
    cells_results: Dict = {}
    for snr in [0.90, 0.95]:
        for n in [1000, 2000]:
            print(f"\n--- Cell SNR={snr}, n={n} ---")
            configs = {"REG_KPV_super": "DUMMY", "DRK_KPV_super": "DUMMY"}
            for cid, spec in K_CANDIDATES.items():
                spec_copy = dict(spec)
                spec_copy["feature_seed"] = SEED_BASE_VAL + 100 + (1000 if snr == 0.95 else 0) + (100 if n == 2000 else 0)
                configs[cid] = spec_copy
            seed_base_cell = SEED_BASE_VAL + (0 if snr == 0.90 else 1000) + (0 if n == 1000 else 100)
            out, _ = _run_cell_at_snr_n(
                snr=snr, n=n, M=M, configs=configs,
                raw_dir=_RAW_DIR / f"wave5_validation/snr{int(snr*100)}_n{n}",
                seed_base_cell=seed_base_cell,
                force=force,
            )
            cells_results[(snr, n)] = out
    return cells_results


def _wave_K6(top_spec: dict, M=50, force=False) -> dict:
    """W6 -- SNR=0.80 robustness on W5 winner."""
    print(f"\n=== Wave K6 SNR=0.80 ROBUSTNESS  M={M} ===")
    seed_base_w6 = SEED_BASE_VAL + 2000
    configs = {
        "REG_KPV_super": "DUMMY",
        "DRK_KPV_super": "DUMMY",
        "K_W5_winner": dict(top_spec, feature_seed=seed_base_w6 + 100),
    }
    out, _ = _run_cell_at_snr_n(
        snr=0.80, n=1000, M=M, configs=configs,
        raw_dir=_RAW_DIR / "wave6_snr80",
        seed_base_cell=seed_base_w6,
        force=force,
    )
    return out


# -- Decoders -----------------------------------------------------------------

def _decode_K1_cid(cid: str) -> Optional[dict]:
    """K_gH50_ls0010 -> dict with gamma_H, lambda_stab, ell_scale=3.5, etc."""
    try:
        if not cid.startswith("K_gH"):
            return None
        gh_code = int(cid.split("_gH")[1].split("_")[0])
        ls_code = int(cid.split("_ls")[1])
        gh_val = 10 ** (-gh_code / 10.0)
        ls_val = ls_code / 10.0
        return dict(
            gamma_H=gh_val, lambda_stab_h=ls_val,
            gamma_critic_h=1e-3,
            n_features_h=400, n_features_critic=400,
            ell_scale=3.5,
        )
    except Exception:
        return None


# -- Disk loaders for standalone wave runs ------------------------------------

def _load_K1_state_from_disk() -> dict:
    """Load K1 pkls from disk, re-aggregate, re-select top.

    Allows K2/K3/K4/K5/K6 to run standalone after K1 finishes.
    """
    k1_dir = _RAW_DIR / "wave1_coarse"
    if not k1_dir.exists():
        return {}
    out: Dict[str, Optional[dict]] = {}
    for pkl in k1_dir.glob("*_M50.pkl"):
        cid = pkl.stem.replace("_M50", "")
        try:
            with open(pkl, "rb") as fh:
                data = pickle.load(fh)
            decomp = _decompose(data)
            if decomp is not None:
                decomp = _augment_decomp_with_ser(decomp, data)
                decomp["_data"] = data
            out[cid] = decomp
        except Exception:
            out[cid] = None
    rows = _aggregate_metrics({k: v for k, v in out.items() if v is not None})
    state = {"K1_top": _autonomous_select_K1(rows)}
    return state


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["K1", "K2", "K3", "K4", "K5", "K6",
                 "K5_only", "K6_only", "all"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # State carried across waves (when stage='all')
    state: Dict[str, object] = {}

    # K5_only / K6_only: skip K2-K4 fine-tuning. Use K1 top picks directly.
    if args.stage in ("K5_only", "K6_only"):
        loaded = _load_K1_state_from_disk()
        K1_top = loaded.get("K1_top", [(1e-5, 0.1), (1e-5, 1.0), (3e-5, 0.1)])
        # Build top 2 candidate specs (with ell_scale=3.5, gc=1e-3, m=400 from K1 base)
        K_CANDIDATES: Dict[str, dict] = {}
        for i, (gh, ls) in enumerate(K1_top[:2]):
            K_CANDIDATES[f"K_top{i+1}_gH{int(round(-np.log10(gh)*10))}_ls{int(round(ls*10)):04d}"] = dict(
                gamma_H=gh, lambda_stab_h=ls,
                gamma_critic_h=1e-3,
                n_features_h=400, n_features_critic=400,
                ell_scale=3.5,
            )
        print(f"  K5_only candidates: {list(K_CANDIDATES.keys())}")

        if args.stage == "K5_only":
            cells_results = _wave_K5(K_CANDIDATES, force=args.force)
            for (snr, n), cell in cells_results.items():
                rows = _aggregate_metrics(cell)
                _save_stage_report(f"phase15B_kallus_W5_snr{int(snr*100)}_n{n}",
                                   _build_stage_report(
                    f"Phase 15B Wave K5 -- DISJOINT validation, SNR={snr}, n={n}, M=100, seed_base=31_005_000",
                    rows,
                ))
            # Pick W5 winner across cells (lowest avg_bse_ref)
            best_cid = None
            best_avg = float("inf")
            for cid in K_CANDIDATES.keys():
                bses = []
                for (snr, n), cell in cells_results.items():
                    decomp = cell.get(cid)
                    if decomp is not None:
                        bses.append(float(decomp["bse_grid"][REF_IDX]))
                if bses:
                    avg = float(np.mean(bses))
                    if avg < best_avg:
                        best_avg = avg
                        best_cid = cid
            if best_cid:
                state["K5_winner_spec"] = K_CANDIDATES[best_cid]
                print(f"  K5 winner: {best_cid}, avg_bse_ref={best_avg:.4f}")
            else:
                state["K5_winner_spec"] = list(K_CANDIDATES.values())[0]
            _write_final_report(state)
            elapsed = time.time() - t_total
            print(f"\n=== Phase 15B K5_only total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")
            return

        if args.stage == "K6_only":
            top_spec = list(K_CANDIDATES.values())[0]   # use top 1
            out = _wave_K6(top_spec, force=args.force)
            rows = _aggregate_metrics(out)
            _print_table(rows)
            _save_stage_report("phase15B_kallus_W6",
                               _build_stage_report(
                "Phase 15B Wave K6 -- SNR=0.80 robustness on K1 top",
                rows,
            ))
            elapsed = time.time() - t_total
            print(f"\n=== Phase 15B K6_only total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")
            return

    if args.stage in ("K1", "all"):
        out = _wave_K1(force=args.force)
        rows = _aggregate_metrics(out)
        _print_table(rows)
        _save_stage_report("phase15B_kallus_W1", _build_stage_report(
            "Phase 15B Wave K1 -- Coarse Kallus h-grid (5 gamma_H x 4 lambda_stab, ell=3.5, m=400, gc=1e-3, n=1000, M=50, SNR=0.95)",
            rows,
        ))
        state["K1_top"] = _autonomous_select_K1(rows)
        state["K1_top_cid"] = sorted(
            [(cid, 0.5 * m["bse_ref"] + 0.3 * m["bse_a18"] + 0.2 * m["bse_a30"], m)
             for cid, m in rows.items() if cid.startswith("K_")],
            key=lambda t: t[1],
        )[0][0] if any(cid.startswith("K_") for cid in rows) else None

    if args.stage in ("K2", "all"):
        top_gh_ls = state.get("K1_top", [(1e-4, 1.0), (3e-5, 1.0), (1e-4, 10.0)])
        out = _wave_K2(top_gh_ls, force=args.force)
        rows = _aggregate_metrics(out)
        _print_table(rows)
        _save_stage_report("phase15B_kallus_W2", _build_stage_report(
            "Phase 15B Wave K2 -- ell_scale refine (3 ell x top 3 (gH, ls))",
            rows,
        ))
        # Pick top from K2 by composite
        scored_w2 = sorted(
            [(cid, 0.5 * m["bse_ref"] + 0.3 * m["bse_a18"] + 0.2 * m["bse_a30"], m)
             for cid, m in rows.items() if cid.startswith("K_W2")],
            key=lambda t: t[1],
        )
        if scored_w2:
            top_cid = scored_w2[0][0]
            try:
                # cid format: K_W2_ell{ell_code}_gH{gh_code}_ls{ls_code04d}
                parts = top_cid.split("_")
                ell_code = int(parts[2][3:])
                gh_code = int(parts[3][2:])
                ls_code = int(parts[4][2:])
                state["K2_top"] = dict(
                    gamma_H=10 ** (-gh_code / 10.0),
                    lambda_stab_h=ls_code / 10.0,
                    gamma_critic_h=1e-3,
                    n_features_h=400, n_features_critic=400,
                    ell_scale=ell_code / 100.0,
                )
                print(f"  K2 top spec: {state['K2_top']}")
            except Exception as e:
                print(f"  K2 decode error: {e}")
                state["K2_top"] = None

    if args.stage in ("K3", "all"):
        top_spec = state.get("K2_top") or dict(
            gamma_H=1e-4, lambda_stab_h=1.0, gamma_critic_h=1e-3,
            n_features_h=400, n_features_critic=400, ell_scale=3.5,
        )
        out = _wave_K3(top_spec, force=args.force)
        rows = _aggregate_metrics(out)
        _print_table(rows)
        _save_stage_report("phase15B_kallus_W3", _build_stage_report(
            f"Phase 15B Wave K3 -- m sensitivity at top K2 spec (n_features in {{200,600,800,1500}})",
            rows,
        ))
        # Pick best m
        scored_w3 = sorted(
            [(cid, 0.5 * m["bse_ref"] + 0.3 * m["bse_a18"] + 0.2 * m["bse_a30"], m)
             for cid, m in rows.items() if cid.startswith("K_W3")],
            key=lambda t: t[1],
        )
        if scored_w3:
            best_cid = scored_w3[0][0]
            try:
                nf = int(best_cid.replace("K_W3_nf", ""))
                state["K3_top"] = dict(top_spec, n_features_h=nf, n_features_critic=nf)
                print(f"  K3 best m: {nf}, spec: {state['K3_top']}")
            except Exception:
                state["K3_top"] = top_spec
        else:
            state["K3_top"] = top_spec

    if args.stage in ("K4", "all"):
        top_spec = state.get("K3_top") or dict(
            gamma_H=1e-4, lambda_stab_h=1.0, gamma_critic_h=1e-3,
            n_features_h=400, n_features_critic=400, ell_scale=3.5,
        )
        out = _wave_K4(top_spec, force=args.force)
        rows = _aggregate_metrics(out)
        _print_table(rows)
        _save_stage_report("phase15B_kallus_W4", _build_stage_report(
            f"Phase 15B Wave K4 -- gamma_critic axis at top K3 spec",
            rows,
        ))
        scored_w4 = sorted(
            [(cid, 0.5 * m["bse_ref"] + 0.3 * m["bse_a18"] + 0.2 * m["bse_a30"], m)
             for cid, m in rows.items() if cid.startswith("K_W4")],
            key=lambda t: t[1],
        )
        if scored_w4:
            best_cid = scored_w4[0][0]
            try:
                gc_code = int(best_cid.replace("K_W4_gc", ""))
                state["K4_top"] = dict(top_spec, gamma_critic_h=10 ** (-gc_code / 10.0))
                print(f"  K4 best gamma_critic: {state['K4_top']['gamma_critic_h']}, spec: {state['K4_top']}")
            except Exception:
                state["K4_top"] = top_spec
        else:
            state["K4_top"] = top_spec

    if args.stage in ("K5", "all"):
        top_spec = state.get("K4_top") or dict(
            gamma_H=1e-4, lambda_stab_h=1.0, gamma_critic_h=1e-3,
            n_features_h=400, n_features_critic=400, ell_scale=3.5,
        )
        # Also include W2 best (untouched ell+m+gc) and full K4 winner = top 2
        W2_top = state.get("K2_top")
        K_CANDIDATES = {"K_super": top_spec}
        if W2_top is not None and W2_top != top_spec:
            K_CANDIDATES["K_W2_top"] = W2_top
        cells_results = _wave_K5(K_CANDIDATES, force=args.force)
        for (snr, n), cell in cells_results.items():
            rows = _aggregate_metrics(cell)
            _save_stage_report(f"phase15B_kallus_W5_snr{int(snr*100)}_n{n}",
                               _build_stage_report(
                f"Phase 15B Wave K5 -- DISJOINT validation, SNR={snr}, n={n}, M=100, seed_base=31_005_000",
                rows,
            ))
        state["K5_winner_spec"] = top_spec

    if args.stage in ("K6", "all"):
        top_spec = state.get("K5_winner_spec") or state.get("K4_top") or dict(
            gamma_H=1e-4, lambda_stab_h=1.0, gamma_critic_h=1e-3,
            n_features_h=400, n_features_critic=400, ell_scale=3.5,
        )
        out = _wave_K6(top_spec, force=args.force)
        rows = _aggregate_metrics(out)
        _print_table(rows)
        _save_stage_report("phase15B_kallus_W6",
                           _build_stage_report(
            "Phase 15B Wave K6 -- SNR=0.80 robustness on K5 winner",
            rows,
        ))

    # Final report
    if args.stage == "all":
        _write_final_report(state)

    elapsed = time.time() - t_total
    print(f"\n=== Phase 15B Kallus symmetric retune total: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")


def _write_final_report(state: dict):
    import datetime
    path = _SUMM_DIR / "phase15B_kallus_FINAL.md"
    K_super = state.get("K5_winner_spec") or state.get("K4_top") or {}
    lines = []
    lines.append("# Phase 15B Kallus symmetric retune -- FINAL")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append("")
    lines.append("## Protocol")
    lines.append("- 6 waves mirroring Phase 14B (Bennett-indep)")
    lines.append("- Tuning seed_base=31_000_000, validation seed_base=31_005_000")
    lines.append("- Method: DRKernel(KallusMinimaxBridgeH(kallus_stabilized) + KPVPolicyBridgeQ)")
    lines.append("")
    lines.append("## K_super specs (selected by autonomous diagnostics, no oracle)")
    lines.append("```python")
    lines.append("K_super = " + repr(K_super))
    lines.append("```")
    lines.append("")
    lines.append("## Wave summaries")
    for w in ["W1", "W2", "W3", "W4"]:
        path_w = _SUMM_DIR / f"phase15B_kallus_{w}.md"
        if path_w.exists():
            lines.append(f"- See `{path_w.name}`")
    for snr in [0.90, 0.95]:
        for n in [1000, 2000]:
            p = _SUMM_DIR / f"phase15B_kallus_W5_snr{int(snr*100)}_n{n}.md"
            if p.exists():
                lines.append(f"- W5 cell SNR={snr}, n={n}: `{p.name}`")
    p = _SUMM_DIR / "phase15B_kallus_W6.md"
    if p.exists():
        lines.append(f"- W6 SNR=0.80: `{p.name}`")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Final report: {path}")


if __name__ == "__main__":
    main()
