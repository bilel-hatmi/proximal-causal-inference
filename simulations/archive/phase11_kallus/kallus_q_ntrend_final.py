"""
Phase 11 -- Final N-trend with HP_winner from kallus_q_explore Step 2.

The exploration uncovered that Kallus_stabilized @ H_super (Phase 9B.4 winner h-bridge)
with tuned hyperparameters (gamma_Q=3e-3, lambda_stab=1.0, n_features=250) achieved
|b|/SE=0.04 vs KPV's 0.24 at n=1000, M=30. The auto-N-trend in kallus_q_explore was
run with HP central (not winner), giving misleading dynamics. This script validates
the HP_winner config across n in {500, 1000, 2000} at M=50.

Three cells are compared head-to-head per n:
    1. Q_kpv_baseline @ H_super        (anchor: Phase 9B.4 winner pure KPV)
    2. Q_kallus_stab @ H_super, HP central (sanity check)
    3. Q_kallus_stab @ H_super, HP winner   (gamma_Q=3e-3, lambda_stab=1.0, nf=250)

Output:
    simulations/results/raw/kallus_q_ntrend/<label>.pkl
    simulations/results/summaries/kallus_q_ntrend_final.md
"""
from __future__ import annotations

import datetime
import time
from pathlib import Path

import numpy as np

from simulations.experiments.dgp2_bias_diagnostics import (
    A_GRID, LAMBDA_Q_DEFAULT, N_FOLDS, RANDOM_STATE, REF_IDX, SNR,
    _compute_J_policy_true, _decompose, _median_bandwidth, _run_block, _silverman_h,
)
from simulations.experiments.kallus_q_explore import HP_CENTRAL, _h_bridge_kwargs

_BASE_DIR = Path(__file__).resolve().parents[2]
_RAW_DIR  = _BASE_DIR / "simulations" / "results" / "raw" / "kallus_q_ntrend"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "kallus_q_ntrend_final.md"

_SEED_BASE = 18_000_000


def make_factory(q_method: str, hp: dict, a_grid, h_KDE, ell_W0, ell_A0, ell_Z0):
    """Always uses H_super h-bridge. q_method in {kpv, kallus_central, kallus_winner}."""
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
    from simulations.methods.kallus_minimax import KallusStabilizedPolicyQ

    bridge_kwargs = _h_bridge_kwargs("H_super", ell_W0, ell_A0, ell_Z0)

    if q_method == "Q_kpv_baseline":
        def make_q():
            return KPVPolicyBridgeQ(a_grid=a_grid, h_KDE=h_KDE,
                                    lambda_Q=LAMBDA_Q_DEFAULT, clip=None)
    else:
        def make_q():
            return KallusStabilizedPolicyQ(a_grid=a_grid, h_KDE=h_KDE,
                                            mode="kallus_stabilized", **hp)

    def factory():
        return DRKernel(
            a_grid=a_grid, q_model=make_q(), bandwidth=h_KDE,
            n_folds=N_FOLDS, random_state=RANDOM_STATE, ref_dose_index=REF_IDX,
            cross_fit_q=True, bridge_kwargs=(bridge_kwargs if bridge_kwargs else None),
        )
    return factory


def main():
    print("=" * 72)
    print("Phase 11 -- Final N-trend with HP_winner")
    print("=" * 72)

    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)

    # Three configs
    HP_winner = {**HP_CENTRAL, "gamma_Q": 3e-3, "lambda_stab": 1.0,
                 "n_features_q": 250, "n_features_h": 250}
    configs = [
        ("Q_kpv_baseline",       "kpv",            None),
        ("Q_kallus_stab",        "kallus_central", HP_CENTRAL),
        ("Q_kallus_stab",        "kallus_winner",  HP_winner),
    ]
    n_list = [500, 1000, 2000]
    M = 50
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    cell_idx = 0
    t_start = time.perf_counter()
    for q, tag, hp in configs:
        for n in n_list:
            ref_sample = dgp.generate(n=n, seed=0)
            h_KDE  = _silverman_h(ref_sample.A)
            ell_W0 = _median_bandwidth(ref_sample.W)
            ell_A0 = _median_bandwidth(ref_sample.A)
            ell_Z0 = _median_bandwidth(ref_sample.Z)
            J_pt   = _compute_J_policy_true(dgp, A_GRID, h_KDE)
            factory = make_factory(q, hp or {}, A_GRID, h_KDE, ell_W0, ell_A0, ell_Z0)

            label = f"kallusnt_{tag}_n{n}_M{M}"
            seed_b = _SEED_BASE + cell_idx * 10_000 + n
            print(f"\n--- {tag} | n={n} | M={M} ---")
            t0 = time.perf_counter()
            data = _run_block(dgp, factory, A_GRID, J_pt, n=n, M=M,
                              seed_base=seed_b, label=label, raw_dir=_RAW_DIR)
            wall = time.perf_counter() - t0
            decomp = _decompose(data)
            if decomp is not None:
                print(f"    bias_DR={decomp['bias_dr'][REF_IDX]:+.4f}  "
                      f"|b|/SE={decomp['bse_grid'][REF_IDX]:.2f}  "
                      f"cov={decomp['coverage_ref']:.3f}  "
                      f"ESS={decomp.get('ess_min_mean', float('nan')):.0f}  "
                      f"wall={wall/60:.1f}min")
            results[(tag, n)] = decomp
            cell_idx += 1

    total_min = (time.perf_counter() - t_start) / 60.0
    print(f"\n{'=' * 72}\nTotal wall-clock: {total_min:.1f} min\n{'=' * 72}")

    # Build report
    lines = []
    lines.append("# Phase 11 -- Final N-trend with HP_winner")
    lines.append(f"# Generated : {datetime.date.today()}")
    lines.append("")
    lines.append("## Configurations")
    lines.append("")
    lines.append("All cells use H_super h-bridge (lambda_h=3e-5, ell_scale=3.5, Phase 9B.4 winner).")
    lines.append("")
    lines.append("- **kpv**: Q_kpv_baseline (lambda_Q=1e-3, no HP tuning)")
    lines.append("- **kallus_central**: Q_kallus_stab with HP central "
                 "(gamma_Q=1e-3, lambda_stab=1.0, n_features=150)")
    lines.append("- **kallus_winner**: Q_kallus_stab with HP_winner from explore Step 2 "
                 "(gamma_Q=3e-3, lambda_stab=1.0, n_features=250)")
    lines.append("")
    lines.append(f"DGP2 MichaelisMenten, snr_W=snr_Z={SNR}, ref dose a={A_GRID[REF_IDX]:.1f}, "
                 f"M={M}, n_folds={N_FOLDS}.")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| config | metric | n=500 | n=1000 | n=2000 |")
    lines.append("|--------|--------|-------|--------|--------|")
    for tag in ["kpv", "kallus_central", "kallus_winner"]:
        # |b|/SE row
        row = f"| {tag} | \\|b\\|/SE |"
        for n in n_list:
            d = results.get((tag, n))
            row += f" {float(d['bse_grid'][REF_IDX]):.2f} |" if d else " --- |"
        lines.append(row)
        # cov row
        row = f"| {tag} | cov |"
        for n in n_list:
            d = results.get((tag, n))
            row += f" {float(d['coverage_ref']):.3f} |" if d else " --- |"
        lines.append(row)
        # bias_DR row
        row = f"| {tag} | bias_DR |"
        for n in n_list:
            d = results.get((tag, n))
            row += f" {float(d['bias_dr'][REF_IDX]):+.4f} |" if d else " --- |"
        lines.append(row)
        # ESS row
        row = f"| {tag} | ESS_min |"
        for n in n_list:
            d = results.get((tag, n))
            row += f" {float(d.get('ess_min_mean', float('nan'))):.0f} |" if d else " --- |"
        lines.append(row)
    lines.append("")

    # Quick verdict
    lines.append("## Verdict")
    lines.append("")
    rows = []
    for n in n_list:
        kpv  = results.get(("kpv", n))
        kw   = results.get(("kallus_winner", n))
        if kpv is None or kw is None: continue
        kpv_bse = float(kpv['bse_grid'][REF_IDX])
        kw_bse  = float(kw['bse_grid'][REF_IDX])
        delta = (kpv_bse - kw_bse) / kpv_bse * 100
        rows.append((n, kpv_bse, kw_bse, delta))
    if rows:
        for n, k, w, d in rows:
            sym = "▼" if w < k else "▲"
            lines.append(f"- n={n}: KPV |b|/SE={k:.2f}, Kallus_winner |b|/SE={w:.2f} "
                         f"({sym} {abs(d):+.0f}% vs KPV)")
    lines.append("")

    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report -> {_REPORT_PATH}")


if __name__ == "__main__":
    main()
