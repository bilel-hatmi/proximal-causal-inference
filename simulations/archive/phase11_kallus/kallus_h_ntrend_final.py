"""
Phase 11B -- Final N-trend validation for KallusMinimaxBridgeH.

Phase 11B screening (kallus_h_pilot, M=30, n=1000) found that F_rich
(KallusMinimaxBridgeH with gamma_H=1e-4, lambda_stab_h=0.1, n_features=400)
achieves bias_DR=+0.007 (vs C_super -0.026) and |b|/SE=0.18 (vs C_super 0.70)
with Q_KPV. With Q_oracle, F_rich and C_super both give |b|/SE=0.23 but
F_rich has cov=0.967 vs C_super 0.933.

This script validates the F_rich winner at n=2000, M=100 (vs screening n=1000 M=30)
and adds a finite-sample sanity at n=1000 (different seeds).

Configurations (all Q_KPV unless noted):
  C_super       : KPV_super (lambda_h=3e-5, ell_scale=3.5)            -- anchor
  F_rich        : KallusMinimaxBridgeH (gamma_H=1e-4, ls=0.1, nf=400) -- winner
  F_rich_qor    : F_rich + Q_oracle                                    -- interaction diagnostic
  C_super_qor   : C_super + Q_oracle                                   -- interaction baseline

Output:
  simulations/results/raw/kallus_h_ntrend/<label>.pkl
  simulations/results/summaries/kallus_h_ntrend_final.md
"""
from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Optional

import numpy as np

from simulations.experiments.dgp2_bias_diagnostics import (
    A_GRID, LAMBDA_Q_DEFAULT, N_FOLDS, RANDOM_STATE, REF_IDX, SNR,
    _compute_J_policy_true, _decompose, _median_bandwidth, _run_block, _silverman_h,
)

_BASE_DIR    = Path(__file__).resolve().parents[2]
_RAW_DIR     = _BASE_DIR / "simulations" / "results" / "raw" / "kallus_h_ntrend"
_SUMM_DIR    = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "kallus_h_ntrend_final.md"

_SEED_BASE = 20_000_000

# Kallus-h F_rich winner config
F_RICH_HP = dict(
    gamma_H=1e-4, lambda_stab_h=0.1, gamma_critic_h=1e-4,
    n_features_h=400, n_features_critic=400,
    ell_scale=3.5,
)

# KPV C_super anchor config
C_SUPER_HP = dict(lambda_h=3e-5, ell_scale=3.5)


def make_factory(tag: str, h_KDE, ell_W0, ell_A0, ell_Z0, dgp):
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
    from simulations.methods.kallus_minimax import KallusMinimaxBridgeH
    from simulations.methods._oracle_bridges import OracleBridgeQ

    bridge_kwargs: dict = {}
    h_factory = None

    if "C_super" in tag:
        bridge_kwargs = dict(
            lambda_1=C_SUPER_HP["lambda_h"], lambda_2=C_SUPER_HP["lambda_h"],
            ell_W=ell_W0 * C_SUPER_HP["ell_scale"],
            ell_A=ell_A0 * C_SUPER_HP["ell_scale"],
            ell_Z=ell_Z0 * C_SUPER_HP["ell_scale"],
        )
    elif "F_rich" in tag:
        def h_factory():
            return KallusMinimaxBridgeH(
                mode="kallus_stabilized", **F_RICH_HP,
                compute_diagnostics="light",
            )

    if tag.endswith("_qor"):
        q_model = OracleBridgeQ(dgp)
        cross_fit_q = False
    else:
        q_model = KPVPolicyBridgeQ(a_grid=A_GRID, h_KDE=h_KDE,
                                    lambda_Q=LAMBDA_Q_DEFAULT, clip=None)
        cross_fit_q = True

    def factory():
        return DRKernel(
            a_grid=A_GRID, q_model=q_model, bridge_kwargs=bridge_kwargs,
            bandwidth=h_KDE, n_folds=N_FOLDS, random_state=RANDOM_STATE,
            ref_dose_index=REF_IDX, cross_fit_q=cross_fit_q,
            h_bridge_factory=h_factory,
        )
    return factory


def _label(tag, n, M):
    return f"kallushnt_{tag}_n{n}_M{M}"


def _row(decomp):
    if decomp is None:
        return [np.nan] * 5
    return [
        float(decomp["bias_dr"][REF_IDX]),
        float(decomp["bse_grid"][REF_IDX]),
        float(decomp["coverage_ref"]),
        float(decomp.get("ess_min_mean", np.nan)),
        float(decomp.get("h_weighted_residual_mean", np.nan)),
    ]


def main():
    print("=" * 72)
    print("Phase 11B -- Final N-trend validation (F_rich winner)")
    print("=" * 72)

    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)

    configs = ["C_super", "F_rich", "C_super_qor", "F_rich_qor"]
    n_list = [1000, 2000]
    M = 100
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    cell_idx = 0
    t_start = time.perf_counter()
    for tag in configs:
        for n in n_list:
            ref_sample = dgp.generate(n=n, seed=0)
            h_KDE  = _silverman_h(ref_sample.A)
            ell_W0 = _median_bandwidth(ref_sample.W)
            ell_A0 = _median_bandwidth(ref_sample.A)
            ell_Z0 = _median_bandwidth(ref_sample.Z)
            J_pt   = _compute_J_policy_true(dgp, A_GRID, h_KDE)
            factory = make_factory(tag, h_KDE, ell_W0, ell_A0, ell_Z0, dgp)
            label = _label(tag, n, M)
            seed_b = _SEED_BASE + cell_idx * 10_000 + n

            print(f"\n--- {tag} | n={n} | M={M} ---")
            t0 = time.perf_counter()
            data = _run_block(dgp, factory, A_GRID, J_pt, n=n, M=M,
                              seed_base=seed_b, label=label, raw_dir=_RAW_DIR)
            wall = time.perf_counter() - t0
            decomp = _decompose(data)
            # Augment with h-extras mean across reps
            if decomp is not None:
                recs = [r for r in data["records"] if r.get("error") is None]
                if recs and recs[0].get("h_kallus_present"):
                    wres_vals = [r["h_weighted_residual_mean"] for r in recs
                                 if np.isfinite(r.get("h_weighted_residual_mean", np.nan))]
                    if wres_vals:
                        decomp["h_weighted_residual_mean"] = float(np.mean(wres_vals))
            if decomp is not None:
                bias, bse, cov, ess, wres = _row(decomp)
                wres_str = f" hwres={wres:.3f}" if np.isfinite(wres) else ""
                print(f"    bias_DR={bias:+.4f}  |b|/SE={bse:.2f}  "
                      f"cov={cov:.3f}  ESS={ess:.0f}{wres_str}  "
                      f"wall={wall/60:.1f}min")
            results[(tag, n)] = decomp
            cell_idx += 1

    total_min = (time.perf_counter() - t_start) / 60.0
    print(f"\n{'=' * 72}\nTotal wall-clock: {total_min:.1f} min\n{'=' * 72}")

    # Build report
    lines = []
    lines.append("# Phase 11B -- Final N-trend validation")
    lines.append(f"# Generated : {datetime.date.today()}")
    lines.append("")
    lines.append("## Configurations")
    lines.append("")
    lines.append("- **C_super** : KPVBridgeH(lambda_h=3e-5, ell_scale=3.5) + Q_KPV  -- Phase 9B.4 winner anchor")
    lines.append("- **F_rich**  : KallusMinimaxBridgeH(gamma_H=1e-4, lambda_stab_h=0.1, n_features=400) + Q_KPV  -- Phase 11B winner")
    lines.append("- **C_super_qor** / **F_rich_qor** : same with Q_oracle (interaction diagnostic)")
    lines.append("")
    lines.append(f"DGP2 MichaelisMenten, snr_W=snr_Z={SNR}, ref dose a={A_GRID[REF_IDX]:.1f}, M=100, n_folds={N_FOLDS}")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| config | metric | n=1000 | n=2000 |")
    lines.append("|--------|--------|--------|--------|")
    for tag in configs:
        for label, idx in [("|b|/SE", 1), ("cov", 2), ("bias_DR", 0), ("ESS_min", 3)]:
            row = f"| {tag} | {label} |"
            for n in n_list:
                d = results.get((tag, n))
                vals = _row(d)
                v = vals[idx]
                if label == "ESS_min":
                    row += f" {v:.0f} |" if np.isfinite(v) else " --- |"
                elif label == "cov":
                    row += f" {v:.3f} |" if np.isfinite(v) else " --- |"
                else:
                    fmt = "{:+.4f}" if "bias" in label else "{:.2f}"
                    row += f" {fmt.format(v)} |" if np.isfinite(v) else " --- |"
            lines.append(row)
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    for n in n_list:
        c_d = results.get(("C_super", n))
        f_d = results.get(("F_rich", n))
        c_q = results.get(("C_super_qor", n))
        f_q = results.get(("F_rich_qor", n))
        if c_d is None or f_d is None:
            continue
        cb = float(c_d["bse_grid"][REF_IDX])
        fb = float(f_d["bse_grid"][REF_IDX])
        cv = float(c_d["coverage_ref"])
        fv = float(f_d["coverage_ref"])
        delta = (cb - fb) / max(cb, 1e-9) * 100
        sym = "▼" if fb < cb else "▲"
        lines.append(f"- n={n} | Q_KPV : C_super |b|/SE={cb:.2f} cov={cv:.3f} -> "
                     f"F_rich |b|/SE={fb:.2f} cov={fv:.3f} ({sym}{abs(delta):.0f}% on |b|/SE)")
        if c_q is not None and f_q is not None:
            cqb = float(c_q["bse_grid"][REF_IDX])
            fqb = float(f_q["bse_grid"][REF_IDX])
            cqv = float(c_q["coverage_ref"])
            fqv = float(f_q["coverage_ref"])
            lines.append(f"- n={n} | Q_oracle : C_super |b|/SE={cqb:.2f} cov={cqv:.3f} -> "
                         f"F_rich |b|/SE={fqb:.2f} cov={fqv:.3f}")
    lines.append("")
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report -> {_REPORT_PATH}")


if __name__ == "__main__":
    main()
