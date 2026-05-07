"""
DGP2 Bias Diagnostics — three targeted sweeps to identify bias levers in DRKernel.

.. deprecated:: 2026-05 (C12)
   This module is retained as a thin shim for backward compatibility. The
   utility helpers it once owned (``_silverman_h``, ``_median_bandwidth``,
   ``_compute_J_policy_true``, ``_run_block``, ``_decompose``,
   ``_build_record``, etc.) were extracted to ``pci.runner.*`` in C5
   (May 2026). New code should import directly from those canonical
   locations:

   - ``from pci.runner.bandwidth_helpers import _silverman_h, ...``
   - ``from pci.runner.block_runner import _run_block``
   - ``from pci.runner.diagnostics import _decompose``

   The diagnostic-sweep functions (Experiments A/B/C) below are not
   referenced by any active S7/S8 essay pipeline. They remain only for
   historical reproducibility.

Context
-------
Exp1A pilot showed DGP2 (Michaelis-Menten) undercoverage for DRKernel xfit:
    |bias|/SE grows: 2.06(n=500) → 2.27(n=1000) → 2.66(n=2000)
    ρ ≈ 0.246 < 0.5 → structural bridge-bias violation
    Correction efficiency ≈ 1% → entire bias in KPVBridgeH, not q

Three diagnostics:
    A — h-bridge tuning  : lambda_1/lambda_2 + lengthscale of KPVBridgeH
    B — q correction tuning : lambda_Q + clip of KPVPolicyBridgeQ
    C — policy bandwidth : h_policy scale (bandwidth of intervention π_{a,h})

Modes
-----
    quick   : n=500, M=5, one config per diagnostic  (smoke-test ~2 min)
    main    : n=1000, M=30, full small grids A/B/C  (~60-90 min total)
    confirm : n=2000, M=30, best configs from main  (selective, ~30 min)

Usage
-----
    python -m simulations.experiments.dgp2_bias_diagnostics --mode quick
    python -m simulations.experiments.dgp2_bias_diagnostics --mode main
    python -m simulations.experiments.dgp2_bias_diagnostics --mode confirm \\
        --best-A "lam1e-03_ell0.5" --best-B "lam1e-04_clip35" --best-C "hpol1.5"

Output
------
    simulations/results/raw/dgp2_bias_diag/  — pkl per bloc
    simulations/results/summaries/dgp2_bias_diagnostics.md
"""
from __future__ import annotations

import argparse
import copy
import datetime
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR = (
    Path(__file__).resolve().parents[2]  # pci_essay/
)
_RAW_DIR  = _BASE_DIR / "simulations" / "results" / "raw" / "dgp2_bias_diag"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"

# ── Constants ─────────────────────────────────────────────────────────────────
A_GRID     = np.array([1.8, 2.2, 2.6, 3.0])
K          = len(A_GRID)
REF_IDX    = 2          # a = 2.6 (as in Exp1A pilot)
SNR        = 0.95
# CHECKPOINT_FREQ moved to pci.runner.block_runner (C5.2). Re-exported below.

# Baseline hyperparameters (from Exp1A _factory_drk_xfit)
LAMBDA_Q_DEFAULT   = 1e-3
CLIP_DEFAULT       = None   # → 5 * n^0.25 at runtime
N_FOLDS            = 5
RANDOM_STATE       = 42


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

# Diagnostic helpers extracted to pci.runner.diagnostics.
# Re-exported here for backward compatibility — please import from the new
# canonical location in new code.
from pci.runner.diagnostics import (  # noqa: F401
    _fmt,
    _pred_cov,
)


# Bandwidth helpers extracted to pci.runner.bandwidth_helpers.
# Re-exported here for backward compatibility — please import from the new
# canonical location in new code.
from pci.runner.bandwidth_helpers import (  # noqa: F401
    _silverman_h,
    _compute_J_policy_true,
    _median_bandwidth,
)


# Block runner extracted to pci.runner.block_runner.
# Re-exported here for backward compatibility — please import from the new
# canonical location in new code.
from pci.runner.block_runner import (  # noqa: F401
    _run_block,
    _build_record,
    CHECKPOINT_FREQ as _CHECKPOINT_FREQ_CANONICAL,
)
# Keep CHECKPOINT_FREQ at module scope for any code that reads it directly.
CHECKPOINT_FREQ = _CHECKPOINT_FREQ_CANONICAL


# ══════════════════════════════════════════════════════════════════════════════
#  Decompose a completed block → summary dict
# ══════════════════════════════════════════════════════════════════════════════

def _decompose(data: dict) -> dict | None:
    """Backward-compat wrapper: passes module-level REF_IDX to the canonical fn.

    Canonical function lives in pci.runner.diagnostics.
    """
    from pci.runner.diagnostics import _decompose as _decompose_canonical
    return _decompose_canonical(data, ref_idx=REF_IDX)


# ══════════════════════════════════════════════════════════════════════════════
#  Diagnostic A — h-bridge (KPVBridgeH) tuning
# ══════════════════════════════════════════════════════════════════════════════

def run_diag_A(dgp, n: int, M: int, ref_sample) -> dict[tuple, dict | None]:
    """
    Sweep lambda_h × ell_scale for KPVBridgeH via bridge_kwargs.
    Returns mapping (lambda_h, ell_scale) -> decomp dict.
    """
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ

    h_ref      = _silverman_h(ref_sample.A)
    J_pt       = _compute_J_policy_true(dgp, A_GRID, h_ref)

    # Baseline bandwidth medians
    ell_W0 = _median_bandwidth(ref_sample.W)
    ell_A0 = _median_bandwidth(ref_sample.A)
    ell_Z0 = _median_bandwidth(ref_sample.Z)

    LAMBDA_H_GRID  = [1e-4, 1e-3, 1e-2]
    ELL_SCALE_GRID = [0.5, 1.0, 2.0]
    SEED_BASE      = 5_000_000

    results: dict[tuple, dict | None] = {}
    print(f"\n[Diagnostic A] h-bridge tuning  n={n}, M={M}")

    for ls, lam_h in enumerate(LAMBDA_H_GRID):
        for es, ell_scale in enumerate(ELL_SCALE_GRID):
            lam_tag  = f"lam{lam_h:.0e}".replace("e-0", "e-").replace("e+0", "e")
            ell_tag  = f"ell{ell_scale:.1f}"
            label    = f"dgp2diag_A_h_{lam_tag}_{ell_tag}_n{n}_M{M}"
            seed_b   = SEED_BASE + ls * 100_000 + es * 10_000 + n

            def make_factory(lam=lam_h, ew=ell_W0*ell_scale, ea=ell_A0*ell_scale, ez=ell_Z0*ell_scale, h=h_ref):
                def factory():
                    q_model = KPVPolicyBridgeQ(
                        a_grid=A_GRID, h_KDE=h, lambda_Q=LAMBDA_Q_DEFAULT,
                    )
                    return DRKernel(
                        a_grid=A_GRID, q_model=q_model, bandwidth=h,
                        n_folds=N_FOLDS, random_state=RANDOM_STATE,
                        ref_dose_index=REF_IDX, cross_fit_q=True,
                        bridge_kwargs=dict(
                            lambda_1=lam, lambda_2=lam,
                            ell_W=ew, ell_A=ea, ell_Z=ez,
                        ),
                    )
                return factory

            data = _run_block(
                dgp, make_factory(), A_GRID, J_pt, n=n, M=M,
                seed_base=seed_b, label=label, raw_dir=_RAW_DIR,
            )
            results[(lam_h, ell_scale)] = _decompose(data)

    return results, J_pt, h_ref


# ══════════════════════════════════════════════════════════════════════════════
#  Diagnostic B — q correction (KPVPolicyBridgeQ) tuning
# ══════════════════════════════════════════════════════════════════════════════

def run_diag_B(dgp, n: int, M: int, ref_sample, h_ref: float, J_pt: np.ndarray,
               best_bridge_kwargs: dict | None = None) -> dict[tuple, dict | None]:
    """
    Sweep lambda_Q × clip for KPVPolicyBridgeQ.
    best_bridge_kwargs: from Diagnostic A winner (None → baseline).
    """
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ

    LAMBDA_Q_GRID  = [1e-4, 1e-3, 1e-2]
    CLIP_GRID      = [None, 20, 35, 70]    # None → 5*n^0.25 default
    SEED_BASE      = 6_000_000
    bkw = best_bridge_kwargs or {}

    results: dict[tuple, dict | None] = {}
    print(f"\n[Diagnostic B] q correction tuning  n={n}, M={M}")

    for lq_i, lam_q in enumerate(LAMBDA_Q_GRID):
        for cl_i, clip in enumerate(CLIP_GRID):
            lam_tag  = f"lam{lam_q:.0e}".replace("e-0", "e-").replace("e+0", "e")
            clip_tag = f"clip{clip}" if clip is not None else "clipNone"
            label    = f"dgp2diag_B_q_{lam_tag}_{clip_tag}_n{n}_M{M}"
            seed_b   = SEED_BASE + lq_i * 100_000 + cl_i * 10_000 + n

            def make_factory(lq=lam_q, cl=clip, h=h_ref, bk=bkw):
                def factory():
                    q_model = KPVPolicyBridgeQ(
                        a_grid=A_GRID, h_KDE=h, lambda_Q=lq, clip=cl,
                    )
                    return DRKernel(
                        a_grid=A_GRID, q_model=q_model, bandwidth=h,
                        n_folds=N_FOLDS, random_state=RANDOM_STATE,
                        ref_dose_index=REF_IDX, cross_fit_q=True,
                        bridge_kwargs=bk,
                    )
                return factory

            data = _run_block(
                dgp, make_factory(), A_GRID, J_pt, n=n, M=M,
                seed_base=seed_b, label=label, raw_dir=_RAW_DIR,
            )
            results[(lam_q, clip)] = _decompose(data)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Diagnostic C — policy bandwidth sweep
# ══════════════════════════════════════════════════════════════════════════════

def run_diag_C(dgp, n: int, M: int, ref_sample,
               best_bridge_kwargs: dict | None = None,
               best_lam_q: float = LAMBDA_Q_DEFAULT,
               best_clip: Optional[float] = CLIP_DEFAULT) -> dict[float, dict | None]:
    """
    Sweep h_policy_scale × {1.0, 1.5, 2.0} relative to Silverman h.
    """
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ

    H_SCALE_GRID = [1.0, 1.5, 2.0]
    SEED_BASE    = 7_000_000
    bkw          = best_bridge_kwargs or {}
    h_base       = _silverman_h(ref_sample.A)

    results: dict[float, dict | None] = {}
    print(f"\n[Diagnostic C] policy bandwidth  n={n}, M={M}")

    for sc_i, scale in enumerate(H_SCALE_GRID):
        h_pol   = h_base * scale
        J_pt_c  = _compute_J_policy_true(dgp, A_GRID, h_pol)
        label   = f"dgp2diag_C_hpol{scale:.1f}_n{n}_M{M}"
        seed_b  = SEED_BASE + sc_i * 10_000 + n

        def make_factory(h=h_pol, lq=best_lam_q, cl=best_clip, bk=bkw):
            def factory():
                q_model = KPVPolicyBridgeQ(
                    a_grid=A_GRID, h_KDE=h, lambda_Q=lq, clip=cl,
                )
                return DRKernel(
                    a_grid=A_GRID, q_model=q_model, bandwidth=h,
                    n_folds=N_FOLDS, random_state=RANDOM_STATE,
                    ref_dose_index=REF_IDX, cross_fit_q=True,
                    bridge_kwargs=bk,
                )
            return factory

        data = _run_block(
            dgp, make_factory(), A_GRID, J_pt_c, n=n, M=M,
            seed_base=seed_b, label=label, raw_dir=_RAW_DIR,
        )
        results[scale] = _decompose(data)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Baseline from Exp1A pkl
# ══════════════════════════════════════════════════════════════════════════════

def _load_baseline() -> dict[int, dict | None]:
    """Load Exp1A pilot DRK_xfit decompositions for n ∈ {500,1000,2000}."""
    from simulations.experiments.exp1A_dgp2_failure import _decompose as _exp1_decompose, _load
    baselines: dict[int, dict | None] = {}
    for n in [500, 1000, 2000]:
        label = f"exp1A_dgp2_drk_xfit_snr095_n{n}_M50"
        data  = _load(label)
        baselines[n] = _exp1_decompose(data) if data is not None else None
    return baselines


# ══════════════════════════════════════════════════════════════════════════════
#  Report builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_report(
    baselines: dict[int, dict | None],
    diag_A:    dict[tuple, dict | None],
    diag_B:    dict[tuple, dict | None],
    diag_C:    dict[float, dict | None],
    diag_A_n2: dict[tuple, dict | None] | None,
    diag_B_n2: dict[tuple, dict | None] | None,
    diag_C_n2: dict[float, dict | None] | None,
    n_main:    int,
    J_true_ref: float | None = None,
) -> list[str]:
    lines: list[str] = []
    add = lines.append

    add("# DGP2 Bias Diagnostics")
    add(f"# Genere : {datetime.date.today()}")
    add("# SNR=0.95, Michaelis-Menten, ref dose a=2.6 (REF_IDX=2)")
    add("# Trois diagnostics : A (h-bridge tuning), B (q correction), C (policy bandwidth)")
    add("")

    # ── Section 1 : Baseline ─────────────────────────────────────────────────
    add("## 1. Baseline Exp1A (DRK_xfit, M=50)")
    add("")
    add("| n | bias_reg (ref) | bias_dr (ref) | SE | |b|/SE | eff | coverage | ESS_min | jensen_gap_ref |")
    add("|---|--------------|-------------|-----|------|-----|---------|--------|----------------|")
    for n in [500, 1000, 2000]:
        d = baselines.get(n)
        if d is None:
            add(f"| {n} | (missing) | | | | | | | |")
            continue
        ref = REF_IDX
        J_pt_ref = float(d["J_pt"][ref]) if "J_pt" in d else np.nan
        jgap = _fmt(J_true_ref - J_pt_ref, "+.5f") \
               if (J_true_ref is not None and np.isfinite(J_pt_ref)) else "---"
        add(f"| {n} | {_fmt(d['bias_reg'][ref],'+.4f')} | {_fmt(d['bias_dr'][ref],'+.4f')} | "
            f"{_fmt(d['SE_grid'][ref],'.4f')} | {_fmt(d['bse_grid'][ref],'.2f')} | "
            f"{_fmt(d['correction_efficiency'][ref],'+.3f')} | {_fmt(d.get('coverage_ref',np.nan),'.3f')} | "
            f"{_fmt(d['ess_min_mean'],'.0f')} | {jgap} |")
    add("")
    add("Recall: correction efficiency ~0.01 for xfit => entire bias in h-bridge.")
    if J_true_ref is not None:
        add("Note: jensen_gap_ref = J_true(a_ref) - J_pt(a_ref)  [smoothing bias of pi_{a,h}].")
        add("      |jensen_gap_ref| << |bias_dr_ref| confirms the failure is not due to smoothing.")
    add("")

    # ── Section 2 : Diagnostic A ─────────────────────────────────────────────
    add("## 2. Diagnostic A — h-bridge tuning (lambda_h × ell_scale)")
    add("")
    add(f"n={n_main}, M per bloc shown. Default lambda_1=lambda_2 from Ying scaling; default ell=median heuristic.")
    add("")
    add("| lambda_h | ell_scale | n | bias_reg (ref) | bias_dr (ref) | eff | SE | |b|/SE | coverage | ESS_min |")
    add("|---------|---------|---|--------------|-------------|-----|-----|------|---------|--------|")
    _print_A_table(lines, diag_A, n_main)
    if diag_A_n2:
        add("")
        add(f"**Confirm n=2000 (best configs):**")
        add("")
        add("| lambda_h | ell_scale | n | bias_reg (ref) | bias_dr (ref) | eff | SE | |b|/SE | coverage | ESS_min |")
        add("|---------|---------|---|--------------|-------------|-----|-----|------|---------|--------|")
        _print_A_table(lines, diag_A_n2, 2000)
    add("")
    add("Critere : abs(bias_reg_ref) < 0.7 * baseline_bias_reg => tuning utile (>30% reduction)")
    add("")

    # ── Section 3 : Diagnostic B ─────────────────────────────────────────────
    add("## 3. Diagnostic B — q correction tuning (lambda_Q × clip)")
    add("")
    add("observed_correction = mean(J_dr - J_reg) ; required_correction = J_pt - mean(J_reg)")
    add("correction_ratio = observed_correction / required_correction (target: > 0.25)")
    add("")
    add("| lambda_Q | clip | n | bias_dr (ref) | corr_ratio | eff | V_corr (ref) | ESS | w_p99 | neg_share | q_clip |")
    add("|---------|------|---|-------------|-----------|-----|------------|-----|-------|---------|--------|")
    _print_B_table(lines, diag_B, n_main)
    if diag_B_n2:
        add("")
        add(f"**Confirm n=2000 (best configs):**")
        add("")
        add("| lambda_Q | clip | n | bias_dr (ref) | corr_ratio | eff | V_corr (ref) | ESS | w_p99 | neg_share | q_clip |")
        add("|---------|------|---|-------------|-----------|-----|------------|-----|-------|---------|--------|")
        _print_B_table(lines, diag_B_n2, 2000)
    add("")

    # ── Section 4 : Diagnostic C ─────────────────────────────────────────────
    add("## 4. Diagnostic C — policy bandwidth scale")
    add("")
    add("h_policy = Silverman(A) * scale  ; J_policy_true recalculated for each scale.")
    add("Jensen gap = J_true(a) - J_policy_true(a)  [increases with h]")
    add("")
    add("| h_scale | n | bias_reg (ref) | bias_dr (ref) | eff | Jensen gap (ref) | coverage | ESS | MISE |")
    add("|--------|---|--------------|-------------|-----|----------------|---------|-----|------|")
    _print_C_table(lines, diag_C, n_main)
    if diag_C_n2:
        add("")
        add(f"**Confirm n=2000:**")
        add("")
        add("| h_scale | n | bias_reg (ref) | bias_dr (ref) | eff | Jensen gap (ref) | coverage | ESS | MISE |")
        add("|--------|---|--------------|-------------|-----|----------------|---------|-----|------|")
        _print_C_table(lines, diag_C_n2, 2000)
    add("")

    # ── Section 5 : Winner table ─────────────────────────────────────────────
    add("## 5. Winner Table — top configurations")
    add("")
    add("Top configurations ranked by coverage_ref, then |bias_dr_ref|.")
    add("")
    add("| Diag | Config | n | coverage_ref | bias_dr_ref | |b|/SE | eff | ESS_min |")
    add("|------|--------|---|------------|-----------|------|-----|--------|")
    _print_winner_table(lines, diag_A, diag_B, diag_C, n_main)
    add("")

    # ── Section 6 : Interpretation ───────────────────────────────────────────
    add("## 6. Interpretation")
    add("")
    _write_interpretation(lines, diag_A, diag_B, diag_C, baselines, n_main)

    return lines


def _print_A_table(lines, diag_A, n):
    add = lines.append
    LAMBDA_H_GRID  = [1e-4, 1e-3, 1e-2]
    ELL_SCALE_GRID = [0.5, 1.0, 2.0]
    for lam_h in LAMBDA_H_GRID:
        for ell_scale in ELL_SCALE_GRID:
            d = diag_A.get((lam_h, ell_scale))
            if d is None:
                add(f"| {lam_h:.0e} | {ell_scale} | {n} | (missing) | | | | | | |")
                continue
            ref = REF_IDX
            add(f"| {lam_h:.0e} | {ell_scale} | {n} | {_fmt(d['bias_reg'][ref],'+.4f')} | "
                f"{_fmt(d['bias_dr'][ref],'+.4f')} | {_fmt(d['correction_efficiency'][ref],'+.3f')} | "
                f"{_fmt(d['SE_grid'][ref],'.4f')} | {_fmt(d['bse_grid'][ref],'.2f')} | "
                f"{_fmt(d['coverage_ref'],'.3f')} | {_fmt(d['ess_min_mean'],'.0f')} |")


def _print_B_table(lines, diag_B, n):
    add = lines.append
    LAMBDA_Q_GRID = [1e-4, 1e-3, 1e-2]
    CLIP_GRID     = [None, 20, 35, 70]
    for lam_q in LAMBDA_Q_GRID:
        for clip in CLIP_GRID:
            d = diag_B.get((lam_q, clip))
            if d is None:
                add(f"| {lam_q:.0e} | {clip or 'None'} | {n} | (missing) | | | | | | | |")
                continue
            ref = REF_IDX
            clip_tag = str(clip) if clip is not None else "None"
            add(f"| {lam_q:.0e} | {clip_tag} | {n} | {_fmt(d['bias_dr'][ref],'+.4f')} | "
                f"{_fmt(d['correction_ratio'][ref],'+.3f')} | {_fmt(d['correction_efficiency'][ref],'+.3f')} | "
                f"{_fmt(d['V_corr'][ref],'.3f')} | {_fmt(d['ess_min_mean'],'.0f')} | "
                f"{_fmt(d['w_p99_mean'],'.2f')} | {_fmt(d['neg_share_mean'],'.3f')} | "
                f"{_fmt(d['q_clip_mean'],'.4f')} |")


def _print_C_table(lines, diag_C, n):
    add = lines.append
    H_SCALE_GRID = [1.0, 1.5, 2.0]
    for scale in H_SCALE_GRID:
        d = diag_C.get(scale)
        if d is None:
            add(f"| {scale} | {n} | (missing) | | | | | | |")
            continue
        ref = REF_IDX
        jg = float(d['jensen_gap_mean'][ref]) if d['jensen_gap_mean'] is not None else np.nan
        add(f"| {scale} | {n} | {_fmt(d['bias_reg'][ref],'+.4f')} | "
            f"{_fmt(d['bias_dr'][ref],'+.4f')} | {_fmt(d['correction_efficiency'][ref],'+.3f')} | "
            f"{_fmt(jg,'+.4f')} | {_fmt(d['coverage_ref'],'.3f')} | "
            f"{_fmt(d['ess_min_mean'],'.0f')} | {_fmt(d['mise_mean'],'.5f')} |")


def _print_winner_table(lines, diag_A, diag_B, diag_C, n):
    add = lines.append
    # Collect all (coverage_ref, |bias_dr_ref|, label, d) tuples
    entries = []
    LAMBDA_H_GRID  = [1e-4, 1e-3, 1e-2]
    ELL_SCALE_GRID = [0.5, 1.0, 2.0]
    LAMBDA_Q_GRID  = [1e-4, 1e-3, 1e-2]
    CLIP_GRID      = [None, 20, 35, 70]
    H_SCALE_GRID   = [1.0, 1.5, 2.0]
    for lam_h in LAMBDA_H_GRID:
        for ell_scale in ELL_SCALE_GRID:
            d = diag_A.get((lam_h, ell_scale))
            if d:
                entries.append((d["coverage_ref"], abs(d["bias_dr"][REF_IDX]),
                                 f"A lam_h={lam_h:.0e} ell={ell_scale}", d))
    for lam_q in LAMBDA_Q_GRID:
        for clip in CLIP_GRID:
            d = diag_B.get((lam_q, clip))
            if d:
                entries.append((d["coverage_ref"], abs(d["bias_dr"][REF_IDX]),
                                 f"B lam_q={lam_q:.0e} clip={clip or 'None'}", d))
    for scale in H_SCALE_GRID:
        d = diag_C.get(scale)
        if d:
            entries.append((d["coverage_ref"], abs(d["bias_dr"][REF_IDX]),
                             f"C h_scale={scale}", d))
    entries.sort(key=lambda e: (-e[0], e[1]))
    for cov, ab, cfg, d in entries[:5]:
        ref = REF_IDX
        add(f"| {cfg.split()[0]} | {cfg[2:]} | {n} | {_fmt(cov,'.3f')} | "
            f"{_fmt(d['bias_dr'][ref],'+.4f')} | {_fmt(d['bse_grid'][ref],'.2f')} | "
            f"{_fmt(d['correction_efficiency'][ref],'+.3f')} | {_fmt(d['ess_min_mean'],'.0f')} |")


def _write_interpretation(lines, diag_A, diag_B, diag_C, baselines, n_main):
    add = lines.append
    ref = REF_IDX

    # Check if any A config reduces bias_reg by >30%
    bl_n = baselines.get(n_main)
    bl_bias_reg = abs(bl_n["bias_reg"][ref]) if bl_n else None

    best_A_bias_reg = None
    best_A_cfg = None
    for (lh, es), d in diag_A.items():
        if d and (best_A_bias_reg is None or abs(d["bias_reg"][ref]) < best_A_bias_reg):
            best_A_bias_reg = abs(d["bias_reg"][ref])
            best_A_cfg = (lh, es)

    best_B_ratio = None
    best_B_cfg = None
    for (lq, cl), d in diag_B.items():
        if d and np.isfinite(d["correction_ratio"][ref]):
            r = float(d["correction_ratio"][ref])
            if best_B_ratio is None or r > best_B_ratio:
                best_B_ratio = r
                best_B_cfg = (lq, cl)

    best_C_cov = None
    best_C_scale = None
    for scale, d in diag_C.items():
        if d and (best_C_cov is None or d["coverage_ref"] > best_C_cov):
            best_C_cov = d["coverage_ref"]
            best_C_scale = scale

    add("### A — h-bridge tuning")
    if bl_bias_reg and best_A_bias_reg:
        pct = 100 * (1 - best_A_bias_reg / bl_bias_reg)
        if pct > 30:
            add(f"Meilleure config ({best_A_cfg}): bias_reg reduit de {pct:.0f}% vs baseline.")
            add("=> Tuning reduce le biais de h. Recommande pour Exp1A DGP2 grande grille.")
        else:
            add(f"Reduction max du biais_reg: {pct:.0f}% < 30% seuil.")
            add("=> Biais de h resistant au tuning => structurel (RKHS rate).")
    add("")

    add("### B — q correction tuning")
    if best_B_ratio is not None:
        if best_B_ratio > 0.25:
            add(f"Meilleure config ({best_B_cfg}): correction_ratio = {best_B_ratio:.3f} > 0.25.")
            add("=> La correction q peut etre rendue plus active. Verifier ESS et V_corr.")
        else:
            add(f"Correction_ratio max: {best_B_ratio:.3f} < 0.25.")
            add("=> La correction q reste trop faible quelle que soit la regularisation.")
            add("=> Le q estime ne parvient pas a corriger le biais structurel du h-bridge.")
    add("")

    add("### C — policy bandwidth")
    if best_C_scale is not None and best_C_cov is not None:
        d_best_C = diag_C[best_C_scale]
        jg_best  = float(d_best_C["jensen_gap_mean"][ref]) if d_best_C else np.nan
        add(f"Best coverage with h_scale={best_C_scale}: coverage_ref={best_C_cov:.3f}.")
        add(f"Corresponding Jensen gap (ref dose): {jg_best:.4f}")
        if best_C_scale > 1.0 and best_C_cov > (diag_C.get(1.0, {}) or {}).get("coverage_ref", 0):
            add("=> Increasing h improves coverage but increases the Jensen gap.")
            add("=> Locality/inference trade-off: smoother target => smaller h-bias.")
            add("=> For S6: useful result to document the trade-off.")
    add("")

    add("### DGP2 final decision")
    add("")
    add("| Case | Condition | Decision |")
    add("|------|-----------|----------|")
    add("| 1 — h tuning effective | bias_reg reduced >30% | Integrate tuning, rerun small DGP2 pilot |")
    add("| 2 — q correction active | correction_ratio > 0.25 with no ESS collapse | Keep q tuning, document |")
    add("| 3 — only h_policy improves | coverage rises with scale but Jensen gap rises | S6 trade-off |")
    add("| 4 — no lever | all preceding cases false | DGP2 = theoretical stress-test ; switch Phase 11 |")
    add("")


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="DGP2 bias diagnostics")
    parser.add_argument("--mode", choices=["quick", "main", "confirm"], default="main")
    parser.add_argument("--best-A", type=str, default=None,
                        help="Tag for best A config to run at n=2000, e.g. 'lam1e-03_ell0.5'")
    parser.add_argument("--best-B", type=str, default=None,
                        help="Tag for best B config to run at n=2000, e.g. 'lam1e-04_clip35'")
    parser.add_argument("--best-C", type=str, default=None,
                        help="Tag for best C config to run at n=2000, e.g. 'hpol1.5'")
    args = parser.parse_args()

    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)

    if args.mode == "quick":
        n_main, M_main = 500, 5
    elif args.mode == "main":
        n_main, M_main = 1000, 30
    else:
        n_main, M_main = 2000, 30

    print(f"Mode: {args.mode}  n={n_main}  M={M_main}")

    # Reference sample for bandwidth estimation (always n=1000)
    ref_sample = dgp.generate(n=1000, seed=0)
    h_ref_base = _silverman_h(ref_sample.A)

    # ── Load baselines ────────────────────────────────────────────────────────
    print("\n[Baseline] Loading Exp1A DRK_xfit baselines ...")
    baselines = _load_baseline()

    # ── Diagnostic A ─────────────────────────────────────────────────────────
    diag_A, J_pt_main, h_ref_n = run_diag_A(dgp, n_main, M_main, ref_sample)

    # Find best A config (minimum |bias_reg| at ref dose)
    best_A_bkw = {}
    best_A_bias = float("inf")
    from simulations.methods.kpv_bridge import _median_bandwidth as _mbw
    ell_W0 = _mbw(ref_sample.W); ell_A0 = _mbw(ref_sample.A); ell_Z0 = _mbw(ref_sample.Z)
    for (lam_h, ell_scale), d in diag_A.items():
        if d and abs(d["bias_reg"][REF_IDX]) < best_A_bias:
            best_A_bias = abs(d["bias_reg"][REF_IDX])
            best_A_bkw = dict(lambda_1=lam_h, lambda_2=lam_h,
                              ell_W=ell_W0*ell_scale, ell_A=ell_A0*ell_scale, ell_Z=ell_Z0*ell_scale)

    # ── Diagnostic B ─────────────────────────────────────────────────────────
    diag_B = run_diag_B(dgp, n_main, M_main, ref_sample,
                        h_ref=h_ref_n, J_pt=J_pt_main,
                        best_bridge_kwargs=best_A_bkw)

    # Find best B config (maximum correction_ratio that keeps ESS > 20)
    best_lam_q = LAMBDA_Q_DEFAULT
    best_clip  = CLIP_DEFAULT
    best_ratio  = float("-inf")
    for (lam_q, clip), d in diag_B.items():
        if d and d["ess_min_mean"] > 20:
            r = float(d["correction_ratio"][REF_IDX])
            if np.isfinite(r) and r > best_ratio:
                best_ratio = r; best_lam_q = lam_q; best_clip = clip

    # ── Diagnostic C ─────────────────────────────────────────────────────────
    diag_C = run_diag_C(dgp, n_main, M_main, ref_sample,
                        best_bridge_kwargs=best_A_bkw,
                        best_lam_q=best_lam_q, best_clip=best_clip)

    # ── Confirm at n=2000 (main mode only, best configs) ─────────────────────
    diag_A_n2 = diag_B_n2 = diag_C_n2 = None
    if args.mode == "main" and n_main < 2000:
        print("\n[Confirm] Best configs at n=2000, M=30 ...")
        ref_sample_2k = dgp.generate(n=2000, seed=0)
        # Run only the best A config
        best_A_entry = min(
            ((k, d) for k, d in diag_A.items() if d),
            key=lambda kd: abs(kd[1]["bias_reg"][REF_IDX]),
            default=None,
        )
        if best_A_entry:
            (lam_h_best, ell_best), _ = best_A_entry
            lam_tag  = f"lam{lam_h_best:.0e}".replace("e-0", "e-").replace("e+0", "e")
            ell_tag  = f"ell{ell_best:.1f}"
            from simulations.methods.dr_kernel import DRKernel
            from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
            h_ref_2k = _silverman_h(ref_sample_2k.A)
            J_pt_2k  = _compute_J_policy_true(dgp, A_GRID, h_ref_2k)
            ew2 = ell_W0 * ell_best; ea2 = ell_A0 * ell_best; ez2 = ell_Z0 * ell_best
            bkw_best = dict(lambda_1=lam_h_best, lambda_2=lam_h_best, ell_W=ew2, ell_A=ea2, ell_Z=ez2)

            def make_A2(lam=lam_h_best, ew=ew2, ea=ea2, ez=ez2, h=h_ref_2k):
                def factory():
                    q_model = KPVPolicyBridgeQ(a_grid=A_GRID, h_KDE=h, lambda_Q=LAMBDA_Q_DEFAULT)
                    return DRKernel(a_grid=A_GRID, q_model=q_model, bandwidth=h,
                                   n_folds=N_FOLDS, random_state=RANDOM_STATE,
                                   ref_dose_index=REF_IDX, cross_fit_q=True,
                                   bridge_kwargs=dict(lambda_1=lam, lambda_2=lam,
                                                      ell_W=ew, ell_A=ea, ell_Z=ez))
                return factory

            label_A2 = f"dgp2diag_A_h_{lam_tag}_{ell_tag}_n2000_M30"
            data_A2  = _run_block(dgp, make_A2(), A_GRID, J_pt_2k, n=2000, M=30,
                                  seed_base=5_900_000, label=label_A2, raw_dir=_RAW_DIR)
            diag_A_n2 = {(lam_h_best, ell_best): _decompose(data_A2)}

            # Best B config at n=2000
            def make_B2(lq=best_lam_q, cl=best_clip, bk=bkw_best, h=h_ref_2k):
                def factory():
                    q_model = KPVPolicyBridgeQ(a_grid=A_GRID, h_KDE=h, lambda_Q=lq, clip=cl)
                    return DRKernel(a_grid=A_GRID, q_model=q_model, bandwidth=h,
                                   n_folds=N_FOLDS, random_state=RANDOM_STATE,
                                   ref_dose_index=REF_IDX, cross_fit_q=True,
                                   bridge_kwargs=bk)
                return factory

            lq_tag  = f"lam{best_lam_q:.0e}".replace("e-0", "e-").replace("e+0", "e")
            cl_tag  = f"clip{best_clip}" if best_clip is not None else "clipNone"
            label_B2 = f"dgp2diag_B_q_{lq_tag}_{cl_tag}_n2000_M30"
            data_B2  = _run_block(dgp, make_B2(), A_GRID, J_pt_2k, n=2000, M=30,
                                  seed_base=6_900_000, label=label_B2, raw_dir=_RAW_DIR)
            diag_B_n2 = {(best_lam_q, best_clip): _decompose(data_B2)}

            # Best C at n=2000
            diag_C_n2 = run_diag_C(dgp, 2000, 30, ref_sample_2k,
                                    best_bridge_kwargs=bkw_best,
                                    best_lam_q=best_lam_q, best_clip=best_clip)

    # ── Compute J_true_ref for Jensen gap column ──────────────────────────────
    J_true_ref = float(dgp.m_true(A_GRID[REF_IDX]))
    print(f"\n  J_true(a={A_GRID[REF_IDX]:.2f}) = {J_true_ref:.5f}")

    # ── Build report ──────────────────────────────────────────────────────────
    print("\n[Report] Building markdown ...")
    lines = _build_report(
        baselines, diag_A, diag_B, diag_C,
        diag_A_n2, diag_B_n2, diag_C_n2,
        n_main, J_true_ref=J_true_ref,
    )
    out = _SUMM_DIR / "dgp2_bias_diagnostics.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport -> {out}")


if __name__ == "__main__":
    main()
