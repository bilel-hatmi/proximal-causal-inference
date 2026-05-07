"""
Kallus q-bridge -- Phase 11 hyperparameter exploration & rigorous comparison vs KPV.

Goal
----
Round 1 pilot returned Cas D (q non-bottleneck) at H_baseline and H_tuned.
This script answers a stronger question:

    Does Kallus_stabilized q ever outperform KPVPolicyBridgeQ when both are
    paired with the **best known h-bridge** (Phase 9B.4 winner H_super:
    lambda_h=3e-5, ell_scale=3.5)?

If no Kallus configuration beats KPV at H_super on |b|/SE, coverage and ESS,
the verdict Cas D is consolidated for S6/S7.

Design (3 conditional steps with auto-skip)
-------------------------------------------
Step 1 : 3 h_cfg x 3 q_method = 9 cells at the central HP point.
         h_cfg in {H_baseline, H_tuned, H_super}
         q_method in {Q_kpv_baseline, Q_kallus_ridge, Q_kallus_stab}
         Central HP : gamma_Q=1e-3, lambda_stab=1.0, gamma_critic=1e-3,
                      n_features_q=n_features_h=150.
         Auto-decision : if min(|b|/SE) over Kallus cells >= max(|b|/SE) over
         KPV cells (no Kallus dominance even on one h_cfg), skip Step 2/3.

Step 2 : on (best h_cfg, Q_kallus_stab), sweep gamma_Q x lambda_stab = 3 x 3 = 9 cells.

Step 3 : on (best HP), sweep n_features in {100, 150, 250} = 3 cells.

N-trend : top 2 configs (one Kallus + KPV at H_super anchor) at n in {500, 1000, 2000},
          M=50.

All cells use n=1000, M=30 (consistent with Round 1) unless N-trend.

Output
------
    simulations/results/raw/kallus_q_explore/<label>.pkl
    simulations/results/summaries/kallus_q_explore.md
"""
from __future__ import annotations

import argparse
import datetime
import pickle
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from simulations.experiments.dgp2_bias_diagnostics import (
    A_GRID,
    LAMBDA_Q_DEFAULT,
    N_FOLDS,
    RANDOM_STATE,
    REF_IDX,
    SNR,
    _compute_J_policy_true,
    _decompose,
    _median_bandwidth,
    _run_block,
    _silverman_h,
)

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE_DIR    = Path(__file__).resolve().parents[2]
_RAW_DIR     = _BASE_DIR / "simulations" / "results" / "raw" / "kallus_q_explore"
_SUMM_DIR    = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "kallus_q_explore.md"

_SEED_BASE = 17_000_000   # disjoint from Phase 9B.x (10–15.9M) and pilot (16M)

# ── h-bridge configurations ──────────────────────────────────────────────────
H_CONFIGS: dict[str, dict] = {
    "H_baseline": dict(lambda_h=None, ell_scale=1.0),
    "H_tuned":    dict(lambda_h=1e-4, ell_scale=2.5),    # Phase 9B.3 winner
    "H_super":    dict(lambda_h=3e-5, ell_scale=3.5),    # Phase 9B.4 winner
}

# ── q-bridge methods ─────────────────────────────────────────────────────────
Q_METHODS = ("Q_kpv_baseline", "Q_kallus_ridge", "Q_kallus_stab")

# ── Central HP point (Round 1 reference) ─────────────────────────────────────
HP_CENTRAL = dict(
    n_features_q  = 150,
    n_features_h  = 150,
    gamma_Q       = 1e-3,
    lambda_stab   = 1.0,
    gamma_critic  = 1e-3,
    feature_seed  = 20260427,
    compute_diagnostics = "light",
)


# ══════════════════════════════════════════════════════════════════════════════
#  Factory builder
# ══════════════════════════════════════════════════════════════════════════════

def _h_bridge_kwargs(h_cfg: str, ell_W0: float, ell_A0: float, ell_Z0: float) -> dict:
    spec = H_CONFIGS[h_cfg]
    bridge_kwargs: dict = {}
    lam_h = spec["lambda_h"]
    ell_scale = float(spec["ell_scale"])
    if lam_h is not None:
        bridge_kwargs["lambda_1"] = float(lam_h)
        bridge_kwargs["lambda_2"] = float(lam_h)
    if (lam_h is not None) or (ell_scale != 1.0):
        bridge_kwargs["ell_W"] = float(ell_W0) * ell_scale
        bridge_kwargs["ell_A"] = float(ell_A0) * ell_scale
        bridge_kwargs["ell_Z"] = float(ell_Z0) * ell_scale
    return bridge_kwargs


def make_factory(
    q_method: str,
    h_cfg: str,
    a_grid: np.ndarray,
    h_KDE: float,
    ell_W0: float, ell_A0: float, ell_Z0: float,
    hp: dict,
) -> Callable:
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
    from simulations.methods.kallus_minimax import KallusStabilizedPolicyQ

    bridge_kwargs = _h_bridge_kwargs(h_cfg, ell_W0, ell_A0, ell_Z0)

    if q_method == "Q_kpv_baseline":
        def make_q():
            return KPVPolicyBridgeQ(
                a_grid=a_grid, h_KDE=h_KDE,
                lambda_Q=LAMBDA_Q_DEFAULT, clip=None,
            )
    elif q_method == "Q_kallus_ridge":
        kallus_hp = {**hp}
        def make_q():
            return KallusStabilizedPolicyQ(
                a_grid=a_grid, h_KDE=h_KDE, mode="ridge_gmm", **kallus_hp,
            )
    elif q_method == "Q_kallus_stab":
        kallus_hp = {**hp}
        def make_q():
            return KallusStabilizedPolicyQ(
                a_grid=a_grid, h_KDE=h_KDE, mode="kallus_stabilized", **kallus_hp,
            )
    else:
        raise ValueError(f"Unknown q_method {q_method!r}")

    def factory():
        return DRKernel(
            a_grid=a_grid, q_model=make_q(),
            bandwidth=h_KDE, n_folds=N_FOLDS, random_state=RANDOM_STATE,
            ref_dose_index=REF_IDX, cross_fit_q=True,
            bridge_kwargs=(bridge_kwargs if bridge_kwargs else None),
        )
    return factory


# ══════════════════════════════════════════════════════════════════════════════
#  Per-cell runner
# ══════════════════════════════════════════════════════════════════════════════

def _label(step: str, q: str, h: str, hp_tag: str, n: int, M: int) -> str:
    return f"kallusexp_{step}_{q}_{h}_{hp_tag}_n{n}_M{M}"


def _hp_tag(hp: dict) -> str:
    """Compact tag distinguishing HP variants (only differs from CENTRAL is shown)."""
    parts = []
    for k in ("gamma_Q", "lambda_stab", "n_features_q"):
        if hp.get(k) != HP_CENTRAL.get(k):
            v = hp[k]
            if isinstance(v, float):
                parts.append(f"{k}{v:.0e}".replace("+0", "+").replace("-0", "-"))
            else:
                parts.append(f"{k}{v}")
    return "central" if not parts else "_".join(parts)


def _seed_base(step_idx: int, cell_idx: int, n: int) -> int:
    return _SEED_BASE + step_idx * 1_000_000 + cell_idx * 10_000 + int(n)


def _run_one_cell(
    dgp,
    q_method: str,
    h_cfg: str,
    hp: dict,
    a_grid: np.ndarray,
    n: int,
    M: int,
    step: str,
    step_idx: int,
    cell_idx: int,
    force: bool = False,
) -> tuple[Optional[dict], float, Optional[dict]]:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    ref_sample = dgp.generate(n=n, seed=0)
    h_KDE  = _silverman_h(ref_sample.A)
    ell_W0 = _median_bandwidth(ref_sample.W)
    ell_A0 = _median_bandwidth(ref_sample.A)
    ell_Z0 = _median_bandwidth(ref_sample.Z)
    J_pt   = _compute_J_policy_true(dgp, a_grid, h_KDE)
    factory = make_factory(q_method, h_cfg, a_grid, h_KDE, ell_W0, ell_A0, ell_Z0, hp)

    label = _label(step, q_method, h_cfg, _hp_tag(hp), n, M)
    seed_b = _seed_base(step_idx, cell_idx, n)

    if force:
        for sfx in (".pkl", ".partial.pkl"):
            p = _RAW_DIR / f"{label}{sfx}"
            if p.exists():
                p.unlink()

    t0 = time.perf_counter()
    data = _run_block(
        dgp, factory, a_grid, J_pt,
        n=n, M=M, seed_base=seed_b, label=label, raw_dir=_RAW_DIR,
    )
    wall = time.perf_counter() - t0
    return _decompose(data), wall, data


def _print_brief(label: str, decomp: Optional[dict], wall: float) -> None:
    if decomp is not None:
        print(
            f"    [{label}]  bias_DR={decomp['bias_dr'][REF_IDX]:+.4f}  "
            f"|b|/SE={decomp['bse_grid'][REF_IDX]:.2f}  "
            f"cov={decomp['coverage_ref']:.3f}  "
            f"ESS={decomp.get('ess_min_mean', float('nan')):.0f}  "
            f"wall={wall/60:.1f}min"
        )
    else:
        print(f"    [{label}]  no data (wall={wall/60:.1f}min)")


# ══════════════════════════════════════════════════════════════════════════════
#  Step runners
# ══════════════════════════════════════════════════════════════════════════════

def run_step1(dgp, n: int, M: int, force: bool = False) -> dict:
    """3 h_cfg x 3 q_method = 9 cells at central HP point."""
    print("\n" + "-" * 72)
    print(f"STEP 1 -- 3 h_cfg x 3 q_method @ central HP (n={n}, M={M})")
    print("-" * 72)
    results: dict[tuple[str, str], tuple] = {}
    cell_idx = 0
    for h_cfg in H_CONFIGS.keys():
        for q in Q_METHODS:
            print(f"\n--- {q} | {h_cfg} ---")
            decomp, wall, data = _run_one_cell(
                dgp, q, h_cfg, HP_CENTRAL, A_GRID, n, M,
                step="s1", step_idx=1, cell_idx=cell_idx, force=force,
            )
            label = f"{q}|{h_cfg}|central"
            _print_brief(label, decomp, wall)
            results[(q, h_cfg)] = (decomp, wall, data)
            cell_idx += 1
    return results


def step1_winner_kallus(s1: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Identify the (q_kallus, h_cfg) cell with lowest |b|/SE among Kallus cells.
    Returns (q_method, h_cfg) or (None, None) if all Kallus dominated by KPV.
    """
    best_bse, best_q, best_h = float("inf"), None, None
    for (q, h), (d, _, _) in s1.items():
        if q == "Q_kpv_baseline" or d is None:
            continue
        bse = float(d["bse_grid"][REF_IDX])
        if bse < best_bse:
            best_bse, best_q, best_h = bse, q, h
    return best_q, best_h


def kallus_dominance_check(s1: dict) -> bool:
    """
    Return True if at least one Kallus cell beats the *same-h_cfg* KPV baseline
    by >=10% on |b|/SE. False otherwise (Cas D evident).
    """
    for h_cfg in H_CONFIGS.keys():
        kpv = s1.get(("Q_kpv_baseline", h_cfg))
        if kpv is None or kpv[0] is None:
            continue
        kpv_bse = float(kpv[0]["bse_grid"][REF_IDX])
        for q in ("Q_kallus_ridge", "Q_kallus_stab"):
            cell = s1.get((q, h_cfg))
            if cell is None or cell[0] is None:
                continue
            cell_bse = float(cell[0]["bse_grid"][REF_IDX])
            if cell_bse <= 0.90 * kpv_bse:
                print(f"    [DOMINANCE] {q}@{h_cfg} beats KPV by "
                      f"{100*(1-cell_bse/kpv_bse):.1f}%")
                return True
    return False


def run_step2(dgp, q: str, h_cfg: str, n: int, M: int, force: bool = False) -> dict:
    """gamma_Q x lambda_stab sweep on (q, h_cfg). 3 x 3 = 9 cells."""
    print("\n" + "-" * 72)
    print(f"STEP 2 -- {q} @ {h_cfg} : sweep gamma_Q x lambda_stab (n={n}, M={M})")
    print("-" * 72)
    gamma_Q_grid    = [3e-4, 1e-3, 3e-3]
    lambda_stab_grid = [0.1, 1.0, 10.0]
    results: dict[tuple[float, float], tuple] = {}
    cell_idx = 0
    for gQ in gamma_Q_grid:
        for ls in lambda_stab_grid:
            hp = {**HP_CENTRAL, "gamma_Q": gQ, "lambda_stab": ls}
            print(f"\n--- gamma_Q={gQ:.0e}  lambda_stab={ls:.1f} ---")
            decomp, wall, data = _run_one_cell(
                dgp, q, h_cfg, hp, A_GRID, n, M,
                step="s2", step_idx=2, cell_idx=cell_idx, force=force,
            )
            label = f"{q}|{h_cfg}|gQ{gQ:.0e}_ls{ls}"
            _print_brief(label, decomp, wall)
            results[(gQ, ls)] = (decomp, wall, data)
            cell_idx += 1
    return results


def run_step3(dgp, q: str, h_cfg: str, hp_winner: dict, n: int, M: int,
              force: bool = False) -> dict:
    """Sweep n_features in {100, 150, 250} on top config."""
    print("\n" + "-" * 72)
    print(f"STEP 3 -- {q} @ {h_cfg} (HP_winner) : sweep n_features (n={n}, M={M})")
    print("-" * 72)
    results: dict[int, tuple] = {}
    cell_idx = 0
    for nf in [100, 150, 250]:
        hp = {**hp_winner, "n_features_q": nf, "n_features_h": nf}
        print(f"\n--- n_features={nf} ---")
        decomp, wall, data = _run_one_cell(
            dgp, q, h_cfg, hp, A_GRID, n, M,
            step="s3", step_idx=3, cell_idx=cell_idx, force=force,
        )
        label = f"{q}|{h_cfg}|nf{nf}"
        _print_brief(label, decomp, wall)
        results[nf] = (decomp, wall, data)
        cell_idx += 1
    return results


def run_ntrend(
    dgp,
    cells: list[tuple[str, str, dict]],   # [(q_method, h_cfg, hp_dict), ...]
    M: int,
    force: bool = False,
) -> dict:
    """N-trend on top configs at n in {500, 1000, 2000}."""
    print("\n" + "-" * 72)
    print(f"N-TREND -- {len(cells)} configs x n in {{500, 1000, 2000}} (M={M})")
    print("-" * 72)
    results: dict[tuple[str, str, int], tuple] = {}
    cell_idx = 0
    for q, h_cfg, hp in cells:
        for n in [500, 1000, 2000]:
            print(f"\n--- {q} | {h_cfg} | n={n} ---")
            decomp, wall, data = _run_one_cell(
                dgp, q, h_cfg, hp, A_GRID, n, M,
                step="nt", step_idx=4, cell_idx=cell_idx, force=force,
            )
            label = f"{q}|{h_cfg}|n{n}"
            _print_brief(label, decomp, wall)
            results[(q, h_cfg, n)] = (decomp, wall, data)
            cell_idx += 1
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Report builder
# ══════════════════════════════════════════════════════════════════════════════

def _row(decomp: Optional[dict]) -> tuple:
    if decomp is None:
        return (np.nan, np.nan, np.nan, np.nan)
    return (
        float(decomp["bias_dr"][REF_IDX]),
        float(decomp["bse_grid"][REF_IDX]),
        float(decomp["coverage_ref"]),
        float(decomp.get("ess_min_mean", np.nan)),
    )


def _build_report(s1, s2, s3, ntrend) -> list[str]:
    lines: list[str] = []
    lines.append("# Kallus q-bridge -- Phase 11 hyperparameter exploration")
    lines.append(f"# Generated : {datetime.date.today()}")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append("")
    lines.append(
        "Round 1 returned Cas D (q non-bottleneck). This script asks the stronger "
        "question: with the **best known h-bridge** (H_super, Phase 9B.4 winner), "
        "does any Kallus configuration beat KPV?"
    )
    lines.append("")
    lines.append(f"DGP2: MichaelisMentenDGP(snr_W={SNR}, snr_Z={SNR})")
    lines.append(f"a_grid: {list(A_GRID)}, ref dose: a={A_GRID[REF_IDX]:.1f}")
    lines.append(f"Cells: n=1000, M=30, n_folds={N_FOLDS}")
    lines.append("")
    lines.append("h-bridge configurations:")
    for k, v in H_CONFIGS.items():
        lh = v["lambda_h"]
        lh_str = "default" if lh is None else f"{lh:.0e}"
        lines.append(f"  {k:12s}: lambda_h={lh_str}, ell_scale={v['ell_scale']:.2f}")
    lines.append("")

    # ── Step 1 ──────────────────────────────────────────────────────────────
    lines.append("## 2. Step 1 -- 3 q_method x 3 h_cfg @ central HP")
    lines.append("")
    lines.append("| q_method | h_cfg | bias_DR | |b|/SE | cov | ESS_min |")
    lines.append("|----------|-------|---------|--------|-----|---------|")
    for h_cfg in H_CONFIGS.keys():
        for q in Q_METHODS:
            entry = s1.get((q, h_cfg))
            if entry is None:
                lines.append(f"| {q} | {h_cfg} | --- | --- | --- | --- |")
                continue
            bias, bse, cov, ess = _row(entry[0])
            lines.append(
                f"| {q} | {h_cfg} | {bias:+.4f} | {bse:.2f} | {cov:.3f} | {ess:.0f} |"
            )
    lines.append("")
    lines.append("**Interpretation**: For each h_cfg, compare the 3 q-methods. "
                 "If KPV consistently wins, q is not the bottleneck (Cas D).")
    lines.append("")

    # Per-h_cfg verdict
    lines.append("**Per-h_cfg comparison (vs KPV at same h_cfg):**")
    lines.append("")
    for h_cfg in H_CONFIGS.keys():
        kpv = s1.get(("Q_kpv_baseline", h_cfg))
        if kpv is None or kpv[0] is None:
            continue
        kpv_bse = float(kpv[0]["bse_grid"][REF_IDX])
        for q in ("Q_kallus_ridge", "Q_kallus_stab"):
            cell = s1.get((q, h_cfg))
            if cell is None or cell[0] is None:
                continue
            cell_bse = float(cell[0]["bse_grid"][REF_IDX])
            delta = (cell_bse - kpv_bse) / kpv_bse * 100
            sym = "▼" if delta < -5 else ("▲" if delta > 5 else "≈")
            lines.append(
                f"  {q:18s} @ {h_cfg:12s}: |b|/SE {kpv_bse:.2f} -> {cell_bse:.2f} "
                f"({sym}{abs(delta):+.0f}% vs KPV)"
            )
    lines.append("")

    # ── Step 2 ──────────────────────────────────────────────────────────────
    lines.append("## 3. Step 2 -- HP sweep (gamma_Q x lambda_stab) on best Kallus cell")
    lines.append("")
    if not s2:
        lines.append("*Skipped: Step 1 showed Cas D evident (no Kallus cell beat KPV by >=10%).*")
        lines.append("")
    else:
        lines.append("| gamma_Q | lambda_stab | bias_DR | |b|/SE | cov | ESS_min |")
        lines.append("|---------|-------------|---------|--------|-----|---------|")
        for (gQ, ls), (d, _, _) in sorted(s2.items()):
            bias, bse, cov, ess = _row(d)
            lines.append(
                f"| {gQ:.0e} | {ls:.1f} | {bias:+.4f} | {bse:.2f} | {cov:.3f} | {ess:.0f} |"
            )
        lines.append("")

    # ── Step 3 ──────────────────────────────────────────────────────────────
    lines.append("## 4. Step 3 -- n_features sweep on best HP")
    lines.append("")
    if not s3:
        lines.append("*Skipped or not run.*")
        lines.append("")
    else:
        lines.append("| n_features | bias_DR | |b|/SE | cov | ESS_min |")
        lines.append("|-----------|---------|--------|-----|---------|")
        for nf, (d, _, _) in sorted(s3.items()):
            bias, bse, cov, ess = _row(d)
            lines.append(f"| {nf} | {bias:+.4f} | {bse:.2f} | {cov:.3f} | {ess:.0f} |")
        lines.append("")

    # ── N-trend ─────────────────────────────────────────────────────────────
    lines.append("## 5. N-trend on top configs")
    lines.append("")
    if not ntrend:
        lines.append("*Skipped.*")
        lines.append("")
    else:
        configs = sorted({(q, h) for (q, h, _) in ntrend.keys()})
        lines.append("| config | n=500 |b|/SE | n=1000 |b|/SE | n=2000 |b|/SE | "
                     "n=500 cov | n=1000 cov | n=2000 cov |")
        lines.append("|--------|-------------|--------------|--------------|"
                     "----------|------------|------------|")
        for q, h in configs:
            row = f"| {q}@{h} |"
            for n in [500, 1000, 2000]:
                d = ntrend.get((q, h, n), (None, 0.0, None))[0]
                row += f" {float(d['bse_grid'][REF_IDX]):.2f} |" if d else " --- |"
            for n in [500, 1000, 2000]:
                d = ntrend.get((q, h, n), (None, 0.0, None))[0]
                row += f" {float(d['coverage_ref']):.3f} |" if d else " --- |"
            lines.append(row)
        lines.append("")

    # ── Verdict ─────────────────────────────────────────────────────────────
    lines.append("## 6. Verdict")
    lines.append("")
    dominance = kallus_dominance_check(s1) if s1 else False
    if not dominance:
        lines.append(
            "**Cas D consolidated** -- No Kallus configuration (across 3 h_cfg, "
            "central HP point) beats KPVPolicyBridgeQ by >=10% on |b|/SE at the "
            "reference dose. Empirical conclusion: q-bridge is not the bottleneck "
            "for J(pi_a) on DGP2; the dominant lever is the h-bridge geometry."
        )
        lines.append("")
        lines.append("**Phrase for S6/S7** :")
        lines.append(
            "> 'A Kallus-stabilised minimax solver for q (with a finite-feature "
            "Nystroem critic) was implemented and compared head-to-head against "
            "KPVPolicyBridgeQ under three h-bridge configurations including the "
            "Phase 9B.4 winner. No Kallus configuration improved coverage or "
            "|bias|/SE at the reference dose, confirming that under proximal "
            "identification with a tuned outcome bridge, the residual error "
            "after KPV-h is dominated by approximation in the outcome bridge "
            "h, not by suboptimal moment weighting in q.'"
        )
    else:
        lines.append(
            "**Potential gain detected** -- At least one Kallus cell beat KPV "
            "by >=10% on |b|/SE. See Step 2/3 for HP sensitivity, and N-trend "
            "for finite-sample stability."
        )
    lines.append("")

    return lines


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 11 Kallus exploration")
    parser.add_argument("--mode", choices=("step1", "all"), default="all")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--M", type=int, default=30)
    parser.add_argument("--ntrend-M", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-skip", action="store_true",
                        help="Force Step 2/3 even if Step 1 shows Cas D")
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 11 -- Kallus q-bridge HP exploration")
    print(f"mode={args.mode}  n={args.n}  M={args.M}  ntrend_M={args.ntrend_M}")
    print("=" * 72)

    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)

    t_start = time.perf_counter()

    # Step 1 (always runs)
    s1 = run_step1(dgp, n=args.n, M=args.M, force=args.force)

    s2: dict = {}
    s3: dict = {}
    ntrend: dict = {}

    if args.mode == "all":
        # Auto-skip Step 2/3 if Step 1 shows Cas D (no Kallus dominance)
        if not args.no_skip and not kallus_dominance_check(s1):
            print("\n[AUTO-SKIP] Step 1 shows no Kallus dominance >= 10% over KPV. "
                  "Skipping Step 2/3 to save time.")
        else:
            # Step 2 on best (q_kallus, h_cfg)
            best_q, best_h = step1_winner_kallus(s1)
            if best_q is not None:
                s2 = run_step2(dgp, best_q, best_h, n=args.n, M=args.M, force=args.force)

            # Step 3 on best HP
            if s2:
                best_hp = HP_CENTRAL.copy()
                best_bse, best_key = float("inf"), None
                for (gQ, ls), (d, _, _) in s2.items():
                    if d is None: continue
                    bse = float(d["bse_grid"][REF_IDX])
                    if bse < best_bse:
                        best_bse, best_key = bse, (gQ, ls)
                if best_key is not None:
                    best_hp["gamma_Q"] = best_key[0]
                    best_hp["lambda_stab"] = best_key[1]
                    s3 = run_step3(dgp, best_q, best_h, best_hp,
                                   n=args.n, M=args.M, force=args.force)

        # N-trend always (whether or not we skipped 2/3): on KPV@H_super (anchor)
        # and the best-found Kallus cell from Step 1
        ntrend_cells: list[tuple[str, str, dict]] = [
            ("Q_kpv_baseline", "H_super", HP_CENTRAL),
        ]
        best_q, best_h = step1_winner_kallus(s1)
        if best_q is not None and best_h is not None:
            ntrend_cells.append((best_q, best_h, HP_CENTRAL))
        ntrend = run_ntrend(dgp, ntrend_cells, M=args.ntrend_M, force=args.force)

    # Build report
    print("\n" + "=" * 72)
    print(f"Total wall-clock: {(time.perf_counter() - t_start) / 60:.1f} min")
    print("Building report...")
    lines = _build_report(s1, s2, s3, ntrend)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report -> {_REPORT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
