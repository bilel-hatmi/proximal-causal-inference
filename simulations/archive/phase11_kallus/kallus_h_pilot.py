"""
Phase 11B -- KallusMinimaxBridgeH screening pilot.

Compares 3 KPV-h variants vs 3 Kallus-h variants on DGP2 at n=1000, M=30.
Then runs a q-oracle diagnostic on (a) KPV_super and (b) the best Kallus-h
to disambiguate "Kallus-h doesn't help" from "h/q interaction problem".

Cells (screening, all with Q_KPV)
----------------------------------
A : KPVBridgeH baseline (lambda_h default, ell_scale=1.0)
B : KPVBridgeH tuned (lambda_h=1e-4, ell_scale=2.5)
C : KPVBridgeH super (lambda_h=3e-5, ell_scale=3.5) -- Phase 9B.4 winner
D : KallusMinimaxBridgeH small (gamma_H=1e-3, lambda_stab_h=10, n_features=150)
E : KallusMinimaxBridgeH medium (gamma_H=3e-4, lambda_stab_h=1, n_features=250)
F : KallusMinimaxBridgeH rich (gamma_H=1e-4, lambda_stab_h=0.1, n_features=400)

q-oracle diagnostic (auto, after screening)
-------------------------------------------
G : C + Q_oracle (anchor pure)
H : KH_top + Q_oracle (interaction test)

Output
------
    simulations/results/raw/kallus_h_pilot/<label>.pkl
    simulations/results/summaries/kallus_h_pilot.md
"""
from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from simulations.experiments.dgp2_bias_diagnostics import (
    A_GRID, LAMBDA_Q_DEFAULT, N_FOLDS, RANDOM_STATE, REF_IDX, SNR,
    _compute_J_policy_true, _decompose, _median_bandwidth, _run_block, _silverman_h,
)

_BASE_DIR    = Path(__file__).resolve().parents[2]
_RAW_DIR     = _BASE_DIR / "simulations" / "results" / "raw" / "kallus_h_pilot"
_SUMM_DIR    = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "kallus_h_pilot.md"

_SEED_BASE = 19_000_000

# ── h-bridge configurations ──────────────────────────────────────────────────
KPV_HCFG = {
    "A_baseline": dict(lambda_h=None, ell_scale=1.0),
    "B_tuned":    dict(lambda_h=1e-4, ell_scale=2.5),
    "C_super":    dict(lambda_h=3e-5, ell_scale=3.5),
}

KH_HCFG = {
    "D_small":  dict(gamma_H=1e-3, lambda_stab_h=10.0, gamma_critic_h=1e-3,
                     n_features_h=150, n_features_critic=150, ell_scale=3.5),
    "E_medium": dict(gamma_H=3e-4, lambda_stab_h=1.0, gamma_critic_h=1e-3,
                     n_features_h=250, n_features_critic=250, ell_scale=3.5),
    "F_rich":   dict(gamma_H=1e-4, lambda_stab_h=0.1, gamma_critic_h=1e-4,
                     n_features_h=400, n_features_critic=400, ell_scale=3.5),
}


def _kpv_bridge_kwargs(h_cfg: str, ell_W0: float, ell_A0: float, ell_Z0: float) -> dict:
    spec = KPV_HCFG[h_cfg]
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
    cell_id: str,
    h_kind: str,         # "kpv" or "kallus"
    h_cfg: str,
    a_grid: np.ndarray,
    h_KDE: float,
    ell_W0: float, ell_A0: float, ell_Z0: float,
    use_q_oracle: bool = False,
    dgp=None,             # required if use_q_oracle
) -> Callable:
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
    from simulations.methods.kallus_minimax import KallusMinimaxBridgeH
    from simulations.methods._oracle_bridges import OracleBridgeQ

    bridge_kwargs = {}
    h_factory = None
    if h_kind == "kpv":
        bridge_kwargs = _kpv_bridge_kwargs(h_cfg, ell_W0, ell_A0, ell_Z0)
    else:
        kh = KH_HCFG[h_cfg]
        def make_kh():
            return KallusMinimaxBridgeH(
                mode="kallus_stabilized",
                **kh,
                compute_diagnostics="light",
            )
        h_factory = make_kh

    if use_q_oracle:
        if dgp is None:
            raise ValueError("dgp required for q-oracle factory")
        q_model = OracleBridgeQ(dgp)
        cross_fit_q = False
    else:
        def _make_q():
            return KPVPolicyBridgeQ(a_grid=a_grid, h_KDE=h_KDE,
                                    lambda_Q=LAMBDA_Q_DEFAULT, clip=None)
        q_model = _make_q()
        cross_fit_q = True

    def factory():
        return DRKernel(
            a_grid=a_grid, q_model=q_model, bridge_kwargs=bridge_kwargs,
            bandwidth=h_KDE, n_folds=N_FOLDS, random_state=RANDOM_STATE,
            ref_dose_index=REF_IDX, cross_fit_q=cross_fit_q,
            h_bridge_factory=h_factory,
        )
    return factory


def _label(cell_id: str, n: int, M: int, suffix: str = "") -> str:
    sfx = f"_{suffix}" if suffix else ""
    return f"kallush_{cell_id}{sfx}_n{n}_M{M}"


def _run_one_cell(
    dgp, cell_id: str, h_kind: str, h_cfg: str,
    a_grid: np.ndarray, n: int, M: int,
    cell_idx: int,
    use_q_oracle: bool = False,
    force: bool = False,
) -> tuple[Optional[dict], float, Optional[dict]]:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    ref_sample = dgp.generate(n=n, seed=0)
    h_KDE  = _silverman_h(ref_sample.A)
    ell_W0 = _median_bandwidth(ref_sample.W)
    ell_A0 = _median_bandwidth(ref_sample.A)
    ell_Z0 = _median_bandwidth(ref_sample.Z)
    J_pt   = _compute_J_policy_true(dgp, a_grid, h_KDE)
    factory = make_factory(
        cell_id, h_kind, h_cfg, a_grid, h_KDE, ell_W0, ell_A0, ell_Z0,
        use_q_oracle=use_q_oracle, dgp=dgp,
    )
    suffix = "qoracle" if use_q_oracle else ""
    label = _label(cell_id, n, M, suffix)
    seed_b = _SEED_BASE + cell_idx * 10_000 + n + (5000 if use_q_oracle else 0)

    if force:
        for sfx in (".pkl", ".partial.pkl"):
            p = _RAW_DIR / f"{label}{sfx}"
            if p.exists():
                p.unlink()

    t0 = time.perf_counter()
    data = _run_block(dgp, factory, a_grid, J_pt,
                      n=n, M=M, seed_base=seed_b, label=label, raw_dir=_RAW_DIR)
    wall = time.perf_counter() - t0
    return _decompose(data), wall, data


def _print_brief(label: str, decomp: Optional[dict], wall: float) -> None:
    if decomp is not None:
        wres = decomp.get("h_weighted_residual_mean", None)
        wres_str = f" hwres={float(wres):.3f}" if wres is not None and np.isfinite(wres) else ""
        print(
            f"    [{label}]  bias_DR={decomp['bias_dr'][REF_IDX]:+.4f}  "
            f"|b|/SE={decomp['bse_grid'][REF_IDX]:.2f}  "
            f"cov={decomp['coverage_ref']:.3f}  "
            f"ESS={decomp.get('ess_min_mean', float('nan')):.0f}{wres_str}  "
            f"wall={wall/60:.1f}min"
        )
    else:
        print(f"    [{label}]  no data (wall={wall/60:.1f}min)")


def _decomp_with_h_extras(decomp: Optional[dict], data: Optional[dict]) -> Optional[dict]:
    """Augment decomp with mean h_* extras across reps (needed since _decompose
    doesn't aggregate them)."""
    if decomp is None or data is None:
        return decomp
    recs = [r for r in data["records"] if r.get("error") is None]
    if not recs:
        return decomp
    if recs[0].get("h_kallus_present"):
        wres_vals = [r["h_weighted_residual_mean"] for r in recs
                     if np.isfinite(r.get("h_weighted_residual_mean", np.nan))]
        raw_vals = [r["h_raw_residual_mean"] for r in recs
                    if np.isfinite(r.get("h_raw_residual_mean", np.nan))]
        if wres_vals:
            decomp["h_weighted_residual_mean"] = float(np.mean(wres_vals))
        if raw_vals:
            decomp["h_raw_residual_mean"] = float(np.mean(raw_vals))
        decomp["h_method"] = recs[0].get("h_method", "?")
    return decomp


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 11B Kallus-h pilot")
    parser.add_argument("--mode", choices=("quick", "screen"), default="screen")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--M", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 11B -- KallusMinimaxBridgeH screening pilot")
    print(f"mode={args.mode}  n={args.n}  M={args.M}")
    print("=" * 72)

    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)

    if args.mode == "quick":
        cells_kpv = [("A_baseline", "kpv", "A_baseline")]
        cells_kh  = [("D_small", "kallus", "D_small")]
        n, M = 300, 5
    else:
        cells_kpv = [("A_baseline", "kpv", "A_baseline"),
                     ("B_tuned",    "kpv", "B_tuned"),
                     ("C_super",    "kpv", "C_super")]
        cells_kh  = [("D_small",  "kallus", "D_small"),
                     ("E_medium", "kallus", "E_medium"),
                     ("F_rich",   "kallus", "F_rich")]
        n, M = args.n, args.M

    t_start = time.perf_counter()
    results: dict = {}

    print("\n" + "-" * 72)
    print(f"SCREENING -- {len(cells_kpv) + len(cells_kh)} cells (Q_KPV)")
    print("-" * 72)
    for cell_idx, (cid, kind, h_cfg) in enumerate(cells_kpv + cells_kh):
        print(f"\n--- {cid} ({kind}) ---")
        decomp, wall, data = _run_one_cell(
            dgp, cid, kind, h_cfg, A_GRID, n, M,
            cell_idx=cell_idx, use_q_oracle=False, force=args.force,
        )
        decomp = _decomp_with_h_extras(decomp, data)
        _print_brief(f"{cid} (Q_KPV)", decomp, wall)
        results[cid] = (decomp, wall, data)

    # q-oracle diagnostic on C and KH_top
    if args.mode == "screen":
        print("\n" + "-" * 72)
        print("Q-ORACLE DIAGNOSTIC")
        print("-" * 72)
        # Find KH_top by best |b|/SE among D/E/F
        kh_cells = [(cid, results[cid][0]) for cid in ["D_small", "E_medium", "F_rich"]
                    if results[cid][0] is not None]
        kh_cells.sort(key=lambda kv: float(kv[1]["bse_grid"][REF_IDX]))
        if kh_cells:
            kh_top_id = kh_cells[0][0]
            print(f"  KH_top selected: {kh_top_id}")
            for cell_idx_off, (cid, kind, h_cfg, suffix) in enumerate([
                ("C_super", "kpv", "C_super", "qoracle"),
                (kh_top_id, "kallus", kh_top_id, "qoracle"),
            ]):
                print(f"\n--- {cid} (Q_oracle) ---")
                decomp, wall, data = _run_one_cell(
                    dgp, cid, kind, h_cfg, A_GRID, n, M,
                    cell_idx=100 + cell_idx_off,
                    use_q_oracle=True, force=args.force,
                )
                decomp = _decomp_with_h_extras(decomp, data)
                _print_brief(f"{cid} (Q_oracle)", decomp, wall)
                results[f"{cid}_qoracle"] = (decomp, wall, data)

    total_min = (time.perf_counter() - t_start) / 60.0
    print(f"\n{'=' * 72}\nTotal wall-clock: {total_min:.1f} min\n{'=' * 72}")

    # Build report
    print("Building report...")
    lines = _build_report(results, n, M)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report -> {_REPORT_PATH}")


def _row(decomp: Optional[dict]) -> tuple:
    if decomp is None:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    bias = float(decomp["bias_dr"][REF_IDX])
    bse  = float(decomp["bse_grid"][REF_IDX])
    cov  = float(decomp["coverage_ref"])
    ess  = float(decomp.get("ess_min_mean", np.nan))
    wres = decomp.get("h_weighted_residual_mean", np.nan)
    raw  = decomp.get("h_raw_residual_mean", np.nan)
    wres = float(wres) if wres is not None else np.nan
    raw  = float(raw)  if raw  is not None else np.nan
    return (bias, bse, cov, ess, wres, raw)


def _build_report(results: dict, n: int, M: int) -> list[str]:
    lines = []
    lines.append("# Phase 11B -- KallusMinimaxBridgeH screening pilot")
    lines.append(f"# Generated : {datetime.date.today()}")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append("")
    lines.append("Phase 11 q-only consolidated Cas B/D for q. Phase 11B asks the same "
                 "question h-side: can a Kallus-stabilised outcome bridge match or beat "
                 "the grid-tuned KPV H_super (Phase 9B.4 winner)?")
    lines.append("")
    lines.append(f"DGP2: MichaelisMentenDGP(snr_W={SNR}, snr_Z={SNR}), "
                 f"a_grid={list(A_GRID)}, ref a={A_GRID[REF_IDX]:.1f}, n={n}, M={M}, "
                 f"n_folds={N_FOLDS}.")
    lines.append("")
    lines.append("## 2. Screening (6 cells, Q_KPV)")
    lines.append("")
    lines.append("| Cell | h-bridge | bias_DR | |b|/SE | cov | ESS_min | h_wres | h_raw |")
    lines.append("|------|----------|---------|--------|-----|---------|--------|-------|")
    for cid in ["A_baseline", "B_tuned", "C_super",
                "D_small", "E_medium", "F_rich"]:
        entry = results.get(cid)
        if entry is None:
            lines.append(f"| {cid} | --- | --- | --- | --- | --- | --- | --- |")
            continue
        d, _, _ = entry
        bias, bse, cov, ess, wres, raw = _row(d)
        h_kind = "KPV" if cid.startswith(("A", "B", "C")) else "Kallus-h"
        wres_str = f"{wres:.3f}" if np.isfinite(wres) else "---"
        raw_str  = f"{raw:.3f}"  if np.isfinite(raw)  else "---"
        lines.append(
            f"| {cid} | {h_kind} | {bias:+.4f} | {bse:.2f} | {cov:.3f} | "
            f"{ess:.0f} | {wres_str} | {raw_str} |"
        )
    lines.append("")

    # Q-oracle diagnostic
    qor = {k: v for k, v in results.items() if k.endswith("_qoracle")}
    lines.append("## 3. Q-oracle diagnostic")
    lines.append("")
    if not qor:
        lines.append("*Skipped (quick mode).*")
        lines.append("")
    else:
        lines.append("| Cell | h-bridge | bias_DR | |b|/SE | cov | ESS_min |")
        lines.append("|------|----------|---------|--------|-----|---------|")
        for cid, entry in sorted(qor.items()):
            d, _, _ = entry
            bias, bse, cov, ess, _, _ = _row(d)
            lines.append(
                f"| {cid} | {'KPV' if 'C_super' in cid else 'Kallus-h'} "
                f"| {bias:+.4f} | {bse:.2f} | {cov:.3f} | {ess:.0f} |"
            )
        lines.append("")

    # Verdict per Section 12 of plan
    lines.append("## 4. Verdict (per Phase 11B plan Section 12)")
    lines.append("")
    C = results.get("C_super")
    if C is None or C[0] is None:
        lines.append("*C_super failed; verdict undetermined.*")
        lines.append("")
        return lines
    C_bse = float(C[0]["bse_grid"][REF_IDX])
    C_cov = float(C[0]["coverage_ref"])
    C_ess = float(C[0].get("ess_min_mean", 0))

    # Find KH_top
    kh_results = [(cid, results.get(cid)) for cid in ["D_small", "E_medium", "F_rich"]]
    kh_results = [(cid, e) for cid, e in kh_results if e is not None and e[0] is not None]
    if not kh_results:
        lines.append("*All Kallus-h cells failed; verdict = Cas 4 (close Kallus).*")
        return lines
    kh_results.sort(key=lambda kv: float(kv[1][0]["bse_grid"][REF_IDX]))
    kh_top_id, (kh_top, _, _) = kh_results[0]
    kh_bse = float(kh_top["bse_grid"][REF_IDX])
    kh_cov = float(kh_top["coverage_ref"])
    kh_ess = float(kh_top.get("ess_min_mean", 0))

    delta_pct = (kh_bse - C_bse) / max(C_bse, 1e-9) * 100
    if kh_bse <= 0.90 * C_bse and kh_cov >= C_cov and kh_ess >= 0.5 * C_ess:
        verdict = "Cas 1 (gain réel) -- KH_top bat C_super >=10%, lancer validation N-trend"
    elif abs(delta_pct) < 10:
        verdict = "Cas 2 (proche) -- KH_top approche C_super (within 10%); analyser blind tuning"
    elif kh_bse < float(results["A_baseline"][0]["bse_grid"][REF_IDX]):
        verdict = "Cas 3 (intermédiaire) -- KH_top bat A_baseline mais perd à C_super"
    else:
        verdict = "Cas 4 (échec) -- KH_top n'améliore rien ; fermer Kallus, Phase 12 Bennett-lite"

    lines.append(f"**Best Kallus-h** : {kh_top_id} (|b|/SE={kh_bse:.2f}, cov={kh_cov:.3f}, ESS={kh_ess:.0f})")
    lines.append(f"**Anchor C_super** : |b|/SE={C_bse:.2f}, cov={C_cov:.3f}, ESS={C_ess:.0f}")
    lines.append(f"**Δ |b|/SE** : {delta_pct:+.0f}% (vs KPV_super)")
    lines.append("")
    lines.append(f"**Verdict** : {verdict}")
    lines.append("")
    return lines


if __name__ == "__main__":
    main()
