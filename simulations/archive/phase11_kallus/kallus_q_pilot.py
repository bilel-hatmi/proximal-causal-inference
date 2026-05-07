"""
Kallus-Stabilized Policy Q -- Phase 11 controlled pilot.

Compares four q-bridge solvers under DRKernel on DGP2 (and a DGP1 sanity slice):
    Q_kpv_baseline   : KPVPolicyBridgeQ (kernel-space ridge/GMM, existing)
    Q_kallus_ridge   : KallusStabilizedPolicyQ(mode="ridge_gmm")
    Q_kallus_stab    : KallusStabilizedPolicyQ(mode="kallus_stabilized")
    (oracle is skipped here -- DGP2 has no closed-form q0; can be re-added
     later for DGP1 sanity if needed)

For each q method, we run two h-bridge configurations:
    H_baseline : KPVBridgeH defaults
    H_tuned    : KPVBridgeH(lambda_1=lambda_2=1e-4, ell_scale=2.5)

Modes
-----
    quick : DGP2 n=300, M=5, 2 methods (kpv_baseline + kallus_stab) under H_baseline
            ~5 min. Smoke pipeline + report generation.
    pilot --round 1 :
            DGP1 sanity (n=500, M=20, 2 methods, H_baseline)
          + DGP2 (n=1000, M=30, 3 methods x 2 h_configs)
            estimated ~30-60 min.
    pilot --round 2 :
            DGP2 (n=1000, M=50, 3 methods x 2 h_configs)
            ~60-90 min. Only if Round 1 shows Cas A or B.

Output
------
    simulations/results/raw/kallus_q_pilot/kallusq_<method>_<hcfg>_n<n>_M<M>.pkl
    simulations/results/summaries/kallus_q_pilot.md

Seeds : 16_000_000 + cfg_idx*10_000 + n   (disjoint from all prior phases).
"""
from __future__ import annotations

import argparse
import datetime
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# ── Reuse helpers from dgp2_bias_diagnostics ─────────────────────────────────
from simulations.experiments.dgp2_bias_diagnostics import (
    A_GRID,
    CHECKPOINT_FREQ,
    K,
    LAMBDA_Q_DEFAULT,
    N_FOLDS,
    RANDOM_STATE,
    REF_IDX,
    SNR,
    _build_record,
    _compute_J_policy_true,
    _decompose,
    _median_bandwidth,
    _run_block,
    _silverman_h,
)

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE_DIR    = Path(__file__).resolve().parents[2]
_RAW_DIR     = _BASE_DIR / "simulations" / "results" / "raw" / "kallus_q_pilot"
_SUMM_DIR    = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "kallus_q_pilot.md"

_SEED_BASE = 16_000_000

# ── h-bridge configurations ──────────────────────────────────────────────────
_H_CONFIGS: dict[str, dict] = {
    "H_baseline": dict(lambda_h=None, ell_scale=1.0),
    "H_tuned":    dict(lambda_h=1e-4,  ell_scale=2.5),
}

# ── q-bridge methods ─────────────────────────────────────────────────────────
_Q_METHODS = ("Q_kpv_baseline", "Q_kallus_ridge", "Q_kallus_stab")

# Default Kallus hyperparameters (Round 1)
_KALLUS_HP = dict(
    n_features_q  = 150,
    n_features_h  = 150,
    gamma_Q       = 1e-3,
    lambda_stab   = 1.0,
    gamma_critic  = 1e-3,
    feature_seed  = 20260427,
    compute_diagnostics = "light",
)

# Configuration order (for seed assignment)
_CONFIGS: list[tuple[str, str]] = [
    (q, h) for q in _Q_METHODS for h in _H_CONFIGS.keys()
]


# ══════════════════════════════════════════════════════════════════════════════
#  Factory builder
# ══════════════════════════════════════════════════════════════════════════════

def _h_bridge_kwargs(h_cfg: str, ell_W0: float, ell_A0: float, ell_Z0: float) -> dict:
    spec = _H_CONFIGS[h_cfg]
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
    ell_W0: float,
    ell_A0: float,
    ell_Z0: float,
) -> Callable:
    """Build a zero-arg factory closing over a (q_method, h_cfg) cell."""
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
    from simulations.methods.kallus_minimax import KallusStabilizedPolicyQ

    bridge_kwargs = _h_bridge_kwargs(h_cfg, ell_W0, ell_A0, ell_Z0)

    if q_method == "Q_kpv_baseline":
        def make_q():
            return KPVPolicyBridgeQ(
                a_grid=a_grid, h_KDE=h_KDE,
                lambda_Q=LAMBDA_Q_DEFAULT,
                clip=None,                 # CLIP_DEFAULT can be heavy; let predictor handle
            )
    elif q_method == "Q_kallus_ridge":
        def make_q():
            return KallusStabilizedPolicyQ(
                a_grid=a_grid, h_KDE=h_KDE,
                mode="ridge_gmm",
                **_KALLUS_HP,
            )
    elif q_method == "Q_kallus_stab":
        def make_q():
            return KallusStabilizedPolicyQ(
                a_grid=a_grid, h_KDE=h_KDE,
                mode="kallus_stabilized",
                **_KALLUS_HP,
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
#  Per-block runner
# ══════════════════════════════════════════════════════════════════════════════

def _seed_base(q_method: str, h_cfg: str, n: int, dgp_tag: str) -> int:
    """Deterministic, disjoint from all prior phases (>= 16_000_000)."""
    cfg_idx = _CONFIGS.index((q_method, h_cfg))
    dgp_off = 0 if dgp_tag == "dgp2" else 5_000_000
    return _SEED_BASE + dgp_off + cfg_idx * 10_000 + int(n)


def _label(q_method: str, h_cfg: str, n: int, M: int, dgp_tag: str) -> str:
    return f"kallusq_{dgp_tag}_{q_method}_{h_cfg}_n{n}_M{M}"


def _run_one_cell(
    dgp,
    dgp_tag: str,
    q_method: str,
    h_cfg: str,
    a_grid: np.ndarray,
    n: int,
    M: int,
    force: bool = False,
) -> tuple[Optional[dict], float, Optional[dict]]:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    ref_sample = dgp.generate(n=n, seed=0)
    h_KDE = _silverman_h(ref_sample.A)
    ell_W0 = _median_bandwidth(ref_sample.W)
    ell_A0 = _median_bandwidth(ref_sample.A)
    ell_Z0 = _median_bandwidth(ref_sample.Z)

    J_pt = _compute_J_policy_true(dgp, a_grid, h_KDE)
    factory = make_factory(q_method, h_cfg, a_grid, h_KDE, ell_W0, ell_A0, ell_Z0)

    label = _label(q_method, h_cfg, n, M, dgp_tag)
    seed_b = _seed_base(q_method, h_cfg, n, dgp_tag)

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


def _print_brief(q_method: str, h_cfg: str, n: int, decomp: Optional[dict], wall: float) -> None:
    if decomp is not None:
        print(
            f"    [{q_method} | {h_cfg}] n={n}: "
            f"bias_DR={decomp['bias_dr'][REF_IDX]:+.4f}  "
            f"|b|/SE={decomp['bse_grid'][REF_IDX]:.2f}  "
            f"cov={decomp['coverage_ref']:.3f}  "
            f"wall={wall/60:.1f}min"
        )
    else:
        print(f"    [{q_method} | {h_cfg}] n={n}: NO DATA (wall={wall/60:.1f}min)")


# ══════════════════════════════════════════════════════════════════════════════
#  Block runners
# ══════════════════════════════════════════════════════════════════════════════

def run_dgp1_sanity(M: int, force: bool = False) -> dict:
    """Run a small DGP1 sanity block. Returns dict of (q,h) -> (decomp, wall, data)."""
    from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP
    dgp = CobbDouglasLinearDGP(snr_W=SNR, snr_Z=SNR)
    a_grid_dgp1 = np.array([1.0, 2.0, 3.0])
    n = 500
    results: dict[tuple[str, str], tuple] = {}
    for q in ("Q_kpv_baseline", "Q_kallus_stab"):
        for h_cfg in ("H_baseline",):
            print(f"\n--- DGP1 sanity | {q} | {h_cfg} | n={n} | M={M} ---")
            decomp, wall, data = _run_one_cell(
                dgp, "dgp1", q, h_cfg, a_grid_dgp1, n, M, force=force,
            )
            _print_brief(q, h_cfg, n, decomp, wall)
            results[(q, h_cfg)] = (decomp, wall, data)
    return results


def run_dgp2_grid(
    methods: list[str],
    h_configs: list[str],
    n: int,
    M: int,
    force: bool = False,
) -> dict:
    """Run the full DGP2 method x h_config grid."""
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)
    results: dict[tuple[str, str], tuple] = {}
    for q in methods:
        for h_cfg in h_configs:
            print(f"\n--- DGP2 | {q} | {h_cfg} | n={n} | M={M} ---")
            decomp, wall, data = _run_one_cell(
                dgp, "dgp2", q, h_cfg, A_GRID, n, M, force=force,
            )
            _print_brief(q, h_cfg, n, decomp, wall)
            results[(q, h_cfg)] = (decomp, wall, data)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Report builder
# ══════════════════════════════════════════════════════════════════════════════

def _row(decomp: Optional[dict], data: Optional[dict]) -> tuple:
    """Extract metric tuple for one cell (decomp, data)."""
    if decomp is None:
        return (np.nan,) * 7
    bias_dr = float(decomp["bias_dr"][REF_IDX])
    bse     = float(decomp["bse_grid"][REF_IDX])
    cov     = float(decomp["coverage_ref"])
    ess     = float(decomp.get("ess_min_mean", np.nan))
    # weighted_residual mean (Kallus only)
    wres = np.nan
    weight_p99_train = np.nan
    if data is not None and "records" in data:
        kallus_w = []
        kallus_wp99 = []
        for rec in data["records"]:
            if not rec.get("ok"):
                continue
            ex = rec.get("extra", {})
            if ex.get("kallus_present", False):
                arr = ex.get("kallus_weighted_residual_grid")
                if arr is not None:
                    kallus_w.append(float(np.nanmean(arr)))
                arr2 = ex.get("kallus_weight_p99_grid")
                if arr2 is not None:
                    kallus_wp99.append(float(np.nanmean(arr2)))
        if kallus_w:
            wres = float(np.mean(kallus_w))
        if kallus_wp99:
            weight_p99_train = float(np.mean(kallus_wp99))
    # weight_p99 from DRKernel test-fold side (always available)
    test_w_p99 = np.nan
    if data is not None and "records" in data:
        vals = []
        for rec in data["records"]:
            if not rec.get("ok"):
                continue
            ex = rec.get("extra", {})
            arr = ex.get("weight_p99_grid")
            if arr is not None:
                vals.append(float(arr[REF_IDX]))
        if vals:
            test_w_p99 = float(np.mean(vals))
    return bias_dr, bse, cov, ess, wres, weight_p99_train, test_w_p99


def _build_report(
    dgp1_res: Optional[dict],
    dgp2_res: dict,
    n_dgp2: int,
    M_dgp2: int,
) -> list[str]:
    lines: list[str] = []
    lines.append("# Kallus-Stabilized Policy Q -- Pilot Report (Phase 11)")
    lines.append(f"# Generated : {datetime.date.today()}")
    lines.append("")

    # ── Section 1 ────────────────────────────────────────────────────────────
    lines.append("## 1. Objective and design")
    lines.append("")
    lines.append(
        "Compare four q-bridge solvers under DRKernel: KPVPolicyBridgeQ "
        "(kernel ridge/GMM, existing baseline), KallusStabilizedPolicyQ in "
        "two modes (ridge_gmm and kallus_stabilized), under two h-bridge "
        "configurations (H_baseline, H_tuned)."
    )
    lines.append("")
    lines.append(f"DGP2: MichaelisMentenDGP(snr_W={SNR}, snr_Z={SNR})")
    lines.append(f"a_grid: {list(A_GRID)}, ref dose: a={A_GRID[REF_IDX]:.1f} (REF_IDX={REF_IDX})")
    lines.append(f"n = {n_dgp2}, M = {M_dgp2}, n_folds = {N_FOLDS}")
    lines.append("")
    lines.append("Kallus hyperparameters (Round 1):")
    for k, v in _KALLUS_HP.items():
        lines.append(f"  {k} = {v}")
    lines.append("")

    # ── Section 2 -- DGP1 sanity ─────────────────────────────────────────────
    lines.append("## 2. DGP1 sanity")
    lines.append("")
    if not dgp1_res:
        lines.append("*Skipped (quick mode or not run).*")
        lines.append("")
    else:
        lines.append("| method | h_cfg | bias_DR | |b|/SE | cov_ref | ESS_min |")
        lines.append("|--------|-------|---------|--------|---------|---------|")
        for (q, h_cfg), (decomp, _, data) in dgp1_res.items():
            bias, bse, cov, ess, _, _, _ = _row(decomp, data)
            lines.append(
                f"| {q} | {h_cfg} | {bias:+.4f} | {bse:.2f} | {cov:.3f} | {ess:.0f} |"
            )
        lines.append("")
        lines.append(
            "DGP1 sanity passes if KallusStabilizedPolicyQ does not catastrophically "
            "underperform KPVPolicyBridgeQ (within a factor 2 on |bias_DR|)."
        )
        lines.append("")

    # ── Section 3 -- DGP2 main comparison ────────────────────────────────────
    lines.append("## 3. DGP2 method x h-config comparison (ref dose)")
    lines.append("")
    lines.append(
        "| method | h_cfg | bias_DR | |b|/SE | cov_ref | ESS_min | "
        "wres (Kallus) | w_p99 (test) |"
    )
    lines.append(
        "|--------|-------|---------|--------|---------|---------|"
        "---------------|--------------|"
    )
    for q in _Q_METHODS:
        for h_cfg in _H_CONFIGS.keys():
            entry = dgp2_res.get((q, h_cfg))
            if entry is None:
                lines.append(f"| {q} | {h_cfg} | --- | --- | --- | --- | --- | --- |")
                continue
            decomp, _, data = entry
            bias, bse, cov, ess, wres, _, tw = _row(decomp, data)
            wres_str = f"{wres:.4f}" if np.isfinite(wres) else "---"
            tw_str   = f"{tw:.2f}"   if np.isfinite(tw)   else "---"
            lines.append(
                f"| {q} | {h_cfg} | {bias:+.4f} | {bse:.2f} | {cov:.3f} | "
                f"{ess:.0f} | {wres_str} | {tw_str} |"
            )
    lines.append("")

    # ── Section 4 -- Verdict ────────────────────────────────────────────────
    lines.append("## 4. Verdict (vs Q_kpv_baseline at same h_cfg)")
    lines.append("")
    lines.append(
        "Cas A (gain reel)        : weighted_residual lower AND |bias_DR|/SE lower by >=10% "
        "AND ESS_min stable (>15) AND test-side w_p99 < 2x baseline."
    )
    lines.append(
        "Cas B (critic without target) : weighted_residual lower BUT |bias_DR|/SE within 5%."
    )
    lines.append(
        "Cas C (instable)         : |bias_DR| lower BUT ESS_min < 10 OR w_p99 > 3x baseline."
    )
    lines.append(
        "Cas D (q non-bottleneck) : nothing moves significantly."
    )
    lines.append("")
    for h_cfg in _H_CONFIGS.keys():
        base = dgp2_res.get(("Q_kpv_baseline", h_cfg))
        if base is None:
            continue
        b_bias, b_bse, b_cov, b_ess, _, _, b_w = _row(base[0], base[2])
        for q in ("Q_kallus_ridge", "Q_kallus_stab"):
            cell = dgp2_res.get((q, h_cfg))
            if cell is None:
                continue
            bias, bse, cov, ess, wres, _, w = _row(cell[0], cell[2])
            verdict = _classify_verdict(
                base_bse=b_bse, base_ess=b_ess, base_w=b_w,
                cell_bse=bse,   cell_ess=ess, cell_w=w,
                wres=wres,
            )
            lines.append(
                f"  {q} vs Q_kpv_baseline @ {h_cfg}: {verdict} "
                f"(|b|/SE: {b_bse:.2f} -> {bse:.2f}, ESS: {b_ess:.0f} -> {ess:.0f})"
            )
        lines.append("")

    return lines


def _classify_verdict(
    base_bse: float, base_ess: float, base_w: float,
    cell_bse: float, cell_ess: float, cell_w: float,
    wres: float,
) -> str:
    """Return Cas A/B/C/D label."""
    if not (np.isfinite(cell_bse) and np.isfinite(base_bse)):
        return "(insufficient data)"
    bse_drop = (base_bse - cell_bse) / max(abs(base_bse), 1e-9)
    ess_ok = (cell_ess >= 15) and (cell_ess > 0.5 * base_ess)
    w_ratio = cell_w / max(base_w, 1e-9) if np.isfinite(base_w) and np.isfinite(cell_w) else np.nan

    if np.isfinite(w_ratio) and w_ratio > 3.0:
        return "Cas C (instable -- w_p99 explodes)"
    if not ess_ok and bse_drop > 0:
        return "Cas C (instable -- ESS collapses)"
    if bse_drop >= 0.10:
        if np.isfinite(w_ratio) and w_ratio > 2.0:
            return "Cas C (instable -- w_p99 inflates)"
        return "Cas A (gain reel)"
    if abs(bse_drop) < 0.05:
        if np.isfinite(wres):
            return "Cas B (critic without target)"
        return "Cas D (q non-bottleneck)"
    return "Cas D (q non-bottleneck)"


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 11 Kallus q pilot")
    parser.add_argument(
        "--mode", choices=("quick", "pilot"), default="quick",
        help="quick = smoke test; pilot = Round 1/2 (use --round)",
    )
    parser.add_argument(
        "--round", type=int, default=1, choices=(1, 2),
        help="pilot round: 1 (n=1000, M=30) or 2 (n=1000, M=50)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if PKLs exist",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Only rebuild report from existing PKLs",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 11 -- Kallus-Stabilized Policy Q pilot")
    print(f"mode={args.mode}  round={args.round}  force={args.force}")
    print("=" * 72)

    if args.report_only:
        # Best-effort: scan the raw dir and rebuild a report only from existing PKLs
        # (left as a future utility; for now require an actual run)
        print("--report-only not yet implemented; run a mode instead.")
        return

    if args.mode == "quick":
        n_dgp2, M_dgp2 = 300, 5
        methods   = ["Q_kpv_baseline", "Q_kallus_stab"]
        h_configs = ["H_baseline"]
        dgp1_res = None
    else:
        # pilot
        n_dgp2 = 1000
        M_dgp2 = 30 if args.round == 1 else 50
        methods   = list(_Q_METHODS)
        h_configs = list(_H_CONFIGS.keys())
        # DGP1 sanity only on Round 1
        if args.round == 1:
            print("\n" + "-" * 72)
            print("DGP1 sanity block")
            print("-" * 72)
            dgp1_res = run_dgp1_sanity(M=20, force=args.force)
        else:
            dgp1_res = None

    print("\n" + "-" * 72)
    print(f"DGP2 grid: {len(methods)} methods x {len(h_configs)} h_configs")
    print(f"  n = {n_dgp2}, M = {M_dgp2}")
    print("-" * 72)
    t0 = time.perf_counter()
    dgp2_res = run_dgp2_grid(
        methods=methods, h_configs=h_configs,
        n=n_dgp2, M=M_dgp2, force=args.force,
    )
    wall_dgp2 = (time.perf_counter() - t0) / 60.0
    print(f"\nDGP2 grid done in {wall_dgp2:.1f} min.")

    # Build report
    print("\nBuilding report...")
    lines = _build_report(dgp1_res, dgp2_res, n_dgp2, M_dgp2)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report -> {_REPORT_PATH}")

    print("\n" + "=" * 72)
    print("Pilot complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
