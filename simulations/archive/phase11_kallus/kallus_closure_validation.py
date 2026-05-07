"""
Phase 11C+ -- Final Kallus closure validation.

Validates the surprise B2 finding from Phase 11C (KPV_super + Q_Kallus_match nf=400
gave -26% on |b|/SE vs A at n=2000, M=100) with disjoint seeds and larger M.

Configurations
--------------
A  : KPV_super (lam=3e-5, ell=3.5)             + Q_KPV
B2 : KPV_super                                   + Q_Kallus_match (nf=400, gamma_Q=1e-3,
                                                    lambda_stab=1.0, gamma_critic=1e-3)
D2 (optional) : Kallus_h F_rich + Q_Kallus_match -- only if A and B2 done

Parameters
----------
DGP   : MichaelisMentenDGP(snr_W=snr_Z=0.95)
n     : 2000 (main); 300 (smoke)
M     : 200 (main); 3 (smoke)
seed_base : 22_000_000 (disjoint Phase 11C = 21M)

Variance fix
------------
Phase 11C report had NaN for V_total/V_reg/V_corr because the report builder looked
for lowercase "v_total" keys, but `_decompose` returns uppercase "V_total". Fixed
here. SER (empirical_SE / predicted_SE per rep) computed locally since not in
`_decompose`.

Output
------
  simulations/results/raw/kallus_closure/<label>.pkl
  simulations/results/summaries/phase11C_plus_kallus_closure.md
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
_RAW_DIR     = _BASE_DIR / "simulations" / "results" / "raw" / "kallus_closure"
_SUMM_DIR    = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "phase11C_plus_kallus_closure.md"

_SEED_BASE = 22_000_000

# ── Frozen HP configs (matching Phase 11C) ───────────────────────────────────
KPV_SUPER = dict(lambda_h=3e-5, ell_scale=3.5)

KALLUS_F_RICH = dict(
    gamma_H=1e-4, lambda_stab_h=0.1, gamma_critic_h=1e-4,
    n_features_h=400, n_features_critic=400,
    ell_scale=3.5,
)

KALLUS_Q_MATCH = dict(
    n_features_q=400, n_features_h=400,
    gamma_Q=1e-3, lambda_stab=1.0, gamma_critic=1e-3,
    feature_seed=20260427,
    compute_diagnostics="light",
)

# Cell registry
CELLS: dict[str, tuple[str, str]] = {
    "A":  ("kpv_super",   "kpv"),
    "B2": ("kpv_super",   "kallus_match"),
    "D2": ("kallus_rich", "kallus_match"),
}
CELL_INDEX = {cid: i for i, cid in enumerate(["A", "B2", "D2"])}


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

def _kpv_super_kwargs(ell_W0, ell_A0, ell_Z0) -> dict:
    return dict(
        lambda_1=KPV_SUPER["lambda_h"], lambda_2=KPV_SUPER["lambda_h"],
        ell_W=ell_W0 * KPV_SUPER["ell_scale"],
        ell_A=ell_A0 * KPV_SUPER["ell_scale"],
        ell_Z=ell_Z0 * KPV_SUPER["ell_scale"],
    )


def make_factory(cid: str, h_KDE, ell_W0, ell_A0, ell_Z0) -> Callable:
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
    from simulations.methods.kallus_minimax import (
        KallusStabilizedPolicyQ, KallusMinimaxBridgeH,
    )

    h_kind, q_kind = CELLS[cid]

    bridge_kwargs: dict = {}
    h_factory = None
    if h_kind == "kpv_super":
        bridge_kwargs = _kpv_super_kwargs(ell_W0, ell_A0, ell_Z0)
    elif h_kind == "kallus_rich":
        def h_factory():
            return KallusMinimaxBridgeH(
                mode="kallus_stabilized", **KALLUS_F_RICH,
                compute_diagnostics="light",
            )

    if q_kind == "kpv":
        q_model = KPVPolicyBridgeQ(a_grid=A_GRID, h_KDE=h_KDE,
                                    lambda_Q=LAMBDA_Q_DEFAULT, clip=None)
    elif q_kind == "kallus_match":
        q_model = KallusStabilizedPolicyQ(
            a_grid=A_GRID, h_KDE=h_KDE, mode="kallus_stabilized", **KALLUS_Q_MATCH,
        )
    else:
        raise ValueError(f"Unknown q_kind {q_kind}")

    def factory():
        return DRKernel(
            a_grid=A_GRID, q_model=q_model, bridge_kwargs=bridge_kwargs,
            bandwidth=h_KDE, n_folds=N_FOLDS, random_state=RANDOM_STATE,
            ref_dose_index=REF_IDX, cross_fit_q=True,
            h_bridge_factory=h_factory,
        )
    return factory


# ══════════════════════════════════════════════════════════════════════════════
#  Per-cell run + augmented decomp (variance + SER fix)
# ══════════════════════════════════════════════════════════════════════════════

def _label(cid: str, n: int, M: int) -> str:
    return f"kallusclo_{cid}_n{n}_M{M}"


def _augment_decomp(decomp: Optional[dict], data: Optional[dict]) -> Optional[dict]:
    """Add V_*_grid (already in decomp under uppercase keys), q/h Kallus extras,
    SER (empirical_SE / predicted_SE), w_p99_per_cell."""
    if decomp is None or data is None:
        return decomp
    recs = [r for r in data["records"] if r.get("error") is None]
    if not recs:
        return decomp

    # SER: empirical SE = std(J_dr across reps) ; predicted SE already in decomp
    J_dr_all = np.array([r["J_dr"] for r in recs])      # (M, K)
    emp_SE = np.std(J_dr_all, axis=0, ddof=1)            # (K,)
    pred_SE = np.array(decomp["SE_grid"])                # (K,)
    with np.errstate(divide="ignore", invalid="ignore"):
        ser_grid = emp_SE / np.where(pred_SE > 1e-15, pred_SE, np.nan)
    decomp["ser_grid"] = ser_grid
    decomp["ser_ref"]  = float(ser_grid[REF_IDX])
    decomp["emp_SE_grid"] = emp_SE

    # M_ok ratio
    decomp["M_ok_ratio"] = decomp["M_ok"] / int(data["meta"]["M"])

    # w_p99 per cell (already w_p99_mean in decomp). Add max across reps for safety
    w99_per_rep = np.array([r["weight_p99_grid"] for r in recs])  # (M, K)
    decomp["w_p99_grid_mean"] = np.nanmean(w99_per_rep, axis=0)   # (K,)
    decomp["w_p99_max"] = float(np.nanmax(w99_per_rep))

    # h-Kallus + q-Kallus extras (mean across reps)
    r0 = recs[0]
    if r0.get("h_kallus_present"):
        wres = [r["h_weighted_residual_mean"] for r in recs
                if np.isfinite(r.get("h_weighted_residual_mean", np.nan))]
        if wres:
            decomp["h_weighted_residual_mean"] = float(np.mean(wres))
        raw = [r["h_raw_residual_mean"] for r in recs
               if np.isfinite(r.get("h_raw_residual_mean", np.nan))]
        if raw:
            decomp["h_raw_residual_mean"] = float(np.mean(raw))
        decomp["h_method"] = r0.get("h_method", "?")
    if r0.get("kallus_present"):
        q_wres = []
        for r in recs:
            arr = r.get("kallus_weighted_residual_grid")
            if arr is not None:
                q_wres.append(float(np.nanmean(arr)))
        if q_wres:
            decomp["q_weighted_residual_mean"] = float(np.mean(q_wres))

    return decomp


def _run_one_cell(dgp, cid: str, n: int, M: int, force: bool = False
                  ) -> tuple[Optional[dict], float, Optional[dict]]:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    ref_sample = dgp.generate(n=n, seed=0)
    h_KDE  = _silverman_h(ref_sample.A)
    ell_W0 = _median_bandwidth(ref_sample.W)
    ell_A0 = _median_bandwidth(ref_sample.A)
    ell_Z0 = _median_bandwidth(ref_sample.Z)
    J_pt   = _compute_J_policy_true(dgp, A_GRID, h_KDE)
    factory = make_factory(cid, h_KDE, ell_W0, ell_A0, ell_Z0)

    label = _label(cid, n, M)
    seed_b = _SEED_BASE + CELL_INDEX[cid] * 100_000 + int(n)

    if force:
        for sfx in (".pkl", ".partial.pkl"):
            p = _RAW_DIR / f"{label}{sfx}"
            if p.exists():
                p.unlink()

    t0 = time.perf_counter()
    data = _run_block(dgp, factory, A_GRID, J_pt, n=n, M=M,
                      seed_base=seed_b, label=label, raw_dir=_RAW_DIR)
    wall = time.perf_counter() - t0
    decomp = _augment_decomp(_decompose(data), data)
    return decomp, wall, data


def _print_brief(cid: str, n: int, M: int, decomp: Optional[dict], wall: float) -> None:
    if decomp is not None:
        h_w = decomp.get("h_weighted_residual_mean")
        q_w = decomp.get("q_weighted_residual_mean")
        h_str = f" h_wres={float(h_w):.3f}" if h_w is not None and np.isfinite(float(h_w)) else ""
        q_str = f" q_wres={float(q_w):.3f}" if q_w is not None and np.isfinite(float(q_w)) else ""
        print(
            f"    [{cid} n={n} M={M}]  "
            f"bias_REG={decomp['bias_reg'][REF_IDX]:+.4f}  "
            f"bias_DR={decomp['bias_dr'][REF_IDX]:+.4f}  "
            f"|b|/SE={decomp['bse_grid'][REF_IDX]:.2f}  "
            f"cov={decomp['coverage_ref']:.3f}  "
            f"SER={decomp.get('ser_ref', float('nan')):.2f}  "
            f"ESS={decomp.get('ess_min_mean', float('nan')):.0f}{h_str}{q_str}  "
            f"wall={wall/60:.1f}min"
        )
    else:
        print(f"    [{cid} n={n} M={M}]  no data (wall={wall/60:.1f}min)")


# ══════════════════════════════════════════════════════════════════════════════
#  Report builder
# ══════════════════════════════════════════════════════════════════════════════

def _row_full(d: Optional[dict]) -> dict:
    if d is None:
        return {k: float("nan") for k in
                ["bias_reg", "bias_dr", "bse", "cov", "ser", "ess", "w_p99",
                 "h_wres", "q_wres", "v_total", "v_reg", "v_corr", "v_cov",
                 "M_ok_ratio"]}
    return dict(
        bias_reg = float(d["bias_reg"][REF_IDX]),
        bias_dr  = float(d["bias_dr"][REF_IDX]),
        bse      = float(d["bse_grid"][REF_IDX]),
        cov      = float(d["coverage_ref"]),
        ser      = float(d.get("ser_ref", float("nan"))),
        ess      = float(d.get("ess_min_mean", float("nan"))),
        w_p99    = float(d["w_p99_grid_mean"][REF_IDX]) if "w_p99_grid_mean" in d else float("nan"),
        h_wres   = float(d.get("h_weighted_residual_mean", float("nan")))
                   if d.get("h_weighted_residual_mean") is not None else float("nan"),
        q_wres   = float(d.get("q_weighted_residual_mean", float("nan")))
                   if d.get("q_weighted_residual_mean") is not None else float("nan"),
        v_total  = float(d["V_total"][REF_IDX]) if "V_total" in d else float("nan"),
        v_reg    = float(d["V_reg"][REF_IDX])   if "V_reg"   in d else float("nan"),
        v_corr   = float(d["V_corr"][REF_IDX])  if "V_corr"  in d else float("nan"),
        v_cov    = float(d["V_cov"][REF_IDX])   if "V_cov"   in d else float("nan"),
        M_ok_ratio = float(d.get("M_ok_ratio", float("nan"))),
    )


def _build_report(results: dict, n: int, M: int) -> list[str]:
    lines: list[str] = []
    lines.append("# Phase 11C+ -- Final Kallus closure validation")
    lines.append(f"# Generated : {datetime.date.today()}")
    lines.append("")

    # Section 1
    lines.append("## 1. Objective")
    lines.append("")
    lines.append(
        "Phase 11C surprise: B2 (KPV_super + Q_Kallus_match nf=400) gave "
        "|b|/SE=0.25 vs A=0.34 at n=2000, M=100 (-26%). This phase validates "
        "B2 vs A on disjoint seeds (seed_base=22_000_000) at M=200 to assess "
        "whether the gain is robust or another small-M artefact (cf. Phase 11 "
        "Q_Kallus winner that did not survive larger M)."
    )
    lines.append("")
    lines.append(f"DGP2 MichaelisMenten, snr={SNR}, ref a={A_GRID[REF_IDX]:.1f}, "
                 f"n={n}, M={M}, n_folds={N_FOLDS}, seed_base={_SEED_BASE}.")
    lines.append("")
    lines.append("**Variance fix**: Phase 11C report had V_total/V_reg/V_corr=NaN "
                 "due to a case mismatch (`v_total` lookup vs `V_total` key in "
                 "`_decompose`). Fixed in this script's report builder.")
    lines.append("")

    # Section 2
    lines.append("## 2. Configurations")
    lines.append("")
    lines.append("| ID | h-bridge | q-bridge |")
    lines.append("|----|----------|----------|")
    lines.append("| A  | KPV_super (lam=3e-5, ell=3.5) | Q_KPV (lambda_Q=1e-3) |")
    lines.append("| B2 | KPV_super | Q_Kallus_match (nf=400, gamma_Q=1e-3, lambda_stab=1.0, gamma_critic=1e-3) |")
    if "D2" in results:
        lines.append("| D2 | Kallus_h F_rich (gH=1e-4, ls=0.1, nf=400) | Q_Kallus_match |")
    lines.append("")

    # Section 3 -- main table
    lines.append(f"## 3. Main validation @ n={n}, M={M}")
    lines.append("")
    lines.append("| ID | M_ok | bias_REG | bias_DR | |b|/SE | cov | SER | ESS_min | w_p99 | q_wres | h_wres |")
    lines.append("|----|------|----------|---------|--------|-----|-----|---------|-------|--------|--------|")
    for cid in ["A", "B2", "D2"]:
        if cid not in results:
            continue
        decomp, _, _ = results[cid]
        r = _row_full(decomp)
        h_str = f"{r['h_wres']:.3f}" if np.isfinite(r['h_wres']) else "---"
        q_str = f"{r['q_wres']:.3f}" if np.isfinite(r['q_wres']) else "---"
        lines.append(
            f"| {cid} | {r['M_ok_ratio']*M:.0f}/{M} | "
            f"{r['bias_reg']:+.4f} | {r['bias_dr']:+.4f} | "
            f"{r['bse']:.2f} | {r['cov']:.3f} | {r['ser']:.2f} | "
            f"{r['ess']:.0f} | {r['w_p99']:.2f} | {q_str} | {h_str} |"
        )
    lines.append("")

    # Section 4 -- difference + decision
    lines.append("## 4. Difference table (B2 vs A)")
    lines.append("")
    A = results.get("A")
    B2 = results.get("B2")
    decision_b2 = "(insufficient data)"
    if A is not None and B2 is not None and A[0] is not None and B2[0] is not None:
        rA, rB2 = _row_full(A[0]), _row_full(B2[0])
        d_bse  = (rA["bse"] - rB2["bse"]) / max(rA["bse"], 1e-9) * 100
        d_cov  = rB2["cov"] - rA["cov"]
        d_bias = rB2["bias_dr"] - rA["bias_dr"]
        d_ess  = rB2["ess"] - rA["ess"]
        d_w99  = rB2["w_p99"] - rA["w_p99"]
        lines.append(f"- Δ |b|/SE   : {-d_bse:+.0f}% (B2 better if positive in absolute terms)")
        lines.append(f"  - A: {rA['bse']:.2f}, B2: {rB2['bse']:.2f}")
        lines.append(f"- Δ coverage : {d_cov:+.3f} (B2 - A)")
        lines.append(f"  - A: {rA['cov']:.3f}, B2: {rB2['cov']:.3f}")
        lines.append(f"- Δ bias_DR  : {d_bias:+.4f} (B2 - A)")
        lines.append(f"- Δ ESS_min  : {d_ess:+.0f} (B2 - A)")
        lines.append(f"- Δ w_p99    : {d_w99:+.2f} (B2 - A)")
        lines.append("")
        # Decision
        if d_bse >= 15.0 and d_cov >= -0.03 and d_w99 < 5.0 and rB2["ess"] >= 50:
            decision_b2 = "**B2 confirmed** (>=15% on |b|/SE, cov drop <= 0.03, weights/ESS stable)"
        elif d_bse >= 10.0 and d_cov >= -0.03:
            decision_b2 = "**B2 partial** (gain 10-15% on |b|/SE, cov stable)"
        elif d_bse >= 0 and (d_w99 > 5.0 or rB2["ess"] < 50):
            decision_b2 = "**B2 unstable** (gain present but weights/ESS degrade materially)"
        elif d_bse < 10.0:
            decision_b2 = "**B2 not robust** (improvement < 10%; consistent with small-M artefact)"
        else:
            decision_b2 = "(see Δ values)"
        lines.append(f"**Decision** : {decision_b2}")
        lines.append("")

    # Section 5 -- variance
    lines.append("## 5. Variance decomposition (V_cov = (V_total - V_reg - V_corr) / 2)")
    lines.append("")
    lines.append("| ID | V_total | V_reg | V_corr | V_cov |")
    lines.append("|----|---------|-------|--------|-------|")
    for cid in ["A", "B2", "D2"]:
        if cid not in results:
            continue
        d = results[cid][0]
        if d is None:
            lines.append(f"| {cid} | --- | --- | --- | --- |")
            continue
        r = _row_full(d)
        lines.append(
            f"| {cid} | {r['v_total']:.4f} | {r['v_reg']:.4f} | "
            f"{r['v_corr']:.4f} | {r['v_cov']:.4f} |"
        )
    lines.append("")

    # Section 6 -- final verdict
    lines.append("## 6. Final Kallus closure verdict")
    lines.append("")
    has_D2 = "D2" in results and results["D2"][0] is not None
    if has_D2:
        rD2 = _row_full(results["D2"][0])
        d_bse_D2 = (rA["bse"] - rD2["bse"]) / max(rA["bse"], 1e-9) * 100
        if "B2 confirmed" in decision_b2:
            if d_bse_D2 < 0:
                # Case 1
                verdict = ("**Case 1** -- B2 confirmed, D2 still worse than A. "
                           "Kallus full does not dominate, but Kallus-q with rich features "
                           "is a useful refinement over KPV-q when paired with KPV-super.")
            else:
                # Case 3 unexpected
                verdict = ("**Case 3** -- D2 unexpectedly beats A and B2. "
                           "Re-open full Kallus with careful validation; do not yet move to Bennett-lite.")
        else:
            # Case 2
            verdict = ("**Case 2** -- KPV-super + KPV-q remains the robust final benchmark. "
                       "Kallus is diagnostic only.")
    else:
        if "B2 confirmed" in decision_b2:
            verdict = ("**Case 1 (no D2)** -- B2 confirmed without further D2 validation. "
                       "Kallus full not retested but Phase 11C showed it does not dominate. "
                       "Kallus-q with rich features is a useful refinement when h is well tuned.")
        else:
            verdict = ("**Case 2** -- KPV-super + KPV-q remains the robust final benchmark. "
                       "Kallus is diagnostic only.")
    lines.append(verdict)
    lines.append("")

    # Section 7 -- essay paragraph
    lines.append("## 7. S6/S7 essay paragraph")
    lines.append("")
    if "B2 confirmed" in decision_b2:
        lines.append(
            "> *\"Across q-only (Phase 11), h-only (Phase 11B), full h+q (Phase 11C), "
            "and a final disjoint-seed validation at M=200 (Phase 11C+), the carefully "
            "tuned KPV DRKernel pipeline (KPV_super + KPV-q) was the robust benchmark "
            "for J(pi) on DGP2 Michaelis-Menten. The most operationally interesting "
            "Kallus contribution was a modest but reproducible refinement on the q-side: "
            "replacing KPV-q with a finite-feature Kallus-stabilised q (n_features=400, "
            "matched in richness with the outcome bridge), while keeping the KPV-super "
            "h-bridge, reduces |bias|/SE by approximately 15-25% at n=2000 with "
            "comparable coverage. Pairing the same Kallus-q with a Kallus minimax "
            "h-bridge does not stack additively: the over-correction of the h-side "
            "minimax under estimated q dilutes the gain. We therefore retain the "
            "KPV+KPV pipeline as the default, document Kallus-q (matched features) as "
            "an optional refinement, and motivate functional-first Bennett-lite "
            "inference (Phase 12) as the natural way to bypass the destructive h/q "
            "interaction at finite n.\"*"
        )
    else:
        lines.append(
            "> *\"Across q-only (Phase 11), h-only (Phase 11B), full h+q (Phase 11C), "
            "and a final disjoint-seed validation at M=200 (Phase 11C+), Kallus did not "
            "robustly improve over the carefully tuned KPV DRKernel pipeline on DGP2. "
            "The Phase 11C surprise (Q_Kallus matched-features at n=2000, M=100, "
            "-26% on |bias|/SE) did not survive disjoint-seed validation at M=200, "
            "consistent with the Phase 11 q-only winner that also evaporated under "
            "stricter MC. Kallus's main contribution in this project is therefore "
            "diagnostic: weighted minimax residuals reveal weak directions but do not "
            "translate into a superior finite-sample estimator of J(pi). This motivates "
            "moving to Bennett-lite functional-first inference (Phase 12).\"*"
        )
    lines.append("")
    return lines


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 11C+ Kallus closure validation")
    parser.add_argument("--mode", choices=("quick", "main"), default="main")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--M", type=int, default=200)
    parser.add_argument("--configs", nargs="+", default=["A", "B2"],
                        help="Configs to run, e.g. A B2 D2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 11C+ -- Final Kallus closure validation")
    print(f"mode={args.mode} n={args.n} M={args.M} configs={args.configs}")
    print("=" * 72)

    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)

    n, M = args.n, args.M
    if args.mode == "quick":
        n, M = 300, 3

    results: dict = {}
    t_start = time.perf_counter()
    for cid in args.configs:
        if cid not in CELLS:
            print(f"[WARN] unknown config {cid!r}, skipping")
            continue
        elapsed_min = (time.perf_counter() - t_start) / 60.0
        print(f"\n--- {cid} (n={n}, M={M})  [elapsed {elapsed_min:.1f}min]  ---")
        decomp, wall, data = _run_one_cell(dgp, cid, n, M, force=args.force)
        _print_brief(cid, n, M, decomp, wall)
        results[cid] = (decomp, wall, data)

    total_min = (time.perf_counter() - t_start) / 60.0
    print(f"\n{'=' * 72}\nTotal wall-clock: {total_min:.1f} min\n{'=' * 72}")

    # Build report (only if main mode and we have at least A+B2)
    if args.mode == "main" and ("A" in results and "B2" in results):
        print("Building closure report...")
        lines = _build_report(results, n=n, M=M)
        _SUMM_DIR.mkdir(parents=True, exist_ok=True)
        _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Report -> {_REPORT_PATH}")
    else:
        print("Skipping report (quick mode or missing A/B2).")


if __name__ == "__main__":
    main()
