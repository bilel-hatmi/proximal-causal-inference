"""
Phase 11C -- Full Kallus final controlled comparison.

Closes the Kallus question by testing the FULL h+q Kallus combination against
the KPV_super benchmark, plus q-only and h-only checks under matched seeds.

Trajectory:
- Phase 11   q-only  : KPV_h + Q_Kallus -> Cas B/D (no q-side gain)
- Phase 11B  h-only  : H_Kallus + Q_KPV -> Cas 2 nuanced (asymptotic only)
- Phase 11C  h+q     : H_Kallus + Q_Kallus  vs  KPV_super + Q_KPV  -> THIS

Configurations (8 total, all run with seed_base=21_000_000)
-----------------------------------------------------------
  A  : KPV_super (lam=3e-5, ell=3.5)            + Q_KPV          -- benchmark
  B1 : KPV_super                                  + Q_Kallus_central (nf=150)
  B2 : KPV_super                                  + Q_Kallus_match   (nf=400)
  C  : Kallus F_rich (gH=1e-4, ls=0.1, nf=400)  + Q_KPV          -- h-only repl
  D1 : Kallus F_rich                              + Q_Kallus_central -- Full Kallus
  D2 : Kallus F_rich                              + Q_Kallus_match   -- Full Kallus
  E  : KPV_super                                  + Q_oracle      -- interaction baseline
  F  : Kallus F_rich                              + Q_oracle      -- interaction H test

Priority ordering (truncate from end if compute too long):
  1. n=2000 : A, C, D1, D2
  2. n=2000 : E, F
  3. n=2000 : B1, B2
  4. n=1000 : A, C, D1, D2
  5. n=1000 : E, F, B1, B2

Output:
  simulations/results/raw/kallus_full_final/<label>.pkl
  simulations/results/summaries/phase11C_full_kallus_FINAL.md
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
_RAW_DIR     = _BASE_DIR / "simulations" / "results" / "raw" / "kallus_full_final"
_SUMM_DIR    = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_PATH = _SUMM_DIR / "phase11C_full_kallus_FINAL.md"

_SEED_BASE = 21_000_000

# ── Hyperparameter configs (frozen) ──────────────────────────────────────────

# h-bridge: KPV_super (Phase 9B.4 winner)
KPV_SUPER = dict(lambda_h=3e-5, ell_scale=3.5)

# h-bridge: Kallus F_rich (Phase 11B winner)
KALLUS_F_RICH = dict(
    gamma_H=1e-4, lambda_stab_h=0.1, gamma_critic_h=1e-4,
    n_features_h=400, n_features_critic=400,
    ell_scale=3.5,
)

# q-bridge: Kallus central (Phase 11 stable)
KALLUS_Q_CENTRAL = dict(
    n_features_q=150, n_features_h=150,
    gamma_Q=1e-3, lambda_stab=1.0, gamma_critic=1e-3,
    feature_seed=20260427,
    compute_diagnostics="light",
)

# q-bridge: Kallus matched (richness aligned with H_Kallus F_rich)
KALLUS_Q_MATCH = dict(
    n_features_q=400, n_features_h=400,
    gamma_Q=1e-3, lambda_stab=1.0, gamma_critic=1e-3,
    feature_seed=20260427,
    compute_diagnostics="light",
)

# Cell registry (id -> (h_kind, q_kind, qkind_extra))
# h_kind in {"kpv_super", "kallus_rich"}
# q_kind in {"kpv", "kallus_central", "kallus_match", "oracle"}
CELLS: dict[str, tuple[str, str]] = {
    "A":  ("kpv_super",   "kpv"),
    "B1": ("kpv_super",   "kallus_central"),
    "B2": ("kpv_super",   "kallus_match"),
    "C":  ("kallus_rich", "kpv"),
    "D1": ("kallus_rich", "kallus_central"),
    "D2": ("kallus_rich", "kallus_match"),
    "E":  ("kpv_super",   "oracle"),
    "F":  ("kallus_rich", "oracle"),
}

CELL_ORDER_PRIORITY: list[tuple[str, int]] = [
    # Tier 1: decisive head-to-head at n=2000
    ("A", 2000), ("C", 2000), ("D1", 2000), ("D2", 2000),
    # Tier 2: oracle diagnostics at n=2000
    ("E", 2000), ("F", 2000),
    # Tier 3: q-only checks at n=2000
    ("B1", 2000), ("B2", 2000),
    # Tier 4: finite-sample sanity at n=1000
    ("A", 1000), ("C", 1000), ("D1", 1000), ("D2", 1000),
    # Tier 5: rest at n=1000
    ("E", 1000), ("F", 1000), ("B1", 1000), ("B2", 1000),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Factory builder
# ══════════════════════════════════════════════════════════════════════════════

def _kpv_super_bridge_kwargs(ell_W0, ell_A0, ell_Z0) -> dict:
    return dict(
        lambda_1=KPV_SUPER["lambda_h"], lambda_2=KPV_SUPER["lambda_h"],
        ell_W=ell_W0 * KPV_SUPER["ell_scale"],
        ell_A=ell_A0 * KPV_SUPER["ell_scale"],
        ell_Z=ell_Z0 * KPV_SUPER["ell_scale"],
    )


def make_factory(cell_id: str, h_KDE, ell_W0, ell_A0, ell_Z0, dgp) -> Callable:
    from simulations.methods.dr_kernel import DRKernel
    from simulations.methods.kpv_bridge import KPVPolicyBridgeQ
    from simulations.methods.kallus_minimax import (
        KallusStabilizedPolicyQ, KallusMinimaxBridgeH,
    )
    from simulations.methods._oracle_bridges import OracleBridgeQ

    h_kind, q_kind = CELLS[cell_id]

    # --- h-bridge config ---
    bridge_kwargs: dict = {}
    h_factory = None
    if h_kind == "kpv_super":
        bridge_kwargs = _kpv_super_bridge_kwargs(ell_W0, ell_A0, ell_Z0)
    elif h_kind == "kallus_rich":
        def h_factory():
            return KallusMinimaxBridgeH(
                mode="kallus_stabilized", **KALLUS_F_RICH,
                compute_diagnostics="light",
            )
    else:
        raise ValueError(f"Unknown h_kind {h_kind!r}")

    # --- q-bridge config ---
    if q_kind == "kpv":
        q_model = KPVPolicyBridgeQ(a_grid=A_GRID, h_KDE=h_KDE,
                                    lambda_Q=LAMBDA_Q_DEFAULT, clip=None)
        cross_fit_q = True
    elif q_kind == "kallus_central":
        q_model = KallusStabilizedPolicyQ(
            a_grid=A_GRID, h_KDE=h_KDE, mode="kallus_stabilized", **KALLUS_Q_CENTRAL,
        )
        cross_fit_q = True
    elif q_kind == "kallus_match":
        q_model = KallusStabilizedPolicyQ(
            a_grid=A_GRID, h_KDE=h_KDE, mode="kallus_stabilized", **KALLUS_Q_MATCH,
        )
        cross_fit_q = True
    elif q_kind == "oracle":
        q_model = OracleBridgeQ(dgp)
        cross_fit_q = False
    else:
        raise ValueError(f"Unknown q_kind {q_kind!r}")

    def factory():
        return DRKernel(
            a_grid=A_GRID, q_model=q_model, bridge_kwargs=bridge_kwargs,
            bandwidth=h_KDE, n_folds=N_FOLDS, random_state=RANDOM_STATE,
            ref_dose_index=REF_IDX, cross_fit_q=cross_fit_q,
            h_bridge_factory=h_factory,
        )
    return factory


# ══════════════════════════════════════════════════════════════════════════════
#  Per-cell runner
# ══════════════════════════════════════════════════════════════════════════════

def _label(cell_id: str, n: int, M: int) -> str:
    return f"kallusfull_{cell_id}_n{n}_M{M}"


def _seed(cell_idx: int, n: int) -> int:
    return _SEED_BASE + cell_idx * 10_000 + int(n)


def _run_one_cell(
    dgp, cell_id: str, n: int, M: int, cell_idx: int, force: bool = False,
) -> tuple[Optional[dict], float, Optional[dict]]:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    ref_sample = dgp.generate(n=n, seed=0)
    h_KDE  = _silverman_h(ref_sample.A)
    ell_W0 = _median_bandwidth(ref_sample.W)
    ell_A0 = _median_bandwidth(ref_sample.A)
    ell_Z0 = _median_bandwidth(ref_sample.Z)
    J_pt   = _compute_J_policy_true(dgp, A_GRID, h_KDE)
    factory = make_factory(cell_id, h_KDE, ell_W0, ell_A0, ell_Z0, dgp)

    label = _label(cell_id, n, M)
    seed_b = _seed(cell_idx, n)

    if force:
        for sfx in (".pkl", ".partial.pkl"):
            p = _RAW_DIR / f"{label}{sfx}"
            if p.exists():
                p.unlink()

    t0 = time.perf_counter()
    data = _run_block(dgp, factory, A_GRID, J_pt, n=n, M=M,
                      seed_base=seed_b, label=label, raw_dir=_RAW_DIR)
    wall = time.perf_counter() - t0
    decomp = _decompose(data)
    # Augment decomp with mean h-extras + q-Kallus extras (across reps)
    if decomp is not None and data is not None:
        recs = [r for r in data["records"] if r.get("error") is None]
        if recs:
            r0 = recs[0]
            if r0.get("h_kallus_present"):
                vals = [r["h_weighted_residual_mean"] for r in recs
                        if np.isfinite(r.get("h_weighted_residual_mean", np.nan))]
                if vals:
                    decomp["h_weighted_residual_mean"] = float(np.mean(vals))
                vals2 = [r["h_raw_residual_mean"] for r in recs
                         if np.isfinite(r.get("h_raw_residual_mean", np.nan))]
                if vals2:
                    decomp["h_raw_residual_mean"] = float(np.mean(vals2))
                decomp["h_method"] = r0.get("h_method", "?")
            if r0.get("kallus_present"):
                wres_arr = []
                for r in recs:
                    arr = r.get("kallus_weighted_residual_grid")
                    if arr is not None:
                        wres_arr.append(float(np.nanmean(arr)))
                if wres_arr:
                    decomp["q_weighted_residual_mean"] = float(np.mean(wres_arr))
                decomp["q_kallus_present"] = True
    return decomp, wall, data


def _print_brief(cell_id: str, n: int, decomp: Optional[dict], wall: float) -> None:
    if decomp is not None:
        h_wres = decomp.get("h_weighted_residual_mean", None)
        q_wres = decomp.get("q_weighted_residual_mean", None)
        h_str = f" h_wres={float(h_wres):.3f}" if h_wres is not None and np.isfinite(h_wres) else ""
        q_str = f" q_wres={float(q_wres):.3f}" if q_wres is not None and np.isfinite(q_wres) else ""
        print(
            f"    [{cell_id} n={n}]  "
            f"bias_DR={decomp['bias_dr'][REF_IDX]:+.4f}  "
            f"|b|/SE={decomp['bse_grid'][REF_IDX]:.2f}  "
            f"cov={decomp['coverage_ref']:.3f}  "
            f"ESS={decomp.get('ess_min_mean', float('nan')):.0f}{h_str}{q_str}  "
            f"wall={wall/60:.1f}min"
        )
    else:
        print(f"    [{cell_id} n={n}]  no data (wall={wall/60:.1f}min)")


# ══════════════════════════════════════════════════════════════════════════════
#  Report builder
# ══════════════════════════════════════════════════════════════════════════════

def _row(decomp: Optional[dict]) -> dict:
    if decomp is None:
        return dict(bias_reg=np.nan, bias_dr=np.nan, bse=np.nan,
                    cov=np.nan, ess=np.nan, w_p99=np.nan,
                    h_wres=np.nan, q_wres=np.nan, ser=np.nan)
    return dict(
        bias_reg = float(decomp["bias_reg"][REF_IDX]),
        bias_dr  = float(decomp["bias_dr"][REF_IDX]),
        bse      = float(decomp["bse_grid"][REF_IDX]),
        cov      = float(decomp["coverage_ref"]),
        ess      = float(decomp.get("ess_min_mean", np.nan)),
        w_p99    = float(decomp.get("w_p99_mean", np.nan)),
        h_wres   = float(decomp.get("h_weighted_residual_mean", np.nan))
                   if decomp.get("h_weighted_residual_mean") is not None else np.nan,
        q_wres   = float(decomp.get("q_weighted_residual_mean", np.nan))
                   if decomp.get("q_weighted_residual_mean") is not None else np.nan,
        ser      = (float(decomp["ratio_emp_pred"]) if "ratio_emp_pred" in decomp else np.nan),
    )


def _classify(results: dict, n: int) -> str:
    """Decide Case A/B/C/D from results at given n."""
    A = results.get(("A", n))
    if A is None or A[0] is None:
        return "(insufficient data)"
    A_row = _row(A[0])

    best_kallus_id, best_kallus_bse = None, float("inf")
    for cid in ("D1", "D2"):
        cell = results.get((cid, n))
        if cell is None or cell[0] is None:
            continue
        bse = float(cell[0]["bse_grid"][REF_IDX])
        if bse < best_kallus_bse:
            best_kallus_bse, best_kallus_id = bse, cid
    if best_kallus_id is None:
        return "(no full-Kallus results)"

    K = _row(results[(best_kallus_id, n)][0])

    bse_drop = (A_row["bse"] - K["bse"]) / max(A_row["bse"], 1e-9)
    cov_drop = A_row["cov"] - K["cov"]
    ess_ratio = K["ess"] / max(A_row["ess"], 1e-9)
    w_ratio = K["w_p99"] / max(A_row["w_p99"], 1e-9) if np.isfinite(A_row["w_p99"]) and np.isfinite(K["w_p99"]) else np.nan

    instab = (
        K["ess"] < 50
        or (np.isfinite(w_ratio) and w_ratio > 2.0)
    )
    if bse_drop >= 0.15 and cov_drop <= 0.03 and not instab:
        return f"Case A (Full Kallus GO @ n={n}, winner {best_kallus_id})"
    if instab and bse_drop > 0:
        return f"Case C (instability @ n={n}, winner {best_kallus_id} but ESS/w_p99 degrade)"
    # check residuals
    if np.isfinite(K["q_wres"]) and (np.isfinite(K["h_wres"])):
        if K["q_wres"] < 0.10 and K["h_wres"] < 0.10 and abs(bse_drop) < 0.10:
            return f"Case B (residuals low but |b|/SE flat @ n={n})"
    if abs(bse_drop) < 0.10:
        return f"Case D (no gain @ n={n}; KPV_super remains benchmark)"
    if bse_drop < 0:
        return f"Case D (Kallus loses @ n={n}; KPV_super remains benchmark)"
    return f"Case A* (partial gain @ n={n}, {best_kallus_id})"


def _build_report(results: dict) -> list[str]:
    lines: list[str] = []
    lines.append("# Phase 11C -- Full Kallus final controlled comparison")
    lines.append(f"# Generated : {datetime.date.today()}")
    lines.append("")

    lines.append("## 1. Objective")
    lines.append("")
    lines.append(
        "Phase 11 q-only (Cas B/D) and Phase 11B h-only (Cas 2 nuanced) tested "
        "Kallus on a single nuisance. Phase 11C tests the FULL combination "
        "h_Kallus + q_Kallus against the KPV_super benchmark, with matched seeds, "
        "matched Monte-Carlo size, and the same DRKernel pipeline."
    )
    lines.append("")
    lines.append(f"DGP2 MichaelisMenten, snr_W=snr_Z={SNR}, ref dose a={A_GRID[REF_IDX]:.1f}, "
                 f"n_folds={N_FOLDS}, seed_base={_SEED_BASE}.")
    lines.append("")

    # ── Section 2 : configurations
    lines.append("## 2. Configurations")
    lines.append("")
    lines.append("| ID | h-bridge | q-bridge | Role |")
    lines.append("|----|----------|----------|------|")
    descr = {
        "A":  ("KPV_super (lam=3e-5, ell=3.5)", "Q_KPV", "**Benchmark**"),
        "B1": ("KPV_super",                       "Q_Kallus_central (nf=150)", "q-only check"),
        "B2": ("KPV_super",                       "Q_Kallus_match (nf=400)",   "q-only check (matched)"),
        "C":  ("Kallus F_rich (gH=1e-4, ls=0.1, nf=400)", "Q_KPV", "h-only check"),
        "D1": ("Kallus F_rich",                   "Q_Kallus_central",          "**Full Kallus (central q)**"),
        "D2": ("Kallus F_rich",                   "Q_Kallus_match",            "**Full Kallus (matched q)**"),
        "E":  ("KPV_super",                       "Q_oracle",                  "Interaction baseline"),
        "F":  ("Kallus F_rich",                   "Q_oracle",                  "Interaction H test"),
    }
    for cid in ["A", "B1", "B2", "C", "D1", "D2", "E", "F"]:
        h, q, role = descr[cid]
        lines.append(f"| {cid} | {h} | {q} | {role} |")
    lines.append("")

    # ── Section 3 : main results
    lines.append("## 3. Main results")
    lines.append("")
    for n in [1000, 2000]:
        lines.append(f"### n={n}")
        lines.append("")
        lines.append("| ID | bias_REG | bias_DR | |b|/SE | cov | ESS_min | h_wres | q_wres |")
        lines.append("|----|----------|---------|--------|-----|---------|--------|--------|")
        for cid in ["A", "B1", "B2", "C", "D1", "D2", "E", "F"]:
            entry = results.get((cid, n))
            if entry is None or entry[0] is None:
                lines.append(f"| {cid} | --- | --- | --- | --- | --- | --- | --- |")
                continue
            r = _row(entry[0])
            h_str = f"{r['h_wres']:.3f}" if np.isfinite(r['h_wres']) else "---"
            q_str = f"{r['q_wres']:.3f}" if np.isfinite(r['q_wres']) else "---"
            lines.append(
                f"| {cid} | {r['bias_reg']:+.4f} | {r['bias_dr']:+.4f} | "
                f"{r['bse']:.2f} | {r['cov']:.3f} | {r['ess']:.0f} | "
                f"{h_str} | {q_str} |"
            )
        lines.append("")

    # ── Section 4 : interaction diagnostic
    lines.append("## 4. h/q interaction diagnostic")
    lines.append("")
    lines.append("Compare delta |b|/SE for isolating contributions:")
    lines.append("")
    lines.append("| n | A (KPV+KPV) | B-best (KPV+Q_K) | C (H_K+KPV) | D-best (Full K) |")
    lines.append("|---|-------------|------------------|-------------|-----------------|")
    for n in [1000, 2000]:
        A = results.get(("A", n))
        B_best, _ = _best_of(results, n, ["B1", "B2"])
        C = results.get(("C", n))
        D_best, _ = _best_of(results, n, ["D1", "D2"])
        cells = []
        for x in (A, B_best, C, D_best):
            if x is None or x[0] is None:
                cells.append("---")
            else:
                cells.append(f"{float(x[0]['bse_grid'][REF_IDX]):.2f}")
        lines.append(f"| {n} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines.append("")

    # ── Section 5 : oracle-q diagnostic
    lines.append("## 5. Oracle-q diagnostic (E vs F)")
    lines.append("")
    lines.append("| n | E (KPV+Q_oracle) | F (Kallus_h+Q_oracle) | Δ |b|/SE |")
    lines.append("|---|------------------|----------------------|--------|")
    for n in [1000, 2000]:
        E = results.get(("E", n))
        F = results.get(("F", n))
        if E is None or F is None or E[0] is None or F[0] is None:
            lines.append(f"| {n} | --- | --- | --- |")
            continue
        e_bse = float(E[0]["bse_grid"][REF_IDX])
        f_bse = float(F[0]["bse_grid"][REF_IDX])
        delta = (e_bse - f_bse) / max(e_bse, 1e-9) * 100
        lines.append(f"| {n} | {e_bse:.2f} | {f_bse:.2f} | {delta:+.0f}% |")
    lines.append("")

    # ── Section 6 : variance + weights
    lines.append("## 6. Variance and weights")
    lines.append("")
    lines.append("| n | ID | V_total | V_reg | V_corr | weight_p99 |")
    lines.append("|---|----|---------|-------|--------|------------|")
    for n in [1000, 2000]:
        for cid in ["A", "C", "D1", "D2"]:
            entry = results.get((cid, n))
            if entry is None or entry[0] is None:
                continue
            d = entry[0]
            v_t = float(d["v_total"][REF_IDX]) if "v_total" in d else float("nan")
            v_r = float(d["v_reg"][REF_IDX]) if "v_reg" in d else float("nan")
            v_c = float(d["v_corr"][REF_IDX]) if "v_corr" in d else float("nan")
            w99 = float(d.get("w_p99_mean", np.nan))
            w99_s = f"{w99:.2f}" if np.isfinite(w99) else "---"
            lines.append(f"| {n} | {cid} | {v_t:.4f} | {v_r:.4f} | {v_c:.4f} | {w99_s} |")
    lines.append("")

    # ── Section 7 : residual diagnostics
    lines.append("## 7. Residual diagnostics")
    lines.append("")
    lines.append("| n | ID | h_weighted_residual | q_weighted_residual | h_raw |")
    lines.append("|---|----|--------------------|--------------------|-------|")
    for n in [1000, 2000]:
        for cid in ["A", "B1", "B2", "C", "D1", "D2", "E", "F"]:
            entry = results.get((cid, n))
            if entry is None or entry[0] is None:
                continue
            d = entry[0]
            h_w = d.get("h_weighted_residual_mean", None)
            q_w = d.get("q_weighted_residual_mean", None)
            h_r = d.get("h_raw_residual_mean", None)
            h_w_s = f"{float(h_w):.3f}" if h_w is not None and np.isfinite(float(h_w)) else "---"
            q_w_s = f"{float(q_w):.3f}" if q_w is not None and np.isfinite(float(q_w)) else "---"
            h_r_s = f"{float(h_r):.3f}" if h_r is not None and np.isfinite(float(h_r)) else "---"
            lines.append(f"| {n} | {cid} | {h_w_s} | {q_w_s} | {h_r_s} |")
    lines.append("")

    # ── Section 8 : decision
    lines.append("## 8. Decision")
    lines.append("")
    for n in [2000, 1000]:
        lines.append(f"- **n={n}** : {_classify(results, n)}")
    lines.append("")

    # ── Section 9 : essay-ready paragraph
    lines.append("## 9. S6/S7 essay-ready paragraph")
    lines.append("")
    verdict_n2000 = _classify(results, 2000)
    if "Case A" in verdict_n2000 and "GO" in verdict_n2000:
        lines.append(
            "> *\"The full Kallus h+q configuration improves over the KPV-super "
            "benchmark, suggesting that adversarial stabilization is useful only "
            "when both nuisances are estimated in a compatible minimax geometry "
            "-- neither nuisance alone exhibited the gain (Phase 11 q-only and "
            "Phase 11B h-only).\"*"
        )
    else:
        lines.append(
            "> *\"After q-only (Phase 11), h-only (Phase 11B), and full h+q "
            "Kallus tests (Phase 11C), Kallus did not robustly dominate the "
            "carefully tuned KPV DRKernel pipeline. Its main contribution in "
            "this project is diagnostic: weighted minimax residuals reveal "
            "weak directions, but they did not translate into a superior "
            "finite-sample estimator of J(pi). This motivates moving to "
            "Bennett-lite functional-first inference (Phase 12).\"*"
        )
    lines.append("")
    return lines


def _best_of(results: dict, n: int, ids: list[str]) -> tuple[Optional[tuple], Optional[str]]:
    best, best_id, best_bse = None, None, float("inf")
    for cid in ids:
        cell = results.get((cid, n))
        if cell is None or cell[0] is None:
            continue
        bse = float(cell[0]["bse_grid"][REF_IDX])
        if bse < best_bse:
            best_bse, best, best_id = bse, cell, cid
    return best, best_id


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 11C Full Kallus final")
    parser.add_argument("--mode", choices=("quick", "main"), default="main")
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-min", type=float, default=180,
                        help="Soft compute budget in minutes; if exceeded, stop further cells.")
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 11C -- Full Kallus final controlled comparison")
    print(f"mode={args.mode}  M={args.M}  budget={args.max_min}min")
    print("=" * 72)

    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)

    if args.mode == "quick":
        # 1 cell per type at small n/M for smoke
        run_list = [("A", 200, 3), ("D1", 200, 3), ("E", 200, 3)]
    else:
        # Priority ordering
        run_list = [(cid, n, args.M) for (cid, n) in CELL_ORDER_PRIORITY]

    # Stable cell index per (cell_id) for seed disjointness
    cell_idx_map = {cid: idx for idx, cid in enumerate(["A","B1","B2","C","D1","D2","E","F"])}

    results: dict = {}
    t_start = time.perf_counter()
    for (cid, n, M) in run_list:
        elapsed_min = (time.perf_counter() - t_start) / 60.0
        if elapsed_min > args.max_min:
            print(f"\n[BUDGET] elapsed={elapsed_min:.1f}min > max={args.max_min}min, stopping further cells.")
            break
        print(f"\n--- {cid}  (n={n}, M={M})  [elapsed {elapsed_min:.1f}min]  ---")
        decomp, wall, data = _run_one_cell(
            dgp, cid, n, M, cell_idx=cell_idx_map[cid], force=args.force,
        )
        _print_brief(cid, n, decomp, wall)
        results[(cid, n)] = (decomp, wall, data)

    total_min = (time.perf_counter() - t_start) / 60.0
    print(f"\n{'=' * 72}\nTotal wall-clock: {total_min:.1f} min\n{'=' * 72}")

    # Build report
    print("Building report...")
    lines = _build_report(results)
    _SUMM_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report -> {_REPORT_PATH}")


if __name__ == "__main__":
    main()
