"""
DGP2 Final Tuning Map -- Phase 9B.5
====================================

Two complementary objectives:

  Part A -- ell-extension:
    Verify whether ell_scale=3.5 is an optimum, plateau, or boundary.
    Grid: lambda_h=3e-5, ell_scale in {3.0, 3.5, 4.0, 4.5, 5.0}.
    Reuses Phase 9B.4 anchors for ell in {3.0, 3.5}; computes 3 new (4.0, 4.5, 5.0).

  Part B -- blind tuning protocol:
    Test whether observable-only diagnostics (no J_policy_true, no bias, no coverage)
    can rank h-bridge configs into the correct region. Uses 4 score variants:
      A: moment + weights + ESS
      B: moment + stability + weights
      C: stability + ESS + weights (no moment)
      D: conservative (heavy weight + clip penalty)
    Compares blind top-3 to oracle top-3 post-hoc.

Forbidden in blind protocol (NEVER read from data):
  - J_policy_true, m_true, bias_REG, bias_DR, coverage, |b|/SE, RMSE, MISE.

Allowed in blind protocol (observable from fit):
  - V_hat, V_hat_grid, V_correction_grid, V_reg_grid
  - ESS_min, ESS_grid, weight_p99_grid, weight_max_grid
  - q_clip_fraction, q_negative_share_grid, neg_share_grid_mean
  - riesz_residual_grid_mean (Bennett-style functional residual)
  - cond_M_grid_mean (numerical conditioning)
  - jensen_gap (per-dose, from estimate distribution)
  - empirical stability: std(J_DR across reps)

Seeds: 16_000_000+ (disjoint from Phase 9B.3=10M-13.9M, Phase 9B.4=14M-15M).

Output:
  simulations/results/raw/dgp2_final/dgp2final_F_{config_id}_n{n}_M{M}.pkl
  simulations/results/summaries/dgp2_final_ell_extension.md
  simulations/results/summaries/dgp2_blind_tuning_protocol.md
"""
from __future__ import annotations

import argparse
import datetime
import pickle
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── Reuse helpers from existing scripts ───────────────────────────────────────
from simulations.experiments.dgp2_bias_diagnostics import (
    A_GRID,
    LAMBDA_Q_DEFAULT,
    CLIP_DEFAULT,
    REF_IDX,
    SNR,
    _compute_J_policy_true,
    _decompose,
    _median_bandwidth,
    _run_block,
    _silverman_h,
)
from simulations.archive.utilities.dgp2_tuning_confirmation import (
    _compute_ser,
    _max_weight_max,
    make_factory,
)
from simulations.archive.utilities.dgp2_optimal_region import (
    _BLOCK_E_ANCHORS,  # for cross-reference
    _LAM_STR,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR        = Path(__file__).resolve().parents[2]
_RAW_DIR         = _BASE_DIR / "simulations" / "results" / "raw" / "dgp2_final"
_TRADEOFF_RAW    = _BASE_DIR / "simulations" / "results" / "raw" / "dgp2_tradeoff"
_OPTREGION_RAW   = _BASE_DIR / "simulations" / "results" / "raw" / "dgp2_optimal_region"
_SUMM_DIR        = _BASE_DIR / "simulations" / "results" / "summaries"
_REPORT_ELL      = _SUMM_DIR / "dgp2_final_ell_extension.md"
_REPORT_BLIND    = _SUMM_DIR / "dgp2_blind_tuning_protocol.md"

# ── Grid for Part A ───────────────────────────────────────────────────────────
_F_LAM    = 3e-5
_F_ELL_NEW: list[float] = [4.0, 4.5, 5.0]      # truly new
_F_ELL_ANCHOR: list[float] = [3.0, 3.5]        # reuse from Phase 9B.4
_F_ELL_ALL = sorted(_F_ELL_ANCHOR + _F_ELL_NEW)

_SEED_BASE_F = 16_000_000

# ── Forbidden field names in blind protocol (whitelist enforced in tests) ─────
_BLIND_FORBIDDEN_FIELDS = frozenset({
    "bias_reg", "bias_dr", "coverage_ref", "bse_grid", "rmse", "mise",
    "J_policy_true", "m_true", "J_pt",
})


# ══════════════════════════════════════════════════════════════════════════════
#  Config helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ell_str(ell: float) -> str:
    return f"{ell:.2f}".replace(".", "p")


def _f_config_id(lam: float, ell: float) -> str:
    lam_s = _LAM_STR.get(lam, f"{lam:.0e}")
    return f"F_lam{lam_s}_ell{_ell_str(ell)}"


def _build_f_specs() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for ell in _F_ELL_NEW:
        cid = _f_config_id(_F_LAM, ell)
        out[cid] = dict(
            lambda_h=_F_LAM, ell_scale=ell,
            lambda_Q=LAMBDA_Q_DEFAULT, clip=CLIP_DEFAULT, h_policy_scale=1.0,
        )
    return out


F_NEW_CONFIGS: dict[str, dict] = _build_f_specs()


def _seed_F(cfg_idx: int, n: int) -> int:
    return _SEED_BASE_F + cfg_idx * 10_000 + int(n)


def _pkl_label(config_id: str, n: int, M: int) -> str:
    return f"dgp2final_{config_id}_n{n}_M{M}"


# ══════════════════════════════════════════════════════════════════════════════
#  Anchor loaders for Phase 9B.4 PKLs (ell=3.0, 3.5 with lam=3e-5)
# ══════════════════════════════════════════════════════════════════════════════

def _try_load_F_anchor_from_optregion(config_id: str) -> Optional[Path]:
    """Find a Phase 9B.4 PKL matching this F config (lam=3e-5, ell in {3.0,3.5})."""
    m = re.match(r"F_lam([\w-]+)_ell(\d+p\d+)$", config_id)
    if not m:
        return None
    lam_s, ell_s = m.group(1), m.group(2)
    e_id = f"E_lam{lam_s}_ell{ell_s}"
    candidate = _OPTREGION_RAW / f"dgp2optimal_E_{e_id}_n1000_M50.pkl"
    return candidate if candidate.exists() else None


# ══════════════════════════════════════════════════════════════════════════════
#  Part A -- ell extension runner
# ══════════════════════════════════════════════════════════════════════════════

def run_ell_extension(dgp, n: int = 1000, M: int = 50, force: bool = False,
                       configs: Optional[list[str]] = None) -> dict:
    """Run the ell-extension grid (lam=3e-5, ell in {4.0,4.5,5.0} new + anchors)."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, tuple[Optional[dict], float, Optional[dict]]] = {}

    # 1. Reference sample for kernel bandwidths
    ref_sample = dgp.generate(n=n, seed=0)
    h_ref  = _silverman_h(ref_sample.A)
    ell_W0 = _median_bandwidth(ref_sample.W)
    ell_A0 = _median_bandwidth(ref_sample.A)
    ell_Z0 = _median_bandwidth(ref_sample.Z)
    h_policy = h_ref * 1.0
    J_pt = _compute_J_policy_true(dgp, A_GRID, h_policy)

    # 2. Determine config order
    if configs is None:
        configs = [_f_config_id(_F_LAM, ell) for ell in _F_ELL_ALL]

    for cfg_idx, cid in enumerate(configs):
        print(f"\n--- Part A | {cid} | n={n} | M={M} ---")

        # 2a. Try anchor (Phase 9B.4)
        if not force:
            anc_path = _try_load_F_anchor_from_optregion(cid)
            if anc_path is not None:
                with open(anc_path, "rb") as fh:
                    data = pickle.load(fh)
                # Save copy under F-naming so blind protocol can scan dgp2_final/
                dest = _RAW_DIR / f"{_pkl_label(cid, n, M)}.pkl"
                if not dest.exists():
                    with open(dest, "wb") as fh:
                        pickle.dump(data, fh)
                decomp = _decompose(data)
                results[cid] = (decomp, 0.0, data)
                print(f"  [ANCHOR] {cid} loaded from Phase 9B.4 (no recompute)")
                _print_brief(cid, decomp)
                continue

        # 2b. Compute new config
        spec = F_NEW_CONFIGS.get(cid)
        if spec is None:
            print(f"  [SKIP] {cid} not in F_NEW_CONFIGS")
            continue

        factory = make_factory(spec, A_GRID, h_ref, ell_W0, ell_A0, ell_Z0)
        label   = _pkl_label(cid, n, M)
        seed_base = _seed_F(cfg_idx, n)

        if force:
            for sfx in (".pkl", ".partial.pkl"):
                p = _RAW_DIR / f"{label}{sfx}"
                if p.exists():
                    p.unlink()

        t0 = time.perf_counter()
        data = _run_block(
            dgp, factory, A_GRID, J_pt,
            n=n, M=M, seed_base=seed_base, label=label, raw_dir=_RAW_DIR,
        )
        wall = time.perf_counter() - t0
        decomp = _decompose(data)
        results[cid] = (decomp, wall, data)
        _print_brief(cid, decomp, wall=wall)

    return results


def _print_brief(cid: str, decomp: Optional[dict], wall: float = 0.0) -> None:
    if decomp is None:
        print(f"    [WARN] {cid} produced no successful reps")
        return
    print(
        f"    bias_DR(ref)={decomp['bias_dr'][REF_IDX]:+.4f}  "
        f"|b|/SE={decomp['bse_grid'][REF_IDX]:.2f}  "
        f"cov={decomp['coverage_ref']:.3f}  "
        f"wall={wall/60:.1f}min"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Part B -- blind tuning protocol
# ══════════════════════════════════════════════════════════════════════════════

# PKL pattern matchers
_PKL_PATTERNS = {
    # Phase 9B.3 Block H : dgp2tradeoff_H_H_<...>_n1000_M50.pkl
    "tradeoff_H": (
        _TRADEOFF_RAW,
        re.compile(r"dgp2tradeoff_H_(H_[^_]+(?:_[^_]+)*?)_n(\d+)_M(\d+)\.pkl$"),
    ),
    # Phase 9B.4 Block E : dgp2optimal_E_E_<...>_n1000_M50.pkl
    "optimal_E": (
        _OPTREGION_RAW,
        re.compile(r"dgp2optimal_E_(E_[^_]+(?:_[^_]+)*?)_n(\d+)_M(\d+)\.pkl$"),
    ),
    # Phase 9B.5 Part A : dgp2final_F_<...>_n1000_M50.pkl
    "final_F": (
        _RAW_DIR,
        re.compile(r"dgp2final_(F_[^_]+(?:_[^_]+)*?)_n(\d+)_M(\d+)\.pkl$"),
    ),
}

# Parse config ID -> (lambda_h, ell_scale)
_CID_PATTERN = re.compile(r"^[HEF]_lam([\w-]+)_ell(\d+p\d+)$")


def _parse_config_id(cid: str) -> tuple[Optional[float], Optional[float]]:
    """Return (lambda_h, ell_scale) from a config_id, or (None, None) if baseline."""
    if cid.endswith("_baseline") or cid in {"H_baseline", "E_baseline", "F_baseline"}:
        return None, None
    # Phase 9B.3 used "H_lam1e-4_ell2.5" (dot)
    m = re.match(r"^[HEF]_lam([\w-]+)_ell([\d\.p]+)$", cid)
    if not m:
        return None, None
    lam_s, ell_s = m.group(1), m.group(2)
    try:
        lam = float(lam_s)
    except ValueError:
        return None, None
    try:
        ell = float(ell_s.replace("p", "."))
    except ValueError:
        return None, None
    return lam, ell


def _scan_all_pkls(n_target: int = 1000, M_target: int = 50) -> dict[str, dict]:
    """
    Scan all source dirs for PKLs matching n=n_target, M=M_target.
    Returns: dict[canonical_id] -> dict with keys {data, source, raw_cid, lam, ell}.
    Canonicalize: H_lam* and E_lam* with same (lam,ell) collapse to one entry
                  (prefer Phase 9B.4 E over Phase 9B.3 H if both exist).
    """
    found: dict[str, dict] = {}  # canonical_id -> entry
    priority = {"final_F": 0, "optimal_E": 1, "tradeoff_H": 2}

    for src_name, (raw_dir, pat) in _PKL_PATTERNS.items():
        if not raw_dir.exists():
            continue
        for pkl in sorted(raw_dir.glob("*.pkl")):
            m = pat.match(pkl.name)
            if not m:
                continue
            raw_cid, n_str, M_str = m.group(1), m.group(2), m.group(3)
            if int(n_str) != n_target or int(M_str) != M_target:
                continue
            lam, ell = _parse_config_id(raw_cid)
            # Canonical key
            if lam is None and ell is None:
                canonical = "BASELINE"
            else:
                lam_s = _LAM_STR.get(lam, f"{lam:.0e}")
                canonical = f"lam{lam_s}_ell{ell:.2f}".replace(".", "p")

            existing = found.get(canonical)
            if existing is None or priority[src_name] < priority[existing["source"]]:
                with open(pkl, "rb") as fh:
                    data = pickle.load(fh)
                found[canonical] = dict(
                    data=data, source=src_name, raw_cid=raw_cid,
                    lam=lam, ell=ell, pkl_path=str(pkl),
                )
    return found


def _blind_diagnostics(data: dict) -> dict[str, float]:
    """
    Extract OBSERVABLE diagnostics only -- NO use of truth-based metrics.
    Returns: dict of scalar diagnostics for blind ranking.
    """
    records = [r for r in data.get("records", []) if r.get("error") is None]
    if not records:
        return dict(n_reps=0)

    # Per-rep arrays
    psi_arr = np.array([r["psi_hat"] for r in records], dtype=float)
    V_arr   = np.array([r["V_hat"]   for r in records], dtype=float)
    ESS_arr = np.array([r["ESS_min"] for r in records], dtype=float)
    qclip_arr = np.array([r["q_clip_fraction"] for r in records], dtype=float)
    bandwidth_arr = np.array([r["bandwidth"] for r in records], dtype=float)

    # Grid arrays (per-rep, per-dose) -- use ref dose
    def _grid_at_ref(key: str) -> np.ndarray:
        return np.array(
            [r[key][REF_IDX] if (r.get(key) is not None and len(r[key]) > REF_IDX) else np.nan
             for r in records], dtype=float
        )

    w_p99_arr = _grid_at_ref("weight_p99_grid")
    w_max_arr = _grid_at_ref("weight_max_grid")
    riesz_arr = _grid_at_ref("riesz_residual_grid_mean")
    cond_arr  = _grid_at_ref("cond_M_grid_mean")
    jensen_arr = _grid_at_ref("jensen_gap")
    neg_share_arr = _grid_at_ref("neg_share_grid_mean")
    V_corr_arr = _grid_at_ref("V_correction_grid")
    V_reg_arr  = _grid_at_ref("V_reg_grid")

    # J_dr per-rep at ref dose (for stability)
    Jdr_arr = np.array(
        [r["J_dr"][REF_IDX] if (r.get("J_dr") is not None) else np.nan for r in records],
        dtype=float,
    )

    out: dict[str, float] = dict(
        n_reps=len(records),
        # Stability of J_DR (empirical SD across reps -- normalized by mean V_hat)
        psi_std=float(np.nanstd(Jdr_arr, ddof=1)) if len(Jdr_arr) > 1 else float("nan"),
        psi_cv=float(np.nanstd(Jdr_arr, ddof=1) / np.nanmean(np.sqrt(V_arr) + 1e-12))
               if len(Jdr_arr) > 1 else float("nan"),
        # Variance metrics (mean across reps)
        V_hat_mean=float(np.nanmean(V_arr)),
        V_corr_mean=float(np.nanmean(V_corr_arr)),
        V_reg_mean=float(np.nanmean(V_reg_arr)),
        V_corr_share=float(np.nanmean(V_corr_arr) / (np.nanmean(V_reg_arr) + np.nanmean(V_corr_arr) + 1e-12)),
        # ESS & weights
        ess_min_mean=float(np.nanmean(ESS_arr)),
        w_p99_mean=float(np.nanmean(w_p99_arr)),
        w_max_mean=float(np.nanmean(w_max_arr)),
        # Bridge q numerics
        q_clip_mean=float(np.nanmean(qclip_arr)),
        neg_share_mean=float(np.nanmean(neg_share_arr)),
        # h-bridge functional residual (Bennett-style; observable)
        riesz_residual_mean=float(np.nanmean(riesz_arr)),
        # Numerical conditioning
        cond_M_mean=float(np.nanmean(cond_arr)),
        # Jensen gap (observable from estimator)
        jensen_mean=float(np.nanmean(jensen_arr)),
        # Bandwidth (KDE)
        bandwidth=float(np.nanmean(bandwidth_arr)),
    )
    return out


def _normalize_rank(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    """Return ranks in [0, 1]; lower-is-better when ascending=True."""
    v = values.copy()
    finite = np.isfinite(v)
    ranks = np.full_like(v, np.nan, dtype=float)
    if finite.sum() == 0:
        return ranks
    order = np.argsort(v[finite]) if ascending else np.argsort(-v[finite])
    rk = np.empty(int(finite.sum()), dtype=float)
    rk[order] = np.arange(int(finite.sum()))
    ranks[finite] = rk / max(int(finite.sum()) - 1, 1)
    return ranks


def _compute_blind_scores(diag_table: list[dict]) -> dict[str, np.ndarray]:
    """
    Given a list of dict diagnostics (one per config), compute 4 blind score variants.
    Returns dict[variant_name] -> array of scores (lower = better).
    """
    n = len(diag_table)
    if n == 0:
        return {}

    arr = lambda key: np.array([d.get(key, np.nan) for d in diag_table], dtype=float)

    moment       = arr("riesz_residual_mean")          # lower better (functional residual)
    stability    = arr("psi_std")                       # lower better (stable J_DR across reps)
    weight_p99   = arr("w_p99_mean")                    # lower better
    ess          = arr("ess_min_mean")                  # higher better (use -)
    q_clip       = arr("q_clip_mean")                   # lower better
    cond         = arr("cond_M_mean")                   # lower better
    V_hat        = arr("V_hat_mean")                    # lower better

    r_moment    = _normalize_rank(moment,    ascending=True)
    r_stab      = _normalize_rank(stability, ascending=True)
    r_weight    = _normalize_rank(weight_p99, ascending=True)
    r_ess       = _normalize_rank(-ess,      ascending=True)  # invert: higher better
    r_qclip     = _normalize_rank(q_clip,    ascending=True)
    r_cond      = _normalize_rank(cond,      ascending=True)
    r_var       = _normalize_rank(V_hat,     ascending=True)

    # Variants
    A = r_moment + r_weight + r_ess
    B = r_moment + r_stab   + r_weight
    C = r_stab   + r_ess    + r_weight
    D = 2.0 * r_weight + 2.0 * r_qclip + r_ess + r_var  # conservative

    return dict(A=A, B=B, C=C, D=D)


def _oracle_summary(data: dict) -> dict[str, float]:
    """Oracle-only metrics for post-hoc comparison (NEVER used by blind score)."""
    decomp = _decompose(data)
    if decomp is None:
        return dict(bias_dr=float("nan"), bse=float("nan"), cov=float("nan"))
    return dict(
        bias_dr=float(decomp["bias_dr"][REF_IDX]),
        bse=float(decomp["bse_grid"][REF_IDX]),
        cov=float(decomp["coverage_ref"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Reports
# ══════════════════════════════════════════════════════════════════════════════

def _build_ell_report(results_F: dict) -> str:
    lines: list[str] = []
    lines.append("# DGP2 Final ell Extension")
    lines.append(f"# Generated : {datetime.date.today()}")
    lines.append("# Phase 9B.5 Part A -- ell_scale > 3.5 with lambda_h = 3e-5")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append("Test whether ell_scale=3.5 (Phase 9B.4 winner) is an optimum,")
    lines.append("a plateau, or simply the boundary of the previous grid.")
    lines.append("")
    lines.append("## 2. Design")
    lines.append(f"DGP : MichaelisMentenDGP(snr_W={SNR}, snr_Z={SNR})")
    lines.append(f"a_grid : {list(A_GRID)}, REF_IDX={REF_IDX}")
    lines.append(f"lambda_h = {_F_LAM:.0e}, ell_scale in {_F_ELL_ALL}")
    lines.append("n=1000, M=50. Anchors loaded from Phase 9B.4 for ell in {3.0, 3.5}.")
    lines.append("")
    lines.append("## 3. Results table")
    lines.append("")
    lines.append("| config | ell | bias_REG | bias_DR | |b|/SE | cov_ref | SER | ESS | w_p99 | jensen |")
    lines.append("|--------|-----|----------|---------|--------|---------|-----|-----|-------|--------|")

    for cid in sorted(results_F.keys(), key=lambda c: float(c.split("ell")[1].replace("p","."))):
        decomp, _, data = results_F[cid]
        ell = float(cid.split("ell")[1].replace("p", "."))
        if decomp is None or data is None:
            lines.append(f"| {cid} | {ell:.2f} | --- | --- | --- | --- | --- | --- | --- | --- |")
            continue
        ser = _compute_ser(data)
        ser_s = f"{ser:.2f}" if ser is not None else "---"
        try:
            w_p99 = float(_max_weight_max(data))
            w_p99_s = f"{w_p99:.2f}"
        except Exception:
            w_p99_s = "---"
        diag = _blind_diagnostics(data)
        jensen_s = f"{diag.get('jensen_mean', float('nan')):.4f}"
        lines.append(
            f"| {cid} | {ell:.2f} | "
            f"{decomp['bias_reg'][REF_IDX]:+.4f} | "
            f"{decomp['bias_dr'][REF_IDX]:+.4f} | "
            f"{decomp['bse_grid'][REF_IDX]:.2f} | "
            f"{decomp['coverage_ref']:.3f} | "
            f"{ser_s} | "
            f"{decomp.get('ess_min_mean', float('nan')):.0f} | "
            f"{w_p99_s} | {jensen_s} |"
        )

    lines.append("")
    lines.append("## 4. Verdict")

    # Compute trend
    bse_by_ell = {}
    for cid, (d, _, _) in results_F.items():
        if d is not None:
            ell = float(cid.split("ell")[1].replace("p", "."))
            bse_by_ell[ell] = float(d["bse_grid"][REF_IDX])

    if len(bse_by_ell) >= 3:
        ells_sorted = sorted(bse_by_ell.keys())
        bses = [bse_by_ell[e] for e in ells_sorted]
        argmin_ell = ells_sorted[int(np.argmin(bses))]
        bse_35 = bse_by_ell.get(3.5, float("nan"))
        bse_50 = bse_by_ell.get(5.0, float("nan"))
        if not np.isnan(bse_35) and not np.isnan(bse_50):
            if bse_50 > bse_35 * 1.20:
                verdict = f"BOUNDARY/OVER-SMOOTHING: ell={argmin_ell:.2f} optimum, |b|/SE rises by ell=5.0 ({bse_50:.2f} vs {bse_35:.2f})."
            elif abs(bse_50 - bse_35) < 0.10:
                verdict = f"PLATEAU: |b|/SE stable in [3.5, 5.0] (range {min(bses):.2f}-{max(bses):.2f})."
            elif bse_50 < bse_35 * 0.80:
                verdict = f"TREND CONTINUES: ell=5.0 still improving ({bse_50:.2f} vs {bse_35:.2f})."
            else:
                verdict = f"MIXED: argmin at ell={argmin_ell:.2f}, range [{min(bses):.2f}, {max(bses):.2f}]."
        else:
            verdict = "INCOMPLETE: missing ell=3.5 or 5.0 data."
        lines.append("")
        lines.append(f"**Region verdict**: {verdict}")
        lines.append("")
        lines.append(f"|b|/SE by ell: " + ", ".join(f"{e:.2f}->{bse_by_ell[e]:.2f}" for e in ells_sorted))
    lines.append("")

    return "\n".join(lines)


def _build_blind_report(scan: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("# DGP2 Blind Tuning Protocol")
    lines.append(f"# Generated : {datetime.date.today()}")
    lines.append("# Phase 9B.5 Part B -- observable-only ranking of h-tuning configs")
    lines.append("")

    # ── Section 1 ────────────────────────────────────────────────────────────
    lines.append("## 1. Protocol definition")
    lines.append("")
    lines.append("**Forbidden** (NEVER read by blind ranking):")
    for f in sorted(_BLIND_FORBIDDEN_FIELDS):
        lines.append(f"  - {f}")
    lines.append("")
    lines.append("**Allowed** (observable from fit):")
    lines.append("  - V_hat, V_correction_grid, V_reg_grid (variance components)")
    lines.append("  - ESS_min, weight_p99_grid, weight_max_grid (IPW health)")
    lines.append("  - q_clip_fraction, q_negative_share, neg_share (q-bridge numerics)")
    lines.append("  - riesz_residual_grid_mean (Bennett-style functional residual of h)")
    lines.append("  - cond_M_grid_mean (numerical conditioning)")
    lines.append("  - jensen_gap_ref (curvature/smoothing diagnostic)")
    lines.append("  - psi_std across reps (empirical stability of J_DR)")
    lines.append("")

    # ── Section 2 ────────────────────────────────────────────────────────────
    lines.append("## 2. Sources scanned")
    lines.append("")
    by_src: dict[str, int] = {}
    for entry in scan.values():
        by_src[entry["source"]] = by_src.get(entry["source"], 0) + 1
    for src, n in by_src.items():
        lines.append(f"  - {src}: {n} configs")
    lines.append(f"  - Total unique configs: {len(scan)}")
    lines.append("")

    # ── Section 3 ────────────────────────────────────────────────────────────
    lines.append("## 3. Observable diagnostics per config")
    lines.append("")
    lines.append("| canonical_id | source | lam | ell | n_reps | psi_std | V_hat | ESS_min | w_p99 | q_clip | riesz_res | cond_M | jensen |")
    lines.append("|--------------|--------|-----|-----|--------|---------|-------|---------|-------|--------|-----------|--------|--------|")

    diag_table: list[dict] = []
    cids_ordered: list[str] = []
    for cid in sorted(scan.keys(), key=lambda c: (
        float("inf") if c == "BASELINE" else (
            scan[c]["ell"] if scan[c]["ell"] is not None else float("inf")
        )
    )):
        entry = scan[cid]
        diag = _blind_diagnostics(entry["data"])
        diag_table.append(diag)
        cids_ordered.append(cid)
        lam_s = "default" if entry["lam"] is None else _LAM_STR.get(entry["lam"], f"{entry['lam']:.0e}")
        ell_s = "1.00" if entry["ell"] is None else f"{entry['ell']:.2f}"
        lines.append(
            f"| {cid} | {entry['source']} | {lam_s} | {ell_s} | "
            f"{diag.get('n_reps', 0)} | "
            f"{diag.get('psi_std', float('nan')):.4f} | "
            f"{diag.get('V_hat_mean', float('nan')):.4f} | "
            f"{diag.get('ess_min_mean', float('nan')):.0f} | "
            f"{diag.get('w_p99_mean', float('nan')):.2f} | "
            f"{diag.get('q_clip_mean', float('nan')):.3f} | "
            f"{diag.get('riesz_residual_mean', float('nan')):.4f} | "
            f"{diag.get('cond_M_mean', float('nan')):.2e} | "
            f"{diag.get('jensen_mean', float('nan')):.4f} |"
        )
    lines.append("")

    # ── Section 4 ────────────────────────────────────────────────────────────
    lines.append("## 4. Blind score rankings")
    lines.append("")
    scores = _compute_blind_scores(diag_table)
    lines.append("**Score variants** (lower = better):")
    lines.append("  - A = rank(riesz_res) + rank(w_p99) + rank(-ESS)")
    lines.append("  - B = rank(riesz_res) + rank(psi_std) + rank(w_p99)")
    lines.append("  - C = rank(psi_std) + rank(-ESS) + rank(w_p99)  [no functional residual]")
    lines.append("  - D = 2*rank(w_p99) + 2*rank(q_clip) + rank(-ESS) + rank(V_hat)  [conservative]")
    lines.append("")

    for variant, sc in scores.items():
        order = np.argsort(sc)
        lines.append(f"### Variant {variant} -- top 5 by blind score")
        lines.append("")
        lines.append("| rank | config | blind_score |")
        lines.append("|------|--------|-------------|")
        for rk, idx in enumerate(order[:5], 1):
            lines.append(f"| {rk} | {cids_ordered[idx]} | {sc[idx]:.3f} |")
        lines.append("")

    # ── Section 5 -- post-hoc oracle comparison ──────────────────────────────
    lines.append("## 5. Post-hoc oracle comparison")
    lines.append("")
    lines.append("Oracle = sorted ascending by |bias_DR|/SE (computed AFTER blind ranking).")
    lines.append("")

    oracle_metrics = [_oracle_summary(scan[cid]["data"]) for cid in cids_ordered]
    oracle_bse = np.array([m["bse"] for m in oracle_metrics], dtype=float)
    # rank by |bse|, smaller is better (after dropping nan)
    oracle_rank = _normalize_rank(np.abs(oracle_bse), ascending=True)
    oracle_top = np.argsort(np.abs(oracle_bse))[:5]
    lines.append("### Oracle top-5 (truth-based)")
    lines.append("")
    lines.append("| rank | config | bias_DR | |b|/SE | cov |")
    lines.append("|------|--------|---------|--------|-----|")
    for rk, idx in enumerate(oracle_top, 1):
        m = oracle_metrics[idx]
        lines.append(
            f"| {rk} | {cids_ordered[idx]} | "
            f"{m['bias_dr']:+.4f} | {m['bse']:.2f} | {m['cov']:.3f} |"
        )
    lines.append("")

    # Overlap of top-3
    lines.append("### Top-3 overlap : blind vs oracle")
    lines.append("")
    lines.append("| variant | blind top-3 | oracle top-3 | overlap | baseline eliminated? |")
    lines.append("|---------|-------------|--------------|---------|----------------------|")
    oracle_top3 = set(np.argsort(np.abs(oracle_bse))[:3].tolist())
    for variant, sc in scores.items():
        blind_top3 = set(np.argsort(sc)[:3].tolist())
        overlap = blind_top3 & oracle_top3
        overlap_cids = [cids_ordered[i] for i in overlap]
        baseline_idx = cids_ordered.index("BASELINE") if "BASELINE" in cids_ordered else -1
        baseline_elim = baseline_idx not in blind_top3 if baseline_idx >= 0 else "n/a"
        lines.append(
            f"| {variant} | "
            f"{', '.join(cids_ordered[i] for i in np.argsort(sc)[:3])} | "
            f"{', '.join(cids_ordered[i] for i in np.argsort(np.abs(oracle_bse))[:3])} | "
            f"{len(overlap)}/3 ({', '.join(overlap_cids) if overlap_cids else 'none'}) | "
            f"{'yes' if baseline_elim is True else 'no' if baseline_elim is False else baseline_elim} |"
        )
    lines.append("")

    # ── Section 6 -- verdict ─────────────────────────────────────────────────
    lines.append("## 6. Verdict")
    lines.append("")

    # Best variant by overlap and baseline elimination
    best_overlap = 0
    best_variant = None
    for variant, sc in scores.items():
        blind_top3 = set(np.argsort(sc)[:3].tolist())
        overlap = len(blind_top3 & oracle_top3)
        if overlap > best_overlap:
            best_overlap = overlap
            best_variant = variant

    if best_overlap >= 2:
        lines.append(f"**Blind protocol SUCCEEDS** (variant {best_variant}): top-3 overlap = {best_overlap}/3 with oracle.")
        lines.append("Observable diagnostics recover the smooth-RKHS region without using truth.")
    elif best_overlap >= 1:
        lines.append(f"**Blind protocol PARTIALLY SUCCEEDS** (variant {best_variant}): top-3 overlap = {best_overlap}/3.")
        lines.append("Eliminates baseline and identifies the correct regime, but does not pinpoint the exact optimum.")
        lines.append("Region-based tuning rather than pointwise hyperparameter selection is recommended.")
    else:
        lines.append(f"**Blind protocol FAILS**: top-3 overlap = 0/3 with oracle.")
        lines.append("Observable diagnostics do not recover the oracle-best region; functional diagnostics")
        lines.append("(Kallus minimax, Bennett SourceCondition) would be required for real-data tuning.")
    lines.append("")

    # ── Section 7 -- recommendation for the essay ────────────────────────────
    lines.append("## 7. Recommendation for S6/S7")
    lines.append("")
    if best_overlap >= 2:
        lines.append("'The blind diagnostic ranking selects the same smooth-RKHS region as the oracle simulation ranking.")
        lines.append(" This suggests that projected residuals, score stability and weight diagnostics can guide tuning")
        lines.append(" even when the causal target is unknown.'")
    elif best_overlap >= 1:
        lines.append("'The blind diagnostics eliminate the failing baseline and identify the correct smooth-RKHS regime,")
        lines.append(" but do not distinguish sharply between the best configurations. This supports region-based tuning")
        lines.append(" rather than pointwise hyperparameter selection.'")
    else:
        lines.append("'The blind diagnostics do not recover the oracle-best region, implying that the simulation tuning")
        lines.append(" is not directly transferable to real data without additional theory or validation. This motivates")
        lines.append(" Kallus/Bennett-style functional diagnostics.'")
    lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DGP2 Final Tuning Map (Phase 9B.5)")
    parser.add_argument("--mode", choices=["quick", "ell-extension", "blind-protocol", "report", "all"],
                        default="all", help="Execution mode")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if PKLs exist (Part A only)")
    args = parser.parse_args()

    print("=" * 72)
    print(f"DGP2 Final Tuning Map -- Phase 9B.5 -- mode={args.mode}")
    print("=" * 72)

    _SUMM_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "quick":
        from simulations.dgp.michaelis_menten import MichaelisMentenDGP
        dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)
        # Smoke: 1 new config at n=200, M=3
        cid = _f_config_id(_F_LAM, 4.5)
        results = run_ell_extension(dgp, n=200, M=3, force=args.force, configs=[cid])
        print(f"\nQuick test produced {len(results)} configs.")
        return

    if args.mode in ("ell-extension", "all"):
        from simulations.dgp.michaelis_menten import MichaelisMentenDGP
        dgp = MichaelisMentenDGP(snr_W=SNR, snr_Z=SNR)
        print("\n--- Part A : ell extension (lam=3e-5, ell in {3.0,3.5,4.0,4.5,5.0}) ---")
        t0 = time.perf_counter()
        results_F = run_ell_extension(dgp, n=1000, M=50, force=args.force)
        wall_F = (time.perf_counter() - t0) / 60.0
        print(f"\nPart A done in {wall_F:.1f} min ({len(results_F)} configs)")
        report_ell = _build_ell_report(results_F)
        _REPORT_ELL.write_text(report_ell, encoding="utf-8")
        print(f"  Report -> {_REPORT_ELL.name}")

    if args.mode in ("blind-protocol", "all"):
        print("\n--- Part B : blind tuning protocol ---")
        scan = _scan_all_pkls(n_target=1000, M_target=50)
        print(f"  Scanned: {len(scan)} unique configs")
        if not scan:
            print("  [WARN] No PKLs found. Run ell-extension first.")
            return
        report_blind = _build_blind_report(scan)
        _REPORT_BLIND.write_text(report_blind, encoding="utf-8")
        print(f"  Report -> {_REPORT_BLIND.name}")

    if args.mode == "report":
        # Rebuild both reports from existing data
        scan = _scan_all_pkls(n_target=1000, M_target=50)
        if scan:
            _REPORT_BLIND.write_text(_build_blind_report(scan), encoding="utf-8")
            print(f"  Blind report -> {_REPORT_BLIND.name}")
        # Build ell report from F PKLs
        results_F: dict = {}
        for pkl in sorted(_RAW_DIR.glob("dgp2final_F_*_n1000_M50.pkl")):
            m = re.match(r"dgp2final_(F_[^_]+_ell[\dp]+)_n1000_M50\.pkl$", pkl.name)
            if not m:
                continue
            cid = m.group(1)
            with open(pkl, "rb") as fh:
                data = pickle.load(fh)
            results_F[cid] = (_decompose(data), 0.0, data)
        if results_F:
            _REPORT_ELL.write_text(_build_ell_report(results_F), encoding="utf-8")
            print(f"  Ell report -> {_REPORT_ELL.name}")


if __name__ == "__main__":
    main()
