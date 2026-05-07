"""
Phase 19 — Win A: widened-grid refinement following Phase 18 blind protocol.

Triggered when Stage 4 winner has axes at grid boundary. Re-applies the same
3-sub-stage blind procedure (axis sweep → refinement → validation) on a
widened grid for the boundary axis. The new winner is declared ONLY if its
blind score strictly improves on the Stage 4 winner's score.

Methodology (to document in S7):
  "After Stage 4 we observed that ell_h was at the lower boundary of the grid
  (1.5). To verify this was not a grid-design artifact, we ran a second
  round of blind tuning on a widened ell_h grid {0.5, 0.75, 1.0, 1.25, 1.5},
  following the same 3-sub-stage protocol (axis sweep M=30 → refinement M=50
  ->validation M=100). The new winner was accepted only if its blind score
  strictly improved on the original."

Usage:
    python -u -m simulations.experiments.real_data_winA_widening \\
        --dataset nlsy79_father --axis ell_h --grid 0.5 0.75 1.0 1.25 1.5
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from simulations.experiments.s8_applications.s8_3_blind_tuning import (
    DATASETS, _RESIDUALIZE_DEFAULT, RealDataDGP,
    _make_bennett_estimator_real, _BASE_DIR,
)
from simulations.experiments.s7_simulations.s7_8_blind_tuning_bennett import (
    bennett_blind_score_v2, extract_diagnostics,
)
from simulations.experiments.dgp2_bias_diagnostics import (
    _silverman_h, _median_bandwidth, _run_block,
)

_RAW_BASE = _BASE_DIR / "simulations" / "results" / "raw" / "s8"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"

ACCEPTANCE_EPSILON = 0.01    # blind score must improve by > 0.01 to accept


def load_winner_config(dataset: str) -> dict:
    """Read winner config from realdata_{dataset}_bennett_FINAL.md JSON block."""
    final_path = _SUMM_DIR / f"realdata_{dataset}_bennett_FINAL.md"
    if not final_path.exists():
        raise FileNotFoundError(f"FINAL.md missing: {final_path}")
    txt = final_path.read_text(encoding="utf-8")
    import re
    m = re.search(r"```json\s*(\{.*?\})\s*```", txt, flags=re.DOTALL)
    if not m:
        raise RuntimeError("No json winner block in FINAL.md")
    return json.loads(m.group(1))


def get_existing_winner_score(dataset: str) -> float:
    """Read Stage 4 winner score from PKL."""
    pkl = _RAW_BASE / dataset / "bennett" / "stage4" / "S4_c0_validation.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"No Stage 4 cfg0 PKL at {pkl}")
    with open(pkl, "rb") as fh:
        data = pickle.load(fh)
    diags = extract_diagnostics(data, ref_idx=int(data["meta"].get("ref_idx", 2)))
    cfg = load_winner_config(dataset)
    diags["lambda_h"] = float(cfg.get("lambda_h", 1e-5))
    diags["m_h"] = float(cfg.get("m_h", 2000))
    return bennett_blind_score_v2(diags, n=data["meta"]["n"])


def run_one(cfg: dict, dgp: RealDataDGP, bw: dict, *,
            M: int, seed_base: int, raw_dir: Path, label: str) -> dict:
    """Single config × M reps. Returns {label, config, diags, score, slope, slope_se}."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    J_pt = np.full(len(dgp.a_grid), np.nan)
    factory = lambda: _make_bennett_estimator_real(
        cfg, dgp, bw["h_KDE"], bw["ell_W"], bw["ell_A"], bw["ell_Z"],
        a_grid=dgp.a_grid, ref_idx=dgp.ref_idx,
    )
    data = _run_block(
        dgp, factory, np.asarray(dgp.a_grid), J_pt,
        n=dgp.N, M=M, seed_base=seed_base, label=label, raw_dir=raw_dir,
    )
    diags = extract_diagnostics(data, ref_idx=dgp.ref_idx)
    diags["lambda_h"] = float(cfg.get("lambda_h", 1e-5))
    diags["m_h"] = float(cfg.get("m_h", 2000))
    score = bennett_blind_score_v2(diags, n=dgp.N)

    # Compute endpoint slope
    records = [r for r in data["records"] if r.get("error") is None]
    J_dr = np.array([r["J_dr"] for r in records])
    a_grid = np.asarray(dgp.a_grid)
    span = float(a_grid[-1] - a_grid[0])
    slope = (J_dr[:, -1] - J_dr[:, 0]) / span
    return {
        "label": label, "config": cfg, "diags": diags,
        "score": float(score),
        "slope_mean": float(slope.mean()),
        "slope_se": float(slope.std(ddof=1)),
        "slope_ci": (float(np.quantile(slope, 0.025)),
                     float(np.quantile(slope, 0.975))),
        "M_ok": len(records),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    p.add_argument("--axis", required=True,
                   choices=["ell_h", "ell_scale_r", "lambda_h", "lambda_r", "gamma_critic"])
    p.add_argument("--grid", nargs="+", type=float, required=True,
                   help="Widened grid values for the axis")
    p.add_argument("--rho_coarsen", type=float, default=0.30)
    p.add_argument("--seed_base", type=int, default=80_800_000,
                   help="Disjoint seed family for Win A")
    p.add_argument("--M_explore", type=int, default=30)
    p.add_argument("--M_refine", type=int, default=50)
    p.add_argument("--M_validate", type=int, default=100)
    args = p.parse_args()

    print(f"\n=== Win A widened-grid refinement: {args.dataset} axis={args.axis} ===")
    print(f"Widened grid: {args.grid}")

    # Load Stage 4 winner config + score
    base_cfg = load_winner_config(args.dataset)
    base_score = get_existing_winner_score(args.dataset)
    print(f"Stage 4 winner config: {base_cfg}")
    print(f"Stage 4 winner blind score: {base_score:.4f}")
    print(f"Acceptance threshold: new score must be < {base_score - ACCEPTANCE_EPSILON:.4f}")

    # Build DGP
    residualize = _RESIDUALIZE_DEFAULT.get(args.dataset, False)
    rho = args.rho_coarsen if DATASETS[args.dataset].get("binary_a") else None
    dgp = RealDataDGP(args.dataset, rho_coarsen=rho, residualize=residualize)
    bw = dgp.bandwidths()
    print(f"Bandwidths: h_KDE={bw['h_KDE']:.4f}  ell_W={bw['ell_W']:.4f}  "
          f"ell_A={bw['ell_A']:.4f}  ell_Z={bw['ell_Z']:.4f}")
    print(f"a_grid: {[round(float(x), 3) for x in dgp.a_grid]}  ref_idx={dgp.ref_idx}")

    raw_dir = _RAW_BASE / args.dataset / "bennett" / "winA"

    # ─── Sub-stage 1bis: axis sweep at M=M_explore ──────────────────────────
    print(f"\n--- Sub-stage 1bis: axis sweep on {args.axis}, M={args.M_explore} ---")
    sub1 = []
    for i, val in enumerate(args.grid):
        cfg = deepcopy(base_cfg)
        cfg[args.axis] = float(val)
        label = f"winA_S1_{args.axis}_{i}"
        seed = args.seed_base + i * 1_000
        t0 = time.time()
        try:
            res = run_one(cfg, dgp, bw, M=args.M_explore, seed_base=seed,
                          raw_dir=raw_dir / "stage1", label=label)
            elapsed = time.time() - t0
            sc = f"{res['score']:.4f}" if np.isfinite(res['score']) else "inf"
            print(f"  {args.axis}={val}  score={sc}  slope={res['slope_mean']*100:+.2f}% "
                  f"+- {res['slope_se']*100:.2f}pp  [{elapsed:.0f}s]")
            sub1.append(res)
        except Exception as e:
            print(f"  {args.axis}={val}  ERROR: {e}")
            sub1.append({"label": label, "config": cfg, "score": float("inf"),
                         "error": str(e)})

    valid_1 = [r for r in sub1 if np.isfinite(r.get("score", float("inf")))]
    valid_1.sort(key=lambda r: r["score"])
    if not valid_1:
        print("\nNO valid configs in sub-stage 1bis — aborting.")
        return
    top2 = valid_1[:2]
    print(f"\nSub-stage 1bis top-2 by blind score:")
    for r in top2:
        print(f"  {args.axis}={r['config'][args.axis]}  score={r['score']:.4f}  "
              f"slope={r['slope_mean']*100:+.2f}%")

    # ─── Sub-stage 2bis: refinement on top-2 with disjoint seeds, M=M_refine ─
    print(f"\n--- Sub-stage 2bis: refinement on top-2, M={args.M_refine} (disjoint seeds) ---")
    sub2 = []
    for ci, r in enumerate(top2):
        label = f"winA_S2_c{ci}"
        seed = args.seed_base + 100_000 + ci * 10_000
        t0 = time.time()
        try:
            res = run_one(r["config"], dgp, bw, M=args.M_refine, seed_base=seed,
                          raw_dir=raw_dir / "stage2", label=label)
            elapsed = time.time() - t0
            print(f"  cfg{ci} ({args.axis}={r['config'][args.axis]}): "
                  f"score={res['score']:.4f}  slope={res['slope_mean']*100:+.2f}% "
                  f"+- {res['slope_se']*100:.2f}pp  [{elapsed:.0f}s]")
            sub2.append(res)
        except Exception as e:
            print(f"  cfg{ci} ERROR: {e}")
            sub2.append({"score": float("inf"), "error": str(e), "config": r["config"]})

    valid_2 = [r for r in sub2 if np.isfinite(r.get("score", float("inf")))]
    valid_2.sort(key=lambda r: r["score"])
    if not valid_2:
        print("\nNO valid configs in sub-stage 2bis — aborting.")
        return
    top1 = valid_2[0]
    print(f"\nSub-stage 2bis top-1: {args.axis}={top1['config'][args.axis]}  score={top1['score']:.4f}")

    # ─── Sub-stage 3bis: validation on top-1 with M=M_validate ──────────────
    print(f"\n--- Sub-stage 3bis: validation on top-1, M={args.M_validate} ---")
    seed = args.seed_base + 500_000
    t0 = time.time()
    val = run_one(top1["config"], dgp, bw, M=args.M_validate, seed_base=seed,
                  raw_dir=raw_dir / "stage3", label=f"winA_S3_validation")
    elapsed = time.time() - t0
    print(f"  Validation: score={val['score']:.4f}  slope={val['slope_mean']*100:+.2f}% "
          f"+- {val['slope_se']*100:.2f}pp")
    print(f"  95% CI = [{val['slope_ci'][0]*100:+.2f}%, {val['slope_ci'][1]*100:+.2f}%]  [{elapsed:.0f}s]")

    # ─── Acceptance criterion ───────────────────────────────────────────────
    new_score = val["score"]
    accepted = (new_score < base_score - ACCEPTANCE_EPSILON)
    print(f"\n=== ACCEPTANCE ===")
    print(f"Stage 4 winner score: {base_score:.4f}")
    print(f"Win A winner score:   {new_score:.4f}")
    print(f"Improvement: {base_score - new_score:+.4f}  (threshold > {ACCEPTANCE_EPSILON})")
    if accepted:
        print(f"  ->ACCEPTED: Win A winner replaces Stage 4 winner.")
    else:
        print(f"  ->REJECTED: blind score not strictly improved. Original winner stands.")

    # ─── Write summary markdown ─────────────────────────────────────────────
    md_path = _SUMM_DIR / f"realdata_{args.dataset}_winA_widening.md"
    import datetime
    lines = [
        f"# Phase 19 — {args.dataset} Win A widened-grid refinement",
        f"*Generated {datetime.datetime.now().isoformat()}*",
        "",
        f"## Methodology",
        "",
        "After Stage 4 we observed `{}` was at the boundary of the original Phase 18 grid. ".format(args.axis) +
        "To verify this was not a grid-design artifact, we ran a second round of blind tuning " +
        "on a widened grid following the same 3-sub-stage protocol (axis sweep M={} → refinement ".format(args.M_explore) +
        "M={} → validation M={}). The new winner is accepted ONLY if its blind score ".format(args.M_refine, args.M_validate) +
        "strictly improves on the original by > {}.".format(ACCEPTANCE_EPSILON),
        "",
        f"## Original Stage 4 winner",
        f"```json",
        json.dumps(base_cfg, indent=2),
        f"```",
        f"- Blind score: **{base_score:.4f}**",
        "",
        f"## Widened grid for axis `{args.axis}`",
        f"- Values tested: {args.grid}",
        f"- Other axes held at Stage 4 winner values",
        "",
        f"## Sub-stage 1bis results (axis sweep, M={args.M_explore})",
        "",
        "| value | blind score | slope (%) | slope SE (pp) |",
        "|---|---|---|---|",
    ]
    for r in sub1:
        if r.get("score", float("inf")) == float("inf"):
            lines.append(f"| {r['config'][args.axis]} | inf | — | — |")
        else:
            lines.append(
                f"| {r['config'][args.axis]} | {r['score']:.4f} | "
                f"{r['slope_mean']*100:+.2f}% | {r['slope_se']*100:.2f} |"
            )
    lines += [
        "",
        f"## Sub-stage 2bis results (top-2, M={args.M_refine}, disjoint seeds)",
        "",
        "| ci | axis value | blind score | slope (%) | slope SE (pp) |",
        "|---|---|---|---|---|",
    ]
    for ci, r in enumerate(sub2):
        if r.get("score", float("inf")) == float("inf"):
            lines.append(f"| {ci} | {r['config'][args.axis]} | inf | — | — |")
        else:
            lines.append(
                f"| {ci} | {r['config'][args.axis]} | {r['score']:.4f} | "
                f"{r['slope_mean']*100:+.2f}% | {r['slope_se']*100:.2f} |"
            )
    lines += [
        "",
        f"## Sub-stage 3bis validation (top-1, M={args.M_validate})",
        "",
        f"- Winner config `{args.axis}` value: **{top1['config'][args.axis]}**",
        f"- Blind score validation: **{val['score']:.4f}**",
        f"- Slope endpoint: **{val['slope_mean']*100:+.3f}%** ± {val['slope_se']*100:.3f}pp",
        f"- 95% CI: **[{val['slope_ci'][0]*100:+.3f}%, {val['slope_ci'][1]*100:+.3f}%]**",
        "",
        f"## Acceptance",
        "",
        f"- Original Stage 4 winner score: {base_score:.4f}",
        f"- Win A winner score: {new_score:.4f}",
        f"- Improvement: {base_score - new_score:+.4f}",
        f"- Threshold (epsilon): {ACCEPTANCE_EPSILON}",
        f"- **{'✅ ACCEPTED' if accepted else '❌ REJECTED'}**: " +
        ("Win A winner replaces Stage 4 winner. The blind tuning genuinely benefits from grid widening on this axis." if accepted else
         "Blind score did not strictly improve. Original winner stands. The boundary was not a grid-design artifact in terms of blind quality, even if a different config gives a slope closer to literature."),
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown written: {md_path}")

    # Save the accepted winner config for present_results.py to pick up
    if accepted:
        winner_json = _RAW_BASE / args.dataset / "bennett" / "winA" / "winner_config.json"
        winner_json.parent.mkdir(parents=True, exist_ok=True)
        out = {"config": top1["config"],
               "score": new_score, "base_score": base_score,
               "slope_mean": val["slope_mean"], "slope_se": val["slope_se"],
               "slope_ci": val["slope_ci"], "M_validate": args.M_validate,
               "axis_widened": args.axis,
               "axis_value": float(top1["config"][args.axis])}
        winner_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Winner config written: {winner_json}")


if __name__ == "__main__":
    main()
