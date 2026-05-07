"""
Phase 19 — Win B: validate winner config with M=200 (vs M=100 default).

Loads Stage 4 winner from FINAL.md state, runs M=200 bootstrap reps with
disjoint seeds (seed_base = 80_750_000 for cfg0, 80_760_000 for cfg1, etc.),
saves to .../bennett/winB/winner_M200.pkl.

Then re-runs present_results pipeline to regenerate PRESENTABLE.md with the
tighter M=200 slope CI.

Usage:
    python -u -m simulations.experiments.real_data_winB_validation --dataset nlsy79_father
    python -u -m simulations.experiments.real_data_winB_validation --dataset nhanes_cadmium
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from simulations.experiments.s8_applications.s8_3_blind_tuning import (
    DATASETS, _RESIDUALIZE_DEFAULT, RealDataDGP,
    _make_bennett_estimator_real, _BASE_DIR,
)
from simulations.experiments.dgp2_bias_diagnostics import (
    _silverman_h, _median_bandwidth, _run_block,
)

_RAW_BASE = _BASE_DIR / "simulations" / "results" / "raw" / "s8"


def load_winner_config(dataset: str) -> dict:
    """Read the winner config from the most recent FINAL.md (json block)."""
    final_path = _BASE_DIR / "simulations" / "results" / "summaries" / f"realdata_{dataset}_bennett_FINAL.md"
    if not final_path.exists():
        raise FileNotFoundError(f"FINAL.md missing: {final_path}")
    txt = final_path.read_text(encoding="utf-8")
    # Pull the JSON block
    import re
    m = re.search(r"```json\s*(\{.*?\})\s*```", txt, flags=re.DOTALL)
    if not m:
        raise RuntimeError("No json winner block in FINAL.md")
    cfg = json.loads(m.group(1))
    return cfg


def run_winB(dataset: str, *, M: int = 200, seed_base: int = 80_750_000,
             rho_coarsen: float = 0.30) -> None:
    print(f"\n=== Win B: M={M} validation on {dataset} winner ===")

    cfg = load_winner_config(dataset)
    print(f"Winner config: {cfg}")

    residualize = _RESIDUALIZE_DEFAULT.get(dataset, False)
    rho = rho_coarsen if DATASETS[dataset].get("binary_a") else None
    print(f"Residualize: {residualize}, rho_coarsen: {rho}")

    dgp = RealDataDGP(dataset, rho_coarsen=rho, residualize=residualize)
    bw = dgp.bandwidths()

    raw_dir = _RAW_BASE / dataset / "bennett" / "winB"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def factory():
        return _make_bennett_estimator_real(
            cfg, dgp, bw["h_KDE"], bw["ell_W"], bw["ell_A"], bw["ell_Z"],
            a_grid=dgp.a_grid, ref_idx=dgp.ref_idx,
        )

    label = f"winB_M{M}"
    print(f"\nLaunching {M} bootstrap reps (seed_base={seed_base}) ...")
    t0 = time.time()
    J_pt = np.full(len(dgp.a_grid), np.nan)
    data = _run_block(
        dgp, factory, np.asarray(dgp.a_grid), J_pt,
        n=dgp.N, M=M, seed_base=seed_base,
        label=label, raw_dir=raw_dir,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed/60:.1f} min")

    # Quick stats
    records = [r for r in data["records"] if r.get("error") is None]
    J_dr = np.array([r["J_dr"] for r in records])
    a_grid = np.asarray(dgp.a_grid)
    span = float(a_grid[-1] - a_grid[0])
    slope = (J_dr[:, -1] - J_dr[:, 0]) / span
    ci_lo, ci_hi = np.quantile(slope, 0.025), np.quantile(slope, 0.975)
    print(f"\nSlope (endpoint, M={len(records)}):")
    print(f"  mean = {slope.mean():+.4f}  SE = {slope.std(ddof=1):.4f}")
    print(f"  95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # Save symlink/copy of the result for present_results to find as Stage 4 cfg0
    # (so that present_results uses M=200 instead of M=100)
    # Strategy: also save a copy at the standard Stage 4 path with a "_winB_M200" suffix
    # that present_results can be told to load, OR overwrite the standard path with backup.
    # For now just leave it in winB/ and patch present_results to optionally use it.
    print(f"\nResult saved at: {raw_dir / (label + '.pkl')}")
    print(f"To re-generate PRESENTABLE.md with M=200 slope, edit present_results.py "
          f"to load from winB/winB_M{M}.pkl instead of stage4/S4_c0_validation.pkl.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",
                   choices=list(DATASETS.keys()), required=True)
    p.add_argument("--M", type=int, default=200)
    p.add_argument("--seed_base", type=int, default=80_750_000)
    p.add_argument("--rho_coarsen", type=float, default=0.30)
    args = p.parse_args()
    run_winB(args.dataset, M=args.M, seed_base=args.seed_base,
             rho_coarsen=args.rho_coarsen)


if __name__ == "__main__":
    main()
