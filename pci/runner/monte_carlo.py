"""
Monte Carlo harness — generic runner for all PCI estimation methods.

Usage
-----
    from pci.runner.monte_carlo import run_monte_carlo, run_pilot

    dgp = CobbDouglasLinearDGP()
    df  = run_monte_carlo(dgp, OracleDirect, n=1000, M=200, seed_base=0)
    # df has columns: psi_hat, V_hat, seed
    # pkl auto-saved to results/raw/OracleDirect_CobbDouglasLinearDGP_n1000_M200.pkl

Parallelism
-----------
    Set n_jobs > 1 (or -1 for all cores) to use joblib.
    Default n_jobs=1 is sequential — safe on all platforms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Type

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from pci.dgps.base import PCIDgp
from pci.estimators.base import PCIEstimator

# ── Paths ─────────────────────────────────────────────────────────────────────
# harness.py lives at  pci_essay/simulations/experiments/harness.py
# parents[0] = experiments/
# parents[1] = simulations/
# parents[2] = pci_essay/
_RAW_DIR: Path = Path(__file__).parents[2] / "simulations" / "results" / "raw"


# ── Single-replication helper (module-level → picklable by joblib) ────────────

def _run_one(
    dgp: PCIDgp,
    method_cls: Type[PCIEstimator],
    n: int,
    seed: int,
) -> dict:
    """Run one MC replication. Returns a dict with psi_hat, V_hat, seed."""
    sample = dgp.generate(n=n, seed=seed)
    result = method_cls().fit(sample).estimate()
    return {"psi_hat": result.psi_hat, "V_hat": result.V_hat, "seed": seed}


# ── Main harness ──────────────────────────────────────────────────────────────

def run_monte_carlo(
    dgp: PCIDgp,
    method_cls: Type[PCIEstimator],
    n: int,
    M: int,
    seed_base: int,
    save_path: Optional[Path] = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Run M Monte Carlo replications and return a tidy DataFrame.

    Parameters
    ----------
    dgp         : DGP instance (PCIDgp subclass)
    method_cls  : estimator CLASS (not instance) — fresh instantiation per rep
    n           : sample size per replication
    M           : number of replications
    seed_base   : replication i uses seed = seed_base + i  (reproducible)
    save_path   : explicit pkl path; if None, auto-named in results/raw/
    n_jobs      : joblib parallelism (1 = sequential, -1 = all cores)

    Returns
    -------
    pd.DataFrame with columns [psi_hat, V_hat, seed]
    Pkl saved to disk BEFORE return (principle P5).
    """
    seeds = [seed_base + i for i in range(M)]

    if n_jobs != 1 and _JOBLIB_AVAILABLE:
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_run_one)(dgp, method_cls, n, s) for s in seeds
        )
    else:
        rows = [_run_one(dgp, method_cls, n, s) for s in seeds]

    df = pd.DataFrame(rows)

    # ── Save pkl ──────────────────────────────────────────────────────────────
    if save_path is None:
        fname = (
            f"{method_cls.__name__}_{dgp.__class__.__name__}"
            f"_n{n}_M{M}.pkl"
        )
        save_path = _RAW_DIR / fname

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(save_path)
    print(f"  Saved -> {save_path}")

    return df


# ── Pilot convenience function ────────────────────────────────────────────────

def run_pilot(n: int = 1_000, M: int = 200, seed_base: int = 0) -> dict:
    """
    Run Oracle + Naive pilot on DGP1 (CobbDouglasLinearDGP defaults).

    Prints a comparison table and checks the coherence triplet:
        |bias_oracle| < |bias_naive| × 0.1
        cv_var_oracle  < 0.3

    Returns
    -------
    dict {"oracle": metrics_dict, "naive": metrics_dict}
    """
    # Local imports to avoid circular deps at module level
    from pci.metrics.canonical import compute_metrics
    from pci.dgps.cobb_douglas import CobbDouglasLinearDGP
    from pci.estimators.naive import NaiveRegression
    from pci.estimators.oracle import OracleDirect

    dgp   = CobbDouglasLinearDGP()
    psi_0 = dgp.psi_true(a=1.0)

    print(f"\n{'=' * 56}")
    print(f"PILOTE MONTE CARLO   n={n}   M={M}   seed_base={seed_base}")
    print(f"DGP : {dgp!r}   psi_0 = {psi_0:.4f}")
    print(f"{'=' * 56}\n")

    df_oracle = run_monte_carlo(dgp, OracleDirect,    n, M, seed_base)
    df_naive  = run_monte_carlo(dgp, NaiveRegression, n, M, seed_base)

    m_o = compute_metrics(
        df_oracle["psi_hat"].values, df_oracle["V_hat"].values, psi_0, n
    )
    m_n = compute_metrics(
        df_naive["psi_hat"].values,  df_naive["V_hat"].values,  psi_0, n
    )

    # ── Print comparison table ────────────────────────────────────────────────
    header = f"\n  {'Metric':<12} {'Oracle':>10} {'Naive':>10}"
    print(header)
    print("  " + "-" * (len(header) - 3))
    for k in ["bias", "variance", "rmse", "coverage", "ser", "cv_var"]:
        print(f"  {k:<12} {m_o[k]:>+10.4f} {m_n[k]:>+10.4f}")

    # ── Coherence checks ──────────────────────────────────────────────────────
    bias_ratio = abs(m_o["bias"]) / (abs(m_n["bias"]) + 1e-9)
    ok_triplet = bias_ratio < 0.1
    ok_cvvar   = m_o["cv_var"] < 0.3

    print()
    status_t = "PASS" if ok_triplet else "FAIL"
    status_c = "PASS" if ok_cvvar   else "FAIL"
    print(f"  [{status_t}]  |bias_oracle| / |bias_naive| = {bias_ratio:.4f}  (target < 0.1)")
    print(f"  [{status_c}]  cv_var_oracle               = {m_o['cv_var']:.4f}  (target < 0.3)")

    n_pass = int(ok_triplet) + int(ok_cvvar)
    print(f"\n  {n_pass}/2 coherence checks passed")
    print(f"{'=' * 56}\n")

    return {"oracle": m_o, "naive": m_n}
