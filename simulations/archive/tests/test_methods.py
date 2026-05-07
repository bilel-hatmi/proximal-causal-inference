"""
Tests unitaires pour les méthodes d'estimation PCI.

verify_method() : diagnostic console réutilisable pour tout PCIEstimator.
test_*          : suites pytest pour chaque méthode concrète.

Convention (Option A, décidée en session) :
    verify_method() est générique et inclut M3 (coverage plausible).
    NaiveRegression échoue M3 intentionnellement — c'est documenté,
    pas un bug. Les tests pytest de Naive vérifient explicitement
    le biais et la coverage basse.
"""

import numpy as np
import pandas as pd
import pytest

from simulations.analysis.metrics import compute_metrics
from simulations.dgp.base import DGPSample
from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP
from simulations.methods.base import EstimationResult, PCIEstimator
from simulations.methods.naive import NaiveRegression
from simulations.methods.oracle import OracleDirect
from simulations.methods.two_stage_linear import TwoStageLeastSquares


# ── verify_method() — réutilisable pour tous les estimateurs ─────────────────

def verify_method(
    method_cls,
    dgp,
    n: int = 2_000,
    M: int = 100,
    seed: int = 42,
) -> dict:
    """
    Run M replications and compute the 6 metrics + 5 sanity checks.

    Checks
    ------
    M1  No NaN in psi_hat
    M2  Finite variance  (< 10.0)
    M3  Coverage plausible  (0.70 < coverage < 1.0)
        NOTE: NaiveRegression fails M3 by design — document, do not suppress.
    M4  SER plausible  (0.5 < SER < 2.0)
    M5  estimate() returns an EstimationResult instance

    Returns
    -------
    dict with keys 'metrics' and 'checks', prints a summary to stdout.
    """
    psi_0 = dgp.psi_true(a=1.0)
    rows = []
    rng = np.random.default_rng(seed)

    # Run one extra estimation to check M5 type
    sample_m5 = dgp.generate(n=n, seed=int(rng.integers(0, 2**31)))
    result_m5 = method_cls().fit(sample_m5).estimate()

    for _ in range(M):
        s = dgp.generate(n=n, seed=int(rng.integers(0, 2**31)))
        res = method_cls().fit(s).estimate()
        rows.append({"psi_hat": res.psi_hat, "V_hat": res.V_hat})

    df = pd.DataFrame(rows)
    metrics = compute_metrics(df["psi_hat"].values, df["V_hat"].values, psi_0, n)

    checks = {
        "M1_not_nan":              bool(not np.any(np.isnan(df["psi_hat"]))),
        "M2_finite_variance":      bool(metrics["variance"] < 10.0),
        "M3_coverage_plausible":   bool(0.70 < metrics["coverage"] < 1.0),
        "M4_ser_plausible":        bool(0.5 < metrics["ser"] < 2.0),
        "M5_returns_EstResult":    isinstance(result_m5, EstimationResult),
    }

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 52}")
    print(f"METHOD VERIFICATION : {method_cls.__name__}")
    print(f"DGP : {dgp!r}  |  n={n}  |  M={M}")
    print(f"{'=' * 52}")
    print(f"  bias     = {metrics['bias']:+.4f}   (psi_0 = {psi_0:.4f})")
    print(f"  variance = {metrics['variance']:.4f}")
    print(f"  rmse     = {metrics['rmse']:.4f}")
    print(f"  coverage = {metrics['coverage']:.3f}  (target: 0.95)")
    print(f"  SER      = {metrics['ser']:.3f}   (target: 1.0)")
    print(f"  cv_var   = {metrics['cv_var']:.3f}")
    print()
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {check_name}")
    n_pass = sum(checks.values())
    print(f"\n  {n_pass}/{len(checks)} checks passed")
    print(f"{'=' * 52}\n")

    return {"metrics": metrics, "checks": checks}


# ── Pytest suite — OracleDirect ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def oracle_verify():
    dgp = CobbDouglasLinearDGP()
    return verify_method(OracleDirect, dgp, n=2_000, M=100, seed=42)


def test_oracle_M1_not_nan(oracle_verify):
    assert oracle_verify["checks"]["M1_not_nan"]


def test_oracle_M2_finite_variance(oracle_verify):
    assert oracle_verify["checks"]["M2_finite_variance"]


def test_oracle_M3_coverage(oracle_verify):
    """Oracle must have coverage in [0.70, 1.0]."""
    assert oracle_verify["checks"]["M3_coverage_plausible"], (
        f"coverage = {oracle_verify['metrics']['coverage']:.3f}"
    )


def test_oracle_M4_ser(oracle_verify):
    assert oracle_verify["checks"]["M4_ser_plausible"], (
        f"SER = {oracle_verify['metrics']['ser']:.3f}"
    )


def test_oracle_M5_type(oracle_verify):
    assert oracle_verify["checks"]["M5_returns_EstResult"]


def test_oracle_low_bias(oracle_verify):
    """Oracle bias must be negligible (< 0.02)."""
    assert abs(oracle_verify["metrics"]["bias"]) < 0.02, (
        f"bias = {oracle_verify['metrics']['bias']:+.4f}"
    )


def test_oracle_coverage_near_95(oracle_verify):
    """Oracle coverage must be in [0.88, 1.0] — well-calibrated CI."""
    cov = oracle_verify["metrics"]["coverage"]
    assert cov > 0.88, f"coverage = {cov:.3f}"


# ── Pytest suite — NaiveRegression ───────────────────────────────────────────

@pytest.fixture(scope="module")
def naive_verify():
    dgp = CobbDouglasLinearDGP()
    return verify_method(NaiveRegression, dgp, n=2_000, M=100, seed=42)


def test_naive_M1_not_nan(naive_verify):
    """Naive must never produce NaN."""
    assert naive_verify["checks"]["M1_not_nan"]


def test_naive_M2_finite_variance(naive_verify):
    assert naive_verify["checks"]["M2_finite_variance"]


def test_naive_M5_type(naive_verify):
    assert naive_verify["checks"]["M5_returns_EstResult"]


def test_naive_is_biased(naive_verify):
    """
    Naive must have a large positive bias (>> 0.05).
    Expected: bias ≈ +1.0 (confounding_bias = Cov(U,A)/Var(A) ≈ 1.0).
    """
    bias = naive_verify["metrics"]["bias"]
    assert bias > 0.05, f"Expected large positive bias, got {bias:+.4f}"


def test_naive_coverage_is_low(naive_verify):
    """
    Naive coverage must be << 0.70 (ICs centred around wrong value).
    This documents the expected failure of M3 for a biased estimator.
    """
    cov = naive_verify["metrics"]["coverage"]
    assert cov < 0.50, f"Expected coverage << 0.50, got {cov:.3f}"


def test_naive_M3_fails_by_design(naive_verify):
    """
    Explicit test that M3 fails for Naive — this is expected, not a bug.
    Documents Option A: verify_method() is generic, Naive fails M3 intentionally.
    """
    assert not naive_verify["checks"]["M3_coverage_plausible"], (
        "M3 should fail for NaiveRegression — if it passes, the DGP has no confounding"
    )


# ── Pytest suite — TwoStageLeastSquares ──────────────────────────────────────

@pytest.fixture(scope="module")
def twostage_verify():
    dgp = CobbDouglasLinearDGP()
    return verify_method(TwoStageLeastSquares, dgp, n=2_000, M=100, seed=42)


def test_twostage_M1_not_nan(twostage_verify):
    assert twostage_verify["checks"]["M1_not_nan"]


def test_twostage_M2_finite_variance(twostage_verify):
    assert twostage_verify["checks"]["M2_finite_variance"]


def test_twostage_M3_coverage(twostage_verify):
    """TwoStage must have coverage in [0.70, 1.0]."""
    assert twostage_verify["checks"]["M3_coverage_plausible"], (
        f"coverage = {twostage_verify['metrics']['coverage']:.3f}"
    )


def test_twostage_M4_ser(twostage_verify):
    assert twostage_verify["checks"]["M4_ser_plausible"], (
        f"SER = {twostage_verify['metrics']['ser']:.3f}"
    )


def test_twostage_M5_type(twostage_verify):
    assert twostage_verify["checks"]["M5_returns_EstResult"]


def test_twostage_low_bias(twostage_verify):
    """TwoStage is consistent on DGP1 — bias must be < 0.05."""
    assert abs(twostage_verify["metrics"]["bias"]) < 0.05, (
        f"bias = {twostage_verify['metrics']['bias']:+.4f}"
    )


def test_twostage_coverage_near_95(twostage_verify):
    """Two-stage sandwich must be well-calibrated: coverage > 0.88."""
    cov = twostage_verify["metrics"]["coverage"]
    assert cov > 0.88, f"coverage = {cov:.3f}"
