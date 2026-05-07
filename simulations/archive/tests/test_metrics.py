"""Tests unitaires pour compute_metrics."""
import numpy as np
import pytest
from simulations.analysis.metrics import compute_metrics


# ── T1 : Bias ──────────────────────────────────────────────────────────────────

def test_bias_deterministic():
    psi_hat = np.full(100, 0.31)
    V_hat = np.full(100, 1.0)
    m = compute_metrics(psi_hat, V_hat, psi_0=0.30, n=1000)
    assert abs(m["bias"] - 0.01) < 1e-10


def test_bias_zero():
    psi_hat = np.full(200, 0.30)
    V_hat = np.full(200, 1.0)
    m = compute_metrics(psi_hat, V_hat, psi_0=0.30, n=1000)
    assert abs(m["bias"]) < 1e-12


# ── T2 : RMSE ──────────────────────────────────────────────────────────────────

def test_rmse_pure_bias():
    """Zero variance -> RMSE = |bias|."""
    psi_hat = np.full(100, 0.35)
    V_hat = np.full(100, 1.0)
    m = compute_metrics(psi_hat, V_hat, psi_0=0.30, n=1000)
    assert abs(m["rmse"] - abs(m["bias"])) < 1e-10


def test_rmse_formula():
    rng = np.random.default_rng(0)
    psi_hat = rng.normal(0.30, 0.05, 1000)
    V_hat = np.full(1000, 1.0)
    m = compute_metrics(psi_hat, V_hat, psi_0=0.30, n=1000)
    expected = np.sqrt(m["bias"] ** 2 + m["variance"])
    assert abs(m["rmse"] - expected) < 1e-12


# ── T3 : Coverage ──────────────────────────────────────────────────────────────

def test_coverage_well_calibrated():
    """Gaussian estimates with true SE -> empirical coverage ~0.95 (+/-0.02)."""
    rng = np.random.default_rng(42)
    M, n, sigma2, psi_0 = 20_000, 1000, 4.0, 0.30
    psi_hat = rng.normal(psi_0, np.sqrt(sigma2 / n), M)
    V_hat = np.full(M, sigma2)
    m = compute_metrics(psi_hat, V_hat, psi_0=psi_0, n=n)
    assert abs(m["coverage"] - 0.95) < 0.02, f"Got {m['coverage']:.4f}"


def test_coverage_underestimated_se():
    """4x underestimated variance -> coverage well below 0.95."""
    rng = np.random.default_rng(1)
    M, n, sigma2, psi_0 = 10_000, 1000, 4.0, 0.30
    psi_hat = rng.normal(psi_0, np.sqrt(sigma2 / n), M)
    V_hat = np.full(M, sigma2 * 0.25)
    m = compute_metrics(psi_hat, V_hat, psi_0=psi_0, n=n)
    assert m["coverage"] < 0.80


# ── T4 : SER ───────────────────────────────────────────────────────────────────

def test_ser_well_calibrated():
    """True variance estimator -> SER ~1 (+/-0.05)."""
    rng = np.random.default_rng(7)
    M, n, sigma2 = 10_000, 500, 2.0
    psi_hat = rng.normal(0.0, np.sqrt(sigma2 / n), M)
    V_hat = np.full(M, sigma2)
    m = compute_metrics(psi_hat, V_hat, psi_0=0.0, n=n)
    assert abs(m["ser"] - 1.0) < 0.05, f"Got {m['ser']:.4f}"


def test_ser_underestimated():
    """Half SE -> SER < 0.7."""
    rng = np.random.default_rng(8)
    M, n, sigma2 = 5_000, 500, 2.0
    psi_hat = rng.normal(0.0, np.sqrt(sigma2 / n), M)
    V_hat = np.full(M, sigma2 * 0.25)
    m = compute_metrics(psi_hat, V_hat, psi_0=0.0, n=n)
    assert m["ser"] < 0.7


# ── T5 : cv_var ────────────────────────────────────────────────────────────────

def test_cv_var_constant():
    """Constant V_hat -> cv_var = 0."""
    psi_hat = np.full(100, 0.3)
    V_hat = np.full(100, 2.5)
    m = compute_metrics(psi_hat, V_hat, psi_0=0.3, n=1000)
    assert m["cv_var"] < 1e-12


def test_cv_var_positive():
    """Variable V_hat -> cv_var > 0."""
    rng = np.random.default_rng(9)
    psi_hat = np.full(200, 0.3)
    V_hat = rng.uniform(0.5, 2.0, 200)
    m = compute_metrics(psi_hat, V_hat, psi_0=0.3, n=1000)
    assert m["cv_var"] > 0.1


# ── T6 : interface ────────────────────────────────────────────────────────────

def test_output_keys():
    m = compute_metrics([0.3, 0.31, 0.29], [1.0, 1.0, 1.0], psi_0=0.30, n=100)
    expected_keys = {"bias", "variance", "rmse", "coverage", "ser", "cv_var", "n", "M", "alpha"}
    assert set(m.keys()) == expected_keys
    assert m["n"] == 100
    assert m["M"] == 3
    assert m["alpha"] == 0.05


def test_accepts_lists():
    """Should accept plain Python lists without error."""
    m = compute_metrics([0.30, 0.31], [1.0, 1.0], psi_0=0.30, n=100)
    assert isinstance(m["bias"], float)
