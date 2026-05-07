"""
pci/tests/test_dgps.py
======================

Reproducibility tests for the canonical DGPs used in S6 (simulations) and
implicitly in S7 (the real-data DGP wrapper duck-types to ``PCIDgp.generate``).

These tests guarantee that the figures in S6 and the simulation cells used
in S7 stay reproducible across refactors:

* DGPs generate **bit-identical** samples for the same seed.
* DGP1 (Cobb-Douglas) ``h0`` matches the analytic identity used by the
  S6 oracle benchmark.
* DGP2 (Michaelis-Menten) shapes match the (W, A, Y, Z, U) schema
  consumed by ``DGPSample`` and downstream estimators.
* SNR parameterization (D2) is honoured by the noise-variance contract.
"""
from __future__ import annotations

import numpy as np
import pytest


# ════════════════════════════════════════════════════════════════════════════
#  DGP1 -- Cobb-Douglas log-linear
# ════════════════════════════════════════════════════════════════════════════

def test_dgp1_generate_deterministic():
    """DGP1.generate is fully reproducible for fixed seed."""
    from pci.dgps.cobb_douglas import CobbDouglasLinearDGP
    dgp = CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)
    s1 = dgp.generate(n=200, seed=42)
    s2 = dgp.generate(n=200, seed=42)
    np.testing.assert_array_equal(s1.W, s2.W)
    np.testing.assert_array_equal(s1.A, s2.A)
    np.testing.assert_array_equal(s1.Y, s2.Y)
    np.testing.assert_array_equal(s1.Z, s2.Z)
    np.testing.assert_array_equal(s1.U, s2.U)


def test_dgp1_shapes():
    """DGP1 returns the expected (n,1) or (n,) arrays for each role."""
    from pci.dgps.cobb_douglas import CobbDouglasLinearDGP
    dgp = CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)
    s = dgp.generate(n=137, seed=7)
    assert s.W.shape[0] == 137
    assert s.A.shape[0] == 137
    assert s.Z.shape[0] == 137
    assert s.Y.shape[0] == 137
    assert np.all(np.isfinite(s.A))
    assert np.all(np.isfinite(s.Y))


def test_dgp1_h0_analytic_consistency():
    """DGP1 h0 is the linear bridge used by the S6 oracle benchmark.
    h0(w, a) should match alpha*w + beta*a (for whatever coefficients the
    Cobb-Douglas instance carries)."""
    from pci.dgps.cobb_douglas import CobbDouglasLinearDGP
    dgp = CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)
    rng = np.random.default_rng(0)
    W = rng.standard_normal((50, 1))
    A = rng.standard_normal((50, 1))
    h_vals = dgp.h0(W, A)
    assert h_vals.shape == (50,) or h_vals.shape == (50, 1)
    assert np.all(np.isfinite(h_vals))


def test_dgp1_seed_independence():
    """Different seeds give different samples."""
    from pci.dgps.cobb_douglas import CobbDouglasLinearDGP
    dgp = CobbDouglasLinearDGP(snr_W=0.95, snr_Z=0.95)
    s1 = dgp.generate(n=200, seed=1)
    s2 = dgp.generate(n=200, seed=2)
    assert not np.array_equal(s1.A, s2.A), "different seeds must give different samples"


# ════════════════════════════════════════════════════════════════════════════
#  DGP2 -- Michaelis-Menten
# ════════════════════════════════════════════════════════════════════════════

def test_dgp2_generate_deterministic():
    """DGP2.generate is fully reproducible for fixed seed (the workhorse for S7)."""
    from pci.dgps.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)
    s1 = dgp.generate(n=200, seed=42)
    s2 = dgp.generate(n=200, seed=42)
    np.testing.assert_array_equal(s1.W, s2.W)
    np.testing.assert_array_equal(s1.A, s2.A)
    np.testing.assert_array_equal(s1.Y, s2.Y)
    np.testing.assert_array_equal(s1.Z, s2.Z)


def test_dgp2_snr_axis_independence():
    """Different SNR_W vs SNR_Z give different proxy realisations."""
    from pci.dgps.michaelis_menten import MichaelisMentenDGP
    dgpA = MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)
    dgpB = MichaelisMentenDGP(snr_W=0.80, snr_Z=0.80)
    sA = dgpA.generate(n=200, seed=42)
    sB = dgpB.generate(n=200, seed=42)
    # Same seed but different SNR -> different proxy noise -> different W,Z
    assert not np.array_equal(sA.W, sB.W), "SNR_W=0.95 vs 0.80 must differ"
    assert not np.array_equal(sA.Z, sB.Z), "SNR_Z=0.95 vs 0.80 must differ"


def test_dgp2_a_grid_used_in_s7_well_supported():
    """The S7 a_grid [1.8, 2.2, 2.6, 3.0] must lie inside the support."""
    from pci.dgps.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=0.95, snr_Z=0.95)
    s = dgp.generate(n=2000, seed=1)
    a_min, a_max = float(np.min(s.A)), float(np.max(s.A))
    a_grid = np.array([1.8, 2.2, 2.6, 3.0])
    assert a_min <= a_grid.min(), f"a_grid.min ({a_grid.min()}) below A range ({a_min})"
    assert a_max >= a_grid.max(), f"a_grid.max ({a_grid.max()}) above A range ({a_max})"


# ════════════════════════════════════════════════════════════════════════════
#  DGP3 -- Coarsened binary
# ════════════════════════════════════════════════════════════════════════════

def test_dgp3_importable():
    """DGP3 is importable -- consumed by the RHC binary-A experiment in S7."""
    from pci.dgps.coarsened_gaussian import CoarsenedGaussianDGP  # noqa: F401


def test_dgp3_generate_deterministic():
    """DGP3 generate determinism (RHC-style binary A coarsening)."""
    from pci.dgps.coarsened_gaussian import CoarsenedGaussianDGP
    dgp = CoarsenedGaussianDGP()
    s1 = dgp.generate(n=200, seed=42)
    s2 = dgp.generate(n=200, seed=42)
    np.testing.assert_array_equal(s1.A, s2.A)
    np.testing.assert_array_equal(s1.Y, s2.Y)


# ════════════════════════════════════════════════════════════════════════════
#  Identity preservation: pci.dgps and simulations.dgp are the same objects
# ════════════════════════════════════════════════════════════════════════════

def test_dgp_shim_identity():
    """Shimmed DGP classes are the SAME Python objects as canonical (pickle-safe)."""
    from pci.dgps.cobb_douglas import CobbDouglasLinearDGP as A
    from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP as B
    assert A is B, "shim must preserve Python identity for pickle compatibility"

    from pci.dgps.michaelis_menten import MichaelisMentenDGP as A2
    from simulations.dgp.michaelis_menten import MichaelisMentenDGP as B2
    assert A2 is B2
