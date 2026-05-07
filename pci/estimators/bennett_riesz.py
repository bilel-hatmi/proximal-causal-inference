"""
pci.estimators.bennett_riesz -- backward-compatibility shim
============================================================

The canonical implementation lives at :mod:`pci.bridges.bennett_riesz`. This module re-exports every public and module-level
private symbol so that:

* ``pci.estimators.bennett_riesz.BennettPolicyRiesz`` is the **same
  Python object** as ``pci.bridges.bennett_riesz.BennettPolicyRiesz``
  (preserves pickle compatibility for cached PKLs in
  ``simulations/results/raw/``).
* All older imports continue to work without code changes.

Symbols re-exported::

    gaussian_raw_moment
    _spd_solve_bennett
    PolynomialPairFeatureMap
    PolicyPolynomialIntegrator
    BennettPolicyRiesz
    RandomFourierPairFeatureMap
    RandomFourierPolicyIntegrator
    StabilizedBennettPolicyRiesz
"""
from pci.bridges.bennett_riesz import (  # noqa: F401
    gaussian_raw_moment,
    _spd_solve_bennett,
    PolynomialPairFeatureMap,
    PolicyPolynomialIntegrator,
    BennettPolicyRiesz,
    RandomFourierPairFeatureMap,
    RandomFourierPolicyIntegrator,
    StabilizedBennettPolicyRiesz,
)
