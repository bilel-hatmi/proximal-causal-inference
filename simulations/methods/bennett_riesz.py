"""Shim — moved to :mod:`pci.estimators.bennett_riesz`.

This module re-exports symbols from their new home for backward
compatibility. Update existing imports to ``from pci.estimators.bennett_riesz import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators.bennett_riesz import *  # noqa: F401,F403
from pci.estimators.bennett_riesz import (  # noqa: F401
    gaussian_raw_moment,
    PolynomialPairFeatureMap,
    PolicyPolynomialIntegrator,
    BennettPolicyRiesz,
    RandomFourierPairFeatureMap,
    RandomFourierPolicyIntegrator,
    StabilizedBennettPolicyRiesz,
)
