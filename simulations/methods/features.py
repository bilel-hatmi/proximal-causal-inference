"""Shim — moved to :mod:`pci.estimators.features`.

This module re-exports symbols from their new home for backward
compatibility. Update existing imports to ``from pci.estimators.features import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators.features import *  # noqa: F401,F403
from pci.estimators.features import (  # noqa: F401
    NystromFeatureMap,
    PairFeatureMap,
    PolicyFeatureIntegrator,
    _rbf_gram_1d,
)
