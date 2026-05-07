"""Shim — moved to :mod:`pci.estimators.base`.

This module re-exports symbols from their new home for backward
compatibility. Update existing imports to ``from pci.estimators.base import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators.base import *  # noqa: F401,F403
from pci.estimators.base import (  # noqa: F401
    EstimationResult,
    PCIEstimator,
)
