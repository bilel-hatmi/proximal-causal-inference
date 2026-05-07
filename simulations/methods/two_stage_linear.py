"""Shim — moved to :mod:`pci.estimators.two_stage_linear`.

This module re-exports symbols from their new home for backward
compatibility. Update existing imports to ``from pci.estimators.two_stage_linear import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators.two_stage_linear import *  # noqa: F401,F403
from pci.estimators.two_stage_linear import (  # noqa: F401
    TwoStageLeastSquares,
)
