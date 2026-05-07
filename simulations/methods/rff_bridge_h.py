"""Shim — moved to :mod:`pci.estimators.rff_bridge_h`.

This module re-exports symbols from their new home for backward
compatibility. Update existing imports to ``from pci.estimators.rff_bridge_h import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators.rff_bridge_h import *  # noqa: F401,F403
from pci.estimators.rff_bridge_h import (  # noqa: F401
    RFFFeatureMap1D,
    RFFBridgeH,
)
