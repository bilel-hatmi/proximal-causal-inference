"""Shim — moved to :mod:`pci.bridges.rff_h`.

This module re-exports the RFF h-bridge from its new canonical home for
backward compatibility. Update existing imports to
``from pci.bridges.rff_h import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.bridges.rff_h import *  # noqa: F401,F403
from pci.bridges.rff_h import (  # noqa: F401
    RFFBridgeH,
    RFFFeatureMap1D,
    _rff_gram,
)
