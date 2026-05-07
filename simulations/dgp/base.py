"""Shim — moved to :mod:`pci.dgps.base`.

This module re-exports symbols from their new home for backward compatibility.
Update existing imports to ``from pci.dgps.base import ...`` at your earliest
convenience; this shim will be removed in a future release.
"""
from __future__ import annotations

from pci.dgps.base import *  # noqa: F401,F403
from pci.dgps.base import DGPSample, PCIDgp  # noqa: F401  (explicit re-export)
