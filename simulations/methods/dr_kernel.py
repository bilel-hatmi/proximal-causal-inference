"""Shim — moved to :mod:`pci.estimators.dr_kernel`.

This module re-exports symbols from their new home for backward
compatibility. Update existing imports to ``from pci.estimators.dr_kernel import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators.dr_kernel import *  # noqa: F401,F403
from pci.estimators.dr_kernel import (  # noqa: F401
    DRKernel,
)
