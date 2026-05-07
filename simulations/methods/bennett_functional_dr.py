"""Shim — moved to :mod:`pci.estimators.bennett_functional_dr`.

This module re-exports symbols from their new home for backward
compatibility. Update existing imports to ``from pci.estimators.bennett_functional_dr import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators.bennett_functional_dr import *  # noqa: F401,F403
from pci.estimators.bennett_functional_dr import (  # noqa: F401
    BennettFunctionalDR,
)
