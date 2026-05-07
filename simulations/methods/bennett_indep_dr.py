"""Shim — moved to :mod:`pci.estimators.bennett_indep_dr`.

This module re-exports symbols from their new home for backward
compatibility. Update existing imports to ``from pci.estimators.bennett_indep_dr import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators.bennett_indep_dr import *  # noqa: F401,F403
from pci.estimators.bennett_indep_dr import (  # noqa: F401
    BennettIndepFunctionalDR,
)
