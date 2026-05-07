"""Shim — moved to :mod:`pci.estimators.dr_dose_response`.

This module re-exports symbols from their new home for backward
compatibility. Update existing imports to ``from pci.estimators.dr_dose_response import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators.dr_dose_response import *  # noqa: F401,F403
from pci.estimators.dr_dose_response import (  # noqa: F401
    DRDoseResponse,
)
