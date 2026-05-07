"""Shim — moved to :mod:`pci.dgps.cobb_douglas`.

This module re-exports symbols from their new home for backward compatibility.
Update existing imports to ``from pci.dgps.cobb_douglas import ...`` at your
earliest convenience.
"""
from __future__ import annotations

from pci.dgps.cobb_douglas import *  # noqa: F401,F403
from pci.dgps.cobb_douglas import CobbDouglasLinearDGP, DEFAULTS_CD  # noqa: F401
