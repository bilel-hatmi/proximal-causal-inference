"""Shim — moved to :mod:`pci.estimators._oracle_bridges`.

This module re-exports symbols from their new home for backward
compatibility. Update existing imports to ``from pci.estimators._oracle_bridges import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators._oracle_bridges import *  # noqa: F401,F403
from pci.estimators._oracle_bridges import (  # noqa: F401
    OracleBridgeH,
    OracleBridgeQ,
    WrongBridgeH_DropsW,
    WrongBridgeQ_Constant,
)
