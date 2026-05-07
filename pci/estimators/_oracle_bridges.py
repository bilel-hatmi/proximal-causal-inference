"""Shim — moved to :mod:`pci.bridges.oracle`.

This module re-exports the oracle and wrong-bridge wrappers from their new
canonical home for backward compatibility. Update existing imports to
``from pci.bridges.oracle import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.bridges.oracle import *  # noqa: F401,F403
from pci.bridges.oracle import (  # noqa: F401
    OracleBridgeH,
    OracleBridgeQ,
    WrongBridgeH_DropsW,
    WrongBridgeQ_Constant,
)
