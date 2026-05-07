"""Shim — moved to :mod:`pci.estimators.kallus_minimax`.

This module re-exports symbols from their new home for backward
compatibility. Update existing imports to ``from pci.estimators.kallus_minimax import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators.kallus_minimax import *  # noqa: F401,F403
from pci.estimators.kallus_minimax import (  # noqa: F401
    KallusStabilizedPolicyQ,
    KallusMinimaxBridgeH,
)
