"""Shim — moved to :mod:`pci.estimators.oracle`.

This module re-exports symbols from their new home for backward
compatibility. Update existing imports to ``from pci.estimators.oracle import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators.oracle import *  # noqa: F401,F403
from pci.estimators.oracle import (  # noqa: F401
    OracleDirect,
)
