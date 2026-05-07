"""Shim — moved to :mod:`pci.estimators.naive`.

This module re-exports symbols from their new home for backward
compatibility. Update existing imports to ``from pci.estimators.naive import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators.naive import *  # noqa: F401,F403
from pci.estimators.naive import (  # noqa: F401
    NaiveRegression,
)
