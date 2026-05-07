"""Shim — moved to :mod:`pci.estimators.method_registry`.

This module re-exports symbols from their new home for backward
compatibility. Update existing imports to ``from pci.estimators.method_registry import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators.method_registry import *  # noqa: F401,F403
from pci.estimators.method_registry import (  # noqa: F401
    OraclePolicy,
    NaivePolicy,
    LinearBridgeREG,
    LinearBridgeDR,
    KPVPolicy,
    make_oracle,
    make_naiveREG,
    make_linearREG,
    make_linearDR,
    make_KPVREG,
    make_DRKPV,
    make_best_bennett,
    make_best_kallus,
    METHOD_REGISTRY,
    list_methods,
    make_method,
)
