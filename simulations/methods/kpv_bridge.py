"""Shim — moved to :mod:`pci.estimators.kpv_bridge`.

This module re-exports symbols (including private helpers used by tests) from
their new home for backward compatibility. Update existing imports to
``from pci.estimators.kpv_bridge import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.estimators.kpv_bridge import *  # noqa: F401,F403
from pci.estimators.kpv_bridge import (  # noqa: F401
    KPVBridgeH,
    KPVPolicyBridgeQ,
    _median_bandwidth,
    _rbf_gram,
    _rbf_kde_convolution,
    _spd_solve,
)
