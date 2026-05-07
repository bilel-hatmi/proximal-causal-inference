"""Shim — moved to :mod:`pci.bridges.kpv`.

This module re-exports KPVBridgeH and KPVPolicyBridgeQ (plus the private
helpers used by tests and other bridges) from their new canonical home for
backward compatibility. Update existing imports to
``from pci.bridges.kpv import ...`` at your earliest convenience.
"""
from __future__ import annotations

from pci.bridges.kpv import *  # noqa: F401,F403
from pci.bridges.kpv import (  # noqa: F401
    KPVBridgeH,
    KPVPolicyBridgeQ,
    _median_bandwidth,
    _rbf_gram,
    _rbf_kde_convolution,
    _spd_solve,
)
