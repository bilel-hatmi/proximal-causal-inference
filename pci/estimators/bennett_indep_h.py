"""
pci.estimators.bennett_indep_h -- backward-compatibility shim
==============================================================

The canonical implementation lives at :mod:`pci.bridges.bennett_h`. This module re-exports every public and module-level
private symbol so that:

* ``pci.estimators.bennett_indep_h.BennettIndepBridgeH`` is the **same
  Python object** as ``pci.bridges.bennett_h.BennettIndepBridgeH``
  (preserves pickle compatibility for cached PKLs in
  ``simulations/results/raw/``).
* All older imports continue to work without code changes.

Symbols re-exported::

    BennettIndepBridgeH
    _spd_solve
"""
from pci.bridges.bennett_h import (  # noqa: F401
    BennettIndepBridgeH,
    _spd_solve,
)
