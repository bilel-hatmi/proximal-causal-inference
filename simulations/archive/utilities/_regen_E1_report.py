"""Regenerate Phase 15C E1 report from existing pkls (no recompute)."""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import pickle
from pathlib import Path
import numpy as np

from simulations.experiments.dgp2_bias_diagnostics import _decompose
from simulations.archive.utilities.bennett_rff_overnight import _augment_decomp_with_ser
from simulations.experiments.simulation.dgp1_sanity import (
    _aggregate_metrics, _write_e1_report, _RAW_DIR, METHOD_REGISTRY,
)


def main():
    per_n = {}
    for n_dir in sorted(_RAW_DIR.glob("n*")):
        n = int(n_dir.name[1:])
        results = {}
        for method_name in METHOD_REGISTRY.keys():
            pkl = n_dir / f"{method_name}_M100.pkl"
            if not pkl.exists():
                results[method_name] = None
                continue
            try:
                with open(pkl, "rb") as fh:
                    data = pickle.load(fh)
                decomp = _decompose(data)
                if decomp is not None:
                    decomp = _augment_decomp_with_ser(decomp, data)
                    decomp["_data"] = data
                results[method_name] = decomp
            except Exception as e:
                print(f"  ERROR {method_name} n={n}: {e}")
                results[method_name] = None
        per_n[n] = results
    _write_e1_report(per_n)


if __name__ == "__main__":
    main()
