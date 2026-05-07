"""Regenerate Phase 15C E3 report from existing pkls (no recompute)."""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import pickle
from pathlib import Path
import numpy as np

from simulations.experiments.dgp2_bias_diagnostics import _decompose
from simulations.archive.utilities.bennett_rff_overnight import _augment_decomp_with_ser
from simulations.experiments.simulation.proxy_asymmetry import (
    _write_e3_report, _RAW_DIR, METHODS_E3, SNR_CELLS,
)


def main():
    per_cell = {}
    for snr_W, snr_Z in SNR_CELLS:
        cell_tag = f"snrW{int(snr_W*100):02d}_snrZ{int(snr_Z*100):02d}"
        cell_dir = _RAW_DIR / cell_tag
        results = {}
        for method_name in METHODS_E3:
            pkl = cell_dir / f"{method_name}_M50.pkl"
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
                print(f"  ERROR {method_name} cell={cell_tag}: {e}")
                results[method_name] = None
        per_cell[(snr_W, snr_Z)] = results
    _write_e3_report(per_cell)


if __name__ == "__main__":
    main()
