"""
pci.inference.crossfit
======================

Cross-fitting helpers shared by the three DR estimators
(``BennettIndepFunctionalDR``, ``BennettFunctionalDR``, ``DRKernel``).

Design note
-----------
The three DR estimators use cross-fitting with **the same K-fold split**
but with bridge-construction logic that diverges in API and diagnostics
(Bennett uses ``h_bridge.plug_in_policy`` + spectral diags, DRKernel
uses ``h_bridge.predict + q_model.predict + ESS``). A unifying
``CrossFitEstimator`` base class would force an awkward common
interface; instead, this module exposes the **fold-split helper**, which
each estimator already calls inline, plus a deterministic per-fold seed
helper.

Because the helper uses ``sklearn.model_selection.KFold(n_splits, shuffle,
random_state)`` exactly as the inline code does, swapping the inline
``KFold(...).split(indices)`` call for ``make_kfold_split(n, K, seed)``
produces **bit-identical** train/test partitions. the each estimators
estimator one at a time without observable behaviour change.

Functions
---------
make_kfold_split(n_samples, n_folds, seed) -> generator
    Yields ``(fold_idx, train_idx, test_idx)`` tuples. Bit-identical to
    ``enumerate(KFold(n_folds, shuffle=True, random_state=seed).split(np.arange(n_samples)))``.

per_fold_seed(base_seed, fold_idx, multiplier=1009) -> int
    Deterministic per-fold seed used by the Bennett estimators
    (``h_seed_base + fold_idx * 1009``). Centralised here so future
    estimators stay consistent.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import KFold


def make_kfold_split(
    n_samples: int,
    n_folds: int,
    seed: int,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(fold_idx, train_idx, test_idx)`` for a shuffled K-fold.

    Bit-identical to the inline pattern used in all three DR estimators:

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.arange(n_samples))):
            ...

    Parameters
    ----------
    n_samples : int
        Number of observations.
    n_folds : int
        Number of folds (typically 5 for DGP2 essais, 2 for fast tests).
    seed : int
        Random seed for the KFold shuffle. Same seed -> same partition
        across reruns and across the three estimators.
    """
    indices = np.arange(int(n_samples))
    kfold = KFold(n_splits=int(n_folds), shuffle=True, random_state=int(seed))
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(indices)):
        yield fold_idx, train_idx, test_idx


def per_fold_seed(base_seed: int, fold_idx: int, multiplier: int = 1009) -> int:
    """Deterministic per-fold seed used by the Bennett bridge constructors.

    The Bennett family builds an RFF h-bridge per fold with seed
    ``h_seed_base + fold_idx * 1009`` so the random Fourier features are
    statistically independent across folds (a primality multiplier
    spreads bits well across small fold_idx).

    Parameters
    ----------
    base_seed : int
        Base seed (e.g. ``self.h_seed_base`` in the estimator).
    fold_idx : int
        Zero-based fold index.
    multiplier : int, default 1009
        Multiplier; defaults to 1009 (prime), matching the inline code.
    """
    return int(base_seed) + int(fold_idx) * int(multiplier)
