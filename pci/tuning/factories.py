"""
pci.tuning.factories
====================

Estimator factories used by the blind tuning protocol. Each factory takes a
configuration dict (hyperparameters) and returns a fitted-ready estimator
instance. Defaults match the Phase 14B winner (BIN_mh2000) for Bennett and
the Phase 17 cfg2 winner for DRKPV.

Module exports:

* :data:`BENNETT_DEFAULTS`     Bennett-RFF DR hyperparameters
* :data:`DRKPV_DEFAULTS`       DRKPV hyperparameters
* :func:`make_bennett_estimator`  ``BennettIndepFunctionalDR`` factory
* :func:`make_drkpv_estimator`    ``DRKernel + KPVPolicyBridgeQ`` factory
* :data:`FACTORIES`            ``{"bennett": ..., "drkpv": ...}``
* :data:`DEFAULTS`             ``{"bennett": BENNETT_DEFAULTS, ...}``

The factories accept ``a_grid`` and ``ref_dose_index`` as keyword arguments;
both default to the DGP2 simulation values used throughout the essay
(a_grid = [1.8, 2.2, 2.6, 3.0], ref_dose_index = 2). Real-data scripts pass
their own a_grid (e.g. NLSY79 schooling years) and ref_dose_index.

Extracted from
``simulations/experiments/simulation/blind_tuning_dgp2_bennett.py``. The original module continues to re-export these as a
backward-compatibility shim.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


# Default DGP2 a_grid and reference dose index (preserved from the original
# Phase 14B winner). Override by passing a_grid / ref_dose_index kwargs to
# the factory functions for non-DGP2 use cases (real-data, DGP1, etc.).
A_GRID_DGP2_DEFAULT = np.array([1.8, 2.2, 2.6, 3.0])
REF_IDX_DGP2_DEFAULT = 2


# Default config templates (BIN_mh2000 oracle-tuned Phase 14B baseline)
BENNETT_DEFAULTS = {
    "m_h": 2000, "m_c": 500,
    "ell_h": 2.75, "ell_c": 2.0,
    "lambda_h": 1e-5, "gamma_critic": 1e-4,
    "lambda_r": 1e-2, "n_features_r": 500, "ell_scale_r": 3.5,
}

DRKPV_DEFAULTS = {
    "lambda_h": 3e-5,
    "ell_scale": 3.5,
    "lambda_Q": 1e-3,
    "clip_factor": None,
}


def make_bennett_estimator(
    config: Dict,
    dgp,
    h_KDE: float,
    ell_W: float,
    ell_A: float,
    ell_Z: float,
    a_grid: Optional[Sequence[float]] = None,
    ref_dose_index: int = REF_IDX_DGP2_DEFAULT,
):
    """Build ``BennettIndepFunctionalDR`` from config dict.

    Parameters
    ----------
    config : dict
        Hyperparameter overrides (defaults from BENNETT_DEFAULTS).
    dgp : object
        Accepted for interface uniformity ; not used by Bennett (which is
        DGP-agnostic).
    h_KDE : float
        Policy KDE bandwidth.
    ell_W, ell_A, ell_Z : float
        Pre-computed length-scales (median heuristic) used as the reference
        scale for the inner ell_h/ell_c products.
    a_grid : sequence of float, optional
        Dose grid. Defaults to DGP2 ``[1.8, 2.2, 2.6, 3.0]``.
    ref_dose_index : int
        Index of the reference dose in a_grid. Defaults to 2.
    """
    from pci.estimators.bennett_indep_dr import BennettIndepFunctionalDR
    if a_grid is None:
        a_grid_list = A_GRID_DGP2_DEFAULT.tolist()
    else:
        a_grid_list = list(a_grid)
    return BennettIndepFunctionalDR(
        m_h=int(config.get("m_h", 2000)),
        m_c=int(config.get("m_c", 500)),
        ell_h=float(config.get("ell_h", 2.75)),
        ell_c=float(config.get("ell_c", 2.0)),
        lambda_h=float(config.get("lambda_h", 1e-5)),
        gamma_critic=float(config.get("gamma_critic", 1e-4)),
        lambda_r=float(config.get("lambda_r", 1e-2)),
        n_features_r=int(config.get("n_features_r", 500)),
        ell_scale_r=float(config.get("ell_scale_r", 3.5)),
        n_folds=5,
        a_grid=a_grid_list,
        bandwidth=h_KDE,
        ref_dose_index=ref_dose_index,
        seed=42,
    )


def make_drkpv_estimator(
    config: Dict,
    dgp,
    h_KDE: float,
    ell_W: float,
    ell_A: float,
    ell_Z: float,
    a_grid: Optional[np.ndarray] = None,
    ref_dose_index: int = REF_IDX_DGP2_DEFAULT,
):
    """Build ``DRKernel`` + ``KPVPolicyBridgeQ`` + ``KPVBridgeH`` from config dict."""
    from pci.estimators.dr_kernel import DRKernel
    from pci.bridges.kpv import KPVPolicyBridgeQ
    if a_grid is None:
        a_grid = A_GRID_DGP2_DEFAULT
    ell_scale = float(config.get("ell_scale", 3.5))
    n = 1000
    clip_factor = config.get("clip_factor", None)
    clip = (clip_factor * (n ** 0.25)) if clip_factor is not None else None
    q_model = KPVPolicyBridgeQ(
        a_grid=a_grid, h_KDE=h_KDE,
        lambda_Q=float(config.get("lambda_Q", 1e-3)),
        ell_W=ell_W * ell_scale,
        ell_A=ell_A * ell_scale,
        ell_Z=ell_Z * ell_scale,
        clip=clip,
        compute_cond=True,
    )
    return DRKernel(
        a_grid=a_grid, q_model=q_model, bandwidth=h_KDE,
        n_folds=5, random_state=42,
        ref_dose_index=ref_dose_index, cross_fit_q=True,
        bridge_kwargs=dict(
            lambda_1=float(config.get("lambda_h", 3e-5)),
            lambda_2=float(config.get("lambda_h", 3e-5)),
            ell_W=ell_W * ell_scale,
            ell_A=ell_A * ell_scale,
            ell_Z=ell_Z * ell_scale,
        ),
    )


FACTORIES = {"bennett": make_bennett_estimator, "drkpv": make_drkpv_estimator}
DEFAULTS = {"bennett": BENNETT_DEFAULTS, "drkpv": DRKPV_DEFAULTS}
