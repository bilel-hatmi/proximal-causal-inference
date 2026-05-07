"""
pci.tuning.blind_score
======================

Blind tuning composite scores used in S6 (DGP2 head-to-head) and S7
(real-data application of the Phase 18 Bennett protocol).

Two scores:

* :func:`bennett_blind_score_v2`  Phase 18 (corrected with `lambda_h x m_h` gate)
* :func:`drkpv_blind_score`        Phase 17 (unchanged)

Plus a dispatch dict :data:`SCORING` mapping method name to score function.

Both scores accept ``n`` and ``K`` for interface uniformity but neither uses
them (they are pure functions of the bridge diagnostics dict).

Extracted from
``simulations/experiments/simulation/blind_tuning_dgp2_bennett.py``. The original module continues to re-export these as a
backward-compatibility shim.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def bennett_blind_score_v2(diags: Dict[str, float], n: int = 1000, K: int = 5) -> float:
    """Composite blind score for Bennett (Phase 18 corrected). Lower is better.

    Changes vs Phase 17 ``bennett_blind_score`` :

    * **Gate 1** (NEW): ``lambda_h * m_h < 0.01`` -> hard fail (interpolation
      protection). Uses m_h (feature space size), not n_fold, so the gate is
      n-independent.
        - BIN_mh2000: lambda_h=1e-5, m_h=2000 -> 0.020 > 0.01 PASS
        - Phase17_blind: lambda_h=1e-6, m_h=2000 -> 0.002 < 0.01 FAIL
    * **Gate 2**: kappa_h hard fail lowered 1e10 -> 1e8; penalty starts at
      1e5 (was 1e6).

    Parameters
    ----------
    diags : Dict[str, float]
        Output of :func:`pci.tuning.extract_diagnostics.extract_diagnostics`.
        MUST contain 'lambda_h' and 'm_h' (injected from the config params).
    n, K : int
        Accepted for interface uniformity, not used in the gates.
    """
    # Gate 1: interpolation protection (NEW in Phase 18)
    lambda_h = diags.get("lambda_h", None)
    m_h = diags.get("m_h", 2000)
    if lambda_h is not None:
        if lambda_h * m_h < 0.01:
            return float("inf")    # Interpolation zone: residual_norm_h -> 0 misleading

    # Gate 2: conditioning (hard fail lowered 1e10 -> 1e8)
    kappa_h = diags.get("kappa_h", float("nan"))
    kappa_r = diags.get("kappa_r", float("nan"))
    if not np.isfinite(kappa_h) or kappa_h > 1e8:
        return float("inf")
    if not np.isfinite(kappa_r) or kappa_r > 1e10:
        return float("inf")

    # Gate 3: Riesz residual
    rr_ref = diags.get("RR_ref", float("nan"))
    if not np.isfinite(rr_ref) or rr_ref > 1e-1:
        return float("inf")

    # Gate 4: residual_norm_h must be finite
    residual_norm_h = diags.get("residual_norm_h", float("nan"))
    if not np.isfinite(residual_norm_h):
        return float("inf")

    eff_rank_h = diags.get("eff_rank_h", float("nan"))
    eff_rank_r = diags.get("eff_rank_r", float("nan"))

    # Composite (weights ~ |rho| from Phase 16 audit, 22 cells)
    s_residual_h = residual_norm_h / 0.005              # target < 0.005
    s_eff_rank_r = max(0, 50 - eff_rank_r) / 50         # target >= 50
    s_eff_rank_h = max(0, 50 - eff_rank_h) / 50
    s_RR         = rr_ref / 0.005                       # target < 0.005
    # kappa_h penalty starts at 1e5 (BIN_mh2000 has kappa_h ~ 87K)
    s_kappa_h    = max(0, np.log10(kappa_h) - 5) / 3    # 1e5 -> 0, 1e8 -> 1
    s_kappa_r    = max(0, np.log10(kappa_r) - 4) / 4

    score = (
        0.30 * s_residual_h +
        0.20 * s_eff_rank_r +
        0.15 * s_eff_rank_h +
        0.15 * s_RR +
        0.10 * s_kappa_h +
        0.10 * s_kappa_r
    )
    return float(score)


def drkpv_blind_score(diags: Dict[str, float], n: int = 1000, K: int = 5) -> float:
    """Composite blind score for DRKPV (unchanged from Phase 17). Lower is better.

    Parameters
    ----------
    diags : Dict[str, float]
        Output of :func:`extract_diagnostics`.
    n, K : int
        Accepted for interface uniformity but not used (DRKPV does not have
        the h-bridge interpolation issue).
    """
    kappa_r = diags.get("kappa_r", float("nan"))
    rr_ref = diags.get("RR_ref", float("nan"))
    residual_norm_r = diags.get("residual_norm_r", float("nan"))
    SER = diags.get("SER", float("nan"))

    if not np.isfinite(rr_ref) or rr_ref > 1e-1:
        return float("inf")
    if np.isfinite(residual_norm_r) and residual_norm_r > 0.5:
        return float("inf")

    s_RR = rr_ref / 0.005
    s_residual_r = (residual_norm_r if np.isfinite(residual_norm_r) else 1.0) / 0.05

    if np.isfinite(kappa_r) and kappa_r > 0:
        s_kappa_r = max(0, (np.log10(kappa_r) - 30) / 40)
    else:
        s_kappa_r = 1.0

    if np.isfinite(SER):
        s_SER = abs(SER - 1.0) / 0.5
    else:
        s_SER = 0.0

    return float(0.30 * s_RR + 0.20 * s_residual_r + 0.20 * s_kappa_r + 0.30 * s_SER)


SCORING = {"bennett": bennett_blind_score_v2, "drkpv": drkpv_blind_score}
