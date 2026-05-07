"""
pci — Proximal Causal Inference library

A library for proximal causal inference with bridges, doubly-robust
estimators, and a blind tuning protocol. Submodules:

    pci.dgps        Data-generating processes for simulation studies.
    pci.estimators  PCI estimators (bridges + inference).
    pci.bridges     First-class bridge functions h(W,A) and q(Z,A,X).
    pci.inference   Cross-fitting, variance decomposition.
    pci.tuning      Blind tuning protocol (Bennett-RFF and DRKPV variants).
    pci.metrics     Six canonical Monte-Carlo metrics
                    (bias, variance, RMSE, coverage, SER, cv_var).
    pci.runner      Monte-Carlo orchestration utilities (parallel runner).

For notation conventions (W = negative-control outcome, Z = negative-control
exposure, GACE = generalized average causal effect, etc.) see :mod:`pci.notation`.
"""
from __future__ import annotations

__version__ = "0.1.0"
