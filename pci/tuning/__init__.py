"""pci.tuning — blind tuning protocol used in S6 (DGP2 head-to-head) and S7
(real-data applications).

Currently empty. The tuning protocol implementation lives in
``simulations/experiments/phase{17,18}_blind_tuning.py`` for historical
reasons; these scripts will be renamed under ``experiments/simulation/`` in
the refactor (see plan file). Once renamed, any genuinely
shared machinery (``AdaptiveTuner``, ``extract_diagnostics``,
``bennett_blind_score_v2``, ``drkpv_blind_score``) can be promoted from those
scripts to canonical modules in this subpackage.

Until then, downstream code that needs the tuning machinery should still
import from ``simulations.experiments.phase18_blind_tuning`` (Bennett) or
``simulations.experiments.phase17_blind_tuning`` (DRKPV).
"""
