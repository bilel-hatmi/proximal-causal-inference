"""
Tests unitaires pour le harnais Monte Carlo.

Deux niveaux :
  - Tests rapides (CI)   : n=200, M=20, <10 secondes
  - Test lent (@slow)    : pilote complet n=1000 M=200, skip par défaut

Lancer les tests rapides :
    python -m pytest simulations/tests/test_harness.py -v -m "not slow"

Lancer le pilote (validation roadmap Phase 4) :
    python -m pytest simulations/tests/test_harness.py -v -m slow
    # ou directement :
    python -c "from simulations.experiments.harness import run_pilot; run_pilot()"
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP
from simulations.experiments.harness import run_monte_carlo, run_pilot
from simulations.methods.oracle import OracleDirect


# ── Fixture : petite run partagée par tous les tests rapides ──────────────────

@pytest.fixture(scope="module")
def small_run():
    """
    n=200, M=20 — shape/type checks only.
    Saved to a temp dir that is discarded after the test session.
    """
    dgp = CobbDouglasLinearDGP()
    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "test_run.pkl"
        df = run_monte_carlo(
            dgp, OracleDirect,
            n=200, M=20, seed_base=0,
            save_path=save_path,
        )
    return df  # df survit à la destruction du répertoire temporaire


# ── Tests rapides ─────────────────────────────────────────────────────────────

def test_harness_returns_dataframe(small_run):
    """run_monte_carlo() doit renvoyer un DataFrame."""
    assert isinstance(small_run, pd.DataFrame)


def test_harness_shape(small_run):
    """DataFrame doit avoir M lignes et 3 colonnes."""
    assert small_run.shape == (20, 3)


def test_harness_columns(small_run):
    """Colonnes exactes : psi_hat, V_hat, seed."""
    assert set(small_run.columns) == {"psi_hat", "V_hat", "seed"}


def test_harness_no_nan(small_run):
    """Aucune valeur NaN dans le DataFrame."""
    assert not small_run.isnull().any().any()


def test_harness_seeds_unique(small_run):
    """Chaque réplication utilise un seed distinct."""
    assert small_run["seed"].nunique() == 20


def test_harness_seeds_consecutive(small_run):
    """Seeds = seed_base + i  →  [0, 1, ..., M-1]."""
    assert sorted(small_run["seed"].tolist()) == list(range(20))


def test_harness_vhat_positive(small_run):
    """V_hat doit être strictement positif."""
    assert (small_run["V_hat"] > 0).all()


def test_harness_pkl_saved(tmp_path):
    """Le pkl doit être créé sur disque, non vide."""
    dgp = CobbDouglasLinearDGP()
    p = tmp_path / "check.pkl"
    run_monte_carlo(dgp, OracleDirect, n=100, M=5, seed_base=99, save_path=p)
    assert p.exists(), "pkl file not created"
    assert p.stat().st_size > 0, "pkl file is empty"


def test_harness_pkl_readable(tmp_path):
    """Le pkl doit pouvoir être relu en DataFrame identique."""
    dgp = CobbDouglasLinearDGP()
    p = tmp_path / "readable.pkl"
    df_orig = run_monte_carlo(dgp, OracleDirect, n=100, M=5, seed_base=7, save_path=p)
    df_reloaded = pd.read_pickle(p)
    pd.testing.assert_frame_equal(df_orig, df_reloaded)


def test_harness_reproducibility():
    """Même seed_base → mêmes psi_hat."""
    dgp = CobbDouglasLinearDGP()
    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "run1.pkl"
        p2 = Path(tmp) / "run2.pkl"
        df1 = run_monte_carlo(dgp, OracleDirect, n=200, M=10, seed_base=42, save_path=p1)
        df2 = run_monte_carlo(dgp, OracleDirect, n=200, M=10, seed_base=42, save_path=p2)
    pd.testing.assert_frame_equal(df1, df2)


# ── Test lent — pilote complet (validation roadmap Phase 4) ──────────────────

@pytest.mark.slow
def test_pilot_coherence():
    """
    Pilote complet n=1000, M=200.
    Valide les deux critères de sortie de la Phase 4 :
      C1 : |bias_oracle| < |bias_naive| × 0.1
      C2 : cv_var_oracle < 0.3
    """
    results = run_pilot(n=1_000, M=200, seed_base=0)
    m_o = results["oracle"]
    m_n = results["naive"]

    bias_ratio = abs(m_o["bias"]) / (abs(m_n["bias"]) + 1e-9)
    assert bias_ratio < 0.1, (
        f"C1 failed: |bias_oracle|/|bias_naive| = {bias_ratio:.4f}  "
        f"(oracle={m_o['bias']:+.4f}, naive={m_n['bias']:+.4f})"
    )
    assert m_o["cv_var"] < 0.3, (
        f"C2 failed: cv_var_oracle = {m_o['cv_var']:.4f}  (target < 0.3)"
    )
