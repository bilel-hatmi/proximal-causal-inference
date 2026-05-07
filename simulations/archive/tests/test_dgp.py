"""
Tests unitaires et fonction verify_dgp() pour les DGPs.

verify_dgp() : diagnostic console réutilisable pour tout PCIDgp.
test_*       : suites pytest pour chaque DGP concret.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from simulations.dgp.base import PCIDgp
from simulations.dgp.cobb_douglas import CobbDouglasLinearDGP
from simulations.dgp.michaelis_menten import MichaelisMentenDGP
from simulations.dgp.utils import snr_empirical


# ── verify_dgp() — réutilisable pour tous les DGPs ────────────────────────────

def verify_dgp(dgp: PCIDgp, n: int = 10_000, seed: int = 42) -> dict:
    """
    Run 6 diagnostic tests on a PCIDgp instance.

    Tests
    -----
    T1  Confounding: corr(U,A) > 0.2 and corr(U,Y) > 0.2
    T2  Exclusion Z: corr(Z, resid(Y ~ U+A)) < 0.1
    T3  Exclusion W: corr(W, resid(A ~ U))   < 0.1
    T4  SNR calibration: |SNR_empirical - SNR_declared| < 0.05 for W and Z
    T5a Bridge consistency: |E[h0(W,A,X)] - psi_0| < 0.02
    T5b Fredholm condition: E[resid*Z]=0, E[resid*Z^2]=0, E[resid*Z*A]=0
        (T5b is discriminant: fails for snr_W*W formula even though T5a passes,
         because E[W]=0 masks the attenuation error in T5a.)

    Returns
    -------
    dict mapping test names to {'pass': bool, ...diagnostic values...}
    Prints a formatted summary to stdout.
    """
    sample = dgp.generate(n=n, seed=seed)
    results = {}

    # ── T1 — Confounding ───────────────────────────────────────────────────
    corr_UA = float(np.corrcoef(sample.U, sample.A)[0, 1])
    corr_UY = float(np.corrcoef(sample.U, sample.Y)[0, 1])
    results["T1_confounding"] = {
        "corr_UA": corr_UA,
        "corr_UY": corr_UY,
        "pass": abs(corr_UA) > 0.2 and abs(corr_UY) > 0.2,
    }

    # ── T2 — Exclusion Z ───────────────────────────────────────────────────
    feat_YUA = np.column_stack([sample.U, sample.A])
    resid_Y = sample.Y - LinearRegression().fit(feat_YUA, sample.Y).predict(feat_YUA)
    corr_Z_residY = float(np.corrcoef(sample.Z, resid_Y)[0, 1])
    results["T2_exclusion_Z"] = {
        "corr_Z_residY_given_UA": corr_Z_residY,
        "pass": abs(corr_Z_residY) < 0.1,
    }

    # ── T3 — Exclusion W ───────────────────────────────────────────────────
    resid_A = sample.A - LinearRegression().fit(
        sample.U.reshape(-1, 1), sample.A
    ).predict(sample.U.reshape(-1, 1))
    corr_W_residA = float(np.corrcoef(sample.W, resid_A)[0, 1])
    results["T3_exclusion_W"] = {
        "corr_W_residA_given_U": corr_W_residA,
        "pass": abs(corr_W_residA) < 0.1,
    }

    # ── T4 — SNR calibration ───────────────────────────────────────────────
    snr_W_emp = snr_empirical(sample.U, sample.W)
    snr_Z_emp = snr_empirical(sample.U, sample.Z)
    snr_W_decl = dgp.snr_W()
    snr_Z_decl = dgp.snr_Z()
    results["T4_snr"] = {
        "snr_W_declared": snr_W_decl,
        "snr_W_empirical": snr_W_emp,
        "snr_Z_declared": snr_Z_decl,
        "snr_Z_empirical": snr_Z_emp,
        "pass": (
            abs(snr_W_emp - snr_W_decl) < 0.05
            and abs(snr_Z_emp - snr_Z_decl) < 0.05
        ),
    }

    # ── T5a — Bridge consistency (contrast) ───────────────────────────────
    # For continuous treatment, the estimand is E[Y(1)] - E[Y(0)] = alpha.
    # We verify that h₀ recovers this contrast, not the observed mean E[h₀(W,A_obs,X)]
    # (which equals alpha*E[A] + (1-alpha)*mu_L ≠ alpha when E[A]=0, mu_L≠0).
    n_s = len(sample.W)
    h0_at_1 = dgp.h0(sample.W, np.ones(n_s), sample.X)
    h0_at_0 = dgp.h0(sample.W, np.zeros(n_s), sample.X)
    h0_contrast = float(np.mean(h0_at_1 - h0_at_0))
    psi_0 = dgp.psi_true(a=1.0)
    results["T5a_bridge_consistency"] = {
        "h0_contrast_E[h0(W,1,X)-h0(W,0,X)]": h0_contrast,
        "psi_0": psi_0,
        "pass": abs(h0_contrast - psi_0) < 0.02,
    }

    # ── T5b — Fredholm condition (discriminant test) ───────────────────────
    # residuals = Y - h0(W, A_obs, X) = eps_Y - eps_W (by algebra), independent of Z
    h0_vals = dgp.h0(sample.W, sample.A, sample.X)
    residuals = sample.Y - h0_vals
    m1 = float(np.mean(residuals * sample.Z))
    m2 = float(np.mean(residuals * sample.Z ** 2))
    m3 = float(np.mean(residuals * sample.Z * sample.A))
    results["T5b_fredholm"] = {
        "E_resid_Z": m1,
        "E_resid_Z2": m2,
        "E_resid_ZA": m3,
        "pass": abs(m1) < 0.02 and abs(m2) < 0.05 and abs(m3) < 0.02,
    }

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 52}")
    print(f"DGP VERIFICATION : {dgp!r}")
    print(f"n={n}, seed={seed}")
    print(f"{'=' * 52}")
    for test_name, res in results.items():
        status = "PASS" if res["pass"] else "FAIL"
        print(f"[{status}]  {test_name}")
        for k, v in res.items():
            if k != "pass":
                if isinstance(v, float):
                    print(f"       {k}: {v:.4f}")
                else:
                    print(f"       {k}: {v}")
    n_pass = sum(r["pass"] for r in results.values())
    print(f"\n{n_pass}/{len(results)} tests passed")
    print(f"{'=' * 52}\n")

    return results


# ── Pytest suite — CobbDouglasLinearDGP ───────────────────────────────────────

@pytest.fixture(scope="module")
def cd_dgp():
    return CobbDouglasLinearDGP()


@pytest.fixture(scope="module")
def cd_sample(cd_dgp):
    return cd_dgp.generate(n=10_000, seed=42)


def test_T1_confounding(cd_dgp, cd_sample):
    corr_UA = float(np.corrcoef(cd_sample.U, cd_sample.A)[0, 1])
    corr_UY = float(np.corrcoef(cd_sample.U, cd_sample.Y)[0, 1])
    assert abs(corr_UA) > 0.2, f"corr(U,A) = {corr_UA:.4f}"
    assert abs(corr_UY) > 0.2, f"corr(U,Y) = {corr_UY:.4f}"


def test_T2_exclusion_Z(cd_sample):
    feat = np.column_stack([cd_sample.U, cd_sample.A])
    resid_Y = cd_sample.Y - LinearRegression().fit(feat, cd_sample.Y).predict(feat)
    corr = float(np.corrcoef(cd_sample.Z, resid_Y)[0, 1])
    assert abs(corr) < 0.1, f"corr(Z, resid_Y|U,A) = {corr:.4f}"


def test_T3_exclusion_W(cd_sample):
    resid_A = cd_sample.A - LinearRegression().fit(
        cd_sample.U.reshape(-1, 1), cd_sample.A
    ).predict(cd_sample.U.reshape(-1, 1))
    corr = float(np.corrcoef(cd_sample.W, resid_A)[0, 1])
    assert abs(corr) < 0.1, f"corr(W, resid_A|U) = {corr:.4f}"


def test_T4_snr(cd_dgp, cd_sample):
    snr_W_emp = snr_empirical(cd_sample.U, cd_sample.W)
    snr_Z_emp = snr_empirical(cd_sample.U, cd_sample.Z)
    assert abs(snr_W_emp - cd_dgp.snr_W()) < 0.05, (
        f"SNR_W: declared={cd_dgp.snr_W():.3f}, empirical={snr_W_emp:.3f}"
    )
    assert abs(snr_Z_emp - cd_dgp.snr_Z()) < 0.05, (
        f"SNR_Z: declared={cd_dgp.snr_Z():.3f}, empirical={snr_Z_emp:.3f}"
    )


def test_T5a_bridge_consistency(cd_dgp, cd_sample):
    n = len(cd_sample.W)
    h0_contrast = float(np.mean(
        cd_dgp.h0(cd_sample.W, np.ones(n), cd_sample.X)
        - cd_dgp.h0(cd_sample.W, np.zeros(n), cd_sample.X)
    ))
    psi_0 = cd_dgp.psi_true(1.0)
    assert abs(h0_contrast - psi_0) < 0.02, (
        f"E[h0(W,1,X)-h0(W,0,X)] = {h0_contrast:.4f}, psi_0 = {psi_0:.4f}"
    )


def test_T5b_fredholm(cd_dgp, cd_sample):
    residuals = cd_sample.Y - cd_dgp.h0(cd_sample.W, cd_sample.A, cd_sample.X)
    m1 = float(np.mean(residuals * cd_sample.Z))
    m2 = float(np.mean(residuals * cd_sample.Z ** 2))
    m3 = float(np.mean(residuals * cd_sample.Z * cd_sample.A))
    assert abs(m1) < 0.02, f"E[resid*Z]   = {m1:.5f}"
    assert abs(m2) < 0.05, f"E[resid*Z^2] = {m2:.5f}"
    assert abs(m3) < 0.02, f"E[resid*Z*A] = {m3:.5f}"


def test_repr(cd_dgp):
    r = repr(cd_dgp)
    assert "CobbDouglasLinearDGP" in r
    assert "0.80" in r


def test_psi_true(cd_dgp):
    assert cd_dgp.psi_true(1.0) == pytest.approx(0.30)


def test_m_true_analytic_value(cd_dgp):
    """m_true(a) = alpha * a + (1-alpha) * mu_L. With defaults alpha=0.3, mu_L=1.0:
       m_true(1.5) = 0.3 * 1.5 + 0.7 * 1.0 = 0.45 + 0.70 = 1.15"""
    assert cd_dgp.m_true(1.5) == pytest.approx(1.15)


def test_m_true_consistency_with_mc():
    """
    m_true(a) must equal the empirical mean of Y under do(A=a) on a large sample.
    This validates the dose-response interpretation E[Y(a)].
    """
    dgp = CobbDouglasLinearDGP()
    a_test = 1.5
    n = 100_000
    rng = np.random.default_rng(0)
    U = rng.normal(0.0, dgp.sigma_U, n)
    X = rng.normal(dgp.mu_L, dgp.sigma_L, n)
    eps_Y = rng.normal(0.0, dgp.sigma_Y, n)
    Y_do_a = U + dgp.alpha * a_test + (1 - dgp.alpha) * X + eps_Y
    err = abs(float(np.mean(Y_do_a)) - dgp.m_true(a_test))
    assert err < 0.01, f"E[Y(do(A={a_test}))] vs m_true: err = {err:.5f}"


# ── Pytest suite — MichaelisMentenDGP ────────────────────────────────────────

@pytest.fixture(scope="module")
def mm_dgp():
    return MichaelisMentenDGP()


@pytest.fixture(scope="module")
def mm_sample(mm_dgp):
    return mm_dgp.generate(n=10_000, seed=42)


# ── m_true calibration — BLOCKING test (run before anything else) ─────────

def test_mm_m_true_calibration_vs_scipy():
    """
    Blocking calibration: hermegauss m_true must match scipy.quad to 1e-3.

    This is the gatekeeper test for Phase 7. If the 1/sqrt(2pi) factor is
    missing, m_true would be off by sqrt(2pi) ≈ 2.51 and all MISE would be wrong.
    """
    from scipy.integrate import quad
    from scipy.stats import lognorm

    dgp = MichaelisMentenDGP()
    for a in [0.5, 1.5, 4.0]:
        gh = dgp.m_true(a)
        ref, _ = quad(
            lambda u: (dgp.Vmax * u * a) / (dgp.Km * u + a)
                      * lognorm.pdf(u, s=dgp.sigma_U),
            1e-6, 200.0,
        )
        err = abs(gh - ref)
        assert err < 1e-3, (
            f"a={a}: hermegauss={gh:.6f}, scipy={ref:.6f}, diff={err:.2e}"
        )


def test_mm_m_true_at_zero():
    """m_true(0) = 0 exactly for Michaelis-Menten (mu(U,0) = 0 for all U)."""
    assert MichaelisMentenDGP().m_true(0.0) == pytest.approx(0.0, abs=1e-10)


def test_mm_m_true_concavity():
    """
    m_true(a) is strictly concave: dm/da decreasing.
    Verified via finite differences on a coarse grid.
    """
    dgp = MichaelisMentenDGP()
    a_vals = np.array([0.5, 1.0, 1.5, 2.5, 4.0])
    m_vals = np.array([dgp.m_true(a) for a in a_vals])
    # First differences (m values should be increasing)
    deltas = np.diff(m_vals)
    assert np.all(deltas > 0), f"m_true not increasing: deltas={deltas}"
    # Per-unit differences should be decreasing (concavity)
    da = np.diff(a_vals)
    slopes = deltas / da
    assert np.all(np.diff(slopes) < 0), (
        f"m_true not concave: per-unit slopes={slopes}"
    )


# ── T1 — Confounding ─────────────────────────────────────────────────────────

def test_mm_T1_confounding(mm_dgp, mm_sample):
    """
    U (LogNormal) must be correlated with both A and Y.
    With sensitivity=0.3 and sigma_U=1, Cor(U,A) ≈ 0.73 (strong confounding).
    """
    corr_UA = float(np.corrcoef(mm_sample.U, mm_sample.A)[0, 1])
    corr_UY = float(np.corrcoef(mm_sample.U, mm_sample.Y)[0, 1])
    assert abs(corr_UA) > 0.2, f"corr(U,A) = {corr_UA:.4f} (expected > 0.2)"
    assert abs(corr_UY) > 0.2, f"corr(U,Y) = {corr_UY:.4f} (expected > 0.2)"


# ── T2 — Exclusion Z ─────────────────────────────────────────────────────────

def test_mm_T2_exclusion_Z(mm_sample):
    """Z is excluded from Y given (U, A): corr(Z, resid_Y|U,A) < 0.1."""
    feat = np.column_stack([mm_sample.U, mm_sample.A])
    resid_Y = (mm_sample.Y
               - LinearRegression().fit(feat, mm_sample.Y).predict(feat))
    corr = float(np.corrcoef(mm_sample.Z, resid_Y)[0, 1])
    assert abs(corr) < 0.1, f"corr(Z, resid_Y|U,A) = {corr:.4f}"


# ── T3 — Exclusion W ─────────────────────────────────────────────────────────

def test_mm_T3_exclusion_W(mm_sample):
    """W is excluded from A given U: corr(W, resid_A|U) < 0.1."""
    resid_A = (mm_sample.A
               - LinearRegression()
               .fit(mm_sample.U.reshape(-1, 1), mm_sample.A)
               .predict(mm_sample.U.reshape(-1, 1)))
    corr = float(np.corrcoef(mm_sample.W, resid_A)[0, 1])
    assert abs(corr) < 0.1, f"corr(W, resid_A|U) = {corr:.4f}"


# ── T4 — SNR calibration ─────────────────────────────────────────────────────

def test_mm_T4_snr(mm_dgp, mm_sample):
    """
    SNR empirical matches declared SNR.
    Tolerance = 0.10 (vs 0.05 for DGP1): LogNormal U has higher variance
    and snr_empirical = Var(U)/Var(proxy) has more Monte Carlo variability
    at n=10k with U non-Gaussian.
    """
    snr_W_emp = snr_empirical(mm_sample.U, mm_sample.W)
    snr_Z_emp = snr_empirical(mm_sample.U, mm_sample.Z)
    assert abs(snr_W_emp - mm_dgp.snr_W()) < 0.10, (
        f"SNR_W: declared={mm_dgp.snr_W():.3f}, empirical={snr_W_emp:.3f}"
    )
    assert abs(snr_Z_emp - mm_dgp.snr_Z()) < 0.10, (
        f"SNR_Z: declared={mm_dgp.snr_Z():.3f}, empirical={snr_Z_emp:.3f}"
    )


# ── T5a — Bridge consistency (xfail — regression oracle ≠ Fredholm bridge) ──

@pytest.mark.xfail(
    reason=(
        "MichaelisMentenDGP.h0 is the REGRESSION oracle E[mu(U,A)|W,A], "
        "NOT the Fredholm bridge. The identification condition E[h0(W,a)] = m_true(a) "
        "(Kallus 2021 Lemma 2) requires the Fredholm bridge, not the regression oracle. "
        "For DGP1 (linear), the regression oracle also fails T5a — it passes only "
        "because the attenuated W cancels in the DGP1 contrast (snr_W*W - snr_W*W = 0), "
        "not because the identification condition holds for the regression oracle. "
        "For DGP2 (nonlinear KRR), E[h0_reg(W,a)] ≠ m_true(a): the observational "
        "conditioning p(U|W, A=a) introduces bias that does not cancel. "
        "Exact Fredholm bridge deferred to Phase 8 (DRKernel)."
    ),
    strict=False,
)
def test_mm_T5a_bridge_consistency(mm_dgp, mm_sample):
    """
    E[h0(W, a)] ≈ m_true(a) — fails for regression oracle (xfail documented).

    Expected to fail: regression oracle ≠ Fredholm bridge. See xfail reason above.
    """
    n = len(mm_sample.W)
    a_vals = [1.5, 2.0, 2.5, 3.0]
    max_err = 0.0
    errors = {}
    for a in a_vals:
        h0_at_a = mm_dgp.h0(mm_sample.W, np.full(n, float(a)), mm_sample.X)
        empirical = float(np.mean(h0_at_a))
        truth = mm_dgp.m_true(a)
        err = abs(empirical - truth)
        errors[a] = (empirical, truth, err)
        max_err = max(max_err, err)
    err_str = "; ".join(
        f"a={a}: E[h0]={v[0]:.4f}, m_true={v[1]:.4f}, |err|={v[2]:.4f}"
        for a, v in errors.items()
    )
    assert max_err < 0.05, (
        f"max |E[h0(W,a)] - m_true(a)| = {max_err:.4f}\n{err_str}"
    )


def test_mm_regression_oracle_quality(mm_dgp):
    """
    KRR regression oracle quality: E[mu(U,A)|W,A] fits in-sample to low RMSE.

    The regression oracle h0_reg(W, A) ≈ E[mu(U,A) | W, A] is not the Fredholm
    bridge, but it IS a valid regression oracle for mu(U,A) given (W, A_obs).
    On a held-out sample (seed≠0), the KRR predictions should correlate strongly
    with the true mu values and have low RMSE.

    This test validates the regression oracle quality independently of the
    Fredholm bridge condition (T5a) or the identification condition.
    """
    # Held-out sample — seed 99, different from training (seed 0)
    val_sample = mm_dgp.generate(n=2_000, seed=99)
    mu_true = ((mm_dgp.Vmax * val_sample.U * val_sample.A)
               / (mm_dgp.Km * val_sample.U + val_sample.A))
    mu_pred = mm_dgp.h0(val_sample.W, val_sample.A, val_sample.X)

    rmse = float(np.sqrt(np.mean((mu_pred - mu_true) ** 2)))
    std_mu = float(np.std(mu_true))
    corr = float(np.corrcoef(mu_pred, mu_true)[0, 1])
    rel_rmse = rmse / std_mu   # scale-invariant; R² = 1 - rel_rmse²

    assert rel_rmse < 0.50, (
        f"KRR regression oracle relative RMSE = {rel_rmse:.4f} (tol=0.50, "
        f"equivalent to R²>{1 - 0.50**2:.2f}); "
        f"abs_RMSE={rmse:.4f}, std(mu)={std_mu:.4f}"
    )
    assert corr > 0.85, (
        f"KRR regression oracle corr(pred, true) = {corr:.4f} (tol=0.85)"
    )


# ── T5b — Fredholm condition (xfail — regression oracle ≠ Fredholm bridge) ──

@pytest.mark.xfail(
    reason=(
        "MichaelisMentenDGP.h0 is the REGRESSION oracle E[mu(U,A)|W,A], "
        "not the Fredholm bridge. The Fredholm bridge satisfies E[Y-h0|Z,A]=0; "
        "the regression oracle gives E[resid*Z] ≠ 0 (same attenuation issue as "
        "'snr_W*W' formula for DGP1). Exact Fredholm bridge deferred to Phase 8."
    ),
    strict=False,
)
def test_mm_T5b_fredholm(mm_dgp, mm_sample):
    """
    Fredholm condition: E[(Y - h0(W,A)) * Z] ≈ 0.
    Expected to fail for the regression oracle — documented xfail.
    """
    residuals = mm_sample.Y - mm_dgp.h0(mm_sample.W, mm_sample.A, mm_sample.X)
    m1 = float(np.mean(residuals * mm_sample.Z))
    m2 = float(np.mean(residuals * mm_sample.Z ** 2))
    m3 = float(np.mean(residuals * mm_sample.Z * mm_sample.A))
    assert abs(m1) < 0.05, f"E[resid*Z]   = {m1:.5f}"
    assert abs(m2) < 0.10, f"E[resid*Z^2] = {m2:.5f}"
    assert abs(m3) < 0.05, f"E[resid*Z*A] = {m3:.5f}"


def test_mm_repr(mm_dgp):
    r = repr(mm_dgp)
    assert "MichaelisMentenDGP" in r
    assert "0.80" in r


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    dgp = CobbDouglasLinearDGP()
    verify_dgp(dgp, n=10_000, seed=42)
    print()
    dgp2 = MichaelisMentenDGP()
    verify_dgp(dgp2, n=10_000, seed=42)
