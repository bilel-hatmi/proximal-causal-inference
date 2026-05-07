"""
Naive estimator — OLS ignoring the latent confounder U.

This is the theoretical floor: an estimator that pretends there is no
unmeasured confounding. Its bias quantifies the cost of ignoring U.

Model fitted:  Y ~ 1 + A + X  (OLS, no U)
Estimand:      coefficient of A  (biased toward alpha + confounding_bias)

Expected bias for CobbDouglasLinearDGP (defaults):
    confounding_bias = Cov(U, A) / Var(A)
                     = (gamma_K * sigma_U^2) / (gamma_K^2 * sigma_U^2 + sigma_K^2)
                     = (0.5 * 1.0) / (0.25 + 0.25)  =  1.0
    psi_hat_naive  ~  alpha + 1.0  =  1.30
"""

from typing import Optional

import numpy as np
import statsmodels.api as sm

from pci.dgps.base import DGPSample
from pci.estimators.base import EstimationResult, PCIEstimator


class NaiveRegression(PCIEstimator):
    """
    Naive estimator: OLS regression of Y on (A, X), ignoring U.

    Intentionally biased. Used as the reference floor in simulations
    (decision D6 / principle P3 of SKILL.md).

    Note: verify_method() check M3 (coverage plausible) will FAIL by design
    for this estimator — that is the expected behaviour, not a bug.
    """

    def __init__(self) -> None:
        self._result: Optional[EstimationResult] = None

    @property
    def name(self) -> str:
        return "NaiveRegression"

    def fit(self, sample: DGPSample) -> 'NaiveRegression':
        n = len(sample.Y)

        # Design matrix: [const, A, X]  — no U
        if sample.X is not None:
            X_design = sm.add_constant(
                np.column_stack([sample.A, sample.X]),
                has_constant='add'
            )
        else:
            X_design = sm.add_constant(
                sample.A.reshape(-1, 1),
                has_constant='add'
            )

        ols = sm.OLS(sample.Y, X_design).fit()

        # Column layout: [const=0, A=1, X=2]
        psi_hat = float(ols.params[1])
        se_A    = float(ols.bse[1])
        V_hat   = se_A ** 2 * n

        self._result = EstimationResult(
            psi_hat=psi_hat,
            V_hat=V_hat,
            method_name=self.name,
            n=n,
            extra={"r2": float(ols.rsquared), "se_A": se_A},
        )
        return self

    def estimate(self) -> EstimationResult:
        if self._result is None:
            raise RuntimeError("Call fit() before estimate().")
        return self._result
