"""
Oracle estimator — OLS with U observed directly.

This is the theoretical ceiling: an estimator that knows the latent
confounder U. No real method can do better.

Model fitted:  Y ~ 1 + A + U + X  (OLS)
Estimand:      coefficient of A  =  alpha  =  psi_0
"""

from typing import Optional

import numpy as np
import statsmodels.api as sm

from pci.dgps.base import DGPSample
from pci.estimators.base import EstimationResult, PCIEstimator


class OracleDirect(PCIEstimator):
    """
    Oracle estimator: OLS regression of Y on (A, U, X).

    Uses the latent confounder U directly — only valid in simulation.
    Returns the coefficient of A as psi_hat.

    Variance: standard OLS formula (homoskedastic assumption matches DGP).
    V_hat = se_A^2 * n  so that  se_hat = sqrt(V_hat / n) = se_A.
    """

    def __init__(self) -> None:
        self._result: Optional[EstimationResult] = None

    @property
    def name(self) -> str:
        return "OracleDirect"

    def fit(self, sample: DGPSample) -> 'OracleDirect':
        n = len(sample.Y)

        # Design matrix: [const, A, U, X]
        if sample.X is not None:
            X_design = sm.add_constant(
                np.column_stack([sample.A, sample.U, sample.X]),
                has_constant='add'
            )
        else:
            X_design = sm.add_constant(
                np.column_stack([sample.A, sample.U]),
                has_constant='add'
            )

        ols = sm.OLS(sample.Y, X_design).fit()

        # Column layout after add_constant: [const=0, A=1, U=2, X=3]
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
