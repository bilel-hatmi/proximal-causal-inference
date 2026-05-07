import numpy as np
import scipy.stats


def compute_metrics(
    psi_hat: np.ndarray,
    V_hat: np.ndarray,
    psi_0: float,
    n: int,
    alpha: float = 0.05,
) -> dict:
    """
    Parameters
    ----------
    psi_hat : (M,) point estimates across M replications
    V_hat   : (M,) variance estimates (NOT divided by n — raw asymptotic variance)
    psi_0   : true parameter value
    n       : sample size used in each replication
    alpha   : nominal level for confidence intervals (default 0.05 -> 95% CI)

    Returns
    -------
    dict with keys: bias, variance, rmse, coverage, ser, cv_var, n, M, alpha
    """
    psi_hat = np.asarray(psi_hat, dtype=float)
    V_hat = np.asarray(V_hat, dtype=float)

    M = len(psi_hat)
    z_crit = scipy.stats.norm.ppf(1 - alpha / 2)

    bias = float(np.mean(psi_hat) - psi_0)
    variance = float(np.var(psi_hat, ddof=1))
    rmse = float(np.sqrt(bias**2 + variance))

    se_hat = np.sqrt(V_hat / n)
    lo = psi_hat - z_crit * se_hat
    hi = psi_hat + z_crit * se_hat
    coverage = float(np.mean((lo <= psi_0) & (psi_0 <= hi)))

    # SER = 1 -> well-calibrated; < 1 -> SE underestimated
    ser = float(np.mean(se_hat) / np.sqrt(variance)) if variance > 0 else float("nan")

    # cv_var: stability of the variance estimator across replications
    mean_V = float(np.mean(V_hat))
    cv_var = float(np.std(V_hat) / mean_V) if mean_V > 0 else float("nan")

    return dict(
        bias=bias,
        variance=variance,
        rmse=rmse,
        coverage=coverage,
        ser=ser,
        cv_var=cv_var,
        n=n,
        M=M,
        alpha=alpha,
    )
