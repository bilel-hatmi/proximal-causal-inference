import numpy as np


def snr_to_sigma(snr: float, sigma_U: float = 1.0) -> float:
    """
    Convert a Signal-to-Noise Ratio to a proxy noise standard deviation.

    SNR = sigma_U² / (sigma_U² + sigma_proxy²)
    => sigma_proxy = sigma_U * sqrt((1 - snr) / snr)

    Parameters
    ----------
    snr     : float in (0, 1) — proxy quality (1 = perfect, 0 = pure noise)
    sigma_U : float — reference scale for U (default 1.0 per decision D2)

    Returns
    -------
    sigma_proxy : float
    """
    if not (0.0 < snr < 1.0):
        raise ValueError(f"SNR must be strictly in (0, 1), got {snr}")
    return sigma_U * np.sqrt((1.0 - snr) / snr)


def snr_empirical(U: np.ndarray, proxy: np.ndarray) -> float:
    """
    Compute the empirical SNR from data.

    SNR_empirical = Var(U) / Var(proxy)

    Used by T4 of verify_dgp() to check that the declared SNR matches the
    generated data. For a proxy = U + eps with eps ~ N(0, sigma_proxy²) and
    U ~ N(0, sigma_U²), the population SNR equals sigma_U² / (sigma_U² + sigma_proxy²),
    which this estimator converges to as n -> infinity.

    Parameters
    ----------
    U     : (n,) array — latent confounder values
    proxy : (n,) array — proxy values (W or Z)

    Returns
    -------
    float
    """
    return float(np.var(U) / np.var(proxy))
