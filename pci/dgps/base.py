from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DGPSample:
    """
    Output of PCIDgp.generate(). All arrays have shape (n,).

    Naming convention (decision D1 — Tchetgen notation):
        W : outcome-inducing proxy (NCO)
        Z : treatment-inducing proxy (NCE)

    All variables are stored on whatever scale the DGP uses internally
    (e.g. log-space for Cobb-Douglas, raw scale for Michaelis-Menten).
    """
    Y: np.ndarray          # outcome
    A: np.ndarray          # treatment
    X: np.ndarray          # observed covariates (may be None if no covariates)
    Z: np.ndarray          # treatment-inducing proxy (NCE)
    W: np.ndarray          # outcome-inducing proxy (NCO)
    U: np.ndarray          # latent confounder (available in simulation only)
    psi_0: float           # true ATE for this DGP instance


class PCIDgp(ABC):
    """
    Abstract base class for all PCI data-generating processes.

    Subclasses must implement generate(), h0(), q0(), and psi_true().
    snr_W() and snr_Z() should be overridden whenever the DGP has proxy SNR
    parameters (required for T4 of verify_dgp()).
    """

    @abstractmethod
    def generate(self, n: int, seed: int) -> DGPSample:
        """Generate n observations from the DGP. Must be reproducible given seed."""

    @abstractmethod
    def h0(self, W: np.ndarray, A: np.ndarray,
           X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        True outcome bridge h₀(W, A, X).

        Satisfies E[Y - h₀(W, A, X) | Z, A, X] = 0  (Fredholm condition).
        This is NOT the regression E[Y | W, A, X] — it is the PCI bridge.
        """

    @abstractmethod
    def q0(self, Z: np.ndarray, A: np.ndarray,
           X: Optional[np.ndarray] = None) -> np.ndarray:
        """True treatment bridge q₀(Z, A, X)."""

    @abstractmethod
    def psi_true(self, a: float) -> float:
        """
        True ATE structural parameter (e.g. slope alpha for linear DGPs).

        NOTE: For DGP1 (Cobb-Douglas linear), psi_true(a) = alpha (slope dm/da)
        regardless of `a`. It is NOT the dose-response level m(a) = E[Y(a)].
        Use m_true(a) for the level. The two coincide only for the contrast
        psi_true * 1 + 0 = m_true(1) - m_true(0) when m is linear through origin
        (which it is NOT in general — m_true(0) = (1-alpha)*mu_L ≠ 0).
        """

    @abstractmethod
    def m_true(self, a: float) -> float:
        """
        True dose-response m(a) = E[Y(a)] at dose a (LEVEL, not slope).

        Identification (Kallus 2021 Lemma 2):  m(a) = E[h₀(W, a, X)].

        For DGP1 (Cobb-Douglas linear):
            m(a) = alpha * a + (1 - alpha) * mu_L     (analytical)

        For DGP2 (Michaelis-Menten, future Phase 7):
            m(a) computed via Gauss-Hermite quadrature against U.
        """

    def snr_W(self) -> float:
        """Signal-to-noise ratio of the outcome proxy W."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement snr_W()"
        )

    def snr_Z(self) -> float:
        """Signal-to-noise ratio of the treatment proxy Z."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement snr_Z()"
        )

    def __repr__(self) -> str:
        try:
            return (f"{self.__class__.__name__}"
                    f"(snr_W={self.snr_W():.2f}, snr_Z={self.snr_Z():.2f})")
        except NotImplementedError:
            return f"{self.__class__.__name__}()"
