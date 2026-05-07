from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pci.dgps.base import DGPSample


@dataclass
class EstimationResult:
    """
    Output of PCIEstimator.estimate().

    V_hat is the estimated asymptotic variance NOT divided by n.
    The 95% CI is: psi_hat ± 1.96 * sqrt(V_hat / n).
    This convention matches compute_metrics() in analysis/metrics.py.
    """
    psi_hat: float           # point estimate of ATE
    V_hat: float             # estimated asymptotic variance (NOT divided by n)
    method_name: str         # human-readable method identifier
    n: int                   # sample size used
    extra: dict = field(default_factory=dict)  # method-specific diagnostics


class PCIEstimator(ABC):
    """
    Abstract base class for all PCI estimation methods.

    Usage pattern (matches verify_method() in SKILL.md):
        est = MethodClass().fit(sample)
        result = est.estimate()
        # result.psi_hat compared to dgp.psi_true(a=1.0)
    """

    @abstractmethod
    def fit(self, sample: DGPSample) -> 'PCIEstimator':
        """Fit the estimator on the given DGPSample. Returns self for chaining."""

    @abstractmethod
    def estimate(self) -> EstimationResult:
        """Return point estimate and variance estimate. Must call fit() first."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable method name, e.g. 'OracleDirect'."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
