"""
Phase 15 — Method registry for the final 8-method comparison on J(pi_{a,h}).

Provides 5 NEW Tier-2 wrapper PCIEstimators that target J(pi):
  1. OraclePolicy     -- DRDoseResponse with OracleBridgeH + OracleBridgeQ
  2. NaivePolicy      -- linear OLS Y ~ A + X (ignores W, Z); plug-in only
  3. LinearBridgeREG  -- TSLS Stage 2 linear h-bridge; plug-in only
  4. LinearBridgeDR   -- TSLS h + OracleBridgeQ (semi-oracle linearDR)
  5. KPVPolicy        -- alias for BennettFunctionalDR(lambda_r=1e10) -- KPV h plug-in only

Plus METHOD_REGISTRY mapping for uniform invocation in Phase 15C experiments.

Design notes
------------
* All targets J(pi_{a_k, h}) for the policy grid via DR Version B formula.
* For REG-only methods (NaivePolicy, LinearBridgeREG), q=0 (ZeroQBridge) so
  J_dr collapses to J_reg.
* LinearBridgeDR uses OracleBridgeQ for semantic clarity: tests whether a linear
  h is adequate, decoupled from q estimation. The h is genuinely estimated.
* KPVPolicy is a thin alias re-exporting BennettFunctionalDR with lambda_r=1e10.

Smoke test
----------
  python -m simulations.methods.method_registry --smoke
"""
from __future__ import annotations

import argparse
from typing import Callable, Optional

import numpy as np

from pci.dgps.base import DGPSample, PCIDgp
from pci.estimators.base import EstimationResult, PCIEstimator
from pci.estimators._oracle_bridges import OracleBridgeH, OracleBridgeQ
from pci.estimators.dr_dose_response import DRDoseResponse


# ── Helper: zero q-bridge for REG-only methods ────────────────────────────────

class _ZeroQBridge:
    """q(Z, A, X) = 0 everywhere. When passed to DRDoseResponse, the DR
    correction term vanishes and J_dr == J_reg.

    Used internally by REG-only wrappers (NaivePolicy, LinearBridgeREG) so they
    return a clean J_reg curve in the standard EstimationResult format.
    """
    def predict(self, Z: np.ndarray, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.zeros_like(A, dtype=float)


# ── Helper: linear h-bridge built from coefficients (linear in A) ─────────────

class _LinearHBridge:
    """h(W, A, X) = beta_0 + beta_A * A + beta_W * W + beta_X * X (no W*A interactions).

    Marked is_linear_in_A=True so DRDoseResponse uses the analytic shortcut
    (integral of K_h * h equals h(W, a_k, X) under linearity).

    Coefficients passed at construction. No fitting here -- this is a thin
    wrapper around an already-fitted linear regression.
    """
    is_linear_in_A: bool = True
    has_rkhs_plug_in: bool = False

    def __init__(self, beta: np.ndarray, use_W: bool, use_X: bool) -> None:
        # beta layout: [intercept, beta_A, beta_W (if use_W), beta_X (if use_X)]
        self.beta = np.asarray(beta, dtype=float).ravel()
        self.use_W = use_W
        self.use_X = use_X

    def predict(self, W: np.ndarray, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        W = np.asarray(W, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        X = np.asarray(X, dtype=float).ravel() if X is not None else np.zeros_like(A)
        n = len(A)
        out = np.full(n, self.beta[0], dtype=float)        # intercept
        out += self.beta[1] * A                             # A coefficient
        idx = 2
        if self.use_W:
            out += self.beta[idx] * W
            idx += 1
        if self.use_X:
            out += self.beta[idx] * X
            idx += 1
        return out


# ── 1. OraclePolicy ────────────────────────────────────────────────────────────

class OraclePolicy(PCIEstimator):
    """
    Oracle estimator targeting J(pi). Uses analytic h_0 and q_0 from the DGP.

    Score: J(pi_{a,h}) computed via DR formula with oracle bridges.
    Since h_0 is correct, the DR correction term has expectation 0 and J_dr ~ J_reg.

    Parameters
    ----------
    dgp        : PCIDgp instance providing h0() and q0()
    a_grid     : (K,) array of dose values
    bandwidth  : optional; default Silverman
    is_linear_in_A : pass True for DGP1 (linear h_0 in A) -- enables shortcut
    """
    def __init__(
        self,
        dgp: PCIDgp,
        a_grid: np.ndarray,
        bandwidth: Optional[float] = None,
        is_linear_in_A: bool = False,
    ) -> None:
        self.dgp = dgp
        self.a_grid = np.asarray(a_grid, dtype=float)
        self.bandwidth = bandwidth
        self.is_linear_in_A = is_linear_in_A
        self._result: Optional[EstimationResult] = None

    @property
    def name(self) -> str:
        return f"OraclePolicy(K={len(self.a_grid)})"

    def fit(self, sample: DGPSample, m_true_fn: Optional[Callable] = None) -> "OraclePolicy":
        h_oracle = OracleBridgeH(self.dgp, is_linear_in_A=self.is_linear_in_A)
        q_oracle = OracleBridgeQ(self.dgp)
        inner = DRDoseResponse(
            a_grid=self.a_grid, h_model=h_oracle, q_model=q_oracle,
            bandwidth=self.bandwidth,
        )
        inner.fit(sample, m_true_fn=m_true_fn)
        res = inner.estimate()
        # Re-wrap with our name
        self._result = EstimationResult(
            psi_hat=res.psi_hat, V_hat=res.V_hat,
            method_name=self.name, n=res.n, extra=res.extra,
        )
        return self

    def estimate(self) -> EstimationResult:
        if self._result is None:
            raise RuntimeError("OraclePolicy.estimate before fit")
        return self._result


# ── 2. NaivePolicy ────────────────────────────────────────────────────────────

class NaivePolicy(PCIEstimator):
    """
    Naive baseline: linear OLS Y ~ 1 + A + X (NO W, NO Z) -> h_naive.
    Plug-in only: J(pi_{a,h}) = mean_i [ T_pi h_naive(W_i, A_i, X_i) ].

    Since h_naive is linear in A, T_pi h_naive at dose a equals h_naive evaluated
    at A=a (DRDoseResponse shortcut). Fits using least squares.
    """
    def __init__(
        self,
        a_grid: np.ndarray,
        bandwidth: Optional[float] = None,
    ) -> None:
        self.a_grid = np.asarray(a_grid, dtype=float)
        self.bandwidth = bandwidth
        self._result: Optional[EstimationResult] = None

    @property
    def name(self) -> str:
        return "NaivePolicy"

    def fit(self, sample: DGPSample, m_true_fn: Optional[Callable] = None) -> "NaivePolicy":
        Y, A = np.asarray(sample.Y), np.asarray(sample.A)
        X = np.asarray(sample.X) if sample.X is not None else np.zeros_like(A)
        # Determine if X has variation; otherwise drop
        use_X = float(np.var(X)) > 1e-12
        n = len(Y)
        # Build design matrix [1, A] or [1, A, X]
        if use_X:
            D = np.column_stack([np.ones(n), A, X])
        else:
            D = np.column_stack([np.ones(n), A])
        # OLS via lstsq (no W, no Z -> intentionally biased)
        beta, *_ = np.linalg.lstsq(D, Y, rcond=None)
        # Build linear bridge; here use_W=False, use_X=use_X
        h_naive = _LinearHBridge(beta=beta, use_W=False, use_X=use_X)
        # Plug-in via DRDoseResponse + ZeroQ
        inner = DRDoseResponse(
            a_grid=self.a_grid, h_model=h_naive, q_model=_ZeroQBridge(),
            bandwidth=self.bandwidth,
        )
        inner.fit(sample, m_true_fn=m_true_fn)
        res = inner.estimate()
        self._result = EstimationResult(
            psi_hat=res.psi_hat, V_hat=res.V_hat,
            method_name=self.name, n=res.n, extra=res.extra,
        )
        return self

    def estimate(self) -> EstimationResult:
        if self._result is None:
            raise RuntimeError("NaivePolicy.estimate before fit")
        return self._result


# ── 3. LinearBridgeREG ─────────────────────────────────────────────────────────

class LinearBridgeREG(PCIEstimator):
    """
    Linear bridge h-only: TSLS-style Stage 1 + Stage 2.

    Stage 1: regress W on (1, A, X, Z) to get W_hat (instrumented W).
    Stage 2: regress Y on (1, A, X, W_hat) to get linear h(w, a, x) =
             beta_0 + beta_A * a + beta_X * x + beta_W * w_hat.

    Plug-in J(pi) = mean_i [ h(W_i, a_k, X_i) ] under linearity in A.
    No DR correction.
    """
    def __init__(
        self,
        a_grid: np.ndarray,
        bandwidth: Optional[float] = None,
    ) -> None:
        self.a_grid = np.asarray(a_grid, dtype=float)
        self.bandwidth = bandwidth
        self._result: Optional[EstimationResult] = None

    @property
    def name(self) -> str:
        return "LinearBridgeREG"

    def fit(self, sample: DGPSample, m_true_fn: Optional[Callable] = None) -> "LinearBridgeREG":
        Y, A, W, Z = (np.asarray(sample.Y), np.asarray(sample.A),
                       np.asarray(sample.W), np.asarray(sample.Z))
        X = np.asarray(sample.X) if sample.X is not None else np.zeros_like(A)
        n = len(Y)
        use_X = float(np.var(X)) > 1e-12
        # Stage 1: W = stage1_intercept + stage1_A*A + stage1_X*X + stage1_Z*Z
        if use_X:
            D1 = np.column_stack([np.ones(n), A, X, Z])
        else:
            D1 = np.column_stack([np.ones(n), A, Z])
        gamma, *_ = np.linalg.lstsq(D1, W, rcond=None)
        W_hat = D1 @ gamma
        # Stage 2: Y = beta_0 + beta_A*A + beta_X*X + beta_W*W_hat
        if use_X:
            D2 = np.column_stack([np.ones(n), A, X, W_hat])
            beta_layout = "with_X_then_W"
        else:
            D2 = np.column_stack([np.ones(n), A, W_hat])
            beta_layout = "no_X_then_W"
        beta_s2, *_ = np.linalg.lstsq(D2, Y, rcond=None)
        # Reorder beta to match _LinearHBridge layout: [intercept, A, W, X]
        # Our _LinearHBridge expects:
        #   beta = [intercept, beta_A, beta_W (if use_W), beta_X (if use_X)]
        if use_X:
            # D2 cols: [1, A, X, W_hat] -> beta_s2 = [b0, bA, bX, bW]
            beta_lin = np.array([beta_s2[0], beta_s2[1], beta_s2[3], beta_s2[2]])
        else:
            # D2 cols: [1, A, W_hat] -> beta_s2 = [b0, bA, bW]
            beta_lin = np.array([beta_s2[0], beta_s2[1], beta_s2[2]])
        h_linear = _LinearHBridge(beta=beta_lin, use_W=True, use_X=use_X)
        inner = DRDoseResponse(
            a_grid=self.a_grid, h_model=h_linear, q_model=_ZeroQBridge(),
            bandwidth=self.bandwidth,
        )
        inner.fit(sample, m_true_fn=m_true_fn)
        res = inner.estimate()
        self._result = EstimationResult(
            psi_hat=res.psi_hat, V_hat=res.V_hat,
            method_name=self.name, n=res.n, extra=res.extra,
        )
        return self

    def estimate(self) -> EstimationResult:
        if self._result is None:
            raise RuntimeError("LinearBridgeREG.estimate before fit")
        return self._result


# ── 4. LinearBridgeDR (semi-oracle q) ─────────────────────────────────────────

class LinearBridgeDR(PCIEstimator):
    """
    Linear h-bridge (TSLS Stage 2) + oracle q-bridge -> DR Version B.

    NOTE: Uses OracleBridgeQ to test specifically whether a linear h is adequate.
    The h is genuinely estimated; q is oracle for fairness. This isolates the
    "linear bridge" question from q estimation.

    DGP-aware: requires `dgp` for OracleBridgeQ.
    """
    def __init__(
        self,
        dgp: PCIDgp,
        a_grid: np.ndarray,
        bandwidth: Optional[float] = None,
    ) -> None:
        self.dgp = dgp
        self.a_grid = np.asarray(a_grid, dtype=float)
        self.bandwidth = bandwidth
        self._linear_reg = LinearBridgeREG(a_grid=a_grid, bandwidth=bandwidth)
        self._result: Optional[EstimationResult] = None

    @property
    def name(self) -> str:
        return "LinearBridgeDR"

    def fit(self, sample: DGPSample, m_true_fn: Optional[Callable] = None) -> "LinearBridgeDR":
        # Re-fit linear h, but with OracleBridgeQ for DR
        Y, A, W, Z = (np.asarray(sample.Y), np.asarray(sample.A),
                       np.asarray(sample.W), np.asarray(sample.Z))
        X = np.asarray(sample.X) if sample.X is not None else np.zeros_like(A)
        n = len(Y)
        use_X = float(np.var(X)) > 1e-12
        # Stage 1 + Stage 2 same as LinearBridgeREG
        if use_X:
            D1 = np.column_stack([np.ones(n), A, X, Z])
        else:
            D1 = np.column_stack([np.ones(n), A, Z])
        gamma, *_ = np.linalg.lstsq(D1, W, rcond=None)
        W_hat = D1 @ gamma
        if use_X:
            D2 = np.column_stack([np.ones(n), A, X, W_hat])
            beta_s2, *_ = np.linalg.lstsq(D2, Y, rcond=None)
            beta_lin = np.array([beta_s2[0], beta_s2[1], beta_s2[3], beta_s2[2]])
        else:
            D2 = np.column_stack([np.ones(n), A, W_hat])
            beta_s2, *_ = np.linalg.lstsq(D2, Y, rcond=None)
            beta_lin = np.array([beta_s2[0], beta_s2[1], beta_s2[2]])
        h_linear = _LinearHBridge(beta=beta_lin, use_W=True, use_X=use_X)
        q_oracle = OracleBridgeQ(self.dgp)
        inner = DRDoseResponse(
            a_grid=self.a_grid, h_model=h_linear, q_model=q_oracle,
            bandwidth=self.bandwidth,
        )
        inner.fit(sample, m_true_fn=m_true_fn)
        res = inner.estimate()
        self._result = EstimationResult(
            psi_hat=res.psi_hat, V_hat=res.V_hat,
            method_name=self.name, n=res.n, extra=res.extra,
        )
        return self

    def estimate(self) -> EstimationResult:
        if self._result is None:
            raise RuntimeError("LinearBridgeDR.estimate before fit")
        return self._result


# ── 5. KPVPolicy ──────────────────────────────────────────────────────────────

class KPVPolicy(PCIEstimator):
    """
    KPV bridge h-only: KPVBridgeH plug-in via BennettFunctionalDR(lambda_r=1e10).

    This is the "clean" KPVREG class — no Riesz correction (lambda_r huge -> r ~ 0).
    Score: J(pi) = mean_i T_pi h_KPV(W_i).

    Re-uses Phase 12B/14B BennettFunctionalDR code path. lambda_r=1e10 effectively
    suppresses the Riesz correction so J_dr == J_reg (plug-in only).
    """
    def __init__(
        self,
        a_grid: np.ndarray,
        lambda_h: float = 3e-5,
        ell_scale: float = 3.5,
        n_folds: int = 5,
        bandwidth: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        self.a_grid = np.asarray(a_grid, dtype=float)
        self.lambda_h = float(lambda_h)
        self.ell_scale = float(ell_scale)
        self.n_folds = int(n_folds)
        self.bandwidth = bandwidth
        self.seed = int(seed)
        self._result: Optional[EstimationResult] = None

    @property
    def name(self) -> str:
        return "KPVPolicy"

    def fit(self, sample: DGPSample, m_true_fn: Optional[Callable] = None) -> "KPVPolicy":
        from pci.estimators.bennett_functional_dr import BennettFunctionalDR
        inner = BennettFunctionalDR(
            lambda_h=self.lambda_h, ell_scale=self.ell_scale,
            lambda_r=1e10, degree=2, feature_map_type="polynomial",
            n_folds=self.n_folds, a_grid=self.a_grid.tolist(),
            bandwidth=self.bandwidth, seed=self.seed, ref_dose_index=2,
        )
        inner.fit(sample, m_true_fn=m_true_fn)
        res = inner.estimate()
        self._result = EstimationResult(
            psi_hat=res.psi_hat, V_hat=res.V_hat,
            method_name=self.name, n=res.n, extra=res.extra,
        )
        return self

    def estimate(self) -> EstimationResult:
        if self._result is None:
            raise RuntimeError("KPVPolicy.estimate before fit")
        return self._result


# ── METHOD_REGISTRY: factory functions with consistent signature ──────────────

def make_oracle(dgp: PCIDgp, a_grid: np.ndarray, h_KDE: float, **kwargs):
    is_linear = bool(kwargs.get("is_linear_in_A", False))
    return OraclePolicy(dgp=dgp, a_grid=a_grid, bandwidth=h_KDE, is_linear_in_A=is_linear)


def make_naiveREG(dgp: PCIDgp, a_grid: np.ndarray, h_KDE: float, **kwargs):
    return NaivePolicy(a_grid=a_grid, bandwidth=h_KDE)


def make_linearREG(dgp: PCIDgp, a_grid: np.ndarray, h_KDE: float, **kwargs):
    return LinearBridgeREG(a_grid=a_grid, bandwidth=h_KDE)


def make_linearDR(dgp: PCIDgp, a_grid: np.ndarray, h_KDE: float, **kwargs):
    return LinearBridgeDR(dgp=dgp, a_grid=a_grid, bandwidth=h_KDE)


def make_KPVREG(dgp: PCIDgp, a_grid: np.ndarray, h_KDE: float, **kwargs):
    return KPVPolicy(
        a_grid=a_grid,
        lambda_h=kwargs.get("lambda_h", 3e-5),
        ell_scale=kwargs.get("ell_scale", 3.5),
        bandwidth=h_KDE,
    )


def make_DRKPV(dgp: PCIDgp, a_grid: np.ndarray, h_KDE: float, ell_W0: float,
                ell_A0: float, ell_Z0: float, **kwargs):
    """DRKernel with KPV-super h + KPV q. Phase 9B+ benchmark."""
    from pci.estimators.dr_kernel import DRKernel
    from pci.estimators.kpv_bridge import KPVPolicyBridgeQ
    q_model = KPVPolicyBridgeQ(a_grid=a_grid, h_KDE=h_KDE, lambda_Q=1e-3, clip=None)
    return DRKernel(
        a_grid=a_grid, q_model=q_model, bandwidth=h_KDE,
        n_folds=5, random_state=42,
        ref_dose_index=kwargs.get("ref_dose_index", 2), cross_fit_q=True,
        bridge_kwargs=dict(
            lambda_1=3e-5, lambda_2=3e-5,
            ell_W=ell_W0 * 3.5, ell_A=ell_A0 * 3.5, ell_Z=ell_Z0 * 3.5,
        ),
    )


def make_best_bennett(dgp: PCIDgp, a_grid: np.ndarray, h_KDE: float, **kwargs):
    """Phase 14B winner: BennettIndepFunctionalDR with BIN_mh2000 specs."""
    from pci.estimators.bennett_indep_dr import BennettIndepFunctionalDR
    return BennettIndepFunctionalDR(
        m_h=2000, m_c=500, ell_h=2.75, ell_c=2.0,
        lambda_h=1e-5, gamma_critic=1e-4,
        lambda_r=1e-2, n_features_r=500, ell_scale_r=3.5,
        rff_seed_base_r=kwargs.get("rff_seed_base_r", 14_200_000),
        h_seed_base=kwargs.get("h_seed_base", 14_100_000),
        n_folds=5, a_grid=a_grid.tolist(),
        bandwidth=h_KDE, seed=42,
        ref_dose_index=kwargs.get("ref_dose_index", 2),
    )


def make_best_kallus(dgp: PCIDgp, a_grid: np.ndarray, h_KDE: float, ell_W0: float,
                       ell_A0: float, ell_Z0: float, **kwargs):
    """Kallus_best: KallusMinimaxBridgeH (h) + KPV q.

    Hyperparameters from Phase 14C.1 K1 (gamma_H=1e-5, lambda_stab=0.1) by default;
    will be UPDATED after Phase 15B retune."""
    from pci.estimators.dr_kernel import DRKernel
    from pci.estimators.kpv_bridge import KPVPolicyBridgeQ
    from pci.estimators.kallus_minimax import KallusMinimaxBridgeH

    gamma_H = kwargs.get("gamma_H", 1e-5)
    lambda_stab_h = kwargs.get("lambda_stab_h", 0.1)
    gamma_critic_h = kwargs.get("gamma_critic_h", 1e-3)
    n_features = kwargs.get("n_features", 400)
    ell_scale = kwargs.get("ell_scale", 3.5)
    feature_seed = kwargs.get("feature_seed", 14_300_000)

    def h_bridge_factory():
        return KallusMinimaxBridgeH(
            mode="kallus_stabilized",
            gamma_H=gamma_H, lambda_stab_h=lambda_stab_h, gamma_critic_h=gamma_critic_h,
            n_features_h=n_features, n_features_critic=n_features,
            ell_scale=ell_scale, feature_seed=feature_seed,
        )

    q_model = KPVPolicyBridgeQ(a_grid=a_grid, h_KDE=h_KDE, lambda_Q=1e-3, clip=None)
    return DRKernel(
        a_grid=a_grid, q_model=q_model, bandwidth=h_KDE,
        n_folds=5, random_state=42,
        ref_dose_index=kwargs.get("ref_dose_index", 2), cross_fit_q=True,
        h_bridge_factory=h_bridge_factory,
    )


METHOD_REGISTRY = {
    "oracle":       make_oracle,
    "naiveREG":     make_naiveREG,
    "linearREG":    make_linearREG,
    "linearDR":     make_linearDR,
    "KPVREG":       make_KPVREG,
    "DRKPV":        make_DRKPV,
    "best_bennett": make_best_bennett,
    "best_kallus":  make_best_kallus,   # Phase 15B retune may UPDATE hyperparams
}


def list_methods():
    return list(METHOD_REGISTRY.keys())


def make_method(name: str, **kwargs):
    if name not in METHOD_REGISTRY:
        raise KeyError(f"unknown method {name!r}. Available: {list_methods()}")
    return METHOD_REGISTRY[name](**kwargs)


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke():
    import time
    from pci.dgps.michaelis_menten import MichaelisMentenDGP
    from pci.dgps.cobb_douglas import CobbDouglasLinearDGP

    a_grid = np.array([1.8, 2.2, 2.6, 3.0])

    print("=" * 72)
    print("Phase 15 method_registry smoke test")
    print("=" * 72)

    for dgp_name, dgp_cls, is_linear in [
        ("DGP1 (linear)", CobbDouglasLinearDGP, True),
        ("DGP2 (MM)",     MichaelisMentenDGP,   False),
    ]:
        dgp = dgp_cls(snr_W=0.95, snr_Z=0.95)
        sample = dgp.generate(n=300, seed=42)
        h_KDE = 1.06 * float(np.std(sample.A)) * 300 ** (-1.0 / 5.0)
        ell_W0 = float(np.std(sample.W))
        ell_A0 = float(np.std(sample.A))
        ell_Z0 = float(np.std(sample.Z))

        print(f"\n--- {dgp_name}, n=300, h_KDE={h_KDE:.4f} ---")
        for name in list_methods():
            try:
                t0 = time.time()
                kwargs = dict(dgp=dgp, a_grid=a_grid, h_KDE=h_KDE,
                               ell_W0=ell_W0, ell_A0=ell_A0, ell_Z0=ell_Z0)
                if name == "oracle":
                    kwargs["is_linear_in_A"] = is_linear
                est = make_method(name, **kwargs)
                est.fit(sample, m_true_fn=dgp.m_true)
                res = est.estimate()
                jdr = res.extra.get("J_dr", np.array([np.nan]))
                jdr_arr = np.asarray(jdr, dtype=float)
                ok_finite = np.all(np.isfinite(jdr_arr))
                elapsed = time.time() - t0
                print(f"  {name:<14}  J_dr=[{jdr_arr[0]:.3f}, {jdr_arr[-1]:.3f}]  "
                      f"finite={ok_finite}  [{elapsed:.1f}s]")
            except Exception as e:
                print(f"  {name:<14}  FAILED: {type(e).__name__}: {e}")
    print("\nSmoke test complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        _smoke()
    else:
        print("Usage: python -m simulations.methods.method_registry --smoke")
