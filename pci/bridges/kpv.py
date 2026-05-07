"""
pci.bridges.kpv
===============

KPVBridgeH — True KPV ridge Fredholm bridge (Mastouri 2021).
KPVPolicyBridgeQ — Policy-weighted action-bridge correction (Kallus 2021, Phase 9B).

Extracted from ``pci/estimators/kpv_bridge.py``. The original
module continues to re-export both classes and the private helpers
(``_median_bandwidth``, ``_rbf_gram``, ``_rbf_kde_convolution``, ``_spd_solve``)
as a backward-compatibility shim with class identity preserved.

Method
------
Two-stage kernel ridge solver for the outcome bridge h₀(W, A) satisfying the
Fredholm condition  E[h₀(W, A) | Z, A] = E[Y | Z, A].

Stage 1: Conditional mean embedding μ_{W|Z,A} via
    Γ = (K_ZA + n·λ₁·I)⁻¹ K_ZA          (n × n)
    where K_ZA = K_Z ⊙ K_A (Hadamard product of marginal Grams =
    Gram of the tensor-product RKHS H_Z ⊗ H_A on stacked features).

Stage 2: Ridge regression on RKHS features ξᵢ = μ̂ᵢ ⊗ φ_A(Aᵢ) with Gram
    G = (Γᵀ K_W Γ) ⊙ K_A                (n × n, SPD)
    α = (G + n·λ₂·I)⁻¹ Y                (Cholesky)

Prediction (Γ in the middle is critical — distinguishes KPV from collocation):
    ĥ(w, a) = Σᵢ αᵢ · ⟨μ̂ᵢ, k_W(w, ·)⟩_{H_W} · k_A(Aᵢ, a)
            = Σᵢ αᵢ · (K_W(w, W_train) Γ)[i] · k_A(Aᵢ, a)

Policy plug-in via closed-form Gauss-Gauss convolution:
    ∫ ĥ(W_i, t) K_h(t - a) dt = (K_W(W_i, W_train) Γ) @ (α ⊙ k̃(A_train, a, h))
    where  k̃(A_j, a, h) = ℓ_A / √(ℓ_A² + h²) · exp(-(A_j - a)² / (2(ℓ_A² + h²)))

NOT collocation
---------------
A common alternative is the Fredholm collocation solver: ansatz
ĥ = Σⱼ αⱼ k_W(W_j, ·) k_A(A_j, ·) + collocation at training points giving
M α = H₁ Y with M = (H₁ K_W) ⊙ K_A. That works as a Fredholm solver but is NOT
KPV — predictions don't pass through Γ. The two methods give different ĥ on
nonlinear kernels. The W² embedding test (T_KPV4) discriminates them.

Phase 8.5: Nyström approximation activated by `n_landmarks` kwarg. Scales to
n ≤ 15000 with m ≈ 200–500 landmarks. Full O(n³) path still used when
n_landmarks is None or n_landmarks >= n.

References
----------
Mastouri, Zhu, Gretton et al. (2021), "Proximal Causal Learning with Kernels:
Two-Stage Estimation and Moment Restriction", ICML 2021.
"""

from __future__ import annotations

from typing import Optional, Tuple
import warnings

import numpy as np
from scipy.spatial.distance import pdist
import scipy.linalg as sla


# ── Module-level utilities (private) ─────────────────────────────────────────

def _median_bandwidth(x: np.ndarray) -> float:
    """
    Median heuristic bandwidth: ℓ = median(|xᵢ - xⱼ|, i<j).

    For a 1-D array of length n, computes the n(n-1)/2 pairwise distances and
    returns their median. Used as the RBF length-scale ℓ in
    k(s, t) = exp(-(s-t)² / (2 ℓ²)).

    For n > 5000, this is O(n²) memory — acceptable for Phase 8 (n ≤ 5000).
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        raise ValueError(f"Need at least 2 samples, got {x.size}.")
    d = pdist(x[:, None])
    med = float(np.median(d))
    if med == 0.0:
        # Degenerate: all points identical. Fall back to a tiny scale.
        med = 1e-6
    return med


def _rbf_gram(x: np.ndarray, y: np.ndarray, ell: float) -> np.ndarray:
    """
    RBF Gram matrix K[i, j] = exp(-(xᵢ - yⱼ)² / (2 ℓ²)).

    Shape: (len(x), len(y)). Length-scale ℓ corresponds to sklearn gamma = 1/(2ℓ²).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if ell <= 0:
        raise ValueError(f"ell must be > 0, got {ell}.")
    diff = x[:, None] - y[None, :]
    return np.exp(-0.5 * (diff / ell) ** 2)


def _rbf_kde_convolution(
    A_train: np.ndarray,
    a: float,
    h: float,
    ell_A: float,
) -> np.ndarray:
    """
    Closed-form convolution of RBF kernel with normalised Gaussian density kernel.

    Computes, for each training point Aⱼ,
        k̃(Aⱼ, a, h) := ∫ k_A(Aⱼ, t) · K_h(t - a) dt
    where:
        k_A(s, t) = exp(-(s - t)² / (2 ℓ_A²))                  (RBF, NOT normalised)
        K_h(u)    = exp(-u² / (2 h²)) / (h · √(2π))             (Gaussian density)

    Closed form (Gauss × Gauss):
        k̃(Aⱼ, a, h) = ℓ_A / √(ℓ_A² + h²) · exp(-(Aⱼ - a)² / (2(ℓ_A² + h²)))

    Used by KPVBridgeH.plug_in_policy via:
        ∫ ĥ(W_i, t) K_h(t - a) dt = (K_W(W_i, W_train) @ Γ) @ (α ⊙ k̃)
    """
    A_train = np.asarray(A_train, dtype=float).ravel()
    if h <= 0 or ell_A <= 0:
        raise ValueError(f"h and ell_A must be > 0, got h={h}, ell_A={ell_A}.")
    var_sum = ell_A ** 2 + h ** 2
    prefactor = ell_A / np.sqrt(var_sum)
    return prefactor * np.exp(-((A_train - a) ** 2) / (2.0 * var_sum))


# ── Helper: SPD solve with Cholesky → fallback chain ─────────────────────────

def _spd_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve A x = b with A SPD (or near-SPD). Tries Cholesky first; falls back to
    np.linalg.solve, then to lstsq with rcond=1e-10. Issues a warning on fallback.

    Used for both Stage 1 (K_ZA + nλ₁I) and Stage 2 (G + nλ₂I).
    """
    try:
        c, low = sla.cho_factor(A, lower=True, check_finite=False)
        return sla.cho_solve((c, low), b, check_finite=False)
    except (sla.LinAlgError, np.linalg.LinAlgError):
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            warnings.warn(
                "KPVBridgeH: ill-conditioned linear system, falling back to lstsq.",
                RuntimeWarning,
            )
            sol, *_ = np.linalg.lstsq(A, b, rcond=1e-10)
            return sol


# ── Main class ───────────────────────────────────────────────────────────────

class KPVBridgeH:
    """
    True KPV ridge bridge (Mastouri 2021) for the outcome bridge h₀(W, A).

    NOT a PCIEstimator (no scalar estimand, no fit(sample) signature). Used as
    the h-bridge inside DRKernel; takes raw arrays (W, A, Z, Y) at fit time.

    See module docstring for the math (Stage 1 + Stage 2 + plug-in convolution).

    Parameters
    ----------
    lambda_1, lambda_2 : float, optional
        Tikhonov regularisation for Stage 1, Stage 2.
        Default (set at fit time): 1e-2 · n^(-0.7) (Ying scaling).
    ell_W, ell_A, ell_Z : float, optional
        RBF length-scales. Default (set at fit time): median heuristic on the
        training slice.
    n_landmarks : int, optional
        If set, activates the Nyström approximation with m = n_landmarks
        landmarks chosen by random uniform subsampling (seed=n). When
        n_landmarks >= n, silently falls back to the full O(n³) path.
        Recommended: 100 ≤ n_landmarks ≤ 500 for n ≤ 15000. Default None.
    """

    has_rkhs_plug_in: bool = True   # signals to DRDoseResponse._policy_integral
    is_linear_in_A:   bool = False

    def __init__(
        self,
        lambda_1: Optional[float] = None,
        lambda_2: Optional[float] = None,
        ell_W: Optional[float] = None,
        ell_A: Optional[float] = None,
        ell_Z: Optional[float] = None,
        n_landmarks: Optional[int] = None,
    ) -> None:
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.ell_W = ell_W
        self.ell_A = ell_A
        self.ell_Z = ell_Z
        self.n_landmarks = n_landmarks

        # Trained state — populated by fit()
        self._fitted: bool = False
        # Full O(n³) mode caches (populated when n_landmarks is None or >= n)
        self._W_train: Optional[np.ndarray] = None
        self._A_train: Optional[np.ndarray] = None
        self._Gamma:   Optional[np.ndarray] = None
        self._alpha:   Optional[np.ndarray] = None
        self._n_train: int = 0
        # Nyström O(nm²) mode caches (populated when n_landmarks < n)
        self._nystrom_mode: bool = False
        self._M:       Optional[np.ndarray] = None   # (m, m) K_W_mm⁻¹ C
        self._alpha_m: Optional[np.ndarray] = None   # (m,) Nyström ridge coefficients
        self._W_lm:   Optional[np.ndarray] = None   # (m,) landmark W values
        self._A_lm:   Optional[np.ndarray] = None   # (m,) landmark A values

    # ── Main interface ──────────────────────────────────────────────────────

    def fit(
        self,
        W: np.ndarray,
        A: np.ndarray,
        Z: np.ndarray,
        Y: np.ndarray,
    ) -> "KPVBridgeH":
        """
        Fit the two-stage KPV ridge bridge on (W, A, Z, Y).

        Full O(n³) path (n_landmarks=None or n_landmarks >= n):
            1. Bandwidths via median heuristic if not user-supplied.
            2. Ridge λ₁, λ₂ default to 1e-2 · n^(-0.7) if not user-supplied.
            3. Build K_W, K_A, K_Z, K_ZA = K_Z ⊙ K_A  (all n×n).
            4. Stage 1: Γ = (K_ZA + n·λ₁·I)⁻¹ K_ZA  (n×n).
            5. Stage 2: G = (Γᵀ K_W Γ) ⊙ K_A; α = (G + n·λ₂·I)⁻¹ Y.
            Caches: _W_train, _A_train, _Gamma (n×n), _alpha (n,).

        Nyström O(nm²) path (n_landmarks=m, m < n):
            1–2. Same bandwidth + λ resolution on full data.
            3. Select m landmarks I_m ⊂ {0,...,n-1} (seed=n, reproducible).
            4. Build sub-matrices K_ZA_nm (n×m), K_ZA_mm (m×m),
               K_W_mn (m×n), K_W_mm (m×m), K_A_mm (m×m).
            5. Stage 1: Gamma_nm = K_ZA_nm (K_ZA_mm + nλ₁I)⁻¹  (n×m).
            6. Stage 2: C = K_W_mn Gamma_nm (m×m);
               M = K_W_mm⁻¹ C (m×m);
               G_mm = C.T M ⊙ K_A_mm  (m×m, SPD);
               alpha_m = (G_mm + nλ₂I)⁻¹ (Gamma_nm.T Y).
            Caches: _M (m×m), _alpha_m (m,), _W_lm (m,), _A_lm (m,).
        """
        W = np.asarray(W, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        Z = np.asarray(Z, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        n = len(Y)
        if not (len(W) == len(A) == len(Z) == n):
            raise ValueError("W, A, Z, Y must all have the same length.")

        # Resolve bandwidths
        ell_W = self.ell_W if self.ell_W is not None else _median_bandwidth(W)
        ell_A = self.ell_A if self.ell_A is not None else _median_bandwidth(A)
        ell_Z = self.ell_Z if self.ell_Z is not None else _median_bandwidth(Z)

        # Resolve regularisation (Ying scaling)
        lam1 = self.lambda_1 if self.lambda_1 is not None else 1e-2 * n ** (-0.7)
        lam2 = self.lambda_2 if self.lambda_2 is not None else 1e-2 * n ** (-0.7)

        # ── Landmark selection ────────────────────────────────────────────────
        # Nyström mode is activated when n_landmarks is set AND < n.
        # When n_landmarks >= n, silently use full mode (no benefit from Nyström).
        m = None
        if self.n_landmarks is not None and int(self.n_landmarks) < n:
            m = int(self.n_landmarks)
            # seed=n: deterministic — same (W,A,Z,Y,n) always gives same landmarks.
            # Each KFold fold has its own n_fold_train, so different folds get
            # different (but reproducible) landmark sets.
            rng_lm = np.random.default_rng(seed=n)
            I_m = rng_lm.choice(n, size=m, replace=False)
            I_m.sort()  # sorted for traceability; distribution unchanged

        if m is None:
            # ── Full O(n³) path ───────────────────────────────────────────────
            K_W = _rbf_gram(W, W, ell_W)
            K_A = _rbf_gram(A, A, ell_A)
            K_Z = _rbf_gram(Z, Z, ell_Z)
            K_ZA = K_Z * K_A   # tensor product RKHS Gram

            # Stage 1: Γ = (K_ZA + n·λ₁·I)⁻¹ K_ZA
            Gamma = _spd_solve(K_ZA + n * lam1 * np.eye(n), K_ZA)

            # Stage 2: G = (Γᵀ K_W Γ) ⊙ K_A  (SPD)
            G = (Gamma.T @ K_W @ Gamma) * K_A
            # Symmetrise to neutralise floating-point asymmetry before Cholesky
            G = 0.5 * (G + G.T)

            # α = (G + n·λ₂·I)⁻¹ Y
            alpha = _spd_solve(G + n * lam2 * np.eye(n), Y)

            # Cache full-mode state
            self._W_train = W
            self._A_train = A
            self._Gamma = Gamma
            self._alpha = alpha
            self._nystrom_mode = False

        else:
            # ── Nyström O(nm²) path ───────────────────────────────────────────
            # Never build the full (n,n) Gram matrices — use (n,m) and (m,m)
            # sub-matrices only.
            K_ZA_nm = (
                _rbf_gram(Z, Z[I_m], ell_Z) * _rbf_gram(A, A[I_m], ell_A)
            )                                                            # (n, m)
            K_ZA_mm = K_ZA_nm[I_m, :]                                   # (m, m)
            K_W_mn  = _rbf_gram(W[I_m], W, ell_W)                      # (m, n)
            K_W_mm  = K_W_mn[:, I_m]                                    # (m, m)
            K_A_mm  = _rbf_gram(A[I_m], A[I_m], ell_A)                 # (m, m)

            # Stage 1 Nyström: Gamma_nm = K_ZA_nm (K_ZA_mm + nλ₁I)⁻¹
            # Solve (K_ZA_mm + nλ₁I) X = K_ZA_nm.T  →  X.T = Gamma_nm
            Gamma_nm = _spd_solve(
                K_ZA_mm + n * lam1 * np.eye(m), K_ZA_nm.T
            ).T                                                          # (n, m)

            # Stage 2:
            #   C    = K_W_mn Gamma_nm                    (m, m)
            #   M    = K_W_mm⁻¹ C                         (m, m)  ← cached
            #   G_mm = C.T M ⊙ K_A_mm  ≈ Γᵀ K_W Γ ⊙ K_A  (m, m)  SPD
            #   rhs  = Gamma_nm.T Y                       (m,)
            #   α_m  = (G_mm + nλ₂I)⁻¹ rhs               (m,)    ← cached
            C    = K_W_mn @ Gamma_nm                                     # (m, m)
            M    = _spd_solve(K_W_mm, C)                                 # (m, m)
            G_mm = C.T @ M * K_A_mm                                      # (m, m)
            G_mm = 0.5 * (G_mm + G_mm.T)

            rhs_m   = Gamma_nm.T @ Y                                     # (m,)
            alpha_m = _spd_solve(G_mm + n * lam2 * np.eye(m), rhs_m)   # (m,)

            # Cache Nyström state
            self._M       = M
            self._alpha_m = alpha_m
            self._W_lm    = W[I_m]
            self._A_lm    = A[I_m]
            self._nystrom_mode = True

        self._n_train = n
        self.ell_W = ell_W
        self.ell_A = ell_A
        self.ell_Z = ell_Z
        self.lambda_1 = lam1
        self.lambda_2 = lam2
        self._fitted = True
        return self

    def predict(self, W: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Evaluate ĥ(Wᵢ, Aᵢ) for each test point i.

        Full mode (n_landmarks=None):
            KW_test = K_W(W_test, W_train)          (n_test, n)
            KA_test = K_A(A_test, A_train)          (n_test, n)
            Ew_test = KW_test @ Γ                   (n_test, n)  — CME projection
            ĥ       = sum(Ew_test * KA_test * α,   axis=1)

        Nyström mode (n_landmarks=m):
            KW_test_m = K_W(W_test, W_lm)           (n_test, m)
            Ew_test   = KW_test_m @ M               (n_test, m)  — M = K_W_mm⁻¹ C
            KA_test_m = K_A(A_test, A_lm)           (n_test, m)
            ĥ         = sum(Ew_test * KA_test_m * α_m, axis=1)

        The M (or Γ in full mode) multiplication is what makes this true KPV.
        """
        if not self._fitted:
            raise RuntimeError("KPVBridgeH.predict called before fit().")
        W = np.asarray(W, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        if len(W) != len(A):
            raise ValueError("W and A must have the same length for prediction.")

        if not self._nystrom_mode:
            # Full O(n_test × n²) path
            KW_test = _rbf_gram(W, self._W_train, self.ell_W)   # (n_test, n)
            KA_test = _rbf_gram(A, self._A_train, self.ell_A)   # (n_test, n)
            Ew_test = KW_test @ self._Gamma                      # (n_test, n)
            return np.sum(Ew_test * KA_test * self._alpha[None, :], axis=1)
        else:
            # Nyström O(n_test × m) path
            KW_test_m = _rbf_gram(W, self._W_lm, self.ell_W)    # (n_test, m)
            Ew_test   = KW_test_m @ self._M                      # (n_test, m)
            KA_test_m = _rbf_gram(A, self._A_lm, self.ell_A)    # (n_test, m)
            return np.sum(Ew_test * KA_test_m * self._alpha_m[None, :], axis=1)

    def plug_in_policy(
        self,
        W: np.ndarray,
        a: float,
        h: float,
    ) -> np.ndarray:
        """
        Compute ∫ ĥ(W_i, t) K_h(t - a) dt for each test point i.

        Closed-form via Gauss-Gauss convolution:
            k̃(A_j, a, h) = ℓ_A / √(ℓ_A²+h²) · exp(-(A_j-a)² / (2(ℓ_A²+h²)))
            ∫ ĥ(W_i, t) K_h(t-a) dt = Ew_test @ (α ⊙ k̃)

        where Ew_test = KW_test @ Γ (full) or KW_test_m @ M (Nyström),
        and k̃ is evaluated at A_train (full) or A_lm (Nyström).

        K_h is the normalised Gaussian density kernel (consistent with
        DRDoseResponse._kernel).
        """
        if not self._fitted:
            raise RuntimeError("KPVBridgeH.plug_in_policy called before fit().")
        W = np.asarray(W, dtype=float).ravel()

        if not self._nystrom_mode:
            # Full path
            KW_test = _rbf_gram(W, self._W_train, self.ell_W)   # (n_test, n)
            Ew_test = KW_test @ self._Gamma                      # (n_test, n)
            k_tilde = _rbf_kde_convolution(
                self._A_train, float(a), float(h), self.ell_A
            )                                                    # (n,)
            return Ew_test @ (self._alpha * k_tilde)
        else:
            # Nyström path — convolution against landmark A values only
            KW_test_m = _rbf_gram(W, self._W_lm, self.ell_W)    # (n_test, m)
            Ew_test   = KW_test_m @ self._M                      # (n_test, m)
            k_tilde_m = _rbf_kde_convolution(
                self._A_lm, float(a), float(h), self.ell_A
            )                                                    # (m,)
            return Ew_test @ (self._alpha_m * k_tilde_m)


# ── KPVPolicyBridgeQ ─────────────────────────────────────────────────────────

class KPVPolicyBridgeQ:
    """
    Policy-weighted action-bridge correction for J(π_{a,h}) — Phase 9B, A2.

    Estimates a policy-specific correction q̂_a satisfying the Kallus (2021)
    policy-weighted moment condition:

        P_n[π_a(A) q̂_a(Z,A) g(W,A)] ≈ P_n[T_{π_a} g(W)]
                                       = P_n[∫ g(W,t) π_a(t) dt]

    for test functions g_j(W,A) = k_W(W,W_j) k_A(A,A_j) in H_WA.

    With RKHS expansion q_a(z,a) = Σ_ℓ α_{ℓ,a} k_Z(Z_ℓ,z) k_A(A_ℓ,a):

        M_a  = (1/n) K_WA^T Λ_a K_ZA            (n,n, NOT symmetric)
        b_a  = k̃_a ⊙ (K_W @ 1_n) / n           (n,)
        α̂_a = argmin ‖M_a α - b_a‖² + λ‖α‖²   → (M_a^T M_a + λI) α = M_a^T b_a

    where:
        K_WA = K_W(W,W) ⊙ K_A(A,A)     test-function gram    (n,n)
        K_ZA = K_Z(Z,Z) ⊙ K_A(A,A)     expansion gram        (n,n)
        Λ_a  = diag(K_h(A_i - a))       policy weights        (n,n diag)
        k̃_a[j] = ∫ k_A(t,A_j) K_h(t-a) dt                   (Gauss-Gauss)

    API contract
    ------------
    predict(Z, A, dose_index) → q̂_a(Z_i, A_i)   shape (n_test,)
    DRKernel forms:  weight = K_h(A - a) * q̂_a
    NEVER multiply by K_h again inside predict().

    Policy-specific: a different α̂_a stored per dose a_k in a_grid.
    q̂_a is NOT universal q₀ — it is validated by the Riesz moment, not q₀ recovery.

    Parameters
    ----------
    a_grid     : 1D array of dose values (one α per dose)
    h_KDE      : Gaussian bandwidth for the policy K_h
    lambda_Q   : Ridge regularisation (default 1e-3)
    ell_W, ell_A, ell_Z : RBF length-scales (default: median heuristic)
    clip       : Upper bound for clipping q̂ in predict(); default 5·n^{0.25}

    References
    ----------
    Kallus (2021) §4, Eqs 12–13 — minimax duality + policy-weighted moment.
    """

    def __init__(
        self,
        a_grid: np.ndarray,
        h_KDE: float,
        lambda_Q: float = 1e-3,
        ell_W: Optional[float] = None,
        ell_A: Optional[float] = None,
        ell_Z: Optional[float] = None,
        clip: Optional[float] = None,
        compute_cond: bool = False,
    ) -> None:
        self.a_grid = np.asarray(a_grid, dtype=float)
        self.h_KDE = float(h_KDE)
        self.lambda_Q = float(lambda_Q)
        self.ell_W = ell_W
        self.ell_A = ell_A
        self.ell_Z = ell_Z
        self.clip = clip
        self.compute_cond = bool(compute_cond)
        # Note: compute_cond=False (default) skips np.linalg.cond calls which are
        # O(n^3) SVD and can take 80%+ of per-rep time at n>=1000.
        # Set compute_cond=True only for small n or explicit diagnostic runs.

        # Trained state
        self._fitted: bool = False
        self._alpha_list: list = []            # [α_k (n,) for k=0..K-1]
        self._Z_train: Optional[np.ndarray] = None
        self._A_train: Optional[np.ndarray] = None
        self._n_train: int = 0

        # Post-fit diagnostics (per dose, shape (K,))
        self._riesz_g1_list: list = []         # E_n[π_k q̂_k] per dose (≈1)
        self._residual_norm_rel: list = []     # ‖M_k α̂_k - b_k‖ / ‖b_k‖
        self._cond_M_list: list = []           # condition number of M_k
        self._cond_Areg_list: list = []        # condition number of M^T M + λI
        self._negative_share_list: list = []   # fraction q̂ < 0 on training data
        self._q_raw_p99_list: list = []        # p99 of |q̂| on training data

    # ── Core fit ─────────────────────────────────────────────────────────────

    def fit(
        self,
        W: np.ndarray,
        A: np.ndarray,
        Z: np.ndarray,
    ) -> "KPVPolicyBridgeQ":
        """
        Fit per-dose α̂_k via ridge/GMM on Kallus (2021) policy-weighted moment.

        Steps (per dose a_k in a_grid):
            1. k̃_k = _rbf_kde_convolution(A, a_k, h_KDE, ell_A)       (n,)
            2. b_k  = k̃_k * (K_W @ 1_n) / n                          (n,) element-wise
            3. Λ_k  = K_h(A - a_k)                                     (n,) density weights
            4. M_k  = (K_WA * Λ_k[:,None]).T @ K_ZA / n               (n,n)
            5. (M_k^T M_k + λI) α_k = M_k^T b_k                      → solve SPD system
        """
        W = np.asarray(W, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        Z = np.asarray(Z, dtype=float).ravel()
        n = len(W)
        if not (len(A) == len(Z) == n):
            raise ValueError("W, A, Z must all have the same length.")

        # Resolve bandwidths
        ell_W = self.ell_W if self.ell_W is not None else _median_bandwidth(W)
        ell_A = self.ell_A if self.ell_A is not None else _median_bandwidth(A)
        ell_Z = self.ell_Z if self.ell_Z is not None else _median_bandwidth(Z)

        # Build full Gram matrices (O(n²) memory — full mode Phase 9B.1)
        K_W  = _rbf_gram(W, W, ell_W)    # (n, n)
        K_A  = _rbf_gram(A, A, ell_A)    # (n, n)
        K_Z  = _rbf_gram(Z, Z, ell_Z)    # (n, n)
        K_WA = K_W * K_A                  # (n, n)  test-function gram H_WA
        K_ZA = K_Z * K_A                  # (n, n)  expansion gram H_ZA

        # Row sums of K_W (precomputed once, shared across doses)
        row_sums_W = K_W @ np.ones(n) / n   # (n,)

        identity_n = np.eye(n)
        lam_Q = self.lambda_Q
        h = self.h_KDE

        alpha_list = []
        riesz_g1_list = []
        residual_norm_rel_list = []
        cond_M_list = []
        cond_Areg_list = []
        negative_share_list = []
        q_raw_p99_list = []

        for a_k in self.a_grid:
            a_k = float(a_k)

            # k̃_k[j] = ∫ k_A(t, A_j) K_h(t-a_k) dt  (Gauss-Gauss closed form)
            k_tilde = _rbf_kde_convolution(A, a_k, h, ell_A)        # (n,)

            # RHS: b_k[j] = k̃_k[j] * (Σ_i K_W(W_j,W_i)) / n
            b_k = k_tilde * row_sums_W                               # (n,)

            # Policy weights Λ_k[i] = K_h(A_i - a_k)  [normalised Gaussian density]
            Lambda_k = (
                np.exp(-0.5 * ((A - a_k) / h) ** 2)
                / (h * np.sqrt(2.0 * np.pi))
            )                                                         # (n,)

            # M_k = (1/n) K_WA.T diag(Lambda_k) K_ZA
            # Efficient: (K_WA * Lambda_k[:,None]).T @ K_ZA / n
            M_k = (K_WA * Lambda_k[:, None]).T @ K_ZA / n           # (n, n)

            # Ridge/GMM normal equations: (M^T M + λI) α = M^T b
            MtM = M_k.T @ M_k + lam_Q * identity_n                  # (n, n) SPD
            Mtb = M_k.T @ b_k                                        # (n,)

            # Condition number of regularised system (Correction 4: lstsq fallback)
            # compute_cond=False skips SVD-based cond (O(n^3)); returns nan for speed.
            cond_Areg = float(np.linalg.cond(MtM)) if self.compute_cond else np.nan
            cond_Areg_list.append(cond_Areg)

            try:
                alpha_k = sla.solve(MtM, Mtb, assume_a='pos')       # (n,)
                if not np.isfinite(alpha_k).all():
                    raise np.linalg.LinAlgError("solve returned NaN/Inf")
            except (np.linalg.LinAlgError, Exception):
                # Fallback: lstsq on the underdetermined system M_k α = b_k
                # (not the normal equations) — more numerically stable when cond>>1
                alpha_k, _, _, _ = np.linalg.lstsq(M_k, b_k, rcond=None)

            alpha_list.append(alpha_k)

            # ── Post-fit diagnostics on training data ─────────────────────────
            q_train = K_ZA @ alpha_k                                 # (n,)

            # Riesz moment g=1: E_n[K_h(A-a) * q̂_a]  (should ≈ 1)
            riesz_g1_list.append(float(np.mean(Lambda_k * q_train)))

            # Relative residual ‖M_k α̂_k - b_k‖ / ‖b_k‖
            residual = M_k @ alpha_k - b_k
            b_norm = float(np.linalg.norm(b_k))
            res_rel = float(np.linalg.norm(residual)) / (b_norm + 1e-15)
            residual_norm_rel_list.append(res_rel)

            # Condition number of M_k (diagnostic — not used in solve)
            # compute_cond=False skips SVD-based cond (O(n^3)); returns nan for speed.
            cond_M_list.append(float(np.linalg.cond(M_k)) if self.compute_cond else np.nan)

            # Fraction of negative predictions on training data
            negative_share_list.append(float(np.mean(q_train < 0.0)))

            # p99 of |q̂| on training data
            q_raw_p99_list.append(float(np.percentile(np.abs(q_train), 99)))

        # Cache fitted state
        self._alpha_list = alpha_list
        self._Z_train = Z.copy()
        self._A_train = A.copy()
        self._n_train = n
        self.ell_W = ell_W
        self.ell_A = ell_A
        self.ell_Z = ell_Z
        self._riesz_g1_list = riesz_g1_list
        self._residual_norm_rel = residual_norm_rel_list
        self._cond_M_list = cond_M_list
        self._cond_Areg_list = cond_Areg_list
        self._negative_share_list = negative_share_list
        self._q_raw_p99_list = q_raw_p99_list
        self._fitted = True
        return self

    # ── Prediction ───────────────────────────────────────────────────────────

    def _q_raw_one(
        self,
        Z: np.ndarray,
        A: np.ndarray,
        dose_index: int,
    ) -> np.ndarray:
        """Unclipped q̂_a for dose a_grid[dose_index]. Shape (n_test,)."""
        if not self._fitted:
            raise RuntimeError("KPVPolicyBridgeQ.predict called before fit().")
        Z = np.asarray(Z, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        K_ZA_test = (
            _rbf_gram(Z, self._Z_train, self.ell_Z)
            * _rbf_gram(A, self._A_train, self.ell_A)
        )                                                             # (n_test, n)
        return K_ZA_test @ self._alpha_list[dose_index]              # (n_test,)

    def _apply_clip(self, q_raw: np.ndarray) -> np.ndarray:
        """
        Clip q̂ from above at M_n (Mode 2: raw target / clipped q).
        No lower-bound clipping by default (negative values are diagnosed,
        not removed — they are informative about estimation quality).
        """
        M_n = self.clip if self.clip is not None else 5.0 * self._n_train ** 0.25
        q_clipped = np.clip(q_raw, -np.inf, M_n)
        frac = float(np.mean(q_raw > M_n))
        if frac > 0.10:
            warnings.warn(
                f"KPVPolicyBridgeQ: {100*frac:.1f}% of q̂ clipped at M_n={M_n:.2f}. "
                "Possible overlap issue — check ESS_grid.",
                RuntimeWarning,
            )
        return q_clipped

    def predict(
        self,
        Z: np.ndarray,
        A: np.ndarray,
        dose_index: int,
    ) -> np.ndarray:
        """
        Return q̂_a(Z_i, A_i) at a_grid[dose_index]. Shape (n_test,).

        DRKernel contract: DRKernel multiplies this by K_h(A_i - a) to form the
        importance weight. Do NOT multiply by K_h here (would be double π).
        """
        return self._apply_clip(self._q_raw_one(Z, A, dose_index))

    def predict_raw(
        self,
        Z: np.ndarray,
        A: np.ndarray,
        dose_index: int,
    ) -> np.ndarray:
        """Return unclipped q̂_a(Z_i, A_i) at dose_index. Shape (n_test,)."""
        return self._q_raw_one(Z, A, dose_index)

    def predict_all(
        self,
        Z: np.ndarray,
        A: np.ndarray,
    ) -> np.ndarray:
        """
        Return q̂_a(Z_i, A_i) for ALL doses (clipped). Shape (n_test, K).
        Precomputes K_ZA_test once for efficiency.
        """
        if not self._fitted:
            raise RuntimeError("KPVPolicyBridgeQ.predict_all called before fit().")
        Z = np.asarray(Z, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        n_test = len(Z)
        K = len(self.a_grid)

        K_ZA_test = (
            _rbf_gram(Z, self._Z_train, self.ell_Z)
            * _rbf_gram(A, self._A_train, self.ell_A)
        )                                                             # (n_test, n)

        q_all = np.zeros((n_test, K))
        for d in range(K):
            q_raw = K_ZA_test @ self._alpha_list[d]                 # (n_test,)
            q_all[:, d] = self._apply_clip(q_raw)
        return q_all

    def predict_raw_all(
        self,
        Z: np.ndarray,
        A: np.ndarray,
    ) -> np.ndarray:
        """
        Return unclipped q̂_a(Z_i, A_i) for ALL doses. Shape (n_test, K).
        Used by DRKernel to compute ESS_raw_grid diagnostics.
        """
        if not self._fitted:
            raise RuntimeError("KPVPolicyBridgeQ.predict_raw_all called before fit().")
        Z = np.asarray(Z, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        n_test = len(Z)
        K = len(self.a_grid)

        K_ZA_test = (
            _rbf_gram(Z, self._Z_train, self.ell_Z)
            * _rbf_gram(A, self._A_train, self.ell_A)
        )
        q_all = np.zeros((n_test, K))
        for d in range(K):
            q_all[:, d] = K_ZA_test @ self._alpha_list[d]
        return q_all
