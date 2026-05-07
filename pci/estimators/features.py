"""
features.py -- Phase 11 utility module.

Provides three building blocks for KallusStabilizedPolicyQ (kallus_minimax.py):

1. NystromFeatureMap     -- whitened Nystroem feature map for an RBF kernel.
2. PairFeatureMap        -- builds joint features on two coordinates by
                            concatenating two NystromFeatureMaps and an
                            optional small set of simple cross terms.
3. PolicyFeatureIntegrator -- computes T_pi Phi(W, .) by trapezoidal quadrature
                              against the Gaussian policy kernel
                              K_h(t-a) = exp(-(t-a)^2/(2 h^2)) / (h sqrt(2 pi)).

Design notes
------------
* All maps fit on the train fold only. `fit(X)` stores landmarks, the whitening
  matrix W = U @ diag(1/sqrt(S)), and (optionally) the StandardScaler. The
  test fold is transformed via `transform(X)` using the same fitted state.
  This is essential for cross-fitting: no leakage.
* Whitening guarantees Phi.T @ Phi / n_train ~ I (up to landmark sub-sampling
  noise). This makes the ridge penalty `gamma * ||alpha||^2` defensible as a
  proxy for the RKHS norm.
* All matrix solves use scipy.linalg with explicit jitter; no np.linalg.inv.

References
----------
- Drineas & Mahoney (2005), "On the Nystroem Method for Approximating a Gram
  Matrix for Improved Kernel-Based Learning". JMLR 6:2153-2175.
- Williams & Seeger (2001), "Using the Nystroem Method to Speed Up Kernel
  Machines". NIPS 13.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import scipy.linalg as sla


# ── RBF Gram (private, mirrors kpv_bridge._rbf_gram for 1-D arrays) ───────────

def _rbf_gram_1d(x: np.ndarray, y: np.ndarray, ell: float) -> np.ndarray:
    """RBF Gram matrix K[i, j] = exp(-(x_i - y_j)^2 / (2 ell^2)).

    Both x and y are flat 1-D arrays. Returns shape (len(x), len(y)).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if ell <= 0:
        raise ValueError(f"ell must be > 0, got {ell}.")
    diff = x[:, None] - y[None, :]
    return np.exp(-0.5 * (diff / ell) ** 2)


# ══════════════════════════════════════════════════════════════════════════════
#  NystromFeatureMap
# ══════════════════════════════════════════════════════════════════════════════

class NystromFeatureMap:
    """
    Whitened Nystroem feature map for the RBF kernel
        k(s, t) = exp(-(s - t)^2 / (2 ell^2))

    fit(X):
        1. Sample n_features landmarks uniformly without replacement from X.
        2. Compute K_MM = k(landmarks, landmarks) + jitter * I  (m, m).
        3. Symmetric eigendecomposition: K_MM = U diag(S) U^T.
        4. Clip eigenvalues at eig_clip to control conditioning.
        5. Store whitening: W_white = U @ diag(1/sqrt(S_clipped))   (m, m).
        6. Optionally fit a StandardScaler on Phi(X_train).

    transform(X):
        K_XM = k(X, landmarks)               (n, m)
        Phi  = K_XM @ W_white                (n, m)   whitened features
        (optional) Phi = (Phi - mu) / sigma  via the fitted StandardScaler

    The whitening choice gives Phi.T @ Phi / n_train ~ I (up to
    landmark/projection noise), so ridge regularisation `gamma * ||alpha||^2`
    is a defensible proxy for the RKHS norm.

    Parameters
    ----------
    ell        : RBF length-scale.
    n_features : number of Nystroem landmarks (m).
    jitter     : added to diag(K_MM) before eigendecomposition.
    eig_clip   : minimum eigenvalue of K_MM kept (smaller values are clipped to
                 this floor). Controls conditioning of W_white.
    seed       : RNG seed for landmark sub-sampling.
    scaler     : if True, additionally standardise Phi(X_train) (zero-mean,
                 unit-variance per coordinate).
    """

    def __init__(
        self,
        ell: float,
        n_features: int,
        jitter: float = 1e-8,
        eig_clip: float = 1e-10,
        seed: int = 42,
        scaler: bool = False,
    ) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}.")
        self.ell        = float(ell)
        self.n_features = int(n_features)
        self.jitter     = float(jitter)
        self.eig_clip   = float(eig_clip)
        self.seed       = int(seed)
        self.scaler     = bool(scaler)

        # Fitted state
        self._fitted: bool = False
        self._landmarks: Optional[np.ndarray] = None       # (m,)
        self._W_white:   Optional[np.ndarray] = None       # (m, m)
        self._mean:      Optional[np.ndarray] = None       # (m,)
        self._std:       Optional[np.ndarray] = None       # (m,)
        self._n_train:   int = 0

    # ── fit / transform ───────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "NystromFeatureMap":
        X = np.asarray(X, dtype=float).ravel()
        n = len(X)
        if n < 2:
            raise ValueError(f"Need at least 2 training points, got {n}.")
        m = min(self.n_features, n)
        if m < self.n_features:
            # Asked for more landmarks than available; degrade gracefully.
            self.n_features = m

        rng = np.random.default_rng(self.seed)
        idx = rng.choice(n, size=m, replace=False)
        landmarks = X[idx]

        K_MM = _rbf_gram_1d(landmarks, landmarks, self.ell)
        K_MM = K_MM + self.jitter * np.eye(m)

        # Symmetric eigendecomposition (eigenvalues ascending)
        S, U = sla.eigh(K_MM, check_finite=False)
        # Clip from below to avoid divide-by-zero when whitening
        S_clip = np.maximum(S, self.eig_clip)
        # W_white = U @ diag(1/sqrt(S_clip))
        W_white = U * (1.0 / np.sqrt(S_clip))[None, :]      # (m, m)

        self._landmarks = landmarks
        self._W_white   = W_white
        self._n_train   = n

        if self.scaler:
            Phi_train = self._raw_transform(X)
            self._mean = Phi_train.mean(axis=0)
            self._std  = Phi_train.std(axis=0)
            self._std[self._std < 1e-12] = 1.0

        self._fitted = True
        return self

    def _raw_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform without the final scaler step (used internally)."""
        if not self._fitted:
            raise RuntimeError("NystromFeatureMap.transform called before fit().")
        X = np.asarray(X, dtype=float).ravel()
        K_XM = _rbf_gram_1d(X, self._landmarks, self.ell)   # (n, m)
        return K_XM @ self._W_white                          # (n, m)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Phi = self._raw_transform(X)
        if self.scaler and self._mean is not None:
            Phi = (Phi - self._mean) / self._std
        return Phi

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ══════════════════════════════════════════════════════════════════════════════
#  PairFeatureMap
# ══════════════════════════════════════════════════════════════════════════════

class PairFeatureMap:
    """
    Joint feature map on two scalar coordinates (X1, X2).

    Construction:
        Phi_1 = NystromFeatureMap(ell_1, n_features_1).fit_transform(X1)   (n, m1)
        Phi_2 = NystromFeatureMap(ell_2, n_features_2).fit_transform(X2)   (n, m2)
        Phi   = [Phi_1, Phi_2, Phi_1[:, :n_cross] * Phi_2[:, :n_cross]]    (n, m_total)

    Optionally augmented with simple polynomial features:
        [1, X1, X2, X1*X2]                                                  (n, +4)

    The cross block uses only the first `n_cross` columns of each Nystroem map
    (so the joint feature dimension stays bounded). `n_cross` defaults to
    min(n_features_1, n_features_2, 32).

    This is a deliberately lightweight joint feature set -- not a full tensor
    product -- because tensor products explode the dimension and KPV-q bridges
    typically need only a few interaction directions.

    Parameters
    ----------
    ell_1, ell_2          : RBF length-scales for X1, X2.
    n_features_1/2        : Nystroem dimensions for X1, X2.
    n_cross               : number of element-wise cross-terms; 0 to disable.
    augment_simple        : if True, append [1, X1, X2, X1*X2].
    seed                  : seeds for the two NystromFeatureMaps (X1 uses seed,
                            X2 uses seed+1).
    scaler                : if True, standardise the final concatenated features.
    jitter, eig_clip      : passed through to each NystromFeatureMap.
    """

    def __init__(
        self,
        ell_1: float,
        ell_2: float,
        n_features_1: int,
        n_features_2: int,
        n_cross: Optional[int] = None,
        augment_simple: bool = True,
        seed: int = 42,
        scaler: bool = False,
        jitter: float = 1e-8,
        eig_clip: float = 1e-10,
    ) -> None:
        self.ell_1          = float(ell_1)
        self.ell_2          = float(ell_2)
        self.n_features_1   = int(n_features_1)
        self.n_features_2   = int(n_features_2)
        self.augment_simple = bool(augment_simple)
        self.seed           = int(seed)
        self.scaler         = bool(scaler)
        self.jitter         = float(jitter)
        self.eig_clip       = float(eig_clip)

        if n_cross is None:
            n_cross = min(self.n_features_1, self.n_features_2, 32)
        self.n_cross = max(0, int(n_cross))

        # Fitted sub-maps
        self._map_1: Optional[NystromFeatureMap] = None
        self._map_2: Optional[NystromFeatureMap] = None
        self._mean:  Optional[np.ndarray] = None
        self._std:   Optional[np.ndarray] = None
        self._fitted: bool = False

    @property
    def n_features_total(self) -> int:
        if not self._fitted:
            raise RuntimeError("PairFeatureMap.n_features_total before fit().")
        n = self._map_1.n_features + self._map_2.n_features + self.n_cross
        if self.augment_simple:
            n += 4
        return n

    def fit(self, X1: np.ndarray, X2: np.ndarray) -> "PairFeatureMap":
        if len(X1) != len(X2):
            raise ValueError("X1 and X2 must have the same length.")
        self._map_1 = NystromFeatureMap(
            ell=self.ell_1, n_features=self.n_features_1,
            jitter=self.jitter, eig_clip=self.eig_clip,
            seed=self.seed, scaler=False,
        ).fit(X1)
        self._map_2 = NystromFeatureMap(
            ell=self.ell_2, n_features=self.n_features_2,
            jitter=self.jitter, eig_clip=self.eig_clip,
            seed=self.seed + 1, scaler=False,
        ).fit(X2)
        # Cap n_cross at actual fitted dims
        self.n_cross = min(self.n_cross,
                           self._map_1.n_features,
                           self._map_2.n_features)

        self._fitted = True
        if self.scaler:
            Phi_train = self._raw_transform(X1, X2)
            self._mean = Phi_train.mean(axis=0)
            self._std  = Phi_train.std(axis=0)
            self._std[self._std < 1e-12] = 1.0
        return self

    def _raw_transform(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PairFeatureMap.transform before fit().")
        X1 = np.asarray(X1, dtype=float).ravel()
        X2 = np.asarray(X2, dtype=float).ravel()
        Phi_1 = self._map_1.transform(X1)              # (n, m1)
        Phi_2 = self._map_2.transform(X2)              # (n, m2)
        blocks = [Phi_1, Phi_2]
        if self.n_cross > 0:
            blocks.append(Phi_1[:, : self.n_cross] * Phi_2[:, : self.n_cross])
        if self.augment_simple:
            n = len(X1)
            simple = np.column_stack([
                np.ones(n),
                X1,
                X2,
                X1 * X2,
            ])
            blocks.append(simple)
        return np.concatenate(blocks, axis=1)

    def transform(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        Phi = self._raw_transform(X1, X2)
        if self.scaler and self._mean is not None:
            Phi = (Phi - self._mean) / self._std
        return Phi

    def fit_transform(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        self.fit(X1, X2)
        return self.transform(X1, X2)

    # ── Per-point T_pi for h-side (Phase 11B) ─────────────────────────────────

    def compute_TPI(
        self,
        X1: np.ndarray,
        a_target: float,
        integrator: "PolicyFeatureIntegrator",
    ) -> np.ndarray:
        """
        Per-point T_pi Phi(X1, .) at a_target, returning shape (n, m_total).

        Same separable decomposition as `compute_TPI_mean` (which is the column
        mean of this matrix), but materialises the full (n, m_total) array.
        Required for h-side `plug_in_policy`, where each test W_i needs its
        own T_pi Phi(W_i, .).

        Identity:
            compute_TPI_mean(X1, a, integ) == compute_TPI(X1, a, integ).mean(axis=0)

        ``scaler`` must be False at construction (the train-side scaler does
        not commute with T_pi). KallusMinimaxBridgeH instantiates Phi_H with
        scaler=False, so this is safe in production.
        """
        if not self._fitted:
            raise RuntimeError("PairFeatureMap.compute_TPI before fit().")
        if self.scaler:
            raise NotImplementedError(
                "compute_TPI requires scaler=False (train-side standardisation "
                "does not commute with T_pi)."
            )
        X1 = np.asarray(X1, dtype=float).ravel()
        n = len(X1)
        a = float(a_target)

        # Block 1: Phi_1(X1) -- independent of t under T_pi
        Phi_1 = self._map_1.transform(X1)             # (n, m1)

        # Block 2: T_pi Phi_2(.) at a -- vector of length m2 (cached on integrator)
        Phi_2_TPI = integrator.policy_integral_of_map(self._map_2, a)  # (m2,)
        # Broadcast to (n, m2) -- every row identical (Phi_2 doesn't depend on X1)
        Phi_2_block = np.broadcast_to(Phi_2_TPI[None, :], (n, len(Phi_2_TPI)))

        blocks: list = [Phi_1, Phi_2_block]

        # Cross block: T_pi (Phi_1[:nc] * Phi_2[:nc]) = Phi_1[:nc] * Phi_2_TPI[:nc]
        if self.n_cross > 0:
            cross = Phi_1[:, : self.n_cross] * Phi_2_TPI[: self.n_cross]   # (n, n_cross)
            blocks.append(cross)

        # Simple block [1, X1, X2, X1*X2]:
        #   T_pi 1 = 1                 -> column of ones
        #   T_pi X1 = X1               -> X1 column
        #   T_pi X2 = a_target         -> column of a_target
        #   T_pi X1*X2 = X1 * a_target -> column X1 * a_target
        if self.augment_simple:
            simple = np.column_stack([
                np.ones(n),
                X1,
                np.full(n, a),
                X1 * a,
            ])
            blocks.append(simple)

        return np.concatenate(blocks, axis=1)

    # ── Closed-form policy integral (Phase 11 optimisation) ───────────────────

    def compute_TPI_mean(
        self,
        X1: np.ndarray,
        a_target: float,
        integrator: "PolicyFeatureIntegrator",
    ) -> np.ndarray:
        """
        Compute  E_X1[ T_pi Phi(X1, .) ]  at a_target, exploiting the
        separable block structure.

        Equivalent to (but ~100x faster than)::

            def feat_fn(W_in, t):
                t_arr = np.full(len(W_in), t)
                return self.transform(W_in, t_arr)
            T = integrator.integrate(feat_fn, X1, a_target)   # (n, m_total)
            return T.mean(axis=0)                              # (m_total,)

        Math: each block of `transform(X1, X2)` is either
          - independent of X2 (Phi_1 block, simple [1, X1] terms): T_pi acts as identity
          - independent of X1 (Phi_2 block, simple [X2] term): T_pi gives a vector via
            quadrature against the Gaussian policy kernel
          - bilinear product (cross block, simple [X1*X2] term): T_pi factorises by linearity

        Only the X2-dependent quadrature is non-trivial -- it reduces to a single
        (G, m2) RBF Gram + a (G,) policy weights inner product.

        ``scaler`` must be False at construction (the train-side scaler does
        not commute with T_pi). KallusStabilizedPolicyQ instantiates Phi_H with
        scaler=False, so this is safe in production.
        """
        if not self._fitted:
            raise RuntimeError("PairFeatureMap.compute_TPI_mean before fit().")
        if self.scaler:
            raise NotImplementedError(
                "compute_TPI_mean requires scaler=False (train-side standardisation "
                "does not commute with T_pi)."
            )
        X1 = np.asarray(X1, dtype=float).ravel()
        n = len(X1)
        a = float(a_target)

        # Block 1: Phi_1(X1) -- independent of t under T_pi
        Phi_1 = self._map_1.transform(X1)             # (n, m1)
        Phi_1_mean = Phi_1.mean(axis=0)               # (m1,)

        # Block 2: T_pi Phi_2(.) at a -- vector of length m2
        Phi_2_TPI = integrator.policy_integral_of_map(self._map_2, a)  # (m2,)

        blocks = [Phi_1_mean, Phi_2_TPI]

        # Cross block: T_pi (Phi_1[:nc] * Phi_2[:nc]) = Phi_1[:nc] * Phi_2_TPI[:nc]
        if self.n_cross > 0:
            cross_mean = Phi_1_mean[: self.n_cross] * Phi_2_TPI[: self.n_cross]
            blocks.append(cross_mean)

        # Simple block [1, X1, X2, X1*X2]:
        #   T_pi 1 = 1
        #   T_pi X1 = X1                  -> mean is X1.mean()
        #   T_pi X2 = a_target            (closed-form: integral of t against K_h(t-a))
        #   T_pi X1*X2 = X1 * a_target    -> mean is X1.mean() * a_target
        if self.augment_simple:
            X1_mean = float(X1.mean())
            simple_mean = np.array([1.0, X1_mean, a, X1_mean * a])
            blocks.append(simple_mean)

        return np.concatenate(blocks)


# ══════════════════════════════════════════════════════════════════════════════
#  PolicyFeatureIntegrator
# ══════════════════════════════════════════════════════════════════════════════

class PolicyFeatureIntegrator:
    """
    Compute T_pi Phi(W, .) by trapezoidal quadrature against the Gaussian
    policy kernel
        K_h(t - a) = exp(-(t - a)^2 / (2 h^2)) / (h sqrt(2 pi)).

    Concretely, for a target dose `a_target` and a feature function
        feature_fn(W, t) -> array of shape (n, m_features)
    where W is a fixed (n,) array of training W's and t is a scalar, this
    returns the integrals
        T[i, k] = integral_t  feature_fn(W_i, t)[k]  K_h(t - a_target)  dt
    approximated on a fixed quadrature grid.

    The grid is chosen at fit_grid time:
        grid = linspace(grid_min, grid_max, grid_size)
        weights = K_h(grid - a_target) * dt   (renormalised to sum to 1 per dose)

    Parameters
    ----------
    h_policy   : Gaussian bandwidth (the same as DRKernel.h_KDE).
    grid_size  : number of quadrature nodes.
    grid_range : tuple (a_min, a_max) or "auto" -- "auto" picks
                 [A_train.min() - 4*h_policy, A_train.max() + 4*h_policy].
    """

    def __init__(
        self,
        h_policy: float,
        grid_size: int = 400,
        grid_range: Tuple = ("auto", "auto"),
    ) -> None:
        if h_policy <= 0:
            raise ValueError(f"h_policy must be > 0, got {h_policy}.")
        if grid_size < 8:
            raise ValueError(f"grid_size must be >= 8, got {grid_size}.")
        self.h_policy   = float(h_policy)
        self.grid_size  = int(grid_size)
        self.grid_range = grid_range

        # Fitted state
        self._fitted: bool = False
        self._grid: Optional[np.ndarray] = None             # (grid_size,)
        self._dt:   float = 0.0                              # uniform spacing

    def fit_grid(self, A_train: np.ndarray) -> "PolicyFeatureIntegrator":
        A_train = np.asarray(A_train, dtype=float).ravel()
        a_min, a_max = self.grid_range
        if a_min == "auto":
            a_min = float(A_train.min()) - 4.0 * self.h_policy
        if a_max == "auto":
            a_max = float(A_train.max()) + 4.0 * self.h_policy
        if a_max <= a_min:
            raise ValueError(f"Empty grid range: a_min={a_min}, a_max={a_max}.")
        self._grid = np.linspace(float(a_min), float(a_max), self.grid_size)
        self._dt   = float(self._grid[1] - self._grid[0])
        self._fitted = True
        return self

    def _policy_weights(self, a_target: float) -> np.ndarray:
        """Trapezoidal weights = K_h(grid - a) * dt, with endpoints halved."""
        if not self._fitted:
            raise RuntimeError("PolicyFeatureIntegrator.fit_grid not called.")
        u = (self._grid - a_target) / self.h_policy
        K = np.exp(-0.5 * u * u) / (self.h_policy * np.sqrt(2.0 * np.pi))
        w = K * self._dt
        # Trapezoid endpoint correction (negligible for typical grids but
        # tightens T_pi 1 = 1 sanity check).
        w[0]  *= 0.5
        w[-1] *= 0.5
        return w

    def integrate(
        self,
        feature_fn: Callable[[np.ndarray, float], np.ndarray],
        W: np.ndarray,
        a_target: float,
    ) -> np.ndarray:
        """
        Compute T_pi Phi(W, .) at a_target.

        Parameters
        ----------
        feature_fn : callable (W, t) -> np.ndarray of shape (len(W), m).
                     For each scalar t in the grid, must return the matrix of
                     feature values at (W_i, t) for all i.
        W          : (n,) array of W values for which to compute the integrals.
        a_target   : scalar dose at which to weight by K_h.

        Returns
        -------
        T : (n, m) array of integrated features.
        """
        if not self._fitted:
            raise RuntimeError("PolicyFeatureIntegrator.fit_grid not called.")
        W = np.asarray(W, dtype=float).ravel()
        weights = self._policy_weights(float(a_target))     # (G,)
        # Lazy-build T by accumulation -- we don't know m until first call.
        T: Optional[np.ndarray] = None
        for k, t in enumerate(self._grid):
            phi_t = feature_fn(W, float(t))                  # (n, m)
            if T is None:
                T = np.zeros((len(W), phi_t.shape[1]), dtype=float)
            T += weights[k] * phi_t
        return T

    def policy_integral_of_map(
        self,
        nystrom_map: "NystromFeatureMap",
        a_target: float,
    ) -> np.ndarray:
        """
        Compute  T_pi Phi(.) = integral_t Phi(t) K_h(t-a) dt
        for a fitted NystromFeatureMap, returning a vector of length m_features.

        Uses a *single* RBF Gram on the quadrature grid (cached per
        NystromFeatureMap instance), so repeat calls at different a_target
        only re-compute the policy weights and the (G,) @ (G, m) inner product.
        """
        if not self._fitted:
            raise RuntimeError("PolicyFeatureIntegrator.fit_grid not called.")
        if not nystrom_map._fitted:
            raise RuntimeError(
                "policy_integral_of_map called on un-fitted NystromFeatureMap."
            )
        # Cache Phi(grid) on the integrator (keyed by id(nystrom_map))
        cache_key = id(nystrom_map)
        cache = getattr(self, "_phi_grid_cache", None)
        if cache is None:
            cache = {}
            self._phi_grid_cache = cache
        if cache_key not in cache:
            # NystromFeatureMap may have a scaler -- we want the SAME features
            # the user gets via .transform() so just call .transform on the grid
            cache[cache_key] = nystrom_map.transform(self._grid)   # (G, m)

        Phi_grid = cache[cache_key]                                # (G, m)
        weights  = self._policy_weights(float(a_target))           # (G,)
        return weights @ Phi_grid                                  # (m,)

    def integrate_separable(
        self,
        Phi_W: np.ndarray,
        feature_A_fn: Callable[[float], np.ndarray],
        a_target: float,
    ) -> np.ndarray:
        """
        Specialised, faster integrator for *separable* features of the form
            phi(W, t)[k] = Phi_W[i, k_W] * Phi_A(t)[k_A]
        encoded as a Khatri-Rao / element-wise product after concat.

        For our use case (PairFeatureMap on (W, A)), most blocks are *not*
        strictly separable. Use the general `integrate` method instead.
        Provided here for future closed-form extensions.
        """
        if not self._fitted:
            raise RuntimeError("PolicyFeatureIntegrator.fit_grid not called.")
        weights = self._policy_weights(float(a_target))
        accum = None
        for k, t in enumerate(self._grid):
            phi_a_t = feature_A_fn(float(t))                 # (m_A,)
            if accum is None:
                accum = np.zeros_like(phi_a_t, dtype=float)
            accum += weights[k] * phi_a_t
        return Phi_W * accum[None, :]                        # (n, m_W * m_A) loosely
