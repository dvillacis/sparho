"""Native FISTA solver for the Group-L1 Lasso (no external dependencies).

scikit-learn ships no Group Lasso and celer's group solver is gated behind
the optional ``[celer]`` extra; this module is the canonical built-in inner
solver for ``Problem(SquaredLoss, GroupL1, X, y)``. The algorithm is FISTA
(Beck-Teboulle 2009) with a fixed step ``1/L`` where ``L = ‖X‖²_op / n`` is
estimated by power iteration. The prox is :func:`sparho._core.prox_group_l1`
(Rust). Convergence is declared on the relative step ``‖β_new − β_old‖_∞``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .. import _core
from ..core.types import Array, DesignMatrix, Hyperparam
from ..problem import GroupL1, Problem, SquaredLoss
from ..state import SolverResult
from ._common import active_set_of, as_scalar


@dataclass(frozen=True, slots=True)
class GroupLassoFista:
    """FISTA adapter for ``Problem(SquaredLoss, GroupL1, X, y)``.

    Pass an explicit ``lipschitz=L`` to skip the per-call power-iteration
    estimate when fitting many times against the same design — typical for
    bilevel outer searches that re-solve the inner problem at neighbouring
    ``α`` values. ``dual_gap`` is reported as the worst-group KKT
    stationarity residual; zero at an exact optimum.
    """

    tol: float = 1e-6
    max_iter: int = 10_000
    lipschitz: float | None = None
    lipschitz_iter: int = 20
    lipschitz_seed: int = 0

    def __call__(
        self,
        problem: Problem,
        hyperparam: Hyperparam,
        /,
        *,
        x0: Array | None = None,
        tol: float | None = None,
    ) -> SolverResult:
        if not isinstance(problem.datafit, SquaredLoss) or not isinstance(
            problem.penalty, GroupL1
        ):
            raise TypeError("GroupLassoFista requires Problem(SquaredLoss, GroupL1, ...)")
        alpha = as_scalar(hyperparam)
        if alpha <= 0:
            raise ValueError("alpha must be strictly positive")
        penalty = problem.penalty
        X = problem.design
        y = np.asarray(problem.target, dtype=np.float64)
        n_samples, n_features = X.shape
        eff_tol = float(self.tol if tol is None else tol)

        weights, group_ptr, group_indices = _group_layout(penalty)

        L = (
            float(self.lipschitz)
            if self.lipschitz is not None
            else _lipschitz_estimate(X, n_samples, self.lipschitz_iter, self.lipschitz_seed)
        )
        step = 1.0 / L
        thr = alpha * step

        beta = (
            np.ascontiguousarray(np.asarray(x0, dtype=np.float64))
            if x0 is not None
            else np.zeros(n_features, dtype=np.float64)
        )
        if beta.shape != (n_features,):
            raise ValueError(f"x0 must have shape ({n_features},), got {beta.shape}")
        y_iter = beta.copy()
        t = 1.0

        n_inv = 1.0 / n_samples
        n_iter = 0
        for k in range(1, self.max_iter + 1):
            n_iter = k
            grad = _rmatvec(X, _matvec(X, y_iter) - y) * n_inv
            z = y_iter - step * grad
            beta_new = _core.prox_group_l1(
                np.ascontiguousarray(z), thr, weights, group_ptr, group_indices
            )
            diff = beta_new - beta
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y_iter = beta_new + ((t - 1.0) / t_new) * diff
            t = t_new
            max_diff = float(np.abs(diff).max()) if diff.size else 0.0
            max_beta = float(np.abs(beta_new).max()) if beta_new.size else 0.0
            beta = beta_new
            if max_diff <= eff_tol * max(1.0, max_beta):
                break

        gap = _kkt_residual(X, y, beta, alpha, penalty, weights, n_inv)
        return SolverResult(
            coef=np.asarray(beta, dtype=np.float64),
            active_set=active_set_of(beta),
            dual_gap=gap,
            n_iter=int(n_iter),
        )


def _group_layout(penalty: GroupL1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR-style layout consumed by ``_core.prox_group_l1``."""
    sizes = [len(g) for g in penalty.groups]
    group_ptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
    if penalty.groups:
        group_indices = np.concatenate(
            [np.asarray(g, dtype=np.int32) for g in penalty.groups]
        )
    else:
        group_indices = np.zeros(0, dtype=np.int32)
    if penalty.weights is not None:
        weights = np.asarray(penalty.weights, dtype=np.float64)
    else:
        weights = np.array([np.sqrt(s) for s in sizes], dtype=np.float64)
    return weights, group_ptr, group_indices


def _matvec(X: DesignMatrix, v: np.ndarray) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X @ v).ravel()
    return np.asarray(X @ v, dtype=np.float64)


def _rmatvec(X: DesignMatrix, v: np.ndarray) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.T @ v).ravel()
    return np.asarray(X.T @ v, dtype=np.float64)


def _lipschitz_estimate(X: DesignMatrix, n_samples: int, n_iter: int, seed: int) -> float:
    """Power iteration on ``X^T X / n``; returns ``‖X‖²_op / n``."""
    n_features = int(X.shape[1])
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n_features)
    nrm = float(np.linalg.norm(v))
    if nrm == 0.0:
        return 1.0
    v /= nrm
    L = 0.0
    for _ in range(max(n_iter, 1)):
        w = _rmatvec(X, _matvec(X, v))
        nrm = float(np.linalg.norm(w))
        if nrm == 0.0:
            return 1.0
        v = w / nrm
        L = nrm
    return float(L / n_samples + 1e-12)


def _kkt_residual(
    X: DesignMatrix,
    y: np.ndarray,
    beta: np.ndarray,
    alpha: float,
    penalty: GroupL1,
    weights: np.ndarray,
    n_inv: float,
) -> float:
    """Worst-group KKT stationarity violation; zero at an exact optimum."""
    grad = _rmatvec(X, _matvec(X, beta) - y) * n_inv
    worst = 0.0
    for k, g in enumerate(penalty.groups):
        idx = np.fromiter(g, dtype=np.int64, count=len(g))
        beta_g = beta[idx]
        grad_g = grad[idx]
        w_k = float(weights[k])
        norm_beta = float(np.linalg.norm(beta_g))
        if norm_beta > 0:
            kkt = float(np.linalg.norm(grad_g + alpha * w_k * beta_g / norm_beta))
        else:
            kkt = max(0.0, float(np.linalg.norm(grad_g)) - alpha * w_k)
        if kkt > worst:
            worst = kkt
    return worst
