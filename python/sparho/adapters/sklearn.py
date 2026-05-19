"""scikit-learn adapters.

Each adapter is a frozen dataclass with a ``__call__(problem, hyperparam) ->
SolverResult`` method. The class holds inner-solver configuration (``tol``,
``max_iter``); the call accepts the current hyperparameter from the outer
search.

All adapters force ``fit_intercept=False`` because ``Problem`` has no intercept
slot; if a user wants centered data they should center it before constructing
the ``Problem``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression

from ..core.types import Array, Hyperparam
from ..problem import (
    L1,
    LogisticLoss,
    Problem,
    SquaredLoss,
    WeightedL1,
)
from ..problem import (
    ElasticNet as ElasticNetPenalty,
)
from ..state import SolverResult
from ._common import active_set_of, as_scalar, as_vector

# ---------------------------------------------------------------- Lasso


@dataclass(frozen=True, slots=True)
class SklearnLasso:
    """Adapter for ``Problem(SquaredLoss, L1, X, y)`` via ``sklearn.linear_model.Lasso``.

    When the criterion uses warm-start, prefer ``tol ≤ 1e-8``: sklearn's
    convergence check is ``dual_gap < tol · ||y||²``, so on small-``||y||``
    problems a loose ``tol`` lets the warm coef pass the check immediately and
    the inner solver returns it unchanged — which makes nearby CV evaluations
    indistinguishable and stalls outer line searches.
    """

    tol: float = 1e-6
    max_iter: int = 10_000

    def __call__(
        self,
        problem: Problem,
        hyperparam: Hyperparam,
        /,
        *,
        x0: Array | None = None,
        tol: float | None = None,
    ) -> SolverResult:
        if not isinstance(problem.datafit, SquaredLoss) or not isinstance(problem.penalty, L1):
            raise TypeError("SklearnLasso requires Problem(SquaredLoss, L1, ...)")
        alpha = as_scalar(hyperparam)
        est = Lasso(
            alpha=alpha,
            fit_intercept=False,
            tol=self.tol if tol is None else float(tol),
            max_iter=self.max_iter,
            selection="cyclic",
            warm_start=x0 is not None,
        )
        if x0 is not None:
            est.coef_ = np.ascontiguousarray(np.asarray(x0, dtype=np.float64))
        est.fit(problem.design, problem.target)
        coef = np.asarray(est.coef_, dtype=np.float64)
        return SolverResult(
            coef=coef,
            active_set=active_set_of(coef),
            dual_gap=float(est.dual_gap_),
            n_iter=int(est.n_iter_),
        )


# ---------------------------------------------------------------- ElasticNet


@dataclass(frozen=True, slots=True)
class SklearnElasticNet:
    """Adapter for ``Problem(SquaredLoss, ElasticNet(rho), X, y)`` via sklearn."""

    tol: float = 1e-6
    max_iter: int = 10_000

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
            problem.penalty, ElasticNetPenalty
        ):
            raise TypeError("SklearnElasticNet requires Problem(SquaredLoss, ElasticNet, ...)")
        alpha = as_scalar(hyperparam)
        est = ElasticNet(
            alpha=alpha,
            l1_ratio=problem.penalty.rho,
            fit_intercept=False,
            tol=self.tol if tol is None else float(tol),
            max_iter=self.max_iter,
            selection="cyclic",
            warm_start=x0 is not None,
        )
        if x0 is not None:
            est.coef_ = np.ascontiguousarray(np.asarray(x0, dtype=np.float64))
        est.fit(problem.design, problem.target)
        coef = np.asarray(est.coef_, dtype=np.float64)
        return SolverResult(
            coef=coef,
            active_set=active_set_of(coef),
            dual_gap=float(est.dual_gap_),
            n_iter=int(est.n_iter_),
        )


# ---------------------------------------------------------------- Weighted Lasso


@dataclass(frozen=True, slots=True)
class SklearnWeightedLasso:
    """Adapter for ``Problem(SquaredLoss, WeightedL1, X, y)`` via column-rescaling.

    sklearn's ``Lasso`` only supports a scalar ``α``. We solve the equivalent
    problem with rescaled design ``X' = X · diag(1/α_vec)`` and unit ``α=1``;
    the recovered ``β'`` is rescaled back as ``β = β' / α_vec``. Coefficients
    where ``α_j = 0`` (no regularization) are not supported in this adapter —
    sparse-ho's behavior was the same.
    """

    tol: float = 1e-6
    max_iter: int = 10_000

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
            problem.penalty, WeightedL1
        ):
            raise TypeError("SklearnWeightedLasso requires Problem(SquaredLoss, WeightedL1, ...)")
        alpha = as_vector(hyperparam, problem.n_features)
        if np.any(alpha <= 0):
            raise ValueError("weighted-Lasso adapter needs strictly positive α_j")
        X = problem.design
        scale = 1.0 / alpha
        if sp.issparse(X):
            # csc_matrix @ diag is column-rescaling; works in O(nnz).
            X_scaled = X @ sp.diags(scale)
        else:
            X_scaled = X * scale  # broadcasts over columns
        est = Lasso(
            alpha=1.0,
            fit_intercept=False,
            tol=self.tol if tol is None else float(tol),
            max_iter=self.max_iter,
            selection="cyclic",
            warm_start=x0 is not None,
        )
        if x0 is not None:
            # β = β' / α  ⇒  β' = α ∘ β.
            est.coef_ = np.ascontiguousarray(np.asarray(x0, dtype=np.float64) * alpha)
        est.fit(X_scaled, problem.target)
        # β' = α ∘ β, so β = β' / α.
        coef = np.asarray(est.coef_, dtype=np.float64) * scale
        return SolverResult(
            coef=coef,
            active_set=active_set_of(coef),
            dual_gap=float(est.dual_gap_),
            n_iter=int(est.n_iter_),
        )


# ---------------------------------------------------------------- Logistic + L1


@dataclass(frozen=True, slots=True)
class SklearnLogisticRegression:
    """Adapter for ``Problem(LogisticLoss, L1, X, y)`` via ``LogisticRegression(penalty='l1')``.

    Assumes binary ``y ∈ {−1, +1}``. sklearn relabels internally. The dual
    gap is not exposed by sklearn for logistic regression; we report a
    stationarity proxy ``||X_A^T (σ(Xβ) − y₀₁) + α sign(β_A)||_∞`` instead,
    which is zero at a KKT-optimal point.

    ``x0`` is accepted for protocol conformance but ignored — sklearn's
    ``LogisticRegression(solver='liblinear')`` does not support warm-start.
    """

    tol: float = 1e-6
    max_iter: int = 10_000

    def __call__(
        self,
        problem: Problem,
        hyperparam: Hyperparam,
        /,
        *,
        x0: Array | None = None,  # noqa: ARG002 — liblinear does not support warm-start
        tol: float | None = None,
    ) -> SolverResult:
        if not isinstance(problem.datafit, LogisticLoss) or not isinstance(problem.penalty, L1):
            raise TypeError("SklearnLogisticRegression requires Problem(LogisticLoss, L1, ...)")
        alpha = as_scalar(hyperparam)
        if alpha <= 0:
            raise ValueError("alpha must be strictly positive")
        # sklearn's L1 LR objective: ||β||₁ + C · Σ log(1 + exp(−y_i Xβ_i));
        # our convention: α||β||₁ + Σ log(1 + exp(−y_i Xβ_i)) ⇒ C = 1/α.
        y = problem.target
        if not np.array_equal(np.unique(y), np.array([-1.0, 1.0])):
            raise ValueError("LogisticLoss expects y ∈ {−1, +1}")
        est = LogisticRegression(
            penalty="l1",
            C=1.0 / alpha,
            fit_intercept=False,
            solver="liblinear",
            tol=self.tol if tol is None else float(tol),
            max_iter=self.max_iter,
        )
        est.fit(problem.design, y)
        coef = np.asarray(est.coef_.ravel(), dtype=np.float64)
        active = active_set_of(coef)
        gap = _logistic_stationarity_gap(problem, coef, active, alpha)
        return SolverResult(
            coef=coef,
            active_set=active,
            dual_gap=gap,
            n_iter=int(np.atleast_1d(est.n_iter_)[0]),
        )


def _logistic_stationarity_gap(
    problem: Problem, coef: np.ndarray, active: np.ndarray, alpha: float
) -> float:
    """``||X_A^T (σ(Xβ) − y₀₁) + α sign(β_A)||_∞`` — zero at KKT optimum."""
    X = problem.design
    y_01 = 0.5 * (problem.target + 1.0)  # {−1, +1} → {0, 1}
    z = X @ coef
    sig = 1.0 / (1.0 + np.exp(-z))
    resid = sig - y_01
    if active.size == 0:
        return 0.0
    if sp.issparse(X):
        grad_A = np.asarray(X[:, active].T @ resid).ravel()
    else:
        grad_A = X[:, active].T @ resid
    return float(np.max(np.abs(grad_A + alpha * np.sign(coef[active]))))
