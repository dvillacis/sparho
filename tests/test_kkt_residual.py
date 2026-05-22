"""Post-solve KKT residual assertions across every (Datafit, Penalty) pair.

Every inner solver in sparho should produce a coefficient vector that
satisfies the proximal fixed-point identity ``β = prox_R(β − ∇L(β); α)``
at convergence. :func:`sparho.testing.kkt_residual` evaluates the infinity
norm of the defect; this test asserts the defect is below a tight tolerance
across the closed ``Datafit × Penalty`` matrix.

Three things this guards against:

1. **Solver regressions** — a wrapper/adapter change that loosens inner
   convergence will surface here even if other tests still pass.
2. **Dispatch holes** — the parametrization enumerates every variant of
   the closed unions in :mod:`sparho.problem`; adding a new penalty
   without wiring it through every match statement will make
   :func:`kkt_residual` raise ``assert_never`` at import time.
3. **Prox / kernel drift** — if the Rust prox kernels and the inner
   solver's prox calls fall out of sync, residual jumps even though
   sklearn-side tests still see "convergence".
"""

from __future__ import annotations

import numpy as np
import pytest
from sparho import (
    L1,
    ElasticNet,
    GroupL1,
    LogisticLoss,
    Problem,
    SquaredLoss,
    WeightedL1,
)
from sparho.adapters import (
    GroupLassoFista,
    SklearnElasticNet,
    SklearnLasso,
    SklearnLogisticRegression,
    SklearnWeightedLasso,
)
from sparho.solver import Solver
from sparho.testing import assert_kkt_optimal, kkt_residual

# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def lasso_problem() -> Problem:
    rng = np.random.default_rng(0)
    n, p, k = 80, 30, 5
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:k] = rng.standard_normal(k) + 2.0
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    return Problem(SquaredLoss(), L1(), X, y)


@pytest.fixture(scope="module")
def enet_problem() -> Problem:
    rng = np.random.default_rng(1)
    n, p = 80, 25
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[[1, 5, 11, 17]] = [2.0, -1.5, 1.0, -2.5]
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    return Problem(SquaredLoss(), ElasticNet(rho=0.7), X, y)


@pytest.fixture(scope="module")
def wlasso_problem() -> tuple[Problem, np.ndarray]:
    rng = np.random.default_rng(2)
    n, p = 60, 20
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[[0, 3, 7, 12]] = [1.5, -2.0, 1.0, -1.5]
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    # Per-feature α: same scalar α replicated, exercises the array path.
    alpha = np.full(p, 0.05, dtype=np.float64)
    return Problem(SquaredLoss(), WeightedL1(), X, y), alpha


@pytest.fixture(scope="module")
def grouplasso_problem() -> Problem:
    rng = np.random.default_rng(3)
    n, p = 100, 30
    X = rng.standard_normal((n, p))
    # 10 groups of 3 consecutive features each.
    groups = tuple(tuple(range(3 * k, 3 * (k + 1))) for k in range(10))
    true_beta = np.zeros(p)
    # First two groups active, all others zero — block-sparse signal.
    true_beta[0:3] = rng.standard_normal(3) + 1.5
    true_beta[3:6] = rng.standard_normal(3) - 1.0
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    return Problem(SquaredLoss(), GroupL1(groups=groups), X, y)


@pytest.fixture(scope="module")
def logreg_problem() -> Problem:
    rng = np.random.default_rng(4)
    n, p = 200, 30
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[[1, 4, 9]] = [2.0, -1.5, 1.0]
    logits = X @ true_beta
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = np.where(rng.uniform(size=n) < probs, 1.0, -1.0)
    return Problem(LogisticLoss(), L1(), X, y)


# ---------------------------------------------------------------- tests


def _solve_tight(solver: Solver, problem: Problem, hp: object) -> np.ndarray:
    """Solve to tight inner tolerance and return ``β̂``."""
    res = solver(problem, hp, tol=1e-8)  # type: ignore[arg-type]
    return np.asarray(res.coef, dtype=np.float64)


def test_kkt_l1_lasso(lasso_problem: Problem) -> None:
    alpha = 0.05
    beta = _solve_tight(SklearnLasso(), lasso_problem, alpha)
    assert_kkt_optimal(lasso_problem, alpha, beta, atol=1e-3)


def test_kkt_elastic_net(enet_problem: Problem) -> None:
    alpha = 0.05
    beta = _solve_tight(SklearnElasticNet(), enet_problem, alpha)
    assert_kkt_optimal(enet_problem, alpha, beta, atol=1e-3)


def test_kkt_weighted_l1(wlasso_problem: tuple[Problem, np.ndarray]) -> None:
    problem, alpha = wlasso_problem
    beta = _solve_tight(SklearnWeightedLasso(), problem, alpha)
    assert_kkt_optimal(problem, alpha, beta, atol=1e-3)


def test_kkt_group_l1(grouplasso_problem: Problem) -> None:
    alpha = 0.05
    beta = _solve_tight(GroupLassoFista(max_iter=5000), grouplasso_problem, alpha)
    # GroupLassoFista is FISTA; sublinear convergence relaxes the tol slightly.
    assert_kkt_optimal(grouplasso_problem, alpha, beta, atol=5e-3)


def test_kkt_logistic_l1(logreg_problem: Problem) -> None:
    alpha = 0.05
    beta = _solve_tight(SklearnLogisticRegression(), logreg_problem, alpha)
    # Logistic has a nonlinear ∇L; residual with τ=1 is looser-bounded than
    # squared loss but still drops to zero at optimum.
    assert_kkt_optimal(logreg_problem, alpha, beta, atol=5e-2)


def test_residual_zero_at_zero_for_large_alpha(lasso_problem: Problem) -> None:
    """At ``α > α_max = (1/n) ‖X^T y‖_∞``, ``β* = 0`` and KKT residual is 0."""
    X, y = lasso_problem.design, lasso_problem.target
    n = lasso_problem.n_samples
    alpha_max = float(np.max(np.abs(X.T @ y))) / n
    alpha = 2.0 * alpha_max  # comfortably above
    beta_zero = np.zeros(lasso_problem.n_features, dtype=np.float64)
    res = kkt_residual(lasso_problem, alpha, beta_zero)
    assert res == 0.0


def test_residual_strictly_positive_off_optimum(lasso_problem: Problem) -> None:
    """A perturbed β has strictly positive residual — sanity check the formula."""
    alpha = 0.05
    beta = _solve_tight(SklearnLasso(), lasso_problem, alpha)
    perturbed = beta + 0.1
    res_opt = kkt_residual(lasso_problem, alpha, beta)
    res_per = kkt_residual(lasso_problem, alpha, perturbed)
    assert res_per > res_opt
    assert res_per > 1e-2
