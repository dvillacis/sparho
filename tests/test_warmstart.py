"""Warm-start tests: the ``x0`` kwarg threads through adapters and CrossVal.

Correctness (warm and cold converge to the same β within tolerance) plus a
sanity check that warm-start actually reduces inner-iter counts when seeded
near the optimum. The big perf wins are validated in
``benchmarks/spike_warmstart.py``; this file only locks in the API.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_regression
from sparho import (
    L1,
    CrossVal,
    ElasticNet,
    LogisticLoss,
    Problem,
    SquaredLoss,
    WeightedL1,
    hoag_search,
    implicit_forward,
)
from sparho.adapters import (
    SklearnElasticNet,
    SklearnLasso,
    SklearnLogisticRegression,
    SklearnWeightedLasso,
    as_solver,
)
from sparho.state import SolverResult


@pytest.fixture(scope="module")
def reg_dense() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=200, n_features=80, n_informative=10, random_state=0)
    return X.astype(np.float64), y.astype(np.float64)


@pytest.fixture(scope="module")
def reg_sparse() -> tuple[sp.csc_matrix, np.ndarray]:
    rng = np.random.default_rng(1)
    X = sp.random(150, 60, density=0.1, format="csc", random_state=rng).astype(np.float64)
    beta = np.zeros(60)
    beta[[3, 7, 12, 30]] = [1.5, -2.0, 0.5, 3.0]
    y = X @ beta + 0.05 * rng.standard_normal(150)
    return X, y


@pytest.fixture(scope="module")
def cls_dense() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((150, 40))
    beta = np.zeros(40)
    beta[[1, 4, 9]] = [2.0, -1.5, 1.0]
    y = np.where(X @ beta > 0, 1.0, -1.0)
    return X, y


# ---------------------------------------------------------------- adapter API


def test_sklearn_lasso_x0_correctness(reg_dense):
    X, y = reg_dense
    problem = Problem(SquaredLoss(), L1(), X, y)
    solver = SklearnLasso(tol=1e-9, max_iter=20_000)
    cold = solver(problem, 0.1)
    seed = cold.coef + 0.01 * np.random.default_rng(0).standard_normal(cold.coef.shape)
    warm = solver(problem, 0.1, x0=seed)
    # Same minimum (Lasso is convex).
    np.testing.assert_allclose(warm.coef, cold.coef, atol=1e-5)
    np.testing.assert_array_equal(warm.active_set, cold.active_set)


def test_sklearn_lasso_x0_reduces_iters(reg_dense):
    X, y = reg_dense
    problem = Problem(SquaredLoss(), L1(), X, y)
    solver = SklearnLasso(tol=1e-8, max_iter=20_000)
    cold = solver(problem, 0.1)
    # Seed from the true optimum → inner solver should converge in fewer iters.
    warm = solver(problem, 0.1, x0=cold.coef)
    assert (
        warm.n_iter < cold.n_iter
    ), f"warm.n_iter={warm.n_iter} not less than cold.n_iter={cold.n_iter}"


def test_sklearn_elastic_net_x0(reg_dense):
    X, y = reg_dense
    problem = Problem(SquaredLoss(), ElasticNet(rho=0.6), X, y)
    solver = SklearnElasticNet(tol=1e-9, max_iter=20_000)
    cold = solver(problem, 0.1)
    warm = solver(problem, 0.1, x0=cold.coef)
    np.testing.assert_allclose(warm.coef, cold.coef, atol=1e-5)


def test_sklearn_weighted_lasso_x0(reg_dense):
    X, y = reg_dense
    n_features = X.shape[1]
    alpha_vec = np.full(n_features, 0.1)
    problem = Problem(SquaredLoss(), WeightedL1(), X, y)
    solver = SklearnWeightedLasso(tol=1e-9, max_iter=20_000)
    cold = solver(problem, alpha_vec)
    warm = solver(problem, alpha_vec, x0=cold.coef)
    np.testing.assert_allclose(warm.coef, cold.coef, atol=1e-5)


def test_sklearn_logistic_x0_accepted_but_ignored(cls_dense):
    X, y = cls_dense
    problem = Problem(LogisticLoss(), L1(), X, y)
    solver = SklearnLogisticRegression(tol=1e-6, max_iter=2_000)
    cold = solver(problem, 0.1)
    # x0 silently ignored (liblinear has no warm-start path); no exception, same answer
    # up to liblinear's own run-to-run jitter.
    seeded = solver(problem, 0.1, x0=cold.coef)
    np.testing.assert_allclose(seeded.coef, cold.coef, atol=1e-3)


def test_sklearn_lasso_x0_sparse(reg_sparse):
    X, y = reg_sparse
    problem = Problem(SquaredLoss(), L1(), X, y)
    solver = SklearnLasso(tol=1e-9, max_iter=20_000)
    cold = solver(problem, 0.05)
    warm = solver(problem, 0.05, x0=cold.coef)
    np.testing.assert_allclose(warm.coef, cold.coef, atol=1e-5)


# ---------------------------------------------------------------- CallableSolver


def test_as_solver_forwards_x0_when_accepted(reg_dense):
    X, y = reg_dense
    problem = Problem(SquaredLoss(), L1(), X, y)
    seen: dict[str, object] = {}

    def fn(p: Problem, hp, *, x0=None) -> SolverResult:
        seen["x0_was"] = None if x0 is None else "array"
        return SklearnLasso()(p, hp, x0=x0)

    s = as_solver(fn, name="probe")
    seed = np.zeros(X.shape[1])
    s(problem, 0.1, x0=seed)
    assert seen["x0_was"] == "array"


def test_as_solver_drops_x0_when_not_accepted(reg_dense):
    X, y = reg_dense
    problem = Problem(SquaredLoss(), L1(), X, y)
    seen: dict[str, object] = {}

    def fn(p: Problem, hp) -> SolverResult:
        seen["called"] = True
        return SklearnLasso()(p, hp)

    s = as_solver(fn, name="legacy")
    # Even passing x0, the wrapper should drop it silently and call fn.
    s(problem, 0.1, x0=np.zeros(X.shape[1]))
    assert seen["called"] is True


# ---------------------------------------------------------------- CrossVal


def test_crossval_warm_start_matches_cold(reg_dense):
    """CV value and hypergrad must agree between warm and cold at the same α."""
    X, y = reg_dense
    problem = Problem(SquaredLoss(), L1(), X, y)
    solver = SklearnLasso(tol=1e-9, max_iter=20_000)

    cold_cv = CrossVal.kfold(X.shape[0], k=4, shuffle=False, warm_start=False)
    warm_cv = CrossVal.kfold(X.shape[0], k=4, shuffle=False, warm_start=True)

    alphas = [0.5, 0.2, 0.1, 0.05]  # decreasing → each warm call seeded by prior
    for alpha in alphas:
        cold_res = cold_cv.value_and_hypergrad(problem, alpha, solver, implicit_forward)
        warm_res = warm_cv.value_and_hypergrad(problem, alpha, solver, implicit_forward)
        np.testing.assert_allclose(warm_res.value, cold_res.value, rtol=1e-6)
        np.testing.assert_allclose(float(warm_res.hypergrad), float(cold_res.hypergrad), rtol=1e-4)


def test_crossval_warm_start_value_only(reg_dense):
    X, y = reg_dense
    problem = Problem(SquaredLoss(), L1(), X, y)
    solver = SklearnLasso(tol=1e-9, max_iter=20_000)

    cold_cv = CrossVal.kfold(X.shape[0], k=3, shuffle=False, warm_start=False)
    warm_cv = CrossVal.kfold(X.shape[0], k=3, shuffle=False, warm_start=True)

    # Prime warm cache.
    warm_cv.value_and_hypergrad(problem, 0.2, solver, implicit_forward)
    # value() at a nearby α should match cold value().
    cold_v = cold_cv.value(problem, 0.18, solver)
    warm_v = warm_cv.value(problem, 0.18, solver)
    np.testing.assert_allclose(warm_v, cold_v, rtol=1e-6)


def test_crossval_warm_cache_excluded_from_equality():
    """Two CrossVal instances with the same folds must compare equal even
    after one of them has populated its warm-start cache."""
    folds = (
        (np.array([0, 1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)),
        (np.array([3, 4], dtype=np.int32), np.array([0, 1, 2], dtype=np.int32)),
    )
    a = CrossVal(folds=folds, warm_start=True)
    b = CrossVal(folds=folds, warm_start=True)
    a._cache.extend([np.zeros(3), np.ones(3)])  # mutate cache on `a` only
    # Cache excluded from comparison.
    assert a == b


# ---------------------------------------------------------------- end-to-end


def test_hoag_search_warm_start_runs_end_to_end():
    """End-to-end smoke: hoag_search with warm-start CrossVal runs and
    produces a finite α and finite CV value."""
    rng = np.random.default_rng(7)
    n, p = 120, 30
    X = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    beta_true[[2, 5, 11]] = [1.5, -2.0, 0.7]
    y = X @ beta_true + 0.5 * rng.standard_normal(n)
    problem = Problem(SquaredLoss(), L1(), X, y)
    solver = SklearnLasso(tol=1e-9, max_iter=20_000)
    warm_cv = CrossVal.kfold(n, k=3, shuffle=False, warm_start=True)

    result = hoag_search(
        problem,
        hp0=0.3,
        solver=solver,
        criterion=warm_cv,
        n_iter=15,
        inner_tol=1e-8,
        outer_tol=1e-6,
    )
    assert np.isfinite(float(result.best_hyperparam))
    assert float(result.best_hyperparam) > 0
    best_value = min(r.value for r in result.history)
    assert np.isfinite(best_value)
