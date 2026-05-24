"""Adapter tests: each adapter must return a converged SolverResult with a
consistent ``active_set`` and small ``dual_gap``, and the sklearn / celer
adapters must agree at the same hyperparameter when both are available."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_regression
from sparho import (
    L1,
    ElasticNet,
    LogisticLoss,
    Problem,
    Solver,
    SolverResult,
    SquaredLoss,
    WeightedL1,
)
from sparho.adapters import (
    SklearnElasticNet,
    SklearnLasso,
    SklearnLogisticRegression,
    SklearnWeightedLasso,
    as_solver,
)

# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def reg_dense() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=200, n_features=80, n_informative=10, random_state=0)
    return X.astype(np.float64), y.astype(np.float64)


@pytest.fixture(scope="module")
def reg_sparse() -> tuple[sp.csc_matrix, np.ndarray]:
    rng = np.random.default_rng(1)
    X = sp.random(150, 60, density=0.1, format="csc", random_state=rng).astype(np.float64)
    true_beta = np.zeros(60)
    true_beta[[3, 7, 12, 30]] = [1.5, -2.0, 0.5, 3.0]
    y = X @ true_beta + 0.05 * rng.standard_normal(150)
    return X, y


@pytest.fixture(scope="module")
def cls_dense() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 50))
    true_beta = np.zeros(50)
    true_beta[[1, 4, 9]] = [2.0, -1.5, 1.0]
    logits = X @ true_beta
    y = np.where(logits > 0, 1.0, -1.0)
    return X, y


# ---------------------------------------------------------------- contract


def _assert_solver_result(result: SolverResult, n_features: int) -> None:
    assert isinstance(result, SolverResult)
    assert result.coef.shape == (n_features,)
    assert result.coef.dtype == np.float64
    assert result.active_set.dtype == np.int32
    # Active set must agree with nonzero pattern of coef.
    np.testing.assert_array_equal(result.active_set, np.flatnonzero(result.coef))
    # Weak duality says dual_gap ≥ 0 at the optimum, but sklearn computes
    # primal − dual via floating-point sums; near-converged solutions can
    # land a few ulps below zero. -1e-10 absorbs that noise while still
    # catching algorithm bugs (which would produce O(1) negative gaps).
    assert result.dual_gap >= -1e-10
    assert result.n_iter >= 0


# ---------------------------------------------------------------- SklearnLasso


def test_sklearn_lasso_dense(reg_dense):
    X, y = reg_dense
    problem = Problem(SquaredLoss(), L1(), X, y)
    solver = SklearnLasso(tol=1e-8, max_iter=10_000)
    result = solver(problem, 0.1)
    _assert_solver_result(result, problem.n_features)
    # `dual_gap_` is in absolute units; what matters is "converged before max_iter".
    assert result.n_iter < solver.max_iter


def test_sklearn_lasso_sparse(reg_sparse):
    X, y = reg_sparse
    problem = Problem(SquaredLoss(), L1(), X, y)
    solver = SklearnLasso(tol=1e-8)
    result = solver(problem, 0.05)
    _assert_solver_result(result, problem.n_features)
    # On a well-conditioned support-recovery problem the active set is small.
    assert len(result.active_set) <= 20


def test_sklearn_lasso_rejects_wrong_problem(reg_dense):
    X, y = reg_dense
    problem = Problem(SquaredLoss(), ElasticNet(rho=0.5), X, y)
    with pytest.raises(TypeError):
        SklearnLasso()(problem, 0.1)


def test_sklearn_lasso_protocol_isinstance():
    assert isinstance(SklearnLasso(), Solver)


# ---------------------------------------------------------------- SklearnElasticNet


def test_sklearn_elasticnet_reduces_to_lasso_at_rho_one(reg_dense):
    X, y = reg_dense
    p_l1 = Problem(SquaredLoss(), L1(), X, y)
    p_en = Problem(SquaredLoss(), ElasticNet(rho=1.0), X, y)
    r_l1 = SklearnLasso(tol=1e-9)(p_l1, 0.1)
    r_en = SklearnElasticNet(tol=1e-9)(p_en, 0.1)
    np.testing.assert_allclose(r_en.coef, r_l1.coef, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------- SklearnWeightedLasso


def test_sklearn_weighted_lasso_reduces_to_lasso_with_constant_weights(reg_dense):
    X, y = reg_dense
    p_l1 = Problem(SquaredLoss(), L1(), X, y)
    p_w = Problem(SquaredLoss(), WeightedL1(), X, y)
    alpha = 0.1
    r_l1 = SklearnLasso(tol=1e-9)(p_l1, alpha)
    r_w = SklearnWeightedLasso(tol=1e-9)(p_w, np.full(p_w.n_features, alpha))
    np.testing.assert_allclose(r_w.coef, r_l1.coef, atol=1e-6, rtol=1e-5)


def test_sklearn_weighted_lasso_sparse(reg_sparse):
    X, y = reg_sparse
    problem = Problem(SquaredLoss(), WeightedL1(), X, y)
    alpha = np.full(problem.n_features, 0.05)
    result = SklearnWeightedLasso(tol=1e-8)(problem, alpha)
    _assert_solver_result(result, problem.n_features)


def test_sklearn_weighted_lasso_rejects_nonpositive(reg_dense):
    X, y = reg_dense
    problem = Problem(SquaredLoss(), WeightedL1(), X, y)
    alpha = np.full(problem.n_features, 0.1)
    alpha[0] = 0.0
    with pytest.raises(ValueError):
        SklearnWeightedLasso()(problem, alpha)


# ---------------------------------------------------------------- SklearnLogisticRegression


def test_sklearn_logreg_recovers_support(cls_dense):
    X, y = cls_dense
    problem = Problem(LogisticLoss(), L1(), X, y)
    result = SklearnLogisticRegression(tol=1e-8)(problem, 0.05)
    _assert_solver_result(result, problem.n_features)
    # Stationarity proxy should be near zero.
    assert result.dual_gap < 1e-2  # liblinear's tol isn't tight
    # Most of the support should be among the planted nonzeros.
    assert {1, 4, 9}.issubset(set(result.active_set.tolist()))


def test_sklearn_logreg_rejects_unscaled_labels(cls_dense):
    X, y = cls_dense
    y_01 = 0.5 * (y + 1.0)
    problem = Problem(LogisticLoss(), L1(), X, y_01)
    with pytest.raises(ValueError):
        SklearnLogisticRegression()(problem, 0.05)


# ---------------------------------------------------------------- celer (optional)


def _celer_or_skip():
    pytest.importorskip("celer", reason="celer extra not installed")
    from sparho.adapters.celer import CelerElasticNet, CelerLasso

    return CelerLasso, CelerElasticNet


def test_celer_lasso_matches_sklearn_lasso(reg_sparse):
    CelerLasso, _ = _celer_or_skip()
    X, y = reg_sparse
    problem = Problem(SquaredLoss(), L1(), X, y)
    r_sk = SklearnLasso(tol=1e-10)(problem, 0.05)
    r_ce = CelerLasso(tol=1e-10)(problem, 0.05)
    np.testing.assert_allclose(r_ce.coef, r_sk.coef, atol=1e-6, rtol=1e-5)


def test_celer_elasticnet_matches_sklearn_elasticnet(reg_dense):
    _, CelerElasticNet = _celer_or_skip()
    X, y = reg_dense
    problem = Problem(SquaredLoss(), ElasticNet(rho=0.7), X, y)
    r_sk = SklearnElasticNet(tol=1e-10)(problem, 0.05)
    r_ce = CelerElasticNet(tol=1e-10)(problem, 0.05)
    np.testing.assert_allclose(r_ce.coef, r_sk.coef, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------- as_solver


def test_as_solver_wraps_plain_callable():
    def fn(problem, hp):
        return SolverResult(
            coef=np.zeros(problem.n_features),
            active_set=np.array([], dtype=np.int32),
            dual_gap=0.0,
            n_iter=0,
        )

    wrapped = as_solver(fn, name="zero-solver")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 3))
    y = rng.standard_normal(5)
    problem = Problem(SquaredLoss(), L1(), X, y)
    result = wrapped(problem, 0.1)
    assert result.coef.shape == (3,)
    assert wrapped.name == "zero-solver"
