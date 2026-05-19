"""Criterion tests: sklearn parity for value, FD checks for hypergradient."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso as SkLasso
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sparho import L1, LogisticLoss, Problem, SquaredLoss
from sparho.adapters import SklearnLasso, SklearnLogisticRegression
from sparho.criteria import (
    Criterion,
    CrossVal,
    HeldOutLogistic,
    HeldOutMSE,
)
from sparho.hypergrad import implicit_forward

# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def reg_problem_and_split():
    X, y = make_regression(n_samples=200, n_features=50, n_informative=10, random_state=0)
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    rng = np.random.default_rng(0)
    idx = np.arange(200)
    rng.shuffle(idx)
    idx_train = idx[:140].astype(np.int32)
    idx_val = idx[140:].astype(np.int32)
    return Problem(SquaredLoss(), L1(), X, y), idx_train, idx_val


@pytest.fixture(scope="module")
def cls_problem_and_split():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 30))
    true_beta = np.zeros(30)
    true_beta[:5] = [2.0, -2.0, 1.5, -1.5, 1.0]
    logits = X @ true_beta
    y = np.where(logits > 0, 1.0, -1.0)
    idx = np.arange(300)
    rng.shuffle(idx)
    idx_train = idx[:200].astype(np.int32)
    idx_val = idx[200:].astype(np.int32)
    return Problem(LogisticLoss(), L1(), X, y), idx_train, idx_val


# ---------------------------------------------------------------- protocol


def test_held_out_mse_satisfies_criterion_protocol(reg_problem_and_split):
    _, idx_tr, idx_val = reg_problem_and_split
    assert isinstance(HeldOutMSE(idx_tr, idx_val), Criterion)


def test_held_out_logistic_satisfies_criterion_protocol(cls_problem_and_split):
    _, idx_tr, idx_val = cls_problem_and_split
    assert isinstance(HeldOutLogistic(idx_tr, idx_val), Criterion)


def test_cross_val_satisfies_criterion_protocol():
    cv = CrossVal.kfold(100, k=3, shuffle=False)
    assert isinstance(cv, Criterion)


# ---------------------------------------------------------------- HeldOutMSE


def test_held_out_mse_value_matches_sklearn(reg_problem_and_split):
    problem, idx_tr, idx_val = reg_problem_and_split
    alpha = 0.1
    crit = HeldOutMSE(idx_tr, idx_val)
    solver = SklearnLasso(tol=1e-12, max_iter=100_000)
    value = crit.value(problem, alpha, solver)
    # Direct sklearn computation.
    est = SkLasso(alpha=alpha, fit_intercept=False, tol=1e-12, max_iter=100_000)
    est.fit(problem.design[idx_tr], problem.target[idx_tr])
    expected = mean_squared_error(problem.target[idx_val], problem.design[idx_val] @ est.coef_)
    assert value == pytest.approx(expected, rel=1e-9)


def test_held_out_mse_hypergrad_finite_difference(reg_problem_and_split):
    problem, idx_tr, idx_val = reg_problem_and_split
    alpha = 0.1
    eps = 1e-5
    crit = HeldOutMSE(idx_tr, idx_val)
    solver = SklearnLasso(tol=1e-12, max_iter=100_000)
    result = crit.value_and_hypergrad(problem, alpha, solver, implicit_forward)
    v_plus = crit.value(problem, alpha + eps, solver)
    v_minus = crit.value(problem, alpha - eps, solver)
    fd = (v_plus - v_minus) / (2 * eps)
    assert result.hypergrad == pytest.approx(fd, rel=1e-3, abs=1e-4)
    # Coef and active set should mirror the converged inner solve.
    np.testing.assert_array_equal(
        result.active_set, np.flatnonzero(result.coef).astype(np.int32)
    )


# ---------------------------------------------------------------- HeldOutLogistic


def test_held_out_logistic_value_matches_formula(cls_problem_and_split):
    problem, idx_tr, idx_val = cls_problem_and_split
    alpha = 0.1
    crit = HeldOutLogistic(idx_tr, idx_val)
    solver = SklearnLogisticRegression(tol=1e-10, max_iter=100_000)
    value = crit.value(problem, alpha, solver)
    # Recompute by hand using the adapter directly on the same subset.
    train_problem = Problem(
        problem.datafit, problem.penalty, problem.design[idx_tr], problem.target[idx_tr]
    )
    result_sk = solver(train_problem, alpha)
    y_val = problem.target[idx_val]
    Xb = problem.design[idx_val] @ result_sk.coef
    expected = float(np.mean(np.logaddexp(0.0, -y_val * Xb)))
    # liblinear runs aren't bit-reproducible across two fits at the same tol;
    # a 1e-6 relative match is the tightest we can rely on.
    assert value == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------- CrossVal


def test_cross_val_value_averages_over_folds(reg_problem_and_split):
    problem, _, _ = reg_problem_and_split
    cv = CrossVal.kfold(problem.n_samples, k=4, shuffle=False)
    solver = SklearnLasso(tol=1e-10, max_iter=100_000)
    value = cv.value(problem, 0.1, solver)
    # Compute the same K HeldOutMSE evaluations manually.
    total = 0.0
    for idx_tr, idx_val in cv.folds:
        total += HeldOutMSE(idx_tr, idx_val).value(problem, 0.1, solver)
    expected = total / len(cv.folds)
    assert value == pytest.approx(expected, rel=1e-12)


def test_cross_val_matches_lasso_cv(reg_problem_and_split):
    """At the α selected by sklearn ``LassoCV``, our CrossVal value matches its
    reported mean MSE within ``rtol=1e-4``."""
    problem, _, _ = reg_problem_and_split
    alphas = np.logspace(-2, 0, 5)
    sk_cv = LassoCV(
        alphas=alphas,
        cv=KFold(5, shuffle=False),
        fit_intercept=False,
        tol=1e-10,
        max_iter=100_000,
    )
    sk_cv.fit(problem.design, problem.target)
    alpha_best = float(sk_cv.alpha_)
    # sklearn's ``mse_path_`` is (n_alphas, n_folds); rows ordered by descending α.
    i_best = int(np.argmin(sk_cv.mse_path_.mean(axis=1)))
    mean_mse_at_best = float(sk_cv.mse_path_[i_best].mean())
    cv = CrossVal.kfold(problem.n_samples, k=5, shuffle=False)
    solver = SklearnLasso(tol=1e-10, max_iter=100_000)
    our_value = cv.value(problem, alpha_best, solver)
    assert our_value == pytest.approx(mean_mse_at_best, rel=1e-4)


def test_cross_val_hypergrad_finite_difference(reg_problem_and_split):
    problem, _, _ = reg_problem_and_split
    cv = CrossVal.kfold(problem.n_samples, k=4, shuffle=False)
    solver = SklearnLasso(tol=1e-12, max_iter=100_000)
    alpha = 0.1
    eps = 1e-5
    result = cv.value_and_hypergrad(problem, alpha, solver, implicit_forward)
    v_plus = cv.value(problem, alpha + eps, solver)
    v_minus = cv.value(problem, alpha - eps, solver)
    fd = (v_plus - v_minus) / (2 * eps)
    assert result.hypergrad == pytest.approx(fd, rel=1e-3, abs=1e-3)


# ---------------------------------------------------------------- end-to-end


def test_held_out_mse_classification_with_logistic_via_held_out_logistic(cls_problem_and_split):
    problem, idx_tr, idx_val = cls_problem_and_split
    crit = HeldOutLogistic(idx_tr, idx_val)
    solver = SklearnLogisticRegression(tol=1e-10, max_iter=100_000)
    result = crit.value_and_hypergrad(problem, 0.1, solver, implicit_forward)
    # Sanity bounds.
    assert 0.0 <= result.value <= np.log(2.0) * 1.1  # ≤ uniform-noise upper bound (loose)
    assert result.coef.shape == (problem.n_features,)
    assert result.active_set.dtype == np.int32
