"""End-to-end ``grad_search`` tests: convergence, sklearn ``LassoCV`` parity,
log-space safety, and vector-hyperparam support."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sparho import (
    L1,
    CrossVal,
    HeldOutMSE,
    Problem,
    SquaredLoss,
    WeightedL1,
    grad_search,
    hoag_search,
)
from sparho.adapters import SklearnLasso, SklearnWeightedLasso

# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def reg_split():
    """Noisy regression — ``noise=10`` puts the held-out MSE optimum at a
    finite ``α``, not at the alphas-grid floor (which would force a
    ``LassoCV``-style infinite descent that small budgets can't match)."""
    X, y = make_regression(
        n_samples=200,
        n_features=60,
        n_informative=10,
        noise=10.0,
        random_state=0,
    )
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    rng = np.random.default_rng(0)
    idx = np.arange(200)
    rng.shuffle(idx)
    idx_train = idx[:140].astype(np.int32)
    idx_val = idx[140:].astype(np.int32)
    return X, y, idx_train, idx_val


# ---------------------------------------------------------------- end-to-end


def test_grad_search_reduces_held_out_mse(reg_split):
    """The final criterion value is no worse than at the starting α."""
    X, y, idx_train, idx_val = reg_split
    problem = Problem(SquaredLoss(), L1(), X, y)
    crit = HeldOutMSE(idx_train, idx_val)
    solver = SklearnLasso(tol=1e-10, max_iter=100_000)

    hp0 = 1.0
    v0 = crit.value(problem, hp0, solver)
    result = grad_search(
        problem,
        hp0=hp0,
        solver=solver,
        criterion=crit,
        lr=0.05,
        n_iter=30,
        tol=1e-6,
    )
    v_final = crit.value(problem, result.best_hyperparam, solver)
    assert v_final <= v0 + 1e-9
    assert result.best_coef.shape == (problem.n_features,)
    assert len(result.history) > 0


def test_hoag_search_parity_with_lasso_cv(reg_split):
    """hoag_search reaches a CV-MSE no more than 5% worse than ``LassoCV``'s grid best."""
    X, y, _, _ = reg_split
    alphas = np.logspace(-2, 1, 20)
    sk_cv = LassoCV(
        alphas=alphas,
        cv=KFold(5, shuffle=False),
        fit_intercept=False,
        tol=1e-8,
        max_iter=100_000,
    )
    sk_cv.fit(X, y)
    sk_mean_mse = float(sk_cv.mse_path_.mean(axis=1).min())

    problem = Problem(SquaredLoss(), L1(), X, y)
    cv = CrossVal.kfold(problem.n_samples, k=5, shuffle=False, warm_start=True)
    solver = SklearnLasso(tol=1e-8, max_iter=100_000)
    result = hoag_search(
        problem,
        hp0=float(alphas[len(alphas) // 2]),
        solver=solver,
        criterion=cv,
        n_iter=60,
        inner_tol=1e-8,
        inner_tol_initial=1e-2,
        tolerance_decrease="exponential",
        outer_tol=1e-6,
    )
    our_mse = cv.value(problem, result.best_hyperparam, solver)
    assert our_mse <= sk_mean_mse * 1.05


# ---------------------------------------------------------------- invariants


def test_grad_search_rejects_nonpositive_scalar_hp(reg_split):
    X, y, idx_train, idx_val = reg_split
    problem = Problem(SquaredLoss(), L1(), X, y)
    crit = HeldOutMSE(idx_train, idx_val)
    solver = SklearnLasso()
    with pytest.raises(ValueError):
        grad_search(
            problem,
            hp0=0.0,
            solver=solver,
            criterion=crit,
            lr=0.1,
            n_iter=1,
        )


def test_grad_search_rejects_nonpositive_vector_hp(reg_split):
    X, y, idx_train, idx_val = reg_split
    problem = Problem(SquaredLoss(), WeightedL1(), X, y)
    crit = HeldOutMSE(idx_train, idx_val)
    hp0 = np.full(problem.n_features, 0.1)
    hp0[0] = -1e-6
    with pytest.raises(ValueError):
        grad_search(
            problem,
            hp0=hp0,
            solver=SklearnWeightedLasso(),
            criterion=crit,
            lr=0.1,
            n_iter=1,
        )


def test_grad_search_keeps_alpha_positive_under_aggressive_steps(reg_split):
    """Even with a large lr in log space, exp(θ) stays strictly positive."""
    X, y, idx_train, idx_val = reg_split
    problem = Problem(SquaredLoss(), L1(), X, y)
    crit = HeldOutMSE(idx_train, idx_val)
    result = grad_search(
        problem,
        hp0=1.0,
        solver=SklearnLasso(tol=1e-10, max_iter=100_000),
        criterion=crit,
        lr=10.0,  # deliberately aggressive
        n_iter=10,
    )
    for rec in result.history:
        # Scalar hp; should remain strictly positive.
        assert float(rec.hyperparam) > 0


def test_grad_search_records_well_formed_history(reg_split):
    X, y, idx_train, idx_val = reg_split
    problem = Problem(SquaredLoss(), L1(), X, y)
    crit = HeldOutMSE(idx_train, idx_val)
    result = grad_search(
        problem,
        hp0=0.5,
        solver=SklearnLasso(tol=1e-10, max_iter=100_000),
        criterion=crit,
        lr=0.1,
        n_iter=8,
    )
    assert 1 <= len(result.history) <= 8
    for rec in result.history:
        assert rec.iteration >= 0
        assert rec.grad_norm >= 0
        assert np.isfinite(rec.value)
    assert result.n_iter == len(result.history)


# ---------------------------------------------------------------- vector hp


def test_grad_search_weighted_lasso_smoke(reg_split):
    """grad_search exercises the vector-hyperparam path on WeightedL1."""
    X, y, idx_train, idx_val = reg_split
    problem = Problem(SquaredLoss(), WeightedL1(), X, y)
    crit = HeldOutMSE(idx_train, idx_val)
    hp0 = np.full(problem.n_features, 0.1)
    result = grad_search(
        problem,
        hp0=hp0,
        solver=SklearnWeightedLasso(tol=1e-8, max_iter=50_000),
        criterion=crit,
        lr=0.1,
        n_iter=5,
    )
    assert isinstance(result.best_hyperparam, np.ndarray)
    assert result.best_hyperparam.shape == (problem.n_features,)
    assert np.all(result.best_hyperparam > 0)
    assert result.best_coef.shape == (problem.n_features,)


def test_grad_search_returns_full_data_refit(reg_split):
    """``best_coef`` is fit on the *full* design, not just the training split."""
    X, y, idx_train, idx_val = reg_split
    problem = Problem(SquaredLoss(), L1(), X, y)
    crit = HeldOutMSE(idx_train, idx_val)
    solver = SklearnLasso(tol=1e-10, max_iter=100_000)
    result = grad_search(
        problem,
        hp0=0.5,
        solver=solver,
        criterion=crit,
        lr=0.05,
        n_iter=5,
    )
    # Refit solver directly on the full data at best_hyperparam.
    full_refit = solver(problem, result.best_hyperparam)
    np.testing.assert_allclose(result.best_coef, full_refit.coef, atol=1e-12, rtol=1e-10)
