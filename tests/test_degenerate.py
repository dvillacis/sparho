"""Degenerate-input regression tests.

Each case is something the library promises to handle gracefully — either
by producing a sensible answer or by raising an actionable error — rather
than crashing, hanging, or returning NaN. These are the inputs that real
users will hit when they wire the library into a pipeline they don't fully
control (sklearn `GridSearchCV` over hyperparameters, `cross_val_score` over
weird folds, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest
from sparho import (
    L1,
    HeldOutMSE,
    LogisticRegressionHO,
    Problem,
    SquaredLoss,
    grad_search,
    hoag_search,
    implicit_forward,
)
from sparho.adapters import SklearnLasso
from sparho.adapters._common import active_set_of
from sparho.state import SolverResult

_RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- Empty active set


def test_implicit_forward_empty_active_set_returns_zero():
    """When β* = 0 the hypergradient must be 0 (β* doesn't move under small α perturbation)."""
    X = _RNG.standard_normal((20, 5))
    y = _RNG.standard_normal(20)
    problem = Problem(SquaredLoss(), L1(), X, y)
    # Synthesize a converged solver result with empty active set.
    coef = np.zeros(5)
    sr = SolverResult(coef=coef, active_set=active_set_of(coef), dual_gap=0.0, n_iter=0)
    grad_C = _RNG.standard_normal(5)
    hg = implicit_forward(problem, 1.0, sr, grad_C)
    assert hg == 0.0


def test_grad_search_with_large_alpha_drives_active_set_empty():
    """Outer search at large α still terminates cleanly (no division by zero)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 4))
    y = rng.standard_normal(30)
    problem = Problem(SquaredLoss(), L1(), X, y)
    result = hoag_search(
        problem,
        1e3,  # huge α → β* = 0
        solver=SklearnLasso(tol=1e-6),
        criterion=HeldOutMSE(idx_train=np.arange(20), idx_val=np.arange(20, 30)),
        n_iter=3,
    )
    assert np.isfinite(result.best_hyperparam)
    assert np.all(np.isfinite(result.best_coef))


# ---------------------------------------------------------------- All-zero target


def test_zero_target_yields_zero_coef():
    """y = 0 → β* = 0 for any α > 0."""
    X = _RNG.standard_normal((20, 4))
    y = np.zeros(20)
    problem = Problem(SquaredLoss(), L1(), X, y)
    result = hoag_search(
        problem,
        0.1,
        solver=SklearnLasso(tol=1e-6),
        criterion=HeldOutMSE(idx_train=np.arange(15), idx_val=np.arange(15, 20)),
        n_iter=3,
    )
    np.testing.assert_array_equal(result.best_coef, np.zeros(4))


# ---------------------------------------------------------------- Single-class logistic


def test_logistic_wrapper_rejects_single_class_target():
    """`LogisticRegressionHO.fit` raises on a degenerate single-class y."""
    X = _RNG.standard_normal((30, 4))
    y = np.zeros(30, dtype=int)  # all class 0
    est = LogisticRegressionHO(alpha_init=0.1, n_iter=3)
    with pytest.raises(ValueError, match="single unique value"):
        est.fit(X, y)


# ---------------------------------------------------------------- Collinear features


def test_collinear_features_dont_crash_outer_search():
    """Three features are exact copies of the first.

    The ridge in ``implicit_forward`` must absorb the singularity.
    """
    rng = np.random.default_rng(2)
    base = rng.standard_normal((40, 1))
    X = np.hstack([base, base, base, rng.standard_normal((40, 1))])  # 3 collinear cols
    y = (base[:, 0] + 0.1 * rng.standard_normal(40)).astype(np.float64)
    problem = Problem(SquaredLoss(), L1(), X, y)
    result = hoag_search(
        problem,
        0.1,
        solver=SklearnLasso(tol=1e-6),
        criterion=HeldOutMSE(idx_train=np.arange(30), idx_val=np.arange(30, 40)),
        n_iter=4,
    )
    # No NaN, no crash, finite outcome.
    assert np.isfinite(result.best_hyperparam)
    assert np.all(np.isfinite(result.best_coef))


# ---------------------------------------------------------------- n << p


def test_n_much_smaller_than_p_runs():
    """Under-determined regime: 10 samples, 100 features.

    Lasso is still well-defined; outer search must run.
    """
    rng = np.random.default_rng(3)
    X = rng.standard_normal((10, 100))
    beta = np.zeros(100)
    beta[:2] = [1.0, -0.7]
    y = X @ beta + 0.1 * rng.standard_normal(10)
    problem = Problem(SquaredLoss(), L1(), X, y)
    result = grad_search(
        problem,
        0.1,
        solver=SklearnLasso(tol=1e-5),
        criterion=HeldOutMSE(idx_train=np.arange(7), idx_val=np.arange(7, 10)),
        n_iter=5,
        lr=0.05,
    )
    assert np.isfinite(result.best_hyperparam)
    assert result.best_coef.shape == (100,)
    # Sparsity: at most ~10 features active (sample count bound).
    assert int(np.count_nonzero(result.best_coef)) <= 10
