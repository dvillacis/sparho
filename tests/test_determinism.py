"""Bit-identical determinism at fixed ``random_state``.

`CrossVal.kfold` (shuffle=True) and `Sure` (RNG-driven δ probe) both consume
a seed. Two fits at the same seed must produce bit-identical
``SearchResult.best_hyperparam`` / ``best_coef`` — required for reproducible
benchmarks, MLflow run comparison, and the `check_fit_idempotent` sklearn
check.
"""

from __future__ import annotations

import numpy as np
from sparho import (
    L1,
    CrossVal,
    HeldOutMSE,
    Problem,
    SquaredLoss,
    Sure,
    grad_search,
    hoag_search,
)
from sparho.adapters import SklearnLasso


def _make_problem(seed: int = 0, n_samples: int = 80, n_features: int = 12):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    beta = np.zeros(n_features)
    beta[:3] = [1.0, -0.5, 0.7]
    y = X @ beta + 0.1 * rng.standard_normal(n_samples)
    return Problem(SquaredLoss(), L1(), X, y)


def test_cross_val_kfold_deterministic_at_fixed_random_state():
    """Same `random_state` → same fold split → bit-identical search result."""
    problem = _make_problem()
    n = problem.n_samples
    solver = SklearnLasso(tol=1e-6)

    def _run(seed: int):
        cv = CrossVal.kfold(n, k=5, shuffle=True, random_state=seed, base=HeldOutMSE)
        return hoag_search(
            problem, 0.1, solver=solver, criterion=cv, n_iter=5, inner_tol=1e-6
        )

    r1 = _run(0)
    r2 = _run(0)
    assert r1.best_hyperparam == r2.best_hyperparam
    np.testing.assert_array_equal(r1.best_coef, r2.best_coef)
    assert r1.n_iter == r2.n_iter


def test_cross_val_different_seeds_differ_in_general():
    """Sanity: different seeds → different fold splits → generally different result.

    Catches the bug where `random_state` is silently dropped (which would
    make the determinism test above pass trivially).
    """
    problem = _make_problem()
    n = problem.n_samples
    solver = SklearnLasso(tol=1e-6)

    def _run(seed: int):
        cv = CrossVal.kfold(n, k=5, shuffle=True, random_state=seed, base=HeldOutMSE)
        return hoag_search(
            problem, 0.1, solver=solver, criterion=cv, n_iter=5, inner_tol=1e-6
        )

    r0 = _run(0)
    r1 = _run(1)
    # Folds differ → α path differs in general (or coefs differ).
    different = (r0.best_hyperparam != r1.best_hyperparam) or (
        not np.array_equal(r0.best_coef, r1.best_coef)
    )
    assert different, "different seeds produced identical results — random_state ignored?"


def test_sure_deterministic_at_fixed_random_state():
    """Same `random_state` → same δ probe → bit-identical search result."""
    problem = _make_problem()
    solver = SklearnLasso(tol=1e-6)

    def _run(seed: int):
        sure = Sure(sigma=0.1, random_state=seed)
        return grad_search(
            problem, 0.1, solver=solver, criterion=sure, n_iter=4, lr=0.05
        )

    r1 = _run(7)
    r2 = _run(7)
    assert r1.best_hyperparam == r2.best_hyperparam
    np.testing.assert_array_equal(r1.best_coef, r2.best_coef)
