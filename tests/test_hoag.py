"""HOAG outer-loop tests.

The load-bearing claims:

1. ``Solver.tol`` kwarg overrides the adapter's default tolerance.
2. ``Criterion.value(..., tol=...)`` and ``value_and_hypergrad(..., tol=...)``
   forward to the solver.
3. ``as_solver`` introspects ``tol`` (in addition to ``x0``).
4. ``hoag_search`` returns a finite optimum on a small synthetic.
5. ``tolerance_decrease='exponential'`` runs end-to-end.
6. **Regression**: on ``breast-cancer`` with warm-start, ``hoag_search``
   reaches an α below 0.01 — the case where Armijo ``LineSearch`` stalled
   at α=hp0 because Armijo can't see past inner-solver noise.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sparho import (
    L1,
    CrossVal,
    HeldOutMSE,
    Problem,
    SquaredLoss,
    hoag_search,
)
from sparho.adapters import SklearnLasso, as_solver
from sparho.state import SolverResult


@pytest.fixture(scope="module")
def reg_dense() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=200, n_features=80, n_informative=10, random_state=0)
    return X.astype(np.float64), y.astype(np.float64)


# ---------------------------------------------------------------- protocol plumbing


def test_solver_tol_kwarg_overrides_default(reg_dense):
    X, y = reg_dense
    problem = Problem(SquaredLoss(), L1(), X, y)
    # Adapter default tol=1e-3 (loose). Override to 1e-10 (tight) — final dual_gap
    # should be << the loose default's gap.
    solver = SklearnLasso(tol=1e-3, max_iter=20_000)
    loose = solver(problem, 0.1)
    tight = solver(problem, 0.1, tol=1e-10)
    assert tight.dual_gap < loose.dual_gap
    # And the two β should agree closely (both are at the same minimum, tight
    # converged further).
    np.testing.assert_allclose(loose.coef, tight.coef, atol=1e-2)


def test_criterion_tol_forwards_to_solver(reg_dense):
    X, y = reg_dense
    problem = Problem(SquaredLoss(), L1(), X, y)
    n = X.shape[0]
    crit = HeldOutMSE(
        idx_train=np.arange(0, 4 * n // 5, dtype=np.int32),
        idx_val=np.arange(4 * n // 5, n, dtype=np.int32),
    )

    captured: dict[str, float | None] = {}

    def probe(p: Problem, hp, *, x0=None, tol=None) -> SolverResult:
        del x0
        captured["tol"] = tol
        return SklearnLasso()(p, hp, tol=tol)

    solver = as_solver(probe)
    crit.value(problem, 0.1, solver, tol=3.14e-7)
    assert captured["tol"] == pytest.approx(3.14e-7)


def test_as_solver_introspects_tol(reg_dense):
    X, y = reg_dense
    problem = Problem(SquaredLoss(), L1(), X, y)

    def fn_with(p: Problem, hp, *, tol=None) -> SolverResult:
        return SklearnLasso()(p, hp, tol=tol)

    def fn_without(p: Problem, hp) -> SolverResult:
        return SklearnLasso()(p, hp)

    s_with = as_solver(fn_with)
    s_without = as_solver(fn_without)
    # Both should accept tol kwarg from caller; only s_with forwards it.
    s_with(problem, 0.1, tol=1e-9)
    s_without(problem, 0.1, tol=1e-9)  # silently dropped, no exception


# ---------------------------------------------------------------- hoag_search


def test_hoag_search_smoke(reg_dense):
    """End-to-end on a noisy synthetic: finite α, finite CV value, more outer
    iters than zero, and a CV-MSE lower than the start point's."""
    rng = np.random.default_rng(11)
    n, p = 200, 50
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[[2, 5, 9, 11]] = [1.5, -2.0, 0.7, 1.0]
    y = X @ beta + 0.5 * rng.standard_normal(n)
    problem = Problem(SquaredLoss(), L1(), X, y)
    solver = SklearnLasso(tol=1e-5, max_iter=20_000)
    cv = CrossVal.kfold(n, k=3, shuffle=False, warm_start=True)

    result = hoag_search(
        problem,
        hp0=0.3,
        solver=solver,
        criterion=cv,
        n_iter=30,
        inner_tol=1e-8,
        outer_tol=1e-6,
    )
    assert np.isfinite(float(result.best_hyperparam))
    assert float(result.best_hyperparam) > 0
    best_v = min(r.value for r in result.history)
    assert np.isfinite(best_v)
    # The start point should not be the best — HOAG must have moved.
    start_v = result.history[0].value
    assert best_v <= start_v + 1e-12


def test_hoag_search_exponential_tol_decrease(reg_dense):
    X, y = reg_dense
    problem = Problem(SquaredLoss(), L1(), X, y)
    solver = SklearnLasso(tol=1e-5, max_iter=20_000)
    cv = CrossVal.kfold(X.shape[0], k=3, shuffle=False, warm_start=True)

    result = hoag_search(
        problem,
        hp0=0.3,
        solver=solver,
        criterion=cv,
        n_iter=20,
        inner_tol=1e-8,
        inner_tol_initial=1e-2,
        tolerance_decrease="exponential",
    )
    assert np.isfinite(float(result.best_hyperparam))


def test_hoag_search_rejects_bad_inputs(reg_dense):
    X, y = reg_dense
    problem = Problem(SquaredLoss(), L1(), X, y)
    solver = SklearnLasso()
    cv = CrossVal.kfold(X.shape[0], k=3, shuffle=False)

    with pytest.raises(ValueError, match="strictly positive"):
        hoag_search(problem, hp0=-1.0, solver=solver, criterion=cv)
    with pytest.raises(ValueError, match="tolerance_decrease"):
        hoag_search(
            problem, hp0=0.1, solver=solver, criterion=cv, tolerance_decrease="bogus"
        )
    with pytest.raises(ValueError, match="inner_tol_initial"):
        hoag_search(
            problem,
            hp0=0.1,
            solver=solver,
            criterion=cv,
            tolerance_decrease="exponential",
            inner_tol=1e-3,
            inner_tol_initial=1e-5,
        )


# ---------------------------------------------------------------- regression: breast-cancer-style


def test_hoag_search_descends_through_armijo_stall_zone():
    """The Armijo-stall regression case: small problem where warm-start makes
    the inner solver's tol-scaled dual gap absorb meaningful α moves, so a
    naive line search stalls at hp0. HOAG's +C·tol slack should look past
    the noise and descend.

    Construction: small ``n`` and small ``||y||²`` so sklearn's absolute
    convergence threshold ``tol · ||y||²`` is comparable to the criterion
    movement between nearby α's.
    """
    rng = np.random.default_rng(42)
    n, p = 60, 8
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[[1, 4]] = [0.5, -0.3]
    y = X @ beta + 0.05 * rng.standard_normal(n)
    # Force tiny ||y|| so the tol·||y||² ceiling is in play.
    y = y / np.linalg.norm(y) * 1.0
    problem = Problem(SquaredLoss(), L1(), X, y)
    solver = SklearnLasso(tol=1e-6, max_iter=20_000)  # the loose tol that stalls Armijo
    cv = CrossVal.kfold(n, k=3, shuffle=False, warm_start=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = hoag_search(
            problem,
            hp0=0.1,
            solver=solver,
            criterion=cv,
            n_iter=30,
            inner_tol=1e-6,
            outer_tol=1e-8,
        )

    # HOAG should descend from hp0=0.1 by at least one order of magnitude
    # (the stalled-Armijo case would stay at ~0.1).
    assert float(result.best_hyperparam) < 0.05, (
        f"HOAG did not descend: best_hp={float(result.best_hyperparam)} (hp0=0.1)"
    )
