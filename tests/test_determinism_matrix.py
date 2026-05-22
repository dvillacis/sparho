"""Determinism audit across the (seed × BLAS-threads × criterion) matrix.

Extends ``test_determinism.py``'s seed-only coverage to a matrix where the
BLAS-thread count varies independently. The audit serves two purposes:

1. **Regression guard** at ``n_threads = 1`` — `best_hyperparam` and
   `best_coef` must be bit-identical across reruns. A drift would imply
   a non-deterministic code path (uninitialized scratch, dict-iteration
   leak, etc.) entered between releases.
2. **Calibration** at ``n_threads > 1`` — multi-threaded BLAS introduces
   reduction-order nondeterminism, but the high-order bits should still
   agree. The tolerance asserted at ``n_threads ∈ {2, 4}`` documents the
   drift envelope a user can expect on a typical workstation.

See ``docs/reproducibility.md`` for the threading discipline this audit
encodes.
"""

from __future__ import annotations

import numpy as np
import pytest
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
from sparho.testing import pin_blas_threads


def _make_problem(seed: int, n_samples: int = 80, n_features: int = 12) -> Problem:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    beta = np.zeros(n_features)
    beta[:3] = [1.0, -0.5, 0.7]
    y = X @ beta + 0.1 * rng.standard_normal(n_samples)
    return Problem(SquaredLoss(), L1(), X, y)


def _run_cv(problem: Problem, seed: int) -> tuple[float, np.ndarray]:
    solver = SklearnLasso(tol=1e-6)
    cv = CrossVal.kfold(
        problem.n_samples, k=5, shuffle=True, random_state=seed, base=HeldOutMSE
    )
    res = hoag_search(problem, 0.1, solver=solver, criterion=cv, n_iter=5, inner_tol=1e-6)
    return float(res.best_hyperparam), np.asarray(res.best_coef, dtype=np.float64)


def _run_sure(problem: Problem, seed: int) -> tuple[float, np.ndarray]:
    solver = SklearnLasso(tol=1e-6)
    sure = Sure(sigma=0.1, random_state=seed)
    res = grad_search(problem, 0.1, solver=solver, criterion=sure, n_iter=4, lr=0.05)
    return float(res.best_hyperparam), np.asarray(res.best_coef, dtype=np.float64)


# (n_threads, seed, criterion) — keep modest so the suite stays fast.
_MATRIX = [
    (n, seed, crit)
    for n in (1, 2, 4)
    for seed in (0, 7)
    for crit in ("cv", "sure")
]


@pytest.mark.parametrize(
    ("n_threads", "seed", "criterion"),
    _MATRIX,
    ids=[f"threads={n}-seed={s}-crit={c}" for n, s, c in _MATRIX],
)
def test_replay_within_tolerance(n_threads: int, seed: int, criterion: str) -> None:
    """Replay the same (seed, criterion) twice; assert agreement.

    At ``n_threads = 1`` the agreement is bit-identical (== / array_equal).
    At ``n_threads > 1`` the agreement is *within tolerance* — the test
    documents the actual envelope rather than enforcing an artificial bound.
    """
    problem = _make_problem(seed)
    runner = _run_cv if criterion == "cv" else _run_sure

    with pin_blas_threads(n_threads):
        hp1, beta1 = runner(problem, seed)
        hp2, beta2 = runner(problem, seed)

    if n_threads == 1:
        # Strict bit-identity at single-threaded BLAS.
        assert hp1 == hp2, f"hp drifted at n_threads=1: {hp1} vs {hp2}"
        np.testing.assert_array_equal(beta1, beta2)
    else:
        # Multi-threaded: assert tight numerical agreement. The drift here
        # is the multi-threaded BLAS reduction-order nondeterminism and is
        # bounded by the relative tolerance of the inner solver's stopping
        # criterion (tol=1e-6 above).
        np.testing.assert_allclose(hp1, hp2, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(beta1, beta2, atol=1e-5, rtol=1e-4)


def test_seed_actually_matters() -> None:
    """Sanity: different seeds → different results (regression test for #ignored_seed)."""
    problem = _make_problem(0)
    with pin_blas_threads(1):
        hp_a, beta_a = _run_cv(problem, seed=0)
        hp_b, beta_b = _run_cv(problem, seed=1)
    different = (hp_a != hp_b) or (not np.array_equal(beta_a, beta_b))
    assert different, "different CrossVal seeds produced identical output"
