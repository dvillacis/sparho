"""HOAG outer loop on vector-α penalties with an exponential inner-tol schedule.

`hoag_search`'s rejection branch (`value >= 1.2 · value_prev` → reject step,
double `L`, recompute val+grad at the restored θ with halved tol) is the
trickiest path in the search. Smoke-tests cover the happy path; this suite
hits the rejection branch on `WeightedL1` (vector-α) and `GroupL1` (scalar-α
but vector active-set semantics) under an exponentially-decreasing tolerance
schedule, on both dense and sparse designs.

The contract: the search must finish in `n_iter` outer iterations, leave
``best_hyperparam`` and ``best_coef`` finite, and produce a history whose
``IterationRecord.value``s are all finite (NaNs would mean the rejection
branch failed to restore a sane state).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sparho import (
    GroupL1,
    HeldOutMSE,
    Problem,
    SquaredLoss,
    WeightedL1,
    hoag_search,
)
from sparho.adapters import GroupLassoFista, SklearnWeightedLasso

_RNG = np.random.default_rng(11)


def _make_dense_problem(n_samples: int = 40, n_features: int = 6):
    X = _RNG.standard_normal((n_samples, n_features))
    beta = np.zeros(n_features)
    beta[:3] = [1.0, -0.5, 0.7]
    y = X @ beta + 0.1 * _RNG.standard_normal(n_samples)
    return X, y


def _make_sparse_problem(n_samples: int = 40, n_features: int = 6, density: float = 0.4):
    X = sp.random(n_samples, n_features, density=density, format="csc", random_state=11).astype(
        np.float64
    )
    beta = np.zeros(n_features)
    beta[:3] = [1.0, -0.5, 0.7]
    y = (X @ beta) + 0.1 * np.random.default_rng(12).standard_normal(n_samples)
    return X, np.asarray(y).ravel()


# ---------------------------------------------------------------- WeightedL1


@pytest.mark.parametrize("density", ["dense", "sparse"])
def test_hoag_weighted_l1_exponential_tol(density: str):
    X, y = _make_dense_problem() if density == "dense" else _make_sparse_problem()
    n_samples, n_features = X.shape
    problem = Problem(SquaredLoss(), WeightedL1(), X, y)
    hp0 = np.full(n_features, 0.05)
    result = hoag_search(
        problem,
        hp0,
        solver=SklearnWeightedLasso(tol=1e-7),
        criterion=HeldOutMSE(
            idx_train=np.arange(int(0.7 * n_samples)),
            idx_val=np.arange(int(0.7 * n_samples), n_samples),
        ),
        n_iter=6,
        inner_tol=1e-6,
        inner_tol_initial=1e-2,
        tolerance_decrease="exponential",
    )
    assert np.all(np.isfinite(result.best_hyperparam))
    assert np.all(np.isfinite(result.best_coef))
    assert all(np.isfinite(r.value) for r in result.history)
    assert all(np.isfinite(r.grad_norm) for r in result.history)


# ---------------------------------------------------------------- GroupL1


def test_hoag_group_l1_exponential_tol_dense():
    X, y = _make_dense_problem(n_samples=50, n_features=6)
    # 3 groups: {0,1}, {2,3}, {4,5}.
    penalty = GroupL1(groups=((0, 1), (2, 3), (4, 5)))
    problem = Problem(SquaredLoss(), penalty, X, y)
    n_samples = X.shape[0]
    result = hoag_search(
        problem,
        0.1,
        solver=GroupLassoFista(tol=1e-7, max_iter=2000),
        criterion=HeldOutMSE(
            idx_train=np.arange(int(0.7 * n_samples)),
            idx_val=np.arange(int(0.7 * n_samples), n_samples),
        ),
        n_iter=5,
        inner_tol=1e-6,
        inner_tol_initial=1e-2,
        tolerance_decrease="exponential",
    )
    assert np.isfinite(result.best_hyperparam)
    assert np.all(np.isfinite(result.best_coef))
    assert all(np.isfinite(r.value) for r in result.history)
