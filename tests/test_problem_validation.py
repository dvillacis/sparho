"""Regression tests for the v0.3.1 Python-side input validation.

`Problem`, `ElasticNet`, and `GroupL1` now validate invariants at
construction. Vector-α length is preflighted in `grad_search`/`hoag_search`.
Adapter helpers produce actionable shape errors.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sparho import (
    L1,
    ElasticNet,
    GroupL1,
    HeldOutMSE,
    Problem,
    SquaredLoss,
    WeightedL1,
    grad_search,
)
from sparho import (
    problem as _problem_mod,
)
from sparho.adapters import SklearnLasso
from sparho.adapters._common import as_scalar, as_vector

# ---------------------------------------------------------------- Problem


def test_problem_rejects_design_not_2d():
    X = np.zeros(10)  # 1-D
    y = np.zeros(10)
    with pytest.raises(ValueError, match="design must be 2-D"):
        Problem(SquaredLoss(), L1(), X, y)


def test_problem_rejects_target_not_1d():
    X = np.zeros((10, 3))
    y = np.zeros((10, 2))
    with pytest.raises(ValueError, match="target must be 1-D"):
        Problem(SquaredLoss(), L1(), X, y)


def test_problem_rejects_target_length_mismatch():
    X = np.zeros((10, 3))
    y = np.zeros(7)
    with pytest.raises(ValueError, match="target length"):
        Problem(SquaredLoss(), L1(), X, y)


def test_problem_rejects_nan_design():
    X = np.zeros((10, 3))
    X[0, 0] = np.nan
    y = np.zeros(10)
    with pytest.raises(ValueError, match="design contains NaN"):
        Problem(SquaredLoss(), L1(), X, y)


def test_problem_rejects_inf_target():
    X = np.zeros((10, 3))
    y = np.zeros(10)
    y[3] = np.inf
    with pytest.raises(ValueError, match="target contains NaN"):
        Problem(SquaredLoss(), L1(), X, y)


def test_problem_rejects_nan_in_sparse_design_data():
    X = sp.random(20, 5, density=0.3, format="csc", random_state=0).astype(np.float64)
    X.data[0] = np.nan
    y = np.zeros(20)
    with pytest.raises(ValueError, match="design contains NaN"):
        Problem(SquaredLoss(), L1(), X, y)


def test_problem_finiteness_opt_out():
    X = np.zeros((10, 3))
    X[0, 0] = np.nan
    y = np.zeros(10)
    # Opt-out: caller takes responsibility for handling the NaNs downstream.
    _problem_mod.CHECK_FINITE = False
    try:
        p = Problem(SquaredLoss(), L1(), X, y)
        assert p.n_samples == 10
    finally:
        _problem_mod.CHECK_FINITE = True


# ---------------------------------------------------------------- ElasticNet rho


def test_elastic_net_rejects_rho_zero():
    with pytest.raises(ValueError, match="rho must lie"):
        ElasticNet(rho=0.0)


def test_elastic_net_rejects_rho_above_one():
    with pytest.raises(ValueError, match="rho must lie"):
        ElasticNet(rho=1.5)


def test_elastic_net_rejects_rho_negative():
    with pytest.raises(ValueError, match="rho must lie"):
        ElasticNet(rho=-0.1)


def test_elastic_net_accepts_rho_at_boundary():
    ElasticNet(rho=1.0)  # boundary allowed; (0, 1] is the spec


# ---------------------------------------------------------------- GroupL1 partition


def test_group_l1_rejects_overlapping_groups():
    with pytest.raises(ValueError, match="more than one group"):
        GroupL1(groups=((0, 1, 2), (1, 3)))


def test_group_l1_rejects_empty_group():
    with pytest.raises(ValueError, match="empty"):
        GroupL1(groups=((0, 1), ()))


def test_group_l1_rejects_negative_index():
    with pytest.raises(ValueError, match="non-negative"):
        GroupL1(groups=((0, -1),))


def test_group_l1_rejects_weight_length_mismatch():
    with pytest.raises(ValueError, match="weights length"):
        GroupL1(groups=((0,), (1,)), weights=(1.0,))


def test_group_l1_accepts_valid_partition():
    p = GroupL1(groups=((0, 1), (2,)))
    assert len(p.groups) == 2


# ---------------------------------------------------------------- Vector-α preflight


def test_grad_search_rejects_vector_alpha_length_mismatch():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 5))
    y = rng.standard_normal(30)
    problem = Problem(SquaredLoss(), WeightedL1(), X, y)
    # Wrong length: 3 entries for a 5-feature problem.
    hp0 = np.array([0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="hp0 length"):
        grad_search(
            problem,
            hp0,
            solver=SklearnLasso(),
            criterion=HeldOutMSE(idx_train=np.arange(20), idx_val=np.arange(20, 30)),
            n_iter=1,
        )


def test_grad_search_rejects_non_positive_alpha():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    y = rng.standard_normal(30)
    problem = Problem(SquaredLoss(), L1(), X, y)
    with pytest.raises(ValueError, match="strictly positive"):
        grad_search(
            problem,
            0.0,
            solver=SklearnLasso(),
            criterion=HeldOutMSE(idx_train=np.arange(20), idx_val=np.arange(20, 30)),
            n_iter=1,
        )


# ---------------------------------------------------------------- as_scalar / as_vector


def test_as_scalar_rejects_vector():
    with pytest.raises(TypeError, match=r"scalar hyperparameter"):
        as_scalar(np.array([1.0, 2.0]))


def test_as_vector_rejects_scalar():
    with pytest.raises(TypeError, match=r"per-feature α"):
        as_vector(0.5, n_features=5)  # type: ignore[arg-type]


def test_as_vector_rejects_length_mismatch():
    with pytest.raises(TypeError, match=r"length mismatch"):
        as_vector(np.array([0.5, 0.5]), n_features=5)
