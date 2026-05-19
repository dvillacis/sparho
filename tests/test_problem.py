from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp
from sparho import (
    L1,
    ElasticNet,
    LogisticLoss,
    Problem,
    SquaredLoss,
    WeightedL1,
)


def test_squared_loss_is_singleton_equal():
    assert SquaredLoss() == SquaredLoss()


def test_logistic_loss_is_singleton_equal():
    assert LogisticLoss() == LogisticLoss()


def test_l1_is_singleton_equal():
    assert L1() == L1()


def test_weighted_l1_is_singleton_equal():
    assert WeightedL1() == WeightedL1()


def test_elastic_net_carries_rho():
    assert ElasticNet(rho=0.5) != ElasticNet(rho=0.7)
    assert ElasticNet(rho=0.5) == ElasticNet(rho=0.5)


def test_problem_basic_construction_dense():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 7))
    y = rng.standard_normal(20)
    p = Problem(SquaredLoss(), L1(), X, y)
    assert p.n_samples == 20
    assert p.n_features == 7
    assert isinstance(p.datafit, SquaredLoss)
    assert isinstance(p.penalty, L1)


def test_problem_basic_construction_sparse():
    rng = np.random.default_rng(1)
    X = sp.random(50, 12, density=0.1, format="csc", random_state=rng).astype(np.float64)
    y = rng.standard_normal(50)
    p = Problem(SquaredLoss(), L1(), X, y)
    assert p.n_samples == 50
    assert p.n_features == 12


def test_problem_is_frozen():
    X = np.zeros((3, 2))
    y = np.zeros(3)
    p = Problem(SquaredLoss(), L1(), X, y)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.target = y  # type: ignore[misc]


def test_problem_replace_round_trip():
    X = np.zeros((3, 2))
    y = np.zeros(3)
    p = Problem(SquaredLoss(), L1(), X, y)
    p2 = dataclasses.replace(p, penalty=ElasticNet(rho=0.3))
    assert p2.datafit is p.datafit
    assert p2.design is p.design
    assert p2.target is p.target
    assert isinstance(p2.penalty, ElasticNet)
    assert p2.penalty.rho == 0.3
