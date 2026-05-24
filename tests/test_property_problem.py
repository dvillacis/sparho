"""Hypothesis property tests for :class:`sparho.Problem` construction.

Sanity-level invariants over random (shape, density, datafit, penalty)
configurations. The existing ``test_problem.py`` and
``test_problem_validation.py`` cover specific fixtures; this file is the
shrinking-and-fuzzing complement.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sparho import (
    L1,
    ElasticNet,
    GroupL1,
    LogisticLoss,
    Problem,
    SquaredLoss,
    WeightedL1,
)

pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# Keep example counts modest so the suite remains fast on CI; coverage of
# shrinking is the goal, not exhaustive sweep.
_DEFAULT_SETTINGS = settings(max_examples=40, deadline=None)


@st.composite
def _dense_design(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    n = draw(st.integers(min_value=2, max_value=20))
    p = draw(st.integers(min_value=1, max_value=15))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    return X, y


@_DEFAULT_SETTINGS
@given(design=_dense_design())
def test_problem_shape_invariants(design: tuple[np.ndarray, np.ndarray]) -> None:
    """``n_samples`` / ``n_features`` always agree with ``design.shape``."""
    X, y = design
    problem = Problem(SquaredLoss(), L1(), X, y)
    assert problem.n_samples == X.shape[0]
    assert problem.n_features == X.shape[1]
    assert problem.target.shape == (X.shape[0],)


@_DEFAULT_SETTINGS
@given(
    design=_dense_design(),
    rho=st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_elastic_net_rho_round_trip(design: tuple[np.ndarray, np.ndarray], rho: float) -> None:
    """Any ``ρ ∈ (0, 1]`` round-trips through ElasticNet construction."""
    X, y = design
    problem = Problem(SquaredLoss(), ElasticNet(rho=rho), X, y)
    assert problem.penalty == ElasticNet(rho=rho)


@_DEFAULT_SETTINGS
@given(
    design=_dense_design(),
    rho=st.floats(allow_nan=False, allow_infinity=False).filter(lambda r: not (0.0 < r <= 1.0)),
)
def test_elastic_net_rejects_invalid_rho(design: tuple[np.ndarray, np.ndarray], rho: float) -> None:
    """Construction raises ``ValueError`` for any ``ρ`` outside ``(0, 1]``."""
    _ = design  # only need the shape strategy for parametrization
    try:
        ElasticNet(rho=rho)
    except ValueError:
        return
    raise AssertionError(f"ElasticNet accepted invalid rho={rho!r}")


@_DEFAULT_SETTINGS
@given(design=_dense_design())
def test_problem_rejects_target_length_mismatch(
    design: tuple[np.ndarray, np.ndarray],
) -> None:
    """``Problem`` raises if ``target.shape[0] != design.shape[0]``."""
    X, y = design
    bad_y = np.concatenate([y, y[:1]])  # length n+1
    try:
        Problem(SquaredLoss(), L1(), X, bad_y)
    except ValueError:
        return
    raise AssertionError("Problem accepted mismatched target length")


@_DEFAULT_SETTINGS
@given(design=_dense_design())
def test_problem_csc_round_trip(design: tuple[np.ndarray, np.ndarray]) -> None:
    """Building a ``Problem`` with a CSC design preserves ``n_samples``/``n_features``."""
    X, y = design
    Xs = sp.csc_matrix(X)
    problem = Problem(SquaredLoss(), L1(), Xs, y)
    assert problem.n_samples == X.shape[0]
    assert problem.n_features == X.shape[1]


@_DEFAULT_SETTINGS
@given(
    design=_dense_design(),
    n_groups=st.integers(min_value=1, max_value=5),
)
def test_group_l1_from_labels_partition(
    design: tuple[np.ndarray, np.ndarray], n_groups: int
) -> None:
    """``GroupL1.from_labels`` produces a covering, disjoint partition."""
    X, _ = design
    p = X.shape[1]
    # Build a label assignment that hits every group at least once.
    base = np.arange(p) % n_groups
    rng = np.random.default_rng(123)
    rng.shuffle(base)
    # Ensure every label 0..K-1 appears at least once.
    if len(set(base)) < n_groups:
        return  # discarded — strategy is allowed to skip degenerate draws
    g = GroupL1.from_labels(base.astype(int))
    seen: set[int] = set()
    for group in g.groups:
        for j in group:
            assert j not in seen
            seen.add(j)
    assert seen == set(range(p))


@_DEFAULT_SETTINGS
@given(design=_dense_design())
def test_logistic_problem_accepts_signed_labels(
    design: tuple[np.ndarray, np.ndarray],
) -> None:
    """LogisticLoss problems accept any ``y ∈ {−1, +1}^n``."""
    X, _ = design
    rng = np.random.default_rng(0)
    y = rng.choice([-1.0, 1.0], size=X.shape[0]).astype(np.float64)
    problem = Problem(LogisticLoss(), L1(), X, y)
    assert np.all(np.isin(problem.target, [-1.0, 1.0]))


@_DEFAULT_SETTINGS
@given(design=_dense_design())
def test_weighted_l1_construction(design: tuple[np.ndarray, np.ndarray]) -> None:
    """WeightedL1 problems carry no extra state; the α vector lives outside."""
    X, y = design
    problem = Problem(SquaredLoss(), WeightedL1(), X, y)
    assert isinstance(problem.penalty, WeightedL1)
    assert problem.n_features == X.shape[1]
