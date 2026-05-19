from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from sparho import (
    IterationRecord,
    SearchResult,
    SearchState,
    SolverResult,
)


def _make_solver_result() -> SolverResult:
    return SolverResult(
        coef=np.zeros(5),
        active_set=np.array([], dtype=np.int32),
        dual_gap=0.0,
        n_iter=0,
    )


def test_solver_result_default_extras_is_independent_dict():
    a = _make_solver_result()
    b = _make_solver_result()
    a.extras["x"] = 1
    assert "x" not in b.extras  # field(default_factory=dict) is per-instance


def test_solver_result_is_frozen():
    r = _make_solver_result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.n_iter = 7  # type: ignore[misc]


def test_iteration_record_round_trip():
    rec = IterationRecord(iteration=2, hyperparam=0.1, value=3.14, grad_norm=0.01, n_inner_iter=42)
    rec2 = dataclasses.replace(rec, iteration=3)
    assert rec2.iteration == 3
    assert rec2.value == rec.value


def test_search_state_holds_opaque_optimizer_state():
    rec = IterationRecord(iteration=0, hyperparam=1.0, value=0.0, grad_norm=0.0, n_inner_iter=0)
    state = SearchState(
        iteration=0,
        hyperparam=1.0,
        value=0.0,
        grad=np.zeros(3),
        solver_result=_make_solver_result(),
        optimizer_state={"lr": 1.0, "step": 0},  # opaque
        history=(rec,),
    )
    assert state.optimizer_state == {"lr": 1.0, "step": 0}
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.iteration = 1  # type: ignore[misc]


def test_search_result_history_is_tuple():
    res = SearchResult(
        best_hyperparam=0.1,
        best_coef=np.zeros(3),
        history=(),
        converged=True,
        n_iter=0,
    )
    assert isinstance(res.history, tuple)
