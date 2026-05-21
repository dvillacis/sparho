"""Coverage for the v0.4 §3 observability surface.

Three contracts:
- `grad_search` / `hoag_search` invoke an optional `callback` once per
  appended `IterationRecord` (and once more on the HOAG rejection branch's
  replacement record).
- `IterationRecord.extras` is populated with `cg_status` (always) plus
  `inner_dual_gap` (when the criterion surfaces it). HOAG records additionally
  carry `step_size` and `L_estimate`.
- `LassoHO(verbose=1)` wires a default `_VerbosePrinter` callback that
  prints one line per outer iter to stdout.
"""

from __future__ import annotations

import numpy as np
import pytest
from sparho import (
    L1,
    CrossVal,
    HeldOutMSE,
    LassoHO,
    Problem,
    SquaredLoss,
    Sure,
    grad_search,
    hoag_search,
)
from sparho.adapters import SklearnLasso

_RNG = np.random.default_rng(0)


def _make_problem(n_samples: int = 60, n_features: int = 8):
    X = _RNG.standard_normal((n_samples, n_features))
    beta = np.zeros(n_features)
    beta[:3] = [1.0, -0.5, 0.7]
    y = X @ beta + 0.1 * _RNG.standard_normal(n_samples)
    return Problem(SquaredLoss(), L1(), X, y)


def _basic_criterion(n_samples: int):
    n_train = int(0.7 * n_samples)
    return HeldOutMSE(
        idx_train=np.arange(n_train),
        idx_val=np.arange(n_train, n_samples),
    )


# ---------------------------------------------------------------- callback


def test_grad_search_invokes_callback_once_per_record():
    problem = _make_problem()
    seen: list[int] = []

    def cb(rec):
        seen.append(rec.iteration)

    result = grad_search(
        problem,
        0.1,
        solver=SklearnLasso(tol=1e-6),
        criterion=_basic_criterion(problem.n_samples),
        n_iter=4,
        lr=0.05,
        callback=cb,
    )
    assert seen == [r.iteration for r in result.history]
    assert len(seen) == len(result.history)


def test_hoag_search_invokes_callback_at_least_once_per_record():
    problem = _make_problem()
    received: list[float] = []

    def cb(rec):
        received.append(rec.value)

    result = hoag_search(
        problem,
        0.1,
        solver=SklearnLasso(tol=1e-6),
        criterion=_basic_criterion(problem.n_samples),
        n_iter=4,
        inner_tol=1e-6,
        callback=cb,
    )
    # `>=`: the rejection branch fires the callback a second time on the
    # replacement record for that iter. The number of *appended* records
    # still equals len(history).
    assert len(received) >= len(result.history)


# ---------------------------------------------------------------- extras content


def test_extras_carry_cg_status_in_grad_search():
    problem = _make_problem()
    result = grad_search(
        problem,
        0.1,
        solver=SklearnLasso(tol=1e-6),
        criterion=_basic_criterion(problem.n_samples),
        n_iter=3,
        lr=0.05,
    )
    for rec in result.history:
        assert rec.extras.get("cg_status") in {"ok", "nonconvergence", "nonfinite"}


def test_hoag_extras_carry_step_size_and_l_estimate():
    problem = _make_problem()
    result = hoag_search(
        problem,
        0.1,
        solver=SklearnLasso(tol=1e-6),
        criterion=_basic_criterion(problem.n_samples),
        n_iter=4,
        inner_tol=1e-6,
    )
    # The break-on-converged record is appended *before* the step is computed
    # only for grad_search; HOAG appends after the cap, so step_size /
    # L_estimate are always present in non-degenerate records.
    seen_step = [rec for rec in result.history if "step_size" in rec.extras]
    assert seen_step, "no HOAG record carried step_size in extras"
    for rec in seen_step:
        assert isinstance(rec.extras["step_size"], float)
        assert isinstance(rec.extras["L_estimate"], float)
        assert float(rec.extras["step_size"]) > 0
        assert float(rec.extras["L_estimate"]) > 0


def test_inner_dual_gap_propagates_from_held_out_mse():
    problem = _make_problem()
    result = grad_search(
        problem,
        0.1,
        solver=SklearnLasso(tol=1e-6),
        criterion=_basic_criterion(problem.n_samples),
        n_iter=2,
        lr=0.05,
    )
    for rec in result.history:
        assert "inner_dual_gap" in rec.extras
        assert isinstance(rec.extras["inner_dual_gap"], float)


def test_inner_dual_gap_propagates_through_cross_val():
    problem = _make_problem()
    cv = CrossVal.kfold(problem.n_samples, k=3, shuffle=True, random_state=0)
    result = hoag_search(
        problem,
        0.1,
        solver=SklearnLasso(tol=1e-6),
        criterion=cv,
        n_iter=3,
        inner_tol=1e-6,
    )
    assert any("inner_dual_gap" in r.extras for r in result.history)


def test_inner_dual_gap_propagates_through_sure():
    problem = _make_problem()
    sure = Sure(sigma=0.1, random_state=0)
    result = grad_search(
        problem,
        0.1,
        solver=SklearnLasso(tol=1e-6),
        criterion=sure,
        n_iter=2,
        lr=0.05,
    )
    assert any("inner_dual_gap" in r.extras for r in result.history)


# ---------------------------------------------------------------- verbose


def test_lasso_ho_verbose_prints_one_line_per_iter(capsys: pytest.CaptureFixture[str]):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 6))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.standard_normal(60)
    LassoHO(alpha_init=0.1, n_iter=3, verbose=1).fit(X, y)
    captured = capsys.readouterr()
    # One line per iter (at least 1; HOAG can terminate early on convergence).
    lines = [line for line in captured.out.splitlines() if "LassoHO" in line]
    assert lines, f"expected verbose output, got {captured.out!r}"
    # Each line includes the iter counter, α, value, |∇θ|, and cg status.
    for line in lines:
        assert "iter" in line
        assert "α=" in line
        assert "value=" in line
        assert "cg=" in line


def test_lasso_ho_verbose_level_2_adds_step_and_l(capsys: pytest.CaptureFixture[str]):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 6))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.standard_normal(60)
    LassoHO(alpha_init=0.1, n_iter=3, verbose=2).fit(X, y)
    captured = capsys.readouterr()
    # At least one verbose=2 line carries the L estimate.
    assert "L=" in captured.out, captured.out
    assert "step=" in captured.out, captured.out


def test_lasso_ho_verbose_zero_is_silent(capsys: pytest.CaptureFixture[str]):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 6))
    y = X[:, 0] + 0.1 * rng.standard_normal(60)
    LassoHO(alpha_init=0.1, n_iter=3, verbose=0).fit(X, y)
    captured = capsys.readouterr()
    assert "LassoHO" not in captured.out
