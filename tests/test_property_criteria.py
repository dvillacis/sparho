"""Hypothesis property tests for criteria and the KKT residual helper.

Coverage focus: ``HeldOutMSE`` matches a brute-force numpy implementation
across random splits; ``kkt_residual`` is non-negative and zero at the
naive optimum ``β = 0`` when ``α`` is large enough to suppress every feature.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from sparho import (  # noqa: E402
    L1,
    HeldOutMSE,
    Problem,
    SquaredLoss,
)
from sparho.adapters import SklearnLasso
from sparho.testing import kkt_residual

_SETTINGS = settings(max_examples=20, deadline=None)


@st.composite
def _split_problem(
    draw: st.DrawFn,
) -> tuple[Problem, np.ndarray, np.ndarray]:
    n = draw(st.integers(min_value=20, max_value=80))
    p = draw(st.integers(min_value=3, max_value=20))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    val_frac = draw(
        st.floats(min_value=0.2, max_value=0.5, allow_nan=False, allow_infinity=False)
    )
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[: max(1, p // 5)] = 1.0
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    perm = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    idx_val = np.asarray(perm[:n_val], dtype=np.int32)
    idx_train = np.asarray(perm[n_val:], dtype=np.int32)
    return Problem(SquaredLoss(), L1(), X, y), idx_train, idx_val


@_SETTINGS
@given(spec=_split_problem())
def test_held_out_mse_value_matches_numpy(
    spec: tuple[Problem, np.ndarray, np.ndarray],
) -> None:
    """``HeldOutMSE.value`` agrees with a direct numpy computation at ``β̂(α)``."""
    problem, idx_train, idx_val = spec
    alpha = 0.05
    crit = HeldOutMSE(idx_train=idx_train, idx_val=idx_val)
    val_from_crit = crit.value(problem, alpha, SklearnLasso(), tol=1e-8)
    # Recompute manually: fit on train, score on val.
    from sparho.criteria import _slice_problem  # internal helper, used by all criteria

    train_problem = _slice_problem(problem, idx_train)
    coef = SklearnLasso()(train_problem, alpha, tol=1e-8).coef
    X_val = problem.design[idx_val]
    y_val = problem.target[idx_val]
    resid = X_val @ coef - y_val
    val_naive = float(resid @ resid) / len(idx_val)
    assert val_from_crit == val_naive  # bit-identical — same code path


@_SETTINGS
@given(spec=_split_problem())
def test_kkt_residual_zero_above_alpha_max(
    spec: tuple[Problem, np.ndarray, np.ndarray],
) -> None:
    """For ``α > (1/n)‖X^T y‖_∞``, ``β = 0`` is optimal and KKT residual = 0."""
    problem, _, _ = spec
    X = problem.design
    y = problem.target
    n = problem.n_samples
    alpha_max = float(np.max(np.abs(X.T @ y))) / n
    alpha = 2.0 * (alpha_max + 1e-12)
    beta_zero = np.zeros(problem.n_features, dtype=np.float64)
    res = kkt_residual(problem, alpha, beta_zero)
    assert res == 0.0, f"KKT residual at β=0, α=2·α_max should be 0, got {res}"


@_SETTINGS
@given(spec=_split_problem())
def test_kkt_residual_nonnegative(
    spec: tuple[Problem, np.ndarray, np.ndarray],
) -> None:
    """``kkt_residual`` returns a non-negative finite scalar for any ``(β, α)``."""
    problem, _, _ = spec
    rng = np.random.default_rng(99)
    beta = rng.standard_normal(problem.n_features).astype(np.float64)
    for alpha in (1e-3, 1e-2, 1e-1, 1.0):
        res = kkt_residual(problem, alpha, beta)
        assert np.isfinite(res)
        assert res >= 0.0
