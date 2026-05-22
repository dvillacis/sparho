"""Hypothesis property tests for :func:`sparho.implicit_forward`.

Coverage focus: the hypergradient is finite, scale-coherent, and consistent
with finite differences across random Lasso problem shapes. ``α`` is
drawn from a log-uniform range to exercise both nearly-fully-active and
nearly-fully-sparse regimes.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sparho import L1, Problem, SquaredLoss
from sparho.adapters import SklearnLasso
from sparho.hypergrad import implicit_forward

_SETTINGS = settings(max_examples=15, deadline=None)


@st.composite
def _well_conditioned_lasso(
    draw: st.DrawFn,
) -> tuple[Problem, float, np.ndarray]:
    n = draw(st.integers(min_value=20, max_value=60))
    p = draw(st.integers(min_value=5, max_value=25))
    k = draw(st.integers(min_value=1, max_value=min(5, p)))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    log_alpha = draw(
        st.floats(min_value=-3.0, max_value=0.0, allow_nan=False, allow_infinity=False)
    )
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:k] = rng.standard_normal(k) + 2.0
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    # α normalized against α_max so we land in a useful regime regardless of n,p.
    alpha_max = float(np.max(np.abs(X.T @ y))) / n
    alpha = alpha_max * 10**log_alpha
    crit_w = np.random.default_rng(seed + 1).standard_normal(p)
    return Problem(SquaredLoss(), L1(), X, y), alpha, crit_w


@_SETTINGS
@given(spec=_well_conditioned_lasso())
def test_implicit_forward_is_finite(
    spec: tuple[Problem, float, np.ndarray],
) -> None:
    """For any well-conditioned Lasso problem, ``implicit_forward`` returns a finite scalar."""
    problem, alpha, crit_w = spec
    solver = SklearnLasso()
    result = solver(problem, alpha, tol=1e-8)
    hg = implicit_forward(problem, alpha, result, crit_w)
    assert np.isfinite(hg), f"hypergrad is not finite: {hg}"
    assert isinstance(hg, float)


@_SETTINGS
@given(spec=_well_conditioned_lasso())
def test_implicit_forward_fd_parity(
    spec: tuple[Problem, float, np.ndarray],
) -> None:
    """Hypergradient agrees with the central FD of a linear criterion ``w^T β̂(α)``.

    The criterion is the *linear* functional ``C(β) = w^T β`` so that
    ``∂C/∂β = w`` is the constant gradient threaded into ``implicit_forward``.
    This isolates the implicit-derivative ``dβ̂/dα`` itself.
    """
    problem, alpha, crit_w = spec
    if alpha <= 1e-6:
        pytest.skip("α too small — sklearn convergence and FD step both shaky")
    solver = SklearnLasso()
    result = solver(problem, alpha, tol=1e-10)
    hg = implicit_forward(problem, alpha, result, crit_w)
    if not np.isfinite(hg):
        pytest.skip("implicit_forward returned non-finite (ill-conditioned draw)")
    # Central FD on β̂(α) ⋅ w  (no scale factor — the criterion is exactly w^T β).
    eps = max(1e-6, 1e-4 * alpha)
    b_plus = solver(problem, alpha + eps, tol=1e-10).coef
    b_minus = solver(problem, alpha - eps, tol=1e-10).coef
    fd = crit_w @ (b_plus - b_minus) / (2.0 * eps)
    if not np.isfinite(fd):
        pytest.skip("FD reference is non-finite")
    # Active-set perturbation between α±ε breaks the linear-system parity;
    # accept either tight match OR active-set drift.
    a_active = np.flatnonzero(np.abs(b_plus) > 1e-10)
    a_active_minus = np.flatnonzero(np.abs(b_minus) > 1e-10)
    if not np.array_equal(a_active, a_active_minus):
        pytest.skip("active set shifted between α±ε — FD vs implicit not comparable")
    # Loose tol to absorb sklearn's inner-solver noise. The closed-form
    # parity test in ``test_hypergrad.py`` already enforces a much tighter
    # bound on a fixed fixture.
    rel = abs(hg - fd) / (abs(fd) + 1e-9)
    assert rel < 0.1, f"hg={hg}, fd={fd}, rel_err={rel}"
