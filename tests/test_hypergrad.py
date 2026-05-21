"""Hypergradient tests.

Two parallel checks for each (datafit, penalty) pair:

1. **Closed-form** — Solve a small problem with high tol, then compute
   ``dC/dα`` by directly inverting the restricted Hessian in numpy. Compare
   to ``implicit_forward`` within ``rtol=1e-6``.
2. **Finite-difference** — Refit at ``α ± ε`` and compare
   ``(C(α+ε) − C(α−ε)) / (2ε)`` to ``implicit_forward``. Looser
   tolerance to absorb the FD truncation + sklearn's inner-solver noise.
"""

from __future__ import annotations

import numpy as np
import pytest
from sparho import (
    L1,
    ElasticNet,
    LogisticLoss,
    Problem,
    SolverResult,
    SquaredLoss,
    WeightedL1,
)
from sparho.adapters import (
    SklearnElasticNet,
    SklearnLasso,
    SklearnLogisticRegression,
    SklearnWeightedLasso,
)
from sparho.hypergrad import implicit_forward

# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def lasso_problem() -> tuple[Problem, np.ndarray]:
    rng = np.random.default_rng(0)
    n, p, k = 60, 25, 5
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:k] = rng.standard_normal(k) + 2.0  # large signal
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    rng_crit = np.random.default_rng(1)
    crit_w = rng_crit.standard_normal(p)
    return Problem(SquaredLoss(), L1(), X, y), crit_w


@pytest.fixture(scope="module")
def enet_problem() -> tuple[Problem, np.ndarray]:
    rng = np.random.default_rng(2)
    n, p = 80, 30
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[[1, 4, 9, 17]] = [2.0, -1.5, 1.0, -2.5]
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    crit_w = np.random.default_rng(3).standard_normal(p)
    return Problem(SquaredLoss(), ElasticNet(rho=0.7), X, y), crit_w


@pytest.fixture(scope="module")
def wlasso_problem() -> tuple[Problem, np.ndarray]:
    rng = np.random.default_rng(4)
    n, p = 60, 20
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[[0, 3, 7, 12]] = [1.5, -2.0, 1.0, -1.5]
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    crit_w = np.random.default_rng(5).standard_normal(p)
    return Problem(SquaredLoss(), WeightedL1(), X, y), crit_w


@pytest.fixture(scope="module")
def logreg_problem() -> tuple[Problem, np.ndarray]:
    rng = np.random.default_rng(6)
    n, p = 200, 30
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[[1, 4, 9]] = [2.5, -2.0, 1.5]
    logits = X @ true_beta
    y = np.where(logits > 0, 1.0, -1.0)
    crit_w = np.random.default_rng(7).standard_normal(p)
    return Problem(LogisticLoss(), L1(), X, y), crit_w


# ---------------------------------------------------------------- helpers


def _closed_form_lasso_hypergrad(X, beta, active, crit_w):
    """``dC/dα = −⟨sign(β_A), ((1/n) X_A^T X_A)^{−1} ∂C/∂β_A⟩`` (scalar α).

    The ``1/n`` factor matches sklearn's ``(1/(2n)) ||y − Xβ||²`` objective.
    """
    n = X.shape[0]
    XA = X[:, active]
    grad_A = crit_w[active]
    v = np.linalg.solve(XA.T @ XA / n, grad_A)
    return float(-np.dot(np.sign(beta[active]), v))


def _closed_form_enet_hypergrad(X, beta, active, crit_w, alpha, rho):
    n = X.shape[0]
    XA = X[:, active]
    H = XA.T @ XA / n + alpha * (1.0 - rho) * np.eye(len(active))
    grad_A = crit_w[active]
    v = np.linalg.solve(H, grad_A)
    sign_A = np.sign(beta[active])
    return float(-np.dot(rho * sign_A + (1.0 - rho) * beta[active], v))


def _closed_form_wlasso_hypergrad(X, beta, active, crit_w):
    """Per-feature: dC/dα_k = −sign(β_k) · v[k_in_A] for k in A, 0 elsewhere."""
    n = X.shape[0]
    XA = X[:, active]
    grad_A = crit_w[active]
    v = np.linalg.solve(XA.T @ XA / n, grad_A)
    out = np.zeros(X.shape[1])
    out[active] = -np.sign(beta[active]) * v
    return out


# ---------------------------------------------------------------- Lasso


def test_implicit_forward_lasso_closed_form(lasso_problem):
    problem, crit_w = lasso_problem
    alpha = 0.05
    solver = SklearnLasso(tol=1e-12)
    result = solver(problem, alpha)
    hg = implicit_forward(problem, alpha, result, crit_w, tol=1e-12)
    cf = _closed_form_lasso_hypergrad(problem.design, result.coef, result.active_set, crit_w)
    assert hg == pytest.approx(cf, rel=1e-6, abs=1e-8)


def test_implicit_forward_lasso_finite_difference(lasso_problem):
    problem, crit_w = lasso_problem
    alpha = 0.05
    eps = 1e-5
    solver = SklearnLasso(tol=1e-12, max_iter=100_000)
    r0 = solver(problem, alpha)
    r_plus = solver(problem, alpha + eps)
    r_minus = solver(problem, alpha - eps)
    # Active set must be stable across the FD perturbation, otherwise the
    # hypergradient assumption (local invariance) is violated.
    np.testing.assert_array_equal(r0.active_set, r_plus.active_set)
    np.testing.assert_array_equal(r0.active_set, r_minus.active_set)
    fd = (np.dot(crit_w, r_plus.coef) - np.dot(crit_w, r_minus.coef)) / (2 * eps)
    hg = implicit_forward(problem, alpha, r0, crit_w, tol=1e-12)
    assert hg == pytest.approx(fd, rel=1e-3, abs=1e-4)


# ---------------------------------------------------------------- ElasticNet


def test_implicit_forward_elasticnet_closed_form(enet_problem):
    problem, crit_w = enet_problem
    alpha = 0.05
    solver = SklearnElasticNet(tol=1e-12)
    result = solver(problem, alpha)
    hg = implicit_forward(problem, alpha, result, crit_w, tol=1e-12)
    cf = _closed_form_enet_hypergrad(
        problem.design, result.coef, result.active_set, crit_w, alpha, problem.penalty.rho
    )
    assert hg == pytest.approx(cf, rel=1e-6, abs=1e-8)


def test_implicit_forward_elasticnet_finite_difference(enet_problem):
    problem, crit_w = enet_problem
    alpha = 0.05
    eps = 1e-5
    solver = SklearnElasticNet(tol=1e-12, max_iter=100_000)
    r0 = solver(problem, alpha)
    r_plus = solver(problem, alpha + eps)
    r_minus = solver(problem, alpha - eps)
    np.testing.assert_array_equal(r0.active_set, r_plus.active_set)
    np.testing.assert_array_equal(r0.active_set, r_minus.active_set)
    fd = (np.dot(crit_w, r_plus.coef) - np.dot(crit_w, r_minus.coef)) / (2 * eps)
    hg = implicit_forward(problem, alpha, r0, crit_w, tol=1e-12)
    assert hg == pytest.approx(fd, rel=1e-3, abs=1e-4)


# ---------------------------------------------------------------- WeightedL1


def test_implicit_forward_wlasso_closed_form(wlasso_problem):
    problem, crit_w = wlasso_problem
    alpha = np.full(problem.n_features, 0.05)
    solver = SklearnWeightedLasso(tol=1e-12)
    result = solver(problem, alpha)
    hg = implicit_forward(problem, alpha, result, crit_w, tol=1e-12)
    cf = _closed_form_wlasso_hypergrad(problem.design, result.coef, result.active_set, crit_w)
    np.testing.assert_allclose(hg, cf, rtol=1e-6, atol=1e-8)


def test_implicit_forward_wlasso_finite_difference(wlasso_problem):
    problem, crit_w = wlasso_problem
    alpha = np.full(problem.n_features, 0.05)
    eps = 1e-5
    solver = SklearnWeightedLasso(tol=1e-12, max_iter=100_000)
    r0 = solver(problem, alpha)
    hg = implicit_forward(problem, alpha, r0, crit_w, tol=1e-12)
    # Perturb three coordinates inside the active set; outside, the hypergradient
    # is exactly zero (no implicit dependence).
    active = r0.active_set
    for k in active[:3]:
        a_plus = alpha.copy()
        a_minus = alpha.copy()
        a_plus[k] += eps
        a_minus[k] -= eps
        r_plus = solver(problem, a_plus)
        r_minus = solver(problem, a_minus)
        # Skip the FD if the perturbation changed the active set.
        if not (
            np.array_equal(r_plus.active_set, active) and np.array_equal(r_minus.active_set, active)
        ):
            continue
        fd = (np.dot(crit_w, r_plus.coef) - np.dot(crit_w, r_minus.coef)) / (2 * eps)
        assert hg[k] == pytest.approx(fd, rel=1e-3, abs=1e-4)


def test_implicit_forward_wlasso_zero_outside_active(wlasso_problem):
    problem, crit_w = wlasso_problem
    alpha = np.full(problem.n_features, 0.05)
    result = SklearnWeightedLasso(tol=1e-12)(problem, alpha)
    hg = implicit_forward(problem, alpha, result, crit_w, tol=1e-12)
    assert isinstance(hg, np.ndarray)
    inactive_mask = np.ones(problem.n_features, dtype=bool)
    inactive_mask[result.active_set] = False
    np.testing.assert_array_equal(hg[inactive_mask], 0.0)


# ---------------------------------------------------------------- Logistic + L1


def test_implicit_forward_logistic_finite_difference(logreg_problem):
    problem, crit_w = logreg_problem
    alpha = 0.05
    eps = 1e-4
    solver = SklearnLogisticRegression(tol=1e-12, max_iter=200_000)
    r0 = solver(problem, alpha)
    r_plus = solver(problem, alpha + eps)
    r_minus = solver(problem, alpha - eps)
    # liblinear's tol is not iron-tight; require active-set stability and
    # compute a robust FD only on the stable support.
    if not (
        np.array_equal(r0.active_set, r_plus.active_set)
        and np.array_equal(r0.active_set, r_minus.active_set)
    ):
        pytest.skip("active set unstable across α±ε — FD check not meaningful")
    fd = (np.dot(crit_w, r_plus.coef) - np.dot(crit_w, r_minus.coef)) / (2 * eps)
    hg = implicit_forward(problem, alpha, r0, crit_w, tol=1e-10)
    # liblinear has loose convergence relative to coord-descent, so widen rtol.
    assert hg == pytest.approx(fd, rel=5e-2, abs=1e-3)


# ---------------------------------------------------------------- edge cases


def test_implicit_forward_empty_active_set_scalar():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((20, 5))
    y = rng.standard_normal(20)
    problem = Problem(SquaredLoss(), L1(), X, y)
    # Construct an "empty active set" SolverResult by hand (would normally come
    # from a very large α).
    empty = SolverResult(
        coef=np.zeros(5),
        active_set=np.array([], dtype=np.int32),
        dual_gap=0.0,
        n_iter=0,
    )
    crit_w = rng.standard_normal(5)
    hg = implicit_forward(problem, 1e3, empty, crit_w)
    assert hg == 0.0


def test_implicit_forward_empty_active_set_vector():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((20, 5))
    y = rng.standard_normal(20)
    problem = Problem(SquaredLoss(), WeightedL1(), X, y)
    empty = SolverResult(
        coef=np.zeros(5),
        active_set=np.array([], dtype=np.int32),
        dual_gap=0.0,
        n_iter=0,
    )
    crit_w = rng.standard_normal(5)
    hg = implicit_forward(problem, np.full(5, 1e3), empty, crit_w)
    assert isinstance(hg, np.ndarray)
    np.testing.assert_array_equal(hg, np.zeros(5))


# ---------------------------------------------------------------- ridge guard


def _near_singular_problem() -> tuple[Problem, SolverResult, np.ndarray]:
    """Synthetic problem whose restricted ``X_A^⊤ X_A`` is exactly singular.

    Three columns are exact duplicates of column 0, and all four sit in the
    active set. The Hessian has a 3-D null space, so vanilla CG cannot
    converge and any back-substitution is ill-defined.
    """
    rng = np.random.default_rng(42)
    n, p = 30, 8
    base = rng.standard_normal((n, p))
    # Duplicate columns 0 → also at indices 5, 6, 7.
    base[:, 5] = base[:, 0]
    base[:, 6] = base[:, 0]
    base[:, 7] = base[:, 0]
    y = rng.standard_normal(n)
    problem = Problem(SquaredLoss(), L1(), base, y)
    coef = np.zeros(p)
    coef[[0, 5, 6, 7]] = [0.1, 0.1, 0.1, 0.1]
    active = np.array([0, 5, 6, 7], dtype=np.int32)
    result = SolverResult(coef=coef, active_set=active, dual_gap=0.0, n_iter=0)
    crit_w = rng.standard_normal(p)
    return problem, result, crit_w


def test_implicit_forward_ridge_auto_recovers_on_singular():
    problem, result, crit_w = _near_singular_problem()
    hg = implicit_forward(problem, 0.05, result, crit_w)  # ridge=None → auto
    assert np.isfinite(hg), "auto-ridge must return a finite hypergradient"


def test_implicit_forward_ridge_zero_warns_and_returns_zero():
    problem, result, crit_w = _near_singular_problem()
    with pytest.warns(RuntimeWarning, match="CG failed"):
        hg = implicit_forward(problem, 0.05, result, crit_w, ridge=0.0)
    assert hg == 0.0, "ridge=0 + singular system must return the zero sentinel"


def test_implicit_forward_ridge_bias_is_benign_on_well_conditioned(lasso_problem):
    problem, crit_w = lasso_problem
    solver = SklearnLasso(tol=1e-9)
    sr = solver(problem, 0.05)
    hg_no_ridge = implicit_forward(problem, 0.05, sr, crit_w, ridge=0.0)
    hg_auto = implicit_forward(problem, 0.05, sr, crit_w)
    # Default ridge_rel=1e-10 should perturb the answer by far less than 1 %.
    np.testing.assert_allclose(hg_auto, hg_no_ridge, rtol=1e-3, atol=1e-8)
