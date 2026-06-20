"""Cross-algorithm equivalence for the hypergradient family.

All algorithms compute the same quantity ``dC/dα`` at a converged ``β*``; they
just differ in *how*. The strongest correctness signal is that independent
derivations agree:

- ``implicit`` — matrix-free CG on the restricted KKT Hessian.
- ``implicit_forward`` — support-restricted BCD Jacobian fixed point (Rust).

and that both agree with a central finite difference of ``wᵀβ̂(α)``. Dense and
CSC code paths must also agree.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sparho import L1, ElasticNet, Problem, SquaredLoss, WarmStartHypergrad, WeightedL1
from sparho.adapters import (
    NativeBcdLasso,
    SklearnElasticNet,
    SklearnLasso,
    SklearnWeightedLasso,
)
from sparho.hypergrad import backward, forward, implicit, implicit_forward


@pytest.fixture(scope="module")
def lasso_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    n, p = 90, 35
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[[1, 5, 9, 14, 22]] = [2.0, -1.5, 0.8, 1.3, -0.6]
    y = X @ beta + 0.1 * rng.standard_normal(n)
    w = rng.standard_normal(p)  # criterion ∂C/∂β
    return X, y, w


def test_implicit_forward_matches_implicit_dense(lasso_data):
    X, y, w = lasso_data
    problem = Problem(SquaredLoss(), L1(), X, y)
    alpha = 0.07
    sr = SklearnLasso(tol=1e-12, max_iter=200_000)(problem, alpha)
    hg_cg = implicit(problem, alpha, sr, w, tol=1e-12)
    hg_if = implicit_forward(problem, alpha, sr, w, tol=1e-12)
    np.testing.assert_allclose(hg_if, hg_cg, rtol=1e-6, atol=1e-9)


def test_implicit_forward_matches_implicit_sparse(lasso_data):
    X, y, w = lasso_data
    problem = Problem(SquaredLoss(), L1(), sp.csc_matrix(X), y)
    alpha = 0.07
    sr = SklearnLasso(tol=1e-12, max_iter=200_000)(problem, alpha)
    hg_cg = implicit(problem, alpha, sr, w, tol=1e-12)
    hg_if = implicit_forward(problem, alpha, sr, w, tol=1e-12)
    np.testing.assert_allclose(hg_if, hg_cg, rtol=1e-6, atol=1e-9)


def test_implicit_forward_dense_csc_agree(lasso_data):
    X, y, w = lasso_data
    alpha = 0.07
    p_dense = Problem(SquaredLoss(), L1(), X, y)
    p_csc = Problem(SquaredLoss(), L1(), sp.csc_matrix(X), y)
    sr_d = NativeBcdLasso(tol=1e-12, max_iter=200_000)(p_dense, alpha)
    sr_s = NativeBcdLasso(tol=1e-12, max_iter=200_000)(p_csc, alpha)
    hg_d = implicit_forward(p_dense, alpha, sr_d, w, tol=1e-12)
    hg_s = implicit_forward(p_csc, alpha, sr_s, w, tol=1e-12)
    np.testing.assert_allclose(hg_d, hg_s, rtol=1e-7, atol=1e-9)


def test_implicit_forward_matches_finite_difference(lasso_data):
    X, y, w = lasso_data
    problem = Problem(SquaredLoss(), L1(), X, y)
    alpha = 0.07
    solver = SklearnLasso(tol=1e-12, max_iter=200_000)
    sr = solver(problem, alpha)
    hg = implicit_forward(problem, alpha, sr, w, tol=1e-12)
    # Central FD of wᵀβ̂(α); the support is stable for a small step here.
    eps = 1e-6
    bp = solver(problem, alpha + eps).coef
    bm = solver(problem, alpha - eps).coef
    fd = float(w @ (bp - bm) / (2 * eps))
    np.testing.assert_allclose(hg, fd, rtol=1e-4, atol=1e-6)


def test_forward_matches_implicit_forward(lasso_data):
    # Forward (joint solve over all features) equals ImplicitForward (support
    # post-solve) at the optimum, dense and CSC.
    X, y, w = lasso_data
    alpha = 0.07
    for design in (X, sp.csc_matrix(X)):
        problem = Problem(SquaredLoss(), L1(), design, y)
        sr = SklearnLasso(tol=1e-12, max_iter=200_000)(problem, alpha)
        hg_fwd = forward(problem, alpha, sr, w, tol=1e-12)
        hg_if = implicit_forward(problem, alpha, sr, w, tol=1e-12)
        np.testing.assert_allclose(hg_fwd, hg_if, rtol=1e-6, atol=1e-9)


def test_forward_matches_finite_difference(lasso_data):
    X, y, w = lasso_data
    problem = Problem(SquaredLoss(), L1(), X, y)
    alpha = 0.07
    solver = SklearnLasso(tol=1e-12, max_iter=200_000)
    sr = solver(problem, alpha)
    hg = forward(problem, alpha, sr, w, tol=1e-12)
    eps = 1e-6
    bp = solver(problem, alpha + eps).coef
    bm = solver(problem, alpha - eps).coef
    fd = float(w @ (bp - bm) / (2 * eps))
    np.testing.assert_allclose(hg, fd, rtol=1e-4, atol=1e-6)


def test_backward_matches_implicit_forward(lasso_data):
    # Reverse-mode replay equals ImplicitForward at the optimum (dense L1).
    X, y, w = lasso_data
    problem = Problem(SquaredLoss(), L1(), X, y)
    alpha = 0.07
    sr = SklearnLasso(tol=1e-12, max_iter=200_000)(problem, alpha)
    hg_back = backward(problem, alpha, sr, w, tol=1e-12)
    hg_if = implicit_forward(problem, alpha, sr, w, tol=1e-12)
    np.testing.assert_allclose(hg_back, hg_if, rtol=1e-6, atol=1e-9)


def test_backward_sparse_delegates_to_implicit_forward(lasso_data):
    # Sparse designs have no native reverse kernel; Backward delegates (same answer).
    X, y, w = lasso_data
    problem = Problem(SquaredLoss(), L1(), sp.csc_matrix(X), y)
    alpha = 0.07
    sr = SklearnLasso(tol=1e-12, max_iter=200_000)(problem, alpha)
    hg_back = backward(problem, alpha, sr, w, tol=1e-12)
    hg_if = implicit_forward(problem, alpha, sr, w, tol=1e-12)
    np.testing.assert_allclose(hg_back, hg_if, rtol=1e-9, atol=1e-12)


def test_implicit_forward_enet_matches_implicit(lasso_data):
    X, y, w = lasso_data
    alpha, rho = 0.07, 0.6
    for design in (X, sp.csc_matrix(X)):
        problem = Problem(SquaredLoss(), ElasticNet(rho=rho), design, y)
        sr = SklearnElasticNet(tol=1e-12, max_iter=300_000)(problem, alpha)
        hg_cg = implicit(problem, alpha, sr, w, tol=1e-12)
        hg_if = implicit_forward(problem, alpha, sr, w, tol=1e-12)
        np.testing.assert_allclose(hg_if, hg_cg, rtol=1e-6, atol=1e-9)


def test_implicit_forward_wl1_matches_implicit(lasso_data):
    X, y, w = lasso_data
    av = np.full(X.shape[1], 0.07)
    for design in (X, sp.csc_matrix(X)):
        problem = Problem(SquaredLoss(), WeightedL1(), design, y)
        sr = SklearnWeightedLasso(tol=1e-12, max_iter=300_000)(problem, av)
        hg_cg = implicit(problem, av, sr, w, tol=1e-12)
        hg_if = implicit_forward(problem, av, sr, w, tol=1e-12)
        np.testing.assert_allclose(hg_if, hg_cg, rtol=1e-6, atol=1e-9)
        # Vector-valued: exactly zero outside the active set.
        inactive = np.setdiff1d(np.arange(X.shape[1]), sr.active_set)
        assert np.all(hg_if[inactive] == 0.0)


def test_warm_start_matches_cold_across_alpha_sweep(lasso_data):
    # Warm-starting the Jacobian must not change the converged answer, even as the
    # active set changes between outer iterations (convexity).
    X, y, w = lasso_data
    problem = Problem(SquaredLoss(), L1(), X, y)
    ws = WarmStartHypergrad(tol=1e-12)
    for alpha in (0.2, 0.1, 0.05, 0.03):
        sr = SklearnLasso(tol=1e-12, max_iter=200_000)(problem, alpha)
        cold = implicit_forward(problem, alpha, sr, w, tol=1e-12)
        warm = ws(problem, alpha, sr, w)
        np.testing.assert_allclose(warm, cold, rtol=1e-6, atol=1e-9)


def test_warm_start_matches_cold_enet_and_wl1(lasso_data):
    X, y, w = lasso_data
    p = X.shape[1]
    ws_e = WarmStartHypergrad(tol=1e-12)
    pe = Problem(SquaredLoss(), ElasticNet(rho=0.5), X, y)
    for alpha in (0.1, 0.05):
        sr = SklearnElasticNet(tol=1e-12, max_iter=200_000)(pe, alpha)
        np.testing.assert_allclose(
            ws_e(pe, alpha, sr, w),
            implicit_forward(pe, alpha, sr, w, tol=1e-12),
            rtol=1e-6,
            atol=1e-9,
        )
    ws_w = WarmStartHypergrad(tol=1e-12)
    pw = Problem(SquaredLoss(), WeightedL1(), X, y)
    for av in (np.full(p, 0.1), np.full(p, 0.05)):
        sr = SklearnWeightedLasso(tol=1e-12, max_iter=200_000)(pw, av)
        np.testing.assert_allclose(
            ws_w(pw, av, sr, w), implicit_forward(pw, av, sr, w, tol=1e-12), rtol=1e-6, atol=1e-9
        )


def test_implicit_forward_native_solver_matches_sklearn_solver(lasso_data):
    # The hypergradient should not depend on which adapter produced β*.
    X, y, w = lasso_data
    problem = Problem(SquaredLoss(), L1(), X, y)
    alpha = 0.07
    sr_sk = SklearnLasso(tol=1e-12, max_iter=200_000)(problem, alpha)
    sr_bcd = NativeBcdLasso(tol=1e-12, max_iter=200_000)(problem, alpha)
    hg_sk = implicit_forward(problem, alpha, sr_sk, w, tol=1e-12)
    hg_bcd = implicit_forward(problem, alpha, sr_bcd, w, tol=1e-12)
    np.testing.assert_allclose(hg_sk, hg_bcd, rtol=1e-6, atol=1e-9)
