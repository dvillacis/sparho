"""Group-L1 penalty: dataclass, FISTA solver, hypergradient.

Three layers of validation:

1. ``GroupL1`` dataclass — construction, ``from_labels`` round-trip,
   default ``w_k = √|G_k|``.
2. ``GroupLassoFista`` — convergence on a synthetic block-sparse design,
   subgradient KKT at the fitted ``β*``, ρ-singleton equivalence to plain
   Lasso, sparse-vs-dense parity.
3. ``implicit_forward`` — closed-form check (manually-inverted block
   Hessian) and finite-difference parity for the hypergradient w.r.t. ``α``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sparho import GroupL1, Problem, SquaredLoss
from sparho.adapters import GroupLassoFista, SklearnLasso
from sparho.hypergrad import implicit_forward

# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def block_problem() -> tuple[Problem, np.ndarray]:
    """Block-sparse regression problem with two active groups out of five."""
    rng = np.random.default_rng(0)
    n, group_size, n_groups = 80, 4, 5
    p = group_size * n_groups
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    # Activate groups 0 and 2 with strong signal.
    true_beta[0:group_size] = rng.standard_normal(group_size) + 2.0
    true_beta[2 * group_size : 3 * group_size] = rng.standard_normal(group_size) - 2.0
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    groups = tuple(
        tuple(range(k * group_size, (k + 1) * group_size)) for k in range(n_groups)
    )
    penalty = GroupL1(groups=groups)
    crit_w = np.random.default_rng(1).standard_normal(p)
    return Problem(SquaredLoss(), penalty, X, y), crit_w


# ---------------------------------------------------------------- dataclass


def test_group_l1_from_labels_round_trip():
    labels = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
    pen = GroupL1.from_labels(labels)
    assert pen.groups == ((0, 1), (2, 3, 4), (5,))
    assert pen.weights is None


def test_group_l1_from_labels_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        GroupL1.from_labels([0, -1, 0])


def test_group_l1_from_labels_rejects_empty_group():
    # Labels skip 1 ⇒ group 1 is empty between groups 0 and 2.
    with pytest.raises(ValueError, match="empty groups"):
        GroupL1.from_labels([0, 0, 2, 2])


def test_group_l1_is_frozen():
    pen = GroupL1(groups=((0, 1), (2,)))
    with pytest.raises(AttributeError):
        pen.weights = (1.0, 2.0)  # type: ignore[misc]


# ---------------------------------------------------------------- FISTA solver


def test_fista_recovers_block_sparse_support(block_problem):
    problem, _ = block_problem
    solver = GroupLassoFista(tol=1e-10, max_iter=20_000)
    result = solver(problem, 0.05)
    # The two true active groups (0 and 2) must be active.
    coef = result.coef
    group_norms = np.array(
        [np.linalg.norm(coef[list(g)]) for g in problem.penalty.groups]  # type: ignore[union-attr]
    )
    active_groups = np.flatnonzero(group_norms > 1e-6)
    np.testing.assert_array_equal(active_groups, np.array([0, 2]))


def test_fista_kkt_residual_decreases_with_tighter_tol(block_problem):
    problem, _ = block_problem
    loose = GroupLassoFista(tol=1e-3, max_iter=20_000)(problem, 0.05)
    tight = GroupLassoFista(tol=1e-10, max_iter=50_000)(problem, 0.05)
    assert tight.dual_gap < loose.dual_gap + 1e-12
    assert tight.dual_gap < 1e-5


def test_fista_singleton_groups_equivalent_to_lasso():
    """With every group a singleton and w=1, GroupL1 reduces to plain L1."""
    rng = np.random.default_rng(3)
    n, p = 100, 30
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[[1, 5, 17]] = [2.0, -1.5, 1.0]
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    groups = tuple((j,) for j in range(p))
    weights = tuple(1.0 for _ in range(p))
    g_problem = Problem(SquaredLoss(), GroupL1(groups=groups, weights=weights), X, y)
    l1_problem = Problem(SquaredLoss(), __import__("sparho").L1(), X, y)
    alpha = 0.1
    g_res = GroupLassoFista(tol=1e-10, max_iter=50_000)(g_problem, alpha)
    l1_res = SklearnLasso(tol=1e-12, max_iter=200_000)(l1_problem, alpha)
    np.testing.assert_allclose(g_res.coef, l1_res.coef, rtol=1e-3, atol=1e-4)


def test_fista_sparse_matches_dense(block_problem):
    problem, _ = block_problem
    X_sp = sp.csc_matrix(problem.design)
    sparse_problem = Problem(problem.datafit, problem.penalty, X_sp, problem.target)
    solver = GroupLassoFista(tol=1e-10, max_iter=20_000)
    r_dense = solver(problem, 0.05)
    r_sparse = solver(sparse_problem, 0.05)
    np.testing.assert_allclose(r_sparse.coef, r_dense.coef, rtol=1e-4, atol=1e-5)


def test_fista_warmstart_does_not_change_solution(block_problem):
    """Cold and warm should reach the same fixed point (Lasso is convex)."""
    problem, _ = block_problem
    solver = GroupLassoFista(tol=1e-10, max_iter=50_000)
    cold = solver(problem, 0.05)
    # Warm-start from a neighbouring α's solution.
    neighbour = solver(problem, 0.06)
    warm = solver(problem, 0.05, x0=neighbour.coef)
    np.testing.assert_allclose(warm.coef, cold.coef, rtol=1e-5, atol=1e-6)


def test_fista_rejects_wrong_problem_shape():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((10, 4))
    y = rng.standard_normal(10)
    pen = GroupL1(groups=((0, 1), (2, 3)))
    problem = Problem(SquaredLoss(), pen, X, y)
    with pytest.raises(ValueError, match="x0 must have shape"):
        GroupLassoFista()(problem, 0.1, x0=np.zeros(5))


# ---------------------------------------------------------------- hypergradient


def _closed_form_group_l1_hypergrad(
    X: np.ndarray,
    coef: np.ndarray,
    penalty: GroupL1,
    crit_w: np.ndarray,
    alpha: float,
) -> float:
    """Reference hypergradient via explicit block-Hessian inverse on active groups."""
    n = X.shape[0]
    active_groups = []
    active_feats: list[int] = []
    block_starts: list[int] = [0]
    norms = []
    weights = (
        np.asarray(penalty.weights, dtype=np.float64)
        if penalty.weights is not None
        else np.array([np.sqrt(len(g)) for g in penalty.groups])
    )
    w_active = []
    u_chunks = []
    for k, g in enumerate(penalty.groups):
        idx = list(g)
        beta_g = coef[idx]
        norm_g = float(np.linalg.norm(beta_g))
        if norm_g == 0.0:
            continue
        active_groups.append(k)
        active_feats.extend(idx)
        block_starts.append(len(active_feats))
        norms.append(norm_g)
        w_active.append(float(weights[k]))
        u_chunks.append(beta_g / norm_g)
    if not active_groups:
        return 0.0
    A = np.asarray(active_feats, dtype=np.int64)
    XA = X[:, A]
    M = XA.T @ XA / n  # data block
    # Add per-group projection curvature: (α w_k / r_k) (I − u_k u_k^T) on block G_k.
    for k_idx, (s, e) in enumerate(zip(block_starts[:-1], block_starts[1:], strict=True)):
        scale = alpha * w_active[k_idx] / norms[k_idx]
        u = u_chunks[k_idx]
        block = scale * (np.eye(e - s) - np.outer(u, u))
        M[s:e, s:e] += block
    grad_A = crit_w[A]
    v = np.linalg.solve(M, grad_A)
    # dC/dα = −Σ_k w_k ⟨u_k, v_{G_k}⟩
    total = 0.0
    for k_idx, (s, e) in enumerate(zip(block_starts[:-1], block_starts[1:], strict=True)):
        total += w_active[k_idx] * float(u_chunks[k_idx] @ v[s:e])
    return -total


def test_implicit_forward_group_l1_closed_form(block_problem):
    problem, crit_w = block_problem
    alpha = 0.05
    solver = GroupLassoFista(tol=1e-12, max_iter=50_000)
    result = solver(problem, alpha)
    hg = implicit_forward(problem, alpha, result, crit_w, tol=1e-12)
    cf = _closed_form_group_l1_hypergrad(
        problem.design, result.coef, problem.penalty, crit_w, alpha  # type: ignore[arg-type]
    )
    assert hg == pytest.approx(cf, rel=1e-4, abs=1e-6)


def test_implicit_forward_group_l1_finite_difference(block_problem):
    problem, crit_w = block_problem
    alpha = 0.05
    eps = 1e-5
    solver = GroupLassoFista(tol=1e-12, max_iter=100_000)
    r0 = solver(problem, alpha)
    r_plus = solver(problem, alpha + eps, x0=r0.coef)
    r_minus = solver(problem, alpha - eps, x0=r0.coef)
    # Active-group stability is the analogue of active-set stability for L1.
    norms_0 = np.array(
        [np.linalg.norm(r0.coef[list(g)]) > 1e-6 for g in problem.penalty.groups]  # type: ignore[union-attr]
    )
    norms_p = np.array(
        [np.linalg.norm(r_plus.coef[list(g)]) > 1e-6 for g in problem.penalty.groups]  # type: ignore[union-attr]
    )
    norms_m = np.array(
        [np.linalg.norm(r_minus.coef[list(g)]) > 1e-6 for g in problem.penalty.groups]  # type: ignore[union-attr]
    )
    if not (np.array_equal(norms_0, norms_p) and np.array_equal(norms_0, norms_m)):
        pytest.skip("active group set unstable across α±ε")
    fd = (np.dot(crit_w, r_plus.coef) - np.dot(crit_w, r_minus.coef)) / (2 * eps)
    hg = implicit_forward(problem, alpha, r0, crit_w, tol=1e-12)
    assert hg == pytest.approx(fd, rel=5e-3, abs=1e-3)


def test_implicit_forward_group_l1_zero_when_no_active_group(block_problem):
    """At huge α, every group is shrunk to zero ⇒ dC/dα = 0."""
    problem, crit_w = block_problem
    # Construct a "no active group" SolverResult by hand.
    from sparho import SolverResult

    empty = SolverResult(
        coef=np.zeros(problem.n_features),
        active_set=np.array([], dtype=np.int32),
        dual_gap=0.0,
        n_iter=0,
    )
    hg = implicit_forward(problem, 1e6, empty, crit_w)
    assert hg == 0.0
