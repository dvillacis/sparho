"""Implicit-differentiation hypergradient.

At v0.1 we ship a single mode — ``implicit_forward`` — which restricts the
KKT linear system to the active set returned by the inner solver and solves
it via matrix-free conjugate gradients (the Hessian is symmetric positive
definite on the active set, at the converged ``β*``).

Math (KKT-derived, no prox-Jacobian needed):

  Inner stationarity on the active set ``A``:
      ∇_A L(β*) + ∂R/∂β_A(β*; α) = 0.
  Differentiating w.r.t. α:
      (H_L,AA + ∂²R/∂β²|_A) · dβ*_A/dα + ∂²R/∂α∂β|_A = 0
  ⇒   M_AA · dβ*_A/dα = − r(β*_A; α)

with ``M_AA`` and ``r`` depending on the (datafit, penalty) pair. The
outer-loop hypergradient ``dC/dα`` follows by chain rule with the criterion
gradient ``∂C/∂β`` passed in by the caller.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import assert_never

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg

from . import _core
from .core.types import Array, Hyperparam
from .problem import (
    L1,
    ElasticNet,
    GroupL1,
    LogisticLoss,
    Problem,
    SquaredLoss,
    WeightedL1,
)
from .state import SolverResult

_MatVec = Callable[[np.ndarray], np.ndarray]


_DEFAULT_RIDGE_REL = 1e-10


@dataclass(frozen=True, slots=True)
class _GroupL1ActiveInfo:
    """Precomputed active-group structure used across hypergrad helpers.

    ``active_features`` is the union of features in active groups, laid out
    one group at a time so ``group_starts[k]:group_starts[k+1]`` slices the
    block belonging to the ``k``-th *active* group. ``u_concat`` stores
    ``β_{G_k}/‖β_{G_k}‖`` in the same layout.
    """

    active_features: np.ndarray  # int32, length |A|
    group_starts: np.ndarray  # int32, length n_active_groups + 1
    u_concat: np.ndarray  # float64, length |A|
    group_norms: np.ndarray  # float64, length n_active_groups
    weights: np.ndarray  # float64, length n_active_groups


def _resolved_group_weights(penalty: GroupL1) -> np.ndarray:
    """Resolve per-group weights, falling back to ``w_k = √|G_k|`` when unset."""
    if penalty.weights is not None:
        w = np.asarray(penalty.weights, dtype=np.float64)
        if w.shape != (len(penalty.groups),):
            raise ValueError(
                f"GroupL1.weights length ({w.shape[0]}) must equal len(groups) "
                f"({len(penalty.groups)})"
            )
        return w
    return np.asarray([np.sqrt(len(g)) for g in penalty.groups], dtype=np.float64)


def _resolve_group_l1_active(penalty: GroupL1, coef: np.ndarray) -> _GroupL1ActiveInfo:
    """Active groups by ``‖β_{G_k}‖ > 0``, with all coords of each active group included.

    Group Lasso's prox produces "whole-block" sparsity generically: either
    ``β_{G_k} = 0`` or ``β_{G_k} ≠ 0`` componentwise. Internal-zero coords inside
    an active group still need to enter the implicit-diff active set (the
    KKT identity ties them to the active-group subgradient), so we expand
    here from the per-group norm rather than ``flatnonzero(coef)``.
    """
    weights = _resolved_group_weights(penalty)
    active_feats: list[int] = []
    starts: list[int] = [0]
    u_chunks: list[np.ndarray] = []
    norms: list[float] = []
    w_active: list[float] = []
    for k, g in enumerate(penalty.groups):
        idx = np.fromiter(g, dtype=np.int64, count=len(g))
        beta_g = coef[idx]
        norm_g = float(np.linalg.norm(beta_g))
        if norm_g == 0.0:
            continue
        active_feats.extend(int(j) for j in idx)
        u_chunks.append(beta_g / norm_g)
        norms.append(norm_g)
        w_active.append(float(weights[k]))
        starts.append(len(active_feats))
    return _GroupL1ActiveInfo(
        active_features=np.asarray(active_feats, dtype=np.int32),
        group_starts=np.asarray(starts, dtype=np.int32),
        u_concat=(
            np.concatenate(u_chunks).astype(np.float64) if u_chunks else np.zeros(0, np.float64)
        ),
        group_norms=np.asarray(norms, dtype=np.float64),
        weights=np.asarray(w_active, dtype=np.float64),
    )


def implicit_forward(
    problem: Problem,
    hyperparam: Hyperparam,
    solver_result: SolverResult,
    criterion_grad_beta: Array,
    *,
    tol: float = 1e-8,
    maxiter: int | None = None,
    ridge: float | None = None,
) -> Hyperparam:
    """Compute ``dC/dα`` by implicit differentiation, restricted to the active set.

    Parameters
    ----------
    problem
        The inner problem.
    hyperparam
        Current ``α``; scalar for ``L1`` / ``ElasticNet``, length-``n_features``
        vector for ``WeightedL1``.
    solver_result
        Converged inner solution. Only ``coef`` and ``active_set`` are read.
    criterion_grad_beta
        ``∂C/∂β`` at ``β*``, as a ``(n_features,)`` array. Entries outside
        ``active_set`` are unused (they multiply zero rows of ``dβ*/dα``).
    tol
        CG absolute and relative tolerance.
    maxiter
        CG maximum iterations. Default ``2 · |A| + 10``.
    ridge
        Tikhonov regularization added to the KKT Hessian as ``M_AA + ε·I``
        to keep CG well-posed when the active-set restricted Hessian is
        near-singular (e.g. dense designs with collinear features). The
        induced hypergradient bias is bounded by ``O(ε / λ_min(M_AA))`` —
        for any direction whose corresponding eigenvalue is well above
        ``ε`` the bias is negligible. ``None`` (default) auto-selects
        ``ε = 1e-10 · trace(M_AA) / |A|`` so ε tracks the operator's
        natural scale; pass ``0.0`` to disable.

    Returns
    -------
    hypergradient
        Scalar for ``L1`` / ``ElasticNet``; ``(n_features,)`` array for
        ``WeightedL1`` (entries outside the active set are exactly zero).
    """
    n_features = problem.n_features
    penalty = problem.penalty

    # GroupL1 has a different active-set semantics ("active groups expanded to
    # all their coords") than the per-feature penalties; resolve it here.
    group_info: _GroupL1ActiveInfo | None
    if isinstance(penalty, GroupL1):
        group_info = _resolve_group_l1_active(penalty, solver_result.coef)
        active = group_info.active_features
    else:
        group_info = None
        active = solver_result.active_set

    # Empty active set ⇒ β* doesn't move under small α perturbations,
    # so dC/dα = ∂C/∂α (which we treat as zero — criteria depending only
    # on β* contribute nothing in that case).
    if active.size == 0:
        if isinstance(penalty, WeightedL1):
            return np.zeros(n_features, dtype=np.float64)
        return 0.0

    beta = solver_result.coef
    beta_A = beta[active]
    grad_C_A = np.ascontiguousarray(criterion_grad_beta[active], dtype=np.float64)
    sign_A = np.sign(beta_A)

    matvec_raw = _build_hess_matvec(problem, hyperparam, active, beta, group_info)
    ridge_eps = _resolve_ridge(ridge, problem, hyperparam, active, beta, group_info)
    matvec = _ridge_wrap(matvec_raw, ridge_eps)
    n_active = active.size
    op = LinearOperator((n_active, n_active), matvec=matvec, dtype=np.float64)
    if maxiter is None:
        maxiter = 2 * n_active + 10
    v, info = cg(op, grad_C_A, rtol=tol, atol=tol, maxiter=maxiter)
    v_finite = bool(np.all(np.isfinite(v)))
    if info != 0 or not v_finite:
        warnings.warn(
            f"implicit_forward: CG failed (info={info}, finite={v_finite}); "
            "returning zero hypergradient for this iter — outer step will stall",
            RuntimeWarning,
            stacklevel=2,
        )
        if isinstance(penalty, WeightedL1):
            return np.zeros(n_features, dtype=np.float64)
        return 0.0

    # Compose with ∂²R/∂α∂β|_A — the penalty's α-Jacobian on the active set.
    match penalty:
        case L1():
            return float(-np.dot(sign_A, v))
        case ElasticNet(rho=rho):
            r = rho * sign_A + (1.0 - rho) * beta_A
            return float(-np.dot(r, v))
        case WeightedL1():
            out = np.zeros(n_features, dtype=np.float64)
            out[active] = -sign_A * v
            return np.asarray(out, dtype=np.float64)
        case GroupL1():
            assert group_info is not None  # set above when isinstance(penalty, GroupL1)
            jac_alpha = np.empty_like(v)
            starts = group_info.group_starts
            for k_idx, w_k in enumerate(group_info.weights):
                s, e = int(starts[k_idx]), int(starts[k_idx + 1])
                jac_alpha[s:e] = w_k * group_info.u_concat[s:e]
            return float(-np.dot(jac_alpha, v))
        case _:
            assert_never(penalty)


def _build_hess_matvec(
    problem: Problem,
    hyperparam: Hyperparam,
    active: np.ndarray,
    beta: np.ndarray,
    group_info: _GroupL1ActiveInfo | None = None,
) -> _MatVec:
    """Construct the augmented Hessian–vector callback restricted to ``active``.

    ``M_AA · v = H_L,AA · v + ∂²R/∂β²|_A · v``.

    For L1 / WeightedL1 the penalty curvature is zero; for ElasticNet it's a
    uniform diagonal ``α·(1−ρ)·I``; for GroupL1 it's *block-diagonal*: each
    active group ``G_k`` contributes ``(α·w_k/r_k) · (I − u_k u_kᵀ)``. The
    GroupL1 case requires the precomputed ``group_info``.
    """
    datafit = problem.datafit
    penalty = problem.penalty

    match datafit:
        case SquaredLoss():
            data_matvec = _build_ls_data_matvec(problem.design, problem.n_samples, active)
        case LogisticLoss():
            data_matvec = _build_logistic_data_matvec(problem.design, problem.target, beta, active)
        case _:
            assert_never(datafit)

    # Penalty curvature on A.
    match penalty:
        case L1() | WeightedL1():
            return data_matvec
        case ElasticNet(rho=rho):
            penalty_curv = float(np.asarray(hyperparam)) * (1.0 - rho)

            def matvec_uniform(v: np.ndarray) -> np.ndarray:
                return data_matvec(v) + penalty_curv * v

            return matvec_uniform
        case GroupL1():
            assert group_info is not None  # required for GroupL1 path
            alpha = float(np.asarray(hyperparam))
            starts = group_info.group_starts
            u_concat = group_info.u_concat
            weights = group_info.weights
            norms = group_info.group_norms

            def matvec_group(v: np.ndarray) -> np.ndarray:
                out = data_matvec(v).copy()
                for k_idx in range(weights.size):
                    s, e = int(starts[k_idx]), int(starts[k_idx + 1])
                    u_k = u_concat[s:e]
                    v_k = v[s:e]
                    scale = alpha * weights[k_idx] / norms[k_idx]
                    # (I − u_k u_kᵀ) v_k = v_k − (u_k·v_k) u_k.
                    out[s:e] += scale * (v_k - (u_k @ v_k) * u_k)
                return out

            return matvec_group
        case _:
            assert_never(penalty)


def _resolve_ridge(
    ridge: float | None,
    problem: Problem,
    hyperparam: Hyperparam,
    active: np.ndarray,
    beta: np.ndarray,
    group_info: _GroupL1ActiveInfo | None = None,
) -> float:
    """Resolve the Tikhonov ε for ``M_AA + ε·I``.

    ``ridge=None`` auto-scales to ``1e-10 · trace(M_AA) / |A|`` so ε tracks
    the operator's natural diagonal magnitude; ``ridge=0.0`` disables;
    explicit ``ridge=ε`` passes through. Diagonal computation is cheap —
    one column-norm pass over ``X_A`` plus the penalty diagonal term.
    """
    if ridge is not None:
        return float(ridge)

    datafit = problem.datafit
    penalty = problem.penalty

    match datafit:
        case SquaredLoss():
            data_diag_mean = _ls_hess_diag_mean(problem.design, problem.n_samples, active)
        case LogisticLoss():
            data_diag_mean = _logistic_hess_diag_mean(problem.design, beta, active)
        case _:
            assert_never(datafit)

    match penalty:
        case L1() | WeightedL1():
            penalty_curv = 0.0
        case ElasticNet(rho=rho):
            penalty_curv = float(np.asarray(hyperparam)) * (1.0 - rho)
        case GroupL1():
            assert group_info is not None
            # Block-projection trace per group is ``(|G_k| − 1) · α w_k / r_k``;
            # averaged across all |A| active features gives the operator's natural
            # diagonal scale.
            if active.size == 0:
                penalty_curv = 0.0
            else:
                alpha = float(np.asarray(hyperparam))
                starts = group_info.group_starts
                sizes = np.diff(starts).astype(np.float64)
                trace_blocks = (sizes - 1.0) * alpha * group_info.weights / group_info.group_norms
                penalty_curv = float(trace_blocks.sum() / active.size)
        case _:
            assert_never(penalty)

    return _DEFAULT_RIDGE_REL * (data_diag_mean + penalty_curv)


def _ridge_wrap(matvec: _MatVec, eps: float) -> _MatVec:
    """Return ``v ↦ matvec(v) + eps·v`` when ε > 0; pass through otherwise."""
    if eps <= 0.0:
        return matvec

    def wrapped(v: np.ndarray) -> np.ndarray:
        return matvec(v) + eps * v

    return wrapped


def _ls_hess_diag_mean(design: object, n_samples: int, active: np.ndarray) -> float:
    """``mean_j (1/n) · ||X[:, A_j]||²`` — average diagonal of the LS Hessian on A."""
    if sp.issparse(design):
        X_A = design[:, active]  # type: ignore[index]
        col_sq = np.asarray(X_A.multiply(X_A).sum(axis=0)).ravel()
    else:
        X_A = np.ascontiguousarray(design[:, active])  # type: ignore[index]
        col_sq = np.einsum("ij,ij->j", X_A, X_A)
    return float(col_sq.mean()) / float(n_samples)


def _logistic_hess_diag_mean(design: object, beta: Array, active: np.ndarray) -> float:
    """``mean_j Σᵢ wᵢ · X[i, A_j]²`` with ``w = σ(Xβ)(1−σ(Xβ))``."""
    z = design @ beta  # type: ignore[operator]
    sig = 1.0 / (1.0 + np.exp(-z))
    w = sig * (1.0 - sig)
    if sp.issparse(design):
        X_A = design[:, active]  # type: ignore[index]
        col_w_sq = np.asarray(X_A.multiply(X_A).T @ w).ravel()
    else:
        X_A = np.ascontiguousarray(design[:, active])  # type: ignore[index]
        col_w_sq = (X_A * X_A).T @ w
    return float(col_w_sq.mean())


def _build_ls_data_matvec(design: object, n_samples: int, active: np.ndarray) -> _MatVec:
    """``v ↦ (1/n) X_A^T (X_A v)``, dispatched on design density.

    The ``1/n`` factor matches sklearn's ``(1/(2n)) ||y − Xβ||²`` convention —
    all v0.1 adapters use this normalization, so all closed-form math here
    inherits it. If we ever add a "raw" SquaredLoss variant the scaling
    will need to be promoted to a property of the datafit tag.

    Dense designs go through numpy/BLAS GEMVs (per CLAUDE.md: don't port
    BLAS-bound matvecs to Rust). ``X_A`` is materialized **once** outside
    the matvec closure and reused across CG iterations. Sparse designs use
    the Rust CSC kernel ``_core.restricted_ls_hessian_matvec``, which
    iterates active columns of the CSC structure directly without
    densification.
    """
    inv_n = 1.0 / n_samples
    if not sp.issparse(design):
        XA = np.ascontiguousarray(np.asarray(design)[:, active], dtype=np.float64)

        def matvec_dense(v: np.ndarray) -> np.ndarray:
            return np.asarray(inv_n * (XA.T @ (XA @ v)), dtype=np.float64)

        return matvec_dense

    if design.format != "csc":  # type: ignore[attr-defined]
        X_csc = design.tocsc()  # type: ignore[attr-defined]
    else:
        X_csc = design
    indptr = X_csc.indptr.astype(np.int32)
    indices = X_csc.indices.astype(np.int32)
    data = np.ascontiguousarray(X_csc.data, dtype=np.float64)
    active_i32 = active.astype(np.int32)

    def matvec_sparse(v: np.ndarray) -> np.ndarray:
        out = _core.restricted_ls_hessian_matvec(
            indptr, indices, data, n_samples, active_i32, np.ascontiguousarray(v)
        )
        return np.asarray(out * inv_n, dtype=np.float64)

    return matvec_sparse


def _build_logistic_data_matvec(
    design: object, target: Array, beta: Array, active: np.ndarray
) -> _MatVec:
    """Logistic Hessian ``X^T diag(w) X`` restricted to ``active``, densified locally.

    The active set is typically small, so we materialize ``√w · X_A`` (shape
    ``n_samples × |A|``) once and use its Gram matrix for matvecs. This trades
    a one-time densification for many cheap Gram-vector products inside CG.

    Note: ``LogisticLoss`` is the unnormalized sum-of-logs (sklearn's
    convention via ``C = 1/α``); no ``1/n`` factor.
    """
    _ = target  # convention check is the adapter's job; here β suffices
    z = design @ beta  # type: ignore[operator]
    sig = 1.0 / (1.0 + np.exp(-z))
    w = sig * (1.0 - sig)
    sqrt_w = np.sqrt(w)
    if sp.issparse(design):
        XA = design[:, active].toarray()  # type: ignore[index]
    else:
        XA = np.ascontiguousarray(design[:, active])  # type: ignore[index]
    XA_w = sqrt_w[:, None] * XA
    gram = XA_w.T @ XA_w  # |A| × |A| dense; small.

    def matvec(v: np.ndarray) -> np.ndarray:
        return np.asarray(gram @ v, dtype=np.float64)

    return matvec
