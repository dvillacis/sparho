"""KKT-restricted conjugate-gradient hypergradient (sparse-ho's ``Implicit``).

Restricts the KKT linear system to the active set returned by the inner solver
and solves it via matrix-free conjugate gradients (the Hessian is symmetric
positive definite on the active set, at the converged ``β*``).

Math (KKT-derived, no prox-Jacobian needed):

  Inner stationarity on the active set ``A``:
      ∇_A L(β*) + ∂R/∂β_A(β*; α) = 0.
  Differentiating w.r.t. α:
      (H_L,AA + ∂²R/∂β²|_A) · dβ*_A/dα + ∂²R/∂α∂β|_A = 0
  ⇒   M_AA · dβ*_A/dα = − r(β*_A; α)

with ``M_AA`` and ``r`` depending on the (datafit, penalty) pair. The
outer-loop hypergradient ``dC/dα`` follows by chain rule with the criterion
gradient ``∂C/∂β`` passed in by the caller.

This module is the universal fallback: :func:`sparho.hypergrad.implicit_forward`
dispatches every ``(datafit, penalty)`` pair that lacks a native BCD kernel
(currently ``LogisticLoss`` and ``GroupL1``) here.
"""

from __future__ import annotations

import warnings
from typing import assert_never

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from ..core.types import Array, Hyperparam
from ..problem import (
    L1,
    ElasticNet,
    GroupL1,
    Problem,
    WeightedL1,
)
from ..state import SolverResult
from ._shared import (
    _build_hess_matvec,
    _GroupL1ActiveInfo,
    _resolve_group_l1_active,
    _resolve_ridge,
    _ridge_wrap,
)


def implicit(
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
        Scalar for ``L1`` / ``ElasticNet`` / ``GroupL1``; ``(n_features,)``
        array for ``WeightedL1`` (entries outside the active set are exactly
        zero).

    See Also
    --------
    sparho.hypergrad.implicit_forward
        The default dispatcher; routes unsupported ``(datafit, penalty)`` pairs
        (``LogisticLoss``, ``GroupL1``) into this function.

    Notes
    -----
    Full derivation of the linear system ``M_AA · dβ*/dα = -r``, the
    active-set restriction argument, the per-penalty curvature terms,
    and the ridge-stabilization bias bound live under :doc:`/theory/index`:
    :doc:`/theory/implicit_diff` (KKT view + ridge),
    :doc:`/theory/active_set` (when ``A`` is locally constant), and
    :doc:`/theory/penalties` (per-variant ``∂_β s`` and ``∂_α s``).
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
            f"implicit: CG failed (info={info}, finite={v_finite}); "
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
