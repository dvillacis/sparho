"""Forward-mode hypergradient (sparse-ho's ``Forward``).

Forward differentiates *through* the inner coordinate descent: it re-solves the
inner problem while propagating the Jacobian ``dβ/dα`` jointly, sweep by sweep
(a native Rust joint solver). At the converged ``β*`` the inactive coordinates
carry ``dβ = 0``, so Forward's full-dimensional Jacobian agrees with
ImplicitForward's support-restricted one — the two produce the same
hypergradient, differing only in *how* it is computed (joint solve vs.
support-restricted post-solve).

Because Forward owns the inner solve, it ignores ``solver_result`` and re-solves
``β`` from cold; it is the less-efficient member of the family (full-dimensional
sweeps) and is provided for completeness / cross-checking. Only the
``SquaredLoss`` Lasso (``L1``) case has a native joint kernel; every other
``(datafit, penalty)`` pair delegates to :func:`implicit_forward` (which is
numerically identical here).
"""

from __future__ import annotations

from ..core.types import Array, Hyperparam
from ..problem import L1, Problem, SquaredLoss
from ..state import SolverResult
from ._bcd import forward_lasso
from .implicit_forward import implicit_forward


def forward(
    problem: Problem,
    hyperparam: Hyperparam,
    solver_result: SolverResult,
    criterion_grad_beta: Array,
    *,
    tol: float = 1e-8,
    maxiter: int | None = None,
    ridge: float | None = None,
) -> Hyperparam:
    """Compute ``dC/dα`` by forward-mode differentiation through coordinate descent.

    Shares the ``HypergradFn`` signature. For ``Problem(SquaredLoss, L1, …)`` this
    runs the native joint β+Jacobian solver; all other pairs delegate to
    :func:`implicit_forward`.
    """
    match (problem.datafit, problem.penalty):
        case (SquaredLoss(), L1()):
            return forward_lasso(
                problem,
                hyperparam,
                criterion_grad_beta,
                tol=tol,
                max_iter=maxiter if maxiter is not None else 10_000,
            )
        case _:
            return implicit_forward(
                problem,
                hyperparam,
                solver_result,
                criterion_grad_beta,
                tol=tol,
                maxiter=maxiter,
                ridge=ridge,
            )
