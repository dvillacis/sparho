"""Reverse-mode hypergradient (sparse-ho's ``Backward``).

Backward replays the inner solver's iterates in reverse to accumulate the
vector–Jacobian product ``∂C/∂β · dβ/dα`` (sparse-ho's ``get_grad_backward``).
It must record every β sweep, so it owns the inner solve via a native Rust
kernel rather than reading ``solver_result`` — and forms the Gram matrix
``XᵀX``, making it the costly member of the family (provided for completeness /
cross-checking; ImplicitForward is the efficient default).

Only the dense ``SquaredLoss`` Lasso (``L1``) case has a native reverse kernel;
every other ``(datafit, penalty)`` — including sparse designs — delegates to
:func:`implicit_forward`, which is numerically identical here.
"""

from __future__ import annotations

import scipy.sparse as sp

from ..core.types import Array, Hyperparam
from ..problem import L1, Problem, SquaredLoss
from ..state import SolverResult
from ._bcd import backward_lasso
from .implicit_forward import implicit_forward


def backward(
    problem: Problem,
    hyperparam: Hyperparam,
    solver_result: SolverResult,
    criterion_grad_beta: Array,
    *,
    tol: float = 1e-8,
    maxiter: int | None = None,
    ridge: float | None = None,
) -> Hyperparam:
    """Compute ``dC/dα`` by reverse-mode replay of the inner coordinate descent.

    Shares the ``HypergradFn`` signature. For dense ``Problem(SquaredLoss, L1, …)``
    this runs the native record-and-replay kernel; all other cases (sparse
    designs, other penalties/datafits) delegate to :func:`implicit_forward`.
    """
    if (
        isinstance(problem.datafit, SquaredLoss)
        and isinstance(problem.penalty, L1)
        and not sp.issparse(problem.design)
    ):
        return backward_lasso(
            problem,
            hyperparam,
            criterion_grad_beta,
            tol=tol,
            max_iter=maxiter if maxiter is not None else 10_000,
        )
    return implicit_forward(
        problem,
        hyperparam,
        solver_result,
        criterion_grad_beta,
        tol=tol,
        maxiter=maxiter,
        ridge=ridge,
    )
