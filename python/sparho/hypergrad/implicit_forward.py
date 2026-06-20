"""ImplicitForward hypergradient — the default algorithm.

ImplicitForward (sparse-ho's headline method) solves the inner problem, then
computes the Jacobian ``dβ*/dα`` by a coordinate-descent fixed-point iteration
*restricted to the support*, and contracts it with the criterion gradient.

This module exposes :func:`implicit_forward`, the **default** hypergradient that
:func:`sparho.grad_search` / :func:`sparho.hoag_search` use. It dispatches on
``(datafit, penalty)``: the native BCD path (a Rust kernel) handles the
``SquaredLoss`` Lasso family, and every other pair falls back to the
matrix-free CG implementation in :func:`sparho.hypergrad.implicit`.
"""

from __future__ import annotations

from typing import assert_never

from ..core.types import Array, Hyperparam
from ..problem import (
    L1,
    ElasticNet,
    GroupL1,
    LogisticLoss,
    Problem,
    SquaredLoss,
    WeightedL1,
)
from ..state import SolverResult
from ._bcd import implicit_forward_enet, implicit_forward_lasso, implicit_forward_wl1
from .implicit import implicit


def dispatch_bcd(
    problem: Problem,
    hyperparam: Hyperparam,
    solver_result: SolverResult,
    criterion_grad_beta: Array,
    *,
    tol: float = 1e-8,
    maxiter: int | None = None,
    ridge: float | None = None,
    jac0: Array | None = None,
) -> tuple[Hyperparam, Array | None]:
    """Dispatch ImplicitForward by ``(datafit, penalty)``, returning ``(dC/dα, jac)``.

    ``jac`` is the support-restricted solve vector (the warm-start payload),
    or ``None`` on the CG-fallback paths / empty support / non-finite solve.
    ``jac0`` warm-starts the BCD fixed point. Used by :func:`implicit_forward`
    (which discards ``jac``) and by :class:`sparho.WarmStartHypergrad` (which
    caches it).
    """

    def _cg_fallback() -> tuple[Hyperparam, Array | None]:
        # Not covered by a native BCD kernel — use the matrix-free CG path, which
        # also owns ``ridge`` (ignored on the BCD path: no Hessian to regularize).
        return (
            implicit(
                problem,
                hyperparam,
                solver_result,
                criterion_grad_beta,
                tol=tol,
                maxiter=maxiter,
                ridge=ridge,
            ),
            None,
        )

    max_iter_jac = maxiter if maxiter is not None else 10_000
    datafit = problem.datafit
    penalty = problem.penalty
    match datafit:
        case SquaredLoss():
            match penalty:
                case L1():
                    return implicit_forward_lasso(
                        problem,
                        hyperparam,
                        solver_result,
                        criterion_grad_beta,
                        tol_jac=tol,
                        max_iter_jac=max_iter_jac,
                        jac0=jac0,
                    )
                case ElasticNet(rho=rho):
                    return implicit_forward_enet(
                        problem,
                        hyperparam,
                        solver_result,
                        criterion_grad_beta,
                        rho=rho,
                        tol_jac=tol,
                        max_iter_jac=max_iter_jac,
                        jac0=jac0,
                    )
                case WeightedL1():
                    return implicit_forward_wl1(
                        problem,
                        hyperparam,
                        solver_result,
                        criterion_grad_beta,
                        tol_jac=tol,
                        max_iter_jac=max_iter_jac,
                        jac0=jac0,
                    )
                case GroupL1():
                    return _cg_fallback()
                case _:
                    assert_never(penalty)
        case LogisticLoss():
            return _cg_fallback()
        case _:
            assert_never(datafit)


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
    """Compute ``dC/dα`` via ImplicitForward, dispatching by ``(datafit, penalty)``.

    For ``SquaredLoss`` with ``L1`` / ``ElasticNet`` / ``WeightedL1`` this runs
    the native BCD Jacobian fixed point. Every other pair — ``LogisticLoss`` and
    ``GroupL1`` — falls back to the matrix-free CG solver
    :func:`sparho.hypergrad.implicit`, which also owns the ``ridge``
    stabilization (``ridge`` is ignored on the BCD path: a coordinate-descent
    fixed point has no Hessian to regularize).

    Parameters mirror :func:`sparho.hypergrad.implicit.implicit` so this slots
    into the ``HypergradFn`` seam criteria call positionally. ``tol`` is the
    Jacobian-iteration tolerance on the BCD path and the CG tolerance on the
    fallback path; ``maxiter`` caps the respective iteration counts.
    """
    hg, _jac = dispatch_bcd(
        problem,
        hyperparam,
        solver_result,
        criterion_grad_beta,
        tol=tol,
        maxiter=maxiter,
        ridge=ridge,
    )
    return hg
