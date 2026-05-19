"""Solver protocol — the typing surface every adapter satisfies structurally."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .core.types import Array, Hyperparam
from .problem import Problem
from .state import SolverResult


@runtime_checkable
class Solver(Protocol):
    """Anything callable with the signature ``(Problem, Hyperparam, *, x0=None) -> SolverResult``.

    The ``Solver`` is the boundary between sparho's bilevel machinery and the
    underlying inner-problem solver (sklearn, celer, or a user callable). It
    has no awareness of the outer loop — it just returns a converged inner
    solution at the given hyperparameter, alongside the active set and dual
    gap needed by the hypergradient.

    Warm-start: ``x0`` is an optional initial guess for the coefficients (e.g.
    ``β*`` from the previous outer iteration at a nearby hyperparameter).
    Solvers may use it to seed the inner iteration and converge faster; those
    that can't are free to ignore it. The criterion / outer loop owns the
    decision of *when* to warm-start; the solver just honors ``x0`` if given.

    Tolerance override: ``tol`` is an optional inner-solver tolerance that
    supersedes whatever default the adapter holds (e.g. ``SklearnLasso.tol``).
    HOAG-style outer loops (:func:`sparho.search.hoag_search`) use this to
    shrink the inner tolerance across iterations — loose early when only the
    gradient direction matters, tight late when the criterion value resolution
    drives convergence. Adapters that have no meaningful tolerance setting may
    ignore it.
    """

    def __call__(
        self,
        problem: Problem,
        hyperparam: Hyperparam,
        /,
        *,
        x0: Array | None = None,
        tol: float | None = None,
    ) -> SolverResult: ...
