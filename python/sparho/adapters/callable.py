"""Wrap an arbitrary ``(problem, hp) -> SolverResult`` callable as a Solver."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.types import Array, Hyperparam
from ..problem import Problem
from ..state import SolverResult

_SolverFn = Callable[..., SolverResult]


def _accepts_kw(fn: Callable[..., Any], name: str) -> bool:
    """Return ``True`` if ``fn``'s signature accepts a keyword argument ``name``."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == name and param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return True
    return False


@dataclass(frozen=True, slots=True)
class _CallableSolver:
    fn: _SolverFn
    name: str
    _accepts_x0: bool = field(default=False, compare=False, repr=False, hash=False)
    _accepts_tol: bool = field(default=False, compare=False, repr=False, hash=False)

    def __call__(
        self,
        problem: Problem,
        hyperparam: Hyperparam,
        /,
        *,
        x0: Array | None = None,
        tol: float | None = None,
    ) -> SolverResult:
        kwargs: dict[str, Any] = {}
        if self._accepts_x0 and x0 is not None:
            kwargs["x0"] = x0
        if self._accepts_tol and tol is not None:
            kwargs["tol"] = tol
        return self.fn(problem, hyperparam, **kwargs)


def as_solver(fn: _SolverFn, *, name: str = "<callable>") -> _CallableSolver:
    """Wrap ``fn`` as a Solver. The wrapper is frozen so it can be hashed and reprd.

    Optional Solver-protocol kwargs (``x0`` for warm-start, ``tol`` for an
    inner-tolerance override) are forwarded only when ``fn`` declares them in
    its signature. A plain ``(problem, hp) -> SolverResult`` callable keeps
    working — the wrapper silently drops kwargs the function doesn't accept.
    """
    return _CallableSolver(
        fn=fn,
        name=name,
        _accepts_x0=_accepts_kw(fn, "x0"),
        _accepts_tol=_accepts_kw(fn, "tol"),
    )
