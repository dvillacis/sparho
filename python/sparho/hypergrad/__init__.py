"""Hypergradient algorithm family (ports of sparse-ho's ``algo`` module).

Each algorithm is a free function conforming to the ``HypergradFn`` seam that
criteria call positionally — ``(train_problem, hp, solver_result, grad_β) →
Hyperparam`` — plus keyword-only knobs. Select an algorithm by passing the
callable to :func:`sparho.grad_search` / :func:`sparho.hoag_search`, or look it
up by name with :func:`get_hypergrad`.

- :func:`implicit_forward` — the default; support-restricted Jacobian
  fixed-point with a CG fallback for unsupported ``(datafit, penalty)`` pairs.
- :func:`implicit` — matrix-free conjugate-gradient on the restricted KKT
  Hessian (sparse-ho's ``Implicit``); the universal fallback.
"""

from __future__ import annotations

from collections.abc import Callable

from ..core.types import Hyperparam
from .backward import backward
from .forward import forward
from .implicit import implicit
from .implicit_forward import implicit_forward
from .warm_start import WarmStartHypergrad

# Hypergradient signature: ``(train_problem, hp, solver_result, grad_β) → Hyperparam``.
HypergradFn = Callable[..., Hyperparam]

_ALGOS: dict[str, HypergradFn] = {
    "implicit_forward": implicit_forward,
    "forward": forward,
    "backward": backward,
    "implicit": implicit,
}


def get_hypergrad(name: str = "implicit_forward") -> HypergradFn:
    """Look up a hypergradient algorithm by name.

    Parameters
    ----------
    name
        One of ``"implicit_forward"`` (default), ``"forward"``, ``"backward"``,
        or ``"implicit"``.

    Returns
    -------
    The selected hypergradient callable (a ``HypergradFn``).
    """
    try:
        return _ALGOS[name]
    except KeyError:
        raise ValueError(
            f"unknown hypergradient algorithm {name!r}; choose from {sorted(_ALGOS)}"
        ) from None


__all__ = [
    "HypergradFn",
    "WarmStartHypergrad",
    "backward",
    "forward",
    "implicit",
    "implicit_forward",
    "get_hypergrad",
]
