"""Result and state dataclasses for the inner solver and outer search.

All dataclasses are ``frozen=True, slots=True``. Outer-search state is built
incrementally as an immutable ``tuple`` of ``IterationRecord``; there is no
mutable monitor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .core.types import Array, Hyperparam, IndexArray

# ---------------------------------------------------------------- Inner solver


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Outcome of one inner solve at a fixed hyperparameter.

    Parameters
    ----------
    coef
        Estimated coefficient vector ``β̂``.
    active_set
        Integer indices ``j`` where ``β̂ⱼ ≠ 0``. Sorted ascending. ``int32``
        to match scipy.sparse CSC index types — the hypergradient linear
        solve restricts to this set.
    dual_gap
        Final duality gap (or a non-negative proxy for it). Used to assert
        inner-loop convergence at the tolerance the adapter targeted.
    n_iter
        Number of inner iterations the adapter consumed.
    extras
        Adapter-specific escape hatch. SVM/SVR adapters in a future phase
        will stash the converged dual variable here.
    """

    coef: Array
    active_set: IndexArray
    dual_gap: float
    n_iter: int
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- Outer search


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """One outer-loop snapshot. Tuples of these form a ``SearchResult.history``."""

    iteration: int
    hyperparam: Hyperparam
    value: float
    grad_norm: float
    n_inner_iter: int


@dataclass(frozen=True, slots=True)
class SearchState:
    """In-flight outer-loop state, threaded through ``grad_search``'s for-loop.

    Each algorithmic step is a pure function ``(state, ...) -> state``; ``grad_search``
    is the only place the loop is written imperatively.
    """

    iteration: int
    hyperparam: Hyperparam
    value: float
    grad: Array
    solver_result: SolverResult
    optimizer_state: Any  # opaque, per-optimizer; pickleable but not introspected
    history: tuple[IterationRecord, ...]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Final outcome of ``grad_search``."""

    best_hyperparam: Hyperparam
    best_coef: Array
    history: tuple[IterationRecord, ...]
    converged: bool
    n_iter: int
