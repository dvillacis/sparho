"""Top-level outer loops: ``grad_search`` (plain GD) and ``hoag_search`` (HOAG).

Both are pure imperative ``for``-loops that thread Solver + Criterion +
Hypergrad into a single call and operate in **log space**: ``hp0`` is
interpreted as a positive ``α``, and the search steps in ``θ = log α``. This
keeps ``α > 0`` without projection and matches how ``α`` typically varies —
across orders of magnitude. The chain rule ``dC/dθ = dC/dα · α`` is applied
inside the loop.

After the search, the inner solver is run once more on the **full** problem
at the best ``α`` seen (``best_value`` minimum) and the resulting ``coef``
goes into ``SearchResult.best_coef``. For ``CrossVal`` this matters — the
per-fold ``coef`` reported by the criterion is the last-fold fit, not the
full-data fit the user wants.

:func:`grad_search` is classical bilevel approximate-gradient descent: plain
``θ ← θ - lr · dC/dθ`` with a fixed learning rate. A baseline for
comparison and for problems where step-size choice is well-understood.

:func:`hoag_search` is the HOAG (Pedregosa 2016) port and the recommended
default for non-smooth bilevel HP optimization. It owns its outer loop: step
size adapted from a Lipschitz proxy ``L``, acceptance test has a ``+C·tol``
slack term that tolerates noise from inner-solver inaccuracy, and inner
tolerance is part of the loop state (optionally decreased exponentially
across iterations).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

from .core.types import Hyperparam
from .criteria import Criterion
from .hypergrad import implicit_forward
from .problem import Problem
from .solver import Solver
from .state import IterationRecord, SearchResult

HypergradFn = Callable[..., Hyperparam]


def grad_search(
    problem: Problem,
    hp0: Hyperparam,
    *,
    solver: Solver,
    criterion: Criterion,
    hypergrad: HypergradFn = implicit_forward,
    n_iter: int = 50,
    lr: float = 0.1,
    tol: float = 1e-6,
) -> SearchResult:
    """Classical bilevel approximate-gradient descent in ``θ = log α`` space.

    ``θ_{k+1} = θ_k - lr · dC/dθ``, where ``dC/dθ = dC/dα · α`` (chain rule).
    Fixed learning rate, one ``value_and_hypergrad`` call per outer iter. No
    step-size adaptation — pair with :func:`hoag_search` if the inner-solver
    accuracy varies with the iteration budget or if you want
    Lipschitz-adaptive steps. Useful as a baseline and for problems where the
    user has prior knowledge of a good ``lr``.

    Parameters
    ----------
    problem
        The full bilevel problem. The criterion handles train/val splits.
    hp0
        Initial hyperparameter — scalar or per-feature vector. **Must be > 0**;
        the outer loop optimizes in ``θ = log α`` space.
    solver
        Inner-problem solver. Called by the criterion and once at the end
        for the full-data refit.
    criterion
        Outer-loop validation oracle (e.g. :class:`sparho.criteria.HeldOutMSE`,
        :class:`sparho.criteria.CrossVal`).
    hypergrad
        Hypergradient function. Defaults to
        :func:`sparho.hypergrad.implicit_forward`.
    n_iter
        Maximum outer iterations.
    lr
        Learning rate applied in ``θ``-space.
    tol
        Stationarity tolerance on the ``log``-space hypergradient norm.

    Returns
    -------
    SearchResult
    """
    hp0_pos = _as_positive(hp0)
    theta = _log(hp0_pos)

    history: list[IterationRecord] = []
    best_value = float("inf")
    best_hp: Hyperparam = hp0_pos
    converged = False

    for k in range(n_iter):
        hp = _exp(theta)
        result = criterion.value_and_hypergrad(problem, hp, solver, hypergrad)
        # Chain rule: ``dC/dθ = dC/dα · α`` (elementwise for vector α).
        hg_theta = _elementwise_mul(result.hypergrad, hp)
        grad_norm = _norm(hg_theta)

        if np.isfinite(result.value) and result.value < best_value:
            best_value = result.value
            best_hp = hp

        history.append(
            IterationRecord(
                iteration=k,
                hyperparam=hp,
                value=result.value,
                grad_norm=grad_norm,
                n_inner_iter=0,
            )
        )

        if not np.all(np.isfinite(np.asarray(hg_theta))):
            warnings.warn(
                f"grad_search: non-finite hypergradient at iter {k}; "
                "holding θ at this iter (outer search will retry next iter)",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if grad_norm < tol:
            converged = True
            break

        theta = _sub(theta, _scale(hg_theta, lr))

    final_result = solver(problem, best_hp)

    return SearchResult(
        best_hyperparam=best_hp,
        best_coef=final_result.coef,
        history=tuple(history),
        converged=converged,
        n_iter=len(history),
    )


# ---------------------------------------------------------------- HOAG


def hoag_search(
    problem: Problem,
    hp0: Hyperparam,
    *,
    solver: Solver,
    criterion: Criterion,
    hypergrad: HypergradFn = implicit_forward,
    n_iter: int = 100,
    inner_tol: float = 1e-5,
    inner_tol_initial: float = 1e-2,
    tolerance_decrease: str = "constant",
    outer_tol: float = 1e-6,
    C: float = 0.25,
    factor: float = 1.0,
    max_step: float = 0.5,
) -> SearchResult:
    """HOAG (Hyperparameter Optimization with Approximate Gradients, Pedregosa 2016).

    One val+grad call per outer iter; step size adapted from a Lipschitz
    proxy ``L``; the acceptance test ``value ≤ value_prev + C·tol +
    tol_prev·(C+factor)·||step·g|| - factor·L·||step·g||²`` carries a slack
    term that absorbs criterion-value noise due to inner-solver inaccuracy.
    On bad descent (``value ≥ 1.2·value_prev``) the step is rejected, ``L``
    is doubled, and val+grad is recomputed at the restored point with
    ``tol/2`` — tightening inner accuracy where the outer search needs it.

    The inner tolerance is part of the loop state and is threaded into the
    criterion → solver call each iter via ``Solver.__call__(tol=...)``. With
    ``tolerance_decrease='exponential'`` the per-iter ``tol`` geometrically
    decreases from ``inner_tol_initial`` to ``inner_tol`` across the
    ``n_iter`` outer steps — loose early when only the gradient *direction*
    matters, tight late when the value *resolution* drives convergence.

    Parameters
    ----------
    problem
        The full bilevel problem. The criterion handles train/val splits.
    hp0
        Initial hyperparameter — scalar or per-feature vector. **Must be > 0**;
        the outer loop optimizes in ``θ = log α`` space.
    solver
        Inner-problem solver. Must accept a ``tol`` keyword (built-in adapters
        and ``as_solver``-wrapped callables both do).
    criterion
        Outer-loop validation oracle.
    hypergrad
        Hypergradient function. Defaults to :func:`implicit_forward`.
    n_iter
        Outer-iteration budget.
    inner_tol
        Final inner-solver tolerance (and the constant value when
        ``tolerance_decrease='constant'``).
    inner_tol_initial
        Starting inner-solver tolerance when
        ``tolerance_decrease='exponential'``. Must be ``≥ inner_tol``.
    tolerance_decrease
        ``'constant'`` (default) or ``'exponential'``.
    outer_tol
        Stationarity tolerance on the ``log``-space hypergradient norm.
    C, factor
        Acceptance-test constants (Pedregosa 2016 calls them ``C`` and the
        scaling on the quadratic term). Defaults match sparse-ho.
    max_step
        Maximum ``θ``-step magnitude per single step (the algorithm takes
        up to two steps per iter; the cap applies to each). Acts as a trust
        region: prevents the first step from overshooting into a
        zero-gradient region (e.g. when the active set collapses) before
        the ``L`` adaptation has had a chance to discipline the step size.

    Returns
    -------
    SearchResult
    """
    if tolerance_decrease not in ("constant", "exponential"):
        raise ValueError(
            f"tolerance_decrease must be 'constant' or 'exponential', got {tolerance_decrease!r}"
        )
    if tolerance_decrease == "exponential" and inner_tol_initial < inner_tol:
        raise ValueError("inner_tol_initial must be ≥ inner_tol for exponential decrease")
    if n_iter < 1:
        raise ValueError("n_iter must be ≥ 1")

    hp0_pos = _as_positive(hp0)
    theta = _log(hp0_pos)
    is_array = isinstance(theta, np.ndarray)

    if tolerance_decrease == "exponential":
        tol_schedule = np.geomspace(inner_tol_initial, inner_tol, n_iter)
    else:
        tol_schedule = np.full(n_iter, float(inner_tol))

    history: list[IterationRecord] = []
    best_value = float("inf")
    best_hp: Hyperparam = hp0_pos
    L: float | None = None
    value_prev = float("inf")
    converged = False

    for k in range(n_iter):
        tol_k = float(tol_schedule[k])
        old_tol = float(tol_schedule[k - 1]) if k > 0 else tol_k
        hp = _exp(theta)

        # 1. val + grad at current θ with current inner tol.
        result = criterion.value_and_hypergrad(
            problem, hp, solver, hypergrad, tol=tol_k
        )
        value = result.value
        grad = _elementwise_mul(result.hypergrad, hp)  # dC/dθ = dC/dα · α
        grad_norm = _norm(grad)

        history.append(
            IterationRecord(
                iteration=k,
                hyperparam=hp,
                value=value,
                grad_norm=grad_norm,
                n_inner_iter=0,
            )
        )
        if np.isfinite(value) and value < best_value:
            best_value = value
            best_hp = hp

        if not np.all(np.isfinite(np.asarray(grad))):
            warnings.warn(
                f"hoag_search: non-finite gradient at iter {k}; "
                "skipping step (outer search will retry next iter)",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        # 2. Init L on first iter from gradient magnitude (sparse-ho heuristic).
        if L is None:
            if grad_norm > 1e-3:
                if is_array:
                    L = grad_norm / float(np.sqrt(np.asarray(theta).size))
                else:
                    L = grad_norm
            else:
                L = 1.0

        if grad_norm < outer_tol:
            converged = True
            break

        # Step-size cap: bound the θ-step magnitude so the first iter can't
        # overshoot into a zero-gradient region before ``L`` adapts. ``L`` is
        # raised, not the step clipped post-hoc, so the acceptance test sees
        # the actual effective L.
        if grad_norm * max_step > 0 and 1.0 / L * grad_norm > max_step:
            L = grad_norm / max_step
        step_size = 1.0 / L
        incr = step_size * grad_norm

        # 3. Tentative step.
        theta_pre = _copy(theta)
        theta = _sub(theta, _scale(grad, step_size))

        # 4. Retrospective acceptance test on the PREVIOUS step
        #    (the step that brought us to this θ from θ_pre_last_iter).
        slack_good = (
            value_prev
            + C * tol_k
            + old_tol * (C + factor) * incr
            - factor * L * incr * incr
        )
        slack_bad = 1.2 * value_prev

        if value <= slack_good:
            # Previous step looked good → take a second step this iter and
            # shrink L (grow step for next iter).
            L *= 0.95
            theta = _sub(theta, _scale(grad, step_size))
        elif value >= slack_bad:
            # Previous step was bad → reject this iter's tentative step,
            # double L, and recompute val+grad at θ_pre with halved tol.
            L *= 2.0
            theta = theta_pre
            tol_retry = tol_k * 0.5
            result = criterion.value_and_hypergrad(
                problem, _exp(theta), solver, hypergrad, tol=tol_retry
            )
            value = result.value
            grad = _elementwise_mul(result.hypergrad, _exp(theta))
            grad_norm = _norm(grad)
            history[-1] = IterationRecord(
                iteration=k,
                hyperparam=_exp(theta),
                value=value,
                grad_norm=grad_norm,
                n_inner_iter=0,
            )
            if np.isfinite(value) and value < best_value:
                best_value = value
                best_hp = _exp(theta)
        else:
            # Neutral → take a second step this iter, leave L alone.
            theta = _sub(theta, _scale(grad, step_size))

        value_prev = value

    final_result = solver(problem, best_hp, tol=float(tol_schedule[-1]))

    return SearchResult(
        best_hyperparam=best_hp,
        best_coef=final_result.coef,
        history=tuple(history),
        converged=converged,
        n_iter=len(history),
    )


# ---------------------------------------------------------------- helpers


def _as_positive(hp: Hyperparam) -> Hyperparam:
    """Validate ``hp > 0`` and return a contiguous float64 copy."""
    if isinstance(hp, np.ndarray):
        arr = np.asarray(hp, dtype=np.float64)
        if np.any(arr <= 0):
            raise ValueError("hp0 must be strictly positive componentwise")
        return arr
    val = float(hp)
    if val <= 0:
        raise ValueError("hp0 must be strictly positive")
    return val


def _log(hp: Hyperparam) -> Hyperparam:
    if isinstance(hp, np.ndarray):
        return np.log(hp)
    return float(np.log(hp))


def _exp(theta: Hyperparam) -> Hyperparam:
    if isinstance(theta, np.ndarray):
        return np.exp(theta)
    return float(np.exp(theta))


def _elementwise_mul(a: Hyperparam, b: Hyperparam) -> Hyperparam:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)
    return float(a) * float(b)


def _norm(x: Hyperparam) -> float:
    if isinstance(x, np.ndarray):
        return float(np.linalg.norm(x))
    return abs(float(x))


def _sub(a: Hyperparam, b: Hyperparam) -> Hyperparam:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(a) - float(b)


def _scale(a: Hyperparam, c: float) -> Hyperparam:
    if isinstance(a, np.ndarray):
        return c * np.asarray(a, dtype=np.float64)
    return c * float(a)


def _copy(a: Hyperparam) -> Hyperparam:
    if isinstance(a, np.ndarray):
        return np.asarray(a, dtype=np.float64).copy()
    return float(a)
