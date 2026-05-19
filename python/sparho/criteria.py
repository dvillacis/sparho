"""Outer-loop criteria: held-out MSE, held-out logistic, K-fold cross-validation.

A ``Criterion`` is responsible for:
- slicing the full ``Problem`` to a training subproblem,
- driving the inner solver,
- evaluating a held-out quantity at the converged ``β*``,
- and (when asked) computing ``dC/dα`` by chaining ``∂C/∂β`` through the
  provided hypergradient function (typically :func:`sparho.hypergrad.implicit_forward`).

The Criterion Protocol exposes two methods. ``value(problem, hp, solver)`` is
the cheap value-only path used by line search trials; ``value_and_hypergrad``
is the full path that also runs the implicit-diff linear solve.

All Criterion classes are frozen dataclasses. ``CrossVal`` wraps a base
single-split Criterion (``HeldOutMSE`` by default, or ``HeldOutLogistic`` for
classification) and averages value + hypergradient across folds.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp

from .core.types import Array, Hyperparam, IndexArray
from .problem import Problem
from .solver import Solver

# Hypergradient signature: ``(train_problem, hp, solver_result, grad_β) → Hyperparam``.
HypergradFn = Callable[..., Hyperparam]


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """Outcome of :meth:`Criterion.value_and_hypergrad`.

    ``coef`` and ``active_set`` are reported from the last (or only) inner
    solve; for ``CrossVal`` they come from the final fold and are diagnostic
    only — the user is expected to refit on the full data at ``best_hyperparam``
    if a single final β is needed.
    """

    value: float
    hypergrad: Hyperparam
    coef: Array
    active_set: IndexArray


@runtime_checkable
class Criterion(Protocol):
    """Outer-loop validation oracle.

    Implementations: :class:`HeldOutMSE`, :class:`HeldOutLogistic`, :class:`CrossVal`.

    ``x0`` is an optional warm-start coefficient guess threaded through to the
    inner solver. Single-split criteria forward it directly; :class:`CrossVal`
    ignores caller-supplied ``x0`` because it manages its own per-fold cache.

    ``tol`` is an optional inner-solver tolerance that overrides the adapter's
    default. Threaded through to ``Solver.__call__(tol=...)``. Used by HOAG-style
    outer loops to schedule inner accuracy across iterations.
    """

    def value(
        self,
        problem: Problem,
        hp: Hyperparam,
        solver: Solver,
        *,
        x0: Array | None = None,
        tol: float | None = None,
    ) -> float: ...
    def value_and_hypergrad(
        self,
        problem: Problem,
        hp: Hyperparam,
        solver: Solver,
        hypergrad_fn: HypergradFn,
        *,
        x0: Array | None = None,
        tol: float | None = None,
    ) -> CriterionResult: ...


# ---------------------------------------------------------------- helpers


def _slice_problem(problem: Problem, idx: IndexArray) -> Problem:
    """Return ``problem`` with ``design`` and ``target`` restricted to ``idx``.

    Row-slicing a CSC matrix densifies the slice in scipy ≤ 1.x; we let scipy
    handle the format choice and re-CSC inside the solver / hypergradient if
    necessary.
    """
    X = problem.design
    y = problem.target
    return dataclasses.replace(problem, design=X[idx], target=y[idx])


def _matvec(X: Any, v: Array) -> Array:
    """``X @ v`` returning a plain ndarray (sparse or dense ``X``)."""
    if sp.issparse(X):
        return np.asarray(X @ v).ravel()
    return np.asarray(X @ v)


def _rmatvec(X: Any, v: Array) -> Array:
    """``X^T @ v`` returning a plain ndarray."""
    if sp.issparse(X):
        return np.asarray(X.T @ v).ravel()
    return np.asarray(X.T @ v)


def _hg_zero_like(hg: Hyperparam) -> Hyperparam:
    if isinstance(hg, np.ndarray):
        return np.zeros_like(hg)
    return 0.0


def _hg_add(a: Hyperparam, b: Hyperparam) -> Hyperparam:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.asarray(np.asarray(a) + np.asarray(b), dtype=np.float64)
    return float(a) + float(b)


def _hg_scale(a: Hyperparam, c: float) -> Hyperparam:
    if isinstance(a, np.ndarray):
        return c * a
    return c * float(a)


# ---------------------------------------------------------------- HeldOutMSE


@dataclass(frozen=True, slots=True)
class HeldOutMSE:
    """Held-out mean-squared-error.

    ``C(β) = (1/|val|) Σ_{i ∈ val} (yᵢ − Xᵢ β)²`` — matches sklearn's
    ``mean_squared_error`` (no ``1/2``). The gradient ``∂C/∂β`` carries the
    factor of ``2``.
    """

    idx_train: IndexArray
    idx_val: IndexArray

    def value(
        self,
        problem: Problem,
        hp: Hyperparam,
        solver: Solver,
        *,
        x0: Array | None = None,
        tol: float | None = None,
    ) -> float:
        train_problem = _slice_problem(problem, self.idx_train)
        result = solver(train_problem, hp, x0=x0, tol=tol)
        return self._mse(problem, result.coef)

    def value_and_hypergrad(
        self,
        problem: Problem,
        hp: Hyperparam,
        solver: Solver,
        hypergrad_fn: HypergradFn,
        *,
        x0: Array | None = None,
        tol: float | None = None,
    ) -> CriterionResult:
        train_problem = _slice_problem(problem, self.idx_train)
        result = solver(train_problem, hp, x0=x0, tol=tol)
        value = self._mse(problem, result.coef)
        grad_beta = self._mse_grad(problem, result.coef)
        hg = hypergrad_fn(train_problem, hp, result, grad_beta)
        return CriterionResult(
            value=value,
            hypergrad=hg,
            coef=result.coef,
            active_set=result.active_set,
        )

    def _mse(self, problem: Problem, beta: Array) -> float:
        X_val = problem.design[self.idx_val]
        y_val = problem.target[self.idx_val]
        resid = _matvec(X_val, beta) - y_val
        return float(resid @ resid) / len(self.idx_val)

    def _mse_grad(self, problem: Problem, beta: Array) -> Array:
        X_val = problem.design[self.idx_val]
        y_val = problem.target[self.idx_val]
        resid = _matvec(X_val, beta) - y_val
        return _rmatvec(X_val, 2.0 * resid) / len(self.idx_val)


# ---------------------------------------------------------------- HeldOutLogistic


@dataclass(frozen=True, slots=True)
class HeldOutLogistic:
    """Held-out logistic loss: ``C(β) = (1/|val|) Σᵢ log(1 + exp(−yᵢ Xᵢβ))``.

    Labels assumed in ``{−1, +1}`` (sparho's ``LogisticLoss`` convention).
    """

    idx_train: IndexArray
    idx_val: IndexArray

    def value(
        self,
        problem: Problem,
        hp: Hyperparam,
        solver: Solver,
        *,
        x0: Array | None = None,
        tol: float | None = None,
    ) -> float:
        train_problem = _slice_problem(problem, self.idx_train)
        result = solver(train_problem, hp, x0=x0, tol=tol)
        return self._loss(problem, result.coef)

    def value_and_hypergrad(
        self,
        problem: Problem,
        hp: Hyperparam,
        solver: Solver,
        hypergrad_fn: HypergradFn,
        *,
        x0: Array | None = None,
        tol: float | None = None,
    ) -> CriterionResult:
        train_problem = _slice_problem(problem, self.idx_train)
        result = solver(train_problem, hp, x0=x0, tol=tol)
        value = self._loss(problem, result.coef)
        grad_beta = self._loss_grad(problem, result.coef)
        hg = hypergrad_fn(train_problem, hp, result, grad_beta)
        return CriterionResult(
            value=value,
            hypergrad=hg,
            coef=result.coef,
            active_set=result.active_set,
        )

    def _loss(self, problem: Problem, beta: Array) -> float:
        X_val = problem.design[self.idx_val]
        y_val = problem.target[self.idx_val]
        Xb = _matvec(X_val, beta)
        # log(1 + exp(−y · Xβ)); numerically stable.
        return float(np.mean(np.logaddexp(0.0, -y_val * Xb)))

    def _loss_grad(self, problem: Problem, beta: Array) -> Array:
        X_val = problem.design[self.idx_val]
        y_val = problem.target[self.idx_val]
        Xb = _matvec(X_val, beta)
        # σ(−y · Xβ) = 1 / (1 + exp(y · Xβ))
        sigma = 1.0 / (1.0 + np.exp(y_val * Xb))
        return -_rmatvec(X_val, y_val * sigma) / len(self.idx_val)


# ---------------------------------------------------------------- CrossVal


_FoldBuilder = Callable[[IndexArray, IndexArray], Criterion]


@dataclass(frozen=True, slots=True)
class CrossVal:
    """K-fold cross-validation aggregator.

    Wraps a single-split base criterion class (typically :class:`HeldOutMSE`)
    over a tuple of ``(train_idx, val_idx)`` pairs. Both value and
    hypergradient are means across folds.

    Build via :meth:`kfold`::

        cv = CrossVal.kfold(problem.n_samples, k=5)

    For classification, pass ``base=HeldOutLogistic`` to ``kfold``.

    Warm-start: with ``warm_start=True``, each fold's previous-iteration ``β*``
    seeds the next inner solve at the same fold. Big wins when the inner
    solver dominates (sparse-X, small α, large active set); converges to the
    same answer as ``warm_start=False`` because Lasso is convex. The cache is
    mutable but excluded from equality / hash so the dataclass remains a
    well-behaved value object.
    """

    folds: tuple[tuple[IndexArray, IndexArray], ...]
    base: _FoldBuilder = HeldOutMSE
    warm_start: bool = False
    _cache: list[Array | None] = field(
        default_factory=list, compare=False, repr=False, hash=False
    )

    @classmethod
    def kfold(
        cls,
        n_samples: int,
        k: int = 5,
        *,
        shuffle: bool = True,
        random_state: int | None = 0,
        base: _FoldBuilder = HeldOutMSE,
        warm_start: bool = False,
    ) -> CrossVal:
        """Build a ``CrossVal`` from ``sklearn.model_selection.KFold``."""
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state if shuffle else None)
        folds = tuple(
            (np.asarray(tr, dtype=np.int32), np.asarray(val, dtype=np.int32))
            for tr, val in kf.split(np.arange(n_samples))
        )
        return cls(folds=folds, base=base, warm_start=warm_start)

    def _ensure_cache(self) -> list[Array | None]:
        if len(self._cache) != len(self.folds):
            self._cache.clear()
            self._cache.extend([None] * len(self.folds))
        return self._cache

    def value(
        self,
        problem: Problem,
        hp: Hyperparam,
        solver: Solver,
        *,
        x0: Array | None = None,  # noqa: ARG002 — CrossVal owns per-fold warm-start
        tol: float | None = None,
    ) -> float:
        cache = self._ensure_cache() if self.warm_start else None
        total = 0.0
        for i, (idx_tr, idx_val) in enumerate(self.folds):
            fold_x0 = cache[i] if cache is not None else None
            total += self.base(idx_tr, idx_val).value(
                problem, hp, solver, x0=fold_x0, tol=tol
            )
        return total / len(self.folds)

    def value_and_hypergrad(
        self,
        problem: Problem,
        hp: Hyperparam,
        solver: Solver,
        hypergrad_fn: HypergradFn,
        *,
        x0: Array | None = None,  # noqa: ARG002 — CrossVal owns per-fold warm-start
        tol: float | None = None,
    ) -> CriterionResult:
        n = len(self.folds)
        cache = self._ensure_cache() if self.warm_start else None
        total_value = 0.0
        total_hg: Hyperparam | None = None
        last_coef: Array | None = None
        last_active: IndexArray | None = None
        for i, (idx_tr, idx_val) in enumerate(self.folds):
            crit = self.base(idx_tr, idx_val)
            fold_x0 = cache[i] if cache is not None else None
            res = crit.value_and_hypergrad(
                problem, hp, solver, hypergrad_fn, x0=fold_x0, tol=tol
            )
            if cache is not None:
                cache[i] = np.asarray(res.coef, dtype=np.float64).copy()
            total_value += res.value
            total_hg = res.hypergrad if total_hg is None else _hg_add(total_hg, res.hypergrad)
            last_coef = res.coef
            last_active = res.active_set
        assert total_hg is not None and last_coef is not None and last_active is not None
        return CriterionResult(
            value=total_value / n,
            hypergrad=_hg_scale(total_hg, 1.0 / n),
            coef=last_coef,
            active_set=last_active,
        )
