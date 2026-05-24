"""Public testing helpers for verifying solver optimality.

This module is part of the *public testing surface* — third-party adapters
that satisfy the :class:`sparho.Solver` Protocol can use these helpers in
their own test suites to gate convergence quality.

The central function is :func:`kkt_residual`, which evaluates the
proximal fixed-point residual at a candidate solution::

    r(β) = β − prox_{R(·; α)}(β − ∇L(β))

For the convex inner problems sparho targets — Lasso, ElasticNet, weighted
L1, Group-L1, and sparse logistic regression — ``r(β*) = 0`` iff ``β*`` is
optimal. The infinity-norm of ``r`` is a relative-tolerance measure of how
far ``β`` is from the KKT manifold and scales monotonically with the
primal–dual gap (up to the local curvature constant). For typical use it
is the *cheapest* post-solve correctness assertion available.

The residual is computed by composing the existing Rust prox kernels
(:mod:`sparho._core.prox_l1`, ``prox_elastic_net``, ``prox_weighted_l1``,
``prox_group_l1``) with the existing CSC matvec primitives — no new
numerical kernels are introduced here. The dispatch is exhaustive: an
``assert_never`` tail on the union :data:`sparho.Penalty` ensures mypy
flags any new variant the moment one is added.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any, assert_never

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from . import _core
from .core.types import Array, Hyperparam
from .problem import (
    L1,
    ElasticNet,
    GroupL1,
    LogisticLoss,
    Penalty,
    Problem,
    SquaredLoss,
    WeightedL1,
)

__all__ = ["assert_kkt_optimal", "kkt_residual", "pin_blas_threads"]

_F64 = NDArray[np.float64]
_I32 = NDArray[np.int32]


def _matvec(X: Any, v: Array) -> Array:
    if sp.issparse(X):
        return np.asarray(X @ v).ravel()
    return np.asarray(X @ v)


def _rmatvec(X: Any, v: Array) -> Array:
    if sp.issparse(X):
        return np.asarray(X.T @ v).ravel()
    return np.asarray(X.T @ v)


def _grad_datafit(problem: Problem, beta: Array) -> Array:
    """Return ``∇_β L(Xβ, y)`` for the problem's datafit.

    Scaling matches sparho's adapter convention (see
    :mod:`sparho.hypergrad`): ``SquaredLoss`` is the sklearn-style
    ``(1/(2n))‖Xβ − y‖²`` (the ``1/n`` factor on the gradient is required
    for the fixed-point identity to hold against solver outputs).
    ``LogisticLoss`` is the unnormalized sum-of-logs (sklearn's ``C =
    1/α`` parametrization) — no ``1/n`` factor.
    """
    if isinstance(problem.datafit, SquaredLoss):
        n = problem.n_samples
        resid = _matvec(problem.design, beta) - problem.target
        return _rmatvec(problem.design, resid) / n
    if isinstance(problem.datafit, LogisticLoss):
        y = problem.target
        xb = _matvec(problem.design, beta)
        sigma = 1.0 / (1.0 + np.exp(y * xb))  # σ(−y · Xβ)
        return -_rmatvec(problem.design, y * sigma)
    assert_never(problem.datafit)


def _group_layout(penalty: GroupL1) -> tuple[_F64, _I32, _I32]:
    """Lower ``GroupL1.groups`` to the CSR-style layout the Rust kernel expects.

    Returns ``(weights, group_ptr, group_indices)``.
    """
    n_groups = len(penalty.groups)
    group_ptr: _I32 = np.zeros(n_groups + 1, dtype=np.int32)
    flat: list[int] = []
    for k, g in enumerate(penalty.groups):
        group_ptr[k + 1] = group_ptr[k] + len(g)
        flat.extend(g)
    group_indices: _I32 = np.asarray(flat, dtype=np.int32)
    weights: _F64
    if penalty.weights is None:
        weights = np.asarray([float(np.sqrt(len(g))) for g in penalty.groups], dtype=np.float64)
    else:
        weights = np.asarray(penalty.weights, dtype=np.float64)
    return weights, group_ptr, group_indices


def _prox_penalty(penalty: Penalty, z: Array, hp: Hyperparam) -> Array:
    """Apply ``prox_{R(·; α)}(z)`` for the supplied penalty and hyperparameter."""
    z64 = np.ascontiguousarray(z, dtype=np.float64)
    if isinstance(penalty, L1):
        return np.asarray(_core.prox_l1(z64, float(hp)))
    if isinstance(penalty, ElasticNet):
        return np.asarray(_core.prox_elastic_net(z64, float(hp), float(penalty.rho)))
    if isinstance(penalty, WeightedL1):
        alpha = np.ascontiguousarray(hp, dtype=np.float64)
        if alpha.shape != z64.shape:
            raise ValueError(
                f"WeightedL1 hyperparam shape {alpha.shape} does not match "
                f"coef shape {z64.shape}"
            )
        return np.asarray(_core.prox_weighted_l1(z64, alpha))
    if isinstance(penalty, GroupL1):
        weights, group_ptr, group_indices = _group_layout(penalty)
        return np.asarray(_core.prox_group_l1(z64, float(hp), weights, group_ptr, group_indices))
    assert_never(penalty)


def kkt_residual(problem: Problem, hp: Hyperparam, coef: Array) -> float:
    """Infinity-norm of the proximal fixed-point residual at ``coef``.

    Computes ``r = β − prox_{R(·; α)}(β − ∇_β L(Xβ, y))`` and returns
    ``‖r‖_∞``. At an optimum ``β*`` the residual is exactly zero; otherwise
    it is a relative-tolerance witness of distance from KKT optimality.

    Parameters
    ----------
    problem
        The inner problem (datafit + penalty + design + target).
    hp
        The hyperparameter ``α`` at which the residual is evaluated —
        scalar for :class:`L1` / :class:`ElasticNet` / :class:`GroupL1`,
        per-feature array for :class:`WeightedL1`.
    coef
        Candidate coefficient vector ``β`` (typically ``solver_result.coef``).

    Returns
    -------
    float
        ``max_j |β_j − prox_{R(·; α)}(β − ∇L)_j|``.

    Notes
    -----
    The proximal operator is taken with unit step (``τ = 1``); the residual
    is therefore *not* a quantitative bound on the primal suboptimality,
    only a qualitative witness that scales with it. For solver-convergence
    assertions in tests, ``kkt_residual < 10 * solver_tol`` is a robust
    rule of thumb across the v0.1 datafit/penalty family.
    """
    beta = np.ascontiguousarray(coef, dtype=np.float64)
    grad_L = _grad_datafit(problem, beta)
    proximal = _prox_penalty(problem.penalty, beta - grad_L, hp)
    return float(np.max(np.abs(beta - proximal)))


def assert_kkt_optimal(
    problem: Problem,
    hp: Hyperparam,
    coef: Array,
    *,
    atol: float = 1e-3,
    msg: str = "",
) -> None:
    """Assert that ``coef`` satisfies the KKT conditions of ``problem`` at ``hp``.

    Wraps :func:`kkt_residual` with a tolerance check. The default
    ``atol=1e-3`` matches a well-tuned inner solver at ``tol≈1e-6``; tighten
    for golden-regression tests, loosen for end-to-end search tests that
    accept looser inner convergence to amortize wall time.
    """
    res = kkt_residual(problem, hp, coef)
    if res > atol:
        prefix = f"{msg}: " if msg else ""
        raise AssertionError(
            f"{prefix}KKT residual {res:.3e} exceeds tolerance {atol:.3e} "
            f"(datafit={type(problem.datafit).__name__}, "
            f"penalty={type(problem.penalty).__name__})"
        )


# ---------------------------------------------------------------- BLAS pinning


_BLAS_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


@contextlib.contextmanager
def pin_blas_threads(n: int = 1) -> Iterator[dict[str, str | None]]:
    """Pin every BLAS-related thread count to ``n`` for the with-block.

    Sets ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``,
    ``VECLIB_MAXIMUM_THREADS``, ``NUMEXPR_NUM_THREADS``, and
    ``BLIS_NUM_THREADS`` to ``str(n)``, and additionally retunes any live
    threadpool managed by ``threadpoolctl`` (which sklearn ships
    transitively). Both halves matter: env vars affect any subprocess
    spawned inside the block (e.g. joblib workers); the threadpool retuning
    affects BLAS calls in the current process.

    Restores the prior environment + threadpool state on exit. Yields the
    mapping of original env-var values so callers can inspect what was
    overridden.

    Parameters
    ----------
    n
        Number of threads. ``1`` is the canonical setting for benchmarks
        and determinism tests — multi-threaded BLAS introduces nondeterminism
        in reduction ordering even at the same seed.

    Notes
    -----
    Numpy reads ``OMP_NUM_THREADS`` etc. at *first* BLAS call, not on every
    call. If the process has already issued a BLAS operation before this
    context manager runs, the env-var changes will not retune the live
    pool — only ``threadpoolctl`` can. This helper does both, so the
    typical "set early, retune late" foot-gun is handled. For absolute
    determinism, set the env vars before the Python process starts (the
    benchmarks documented in ``docs/reproducibility.md`` follow that
    pattern).
    """
    if n < 1:
        raise ValueError(f"pin_blas_threads: n must be >= 1, got {n}")
    saved_env: dict[str, str | None] = {var: os.environ.get(var) for var in _BLAS_ENV_VARS}
    for var in _BLAS_ENV_VARS:
        os.environ[var] = str(n)
    # threadpoolctl is optional — sparho doesn't declare it directly, but it
    # ships transitively with scikit-learn. If it's missing, fall back to
    # env-var-only mode (still useful for subprocess control).
    threadpool_ctx: contextlib.AbstractContextManager[Any] = contextlib.nullcontext()
    try:
        from threadpoolctl import threadpool_limits

        threadpool_ctx = threadpool_limits(limits=n)
    except ImportError:
        pass
    try:
        with threadpool_ctx:
            yield saved_env
    finally:
        for var, prior in saved_env.items():
            if prior is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prior
