"""Opt-in Jacobian warm-starting across outer iterations.

The ImplicitForward Jacobian fixed point converges faster when seeded from the
previous outer iteration's solution. :class:`WarmStartHypergrad` wraps the BCD
dispatch, caches the support-restricted solve vector keyed by the active set,
and remaps it onto each new support via :func:`sparho.hypergrad._shared.init_dbeta0_new`
(carry shared coordinates, zero-fill new ones, drop dropped ones).

It conforms to the ``HypergradFn`` seam, so it drops into ``grad_search`` /
``hoag_search`` exactly like a plain hypergradient::

    search(..., hypergrad=WarmStartHypergrad())

The cache lives in a deliberately-mutable, hash/eq-excluded field — the same
idiom as :class:`sparho.CrossVal`'s per-fold cache — so the dataclass stays a
well-behaved value object. Warm-starting changes only the iteration count, not
the converged answer (the inner problem is convex), so results match cold-start.
Only the native BCD penalties (``SquaredLoss × {L1, ElasticNet, WeightedL1}``)
warm-start; the CG-fallback pairs produce no cacheable Jacobian and run cold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.types import Array, Hyperparam
from ..problem import Problem
from ..state import SolverResult
from ._shared import init_dbeta0_new
from .implicit_forward import dispatch_bcd


@dataclass(frozen=True, slots=True)
class WarmStartHypergrad:
    """A ``HypergradFn`` that warm-starts the ImplicitForward Jacobian fixed point.

    Parameters
    ----------
    tol, maxiter, ridge
        Forwarded to the underlying dispatch (Jacobian-iteration tolerance / cap;
        ``ridge`` applies only on the CG-fallback pairs).
    """

    tol: float = 1e-8
    maxiter: int | None = None
    ridge: float | None = None
    # At most one (support_mask, jac) tuple. Mutable, excluded from eq/hash/repr.
    _cache: list[tuple[np.ndarray, Array]] = field(
        default_factory=list, compare=False, repr=False, hash=False
    )

    def __call__(
        self,
        problem: Problem,
        hyperparam: Hyperparam,
        solver_result: SolverResult,
        criterion_grad_beta: Array,
    ) -> Hyperparam:
        """Compute ``dC/dα``, seeding the Jacobian solve from the cached solution."""
        n_features = problem.n_features
        active = solver_result.active_set
        mask_new = np.zeros(n_features, dtype=bool)
        mask_new[active] = True

        jac0: Array | None = None
        if self._cache:
            mask_old, jac_old = self._cache[0]
            jac0 = init_dbeta0_new(jac_old, mask_new, mask_old)

        hg, jac = dispatch_bcd(
            problem,
            hyperparam,
            solver_result,
            criterion_grad_beta,
            tol=self.tol,
            maxiter=self.maxiter,
            ridge=self.ridge,
            jac0=jac0,
        )

        self._cache.clear()
        if jac is not None:
            self._cache.append((mask_new, np.asarray(jac, dtype=np.float64).copy()))
        return hg
