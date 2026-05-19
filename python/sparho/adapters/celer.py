"""celer adapters (optional, behind the ``[celer]`` extra).

celer is a coordinate-descent solver for Lasso-family problems with extrapolation
and working-set screening; faster than sklearn on large sparse problems. The
adapter exposes ``CelerLasso`` and ``CelerElasticNet``; ``celer`` itself is
imported lazily so the module loads even when the extra isn't installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import Array, Hyperparam
from ..problem import L1, Problem, SquaredLoss
from ..problem import (
    ElasticNet as ElasticNetPenalty,
)
from ..state import SolverResult
from ._common import active_set_of, as_scalar


def _require_celer() -> None:
    try:
        import celer  # noqa: F401
    except ImportError as exc:  # pragma: no cover — import guard
        raise ImportError(
            "celer adapters require the `[celer]` extra: `pip install sparho[celer]`"
        ) from exc


@dataclass(frozen=True, slots=True)
class CelerLasso:
    """Adapter for ``Problem(SquaredLoss, L1, X, y)`` via ``celer.Lasso``."""

    tol: float = 1e-6
    max_iter: int = 100  # celer's iter counter is outer-iterations; small is fine

    def __call__(
        self,
        problem: Problem,
        hyperparam: Hyperparam,
        /,
        *,
        x0: Array | None = None,
        tol: float | None = None,
    ) -> SolverResult:
        _require_celer()
        from celer import Lasso as _CelerLasso

        if not isinstance(problem.datafit, SquaredLoss) or not isinstance(problem.penalty, L1):
            raise TypeError("CelerLasso requires Problem(SquaredLoss, L1, ...)")
        alpha = as_scalar(hyperparam)
        est = _CelerLasso(
            alpha=alpha,
            fit_intercept=False,
            tol=self.tol if tol is None else float(tol),
            max_iter=self.max_iter,
            warm_start=x0 is not None,
        )
        if x0 is not None:
            est.coef_ = np.ascontiguousarray(np.asarray(x0, dtype=np.float64))
        est.fit(problem.design, problem.target)
        coef = np.asarray(est.coef_, dtype=np.float64)
        return SolverResult(
            coef=coef,
            active_set=active_set_of(coef),
            dual_gap=float(getattr(est, "dual_gap_", 0.0)),
            n_iter=int(np.atleast_1d(getattr(est, "n_iter_", 0))[0] or 0),
        )


@dataclass(frozen=True, slots=True)
class CelerElasticNet:
    """Adapter for ``Problem(SquaredLoss, ElasticNet(rho), X, y)`` via celer."""

    tol: float = 1e-6
    max_iter: int = 100

    def __call__(
        self,
        problem: Problem,
        hyperparam: Hyperparam,
        /,
        *,
        x0: Array | None = None,
        tol: float | None = None,
    ) -> SolverResult:
        _require_celer()
        from celer import ElasticNet as _CelerEN

        if not isinstance(problem.datafit, SquaredLoss) or not isinstance(
            problem.penalty, ElasticNetPenalty
        ):
            raise TypeError("CelerElasticNet requires Problem(SquaredLoss, ElasticNet, ...)")
        alpha = as_scalar(hyperparam)
        est = _CelerEN(
            alpha=alpha,
            l1_ratio=problem.penalty.rho,
            fit_intercept=False,
            tol=self.tol if tol is None else float(tol),
            max_iter=self.max_iter,
            warm_start=x0 is not None,
        )
        if x0 is not None:
            est.coef_ = np.ascontiguousarray(np.asarray(x0, dtype=np.float64))
        est.fit(problem.design, problem.target)
        coef = np.asarray(est.coef_, dtype=np.float64)
        return SolverResult(
            coef=coef,
            active_set=active_set_of(coef),
            dual_gap=float(getattr(est, "dual_gap_", 0.0)),
            n_iter=int(np.atleast_1d(getattr(est, "n_iter_", 0))[0] or 0),
        )
