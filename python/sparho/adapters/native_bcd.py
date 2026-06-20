"""Native block-coordinate-descent inner solvers (Rust kernels).

These adapters wrap sparho's own BCD coordinate-descent solver (the Rust
``_core.bcd_*`` kernels, ports of sparse-ho's ``compute_beta``) behind the
``Solver`` protocol. Unlike the sklearn/celer adapters they own the inner
solve end-to-end, which lets the ImplicitForward / Forward hypergradients
reuse the same machinery.

The dense path passes ``X`` in column-major (Fortran) order; the sparse path
passes CSC triplets directly — no densification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .. import _core
from .._linalg import column_lipschitz
from ..core.types import Array, Hyperparam
from ..problem import L1, Problem, SquaredLoss
from ..state import SolverResult
from ._common import active_set_of, as_scalar


@dataclass(frozen=True, slots=True)
class NativeBcdLasso:
    """Adapter for ``Problem(SquaredLoss, L1, X, y)`` via the native Rust BCD solver.

    Solves ``(1/2n)‖y − Xβ‖² + α‖β‖₁`` by cyclic coordinate descent with a
    duality-gap stopping test, returning the true gap in ``SolverResult``.

    As with the sklearn adapter, prefer ``tol ≤ 1e-8`` when the criterion uses
    warm-start: the gap test is relative to ``‖y‖²/(2n)``, so a loose ``tol``
    can accept a warm coefficient unchanged and stall outer line searches.
    """

    tol: float = 1e-8
    max_iter: int = 10_000
    gap_freq: int = 10

    def __call__(
        self,
        problem: Problem,
        hyperparam: Hyperparam,
        /,
        *,
        x0: Array | None = None,
        tol: float | None = None,
    ) -> SolverResult:
        if not isinstance(problem.datafit, SquaredLoss) or not isinstance(problem.penalty, L1):
            raise TypeError("NativeBcdLasso requires Problem(SquaredLoss, L1, ...)")
        alpha = as_scalar(hyperparam)
        n_samples = problem.n_samples
        n_features = problem.n_features
        y = np.ascontiguousarray(problem.target, dtype=np.float64)
        beta0 = (
            np.zeros(n_features, dtype=np.float64)
            if x0 is None
            else np.ascontiguousarray(x0, dtype=np.float64)
        )
        lipschitz = column_lipschitz(problem.design, n_samples)
        tol_eff = self.tol if tol is None else float(tol)

        design = problem.design
        if sp.issparse(design):
            X_csc = design.tocsc()  # type: ignore[union-attr]
            coef, n_iter, gap = _core.bcd_lasso_csc(
                X_csc.indptr.astype(np.int32),
                X_csc.indices.astype(np.int32),
                np.ascontiguousarray(X_csc.data, dtype=np.float64),
                n_samples,
                y,
                alpha,
                beta0,
                lipschitz,
                self.max_iter,
                tol_eff,
                self.gap_freq,
            )
        else:
            x_flat = np.ravel(np.asarray(design, dtype=np.float64), order="F")
            coef, n_iter, gap = _core.bcd_lasso_dense(
                x_flat,
                n_samples,
                n_features,
                y,
                alpha,
                beta0,
                lipschitz,
                self.max_iter,
                tol_eff,
                self.gap_freq,
            )
        coef = np.asarray(coef, dtype=np.float64)
        return SolverResult(
            coef=coef,
            active_set=active_set_of(coef),
            dual_gap=float(gap),
            n_iter=int(n_iter),
        )
