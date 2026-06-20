"""Python orchestration of the native BCD ImplicitForward kernels.

Every Lasso-family hypergradient reduces to one primitive: solve the
implicit-differentiation linear system restricted to the active set,
``(XsᵀXs/n + c·I) x = b``, via the Rust ``_core.solve_restricted_normal_*``
coordinate-descent kernel. Each penalty supplies its own ``(b, c)`` and
contracts the result with the criterion gradient ``∂C/∂β``:

- ``L1``: ``c = 0``, ``b = −sign`` ⇒ ``x = dβ/dα``; ``dC/dα = ∂C/∂β_A · x``.
- ``ElasticNet(ρ)``: ``c = α(1−ρ)``, ``b = −(ρ·sign + (1−ρ)·β_A)`` ⇒ ``x = dβ/dα``;
  same contraction. (sparho's single-``α`` convention maps to sparse-ho's
  ``(α₁, α₂)`` via ``α₁ = αρ``, ``α₂ = α(1−ρ)``.)
- ``WeightedL1``: ``c = 0``, ``b = ∂C/∂β_A`` ⇒ ``x = M_AA⁻¹ ∂C/∂β_A`` (adjoint
  solve); ``dC/dα_j = −sign_j · x_j`` on the support, zero elsewhere.

All kernels compute ``dβ/dα`` (no ``α`` prefactor), matching
:func:`sparho.hypergrad.implicit` — the outer loop applies the log-space ``·α``
factor.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

from .. import _core
from ..core.types import Array, Hyperparam
from ..problem import Problem
from ..state import SolverResult

# Default cap for the fixed point. Each sweep is cheap (one BCD pass over the
# small active set) and converges linearly, so a generous cap costs little and
# keeps tight tolerances satisfiable.
_DEFAULT_MAX_ITER_JAC = 10_000


def _warn_nonfinite() -> None:
    # Mirror the CG path's warning vocabulary so search._cg_status_from_warnings
    # surfaces this as ``cg_status="nonfinite"`` without any change there.
    warnings.warn(
        "implicit_forward: CG failed (finite=False); returning zero hypergradient "
        "for this iter — outer step will stall",
        RuntimeWarning,
        stacklevel=3,
    )


def _solve_restricted_normal(
    problem: Problem,
    active: np.ndarray,
    b: Array,
    diag_shift: float,
    *,
    tol: float,
    max_iter: int,
    x0: Array | None,
) -> Array:
    """Solve ``(XsᵀXs/n + diag_shift·I) x = b`` on ``active`` via the Rust kernel."""
    n_samples = problem.n_samples
    design = problem.design
    b_arr = np.ascontiguousarray(b, dtype=np.float64)
    x0_arr = (
        np.zeros(active.size, dtype=np.float64)
        if x0 is None
        else np.ascontiguousarray(x0, dtype=np.float64)
    )
    if sp.issparse(design):
        X_csc = design.tocsc()  # type: ignore[union-attr]
        active_i32 = active.astype(np.int32)
        cols = X_csc[:, active_i32]
        col_sq = np.asarray(cols.multiply(cols).sum(axis=0)).ravel()
        lipschitz = np.ascontiguousarray(col_sq / n_samples, dtype=np.float64)
        x, _n_iter = _core.solve_restricted_normal_csc(
            X_csc.indptr.astype(np.int32),
            X_csc.indices.astype(np.int32),
            np.ascontiguousarray(X_csc.data, dtype=np.float64),
            n_samples,
            active_i32,
            b_arr,
            float(diag_shift),
            x0_arr,
            lipschitz,
            max_iter,
            tol,
        )
    else:
        Xs = np.ascontiguousarray(np.asarray(design, dtype=np.float64)[:, active])
        col_sq = np.einsum("ij,ij->j", Xs, Xs)
        lipschitz = np.ascontiguousarray(col_sq / n_samples, dtype=np.float64)
        xs_flat = np.ravel(Xs, order="F")
        x, _n_iter = _core.solve_restricted_normal_dense(
            xs_flat,
            n_samples,
            int(active.size),
            b_arr,
            float(diag_shift),
            x0_arr,
            lipschitz,
            max_iter,
            tol,
        )
    return np.asarray(x, dtype=np.float64)


def forward_lasso(
    problem: Problem,
    hyperparam: Hyperparam,
    criterion_grad_beta: Array,
    *,
    tol: float = 1e-8,
    max_iter: int = _DEFAULT_MAX_ITER_JAC,
    gap_freq: int = 10,
) -> float:
    """Forward-mode hypergradient for ``Problem(SquaredLoss, L1, …)``.

    Re-solves the inner problem from cold while tracking the full Jacobian
    ``dβ/dα`` jointly (sparse-ho's ``compute_beta`` with ``compute_jac=True``),
    then contracts ``∂C/∂β · dβ``. Equivalent in result to
    :func:`implicit_forward_lasso`, but computed by differentiating *through* the
    coordinate descent rather than by a support-restricted post-solve — hence it
    does not read ``solver_result`` and re-solves β itself.
    """
    alpha = float(np.asarray(hyperparam))
    n_samples = problem.n_samples
    n_features = problem.n_features
    y = np.ascontiguousarray(problem.target, dtype=np.float64)
    beta0 = np.zeros(n_features, dtype=np.float64)
    dbeta0 = np.zeros(n_features, dtype=np.float64)
    design = problem.design
    if sp.issparse(design):
        X_csc = design.tocsc()  # type: ignore[union-attr]
        col_sq = np.asarray(X_csc.multiply(X_csc).sum(axis=0)).ravel()
        lipschitz = np.ascontiguousarray(col_sq / n_samples, dtype=np.float64)
        _beta, dbeta, _n_iter, _gap = _core.bcd_lasso_jac_csc(
            X_csc.indptr.astype(np.int32),
            X_csc.indices.astype(np.int32),
            np.ascontiguousarray(X_csc.data, dtype=np.float64),
            n_samples,
            y,
            alpha,
            beta0,
            dbeta0,
            lipschitz,
            max_iter,
            tol,
            gap_freq,
        )
    else:
        Xa = np.asarray(design, dtype=np.float64)
        col_sq = np.einsum("ij,ij->j", Xa, Xa)
        lipschitz = np.ascontiguousarray(col_sq / n_samples, dtype=np.float64)
        x_flat = np.ravel(Xa, order="F")
        _beta, dbeta, _n_iter, _gap = _core.bcd_lasso_jac_dense(
            x_flat,
            n_samples,
            n_features,
            y,
            alpha,
            beta0,
            dbeta0,
            lipschitz,
            max_iter,
            tol,
            gap_freq,
        )
    dbeta = np.asarray(dbeta, dtype=np.float64)
    if not np.all(np.isfinite(dbeta)):
        _warn_nonfinite()
        return 0.0
    grad = np.ascontiguousarray(criterion_grad_beta, dtype=np.float64)
    return float(np.dot(grad, dbeta))


def backward_lasso(
    problem: Problem,
    hyperparam: Hyperparam,
    criterion_grad_beta: Array,
    *,
    tol: float = 1e-8,
    max_iter: int = _DEFAULT_MAX_ITER_JAC,
    gap_freq: int = 10,
) -> float:
    """Reverse-mode hypergradient for dense ``Problem(SquaredLoss, L1, …)``.

    Solves while recording β sweeps, then reverse-replays to accumulate
    ``∂C/∂β · dβ/dα`` (sparse-ho's ``get_grad_backward``). Dense only — the
    reverse replay forms the Gram matrix; the caller delegates other cases.
    """
    alpha = float(np.asarray(hyperparam))
    n_samples = problem.n_samples
    n_features = problem.n_features
    y = np.ascontiguousarray(problem.target, dtype=np.float64)
    v = np.ascontiguousarray(criterion_grad_beta, dtype=np.float64)
    Xa = np.asarray(problem.design, dtype=np.float64)
    col_sq = np.einsum("ij,ij->j", Xa, Xa)
    lipschitz = np.ascontiguousarray(col_sq / n_samples, dtype=np.float64)
    x_flat = np.ravel(Xa, order="F")
    grad, _n_iter = _core.bcd_lasso_backward_dense(
        x_flat, n_samples, n_features, y, alpha, v, lipschitz, max_iter, tol, gap_freq
    )
    if not np.isfinite(grad):
        _warn_nonfinite()
        return 0.0
    return float(grad)


def implicit_forward_lasso(
    problem: Problem,
    hyperparam: Hyperparam,
    solver_result: SolverResult,
    criterion_grad_beta: Array,
    *,
    tol_jac: float = 1e-8,
    max_iter_jac: int = _DEFAULT_MAX_ITER_JAC,
    jac0: Array | None = None,
) -> tuple[float, Array | None]:
    """ImplicitForward hypergradient for ``Problem(SquaredLoss, L1, …)``.

    Returns ``(dC/dα, dβ)`` — the scalar hypergradient and the support Jacobian
    (the latter is the warm-start payload; ``None`` on an empty support or a
    non-finite solve).
    """
    del hyperparam  # L1 Jacobian needs only the support + sign, carried by solver_result.
    active = solver_result.active_set
    if active.size == 0:
        return 0.0, None
    sign_beta = np.sign(solver_result.coef[active]).astype(np.float64)
    grad_C_A = np.ascontiguousarray(criterion_grad_beta[active], dtype=np.float64)
    dbeta = _solve_restricted_normal(
        problem, active, -sign_beta, 0.0, tol=tol_jac, max_iter=max_iter_jac, x0=jac0
    )
    if not np.all(np.isfinite(dbeta)):
        _warn_nonfinite()
        return 0.0, None
    return float(np.dot(grad_C_A, dbeta)), dbeta


def implicit_forward_enet(
    problem: Problem,
    hyperparam: Hyperparam,
    solver_result: SolverResult,
    criterion_grad_beta: Array,
    *,
    rho: float,
    tol_jac: float = 1e-8,
    max_iter_jac: int = _DEFAULT_MAX_ITER_JAC,
    jac0: Array | None = None,
) -> tuple[float, Array | None]:
    """ImplicitForward hypergradient for ``Problem(SquaredLoss, ElasticNet(ρ), …)``.

    Returns ``(dC/dα, dβ)``; see :func:`implicit_forward_lasso`.
    """
    active = solver_result.active_set
    if active.size == 0:
        return 0.0, None
    alpha = float(np.asarray(hyperparam))
    beta_A = np.ascontiguousarray(solver_result.coef[active], dtype=np.float64)
    sign_beta = np.sign(beta_A)
    grad_C_A = np.ascontiguousarray(criterion_grad_beta[active], dtype=np.float64)
    diag_shift = alpha * (1.0 - rho)
    b = -(rho * sign_beta + (1.0 - rho) * beta_A)
    dbeta = _solve_restricted_normal(
        problem, active, b, diag_shift, tol=tol_jac, max_iter=max_iter_jac, x0=jac0
    )
    if not np.all(np.isfinite(dbeta)):
        _warn_nonfinite()
        return 0.0, None
    return float(np.dot(grad_C_A, dbeta)), dbeta


def implicit_forward_wl1(
    problem: Problem,
    hyperparam: Hyperparam,
    solver_result: SolverResult,
    criterion_grad_beta: Array,
    *,
    tol_jac: float = 1e-8,
    max_iter_jac: int = _DEFAULT_MAX_ITER_JAC,
    jac0: Array | None = None,
) -> tuple[Array, Array | None]:
    """ImplicitForward hypergradient for ``Problem(SquaredLoss, WeightedL1, …)``.

    Returns ``(dC/dα, v)``: the ``(n_features,)`` hypergradient (zero outside the
    active set) and the support adjoint solve ``v = M_AA⁻¹ ∂C/∂β_A`` (the
    warm-start payload). The per-feature α has a diagonal α-Jacobian, so
    ``dC/dα_j = −sign_j · v_j``.
    """
    del hyperparam  # WeightedL1 curvature is zero; α does not enter M_AA.
    n_features = problem.n_features
    active = solver_result.active_set
    if active.size == 0:
        return np.zeros(n_features, dtype=np.float64), None
    sign_beta = np.sign(solver_result.coef[active]).astype(np.float64)
    grad_C_A = np.ascontiguousarray(criterion_grad_beta[active], dtype=np.float64)
    v = _solve_restricted_normal(
        problem, active, grad_C_A, 0.0, tol=tol_jac, max_iter=max_iter_jac, x0=jac0
    )
    out = np.zeros(n_features, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        _warn_nonfinite()
        return out, None
    out[active] = -sign_beta * v
    return out, v
