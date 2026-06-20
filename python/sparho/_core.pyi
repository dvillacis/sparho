"""Type stubs for the PyO3 extension module ``sparho._core``."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_F64 = NDArray[np.float64]
_I32 = NDArray[np.int32]

def version() -> str: ...

# kernels (element-wise scalar; mostly useful for tests)
def soft_threshold(x: float, alpha: float) -> float: ...
def sigmoid(z: float) -> float: ...

# prox + Jacobians
def prox_l1(z: _F64, alpha: float) -> _F64: ...
def prox_jacobian_l1(z: _F64, alpha: float) -> tuple[_F64, _F64]: ...
def prox_elastic_net(z: _F64, alpha: float, rho: float) -> _F64: ...
def prox_jacobian_elastic_net(z: _F64, alpha: float, rho: float) -> tuple[_F64, _F64]: ...
def prox_weighted_l1(z: _F64, alpha: _F64) -> _F64: ...
def prox_jacobian_weighted_l1(z: _F64, alpha: _F64) -> tuple[_F64, _F64]: ...
def prox_group_l1(
    z: _F64,
    alpha: float,
    weights: _F64,
    group_ptr: _I32,
    group_indices: _I32,
) -> _F64: ...

# csc sparse matrix–vector
def csc_matvec(indptr: _I32, indices: _I32, data: _F64, n_samples: int, x: _F64) -> _F64: ...
def csc_rmatvec(indptr: _I32, indices: _I32, data: _F64, y: _F64) -> _F64: ...

# restricted Hessian–vector product (least-squares; v0.1 Lasso family)
def restricted_ls_hessian_matvec(
    indptr: _I32,
    indices: _I32,
    data: _F64,
    n_samples: int,
    active: _I32,
    v: _F64,
) -> _F64: ...

# block-coordinate-descent inner solvers (β-only; returns (beta, n_iter, dual_gap)).
# Dense ``x`` is column-major (Fortran) flattened: column j is x[j*n_samples:(j+1)*n_samples].
def bcd_lasso_dense(
    x: _F64,
    n_samples: int,
    n_features: int,
    y: _F64,
    alpha: float,
    beta0: _F64,
    lipschitz: _F64,
    max_iter: int,
    tol: float,
    gap_freq: int,
) -> tuple[_F64, int, float]: ...
def bcd_lasso_csc(
    indptr: _I32,
    indices: _I32,
    data: _F64,
    n_samples: int,
    y: _F64,
    alpha: float,
    beta0: _F64,
    lipschitz: _F64,
    max_iter: int,
    tol: float,
    gap_freq: int,
) -> tuple[_F64, int, float]: ...

# Joint β+Jacobian BCD solve (Forward); returns (beta, dbeta, n_iter, dual_gap).
# Tracks dβ/dα over all features alongside β. Dense ``x`` is column-major.
def bcd_lasso_jac_dense(
    x: _F64,
    n_samples: int,
    n_features: int,
    y: _F64,
    alpha: float,
    beta0: _F64,
    dbeta0: _F64,
    lipschitz: _F64,
    max_iter: int,
    tol: float,
    gap_freq: int,
) -> tuple[_F64, _F64, int, float]: ...
def bcd_lasso_jac_csc(
    indptr: _I32,
    indices: _I32,
    data: _F64,
    n_samples: int,
    y: _F64,
    alpha: float,
    beta0: _F64,
    dbeta0: _F64,
    lipschitz: _F64,
    max_iter: int,
    tol: float,
    gap_freq: int,
) -> tuple[_F64, _F64, int, float]: ...

# Reverse-mode (Backward) hypergradient for dense Lasso; returns (dC_dalpha, n_iter).
# Solves while recording β sweeps, then reverse-replays. ``v`` is ∂C/∂β (length n_features).
def bcd_lasso_backward_dense(
    x: _F64,
    n_samples: int,
    n_features: int,
    y: _F64,
    alpha: float,
    v: _F64,
    lipschitz: _F64,
    max_iter: int,
    tol: float,
    gap_freq: int,
) -> tuple[float, int]: ...

# Restricted normal-equation solve (ImplicitForward primitive); returns (x, n_iter).
# Solves (XsᵀXs/n + diag_shift·I) x = b on the support by coordinate descent.
# Dense ``xs`` is column-major; the CSC variant takes the full matrix + ``active``.
def solve_restricted_normal_dense(
    xs: _F64,
    n_samples: int,
    n_active: int,
    b: _F64,
    diag_shift: float,
    x0: _F64,
    lipschitz: _F64,
    max_iter: int,
    tol: float,
) -> tuple[_F64, int]: ...
def solve_restricted_normal_csc(
    indptr: _I32,
    indices: _I32,
    data: _F64,
    n_samples: int,
    active: _I32,
    b: _F64,
    diag_shift: float,
    x0: _F64,
    lipschitz: _F64,
    max_iter: int,
    tol: float,
) -> tuple[_F64, int]: ...
