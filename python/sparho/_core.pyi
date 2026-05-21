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
def prox_jacobian_elastic_net(
    z: _F64, alpha: float, rho: float
) -> tuple[_F64, _F64]: ...
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
def csc_matvec(
    indptr: _I32, indices: _I32, data: _F64, n_samples: int, x: _F64
) -> _F64: ...
def csc_rmatvec(
    indptr: _I32, indices: _I32, data: _F64, y: _F64
) -> _F64: ...

# restricted Hessian–vector product (least-squares; v0.1 Lasso family)
def restricted_ls_hessian_matvec(
    indptr: _I32,
    indices: _I32,
    data: _F64,
    n_samples: int,
    active: _I32,
    v: _F64,
) -> _F64: ...
