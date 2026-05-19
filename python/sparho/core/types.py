"""Public type aliases used across sparho.

Imports are kept to numpy + scipy.sparse to honor the Phase-1 constraint that
core types do not depend on anything heavier than numpy.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

Scalar: TypeAlias = float | np.floating
"""A real scalar — either a Python float or any numpy floating type."""

Array: TypeAlias = NDArray[np.floating]
"""A real-valued numpy array of arbitrary shape."""

Hyperparam: TypeAlias = float | Array
"""Outer-loop hyperparameter — scalar (e.g. Lasso `alpha`) or per-feature vector
(e.g. weighted Lasso `alpha_j`)."""

DesignMatrix: TypeAlias = NDArray[np.floating] | sp.csc_matrix | sp.csc_array
"""Either a dense numpy matrix or a scipy.sparse CSC matrix/array.

CSC is the v0.1 sparse format. CSR and COO inputs are expected to be converted
by the caller via `.tocsc()` before reaching the inner solvers."""

IndexArray: TypeAlias = NDArray[np.int32]
"""Index arrays (active set, fold indices). int32 to match CSC indices."""
