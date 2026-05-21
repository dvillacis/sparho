"""Internal helpers shared across adapter modules."""

from __future__ import annotations

from typing import cast

import numpy as np

from ..core.types import Array, Hyperparam


def as_scalar(hp: Hyperparam) -> float:
    """Coerce a scalar hyperparameter (Python float, numpy float, or 0-d array)."""
    if isinstance(hp, int | float | np.floating):
        return float(hp)
    arr = cast(np.ndarray, hp)
    if isinstance(arr, np.ndarray) and arr.ndim == 0:
        return float(arr)
    actual = np.asarray(hp).shape
    raise TypeError(
        f"expected scalar hyperparameter α (Python float or 0-d ndarray), "
        f"got ndarray of shape {actual}; this adapter only supports the scalar-α "
        "penalties (L1 / ElasticNet / GroupL1) — use a WeightedL1 adapter for "
        "per-feature α"
    )


def as_vector(hp: Hyperparam, n_features: int) -> Array:
    """Coerce a per-feature hyperparameter vector of length ``n_features``."""
    arr = np.asarray(hp, dtype=np.float64)
    if arr.ndim != 1:
        raise TypeError(
            f"expected per-feature α as a 1-D ndarray of shape ({n_features},), "
            f"got ndim={arr.ndim} (shape {arr.shape}); this adapter is for WeightedL1"
        )
    if arr.shape[0] != n_features:
        raise TypeError(
            f"per-feature α length mismatch: expected ({n_features},) to match "
            f"design.shape[1], got ({arr.shape[0]},)"
        )
    return arr


def active_set_of(coef: Array) -> np.ndarray:
    """``np.flatnonzero(coef)`` as a sorted int32 index array."""
    return np.flatnonzero(coef).astype(np.int32)
