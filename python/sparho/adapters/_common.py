"""Internal helpers shared across adapter modules."""

from __future__ import annotations

from typing import cast

import numpy as np

from ..core.types import Array, Hyperparam


def as_scalar(hp: Hyperparam) -> float:
    """Coerce a scalar hyperparameter (Python float, numpy float, or 0-d array)."""
    if isinstance(hp, (int, float, np.floating)):
        return float(hp)
    arr = cast(np.ndarray, hp)
    if isinstance(arr, np.ndarray) and arr.ndim == 0:
        return float(arr)
    raise TypeError(f"expected scalar hyperparameter, got shape {np.asarray(hp).shape}")


def as_vector(hp: Hyperparam, n_features: int) -> Array:
    """Coerce a per-feature hyperparameter vector of length ``n_features``."""
    arr = np.asarray(hp, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != n_features:
        raise TypeError(
            f"expected per-feature hyperparameter of shape ({n_features},), "
            f"got {arr.shape}"
        )
    return arr


def active_set_of(coef: Array) -> np.ndarray:
    """``np.flatnonzero(coef)`` as a sorted int32 index array."""
    return np.flatnonzero(coef).astype(np.int32)
