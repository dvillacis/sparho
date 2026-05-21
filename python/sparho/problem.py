"""Bilevel problem definition: ``Problem = (datafit, penalty, design, target)``.

The Datafit and Penalty families are tagged unions of frozen dataclasses. The
v0.1 set is closed; algorithms exhaustively dispatch via ``match`` statements
with ``typing.assert_never`` on the default branch so mypy will flag any
unhandled case.

Extending the library with a new datafit/penalty means: (1) add a new frozen
dataclass to the union here, (2) implement the corresponding Rust kernel
under ``crates/sparho-core``, (3) add a new match arm in each algorithm that
dispatches on it. There is no inheritance hierarchy to subclass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import scipy.sparse as sp

from .core.types import Array, DesignMatrix

# Module-level kill switch for the finiteness checks added in v0.3.1. Users
# with deliberately NaN-padded designs (e.g. missing-data masks they handle
# downstream) can flip this to ``False`` to skip the ``np.isfinite`` scan.
# Shape and dtype checks are non-negotiable.
CHECK_FINITE: bool = True

# ---------------------------------------------------------------- Datafit


@dataclass(frozen=True, slots=True)
class SquaredLoss:
    """``L(Xβ, y) = 0.5 · ‖Xβ − y‖²``."""


@dataclass(frozen=True, slots=True)
class LogisticLoss:
    """``L(Xβ, y) = Σᵢ log(1 + exp(−yᵢ (Xβ)ᵢ))`` with ``yᵢ ∈ {−1, +1}``."""


Datafit: TypeAlias = SquaredLoss | LogisticLoss
"""v0.1 datafit family. ``SmoothHinge`` for SVM/SVR is deliberately out of scope."""

# ---------------------------------------------------------------- Penalty


@dataclass(frozen=True, slots=True)
class L1:
    """``R(β; α) = α · ‖β‖₁`` with scalar hyperparameter ``α > 0``."""


@dataclass(frozen=True, slots=True)
class ElasticNet:
    """``R(β; α) = α · (ρ · ‖β‖₁ + (1 − ρ)/2 · ‖β‖²)``.

    The mixing weight ``ρ ∈ (0, 1]`` is structural (carried here, not tuned).
    The hyperparameter optimized by ``grad_search`` is the scalar ``α``.
    """

    rho: float

    def __post_init__(self) -> None:
        """Validate ``rho ∈ (0, 1]`` at construction."""
        if not (0.0 < self.rho <= 1.0):
            raise ValueError(f"ElasticNet.rho must lie in (0, 1], got {self.rho!r}")


@dataclass(frozen=True, slots=True)
class WeightedL1:
    """``R(β; α) = Σⱼ αⱼ · |βⱼ|`` with per-feature hyperparameter vector ``α``."""


@dataclass(frozen=True, slots=True)
class GroupL1:
    """``R(β; α) = α · Σ_k w_k · ‖β_{G_k}‖_2`` — block-sparsity penalty.

    Each group ``G_k ⊆ {0, …, p−1}`` is shrunk together by a block
    soft-threshold: generically either all of ``β_{G_k}`` is zero, or all of
    it is nonzero. With ``w_k = √|G_k|`` (the default) the penalty is invariant
    to group size — Yuan & Lin 2006's standard scaling.

    The hyperparameter optimized by ``grad_search`` is the scalar ``α``;
    ``groups`` and ``weights`` are structural (carried, not tuned).

    Parameters
    ----------
    groups
        Tuple of tuples — ``groups[k]`` is the feature indices of the ``k``-th
        group. Required to partition ``{0, …, n_features − 1}`` (disjoint and
        covering all features). Order is structural and indexes ``weights``.
        Use :meth:`from_labels` to build from a length-``n_features`` array
        of group labels.
    weights
        Optional per-group multipliers; ``None`` (default) resolves to
        ``√|G_k|`` at use site.
    """

    groups: tuple[tuple[int, ...], ...]
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the partition (disjoint, non-empty, in-range) at construction."""
        seen: set[int] = set()
        for k, g in enumerate(self.groups):
            if not g:
                raise ValueError(f"GroupL1: group {k} is empty")
            for j in g:
                if not isinstance(j, int) or j < 0:
                    raise ValueError(
                        f"GroupL1: group {k} contains non-negative-int index {j!r}"
                    )
                if j in seen:
                    raise ValueError(f"GroupL1: feature {j} appears in more than one group")
                seen.add(j)
        if self.weights is not None and len(self.weights) != len(self.groups):
            raise ValueError(
                f"GroupL1.weights length ({len(self.weights)}) must equal "
                f"len(groups) ({len(self.groups)})"
            )

    @classmethod
    def from_labels(
        cls,
        labels: Array | Sequence[int],
        *,
        weights: tuple[float, ...] | None = None,
    ) -> GroupL1:
        """Build from a length-``n_features`` integer label array.

        ``labels[j] = k`` ⇒ feature ``j`` is in group ``k``. Labels must be
        a contiguous range ``0 … K−1``.
        """
        arr = np.asarray(labels, dtype=np.int64)
        if arr.size == 0:
            return cls(groups=(), weights=weights)
        k_max = int(arr.max())
        if int(arr.min()) < 0:
            raise ValueError("group labels must be non-negative")
        groups = tuple(
            tuple(int(j) for j in np.flatnonzero(arr == k)) for k in range(k_max + 1)
        )
        empty = [k for k, g in enumerate(groups) if not g]
        if empty:
            raise ValueError(f"empty groups not allowed; labels missing: {empty}")
        return cls(groups=groups, weights=weights)


Penalty: TypeAlias = L1 | ElasticNet | WeightedL1 | GroupL1

# ---------------------------------------------------------------- Problem


@dataclass(frozen=True, slots=True)
class Problem:
    """A bilevel inner problem ``argmin_β  L(Xβ, y) + R(β; α)``.

    The hyperparameter ``α`` is **not** stored here — it is what the outer
    search tunes. The problem captures the fixed structure: which loss, which
    regularizer family, which design matrix, which target vector.
    """

    datafit: Datafit
    penalty: Penalty
    design: DesignMatrix
    target: Array

    def __post_init__(self) -> None:
        """Validate shape, dtype, and finiteness of design/target at construction."""
        if getattr(self.design, "ndim", None) != 2:
            raise ValueError(
                f"Problem.design must be 2-D, got ndim={getattr(self.design, 'ndim', None)!r}"
            )
        target = np.asarray(self.target)
        if target.ndim != 1:
            raise ValueError(f"Problem.target must be 1-D, got ndim={target.ndim}")
        if target.shape[0] != self.design.shape[0]:
            raise ValueError(
                f"Problem.target length ({target.shape[0]}) must equal "
                f"design.shape[0] ({self.design.shape[0]})"
            )
        if CHECK_FINITE:
            if sp.issparse(self.design):
                # ``.data`` holds the explicit non-zeros; implicit zeros are
                # finite by definition.
                design_data = np.asarray(self.design.data)
            else:
                design_data = np.asarray(self.design)
            if design_data.size and not np.isfinite(design_data).all():
                raise ValueError(
                    "Problem.design contains NaN/Inf; set sparho.problem.CHECK_FINITE = False "
                    "to opt out (e.g. for masked-input pipelines)"
                )
            if target.size and not np.isfinite(target).all():
                raise ValueError(
                    "Problem.target contains NaN/Inf; set sparho.problem.CHECK_FINITE = False "
                    "to opt out"
                )

    @property
    def n_samples(self) -> int:
        """Number of observations (``X.shape[0]``)."""
        return int(self.design.shape[0])

    @property
    def n_features(self) -> int:
        """Number of features (``X.shape[1]``)."""
        return int(self.design.shape[1])
