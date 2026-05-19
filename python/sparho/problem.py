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

from dataclasses import dataclass
from typing import TypeAlias

from .core.types import Array, DesignMatrix

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


@dataclass(frozen=True, slots=True)
class WeightedL1:
    """``R(β; α) = Σⱼ αⱼ · |βⱼ|`` with per-feature hyperparameter vector ``α``."""


Penalty: TypeAlias = L1 | ElasticNet | WeightedL1

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

    @property
    def n_samples(self) -> int:
        """Number of observations (``X.shape[0]``)."""
        return int(self.design.shape[0])

    @property
    def n_features(self) -> int:
        """Number of features (``X.shape[1]``)."""
        return int(self.design.shape[1])
