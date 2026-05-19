"""Solver adapters — convert third-party fitters into a ``Solver`` callable."""

from __future__ import annotations

from .callable import as_solver
from .sklearn import (
    SklearnElasticNet,
    SklearnLasso,
    SklearnLogisticRegression,
    SklearnWeightedLasso,
)

__all__ = [
    "SklearnElasticNet",
    "SklearnLasso",
    "SklearnLogisticRegression",
    "SklearnWeightedLasso",
    "as_solver",
]
