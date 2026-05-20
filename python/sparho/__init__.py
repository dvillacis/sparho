"""sparho — nonsmooth bilevel hyperparameter optimization."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from sparho import _core

from . import adapters
from .core.types import Array, DesignMatrix, Hyperparam, IndexArray, Scalar
from .criteria import (
    Criterion,
    CriterionResult,
    CrossVal,
    HeldOutLogistic,
    HeldOutMSE,
    Sure,
)
from .hypergrad import implicit_forward
from .problem import (
    L1,
    Datafit,
    ElasticNet,
    LogisticLoss,
    Penalty,
    Problem,
    SquaredLoss,
    WeightedL1,
)
from .search import grad_search, hoag_search
from .solver import Solver
from .state import IterationRecord, SearchResult, SearchState, SolverResult

try:
    __version__ = _pkg_version("sparho")
except PackageNotFoundError:  # pragma: no cover — editable installs without metadata
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "_core",
    "adapters",
    # types
    "Array",
    "DesignMatrix",
    "Hyperparam",
    "IndexArray",
    "Scalar",
    # problem
    "Datafit",
    "ElasticNet",
    "L1",
    "LogisticLoss",
    "Penalty",
    "Problem",
    "SquaredLoss",
    "WeightedL1",
    # state
    "IterationRecord",
    "SearchResult",
    "SearchState",
    "SolverResult",
    # solver protocol
    "Solver",
    # criterion + criteria
    "Criterion",
    "CriterionResult",
    "CrossVal",
    "HeldOutLogistic",
    "HeldOutMSE",
    "Sure",
    # hypergradient + search
    "grad_search",
    "hoag_search",
    "implicit_forward",
]
