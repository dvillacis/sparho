"""Regenerate the golden numerical-regression fixtures.

Each fixture pins ``(β*, training loss, KKT residual)`` for one
``(datafit, penalty, solver, α)`` triple on a deterministic synthetic
problem. The runner in ``tests/test_golden.py`` re-solves each one and
asserts agreement at tight tolerance.

When this script is rerun (e.g. after an intentional inner-solver
algorithmic change), commit the updated JSONs as a discrete numerical
behaviour change reviewable in PR. Do not silently regenerate to make a
failing test pass; investigate first.

Run from the repo root::

    uv run python tests/golden/generate.py

Pinned values are computed at inner tolerance ``1e-10``; the runner
asserts at ``atol=1e-8, rtol=1e-6`` to absorb BLAS-level numerical noise
across platforms while still catching real regressions.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sparho import (
    L1,
    ElasticNet,
    GroupL1,
    LogisticLoss,
    Problem,
    SquaredLoss,
    WeightedL1,
)
from sparho.adapters import (
    GroupLassoFista,
    SklearnElasticNet,
    SklearnLasso,
    SklearnLogisticRegression,
    SklearnWeightedLasso,
)
from sparho.solver import Solver
from sparho.testing import kkt_residual

GOLDEN_DIR = Path(__file__).resolve().parent
SOLVE_TOL = 1e-10


@dataclass(frozen=True, slots=True)
class GoldenSpec:
    """One regression-pinned problem."""

    name: str
    builder: Callable[[], tuple[Problem, Any]]  # returns (problem, hyperparam)
    solver_factory: Callable[[], Solver]


def _lasso_small() -> tuple[Problem, float]:
    rng = np.random.default_rng(0)
    n, p, k = 50, 20, 4
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:k] = rng.standard_normal(k) + 2.0
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    return Problem(SquaredLoss(), L1(), X, y), 0.05


def _elastic_net_small() -> tuple[Problem, float]:
    rng = np.random.default_rng(1)
    n, p = 60, 18
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[[1, 4, 9, 13]] = [2.0, -1.5, 1.0, -2.5]
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    return Problem(SquaredLoss(), ElasticNet(rho=0.7), X, y), 0.05


def _weighted_lasso_small() -> tuple[Problem, np.ndarray]:
    rng = np.random.default_rng(2)
    n, p = 50, 15
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[[0, 3, 7]] = [1.5, -2.0, 1.0]
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    alpha = np.full(p, 0.05, dtype=np.float64)
    return Problem(SquaredLoss(), WeightedL1(), X, y), alpha


def _group_lasso_small() -> tuple[Problem, float]:
    rng = np.random.default_rng(3)
    n, p = 80, 24
    X = rng.standard_normal((n, p))
    groups = tuple(tuple(range(3 * k, 3 * (k + 1))) for k in range(8))
    true_beta = np.zeros(p)
    true_beta[0:3] = [1.5, -1.0, 0.8]
    true_beta[3:6] = [-1.2, 0.9, 1.4]
    y = X @ true_beta + 0.05 * rng.standard_normal(n)
    return Problem(SquaredLoss(), GroupL1(groups=groups), X, y), 0.05


def _logreg_small() -> tuple[Problem, float]:
    rng = np.random.default_rng(4)
    n, p = 150, 20
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[[1, 4, 9]] = [2.0, -1.5, 1.0]
    logits = X @ true_beta
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = np.where(rng.uniform(size=n) < probs, 1.0, -1.0)
    return Problem(LogisticLoss(), L1(), X, y), 0.05


SPECS: tuple[GoldenSpec, ...] = (
    GoldenSpec("lasso_small", _lasso_small, SklearnLasso),
    GoldenSpec("elastic_net_small", _elastic_net_small, SklearnElasticNet),
    GoldenSpec("weighted_lasso_small", _weighted_lasso_small, SklearnWeightedLasso),
    GoldenSpec("group_lasso_small", _group_lasso_small, lambda: GroupLassoFista(max_iter=5000)),
    GoldenSpec("logreg_small", _logreg_small, SklearnLogisticRegression),
)


def _training_loss(problem: Problem, beta: np.ndarray) -> float:
    """Datafit value at ``β`` (no penalty term)."""
    if isinstance(problem.datafit, SquaredLoss):
        resid = problem.design @ beta - problem.target
        return float(resid @ resid) / (2.0 * problem.n_samples)
    if isinstance(problem.datafit, LogisticLoss):
        xb = problem.design @ beta
        return float(np.mean(np.logaddexp(0.0, -problem.target * xb)))
    raise TypeError(f"unsupported datafit: {type(problem.datafit).__name__}")


def _hp_to_json(hp: Any) -> Any:
    if isinstance(hp, np.ndarray):
        return {"kind": "vector", "values": hp.tolist()}
    return {"kind": "scalar", "value": float(hp)}


def _compute_fixture(spec: GoldenSpec) -> dict[str, Any]:
    problem, hp = spec.builder()
    solver = spec.solver_factory()
    result = solver(problem, hp, tol=SOLVE_TOL)
    coef = np.asarray(result.coef, dtype=np.float64)
    return {
        "name": spec.name,
        "datafit": type(problem.datafit).__name__,
        "penalty": type(problem.penalty).__name__,
        "solver": type(solver).__name__,
        "solve_tol": SOLVE_TOL,
        "n_samples": problem.n_samples,
        "n_features": problem.n_features,
        "hyperparam": _hp_to_json(hp),
        "active_set": sorted(int(j) for j in result.active_set),
        "coef": coef.tolist(),
        "coef_l2": float(np.linalg.norm(coef)),
        "coef_linf": float(np.max(np.abs(coef))),
        "training_loss": _training_loss(problem, coef),
        "kkt_residual": kkt_residual(problem, hp, coef),
    }


def main() -> None:
    for spec in SPECS:
        fixture = _compute_fixture(spec)
        path = GOLDEN_DIR / f"{spec.name}.json"
        path.write_text(json.dumps(fixture, indent=2) + "\n")
        print(
            f"wrote {path.name:30s}  "
            f"|β|₂={fixture['coef_l2']:.6f}  "
            f"loss={fixture['training_loss']:.6e}  "
            f"kkt={fixture['kkt_residual']:.3e}"
        )


if __name__ == "__main__":
    main()
