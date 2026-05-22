"""Golden numerical-regression suite.

Each fixture in ``tests/golden/*.json`` pins ``(β*, loss, KKT residual)``
for a deterministic ``(datafit, penalty, solver, α)`` triple. This test
loads every fixture, re-builds the problem, re-solves at the same inner
tolerance, and asserts agreement.

Tolerances:
- ``coef``: atol=1e-8, rtol=1e-6 — tight enough to catch real
  algorithmic regressions, loose enough to survive BLAS jitter across
  platforms.
- ``training_loss``: atol=1e-10, rtol=1e-8 — much tighter; the loss is a
  scalar function of β and far less BLAS-sensitive.
- ``kkt_residual``: atol=1e-6 — qualitative, since the residual itself
  is a noise-amplified function near the optimum.

To intentionally update a fixture (e.g. after an inner-solver algorithm
change), rerun ``uv run python tests/golden/generate.py`` and commit
the diff as a discrete numerical-behaviour change.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sparho.adapters import (
    GroupLassoFista,
    SklearnElasticNet,
    SklearnLasso,
    SklearnLogisticRegression,
    SklearnWeightedLasso,
)
from sparho.solver import Solver
from sparho.testing import kkt_residual

# Load tests/golden/generate.py as an anonymous module — tests/ is not a
# package, so `from tests.golden.generate import ...` would require an
# `__init__.py` at the tests root that isn't there by convention.
_GENERATE_PATH = Path(__file__).resolve().parent / "golden" / "generate.py"
_spec = importlib.util.spec_from_file_location("_golden_generate", _GENERATE_PATH)
assert _spec is not None and _spec.loader is not None
_generate = importlib.util.module_from_spec(_spec)
# Register in sys.modules so @dataclass(slots=True) can resolve the module
# back through `cls.__module__`; otherwise dataclass introspection blows up.
import sys  # noqa: E402

sys.modules[_spec.name] = _generate
_spec.loader.exec_module(_generate)
SPECS = _generate.SPECS
_training_loss = _generate._training_loss

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
FIXTURE_FILES = sorted(GOLDEN_DIR.glob("*.json"))

assert FIXTURE_FILES, "no golden fixtures found — run tests/golden/generate.py"

# Map fixture name → (builder, solver_factory) by reusing the same spec
# objects the generator uses. Keeps the runner and generator in lockstep.
_SPECS_BY_NAME = {s.name: s for s in SPECS}


_SOLVER_REGISTRY: dict[str, type[Solver]] = {
    "SklearnLasso": SklearnLasso,
    "SklearnElasticNet": SklearnElasticNet,
    "SklearnWeightedLasso": SklearnWeightedLasso,
    "SklearnLogisticRegression": SklearnLogisticRegression,
    "GroupLassoFista": GroupLassoFista,
}


def _hp_from_json(blob: dict[str, Any]) -> Any:
    if blob["kind"] == "vector":
        return np.asarray(blob["values"], dtype=np.float64)
    return float(blob["value"])


@pytest.mark.parametrize(
    "fixture_path", FIXTURE_FILES, ids=lambda p: p.stem
)
def test_golden_regression(fixture_path: Path) -> None:
    """Re-solve the pinned problem and assert agreement on β, loss, KKT residual."""
    fixture = json.loads(fixture_path.read_text())
    name = fixture["name"]

    spec = _SPECS_BY_NAME.get(name)
    if spec is None:
        pytest.skip(
            f"fixture {name!r} has no matching GoldenSpec — regenerate fixtures"
        )

    problem, hp = spec.builder()
    solver = spec.solver_factory()
    result = solver(problem, hp, tol=fixture["solve_tol"])

    # 1. β agreement (tight, but BLAS-tolerant).
    coef = np.asarray(result.coef, dtype=np.float64)
    expected_coef = np.asarray(fixture["coef"], dtype=np.float64)
    assert coef.shape == expected_coef.shape, (
        f"coef shape drift: {coef.shape} vs {expected_coef.shape}"
    )
    np.testing.assert_allclose(
        coef, expected_coef, atol=1e-8, rtol=1e-6, err_msg=f"{name}: coef regression"
    )

    # 2. Loss agreement (very tight — scalar, BLAS-insensitive).
    loss = _training_loss(problem, coef)
    np.testing.assert_allclose(
        loss,
        fixture["training_loss"],
        atol=1e-10,
        rtol=1e-8,
        err_msg=f"{name}: training_loss regression",
    )

    # 3. KKT residual (qualitative — re-solve produces a similarly small residual).
    res = kkt_residual(problem, hp, coef)
    assert res < max(1e-5, 10.0 * fixture["kkt_residual"]), (
        f"{name}: KKT residual {res:.3e} is much larger than pinned "
        f"{fixture['kkt_residual']:.3e}"
    )

    # 4. Active set agreement (sparsity pattern is the strongest invariant
    #    against algorithmic drift; coordinate values can wobble, but a
    #    feature jumping in/out of the support is always a real change).
    active = sorted(int(j) for j in result.active_set)
    assert active == fixture["active_set"], (
        f"{name}: active-set drift\n  was {fixture['active_set']}\n  got {active}"
    )
