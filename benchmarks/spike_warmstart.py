#!/usr/bin/env python
"""Spike: does warm-starting the inner Lasso across outer iterations close the gap?

Diagnostic-only. Compares two ``Solver`` wrappers on the same outer search:

- ``ColdLasso``  — exactly what ``SklearnLasso`` does today: a fresh
  ``Lasso`` per outer iteration, β starts at zero every time.
- ``WarmLasso``  — keeps a per-fold ``Lasso(warm_start=True)`` cache keyed
  by the train-target bytes, so β*_prev seeds the next inner solve.

Both run the same ``grad_search`` config; we report wall time, total inner
coordinate-descent iterations summed across all fold solves, and final α / CV-MSE
to confirm the two paths converge to the same answer.

This script does NOT modify library code — the warm-start logic is local to
``WarmLasso`` here. If the speedup hypothesis holds, we promote it into a
proper API change for v0.2.

Usage::

    uv run python benchmarks/spike_warmstart.py
    uv run python benchmarks/spike_warmstart.py --datasets leukemia
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from libsvmdata import fetch_libsvm
from sklearn.linear_model import Lasso
from sparho import (
    L1,
    CrossVal,
    Problem,
    SquaredLoss,
    hoag_search,
)
from sparho.adapters._common import active_set_of, as_scalar
from sparho.state import SolverResult

DATASETS: dict[str, dict] = {
    "breast-cancer": {"sparse_required": False, "alpha0": 1e-1},
    "leukemia": {"sparse_required": False, "alpha0": 1e-1},
    "rcv1.binary": {"sparse_required": True, "alpha0": 1e-2},
}


def _load_dataset(name: str, *, must_be_sparse: bool):
    X, y = fetch_libsvm(name)
    y = np.asarray(y, dtype=np.float64)
    if must_be_sparse:
        if not sp.issparse(X):
            raise RuntimeError(f"{name}: expected sparse X")
        if X.format != "csc":
            X = X.tocsc()
    else:
        if sp.issparse(X):
            X = np.asarray(X.toarray(), dtype=np.float64)
        else:
            X = np.asarray(X, dtype=np.float64)
    return X, y


def _fold_key(problem: Problem) -> bytes:
    """Stable per-fold key.

    ``CrossVal`` constructs a fresh ``Problem`` per fold per outer iter via
    ``dataclasses.replace``, but ``y[idx_train]`` has identical contents
    across outer iters for a given fold — hash those bytes.
    """
    y = problem.target
    return bytes(np.asarray(y, dtype=np.float64).tobytes())


@dataclass
class ColdLasso:
    """Stateless cold-start wrapper — equivalent to the current SklearnLasso."""

    tol: float = 1e-6
    max_iter: int = 10_000
    total_inner_iters: int = 0
    n_calls: int = 0

    def __call__(self, problem: Problem, hyperparam: Any, /) -> SolverResult:
        alpha = as_scalar(hyperparam)
        est = Lasso(
            alpha=alpha,
            fit_intercept=False,
            tol=self.tol,
            max_iter=self.max_iter,
            selection="cyclic",
        )
        est.fit(problem.design, problem.target)
        coef = np.asarray(est.coef_, dtype=np.float64)
        self.total_inner_iters += int(est.n_iter_)
        self.n_calls += 1
        return SolverResult(
            coef=coef,
            active_set=active_set_of(coef),
            dual_gap=float(est.dual_gap_),
            n_iter=int(est.n_iter_),
        )


@dataclass
class WarmLasso:
    """Per-fold warm-started wrapper.

    Holds a dict of fold-key → ``Lasso(warm_start=True)`` estimator. On each
    call, looks up the estimator for this fold (creating it if missing),
    mutates ``alpha`` to the current hyperparameter, and refits. sklearn
    reuses ``self.coef_`` as the starting point when ``warm_start=True``.
    """

    tol: float = 1e-6
    max_iter: int = 10_000
    _cache: dict[bytes, Lasso] = field(default_factory=dict)
    total_inner_iters: int = 0
    n_calls: int = 0

    def __call__(self, problem: Problem, hyperparam: Any, /) -> SolverResult:
        alpha = as_scalar(hyperparam)
        key = _fold_key(problem)
        est = self._cache.get(key)
        if est is None:
            est = Lasso(
                alpha=alpha,
                fit_intercept=False,
                tol=self.tol,
                max_iter=self.max_iter,
                selection="cyclic",
                warm_start=True,
            )
            self._cache[key] = est
        else:
            est.alpha = alpha
        est.fit(problem.design, problem.target)
        coef = np.asarray(est.coef_, dtype=np.float64)
        self.total_inner_iters += int(est.n_iter_)
        self.n_calls += 1
        return SolverResult(
            coef=coef,
            active_set=active_set_of(coef),
            dual_gap=float(est.dual_gap_),
            n_iter=int(est.n_iter_),
        )


def _run(X, y, *, solver_factory, hp0: float, n_iter: int):
    problem = Problem(SquaredLoss(), L1(), X, y)
    cv = CrossVal.kfold(X.shape[0], k=5, shuffle=False)
    solver = solver_factory()
    t0 = time.perf_counter()
    result = hoag_search(
        problem,
        hp0=hp0,
        solver=solver,
        criterion=cv,
        n_iter=n_iter,
        inner_tol=1e-6,
        inner_tol_initial=1e-2,
        tolerance_decrease="exponential",
        outer_tol=1e-4,
    )
    elapsed = time.perf_counter() - t0
    final_mse = cv.value(problem, result.best_hyperparam, solver)
    return {
        "alpha": float(result.best_hyperparam),
        "mse": float(final_mse),
        "elapsed": float(elapsed),
        "outer_iters": int(result.n_iter),
        "inner_iters": int(solver.total_inner_iters),
        "n_calls": int(solver.n_calls),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", nargs="+", default=["breast-cancer", "leukemia"])
    parser.add_argument("--n-iter", type=int, default=30, help="outer iterations")
    args = parser.parse_args()

    header = (
        f"{'dataset':<16}  {'variant':<6}  {'time':>8}  {'outer':>5}  "
        f"{'inner':>7}  {'calls':>5}  {'alpha':>9}  {'mse':>9}"
    )
    print(header)
    print("-" * 88)

    for ds in args.datasets:
        meta = DATASETS.get(ds)
        if meta is None:
            print(f"Skipping unknown dataset: {ds}", file=sys.stderr)
            continue
        X, y = _load_dataset(ds, must_be_sparse=meta["sparse_required"])

        cold = _run(X, y, solver_factory=ColdLasso, hp0=meta["alpha0"], n_iter=args.n_iter)
        warm = _run(X, y, solver_factory=WarmLasso, hp0=meta["alpha0"], n_iter=args.n_iter)

        for label, r in [("cold", cold), ("warm", warm)]:
            print(
                f"{ds:<16}  {label:<6}  {r['elapsed']:>7.2f}s  "
                f"{r['outer_iters']:>5d}  {r['inner_iters']:>7d}  {r['n_calls']:>5d}  "
                f"{r['alpha']:>9.4g}  {r['mse']:>9.4g}"
            )
        speedup = cold["elapsed"] / warm["elapsed"] if warm["elapsed"] > 0 else float("inf")
        cold_inner, warm_inner = cold["inner_iters"], warm["inner_iters"]
        iter_ratio = cold_inner / warm_inner if warm_inner > 0 else float("inf")
        d_alpha = abs(cold["alpha"] - warm["alpha"]) / max(abs(cold["alpha"]), 1e-30)
        d_mse = abs(cold["mse"] - warm["mse"]) / max(abs(cold["mse"]), 1e-30)
        print(
            f"  → warm/cold: {speedup:.2f}× wall, "
            f"{iter_ratio:.2f}× inner iters, "
            f"|Δα|/α = {d_alpha:.2e}, |ΔMSE|/MSE = {d_mse:.2e}"
        )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
