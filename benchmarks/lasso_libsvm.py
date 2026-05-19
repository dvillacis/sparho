#!/usr/bin/env python
"""Benchmark ``sparho`` vs ``sklearn.LassoCV`` on libsvm Lasso problems.

Each library tunes the Lasso ``α`` by minimizing 5-fold CV-MSE. Wall-time is
measured from "data loaded" to "α* selected". A small markdown table prints
to stdout at the end.

Usage::

    python benchmarks/lasso_libsvm.py
    python benchmarks/lasso_libsvm.py --datasets leukemia
    python benchmarks/lasso_libsvm.py --quick           # fewer outer iters

Datasets (downloaded once via libsvmdata, cached on disk):

- ``breast-cancer``  — 683 × 10 (small, < 1 s)
- ``leukemia``       — 72 × 7129 (high-dim, n ≪ p)
- ``rcv1.binary``    — 20242 × 47236 (sparse; only run with ``--rcv1``)

``rcv1.binary`` is gated behind ``--rcv1`` because the download is ~ 300 MB
and the full bench takes several minutes; the script also asserts the
sparse-X path is preserved end-to-end (no densification) on that dataset.

Note: ``sparse-ho`` is not on PyPI. If installed from source, add ``--with-sparse-ho``
to include a third comparison column (TODO at v0.1 — currently only the
sklearn baseline is included).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import scipy.sparse as sp
from libsvmdata import fetch_libsvm
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sparho import (
    L1,
    CrossVal,
    Problem,
    SquaredLoss,
    hoag_search,
)
from sparho.adapters import SklearnLasso

DATASETS: dict[str, dict] = {
    "breast-cancer": {
        "sparse_required": False,
        "alpha_grid_log": (-3, 1, 20),
    },
    "leukemia": {
        "sparse_required": False,
        "alpha_grid_log": (-3, 1, 20),
    },
    "rcv1.binary": {
        "sparse_required": True,
        "alpha_grid_log": (-4, 0, 15),
    },
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
        # Small / medium datasets — densify so sklearn's Lasso path is comparable.
        if sp.issparse(X):
            X = np.asarray(X.toarray(), dtype=np.float64)
        else:
            X = np.asarray(X, dtype=np.float64)
    return X, y


def _run_sparho(
    X,
    y,
    *,
    hp0: float,
    n_iter: int,
    sparse_required: bool,
    warm_start: bool,
):
    problem = Problem(SquaredLoss(), L1(), X, y)
    cv = CrossVal.kfold(X.shape[0], k=5, shuffle=False, warm_start=warm_start)
    # SklearnLasso default tol is overridden per-iter by hoag_search.
    solver = SklearnLasso(tol=1e-6, max_iter=50_000)
    if sparse_required:
        assert sp.issparse(problem.design), "sparho: design must remain sparse for rcv1"
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
        outer_tol=1e-6,
    )
    elapsed = time.perf_counter() - t0
    final_mse = cv.value(problem, result.best_hyperparam, solver)
    return {
        "alpha": float(result.best_hyperparam),
        "mse": float(final_mse),
        "elapsed": float(elapsed),
        "iters": int(result.n_iter),
    }


def _run_lasso_cv(X, y, *, alpha_grid: np.ndarray):
    t0 = time.perf_counter()
    est = LassoCV(
        alphas=alpha_grid,
        cv=KFold(5, shuffle=False),
        fit_intercept=False,
        tol=1e-6,
        max_iter=10_000,
    )
    est.fit(X, y)
    elapsed = time.perf_counter() - t0
    i_best = int(np.argmin(est.mse_path_.mean(axis=1)))
    return {
        "alpha": float(est.alphas_[i_best]),
        "mse": float(est.mse_path_[i_best].mean()),
        "elapsed": float(elapsed),
        "iters": int(len(alpha_grid)),
    }


def _print_row(ds: str, shape: tuple[int, int], s: dict, lcv: dict) -> str:
    speedup = lcv["elapsed"] / s["elapsed"] if s["elapsed"] > 0 else float("inf")
    return (
        f"| `{ds}` | {shape[0]}×{shape[1]} "
        f"| {s['alpha']:.4g} | {s['mse']:.4g} | {s['elapsed']:.2f}s | {s['iters']} "
        f"| {lcv['alpha']:.4g} | {lcv['mse']:.4g} | {lcv['elapsed']:.2f}s | {lcv['iters']} "
        f"| **{speedup:.2f}×** |"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["breast-cancer", "leukemia"],
        help="libsvm dataset names",
    )
    parser.add_argument(
        "--rcv1",
        action="store_true",
        help="include rcv1.binary (large download, slow)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run with fewer outer iters (CI smoke)",
    )
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="disable inner-solver warm-start across outer iterations "
        "(use to reproduce the v0.1 baseline)",
    )
    args = parser.parse_args()
    warm_start = not args.cold_start

    datasets = list(args.datasets)
    if args.rcv1 and "rcv1.binary" not in datasets:
        datasets.append("rcv1.binary")

    n_iter = 10 if args.quick else 30

    mode = "cold-start" if args.cold_start else "warm-start"
    print(f"sparho mode: {mode}  (hoag_search + exponential inner-tol decrease)")
    table_rows: list[str] = []
    for ds in datasets:
        meta = DATASETS.get(ds)
        if meta is None:
            print(f"Skipping unknown dataset: {ds}", file=sys.stderr)
            continue
        print(f"\n=== {ds} ===")
        X, y = _load_dataset(ds, must_be_sparse=meta["sparse_required"])
        nnz = X.nnz if sp.issparse(X) else "dense"
        print(f"  shape: {X.shape}, sparse: {sp.issparse(X)}, nnz: {nnz}")

        alpha_grid = np.logspace(*meta["alpha_grid_log"])
        hp0 = float(alpha_grid[len(alpha_grid) // 2])

        s = _run_sparho(
            X,
            y,
            hp0=hp0,
            n_iter=n_iter,
            sparse_required=meta["sparse_required"],
            warm_start=warm_start,
        )
        lcv = _run_lasso_cv(X, y, alpha_grid=alpha_grid)

        print(
            f"  sparho:   α*={s['alpha']:.4g}  MSE={s['mse']:.4g}  "
            f"iters={s['iters']:3d}  {s['elapsed']:6.2f}s"
        )
        print(
            f"  LassoCV:  α*={lcv['alpha']:.4g}  MSE={lcv['mse']:.4g}  "
            f"grid={lcv['iters']:3d}  {lcv['elapsed']:6.2f}s"
        )
        print(f"  speedup:  {lcv['elapsed'] / s['elapsed']:.2f}×")
        table_rows.append(_print_row(ds, X.shape, s, lcv))

    print("\n## Results\n")
    print(
        "| dataset | shape "
        "| sparho α* | sparho MSE | sparho time | sparho iters "
        "| LassoCV α* | LassoCV MSE | LassoCV time | LassoCV grid "
        "| speedup |"
    )
    print(
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for row in table_rows:
        print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
