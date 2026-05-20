#!/usr/bin/env python
"""Benchmark ``sparho`` vs ``sklearn.LassoCV`` on libsvm Lasso problems.

Each library tunes the Lasso ``α`` by minimizing 5-fold CV-MSE. Wall-time is
measured from "data loaded" to "α* selected". A small markdown table prints
to stdout at the end.

Usage::

    python benchmarks/lasso_libsvm.py
    python benchmarks/lasso_libsvm.py --datasets leukemia
    python benchmarks/lasso_libsvm.py --quick           # fewer outer iters
    python benchmarks/lasso_libsvm.py --repeat 5 --cooldown 2  # reproducibility mode

Datasets (downloaded once via libsvmdata, cached on disk):

- ``breast-cancer``  — 683 × 10 (small, < 1 s)
- ``leukemia``       — 72 × 7129 (high-dim, n ≪ p)
- ``rcv1.binary``    — 20242 × 47236 (sparse; only run with ``--rcv1``)

``rcv1.binary`` is gated behind ``--rcv1`` because the download is ~ 300 MB
and the full bench takes several minutes; the script also asserts the
sparse-X path is preserved end-to-end (no densification) on that dataset.

Reproducibility mode (``--repeat N``) runs each timed section N times and
reports the median wall + relative spread ``(max - min) / median``.
``--warmup K`` (default 1 when ``--repeat > 1``) drops the first K samples
to amortize cold-cache effects. ``--cooldown S`` sleeps S seconds between
iters to let macOS thermal state settle. sparho and ``LassoCV`` are
interleaved per iteration so thermal load is shared fairly.

Note: ``sparse-ho`` is not on PyPI. If installed from source, add ``--with-sparse-ho``
to include a third comparison column (TODO at v0.1 — currently only the
sklearn baseline is included).
"""

from __future__ import annotations

import argparse
import gc
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
from sparho.adapters import CelerLasso, SklearnLasso

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
    solver_name: str,
):
    problem = Problem(SquaredLoss(), L1(), X, y)
    cv = CrossVal.kfold(X.shape[0], k=5, shuffle=False, warm_start=warm_start)
    # Inner-solver tol is overridden per-iter by hoag_search.
    if solver_name == "celer":
        solver = CelerLasso(tol=1e-6, max_iter=100)
    elif solver_name == "sklearn":
        solver = SklearnLasso(tol=1e-6, max_iter=50_000)
    else:
        raise ValueError(f"unknown solver: {solver_name!r}")
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


def _aggregate(samples: list[dict]) -> dict:
    """Combine N single-run dicts into median + spread."""
    times = [s["elapsed"] for s in samples]
    median = float(np.median(times))
    spread = (max(times) - min(times)) / median if median > 0 and len(times) > 1 else 0.0
    return {
        "alpha": samples[0]["alpha"],  # deterministic across runs
        "mse": samples[0]["mse"],
        "iters": samples[0]["iters"],
        "elapsed": median,
        "elapsed_min": float(min(times)),
        "elapsed_max": float(max(times)),
        "elapsed_spread": float(spread),
        "n_samples": len(times),
    }


def _fmt_time_cell(d: dict) -> str:
    if d["n_samples"] <= 1:
        return f"{d['elapsed']:.2f}s"
    return f"{d['elapsed']:.2f}s ±{100 * d['elapsed_spread']:.1f}%"


def _print_row(ds: str, shape: tuple[int, int], s: dict, lcv: dict) -> str:
    speedup = lcv["elapsed"] / s["elapsed"] if s["elapsed"] > 0 else float("inf")
    return (
        f"| `{ds}` | {shape[0]}×{shape[1]} "
        f"| {s['alpha']:.4g} | {s['mse']:.4g} | {_fmt_time_cell(s)} | {s['iters']} "
        f"| {lcv['alpha']:.4g} | {lcv['mse']:.4g} | {_fmt_time_cell(lcv)} | {lcv['iters']} "
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
    parser.add_argument(
        "--solver",
        choices=("sklearn", "celer"),
        default="sklearn",
        help="inner solver for sparho's hoag_search (default: sklearn)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run each timed section N times and report median + spread (default: 1)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="drop the first K samples (default: 0 if --repeat=1, else 1)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help="seconds to sleep between iters to let thermal state settle (default: 0)",
    )
    args = parser.parse_args()
    warm_start = not args.cold_start

    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    warmup = args.warmup if args.warmup is not None else (1 if args.repeat > 1 else 0)
    if warmup < 0 or warmup >= args.repeat:
        parser.error(f"--warmup must be in [0, {args.repeat - 1}]")

    datasets = list(args.datasets)
    if args.rcv1 and "rcv1.binary" not in datasets:
        datasets.append("rcv1.binary")

    n_iter = 10 if args.quick else 30

    mode = "cold-start" if args.cold_start else "warm-start"
    repeat_str = (
        "single-run"
        if args.repeat == 1
        else f"repeat={args.repeat} warmup={warmup} cooldown={args.cooldown}s"
    )
    print(
        f"sparho mode: {mode}  inner-solver: {args.solver}  {repeat_str}  "
        f"(hoag_search + exponential inner-tol decrease)"
    )
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

        sparho_samples: list[dict] = []
        lcv_samples: list[dict] = []
        for rep in range(args.repeat):
            if rep > 0 and args.cooldown > 0:
                time.sleep(args.cooldown)
            gc.collect()
            s_one = _run_sparho(
                X,
                y,
                hp0=hp0,
                n_iter=n_iter,
                sparse_required=meta["sparse_required"],
                warm_start=warm_start,
                solver_name=args.solver,
            )
            sparho_samples.append(s_one)
            if args.cooldown > 0:
                time.sleep(args.cooldown)
            gc.collect()
            lcv_one = _run_lasso_cv(X, y, alpha_grid=alpha_grid)
            lcv_samples.append(lcv_one)
            if args.repeat > 1:
                marker = " (warmup)" if rep < warmup else ""
                print(
                    f"  rep {rep + 1:2d}/{args.repeat}: "
                    f"sparho {s_one['elapsed']:7.2f}s   "
                    f"LassoCV {lcv_one['elapsed']:7.2f}s{marker}"
                )

        s = _aggregate(sparho_samples[warmup:])
        lcv = _aggregate(lcv_samples[warmup:])

        spread_tag = ""
        if s["n_samples"] > 1:
            spread_tag = (
                f"  spread[sparho]={100 * s['elapsed_spread']:.1f}%  "
                f"spread[LassoCV]={100 * lcv['elapsed_spread']:.1f}%"
            )
        print(
            f"  sparho:   α*={s['alpha']:.4g}  MSE={s['mse']:.4g}  "
            f"iters={s['iters']:3d}  {_fmt_time_cell(s)}"
        )
        print(
            f"  LassoCV:  α*={lcv['alpha']:.4g}  MSE={lcv['mse']:.4g}  "
            f"grid={lcv['iters']:3d}  {_fmt_time_cell(lcv)}"
        )
        print(f"  speedup:  {lcv['elapsed'] / s['elapsed']:.2f}×{spread_tag}")
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
