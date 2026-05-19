#!/usr/bin/env python
"""Spike: does Tikhonov-ridged CG stabilize ``implicit_forward`` on rcv1?

For each dataset and each ``ridge`` value, run ``grad_search`` with
``hypergrad = partial(implicit_forward, ridge=ε)`` and record:

- did the search complete (no NaN α, no crash)?
- final α*, final CV-MSE, wall time, outer iters, stall iters
  (`history` entries where `grad_norm` is non-finite or zero)

Two purposes:

1. **leukemia** — bias check. Reference is ``ridge=0`` (no perturbation).
   Compare α* across `ridge ∈ {1e-12 ... 1e-6}` to verify the auto-default
   (`1e-10 · trace/|A|`) sits in the "no detectable bias" range.
2. **rcv1.binary** — convergence check. Verify the search actually
   completes for some ε > 0, and find the smallest ε that does so.

Usage::

    uv run python benchmarks/spike_cg_stability.py
    uv run python benchmarks/spike_cg_stability.py --datasets leukemia
    uv run python benchmarks/spike_cg_stability.py --skip-rcv1
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from functools import partial

import numpy as np
import scipy.sparse as sp
from libsvmdata import fetch_libsvm
from sparho import (
    L1,
    CrossVal,
    Problem,
    SquaredLoss,
    hoag_search,
)
from sparho.adapters import SklearnLasso
from sparho.hypergrad import implicit_forward

DATASETS: dict[str, dict] = {
    "leukemia": {"sparse_required": False, "alpha0": 1e-1, "n_iter": 30},
    "rcv1.binary": {"sparse_required": True, "alpha0": 1e-2, "n_iter": 5},
}

# Per-dataset ridge sweeps. Leukemia is well-conditioned so ridge=0 still
# completes and serves as the bias reference. rcv1 is the pathological case;
# we already know ridge=0 burns minutes on each outer iter without converging,
# so we skip it and the equally-bad ridge=1e-12 to keep the spike under a few
# minutes.
RIDGE_SETTINGS_LEUKEMIA = [
    ("auto", None),
    ("0", 0.0),
    ("1e-12", 1e-12),
    ("1e-10", 1e-10),
    ("1e-8", 1e-8),
    ("1e-6", 1e-6),
]
RIDGE_SETTINGS_RCV1 = [
    ("auto", None),
    ("1e-10", 1e-10),
    ("1e-8", 1e-8),
    ("1e-6", 1e-6),
]

RIDGE_SETTINGS_BY_DATASET = {
    "leukemia": RIDGE_SETTINGS_LEUKEMIA,
    "rcv1.binary": RIDGE_SETTINGS_RCV1,
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


def _run(X, y, *, ridge, hp0: float, n_iter: int) -> dict:
    problem = Problem(SquaredLoss(), L1(), X, y)
    cv = CrossVal.kfold(X.shape[0], k=5, shuffle=False)
    solver = SklearnLasso(tol=1e-6, max_iter=10_000)
    hg_fn = partial(implicit_forward, ridge=ridge) if ridge != "default" else implicit_forward

    cg_failures = 0
    stalls = 0
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            result = hoag_search(
                problem,
                hp0=hp0,
                solver=solver,
                criterion=cv,
                hypergrad=hg_fn,
                n_iter=n_iter,
                inner_tol=1e-6,
                outer_tol=1e-4,
            )
        elapsed = time.perf_counter() - t0
        for w in caught:
            msg = str(w.message)
            if "CG failed" in msg:
                cg_failures += 1
            if "non-finite hypergradient" in msg or "non-finite gradient" in msg:
                stalls += 1
        final_mse = cv.value(problem, result.best_hyperparam, solver)
        return {
            "ok": True,
            "alpha": float(result.best_hyperparam),
            "mse": float(final_mse),
            "elapsed": float(elapsed),
            "outer_iters": int(result.n_iter),
            "cg_failures": cg_failures,
            "stalls": stalls,
            "error": None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "ok": False,
            "alpha": float("nan"),
            "mse": float("nan"),
            "elapsed": float(elapsed),
            "outer_iters": 0,
            "cg_failures": cg_failures,
            "stalls": stalls,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _print_header(ds: str) -> None:
    print(f"\n### {ds}\n", flush=True)
    print("| ridge | ok | α* | MSE | time | outer | cg_fail | stalls | error |", flush=True)
    print("|---|---|---|---|---|---|---|---|---|", flush=True)


def _print_row(label: str, r: dict) -> None:
    if r["ok"]:
        print(
            f"| {label} | ✓ | {r['alpha']:.4g} | {r['mse']:.4g} "
            f"| {r['elapsed']:.2f}s | {r['outer_iters']} "
            f"| {r['cg_failures']} | {r['stalls']} | — |",
            flush=True,
        )
    else:
        print(
            f"| {label} | ✗ | — | — | {r['elapsed']:.2f}s | {r['outer_iters']} "
            f"| {r['cg_failures']} | {r['stalls']} | `{r['error']}` |",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--skip-rcv1", action="store_true")
    args = parser.parse_args()

    datasets = list(args.datasets)
    if args.skip_rcv1 and "rcv1.binary" in datasets:
        datasets.remove("rcv1.binary")

    for ds in datasets:
        meta = DATASETS.get(ds)
        if meta is None:
            print(f"Skipping unknown dataset: {ds}", file=sys.stderr, flush=True)
            continue
        print(f"\nLoading {ds} …", file=sys.stderr, flush=True)
        X, y = _load_dataset(ds, must_be_sparse=meta["sparse_required"])
        _print_header(ds)
        for label, eps in RIDGE_SETTINGS_BY_DATASET[ds]:
            print(f"  ridge={label} …", file=sys.stderr, flush=True)
            r = _run(X, y, ridge=eps, hp0=meta["alpha0"], n_iter=meta["n_iter"])
            _print_row(label, r)

    return 0


if __name__ == "__main__":
    sys.exit(main())
