# Reproducibility

Bit-identical replay of a `sparho` run on a different machine — or three
months later on the same machine — requires three things to line up:
the **seed**, the **BLAS thread count**, and the **dependency versions**.
This page documents the discipline `sparho` uses to make that possible
and what you should do to consume it.

## TL;DR

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
  uv run python your_script.py
```

For tests and benchmarks, prefer the helper:

```python
from sparho.testing import pin_blas_threads

with pin_blas_threads(1):
    # everything inside this block runs single-threaded BLAS.
    result = hoag_search(...)
```

## Why BLAS threads matter

Multi-threaded BLAS (`OpenBLAS`, `MKL`, `Accelerate`) computes reductions
(dot products, matrix–matrix multiplies) by splitting the work across
threads and summing the partial results in the order the threads happen
to finish. Floating-point addition is not associative, so the final sum
is *not* bit-identical run-to-run — the high-order bits agree, the last
few mantissa bits drift. For `sparho`'s inner-solver tolerance regime
(`1e-8` to `1e-10`), that drift is enough to push the active set across
its threshold and flip the entire downstream search trajectory.

Single-threaded BLAS is deterministic: the reduction order is fixed,
the same inputs produce the same bits. The cost is wall-time
(BLAS-bound operations no longer scale across cores), but for the inner
solvers `sparho` targets — small-active-set Lasso/ElasticNet/Group —
the wall-time hit is modest.

## The four environment variables

Numpy and scipy read these on **first BLAS call**, not on every call:

| Variable                  | Backend                                       |
|---------------------------|-----------------------------------------------|
| `OMP_NUM_THREADS`         | OpenMP (used by OpenBLAS, MKL fallback)       |
| `MKL_NUM_THREADS`         | Intel MKL                                     |
| `OPENBLAS_NUM_THREADS`    | OpenBLAS                                      |
| `VECLIB_MAXIMUM_THREADS`  | Apple Accelerate (macOS)                      |
| `NUMEXPR_NUM_THREADS`     | NumExpr (transitive dep of pandas / xarray)   |
| `BLIS_NUM_THREADS`        | BLIS (some scientific Linux distros)          |

If your script imports numpy *before* the env vars are set, the
threadpool is already baked in. `sparho.testing.pin_blas_threads()`
handles this by additionally calling
[`threadpoolctl.threadpool_limits()`](https://github.com/joblib/threadpoolctl),
which retunes the *live* pool. Without `threadpoolctl`, only future
subprocesses see the change.

## Seed discipline

Every `sparho` API that consumes randomness exposes a `random_state`
keyword and is bit-identical at the same seed (under single-threaded
BLAS). The places randomness enters:

- `CrossVal.kfold(..., random_state=seed)` — shuffled fold splits.
- `Sure(sigma=..., random_state=seed)` — the FDMC probe `δ`.
- Inner-solver warm-start cache: deterministic given the outer-loop
  trajectory, so seeding `CrossVal` / `Sure` covers it.

The test suite asserts bit-equality of `SearchResult.best_hyperparam`
and `best_coef` at fixed seed in `tests/test_determinism.py` and across
the BLAS-threads × seed matrix in `tests/test_determinism_matrix.py`.

## Dependency versions

For exact reproducibility, pin the dependency closure. The repository
provides:

- `pyproject.toml` — the *floor* versions (`numpy>=1.24`, `scipy>=1.10`,
  `scikit-learn>=1.3`). The `ci-min-deps` job exercises these to guard
  against silent floor regressions.
- `uv.lock` — the *resolved* lockfile for the dev/test environment.
  Run `uv sync --extra dev` to materialize it.
- `requirements-bench.txt` — a `uv pip compile`-derived lockfile for
  the benchmark suite specifically (numpy + scipy + sklearn + celer +
  libsvmdata + matplotlib + pandas at known-good versions). Refresh
  quarterly with `uv pip compile pyproject.toml --extra bench -o
  requirements-bench.txt`.

## What `pin_blas_threads` does (and does not) guarantee

| Guarantee                                                | Mechanism                          |
|----------------------------------------------------------|------------------------------------|
| Future subprocesses see `n` threads                      | Env vars updated in `os.environ`   |
| BLAS calls inside this process see `n` threads           | `threadpoolctl.threadpool_limits`  |
| Env vars restored on context-manager exit                | `try/finally` saves prior values   |
| Bit-identical `np.dot` / `X.T @ y` across runs at `n=1`  | Single-threaded BLAS               |
| Bit-identical floats across BLAS backends (MKL vs OB)    | **No.** Different rounding paths.  |
| Bit-identical across CPU microarchitectures              | **No.** AVX-512 vs AVX2 differ.    |
| Bit-identical across numpy/scipy versions                | **No.** Backed by the lockfile.    |

For cross-backend reproducibility, pin both the BLAS backend (e.g.
install numpy from `numpy/openblas` vs `numpy/mkl-fft`) and the
version. For most academic-publication purposes, pinning the lockfile
and `OMP_NUM_THREADS=1` is enough — the active set, β\*, and search
trajectory will match. Last-bit numerical agreement on residuals is
rarely the meaningful invariant.

## Determinism audit matrix

`tests/test_determinism_matrix.py` re-runs the canonical
`grad_search` / `hoag_search` paths at three BLAS-thread counts
(`{1, 2, 4}`) × two seeds × two criteria (`CrossVal`, `Sure`). At
`n_threads=1` it asserts bit-equality of `best_hyperparam` and
`best_coef` across reruns. At `n_threads > 1` it asserts equality
*within a tight tolerance* — the test serves as both a regression
guard and a calibration of the multi-thread drift envelope.

Run locally with:

```bash
uv run pytest tests/test_determinism_matrix.py -v
```

## Benchmark provenance

Every run of `benchmarks/lasso_libsvm.py` emits a `provenance.json`
alongside its `results.json`. The provenance records CPU model, OS,
Python / numpy / scipy / sklearn / celer versions, the BLAS backend
resolved via `np.show_config()`, every BLAS env var at run time, and
the git SHA of the working tree. Reviewers reproducing a published
number should be able to diff their provenance against the one in the
paper artifact and immediately see what changed.

`benchmarks/render_tables.py` regenerates the Markdown tables in
`benchmarks/README.md` from the result JSONs, so the published numbers
never drift from the underlying measurements.
