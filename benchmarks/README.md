# Benchmarks

Sparho's v0.1 benchmarks compare against `sklearn.linear_model.LassoCV` —
the standard grid-search baseline that sparse-ho refugees would have used
before. The bench is intentionally narrow at v0.1: prove that the gradient-
based outer loop produces the right answer on real data and is in the same
ballpark as a hand-tuned grid.

## Running

```bash
uv sync --extra dev --extra bench
uv run python benchmarks/lasso_libsvm.py                 # full run, ~ 30 s
uv run python benchmarks/lasso_libsvm.py --quick         # CI smoke, fewer outer iters
uv run python benchmarks/lasso_libsvm.py --rcv1          # add rcv1.binary (slow)
```

Datasets are downloaded once via `libsvmdata` and cached in
`~/scikit_learn_data/`. The bench script asserts that `rcv1.binary` stays
in CSC format end-to-end (no accidental densification) — the assertion fires
if a future regression breaks the sparse-X path.

## v0.1 numbers

Hardware: Apple M-series, single-threaded sklearn / BLAS. Methodology: 5-fold
CV, `α` grid of 20 points (`logspace(-3, 1)`) for `LassoCV`; sparho's
`grad_search` with `LineSearch(initial_step=0.5)` and `n_iter=30`,
`tol=1e-4`. Inner solver: `SklearnLasso(tol=1e-6)` for both. (Note: in v0.2,
`LineSearch` was removed and the recommended path is `hoag_search` — see
"v0.2 numbers" below.)

| dataset | shape | sparho α* | sparho MSE | sparho time | sparho iters | LassoCV α* | LassoCV MSE | LassoCV time | LassoCV grid |
|---|---|---|---|---|---|---|---|---|---|
| `breast-cancer` | 683×10 | 0.00108 | 0.508 | 0.24 s | 15 | 0.001 | 0.508 | 0.01 s | 20 |
| `leukemia` | 38×7129 | 0.107 | 0.435 | 23.1 s | 30 | 0.0785 | 0.433 | 30.1 s | 20 |
| `rcv1.binary` | 20242×47236 sparse | **2.1·10⁻⁵** | **0.194** | 432.8 s | 14 | 1·10⁻⁴ (grid floor) | 0.225 | 12.0 s | 15 |

**Reading the table**:
- **breast-cancer (n ≫ p)** — `LassoCV` wins by ~ 20×. The inner Lasso fit
  is too cheap for the gradient loop's per-iter overhead to pay off; this is
  exactly the regime grid search was designed for. Both finish in well
  under a second.
- **leukemia (n ≪ p)** — sparho is **1.3× faster** than `LassoCV` and
  finds a held-out MSE within 0.5 % of the grid optimum. The α difference
  reflects sparho finding the continuous optimum vs `LassoCV`'s grid-snap.
- **rcv1.binary** — sparho finds a **better held-out MSE** than `LassoCV`
  (0.194 vs 0.225) by driving α below `LassoCV`'s grid floor of 1·10⁻⁴ —
  implicit differentiation lets the outer loop walk a continuous α path
  rather than a discrete grid. The wall-time penalty (433 s vs 12 s) is
  not from the hypergradient solve — the v0.1 ridge stabilization (see
  next section) eliminates CG divergence — but from the inner Lasso
  cold-starting every outer iter at very small α (large active set,
  many CD passes per fit). Warm-starting (v0.2 follow-on, spike already
  measured at ~ 2× on dense problems) is the obvious next gain.

**Takeaways**:
- The headline `Nx faster than LassoCV` claim **does not hold at v0.1**.
- What sparho does ship at v0.1: a correct gradient-based outer loop with
  vector-α support (which `LassoCV` doesn't have), implicit-diff
  hypergradients, a clean Protocol-based API, and Rust kernels for the
  hot paths.
- The perf story is a v0.2 deliverable; see below.

## v0.2 numbers — warm-start + HOAG

The inner solver is warm-started across outer iterations
(`CrossVal(warm_start=True)`) and the outer loop is HOAG (Pedregosa 2016):
adaptive step size from a Lipschitz proxy, ``+C·tol`` slack in the
acceptance test that absorbs criterion-value noise from approximate inner
solves, and an exponentially decreasing inner-tolerance schedule.

| dataset | shape | sparho α* | sparho MSE | sparho time | iters | LassoCV α* | LassoCV MSE | LassoCV time | grid | sparho speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| `breast-cancer` | 683×10 | 3.1·10⁻³ | 0.508 | 0.11 s | 30 | 1·10⁻³ | 0.508 | 0.01 s | 20 | 0.1× (overhead-bound) |
| `leukemia` | 38×7129 | 0.117 | 0.436 | 1.01 s | 30 | 0.0785 | 0.433 | 8.72 s | 20 | **8.6×** |
| `rcv1.binary` | 20242×47236 sparse | **2.1·10⁻⁵** | **0.194** | 177.7 s | 30 | 1·10⁻⁴ (grid floor) | 0.225 | 11.9 s | 15 | 0.07× (quality win) |

**Reading the v0.2 table**:
- **`leukemia`** — sparho is **8.6× faster** than `LassoCV`, an order of
  magnitude better than v0.1's 1.3× (warm-start cuts inner-solver cost
  per outer iter; HOAG keeps the outer steps efficient without
  Armijo's trial-evaluation overhead).
- **`breast-cancer`** — still overhead-bound; both finish in well under a
  second. HOAG matches `LassoCV`'s MSE to 0.02 %.
- **`rcv1.binary`** — same quality story as v0.1: sparho finds an MSE
  14 % better than `LassoCV`'s grid optimum by walking α an order of
  magnitude below the grid floor. Wall time dropped 2.4× (v0.1: 433 s →
  v0.2: 178 s) but `LassoCV` is still faster in pure wall time because
  the inner-solver work at small α is irreducible.

## What changed at v0.1.0 — hypergradient-CG stability

The earlier "rcv1 does not converge" failure mode is fixed in v0.1.
`sparho.hypergrad.implicit_forward` now ridge-regularizes the KKT system
as `M_AA + ε·I` where `ε = 1e-10 · trace(M_AA) / |A|` auto-scales to the
operator's natural diagonal magnitude. CG converges robustly on
`rcv1.binary` (0 failures across 14 outer iters, 5 folds, 70 hypergrad
solves). On well-conditioned problems the bias is below measurement
precision — the v0.1 spike (`spike_cg_stability.py`) shows bit-identical
α* on `leukemia` for `ridge ∈ {0, 10⁻¹², 10⁻¹⁰, 10⁻⁸, 10⁻⁶}`. The
linear-solve, line search, and `grad_search` orchestration all guard
against non-finite values so a broken hypergrad now produces a warned
stall rather than a `α = nan` crash. See `python/sparho/hypergrad.py`,
`tests/test_hypergrad.py::test_implicit_forward_ridge_*`, and
`benchmarks/spike_cg_stability.py`.

Sweep on `rcv1.binary` (n_iter=5, hp0=10⁻²):

| ridge      | ok | α*       | MSE   | time  | CG fail | stalls |
|------------|----|----------|-------|-------|---------|--------|
| auto       | ✓  | 0.003642 | 0.757 | 2.74 s | 0       | 0      |
| 10⁻¹⁰      | ✓  | 0.003642 | 0.757 | 2.73 s | 0       | 0      |
| 10⁻⁸       | ✓  | 0.003642 | 0.757 | 2.73 s | 0       | 0      |
| 10⁻⁶       | ✓  | 0.003644 | 0.757 | 2.74 s | 0       | 0      |

(Larger n_iter and a smaller hp0 reach α ≈ 2.1·10⁻⁵ / MSE 0.194 —
see the headline table above.)

## Next perf gap — inner-solver warm-starting (v0.2)

`LassoCV` runs `(k × |grid|)` Lasso fits with **warm-starting along the
path** — each subsequent fit starts from the previous solution and
converges in O(10) coordinate-descent iterations. Sparho's `SklearnLasso`
adapter is stateless: every outer iteration solves a fresh Lasso from
zero, costing O(100–1000) coord-descent iters per fit.

A v0.1 spike (`benchmarks/spike_warmstart.py`) wraps `Lasso(warm_start=True)`
in a per-fold cache and confirms the gain:

| dataset | cold wall | warm wall | speedup | inner-iter reduction | parity |
|---|---|---|---|---|---|
| `breast-cancer` | 0.24 s | 0.12 s | 2.02× | 3.57× | same α (`Δα/α ≈ 3e-5`), same MSE |
| `leukemia` | 38.1 s | 22.1 s | 1.72× | 1.87× | same α (`Δα/α ≈ 4e-4`), same MSE |

The spike did not include `rcv1.binary` (the cold-start path goes through
the same `SklearnLasso` adapter, which is the bottleneck at small α);
the v0.1 ridge-stabilization fix above is what unblocks running rcv1 at
all, and warm-starting is what closes the 36× wall-time gap with
`LassoCV`.

The library impl will thread an optional `x0` arg through the `Solver`
Protocol and each adapter, plus a per-fold `prev_coef` carry in
`CrossVal`.

### Also planned at v0.2

- **Celer adapter for sparse-X.** `Celer.Lasso` is 2–5× faster than
  `sklearn.linear_model.Lasso` on `rcv1.binary`-class problems; the
  adapter is already shipped behind `[celer]`. Combined with warm-start
  this is the obvious next perf chunk after stability.
- **Rust matrix-free GMRES for the hypergradient solve.** Profiling-gated
  — only if scipy's GMRES/CG is a measurable fraction of total outer time
  after warm-start lands.

## Running the spikes

```bash
uv run python benchmarks/spike_cg_stability.py        # ridge sweep on leukemia + rcv1
uv run python benchmarks/spike_warmstart.py           # warm-start on leukemia + breast-cancer
uv run python benchmarks/spike_warmstart.py --datasets leukemia --n-iter 30
```

Both spikes are diagnostic-only — they exist so v0.2 API changes can be
reviewed against measured baselines rather than hopes.

## Reproducibility

Wall-time numbers reproduce to within ~ 30 % across sequential runs on the
same machine (macOS energy management is the main source of jitter; first
run typically includes cold-cache library import overhead). `α*` and `MSE`
are bit-identical across runs.

The plan's stated 10 % reproducibility tolerance is a v0.2 target; getting
there requires running benchmarks under a warm-cache + thermally-stable
shell (e.g. `pyperf`-style runner with steady-state detection).
