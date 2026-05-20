# Benchmarks

Sparho's v0.1 benchmarks compare against `sklearn.linear_model.LassoCV` —
the standard grid-search baseline that sparse-ho refugees would have used
before. The bench is intentionally narrow at v0.1: prove that the gradient-
based outer loop produces the right answer on real data and is in the same
ballpark as a hand-tuned grid.

## Running

```bash
uv sync --extra dev --extra bench --extra celer
uv run python benchmarks/lasso_libsvm.py                            # full run, ~ 30 s
uv run python benchmarks/lasso_libsvm.py --quick                    # CI smoke, fewer outer iters
uv run python benchmarks/lasso_libsvm.py --rcv1                     # add rcv1.binary (slow)
uv run python benchmarks/lasso_libsvm.py --solver celer --rcv1      # v0.2 headline config
uv run python benchmarks/lasso_libsvm.py --cold-start                # reproduce the v0.1 baseline
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

## v0.2 numbers — HOAG + warm-start + celer

The inner solver is `CelerLasso` (working-set CD with extrapolation),
warm-started across outer iterations (`CrossVal(warm_start=True)`), and
the outer loop is HOAG (Pedregosa 2016): adaptive step size from a
Lipschitz proxy, ``+C·tol`` slack in the acceptance test that absorbs
criterion-value noise from approximate inner solves, and an
exponentially decreasing inner-tolerance schedule. Run with:

```bash
uv run python benchmarks/lasso_libsvm.py --datasets breast-cancer leukemia --rcv1 --solver celer
```

| dataset | shape | sparho α* | sparho MSE | sparho time | iters | LassoCV α* | LassoCV MSE | LassoCV time | grid | sparho speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| `breast-cancer` | 683×10 | 2.98·10⁻³ | 0.508 | 0.26 s | 30 | 1·10⁻³ | 0.508 | 0.02 s | 20 | 0.06× (overhead-bound) |
| `leukemia` | 38×7129 | 0.113 | 0.435 | 0.58 s | 30 | 0.0785 | 0.433 | 19.0 s | 20 | **32.8×** |
| `rcv1.binary` | 20242×47236 sparse | **2.12·10⁻⁵** | **0.194** | 211 s | 30 | 1·10⁻⁴ (grid floor) | 0.225 | 22.6 s | 15 | 0.11× (quality win) |

**Reading the v0.2 table**:
- **`leukemia`** — sparho is **32.8× faster** than `LassoCV`, up from
  1.3× at v0.1 and 8.6× with the sklearn inner solver. Three compounding
  wins: (1) HOAG removes Armijo's trial-evaluation overhead, (2)
  warm-start cuts inner-solver iterations per outer step, (3) celer's
  working-set CD with extrapolation is roughly 5–10× faster than
  sklearn's coordinate descent on this dense `n ≪ p` regime.
- **`breast-cancer`** — still overhead-bound; both finish in well under a
  second. HOAG matches `LassoCV`'s MSE to 0.02 %.
- **`rcv1.binary`** — same quality story as v0.1: sparho finds an MSE
  **14 % better** than `LassoCV`'s grid optimum by walking α an order of
  magnitude below the grid floor. Wall time dropped 2× (v0.1: 433 s →
  v0.2: 211 s) — celer is ~1.65× faster than sklearn for the inner
  solves on this dataset (vs 347 s with sklearn at v0.2). `LassoCV` is
  still faster in pure wall time because at α ≈ 2·10⁻⁵ the active set
  is large and the inner-solver work is irreducible — sparho wins on
  *quality* per outer iter rather than on raw wall time.

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

## Landed at v0.2 — warm-start + celer

Both of the perf gaps called out at v0.1 are now closed:

- **Inner-solver warm-starting.** The `Solver` Protocol grew an optional
  `x0` keyword; `SklearnLasso` / `SklearnElasticNet` /
  `SklearnWeightedLasso` / `CelerLasso` / `CelerElasticNet` honor it via
  `warm_start=True`. `CrossVal(warm_start=True)` carries per-fold
  `prev_coef` across outer iterations. The v0.1 spike
  (`benchmarks/spike_warmstart.py`) measured 1.7–2× on dense designs;
  the v0.2 end-to-end numbers fold this in.
- **Celer adapter for sparse-X.** `CelerLasso` is the recommended inner
  solver for the v0.2 perf story: ~5–10× faster than `SklearnLasso` on
  `leukemia` (compounding the warm-start win into 32.8× vs `LassoCV`),
  ~1.65× faster on `rcv1.binary` (211 s vs 347 s with sklearn at v0.2).

The remaining v0.2 item is profiling-gated:

- **Rust matrix-free GMRES for the hypergradient solve.** Only if
  scipy's GMRES/CG is a measurable fraction of total outer time. After
  the dense-matvec fix (`implicit_forward` is 56 ms/call on `leukemia`),
  it is not — deferred.

## Running the spikes

```bash
uv run python benchmarks/spike_cg_stability.py        # ridge sweep on leukemia + rcv1
uv run python benchmarks/spike_warmstart.py           # warm-start on leukemia + breast-cancer
uv run python benchmarks/spike_warmstart.py --datasets leukemia --n-iter 30
```

Both spikes are diagnostic-only — they exist so v0.2 API changes can be
reviewed against measured baselines rather than hopes.

## Reproducibility

`α*` and `MSE` are bit-identical across runs (sparho's outer loop is
deterministic; `LassoCV` is deterministic on a fixed alpha grid). Only
wall time varies.

The bench supports a repeatable mode (landed at v0.2):

```bash
uv run python benchmarks/lasso_libsvm.py --solver celer --repeat 5 --cooldown 2
```

`--repeat N` runs each timed section N times; sparho and `LassoCV` are
interleaved per iteration so thermal load is shared fairly. `--warmup K`
(defaults to 1 when `--repeat > 1`) drops the first K samples to amortize
cold-cache effects. `--cooldown S` sleeps S seconds between iters so
macOS thermal state can settle. `gc.collect()` is called between iters
to keep collection out of the timed sections. The reported wall is the
**median** of the post-warmup samples; the **spread** is reported as
`(max − min) / median`.

### v0.2 reproducibility measurement

Single-thread Apple M-series, single Python process. Dense datasets
sampled with `--repeat 5 --cooldown 2`; `rcv1.binary` sampled with
`--repeat 3 --cooldown 3` (each sample is ~3 min so the budget is
tighter):

| dataset | sparho spread | `LassoCV` spread | within 10 % target? |
|---|---|---|---|
| `breast-cancer` 683×10 | ±0.9 % | ±1.8 % | ✅ both |
| `leukemia` 38×7129 | ±6.9 % | ±19.9 % | ✅ sparho only |
| `rcv1.binary` 20242×47236 | ±33.1 % | ±16.0 % | ❌ neither |

**Sparho hits the 10 % target on both dense datasets** — that's enough
to detect a sparho-side regression across releases on the headline
benchmarks. `LassoCV` jitters more than 10 % on `leukemia`, so the
speedup *ratio* on that dataset is bounded by sklearn's own variance.

`rcv1.binary` is the hard case: each sparho sample is ~3 minutes, and a
3-second cooldown is not enough for macOS to dissipate the thermal load
from a multi-minute burn. The per-iter timings show a clear
warm-then-throttle pattern (rep 1: 175 s, rep 2: 176 s, rep 3: 245 s —
the third iter ran while the CPU was already warm from the first two).
Longer cooldowns help but don't fully eliminate this on macOS without
OS-level controls.

The residual jitter is irreducible on macOS without process-level
isolation (no CPU pinning, no real-time scheduling). Tightening it
further would require a Linux host running `taskset` + `pyperf
--isolated`, which is future work, not a v0.2 deliverable. The
bit-identical `α*` / `MSE` outputs are the durable correctness check
across releases; the wall-time medians from `--repeat 5 --cooldown 2`
(or `--repeat 3 --cooldown 30` for `rcv1.binary` if you want a longer
soak) are the durable perf check.
