# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and Semantic Versioning.

## [Unreleased]

## [0.1.0] — 2026-05-19

First public release. Functionally-complete nonsmooth bilevel HP
optimization library; Lasso / ElasticNet / WeightedL1 over SquaredLoss and
sparse logistic regression over LogisticLoss. The "Nx faster than
sparse-ho on libsvm" perf headline is a v0.2 deliverable — v0.1 ships
correctness, sparse-X-first-class kernels, and the gradient-based outer
loop (`grad_search` + `hoag_search`) with verified FD parity. See
`ROADMAP.md` and the v0.1 entries below.

### Added
- Sphinx documentation site (Phase 9). `docs/` carries a Furo-themed
  Sphinx build with numpydoc autodoc for the public API, a MyST narrative
  layer (installation / quickstart / concepts / protocols / migration
  guide), and a sphinx-gallery section that runs four end-to-end examples
  (`plot_held_out_lasso`, `plot_weighted_lasso`, `plot_sparse_logreg`,
  `plot_cv_lasso`) at build time. CI gained a `docs` job that runs
  `sphinx-build -W` on every PR and uploads the rendered HTML as an
  artifact; `[docs]` extra in `pyproject.toml` already covered the deps.
- HOAG outer loop (`sparho.hoag_search`). Faithful port of sparse-ho's
  `LineSearch` (which is HOAG, Pedregosa 2016): one val+grad call per outer
  iter, Lipschitz-proxy step adaptation, ``+C·tol`` slack in the acceptance
  test that tolerates criterion-value noise from approximate inner solves,
  optional exponentially-decreasing inner-tolerance schedule, and a
  ``max_step`` trust-region cap on the initial θ-step. `Solver.__call__`
  gains a `tol: float | None` kwarg; `HeldOutMSE` / `HeldOutLogistic` /
  `CrossVal` thread it through; `as_solver` introspects for `tol` like it
  does for `x0`. End-to-end on the leukemia libsvm benchmark: **8.6× faster
  than `LassoCV`** (up from 1.3× at v0.1).

### Changed
- `grad_search` is now plain classical bilevel approximate-gradient descent:
  fixed `lr` in ``θ = log α`` space, no Optimizer plug-in. Useful as a
  baseline; the production workhorse is `hoag_search`.

### Fixed
- `hypergrad._build_ls_data_matvec` was unconditionally converting the
  design to scipy CSC even when the input was already a dense ndarray, so
  every outer iter paid ~400 ms of `coo_tocsr` + `np.nonzero` overhead per
  fold on the `leukemia` benchmark. Split into explicit dense and sparse
  branches: dense uses numpy/BLAS GEMVs over a one-time-extracted `X_A`
  (per CLAUDE.md's "don't port BLAS-bound matvecs to Rust" policy); sparse
  keeps the Rust CSC kernel. `leukemia` wall time **1.05 s → 0.56 s**
  (1.8× faster, **15.4× vs `LassoCV`**); `implicit_forward` itself went
  from 472 ms → 56 ms (8.4× faster).

### Removed
- `LineSearch` Armijo backtracking optimizer. Replaced by `hoag_search`:
  Armijo's strict sufficient-decrease test stalls on small-``||y||²``
  problems with warm-started inner solves because the inner solver's
  ``tol · ||y||²`` convergence ceiling absorbs the per-α value change, so
  trial points are indistinguishable from the current point. HOAG's slack
  term handles this directly.
- `Optimizer` protocol, `GradDescent` class, `GDState`, and the entire
  `sparho.optimizer` module. The protocol was an extension point for
  pluggable step rules that earned its keep only on speculation; with
  HOAG owning its own outer loop and `grad_search` being a 20-line plain
  GD, the abstraction had no live users. Can be reintroduced if a real
  custom-optimizer use case appears.

### Added (earlier in unreleased)
- Warm-start support for the inner solver across outer iterations
  (v0.2 perf item 1). `Solver.__call__` gains an optional keyword-only
  `x0: Array | None = None` parameter; `SklearnLasso`,
  `SklearnElasticNet`, `SklearnWeightedLasso`, `CelerLasso`, and
  `CelerElasticNet` honor it via their estimators' `warm_start=True` path.
  `SklearnLogisticRegression` accepts `x0` for protocol conformance but
  ignores it (liblinear has no warm-start). `as_solver` introspects the
  wrapped callable and forwards `x0` only when accepted, so existing
  `(problem, hp) -> SolverResult` user callables keep working.
  `HeldOutMSE` / `HeldOutLogistic` thread `x0` through to the solver;
  `CrossVal` gains a `warm_start: bool` flag and an internal per-fold
  cache (excluded from equality / hash / repr) that seeds each fold's
  next inner solve from the previous outer iteration's `β*`. End-to-end
  on a dense 400×200 synthetic this gives ~1.9× wall-time vs cold-start
  with α agreeing to 0.3 %.
- Phase 0 scaffold: pyproject.toml (maturin backend), cargo workspace, PyO3
  ABI3 py311 bindings, smoke-test for round-trip wheel build, CI matrix
  (ubuntu + macos × 3.11/3.12/3.13), release workflow with Trusted Publishing.
