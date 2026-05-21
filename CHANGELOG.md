# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and Semantic Versioning.

## [Unreleased]

### Security
- **FFI hardening (v0.3.1).** Every Rust kernel that previously did a bare
  `as usize` cast on caller-supplied `i32` indices now validates the input
  first and returns `Result<(), &'static str>` instead of `assert!`-ing —
  malformed `scipy.sparse.csc_matrix` data from Python reaches the user as
  `ValueError`, never as a Rust panic unwinding through CPython (undefined
  behavior on a release build with `panic = "unwind"`). Affects
  `csc::{matvec,rmatvec}`, `residual::restricted_ls_hessian_matvec`, and
  `prox::{prox_l1,prox_jacobian_l1,prox_elastic_net,prox_jacobian_elastic_net,
  prox_weighted_l1,prox_jacobian_weighted_l1,prox_group_l1}`. The PyO3
  wrappers in `crates/sparho-py/src/lib.rs` translate each kernel error to
  `PyValueError`; `PyReadonlyArray1::as_slice()` already required C-contiguity
  on 1-D inputs. Cargo release profile sets `panic = "abort"` + `strip =
  true` as the safety net for any remaining bug. 15 new regression tests in
  `tests/test_ffi_safety.py` cover bad CSC structure, out-of-range active
  indices, non-contiguous slices, and bad group partitions.

### Observability
- **Observability hooks (v0.4 §3).** `grad_search` and `hoag_search` gain
  an optional `callback: Callable[[IterationRecord], None] | None = None`
  kwarg, invoked once per appended `IterationRecord` (and once more on the
  HOAG rejection branch's replacement record). `IterationRecord.extras` is
  now populated systematically: `cg_status` (`"ok"` /
  `"nonconvergence"` / `"nonfinite"`, replaces the v0.3.1
  `cg_nonconvergence` / `cg_nonfinite` boolean keys); `inner_dual_gap`
  (float, propagated from `HeldOutMSE` / `HeldOutLogistic`; max-across
  for `CrossVal` / `Sure`); HOAG records additionally carry `step_size`
  and `L_estimate`. `CriterionResult` gained an
  `inner_dual_gap: float | None = None` field. The three wrappers
  (`LassoHO`, `ElasticNetHO`, `LogisticRegressionHO`) gain `verbose:
  int = 0`; `verbose=1` wires a default `_VerbosePrinter` callback that
  prints one line per outer iter (`verbose=2` also prints `step_size` /
  `L_estimate`). New `docs/stability.md` declares the frozen-stable /
  experimental / private surfaces, with the `extras` schema tabled out
  and flagged stability-experimental. 10 new tests in
  `tests/test_observability.py`. Closes ROADMAP v0.4 §3.

### CI
- **CI hardening (v0.4 §2).** PR matrix now includes `windows-latest`
  (both the rust and python jobs; Windows tested on Python 3.12 with the
  full Linux/macOS sweep across 3.11–3.13). New `cargo-audit` job
  (Linux-only; one ignore for the known PyO3 0.22 advisory
  RUSTSEC-2025-0020 pending an `pyo3 → 0.24` bump). New `pre-commit` job
  running `pre-commit run --all-files`. New `pytest --doctest-modules
  python/sparho` step so future doctests are automatically gated. New
  `ci-celer` job exercising the optional `[celer]` extra. New
  `ci-min-deps` job pinning `numpy==1.24 / scipy==1.10 /
  scikit-learn==1.3` so silent floor-version regressions surface on PR.
  New `.github/workflows/perf.yml` with `ci-linux-isolated`: runs the
  `leukemia` libsvm benchmark under `taskset -c 0` with `--repeat 5
  --cooldown 2`, fails the job if sparho's wall-time spread ≥ 10 %
  (the v0.2 §5 reproducibility target macOS jitter couldn't hit).
  Triggers on `workflow_dispatch` + tags, not PR.
  `.github/workflows/release.yml` gained a `wheel_smoke` job (Linux +
  macOS arm64 + Windows) that downloads each cibuildwheel artifact into
  a fresh venv and runs `LassoHO().fit(X, y)` end-to-end before the PyPI
  publish step. `pyproject.toml` per-file-ignores extended for
  `docs/examples/**` (sphinx-gallery title format conflicts with
  pydocstyle D205 / D400). Closes ROADMAP v0.4 §2.

### Tests
- **Coverage expansion (v0.4 §1).** New `proptest` dev-dependency drives 11
  property suites (~256 cases each) in
  `crates/sparho-core/tests/proptests.rs`: prox-kernel algebraic identities
  (`prox_l1` ≡ elementwise `soft_threshold`, sign preservation +
  non-expansion, below-threshold zeroing; `prox_elastic_net@rho=1` ≡
  `prox_l1`; uniform-`α` `prox_weighted_l1` ≡ `prox_l1`; singleton-group
  `prox_group_l1` ≡ `prox_l1`; block non-expansion), and CSC parity
  against a dense reference (`csc::matvec`, `csc::rmatvec`,
  `restricted_ls_hessian_matvec` on the full active set). New pytest
  files: `tests/test_wrappers_pickle.py` (9 cases: unfit/fit pickle
  round-trip + `sklearn.clone` for each of LassoHO / ElasticNetHO /
  LogisticRegressionHO), `tests/test_determinism.py` (3 cases: same
  `random_state` → bit-identical `SearchResult` for `CrossVal.kfold` and
  `Sure`; different seeds detect-ably differ), `tests/test_degenerate.py`
  (6 cases: empty active set, all-zero target, single-class logistic
  rejection, fully-collinear features, `n ≪ p`), and
  `tests/test_hoag_schedule.py` (3 cases: `WeightedL1` dense+sparse and
  `GroupL1` under `hoag_search` + `tolerance_decrease='exponential'`,
  exercising the rejection branch). Total: 197 pytest (+21 over v0.3.1),
  33 cargo unit + property tests (+11). Closes ROADMAP v0.4 §1.

### Added
- **Input validation (v0.3.1).** `Problem.__post_init__` now enforces
  `design.ndim == 2`, `target.ndim == 1`, matching first axis, and
  `np.isfinite` on both (sparse designs check `.data`; opt out globally via
  `sparho.problem.CHECK_FINITE = False` for masked-input pipelines).
  `ElasticNet.__post_init__` enforces `rho ∈ (0, 1]`. `GroupL1.__post_init__`
  validates the partition (disjoint, non-empty groups, non-negative indices,
  weights length matches groups) at construction — previously this was only
  checked inside `from_labels`. `grad_search` / `hoag_search` preflight
  vector-α length against `problem.n_features` before the outer loop starts.
  `adapters._common.as_scalar` / `as_vector` produce actionable error
  messages that name the expected shape and point at the right penalty
  (scalar-α vs WeightedL1). 21 new tests in `tests/test_problem_validation.py`.
- `IterationRecord.extras: Mapping[str, object]` field — populated by the
  search loop when implicit-diff fails (`"cg_nonconvergence"`,
  `"cg_nonfinite"` keys today). Defaults to an empty dict; schema is
  stability-experimental.
- `sparho.GroupL1` penalty — Yuan & Lin's block-sparsity regularizer
  `R(β; α) = α · Σ_k w_k · ‖β_{G_k}‖_2`, with default `w_k = √|G_k|` so the
  penalty is invariant to group size. New variant of the closed `Penalty`
  union; mypy strict flags any algorithm that forgets to dispatch on it.
  Build via `GroupL1(groups=((0, 1), (2, 3, 4), ...))` or the convenience
  `GroupL1.from_labels(label_array)` factory.
  `sparho._core.prox_group_l1` is the new Rust block-soft-threshold kernel
  (allocation-free, CSR-style group layout). `implicit_forward` gains a
  `case GroupL1()` arm that handles the *block-diagonal* penalty curvature
  `(α·w_k/r_k)·(I − u_k u_kᵀ)` on each active group — distinct from the
  uniform-diagonal curvature L1 / ElasticNet / WeightedL1 use. The active
  set is expanded from "active groups" (groups with `‖β_{G_k}‖ > 0`) so
  internal-zero coords still enter the KKT system, per the Group-Lasso
  optimality conditions. `sparho.adapters.GroupLassoFista` is the canonical
  built-in inner solver — accelerated proximal gradient (Beck-Teboulle 2009)
  on top of the Rust prox kernel, Lipschitz constant estimated by power
  iteration (pass `lipschitz=L` to skip when re-fitting many α values on
  the same design), warm-start via `x0`, KKT-stationarity `dual_gap` proxy.
  Works with dense and CSC-sparse `X`. 13 new tests in
  `tests/test_group_lasso.py` (dataclass + `from_labels` round-trip,
  block-sparse support recovery, ρ-singleton equivalence to plain Lasso,
  sparse↔dense parity, warm-start invariance, closed-form and FD-validated
  hypergradients) plus 4 kernel-level tests in `tests/test_kernels.py`.
  Closes ROADMAP v0.3 §4.
- `sparho.LassoHO`, `sparho.ElasticNetHO`, `sparho.LogisticRegressionHO` —
  sklearn-compatible wrapper estimators (`BaseEstimator + RegressorMixin /
  ClassifierMixin`). Each exposes `fit` / `predict` / `score` / `coef_` /
  `intercept_` / `alpha_` / `n_iter_` / `feature_names_in_` /
  `n_features_in_` and survives `sklearn.utils.estimator_checks.check_estimator`
  with seven sample-weight checks marked as known-skip (ROADMAP item M).
  Drops into `Pipeline`, `GridSearchCV`, `cross_val_score`, `clone`,
  `permutation_importance`, MLflow autolog, and EconML/DoubleML without
  user code changes. pandas DataFrames in → `feature_names_in_` capture
  via sklearn's `validate_data`. **Standardization decision (2026-05-20):**
  no `standardize=` parameter; users compose `Pipeline([StandardScaler(),
  LassoHO()])` per the post-sklearn-1.0 norm — see the new how-to doc.
  `fit()` emits a `UserWarning` when feature scales are uneven
  (`ptp(X.std(axis=0)) > 10 * X.std(axis=0).mean()`). `fit_intercept=True`
  is supported on dense X via upfront centering; sparse X + `fit_intercept=
  True` raises with a redirect to `StandardScaler(with_mean=False)` or
  `fit_intercept=False` (sparse-aware offset-adjusted matvecs deferred to
  v0.4). `LogisticRegressionHO` accepts arbitrary 2-class labels and maps
  internally to the `{-1, +1}` `LogisticLoss` convention; `predict_proba`
  exposed for ecosystem compatibility. `sample_weight` is rejected with a
  `NotImplementedError` pointing at ROADMAP item M. 20 new tests in
  `tests/test_wrappers.py`. Closes ROADMAP v0.3 §3 (modulo the
  sparse-aware-centering deferral).
- `docs/how-to/standardization-and-leakage.md` — recipe for
  `Pipeline([StandardScaler(), LassoHO()])` plus an explicit warning that
  the wrapper's internal CV sees post-scaler folds (same leakage trap as
  `LassoCV` inside a Pipeline, sklearn#26359); recommends
  `HeldOutMSE` + outer `cross_validate` when leakage matters.
- `random_state` parameter on `sparho.adapters.SklearnLogisticRegression`
  (default `0`). Threads through to liblinear's internal RNG so re-fits
  at the same hyperparameter are bit-identical — prerequisite for
  `LogisticRegressionHO` to pass sklearn's `check_fit_idempotent`.
- `sparho.Sure` — Stein's Unbiased Risk Estimator criterion, FDMC
  (finite-difference Monte Carlo) variant after Deledalle et al. 2014 (SUGAR).
  Tunes α for `SquaredLoss` problems with known Gaussian noise σ when no
  held-out set exists (denoising, signal recovery, single-fold). Two inner
  solves per evaluation (β̂(y) and β̂(y+εδ)); `value_and_hypergrad` sums two
  `hypergrad_fn` calls. Default ε = `2σ/n^0.3` (Deledalle heuristic), single
  deterministic δ probe per instance for line-search monotonicity, optional
  internal warm-start of both inner solves. Refuses non-`SquaredLoss`
  problems at call time with a `TypeError`. Reinstates the
  `FiniteDiffMonteCarloSure` criterion that v0.1 had dropped from Phase 6.
  Six new tests in `tests/test_criteria.py` (protocol conformance, datafit
  guard, seed reproducibility, MC-averaged DOF concentration at the Lasso
  closed-form `|support(β̂)|`, FD-parity on the hypergradient, warm-start
  ≡ cold-start convergence, vector-α / `WeightedL1` smoke); one new
  sphinx-gallery example `plot_sure_lasso.py`. Closes ROADMAP v0.3 §2.

## [0.2.0] — 2026-05-20

Closes the v0.2 perf-story arc: HOAG outer loop, inner-solver warm-start,
dense-matvec fix, and `CelerLasso` as the recommended inner solver
landed earlier under [0.1.0]; this release lands the benchmark refresh,
reproducibility harness, and adapter re-exports that round it out.
Headline (single-thread, Apple M-series): `leukemia` is **32.8× faster
than `LassoCV`** (up from 1.3× at v0.1); `rcv1.binary` wall halved (433 s
→ 211 s) at the same MSE win (0.194 vs `LassoCV`'s 0.225). See
`ROADMAP.md` §v0.2 and `benchmarks/README.md` for the full numbers and
methodology. 94 pytest, 11 cargo, mypy strict, clippy clean, single
ABI3 wheel.

### Added
- `--solver {sklearn,celer}` flag in `benchmarks/lasso_libsvm.py` so the
  v0.2 perf story can swap in `CelerLasso` as the inner solver. `CelerLasso`
  and `CelerElasticNet` are now re-exported from `sparho.adapters`.
- Reproducibility mode in `benchmarks/lasso_libsvm.py`: `--repeat N` runs
  each timed section N times with interleaved sparho/LassoCV iterations
  (so thermal load is shared fairly), `--warmup K` drops the first K
  samples (defaults to 1 when `--repeat > 1`), `--cooldown S` sleeps S
  seconds between iters so macOS thermal state can settle. `gc.collect()`
  runs between iters to keep GC out of the timed sections. Median wall +
  `(max-min)/median` spread is reported per dataset. With
  `--repeat 5 --cooldown 2`, sparho's own wall-time spread is 0.9–6.9 %
  on the dense libsvm datasets — within the ROADMAP's 10 % target for
  detecting sparho-side regressions across releases. (`LassoCV`'s own
  jitter is the residual: ~20 % on `leukemia`, ~16 % on `rcv1.binary`;
  the `rcv1.binary` sparho row also jitters ~33 % because each sample
  is multi-minute and macOS throttles under sustained load. Both are
  irreducible on macOS without process-level isolation.)

### Changed
- v0.2 benchmark numbers refreshed across all three libsvm Lasso datasets
  with HOAG + warm-start + `CelerLasso`. Headlines (single-thread, M-series):
  `leukemia` **32.8× vs `LassoCV`** (0.58 s vs 19.0 s — up from 8.6× with
  the sklearn inner solver, 1.3× at v0.1); `rcv1.binary` wall halved
  (211 s vs v0.1's 433 s) with the same quality win (sparho MSE 0.194 vs
  `LassoCV` 0.225) — celer is ~1.65× faster than sklearn on rcv1 (347 s
  with sklearn at v0.2); `breast-cancer` still overhead-bound. Top-level
  `README.md`, `benchmarks/README.md`, and `ROADMAP.md` updated.

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
