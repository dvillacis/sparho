# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and Semantic Versioning.

## [Unreleased]

### Hypergradient redesign — full sparse-ho algo family
- **`hypergrad.py` is now the `hypergrad/` package.** It ports sparse-ho's
  `algo` module as four free functions on the `HypergradFn` seam:
  `implicit_forward` (the **default**), `forward`, `backward`, and `implicit`.
  Select one by passing it to `grad_search` / `hoag_search`, or look it up with
  `get_hypergrad(name)`.
- **`implicit_forward` now runs a native BCD Jacobian fixed point** (Rust
  `crates/sparho-core/src/bcd.rs`) restricted to the active set, for
  `SquaredLoss × {L1, ElasticNet, WeightedL1}`, dense and CSC. It matches the
  previous CG result and finite differences to ~1e-10. `LogisticLoss` and
  `GroupL1` fall back to the CG path.
- **BREAKING: `implicit_forward`'s algorithm changed.** It was sparse-ho's
  `Implicit` (matrix-free CG on the restricted KKT Hessian); that exact code now
  ships unchanged as `sparho.hypergrad.implicit`. The `ridge` stabilization knob
  lives only on `implicit` (a BCD fixed point has no Hessian to regularize); it
  is accepted-but-ignored on the BCD paths. Code that relied on `ridge` or on
  CG-specific singular-system behavior should call `implicit` directly.
- **`forward`** re-solves the inner problem from cold while propagating the full
  Jacobian (joint solve); **`backward`** records β sweeps and reverse-replays
  (dense L1; other cases delegate). Both equal `implicit_forward` at the optimum.
- **`NativeBcdLasso`** adapter — sparho's own Rust coordinate-descent inner
  solver for `Problem(SquaredLoss, L1, …)` (matches `SklearnLasso` to ~1e-12),
  alongside the existing external adapters.
- **`WarmStartHypergrad`** — opt-in wrapper that caches the support Jacobian and
  remaps it across outer iterations (warm-starts the fixed point; same answer,
  fewer sweeps).
- New Rust kernels exposed via `_core`: `bcd_lasso_{dense,csc}`,
  `bcd_lasso_jac_{dense,csc}`, `bcd_lasso_backward_dense`,
  `solve_restricted_normal_{dense,csc}` (the `(XsᵀXs/n + cI)x = b` primitive).

### Dependencies
- **scikit-learn floor bumped to `>=1.6`** (was `>=1.3`). The v0.3
  sklearn-compatible wrappers (`LassoHO` / `ElasticNetHO` /
  `LogisticRegressionHO`) use `sklearn.utils.validation.validate_data`
  and `__sklearn_tags__`, both introduced in sklearn 1.6 (Dec 2024).
  The `ci-min-deps` floor-pin job in `.github/workflows/ci.yml` is
  bumped to match (`scikit-learn==1.6.*`). numpy floor unchanged.
- **scipy floor bumped to `>=1.12`** (was `>=1.10`).
  `hypergrad.implicit_forward` passes `rtol=` and `atol=` to
  `scipy.sparse.linalg.cg`, both introduced in scipy 1.12 (Jan 2024);
  earlier scipy only knew the now-deprecated `tol=`. The `ci-min-deps`
  pin matches at `scipy==1.12.*`.

### Theory in docs (v0.8)
- **`docs/theory/` section.** New self-contained derivations covering
  the bilevel setup and notation (`theory/index.md`), the KKT-restricted
  implicit-diff linear system `M_AA · dβ*/dα = -r` and its ridge
  stabilization (`theory/implicit_diff.md`), the active-set restriction
  argument under strict subgradient inequality and GroupL1's
  active-group expansion (`theory/active_set.md`), per-penalty prox /
  Jacobian / subdifferential / α-Jacobian tables linked into
  `crates/sparho-core/src/prox.rs` (`theory/penalties.md`), the
  outer-criterion chain rule plus a full SURE / SUGAR FDMC derivation
  from Stein's identity (`theory/criteria.md`), and the HOAG
  convergence sketch with sparho-specific deviations (`theory/convergence.md`).
- **`docs/refs.bib` + `sphinxcontrib-bibtex`.** Single bibliography
  backing theory pages and docstrings. Seeded with the foundational
  papers (Pedregosa 2016 HOAG, Bertrand 2020/2022 implicit-diff,
  Deledalle 2014 SUGAR, Stein 1981, Zou-Hastie-Tibshirani 2007
  Lasso-DOF, Yuan-Lin 2006 Group Lasso, Beck-Teboulle 2009 FISTA,
  Krantz-Parks 2013 IFT, Bolte 2021 nonsmooth IFT, Massias 2018 celer,
  and supporting references). New `docs/theory/references.md` renders
  the full bibliography. MyST `dollarmath` + `amsmath` extensions
  enabled in `docs/conf.py` for inline / display math.
- **Docstring cross-links.** `implicit_forward`, `Sure`, `CrossVal`,
  and `hoag_search` docstrings gained Sphinx `:doc:` references into
  the new theory pages; `docs/concepts.md` opens with an intro
  paragraph pointing into `theory/`; main `docs/index.md` exposes a
  new "Theory" toctree.

### Reproducibility (v0.7)
- **BLAS-thread discipline.** New `sparho.testing.pin_blas_threads(n)`
  context manager pins `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`,
  and `BLIS_NUM_THREADS` for the duration of a block and additionally
  retunes the live `threadpoolctl` pool. Restores prior env state on
  exit. New `docs/reproducibility.md` documents the discipline,
  guarantees and non-guarantees, and the audit matrix.
- **Autouse BLAS pin in pytest.** New `tests/conftest.py` autouses
  `pin_blas_threads(1)` for the entire test session; opt out with
  `SPARHO_TEST_RESPECT_BLAS=1`. Stabilizes the determinism + golden
  suites across BLAS backends.
- **Benchmark provenance + JSON output.**
  `benchmarks/lasso_libsvm.py` gains `--blas-threads`, `--results-json
  PATH`, `--provenance-json PATH` flags. Provenance captures git SHA,
  CPU model, OS, Python implementation/version, BLAS backend resolved
  via `np.show_config()`, BLAS env vars *at capture time* (inside the
  pin block), and pinned package versions (`numpy`, `scipy`,
  `scikit-learn`, `celer`, `libsvmdata`, `pandas`, `matplotlib`).
  Schema versioned as `sparho-bench-provenance@1`.
- **Deterministic table rendering.** New `benchmarks/render_tables.py`
  reads results JSON + provenance JSON and emits the canonical
  Markdown tables for `benchmarks/README.md` and the top-level
  `README.md`. Output is byte-deterministic for fixed inputs;
  re-running on the same JSONs produces an identical file. Wired into
  `.github/workflows/perf.yml` after the bench step; both JSONs and
  rendered tables upload as the `perf-artifacts` artifact (replaces
  the old `perf-log`).
- **Determinism audit matrix.** New `tests/test_determinism_matrix.py`
  runs `(BLAS threads ∈ {1, 2, 4}) × (seed ∈ {0, 7}) × (criterion ∈
  {CrossVal, Sure})`. At `n_threads=1`, asserts bit-equality of
  `best_hyperparam` and `best_coef` across replays; at `n_threads > 1`,
  asserts agreement within `atol=1e-5, rtol=1e-4` so the test
  documents the multi-thread drift envelope. 12 matrix cases +
  1 sanity test ("different seeds → different results").
- **Dataset hash verification.** New `tests/fixtures/datasets.py`
  wraps `libsvmdata.fetch_libsvm` with SHA256 verification against
  a pinned manifest `tests/fixtures/libsvm_manifest.json`. Missing
  entries print the observed hashes for the contributor to commit
  (bootstrap mode); hash drift raises `DatasetHashMismatch`.
  `breast-cancer` bootstrap-pinned in this release; `leukemia` and
  `rcv1.binary` pin-on-first-use. Workflow documented in
  `CONTRIBUTING.md`.
- **Bench lockfile.** New `requirements-bench.txt` generated by
  `uv pip compile pyproject.toml --extra bench` — pins the exact
  numpy/scipy/sklearn/celer/libsvmdata/matplotlib/pandas closure used
  to produce the published benchmark numbers. Quarterly regeneration
  cadence documented in `docs/reproducibility.md`.
- **mypy / threadpoolctl override.** Added to the `ignore_missing_imports`
  list since the package has no public stubs.
- Test count 229 → 242.

### Citability (v0.5)
- **Citation infrastructure.** New `CITATION.cff` (schema 1.2.0, passes
  `cffconvert --validate`) and `.zenodo.json` so a tagged release auto-
  mints a Zenodo DOI via the GitHub–Zenodo integration; new
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1 by reference);
  `README.md` gains "How to cite" + BibTeX entries for sparho and the
  underlying `sparse-ho` ICML 2020 paper, plus a "Community" link block.
  `RELEASE.md` documents the one-time GitHub–Zenodo setup and the
  per-tag DOI-minting procedure; `pyproject.toml` keywords expand 5→13
  and `project.urls` 4→8 (Homepage, Release Notes, Code of Conduct,
  Security Policy). New `citation` CI job runs `cffconvert --validate`
  on every PR. ORCID and concept-DOI placeholders are filled in once at
  the first v0.5.0 tag.

### Numerical rigor (v0.6)
- **Coverage gates.** `pytest --cov=sparho` wired into every CI
  matrix entry with Codecov upload (one flag per matrix cell);
  separate `rust-coverage` job runs `cargo llvm-cov --lib` on Linux
  and uploads under the `rust` flag. New `codecov.yml` declares
  per-module floors in informational mode (75% python / 70% rust);
  Codecov badge added to `README.md`.
- **KKT post-solve assertions.** New public testing surface
  `sparho.testing.kkt_residual` / `assert_kkt_optimal`, computing the
  proximal fixed-point residual `‖β − prox_R(β − ∇L(β); α)‖_∞` by
  composing the existing Rust prox kernels (`_core.prox_{l1,
  elastic_net,weighted_l1,group_l1}`) with CSC matvecs — no new
  numerical kernels. Exhaustive `match` on the closed `Datafit ×
  Penalty` union with `assert_never` tail so mypy flags missed
  dispatch when a new variant lands. `tests/test_kkt_residual.py`
  parametrizes across every (Datafit, Penalty) combination including
  `L1`/`ElasticNet`/`WeightedL1`/`GroupL1` × `SquaredLoss`/`LogisticLoss`
  (5 solver-level assertions plus boundary cases at α > α_max and
  off-optimum perturbations).
- **Hypothesis property tests on the Python surface.** Three new test
  files (`test_property_problem.py`, `test_property_hypergrad.py`,
  `test_property_criteria.py`) using `hypothesis>=6.0`. Cover Problem
  construction invariants under random shape/sparsity, FD parity of
  `implicit_forward` over random Lasso problems (log-uniform α
  exercising both nearly-fully-active and nearly-fully-sparse
  regimes), and `HeldOutMSE` parity against a direct numpy
  implementation. Modest `max_examples` (15–40) keeps CI fast.
- **Golden numerical regression suite.** New `tests/golden/` with five
  JSON fixtures pinning `(β*, training_loss, KKT residual, active
  set)` at `tol=1e-10` for canonical (datafit, penalty, solver, α)
  triples — Lasso, ElasticNet, weighted Lasso, group Lasso, sparse
  logistic regression. `tests/golden/generate.py` regenerates the
  fixtures; the runner in `tests/test_golden.py` re-solves and
  asserts agreement at `atol=1e-8, rtol=1e-6` on coef, `atol=1e-10,
  rtol=1e-8` on training loss, and active-set bit-equality.
- **Test environment completeness.** `hypothesis>=6` and `pandas>=2.1`
  added to the `[project.optional-dependencies].dev` extra. The
  pandas pin fixes a pre-existing collection error in
  `tests/test_wrappers.py` (DataFrame round-trip coverage) that the
  full-suite gate from B.1 surfaced. Total test count 197 → 229.

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

### Docs & community
- **Docs & community surface (v0.4 §6).** New top-level `CONTRIBUTING.md`
  (dev setup via `uv sync --extra dev` + `maturin develop`, the gate
  commands, the architecture pillars, the closed-union extension
  recipe, commit norms) and `SECURITY.md` (private email reporting,
  90-day disclosure window, explicit threat-model statement). New
  GitHub issue templates (`.github/ISSUE_TEMPLATE/bug.md`, `feature.md`)
  and `.github/PULL_REQUEST_TEMPLATE.md` carrying the gate checklist
  + CHANGELOG reminder. Two new sphinx-gallery examples:
  `docs/examples/plot_group_lasso.py` (Group-L1 end-to-end with HOAG;
  recovers the right active groups on synthetic block-sparse data) and
  `docs/examples/plot_migration_from_sparse_ho.py` (runnable
  side-by-side translation that pairs with the prose
  `migration_from_sparse_ho.md` table). `pyproject.toml [project.urls]`
  gained `Documentation` (RTD), `Issues` (GitHub), `Changelog`
  (GitHub) — the metadata PyPI uses for the project sidebar.
  `migration_from_sparse_ho.md` row for `FiniteDiffMonteCarloSure`
  updated to point at the v0.3 §2 `Sure` landing. The frozen-stable /
  experimental / private surface declaration that was originally
  scoped here landed in v0.4 §3 (`docs/stability.md`).
  Closes ROADMAP v0.4 §6.

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
