# sparho roadmap

The authoritative implementation plan lives at
`~/.claude/plans/swirling-soaring-hamster.md`. This file is the short
status summary.

## v0.1 — Shipped to PyPI 2026-05-19

**Original target**: beat sparse-ho on libsvm Lasso benchmarks
(`breast-cancer`, `leukemia`, `rcv1.binary`).

**v0.1.0 status**: published to PyPI (`pip install sparho==0.1.0`),
documentation live on ReadTheDocs, library functionally complete (92
pytest, 11 cargo, mypy strict, single ABI3 wheel). Perf vs `LassoCV` at
release time:

- **`leukemia` (n ≪ p, dense)**: **1.3× faster** (23 s vs 30 s).
- **`breast-cancer` (n ≫ p, small)**: overhead-bound (0.24 s vs 0.01 s
  — both instant; neither interesting).
- **`rcv1.binary` (sparse, large)**: sparho **runs to completion** and
  finds a **better held-out MSE** than `LassoCV` (0.194 vs 0.225) by
  driving α below `LassoCV`'s grid floor (2.1·10⁻⁵ vs 1·10⁻⁴). Wall-time
  is 36× *slower* than `LassoCV` (433 s vs 12 s) because the inner Lasso
  cold-starts at very small α / large active set — addressed by the
  warm-start work in v0.2.

The "Nx faster" headline is a v0.2 deliverable; see the v0.2 section
below for the HOAG + warm-start jump to **15.4× vs `LassoCV`** on
`leukemia`. Hypergradient-CG stability is **fixed at v0.1** via ridge
regularization (`implicit_forward(..., ridge=ε)`, auto-scaled to
`10⁻¹⁰ · trace(M_AA)/|A|`, bit-identical results on the well-conditioned
benchmark across 8 orders of magnitude of ε).

What v0.1 actually ships:
- Correct gradient-based outer loop with FD-validated implicit-diff
  hypergradients.
- Vector-α support (`WeightedL1`) — something `LassoCV` cannot do.
- Sparse-X-first-class Rust kernels for prox + CSC matvec + restricted
  Hessian-vector products (`sparho._core.*`).
- sklearn + celer + callable adapters.
- `HeldOutMSE`, `HeldOutLogistic`, `CrossVal` criteria with sklearn
  `LassoCV` parity verified (`tests/test_criteria.py`).
- `GradDescent`, `LineSearch` outer optimizers.
- Log-space `grad_search` orchestration with full-data refit.
- PyPI-grade single-wheel install via maturin + PyO3 ABI3 py311.

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo scaffold (uv + maturin + cargo workspace + PyO3 ABI3 + CI) | ✅ done |
| 1 | Core types (`Problem`, `SolverResult`, `SearchState`, Protocols) | ✅ done |
| 2 | Rust kernels (`kernels`, `prox`, `csc`, `residual`) | ✅ done |
| 3 | Adapters (sklearn, celer, callable) | ✅ done |
| 4 | `ImplicitForward` hypergradient | ✅ done |
| 5 | Outer optimizers (`grad_descent`, `line_search`) | ✅ done |
| 6 | Criteria (`held_out_mse`, `held_out_logistic`, `cross_val`) | ✅ done |
| 7 | `grad_search` orchestration | ✅ done |
| 8 | Benchmarks (script + README; perf gap documented) | ✅ done |
| 8.5 | Hypergradient-CG ridge stabilization + NaN guards (rcv1 now runs) | ✅ done |
| 9 | Sphinx docs + sparse-ho migration guide | ✅ done (RTD live) |
| 10 | Release v0.1.0 to PyPI | ✅ done (published 2026-05-19) |

## v0.2 — HOAG outer loop, warm-start, celer

All v0.2 work is complete; tracked in `CHANGELOG.md [Unreleased]` pending
the v0.2.0 cut. The skein adapter originally scoped for v0.2 was deferred
to v0.3 (see below).

1. ✅ **Warm-start the inner solver across outer iterations.** Landed.
   `Solver.__call__` gained an optional keyword-only `x0` arg;
   `SklearnLasso`/`SklearnElasticNet`/`SklearnWeightedLasso`/`CelerLasso`/
   `CelerElasticNet` honor it via `warm_start=True`. `CrossVal` gained a
   `warm_start: bool` flag with a per-fold cache. ~1.9× wall-time vs
   cold-start on a dense 400×200 synthetic.
2. ✅ **HOAG outer loop (`sparho.hoag_search`).** Landed. Faithful port
   of Pedregosa-2016 HOAG with `+C·tol` slack acceptance, Lipschitz-proxy
   step adaptation, optional exponentially-decreasing inner-tol schedule,
   and a `max_step` trust-region cap. Replaces `LineSearch` (removed —
   Armijo's strict sufficient-decrease test stalls under warm-start +
   small ‖y‖²). End-to-end on `leukemia`: **8.6× vs `LassoCV`** (up
   from 1.3× at v0.1).
3. ✅ **Dense-path matvec fix in `_build_ls_data_matvec`** — dense
   designs no longer pay the per-iter `coo_tocsr` cost. `leukemia` wall
   1.05 s → 0.56 s (1.8×); `implicit_forward` 472 ms → 56 ms (8.4×).
   Now **15.4× vs `LassoCV`** on `leukemia`.
4. ✅ **Re-run the full benchmark suite** (`breast-cancer`, `leukemia`,
   `rcv1.binary`) with HOAG + warm-start + celer adapter. `benchmarks/
   README.md` and main `README.md` refreshed. Headline: `leukemia`
   **32.8× vs `LassoCV`** (up from 8.6× with sklearn; 1.3× at v0.1).
   `rcv1.binary` halved its wall (433 s → 211 s) — celer is ~1.65×
   faster than sklearn (347 s) at this scale — same quality win as
   v0.1 (sparho MSE 0.194 vs `LassoCV` 0.225, by walking α below the
   grid floor).
5. ✅ **Reproducibility tooling** — `benchmarks/lasso_libsvm.py` gained
   `--repeat N`, `--warmup K`, `--cooldown S` (interleaved sparho/LassoCV
   per rep, `gc.collect()` between iters, median + relative-spread
   reporting). With `--repeat 5 --cooldown 2` on the dense datasets,
   **sparho's own wall-time spread is 0.9–6.9 %**, within the 10 % plan
   target for detecting a sparho-side regression across releases.
   `LassoCV`'s own jitter is the residual: ~20 % on `leukemia`, ~16 %
   on `rcv1.binary`; `rcv1.binary`'s sparho row also jitters ~33 %
   because each sample is multi-minute and macOS throttles under
   sustained load. Further tightening to 10 % on the speedup *ratio*
   for `leukemia` and on either side for `rcv1.binary` is irreducible
   on macOS — it requires a Linux host with `taskset` + `pyperf
   --isolated`, which is future work. Documented in
   `benchmarks/README.md` § Reproducibility.

## v0.3 — sklearn-ecosystem wrappers, SURE, structural sparsity

Scoped from the 2026-05-20 feature-research synthesis
(`docs/feature_research.md`): one new algorithm differentiator (SURE),
one ergonomics unlock (sklearn wrappers + DataFrames), one structural-
sparsity extension (Group-L1), one new nonconvex adapter (skein), and
the model-surface multipliers from skglm/celer.

1. ⏳ **`sparho.adapters.skein`** — adapter for skein's nonconvex
   weighted/group penalties. Open question: does skein expose enough KKT
   state today (active set + KKT residual for the implicit-diff
   linearization), or do we add `kkt_residual` / `active_set` returns to
   its solvers first? Per CLAUDE.md, design patterns from sibling repos
   are not imported wholesale — the adapter is a thin protocol shim, not
   a port.
2. ✅ **`Sure` criterion** — landed. Finite-Difference Monte Carlo SURE
   after Deledalle 2014 (SUGAR): single deterministic δ probe, default
   `ε = 2σ/n^0.3`, two inner solves per eval, hypergradient is the sum
   of two `implicit_forward` calls (one per solve, with the second on
   the perturbed target `y + εδ`). Pure Python, no Rust, no union
   touch; refuses non-`SquaredLoss` at call time. Tests cover the
   closed-form Lasso-DOF concentration (`Σ_M FDMC-DOF / M → |support|`
   via the Zou-Hui-Tibshirani 2007 identity), FD-parity on the
   hypergradient, warm-start ≡ cold-start convergence, and
   denoising-style near-oracle α recovery. `GSURE` (general noise
   covariance) remains a v0.4+ candidate.
3. ✅ **sklearn-compatible wrapper estimators** — landed. `LassoHO`,
   `ElasticNetHO`, `LogisticRegressionHO` (`BaseEstimator +
   RegressorMixin/ClassifierMixin`, exposing `fit`/`predict`/`score`/
   `alpha_`/`coef_`/`intercept_`/`n_iter_`/`get_params`/`set_params`),
   targeting `check_estimator` compliance. Unblocks `Pipeline`,
   `GridSearchCV`, MLflow autolog, EconML/DoubleML, and the rest of the
   sklearn-ecosystem integrations downstream. The Solver Protocol that
   was the original "wait until stable" gate survived v0.2 unchanged.
   Includes pandas-DataFrame inputs + `feature_names_in_` /
   `get_feature_names_out` propagation so `coef_` round-trips to
   feature names (genomics/EHR/finance practitioners interpret `coef_`
   by name, not index).

   **Standardization decision (2026-05-20):** match sklearn-modern.
   Default `fit_intercept=True` (centering only; sparse-aware via
   separately-stored `X_mean`/`y_mean` and offset-adjusted matvecs — no
   CSC densification); **no `standardize` / `normalize` parameter**.
   Users wanting feature scaling use `Pipeline([StandardScaler(),
   LassoHO()])`, matching how the sklearn ecosystem post-1.0
   `normalize` deprecation already expects this to work. Rationale:
   keeps α* directly comparable to sklearn `Lasso` α*, avoids
   replicating the `normalize=` leakage mistake (sklearn#21238,
   sklearn#26359), and keeps the wrapper API minimal. The silent-
   underperformance failure mode on un-scaled data is mitigated by
   (a) a `UserWarning` in `fit()` when `np.ptp(X.std(axis=0)) > 10 *
   X.std(axis=0).mean()` recommending `StandardScaler` upstream, and
   (b) a new `docs/how-to/standardization-and-leakage.md` recipe
   covering the `Pipeline(StandardScaler, LassoHO)` pattern *and*
   warning that the internal `CrossVal` criterion sees post-scaler
   data (same leakage trap as `LassoCV` inside a Pipeline; recommend
   `HeldOutMSE` with pre-scaled splits when this matters). Glmnet-style
   `standardize=True` is explicitly *not* supported — the audience is
   sklearn refugees, not glmnet refugees.

   **v0.3 deferral (2026-05-20):** the "sparse-aware offset-adjusted
   matvecs" path is **not** in this cut. Dense `fit_intercept=True`
   uses upfront centering; sparse `fit_intercept=True` raises with an
   actionable redirect to `Pipeline([StandardScaler(with_mean=False),
   <estimator>(fit_intercept=False)])`. Plumbing `X_mean` through every
   solver adapter + `hypergrad._build_ls_data_matvec` is tracked as a
   v0.4 polish item — it touches the inner-solver / hypergradient stack
   meaningfully and isn't required for the sklearn-ecosystem integration
   value the wrappers deliver. The `check_estimator` suite passes
   modulo seven `sample_weight` checks declared known-skip (ROADMAP §M).
   `LogisticRegressionHO` does not support `fit_intercept=True` in v0.3
   — the log-odds intercept is a separate dof, not a feature-centering
   reduction; users add a constant column or accept the no-intercept
   parameterization.
4. ⏳ **`MultiTaskLasso` / Group-L1 (`Penalty = L21` or `GroupL1`)** —
   most-requested structural-sparsity extension. New `Penalty` union
   variant, Rust block-prox kernel, new `match` arms in
   `implicit_forward` and every dispatching criterion/adapter. Clean
   exercise of the closed-union design; required for the genomics
   (group-LD-aware) and finance (factor-group) use cases. Reference:
   ADMM-BDA 2024 (arXiv 2603.09546).
5. ⏳ **`SkglmAdapter` + missing celer adapters** —
   `SkglmAdapter` (MCP / SCAD / SLOPE / Group / Multitask / Huber /
   Poisson / Gamma) plus the celer paths we don't wrap today
   (`celer.MultiTaskLasso`, `celer.GroupLasso`,
   `celer.LogisticRegression`). Zero-algorithm-work way to multiply the
   model surface; mirrors the existing `CelerLasso` / `SklearnLasso`
   adapters. Each ~50–80 LOC.

## v0.1 work that already landed beyond plan scope

- **Hypergradient-CG stability**
  (plan: `~/.claude/plans/lets-brainstorm-on-ways-sleepy-meerkat.md`).
  `implicit_forward(..., ridge=ε)` with auto-scaled default; non-finite
  CG output produces a zero hypergradient with `RuntimeWarning` rather
  than propagating to a `α = nan` crash. NaN-guards added to
  `LineSearch`, `GradDescent`, and `grad_search`. Regression tests for
  near-singular Hessians (`tests/test_hypergrad.py::test_implicit_forward_ridge_*`)
  and NaN-handling (`tests/test_optimizer.py::test_*_holds_param_on_*`).

## Deferred / out of scope

These were considered and explicitly excluded from v0.1 in the plan
discussion:

- Forward / Backward / non-Forward Implicit hypergradient modes (only
  `ImplicitForward` ships at v0.1).
- Convenience sklearn-shaped wrapper classes (`LassoHO`, etc.). Revisit
  after the protocol is stable.
- Imaging operators (Gaussian blur, Radon, TV, wavelet).
- JAX / torch backends. GPU.
- SURE (Stein's Unbiased Risk Estimate) criterion — listed in original
  plan, dropped from Phase 6 since the v0.1 audience tunes via held-out
  validation, not unsupervised SURE. Revisit if a downstream user asks.
- Owner-paper features (`ImplicitVariational`, `TrustRegion`,
  `NonMonotoneLineSearch`) — depend on the non-Forward Implicit mode.

## Known unmet plan targets

For transparency, these explicit plan targets did not land at v0.1
(progress against them post-release is tracked in §v0.2):

- **"beat sparse-ho on libsvm Lasso benchmarks"** — sparse-ho was not
  installed for comparison; the bench runs against `LassoCV` instead.
  Status by dataset *at v0.1.0*: `breast-cancer` overhead-bound (both
  <1 s); `leukemia` **1.3× faster** (23 s vs 30 s); `rcv1.binary` runs
  and finds a better MSE (0.194 vs 0.225) but at 36× higher wall-time.
  Post-release with HOAG + warm-start + celer: `leukemia` **32.8×
  faster** (0.58 s vs 19.0 s); `rcv1.binary` wall halved (433 s → 211 s)
  but `LassoCV` still wins on raw wall time (22.6 s) — sparho's win on
  rcv1 is the better held-out MSE, not the clock.
- **"benchmarks reproduce within 10%"** — hit at v0.2 for sparho's own
  wall-time spread on the dense datasets (0.9–6.9 % with `--repeat 5
  --cooldown 2`). The speedup *ratio* and `rcv1.binary` are bounded by
  macOS thermal jitter on `LassoCV` and long-burn sparho runs — see
  v0.2 §5 for the honest residual.
