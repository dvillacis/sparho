---
orphan: true
---

# Feature research — v0.3 and beyond

Research compiled 2026-05-20, day after the v0.1.0 release and the v0.2
perf-story milestone (HOAG + warm-start + celer + reproducibility
harness). Methodology: three parallel literature/landscape threads —
(1) competitive audit of sibling/adjacent libraries, (2) academic
literature scan covering 2022–2026, (3) practitioner-needs review
grounded in GitHub issues and real-world use cases. Findings below are
filtered against the CLAUDE.md design pillars: closed tagged unions for
`Datafit` / `Penalty`, scipy-stack-native (no JAX/torch/GPU), no legacy
compatibility shims, single `step()` outer-optimizer protocol.

## Convergent high-confidence picks

Each appeared on multiple threads with strong fit-with-sparho. The v0.3
section of `ROADMAP.md` adopts A–D alongside the existing skein adapter.

### A. `SURE` / `GSURE` criterion *(literature + competitive)*

Stein's Unbiased Risk Estimator — closed-form upper-level objective
requiring `∂²L/∂y²` (closed-form for `SquaredLoss`) and the trace of the
active-set projection that `implicit_forward` already builds. Uniquely
enables tuning when no held-out set exists (denoising, signal recovery,
single-fold scenarios) — the cleanest differentiator vs `LassoCV`. Was
on sparse-ho's flagship feature list as `FiniteDiffMonteCarloSure`;
dropped from sparho v0.1's Phase 6 in the plan discussion.

- **Fit:** pure-Python new `Criterion`. No Rust, no union touch.
- **Effort:** small. Reuses restricted Jacobian.
- **References:** sparse-ho `FiniteDiffMonteCarloSure`; Boyd group 2023
  *"Tractable Evaluation of SURE with Convex Regularizers"*.

### B. sklearn-compatible wrapper estimators *(practitioner + competitive)*

`LassoHO`, `ElasticNetHO`, `LogisticRegressionHO` — `BaseEstimator +
RegressorMixin/ClassifierMixin` subclasses exposing `fit(X, y) -> self`,
`predict`, `score`, `alpha_`, `coef_`, `intercept_`, `n_iter_`,
`get_params`/`set_params`. `check_estimator` compliance is the contract
that lets sparho drop into `Pipeline`, `GridSearchCV`,
`cross_val_score`, `clone`, `permutation_importance`, MLflow autolog,
and EconML/DoubleML without code changes downstream.

This was the single biggest adoption blocker called out by the
practitioner thread. `celer.Lasso` and `skglm.Lasso` already subclass
`sklearn.linear_model.Lasso` directly — sparho is the outlier today.
The wrapper class was explicitly deferred in the original ROADMAP
"until protocol stable" — protocol shipped at v0.1 and survived v0.2
unchanged, so the deferral is now resolvable.

- **Fit:** additive layer on top of the existing orchestration.
- **Effort:** medium. `check_estimator` surfaces dozens of edge cases
  (1-D y, sparse y, sample_weight, multi-output, dtype preservation).
- **References:** scikit-learn-contrib project-template; *Developing
  scikit-learn estimators*; QB3/sparse-ho issues #114, #115 (upstream
  never closed the wrapper-API question — debt is inherited).
- **Standardization decision (2026-05-20):** match sklearn-modern.
  Default `fit_intercept=True` (sparse-aware centering); no
  `standardize` / `normalize` parameter. Users wanting feature scaling
  use `Pipeline([StandardScaler(), LassoHO()])`. Rationale: keeps α*
  comparable to sklearn `Lasso` α*, avoids the `normalize=` leakage
  mistake (sklearn#21238, sklearn#26359), keeps the API minimal. The
  silent-failure mode on un-scaled data is mitigated by a runtime
  `UserWarning` (uneven column scales) plus a recipe doc covering both
  the standardization pattern and the residual leakage trap in
  `Pipeline(StandardScaler, LassoHO_internal_CV)`. Glmnet-style
  default-standardize is explicitly out — audience is sklearn refugees,
  not glmnet refugees.

### C. `MultiTaskLasso` / Group-L1 (`Penalty = L21` or `GroupL1`) *(literature + competitive)*

Most-requested structural-sparsity extension. Both skglm and celer ship
it; explicit bilevel precedent (ADMM-BDA 2024, HAL-04183917). Critical
for the genomics (group-LD-aware penalties) and finance (factor-group
selection) practitioner use cases.

- **Fit:** new `Penalty` variant; Rust block-prox kernel; new `match`
  arms in `implicit_forward`, every criterion/adapter that dispatches.
  Clean exercise of the closed-union design.
- **Effort:** medium-large.
- **References:** ADMM-based Bilevel Descent Aggregation (arXiv
  2603.09546); HAL-04183917 (bilevel mixed-binary SGL).

### D. `SkglmAdapter` + unwrapped celer paths *(competitive, practitioner-adjacent)*

Adapter for skglm (MCP / SCAD / SLOPE / Group / Multitask / Huber /
Poisson / Gamma) and the missing celer adapters (`celer.MultiTaskLasso`,
`celer.GroupLasso`, `celer.LogisticRegression`). Zero-algorithm-work way
to multiply the model surface; consistent with the `adapters/` pillar.

- **Fit:** mechanical, mirrors `CelerLasso` / `SklearnLasso`.
- **Effort:** small. Each adapter ~50–80 LOC.
- **References:** `skglm` (scikit-learn-contrib); `celer` repo.

## Algorithm / research wins (literature-driven, v0.4+ candidates)

### E. Ehrhardt–Roberts a posteriori inner-tol schedule

Replaces HOAG's Pedregosa-2016 *a priori* geometric schedule with a
*computable* error bound on the hypergradient. Defensible stopping
criterion, citation magnet, no algorithmic risk. Upgrades `hoag_search`
in place; pure orchestration.

- **Fit:** small Python diff to `hoag_search` and the inner-tol
  pipeline.
- **References:** Ehrhardt & Roberts, IMA J. Appl. Math. 2024 (arXiv
  2301.04764).

### F. Salehi–Ehrhardt adaptive backtracking

Backtracking line search that *jointly* adapts outer step size and inner
accuracy when the hypergradient Lipschitz constant is unknown. Tested
on TV + multinomial logistic.

- **Fit:** new outer optimizer satisfying the existing `step()`
  protocol.
- **References:** Salehi et al., SIAM J. Math. Data Sci. 2024 (arXiv
  2308.10098).

### G. Neumann-series `HypergradMode`

Truncated `(I − ηH)^k` series as alternative to restricted CG.
Autograd-free; cheaper than CG when the restricted Hessian is
ill-conditioned. Matches the betty/torchopt API expectation. ROADMAP's
"Deferred / out of scope" lists Forward/Backward/Implicit-non-restricted
— Neumann is a *new* mode that doesn't require lifting the
implicit-only pillar.

- **Fit:** new `HypergradMode` slot. Pure Python or thin Rust.
- **References:** betty, torchopt.

### H. `Datafit = HuberLoss`

Smooth datafit with closed-form Jacobian. Opens robust-regression
direction; active research thread (arXiv 2506.12591, 2025 — auto-tuning
sparse Huber).

- **Fit:** new `Datafit` variant + Rust kernel.
- **Effort:** medium.

### I. `Penalty = SortedL1` (SLOPE)

Single Rust prox kernel (sorted-then-isotonic). Tests the framework on
a permutation-dependent active set — useful stress test before SGL
ships.

- **Effort:** medium.

### J. Multinomial / multiclass sparse logistic

sparse-ho's `LogisticMulticlass` criterion + `SparseLogreg` with
multinomial head. Closes the biggest missing-model gap; unlocks EHR /
TF-IDF text-classification practitioner use cases.

- **Fit:** new `Datafit = MultinomialLogistic` variant; current
  `LogisticLoss` is binary only.
- **Effort:** medium.

## Practitioner ergonomics (sklearn ecosystem)

### K. `fit_intercept=True` + sparse-aware centering

**Decided (2026-05-20):** `fit_intercept=True` default; no
`standardize` parameter. Centering uses separately-stored
`X_mean`/`y_mean` + offset-adjusted matvecs so sparse CSC stays sparse
on the hot path. Standardization is handled by the user via
`Pipeline([StandardScaler(), LassoHO()])`; the wrapper emits a
`UserWarning` at fit-time on uneven column scales (`np.ptp(X.std(
axis=0)) > 10 * X.std(axis=0).mean()`) to mitigate the silent-failure
mode. See ROADMAP v0.3 §3 for the full decision record.

- **Effort:** medium (Rust path for sparse-aware centering).
- **References:** scikit-learn discussion #21238 (`normalize`
  deprecation); issue #26359 (LassoCV leakage).

### L. pandas DataFrame inputs + `feature_names_in_` / `get_feature_names_out`

Required to live downstream of `ColumnTransformer`. GWAS / EHR / finance
practitioners interpret `coef_` by name, not index.

- **Effort:** small.

### M. `sample_weight=` in `fit`

Heavily requested for class imbalance (EHR risk scoring) and survey
weighting.

- **Effort:** medium — touches kernels and criterion paths.

### N. `n_jobs` / joblib parallelism for `CrossVal`

Folds are embarrassingly parallel between outer steps but share
warm-starts within a step — real design call.

- **Effort:** medium.

### O. Convergence diagnostics

sklearn-compatible `ConvergenceWarning`, `SearchResult.converged: bool`,
`.message: str`. Practitioners grep logs for `ConvergenceWarning`.

- **Effort:** small.

## Documentation patterns

### P. Decision-tree page: "sparho vs `LassoCV` vs skglm vs `celer` vs Optuna"

Keyed on `(p, sparsity, n_HPs, criterion type)`. Converts the curious
sklearn user.

### Q. Recipe-style how-tos

Separate `docs/how-to/` tree from the sphinx-gallery examples. Short,
copy-pasteable, one task per page:

- *"Tune α for sparse logistic on TF-IDF features"*
- *"Plug sparho into DoubleML / EconML"*
- *"Get a reproducible α across runs"* (harness landed at v0.2)
- *"Migrate a sparse-ho project"* (extend the existing migration guide
  with a "common breakages" section as users report them)

### R. Comparison-features table

sparho × celer × skglm × sklearn × glmnet. Rows = features (sparse-X,
weighted-L1, gradient HP search, sklearn-compatible, MLflow autolog,
fit_intercept, sample_weight). Columns = libraries. The screenshot
reviewers paste into Slack.

### S. Benchmark gallery beyond libsvm

Add at least one GWAS-scale simulated dataset (n ≈ 10⁵, p ≈ 10⁶, very
sparse one-hot) and one text-TF-IDF dataset (e.g. 20newsgroups binary).
Establishes the speed claim outside the libsvm reference set.

## Use-case landscape (practitioner thread, abridged)

- **GWAS / eQTL** — n ≈ 10⁴–10⁶, p ≈ 10⁵–10⁷ SNPs, CSC-sparse. Permutation-
  assisted CV is the brutal-slow incumbent. sparho's CSC-in-Rust +
  gradient HP search is the clearest "10–100× vs `LassoCV`" pitch.
  Needs `WeightedL1` (shipped) + stable `coef_` ↔ SNP-ID round-trip.
- **Single-cell RNA-seq feature selection** — ElasticNet with 2-D HP
  search; gradient dominates 2-D grid.
- **EHR risk scoring** — sparse logistic + sample weights + clinical
  priors via `WeightedL1`. Contingent on the wrapper estimator.
- **Double/debiased ML for causal inference** — `econml.SparseLinearDML`
  currently uses CV-tuned Lasso per fold; sparho fits naturally as the
  nuisance-Lasso backend.
- **Marketing mix modeling** — small dataset; sparho's win is
  reproducibility and intercept/scale handling rather than speed.
- **Quantitative finance factor models** — small p, vendor-risk-averse;
  win is reputational rather than perf.
- **Text classification with TF-IDF + sparse logistic** — depends on
  `LogisticRegressionHO`.

## Explicitly flagged — tempting but conflicts with design pillars

- **JAXopt `custom_root` decorator (open extension API)** — conflicts
  with the closed-tagged-union pillar. Re-evaluate only if external
  users request escape-hatch extensibility for custom `(datafit,
  penalty)` pairs that don't justify a union variant.
- **Weighted Graphical Lasso, Trace-norm / nuclear-norm low-rank** —
  matrix-valued β doesn't fit sparho's vectorized internals. Defer
  indefinitely.
- **MCP / SCAD / log-sum (nonconvex sparse)** — `skein` is the v0.3
  home.
- **Generalized Lasso / TV penalty** — needs dual reformulation;
  imaging-adjacent (ROADMAP excludes imaging).
- **Adam outer optimizer** — sparse-ho had it; HOAG covers
  adaptive-step needs and Salehi (F) covers tuning-free.
- **Forward / Backward hypergradient modes** — sparse-ho had these;
  competitive audit suggests them as ground-truth baselines for testing
  only; ROADMAP excludes them as full citizens at v0.1+.
- **Multi-Objective Bilevel (Pareto frontier hypergrad)** — premature;
  waits until 2+ stable criteria are commonly composed.
- **GPU / out-of-core / JAX / torch backends** — pillar exclusion.

## Sources

### Competitive

- sparse-ho — https://github.com/QB3/sparse-ho ; https://qb3.github.io/sparse-ho/
- JAXopt `implicit_diff` — https://jaxopt.github.io/stable/implicit_diff.html
- betty — https://github.com/leopard-ai/betty
- torchopt — https://github.com/metaopt/torchopt
- skglm — https://contrib.scikit-learn.org/skglm/
- celer — https://github.com/mathurinm/celer

### Literature

- Grazzi, Pontil & Salzo, *Nonsmooth Implicit Differentiation*, ICML 2024 — arXiv 2403.11687
- Ehrhardt & Roberts, *Analyzing Inexact Hypergradients for Bilevel Learning*, IMA J. Appl. Math. 2024 — arXiv 2301.04764
- Salehi, Ehrhardt et al., *An Adaptively Inexact First-Order Method for Bilevel Optimization*, SIAM J. Math. Data Sci. 2024 — arXiv 2308.10098
- Bolte & Pauwels, *One-step differentiation of iterative algorithms*, 2023 — arXiv 2305.13768
- Boyd group, *Tractable Evaluation of SURE with Convex Regularizers*, 2023
- ADMM-based Bilevel Descent Aggregation for SGL — arXiv 2603.09546
- *Sparse Reduced Rank Huber Regression*, PMC 10812838, 2024
- *An Easily Tunable Approach to Robust and Sparse High-Dim Linear Regression*, 2025 — arXiv 2506.12591
- Bertrand et al., *Implicit differentiation for fast HP selection in non-smooth convex learning*, JMLR 2022

### Practitioner

- scikit-learn discussion #21238 (`normalize` deprecation)
- scikit-learn issue #26359 (LassoCV leakage in Pipeline)
- scikit-learn issue #24877 (LassoCV vs GridSearchCV α* disagreement)
- QB3/sparse-ho issues #114, #115 (wrapper-API debt)
- Yang et al., *permutation-assisted lasso tuning for GWAS*, Bioinformatics 2020
- EconML `SparseLinearDML` — https://www.pywhy.org/EconML/spec/estimation/dml.html
- MLflow sklearn integration — https://mlflow.org/docs/latest/ml/traditional-ml/sklearn/
