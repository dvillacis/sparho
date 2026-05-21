# API stability

`sparho` is pre-alpha; the public API may change between minor versions
until v1.0. This page declares which surfaces are committed to in the
current release line and which are not, so downstream packages know what
they can lean on.

## Stable

These names and signatures are committed to within a given minor version.
Breaking changes require a deprecation cycle: a new release introduces
the replacement and emits `DeprecationWarning` on the old surface; the
old surface is removed in the next minor (not patch) bump.

- `sparho.Problem`, `sparho.SquaredLoss`, `sparho.LogisticLoss`.
- The `sparho.Penalty` union and its variants `L1`, `ElasticNet`,
  `WeightedL1`, `GroupL1` (including the `from_labels` factory).
- The `sparho.Solver` protocol — `(problem, hyperparam) -> SolverResult`,
  with optional keyword-only `x0` and `tol`.
- `sparho.SolverResult`, `sparho.SearchResult`, `sparho.SearchState`,
  `sparho.IterationRecord` *as dataclasses* — field names, types, and
  default values. See the experimental section below for
  `IterationRecord.extras`.
- `sparho.grad_search`, `sparho.hoag_search` — positional / keyword
  arguments, including the v0.4 `callback` kwarg.
- `sparho.implicit_forward`.
- The three sklearn-compatible wrapper estimators
  (`sparho.LassoHO`, `sparho.ElasticNetHO`, `sparho.LogisticRegressionHO`)
  and their fitted attributes (`coef_`, `intercept_`, `alpha_`,
  `n_iter_`, `feature_names_in_`, `n_features_in_`,
  `search_result_`, `classes_` for the classifier).

## Experimental

Subject to change in any release, including patch. Pin your version if
you depend on these.

### `IterationRecord.extras` schema

`IterationRecord.extras` is a `Mapping[str, object]` populated by the
search loops. Keys and value types may be added, removed, or renamed
between releases. As of v0.4 the following keys are populated when
available:

| key | type | populated by | meaning |
|---|---|---|---|
| `cg_status` | `Literal["ok", "nonconvergence", "nonfinite"]` | always | summary of the inner CG solve inside `implicit_forward`. `"nonfinite"` means CG returned NaN/Inf (the search treats this as a zero hypergradient and stalls one iter); `"nonconvergence"` means CG returned a finite but unconverged result. |
| `inner_dual_gap` | `float` | when the criterion surfaces it (`HeldOutMSE` / `HeldOutLogistic` directly; `CrossVal` takes max across folds; `Sure` takes max across its two solves) | inner-solver convergence proxy. Compare against the configured `inner_tol`. |
| `step_size` | `float` | `hoag_search` only | `1 / L_estimate` — the θ-space step magnitude that drove this iter's update. |
| `L_estimate` | `float` | `hoag_search` only | Lipschitz proxy used by HOAG's acceptance test. Doubles on rejected steps, multiplied by 0.95 on accepted steps. |

Future keys under consideration but not committed: `n_inner_iter` per
solve, `active_set_size`, `objective` (current outer-loop criterion
value). Treat anything not listed above as private.

### Adapter internals

`sparho.adapters` re-exports `SklearnLasso`, `CelerLasso`, etc. as a
convenience. The internal layout (`adapters._common`, the per-adapter
modules) is not stable.

### Hypergradient ridge default

`implicit_forward(..., ridge=None)` resolves to
`1e-10 · trace(M_AA) / |A|`. The coefficient and the auto-scale formula
may change as we accumulate more empirical evidence on near-singular
designs.

## Private

Anything under a leading underscore (`_VerbosePrinter`,
`_extras_from_warnings`, etc.) and the entire `sparho._core` extension
module. Calling these from outside the package is unsupported.

## Optional dependencies

The `[celer]` extra is tracked by a dedicated CI job (`ci-celer`). The
floor-version pins are tracked by `ci-min-deps`. Bumps to either are
documented in `CHANGELOG.md`.
