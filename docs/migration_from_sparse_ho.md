# Migrating from sparse-ho

`sparho` is a clean-break successor to
[`sparse-ho`](https://github.com/QB3/sparse-ho), not a drop-in replacement.
There is no `compat` module — porting code is a manual rewrite. This page
is the translation table.

## The shape of the rewrite

sparse-ho's outer call took four objects:

```python
sparse_ho.grad_search(algo, criterion, model, optimizer, X, y, alpha0, monitor)
```

sparho rolls `algo` (hypergradient) and `optimizer` (outer step rule) into
the search function itself, and folds `model` (which estimator) into the
`Solver` adapter you pass:

```python
sparho.hoag_search(
    problem,         # ← Problem(datafit, penalty, X, y)
    hp0=alpha0,      # ← in log α space internally; pass α > 0
    solver=...,      # ← was `model` + adapter choice
    criterion=...,   # ← was `criterion`
    hypergrad=...,   # ← was `algo`, default `implicit_forward`
)
```

`monitor` becomes the return value: `result.history` is an immutable tuple
of `IterationRecord`s. Use `grad_search` instead of `hoag_search` for the
naive fixed-`lr` outer loop (sparse-ho's `GradientDescent` optimizer).

## Translation table

### Models → `Problem` + adapter

| sparse-ho | sparho | Notes |
|---|---|---|
| `Lasso(X, y, estimator=None)` | `Problem(SquaredLoss(), L1(), X, y)` + `SklearnLasso()` | Default. Use `CelerLasso()` from the `[celer]` extra for sparse-X. |
| `Lasso(..., estimator=celer.Lasso(...))` | `Problem(SquaredLoss(), L1(), X, y)` + `CelerLasso(tol=...)` | |
| `ElasticNet(X, y, estimator=...)` | `Problem(SquaredLoss(), ElasticNet(rho), X, y)` + `SklearnElasticNet()` | sparse-ho's `l1_ratio` is sparho's `rho`. |
| `WeightedLasso(X, y, ...)` | `Problem(SquaredLoss(), WeightedL1(), X, y)` + `SklearnWeightedLasso()` | Per-feature α. |
| `WeightedElasticNet(X, y, ...)` | not yet at v0.1 | Open a discussion if you need it. |
| `SparseLogreg(X, y, ...)` | `Problem(LogisticLoss(), L1(), X, y)` + `SklearnLogisticRegression()` | sparse-ho's `C = 1/α`; sparho's `α` matches the math directly. |
| `SVM(...)`, `SVR(...)`, `SimplexSVR(...)` | not at v0.1 | `SmoothHinge` deferred. |
| `Ridge(...)` | not at v0.1 | The hypergradient is closed-form; out of scope. |

### Algorithms → `hypergrad=`

| sparse-ho `algo` | sparho `hypergrad=` | Notes |
|---|---|---|
| `ImplicitForward(...)` | `implicit_forward` (the default) | The only mode at v0.1. |
| `Implicit(...)`, `ImplicitVariational(...)` | not at v0.1 | Deferred. |
| `Forward(...)`, `Backward(...)` | not at v0.1 | Unrolled modes deferred. |

### Criteria

| sparse-ho | sparho | Notes |
|---|---|---|
| `HeldOutMSE(idx_train, idx_val)` | `HeldOutMSE(idx_train, idx_val)` | int32 indices required. |
| `HeldOutLogistic(idx_train, idx_val)` | `HeldOutLogistic(idx_train, idx_val)` | `y ∈ {−1, +1}` (same convention). |
| `CrossVal(cv, criterion=HeldOutMSE)` | `CrossVal.kfold(n_samples, k=...)` or `CrossVal(folds=..., base=HeldOutMSE)` | sparho's `CrossVal` is a frozen dataclass; build it once and reuse. Opt-in `warm_start=True` reuses per-fold `β*` across outer iters. |
| `HeldOutSmoothedHinge(...)` | not at v0.1 | SVM/SVR family deferred. |
| `FiniteDiffMonteCarloSure(...)` | not at v0.1 | SURE deferred — see "Out of scope" in `ROADMAP.md`. |

### Optimizers → search function

| sparse-ho optimizer | sparho equivalent | Notes |
|---|---|---|
| `GradientDescent(n_outer, step_size=lr)` | `grad_search(..., n_iter=n_outer, lr=lr)` | Plain GD baseline. |
| `LineSearch(n_outer)` | `hoag_search(..., n_iter=n_outer)` | sparho's recommended default — Lipschitz-adaptive steps + inner-tolerance scheduling. The original Armijo `LineSearch` is **not** a sparho v0.1 deliverable; HOAG subsumes it in practice. |
| `Adam(...)`, `TrustRegion(...)`, `NonMonotoneLineSearch(...)` | not at v0.1 | Owner-paper features; depend on `ImplicitVariational`. |

### Monitor → `SearchResult.history`

sparse-ho's `Monitor` mutated as the loop ran:

```python
monitor = Monitor()
sparse_ho.grad_search(algo, crit, model, opt, X, y, alpha0, monitor)
alphas, mses = monitor.alphas, monitor.objs
```

sparho returns an immutable `SearchResult`:

```python
result = sparho.hoag_search(problem, hp0=alpha0, solver=..., criterion=...)
alphas = [r.hyperparam for r in result.history]
mses = [r.value for r in result.history]
best_alpha = result.best_hyperparam
best_coef = result.best_coef        # refit on the full problem
```

## A worked example

sparse-ho:

```python
from celer import Lasso as CelerLasso
from sklearn.datasets import make_regression
from sparse_ho import grad_search
from sparse_ho.algo import ImplicitForward
from sparse_ho.criterion import HeldOutMSE
from sparse_ho.models import Lasso
from sparse_ho.optimizers import LineSearch
from sparse_ho.utils import Monitor

X, y = make_regression(n_samples=300, n_features=100, noise=1.0, random_state=0)
idx_train, idx_val = np.arange(200), np.arange(200, 300)

model = Lasso(X, y, estimator=CelerLasso(fit_intercept=False))
algo = ImplicitForward(criterion="HO")
criterion = HeldOutMSE(idx_train, idx_val)
optimizer = LineSearch(n_outer=30)
monitor = Monitor()

grad_search(algo, criterion, model, optimizer, X, y, alpha0=1e-2, monitor=monitor)
print(monitor.alphas[-1], monitor.objs[-1])
```

sparho:

```python
import numpy as np
from sklearn.datasets import make_regression

from sparho import HeldOutMSE, L1, Problem, SquaredLoss, hoag_search
from sparho.adapters.celer import CelerLasso

X, y = make_regression(n_samples=300, n_features=100, noise=1.0, random_state=0)
idx_train = np.arange(200, dtype=np.int32)
idx_val = np.arange(200, 300, dtype=np.int32)

result = hoag_search(
    Problem(SquaredLoss(), L1(), X, y),
    hp0=1e-2,
    solver=CelerLasso(tol=1e-8),
    criterion=HeldOutMSE(idx_train, idx_val),
    n_iter=30,
)
print(result.best_hyperparam, result.history[-1].value)
```

## Behavior differences to know about

- **`α` lives in log space.** Both `grad_search` and `hoag_search` step in
  `θ = log α`; `hp0` must be strictly positive. The chain rule
  `dC/dθ = dC/dα · α` is applied internally. sparse-ho left this to the
  user via `log_alpha_max`.
- **Full-data refit at the end.** `SearchResult.best_coef` is the inner
  solver run on the full `Problem` at `best_hyperparam`. sparse-ho left
  `monitor.alphas[-1]` as the only output; a refit was on the user.
- **Sparse-X is CSC, not CSR.** Convert before constructing `Problem`.
- **No `Monitor`.** The history is immutable. If you need streaming
  observation, wrap the criterion (or the solver) yourself.
- **No `sure` criterion.** Dropped at v0.1; the v0.1 audience tunes
  validation, not unsupervised SURE. Revisit if asked.
- **No grid-search fallback.** sparse-ho had a grid `ho.grid_search`;
  sparho doesn't ship one. Use `sklearn.linear_model.LassoCV` if you
  want a grid baseline.

## What's not yet ported

These were sparse-ho features that v0.1 deliberately leaves on the floor.
Some are slated for v0.2 (`skein` adapter, more datafits/penalties); others
are out of scope (`SVM/SVR`, imaging operators). See `ROADMAP.md` for the
full picture.
