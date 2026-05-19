# Quickstart

This page walks the held-out Lasso example end-to-end. The goal: find the
Lasso regularization strength `α` that minimizes mean-squared error on a
held-out validation set, using one gradient-based outer search instead of a
grid sweep.

## 1. Build a `Problem`

A bilevel problem is the inner loss + penalty + data; the outer
hyperparameter `α` is **not** stored on it.

```python
import numpy as np
from sklearn.datasets import make_regression

from sparho import L1, Problem, SquaredLoss

X, y = make_regression(n_samples=300, n_features=100, n_informative=10,
                       noise=1.0, random_state=0)
problem = Problem(SquaredLoss(), L1(), X, y)
```

`SquaredLoss` and `L1` are dataclass *tags* — the algorithms `match` on them
to dispatch the right Rust kernels. The full v0.1 set is
`SquaredLoss | LogisticLoss` × `L1 | ElasticNet | WeightedL1`.

## 2. Pick an inner solver

`SklearnLasso` wraps `sklearn.linear_model.Lasso` to satisfy the
{py:class}`sparho.Solver` protocol. For sparse-X you'd typically prefer
`CelerLasso` from the `[celer]` extra.

```python
from sparho.adapters import SklearnLasso

solver = SklearnLasso(tol=1e-8)
```

The tight `tol` matters: criteria that rely on tiny coefficient
movements between outer iters will stall if the inner solver short-circuits
on a loose tolerance check.

## 3. Pick an outer criterion

```python
from sparho import HeldOutMSE

rng = np.random.default_rng(0)
idx = rng.permutation(X.shape[0]).astype(np.int32)
idx_train, idx_val = idx[:200], idx[200:]
criterion = HeldOutMSE(idx_train, idx_val)
```

The criterion owns the train/val split. It tells the inner solver to train
on `idx_train` and evaluates the held-out MSE on `idx_val`. For K-fold use
`CrossVal.kfold(problem.n_samples, k=5)`.

## 4. Run the outer search

```python
from sparho import hoag_search

result = hoag_search(
    problem,
    hp0=1e-2,
    solver=solver,
    criterion=criterion,
    n_iter=30,
)

print(f"best α = {result.best_hyperparam:.4g}")
print(f"history length = {len(result.history)}")
print(f"converged = {result.converged}")
```

`hoag_search` runs in `θ = log α` space (so `α` stays positive without
projection), adapts its step size from a Lipschitz proxy, and tolerates
noise from the inner solver via a slack term in its acceptance test. After
the loop it refits the inner solver on the **full** problem at the best `α`
seen and stuffs that coefficient vector into `result.best_coef`.

`grad_search` is the simpler alternative — fixed learning rate, no step
adaptation; useful as a baseline. Both functions share the same signature.

## 5. Inspect the trajectory

```python
import matplotlib.pyplot as plt

xs = [r.hyperparam for r in result.history]
ys = [r.value for r in result.history]
plt.semilogx(xs, ys, marker="o")
plt.xlabel("α"); plt.ylabel("held-out MSE"); plt.show()
```

Each `IterationRecord` carries the hyperparameter, criterion value, and
gradient norm at one outer iter. The history is an immutable
`tuple[IterationRecord, ...]` — there is no mutable monitor.

## What to read next

- [Concepts](concepts.md) — what's actually being computed at each step.
- [Extending sparho](protocols.md) — adding a custom solver, criterion, or
  penalty.
- [Migration from sparse-ho](migration_from_sparse_ho.md) — translation
  table if you're coming from the older 4-tuple API.
- The [API reference](api/index.md) and the
  [examples gallery](examples_built/index.rst).
