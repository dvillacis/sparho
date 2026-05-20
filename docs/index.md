# sparho

Nonsmooth bilevel hyperparameter optimization via implicit differentiation.

`sparho` is a maintained, performant successor to
[`sparse-ho`](https://github.com/QB3/sparse-ho) (ICML 2020, dormant since
2022). It tunes hyperparameters of non-smooth estimators (Lasso, ElasticNet,
weighted Lasso, sparse logistic regression) by computing the hypergradient
via implicit differentiation rather than grid or random search.

```{warning}
Pre-alpha. The public API may change between minor versions until v1.0.
```

## At a glance

- **One core type.** `Problem(datafit, penalty, design, target)` plus free
  functions; no inheritance tower.
- **Implicit-only at v0.1.** A single hypergradient mode (`implicit_forward`).
- **Sparse-X first class.** CSC is iterated directly in Rust kernels — no
  densification anywhere on the hot path.
- **Adapters, not wrappers.** `SklearnLasso`, `CelerLasso`, and `as_solver`
  bring third-party fitters under the `Solver` protocol.
- **Two outer loops.** `grad_search` is plain gradient descent in `log α`
  space; `hoag_search` is the (Pedregosa 2016) algorithm with
  Lipschitz-adaptive steps and inner-tolerance scheduling.

## Quickstart

```python
import numpy as np
from sklearn.datasets import make_regression

from sparho import (
    HeldOutMSE,
    L1,
    Problem,
    SquaredLoss,
    hoag_search,
)
from sparho.adapters import SklearnLasso

X, y = make_regression(n_samples=200, n_features=80, noise=1.0, random_state=0)
n_train = 150
idx_train = np.arange(n_train, dtype=np.int32)
idx_val = np.arange(n_train, X.shape[0], dtype=np.int32)

problem = Problem(SquaredLoss(), L1(), X, y)
result = hoag_search(
    problem,
    hp0=1e-2,
    solver=SklearnLasso(tol=1e-8),
    criterion=HeldOutMSE(idx_train, idx_val),
    n_iter=30,
)
print(result.best_hyperparam, result.converged)
```

See [Quickstart](quickstart.md) for an annotated walkthrough,
[Concepts](concepts.md) for the math, and the
[Gallery](examples_built/index.rst) for runnable end-to-end examples.

## Contents

```{toctree}
:caption: User guide
:maxdepth: 2

installation
quickstart
concepts
protocols
migration_from_sparse_ho
```

```{toctree}
:caption: How-to
:maxdepth: 1

how-to/standardization-and-leakage
```

```{toctree}
:caption: API reference
:maxdepth: 2

api/index
```

```{toctree}
:caption: Examples
:maxdepth: 1

examples_built/index
```
