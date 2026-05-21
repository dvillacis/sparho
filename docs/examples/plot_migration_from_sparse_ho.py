"""
Migrating from sparse-ho to sparho
==================================

Runnable companion to ``docs/migration_from_sparse_ho.md``. We don't
import ``sparse_ho`` (it's optional, not on PyPI) — instead each section
shows the sparse-ho idiom in a comment and the sparho equivalent in
runnable code. The prose guide has the full translation table.

The problem: tune the Lasso α on a 200-sample / 50-feature regression
with a held-out validation split, compute a hypergradient by implicit
differentiation, drive the outer search with HOAG, and refit on the full
problem at the chosen α*.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

from sparho import (
    HeldOutMSE,
    L1,
    Problem,
    SquaredLoss,
    hoag_search,
)
from sparho.adapters import SklearnLasso

# %%
# Data + split: 200 samples × 50 features, 150 train / 50 val. The split
# is what `HeldOutMSE` reads — the inner solver sees only the train slice;
# the criterion evaluates on the val slice.
X, y = make_regression(
    n_samples=200,
    n_features=50,
    n_informative=8,
    noise=1.0,
    random_state=0,
)
idx_train = np.arange(150, dtype=np.int32)
idx_val = np.arange(150, 200, dtype=np.int32)

# %%
# Step 1 — pick the **model**.
#
# sparse-ho:
#
# .. code-block:: python
#
#     model = sparse_ho.Lasso(X, y, estimator=None)
#
# sparho splits "what's being optimized" from "how it's solved": a
# ``Problem`` (the bilevel inner problem in math) plus a ``Solver`` (an
# adapter wrapping the actual numerical fitter).
problem = Problem(SquaredLoss(), L1(), X, y)
solver = SklearnLasso(tol=1e-8)

# %%
# Step 2 — pick the **criterion**.
#
# sparse-ho:
#
# .. code-block:: python
#
#     criterion = sparse_ho.HeldOutMSE(idx_train, idx_val)
#
# Same name, same idea, same int32-index convention. ``CrossVal`` and
# ``HeldOutLogistic`` carry over too — see the translation table in
# ``migration_from_sparse_ho.md``.
criterion = HeldOutMSE(idx_train=idx_train, idx_val=idx_val)

# %%
# Step 3 — pick the **hypergradient algorithm**.
#
# sparse-ho:
#
# .. code-block:: python
#
#     algo = sparse_ho.ImplicitForward(tol_jac=1e-8, n_iter_jac=200)
#
# sparho ships ``implicit_forward`` only at v0.x and uses it by default —
# nothing to pass unless you want to override the CG tolerance:
#
# .. code-block:: python
#
#     from sparho import implicit_forward
#     hoag_search(..., hypergrad=implicit_forward)

# %%
# Step 4 — pick the **outer optimizer**, and run.
#
# sparse-ho:
#
# .. code-block:: python
#
#     optimizer = sparse_ho.LineSearch(n_outer=20)
#     monitor = sparse_ho.Monitor()
#     sparse_ho.grad_search(algo, criterion, model, optimizer, X, y, alpha0,
#                           monitor=monitor)
#
# sparho rolls ``algo`` + ``optimizer`` + the outer loop into a single
# call. The ``LineSearch`` outer becomes ``hoag_search``; the ``Monitor``
# becomes ``SearchResult.history`` (an immutable tuple).
result = hoag_search(
    problem,
    hp0=1e-2,
    solver=solver,
    criterion=criterion,
    n_iter=20,
    inner_tol=1e-7,
)
print(f"α* = {result.best_hyperparam:.4g}   converged: {result.converged}")
print(f"outer iterations: {result.n_iter}")

# %%
# Step 5 — inspect the trajectory.
#
# sparse-ho stored the per-iter α / value / time in ``monitor.alphas``,
# ``monitor.objs``, ``monitor.times``. sparho returns the immutable
# ``history`` tuple of :class:`IterationRecord`. Each record carries
# ``iteration``, ``hyperparam``, ``value``, ``grad_norm``,
# ``n_inner_iter``, and an ``extras`` mapping (HOAG records also include
# ``step_size`` / ``L_estimate``; see ``docs/stability.md``).
for rec in result.history[:5]:
    print(
        f"iter {rec.iteration:2d}: "
        f"α={float(rec.hyperparam):.4g}  "
        f"value={rec.value:.4g}  "
        f"|∇θ|={rec.grad_norm:.3g}"
    )

# %%
# Step 6 — use the result.
#
# Refit on the **full** problem at α* — by default the search returns
# ``best_coef`` already refitted, so no extra step is needed.
yhat_val = X[idx_val] @ result.best_coef
mse_val = mean_squared_error(y[idx_val], yhat_val)
print(f"held-out MSE at α*: {mse_val:.4g}")

# %%
# That's the whole story. The detailed translation table for every
# sparse-ho symbol (Models, Criteria, Algorithms, Optimizers, Monitor)
# lives at ``docs/migration_from_sparse_ho.md``. The Sphinx-rendered
# version is on Read the Docs.
