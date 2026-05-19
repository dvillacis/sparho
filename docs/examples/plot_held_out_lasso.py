"""
Held-out Lasso with HOAG
========================

The canonical sparho example: tune the Lasso regularization strength ``α``
to minimize the mean-squared error on a fixed held-out set, using one
gradient-based outer search instead of a grid sweep.

This is what ``hoag_search`` was written for. Sklearn's ``LassoCV`` is the
grid-search baseline; we compare against it at the end.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

from sparho import HeldOutMSE, L1, Problem, SquaredLoss, hoag_search
from sparho.adapters import SklearnLasso

# %%
# Synthetic data — 300 samples × 100 features, 10 informative, mild noise.
X, y = make_regression(
    n_samples=300, n_features=100, n_informative=10,
    noise=1.0, random_state=0,
)

rng = np.random.default_rng(0)
perm = rng.permutation(X.shape[0]).astype(np.int32)
idx_train, idx_val = perm[:200], perm[200:]

# %%
# Run sparho — one HOAG outer search in ``log α`` space, starting at α = 1e-2.
problem = Problem(SquaredLoss(), L1(), X, y)
result = hoag_search(
    problem,
    hp0=1e-2,
    solver=SklearnLasso(tol=1e-8),
    criterion=HeldOutMSE(idx_train, idx_val),
    n_iter=30,
)

best_alpha = float(result.best_hyperparam)
best_mse = mean_squared_error(y[idx_val], X[idx_val] @ result.best_coef)
print(f"sparho: α = {best_alpha:.4g}   held-out MSE = {best_mse:.4f}")

# %%
# Baseline: ``LassoCV`` on a 30-point log grid using the same fold.
alphas_grid = np.logspace(-4, 1, 30)
cv = LassoCV(
    alphas=alphas_grid, cv=[(idx_train, idx_val)],
    fit_intercept=False, tol=1e-8, max_iter=10_000,
)
cv.fit(X, y)
grid_mse = mean_squared_error(y[idx_val], X[idx_val] @ cv.coef_)
print(f"LassoCV: α = {cv.alpha_:.4g}   held-out MSE = {grid_mse:.4f}")

# %%
# Plot the sparho trajectory against the grid baseline.
fig, ax = plt.subplots(figsize=(6, 4))

xs = [float(r.hyperparam) for r in result.history]
ys = [r.value for r in result.history]
ax.plot(xs, ys, "o-", color="C0", label="sparho HOAG trajectory")
ax.axvline(best_alpha, color="C0", linestyle="--", alpha=0.5)
ax.axvline(cv.alpha_, color="C1", linestyle="--", alpha=0.5,
           label=f"LassoCV α = {cv.alpha_:.2g}")
ax.set_xscale("log")
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("held-out MSE")
ax.set_title("Held-out Lasso: gradient-based search vs. grid")
ax.legend()
fig.tight_layout()
plt.show()
