"""
Cross-validated Lasso
=====================

K-fold cross-validation is wrapped by ``CrossVal``, which averages the
value and the hypergradient across folds. With ``warm_start=True`` each
fold reuses its previous-iteration ``β*`` as the next inner solve's
starting point — a substantial speed-up when the inner solver dominates.

We compare against ``sklearn.LassoCV`` on the same K-fold split.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from sparho import CrossVal, L1, Problem, SquaredLoss, hoag_search
from sparho.adapters import SklearnLasso

# %%
# Synthetic data — 300 samples × 80 features, 10 informative.
X, y = make_regression(
    n_samples=300, n_features=80, n_informative=10,
    noise=1.0, random_state=0,
)

# %%
# Build the 5-fold splitter once, then feed the same folds to both pipelines
# so the comparison is apples-to-apples.
kf = KFold(n_splits=5, shuffle=True, random_state=0)
folds = tuple(
    (np.asarray(tr, dtype=np.int32), np.asarray(val, dtype=np.int32))
    for tr, val in kf.split(np.arange(X.shape[0]))
)

problem = Problem(SquaredLoss(), L1(), X, y)
result = hoag_search(
    problem,
    hp0=1e-2,
    solver=SklearnLasso(tol=1e-8),
    criterion=CrossVal(folds=folds, warm_start=True),
    n_iter=20,
)
best_alpha = float(result.best_hyperparam)
print(f"sparho: α = {best_alpha:.4g}   mean CV MSE = "
      f"{min(r.value for r in result.history):.4f}")

# %%
# LassoCV baseline on a 30-point grid using the same folds.
alphas_grid = np.logspace(-4, 1, 30)
cv = LassoCV(
    alphas=alphas_grid, cv=list(folds), fit_intercept=False,
    tol=1e-8, max_iter=10_000,
)
cv.fit(X, y)
cv_mse = float(cv.mse_path_.mean(axis=1).min())
print(f"LassoCV: α = {cv.alpha_:.4g}   mean CV MSE = {cv_mse:.4f}")

# %%
# Refit metrics on the full data at each method's best α.
fit_full = mean_squared_error(y, X @ result.best_coef)
fit_cv = mean_squared_error(y, X @ cv.coef_)

fig, ax = plt.subplots(figsize=(6, 4))
xs = [float(r.hyperparam) for r in result.history]
ys = [r.value for r in result.history]
ax.plot(xs, ys, "o-", color="C0", label="sparho HOAG (5-fold CV)")
ax.axvline(best_alpha, color="C0", linestyle="--", alpha=0.5)
ax.axvline(cv.alpha_, color="C1", linestyle="--", alpha=0.5,
           label=f"LassoCV α = {cv.alpha_:.2g}")
ax.set_xscale("log")
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("mean CV MSE")
ax.set_title("Cross-validated Lasso: gradient-based vs grid")
ax.legend()
fig.tight_layout()
plt.show()

print(f"\nfull-data refit MSE — sparho: {fit_full:.4f}   LassoCV: {fit_cv:.4f}")
