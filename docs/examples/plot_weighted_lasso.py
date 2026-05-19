"""
Weighted Lasso (per-feature α)
==============================

``WeightedL1`` carries one regularization strength per feature: a length-``p``
vector ``α``. Grid search over the resulting ``p``-dimensional hyperparameter
space is intractable, but the implicit-differentiation hypergradient is just
a vector, and ``hoag_search`` steps along it the same way it would for a
scalar ``α``.

This is something ``sklearn.LassoCV`` cannot do.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

from sparho import HeldOutMSE, Problem, SquaredLoss, WeightedL1, hoag_search
from sparho.adapters import SklearnWeightedLasso

# %%
# Synthetic data — 200 samples × 40 features, 5 informative.
X, y = make_regression(
    n_samples=200, n_features=40, n_informative=5,
    noise=1.0, random_state=0,
)

rng = np.random.default_rng(0)
perm = rng.permutation(X.shape[0]).astype(np.int32)
idx_train, idx_val = perm[:140], perm[140:]

# %%
# Initial vector ``α`` — uniform 1e-2 across features. The outer search will
# pull informative features' α toward 0 (less shrinkage) and push noise
# features' α up.
n_features = X.shape[1]
alpha0 = np.full(n_features, 1e-2, dtype=np.float64)

problem = Problem(SquaredLoss(), WeightedL1(), X, y)
result = hoag_search(
    problem,
    hp0=alpha0,
    solver=SklearnWeightedLasso(tol=1e-8),
    criterion=HeldOutMSE(idx_train, idx_val),
    n_iter=20,
)

alpha_star = np.asarray(result.best_hyperparam)
nonzero = np.flatnonzero(np.abs(result.best_coef) > 1e-8)
print(f"|active set| = {nonzero.size}")
print(f"α (min/median/max): {alpha_star.min():.3g} / "
      f"{np.median(alpha_star):.3g} / {alpha_star.max():.3g}")

# %%
# Visualize the per-feature α vector after the search.
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
features = np.arange(n_features)
ax_a.stem(features, alpha_star, basefmt=" ")
ax_a.set_xlabel("feature index")
ax_a.set_ylabel(r"$\alpha_j$ at convergence")
ax_a.set_title("Per-feature regularization strength")

ax_b.stem(features, result.best_coef, basefmt=" ")
ax_b.set_xlabel("feature index")
ax_b.set_ylabel(r"$\beta_j^\star$")
ax_b.set_title("Recovered coefficients")
fig.tight_layout()
plt.show()
