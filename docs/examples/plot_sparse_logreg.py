"""
Sparse logistic regression
==========================

Held-out logistic loss on a binary classification problem with an L1
penalty. The Problem is ``(LogisticLoss, L1, X, y)`` with ``y ∈ {−1, +1}``;
the rest of the bilevel machinery is unchanged.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sparho import L1, HeldOutLogistic, LogisticLoss, Problem, hoag_search
from sparho.adapters import SklearnLogisticRegression

# %%
# Binary classification: 250 samples × 50 features, 8 informative.
X, y01 = make_classification(
    n_samples=250,
    n_features=50,
    n_informative=8,
    n_redundant=5,
    random_state=0,
)
# sparho's LogisticLoss expects ±1 labels.
y = (2 * y01 - 1).astype(np.float64)

rng = np.random.default_rng(0)
perm = rng.permutation(X.shape[0]).astype(np.int32)
idx_train, idx_val = perm[:180], perm[180:]

# %%
# Outer search. ``SklearnLogisticRegression`` uses liblinear under the hood;
# liblinear doesn't support warm-start, so ``x0`` is ignored — only
# ``tol`` is meaningful here.
problem = Problem(LogisticLoss(), L1(), X, y)
result = hoag_search(
    problem,
    hp0=1e-1,
    solver=SklearnLogisticRegression(tol=1e-8),
    criterion=HeldOutLogistic(idx_train, idx_val),
    n_iter=20,
)

print(f"sparho: α = {float(result.best_hyperparam):.4g}")
print(f"        held-out loss = {result.history[-1].value:.4f}")
print(f"        |active set|  = {np.count_nonzero(result.best_coef)}")

# %%
# Plot the outer-loop trajectory.
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))

xs = [float(r.hyperparam) for r in result.history]
ys = [r.value for r in result.history]
ax_a.semilogx(xs, ys, "o-")
ax_a.set_xlabel(r"$\alpha$")
ax_a.set_ylabel("held-out logistic loss")
ax_a.set_title("Outer-search trajectory")

ax_b.stem(np.arange(X.shape[1]), result.best_coef, basefmt=" ")
ax_b.set_xlabel("feature index")
ax_b.set_ylabel(r"$\beta_j^\star$")
ax_b.set_title("Recovered coefficients")
fig.tight_layout()
plt.show()
