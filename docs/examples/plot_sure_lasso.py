# ruff: noqa: D205, D400 -- sphinx-gallery title-block docstring format
"""
SURE-tuned Lasso (no held-out set)
==================================

Stein's Unbiased Risk Estimator (SURE) lets you tune the Lasso regularization
strength when no held-out set exists — denoising, signal recovery, single-fold
settings — by directly estimating the prediction MSE from the *training* data.

This example sets up a sparse-denoising problem with a known noise level σ,
tunes α via :class:`sparho.Sure` + :func:`sparho.hoag_search`, and overlays
the SURE curve against the oracle prediction-MSE curve (which we can only
compute here because we know the noise-free signal).
"""

import matplotlib.pyplot as plt
import numpy as np
from sparho import L1, Problem, SquaredLoss, Sure, hoag_search
from sparho.adapters import SklearnLasso

# %%
# Sparse-denoising fixture — 200 obs × 150 features, 12 informative,
# i.i.d. Gaussian noise with σ = 0.2.
rng = np.random.default_rng(0)
n, p, k = 200, 150, 12
sigma = 0.2
X = rng.standard_normal((n, p)) / np.sqrt(n)
beta_star = np.zeros(p)
support = rng.choice(p, size=k, replace=False)
beta_star[support] = rng.choice([-1.0, 1.0], size=k) * (1.0 + rng.random(k))
y_clean = X @ beta_star
y = y_clean + sigma * rng.standard_normal(n)

problem = Problem(SquaredLoss(), L1(), X, y)

# %%
# Gradient-based search using SURE as the outer criterion.
solver = SklearnLasso(tol=1e-10, max_iter=100_000)
result = hoag_search(
    problem,
    hp0=1e-2,
    solver=solver,
    criterion=Sure(sigma=sigma, random_state=0),
    n_iter=30,
)
alpha_sure = float(result.best_hyperparam)
print(f"sparho SURE: α* = {alpha_sure:.4g}")

# %%
# Oracle curve: prediction MSE against the noise-free signal Xβ*. We can only
# compute this because the example knows the ground truth; in practice SURE is
# what stands in for it.
alphas = np.logspace(-3, 0, 40)
sure_values, oracle_mse = [], []
crit = Sure(sigma=sigma, random_state=0)
for a in alphas:
    sure_values.append(crit.value(problem, float(a), solver))
    coef = solver(problem, float(a)).coef
    err = X @ coef - y_clean
    oracle_mse.append(float(err @ err) / n)

alpha_oracle = float(alphas[int(np.argmin(oracle_mse))])
print(f"oracle:      α* = {alpha_oracle:.4g}")

# %%
# Plot both curves on a shared log-α axis. SURE tracks the oracle near the
# optimum; the FDMC variance shows up as wiggles at very small α (large
# active set ⇒ noisier DOF estimate).
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(alphas, oracle_mse, "o-", color="C1", label="oracle pred. MSE (uses true signal)")
ax.plot(alphas, sure_values, "o-", color="C0", label="SURE (data only)")
ax.axvline(alpha_sure, color="C0", linestyle="--", alpha=0.5, label=f"sparho α* = {alpha_sure:.2g}")
ax.axvline(
    alpha_oracle, color="C1", linestyle="--", alpha=0.5, label=f"oracle α* = {alpha_oracle:.2g}"
)
ax.set_xscale("log")
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("estimated prediction MSE")
ax.set_title("SURE recovers a near-oracle α without a held-out set")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
