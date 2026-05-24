"""
Group-Lasso with HOAG
=====================

`GroupL1` is sparho's block-sparsity penalty: features split into groups
``G_k`` get shrunk together by a block soft-threshold, so either *all*
coordinates in a group are zero or *all* are nonzero. Useful whenever
features come in natural batches — dummy-encoded categoricals, multi-
output regression coefficient rows, multi-frequency Fourier bases, etc.

This example fits a 30-feature regression with three true-active groups
out of ten total, and tunes the single penalty strength ``α`` via
:func:`sparho.hoag_search`. The inner solver is
:class:`sparho.adapters.GroupLassoFista` — a native FISTA on the Rust
block-prox kernel, no celer dependency required.
"""

import matplotlib.pyplot as plt
import numpy as np
from sparho import GroupL1, HeldOutMSE, Problem, SquaredLoss, hoag_search
from sparho.adapters import GroupLassoFista

# %%
# Synthetic data — 10 groups of 3 features each (30 features total). The
# first three groups are active; the remaining seven are noise. Within an
# active group, all three coefficients are nonzero (Group-Lasso's
# structural assumption).
rng = np.random.default_rng(0)
n_samples, n_groups, group_size = 300, 10, 3
n_features = n_groups * group_size

X = rng.standard_normal((n_samples, n_features))

beta_true = np.zeros(n_features)
beta_true[0:3] = [1.5, -0.8, 1.2]  # group 0 active
beta_true[3:6] = [-1.0, 0.7, -0.9]  # group 1 active
beta_true[6:9] = [0.6, 1.1, -0.5]  # group 2 active
y = X @ beta_true + 0.3 * rng.standard_normal(n_samples)

# %%
# Declare the partition. ``GroupL1`` requires a *disjoint, complete* cover
# of ``{0, ..., n_features - 1}``; ``from_labels`` is the convenience for
# the common "block-of-K consecutive features" pattern.
labels = np.repeat(np.arange(n_groups), group_size)
penalty = GroupL1.from_labels(labels)
print(f"groups: {[len(g) for g in penalty.groups for _ in [list(g)]]}  (size each)")

# %%
# Bilevel setup: held-out MSE on a 70/30 split, ``α`` tuned by HOAG.
idx_train = np.arange(int(0.7 * n_samples), dtype=np.int32)
idx_val = np.arange(int(0.7 * n_samples), n_samples, dtype=np.int32)
problem = Problem(SquaredLoss(), penalty, X, y)

result = hoag_search(
    problem,
    hp0=0.1,
    solver=GroupLassoFista(tol=1e-7, max_iter=2000),
    criterion=HeldOutMSE(idx_train=idx_train, idx_val=idx_val),
    n_iter=20,
    inner_tol=1e-6,
    tolerance_decrease="exponential",
    inner_tol_initial=1e-2,
)
print(f"best α = {result.best_hyperparam:.4g}   converged: {result.converged}")

# %%
# Inspect block-sparsity: each group should be entirely zero or entirely
# nonzero, modulo solver tolerance. Plot the recovered coefficient
# magnitudes side-by-side with the true coefficients, banded by group.
beta_hat = result.best_coef
group_norms_true = np.array([np.linalg.norm(beta_true[list(g)]) for g in penalty.groups])
group_norms_hat = np.array([np.linalg.norm(beta_hat[list(g)]) for g in penalty.groups])

fig, (ax_coef, ax_groups) = plt.subplots(1, 2, figsize=(10, 4))
xs = np.arange(n_features)
ax_coef.bar(xs - 0.2, np.abs(beta_true), width=0.4, label="|β true|", alpha=0.85)
ax_coef.bar(xs + 0.2, np.abs(beta_hat), width=0.4, label="|β̂|", alpha=0.85)
for k in range(n_groups):
    ax_coef.axvline(group_size * k - 0.5, color="grey", lw=0.5, alpha=0.5)
ax_coef.set_xlabel("feature index")
ax_coef.set_title("Coefficient magnitudes (banded by group)")
ax_coef.legend()

ax_groups.bar(np.arange(n_groups) - 0.2, group_norms_true, width=0.4, label="‖β_{G_k}‖ true")
ax_groups.bar(np.arange(n_groups) + 0.2, group_norms_hat, width=0.4, label="‖β̂_{G_k}‖")
ax_groups.set_xlabel("group k")
ax_groups.set_title("Per-group ℓ₂ norms — active groups recovered")
ax_groups.legend()
fig.tight_layout()

# %%
# Block-sparsity check: the recovered active set should match the true
# active set at the group level. Internal-zero coordinates inside an
# active group are allowed (GroupL1 doesn't enforce within-group
# sparsity), so we report at the group granularity.
active_true = {k for k, n in enumerate(group_norms_true) if n > 0}
active_hat = {k for k, n in enumerate(group_norms_hat) if n > 1e-6}
print(f"true active groups:      {sorted(active_true)}")
print(f"recovered active groups: {sorted(active_hat)}")
