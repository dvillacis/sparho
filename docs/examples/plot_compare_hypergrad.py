"""
Comparing the hypergradient algorithms
=======================================

sparho ships the full sparse-ho ``algo`` family as interchangeable
``hypergrad=`` choices — ``implicit_forward`` (the default), ``forward``,
``backward``, and ``implicit``. They all compute the **same** quantity, the
hypergradient ``dC/dα`` of an outer criterion ``C`` through the Lasso solution
``β*(α)``; they differ only in *how*:

- ``implicit_forward`` — a coordinate-descent fixed point for the Jacobian,
  restricted to the active set (Rust). The efficient default.
- ``forward`` — re-solves the inner problem while propagating the full Jacobian
  jointly (forward-mode through coordinate descent).
- ``backward`` — records the inner iterates and replays them in reverse
  (reverse-mode); forms the Gram matrix, so it is the costly member.
- ``implicit`` — matrix-free conjugate gradients on the restricted KKT Hessian
  (sparse-ho's *Implicit*); the universal fallback that also handles
  ``LogisticLoss`` / ``GroupL1`` and owns the ``ridge`` knob.

Because the inner Lasso is convex, all four agree at the optimum — and they
agree with a finite-difference of ``C(α)``. This example verifies that, and
sketches the efficiency trade-off.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sparho import L1, Problem, SquaredLoss
from sparho.adapters import SklearnLasso
from sparho.hypergrad import backward, forward, implicit, implicit_forward

# %%
# Synthetic regression with a fixed train/validation split. The outer criterion
# is the held-out mean-squared error ``C(α) = (1/|val|) ‖y_val − X_val β*(α)‖²``.
X, y = make_regression(n_samples=300, n_features=120, n_informative=10, noise=1.0, random_state=0)

rng = np.random.default_rng(0)
perm = rng.permutation(X.shape[0]).astype(np.int32)
idx_train, idx_val = perm[:200], perm[200:]

X_train, y_train = X[idx_train], y[idx_train]
X_val, y_val = X[idx_val], y[idx_val]

alpha = 0.05
train_problem = Problem(SquaredLoss(), L1(), X_train, y_train)
solver = SklearnLasso(tol=1e-12, max_iter=200_000)


# %%
# The hypergradient functions take ``∂C/∂β`` evaluated at ``β*`` and chain it
# through ``dβ*/dα``. For held-out MSE that gradient is
# ``(2/|val|) X_valᵀ (X_val β* − y_val)`` — exactly what ``HeldOutMSE`` computes
# internally. We solve ``β*`` once and reuse it for every algorithm.
def held_out_mse(beta: np.ndarray) -> float:
    resid = X_val @ beta - y_val
    return float(resid @ resid) / len(idx_val)


def grad_beta(beta: np.ndarray) -> np.ndarray:
    return 2.0 * (X_val.T @ (X_val @ beta - y_val)) / len(idx_val)


sr = solver(train_problem, alpha)
gradC = grad_beta(sr.coef)

algos = {
    "implicit_forward": implicit_forward,
    "forward": forward,
    "backward": backward,
    "implicit": implicit,
}
values = {name: float(fn(train_problem, alpha, sr, gradC, tol=1e-12)) for name, fn in algos.items()}

# %%
# Ground truth: a central finite difference of ``C(α)``. The active set is
# stable for a small step, so ``[C(α+ε) − C(α−ε)] / 2ε`` is an accurate proxy
# for ``dC/dα``.
eps = 1e-6
beta_plus = solver(train_problem, alpha + eps).coef
beta_minus = solver(train_problem, alpha - eps).coef
fd = (held_out_mse(beta_plus) - held_out_mse(beta_minus)) / (2.0 * eps)

print(f"finite difference  dC/dα ≈ {fd:.10f}\n")
for name, val in values.items():
    print(f"{name:<18} {val:.10f}   |Δ vs FD| = {abs(val - fd):.2e}")

# %%
# All four land on the same number, to a relative error set by the inner-solver
# tolerance — three independent derivations (BCD forward, BCD reverse, CG)
# agreeing with the finite difference is a strong correctness signal.
#
# They differ in cost. ``implicit_forward`` works on the small active set;
# ``implicit`` runs CG against a matrix-free operator; ``forward`` re-solves the
# inner problem over all features; ``backward`` additionally forms ``XᵀX``. We
# time each (median of a few runs — wall-clock in a doc build is only
# illustrative).
reps = 5
timings = {}
for name, fn in algos.items():
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(train_problem, alpha, sr, gradC, tol=1e-12)
        samples.append(time.perf_counter() - t0)
    timings[name] = float(np.median(samples))

# %%
# Plot the agreement (left, log scale) and the relative timing (right).
names = list(algos)
errors = [max(abs(values[n] - fd), 1e-16) for n in names]
times_ms = [timings[n] * 1e3 for n in names]
colors = ["C0", "C1", "C2", "C3"]

fig, (ax_err, ax_time) = plt.subplots(1, 2, figsize=(10, 4))

ax_err.bar(names, errors, color=colors)
ax_err.axhline(abs(fd) * 1e-6, color="k", ls="--", alpha=0.5, label="1e-6 relative")
ax_err.set_yscale("log")
ax_err.set_ylabel(r"$|\,\mathrm{hypergrad} - \mathrm{FD}\,|$")
ax_err.set_title("Agreement with finite difference")
ax_err.tick_params(axis="x", rotation=30)
ax_err.legend()

ax_time.bar(names, times_ms, color=colors)
ax_time.set_ylabel("time per call (ms)")
ax_time.set_title("Cost per hypergradient (illustrative)")
ax_time.tick_params(axis="x", rotation=30)

fig.suptitle(f"Hypergradient algorithms agree at α = {alpha}  (dC/dα ≈ {fd:.4f})")
fig.tight_layout()
plt.show()

# %%
# Practical guidance:
#
# - **Use the default** (``implicit_forward``) unless you have a reason not to —
#   it is the cheapest for the sparse Lasso family.
# - **``implicit``** is the fallback for ``LogisticLoss`` / ``GroupL1`` and the
#   one with ``ridge`` stabilization for near-singular active-set Hessians.
# - **``forward`` / ``backward``** are provided for completeness and
#   cross-checking; they recompute the inner solve and are slower here.
# - To warm-start the Jacobian across a search, wrap any of them in
#   :class:`sparho.WarmStartHypergrad` and pass it as ``hypergrad=``.
