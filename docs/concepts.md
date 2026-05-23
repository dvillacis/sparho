# Concepts

This page is the two-screen user-facing summary of the math. If you
want the derivations — KKT-based implicit differentiation, the
active-set restriction argument, per-penalty prox/Jacobian formulas,
the SURE/SUGAR FDMC estimator, the HOAG convergence sketch — those
live in the [Theory](theory/index.md) section. The
{doc}`theory/index` page also fixes the notation used throughout.

## The bilevel problem

Choosing a hyperparameter `α` for a non-smooth estimator can be written as

$$
\min_{\alpha > 0} \; C(\beta^\star(\alpha))
\quad \text{s.t.} \quad
\beta^\star(\alpha) = \arg\min_{\beta} \; L(X\beta, y) + R(\beta; \alpha).
$$

`L` is the datafit (e.g. squared loss for Lasso, logistic loss for sparse
logistic regression); `R` is the non-smooth regularizer (`α‖β‖₁` for Lasso,
elastic net, weighted L1). `C` is an outer **criterion** — typically a
held-out MSE or cross-validated MSE — that we want to minimize over `α`.

The inner problem is convex, has a unique solution `β*(α)` for each `α`, and
is the same problem sklearn / celer / friends solve.

## Why implicit differentiation

Grid search evaluates `C(β*(α))` at a finite set of `α` values; random
search samples them. Both pay a full inner-solve per `α`, and the
"resolution" of `α` is bounded below by the spacing of the grid.

If we can compute `dC/dα` we can run any first-order optimizer on it instead.
That gives:

- **Many fewer inner solves** — one per outer step, not one per grid point.
- **`α` adapts continuously** — no grid floor; on `rcv1.binary` sparho's
  search drives `α` two orders of magnitude below `LassoCV`'s default grid
  and lands on a strictly better held-out MSE.
- **Vector-valued `α`** — weighted Lasso (`WeightedL1`) has one `α_j` per
  feature; grid search is intractable, but the hypergradient is a vector
  the optimizer can step along.

## Computing the hypergradient

At `β*(α)` the inner KKT conditions hold on the active set
`A = { j : β*_j ≠ 0 }`:

$$
\nabla_{\!A}\, L(X\beta^\star, y) \;+\; \partial R(\beta^\star_A; \alpha) \;=\; 0.
$$

Differentiating implicitly in `α`:

$$
\bigl(H_{L,AA} + \nabla^2_{\beta\beta} R |_A\bigr)\,
  \frac{d\beta^\star_A}{d\alpha}
\;+\;
\nabla^2_{\alpha\beta} R |_A
\;=\; 0.
$$

Set `M_AA = H_{L,AA} + diag(curvature of R on A)`. The hypergradient by
chain rule is

$$
\frac{dC}{d\alpha}
\;=\;
\Bigl(\frac{\partial C}{\partial \beta_A}\Bigr)^{\!\top}
\bigl(-M_{AA}^{-1}\bigr)\,
\nabla^2_{\alpha\beta} R |_A.
$$

{py:func}`sparho.implicit_forward` solves
`M_AA v = ∂C/∂β_A` by matrix-free conjugate gradients on the active set;
the matvec is done in Rust (`sparho._core.restricted_ls_hessian_matvec` for
squared loss, a small dense Gram for logistic). Sparse-X stays sparse
end-to-end.

A small Tikhonov ridge `M_AA + εI` keeps CG well-posed on near-singular
restricted Hessians (collinear features in a dense design). The default
`ε = 10⁻¹⁰ · trace(M_AA)/|A|` scales with the operator and is
bit-identical to `ε = 0` on well-conditioned problems.

## The outer loop

Both `grad_search` and `hoag_search` step in `θ = log α` so `α` stays
strictly positive without projection. The chain rule `dC/dθ = dC/dα · α`
is applied internally.

- {py:func}`sparho.grad_search` — plain
  `θ ← θ − lr · dC/dθ` with a fixed learning rate. One val+grad call per
  outer iter. Use as a baseline or when you have prior knowledge of a good
  `lr`.
- {py:func}`sparho.hoag_search` — Pedregosa
  (2016). Adapts step size from a Lipschitz proxy `L`; an acceptance test
  with a `C·tol` slack term tolerates inner-solver noise; bad descent
  doubles `L` and recomputes the val+grad with a tighter inner tolerance.
  Recommended default.

After the loop, the solver runs once more on the **full** problem at the
best `α` seen, and `SearchResult.best_coef` holds the resulting `β`. For
`CrossVal` this matters — the per-fold `coef` reported by the criterion is
the last-fold fit, not what the user actually wants.

## Criteria

- {py:class}`sparho.HeldOutMSE` — squared error on a fixed validation
  index set. Matches `sklearn.mean_squared_error` (no `1/2`).
- {py:class}`sparho.HeldOutLogistic` — logistic loss on `y ∈ {−1, +1}`,
  numerically stable via `logaddexp`.
- {py:class}`sparho.CrossVal` — K-fold
  aggregator over any single-split base criterion. Value and hypergradient
  are means across folds. Opt-in `warm_start=True` lets each fold reuse its
  previous `β*` as the next inner solve's starting point — big speedup
  when the inner solver dominates.

## When not to use this

Implicit differentiation needs an inner problem with a continuous
`β*(α)` and a usable second-order structure on the active set. v0.1 ships
the cases that sparse-ho's audience actually uses; non-convex inner
problems and constrained inner problems are not supported.

For very small data (`breast-cancer` 683 × 10) the FFI overhead and the
fixed outer-iter budget dominate the inner solve; `LassoCV` finishes
instantly. The pay-off shows up where the inner solver is the bottleneck —
high-dimensional, sparse, or many-fold CV settings.
