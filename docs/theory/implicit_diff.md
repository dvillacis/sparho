# Implicit differentiation

We derive the linear system
$M_{AA}\,d\beta^\star_A/d\alpha = -r$
that {func}`sparho.implicit_forward` solves, starting from either of two
equivalent fixed points: KKT stationarity of the inner problem on the
active set, or the proximal-gradient fixed point. The two viewpoints
give the same system; we present the KKT view first (cleaner, no
prox-Jacobian) and then sketch the prox view for context.

## Setup

The inner problem is

$$
\beta^\star(\alpha) \;=\; \arg\min_{\beta \in \mathbb{R}^p}
\; F(\beta;\alpha)
\;:=\; L(X\beta, y) \;+\; R(\beta; \alpha),
$$

with $L$ smooth and convex, $R$ convex but generally non-smooth. We
assume:

1. **Inner uniqueness.** $F(\cdot;\alpha)$ has a unique minimizer
   $\beta^\star(\alpha)$. This holds for `SquaredLoss + L1` with full
   column rank on the active columns; for `LogisticLoss` $L$ is
   strictly convex.
2. **Strict activity on $A$.** At $\beta^\star$, the subgradient of
   $R$ at every $j \in A$ is single-valued — i.e. no coordinate sits
   exactly at a kink ($\beta^\star_j = 0$ for separable L1-type, or
   $\|\beta^\star_{G_k}\| = 0$ for `GroupL1`). This is the
   measure-zero genericity assumption discussed in
   {doc}`active_set`.
3. **Local $C^1$ regularity of $\alpha \mapsto \beta^\star(\alpha)$
   on $A$.** Under (1)+(2) the inner problem on $A$ reduces to a
   smooth strongly-convex problem and the classical implicit
   function theorem
   {cite}`Krantz2013Implicit` applies. See
   {cite}`Bertrand2022Implicit` for a careful treatment in this exact
   setting, and {cite}`Bolte2021NonsmoothImplicit` for a generalized
   nonsmooth IFT covering the boundary cases.

Under (1)–(3), $\beta^\star_I \equiv 0$ on the inactive set
$I = \{0,\dots,p-1\} \setminus A$ in a neighborhood of $\alpha$ (see
{doc}`active_set`), so $d\beta^\star_I/d\alpha = 0$ identically and
the implicit-diff machinery only needs to track $A$.

## KKT view

On $A$, the subdifferential $\partial R$ is single-valued; the inner
KKT stationarity condition is

$$
\nabla_{\!A}\, L(X\beta^\star, y) \;+\; s_A(\beta^\star_A; \alpha)
\;=\; 0,
\qquad
s_A(\beta_A;\alpha) := \partial_\beta R(\beta;\alpha)\big|_A.
$$

For `L1`, $s_A = \alpha \operatorname{sign}(\beta_A)$, locally constant
on $A$ (its derivative in $\beta_A$ is zero); for `WeightedL1` it is
$\alpha_A \odot \operatorname{sign}(\beta_A)$; for `ElasticNet`
$s_A = \alpha (\rho \operatorname{sign}(\beta_A) + (1-\rho)\beta_A)$;
for `GroupL1` $s_{G_k} = \alpha w_k \, \beta_{G_k}/\|\beta_{G_k}\|$ —
a block-structured map enumerated explicitly in {doc}`penalties`.

Both sides are $C^1$ in $(\alpha, \beta_A)$ on a neighborhood of the
solution (by strict activity), so we differentiate in $\alpha$:

$$
\underbrace{\Big(\nabla_{\!\beta_A \beta_A}^{\!2}\,L
   \;+\; \partial_\beta s_A(\beta_A;\alpha)\Big)}_{=:\,M_{AA}}
\;\frac{d\beta^\star_A}{d\alpha}
\;+\;
\underbrace{\partial_\alpha s_A(\beta_A;\alpha)}_{=:\,r}
\;=\; 0.
$$

Rearranged:

$$
\boxed{\;M_{AA} \, \frac{d\beta^\star_A}{d\alpha} \;=\; -\,r.\;}
$$

This is the system {func}`sparho.implicit_forward` builds and inverts.
The two operators on its left:

- $H_{L,AA} = \nabla^2_{\beta_A \beta_A} L(X\beta^\star, y)$ —
  data-side Hessian. For `SquaredLoss`,
  $H_{L,AA} = \tfrac{1}{n} X_A^\top X_A$; for `LogisticLoss`,
  $H_{L,AA} = X_A^\top \operatorname{diag}(w)\, X_A$ with
  $w_i = \sigma(z_i)(1-\sigma(z_i))$, $z = X\beta^\star$.
- $\partial_\beta s_A$ — penalty curvature on $A$. Zero for `L1` /
  `WeightedL1`; uniform diagonal $\alpha(1-\rho) I$ for `ElasticNet`;
  block-diagonal $(\alpha w_k / \|\beta_{G_k}\|)\,(I - u_k u_k^\top)$
  per active group for `GroupL1`, with $u_k = \beta_{G_k} / \|\beta_{G_k}\|$.

The right-hand side $r = \partial_\alpha s_A$ is what
{func}`sparho.implicit_forward` calls the "penalty α-Jacobian on the
active set" — `sign(β_A)` for `L1`, `ρ sign(β_A) + (1-ρ)β_A` for
`ElasticNet`, etc. See {doc}`penalties` for the per-variant list.

## Chain rule for $dC/d\alpha$

The outer criterion $C$ enters only through its gradient
$\partial C/\partial \beta$ at $\beta^\star$. By chain rule
(with $\beta^\star_I \equiv 0$ near $\alpha$):

$$
\frac{dC}{d\alpha} \;=\;
\Big(\frac{\partial C}{\partial \beta}\Big)^{\!\top}
\frac{d\beta^\star}{d\alpha}
\;=\;
\Big(\frac{\partial C}{\partial \beta_A}\Big)^{\!\top}
\frac{d\beta^\star_A}{d\alpha}
\;=\;
-\,\Big(\frac{\partial C}{\partial \beta_A}\Big)^{\!\top}
M_{AA}^{-1}\, r.
$$

The standard adjoint trick avoids materializing $M_{AA}^{-1}$:
solve once for $v = M_{AA}^{-1}\,\partial C/\partial \beta_A$
(symmetric system, $M_{AA} = M_{AA}^\top$), then return $-v^\top r$.
This is exactly the structure of
{func}`sparho.implicit_forward`:

```{code-block} python
v, info = cg(op, grad_C_A, ...)         # M_AA · v = ∂C/∂β_A
return -np.dot(jac_alpha, v)             # = -rᵀ v
```

where `op` is the matrix-free operator wrapping
$H_{L,AA}+\partial_\beta s_A$ and `jac_alpha` is $r$. For
`WeightedL1` $r$ is a vector lifted back to $\mathbb{R}^p$ rather than
scalar-summed.

## Proximal-gradient fixed-point view

The same linear system can be derived from the proximal-gradient
fixed point

$$
\beta^\star \;=\; \operatorname{prox}_{\gamma R(\cdot;\alpha)}
   \!\Big(\beta^\star - \gamma\,\nabla L(X\beta^\star, y)\Big),
$$

valid for any step $\gamma \in (0, 2/\!\operatorname{Lip}(\nabla L))$.
Differentiating in $\alpha$ and using

$$
\frac{d\beta^\star}{d\alpha}
\;=\;
J_z\!\Big(\beta^\star - \gamma\nabla L; \alpha\Big)
\Big(\frac{d\beta^\star}{d\alpha} - \gamma H_L \frac{d\beta^\star}{d\alpha}\Big)
\;+\;
J_\alpha\!\Big(\beta^\star - \gamma\nabla L; \alpha\Big),
$$

where $J_z$ and $J_\alpha$ are the prox Jacobians w.r.t. its input and
$\alpha$ respectively (see {doc}`penalties` and `crates/sparho-core/src/prox.rs`),
one arrives at the same restricted system after using the fact that
$J_z$ is the orthogonal projector onto $A$ on the active set
(and zero on the inactive set, hence killing inactive rows automatically).
sparho takes the KKT route because it avoids materializing the prox
Jacobian inside `hypergrad.py` — the Rust kernels in
`crates/sparho-core/src/prox.rs` expose `prox_jacobian_*` for *checking*
the math in unit tests and for any future hypergradient mode that wants
them, but {func}`sparho.implicit_forward` itself does not call them.

The two derivations are equivalent and both appear in the
sparse-ho line of work; the KKT-restricted version is the practical
formulation introduced in {cite}`Bertrand2020Implicit` and developed in
{cite}`Bertrand2022Implicit`.

## Why $M_{AA}$ is SPD (generically)

Inside CG we need $M_{AA}$ to be symmetric positive definite (SPD) so
that conjugate gradients converges and the system has a unique
solution.

- **Symmetry.** $H_{L,AA}$ is symmetric since $L$ is twice-differentiable
  (and is itself a Gram matrix for `SquaredLoss` / a weighted Gram for
  `LogisticLoss`). $\partial_\beta s_A$ is symmetric for each penalty
  variant: zero / scalar diagonal for L1 / ElasticNet, and
  $(I - u_k u_k^\top)$ is symmetric for `GroupL1`.
- **Positive semidefiniteness.** $H_{L,AA}$ is PSD by convexity of $L$;
  $\partial_\beta s_A$ is PSD because $R$ is convex and we evaluated on
  the smooth branch (for L1 / WL1 it is identically zero; for ElasticNet
  it is a non-negative scalar diagonal; for GroupL1
  $I - u_k u_k^\top$ is the orthogonal projector onto $u_k^\perp$, PSD).
- **Strict definiteness.** Generic active sets give $X_A$ full column
  rank ($|A| \leq n$), so $H_{L,AA} \succ 0$ for SquaredLoss; the
  weighted Gram $X_A^\top \operatorname{diag}(w) X_A$ is positive
  definite for Logistic when $w_i > 0$ (always true) and the same rank
  condition holds. The penalty curvature is PSD on top — `ElasticNet`
  adds a positive scalar shift; `GroupL1` adds a PSD block — so
  $M_{AA}$ inherits strict positive definiteness from $H_{L,AA}$.

## Ridge stabilization

When $X_A$ is near-rank-deficient (highly collinear features, common
on dense designs with $|A|$ approaching $n$), $H_{L,AA}$ becomes
ill-conditioned and CG either stalls or returns a non-finite iterate.
sparho stabilizes by replacing $M_{AA}$ with $M_{AA} + \varepsilon I$:

$$
(M_{AA} + \varepsilon I)\,v_\varepsilon \;=\; \partial C/\partial \beta_A.
$$

The induced bias in $v$, and hence in $dC/d\alpha$, is bounded as
follows. Let $M_{AA}$ have eigendecomposition
$M_{AA} = U \Lambda U^\top$ with $\Lambda = \operatorname{diag}(\lambda_i)$,
$\lambda_i > 0$. Then

$$
\|v - v_\varepsilon\| \;\leq\;
\max_i \frac{\varepsilon}{\lambda_i (\lambda_i + \varepsilon)}
\;\|\partial C/\partial \beta_A\|.
$$

For directions whose eigenvalue $\lambda_i \gg \varepsilon$ the bias is
$O(\varepsilon/\lambda_i^2) \cdot \lambda_i = O(\varepsilon/\lambda_i)$
— negligible. For directions where $\lambda_i \approx \varepsilon$ the
ridge effectively replaces the answer with a soft-pseudoinverse, which
is the desired behavior: the original problem was undefined there.

sparho auto-scales $\varepsilon$ to the operator's natural diagonal
magnitude:

$$
\varepsilon \;=\; 10^{-10} \cdot \frac{\operatorname{tr}(M_{AA})}{|A|},
$$

so on well-conditioned problems CG returns results bit-identical to
$\varepsilon = 0$ across eight orders of magnitude (verified at v0.1.0
release on the libsvm Lasso benchmarks). Pass `ridge=0.0` to
{func}`sparho.implicit_forward` to disable; pass an explicit value to
override the auto-scaling.

## What happens when CG fails

`hypergrad.implicit_forward` treats `scipy.sparse.linalg.cg` failure
(non-zero `info`, or non-finite output) as a hard miss: it emits a
`RuntimeWarning` and **returns a zero hypergradient** for that outer
iteration. The outer loop then takes a zero step, the next iteration
retries with potentially-different inner state (warm-start drift,
HOAG's tolerance schedule reducing inner tolerance, etc.). This is
safer than propagating a NaN — see also the NaN-guards in
{func}`sparho.grad_search` and `sparho.hoag_search`.

## What this enables

A single CG solve of an $|A| \times |A|$ SPD system per outer
iteration replaces the $O(p)$-dimensional fixed-point iterations
that a naïve unrolled hypergradient would need. For sparse problems
where $|A| \ll p$ — the regime sparse-ho was designed for — this
turns hyperparameter tuning into a tractable bilevel problem, which
is the core contribution of the implicit-differentiation line
{cite}`Pedregosa2016Hoag,Bertrand2020Implicit,Bertrand2022Implicit`.

See {doc}`active_set` for the assumptions under which the active set
is locally constant, {doc}`penalties` for the per-variant breakdown
of $\partial_\beta s$ and $\partial_\alpha s$, and {doc}`convergence`
for the HOAG outer-loop analysis.
