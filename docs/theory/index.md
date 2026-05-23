# Theory

This section derives the math behind sparho's hypergradient, active-set
restriction, prox kernels, and criteria, at enough depth that a reader
should **not** need to fetch the underlying papers. Each page is a
self-contained derivation with pointers into the implementation it
backs.

If you only need the user-facing summary, [Concepts](../concepts.md)
covers the same ideas in two screens. Use this section when you want to
understand *why* a particular line in `hypergrad.py` looks the way it
does, or to extend sparho with a new datafit/penalty/criterion.

## Contents

```{toctree}
:maxdepth: 1

implicit_diff
active_set
penalties
criteria
convergence
references
```

## The bilevel problem

sparho solves

$$
\min_{\alpha \,\in\, \mathcal{A}} \; C(\beta^\star(\alpha))
\quad \text{s.t.} \quad
\beta^\star(\alpha) \;=\; \arg\min_{\beta \in \mathbb{R}^p} \;
  L(X\beta, y) + R(\beta; \alpha).
$$

Two layers: an **inner** convex (typically non-smooth) problem in
$\beta$ with a closed-form Bayes-style structure (`Lasso`,
`ElasticNet`, `WeightedL1`, `GroupL1`, possibly with `LogisticLoss`),
and an **outer** problem in the hyperparameter $\alpha$ governed by a
**criterion** $C$ — held-out MSE, K-fold CV, SURE.

The unique inner solution $\beta^\star(\alpha)$ defines a map
$\alpha \mapsto \beta^\star(\alpha)$. Computing $dC/d\alpha$ through that
map is the central technical problem; sections {doc}`implicit_diff` and
{doc}`active_set` derive the linear system sparho solves. The pages
{doc}`penalties` and {doc}`criteria` enumerate the (datafit, penalty)
dispatches and the criterion chain rule; {doc}`convergence` sketches
why the HOAG outer loop converges to a stationary point of the
(noisy) outer objective.

## Notation

The following symbols are used consistently across this section unless
explicitly re-defined.

```{list-table}
:header-rows: 1
:widths: 18 82

* - Symbol
  - Meaning
* - $n,\,p$
  - Number of samples / features. $n = $ `problem.n_samples`,
    $p = $ `problem.n_features`.
* - $X \in \mathbb{R}^{n \times p}$
  - Design matrix. `problem.design`. May be dense or
    `scipy.sparse.csc_matrix`.
* - $y \in \mathbb{R}^n$ or $\{-1, +1\}^n$
  - Target. `problem.target`. Regression vs. binary classification
    by datafit.
* - $\beta \in \mathbb{R}^p$
  - Inner parameter (coefficient vector). $\beta^\star(\alpha)$ at
    optimum.
* - $\alpha$
  - Hyperparameter. Scalar for `L1` / `ElasticNet` / `GroupL1`,
    length-$p$ for `WeightedL1`. Strictly positive componentwise.
* - $\theta = \log\alpha$
  - Log-parametrization. Outer loops in `search.py` step in $\theta$;
    chain rule $dC/d\theta = dC/d\alpha \cdot \alpha$.
* - $L(\cdot,\cdot)$
  - Datafit. $\tfrac{1}{2n}\|X\beta - y\|^2$ for `SquaredLoss`
    (sklearn convention), $\sum_i \log(1 + e^{-y_i (X\beta)_i})$ for
    `LogisticLoss`.
* - $R(\beta;\alpha)$
  - Penalty. `L1`, `WeightedL1`, `ElasticNet`, `GroupL1` —
    enumerated in {doc}`penalties`.
* - $C(\beta)$
  - Outer criterion. `HeldOutMSE`, `HeldOutLogistic`, `CrossVal`,
    `Sure` — enumerated in {doc}`criteria`.
* - $A \subseteq \{0, \dots, p-1\}$
  - Active set. $A = \{j : \beta^\star_j \neq 0\}$ for separable
    penalties; for `GroupL1` it is the union of coordinates in
    nonzero groups (see {doc}`active_set`).
* - $H_L = \nabla_{\beta\beta}^2 L(X\beta^\star, y)$
  - Inner-loss Hessian at $\beta^\star$.
* - $H_{L,AA}$
  - Submatrix of $H_L$ on rows/cols in $A$. Restricted Hessian.
* - $M_{AA}$
  - $H_{L,AA} + \nabla_{\beta\beta}^2 R\big|_A$ — the augmented Hessian
    sparho's CG solver inverts; see {doc}`implicit_diff`.
* - $r$
  - Right-hand side $\nabla^2_{\alpha\beta} R\big|_A$ of the
    implicit-diff linear system; penalty-specific column built in
    {func}`sparho.implicit_forward`.
```

The key load-bearing fact across this section: every algorithm sparho
exposes operates *only* on $A$. The inactive coordinates contribute
zero to $d\beta^\star/d\alpha$ — see {doc}`active_set` — so the
implicit-diff linear system is $|A| \times |A|$ rather than $p \times p$.
For sparse problems where $|A| \ll p$ this is the difference between a
tractable hypergradient and an intractable one.
