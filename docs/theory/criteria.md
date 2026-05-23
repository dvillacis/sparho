# Criteria and the outer chain rule

A **criterion** $C$ defines what we want to minimize over $\alpha$.
Each criterion is responsible for two things:

1. **Value**: compute $C(\beta^\star(\alpha))$.
2. **Hypergradient ingredients**: produce $\partial C/\partial \beta$
   at $\beta^\star$ so that {func}`sparho.implicit_forward` can chain
   it through the implicit-diff linear system to get $dC/d\alpha$.

This page lays out the chain rule for each criterion sparho ships and
derives the SURE / SUGAR FDMC estimator, which is the only criterion
whose hypergradient is non-obvious.

## The outer chain rule

Across all four criteria the hypergradient factors as

$$
\frac{dC}{d\alpha}
\;=\;
\Big(\frac{\partial C}{\partial \beta}\Big)^{\!\top}
\frac{d\beta^\star}{d\alpha},
$$

where $d\beta^\star/d\alpha$ comes from {doc}`implicit_diff`. Each
criterion is responsible only for $\partial C/\partial \beta$ — it
does *not* know about the inner problem's KKT structure. The
{class}`sparho.criteria.CriterionResult` returned by
`Criterion.value_and_hypergrad` carries `value` and `hypergrad`,
and the latter is the criterion's call to `hypergrad_fn(train_problem,
hp, solver_result, grad_beta)` — for the v0.1 default this is
{func}`sparho.implicit_forward`.

## HeldOutMSE

$$
C(\beta) \;=\; \frac{1}{|\mathrm{val}|}
\sum_{i \in \mathrm{val}} \big(y_i - X_{i,:}\,\beta\big)^2.
$$

Matches `sklearn.mean_squared_error` (no $1/2$). The factor of $2$
shows up in the gradient:

$$
\frac{\partial C}{\partial \beta}
\;=\;
\frac{2}{|\mathrm{val}|}\, X_{\mathrm{val}}^\top
   \big(X_{\mathrm{val}}\,\beta - y_{\mathrm{val}}\big).
$$

`HeldOutMSE` slices the full problem to the **training** subproblem
($\beta^\star$ depends only on $X_{\mathrm{tr}}$, $y_{\mathrm{tr}}$),
solves it, then evaluates the held-out MSE on the **validation**
subset. Both index sets are user-supplied.

## HeldOutLogistic

$$
C(\beta) \;=\;
\frac{1}{|\mathrm{val}|}\, \sum_{i \in \mathrm{val}}
\log\!\big(1 + e^{-y_i\, X_{i,:}\,\beta}\big),
\qquad y_i \in \{-1, +1\}.
$$

Numerically stable form: `np.logaddexp(0, -y · Xβ)`. Gradient
(letting $\sigma(t) = 1/(1 + e^t)$):

$$
\frac{\partial C}{\partial \beta}
\;=\;
-\,\frac{1}{|\mathrm{val}|}\, X_{\mathrm{val}}^\top
   \big(y_{\mathrm{val}} \odot \sigma(y_{\mathrm{val}} \odot X_{\mathrm{val}}\beta)\big).
$$

Used with `LogisticLoss` inner problems.

## CrossVal

K-fold aggregation over a single-split base criterion (default
`HeldOutMSE`, optionally `HeldOutLogistic` for classification):

$$
C(\beta) \;=\; \frac{1}{K}\sum_{k=1}^{K} C_k(\beta^{(k)}),
\qquad
\frac{dC}{d\alpha} \;=\; \frac{1}{K}\sum_{k=1}^{K}
   \frac{dC_k}{d\alpha}.
$$

Each fold $k$ has its own train/val split $(I_{\mathrm{tr}}^{(k)},
I_{\mathrm{val}}^{(k)})$ and its own inner solve, so $\beta^{(k)}$
differs across folds. Both the value and the hypergradient average
linearly across folds. With `warm_start=True`, fold $k$'s previous-
iteration $\beta^\star$ seeds the next iteration's fold-$k$ inner
solve. Since the inner problem is convex this is **safe** —
convergence is to the same $\beta^\star(\alpha)$ regardless of init —
and on the bench it is a $1.5\times$–$3\times$ speedup when the inner
solver dominates.

## SURE / SUGAR (FDMC)

`Sure` is the only sparho criterion that needs a derivation. It is a
single-split, **self-supervised** alternative to held-out validation:
it doesn't need a validation set, only the noise standard deviation
$\sigma$. Useful in denoising / signal-recovery settings where holding
out data is wasteful.

### Stein's identity → SURE for the linear case

Suppose $y = X\beta_{\mathrm{true}} + \xi$ with
$\xi \sim \mathcal{N}(0, \sigma^2 I_n)$. The **prediction MSE**

$$
R_{\mathrm{pred}}(\hat\beta) \;=\;
\mathbb{E}\!\left[\frac{1}{n}\|X\beta_{\mathrm{true}} - X\hat\beta\|^2\right]
$$

is what we ultimately want to minimize over $\alpha$. Stein's lemma
{cite}`Stein1981Estimation` gives the unbiased identity (after
expansion):

$$
\mathbb{E}\!\left[\frac{1}{n}\|y - X\hat\beta\|^2\right]
\;=\;
R_{\mathrm{pred}}(\hat\beta)
\;+\;\sigma^2
\;-\;\frac{2\sigma^2}{n}\,\mathrm{DOF}(\hat\beta),
$$

where the **effective degrees of freedom** is

$$
\mathrm{DOF}(\hat\beta)
\;=\;
\frac{1}{\sigma^2}\,\mathbb{E}\!\big[\xi^\top X\hat\beta(\,y\,)\big]
\;=\;
\mathbb{E}\!\left[\sum_{i=1}^{n}
   \frac{\partial (X\hat\beta)_i}{\partial y_i}\right].
$$

Solving for $R_{\mathrm{pred}}$:

$$
\boxed{\;
\mathrm{SURE}(\hat\beta)
\;:=\;
\frac{1}{n}\|y - X\hat\beta\|^2 \;-\;\sigma^2
\;+\;\frac{2\sigma^2}{n}\,\mathrm{DOF}(\hat\beta)
\;}
$$

satisfies $\mathbb{E}[\mathrm{SURE}(\hat\beta)] = R_{\mathrm{pred}}(\hat\beta)$
— unbiased estimate of the prediction risk, *no held-out data
required*. For the Lasso, $\mathrm{DOF} = |A|$ in expectation
{cite}`Zou2007DegreesOfFreedom`; sparho's tests use this fact as a
sanity check (see below).

### Finite-Difference Monte Carlo DOF

The catch is the divergence $\sum_i \partial (X\hat\beta)_i /
\partial y_i$. For smooth $\hat\beta$ we could differentiate
analytically; for a Lasso / ElasticNet solution the dependence on
$y$ is non-differentiable at the active-set transitions. The
**SUGAR** estimator {cite}`Deledalle2014Sugar` sidesteps this with a
single Monte-Carlo probe:

$$
\widehat{\mathrm{DOF}}(\hat\beta; \delta, \varepsilon)
\;=\;
\frac{1}{\varepsilon}\,
\delta^\top\, X\,\big(\hat\beta(y + \varepsilon\delta) - \hat\beta(y)\big),
\quad
\delta \sim \mathcal{N}(0, I_n).
$$

As $\varepsilon \to 0$ this converges to
$\delta^\top \nabla_y (X\hat\beta) \delta$; taking expectation over
$\delta$ recovers
$\sum_i \partial(X\hat\beta)_i/\partial y_i$ (Hutchinson trace
estimator). A finite $\varepsilon$ trades MC variance against
finite-difference bias. {cite}`Deledalle2014Sugar` recommends

$$
\varepsilon \;=\; \frac{2\sigma}{n^{0.3}},
$$

sparho's default when `Sure(epsilon=None)`. The probe $\delta$ and
$\varepsilon$ are **fixed for the lifetime of the `Sure` instance**
so the criterion is a deterministic function of $\alpha$ — required
for line-search monotonicity and FD gradient checks.

### sparho's SURE estimator

Plugging FDMC into the SURE expression and identifying the two inner
solves ($\hat\beta_1 = \hat\beta(y)$, $\hat\beta_2 = \hat\beta(y +
\varepsilon\delta)$):

$$
\mathrm{SURE}(\alpha)
\;=\;
\underbrace{\frac{1}{n}\|y - X\hat\beta_1\|^2 - \sigma^2}_{\text{data term}}
\;+\;
\underbrace{\frac{2\sigma^2}{n\varepsilon}\,
   \delta^\top X\,(\hat\beta_2 - \hat\beta_1)}_{\text{DOF correction}}.
$$

That is exactly `Sure._sure_value` (compare line-for-line). Two inner
solves per criterion evaluation; `warm_start=True` seeds both from
the previous iter's $\hat\beta_1, \hat\beta_2$.

### SURE's hypergradient: two `implicit_forward` calls

The hypergradient is the chain rule applied to both $\hat\beta_1$
and $\hat\beta_2$ — they depend on $\alpha$ via two different
problems (training target $y$ vs. perturbed target $y +
\varepsilon\delta$). Letting

$$
\Phi(\beta_1, \beta_2)
\;=\;
\frac{1}{n}\|y - X\beta_1\|^2 - \sigma^2
\;+\;
\frac{2\sigma^2}{n\varepsilon}\,\delta^\top X (\beta_2 - \beta_1),
$$

the partial gradients are

$$
\frac{\partial \Phi}{\partial \beta_1}
\;=\;
\frac{2}{n}\, X^\top (X\beta_1 - y)
\;-\;
\frac{2\sigma^2}{n\varepsilon}\, X^\top \delta,
\qquad
\frac{\partial \Phi}{\partial \beta_2}
\;=\;
\frac{2\sigma^2}{n\varepsilon}\, X^\top \delta.
$$

The hypergradient is then

$$
\frac{d\,\mathrm{SURE}}{d\alpha}
\;=\;
\Big(\frac{\partial \Phi}{\partial \beta_1}\Big)^{\!\top}
\frac{d\hat\beta_1}{d\alpha}
\;+\;
\Big(\frac{\partial \Phi}{\partial \beta_2}\Big)^{\!\top}
\frac{d\hat\beta_2}{d\alpha},
$$

which is exactly two `implicit_forward` calls — one against the
original problem with $\partial \Phi/\partial \beta_1$, one against
the perturbed problem with $\partial \Phi/\partial \beta_2$. In
{class}`sparho.criteria.Sure`'s `value_and_hypergrad`:

```{code-block} python
coupling = (2.0 * sigma_sq / (n * eps)) * X.T @ delta
grad_beta_1 = (2.0 / n) * X.T @ (X @ r1.coef - y) - coupling
grad_beta_2 = coupling
hg1 = hypergrad_fn(problem,   hp, r1, grad_beta_1)
hg2 = hypergrad_fn(perturbed, hp, r2, grad_beta_2)
return CriterionResult(..., hypergrad=hg1 + hg2, ...)
```

### Why SURE only supports `SquaredLoss`

SURE's derivation rests on Stein's lemma, which assumes Gaussian
observation noise on a *linear* predictor. There is no
distribution-free analogue for `LogisticLoss` that admits a useful
finite-sample estimator. sparho raises `TypeError` if you try.

### Sanity check via the Lasso-DOF identity

For Lasso under a continuous-density design, $\mathrm{DOF} = |A|$ in
expectation {cite}`Zou2007DegreesOfFreedom`. sparho's
`tests/test_sure.py` exploits this: average the FDMC DOF over $M$
probes; it should concentrate on the average $|A|$. That test
catches sign / scaling errors in the SURE math end-to-end.

## Recap

- Every criterion in sparho exposes the same two-method protocol
  (`value`, `value_and_hypergrad`) and reduces its hypergradient to a
  call to `hypergrad_fn` with the right $\partial C/\partial \beta$
  — see {class}`sparho.Criterion`.
- `HeldOutMSE`, `HeldOutLogistic` are straightforward chain rules.
- `CrossVal` averages.
- `Sure` is the substantive case: SURE + FDMC, two inner solves and
  two `implicit_forward` calls per evaluation, deterministic-by-seed
  probe.

See {doc}`convergence` for how these criterion values plug into the
HOAG outer loop's acceptance test.
