# Convergence: HOAG outer loop

The HOAG algorithm of {cite}`Pedregosa2016Hoag` is the outer loop
sparho recommends for non-smooth bilevel HP optimization. This page
sketches why it converges to a stationary point of the (noisy) outer
objective in spite of using only approximate inner gradients, and
maps each piece of the proof to the corresponding line in
`python/sparho/search.py`.

This is a **sketch**: we state the assumptions, present the key
descent inequality, and note where sparho's implementation deviates
from the published algorithm. For the formal theorem we point to
{cite}`Pedregosa2016Hoag` and to {cite}`Bertrand2022Implicit` for the
implicit-diff branch of the analysis. {cite}`HastieTibshiraniWainwright2015`
covers the inner-problem optimization landscape (Lasso convergence,
active-set behaviour) that grounds the assumptions.

## The setup

We optimize in $\theta = \log \alpha$ space (so $\alpha > 0$ without
projection); writing $g(\theta) := C(\beta^\star(e^\theta))$ for the
outer objective:

$$
\theta^\star \;\in\; \arg\min_{\theta \in \mathbb{R}^q} \; g(\theta),
$$

with $q = 1$ for scalar $\alpha$ or $q = p$ for `WeightedL1`. The
chain rule gives $\nabla g(\theta) = (dC/d\alpha)\odot \alpha$. We
have access to:

- **Inexact value** $\tilde g(\theta; \tau) \approx g(\theta)$ from
  the criterion's `value` path, with the inner solver run at tolerance
  $\tau$.
- **Inexact gradient** $\tilde\nabla g(\theta; \tau) \approx \nabla g(\theta)$
  from `value_and_hypergrad` → `implicit_forward`. The error has two
  sources: (a) the inner solve is only $\tau$-accurate, so
  $\beta^\star$ is approximate; (b) CG truncation in
  `implicit_forward` adds its own error.

Both errors are controlled by $\tau$ — tightening $\tau$ improves
both. {cite}`Pedregosa2016Hoag` proves a quantitative bound of the
form $\|\tilde\nabla g - \nabla g\| \leq c\,\tau^{1/2}$ for
SquaredLoss + L1; we elide the exact constant.

## The acceptance test

HOAG's key technical move is to allow inexact gradients but keep
descent under a **slack-augmented** Armijo-like condition. Let
$g_{k-1} = \tilde g(\theta_{k-1}; \tau_{k-1})$ be the criterion value
from the previous iteration, $L_k$ the current Lipschitz proxy,
$\Delta_k = \|s_k \tilde\nabla g_k\|$ where $s_k = 1/L_k$ is the step
size, and $\tau_k$ the current inner tolerance. The **good-step**
acceptance condition checked retrospectively at iter $k$:

$$
\tilde g_k \;\leq\;
g_{k-1}
\;+\; C\,\tau_k
\;+\; \tau_{k-1}\,(C + \kappa)\,\Delta_k
\;-\; \kappa\, L_k\, \Delta_k^2,
$$

with $C, \kappa$ small positive constants (sparho defaults
$C = 0.25$, $\kappa = 1$ via the `factor` knob; same as sparse-ho).
This is the line

```{code-block} python
slack_good = (
    value_prev + C * tol_k + old_tol * (C + factor) * incr
    - factor * L * incr * incr
)
```

in `search.py:hoag_search`. The two slack terms are what makes the
algorithm tolerant of inexact gradients:

- $C\,\tau_k$ — the criterion value itself has error $O(\tau_k)$, so
  we allow that much "free" non-monotonicity.
- $\tau_{k-1}\,(C+\kappa)\,\Delta_k$ — the gradient direction used
  for the previous step had error $O(\tau_{k-1})$, costing up to
  $\tau_{k-1}\,\|\text{step}\|$ in objective; we allow that too.

The negative quadratic $-\kappa L_k \Delta_k^2$ is the genuine
descent we *do* require — the step has to make progress against the
local Lipschitz proxy.

The **bad-step** condition is the much simpler one-shot test:

$$
\tilde g_k \;\geq\; 1.2\, g_{k-1}.
$$

If the value blew up by more than 20 %, the previous step is
rejected, $L$ is doubled, and we **recompute** value + gradient at
$\theta_{k-1}$ with $\tau \leftarrow \tau/2$:

```{code-block} python
elif value >= slack_bad:
    L *= 2.0
    theta = theta_pre
    tol_retry = tol_k * 0.5
    result = criterion.value_and_hypergrad(problem, _exp(theta), solver,
                                           hypergrad, tol=tol_retry)
```

The doubled $L$ shrinks the next step ($1/L \to 1/(2L)$); the halved
$\tau$ tightens inner accuracy. The combination is what
{cite}`Pedregosa2016Hoag` shows is sufficient to recover descent.

## Why this converges

The Pedregosa argument has three pieces, summarized informally:

1. **Bounded inexactness.** Inner-solver error in both value and
   gradient is bounded by a $O(\tau^{1/2})$ term. As long as the
   tolerance schedule $\{\tau_k\}$ is summable
   ($\sum_k \tau_k^{1/2} < \infty$), the cumulative error is
   bounded.
2. **Descent on acceptance.** Whenever the good-step condition holds,
   the genuine descent $\kappa L_k \Delta_k^2$ dominates the slack
   terms in the long run, giving a Lyapunov decrease in the running
   minimum.
3. **Eventual acceptance.** The bad-step branch doubles $L$ each
   reject; after finitely many doublings $L_k$ exceeds the actual
   local Lipschitz constant of $\nabla g$ and the acceptance test
   must pass. Combined with the inner-tolerance tightening, this
   bounds the total reject count.

The conclusion is convergence of $g_k$ to a stationary value
$g^\star$ along a subsequence. (Stationary, not global minimum —
$g$ is generally non-convex in $\theta$ even when the inner problem
is convex.) See {cite}`Pedregosa2016Hoag`, Theorem 4 for the
quantitative version.

## Tolerance schedule

The schedule $\{\tau_k\}$ matters. sparho exposes two modes:

- **`tolerance_decrease='constant'`** (default). $\tau_k = \tau$
  for all $k$. Simplest; appropriate when the user has a sense of the
  right inner accuracy and doesn't want to spend the budget on early
  iterations.
- **`tolerance_decrease='exponential'`**. Geometric schedule from
  `inner_tol_initial` down to `inner_tol` across `n_iter` outer
  steps:

  $$
  \tau_k \;=\; \tau_{\mathrm{initial}}\,
     \big(\tau_{\mathrm{final}}/\tau_{\mathrm{initial}}\big)^{k/n_{\mathrm{iter}}}.
  $$

  This is what {cite}`Pedregosa2016Hoag`'s convergence proof uses
  and what sparse-ho exposes by default. The intuition: only the
  gradient **direction** matters early (we are far from $\theta^\star$);
  late, the value **resolution** matters because we are close. The
  bad-step branch also locally tightens $\tau$, so even constant mode
  picks up dynamic tightening when the search gets into trouble.

## Lipschitz proxy initialization and step cap

The proxy $L$ has no closed-form starting value (it depends on the
unknown local curvature of $g$). sparho initializes from the
first-iteration gradient magnitude:

```{code-block} python
if L is None:
    if grad_norm > 1e-3:
        L = grad_norm / sqrt(theta.size)  # vector case
        # or grad_norm                     # scalar case
    else:
        L = 1.0
```

then applies a **step-size cap** `max_step` (default $0.5$ in
$\theta$-space, i.e. a factor of $e^{0.5} \approx 1.65$ in $\alpha$):

```{code-block} python
if 1.0 / L * grad_norm > max_step:
    L = grad_norm / max_step
step_size = 1.0 / L
```

The cap is implemented by **raising $L$** rather than clipping the
step after the fact, so the acceptance test sees the same $L$ that
drove the step. Practically, the cap prevents the first iteration
from overshooting into a zero-gradient region before $L$ has had a
chance to adapt — a failure mode we hit on `leukemia` before the cap
was added.

## What sparho's HOAG diverges from in the paper

A handful of pragmatic adjustments:

- **Log-parametrization.** The paper presents HOAG in $\alpha$-space
  with explicit positivity projection. sparho works in $\theta =
  \log\alpha$ so $\alpha > 0$ is automatic. The chain rule
  $\nabla_\theta g = \nabla_\alpha g \cdot \alpha$ is applied
  internally.
- **Two-steps-per-iter on success.** When the acceptance test passes
  and $L$ shrinks, sparho takes an *extra* step in the same direction
  this iter rather than waiting for the next outer cycle:

  ```{code-block} python
  if value <= slack_good:
      L *= 0.95
      theta = _sub(theta, _scale(grad, step_size))   # second step
  ```

  This accelerates the early descent — empirically the leukemia
  speedup is largely from this. It's a heuristic on top of the
  paper's algorithm; the paper's convergence guarantees still apply
  to the slower one-step-per-iter variant.
- **NaN / non-finite guards.** `implicit_forward` may return a zero
  hypergradient on CG failure (see {doc}`implicit_diff`); HOAG
  detects non-finite gradients and skips the step rather than
  propagating NaN through $\theta$. Effectively this is a free
  "reject without rebuilding".
- **Step-size cap (`max_step`).** Not in the published algorithm.
  Sparse-ho has the same cap with the same default. We adopted both.

## Stationarity test and stopping

The outer loop stops on $\|\nabla_\theta \tilde g\| < \mathtt{outer\_tol}$
(default $10^{-6}$). This is checked *before* taking the step in a
given iteration, so a converged iteration still records its value
and Lipschitz estimate.

After the loop terminates, sparho runs the inner solver one more time
on the **full** problem (no train/val split) at `best_hp` — the $\alpha$
attaining the lowest seen value, not necessarily the final $\alpha$.
This last solve is what populates `SearchResult.best_coef`. For
`CrossVal` it is the difference between a per-fold $\beta$ and the
$\beta$ a user actually wants.

## A note on grad_search

{func}`sparho.grad_search` is plain gradient descent in
$\theta$-space with a fixed learning rate. It has none of HOAG's
adaptive machinery; it is included as a baseline and for problems
where the user has a strong prior on `lr`. There is no convergence
guarantee under inexact gradients beyond the general
{cite}`Beck2009Fista`-style proximal-gradient analysis on smooth
objectives, which $g$ is not (it has finite jumps at active-set
transitions). Use HOAG unless you have a specific reason not to.

## Recap

- HOAG handles inexact gradients via slack-augmented acceptance and
  retrospective $L$ doubling.
- Tolerance scheduling (constant or exponential) controls the
  inner-vs-outer accuracy trade-off.
- sparho adds a step-size cap, log-parametrization, and the
  two-steps-on-success heuristic; none invalidate the underlying
  convergence argument.
- The recommended default is `hoag_search` with the constant
  tolerance schedule; switch to exponential when the inner solver is
  expensive and the outer loop is long.
