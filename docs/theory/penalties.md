# Penalties: prox, Jacobian, $\partial_\beta s$, $\partial_\alpha s$

For each penalty variant in the closed `Penalty` union we list:

- the **regularizer** $R(\beta;\alpha)$,
- its **proximal operator** and **subdifferential**, with pointer to
  the Rust kernel in `crates/sparho-core/src/prox.rs`,
- the **prox Jacobian** in input $z$ and in $\alpha$ (used by
  hypergradient modes that need it; sparho's `implicit_forward`
  does *not* use the prox Jacobian directly — see
  {doc}`implicit_diff`),
- the **$\partial_\beta s_A$** ("penalty curvature on $A$") and
  **$\partial_\alpha s_A$** ("penalty α-Jacobian on $A$") that feed
  $M_{AA}$ and $r$ in the implicit-diff linear system.

The naming follows {doc}`implicit_diff` and `hypergrad.py`. All
formulas hold on the active set after the strict-activity assumption
($\star$) of {doc}`active_set`.

## Soft-thresholding primitive

For scalar $z, t \geq 0$ define

$$
\operatorname{soft}(z, t)
\;=\;
\operatorname{sign}(z)\,\max(|z| - t,\, 0).
$$

Every separable penalty in sparho is built from this primitive
(`crates/sparho-core/src/kernels.rs:soft_threshold`). At the kink
$|z| = t$ we follow sparse-ho in declaring the coordinate inactive
(Jacobian = 0). This is a measure-zero ambiguity.

## L1

$$
R(\beta;\alpha) \;=\; \alpha \,\|\beta\|_1
\;=\; \alpha \sum_j |\beta_j|.
$$

Standard Lasso {cite}`Tibshirani1996Lasso`.

**Prox** (`prox_l1`):
$\operatorname{prox}_{\gamma\alpha\|\cdot\|_1}(z)_j = \operatorname{soft}(z_j,\, \gamma\alpha)$.

**Subdifferential**: $\partial_j R(\beta;\alpha) = \alpha\,\operatorname{sign}(\beta_j)$ when
$\beta_j \neq 0$; $[-\alpha, \alpha]$ when $\beta_j = 0$.

**Prox Jacobian** (`prox_jacobian_l1`), diagonal:

$$
J_z(z, \alpha)_{jj}
\;=\;
\begin{cases}
1, & |z_j| > \alpha,\\
0, & |z_j| \leq \alpha,
\end{cases}
\qquad
J_\alpha(z, \alpha)_j
\;=\;
\begin{cases}
-\operatorname{sign}(z_j), & |z_j| > \alpha,\\
0, & |z_j| \leq \alpha.
\end{cases}
$$

**KKT view inputs.** On $A$, $s_A(\beta_A;\alpha) = \alpha\,\operatorname{sign}(\beta_A)$. Hence

$$
\partial_\beta s_A \;=\; 0, \qquad
\partial_\alpha s_A \;=\; \operatorname{sign}(\beta_A) =: r_{L_1}.
$$

Plug into the linear system: $M_{AA} = H_{L,AA}$ (no penalty
curvature), $r = \operatorname{sign}(\beta_A)$.
{func}`sparho.implicit_forward`'s `match` arm:

```{code-block} python
case L1():
    return float(-np.dot(sign_A, v))
```

## ElasticNet

$$
R(\beta;\alpha) \;=\;
\alpha \,\Big(\rho \,\|\beta\|_1 \;+\; \frac{1-\rho}{2}\,\|\beta\|^2\Big),
\qquad \rho \in (0, 1].
$$

{cite}`Zou2005ElasticNet`. `ρ` is structural; the optimized scalar
is `α`. The variant `ρ = 1` recovers L1 exactly (the prox kernel
checks this case for free).

**Prox** (`prox_elastic_net`):
$\operatorname{prox}_{\gamma R}(z)_j = \operatorname{soft}(z_j,\, \gamma\alpha\rho) \;/\;
   \big(1 + \gamma\alpha(1-\rho)\big)$.

**Prox Jacobian** (`prox_jacobian_elastic_net`):
let $d = 1 + \alpha(1-\rho)$, $t = \alpha\rho$. On the active branch
$|z_j| > t$:

$$
J_z(z,\alpha)_{jj} \;=\; \frac{1}{d}, \qquad
J_\alpha(z,\alpha)_j \;=\;
   -\frac{\rho\,\operatorname{sign}(z_j)}{d}
   \;-\;
   \frac{(z_j - t\,\operatorname{sign}(z_j))(1-\rho)}{d^2}.
$$

**KKT view inputs.** On $A$,
$s_A(\beta_A;\alpha) = \alpha\,(\rho\,\operatorname{sign}(\beta_A) + (1-\rho)\,\beta_A)$. Hence

$$
\partial_\beta s_A \;=\; \alpha(1-\rho)\, I,
\qquad
\partial_\alpha s_A \;=\; \rho\,\operatorname{sign}(\beta_A) + (1-\rho)\,\beta_A.
$$

`implicit_forward`'s `match` arm:

```{code-block} python
case ElasticNet(rho=rho):
    r = rho * sign_A + (1.0 - rho) * beta_A
    return float(-np.dot(r, v))
```

and `_build_hess_matvec` adds `α(1-ρ)·v` to the data-side matvec.

## WeightedL1

$$
R(\beta;\alpha) \;=\; \sum_{j=1}^{p} \alpha_j\, |\beta_j|,
\qquad \alpha \in \mathbb{R}^p_{>0}.
$$

Per-feature shrinkage; underlies the adaptive Lasso
{cite}`Zou2006Adaptive`. The optimized hyperparameter is the vector
$\alpha$ — one knob per feature — which makes grid search intractable
and is the canonical motivation for hypergradient-based tuning.

**Prox** (`prox_weighted_l1`):
$\operatorname{prox}(z)_j = \operatorname{soft}(z_j,\, \gamma\alpha_j)$.

**Prox Jacobian** (`prox_jacobian_weighted_l1`): diagonal in $z$,
$J_z(z,\alpha)_{jj} = \mathbf{1}\{|z_j| > \alpha_j\}$; diagonal in
$\alpha$, $J_\alpha(z,\alpha)_j = -\operatorname{sign}(z_j) \cdot \mathbf{1}\{|z_j| > \alpha_j\}$.

**KKT view inputs.** On $A$, $s_{A,j} = \alpha_j\, \operatorname{sign}(\beta_j)$. So

$$
\partial_\beta s_A \;=\; 0, \qquad
\big(\partial_\alpha s_A\big)_{jk}
   \;=\;
   \operatorname{sign}(\beta_j)\,\delta_{jk}.
$$

$\partial_\alpha s_A$ is diagonal: scaling $\alpha_j$ only affects
$s_{A,j}$. The output hypergradient is a length-$p$ vector with the
inactive entries identically zero — which is exactly the
`implicit_forward` `match` arm:

```{code-block} python
case WeightedL1():
    out = np.zeros(n_features, dtype=np.float64)
    out[active] = -sign_A * v       # entrywise: -sign(β_j) · v_j
    return out
```

`v_j` from the CG solve is $\big(M_{AA}^{-1}\,\partial C/\partial \beta_A\big)_j$; the chain rule unrolls per coordinate.

## GroupL1

$$
R(\beta;\alpha) \;=\; \alpha \sum_{k=1}^{K} w_k \,\|\beta_{G_k}\|_2,
\qquad w_k > 0,
$$

with $\{G_k\}$ a partition of $\{0,\dots,p-1\}$. Default
$w_k = \sqrt{|G_k|}$ ({cite}`Yuan2006GroupLasso`), which makes the
penalty invariant to group size.

**Prox** (`prox_group_l1`): block soft-thresholding. For each
group, with $r_k = \|z_{G_k}\|$,

$$
\operatorname{prox}_{\gamma R}(z)_{G_k}
\;=\;
\max\!\Big(0,\,1 - \frac{\gamma \alpha w_k}{r_k}\Big)\, z_{G_k}.
$$

If $r_k \leq \gamma \alpha w_k$ the whole block is zeroed.

**Subdifferential** at the optimum, on active groups:
$\partial_{\beta_{G_k}} R = \alpha w_k\, \beta_{G_k}/\|\beta_{G_k}\|
   = \alpha w_k\, u_k$, with $u_k = \beta_{G_k}/\|\beta_{G_k}\|$.

**KKT view inputs.** On an active group $G_k$ with $r_k = \|\beta_{G_k}\|$,
$s_{G_k} = \alpha w_k\,\beta_{G_k}/\|\beta_{G_k}\|$. Differentiating
in $\beta_{G_k}$ (using $\partial(x/\|x\|)/\partial x = (I - x x^\top/\|x\|^2)/\|x\|$):

$$
\partial_{\beta_{G_k}} s_{G_k}
\;=\;
\frac{\alpha w_k}{r_k}\,\big(I_{|G_k|} - u_k u_k^\top\big).
$$

This is the **block-diagonal penalty curvature** in
`_build_hess_matvec`'s `GroupL1` arm. $I - u_k u_k^\top$ is the
orthogonal projector onto $u_k^\perp \subset \mathbb{R}^{|G_k|}$; it
has eigenvalues $1$ (multiplicity $|G_k|-1$) and $0$ (multiplicity 1,
the $u_k$ direction itself). Geometrically: shrinking $\beta_{G_k}$
along $u_k$ doesn't change the subgradient direction, so the
curvature acts only orthogonal to $u_k$.

For the **$\alpha$ derivative**: $\partial_\alpha s_{G_k} =
   w_k\, u_k$, a length-$|G_k|$ block lifted into the concatenated
active layout. {func}`sparho.implicit_forward`'s `match` arm builds
this block-wise:

```{code-block} python
case GroupL1():
    jac_alpha = np.empty_like(v)
    for k_idx, w_k in enumerate(group_info.weights):
        s, e = int(starts[k_idx]), int(starts[k_idx + 1])
        jac_alpha[s:e] = w_k * group_info.u_concat[s:e]
    return float(-np.dot(jac_alpha, v))
```

The trace of each block on $G_k$ is $(|G_k| - 1)\,\alpha w_k / r_k$,
used by `_resolve_ridge` to compute the operator's natural diagonal
scale for auto-ridge.

## Summary table

```{list-table}
:header-rows: 1
:widths: 15 28 28 14 15

* - Penalty
  - $\partial_\beta s_A$
  - $\partial_\alpha s_A = r$
  - Hypergrad
  - Prox kernel
* - `L1`
  - $0$
  - $\operatorname{sign}(\beta_A)$
  - scalar
  - `prox_l1`
* - `ElasticNet(ρ)`
  - $\alpha(1-\rho)\,I$
  - $\rho\,\operatorname{sign}(\beta_A) + (1-\rho)\,\beta_A$
  - scalar
  - `prox_elastic_net`
* - `WeightedL1`
  - $0$
  - $\operatorname{diag}(\operatorname{sign}(\beta_A))$
  - vector
  - `prox_weighted_l1`
* - `GroupL1(w)`
  - block-diag $\tfrac{\alpha w_k}{r_k}(I - u_k u_k^\top)$
  - block-stacked $w_k u_k$
  - scalar
  - `prox_group_l1`
```

Three observations from the table:

1. **L1 and WeightedL1 have zero penalty curvature.** Their
   $M_{AA}$ is just $H_{L,AA}$ — pure data side. Ridge stabilization
   is more important for these because the curvature has nothing to
   add to ill-conditioned $X_A^\top X_A$.
2. **ElasticNet adds a uniform positive shift.** $\alpha(1-\rho)\,I$
   is the cheapest possible regularizer of the linear system; this is
   why ElasticNet is numerically friendlier than Lasso for
   implicit-diff even before we add the ridge.
3. **GroupL1's penalty curvature is rank-deficient** — it is zero
   along each $u_k$. The data Hessian $H_{L,AA}$ has to supply the
   strict positive definiteness on that direction (which it does
   under the same generic full-rank-$X_A$ assumption).

## Extending: how to add a new penalty

The closed `Penalty` union is meant to give mypy enough information
to flag missed dispatch. To add `MyPenalty`:

1. Add a `@dataclass(frozen=True, slots=True)` variant in
   `python/sparho/problem.py` and export from `__init__.py`.
2. Implement the prox (and, if needed, prox-Jacobian) kernels in
   `crates/sparho-core/src/prox.rs`. Expose via
   `crates/sparho-py/src/lib.rs`. Update the typed stub
   `python/sparho/_core.pyi`.
3. Derive $\partial_\beta s_A$ and $\partial_\alpha s_A$. Add a
   `case MyPenalty(): ...` arm in **every** `match` over `Penalty`
   in `python/sparho/hypergrad.py`. mypy strict mode will flag any
   you miss via the trailing `case _: assert_never(penalty)`.
4. Update this page's summary table.

See {doc}`implicit_diff` for the structural argument, and
`crates/sparho-core/src/prox.rs` for the Rust style.
