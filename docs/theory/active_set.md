# Active sets and why we restrict

The implicit-diff linear system in {doc}`implicit_diff` is $|A| \times
|A|$, not $p \times p$. This page justifies that reduction: the
**inactive** coordinates contribute identically zero to $d\beta^\star/
d\alpha$ under a strict-activity assumption that holds on a
measure-one set of hyperparameters, and the **restricted Hessian** is
SPD generically. For `GroupL1` the active-set definition is slightly
different — "active *groups*" rather than "active coordinates" — and
we cover that variant explicitly.

## Active set, formally

For separable per-feature penalties (`L1`, `WeightedL1`, `ElasticNet`):

$$
A(\alpha) \;:=\; \{\, j : \beta^\star_j(\alpha) \neq 0 \,\},
\qquad
I(\alpha) \;:=\; \{0,\dots,p-1\} \setminus A(\alpha).
$$

In code this is exactly `SolverResult.active_set` populated by the
inner solver (sklearn / celer / callable adapter) and read by
{func}`sparho.implicit_forward`. The set $A$ is data-dependent and
$\alpha$-dependent; the implicit-diff derivation only needs it to be
locally constant in $\alpha$, which is what we establish next.

For `GroupL1` with groups $G_1, \dots, G_K$:

$$
A(\alpha) \;:=\; \bigcup_{k\,:\,\|\beta^\star_{G_k}\| > 0}\, G_k,
$$

i.e. the union of all coordinates in active *groups*. This is
slightly more than `np.flatnonzero(coef)` because, generically,
`GroupL1`'s prox produces whole-block sparsity (either
$\beta_{G_k} = 0$ componentwise or $\beta_{G_k} \neq 0$ componentwise)
— but it is technically possible for an internal coordinate to land at
zero while its group is active. Those internal-zero coordinates must
still enter the KKT system because the active-group subgradient
$s_{G_k} = \alpha w_k\, \beta_{G_k}/\|\beta_{G_k}\|$ couples every
coordinate of $G_k$ to every other, and zeroing one row would
silently drop a constraint. `hypergrad._resolve_group_l1_active` does
this expansion.

## Why inactive coords have zero hypergradient

Let $j \in I(\alpha_0)$ for some fixed $\alpha_0$. We claim
$\beta^\star_j(\alpha) = 0$ for all $\alpha$ in a neighborhood of
$\alpha_0$ — i.e. the inactive set is locally constant — under the
**strict subgradient inequality**

$$
\|\nabla_j L(X\beta^\star, y)\| \;<\; \partial_j R(0;\alpha)
\quad \text{for all } j \in I(\alpha_0).
\tag{$\star$}
$$

For `L1` this is $|\nabla_j L| < \alpha$; for `WeightedL1` it is
$|\nabla_j L| < \alpha_j$. ($\star$) is the *strict* form of the
KKT optimality condition at zero. Strictness is the active-set
analogue of strict complementarity in interior-point theory.

**Argument.** $\beta^\star(\alpha)$ is continuous in $\alpha$ (by
inner uniqueness + convexity), so $\nabla_j L(X\beta^\star(\alpha), y)$
is continuous too. The function $\alpha \mapsto \partial_j R(0;\alpha)$
is continuous (in fact linear in $\alpha$ for our family of penalties).
By ($\star$) the strict inequality $|\nabla_j L| < \partial_j R(0;\alpha)$
persists in a neighborhood of $\alpha_0$, and the optimality condition
forces $\beta^\star_j(\alpha) = 0$ for every $\alpha$ in that
neighborhood. Hence $d\beta^\star_j/d\alpha = 0$ on the neighborhood.
$\quad\blacksquare$

Symmetrically, for $j \in A(\alpha_0)$ with $\beta^\star_j(\alpha_0)
\neq 0$, continuity keeps $\beta^\star_j(\alpha)$ away from zero in a
neighborhood. The two arguments together: under ($\star$), the active
set is locally constant, so we can treat $A$ as fixed when
differentiating in $\alpha$. This is the "active-set restriction"
underpinning sparho's implementation.

## When ($\star$) fails

($\star$) fails precisely at the **transition hyperparameters** —
values of $\alpha$ where a coordinate enters or leaves $A$. For
`L1` these form a discrete set of "knots" along the regularization
path. They are a measure-zero set in $\alpha$-space, so a generic
outer-loop trajectory traverses them only transiently. Two practical
consequences:

- **The hypergradient may flicker at a transition.** Step across a
  knot and the active set changes; the linear system jumps
  discontinuously and $dC/d\alpha$ has a finite jump. HOAG's
  acceptance test absorbs this — a bad step is rejected and the
  Lipschitz proxy $L$ doubles. See {doc}`convergence`.
- **At an exact transition, the analytic IFT does not apply.**
  Nonsmooth-IFT theory {cite}`Bolte2021NonsmoothImplicit` recovers
  a generalized Clarke Jacobian here, but sparho does not implement
  this — it falls back to the active set the inner solver reports
  and treats the result as one-sided. Bit-for-bit, this is the same
  choice as sparse-ho.

The corresponding measure-zero issue in the prox is documented inline
in `crates/sparho-core/src/prox.rs`: at $|z| = \alpha$ exactly we
follow sparse-ho in calling the coordinate inactive (Jacobian = 0).
That is consistent with the active-set restriction here — the
coordinate sits *on* the kink, but our subgradient choice puts it on
the inactive side.

## $|A| \ll p$ — the regime sparho was designed for

The Lasso solution under a generic design has $|A| \leq n$ almost
surely {cite}`Tibshirani1996Lasso,Zou2007DegreesOfFreedom`, and on the
sparse-recovery regime
$|A| \approx s^\star \ll p$ where $s^\star$ is the true sparsity. The
implicit-diff linear system is then *much* smaller than the inner
problem itself, and CG with the matrix-free operator costs

$$
O\!\big(|A| \cdot (\text{matvec cost on } A)\big)
\;=\;
O\!\big(|A| \cdot n \cdot \overline{\mathrm{nnz}}_A\big)
$$

per outer iteration, where $\overline{\mathrm{nnz}}_A$ is the average
non-zero density of an active column. For `rcv1.binary` (sparse,
$|A| \sim 100$, $n = 20{,}000$, $p = 47{,}236$, $\overline{\mathrm{nnz}}
\approx 80$) this is a few-million-flop CG solve per outer iter — small
compared to one inner Lasso fit.

## SPD generic on $A$

{doc}`implicit_diff` summarizes the SPD argument for $M_{AA}$:

- **`SquaredLoss`.** $H_{L,AA} = \tfrac{1}{n} X_A^\top X_A$ is the
  Gram matrix of the active columns. SPD iff $X_A$ has full column
  rank. For dense designs this holds generically when
  $|A| \leq n$ and the columns are drawn from a continuous
  distribution. For sparse designs the same holds whenever the active
  columns are linearly independent — typical at the relevant
  sparsity regime.
- **`LogisticLoss`.** $H_{L,AA} = X_A^\top \operatorname{diag}(w)
  X_A$ with $w_i = \sigma(z_i)(1-\sigma(z_i)) \in (0, 1/4]$ strictly
  positive at every sample. Same rank condition on $X_A$ ⇒ SPD.
- **Penalty curvature.** PSD by convexity, on the smooth branch. For
  L1 / WL1 it is identically zero; for ElasticNet it is a positive
  scalar shift; for `GroupL1` it adds a PSD block (the
  orthogonal projector $I - u_k u_k^\top$ scaled by a positive
  factor). PSD on top of SPD stays SPD.

The pathological case is dense designs with collinear features at
the boundary of $|A| = n$. The auto-scaled ridge in
{func}`sparho.implicit_forward` handles this (see
{doc}`implicit_diff`).

## `GroupL1` active-set expansion in code

The cleanest way to see the per-group active-set expansion is to read
`hypergrad._resolve_group_l1_active`:

```{code-block} python
for k, g in enumerate(penalty.groups):
    idx = np.fromiter(g, dtype=np.int64, count=len(g))
    beta_g = coef[idx]
    norm_g = float(np.linalg.norm(beta_g))
    if norm_g == 0.0:
        continue                     # whole group inactive
    active_feats.extend(int(j) for j in idx)
    u_chunks.append(beta_g / norm_g) # u_k = β_{G_k}/||β_{G_k}||
    norms.append(norm_g)             # r_k = ||β_{G_k}||
```

The returned `_GroupL1ActiveInfo` carries `active_features` (the union
of all $G_k$ with $\|\beta_{G_k}\| > 0$), per-group `u_concat` and
`group_norms`, and `weights`. The block Hessian curvature in
`_build_hess_matvec` for GroupL1 then iterates over active groups:

```{code-block} python
for k_idx in range(weights.size):
    s, e = int(starts[k_idx]), int(starts[k_idx + 1])
    u_k = u_concat[s:e]
    scale = alpha * weights[k_idx] / norms[k_idx]
    # (I − u_k u_k^T) v_k = v_k − (u_k·v_k) u_k.
    out[s:e] += scale * (v_k - (u_k @ v_k) * u_k)
```

The trace of each block on $G_k$ is $(|G_k|-1)\,\alpha w_k /
\|\beta_{G_k}\|$ — the rank-$(|G_k|-1)$ projector contributes $|G_k|-1$
ones to its eigenvalue spectrum, scaled by $\alpha w_k/r_k$. This shows
up in `_resolve_ridge` when sparho computes the operator's natural
diagonal scale for auto-ridge.

## Recap

- $A$ is the active set of the inner solution, reported by the
  inner solver (and expanded to "active groups" for `GroupL1`).
- Under ($\star$) — strict subgradient inequality on $I$ — the active
  set is locally constant in $\alpha$, so $d\beta^\star/d\alpha$ has
  support contained in $A$.
- $M_{AA}$ is SPD generically (full-column-rank $X_A$), so
  {func}`sparho.implicit_forward`'s CG converges; auto-scaled ridge
  handles the boundary.
- The reduction $p \to |A|$ is the difference between tractable and
  intractable on sparse-recovery problems and is the operational
  reason implicit diff works.

See {doc}`penalties` for the explicit per-variant formulas plugged
into $M_{AA}$ and $r$.
