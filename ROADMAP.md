# sparho roadmap

The authoritative implementation plan lives at
`~/.claude/plans/swirling-soaring-hamster.md`. This file is the short
status summary.

## v0.1 — Library shipped; perf still WIP

**Original target**: beat sparse-ho on libsvm Lasso benchmarks
(`breast-cancer`, `leukemia`, `rcv1.binary`).

**Actual v0.1 status**: library is functionally complete (92 pytest, 11
cargo, mypy strict, single wheel). Perf vs `LassoCV`:

- **`leukemia` (n ≪ p, dense)**: **1.3× faster** (23 s vs 30 s).
- **`breast-cancer` (n ≫ p, small)**: overhead-bound (0.24 s vs 0.01 s
  — both instant; neither interesting).
- **`rcv1.binary` (sparse, large)**: sparho **runs to completion** and
  finds a **better held-out MSE** than `LassoCV` (0.194 vs 0.225) by
  driving α below `LassoCV`'s grid floor (2.1·10⁻⁵ vs 1·10⁻⁴). Wall-time
  is 36× *slower* than `LassoCV` (433 s vs 12 s) because the inner Lasso
  cold-starts at very small α / large active set — addressed by the
  warm-start work in v0.2.

The "Nx faster" headline is a v0.2 deliverable that needs warm-starting
of the inner solver. Hypergradient-CG stability is **fixed at v0.1** via
ridge regularization (`implicit_forward(..., ridge=ε)`, auto-scaled to
`10⁻¹⁰ · trace(M_AA)/|A|`, bit-identical results on the well-conditioned
benchmark across 8 orders of magnitude of ε).

What v0.1 actually ships:
- Correct gradient-based outer loop with FD-validated implicit-diff
  hypergradients.
- Vector-α support (`WeightedL1`) — something `LassoCV` cannot do.
- Sparse-X-first-class Rust kernels for prox + CSC matvec + restricted
  Hessian-vector products (`sparho._core.*`).
- sklearn + celer + callable adapters.
- `HeldOutMSE`, `HeldOutLogistic`, `CrossVal` criteria with sklearn
  `LassoCV` parity verified (`tests/test_criteria.py`).
- `GradDescent`, `LineSearch` outer optimizers.
- Log-space `grad_search` orchestration with full-data refit.
- PyPI-grade single-wheel install via maturin + PyO3 ABI3 py311.

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo scaffold (uv + maturin + cargo workspace + PyO3 ABI3 + CI) | ✅ done |
| 1 | Core types (`Problem`, `SolverResult`, `SearchState`, Protocols) | ✅ done |
| 2 | Rust kernels (`kernels`, `prox`, `csc`, `residual`) | ✅ done |
| 3 | Adapters (sklearn, celer, callable) | ✅ done |
| 4 | `ImplicitForward` hypergradient | ✅ done |
| 5 | Outer optimizers (`grad_descent`, `line_search`) | ✅ done |
| 6 | Criteria (`held_out_mse`, `held_out_logistic`, `cross_val`) | ✅ done |
| 7 | `grad_search` orchestration | ✅ done |
| 8 | Benchmarks (script + README; perf gap documented) | ✅ done |
| 8.5 | Hypergradient-CG ridge stabilization + NaN guards (rcv1 now runs) | ✅ done |
| 9 | Sphinx docs + sparse-ho migration guide | ✅ done |
| 10 | Release v0.1.0 to PyPI | ⏳ local prep done; external steps pending — see `RELEASE.md` |

## v0.2 — warm-start, celer, skein backend

Hypergradient-CG stability landed at v0.1 (see above); warm-starting is
the next perf gap to close, supported by spike measurements.

1. **Warm-start the inner solver across outer iterations.** Pass
   `β*_prev` as the starting point for the next inner solve. v0.1 spike
   (`benchmarks/spike_warmstart.py`) measured ~ 2× wall on dense
   (`leukemia` 38 s → 22 s; `breast-cancer` 0.24 s → 0.12 s, same α/MSE
   on both). On `rcv1.binary` the gain is expected to be much larger
   because the inner solver at small α / large active set dominates the
   433 s wall time; warm-starting from `β*_prev` should cut inner-iter
   count by an order of magnitude on the late outer iters. Touches
   `Solver` Protocol (optional `x0` arg) and a per-fold `prev_coef`
   carry through `CrossVal`. Plan: `~/.claude/plans/<separate-plan>.md`.
2. **`sparho.adapters.skein`** — adapter for skein's nonconvex
   weighted/group penalties. Open question: does skein expose enough KKT
   state today, or do we add `kkt_residual` / `active_set` returns to
   its solvers first?
3. **Re-run benchmarks** with (1) + celer adapter as the v0.2 perf
   story; refresh `benchmarks/README.md` and main `README.md` headline.
4. **Reproducibility tooling** — wall-time numbers reproduce within
   30 % across runs at v0.1; the plan's 10 % tolerance is a v0.2 target
   that needs a `pyperf`-style steady-state runner.

## v0.1 work that already landed beyond plan scope

- **Hypergradient-CG stability**
  (plan: `~/.claude/plans/lets-brainstorm-on-ways-sleepy-meerkat.md`).
  `implicit_forward(..., ridge=ε)` with auto-scaled default; non-finite
  CG output produces a zero hypergradient with `RuntimeWarning` rather
  than propagating to a `α = nan` crash. NaN-guards added to
  `LineSearch`, `GradDescent`, and `grad_search`. Regression tests for
  near-singular Hessians (`tests/test_hypergrad.py::test_implicit_forward_ridge_*`)
  and NaN-handling (`tests/test_optimizer.py::test_*_holds_param_on_*`).

## Deferred / out of scope

These were considered and explicitly excluded from v0.1 in the plan
discussion:

- Forward / Backward / non-Forward Implicit hypergradient modes (only
  `ImplicitForward` ships at v0.1).
- Convenience sklearn-shaped wrapper classes (`LassoHO`, etc.). Revisit
  after the protocol is stable.
- Imaging operators (Gaussian blur, Radon, TV, wavelet).
- JAX / torch backends. GPU.
- SURE (Stein's Unbiased Risk Estimate) criterion — listed in original
  plan, dropped from Phase 6 since the v0.1 audience tunes via held-out
  validation, not unsupervised SURE. Revisit if a downstream user asks.
- Owner-paper features (`ImplicitVariational`, `TrustRegion`,
  `NonMonotoneLineSearch`) — depend on the non-Forward Implicit mode.

## Known unmet plan targets

For transparency, these explicit plan targets did not land at v0.1:

- **"beat sparse-ho on libsvm Lasso benchmarks"** — sparse-ho was not
  installed for comparison; the bench runs against `LassoCV` instead.
  Status by dataset: `breast-cancer` overhead-bound (both <1 s);
  `leukemia` **1.3× faster** (23 s vs 30 s); `rcv1.binary` **runs and
  finds a better MSE** (0.194 vs 0.225) but at 36× higher wall-time —
  the inner solver cold-starts at very small α / large active set, which
  warm-starting (v0.2 item 1) is expected to address.
- **"benchmarks reproduce within 10%"** — currently ~ 30 % jitter across
  runs. Needs `pyperf`-style steady-state methodology.
