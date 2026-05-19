# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project

`sparho` is a Python + Rust library for nonsmooth bilevel hyperparameter
optimization via implicit differentiation. It is the maintained,
performant successor to `sparse-ho` (QB3, ICML 2020, dormant since 2022).
Target audience: people who would have reached for `sparse-ho`.

Two-layer package: Rust core (`sparho._core`, compiled via PyO3) houses
the compute kernels — prox operators, CSC iteration, KKT residuals;
Python keeps the orchestration (problem definition, hypergradient
linear-solve, criteria, optimizers, `grad_search`).

Authoritative plan: `~/.claude/plans/swirling-soaring-hamster.md`.
Short status: `ROADMAP.md`. Per-release narrative: `CHANGELOG.md`.

## Module map

Python side (`python/sparho/`):
- `problem.py` — `Problem` dataclass + closed tagged unions `Datafit =
  SquaredLoss | LogisticLoss` and `Penalty = L1 | ElasticNet | WeightedL1`.
- `state.py` — frozen `SolverResult`, `IterationRecord`, `SearchState`,
  `SearchResult`. All state is immutable; outer history is a `tuple`.
- `solver.py` — `Solver` protocol: `(Problem, Hyperparam) -> SolverResult`.
- `hypergrad.py` — `implicit_forward`: KKT-restricted CG on the active set,
  dispatches on `(datafit, penalty)` via `match` + `assert_never`.
- `criteria.py` — `HeldOutMSE`, `HeldOutLogistic`, `CrossVal`; each returns
  `CriterionResult(value, grad_beta)`.
- `optimizer.py` — `GradDescent`, `LineSearch`; both satisfy the
  `step(value, grad, state) -> (new_param, new_state)` protocol.
- `search.py` — `grad_search`, the only imperative for-loop in the package.
- `adapters/` — `SklearnLasso`, `CelerLasso`, `CallableSolver`. Each adapts
  an external inner solver to the `Solver` protocol.
- `core/types.py` — `Array`, `DesignMatrix`, `Hyperparam`, `IndexArray`,
  `Scalar` type aliases used everywhere.
- `_core.pyi` — authoritative signature surface for the Rust extension
  (`sparho._core`). Update it whenever you add a Rust-exposed function.

Rust side (`crates/`):
- `sparho-core` — pure-Rust kernels: `prox.rs`, `kernels.rs`, `csc.rs`,
  `residual.rs`. No PyO3 dependency; unit-testable with `cargo test`.
- `sparho-py` — thin PyO3 bindings; `src/lib.rs` is the `#[pymodule]`.

## Adding a new datafit or penalty

The unions in `problem.py` are deliberately closed so mypy flags missed
dispatch. To add one:
1. Add a `@dataclass(frozen=True, slots=True)` variant to `Datafit` or
   `Penalty` in `problem.py` and export it from `__init__.py`.
2. Implement the kernel (prox, Jacobian, residual term) in
   `crates/sparho-core` and expose it via `crates/sparho-py/src/lib.rs`.
   Update `python/sparho/_core.pyi` with the signature.
3. Add a `case` arm in every `match` over the union — at minimum
   `hypergrad.implicit_forward`, plus any criterion/adapter that
   dispatches. Leave the `case _: assert_never(x)` tail intact; mypy
   strict mode is your safety net.

## Design pillars (non-negotiable)

1. **One core type, not seven.** `Problem(datafit, penalty, design, target)`
   dataclass + free-function algorithms. No ABC tower. Typing via
   `typing.Protocol`, not inheritance.
2. **Functional core, imperative shell.** State is an immutable dataclass;
   algorithms are pure functions; the outer loop is a plain `for`.
3. **Implicit-only at v0.1.** Only `ImplicitForward` ships.
4. **Sparse-X first-class.** CSC iterated directly in Rust; no densification.
5. **Rust kernels via PyO3 + maturin + ABI3.** No numba, no pure-Python
   fallback.
6. **Single outer-optimizer protocol.** `step(value, grad, state) -> (new_param, new_state)`.
7. **No legacy compatibility shims.** Clean-break API; sparse-ho migration
   is a translation table in `docs/migration_from_sparse_ho.md`.

## Build & test

```bash
uv sync --extra dev
uv run maturin develop --release        # iterative Rust rebuild
uv run pytest                           # all python tests
uv run pytest tests/test_smoke.py       # single file
uv run ruff check python/ tests/
uv run mypy
uv run pre-commit run --all-files

cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p sparho-core --lib
```

## When to port to Rust

Profiling-driven. Port:
1. Element-wise piecewise math (`np.where(np.where(...))` patterns).
2. Python-loop coordinate updates that don't vectorize to BLAS.
3. Hot kernels called many times per outer iteration.

Don't port:
- BLAS-bound matvecs — numpy already calls BLAS.
- Once-per-outer-iter scalar/tiny-vector math — FFI overhead exceeds work.
- scipy iterative solvers (GMRES / CG) — already BLAS-bound through
  matrix-free callbacks.
- Pure orchestration (`grad_search`, `Monitor`, optimizers).

## Conventions

- Style: ruff with pydocstyle (numpy convention); strict mypy on
  `python/sparho`. Tests exempt from `D` rules. Line length 100.
  `from __future__ import annotations` everywhere.
- Rust style: `cargo fmt` + `cargo clippy -- -D warnings`.
- Optional deps grouped: `celer`, `dev`, `docs`, `bench`.
- Do **not** import design patterns from sibling repos in
  `/Users/davidvillacis/workspace/2026/` (bihop, sparse-ho-*). Substrate
  reuse from `skein` (build config) is fine; API/abstraction reuse is not.
