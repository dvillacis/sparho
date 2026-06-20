# Protocols & extension points

sparho deliberately exposes a small extension surface — four pluggable
shapes that cover the things downstream users actually want to change.

| Surface | Kind | What it does |
|---|---|---|
| `Solver` | Protocol | Drive the inner problem at a fixed `α`. |
| `Criterion` | Protocol | Evaluate the outer loss + gradient `∂C/∂β`. |
| Datafit / Penalty | Tagged unions | Add new `match`-dispatched problem variants. |
| Hypergrad function | Callable | Swap the implicit-diff solver itself (rarely needed at v0.1). |

The `Optimizer` Protocol that was sketched in the original plan was
folded into `grad_search` / `hoag_search`. If you need a fundamentally
different outer step (Adam, L-BFGS), write a new search function — the
shared boundary is the `Criterion.value_and_hypergrad` call, not an
optimizer state object.

## `Solver`

A solver is anything callable with the canonical signature:

```python
from sparho import Problem, SolverResult
from sparho.core.types import Array, Hyperparam

def my_solver(
    problem: Problem,
    hyperparam: Hyperparam,
    /,
    *,
    x0: Array | None = None,
    tol: float | None = None,
) -> SolverResult:
    ...
```

Return a {py:class}`sparho.SolverResult` with the
estimated `β`, the integer `active_set` (sorted ascending, `int32`),
a dual-gap (or stationarity proxy), and the inner iteration count.

- `x0` is an **optional** warm-start guess. If your solver supports it,
  use it. If not, ignore it — the outer loop will still work, just slower.
- `tol` is an **optional** override of the adapter's default inner
  tolerance. `hoag_search` uses this to schedule inner accuracy across
  outer iterations. Adapters with no meaningful `tol` may ignore it.

The cheapest way to plug in is {py:func}`sparho.adapters.as_solver`:

```python
from sparho.adapters import as_solver

solver = as_solver(my_solver, name="my-coord-descent")
```

The wrapper introspects your function's signature and forwards `x0` /
`tol` only when they are declared — a plain `(problem, hp) -> SolverResult`
callable keeps working.

If you can't satisfy the protocol (e.g. you want to validate inputs in
`__post_init__`), write a frozen dataclass with `__call__` matching the
signature above — that's how the bundled adapters are written.

### What `active_set` has to be

The implicit-differentiation hypergradient is restricted to the active
set you return. A wrong active set produces a wrong hypergradient — there
is no separate sanity check. Use the bundled helper:

```python
from sparho.adapters._common import active_set_of

active = active_set_of(coef)  # int32 indices where |coef| > 0
```

For penalties whose support isn't simply `coef != 0` (group lasso, fused
lasso), report the *generalized* active set — the set of indices the inner
solver considers "free" at convergence. Subgradient slack on the boundary
is documented as a v0.2 concern.

## `Criterion`

A criterion exposes two methods:

```python
class MyCriterion:
    def value(self, problem, hp, solver, *, x0=None, tol=None) -> float:
        ...

    def value_and_hypergrad(
        self, problem, hp, solver, hypergrad_fn, *, x0=None, tol=None
    ) -> CriterionResult:
        ...
```

- `value` is the cheap value-only path used by trial steps. Don't compute
  derivatives here.
- `value_and_hypergrad` runs the inner solver, evaluates the outer loss,
  computes `∂C/∂β` at the converged `β*`, and calls `hypergrad_fn` to chain
  it through the implicit-diff solve. Return a
  {py:class}`sparho.CriterionResult`.

The bundled `HeldOutMSE`, `HeldOutLogistic`, `CrossVal` are 60-line
frozen dataclasses; copy and adapt. A common case is "the same MSE but
weighted observations" — write a new dataclass; don't subclass.

## Datafit / Penalty (tagged unions)

The v0.1 `Datafit = SquaredLoss | LogisticLoss` and
`Penalty = L1 | ElasticNet | WeightedL1` unions are closed. To add a new
variant:

1. Add a `@dataclass(frozen=True, slots=True)` to
   `python/sparho/problem.py` and export it from `python/sparho/__init__.py`.
2. Implement the corresponding Rust kernels in `crates/sparho-core` and
   expose them via `crates/sparho-py/src/lib.rs`. Update the type stubs
   in `python/sparho/_core.pyi`.
3. Add a `case` arm in every `match` over the union. At minimum
   {py:func}`sparho.implicit_forward`; plus
   any adapter / criterion that dispatches on the union. Leave the
   `case _: assert_never(x)` tail intact — mypy strict mode is the safety
   net for missed cases.

There is no inheritance hierarchy to subclass. If a new datafit doesn't
fit (e.g. a non-separable penalty needs a `prox` rather than a closed-form
Jacobian) lift the abstraction one notch higher, but do it in a PR — the
union shape is a deliberate v0.1 constraint.

## Hypergradient function

`implicit_forward` is the default; `forward`, `backward`, and `implicit` ship
alongside it (and `WarmStartHypergrad` wraps any of them). The shared signature is

```python
HypergradFn = Callable[..., Hyperparam]
# (train_problem, hp, solver_result, criterion_grad_beta, **opts) -> Hyperparam
```

Both `grad_search` and `hoag_search` accept it as an optional kwarg:

```python
hoag_search(problem, hp0=1e-2, solver=..., criterion=..., hypergrad=my_hg)
```

Anything with that signature works. If you're writing one, look at the
`python/sparho/hypergrad/` package — the Rust BCD / restricted-Hessian kernels
in `crates/sparho-core/src/bcd.rs` (and `residual.rs`) are usable as-is.
