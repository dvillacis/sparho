# sparho

Nonsmooth bilevel hyperparameter optimization via implicit differentiation.

A maintained, performant successor to [`sparse-ho`](https://github.com/QB3/sparse-ho)
(ICML 2020, dormant since 2022). Tunes hyperparameters of non-smooth estimators
(Lasso, ElasticNet, weighted Lasso, sparse logistic regression) by computing
the hypergradient via implicit differentiation rather than grid/random search.

**Status:** pre-alpha. Public API may change between minor versions until v1.0.

## Why

Implicit-differentiation HP optimization can be orders of magnitude faster
than `LassoCV`-style grid search when you have a held-out criterion, but
the existing libraries are dormant (`sparse-ho`) or no longer maintained
(`JAXopt`). `sparho` is a clean-break, scipy-stack-native Rust+Python
implementation built for the same target audience.

## v0.1 honest perf summary

| dataset | shape | sparho | `LassoCV` | notes |
|---|---|---|---|---|
| `breast-cancer` | 683×10 | 0.24 s | 0.01 s | overhead-bound; both finish instantly |
| `leukemia` | 38×7129 | 23.1 s | 30.1 s | **1.3× faster**; vector-α uniquely supported |
| `rcv1.binary` | 20242×47236 sparse | 433 s, MSE **0.194** | 12 s, MSE 0.225 | **better MSE, slower wall** (see below) |

What v0.1 delivers:

- Correct gradient-based outer loop with full FD parity vs analytic
  hypergradients.
- **Vector-α support** (`WeightedL1`, per-feature regularization) — something
  `LassoCV` cannot do.
- Implicit-diff hypergradients via Rust kernels on the hot path; ridge-
  stabilized CG so the active-set-restricted linear solve does not
  diverge on ill-conditioned sparse designs.
- Clean Protocol-based API for sparse-ho refugees.
- 92 pytest, mypy strict, clippy clean, single wheel via maturin.

**The rcv1 story.** Implicit differentiation lets sparho search past
`LassoCV`'s discrete grid: on `rcv1.binary`, sparho's outer loop drives
α down to `2.1·10⁻⁵`, well below `LassoCV`'s grid floor of `1·10⁻⁴`,
and lands on a **better held-out MSE** (0.194 vs 0.225). The wall-time
penalty (433 s vs 12 s) comes from the inner Lasso solver being cold-
started at every outer iter — see the v0.2 plan for warm-starting, which
the v0.1 spike (`benchmarks/spike_warmstart.py`) measured at ~2× on
dense cases.

## Install (planned)

```bash
pip install sparho               # release wheel, no Rust toolchain needed
pip install "sparho[celer]"      # add celer as a fast Lasso adapter
```

## Quickstart (planned API)

```python
from sparho import Problem, grad_search
from sparho.adapters import SklearnLasso
from sparho.criteria import held_out_mse
from sparho.optimizer import grad_descent
from sparho.hypergrad import implicit_forward

problem = Problem.lasso(X, y)
result = grad_search(
    problem,
    hp0=1e-3,
    solver=SklearnLasso(),
    hypergrad=implicit_forward,
    criterion=held_out_mse(idx_train, idx_val, X, y),
    optimizer=grad_descent(lr=1.0),
)
print(result.best_hyperparam, result.best_coef)
```

See `docs/migration_from_sparse_ho.md` for translation from sparse-ho's API.

## Design

- **One `Problem` dataclass.** No abstract base class tower. Algorithms are
  free functions over `Problem`. Typing via `typing.Protocol`.
- **Implicit-only at v0.1.** ImplicitForward is the only hypergradient mode.
- **Sparse-X first class.** CSC iterated directly in Rust; no densification.
- **Rust kernels via PyO3 + maturin + ABI3.** Single binary wheel, no numba.
- **Clean break from sparse-ho.** Migration guide rather than compat shim.

## Roadmap

See `ROADMAP.md`. v0.1 ships sklearn + celer + callable adapters with
verified correctness and dense-high-d parity vs `LassoCV`. v0.2 closes the
two remaining perf gaps — inner-solver warm-starting and hypergradient
stability on large sparse designs — and adds a `skein` backend for
nonconvex weighted/group penalties.

## License

BSD 3-Clause.
