# sparho

[![CI](https://github.com/dvillacis/sparho/actions/workflows/ci.yml/badge.svg)](https://github.com/dvillacis/sparho/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/dvillacis/sparho/branch/main/graph/badge.svg)](https://codecov.io/gh/dvillacis/sparho)
[![PyPI](https://img.shields.io/pypi/v/sparho.svg)](https://pypi.org/project/sparho/)
[![Python](https://img.shields.io/pypi/pyversions/sparho.svg)](https://pypi.org/project/sparho/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

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

## Perf summary

v0.2 numbers (HOAG + warm-start + celer inner solver), single-threaded on
an Apple M-series; see `benchmarks/README.md` for the methodology and the
v0.1 historical row.

| dataset | shape | sparho | `LassoCV` | notes |
|---|---|---|---|---|
| `breast-cancer` | 683×10 | 0.26 s | 0.02 s | overhead-bound; both finish instantly |
| `leukemia` | 38×7129 | **0.58 s** | 19.0 s | **32.8× faster** (was 1.3× at v0.1) |
| `rcv1.binary` | 20242×47236 sparse | 211 s, MSE **0.194** | 22.6 s, MSE 0.225 | **better MSE** (see below); 2× wall faster than v0.1 |

What v0.2 delivers on top of v0.1:

- `hoag_search` outer loop (Pedregosa 2016): adaptive step from a
  Lipschitz proxy, `+C·tol` slack acceptance, exponentially-decreasing
  inner-tol schedule. Replaces `LineSearch`.
- Inner-solver warm-starting threaded through the `Solver` Protocol +
  every adapter + `CrossVal(warm_start=True)`.
- celer adapter recommended for the high-d regime — compounding the
  HOAG/warm-start win into 32.8× on `leukemia` and 1.65× faster than
  sklearn on `rcv1.binary`.
- Dense-matvec fix in `implicit_forward` (no `coo_tocsr` round-trip on
  dense designs): 8.4× faster hypergradient solve on `leukemia`.

Everything v0.1 delivered still holds: gradient-based outer loop with
full FD parity, vector-α (`WeightedL1`) which `LassoCV` cannot do,
ridge-stabilized hypergradient-CG on ill-conditioned sparse designs,
clean Protocol-based API, mypy strict + clippy clean, single wheel via
maturin.

**The rcv1 story.** Implicit differentiation lets sparho search past
`LassoCV`'s discrete grid: on `rcv1.binary`, sparho's outer loop drives
α down to `2.1·10⁻⁵`, well below `LassoCV`'s grid floor of `1·10⁻⁴`,
and lands on a **14 % better held-out MSE** (0.194 vs 0.225). The
wall-time gap halved at v0.2 (433 s → 211 s) thanks to warm-start +
celer; the remaining gap is irreducible inner-solver work at very small
α / large active set. sparho's win on this dataset is *quality per
outer iter*, not raw wall time.

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
- **Full hypergradient family; ImplicitForward by default.** `implicit_forward`,
  `forward`, `backward`, and `implicit` all ship as `hypergrad=` choices.
- **Sparse-X first class.** CSC iterated directly in Rust; no densification.
- **Rust kernels via PyO3 + maturin + ABI3.** Single binary wheel, no numba.
- **Clean break from sparse-ho.** Migration guide rather than compat shim.

## Roadmap

See `ROADMAP.md`. v0.1 shipped sklearn + celer + callable adapters with
verified correctness and dense-high-d parity vs `LassoCV`. v0.2 closes
the inner-solver warm-starting and hypergradient-stability gaps and
lands the HOAG outer loop — 32.8× on `leukemia`, 2× wall on
`rcv1.binary`. v0.3 lands the sklearn-ecosystem wrappers (`LassoHO`,
`ElasticNetHO`, `LogisticRegressionHO`) so sparho slots into
`Pipeline` / EconML / MLflow, the `SURE` / `GSURE` criterion for
unsupervised tuning, a `MultiTaskLasso` / Group-L1 penalty, and adapters
for `skein` (nonconvex weighted/group) and `skglm` (MCP / SCAD / SLOPE /
Group / Huber / Poisson). See `docs/feature_research.md` for the
2026-05-20 landscape synthesis behind these picks.

## How to cite

If you use `sparho` in academic work, please cite it. The repository ships a
[`CITATION.cff`](CITATION.cff) — GitHub renders a "Cite this repository" widget
in the right-hand sidebar that produces BibTeX, APA, and other formats from it.

Each tagged release also mints a [Zenodo](https://zenodo.org/) DOI via the
GitHub–Zenodo integration. Cite the *concept* DOI (resolves to all versions)
when you want to refer to the project as a whole, or a *version* DOI for
reproducibility. The DOI badge below is a placeholder until the first tagged
release (`v0.5.0`):

<!-- Replace `XXXXXXX` with the concept DOI after the first Zenodo release.
     See RELEASE.md § Zenodo DOI integration for the procedure. -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

```bibtex
@software{sparho,
  author  = {Villacis, David},
  title   = {sparho: nonsmooth bilevel hyperparameter optimization via implicit differentiation},
  url     = {https://github.com/dvillacis/sparho},
  doi     = {10.5281/zenodo.XXXXXXX},
  version = {0.5.0},
  year    = {2026}
}
```

The original `sparse-ho` algorithm should be cited alongside `sparho`:

```bibtex
@inproceedings{bertrand2020implicit,
  author    = {Bertrand, Quentin and Klopfenstein, Quentin and Blondel, Mathurin
               and Vaiter, Samuel and Gramfort, Alexandre and Salmon, Joseph},
  title     = {Implicit Differentiation of Lasso-Type Models for Hyperparameter Optimization},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
  year      = {2020},
  url       = {https://arxiv.org/abs/2002.08943}
}
```

## Community

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — dev setup, gates, contribution flow.
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1.
- [`SECURITY.md`](SECURITY.md) — vulnerability disclosure policy.

## License

BSD 3-Clause.
