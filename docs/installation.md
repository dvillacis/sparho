# Installation

`sparho` ships as a single binary wheel via [maturin](https://www.maturin.rs)
+ [PyO3](https://pyo3.rs) (ABI3 — one wheel per OS/arch, all Python ≥ 3.11).
You do **not** need a Rust toolchain to install from PyPI.

## From PyPI (planned for v0.1.0)

```bash
pip install sparho
```

The `[celer]` extra pulls in [celer](https://mathurinm.github.io/celer/) as a
fast coordinate-descent Lasso solver:

```bash
pip install "sparho[celer]"
```

## From source

A Rust toolchain (stable, ≥ 1.75) and Python ≥ 3.11 are required. Editable
development installs use [`uv`](https://docs.astral.sh/uv/) and `maturin
develop`:

```bash
git clone https://github.com/dvillacis/sparho.git
cd sparho
uv sync --extra dev
uv run maturin develop --release
uv run pytest
```

## Optional extras

| Extra | Purpose |
|---|---|
| `celer` | Fast coordinate-descent solver adapters. |
| `dev` | pytest, ruff, mypy, pre-commit. |
| `docs` | Sphinx + Furo + sphinx-gallery + numpydoc + myst-parser. |
| `bench` | `libsvmdata`, `pandas`, `matplotlib` for the benchmark scripts. |

## Verifying the install

```python
import sparho
print(sparho.__version__)

from sparho import _core
import numpy as np

z = np.array([0.5, -0.2, 1.5])
print(_core.prox_l1(z, 0.3))  # → [0.2, 0.0, 1.2]
```

If the Rust extension fails to load, re-run `maturin develop --release`
(source install) or report the wheel architecture mismatch on the issue
tracker.
