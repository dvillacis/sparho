# Contributing to sparho

Thanks for considering a contribution. `sparho` is a small project; the
fastest path to merge is a self-contained PR that lands inside one of the
ROADMAP items.

## Dev setup

Requirements: Python ≥ 3.11, a stable Rust toolchain (1.75+), and
[`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/dvillacis/sparho.git
cd sparho

uv sync --extra dev                  # creates .venv, installs dev tools
uv run maturin develop --release     # builds the Rust extension in place
uv run pre-commit install            # installs the per-commit hook chain
```

`maturin develop` is iterative — re-run it after editing Rust code. Python
changes are picked up automatically by the editable install.

## Gates

Every PR has to pass the same six checks CI runs. Run them locally before
pushing — `pre-commit run --all-files` covers the linters and rustfmt.

```bash
uv run pytest                                            # Python suite
uv run ruff check python/ tests/                         # lint
uv run mypy                                              # strict typing
uv run pytest --doctest-modules python/sparho            # doctests
cargo test --workspace --all-targets                     # Rust suite (incl. proptests)
cargo clippy --workspace --all-targets -- -D warnings    # Rust lint
```

Plus on the workflow side: `pre-commit run --all-files` (Python +
Rust hooks), and `cargo audit` (Linux job, with a ROADMAP-tracked ignore
for the PyO3 0.22 advisory).

## Architecture pillars

These are described in `CLAUDE.md` and are non-negotiable in PRs:

1. **One core type.** `Problem(datafit, penalty, design, target)` plus
   free functions. No ABC tower, no estimator class hierarchy.
2. **Functional core, imperative shell.** State is an immutable dataclass;
   algorithms are pure functions; the outer loop is a plain `for`.
3. **Closed unions.** `Datafit` and `Penalty` are tagged unions — every
   algorithm `match`es with `assert_never` on the default branch so mypy
   strict catches missed dispatch.
4. **Sparse-X first-class.** CSC iterated directly in Rust; no
   densification on the hot path.
5. **Rust kernels via PyO3 + maturin + ABI3.** No numba, no pure-Python
   fallback.

## Adding a new datafit or penalty

`python/sparho/problem.py` declares the closed unions. To extend either:

1. Add a `@dataclass(frozen=True, slots=True)` variant in `problem.py`
   and export it from `__init__.py`.
2. Implement the kernel (prox, Jacobian, residual term) in
   `crates/sparho-core` and expose it via `crates/sparho-py/src/lib.rs`.
   Update `python/sparho/_core.pyi` with the signature.
3. Add a `case` arm in every `match` over the union — at minimum
   `hypergrad.implicit_forward`, plus any criterion/adapter that
   dispatches. Leave the `case _: assert_never(x)` tail intact.

## API stability

Read `docs/stability.md` before changing a public surface. Anything in
the frozen-stable list needs a deprecation cycle: introduce the
replacement and emit `DeprecationWarning`; remove the old surface in the
next minor (not patch) bump. Experimental surfaces (the
`IterationRecord.extras` schema, adapter internals) can change in any
release.

## Commit and PR norms

- One logical change per PR. Refactors that touch many files are easier
  to review if the refactor is a separate commit from the feature.
- Reference the ROADMAP item or issue in the PR description (e.g.
  "closes ROADMAP v0.4 §3").
- CHANGELOG.md `[Unreleased]` entry required for user-visible changes.
- We don't squash on merge — write commits like they'll be read in
  `git log`.

## Updating the libsvm dataset manifest

`tests/fixtures/datasets.py` verifies fetched libsvm datasets against
SHA256 hashes pinned in `tests/fixtures/libsvm_manifest.json`. Drift in
those bytes raises `DatasetHashMismatch` rather than silently consuming
different data.

To **add a new dataset** to the manifest:

```bash
uv run python -c \
  'from tests.fixtures.datasets import regenerate_manifest_entry; \
   import json; print(json.dumps(regenerate_manifest_entry("<name>"), indent=2))'
```

Paste the printed entry into `libsvm_manifest.json` under its dataset
key. Commit alongside whatever consumer change motivated the addition.

To **update an existing dataset** (legitimate upstream rehost):

1. Run the regenerate command above and capture the new hashes.
2. Verify the change is real (libsvm's GitHub issue tracker, upstream
   changelog, or by re-downloading on a second machine and confirming
   the new hashes match).
3. Update the manifest entry and document the rationale in the commit
   message — "libsvmdata v0.6 switched from CC-BY hosting to Zenodo"
   etc.

Never update a hash to make a failing test pass without step 2 —
silent byte drift is the failure mode this guard exists to catch.

## Updating the golden numerical fixtures

`tests/golden/*.json` pins `(β*, training_loss, KKT residual, active
set)` for canonical inner-solver runs (see `tests/golden/generate.py`).
If an intentional algorithmic change shifts these numbers, regenerate
the JSONs and commit them as a discrete numerical-behaviour change:

```bash
uv run python tests/golden/generate.py
```

Same discipline as the dataset manifest: never regenerate to make a
failing test pass without confirming the change is intentional and
correct.

## Security

See `SECURITY.md` for the vulnerability disclosure path.
