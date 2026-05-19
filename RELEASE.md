# Releasing sparho v0.1.0

Local prep is done (version bumped, gates green, sdist + wheel built,
fresh-venv install verified). The remaining steps need actions on
GitHub and PyPI that have to be driven by hand.

## Pre-release state (already in this repo)

- `pyproject.toml`: `version = "0.1.0"`.
- `Cargo.toml`: `[workspace.package] version = "0.1.0"`.
- `CHANGELOG.md`: `[0.1.0]` section dated `2026-05-19`.
- `.github/workflows/release.yml`: cibuildwheel + Trusted Publishers,
  tag-triggered PyPI publish, manual-dispatch TestPyPI publish.
- Local artifacts in `/tmp/sparho-release-dist/`:
  `sparho-0.1.0-cp311-abi3-macosx_11_0_arm64.whl` (260 KB),
  `sparho-0.1.0.tar.gz` (30 KB). Fresh-venv install + smoke test passed.

## External steps the user has to drive

### 1. First git commit + GitHub remote

The repo has zero commits. Pick a starting line (single big "v0.1.0
foundation" commit, or stage-by-stage history). Then:

```bash
gh repo create dvillacis/sparho --public --source . --remote origin
git push -u origin main
```

### 2. Configure Trusted Publishers on PyPI / TestPyPI

Without these the publish jobs fail with an OIDC mismatch.

PyPI: <https://pypi.org/manage/account/publishing/> → **Add a new
pending publisher** → fill in:

| Field | Value |
|---|---|
| PyPI project name | `sparho` |
| Owner | `dvillacis` |
| Repository name | `sparho` |
| Workflow filename | `release.yml` |
| Environment name | `pypi` |

Repeat at <https://test.pypi.org/manage/account/publishing/> with
environment name `testpypi`.

### 3. Create the GitHub environments

GitHub → **Settings → Environments** → create `pypi` and `testpypi`.
For `pypi` add a "Required reviewers" rule if you want a manual approval
gate before the publish step runs.

### 4. Smoke-test via TestPyPI

```bash
# On main, no tag — just trigger the workflow.
gh workflow run release.yml -f target=testpypi
gh run watch
```

When green:

```bash
python3.12 -m venv /tmp/sparho-testpypi
source /tmp/sparho-testpypi/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            sparho==0.1.0
python -c "
import sparho, numpy as np
from sklearn.datasets import make_regression
from sparho import HeldOutMSE, L1, Problem, SquaredLoss, hoag_search
from sparho.adapters import SklearnLasso
X, y = make_regression(200, 80, noise=1.0, random_state=0)
idx_train = np.arange(150, dtype=np.int32)
idx_val = np.arange(150, 200, dtype=np.int32)
r = hoag_search(Problem(SquaredLoss(), L1(), X, y), hp0=1e-2,
                solver=SklearnLasso(tol=1e-8),
                criterion=HeldOutMSE(idx_train, idx_val), n_iter=20)
print('OK', sparho.__version__, float(r.best_hyperparam))
"
```

If anything fails (wheel build error, install error, smoke-test error),
fix in a follow-up commit and re-dispatch `release.yml`.

### 5. Promote to PyPI

Once TestPyPI looks good:

```bash
git tag -a v0.1.0 -m "v0.1.0 — first public release"
git push origin v0.1.0
```

This triggers `release.yml`, which runs the wheel matrix and publishes
to PyPI on green.

### 6. Verify the PyPI install

```bash
python3.12 -m venv /tmp/sparho-pypi
source /tmp/sparho-pypi/bin/activate
pip install sparho==0.1.0
python -c "import sparho; print(sparho.__version__)"
```

Then run the held-out Lasso gallery example end-to-end (the file is at
`docs/examples/plot_held_out_lasso.py`).

### 7. Post-release

```bash
gh release create v0.1.0 --title "sparho v0.1.0" --notes-file CHANGELOG.md
```

Edit the release notes to point at the `[0.1.0]` block.

Then bump `pyproject.toml` to `0.2.0.dev0` on a `bump-dev` commit and
add an empty `[Unreleased]` section back to `CHANGELOG.md`.

## What's not in scope at v0.1.0

- aarch64 Linux wheels (build from sdist).
- macOS Intel wheels (build from sdist).
- Conda-forge feedstock (revisit once PyPI install stabilizes).
- A perf headline beating sparse-ho/`LassoCV` across the board — that's
  the v0.2 milestone tracked in `ROADMAP.md`.
