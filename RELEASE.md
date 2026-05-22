# Releasing sparho

This file is part procedure (the per-release one-time-each section at the
top) and part audit trail (the per-version sections below — they document
the actual steps taken for each release). When cutting a new release, copy
the most recent per-version section as a template.

## Zenodo DOI integration (applies to every tagged release)

`sparho` is registered with [Zenodo](https://zenodo.org/) via the
GitHub–Zenodo integration. Every `vX.Y.Z` tag pushed to the GitHub
repository auto-mints a version-specific DOI; the *concept* DOI
(`10.5281/zenodo.<concept>`) is stable and resolves to whichever version
is latest. The concept DOI is the one cited in academic work that refers
to the library as a whole.

### One-time setup (do once, then never again)

1. Sign in at <https://zenodo.org/> with GitHub OAuth (the same GitHub
   account that owns the `dvillacis/sparho` repository).
2. <https://zenodo.org/account/settings/github/> → flip the
   `dvillacis/sparho` toggle to **On**.
3. Confirm `CITATION.cff` and `.zenodo.json` are committed at repo root
   — Zenodo reads both at tag time. `.zenodo.json` is authoritative for
   the Zenodo record; `CITATION.cff` is what GitHub renders in the
   "Cite this repository" sidebar widget.
4. Sanity check with a *pre-release* tag (`v0.5.0-rc1`): tag, push, watch
   for a Zenodo sandbox record at <https://sandbox.zenodo.org/>, confirm
   metadata renders, then delete the rc tag locally and remotely.

### Per-release tasks (every tag)

Once setup is complete the Zenodo work per release is small:

1. Before tagging, bump the `version:` placeholder in the README's
   `@software` BibTeX entry to the new version.
2. Tag with `git tag -a vX.Y.Z -m "..."` and push — the Zenodo webhook
   fires automatically; nothing else to do.
3. After Zenodo mints the version DOI (usually within a minute), copy
   the concept DOI back into:
   - `README.md` — the `[![DOI](...)](...)` badge placeholder
     (`10.5281/zenodo.XXXXXXX`).
   - `CITATION.cff` — uncomment the `identifiers:` block.
   - The GitHub release notes, prefixed with "Cite this release: …".
   These three placeholders only need filling in **once** (at the first
   tagged release — `v0.5.0`); subsequent releases pick up the same
   concept DOI automatically.
4. Verify the citation widget: open the GitHub repo page logged out;
   the right sidebar should show "Cite this repository" rendering
   `CITATION.cff` and offering BibTeX / APA / etc.

### Validation

- `pip install cffconvert && cffconvert --validate -i CITATION.cff`
  — runs in CI as a soft job; you can also run locally before tagging.
- `python -m json.tool .zenodo.json` for JSON syntax.

---

# Releasing sparho v0.2.0

Local prep is done (version bumped, gates green, sdist + wheel built,
fresh-venv install verified). The Trusted Publisher and GitHub `pypi`
environment configured for v0.1.0 carry over — the remaining external
steps are just tag → CI → publish → verify.

## Pre-release state (already in this repo)

- `pyproject.toml`: `version = "0.2.0"`.
- `Cargo.toml`: `[workspace.package] version = "0.2.0"`.
- `CHANGELOG.md`: `[0.2.0]` section dated `2026-05-20`; fresh empty
  `[Unreleased]` above it.
- `ROADMAP.md`: v0.2 items 1–5 ✅; v0.3 section added (skein, SURE,
  sklearn wrappers, MultiTask/Group-L1, SkglmAdapter).
- `.github/workflows/release.yml`: unchanged from v0.1.0 — cibuildwheel
  + Trusted Publishers, tag-triggered PyPI publish, manual
  `workflow_dispatch` for a dry-run wheel build (no publish).
- Local artifacts in `/tmp/sparho-release-dist/`:
  `sparho-0.2.0-cp311-abi3-macosx_11_0_arm64.whl` (232 KB),
  `sparho-0.2.0.tar.gz` (32 KB). Fresh-venv install + end-to-end
  `hoag_search` smoke test passed.
- Gates green: 94 pytest, 11 cargo, mypy strict, ruff, clippy
  (`-D warnings`), `sphinx-build -W`.

## What v0.2.0 ships (user-visible delta since 0.1.0)

The HOAG outer loop, inner-solver warm-start, dense-matvec fix, and
`CelerLasso` adapter landed under [0.1.0] in the CHANGELOG (they were
in the 2026-05-19 wheel). What's new at the 0.2.0 tag:

- `--solver {sklearn,celer}` flag in `benchmarks/lasso_libsvm.py`.
- `CelerLasso` and `CelerElasticNet` re-exported from
  `sparho.adapters`.
- Reproducibility mode: `--repeat N`, `--warmup K`, `--cooldown S` with
  median + spread reporting, interleaved sparho/`LassoCV` per iter,
  `gc.collect()` between iters.
- Refreshed v0.2 benchmark numbers across all three libsvm Lasso
  datasets with HOAG + warm-start + `CelerLasso`. Headlines:
  `leukemia` **32.8× vs `LassoCV`** (up from 8.6× with sklearn,
  1.3× at v0.1.0); `rcv1.binary` wall halved (433 s → 211 s) at the
  same MSE win (0.194 vs 0.225).
- Top-level `README.md`, `benchmarks/README.md`, `ROADMAP.md`,
  `docs/feature_research.md` (new) all updated.

## External steps the user has to drive

### 1. Commit the v0.2.0 prep

```bash
git add pyproject.toml Cargo.toml CHANGELOG.md ROADMAP.md README.md \
        RELEASE.md benchmarks/ docs/feature_research.md \
        python/sparho/adapters/__init__.py
git commit -m "v0.2.0: bench refresh + repro harness + celer re-exports"
git push origin main
```

(Pick the file list that matches what's actually in your worktree —
`git status` is authoritative.)

### 2. Dry-run the wheel matrix on CI

Same as v0.1.0 — workflow_dispatch builds wheels on Linux + Windows
without publishing. Catches cibuildwheel / Rust-toolchain regressions
that the local macOS build doesn't exercise.

```bash
gh workflow run release.yml
gh run watch
```

If anything fails, fix in a follow-up commit and re-dispatch.

### 3. Tag + publish to PyPI

Once the dry-run is green:

```bash
git tag -a v0.2.0 -m "v0.2.0 — HOAG perf story + reproducibility harness"
git push origin v0.2.0
```

This re-runs `release.yml`; on green, the `publish` job uploads to
PyPI via the Trusted Publisher OIDC configured at v0.1.0 (no token
required).

### 4. Verify the PyPI install

```bash
python3.12 -m venv /tmp/sparho-pypi
source /tmp/sparho-pypi/bin/activate
pip install sparho==0.2.0
python -c "import sparho; print(sparho.__version__)"
python -c "from sparho.adapters import CelerLasso; print('celer re-export OK')"
```

Then run the held-out Lasso gallery example end-to-end:
`docs/examples/plot_held_out_lasso.py`.

### 5. GitHub release

```bash
gh release create v0.2.0 --title "sparho v0.2.0" --notes-file CHANGELOG.md
```

Edit the rendered notes to focus on the `[0.2.0]` block.

### 6. Post-release

Bump `pyproject.toml` to `0.3.0.dev0` on a `bump-dev` commit (the
`[Unreleased]` CHANGELOG section was already re-added during prep).

## What's not in scope at v0.2.0

- aarch64 Linux wheels (build from sdist).
- macOS Intel wheels (build from sdist).
- Conda-forge feedstock — defer until PyPI install pattern stabilizes
  across the v0.1 → v0.2 transition.
- `skein` adapter, SURE criterion, `LassoHO` sklearn wrappers,
  `MultiTaskLasso` / Group-L1, `SkglmAdapter` — all v0.3, see
  `ROADMAP.md` and `docs/feature_research.md`.
