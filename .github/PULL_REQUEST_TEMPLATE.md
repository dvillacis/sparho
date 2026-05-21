<!--
Thanks for the PR! A few quick reminders from CONTRIBUTING.md:
- One logical change per PR.
- CHANGELOG.md [Unreleased] entry for user-visible changes.
- Reference the ROADMAP item or issue (e.g. "closes ROADMAP v0.4 §3").
- Tests for new behavior; regression test for any bug fix.
-->

## Summary

<!-- One paragraph. What changed, and why. -->

## ROADMAP / issue

Closes <!-- issue # / ROADMAP §x -->

## Test plan

- [ ] `uv run pytest`
- [ ] `uv run ruff check python/ tests/`
- [ ] `uv run mypy`
- [ ] `cargo test --workspace --all-targets`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `uv run pre-commit run --all-files`

## API surface

<!-- Tick the row that applies. If you're touching a frozen-stable surface,
     describe the deprecation cycle in the summary. -->

- [ ] No public API change.
- [ ] Experimental surface only (extras schema, adapter internals, etc.).
- [ ] Frozen-stable surface change — deprecation cycle planned.

## CHANGELOG

- [ ] `[Unreleased]` entry added.
- [ ] N/A (pure internal change).
