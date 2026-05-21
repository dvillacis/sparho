# Security policy

## Supported versions

Only the latest `0.x` release on PyPI receives security updates. Patches
land in a new `0.x.y` release; older versions are not backported.

## Reporting a vulnerability

**Do not open a public GitHub issue for security reports.**

Email the maintainer at **david@villacis.net** with:

- A description of the vulnerability and its impact.
- Reproduction steps (a minimal failing case is ideal — the simpler, the
  faster we can confirm and patch).
- The affected version (`pip show sparho` or `sparho.__version__`).
- Your preferred attribution in the resulting advisory, or anonymous.

Expected response: acknowledgement within 5 business days; an initial
triage assessment within 14 days. We follow a **90-day coordinated
disclosure** window — if the fix lands sooner the advisory is published
on the release date, otherwise on day 90 regardless.

## Threat model

`sparho` runs locally inside a user's Python process and consumes
in-memory numpy / scipy data structures. The realistic attack surface is:

- **FFI boundary.** Rust kernels validate every caller-supplied `i32`
  index against the matching slice length before any `as usize` cast
  (v0.3.1 hardening). Malformed `scipy.sparse.csc_matrix` input produces
  a `ValueError`, not a Rust panic — the release profile sets `panic =
  "abort"` as the safety net for any remaining bug.
- **Dependency advisories.** A `cargo-audit` job runs on every PR; the
  one current ignore (PyO3 0.22 `PyString::from_object` —
  RUSTSEC-2025-0020) is tracked on the ROADMAP behind a pyo3 → 0.24
  bump. Python deps are checked transitively via the floor-version
  `ci-min-deps` matrix job.

Out of scope (these are the user's responsibility):

- Loading untrusted pickle artifacts. `SearchResult` and the wrapper
  estimators *are* picklable, but pickle is unsafe with attacker-
  controlled input by construction.
- Adversarial training data. The library makes no robustness guarantees
  against data poisoning or membership inference.
- Side-channel attacks on the inner solver (timing, cache).

## Hall of fame

None yet. Be the first.
