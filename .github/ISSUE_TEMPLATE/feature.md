---
name: Feature request
about: A new datafit / penalty / solver adapter / outer-loop variant
labels: enhancement
---

## What you want to do

<!-- One paragraph. Concrete use case, not just the mechanism. What does
     this unlock that sparho can't do today? -->

## Proposed API surface

<!-- Optional, but it helps. What would the call site look like? Which
     existing type would the new variant slot into (a `Penalty` union
     arm? a new `Criterion`? a new wrapper estimator)? -->

```python
# sketch
```

## Prior art

<!-- Has sparse-ho, skglm, celer, glmnet, or a paper covered this? Link
     where helpful — sparho is happy to be a clean reimplementation but
     prefers not to invent new math. -->

## Why not just wrap an existing solver?

<!-- sparho's adapter pattern means many "I want X" requests resolve to
     "write an X adapter, no library change needed." If that doesn't
     work for your case, explain why — e.g. the bilevel hypergradient
     needs a kernel that doesn't exist yet, or the closed `Penalty`
     union needs a new arm. -->

## Out of scope reminders

These are intentional v0.x non-goals; please confirm your request is
*not* one of them before opening:

- JAX / torch backends, GPU support.
- A `compat` shim for sparse-ho (see `docs/migration_from_sparse_ho.md`).
- ABC tower / estimator class hierarchy (`sparho` is dataclasses +
  protocols on purpose).
- `Forward` / `Backward` unrolled hypergradient modes.
