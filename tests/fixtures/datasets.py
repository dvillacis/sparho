"""Verified-hash wrappers around ``libsvmdata.fetch_libsvm``.

Downloading from libsvm's HTTPS host is not bit-stable forever — the
project has rehosted, mirrored, and even regenerated some files in the
past. For the benchmarks and reproducibility tests in this repo to mean
anything, the bytes we operate on at run-time must match the bytes the
quoted numbers were measured on.

This module wraps ``libsvmdata.fetch_libsvm`` with SHA256 verification
against a pinned manifest at ``tests/fixtures/libsvm_manifest.json``. On
a hash mismatch, the helper raises rather than silently using drifted
data.

The manifest is bootstrapped lazily: if a dataset has no entry, the
helper prints the hashes it observed and asks the contributor to either
commit them (after verifying against the upstream changelog) or
investigate why the bytes drifted. This avoids the chicken-and-egg
problem of "I need to know the hash to commit the manifest, but I need
the manifest to know the hash".

Usage::

    from tests.fixtures.datasets import fetch_verified

    X, y = fetch_verified("leukemia")
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

MANIFEST_PATH = Path(__file__).resolve().parent / "libsvm_manifest.json"


def _load_manifest() -> dict[str, dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return {}
    return dict(json.loads(MANIFEST_PATH.read_text()))


def _hash_array(arr: np.ndarray) -> str:
    """SHA256 of a numpy array's *bytes* in C-contiguous, dtype-normalised form.

    Normalising to C-contiguous + float64 (for ``data``) and int32 (for
    indices/indptr) makes the hash invariant to scipy/numpy implementation
    details that don't change the semantics.
    """
    contig = np.ascontiguousarray(arr)
    return hashlib.sha256(contig.tobytes()).hexdigest()


def _hash_csc(X: sp.csc_matrix) -> dict[str, str]:
    return {
        "data": _hash_array(X.data),
        "indices": _hash_array(X.indices),
        "indptr": _hash_array(X.indptr),
    }


def _observed_manifest_entry(name: str, X: Any, y: np.ndarray) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "shape": list(X.shape),
        "sparse": bool(sp.issparse(X)),
        "y_sha256": _hash_array(np.asarray(y)),
    }
    if sp.issparse(X):
        entry["X_csc"] = _hash_csc(sp.csc_matrix(X))
    else:
        entry["X_sha256"] = _hash_array(np.asarray(X))
    return entry


class DatasetHashMismatch(RuntimeError):
    """Raised when fetched dataset bytes don't match the pinned manifest."""


def fetch_verified(name: str) -> tuple[Any, np.ndarray]:
    """Fetch ``name`` via libsvmdata and verify against the pinned manifest.

    If the manifest has no entry for ``name``, prints the observed hashes
    for the contributor to commit (and returns the dataset). Subsequent
    runs will then enforce the pin.

    Raises
    ------
    DatasetHashMismatch
        When the observed hashes disagree with the pinned manifest.
    """
    from libsvmdata import fetch_libsvm

    X, y = fetch_libsvm(name)
    observed = _observed_manifest_entry(name, X, y)
    manifest = _load_manifest()

    pinned = manifest.get(name)
    if pinned is None:
        # Bootstrap mode: contributor hasn't pinned this dataset yet.
        # Print the observed entry so it can be committed verbatim.
        print(
            f"[fetch_verified] '{name}' has no manifest entry yet. "
            f"Observed hashes (commit to {MANIFEST_PATH.name} to pin):"
        )
        print(json.dumps({name: observed}, indent=2))
        return X, y

    if pinned.get("shape") != observed["shape"]:
        raise DatasetHashMismatch(
            f"{name}: shape drift  pinned={pinned['shape']}  observed={observed['shape']}"
        )
    if pinned.get("sparse") != observed["sparse"]:
        raise DatasetHashMismatch(
            f"{name}: sparsity drift  pinned={pinned['sparse']}  observed={observed['sparse']}"
        )
    if pinned.get("y_sha256") != observed["y_sha256"]:
        raise DatasetHashMismatch(
            f"{name}: y bytes drifted  pinned={pinned['y_sha256'][:16]}…  "
            f"observed={observed['y_sha256'][:16]}…"
        )
    if observed["sparse"]:
        for key in ("data", "indices", "indptr"):
            if pinned["X_csc"][key] != observed["X_csc"][key]:
                raise DatasetHashMismatch(
                    f"{name}: X.{key} bytes drifted  "
                    f"pinned={pinned['X_csc'][key][:16]}…  "
                    f"observed={observed['X_csc'][key][:16]}…"
                )
    else:
        if pinned.get("X_sha256") != observed["X_sha256"]:
            raise DatasetHashMismatch(
                f"{name}: X bytes drifted  "
                f"pinned={pinned.get('X_sha256', '<missing>')[:16]}…  "
                f"observed={observed['X_sha256'][:16]}…"
            )
    return X, y


def regenerate_manifest_entry(name: str) -> dict[str, Any]:
    """Compute the manifest entry for ``name`` without verifying. Returns the dict.

    Intended for use from the bootstrap workflow documented in
    ``CONTRIBUTING.md`` — never from test code.
    """
    from libsvmdata import fetch_libsvm

    X, y = fetch_libsvm(name)
    return _observed_manifest_entry(name, X, y)
