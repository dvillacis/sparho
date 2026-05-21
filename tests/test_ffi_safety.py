"""Regression tests for the v0.3.1 FFI hardening.

Every malformed input that previously triggered a Rust `assert!` panic
(unwinding through CPython — undefined behavior on a `panic = "unwind"`
extension) now reaches Python as `ValueError`. We assert on the exception
type only; the kernel error strings are stability-experimental.
"""

from __future__ import annotations

import numpy as np
import pytest
from sparho import _core

# ---------------------------------------------------------------- csc_matvec


def test_csc_matvec_rejects_out_of_range_indices():
    indptr = np.array([0, 1], dtype=np.int32)
    indices = np.array([5], dtype=np.int32)  # n_samples = 3 → out of range
    data = np.array([1.0], dtype=np.float64)
    x = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        _core.csc_matvec(indptr, indices, data, 3, x)


def test_csc_matvec_rejects_negative_indices():
    indptr = np.array([0, 1], dtype=np.int32)
    indices = np.array([-1], dtype=np.int32)
    data = np.array([1.0], dtype=np.float64)
    x = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        _core.csc_matvec(indptr, indices, data, 3, x)


def test_csc_matvec_rejects_non_monotone_indptr():
    indptr = np.array([0, 2, 1], dtype=np.int32)  # not monotone
    indices = np.array([0, 1], dtype=np.int32)
    data = np.array([1.0, 1.0], dtype=np.float64)
    x = np.array([0.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError):
        _core.csc_matvec(indptr, indices, data, 3, x)


def test_csc_matvec_rejects_indptr_nonzero_first():
    indptr = np.array([1, 2], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    data = np.array([1.0], dtype=np.float64)
    x = np.array([0.0], dtype=np.float64)
    with pytest.raises(ValueError):
        _core.csc_matvec(indptr, indices, data, 3, x)


def test_csc_matvec_rejects_length_mismatch():
    indptr = np.array([0, 1], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    data = np.array([1.0, 2.0], dtype=np.float64)  # wrong length (should be 1)
    x = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        _core.csc_matvec(indptr, indices, data, 1, x)


def test_csc_matvec_rejects_x_length_mismatch():
    indptr = np.array([0, 1, 2], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    data = np.array([1.0, 1.0], dtype=np.float64)
    x = np.array([1.0], dtype=np.float64)  # should be length 2
    with pytest.raises(ValueError):
        _core.csc_matvec(indptr, indices, data, 2, x)


def test_csc_matvec_rejects_non_contiguous_slice():
    indptr = np.array([0, 1, 2], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    data = np.array([1.0, 1.0], dtype=np.float64)
    x = np.arange(4, dtype=np.float64)[::2]  # length 2, stride 16
    assert not x.flags["C_CONTIGUOUS"]
    # `PyReadonlyArray1::as_slice` requires contiguity → BufferError or
    # similar; either way it must not panic.
    with pytest.raises((ValueError, BufferError, TypeError)):
        _core.csc_matvec(indptr, indices, data, 3, x)


# ---------------------------------------------------------------- restricted_ls_hessian_matvec


def test_restricted_hess_rejects_out_of_range_active():
    indptr = np.array([0, 1, 2, 3], dtype=np.int32)  # n_features = 3
    indices = np.array([0, 0, 0], dtype=np.int32)
    data = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    active = np.array([5], dtype=np.int32)  # out of [0, 3)
    v = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        _core.restricted_ls_hessian_matvec(indptr, indices, data, 1, active, v)


def test_restricted_hess_rejects_negative_active():
    indptr = np.array([0, 1, 2, 3], dtype=np.int32)
    indices = np.array([0, 0, 0], dtype=np.int32)
    data = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    active = np.array([-1], dtype=np.int32)
    v = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        _core.restricted_ls_hessian_matvec(indptr, indices, data, 1, active, v)


def test_restricted_hess_rejects_active_v_length_mismatch():
    indptr = np.array([0, 1, 2], dtype=np.int32)
    indices = np.array([0, 0], dtype=np.int32)
    data = np.array([1.0, 1.0], dtype=np.float64)
    active = np.array([0, 1], dtype=np.int32)
    v = np.array([1.0], dtype=np.float64)  # length 1 vs. active length 2
    with pytest.raises(ValueError):
        _core.restricted_ls_hessian_matvec(indptr, indices, data, 1, active, v)


def test_restricted_hess_rejects_out_of_range_csc_indices():
    indptr = np.array([0, 1], dtype=np.int32)
    indices = np.array([99], dtype=np.int32)  # row 99 with n_samples = 3
    data = np.array([1.0], dtype=np.float64)
    active = np.array([0], dtype=np.int32)
    v = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        _core.restricted_ls_hessian_matvec(indptr, indices, data, 3, active, v)


# ---------------------------------------------------------------- prox_group_l1


def test_prox_group_l1_rejects_out_of_range_group_indices():
    z = np.array([1.0, 1.0], dtype=np.float64)
    weights = np.array([1.0], dtype=np.float64)
    group_ptr = np.array([0, 2], dtype=np.int32)
    group_indices = np.array([0, 5], dtype=np.int32)  # 5 ∉ [0, 2)
    with pytest.raises(ValueError):
        _core.prox_group_l1(z, 0.5, weights, group_ptr, group_indices)


def test_prox_group_l1_rejects_bad_partition_length():
    z = np.array([1.0], dtype=np.float64)
    weights = np.array([1.0, 1.0], dtype=np.float64)  # 2 groups, but
    group_ptr = np.array([0, 1], dtype=np.int32)  # only 1 ptr boundary
    group_indices = np.array([0], dtype=np.int32)
    with pytest.raises(ValueError):
        _core.prox_group_l1(z, 0.5, weights, group_ptr, group_indices)


def test_prox_group_l1_rejects_non_monotone_group_ptr():
    z = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    weights = np.array([1.0, 1.0], dtype=np.float64)
    group_ptr = np.array([0, 2, 1], dtype=np.int32)  # not monotone
    group_indices = np.array([0, 1, 2], dtype=np.int32)
    with pytest.raises(ValueError):
        _core.prox_group_l1(z, 0.5, weights, group_ptr, group_indices)


# ---------------------------------------------------------------- prox length mismatches


def test_prox_weighted_l1_rejects_alpha_length_mismatch():
    z = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    alpha = np.array([0.5, 0.5], dtype=np.float64)  # length 2 ≠ 3
    with pytest.raises(ValueError):
        _core.prox_weighted_l1(z, alpha)
