"""End-to-end checks of the Rust kernels through the PyO3 binding layer.

Each test compares the Rust output to a numpy reference within rtol=1e-12.
The kernels are deterministic and the math is exact in double precision aside
from accumulation order, so tolerances are tight."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sparho import _core

# ---------------------------------------------------------------- kernels


def test_soft_threshold_scalar_matches_numpy():
    for x in (-3.0, -0.5, 0.0, 0.5, 3.0):
        expected = np.sign(x) * max(abs(x) - 0.7, 0.0)
        assert _core.soft_threshold(x, 0.7) == pytest.approx(expected, abs=1e-15)


def test_sigmoid_scalar_overflow_safe():
    assert _core.sigmoid(0.0) == pytest.approx(0.5, abs=1e-15)
    # Saturation is fine; what matters is "no overflow / underflow exception".
    assert 0.0 < _core.sigmoid(1000.0) <= 1.0
    assert 0.0 <= _core.sigmoid(-1000.0) < 1.0
    # Mid-magnitude where stability matters.
    assert _core.sigmoid(50.0) == pytest.approx(1.0, abs=1e-15)
    assert _core.sigmoid(-50.0) == pytest.approx(0.0, abs=1e-21)


# ---------------------------------------------------------------- prox L1


def test_prox_l1_matches_numpy_soft_threshold():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(100)
    alpha = 0.4
    expected = np.sign(z) * np.maximum(np.abs(z) - alpha, 0.0)
    out = _core.prox_l1(z, alpha)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-15)


def test_prox_jacobian_l1_shapes_and_values():
    z = np.array([2.0, -2.0, 0.3, -0.3, 0.0])
    wz, wa = _core.prox_jacobian_l1(z, 0.5)
    np.testing.assert_array_equal(wz, [1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(wa, [-1.0, 1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------- prox ElasticNet


def test_prox_elastic_net_reduces_to_l1_at_rho_one():
    rng = np.random.default_rng(1)
    z = rng.standard_normal(50)
    alpha = 0.3
    enet = _core.prox_elastic_net(z, alpha, 1.0)
    l1 = _core.prox_l1(z, alpha)
    np.testing.assert_allclose(enet, l1, rtol=1e-12, atol=1e-15)


def test_prox_elastic_net_matches_formula():
    rng = np.random.default_rng(2)
    z = rng.standard_normal(50)
    alpha, rho = 0.4, 0.7
    denom = 1.0 + alpha * (1.0 - rho)
    thr = alpha * rho
    expected = np.sign(z) * np.maximum(np.abs(z) - thr, 0.0) / denom
    out = _core.prox_elastic_net(z, alpha, rho)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-15)


def test_prox_elastic_net_jacobian_finite_difference():
    """Centered FD against the analytic Jacobian for one perturbation each."""
    rng = np.random.default_rng(3)
    z = rng.standard_normal(20)
    alpha, rho = 0.3, 0.6
    wz, wa = _core.prox_jacobian_elastic_net(z, alpha, rho)
    eps = 1e-7
    # wrt α
    plus = _core.prox_elastic_net(z, alpha + eps, rho)
    minus = _core.prox_elastic_net(z, alpha - eps, rho)
    fd_a = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(fd_a, wa, atol=1e-6, rtol=1e-5)
    # wrt z, one coord at a time (only one perturbation here for compactness)
    for j in (0, 5, 12):
        zp = z.copy()
        zm = z.copy()
        zp[j] += eps
        zm[j] -= eps
        plus_j = _core.prox_elastic_net(zp, alpha, rho)[j]
        minus_j = _core.prox_elastic_net(zm, alpha, rho)[j]
        fd_zj = (plus_j - minus_j) / (2 * eps)
        # Boundary cases: skip if FD straddles the kink.
        if abs(abs(z[j]) - alpha * rho) > 1e-3:
            assert fd_zj == pytest.approx(wz[j], abs=1e-6)


def test_prox_elastic_net_rejects_bad_rho():
    z = np.zeros(3)
    with pytest.raises(ValueError):
        _core.prox_elastic_net(z, 0.1, 0.0)
    with pytest.raises(ValueError):
        _core.prox_elastic_net(z, 0.1, 1.5)


# ---------------------------------------------------------------- prox WeightedL1


def test_prox_weighted_l1_matches_per_feature_st():
    rng = np.random.default_rng(4)
    z = rng.standard_normal(50)
    alpha = rng.uniform(0.05, 0.5, size=50)
    expected = np.sign(z) * np.maximum(np.abs(z) - alpha, 0.0)
    out = _core.prox_weighted_l1(z, alpha)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-15)


def test_prox_jacobian_weighted_l1_per_feature():
    z = np.array([2.0, -2.0, 0.1])
    alpha = np.array([0.5, 0.5, 0.5])
    wz, wa = _core.prox_jacobian_weighted_l1(z, alpha)
    np.testing.assert_array_equal(wz, [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(wa, [-1.0, 1.0, 0.0])


def test_prox_weighted_l1_length_mismatch_raises():
    with pytest.raises(ValueError):
        _core.prox_weighted_l1(np.zeros(3), np.zeros(4))


# ---------------------------------------------------------------- CSC


def test_csc_matvec_matches_scipy():
    rng = np.random.default_rng(5)
    X = sp.random(80, 30, density=0.1, format="csc", random_state=rng).astype(np.float64)
    x = rng.standard_normal(30)
    expected = X @ x
    got = _core.csc_matvec(
        X.indptr.astype(np.int32),
        X.indices.astype(np.int32),
        X.data,
        X.shape[0],
        x,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-13)


def test_csc_rmatvec_matches_scipy():
    rng = np.random.default_rng(6)
    X = sp.random(80, 30, density=0.1, format="csc", random_state=rng).astype(np.float64)
    y = rng.standard_normal(80)
    expected = X.T @ y
    got = _core.csc_rmatvec(
        X.indptr.astype(np.int32),
        X.indices.astype(np.int32),
        X.data,
        y,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-13)


# ---------------------------------------------------------------- restricted Hessian


def test_restricted_ls_hessian_matvec_matches_dense():
    rng = np.random.default_rng(7)
    Xd = rng.standard_normal((50, 20)).astype(np.float64)
    # Inject some zero columns to keep it sparse-flavored.
    Xd[:, [3, 7, 11]] = 0.0
    X = sp.csc_matrix(Xd)
    active = np.array([0, 1, 4, 8, 15], dtype=np.int32)
    v = rng.standard_normal(len(active))
    expected = Xd[:, active].T @ (Xd[:, active] @ v)
    got = _core.restricted_ls_hessian_matvec(
        X.indptr.astype(np.int32),
        X.indices.astype(np.int32),
        X.data,
        X.shape[0],
        active,
        v,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_restricted_ls_hessian_matvec_length_mismatch_raises():
    X = sp.csc_matrix(np.eye(3))
    with pytest.raises(ValueError):
        _core.restricted_ls_hessian_matvec(
            X.indptr.astype(np.int32),
            X.indices.astype(np.int32),
            X.data,
            X.shape[0],
            np.array([0, 1], dtype=np.int32),
            np.zeros(3),  # wrong length
        )
