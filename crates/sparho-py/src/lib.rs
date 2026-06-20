//! Python bindings for `sparho-core`.
//!
//! Naming convention: prefixed flat namespace (no PyO3 submodules) — every
//! binding lives at `sparho._core.<name>`. The `.pyi` stub at
//! `python/sparho/_core.pyi` is the source of truth for the public surface.
//!
//! Every kernel returns `Result<(), &'static str>`; we translate any error
//! to `PyValueError` so that malformed inputs from Python never trigger a
//! Rust panic (the release profile sets `panic = "abort"`, so a stray panic
//! would terminate the interpreter — that is the safety net, not the error
//! path).

// PyO3 0.22's `#[pyfunction]` macro expands a `PyResult -> PyResult` round-trip
// that clippy 1.84+ flags as `useless_conversion`. Crate-level suppression;
// remove when bumping to pyo3 0.23+.
#![allow(clippy::useless_conversion)]

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Return type of every prox-Jacobian binding: `(d prox / d z, d prox / d α)`,
/// both diagonal-encoded as 1-D float64 arrays.
type JacobianPair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Map a kernel's `&'static str` error into a `PyValueError`.
fn map_kernel_err(e: &'static str) -> PyErr {
    PyValueError::new_err(e)
}

#[pyfunction]
fn version() -> &'static str {
    sparho_core::version()
}

// ---------------------------------------------------------------- kernels

#[pyfunction]
fn soft_threshold(x: f64, alpha: f64) -> f64 {
    sparho_core::kernels::soft_threshold(x, alpha)
}

#[pyfunction]
fn sigmoid(z: f64) -> f64 {
    sparho_core::kernels::sigmoid(z)
}

// ---------------------------------------------------------------- prox

#[pyfunction]
fn prox_l1<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    alpha: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let z_slice = z.as_slice()?;
    let n = z_slice.len();
    let out = PyArray1::<f64>::zeros_bound(py, n, false);
    {
        let mut out_rw = out.readwrite();
        sparho_core::prox::prox_l1(z_slice, alpha, out_rw.as_slice_mut()?)
            .map_err(map_kernel_err)?;
    }
    Ok(out)
}

#[pyfunction]
fn prox_jacobian_l1<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    alpha: f64,
) -> PyResult<JacobianPair<'py>> {
    let z_slice = z.as_slice()?;
    let n = z_slice.len();
    let out_z = PyArray1::<f64>::zeros_bound(py, n, false);
    let out_a = PyArray1::<f64>::zeros_bound(py, n, false);
    {
        let mut oz_rw = out_z.readwrite();
        let mut oa_rw = out_a.readwrite();
        sparho_core::prox::prox_jacobian_l1(
            z_slice,
            alpha,
            oz_rw.as_slice_mut()?,
            oa_rw.as_slice_mut()?,
        )
        .map_err(map_kernel_err)?;
    }
    Ok((out_z, out_a))
}

#[pyfunction]
fn prox_elastic_net<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    rho: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if !(0.0 < rho && rho <= 1.0) {
        return Err(PyValueError::new_err("rho must lie in (0, 1]"));
    }
    let z_slice = z.as_slice()?;
    let n = z_slice.len();
    let out = PyArray1::<f64>::zeros_bound(py, n, false);
    {
        let mut out_rw = out.readwrite();
        sparho_core::prox::prox_elastic_net(z_slice, alpha, rho, out_rw.as_slice_mut()?)
            .map_err(map_kernel_err)?;
    }
    Ok(out)
}

#[pyfunction]
fn prox_jacobian_elastic_net<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    rho: f64,
) -> PyResult<JacobianPair<'py>> {
    if !(0.0 < rho && rho <= 1.0) {
        return Err(PyValueError::new_err("rho must lie in (0, 1]"));
    }
    let z_slice = z.as_slice()?;
    let n = z_slice.len();
    let out_z = PyArray1::<f64>::zeros_bound(py, n, false);
    let out_a = PyArray1::<f64>::zeros_bound(py, n, false);
    {
        let mut oz_rw = out_z.readwrite();
        let mut oa_rw = out_a.readwrite();
        sparho_core::prox::prox_jacobian_elastic_net(
            z_slice,
            alpha,
            rho,
            oz_rw.as_slice_mut()?,
            oa_rw.as_slice_mut()?,
        )
        .map_err(map_kernel_err)?;
    }
    Ok((out_z, out_a))
}

#[pyfunction]
fn prox_weighted_l1<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    alpha: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let z_slice = z.as_slice()?;
    let a_slice = alpha.as_slice()?;
    if z_slice.len() != a_slice.len() {
        return Err(PyValueError::new_err(
            "z and alpha must have the same length",
        ));
    }
    let n = z_slice.len();
    let out = PyArray1::<f64>::zeros_bound(py, n, false);
    {
        let mut out_rw = out.readwrite();
        sparho_core::prox::prox_weighted_l1(z_slice, a_slice, out_rw.as_slice_mut()?)
            .map_err(map_kernel_err)?;
    }
    Ok(out)
}

#[pyfunction]
fn prox_jacobian_weighted_l1<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    alpha: PyReadonlyArray1<'py, f64>,
) -> PyResult<JacobianPair<'py>> {
    let z_slice = z.as_slice()?;
    let a_slice = alpha.as_slice()?;
    if z_slice.len() != a_slice.len() {
        return Err(PyValueError::new_err(
            "z and alpha must have the same length",
        ));
    }
    let n = z_slice.len();
    let out_z = PyArray1::<f64>::zeros_bound(py, n, false);
    let out_a = PyArray1::<f64>::zeros_bound(py, n, false);
    {
        let mut oz_rw = out_z.readwrite();
        let mut oa_rw = out_a.readwrite();
        sparho_core::prox::prox_jacobian_weighted_l1(
            z_slice,
            a_slice,
            oz_rw.as_slice_mut()?,
            oa_rw.as_slice_mut()?,
        )
        .map_err(map_kernel_err)?;
    }
    Ok((out_z, out_a))
}

#[pyfunction]
fn prox_group_l1<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    weights: PyReadonlyArray1<'py, f64>,
    group_ptr: PyReadonlyArray1<'py, i32>,
    group_indices: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let z_s = z.as_slice()?;
    let w_s = weights.as_slice()?;
    let gp_s = group_ptr.as_slice()?;
    let gi_s = group_indices.as_slice()?;
    let n = z_s.len();
    let out = PyArray1::<f64>::zeros_bound(py, n, false);
    {
        let mut out_rw = out.readwrite();
        sparho_core::prox::prox_group_l1(z_s, alpha, w_s, gp_s, gi_s, out_rw.as_slice_mut()?)
            .map_err(map_kernel_err)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------- csc

#[pyfunction]
fn csc_matvec<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i32>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f64>,
    n_samples: usize,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let indptr_s = indptr.as_slice()?;
    let indices_s = indices.as_slice()?;
    let data_s = data.as_slice()?;
    let x_s = x.as_slice()?;
    let out = PyArray1::<f64>::zeros_bound(py, n_samples, false);
    {
        let mut out_rw = out.readwrite();
        sparho_core::csc::matvec(
            indptr_s,
            indices_s,
            data_s,
            n_samples,
            x_s,
            out_rw.as_slice_mut()?,
        )
        .map_err(map_kernel_err)?;
    }
    Ok(out)
}

#[pyfunction]
fn csc_rmatvec<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i32>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let indptr_s = indptr.as_slice()?;
    let indices_s = indices.as_slice()?;
    let data_s = data.as_slice()?;
    let y_s = y.as_slice()?;
    if indptr_s.is_empty() {
        return Err(PyValueError::new_err("indptr must be non-empty"));
    }
    let n_features = indptr_s.len() - 1;
    let out = PyArray1::<f64>::zeros_bound(py, n_features, false);
    {
        let mut out_rw = out.readwrite();
        sparho_core::csc::rmatvec(indptr_s, indices_s, data_s, y_s, out_rw.as_slice_mut()?)
            .map_err(map_kernel_err)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------- residual

#[pyfunction]
fn restricted_ls_hessian_matvec<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i32>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f64>,
    n_samples: usize,
    active: PyReadonlyArray1<'py, i32>,
    v: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let indptr_s = indptr.as_slice()?;
    let indices_s = indices.as_slice()?;
    let data_s = data.as_slice()?;
    let active_s = active.as_slice()?;
    let v_s = v.as_slice()?;
    if active_s.len() != v_s.len() {
        return Err(PyValueError::new_err(
            "active and v must have the same length",
        ));
    }
    let out = PyArray1::<f64>::zeros_bound(py, active_s.len(), false);
    let mut scratch = vec![0.0; n_samples];
    {
        let mut out_rw = out.readwrite();
        sparho_core::residual::restricted_ls_hessian_matvec(
            indptr_s,
            indices_s,
            data_s,
            n_samples,
            active_s,
            v_s,
            out_rw.as_slice_mut()?,
            &mut scratch,
        )
        .map_err(map_kernel_err)?;
    }
    let _ = py; // kept for the lifetime; explicit usage avoids the unused warning
    Ok(out)
}

// ---------------------------------------------------------------- bcd

/// Return type of a BCD inner solve: `(beta, n_iter, dual_gap)`.
type BcdResult<'py> = (Bound<'py, PyArray1<f64>>, usize, f64);

/// Return type of a joint β+Jacobian solve: `(beta, dbeta, n_iter, dual_gap)`.
type BcdJacResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    f64,
);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bcd_lasso_dense<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    n_samples: usize,
    n_features: usize,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    beta0: PyReadonlyArray1<'py, f64>,
    lipschitz: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
    gap_freq: usize,
) -> PyResult<BcdResult<'py>> {
    let x_s = x.as_slice()?;
    let y_s = y.as_slice()?;
    let beta0_s = beta0.as_slice()?;
    let l_s = lipschitz.as_slice()?;
    if beta0_s.len() != n_features {
        return Err(PyValueError::new_err("beta0 length must equal n_features"));
    }
    let beta = PyArray1::<f64>::zeros_bound(py, n_features, false);
    let mut resid = vec![0.0; n_samples];
    let (n_iter, gap);
    {
        let mut beta_rw = beta.readwrite();
        let beta_mut = beta_rw.as_slice_mut()?;
        beta_mut.copy_from_slice(beta0_s);
        let out = sparho_core::bcd::bcd_lasso_dense(
            x_s, n_samples, n_features, y_s, alpha, beta_mut, &mut resid, l_s, max_iter, tol,
            gap_freq,
        )
        .map_err(map_kernel_err)?;
        n_iter = out.0;
        gap = out.1;
    }
    Ok((beta, n_iter, gap))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bcd_lasso_csc<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i32>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f64>,
    n_samples: usize,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    beta0: PyReadonlyArray1<'py, f64>,
    lipschitz: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
    gap_freq: usize,
) -> PyResult<BcdResult<'py>> {
    let indptr_s = indptr.as_slice()?;
    let indices_s = indices.as_slice()?;
    let data_s = data.as_slice()?;
    let y_s = y.as_slice()?;
    let beta0_s = beta0.as_slice()?;
    let l_s = lipschitz.as_slice()?;
    if indptr_s.is_empty() {
        return Err(PyValueError::new_err("indptr must be non-empty"));
    }
    let n_features = indptr_s.len() - 1;
    if beta0_s.len() != n_features {
        return Err(PyValueError::new_err("beta0 length must equal n_features"));
    }
    let beta = PyArray1::<f64>::zeros_bound(py, n_features, false);
    let mut resid = vec![0.0; n_samples];
    let (n_iter, gap);
    {
        let mut beta_rw = beta.readwrite();
        let beta_mut = beta_rw.as_slice_mut()?;
        beta_mut.copy_from_slice(beta0_s);
        let out = sparho_core::bcd::bcd_lasso_csc(
            indptr_s, indices_s, data_s, n_samples, y_s, alpha, beta_mut, &mut resid, l_s,
            max_iter, tol, gap_freq,
        )
        .map_err(map_kernel_err)?;
        n_iter = out.0;
        gap = out.1;
    }
    Ok((beta, n_iter, gap))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bcd_lasso_jac_dense<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    n_samples: usize,
    n_features: usize,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    beta0: PyReadonlyArray1<'py, f64>,
    dbeta0: PyReadonlyArray1<'py, f64>,
    lipschitz: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
    gap_freq: usize,
) -> PyResult<BcdJacResult<'py>> {
    let x_s = x.as_slice()?;
    let y_s = y.as_slice()?;
    let beta0_s = beta0.as_slice()?;
    let dbeta0_s = dbeta0.as_slice()?;
    let l_s = lipschitz.as_slice()?;
    if beta0_s.len() != n_features || dbeta0_s.len() != n_features {
        return Err(PyValueError::new_err(
            "beta0 and dbeta0 length must equal n_features",
        ));
    }
    let beta = PyArray1::<f64>::zeros_bound(py, n_features, false);
    let dbeta = PyArray1::<f64>::zeros_bound(py, n_features, false);
    let mut resid = vec![0.0; n_samples];
    let mut dresid = vec![0.0; n_samples];
    let (n_iter, gap);
    {
        let mut beta_rw = beta.readwrite();
        let mut dbeta_rw = dbeta.readwrite();
        let beta_mut = beta_rw.as_slice_mut()?;
        let dbeta_mut = dbeta_rw.as_slice_mut()?;
        beta_mut.copy_from_slice(beta0_s);
        dbeta_mut.copy_from_slice(dbeta0_s);
        let out = sparho_core::bcd::bcd_lasso_jac_dense(
            x_s,
            n_samples,
            n_features,
            y_s,
            alpha,
            beta_mut,
            dbeta_mut,
            &mut resid,
            &mut dresid,
            l_s,
            max_iter,
            tol,
            gap_freq,
        )
        .map_err(map_kernel_err)?;
        n_iter = out.0;
        gap = out.1;
    }
    Ok((beta, dbeta, n_iter, gap))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bcd_lasso_jac_csc<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i32>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f64>,
    n_samples: usize,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    beta0: PyReadonlyArray1<'py, f64>,
    dbeta0: PyReadonlyArray1<'py, f64>,
    lipschitz: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
    gap_freq: usize,
) -> PyResult<BcdJacResult<'py>> {
    let indptr_s = indptr.as_slice()?;
    let indices_s = indices.as_slice()?;
    let data_s = data.as_slice()?;
    let y_s = y.as_slice()?;
    let beta0_s = beta0.as_slice()?;
    let dbeta0_s = dbeta0.as_slice()?;
    let l_s = lipschitz.as_slice()?;
    if indptr_s.is_empty() {
        return Err(PyValueError::new_err("indptr must be non-empty"));
    }
    let n_features = indptr_s.len() - 1;
    if beta0_s.len() != n_features || dbeta0_s.len() != n_features {
        return Err(PyValueError::new_err(
            "beta0 and dbeta0 length must equal n_features",
        ));
    }
    let beta = PyArray1::<f64>::zeros_bound(py, n_features, false);
    let dbeta = PyArray1::<f64>::zeros_bound(py, n_features, false);
    let mut resid = vec![0.0; n_samples];
    let mut dresid = vec![0.0; n_samples];
    let (n_iter, gap);
    {
        let mut beta_rw = beta.readwrite();
        let mut dbeta_rw = dbeta.readwrite();
        let beta_mut = beta_rw.as_slice_mut()?;
        let dbeta_mut = dbeta_rw.as_slice_mut()?;
        beta_mut.copy_from_slice(beta0_s);
        dbeta_mut.copy_from_slice(dbeta0_s);
        let out = sparho_core::bcd::bcd_lasso_jac_csc(
            indptr_s,
            indices_s,
            data_s,
            n_samples,
            y_s,
            alpha,
            beta_mut,
            dbeta_mut,
            &mut resid,
            &mut dresid,
            l_s,
            max_iter,
            tol,
            gap_freq,
        )
        .map_err(map_kernel_err)?;
        n_iter = out.0;
        gap = out.1;
    }
    Ok((beta, dbeta, n_iter, gap))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bcd_lasso_backward_dense(
    x: PyReadonlyArray1<'_, f64>,
    n_samples: usize,
    n_features: usize,
    y: PyReadonlyArray1<'_, f64>,
    alpha: f64,
    v: PyReadonlyArray1<'_, f64>,
    lipschitz: PyReadonlyArray1<'_, f64>,
    max_iter: usize,
    tol: f64,
    gap_freq: usize,
) -> PyResult<(f64, usize)> {
    let x_s = x.as_slice()?;
    let y_s = y.as_slice()?;
    let v_s = v.as_slice()?;
    let l_s = lipschitz.as_slice()?;
    let (grad, n_iter) = sparho_core::bcd::bcd_lasso_backward_dense(
        x_s, n_samples, n_features, y_s, alpha, v_s, l_s, max_iter, tol, gap_freq,
    )
    .map_err(map_kernel_err)?;
    Ok((grad, n_iter))
}

/// Return type of a restricted normal-equation solve: `(x, n_iter)`.
type SolveResult<'py> = (Bound<'py, PyArray1<f64>>, usize);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_restricted_normal_dense<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<'py, f64>,
    n_samples: usize,
    n_active: usize,
    b: PyReadonlyArray1<'py, f64>,
    diag_shift: f64,
    x0: PyReadonlyArray1<'py, f64>,
    lipschitz: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<SolveResult<'py>> {
    let xs_s = xs.as_slice()?;
    let b_s = b.as_slice()?;
    let x0_s = x0.as_slice()?;
    let l_s = lipschitz.as_slice()?;
    if x0_s.len() != n_active {
        return Err(PyValueError::new_err("x0 length must equal n_active"));
    }
    let x = PyArray1::<f64>::zeros_bound(py, n_active, false);
    let mut q = vec![0.0; n_samples];
    let n_iter;
    {
        let mut x_rw = x.readwrite();
        let x_mut = x_rw.as_slice_mut()?;
        x_mut.copy_from_slice(x0_s);
        n_iter = sparho_core::bcd::solve_restricted_normal_dense(
            xs_s, n_samples, n_active, b_s, diag_shift, x_mut, &mut q, l_s, max_iter, tol,
        )
        .map_err(map_kernel_err)?;
    }
    Ok((x, n_iter))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_restricted_normal_csc<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i32>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f64>,
    n_samples: usize,
    active: PyReadonlyArray1<'py, i32>,
    b: PyReadonlyArray1<'py, f64>,
    diag_shift: f64,
    x0: PyReadonlyArray1<'py, f64>,
    lipschitz: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<SolveResult<'py>> {
    let indptr_s = indptr.as_slice()?;
    let indices_s = indices.as_slice()?;
    let data_s = data.as_slice()?;
    let active_s = active.as_slice()?;
    let b_s = b.as_slice()?;
    let x0_s = x0.as_slice()?;
    let l_s = lipschitz.as_slice()?;
    let n_active = active_s.len();
    if x0_s.len() != n_active {
        return Err(PyValueError::new_err("x0 length must equal active length"));
    }
    let x = PyArray1::<f64>::zeros_bound(py, n_active, false);
    let mut q = vec![0.0; n_samples];
    let n_iter;
    {
        let mut x_rw = x.readwrite();
        let x_mut = x_rw.as_slice_mut()?;
        x_mut.copy_from_slice(x0_s);
        n_iter = sparho_core::bcd::solve_restricted_normal_csc(
            indptr_s, indices_s, data_s, n_samples, active_s, b_s, diag_shift, x_mut, &mut q, l_s,
            max_iter, tol,
        )
        .map_err(map_kernel_err)?;
    }
    Ok((x, n_iter))
}

// ---------------------------------------------------------------- module

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    // kernels
    m.add_function(wrap_pyfunction!(soft_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    // prox
    m.add_function(wrap_pyfunction!(prox_l1, m)?)?;
    m.add_function(wrap_pyfunction!(prox_jacobian_l1, m)?)?;
    m.add_function(wrap_pyfunction!(prox_elastic_net, m)?)?;
    m.add_function(wrap_pyfunction!(prox_jacobian_elastic_net, m)?)?;
    m.add_function(wrap_pyfunction!(prox_weighted_l1, m)?)?;
    m.add_function(wrap_pyfunction!(prox_jacobian_weighted_l1, m)?)?;
    m.add_function(wrap_pyfunction!(prox_group_l1, m)?)?;
    // csc
    m.add_function(wrap_pyfunction!(csc_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(csc_rmatvec, m)?)?;
    // residual
    m.add_function(wrap_pyfunction!(restricted_ls_hessian_matvec, m)?)?;
    // bcd
    m.add_function(wrap_pyfunction!(bcd_lasso_dense, m)?)?;
    m.add_function(wrap_pyfunction!(bcd_lasso_csc, m)?)?;
    m.add_function(wrap_pyfunction!(bcd_lasso_jac_dense, m)?)?;
    m.add_function(wrap_pyfunction!(bcd_lasso_jac_csc, m)?)?;
    m.add_function(wrap_pyfunction!(bcd_lasso_backward_dense, m)?)?;
    m.add_function(wrap_pyfunction!(solve_restricted_normal_dense, m)?)?;
    m.add_function(wrap_pyfunction!(solve_restricted_normal_csc, m)?)?;
    Ok(())
}
