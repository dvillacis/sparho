//! Block-coordinate-descent inner solvers (ports of sparse-ho's `compute_beta`).
//!
//! These kernels solve the Lasso-family inner problem by cyclic coordinate
//! descent, maintaining the residual `r = y − Xβ` incrementally. The whole
//! sweep loop — including the duality-gap stopping test — lives in Rust so the
//! FFI boundary is crossed once per inner solve, not once per epoch.
//!
//! Conventions (match sparse-ho and the rest of sparho):
//! - Objective `(1/2n)‖y − Xβ‖² + α‖β‖₁` (the `1/n` matches sklearn's `1/2n`).
//! - Per-coordinate Lipschitz `L_j = ‖X_j‖² / n`, supplied by the caller. A
//!   zero `L_j` marks a zero column and the coordinate is skipped.
//! - Dense `X` is **column-major** (Fortran order): column `j` occupies
//!   `x[j·n_samples .. (j+1)·n_samples]`. Sparse `X` is CSC, like every other
//!   kernel here.
//!
//! Every entry point validates its inputs and returns `Result`; malformed input
//! from Python becomes a `PyValueError`, never a panic.

use crate::csc::validate_csc;
use crate::kernels::soft_threshold;

/// Outcome of an inner solve: number of sweeps run and the final duality gap.
pub type SolveOutcome = (usize, f64);

/// Duality gap of the Lasso primal `(1/2n)‖r‖² + α‖β‖₁` against its dual, with
/// the dual point `θ = r/(αn)` rescaled into the feasible set
/// `‖Xᵀθ‖_∞ ≤ 1`. Dense, column-major `X`.
fn lasso_dual_gap_dense(
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    y: &[f64],
    beta: &[f64],
    resid: &[f64],
    alpha: f64,
) -> f64 {
    let n = n_samples as f64;
    let r2: f64 = resid.iter().map(|&r| r * r).sum();
    let l1: f64 = beta.iter().map(|&b| b.abs()).sum();
    let pobj = 0.5 * r2 / n + alpha * l1;

    // ‖Xᵀr‖_∞.
    let mut max_abs = 0.0f64;
    for j in 0..n_features {
        let col = &x[j * n_samples..(j + 1) * n_samples];
        let s: f64 = col.iter().zip(resid.iter()).map(|(&c, &r)| c * r).sum();
        let a = s.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let dobj = lasso_dual_objective(alpha, n, r2, dot(y, resid), max_abs);
    pobj - dobj
}

/// CSC variant of [`lasso_dual_gap_dense`].
#[allow(clippy::too_many_arguments)]
fn lasso_dual_gap_csc(
    indptr: &[i32],
    indices: &[i32],
    data: &[f64],
    n_samples: usize,
    y: &[f64],
    beta: &[f64],
    resid: &[f64],
    alpha: f64,
) -> f64 {
    let n = n_samples as f64;
    let n_features = indptr.len() - 1;
    let r2: f64 = resid.iter().map(|&r| r * r).sum();
    let l1: f64 = beta.iter().map(|&b| b.abs()).sum();
    let pobj = 0.5 * r2 / n + alpha * l1;

    let mut max_abs = 0.0f64;
    for j in 0..n_features {
        let start = indptr[j] as usize;
        let end = indptr[j + 1] as usize;
        let mut s = 0.0;
        for k in start..end {
            s += data[k] * resid[indices[k] as usize];
        }
        let a = s.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let dobj = lasso_dual_objective(alpha, n, r2, dot(y, resid), max_abs);
    pobj - dobj
}

/// Shared dual-objective evaluation given precomputed scalars.
///
/// `θ = scale · r` with `scale = 1/(αn)`, rescaled by `‖Xᵀθ‖_∞ = max_abs·scale`
/// when that exceeds 1. Then `D(θ) = α·yᵀθ − (α²n/2)‖θ‖²`.
fn lasso_dual_objective(alpha: f64, n: f64, r2: f64, y_dot_r: f64, max_abs: f64) -> f64 {
    if alpha <= 0.0 {
        return 0.0;
    }
    let scale = 1.0 / (alpha * n);
    let norm_inf = max_abs * scale;
    let theta_scale = if norm_inf > 1.0 {
        scale / norm_inf
    } else {
        scale
    };
    alpha * theta_scale * y_dot_r - 0.5 * alpha * alpha * n * theta_scale * theta_scale * r2
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Validate the shared shapes of a dense BCD call. Returns `1/n_samples`.
fn check_dense_shapes(
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    y: &[f64],
    beta: &[f64],
    resid: &[f64],
    lipschitz: &[f64],
) -> Result<(), &'static str> {
    if n_samples == 0 {
        return Err("n_samples must be positive");
    }
    let expected = n_samples
        .checked_mul(n_features)
        .ok_or("n_samples * n_features overflows usize")?;
    if x.len() != expected {
        return Err("x length must equal n_samples * n_features (column-major)");
    }
    if y.len() != n_samples {
        return Err("y length must equal n_samples");
    }
    if beta.len() != n_features {
        return Err("beta length must equal n_features");
    }
    if resid.len() != n_samples {
        return Err("resid length must equal n_samples");
    }
    if lipschitz.len() != n_features {
        return Err("lipschitz length must equal n_features");
    }
    Ok(())
}

/// Solve the dense Lasso inner problem by cyclic BCD (β-only, no Jacobian).
///
/// `beta` carries the warm start in and the solution out; `resid` is
/// caller-owned scratch (length `n_samples`) reinitialized to `y − Xβ` here.
/// Returns `(n_sweeps, final_dual_gap)`.
#[allow(clippy::too_many_arguments)]
pub fn bcd_lasso_dense(
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    y: &[f64],
    alpha: f64,
    beta: &mut [f64],
    resid: &mut [f64],
    lipschitz: &[f64],
    max_iter: usize,
    tol: f64,
    gap_freq: usize,
) -> Result<SolveOutcome, &'static str> {
    check_dense_shapes(x, n_samples, n_features, y, beta, resid, lipschitz)?;
    let nf = n_samples as f64;

    // resid = y − Xβ for the (possibly warm-started) β.
    resid.copy_from_slice(y);
    for j in 0..n_features {
        let bj = beta[j];
        if bj != 0.0 {
            let col = &x[j * n_samples..(j + 1) * n_samples];
            for (r, &c) in resid.iter_mut().zip(col.iter()) {
                *r -= c * bj;
            }
        }
    }
    let pobj0 = 0.5 * dot(y, y) / nf;
    let stop = pobj0 * tol;

    let mut it = 0usize;
    while it < max_iter {
        for j in 0..n_features {
            let lj = lipschitz[j];
            if lj == 0.0 {
                continue;
            }
            let col = &x[j * n_samples..(j + 1) * n_samples];
            let beta_old = beta[j];
            let grad: f64 = col.iter().zip(resid.iter()).map(|(&c, &r)| c * r).sum();
            let zj = beta_old + grad / (lj * nf);
            let bj = soft_threshold(zj, alpha / lj);
            beta[j] = bj;
            let diff = bj - beta_old;
            if diff != 0.0 {
                for (r, &c) in resid.iter_mut().zip(col.iter()) {
                    *r -= c * diff;
                }
            }
        }
        it += 1;
        if gap_freq != 0 && (it.is_multiple_of(gap_freq) || it == 1) {
            let gap = lasso_dual_gap_dense(x, n_samples, n_features, y, beta, resid, alpha);
            if gap <= stop {
                return Ok((it, gap));
            }
        }
    }
    let gap = lasso_dual_gap_dense(x, n_samples, n_features, y, beta, resid, alpha);
    Ok((it, gap))
}

/// Sign with a true zero at the origin (unlike `f64::signum`, which returns ±1).
#[inline]
fn sgn(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Joint β + Jacobian BCD solve for Lasso (the `Forward` algorithm).
///
/// One sweep loop updates `beta` AND `dbeta = dβ/dα` together over *all*
/// features — sparse-ho's `_update_beta_jac_bcd`, forward-mode autodiff through
/// coordinate descent. The active-set gate `|sign(β_j)|` zeroes the Jacobian off
/// the support, so inactive coordinates carry `dβ_j = 0`. The Jacobian source is
/// `−sign/L` (no `α` factor), yielding `dβ/dα` directly.
///
/// `beta`/`dbeta` carry warm starts in and solutions out; `resid`/`dresid` are
/// caller-owned scratch reinitialized to `y − Xβ` and `−Xs·dβ` here. Dense `x`
/// is column-major. Returns `(n_sweeps, final_dual_gap)`. The stop is the β
/// duality gap (the Jacobian co-converges as β's support stabilizes), so this is
/// run from a cold β rather than a converged warm start.
#[allow(clippy::too_many_arguments)]
pub fn bcd_lasso_jac_dense(
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    y: &[f64],
    alpha: f64,
    beta: &mut [f64],
    dbeta: &mut [f64],
    resid: &mut [f64],
    dresid: &mut [f64],
    lipschitz: &[f64],
    max_iter: usize,
    tol: f64,
    gap_freq: usize,
) -> Result<SolveOutcome, &'static str> {
    check_dense_shapes(x, n_samples, n_features, y, beta, resid, lipschitz)?;
    if dbeta.len() != n_features {
        return Err("dbeta length must equal n_features");
    }
    if dresid.len() != n_samples {
        return Err("dresid length must equal n_samples");
    }
    let nf = n_samples as f64;

    resid.copy_from_slice(y);
    for j in 0..n_features {
        let bj = beta[j];
        if bj != 0.0 {
            let col = &x[j * n_samples..(j + 1) * n_samples];
            for (r, &c) in resid.iter_mut().zip(col.iter()) {
                *r -= c * bj;
            }
        }
    }
    for v in dresid.iter_mut() {
        *v = 0.0;
    }
    for j in 0..n_features {
        let dbj = dbeta[j];
        if dbj != 0.0 {
            let col = &x[j * n_samples..(j + 1) * n_samples];
            for (r, &c) in dresid.iter_mut().zip(col.iter()) {
                *r -= c * dbj;
            }
        }
    }
    let pobj0 = 0.5 * dot(y, y) / nf;
    let stop = pobj0 * tol;

    let mut it = 0usize;
    while it < max_iter {
        for j in 0..n_features {
            let lj = lipschitz[j];
            if lj == 0.0 {
                continue;
            }
            let col = &x[j * n_samples..(j + 1) * n_samples];
            let beta_old = beta[j];
            let dbeta_old = dbeta[j];
            let grad: f64 = col.iter().zip(resid.iter()).map(|(&c, &r)| c * r).sum();
            let bj = soft_threshold(beta_old + grad / (lj * nf), alpha / lj);
            beta[j] = bj;
            let bdiff = bj - beta_old;
            if bdiff != 0.0 {
                for (r, &c) in resid.iter_mut().zip(col.iter()) {
                    *r -= c * bdiff;
                }
            }
            // Jacobian update, gated by the support.
            let s = sgn(bj);
            let dgrad: f64 = col.iter().zip(dresid.iter()).map(|(&c, &r)| c * r).sum();
            let dbj = if bj == 0.0 {
                0.0
            } else {
                (dbeta_old + dgrad / (lj * nf)) - s / lj
            };
            dbeta[j] = dbj;
            let ddiff = dbj - dbeta_old;
            if ddiff != 0.0 {
                for (r, &c) in dresid.iter_mut().zip(col.iter()) {
                    *r -= c * ddiff;
                }
            }
        }
        it += 1;
        if gap_freq != 0 && (it.is_multiple_of(gap_freq) || it == 1) {
            let gap = lasso_dual_gap_dense(x, n_samples, n_features, y, beta, resid, alpha);
            if gap <= stop {
                return Ok((it, gap));
            }
        }
    }
    let gap = lasso_dual_gap_dense(x, n_samples, n_features, y, beta, resid, alpha);
    Ok((it, gap))
}

/// CSC variant of [`bcd_lasso_jac_dense`].
#[allow(clippy::too_many_arguments)]
pub fn bcd_lasso_jac_csc(
    indptr: &[i32],
    indices: &[i32],
    data: &[f64],
    n_samples: usize,
    y: &[f64],
    alpha: f64,
    beta: &mut [f64],
    dbeta: &mut [f64],
    resid: &mut [f64],
    dresid: &mut [f64],
    lipschitz: &[f64],
    max_iter: usize,
    tol: f64,
    gap_freq: usize,
) -> Result<SolveOutcome, &'static str> {
    validate_csc(indptr, indices, data, n_samples)?;
    if n_samples == 0 {
        return Err("n_samples must be positive");
    }
    let n_features = indptr.len() - 1;
    if y.len() != n_samples {
        return Err("y length must equal n_samples");
    }
    if beta.len() != n_features || dbeta.len() != n_features {
        return Err("beta and dbeta length must equal n_features");
    }
    if resid.len() != n_samples || dresid.len() != n_samples {
        return Err("resid and dresid length must equal n_samples");
    }
    if lipschitz.len() != n_features {
        return Err("lipschitz length must equal n_features");
    }
    let nf = n_samples as f64;

    resid.copy_from_slice(y);
    for j in 0..n_features {
        let bj = beta[j];
        if bj != 0.0 {
            let start = indptr[j] as usize;
            let end = indptr[j + 1] as usize;
            for k in start..end {
                resid[indices[k] as usize] -= data[k] * bj;
            }
        }
    }
    for v in dresid.iter_mut() {
        *v = 0.0;
    }
    for j in 0..n_features {
        let dbj = dbeta[j];
        if dbj != 0.0 {
            let start = indptr[j] as usize;
            let end = indptr[j + 1] as usize;
            for k in start..end {
                dresid[indices[k] as usize] -= data[k] * dbj;
            }
        }
    }
    let pobj0 = 0.5 * dot(y, y) / nf;
    let stop = pobj0 * tol;

    let mut it = 0usize;
    while it < max_iter {
        for j in 0..n_features {
            let lj = lipschitz[j];
            if lj == 0.0 {
                continue;
            }
            let start = indptr[j] as usize;
            let end = indptr[j + 1] as usize;
            let beta_old = beta[j];
            let dbeta_old = dbeta[j];
            let mut grad = 0.0;
            for k in start..end {
                grad += data[k] * resid[indices[k] as usize];
            }
            let bj = soft_threshold(beta_old + grad / (lj * nf), alpha / lj);
            beta[j] = bj;
            let bdiff = bj - beta_old;
            if bdiff != 0.0 {
                for k in start..end {
                    resid[indices[k] as usize] -= data[k] * bdiff;
                }
            }
            let s = sgn(bj);
            let mut dgrad = 0.0;
            for k in start..end {
                dgrad += data[k] * dresid[indices[k] as usize];
            }
            let dbj = if bj == 0.0 {
                0.0
            } else {
                (dbeta_old + dgrad / (lj * nf)) - s / lj
            };
            dbeta[j] = dbj;
            let ddiff = dbj - dbeta_old;
            if ddiff != 0.0 {
                for k in start..end {
                    dresid[indices[k] as usize] -= data[k] * ddiff;
                }
            }
        }
        it += 1;
        if gap_freq != 0 && (it.is_multiple_of(gap_freq) || it == 1) {
            let gap = lasso_dual_gap_csc(indptr, indices, data, n_samples, y, beta, resid, alpha);
            if gap <= stop {
                return Ok((it, gap));
            }
        }
    }
    let gap = lasso_dual_gap_csc(indptr, indices, data, n_samples, y, beta, resid, alpha);
    Ok((it, gap))
}

/// CSC variant of [`bcd_lasso_dense`].
#[allow(clippy::too_many_arguments)]
pub fn bcd_lasso_csc(
    indptr: &[i32],
    indices: &[i32],
    data: &[f64],
    n_samples: usize,
    y: &[f64],
    alpha: f64,
    beta: &mut [f64],
    resid: &mut [f64],
    lipschitz: &[f64],
    max_iter: usize,
    tol: f64,
    gap_freq: usize,
) -> Result<SolveOutcome, &'static str> {
    validate_csc(indptr, indices, data, n_samples)?;
    if n_samples == 0 {
        return Err("n_samples must be positive");
    }
    let n_features = indptr.len() - 1;
    if y.len() != n_samples {
        return Err("y length must equal n_samples");
    }
    if beta.len() != n_features {
        return Err("beta length must equal n_features");
    }
    if resid.len() != n_samples {
        return Err("resid length must equal n_samples");
    }
    if lipschitz.len() != n_features {
        return Err("lipschitz length must equal n_features");
    }
    let nf = n_samples as f64;

    resid.copy_from_slice(y);
    for j in 0..n_features {
        let bj = beta[j];
        if bj != 0.0 {
            let start = indptr[j] as usize;
            let end = indptr[j + 1] as usize;
            for k in start..end {
                resid[indices[k] as usize] -= data[k] * bj;
            }
        }
    }
    let pobj0 = 0.5 * dot(y, y) / nf;
    let stop = pobj0 * tol;

    let mut it = 0usize;
    while it < max_iter {
        for j in 0..n_features {
            let lj = lipschitz[j];
            if lj == 0.0 {
                continue;
            }
            let start = indptr[j] as usize;
            let end = indptr[j + 1] as usize;
            let beta_old = beta[j];
            let mut grad = 0.0;
            for k in start..end {
                grad += data[k] * resid[indices[k] as usize];
            }
            let zj = beta_old + grad / (lj * nf);
            let bj = soft_threshold(zj, alpha / lj);
            beta[j] = bj;
            let diff = bj - beta_old;
            if diff != 0.0 {
                for k in start..end {
                    resid[indices[k] as usize] -= data[k] * diff;
                }
            }
        }
        it += 1;
        if gap_freq != 0 && (it.is_multiple_of(gap_freq) || it == 1) {
            let gap = lasso_dual_gap_csc(indptr, indices, data, n_samples, y, beta, resid, alpha);
            if gap <= stop {
                return Ok((it, gap));
            }
        }
    }
    let gap = lasso_dual_gap_csc(indptr, indices, data, n_samples, y, beta, resid, alpha);
    Ok((it, gap))
}

/// Reverse-mode (backward) hypergradient for dense Lasso (the `Backward` algorithm).
///
/// Solves the inner problem while recording every β sweep, then replays the
/// recorded iterates in reverse to accumulate the VJP `vᵀ (dβ/dα) = dC/dα`,
/// where `v = ∂C/∂β` (length `n_features`). This is sparse-ho's
/// `get_grad_backward`, with the `α` factor dropped (sparho returns `dC/dα`).
///
/// Reverse-mode is the costly member of the family — it forms the Gram matrix
/// `G = XᵀX` (`p×p`) and does an `O(p²)` update per coordinate per recorded
/// sweep — and is provided for completeness / cross-checking. Returns
/// `(dC/dα, n_sweeps)`. Dense `x` is column-major.
#[allow(clippy::too_many_arguments)]
pub fn bcd_lasso_backward_dense(
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    y: &[f64],
    alpha: f64,
    v: &[f64],
    lipschitz: &[f64],
    max_iter: usize,
    tol: f64,
    gap_freq: usize,
) -> Result<(f64, usize), &'static str> {
    if v.len() != n_features {
        return Err("v length must equal n_features");
    }
    let nf = n_samples as f64;
    let mut beta = vec![0.0; n_features];
    let mut resid = vec![0.0; n_samples];

    // Forward solve, recording β after every sweep.
    check_dense_shapes(x, n_samples, n_features, y, &beta, &resid, lipschitz)?;
    resid.copy_from_slice(y);
    let pobj0 = 0.5 * dot(y, y) / nf;
    let stop = pobj0 * tol;
    let mut iterates: Vec<Vec<f64>> = Vec::new();
    let mut it = 0usize;
    while it < max_iter {
        for j in 0..n_features {
            let lj = lipschitz[j];
            if lj == 0.0 {
                continue;
            }
            let col = &x[j * n_samples..(j + 1) * n_samples];
            let beta_old = beta[j];
            let grad: f64 = col.iter().zip(resid.iter()).map(|(&c, &r)| c * r).sum();
            let bj = soft_threshold(beta_old + grad / (lj * nf), alpha / lj);
            beta[j] = bj;
            let diff = bj - beta_old;
            if diff != 0.0 {
                for (r, &c) in resid.iter_mut().zip(col.iter()) {
                    *r -= c * diff;
                }
            }
        }
        it += 1;
        iterates.push(beta.clone());
        if gap_freq != 0 && (it.is_multiple_of(gap_freq) || it == 1) {
            let gap = lasso_dual_gap_dense(x, n_samples, n_features, y, &beta, &resid, alpha);
            if gap <= stop {
                break;
            }
        }
    }

    // Gram matrix G = XᵀX (p×p), reused across the reverse replay.
    let mut g = vec![0.0; n_features * n_features];
    for j in 0..n_features {
        let cj = &x[j * n_samples..(j + 1) * n_samples];
        for i in j..n_features {
            let ci = &x[i * n_samples..(i + 1) * n_samples];
            let d: f64 = cj.iter().zip(ci.iter()).map(|(&a, &b)| a * b).sum();
            g[j * n_features + i] = d;
            g[i * n_features + j] = d;
        }
    }

    // Reverse replay over recorded iterates.
    let mut grad = 0.0;
    let mut v_t = v.to_vec();
    for beta_k in iterates.iter().rev() {
        for j in (0..n_features).rev() {
            let lj = lipschitz[j];
            if lj == 0.0 {
                continue;
            }
            let s = sgn(beta_k[j]);
            grad -= v_t[j] * s / lj;
            v_t[j] *= s.abs();
            let cste = v_t[j] / (lj * nf);
            if cste != 0.0 {
                let row = &g[j * n_features..(j + 1) * n_features];
                for (vt, &gji) in v_t.iter_mut().zip(row.iter()) {
                    *vt -= cste * gji;
                }
            }
        }
    }
    Ok((grad, it))
}

/// Projected residual `|‖q‖² + n·c·‖x‖² − n·(x·b)|`, which equals
/// `|n·xᵀ(A x − b)|` with `A = XsᵀXs/n + c·I` and `q = Xs·x`. It vanishes at the
/// solution `A x = b` and drives the relative stopping test. (At `x = 0` it is
/// also zero — harmless, since a `b ≠ 0` system moves `x` on the first sweep.)
fn normal_residual(q: &[f64], x: &[f64], b: &[f64], diag_shift: f64, n: f64) -> f64 {
    let qq: f64 = q.iter().map(|&v| v * v).sum();
    let xx: f64 = x.iter().map(|&v| v * v).sum();
    let xb: f64 = x.iter().zip(b.iter()).map(|(&xi, &bi)| xi * bi).sum();
    (qq + n * diag_shift * xx - n * xb).abs()
}

/// Solve `(Xsᵀ Xs / n + c·I) x = b` on the support by cyclic coordinate descent.
///
/// This is the single primitive behind every ImplicitForward hypergradient: the
/// implicit-differentiation linear system restricted to the active set. Each
/// penalty supplies its own `(b, c)` and contracts the result:
/// - L1: `c = 0`, `b = −sign` ⇒ `x = dβ/dα`.
/// - ElasticNet: `c = α(1−ρ)`, `b = −(ρ·sign + (1−ρ)·β_S)` ⇒ `x = dβ/dα`.
/// - WeightedL1: `c = 0`, `b = ∂C/∂β_A` ⇒ `x = M_AA⁻¹ ∂C/∂β_A` (adjoint solve).
///
/// `x` carries the warm start in and the solution out; `q` is caller-owned
/// scratch (length `n_samples`) reinitialized to `Xs·x` here. Dense `xs` is
/// column-major. Returns the number of sweeps run.
#[allow(clippy::too_many_arguments)]
pub fn solve_restricted_normal_dense(
    xs: &[f64],
    n_samples: usize,
    n_active: usize,
    b: &[f64],
    diag_shift: f64,
    x: &mut [f64],
    q: &mut [f64],
    lipschitz: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<usize, &'static str> {
    if n_samples == 0 {
        return Err("n_samples must be positive");
    }
    let expected = n_samples
        .checked_mul(n_active)
        .ok_or("n_samples * n_active overflows usize")?;
    if xs.len() != expected {
        return Err("xs length must equal n_samples * n_active (column-major)");
    }
    if b.len() != n_active {
        return Err("b length must equal n_active");
    }
    if x.len() != n_active {
        return Err("x length must equal n_active");
    }
    if q.len() != n_samples {
        return Err("q length must equal n_samples");
    }
    if lipschitz.len() != n_active {
        return Err("lipschitz length must equal n_active");
    }
    let nf = n_samples as f64;

    // q = Xs·x for the (possibly warm-started) x.
    for v in q.iter_mut() {
        *v = 0.0;
    }
    for j in 0..n_active {
        let xj = x[j];
        if xj != 0.0 {
            let col = &xs[j * n_samples..(j + 1) * n_samples];
            for (qv, &c) in q.iter_mut().zip(col.iter()) {
                *qv += c * xj;
            }
        }
    }

    let mut res_prev = f64::INFINITY;
    let mut it = 0usize;
    while it < max_iter {
        for j in 0..n_active {
            let ljj = lipschitz[j] + diag_shift;
            if ljj == 0.0 {
                continue;
            }
            let col = &xs[j * n_samples..(j + 1) * n_samples];
            let x_old = x[j];
            let qdot: f64 = col.iter().zip(q.iter()).map(|(&c, &qv)| c * qv).sum();
            // x_j ← x_j − ((A x)_j − b_j)/A_jj,  (A x)_j = qdot/n + c·x_j.
            let ax_j = qdot / nf + diag_shift * x_old;
            let x_new = x_old - (ax_j - b[j]) / ljj;
            x[j] = x_new;
            let diff = x_new - x_old;
            if diff != 0.0 {
                for (qv, &c) in q.iter_mut().zip(col.iter()) {
                    *qv += c * diff;
                }
            }
        }
        it += 1;
        let res = normal_residual(q, x, b, diag_shift, nf);
        if it >= 2 && (res < 1e-10 || (res_prev - res).abs() < res * tol) {
            break;
        }
        res_prev = res;
    }
    Ok(it)
}

/// CSC variant of [`solve_restricted_normal_dense`]. Takes the full CSC matrix
/// plus the `active` column indices (no sub-matrix materialized in Python).
#[allow(clippy::too_many_arguments)]
pub fn solve_restricted_normal_csc(
    indptr: &[i32],
    indices: &[i32],
    data: &[f64],
    n_samples: usize,
    active: &[i32],
    b: &[f64],
    diag_shift: f64,
    x: &mut [f64],
    q: &mut [f64],
    lipschitz: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<usize, &'static str> {
    validate_csc(indptr, indices, data, n_samples)?;
    if n_samples == 0 {
        return Err("n_samples must be positive");
    }
    let n_active = active.len();
    if b.len() != n_active {
        return Err("b length must equal active length");
    }
    if x.len() != n_active {
        return Err("x length must equal active length");
    }
    if q.len() != n_samples {
        return Err("q length must equal n_samples");
    }
    if lipschitz.len() != n_active {
        return Err("lipschitz length must equal active length");
    }
    let n_features_i =
        i32::try_from(indptr.len() - 1).map_err(|_| "n_features too large for i32")?;
    for &j in active.iter() {
        if j < 0 || j >= n_features_i {
            return Err("active entry out of range [0, n_features)");
        }
    }
    let nf = n_samples as f64;

    for v in q.iter_mut() {
        *v = 0.0;
    }
    for idx in 0..n_active {
        let xj = x[idx];
        if xj != 0.0 {
            let j = active[idx] as usize;
            let start = indptr[j] as usize;
            let end = indptr[j + 1] as usize;
            for k in start..end {
                q[indices[k] as usize] += data[k] * xj;
            }
        }
    }

    let mut res_prev = f64::INFINITY;
    let mut it = 0usize;
    while it < max_iter {
        for idx in 0..n_active {
            let ljj = lipschitz[idx] + diag_shift;
            if ljj == 0.0 {
                continue;
            }
            let j = active[idx] as usize;
            let start = indptr[j] as usize;
            let end = indptr[j + 1] as usize;
            let x_old = x[idx];
            let mut qdot = 0.0;
            for k in start..end {
                qdot += data[k] * q[indices[k] as usize];
            }
            let ax_j = qdot / nf + diag_shift * x_old;
            let x_new = x_old - (ax_j - b[idx]) / ljj;
            x[idx] = x_new;
            let diff = x_new - x_old;
            if diff != 0.0 {
                for k in start..end {
                    q[indices[k] as usize] += data[k] * diff;
                }
            }
        }
        it += 1;
        let res = normal_residual(q, x, b, diag_shift, nf);
        if it >= 2 && (res < 1e-10 || (res_prev - res).abs() < res * tol) {
            break;
        }
        res_prev = res;
    }
    Ok(it)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Column-major flatten of ``X = [[1, 0, 4], [0, 2, 0], [0, 3, 5]]`` (3×3).
    fn dense_fixture() -> Vec<f64> {
        // columns: [1,0,0], [0,2,3], [4,0,5]
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0, 5.0]
    }

    /// CSC of the same 3×3 ``X``.
    fn csc_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        (
            vec![0, 1, 3, 5],
            vec![0, 1, 2, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
    }

    fn lipschitz(x: &[f64], n: usize, p: usize) -> Vec<f64> {
        (0..p)
            .map(|j| {
                let col = &x[j * n..(j + 1) * n];
                col.iter().map(|&c| c * c).sum::<f64>() / n as f64
            })
            .collect()
    }

    #[test]
    fn identity_design_recovers_soft_threshold() {
        // X = I_3, objective (1/2n)||y-β||² + α||β||₁ ⇒ β_j = ST(y_j, nα).
        let n = 3;
        let x = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let y = vec![3.0, -6.0, 0.3];
        let alpha = 0.1; // threshold = nα = 0.3
        let l = lipschitz(&x, n, 3);
        let mut beta = vec![0.0; 3];
        let mut resid = vec![0.0; n];
        let (_n_iter, gap) = bcd_lasso_dense(
            &x, n, 3, &y, alpha, &mut beta, &mut resid, &l, 1000, 1e-12, 5,
        )
        .unwrap();
        assert!((beta[0] - 2.7).abs() < 1e-9, "beta0 = {}", beta[0]);
        assert!((beta[1] + 5.7).abs() < 1e-9, "beta1 = {}", beta[1]);
        assert!(beta[2].abs() < 1e-12, "beta2 = {}", beta[2]);
        assert!(gap <= 1e-9, "gap = {gap}");
    }

    #[test]
    fn dense_and_csc_agree() {
        let n = 3;
        let p = 3;
        let xd = dense_fixture();
        let (indptr, indices, data) = csc_fixture();
        let y = vec![1.5, -2.0, 3.0];
        let alpha = 0.05;
        let ld = lipschitz(&xd, n, p);

        let mut beta_d = vec![0.0; p];
        let mut resid_d = vec![0.0; n];
        bcd_lasso_dense(
            &xd,
            n,
            p,
            &y,
            alpha,
            &mut beta_d,
            &mut resid_d,
            &ld,
            5000,
            1e-12,
            10,
        )
        .unwrap();

        let mut beta_s = vec![0.0; p];
        let mut resid_s = vec![0.0; n];
        bcd_lasso_csc(
            &indptr,
            &indices,
            &data,
            n,
            &y,
            alpha,
            &mut beta_s,
            &mut resid_s,
            &ld,
            5000,
            1e-12,
            10,
        )
        .unwrap();

        for j in 0..p {
            assert!(
                (beta_d[j] - beta_s[j]).abs() < 1e-8,
                "coord {j}: dense {} vs csc {}",
                beta_d[j],
                beta_s[j]
            );
        }
    }

    #[test]
    fn warm_start_is_idempotent() {
        // Re-solving from the converged β must not move it and must report a tiny gap.
        let n = 3;
        let p = 3;
        let xd = dense_fixture();
        let y = vec![1.5, -2.0, 3.0];
        let alpha = 0.05;
        let ld = lipschitz(&xd, n, p);
        let mut beta = vec![0.0; p];
        let mut resid = vec![0.0; n];
        bcd_lasso_dense(
            &xd, n, p, &y, alpha, &mut beta, &mut resid, &ld, 5000, 1e-12, 10,
        )
        .unwrap();
        let beta_cold = beta.clone();
        let (_it, gap) = bcd_lasso_dense(
            &xd, n, p, &y, alpha, &mut beta, &mut resid, &ld, 5000, 1e-12, 10,
        )
        .unwrap();
        for j in 0..p {
            assert!((beta[j] - beta_cold[j]).abs() < 1e-9);
        }
        assert!(gap <= 1e-9, "gap = {gap}");
    }

    #[test]
    fn rejects_bad_dense_shapes() {
        let n = 3;
        let x = dense_fixture();
        let y = vec![1.0, 2.0, 3.0];
        let l = vec![1.0, 1.0, 1.0];
        let mut beta = vec![0.0; 3];
        let mut resid = vec![0.0; n];
        // wrong x length
        assert!(bcd_lasso_dense(
            &x[..8],
            n,
            3,
            &y,
            0.1,
            &mut beta,
            &mut resid,
            &l,
            10,
            1e-8,
            10
        )
        .is_err());
        // wrong beta length
        let mut beta2 = vec![0.0; 2];
        assert!(
            bcd_lasso_dense(&x, n, 3, &y, 0.1, &mut beta2, &mut resid, &l, 10, 1e-8, 10).is_err()
        );
    }

    #[test]
    fn rejects_bad_csc() {
        let (indptr, indices, data) = csc_fixture();
        let y = vec![1.0, 2.0, 3.0];
        let l = vec![1.0, 1.0, 1.0];
        let mut beta = vec![0.0; 3];
        let mut resid = vec![0.0; 3];
        // n_samples mismatch with indices range (index 2 out of [0,2))
        assert!(bcd_lasso_csc(
            &indptr, &indices, &data, 2, &y, 0.1, &mut beta, &mut resid, &l, 10, 1e-8, 10
        )
        .is_err());
    }

    #[test]
    fn solve_normal_single_coordinate() {
        // Xs = single column [3, 4] (n=2). M = ‖Xs‖²/n = 12.5.
        // L1 system: M dβ = −sign ⇒ dβ = −0.08 (b = −sign, c = 0).
        let n = 2;
        let xs = vec![3.0, 4.0];
        let l = vec![25.0 / 2.0];
        let b = vec![-1.0];
        let mut x = vec![0.0];
        let mut q = vec![0.0; n];
        let it = solve_restricted_normal_dense(&xs, n, 1, &b, 0.0, &mut x, &mut q, &l, 1000, 1e-12)
            .unwrap();
        assert!((x[0] + 0.08).abs() < 1e-12, "x = {} (it {it})", x[0]);
    }

    #[test]
    fn solve_normal_matches_direct_inverse() {
        // 4×2 support; x solves (XsᵀXs/n + cI) x = b, checked against a 2×2 inverse.
        let n = 4;
        let p = 2;
        let xs = vec![1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 1.0]; // col0, col1
        let l = vec![6.0 / 4.0, 11.0 / 4.0];
        let b = vec![-1.0, 1.0];
        let c = 0.3; // ElasticNet-style diagonal shift
        let mut x = vec![0.0; p];
        let mut q = vec![0.0; n];
        solve_restricted_normal_dense(&xs, n, p, &b, c, &mut x, &mut q, &l, 5000, 1e-13).unwrap();

        let a00 = 6.0 / 4.0 + c;
        let a01 = 3.0 / 4.0;
        let a11 = 11.0 / 4.0 + c;
        let det = a00 * a11 - a01 * a01;
        let exp0 = (a11 * b[0] - a01 * b[1]) / det;
        let exp1 = (-a01 * b[0] + a00 * b[1]) / det;
        assert!((x[0] - exp0).abs() < 1e-8, "x0 {} vs {}", x[0], exp0);
        assert!((x[1] - exp1).abs() < 1e-8, "x1 {} vs {}", x[1], exp1);
    }

    #[test]
    fn solve_normal_dense_csc_agree() {
        let n = 3;
        let xd = dense_fixture();
        let (indptr, indices, data) = csc_fixture();
        let active = vec![0i32, 1, 2];
        let l = lipschitz(&xd, n, 3);
        let b = vec![-1.0, 1.0, -1.0];
        let c = 0.2;

        let mut x_d = vec![0.0; 3];
        let mut q_d = vec![0.0; n];
        solve_restricted_normal_dense(&xd, n, 3, &b, c, &mut x_d, &mut q_d, &l, 5000, 1e-13)
            .unwrap();

        let mut x_s = vec![0.0; 3];
        let mut q_s = vec![0.0; n];
        solve_restricted_normal_csc(
            &indptr, &indices, &data, n, &active, &b, c, &mut x_s, &mut q_s, &l, 5000, 1e-13,
        )
        .unwrap();

        for j in 0..3 {
            assert!(
                (x_d[j] - x_s[j]).abs() < 1e-8,
                "coord {j}: {} vs {}",
                x_d[j],
                x_s[j]
            );
        }
    }

    #[test]
    fn joint_jacobian_matches_support_solve() {
        // Forward's joint dβ (over all features) must match the support-restricted
        // solve on the active coords, and be zero off-support.
        let n = 3;
        let p = 3;
        let xd = dense_fixture();
        let y = vec![1.5, -2.0, 3.0];
        let alpha = 0.05;
        let l = lipschitz(&xd, n, p);

        let mut beta = vec![0.0; p];
        let mut dbeta = vec![0.0; p];
        let mut resid = vec![0.0; n];
        let mut dresid = vec![0.0; n];
        bcd_lasso_jac_dense(
            &xd,
            n,
            p,
            &y,
            alpha,
            &mut beta,
            &mut dbeta,
            &mut resid,
            &mut dresid,
            &l,
            5000,
            1e-13,
            10,
        )
        .unwrap();

        // Active set + support-restricted dβ via the normal-equation solver.
        let active: Vec<i32> = (0..p as i32).filter(|&j| beta[j as usize] != 0.0).collect();
        if !active.is_empty() {
            let na = active.len();
            let sign: Vec<f64> = active.iter().map(|&j| sgn(beta[j as usize])).collect();
            // Restricted dense columns.
            let mut xs = vec![0.0; n * na];
            let mut la = vec![0.0; na];
            for (idx, &j) in active.iter().enumerate() {
                let jc = j as usize;
                xs[idx * n..(idx + 1) * n].copy_from_slice(&xd[jc * n..(jc + 1) * n]);
                la[idx] = l[jc];
            }
            let b: Vec<f64> = sign.iter().map(|&s| -s).collect();
            let mut x = vec![0.0; na];
            let mut q = vec![0.0; n];
            solve_restricted_normal_dense(&xs, n, na, &b, 0.0, &mut x, &mut q, &la, 5000, 1e-13)
                .unwrap();
            for (idx, &j) in active.iter().enumerate() {
                assert!(
                    (dbeta[j as usize] - x[idx]).abs() < 1e-7,
                    "active coord {j}: joint {} vs solve {}",
                    dbeta[j as usize],
                    x[idx]
                );
            }
        }
        // Off-support entries are exactly zero.
        for j in 0..p {
            if beta[j] == 0.0 {
                assert_eq!(dbeta[j], 0.0, "inactive coord {j} has nonzero dβ");
            }
        }
    }

    #[test]
    fn backward_matches_forward_jacobian() {
        // Backward's vᵀ(dβ/dα) must equal v·dβ from the joint forward solve.
        let n = 3;
        let p = 3;
        let xd = dense_fixture();
        let y = vec![1.5, -2.0, 3.0];
        let alpha = 0.05;
        let l = lipschitz(&xd, n, p);
        let v = vec![0.7, -1.3, 0.4];

        // Forward Jacobian.
        let mut beta = vec![0.0; p];
        let mut dbeta = vec![0.0; p];
        let mut resid = vec![0.0; n];
        let mut dresid = vec![0.0; n];
        bcd_lasso_jac_dense(
            &xd,
            n,
            p,
            &y,
            alpha,
            &mut beta,
            &mut dbeta,
            &mut resid,
            &mut dresid,
            &l,
            5000,
            1e-13,
            10,
        )
        .unwrap();
        let fwd: f64 = v.iter().zip(dbeta.iter()).map(|(&a, &b)| a * b).sum();

        let (back, _it) =
            bcd_lasso_backward_dense(&xd, n, p, &y, alpha, &v, &l, 5000, 1e-13, 10).unwrap();
        assert!(
            (back - fwd).abs() < 1e-7,
            "backward {back} vs forward {fwd}"
        );
    }

    #[test]
    fn solve_normal_rejects_bad_shapes() {
        let n = 2;
        let xs = vec![3.0, 4.0];
        let l = vec![12.5];
        let b = vec![-1.0];
        let mut x = vec![0.0; 2]; // wrong: should be 1
        let mut q = vec![0.0; n];
        assert!(
            solve_restricted_normal_dense(&xs, n, 1, &b, 0.0, &mut x, &mut q, &l, 10, 1e-8)
                .is_err()
        );
    }
}
