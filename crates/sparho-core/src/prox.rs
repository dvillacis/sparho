//! Proximal operators and their Jacobians (w.r.t. input and w.r.t. hyperparameter).
//!
//! Jacobians are written into caller-provided slices to avoid allocations on
//! the PyO3 hot path. The convention is `wrt_z` first, `wrt_alpha` second.
//! Both are diagonal in 1-D (separable regularizers) so they are returned as
//! length-n vectors rather than n×n matrices.
//!
//! At a kink (|z| = α exactly), the prox is set-valued; we follow sparse-ho
//! in calling such coordinates inactive (Jacobian = 0). This is a
//! measure-zero ambiguity that does not affect any well-conditioned outer
//! search.
//!
//! Every public entry point returns `Result<(), &'static str>` so length and
//! index-range violations propagate to Python as `PyValueError` rather than
//! unwinding through the FFI boundary.

use crate::kernels::soft_threshold;

// ---------------------------------------------------------------- L1

pub fn prox_l1(z: &[f64], alpha: f64, out: &mut [f64]) -> Result<(), &'static str> {
    if z.len() != out.len() {
        return Err("z and out must have the same length");
    }
    for (zi, oi) in z.iter().zip(out.iter_mut()) {
        *oi = soft_threshold(*zi, alpha);
    }
    Ok(())
}

pub fn prox_jacobian_l1(
    z: &[f64],
    alpha: f64,
    out_z: &mut [f64],
    out_alpha: &mut [f64],
) -> Result<(), &'static str> {
    if z.len() != out_z.len() || z.len() != out_alpha.len() {
        return Err("z, out_z, and out_alpha must have the same length");
    }
    for (i, &zi) in z.iter().enumerate() {
        if zi > alpha {
            out_z[i] = 1.0;
            out_alpha[i] = -1.0;
        } else if zi < -alpha {
            out_z[i] = 1.0;
            out_alpha[i] = 1.0;
        } else {
            out_z[i] = 0.0;
            out_alpha[i] = 0.0;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- ElasticNet

/// Prox of ``α · (ρ |β|₁ + (1−ρ)/2 ||β||²)``: shrink L1 then divide.
pub fn prox_elastic_net(
    z: &[f64],
    alpha: f64,
    rho: f64,
    out: &mut [f64],
) -> Result<(), &'static str> {
    if z.len() != out.len() {
        return Err("z and out must have the same length");
    }
    let denom = 1.0 + alpha * (1.0 - rho);
    let thr = alpha * rho;
    for (zi, oi) in z.iter().zip(out.iter_mut()) {
        *oi = soft_threshold(*zi, thr) / denom;
    }
    Ok(())
}

pub fn prox_jacobian_elastic_net(
    z: &[f64],
    alpha: f64,
    rho: f64,
    out_z: &mut [f64],
    out_alpha: &mut [f64],
) -> Result<(), &'static str> {
    if z.len() != out_z.len() || z.len() != out_alpha.len() {
        return Err("z, out_z, and out_alpha must have the same length");
    }
    let denom = 1.0 + alpha * (1.0 - rho);
    let denom_sq = denom * denom;
    let thr = alpha * rho;
    for (i, &zi) in z.iter().enumerate() {
        if zi.abs() > thr {
            let sign = if zi > 0.0 { 1.0 } else { -1.0 };
            out_z[i] = 1.0 / denom;
            // u(α) = (z − α·ρ·sign(z)) / (1 + α·(1−ρ));
            // du/dα = [(−ρ·sign(z))·denom − (z − α·ρ·sign(z))·(1−ρ)] / denom²
            let numerator = zi - alpha * rho * sign;
            out_alpha[i] = -rho * sign / denom - numerator * (1.0 - rho) / denom_sq;
        } else {
            out_z[i] = 0.0;
            out_alpha[i] = 0.0;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- Weighted L1

pub fn prox_weighted_l1(z: &[f64], alpha: &[f64], out: &mut [f64]) -> Result<(), &'static str> {
    if z.len() != alpha.len() || z.len() != out.len() {
        return Err("z, alpha, and out must have the same length");
    }
    for (i, &zi) in z.iter().enumerate() {
        out[i] = soft_threshold(zi, alpha[i]);
    }
    Ok(())
}

/// For weighted L1 each ``αⱼ`` only affects ``βⱼ``, so the Jacobian
/// w.r.t. ``α`` is diagonal and is returned as a length-n vector.
pub fn prox_jacobian_weighted_l1(
    z: &[f64],
    alpha: &[f64],
    out_z: &mut [f64],
    out_alpha: &mut [f64],
) -> Result<(), &'static str> {
    if z.len() != alpha.len() || z.len() != out_z.len() || z.len() != out_alpha.len() {
        return Err("z, alpha, out_z, and out_alpha must have the same length");
    }
    for (i, &zi) in z.iter().enumerate() {
        let ai = alpha[i];
        if zi > ai {
            out_z[i] = 1.0;
            out_alpha[i] = -1.0;
        } else if zi < -ai {
            out_z[i] = 1.0;
            out_alpha[i] = 1.0;
        } else {
            out_z[i] = 0.0;
            out_alpha[i] = 0.0;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- Group L1

/// Block soft-thresholding for the group-L1 penalty.
///
/// For each group ``G_k`` indexed by ``group_indices[group_ptr[k]..group_ptr[k+1]]``:
///   r_k = ||z_{G_k}||_2
///   out_{G_k} = max(0, 1 - alpha * w_k / r_k) * z_{G_k}
///
/// CSR-style storage matches scipy's ``indptr`` / ``indices`` convention: groups
/// are *required* to partition ``0..z.len()``; coordinates not covered by any
/// group are left at zero (defensive — typical usage covers every feature).
///
/// Validates `group_ptr` monotonicity and that every `group_indices` entry is
/// in `[0, z.len())` before any `as usize` cast, so a malformed partition
/// from Python becomes `PyValueError`, not a Rust panic.
pub fn prox_group_l1(
    z: &[f64],
    alpha: f64,
    weights: &[f64],
    group_ptr: &[i32],
    group_indices: &[i32],
    out: &mut [f64],
) -> Result<(), &'static str> {
    if z.len() != out.len() {
        return Err("z and out must have the same length");
    }
    if weights.len() + 1 != group_ptr.len() {
        return Err("group_ptr must have length weights.len() + 1");
    }
    if group_ptr.is_empty() {
        return Err("group_ptr must be non-empty");
    }
    if group_ptr[0] != 0 {
        return Err("group_ptr[0] must be 0");
    }
    let mut prev = 0i32;
    for &p in group_ptr.iter() {
        if p < prev {
            return Err("group_ptr must be non-decreasing");
        }
        prev = p;
    }
    let total_i = *group_ptr.last().expect("non-empty checked above");
    if total_i < 0 || (total_i as usize) != group_indices.len() {
        return Err("group_ptr.last() must equal group_indices.len()");
    }
    let n_features_i = i32::try_from(z.len()).map_err(|_| "z too large for i32")?;
    for &j in group_indices.iter() {
        if j < 0 || j >= n_features_i {
            return Err("group_indices entry out of range [0, z.len())");
        }
    }
    for o in out.iter_mut() {
        *o = 0.0;
    }
    for (k, &w_k) in weights.iter().enumerate() {
        let start = group_ptr[k] as usize;
        let end = group_ptr[k + 1] as usize;
        let group = &group_indices[start..end];
        let mut norm_sq = 0.0;
        for &j in group {
            let zj = z[j as usize];
            norm_sq += zj * zj;
        }
        let norm = norm_sq.sqrt();
        let thr = alpha * w_k;
        if norm <= thr || norm == 0.0 {
            for &j in group {
                out[j as usize] = 0.0;
            }
        } else {
            let shrink = 1.0 - thr / norm;
            for &j in group {
                out[j as usize] = shrink * z[j as usize];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prox_l1_matches_soft_threshold() {
        let z = [2.0, -2.0, 0.3, -0.3, 0.0];
        let mut out = vec![0.0; z.len()];
        prox_l1(&z, 0.5, &mut out).unwrap();
        assert_eq!(out, vec![1.5, -1.5, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn prox_jacobian_l1_signs() {
        let z = [2.0, -2.0, 0.3];
        let mut wz = vec![0.0; 3];
        let mut wa = vec![0.0; 3];
        prox_jacobian_l1(&z, 0.5, &mut wz, &mut wa).unwrap();
        assert_eq!(wz, vec![1.0, 1.0, 0.0]);
        assert_eq!(wa, vec![-1.0, 1.0, 0.0]);
    }

    #[test]
    fn prox_elastic_net_reduces_to_l1_when_rho_one() {
        let z = [2.0, -2.0, 0.3];
        let mut out_en = vec![0.0; 3];
        let mut out_l1 = vec![0.0; 3];
        prox_elastic_net(&z, 0.5, 1.0, &mut out_en).unwrap();
        prox_l1(&z, 0.5, &mut out_l1).unwrap();
        for (a, b) in out_en.iter().zip(out_l1.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn prox_weighted_l1_per_feature() {
        let z = [2.0, -2.0, 0.3];
        let alpha = [0.5, 1.5, 0.2];
        let mut out = vec![0.0; 3];
        prox_weighted_l1(&z, &alpha, &mut out).unwrap();
        let expected = [1.5, -0.5, 0.1];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn prox_group_l1_kills_below_threshold() {
        // Single group of 4 features. ‖z‖₂ = √(0.01 + 0.01 + 0.01 + 0.01) = 0.2;
        // alpha·w = 0.5·1 > 0.2 ⇒ output is zero.
        let z = [0.1, -0.1, 0.1, -0.1];
        let weights = [1.0];
        let group_ptr = [0i32, 4];
        let group_indices = [0i32, 1, 2, 3];
        let mut out = vec![1.0; 4];
        prox_group_l1(&z, 0.5, &weights, &group_ptr, &group_indices, &mut out).unwrap();
        for o in out.iter() {
            assert_eq!(*o, 0.0);
        }
    }

    #[test]
    fn prox_group_l1_block_shrinks_above_threshold() {
        // ‖z‖₂ = 2; alpha·w = 0.5 ⇒ shrink = 1 - 0.5/2 = 0.75.
        let z = [1.0, -1.0, 1.0, -1.0];
        let weights = [1.0];
        let group_ptr = [0i32, 4];
        let group_indices = [0i32, 1, 2, 3];
        let mut out = vec![0.0; 4];
        prox_group_l1(&z, 0.5, &weights, &group_ptr, &group_indices, &mut out).unwrap();
        let expected = [0.75, -0.75, 0.75, -0.75];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} ≠ {b}");
        }
    }

    #[test]
    fn prox_group_l1_singleton_groups_match_l1() {
        // Groups = {0}, {1}, {2}; weights = 1 each ⇒ same as plain L1.
        let z = [2.0, -2.0, 0.3];
        let weights = [1.0, 1.0, 1.0];
        let group_ptr = [0i32, 1, 2, 3];
        let group_indices = [0i32, 1, 2];
        let mut out_grp = vec![0.0; 3];
        prox_group_l1(&z, 0.5, &weights, &group_ptr, &group_indices, &mut out_grp).unwrap();
        let mut out_l1 = vec![0.0; 3];
        prox_l1(&z, 0.5, &mut out_l1).unwrap();
        for (a, b) in out_grp.iter().zip(out_l1.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn prox_group_l1_two_groups_with_size_weights() {
        // Group 0 = {0, 1} (size-2, w = √2); group 1 = {2, 3, 4} (size-3, w = √3).
        // ‖z₀‖ = 1, alpha·w₀ = 0.5·√2 ≈ 0.7071 < 1 ⇒ shrink₀ = 1 − √2/2.
        // ‖z₁‖ = 2√3 ≈ 3.464, alpha·w₁ = 0.5·√3 ≈ 0.866 ⇒ shrink₁ = 1 − 0.5·√3/(2·√3) = 0.75.
        let z = [0.6, 0.8, 2.0, -2.0, 2.0];
        let sqrt2 = 2.0_f64.sqrt();
        let sqrt3 = 3.0_f64.sqrt();
        let weights = [sqrt2, sqrt3];
        let group_ptr = [0i32, 2, 5];
        let group_indices = [0i32, 1, 2, 3, 4];
        let mut out = vec![0.0; 5];
        prox_group_l1(&z, 0.5, &weights, &group_ptr, &group_indices, &mut out).unwrap();
        let s0 = 1.0 - 0.5 * sqrt2 / 1.0;
        let expected = [s0 * 0.6, s0 * 0.8, 1.5, -1.5, 1.5];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} ≠ {b}");
        }
    }

    #[test]
    fn prox_group_l1_rejects_out_of_range_index() {
        let z = [1.0, 1.0];
        let weights = [1.0];
        let group_ptr = [0i32, 2];
        let group_indices = [0i32, 5]; // n_features = 2 → out of range
        let mut out = vec![0.0; 2];
        assert!(prox_group_l1(&z, 0.5, &weights, &group_ptr, &group_indices, &mut out).is_err());
    }

    #[test]
    fn prox_group_l1_rejects_bad_partition_length() {
        let z = [1.0];
        let weights = [1.0, 1.0]; // implies 2 groups, but group_ptr has 2 entries
        let group_ptr = [0i32, 1];
        let group_indices = [0i32];
        let mut out = vec![0.0; 1];
        assert!(prox_group_l1(&z, 0.5, &weights, &group_ptr, &group_indices, &mut out).is_err());
    }
}
