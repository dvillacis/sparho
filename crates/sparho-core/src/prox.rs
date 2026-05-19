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

use crate::kernels::soft_threshold;

// ---------------------------------------------------------------- L1

pub fn prox_l1(z: &[f64], alpha: f64, out: &mut [f64]) {
    assert_eq!(z.len(), out.len());
    for (zi, oi) in z.iter().zip(out.iter_mut()) {
        *oi = soft_threshold(*zi, alpha);
    }
}

pub fn prox_jacobian_l1(z: &[f64], alpha: f64, out_z: &mut [f64], out_alpha: &mut [f64]) {
    assert_eq!(z.len(), out_z.len());
    assert_eq!(z.len(), out_alpha.len());
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
}

// ---------------------------------------------------------------- ElasticNet

/// Prox of ``α · (ρ |β|₁ + (1−ρ)/2 ||β||²)``: shrink L1 then divide.
pub fn prox_elastic_net(z: &[f64], alpha: f64, rho: f64, out: &mut [f64]) {
    assert_eq!(z.len(), out.len());
    let denom = 1.0 + alpha * (1.0 - rho);
    let thr = alpha * rho;
    for (zi, oi) in z.iter().zip(out.iter_mut()) {
        *oi = soft_threshold(*zi, thr) / denom;
    }
}

pub fn prox_jacobian_elastic_net(
    z: &[f64],
    alpha: f64,
    rho: f64,
    out_z: &mut [f64],
    out_alpha: &mut [f64],
) {
    assert_eq!(z.len(), out_z.len());
    assert_eq!(z.len(), out_alpha.len());
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
}

// ---------------------------------------------------------------- Weighted L1

pub fn prox_weighted_l1(z: &[f64], alpha: &[f64], out: &mut [f64]) {
    assert_eq!(z.len(), alpha.len());
    assert_eq!(z.len(), out.len());
    for (i, &zi) in z.iter().enumerate() {
        out[i] = soft_threshold(zi, alpha[i]);
    }
}

/// For weighted L1 each ``αⱼ`` only affects ``βⱼ``, so the Jacobian
/// w.r.t. ``α`` is diagonal and is returned as a length-n vector.
pub fn prox_jacobian_weighted_l1(
    z: &[f64],
    alpha: &[f64],
    out_z: &mut [f64],
    out_alpha: &mut [f64],
) {
    assert_eq!(z.len(), alpha.len());
    assert_eq!(z.len(), out_z.len());
    assert_eq!(z.len(), out_alpha.len());
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prox_l1_matches_soft_threshold() {
        let z = [2.0, -2.0, 0.3, -0.3, 0.0];
        let mut out = vec![0.0; z.len()];
        prox_l1(&z, 0.5, &mut out);
        assert_eq!(out, vec![1.5, -1.5, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn prox_jacobian_l1_signs() {
        let z = [2.0, -2.0, 0.3];
        let mut wz = vec![0.0; 3];
        let mut wa = vec![0.0; 3];
        prox_jacobian_l1(&z, 0.5, &mut wz, &mut wa);
        assert_eq!(wz, vec![1.0, 1.0, 0.0]);
        assert_eq!(wa, vec![-1.0, 1.0, 0.0]);
    }

    #[test]
    fn prox_elastic_net_reduces_to_l1_when_rho_one() {
        let z = [2.0, -2.0, 0.3];
        let mut out_en = vec![0.0; 3];
        let mut out_l1 = vec![0.0; 3];
        prox_elastic_net(&z, 0.5, 1.0, &mut out_en);
        prox_l1(&z, 0.5, &mut out_l1);
        for (a, b) in out_en.iter().zip(out_l1.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn prox_weighted_l1_per_feature() {
        let z = [2.0, -2.0, 0.3];
        let alpha = [0.5, 1.5, 0.2];
        let mut out = vec![0.0; 3];
        prox_weighted_l1(&z, &alpha, &mut out);
        let expected = [1.5, -0.5, 0.1];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
