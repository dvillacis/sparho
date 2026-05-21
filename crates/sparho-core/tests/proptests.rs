//! Property-based invariants for the public kernel surface.
//!
//! These tests complement the small fixture-based unit tests in each module
//! by hitting each kernel with ~256 randomly-generated cases per invariant.
//! Their job is to flush out the integer-cast + range-check + length-mismatch
//! classes of bug that the v0.3.1 FFI hardening targeted, plus algebraic
//! identities that small fixtures may not exercise (e.g. `ElasticNet@rho=1 ≡
//! prox_l1`).
//!
//! Cases are generated as `f64` slices and `i32` index buffers; ranges are
//! restricted to the well-conditioned regime (`α ≥ 0`, `|z| ≤ 1e6`,
//! `n ≤ 32`) so the assertions remain exact under float arithmetic.

use proptest::prelude::*;
use sparho_core::csc;
use sparho_core::kernels::soft_threshold;
use sparho_core::prox;
use sparho_core::residual;

// ---------------------------------------------------------------- generators

fn vec_f64(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-1e3_f64..1e3, min_len..=max_len)
}

fn small_alpha() -> impl Strategy<Value = f64> {
    0.0_f64..10.0
}

/// Generate a well-formed CSC structure `(indptr, indices, data, n_samples,
/// n_features)`. Densities are intentionally moderate (≤ 50 %) so the
/// fixtures exercise the iterate-active-columns path without degenerating
/// into "all-dense" or "all-empty".
fn csc_structure() -> impl Strategy<
    Value = (
        Vec<i32>,
        Vec<i32>,
        Vec<f64>,
        usize, // n_samples
        usize, // n_features
    ),
> {
    (2usize..8, 2usize..8).prop_flat_map(|(n_samples, n_features)| {
        // For each column, pick a subset of row indices (each ∈ [0, n_samples)).
        let cols = prop::collection::vec(
            prop::collection::vec(0i32..(n_samples as i32), 0..n_samples),
            n_features..=n_features,
        );
        cols.prop_map(move |mut cols| {
            // Sort + dedupe each column to mirror scipy's canonical CSC.
            for col in cols.iter_mut() {
                col.sort_unstable();
                col.dedup();
            }
            let mut indptr = Vec::with_capacity(n_features + 1);
            let mut indices = Vec::new();
            indptr.push(0i32);
            for col in &cols {
                indices.extend_from_slice(col);
                indptr.push(indices.len() as i32);
            }
            let data: Vec<f64> = (0..indices.len()).map(|k| (k as f64 + 1.0) * 0.5).collect();
            (indptr, indices, data, n_samples, n_features)
        })
    })
}

/// Dense reference: materialize `(n_samples, n_features)` matrix from CSC.
fn csc_to_dense(
    indptr: &[i32],
    indices: &[i32],
    data: &[f64],
    n_samples: usize,
    n_features: usize,
) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n_features]; n_samples];
    for j in 0..n_features {
        for k in indptr[j] as usize..indptr[j + 1] as usize {
            m[indices[k] as usize][j] = data[k];
        }
    }
    m
}

fn dense_matvec(m: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

fn dense_rmatvec(m: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n_features = m.first().map(|r| r.len()).unwrap_or(0);
    (0..n_features)
        .map(|j| m.iter().zip(y.iter()).map(|(row, yi)| row[j] * yi).sum())
        .collect()
}

// ---------------------------------------------------------------- prox_l1

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// `prox_l1` ≡ elementwise `soft_threshold`.
    #[test]
    fn prox_l1_matches_soft_threshold(z in vec_f64(0, 32), alpha in small_alpha()) {
        let mut out = vec![0.0; z.len()];
        prox::prox_l1(&z, alpha, &mut out).unwrap();
        for (i, &zi) in z.iter().enumerate() {
            prop_assert_eq!(out[i], soft_threshold(zi, alpha));
        }
    }

    /// Sign preservation + non-expansion: `sign(out) ∈ {0, sign(z)}` and
    /// `|out| ≤ |z|` elementwise.
    #[test]
    fn prox_l1_sign_and_shrinkage(z in vec_f64(0, 32), alpha in small_alpha()) {
        let mut out = vec![0.0; z.len()];
        prox::prox_l1(&z, alpha, &mut out).unwrap();
        for (zi, oi) in z.iter().zip(out.iter()) {
            prop_assert!(oi.abs() <= zi.abs() + 1e-12);
            if *oi != 0.0 {
                prop_assert_eq!(oi.signum(), zi.signum());
            }
        }
    }

    /// Below-threshold inputs are zeroed.
    #[test]
    fn prox_l1_zeroes_below_threshold(z in vec_f64(0, 32), alpha in small_alpha()) {
        let mut out = vec![0.0; z.len()];
        prox::prox_l1(&z, alpha, &mut out).unwrap();
        for (zi, oi) in z.iter().zip(out.iter()) {
            if zi.abs() <= alpha {
                prop_assert_eq!(*oi, 0.0);
            }
        }
    }
}

// ---------------------------------------------------------------- prox_elastic_net

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// `ElasticNet` at `rho == 1` reduces to plain L1 prox.
    #[test]
    fn prox_elastic_net_rho_one_matches_l1(z in vec_f64(0, 32), alpha in small_alpha()) {
        let mut out_en = vec![0.0; z.len()];
        let mut out_l1 = vec![0.0; z.len()];
        prox::prox_elastic_net(&z, alpha, 1.0, &mut out_en).unwrap();
        prox::prox_l1(&z, alpha, &mut out_l1).unwrap();
        for (a, b) in out_en.iter().zip(out_l1.iter()) {
            prop_assert!((a - b).abs() < 1e-12);
        }
    }

    /// Sign preservation + non-expansion (the L2 component only shrinks more).
    #[test]
    fn prox_elastic_net_sign_and_shrinkage(
        z in vec_f64(0, 32),
        alpha in small_alpha(),
        rho in 0.01_f64..1.0,
    ) {
        let mut out = vec![0.0; z.len()];
        prox::prox_elastic_net(&z, alpha, rho, &mut out).unwrap();
        for (zi, oi) in z.iter().zip(out.iter()) {
            prop_assert!(oi.abs() <= zi.abs() + 1e-12);
            if *oi != 0.0 {
                prop_assert_eq!(oi.signum(), zi.signum());
            }
        }
    }
}

// ---------------------------------------------------------------- prox_weighted_l1

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// `prox_weighted_l1(z, c·𝟏) ≡ prox_l1(z, c)`.
    #[test]
    fn prox_weighted_l1_uniform_matches_scalar(z in vec_f64(1, 32), c in small_alpha()) {
        let alpha_vec = vec![c; z.len()];
        let mut out_w = vec![0.0; z.len()];
        let mut out_s = vec![0.0; z.len()];
        prox::prox_weighted_l1(&z, &alpha_vec, &mut out_w).unwrap();
        prox::prox_l1(&z, c, &mut out_s).unwrap();
        for (a, b) in out_w.iter().zip(out_s.iter()) {
            prop_assert_eq!(a, b);
        }
    }
}

// ---------------------------------------------------------------- prox_group_l1

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Singleton groups with `w_k = 1` reduce to plain L1.
    #[test]
    fn prox_group_l1_singletons_match_l1(z in vec_f64(1, 16), alpha in small_alpha()) {
        let n = z.len();
        let weights: Vec<f64> = vec![1.0; n];
        let group_ptr: Vec<i32> = (0..=n as i32).collect();
        let group_indices: Vec<i32> = (0..n as i32).collect();
        let mut out_g = vec![0.0; n];
        let mut out_l1 = vec![0.0; n];
        prox::prox_group_l1(&z, alpha, &weights, &group_ptr, &group_indices, &mut out_g).unwrap();
        prox::prox_l1(&z, alpha, &mut out_l1).unwrap();
        // `prox_l1` computes `z - α·sign(z)` directly; `prox_group_l1` factors
        // through a norm/sqrt, so the result rounds differently by a few ulps
        // for large `|z|`. Compare with a relative tolerance.
        for (a, b) in out_g.iter().zip(out_l1.iter()) {
            let scale = b.abs().max(1.0);
            prop_assert!((a - b).abs() <= 1e-12 * scale);
        }
    }

    /// Block non-expansion: `‖out_G‖ ≤ ‖z_G‖` per group.
    #[test]
    fn prox_group_l1_block_nonexpansive(
        z in vec_f64(2, 12),
        alpha in small_alpha(),
        weight in 0.5_f64..3.0,
    ) {
        // Single all-features group for simplicity; the per-group invariant
        // is a consequence of the algorithm so one group suffices to
        // exercise the kernel's contractive guarantee.
        let n = z.len();
        let weights = vec![weight];
        let group_ptr = vec![0i32, n as i32];
        let group_indices: Vec<i32> = (0..n as i32).collect();
        let mut out = vec![0.0; n];
        prox::prox_group_l1(&z, alpha, &weights, &group_ptr, &group_indices, &mut out).unwrap();
        let nrm_in: f64 = z.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nrm_out: f64 = out.iter().map(|x| x * x).sum::<f64>().sqrt();
        prop_assert!(nrm_out <= nrm_in + 1e-12);
    }
}

// ---------------------------------------------------------------- csc kernels

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// `csc::matvec` reproduces the dense reference within float tolerance.
    #[test]
    fn csc_matvec_matches_dense(
        (indptr, indices, data, n_samples, n_features) in csc_structure(),
        x_seed in vec_f64(0, 8),
    ) {
        // Tile / truncate `x_seed` to length `n_features`.
        let mut x = vec![0.0; n_features];
        for (i, val) in x.iter_mut().enumerate() {
            if !x_seed.is_empty() {
                *val = x_seed[i % x_seed.len()];
            }
        }
        let dense = csc_to_dense(&indptr, &indices, &data, n_samples, n_features);
        let expected = dense_matvec(&dense, &x);
        let mut out = vec![0.0; n_samples];
        csc::matvec(&indptr, &indices, &data, n_samples, &x, &mut out).unwrap();
        for (a, b) in out.iter().zip(expected.iter()) {
            prop_assert!((a - b).abs() < 1e-9, "matvec mismatch {a} vs {b}");
        }
    }

    /// `csc::rmatvec` reproduces the dense reference.
    #[test]
    fn csc_rmatvec_matches_dense(
        (indptr, indices, data, n_samples, n_features) in csc_structure(),
        y_seed in vec_f64(0, 8),
    ) {
        let mut y = vec![0.0; n_samples];
        for (i, val) in y.iter_mut().enumerate() {
            if !y_seed.is_empty() {
                *val = y_seed[i % y_seed.len()];
            }
        }
        let dense = csc_to_dense(&indptr, &indices, &data, n_samples, n_features);
        let expected = dense_rmatvec(&dense, &y);
        let mut out = vec![0.0; n_features];
        csc::rmatvec(&indptr, &indices, &data, &y, &mut out).unwrap();
        for (a, b) in out.iter().zip(expected.iter()) {
            prop_assert!((a - b).abs() < 1e-9, "rmatvec mismatch {a} vs {b}");
        }
    }

    /// `restricted_ls_hessian_matvec` on the full active set equals `X^T X v`.
    #[test]
    fn restricted_hess_full_active_matches_xtx(
        (indptr, indices, data, n_samples, n_features) in csc_structure(),
        v_seed in vec_f64(0, 8),
    ) {
        let active: Vec<i32> = (0..n_features as i32).collect();
        let mut v = vec![0.0; n_features];
        for (i, val) in v.iter_mut().enumerate() {
            if !v_seed.is_empty() {
                *val = v_seed[i % v_seed.len()];
            }
        }
        let dense = csc_to_dense(&indptr, &indices, &data, n_samples, n_features);
        let xv = dense_matvec(&dense, &v);
        let expected = dense_rmatvec(&dense, &xv);
        let mut out = vec![0.0; n_features];
        let mut scratch = vec![0.0; n_samples];
        residual::restricted_ls_hessian_matvec(
            &indptr, &indices, &data, n_samples, &active, &v, &mut out, &mut scratch,
        )
        .unwrap();
        for (a, b) in out.iter().zip(expected.iter()) {
            prop_assert!((a - b).abs() < 1e-9, "H_AA mismatch {a} vs {b}");
        }
    }
}
