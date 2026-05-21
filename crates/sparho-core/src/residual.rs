//! Restricted Hessian–vector products consumed by the hypergradient GMRES.
//!
//! At v0.1 only the least-squares Hessian ``H = X^T X`` is supported (Lasso,
//! ElasticNet, WeightedLasso). Logistic loss adds a sample-weighted diagonal
//! and is deferred — the Python layer may densify locally for logistic.

use crate::csc::validate_csc;

/// ``H_AA · v`` where ``H = X^T X`` and ``X`` is CSC; ``A`` is `active`.
///
/// Two-pass implementation: ``y = X_A · v`` into `scratch` (dense
/// length-`n_samples`), then ``out = X_A^T · y``. Caller-owned scratch keeps
/// the hot path allocation-free.
#[allow(clippy::too_many_arguments)]
pub fn restricted_ls_hessian_matvec(
    indptr: &[i32],
    indices: &[i32],
    data: &[f64],
    n_samples: usize,
    active: &[i32],
    v: &[f64],
    out: &mut [f64],
    scratch: &mut [f64],
) -> Result<(), &'static str> {
    validate_csc(indptr, indices, data, n_samples)?;
    if active.len() != v.len() {
        return Err("active and v must have the same length");
    }
    if active.len() != out.len() {
        return Err("out length must equal active length");
    }
    if scratch.len() != n_samples {
        return Err("scratch length must equal n_samples");
    }
    let n_features_i =
        i32::try_from(indptr.len() - 1).map_err(|_| "n_features too large for i32")?;
    for &j in active.iter() {
        if j < 0 || j >= n_features_i {
            return Err("active entry out of range [0, n_features)");
        }
    }
    for s in scratch.iter_mut() {
        *s = 0.0;
    }
    // y = X[:, active] @ v  (gather into scratch)
    for (idx, &j) in active.iter().enumerate() {
        let j = j as usize;
        let start = indptr[j] as usize;
        let end = indptr[j + 1] as usize;
        let vj = v[idx];
        if vj == 0.0 {
            continue;
        }
        for k in start..end {
            let i = indices[k] as usize;
            scratch[i] += data[k] * vj;
        }
    }
    // out = X[:, active]^T @ y
    for (idx, &j) in active.iter().enumerate() {
        let j = j as usize;
        let start = indptr[j] as usize;
        let end = indptr[j + 1] as usize;
        let mut sum = 0.0;
        for k in start..end {
            sum += data[k] * scratch[indices[k] as usize];
        }
        out[idx] = sum;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ``X = [[1, 0, 4], [0, 2, 0], [0, 3, 5]]`` (3×3).
    /// ``X^T X = [[1, 0, 4], [0, 13, 15], [4, 15, 41]]``.
    fn fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        (
            vec![0, 1, 3, 5],
            vec![0, 1, 2, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
    }

    #[test]
    fn full_active_set_recovers_xtx() {
        let (indptr, indices, data) = fixture();
        let active = [0i32, 1, 2];
        // Multiply H_AA by each unit vector and check the resulting column.
        let mut expected_cols: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, 4.0],
            vec![0.0, 13.0, 15.0],
            vec![4.0, 15.0, 41.0],
        ];
        for (j, expected) in expected_cols.iter_mut().enumerate() {
            let mut v = vec![0.0; 3];
            v[j] = 1.0;
            let mut out = vec![0.0; 3];
            let mut scratch = vec![0.0; 3];
            restricted_ls_hessian_matvec(
                &indptr,
                &indices,
                &data,
                3,
                &active,
                &v,
                &mut out,
                &mut scratch,
            )
            .unwrap();
            assert_eq!(out, *expected);
        }
    }

    #[test]
    fn subset_active_extracts_submatrix() {
        let (indptr, indices, data) = fixture();
        // active = {0, 2}; expected H_AA = [[1, 4], [4, 41]].
        let active = [0i32, 2];
        let mut v = vec![1.0, 0.0];
        let mut out = vec![0.0; 2];
        let mut scratch = vec![0.0; 3];
        restricted_ls_hessian_matvec(
            &indptr,
            &indices,
            &data,
            3,
            &active,
            &v,
            &mut out,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(out, vec![1.0, 4.0]);
        v = vec![0.0, 1.0];
        restricted_ls_hessian_matvec(
            &indptr,
            &indices,
            &data,
            3,
            &active,
            &v,
            &mut out,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(out, vec![4.0, 41.0]);
    }

    #[test]
    fn rejects_out_of_range_active() {
        let (indptr, indices, data) = fixture();
        let active = [5i32]; // n_features = 3
        let v = vec![1.0];
        let mut out = vec![0.0; 1];
        let mut scratch = vec![0.0; 3];
        assert!(restricted_ls_hessian_matvec(
            &indptr,
            &indices,
            &data,
            3,
            &active,
            &v,
            &mut out,
            &mut scratch,
        )
        .is_err());
    }

    #[test]
    fn rejects_negative_active() {
        let (indptr, indices, data) = fixture();
        let active = [-1i32];
        let v = vec![1.0];
        let mut out = vec![0.0; 1];
        let mut scratch = vec![0.0; 3];
        assert!(restricted_ls_hessian_matvec(
            &indptr,
            &indices,
            &data,
            3,
            &active,
            &v,
            &mut out,
            &mut scratch,
        )
        .is_err());
    }
}
