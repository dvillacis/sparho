//! Compressed Sparse Column (CSC) matrix–vector products.
//!
//! Wire format mirrors `scipy.sparse.csc_matrix`:
//! - `indptr`: column pointers, length `n_features + 1`, `i32`.
//! - `indices`: row indices for non-zeros, length `nnz`, `i32`.
//! - `data`: non-zero values, length `nnz`, `f64`.
//!
//! The caller passes `n_samples` (rows) explicitly so we don't have to scan
//! `indices` to recover it. `n_features = indptr.len() - 1`.

/// ``y = X · x`` where ``X`` is CSC of shape ``(n_samples, n_features)``.
pub fn matvec(
    indptr: &[i32],
    indices: &[i32],
    data: &[f64],
    n_samples: usize,
    x: &[f64],
    out: &mut [f64],
) {
    let n_features = indptr.len() - 1;
    assert_eq!(x.len(), n_features);
    assert_eq!(out.len(), n_samples);
    for o in out.iter_mut() {
        *o = 0.0;
    }
    for j in 0..n_features {
        let start = indptr[j] as usize;
        let end = indptr[j + 1] as usize;
        let xj = x[j];
        if xj == 0.0 {
            continue;
        }
        for k in start..end {
            let i = indices[k] as usize;
            out[i] += data[k] * xj;
        }
    }
}

/// ``z = X^T · y`` where ``X`` is CSC of shape ``(n_samples, n_features)``.
/// Output length is ``indptr.len() − 1``.
pub fn rmatvec(indptr: &[i32], indices: &[i32], data: &[f64], y: &[f64], out: &mut [f64]) {
    let n_features = indptr.len() - 1;
    assert_eq!(out.len(), n_features);
    for j in 0..n_features {
        let start = indptr[j] as usize;
        let end = indptr[j + 1] as usize;
        let mut sum = 0.0;
        for k in start..end {
            sum += data[k] * y[indices[k] as usize];
        }
        out[j] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small CSC fixture for
    /// ``X = [[1, 0, 4], [0, 2, 0], [0, 3, 5]]``  (3×3).
    fn fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let indptr = vec![0, 1, 3, 5];
        let indices = vec![0, 1, 2, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        (indptr, indices, data)
    }

    #[test]
    fn matvec_round_trip() {
        let (indptr, indices, data) = fixture();
        let x = [1.0, 1.0, 1.0];
        let mut y = vec![0.0; 3];
        matvec(&indptr, &indices, &data, 3, &x, &mut y);
        assert_eq!(y, vec![5.0, 2.0, 8.0]); // X @ [1,1,1]
    }

    #[test]
    fn rmatvec_round_trip() {
        let (indptr, indices, data) = fixture();
        let y = [1.0, 1.0, 1.0];
        let mut z = vec![0.0; 3];
        rmatvec(&indptr, &indices, &data, &y, &mut z);
        assert_eq!(z, vec![1.0, 5.0, 9.0]); // X^T @ [1,1,1] = column sums
    }
}
