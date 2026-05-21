//! Compressed Sparse Column (CSC) matrix–vector products.
//!
//! Wire format mirrors `scipy.sparse.csc_matrix`:
//! - `indptr`: column pointers, length `n_features + 1`, `i32`.
//! - `indices`: row indices for non-zeros, length `nnz`, `i32`.
//! - `data`: non-zero values, length `nnz`, `f64`.
//!
//! The caller passes `n_samples` (rows) explicitly so we don't have to scan
//! `indices` to recover it. `n_features = indptr.len() - 1`.
//!
//! All public entry points validate caller-supplied indices before any `as
//! usize` cast: malformed CSC input from Python crosses the FFI boundary as
//! `PyValueError`, never as a Rust panic. (Unwinding through CPython is
//! undefined behavior; the release profile sets `panic = "abort"` so any
//! residual bug terminates the process cleanly.)

/// Validate CSC structure (`indptr` monotone non-decreasing, in-range
/// `indices` for `n_samples` rows, `data.len() == nnz`). Returns the `nnz`
/// implied by `indptr.last()` on success.
pub(crate) fn validate_csc(
    indptr: &[i32],
    indices: &[i32],
    data: &[f64],
    n_samples: usize,
) -> Result<usize, &'static str> {
    if indptr.is_empty() {
        return Err("indptr must have length n_features + 1 (>= 1)");
    }
    if indptr[0] != 0 {
        return Err("indptr[0] must be 0");
    }
    let mut prev = 0i32;
    for &p in indptr.iter() {
        if p < prev {
            return Err("indptr must be non-decreasing");
        }
        prev = p;
    }
    let nnz_i = *indptr.last().expect("non-empty checked above");
    if nnz_i < 0 {
        return Err("indptr entries must be non-negative");
    }
    let nnz = nnz_i as usize;
    if indices.len() != nnz {
        return Err("indices length must equal indptr.last()");
    }
    if data.len() != nnz {
        return Err("data length must equal indptr.last()");
    }
    let n_samples_i = i32::try_from(n_samples).map_err(|_| "n_samples too large for i32")?;
    for &i in indices.iter() {
        if i < 0 || i >= n_samples_i {
            return Err("indices entry out of range [0, n_samples)");
        }
    }
    Ok(nnz)
}

/// ``y = X · x`` where ``X`` is CSC of shape ``(n_samples, n_features)``.
pub fn matvec(
    indptr: &[i32],
    indices: &[i32],
    data: &[f64],
    n_samples: usize,
    x: &[f64],
    out: &mut [f64],
) -> Result<(), &'static str> {
    validate_csc(indptr, indices, data, n_samples)?;
    let n_features = indptr.len() - 1;
    if x.len() != n_features {
        return Err("x length must equal n_features (indptr.len() - 1)");
    }
    if out.len() != n_samples {
        return Err("out length must equal n_samples");
    }
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
    Ok(())
}

/// ``z = X^T · y`` where ``X`` is CSC of shape ``(n_samples, n_features)``.
/// Output length is ``indptr.len() − 1``; ``n_samples`` is derived from
/// ``y.len()`` (this entry point doesn't carry it separately, matching the
/// public Python signature).
pub fn rmatvec(
    indptr: &[i32],
    indices: &[i32],
    data: &[f64],
    y: &[f64],
    out: &mut [f64],
) -> Result<(), &'static str> {
    validate_csc(indptr, indices, data, y.len())?;
    let n_features = indptr.len() - 1;
    if out.len() != n_features {
        return Err("out length must equal n_features (indptr.len() - 1)");
    }
    for j in 0..n_features {
        let start = indptr[j] as usize;
        let end = indptr[j + 1] as usize;
        let mut sum = 0.0;
        for k in start..end {
            sum += data[k] * y[indices[k] as usize];
        }
        out[j] = sum;
    }
    Ok(())
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
        matvec(&indptr, &indices, &data, 3, &x, &mut y).unwrap();
        assert_eq!(y, vec![5.0, 2.0, 8.0]); // X @ [1,1,1]
    }

    #[test]
    fn rmatvec_round_trip() {
        let (indptr, indices, data) = fixture();
        let y = [1.0, 1.0, 1.0];
        let mut z = vec![0.0; 3];
        rmatvec(&indptr, &indices, &data, &y, &mut z).unwrap();
        assert_eq!(z, vec![1.0, 5.0, 9.0]); // X^T @ [1,1,1] = column sums
    }

    #[test]
    fn matvec_rejects_out_of_range_indices() {
        let indptr = vec![0, 1];
        let indices = vec![5]; // n_samples = 3 → out of range
        let data = vec![1.0];
        let x = [1.0];
        let mut y = vec![0.0; 3];
        assert!(matvec(&indptr, &indices, &data, 3, &x, &mut y).is_err());
    }

    #[test]
    fn matvec_rejects_non_monotone_indptr() {
        let indptr = vec![0, 2, 1]; // not monotone
        let indices = vec![0, 1];
        let data = vec![1.0, 1.0];
        let x = [0.0, 0.0];
        let mut y = vec![0.0; 3];
        assert!(matvec(&indptr, &indices, &data, 3, &x, &mut y).is_err());
    }

    #[test]
    fn matvec_rejects_length_mismatch() {
        let indptr = vec![0, 1];
        let indices = vec![0];
        let data = vec![1.0, 2.0]; // wrong length
        let x = [1.0];
        let mut y = vec![0.0; 1];
        assert!(matvec(&indptr, &indices, &data, 1, &x, &mut y).is_err());
    }
}
