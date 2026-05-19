//! Element-wise scalar primitives. Inlined into the hot kernels in `prox`.

/// Soft-thresholding ``ST(x, α) = sign(x) · max(|x| − α, 0)``. The Lasso prox
/// of a single coordinate.
#[inline]
pub fn soft_threshold(x: f64, alpha: f64) -> f64 {
    if x > alpha {
        x - alpha
    } else if x < -alpha {
        x + alpha
    } else {
        0.0
    }
}

/// Numerically stable logistic sigmoid ``σ(z) = 1 / (1 + exp(−z))``.
#[inline]
pub fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_threshold_branches() {
        assert_eq!(soft_threshold(2.0, 0.5), 1.5);
        assert_eq!(soft_threshold(-2.0, 0.5), -1.5);
        assert_eq!(soft_threshold(0.3, 0.5), 0.0);
        assert_eq!(soft_threshold(-0.3, 0.5), 0.0);
        assert_eq!(soft_threshold(0.0, 0.5), 0.0);
    }

    #[test]
    fn sigmoid_basic() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-15);
        // Large-magnitude inputs must not overflow.
        assert!(sigmoid(1000.0) > 0.999999);
        assert!(sigmoid(-1000.0) < 1e-6);
    }
}
