//! sparho-core: pure-Rust kernels for sparho.
//!
//! All public functions take input slices and write into caller-provided
//! output slices to keep the PyO3 hot path allocation-free.

pub mod bcd;
pub mod csc;
pub mod kernels;
pub mod prox;
pub mod residual;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::version;

    #[test]
    fn version_is_nonempty() {
        assert!(!version().is_empty());
    }
}
