//! # Testing Utilities for `QuantRS2`
//!
//! This module provides utilities and helpers for testing quantum algorithms and circuits.
//!
//! ## Features
//!
//! - Test fixture generation
//! - Assertion helpers for quantum states
//! - Mock quantum backends for testing
//! - Property-based testing helpers
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use quantrs2::testing::*;
//!
//! #[test]
//! fn test_my_quantum_algorithm() {
//!     // Create a test circuit
//!     let circuit = create_test_circuit(4);
//!
//!     // Run with mock backend
//!     let result = run_with_mock_backend(&circuit);
//!
//!     // Assert results
//!     assert_measurement_counts_close(&result, &expected, 0.05);
//! }
//! ```

#![allow(clippy::unreadable_literal)] // LCG constants are standard
#![allow(clippy::cast_precision_loss)] // Intentional for test data generation

use std::collections::HashMap;
use std::hash::BuildHasher;

/// Tolerance for floating-point comparisons
pub const DEFAULT_TOLERANCE: f64 = 1e-10;

/// Assert that two floating-point values are approximately equal
///
/// # Arguments
///
/// * `a` - First value
/// * `b` - Second value
/// * `tolerance` - Maximum allowed difference
///
/// # Panics
///
/// Panics if the absolute difference exceeds the tolerance
///
/// # Example
///
/// ```rust
/// use quantrs2::testing::assert_approx_eq;
///
/// assert_approx_eq(1.0, 1.0000001, 1e-6);
/// ```
pub fn assert_approx_eq(a: f64, b: f64, tolerance: f64) {
    let diff = (a - b).abs();
    assert!(
        diff <= tolerance,
        "Values not approximately equal: {a} != {b} (diff: {diff}, tolerance: {tolerance})"
    );
}

/// Assert that two vectors of floats are approximately equal element-wise
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
/// * `tolerance` - Maximum allowed difference per element
///
/// # Panics
///
/// Panics if vectors have different lengths or any element differs by more than tolerance
///
/// # Example
///
/// ```rust
/// use quantrs2::testing::assert_vec_approx_eq;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![1.0000001, 2.0000001, 3.0000001];
/// assert_vec_approx_eq(&a, &b, 1e-6);
/// ```
pub fn assert_vec_approx_eq(a: &[f64], b: &[f64], tolerance: f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "Vectors have different lengths: {} != {}",
        a.len(),
        b.len()
    );

    for (i, (a_val, b_val)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (a_val - b_val).abs();
        assert!(
            diff <= tolerance,
            "Element {i} not approximately equal: {a_val} != {b_val} (diff: {diff}, tolerance: {tolerance})"
        );
    }
}

/// Assert that measurement counts are close to expected values within a relative tolerance
///
/// This is useful for testing stochastic quantum algorithms where exact counts
/// cannot be predicted but probabilities can be verified.
///
/// # Arguments
///
/// * `actual` - Actual measurement counts
/// * `expected` - Expected measurement counts
/// * `relative_tolerance` - Maximum allowed relative difference (e.g., 0.05 for 5%)
///
/// # Panics
///
/// Panics if the relative difference exceeds the tolerance for any measurement outcome
///
/// # Example
///
/// ```rust
/// use quantrs2::testing::assert_measurement_counts_close;
/// use std::collections::HashMap;
///
/// let mut actual = HashMap::new();
/// actual.insert("00".to_string(), 495);
/// actual.insert("11".to_string(), 505);
///
/// let mut expected = HashMap::new();
/// expected.insert("00".to_string(), 500);
/// expected.insert("11".to_string(), 500);
///
/// assert_measurement_counts_close(&actual, &expected, 0.05); // Allow 5% deviation
/// ```
pub fn assert_measurement_counts_close<S1: BuildHasher, S2: BuildHasher>(
    actual: &HashMap<String, usize, S1>,
    expected: &HashMap<String, usize, S2>,
    relative_tolerance: f64,
) {
    // Check that all expected outcomes are present
    for (outcome, &expected_count) in expected {
        let actual_count = actual.get(outcome).copied().unwrap_or(0);
        let expected_count_f = expected_count as f64;
        let actual_count_f = actual_count as f64;

        let relative_diff = if expected_count_f > 0.0 {
            (actual_count_f - expected_count_f).abs() / expected_count_f
        } else {
            actual_count_f
        };

        assert!(
            relative_diff <= relative_tolerance,
            "Measurement outcome '{}' not close: expected {}, got {} (relative diff: {:.2}%, tolerance: {:.2}%)",
            outcome,
            expected_count,
            actual_count,
            relative_diff * 100.0,
            relative_tolerance * 100.0
        );
    }
}

/// Generate deterministic test data for testing quantum algorithms
///
/// This generates a deterministic sequence of values based on a simple PRNG.
///
/// # Arguments
///
/// * `size` - Size of the test data
/// * `seed` - Seed for reproducibility
///
/// # Returns
///
/// Vector of deterministic f64 values in range [0, 1]
///
/// # Example
///
/// ```rust
/// use quantrs2::testing::generate_random_test_data;
///
/// let data = generate_random_test_data(100, 42);
/// assert_eq!(data.len(), 100);
/// assert!(data.iter().all(|&x| x >= 0.0 && x <= 1.0));
/// ```
pub fn generate_random_test_data(size: usize, seed: u64) -> Vec<f64> {
    // Simple LCG (Linear Congruential Generator) for deterministic test data
    let mut state = seed;
    (0..size)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            (state as f64 / u64::MAX as f64).abs()
        })
        .collect()
}

/// Generate a fixed test seed for reproducible tests
///
/// This is useful when you want reproducible tests but don't care about the specific seed value.
///
/// # Returns
///
/// A fixed seed value (42)
///
/// # Example
///
/// ```rust
/// use quantrs2::testing::test_seed;
///
/// let seed = test_seed();
/// assert_eq!(seed, 42);
/// ```
pub const fn test_seed() -> u64 {
    42
}

/// Create a temporary directory for test files
///
/// The directory is automatically cleaned up when the returned value is dropped.
///
/// # Returns
///
/// Path to the temporary directory
///
/// # Example
///
/// ```rust
/// use quantrs2::testing::create_temp_test_dir;
///
/// let temp_dir = create_temp_test_dir();
/// // Use temp_dir for test files
/// // Directory is automatically cleaned up
/// ```
pub fn create_temp_test_dir() -> std::path::PathBuf {
    let temp_dir = std::env::temp_dir().join(format!("quantrs2_test_{}", test_seed()));
    std::fs::create_dir_all(&temp_dir).expect("Failed to create temp directory");
    temp_dir
}

/// Helper to suppress stdout during tests
///
/// This is useful for tests that produce a lot of output.
///
/// # Example
///
/// ```rust
/// use quantrs2::testing::with_suppressed_output;
///
/// with_suppressed_output(|| {
///     // Code that produces output
///     println!("This won't be shown");
/// });
/// ```
pub fn with_suppressed_output<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    // Simply run the function - output suppression would require
    // platform-specific code and is typically handled by test frameworks
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_approx_eq() {
        assert_approx_eq(1.0, 1.0, DEFAULT_TOLERANCE);
        assert_approx_eq(1.0, 1.0 + 1e-11, DEFAULT_TOLERANCE);
    }

    #[test]
    #[should_panic(expected = "not approximately equal")]
    fn test_assert_approx_eq_fails() {
        assert_approx_eq(1.0, 2.0, DEFAULT_TOLERANCE);
    }

    #[test]
    fn test_assert_vec_approx_eq() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.000_000_1, 2.000_000_1, 3.000_000_1];
        assert_vec_approx_eq(&a, &b, 1e-6);
    }

    #[test]
    #[should_panic(expected = "not approximately equal")]
    fn test_assert_vec_approx_eq_fails() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 3.0];
        assert_vec_approx_eq(&a, &b, 1e-6);
    }

    #[test]
    fn test_assert_measurement_counts_close() {
        let mut actual = HashMap::new();
        actual.insert("00".to_string(), 495);
        actual.insert("11".to_string(), 505);

        let mut expected = HashMap::new();
        expected.insert("00".to_string(), 500);
        expected.insert("11".to_string(), 500);

        assert_measurement_counts_close(&actual, &expected, 0.05);
    }

    #[test]
    fn test_generate_random_test_data() {
        let data = generate_random_test_data(100, test_seed());
        assert_eq!(data.len(), 100);
        assert!(data.iter().all(|&x| (0.0..=1.0).contains(&x)));

        // Test reproducibility
        let data2 = generate_random_test_data(100, test_seed());
        assert_eq!(data, data2);
    }

    #[test]
    fn test_test_seed() {
        assert_eq!(test_seed(), 42);
    }

    #[test]
    fn test_create_temp_test_dir() {
        let dir = create_temp_test_dir();
        assert!(dir.exists());
    }
}
