//! Testing utilities demonstration
//!
//! This example shows how to use QuantRS2's testing helpers for quantum algorithm testing.
//!
//! Run with: cargo run --example testing_helpers

use quantrs2::testing;
use std::collections::HashMap;

fn main() {
    println!("=== QuantRS2 Testing Helpers Example ===\n");

    // 1. Floating-point assertions
    println!("1. Floating-Point Assertions:");
    demonstrate_float_assertions();
    println!();

    // 2. Vector assertions
    println!("2. Vector Assertions:");
    demonstrate_vector_assertions();
    println!();

    // 3. Measurement count assertions
    println!("3. Measurement Count Assertions (for stochastic algorithms):");
    demonstrate_measurement_assertions();
    println!();

    // 4. Deterministic test data generation
    println!("4. Deterministic Test Data Generation:");
    demonstrate_test_data_generation();
    println!();

    // 5. Test utilities
    println!("5. Test Utilities:");
    demonstrate_test_utilities();
    println!();

    println!("=== Example Complete ===");
}

fn demonstrate_float_assertions() {
    // Exact equality
    let a = 1.0;
    let b = 1.0;
    testing::assert_approx_eq(a, b, testing::DEFAULT_TOLERANCE);
    println!("   ✓ {a} ≈ {b} (exact)");

    // Close enough
    let a = 1.0;
    let b = 1.0000001;
    testing::assert_approx_eq(a, b, 1e-6);
    println!("   ✓ {a} ≈ {b} (tolerance: 1e-6)");

    // Quantum probability (should sum to 1)
    let probs = vec![0.5, 0.3, 0.2];
    let sum: f64 = probs.iter().sum();
    testing::assert_approx_eq(sum, 1.0, 1e-10);
    println!("   ✓ Probabilities sum to 1.0: {probs:?}");

    // Example: Would panic if tolerance exceeded
    // testing::assert_approx_eq(1.0, 2.0, 1e-6); // Panics!
}

fn demonstrate_vector_assertions() {
    // Quantum state vector comparison
    let state_a = vec![0.707, 0.0, 0.0, 0.707]; // |00⟩ + |11⟩ (unnormalized)
    let state_b = vec![0.707_000_1, 0.0, 0.0, 0.707_000_1];

    testing::assert_vec_approx_eq(&state_a, &state_b, 1e-5);
    println!("   ✓ State vectors are approximately equal");
    println!("     state_a: {state_a:?}");
    println!("     state_b: {state_b:?}");

    // Probability distribution comparison
    let dist_a = vec![0.25, 0.25, 0.25, 0.25];
    let dist_b = vec![0.250_001, 0.249_999, 0.250_001, 0.249_999];

    testing::assert_vec_approx_eq(&dist_a, &dist_b, 1e-5);
    println!("   ✓ Probability distributions are approximately equal");
}

fn demonstrate_measurement_assertions() {
    // Simulate Bell state measurements: |00⟩ + |11⟩
    // Expected: 50% |00⟩, 50% |11⟩
    let mut actual = HashMap::new();
    actual.insert("00".to_string(), 495); // ~50%
    actual.insert("11".to_string(), 505); // ~50%

    let mut expected = HashMap::new();
    expected.insert("00".to_string(), 500);
    expected.insert("11".to_string(), 500);

    // Allow 5% deviation (stochastic sampling)
    testing::assert_measurement_counts_close(&actual, &expected, 0.05);
    println!("   ✓ Bell state measurements within tolerance:");
    println!("     Actual:   {actual:?}");
    println!("     Expected: {expected:?}");
    println!();

    // GHZ state example: |000⟩ + |111⟩
    let mut actual = HashMap::new();
    actual.insert("000".to_string(), 487);
    actual.insert("111".to_string(), 513);

    let mut expected = HashMap::new();
    expected.insert("000".to_string(), 500);
    expected.insert("111".to_string(), 500);

    testing::assert_measurement_counts_close(&actual, &expected, 0.10);
    println!("   ✓ GHZ state measurements within tolerance:");
    println!("     Actual:   {actual:?}");
    println!("     Expected: {expected:?}");
}

fn demonstrate_test_data_generation() {
    // Generate reproducible test data
    let seed = testing::test_seed(); // Always returns 42
    println!("   Test seed: {seed}");

    let data = testing::generate_random_test_data(10, seed);
    println!("   Generated {} values: {:?}", data.len(), data);

    // Verify reproducibility
    let data2 = testing::generate_random_test_data(10, seed);
    assert_eq!(data, data2);
    println!("   ✓ Data is reproducible with same seed");
    println!();

    // Different seed produces different data
    let data3 = testing::generate_random_test_data(10, 123);
    assert_ne!(data, data3);
    println!("   ✓ Different seed produces different data");

    // All values in [0, 1]
    assert!(data.iter().all(|&x| (0.0..=1.0).contains(&x)));
    println!("   ✓ All values in range [0, 1]");
}

fn demonstrate_test_utilities() {
    // Temporary directory for test files
    let temp_dir = testing::create_temp_test_dir();
    println!("   Created temp directory: {}", temp_dir.display());
    println!("   ✓ Directory exists: {}", temp_dir.exists());

    // Test seed is consistent
    let seed1 = testing::test_seed();
    let seed2 = testing::test_seed();
    assert_eq!(seed1, seed2);
    println!("   ✓ Test seed is consistent: {seed1}");

    // Suppressed output example
    testing::with_suppressed_output(|| {
        // This output won't appear in test runs
        println!("This is suppressed in tests");
    });
    println!("   ✓ Output suppression works for noisy tests");
}

// Example test function showing real usage
#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2::testing;

    #[test]
    fn test_quantum_algorithm_convergence() {
        // Simulate some quantum algorithm result
        let expected_energy = -1.137;
        let computed_energy = -1.136_999;

        // Assert convergence within tolerance
        testing::assert_approx_eq(expected_energy, computed_energy, 1e-5);
    }

    #[test]
    fn test_bell_state_probabilities() {
        let mut counts = HashMap::new();
        counts.insert("00".to_string(), 502);
        counts.insert("11".to_string(), 498);

        let mut expected = HashMap::new();
        expected.insert("00".to_string(), 500);
        expected.insert("11".to_string(), 500);

        testing::assert_measurement_counts_close(&counts, &expected, 0.10);
    }
}
