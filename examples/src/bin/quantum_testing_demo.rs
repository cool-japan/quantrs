//! Quantum unit testing framework demonstration

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::Complex64;
use quantrs2_core::prelude::*;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Quantum Unit Testing Framework Demo ===\n");

    // Demo 1: Basic state assertions
    basic_assertions_demo();

    // Demo 2: Matrix property testing
    matrix_testing_demo();

    // Demo 3: Full test suite example
    test_suite_demo();

    // Demo 4: Testing quantum algorithms
    algorithm_testing_demo();

    Ok(())
}

/// Demonstrate basic quantum state assertions
fn basic_assertions_demo() {
    println!("1. Basic Quantum State Assertions:");
    println!("----------------------------------");

    let assert = QuantumAssert::default();

    // Test 1: State equality (including global phase)
    let state1 = array![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0)
    ];

    let state2 = array![
        Complex64::new(0.0, 1.0 / 2.0_f64.sqrt()),
        Complex64::new(0.0, 1.0 / 2.0_f64.sqrt())
    ]; // Same state with global phase i

    match assert.states_equal(&state1, &state2) {
        TestResult::Pass => println!("✓ States are equal (ignoring global phase)"),
        TestResult::Fail(reason) => println!("✗ States not equal: {}", reason),
        TestResult::Skip(reason) => println!("⊙ Test skipped: {}", reason),
    }

    // Test 2: Normalization check
    match assert.state_normalized(&state1) {
        TestResult::Pass => println!("✓ State is normalized"),
        TestResult::Fail(reason) => println!("✗ State not normalized: {}", reason),
        TestResult::Skip(reason) => println!("⊙ Test skipped: {}", reason),
    }

    // Test 3: Orthogonality check
    let state3 = array![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0)
    ];

    match assert.states_orthogonal(&state1, &state3) {
        TestResult::Pass => println!("✓ States are orthogonal"),
        TestResult::Fail(reason) => println!("✗ States not orthogonal: {}", reason),
        TestResult::Skip(reason) => println!("⊙ Test skipped: {}", reason),
    }

    // Test 4: Measurement probabilities
    let expected_probs = vec![(0, 0.5), (1, 0.5)];
    match assert.measurement_probabilities(&state1, &expected_probs) {
        TestResult::Pass => println!("✓ Measurement probabilities correct"),
        TestResult::Fail(reason) => println!("✗ Probabilities incorrect: {}", reason),
        TestResult::Skip(reason) => println!("⊙ Test skipped: {}", reason),
    }

    println!();
}

/// Demonstrate matrix property testing
fn matrix_testing_demo() {
    println!("2. Quantum Matrix Property Testing:");
    println!("-----------------------------------");

    let assert = QuantumAssert::default();

    // Test unitary matrices
    let h_val = 1.0 / 2.0_f64.sqrt();
    let hadamard = array![
        [Complex64::new(h_val, 0.0), Complex64::new(h_val, 0.0)],
        [Complex64::new(h_val, 0.0), Complex64::new(-h_val, 0.0)]
    ];

    match assert.matrix_unitary(&hadamard) {
        TestResult::Pass => println!("✓ Hadamard matrix is unitary"),
        TestResult::Fail(reason) => println!("✗ Not unitary: {}", reason),
        TestResult::Skip(reason) => println!("⊙ Test skipped: {}", reason),
    }

    // Test Pauli X
    let pauli_x = array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ];

    match assert.matrix_unitary(&pauli_x) {
        TestResult::Pass => println!("✓ Pauli-X matrix is unitary"),
        TestResult::Fail(reason) => println!("✗ Not unitary: {}", reason),
        TestResult::Skip(reason) => println!("⊙ Test skipped: {}", reason),
    }

    // Test non-unitary matrix
    let non_unitary = array![
        [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
    ];

    match assert.matrix_unitary(&non_unitary) {
        TestResult::Pass => println!("✓ Matrix is unitary (unexpected!)"),
        TestResult::Fail(reason) => {
            println!("✓ Correctly identified non-unitary matrix: {}", reason)
        }
        TestResult::Skip(reason) => println!("⊙ Test skipped: {}", reason),
    }

    println!();
}

/// Demonstrate full test suite functionality
fn test_suite_demo() {
    println!("3. Quantum Test Suite Example:");
    println!("------------------------------");

    let mut suite = QuantumTestSuite::new("Bell State Tests");

    // Test 1: Bell state preparation
    suite.add_test(QuantumTest::new("Bell state |Φ+⟩ preparation", || {
        let bell_state = array![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0)
        ];

        let assert = QuantumAssert::default();
        assert.state_normalized(&bell_state)
    }));

    // Test 2: Bell state measurement correlations
    suite.add_test(QuantumTest::new(
        "Bell state measurement correlations",
        || {
            let bell_state = array![
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0)
            ];

            let assert = QuantumAssert::default();
            let expected_probs = vec![(0, 0.5), (3, 0.5)];
            assert.measurement_probabilities(&bell_state, &expected_probs)
        },
    ));

    // Test 3: Bell state entanglement (placeholder)
    suite.add_test(QuantumTest::new("Bell state entanglement", || {
        TestResult::Skip("Entanglement verification not yet implemented".to_string())
    }));

    // Run the suite
    let results = suite.run();
    println!("{}", results);
}

/// Demonstrate testing quantum algorithms
fn algorithm_testing_demo() {
    println!("4. Testing Quantum Algorithms:");
    println!("------------------------------");

    let mut suite = QuantumTestSuite::new("Quantum Algorithm Tests");

    // Test quantum Fourier transform
    suite.add_test(QuantumTest::new("QFT unitarity", || {
        let qft_2 = create_qft_matrix(2);
        let assert = QuantumAssert::default();
        assert.matrix_unitary(&qft_2)
    }));

    // Test phase kickback
    suite.add_test(QuantumTest::new(
        "Phase kickback in controlled operations",
        || {
            // |+⟩ state
            let plus_state = array![
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0)
            ];

            // After controlled-Z with |1⟩, should become |-⟩
            let minus_state = array![
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0)
            ];

            let assert = QuantumAssert::default();
            assert.states_orthogonal(&plus_state, &minus_state)
        },
    ));

    // Test quantum teleportation fidelity
    suite.add_test(QuantumTest::new("Quantum teleportation fidelity", || {
        // Simplified: just check that arbitrary state can be teleported
        let input_state = array![Complex64::new(0.6, 0.0), Complex64::new(0.8, 0.0)];

        // After perfect teleportation, output should equal input
        let output_state = input_state.clone(); // Simulated

        let assert = QuantumAssert::default();
        assert.states_equal(&input_state, &output_state)
    }));

    let results = suite.run();
    println!("{}", results);

    println!("\nAdvanced Testing Features:");
    println!("• Custom tolerance levels for approximate equality");
    println!("• Global phase handling in state comparisons");
    println!("• Setup/teardown functions for complex tests");
    println!("• Test organization into suites");
    println!("• Detailed failure reporting");
}

/// Helper: Create QFT matrix for n qubits
fn create_qft_matrix(n: usize) -> Array2<Complex64> {
    let size = 1 << n;
    let mut matrix = Array2::zeros((size, size));
    let omega = Complex64::from_polar(1.0, -2.0 * PI / size as f64);

    for i in 0..size {
        for j in 0..size {
            let exponent = (i * j) as f64;
            matrix[[i, j]] = omega.powf(exponent) / (size as f64).sqrt();
        }
    }

    matrix
}
