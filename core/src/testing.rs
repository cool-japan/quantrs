//! Quantum unit testing framework
//!
//! This module provides tools for testing quantum circuits, states, and operations
//! with proper handling of quantum-specific properties like phase and entanglement.

use crate::error::QuantRS2Error;
use crate::complex_ext::QuantumComplexExt;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::fmt;

/// Tolerance for quantum state comparisons
pub const DEFAULT_TOLERANCE: f64 = 1e-10;

/// Result of a quantum test
#[derive(Debug, Clone, PartialEq)]
pub enum TestResult {
    /// Test passed
    Pass,
    /// Test failed with reason
    Fail(String),
    /// Test skipped
    Skip(String),
}

impl TestResult {
    pub fn is_pass(&self) -> bool {
        matches!(self, TestResult::Pass)
    }
}

/// Quantum state assertion helper
pub struct QuantumAssert {
    tolerance: f64,
}

impl Default for QuantumAssert {
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
        }
    }
}

impl QuantumAssert {
    /// Create with custom tolerance
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Assert two quantum states are equal (up to global phase)
    pub fn states_equal(
        &self,
        state1: &Array1<Complex64>,
        state2: &Array1<Complex64>,
    ) -> TestResult {
        if state1.len() != state2.len() {
            return TestResult::Fail(format!(
                "State dimensions mismatch: {} vs {}",
                state1.len(),
                state2.len()
            ));
        }

        // Find first non-zero amplitude to determine global phase
        let mut phase_factor = None;
        for i in 0..state1.len() {
            if state1[i].norm() > self.tolerance && state2[i].norm() > self.tolerance {
                phase_factor = Some(state2[i] / state1[i]);
                break;
            }
        }

        let phase = phase_factor.unwrap_or(Complex64::new(1.0, 0.0));

        // Check all amplitudes match after phase correction
        for i in 0..state1.len() {
            let expected = state1[i] * phase;
            if (expected - state2[i]).norm() > self.tolerance {
                return TestResult::Fail(format!(
                    "States differ at index {}: expected {}, got {}",
                    i, expected, state2[i]
                ));
            }
        }

        TestResult::Pass
    }

    /// Assert a state is normalized
    pub fn state_normalized(&self, state: &Array1<Complex64>) -> TestResult {
        let norm_squared: f64 = state.iter().map(|c| c.norm_sqr()).sum();

        if (norm_squared - 1.0).abs() > self.tolerance {
            TestResult::Fail(format!(
                "State not normalized: norm^2 = {} (expected 1.0)",
                norm_squared
            ))
        } else {
            TestResult::Pass
        }
    }

    /// Assert two states are orthogonal
    pub fn states_orthogonal(
        &self,
        state1: &Array1<Complex64>,
        state2: &Array1<Complex64>,
    ) -> TestResult {
        if state1.len() != state2.len() {
            return TestResult::Fail("State dimensions mismatch".to_string());
        }

        let inner_product: Complex64 = state1
            .iter()
            .zip(state2.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();

        if inner_product.norm() > self.tolerance {
            TestResult::Fail(format!(
                "States not orthogonal: inner product = {}",
                inner_product
            ))
        } else {
            TestResult::Pass
        }
    }

    /// Assert a matrix is unitary
    pub fn matrix_unitary(&self, matrix: &Array2<Complex64>) -> TestResult {
        let (rows, cols) = matrix.dim();
        if rows != cols {
            return TestResult::Fail(format!("Matrix not square: {}x{}", rows, cols));
        }

        // Compute U† U
        let conjugate_transpose = matrix.t().mapv(|c| c.conj());
        let product = conjugate_transpose.dot(matrix);

        // Check if it's identity
        for i in 0..rows {
            for j in 0..cols {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };

                if (product[[i, j]] - expected).norm() > self.tolerance {
                    return TestResult::Fail(format!(
                        "U†U not identity at ({},{}): got {}",
                        i,
                        j,
                        product[[i, j]]
                    ));
                }
            }
        }

        TestResult::Pass
    }

    /// Assert a state has specific measurement probabilities
    pub fn measurement_probabilities(
        &self,
        state: &Array1<Complex64>,
        expected_probs: &[(usize, f64)],
    ) -> TestResult {
        for &(index, expected_prob) in expected_probs {
            if index >= state.len() {
                return TestResult::Fail(format!(
                    "Index {} out of bounds for state of length {}",
                    index,
                    state.len()
                ));
            }

            let actual_prob = state[index].probability();
            if (actual_prob - expected_prob).abs() > self.tolerance {
                return TestResult::Fail(format!(
                    "Probability mismatch at index {}: expected {}, got {}",
                    index, expected_prob, actual_prob
                ));
            }
        }

        TestResult::Pass
    }

    /// Assert entanglement properties
    pub fn is_entangled(&self, state: &Array1<Complex64>, qubit_indices: &[usize]) -> TestResult {
        // For a 2-qubit system, check if state can be written as |ψ⟩ = |a⟩ ⊗ |b⟩
        if qubit_indices.len() != 2 {
            return TestResult::Skip(
                "Entanglement check only implemented for 2-qubit subsystems".to_string(),
            );
        }

        let n_qubits = (state.len() as f64).log2() as usize;
        if qubit_indices.iter().any(|&i| i >= n_qubits) {
            return TestResult::Fail("Qubit index out of bounds".to_string());
        }

        // Simplified check: for computational basis states
        // A separable state has rank-1 reduced density matrix
        // This is a placeholder - full implementation would compute partial trace

        TestResult::Skip("Full entanglement check not yet implemented".to_string())
    }
}

/// Quantum circuit test builder
pub struct QuantumTest {
    name: String,
    setup: Option<Box<dyn Fn() -> Result<(), QuantRS2Error>>>,
    test: Box<dyn Fn() -> TestResult>,
    teardown: Option<Box<dyn Fn()>>,
}

impl QuantumTest {
    /// Create a new quantum test
    pub fn new(name: impl Into<String>, test: impl Fn() -> TestResult + 'static) -> Self {
        Self {
            name: name.into(),
            setup: None,
            test: Box::new(test),
            teardown: None,
        }
    }

    /// Add setup function
    pub fn with_setup(mut self, setup: impl Fn() -> Result<(), QuantRS2Error> + 'static) -> Self {
        self.setup = Some(Box::new(setup));
        self
    }

    /// Add teardown function
    pub fn with_teardown(mut self, teardown: impl Fn() + 'static) -> Self {
        self.teardown = Some(Box::new(teardown));
        self
    }

    /// Run the test
    pub fn run(&self) -> TestResult {
        // Run setup
        if let Some(setup) = &self.setup {
            if let Err(e) = setup() {
                return TestResult::Fail(format!("Setup failed: {}", e));
            }
        }

        // Run test
        let result = (self.test)();

        // Run teardown
        if let Some(teardown) = &self.teardown {
            teardown();
        }

        result
    }
}

/// Test suite for organizing multiple quantum tests
pub struct QuantumTestSuite {
    name: String,
    tests: Vec<QuantumTest>,
}

impl QuantumTestSuite {
    /// Create a new test suite
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tests: Vec::new(),
        }
    }

    /// Add a test to the suite
    pub fn add_test(&mut self, test: QuantumTest) {
        self.tests.push(test);
    }

    /// Run all tests in the suite
    pub fn run(&self) -> TestSuiteResult {
        let mut results = Vec::new();

        for test in &self.tests {
            let result = test.run();
            results.push((test.name.clone(), result));
        }

        TestSuiteResult {
            suite_name: self.name.clone(),
            results,
        }
    }
}

/// Results from running a test suite
pub struct TestSuiteResult {
    suite_name: String,
    results: Vec<(String, TestResult)>,
}

impl TestSuiteResult {
    /// Get number of passed tests
    pub fn passed(&self) -> usize {
        self.results.iter().filter(|(_, r)| r.is_pass()).count()
    }

    /// Get number of failed tests
    pub fn failed(&self) -> usize {
        self.results
            .iter()
            .filter(|(_, r)| matches!(r, TestResult::Fail(_)))
            .count()
    }

    /// Get number of skipped tests
    pub fn skipped(&self) -> usize {
        self.results
            .iter()
            .filter(|(_, r)| matches!(r, TestResult::Skip(_)))
            .count()
    }

    /// Get total number of tests
    pub fn total(&self) -> usize {
        self.results.len()
    }

    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.failed() == 0
    }
}

impl fmt::Display for TestSuiteResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{} Test Results:", self.suite_name)?;
        writeln!(f, "{}", "=".repeat(50))?;

        for (name, result) in &self.results {
            let status = match result {
                TestResult::Pass => "✓ PASS",
                TestResult::Fail(_) => "✗ FAIL",
                TestResult::Skip(_) => "⊙ SKIP",
            };

            writeln!(f, "{:<6} {}", status, name)?;

            if let TestResult::Fail(reason) = result {
                writeln!(f, "       Reason: {}", reason)?;
            } else if let TestResult::Skip(reason) = result {
                writeln!(f, "       Reason: {}", reason)?;
            }
        }

        writeln!(f, "{}", "=".repeat(50))?;
        writeln!(
            f,
            "Total: {} | Passed: {} | Failed: {} | Skipped: {}",
            self.total(),
            self.passed(),
            self.failed(),
            self.skipped()
        )?;

        Ok(())
    }
}

/// Macros for quantum testing
#[macro_export]
macro_rules! quantum_test {
    ($name:expr, $test:expr) => {
        QuantumTest::new($name, $test)
    };
}

#[macro_export]
macro_rules! assert_states_equal {
    ($state1:expr, $state2:expr) => {{
        let assert = QuantumAssert::default();
        assert.states_equal($state1, $state2)
    }};
    ($state1:expr, $state2:expr, $tolerance:expr) => {{
        let assert = QuantumAssert::with_tolerance($tolerance);
        assert.states_equal($state1, $state2)
    }};
}

#[macro_export]
macro_rules! assert_unitary {
    ($matrix:expr) => {{
        let assert = QuantumAssert::default();
        assert.matrix_unitary($matrix)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_quantum_assert_states_equal() {
        let assert = QuantumAssert::default();

        // Test equal states
        let state1 = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let state2 = state1.clone();
        assert!(assert.states_equal(&state1, &state2).is_pass());

        // Test states with global phase
        let state3 = array![Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)]; // i|0⟩
        assert!(assert.states_equal(&state1, &state3).is_pass());

        // Test different states
        let state4 = array![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
        assert!(!assert.states_equal(&state1, &state4).is_pass());
    }

    #[test]
    fn test_quantum_assert_normalized() {
        let assert = QuantumAssert::default();

        // Normalized state
        let state1 = array![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0)
        ];
        assert!(assert.state_normalized(&state1).is_pass());

        // Not normalized
        let state2 = array![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        assert!(!assert.state_normalized(&state2).is_pass());
    }

    #[test]
    fn test_quantum_test_suite() {
        let mut suite = QuantumTestSuite::new("Example Suite");

        suite.add_test(QuantumTest::new("Test 1", || TestResult::Pass));
        suite.add_test(QuantumTest::new("Test 2", || {
            TestResult::Fail("Expected failure".to_string())
        }));
        suite.add_test(QuantumTest::new("Test 3", || {
            TestResult::Skip("Not implemented".to_string())
        }));

        let results = suite.run();
        assert_eq!(results.total(), 3);
        assert_eq!(results.passed(), 1);
        assert_eq!(results.failed(), 1);
        assert_eq!(results.skipped(), 1);
    }
}
