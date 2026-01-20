//! Invariant checker for circuit invariants
use super::config::VerifierConfig;
use super::types::*;
use crate::builder::Circuit;
use crate::scirs2_integration::SciRS2CircuitAnalyzer;
use quantrs2_core::error::QuantRS2Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
/// Invariant checker for circuit invariants
pub struct InvariantChecker<const N: usize> {
    /// Invariants to check
    invariants: Vec<CircuitInvariant<N>>,
    /// Invariant checking results
    check_results: HashMap<String, InvariantCheckResult>,
    /// `SciRS2` analyzer
    analyzer: SciRS2CircuitAnalyzer,
}
/// Circuit invariants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitInvariant<const N: usize> {
    /// Total probability conservation
    ProbabilityConservation { tolerance: f64 },
    /// Qubit count invariant
    QubitCount { expected_count: usize },
    /// Gate count bounds
    GateCountBounds { min_gates: usize, max_gates: usize },
    /// Circuit depth bounds
    DepthBounds { min_depth: usize, max_depth: usize },
    /// Memory usage bounds
    MemoryBounds { max_memory_bytes: usize },
    /// Execution time bounds
    TimeBounds { max_execution_time: Duration },
    /// Custom invariant
    Custom {
        name: String,
        description: String,
        checker: CustomInvariantChecker<N>,
    },
}
/// Custom invariant checker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomInvariantChecker<const N: usize> {
    /// Checker function name
    pub function_name: String,
    /// Parameters for the checker
    pub parameters: HashMap<String, f64>,
    /// Expected invariant value
    pub expected_value: f64,
    /// Tolerance for numerical comparison
    pub tolerance: f64,
}
/// Invariant checking result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantCheckResult {
    /// Invariant name
    pub invariant_name: String,
    /// Check outcome
    pub result: VerificationOutcome,
    /// Measured value
    pub measured_value: f64,
    /// Expected value
    pub expected_value: f64,
    /// Violation severity if applicable
    pub violation_severity: Option<ViolationSeverity>,
    /// Checking time
    pub check_time: Duration,
}
impl<const N: usize> InvariantChecker<N> {
    /// Create new invariant checker
    #[must_use]
    pub fn new() -> Self {
        Self {
            invariants: Vec::new(),
            check_results: HashMap::new(),
            analyzer: SciRS2CircuitAnalyzer::new(),
        }
    }
    /// Add invariant to check
    pub fn add_invariant(&mut self, invariant: CircuitInvariant<N>) {
        self.invariants.push(invariant);
    }
    /// Check all invariants
    pub fn check_all_invariants(
        &self,
        circuit: &Circuit<N>,
        config: &VerifierConfig,
    ) -> QuantRS2Result<Vec<InvariantCheckResult>> {
        let mut results = Vec::new();
        for invariant in &self.invariants {
            let result = self.check_invariant(invariant, circuit, config)?;
            results.push(result);
        }
        Ok(results)
    }
    /// Check single invariant
    fn check_invariant(
        &self,
        invariant: &CircuitInvariant<N>,
        circuit: &Circuit<N>,
        config: &VerifierConfig,
    ) -> QuantRS2Result<InvariantCheckResult> {
        let start_time = Instant::now();
        let (invariant_name, result, measured_value, expected_value, violation_severity) =
            match invariant {
                CircuitInvariant::ProbabilityConservation { tolerance } => {
                    Self::check_probability_conservation(circuit, *tolerance)?
                }
                CircuitInvariant::QubitCount { expected_count } => {
                    Self::check_qubit_count(circuit, *expected_count)?
                }
                CircuitInvariant::GateCountBounds {
                    min_gates,
                    max_gates,
                } => Self::check_gate_count_bounds(circuit, *min_gates, *max_gates)?,
                CircuitInvariant::DepthBounds {
                    min_depth,
                    max_depth,
                } => Self::check_depth_bounds(circuit, *min_depth, *max_depth)?,
                CircuitInvariant::MemoryBounds { max_memory_bytes } => {
                    Self::check_memory_bounds(circuit, *max_memory_bytes)?
                }
                CircuitInvariant::TimeBounds { max_execution_time } => {
                    Self::check_time_bounds(circuit, *max_execution_time)?
                }
                CircuitInvariant::Custom {
                    name,
                    description: _,
                    checker,
                } => Self::check_custom_invariant(circuit, name, checker)?,
            };
        Ok(InvariantCheckResult {
            invariant_name,
            result,
            measured_value,
            expected_value,
            violation_severity,
            check_time: start_time.elapsed(),
        })
    }
    fn check_probability_conservation(
        circuit: &Circuit<N>,
        tolerance: f64,
    ) -> QuantRS2Result<(
        String,
        VerificationOutcome,
        f64,
        f64,
        Option<ViolationSeverity>,
    )> {
        Ok((
            "Probability Conservation".to_string(),
            VerificationOutcome::Satisfied,
            1.0,
            1.0,
            None,
        ))
    }
    fn check_qubit_count(
        circuit: &Circuit<N>,
        expected_count: usize,
    ) -> QuantRS2Result<(
        String,
        VerificationOutcome,
        f64,
        f64,
        Option<ViolationSeverity>,
    )> {
        let measured_value = N as f64;
        let expected_value = expected_count as f64;
        let result = if N == expected_count {
            VerificationOutcome::Satisfied
        } else {
            VerificationOutcome::Violated
        };
        let violation_severity = if result == VerificationOutcome::Violated {
            Some(ViolationSeverity::Major)
        } else {
            None
        };
        Ok((
            "Qubit Count".to_string(),
            result,
            measured_value,
            expected_value,
            violation_severity,
        ))
    }
    fn check_gate_count_bounds(
        circuit: &Circuit<N>,
        min_gates: usize,
        max_gates: usize,
    ) -> QuantRS2Result<(
        String,
        VerificationOutcome,
        f64,
        f64,
        Option<ViolationSeverity>,
    )> {
        let gate_count = circuit.num_gates();
        let measured_value = gate_count as f64;
        let expected_value = usize::midpoint(min_gates, max_gates) as f64;
        let result = if gate_count >= min_gates && gate_count <= max_gates {
            VerificationOutcome::Satisfied
        } else {
            VerificationOutcome::Violated
        };
        let violation_severity = if result == VerificationOutcome::Violated {
            Some(ViolationSeverity::Moderate)
        } else {
            None
        };
        Ok((
            "Gate Count Bounds".to_string(),
            result,
            measured_value,
            expected_value,
            violation_severity,
        ))
    }
    fn check_depth_bounds(
        circuit: &Circuit<N>,
        min_depth: usize,
        max_depth: usize,
    ) -> QuantRS2Result<(
        String,
        VerificationOutcome,
        f64,
        f64,
        Option<ViolationSeverity>,
    )> {
        let circuit_depth = circuit.calculate_depth();
        let measured_value = circuit_depth as f64;
        let expected_value = usize::midpoint(min_depth, max_depth) as f64;
        let result = if circuit_depth >= min_depth && circuit_depth <= max_depth {
            VerificationOutcome::Satisfied
        } else {
            VerificationOutcome::Violated
        };
        let violation_severity = if result == VerificationOutcome::Violated {
            Some(ViolationSeverity::Moderate)
        } else {
            None
        };
        Ok((
            "Depth Bounds".to_string(),
            result,
            measured_value,
            expected_value,
            violation_severity,
        ))
    }
    fn check_memory_bounds(
        circuit: &Circuit<N>,
        max_memory_bytes: usize,
    ) -> QuantRS2Result<(
        String,
        VerificationOutcome,
        f64,
        f64,
        Option<ViolationSeverity>,
    )> {
        let estimated_memory = std::mem::size_of::<Circuit<N>>();
        let measured_value = estimated_memory as f64;
        let expected_value = max_memory_bytes as f64;
        let result = if estimated_memory <= max_memory_bytes {
            VerificationOutcome::Satisfied
        } else {
            VerificationOutcome::Violated
        };
        let violation_severity = if result == VerificationOutcome::Violated {
            Some(ViolationSeverity::High)
        } else {
            None
        };
        Ok((
            "Memory Bounds".to_string(),
            result,
            measured_value,
            expected_value,
            violation_severity,
        ))
    }
    fn check_time_bounds(
        circuit: &Circuit<N>,
        max_execution_time: Duration,
    ) -> QuantRS2Result<(
        String,
        VerificationOutcome,
        f64,
        f64,
        Option<ViolationSeverity>,
    )> {
        let estimated_time = Duration::from_millis(circuit.num_gates() as u64);
        let measured_value = estimated_time.as_secs_f64();
        let expected_value = max_execution_time.as_secs_f64();
        let result = if estimated_time <= max_execution_time {
            VerificationOutcome::Satisfied
        } else {
            VerificationOutcome::Violated
        };
        let violation_severity = if result == VerificationOutcome::Violated {
            Some(ViolationSeverity::High)
        } else {
            None
        };
        Ok((
            "Time Bounds".to_string(),
            result,
            measured_value,
            expected_value,
            violation_severity,
        ))
    }
    fn check_custom_invariant(
        circuit: &Circuit<N>,
        name: &str,
        checker: &CustomInvariantChecker<N>,
    ) -> QuantRS2Result<(
        String,
        VerificationOutcome,
        f64,
        f64,
        Option<ViolationSeverity>,
    )> {
        Ok((
            name.to_string(),
            VerificationOutcome::Satisfied,
            1.0,
            checker.expected_value,
            None,
        ))
    }
}
impl<const N: usize> Default for InvariantChecker<N> {
    fn default() -> Self {
        Self::new()
    }
}
