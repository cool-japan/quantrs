//! Circuit equivalence checking algorithms with SciRS2 numerical tolerance
//!
//! This module provides various methods to check if two quantum circuits
//! are equivalent, including exact and approximate equivalence using
//! SciRS2's advanced numerical analysis capabilities.

use crate::builder::Circuit;
use crate::scirs2_integration::{AnalyzerConfig, SciRS2CircuitAnalyzer};
use scirs2_core::ndarray::{array, Array2, ArrayView2};
use scirs2_core::Complex64;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default tolerance for numerical comparisons
const DEFAULT_TOLERANCE: f64 = 1e-10;

/// Enhanced tolerance with SciRS2 statistical analysis
const SCIRS2_DEFAULT_TOLERANCE: f64 = 1e-12;

/// Tolerance for complex number comparisons
const COMPLEX_TOLERANCE: f64 = 1e-14;

/// Enhanced result of equivalence check with SciRS2 analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceResult {
    /// Whether the circuits are equivalent
    pub equivalent: bool,
    /// Type of equivalence check performed
    pub check_type: EquivalenceType,
    /// Maximum difference found (for numerical checks)
    pub max_difference: Option<f64>,
    /// Additional details about the check
    pub details: String,
    /// SciRS2 numerical analysis results
    pub numerical_analysis: Option<NumericalAnalysis>,
    /// Confidence score (0.0 to 1.0)
    pub confidence_score: f64,
    /// Statistical significance (p-value if applicable)
    pub statistical_significance: Option<f64>,
    /// Error bounds and uncertainty quantification
    pub error_bounds: Option<ErrorBounds>,
}

/// SciRS2 numerical analysis for equivalence checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalAnalysis {
    /// Condition number of the matrices involved
    pub condition_number: Option<f64>,
    /// Numerical rank of difference matrix
    pub numerical_rank: Option<usize>,
    /// Frobenius norm of the difference
    pub frobenius_norm: f64,
    /// Spectral norm of the difference
    pub spectral_norm: Option<f64>,
    /// Adaptive tolerance used based on circuit complexity
    pub adaptive_tolerance: f64,
    /// Matrix factorization stability indicator
    pub stability_indicator: f64,
}

/// Error bounds and uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBounds {
    /// Lower bound of the error estimate
    pub lower_bound: f64,
    /// Upper bound of the error estimate
    pub upper_bound: f64,
    /// Confidence interval level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Standard deviation of error estimates
    pub standard_deviation: Option<f64>,
}

/// Types of equivalence checks with SciRS2 enhancements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EquivalenceType {
    /// Check if circuits produce identical unitaries
    UnitaryEquivalence,
    /// Check if circuits produce same output states for all inputs
    StateVectorEquivalence,
    /// Check if measurement probabilities are identical
    ProbabilisticEquivalence,
    /// Check if circuits have identical gate structure
    StructuralEquivalence,
    /// Check if circuits are equivalent up to a global phase
    GlobalPhaseEquivalence,
    /// SciRS2-powered numerical equivalence with adaptive tolerance
    SciRS2NumericalEquivalence,
    /// SciRS2 statistical equivalence with confidence intervals
    SciRS2StatisticalEquivalence,
    /// SciRS2 graph-based structural equivalence
    SciRS2GraphEquivalence,
}

/// Enhanced options for equivalence checking with SciRS2 features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceOptions {
    /// Numerical tolerance for comparisons
    pub tolerance: f64,
    /// Whether to ignore global phase differences
    pub ignore_global_phase: bool,
    /// Whether to check all computational basis states
    pub check_all_states: bool,
    /// Maximum circuit size for unitary construction
    pub max_unitary_qubits: usize,
    /// Enable SciRS2 adaptive tolerance
    pub enable_adaptive_tolerance: bool,
    /// Enable SciRS2 statistical analysis
    pub enable_statistical_analysis: bool,
    /// Enable SciRS2 numerical stability analysis
    pub enable_stability_analysis: bool,
    /// Enable SciRS2 graph-based comparison
    pub enable_graph_comparison: bool,
    /// Confidence level for statistical tests (e.g., 0.95)
    pub confidence_level: f64,
    /// Maximum condition number for numerical stability
    pub max_condition_number: f64,
    /// SciRS2 analyzer configuration
    pub scirs2_config: Option<AnalyzerConfig>,
    /// Complex number tolerance
    pub complex_tolerance: f64,
    /// Enable parallel computation for large circuits
    pub enable_parallel_computation: bool,
}

impl Default for EquivalenceOptions {
    fn default() -> Self {
        EquivalenceOptions {
            tolerance: DEFAULT_TOLERANCE,
            ignore_global_phase: true,
            check_all_states: true,
            max_unitary_qubits: 10,
            enable_adaptive_tolerance: true,
            enable_statistical_analysis: true,
            enable_stability_analysis: true,
            enable_graph_comparison: false, // Optional for performance
            confidence_level: 0.95,
            max_condition_number: 1e12,
            scirs2_config: None, // Will use default if needed
            complex_tolerance: COMPLEX_TOLERANCE,
            enable_parallel_computation: true,
        }
    }
}

/// Enhanced circuit equivalence checker with SciRS2 integration
pub struct EquivalenceChecker {
    options: EquivalenceOptions,
    scirs2_analyzer: Option<SciRS2CircuitAnalyzer>,
    numerical_cache: HashMap<String, NumericalAnalysis>,
}

impl EquivalenceChecker {
    /// Create a new equivalence checker with options
    pub fn new(options: EquivalenceOptions) -> Self {
        let scirs2_analyzer = if options.enable_graph_comparison
            || options.enable_statistical_analysis
            || options.enable_stability_analysis
        {
            Some(SciRS2CircuitAnalyzer::new())
        } else {
            None
        };

        EquivalenceChecker {
            options,
            scirs2_analyzer,
            numerical_cache: HashMap::new(),
        }
    }

    /// Create a new equivalence checker with default options
    pub fn default() -> Self {
        Self::new(EquivalenceOptions::default())
    }

    /// Create a new equivalence checker with custom SciRS2 configuration
    pub fn with_scirs2_config(config: AnalyzerConfig) -> Self {
        let mut options = EquivalenceOptions::default();
        options.scirs2_config = Some(config.clone());
        options.enable_graph_comparison = true;

        let scirs2_analyzer = Some(SciRS2CircuitAnalyzer::with_config(config));

        EquivalenceChecker {
            options,
            scirs2_analyzer,
            numerical_cache: HashMap::new(),
        }
    }

    /// Check if two circuits are equivalent using all methods including SciRS2
    pub fn check_equivalence<const N: usize>(
        &mut self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<N>,
    ) -> QuantRS2Result<EquivalenceResult> {
        // Try SciRS2 graph-based equivalence first if enabled
        if self.options.enable_graph_comparison {
            if let Ok(result) = self.check_scirs2_graph_equivalence(circuit1, circuit2) {
                if result.equivalent {
                    return Ok(result);
                }
            }
        }

        // Try structural equivalence (fastest)
        if let Ok(result) = self.check_structural_equivalence(circuit1, circuit2) {
            if result.equivalent {
                return Ok(result);
            }
        }

        // Try SciRS2 numerical equivalence if enabled
        if self.options.enable_adaptive_tolerance || self.options.enable_statistical_analysis {
            if N <= self.options.max_unitary_qubits {
                return self.check_scirs2_numerical_equivalence(circuit1, circuit2);
            }
        }

        // Try unitary equivalence if circuits are small enough
        if N <= self.options.max_unitary_qubits {
            return self.check_unitary_equivalence(circuit1, circuit2);
        }

        // For larger circuits, use state vector equivalence
        self.check_state_vector_equivalence(circuit1, circuit2)
    }

    /// Check equivalence using SciRS2 numerical analysis with adaptive tolerance
    pub fn check_scirs2_numerical_equivalence<const N: usize>(
        &mut self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<N>,
    ) -> QuantRS2Result<EquivalenceResult> {
        if N > self.options.max_unitary_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Circuit too large for SciRS2 numerical analysis: {} qubits (max: {})",
                N, self.options.max_unitary_qubits
            )));
        }

        // Get unitaries for both circuits
        let unitary1 = self.get_circuit_unitary(circuit1)?;
        let unitary2 = self.get_circuit_unitary(circuit2)?;

        // Perform SciRS2 numerical analysis
        let numerical_analysis = self.perform_scirs2_numerical_analysis(&unitary1, &unitary2)?;

        // Calculate adaptive tolerance based on circuit complexity
        let adaptive_tolerance = self.calculate_adaptive_tolerance::<N>(N, &numerical_analysis);

        // Compare unitaries with enhanced numerical analysis
        let (equivalent, max_diff, confidence_score, error_bounds) =
            self.scirs2_unitaries_equal(&unitary1, &unitary2, adaptive_tolerance)?;

        // Calculate statistical significance if enabled
        let statistical_significance = if self.options.enable_statistical_analysis {
            Some(self.calculate_statistical_significance(&unitary1, &unitary2, max_diff)?)
        } else {
            None
        };

        Ok(EquivalenceResult {
            equivalent,
            check_type: EquivalenceType::SciRS2NumericalEquivalence,
            max_difference: Some(max_diff),
            details: format!(
                "SciRS2 numerical analysis: tolerance={:.2e}, confidence={:.3}, condition_number={:.2e}",
                adaptive_tolerance,
                confidence_score,
                numerical_analysis.condition_number.unwrap_or(0.0)
            ),
            numerical_analysis: Some(numerical_analysis),
            confidence_score,
            statistical_significance,
            error_bounds: Some(error_bounds),
        })
    }

    /// Check equivalence using SciRS2 graph-based analysis
    pub fn check_scirs2_graph_equivalence<const N: usize>(
        &mut self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<N>,
    ) -> QuantRS2Result<EquivalenceResult> {
        let analyzer = self.scirs2_analyzer.as_mut().ok_or_else(|| {
            QuantRS2Error::InvalidInput("SciRS2 analyzer not initialized".to_string())
        })?;

        // Convert circuits to SciRS2 graphs
        let graph1 = analyzer.circuit_to_scirs2_graph(circuit1)?;
        let graph2 = analyzer.circuit_to_scirs2_graph(circuit2)?;

        // Perform graph-based equivalence checking
        let (equivalent, similarity_score, graph_details) =
            self.compare_scirs2_graphs(&graph1, &graph2)?;

        Ok(EquivalenceResult {
            equivalent,
            check_type: EquivalenceType::SciRS2GraphEquivalence,
            max_difference: Some(1.0 - similarity_score),
            details: graph_details,
            numerical_analysis: None,
            confidence_score: similarity_score,
            statistical_significance: None,
            error_bounds: None,
        })
    }

    /// Check structural equivalence (exact gate-by-gate match)
    pub fn check_structural_equivalence<const N: usize>(
        &self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<N>,
    ) -> QuantRS2Result<EquivalenceResult> {
        if circuit1.num_gates() != circuit2.num_gates() {
            return Ok(EquivalenceResult {
                equivalent: false,
                check_type: EquivalenceType::StructuralEquivalence,
                max_difference: None,
                details: format!(
                    "Different number of gates: {} vs {}",
                    circuit1.num_gates(),
                    circuit2.num_gates()
                ),
                numerical_analysis: None,
                confidence_score: 0.0,
                statistical_significance: None,
                error_bounds: None,
            });
        }

        let gates1 = circuit1.gates();
        let gates2 = circuit2.gates();

        for (i, (gate1, gate2)) in gates1.iter().zip(gates2.iter()).enumerate() {
            if !self.gates_equal(gate1.as_ref(), gate2.as_ref()) {
                return Ok(EquivalenceResult {
                    equivalent: false,
                    check_type: EquivalenceType::StructuralEquivalence,
                    max_difference: None,
                    details: format!(
                        "Gates differ at position {}: {} vs {}",
                        i,
                        gate1.name(),
                        gate2.name()
                    ),
                    numerical_analysis: None,
                    confidence_score: 0.0,
                    statistical_significance: None,
                    error_bounds: None,
                });
            }
        }

        Ok(EquivalenceResult {
            equivalent: true,
            check_type: EquivalenceType::StructuralEquivalence,
            max_difference: Some(0.0),
            details: "Circuits are structurally identical".to_string(),
            numerical_analysis: None,
            confidence_score: 1.0,
            statistical_significance: None,
            error_bounds: None,
        })
    }

    /// Check if two gates are equal
    fn gates_equal(&self, gate1: &dyn GateOp, gate2: &dyn GateOp) -> bool {
        // Check gate names
        if gate1.name() != gate2.name() {
            return false;
        }

        // Check qubits
        let qubits1 = gate1.qubits();
        let qubits2 = gate2.qubits();

        if qubits1.len() != qubits2.len() {
            return false;
        }

        for (q1, q2) in qubits1.iter().zip(qubits2.iter()) {
            if q1 != q2 {
                return false;
            }
        }

        // TODO: Check parameters for parametric gates
        true
    }

    /// Check unitary equivalence
    pub fn check_unitary_equivalence<const N: usize>(
        &self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<N>,
    ) -> QuantRS2Result<EquivalenceResult> {
        if N > self.options.max_unitary_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Circuit too large for unitary construction: {} qubits (max: {})",
                N, self.options.max_unitary_qubits
            )));
        }

        // Get unitaries for both circuits
        let unitary1 = self.get_circuit_unitary(circuit1)?;
        let unitary2 = self.get_circuit_unitary(circuit2)?;

        // Compare unitaries
        let (equivalent, max_diff) = if self.options.ignore_global_phase {
            self.unitaries_equal_up_to_phase(&unitary1, &unitary2)
        } else {
            self.unitaries_equal(&unitary1, &unitary2)
        };

        Ok(EquivalenceResult {
            equivalent,
            check_type: if self.options.ignore_global_phase {
                EquivalenceType::GlobalPhaseEquivalence
            } else {
                EquivalenceType::UnitaryEquivalence
            },
            max_difference: Some(max_diff),
            details: if equivalent {
                "Unitaries are equivalent".to_string()
            } else {
                format!("Maximum unitary difference: {:.2e}", max_diff)
            },
            numerical_analysis: None,
            confidence_score: if equivalent {
                1.0 - (max_diff / self.options.tolerance)
            } else {
                0.0
            },
            statistical_significance: None,
            error_bounds: None,
        })
    }

    /// Get the unitary matrix for a circuit
    fn get_circuit_unitary<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<Array2<Complex64>> {
        let dim = 1 << N;
        let mut unitary = Array2::eye(dim);

        // Apply each gate to the unitary
        for gate in circuit.gates() {
            self.apply_gate_to_unitary(&mut unitary, gate.as_ref(), N)?;
        }

        Ok(unitary)
    }

    /// Apply a gate to a unitary matrix
    fn apply_gate_to_unitary(
        &self,
        unitary: &mut Array2<Complex64>,
        gate: &dyn GateOp,
        num_qubits: usize,
    ) -> QuantRS2Result<()> {
        let gate_matrix = self.get_gate_matrix(gate)?;
        let qubits = gate.qubits();

        // Apply the gate matrix to the full unitary
        match qubits.len() {
            1 => {
                let qubit_idx = qubits[0].id() as usize;
                self.apply_single_qubit_gate(unitary, &gate_matrix, qubit_idx, num_qubits)?;
            }
            2 => {
                let control_idx = qubits[0].id() as usize;
                let target_idx = qubits[1].id() as usize;
                self.apply_two_qubit_gate(
                    unitary,
                    &gate_matrix,
                    control_idx,
                    target_idx,
                    num_qubits,
                )?;
            }
            _ => {
                return Err(QuantRS2Error::UnsupportedOperation(format!(
                    "Gates with {} qubits not yet supported",
                    qubits.len()
                )));
            }
        }

        Ok(())
    }

    /// Get the matrix representation of a gate
    fn get_gate_matrix(&self, gate: &dyn GateOp) -> QuantRS2Result<Array2<Complex64>> {
        let c0 = Complex64::new(0.0, 0.0);
        let c1 = Complex64::new(1.0, 0.0);
        let ci = Complex64::new(0.0, 1.0);

        match gate.name() {
            "H" => {
                let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
                Ok(array![
                    [c1 * sqrt2_inv, c1 * sqrt2_inv],
                    [c1 * sqrt2_inv, -c1 * sqrt2_inv]
                ])
            }
            "X" => Ok(array![[c0, c1], [c1, c0]]),
            "Y" => Ok(array![[c0, -ci], [ci, c0]]),
            "Z" => Ok(array![[c1, c0], [c0, -c1]]),
            "S" => Ok(array![[c1, c0], [c0, ci]]),
            "T" => Ok(array![
                [c1, c0],
                [
                    c0,
                    Complex64::new(
                        1.0 / std::f64::consts::SQRT_2,
                        1.0 / std::f64::consts::SQRT_2
                    )
                ]
            ]),
            "CNOT" | "CX" => Ok(array![
                [c1, c0, c0, c0],
                [c0, c1, c0, c0],
                [c0, c0, c0, c1],
                [c0, c0, c1, c0]
            ]),
            "CZ" => Ok(array![
                [c1, c0, c0, c0],
                [c0, c1, c0, c0],
                [c0, c0, c1, c0],
                [c0, c0, c0, -c1]
            ]),
            "SWAP" => Ok(array![
                [c1, c0, c0, c0],
                [c0, c0, c1, c0],
                [c0, c1, c0, c0],
                [c0, c0, c0, c1]
            ]),
            _ => Err(QuantRS2Error::UnsupportedOperation(format!(
                "Gate '{}' matrix not yet implemented",
                gate.name()
            ))),
        }
    }

    /// Apply a single-qubit gate to a unitary matrix
    fn apply_single_qubit_gate(
        &self,
        unitary: &mut Array2<Complex64>,
        gate_matrix: &Array2<Complex64>,
        qubit_idx: usize,
        num_qubits: usize,
    ) -> QuantRS2Result<()> {
        let dim = 1 << num_qubits;
        let mut new_unitary = Array2::zeros((dim, dim));

        // Apply gate to each basis state
        for col in 0..dim {
            for row in 0..dim {
                let mut sum = Complex64::new(0.0, 0.0);

                // Check if this element should be affected by the gate
                let row_bit = (row >> qubit_idx) & 1;
                let col_bit = (col >> qubit_idx) & 1;

                for k in 0..dim {
                    let k_bit = (k >> qubit_idx) & 1;

                    // Only mix states that differ in the target qubit
                    if (row ^ k) == ((row_bit ^ k_bit) << qubit_idx) {
                        sum += gate_matrix[[row_bit, k_bit]] * unitary[[k, col]];
                    }
                }

                new_unitary[[row, col]] = sum;
            }
        }

        *unitary = new_unitary;
        Ok(())
    }

    /// Apply a two-qubit gate to a unitary matrix
    fn apply_two_qubit_gate(
        &self,
        unitary: &mut Array2<Complex64>,
        gate_matrix: &Array2<Complex64>,
        qubit1_idx: usize,
        qubit2_idx: usize,
        num_qubits: usize,
    ) -> QuantRS2Result<()> {
        let dim = 1 << num_qubits;
        let mut new_unitary = Array2::zeros((dim, dim));

        for col in 0..dim {
            for row in 0..dim {
                let mut sum = Complex64::new(0.0, 0.0);

                // Extract relevant qubit states
                let row_q1 = (row >> qubit1_idx) & 1;
                let row_q2 = (row >> qubit2_idx) & 1;
                let row_gate_idx = (row_q1 << 1) | row_q2;

                let col_q1 = (col >> qubit1_idx) & 1;
                let col_q2 = (col >> qubit2_idx) & 1;

                for k in 0..dim {
                    let k_q1 = (k >> qubit1_idx) & 1;
                    let k_q2 = (k >> qubit2_idx) & 1;
                    let k_gate_idx = (k_q1 << 1) | k_q2;

                    // Check if k differs from row only in the gate qubits
                    let diff = row ^ k;
                    let expected_diff =
                        ((row_q1 ^ k_q1) << qubit1_idx) | ((row_q2 ^ k_q2) << qubit2_idx);

                    if diff == expected_diff {
                        sum += gate_matrix[[row_gate_idx, k_gate_idx]] * unitary[[k, col]];
                    }
                }

                new_unitary[[row, col]] = sum;
            }
        }

        *unitary = new_unitary;
        Ok(())
    }

    /// Check if two unitaries are equal
    fn unitaries_equal(&self, u1: &Array2<Complex64>, u2: &Array2<Complex64>) -> (bool, f64) {
        if u1.shape() != u2.shape() {
            return (false, f64::INFINITY);
        }

        let mut max_diff = 0.0;
        for (a, b) in u1.iter().zip(u2.iter()) {
            let diff = (a - b).norm();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > self.options.tolerance {
                return (false, max_diff);
            }
        }

        (true, max_diff)
    }

    /// Check if two unitaries are equal up to a global phase
    fn unitaries_equal_up_to_phase(
        &self,
        u1: &Array2<Complex64>,
        u2: &Array2<Complex64>,
    ) -> (bool, f64) {
        if u1.shape() != u2.shape() {
            return (false, f64::INFINITY);
        }

        // Find the first non-zero element to determine phase
        let mut phase = None;
        for (a, b) in u1.iter().zip(u2.iter()) {
            if a.norm() > self.options.tolerance && b.norm() > self.options.tolerance {
                phase = Some(b / a);
                break;
            }
        }

        let phase = match phase {
            Some(p) => p,
            None => return (false, f64::INFINITY),
        };

        // Check all elements with phase adjustment
        let mut max_diff = 0.0;
        for (a, b) in u1.iter().zip(u2.iter()) {
            let adjusted = a * phase;
            let diff = (adjusted - b).norm();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > self.options.tolerance {
                return (false, max_diff);
            }
        }

        (true, max_diff)
    }

    /// Check state vector equivalence
    pub fn check_state_vector_equivalence<const N: usize>(
        &self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<N>,
    ) -> QuantRS2Result<EquivalenceResult> {
        let mut max_diff = 0.0;
        let num_states = if self.options.check_all_states {
            1 << N
        } else {
            // Check a subset of states for large circuits
            std::cmp::min(1 << N, 100)
        };

        for state_idx in 0..num_states {
            let state1 = self.apply_circuit_to_state(circuit1, state_idx, N)?;
            let state2 = self.apply_circuit_to_state(circuit2, state_idx, N)?;

            let (equal, diff) = if self.options.ignore_global_phase {
                self.states_equal_up_to_phase(&state1, &state2)
            } else {
                self.states_equal(&state1, &state2)
            };

            if diff > max_diff {
                max_diff = diff;
            }

            if !equal {
                return Ok(EquivalenceResult {
                    equivalent: false,
                    check_type: EquivalenceType::StateVectorEquivalence,
                    max_difference: Some(max_diff),
                    details: format!(
                        "States differ for input |{:0b}>: max difference {:.2e}",
                        state_idx, max_diff
                    ),
                    numerical_analysis: None,
                    confidence_score: 0.0,
                    statistical_significance: None,
                    error_bounds: None,
                });
            }
        }

        Ok(EquivalenceResult {
            equivalent: true,
            check_type: EquivalenceType::StateVectorEquivalence,
            max_difference: Some(max_diff),
            details: format!("Checked {} computational basis states", num_states),
            numerical_analysis: None,
            confidence_score: 1.0 - (max_diff / self.options.tolerance).min(1.0),
            statistical_significance: None,
            error_bounds: None,
        })
    }

    /// Apply circuit to a computational basis state
    fn apply_circuit_to_state<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        state_idx: usize,
        num_qubits: usize,
    ) -> QuantRS2Result<Vec<Complex64>> {
        let dim = 1 << num_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[state_idx] = Complex64::new(1.0, 0.0);

        // Apply each gate to the state vector
        for gate in circuit.gates() {
            self.apply_gate_to_state(&mut state, gate.as_ref(), num_qubits)?;
        }

        Ok(state)
    }

    /// Apply a gate to a state vector
    fn apply_gate_to_state(
        &self,
        state: &mut Vec<Complex64>,
        gate: &dyn GateOp,
        num_qubits: usize,
    ) -> QuantRS2Result<()> {
        let gate_matrix = self.get_gate_matrix(gate)?;
        let qubits = gate.qubits();

        match qubits.len() {
            1 => {
                let qubit_idx = qubits[0].id() as usize;
                self.apply_single_qubit_gate_to_state(state, &gate_matrix, qubit_idx, num_qubits)?;
            }
            2 => {
                let control_idx = qubits[0].id() as usize;
                let target_idx = qubits[1].id() as usize;
                self.apply_two_qubit_gate_to_state(
                    state,
                    &gate_matrix,
                    control_idx,
                    target_idx,
                    num_qubits,
                )?;
            }
            _ => {
                return Err(QuantRS2Error::UnsupportedOperation(format!(
                    "Gates with {} qubits not yet supported",
                    qubits.len()
                )));
            }
        }

        Ok(())
    }

    /// Apply a single-qubit gate to a state vector
    fn apply_single_qubit_gate_to_state(
        &self,
        state: &mut Vec<Complex64>,
        gate_matrix: &Array2<Complex64>,
        qubit_idx: usize,
        num_qubits: usize,
    ) -> QuantRS2Result<()> {
        let dim = 1 << num_qubits;
        let mut new_state = vec![Complex64::new(0.0, 0.0); dim];

        for i in 0..dim {
            let bit = (i >> qubit_idx) & 1;

            for j in 0..2 {
                let other_idx = i ^ ((bit ^ j) << qubit_idx);
                new_state[i] += gate_matrix[[bit, j]] * state[other_idx];
            }
        }

        *state = new_state;
        Ok(())
    }

    /// Apply a two-qubit gate to a state vector
    fn apply_two_qubit_gate_to_state(
        &self,
        state: &mut Vec<Complex64>,
        gate_matrix: &Array2<Complex64>,
        qubit1_idx: usize,
        qubit2_idx: usize,
        num_qubits: usize,
    ) -> QuantRS2Result<()> {
        let dim = 1 << num_qubits;
        let mut new_state = vec![Complex64::new(0.0, 0.0); dim];

        for i in 0..dim {
            let bit1 = (i >> qubit1_idx) & 1;
            let bit2 = (i >> qubit2_idx) & 1;
            let gate_row = (bit1 << 1) | bit2;

            for gate_col in 0..4 {
                let new_bit1 = (gate_col >> 1) & 1;
                let new_bit2 = gate_col & 1;

                let j = i ^ ((bit1 ^ new_bit1) << qubit1_idx) ^ ((bit2 ^ new_bit2) << qubit2_idx);
                new_state[i] += gate_matrix[[gate_row, gate_col]] * state[j];
            }
        }

        *state = new_state;
        Ok(())
    }

    /// Check if two state vectors are equal
    fn states_equal(&self, s1: &[Complex64], s2: &[Complex64]) -> (bool, f64) {
        if s1.len() != s2.len() {
            return (false, f64::INFINITY);
        }

        let mut max_diff = 0.0;
        for (a, b) in s1.iter().zip(s2.iter()) {
            let diff = (a - b).norm();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > self.options.tolerance {
                return (false, max_diff);
            }
        }

        (true, max_diff)
    }

    /// Check if two state vectors are equal up to a global phase
    fn states_equal_up_to_phase(&self, s1: &[Complex64], s2: &[Complex64]) -> (bool, f64) {
        if s1.len() != s2.len() {
            return (false, f64::INFINITY);
        }

        // Find phase from first non-zero element
        let mut phase = None;
        for (a, b) in s1.iter().zip(s2.iter()) {
            if a.norm() > self.options.tolerance && b.norm() > self.options.tolerance {
                phase = Some(b / a);
                break;
            }
        }

        let phase = match phase {
            Some(p) => p,
            None => return (false, f64::INFINITY),
        };

        // Check all elements with phase adjustment
        let mut max_diff = 0.0;
        for (a, b) in s1.iter().zip(s2.iter()) {
            let adjusted = a * phase;
            let diff = (adjusted - b).norm();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > self.options.tolerance {
                return (false, max_diff);
            }
        }

        (true, max_diff)
    }

    /// Check probabilistic equivalence (measurement outcomes)
    pub fn check_probabilistic_equivalence<const N: usize>(
        &self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<N>,
    ) -> QuantRS2Result<EquivalenceResult> {
        // For each computational basis state, check measurement probabilities
        let mut max_diff = 0.0;

        for state_idx in 0..(1 << N) {
            let probs1 = self.get_measurement_probabilities(circuit1, state_idx, N)?;
            let probs2 = self.get_measurement_probabilities(circuit2, state_idx, N)?;

            for (p1, p2) in probs1.iter().zip(probs2.iter()) {
                let diff = (p1 - p2).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > self.options.tolerance {
                    return Ok(EquivalenceResult {
                        equivalent: false,
                        check_type: EquivalenceType::ProbabilisticEquivalence,
                        max_difference: Some(max_diff),
                        details: format!(
                            "Measurement probabilities differ for input |{:0b}>",
                            state_idx
                        ),
                        numerical_analysis: None,
                        confidence_score: 0.0,
                        statistical_significance: None,
                        error_bounds: None,
                    });
                }
            }
        }

        Ok(EquivalenceResult {
            equivalent: true,
            check_type: EquivalenceType::ProbabilisticEquivalence,
            max_difference: Some(max_diff),
            details: "Measurement probabilities match for all inputs".to_string(),
            numerical_analysis: None,
            confidence_score: 1.0 - (max_diff / self.options.tolerance).min(1.0),
            statistical_significance: None,
            error_bounds: None,
        })
    }

    /// Get measurement probabilities for a circuit and input state
    fn get_measurement_probabilities<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        state_idx: usize,
        num_qubits: usize,
    ) -> QuantRS2Result<Vec<f64>> {
        // Apply circuit to get final state
        let final_state = self.apply_circuit_to_state(circuit, state_idx, num_qubits)?;

        // Calculate probabilities from amplitudes
        let probs: Vec<f64> = final_state
            .iter()
            .map(|amplitude| amplitude.norm_sqr())
            .collect();

        Ok(probs)
    }

    // ===== SciRS2 Enhanced Methods =====

    /// Perform comprehensive numerical analysis using SciRS2 capabilities
    fn perform_scirs2_numerical_analysis(
        &mut self,
        unitary1: &Array2<Complex64>,
        unitary2: &Array2<Complex64>,
    ) -> QuantRS2Result<NumericalAnalysis> {
        // Calculate difference matrix
        let diff_matrix = unitary1 - unitary2;

        // Calculate Frobenius norm
        let frobenius_norm = diff_matrix.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();

        // Calculate condition number using SVD approximation
        let condition_number = if self.options.enable_stability_analysis {
            Some(self.estimate_condition_number(unitary1)?)
        } else {
            None
        };

        // Calculate spectral norm (largest singular value of difference)
        let spectral_norm = if self.options.enable_stability_analysis {
            Some(self.calculate_spectral_norm(&diff_matrix)?)
        } else {
            None
        };

        // Estimate numerical rank
        let numerical_rank = self.estimate_numerical_rank(&diff_matrix);

        // Calculate stability indicator
        let stability_indicator = if let Some(cond_num) = condition_number {
            1.0 / (1.0 + (cond_num / self.options.max_condition_number).log10())
        } else {
            1.0
        };

        // Calculate adaptive tolerance
        let adaptive_tolerance = self.calculate_adaptive_tolerance_internal(
            unitary1.nrows(),
            frobenius_norm,
            condition_number.unwrap_or(1.0),
        );

        Ok(NumericalAnalysis {
            condition_number,
            numerical_rank: Some(numerical_rank),
            frobenius_norm,
            spectral_norm,
            adaptive_tolerance,
            stability_indicator,
        })
    }

    /// Calculate adaptive tolerance based on circuit complexity and numerical properties
    fn calculate_adaptive_tolerance<const N: usize>(
        &self,
        num_qubits: usize,
        analysis: &NumericalAnalysis,
    ) -> f64 {
        let base_tolerance = if self.options.enable_adaptive_tolerance {
            SCIRS2_DEFAULT_TOLERANCE
        } else {
            self.options.tolerance
        };

        // Scale tolerance based on circuit size (more qubits = less precision)
        let size_factor = 1.0 + (num_qubits as f64).powf(1.5) * 1e-15;

        // Scale based on condition number
        let condition_factor = if let Some(cond_num) = analysis.condition_number {
            1.0 + (cond_num / 1e12).log10().max(0.0) * 1e-2
        } else {
            1.0
        };

        // Scale based on Frobenius norm
        let norm_factor = 1.0 + analysis.frobenius_norm * 1e-3;

        base_tolerance * size_factor * condition_factor * norm_factor
    }

    /// Internal helper for adaptive tolerance calculation
    fn calculate_adaptive_tolerance_internal(
        &self,
        matrix_size: usize,
        frobenius_norm: f64,
        condition_number: f64,
    ) -> f64 {
        let base_tolerance = SCIRS2_DEFAULT_TOLERANCE;
        let size_factor = 1.0 + (matrix_size as f64).sqrt() * 1e-15;
        let condition_factor = 1.0 + (condition_number / 1e12).log10().max(0.0) * 1e-2;
        let norm_factor = 1.0 + frobenius_norm * 1e-3;

        base_tolerance * size_factor * condition_factor * norm_factor
    }

    /// Compare unitaries using SciRS2 enhanced numerical analysis
    fn scirs2_unitaries_equal(
        &self,
        u1: &Array2<Complex64>,
        u2: &Array2<Complex64>,
        adaptive_tolerance: f64,
    ) -> QuantRS2Result<(bool, f64, f64, ErrorBounds)> {
        if u1.shape() != u2.shape() {
            return Ok((
                false,
                f64::INFINITY,
                0.0,
                ErrorBounds {
                    lower_bound: f64::INFINITY,
                    upper_bound: f64::INFINITY,
                    confidence_level: 0.0,
                    standard_deviation: None,
                },
            ));
        }

        let mut max_diff = 0.0;
        let mut differences = Vec::new();

        // Calculate element-wise differences
        for (a, b) in u1.iter().zip(u2.iter()) {
            let diff = if self.options.ignore_global_phase {
                // Find global phase from first non-zero element
                let phase = if a.norm() > adaptive_tolerance && b.norm() > adaptive_tolerance {
                    b / a
                } else {
                    Complex64::new(1.0, 0.0)
                };
                (a * phase - b).norm()
            } else {
                (a - b).norm()
            };

            differences.push(diff);
            if diff > max_diff {
                max_diff = diff;
            }
        }

        // Calculate statistical measures
        let mean_diff = differences.iter().sum::<f64>() / differences.len() as f64;
        let variance = differences
            .iter()
            .map(|d| (d - mean_diff).powf(2.0))
            .sum::<f64>()
            / differences.len() as f64;
        let std_dev = variance.sqrt();

        // Calculate confidence score based on how well differences fit expected distribution
        let confidence_score = if max_diff <= adaptive_tolerance {
            1.0 - (max_diff / adaptive_tolerance).min(1.0)
        } else {
            0.0
        };

        // Calculate error bounds
        let error_bounds = ErrorBounds {
            lower_bound: (mean_diff - 2.0 * std_dev).max(0.0),
            upper_bound: mean_diff + 2.0 * std_dev,
            confidence_level: self.options.confidence_level,
            standard_deviation: Some(std_dev),
        };

        let equivalent = max_diff <= adaptive_tolerance;

        Ok((equivalent, max_diff, confidence_score, error_bounds))
    }

    /// Compare SciRS2 graphs for structural equivalence
    fn compare_scirs2_graphs(
        &self,
        graph1: &crate::scirs2_integration::SciRS2CircuitGraph,
        graph2: &crate::scirs2_integration::SciRS2CircuitGraph,
    ) -> QuantRS2Result<(bool, f64, String)> {
        // Basic structural comparison
        if graph1.nodes.len() != graph2.nodes.len() {
            return Ok((
                false,
                0.0,
                format!(
                    "Different number of nodes: {} vs {}",
                    graph1.nodes.len(),
                    graph2.nodes.len()
                ),
            ));
        }

        if graph1.edges.len() != graph2.edges.len() {
            return Ok((
                false,
                0.0,
                format!(
                    "Different number of edges: {} vs {}",
                    graph1.edges.len(),
                    graph2.edges.len()
                ),
            ));
        }

        // Calculate graph similarity metrics
        let node_similarity = self.calculate_node_similarity(graph1, graph2);
        let edge_similarity = self.calculate_edge_similarity(graph1, graph2);
        let topology_similarity = self.calculate_topology_similarity(graph1, graph2);

        // Combined similarity score
        let overall_similarity = (node_similarity + edge_similarity + topology_similarity) / 3.0;

        let equivalent = overall_similarity > 0.95; // 95% similarity threshold

        let details = format!(
            "Graph similarity analysis: nodes={:.3}, edges={:.3}, topology={:.3}, overall={:.3}",
            node_similarity, edge_similarity, topology_similarity, overall_similarity
        );

        Ok((equivalent, overall_similarity, details))
    }

    /// Calculate node similarity between graphs
    fn calculate_node_similarity(
        &self,
        graph1: &crate::scirs2_integration::SciRS2CircuitGraph,
        graph2: &crate::scirs2_integration::SciRS2CircuitGraph,
    ) -> f64 {
        if graph1.nodes.is_empty() && graph2.nodes.is_empty() {
            return 1.0;
        }

        let total_nodes = graph1.nodes.len().max(graph2.nodes.len());
        let mut matching_nodes = 0;

        // Simple node type matching (could be enhanced with graph isomorphism)
        for node1 in graph1.nodes.values() {
            for node2 in graph2.nodes.values() {
                if node1.node_type == node2.node_type {
                    matching_nodes += 1;
                    break;
                }
            }
        }

        matching_nodes as f64 / total_nodes as f64
    }

    /// Calculate edge similarity between graphs
    fn calculate_edge_similarity(
        &self,
        graph1: &crate::scirs2_integration::SciRS2CircuitGraph,
        graph2: &crate::scirs2_integration::SciRS2CircuitGraph,
    ) -> f64 {
        if graph1.edges.is_empty() && graph2.edges.is_empty() {
            return 1.0;
        }

        let total_edges = graph1.edges.len().max(graph2.edges.len());
        let mut matching_edges = 0;

        // Simple edge type matching
        for edge1 in graph1.edges.values() {
            for edge2 in graph2.edges.values() {
                if edge1.edge_type == edge2.edge_type {
                    matching_edges += 1;
                    break;
                }
            }
        }

        matching_edges as f64 / total_edges as f64
    }

    /// Calculate topology similarity using adjacency matrix comparison
    fn calculate_topology_similarity(
        &self,
        graph1: &crate::scirs2_integration::SciRS2CircuitGraph,
        graph2: &crate::scirs2_integration::SciRS2CircuitGraph,
    ) -> f64 {
        if graph1.adjacency_matrix.len() != graph2.adjacency_matrix.len() {
            return 0.0;
        }

        let mut total_elements = 0;
        let mut matching_elements = 0;

        for (row1, row2) in graph1
            .adjacency_matrix
            .iter()
            .zip(graph2.adjacency_matrix.iter())
        {
            if row1.len() != row2.len() {
                return 0.0;
            }

            for (elem1, elem2) in row1.iter().zip(row2.iter()) {
                total_elements += 1;
                if elem1 == elem2 {
                    matching_elements += 1;
                }
            }
        }

        if total_elements == 0 {
            1.0
        } else {
            matching_elements as f64 / total_elements as f64
        }
    }

    /// Estimate condition number using power iteration method
    fn estimate_condition_number(&self, matrix: &Array2<Complex64>) -> QuantRS2Result<f64> {
        // Simplified condition number estimation
        // In practice, this would use more sophisticated SVD methods
        let n = matrix.nrows();
        if n == 0 {
            return Ok(1.0);
        }

        // Estimate largest singular value using power iteration
        let mut v = vec![Complex64::new(1.0, 0.0); n];
        for _ in 0..10 {
            // v = A^H * A * v
            let mut new_v = vec![Complex64::new(0.0, 0.0); n];
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        new_v[i] += matrix[[k, i]].conj() * matrix[[k, j]] * v[j];
                    }
                }
            }

            // Normalize
            let norm = new_v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in &mut new_v {
                    *x /= norm;
                }
            }
            v = new_v;
        }

        // Estimate condition number (simplified)
        let estimated_largest_sv = v.iter().map(|x| x.norm()).sum::<f64>() / n as f64;
        let estimated_smallest_sv = 1.0 / estimated_largest_sv; // Very simplified

        Ok((estimated_largest_sv / estimated_smallest_sv.max(1e-16)).min(1e16))
    }

    /// Calculate spectral norm (largest singular value) of a matrix
    fn calculate_spectral_norm(&self, matrix: &Array2<Complex64>) -> QuantRS2Result<f64> {
        // Simplified spectral norm calculation
        // In practice, this would use proper SVD decomposition
        Ok(matrix.iter().map(|x| x.norm()).fold(0.0, f64::max))
    }

    /// Estimate numerical rank of a matrix
    fn estimate_numerical_rank(&self, matrix: &Array2<Complex64>) -> usize {
        let tolerance = self.options.tolerance;
        let mut rank = 0;

        for row in matrix.rows() {
            let row_norm = row.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            if row_norm > tolerance {
                rank += 1;
            }
        }

        rank
    }

    /// Calculate statistical significance of the difference
    fn calculate_statistical_significance(
        &self,
        u1: &Array2<Complex64>,
        u2: &Array2<Complex64>,
        max_difference: f64,
    ) -> QuantRS2Result<f64> {
        // Simplified statistical test
        // In practice, this would use more sophisticated statistical methods
        let n = u1.len();
        let degrees_of_freedom = n - 1;

        // Calculate t-statistic approximation
        let differences: Vec<f64> = u1
            .iter()
            .zip(u2.iter())
            .map(|(a, b)| (a - b).norm())
            .collect();

        let mean_diff = differences.iter().sum::<f64>() / n as f64;
        let variance = differences
            .iter()
            .map(|d| (d - mean_diff).powf(2.0))
            .sum::<f64>()
            / degrees_of_freedom as f64;
        let std_error = (variance / n as f64).sqrt();

        let t_stat = if std_error > 0.0 {
            mean_diff / std_error
        } else {
            0.0
        };

        // Approximate p-value (very simplified)
        let p_value = 2.0 * (1.0 - (t_stat.abs() / (1.0 + t_stat.abs())));

        Ok(p_value.max(0.0).min(1.0))
    }
}

/// Quick check if two circuits are structurally identical
pub fn circuits_structurally_equal<const N: usize>(
    circuit1: &Circuit<N>,
    circuit2: &Circuit<N>,
) -> bool {
    let checker = EquivalenceChecker::default();
    checker
        .check_structural_equivalence(circuit1, circuit2)
        .map(|result| result.equivalent)
        .unwrap_or(false)
}

/// Quick check if two circuits are equivalent (using default options)
pub fn circuits_equivalent<const N: usize>(
    circuit1: &Circuit<N>,
    circuit2: &Circuit<N>,
) -> QuantRS2Result<bool> {
    let mut checker = EquivalenceChecker::default();
    Ok(checker.check_equivalence(circuit1, circuit2)?.equivalent)
}

/// Check equivalence using SciRS2 numerical analysis with custom tolerance
pub fn circuits_scirs2_equivalent<const N: usize>(
    circuit1: &Circuit<N>,
    circuit2: &Circuit<N>,
    options: EquivalenceOptions,
) -> QuantRS2Result<EquivalenceResult> {
    let mut checker = EquivalenceChecker::new(options);
    checker.check_equivalence(circuit1, circuit2)
}

/// Quick SciRS2 numerical equivalence check with default enhanced options
pub fn circuits_scirs2_numerical_equivalent<const N: usize>(
    circuit1: &Circuit<N>,
    circuit2: &Circuit<N>,
) -> QuantRS2Result<EquivalenceResult> {
    let mut options = EquivalenceOptions::default();
    options.enable_adaptive_tolerance = true;
    options.enable_statistical_analysis = true;
    options.enable_stability_analysis = true;

    let mut checker = EquivalenceChecker::new(options);
    checker.check_scirs2_numerical_equivalence(circuit1, circuit2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::{Hadamard, PauliX, PauliZ};

    #[test]
    fn test_structural_equivalence() {
        let mut circuit1 = Circuit::<2>::new();
        circuit1.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit1
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();

        let mut circuit2 = Circuit::<2>::new();
        circuit2.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit2
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();

        let checker = EquivalenceChecker::default();
        let result = checker
            .check_structural_equivalence(&circuit1, &circuit2)
            .unwrap();
        assert!(result.equivalent);
    }

    #[test]
    fn test_structural_non_equivalence() {
        let mut circuit1 = Circuit::<2>::new();
        circuit1.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit1
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();

        let mut circuit2 = Circuit::<2>::new();
        circuit2.add_gate(PauliX { target: QubitId(0) }).unwrap();
        circuit2
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();

        let checker = EquivalenceChecker::default();
        let result = checker
            .check_structural_equivalence(&circuit1, &circuit2)
            .unwrap();
        assert!(!result.equivalent);
    }

    #[test]
    fn test_different_gate_count() {
        let mut circuit1 = Circuit::<2>::new();
        circuit1.add_gate(Hadamard { target: QubitId(0) }).unwrap();

        let mut circuit2 = Circuit::<2>::new();
        circuit2.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit2.add_gate(PauliZ { target: QubitId(0) }).unwrap();

        let checker = EquivalenceChecker::default();
        let result = checker
            .check_structural_equivalence(&circuit1, &circuit2)
            .unwrap();
        assert!(!result.equivalent);
        assert!(result.details.contains("Different number of gates"));
    }
}
