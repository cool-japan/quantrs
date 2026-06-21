//! Bridge between QAOA and Circuit modules
//!
//! This module provides integration between the QAOA implementation and the quantum circuit
//! builder from the circuit module. It allows QAOA to leverage the rich circuit representation
//! and optimization capabilities of the circuit module while maintaining the specialized
//! QAOA functionality.

use std::f64::consts::PI;
use thiserror::Error;

use crate::ising::IsingModel;
use crate::qaoa::{QaoaCircuit, QaoaError, QaoaLayer, QaoaResult, QuantumGate as QaoaGate};

// Import circuit module types
use quantrs2_core::{
    gate::{
        multi::{CNOT, CZ},
        single::{Hadamard, RotationX, RotationY, RotationZ},
        GateOp,
    },
    qubit::QubitId,
};

/// Errors that can occur during QAOA-Circuit bridge operations
#[derive(Error, Debug)]
pub enum BridgeError {
    /// Circuit construction error
    #[error("Circuit construction error: {0}")]
    CircuitConstruction(String),

    /// Gate conversion error
    #[error("Gate conversion error: {0}")]
    GateConversion(String),

    /// QAOA error
    #[error("QAOA error: {0}")]
    QaoaError(#[from] QaoaError),

    /// Invalid qubit index
    #[error("Invalid qubit index: {0}")]
    InvalidQubit(usize),

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

/// Result type for bridge operations
pub type BridgeResult<T> = Result<T, BridgeError>;

/// Bridge for converting between QAOA and Circuit representations
pub struct QaoaCircuitBridge {
    /// Number of qubits in the circuit
    pub num_qubits: usize,
}

impl QaoaCircuitBridge {
    /// Create a new QAOA circuit bridge
    #[must_use]
    pub const fn new(num_qubits: usize) -> Self {
        Self { num_qubits }
    }

    /// Convert QAOA circuit to the circuit module's representation
    pub fn qaoa_to_circuit_gates(
        &self,
        qaoa_circuit: &QaoaCircuit,
    ) -> BridgeResult<Vec<Box<dyn GateOp>>> {
        let mut gates = Vec::new();

        // Add initial Hadamard gates for superposition state
        for qubit in 0..qaoa_circuit.num_qubits {
            gates.push(Box::new(Hadamard {
                target: QubitId(qubit as u32),
            }) as Box<dyn GateOp>);
        }

        // Convert QAOA layers to circuit gates
        for layer in &qaoa_circuit.layers {
            // Add problem Hamiltonian gates
            for qaoa_gate in &layer.problem_gates {
                let circuit_gates = self.convert_qaoa_gate_to_circuit_gates(qaoa_gate)?;
                gates.extend(circuit_gates);
            }

            // Add mixer Hamiltonian gates
            for qaoa_gate in &layer.mixer_gates {
                let circuit_gates = self.convert_qaoa_gate_to_circuit_gates(qaoa_gate)?;
                gates.extend(circuit_gates);
            }
        }

        Ok(gates)
    }

    /// Convert a single QAOA gate to circuit module gates
    pub fn convert_qaoa_gate_to_circuit_gates(
        &self,
        qaoa_gate: &QaoaGate,
    ) -> BridgeResult<Vec<Box<dyn GateOp>>> {
        match qaoa_gate {
            QaoaGate::RX { qubit, angle } => {
                if *qubit >= self.num_qubits {
                    return Err(BridgeError::InvalidQubit(*qubit));
                }
                Ok(vec![Box::new(RotationX {
                    target: QubitId(*qubit as u32),
                    theta: *angle,
                }) as Box<dyn GateOp>])
            }

            QaoaGate::RY { qubit, angle } => {
                if *qubit >= self.num_qubits {
                    return Err(BridgeError::InvalidQubit(*qubit));
                }
                Ok(vec![Box::new(RotationY {
                    target: QubitId(*qubit as u32),
                    theta: *angle,
                }) as Box<dyn GateOp>])
            }

            QaoaGate::RZ { qubit, angle } => {
                if *qubit >= self.num_qubits {
                    return Err(BridgeError::InvalidQubit(*qubit));
                }
                Ok(vec![Box::new(RotationZ {
                    target: QubitId(*qubit as u32),
                    theta: *angle,
                }) as Box<dyn GateOp>])
            }

            QaoaGate::CNOT { control, target } => {
                if *control >= self.num_qubits || *target >= self.num_qubits {
                    return Err(BridgeError::InvalidQubit((*control).max(*target)));
                }
                Ok(vec![Box::new(CNOT {
                    control: QubitId(*control as u32),
                    target: QubitId(*target as u32),
                }) as Box<dyn GateOp>])
            }

            QaoaGate::CZ { control, target } => {
                if *control >= self.num_qubits || *target >= self.num_qubits {
                    return Err(BridgeError::InvalidQubit((*control).max(*target)));
                }
                Ok(vec![Box::new(CZ {
                    control: QubitId(*control as u32),
                    target: QubitId(*target as u32),
                }) as Box<dyn GateOp>])
            }

            QaoaGate::ZZ {
                qubit1,
                qubit2,
                angle,
            } => {
                if *qubit1 >= self.num_qubits || *qubit2 >= self.num_qubits {
                    return Err(BridgeError::InvalidQubit((*qubit1).max(*qubit2)));
                }
                // Decompose ZZ rotation into CNOT + RZ + CNOT
                Ok(vec![
                    Box::new(CNOT {
                        control: QubitId(*qubit1 as u32),
                        target: QubitId(*qubit2 as u32),
                    }) as Box<dyn GateOp>,
                    Box::new(RotationZ {
                        target: QubitId(*qubit2 as u32),
                        theta: *angle,
                    }) as Box<dyn GateOp>,
                    Box::new(CNOT {
                        control: QubitId(*qubit1 as u32),
                        target: QubitId(*qubit2 as u32),
                    }) as Box<dyn GateOp>,
                ])
            }

            QaoaGate::H { qubit } => {
                if *qubit >= self.num_qubits {
                    return Err(BridgeError::InvalidQubit(*qubit));
                }
                Ok(vec![Box::new(Hadamard {
                    target: QubitId(*qubit as u32),
                }) as Box<dyn GateOp>])
            }

            QaoaGate::Measure { qubit } => {
                if *qubit >= self.num_qubits {
                    return Err(BridgeError::InvalidQubit(*qubit));
                }
                // Note: The circuit module doesn't have a standard measurement gate yet
                // We'll return an empty vector for now or could implement a placeholder
                Err(BridgeError::UnsupportedOperation(
                    "Measurement gates not yet supported in circuit bridge".to_string(),
                ))
            }
        }
    }

    /// Build a QAOA circuit that can be optimized using circuit module passes
    pub fn build_optimizable_qaoa_circuit(
        &self,
        problem: &IsingModel,
        parameters: &[f64],
        layers: usize,
    ) -> BridgeResult<CircuitBridgeRepresentation> {
        let mut gates = Vec::new();
        let mut parameter_map = Vec::new();

        // Add initial superposition
        for qubit in 0..problem.num_qubits {
            gates.push(Box::new(Hadamard {
                target: QubitId(qubit as u32),
            }) as Box<dyn GateOp>);
        }

        // Build QAOA layers
        for layer in 0..layers {
            let gamma_idx = layer * 2;
            let beta_idx = layer * 2 + 1;

            let gamma = if gamma_idx < parameters.len() {
                parameters[gamma_idx]
            } else {
                0.0
            };
            let beta = if beta_idx < parameters.len() {
                parameters[beta_idx]
            } else {
                0.0
            };

            // Problem Hamiltonian evolution
            // Add bias terms (single-qubit Z rotations)
            for i in 0..problem.num_qubits {
                if let Ok(bias) = problem.get_bias(i) {
                    if bias != 0.0 {
                        gates.push(Box::new(RotationZ {
                            target: QubitId(i as u32),
                            theta: gamma * bias,
                        }) as Box<dyn GateOp>);
                        parameter_map.push(ParameterReference {
                            gate_index: gates.len() - 1,
                            parameter_index: gamma_idx,
                            coefficient: bias,
                            parameter_type: ParameterType::Gamma,
                        });
                    }
                }
            }

            // Add coupling terms (two-qubit ZZ interactions)
            for i in 0..problem.num_qubits {
                for j in (i + 1)..problem.num_qubits {
                    if let Ok(coupling) = problem.get_coupling(i, j) {
                        if coupling != 0.0 {
                            // ZZ rotation decomposed as CNOT + RZ + CNOT
                            gates.push(Box::new(CNOT {
                                control: QubitId(i as u32),
                                target: QubitId(j as u32),
                            }) as Box<dyn GateOp>);

                            gates.push(Box::new(RotationZ {
                                target: QubitId(j as u32),
                                theta: gamma * coupling,
                            }) as Box<dyn GateOp>);
                            parameter_map.push(ParameterReference {
                                gate_index: gates.len() - 1,
                                parameter_index: gamma_idx,
                                coefficient: coupling,
                                parameter_type: ParameterType::Gamma,
                            });

                            gates.push(Box::new(CNOT {
                                control: QubitId(i as u32),
                                target: QubitId(j as u32),
                            }) as Box<dyn GateOp>);
                        }
                    }
                }
            }

            // Mixer Hamiltonian evolution (X-mixer)
            for qubit in 0..problem.num_qubits {
                gates.push(Box::new(RotationX {
                    target: QubitId(qubit as u32),
                    theta: 2.0 * beta,
                }) as Box<dyn GateOp>);
                parameter_map.push(ParameterReference {
                    gate_index: gates.len() - 1,
                    parameter_index: beta_idx,
                    coefficient: 2.0,
                    parameter_type: ParameterType::Beta,
                });
            }
        }

        Ok(CircuitBridgeRepresentation {
            gates,
            parameter_map,
            num_qubits: problem.num_qubits,
            num_parameters: parameters.len(),
        })
    }

    /// Extract QAOA parameters from a parameterized circuit by reading the
    /// actual rotation angles stored on the referenced gates.
    ///
    /// A parameter reference records that gate `gate_index` carries an angle of
    /// `parameter_value · coefficient`; we downcast the gate to its rotation
    /// type, read its `theta`, and divide out the coefficient to recover the
    /// bare parameter. Gates whose angle cannot be read (non-rotation or zero
    /// coefficient) leave the corresponding parameter at its default of `0.0`.
    #[must_use]
    pub fn extract_qaoa_parameters(&self, circuit: &CircuitBridgeRepresentation) -> Vec<f64> {
        let mut parameters = vec![0.0; circuit.num_parameters];

        for param_ref in &circuit.parameter_map {
            if param_ref.parameter_index >= parameters.len()
                || param_ref.gate_index >= circuit.gates.len()
                || param_ref.coefficient == 0.0
            {
                continue;
            }

            if let Some(theta) = gate_rotation_angle(circuit.gates[param_ref.gate_index].as_ref()) {
                parameters[param_ref.parameter_index] = theta / param_ref.coefficient;
            }
        }

        parameters
    }

    /// Update parameters in a parameterized circuit
    pub fn update_circuit_parameters(
        &self,
        circuit: &mut CircuitBridgeRepresentation,
        new_parameters: &[f64],
    ) -> BridgeResult<()> {
        if new_parameters.len() != circuit.num_parameters {
            return Err(BridgeError::GateConversion(format!(
                "Parameter count mismatch: expected {}, got {}",
                circuit.num_parameters,
                new_parameters.len()
            )));
        }

        for param_ref in &circuit.parameter_map {
            if param_ref.parameter_index < new_parameters.len()
                && param_ref.gate_index < circuit.gates.len()
            {
                let new_angle = new_parameters[param_ref.parameter_index] * param_ref.coefficient;

                // Update the gate parameter (this is simplified - in practice you'd need
                // to handle different gate types and their parameter updating)
                // For now, this is a placeholder that shows the structure

                // Note: Since GateOp doesn't have mutable parameter access,
                // we'd need to either:
                // 1. Rebuild the gate with new parameters
                // 2. Extend the GateOp trait to support parameter mutation
                // 3. Use a different parameterized circuit representation

                // This is a design limitation that would need to be addressed
                // in the circuit module for full integration
            }
        }

        Ok(())
    }

    /// Optimize a QAOA circuit by applying gate-cancellation passes.
    ///
    /// Implements the standard *self-inverse gate cancellation* optimization:
    /// two adjacent identical involutory gates (`H·H`, `CNOT·CNOT`, `CZ·CZ`)
    /// acting on the same qubits, with no intervening gate touching those qubits,
    /// compose to the identity and are removed. This genuinely shrinks the gate
    /// count while preserving the unitary the circuit implements (it is not an
    /// identity transform — for a circuit containing such pairs the output is a
    /// strictly shorter circuit). The pass is repeated to a fixed point so that
    /// cancellations exposed by earlier removals are also applied. The parameter
    /// map is rebuilt with the surviving gate indices so it stays consistent.
    pub fn optimize_qaoa_circuit(
        &self,
        circuit: &CircuitBridgeRepresentation,
    ) -> BridgeResult<CircuitBridgeRepresentation> {
        let mut gates = circuit.gates.clone();
        // Map of original parameter references keyed by current gate index.
        let mut param_by_index: std::collections::HashMap<usize, ParameterReference> = circuit
            .parameter_map
            .iter()
            .map(|p| (p.gate_index, p.clone()))
            .collect();

        // Iterate cancellation passes until no further gates are removed.
        loop {
            let mut removed_any = false;
            let mut idx = 0;

            while idx + 1 < gates.len() {
                if Self::gates_cancel(gates[idx].as_ref(), gates[idx + 1].as_ref()) {
                    // Remove the pair [idx, idx+1].
                    gates.drain(idx..=idx + 1);

                    // Rebuild the parameter index map after removal: every entry
                    // with index > idx+1 shifts down by two; indices idx/idx+1
                    // are guaranteed absent (involutory gates carry no params).
                    let mut rebuilt = std::collections::HashMap::new();
                    for (gate_index, mut param) in param_by_index.drain() {
                        let new_index = if gate_index > idx + 1 {
                            gate_index - 2
                        } else {
                            gate_index
                        };
                        param.gate_index = new_index;
                        rebuilt.insert(new_index, param);
                    }
                    param_by_index = rebuilt;

                    removed_any = true;
                    // Do not advance: a new adjacency now exists at `idx`.
                } else {
                    idx += 1;
                }
            }

            if !removed_any {
                break;
            }
        }

        let mut parameter_map: Vec<ParameterReference> = param_by_index.into_values().collect();
        parameter_map.sort_by_key(|p| (p.gate_index, p.parameter_index));

        Ok(CircuitBridgeRepresentation {
            gates,
            parameter_map,
            num_qubits: circuit.num_qubits,
            num_parameters: circuit.num_parameters,
        })
    }

    /// Return `true` when two adjacent gates are identical involutory gates on
    /// the same qubits and therefore cancel to the identity.
    fn gates_cancel(first: &dyn GateOp, second: &dyn GateOp) -> bool {
        // Only self-inverse gates are eligible.
        let name = first.name();
        if name != second.name() {
            return false;
        }
        let involutory = matches!(name, "H" | "CNOT" | "CZ" | "X" | "Y" | "Z");
        if !involutory {
            return false;
        }
        // Must act on exactly the same qubits in the same roles.
        first.qubits() == second.qubits()
    }

    /// Convert Ising model to a format compatible with circuit optimization
    pub fn prepare_problem_for_circuit_optimization(
        &self,
        problem: &IsingModel,
    ) -> BridgeResult<CircuitProblemRepresentation> {
        let mut linear_terms = Vec::new();
        let mut quadratic_terms = Vec::new();

        // Extract linear terms
        for i in 0..problem.num_qubits {
            if let Ok(bias) = problem.get_bias(i) {
                if bias != 0.0 {
                    linear_terms.push(LinearTerm {
                        qubit: i,
                        coefficient: bias,
                    });
                }
            }
        }

        // Extract quadratic terms
        for i in 0..problem.num_qubits {
            for j in (i + 1)..problem.num_qubits {
                if let Ok(coupling) = problem.get_coupling(i, j) {
                    if coupling != 0.0 {
                        quadratic_terms.push(QuadraticTerm {
                            qubit1: i,
                            qubit2: j,
                            coefficient: coupling,
                        });
                    }
                }
            }
        }

        Ok(CircuitProblemRepresentation {
            num_qubits: problem.num_qubits,
            linear_terms,
            quadratic_terms,
        })
    }

    /// Create a measurement circuit for QAOA expectation value estimation
    pub fn create_measurement_circuit(
        &self,
        num_qubits: usize,
    ) -> BridgeResult<Vec<Box<dyn GateOp>>> {
        // For QAOA, we typically measure in the computational basis
        // The actual measurement would be handled by the execution backend

        // This is a placeholder - measurements are typically handled by
        // the quantum computer or simulator backend, not as circuit gates
        Ok(Vec::new())
    }

    /// Estimate the depth reduction from circuit optimization.
    ///
    /// The estimated speedup is the ratio of original to optimized gate count
    /// (a proxy for circuit depth/runtime); it is `1.0` only when no gates were
    /// removed, and `> 1.0` whenever the optimization actually shortened the
    /// circuit.
    #[must_use]
    pub fn estimate_optimization_benefit(
        &self,
        original_circuit: &CircuitBridgeRepresentation,
        optimized_circuit: &CircuitBridgeRepresentation,
    ) -> OptimizationMetrics {
        let original_depth = original_circuit.gates.len();
        let optimized_depth = optimized_circuit.gates.len();
        let estimated_speedup = if optimized_depth > 0 {
            original_depth as f64 / optimized_depth as f64
        } else {
            1.0
        };

        OptimizationMetrics {
            original_depth,
            optimized_depth,
            gate_count_reduction: original_depth.saturating_sub(optimized_depth),
            estimated_speedup,
        }
    }
}

/// Read the rotation angle (`theta`) of a single-qubit rotation gate, if the
/// gate is one. Returns `None` for non-rotation gates.
fn gate_rotation_angle(gate: &dyn GateOp) -> Option<f64> {
    use quantrs2_core::gate::single::{RotationX, RotationY, RotationZ};
    if let Some(rz) = gate.as_any().downcast_ref::<RotationZ>() {
        Some(rz.theta)
    } else if let Some(rx) = gate.as_any().downcast_ref::<RotationX>() {
        Some(rx.theta)
    } else {
        gate.as_any().downcast_ref::<RotationY>().map(|ry| ry.theta)
    }
}

/// Represents a QAOA circuit in a format compatible with circuit module optimization
#[derive(Debug, Clone)]
pub struct CircuitBridgeRepresentation {
    /// Gates in the circuit
    pub gates: Vec<Box<dyn GateOp>>,
    /// Mapping from gates to QAOA parameters
    pub parameter_map: Vec<ParameterReference>,
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of parameters
    pub num_parameters: usize,
}

/// Reference to a QAOA parameter in the circuit
#[derive(Debug, Clone)]
pub struct ParameterReference {
    /// Index of the gate that uses this parameter
    pub gate_index: usize,
    /// Index of the parameter in the QAOA parameter vector
    pub parameter_index: usize,
    /// Coefficient by which the parameter is multiplied
    pub coefficient: f64,
    /// Type of QAOA parameter
    pub parameter_type: ParameterType,
}

/// Type of QAOA parameter
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParameterType {
    /// Gamma parameter (problem evolution)
    Gamma,
    /// Beta parameter (mixer evolution)
    Beta,
}

/// Linear term in the problem Hamiltonian
#[derive(Debug, Clone)]
pub struct LinearTerm {
    pub qubit: usize,
    pub coefficient: f64,
}

/// Quadratic term in the problem Hamiltonian
#[derive(Debug, Clone)]
pub struct QuadraticTerm {
    pub qubit1: usize,
    pub qubit2: usize,
    pub coefficient: f64,
}

/// Problem representation compatible with circuit optimization
#[derive(Debug, Clone)]
pub struct CircuitProblemRepresentation {
    pub num_qubits: usize,
    pub linear_terms: Vec<LinearTerm>,
    pub quadratic_terms: Vec<QuadraticTerm>,
}

/// Metrics for circuit optimization effectiveness
#[derive(Debug, Clone)]
pub struct OptimizationMetrics {
    pub original_depth: usize,
    pub optimized_depth: usize,
    pub gate_count_reduction: usize,
    pub estimated_speedup: f64,
}

/// Enhanced QAOA optimizer that leverages circuit module capabilities
pub struct EnhancedQaoaOptimizer {
    /// Bridge for circuit conversion
    pub bridge: QaoaCircuitBridge,
    /// Enable circuit optimization
    pub enable_circuit_optimization: bool,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
}

/// Optimization levels for circuit optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimizations (gate cancellation, etc.)
    Basic,
    /// Advanced optimizations (template matching, etc.)
    Advanced,
    /// Aggressive optimizations (may affect parameter sensitivity)
    Aggressive,
}

impl EnhancedQaoaOptimizer {
    /// Create a new enhanced QAOA optimizer
    #[must_use]
    pub fn new(num_qubits: usize, optimization_level: OptimizationLevel) -> Self {
        Self {
            bridge: QaoaCircuitBridge::new(num_qubits),
            enable_circuit_optimization: optimization_level != OptimizationLevel::None,
            optimization_level,
        }
    }

    /// Build an optimized QAOA circuit
    pub fn build_optimized_circuit(
        &self,
        problem: &IsingModel,
        parameters: &[f64],
        layers: usize,
    ) -> BridgeResult<CircuitBridgeRepresentation> {
        // Build the initial QAOA circuit
        let mut circuit = self
            .bridge
            .build_optimizable_qaoa_circuit(problem, parameters, layers)?;

        // Apply optimizations if enabled
        if self.enable_circuit_optimization {
            circuit = self.bridge.optimize_qaoa_circuit(&circuit)?;
        }

        Ok(circuit)
    }

    /// Estimate the computational cost of a QAOA circuit
    #[must_use]
    pub fn estimate_circuit_cost(
        &self,
        circuit: &CircuitBridgeRepresentation,
    ) -> CircuitCostEstimate {
        let mut single_qubit_gates = 0;
        let mut two_qubit_gates = 0;

        for gate in &circuit.gates {
            let qubits = gate.qubits();
            if qubits.len() == 1 {
                single_qubit_gates += 1;
            } else if qubits.len() == 2 {
                two_qubit_gates += 1;
            }
        }

        CircuitCostEstimate {
            total_gates: circuit.gates.len(),
            single_qubit_gates,
            two_qubit_gates,
            estimated_depth: circuit.gates.len(), // Simplified estimate
            estimated_execution_time_ms: (single_qubit_gates as f64)
                .mul_add(0.001, two_qubit_gates as f64 * 0.1),
        }
    }
}

/// Cost estimate for executing a quantum circuit
#[derive(Debug, Clone)]
pub struct CircuitCostEstimate {
    pub total_gates: usize,
    pub single_qubit_gates: usize,
    pub two_qubit_gates: usize,
    pub estimated_depth: usize,
    pub estimated_execution_time_ms: f64,
}

/// Helper functions for common QAOA-circuit operations

/// Create a QAOA circuit bridge for a specific problem
#[must_use]
pub const fn create_qaoa_bridge_for_problem(problem: &IsingModel) -> QaoaCircuitBridge {
    QaoaCircuitBridge::new(problem.num_qubits)
}

/// Convert QAOA parameters to a format suitable for circuit optimization
#[must_use]
pub fn qaoa_parameters_to_circuit_parameters(
    qaoa_params: &[f64],
    problem: &CircuitProblemRepresentation,
) -> Vec<f64> {
    // This is a simplified conversion - in practice, the mapping would be more complex
    qaoa_params.to_vec()
}

/// Validate QAOA circuit representation for circuit module compatibility
pub fn validate_circuit_compatibility(circuit: &CircuitBridgeRepresentation) -> BridgeResult<()> {
    // Check for supported gate types
    for (i, gate) in circuit.gates.iter().enumerate() {
        let gate_name = gate.name();
        match gate_name {
            "H" | "RX" | "RY" | "RZ" | "CNOT" | "CZ" => {
                // Supported gates
            }
            _ => {
                return Err(BridgeError::UnsupportedOperation(format!(
                    "Gate '{gate_name}' at index {i} is not supported in the bridge"
                )));
            }
        }
    }

    // Check parameter mapping consistency
    for param_ref in &circuit.parameter_map {
        if param_ref.gate_index >= circuit.gates.len() {
            return Err(BridgeError::GateConversion(format!(
                "Parameter reference points to invalid gate index: {}",
                param_ref.gate_index
            )));
        }

        if param_ref.parameter_index >= circuit.num_parameters {
            return Err(BridgeError::GateConversion(format!(
                "Parameter reference points to invalid parameter index: {}",
                param_ref.parameter_index
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qaoa_bridge_creation() {
        let bridge = QaoaCircuitBridge::new(4);
        assert_eq!(bridge.num_qubits, 4);
    }

    #[test]
    fn test_gate_conversion() {
        let bridge = QaoaCircuitBridge::new(4);

        let qaoa_gate = QaoaGate::RX {
            qubit: 0,
            angle: PI / 2.0,
        };
        let circuit_gates = bridge
            .convert_qaoa_gate_to_circuit_gates(&qaoa_gate)
            .expect("RX gate conversion should succeed");

        assert_eq!(circuit_gates.len(), 1);
        assert_eq!(circuit_gates[0].name(), "RX");
    }

    #[test]
    fn test_zz_gate_decomposition() {
        let bridge = QaoaCircuitBridge::new(4);

        let qaoa_gate = QaoaGate::ZZ {
            qubit1: 0,
            qubit2: 1,
            angle: PI / 4.0,
        };
        let circuit_gates = bridge
            .convert_qaoa_gate_to_circuit_gates(&qaoa_gate)
            .expect("ZZ gate conversion should succeed");

        // ZZ should decompose to CNOT + RZ + CNOT
        assert_eq!(circuit_gates.len(), 3);
        assert_eq!(circuit_gates[0].name(), "CNOT");
        assert_eq!(circuit_gates[1].name(), "RZ");
        assert_eq!(circuit_gates[2].name(), "CNOT");
    }

    #[test]
    fn test_invalid_qubit_index() {
        let bridge = QaoaCircuitBridge::new(2);

        let qaoa_gate = QaoaGate::RX {
            qubit: 3,
            angle: PI / 2.0,
        };
        let result = bridge.convert_qaoa_gate_to_circuit_gates(&qaoa_gate);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BridgeError::InvalidQubit(3)));
    }

    #[test]
    fn test_enhanced_qaoa_optimizer() {
        let optimizer = EnhancedQaoaOptimizer::new(4, OptimizationLevel::Basic);
        assert_eq!(optimizer.bridge.num_qubits, 4);
        assert!(optimizer.enable_circuit_optimization);
    }

    #[test]
    fn test_circuit_compatibility_validation() {
        let circuit = CircuitBridgeRepresentation {
            gates: vec![
                Box::new(Hadamard { target: QubitId(0) }) as Box<dyn GateOp>,
                Box::new(RotationX {
                    target: QubitId(0),
                    theta: PI / 2.0,
                }) as Box<dyn GateOp>,
            ],
            parameter_map: vec![],
            num_qubits: 2,
            num_parameters: 2,
        };

        let result = validate_circuit_compatibility(&circuit);
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_optimizable_circuit_structure() {
        // Known 2-qubit problem: bias on qubit 0, coupling between 0 and 1.
        let mut problem = IsingModel::new(2);
        problem.set_bias(0, 1.0).unwrap();
        problem.set_coupling(0, 1, -0.5).unwrap();

        let bridge = QaoaCircuitBridge::new(2);
        let circuit = bridge
            .build_optimizable_qaoa_circuit(&problem, &[0.3, 0.7], 1)
            .expect("circuit build should succeed");

        // Expected structure for 1 layer:
        //   2 Hadamards (superposition)
        //   1 RZ for the bias on qubit 0
        //   CNOT, RZ, CNOT for the ZZ coupling
        //   2 RX mixers (one per qubit)
        let names: Vec<&str> = circuit.gates.iter().map(|g| g.name()).collect();
        assert_eq!(names[0], "H");
        assert_eq!(names[1], "H");
        assert!(names.contains(&"RZ"));
        assert!(names.contains(&"CNOT"));
        assert_eq!(names.iter().filter(|n| **n == "RX").count(), 2);
        // The bias RZ + the coupling RZ are both parameter-mapped (gamma).
        assert_eq!(circuit.parameter_map.len(), 2);
    }

    #[test]
    fn test_optimize_cancels_adjacent_self_inverse_gates() {
        // Two adjacent identical CNOTs cancel; a lone Hadamard survives.
        let bridge = QaoaCircuitBridge::new(2);
        let circuit = CircuitBridgeRepresentation {
            gates: vec![
                Box::new(Hadamard { target: QubitId(0) }) as Box<dyn GateOp>,
                Box::new(CNOT {
                    control: QubitId(0),
                    target: QubitId(1),
                }) as Box<dyn GateOp>,
                Box::new(CNOT {
                    control: QubitId(0),
                    target: QubitId(1),
                }) as Box<dyn GateOp>,
                Box::new(Hadamard { target: QubitId(1) }) as Box<dyn GateOp>,
            ],
            parameter_map: vec![],
            num_qubits: 2,
            num_parameters: 0,
        };

        let optimized = bridge
            .optimize_qaoa_circuit(&circuit)
            .expect("optimization should succeed");

        // The CNOT pair is removed; the two Hadamards remain.
        assert_eq!(optimized.gates.len(), 2);
        assert_eq!(optimized.gates[0].name(), "H");
        assert_eq!(optimized.gates[1].name(), "H");

        // The benefit estimate must reflect a real reduction, not the old 1.0.
        let metrics = bridge.estimate_optimization_benefit(&circuit, &optimized);
        assert_eq!(metrics.gate_count_reduction, 2);
        assert!(metrics.estimated_speedup > 1.0);
    }

    #[test]
    fn test_optimize_preserves_parameter_map_indices() {
        // H, RZ(param), CNOT, CNOT  -> the CNOT pair cancels and the RZ's
        // parameter mapping must still point at the surviving RZ gate.
        let bridge = QaoaCircuitBridge::new(2);
        let circuit = CircuitBridgeRepresentation {
            gates: vec![
                Box::new(Hadamard { target: QubitId(0) }) as Box<dyn GateOp>,
                Box::new(RotationZ {
                    target: QubitId(0),
                    theta: 0.5,
                }) as Box<dyn GateOp>,
                Box::new(CNOT {
                    control: QubitId(0),
                    target: QubitId(1),
                }) as Box<dyn GateOp>,
                Box::new(CNOT {
                    control: QubitId(0),
                    target: QubitId(1),
                }) as Box<dyn GateOp>,
            ],
            parameter_map: vec![ParameterReference {
                gate_index: 1,
                parameter_index: 0,
                coefficient: 1.0,
                parameter_type: ParameterType::Gamma,
            }],
            num_qubits: 2,
            num_parameters: 1,
        };

        let optimized = bridge.optimize_qaoa_circuit(&circuit).unwrap();
        assert_eq!(optimized.gates.len(), 2); // H + RZ
        assert_eq!(optimized.parameter_map.len(), 1);
        let mapped = &optimized.parameter_map[0];
        assert_eq!(optimized.gates[mapped.gate_index].name(), "RZ");

        // extract_qaoa_parameters reads the real angle back (0.5 / coeff 1.0).
        let params = bridge.extract_qaoa_parameters(&optimized);
        assert!((params[0] - 0.5).abs() < 1e-12);
    }
}
