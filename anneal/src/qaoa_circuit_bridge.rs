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

    /// Extract QAOA parameters from a parameterized circuit
    #[must_use]
    pub fn extract_qaoa_parameters(&self, circuit: &CircuitBridgeRepresentation) -> Vec<f64> {
        let mut parameters = vec![0.0; circuit.num_parameters];

        for param_ref in &circuit.parameter_map {
            if param_ref.parameter_index < parameters.len() {
                // This is simplified - in practice, you'd extract the actual parameter
                // from the gate at gate_index
                parameters[param_ref.parameter_index] = 0.1; // Placeholder
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

    /// Optimize a QAOA circuit using circuit module optimization passes
    pub fn optimize_qaoa_circuit(
        &self,
        circuit: &CircuitBridgeRepresentation,
    ) -> BridgeResult<CircuitBridgeRepresentation> {
        // This would integrate with the circuit module's optimization passes
        // For now, return the circuit unchanged as a placeholder

        // In a full implementation, this would:
        // 1. Convert to the circuit module's format
        // 2. Apply optimization passes (gate cancellation, rotation merging, etc.)
        // 3. Convert back to our bridge representation

        Ok(circuit.clone())
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

    /// Estimate the depth reduction from circuit optimization
    #[must_use]
    pub fn estimate_optimization_benefit(
        &self,
        original_circuit: &CircuitBridgeRepresentation,
        optimized_circuit: &CircuitBridgeRepresentation,
    ) -> OptimizationMetrics {
        OptimizationMetrics {
            original_depth: original_circuit.gates.len(),
            optimized_depth: optimized_circuit.gates.len(),
            gate_count_reduction: original_circuit
                .gates
                .len()
                .saturating_sub(optimized_circuit.gates.len()),
            estimated_speedup: 1.0, // Placeholder
        }
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
}
