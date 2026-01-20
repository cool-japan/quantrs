//! Parametric circuit definitions and execution for VQA
//!
//! This module provides parametric quantum circuits commonly used
//! in variational quantum algorithms.

use crate::DeviceResult;
use quantrs2_core::qubit::QubitId;
use scirs2_core::random::prelude::*;
use std::collections::HashMap;

/// Parametric circuit configuration
#[derive(Debug, Clone)]
pub struct ParametricCircuitConfig {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub depth: usize,
    /// Ansatz type
    pub ansatz: AnsatzType,
    /// Parameter mapping
    pub parameter_map: HashMap<String, usize>,
}

/// Available ansatz types
#[derive(Debug, Clone)]
pub enum AnsatzType {
    /// Hardware efficient ansatz
    HardwareEfficient,
    /// QAOA ansatz
    QAOA,
    /// Real amplitudes ansatz
    RealAmplitudes,
    /// Custom ansatz
    Custom(String),
}

impl Default for ParametricCircuitConfig {
    fn default() -> Self {
        Self {
            num_qubits: 4,
            depth: 3,
            ansatz: AnsatzType::HardwareEfficient,
            parameter_map: HashMap::new(),
        }
    }
}

/// Parametric circuit representation
#[derive(Debug, Clone)]
pub struct ParametricCircuit {
    /// Configuration
    pub config: ParametricCircuitConfig,
    /// Current parameters
    pub parameters: Vec<f64>,
    /// Parameter bounds
    pub bounds: Vec<(f64, f64)>,
    /// Circuit structure metadata
    pub structure: CircuitStructure,
}

/// Circuit structure metadata
#[derive(Debug, Clone)]
pub struct CircuitStructure {
    /// Gate sequence description
    pub gates: Vec<ParametricGate>,
    /// Qubit connectivity required
    pub connectivity: Vec<(usize, usize)>,
    /// Circuit depth estimate
    pub estimated_depth: usize,
}

/// Parametric gate representation
#[derive(Debug, Clone)]
pub struct ParametricGate {
    /// Gate type
    pub gate_type: GateType,
    /// Qubits involved
    pub qubits: Vec<usize>,
    /// Parameter indices
    pub parameter_indices: Vec<usize>,
    /// Gate metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Gate types for VQA circuits
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateType {
    /// Rotation X gate
    RX,
    /// Rotation Y gate
    RY,
    /// Rotation Z gate
    RZ,
    /// CNOT gate
    CNOT,
    /// CZ gate
    CZ,
    /// Hadamard gate
    H,
    /// Pauli X gate
    X,
    /// Pauli Y gate
    Y,
    /// Pauli Z gate
    Z,
    /// Custom parameterized gate
    Custom(String),
}

impl ParametricCircuit {
    /// Create new parametric circuit
    pub fn new(config: ParametricCircuitConfig) -> Self {
        let (num_params, structure) = Self::generate_circuit_structure(&config);

        Self {
            config,
            parameters: vec![0.0; num_params],
            bounds: vec![(-std::f64::consts::PI, std::f64::consts::PI); num_params],
            structure,
        }
    }

    /// Generate circuit structure based on ansatz type
    fn generate_circuit_structure(config: &ParametricCircuitConfig) -> (usize, CircuitStructure) {
        match &config.ansatz {
            AnsatzType::HardwareEfficient => Self::generate_hardware_efficient(config),
            AnsatzType::QAOA => Self::generate_qaoa(config),
            AnsatzType::RealAmplitudes => Self::generate_real_amplitudes(config),
            AnsatzType::Custom(name) => Self::generate_custom(config, name),
        }
    }

    /// Generate hardware-efficient ansatz
    fn generate_hardware_efficient(config: &ParametricCircuitConfig) -> (usize, CircuitStructure) {
        let mut gates = Vec::new();
        let mut connectivity = Vec::new();
        let mut param_count = 0;

        for layer in 0..config.depth {
            // Single-qubit rotations
            for qubit in 0..config.num_qubits {
                // RY rotation for each qubit
                gates.push(ParametricGate {
                    gate_type: GateType::RY,
                    qubits: vec![qubit],
                    parameter_indices: vec![param_count],
                    metadata: std::iter::once(("layer".to_string(), layer.to_string())).collect(),
                });
                param_count += 1;

                // RZ rotation for each qubit
                gates.push(ParametricGate {
                    gate_type: GateType::RZ,
                    qubits: vec![qubit],
                    parameter_indices: vec![param_count],
                    metadata: std::iter::once(("layer".to_string(), layer.to_string())).collect(),
                });
                param_count += 1;
            }

            // Entangling layer with CNOT gates
            for qubit in 0..config.num_qubits {
                let target = (qubit + 1) % config.num_qubits;

                gates.push(ParametricGate {
                    gate_type: GateType::CNOT,
                    qubits: vec![qubit, target],
                    parameter_indices: vec![], // Non-parametric
                    metadata: std::iter::once(("layer".to_string(), layer.to_string())).collect(),
                });

                connectivity.push((qubit, target));
            }
        }

        let structure = CircuitStructure {
            gates,
            connectivity,
            estimated_depth: config.depth * 3, // RY + RZ + CNOT per layer
        };

        (param_count, structure)
    }

    /// Generate QAOA ansatz
    fn generate_qaoa(config: &ParametricCircuitConfig) -> (usize, CircuitStructure) {
        let mut gates = Vec::new();
        let mut connectivity = Vec::new();
        let param_count = 2 * config.depth; // 2 parameters per layer (γ, β)

        // Initial superposition layer
        for qubit in 0..config.num_qubits {
            gates.push(ParametricGate {
                gate_type: GateType::H,
                qubits: vec![qubit],
                parameter_indices: vec![],
                metadata: std::iter::once(("layer".to_string(), "initial".to_string())).collect(),
            });
        }

        // QAOA layers
        for layer in 0..config.depth {
            let gamma_idx = layer * 2;
            let beta_idx = layer * 2 + 1;

            // Problem Hamiltonian layer (ZZ interactions)
            for qubit in 0..config.num_qubits {
                let neighbor = (qubit + 1) % config.num_qubits;

                // ZZ interaction implemented as CNOT-RZ-CNOT
                gates.push(ParametricGate {
                    gate_type: GateType::CNOT,
                    qubits: vec![qubit, neighbor],
                    parameter_indices: vec![],
                    metadata: [
                        ("layer".to_string(), layer.to_string()),
                        ("type".to_string(), "problem".to_string()),
                    ]
                    .into_iter()
                    .collect(),
                });

                gates.push(ParametricGate {
                    gate_type: GateType::RZ,
                    qubits: vec![neighbor],
                    parameter_indices: vec![gamma_idx],
                    metadata: [
                        ("layer".to_string(), layer.to_string()),
                        ("type".to_string(), "problem".to_string()),
                    ]
                    .into_iter()
                    .collect(),
                });

                gates.push(ParametricGate {
                    gate_type: GateType::CNOT,
                    qubits: vec![qubit, neighbor],
                    parameter_indices: vec![],
                    metadata: [
                        ("layer".to_string(), layer.to_string()),
                        ("type".to_string(), "problem".to_string()),
                    ]
                    .into_iter()
                    .collect(),
                });

                connectivity.push((qubit, neighbor));
            }

            // Mixer Hamiltonian layer (X rotations)
            for qubit in 0..config.num_qubits {
                gates.push(ParametricGate {
                    gate_type: GateType::RX,
                    qubits: vec![qubit],
                    parameter_indices: vec![beta_idx],
                    metadata: [
                        ("layer".to_string(), layer.to_string()),
                        ("type".to_string(), "mixer".to_string()),
                    ]
                    .into_iter()
                    .collect(),
                });
            }
        }

        let structure = CircuitStructure {
            gates,
            connectivity,
            estimated_depth: config.num_qubits
                + config.depth * (3 * config.num_qubits + config.num_qubits), // H + (CNOT-RZ-CNOT + RX) per layer
        };

        (param_count, structure)
    }

    /// Generate real amplitudes ansatz
    fn generate_real_amplitudes(config: &ParametricCircuitConfig) -> (usize, CircuitStructure) {
        let mut gates = Vec::new();
        let mut connectivity = Vec::new();
        let mut param_count = 0;

        for layer in 0..config.depth {
            // RY rotations only (for real amplitudes)
            for qubit in 0..config.num_qubits {
                gates.push(ParametricGate {
                    gate_type: GateType::RY,
                    qubits: vec![qubit],
                    parameter_indices: vec![param_count],
                    metadata: std::iter::once(("layer".to_string(), layer.to_string())).collect(),
                });
                param_count += 1;
            }

            // Linear entangling layer
            for qubit in 0..(config.num_qubits - 1) {
                gates.push(ParametricGate {
                    gate_type: GateType::CNOT,
                    qubits: vec![qubit, qubit + 1],
                    parameter_indices: vec![],
                    metadata: std::iter::once(("layer".to_string(), layer.to_string())).collect(),
                });
                connectivity.push((qubit, qubit + 1));
            }
        }

        // Final RY layer
        for qubit in 0..config.num_qubits {
            gates.push(ParametricGate {
                gate_type: GateType::RY,
                qubits: vec![qubit],
                parameter_indices: vec![param_count],
                metadata: std::iter::once(("layer".to_string(), "final".to_string())).collect(),
            });
            param_count += 1;
        }

        let structure = CircuitStructure {
            gates,
            connectivity,
            estimated_depth: config.depth * 2 + 1, // RY + CNOT per layer + final RY
        };

        (param_count, structure)
    }

    /// Generate custom ansatz
    const fn generate_custom(
        _config: &ParametricCircuitConfig,
        _name: &str,
    ) -> (usize, CircuitStructure) {
        // Placeholder for custom ansatz - would be implemented based on specific requirements
        let structure = CircuitStructure {
            gates: Vec::new(),
            connectivity: Vec::new(),
            estimated_depth: 1,
        };
        (0, structure)
    }

    /// Update circuit parameters
    pub fn set_parameters(&mut self, params: Vec<f64>) -> DeviceResult<()> {
        if params.len() != self.parameters.len() {
            return Err(crate::DeviceError::InvalidInput(format!(
                "Parameter count mismatch: expected {}, got {}",
                self.parameters.len(),
                params.len()
            )));
        }
        self.parameters = params;
        Ok(())
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    /// Get circuit depth estimate
    pub const fn circuit_depth(&self) -> usize {
        self.structure.estimated_depth
    }

    /// Get required connectivity
    pub fn required_connectivity(&self) -> &[(usize, usize)] {
        &self.structure.connectivity
    }

    /// Get gate sequence
    pub fn gates(&self) -> &[ParametricGate] {
        &self.structure.gates
    }

    /// Generate random initial parameters
    pub fn random_parameters(&self) -> Vec<f64> {
        use scirs2_core::random::prelude::*;
        let mut rng = thread_rng();

        self.bounds
            .iter()
            .map(|(min, max)| rng.gen_range(*min..*max))
            .collect()
    }

    /// Set parameter bounds
    pub fn set_bounds(&mut self, bounds: Vec<(f64, f64)>) -> DeviceResult<()> {
        if bounds.len() != self.parameters.len() {
            return Err(crate::DeviceError::InvalidInput(
                "Bounds count mismatch".to_string(),
            ));
        }
        self.bounds = bounds;
        Ok(())
    }

    /// Validate parameters are within bounds
    pub fn validate_parameters(&self) -> DeviceResult<()> {
        for (i, (&param, &(min, max))) in self.parameters.iter().zip(self.bounds.iter()).enumerate()
        {
            if param < min || param > max {
                return Err(crate::DeviceError::InvalidInput(format!(
                    "Parameter {i} ({param}) out of bounds [{min}, {max}]"
                )));
            }
        }
        Ok(())
    }
}
