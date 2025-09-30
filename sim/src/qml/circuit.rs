//! Parameterized quantum circuits for machine learning applications.
//!
//! This module provides parameterized quantum circuit structures with
//! hardware-aware optimizations for different quantum computing architectures.

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

use super::config::HardwareArchitecture;
use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGateType};

/// Parameterized quantum circuit for machine learning
#[derive(Debug, Clone)]
pub struct ParameterizedQuantumCircuit {
    /// Circuit structure
    pub circuit: InterfaceCircuit,
    /// Parameter vector
    pub parameters: Array1<f64>,
    /// Parameter names for identification
    pub parameter_names: Vec<String>,
    /// Gate-to-parameter mapping
    pub gate_parameter_map: HashMap<usize, Vec<usize>>,
    /// Hardware-specific optimizations
    pub hardware_optimizations: HardwareOptimizations,
}

/// Hardware-specific optimizations
#[derive(Debug, Clone)]
pub struct HardwareOptimizations {
    /// Connectivity graph
    pub connectivity_graph: Array2<bool>,
    /// Gate fidelities
    pub gate_fidelities: HashMap<String, f64>,
    /// Decoherence times
    pub decoherence_times: Array1<f64>,
    /// Gate times
    pub gate_times: HashMap<String, f64>,
    /// Crosstalk matrix
    pub crosstalk_matrix: Array2<f64>,
}

impl ParameterizedQuantumCircuit {
    /// Create a new parameterized quantum circuit
    pub fn new(
        circuit: InterfaceCircuit,
        parameters: Array1<f64>,
        parameter_names: Vec<String>,
        hardware_architecture: HardwareArchitecture,
    ) -> Self {
        let num_qubits = circuit.num_qubits;
        let hardware_optimizations =
            HardwareOptimizations::for_hardware(hardware_architecture, num_qubits);

        Self {
            circuit,
            parameters,
            parameter_names,
            gate_parameter_map: HashMap::new(),
            hardware_optimizations,
        }
    }

    /// Update circuit parameters
    pub fn update_parameters(&mut self, new_parameters: Array1<f64>) -> Result<(), String> {
        if new_parameters.len() != self.parameters.len() {
            return Err(format!(
                "Parameter count mismatch: expected {}, got {}",
                self.parameters.len(),
                new_parameters.len()
            ));
        }
        self.parameters = new_parameters;
        Ok(())
    }

    /// Get parameter at specific index
    pub fn get_parameter(&self, index: usize) -> Option<f64> {
        self.parameters.get(index).copied()
    }

    /// Set parameter at specific index
    pub fn set_parameter(&mut self, index: usize, value: f64) -> Result<(), String> {
        if index >= self.parameters.len() {
            return Err(format!("Parameter index {} out of bounds", index));
        }
        self.parameters[index] = value;
        Ok(())
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.circuit.num_qubits
    }

    /// Get circuit depth
    pub fn depth(&self) -> usize {
        self.circuit.gates.len()
    }

    /// Add parameter mapping for a gate
    pub fn add_parameter_mapping(&mut self, gate_index: usize, parameter_indices: Vec<usize>) {
        self.gate_parameter_map
            .insert(gate_index, parameter_indices);
    }

    /// Get parameter mapping for a gate
    pub fn get_parameter_mapping(&self, gate_index: usize) -> Option<&Vec<usize>> {
        self.gate_parameter_map.get(&gate_index)
    }

    /// Estimate circuit fidelity based on hardware optimizations
    pub fn estimate_fidelity(&self) -> f64 {
        let mut total_fidelity = 1.0;

        for gate in &self.circuit.gates {
            let gate_name = Self::gate_type_to_string(&gate.gate_type);
            if let Some(&fidelity) = self.hardware_optimizations.gate_fidelities.get(&gate_name) {
                total_fidelity *= fidelity;
            } else {
                // Default fidelity for unknown gates
                total_fidelity *= 0.99;
            }
        }

        total_fidelity
    }

    /// Estimate total execution time
    pub fn estimate_execution_time(&self) -> f64 {
        let mut total_time = 0.0;

        for gate in &self.circuit.gates {
            let gate_name = Self::gate_type_to_string(&gate.gate_type);
            if let Some(&time) = self.hardware_optimizations.gate_times.get(&gate_name) {
                total_time += time;
            } else {
                // Default time for unknown gates
                total_time += 1e-6;
            }
        }

        total_time
    }

    /// Check if two qubits are connected according to hardware topology
    pub fn are_qubits_connected(&self, qubit1: usize, qubit2: usize) -> bool {
        if qubit1 >= self.num_qubits() || qubit2 >= self.num_qubits() {
            return false;
        }
        self.hardware_optimizations.connectivity_graph[[qubit1, qubit2]]
    }

    /// Get decoherence time for a specific qubit
    pub fn get_decoherence_time(&self, qubit: usize) -> Option<f64> {
        self.hardware_optimizations
            .decoherence_times
            .get(qubit)
            .copied()
    }

    /// Clone the circuit with new parameters
    pub fn with_parameters(&self, parameters: Array1<f64>) -> Result<Self, String> {
        let mut new_circuit = self.clone();
        new_circuit.update_parameters(parameters)?;
        Ok(new_circuit)
    }

    /// Convert InterfaceGateType to string
    fn gate_type_to_string(gate_type: &InterfaceGateType) -> String {
        match gate_type {
            InterfaceGateType::Identity => "I".to_string(),
            InterfaceGateType::PauliX | InterfaceGateType::X => "X".to_string(),
            InterfaceGateType::PauliY => "Y".to_string(),
            InterfaceGateType::PauliZ => "Z".to_string(),
            InterfaceGateType::Hadamard | InterfaceGateType::H => "H".to_string(),
            InterfaceGateType::S => "S".to_string(),
            InterfaceGateType::T => "T".to_string(),
            InterfaceGateType::Phase(_) => "Phase".to_string(),
            InterfaceGateType::RX(_) => "RX".to_string(),
            InterfaceGateType::RY(_) => "RY".to_string(),
            InterfaceGateType::RZ(_) => "RZ".to_string(),
            InterfaceGateType::U1(_) => "U1".to_string(),
            InterfaceGateType::U2(_, _) => "U2".to_string(),
            InterfaceGateType::U3(_, _, _) => "U3".to_string(),
            InterfaceGateType::CNOT => "CNOT".to_string(),
            InterfaceGateType::CZ => "CZ".to_string(),
            InterfaceGateType::CY => "CY".to_string(),
            InterfaceGateType::SWAP => "SWAP".to_string(),
            InterfaceGateType::ISwap => "ISwap".to_string(),
            InterfaceGateType::CRX(_) => "CRX".to_string(),
            InterfaceGateType::CRY(_) => "CRY".to_string(),
            InterfaceGateType::CRZ(_) => "CRZ".to_string(),
            InterfaceGateType::CPhase(_) => "CPhase".to_string(),
            InterfaceGateType::Toffoli => "Toffoli".to_string(),
            InterfaceGateType::Fredkin => "Fredkin".to_string(),
            InterfaceGateType::MultiControlledX(_) => "MCX".to_string(),
            InterfaceGateType::MultiControlledZ(_) => "MCZ".to_string(),
            InterfaceGateType::Custom(name, _) => name.clone(),
            InterfaceGateType::Measure => "Measure".to_string(),
            InterfaceGateType::Reset => "Reset".to_string(),
        }
    }
}

impl HardwareOptimizations {
    /// Create optimizations for specific hardware
    pub fn for_hardware(architecture: HardwareArchitecture, num_qubits: usize) -> Self {
        let connectivity_graph = match architecture {
            HardwareArchitecture::Superconducting => {
                // Linear connectivity typical of superconducting systems
                let mut graph = Array2::from_elem((num_qubits, num_qubits), false);
                for i in 0..num_qubits.saturating_sub(1) {
                    graph[[i, i + 1]] = true;
                    graph[[i + 1, i]] = true;
                }
                graph
            }
            HardwareArchitecture::TrappedIon => {
                // All-to-all connectivity for trapped ions
                Array2::from_elem((num_qubits, num_qubits), true)
            }
            HardwareArchitecture::Photonic => {
                // Limited connectivity for photonic systems
                let mut graph = Array2::from_elem((num_qubits, num_qubits), false);
                for i in 0..num_qubits {
                    for j in 0..num_qubits {
                        if (i as i32 - j as i32).abs() <= 2 {
                            graph[[i, j]] = true;
                        }
                    }
                }
                graph
            }
            _ => Array2::from_elem((num_qubits, num_qubits), true),
        };

        let gate_fidelities = match architecture {
            HardwareArchitecture::Superconducting => {
                let mut fidelities = HashMap::new();
                fidelities.insert("X".to_string(), 0.999);
                fidelities.insert("Y".to_string(), 0.999);
                fidelities.insert("Z".to_string(), 0.9999);
                fidelities.insert("H".to_string(), 0.998);
                fidelities.insert("CNOT".to_string(), 0.995);
                fidelities.insert("CZ".to_string(), 0.996);
                fidelities
            }
            HardwareArchitecture::TrappedIon => {
                let mut fidelities = HashMap::new();
                fidelities.insert("X".to_string(), 0.9999);
                fidelities.insert("Y".to_string(), 0.9999);
                fidelities.insert("Z".to_string(), 0.99999);
                fidelities.insert("H".to_string(), 0.9999);
                fidelities.insert("CNOT".to_string(), 0.999);
                fidelities.insert("CZ".to_string(), 0.999);
                fidelities
            }
            _ => {
                let mut fidelities = HashMap::new();
                fidelities.insert("X".to_string(), 0.99);
                fidelities.insert("Y".to_string(), 0.99);
                fidelities.insert("Z".to_string(), 0.999);
                fidelities.insert("H".to_string(), 0.99);
                fidelities.insert("CNOT".to_string(), 0.98);
                fidelities.insert("CZ".to_string(), 0.98);
                fidelities
            }
        };

        let decoherence_times = match architecture {
            HardwareArchitecture::Superconducting => {
                Array1::from_vec(vec![50e-6; num_qubits]) // 50 microseconds
            }
            HardwareArchitecture::TrappedIon => {
                Array1::from_vec(vec![100e-3; num_qubits]) // 100 milliseconds
            }
            _ => Array1::from_vec(vec![10e-6; num_qubits]),
        };

        let gate_times = match architecture {
            HardwareArchitecture::Superconducting => {
                let mut times = HashMap::new();
                times.insert("X".to_string(), 20e-9);
                times.insert("Y".to_string(), 20e-9);
                times.insert("Z".to_string(), 0.0);
                times.insert("H".to_string(), 20e-9);
                times.insert("CNOT".to_string(), 40e-9);
                times.insert("CZ".to_string(), 40e-9);
                times
            }
            HardwareArchitecture::TrappedIon => {
                let mut times = HashMap::new();
                times.insert("X".to_string(), 10e-6);
                times.insert("Y".to_string(), 10e-6);
                times.insert("Z".to_string(), 0.0);
                times.insert("H".to_string(), 10e-6);
                times.insert("CNOT".to_string(), 100e-6);
                times.insert("CZ".to_string(), 100e-6);
                times
            }
            _ => {
                let mut times = HashMap::new();
                times.insert("X".to_string(), 1e-6);
                times.insert("Y".to_string(), 1e-6);
                times.insert("Z".to_string(), 0.0);
                times.insert("H".to_string(), 1e-6);
                times.insert("CNOT".to_string(), 10e-6);
                times.insert("CZ".to_string(), 10e-6);
                times
            }
        };

        let crosstalk_matrix = Array2::zeros((num_qubits, num_qubits));

        Self {
            connectivity_graph,
            gate_fidelities,
            decoherence_times,
            gate_times,
            crosstalk_matrix,
        }
    }

    /// Update gate fidelity for a specific gate
    pub fn set_gate_fidelity(&mut self, gate_name: &str, fidelity: f64) {
        self.gate_fidelities.insert(gate_name.to_string(), fidelity);
    }

    /// Update gate time for a specific gate
    pub fn set_gate_time(&mut self, gate_name: &str, time: f64) {
        self.gate_times.insert(gate_name.to_string(), time);
    }

    /// Update decoherence time for a specific qubit
    pub fn set_decoherence_time(&mut self, qubit: usize, time: f64) {
        if qubit < self.decoherence_times.len() {
            self.decoherence_times[qubit] = time;
        }
    }

    /// Set connectivity between two qubits
    pub fn set_connectivity(&mut self, qubit1: usize, qubit2: usize, connected: bool) {
        if qubit1 < self.connectivity_graph.nrows() && qubit2 < self.connectivity_graph.ncols() {
            self.connectivity_graph[[qubit1, qubit2]] = connected;
            self.connectivity_graph[[qubit2, qubit1]] = connected; // Symmetric
        }
    }

    /// Get average gate fidelity
    pub fn average_gate_fidelity(&self) -> f64 {
        let fidelities: Vec<f64> = self.gate_fidelities.values().cloned().collect();
        if fidelities.is_empty() {
            1.0
        } else {
            fidelities.iter().sum::<f64>() / fidelities.len() as f64
        }
    }

    /// Get connectivity degree (number of connections) for a qubit
    pub fn connectivity_degree(&self, qubit: usize) -> usize {
        if qubit >= self.connectivity_graph.nrows() {
            return 0;
        }
        self.connectivity_graph
            .row(qubit)
            .iter()
            .map(|&x| if x { 1 } else { 0 })
            .sum()
    }
}
