//! Quantum Hardware Constraints
//!
//! This module defines quantum hardware constraints and topology configurations.

/// Quantum resource constraints
#[derive(Debug, Clone)]
pub struct QuantumConstraints {
    /// Available qubits
    pub available_qubits: usize,

    /// Maximum circuit depth
    pub max_circuit_depth: usize,

    /// Gate set constraints
    pub gate_set: Vec<String>,

    /// Coherence time constraints
    pub coherence_time: f64,

    /// Error rate constraints
    pub max_error_rate: f64,

    /// Hardware topology
    pub topology: QuantumTopology,
}

/// Quantum hardware topology
#[derive(Debug, Clone)]
pub enum QuantumTopology {
    FullyConnected,
    Linear,
    Grid { rows: usize, cols: usize },
    HeavyHex,
    Custom { connectivity: Vec<(usize, usize)> },
}

impl Default for QuantumConstraints {
    fn default() -> Self {
        Self {
            available_qubits: 16,
            max_circuit_depth: 20,
            gate_set: vec![
                "X".to_string(),
                "Y".to_string(),
                "Z".to_string(),
                "H".to_string(),
                "RX".to_string(),
                "RY".to_string(),
                "RZ".to_string(),
                "CNOT".to_string(),
            ],
            coherence_time: 100.0, // microseconds
            max_error_rate: 0.01,
            topology: QuantumTopology::FullyConnected,
        }
    }
}

impl QuantumConstraints {
    /// IBM quantum computer constraints
    pub fn ibm_quantum() -> Self {
        Self {
            available_qubits: 27,
            max_circuit_depth: 1000,
            gate_set: vec![
                "I".to_string(),
                "X".to_string(),
                "SX".to_string(),
                "RZ".to_string(),
                "CNOT".to_string(),
            ],
            coherence_time: 150.0,
            max_error_rate: 0.005,
            topology: QuantumTopology::HeavyHex,
        }
    }

    /// Google quantum computer constraints
    pub fn google_quantum() -> Self {
        Self {
            available_qubits: 70,
            max_circuit_depth: 40,
            gate_set: vec![
                "X".to_string(),
                "Y".to_string(),
                "Z".to_string(),
                "PhasedXPow".to_string(),
                "XPow".to_string(),
                "YPow".to_string(),
                "ZPow".to_string(),
                "CZ".to_string(),
            ],
            coherence_time: 80.0,
            max_error_rate: 0.002,
            topology: QuantumTopology::Grid { rows: 9, cols: 8 },
        }
    }

    /// Trapped ion quantum computer constraints
    pub fn trapped_ion() -> Self {
        Self {
            available_qubits: 32,
            max_circuit_depth: 100,
            gate_set: vec![
                "RX".to_string(),
                "RY".to_string(),
                "RZ".to_string(),
                "XX".to_string(),
                "MS".to_string(), // Mølmer-Sørensen gate
            ],
            coherence_time: 1000.0, // much longer coherence
            max_error_rate: 0.001,
            topology: QuantumTopology::FullyConnected,
        }
    }

    /// Photonic quantum computer constraints
    pub fn photonic() -> Self {
        Self {
            available_qubits: 216,
            max_circuit_depth: 12,
            gate_set: vec![
                "X".to_string(),
                "Z".to_string(),
                "S".to_string(),
                "H".to_string(),
                "CNOT".to_string(),
                "BS".to_string(), // Beam splitter
                "PS".to_string(), // Phase shifter
            ],
            coherence_time: f64::INFINITY, // photons don't decohere
            max_error_rate: 0.05,          // higher gate errors
            topology: QuantumTopology::Grid { rows: 12, cols: 18 },
        }
    }

    /// Simulator constraints (unrestricted)
    pub fn simulator() -> Self {
        Self {
            available_qubits: 64,
            max_circuit_depth: 1000,
            gate_set: vec![
                "I".to_string(),
                "X".to_string(),
                "Y".to_string(),
                "Z".to_string(),
                "H".to_string(),
                "S".to_string(),
                "T".to_string(),
                "RX".to_string(),
                "RY".to_string(),
                "RZ".to_string(),
                "CNOT".to_string(),
                "CZ".to_string(),
                "SWAP".to_string(),
                "CCX".to_string(), // Toffoli
            ],
            coherence_time: f64::INFINITY,
            max_error_rate: 0.0,
            topology: QuantumTopology::FullyConnected,
        }
    }

    /// Production constraints (realistic but conservative)
    pub fn production() -> Self {
        Self {
            available_qubits: 20,
            max_circuit_depth: 30,
            gate_set: vec![
                "X".to_string(),
                "Y".to_string(),
                "Z".to_string(),
                "H".to_string(),
                "RX".to_string(),
                "RY".to_string(),
                "RZ".to_string(),
                "CNOT".to_string(),
                "CZ".to_string(),
            ],
            coherence_time: 120.0,
            max_error_rate: 0.003,
            topology: QuantumTopology::Linear,
        }
    }

    /// NISQ era constraints
    pub fn nisq() -> Self {
        Self {
            available_qubits: 50,
            max_circuit_depth: 50,
            gate_set: vec![
                "X".to_string(),
                "Y".to_string(),
                "Z".to_string(),
                "H".to_string(),
                "RX".to_string(),
                "RY".to_string(),
                "RZ".to_string(),
                "CNOT".to_string(),
            ],
            coherence_time: 100.0,
            max_error_rate: 0.01,
            topology: QuantumTopology::Grid { rows: 7, cols: 7 },
        }
    }
}

impl QuantumTopology {
    /// Get the connectivity graph for this topology
    pub fn get_connectivity(&self, num_qubits: usize) -> Vec<(usize, usize)> {
        match self {
            QuantumTopology::FullyConnected => {
                let mut connections = Vec::new();
                for i in 0..num_qubits {
                    for j in (i + 1)..num_qubits {
                        connections.push((i, j));
                    }
                }
                connections
            }
            QuantumTopology::Linear => {
                let mut connections = Vec::new();
                for i in 0..(num_qubits - 1) {
                    connections.push((i, i + 1));
                }
                connections
            }
            QuantumTopology::Grid { rows, cols } => {
                let mut connections = Vec::new();
                for row in 0..*rows {
                    for col in 0..*cols {
                        let qubit = row * cols + col;
                        if qubit >= num_qubits {
                            break;
                        }

                        // Horizontal connections
                        if col + 1 < *cols {
                            let neighbor = row * cols + col + 1;
                            if neighbor < num_qubits {
                                connections.push((qubit, neighbor));
                            }
                        }

                        // Vertical connections
                        if row + 1 < *rows {
                            let neighbor = (row + 1) * cols + col;
                            if neighbor < num_qubits {
                                connections.push((qubit, neighbor));
                            }
                        }
                    }
                }
                connections
            }
            QuantumTopology::HeavyHex => {
                // Simplified heavy-hex connectivity for IBM quantum computers
                let mut connections = Vec::new();

                // This is a simplified version - real heavy-hex is more complex
                for i in 0..num_qubits {
                    if i % 3 == 0 && i + 1 < num_qubits {
                        connections.push((i, i + 1));
                    }
                    if i % 3 == 1 && i + 1 < num_qubits {
                        connections.push((i, i + 1));
                    }
                    if i + 3 < num_qubits {
                        connections.push((i, i + 3));
                    }
                }

                connections
            }
            QuantumTopology::Custom { connectivity } => connectivity.clone(),
        }
    }

    /// Check if two qubits are connected in this topology
    pub fn are_connected(&self, qubit1: usize, qubit2: usize, num_qubits: usize) -> bool {
        let connections = self.get_connectivity(num_qubits);
        connections
            .iter()
            .any(|&(a, b)| (a == qubit1 && b == qubit2) || (a == qubit2 && b == qubit1))
    }
}
