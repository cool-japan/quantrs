//! Gate properties and relationships
//!
//! This module defines properties of quantum gates that are used for optimization,
//! including cost models, error rates, decomposition rules, and commutation relations.

use std::collections::HashMap;
use quantrs2_core::gate::{GateOp, single, multi};
use quantrs2_core::qubit::QubitId;
use std::any::Any;

/// Gate cost information
#[derive(Debug, Clone, Copy)]
pub struct GateCost {
    /// Time duration in nanoseconds
    pub duration_ns: f64,
    /// Gate count (e.g., 1 for native gates, >1 for decomposed gates)
    pub gate_count: u32,
    /// Energy or computational cost (arbitrary units)
    pub computational_cost: f64,
}

impl GateCost {
    /// Create a new gate cost
    pub fn new(duration_ns: f64, gate_count: u32, computational_cost: f64) -> Self {
        Self {
            duration_ns,
            gate_count,
            computational_cost,
        }
    }

    /// Total cost (weighted sum)
    pub fn total_cost(&self, time_weight: f64, count_weight: f64, comp_weight: f64) -> f64 {
        time_weight * self.duration_ns 
            + count_weight * self.gate_count as f64 
            + comp_weight * self.computational_cost
    }
}

/// Gate error information
#[derive(Debug, Clone, Copy)]
pub struct GateError {
    /// Average gate fidelity
    pub fidelity: f64,
    /// Depolarizing error rate
    pub error_rate: f64,
    /// Coherent error contribution
    pub coherent_error: f64,
}

impl GateError {
    /// Create new gate error info
    pub fn new(fidelity: f64, error_rate: f64, coherent_error: f64) -> Self {
        Self {
            fidelity,
            error_rate,
            coherent_error,
        }
    }

    /// Combined error metric
    pub fn total_error(&self) -> f64 {
        1.0 - self.fidelity + self.error_rate + self.coherent_error
    }
}

/// Decomposition rule for gates
#[derive(Debug, Clone)]
pub struct DecompositionRule {
    /// Name of the decomposition
    pub name: String,
    /// Target gate set for decomposition
    pub target_gates: Vec<String>,
    /// Cost of the decomposition
    pub cost: GateCost,
    /// Priority (lower is better)
    pub priority: u32,
}

/// Properties of a quantum gate
#[derive(Debug, Clone)]
pub struct GateProperties {
    /// Gate name
    pub name: String,
    /// Number of qubits
    pub num_qubits: usize,
    /// Is this a native gate on hardware?
    pub is_native: bool,
    /// Cost information
    pub cost: GateCost,
    /// Error information
    pub error: GateError,
    /// Available decomposition rules
    pub decompositions: Vec<DecompositionRule>,
    /// Gates that this gate commutes with
    pub commutes_with: Vec<String>,
    /// Is self-inverse (G·G = I)?
    pub is_self_inverse: bool,
    /// Is diagonal in computational basis?
    pub is_diagonal: bool,
    /// Is parameterized?
    pub is_parameterized: bool,
}

impl GateProperties {
    /// Create properties for a single-qubit gate
    pub fn single_qubit(name: &str) -> Self {
        match name {
            "H" => Self {
                name: name.to_string(),
                num_qubits: 1,
                is_native: true,
                cost: GateCost::new(20.0, 1, 1.0),
                error: GateError::new(0.9999, 0.0001, 0.00001),
                decompositions: vec![],
                commutes_with: vec![],
                is_self_inverse: true,
                is_diagonal: false,
                is_parameterized: false,
            },
            "X" | "Y" | "Z" => Self {
                name: name.to_string(),
                num_qubits: 1,
                is_native: true,
                cost: GateCost::new(20.0, 1, 1.0),
                error: GateError::new(0.9999, 0.0001, 0.00001),
                decompositions: vec![],
                commutes_with: if name == "Z" { 
                    vec!["Z".to_string(), "RZ".to_string(), "S".to_string(), "T".to_string()]
                } else { 
                    vec![] 
                },
                is_self_inverse: true,
                is_diagonal: name == "Z",
                is_parameterized: false,
            },
            "S" => Self {
                name: name.to_string(),
                num_qubits: 1,
                is_native: true,
                cost: GateCost::new(20.0, 1, 1.0),
                error: GateError::new(0.9999, 0.0001, 0.00001),
                decompositions: vec![
                    DecompositionRule {
                        name: "Z-rotation".to_string(),
                        target_gates: vec!["RZ".to_string()],
                        cost: GateCost::new(20.0, 1, 1.0),
                        priority: 1,
                    }
                ],
                commutes_with: vec!["Z".to_string(), "RZ".to_string(), "S".to_string(), "T".to_string()],
                is_self_inverse: false,
                is_diagonal: true,
                is_parameterized: false,
            },
            "T" => Self {
                name: name.to_string(),
                num_qubits: 1,
                is_native: true,
                cost: GateCost::new(20.0, 1, 1.0),
                error: GateError::new(0.9999, 0.0001, 0.00001),
                decompositions: vec![
                    DecompositionRule {
                        name: "Z-rotation".to_string(),
                        target_gates: vec!["RZ".to_string()],
                        cost: GateCost::new(20.0, 1, 1.0),
                        priority: 1,
                    }
                ],
                commutes_with: vec!["Z".to_string(), "RZ".to_string(), "S".to_string(), "T".to_string()],
                is_self_inverse: false,
                is_diagonal: true,
                is_parameterized: false,
            },
            "RX" | "RY" | "RZ" => Self {
                name: name.to_string(),
                num_qubits: 1,
                is_native: true,
                cost: GateCost::new(40.0, 1, 1.5),
                error: GateError::new(0.9998, 0.0002, 0.00002),
                decompositions: vec![],
                commutes_with: if name == "RZ" { 
                    vec!["Z".to_string(), "RZ".to_string(), "S".to_string(), "T".to_string()]
                } else { 
                    vec![] 
                },
                is_self_inverse: false,
                is_diagonal: name == "RZ",
                is_parameterized: true,
            },
            _ => Self::default_single_qubit(name),
        }
    }

    /// Create properties for a two-qubit gate
    pub fn two_qubit(name: &str) -> Self {
        match name {
            "CNOT" => Self {
                name: name.to_string(),
                num_qubits: 2,
                is_native: true,
                cost: GateCost::new(300.0, 1, 3.0),
                error: GateError::new(0.999, 0.001, 0.0001),
                decompositions: vec![],
                commutes_with: vec![],
                is_self_inverse: true,
                is_diagonal: false,
                is_parameterized: false,
            },
            "CZ" => Self {
                name: name.to_string(),
                num_qubits: 2,
                is_native: true,
                cost: GateCost::new(200.0, 1, 2.5),
                error: GateError::new(0.999, 0.001, 0.0001),
                decompositions: vec![
                    DecompositionRule {
                        name: "H-CNOT-H".to_string(),
                        target_gates: vec!["H".to_string(), "CNOT".to_string()],
                        cost: GateCost::new(340.0, 3, 5.0),
                        priority: 1,
                    }
                ],
                commutes_with: vec!["CZ".to_string()],
                is_self_inverse: true,
                is_diagonal: true,
                is_parameterized: false,
            },
            "SWAP" => Self {
                name: name.to_string(),
                num_qubits: 2,
                is_native: false,
                cost: GateCost::new(900.0, 3, 9.0),
                error: GateError::new(0.997, 0.003, 0.0003),
                decompositions: vec![
                    DecompositionRule {
                        name: "3-CNOT".to_string(),
                        target_gates: vec!["CNOT".to_string()],
                        cost: GateCost::new(900.0, 3, 9.0),
                        priority: 1,
                    }
                ],
                commutes_with: vec![],
                is_self_inverse: true,
                is_diagonal: false,
                is_parameterized: false,
            },
            _ => Self::default_two_qubit(name),
        }
    }

    /// Create properties for a multi-qubit gate
    pub fn multi_qubit(name: &str, num_qubits: usize) -> Self {
        match name {
            "Toffoli" => Self {
                name: name.to_string(),
                num_qubits: 3,
                is_native: false,
                cost: GateCost::new(2000.0, 15, 20.0),
                error: GateError::new(0.99, 0.01, 0.001),
                decompositions: vec![
                    DecompositionRule {
                        name: "Standard".to_string(),
                        target_gates: vec!["H".to_string(), "CNOT".to_string(), "T".to_string(), "T†".to_string()],
                        cost: GateCost::new(2000.0, 15, 20.0),
                        priority: 1,
                    }
                ],
                commutes_with: vec![],
                is_self_inverse: true,
                is_diagonal: false,
                is_parameterized: false,
            },
            _ => Self::default_multi_qubit(name, num_qubits),
        }
    }

    fn default_single_qubit(name: &str) -> Self {
        Self {
            name: name.to_string(),
            num_qubits: 1,
            is_native: false,
            cost: GateCost::new(50.0, 1, 2.0),
            error: GateError::new(0.999, 0.001, 0.0001),
            decompositions: vec![],
            commutes_with: vec![],
            is_self_inverse: false,
            is_diagonal: false,
            is_parameterized: false,
        }
    }

    fn default_two_qubit(name: &str) -> Self {
        Self {
            name: name.to_string(),
            num_qubits: 2,
            is_native: false,
            cost: GateCost::new(500.0, 2, 5.0),
            error: GateError::new(0.995, 0.005, 0.0005),
            decompositions: vec![],
            commutes_with: vec![],
            is_self_inverse: false,
            is_diagonal: false,
            is_parameterized: false,
        }
    }

    fn default_multi_qubit(name: &str, num_qubits: usize) -> Self {
        Self {
            name: name.to_string(),
            num_qubits,
            is_native: false,
            cost: GateCost::new(1000.0 * num_qubits as f64, num_qubits as u32 * 5, 10.0 * num_qubits as f64),
            error: GateError::new(0.99 / num_qubits as f64, 0.01 * num_qubits as f64, 0.001 * num_qubits as f64),
            decompositions: vec![],
            commutes_with: vec![],
            is_self_inverse: false,
            is_diagonal: false,
            is_parameterized: false,
        }
    }
}

/// Commutation table for gates
pub struct CommutationTable {
    table: HashMap<(String, String), bool>,
}

impl CommutationTable {
    /// Create a new commutation table
    pub fn new() -> Self {
        let mut table = HashMap::new();
        
        // Pauli commutation relations
        table.insert(("X".to_string(), "X".to_string()), true);
        table.insert(("Y".to_string(), "Y".to_string()), true);
        table.insert(("Z".to_string(), "Z".to_string()), true);
        table.insert(("X".to_string(), "Y".to_string()), false);
        table.insert(("Y".to_string(), "X".to_string()), false);
        table.insert(("X".to_string(), "Z".to_string()), false);
        table.insert(("Z".to_string(), "X".to_string()), false);
        table.insert(("Y".to_string(), "Z".to_string()), false);
        table.insert(("Z".to_string(), "Y".to_string()), false);
        
        // Diagonal gates commute
        for diag1 in &["Z", "S", "T", "RZ"] {
            for diag2 in &["Z", "S", "T", "RZ"] {
                table.insert((diag1.to_string(), diag2.to_string()), true);
            }
        }
        
        // CNOT commutation
        table.insert(("CNOT".to_string(), "CNOT".to_string()), false); // In general
        
        Self { table }
    }

    /// Check if two gates commute
    pub fn commutes(&self, gate1: &str, gate2: &str) -> bool {
        if let Some(&result) = self.table.get(&(gate1.to_string(), gate2.to_string())) {
            result
        } else if let Some(&result) = self.table.get(&(gate2.to_string(), gate1.to_string())) {
            result
        } else {
            false // Conservative default
        }
    }

    /// Check if two gate operations commute (considering qubit indices)
    pub fn gates_commute(&self, gate1: &dyn GateOp, gate2: &dyn GateOp) -> bool {
        let qubits1 = gate1.qubits();
        let qubits2 = gate2.qubits();
        
        // Gates on disjoint qubits always commute
        if qubits1.iter().all(|q| !qubits2.contains(q)) {
            return true;
        }
        
        // Check specific commutation rules
        self.commutes(gate1.name(), gate2.name())
    }
}

impl Default for CommutationTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Get properties for a gate operation
pub fn get_gate_properties(gate: &dyn GateOp) -> GateProperties {
    let num_qubits = gate.num_qubits();
    
    match num_qubits {
        1 => GateProperties::single_qubit(gate.name()),
        2 => GateProperties::two_qubit(gate.name()),
        n => GateProperties::multi_qubit(gate.name(), n),
    }
}