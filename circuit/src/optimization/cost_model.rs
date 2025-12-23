//! Cost models for circuit optimization
//!
//! This module defines different cost models used to evaluate and optimize quantum circuits.

use crate::builder::Circuit;
use crate::optimization::gate_properties::{get_gate_properties, GateCost, GateError};
use quantrs2_core::gate::GateOp;
use std::collections::HashMap;

/// Trait for cost models
pub trait CostModel: Send + Sync {
    /// Calculate the cost of a single gate
    fn gate_cost(&self, gate: &dyn GateOp) -> f64;

    /// Calculate the total cost of a circuit (using gate list)
    fn circuit_cost_from_gates(&self, gates: &[Box<dyn GateOp>]) -> f64;

    /// Calculate the total cost of a list of gates (alias for `circuit_cost_from_gates`)
    fn gates_cost(&self, gates: &[Box<dyn GateOp>]) -> f64 {
        self.circuit_cost_from_gates(gates)
    }

    /// Get the weights used in cost calculation
    fn weights(&self) -> CostWeights;

    /// Check if a gate is native on the target hardware
    fn is_native(&self, gate: &dyn GateOp) -> bool;
}

/// Extension trait for circuit cost calculation
pub trait CircuitCostExt<const N: usize> {
    fn circuit_cost(&self, circuit: &Circuit<N>) -> f64;
}

impl<T: CostModel + ?Sized, const N: usize> CircuitCostExt<N> for T {
    fn circuit_cost(&self, circuit: &Circuit<N>) -> f64 {
        let mut total_cost = 0.0;

        // Calculate cost for each gate
        for gate in circuit.gates() {
            total_cost += self.gate_cost(gate.as_ref());
        }

        // Add depth penalty for deep circuits
        let depth = circuit.calculate_depth();
        total_cost += depth as f64 * 0.1; // Small penalty per depth unit

        // Add two-qubit gate penalty (they're expensive)
        let two_qubit_gates = circuit.count_two_qubit_gates();
        total_cost += two_qubit_gates as f64 * 5.0; // Extra cost for two-qubit gates

        total_cost
    }
}

/// Weights for different cost components
#[derive(Debug, Clone, Copy)]
pub struct CostWeights {
    pub gate_count: f64,
    pub execution_time: f64,
    pub error_rate: f64,
    pub circuit_depth: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            gate_count: 1.0,
            execution_time: 1.0,
            error_rate: 10.0,
            circuit_depth: 0.5,
        }
    }
}

/// Abstract cost model (hardware-agnostic)
pub struct AbstractCostModel {
    weights: CostWeights,
    native_gates: Vec<String>,
}

impl AbstractCostModel {
    /// Create a new abstract cost model
    #[must_use]
    pub fn new(weights: CostWeights) -> Self {
        Self {
            weights,
            native_gates: vec!["H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ", "CNOT", "CZ"]
                .into_iter()
                .map(std::string::ToString::to_string)
                .collect(),
        }
    }
}

impl Default for AbstractCostModel {
    fn default() -> Self {
        Self::new(CostWeights::default())
    }
}

impl CostModel for AbstractCostModel {
    fn gate_cost(&self, gate: &dyn GateOp) -> f64 {
        let props = get_gate_properties(gate);

        props.cost.total_cost(
            self.weights.execution_time,
            self.weights.gate_count,
            self.weights.error_rate,
        )
    }

    fn circuit_cost_from_gates(&self, gates: &[Box<dyn GateOp>]) -> f64 {
        gates.iter().map(|g| self.gate_cost(g.as_ref())).sum()
    }

    fn weights(&self) -> CostWeights {
        self.weights
    }

    fn is_native(&self, gate: &dyn GateOp) -> bool {
        self.native_gates.contains(&gate.name().to_string())
    }
}

/// Hardware-specific cost model
pub struct HardwareCostModel {
    backend_name: String,
    weights: CostWeights,
    gate_costs: HashMap<String, GateCost>,
    gate_errors: HashMap<String, GateError>,
    native_gates: Vec<String>,
}

impl HardwareCostModel {
    /// Create a cost model for a specific backend
    #[must_use]
    pub fn for_backend(backend: &str) -> Self {
        let (weights, gate_costs, gate_errors, native_gates) = match backend {
            "ibm" => Self::ibm_config(),
            "google" => Self::google_config(),
            "aws" => Self::aws_config(),
            _ => Self::default_config(),
        };

        Self {
            backend_name: backend.to_string(),
            weights,
            gate_costs,
            gate_errors,
            native_gates,
        }
    }

    fn ibm_config() -> (
        CostWeights,
        HashMap<String, GateCost>,
        HashMap<String, GateError>,
        Vec<String>,
    ) {
        let weights = CostWeights {
            gate_count: 0.5,
            execution_time: 1.5,
            error_rate: 20.0,
            circuit_depth: 1.0,
        };

        let mut gate_costs = HashMap::new();
        gate_costs.insert("X".to_string(), GateCost::new(35.0, 1, 1.0));
        gate_costs.insert("Y".to_string(), GateCost::new(35.0, 1, 1.0));
        gate_costs.insert("Z".to_string(), GateCost::new(0.0, 0, 0.0)); // Virtual Z
        gate_costs.insert("H".to_string(), GateCost::new(35.0, 1, 1.0));
        gate_costs.insert("S".to_string(), GateCost::new(35.0, 1, 1.0));
        gate_costs.insert("T".to_string(), GateCost::new(35.0, 1, 1.0));
        gate_costs.insert("RZ".to_string(), GateCost::new(0.0, 0, 0.0)); // Virtual RZ
        gate_costs.insert("CNOT".to_string(), GateCost::new(300.0, 1, 3.0));
        gate_costs.insert("CZ".to_string(), GateCost::new(300.0, 1, 3.0));

        let mut gate_errors = HashMap::new();
        gate_errors.insert("X".to_string(), GateError::new(0.99975, 0.00025, 0.00002));
        gate_errors.insert("CNOT".to_string(), GateError::new(0.9985, 0.0015, 0.0001));

        let native_gates = vec!["X", "Y", "Z", "H", "S", "T", "RZ", "CNOT", "CZ"]
            .into_iter()
            .map(std::string::ToString::to_string)
            .collect();

        (weights, gate_costs, gate_errors, native_gates)
    }

    fn google_config() -> (
        CostWeights,
        HashMap<String, GateCost>,
        HashMap<String, GateError>,
        Vec<String>,
    ) {
        let weights = CostWeights {
            gate_count: 0.8,
            execution_time: 1.0,
            error_rate: 15.0,
            circuit_depth: 0.8,
        };

        let mut gate_costs = HashMap::new();
        gate_costs.insert("X".to_string(), GateCost::new(25.0, 1, 1.0));
        gate_costs.insert("Y".to_string(), GateCost::new(25.0, 1, 1.0));
        gate_costs.insert("Z".to_string(), GateCost::new(0.0, 0, 0.0)); // Virtual
        gate_costs.insert("H".to_string(), GateCost::new(25.0, 1, 1.0));
        gate_costs.insert("RZ".to_string(), GateCost::new(0.0, 0, 0.0)); // Virtual
        gate_costs.insert("SQRT_X".to_string(), GateCost::new(25.0, 1, 1.0));
        gate_costs.insert("CZ".to_string(), GateCost::new(30.0, 1, 2.0));

        let mut gate_errors = HashMap::new();
        gate_errors.insert("X".to_string(), GateError::new(0.9998, 0.0002, 0.00001));
        gate_errors.insert("CZ".to_string(), GateError::new(0.994, 0.006, 0.0003));

        let native_gates = vec!["X", "Y", "Z", "H", "RZ", "SQRT_X", "CZ"]
            .into_iter()
            .map(std::string::ToString::to_string)
            .collect();

        (weights, gate_costs, gate_errors, native_gates)
    }

    fn aws_config() -> (
        CostWeights,
        HashMap<String, GateCost>,
        HashMap<String, GateError>,
        Vec<String>,
    ) {
        let weights = CostWeights {
            gate_count: 1.0,
            execution_time: 1.2,
            error_rate: 10.0,
            circuit_depth: 0.6,
        };

        let mut gate_costs = HashMap::new();
        gate_costs.insert("X".to_string(), GateCost::new(50.0, 1, 1.0));
        gate_costs.insert("Y".to_string(), GateCost::new(50.0, 1, 1.0));
        gate_costs.insert("Z".to_string(), GateCost::new(50.0, 1, 1.0));
        gate_costs.insert("H".to_string(), GateCost::new(50.0, 1, 1.0));
        gate_costs.insert("RX".to_string(), GateCost::new(50.0, 1, 1.2));
        gate_costs.insert("RY".to_string(), GateCost::new(50.0, 1, 1.2));
        gate_costs.insert("RZ".to_string(), GateCost::new(50.0, 1, 1.2));
        gate_costs.insert("CNOT".to_string(), GateCost::new(500.0, 1, 4.0));

        let mut gate_errors = HashMap::new();
        gate_errors.insert("X".to_string(), GateError::new(0.9997, 0.0003, 0.00002));
        gate_errors.insert("CNOT".to_string(), GateError::new(0.997, 0.003, 0.0002));

        let native_gates = vec!["X", "Y", "Z", "H", "RX", "RY", "RZ", "CNOT", "CZ"]
            .into_iter()
            .map(std::string::ToString::to_string)
            .collect();

        (weights, gate_costs, gate_errors, native_gates)
    }

    fn default_config() -> (
        CostWeights,
        HashMap<String, GateCost>,
        HashMap<String, GateError>,
        Vec<String>,
    ) {
        let weights = CostWeights::default();
        let gate_costs = HashMap::new();
        let gate_errors = HashMap::new();
        let native_gates = vec!["H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ", "CNOT", "CZ"]
            .into_iter()
            .map(std::string::ToString::to_string)
            .collect();

        (weights, gate_costs, gate_errors, native_gates)
    }
}

impl CostModel for HardwareCostModel {
    fn gate_cost(&self, gate: &dyn GateOp) -> f64 {
        let gate_name = gate.name().to_string();

        // Use hardware-specific cost if available
        if let Some(cost) = self.gate_costs.get(&gate_name) {
            let error_cost = if let Some(error) = self.gate_errors.get(&gate_name) {
                error.total_error() * self.weights.error_rate
            } else {
                0.0
            };

            cost.total_cost(
                self.weights.execution_time,
                self.weights.gate_count,
                0.0, // Use error_cost separately
            ) + error_cost
        } else {
            // Fall back to generic properties
            let props = get_gate_properties(gate);
            props.cost.total_cost(
                self.weights.execution_time,
                self.weights.gate_count,
                self.weights.error_rate,
            )
        }
    }

    fn circuit_cost_from_gates(&self, gates: &[Box<dyn GateOp>]) -> f64 {
        gates.iter().map(|g| self.gate_cost(g.as_ref())).sum()
    }

    fn weights(&self) -> CostWeights {
        self.weights
    }

    fn is_native(&self, gate: &dyn GateOp) -> bool {
        self.native_gates.contains(&gate.name().to_string())
    }
}
