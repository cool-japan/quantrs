//! Routing pass implementation and result types

use crate::builder::Circuit;
use quantrs2_core::{error::QuantRS2Result, gate::GateOp};
use std::collections::HashMap;

/// Result of a routing operation
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// Total number of SWAP gates inserted
    pub total_swaps: usize,
    /// Final circuit depth after routing
    pub circuit_depth: usize,
    /// Routing overhead as a fraction
    pub routing_overhead: f64,
}

impl RoutingResult {
    /// Calculate total cost (combination of swaps and depth)
    pub fn total_cost(&self) -> f64 {
        self.total_swaps as f64 + self.circuit_depth as f64 * 0.1
    }
}

/// A routed circuit with mapping information
#[derive(Debug)]
pub struct RoutedCircuit<const N: usize> {
    /// The routed gates in execution order
    pub gates: Vec<Box<dyn GateOp>>,
    /// Final mapping from logical to physical qubits
    pub logical_to_physical: HashMap<usize, usize>,
    /// Routing statistics
    pub result: RoutingResult,
}

impl<const N: usize> RoutedCircuit<N> {
    /// Create a new routed circuit
    pub fn new(
        gates: Vec<Box<dyn GateOp>>,
        logical_to_physical: HashMap<usize, usize>,
        result: RoutingResult,
    ) -> Self {
        Self {
            gates,
            logical_to_physical,
            result,
        }
    }

    /// Get total cost of the routed circuit
    pub fn total_cost(&self) -> f64 {
        self.result.total_cost()
    }

    /// Get the number of gates in the routed circuit
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    /// Get the number of SWAP gates
    pub fn num_swaps(&self) -> usize {
        self.gates.iter().filter(|g| g.name() == "SWAP").count()
    }

    /// Get the routing overhead (SWAPs / total gates)
    pub fn routing_overhead(&self) -> f64 {
        if self.gates.is_empty() {
            0.0
        } else {
            self.num_swaps() as f64 / self.gates.len() as f64
        }
    }

    /// Get gates by type
    pub fn gates_by_type(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for gate in &self.gates {
            *counts.entry(gate.name().to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Convert back to a Circuit (if possible)
    pub fn to_circuit(&self) -> QuantRS2Result<Circuit<N>> {
        let mut circuit = Circuit::<N>::new();

        // Note: This is a simplified conversion - in practice we'd need proper gate conversion
        // For now, we'll just return an empty circuit as this is mainly for demonstration
        // TODO: Implement proper gate conversion from boxed gates back to circuit

        Ok(circuit)
    }

    /// Get the final qubit mapping
    pub fn get_mapping(&self) -> &HashMap<usize, usize> {
        &self.logical_to_physical
    }

    /// Get the inverse mapping (physical to logical)
    pub fn get_inverse_mapping(&self) -> HashMap<usize, usize> {
        self.logical_to_physical
            .iter()
            .map(|(&logical, &physical)| (physical, logical))
            .collect()
    }

    /// Calculate circuit statistics
    pub fn statistics(&self) -> RoutingStatistics {
        let mut two_qubit_gates = 0;
        let mut single_qubit_gates = 0;
        let mut swap_gates = 0;

        for gate in &self.gates {
            match gate.qubits().len() {
                1 => single_qubit_gates += 1,
                2 => {
                    if gate.name() == "SWAP" {
                        swap_gates += 1;
                    } else {
                        two_qubit_gates += 1;
                    }
                }
                _ => {}
            }
        }

        RoutingStatistics {
            total_gates: self.gates.len(),
            single_qubit_gates,
            two_qubit_gates,
            swap_gates,
            circuit_depth: self.result.circuit_depth,
            routing_overhead: self.routing_overhead(),
        }
    }
}

/// Detailed statistics about a routed circuit
#[derive(Debug, Clone)]
pub struct RoutingStatistics {
    pub total_gates: usize,
    pub single_qubit_gates: usize,
    pub two_qubit_gates: usize,
    pub swap_gates: usize,
    pub circuit_depth: usize,
    pub routing_overhead: f64,
}

impl RoutingStatistics {
    /// Calculate the improvement ratio compared to another statistic
    pub fn improvement_ratio(&self, other: &Self) -> f64 {
        if other.total_gates == 0 {
            return 0.0;
        }

        (other.total_gates as f64 - self.total_gates as f64) / other.total_gates as f64
    }

    /// Calculate SWAP efficiency (two-qubit gates / total gates)
    pub fn swap_efficiency(&self) -> f64 {
        if self.total_gates == 0 {
            0.0
        } else {
            self.two_qubit_gates as f64 / self.total_gates as f64
        }
    }
}

/// Routing pass enum for type-safe routing
#[derive(Debug, Clone)]
pub enum RoutingPassType {
    Sabre {
        coupling_map: super::CouplingMap,
        config: super::SabreConfig,
    },
    Lookahead {
        coupling_map: super::CouplingMap,
        config: super::LookaheadConfig,
    },
}

impl RoutingPassType {
    /// Get the name of the routing pass
    pub fn name(&self) -> &str {
        match self {
            Self::Sabre { .. } => "SABRE",
            Self::Lookahead { .. } => "Lookahead",
        }
    }

    /// Apply routing to a circuit
    pub fn route<const N: usize>(&self, circuit: &Circuit<N>) -> QuantRS2Result<RoutedCircuit<N>> {
        match self {
            Self::Sabre {
                coupling_map,
                config,
            } => {
                let router = super::SabreRouter::new(coupling_map.clone(), config.clone());
                router.route(circuit)
            }
            Self::Lookahead {
                coupling_map,
                config,
            } => {
                let router = super::LookaheadRouter::new(coupling_map.clone(), config.clone());
                router.route(circuit)
            }
        }
    }

    /// Check if this pass should be applied
    pub fn should_apply<const N: usize>(&self, _circuit: &Circuit<N>) -> bool {
        true
    }

    /// Get pass configuration as string (for debugging)
    pub fn config_string(&self) -> String {
        match self {
            Self::Sabre { config, .. } => {
                format!(
                    "SABRE(depth={}, max_iter={}, stochastic={})",
                    config.lookahead_depth, config.max_iterations, config.stochastic
                )
            }
            Self::Lookahead { config, .. } => {
                format!(
                    "Lookahead(depth={}, candidates={})",
                    config.lookahead_depth, config.max_swap_candidates
                )
            }
        }
    }
}

/// Routing pass manager for handling multiple routing strategies
pub struct RoutingPassManager {
    passes: Vec<RoutingPassType>,
}

impl RoutingPassManager {
    /// Create a new routing pass manager
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add a routing pass
    pub fn add_pass(&mut self, pass: RoutingPassType) {
        self.passes.push(pass);
    }

    /// Apply the best routing pass to a circuit
    pub fn route_with_best_pass<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<RoutedCircuit<N>> {
        if self.passes.is_empty() {
            return Err(quantrs2_core::error::QuantRS2Error::RoutingError(
                "No routing passes configured".to_string(),
            ));
        }

        let mut best_result = None;
        let mut best_cost = f64::INFINITY;

        for pass in &self.passes {
            if pass.should_apply(circuit) {
                match pass.route(circuit) {
                    Ok(result) => {
                        let cost = result.result.total_cost();
                        if cost < best_cost {
                            best_cost = cost;
                            best_result = Some(result);
                        }
                    }
                    Err(e) => {
                        eprintln!("Routing pass {} failed: {}", pass.name(), e);
                    }
                }
            }
        }

        best_result.ok_or_else(|| {
            quantrs2_core::error::QuantRS2Error::RoutingError(
                "All routing passes failed".to_string(),
            )
        })
    }

    /// Apply a specific routing pass by name
    pub fn route_with_pass<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        pass_name: &str,
    ) -> QuantRS2Result<RoutedCircuit<N>> {
        for pass in &self.passes {
            if pass.name() == pass_name {
                return pass.route(circuit);
            }
        }

        Err(quantrs2_core::error::QuantRS2Error::RoutingError(format!(
            "Routing pass '{}' not found",
            pass_name
        )))
    }

    /// Get available pass names
    pub fn available_passes(&self) -> Vec<&str> {
        self.passes.iter().map(|p| p.name()).collect()
    }
}

impl Default for RoutingPassManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::{
        multi::{CNOT, SWAP},
        single::Hadamard,
    };
    use quantrs2_core::qubit::QubitId;

    #[test]
    fn test_routed_circuit_statistics() {
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: QubitId(0) }),
            Box::new(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            }),
            Box::new(SWAP {
                qubit1: QubitId(1),
                qubit2: QubitId(2),
            }),
        ];

        let mapping = [(0, 0), (1, 1), (2, 2)].iter().cloned().collect();
        let result = RoutingResult {
            total_swaps: 1,
            circuit_depth: 3,
            routing_overhead: 0.33,
        };

        let routed_circuit = RoutedCircuit::<3>::new(gates, mapping, result);
        let stats = routed_circuit.statistics();

        assert_eq!(stats.total_gates, 3);
        assert_eq!(stats.single_qubit_gates, 1);
        assert_eq!(stats.two_qubit_gates, 1);
        assert_eq!(stats.swap_gates, 1);
    }

    #[test]
    fn test_routing_pass_manager() {
        let mut manager = RoutingPassManager::new();
        let coupling_map = super::super::CouplingMap::linear(3);
        let sabre_config = super::super::SabreConfig::default();

        manager.add_pass(RoutingPassType::Sabre {
            coupling_map,
            config: sabre_config,
        });

        assert_eq!(manager.available_passes(), vec!["SABRE"]);
    }
}
