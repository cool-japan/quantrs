//! Circuit routing algorithms for mapping logical qubits to physical qubits
//!
//! This module implements various routing algorithms including SABRE and lookahead
//! routing for quantum circuits with limited connectivity.

mod coupling_map;
mod lookahead;
mod routing_pass;
mod sabre;
mod swap_network;

pub use coupling_map::{CouplingMap, Distance};
pub use lookahead::{LookaheadConfig, LookaheadRouter};
pub use routing_pass::{RoutedCircuit, RoutingPassType, RoutingResult, RoutingStatistics};
pub use sabre::{SabreConfig, SabreRouter};
pub use swap_network::{SwapLayer, SwapNetwork};

use crate::builder::Circuit;
use quantrs2_core::error::QuantRS2Result;
use std::collections::HashMap;

/// Routing strategy for quantum circuits
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// SABRE (SWAP-based `BidiREctional`) routing
    Sabre,
    /// Lookahead routing with configurable depth
    Lookahead { depth: usize },
    /// Basic greedy routing
    Basic,
    /// Stochastic routing with multiple trials
    Stochastic { trials: usize },
}

/// Main router interface
pub struct CircuitRouter {
    strategy: RoutingStrategy,
    coupling_map: CouplingMap,
}

impl CircuitRouter {
    /// Create a new router with the specified strategy and coupling map
    #[must_use]
    pub const fn new(strategy: RoutingStrategy, coupling_map: CouplingMap) -> Self {
        Self {
            strategy,
            coupling_map,
        }
    }

    /// Create a router for a specific backend
    #[must_use]
    pub fn for_backend(backend: &str) -> Self {
        let coupling_map = match backend {
            "ibm_lagos" => CouplingMap::ibm_lagos(),
            "ibm_nairobi" => CouplingMap::ibm_nairobi(),
            "google_sycamore" => CouplingMap::google_sycamore(),
            _ => CouplingMap::linear(5), // Default to 5-qubit linear
        };

        Self {
            strategy: RoutingStrategy::Sabre,
            coupling_map,
        }
    }

    /// Route a circuit to the physical device layout
    pub fn route<const N: usize>(&self, circuit: &Circuit<N>) -> QuantRS2Result<RoutedCircuit<N>> {
        match self.strategy {
            RoutingStrategy::Sabre => {
                let config = SabreConfig::default();
                let router = SabreRouter::new(self.coupling_map.clone(), config);
                router.route(circuit)
            }
            RoutingStrategy::Lookahead { depth } => {
                let config = LookaheadConfig::new(depth);
                let router = LookaheadRouter::new(self.coupling_map.clone(), config);
                router.route(circuit)
            }
            RoutingStrategy::Basic => {
                // Use SABRE with basic config for now
                let config = SabreConfig::basic();
                let router = SabreRouter::new(self.coupling_map.clone(), config);
                router.route(circuit)
            }
            RoutingStrategy::Stochastic { trials } => {
                // Run multiple trials and pick the best
                let mut best_result = None;
                let mut best_cost = f64::INFINITY;

                for _ in 0..trials {
                    let config = SabreConfig::stochastic();
                    let router = SabreRouter::new(self.coupling_map.clone(), config);
                    let result = router.route(circuit)?;
                    let cost = result.total_cost();

                    if cost < best_cost {
                        best_cost = cost;
                        best_result = Some(result);
                    }
                }

                best_result.ok_or_else(|| {
                    quantrs2_core::error::QuantRS2Error::RoutingError(
                        "Failed to find valid routing".to_string(),
                    )
                })
            }
        }
    }

    /// Get the coupling map
    #[must_use]
    pub const fn coupling_map(&self) -> &CouplingMap {
        &self.coupling_map
    }
}

/// Utilities for analyzing routing complexity
pub mod analysis {
    use super::{Circuit, CouplingMap};
    use crate::dag::{circuit_to_dag, CircuitDag};

    /// Analyze routing complexity for a circuit
    pub struct RoutingAnalyzer {
        coupling_map: CouplingMap,
    }

    impl RoutingAnalyzer {
        #[must_use]
        pub const fn new(coupling_map: CouplingMap) -> Self {
            Self { coupling_map }
        }

        /// Estimate the number of SWAPs needed
        #[must_use]
        pub fn estimate_swaps<const N: usize>(&self, circuit: &Circuit<N>) -> usize {
            let dag = circuit_to_dag(circuit);
            let mut swap_count = 0;

            // Simple heuristic: count non-adjacent two-qubit gates
            for node in dag.nodes() {
                if node.gate.qubits().len() == 2 {
                    let q1 = node.gate.qubits()[0];
                    let q2 = node.gate.qubits()[1];

                    if !self
                        .coupling_map
                        .are_connected(q1.id() as usize, q2.id() as usize)
                    {
                        // Estimate distance-based swaps
                        let distance = self
                            .coupling_map
                            .distance(q1.id() as usize, q2.id() as usize);
                        swap_count += distance.saturating_sub(1);
                    }
                }
            }

            swap_count
        }

        /// Calculate interaction graph density
        #[must_use]
        pub fn interaction_density<const N: usize>(&self, circuit: &Circuit<N>) -> f64 {
            let mut interactions = std::collections::HashSet::new();

            for gate in circuit.gates() {
                if gate.qubits().len() == 2 {
                    let q1 = gate.qubits()[0].id() as usize;
                    let q2 = gate.qubits()[1].id() as usize;
                    interactions.insert((q1.min(q2), q1.max(q2)));
                }
            }

            let max_interactions = (N * (N - 1)) / 2;
            if max_interactions == 0 {
                0.0
            } else {
                interactions.len() as f64 / max_interactions as f64
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::{multi::CNOT, single::Hadamard};
    use quantrs2_core::qubit::QubitId;

    #[test]
    fn test_basic_routing() {
        let coupling_map = CouplingMap::linear(3);
        let router = CircuitRouter::new(RoutingStrategy::Basic, coupling_map);

        let mut circuit = Circuit::<3>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("add H gate to circuit");
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(2),
            })
            .expect("add CNOT gate to circuit");

        let result = router.route(&circuit);
        assert!(result.is_ok());
    }
}
