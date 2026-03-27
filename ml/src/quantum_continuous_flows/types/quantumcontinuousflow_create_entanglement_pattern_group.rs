//! # QuantumContinuousFlow - create_entanglement_pattern_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::{
    ConnectivityGraph, EntanglementPattern, EntanglementPatternType, QuantumContinuousFlowConfig,
};

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Create entanglement pattern
    pub(super) fn create_entanglement_pattern(
        config: &QuantumContinuousFlowConfig,
    ) -> Result<EntanglementPattern> {
        let connectivity = ConnectivityGraph {
            adjacency_matrix: Array2::<f64>::eye(config.num_qubits).mapv(|x| x != 0.0),
            edge_weights: Array2::ones((config.num_qubits, config.num_qubits)),
            num_nodes: config.num_qubits,
        };
        Ok(EntanglementPattern {
            pattern_type: EntanglementPatternType::Circular,
            connectivity,
            entanglement_strength: Array1::ones(config.num_qubits)
                * config.entanglement_coupling_strength,
        })
    }
}
