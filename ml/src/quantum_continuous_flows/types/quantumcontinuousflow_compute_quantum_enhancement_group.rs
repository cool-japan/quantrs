//! # QuantumContinuousFlow - compute_quantum_enhancement_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::{QuantumEnhancement, QuantumLayerState};

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Compute quantum enhancement
    pub(super) fn compute_quantum_enhancement(
        &self,
        quantum_states: &[QuantumLayerState],
    ) -> Result<QuantumEnhancement> {
        let average_entanglement = quantum_states
            .iter()
            .map(|state| state.entanglement_measure)
            .sum::<f64>()
            / quantum_states.len() as f64;
        let average_fidelity = quantum_states
            .iter()
            .map(|state| state.quantum_fidelity)
            .sum::<f64>()
            / quantum_states.len() as f64;
        let average_coherence = quantum_states
            .iter()
            .map(|state| state.coherence_time)
            .sum::<f64>()
            / quantum_states.len() as f64;
        let log_enhancement = 0.1 * (average_entanglement + average_fidelity + average_coherence);
        let quantum_advantage_ratio = 1.0 + average_entanglement * 2.0 + average_fidelity;
        Ok(QuantumEnhancement {
            log_enhancement,
            entanglement_contribution: average_entanglement,
            fidelity_contribution: average_fidelity,
            coherence_contribution: average_coherence,
            quantum_advantage_ratio,
        })
    }
}
