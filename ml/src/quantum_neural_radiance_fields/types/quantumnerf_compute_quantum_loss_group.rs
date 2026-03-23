//! # QuantumNeRF - compute_quantum_loss_group Methods
//!
//! This module contains method implementations for `QuantumNeRF`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::QuantumMLPState;

use super::quantumnerf_type::QuantumNeRF;

impl QuantumNeRF {
    /// Compute quantum loss
    pub(super) fn compute_quantum_loss(&self, quantum_state: &QuantumMLPState) -> Result<f64> {
        let target_entanglement = 0.7;
        let entanglement_loss = (quantum_state.entanglement_measure - target_entanglement).powi(2);
        let fidelity_loss = 1.0 - quantum_state.quantum_fidelity;
        let coherence_loss = quantum_state
            .quantum_amplitudes
            .iter()
            .map(|amp| 1.0 - amp.norm())
            .sum::<f64>()
            / quantum_state.quantum_amplitudes.len() as f64;
        Ok(entanglement_loss + fidelity_loss + coherence_loss)
    }
}
