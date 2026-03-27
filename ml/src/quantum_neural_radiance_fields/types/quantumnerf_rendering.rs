//! # QuantumNeRF - rendering Methods
//!
//! This module contains method implementations for `QuantumNeRF`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::{QuantumMLPState, RenderingMetrics};

use super::quantumnerf_type::QuantumNeRF;

impl QuantumNeRF {
    /// Compute rendering metrics
    pub(super) fn compute_rendering_metrics(
        &self,
        rendered_image: &Array3<f64>,
        pixel_quantum_states: &[QuantumMLPState],
    ) -> Result<RenderingMetrics> {
        let average_entanglement = pixel_quantum_states
            .iter()
            .map(|state| state.entanglement_measure)
            .sum::<f64>()
            / pixel_quantum_states.len() as f64;
        let average_fidelity = pixel_quantum_states
            .iter()
            .map(|state| state.quantum_fidelity)
            .sum::<f64>()
            / pixel_quantum_states.len() as f64;
        Ok(RenderingMetrics {
            average_pixel_entanglement: average_entanglement,
            average_quantum_fidelity: average_fidelity,
            rendering_quantum_advantage: 1.0 + average_entanglement * 2.0,
            coherence_preservation: average_fidelity,
        })
    }
}
