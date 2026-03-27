//! # QuantumNeRF - rendering Methods
//!
//! This module contains method implementations for `QuantumNeRF`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::NeRFTrainingMetrics;

use super::quantumnerf_type::QuantumNeRF;

impl QuantumNeRF {
    /// Update quantum rendering metrics
    pub(super) fn update_quantum_rendering_metrics(
        &mut self,
        epoch_metrics: &NeRFTrainingMetrics,
    ) -> Result<()> {
        self.quantum_rendering_metrics.entanglement_utilization = 0.9
            * self.quantum_rendering_metrics.entanglement_utilization
            + 0.1 * epoch_metrics.entanglement_measure;
        self.quantum_rendering_metrics.coherence_preservation = 0.9
            * self.quantum_rendering_metrics.coherence_preservation
            + 0.1 * epoch_metrics.quantum_fidelity;
        self.quantum_rendering_metrics.quantum_acceleration_factor =
            epoch_metrics.quantum_advantage_ratio;
        Ok(())
    }
}
