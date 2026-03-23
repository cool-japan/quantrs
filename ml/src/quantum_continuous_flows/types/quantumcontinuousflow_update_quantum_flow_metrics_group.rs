//! # QuantumContinuousFlow - update_quantum_flow_metrics_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::FlowTrainingMetrics;

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Update quantum flow metrics
    pub(super) fn update_quantum_flow_metrics(
        &mut self,
        epoch_metrics: &FlowTrainingMetrics,
    ) -> Result<()> {
        self.quantum_flow_metrics.average_entanglement = 0.9
            * self.quantum_flow_metrics.average_entanglement
            + 0.1 * epoch_metrics.entanglement_measure;
        self.quantum_flow_metrics.coherence_preservation = 0.9
            * self.quantum_flow_metrics.coherence_preservation
            + 0.1 * epoch_metrics.coherence_time;
        self.quantum_flow_metrics.invertibility_accuracy = epoch_metrics.invertibility_score;
        self.quantum_flow_metrics.quantum_speedup_factor = epoch_metrics.quantum_advantage_ratio;
        Ok(())
    }
}
