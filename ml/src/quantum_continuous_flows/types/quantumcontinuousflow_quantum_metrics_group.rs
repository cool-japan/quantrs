//! # QuantumContinuousFlow - quantum_metrics_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::QuantumFlowMetrics;

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Get current quantum metrics
    pub fn quantum_metrics(&self) -> &QuantumFlowMetrics {
        &self.quantum_flow_metrics
    }
}
