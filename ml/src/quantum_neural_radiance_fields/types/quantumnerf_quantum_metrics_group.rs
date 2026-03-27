//! # QuantumNeRF - quantum_metrics_group Methods
//!
//! This module contains method implementations for `QuantumNeRF`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::QuantumRenderingMetrics;

use super::quantumnerf_type::QuantumNeRF;

impl QuantumNeRF {
    /// Get current quantum metrics
    pub fn quantum_metrics(&self) -> &QuantumRenderingMetrics {
        &self.quantum_rendering_metrics
    }
}
