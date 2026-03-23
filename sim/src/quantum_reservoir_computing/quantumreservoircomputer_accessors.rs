//! # QuantumReservoirComputer - accessors Methods
//!
//! This module contains method implementations for `QuantumReservoirComputer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::ReservoirMetrics;

use super::quantumreservoircomputer_type::QuantumReservoirComputer;

impl QuantumReservoirComputer {
    /// Get current metrics
    pub const fn get_metrics(&self) -> &ReservoirMetrics {
        &self.metrics
    }
}
