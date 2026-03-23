//! # EnhancedTensorNetworkSimulator - accessors Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::TensorNetworkStats;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    /// Get performance statistics
    #[must_use]
    pub const fn get_stats(&self) -> &TensorNetworkStats {
        &self.stats
    }
}
