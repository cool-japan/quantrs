//! # EnhancedResourceEstimator - start_monitoring_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::QuantRS2Error;
use crate::parallel_ops_stubs::*;

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Monitor resources in real-time
    pub const fn start_monitoring(&mut self) -> Result<(), QuantRS2Error> {
        self.realtime_tracker.start_monitoring()
    }
}
