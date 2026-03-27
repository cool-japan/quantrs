//! # EnhancedResourceEstimator - stop_monitoring_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::QuantRS2Error;
use crate::parallel_ops_stubs::*;

use super::types::MonitoringReport;

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Stop resource monitoring
    pub const fn stop_monitoring(&mut self) -> Result<MonitoringReport, QuantRS2Error> {
        self.realtime_tracker.stop_monitoring()
    }
}
