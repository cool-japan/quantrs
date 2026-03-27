//! # EnhancedResourceEstimator - estimate_code_distance_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::QuantRS2Error;
use crate::parallel_ops_stubs::*;

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Estimate code distance
    pub(super) fn estimate_code_distance(&self) -> Result<usize, QuantRS2Error> {
        let p = self.config.base_config.physical_error_rate;
        let p_target = self.config.base_config.target_logical_error_rate;
        let threshold = 0.01;
        if p > threshold {
            return Err(QuantRS2Error::InvalidInput(
                "Physical error rate too high".into(),
            ));
        }
        let distance = ((-p_target.log10()) / (-p.log10())).ceil() as usize;
        Ok(distance.max(3))
    }
}
