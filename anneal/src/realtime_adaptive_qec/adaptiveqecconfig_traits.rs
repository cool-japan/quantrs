//! # AdaptiveQecConfig - Trait Implementations
//!
//! This module contains trait implementations for `AdaptiveQecConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{
    AdaptiveQecConfig, HierarchyConfig, MLNoiseConfig, PredictionConfig, ResourceManagementConfig,
};

impl Default for AdaptiveQecConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_millis(100),
            adaptation_threshold: 0.1,
            performance_window: Duration::from_secs(30),
            ml_config: MLNoiseConfig::default(),
            hierarchy_config: HierarchyConfig::default(),
            resource_config: ResourceManagementConfig::default(),
            prediction_config: PredictionConfig::default(),
        }
    }
}
