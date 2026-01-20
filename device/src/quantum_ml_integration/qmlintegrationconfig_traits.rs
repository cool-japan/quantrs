//! # QMLIntegrationConfig - Trait Implementations
//!
//! This module contains trait implementations for `QMLIntegrationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

// Import types from sibling modules
use super::types::*;
// Merged into super::types
// Merged into super::types

/// Default implementations
impl Default for QMLIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_qnn: true,
            enable_hybrid_training: true,
            enable_autodiff: true,
            enabled_frameworks: vec![
                MLFramework::TensorFlow,
                MLFramework::PyTorch,
                MLFramework::PennyLane,
            ],
            training_config: QMLTrainingConfig::default(),
            optimization_config: QMLOptimizationConfig::default(),
            resource_config: QMLResourceConfig::default(),
            monitoring_config: QMLMonitoringConfig::default(),
        }
    }
}
