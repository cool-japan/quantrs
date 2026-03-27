//! # MetaLearningConfig - Trait Implementations
//!
//! This module contains trait implementations for `MetaLearningConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{MetaLearningConfig, TaskDistribution};

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            inner_steps: 5,
            meta_lr: 0.001,
            task_distribution: TaskDistribution::Uniform,
        }
    }
}
