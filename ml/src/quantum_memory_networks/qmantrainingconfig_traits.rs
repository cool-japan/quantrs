//! # QMANTrainingConfig - Trait Implementations
//!
//! This module contains trait implementations for `QMANTrainingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{MemoryReplayStrategy, MetaLearningConfig, QMANTrainingConfig};

impl Default for QMANTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 300,
            learning_rate: 0.0005,
            batch_size: 16,
            memory_replay: MemoryReplayStrategy::QuantumPrioritized,
            curriculum_learning: true,
            meta_learning: Some(MetaLearningConfig::default()),
        }
    }
}
