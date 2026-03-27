//! # NoiseCharacterizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `NoiseCharacterizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::NoiseCharacterizationConfig;

impl Default for NoiseCharacterizationConfig {
    fn default() -> Self {
        Self {
            num_sequences: 100,
            sequence_lengths: vec![1, 2, 4, 8, 16, 32, 64, 128],
            shots_per_sequence: 1000,
            confidence_level: 0.95,
        }
    }
}
