//! # TensorNetworkConfig - Trait Implementations
//!
//! This module contains trait implementations for `TensorNetworkConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{ContractionMethod, TensorNetworkConfig, TensorTopology};

impl Default for TensorNetworkConfig {
    fn default() -> Self {
        Self {
            bond_dimension: 64,
            topology: TensorTopology::MPS,
            contraction_method: ContractionMethod::OptimalContraction,
            truncation_threshold: 1e-12,
        }
    }
}
