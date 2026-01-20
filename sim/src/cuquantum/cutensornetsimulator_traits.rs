//! # CuTensorNetSimulator - Trait Implementations
//!
//! This module contains trait implementations for `CuTensorNetSimulator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CuQuantumConfig, CuTensorNetSimulator, SimulationStats};

impl Default for CuTensorNetSimulator {
    fn default() -> Self {
        Self::new(CuQuantumConfig::default()).unwrap_or_else(|_| Self {
            config: CuQuantumConfig::default(),
            device_info: None,
            stats: SimulationStats::default(),
            tensor_network: None,
        })
    }
}
