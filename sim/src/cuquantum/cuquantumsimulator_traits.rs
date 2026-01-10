//! # CuQuantumSimulator - Trait Implementations
//!
//! This module contains trait implementations for `CuQuantumSimulator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CuQuantumConfig, CuQuantumSimulator};

impl Default for CuQuantumSimulator {
    fn default() -> Self {
        Self::new(CuQuantumConfig::default()).unwrap_or_else(|_| Self {
            statevec: None,
            tensornet: None,
            config: CuQuantumConfig::default(),
            tensornet_threshold: 30,
        })
    }
}
