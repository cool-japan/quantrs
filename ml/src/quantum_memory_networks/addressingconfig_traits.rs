//! # AddressingConfig - Trait Implementations
//!
//! This module contains trait implementations for `AddressingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{AddressingConfig, AddressingType};

impl Default for AddressingConfig {
    fn default() -> Self {
        Self {
            addressing_type: AddressingType::QuantumContentBased,
            content_addressing: true,
            location_addressing: true,
            sharpening_factor: 2.0,
            quantum_superposition: true,
        }
    }
}
