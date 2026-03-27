//! # VQEConfig - Trait Implementations
//!
//! This module contains trait implementations for `VQEConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{ChemistryAnsatz, ChemistryOptimizer, VQEConfig};

impl Default for VQEConfig {
    fn default() -> Self {
        Self {
            ansatz: ChemistryAnsatz::UCCSD,
            optimizer: ChemistryOptimizer::COBYLA,
            max_iterations: 100,
            energy_threshold: 1e-6,
            gradient_threshold: 1e-4,
            shots: 10_000,
            enable_noise_mitigation: true,
        }
    }
}
