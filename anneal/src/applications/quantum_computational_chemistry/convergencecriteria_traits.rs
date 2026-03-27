//! # ConvergenceCriteria - Trait Implementations
//!
//! This module contains trait implementations for `ConvergenceCriteria`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ConvergenceCriteria;

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            energy_threshold: 1e-8,
            density_threshold: 1e-6,
            max_scf_iterations: 100,
            gradient_threshold: 1e-6,
            use_diis: true,
        }
    }
}
