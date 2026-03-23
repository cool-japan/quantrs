//! # StateTomographyConfig - Trait Implementations
//!
//! This module contains trait implementations for `StateTomographyConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{StateTomographyConfig, TomographyMethod};

impl Default for StateTomographyConfig {
    fn default() -> Self {
        Self {
            shots_per_basis: 1000,
            method: TomographyMethod::LinearInversion,
            physical_constraints: true,
            threshold: 1e-10,
        }
    }
}
