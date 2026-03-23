//! # LinalgConfig - Trait Implementations
//!
//! This module contains trait implementations for `LinalgConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{LinalgAlgorithm, LinalgConfig};

impl Default for LinalgConfig {
    fn default() -> Self {
        Self {
            algorithm_type: LinalgAlgorithm::QuantumInspiredLinearSolver,
            matrix_dimension: 1024,
            precision: 1e-8,
            max_iterations: 1000,
            krylov_dimension: 50,
        }
    }
}
