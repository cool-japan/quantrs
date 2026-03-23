//! # ZXOptimizer - Trait Implementations
//!
//! This module contains trait implementations for `ZXOptimizer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{single::*, GateOp},
    qubit::QubitId,
};

use super::types::ZXOptimizer;

impl Default for ZXOptimizer {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            enable_advanced: true,
            verbose: false,
            tolerance: 1e-10,
        }
    }
}
