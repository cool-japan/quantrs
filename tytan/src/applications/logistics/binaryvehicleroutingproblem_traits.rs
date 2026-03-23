//! # BinaryVehicleRoutingProblem - Trait Implementations
//!
//! This module contains trait implementations for `BinaryVehicleRoutingProblem`.
//!
//! ## Implemented Traits
//!
//! - `OptimizationProblem`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;
use scirs2_core::random::prelude::*;

use super::functions::OptimizationProblem;
use super::types::BinaryVehicleRoutingProblem;

impl OptimizationProblem for BinaryVehicleRoutingProblem {
    type Solution = Vec<i8>;
    fn evaluate(&self, solution: &Self::Solution) -> f64 {
        self.evaluate_binary(solution)
    }
}
