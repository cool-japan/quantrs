//! Optimization strategies for quantum annealing
//!
//! This module provides advanced optimization techniques including
//! penalty function optimization, constraint handling, and parameter tuning.

pub mod adaptive;
pub mod constraints;
pub mod penalty;
pub mod tuning;

pub use self::adaptive::{AdaptiveOptimizer, AdaptiveStrategy};
pub use self::constraints::{Constraint, ConstraintHandler, ConstraintType};
pub use self::penalty::{PenaltyConfig, PenaltyOptimizer};
pub use self::tuning::{ParameterTuner, TuningResult};

/// Prelude for common optimization imports
pub mod prelude {
    pub use super::{
        AdaptiveStrategy, Constraint, ConstraintType, ParameterTuner, PenaltyConfig,
        PenaltyOptimizer, TuningResult,
    };
}
