//! Optimization strategies for quantum annealing
//!
//! This module provides advanced optimization techniques including
//! penalty function optimization, constraint handling, and parameter tuning.

pub mod penalty;
pub mod tuning;
pub mod constraints;
pub mod adaptive;

pub use self::penalty::{PenaltyOptimizer, PenaltyConfig};
pub use self::tuning::{ParameterTuner, TuningResult};
pub use self::constraints::{Constraint, ConstraintType, ConstraintHandler};
pub use self::adaptive::{AdaptiveStrategy, AdaptiveOptimizer};

/// Prelude for common optimization imports
pub mod prelude {
    pub use super::{
        PenaltyOptimizer, PenaltyConfig,
        ParameterTuner, TuningResult,
        Constraint, ConstraintType,
        AdaptiveStrategy,
    };
}