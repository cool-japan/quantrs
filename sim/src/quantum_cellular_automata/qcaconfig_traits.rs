//! # QCAConfig - Trait Implementations
//!
//! This module contains trait implementations for `QCAConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    BoundaryConditions, MeasurementStrategy, NeighborhoodType, QCAConfig, QCARuleType,
};

impl Default for QCAConfig {
    fn default() -> Self {
        Self {
            dimensions: vec![16],
            boundary_conditions: BoundaryConditions::Periodic,
            neighborhood: NeighborhoodType::Moore,
            rule_type: QCARuleType::Partitioned,
            evolution_steps: 100,
            parallel_evolution: true,
            measurement_strategy: MeasurementStrategy::Probabilistic,
        }
    }
}
