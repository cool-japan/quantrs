//! # ScheduleConstraints - Trait Implementations
//!
//! This module contains trait implementations for `ScheduleConstraints`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    EnergyGapConstraints, GapPreservationStrategy, HardwareConstraints, ScheduleConstraints,
    SmoothnessConstraints,
};

impl Default for ScheduleConstraints {
    fn default() -> Self {
        Self {
            min_annealing_time: 1.0,
            max_annealing_time: 10_000.0,
            smoothness_constraints: SmoothnessConstraints {
                max_derivative: 1.0,
                max_second_derivative: 1.0,
                penalty_weight: 0.1,
            },
            hardware_constraints: HardwareConstraints {
                control_precision: 1e-6,
                max_update_rate: 1e6,
                field_strength_limits: (0.0, 1.0),
                bandwidth_constraints: 1e6,
            },
            energy_gap_constraints: EnergyGapConstraints {
                min_gap_threshold: 1e-6,
                gap_preservation: GapPreservationStrategy::Adaptive,
                avoid_diabatic_transitions: true,
            },
        }
    }
}
