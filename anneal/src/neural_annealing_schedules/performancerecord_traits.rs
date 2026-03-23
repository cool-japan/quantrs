//! # PerformanceRecord - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceRecord`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;
use std::time::{Duration, Instant};

use super::types::{ConvergenceMetrics, EnergyStatistics, PerformanceRecord};

impl Default for PerformanceRecord {
    fn default() -> Self {
        Self {
            solution_quality: 0.0,
            time_to_solution: Duration::from_secs(1),
            success_rate: 0.0,
            energy_stats: EnergyStatistics {
                final_energy: 0.0,
                energy_gap: 0.0,
                energy_trajectory: Vec::new(),
                energy_variance: 0.0,
            },
            convergence_metrics: ConvergenceMetrics {
                convergence_time: Duration::from_secs(1),
                convergence_rate: 0.0,
                plateau_detected: false,
                oscillation_amplitude: 0.0,
            },
        }
    }
}
