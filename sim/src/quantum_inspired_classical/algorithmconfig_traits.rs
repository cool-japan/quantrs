//! # AlgorithmConfig - Trait Implementations
//!
//! This module contains trait implementations for `AlgorithmConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{AlgorithmConfig, QuantumParameters, TemperatureSchedule};

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            population_size: 100,
            elite_ratio: 0.1,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            temperature_schedule: TemperatureSchedule::Exponential,
            quantum_parameters: QuantumParameters::default(),
        }
    }
}
