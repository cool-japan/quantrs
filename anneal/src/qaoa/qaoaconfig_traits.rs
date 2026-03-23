//! # QaoaConfig - Trait Implementations
//!
//! This module contains trait implementations for `QaoaConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;
use std::time::{Duration, Instant};

use super::types::{
    MixerType, ParameterInitialization, ProblemEncoding, QaoaClassicalOptimizer, QaoaConfig,
    QaoaVariant,
};

impl Default for QaoaConfig {
    fn default() -> Self {
        Self {
            variant: QaoaVariant::Standard { layers: 1 },
            mixer_type: MixerType::XMixer,
            problem_encoding: ProblemEncoding::Ising,
            optimizer: QaoaClassicalOptimizer::NelderMead {
                initial_size: 0.5,
                tolerance: 1e-6,
                max_iterations: 1000,
            },
            num_shots: 1000,
            parameter_init: ParameterInitialization::Random {
                range: (-std::f64::consts::PI, std::f64::consts::PI),
            },
            convergence_tolerance: 1e-6,
            max_optimization_time: Some(Duration::from_secs(3600)),
            seed: None,
            detailed_logging: false,
            track_optimization_history: true,
            max_circuit_depth: None,
            use_symmetry_reduction: false,
        }
    }
}
