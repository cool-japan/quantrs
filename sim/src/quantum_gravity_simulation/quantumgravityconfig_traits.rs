//! # QuantumGravityConfig - Trait Implementations
//!
//! This module contains trait implementations for `QuantumGravityConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    AdSCFTConfig, AsymptoticSafetyConfig, BackgroundMetric, CDTConfig, GravityApproach, LQGConfig,
    QuantumGravityConfig,
};

impl Default for QuantumGravityConfig {
    fn default() -> Self {
        Self {
            gravity_approach: GravityApproach::LoopQuantumGravity,
            planck_length: 1.616e-35,
            planck_time: 5.391e-44,
            spatial_dimensions: 3,
            lorentz_invariant: true,
            background_metric: BackgroundMetric::Minkowski,
            cosmological_constant: 0.0,
            gravitational_constant: 6.674e-11,
            speed_of_light: 299_792_458.0,
            reduced_planck_constant: 1.055e-34,
            quantum_corrections: true,
            lqg_config: Some(LQGConfig::default()),
            cdt_config: Some(CDTConfig::default()),
            asymptotic_safety_config: Some(AsymptoticSafetyConfig::default()),
            ads_cft_config: Some(AdSCFTConfig::default()),
        }
    }
}
