//! # Circuitoptimizationconfig - Trait Implementations
//!
//! This module contains trait implementations for `Circuitoptimizationconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for CircuitOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimization_passes: vec![
                OptimizationPass::GateFusion,
                OptimizationPass::CircuitDepthReduction,
                OptimizationPass::NoiseAwareOptimization,
            ],
            optimization_level: OptimizationLevel::Moderate,
            target_platform_optimization: true,
        }
    }
}
