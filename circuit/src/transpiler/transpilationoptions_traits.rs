//! # TranspilationOptions - Trait Implementations
//!
//! This module contains trait implementations for `TranspilationOptions`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    GraphOptimizationStrategy, HardwareSpec, SciRS2TranspilerConfig, TranspilationOptions,
    TranspilationStrategy,
};

impl Default for TranspilationOptions {
    fn default() -> Self {
        Self {
            hardware_spec: HardwareSpec::generic(),
            strategy: TranspilationStrategy::SciRS2GraphOptimized {
                graph_strategy: GraphOptimizationStrategy::MultiObjective,
                parallel_processing: true,
                advanced_connectivity: true,
            },
            max_iterations: 10,
            aggressive: false,
            seed: None,
            initial_layout: None,
            skip_routing_if_connected: true,
            scirs2_config: SciRS2TranspilerConfig::default(),
        }
    }
}
