//! # AutoParallelConfig - Trait Implementations
//!
//! This module contains trait implementations for `AutoParallelConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    AutoParallelConfig, LoadBalancingConfig, OptimizationLevel, ParallelizationStrategy,
    ResourceConstraints,
};
use scirs2_core::parallel_ops::current_num_threads;

impl Default for AutoParallelConfig {
    fn default() -> Self {
        Self {
            max_threads: current_num_threads(),
            min_gates_for_parallel: 10,
            strategy: ParallelizationStrategy::DependencyAnalysis,
            resource_constraints: ResourceConstraints::default(),
            enable_inter_layer_parallel: true,
            enable_gate_fusion: true,
            scirs2_optimization_level: OptimizationLevel::Aggressive,
            load_balancing: LoadBalancingConfig::default(),
            enable_analysis_caching: true,
            memory_budget: 4 * 1024 * 1024 * 1024,
        }
    }
}
