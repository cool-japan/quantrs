//! # ProviderOptimizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `ProviderOptimizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::traits::ProviderOptimizer;
use super::types::*;
use crate::prelude::CloudProvider;
use crate::DeviceResult;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

impl Default for ProviderOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimization_level: OptimizationLevel::Balanced,
            target_metrics: vec![
                OptimizationMetric::ExecutionTime,
                OptimizationMetric::Cost,
                OptimizationMetric::Fidelity,
            ],
            cost_constraints: CostConstraints {
                max_cost_per_execution: Some(100.0),
                max_daily_budget: Some(1000.0),
                max_monthly_budget: Some(10000.0),
                cost_optimization_priority: 0.3,
                cost_tolerance: 0.1,
            },
            performance_targets: PerformanceTargets {
                max_execution_time: Some(Duration::from_secs(3600)),
                min_fidelity: Some(0.95),
                max_queue_time: Some(Duration::from_secs(1800)),
                min_throughput: Some(10.0),
                max_error_rate: Some(0.05),
            },
            caching_enabled: true,
            adaptive_optimization: true,
            real_time_optimization: false,
        }
    }
}
