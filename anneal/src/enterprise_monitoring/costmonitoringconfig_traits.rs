//! # CostMonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `CostMonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{BudgetAction, BudgetThreshold, CostMonitoringConfig, CostOptimizationRule};

impl Default for CostMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_cost_monitoring: true,
            enable_finops: true,
            cost_allocation_tags: vec![
                "team".to_string(),
                "project".to_string(),
                "environment".to_string(),
                "quantum_algorithm".to_string(),
            ],
            budget_thresholds: vec![
                BudgetThreshold {
                    percentage: 80.0,
                    action: BudgetAction::Alert,
                },
                BudgetThreshold {
                    percentage: 95.0,
                    action: BudgetAction::Restrict,
                },
            ],
            optimization_rules: vec![
                CostOptimizationRule::IdleResourceShutdown,
                CostOptimizationRule::RightSizing,
                CostOptimizationRule::ReservedInstanceOptimization,
            ],
        }
    }
}
