//! Dashboard Optimization Configuration Types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Dashboard optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardOptimizationConfig {
    /// Enable automatic optimization recommendations
    pub enable_auto_recommendations: bool,
    /// Optimization objectives
    pub optimization_objectives: Vec<OptimizationObjective>,
    /// Recommendation confidence threshold
    pub confidence_threshold: f64,
    /// Implementation priority weighting
    pub priority_weighting: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptimizationObjective {
    MaximizeThroughput,
    MinimizeLatency,
    MaximizeFidelity,
    MinimizeError,
    MinimizeCost,
    MaximizeReliability,
    BalancedPerformance,
}
