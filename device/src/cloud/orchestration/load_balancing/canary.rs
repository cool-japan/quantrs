//! Canary deployment configurations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Canary deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryConfig {
    /// Enable canary deployments
    pub enabled: bool,
    /// Initial traffic percentage
    pub initial_percentage: f64,
    /// Increment percentage
    pub increment_percentage: f64,
    /// Promotion criteria
    pub promotion_criteria: PromotionCriteria,
}

/// Promotion criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionCriteria {
    /// Success metrics
    pub success_metrics: HashMap<String, f64>,
    /// Failure thresholds
    pub failure_thresholds: HashMap<String, f64>,
    /// Observation period
    pub observation_period: Duration,
}
