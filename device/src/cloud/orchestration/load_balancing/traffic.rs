//! Traffic distribution and splitting configurations

use super::canary::CanaryConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Traffic distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficDistributionConfig {
    /// Distribution algorithm
    pub algorithm: DistributionAlgorithm,
    /// Weight assignments
    pub weights: HashMap<String, f64>,
    /// Traffic splitting
    pub splitting: TrafficSplittingConfig,
}

/// Distribution algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DistributionAlgorithm {
    Weighted,
    Random,
    Consistent,
    Geographic,
    Custom(String),
}

/// Traffic splitting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficSplittingConfig {
    /// Enable traffic splitting
    pub enabled: bool,
    /// Split rules
    pub rules: Vec<SplitRule>,
    /// Canary deployment
    pub canary: CanaryConfig,
}

/// Split rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitRule {
    /// Rule name
    pub name: String,
    /// Condition
    pub condition: SplitCondition,
    /// Target percentage
    pub percentage: f64,
    /// Target destination
    pub destination: String,
}

/// Split condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitCondition {
    /// Header conditions
    pub headers: HashMap<String, String>,
    /// Query parameters
    pub query_params: HashMap<String, String>,
    /// User attributes
    pub user_attributes: HashMap<String, String>,
}
