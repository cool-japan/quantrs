//! Resource scaling configurations

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Resource scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceScalingConfig {
    /// Auto-scaling
    pub auto_scaling: AutoScalingConfig,
    /// Manual scaling
    pub manual_scaling: ManualScalingConfig,
    /// Predictive scaling
    pub predictive_scaling: PredictiveScalingConfig,
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Enable auto-scaling
    pub enabled: bool,
    /// Scaling policies
    pub policies: Vec<ScalingPolicy>,
    /// Cooldown period
    pub cooldown_period: Duration,
    /// Scaling limits
    pub limits: ScalingLimits,
}

/// Scaling policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPolicy {
    /// Policy name
    pub name: String,
    /// Trigger conditions
    pub conditions: Vec<ScalingCondition>,
    /// Scaling action
    pub action: ScalingAction,
    /// Priority
    pub priority: u8,
}

/// Scaling condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingCondition {
    /// Metric name
    pub metric: String,
    /// Operator
    pub operator: ComparisonOperator,
    /// Threshold
    pub threshold: f64,
    /// Duration
    pub duration: Duration,
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    GreaterThanOrEqual,
    LessThanOrEqual,
    NotEqual,
}

/// Scaling actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingAction {
    /// Action type
    pub action_type: ScalingActionType,
    /// Scale amount
    pub amount: ScalingAmount,
    /// Target group
    pub target: String,
}

/// Scaling action types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingActionType {
    ScaleUp,
    ScaleDown,
    ScaleOut,
    ScaleIn,
}

/// Scaling amount
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingAmount {
    Absolute(u32),
    Percentage(f64),
    Capacity(f64),
}

/// Scaling limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingLimits {
    /// Minimum instances
    pub min_instances: usize,
    /// Maximum instances
    pub max_instances: usize,
    /// Maximum scaling rate
    pub max_scaling_rate: f64,
}

/// Manual scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualScalingConfig {
    /// Default instance count
    pub default_instances: usize,
    /// Scaling increments
    pub scaling_increments: Vec<usize>,
    /// Approval required
    pub approval_required: bool,
}

/// Predictive scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveScalingConfig {
    /// Enable predictive scaling
    pub enabled: bool,
    /// Prediction models
    pub models: Vec<PredictionModel>,
    /// Forecast horizon
    pub forecast_horizon: Duration,
    /// Confidence threshold
    pub confidence_threshold: f64,
}

/// Prediction models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionModel {
    ARIMA,
    LinearRegression,
    NeuralNetwork,
    EnsembleModel,
    Custom(String),
}
