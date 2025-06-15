//! Hardware Prediction Configuration Types

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Hardware prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwarePredictionConfig {
    /// Enable hardware prediction
    pub enable_prediction: bool,
    /// Prediction targets
    pub prediction_targets: Vec<PredictionTarget>,
    /// Prediction horizon
    pub prediction_horizon: Duration,
    /// Uncertainty quantification
    pub uncertainty_quantification: bool,
    /// Multi-step prediction
    pub multi_step_prediction: bool,
    /// Hardware adaptation
    pub hardware_adaptation: HardwareAdaptationConfig,
}

/// Prediction targets
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PredictionTarget {
    CircuitFidelity,
    ExecutionTime,
    ErrorRate,
    ResourceUtilization,
    CircuitDepth,
    SwapCount,
    GateCount,
    OptimalMapping,
    HardwareBottlenecks,
    CalibrationDrift,
}

/// Hardware adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareAdaptationConfig {
    /// Enable adaptive optimization
    pub enable_adaptation: bool,
    /// Adaptation frequency
    pub adaptation_frequency: Duration,
    /// Adaptation triggers
    pub adaptation_triggers: Vec<AdaptationTrigger>,
    /// Learning rate adaptation
    pub learning_rate_adaptation: bool,
}

/// Adaptation triggers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AdaptationTrigger {
    PerformanceDegradation,
    CalibrationDrift,
    EnvironmentalChange,
    NewHardwareData,
    ScheduledUpdate,
    ThresholdBreach,
}
