//! Alerting Configuration Types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Enable alerting system
    pub enable_alerting: bool,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, AlertThreshold>,
    /// Notification channels
    pub notification_channels: Vec<NotificationChannel>,
    /// Alert escalation rules
    pub escalation_rules: Vec<EscalationRule>,
    /// Anomaly detection settings
    pub anomaly_detection: AnomalyDetectionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThreshold {
    pub metric_name: String,
    pub threshold_value: f64,
    pub comparison_operator: ComparisonOperator,
    pub severity: AlertSeverity,
    pub duration_threshold: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NotificationChannel {
    Email,
    Slack,
    SMS,
    Webhook,
    PushNotification,
    Dashboard,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRule {
    pub trigger_condition: String,
    pub escalation_delay: Duration,
    pub escalation_targets: Vec<String>,
    pub escalation_severity: AlertSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    pub detection_algorithms: Vec<AnomalyDetectionAlgorithm>,
    pub sensitivity: f64,
    pub baseline_window: Duration,
    pub detection_window: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AnomalyDetectionAlgorithm {
    StatisticalOutlier,
    MovingAverage,
    ExponentialSmoothing,
    IsolationForest,
    LocalOutlierFactor,
    MachineLearning,
}
