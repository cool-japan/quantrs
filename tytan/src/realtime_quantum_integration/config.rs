//! Configuration types for Real-time Quantum Computing Integration
//!
//! This module provides configuration structs.

use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::types::{AlertChannel, AllocationStrategy, MetricType};

/// Real-time system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeConfig {
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Resource allocation strategy
    pub allocation_strategy: AllocationStrategy,
    /// Fault detection sensitivity
    pub fault_detection_sensitivity: f64,
    /// Performance analytics settings
    pub analytics_config: AnalyticsConfig,
    /// Auto-recovery enabled
    pub auto_recovery_enabled: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Data retention period
    pub data_retention_period: Duration,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_millis(100),
            max_queue_size: 1000,
            allocation_strategy: AllocationStrategy::LoadBalanced,
            fault_detection_sensitivity: 0.95,
            analytics_config: AnalyticsConfig::default(),
            auto_recovery_enabled: true,
            alert_thresholds: AlertThresholds::default(),
            data_retention_period: Duration::from_secs(24 * 3600), // 24 hours
        }
    }
}

/// Analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable real-time metrics collection
    pub real_time_metrics: bool,
    /// Enable predictive analytics
    pub predictive_analytics: bool,
    /// Metrics aggregation interval
    pub aggregation_interval: Duration,
    /// Historical data analysis depth
    pub analysis_depth: Duration,
    /// Performance prediction horizon
    pub prediction_horizon: Duration,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            real_time_metrics: true,
            predictive_analytics: true,
            aggregation_interval: Duration::from_secs(60),
            analysis_depth: Duration::from_secs(3600), // 1 hour
            prediction_horizon: Duration::from_secs(1800), // 30 minutes
        }
    }
}

/// Alert thresholds configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// CPU utilization threshold
    pub cpu_threshold: f64,
    /// Memory utilization threshold
    pub memory_threshold: f64,
    /// Queue length threshold
    pub queue_threshold: usize,
    /// Error rate threshold
    pub error_rate_threshold: f64,
    /// Response time threshold
    pub response_time_threshold: Duration,
    /// Hardware failure threshold
    pub hardware_failure_threshold: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 0.85,
            memory_threshold: 0.90,
            queue_threshold: 100,
            error_rate_threshold: 0.05,
            response_time_threshold: Duration::from_secs(300),
            hardware_failure_threshold: 0.01,
        }
    }
}

/// Monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Monitoring frequency
    pub monitoring_frequency: Duration,
    /// Metrics to collect
    pub metrics_to_collect: Vec<MetricType>,
    /// Alert configuration
    pub alert_config: AlertConfig,
    /// Data retention period
    pub data_retention: Duration,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            monitoring_frequency: Duration::from_secs(60),
            metrics_to_collect: vec![MetricType::All],
            alert_config: AlertConfig {
                alerts_enabled: true,
                alert_channels: vec![AlertChannel::Dashboard],
                alert_rules: vec![],
                escalation_policy: EscalationPolicy {
                    levels: vec![],
                    auto_acknowledge_timeout: Duration::from_secs(300),
                    max_level: 3,
                },
            },
            data_retention: Duration::from_secs(7 * 24 * 3600), // 7 days
        }
    }
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable alerts
    pub alerts_enabled: bool,
    /// Alert delivery channels
    pub alert_channels: Vec<AlertChannel>,
    /// Alert rules
    pub alert_rules: Vec<AlertRule>,
    /// Escalation policy
    pub escalation_policy: EscalationPolicy,
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Condition expression
    pub condition: String,
    /// Threshold value
    pub threshold: f64,
    /// Alert severity
    pub severity: super::types::AlertSeverity,
    /// Cooldown period
    pub cooldown: Duration,
}

/// Escalation policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicy {
    /// Escalation levels
    pub levels: Vec<EscalationLevel>,
    /// Auto-acknowledge timeout
    pub auto_acknowledge_timeout: Duration,
    /// Maximum escalation level
    pub max_level: usize,
}

/// Escalation level definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    /// Level number
    pub level: usize,
    /// Time before escalation
    pub delay: Duration,
    /// Notification targets
    pub targets: Vec<String>,
    /// Channels for this level
    pub channels: Vec<AlertChannel>,
}
