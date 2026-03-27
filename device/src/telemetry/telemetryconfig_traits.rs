//! # TelemetryConfig - Trait Implementations
//!
//! This module contains trait implementations for `TelemetryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(not(feature = "scirs2"))]
use fallback_scirs2::*;
use quantrs2_circuit::prelude::*;
use std::collections::{BTreeMap, HashMap, VecDeque};

use super::types::{
    AlertConfig, AnalyticsConfig, ExportConfig, MetricConfig, MonitoringConfig, RetentionConfig,
    TelemetryConfig,
};

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval: 30,
            enable_realtime_monitoring: true,
            enable_analytics: true,
            enable_alerting: true,
            retention_config: RetentionConfig {
                realtime_retention_hours: 24,
                historical_retention_days: 30,
                aggregated_retention_months: 12,
                enable_compression: true,
                archive_threshold_gb: 10.0,
            },
            metric_config: MetricConfig {
                enable_performance_metrics: true,
                enable_resource_metrics: true,
                enable_error_metrics: true,
                enable_cost_metrics: true,
                enable_custom_metrics: true,
                sampling_rate: 1.0,
                batch_size: 100,
            },
            monitoring_config: MonitoringConfig {
                dashboard_refresh_rate: 5,
                health_check_interval: 60,
                anomaly_sensitivity: 0.8,
                enable_trend_analysis: true,
                monitoring_targets: Vec::new(),
            },
            analytics_config: AnalyticsConfig {
                enable_statistical_analysis: true,
                enable_predictive_analytics: true,
                enable_correlation_analysis: true,
                processing_interval_minutes: 15,
                confidence_level: 0.95,
                prediction_horizon_hours: 24,
            },
            alert_config: AlertConfig {
                enable_email_alerts: true,
                enable_sms_alerts: false,
                enable_webhook_alerts: true,
                enable_slack_alerts: false,
                thresholds: HashMap::new(),
                escalation_rules: Vec::new(),
                suppression_rules: Vec::new(),
            },
            export_config: ExportConfig {
                enable_prometheus: true,
                enable_influxdb: false,
                enable_grafana: false,
                enable_custom_exports: false,
                export_endpoints: HashMap::new(),
            },
        }
    }
}
