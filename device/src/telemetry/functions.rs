//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    backend_traits::BackendCapabilities, calibration::DeviceCalibration,
    topology::HardwareTopology, DeviceError, DeviceResult,
};
#[cfg(not(feature = "scirs2"))]
use fallback_scirs2::*;
use quantrs2_circuit::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::types::{
    Alert, AlertConfig, AlertSeverity, AnalyticsConfig, AnomalyDetectorConfig, AnomalyResult,
    ExportConfig, Metric, MetricConfig, MetricType, MonitoringConfig, QuantumTelemetrySystem,
    RetentionConfig, TelemetryConfig,
};

#[cfg(not(feature = "scirs2"))]
mod fallback_scirs2 {
    use scirs2_core::ndarray::{Array1, ArrayView1};
    pub fn mean(_data: &ArrayView1<f64>) -> Result<f64, String> {
        Ok(0.0)
    }
    pub fn std(_data: &ArrayView1<f64>, _ddof: i32) -> Result<f64, String> {
        Ok(1.0)
    }
    pub fn var(_data: &ArrayView1<f64>, _ddof: i32) -> Result<f64, String> {
        Ok(1.0)
    }
    pub fn percentile(_data: &ArrayView1<f64>, _q: f64) -> Result<f64, String> {
        Ok(0.0)
    }
}
/// Metric collector trait
pub trait MetricCollector: Send + Sync {
    /// Collect metrics
    fn collect(&self) -> DeviceResult<Vec<Metric>>;
    /// Get collector name
    fn name(&self) -> &str;
    /// Get collection interval
    fn interval(&self) -> Duration;
    /// Check if collector is enabled
    fn is_enabled(&self) -> bool;
}
/// Anomaly detector trait
pub trait AnomalyDetector: Send + Sync {
    /// Detect anomalies in metric data
    fn detect(&self, data: &[f64]) -> Vec<AnomalyResult>;
    /// Update detector with new data
    fn update(&mut self, data: &[f64]);
    /// Get detector configuration
    fn config(&self) -> AnomalyDetectorConfig;
}
/// Notification channel trait
pub trait NotificationChannel: Send + Sync {
    /// Send notification
    fn send(&self, alert: &Alert) -> DeviceResult<()>;
    /// Get channel name
    fn name(&self) -> &str;
    /// Check if channel is enabled
    fn is_enabled(&self) -> bool;
}
/// Create a default telemetry system
pub fn create_telemetry_system() -> QuantumTelemetrySystem {
    QuantumTelemetrySystem::new(TelemetryConfig::default())
}
/// Create a high-performance telemetry configuration
pub fn create_high_performance_telemetry_config() -> TelemetryConfig {
    TelemetryConfig {
        enabled: true,
        collection_interval: 10,
        enable_realtime_monitoring: true,
        enable_analytics: true,
        enable_alerting: true,
        retention_config: RetentionConfig {
            realtime_retention_hours: 48,
            historical_retention_days: 90,
            aggregated_retention_months: 24,
            enable_compression: true,
            archive_threshold_gb: 50.0,
        },
        metric_config: MetricConfig {
            enable_performance_metrics: true,
            enable_resource_metrics: true,
            enable_error_metrics: true,
            enable_cost_metrics: true,
            enable_custom_metrics: true,
            sampling_rate: 1.0,
            batch_size: 500,
        },
        monitoring_config: MonitoringConfig {
            dashboard_refresh_rate: 1,
            health_check_interval: 30,
            anomaly_sensitivity: 0.9,
            enable_trend_analysis: true,
            monitoring_targets: Vec::new(),
        },
        analytics_config: AnalyticsConfig {
            enable_statistical_analysis: true,
            enable_predictive_analytics: true,
            enable_correlation_analysis: true,
            processing_interval_minutes: 5,
            confidence_level: 0.99,
            prediction_horizon_hours: 72,
        },
        alert_config: AlertConfig {
            enable_email_alerts: true,
            enable_sms_alerts: true,
            enable_webhook_alerts: true,
            enable_slack_alerts: true,
            thresholds: HashMap::new(),
            escalation_rules: Vec::new(),
            suppression_rules: Vec::new(),
        },
        export_config: ExportConfig {
            enable_prometheus: true,
            enable_influxdb: true,
            enable_grafana: true,
            enable_custom_exports: true,
            export_endpoints: HashMap::new(),
        },
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_telemetry_config_default() {
        let config = TelemetryConfig::default();
        assert!(config.enabled);
        assert_eq!(config.collection_interval, 30);
        assert!(config.enable_realtime_monitoring);
        assert!(config.enable_analytics);
        assert!(config.enable_alerting);
    }
    #[test]
    fn test_metric_creation() {
        let metric = Metric {
            name: "test_metric".to_string(),
            value: 42.0,
            unit: "units".to_string(),
            metric_type: MetricType::Gauge,
            timestamp: SystemTime::now(),
            labels: HashMap::new(),
            metadata: HashMap::new(),
        };
        assert_eq!(metric.name, "test_metric");
        assert_eq!(metric.value, 42.0);
        assert_eq!(metric.metric_type, MetricType::Gauge);
    }
    #[test]
    fn test_telemetry_system_creation() {
        let config = TelemetryConfig::default();
        let system = QuantumTelemetrySystem::new(config);
    }
    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Critical);
        assert!(AlertSeverity::Critical < AlertSeverity::Emergency);
    }
    #[tokio::test]
    async fn test_telemetry_start_stop() {
        let config = TelemetryConfig::default();
        let system = QuantumTelemetrySystem::new(config);
        let start_result = system.start().await;
        assert!(start_result.is_ok());
        let stop_result = system.stop().await;
        assert!(stop_result.is_ok());
    }
}
