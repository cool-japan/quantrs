//! Marketplace Analytics Configuration Types

use serde::{Deserialize, Serialize};

/// Marketplace analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceAnalyticsConfig {
    /// Enable advanced analytics
    pub enable_analytics: bool,
    /// Usage analytics configuration
    pub usage_analytics: UsageAnalyticsConfig,
    /// Performance analytics settings
    pub performance_analytics: PerformanceAnalyticsConfig,
    /// Trend analysis configuration
    pub trend_analysis_config: TrendAnalysisConfig,
    /// Predictive analytics settings
    pub predictive_analytics: PredictiveAnalyticsConfig,
    /// Real-time monitoring
    pub realtime_monitoring: RealtimeMonitoringConfig,
}

/// Usage analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageAnalyticsConfig {
    pub track_usage: bool,
    pub metrics: Vec<String>,
}

/// Performance analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalyticsConfig {
    pub enable_performance_tracking: bool,
    pub performance_metrics: Vec<String>,
}

/// Predictive analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveAnalyticsConfig {
    pub enable_predictions: bool,
    pub prediction_models: Vec<String>,
}

/// Realtime monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeMonitoringConfig {
    pub enable_realtime: bool,
    pub monitoring_interval: u64,
}

// Re-export from discovery module
use super::discovery::TrendAnalysisConfig;
