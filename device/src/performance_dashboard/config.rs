//! Performance Dashboard Configuration Types

use serde::{Deserialize, Serialize};

/// Configuration for the Performance Analytics Dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Enable real-time monitoring
    pub enable_realtime_monitoring: bool,
    /// Data collection and analysis configuration
    pub data_config: DataCollectionConfig,
    /// Visualization configuration
    pub visualization_config: VisualizationConfig,
    /// Alerting and notification configuration
    pub alerting_config: AlertingConfig,
    /// Machine learning and prediction configuration
    pub ml_config: MLAnalyticsConfig,
    /// Performance optimization configuration
    pub optimization_config: DashboardOptimizationConfig,
    /// Export and reporting configuration
    pub reporting_config: ReportingConfig,
}

// Forward declarations for types that will be defined in other modules
use super::{
    alerting::AlertingConfig, data_collection::DataCollectionConfig,
    ml_analytics::MLAnalyticsConfig, optimization::DashboardOptimizationConfig,
    reporting::ReportingConfig, visualization::VisualizationConfig,
};
