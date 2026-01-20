//! Core types for enhanced monitoring system

use super::components::*;
use super::storage::*;
use super::types::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Comprehensive enhanced monitoring system for quantum networks
#[derive(Debug)]
pub struct EnhancedQuantumNetworkMonitor {
    /// Real-time metrics collection engine
    pub metrics_collector: Arc<RealTimeMetricsCollector>,
    /// Advanced analytics engine with ML capabilities
    pub analytics_engine: Arc<QuantumNetworkAnalyticsEngine>,
    /// Anomaly detection system
    pub anomaly_detector: Arc<QuantumAnomalyDetector>,
    /// Predictive analytics system
    pub predictive_analytics: Arc<QuantumNetworkPredictor>,
    /// Alert and notification system
    pub alert_system: Arc<QuantumNetworkAlertSystem>,
    /// Historical data manager
    pub historical_data_manager: Arc<QuantumHistoricalDataManager>,
    /// Performance optimization recommender
    pub optimization_recommender: Arc<QuantumOptimizationRecommender>,
    /// Real-time dashboard system
    pub dashboard_system: Arc<QuantumNetworkDashboard>,
    /// Configuration manager
    pub config_manager: Arc<EnhancedMonitoringConfig>,
}

/// Enhanced monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedMonitoringConfig {
    /// General monitoring settings
    pub general_settings: GeneralMonitoringSettings,
    /// Metrics collection configuration
    pub metrics_config: MetricsCollectionConfig,
    /// Analytics engine configuration
    pub analytics_config: AnalyticsEngineConfig,
    /// Anomaly detection configuration
    pub anomaly_detection_config: AnomalyDetectionConfig,
    /// Predictive analytics configuration
    pub predictive_config: PredictiveAnalyticsConfig,
    /// Alert system configuration
    pub alert_config: AlertSystemConfig,
    /// Storage and retention configuration
    pub storage_config: StorageConfig,
}
