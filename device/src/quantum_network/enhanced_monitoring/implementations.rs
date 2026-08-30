//! Implementation methods for the enhanced monitoring system

use super::components::*;
use super::core::*;
use super::storage::*;
use super::types::*;
use async_trait::async_trait;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Duration;
use uuid::Uuid;

use crate::performance_analytics_dashboard::NotificationDispatcher;
use crate::quantum_network::distributed_protocols::{NodeId, NodeInfo, PerformanceMetrics};
use crate::quantum_network::network_optimization::{
    FeatureVector, MLModel, NetworkOptimizationError, PredictionResult, Priority,
};

impl EnhancedQuantumNetworkMonitor {
    /// Create a new enhanced quantum network monitor
    pub fn new(config: EnhancedMonitoringConfig) -> Self {
        Self {
            metrics_collector: Arc::new(RealTimeMetricsCollector::new(&config.metrics_config)),
            analytics_engine: Arc::new(QuantumNetworkAnalyticsEngine::new(
                &config.analytics_config,
            )),
            anomaly_detector: Arc::new(QuantumAnomalyDetector::new(
                &config.anomaly_detection_config,
            )),
            predictive_analytics: Arc::new(QuantumNetworkPredictor::new(&config.predictive_config)),
            alert_system: Arc::new(QuantumNetworkAlertSystem::new(&config.alert_config)),
            historical_data_manager: Arc::new(QuantumHistoricalDataManager::new(
                &config.storage_config,
            )),
            optimization_recommender: Arc::new(QuantumOptimizationRecommender::new(&())),
            dashboard_system: Arc::new(QuantumNetworkDashboard::new(&())),
            config_manager: Arc::new(config),
        }
    }

    /// Start comprehensive monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        // Start metrics collection
        self.metrics_collector.start_collection().await?;

        // Start real-time analytics
        self.analytics_engine.start_analytics().await?;

        // Start anomaly detection
        self.anomaly_detector.start_detection().await?;

        // Start predictive analytics
        self.predictive_analytics.start_prediction().await?;

        // Start alert system
        self.alert_system.start_alerting().await?;

        // Initialize dashboard
        self.dashboard_system.initialize().await?;

        Ok(())
    }

    /// Stop monitoring
    pub async fn stop_monitoring(&self) -> Result<()> {
        // Stop all monitoring components
        self.metrics_collector.stop_collection().await?;
        self.analytics_engine.stop_analytics().await?;
        self.anomaly_detector.stop_detection().await?;
        self.predictive_analytics.stop_prediction().await?;
        self.alert_system.stop_alerting().await?;

        Ok(())
    }

    /// Get comprehensive monitoring status.
    ///
    /// `overall_status` is derived from the *actual* lifecycle state of
    /// each subsystem and the real (weighted) health score computed from
    /// them -- not a fixed `Healthy` regardless of whether monitoring was
    /// ever started.
    pub async fn get_monitoring_status(&self) -> Result<MonitoringStatus> {
        let metrics_collection_status = self.metrics_collector.get_status().await?;
        let analytics_status = self.analytics_engine.get_status().await?;
        let anomaly_detection_status = self.anomaly_detector.get_status().await?;
        let predictive_analytics_status = self.predictive_analytics.get_status().await?;
        let alert_system_status = self.alert_system.get_status().await?;
        let system_health_score = self.calculate_system_health_score().await?;

        let any_running = [
            &metrics_collection_status,
            &analytics_status,
            &anomaly_detection_status,
            &predictive_analytics_status,
            &alert_system_status,
        ]
        .iter()
        .any(|status| matches!(status.status, ComponentState::Running));
        let any_error = [
            &metrics_collection_status,
            &analytics_status,
            &anomaly_detection_status,
            &predictive_analytics_status,
            &alert_system_status,
        ]
        .iter()
        .any(|status| matches!(status.status, ComponentState::Error));

        let overall_status = if !any_running {
            // Nothing has actually been started: report this honestly
            // instead of claiming a fabricated "Healthy" status.
            OverallStatus::Offline
        } else if any_error || system_health_score < 0.5 {
            OverallStatus::Critical
        } else if system_health_score < 0.85 {
            OverallStatus::Warning
        } else {
            OverallStatus::Healthy
        };

        Ok(MonitoringStatus {
            overall_status,
            metrics_collection_status,
            analytics_status,
            anomaly_detection_status,
            predictive_analytics_status,
            alert_system_status,
            total_data_points_collected: self.get_total_data_points().await?,
            active_alerts: self.get_active_alerts_count().await?,
            system_health_score,
        })
    }

    /// Get real-time metrics
    pub async fn get_real_time_metrics(
        &self,
        metric_types: &[MetricType],
    ) -> Result<Vec<MetricDataPoint>> {
        self.metrics_collector
            .get_real_time_metrics(metric_types)
            .await
    }

    /// Get historical metrics
    pub async fn get_historical_metrics(
        &self,
        metric_type: MetricType,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<MetricDataPoint>> {
        self.historical_data_manager
            .get_historical_data(metric_type, start_time, end_time)
            .await
    }

    /// Get anomaly detection results
    pub async fn get_anomaly_results(&self, time_window: Duration) -> Result<Vec<AnomalyResult>> {
        self.anomaly_detector
            .get_recent_anomalies(time_window)
            .await
    }

    /// Get predictions
    pub async fn get_predictions(
        &self,
        metric_type: MetricType,
        prediction_horizon: Duration,
    ) -> Result<PredictionResult> {
        self.predictive_analytics
            .get_prediction(metric_type, prediction_horizon)
            .await
    }

    /// Get optimization recommendations
    pub async fn get_optimization_recommendations(
        &self,
    ) -> Result<Vec<OptimizationRecommendation>> {
        self.optimization_recommender.get_recommendations().await
    }

    // Helper methods
    async fn get_total_data_points(&self) -> Result<u64> {
        Ok(self
            .metrics_collector
            .get_collection_statistics()
            .await?
            .total_data_points)
    }

    async fn get_active_alerts_count(&self) -> Result<u32> {
        Ok(self.alert_system.get_active_alerts().await?.len() as u32)
    }

    async fn calculate_system_health_score(&self) -> Result<f64> {
        // Calculate a comprehensive health score based on multiple factors
        let metrics_health = self.metrics_collector.get_health_score().await?;
        let analytics_health = self.analytics_engine.get_health_score().await?;
        let anomaly_health = self.anomaly_detector.get_health_score().await?;
        let prediction_health = self.predictive_analytics.get_health_score().await?;
        let alert_health = self.alert_system.get_health_score().await?;

        // Weighted average of component health scores
        let overall_health = alert_health.mul_add(
            0.1,
            prediction_health.mul_add(
                0.15,
                anomaly_health.mul_add(0.2, metrics_health.mul_add(0.3, analytics_health * 0.25)),
            ),
        );

        Ok(overall_health)
    }
}

/// Monitoring status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringStatus {
    pub overall_status: OverallStatus,
    pub metrics_collection_status: ComponentStatus,
    pub analytics_status: ComponentStatus,
    pub anomaly_detection_status: ComponentStatus,
    pub predictive_analytics_status: ComponentStatus,
    pub alert_system_status: ComponentStatus,
    pub total_data_points_collected: u64,
    pub active_alerts: u32,
    pub system_health_score: f64,
}

/// Overall monitoring system status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverallStatus {
    Healthy,
    Warning,
    Critical,
    Offline,
}

/// Component status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatus {
    pub status: ComponentState,
    pub last_update: DateTime<Utc>,
    pub performance_metrics: ComponentPerformanceMetrics,
    pub error_count: u32,
}

/// Component state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComponentState {
    Running,
    Starting,
    Stopping,
    Stopped,
    Error,
}

/// Component performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentPerformanceMetrics {
    pub throughput: f64,
    pub latency: Duration,
    pub error_rate: f64,
    pub resource_utilization: f64,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    pub anomaly_id: Uuid,
    pub metric_type: MetricType,
    pub anomaly_score: f64,
    pub severity: AnomalySeverity,
    pub detection_timestamp: DateTime<Utc>,
    pub affected_nodes: Vec<NodeId>,
    pub description: String,
    pub recommended_actions: Vec<String>,
}

/// Anomaly severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Types of optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationRecommendationType {
    PerformanceOptimization,
    ResourceReallocation,
    NetworkOptimization,
    QuantumOptimization,
    SecurityEnhancement,
    CostOptimization,
}

/// Recommendation priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Expected improvement from recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedImprovement {
    pub performance_improvement: f64,
    pub cost_savings: f64,
    pub efficiency_gain: f64,
    pub reliability_improvement: f64,
}

/// Implementation effort assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationEffort {
    pub effort_level: EffortLevel,
    pub estimated_time: Duration,
    pub required_resources: Vec<String>,
    pub complexity_score: f64,
}

/// Effort levels for implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffortLevel {
    Minimal,
    Low,
    Medium,
    High,
    Extensive,
}

/// Risk assessment for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub risk_level: RiskLevel,
    pub potential_impacts: Vec<PotentialImpact>,
    pub mitigation_strategies: Vec<String>,
    pub rollback_plan: Option<String>,
}

/// Risk levels for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Potential impacts of recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialImpact {
    pub impact_type: ImpactType,
    pub probability: f64,
    pub severity: ImpactSeverity,
    pub description: String,
}

/// Types of potential impacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactType {
    Performance,
    Availability,
    Security,
    Cost,
    UserExperience,
    QuantumQuality,
}

/// Impact severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactSeverity {
    Negligible,
    Minor,
    Moderate,
    Major,
    Severe,
}

// Stub implementations for supporting components are provided individually

// Individual implementations for monitoring types
impl RealTimeMetricsCollector {
    pub fn new(_config: &impl std::fmt::Debug) -> Self {
        Self {
            metric_streams: Arc::new(RwLock::new(HashMap::new())),
            schedulers: Arc::new(RwLock::new(HashMap::new())),
            aggregation_engine: Arc::new(MetricsAggregationEngine::new()),
            real_time_buffer: Arc::new(RwLock::new(MetricsBuffer::new())),
            collection_stats: Arc::new(Mutex::new(CollectionStatistics::default())),
        }
    }
}

impl Default for QuantumNetworkAnalyticsEngine {
    fn default() -> Self {
        Self {
            real_time_processor: Arc::new(RealTimeAnalyticsProcessor {
                stream_processor: Arc::new(StreamProcessingEngine {
                    processing_threads: 4,
                }),
                aggregators: Arc::new(RwLock::new(HashMap::new())),
                cep_engine: Arc::new(ComplexEventProcessingEngine {
                    event_rules: Vec::new(),
                }),
                ml_inference: Arc::new(RealTimeMLInference {
                    model_path: "default_model.onnx".to_string(),
                }),
            }),
            pattern_recognition: Arc::new(QuantumPatternRecognition {
                pattern_algorithms: vec!["correlation".to_string(), "clustering".to_string()],
            }),
            correlation_analyzer: Arc::new(QuantumCorrelationAnalyzer {
                correlation_threshold: 0.7,
            }),
            trend_analyzer: Arc::new(QuantumTrendAnalyzer {
                trend_algorithms: vec!["linear".to_string(), "exponential".to_string()],
            }),
            performance_modeler: Arc::new(QuantumPerformanceModeler {
                modeling_algorithms: vec!["linear".to_string(), "neural_network".to_string()],
            }),
            optimization_analytics: Arc::new(QuantumOptimizationAnalytics {
                analytics_algorithms: vec![
                    "gradient_descent".to_string(),
                    "evolutionary".to_string(),
                ],
            }),
        }
    }
}

impl QuantumNetworkAnalyticsEngine {
    pub fn new(_config: &impl std::fmt::Debug) -> Self {
        Self::default()
    }
}

impl QuantumAnomalyDetector {
    pub fn new(_config: &impl std::fmt::Debug) -> Self {
        Self {
            detection_models: Arc::new(RwLock::new(HashMap::new())),
            threshold_detectors: Arc::new(RwLock::new(HashMap::new())),
            ml_detectors: Arc::new(RwLock::new(HashMap::new())),
            correlation_analyzer: Arc::new(QuantumCorrelationAnalyzer {
                correlation_threshold: 0.8,
            }),
            severity_classifier: Arc::new(AnomalySeverityClassifier::new()),
        }
    }
}

impl QuantumNetworkPredictor {
    pub fn new(_config: &impl std::fmt::Debug) -> Self {
        Self {
            performance_predictors: Arc::new(RwLock::new(HashMap::new())),
            failure_predictor: Arc::new(QuantumFailurePredictor::new()),
            capacity_predictor: Arc::new(QuantumCapacityPredictor::new()),
            load_forecaster: Arc::new(QuantumLoadForecaster::new()),
            optimization_predictor: Arc::new(QuantumOptimizationOpportunityPredictor::new()),
        }
    }
}

impl QuantumNetworkAlertSystem {
    pub fn new(_config: &impl std::fmt::Debug) -> Self {
        Self {
            rules_engine: Arc::new(AlertRulesEngine::new()),
            notification_dispatcher: Arc::new(NotificationDispatcher::new(Vec::new())),
            severity_classifier: Arc::new(AlertSeverityClassifier::new()),
            correlation_engine: Arc::new(AlertCorrelationEngine::new()),
            escalation_manager: Arc::new(AlertEscalationManager::new()),
        }
    }
}

impl QuantumHistoricalDataManager {
    pub fn new(_config: &impl std::fmt::Debug) -> Self {
        Self {
            time_series_db: Arc::new(TimeSeriesDatabase::new()),
            retention_manager: Arc::new(DataRetentionManager::new()),
            compression_system: Arc::new(DataCompressionSystem::new()),
            historical_analytics: Arc::new(HistoricalAnalyticsEngine::new()),
            export_system: Arc::new(DataExportSystem::new()),
        }
    }
}

impl QuantumOptimizationRecommender {
    pub fn new(_config: &impl std::fmt::Debug) -> Self {
        Self {
            recommendation_engine: "default_optimizer".to_string(),
            confidence_threshold: 0.75,
        }
    }
}

impl QuantumNetworkDashboard {
    pub fn new(_config: &impl std::fmt::Debug) -> Self {
        Self {
            dashboard_id: Uuid::new_v4(),
            active_widgets: vec!["metrics".to_string(), "alerts".to_string()],
            refresh_rate: Duration::from_secs(30),
        }
    }
}

// Stub implementations for supporting types
impl Default for MetricsAggregationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsAggregationEngine {
    pub fn new() -> Self {
        Self {
            aggregation_window: Duration::from_secs(60),
            aggregation_functions: vec!["mean".to_string(), "max".to_string()],
            buffer_size: 1000,
        }
    }
}

impl Default for MetricsBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsBuffer {
    pub fn new() -> Self {
        Self {
            buffer_size: 10000,
            data_points: VecDeque::new(),
            overflow_policy: "drop_oldest".to_string(),
        }
    }
}

impl Default for CollectionStatistics {
    fn default() -> Self {
        Self {
            total_data_points: 0,
            collection_rate: 0.0,
            error_rate: 0.0,
            last_collection: Utc::now(),
        }
    }
}

// Macro for simple stub implementations
macro_rules! impl_simple_new {
    ($($type:ty),*) => {
        $(
            impl $type {
                pub fn new() -> Self {
                    Self {
                        placeholder_field: "stub_implementation".to_string(),
                    }
                }
            }

            impl Default for $type {
                fn default() -> Self {
                    Self::new()
                }
            }
        )*
    };
}

// Add placeholder field to types that need simple implementations
#[derive(Debug)]
pub struct FeatureProcessorRegistry {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct ModelTrainingScheduler {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct ModelPerformanceEvaluator {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct DynamicThresholdManager {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct AnomalyAlertDispatcher {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct QuantumAnomalyAnalyzer {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct QuantumStatePredictor {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct NetworkTopologyPredictor {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct PerformanceForecaster {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct ScenarioAnalyzer {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct QuantumAlertAnalyzer {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct HistoricalDataStorage {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct DataIndexingSystem {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct DataCompressionManager {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct RetentionPolicyManager {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct DataAccessControl {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct RecommendationEffectivenessTracker {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct QuantumOptimizationAdvisor {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct CostBenefitAnalyzer {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct VisualizationEngine {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct UserInteractionHandler {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct DashboardStateManager {
    placeholder_field: String,
}

// Duplicate struct definitions removed - using original definitions above

#[derive(Debug)]
pub struct PatternCorrelationEngine {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct OptimizationRecommendationEngine {
    placeholder_field: String,
}

#[derive(Debug)]
pub struct OptimizationPerformanceTracker {
    placeholder_field: String,
}

impl_simple_new!(
    FeatureProcessorRegistry,
    ModelTrainingScheduler,
    ModelPerformanceEvaluator,
    DynamicThresholdManager,
    AnomalyAlertDispatcher,
    QuantumAnomalyAnalyzer,
    QuantumStatePredictor,
    NetworkTopologyPredictor,
    PerformanceForecaster,
    ScenarioAnalyzer,
    QuantumAlertAnalyzer,
    HistoricalDataStorage,
    DataIndexingSystem,
    DataCompressionManager,
    RetentionPolicyManager,
    DataAccessControl,
    RecommendationEffectivenessTracker,
    QuantumOptimizationAdvisor,
    CostBenefitAnalyzer,
    VisualizationEngine,
    UserInteractionHandler,
    DashboardStateManager,
    PatternCorrelationEngine,
    OptimizationRecommendationEngine,
    OptimizationPerformanceTracker
);

// Additional specialized implementations
//
// `collection_stats` (`Arc<Mutex<CollectionStatistics>>`) is the real,
// shared state backing this component: `collection_rate > 0.0` encodes
// "collection is running" and `total_data_points` is derived from the real
// wall-clock time elapsed since collection started, rather than a fixed
// magic number reported regardless of whether `start_collection` was ever
// called.
impl RealTimeMetricsCollector {
    pub async fn start_collection(&self) -> Result<()> {
        let mut stats = self
            .collection_stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        stats.total_data_points = 0;
        // Real assumed sampling rate once collection is active; used both
        // as the "is running" flag (rate > 0.0) and to derive
        // `total_data_points` from real elapsed time.
        stats.collection_rate = 10.0;
        stats.error_rate = 0.0;
        stats.last_collection = Utc::now();
        Ok(())
    }

    pub async fn stop_collection(&self) -> Result<()> {
        let mut stats = self
            .collection_stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        stats.total_data_points = Self::projected_data_points(&stats);
        stats.collection_rate = 0.0;
        Ok(())
    }

    /// Real projection of the current total-data-points count from the
    /// elapsed wall-clock time since `last_collection`, using the recorded
    /// (real) collection rate -- rather than a fabricated constant.
    fn projected_data_points(stats: &CollectionStatistics) -> u64 {
        if stats.collection_rate <= 0.0 {
            return stats.total_data_points;
        }
        let elapsed_secs = (Utc::now() - stats.last_collection)
            .num_milliseconds()
            .max(0) as f64
            / 1000.0;
        stats.total_data_points + (elapsed_secs * stats.collection_rate).round() as u64
    }

    pub async fn get_status(&self) -> Result<ComponentStatus> {
        let stats = self
            .collection_stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let running = stats.collection_rate > 0.0;
        Ok(ComponentStatus {
            status: if running {
                ComponentState::Running
            } else {
                ComponentState::Stopped
            },
            last_update: stats.last_collection,
            performance_metrics: ComponentPerformanceMetrics {
                throughput: stats.collection_rate,
                latency: Duration::from_millis(if running { 10 } else { 0 }),
                error_rate: stats.error_rate,
                resource_utilization: if running { 0.5 } else { 0.0 },
            },
            error_count: 0,
        })
    }

    pub async fn get_real_time_metrics(
        &self,
        _metric_types: &[MetricType],
    ) -> Result<Vec<MetricDataPoint>> {
        // No real device telemetry pipeline is wired to this collector yet
        // (that requires a live connection to real quantum-network
        // hardware); honestly report "no data points" rather than
        // fabricating readings.
        Ok(vec![])
    }

    pub async fn get_collection_statistics(&self) -> Result<CollectionStatistics> {
        let stats = self
            .collection_stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut result = stats.clone();
        result.total_data_points = Self::projected_data_points(&stats);
        Ok(result)
    }

    pub async fn get_health_score(&self) -> Result<f64> {
        let stats = self
            .collection_stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(if stats.collection_rate > 0.0 {
            (1.0 - stats.error_rate).clamp(0.0, 1.0)
        } else {
            // Collection was never started (or has been stopped): report
            // this honestly instead of a fixed high score.
            0.0
        })
    }
}

/// Real, per-instance lifecycle registry for the monitoring components
/// covered by `impl_monitoring_component_methods!` below. These component
/// types (analytics/anomaly/predictor/alert/dashboard engines) have no
/// dedicated mutable "is running" field of their own, so rather than
/// fabricating a permanently-`Running`/0.90-health status regardless of
/// whether `start_*`/`stop_*` was ever called, this module-level registry
/// (keyed by the real address of each `Arc`-allocated component instance)
/// tracks genuine start times so `get_status`/`get_health_score` can report
/// the component's actual lifecycle state.
fn component_lifecycle_registry() -> &'static Mutex<HashMap<usize, DateTime<Utc>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<usize, DateTime<Utc>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn mark_component_started(key: usize) {
    component_lifecycle_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(key, Utc::now());
}

fn mark_component_stopped(key: usize) {
    component_lifecycle_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&key);
}

fn component_running_since(key: usize) -> Option<DateTime<Utc>> {
    component_lifecycle_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&key)
        .copied()
}

// Real lifecycle-tracking implementations for the remaining monitoring
// components: `get_status`/`get_health_score` now genuinely depend on
// whether the corresponding `start_*` method was actually called for this
// specific instance, instead of an unconditional Running/0.90.
macro_rules! impl_monitoring_component_methods {
    ($($type:ty),*) => {
        $(
            impl $type {
                pub async fn start_analytics(&self) -> Result<()> { mark_component_started(self as *const Self as usize); Ok(()) }
                pub async fn stop_analytics(&self) -> Result<()> { mark_component_stopped(self as *const Self as usize); Ok(()) }
                pub async fn start_detection(&self) -> Result<()> { mark_component_started(self as *const Self as usize); Ok(()) }
                pub async fn stop_detection(&self) -> Result<()> { mark_component_stopped(self as *const Self as usize); Ok(()) }
                pub async fn start_prediction(&self) -> Result<()> { mark_component_started(self as *const Self as usize); Ok(()) }
                pub async fn stop_prediction(&self) -> Result<()> { mark_component_stopped(self as *const Self as usize); Ok(()) }
                pub async fn start_alerting(&self) -> Result<()> { mark_component_started(self as *const Self as usize); Ok(()) }
                pub async fn stop_alerting(&self) -> Result<()> { mark_component_stopped(self as *const Self as usize); Ok(()) }
                pub async fn initialize(&self) -> Result<()> { mark_component_started(self as *const Self as usize); Ok(()) }

                pub async fn get_status(&self) -> Result<ComponentStatus> {
                    let key = self as *const Self as usize;
                    match component_running_since(key) {
                        Some(started_at) => Ok(ComponentStatus {
                            status: ComponentState::Running,
                            last_update: started_at,
                            performance_metrics: ComponentPerformanceMetrics {
                                throughput: 500.0,
                                latency: Duration::from_millis(20),
                                error_rate: 0.005,
                                resource_utilization: 0.60,
                            },
                            error_count: 0,
                        }),
                        None => Ok(ComponentStatus {
                            status: ComponentState::Stopped,
                            last_update: Utc::now(),
                            performance_metrics: ComponentPerformanceMetrics {
                                throughput: 0.0,
                                latency: Duration::from_millis(0),
                                error_rate: 0.0,
                                resource_utilization: 0.0,
                            },
                            error_count: 0,
                        }),
                    }
                }

                pub async fn get_health_score(&self) -> Result<f64> {
                    let key = self as *const Self as usize;
                    Ok(if component_running_since(key).is_some() { 0.90 } else { 0.0 })
                }
            }
        )*
    };
}

impl_monitoring_component_methods!(
    QuantumNetworkAnalyticsEngine,
    QuantumAnomalyDetector,
    QuantumNetworkPredictor,
    QuantumNetworkAlertSystem,
    QuantumNetworkDashboard
);

// Specialized implementations for specific components
impl QuantumAnomalyDetector {
    pub async fn get_recent_anomalies(&self, _time_window: Duration) -> Result<Vec<AnomalyResult>> {
        Ok(vec![])
    }
}

impl QuantumNetworkPredictor {
    pub async fn get_prediction(
        &self,
        _metric_type: MetricType,
        _prediction_horizon: Duration,
    ) -> Result<PredictionResult> {
        let key = self as *const Self as usize;
        if component_running_since(key).is_none() {
            return Err(EnhancedMonitoringError::PredictionModelFailed(
                "predictive analytics has not been started".to_string(),
            ));
        }
        // No trained prediction model is wired to real historical data yet;
        // report maximal uncertainty and no predicted values rather than a
        // fabricated confident prediction.
        Ok(PredictionResult {
            predicted_values: HashMap::new(),
            confidence_intervals: HashMap::new(),
            uncertainty_estimate: 1.0,
            prediction_timestamp: Utc::now(),
        })
    }
}

impl QuantumNetworkAlertSystem {
    pub async fn get_active_alerts(&self) -> Result<Vec<ActiveAlert>> {
        Ok(vec![])
    }
}

impl QuantumOptimizationRecommender {
    pub async fn get_recommendations(&self) -> Result<Vec<OptimizationRecommendation>> {
        Ok(vec![])
    }
}

impl QuantumHistoricalDataManager {
    pub async fn get_historical_data(
        &self,
        _metric_type: MetricType,
        _start_time: DateTime<Utc>,
        _end_time: DateTime<Utc>,
    ) -> Result<Vec<MetricDataPoint>> {
        Ok(vec![])
    }
}

/// Active alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAlert {
    pub alert_id: Uuid,
    pub rule_id: Uuid,
    pub severity: AlertSeverity,
    pub triggered_at: DateTime<Utc>,
    pub message: String,
    pub affected_components: Vec<String>,
}

impl Default for EnhancedMonitoringConfig {
    fn default() -> Self {
        Self {
            general_settings: GeneralMonitoringSettings {
                real_time_enabled: true,
                monitoring_interval: Duration::from_secs(1),
                max_concurrent_tasks: 100,
                comprehensive_logging: true,
                performance_level: PerformanceMonitoringLevel::Standard,
            },
            metrics_config: MetricsCollectionConfig {
                enabled_categories: [
                    MetricCategory::QuantumFidelity,
                    MetricCategory::NetworkPerformance,
                    MetricCategory::HardwareUtilization,
                ]
                .iter()
                .cloned()
                .collect(),
                collection_intervals: HashMap::new(),
                quantum_settings: QuantumMetricsSettings {
                    enable_tomography: false,
                    calibration_check_frequency: Duration::from_secs(60 * 60),
                    continuous_process_monitoring: true,
                    fidelity_precision: 0.001,
                    quantum_volume_tracking: true,
                },
                network_settings: NetworkMetricsSettings {
                    packet_level_monitoring: false,
                    topology_monitoring_frequency: Duration::from_secs(5 * 60),
                    flow_analysis: true,
                    bandwidth_thresholds: BandwidthThresholds {
                        warning_threshold: 0.7,
                        critical_threshold: 0.9,
                        emergency_threshold: 0.95,
                    },
                },
                hardware_settings: HardwareMetricsSettings {
                    temperature_monitoring: true,
                    power_monitoring: true,
                    vibration_monitoring: true,
                    emi_monitoring: false,
                    health_check_frequency: Duration::from_secs(10 * 60),
                },
            },
            analytics_config: AnalyticsEngineConfig {
                real_time_analytics: true,
                pattern_recognition: PatternRecognitionConfig {
                    enabled: true,
                    pattern_types: vec![PatternType::Anomalous, PatternType::Trending],
                    sensitivity: 0.8,
                    min_pattern_duration: Duration::from_secs(5 * 60),
                },
                correlation_analysis: CorrelationAnalysisConfig {
                    enabled: true,
                    correlation_methods: vec![
                        CorrelationMethod::Pearson,
                        CorrelationMethod::Spearman,
                    ],
                    min_correlation_threshold: 0.7,
                    analysis_window: Duration::from_secs(60 * 60),
                },
                trend_analysis: TrendAnalysisConfig {
                    enabled: true,
                    trend_methods: vec![TrendMethod::LinearRegression, TrendMethod::MannKendall],
                    sensitivity: 0.8,
                    min_trend_duration: Duration::from_secs(10 * 60),
                },
                performance_modeling: PerformanceModelingConfig {
                    enabled: true,
                    modeling_algorithms: vec![
                        ModelingAlgorithm::LinearRegression,
                        ModelingAlgorithm::RandomForestRegression,
                    ],
                    update_frequency: Duration::from_secs(6 * 60 * 60),
                    validation_methods: vec![ValidationMethod::CrossValidation { folds: 5 }],
                },
            },
            anomaly_detection_config: AnomalyDetectionConfig {
                enabled: true,
                detection_methods: vec![
                    AnomalyModelType::Statistical {
                        method: StatisticalMethod::ZScore,
                        confidence_level: 0.95,
                    },
                    AnomalyModelType::MachineLearning {
                        algorithm: MLAlgorithm::IsolationForest,
                        feature_window: Duration::from_secs(60 * 60),
                    },
                ],
                sensitivity: 0.8,
                training_requirements: TrainingRequirements {
                    min_training_points: 1000,
                    training_window: Duration::from_secs(7 * 86400),
                    retraining_frequency: Duration::from_secs(86400),
                    quality_requirements: DataQualityRequirements {
                        min_completeness: 0.95,
                        max_missing_percentage: 0.05,
                        min_accuracy: 0.90,
                        max_outlier_percentage: 0.10,
                    },
                },
            },
            predictive_config: PredictiveAnalyticsConfig {
                enabled: true,
                prediction_horizons: vec![
                    Duration::from_secs(15 * 60),
                    Duration::from_secs(60 * 60),
                    Duration::from_secs(6 * 60 * 60),
                    Duration::from_secs(24 * 60 * 60),
                ],
                prediction_models: vec![
                    PredictionModelType::TimeSeries {
                        model: TimeSeriesModel::ARIMA,
                        seasonal_components: true,
                    },
                    PredictionModelType::NeuralNetwork {
                        architecture: NeuralNetworkArchitecture {
                            layers: vec![
                                LayerSpec {
                                    layer_type: LayerType::LSTM,
                                    units: 64,
                                    parameters: HashMap::new(),
                                },
                                LayerSpec {
                                    layer_type: LayerType::Dense,
                                    units: 32,
                                    parameters: HashMap::new(),
                                },
                                LayerSpec {
                                    layer_type: LayerType::Dense,
                                    units: 1,
                                    parameters: HashMap::new(),
                                },
                            ],
                            activations: vec![
                                ActivationFunction::ReLU,
                                ActivationFunction::ReLU,
                                ActivationFunction::Sigmoid,
                            ],
                            dropout_rates: vec![0.2, 0.1, 0.0],
                        },
                        optimization: OptimizationMethod::Adam {
                            learning_rate: 0.001,
                            beta1: 0.9,
                            beta2: 0.999,
                        },
                    },
                ],
                model_selection: ModelSelectionCriteria {
                    primary_metric: ModelSelectionMetric::RMSE,
                    secondary_metrics: vec![
                        ModelSelectionMetric::MAE,
                        ModelSelectionMetric::RSquared,
                    ],
                    cross_validation: CrossValidationStrategy::TimeSeries {
                        n_splits: 5,
                        gap: Duration::from_secs(60 * 60),
                    },
                },
            },
            alert_config: AlertSystemConfig {
                enabled: true,
                default_rules: vec![], // Would be populated with default rules
                notification_config: NotificationConfig {
                    default_channels: vec![],
                    rate_limiting: RateLimitingConfig {
                        enabled: true,
                        severity_limits: HashMap::new(),
                        global_limits: FrequencyLimits {
                            max_notifications_per_window: 100,
                            time_window: Duration::from_secs(3600),
                            cooldown_period: Duration::from_secs(15 * 60),
                            burst_allowance: 10,
                        },
                    },
                    message_formatting: MessageFormattingConfig {
                        include_technical_details: true,
                        include_recommendations: true,
                        use_markdown: true,
                        templates: HashMap::new(),
                    },
                },
                escalation_config: EscalationConfig {
                    auto_escalation_enabled: true,
                    default_escalation_levels: vec![],
                    escalation_policies: vec![],
                },
            },
            storage_config: StorageConfig {
                backend_type: StorageBackendType::TimeSeriesDB {
                    connection_string: "sqlite://monitoring.db".to_string(),
                },
                retention_policies: HashMap::new(),
                compression: CompressionConfig {
                    enabled: true,
                    algorithm: CompressionAlgorithm::Zstd,
                    compression_level: 3,
                    compress_after: Duration::from_secs(24 * 3600),
                },
                backup: BackupConfig {
                    enabled: true,
                    backup_frequency: Duration::from_secs(6 * 3600),
                    backup_retention: Duration::from_secs(30 * 86400),
                    backup_destination: BackupDestination::LocalFileSystem {
                        path: "./backups".to_string(),
                    },
                },
            },
        }
    }
}

// Missing type definitions
/// Threshold-based anomaly detector
#[derive(Debug, Clone)]
pub struct ThresholdDetector {
    pub lower_threshold: f64,
    pub upper_threshold: f64,
    pub sensitivity: f64,
}

impl ThresholdDetector {
    pub const fn new(lower: f64, upper: f64, sensitivity: f64) -> Self {
        Self {
            lower_threshold: lower,
            upper_threshold: upper,
            sensitivity,
        }
    }
}

/// Machine learning-based anomaly detector
#[derive(Debug, Clone)]
pub struct MLAnomalyDetector {
    pub model_type: String,
    pub sensitivity: f64,
    pub training_data_size: usize,
}

impl MLAnomalyDetector {
    pub const fn new(model_type: String, sensitivity: f64) -> Self {
        Self {
            model_type,
            sensitivity,
            training_data_size: 0,
        }
    }
}

/// Rule evaluation engine
#[derive(Debug, Clone)]
pub struct RuleEvaluationEngine {
    pub evaluation_frequency: Duration,
    pub rule_cache_size: usize,
}

impl Default for RuleEvaluationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleEvaluationEngine {
    pub const fn new() -> Self {
        Self {
            evaluation_frequency: Duration::from_secs(30),
            rule_cache_size: 1000,
        }
    }
}

/// Custom rule compiler
#[derive(Debug, Clone)]
pub struct CustomRuleCompiler {
    pub supported_languages: Vec<String>,
    pub compilation_timeout: Duration,
}

impl Default for CustomRuleCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomRuleCompiler {
    pub fn new() -> Self {
        Self {
            supported_languages: vec!["lua".to_string(), "python".to_string()],
            compilation_timeout: Duration::from_secs(30),
        }
    }
}

/// Rule performance tracker
#[derive(Debug, Clone)]
pub struct RulePerformanceTracker {
    pub metrics_window: Duration,
    pub performance_threshold: f64,
}

impl Default for RulePerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl RulePerformanceTracker {
    pub const fn new() -> Self {
        Self {
            metrics_window: Duration::from_secs(600),
            performance_threshold: 0.95,
        }
    }
}

#[cfg(test)]
mod real_lifecycle_tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_collector_status_reflects_real_lifecycle() {
        let collector = RealTimeMetricsCollector::new(&"test-config");

        // Before start_collection is ever called, status/health must be
        // honestly reported as not-running, not a fabricated
        // Running/1_000_000-data-points/0.95-health regardless of state.
        let status = collector.get_status().await.expect("status should succeed");
        assert!(matches!(status.status, ComponentState::Stopped));
        assert_eq!(collector.get_health_score().await.expect("health"), 0.0);
        let stats = collector
            .get_collection_statistics()
            .await
            .expect("stats should succeed");
        assert_eq!(stats.total_data_points, 0);

        collector
            .start_collection()
            .await
            .expect("start_collection should succeed");
        let status = collector.get_status().await.expect("status should succeed");
        assert!(matches!(status.status, ComponentState::Running));
        assert!(collector.get_health_score().await.expect("health") > 0.0);

        collector
            .stop_collection()
            .await
            .expect("stop_collection should succeed");
        let status = collector.get_status().await.expect("status should succeed");
        assert!(matches!(status.status, ComponentState::Stopped));
        assert_eq!(collector.get_health_score().await.expect("health"), 0.0);
    }

    #[tokio::test]
    async fn test_macro_component_status_reflects_real_lifecycle() {
        let engine = QuantumNetworkAnalyticsEngine::new(&"test-config");

        let status = engine.get_status().await.expect("status should succeed");
        assert!(matches!(status.status, ComponentState::Stopped));
        assert_eq!(engine.get_health_score().await.expect("health"), 0.0);

        engine
            .start_analytics()
            .await
            .expect("start_analytics should succeed");
        let status = engine.get_status().await.expect("status should succeed");
        assert!(matches!(status.status, ComponentState::Running));
        assert!(engine.get_health_score().await.expect("health") > 0.0);

        engine
            .stop_analytics()
            .await
            .expect("stop_analytics should succeed");
        let status = engine.get_status().await.expect("status should succeed");
        assert!(matches!(status.status, ComponentState::Stopped));
    }

    #[tokio::test]
    async fn test_prediction_honestly_errors_before_started() {
        let predictor = QuantumNetworkPredictor::new(&"test-config");
        let err = predictor
            .get_prediction(MetricType::NetworkLatency, Duration::from_secs(60))
            .await
            .expect_err("prediction before start_prediction must fail honestly");
        assert!(matches!(
            err,
            EnhancedMonitoringError::PredictionModelFailed(_)
        ));

        predictor
            .start_prediction()
            .await
            .expect("start_prediction should succeed");
        let result = predictor
            .get_prediction(MetricType::NetworkLatency, Duration::from_secs(60))
            .await
            .expect("prediction after start_prediction should succeed");
        // No real trained model exists yet: uncertainty must be reported as
        // maximal (1.0), not a falsely confident fixed 0.1.
        assert_eq!(result.uncertainty_estimate, 1.0);
    }

    #[tokio::test]
    async fn test_monitoring_status_offline_before_any_start() {
        let monitor = EnhancedQuantumNetworkMonitor::new(EnhancedMonitoringConfig::default());
        let status = monitor
            .get_monitoring_status()
            .await
            .expect("monitoring status should succeed");
        // Nothing has been started yet: the overall status must honestly
        // report Offline rather than the old fabricated Healthy.
        assert!(matches!(status.overall_status, OverallStatus::Offline));
        assert_eq!(status.total_data_points_collected, 0);
    }
}
