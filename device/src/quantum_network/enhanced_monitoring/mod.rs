//! Enhanced Real-Time Monitoring and Analytics for Distributed Quantum Networks
//!
//! This module provides comprehensive real-time monitoring, analytics, and predictive
//! capabilities for distributed quantum computing networks, including ML-based anomaly
//! detection, performance prediction, and automated optimization recommendations.

pub mod components;
pub mod core;
pub mod implementations;
pub mod storage;
pub mod types;

// Re-export main types and errors
pub use types::{
    BandwidthThresholds, EnhancedMonitoringError, GeneralMonitoringSettings,
    HardwareMetricsSettings, MetricCategory, MetricType, MetricsCollectionConfig,
    NetworkMetricsSettings, PerformanceMonitoringLevel, QuantumMetricsSettings, Result,
};

// Re-export core types
pub use core::{EnhancedMonitoringConfig, EnhancedQuantumNetworkMonitor};

// Re-export component types
pub use components::{
    ActivationFunction, AlertRule, AlertRulesEngine, AlertSeverity, AnomalyDetectionModel,
    AnomalyModelType, AutoEscalationRule, CalibrationStatus, CollectionStatistics,
    ComparisonOperator, ComplexEventProcessingEngine, ConfidenceIntervals,
    DashboardNotificationType, DataQuality, DataQualityIndicators, EnsembleCombinationMethod,
    EnvironmentalConditions, EscalationCondition, EscalationLevel, EscalationSettings,
    FeatureExtractor, FeatureType, FrequencyLimits, LayerSpec, LayerType, MLAlgorithm,
    MeasurementContext, MetricCollectionScheduler, MetricDataPoint, MetricMetadata, MetricStream,
    MetricsAggregationEngine, MetricsBuffer, ModelAccuracyMetrics, NeuralNetworkArchitecture,
    NotificationChannel, NotificationSettings, OptimizationMethod, OptimizationRecommendation,
    ParameterOptimization, PerformancePredictionModel, PredictionModelType, QuantumAnomalyDetector,
    QuantumAnomalyModel, QuantumAnsatz, QuantumCondition, QuantumCorrelationAnalyzer,
    QuantumNetworkAlertSystem, QuantumNetworkAnalyticsEngine, QuantumNetworkPredictor,
    QuantumOptimizationAnalytics, QuantumPatternRecognition, QuantumPerformanceModeler,
    QuantumTrendAnalyzer, RealTimeAggregator, RealTimeAnalyticsProcessor, RealTimeMetricsCollector,
    RegularizationType, RuleCategory, RuleCondition, RuleMetadata, StatisticalMethod,
    StreamProcessingEngine, StreamStatistics, TimeSeriesModel, TrendDirection,
};

// Re-export storage types
pub use storage::{
    AlertSystemConfig, AnalyticsEngineConfig, AnomalyDetectionConfig, BackupConfig,
    BackupDestination, CompressionAlgorithm, CompressionConfig, CorrelationAnalysisConfig,
    CorrelationMethod, CrossValidationStrategy, DataCompressionSystem, DataExportSystem,
    DataQualityRequirements, DataRetentionManager, EscalationAction, EscalationConfig,
    EscalationPolicy, HistoricalAnalyticsEngine, MessageFormattingConfig, ModelSelectionCriteria,
    ModelSelectionMetric, ModelingAlgorithm, NotificationConfig, PatternRecognitionConfig,
    PatternType, PerformanceModelingConfig, PredictiveAnalyticsConfig,
    QuantumHistoricalDataManager, QuantumNetworkDashboard, QuantumOptimizationRecommender,
    RateLimitingConfig, RetentionPolicy, StorageBackendType, StorageConfig, TimeSeriesDatabase,
    TrainingRequirements, TrendAnalysisConfig, TrendMethod, ValidationMethod,
};

// Re-export implementation types
pub use implementations::{
    ActiveAlert, AnomalyAlertDispatcher, AnomalyResult, AnomalySeverity,
    ComponentPerformanceMetrics, ComponentState, ComponentStatus, CostBenefitAnalyzer,
    CustomRuleCompiler, DashboardStateManager, DataAccessControl, DataCompressionManager,
    DataIndexingSystem, DynamicThresholdManager, EffortLevel, ExpectedImprovement,
    FeatureProcessorRegistry, HistoricalDataStorage, ImpactSeverity, ImpactType,
    ImplementationEffort, MLAnomalyDetector, ModelPerformanceEvaluator, ModelTrainingScheduler,
    MonitoringStatus, NetworkTopologyPredictor, OptimizationPerformanceTracker,
    OptimizationRecommendationEngine, OptimizationRecommendationType, OverallStatus,
    PatternCorrelationEngine, PerformanceForecaster, PotentialImpact, QuantumAlertAnalyzer,
    QuantumAnomalyAnalyzer, QuantumOptimizationAdvisor, QuantumStatePredictor,
    RecommendationEffectivenessTracker, RecommendationPriority, RetentionPolicyManager,
    RiskAssessment, RiskLevel, RuleEvaluationEngine, RulePerformanceTracker, ScenarioAnalyzer,
    ThresholdDetector, UserInteractionHandler, VisualizationEngine,
};
