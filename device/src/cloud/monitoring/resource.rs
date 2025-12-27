//! Resource monitoring and usage tracking configuration.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Resource monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudResourceMonitoringConfig {
    /// Enable resource monitoring
    pub enabled: bool,
    /// Resource types to monitor
    pub resource_types: Vec<ResourceType>,
    /// Monitoring granularity
    pub granularity: MonitoringGranularity,
    /// Usage tracking
    pub usage_tracking: UsageTrackingConfig,
    /// Capacity planning
    pub capacity_planning: CapacityPlanningConfig,
    /// Resource optimization
    pub optimization: ResourceOptimizationConfig,
}

/// Resource types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    Compute,
    Storage,
    Network,
    Quantum,
    Database,
    Memory,
    GPU,
    FPGA,
    Custom(String),
}

/// Monitoring granularity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonitoringGranularity {
    PerSecond,
    PerMinute,
    PerHour,
    PerDay,
    Custom(Duration),
}

/// Usage tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageTrackingConfig {
    /// Track resource utilization
    pub track_utilization: bool,
    /// Track resource allocation
    pub track_allocation: bool,
    /// Track resource efficiency
    pub track_efficiency: bool,
    /// Usage analytics
    pub analytics: UsageAnalyticsConfig,
    /// Resource tagging
    pub tagging: ResourceTaggingConfig,
}

/// Usage analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageAnalyticsConfig {
    /// Enable analytics
    pub enabled: bool,
    /// Analytics methods
    pub methods: Vec<UsageAnalyticsMethod>,
    /// Analysis frequency
    pub frequency: Duration,
    /// Historical analysis
    pub historical_analysis: HistoricalAnalysisConfig,
    /// Predictive analytics
    pub predictive_analytics: PredictiveAnalyticsConfig,
}

/// Usage analytics methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsageAnalyticsMethod {
    TrendAnalysis,
    SeasonalAnalysis,
    AnomalyDetection,
    CapacityPlanning,
    EfficiencyAnalysis,
    CostAnalysis,
    ResourceRightSizing,
    Custom(String),
}

/// Historical analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalAnalysisConfig {
    /// Analysis window
    pub window: Duration,
    /// Comparison periods
    pub comparison_periods: Vec<Duration>,
    /// Statistical methods
    pub statistical_methods: Vec<StatisticalMethod>,
    /// Data retention
    pub retention: Duration,
}

/// Statistical methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatisticalMethod {
    Mean,
    Median,
    StandardDeviation,
    Percentiles,
    Correlation,
    Regression,
    TimeSeries,
    SeasonalDecomposition,
    Custom(String),
}

/// Predictive analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveAnalyticsConfig {
    /// Enable predictive analytics
    pub enabled: bool,
    /// Prediction horizon
    pub horizon: Duration,
    /// Models to use
    pub models: Vec<PredictiveModel>,
    /// Model accuracy threshold
    pub accuracy_threshold: f64,
}

/// Predictive models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictiveModel {
    LinearRegression,
    TimeSeriesForecasting,
    MachineLearning,
    NeuralNetwork,
    Custom(String),
}

/// Resource tagging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceTaggingConfig {
    /// Enable automatic tagging
    pub auto_tagging: bool,
    /// Required tags
    pub required_tags: Vec<String>,
    /// Tag categories
    pub categories: Vec<TagCategory>,
    /// Tag validation rules
    pub validation_rules: Vec<TagValidationRule>,
}

/// Tag categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagCategory {
    /// Category name
    pub name: String,
    /// Category description
    pub description: String,
    /// Allowed values
    pub allowed_values: Vec<String>,
    /// Required category
    pub required: bool,
}

/// Tag validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagValidationRule {
    /// Rule name
    pub name: String,
    /// Tag key pattern
    pub key_pattern: String,
    /// Value validation
    pub value_validation: ValueValidation,
}

/// Value validation types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValueValidation {
    Regex(String),
    Enum(Vec<String>),
    Range { min: f64, max: f64 },
    Custom(String),
}

/// Capacity planning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityPlanningConfig {
    /// Enable capacity planning
    pub enabled: bool,
    /// Planning horizon
    pub horizon: Duration,
    /// Growth models
    pub growth_models: Vec<GrowthModel>,
    /// Threshold-based planning
    pub threshold_planning: ThresholdPlanningConfig,
}

/// Growth models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrowthModel {
    Linear,
    Exponential,
    Seasonal,
    MachineLearning,
    Custom(String),
}

/// Threshold-based planning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdPlanningConfig {
    /// Utilization thresholds
    pub utilization_thresholds: std::collections::HashMap<ResourceType, f64>,
    /// Lead time for provisioning
    pub provisioning_lead_time: Duration,
    /// Buffer capacity percentage
    pub buffer_capacity: f64,
}

/// Resource optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOptimizationConfig {
    /// Enable optimization recommendations
    pub enabled: bool,
    /// Optimization strategies
    pub strategies: Vec<OptimizationStrategy>,
    /// Automation level
    pub automation_level: AutomationLevel,
    /// Optimization frequency
    pub frequency: Duration,
}

/// Optimization strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    RightSizing,
    Consolidation,
    ScheduledScaling,
    AutoScaling,
    ResourcePooling,
    Custom(String),
}

/// Automation levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutomationLevel {
    Manual,
    SemiAutomatic,
    Automatic,
    FullyAutomated,
}

impl Default for CloudResourceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            resource_types: vec![
                ResourceType::Compute,
                ResourceType::Storage,
                ResourceType::Network,
                ResourceType::Memory,
            ],
            granularity: MonitoringGranularity::PerMinute,
            usage_tracking: UsageTrackingConfig::default(),
            capacity_planning: CapacityPlanningConfig::default(),
            optimization: ResourceOptimizationConfig::default(),
        }
    }
}

impl Default for UsageTrackingConfig {
    fn default() -> Self {
        Self {
            track_utilization: true,
            track_allocation: true,
            track_efficiency: false,
            analytics: UsageAnalyticsConfig::default(),
            tagging: ResourceTaggingConfig::default(),
        }
    }
}

impl Default for UsageAnalyticsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            methods: vec![
                UsageAnalyticsMethod::TrendAnalysis,
                UsageAnalyticsMethod::AnomalyDetection,
            ],
            frequency: Duration::from_secs(3600), // hourly
            historical_analysis: HistoricalAnalysisConfig::default(),
            predictive_analytics: PredictiveAnalyticsConfig::default(),
        }
    }
}

impl Default for HistoricalAnalysisConfig {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(86400 * 30), // 30 days
            comparison_periods: vec![
                Duration::from_secs(86400),      // daily
                Duration::from_secs(86400 * 7),  // weekly
                Duration::from_secs(86400 * 30), // monthly
            ],
            statistical_methods: vec![StatisticalMethod::Mean, StatisticalMethod::Percentiles],
            retention: Duration::from_secs(86400 * 365), // 1 year
        }
    }
}

impl Default for PredictiveAnalyticsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            horizon: Duration::from_secs(86400 * 30), // 30 days
            models: vec![PredictiveModel::LinearRegression],
            accuracy_threshold: 0.8,
        }
    }
}

impl Default for ResourceTaggingConfig {
    fn default() -> Self {
        Self {
            auto_tagging: false,
            required_tags: vec!["Environment".to_string(), "Project".to_string()],
            categories: vec![],
            validation_rules: vec![],
        }
    }
}

impl Default for CapacityPlanningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            horizon: Duration::from_secs(86400 * 90), // 90 days
            growth_models: vec![GrowthModel::Linear],
            threshold_planning: ThresholdPlanningConfig::default(),
        }
    }
}

impl Default for ThresholdPlanningConfig {
    fn default() -> Self {
        let mut thresholds = std::collections::HashMap::new();
        thresholds.insert(ResourceType::Compute, 80.0);
        thresholds.insert(ResourceType::Storage, 85.0);
        thresholds.insert(ResourceType::Memory, 75.0);

        Self {
            utilization_thresholds: thresholds,
            provisioning_lead_time: Duration::from_secs(86400), // 1 day
            buffer_capacity: 20.0,
        }
    }
}

impl Default for ResourceOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategies: vec![OptimizationStrategy::RightSizing],
            automation_level: AutomationLevel::Manual,
            frequency: Duration::from_secs(86400 * 7), // weekly
        }
    }
}
