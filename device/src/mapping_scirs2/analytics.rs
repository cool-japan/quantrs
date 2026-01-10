//! Analytics and monitoring for mapping performance

use super::*;

/// Mapping analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingAnalyticsConfig {
    /// Enable performance analytics
    pub enable_analytics: bool,
    /// Level of analysis detail
    pub tracking_level: AnalysisDepth,
    /// Metrics to track
    pub metrics_to_track: Vec<TrackingMetric>,
    /// Anomaly detection configuration
    pub anomaly_detection: AnomalyDetectionConfig,
    /// Alert configuration
    pub alerting: AlertConfig,
    /// Reporting configuration
    pub reporting: ReportingConfig,
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    /// Enable anomaly detection
    pub enable_detection: bool,
    /// Detection method
    pub detection_method: AnomalyDetectionMethod,
    /// Anomaly threshold
    pub threshold: f64,
    /// Window size for analysis
    pub window_size: usize,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable alerting
    pub enable_alerts: bool,
    /// Severity threshold for alerts
    pub severity_threshold: f64,
    /// Notification methods
    pub notification_methods: Vec<NotificationMethod>,
    /// Cooldown period between alerts
    pub cooldown_period: Duration,
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Enable automated reporting
    pub enable_reporting: bool,
    /// Report generation frequency
    pub report_frequency: Duration,
    /// Report format
    pub report_format: ReportFormat,
    /// Report content configuration
    pub content_config: ReportContentConfig,
}

/// Report content configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportContentConfig {
    /// Include performance metrics
    pub include_performance_metrics: bool,
    /// Include trend analysis
    pub include_trend_analysis: bool,
    /// Include optimization recommendations
    pub include_recommendations: bool,
    /// Include visualizations
    pub include_visualizations: bool,
}

/// Real-time analytics results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeAnalyticsResult {
    /// Current performance metrics
    pub current_metrics: HashMap<String, f64>,
    /// Performance trends over time
    pub performance_trends: HashMap<String, Vec<(SystemTime, f64)>>,
    /// Detected anomalies
    pub anomalies: Vec<DetectedAnomaly>,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Quality assessments
    pub quality_assessments: Vec<QualityAssessment>,
}

/// Detected anomaly information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedAnomaly {
    /// Timestamp of detection
    pub timestamp: SystemTime,
    /// Anomaly type
    pub anomaly_type: String,
    /// Severity score
    pub severity: f64,
    /// Description
    pub description: String,
    /// Affected metrics
    pub affected_metrics: Vec<String>,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu_usage: f64,
    /// Memory utilization percentage
    pub memory_usage: f64,
    /// Disk I/O metrics
    pub disk_io: f64,
    /// Network utilization
    pub network_usage: f64,
    /// GPU utilization (if applicable)
    pub gpu_usage: Option<f64>,
}

/// Quality assessment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Assessment timestamp
    pub timestamp: SystemTime,
    /// Quality metric type
    pub metric_type: QualityMetricType,
    /// Quality score
    pub score: f64,
    /// Confidence in assessment
    pub confidence: f64,
    /// Comparison to baseline
    pub baseline_comparison: f64,
}

/// Optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendations {
    /// Algorithm recommendations
    pub algorithm_recommendations: Vec<AlgorithmRecommendation>,
    /// Parameter tuning suggestions
    pub parameter_suggestions: Vec<ParameterSuggestion>,
    /// Hardware-specific optimizations
    pub hardware_optimizations: Vec<HardwareOptimization>,
    /// Performance improvement predictions
    pub improvement_predictions: HashMap<String, f64>,
}

/// Algorithm recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmRecommendation {
    /// Recommended algorithm
    pub algorithm: String,
    /// Confidence score
    pub confidence: f64,
    /// Expected performance gain
    pub expected_gain: f64,
    /// Reasoning
    pub reasoning: String,
}

/// Parameter tuning suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSuggestion {
    /// Parameter name
    pub parameter: String,
    /// Suggested value range
    pub value_range: (f64, f64),
    /// Priority level
    pub priority: SuggestionPriority,
    /// Impact assessment
    pub impact: String,
}

/// Hardware-specific optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOptimization {
    /// Optimization type
    pub optimization_type: String,
    /// Target hardware component
    pub target_component: String,
    /// Optimization parameters
    pub parameters: HashMap<String, f64>,
    /// Expected benefit
    pub expected_benefit: f64,
    /// Implementation complexity
    pub complexity: String,
}
