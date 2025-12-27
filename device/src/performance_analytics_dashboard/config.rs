//! Configuration Management for Performance Analytics Dashboard
//!
//! This module contains all configuration structures and types for the dashboard,
//! including analytics, visualization, alerting, and export configurations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Configuration for the Performance Analytics Dashboard
#[derive(Debug, Clone)]
pub struct PerformanceDashboardConfig {
    /// Enable real-time monitoring
    pub enable_realtime_monitoring: bool,
    /// Data collection interval in seconds
    pub collection_interval: u64,
    /// Historical data retention in days
    pub retention_days: u32,
    /// Dashboard refresh rate in seconds
    pub dashboard_refresh_rate: u64,
    /// Analytics configuration
    pub analytics_config: AnalyticsConfig,
    /// Visualization configuration
    pub visualization_config: VisualizationConfig,
    /// Alert configuration
    pub alert_config: AlertConfig,
    /// Export configuration
    pub export_config: ExportConfig,
    /// Prediction configuration
    pub prediction_config: PredictionConfig,
}

/// Advanced analytics configuration
#[derive(Debug, Clone)]
pub struct AnalyticsConfig {
    /// Enable statistical analysis
    pub enable_statistical_analysis: bool,
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
    /// Enable correlation analysis
    pub enable_correlation_analysis: bool,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Enable performance modeling
    pub enable_performance_modeling: bool,
    /// Statistical confidence level
    pub confidence_level: f64,
    /// Anomaly detection sensitivity
    pub anomaly_sensitivity: f64,
    /// Trend analysis window size
    pub trend_window_size: usize,
}

/// Visualization configuration
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Enable interactive charts
    pub enable_interactive_charts: bool,
    /// Chart types to display
    pub chart_types: Vec<ChartType>,
    /// Dashboard layout
    pub dashboard_layout: DashboardLayout,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Aggregation levels
    pub aggregation_levels: Vec<AggregationLevel>,
    /// Custom visualizations
    pub custom_visualizations: Vec<CustomVisualization>,
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Enable alerting
    pub enable_alerts: bool,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, AlertThreshold>,
    /// Notification channels
    pub notification_channels: Vec<NotificationChannel>,
    /// Alert suppression rules
    pub suppression_rules: Vec<SuppressionRule>,
    /// Escalation policies
    pub escalation_policies: Vec<EscalationPolicy>,
}

/// Export configuration
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Enable data export
    pub enable_export: bool,
    /// Export formats
    pub export_formats: Vec<ExportFormat>,
    /// Automatic export schedule
    pub auto_export_schedule: Option<ExportSchedule>,
    /// Report templates
    pub report_templates: Vec<ReportTemplate>,
}

/// Prediction configuration
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    /// Enable performance predictions
    pub enable_predictions: bool,
    /// Prediction horizon in hours
    pub prediction_horizon_hours: u32,
    /// Prediction models
    pub prediction_models: Vec<PredictionModel>,
    /// Prediction confidence threshold
    pub confidence_threshold: f64,
    /// Enable adaptive learning
    pub enable_adaptive_learning: bool,
}

/// Chart types for visualization
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChartType {
    LineChart,
    BarChart,
    ScatterPlot,
    HeatMap,
    Histogram,
    BoxPlot,
    ViolinPlot,
    Waterfall,
    Gauge,
    TreeMap,
    Sankey,
    Radar,
    Candlestick,
    Custom(String),
}

/// Dashboard layout options
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DashboardLayout {
    Grid,
    Fluid,
    Tabbed,
    Stacked,
    Custom(String),
}

/// Color schemes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColorScheme {
    Default,
    Dark,
    Light,
    HighContrast,
    Scientific,
    Custom(Vec<String>),
}

/// Aggregation levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AggregationLevel {
    RealTime,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Quarter,
    Year,
}

/// Custom visualization definition
#[derive(Debug, Clone)]
pub struct CustomVisualization {
    pub name: String,
    pub visualization_type: String,
    pub data_sources: Vec<String>,
    pub configuration: HashMap<String, String>,
    pub filters: Vec<VisualizationFilter>,
}

/// Visualization filter
#[derive(Debug, Clone)]
pub struct VisualizationFilter {
    pub field: String,
    pub operator: String,
    pub value: String,
}

/// Alert threshold definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThreshold {
    pub metric_name: String,
    pub threshold_type: ThresholdType,
    pub value: f64,
    pub severity: AlertSeverity,
    pub duration: Duration,
    pub enabled: bool,
}

/// Threshold types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdType {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    PercentageChange,
    StandardDeviation,
    Custom(String),
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Notification channels
#[derive(Debug, Clone)]
pub struct NotificationChannel {
    pub channel_type: ChannelType,
    pub configuration: HashMap<String, String>,
    pub enabled: bool,
    pub filters: Vec<NotificationFilter>,
}

/// Channel types for notifications
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChannelType {
    Email,
    Slack,
    SMS,
    Webhook,
    PagerDuty,
    Discord,
    Teams,
    Custom(String),
}

/// Notification filter
#[derive(Debug, Clone)]
pub struct NotificationFilter {
    pub filter_type: String,
    pub condition: String,
    pub value: String,
}

/// Alert suppression rules
#[derive(Debug, Clone)]
pub struct SuppressionRule {
    pub rule_name: String,
    pub conditions: Vec<SuppressionCondition>,
    pub duration: Duration,
    pub enabled: bool,
}

/// Suppression condition
#[derive(Debug, Clone)]
pub struct SuppressionCondition {
    pub field: String,
    pub operator: String,
    pub value: String,
}

/// Escalation policies
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    pub policy_name: String,
    pub steps: Vec<EscalationStep>,
    pub enabled: bool,
}

/// Escalation step
#[derive(Debug, Clone)]
pub struct EscalationStep {
    pub step_number: usize,
    pub delay: Duration,
    pub notification_channels: Vec<String>,
    pub action_type: EscalationActionType,
}

/// Escalation action types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EscalationActionType {
    Notify,
    AutoRemediate,
    CreateTicket,
    CallOnDuty,
    Custom(String),
}

/// Export formats
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportFormat {
    JSON,
    CSV,
    Excel,
    PDF,
    PNG,
    SVG,
    HTML,
    Custom(String),
}

/// Export schedule
#[derive(Debug, Clone)]
pub struct ExportSchedule {
    pub frequency: ExportFrequency,
    pub time_of_day: Option<chrono::NaiveTime>,
    pub recipients: Vec<String>,
    pub formats: Vec<ExportFormat>,
}

/// Export frequency
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportFrequency {
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Custom(String),
}

/// Report templates
#[derive(Debug, Clone)]
pub struct ReportTemplate {
    pub template_name: String,
    pub template_type: ReportType,
    pub sections: Vec<ReportSection>,
    pub layout: ReportLayout,
    pub enabled: bool,
}

/// Report types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportType {
    Executive,
    Technical,
    Operational,
    Compliance,
    Custom(String),
}

/// Report section
#[derive(Debug, Clone)]
pub struct ReportSection {
    pub section_name: String,
    pub section_type: ReportSectionType,
    pub data_sources: Vec<String>,
    pub charts: Vec<ChartType>,
    pub order: usize,
}

/// Report section types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportSectionType {
    Summary,
    Charts,
    Tables,
    Analysis,
    Recommendations,
    Custom(String),
}

/// Report layout
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportLayout {
    Standard,
    Compact,
    Detailed,
    Custom(String),
}

/// Prediction models
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictionModel {
    LinearRegression,
    ARIMA,
    LSTM,
    Prophet,
    ExponentialSmoothing,
    RandomForest,
    GradientBoosting,
    Custom(String),
}

impl Default for PerformanceDashboardConfig {
    fn default() -> Self {
        Self {
            enable_realtime_monitoring: true,
            collection_interval: 30,
            retention_days: 90,
            dashboard_refresh_rate: 5,
            analytics_config: AnalyticsConfig {
                enable_statistical_analysis: true,
                enable_trend_analysis: true,
                enable_correlation_analysis: true,
                enable_anomaly_detection: true,
                enable_performance_modeling: true,
                confidence_level: 0.95,
                anomaly_sensitivity: 0.05,
                trend_window_size: 100,
            },
            visualization_config: VisualizationConfig {
                enable_interactive_charts: true,
                chart_types: vec![
                    ChartType::LineChart,
                    ChartType::BarChart,
                    ChartType::HeatMap,
                    ChartType::Gauge,
                    ChartType::BoxPlot,
                ],
                dashboard_layout: DashboardLayout::Grid,
                color_scheme: ColorScheme::Scientific,
                aggregation_levels: vec![
                    AggregationLevel::RealTime,
                    AggregationLevel::Minute,
                    AggregationLevel::Hour,
                    AggregationLevel::Day,
                ],
                custom_visualizations: Vec::new(),
            },
            alert_config: AlertConfig {
                enable_alerts: true,
                alert_thresholds: [
                    (
                        "fidelity".to_string(),
                        AlertThreshold {
                            metric_name: "fidelity".to_string(),
                            threshold_type: ThresholdType::LessThan,
                            value: 0.9,
                            severity: AlertSeverity::Warning,
                            duration: Duration::from_secs(300),
                            enabled: true,
                        },
                    ),
                    (
                        "error_rate".to_string(),
                        AlertThreshold {
                            metric_name: "error_rate".to_string(),
                            threshold_type: ThresholdType::GreaterThan,
                            value: 0.05,
                            severity: AlertSeverity::Error,
                            duration: Duration::from_secs(60),
                            enabled: true,
                        },
                    ),
                ]
                .iter()
                .cloned()
                .collect(),
                notification_channels: vec![NotificationChannel {
                    channel_type: ChannelType::Email,
                    configuration: [("recipients".to_string(), "admin@example.com".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    enabled: true,
                    filters: Vec::new(),
                }],
                suppression_rules: Vec::new(),
                escalation_policies: Vec::new(),
            },
            export_config: ExportConfig {
                enable_export: true,
                export_formats: vec![ExportFormat::JSON, ExportFormat::CSV, ExportFormat::PDF],
                auto_export_schedule: Some(ExportSchedule {
                    frequency: ExportFrequency::Daily,
                    // 9:00:00 is a valid time, use expect for infallible case
                    time_of_day: chrono::NaiveTime::from_hms_opt(9, 0, 0),
                    recipients: vec!["reports@example.com".to_string()],
                    formats: vec![ExportFormat::PDF],
                }),
                report_templates: vec![ReportTemplate {
                    template_name: "Daily Performance Summary".to_string(),
                    template_type: ReportType::Executive,
                    sections: Vec::new(),
                    layout: ReportLayout::Standard,
                    enabled: true,
                }],
            },
            prediction_config: PredictionConfig {
                enable_predictions: true,
                prediction_horizon_hours: 24,
                prediction_models: vec![
                    PredictionModel::ARIMA,
                    PredictionModel::Prophet,
                    PredictionModel::ExponentialSmoothing,
                ],
                confidence_threshold: 0.8,
                enable_adaptive_learning: true,
            },
        }
    }
}

impl AnalyticsConfig {
    /// Create a basic analytics configuration
    pub const fn basic() -> Self {
        Self {
            enable_statistical_analysis: true,
            enable_trend_analysis: false,
            enable_correlation_analysis: false,
            enable_anomaly_detection: true,
            enable_performance_modeling: false,
            confidence_level: 0.95,
            anomaly_sensitivity: 0.1,
            trend_window_size: 50,
        }
    }

    /// Create a comprehensive analytics configuration
    pub const fn comprehensive() -> Self {
        Self {
            enable_statistical_analysis: true,
            enable_trend_analysis: true,
            enable_correlation_analysis: true,
            enable_anomaly_detection: true,
            enable_performance_modeling: true,
            confidence_level: 0.99,
            anomaly_sensitivity: 0.01,
            trend_window_size: 200,
        }
    }
}

impl VisualizationConfig {
    /// Create a minimal visualization configuration
    pub fn minimal() -> Self {
        Self {
            enable_interactive_charts: false,
            chart_types: vec![ChartType::LineChart, ChartType::BarChart],
            dashboard_layout: DashboardLayout::Stacked,
            color_scheme: ColorScheme::Default,
            aggregation_levels: vec![AggregationLevel::Hour, AggregationLevel::Day],
            custom_visualizations: Vec::new(),
        }
    }

    /// Create a comprehensive visualization configuration
    pub fn comprehensive() -> Self {
        Self {
            enable_interactive_charts: true,
            chart_types: vec![
                ChartType::LineChart,
                ChartType::BarChart,
                ChartType::ScatterPlot,
                ChartType::HeatMap,
                ChartType::Histogram,
                ChartType::BoxPlot,
                ChartType::ViolinPlot,
                ChartType::Gauge,
                ChartType::Radar,
            ],
            dashboard_layout: DashboardLayout::Grid,
            color_scheme: ColorScheme::Scientific,
            aggregation_levels: vec![
                AggregationLevel::RealTime,
                AggregationLevel::Minute,
                AggregationLevel::Hour,
                AggregationLevel::Day,
                AggregationLevel::Week,
                AggregationLevel::Month,
            ],
            custom_visualizations: Vec::new(),
        }
    }
}

impl AlertConfig {
    /// Create a basic alert configuration with essential thresholds
    pub fn basic() -> Self {
        Self {
            enable_alerts: true,
            alert_thresholds: [(
                "fidelity".to_string(),
                AlertThreshold {
                    metric_name: "fidelity".to_string(),
                    threshold_type: ThresholdType::LessThan,
                    value: 0.95,
                    severity: AlertSeverity::Warning,
                    duration: Duration::from_secs(300),
                    enabled: true,
                },
            )]
            .iter()
            .cloned()
            .collect(),
            notification_channels: vec![NotificationChannel {
                channel_type: ChannelType::Email,
                configuration: HashMap::new(),
                enabled: true,
                filters: Vec::new(),
            }],
            suppression_rules: Vec::new(),
            escalation_policies: Vec::new(),
        }
    }
}
