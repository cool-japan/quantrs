//! Analytics and reporting configuration.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Cloud analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudAnalyticsConfig {
    /// Enable analytics
    pub enabled: bool,
    /// Data collection
    pub data_collection: DataCollectionConfig,
    /// Analysis engines
    pub analysis_engines: Vec<AnalysisEngine>,
    /// Reporting
    pub reporting: ReportingConfig,
    /// Anomaly detection
    pub anomaly_detection: AnomalyDetectionConfig,
    /// Dashboards
    pub dashboards: DashboardConfig,
}

/// Data collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCollectionConfig {
    /// Collection frequency
    pub frequency: Duration,
    /// Data sources
    pub sources: Vec<DataSource>,
    /// Data retention
    pub retention: DataRetentionConfig,
    /// Data quality
    pub quality: DataQualityConfig,
}

/// Data sources
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataSource {
    Metrics,
    Logs,
    Events,
    Traces,
    Custom(String),
}

/// Data retention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRetentionConfig {
    /// Raw data retention
    pub raw_data: Duration,
    /// Aggregated data retention
    pub aggregated_data: Duration,
    /// Archive policy
    pub archive: ArchivePolicy,
}

/// Archive policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivePolicy {
    /// Enable archiving
    pub enabled: bool,
    /// Archive after
    pub archive_after: Duration,
    /// Compression settings
    pub compression: CompressionSettings,
}

/// Compression settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level
    pub level: u8,
    /// Block size
    pub block_size: usize,
}

/// Compression algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Lz4,
    Zstd,
    Custom(String),
}

/// Data quality configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityConfig {
    /// Enable quality checks
    pub enabled: bool,
    /// Quality rules
    pub rules: Vec<QualityRule>,
    /// Validation frequency
    pub validation_frequency: Duration,
}

/// Quality rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRule {
    /// Rule name
    pub name: String,
    /// Rule type
    pub rule_type: QualityRuleType,
    /// Parameters
    pub parameters: std::collections::HashMap<String, String>,
    /// Actions on failure
    pub failure_actions: Vec<QualityFailureAction>,
}

/// Quality rule types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityRuleType {
    Completeness,
    Accuracy,
    Consistency,
    Timeliness,
    Validity,
    Custom(String),
}

/// Quality failure actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityFailureAction {
    Alert,
    Quarantine,
    Discard,
    Repair,
    Custom(String),
}

/// Analysis engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisEngine {
    /// Engine name
    pub name: String,
    /// Engine type
    pub engine_type: AnalysisEngineType,
    /// Configuration
    pub config: EngineConfig,
    /// Enabled analyses
    pub analyses: Vec<AnalysisType>,
}

/// Analysis engine types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisEngineType {
    Statistical,
    MachineLearning,
    StreamProcessing,
    BatchProcessing,
    Custom(String),
}

/// Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Engine parameters
    pub parameters: std::collections::HashMap<String, String>,
    /// Resource allocation
    pub resources: ResourceAllocation,
    /// Performance settings
    pub performance: PerformanceSettings,
}

/// Resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// CPU cores
    pub cpu_cores: u32,
    /// Memory (MB)
    pub memory_mb: u32,
    /// Disk space (GB)
    pub disk_gb: u32,
    /// GPU units
    pub gpu_units: Option<u32>,
}

/// Performance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    /// Parallelism level
    pub parallelism: u32,
    /// Batch size
    pub batch_size: usize,
    /// Timeout
    pub timeout: Duration,
    /// Cache settings
    pub cache: CacheSettings,
}

/// Cache settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSettings {
    /// Enable caching
    pub enabled: bool,
    /// Cache size (MB)
    pub size_mb: u32,
    /// Cache TTL
    pub ttl: Duration,
    /// Cache policy
    pub policy: CachePolicy,
}

/// Cache policies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CachePolicy {
    LRU,
    LFU,
    FIFO,
    Custom(String),
}

/// Analysis types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisType {
    TrendAnalysis,
    SeasonalAnalysis,
    AnomalyDetection,
    CorrelationAnalysis,
    PredictiveAnalysis,
    RootCauseAnalysis,
    Custom(String),
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReportingConfig {
    /// Enable automated reporting
    pub enabled: bool,
    /// Report schedules
    pub schedules: Vec<ReportSchedule>,
    /// Report templates
    pub templates: Vec<ReportTemplate>,
    /// Distribution
    pub distribution: DistributionConfig,
}

/// Report schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSchedule {
    /// Schedule name
    pub name: String,
    /// Report template
    pub template: String,
    /// Frequency
    pub frequency: ReportFrequency,
    /// Recipients
    pub recipients: Vec<String>,
    /// Filters
    pub filters: std::collections::HashMap<String, String>,
}

/// Report frequency
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFrequency {
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Custom(Duration),
}

/// Report template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTemplate {
    /// Template name
    pub name: String,
    /// Template type
    pub template_type: ReportTemplateType,
    /// Sections
    pub sections: Vec<ReportSection>,
    /// Formatting
    pub formatting: ReportFormatting,
}

/// Report template types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportTemplateType {
    Executive,
    Technical,
    Operational,
    Compliance,
    Custom(String),
}

/// Report section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    /// Section name
    pub name: String,
    /// Section type
    pub section_type: ReportSectionType,
    /// Data queries
    pub queries: Vec<String>,
    /// Visualizations
    pub visualizations: Vec<VisualizationType>,
}

/// Report section types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportSectionType {
    Summary,
    Metrics,
    Trends,
    Alerts,
    Recommendations,
    Custom(String),
}

/// Visualization types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualizationType {
    Table,
    LineChart,
    BarChart,
    PieChart,
    Heatmap,
    Gauge,
    Custom(String),
}

/// Report formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportFormatting {
    /// Output format
    pub format: OutputFormat,
    /// Styling
    pub styling: StylingOptions,
    /// Layout
    pub layout: LayoutOptions,
}

/// Output formats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    PDF,
    HTML,
    Excel,
    CSV,
    JSON,
    Custom(String),
}

/// Styling options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StylingOptions {
    /// Theme
    pub theme: String,
    /// Colors
    pub colors: std::collections::HashMap<String, String>,
    /// Fonts
    pub fonts: FontSettings,
}

/// Font settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontSettings {
    /// Primary font
    pub primary: String,
    /// Secondary font
    pub secondary: String,
    /// Font sizes
    pub sizes: std::collections::HashMap<String, u32>,
}

/// Layout options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutOptions {
    /// Page orientation
    pub orientation: PageOrientation,
    /// Margins
    pub margins: Margins,
    /// Header/footer
    pub header_footer: HeaderFooterSettings,
}

/// Page orientation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PageOrientation {
    Portrait,
    Landscape,
}

/// Margins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Margins {
    /// Top margin
    pub top: u32,
    /// Bottom margin
    pub bottom: u32,
    /// Left margin
    pub left: u32,
    /// Right margin
    pub right: u32,
}

/// Header/footer settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderFooterSettings {
    /// Include header
    pub header: bool,
    /// Include footer
    pub footer: bool,
    /// Header content
    pub header_content: String,
    /// Footer content
    pub footer_content: String,
}

/// Distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionConfig {
    /// Distribution channels
    pub channels: Vec<DistributionChannel>,
    /// Access control
    pub access_control: AccessControlConfig,
}

/// Distribution channels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributionChannel {
    Email,
    S3,
    SharePoint,
    Slack,
    Teams,
    Custom(String),
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccessControlConfig {
    /// Enable access control
    pub enabled: bool,
    /// Permissions
    pub permissions: Vec<Permission>,
    /// Roles
    pub roles: Vec<Role>,
}

/// Permission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    /// Resource
    pub resource: String,
    /// Actions
    pub actions: Vec<Action>,
    /// Conditions
    pub conditions: Vec<String>,
}

/// Actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    Read,
    Write,
    Execute,
    Delete,
    Custom(String),
}

/// Role
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name
    pub name: String,
    /// Permissions
    pub permissions: Vec<String>,
    /// Users
    pub users: Vec<String>,
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    /// Enable anomaly detection
    pub enabled: bool,
    /// Detection algorithms
    pub algorithms: Vec<AnomalyAlgorithm>,
    /// Sensitivity settings
    pub sensitivity: SensitivitySettings,
    /// Notifications
    pub notifications: AnomalyNotificationConfig,
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyAlgorithm {
    Statistical,
    IsolationForest,
    OneClassSVM,
    AutoEncoder,
    LSTM,
    Custom(String),
}

/// Sensitivity settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivitySettings {
    /// Detection threshold
    pub threshold: f64,
    /// Confidence level
    pub confidence: f64,
    /// Minimum severity
    pub min_severity: AnomalySeverity,
}

/// Anomaly severity
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// Anomaly notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyNotificationConfig {
    /// Notification channels
    pub channels: Vec<super::alerting::NotificationChannel>,
    /// Recipients
    pub recipients: Vec<String>,
    /// Frequency limits
    pub frequency_limits: NotificationFrequencyLimits,
}

/// Notification frequency limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationFrequencyLimits {
    /// Maximum notifications per hour
    pub max_per_hour: u32,
    /// Maximum notifications per day
    pub max_per_day: u32,
    /// Cooldown period
    pub cooldown: Duration,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DashboardConfig {
    /// Enable dashboards
    pub enabled: bool,
    /// Dashboard definitions
    pub dashboards: Vec<Dashboard>,
    /// Refresh settings
    pub refresh: RefreshSettings,
}

/// Dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dashboard {
    /// Dashboard name
    pub name: String,
    /// Dashboard type
    pub dashboard_type: DashboardType,
    /// Widgets
    pub widgets: Vec<Widget>,
    /// Layout
    pub layout: DashboardLayout,
}

/// Dashboard type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DashboardType {
    Overview,
    Detailed,
    RealTime,
    Historical,
    Custom(String),
}

/// Widget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Widget {
    /// Widget name
    pub name: String,
    /// Widget type
    pub widget_type: WidgetType,
    /// Data source
    pub data_source: String,
    /// Configuration
    pub config: WidgetConfig,
}

/// Widget type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WidgetType {
    Metric,
    Chart,
    Table,
    Alert,
    Custom(String),
}

/// Widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetConfig {
    /// Query
    pub query: String,
    /// Visualization settings
    pub visualization: VisualizationSettings,
    /// Refresh rate
    pub refresh_rate: Duration,
}

/// Visualization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationSettings {
    /// Chart type
    pub chart_type: VisualizationType,
    /// Color scheme
    pub colors: Vec<String>,
    /// Axes settings
    pub axes: AxesSettings,
}

/// Axes settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxesSettings {
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Axis ranges
    pub ranges: std::collections::HashMap<String, (f64, f64)>,
}

/// Dashboard layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardLayout {
    /// Grid columns
    pub columns: u32,
    /// Grid rows
    pub rows: u32,
    /// Widget positions
    pub positions: std::collections::HashMap<String, (u32, u32, u32, u32)>, // (x, y, w, h)
}

/// Refresh settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshSettings {
    /// Auto refresh
    pub auto_refresh: bool,
    /// Default refresh rate
    pub default_rate: Duration,
    /// Minimum refresh rate
    pub min_rate: Duration,
}

impl Default for CloudAnalyticsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            data_collection: DataCollectionConfig::default(),
            analysis_engines: vec![],
            reporting: ReportingConfig::default(),
            anomaly_detection: AnomalyDetectionConfig::default(),
            dashboards: DashboardConfig::default(),
        }
    }
}

impl Default for DataCollectionConfig {
    fn default() -> Self {
        Self {
            frequency: Duration::from_secs(60),
            sources: vec![DataSource::Metrics, DataSource::Logs],
            retention: DataRetentionConfig::default(),
            quality: DataQualityConfig::default(),
        }
    }
}

impl Default for DataRetentionConfig {
    fn default() -> Self {
        Self {
            raw_data: Duration::from_secs(86400 * 7),         // 7 days
            aggregated_data: Duration::from_secs(86400 * 90), // 90 days
            archive: ArchivePolicy::default(),
        }
    }
}

impl Default for ArchivePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            archive_after: Duration::from_secs(86400 * 30), // 30 days
            compression: CompressionSettings::default(),
        }
    }
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Gzip,
            level: 6,
            block_size: 65536,
        }
    }
}

impl Default for DataQualityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            rules: vec![],
            validation_frequency: Duration::from_secs(3600), // hourly
        }
    }
}

impl Default for DistributionConfig {
    fn default() -> Self {
        Self {
            channels: vec![DistributionChannel::Email],
            access_control: AccessControlConfig::default(),
        }
    }
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithms: vec![AnomalyAlgorithm::Statistical],
            sensitivity: SensitivitySettings::default(),
            notifications: AnomalyNotificationConfig::default(),
        }
    }
}

impl Default for SensitivitySettings {
    fn default() -> Self {
        Self {
            threshold: 0.95,
            confidence: 0.99,
            min_severity: AnomalySeverity::Medium,
        }
    }
}

impl Default for AnomalyNotificationConfig {
    fn default() -> Self {
        Self {
            channels: vec![super::alerting::NotificationChannel::Email],
            recipients: vec![],
            frequency_limits: NotificationFrequencyLimits::default(),
        }
    }
}

impl Default for NotificationFrequencyLimits {
    fn default() -> Self {
        Self {
            max_per_hour: 10,
            max_per_day: 50,
            cooldown: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl Default for RefreshSettings {
    fn default() -> Self {
        Self {
            auto_refresh: true,
            default_rate: Duration::from_secs(30),
            min_rate: Duration::from_secs(5),
        }
    }
}
