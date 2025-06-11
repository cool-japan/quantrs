//! Alerting and notification configuration.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Cloud alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudAlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Alert rules
    pub rules: Vec<AlertRule>,
    /// Notification channels
    pub notification_channels: Vec<NotificationChannel>,
    /// Escalation policies
    pub escalation_policies: Vec<EscalationPolicy>,
    /// Alert management
    pub management: AlertManagementConfig,
}

/// Alert rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Condition
    pub condition: AlertCondition,
    /// Severity
    pub severity: AlertSeverity,
    /// Notification channels
    pub channels: Vec<String>,
    /// Suppression rules
    pub suppression: Option<SuppressionRule>,
    /// Tags
    pub tags: std::collections::HashMap<String, String>,
}

/// Alert condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCondition {
    /// Metric name
    pub metric: String,
    /// Operator
    pub operator: ComparisonOperator,
    /// Threshold value
    pub threshold: f64,
    /// Evaluation window
    pub window: Duration,
    /// Aggregation function
    pub aggregation: AggregationFunction,
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

/// Aggregation functions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregationFunction {
    Average,
    Sum,
    Count,
    Max,
    Min,
    Percentile(f64),
    Custom(String),
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Suppression rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionRule {
    /// Suppression window
    pub window: Duration,
    /// Suppression conditions
    pub conditions: Vec<SuppressionCondition>,
    /// Max suppressions
    pub max_suppressions: Option<u32>,
}

/// Suppression condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionCondition {
    /// Field name
    pub field: String,
    /// Field value
    pub value: String,
    /// Match type
    pub match_type: MatchType,
}

/// Match types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MatchType {
    Exact,
    Pattern,
    Regex,
    Contains,
}

/// Notification channels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email,
    SMS,
    Slack,
    Teams,
    PagerDuty,
    Webhook,
    Custom(String),
}

/// Escalation policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicy {
    /// Policy name
    pub name: String,
    /// Escalation steps
    pub steps: Vec<EscalationStep>,
    /// Repeat policy
    pub repeat: Option<RepeatPolicy>,
}

/// Escalation step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationStep {
    /// Step number
    pub step: u32,
    /// Delay before this step
    pub delay: Duration,
    /// Notification targets
    pub targets: Vec<NotificationTarget>,
    /// Step conditions
    pub conditions: Vec<StepCondition>,
}

/// Notification target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationTarget {
    /// Target type
    pub target_type: TargetType,
    /// Target identifier
    pub identifier: String,
    /// Channel preferences
    pub channels: Vec<NotificationChannel>,
}

/// Target types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TargetType {
    User,
    Group,
    Role,
    External,
}

/// Step condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepCondition {
    /// Condition type
    pub condition_type: StepConditionType,
    /// Condition value
    pub value: String,
}

/// Step condition types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StepConditionType {
    AlertSeverity,
    AlertSource,
    TimeOfDay,
    DayOfWeek,
    Custom(String),
}

/// Repeat policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepeatPolicy {
    /// Enable repeat
    pub enabled: bool,
    /// Repeat interval
    pub interval: Duration,
    /// Maximum repeats
    pub max_repeats: Option<u32>,
    /// Repeat conditions
    pub conditions: Vec<RepeatCondition>,
}

/// Repeat condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepeatCondition {
    /// Condition type
    pub condition_type: RepeatConditionType,
    /// Condition value
    pub value: String,
}

/// Repeat condition types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RepeatConditionType {
    NoAcknowledgment,
    NoResolution,
    ContinuedViolation,
    Custom(String),
}

/// Alert management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertManagementConfig {
    /// Auto-resolution
    pub auto_resolution: AutoResolutionConfig,
    /// Alert grouping
    pub grouping: AlertGroupingConfig,
    /// Alert correlation
    pub correlation: AlertCorrelationConfig,
    /// Alert history
    pub history: AlertHistoryConfig,
}

/// Auto-resolution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoResolutionConfig {
    /// Enable auto-resolution
    pub enabled: bool,
    /// Resolution timeout
    pub timeout: Duration,
    /// Resolution conditions
    pub conditions: Vec<ResolutionCondition>,
}

/// Resolution condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionCondition {
    /// Condition type
    pub condition_type: ResolutionConditionType,
    /// Condition parameters
    pub parameters: std::collections::HashMap<String, String>,
}

/// Resolution condition types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolutionConditionType {
    MetricBelowThreshold,
    NoNewViolations,
    ManualResolution,
    Custom(String),
}

/// Alert grouping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertGroupingConfig {
    /// Enable grouping
    pub enabled: bool,
    /// Grouping criteria
    pub criteria: Vec<GroupingCriterion>,
    /// Grouping window
    pub window: Duration,
    /// Maximum group size
    pub max_group_size: Option<u32>,
}

/// Grouping criterion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupingCriterion {
    /// Field name
    pub field: String,
    /// Grouping method
    pub method: GroupingMethod,
}

/// Grouping methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GroupingMethod {
    Exact,
    Similar,
    Pattern,
    Custom(String),
}

/// Alert correlation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCorrelationConfig {
    /// Enable correlation
    pub enabled: bool,
    /// Correlation rules
    pub rules: Vec<CorrelationRule>,
    /// Correlation window
    pub window: Duration,
}

/// Correlation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationRule {
    /// Rule name
    pub name: String,
    /// Rule type
    pub rule_type: CorrelationRuleType,
    /// Pattern
    pub pattern: String,
    /// Actions
    pub actions: Vec<CorrelationAction>,
}

/// Correlation rule types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CorrelationRuleType {
    Sequence,
    Temporal,
    Causal,
    Statistical,
    Custom(String),
}

/// Correlation action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAction {
    /// Action type
    pub action_type: CorrelationActionType,
    /// Action parameters
    pub parameters: std::collections::HashMap<String, String>,
}

/// Correlation action types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CorrelationActionType {
    CreateIncident,
    UpdateSeverity,
    SuppressAlerts,
    TriggerRunbook,
    Custom(String),
}

/// Alert history configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertHistoryConfig {
    /// Enable history tracking
    pub enabled: bool,
    /// Retention period
    pub retention: Duration,
    /// Archive configuration
    pub archive: ArchiveConfig,
    /// Analytics
    pub analytics: HistoryAnalyticsConfig,
}

/// Archive configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveConfig {
    /// Enable archiving
    pub enabled: bool,
    /// Archive after
    pub archive_after: Duration,
    /// Archive storage
    pub storage_type: ArchiveStorageType,
}

/// Archive storage types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArchiveStorageType {
    Local,
    S3,
    Database,
    Custom(String),
}

/// History analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryAnalyticsConfig {
    /// Enable analytics
    pub enabled: bool,
    /// Analytics types
    pub types: Vec<AnalyticsType>,
    /// Analysis frequency
    pub frequency: Duration,
}

/// Analytics types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnalyticsType {
    AlertTrends,
    MeanTimeToResolution,
    FalsePositiveRate,
    AlertVelocity,
    Custom(String),
}

impl Default for CloudAlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rules: vec![],
            notification_channels: vec![NotificationChannel::Email],
            escalation_policies: vec![],
            management: AlertManagementConfig::default(),
        }
    }
}

impl Default for AlertManagementConfig {
    fn default() -> Self {
        Self {
            auto_resolution: AutoResolutionConfig::default(),
            grouping: AlertGroupingConfig::default(),
            correlation: AlertCorrelationConfig::default(),
            history: AlertHistoryConfig::default(),
        }
    }
}

impl Default for AutoResolutionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            timeout: Duration::from_secs(3600), // 1 hour
            conditions: vec![],
        }
    }
}

impl Default for AlertGroupingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            criteria: vec![],
            window: Duration::from_secs(300), // 5 minutes
            max_group_size: Some(50),
        }
    }
}

impl Default for AlertCorrelationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            rules: vec![],
            window: Duration::from_secs(600), // 10 minutes
        }
    }
}

impl Default for AlertHistoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            retention: Duration::from_secs(86400 * 90), // 90 days
            archive: ArchiveConfig::default(),
            analytics: HistoryAnalyticsConfig::default(),
        }
    }
}

impl Default for ArchiveConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            archive_after: Duration::from_secs(86400 * 30), // 30 days
            storage_type: ArchiveStorageType::Local,
        }
    }
}

impl Default for HistoryAnalyticsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            types: vec![AnalyticsType::AlertTrends],
            frequency: Duration::from_secs(86400), // daily
        }
    }
}
