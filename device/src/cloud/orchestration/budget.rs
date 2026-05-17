//! Cloud Orchestration — Budget Management Configuration
//!
//! Budget tracking, approval workflows, cost allocation, monitoring,
//! and forecasting configuration types for cloud orchestration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Budget management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetManagementConfig {
    /// Global budget settings
    pub global_budget: GlobalBudgetConfig,
    /// Department budgets
    pub department_budgets: HashMap<String, DepartmentBudgetConfig>,
    /// Project budgets
    pub project_budgets: HashMap<String, ProjectBudgetConfig>,
    /// Budget monitoring
    pub monitoring: BudgetMonitoringConfig,
}

/// Global budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalBudgetConfig {
    /// Total budget
    pub total_budget: f64,
    /// Budget period
    pub period: BudgetPeriod,
    /// Currency
    pub currency: String,
    /// Rollover policy
    pub rollover_policy: RolloverPolicy,
}

/// Budget periods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetPeriod {
    Monthly,
    Quarterly,
    Annual,
    Custom(Duration),
}

/// Rollover policies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RolloverPolicy {
    NoRollover,
    FullRollover,
    PartialRollover(f64),
    ConditionalRollover,
}

/// Department budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepartmentBudgetConfig {
    /// Department name
    pub name: String,
    /// Allocated budget
    pub allocated_budget: f64,
    /// Spending limits
    pub spending_limits: SpendingLimits,
    /// Approval workflow
    pub approval_workflow: BudgetApprovalWorkflow,
}

/// Spending limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendingLimits {
    /// Daily limit
    pub daily_limit: Option<f64>,
    /// Weekly limit
    pub weekly_limit: Option<f64>,
    /// Monthly limit
    pub monthly_limit: Option<f64>,
    /// Per-transaction limit
    pub per_transaction_limit: Option<f64>,
}

/// Budget approval workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetApprovalWorkflow {
    /// Approval levels
    pub levels: Vec<BudgetApprovalLevel>,
    /// Auto-approval thresholds
    pub auto_approval_thresholds: HashMap<String, f64>,
    /// Escalation timeouts
    pub escalation_timeouts: HashMap<String, Duration>,
}

/// Budget approval level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetApprovalLevel {
    /// Level name
    pub name: String,
    /// Approvers
    pub approvers: Vec<String>,
    /// Spending thresholds
    pub thresholds: HashMap<String, f64>,
}

/// Project budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectBudgetConfig {
    /// Project name
    pub name: String,
    /// Project budget
    pub budget: f64,
    /// Cost tracking
    pub cost_tracking: ProjectCostTracking,
    /// Budget alerts
    pub alerts: ProjectBudgetAlerts,
}

/// Project cost tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectCostTracking {
    /// Granularity
    pub granularity: CostTrackingGranularity,
    /// Cost categories
    pub categories: Vec<CostCategory>,
    /// Allocation rules
    pub allocation_rules: Vec<CostAllocationRule>,
}

/// Cost tracking granularity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CostTrackingGranularity {
    Hourly,
    Daily,
    Weekly,
    Monthly,
}

/// Cost category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostCategory {
    /// Category name
    pub name: String,
    /// Description
    pub description: String,
    /// Budget allocation
    pub budget_allocation: f64,
}

/// Cost allocation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAllocationRule {
    /// Rule name
    pub name: String,
    /// Source category
    pub source: String,
    /// Target category
    pub target: String,
    /// Allocation percentage
    pub percentage: f64,
}

/// Project budget alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectBudgetAlerts {
    /// Alert thresholds
    pub thresholds: Vec<f64>,
    /// Alert recipients
    pub recipients: Vec<String>,
    /// Alert frequency
    pub frequency: AlertFrequency,
}

/// Alert frequency
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertFrequency {
    Immediate,
    Daily,
    Weekly,
    OnThreshold,
}

/// Budget monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetMonitoringConfig {
    /// Real-time monitoring
    pub real_time: bool,
    /// Reporting frequency
    pub reporting_frequency: ReportingFrequency,
    /// Variance analysis
    pub variance_analysis: BudgetVarianceAnalysis,
    /// Forecasting
    pub forecasting: BudgetForecastingConfig,
}

/// Reporting frequency
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportingFrequency {
    RealTime,
    Hourly,
    Daily,
    Weekly,
    Monthly,
}

/// Budget variance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetVarianceAnalysis {
    /// Enable analysis
    pub enabled: bool,
    /// Variance thresholds
    pub thresholds: BudgetVarianceThresholds,
    /// Analysis methods
    pub methods: Vec<VarianceAnalysisMethod>,
}

/// Budget variance thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetVarianceThresholds {
    /// Warning threshold
    pub warning: f64,
    /// Critical threshold
    pub critical: f64,
    /// Emergency threshold
    pub emergency: f64,
}

/// Variance analysis methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VarianceAnalysisMethod {
    AbsoluteVariance,
    PercentageVariance,
    TrendAnalysis,
    SeasonalAnalysis,
}

/// Budget forecasting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetForecastingConfig {
    /// Enable forecasting
    pub enabled: bool,
    /// Forecasting models
    pub models: Vec<BudgetForecastingModel>,
    /// Forecast horizon
    pub horizon: Duration,
    /// Update frequency
    pub update_frequency: Duration,
}

/// Budget forecasting models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetForecastingModel {
    LinearTrend,
    ExponentialSmoothing,
    ARIMA,
    MachineLearning,
    Custom(String),
}
