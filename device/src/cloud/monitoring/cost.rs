//! Cost monitoring and budget tracking configuration.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Cost monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudCostMonitoringConfig {
    /// Enable cost monitoring
    pub enabled: bool,
    /// Cost tracking granularity
    pub granularity: CostTrackingGranularity,
    /// Budget tracking
    pub budget_tracking: BudgetTrackingConfig,
    /// Cost optimization tracking
    pub optimization_tracking: CostOptimizationTrackingConfig,
    /// Cost allocation
    pub allocation: CostAllocationConfig,
    /// Billing integration
    pub billing: BillingIntegrationConfig,
}

/// Cost tracking granularity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CostTrackingGranularity {
    PerHour,
    PerDay,
    PerWeek,
    PerMonth,
    PerProject,
    PerDepartment,
    PerUser,
    PerResource,
    Custom(String),
}

/// Budget tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetTrackingConfig {
    /// Track budget utilization
    pub track_utilization: bool,
    /// Track budget variance
    pub track_variance: bool,
    /// Forecast budget consumption
    pub forecast_consumption: bool,
    /// Alert thresholds
    pub alert_thresholds: Vec<f64>,
    /// Budget periods
    pub budget_periods: Vec<BudgetPeriod>,
    /// Multi-level budgets
    pub multi_level: MultilevelBudgetConfig,
}

/// Budget period configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetPeriod {
    /// Period name
    pub name: String,
    /// Period duration
    pub duration: Duration,
    /// Budget amount
    pub amount: f64,
    /// Currency
    pub currency: String,
    /// Rollover policy
    pub rollover_policy: RolloverPolicy,
}

/// Budget rollover policies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RolloverPolicy {
    NoRollover,
    FullRollover,
    PartialRollover(f64), // percentage
    Custom(String),
}

/// Multi-level budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultilevelBudgetConfig {
    /// Enable hierarchical budgets
    pub enabled: bool,
    /// Budget hierarchy levels
    pub levels: Vec<BudgetLevel>,
    /// Allocation strategies
    pub allocation_strategies: Vec<AllocationStrategy>,
}

/// Budget levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetLevel {
    /// Level name
    pub name: String,
    /// Parent level
    pub parent: Option<String>,
    /// Budget amount
    pub amount: f64,
    /// Allocation rules
    pub allocation_rules: Vec<AllocationRule>,
}

/// Allocation strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    EqualDistribution,
    WeightedDistribution,
    UsageBased,
    PriorityBased,
    Custom(String),
}

/// Allocation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationRule {
    /// Rule name
    pub name: String,
    /// Target entities
    pub targets: Vec<String>,
    /// Allocation percentage
    pub percentage: f64,
    /// Conditions
    pub conditions: Vec<AllocationCondition>,
}

/// Allocation conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Value
    pub value: String,
    /// Operator
    pub operator: ComparisonOperator,
}

/// Condition types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionType {
    ResourceType,
    Department,
    Project,
    User,
    Tag,
    Custom(String),
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equals,
    NotEquals,
    Contains,
    StartsWith,
    EndsWith,
    GreaterThan,
    LessThan,
    In(Vec<String>),
}

/// Cost optimization tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizationTrackingConfig {
    /// Track savings opportunities
    pub track_opportunities: bool,
    /// Track implemented optimizations
    pub track_implementations: bool,
    /// ROI tracking
    pub roi_tracking: ROITrackingConfig,
    /// Optimization recommendations
    pub recommendations: OptimizationRecommendationConfig,
}

/// ROI tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ROITrackingConfig {
    /// Calculate ROI
    pub calculate_roi: bool,
    /// ROI calculation period
    pub calculation_period: Duration,
    /// Include indirect benefits
    pub include_indirect_benefits: bool,
    /// ROI metrics
    pub metrics: Vec<ROIMetric>,
}

/// ROI metrics
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ROIMetric {
    CostSavings,
    ProductivityGains,
    QualityImprovements,
    RiskReduction,
    TimeToMarket,
    Custom(String),
}

/// Optimization recommendation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendationConfig {
    /// Enable recommendations
    pub enabled: bool,
    /// Recommendation types
    pub types: Vec<RecommendationType>,
    /// Recommendation frequency
    pub frequency: Duration,
    /// Minimum potential savings threshold
    pub savings_threshold: f64,
}

/// Recommendation types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    RightSizing,
    ReservedInstances,
    SpotInstances,
    ScheduledShutdown,
    StorageOptimization,
    NetworkOptimization,
    QuantumCircuitOptimization,
    Custom(String),
}

/// Cost allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAllocationConfig {
    /// Enable cost allocation
    pub enabled: bool,
    /// Allocation methods
    pub methods: Vec<CostAllocationMethod>,
    /// Tag-based allocation
    pub tag_based: TagBasedAllocationConfig,
    /// Usage-based allocation
    pub usage_based: UsageBasedAllocationConfig,
}

/// Cost allocation methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CostAllocationMethod {
    Direct,
    Proportional,
    ActivityBased,
    TagBased,
    UsageBased,
    Custom(String),
}

/// Tag-based allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagBasedAllocationConfig {
    /// Primary allocation tags
    pub primary_tags: Vec<String>,
    /// Secondary allocation tags
    pub secondary_tags: Vec<String>,
    /// Default allocation for untagged resources
    pub default_allocation: String,
}

/// Usage-based allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageBasedAllocationConfig {
    /// Allocation metrics
    pub metrics: Vec<AllocationMetric>,
    /// Weighting factors
    pub weights: std::collections::HashMap<AllocationMetric, f64>,
    /// Calculation frequency
    pub frequency: Duration,
}

/// Allocation metrics
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllocationMetric {
    CPUHours,
    MemoryHours,
    StorageGB,
    NetworkGBTransferred,
    QuantumCircuitExecutions,
    DatabaseQueries,
    Custom(String),
}

/// Billing integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingIntegrationConfig {
    /// Enable billing integration
    pub enabled: bool,
    /// Billing providers
    pub providers: Vec<BillingProvider>,
    /// Billing frequency
    pub frequency: BillingFrequency,
    /// Cost center mapping
    pub cost_center_mapping: CostCenterMappingConfig,
}

/// Billing providers
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BillingProvider {
    AWS,
    Azure,
    GCP,
    Custom(String),
}

/// Billing frequency
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BillingFrequency {
    RealTime,
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Custom(Duration),
}

/// Cost center mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostCenterMappingConfig {
    /// Mapping rules
    pub rules: Vec<CostCenterRule>,
    /// Default cost center
    pub default_cost_center: String,
    /// Validation rules
    pub validation: CostCenterValidationConfig,
}

/// Cost center rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostCenterRule {
    /// Rule name
    pub name: String,
    /// Conditions
    pub conditions: Vec<AllocationCondition>,
    /// Target cost center
    pub target_cost_center: String,
    /// Priority
    pub priority: i32,
}

/// Cost center validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostCenterValidationConfig {
    /// Require valid cost center
    pub require_valid: bool,
    /// Valid cost centers
    pub valid_centers: Vec<String>,
    /// Validation frequency
    pub validation_frequency: Duration,
}

impl Default for CloudCostMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            granularity: CostTrackingGranularity::PerDay,
            budget_tracking: BudgetTrackingConfig::default(),
            optimization_tracking: CostOptimizationTrackingConfig::default(),
            allocation: CostAllocationConfig::default(),
            billing: BillingIntegrationConfig::default(),
        }
    }
}

impl Default for BudgetTrackingConfig {
    fn default() -> Self {
        Self {
            track_utilization: true,
            track_variance: true,
            forecast_consumption: false,
            alert_thresholds: vec![50.0, 75.0, 90.0, 100.0],
            budget_periods: vec![],
            multi_level: MultilevelBudgetConfig::default(),
        }
    }
}

impl Default for MultilevelBudgetConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            levels: vec![],
            allocation_strategies: vec![AllocationStrategy::EqualDistribution],
        }
    }
}

impl Default for CostOptimizationTrackingConfig {
    fn default() -> Self {
        Self {
            track_opportunities: true,
            track_implementations: false,
            roi_tracking: ROITrackingConfig::default(),
            recommendations: OptimizationRecommendationConfig::default(),
        }
    }
}

impl Default for ROITrackingConfig {
    fn default() -> Self {
        Self {
            calculate_roi: false,
            calculation_period: Duration::from_secs(86400 * 30), // 30 days
            include_indirect_benefits: false,
            metrics: vec![ROIMetric::CostSavings],
        }
    }
}

impl Default for OptimizationRecommendationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            types: vec![RecommendationType::RightSizing],
            frequency: Duration::from_secs(86400 * 7), // weekly
            savings_threshold: 100.0,                  // $100
        }
    }
}

impl Default for CostAllocationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            methods: vec![CostAllocationMethod::TagBased],
            tag_based: TagBasedAllocationConfig::default(),
            usage_based: UsageBasedAllocationConfig::default(),
        }
    }
}

impl Default for TagBasedAllocationConfig {
    fn default() -> Self {
        Self {
            primary_tags: vec!["Project".to_string(), "Department".to_string()],
            secondary_tags: vec!["Environment".to_string()],
            default_allocation: "Unallocated".to_string(),
        }
    }
}

impl Default for UsageBasedAllocationConfig {
    fn default() -> Self {
        let mut weights = std::collections::HashMap::new();
        weights.insert(AllocationMetric::CPUHours, 0.4);
        weights.insert(AllocationMetric::MemoryHours, 0.3);
        weights.insert(AllocationMetric::StorageGB, 0.2);
        weights.insert(AllocationMetric::NetworkGBTransferred, 0.1);

        Self {
            metrics: vec![
                AllocationMetric::CPUHours,
                AllocationMetric::MemoryHours,
                AllocationMetric::StorageGB,
            ],
            weights,
            frequency: Duration::from_secs(3600), // hourly
        }
    }
}

impl Default for BillingIntegrationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            providers: vec![],
            frequency: BillingFrequency::Daily,
            cost_center_mapping: CostCenterMappingConfig::default(),
        }
    }
}

impl Default for CostCenterMappingConfig {
    fn default() -> Self {
        Self {
            rules: vec![],
            default_cost_center: "Default".to_string(),
            validation: CostCenterValidationConfig::default(),
        }
    }
}

impl Default for CostCenterValidationConfig {
    fn default() -> Self {
        Self {
            require_valid: false,
            valid_centers: vec![],
            validation_frequency: Duration::from_secs(86400), // daily
        }
    }
}
