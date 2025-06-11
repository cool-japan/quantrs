//! Cost Management and Optimization Configuration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Cost management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostManagementConfig {
    /// Enable cost optimization
    pub enable_cost_optimization: bool,
    /// Cost optimization strategies
    pub optimization_strategies: Vec<CostOptimizationStrategy>,
    /// Pricing models
    pub pricing_models: HashMap<String, PricingModel>,
    /// Cost prediction and forecasting
    pub cost_prediction: CostPredictionConfig,
    /// Budget management
    pub budget_management: BudgetConfig,
    /// Cost alerting
    pub cost_alerting: CostAlertingConfig,
    /// Financial reporting
    pub reporting: FinancialReportingConfig,
    /// Cost allocation
    pub allocation: CostAllocationConfig,
}

/// Cost optimization strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CostOptimizationStrategy {
    SpotInstanceOptimization,
    ReservedInstanceOptimization,
    RightSizing,
    SchedulingOptimization,
    LoadBalancingOptimization,
    ResourcePooling,
    AutoShutdown,
    PriceComparison,
    ContractNegotiation,
    CustomStrategy(String),
}

/// Pricing model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingModel {
    pub model_type: String,
    pub base_rate: f64,
    pub scaling_factors: HashMap<String, f64>,
    pub discount_tiers: Vec<f64>,
    /// Pricing structure
    pub structure: PricingStructure,
    /// Contract terms
    pub contract_terms: ContractTerms,
    /// Dynamic pricing
    pub dynamic_pricing: DynamicPricingConfig,
}

/// Pricing structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingStructure {
    /// Billing model
    pub billing_model: BillingModel,
    /// Rate components
    pub rate_components: Vec<RateComponent>,
    /// Volume discounts
    pub volume_discounts: Vec<VolumeDiscount>,
    /// Time-based pricing
    pub time_based: TimeBasedPricingConfig,
}

/// Billing models
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BillingModel {
    PayPerUse,
    Subscription,
    Reserved,
    Spot,
    Prepaid,
    Hybrid,
}

/// Rate component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateComponent {
    /// Component name
    pub name: String,
    /// Rate per unit
    pub rate: f64,
    /// Unit type
    pub unit: String,
    /// Minimum charge
    pub minimum_charge: Option<f64>,
    /// Maximum charge
    pub maximum_charge: Option<f64>,
}

/// Volume discount
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeDiscount {
    /// Minimum volume
    pub min_volume: f64,
    /// Discount percentage
    pub discount_percent: f64,
    /// Discount type
    pub discount_type: DiscountType,
}

/// Discount types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiscountType {
    Percentage,
    FixedAmount,
    TieredDiscount,
    VolumeDiscount,
}

/// Time-based pricing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeBasedPricingConfig {
    /// Peak hours pricing
    pub peak_hours: PeakHoursPricing,
    /// Off-peak hours pricing
    pub off_peak_hours: OffPeakHoursPricing,
    /// Weekend pricing
    pub weekend_pricing: WeekendPricing,
    /// Holiday pricing
    pub holiday_pricing: HolidayPricing,
}

/// Peak hours pricing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakHoursPricing {
    /// Start time
    pub start_time: String,
    /// End time
    pub end_time: String,
    /// Rate multiplier
    pub rate_multiplier: f64,
    /// Days of week
    pub days_of_week: Vec<String>,
}

/// Off-peak hours pricing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffPeakHoursPricing {
    /// Rate multiplier
    pub rate_multiplier: f64,
    /// Minimum discount
    pub min_discount: f64,
}

/// Weekend pricing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeekendPricing {
    /// Weekend rate multiplier
    pub rate_multiplier: f64,
    /// Weekend days
    pub weekend_days: Vec<String>,
}

/// Holiday pricing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolidayPricing {
    /// Holiday rate multiplier
    pub rate_multiplier: f64,
    /// Holiday calendar
    pub holiday_calendar: Vec<String>,
}

/// Contract terms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractTerms {
    /// Contract duration
    pub duration: Duration,
    /// Commitment level
    pub commitment_level: CommitmentLevel,
    /// Early termination fees
    pub early_termination_fees: EarlyTerminationFees,
    /// Service level agreements
    pub sla: ServiceLevelAgreements,
}

/// Commitment levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CommitmentLevel {
    NoCommitment,
    MonthlyCommitment,
    YearlyCommitment,
    MultiYearCommitment,
    CustomCommitment(Duration),
}

/// Early termination fees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyTerminationFees {
    /// Fee structure
    pub fee_structure: FeeStructure,
    /// Calculation method
    pub calculation_method: FeeCalculationMethod,
    /// Waiver conditions
    pub waiver_conditions: Vec<WaiverCondition>,
}

/// Fee structures
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeeStructure {
    FixedFee(f64),
    PercentageOfRemaining(f64),
    ProRatedFee,
    NoFee,
}

/// Fee calculation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeeCalculationMethod {
    RemainingMonths,
    UnusedCommitment,
    TotalContractValue,
    Custom(String),
}

/// Waiver conditions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WaiverCondition {
    ServiceFailure,
    ForCause,
    MutualAgreement,
    Custom(String),
}

/// Service level agreements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceLevelAgreements {
    /// Uptime SLA
    pub uptime: f64,
    /// Performance SLA
    pub performance: HashMap<String, f64>,
    /// Support SLA
    pub support: SupportSLA,
    /// Penalties
    pub penalties: SLAPenalties,
}

/// Support SLA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportSLA {
    /// Response time by severity
    pub response_times: HashMap<String, Duration>,
    /// Resolution time by severity
    pub resolution_times: HashMap<String, Duration>,
    /// Escalation procedures
    pub escalation: EscalationProcedures,
}

/// Escalation procedures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationProcedures {
    /// Escalation levels
    pub levels: Vec<EscalationLevel>,
    /// Automatic escalation
    pub automatic: bool,
    /// Escalation triggers
    pub triggers: Vec<EscalationTrigger>,
}

/// Escalation level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    /// Level number
    pub level: u8,
    /// Contact information
    pub contacts: Vec<String>,
    /// Escalation time
    pub escalation_time: Duration,
}

/// Escalation triggers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EscalationTrigger {
    TimeElapsed,
    NoResponse,
    CustomerRequest,
    SeverityIncrease,
}

/// SLA penalties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLAPenalties {
    /// Penalty structure
    pub structure: PenaltyStructure,
    /// Credits
    pub credits: CreditStructure,
    /// Maximum penalties
    pub max_penalties: MaxPenalties,
}

/// Penalty structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyStructure {
    /// Uptime penalties
    pub uptime_penalties: Vec<UptimePenalty>,
    /// Performance penalties
    pub performance_penalties: Vec<PerformancePenalty>,
}

/// Uptime penalty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UptimePenalty {
    /// Minimum uptime
    pub min_uptime: f64,
    /// Penalty percentage
    pub penalty_percent: f64,
}

/// Performance penalty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePenalty {
    /// Metric name
    pub metric: String,
    /// Threshold
    pub threshold: f64,
    /// Penalty percentage
    pub penalty_percent: f64,
}

/// Credit structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditStructure {
    /// Service credits
    pub service_credits: Vec<ServiceCredit>,
    /// Credit application
    pub application: CreditApplication,
}

/// Service credit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceCredit {
    /// Condition
    pub condition: String,
    /// Credit amount
    pub amount: f64,
    /// Credit type
    pub credit_type: CreditType,
}

/// Credit types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CreditType {
    Percentage,
    FixedAmount,
    FreeUsage,
}

/// Credit application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditApplication {
    /// Application method
    pub method: CreditApplicationMethod,
    /// Processing time
    pub processing_time: Duration,
    /// Verification required
    pub verification_required: bool,
}

/// Credit application methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CreditApplicationMethod {
    Automatic,
    Manual,
    OnRequest,
}

/// Maximum penalties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxPenalties {
    /// Monthly maximum
    pub monthly_max: Option<f64>,
    /// Annual maximum
    pub annual_max: Option<f64>,
    /// Per incident maximum
    pub per_incident_max: Option<f64>,
}

/// Dynamic pricing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicPricingConfig {
    /// Enable dynamic pricing
    pub enabled: bool,
    /// Pricing algorithms
    pub algorithms: Vec<PricingAlgorithm>,
    /// Update frequency
    pub update_frequency: Duration,
    /// Price bounds
    pub price_bounds: PriceBounds,
}

/// Pricing algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PricingAlgorithm {
    SupplyDemand,
    Competition,
    CostPlus,
    ValueBased,
    MarketBased,
    Custom(String),
}

/// Price bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceBounds {
    /// Minimum price
    pub min_price: f64,
    /// Maximum price
    pub max_price: f64,
    /// Price change limits
    pub change_limits: PriceChangeLimits,
}

/// Price change limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceChangeLimits {
    /// Maximum increase percentage
    pub max_increase_percent: f64,
    /// Maximum decrease percentage
    pub max_decrease_percent: f64,
    /// Change frequency limit
    pub change_frequency: Duration,
}

/// Cost prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostPredictionConfig {
    pub enable_prediction: bool,
    pub prediction_algorithms: Vec<String>,
    pub prediction_horizon: u64,
    /// Prediction models
    pub models: Vec<CostPredictionModel>,
    /// Forecasting settings
    pub forecasting: ForecastingConfig,
    /// Accuracy monitoring
    pub accuracy_monitoring: AccuracyMonitoringConfig,
}

/// Cost prediction models
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CostPredictionModel {
    LinearRegression,
    ARIMA,
    LSTM,
    RandomForest,
    EnsembleModel,
    Custom(String),
}

/// Forecasting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastingConfig {
    /// Forecast horizons
    pub horizons: Vec<ForecastHorizon>,
    /// Confidence intervals
    pub confidence_intervals: Vec<f64>,
    /// Scenario analysis
    pub scenario_analysis: ScenarioAnalysisConfig,
}

/// Forecast horizon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastHorizon {
    /// Horizon name
    pub name: String,
    /// Duration
    pub duration: Duration,
    /// Update frequency
    pub update_frequency: Duration,
}

/// Scenario analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioAnalysisConfig {
    /// Enable scenario analysis
    pub enabled: bool,
    /// Scenarios
    pub scenarios: Vec<CostScenario>,
    /// Probability weights
    pub probability_weights: HashMap<String, f64>,
}

/// Cost scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostScenario {
    /// Scenario name
    pub name: String,
    /// Description
    pub description: String,
    /// Parameters
    pub parameters: HashMap<String, f64>,
    /// Expected impact
    pub expected_impact: f64,
}

/// Accuracy monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// Accuracy metrics
    pub metrics: Vec<AccuracyMetric>,
    /// Threshold alerts
    pub threshold_alerts: Vec<AccuracyAlert>,
    /// Model retraining
    pub retraining: ModelRetrainingConfig,
}

/// Accuracy metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AccuracyMetric {
    MAPE, // Mean Absolute Percentage Error
    RMSE, // Root Mean Square Error
    MAE,  // Mean Absolute Error
    R2,   // R-squared
    Custom(String),
}

/// Accuracy alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyAlert {
    /// Metric
    pub metric: AccuracyMetric,
    /// Threshold
    pub threshold: f64,
    /// Alert action
    pub action: AlertAction,
}

/// Alert actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertAction {
    Notify,
    RetainModel,
    SwitchModel,
    Custom(String),
}

/// Model retraining configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRetrainingConfig {
    /// Auto retraining
    pub auto_retrain: bool,
    /// Retraining triggers
    pub triggers: Vec<RetrainingTrigger>,
    /// Training frequency
    pub frequency: Duration,
}

/// Retraining triggers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RetrainingTrigger {
    AccuracyDrop,
    DataDrift,
    ScheduledRetrain,
    ManualTrigger,
}

/// Budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    pub daily_budget: Option<f64>,
    pub monthly_budget: Option<f64>,
    pub auto_scaling_budget: bool,
    pub budget_alerts: Vec<String>,
    /// Budget allocation
    pub allocation: BudgetAllocation,
    /// Budget tracking
    pub tracking: BudgetTrackingConfig,
    /// Budget enforcement
    pub enforcement: BudgetEnforcementConfig,
}

/// Budget allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAllocation {
    /// Department allocations
    pub departments: HashMap<String, f64>,
    /// Project allocations
    pub projects: HashMap<String, f64>,
    /// Provider allocations
    pub providers: HashMap<String, f64>,
    /// Reserve allocation
    pub reserve_percentage: f64,
}

/// Budget tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetTrackingConfig {
    /// Tracking frequency
    pub frequency: Duration,
    /// Variance analysis
    pub variance_analysis: VarianceAnalysisConfig,
    /// Trend analysis
    pub trend_analysis: TrendAnalysisConfig,
}

/// Variance analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceAnalysisConfig {
    /// Enable variance analysis
    pub enabled: bool,
    /// Variance thresholds
    pub thresholds: VarianceThresholds,
    /// Root cause analysis
    pub root_cause_analysis: bool,
}

/// Variance thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceThresholds {
    /// Warning threshold
    pub warning_threshold: f64,
    /// Critical threshold
    pub critical_threshold: f64,
    /// Absolute vs percentage
    pub is_percentage: bool,
}

/// Trend analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisConfig {
    /// Enable trend analysis
    pub enabled: bool,
    /// Analysis window
    pub analysis_window: Duration,
    /// Trend detection methods
    pub detection_methods: Vec<TrendDetectionMethod>,
}

/// Trend detection methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendDetectionMethod {
    MovingAverage,
    LinearRegression,
    ExponentialSmoothing,
    SeasonalDecomposition,
}

/// Budget enforcement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetEnforcementConfig {
    /// Enforcement policy
    pub policy: EnforcementPolicy,
    /// Actions
    pub actions: Vec<EnforcementAction>,
    /// Override permissions
    pub override_permissions: OverridePermissions,
}

/// Enforcement policies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnforcementPolicy {
    SoftLimit,
    HardLimit,
    GracePeriod,
    Escalation,
}

/// Enforcement actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnforcementAction {
    Alert,
    Throttle,
    Block,
    Approve,
    RequestApproval,
}

/// Override permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverridePermissions {
    /// Authorized users
    pub authorized_users: Vec<String>,
    /// Approval workflow
    pub approval_workflow: ApprovalWorkflow,
    /// Override limits
    pub override_limits: OverrideLimits,
}

/// Approval workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalWorkflow {
    /// Required approvers
    pub required_approvers: usize,
    /// Approval levels
    pub levels: Vec<ApprovalLevel>,
    /// Timeout
    pub timeout: Duration,
}

/// Approval level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalLevel {
    /// Level name
    pub name: String,
    /// Approvers
    pub approvers: Vec<String>,
    /// Budget limits
    pub budget_limits: HashMap<String, f64>,
}

/// Override limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverrideLimits {
    /// Maximum override percentage
    pub max_override_percent: f64,
    /// Override frequency limits
    pub frequency_limits: FrequencyLimits,
    /// Justification required
    pub justification_required: bool,
}

/// Frequency limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyLimits {
    /// Per day
    pub per_day: usize,
    /// Per week
    pub per_week: usize,
    /// Per month
    pub per_month: usize,
}

/// Cost alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAlertingConfig {
    pub enable_alerts: bool,
    pub alert_thresholds: Vec<f64>,
    pub notification_channels: Vec<String>,
    /// Alert rules
    pub rules: Vec<CostAlertRule>,
    /// Escalation policies
    pub escalation_policies: Vec<AlertEscalationPolicy>,
    /// Suppression rules
    pub suppression_rules: Vec<AlertSuppressionRule>,
}

/// Cost alert rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAlertRule {
    /// Rule name
    pub name: String,
    /// Condition
    pub condition: AlertCondition,
    /// Severity
    pub severity: AlertSeverity,
    /// Actions
    pub actions: Vec<AlertAction>,
}

/// Alert condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCondition {
    /// Metric
    pub metric: CostMetric,
    /// Operator
    pub operator: ComparisonOperator,
    /// Threshold
    pub threshold: f64,
    /// Time window
    pub time_window: Duration,
}

/// Cost metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CostMetric {
    TotalCost,
    HourlyCost,
    DailyCost,
    MonthlyCost,
    CostPerJob,
    CostPerQubit,
    BudgetUtilization,
    Custom(String),
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    GreaterThanOrEqual,
    LessThanOrEqual,
    NotEqual,
}

/// Alert severity
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Alert escalation policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEscalationPolicy {
    /// Policy name
    pub name: String,
    /// Escalation levels
    pub levels: Vec<EscalationStep>,
    /// Escalation timeouts
    pub timeouts: HashMap<String, Duration>,
}

/// Escalation step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationStep {
    /// Step name
    pub name: String,
    /// Contacts
    pub contacts: Vec<String>,
    /// Actions
    pub actions: Vec<EscalationAction>,
    /// Delay before escalation
    pub delay: Duration,
}

/// Escalation actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EscalationAction {
    Notify,
    AutoRemediate,
    ManualIntervention,
    EmergencyShutdown,
}

/// Alert suppression rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertSuppressionRule {
    /// Rule name
    pub name: String,
    /// Conditions
    pub conditions: Vec<SuppressionCondition>,
    /// Duration
    pub duration: Duration,
}

/// Suppression condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionCondition {
    /// Field
    pub field: String,
    /// Operator
    pub operator: ComparisonOperator,
    /// Value
    pub value: String,
}

/// Financial reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialReportingConfig {
    /// Enable reporting
    pub enabled: bool,
    /// Report types
    pub report_types: Vec<ReportType>,
    /// Report frequency
    pub frequency: ReportFrequency,
    /// Report delivery
    pub delivery: ReportDeliveryConfig,
}

/// Report types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportType {
    CostSummary,
    DetailedUsage,
    BudgetVariance,
    TrendAnalysis,
    CostOptimization,
    CustomReport(String),
}

/// Report frequency
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportFrequency {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    OnDemand,
}

/// Report delivery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportDeliveryConfig {
    /// Delivery methods
    pub methods: Vec<DeliveryMethod>,
    /// Recipients
    pub recipients: Vec<String>,
    /// Format options
    pub formats: Vec<ReportFormat>,
}

/// Delivery methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeliveryMethod {
    Email,
    Dashboard,
    API,
    FileSystem,
    CloudStorage,
}

/// Report formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportFormat {
    PDF,
    CSV,
    JSON,
    Excel,
    HTML,
}

/// Cost allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAllocationConfig {
    /// Allocation methods
    pub methods: Vec<AllocationMethod>,
    /// Allocation rules
    pub rules: Vec<AllocationRule>,
    /// Chargeback configuration
    pub chargeback: ChargebackConfig,
}

/// Allocation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AllocationMethod {
    DirectAllocation,
    ProportionalAllocation,
    ActivityBasedAllocation,
    UsageBasedAllocation,
    CustomAllocation(String),
}

/// Allocation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationRule {
    /// Rule name
    pub name: String,
    /// Source
    pub source: AllocationSource,
    /// Target
    pub target: AllocationTarget,
    /// Method
    pub method: AllocationMethod,
    /// Percentage
    pub percentage: Option<f64>,
}

/// Allocation source
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AllocationSource {
    SharedResources,
    Infrastructure,
    Support,
    Overhead,
    Custom(String),
}

/// Allocation target
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AllocationTarget {
    Department,
    Project,
    CostCenter,
    User,
    Custom(String),
}

/// Chargeback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChargebackConfig {
    /// Enable chargeback
    pub enabled: bool,
    /// Chargeback model
    pub model: ChargebackModel,
    /// Billing cycle
    pub billing_cycle: BillingCycle,
    /// Rate cards
    pub rate_cards: Vec<RateCard>,
}

/// Chargeback models
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChargebackModel {
    FullChargeback,
    Showback,
    HybridModel,
    CustomModel(String),
}

/// Billing cycle
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BillingCycle {
    Monthly,
    Quarterly,
    Annual,
    Custom(Duration),
}

/// Rate card
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateCard {
    /// Service name
    pub service: String,
    /// Unit rate
    pub rate: f64,
    /// Unit type
    pub unit: String,
    /// Effective date
    pub effective_date: String,
    /// Expiration date
    pub expiration_date: Option<String>,
}

impl Default for CostManagementConfig {
    fn default() -> Self {
        Self {
            enable_cost_optimization: true,
            optimization_strategies: vec![
                CostOptimizationStrategy::RightSizing,
                CostOptimizationStrategy::SchedulingOptimization,
            ],
            pricing_models: HashMap::new(),
            cost_prediction: CostPredictionConfig::default(),
            budget_management: BudgetConfig::default(),
            cost_alerting: CostAlertingConfig::default(),
            reporting: FinancialReportingConfig::default(),
            allocation: CostAllocationConfig::default(),
        }
    }
}

impl Default for PricingModel {
    fn default() -> Self {
        Self {
            model_type: "pay_per_use".to_string(),
            base_rate: 0.1,
            scaling_factors: HashMap::new(),
            discount_tiers: vec![],
            structure: PricingStructure::default(),
            contract_terms: ContractTerms::default(),
            dynamic_pricing: DynamicPricingConfig::default(),
        }
    }
}

impl Default for PricingStructure {
    fn default() -> Self {
        Self {
            billing_model: BillingModel::PayPerUse,
            rate_components: vec![],
            volume_discounts: vec![],
            time_based: TimeBasedPricingConfig::default(),
        }
    }
}

impl Default for TimeBasedPricingConfig {
    fn default() -> Self {
        Self {
            peak_hours: PeakHoursPricing::default(),
            off_peak_hours: OffPeakHoursPricing::default(),
            weekend_pricing: WeekendPricing::default(),
            holiday_pricing: HolidayPricing::default(),
        }
    }
}

impl Default for PeakHoursPricing {
    fn default() -> Self {
        Self {
            start_time: "09:00".to_string(),
            end_time: "17:00".to_string(),
            rate_multiplier: 1.5,
            days_of_week: vec![
                "Mon".to_string(),
                "Tue".to_string(),
                "Wed".to_string(),
                "Thu".to_string(),
                "Fri".to_string(),
            ],
        }
    }
}

impl Default for OffPeakHoursPricing {
    fn default() -> Self {
        Self {
            rate_multiplier: 0.8,
            min_discount: 0.1,
        }
    }
}

impl Default for WeekendPricing {
    fn default() -> Self {
        Self {
            rate_multiplier: 0.7,
            weekend_days: vec!["Sat".to_string(), "Sun".to_string()],
        }
    }
}

impl Default for HolidayPricing {
    fn default() -> Self {
        Self {
            rate_multiplier: 0.5,
            holiday_calendar: vec![],
        }
    }
}

impl Default for ContractTerms {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(86400 * 365), // 1 year
            commitment_level: CommitmentLevel::NoCommitment,
            early_termination_fees: EarlyTerminationFees::default(),
            sla: ServiceLevelAgreements::default(),
        }
    }
}

impl Default for EarlyTerminationFees {
    fn default() -> Self {
        Self {
            fee_structure: FeeStructure::NoFee,
            calculation_method: FeeCalculationMethod::RemainingMonths,
            waiver_conditions: vec![],
        }
    }
}

impl Default for ServiceLevelAgreements {
    fn default() -> Self {
        Self {
            uptime: 0.99,
            performance: HashMap::new(),
            support: SupportSLA::default(),
            penalties: SLAPenalties::default(),
        }
    }
}

impl Default for SupportSLA {
    fn default() -> Self {
        Self {
            response_times: HashMap::new(),
            resolution_times: HashMap::new(),
            escalation: EscalationProcedures::default(),
        }
    }
}

impl Default for EscalationProcedures {
    fn default() -> Self {
        Self {
            levels: vec![],
            automatic: false,
            triggers: vec![],
        }
    }
}

impl Default for SLAPenalties {
    fn default() -> Self {
        Self {
            structure: PenaltyStructure::default(),
            credits: CreditStructure::default(),
            max_penalties: MaxPenalties::default(),
        }
    }
}

impl Default for PenaltyStructure {
    fn default() -> Self {
        Self {
            uptime_penalties: vec![],
            performance_penalties: vec![],
        }
    }
}

impl Default for CreditStructure {
    fn default() -> Self {
        Self {
            service_credits: vec![],
            application: CreditApplication::default(),
        }
    }
}

impl Default for CreditApplication {
    fn default() -> Self {
        Self {
            method: CreditApplicationMethod::Manual,
            processing_time: Duration::from_secs(86400 * 7), // 1 week
            verification_required: true,
        }
    }
}

impl Default for MaxPenalties {
    fn default() -> Self {
        Self {
            monthly_max: Some(0.1), // 10% of monthly bill
            annual_max: Some(0.05), // 5% of annual bill
            per_incident_max: None,
        }
    }
}

impl Default for DynamicPricingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithms: vec![],
            update_frequency: Duration::from_secs(3600), // 1 hour
            price_bounds: PriceBounds::default(),
        }
    }
}

impl Default for PriceBounds {
    fn default() -> Self {
        Self {
            min_price: 0.01,
            max_price: 10.0,
            change_limits: PriceChangeLimits::default(),
        }
    }
}

impl Default for PriceChangeLimits {
    fn default() -> Self {
        Self {
            max_increase_percent: 0.1,                   // 10%
            max_decrease_percent: 0.1,                   // 10%
            change_frequency: Duration::from_secs(3600), // 1 hour
        }
    }
}

impl Default for CostPredictionConfig {
    fn default() -> Self {
        Self {
            enable_prediction: true,
            prediction_algorithms: vec!["linear_regression".to_string()],
            prediction_horizon: 86400 * 30, // 30 days
            models: vec![CostPredictionModel::LinearRegression],
            forecasting: ForecastingConfig::default(),
            accuracy_monitoring: AccuracyMonitoringConfig::default(),
        }
    }
}

impl Default for ForecastingConfig {
    fn default() -> Self {
        Self {
            horizons: vec![],
            confidence_intervals: vec![0.95],
            scenario_analysis: ScenarioAnalysisConfig::default(),
        }
    }
}

impl Default for ScenarioAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            scenarios: vec![],
            probability_weights: HashMap::new(),
        }
    }
}

impl Default for AccuracyMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics: vec![AccuracyMetric::MAPE],
            threshold_alerts: vec![],
            retraining: ModelRetrainingConfig::default(),
        }
    }
}

impl Default for ModelRetrainingConfig {
    fn default() -> Self {
        Self {
            auto_retrain: true,
            triggers: vec![RetrainingTrigger::AccuracyDrop],
            frequency: Duration::from_secs(86400 * 7), // weekly
        }
    }
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            daily_budget: None,
            monthly_budget: Some(1000.0),
            auto_scaling_budget: false,
            budget_alerts: vec![],
            allocation: BudgetAllocation::default(),
            tracking: BudgetTrackingConfig::default(),
            enforcement: BudgetEnforcementConfig::default(),
        }
    }
}

impl Default for BudgetAllocation {
    fn default() -> Self {
        Self {
            departments: HashMap::new(),
            projects: HashMap::new(),
            providers: HashMap::new(),
            reserve_percentage: 0.1, // 10% reserve
        }
    }
}

impl Default for BudgetTrackingConfig {
    fn default() -> Self {
        Self {
            frequency: Duration::from_secs(3600), // hourly
            variance_analysis: VarianceAnalysisConfig::default(),
            trend_analysis: TrendAnalysisConfig::default(),
        }
    }
}

impl Default for VarianceAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            thresholds: VarianceThresholds::default(),
            root_cause_analysis: false,
        }
    }
}

impl Default for VarianceThresholds {
    fn default() -> Self {
        Self {
            warning_threshold: 0.1,  // 10%
            critical_threshold: 0.2, // 20%
            is_percentage: true,
        }
    }
}

impl Default for TrendAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            analysis_window: Duration::from_secs(86400 * 30), // 30 days
            detection_methods: vec![TrendDetectionMethod::MovingAverage],
        }
    }
}

impl Default for BudgetEnforcementConfig {
    fn default() -> Self {
        Self {
            policy: EnforcementPolicy::SoftLimit,
            actions: vec![EnforcementAction::Alert],
            override_permissions: OverridePermissions::default(),
        }
    }
}

impl Default for OverridePermissions {
    fn default() -> Self {
        Self {
            authorized_users: vec![],
            approval_workflow: ApprovalWorkflow::default(),
            override_limits: OverrideLimits::default(),
        }
    }
}

impl Default for ApprovalWorkflow {
    fn default() -> Self {
        Self {
            required_approvers: 1,
            levels: vec![],
            timeout: Duration::from_secs(86400), // 24 hours
        }
    }
}

impl Default for OverrideLimits {
    fn default() -> Self {
        Self {
            max_override_percent: 0.2, // 20%
            frequency_limits: FrequencyLimits::default(),
            justification_required: true,
        }
    }
}

impl Default for FrequencyLimits {
    fn default() -> Self {
        Self {
            per_day: 3,
            per_week: 10,
            per_month: 20,
        }
    }
}

impl Default for CostAlertingConfig {
    fn default() -> Self {
        Self {
            enable_alerts: true,
            alert_thresholds: vec![0.8, 0.9, 1.0], // 80%, 90%, 100% of budget
            notification_channels: vec!["email".to_string()],
            rules: vec![],
            escalation_policies: vec![],
            suppression_rules: vec![],
        }
    }
}

impl Default for FinancialReportingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            report_types: vec![ReportType::CostSummary, ReportType::BudgetVariance],
            frequency: ReportFrequency::Monthly,
            delivery: ReportDeliveryConfig::default(),
        }
    }
}

impl Default for ReportDeliveryConfig {
    fn default() -> Self {
        Self {
            methods: vec![DeliveryMethod::Email],
            recipients: vec![],
            formats: vec![ReportFormat::PDF],
        }
    }
}

impl Default for CostAllocationConfig {
    fn default() -> Self {
        Self {
            methods: vec![AllocationMethod::UsageBasedAllocation],
            rules: vec![],
            chargeback: ChargebackConfig::default(),
        }
    }
}

impl Default for ChargebackConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: ChargebackModel::Showback,
            billing_cycle: BillingCycle::Monthly,
            rate_cards: vec![],
        }
    }
}
