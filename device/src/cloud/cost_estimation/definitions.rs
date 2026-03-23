//! Cost Estimation APIs for Quantum Cloud Services — type definitions.
//!
//! Split from cost_estimation.rs for size compliance.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as TokioRwLock;
use uuid::Uuid;

use super::super::{CloudProvider, ExecutionConfig, QuantumCloudConfig, WorkloadSpec};
use crate::{DeviceError, DeviceResult};

/// Cost estimation engine for quantum cloud services
pub struct CostEstimationEngine {
    pub config: CostEstimationConfig,
    pub pricing_models: HashMap<CloudProvider, ProviderPricingModel>,
    pub cost_predictors: HashMap<String, Box<dyn CostPredictor + Send + Sync>>,
    pub budget_analyzer: Arc<TokioRwLock<BudgetAnalyzer>>,
    pub cost_optimizer: Arc<TokioRwLock<CostOptimizer>>,
    pub pricing_cache: Arc<TokioRwLock<PricingCache>>,
    pub cost_history: Arc<TokioRwLock<CostHistory>>,
}

/// Cost estimation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimationConfig {
    pub enabled: bool,
    pub estimation_accuracy_level: EstimationAccuracyLevel,
    pub pricing_update_frequency: Duration,
    pub include_hidden_costs: bool,
    pub currency: String,
    pub tax_rate: f64,
    pub discount_thresholds: Vec<DiscountThreshold>,
    pub cost_categories: Vec<CostCategory>,
    pub predictive_modeling: PredictiveModelingConfig,
    pub budget_tracking: BudgetTrackingConfig,
}

/// Estimation accuracy levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EstimationAccuracyLevel {
    Quick,
    Standard,
    Detailed,
    Comprehensive,
    RealTime,
}

/// Discount thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscountThreshold {
    pub threshold_type: ThresholdType,
    pub minimum_amount: f64,
    pub discount_percentage: f64,
    pub applicable_services: Vec<String>,
    pub time_period: Duration,
}

/// Threshold types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdType {
    Volume,
    Frequency,
    Duration,
    Commitment,
    Loyalty,
}

/// Cost categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CostCategory {
    Compute,
    Storage,
    Network,
    Management,
    Support,
    Licensing,
    Compliance,
    DataTransfer,
    Backup,
    Security,
}

/// Predictive modeling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveModelingConfig {
    pub enabled: bool,
    pub model_types: Vec<PredictiveModelType>,
    pub forecast_horizon: Duration,
    pub confidence_intervals: bool,
    pub seasonal_adjustments: bool,
    pub trend_analysis: bool,
    pub anomaly_detection: bool,
}

/// Predictive model types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictiveModelType {
    Linear,
    TimeSeries,
    MachineLearning,
    StatisticalRegression,
    NeuralNetwork,
    EnsembleMethods,
}

/// Budget tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetTrackingConfig {
    pub enabled: bool,
    pub budget_periods: Vec<BudgetPeriod>,
    pub alert_thresholds: Vec<f64>,
    pub auto_scaling_on_budget: bool,
    pub cost_allocation_tracking: bool,
    pub variance_analysis: bool,
}

/// Budget periods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetPeriod {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
    Custom(Duration),
}

/// Provider pricing model
#[derive(Debug, Clone)]
pub struct ProviderPricingModel {
    pub provider: CloudProvider,
    pub pricing_structure: PricingStructure,
    pub service_pricing: HashMap<String, ServicePricing>,
    pub volume_discounts: Vec<VolumeDiscount>,
    pub promotional_offers: Vec<PromotionalOffer>,
    pub regional_pricing: HashMap<String, RegionalPricingAdjustment>,
    pub last_updated: SystemTime,
}

/// Pricing structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingStructure {
    pub base_pricing_model: BasePricingModel,
    pub billing_granularity: BillingGranularity,
    pub minimum_charges: HashMap<String, f64>,
    pub setup_fees: HashMap<String, f64>,
    pub termination_fees: HashMap<String, f64>,
}

/// Base pricing models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BasePricingModel {
    PayPerUse,
    Subscription,
    Reserved,
    Spot,
    Hybrid,
    Freemium,
}

/// Billing granularity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BillingGranularity {
    Second,
    Minute,
    Hour,
    Day,
    Month,
    Transaction,
    Resource,
}

/// Service pricing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePricing {
    pub service_name: String,
    pub pricing_tiers: Vec<PricingTier>,
    pub usage_metrics: Vec<UsageMetric>,
    pub cost_components: Vec<CostComponent>,
    pub billing_model: ServiceBillingModel,
}

/// Pricing tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingTier {
    pub tier_name: String,
    pub tier_level: usize,
    pub usage_range: UsageRange,
    pub unit_price: f64,
    pub included_quota: Option<f64>,
    pub overage_price: Option<f64>,
}

/// Usage range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRange {
    pub min_usage: f64,
    pub max_usage: Option<f64>,
    pub unit: String,
}

/// Usage metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetric {
    pub metric_name: String,
    pub metric_type: UsageMetricType,
    pub unit: String,
    pub measurement_interval: Duration,
    pub aggregation_method: AggregationMethod,
}

/// Usage metric types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsageMetricType {
    Cumulative,
    Peak,
    Average,
    Minimum,
    Maximum,
    Count,
    Duration,
}

/// Aggregation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregationMethod {
    Sum,
    Average,
    Maximum,
    Minimum,
    Count,
    Percentile(f64),
}

/// Cost component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostComponent {
    pub component_name: String,
    pub component_type: CostComponentType,
    pub pricing_formula: PricingFormula,
    pub dependencies: Vec<String>,
    pub optional: bool,
}

/// Cost component types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CostComponentType {
    Fixed,
    Variable,
    Tiered,
    StepFunction,
    Custom,
}

/// Pricing formula
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingFormula {
    pub formula_type: FormulaType,
    pub parameters: HashMap<String, f64>,
    pub variables: Vec<String>,
    pub expression: String,
}

/// Formula types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FormulaType {
    Linear,
    Polynomial,
    Exponential,
    Logarithmic,
    Piecewise,
    Custom,
}

/// Service billing model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceBillingModel {
    pub billing_cycle: BillingCycle,
    pub payment_terms: PaymentTerms,
    pub late_fees: Option<LateFees>,
    pub refund_policy: RefundPolicy,
}

/// Billing cycle
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BillingCycle {
    RealTime,
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annually,
}

/// Payment terms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentTerms {
    pub payment_due_days: u32,
    pub early_payment_discount: Option<f64>,
    pub accepted_payment_methods: Vec<PaymentMethod>,
    pub automatic_payment: bool,
}

/// Payment methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaymentMethod {
    CreditCard,
    BankTransfer,
    DigitalWallet,
    Cryptocurrency,
    InvoiceBilling,
    PurchaseOrder,
}

/// Late fees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateFees {
    pub late_fee_percentage: f64,
    pub grace_period_days: u32,
    pub maximum_late_fee: Option<f64>,
    pub compound_interest: bool,
}

/// Refund policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefundPolicy {
    pub refund_eligibility_days: u32,
    pub partial_refunds_allowed: bool,
    pub refund_processing_time: Duration,
    pub refund_conditions: Vec<String>,
}

/// Volume discount
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeDiscount {
    pub discount_name: String,
    pub volume_threshold: f64,
    pub discount_percentage: f64,
    pub applicable_services: Vec<String>,
    pub discount_cap: Option<f64>,
    pub validity_period: Option<Duration>,
}

/// Promotional offer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionalOffer {
    pub offer_name: String,
    pub offer_type: OfferType,
    pub discount_value: f64,
    pub applicable_services: Vec<String>,
    pub eligibility_criteria: Vec<String>,
    pub start_date: SystemTime,
    pub end_date: SystemTime,
    pub usage_limit: Option<f64>,
}

/// Offer types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OfferType {
    PercentageDiscount,
    FixedAmountDiscount,
    FreeUsage,
    UpgradePromotion,
    BundleDiscount,
}

/// Regional pricing adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalPricingAdjustment {
    pub region: String,
    pub currency: String,
    pub exchange_rate: f64,
    pub tax_rate: f64,
    pub regulatory_fees: f64,
    pub local_adjustments: HashMap<String, f64>,
}

/// Cost predictor trait
pub trait CostPredictor {
    fn predict_cost(
        &self,
        workload: &WorkloadSpec,
        config: &ExecutionConfig,
        time_horizon: Duration,
    ) -> DeviceResult<CostPrediction>;
    fn get_predictor_name(&self) -> String;
    fn get_confidence_level(&self) -> f64;
}

/// Cost prediction
#[derive(Debug, Clone)]
pub struct CostPrediction {
    pub prediction_id: String,
    pub workload_id: String,
    pub predicted_cost: f64,
    pub cost_breakdown: DetailedCostBreakdown,
    pub confidence_interval: (f64, f64),
    pub prediction_horizon: Duration,
    pub assumptions: Vec<CostAssumption>,
    pub risk_factors: Vec<RiskFactor>,
    pub optimization_opportunities: Vec<CostOptimizationOpportunity>,
}

/// Detailed cost breakdown
#[derive(Debug, Clone)]
pub struct DetailedCostBreakdown {
    pub base_costs: HashMap<CostCategory, f64>,
    pub variable_costs: HashMap<String, f64>,
    pub fixed_costs: HashMap<String, f64>,
    pub taxes_and_fees: f64,
    pub discounts_applied: f64,
    pub total_cost: f64,
    pub cost_per_unit: HashMap<String, f64>,
}

/// Cost assumption
#[derive(Debug, Clone)]
pub struct CostAssumption {
    pub assumption_type: AssumptionType,
    pub description: String,
    pub impact_on_cost: f64,
    pub confidence: f64,
    pub sensitivity: f64,
}

/// Assumption types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssumptionType {
    PricingStability,
    UsagePattern,
    ResourceAvailability,
    PerformanceCharacteristics,
    ExternalFactors,
}

/// Risk factor
#[derive(Debug, Clone)]
pub struct RiskFactor {
    pub risk_type: RiskType,
    pub description: String,
    pub probability: f64,
    pub potential_cost_impact: f64,
    pub mitigation_strategies: Vec<String>,
}

/// Risk types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RiskType {
    PriceVolatility,
    DemandSpike,
    ResourceScarcity,
    TechnicalFailure,
    PolicyChange,
    MarketConditions,
}

/// Cost optimization opportunity
#[derive(Debug, Clone)]
pub struct CostOptimizationOpportunity {
    pub opportunity_id: String,
    pub opportunity_type: OptimizationType,
    pub description: String,
    pub potential_savings: f64,
    pub implementation_effort: ImplementationEffort,
    pub implementation_time: Duration,
    pub confidence: f64,
}

/// Optimization types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationType {
    ProviderSwitch,
    ServiceTierChange,
    UsageOptimization,
    SchedulingOptimization,
    VolumeDiscount,
    ReservedCapacity,
    SpotInstances,
}

/// Implementation effort
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Budget analyzer
pub struct BudgetAnalyzer {
    pub current_budgets: HashMap<String, Budget>,
    pub budget_performance: HashMap<String, BudgetPerformance>,
    pub variance_analyzer: VarianceAnalyzer,
    pub forecast_engine: BudgetForecastEngine,
}

/// Budget
#[derive(Debug, Clone)]
pub struct Budget {
    pub budget_id: String,
    pub budget_name: String,
    pub budget_period: BudgetPeriod,
    pub allocated_amount: f64,
    pub spent_amount: f64,
    pub remaining_amount: f64,
    pub cost_centers: HashMap<String, f64>,
    pub alert_thresholds: Vec<AlertThreshold>,
    pub auto_adjustments: Vec<AutoAdjustment>,
}

/// Budget performance
#[derive(Debug, Clone)]
pub struct BudgetPerformance {
    pub budget_id: String,
    pub utilization_rate: f64,
    pub spending_velocity: f64,
    pub variance_from_plan: f64,
    pub efficiency_score: f64,
    pub trend_direction: TrendDirection,
    pub performance_metrics: HashMap<String, f64>,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Alert threshold
#[derive(Debug, Clone)]
pub struct AlertThreshold {
    pub threshold_type: ThresholdType,
    pub threshold_value: f64,
    pub alert_severity: AlertSeverity,
    pub notification_channels: Vec<NotificationChannel>,
    pub auto_actions: Vec<AutoAction>,
}

/// Alert severity
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Notification channels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotificationChannel {
    Email,
    SMS,
    Slack,
    Webhook,
    Dashboard,
    PagerDuty,
}

/// Auto action
#[derive(Debug, Clone)]
pub struct AutoAction {
    pub action_type: AutoActionType,
    pub action_parameters: HashMap<String, String>,
    pub conditions: Vec<String>,
    pub approval_required: bool,
}

/// Auto action types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoActionType {
    ScaleDown,
    SuspendJobs,
    SwitchProvider,
    NotifyAdministrator,
    ActivateBackupPlan,
    ApplyEmergencyBudget,
}

/// Auto adjustment
#[derive(Debug, Clone)]
pub struct AutoAdjustment {
    pub adjustment_type: AdjustmentType,
    pub trigger_conditions: Vec<String>,
    pub adjustment_amount: f64,
    pub maximum_adjustments: usize,
    pub cool_down_period: Duration,
}

/// Adjustment types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdjustmentType {
    BudgetIncrease,
    BudgetDecrease,
    Reallocation,
    TemporaryExtension,
    EmergencyFund,
}

/// Variance analyzer
pub struct VarianceAnalyzer {
    pub variance_models: Vec<Box<dyn VarianceModel + Send + Sync>>,
    pub statistical_analyzers: Vec<Box<dyn StatisticalAnalyzer + Send + Sync>>,
    pub trend_detectors: Vec<Box<dyn TrendDetector + Send + Sync>>,
}

/// Variance model trait
pub trait VarianceModel {
    fn analyze_variance(
        &self,
        budget: &Budget,
        actual_spending: &[SpendingRecord],
    ) -> DeviceResult<VarianceAnalysis>;
    fn get_model_name(&self) -> String;
}

/// Variance analysis
#[derive(Debug, Clone)]
pub struct VarianceAnalysis {
    pub total_variance: f64,
    pub variance_percentage: f64,
    pub variance_components: HashMap<String, f64>,
    pub variance_causes: Vec<VarianceCause>,
    pub statistical_significance: f64,
    pub recommendations: Vec<VarianceRecommendation>,
}

/// Variance cause
#[derive(Debug, Clone)]
pub struct VarianceCause {
    pub cause_type: VarianceCauseType,
    pub description: String,
    pub contribution_percentage: f64,
    pub controllability: ControllabilityLevel,
}

/// Variance cause types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VarianceCauseType {
    VolumeVariance,
    RateVariance,
    MixVariance,
    EfficiencyVariance,
    TimingVariance,
    ExternalFactors,
}

/// Controllability level
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControllabilityLevel {
    FullyControllable,
    PartiallyControllable,
    Influenceable,
    Uncontrollable,
}

/// Variance recommendation
#[derive(Debug, Clone)]
pub struct VarianceRecommendation {
    pub recommendation_type: VarianceRecommendationType,
    pub description: String,
    pub priority: RecommendationPriority,
    pub expected_impact: f64,
    pub implementation_timeline: Duration,
}

/// Variance recommendation types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VarianceRecommendationType {
    BudgetAdjustment,
    ProcessImprovement,
    CostControl,
    ResourceOptimization,
    ForecastRefinement,
}

/// Recommendation priority
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Statistical analyzer trait
pub trait StatisticalAnalyzer {
    fn analyze_spending_patterns(
        &self,
        spending_data: &[SpendingRecord],
    ) -> DeviceResult<StatisticalAnalysis>;
    fn get_analyzer_name(&self) -> String;
}

/// Statistical analysis
#[derive(Debug, Clone)]
pub struct StatisticalAnalysis {
    pub descriptive_statistics: DescriptiveStatistics,
    pub correlation_analysis: CorrelationAnalysis,
    pub regression_analysis: Option<RegressionAnalysis>,
    pub anomaly_detection: AnomalyDetectionResult,
    pub seasonal_decomposition: Option<SeasonalDecomposition>,
}

/// Descriptive statistics
#[derive(Debug, Clone)]
pub struct DescriptiveStatistics {
    pub mean: f64,
    pub median: f64,
    pub standard_deviation: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub percentiles: HashMap<f64, f64>,
}

/// Correlation analysis
#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    pub correlationmatrix: Vec<Vec<f64>>,
    pub variable_names: Vec<String>,
    pub significant_correlations: Vec<(String, String, f64, f64)>, // (var1, var2, correlation, p_value)
}

/// Regression analysis
#[derive(Debug, Clone)]
pub struct RegressionAnalysis {
    pub model_type: RegressionModelType,
    pub coefficients: Vec<f64>,
    pub r_squared: f64,
    pub adjusted_r_squared: f64,
    pub f_statistic: f64,
    pub p_values: Vec<f64>,
    pub residual_analysis: ResidualAnalysis,
}

/// Regression model types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegressionModelType {
    Linear,
    Polynomial,
    Logistic,
    MultipleRegression,
    StepwiseRegression,
}

/// Residual analysis
#[derive(Debug, Clone)]
pub struct ResidualAnalysis {
    pub residuals: Vec<f64>,
    pub standardized_residuals: Vec<f64>,
    pub residual_patterns: Vec<ResidualPattern>,
    pub outliers: Vec<usize>,
    pub normality_test: NormalityTest,
}

/// Residual patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidualPattern {
    Random,
    Heteroscedastic,
    AutoCorrelated,
    NonLinear,
    Seasonal,
}

/// Normality test
#[derive(Debug, Clone)]
pub struct NormalityTest {
    pub test_name: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub is_normal: bool,
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct AnomalyDetectionResult {
    pub anomalies_detected: Vec<Anomaly>,
    pub anomaly_score: f64,
    pub detection_method: String,
    pub confidence_level: f64,
}

/// Anomaly
#[derive(Debug, Clone)]
pub struct Anomaly {
    pub anomaly_id: String,
    pub timestamp: SystemTime,
    pub anomaly_type: AnomalyType,
    pub severity: f64,
    pub description: String,
    pub affected_metrics: Vec<String>,
}

/// Anomaly types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnomalyType {
    PointAnomaly,
    ContextualAnomaly,
    CollectiveAnomaly,
    TrendAnomaly,
    SeasonalAnomaly,
}

/// Seasonal decomposition
#[derive(Debug, Clone)]
pub struct SeasonalDecomposition {
    pub trend_component: Vec<f64>,
    pub seasonal_component: Vec<f64>,
    pub residual_component: Vec<f64>,
    pub seasonal_periods: Vec<Duration>,
    pub trend_direction: TrendDirection,
}

/// Trend detector trait
pub trait TrendDetector {
    fn detect_trends(&self, spending_data: &[SpendingRecord])
        -> DeviceResult<TrendDetectionResult>;
    fn get_detector_name(&self) -> String;
}

/// Trend detection result
#[derive(Debug, Clone)]
pub struct TrendDetectionResult {
    pub trends_detected: Vec<Trend>,
    pub overall_trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub trend_significance: f64,
}

/// Trend
#[derive(Debug, Clone)]
pub struct Trend {
    pub trend_id: String,
    pub trend_type: TrendType,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub trend_direction: TrendDirection,
    pub trend_magnitude: f64,
    pub affected_categories: Vec<CostCategory>,
}

/// Trend types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendType {
    Linear,
    Exponential,
    Cyclical,
    Seasonal,
    Irregular,
}

/// Budget forecast engine
pub struct BudgetForecastEngine {
    pub forecast_models: Vec<Box<dyn ForecastModel + Send + Sync>>,
    pub scenario_generators: Vec<Box<dyn ScenarioGenerator + Send + Sync>>,
    pub uncertainty_quantifiers: Vec<Box<dyn UncertaintyQuantifier + Send + Sync>>,
}

/// Forecast model trait
pub trait ForecastModel {
    fn generate_forecast(
        &self,
        historical_data: &[SpendingRecord],
        forecast_horizon: Duration,
    ) -> DeviceResult<BudgetForecast>;
    fn get_model_name(&self) -> String;
    fn get_model_accuracy(&self) -> f64;
}

/// Budget forecast
#[derive(Debug, Clone)]
pub struct BudgetForecast {
    pub forecast_id: String,
    pub forecast_horizon: Duration,
    pub forecasted_values: Vec<ForecastPoint>,
    pub confidence_intervals: Vec<ConfidenceInterval>,
    pub forecast_accuracy_metrics: ForecastAccuracyMetrics,
    pub scenarios: Vec<ForecastScenario>,
}

/// Forecast point
#[derive(Debug, Clone)]
pub struct ForecastPoint {
    pub timestamp: SystemTime,
    pub predicted_value: f64,
    pub prediction_interval: (f64, f64),
    pub contributing_factors: HashMap<String, f64>,
}

/// Confidence interval
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    pub confidence_level: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub interval_width: f64,
}

/// Forecast accuracy metrics
#[derive(Debug, Clone)]
pub struct ForecastAccuracyMetrics {
    pub mean_absolute_error: f64,
    pub mean_squared_error: f64,
    pub mean_absolute_percentage_error: f64,
    pub symmetric_mean_absolute_percentage_error: f64,
    pub theil_u_statistic: f64,
}

/// Forecast scenario
#[derive(Debug, Clone)]
pub struct ForecastScenario {
    pub scenario_name: String,
    pub scenario_type: ScenarioType,
    pub probability: f64,
    pub forecasted_values: Vec<f64>,
    pub scenario_assumptions: Vec<String>,
}

/// Scenario types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScenarioType {
    Optimistic,
    Pessimistic,
    MostLikely,
    WorstCase,
    BestCase,
    Stress,
}

/// Scenario generator trait
pub trait ScenarioGenerator {
    fn generate_scenarios(
        &self,
        base_forecast: &BudgetForecast,
        num_scenarios: usize,
    ) -> DeviceResult<Vec<ForecastScenario>>;
    fn get_generator_name(&self) -> String;
}

/// Uncertainty quantifier trait
pub trait UncertaintyQuantifier {
    fn quantify_uncertainty(&self, forecast: &BudgetForecast) -> DeviceResult<UncertaintyAnalysis>;
    fn get_quantifier_name(&self) -> String;
}

/// Uncertainty analysis
#[derive(Debug, Clone)]
pub struct UncertaintyAnalysis {
    pub total_uncertainty: f64,
    pub uncertainty_sources: Vec<UncertaintySource>,
    pub uncertainty_propagation: UncertaintyPropagation,
    pub sensitivity_analysis: SensitivityAnalysis,
}

/// Uncertainty source
#[derive(Debug, Clone)]
pub struct UncertaintySource {
    pub source_name: String,
    pub source_type: UncertaintySourceType,
    pub contribution_percentage: f64,
    pub uncertainty_distribution: UncertaintyDistribution,
}

/// Uncertainty source types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UncertaintySourceType {
    ModelUncertainty,
    ParameterUncertainty,
    InputDataUncertainty,
    StructuralUncertainty,
    ExternalFactors,
}

/// Uncertainty distribution
#[derive(Debug, Clone)]
pub struct UncertaintyDistribution {
    pub distribution_type: DistributionType,
    pub parameters: HashMap<String, f64>,
    pub confidence_level: f64,
}

/// Distribution types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributionType {
    Normal,
    LogNormal,
    Uniform,
    Triangular,
    Beta,
    Gamma,
    Exponential,
}

/// Uncertainty propagation
#[derive(Debug, Clone)]
pub struct UncertaintyPropagation {
    pub propagation_method: PropagationMethod,
    pub correlation_effects: Vec<CorrelationEffect>,
    pub amplification_factors: HashMap<String, f64>,
}

/// Propagation methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropagationMethod {
    MonteCarlo,
    LinearPropagation,
    TaylorSeries,
    Polynomial,
    Sampling,
}

/// Correlation effect
#[derive(Debug, Clone)]
pub struct CorrelationEffect {
    pub source1: String,
    pub source2: String,
    pub correlation_coefficient: f64,
    pub effect_magnitude: f64,
}

/// Sensitivity analysis
#[derive(Debug, Clone)]
pub struct SensitivityAnalysis {
    pub sensitivity_indices: HashMap<String, f64>,
    pub interaction_effects: Vec<InteractionEffect>,
    pub robust_parameters: Vec<String>,
    pub critical_parameters: Vec<String>,
}

/// Interaction effect
#[derive(Debug, Clone)]
pub struct InteractionEffect {
    pub parameters: Vec<String>,
    pub interaction_strength: f64,
    pub effect_type: InteractionType,
}

/// Interaction types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InteractionType {
    Synergistic,
    Antagonistic,
    Additive,
    Multiplicative,
}

/// Cost optimizer
pub struct CostOptimizer {
    pub optimization_strategies: Vec<Box<dyn CostOptimizationStrategy + Send + Sync>>,
    pub recommendation_engine: RecommendationEngine,
    pub savings_calculator: SavingsCalculator,
}

/// Cost optimization strategy trait
pub trait CostOptimizationStrategy {
    fn optimize_costs(
        &self,
        cost_analysis: &CostAnalysis,
    ) -> DeviceResult<OptimizationRecommendation>;
    fn get_strategy_name(&self) -> String;
    fn get_potential_savings(&self, cost_analysis: &CostAnalysis) -> DeviceResult<f64>;
}

/// Cost analysis
#[derive(Debug, Clone)]
pub struct CostAnalysis {
    pub total_costs: f64,
    pub cost_breakdown: DetailedCostBreakdown,
    pub cost_trends: Vec<CostTrend>,
    pub cost_drivers: Vec<CostDriver>,
    pub benchmark_comparison: BenchmarkComparison,
    pub inefficiencies: Vec<CostInefficiency>,
}

/// Cost trend
#[derive(Debug, Clone)]
pub struct CostTrend {
    pub trend_id: String,
    pub cost_category: CostCategory,
    pub trend_direction: TrendDirection,
    pub trend_rate: f64,
    pub trend_duration: Duration,
    pub projected_impact: f64,
}

/// Cost driver
#[derive(Debug, Clone)]
pub struct CostDriver {
    pub driver_name: String,
    pub driver_type: CostDriverType,
    pub impact_magnitude: f64,
    pub controllability: ControllabilityLevel,
    pub optimization_potential: f64,
}

/// Cost driver types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CostDriverType {
    Volume,
    Complexity,
    Quality,
    Speed,
    Flexibility,
    Risk,
}

/// Benchmark comparison
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    pub benchmark_type: BenchmarkType,
    pub comparison_metrics: HashMap<String, BenchmarkMetric>,
    pub relative_performance: f64,
    pub improvement_opportunities: Vec<ImprovementOpportunity>,
}

/// Benchmark types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkType {
    IndustryAverage,
    BestInClass,
    Historical,
    Internal,
    Competitor,
}

/// Benchmark metric
#[derive(Debug, Clone)]
pub struct BenchmarkMetric {
    pub metric_name: String,
    pub current_value: f64,
    pub benchmark_value: f64,
    pub variance_percentage: f64,
    pub performance_gap: f64,
}

/// Improvement opportunity
#[derive(Debug, Clone)]
pub struct ImprovementOpportunity {
    pub opportunity_id: String,
    pub opportunity_area: String,
    pub current_performance: f64,
    pub target_performance: f64,
    pub potential_savings: f64,
    pub implementation_complexity: f64,
}

/// Cost inefficiency
#[derive(Debug, Clone)]
pub struct CostInefficiency {
    pub inefficiency_type: InefficiencyType,
    pub description: String,
    pub cost_impact: f64,
    pub frequency: f64,
    pub root_causes: Vec<String>,
    pub remediation_options: Vec<RemediationOption>,
}

/// Inefficiency types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InefficiencyType {
    Overprovisioning,
    Underutilization,
    Redundancy,
    ProcessInefficiency,
    TechnologyMismatch,
    VendorInefficiency,
}

/// Remediation option
#[derive(Debug, Clone)]
pub struct RemediationOption {
    pub option_name: String,
    pub implementation_effort: ImplementationEffort,
    pub expected_savings: f64,
    pub implementation_timeline: Duration,
    pub success_probability: f64,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub recommendation_id: String,
    pub recommendation_type: OptimizationType,
    pub priority: RecommendationPriority,
    pub description: String,
    pub potential_savings: f64,
    pub implementation_plan: ImplementationPlan,
    pub risk_assessment: RiskAssessment,
    pub roi_analysis: ROIAnalysis,
}

/// Implementation plan
#[derive(Debug, Clone)]
pub struct ImplementationPlan {
    pub phases: Vec<ImplementationPhase>,
    pub total_duration: Duration,
    pub resource_requirements: Vec<ResourceRequirement>,
    pub dependencies: Vec<String>,
    pub milestones: Vec<Milestone>,
}

/// Implementation phase
#[derive(Debug, Clone)]
pub struct ImplementationPhase {
    pub phase_name: String,
    pub phase_duration: Duration,
    pub phase_activities: Vec<String>,
    pub deliverables: Vec<String>,
    pub success_criteria: Vec<String>,
}

/// Resource requirement
#[derive(Debug, Clone)]
pub struct ResourceRequirement {
    pub resource_type: String,
    pub quantity: f64,
    pub duration: Duration,
    pub cost: f64,
    pub availability: f64,
}

/// Milestone
#[derive(Debug, Clone)]
pub struct Milestone {
    pub milestone_name: String,
    pub target_date: SystemTime,
    pub completion_criteria: Vec<String>,
    pub deliverables: Vec<String>,
}

/// Risk assessment
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    pub overall_risk_score: f64,
    pub risk_factors: Vec<RiskFactor>,
    pub mitigation_strategies: Vec<MitigationStrategy>,
    pub contingency_plans: Vec<ContingencyPlan>,
}

/// Mitigation strategy
#[derive(Debug, Clone)]
pub struct MitigationStrategy {
    pub strategy_name: String,
    pub risk_reduction: f64,
    pub implementation_cost: f64,
    pub effectiveness: f64,
}

/// Contingency plan
#[derive(Debug, Clone)]
pub struct ContingencyPlan {
    pub plan_name: String,
    pub trigger_conditions: Vec<String>,
    pub response_actions: Vec<String>,
    pub resource_requirements: Vec<ResourceRequirement>,
}

/// ROI analysis
#[derive(Debug, Clone)]
pub struct ROIAnalysis {
    pub initial_investment: f64,
    pub annual_savings: f64,
    pub payback_period: Duration,
    pub net_present_value: f64,
    pub internal_rate_of_return: f64,
    pub roi_percentage: f64,
}

/// Recommendation engine
pub struct RecommendationEngine {
    pub recommendation_algorithms: Vec<Box<dyn RecommendationAlgorithm + Send + Sync>>,
    pub scoring_models: Vec<Box<dyn ScoringModel + Send + Sync>>,
    pub prioritization_engine: PrioritizationEngine,
}

/// Recommendation algorithm trait
pub trait RecommendationAlgorithm {
    fn generate_recommendations(
        &self,
        analysis: &CostAnalysis,
    ) -> DeviceResult<Vec<OptimizationRecommendation>>;
    fn get_algorithm_name(&self) -> String;
}

/// Scoring model trait
pub trait ScoringModel {
    fn score_recommendation(
        &self,
        recommendation: &OptimizationRecommendation,
    ) -> DeviceResult<f64>;
    fn get_model_name(&self) -> String;
}

/// Prioritization engine
pub struct PrioritizationEngine {
    pub prioritization_criteria: Vec<PrioritizationCriterion>,
    pub weighting_scheme: WeightingScheme,
    pub decision_matrix: DecisionMatrix,
}

/// Prioritization criterion
#[derive(Debug, Clone)]
pub struct PrioritizationCriterion {
    pub criterion_name: String,
    pub criterion_type: CriterionType,
    pub weight: f64,
    pub scoring_function: ScoringFunction,
}

/// Criterion types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CriterionType {
    Financial,
    Strategic,
    Operational,
    Risk,
    Feasibility,
}

/// Scoring function
#[derive(Debug, Clone)]
pub struct ScoringFunction {
    pub function_type: FunctionType,
    pub parameters: HashMap<String, f64>,
    pub range: (f64, f64),
}

/// Function types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionType {
    Linear,
    Exponential,
    Logarithmic,
    Sigmoid,
    Step,
}

/// Weighting scheme
#[derive(Debug, Clone)]
pub struct WeightingScheme {
    pub scheme_type: WeightingSchemeType,
    pub weights: HashMap<String, f64>,
    pub normalization_method: NormalizationMethod,
}

/// Weighting scheme types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightingSchemeType {
    Equal,
    Hierarchical,
    Expert,
    DataDriven,
    Hybrid,
}

/// Normalization methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizationMethod {
    MinMax,
    ZScore,
    UnitVector,
    Sum,
    Max,
}

/// Decision matrix
#[derive(Debug, Clone)]
pub struct DecisionMatrix {
    pub alternatives: Vec<String>,
    pub criteria: Vec<String>,
    pub scores: Vec<Vec<f64>>,
    pub weights: Vec<f64>,
    pub aggregation_method: AggregationMethod,
}

/// Savings calculator
pub struct SavingsCalculator {
    pub calculation_methods: Vec<Box<dyn SavingsCalculationMethod + Send + Sync>>,
    pub validation_rules: Vec<ValidationRule>,
    pub adjustment_factors: AdjustmentFactors,
}

/// Savings calculation method trait
pub trait SavingsCalculationMethod {
    fn calculate_savings(
        &self,
        baseline_cost: f64,
        optimized_cost: f64,
        implementation_cost: f64,
    ) -> DeviceResult<SavingsCalculation>;
    fn get_method_name(&self) -> String;
}

/// Savings calculation
#[derive(Debug, Clone)]
pub struct SavingsCalculation {
    pub gross_savings: f64,
    pub net_savings: f64,
    pub savings_percentage: f64,
    pub implementation_cost: f64,
    pub ongoing_costs: f64,
    pub payback_period: Duration,
    pub cumulative_savings: Vec<(SystemTime, f64)>,
}

/// Validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub rule_name: String,
    pub rule_type: ValidationRuleType,
    pub condition: String,
    pub action: String,
    pub severity: ValidationSeverity,
}

/// Validation rule types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationRuleType {
    BusinessRule,
    TechnicalConstraint,
    RegulatoryRequirement,
    PolicyCompliance,
    DataQuality,
}

/// Validation severity
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationSeverity {
    Warning,
    Error,
    Critical,
    Info,
}

/// Adjustment factors
#[derive(Debug, Clone)]
pub struct AdjustmentFactors {
    pub risk_adjustment: f64,
    pub confidence_adjustment: f64,
    pub market_adjustment: f64,
    pub seasonal_adjustment: f64,
    pub inflation_adjustment: f64,
}

/// Pricing cache
pub struct PricingCache {
    pub cache_entries: HashMap<String, PricingCacheEntry>,
    pub cache_statistics: CacheStatistics,
    pub eviction_policy: EvictionPolicy,
}

/// Pricing cache entry
#[derive(Debug, Clone)]
pub struct PricingCacheEntry {
    pub entry_id: String,
    pub provider: CloudProvider,
    pub service: String,
    pub pricing_data: ServicePricing,
    pub timestamp: SystemTime,
    pub validity_period: Duration,
    pub access_count: usize,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub hit_rate: f64,
    pub miss_rate: f64,
    pub eviction_rate: f64,
    pub average_lookup_time: Duration,
    pub total_entries: usize,
}

/// Eviction policy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    TTL,
    Random,
}

/// Cost history
pub struct CostHistory {
    pub spending_records: Vec<SpendingRecord>,
    pub aggregated_costs: HashMap<String, Vec<AggregatedCost>>,
    pub cost_trends: HashMap<String, CostTrend>,
    pub historical_analysis: HistoricalAnalysis,
}

/// Spending record
#[derive(Debug, Clone)]
pub struct SpendingRecord {
    pub record_id: String,
    pub timestamp: SystemTime,
    pub provider: CloudProvider,
    pub service: String,
    pub cost_amount: f64,
    pub currency: String,
    pub cost_category: CostCategory,
    pub usage_metrics: HashMap<String, f64>,
    pub billing_period: (SystemTime, SystemTime),
}

/// Aggregated cost
#[derive(Debug, Clone)]
pub struct AggregatedCost {
    pub aggregation_period: (SystemTime, SystemTime),
    pub total_cost: f64,
    pub cost_breakdown: HashMap<CostCategory, f64>,
    pub usage_summary: HashMap<String, f64>,
    pub cost_per_unit: HashMap<String, f64>,
}

/// Historical analysis
#[derive(Debug, Clone)]
pub struct HistoricalAnalysis {
    pub cost_growth_rate: f64,
    pub seasonal_patterns: Vec<SeasonalPattern>,
    pub cost_volatility: f64,
    pub efficiency_trends: Vec<EfficiencyTrend>,
    pub comparative_analysis: ComparativeAnalysis,
}

/// Seasonal pattern
#[derive(Debug, Clone)]
pub struct SeasonalPattern {
    pub pattern_name: String,
    pub period: Duration,
    pub amplitude: f64,
    pub phase_shift: Duration,
    pub significance: f64,
}

/// Efficiency trend
#[derive(Debug, Clone)]
pub struct EfficiencyTrend {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub improvement_rate: f64,
    pub efficiency_score: f64,
}

/// Comparative analysis
#[derive(Debug, Clone)]
pub struct ComparativeAnalysis {
    pub period_comparisons: Vec<PeriodComparison>,
    pub provider_comparisons: Vec<ProviderComparison>,
    pub service_comparisons: Vec<ServiceComparison>,
}

/// Period comparison
#[derive(Debug, Clone)]
pub struct PeriodComparison {
    pub period1: (SystemTime, SystemTime),
    pub period2: (SystemTime, SystemTime),
    pub cost_difference: f64,
    pub percentage_change: f64,
    pub significant_changes: Vec<String>,
}

/// Provider comparison
#[derive(Debug, Clone)]
pub struct ProviderComparison {
    pub provider1: CloudProvider,
    pub provider2: CloudProvider,
    pub cost_comparison: HashMap<String, f64>,
    pub service_comparison: HashMap<String, f64>,
    pub efficiency_comparison: HashMap<String, f64>,
}

/// Service comparison
#[derive(Debug, Clone)]
pub struct ServiceComparison {
    pub service_name: String,
    pub cost_trend: TrendDirection,
    pub usage_trend: TrendDirection,
    pub efficiency_trend: TrendDirection,
    pub optimization_opportunities: Vec<String>,
}
