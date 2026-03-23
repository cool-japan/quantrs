//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

use super::functions::{MLCostModel, NotificationHandler, OptimizationAlgorithm, PredictiveModel};
use crate::translation::HardwareBackend;
use crate::DeviceResult;

/// Cost estimation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimationMetadata {
    /// Model used for estimation
    pub model_used: String,
    /// Estimation timestamp
    pub timestamp: SystemTime,
    /// Confidence level
    pub confidence_level: f64,
    /// Historical accuracy
    pub historical_accuracy: Option<f64>,
    /// Factors considered
    pub factors_considered: Vec<String>,
}
/// Model performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    accuracy_metrics: AccuracyMetrics,
    performance_over_time: VecDeque<(SystemTime, f64)>,
    prediction_distribution: PredictionDistribution,
    feature_drift: HashMap<String, f64>,
}
/// Daily budget status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyBudgetStatus {
    pub daily_budget: f64,
    pub daily_used: f64,
    pub daily_remaining: f64,
    pub projected_daily_usage: f64,
}
/// Resource optimization algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceOptimizationAlgorithm {
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Particle swarm optimization
    ParticleSwarmOptimization,
    /// Linear programming
    LinearProgramming,
    /// Integer programming
    IntegerProgramming,
    /// Constraint satisfaction
    ConstraintSatisfaction,
    /// SciRS2-powered optimization
    SciRS2Optimization,
}
/// Custom visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomVisualization {
    /// Visualization name
    pub name: String,
    /// Chart type
    pub chart_type: ChartType,
    /// Data source
    pub data_source: String,
    /// Configuration parameters
    pub parameters: HashMap<String, serde_json::Value>,
}
/// Multi-objective optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveConfig {
    /// Objective functions
    pub objectives: Vec<OptimizationObjective>,
    /// Pareto frontier configuration
    pub pareto_config: ParetoConfig,
    /// Solution selection strategy
    pub selection_strategy: SolutionSelectionStrategy,
}
/// Time-based features for prediction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimeFeature {
    HourOfDay,
    DayOfWeek,
    DayOfMonth,
    Month,
    Season,
    IsWeekend,
    IsHoliday,
    TimeToDeadline,
}
/// Escalation policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicy {
    escalation_levels: Vec<EscalationLevel>,
    max_escalation_attempts: usize,
    escalation_timeout: Duration,
}
/// Cost reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostReportingConfig {
    /// Enable automated reports
    pub automated_reports: bool,
    /// Report frequency
    pub report_frequency: Duration,
    /// Report types
    pub report_types: Vec<ReportType>,
    /// Report recipients
    pub recipients: Vec<String>,
    /// Report format
    pub format: ReportFormat,
}
/// Variable definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub variable_type: VariableType,
    pub bounds: (f64, f64),
    pub initial_value: Option<f64>,
}
/// Training result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub training_score: f64,
    pub validation_score: f64,
    pub feature_importance: HashMap<String, f64>,
    pub training_time: Duration,
    pub model_parameters: HashMap<String, f64>,
}
/// Time-based pricing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeBasedPricing {
    /// Peak hours pricing multiplier
    pub peak_multiplier: f64,
    /// Off-peak hours pricing multiplier
    pub off_peak_multiplier: f64,
    /// Peak hours definition
    pub peak_hours: Vec<(u8, u8)>,
    /// Weekend pricing multiplier
    pub weekend_multiplier: Option<f64>,
    /// Holiday pricing multiplier
    pub holiday_multiplier: Option<f64>,
}
/// Prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub predicted_value: f64,
    pub confidence_interval: (f64, f64),
    pub feature_contributions: HashMap<String, f64>,
    pub model_used: String,
    pub prediction_timestamp: SystemTime,
}
/// Ensemble configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    pub ensemble_method: EnsembleMethod,
    pub model_weights: HashMap<String, f64>,
    pub voting_strategy: VotingStrategy,
    pub diversity_threshold: f64,
}
/// Real-time metrics per provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMetrics {
    current_queue_length: usize,
    average_execution_time: Duration,
    current_error_rate: f64,
    availability_status: AvailabilityStatus,
    cost_fluctuation: f64,
    last_updated: SystemTime,
}
/// Trend model types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendModelType {
    Linear,
    Exponential,
    Polynomial,
    Seasonal,
    ARIMA,
    ExponentialSmoothing,
}
/// Cost models for different providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    /// Model type
    pub model_type: CostModelType,
    /// Base cost per shot
    pub base_cost_per_shot: f64,
    /// Cost per qubit
    pub cost_per_qubit: f64,
    /// Cost per gate
    pub cost_per_gate: f64,
    /// Cost per second of execution
    pub cost_per_second: f64,
    /// Setup/teardown cost
    pub setup_cost: f64,
    /// Queue time multiplier
    pub queue_time_multiplier: f64,
    /// Peak/off-peak pricing
    pub time_based_pricing: Option<TimeBasedPricing>,
    /// Volume discounts
    pub volume_discounts: Vec<VolumeDiscount>,
    /// Custom cost factors
    pub custom_factors: HashMap<String, f64>,
}
/// Types of predictive models
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PredictiveModelType {
    /// Linear regression
    LinearRegression,
    /// Random forest
    RandomForest,
    /// Neural network
    NeuralNetwork,
    /// ARIMA time series model
    ARIMA,
    /// Support vector machine
    SVM,
    /// Gradient boosting
    GradientBoosting,
    /// SciRS2-powered models
    SciRS2Enhanced,
}
/// Cross-validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResult {
    pub mean_score: f64,
    pub std_score: f64,
    pub fold_scores: Vec<f64>,
    pub best_parameters: HashMap<String, f64>,
}
/// Resource optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOptimizationConfig {
    /// Enable resource optimization
    pub enabled: bool,
    /// Optimization algorithms
    pub algorithms: Vec<ResourceOptimizationAlgorithm>,
    /// Constraint types
    pub constraints: Vec<ResourceConstraint>,
    /// Optimization frequency
    pub optimization_frequency: Duration,
    /// Parallel optimization
    pub enable_parallel_optimization: bool,
    /// Multi-objective optimization
    pub multi_objective_config: MultiObjectiveConfig,
}
/// Usage-based features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UsageFeature {
    HistoricalCosts,
    UsagePatterns,
    PeakUsageTimes,
    VolumeDiscounts,
    BudgetUtilization,
    CostTrends,
}
/// Cost monitor component
pub struct CostMonitor {
    pub monitoring_metrics: HashMap<MonitoringMetric, MetricTimeSeries>,
    pub anomaly_detector: AnomalyDetector,
    pub trend_analyzer: TrendAnalyzer,
    pub dashboard_data: DashboardData,
}
/// Budget management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Total budget limit
    pub total_budget: f64,
    /// Daily budget limit
    pub daily_budget: Option<f64>,
    /// Monthly budget limit
    pub monthly_budget: Option<f64>,
    /// Budget allocation per provider
    pub provider_budgets: HashMap<HardwareBackend, f64>,
    /// Budget allocation per circuit type
    pub circuit_type_budgets: HashMap<String, f64>,
    /// Enable automatic budget management
    pub auto_budget_management: bool,
    /// Budget rollover policy
    pub rollover_policy: BudgetRolloverPolicy,
}
/// Feature store for predictive modeling
pub struct FeatureStore {
    pub features: HashMap<String, FeatureTimeSeries>,
    pub feature_metadata: HashMap<String, FeatureMetadata>,
    pub derived_features: HashMap<String, DerivedFeature>,
}
/// Budget periods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BudgetPeriod {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
}
/// Optimization objectives
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationObjective {
    MinimizeCost,
    MinimizeTime,
    MaximizeQuality,
    MaximizeReliability,
    MinimizeRisk,
    Custom(String),
}
/// Pattern types in spending
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatternType {
    Constant,
    Linear,
    Exponential,
    Periodic,
    Random,
    Composite,
}
/// Predictive modeling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveModelingConfig {
    /// Enable predictive cost modeling
    pub enabled: bool,
    /// Prediction horizon
    pub prediction_horizon: Duration,
    /// Model types to use
    pub model_types: Vec<PredictiveModelType>,
    /// Feature engineering settings
    pub feature_engineering: FeatureEngineeringConfig,
    /// Model training frequency
    pub training_frequency: Duration,
    /// Prediction confidence threshold
    pub confidence_threshold: f64,
    /// Enable ensemble predictions
    pub enable_ensemble: bool,
}
/// Feature data types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeatureDataType {
    Numerical,
    Categorical,
    Binary,
    TimeStamp,
    Text,
}
/// Cost monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostMonitoringConfig {
    /// Enable real-time monitoring
    pub real_time_monitoring: bool,
    /// Monitoring frequency
    pub monitoring_frequency: Duration,
    /// Metrics to track
    pub tracked_metrics: Vec<MonitoringMetric>,
    /// Reporting configuration
    pub reporting_config: CostReportingConfig,
    /// Dashboard configuration
    pub dashboard_config: Option<DashboardConfig>,
}
/// Anomaly detection methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnomalyDetectionMethod {
    StatisticalOutlier,
    IsolationForest,
    OneClassSVM,
    LocalOutlierFactor,
    DBSCAN,
    TimeSeriesDecomposition,
}
/// Metric time series data
#[derive(Debug, Clone)]
pub struct MetricTimeSeries {
    data_points: VecDeque<(SystemTime, f64)>,
    sampling_frequency: Duration,
    aggregation_method: AggregationMethod,
    retention_period: Duration,
}
/// Volume discount configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeDiscount {
    /// Minimum volume threshold
    pub min_volume: f64,
    /// Maximum volume threshold (None for unlimited)
    pub max_volume: Option<f64>,
    /// Discount percentage (0.0 to 1.0)
    pub discount_percentage: f64,
    /// Discount type
    pub discount_type: DiscountType,
}
/// Spending patterns analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendingPattern {
    pattern_type: PatternType,
    frequency: f64,
    amplitude: f64,
    trend: f64,
    seasonality: Option<SeasonalityPattern>,
}
/// Optimization problem definition
#[derive(Debug, Clone)]
pub struct OptimizationProblem {
    pub objectives: Vec<ObjectiveFunction>,
    pub constraints: Vec<Constraint>,
    pub variables: Vec<Variable>,
    pub problem_type: ProblemType,
}
/// Constraint solver
pub struct ConstraintSolver {
    pub solver_type: SolverType,
    pub tolerance: f64,
    pub max_iterations: usize,
}
/// Constraint definition
pub struct Constraint {
    pub name: String,
    pub constraint_function: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
    pub constraint_type: ConstraintType,
    pub bound: f64,
}
/// Reliability tracker
pub struct ReliabilityTracker {
    pub provider_reliability: HashMap<HardwareBackend, ReliabilityMetrics>,
    pub incident_history: VecDeque<ReliabilityIncident>,
}
/// Notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email {
        recipients: Vec<String>,
        template: String,
    },
    Slack {
        webhook_url: String,
        channel: String,
    },
    Webhook {
        url: String,
        headers: HashMap<String, String>,
    },
    SMS {
        phone_numbers: Vec<String>,
    },
}
/// Cost record for historical tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRecord {
    pub provider: HardwareBackend,
    pub circuit_hash: String,
    pub actual_cost: f64,
    pub estimated_cost: f64,
    pub execution_time: Duration,
    pub timestamp: SystemTime,
    pub metadata: HashMap<String, String>,
}
/// Alert conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    /// Budget threshold exceeded
    BudgetThreshold { threshold: f64, percentage: bool },
    /// Cost spike detected
    CostSpike { multiplier: f64, window: Duration },
    /// Prediction error high
    PredictionError { threshold: f64 },
    /// Optimization opportunity
    OptimizationOpportunity { savings_threshold: f64 },
    /// Custom condition
    Custom { expression: String },
}
/// Training data structure
#[derive(Debug, Clone)]
pub struct TrainingData {
    pub features: Array2<f64>,
    pub targets: Array1<f64>,
    pub feature_names: Vec<String>,
    pub timestamps: Vec<SystemTime>,
}
/// Cost estimator component
pub struct CostEstimator {
    pub models: HashMap<HardwareBackend, CostModel>,
    pub historical_data: VecDeque<CostRecord>,
    pub ml_models: HashMap<String, Box<dyn MLCostModel + Send + Sync>>,
    pub estimation_cache: HashMap<String, CachedEstimate>,
}
/// Cost optimization strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CostOptimizationStrategy {
    /// Minimize total cost
    MinimizeCost,
    /// Maximize cost-performance ratio
    MaximizeCostPerformance,
    /// Stay within budget constraints
    BudgetConstrained,
    /// Optimize for specific metrics
    MetricOptimized {
        cost_weight: f64,
        time_weight: f64,
        quality_weight: f64,
    },
    /// Machine learning-driven optimization
    MLOptimized,
    /// Custom optimization with SciRS2
    SciRS2Optimized {
        objectives: Vec<String>,
        constraints: Vec<String>,
    },
}
/// Predictive modeler component
pub struct PredictiveModeler {
    pub models: HashMap<String, Box<dyn PredictiveModel + Send + Sync>>,
    pub feature_store: FeatureStore,
    pub model_performance: HashMap<String, ModelPerformance>,
    pub ensemble_config: EnsembleConfig,
}
/// Algorithm information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmInfo {
    pub name: String,
    pub description: String,
    pub problem_types: Vec<ProblemType>,
    pub parameters: HashMap<String, ParameterInfo>,
}
/// Optimization directions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationDirection {
    Minimize,
    Maximize,
}
/// Parameter information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    pub name: String,
    pub description: String,
    pub parameter_type: ParameterType,
    pub default_value: f64,
    pub bounds: Option<(f64, f64)>,
}
/// Objective function
pub struct ObjectiveFunction {
    pub name: String,
    pub function: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
    pub optimization_direction: OptimizationDirection,
    pub weight: f64,
}
/// Budget manager component
pub struct BudgetManager {
    pub current_budget: BudgetStatus,
    pub budget_history: VecDeque<BudgetSnapshot>,
    pub spending_patterns: HashMap<String, SpendingPattern>,
    pub budget_alerts: Vec<ActiveBudgetAlert>,
}
/// Provider comparator component
pub struct ProviderComparator {
    pub comparison_cache: HashMap<String, CachedComparison>,
    pub real_time_metrics: HashMap<HardwareBackend, RealTimeMetrics>,
    pub reliability_tracker: ReliabilityTracker,
}
/// Feature engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    /// Time-based features
    pub time_features: Vec<TimeFeature>,
    /// Circuit-based features
    pub circuit_features: Vec<CircuitFeature>,
    /// Provider-based features
    pub provider_features: Vec<ProviderFeature>,
    /// Historical usage features
    pub usage_features: Vec<UsageFeature>,
    /// Feature selection method
    pub feature_selection: FeatureSelectionMethod,
}
/// Ensemble methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnsembleMethod {
    Voting,
    Averaging,
    Stacking,
    Boosting,
    Bagging,
}
/// Alert manager component
pub struct AlertManager {
    pub active_alerts: HashMap<String, ActiveAlert>,
    pub alert_history: VecDeque<AlertHistoryEntry>,
    pub notification_handlers: HashMap<String, Box<dyn NotificationHandler + Send + Sync>>,
    pub escalation_policies: HashMap<AlertSeverity, EscalationPolicy>,
}
/// Alert aggregation strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregationStrategy {
    Count,
    SeverityBased,
    TypeBased,
    Custom(String),
}
/// Cached optimization result
#[derive(Debug, Clone)]
pub struct CachedOptimization {
    result: OptimizationResult,
    input_hash: u64,
    created_at: SystemTime,
    access_count: usize,
    expiry_time: SystemTime,
}
/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub solution: Vec<f64>,
    pub objective_values: Vec<f64>,
    pub constraint_violations: Vec<f64>,
    pub optimization_status: OptimizationStatus,
    pub iterations: usize,
    pub execution_time: Duration,
    pub algorithm_used: String,
}
/// Pareto frontier configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoConfig {
    /// Maximum solutions to maintain
    pub max_solutions: usize,
    /// Convergence criteria
    pub convergence_criteria: ConvergenceCriteria,
    /// Diversity preservation
    pub diversity_preservation: bool,
}
/// Budget rollover policies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BudgetRolloverPolicy {
    /// No rollover - unused budget is lost
    NoRollover,
    /// Full rollover - all unused budget carries over
    FullRollover,
    /// Percentage rollover
    PercentageRollover(f64),
    /// Fixed amount rollover
    FixedAmountRollover(f64),
    /// Capped rollover with maximum
    CappedRollover { percentage: f64, max_amount: f64 },
}
/// Variable types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VariableType {
    Continuous,
    Integer,
    Binary,
}
/// Detailed metrics for provider comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderMetrics {
    /// Cost metrics
    pub cost_metrics: HashMap<ComparisonMetric, f64>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
    /// Reliability metrics
    pub reliability_metrics: HashMap<String, f64>,
    /// Overall score
    pub overall_score: f64,
}
/// Cost estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Total estimated cost
    pub total_cost: f64,
    /// Cost breakdown by component
    pub cost_breakdown: CostBreakdown,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Estimation metadata
    pub metadata: CostEstimationMetadata,
}
/// Budget snapshot for historical tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetSnapshot {
    timestamp: SystemTime,
    budget_status: BudgetStatus,
    period_type: BudgetPeriod,
}
/// Escalation level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    level: usize,
    notification_channels: Vec<String>,
    delay: Duration,
    repeat_frequency: Option<Duration>,
}
/// Pareto frontier for multi-objective optimization
#[derive(Debug, Clone)]
pub struct ParetoFrontier {
    solutions: Vec<ParetoSolution>,
    objectives: Vec<String>,
    generation: usize,
    last_updated: SystemTime,
}
/// Feature metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetadata {
    name: String,
    description: String,
    data_type: FeatureDataType,
    update_frequency: Duration,
    importance_score: f64,
}
/// Trend analyzer
pub struct TrendAnalyzer {
    pub trend_models: HashMap<MonitoringMetric, TrendModel>,
    pub trend_detection_sensitivity: f64,
    pub forecasting_horizon: Duration,
}
/// Seasonality patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityPattern {
    period: Duration,
    phase: f64,
    strength: f64,
}
/// Time series data for features
#[derive(Debug, Clone)]
pub struct FeatureTimeSeries {
    values: VecDeque<(SystemTime, f64)>,
    statistics: FeatureStatistics,
}
/// Dashboard data structure
#[derive(Debug, Clone)]
pub struct DashboardData {
    pub widget_data: HashMap<DashboardWidget, serde_json::Value>,
    pub last_updated: SystemTime,
    pub update_frequency: Duration,
}
/// Resource optimizer component
pub struct ResourceOptimizer {
    pub optimization_algorithms: HashMap<String, Box<dyn OptimizationAlgorithm + Send + Sync>>,
    pub constraint_solver: ConstraintSolver,
    pub optimization_history: VecDeque<OptimizationResult>,
    pub pareto_frontiers: HashMap<String, ParetoFrontier>,
}
/// Solver types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SolverType {
    Simplex,
    InteriorPoint,
    ActiveSet,
    BarrierMethod,
    AugmentedLagrangian,
}
/// Prediction distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionDistribution {
    histogram: Vec<(f64, usize)>,
    quantiles: HashMap<String, f64>,
    outlier_threshold: f64,
    outlier_count: usize,
}
/// Cost alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAlertConfig {
    /// Enable alerts
    pub enabled: bool,
    /// Alert rules
    pub alert_rules: Vec<CostAlertRule>,
    /// Notification channels
    pub notification_channels: Vec<NotificationChannel>,
    /// Alert aggregation
    pub aggregation_config: AlertAggregationConfig,
}
/// Reliability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityMetrics {
    uptime_percentage: f64,
    mean_time_between_failures: Duration,
    mean_time_to_recovery: Duration,
    error_rate_trend: f64,
    consistency_score: f64,
}
/// Acknowledgment status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AcknowledgmentStatus {
    Unacknowledged,
    Acknowledged { by: String, at: SystemTime },
    Resolved { by: String, at: SystemTime },
}
/// Notification frequencies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NotificationFrequency {
    Immediate,
    Throttled(Duration),
    Daily,
    Weekly,
    Custom(Duration),
}
/// Solution selection strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SolutionSelectionStrategy {
    /// Select by cost
    ByCost,
    /// Select by time
    ByTime,
    /// Select by quality
    ByQuality,
    /// Weighted selection
    Weighted(HashMap<OptimizationObjective, f64>),
    /// User preference
    UserPreference,
}
/// Optimization status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unbounded,
    TimeLimit,
    IterationLimit,
    Error(String),
}
/// Alert history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertHistoryEntry {
    alert_id: String,
    rule_name: String,
    triggered_at: SystemTime,
    resolved_at: Option<SystemTime>,
    duration: Option<Duration>,
    severity: AlertSeverity,
    notification_count: usize,
}
/// Types of volume discounts
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiscountType {
    /// Percentage discount
    Percentage,
    /// Fixed amount discount
    FixedAmount,
    /// Tiered pricing
    TieredPricing,
    /// Custom discount formula
    Custom(String),
}
/// Dashboard widgets
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DashboardWidget {
    CostGauge,
    BudgetProgress,
    ProviderComparison,
    CostTrends,
    TopCostConsumers,
    OptimizationSavings,
    PredictionAccuracy,
}
/// Active budget alert
#[derive(Debug, Clone)]
pub struct ActiveBudgetAlert {
    rule: CostAlertRule,
    triggered_at: SystemTime,
    last_notification: SystemTime,
    trigger_count: usize,
}
/// Cached provider comparison
#[derive(Debug, Clone)]
pub struct CachedComparison {
    result: ProviderComparisonResult,
    created_at: SystemTime,
    cache_key: String,
}
/// Report formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportFormat {
    PDF,
    HTML,
    JSON,
    CSV,
    Excel,
}
/// Availability status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AvailabilityStatus {
    Available,
    Busy,
    Maintenance,
    Unavailable,
}
/// Provider comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderComparisonResult {
    /// Comparison scores per provider
    pub provider_scores: HashMap<HardwareBackend, f64>,
    /// Detailed metrics per provider
    pub detailed_metrics: HashMap<HardwareBackend, ProviderMetrics>,
    /// Recommended provider
    pub recommended_provider: HardwareBackend,
    /// Comparison timestamp
    pub timestamp: SystemTime,
}
/// Resource constraints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceConstraint {
    /// Budget constraint
    Budget(f64),
    /// Time constraint
    Time(Duration),
    /// Quality constraint
    Quality(f64),
    /// Provider constraint
    Provider(Vec<HardwareBackend>),
    /// Custom constraint
    Custom { name: String, value: f64 },
}
/// Chart types for visualizations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChartType {
    LineChart,
    BarChart,
    PieChart,
    Histogram,
    ScatterPlot,
    Heatmap,
    TreeMap,
    Gauge,
}
/// Active alert
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    alert_id: String,
    rule: CostAlertRule,
    triggered_at: SystemTime,
    last_notification: SystemTime,
    escalation_level: usize,
    notification_count: usize,
    acknowledgment_status: AcknowledgmentStatus,
}
/// Feature selection methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeatureSelectionMethod {
    /// Use all features
    All,
    /// Correlation-based selection
    Correlation(f64),
    /// Mutual information
    MutualInformation,
    /// Recursive feature elimination
    RecursiveElimination,
    /// L1 regularization (Lasso)
    L1Regularization,
    /// Custom selection
    Custom(Vec<String>),
}
/// Incident severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IncidentSeverity {
    Low,
    Medium,
    High,
    Critical,
}
/// Main cost optimization engine
pub struct CostOptimizationEngine {
    pub config: CostOptimizationConfig,
    pub cost_estimator: Arc<RwLock<CostEstimator>>,
    pub budget_manager: Arc<RwLock<BudgetManager>>,
    pub provider_comparator: Arc<RwLock<ProviderComparator>>,
    pub predictive_modeler: Arc<RwLock<PredictiveModeler>>,
    pub resource_optimizer: Arc<RwLock<ResourceOptimizer>>,
    pub cost_monitor: Arc<RwLock<CostMonitor>>,
    pub alert_manager: Arc<RwLock<AlertManager>>,
    pub optimization_cache: Arc<RwLock<HashMap<String, CachedOptimization>>>,
}
/// Convergence criteria for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceCriteria {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Tolerance
    pub tolerance: f64,
    /// Patience (iterations without improvement)
    pub patience: usize,
}
/// Provider-based features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProviderFeature {
    ProviderType,
    QueueLength,
    SystemLoad,
    ErrorRates,
    Calibration,
    Availability,
    PastPerformance,
}
/// Normalization methods for comparison
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Min-max normalization
    MinMax,
    /// Z-score normalization
    ZScore,
    /// Robust normalization
    Robust,
    /// Percentile-based normalization
    Percentile(f64),
    /// Custom normalization
    Custom(String),
}
/// Cached cost estimate
#[derive(Debug, Clone)]
pub struct CachedEstimate {
    estimate: CostEstimate,
    created_at: SystemTime,
    access_count: usize,
}
/// Types of reliability incidents
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IncidentType {
    Outage,
    Degradation,
    Maintenance,
    ErrorSpike,
    QueueOverload,
}
/// Constraint types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintType {
    Equality,
    Inequality,
    Box,
}
/// Accuracy metrics for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    mae: f64,
    mse: f64,
    rmse: f64,
    mape: f64,
    r2_score: f64,
}
/// Anomaly detector
pub struct AnomalyDetector {
    pub detection_methods: Vec<AnomalyDetectionMethod>,
    pub anomaly_threshold: f64,
    pub detected_anomalies: VecDeque<Anomaly>,
}
/// Notification handler information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationHandlerInfo {
    pub name: String,
    pub description: String,
    pub supported_formats: Vec<String>,
    pub delivery_guarantee: DeliveryGuarantee,
}
/// Monthly budget status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlyBudgetStatus {
    pub monthly_budget: f64,
    pub monthly_used: f64,
    pub monthly_remaining: f64,
    pub projected_monthly_usage: f64,
}
/// Types of cost models
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CostModelType {
    /// Linear cost model
    Linear,
    /// Step-based pricing
    StepBased,
    /// Exponential pricing
    Exponential,
    /// Custom formula
    Custom(String),
    /// Machine learning model
    MachineLearning,
    /// Hybrid model combining multiple approaches
    Hybrid(Vec<CostModelType>),
}
/// Trend model
#[derive(Debug, Clone)]
pub struct TrendModel {
    model_type: TrendModelType,
    parameters: HashMap<String, f64>,
    goodness_of_fit: f64,
    last_updated: SystemTime,
}
/// Cost alert rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAlertRule {
    /// Rule name
    pub name: String,
    /// Alert condition
    pub condition: AlertCondition,
    /// Severity level
    pub severity: AlertSeverity,
    /// Notification frequency
    pub frequency: NotificationFrequency,
    /// Enabled flag
    pub enabled: bool,
}
/// Parameter types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterType {
    Real,
    Integer,
    Boolean,
    Categorical(Vec<String>),
}
/// Voting strategies for ensembles
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VotingStrategy {
    Majority,
    Weighted,
    Confidence,
    Dynamic,
}
/// Problem types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProblemType {
    LinearProgramming,
    QuadraticProgramming,
    NonlinearProgramming,
    IntegerProgramming,
    ConstraintSatisfaction,
    MultiObjective,
}
/// Cost estimation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimationConfig {
    /// Estimation models per provider
    pub provider_models: HashMap<HardwareBackend, CostModel>,
    /// Include queue time in estimates
    pub include_queue_time: bool,
    /// Include setup/teardown costs
    pub include_overhead_costs: bool,
    /// Estimation accuracy target
    pub accuracy_target: f64,
    /// Update frequency for cost models
    pub model_update_frequency: Duration,
    /// Enable machine learning-based estimation
    pub enable_ml_estimation: bool,
    /// Historical data retention period
    pub data_retention_period: Duration,
}
/// Cost breakdown by components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    /// Base execution cost
    pub execution_cost: f64,
    /// Queue time cost
    pub queue_cost: f64,
    /// Setup/teardown cost
    pub setup_cost: f64,
    /// Data transfer cost
    pub data_transfer_cost: f64,
    /// Storage cost
    pub storage_cost: f64,
    /// Additional fees
    pub additional_fees: HashMap<String, f64>,
}
/// Aggregation methods for metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregationMethod {
    Average,
    Sum,
    Count,
    Min,
    Max,
    Median,
    Percentile(f64),
}
/// Alert aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertAggregationConfig {
    /// Enable aggregation
    pub enabled: bool,
    /// Aggregation window
    pub window: Duration,
    /// Maximum alerts per window
    pub max_alerts_per_window: usize,
    /// Aggregation strategy
    pub strategy: AggregationStrategy,
}
/// Derived feature definition
pub struct DerivedFeature {
    pub name: String,
    pub computation: Box<dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync>,
    pub dependencies: Vec<String>,
}
/// Delivery guarantees
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeliveryGuarantee {
    BestEffort,
    AtLeastOnce,
    ExactlyOnce,
}
/// Circuit-based features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CircuitFeature {
    QubitCount,
    GateCount,
    CircuitDepth,
    GateComplexity,
    ConnectivityRequirements,
    EstimatedFidelity,
    CircuitType,
}
/// Reliability incident
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityIncident {
    provider: HardwareBackend,
    incident_type: IncidentType,
    start_time: SystemTime,
    end_time: Option<SystemTime>,
    impact_severity: IncidentSeverity,
    description: String,
}
/// Types of cost reports
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportType {
    CostSummary,
    BudgetAnalysis,
    ProviderComparison,
    TrendAnalysis,
    OptimizationReport,
    ForecastReport,
    AnomalyReport,
}
/// Cost optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizationConfig {
    /// Budget management settings
    pub budget_config: BudgetConfig,
    /// Cost estimation settings
    pub estimation_config: CostEstimationConfig,
    /// Optimization strategy
    pub optimization_strategy: CostOptimizationStrategy,
    /// Provider comparison settings
    pub provider_comparison: ProviderComparisonConfig,
    /// Predictive modeling settings
    pub predictive_modeling: PredictiveModelingConfig,
    /// Resource allocation optimization
    pub resource_optimization: ResourceOptimizationConfig,
    /// Real-time monitoring settings
    pub monitoring_config: CostMonitoringConfig,
    /// Alert and notification settings
    pub alert_config: CostAlertConfig,
}
/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}
/// Budget tracking information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetStatus {
    /// Total budget
    pub total_budget: f64,
    /// Used budget
    pub used_budget: f64,
    /// Remaining budget
    pub remaining_budget: f64,
    /// Budget utilization percentage
    pub utilization_percentage: f64,
    /// Daily budget status
    pub daily_status: Option<DailyBudgetStatus>,
    /// Monthly budget status
    pub monthly_status: Option<MonthlyBudgetStatus>,
    /// Provider budget breakdown
    pub provider_breakdown: HashMap<HardwareBackend, f64>,
}
/// Provider comparison configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderComparisonConfig {
    /// Comparison metrics
    pub comparison_metrics: Vec<ComparisonMetric>,
    /// Normalization method
    pub normalization_method: NormalizationMethod,
    /// Weighting scheme
    pub metric_weights: HashMap<ComparisonMetric, f64>,
    /// Enable real-time comparison
    pub real_time_comparison: bool,
    /// Comparison update frequency
    pub update_frequency: Duration,
    /// Include provider reliability in comparison
    pub include_reliability: bool,
}
/// Feature statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStatistics {
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    trend: f64,
    autocorrelation: f64,
}
/// Detected anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    metric: MonitoringMetric,
    timestamp: SystemTime,
    anomaly_score: f64,
    detected_value: f64,
    expected_value: f64,
    description: String,
}
/// Metrics for provider comparison
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComparisonMetric {
    /// Total cost
    TotalCost,
    /// Cost per shot
    CostPerShot,
    /// Cost per qubit
    CostPerQubit,
    /// Queue time
    QueueTime,
    /// Execution time
    ExecutionTime,
    /// Fidelity
    Fidelity,
    /// Availability
    Availability,
    /// Reliability
    Reliability,
    /// Custom metric
    Custom(String),
}
/// Metrics for cost monitoring
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MonitoringMetric {
    TotalCost,
    CostPerProvider,
    CostPerCircuit,
    BudgetUtilization,
    CostTrends,
    CostEfficiency,
    PredictionAccuracy,
    OptimizationSavings,
}
/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Enable real-time dashboard
    pub enabled: bool,
    /// Update frequency
    pub update_frequency: Duration,
    /// Dashboard widgets
    pub widgets: Vec<DashboardWidget>,
    /// Custom visualizations
    pub custom_visualizations: Vec<CustomVisualization>,
}
/// Pareto solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoSolution {
    pub variables: Vec<f64>,
    pub objectives: Vec<f64>,
    pub dominance_count: usize,
    pub crowding_distance: f64,
}
