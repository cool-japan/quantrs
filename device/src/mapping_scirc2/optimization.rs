//! Advanced optimization configurations and algorithms

use super::*;

/// Advanced optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedOptimizationConfig {
    /// Enable advanced optimization features
    pub enable_advanced: bool,
    /// Multi-objective optimization configuration
    pub multi_objective: MultiObjectiveConfig,
    /// Constraint handling configuration
    pub constraint_handling: ConstraintHandlingConfig,
    /// Search strategy configuration
    pub search_strategy: SearchStrategyConfig,
    /// Parallel optimization configuration
    pub parallel_optimization: ParallelOptimizationConfig,
}

/// Multi-objective optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveConfig {
    /// Enable multi-objective optimization
    pub enable_multi_objective: bool,
    /// List of objectives to optimize
    pub objectives: Vec<OptimizationObjective>,
    /// Pareto optimization configuration
    pub pareto_config: ParetoConfig,
}

/// Pareto optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoConfig {
    /// Population size for genetic algorithms
    pub population_size: usize,
    /// Number of generations
    pub generations: usize,
    /// Crossover rate
    pub crossover_rate: f64,
    /// Mutation rate
    pub mutation_rate: f64,
    /// Selection method
    pub selection_method: SelectionMethod,
    /// Scalarization method for multi-objective
    pub scalarization: ScalarizationMethod,
}

/// Constraint handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintHandlingConfig {
    /// Enable constraint handling
    pub enable_constraints: bool,
    /// Types of constraints to handle
    pub constraint_types: Vec<ConstraintType>,
    /// Penalty method for constraint violations
    pub penalty_method: PenaltyMethod,
    /// Tolerance for constraint satisfaction
    pub tolerance: f64,
}

/// Search strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchStrategyConfig {
    /// Primary search strategy
    pub strategy: SearchStrategy,
    /// Hybrid search configuration
    pub hybrid_config: HybridSearchConfig,
    /// Search budget configuration
    pub budget: SearchBudgetConfig,
}

/// Hybrid search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchConfig {
    /// List of strategies to combine
    pub strategies: Vec<SearchStrategy>,
    /// Criteria for switching between strategies
    pub switching_criteria: SwitchingCriteria,
}

/// Switching criteria for hybrid search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchingCriteria {
    /// Performance improvement threshold
    pub performance_threshold: f64,
    /// Number of iterations without improvement
    pub stagnation_limit: usize,
    /// Time limit for each strategy
    pub time_limit: Duration,
}

/// Search budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchBudgetConfig {
    /// Maximum number of evaluations
    pub max_evaluations: usize,
    /// Maximum time budget
    pub max_time: Duration,
    /// Target quality threshold
    pub target_quality: f64,
}

/// Parallel optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelOptimizationConfig {
    /// Enable parallel optimization
    pub enable_parallel: bool,
    /// Number of worker threads
    pub num_workers: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Synchronization method
    pub synchronization: SynchronizationMethod,
}

/// Optimization metrics and performance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationMetrics {
    /// Total optimization time
    pub optimization_time: Duration,
    /// Number of iterations performed
    pub iterations: usize,
    /// Convergence achieved
    pub converged: bool,
    /// Final objective value
    pub final_objective: f64,
    /// Best objective value achieved
    pub best_objective: f64,
    /// Improvement over initial solution
    pub improvement_ratio: f64,
    /// Constraint violations
    pub constraint_violations: f64,
    /// Algorithm-specific metrics
    pub algorithm_metrics: HashMap<String, f64>,
    /// Resource utilization during optimization
    pub resource_usage: ResourceUsageMetrics,
}

/// Resource usage metrics during optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageMetrics {
    /// Peak memory usage
    pub peak_memory: usize,
    /// Average CPU utilization
    pub average_cpu: f64,
    /// Total energy consumption (if measurable)
    pub energy_consumption: Option<f64>,
    /// Network communication overhead
    pub network_overhead: Option<f64>,
}

/// Result analysis for optimization outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultAnalysis {
    /// Analysis type
    pub analysis_type: ResultAnalysisType,
    /// Statistical summary
    pub statistical_summary: StatisticalSummary,
    /// Trend analysis over iterations
    pub trend_analysis: TrendAnalysis,
    /// Comparative analysis with baselines
    pub comparative_analysis: ComparativeAnalysis,
    /// Sensitivity analysis results
    pub sensitivity_analysis: SensitivityAnalysis,
}

/// Statistical summary of results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    /// Mean objective value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Median value
    pub median: f64,
    /// 95% confidence interval
    pub confidence_interval_95: (f64, f64),
}

/// Trend analysis over optimization iterations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Overall trend direction
    pub trend_direction: TrendDirection,
    /// Rate of improvement
    pub improvement_rate: f64,
    /// Convergence characteristics
    pub convergence_pattern: String,
    /// Plateaus and breakthroughs
    pub plateau_analysis: Vec<PlateauInfo>,
}

/// Information about plateaus in optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlateauInfo {
    /// Start iteration of plateau
    pub start_iteration: usize,
    /// End iteration of plateau
    pub end_iteration: usize,
    /// Duration of plateau
    pub duration: usize,
    /// Objective value during plateau
    pub plateau_value: f64,
}

/// Comparative analysis with baseline methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeAnalysis {
    /// Baseline method comparisons
    pub baseline_comparisons: Vec<BaselineComparison>,
    /// Performance ranking
    pub performance_ranking: usize,
    /// Statistical significance tests
    pub significance_tests: HashMap<String, f64>,
}

/// Baseline comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    /// Baseline method name
    pub baseline_name: String,
    /// Performance improvement factor
    pub improvement_factor: f64,
    /// Statistical significance
    pub p_value: f64,
    /// Effect size
    pub effect_size: f64,
    /// Confidence in comparison
    pub confidence: f64,
}

/// Sensitivity analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis {
    /// Parameter sensitivity scores
    pub parameter_sensitivity: HashMap<String, f64>,
    /// Robust parameter ranges
    pub robust_ranges: HashMap<String, (f64, f64)>,
    /// Critical parameters
    pub critical_parameters: Vec<String>,
    /// Interaction effects
    pub interaction_effects: HashMap<String, f64>,
}