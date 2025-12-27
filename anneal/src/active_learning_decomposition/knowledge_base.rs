//! Knowledge base components for decomposition

use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::{
    ActionType, ConditionType, DecompositionStrategy, LogicalOperator, PerformanceRecord,
    RequirementType, RequirementValue, ResourceConstraints, SideEffectType, TrendDirection,
};

/// Decomposition knowledge base
#[derive(Debug, Clone)]
pub struct DecompositionKnowledgeBase {
    /// Strategy database
    pub strategy_database: StrategyDatabase,
    /// Pattern library
    pub pattern_library: PatternLibrary,
    /// Performance repository
    pub performance_repository: PerformanceRepository,
    /// Rule engine
    pub rule_engine: RuleEngine,
}

impl DecompositionKnowledgeBase {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            strategy_database: StrategyDatabase::new(),
            pattern_library: PatternLibrary::new(),
            performance_repository: PerformanceRepository::new(),
            rule_engine: RuleEngine::new(),
        })
    }
}

/// Strategy database
#[derive(Debug, Clone)]
pub struct StrategyDatabase {
    /// Available strategies
    pub strategies: Vec<DecompositionStrategy>,
    /// Strategy relationships
    pub strategy_relationships: HashMap<String, Vec<String>>,
    /// Strategy success rates
    pub success_rates: HashMap<DecompositionStrategy, f64>,
}

impl StrategyDatabase {
    #[must_use]
    pub fn new() -> Self {
        Self {
            strategies: vec![
                DecompositionStrategy::GraphPartitioning,
                DecompositionStrategy::CommunityDetection,
                DecompositionStrategy::SpectralClustering,
                DecompositionStrategy::Hierarchical,
                DecompositionStrategy::NoDecomposition,
            ],
            strategy_relationships: HashMap::new(),
            success_rates: HashMap::new(),
        }
    }
}

/// Pattern library
#[derive(Debug, Clone)]
pub struct PatternLibrary {
    /// Known patterns
    pub patterns: Vec<KnownPattern>,
    /// Pattern index
    pub pattern_index: HashMap<String, usize>,
    /// Pattern similarity matrix
    pub similarity_matrix: Array2<f64>,
}

impl PatternLibrary {
    #[must_use]
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            pattern_index: HashMap::new(),
            similarity_matrix: Array2::zeros((0, 0)),
        }
    }
}

/// Known pattern
#[derive(Debug, Clone)]
pub struct KnownPattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Pattern description
    pub description: String,
    /// Pattern features
    pub features: scirs2_core::ndarray::Array1<f64>,
    /// Optimal strategies for this pattern
    pub optimal_strategies: Vec<DecompositionStrategy>,
    /// Pattern frequency
    pub frequency: f64,
}

/// Performance repository
#[derive(Debug, Clone)]
pub struct PerformanceRepository {
    /// Historical performance data
    pub historical_data: Vec<HistoricalPerformance>,
    /// Performance trends
    pub performance_trends: HashMap<String, PerformanceTrend>,
    /// Benchmark results
    pub benchmark_results: Vec<BenchmarkResult>,
}

impl PerformanceRepository {
    #[must_use]
    pub fn new() -> Self {
        Self {
            historical_data: Vec::new(),
            performance_trends: HashMap::new(),
            benchmark_results: Vec::new(),
        }
    }
}

/// Historical performance data
#[derive(Debug, Clone)]
pub struct HistoricalPerformance {
    /// Problem characteristics
    pub problem_characteristics: ProblemCharacteristics,
    /// Strategy applied
    pub strategy_applied: DecompositionStrategy,
    /// Performance achieved
    pub performance_achieved: PerformanceRecord,
    /// Context information
    pub context: PerformanceContext,
}

/// Problem characteristics
#[derive(Debug, Clone)]
pub struct ProblemCharacteristics {
    /// Problem size
    pub problem_size: usize,
    /// Problem type
    pub problem_type: String,
    /// Structural features
    pub structural_features: scirs2_core::ndarray::Array1<f64>,
    /// Complexity indicators
    pub complexity_indicators: HashMap<String, f64>,
}

/// Performance context
#[derive(Debug, Clone)]
pub struct PerformanceContext {
    /// Hardware configuration
    pub hardware_config: String,
    /// Software configuration
    pub software_config: String,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Environmental factors
    pub environmental_factors: HashMap<String, f64>,
}

/// Performance trend
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Trend strength
    pub trend_strength: f64,
    /// Trend data points
    pub data_points: Vec<(f64, f64)>, // (time, performance)
    /// Trend prediction
    pub prediction: Option<TrendPrediction>,
}

/// Trend prediction
#[derive(Debug, Clone)]
pub struct TrendPrediction {
    /// Predicted value
    pub predicted_value: f64,
    /// Prediction confidence
    pub confidence: f64,
    /// Prediction horizon
    pub horizon: Duration,
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub benchmark_name: String,
    /// Problem set
    pub problem_set: Vec<String>,
    /// Strategy results
    pub strategy_results: HashMap<DecompositionStrategy, f64>,
    /// Best performing strategy
    pub best_strategy: DecompositionStrategy,
}

/// Rule engine for decomposition decisions
#[derive(Debug, Clone)]
pub struct RuleEngine {
    /// Rule set
    pub rules: Vec<DecompositionRule>,
    /// Rule priorities
    pub rule_priorities: HashMap<String, f64>,
    /// Rule application history
    pub application_history: Vec<RuleApplication>,
}

impl RuleEngine {
    #[must_use]
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            rule_priorities: HashMap::new(),
            application_history: Vec::new(),
        }
    }
}

/// Decomposition rule
#[derive(Debug, Clone)]
pub struct DecompositionRule {
    /// Rule identifier
    pub rule_id: String,
    /// Rule condition
    pub condition: RuleCondition,
    /// Rule action
    pub action: RuleAction,
    /// Rule confidence
    pub confidence: f64,
    /// Rule applicability
    pub applicability: RuleApplicability,
}

/// Rule condition
#[derive(Debug, Clone)]
pub struct RuleCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Condition parameters
    pub parameters: HashMap<String, f64>,
    /// Logical operator
    pub logical_operator: LogicalOperator,
}

/// Rule action
#[derive(Debug, Clone)]
pub struct RuleAction {
    /// Action type
    pub action_type: ActionType,
    /// Action parameters
    pub parameters: HashMap<String, f64>,
    /// Expected outcome
    pub expected_outcome: ExpectedOutcome,
}

/// Expected outcome
#[derive(Debug, Clone)]
pub struct ExpectedOutcome {
    /// Performance improvement
    pub performance_improvement: f64,
    /// Confidence in outcome
    pub outcome_confidence: f64,
    /// Side effects
    pub side_effects: Vec<SideEffect>,
}

/// Side effect
#[derive(Debug, Clone)]
pub struct SideEffect {
    /// Effect type
    pub effect_type: SideEffectType,
    /// Effect magnitude
    pub magnitude: f64,
    /// Effect probability
    pub probability: f64,
}

/// Rule applicability
#[derive(Debug, Clone)]
pub struct RuleApplicability {
    /// Problem types where rule applies
    pub applicable_problem_types: Vec<String>,
    /// Size range where rule applies
    pub applicable_size_range: (usize, usize),
    /// Context requirements
    pub context_requirements: Vec<ContextRequirement>,
}

/// Context requirement
#[derive(Debug, Clone)]
pub struct ContextRequirement {
    /// Requirement type
    pub requirement_type: RequirementType,
    /// Required value or range
    pub required_value: RequirementValue,
}

/// Rule application record
#[derive(Debug, Clone)]
pub struct RuleApplication {
    /// Application timestamp
    pub timestamp: Instant,
    /// Rule applied
    pub rule_id: String,
    /// Problem context
    pub problem_context: ProblemCharacteristics,
    /// Application result
    pub result: ApplicationResult,
}

/// Application result
#[derive(Debug, Clone)]
pub struct ApplicationResult {
    /// Success status
    pub success: bool,
    /// Performance impact
    pub performance_impact: f64,
    /// User satisfaction
    pub user_satisfaction: Option<f64>,
    /// Lessons learned
    pub lessons_learned: Vec<String>,
}
