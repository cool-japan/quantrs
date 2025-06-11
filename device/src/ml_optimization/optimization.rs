//! Optimization Strategy Configuration Types

use std::collections::HashMap;

/// Optimization strategy configuration
#[derive(Debug, Clone)]
pub struct OptimizationStrategyConfig {
    /// Multi-objective optimization
    pub multi_objective: MultiObjectiveConfig,
    /// Constraint handling
    pub constraint_handling: ConstraintHandlingConfig,
    /// Search strategies
    pub search_strategies: Vec<SearchStrategy>,
    /// Exploration-exploitation balance
    pub exploration_exploitation: ExplorationExploitationConfig,
    /// Adaptive strategies
    pub adaptive_strategies: AdaptiveStrategyConfig,
}

/// Multi-objective optimization configuration
#[derive(Debug, Clone)]
pub struct MultiObjectiveConfig {
    /// Enable multi-objective optimization
    pub enable_multi_objective: bool,
    /// Objectives and weights
    pub objectives: HashMap<String, f64>,
    /// Pareto optimization
    pub pareto_optimization: bool,
    /// Scalarization methods
    pub scalarization_methods: Vec<ScalarizationMethod>,
}

/// Scalarization methods
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarizationMethod {
    WeightedSum,
    Chebyshev,
    AugmentedChebyshev,
    BoundaryIntersection,
    AchievementFunction,
}

/// Constraint handling configuration
#[derive(Debug, Clone)]
pub struct ConstraintHandlingConfig {
    /// Constraint types
    pub constraint_types: Vec<ConstraintType>,
    /// Penalty methods
    pub penalty_methods: Vec<PenaltyMethod>,
    /// Constraint tolerance
    pub constraint_tolerance: f64,
    /// Feasibility preservation
    pub feasibility_preservation: bool,
}

/// Constraint types
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    Equality,
    Inequality,
    Box,
    Nonlinear,
    Integer,
}

/// Penalty methods
#[derive(Debug, Clone, PartialEq)]
pub enum PenaltyMethod {
    ExteriorPenalty,
    InteriorPenalty,
    AugmentedLagrangian,
    BarrierMethod,
    FilterMethod,
}

/// Search strategies
#[derive(Debug, Clone, PartialEq)]
pub enum SearchStrategy {
    GradientBased,
    EvolutionaryAlgorithm,
    SwarmIntelligence,
    SimulatedAnnealing,
    BayesianOptimization,
    ReinforcementLearning,
    HybridMethods,
}

/// Exploration-exploitation configuration
#[derive(Debug, Clone)]
pub struct ExplorationExploitationConfig {
    /// Initial exploration rate
    pub initial_exploration_rate: f64,
    /// Exploration decay
    pub exploration_decay: f64,
    /// Minimum exploration rate
    pub min_exploration_rate: f64,
    /// Exploitation threshold
    pub exploitation_threshold: f64,
    /// Adaptive balancing
    pub adaptive_balancing: bool,
}

/// Adaptive strategy configuration
#[derive(Debug, Clone)]
pub struct AdaptiveStrategyConfig {
    /// Enable adaptive strategies
    pub enable_adaptive: bool,
    /// Strategy selection methods
    pub strategy_selection: Vec<StrategySelectionMethod>,
    /// Performance feedback
    pub performance_feedback: bool,
    /// Strategy mutation
    pub strategy_mutation: bool,
}

/// Strategy selection methods
#[derive(Debug, Clone, PartialEq)]
pub enum StrategySelectionMethod {
    PerformanceBased,
    BanditAlgorithm,
    ReinforcementLearning,
    HeuristicRules,
    MachineLearning,
}
