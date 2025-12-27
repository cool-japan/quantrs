//! Core Configuration Types for Adaptive Compilation

use std::collections::HashMap;
use std::time::Duration;

use super::hardware_adaptation::*;
use super::ml_integration::*;
use super::monitoring::*;
use super::strategies::*;

/// Configuration for adaptive compilation pipeline
#[derive(Debug, Clone, Default)]
pub struct AdaptiveCompilationConfig {
    /// Real-time optimization settings
    pub realtime_optimization: RealtimeOptimizationConfig,
    /// Adaptive strategies configuration
    pub adaptive_strategies: AdaptiveStrategiesConfig,
    /// Performance monitoring configuration
    pub performance_monitoring: PerformanceMonitoringConfig,
    /// Machine learning configuration
    pub ml_optimization: MLOptimizationConfig,
    /// Circuit analysis configuration
    pub circuit_analysis: CircuitAnalysisConfig,
    /// Hardware adaptation configuration
    pub hardware_adaptation: HardwareAdaptationConfig,
    /// Optimization objectives and weights
    pub optimization_objectives: OptimizationObjectivesConfig,
    /// Caching and learning configuration
    pub caching_learning: CachingLearningConfig,
}

/// Real-time optimization configuration
#[derive(Debug, Clone)]
pub struct RealtimeOptimizationConfig {
    /// Enable real-time optimization
    pub enable_realtime: bool,
    /// Optimization update interval
    pub update_interval: Duration,
    /// Performance threshold for triggering re-optimization
    pub performance_threshold: f64,
    /// Maximum optimization time per iteration
    pub max_optimization_time: Duration,
    /// Optimization algorithms to use
    pub optimization_algorithms: Vec<OptimizationAlgorithm>,
    /// Parallel optimization settings
    pub parallel_optimization: ParallelOptimizationConfig,
    /// Adaptive algorithm selection
    pub adaptive_algorithm_selection: bool,
}

/// Optimization objectives configuration
#[derive(Debug, Clone)]
pub struct OptimizationObjectivesConfig {
    /// Primary objectives and weights
    pub primary_objectives: HashMap<String, f64>,
    /// Secondary objectives and weights
    pub secondary_objectives: HashMap<String, f64>,
    /// Constraint definitions
    pub constraints: Vec<OptimizationConstraint>,
    /// Multi-objective optimization settings
    pub multi_objective: MultiObjectiveConfig,
    /// Dynamic objective weighting
    pub dynamic_weighting: DynamicWeightingConfig,
}

/// Caching and learning configuration
#[derive(Debug, Clone)]
pub struct CachingLearningConfig {
    /// Enable intelligent caching
    pub enable_caching: bool,
    /// Cache size limits
    pub cache_size_limits: CacheSizeLimits,
    /// Cache eviction policies
    pub eviction_policies: EvictionPolicies,
    /// Learning from cached results
    pub cache_learning: CacheLearningConfig,
    /// Distributed caching settings
    pub distributed_caching: DistributedCachingConfig,
}

/// Optimization constraint definition
#[derive(Debug, Clone)]
pub struct OptimizationConstraint {
    /// Constraint name
    pub name: String,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Constraint value or bound
    pub value: f64,
    /// Priority level
    pub priority: f64,
}

/// Types of optimization constraints
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintType {
    MaxExecutionTime,
    MinFidelity,
    MaxResourceUsage,
    MaxCircuitDepth,
    MaxGateCount,
    EnergyBudget,
    Custom(String),
}

/// Multi-objective optimization configuration
#[derive(Debug, Clone)]
pub struct MultiObjectiveConfig {
    /// Enable multi-objective optimization
    pub enable_multi_objective: bool,
    /// Pareto frontier exploration
    pub pareto_exploration: ParetoExplorationConfig,
    /// Objective trade-off strategies
    pub trade_off_strategies: Vec<TradeOffStrategy>,
    /// Solution ranking method
    pub ranking_method: RankingMethod,
}

/// Pareto frontier exploration configuration
#[derive(Debug, Clone)]
pub struct ParetoExplorationConfig {
    /// Number of solutions to maintain
    pub solution_count: usize,
    /// Diversity preservation
    pub diversity_preservation: bool,
    /// Convergence criteria
    pub convergence_criteria: ConvergenceCriteria,
}

/// Trade-off strategies between objectives
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TradeOffStrategy {
    WeightedSum,
    EpsilonConstraint,
    Lexicographic,
    GoalProgramming,
    NSGA2,
    MOEA,
}

/// Solution ranking methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RankingMethod {
    Dominance,
    TOPSIS,
    WeightedSum,
    Utility,
    Custom(String),
}

/// Convergence criteria for optimization
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Objective improvement threshold
    pub improvement_threshold: f64,
    /// Stagnation detection
    pub stagnation_tolerance: usize,
    /// Time limit
    pub time_limit: Duration,
}

/// Dynamic objective weighting configuration
#[derive(Debug, Clone)]
pub struct DynamicWeightingConfig {
    /// Enable dynamic weighting
    pub enable_dynamic: bool,
    /// Weighting adaptation strategy
    pub adaptation_strategy: WeightingStrategy,
    /// Update frequency
    pub update_frequency: Duration,
    /// Historical influence
    pub historical_influence: f64,
}

/// Weighting adaptation strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightingStrategy {
    PerformanceBased,
    AdaptiveBayesian,
    ReinforcementLearning,
    GradientBased,
    Evolutionary,
}

/// Cache size limits configuration
#[derive(Debug, Clone)]
pub struct CacheSizeLimits {
    /// Circuit cache size (number of circuits)
    pub circuit_cache_size: usize,
    /// Optimization result cache size
    pub optimization_cache_size: usize,
    /// Performance data cache size
    pub performance_cache_size: usize,
    /// Memory limit for caching (in MB)
    pub memory_limit_mb: usize,
}

/// Cache eviction policies
#[derive(Debug, Clone)]
pub struct EvictionPolicies {
    /// Policy for circuit cache
    pub circuit_cache_policy: EvictionPolicy,
    /// Policy for optimization cache
    pub optimization_cache_policy: EvictionPolicy,
    /// Policy for performance cache
    pub performance_cache_policy: EvictionPolicy,
}

/// Cache eviction policy types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    PerformanceBased,
    AgeBased,
    SizeBased,
    Hybrid,
}

/// Cache learning configuration
#[derive(Debug, Clone)]
pub struct CacheLearningConfig {
    /// Enable learning from cache
    pub enable_cache_learning: bool,
    /// Pattern recognition in cached results
    pub pattern_recognition: bool,
    /// Predictive pre-caching
    pub predictive_precaching: bool,
    /// Cache hit prediction
    pub cache_hit_prediction: bool,
}

/// Distributed caching configuration
#[derive(Debug, Clone)]
pub struct DistributedCachingConfig {
    /// Enable distributed caching
    pub enable_distributed: bool,
    /// Cache consistency model
    pub consistency_model: ConsistencyModel,
    /// Cache replication factor
    pub replication_factor: usize,
    /// Cache synchronization strategy
    pub synchronization_strategy: SynchronizationStrategy,
}

/// Cache consistency models
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsistencyModel {
    StrongConsistency,
    EventualConsistency,
    WeakConsistency,
    SessionConsistency,
}

/// Cache synchronization strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynchronizationStrategy {
    Immediate,
    Batch,
    LazyPropagation,
    ConflictFree,
}

impl Default for RealtimeOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_realtime: true,
            update_interval: Duration::from_millis(100),
            performance_threshold: 0.95,
            max_optimization_time: Duration::from_secs(30),
            optimization_algorithms: vec![
                OptimizationAlgorithm::GradientDescent,
                OptimizationAlgorithm::BayesianOptimization,
            ],
            parallel_optimization: ParallelOptimizationConfig::default(),
            adaptive_algorithm_selection: true,
        }
    }
}

impl Default for OptimizationObjectivesConfig {
    fn default() -> Self {
        let mut primary_objectives = HashMap::new();
        primary_objectives.insert("fidelity".to_string(), 0.6);
        primary_objectives.insert("execution_time".to_string(), 0.4);

        Self {
            primary_objectives,
            secondary_objectives: HashMap::new(),
            constraints: vec![],
            multi_objective: MultiObjectiveConfig::default(),
            dynamic_weighting: DynamicWeightingConfig::default(),
        }
    }
}

impl Default for CachingLearningConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_size_limits: CacheSizeLimits::default(),
            eviction_policies: EvictionPolicies::default(),
            cache_learning: CacheLearningConfig::default(),
            distributed_caching: DistributedCachingConfig::default(),
        }
    }
}

impl Default for MultiObjectiveConfig {
    fn default() -> Self {
        Self {
            enable_multi_objective: true,
            pareto_exploration: ParetoExplorationConfig::default(),
            trade_off_strategies: vec![TradeOffStrategy::NSGA2],
            ranking_method: RankingMethod::Dominance,
        }
    }
}

impl Default for ParetoExplorationConfig {
    fn default() -> Self {
        Self {
            solution_count: 50,
            diversity_preservation: true,
            convergence_criteria: ConvergenceCriteria::default(),
        }
    }
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            improvement_threshold: 1e-6,
            stagnation_tolerance: 50,
            time_limit: Duration::from_secs(300),
        }
    }
}

impl Default for DynamicWeightingConfig {
    fn default() -> Self {
        Self {
            enable_dynamic: true,
            adaptation_strategy: WeightingStrategy::PerformanceBased,
            update_frequency: Duration::from_secs(60),
            historical_influence: 0.3,
        }
    }
}

impl Default for CacheSizeLimits {
    fn default() -> Self {
        Self {
            circuit_cache_size: 1000,
            optimization_cache_size: 500,
            performance_cache_size: 2000,
            memory_limit_mb: 1024,
        }
    }
}

impl Default for EvictionPolicies {
    fn default() -> Self {
        Self {
            circuit_cache_policy: EvictionPolicy::PerformanceBased,
            optimization_cache_policy: EvictionPolicy::LRU,
            performance_cache_policy: EvictionPolicy::LFU,
        }
    }
}

impl Default for CacheLearningConfig {
    fn default() -> Self {
        Self {
            enable_cache_learning: true,
            pattern_recognition: true,
            predictive_precaching: true,
            cache_hit_prediction: true,
        }
    }
}

impl Default for DistributedCachingConfig {
    fn default() -> Self {
        Self {
            enable_distributed: false,
            consistency_model: ConsistencyModel::EventualConsistency,
            replication_factor: 3,
            synchronization_strategy: SynchronizationStrategy::Batch,
        }
    }
}
