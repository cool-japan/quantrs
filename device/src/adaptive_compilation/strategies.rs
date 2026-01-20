//! Optimization Strategies and Adaptive Algorithm Configuration

use std::time::Duration;

/// Optimization algorithms available
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationAlgorithm {
    GradientDescent,
    LBFGS,
    DifferentialEvolution,
    GeneticAlgorithm,
    ParticleSwarm,
    SimulatedAnnealing,
    BayesianOptimization,
    ReinforcementLearning,
    QuantumApproximateOptimization,
    HybridClassicalQuantum,
}

/// Adaptive strategies configuration
#[derive(Debug, Clone)]
pub struct AdaptiveStrategiesConfig {
    /// Enable adaptive circuit optimization
    pub enable_adaptive_circuits: bool,
    /// Enable adaptive resource allocation
    pub enable_adaptive_resources: bool,
    /// Enable adaptive error mitigation
    pub enable_adaptive_error_mitigation: bool,
    /// Enable adaptive scheduling
    pub enable_adaptive_scheduling: bool,
    /// Adaptation triggers
    pub adaptation_triggers: Vec<AdaptationTrigger>,
    /// Learning rate for adaptation
    pub adaptation_learning_rate: f64,
    /// Exploration vs exploitation balance
    pub exploration_exploitation_balance: f64,
}

/// Triggers for adaptive optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptationTrigger {
    PerformanceDegradation,
    TimeInterval,
    ErrorRateIncrease,
    ResourceConstraintViolation,
    UserRequest,
    AutomaticDetection,
}

/// Parallel optimization configuration
#[derive(Debug, Clone)]
pub struct ParallelOptimizationConfig {
    pub enable_parallel: bool,
    pub max_parallel_jobs: usize,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub resource_allocation: ResourceAllocationStrategy,
}

/// Load balancing strategies for parallel optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    PerformanceBased,
    PredictiveBased,
}

/// Resource allocation strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceAllocationStrategy {
    Static,
    Dynamic,
    PredictiveBased,
    LoadAware,
    CapacityBased,
}

/// Circuit analysis configuration
#[derive(Debug, Clone)]
pub struct CircuitAnalysisConfig {
    /// Enable comprehensive circuit analysis
    pub enable_analysis: bool,
    /// Analysis depth level
    pub analysis_depth: AnalysisDepth,
    /// Circuit complexity analysis
    pub complexity_analysis: ComplexityAnalysisConfig,
    /// Circuit structure analysis
    pub structure_analysis: StructureAnalysisConfig,
    /// Circuit optimization potential analysis
    pub optimization_potential: OptimizationPotentialConfig,
}

/// Analysis depth levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisDepth {
    Basic,
    Intermediate,
    Comprehensive,
    ExhaustiveSearch,
}

/// Circuit complexity analysis configuration
#[derive(Debug, Clone)]
pub struct ComplexityAnalysisConfig {
    /// Complexity metrics to compute
    pub complexity_metrics: Vec<ComplexityMetric>,
    /// Enable parallel complexity analysis
    pub enable_parallel_analysis: bool,
    /// Analysis timeout
    pub analysis_timeout: Duration,
    /// Resource limits for analysis
    pub resource_limits: AnalysisResourceLimits,
}

/// Complexity metrics for circuit analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComplexityMetric {
    CircuitDepth,
    GateCount,
    TwoQubitGateCount,
    ParameterCount,
    ConnectivityComplexity,
    EntanglementComplexity,
    ComputationalComplexity,
    SimulationComplexity,
}

/// Circuit structure analysis configuration
#[derive(Debug, Clone)]
pub struct StructureAnalysisConfig {
    /// Pattern recognition
    pub pattern_recognition: bool,
    /// Symmetry detection
    pub symmetry_detection: bool,
    /// Subcircuit identification
    pub subcircuit_identification: bool,
    /// Critical path analysis
    pub critical_path_analysis: bool,
}

/// Optimization potential analysis configuration
#[derive(Debug, Clone)]
pub struct OptimizationPotentialConfig {
    /// Potential metrics to evaluate
    pub potential_metrics: Vec<PotentialMetric>,
    /// Optimization opportunity detection
    pub opportunity_detection: bool,
    /// Bottleneck identification
    pub bottleneck_identification: bool,
    /// Resource utilization analysis
    pub resource_utilization_analysis: bool,
}

/// Metrics for evaluating optimization potential
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PotentialMetric {
    GateReduction,
    DepthReduction,
    ParallelizationOpportunity,
    ResourceOptimization,
    ErrorMitigationPotential,
    ExecutionTimeImprovement,
}

/// Resource limits for circuit analysis
#[derive(Debug, Clone)]
pub struct AnalysisResourceLimits {
    /// Maximum memory usage (MB)
    pub max_memory_mb: usize,
    /// Maximum CPU time
    pub max_cpu_time: Duration,
    /// Maximum parallel threads
    pub max_threads: usize,
}

impl Default for AdaptiveStrategiesConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_circuits: true,
            enable_adaptive_resources: true,
            enable_adaptive_error_mitigation: true,
            enable_adaptive_scheduling: true,
            adaptation_triggers: vec![
                AdaptationTrigger::PerformanceDegradation,
                AdaptationTrigger::TimeInterval,
                AdaptationTrigger::AutomaticDetection,
            ],
            adaptation_learning_rate: 0.01,
            exploration_exploitation_balance: 0.1,
        }
    }
}

impl Default for ParallelOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            max_parallel_jobs: 4,
            load_balancing_strategy: LoadBalancingStrategy::PerformanceBased,
            resource_allocation: ResourceAllocationStrategy::Dynamic,
        }
    }
}

impl Default for CircuitAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_analysis: true,
            analysis_depth: AnalysisDepth::Intermediate,
            complexity_analysis: ComplexityAnalysisConfig::default(),
            structure_analysis: StructureAnalysisConfig::default(),
            optimization_potential: OptimizationPotentialConfig::default(),
        }
    }
}

impl Default for ComplexityAnalysisConfig {
    fn default() -> Self {
        Self {
            complexity_metrics: vec![
                ComplexityMetric::CircuitDepth,
                ComplexityMetric::GateCount,
                ComplexityMetric::TwoQubitGateCount,
                ComplexityMetric::ConnectivityComplexity,
            ],
            enable_parallel_analysis: true,
            analysis_timeout: Duration::from_secs(60),
            resource_limits: AnalysisResourceLimits::default(),
        }
    }
}

impl Default for StructureAnalysisConfig {
    fn default() -> Self {
        Self {
            pattern_recognition: true,
            symmetry_detection: true,
            subcircuit_identification: true,
            critical_path_analysis: true,
        }
    }
}

impl Default for OptimizationPotentialConfig {
    fn default() -> Self {
        Self {
            potential_metrics: vec![
                PotentialMetric::GateReduction,
                PotentialMetric::DepthReduction,
                PotentialMetric::ParallelizationOpportunity,
            ],
            opportunity_detection: true,
            bottleneck_identification: true,
            resource_utilization_analysis: true,
        }
    }
}

impl Default for AnalysisResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 2048,
            max_cpu_time: Duration::from_secs(300),
            max_threads: 8,
        }
    }
}
