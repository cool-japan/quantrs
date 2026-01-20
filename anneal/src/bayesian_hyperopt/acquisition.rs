//! Acquisition Function Configuration Types

/// Acquisition function configuration
#[derive(Debug, Clone)]
pub struct AcquisitionConfig {
    /// Type of acquisition function
    pub function_type: AcquisitionFunctionType,
    /// Exploration-exploitation balance parameter
    pub exploration_factor: f64,
    /// Number of random restarts for acquisition optimization
    pub num_restarts: usize,
    /// Batch acquisition strategy
    pub batch_strategy: BatchAcquisitionStrategy,
    /// Acquisition function optimization method
    pub optimization_method: AcquisitionOptimizationMethod,
}

impl Default for AcquisitionConfig {
    fn default() -> Self {
        Self {
            function_type: AcquisitionFunctionType::ExpectedImprovement,
            exploration_factor: 0.1,
            num_restarts: 10,
            batch_strategy: BatchAcquisitionStrategy::LocalPenalization,
            optimization_method: AcquisitionOptimizationMethod::LBFGS,
        }
    }
}

/// Types of acquisition functions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcquisitionFunctionType {
    /// Expected Improvement
    ExpectedImprovement,
    /// Upper Confidence Bound
    UpperConfidenceBound,
    /// Probability of Improvement
    ProbabilityOfImprovement,
    /// Entropy Search
    EntropySearch,
    /// Knowledge Gradient
    KnowledgeGradient,
    /// Multi-objective Expected Hypervolume Improvement
    ExpectedHypervolumeImprovement,
    /// Thompson Sampling
    ThompsonSampling,
}

/// Batch acquisition strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchAcquisitionStrategy {
    /// Local penalization method
    LocalPenalization,
    /// Constant liar strategy
    ConstantLiar,
    /// Kriging believer approach
    KrigingBeliever,
    /// Monte Carlo acquisition functions
    MonteCarlo,
}

/// Acquisition function optimization methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcquisitionOptimizationMethod {
    /// Limited-memory BFGS
    LBFGS,
    /// Differential Evolution
    DifferentialEvolution,
    /// Particle Swarm Optimization
    ParticleSwarm,
    /// Random search
    RandomSearch,
    /// Gradient-free methods
    GradientFree,
}

/// Acquisition function implementation
#[derive(Debug, Clone)]
pub struct AcquisitionFunction {
    pub function_type: AcquisitionFunctionType,
    pub exploration_factor: f64,
}

impl Default for AcquisitionFunction {
    fn default() -> Self {
        Self {
            function_type: AcquisitionFunctionType::ExpectedImprovement,
            exploration_factor: 0.1,
        }
    }
}
