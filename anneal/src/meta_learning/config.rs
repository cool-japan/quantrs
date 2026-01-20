//! Configuration types for Meta-Learning Optimization
//!
//! This module contains all configuration structures and enums used throughout
//! the meta-learning optimization system.

use std::collections::HashMap;
use std::time::Duration;

/// Meta-learning optimization engine configuration
#[derive(Debug, Clone)]
pub struct MetaLearningConfig {
    /// Enable transfer learning
    pub enable_transfer_learning: bool,
    /// Enable few-shot learning
    pub enable_few_shot_learning: bool,
    /// Experience buffer size
    pub experience_buffer_size: usize,
    /// Learning rate for meta-updates
    pub meta_learning_rate: f64,
    /// Number of inner optimization steps
    pub inner_steps: usize,
    /// Feature extraction configuration
    pub feature_config: FeatureExtractionConfig,
    /// Neural architecture search settings
    pub nas_config: NeuralArchitectureSearchConfig,
    /// Portfolio management settings
    pub portfolio_config: PortfolioManagementConfig,
    /// Multi-objective optimization settings
    pub multi_objective_config: MultiObjectiveConfig,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            enable_transfer_learning: true,
            enable_few_shot_learning: true,
            experience_buffer_size: 10_000,
            meta_learning_rate: 0.001,
            inner_steps: 5,
            feature_config: FeatureExtractionConfig::default(),
            nas_config: NeuralArchitectureSearchConfig::default(),
            portfolio_config: PortfolioManagementConfig::default(),
            multi_objective_config: MultiObjectiveConfig::default(),
        }
    }
}

/// Feature extraction configuration
#[derive(Debug, Clone)]
pub struct FeatureExtractionConfig {
    /// Enable graph-based features
    pub enable_graph_features: bool,
    /// Enable statistical features
    pub enable_statistical_features: bool,
    /// Enable spectral features
    pub enable_spectral_features: bool,
    /// Enable domain-specific features
    pub enable_domain_features: bool,
    /// Feature selection method
    pub selection_method: FeatureSelectionMethod,
    /// Dimensionality reduction method
    pub reduction_method: DimensionalityReduction,
    /// Feature normalization
    pub normalization: FeatureNormalization,
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            enable_graph_features: true,
            enable_statistical_features: true,
            enable_spectral_features: true,
            enable_domain_features: true,
            selection_method: FeatureSelectionMethod::AutomaticRelevance,
            reduction_method: DimensionalityReduction::PCA,
            normalization: FeatureNormalization::StandardScaling,
        }
    }
}

/// Feature selection methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureSelectionMethod {
    /// Automatic relevance determination
    AutomaticRelevance,
    /// Mutual information
    MutualInformation,
    /// Recursive feature elimination
    RecursiveElimination,
    /// LASSO regularization
    LASSO,
    /// Random forest importance
    RandomForestImportance,
}

/// Dimensionality reduction methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimensionalityReduction {
    /// Principal Component Analysis
    PCA,
    /// Independent Component Analysis
    ICA,
    /// t-Distributed Stochastic Neighbor Embedding
    tSNE,
    /// Uniform Manifold Approximation and Projection
    UMAP,
    /// Linear Discriminant Analysis
    LDA,
    /// No reduction
    None,
}

/// Feature normalization methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureNormalization {
    /// Standard scaling (z-score)
    StandardScaling,
    /// Min-max scaling
    MinMaxScaling,
    /// Robust scaling
    RobustScaling,
    /// Unit vector scaling
    UnitVector,
    /// No normalization
    None,
}

/// Neural Architecture Search configuration
#[derive(Debug, Clone)]
pub struct NeuralArchitectureSearchConfig {
    /// Enable NAS
    pub enable_nas: bool,
    /// Search space definition
    pub search_space: SearchSpace,
    /// Search strategy
    pub search_strategy: SearchStrategy,
    /// Maximum search iterations
    pub max_iterations: usize,
    /// Early stopping criteria
    pub early_stopping: EarlyStoppingCriteria,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}

impl Default for NeuralArchitectureSearchConfig {
    fn default() -> Self {
        Self {
            enable_nas: true,
            search_space: SearchSpace::default(),
            search_strategy: SearchStrategy::DifferentiableNAS,
            max_iterations: 100,
            early_stopping: EarlyStoppingCriteria::default(),
            resource_constraints: ResourceConstraints::default(),
        }
    }
}

/// Neural architecture search space
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Layer types to consider
    pub layer_types: Vec<LayerType>,
    /// Number of layers range
    pub num_layers_range: (usize, usize),
    /// Hidden dimension options
    pub hidden_dims: Vec<usize>,
    /// Activation functions
    pub activations: Vec<ActivationFunction>,
    /// Dropout rates
    pub dropout_rates: Vec<f64>,
    /// Skip connection options
    pub skip_connections: bool,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            layer_types: vec![
                LayerType::Dense,
                LayerType::LSTM,
                LayerType::GRU,
                LayerType::Attention,
                LayerType::Convolution1D,
            ],
            num_layers_range: (2, 8),
            hidden_dims: vec![64, 128, 256, 512],
            activations: vec![
                ActivationFunction::ReLU,
                ActivationFunction::Tanh,
                ActivationFunction::Swish,
                ActivationFunction::GELU,
            ],
            dropout_rates: vec![0.0, 0.1, 0.2, 0.3],
            skip_connections: true,
        }
    }
}

/// Neural network layer types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerType {
    /// Dense/Linear layer
    Dense,
    /// LSTM layer
    LSTM,
    /// GRU layer
    GRU,
    /// Attention layer
    Attention,
    /// 1D Convolution layer
    Convolution1D,
    /// Normalization layer
    Normalization,
    /// Residual block
    ResidualBlock,
}

/// Activation functions
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationFunction {
    ReLU,
    Tanh,
    Sigmoid,
    Swish,
    GELU,
    LeakyReLU(f64),
    ELU(f64),
}

/// Search strategies for NAS
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Differentiable NAS
    DifferentiableNAS,
    /// Evolutionary search
    EvolutionarySearch,
    /// Reinforcement learning
    ReinforcementLearning,
    /// Bayesian optimization
    BayesianOptimization,
    /// Random search
    RandomSearch,
    /// Progressive search
    ProgressiveSearch,
}

/// Early stopping criteria
#[derive(Debug, Clone)]
pub struct EarlyStoppingCriteria {
    /// Patience (iterations without improvement)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_improvement: f64,
    /// Maximum runtime
    pub max_runtime: Duration,
    /// Target performance threshold
    pub target_performance: Option<f64>,
}

impl Default for EarlyStoppingCriteria {
    fn default() -> Self {
        Self {
            patience: 10,
            min_improvement: 0.001,
            max_runtime: Duration::from_secs(2 * 3600),
            target_performance: None,
        }
    }
}

/// Resource constraints for NAS
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum memory usage (MB)
    pub max_memory: usize,
    /// Maximum training time per architecture
    pub max_training_time: Duration,
    /// Maximum model parameters
    pub max_parameters: usize,
    /// Maximum FLOPs
    pub max_flops: usize,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_memory: 2048,
            max_training_time: Duration::from_secs(10 * 60),
            max_parameters: 1_000_000,
            max_flops: 1_000_000_000,
        }
    }
}

/// Portfolio management configuration
#[derive(Debug, Clone)]
pub struct PortfolioManagementConfig {
    /// Enable dynamic portfolio
    pub enable_dynamic_portfolio: bool,
    /// Maximum portfolio size
    pub max_portfolio_size: usize,
    /// Algorithm selection strategy
    pub selection_strategy: AlgorithmSelectionStrategy,
    /// Performance evaluation window
    pub evaluation_window: Duration,
    /// Diversity criteria
    pub diversity_criteria: DiversityCriteria,
}

impl Default for PortfolioManagementConfig {
    fn default() -> Self {
        Self {
            enable_dynamic_portfolio: true,
            max_portfolio_size: 10,
            selection_strategy: AlgorithmSelectionStrategy::MultiArmedBandit,
            evaluation_window: Duration::from_secs(24 * 3600),
            diversity_criteria: DiversityCriteria::default(),
        }
    }
}

/// Algorithm selection strategies
#[derive(Debug, Clone, PartialEq)]
pub enum AlgorithmSelectionStrategy {
    /// Multi-armed bandit
    MultiArmedBandit,
    /// Upper confidence bound
    UpperConfidenceBound,
    /// Thompson sampling
    ThompsonSampling,
    /// ε-greedy
    EpsilonGreedy(f64),
    /// Collaborative filtering
    CollaborativeFiltering,
    /// Meta-learning based
    MetaLearningBased,
}

/// Diversity criteria for portfolio
#[derive(Debug, Clone)]
pub struct DiversityCriteria {
    /// Minimum performance diversity
    pub min_performance_diversity: f64,
    /// Minimum algorithmic diversity
    pub min_algorithmic_diversity: f64,
    /// Diversity measurement method
    pub diversity_method: DiversityMethod,
}

impl Default for DiversityCriteria {
    fn default() -> Self {
        Self {
            min_performance_diversity: 0.1,
            min_algorithmic_diversity: 0.2,
            diversity_method: DiversityMethod::KullbackLeibler,
        }
    }
}

/// Diversity measurement methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiversityMethod {
    /// Kullback-Leibler divergence
    KullbackLeibler,
    /// Jensen-Shannon divergence
    JensenShannon,
    /// Cosine distance
    CosineDistance,
    /// Euclidean distance
    EuclideanDistance,
    /// Hamming distance
    HammingDistance,
}

/// Multi-objective optimization configuration
#[derive(Debug, Clone)]
pub struct MultiObjectiveConfig {
    /// Enable multi-objective optimization
    pub enable_multi_objective: bool,
    /// Objectives to optimize
    pub objectives: Vec<OptimizationObjective>,
    /// Pareto frontier management
    pub pareto_config: ParetoFrontierConfig,
    /// Scalarization method
    pub scalarization: ScalarizationMethod,
    /// Constraint handling
    pub constraint_handling: ConstraintHandling,
}

impl Default for MultiObjectiveConfig {
    fn default() -> Self {
        Self {
            enable_multi_objective: true,
            objectives: vec![
                OptimizationObjective::SolutionQuality,
                OptimizationObjective::Runtime,
                OptimizationObjective::ResourceUsage,
            ],
            pareto_config: ParetoFrontierConfig::default(),
            scalarization: ScalarizationMethod::WeightedSum,
            constraint_handling: ConstraintHandling::PenaltyMethod,
        }
    }
}

/// Optimization objectives
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationObjective {
    /// Solution quality
    SolutionQuality,
    /// Runtime performance
    Runtime,
    /// Resource usage
    ResourceUsage,
    /// Energy consumption
    EnergyConsumption,
    /// Robustness
    Robustness,
    /// Scalability
    Scalability,
    /// Custom objective
    Custom(String),
}

/// Pareto frontier configuration
#[derive(Debug, Clone)]
pub struct ParetoFrontierConfig {
    /// Maximum frontier size
    pub max_frontier_size: usize,
    /// Dominance tolerance
    pub dominance_tolerance: f64,
    /// Frontier update strategy
    pub update_strategy: FrontierUpdateStrategy,
    /// Crowding distance weight
    pub crowding_weight: f64,
}

impl Default for ParetoFrontierConfig {
    fn default() -> Self {
        Self {
            max_frontier_size: 100,
            dominance_tolerance: 1e-6,
            update_strategy: FrontierUpdateStrategy::NonDominatedSort,
            crowding_weight: 0.5,
        }
    }
}

/// Frontier update strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrontierUpdateStrategy {
    /// Non-dominated sorting
    NonDominatedSort,
    /// ε-dominance
    EpsilonDominance,
    /// Hypervolume-based
    HypervolumeBased,
    /// Reference point-based
    ReferencePointBased,
}

/// Scalarization methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarizationMethod {
    /// Weighted sum
    WeightedSum,
    /// Weighted Tchebycheff
    WeightedTchebycheff,
    /// Achievement scalarizing function
    AchievementScalarizing,
    /// Penalty-based boundary intersection
    PenaltyBoundaryIntersection,
    /// Reference point method
    ReferencePoint,
}

/// Constraint handling methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintHandling {
    /// Penalty method
    PenaltyMethod,
    /// Barrier method
    BarrierMethod,
    /// Lagrangian method
    LagrangianMethod,
    /// Feasibility rules
    FeasibilityRules,
    /// Multi-objective constraint handling
    MultiObjectiveConstraint,
}

/// Algorithm types
#[derive(Debug, Clone, PartialEq)]
pub enum AlgorithmType {
    /// Simulated annealing
    SimulatedAnnealing,
    /// Quantum annealing
    QuantumAnnealing,
    /// Tabu search
    TabuSearch,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Particle swarm optimization
    ParticleSwarm,
    /// Ant colony optimization
    AntColony,
    /// Variable neighborhood search
    VariableNeighborhood,
    /// Hybrid algorithm
    Hybrid(Vec<Self>),
}

/// Architecture specification
#[derive(Debug, Clone)]
pub struct ArchitectureSpec {
    /// Layer specifications
    pub layers: Vec<LayerSpec>,
    /// Connection pattern
    pub connections: ConnectionPattern,
    /// Optimization settings
    pub optimization: OptimizationSettings,
}

/// Layer specification
#[derive(Debug, Clone)]
pub struct LayerSpec {
    /// Layer type
    pub layer_type: LayerType,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function
    pub activation: ActivationFunction,
    /// Dropout rate
    pub dropout: f64,
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
}

/// Connection patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionPattern {
    /// Sequential connections
    Sequential,
    /// Skip connections
    SkipConnections,
    /// Dense connections
    DenseConnections,
    /// Residual connections
    ResidualConnections,
    /// Custom pattern
    Custom(Vec<(usize, usize)>),
}

/// Optimization settings
#[derive(Debug, Clone)]
pub struct OptimizationSettings {
    /// Optimizer type
    pub optimizer: OptimizerType,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Regularization
    pub regularization: RegularizationConfig,
}

/// Optimizer types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizerType {
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    Adadelta,
    LBFGS,
}

/// Regularization configuration
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// L1 regularization weight
    pub l1_weight: f64,
    /// L2 regularization weight
    pub l2_weight: f64,
    /// Dropout rate
    pub dropout: f64,
    /// Batch normalization
    pub batch_norm: bool,
    /// Early stopping
    pub early_stopping: bool,
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfiguration {
    /// Algorithm used
    pub algorithm: AlgorithmType,
    /// Hyperparameters
    pub hyperparameters: HashMap<String, f64>,
    /// Architecture specification
    pub architecture: Option<ArchitectureSpec>,
    /// Resource allocation
    pub resources: ResourceAllocation,
}

/// Resource allocation
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// CPU allocation
    pub cpu: f64,
    /// Memory allocation (MB)
    pub memory: usize,
    /// GPU allocation
    pub gpu: f64,
    /// Time allocation
    pub time: Duration,
}

/// Problem domains
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProblemDomain {
    /// Combinatorial optimization
    Combinatorial,
    /// Portfolio optimization
    Portfolio,
    /// Scheduling
    Scheduling,
    /// Graph problems
    Graph,
    /// Machine learning
    MachineLearning,
    /// Physics simulation
    Physics,
    /// Chemistry
    Chemistry,
    /// Custom domain
    Custom(String),
}
