//! Core types and data structures for RL embedding optimization

use crate::embedding::{Embedding, HardwareTopology};
use crate::hardware_compilation::HardwareType;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for RL embedding optimizer
#[derive(Debug, Clone)]
pub struct RLEmbeddingConfig {
    /// DQN network architecture
    pub dqn_layers: Vec<usize>,
    /// Policy network architecture
    pub policy_layers: Vec<usize>,
    /// Learning rate
    pub learning_rate: f64,
    /// Experience buffer size
    pub buffer_size: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Exploration parameters
    pub exploration_config: ExplorationConfig,
    /// Target network update frequency
    pub target_update_frequency: usize,
    /// Discount factor
    pub discount_factor: f64,
    /// Multi-objective weights
    pub objective_weights: ObjectiveWeights,
    /// Transfer learning configuration
    pub transfer_learning: TransferLearningConfig,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for RLEmbeddingConfig {
    fn default() -> Self {
        Self {
            dqn_layers: vec![128, 256, 128, 64],
            policy_layers: vec![128, 256, 128, 32],
            learning_rate: 0.0001,
            buffer_size: 10_000,
            batch_size: 64,
            exploration_config: ExplorationConfig::default(),
            target_update_frequency: 1000,
            discount_factor: 0.99,
            objective_weights: ObjectiveWeights::default(),
            transfer_learning: TransferLearningConfig::default(),
            seed: None,
        }
    }
}

/// Exploration configuration for RL
#[derive(Debug, Clone)]
pub struct ExplorationConfig {
    /// Initial epsilon for epsilon-greedy
    pub initial_epsilon: f64,
    /// Final epsilon
    pub final_epsilon: f64,
    /// Epsilon decay steps
    pub epsilon_decay_steps: usize,
    /// Exploration noise for policy
    pub policy_noise: f64,
    /// Curiosity-driven exploration weight
    pub curiosity_weight: f64,
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        Self {
            initial_epsilon: 1.0,
            final_epsilon: 0.01,
            epsilon_decay_steps: 10_000,
            policy_noise: 0.1,
            curiosity_weight: 0.1,
        }
    }
}

/// Multi-objective optimization weights
#[derive(Debug, Clone)]
pub struct ObjectiveWeights {
    /// Chain length minimization weight
    pub chain_length_weight: f64,
    /// Embedding efficiency weight
    pub efficiency_weight: f64,
    /// Hardware utilization weight
    pub utilization_weight: f64,
    /// Connectivity preservation weight
    pub connectivity_weight: f64,
    /// Performance prediction weight
    pub performance_weight: f64,
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self {
            chain_length_weight: 0.3,
            efficiency_weight: 0.25,
            utilization_weight: 0.2,
            connectivity_weight: 0.15,
            performance_weight: 0.1,
        }
    }
}

/// Transfer learning configuration
#[derive(Debug, Clone)]
pub struct TransferLearningConfig {
    /// Enable transfer learning
    pub enabled: bool,
    /// Source domain weight decay
    pub source_weight_decay: f64,
    /// Adaptation learning rate
    pub adaptation_lr: f64,
    /// Fine-tuning epochs
    pub fine_tuning_epochs: usize,
    /// Domain similarity threshold
    pub similarity_threshold: f64,
}

impl Default for TransferLearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            source_weight_decay: 0.9,
            adaptation_lr: 0.00_001,
            fine_tuning_epochs: 100,
            similarity_threshold: 0.7,
        }
    }
}

/// State representation for embedding RL
#[derive(Debug, Clone)]
pub struct EmbeddingState {
    /// Problem graph features
    pub problem_features: ProblemGraphFeatures,
    /// Hardware topology features
    pub hardware_features: HardwareFeatures,
    /// Current embedding state
    pub embedding_state: CurrentEmbeddingState,
    /// Performance history
    pub performance_history: Vec<f64>,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Problem graph characteristics
#[derive(Debug, Clone)]
pub struct ProblemGraphFeatures {
    /// Number of vertices
    pub num_vertices: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Graph density
    pub density: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Average degree
    pub average_degree: f64,
    /// Degree distribution moments
    pub degree_moments: Vec<f64>,
    /// Graph diameter
    pub diameter: usize,
    /// Modularity
    pub modularity: f64,
    /// Spectral properties
    pub spectral_features: SpectralFeatures,
}

/// Spectral graph features
#[derive(Debug, Clone)]
pub struct SpectralFeatures {
    /// Largest eigenvalue
    pub largest_eigenvalue: f64,
    /// Second largest eigenvalue
    pub second_largest_eigenvalue: f64,
    /// Spectral gap
    pub spectral_gap: f64,
    /// Algebraic connectivity
    pub algebraic_connectivity: f64,
    /// Eigenvalue distribution moments
    pub eigenvalue_moments: Vec<f64>,
}

/// Hardware topology features
#[derive(Debug, Clone)]
pub struct HardwareFeatures {
    /// Hardware type
    pub hardware_type: HardwareType,
    /// Number of physical qubits
    pub num_physical_qubits: usize,
    /// Connectivity graph properties
    pub connectivity_features: ConnectivityFeatures,
    /// Hardware constraints
    pub constraints: HardwareConstraints,
    /// Performance characteristics
    pub performance_chars: HardwarePerformanceChars,
}

/// Hardware connectivity features
#[derive(Debug, Clone)]
pub struct ConnectivityFeatures {
    /// Connectivity degree distribution
    pub degree_distribution: Vec<usize>,
    /// Average connectivity
    pub average_connectivity: f64,
    /// Maximum connectivity
    pub max_connectivity: usize,
    /// Connectivity variance
    pub connectivity_variance: f64,
    /// Topology regularity
    pub regularity_measure: f64,
}

/// Hardware constraints
#[derive(Debug, Clone)]
pub struct HardwareConstraints {
    /// Maximum chain length allowed
    pub max_chain_length: usize,
    /// Coupling strength limitations
    pub coupling_constraints: Vec<f64>,
    /// Noise characteristics
    pub noise_profile: NoiseProfile,
    /// Timing constraints
    pub timing_constraints: TimingConstraints,
}

/// Noise profile of hardware
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// Coherence times
    pub coherence_times: Vec<f64>,
    /// Error rates
    pub error_rates: Vec<f64>,
    /// Cross-talk coefficients
    pub crosstalk_matrix: Vec<Vec<f64>>,
    /// Temperature stability
    pub temperature_stability: f64,
}

/// Timing constraints
#[derive(Debug, Clone)]
pub struct TimingConstraints {
    /// Minimum annealing time
    pub min_anneal_time: f64,
    /// Maximum annealing time
    pub max_anneal_time: f64,
    /// Readout time
    pub readout_time: f64,
    /// Programming time
    pub programming_time: f64,
}

/// Hardware performance characteristics
#[derive(Debug, Clone)]
pub struct HardwarePerformanceChars {
    /// Success probability estimates
    pub success_probabilities: Vec<f64>,
    /// Energy scale factors
    pub energy_scales: Vec<f64>,
    /// Bandwidth limitations
    pub bandwidth_limits: Vec<f64>,
    /// Calibration quality metrics
    pub calibration_quality: CalibrationQuality,
}

/// Calibration quality metrics
#[derive(Debug, Clone)]
pub struct CalibrationQuality {
    /// Bias calibration accuracy
    pub bias_accuracy: f64,
    /// Coupling calibration accuracy
    pub coupling_accuracy: f64,
    /// Frequency drift
    pub frequency_drift: f64,
    /// Last calibration time
    pub last_calibration: Duration,
}

/// Current embedding state
#[derive(Debug, Clone)]
pub struct CurrentEmbeddingState {
    /// Current mapping of logical to physical qubits
    pub logical_to_physical: HashMap<usize, Vec<usize>>,
    /// Chain lengths
    pub chain_lengths: Vec<usize>,
    /// Embedding efficiency metrics
    pub efficiency_metrics: EmbeddingEfficiencyMetrics,
    /// Unutilized hardware resources
    pub unused_resources: Vec<usize>,
    /// Embedding quality score
    pub quality_score: f64,
}

/// Embedding efficiency metrics
#[derive(Debug, Clone)]
pub struct EmbeddingEfficiencyMetrics {
    /// Hardware utilization ratio
    pub utilization_ratio: f64,
    /// Average chain length
    pub avg_chain_length: f64,
    /// Maximum chain length
    pub max_chain_length: usize,
    /// Connectivity preservation ratio
    pub connectivity_preservation: f64,
    /// Embedding compactness
    pub compactness: f64,
}

/// Resource utilization information
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// Physical qubit usage
    pub qubit_usage: Vec<bool>,
    /// Coupling usage
    pub coupling_usage: Vec<bool>,
    /// Memory usage
    pub memory_usage: f64,
    /// Computational overhead
    pub computational_overhead: f64,
    /// Energy consumption estimate
    pub energy_consumption: f64,
}

/// Actions for embedding optimization
#[derive(Debug, Clone)]
pub enum EmbeddingAction {
    /// Discrete actions for DQN
    Discrete(DiscreteEmbeddingAction),
    /// Continuous actions for policy networks
    Continuous(ContinuousEmbeddingAction),
    /// Hybrid discrete-continuous actions
    Hybrid {
        discrete: DiscreteEmbeddingAction,
        continuous: ContinuousEmbeddingAction,
    },
}

/// Discrete embedding actions
#[derive(Debug, Clone)]
pub enum DiscreteEmbeddingAction {
    /// Add qubit to chain
    AddToChain {
        logical_qubit: usize,
        physical_qubit: usize,
    },
    /// Remove qubit from chain
    RemoveFromChain {
        logical_qubit: usize,
        physical_qubit: usize,
    },
    /// Swap chain assignments
    SwapChains { chain1: usize, chain2: usize },
    /// Relocate entire chain
    RelocateChain {
        logical_qubit: usize,
        new_location: Vec<usize>,
    },
    /// Merge adjacent chains
    MergeChains { chain1: usize, chain2: usize },
    /// Split long chain
    SplitChain { chain: usize, split_point: usize },
    /// Optimize chain ordering
    OptimizeOrdering { chain: usize },
    /// No operation
    NoOp,
}

/// Continuous embedding actions
#[derive(Debug, Clone)]
pub struct ContinuousEmbeddingAction {
    /// Chain strength adjustments
    pub chain_strength_adjustments: Vec<f64>,
    /// Coupling strength modifications
    pub coupling_modifications: Vec<f64>,
    /// Bias adjustments
    pub bias_adjustments: Vec<f64>,
    /// Annealing parameter modifications
    pub annealing_adjustments: Vec<f64>,
}

/// Experience tuple for reinforcement learning
#[derive(Debug, Clone)]
pub struct EmbeddingExperience {
    /// Current state (problem features + current embedding state)
    pub state: EmbeddingState,
    /// Action taken
    pub action: EmbeddingAction,
    /// Reward received
    pub reward: f64,
    /// Next state
    pub next_state: EmbeddingState,
    /// Episode termination flag
    pub done: bool,
    /// Additional context
    pub context: ExperienceContext,
}

/// Experience context
#[derive(Debug, Clone)]
pub struct ExperienceContext {
    /// Problem type identifier
    pub problem_type: String,
    /// Hardware identifier
    pub hardware_id: String,
    /// Timestamp
    pub timestamp: Instant,
    /// Episode identifier
    pub episode_id: usize,
    /// Step within episode
    pub step: usize,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

/// Training statistics for RL
#[derive(Debug, Clone)]
pub struct RLTrainingStats {
    /// Episode rewards over time
    pub episode_rewards: Vec<f64>,
    /// Average reward (rolling)
    pub average_reward: f64,
    /// Loss values over time
    pub loss_history: Vec<f64>,
    /// Exploration rate over time
    pub exploration_history: Vec<f64>,
    /// Q-value statistics
    pub q_value_stats: QValueStatistics,
    /// Policy statistics
    pub policy_stats: PolicyStatistics,
    /// Transfer learning statistics
    pub transfer_stats: TransferLearningStats,
}

/// Q-value statistics
#[derive(Debug, Clone)]
pub struct QValueStatistics {
    /// Average Q-values
    pub average_q_values: Vec<f64>,
    /// Q-value variance
    pub q_value_variance: Vec<f64>,
    /// Action-value distribution
    pub action_value_distribution: HashMap<String, Vec<f64>>,
    /// Temporal difference errors
    pub td_errors: Vec<f64>,
}

/// Policy statistics
#[derive(Debug, Clone)]
pub struct PolicyStatistics {
    /// Action probabilities over time
    pub action_probabilities: Vec<HashMap<String, f64>>,
    /// Policy entropy
    pub policy_entropy: Vec<f64>,
    /// Actor loss
    pub actor_loss: Vec<f64>,
    /// Critic loss
    pub critic_loss: Vec<f64>,
}

/// Transfer learning statistics
#[derive(Debug, Clone)]
pub struct TransferLearningStats {
    /// Source domain performance
    pub source_performance: Vec<f64>,
    /// Target domain performance
    pub target_performance: Vec<f64>,
    /// Transfer effectiveness
    pub transfer_effectiveness: f64,
    /// Domain similarity measures
    pub domain_similarities: Vec<f64>,
}

/// Cached embedding information
#[derive(Debug, Clone)]
pub struct CachedEmbedding {
    /// The embedding itself
    pub embedding: Embedding,
    /// Quality metrics
    pub quality_metrics: EmbeddingQualityMetrics,
    /// Performance results
    pub performance_results: EmbeddingPerformanceResults,
    /// Cache metadata
    pub cache_metadata: CacheMetadata,
}

/// Embedding quality metrics
#[derive(Debug, Clone)]
pub struct EmbeddingQualityMetrics {
    /// Overall quality score
    pub overall_score: f64,
    /// Chain length penalty
    pub chain_length_penalty: f64,
    /// Connectivity preservation score
    pub connectivity_score: f64,
    /// Hardware utilization efficiency
    pub utilization_efficiency: f64,
    /// Predicted performance score
    pub predicted_performance: f64,
}

/// Embedding performance results
#[derive(Debug, Clone)]
pub struct EmbeddingPerformanceResults {
    /// Success probability
    pub success_probability: f64,
    /// Average energy gap
    pub average_energy_gap: f64,
    /// Solution quality distribution
    pub solution_quality: Vec<f64>,
    /// Runtime statistics
    pub runtime_stats: RuntimeStatistics,
}

/// Runtime statistics
#[derive(Debug, Clone)]
pub struct RuntimeStatistics {
    /// Embedding computation time
    pub embedding_time: Duration,
    /// Hardware execution time
    pub execution_time: Duration,
    /// Total optimization time
    pub total_time: Duration,
    /// Memory usage
    pub memory_usage: usize,
}

/// Cache metadata
#[derive(Debug, Clone)]
pub struct CacheMetadata {
    /// Creation time
    pub created_at: Instant,
    /// Last accessed time
    pub last_accessed: Instant,
    /// Access count
    pub access_count: usize,
    /// Cache hit rate
    pub hit_rate: f64,
}

/// Performance metrics for RL embedding optimizer
#[derive(Debug, Clone)]
pub struct RLPerformanceMetrics {
    /// Total problems solved
    pub problems_solved: usize,
    /// Average improvement over baseline
    pub average_improvement: f64,
    /// Best improvement achieved
    pub best_improvement: f64,
    /// Learning convergence rate
    pub convergence_rate: f64,
    /// Transfer learning effectiveness
    pub transfer_effectiveness: f64,
    /// Computational efficiency
    pub computational_efficiency: f64,
}

impl RLTrainingStats {
    /// Create new training statistics
    #[must_use]
    pub fn new() -> Self {
        Self {
            episode_rewards: Vec::new(),
            average_reward: 0.0,
            loss_history: Vec::new(),
            exploration_history: Vec::new(),
            q_value_stats: QValueStatistics {
                average_q_values: Vec::new(),
                q_value_variance: Vec::new(),
                action_value_distribution: HashMap::new(),
                td_errors: Vec::new(),
            },
            policy_stats: PolicyStatistics {
                action_probabilities: Vec::new(),
                policy_entropy: Vec::new(),
                actor_loss: Vec::new(),
                critic_loss: Vec::new(),
            },
            transfer_stats: TransferLearningStats {
                source_performance: Vec::new(),
                target_performance: Vec::new(),
                transfer_effectiveness: 0.0,
                domain_similarities: Vec::new(),
            },
        }
    }
}

impl RLPerformanceMetrics {
    /// Create new performance metrics
    #[must_use]
    pub const fn new() -> Self {
        Self {
            problems_solved: 0,
            average_improvement: 0.0,
            best_improvement: 0.0,
            convergence_rate: 0.0,
            transfer_effectiveness: 0.0,
            computational_efficiency: 0.0,
        }
    }
}
