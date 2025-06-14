//! Type definitions for SciRS2 mapping

use super::*;

/// Initial mapping algorithms
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InitialMappingAlgorithm {
    /// Spectral embedding for optimal initial placement
    SpectralEmbedding,
    /// Community detection based mapping
    CommunityBased,
    /// Centrality-weighted assignment
    CentralityWeighted,
    /// Minimum spanning tree based
    MSTreeBased,
    /// PageRank weighted assignment
    PageRankWeighted,
    /// Bipartite matching for optimal assignment
    BipartiteMatching,
    /// Multi-level graph partitioning
    MultilevelPartitioning,
}

/// SciRS2 routing algorithms
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SciRS2RoutingAlgorithm {
    /// Enhanced A* with graph metrics
    AStarEnhanced,
    /// Community-aware routing
    CommunityAware,
    /// Spectral-based shortest paths
    SpectralRouting,
    /// Centrality-guided routing
    CentralityGuided,
    /// Multi-path routing with load balancing
    MultiPath,
    /// Adaptive routing based on real-time metrics
    AdaptiveRouting,
}

/// Optimization objectives
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Minimize number of SWAP operations
    MinimizeSwaps,
    /// Minimize circuit depth
    MinimizeDepth,
    /// Maximize gate fidelity
    MaximizeFidelity,
    /// Minimize execution time
    MinimizeTime,
    /// Hybrid objective combining multiple factors
    HybridObjective,
    /// Custom objective function
    CustomObjective,
}

/// Community detection methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CommunityMethod {
    /// Louvain community detection
    Louvain,
    /// Leiden algorithm
    Leiden,
    /// Label propagation
    LabelPropagation,
    /// Spectral clustering
    SpectralClustering,
    /// Walktrap algorithm
    Walktrap,
}

/// ML model types for mapping
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MLModelType {
    GraphNeuralNetwork { hidden_dims: Vec<usize> },
    GraphConvolutionalNetwork { layers: usize },
    GraphAttentionNetwork { heads: usize },
    DeepQLearning { experience_buffer_size: usize },
    PolicyGradient { actor_critic: bool },
    TreeSearch { simulation_count: usize },
    EnsembleMethod { base_models: Vec<String> },
}

/// Feature selection methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeatureSelectionMethod {
    VarianceThreshold { threshold: f64 },
    UnivariateSelection { k_best: usize },
    RecursiveElimination { step_size: usize },
    LassoRegularization { alpha: f64 },
    MutualInformation { bins: usize },
    PrincipalComponentAnalysis { n_components: usize },
}

/// Analysis depth levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnalysisDepth {
    Basic,
    Standard,
    Comprehensive,
    Expert,
}

/// Tracking metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrackingMetric {
    ExecutionTime,
    MemoryUsage,
    MappingQuality,
    SwapCount,
    FidelityLoss,
    CommunicationOverhead,
    ResourceUtilization,
    ConvergenceRate,
}

/// Anomaly detection methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnomalyDetectionMethod {
    IsolationForest,
    OneClassSVM,
    LocalOutlierFactor,
    EllipticEnvelope,
    StatisticalThreshold,
}

/// Notification methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NotificationMethod {
    Log,
    Email,
    Webhook,
    MQTT,
    Console,
}

/// Report formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportFormat {
    JSON,
    XML,
    CSV,
    HTML,
    PDF,
}

/// Selection methods for multi-objective optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SelectionMethod {
    Tournament { size: usize },
    Roulette,
    Rank,
    Random,
}

/// Scalarization methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalarizationMethod {
    WeightedSum { weights: Vec<f64> },
    Tchebycheff { reference_point: Vec<f64> },
    AugmentedTchebycheff { weights: Vec<f64>, rho: f64 },
    WeightedMetric { p: f64, weights: Vec<f64> },
}

/// Constraint types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintType {
    ConnectivityConstraint,
    TimingConstraint,
    ResourceConstraint,
    FidelityConstraint,
    PowerConstraint,
}

/// Penalty methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PenaltyMethod {
    QuadraticPenalty { coefficient: f64 },
    ExponentialPenalty { base: f64 },
    AdaptivePenalty { initial_penalty: f64 },
    BarrierMethod { barrier_parameter: f64 },
}

/// Search strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SearchStrategy {
    GeneticAlgorithm,
    SimulatedAnnealing,
    ParticleSwarm,
    DifferentialEvolution,
    HybridSearch,
    TabuSearch,
}

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    Static,
    Dynamic,
    WorkStealing,
    RoundRobin,
}

/// Synchronization methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SynchronizationMethod {
    Synchronous,
    Asynchronous,
    BulkSynchronous,
    EventDriven,
}

/// Domain adaptation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DomainAdaptationMethod {
    FineTuning,
    FeatureAlignment,
    AdversarialTraining,
    DomainAdversarialNeuralNetwork,
    GradientReversal,
}

/// Calibration methods for prediction confidence
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CalibrationMethod {
    PlattScaling,
    IsotonicRegression,
    TemperatureScaling,
    BetaCalibration,
    HistogramBinning,
}

/// Suggestion priority levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SuggestionPriority {
    High,
    Medium,
    Low,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
}

/// Quality metric types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QualityMetricType {
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUC,
    MeanAbsoluteError,
    RootMeanSquareError,
    R2Score,
}

/// Result analysis types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResultAnalysisType {
    Statistical,
    Temporal,
    Comparative,
    Predictive,
    Causal,
}

/// Optimization phases
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationPhase {
    Initialization,
    Exploration,
    Exploitation,
    Convergence,
    PostProcessing,
}

/// Risk assessment levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PerformanceCategory {
    Excellent,
    Good,
    Average,
    Poor,
    Critical,
}

/// Resource utilization states
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceState {
    Underutilized,
    Optimal,
    Overutilized,
    Saturated,
    Unavailable,
}

/// Learning phases
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LearningPhase {
    Initialization,
    Training,
    Validation,
    Testing,
    Deployment,
}

/// Adaptation triggers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AdaptationTrigger {
    PerformanceDegradation,
    EnvironmentChange,
    ResourceConstraints,
    QualityThreshold,
    TimeInterval,
    ManualTrigger,
}