//! Type definitions for active learning decomposition

use super::{DecompositionStrategy, PerformanceRecord};
use crate::ising::IsingModel;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Types of learning models
#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    /// Linear model
    Linear,
    /// Random forest
    RandomForest,
    /// Neural network
    NeuralNetwork,
    /// Gaussian process
    GaussianProcess,
    /// Ensemble model
    Ensemble,
}

/// Query strategies for active learning
#[derive(Debug, Clone, PartialEq)]
pub enum QueryStrategy {
    /// Uncertainty sampling
    UncertaintySampling,
    /// Expected improvement
    ExpectedImprovement,
    /// Information gain
    InformationGain,
    /// Diversity sampling
    DiversitySampling,
    /// Hybrid strategy
    Hybrid,
}

/// Diversity metrics
#[derive(Debug, Clone, PartialEq)]
pub enum DiversityMetric {
    /// Euclidean distance
    Euclidean,
    /// Cosine similarity
    Cosine,
    /// Jaccard similarity
    Jaccard,
    /// Graph edit distance
    GraphEditDistance,
}

/// Structure types in problems
#[derive(Debug, Clone, PartialEq)]
pub enum StructureType {
    /// Grid structure
    Grid,
    /// Tree structure
    Tree,
    /// Bipartite structure
    Bipartite,
    /// Community structure
    Community,
    /// Random structure
    Random,
    /// Custom structure
    Custom(String),
}

/// Domain adaptation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum DomainAdaptationStrategy {
    /// Fine-tuning
    FineTuning,
    /// Feature adaptation
    FeatureAdaptation,
    /// Model ensemble
    ModelEnsemble,
    /// Domain adversarial training
    DomainAdversarial,
    /// Meta-learning
    MetaLearning,
}

/// Community detection algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum CommunityDetectionAlgorithm {
    /// Louvain algorithm
    Louvain,
    /// Leiden algorithm
    Leiden,
    /// Spectral clustering
    SpectralClustering,
    /// Label propagation
    LabelPropagation,
    /// Modularity optimization
    ModularityOptimization,
}

/// Path finding algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum PathFindingAlgorithm {
    /// Dijkstra's algorithm
    Dijkstra,
    /// A* algorithm
    AStar,
    /// Bellman-Ford algorithm
    BellmanFord,
    /// Floyd-Warshall algorithm
    FloydWarshall,
}

/// Weight calculation methods
#[derive(Debug, Clone, PartialEq)]
pub enum WeightCalculationMethod {
    /// Coupling strength based
    CouplingStrength,
    /// Inverse coupling strength
    InverseCouplingStrength,
    /// Uniform weights
    Uniform,
    /// Custom weights
    Custom,
}

/// Types of bottlenecks
#[derive(Debug, Clone, PartialEq)]
pub enum BottleneckType {
    /// Vertex bottleneck
    Vertex,
    /// Edge bottleneck
    Edge,
    /// Community bridge
    CommunityBridge,
    /// High-degree vertex
    HighDegreeVertex,
}

/// Decomposition actions
#[derive(Debug, Clone, PartialEq)]
pub enum DecompositionAction {
    /// Split at bottleneck
    SplitAtBottleneck,
    /// Isolate bottleneck
    IsolateBottleneck,
    /// Replicate bottleneck
    ReplicateBottleneck,
    /// Bridge decomposition
    BridgeDecomposition,
    /// No action needed
    NoAction,
}

/// Pattern types
#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    /// Grid pattern
    Grid,
    /// Tree pattern
    Tree,
    /// Star pattern
    Star,
    /// Clique pattern
    Clique,
    /// Bipartite pattern
    Bipartite,
    /// Custom pattern
    Custom(String),
}

/// Pattern matching algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum PatternMatchingAlgorithm {
    /// Subgraph isomorphism
    SubgraphIsomorphism,
    /// Graph neural network
    GraphNeuralNetwork,
    /// Template matching
    TemplateMatching,
    /// Statistical matching
    StatisticalMatching,
}

/// Constraint types
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    /// Size constraint
    Size,
    /// Density constraint
    Density,
    /// Degree constraint
    Degree,
    /// Distance constraint
    Distance,
}

/// Complexity metrics
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ComplexityMetric {
    /// Time complexity
    TimeComplexity,
    /// Space complexity
    SpaceComplexity,
    /// Approximation hardness
    ApproximationHardness,
    /// Parameterized complexity
    ParameterizedComplexity,
}

/// Complexity model types
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityModelType {
    /// Polynomial model
    Polynomial,
    /// Exponential model
    Exponential,
    /// Machine learning model
    MachineLearning,
    /// Empirical model
    Empirical,
}

/// Complexity classes
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityClass {
    /// Polynomial time
    P,
    /// Nondeterministic polynomial time
    NP,
    /// NP-Complete
    NPComplete,
    /// NP-Hard
    NPHard,
    /// PSPACE
    PSPACE,
    /// EXPTIME
    EXPTIME,
}

/// Scoring function types
#[derive(Debug, Clone, PartialEq)]
pub enum ScoringFunctionType {
    /// Modularity-based scoring
    Modularity,
    /// Cut-based scoring
    CutBased,
    /// Balance-based scoring
    BalanceBased,
    /// Connectivity-based scoring
    ConnectivityBased,
    /// Custom scoring function
    Custom(String),
}

/// Types of cuts
#[derive(Debug, Clone, PartialEq)]
pub enum CutType {
    /// Minimum cut
    MinimumCut,
    /// Balanced cut
    BalancedCut,
    /// Sparse cut
    SparseCut,
    /// Spectral cut
    SpectralCut,
}

/// Risk levels
#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
    /// Very high risk
    VeryHigh,
}

/// Risk factor types
#[derive(Debug, Clone, PartialEq)]
pub enum RiskFactorType {
    /// Solution quality degradation
    QualityDegradation,
    /// Increased computation time
    TimeIncrease,
    /// Memory overhead
    MemoryOverhead,
    /// Coordination complexity
    CoordinationComplexity,
    /// Information loss
    InformationLoss,
}

/// Mitigation strategy types
#[derive(Debug, Clone, PartialEq)]
pub enum MitigationStrategyType {
    /// Overlap regions
    OverlapRegions,
    /// Iterative refinement
    IterativeRefinement,
    /// Global coordination
    GlobalCoordination,
    /// Redundant computation
    RedundantComputation,
    /// Quality monitoring
    QualityMonitoring,
}

/// Generation strategy types
#[derive(Debug, Clone, PartialEq)]
pub enum GenerationStrategyType {
    /// Graph partitioning
    GraphPartitioning,
    /// Community-based decomposition
    CommunityBased,
    /// Hierarchical decomposition
    Hierarchical,
    /// Random decomposition
    Random,
    /// Greedy decomposition
    Greedy,
    /// Spectral decomposition
    Spectral,
}

/// Overlap strategies
#[derive(Debug, Clone, PartialEq)]
pub enum OverlapStrategy {
    /// No overlap
    NoOverlap,
    /// Fixed overlap
    FixedOverlap,
    /// Adaptive overlap
    AdaptiveOverlap,
    /// Critical vertex overlap
    CriticalVertexOverlap,
}

/// Overlap resolution methods
#[derive(Debug, Clone, PartialEq)]
pub enum OverlapResolutionMethod {
    /// Voting
    Voting,
    /// Weighted average
    WeightedAverage,
    /// Best solution
    BestSolution,
    /// Consensus building
    ConsensusBuilding,
}

/// Size balancing strategies
#[derive(Debug, Clone, PartialEq)]
pub enum SizeBalancingStrategy {
    /// Strict balancing
    Strict,
    /// Flexible balancing
    Flexible,
    /// Quality-first balancing
    QualityFirst,
    /// No balancing
    NoBalancing,
}

/// Validation criterion types
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationCriterionType {
    /// Connectivity preservation
    ConnectivityPreservation,
    /// Information preservation
    InformationPreservation,
    /// Size balance
    SizeBalance,
    /// Cut quality
    CutQuality,
    /// Decomposition feasibility
    DecompositionFeasibility,
}

/// Evaluation metrics
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EvaluationMetric {
    /// Solution quality
    SolutionQuality,
    /// Computation time
    ComputationTime,
    /// Memory usage
    MemoryUsage,
    /// Parallelization efficiency
    ParallelizationEfficiency,
    /// Decomposition overhead
    DecompositionOverhead,
}

/// Trend directions
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Improving performance
    Improving,
    /// Declining performance
    Declining,
    /// Stable performance
    Stable,
    /// Oscillating performance
    Oscillating,
}

/// Condition types
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionType {
    /// Size-based condition
    SizeBased,
    /// Structure-based condition
    StructureBased,
    /// Performance-based condition
    PerformanceBased,
    /// Context-based condition
    ContextBased,
}

/// Logical operators
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOperator {
    /// AND
    And,
    /// OR
    Or,
    /// NOT
    Not,
    /// Implies
    Implies,
}

/// Action types
#[derive(Debug, Clone, PartialEq)]
pub enum ActionType {
    /// Recommend strategy
    RecommendStrategy,
    /// Adjust parameters
    AdjustParameters,
    /// Trigger learning
    TriggerLearning,
    /// Request feedback
    RequestFeedback,
}

/// Side effect types
#[derive(Debug, Clone, PartialEq)]
pub enum SideEffectType {
    /// Increased computation time
    IncreasedTime,
    /// Increased memory usage
    IncreasedMemory,
    /// Reduced solution quality
    ReducedQuality,
    /// Coordination overhead
    CoordinationOverhead,
}

/// Requirement types
#[derive(Debug, Clone, PartialEq)]
pub enum RequirementType {
    /// Hardware requirement
    Hardware,
    /// Software requirement
    Software,
    /// Performance requirement
    Performance,
    /// Resource requirement
    Resource,
}

/// Requirement value
#[derive(Debug, Clone)]
pub enum RequirementValue {
    /// Exact value
    Exact(f64),
    /// Range
    Range(f64, f64),
    /// Minimum value
    Minimum(f64),
    /// Maximum value
    Maximum(f64),
}

/// Graph metrics
#[derive(Debug, Clone)]
pub struct GraphMetrics {
    /// Number of vertices
    pub num_vertices: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Graph density
    pub density: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Average path length
    pub avg_path_length: f64,
    /// Modularity
    pub modularity: f64,
    /// Spectral gap
    pub spectral_gap: f64,
    /// Treewidth estimate
    pub treewidth_estimate: usize,
}

/// Detected structure
#[derive(Debug, Clone)]
pub struct DetectedStructure {
    /// Structure type
    pub structure_type: StructureType,
    /// Vertices in structure
    pub vertices: Vec<usize>,
    /// Structure confidence
    pub confidence: f64,
    /// Recommended decomposition
    pub recommended_decomposition: DecompositionStrategy,
}

/// Complexity estimate
#[derive(Debug, Clone)]
pub struct ComplexityEstimate {
    /// Estimated complexity class
    pub complexity_class: ComplexityClass,
    /// Numeric complexity estimate
    pub numeric_estimate: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Estimation method used
    pub estimation_method: String,
}

/// Decomposability score
#[derive(Debug, Clone)]
pub struct DecomposabilityScore {
    /// Overall score
    pub overall_score: f64,
    /// Individual component scores
    pub component_scores: HashMap<String, f64>,
    /// Decomposition recommendation
    pub recommendation: DecompositionRecommendation,
    /// Confidence level
    pub confidence: f64,
}

/// Decomposition recommendation
#[derive(Debug, Clone)]
pub struct DecompositionRecommendation {
    /// Recommended strategy
    pub strategy: DecompositionStrategy,
    /// Recommended cut points
    pub cut_points: Vec<CutPoint>,
    /// Expected benefit
    pub expected_benefit: f64,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
}

/// Cut point for decomposition
#[derive(Debug, Clone)]
pub struct CutPoint {
    /// Cut type
    pub cut_type: CutType,
    /// Vertices to separate
    pub vertices: Vec<usize>,
    /// Edges to cut
    pub edges: Vec<(usize, usize)>,
    /// Cut weight
    pub weight: f64,
}

/// Risk assessment for decomposition
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    /// Risk level
    pub risk_level: RiskLevel,
    /// Risk factors
    pub risk_factors: Vec<RiskFactor>,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<MitigationStrategy>,
}

/// Risk factor
#[derive(Debug, Clone)]
pub struct RiskFactor {
    /// Factor type
    pub factor_type: RiskFactorType,
    /// Severity
    pub severity: f64,
    /// Probability
    pub probability: f64,
    /// Impact assessment
    pub impact: String,
}

/// Mitigation strategy
#[derive(Debug, Clone)]
pub struct MitigationStrategy {
    /// Strategy type
    pub strategy_type: MitigationStrategyType,
    /// Implementation cost
    pub implementation_cost: f64,
    /// Expected effectiveness
    pub effectiveness: f64,
    /// Strategy description
    pub description: String,
}
