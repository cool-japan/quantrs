//! Active Learning for Problem Decomposition
//!
//! This module implements active learning techniques for intelligent decomposition
//! of complex optimization problems into smaller, more manageable subproblems.
//! It uses machine learning to guide the decomposition process and adaptively
//! improve decomposition strategies based on performance feedback.
//!
//! Key features:
//! - Intelligent problem decomposition using graph analysis
//! - Active learning for decomposition strategy selection
//! - Hierarchical decomposition with adaptive granularity
//! - Performance-guided decomposition refinement
//! - Multi-objective decomposition optimization
//! - Transfer learning across problem domains

use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::ising::{IsingModel};
use crate::simulator::{AnnealingResult};

/// Active learning decomposer for optimization problems
#[derive(Debug, Clone)]
pub struct ActiveLearningDecomposer {
    /// Decomposition strategy learner
    pub strategy_learner: DecompositionStrategyLearner,
    /// Problem analyzer
    pub problem_analyzer: ProblemAnalyzer,
    /// Subproblem generator
    pub subproblem_generator: SubproblemGenerator,
    /// Performance evaluator
    pub performance_evaluator: PerformanceEvaluator,
    /// Knowledge base
    pub knowledge_base: DecompositionKnowledgeBase,
    /// Configuration
    pub config: ActiveLearningConfig,
}

/// Configuration for active learning decomposition
#[derive(Debug, Clone)]
pub struct ActiveLearningConfig {
    /// Enable online learning
    pub enable_online_learning: bool,
    /// Maximum decomposition depth
    pub max_decomposition_depth: usize,
    /// Minimum subproblem size
    pub min_subproblem_size: usize,
    /// Maximum subproblem size
    pub max_subproblem_size: usize,
    /// Learning rate for strategy updates
    pub learning_rate: f64,
    /// Exploration rate for active learning
    pub exploration_rate: f64,
    /// Performance threshold for decomposition
    pub performance_threshold: f64,
    /// Enable transfer learning
    pub enable_transfer_learning: bool,
    /// Active learning query budget
    pub query_budget: usize,
    /// Decomposition overlap tolerance
    pub overlap_tolerance: f64,
}

/// Decomposition strategy learner
#[derive(Debug, Clone)]
pub struct DecompositionStrategyLearner {
    /// Strategy selection model
    pub selection_model: StrategySelectionModel,
    /// Strategy performance history
    pub performance_history: HashMap<String, Vec<PerformanceRecord>>,
    /// Active learning query selector
    pub query_selector: QuerySelector,
    /// Transfer learning manager
    pub transfer_learning: TransferLearningManager,
    /// Learning statistics
    pub learning_stats: LearningStatistics,
}

/// Strategy selection model
#[derive(Debug, Clone)]
pub struct StrategySelectionModel {
    /// Model type
    pub model_type: ModelType,
    /// Feature weights
    pub feature_weights: Array1<f64>,
    /// Strategy preferences
    pub strategy_preferences: HashMap<DecompositionStrategy, f64>,
    /// Uncertainty estimates
    pub uncertainty_estimates: HashMap<String, f64>,
    /// Model parameters
    pub model_parameters: ModelParameters,
}

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

/// Model parameters
#[derive(Debug, Clone)]
pub struct ModelParameters {
    /// Model-specific parameters
    pub parameters: HashMap<String, f64>,
    /// Regularization parameters
    pub regularization: RegularizationParameters,
    /// Training configuration
    pub training_config: ModelTrainingConfig,
}

/// Regularization parameters
#[derive(Debug, Clone)]
pub struct RegularizationParameters {
    /// L1 regularization weight
    pub l1_weight: f64,
    /// L2 regularization weight
    pub l2_weight: f64,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Early stopping patience
    pub early_stopping_patience: usize,
}

/// Model training configuration
#[derive(Debug, Clone)]
pub struct ModelTrainingConfig {
    /// Number of training epochs
    pub num_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Validation split
    pub validation_split: f64,
}

/// Query selector for active learning
#[derive(Debug, Clone)]
pub struct QuerySelector {
    /// Query strategy
    pub query_strategy: QueryStrategy,
    /// Uncertainty threshold
    pub uncertainty_threshold: f64,
    /// Diversity constraint
    pub diversity_constraint: DiversityConstraint,
    /// Query history
    pub query_history: Vec<QueryRecord>,
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

/// Diversity constraint for query selection
#[derive(Debug, Clone)]
pub struct DiversityConstraint {
    /// Minimum distance between queries
    pub min_distance: f64,
    /// Diversity metric
    pub diversity_metric: DiversityMetric,
    /// Maximum similarity allowed
    pub max_similarity: f64,
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

/// Query record
#[derive(Debug, Clone)]
pub struct QueryRecord {
    /// Query timestamp
    pub timestamp: Instant,
    /// Queried problem features
    pub problem_features: Array1<f64>,
    /// Recommended strategy
    pub recommended_strategy: DecompositionStrategy,
    /// Query outcome
    pub query_outcome: QueryOutcome,
    /// Performance feedback
    pub performance_feedback: Option<PerformanceRecord>,
}

/// Query outcome
#[derive(Debug, Clone)]
pub struct QueryOutcome {
    /// Strategy actually used
    pub strategy_used: DecompositionStrategy,
    /// User accepted recommendation
    pub accepted_recommendation: bool,
    /// Performance achieved
    pub performance_achieved: f64,
    /// Feedback quality
    pub feedback_quality: f64,
}

/// Transfer learning manager
#[derive(Debug, Clone)]
pub struct TransferLearningManager {
    /// Source domain models
    pub source_models: Vec<SourceDomainModel>,
    /// Domain adaptation strategy
    pub adaptation_strategy: DomainAdaptationStrategy,
    /// Knowledge transfer weights
    pub transfer_weights: Array1<f64>,
    /// Transfer learning statistics
    pub transfer_stats: TransferStatistics,
}

/// Source domain model
#[derive(Debug, Clone)]
pub struct SourceDomainModel {
    /// Domain identifier
    pub domain_id: String,
    /// Model for this domain
    pub model: StrategySelectionModel,
    /// Domain characteristics
    pub domain_characteristics: DomainCharacteristics,
    /// Transfer applicability score
    pub applicability_score: f64,
}

/// Domain characteristics
#[derive(Debug, Clone)]
pub struct DomainCharacteristics {
    /// Problem types in domain
    pub problem_types: Vec<String>,
    /// Average problem size
    pub avg_problem_size: f64,
    /// Problem complexity distribution
    pub complexity_distribution: Array1<f64>,
    /// Common structures
    pub common_structures: Vec<StructureType>,
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

/// Transfer learning statistics
#[derive(Debug, Clone)]
pub struct TransferStatistics {
    /// Successful transfers
    pub successful_transfers: usize,
    /// Failed transfers
    pub failed_transfers: usize,
    /// Average transfer benefit
    pub avg_transfer_benefit: f64,
    /// Transfer time overhead
    pub transfer_time_overhead: Duration,
}

/// Learning statistics
#[derive(Debug, Clone)]
pub struct LearningStatistics {
    /// Total queries made
    pub total_queries: usize,
    /// Successful predictions
    pub successful_predictions: usize,
    /// Average prediction accuracy
    pub avg_prediction_accuracy: f64,
    /// Learning curve data
    pub learning_curve: Vec<(usize, f64)>, // (query_count, accuracy)
    /// Exploration vs exploitation ratio
    pub exploration_exploitation_ratio: f64,
}

/// Problem analyzer for decomposition
#[derive(Debug, Clone)]
pub struct ProblemAnalyzer {
    /// Graph analyzer
    pub graph_analyzer: GraphAnalyzer,
    /// Structure detector
    pub structure_detector: StructureDetector,
    /// Complexity estimator
    pub complexity_estimator: ComplexityEstimator,
    /// Decomposability scorer
    pub decomposability_scorer: DecomposabilityScorer,
}

/// Graph analyzer for problem structure
#[derive(Debug, Clone)]
pub struct GraphAnalyzer {
    /// Graph metrics calculator
    pub metrics_calculator: GraphMetricsCalculator,
    /// Community detection algorithm
    pub community_detector: CommunityDetector,
    /// Critical path analyzer
    pub critical_path_analyzer: CriticalPathAnalyzer,
    /// Bottleneck detector
    pub bottleneck_detector: BottleneckDetector,
}

/// Graph metrics calculator
#[derive(Debug, Clone)]
pub struct GraphMetricsCalculator {
    /// Cached metrics
    pub cached_metrics: HashMap<String, GraphMetrics>,
    /// Metric computation config
    pub computation_config: MetricComputationConfig,
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

/// Metric computation configuration
#[derive(Debug, Clone)]
pub struct MetricComputationConfig {
    /// Enable expensive metrics
    pub enable_expensive_metrics: bool,
    /// Approximation algorithms enabled
    pub enable_approximation: bool,
    /// Sampling ratio for large graphs
    pub sampling_ratio: f64,
    /// Timeout for metric computation
    pub computation_timeout: Duration,
}

/// Community detection
#[derive(Debug, Clone)]
pub struct CommunityDetector {
    /// Detection algorithm
    pub algorithm: CommunityDetectionAlgorithm,
    /// Resolution parameter
    pub resolution: f64,
    /// Minimum community size
    pub min_community_size: usize,
    /// Maximum community size
    pub max_community_size: usize,
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

/// Critical path analyzer
#[derive(Debug, Clone)]
pub struct CriticalPathAnalyzer {
    /// Path finding algorithm
    pub algorithm: PathFindingAlgorithm,
    /// Weight calculation method
    pub weight_method: WeightCalculationMethod,
    /// Critical path cache
    pub path_cache: HashMap<String, CriticalPath>,
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

/// Critical path information
#[derive(Debug, Clone)]
pub struct CriticalPath {
    /// Path vertices
    pub vertices: Vec<usize>,
    /// Path weight
    pub weight: f64,
    /// Bottleneck edges
    pub bottleneck_edges: Vec<(usize, usize)>,
    /// Alternative paths
    pub alternative_paths: Vec<AlternativePath>,
}

/// Alternative path
#[derive(Debug, Clone)]
pub struct AlternativePath {
    /// Path vertices
    pub vertices: Vec<usize>,
    /// Path weight
    pub weight: f64,
    /// Overlap with critical path
    pub overlap_ratio: f64,
}

/// Bottleneck detector
#[derive(Debug, Clone)]
pub struct BottleneckDetector {
    /// Detection threshold
    pub detection_threshold: f64,
    /// Bottleneck types to detect
    pub bottleneck_types: Vec<BottleneckType>,
    /// Detected bottlenecks cache
    pub bottlenecks_cache: HashMap<String, Vec<Bottleneck>>,
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

/// Bottleneck information
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Affected vertices
    pub affected_vertices: Vec<usize>,
    /// Affected edges
    pub affected_edges: Vec<(usize, usize)>,
    /// Severity score
    pub severity: f64,
    /// Suggested decomposition action
    pub decomposition_action: DecompositionAction,
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

/// Structure detector for problem patterns
#[derive(Debug, Clone)]
pub struct StructureDetector {
    /// Pattern matching algorithms
    pub pattern_matchers: Vec<PatternMatcher>,
    /// Structure templates
    pub structure_templates: Vec<StructureTemplate>,
    /// Detection confidence threshold
    pub confidence_threshold: f64,
    /// Detected structures cache
    pub structures_cache: HashMap<String, Vec<DetectedStructure>>,
}

/// Pattern matcher
#[derive(Debug, Clone)]
pub struct PatternMatcher {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Matching algorithm
    pub algorithm: PatternMatchingAlgorithm,
    /// Matching parameters
    pub parameters: PatternMatchingParameters,
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

/// Pattern matching parameters
#[derive(Debug, Clone)]
pub struct PatternMatchingParameters {
    /// Matching tolerance
    pub tolerance: f64,
    /// Minimum pattern size
    pub min_pattern_size: usize,
    /// Maximum pattern size
    pub max_pattern_size: usize,
    /// Allow overlapping patterns
    pub allow_overlap: bool,
}

/// Structure template
#[derive(Debug, Clone)]
pub struct StructureTemplate {
    /// Template name
    pub name: String,
    /// Template graph
    pub template_graph: TemplateGraph,
    /// Decomposition strategy for this structure
    pub decomposition_strategy: DecompositionStrategy,
    /// Expected performance gain
    pub expected_gain: f64,
}

/// Template graph representation
#[derive(Debug, Clone)]
pub struct TemplateGraph {
    /// Template adjacency matrix
    pub adjacency_matrix: Array2<u8>,
    /// Template features
    pub features: Array1<f64>,
    /// Template constraints
    pub constraints: Vec<TemplateConstraint>,
}

/// Template constraints
#[derive(Debug, Clone)]
pub struct TemplateConstraint {
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Constraint parameters
    pub parameters: HashMap<String, f64>,
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

/// Complexity estimator
#[derive(Debug, Clone)]
pub struct ComplexityEstimator {
    /// Complexity metrics
    pub complexity_metrics: Vec<ComplexityMetric>,
    /// Estimation models
    pub estimation_models: HashMap<ComplexityMetric, ComplexityModel>,
    /// Complexity cache
    pub complexity_cache: HashMap<String, ComplexityEstimate>,
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

/// Complexity model
#[derive(Debug, Clone)]
pub struct ComplexityModel {
    /// Model type
    pub model_type: ComplexityModelType,
    /// Model parameters
    pub parameters: Array1<f64>,
    /// Prediction accuracy
    pub accuracy: f64,
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

/// Decomposability scorer
#[derive(Debug, Clone)]
pub struct DecomposabilityScorer {
    /// Scoring functions
    pub scoring_functions: Vec<ScoringFunction>,
    /// Score weights
    pub score_weights: Array1<f64>,
    /// Scoring cache
    pub scoring_cache: HashMap<String, DecomposabilityScore>,
}

/// Scoring function
#[derive(Debug, Clone)]
pub struct ScoringFunction {
    /// Function type
    pub function_type: ScoringFunctionType,
    /// Function parameters
    pub parameters: HashMap<String, f64>,
    /// Function weight
    pub weight: f64,
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

/// Subproblem generator
#[derive(Debug, Clone)]
pub struct SubproblemGenerator {
    /// Generation strategies
    pub generation_strategies: Vec<GenerationStrategy>,
    /// Overlap manager
    pub overlap_manager: OverlapManager,
    /// Size controller
    pub size_controller: SizeController,
    /// Quality validator
    pub quality_validator: QualityValidator,
}

/// Generation strategy
#[derive(Debug, Clone)]
pub struct GenerationStrategy {
    /// Strategy type
    pub strategy_type: GenerationStrategyType,
    /// Strategy parameters
    pub parameters: GenerationParameters,
    /// Success rate
    pub success_rate: f64,
    /// Average quality
    pub average_quality: f64,
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

/// Generation parameters
#[derive(Debug, Clone)]
pub struct GenerationParameters {
    /// Target number of subproblems
    pub target_num_subproblems: usize,
    /// Size balance tolerance
    pub size_balance_tolerance: f64,
    /// Quality threshold
    pub quality_threshold: f64,
    /// Maximum iterations
    pub max_iterations: usize,
}

/// Overlap manager
#[derive(Debug, Clone)]
pub struct OverlapManager {
    /// Overlap strategy
    pub overlap_strategy: OverlapStrategy,
    /// Overlap size
    pub overlap_size: usize,
    /// Overlap resolution method
    pub resolution_method: OverlapResolutionMethod,
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

/// Size controller
#[derive(Debug, Clone)]
pub struct SizeController {
    /// Size constraints
    pub size_constraints: SizeConstraints,
    /// Size balancing strategy
    pub balancing_strategy: SizeBalancingStrategy,
    /// Adaptive sizing enabled
    pub adaptive_sizing: bool,
}

/// Size constraints
#[derive(Debug, Clone)]
pub struct SizeConstraints {
    /// Minimum subproblem size
    pub min_size: usize,
    /// Maximum subproblem size
    pub max_size: usize,
    /// Target size
    pub target_size: usize,
    /// Size tolerance
    pub size_tolerance: f64,
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

/// Quality validator
#[derive(Debug, Clone)]
pub struct QualityValidator {
    /// Validation criteria
    pub validation_criteria: Vec<ValidationCriterion>,
    /// Validation threshold
    pub validation_threshold: f64,
    /// Strict validation enabled
    pub strict_validation: bool,
}

/// Validation criterion
#[derive(Debug, Clone)]
pub struct ValidationCriterion {
    /// Criterion type
    pub criterion_type: ValidationCriterionType,
    /// Weight in overall validation
    pub weight: f64,
    /// Threshold for this criterion
    pub threshold: f64,
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

/// Performance evaluator
#[derive(Debug, Clone)]
pub struct PerformanceEvaluator {
    /// Evaluation metrics
    pub evaluation_metrics: Vec<EvaluationMetric>,
    /// Baseline comparisons
    pub baseline_comparisons: Vec<BaselineComparison>,
    /// Performance history
    pub performance_history: Vec<PerformanceRecord>,
    /// Evaluation cache
    pub evaluation_cache: HashMap<String, EvaluationResult>,
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

/// Baseline comparison
#[derive(Debug, Clone)]
pub struct BaselineComparison {
    /// Baseline name
    pub baseline_name: String,
    /// Baseline strategy
    pub baseline_strategy: DecompositionStrategy,
    /// Performance comparison
    pub performance_comparison: PerformanceComparison,
}

/// Performance comparison
#[derive(Debug, Clone)]
pub struct PerformanceComparison {
    /// Improvement factor
    pub improvement_factor: f64,
    /// Statistical significance
    pub statistical_significance: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

/// Performance record
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    /// Timestamp
    pub timestamp: Instant,
    /// Problem identifier
    pub problem_id: String,
    /// Strategy used
    pub strategy_used: DecompositionStrategy,
    /// Performance metrics
    pub metrics: HashMap<EvaluationMetric, f64>,
    /// Overall performance score
    pub overall_score: f64,
}

/// Evaluation result
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Individual metric scores
    pub metric_scores: HashMap<EvaluationMetric, f64>,
    /// Overall evaluation score
    pub overall_score: f64,
    /// Performance improvement
    pub improvement_over_baseline: f64,
    /// Evaluation confidence
    pub confidence: f64,
}

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

/// Known pattern
#[derive(Debug, Clone)]
pub struct KnownPattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Pattern description
    pub description: String,
    /// Pattern features
    pub features: Array1<f64>,
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
    pub structural_features: Array1<f64>,
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

/// Resource constraints
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum computation time
    pub max_computation_time: Duration,
    /// Maximum memory usage
    pub max_memory_usage: usize,
    /// Number of available processors
    pub num_processors: usize,
    /// Communication bandwidth
    pub communication_bandwidth: f64,
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

/// Decomposition strategies
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DecompositionStrategy {
    /// Graph partitioning based
    GraphPartitioning,
    /// Community detection based
    CommunityDetection,
    /// Spectral clustering
    SpectralClustering,
    /// Hierarchical decomposition
    Hierarchical,
    /// Random decomposition
    Random,
    /// Greedy decomposition
    Greedy,
    /// No decomposition
    NoDecomposition,
    /// Custom strategy
    Custom(String),
}

impl ActiveLearningDecomposer {
    /// Create new active learning decomposer
    pub fn new(config: ActiveLearningConfig) -> Result<Self, String> {
        let strategy_learner = DecompositionStrategyLearner::new()?;
        let problem_analyzer = ProblemAnalyzer::new()?;
        let subproblem_generator = SubproblemGenerator::new()?;
        let performance_evaluator = PerformanceEvaluator::new()?;
        let knowledge_base = DecompositionKnowledgeBase::new()?;

        Ok(Self {
            strategy_learner,
            problem_analyzer,
            subproblem_generator,
            performance_evaluator,
            knowledge_base,
            config,
        })
    }

    /// Decompose problem using active learning
    pub fn decompose_problem(
        &mut self,
        problem: &IsingModel,
    ) -> Result<DecompositionResult, String> {
        // Analyze problem structure and characteristics
        let problem_analysis = self.analyze_problem(problem)?;

        // Select decomposition strategy using active learning
        let strategy = self.select_strategy(problem, &problem_analysis)?;

        // Generate subproblems
        let subproblems = self.generate_subproblems(problem, &strategy, &problem_analysis)?;

        // Validate decomposition quality
        let quality_score = self.validate_decomposition_quality(&subproblems, problem)?;

        // Update knowledge base if learning is enabled
        if self.config.enable_online_learning {
            self.update_knowledge_base(problem, &strategy, quality_score)?;
        }

        Ok(DecompositionResult {
            subproblems,
            strategy_used: strategy,
            quality_score,
            analysis: problem_analysis,
            metadata: DecompositionMetadata::new(),
        })
    }

    /// Analyze problem structure and characteristics
    fn analyze_problem(&mut self, problem: &IsingModel) -> Result<ProblemAnalysis, String> {
        // Calculate graph metrics
        let graph_metrics = self.problem_analyzer.graph_analyzer.calculate_metrics(problem)?;

        // Detect communities
        let communities = self.problem_analyzer.graph_analyzer.detect_communities(problem)?;

        // Detect structures
        let structures = self.problem_analyzer.structure_detector.detect_structures(problem)?;

        // Estimate complexity
        let complexity = self.problem_analyzer.complexity_estimator.estimate_complexity(problem)?;

        // Score decomposability
        let decomposability = self.problem_analyzer.decomposability_scorer.score_decomposability(problem)?;

        Ok(ProblemAnalysis {
            graph_metrics,
            communities,
            structures,
            complexity,
            decomposability,
            problem_features: self.extract_problem_features(problem)?,
        })
    }

    /// Extract problem features for learning
    fn extract_problem_features(&self, problem: &IsingModel) -> Result<Array1<f64>, String> {
        let mut features = Vec::new();

        // Basic features
        features.push(problem.num_qubits as f64);
        
        // Count non-zero couplings
        let mut num_couplings = 0;
        for i in 0..problem.num_qubits {
            for j in (i+1)..problem.num_qubits {
                if problem.get_coupling(i, j).unwrap_or(0.0).abs() > 1e-10 {
                    num_couplings += 1;
                }
            }
        }
        features.push(num_couplings as f64);

        // Density
        let max_couplings = problem.num_qubits * (problem.num_qubits - 1) / 2;
        let density = if max_couplings > 0 { num_couplings as f64 / max_couplings as f64 } else { 0.0 };
        features.push(density);

        // Bias statistics
        let mut bias_sum = 0.0;
        let mut bias_var = 0.0;
        for i in 0..problem.num_qubits {
            let bias = problem.get_bias(i).unwrap_or(0.0);
            bias_sum += bias;
            bias_var += bias * bias;
        }
        let bias_mean = bias_sum / problem.num_qubits as f64;
        bias_var = bias_var / problem.num_qubits as f64 - bias_mean * bias_mean;
        features.extend_from_slice(&[bias_mean, bias_var.sqrt()]);

        // Coupling statistics
        let mut coupling_sum = 0.0;
        let mut coupling_var = 0.0;
        if num_couplings > 0 {
            for i in 0..problem.num_qubits {
                for j in (i+1)..problem.num_qubits {
                    let coupling = problem.get_coupling(i, j).unwrap_or(0.0);
                    if coupling.abs() > 1e-10 {
                        coupling_sum += coupling;
                        coupling_var += coupling * coupling;
                    }
                }
            }
            let coupling_mean = coupling_sum / num_couplings as f64;
            coupling_var = coupling_var / num_couplings as f64 - coupling_mean * coupling_mean;
            features.extend_from_slice(&[coupling_mean, coupling_var.sqrt()]);
        } else {
            features.extend_from_slice(&[0.0, 0.0]);
        }

        // Pad to fixed size
        features.resize(20, 0.0);
        Ok(Array1::from_vec(features))
    }

    /// Select decomposition strategy using active learning
    fn select_strategy(
        &mut self,
        problem: &IsingModel,
        analysis: &ProblemAnalysis,
    ) -> Result<DecompositionStrategy, String> {
        // Get strategy recommendation from model
        let recommendation = self.strategy_learner.recommend_strategy(problem, analysis)?;

        // Decide whether to explore or exploit
        let should_explore = self.should_explore(&analysis.problem_features)?;

        if should_explore {
            // Exploration: try a different strategy or query for feedback
            self.explore_strategy(problem, analysis, &recommendation)
        } else {
            // Exploitation: use the recommended strategy
            Ok(recommendation)
        }
    }

    /// Decide whether to explore or exploit
    fn should_explore(&self, features: &Array1<f64>) -> Result<bool, String> {
        // Get uncertainty estimate for this problem
        let uncertainty = self.strategy_learner.selection_model.get_uncertainty(features)?;
        
        // Compare with threshold and exploration rate
        let explore_threshold = self.config.exploration_rate;
        let uncertainty_threshold = 0.5; // High uncertainty threshold

        Ok(uncertainty > uncertainty_threshold || 
           rand::thread_rng().gen::<f64>() < explore_threshold)
    }

    /// Explore strategy selection
    fn explore_strategy(
        &mut self,
        problem: &IsingModel,
        analysis: &ProblemAnalysis,
        base_recommendation: &DecompositionStrategy,
    ) -> Result<DecompositionStrategy, String> {
        // Select query strategy
        match self.strategy_learner.query_selector.query_strategy {
            QueryStrategy::UncertaintySampling => {
                self.uncertainty_sampling_exploration(analysis)
            }
            QueryStrategy::DiversitySampling => {
                self.diversity_sampling_exploration(analysis)
            }
            _ => {
                // Default: slightly perturb the base recommendation
                self.perturb_strategy(base_recommendation)
            }
        }
    }

    /// Uncertainty sampling exploration
    fn uncertainty_sampling_exploration(&self, analysis: &ProblemAnalysis) -> Result<DecompositionStrategy, String> {
        // Find strategy with highest uncertainty
        let strategies = &self.knowledge_base.strategy_database.strategies;
        let mut best_strategy = DecompositionStrategy::GraphPartitioning;
        let mut highest_uncertainty = 0.0;

        for strategy in strategies {
            let uncertainty = self.strategy_learner.selection_model
                .get_strategy_uncertainty(strategy, &analysis.problem_features)?;
            
            if uncertainty > highest_uncertainty {
                highest_uncertainty = uncertainty;
                best_strategy = strategy.clone();
            }
        }

        Ok(best_strategy)
    }

    /// Diversity sampling exploration
    fn diversity_sampling_exploration(&self, analysis: &ProblemAnalysis) -> Result<DecompositionStrategy, String> {
        // Find strategy most different from recently used strategies
        let recent_strategies: Vec<_> = self.strategy_learner.query_selector.query_history
            .iter()
            .rev()
            .take(10)
            .map(|record| &record.recommended_strategy)
            .collect();

        let strategies = &self.knowledge_base.strategy_database.strategies;
        let mut best_strategy = DecompositionStrategy::GraphPartitioning;
        let mut max_diversity = 0.0;

        for strategy in strategies {
            let diversity = self.calculate_strategy_diversity(strategy, &recent_strategies)?;
            
            if diversity > max_diversity {
                max_diversity = diversity;
                best_strategy = strategy.clone();
            }
        }

        Ok(best_strategy)
    }

    /// Calculate strategy diversity
    fn calculate_strategy_diversity(
        &self,
        strategy: &DecompositionStrategy,
        recent_strategies: &[&DecompositionStrategy],
    ) -> Result<f64, String> {
        if recent_strategies.is_empty() {
            return Ok(1.0);
        }

        let mut total_distance = 0.0;
        for &recent_strategy in recent_strategies {
            total_distance += self.strategy_distance(strategy, recent_strategy)?;
        }

        Ok(total_distance / recent_strategies.len() as f64)
    }

    /// Calculate distance between strategies
    fn strategy_distance(
        &self,
        strategy1: &DecompositionStrategy,
        strategy2: &DecompositionStrategy,
    ) -> Result<f64, String> {
        // Simplified strategy distance
        if strategy1 == strategy2 {
            Ok(0.0)
        } else {
            Ok(1.0)
        }
    }

    /// Perturb strategy selection
    fn perturb_strategy(&self, base_strategy: &DecompositionStrategy) -> Result<DecompositionStrategy, String> {
        let strategies = &self.knowledge_base.strategy_database.strategies;
        let mut rng = rand::thread_rng();
        
        // Select a random strategy with some probability
        if rng.gen::<f64>() < 0.3 {
            let random_idx = rng.gen_range(0..strategies.len());
            Ok(strategies[random_idx].clone())
        } else {
            Ok(base_strategy.clone())
        }
    }

    /// Generate subproblems using selected strategy
    fn generate_subproblems(
        &mut self,
        problem: &IsingModel,
        strategy: &DecompositionStrategy,
        analysis: &ProblemAnalysis,
    ) -> Result<Vec<Subproblem>, String> {
        match strategy {
            DecompositionStrategy::GraphPartitioning => {
                self.graph_partitioning_decomposition(problem, analysis)
            }
            DecompositionStrategy::CommunityDetection => {
                self.community_detection_decomposition(problem, analysis)
            }
            DecompositionStrategy::SpectralClustering => {
                self.spectral_clustering_decomposition(problem, analysis)
            }
            DecompositionStrategy::NoDecomposition => {
                Ok(vec![Subproblem::from_full_problem(problem)])
            }
            _ => {
                // Default to graph partitioning
                self.graph_partitioning_decomposition(problem, analysis)
            }
        }
    }

    /// Graph partitioning decomposition
    fn graph_partitioning_decomposition(
        &mut self,
        problem: &IsingModel,
        analysis: &ProblemAnalysis,
    ) -> Result<Vec<Subproblem>, String> {
        // Simple bisection for demonstration
        let n = problem.num_qubits;
        let mid = n / 2;

        let partition1: Vec<usize> = (0..mid).collect();
        let partition2: Vec<usize> = (mid..n).collect();

        let subproblem1 = self.create_subproblem(problem, &partition1, 0)?;
        let subproblem2 = self.create_subproblem(problem, &partition2, 1)?;

        Ok(vec![subproblem1, subproblem2])
    }

    /// Community detection decomposition
    fn community_detection_decomposition(
        &mut self,
        problem: &IsingModel,
        analysis: &ProblemAnalysis,
    ) -> Result<Vec<Subproblem>, String> {
        // Use detected communities from analysis
        let communities = &analysis.communities;
        let mut subproblems = Vec::new();

        for (i, community) in communities.iter().enumerate() {
            if community.vertices.len() >= self.config.min_subproblem_size {
                let subproblem = self.create_subproblem(problem, &community.vertices, i)?;
                subproblems.push(subproblem);
            }
        }

        // If no valid communities, fall back to graph partitioning
        if subproblems.is_empty() {
            self.graph_partitioning_decomposition(problem, analysis)
        } else {
            Ok(subproblems)
        }
    }

    /// Spectral clustering decomposition
    fn spectral_clustering_decomposition(
        &mut self,
        problem: &IsingModel,
        analysis: &ProblemAnalysis,
    ) -> Result<Vec<Subproblem>, String> {
        // Simplified spectral clustering - in practice would use eigendecomposition
        self.graph_partitioning_decomposition(problem, analysis)
    }

    /// Create subproblem from vertex subset
    fn create_subproblem(
        &self,
        problem: &IsingModel,
        vertices: &[usize],
        subproblem_id: usize,
    ) -> Result<Subproblem, String> {
        // Create Ising model for subproblem
        let mut subproblem_model = IsingModel::new(vertices.len());

        // Map vertex indices
        let vertex_map: HashMap<usize, usize> = vertices.iter()
            .enumerate()
            .map(|(new_idx, &old_idx)| (old_idx, new_idx))
            .collect();

        // Copy biases
        for (new_idx, &old_idx) in vertices.iter().enumerate() {
            let bias = problem.get_bias(old_idx).unwrap_or(0.0);
            subproblem_model.set_bias(new_idx, bias).map_err(|e| e.to_string())?;
        }

        // Copy couplings within subproblem
        for (i, &old_i) in vertices.iter().enumerate() {
            for (j, &old_j) in vertices.iter().enumerate().skip(i + 1) {
                let coupling = problem.get_coupling(old_i, old_j).unwrap_or(0.0);
                if coupling.abs() > 1e-10 {
                    subproblem_model.set_coupling(i, j, coupling).map_err(|e| e.to_string())?;
                }
            }
        }

        // Identify boundary edges (edges connecting to other subproblems)
        let mut boundary_edges = Vec::new();
        for &vertex in vertices {
            for other_vertex in 0..problem.num_qubits {
                if !vertices.contains(&other_vertex) {
                    let coupling = problem.get_coupling(vertex, other_vertex).unwrap_or(0.0);
                    if coupling.abs() > 1e-10 {
                        boundary_edges.push(BoundaryEdge {
                            internal_vertex: vertex_map[&vertex],
                            external_vertex: other_vertex,
                            coupling_strength: coupling,
                        });
                    }
                }
            }
        }

        Ok(Subproblem {
            id: subproblem_id,
            model: subproblem_model,
            vertices: vertices.to_vec(),
            boundary_edges,
            metadata: SubproblemMetadata::new(),
        })
    }

    /// Validate decomposition quality
    fn validate_decomposition_quality(
        &self,
        subproblems: &[Subproblem],
        original_problem: &IsingModel,
    ) -> Result<f64, String> {
        let mut total_score = 0.0;
        let mut num_criteria = 0;

        // Size balance
        let sizes: Vec<usize> = subproblems.iter().map(|sp| sp.vertices.len()).collect();
        let avg_size = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        let size_variance = sizes.iter()
            .map(|&size| (size as f64 - avg_size).powi(2))
            .sum::<f64>() / sizes.len() as f64;
        let size_balance_score = 1.0 / (1.0 + size_variance / avg_size);
        total_score += size_balance_score;
        num_criteria += 1;

        // Cut quality (minimize boundary edges)
        let total_boundary_edges: usize = subproblems.iter()
            .map(|sp| sp.boundary_edges.len())
            .sum();
        let total_edges = original_problem.num_qubits * (original_problem.num_qubits - 1) / 2;
        let cut_quality_score = 1.0 - (total_boundary_edges as f64 / total_edges as f64);
        total_score += cut_quality_score;
        num_criteria += 1;

        // Coverage (all vertices included)
        let covered_vertices: HashSet<usize> = subproblems.iter()
            .flat_map(|sp| sp.vertices.iter())
            .cloned()
            .collect();
        let coverage_score = covered_vertices.len() as f64 / original_problem.num_qubits as f64;
        total_score += coverage_score;
        num_criteria += 1;

        Ok(total_score / num_criteria as f64)
    }

    /// Update knowledge base with new experience
    fn update_knowledge_base(
        &mut self,
        problem: &IsingModel,
        strategy: &DecompositionStrategy,
        quality_score: f64,
    ) -> Result<(), String> {
        // Update strategy performance history
        let performance_record = PerformanceRecord {
            timestamp: Instant::now(),
            problem_id: format!("problem_{}", problem.num_qubits),
            strategy_used: strategy.clone(),
            metrics: {
                let mut metrics = HashMap::new();
                metrics.insert(EvaluationMetric::SolutionQuality, quality_score);
                metrics
            },
            overall_score: quality_score,
        };

        self.strategy_learner.performance_history
            .entry(format!("{:?}", strategy))
            .or_insert_with(Vec::new)
            .push(performance_record.clone());

        self.performance_evaluator.performance_history.push(performance_record);

        // Update learning statistics
        self.strategy_learner.learning_stats.total_queries += 1;
        if quality_score > self.config.performance_threshold {
            self.strategy_learner.learning_stats.successful_predictions += 1;
        }

        // Update success rates
        let strategy_key = format!("{:?}", strategy);
        let history = &self.strategy_learner.performance_history[&strategy_key];
        let success_count = history.iter()
            .filter(|record| record.overall_score > self.config.performance_threshold)
            .count();
        let success_rate = success_count as f64 / history.len() as f64;
        
        self.knowledge_base.strategy_database.success_rates.insert(strategy.clone(), success_rate);

        Ok(())
    }
}

// Implementation of supporting types and traits

impl DecompositionStrategyLearner {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            selection_model: StrategySelectionModel::new(),
            performance_history: HashMap::new(),
            query_selector: QuerySelector::new(),
            transfer_learning: TransferLearningManager::new(),
            learning_stats: LearningStatistics::new(),
        })
    }

    pub fn recommend_strategy(
        &self,
        problem: &IsingModel,
        analysis: &ProblemAnalysis,
    ) -> Result<DecompositionStrategy, String> {
        // Simplified strategy recommendation based on problem size
        if problem.num_qubits < 10 {
            Ok(DecompositionStrategy::NoDecomposition)
        } else if problem.num_qubits < 50 {
            Ok(DecompositionStrategy::GraphPartitioning)
        } else {
            Ok(DecompositionStrategy::CommunityDetection)
        }
    }
}

impl StrategySelectionModel {
    pub fn new() -> Self {
        Self {
            model_type: ModelType::Linear,
            feature_weights: Array1::ones(20),
            strategy_preferences: HashMap::new(),
            uncertainty_estimates: HashMap::new(),
            model_parameters: ModelParameters::default(),
        }
    }

    pub fn get_uncertainty(&self, features: &Array1<f64>) -> Result<f64, String> {
        // Simplified uncertainty calculation
        let feature_sum = features.sum();
        Ok(1.0 / (1.0 + feature_sum.abs()))
    }

    pub fn get_strategy_uncertainty(
        &self,
        strategy: &DecompositionStrategy,
        features: &Array1<f64>,
    ) -> Result<f64, String> {
        let base_uncertainty = self.get_uncertainty(features)?;
        let strategy_key = format!("{:?}", strategy);
        
        if let Some(&stored_uncertainty) = self.uncertainty_estimates.get(&strategy_key) {
            Ok((base_uncertainty + stored_uncertainty) / 2.0)
        } else {
            Ok(base_uncertainty)
        }
    }
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            regularization: RegularizationParameters {
                l1_weight: 0.01,
                l2_weight: 0.01,
                dropout_rate: 0.1,
                early_stopping_patience: 10,
            },
            training_config: ModelTrainingConfig {
                num_epochs: 100,
                batch_size: 32,
                learning_rate: 0.001,
                validation_split: 0.2,
            },
        }
    }
}

impl QuerySelector {
    pub fn new() -> Self {
        Self {
            query_strategy: QueryStrategy::UncertaintySampling,
            uncertainty_threshold: 0.5,
            diversity_constraint: DiversityConstraint {
                min_distance: 0.1,
                diversity_metric: DiversityMetric::Euclidean,
                max_similarity: 0.8,
            },
            query_history: Vec::new(),
        }
    }
}

impl TransferLearningManager {
    pub fn new() -> Self {
        Self {
            source_models: Vec::new(),
            adaptation_strategy: DomainAdaptationStrategy::FineTuning,
            transfer_weights: Array1::ones(5),
            transfer_stats: TransferStatistics {
                successful_transfers: 0,
                failed_transfers: 0,
                avg_transfer_benefit: 0.0,
                transfer_time_overhead: Duration::from_secs(0),
            },
        }
    }
}

impl LearningStatistics {
    pub fn new() -> Self {
        Self {
            total_queries: 0,
            successful_predictions: 0,
            avg_prediction_accuracy: 0.0,
            learning_curve: Vec::new(),
            exploration_exploitation_ratio: 0.5,
        }
    }
}

impl ProblemAnalyzer {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            graph_analyzer: GraphAnalyzer::new(),
            structure_detector: StructureDetector::new(),
            complexity_estimator: ComplexityEstimator::new(),
            decomposability_scorer: DecomposabilityScorer::new(),
        })
    }
}

impl GraphAnalyzer {
    pub fn new() -> Self {
        Self {
            metrics_calculator: GraphMetricsCalculator::new(),
            community_detector: CommunityDetector::new(),
            critical_path_analyzer: CriticalPathAnalyzer::new(),
            bottleneck_detector: BottleneckDetector::new(),
        }
    }

    pub fn calculate_metrics(&mut self, problem: &IsingModel) -> Result<GraphMetrics, String> {
        let problem_key = format!("problem_{}", problem.num_qubits);
        
        if let Some(cached_metrics) = self.metrics_calculator.cached_metrics.get(&problem_key) {
            return Ok(cached_metrics.clone());
        }

        // Calculate metrics
        let num_vertices = problem.num_qubits;
        let mut num_edges = 0;
        
        for i in 0..problem.num_qubits {
            for j in (i+1)..problem.num_qubits {
                if problem.get_coupling(i, j).unwrap_or(0.0).abs() > 1e-10 {
                    num_edges += 1;
                }
            }
        }

        let max_edges = num_vertices * (num_vertices - 1) / 2;
        let density = if max_edges > 0 { num_edges as f64 / max_edges as f64 } else { 0.0 };

        let metrics = GraphMetrics {
            num_vertices,
            num_edges,
            density,
            clustering_coefficient: 0.0, // Would compute actual clustering coefficient
            avg_path_length: 0.0, // Would compute actual average path length
            modularity: 0.0, // Would compute actual modularity
            spectral_gap: 0.1, // Simplified estimate
            treewidth_estimate: (num_vertices as f64).sqrt() as usize, // Rough estimate
        };

        self.metrics_calculator.cached_metrics.insert(problem_key, metrics.clone());
        Ok(metrics)
    }

    pub fn detect_communities(&mut self, problem: &IsingModel) -> Result<Vec<DetectedCommunity>, String> {
        // Simplified community detection - in practice would use sophisticated algorithms
        let n = problem.num_qubits;
        let community_size = (n as f64).sqrt() as usize;
        let mut communities = Vec::new();

        for i in (0..n).step_by(community_size) {
            let end = (i + community_size).min(n);
            let vertices: Vec<usize> = (i..end).collect();
            
            if vertices.len() >= 2 {
                communities.push(DetectedCommunity {
                    id: communities.len(),
                    vertices,
                    modularity: 0.5, // Simplified
                    internal_density: 0.7,
                    external_density: 0.2,
                });
            }
        }

        Ok(communities)
    }
}

impl GraphMetricsCalculator {
    pub fn new() -> Self {
        Self {
            cached_metrics: HashMap::new(),
            computation_config: MetricComputationConfig {
                enable_expensive_metrics: false,
                enable_approximation: true,
                sampling_ratio: 0.1,
                computation_timeout: Duration::from_secs(60),
            },
        }
    }
}

impl CommunityDetector {
    pub fn new() -> Self {
        Self {
            algorithm: CommunityDetectionAlgorithm::Louvain,
            resolution: 1.0,
            min_community_size: 2,
            max_community_size: 100,
        }
    }
}

impl CriticalPathAnalyzer {
    pub fn new() -> Self {
        Self {
            algorithm: PathFindingAlgorithm::Dijkstra,
            weight_method: WeightCalculationMethod::CouplingStrength,
            path_cache: HashMap::new(),
        }
    }
}

impl BottleneckDetector {
    pub fn new() -> Self {
        Self {
            detection_threshold: 0.8,
            bottleneck_types: vec![
                BottleneckType::Vertex,
                BottleneckType::Edge,
                BottleneckType::CommunityBridge,
            ],
            bottlenecks_cache: HashMap::new(),
        }
    }
}

impl StructureDetector {
    pub fn new() -> Self {
        Self {
            pattern_matchers: Vec::new(),
            structure_templates: Vec::new(),
            confidence_threshold: 0.7,
            structures_cache: HashMap::new(),
        }
    }

    pub fn detect_structures(&mut self, problem: &IsingModel) -> Result<Vec<DetectedStructure>, String> {
        // Simplified structure detection
        let structures = vec![
            DetectedStructure {
                structure_type: StructureType::Random,
                vertices: (0..problem.num_qubits).collect(),
                confidence: 0.5,
                recommended_decomposition: DecompositionStrategy::GraphPartitioning,
            }
        ];
        
        Ok(structures)
    }
}

impl ComplexityEstimator {
    pub fn new() -> Self {
        Self {
            complexity_metrics: vec![
                ComplexityMetric::TimeComplexity,
                ComplexityMetric::SpaceComplexity,
            ],
            estimation_models: HashMap::new(),
            complexity_cache: HashMap::new(),
        }
    }

    pub fn estimate_complexity(&mut self, problem: &IsingModel) -> Result<ComplexityEstimate, String> {
        // Simplified complexity estimation
        let n = problem.num_qubits;
        let complexity_class = if n < 20 {
            ComplexityClass::P
        } else if n < 100 {
            ComplexityClass::NP
        } else {
            ComplexityClass::NPComplete
        };

        Ok(ComplexityEstimate {
            complexity_class,
            numeric_estimate: (n as f64).powf(2.0),
            confidence_interval: (n as f64, (n * n) as f64),
            estimation_method: "simplified".to_string(),
        })
    }
}

impl DecomposabilityScorer {
    pub fn new() -> Self {
        Self {
            scoring_functions: vec![
                ScoringFunction {
                    function_type: ScoringFunctionType::Modularity,
                    parameters: HashMap::new(),
                    weight: 0.4,
                },
                ScoringFunction {
                    function_type: ScoringFunctionType::CutBased,
                    parameters: HashMap::new(),
                    weight: 0.3,
                },
                ScoringFunction {
                    function_type: ScoringFunctionType::BalanceBased,
                    parameters: HashMap::new(),
                    weight: 0.3,
                },
            ],
            score_weights: Array1::from_vec(vec![0.4, 0.3, 0.3]),
            scoring_cache: HashMap::new(),
        }
    }

    pub fn score_decomposability(&mut self, problem: &IsingModel) -> Result<DecomposabilityScore, String> {
        // Simplified decomposability scoring
        let n = problem.num_qubits;
        let overall_score = if n < 10 {
            0.2 // Small problems don't benefit much from decomposition
        } else if n < 50 {
            0.7 // Medium problems benefit significantly
        } else {
            0.9 // Large problems benefit greatly
        };

        let mut component_scores = HashMap::new();
        component_scores.insert("modularity".to_string(), overall_score * 0.8);
        component_scores.insert("cut_quality".to_string(), overall_score * 0.9);
        component_scores.insert("balance".to_string(), overall_score * 0.7);

        let recommendation = DecompositionRecommendation {
            strategy: if n < 10 {
                DecompositionStrategy::NoDecomposition
            } else {
                DecompositionStrategy::GraphPartitioning
            },
            cut_points: Vec::new(),
            expected_benefit: overall_score,
            risk_assessment: RiskAssessment {
                risk_level: RiskLevel::Low,
                risk_factors: Vec::new(),
                mitigation_strategies: Vec::new(),
            },
        };

        Ok(DecomposabilityScore {
            overall_score,
            component_scores,
            recommendation,
            confidence: 0.8,
        })
    }
}

impl SubproblemGenerator {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            generation_strategies: Vec::new(),
            overlap_manager: OverlapManager::new(),
            size_controller: SizeController::new(),
            quality_validator: QualityValidator::new(),
        })
    }
}

impl OverlapManager {
    pub fn new() -> Self {
        Self {
            overlap_strategy: OverlapStrategy::NoOverlap,
            overlap_size: 0,
            resolution_method: OverlapResolutionMethod::Voting,
        }
    }
}

impl SizeController {
    pub fn new() -> Self {
        Self {
            size_constraints: SizeConstraints {
                min_size: 2,
                max_size: 100,
                target_size: 20,
                size_tolerance: 0.2,
            },
            balancing_strategy: SizeBalancingStrategy::Flexible,
            adaptive_sizing: true,
        }
    }
}

impl QualityValidator {
    pub fn new() -> Self {
        Self {
            validation_criteria: vec![
                ValidationCriterion {
                    criterion_type: ValidationCriterionType::ConnectivityPreservation,
                    weight: 0.3,
                    threshold: 0.8,
                },
                ValidationCriterion {
                    criterion_type: ValidationCriterionType::SizeBalance,
                    weight: 0.2,
                    threshold: 0.7,
                },
                ValidationCriterion {
                    criterion_type: ValidationCriterionType::CutQuality,
                    weight: 0.5,
                    threshold: 0.6,
                },
            ],
            validation_threshold: 0.7,
            strict_validation: false,
        }
    }
}

impl PerformanceEvaluator {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            evaluation_metrics: vec![
                EvaluationMetric::SolutionQuality,
                EvaluationMetric::ComputationTime,
                EvaluationMetric::MemoryUsage,
            ],
            baseline_comparisons: Vec::new(),
            performance_history: Vec::new(),
            evaluation_cache: HashMap::new(),
        })
    }
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

impl StrategyDatabase {
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

impl PatternLibrary {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            pattern_index: HashMap::new(),
            similarity_matrix: Array2::zeros((0, 0)),
        }
    }
}

impl PerformanceRepository {
    pub fn new() -> Self {
        Self {
            historical_data: Vec::new(),
            performance_trends: HashMap::new(),
            benchmark_results: Vec::new(),
        }
    }
}

impl RuleEngine {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            rule_priorities: HashMap::new(),
            application_history: Vec::new(),
        }
    }
}

impl Subproblem {
    pub fn from_full_problem(problem: &IsingModel) -> Self {
        Self {
            id: 0,
            model: problem.clone(),
            vertices: (0..problem.num_qubits).collect(),
            boundary_edges: Vec::new(),
            metadata: SubproblemMetadata::new(),
        }
    }
}

impl SubproblemMetadata {
    pub fn new() -> Self {
        Self {
            creation_time: Instant::now(),
            size: 0,
            complexity_estimate: 0.0,
            expected_solution_time: Duration::from_secs(1),
        }
    }
}

impl DecompositionMetadata {
    pub fn new() -> Self {
        Self {
            decomposition_time: Duration::from_secs(0),
            strategy_selection_time: Duration::from_secs(0),
            total_subproblems: 0,
            decomposition_depth: 1,
        }
    }
}

impl Default for ActiveLearningConfig {
    fn default() -> Self {
        Self {
            enable_online_learning: true,
            max_decomposition_depth: 3,
            min_subproblem_size: 2,
            max_subproblem_size: 100,
            learning_rate: 0.01,
            exploration_rate: 0.1,
            performance_threshold: 0.7,
            enable_transfer_learning: true,
            query_budget: 100,
            overlap_tolerance: 0.1,
        }
    }
}

/// Problem analysis result
#[derive(Debug, Clone)]
pub struct ProblemAnalysis {
    /// Graph metrics
    pub graph_metrics: GraphMetrics,
    /// Detected communities
    pub communities: Vec<DetectedCommunity>,
    /// Detected structures
    pub structures: Vec<DetectedStructure>,
    /// Complexity estimate
    pub complexity: ComplexityEstimate,
    /// Decomposability score
    pub decomposability: DecomposabilityScore,
    /// Extracted features
    pub problem_features: Array1<f64>,
}

/// Detected community
#[derive(Debug, Clone)]
pub struct DetectedCommunity {
    /// Community ID
    pub id: usize,
    /// Vertices in community
    pub vertices: Vec<usize>,
    /// Community modularity
    pub modularity: f64,
    /// Internal density
    pub internal_density: f64,
    /// External density
    pub external_density: f64,
}

/// Subproblem representation
#[derive(Debug, Clone)]
pub struct Subproblem {
    /// Subproblem ID
    pub id: usize,
    /// Ising model for subproblem
    pub model: IsingModel,
    /// Original vertex indices
    pub vertices: Vec<usize>,
    /// Boundary edges to other subproblems
    pub boundary_edges: Vec<BoundaryEdge>,
    /// Subproblem metadata
    pub metadata: SubproblemMetadata,
}

/// Boundary edge connecting subproblems
#[derive(Debug, Clone)]
pub struct BoundaryEdge {
    /// Vertex index within this subproblem
    pub internal_vertex: usize,
    /// Vertex index in original problem
    pub external_vertex: usize,
    /// Coupling strength
    pub coupling_strength: f64,
}

/// Subproblem metadata
#[derive(Debug, Clone)]
pub struct SubproblemMetadata {
    /// Creation timestamp
    pub creation_time: Instant,
    /// Subproblem size
    pub size: usize,
    /// Complexity estimate
    pub complexity_estimate: f64,
    /// Expected solution time
    pub expected_solution_time: Duration,
}

/// Decomposition result
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// Generated subproblems
    pub subproblems: Vec<Subproblem>,
    /// Strategy used
    pub strategy_used: DecompositionStrategy,
    /// Quality score
    pub quality_score: f64,
    /// Problem analysis
    pub analysis: ProblemAnalysis,
    /// Decomposition metadata
    pub metadata: DecompositionMetadata,
}

/// Decomposition metadata
#[derive(Debug, Clone)]
pub struct DecompositionMetadata {
    /// Time spent on decomposition
    pub decomposition_time: Duration,
    /// Time spent on strategy selection
    pub strategy_selection_time: Duration,
    /// Total number of subproblems
    pub total_subproblems: usize,
    /// Decomposition depth
    pub decomposition_depth: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_learning_decomposer_creation() {
        let config = ActiveLearningConfig::default();
        let decomposer = ActiveLearningDecomposer::new(config);
        assert!(decomposer.is_ok());
    }

    #[test]
    fn test_problem_analysis() {
        let config = ActiveLearningConfig::default();
        let mut decomposer = ActiveLearningDecomposer::new(config).unwrap();
        
        let mut problem = IsingModel::new(4);
        problem.set_bias(0, 1.0).unwrap();
        problem.set_coupling(0, 1, -0.5).unwrap();
        
        let analysis = decomposer.analyze_problem(&problem);
        assert!(analysis.is_ok());
        
        let analysis = analysis.unwrap();
        assert_eq!(analysis.graph_metrics.num_vertices, 4);
        assert_eq!(analysis.problem_features.len(), 20);
    }

    #[test]
    fn test_feature_extraction() {
        let config = ActiveLearningConfig::default();
        let decomposer = ActiveLearningDecomposer::new(config).unwrap();
        
        let mut problem = IsingModel::new(3);
        problem.set_bias(0, 1.0).unwrap();
        problem.set_coupling(0, 1, -0.5).unwrap();
        problem.set_coupling(1, 2, 0.3).unwrap();
        
        let features = decomposer.extract_problem_features(&problem).unwrap();
        assert_eq!(features.len(), 20);
        assert_eq!(features[0], 3.0); // num_qubits
        assert_eq!(features[1], 2.0); // num_couplings
    }

    #[test]
    fn test_problem_decomposition() {
        let config = ActiveLearningConfig::default();
        let mut decomposer = ActiveLearningDecomposer::new(config).unwrap();
        
        let mut problem = IsingModel::new(6);
        problem.set_bias(0, 1.0).unwrap();
        problem.set_coupling(0, 1, -0.5).unwrap();
        problem.set_coupling(2, 3, 0.3).unwrap();
        
        let result = decomposer.decompose_problem(&problem);
        assert!(result.is_ok());
        
        let decomposition = result.unwrap();
        assert!(!decomposition.subproblems.is_empty());
        assert!(decomposition.quality_score >= 0.0);
        assert!(decomposition.quality_score <= 1.0);
    }

    #[test]
    fn test_subproblem_creation() {
        let config = ActiveLearningConfig::default();
        let decomposer = ActiveLearningDecomposer::new(config).unwrap();
        
        let mut problem = IsingModel::new(4);
        problem.set_bias(0, 1.0).unwrap();
        problem.set_coupling(0, 1, -0.5).unwrap();
        problem.set_coupling(1, 2, 0.3).unwrap();
        
        let vertices = vec![0, 1];
        let subproblem = decomposer.create_subproblem(&problem, &vertices, 0).unwrap();
        
        assert_eq!(subproblem.id, 0);
        assert_eq!(subproblem.vertices, vec![0, 1]);
        assert_eq!(subproblem.model.num_qubits, 2);
        assert!(!subproblem.boundary_edges.is_empty()); // Should have boundary to vertex 2
    }

    #[test]
    fn test_decomposition_quality_validation() {
        let config = ActiveLearningConfig::default();
        let decomposer = ActiveLearningDecomposer::new(config).unwrap();
        
        let problem = IsingModel::new(6);
        
        // Create test subproblems
        let subproblem1 = Subproblem {
            id: 0,
            model: IsingModel::new(3),
            vertices: vec![0, 1, 2],
            boundary_edges: Vec::new(),
            metadata: SubproblemMetadata::new(),
        };
        
        let subproblem2 = Subproblem {
            id: 1,
            model: IsingModel::new(3),
            vertices: vec![3, 4, 5],
            boundary_edges: Vec::new(),
            metadata: SubproblemMetadata::new(),
        };
        
        let subproblems = vec![subproblem1, subproblem2];
        let quality = decomposer.validate_decomposition_quality(&subproblems, &problem).unwrap();
        
        assert!(quality >= 0.0);
        assert!(quality <= 1.0);
    }

    #[test]
    fn test_strategy_selection() {
        let mut learner = DecompositionStrategyLearner::new().unwrap();
        let problem = IsingModel::new(10);
        let analysis = ProblemAnalysis {
            graph_metrics: GraphMetrics {
                num_vertices: 10,
                num_edges: 15,
                density: 0.3,
                clustering_coefficient: 0.5,
                avg_path_length: 2.5,
                modularity: 0.4,
                spectral_gap: 0.2,
                treewidth_estimate: 3,
            },
            communities: Vec::new(),
            structures: Vec::new(),
            complexity: ComplexityEstimate {
                complexity_class: ComplexityClass::NP,
                numeric_estimate: 100.0,
                confidence_interval: (50.0, 200.0),
                estimation_method: "test".to_string(),
            },
            decomposability: DecomposabilityScore {
                overall_score: 0.8,
                component_scores: HashMap::new(),
                recommendation: DecompositionRecommendation {
                    strategy: DecompositionStrategy::GraphPartitioning,
                    cut_points: Vec::new(),
                    expected_benefit: 0.8,
                    risk_assessment: RiskAssessment {
                        risk_level: RiskLevel::Low,
                        risk_factors: Vec::new(),
                        mitigation_strategies: Vec::new(),
                    },
                },
                confidence: 0.9,
            },
            problem_features: Array1::ones(20),
        };
        
        let strategy = learner.recommend_strategy(&problem, &analysis).unwrap();
        // For a 10-qubit problem, should recommend GraphPartitioning
        assert_eq!(strategy, DecompositionStrategy::GraphPartitioning);
    }
}