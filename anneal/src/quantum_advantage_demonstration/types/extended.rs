//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::functions::{
    CertificationCriterionEvaluator, ClassicalSolver, EffectSizeCalculator, EvidenceEvaluator,
    ProblemGenerator, StatisticalTest,
};
use super::core::*;
use crate::ising::{IsingModel, QuboModel};
use scirs2_core::random::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Bootstrap methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BootstrapMethod {
    /// Percentile bootstrap
    Percentile,
    /// Bias-corrected bootstrap
    BiasCorrected,
    /// Accelerated bootstrap
    BCa,
    /// Studentized bootstrap
    Studentized,
}
/// Performance record
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    /// Problem identifier
    pub problem_id: String,
    /// Device used
    pub device: QuantumDevice,
    /// Performance metrics
    pub metrics: QuantumPerformanceMetrics,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Timestamp
    pub timestamp: Instant,
}
/// Query metadata
#[derive(Debug, Clone)]
pub struct QueryMetadata {
    /// Query string
    pub query: String,
    /// Result count
    pub result_count: usize,
    /// Query timestamp
    pub timestamp: Instant,
}
/// Problem instance representation
#[derive(Debug, Clone)]
pub struct ProblemInstance {
    /// Instance identifier
    pub id: String,
    /// Problem size
    pub size: usize,
    /// Problem representation
    pub problem: ProblemRepresentation,
    /// Instance properties
    pub properties: InstanceProperties,
    /// Generation parameters
    pub generation_info: GenerationInfo,
}
/// Tuning methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuningMethod {
    /// Grid search
    GridSearch,
    /// Random search
    RandomSearch,
    /// Bayesian optimization
    BayesianOptimization,
    /// Evolutionary search
    EvolutionarySearch,
}
/// Device performance model
#[derive(Debug, Clone)]
pub struct DevicePerformanceModel {
    /// Device specifications
    pub device_specs: DeviceSpecifications,
    /// Performance characteristics
    pub performance_characteristics: DevicePerformanceCharacteristics,
    /// Calibration data
    pub calibration_data: DeviceCalibrationData,
    /// Usage statistics
    pub usage_statistics: DeviceUsageStatistics,
}
/// Error characteristics
#[derive(Debug, Clone)]
pub struct ErrorCharacteristics {
    /// Bit flip probability
    pub bit_flip_probability: f64,
    /// Phase flip probability
    pub phase_flip_probability: f64,
    /// Correlated error probability
    pub correlated_error_probability: f64,
    /// Readout error probability
    pub readout_error_probability: f64,
}
/// Solution representation
#[derive(Debug, Clone)]
pub struct Solution {
    /// Solution vector
    pub solution_vector: Vec<i32>,
    /// Objective value
    pub objective_value: f64,
    /// Solution quality
    pub quality: f64,
    /// Verification status
    pub verified: bool,
}
/// Problem representation formats
#[derive(Debug, Clone)]
pub enum ProblemRepresentation {
    /// Ising model
    Ising(IsingModel),
    /// QUBO model
    QUBO(QuboModel),
    /// Graph representation
    Graph(GraphProblem),
    /// Constraint satisfaction
    CSP(CSPProblem),
    /// Custom format
    Custom(String, Vec<u8>),
}
/// Operating parameters
#[derive(Debug, Clone)]
pub struct OperatingParameters {
    /// Operating temperature
    pub temperature: f64,
    /// Annealing time range
    pub anneal_time_range: (Duration, Duration),
    /// Programming time
    pub programming_time: Duration,
    /// Readout time
    pub readout_time: Duration,
}
/// Problem generation parameters
#[derive(Debug, Clone)]
pub struct GenerationParameters {
    /// Random seed for reproducibility
    pub random_seed: u64,
    /// Problem density parameters
    pub density_range: (f64, f64),
    /// Constraint tightness range
    pub constraint_tightness: (f64, f64),
    /// Hardness parameters
    pub hardness_params: HardnessParameters,
}
/// Criterion evaluation
#[derive(Debug, Clone)]
pub struct CriterionEvaluation {
    /// Criterion met
    pub criterion_met: bool,
    /// Evaluation score
    pub score: f64,
    /// Confidence in evaluation
    pub confidence: f64,
    /// Supporting evidence
    pub evidence: Vec<String>,
    /// Limitations identified
    pub limitations: Vec<String>,
}
/// Device calibration data
#[derive(Debug, Clone)]
pub struct DeviceCalibrationData {
    /// Last calibration time
    pub last_calibration: Instant,
    /// Calibration parameters
    pub calibration_parameters: HashMap<String, f64>,
    /// Calibration quality metrics
    pub quality_metrics: CalibrationQualityMetrics,
}
/// Aggregate statistics
#[derive(Debug, Clone)]
pub struct AggregateStatistics {
    /// Mean performance
    pub mean_performance: HashMap<String, f64>,
    /// Standard deviations
    pub standard_deviations: HashMap<String, f64>,
    /// Percentiles
    pub percentiles: HashMap<String, Vec<f64>>,
    /// Correlations
    pub correlations: CorrelationMatrix,
}
/// Benchmark metadata
#[derive(Debug, Clone)]
pub struct BenchmarkMetadata {
    /// Suite version
    pub version: String,
    /// Total benchmarks
    pub total_benchmarks: usize,
    /// Categories covered
    pub categories: Vec<ProblemCategory>,
    /// Size range
    pub size_range: (usize, usize),
    /// Creation timestamp
    pub creation_timestamp: Instant,
}
/// Advantage demonstration result
#[derive(Debug, Clone)]
pub struct AdvantageDemonstrationResult {
    /// Demonstration identifier
    pub id: String,
    /// Problem benchmarked
    pub problem_id: String,
    /// Quantum results
    pub quantum_results: HashMap<QuantumDevice, QuantumPerformanceMetrics>,
    /// Classical results
    pub classical_results: HashMap<ClassicalAlgorithm, PerformanceMetrics>,
    /// Statistical analysis
    pub statistical_analysis: StatisticalAnalysisResult,
    /// Advantage certification
    pub certification: AdvantageCertification,
    /// Metadata
    pub metadata: ResultMetadata,
}
/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Time to solution
    pub time_to_solution: Duration,
    /// Solution quality
    pub solution_quality: f64,
    /// Success rate
    pub success_rate: f64,
    /// Convergence rate
    pub convergence_rate: f64,
    /// Resource efficiency
    pub resource_efficiency: f64,
}
/// Certification levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CertificationLevel {
    /// No advantage demonstrated
    NoAdvantage,
    /// Weak evidence of advantage
    WeakEvidence,
    /// Moderate evidence of advantage
    ModerateEvidence,
    /// Strong evidence of advantage
    StrongEvidence,
    /// Definitive advantage
    DefinitiveAdvantage,
}
/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// CPU time used
    pub cpu_time: Duration,
    /// Memory usage (MB)
    pub memory_usage: f64,
    /// Energy consumption (Joules)
    pub energy_consumption: f64,
    /// Cost (monetary units)
    pub cost: f64,
}
/// CSP variable
#[derive(Debug, Clone)]
pub struct CSPVariable {
    /// Variable identifier
    pub id: usize,
    /// Variable domain
    pub domain: Vec<i32>,
    /// Variable type
    pub variable_type: CSPVariableType,
}
/// Validation strategies for tuning
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationStrategy {
    /// Cross-validation
    CrossValidation { folds: usize },
    /// Hold-out validation
    HoldOut { split_ratio: f64 },
    /// Bootstrap validation
    Bootstrap { samples: usize },
}
/// Scaling fit types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalingFitType {
    /// Polynomial fit
    Polynomial { degree: usize },
    /// Exponential fit
    Exponential,
    /// Power law fit
    PowerLaw,
    /// Logarithmic fit
    Logarithmic,
}
/// Problem categories for benchmarking
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProblemCategory {
    /// General optimization problems
    Optimization,
    /// Sampling problems
    Sampling,
    /// Constraint satisfaction problems
    ConstraintSatisfaction,
    /// Machine learning problems
    MachineLearning,
    /// Scientific computing problems
    ScientificComputing,
    /// Graph problems
    GraphProblems,
    /// Financial optimization
    FinancialOptimization,
}
/// Tuning configuration
#[derive(Debug, Clone)]
pub struct TuningConfig {
    /// Tuning method
    pub method: TuningMethod,
    /// Number of tuning iterations
    pub num_iterations: usize,
    /// Validation strategy
    pub validation_strategy: ValidationStrategy,
    /// Objective function
    pub objective_function: TuningObjective,
}
/// Classical performance record
#[derive(Debug, Clone)]
pub struct ClassicalPerformanceRecord {
    /// Algorithm
    pub algorithm: ClassicalAlgorithm,
    /// Problem identifier
    pub problem_id: String,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Timestamp
    pub timestamp: Instant,
}
/// Results database for storing and querying results
pub struct ResultsDatabase {
    /// Database configuration
    pub config: DatabaseConfig,
    /// Stored results
    pub results: HashMap<String, AdvantageDemonstrationResult>,
    /// Benchmark results
    pub benchmark_results: HashMap<String, BenchmarkResult>,
    /// Query engine
    pub query_engine: QueryEngine,
    /// Metadata index
    pub metadata_index: MetadataIndex,
}
impl ResultsDatabase {
    pub fn new() -> Self {
        Self {
            config: DatabaseConfig {
                enable_caching: true,
                max_cache_size: 10_000,
                enable_compression: true,
                backup_frequency: Duration::from_secs(3600),
            },
            results: HashMap::new(),
            benchmark_results: HashMap::new(),
            query_engine: QueryEngine {
                query_types: vec![QueryType::Filter, QueryType::Aggregate, QueryType::Compare],
                query_cache: HashMap::new(),
                indices: HashMap::new(),
            },
            metadata_index: MetadataIndex {
                category_index: HashMap::new(),
                device_index: HashMap::new(),
                algorithm_index: HashMap::new(),
                time_index: BTreeMap::new(),
            },
        }
    }
}
/// Energy landscape characteristics
#[derive(Debug, Clone)]
pub struct LandscapeCharacteristics {
    /// Number of local minima
    pub num_local_minima: usize,
    /// Barrier heights
    pub barrier_heights: Vec<f64>,
    /// Basin sizes
    pub basin_sizes: Vec<usize>,
    /// Ruggedness measures
    pub ruggedness: f64,
}
/// Evidence representation
#[derive(Debug, Clone)]
pub struct Evidence {
    /// Evidence type
    pub evidence_type: EvidenceType,
    /// Evidence data
    pub data: Vec<u8>,
    /// Evidence metadata
    pub metadata: HashMap<String, String>,
}
/// Quantum performance analyzer
pub struct QuantumPerformanceAnalyzer {
    /// Analyzer configuration
    pub config: QuantumAnalyzerConfig,
    /// Device performance models
    pub device_models: HashMap<QuantumDevice, DevicePerformanceModel>,
    /// Performance database
    pub performance_database: PerformanceDatabase,
    /// Error models
    pub error_models: HashMap<QuantumDevice, ErrorModel>,
}
impl QuantumPerformanceAnalyzer {
    pub fn new() -> Self {
        Self {
            config: QuantumAnalyzerConfig {
                enable_error_modeling: true,
                track_resource_usage: true,
                analyze_scaling: true,
                compare_devices: true,
            },
            device_models: HashMap::new(),
            performance_database: PerformanceDatabase {
                records: HashMap::new(),
                scaling_data: HashMap::new(),
                comparison_data: vec![],
            },
            error_models: HashMap::new(),
        }
    }
}
/// CSP constraint
#[derive(Debug, Clone)]
pub struct CSPConstraint {
    /// Constraint identifier
    pub id: usize,
    /// Variables involved
    pub variables: Vec<usize>,
    /// Constraint type
    pub constraint_type: CSPConstraintType,
    /// Constraint parameters
    pub parameters: Vec<f64>,
}
/// Bias levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BiasLevel {
    /// High bias
    High,
    /// Medium bias
    Medium,
    /// Low bias
    Low,
    /// Very low bias
    VeryLow,
}
/// Connectivity patterns for problem generation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectivityPattern {
    /// Random connectivity
    Random,
    /// Small-world networks
    SmallWorld,
    /// Scale-free networks
    ScaleFree,
    /// Grid connectivity
    Grid,
    /// Complete graphs
    Complete,
    /// Sparse connectivity
    Sparse,
}
/// Spectral properties of problems
#[derive(Debug, Clone)]
pub struct SpectralProperties {
    /// Eigenvalue spectrum
    pub eigenvalues: Vec<f64>,
    /// Spectral gap
    pub spectral_gap: f64,
    /// Condition number
    pub condition_number: f64,
}
/// Instance result
#[derive(Debug, Clone)]
pub struct InstanceResult {
    /// Instance identifier
    pub instance_id: String,
    /// Quantum results
    pub quantum_results: HashMap<QuantumDevice, QuantumPerformanceMetrics>,
    /// Classical results
    pub classical_results: HashMap<ClassicalAlgorithm, PerformanceMetrics>,
    /// Advantage metrics
    pub advantage_metrics: HashMap<AdvantageMetric, f64>,
}
/// Tuning objectives
#[derive(Debug, Clone, PartialEq)]
pub enum TuningObjective {
    /// Minimize time to solution
    MinimizeTime,
    /// Maximize solution quality
    MaximizeQuality,
    /// Multi-objective
    MultiObjective { weights: Vec<f64> },
}
/// Execution environment
#[derive(Debug, Clone)]
pub struct ExecutionEnvironment {
    /// Hardware specifications
    pub hardware_specs: HashMap<String, String>,
    /// Software versions
    pub software_versions: HashMap<String, String>,
    /// Environmental conditions
    pub environmental_conditions: HashMap<String, f64>,
}
/// Certification record
#[derive(Debug, Clone)]
pub struct CertificationRecord {
    /// Record identifier
    pub id: String,
    /// Result certified
    pub result_id: String,
    /// Certification outcome
    pub certification: AdvantageCertification,
    /// Certification process
    pub process: CertificationProcess,
    /// Timestamp
    pub timestamp: Instant,
}
/// Parameter types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParameterType {
    /// Continuous parameter
    Continuous,
    /// Integer parameter
    Integer,
    /// Categorical parameter
    Categorical,
    /// Boolean parameter
    Boolean,
}
/// Classical optimizer configuration
#[derive(Debug, Clone)]
pub struct ClassicalOptimizerConfig {
    /// Algorithms to include
    pub enabled_algorithms: Vec<ClassicalAlgorithm>,
    /// Time limit per algorithm
    pub time_limit_per_algorithm: Duration,
    /// Enable algorithm tuning
    pub enable_tuning: bool,
    /// Tuning budget
    pub tuning_budget: Duration,
    /// Parallel execution
    pub parallel_execution: bool,
}
/// Individual benchmark specification
#[derive(Debug, Clone)]
pub struct Benchmark {
    /// Benchmark identifier
    pub id: String,
    /// Benchmark name
    pub name: String,
    /// Problem category
    pub category: ProblemCategory,
    /// Problem instances
    pub instances: Vec<ProblemInstance>,
    /// Benchmark metadata
    pub metadata: BenchmarkInstanceMetadata,
    /// Expected difficulty
    pub expected_difficulty: DifficultyLevel,
    /// Known optimal solutions (if available)
    pub known_solutions: Option<Vec<Solution>>,
}
/// Evidence types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceType {
    /// Performance measurements
    PerformanceMeasurements,
    /// Statistical analysis
    StatisticalAnalysis,
    /// Scaling analysis
    ScalingAnalysis,
    /// Error analysis
    ErrorAnalysis,
    /// Cost analysis
    CostAnalysis,
}
/// Effect size interpretation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectSizeInterpretation {
    /// Negligible effect
    Negligible,
    /// Small effect
    Small,
    /// Medium effect
    Medium,
    /// Large effect
    Large,
    /// Very large effect
    VeryLarge,
}
/// Problem structure features
#[derive(Debug, Clone)]
pub struct StructureFeatures {
    /// Symmetry measures
    pub symmetry_measures: Vec<f64>,
    /// Modularity
    pub modularity: f64,
    /// Spectral properties
    pub spectral_properties: SpectralProperties,
    /// Frustration indicators
    pub frustration_indicators: FrustrationIndicators,
}
/// Frustration indicators
#[derive(Debug, Clone)]
pub struct FrustrationIndicators {
    /// Frustration index
    pub frustration_index: f64,
    /// Conflict density
    pub conflict_density: f64,
    /// Backbone fraction
    pub backbone_fraction: f64,
}
/// Cost statistics
#[derive(Debug, Clone)]
pub struct CostStatistics {
    /// Average cost per problem
    pub average_cost_per_problem: f64,
    /// Cost per qubit usage
    pub cost_per_qubit: f64,
    /// Total cost
    pub total_cost: f64,
    /// Cost efficiency trend
    pub cost_efficiency_trend: Vec<f64>,
}
/// Device performance characteristics
#[derive(Debug, Clone)]
pub struct DevicePerformanceCharacteristics {
    /// Typical solution time
    pub typical_solution_time: Duration,
    /// Success probability vs problem size
    pub success_probability_curve: Vec<(usize, f64)>,
    /// Time-to-solution scaling
    pub time_to_solution_scaling: ScalingCharacteristics,
    /// Energy efficiency
    pub energy_efficiency: f64,
}
/// Query result
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Result data
    pub data: Vec<HashMap<String, String>>,
    /// Query metadata
    pub metadata: QueryMetadata,
    /// Execution time
    pub execution_time: Duration,
}
/// Query engine for database queries
#[derive(Debug)]
pub struct QueryEngine {
    /// Supported query types
    pub query_types: Vec<QueryType>,
    /// Query cache
    pub query_cache: HashMap<String, QueryResult>,
    /// Index structures
    pub indices: HashMap<String, Index>,
}
/// Index types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexType {
    /// Hash index
    Hash,
    /// B-tree index
    BTree,
    /// Full-text index
    FullText,
}
/// Quantum advantage metrics
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdvantageMetric {
    /// Time to solution
    TimeToSolution,
    /// Solution quality
    SolutionQuality,
    /// Energy consumption
    EnergyConsumption,
    /// Cost efficiency
    CostEfficiency,
    /// Scalability
    Scalability,
    /// Success probability
    SuccessProbability,
    /// Convergence rate
    ConvergenceRate,
}
/// Classical algorithms for baseline comparison
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClassicalAlgorithm {
    /// Simulated annealing
    SimulatedAnnealing,
    /// Tabu search
    TabuSearch,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Particle swarm optimization
    ParticleSwarmOptimization,
    /// Branch and bound
    BranchAndBound,
    /// Variable neighborhood search
    VariableNeighborhoodSearch,
    /// GRASP (Greedy Randomized Adaptive Search)
    GRASP,
    /// Ant colony optimization
    AntColonyOptimization,
    /// Large neighborhood search
    LargeNeighborhoodSearch,
}
/// Multiple comparison correction methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorrectionMethod {
    /// Bonferroni correction
    Bonferroni,
    /// Benjamini-Hochberg correction
    BenjaminiHochberg,
    /// Holm-Bonferroni correction
    HolmBonferroni,
    /// False discovery rate control
    FDR,
}
/// Completeness levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletenessLevel {
    /// Incomplete
    Incomplete,
    /// Partially complete
    PartiallyComplete,
    /// Mostly complete
    MostlyComplete,
    /// Complete
    Complete,
}
/// Parameter definition
#[derive(Debug, Clone)]
pub struct ParameterDefinition {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub parameter_type: ParameterType,
    /// Valid range
    pub range: ParameterRange,
    /// Default value
    pub default_value: ParameterValue,
}
/// Instance properties for characterization
#[derive(Debug, Clone)]
pub struct InstanceProperties {
    /// Connectivity density
    pub connectivity_density: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Constraint tightness
    pub constraint_tightness: f64,
    /// Estimated hardness
    pub estimated_hardness: f64,
    /// Problem structure features
    pub structure_features: StructureFeatures,
}
/// Database index
#[derive(Debug, Clone)]
pub struct Index {
    /// Index name
    pub name: String,
    /// Index type
    pub index_type: IndexType,
    /// Index data
    pub data: HashMap<String, Vec<String>>,
}
/// Error rates
#[derive(Debug, Clone)]
pub struct ErrorRates {
    /// Single qubit error rate
    pub single_qubit_error_rate: f64,
    /// Two qubit error rate
    pub two_qubit_error_rate: f64,
    /// Readout error rate
    pub readout_error_rate: f64,
    /// Coherence time
    pub coherence_time: Duration,
}
/// Classical baseline optimizer
pub struct ClassicalBaselineOptimizer {
    /// Optimizer configuration
    pub config: ClassicalOptimizerConfig,
    /// Available algorithms
    pub algorithms: HashMap<ClassicalAlgorithm, Box<dyn ClassicalSolver>>,
    /// Performance history
    pub performance_history: VecDeque<ClassicalPerformanceRecord>,
    /// Algorithm tuning system
    pub tuning_system: AlgorithmTuningSystem,
}
impl ClassicalBaselineOptimizer {
    pub fn new() -> Self {
        Self {
            config: ClassicalOptimizerConfig {
                enabled_algorithms: vec![
                    ClassicalAlgorithm::SimulatedAnnealing,
                    ClassicalAlgorithm::TabuSearch,
                    ClassicalAlgorithm::GeneticAlgorithm,
                ],
                time_limit_per_algorithm: Duration::from_secs(300),
                enable_tuning: true,
                tuning_budget: Duration::from_secs(3600),
                parallel_execution: true,
            },
            algorithms: HashMap::new(),
            performance_history: VecDeque::new(),
            tuning_system: AlgorithmTuningSystem {
                config: TuningConfig {
                    method: TuningMethod::BayesianOptimization,
                    num_iterations: 100,
                    validation_strategy: ValidationStrategy::CrossValidation { folds: 5 },
                    objective_function: TuningObjective::MultiObjective {
                        weights: vec![0.5, 0.5],
                    },
                },
                parameter_spaces: HashMap::new(),
                tuning_history: HashMap::new(),
                best_parameters: HashMap::new(),
            },
        }
    }
}
/// Constraint types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintType {
    /// Linear constraint
    Linear,
    /// Nonlinear constraint
    Nonlinear,
    /// Conditional constraint
    Conditional,
}
/// Constraint satisfaction problem
#[derive(Debug, Clone)]
pub struct CSPProblem {
    /// Variables
    pub variables: Vec<CSPVariable>,
    /// Constraints
    pub constraints: Vec<CSPConstraint>,
    /// Domain sizes
    pub domain_sizes: Vec<usize>,
}
/// Correlation matrix
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    /// Variable names
    pub variables: Vec<String>,
    /// Correlation coefficients
    pub correlations: Vec<Vec<f64>>,
    /// p-values
    pub p_values: Vec<Vec<f64>>,
}
/// Quantum advantage demonstration configuration
#[derive(Debug, Clone)]
pub struct AdvantageConfig {
    /// Statistical confidence level
    pub confidence_level: f64,
    /// Number of repetitions for statistical significance
    pub num_repetitions: usize,
    /// Problem size range for scaling analysis
    pub problem_size_range: (usize, usize),
    /// Time limit per optimization
    pub time_limit: Duration,
    /// Classical algorithms to compare against
    pub classical_algorithms: Vec<ClassicalAlgorithm>,
    /// Quantum devices to test
    pub quantum_devices: Vec<QuantumDevice>,
    /// Advantage metrics to evaluate
    pub advantage_metrics: Vec<AdvantageMetric>,
    /// Problem categories to benchmark
    pub problem_categories: Vec<ProblemCategory>,
}
/// Quantum devices for testing
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QuantumDevice {
    /// D-Wave Advantage system
    DWaveAdvantage,
    /// AWS Braket quantum devices
    AWSBraket,
    /// Local quantum simulator
    Simulator,
    /// IBM Quantum
    IBMQuantum,
    /// `IonQ`
    IonQ,
    /// Rigetti
    Rigetti,
}
/// Certification configuration
#[derive(Debug, Clone)]
pub struct CertificationConfig {
    /// Required confidence level
    pub required_confidence_level: f64,
    /// Minimum effect size
    pub minimum_effect_size: f64,
    /// Required robustness level
    pub required_robustness_level: f64,
    /// Enable peer review
    pub enable_peer_review: bool,
}
