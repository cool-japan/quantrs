//! Error Mitigation Strategies and Configurations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Error mitigation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMitigationConfig {
    /// Enable zero noise extrapolation
    pub enable_zne: bool,
    /// Enable symmetry verification
    pub enable_symmetry_verification: bool,
    /// Enable readout correction
    pub enable_readout_correction: bool,
    /// Enable dynamical decoupling
    pub enable_dynamical_decoupling: bool,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<MitigationStrategy>,
    /// ZNE configuration
    pub zne_config: ZNEConfig,
    /// Enable error mitigation
    pub enable_mitigation: bool,
    /// Mitigation strategies (legacy)
    pub strategies: Vec<MitigationStrategy>,
    /// Zero noise extrapolation
    pub zne: ZNEConfig,
    /// Readout mitigation
    pub readout_mitigation: ReadoutMitigationConfig,
    /// Gate mitigation
    pub gate_mitigation: GateMitigationConfig,
    /// Symmetry verification
    pub symmetry_verification: SymmetryVerificationConfig,
    /// Virtual distillation
    pub virtual_distillation: VirtualDistillationConfig,
}

/// Mitigation strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MitigationStrategy {
    ZeroNoiseExtrapolation,
    SymmetryVerification,
    ReadoutErrorMitigation,
    ReadoutMitigation,
    GateMitigation,
    VirtualDistillation,
    ProbabilisticErrorCancellation,
    CliffordDeRandomization,
}

/// Zero noise extrapolation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZNEConfig {
    /// Noise factors for scaling
    pub noise_factors: Vec<f64>,
    /// Extrapolation method
    pub extrapolation_method: ExtrapolationMethod,
    /// Circuit folding method
    pub circuit_folding: CircuitFoldingMethod,
    /// Enable ZNE
    pub enable_zne: bool,
    /// Noise scaling factors
    pub noise_scaling_factors: Vec<f64>,
    /// Folding configuration
    pub folding: FoldingConfig,
    /// Richardson extrapolation
    pub richardson: RichardsonConfig,
}

/// Circuit folding methods for ZNE
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CircuitFoldingMethod {
    GlobalFolding,
    LocalFolding,
    UniformFolding,
    RandomFolding,
}

/// Extrapolation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExtrapolationMethod {
    Linear,
    Polynomial,
    Exponential,
    Richardson,
    AdaptiveExtrapolation,
}

/// Folding configuration for ZNE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldingConfig {
    /// Folding type
    pub folding_type: FoldingType,
    /// Global folding
    pub global_folding: bool,
    /// Local folding configuration
    pub local_folding: LocalFoldingConfig,
    /// Gate-specific folding
    pub gate_specific: GateSpecificFoldingConfig,
}

/// Types of folding
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FoldingType {
    Global,
    Local,
    GateSpecific,
    RandomFolding,
    IdentityInsertion,
}

/// Local folding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalFoldingConfig {
    /// Local folding regions
    pub regions: Vec<FoldingRegion>,
    /// Region selection strategy
    pub selection_strategy: RegionSelectionStrategy,
    /// Overlap handling
    pub overlap_handling: OverlapHandling,
}

/// Folding region definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldingRegion {
    /// Start qubit
    pub start_qubit: usize,
    /// End qubit
    pub end_qubit: usize,
    /// Start time
    pub start_time: f64,
    /// End time
    pub end_time: f64,
}

/// Region selection strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegionSelectionStrategy {
    Random,
    HighErrorRate,
    CriticalPath,
    Uniform,
    Adaptive,
    Automatic,
}

/// Overlap handling strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OverlapHandling {
    Ignore,
    Merge,
    Prioritize,
    Exclude,
}

/// Gate-specific folding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateSpecificFoldingConfig {
    /// Gate folding rules
    pub folding_rules: HashMap<String, GateFoldingRule>,
    /// Priority ordering
    pub priority_ordering: Vec<String>,
    /// Error rate weighting
    pub error_rate_weighting: bool,
    /// Folding strategies (alias for folding_rules)
    pub folding_strategies: HashMap<String, GateFoldingRule>,
    /// Default folding strategy
    pub default_strategy: DefaultFoldingStrategy,
    /// Prioritized gates (alias for priority_ordering)
    pub prioritized_gates: Vec<String>,
}

/// Default folding strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DefaultFoldingStrategy {
    Identity,
    Inverse,
    Decomposition,
    Random,
    None,
}

/// Gate folding rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateFoldingRule {
    /// Folding factor
    pub folding_factor: f64,
    /// Folding probability
    pub probability: f64,
    /// Replacement strategy
    pub replacement: GateReplacementStrategy,
}

/// Gate replacement strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GateReplacementStrategy {
    Identity,
    Inverse,
    Decomposition,
    Equivalent,
}

/// Richardson extrapolation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RichardsonConfig {
    /// Enable Richardson extrapolation
    pub enable_richardson: bool,
    /// Order of extrapolation
    pub order: usize,
    /// Stability check
    pub stability_check: bool,
    /// Error estimation
    pub error_estimation: ErrorEstimationConfig,
}

/// Error estimation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEstimationConfig {
    /// Estimation method
    pub method: ErrorEstimationMethod,
    /// Bootstrap samples
    pub bootstrap_samples: usize,
    /// Confidence level
    pub confidence_level: f64,
}

/// Error estimation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ErrorEstimationMethod {
    Bootstrap,
    Jackknife,
    CrossValidation,
    AnalyticalEstimate,
}

/// Readout mitigation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadoutMitigationConfig {
    /// Enable readout mitigation
    pub enable_mitigation: bool,
    /// Mitigation methods
    pub methods: Vec<ReadoutMitigationMethod>,
    /// Calibration configuration
    pub calibration: ReadoutCalibrationConfig,
    /// Matrix inversion
    pub matrix_inversion: MatrixInversionConfig,
    /// Tensored mitigation
    pub tensored_mitigation: TensoredMitigationConfig,
}

/// Readout mitigation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReadoutMitigationMethod {
    CompleteMitigation,
    TensoredMitigation,
    LocalMitigation,
    ClusterMitigation,
    PartialMitigation,
}

/// Readout calibration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadoutCalibrationConfig {
    /// Calibration frequency
    pub frequency: CalibrationFrequency,
    /// Calibration states
    pub states: Vec<CalibrationState>,
    /// Quality metrics
    pub quality_metrics: Vec<QualityMetric>,
}

/// Calibration frequency options
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CalibrationFrequency {
    BeforeEachExperiment,
    Periodic(Duration),
    OnDemand,
    Adaptive,
}

/// Calibration state definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationState {
    /// State label
    pub label: String,
    /// Preparation circuit
    pub preparation: Vec<String>,
    /// Expected result
    pub expected_result: Vec<f64>,
}

/// Quality metrics for calibration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QualityMetric {
    Fidelity,
    CrossTalk,
    TemporalStability,
    SpatialUniformity,
}

/// Matrix inversion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixInversionConfig {
    /// Inversion method
    pub method: InversionMethod,
    /// Regularization
    pub regularization: RegularizationConfig,
    /// Numerical stability
    pub stability: NumericalStabilityConfig,
}

/// Matrix inversion methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InversionMethod {
    DirectInversion,
    PseudoInverse,
    IterativeInversion,
    RegularizedInversion,
}

/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// Regularization type
    pub regularization_type: RegularizationType,
    /// Regularization parameter
    pub parameter: f64,
    /// Adaptive regularization
    pub adaptive: bool,
}

/// Regularization types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegularizationType {
    L1,
    L2,
    ElasticNet,
    Tikhonov,
    None,
}

/// Numerical stability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalStabilityConfig {
    /// Condition number threshold
    pub condition_threshold: f64,
    /// Pivoting strategy
    pub pivoting: PivotingStrategy,
    /// Scaling
    pub scaling: bool,
}

/// Pivoting strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PivotingStrategy {
    Partial,
    Complete,
    Rook,
    None,
}

/// Tensored mitigation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensoredMitigationConfig {
    /// Mitigation groups
    pub groups: Vec<MitigationGroup>,
    /// Group formation strategy
    pub group_strategy: GroupFormationStrategy,
    /// Cross-talk handling
    pub crosstalk_handling: CrosstalkHandling,
}

/// Mitigation group definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationGroup {
    /// Group qubits
    pub qubits: Vec<usize>,
    /// Group label
    pub label: String,
    /// Independent mitigation
    pub independent: bool,
}

/// Group formation strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GroupFormationStrategy {
    Topology,
    ErrorRate,
    Connectivity,
    Manual,
    Adaptive,
}

/// Cross-talk handling methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CrosstalkHandling {
    Ignore,
    Model,
    Compensate,
    Avoid,
}

/// Gate mitigation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateMitigationConfig {
    /// Enable gate mitigation
    pub enable_mitigation: bool,
    /// Gate-specific configurations
    pub gate_configs: HashMap<String, GateSpecificConfig>,
    /// Twirling configuration
    pub twirling: TwirlingConfig,
    /// Randomized compiling
    pub randomized_compiling: RandomizedCompilingConfig,
}

/// Gate-specific mitigation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateSpecificConfig {
    /// Error model
    pub error_model: GateErrorModel,
    /// Mitigation method
    pub mitigation_method: GateMitigationMethod,
    /// Compensation parameters
    pub compensation: CompensationParameters,
}

/// Gate error models
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GateErrorModel {
    Depolarizing,
    Coherent,
    Incoherent,
    Composite,
    Custom(String),
}

/// Gate mitigation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GateMitigationMethod {
    PulseOptimization,
    CompositePulses,
    DynamicalDecoupling,
    ErrorCorrection,
    VirtualZ,
}

/// Compensation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompensationParameters {
    /// Phase corrections
    pub phase_corrections: HashMap<String, f64>,
    /// Amplitude corrections
    pub amplitude_corrections: HashMap<String, f64>,
    /// Timing corrections
    pub timing_corrections: HashMap<String, f64>,
}

/// Twirling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwirlingConfig {
    /// Enable twirling
    pub enable_twirling: bool,
    /// Twirling type
    pub twirling_type: TwirlingType,
    /// Twirling groups
    pub groups: Vec<TwirlingGroup>,
    /// Randomization strategy
    pub randomization: RandomizationStrategy,
}

/// Types of twirling
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TwirlingType {
    Pauli,
    Clifford,
    Gaussian,
    Custom(String),
}

/// Twirling group definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwirlingGroup {
    /// Group elements
    pub elements: Vec<String>,
    /// Group operation
    pub operation: String,
    /// Sampling distribution
    pub distribution: SamplingDistribution,
}

/// Sampling distributions for twirling
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SamplingDistribution {
    Uniform,
    Weighted,
    Adaptive,
    Custom(String),
}

/// Randomization strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RandomizationStrategy {
    FullRandomization,
    PartialRandomization,
    StructuredRandomization,
    AdaptiveRandomization,
}

/// Randomized compiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomizedCompilingConfig {
    /// Enable randomized compiling
    pub enable_rc: bool,
    /// Compilation strategies
    pub strategies: Vec<CompilationStrategy>,
    /// Gate replacement rules
    pub replacement_rules: HashMap<String, Vec<String>>,
    /// Randomization level
    pub randomization_level: RandomizationLevel,
}

/// Compilation strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompilationStrategy {
    GateReplacement,
    CircuitReordering,
    ParameterRandomization,
    HybridApproach,
}

/// Randomization levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RandomizationLevel {
    Low,
    Medium,
    High,
    Adaptive,
}

/// Symmetry verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryVerificationConfig {
    /// Enable symmetry verification
    pub enable_verification: bool,
    /// Symmetry types
    pub symmetry_types: Vec<SymmetryType>,
    /// Verification protocols
    pub protocols: Vec<VerificationProtocol>,
    /// Tolerance settings
    pub tolerance: ToleranceSettings,
}

/// Symmetry types for verification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SymmetryType {
    UnitarySymmetry,
    HamiltonianSymmetry,
    StateSymmetry,
    OperatorSymmetry,
}

/// Verification protocols
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VerificationProtocol {
    DirectVerification,
    RandomizedBenchmarking,
    ProcessTomography,
    ShadowEstimation,
}

/// Tolerance settings for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceSettings {
    /// Symmetry tolerance
    pub symmetry_tolerance: f64,
    /// Statistical tolerance
    pub statistical_tolerance: f64,
    /// Confidence level
    pub confidence_level: f64,
}

/// Virtual distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualDistillationConfig {
    /// Enable virtual distillation
    pub enable_distillation: bool,
    /// Distillation protocols
    pub protocols: Vec<DistillationProtocol>,
    /// Resource requirements
    pub resources: ResourceRequirements,
    /// Quality metrics
    pub quality_metrics: Vec<DistillationQualityMetric>,
}

/// Distillation protocols
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DistillationProtocol {
    Standard,
    Accelerated,
    ResourceOptimized,
    FaultTolerant,
}

/// Resource requirements for distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Auxiliary qubits
    pub auxiliary_qubits: usize,
    /// Measurement rounds
    pub measurement_rounds: usize,
    /// Classical processing
    pub classical_processing: ProcessingRequirements,
}

/// Processing requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingRequirements {
    /// Memory requirements
    pub memory_mb: usize,
    /// Computation time
    pub computation_time: Duration,
    /// Parallel processing
    pub parallel_processing: bool,
}

/// Quality metrics for distillation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DistillationQualityMetric {
    Fidelity,
    SuccessProbability,
    ResourceEfficiency,
    FaultTolerance,
}

/// Probabilistic error cancellation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PECConfig {
    /// Enable PEC
    pub enable_pec: bool,
    /// Quasi-probability decomposition
    pub quasi_probability: QuasiProbabilityConfig,
    /// Sampling strategy
    pub sampling: SamplingConfig,
    /// Optimization settings
    pub optimization: PECOptimizationConfig,
}

/// Quasi-probability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuasiProbabilityConfig {
    /// Decomposition method
    pub method: DecompositionMethod,
    /// Basis operations
    pub basis_operations: Vec<String>,
    /// Normalization
    pub normalization: bool,
}

/// Decomposition methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecompositionMethod {
    Optimal,
    Greedy,
    Heuristic,
    MachineLearning,
}

/// Sampling configuration for PEC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Sampling method
    pub method: SamplingMethod,
    /// Sample size
    pub sample_size: usize,
    /// Importance sampling
    pub importance_sampling: bool,
}

/// Sampling methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SamplingMethod {
    MonteCarlo,
    ImportanceSampling,
    StratifiedSampling,
    AdaptiveSampling,
}

/// PEC optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PECOptimizationConfig {
    /// Objective function
    pub objective: ObjectiveFunction,
    /// Constraints
    pub constraints: Vec<OptimizationConstraint>,
    /// Optimization algorithm
    pub algorithm: OptimizationAlgorithm,
}

/// Objective functions for PEC
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObjectiveFunction {
    MinimizeVariance,
    MinimizeBias,
    MaximizeEfficiency,
    BalancedObjective,
}

/// Optimization constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConstraint {
    /// Constraint type
    pub constraint_type: String,
    /// Constraint value
    pub value: f64,
    /// Tolerance
    pub tolerance: f64,
}

/// Optimization algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    GradientDescent,
    Newton,
    QuasiNewton,
    EvolutionaryAlgorithm,
    BayesianOptimization,
}

// Default implementations

impl Default for FoldingConfig {
    fn default() -> Self {
        Self {
            folding_type: FoldingType::Global,
            global_folding: true,
            local_folding: LocalFoldingConfig::default(),
            gate_specific: GateSpecificFoldingConfig::default(),
        }
    }
}

impl Default for LocalFoldingConfig {
    fn default() -> Self {
        Self {
            regions: vec![],
            selection_strategy: RegionSelectionStrategy::Automatic,
            overlap_handling: OverlapHandling::Merge,
        }
    }
}

impl Default for GateSpecificFoldingConfig {
    fn default() -> Self {
        Self {
            folding_rules: HashMap::new(),
            priority_ordering: vec![],
            error_rate_weighting: false,
            folding_strategies: HashMap::new(),
            default_strategy: DefaultFoldingStrategy::Identity,
            prioritized_gates: vec![],
        }
    }
}
