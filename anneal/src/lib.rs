//! Quantum annealing support for the QuantRS2 framework.
//!
//! This crate provides types and functions for quantum annealing,
//! including Ising model representation, QUBO problem formulation,
//! simulated quantum annealing, and cloud quantum annealing services.
//!
//! # Features
//!
//! - Ising model representation with biases and couplings
//! - QUBO problem formulation with constraints
//! - Simulated quantum annealing using path integral Monte Carlo
//! - Classical simulated annealing using Metropolis algorithm
//! - D-Wave API client for connecting to quantum annealing hardware
//! - AWS Braket client for accessing Amazon's quantum computing services
//!
//! # Example
//!
//! ```rust
//! use quantrs2_anneal::{
//!     ising::IsingModel,
//!     simulator::{ClassicalAnnealingSimulator, AnnealingParams}
//! };
//!
//! // Create a simple 3-qubit Ising model
//! let mut model = IsingModel::new(3);
//! model.set_bias(0, 1.0).unwrap();
//! model.set_coupling(0, 1, -1.0).unwrap();
//!
//! // Configure annealing parameters
//! let mut params = AnnealingParams::new();
//! params.num_sweeps = 1000;
//! params.num_repetitions = 10;
//!
//! // Create an annealing simulator and solve the model
//! let simulator = ClassicalAnnealingSimulator::new(params).unwrap();
//! let result = simulator.solve(&model).unwrap();
//!
//! println!("Best energy: {}", result.best_energy);
//! println!("Best solution: {:?}", result.best_spins);
//! ```

// Export modules
pub mod applications;
pub mod braket;
pub mod chain_break;
pub mod coherent_ising_machine;
pub mod compression;
pub mod continuous_variable;
pub mod csp_compiler;
pub mod dsl;
pub mod dwave;
pub mod embedding;
pub mod flux_bias;
#[cfg(feature = "fujitsu")]
pub mod fujitsu;
pub mod hardware_compilation;
pub mod hobo;
pub mod hybrid_solvers;
pub mod ising;
pub mod layout_embedding;
pub mod multi_objective;
pub mod non_stoquastic;
pub mod partitioning;
pub mod penalty_optimization;
pub mod photonic_annealing;
pub mod population_annealing;
pub mod problem_schedules;
pub mod qaoa;
pub mod qaoa_circuit_bridge;
pub mod qubo;
pub mod qubo_decomposition;
pub mod quantum_boltzmann_machine;
pub mod quantum_machine_learning;
pub mod quantum_walk;
pub mod reverse_annealing;
pub mod simulator;
pub mod solution_clustering;
pub mod variational_quantum_annealing;
pub mod visualization;

// Re-export key types for convenience
pub use applications::{
    ApplicationError, ApplicationResult, OptimizationProblem, IndustrySolution, Benchmarkable,
    ProblemCategory, IndustryConstraint, IndustryObjective, create_benchmark_suite,
    generate_performance_report, validate_constraints,
    finance, logistics, energy, manufacturing, healthcare, telecommunications, transportation,
};
pub use braket::{
    is_available as is_braket_available, BraketClient, BraketError, BraketResult,
    DeviceType, DeviceStatus, TaskStatus, BraketDevice, DeviceSelector,
    AdvancedAnnealingParams, TaskResult, TaskMetrics, BatchTaskResult, CostTracker,
};
pub use chain_break::{
    ChainBreakResolver, ChainBreakStats, ChainStrengthOptimizer, HardwareSolution,
    LogicalProblem, ResolutionMethod, ResolvedSolution,
};
pub use coherent_ising_machine::{
    CoherentIsingMachine, CimConfig, CimResults, CimError, CimResult,
    OpticalParametricOscillator, OpticalCoupling, Complex, PumpSchedule, NetworkTopology,
    NoiseConfig, MeasurementConfig, ConvergenceConfig, OpticalStatistics, CimPerformanceMetrics,
    create_standard_cim_config, create_low_noise_cim_config, create_realistic_cim_config,
};
pub use compression::{
    BlockDetector, CompressedQubo, CompressionStats, CooCompressor, ReductionMapping,
    VariableReducer,
};
pub use continuous_variable::{
    ContinuousVariableAnnealer, ContinuousAnnealingConfig, ContinuousOptimizationProblem,
    ContinuousVariable, ContinuousConstraint, ContinuousSolution, ContinuousOptimizationStats,
    ContinuousVariableError, ContinuousVariableResult, create_quadratic_problem,
};
pub use csp_compiler::{
    CspProblem, CspVariable, CspConstraint, CspObjective, CspSolution, CspValue, Domain,
    ComparisonOp, CompilationParams, CspCompilationInfo, CspError, CspResult,
};
pub use dsl::{
    OptimizationModel, Variable, VariableVector, VariableType, Expression, BooleanExpression,
    Constraint, Objective, ObjectiveDirection, ModelSummary, DslError, DslResult, patterns,
};
pub use dwave::{
    is_available as is_dwave_available, DWaveClient, DWaveError, DWaveResult, ProblemParams,
    // Enhanced Leap types
    SolverType, SolverCategory, ProblemStatus, LeapSolverInfo, ProblemInfo, SolverSelector,
    EmbeddingConfig, ChainStrengthMethod, AdvancedProblemParams, HybridSolverParams,
    AnnealingSchedule, ProblemMetrics, BatchSubmissionResult,
};
pub use embedding::{Embedding, HardwareGraph, HardwareTopology, MinorMiner};
pub use flux_bias::{
    FluxBiasOptimizer, FluxBiasConfig, FluxBiasResult, CalibrationData, MLFluxBiasOptimizer,
};
#[cfg(feature = "fujitsu")]
pub use fujitsu::{
    FujitsuClient, FujitsuError, FujitsuResult, FujitsuAnnealingParams, FujitsuHardwareSpec,
    GuidanceConfig, is_available as is_fujitsu_available,
};
pub use hardware_compilation::{
    HardwareCompiler, CompilerConfig, CompilationTarget, CompilationResult, HardwareType,
    HardwareCharacteristics, OptimizationObjective, HardwareCompilationError, HardwareCompilationResult,
    EmbeddingInfo, HardwareMapping, PerformancePrediction, EmbeddingAlgorithm, QubitAllocationStrategy,
    CouplingUtilization, ParallelizationStrategy, ConnectivityPattern, TopologyType,
    create_chimera_target, create_ideal_target,
};
pub use hobo::{
    AuxiliaryVariable, ConstraintViolations, HigherOrderTerm, HoboAnalyzer, HoboProblem,
    HoboStats, QuboReduction, ReductionMethod, ReductionType,
};
pub use hybrid_solvers::{
    HybridQuantumClassicalSolver, HybridSolverConfig, HybridSolverResult,
    VariationalHybridSolver,
};
pub use ising::{IsingError, IsingModel, IsingResult, QuboModel};
pub use layout_embedding::{
    LayoutAwareEmbedder, LayoutConfig, LayoutStats, MultiLevelEmbedder,
};
pub use multi_objective::{
    MultiObjectiveOptimizer, MultiObjectiveConfig, MultiObjectiveResults, MultiObjectiveSolution,
    MultiObjectiveStats, ScalarizationMethod, QualityMetrics, MultiObjectiveError,
    MultiObjectiveResult, MultiObjectiveFunction,
};
pub use non_stoquastic::{
    NonStoquasticHamiltonian, HamiltonianType, ComplexCoupling, InteractionType,
    NonStoquasticQMCConfig, SignMitigationStrategy, NonStoquasticResults,
    QMCStatistics, ConvergenceInfo, NonStoquasticSimulator, QuantumState as NonStoquasticQuantumState,
    NonStoquasticError, NonStoquasticResult, is_hamiltonian_stoquastic, xy_to_ising_approximation,
    create_xy_chain, create_tfxy_model, create_frustrated_xy_triangle,
};
pub use partitioning::{
    BipartitionMethod, KernighanLinPartitioner, Partition, RecursiveBisectionPartitioner,
    SpectralPartitioner,
};
pub use penalty_optimization::{
    PenaltyOptimizer, PenaltyConfig, PenaltyStats, AdvancedPenaltyOptimizer,
    ConstraintPenaltyOptimizer, Constraint as PenaltyConstraint, ConstraintType,
};
pub use photonic_annealing::{
    PhotonicAnnealer, PhotonicAnnealingConfig, PhotonicAnnealingResults, PhotonicError,
    PhotonicResult, PhotonicArchitecture, ConnectivityType, MeasurementType,
    PhotonicComponent, PhotonicState, InitialStateType, PumpPowerSchedule,
    MeasurementStrategy, MeasurementOutcome, EvolutionHistory, PhotonicMetrics,
    create_coherent_state_config, create_squeezed_state_config, create_temporal_multiplexing_config,
    create_measurement_based_config, create_low_noise_config, create_realistic_config,
};
pub use population_annealing::{
    PopulationAnnealingSimulator, PopulationAnnealingConfig, PopulationAnnealingSolution,
    PopulationMember, EnergyStatistics, MpiConfig, PopulationAnnealingError,
};
pub use problem_schedules::{
    ProblemSpecificScheduler, ProblemType, ScheduleTemplate, AdaptiveScheduleOptimizer,
};
pub use qaoa::{
    QaoaOptimizer, QaoaConfig, QaoaResults, QaoaError, QaoaResult, QaoaVariant, MixerType as QaoaMixerType,
    ProblemEncoding, QaoaClassicalOptimizer, ParameterInitialization as QaoaParameterInitialization,
    QaoaCircuit, QaoaLayer, QuantumGate as QaoaQuantumGate, QuantumState as QaoaQuantumState, QaoaCircuitStats, QuantumStateStats,
    QaoaPerformanceMetrics, create_standard_qaoa_config, create_qaoa_plus_config,
    create_warm_start_qaoa_config, create_constrained_qaoa_config,
};
pub use qaoa_circuit_bridge::{
    QaoaCircuitBridge, BridgeError, BridgeResult, CircuitBridgeRepresentation, ParameterReference,
    ParameterType, LinearTerm, QuadraticTerm, CircuitProblemRepresentation, OptimizationMetrics,
    EnhancedQaoaOptimizer, OptimizationLevel, CircuitCostEstimate, create_qaoa_bridge_for_problem,
    qaoa_parameters_to_circuit_parameters, validate_circuit_compatibility,
};
pub use qubo_decomposition::{
    QuboDecomposer, DecompositionConfig, DecompositionStrategy, DecomposedSolution,
    SubProblem, SubSolution, DecompositionStats, DecompositionError,
};
pub use quantum_boltzmann_machine::{
    QuantumRestrictedBoltzmannMachine, LayerConfig, UnitType, QbmTrainingConfig,
    QbmTrainingStats, QuantumSamplingStats, TrainingSample, QbmInferenceResult,
    QbmError, QbmResult, create_binary_rbm, create_gaussian_bernoulli_rbm,
};
pub use quantum_machine_learning::{
    VariationalQuantumClassifier, VqcConfig, TrainingSample as QmlTrainingSample, TrainingHistory,
    QuantumNeuralNetwork, QuantumNeuralLayer, QnnConfig, ActivationType,
    QuantumFeatureMap, FeatureMapType, EntanglementType, QuantumCircuit, QuantumLayer, QuantumGate as QmlQuantumGate,
    QuantumKernelMethod, KernelMethodType, QuantumGAN, QGanConfig, QGanTrainingHistory,
    QuantumRLAgent, QRLConfig, Experience, QRLStats, QuantumAutoencoder, QAutoencoderConfig,
    QmlMetrics, QmlError, QmlResult, create_binary_classifier, create_zz_feature_map,
    create_quantum_svm, evaluate_qml_model,
};
pub use quantum_walk::{
    QuantumWalkOptimizer, QuantumWalkConfig, QuantumWalkAlgorithm, QuantumState as QuantumWalkState,
    CoinOperator, AdiabaticHamiltonian, QuantumWalkError, QuantumWalkResult,
};
pub use qubo::{QuboBuilder, QuboError, QuboFormulation, QuboResult};
pub use reverse_annealing::{
    ReverseAnnealingSimulator, ReverseAnnealingParams, ReverseAnnealingSchedule,
    ReverseAnnealingScheduleBuilder,
};
pub use simulator::{
    AnnealingError, AnnealingParams, AnnealingResult, AnnealingSolution,
    ClassicalAnnealingSimulator, QuantumAnnealingSimulator, TemperatureSchedule,
    TransverseFieldSchedule,
};
pub use solution_clustering::{
    SolutionClusteringAnalyzer, ClusteringConfig, ClusteringResults, SolutionCluster,
    ClusteringAlgorithm, DistanceMetric, FeatureExtractionMethod, AnalysisDepth,
    SolutionPoint, LandscapeAnalysis, StatisticalSummary, OptimizationRecommendation,
    ClusteringError, ClusteringResult, LinkageType, DimensionalityReduction,
    create_basic_clustering_config, create_comprehensive_clustering_config,
    analyze_solution_diversity, find_representative_solution,
};
pub use variational_quantum_annealing::{
    VariationalQuantumAnnealer, VqaConfig, VqaResults, VqaStatistics, ParameterStatistics,
    OptimizerStatistics, AnsatzType, ClassicalOptimizer, EntanglingGateType, MixerType as VqaMixerType,
    QuantumGate as VqaQuantumGate, QuantumCircuit as VqaQuantumCircuit, ParameterRef, VqaError, VqaResult,
    create_qaoa_vqa_config, create_hardware_efficient_vqa_config, create_adiabatic_vqa_config,
};
pub use visualization::{
    BasinAnalyzer, LandscapeAnalyzer, LandscapePoint, LandscapeStats, VisualizationError,
    VisualizationResult, calculate_landscape_stats, plot_energy_histogram, plot_energy_landscape,
};

/// Check if quantum annealing support is available
///
/// This function always returns `true` since the simulation capabilities
/// are always available.
pub fn is_available() -> bool {
    true
}

/// Check if hardware quantum annealing is available
///
/// This function checks if any quantum annealing hardware API clients are available
/// and enabled via their respective features (D-Wave or AWS Braket).
pub fn is_hardware_available() -> bool {
    dwave::is_available() || braket::is_available()
}
