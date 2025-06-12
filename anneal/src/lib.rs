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
pub mod active_learning_decomposition;
pub mod advanced_quantum_algorithms;
pub mod applications;
pub mod bayesian_hyperopt;
pub mod braket;
pub mod chain_break;
pub mod climate_modeling_optimization;
pub mod coherent_ising_machine;
pub mod compression;
pub mod comprehensive_integration_testing;
pub mod continuous_variable;
pub mod csp_compiler;
pub mod dsl;
pub mod dwave;
pub mod embedding;
pub mod flux_bias;
#[cfg(feature = "fujitsu")]
pub mod fujitsu;
pub mod hardware_compilation;
pub mod heterogeneous_hybrid_engine;
pub mod hobo;
pub mod hybrid_solvers;
pub mod ising;
pub mod layout_embedding;
pub mod multi_chip_embedding;
pub mod multi_objective;
pub mod neural_annealing_schedules;
pub mod non_stoquastic;
pub mod partitioning;
pub mod penalty_optimization;
pub mod photonic_annealing;
pub mod population_annealing;
pub mod problem_schedules;
pub mod qaoa;
pub mod qaoa_circuit_bridge;
pub mod quantum_advantage_demonstration;
pub mod quantum_boltzmann_machine;
pub mod quantum_error_correction;
pub mod quantum_machine_learning;
pub mod quantum_walk;
pub mod qubo;
pub mod qubo_decomposition;
pub mod realtime_adaptive_qec;
pub mod realtime_hardware_monitoring;
pub mod reverse_annealing;
pub mod rl_embedding_optimizer;
pub mod scientific_performance_optimization;
pub mod simulator;
pub mod solution_clustering;
pub mod variational_quantum_annealing;
pub mod visualization;

// Re-export key types for convenience
pub use active_learning_decomposition::{
    ActiveLearningConfig, ActiveLearningDecomposer, DecompositionMetadata, DecompositionResult,
    DecompositionStrategy as ActiveDecompositionStrategy, ProblemAnalysis, Subproblem, BoundaryEdge, SubproblemMetadata,
};
pub use advanced_quantum_algorithms::{
    create_custom_infinite_qaoa, create_custom_zeno_annealer, create_infinite_qaoa_optimizer,
    create_quantum_zeno_annealer, AdiabaticShortcutsOptimizer, AdvancedQuantumError,
    AdvancedQuantumResult, ControlOptimizationMethod, ConvergenceMetrics,
    CounterdiabaticApproximation, CounterdiabaticConfig, CounterdiabaticDrivingOptimizer,
    CounterdiabaticMetrics, DepthIncrementStrategy, InfiniteDepthQAOA, InfiniteQAOAConfig,
    InfiniteQAOAStats, ParameterInitializationMethod, QuantumZenoAnnealer, ShortcutMethod,
    ShortcutsConfig, ShortcutsPerformanceStats, ZenoAdaptiveStrategy, ZenoConfig,
    ZenoPerformanceMetrics, ZenoSubspaceProjection,
};
pub use applications::{
    create_benchmark_suite, energy, finance, generate_performance_report, healthcare, logistics,
    manufacturing, telecommunications, transportation, validate_constraints, ApplicationError,
    ApplicationResult, Benchmarkable, IndustryConstraint, IndustryObjective, IndustrySolution,
    OptimizationProblem, ProblemCategory as ApplicationProblemCategory,
};
pub use bayesian_hyperopt::{
    create_annealing_parameter_space, create_bayesian_optimizer, create_custom_bayesian_optimizer,
    AcquisitionFunction, AcquisitionFunctionType, BayesianHyperoptimizer, BayesianOptConfig,
    BayesianOptError, BayesianOptMetrics, BayesianOptResult, ConstraintHandlingMethod,
    GaussianProcessSurrogate, KernelFunction, ObjectiveFunction, OptimizationHistory, Parameter,
    ParameterBounds, ParameterSpace, ParameterType, ParameterValue, ScalarizationMethod,
};
pub use braket::{
    is_available as is_braket_available, AdvancedAnnealingParams, BatchTaskResult, BraketClient,
    BraketDevice, BraketError, BraketResult, CostTracker as BraketCostTracker, DeviceSelector, DeviceStatus as BraketDeviceStatus, DeviceType,
    TaskMetrics, TaskResult, TaskStatus,
};
pub use chain_break::{
    ChainBreakResolver, ChainBreakStats, ChainStrengthOptimizer, HardwareSolution, LogicalProblem,
    ResolutionMethod, ResolvedSolution,
};
pub use coherent_ising_machine::{
    create_low_noise_cim_config, create_realistic_cim_config, create_standard_cim_config,
    CimConfig, CimError, CimPerformanceMetrics, CimResult, CimResults, CoherentIsingMachine,
    Complex, ConvergenceConfig, MeasurementConfig, NetworkTopology, NoiseConfig, OpticalCoupling,
    OpticalParametricOscillator, OpticalStatistics, PumpSchedule,
};
pub use climate_modeling_optimization::{
    AtmosphericDynamicsOptimizer, AtmosphericTarget, CarbonCycleOptimizer, ClimateModelingOptimizer,
    ClimateOptimizationConfig, ClimateOptimizationResult, ClimateParameterSpace, ClimatePerformanceMetrics,
    ConvectionScheme, CouplingMethod, EnergyBalanceOptimizer, GlobalClimateModel, LandSurfaceScheme,
    OceanDynamicsOptimizer, OptimizationMethod, ParameterInfo, RadiationScheme, UncertaintyEstimates,
    ValidationMetric, ValidationResults, create_example_climate_optimizer,
};
pub use compression::{
    BlockDetector, CompressedQubo, CompressionStats, CooCompressor, ReductionMapping,
    VariableReducer,
};
pub use comprehensive_integration_testing::{
    BenchmarkConfig, BenchmarkResults, ComponentIntegrationResults, ComprehensiveIntegrationTesting,
    ComprehensiveValidationResult, EnvironmentRequirements, ExpectedOutcomes, FaultInjectionConfig,
    IntegrationTestCase, IntegrationTestConfig, IntegrationTestResult, PerformanceTestResult,
    StressTestConfig, StressTestResult, SystemIntegrationResults, TestCategory, TestExecutionSpec,
    TestMetadata, TestPriority, TestRegistry, TestResult, TestStorageConfig, ValidationStatus as TestValidationStatus,
    create_example_integration_testing,
};
pub use continuous_variable::{
    create_quadratic_problem, ContinuousAnnealingConfig, ContinuousConstraint,
    ContinuousOptimizationProblem, ContinuousOptimizationStats, ContinuousSolution,
    ContinuousVariable, ContinuousVariableAnnealer, ContinuousVariableError,
    ContinuousVariableResult,
};
pub use csp_compiler::{
    ComparisonOp, CompilationParams, CspCompilationInfo, CspConstraint, CspError, CspObjective,
    CspProblem, CspResult, CspSolution, CspValue, CspVariable, Domain,
};
pub use dsl::{
    patterns, BooleanExpression, Constraint, DslError, DslResult, Expression, ModelSummary,
    Objective, ObjectiveDirection, OptimizationModel, Variable, VariableType, VariableVector,
};
pub use dwave::{
    is_available as is_dwave_available,
    AdvancedProblemParams,
    AnnealingSchedule,
    BatchSubmissionResult,
    ChainStrengthMethod,
    DWaveClient,
    DWaveError,
    DWaveResult,
    EmbeddingConfig,
    HybridSolverParams,
    LeapSolverInfo,
    ProblemInfo,
    ProblemMetrics,
    ProblemParams,
    ProblemStatus,
    SolverCategory,
    SolverSelector,
    // Enhanced Leap types
    SolverType,
};
pub use embedding::{Embedding, HardwareGraph, HardwareTopology, MinorMiner};
pub use flux_bias::{
    CalibrationData, FluxBiasConfig, FluxBiasOptimizer, FluxBiasResult, MLFluxBiasOptimizer,
};
#[cfg(feature = "fujitsu")]
pub use fujitsu::{
    is_available as is_fujitsu_available, FujitsuAnnealingParams, FujitsuClient, FujitsuError,
    FujitsuHardwareSpec, FujitsuResult, GuidanceConfig,
};
pub use hardware_compilation::{
    create_chimera_target, create_ideal_target, CompilationResult, CompilationTarget,
    CompilerConfig, ConnectivityPattern, CouplingUtilization, EmbeddingAlgorithm, EmbeddingInfo,
    HardwareCharacteristics, HardwareCompilationError, HardwareCompilationResult, HardwareCompiler,
    HardwareMapping, HardwareType, OptimizationObjective, ParallelizationStrategy,
    PerformancePrediction, QubitAllocationStrategy, TopologyType,
};
pub use heterogeneous_hybrid_engine::{
    ActiveExecution, ComputeResource, ConsensusAlgorithm, ConsensusResult, CostTracker as HybridCostTracker, 
    ExecutionMetadata, ExecutionPlan, ExecutionStatus, ExecutionStrategy, GeographicConstraints,
    HeterogeneousHybridEngine, HybridEngineConfig, HybridExecutionResult, HybridExecutionTask,
    HybridFaultToleranceConfig, HybridMonitoringConfig, HybridPerformanceMonitor, 
    HybridSystemMetrics, IndividualResult, LoadBalancingDecisions, OptimizationSettings,
    PerformanceEntry, PerformanceRequirements, ProblemComplexity, QualityAssessmentMethod,
    QualityAssessor, QualityMeasurement, QualityRequirements, ResourceAllocationStrategy,
    ResourceAvailability, ResourceConnection, ResourceConstraints, ResourceCost,
    ResourceMetrics, ResourcePerformance, ResourceRequirements, ResourceScheduler,
    ResourceType, ResourceUtilization as HybridResourceUtilization, ResourceWorkload, ResultAggregationStrategy,
    ResultAggregator, SchedulingDecision, TaskPriority, create_example_hybrid_engine,
};
pub use hobo::{
    AuxiliaryVariable, ConstraintViolations, HigherOrderTerm, HoboAnalyzer, HoboProblem, HoboStats,
    QuboReduction, ReductionMethod, ReductionType,
};
pub use hybrid_solvers::{
    HybridQuantumClassicalSolver, HybridSolverConfig, HybridSolverResult, VariationalHybridSolver,
};
pub use ising::{IsingError, IsingModel, IsingResult, QuboModel};
pub use layout_embedding::{LayoutAwareEmbedder, LayoutConfig, LayoutStats, MultiLevelEmbedder};
pub use multi_chip_embedding::{
    ChipMetrics, ChipPerformance, ChipStatus, ChipWorkload, CommunicationChannel as ChipCommunicationChannel, 
    CommunicationProtocol, ConnectionStatus, FaultToleranceConfig, LoadBalancer, 
    LoadBalancingDecision, LoadBalancingStrategy, Message, MessageType as ChipMessageType, MonitoringConfig as ChipMonitoringConfig,
    MultiChipConfig, MultiChipCoordinator, PerformanceMonitor, PerformanceSnapshot,
    PerformanceThresholds, ProblemPartition, QuantumChip, RecoveryStrategy, 
    ResourceUtilization as ChipResourceUtilization, SystemMetrics, WorkTransfer, create_example_multi_chip_system,
};
pub use multi_objective::{
    MultiObjectiveError, MultiObjectiveFunction, MultiObjectiveOptimizer, MultiObjectiveResult,
    MultiObjectiveResults, MultiObjectiveSolution, MultiObjectiveStats, QualityMetrics,
};
pub use neural_annealing_schedules::{
    ActivationFunction, AnnealingSchedule as NeuralAnnealingSchedule, AttentionMechanism,
    DatabaseStatistics, DenseLayer, FeatureExtractor as NeuralFeatureExtractor, GenerationMethod, LearningRateSchedule,
    NetworkArchitecture, NeuralAnnealingScheduler, NeuralSchedulerConfig,
    OptimizerType, PerformanceMetric, PerformanceRecord, PerformanceTarget, ProblemEncoderNetwork,
    ScheduleConstraints, ScheduleDatabase, ScheduleGenerationNetwork, TrainingConfig,
    TrainingManager, ValidationStatus as NeuralValidationStatus,
};
pub use non_stoquastic::{
    create_frustrated_xy_triangle, create_tfxy_model, create_xy_chain, is_hamiltonian_stoquastic,
    xy_to_ising_approximation, ComplexCoupling, ConvergenceInfo, HamiltonianType, InteractionType,
    NonStoquasticError, NonStoquasticHamiltonian, NonStoquasticQMCConfig, NonStoquasticResult,
    NonStoquasticResults, NonStoquasticSimulator, QMCStatistics,
    QuantumState as NonStoquasticQuantumState, SignMitigationStrategy,
};
pub use partitioning::{
    BipartitionMethod, KernighanLinPartitioner, Partition, RecursiveBisectionPartitioner,
    SpectralPartitioner,
};
pub use penalty_optimization::{
    AdvancedPenaltyOptimizer, Constraint as PenaltyConstraint, ConstraintPenaltyOptimizer,
    ConstraintType, PenaltyConfig, PenaltyOptimizer, PenaltyStats,
};
pub use photonic_annealing::{
    create_coherent_state_config, create_low_noise_config, create_measurement_based_config,
    create_realistic_config, create_squeezed_state_config, create_temporal_multiplexing_config,
    ConnectivityType, EvolutionHistory, InitialStateType, MeasurementOutcome, MeasurementStrategy,
    MeasurementType, PhotonicAnnealer, PhotonicAnnealingConfig, PhotonicAnnealingResults,
    PhotonicArchitecture, PhotonicComponent, PhotonicError, PhotonicMetrics, PhotonicResult,
    PhotonicState, PumpPowerSchedule,
};
pub use population_annealing::{
    EnergyStatistics, MpiConfig, PopulationAnnealingConfig, PopulationAnnealingError,
    PopulationAnnealingSimulator, PopulationAnnealingSolution, PopulationMember,
};
pub use problem_schedules::{
    AdaptiveScheduleOptimizer, ProblemSpecificScheduler, ProblemType, ScheduleTemplate,
};
pub use qaoa::{
    create_constrained_qaoa_config, create_qaoa_plus_config, create_standard_qaoa_config,
    create_warm_start_qaoa_config, MixerType as QaoaMixerType,
    ParameterInitialization as QaoaParameterInitialization, ProblemEncoding, QaoaCircuit,
    QaoaCircuitStats, QaoaClassicalOptimizer, QaoaConfig, QaoaError, QaoaLayer, QaoaOptimizer,
    QaoaPerformanceMetrics, QaoaResult, QaoaResults, QaoaVariant, QuantumGate as QaoaQuantumGate,
    QuantumState as QaoaQuantumState, QuantumStateStats,
};
pub use qaoa_circuit_bridge::{
    create_qaoa_bridge_for_problem, qaoa_parameters_to_circuit_parameters,
    validate_circuit_compatibility, BridgeError, BridgeResult, CircuitBridgeRepresentation,
    CircuitCostEstimate, CircuitProblemRepresentation, EnhancedQaoaOptimizer, LinearTerm,
    OptimizationLevel, OptimizationMetrics, ParameterReference, QaoaCircuitBridge, QuadraticTerm,
};
pub use quantum_boltzmann_machine::{
    create_binary_rbm, create_gaussian_bernoulli_rbm, LayerConfig, QbmError, QbmInferenceResult,
    QbmResult, QbmTrainingConfig, QbmTrainingStats, QuantumRestrictedBoltzmannMachine,
    QuantumSamplingStats, TrainingSample, UnitType,
};
pub use quantum_error_correction::{
    ErrorMitigationConfig, ErrorMitigationManager, MitigationResult, MitigationTechnique,
    QECConfig, QuantumErrorCorrectionError, SyndromeDetector,
    LogicalAnnealingEncoder, NoiseResilientAnnealingProtocol, ErrorCorrectionCode,
};
pub use realtime_adaptive_qec::{
    AdaptationAction, AdaptationCondition, AdaptationDecision, AdaptationRule, AdaptiveProtocol,
    AdaptiveProtocolManager, AdaptiveQecConfig, AdaptiveQecMetrics, AdaptiveResourceManager,
    AdaptiveStrategyConfig, AnalysisAlgorithm, AnalysisType, CommunicationChannel as QecCommunicationChannel,
    CommunicationStatistics, CoordinationAlgorithm, CoordinationStrategy, CorrectedProblem,
    CorrectionConfig, CorrectionMetadata, CorrectionResult, DetectionAction, DetectionConfig,
    DetectionMethod, ErrorCorrectionStrategy, ErrorInfo, FeatureConfig, FeatureDefinition,
    FeatureExtractor as QecFeatureExtractor, FeatureNormalization, FeatureSelection, FeatureType, HierarchyConfig,
    HierarchyCoordinator, HierarchyLevel, HierarchyMessage, HybridConfig, MLNoiseConfig,
    MessagePayload, MessageType as QecMessageType, ModelEnsemble, NeuralArchitecture, NoiseAnalyzer,
    NoiseAssessment, NoiseCharacteristics, NoiseDataPoint, NoiseMonitor, NoisePrediction,
    NoisePredictionModel, NoisePredictionSystem, NoiseSensor, NoiseSeverity, NoiseType,
    PredictionConfig, PredictionResult, RealTimeAdaptiveQec, ResourceAllocationStrategy as QecResourceAllocationStrategy,
    ResourceConstraints as QecResourceConstraints, ResourceOptimizer, ResourceReallocation, SensorType, StrategyPerformance,
    SwitchingCriteria, TrendDirection as QecTrendDirection, create_example_adaptive_qec,
};
pub use realtime_hardware_monitoring::{
    AdaptationStatus, AdaptiveAction, AdaptiveActionType, AdaptiveCompiler, AdaptiveCompilerConfig,
    Alert, AlertHandler, AlertLevel, AlertSystem, AlertThresholds, CalibrationData as MonitorCalibrationData, ChangeType,
    CoherenceCharacteristics, DecoherenceSource, DeviceConnection, DeviceInfo, DevicePerformanceMetrics,
    DeviceStatus as MonitorDeviceStatus, FailureDetectionConfig, FailurePrediction, FailureSeverity, FailureType,
    MetricType, MonitoredDevice, MonitoringConfig as HardwareMonitoringConfig, MonitoringSnapshot, NoiseModel, NoiseProfile,
    PredictiveFailureDetector, QuantumOperation, RealTimeHardwareMonitor, RealTimePerformanceOptimizer,
    TrendDirection as MonitorTrendDirection, create_example_hardware_monitor,
};
pub use quantum_advantage_demonstration::{
    AdvantageConfig, AdvantageMetric, AdvantageCertification, AdvantageDemonstrationResult,
    BenchmarkSuite, CertificationLevel, ClassicalAlgorithm, ClassicalBaselineOptimizer,
    ProblemCategory as AdvantageProblemCategory, QuantumAdvantageDemonstrator, QuantumDevice, QuantumPerformanceAnalyzer,
    QuantumPerformanceMetrics, ResultsDatabase, StatisticalAnalyzer, create_example_advantage_demonstrator,
};
pub use quantum_machine_learning::{
    create_binary_classifier, create_quantum_svm, create_zz_feature_map, evaluate_qml_model,
    ActivationType, EntanglementType, Experience, FeatureMapType, KernelMethodType,
    QAutoencoderConfig, QGanConfig, QGanTrainingHistory, QRLConfig, QRLStats, QmlError, QmlMetrics,
    QmlResult, QnnConfig, QuantumAutoencoder, QuantumCircuit, QuantumFeatureMap, QuantumGAN,
    QuantumGate as QmlQuantumGate, QuantumKernelMethod, QuantumLayer, QuantumNeuralLayer,
    QuantumNeuralNetwork, QuantumRLAgent, TrainingHistory, TrainingSample as QmlTrainingSample,
    VariationalQuantumClassifier, VqcConfig,
};
pub use quantum_walk::{
    AdiabaticHamiltonian, CoinOperator, QuantumState as QuantumWalkState, QuantumWalkAlgorithm,
    QuantumWalkConfig, QuantumWalkError, QuantumWalkOptimizer, QuantumWalkResult,
};
pub use qubo::{QuboBuilder, QuboError, QuboFormulation, QuboResult};
pub use qubo_decomposition::{
    DecomposedSolution, DecompositionConfig, DecompositionError, DecompositionStats,
    DecompositionStrategy as QuboDecompositionStrategy, QuboDecomposer, SubProblem, SubSolution,
};
pub use reverse_annealing::{
    ReverseAnnealingParams, ReverseAnnealingSchedule, ReverseAnnealingScheduleBuilder,
    ReverseAnnealingSimulator,
};
pub use rl_embedding_optimizer::{
    create_custom_rl_embedding_optimizer, create_rl_embedding_optimizer, ContinuousEmbeddingAction,
    DiscreteEmbeddingAction, EmbeddingAction, EmbeddingDQN, EmbeddingExperience,
    EmbeddingPolicyNetwork, EmbeddingQualityMetrics, EmbeddingState, HardwareFeatures,
    ObjectiveWeights, ProblemGraphFeatures, RLEmbeddingConfig, RLEmbeddingError,
    RLEmbeddingOptimizer, RLEmbeddingResult, RLPerformanceMetrics, RLTrainingStats,
};
pub use scientific_performance_optimization::{
    AlgorithmOptimizationConfig, ApproximationConfig, BottleneckAnalysis, CachingConfig,
    CompressionConfig, ComprehensivePerformanceReport, DecompositionConfig as PerfDecompositionConfig, DistributedComputingConfig,
    GPUAccelerationConfig, HierarchicalMemoryManager, LoadBalancingConfig, MemoryOptimizationConfig,
    OptimizationCategory, OptimizationImpact, OptimizationRecommendation as PerfOptimizationRecommendation, OptimizedDrugDiscoveryResult,
    OptimizedMaterialsScienceResult, OptimizedProteinFoldingResult, ParallelProcessingConfig,
    PerformanceOptimizationConfig, ProfilingConfig, ResourceUtilizationAnalysis, ScientificPerformanceOptimizer,
    StreamingConfig, SystemPerformanceMetrics, ThreadPoolConfig, create_example_performance_optimizer,
};
pub use simulator::{
    AnnealingError, AnnealingParams, AnnealingResult, AnnealingSolution,
    ClassicalAnnealingSimulator, QuantumAnnealingSimulator, TemperatureSchedule,
    TransverseFieldSchedule,
};
pub use solution_clustering::{
    analyze_solution_diversity, create_basic_clustering_config,
    create_comprehensive_clustering_config, find_representative_solution, AnalysisDepth,
    ClusteringAlgorithm, ClusteringConfig, ClusteringError, ClusteringResult, ClusteringResults,
    DimensionalityReduction, DistanceMetric, FeatureExtractionMethod, LandscapeAnalysis,
    LinkageType, OptimizationRecommendation as ClusteringOptimizationRecommendation, SolutionCluster, SolutionClusteringAnalyzer,
    SolutionPoint, StatisticalSummary,
};
pub use variational_quantum_annealing::{
    create_adiabatic_vqa_config, create_hardware_efficient_vqa_config, create_qaoa_vqa_config,
    AnsatzType, ClassicalOptimizer, EntanglingGateType, MixerType as VqaMixerType,
    OptimizerStatistics, ParameterRef, ParameterStatistics, QuantumCircuit as VqaQuantumCircuit,
    QuantumGate as VqaQuantumGate, VariationalQuantumAnnealer, VqaConfig, VqaError, VqaResult,
    VqaResults, VqaStatistics,
};
pub use visualization::{
    calculate_landscape_stats, plot_energy_histogram, plot_energy_landscape, BasinAnalyzer,
    LandscapeAnalyzer, LandscapePoint, LandscapeStats, VisualizationError, VisualizationResult,
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
