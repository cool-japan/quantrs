//! Organized public API for QuantRS2 Simulation
//!
//! This module provides a hierarchical organization of the simulation crate's public API,
//! designed for the 1.0 release with clear naming conventions and logical grouping.

pub mod prelude;

/// Core simulation interfaces and backends
pub mod simulation {
    pub use crate::enhanced_statevector::EnhancedStateVectorSimulator;
    pub use crate::error::{Result, SimulatorError};
    pub use crate::simulator::{Simulator, SimulatorResult};
    /// Basic simulators
    pub use crate::statevector::StateVectorSimulator;

    pub use crate::clifford_sparse::{CliffordGate, SparseCliffordSimulator};
    pub use crate::mps_basic::{BasicMPS, BasicMPSConfig, BasicMPSSimulator};
    pub use crate::mps_simulator::{MPSSimulator, MPS};
    /// Specialized simulators
    pub use crate::specialized_simulator::{
        benchmark_specialization, SpecializationStats, SpecializedSimulatorConfig,
        SpecializedStateVectorSimulator,
    };
    pub use crate::stabilizer::{is_clifford_circuit, StabilizerGate, StabilizerSimulator};

    #[cfg(feature = "mps")]
    pub use crate::mps_enhanced::{utils::*, EnhancedMPS, EnhancedMPSSimulator, MPSConfig};
}

/// GPU and accelerated computing backends
pub mod gpu {
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pub use crate::gpu_linalg::{benchmark_gpu_linalg, GpuLinearAlgebra};

    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pub use crate::cuda_kernels::{
        CudaKernelConfig, CudaKernelStats, CudaQuantumKernels, GateType as CudaGateType,
        OptimizationLevel as CudaOptimizationLevel,
    };

    #[cfg(all(feature = "advanced_math", not(target_os = "macos")))]
    pub use crate::cuda_kernels::{CudaContext, CudaDeviceProperties, CudaKernel};

    pub use crate::opencl_amd_backend::{
        benchmark_amd_opencl_backend, AMDOpenCLSimulator, KernelArg, MemoryFlags, OpenCLBuffer,
        OpenCLConfig, OpenCLDevice, OpenCLDeviceType, OpenCLKernel, OpenCLPlatform, OpenCLStats,
        OptimizationLevel as OpenCLOptimizationLevel,
    };

    pub use crate::tpu_acceleration::{
        benchmark_tpu_acceleration, CommunicationBackend, DistributedContext, MemoryOptimization,
        TPUConfig, TPUDataType, TPUDeviceInfo, TPUDeviceType, TPUMemoryManager,
        TPUQuantumSimulator, TPUStats, TPUTensorBuffer, TPUTopology, XLAComputation,
    };
}

/// Large-scale and distributed simulation
pub mod distributed {
    pub use crate::large_scale_simulator::{
        CompressedQuantumState, CompressionAlgorithm, CompressionMetadata,
        LargeScaleQuantumSimulator, LargeScaleSimulatorConfig, MemoryMappedQuantumState,
        MemoryStatistics as LargeScaleMemoryStatistics, QuantumStateRepresentation,
        SparseQuantumState,
    };

    pub use crate::distributed_simulator::{
        benchmark_distributed_simulation, ChunkMetadata, CommunicationConfig, CommunicationManager,
        CommunicationPattern, CommunicationRequirements, DistributedGateOperation,
        DistributedPerformanceStats, DistributedQuantumSimulator, DistributedSimulatorConfig,
        DistributionStrategy, FaultToleranceConfig, FaultToleranceMessage, FaultToleranceStats,
        LoadBalancer, LoadBalancingCommand, LoadBalancingConfig,
        LoadBalancingStrategy as DistributedLoadBalancingStrategy, NetworkConfig, NetworkMessage,
        NetworkStats, NodeCapabilities, NodeInfo, NodePerformanceStats, NodeStatus,
        OperationPriority, RebalancingStats, SimulationState, StateChunk, SynchronizationLevel,
        WorkDistribution,
    };

    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pub use crate::distributed_gpu::*;
}

/// Optimization and performance tools
pub mod optimization {
    pub use crate::circuit_optimization::{
        optimize_circuit, optimize_circuit_with_config, CircuitOptimizer, OptimizationConfig,
        OptimizationResult, OptimizationStatistics,
    };

    pub use crate::auto_optimizer::{
        execute_with_auto_optimization, recommend_backend_for_circuit, AnalysisDepth,
        AutoOptimizer, AutoOptimizerConfig, BackendRecommendation, BackendType,
        CircuitCharacteristics, ConnectivityProperties, FallbackStrategy,
        OptimizationLevel as AutoOptimizationLevel, PerformanceHistory,
        PerformanceMetrics as AutoOptimizerPerformanceMetrics,
    };

    pub use crate::performance_prediction::{
        create_performance_predictor, predict_circuit_execution_time, ComplexityMetrics,
        ExecutionDataPoint, ModelType, PerformanceHardwareSpecs, PerformancePredictionConfig,
        PerformancePredictionEngine, PerformanceTimingStatistics, PredictionMetadata,
        PredictionResult, PredictionStatistics, PredictionStrategy, ResourceMetrics, TrainedModel,
        TrainingStatistics,
    };

    pub use crate::compilation_optimization::{
        CompilationAnalysis, CompilationOptimizer, CompilationOptimizerConfig,
        OptimizationRecommendation, OptimizationType, RecommendationPriority,
    };

    pub use crate::automatic_parallelization::{
        benchmark_automatic_parallelization, AutoParallelBenchmarkResults, AutoParallelConfig,
        AutoParallelEngine, CircuitParallelResult, DependencyGraph, GateNode,
        LoadBalancingConfig as AutoLoadBalancingConfig, OptimizationLevel,
        OptimizationRecommendation as ParallelOptimizationRecommendation, ParallelPerformanceStats,
        ParallelTask, ParallelizationAnalysis, ParallelizationStrategy, RecommendationComplexity,
        RecommendationType, ResourceConstraints, ResourceSnapshot, ResourceUtilization,
        TaskCompletionStats, TaskPriority, WorkStealingStrategy,
    };
}

/// Performance analysis and profiling
pub mod profiling {
    pub use crate::performance_benchmark::{
        run_comprehensive_benchmark, run_quick_benchmark, BenchmarkComparison, BenchmarkConfig,
        BenchmarkResult, MemoryStats as BenchmarkMemoryStats, QuantumBenchmarkSuite,
        ScalabilityAnalysis, ThroughputStats, TimingStats,
    };

    pub use crate::telemetry::{
        benchmark_telemetry, Alert, AlertLevel, AlertThresholds, DiskIOStats, MetricsSummary,
        NetworkIOStats, PerformanceSnapshot, QuantumMetrics, TelemetryCollector, TelemetryConfig,
        TelemetryExportFormat, TelemetryMetric,
    };

    pub use crate::benchmark::*;
}

/// Developer tools and debugging
pub mod dev_tools {
    pub use crate::debugger::{
        BreakCondition, DebugConfig, DebugReport, PerformanceMetrics, QuantumDebugger, StepResult,
        WatchFrequency, WatchProperty, Watchpoint,
    };

    pub use crate::diagnostics::*;

    pub use crate::visualization_hooks::{
        benchmark_visualization, ASCIIVisualizationHook, ColorScheme, GateVisualizationData,
        JSONVisualizationHook, VisualizationConfig, VisualizationData, VisualizationFramework,
        VisualizationHook, VisualizationManager,
    };
}

/// Noise models and error correction
pub mod noise {
    pub use crate::noise::*;
    pub use crate::noise::{NoiseChannel, NoiseModel};
    pub use crate::noise_advanced::*;
    pub use crate::noise_advanced::{AdvancedNoiseModel, RealisticNoiseModelBuilder};

    pub use crate::device_noise_models::{
        CalibrationData, CoherenceParameters, DeviceNoiseConfig, DeviceNoiseModel,
        DeviceNoiseSimulator, DeviceNoiseUtils, DeviceTopology, DeviceType, FrequencyDrift,
        GateErrorRates, GateTimes, NoiseBenchmarkResults, NoiseSimulationStats,
        SuperconductingNoiseModel,
    };

    pub use crate::noise_extrapolation::{
        benchmark_noise_extrapolation, DistillationProtocol, ExtrapolationMethod, FitStatistics,
        NoiseScalingMethod, SymmetryOperation, SymmetryVerification, SymmetryVerificationResult,
        VirtualDistillation, VirtualDistillationResult, ZNEResult, ZeroNoiseExtrapolator,
    };

    pub use crate::open_quantum_systems::{
        quantum_fidelity, CompositeNoiseModel, EvolutionResult, IntegrationMethod, LindladOperator,
        LindladSimulator, NoiseModelBuilder, ProcessTomography, QuantumChannel,
    };
}

/// Error correction codes and protocols
pub mod error_correction {
    #[allow(unused_imports)]
    pub use crate::error_correction::*;

    pub use crate::adaptive_ml_error_correction::{
        benchmark_adaptive_ml_error_correction, AdaptiveCorrectionResult, AdaptiveMLConfig,
        AdaptiveMLErrorCorrection, CorrectionMetrics, ErrorCorrectionAgent,
        FeatureExtractionMethod, FeatureExtractor, LearningStrategy, MLModelType,
        SyndromeClassificationNetwork, TrainingExample as MLTrainingExample,
    };

    pub use crate::concatenated_error_correction::{
        benchmark_concatenated_error_correction, create_standard_concatenated_code, CodeParameters,
        ConcatenatedCodeConfig, ConcatenatedCorrectionResult, ConcatenatedErrorCorrection,
        ConcatenationLevel, ConcatenationStats, DecodingResult, ErrorCorrectionCode, ErrorType,
        HierarchicalDecodingMethod, LevelDecodingResult,
    };

    pub use crate::holographic_quantum_error_correction::{
        benchmark_holographic_qec, BulkReconstructionMethod, BulkReconstructionResult,
        HolographicCodeType, HolographicQECBenchmarkResults, HolographicQECConfig,
        HolographicQECResult, HolographicQECSimulator, HolographicQECStats, HolographicQECUtils,
    };

    pub use crate::quantum_ldpc_codes::{
        benchmark_quantum_ldpc_codes, BPDecodingResult, BeliefPropagationAlgorithm, CheckNode,
        LDPCConfig, LDPCConstructionMethod, LDPCStats, QuantumLDPCCode, TannerGraph, VariableNode,
    };
}

/// Advanced algorithms and applications
pub mod algorithms {
    pub use crate::adiabatic_quantum_computing::{
        AdiabaticBenchmarkResults, AdiabaticConfig, AdiabaticQuantumComputer, AdiabaticResult,
        AdiabaticSnapshot, AdiabaticStats, AdiabaticUtils, GapMeasurement, GapTrackingConfig,
        ScheduleType,
    };

    pub use crate::advanced_variational_algorithms::{
        benchmark_advanced_vqa, AcquisitionFunction, AdvancedOptimizerType, AdvancedVQATrainer,
        BayesianModel, CompressionMethod, CostFunction, FiniteDifferenceGradient,
        GradientCalculator, GrowthCriterion, HamiltonianTerm as VQAHamiltonianTerm,
        IsingCostFunction, MixerHamiltonian, MixerType, NetworkConnectivity,
        OptimizationProblemType, OptimizerState as VQAOptimizerState, ParameterShiftGradient,
        ProblemHamiltonian, QuantumActivation, TensorTopology, VQAConfig, VQAResult,
        VQATrainerState, VQATrainingStats, VariationalAnsatz, WarmRestartConfig,
    };

    pub use crate::autodiff_vqe::{
        ansatze, AutoDiffContext, ConvergenceCriteria, GradientMethod, ParametricCircuit,
        ParametricGate, ParametricRX, ParametricRY, ParametricRZ, VQEIteration, VQEResult,
        VQEWithAutodiff,
    };

    pub use crate::quantum_algorithms::{
        benchmark_quantum_algorithms, AlgorithmResourceStats, EnhancedPhaseEstimation,
        GroverResult, OptimizationLevel as AlgorithmOptimizationLevel, OptimizedGroverAlgorithm,
        OptimizedShorAlgorithm, PhaseEstimationResult, QuantumAlgorithmConfig, ShorResult,
    };

    pub use crate::quantum_annealing::{
        AnnealingBenchmarkResults, AnnealingResult, AnnealingScheduleType, AnnealingSolution,
        AnnealingStats, AnnealingTopology, IsingProblem, ProblemType, QUBOProblem,
        QuantumAnnealingConfig, QuantumAnnealingSimulator, QuantumAnnealingUtils,
    };

    pub use crate::qaoa_optimization::{
        benchmark_qaoa, LevelTransitionCriteria, MultiLevelQAOAConfig, QAOAConfig, QAOAConstraint,
        QAOAGraph, QAOAInitializationStrategy, QAOALevel, QAOAMixerType, QAOAOptimizationStrategy,
        QAOAOptimizer, QAOAProblemType, QAOAResult, QAOAStats,
        QuantumAdvantageMetrics as QAOAQuantumAdvantageMetrics, SolutionQuality,
    };
}

/// Quantum machine learning and AI
pub mod quantum_ml {
    pub use crate::qml_integration::{
        AdamOptimizer, LossFunction, OptimizerType, QMLBenchmarkResults, QMLFramework,
        QMLIntegration, QMLIntegrationConfig, QMLLayer, QMLLayerType, QMLOptimizer,
        QMLTrainingStats, QMLUtils, QuantumNeuralNetwork, SGDOptimizer, TrainingConfig,
        TrainingExample, TrainingResult,
    };

    pub use crate::quantum_machine_learning_layers::{
        benchmark_quantum_ml_layers, AdversarialAttackMethod, AdversarialDefenseMethod,
        AdversarialTrainingConfig, AlternatingSchedule, AnsatzType, AttentionHead,
        BenchmarkingProtocols, CachingConfig, CalibrationFrequency, ClassicalArchitecture,
        ClassicalPreprocessingConfig, ComputationOptimizationConfig, ConnectivityConstraints,
        ConvolutionalFilter, DataEncodingMethod, DenseConnection,
        DistillationProtocol as QMLDistillationProtocol, EarlyStoppingConfig, EnsembleMethod,
        EnsembleMethodsConfig, EntanglementPattern, ErrorMitigationConfig, FeatureSelectionConfig,
        FeatureSelectionMethod, GradientFlowConfig, GradientMethod as QMLGradientMethod,
        HardwareOptimizationConfig, HardwareOptimizationLevel, HybridTrainingConfig, LSTMGate,
        LSTMGateType, LearningRateSchedule,
        MemoryOptimizationConfig as QMLMemoryOptimizationConfig, NoiseAwareTrainingConfig,
        NoiseCharacterizationConfig, NoiseCharacterizationMethod, NoiseInjectionConfig,
        NoiseParameters, NoiseType, OptimizerType as QMLOptimizerType, PQCGate, PQCGateType,
        ParallelizationConfig, ParameterizedQuantumCircuitLayer, PerformanceOptimizationConfig,
        QMLArchitectureType, QMLBenchmarkResults as QMLLayersQMLBenchmarkResults, QMLConfig,
        QMLEpochMetrics, QMLLayer as QMLLayersQMLLayer, QMLLayerConfig,
        QMLLayerType as QMLLayersQMLLayerType, QMLStats, QMLTrainingAlgorithm, QMLTrainingConfig,
        QMLTrainingResult, QMLTrainingState, QMLUtils as QMLLayersQMLUtils,
        QuantumAdvantageMetrics as QMLQuantumAdvantageMetrics, QuantumAttentionLayer,
        QuantumClassicalInterface, QuantumConvolutionalLayer, QuantumDenseLayer,
        QuantumHardwareTarget, QuantumLSTMLayer, QuantumMLFramework, RegularizationConfig,
        RobustTrainingConfig, RotationGate, ScalingMethod, TwoQubitGate, VirtualDistillationConfig,
        VotingStrategy,
    };

    pub use crate::quantum_ml_algorithms::{
        benchmark_quantum_ml_algorithms, GradientMethod as QMLAlgorithmsGradientMethod,
        HardwareArchitecture, HardwareAwareCompiler, HardwareMetrics, HardwareOptimizations,
        OptimizerState, OptimizerType as QMLAlgorithmsOptimizerType, ParameterizedQuantumCircuit,
        QMLAlgorithmType, QMLConfig as QMLAlgorithmsConfig, QuantumMLTrainer, TrainingHistory,
        TrainingResult as QMLAlgorithmsTrainingResult,
    };

    pub use crate::quantum_reservoir_computing::{
        benchmark_quantum_reservoir_computing, InputEncoding, OutputMeasurement,
        QuantumReservoirArchitecture, QuantumReservoirComputer, QuantumReservoirConfig,
        QuantumReservoirState, ReservoirDynamics, ReservoirMetrics, ReservoirTrainingData,
        TrainingResult as ReservoirTrainingResult,
    };

    pub use crate::quantum_reservoir_computing_enhanced::{
        benchmark_enhanced_quantum_reservoir_computing, ARIMAParams,
        ActivationFunction as ReservoirActivationFunction, AdvancedLearningConfig, IPCFunction,
        LearningAlgorithm, MemoryAnalysisConfig, MemoryAnalyzer, MemoryKernel, MemoryMetrics,
        MemoryTask, NARState, QuantumReservoirComputerEnhanced,
        ReservoirTrainingData as EnhancedReservoirTrainingData, TimeSeriesConfig,
        TimeSeriesPredictor, TrainingExample as ReservoirTrainingExample,
        TrainingResult as EnhancedTrainingResult, TrendModel,
    };
}

/// Specialized simulation methods
pub mod specialized {
    pub use crate::fermionic_simulation::{
        benchmark_fermionic_simulation, FermionicHamiltonian, FermionicOperator,
        FermionicSimulator, FermionicStats, FermionicString, JordanWignerTransform,
    };

    pub use crate::photonic::{
        benchmark_photonic_methods, FockState, PhotonicConfig, PhotonicMethod, PhotonicOperator,
        PhotonicResult, PhotonicSimulator, PhotonicState, PhotonicStats, PhotonicUtils,
    };

    pub use crate::path_integral::{
        benchmark_path_integral_methods, ConvergenceStats, PathIntegralConfig, PathIntegralMethod,
        PathIntegralResult, PathIntegralSimulator, PathIntegralStats, PathIntegralUtils,
        QuantumPath,
    };

    pub use crate::qmc::{DMCResult, PIMCResult, VMCResult, Walker, WaveFunction, DMC, PIMC, VMC};

    pub use crate::decision_diagram::{
        benchmark_dd_simulator, DDNode, DDOptimizer, DDSimulator, DDStats, DecisionDiagram, Edge,
    };

    pub use crate::topological_quantum_simulation::{
        AnyonModel, AnyonType, LatticeType, TopologicalBoundaryConditions, TopologicalConfig,
        TopologicalErrorCode, TopologicalQuantumSimulator,
    };
}

/// Tensor network methods
pub mod tensor_networks {
    #[cfg(feature = "advanced_math")]
    pub use crate::tensor_network::*;

    pub use crate::enhanced_tensor_networks::*;

    pub use crate::parallel_tensor_optimization::{
        ContractionPair, LoadBalancingStrategy, NumaTopology, ParallelTensorConfig,
        ParallelTensorEngine, ParallelTensorStats, TensorWorkQueue, TensorWorkUnit,
        ThreadAffinityConfig,
    };
}

/// Memory optimization and management
pub mod memory {
    pub use crate::memory_optimization::{
        AdvancedMemoryPool, MemoryStats as AdvancedMemoryStats, NumaAwareAllocator,
    };

    pub use crate::memory_bandwidth_optimization::{
        BandwidthMonitor, MemoryBandwidthOptimizer, MemoryLayout, MemoryOptimizationConfig,
        MemoryOptimizationReport, MemoryStats, OptimizedStateVector,
    };

    pub use crate::memory_prefetching_optimization::{
        AccessPatternPredictor, AccessPatternType, DataLocalityOptimizer,
        LocalityOptimizationResult, LocalityStrategy, LoopPattern, MemoryPrefetcher, NUMATopology,
        PerformanceFeedback, PrefetchConfig, PrefetchHint, PrefetchStats, PrefetchStrategy,
    };

    pub use crate::cache_optimized_layouts::{
        CacheHierarchyConfig, CacheLayoutAdaptationResult, CacheOperationStats,
        CacheOptimizedGateManager, CacheOptimizedLayout, CacheOptimizedStateVector,
        CachePerformanceStats, CacheReplacementPolicy,
    };
}

/// SIMD and high-performance operations
pub mod simd {
    pub use crate::scirs2_complex_simd::{
        apply_cnot_complex_simd, apply_hadamard_gate_complex_simd,
        apply_single_qubit_gate_complex_simd, benchmark_complex_simd_operations, ComplexSimdOps,
        ComplexSimdVector,
    };

    pub use crate::scirs2_integration::{
        BackendStats as SciRS2BackendStats, SciRS2Backend, SciRS2Matrix, SciRS2MemoryAllocator,
        SciRS2ParallelContext, SciRS2SimdConfig, SciRS2SimdContext, SciRS2Vector,
        SciRS2VectorizedFFT,
    };
}

/// Gate operations and fusion
pub mod gates {
    pub use crate::fusion::{
        benchmark_fusion_strategies, FusedGate, FusionStats, FusionStrategy, GateFusion, GateGroup,
        OptimizedCircuit, OptimizedGate,
    };

    pub use crate::adaptive_gate_fusion::*;

    pub use crate::specialized_gates::{
        specialize_gate, CNOTSpecialized, CPhaseSpecialized, CZSpecialized, FredkinSpecialized,
        HadamardSpecialized, PauliXSpecialized, PauliYSpecialized, PauliZSpecialized,
        PhaseSpecialized, RXSpecialized, RYSpecialized, RZSpecialized, SGateSpecialized,
        SWAPSpecialized, SpecializedGate, TGateSpecialized, ToffoliSpecialized,
    };

    pub use crate::operation_cache::{
        CacheConfig, CacheStats, CachedData, CachedOperation, EvictionPolicy, GateMatrixCache,
        OperationKey, QuantumOperationCache,
    };
}

/// Measurement and sampling
pub mod measurement {
    pub use crate::shot_sampling::{
        analysis, BitString, ComparisonResult, ConvergenceResult, ExpectationResult,
        MeasurementStatistics, NoiseModel as SamplingNoiseModel, QuantumSampler,
        SamplingConfig as ShotSamplingConfig, ShotResult, SimpleReadoutNoise,
    };
}

/// SciRS2 integration and enhancements
pub mod scirs2 {
    pub use crate::scirs2_eigensolvers::{
        benchmark_spectral_analysis, BandStructureResult, EntanglementSpectrumResult,
        PhaseTransitionResult, QuantumHamiltonianLibrary, SciRS2SpectralAnalyzer,
        SpectralAnalysisResult, SpectralConfig, SpectralDensityResult, SpectralStatistics,
    };

    pub use crate::scirs2_qft::{
        benchmark_qft_methods, compare_qft_accuracy, QFTConfig, QFTMethod, QFTStats, QFTUtils,
        SciRS2QFT,
    };

    pub use crate::scirs2_sparse::{
        benchmark_sparse_solvers, compare_sparse_solver_accuracy, Preconditioner,
        SciRS2SparseSolver, SparseEigenResult, SparseFormat, SparseMatrix, SparseMatrixUtils,
        SparseSolverConfig, SparseSolverMethod, SparseSolverStats,
    };
}

/// Utility functions and common operations
pub mod utils {
    pub use crate::pauli::{PauliOperator, PauliOperatorSum, PauliString, PauliUtils};
    pub use crate::sparse::{apply_sparse_gate, CSRMatrix, SparseGates, SparseMatrixBuilder};
    pub use crate::trotter::{
        Hamiltonian, HamiltonianLibrary, HamiltonianTerm, TrotterDecomposer, TrotterMethod,
    };
    pub use crate::utils::*;
}

/// Dynamic simulation and JIT compilation
pub mod dynamic {
    pub use crate::dynamic::*;

    pub use crate::jit_compilation::{
        benchmark_jit_compilation, CompilationPriority, CompilationStatus, CompiledFunction,
        CompiledGateSequence, GateSequencePattern, JITBenchmarkResults, JITCompiler, JITConfig,
        JITOptimization, JITOptimizationLevel, JITPerformanceStats, JITQuantumSimulator,
        JITSimulatorStats, OptimizationSuggestion, PatternAnalysisResult, PatternComplexity,
        RuntimeProfiler, RuntimeProfilerStats,
    };
}

/// Mixed precision and adaptive algorithms
pub mod precision {
    pub use crate::precision::{
        benchmark_precisions, AdaptivePrecisionConfig, AdaptiveStateVector, ComplexAmplitude,
        ComplexF16, Precision, PrecisionStats, PrecisionTracker,
    };

    pub use crate::mixed_precision::*;
    pub use crate::mixed_precision_impl::*;
}
