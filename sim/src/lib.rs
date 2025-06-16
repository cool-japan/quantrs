//! Quantum circuit simulators for the QuantRS2 framework.
//!
//! This crate provides various simulation backends for quantum circuits,
//! including state vector simulation on CPU and optionally GPU.
//!
//! It includes both standard and optimized implementations, with the optimized
//! versions leveraging SIMD, memory-efficient algorithms, and parallel processing
//! to enable simulation of larger qubit counts (30+).

pub mod adaptive_gate_fusion;
pub mod adaptive_ml_error_correction;
pub mod adiabatic_quantum_computing;
pub mod advanced_ml_error_mitigation;
pub mod advanced_variational_algorithms;
pub mod autodiff_vqe;
pub mod cache_optimized_layouts;
pub mod circuit_interfaces;
pub mod concatenated_error_correction;
pub mod cuda;
pub mod cuda_kernels;
pub mod debugger;
pub mod decision_diagram;
pub mod device_noise_models;
pub mod distributed_gpu;
pub mod dynamic;
pub mod enhanced_statevector;
pub mod enhanced_tensor_networks;
pub mod error;
pub mod fault_tolerant_synthesis;
pub mod fermionic_simulation;
pub mod fpga_acceleration;
pub mod fusion;
pub mod hardware_aware_qml;
pub mod jit_compilation;
pub mod linalg_ops;
pub mod memory_bandwidth_optimization;
pub mod memory_prefetching_optimization;
pub mod mixed_precision;
pub mod mixed_precision_impl;
pub mod mps_basic;
#[cfg(feature = "mps")]
pub mod mps_enhanced;
pub mod mps_simulator;
pub mod noise_extrapolation;
pub mod open_quantum_systems;
pub mod opencl_amd_backend;
pub mod path_integral;
pub mod pauli;
pub mod photonic;
pub mod precision;
pub mod qmc;
pub mod qml;
pub mod qml_integration;
pub mod qaoa_optimization;
pub mod quantum_advantage_demonstration;
pub mod quantum_algorithms;
pub mod quantum_annealing;
pub mod quantum_chemistry;
pub mod quantum_cellular_automata;
pub mod quantum_cloud_integration;
pub mod quantum_ldpc_codes;
pub mod quantum_ml_algorithms;
pub mod quantum_reservoir_computing;
pub mod quantum_supremacy;
pub mod quantum_volume;
pub mod scirs2_eigensolvers;
pub mod scirs2_integration;
pub mod scirs2_qft;
pub mod scirs2_sparse;
pub mod shot_sampling;
pub mod simulator;
pub mod sparse;
pub mod specialized_gates;
pub mod specialized_simulator;
pub mod stabilizer;
pub mod statevector;
pub mod tensor;
pub mod tpu_acceleration;
pub mod trotter;
pub mod visualization_hooks;
pub mod telemetry;

#[cfg(feature = "advanced_math")]
pub mod tensor_network;
pub mod utils;
// pub mod optimized;  // Temporarily disabled due to implementation issues
// pub mod optimized_simulator;  // Temporarily disabled due to implementation issues
pub mod benchmark;
pub mod circuit_optimization;
pub mod clifford_sparse;
pub mod diagnostics;
pub mod memory_verification_simple;
pub mod optimized_chunked;
pub mod optimized_simd;
pub mod optimized_simple;
pub mod optimized_simulator;
pub mod optimized_simulator_chunked;
pub mod optimized_simulator_simple;
pub mod performance_benchmark;
#[cfg(test)]
pub mod tests;
#[cfg(test)]
pub mod tests_optimized;
#[cfg(test)]
pub mod tests_simple;
#[cfg(test)]
pub mod tests_tensor_network;
#[cfg(test)]
pub mod tests_ultrathink_implementations;

/// Noise models for quantum simulation
pub mod noise;

/// Advanced noise models for realistic device simulation
pub mod noise_advanced;

#[allow(clippy::module_inception)]
pub mod error_correction {
    //! Quantum error correction codes and utilities
    //!
    //! This module will provide error correction codes like the Steane code,
    //! Surface code, and related utilities. For now, it's a placeholder.
}

/// Prelude module that re-exports common types and traits
pub mod prelude {
    pub use crate::adaptive_ml_error_correction::{
        benchmark_adaptive_ml_error_correction, AdaptiveCorrectionResult, AdaptiveMLConfig,
        AdaptiveMLErrorCorrection, CorrectionMetrics, ErrorCorrectionAgent,
        FeatureExtractionMethod, FeatureExtractor, LearningStrategy, MLModelType,
        SyndromeClassificationNetwork, TrainingExample as MLTrainingExample,
    };
    pub use crate::adiabatic_quantum_computing::{
        AdiabaticBenchmarkResults, AdiabaticConfig, AdiabaticQuantumComputer, AdiabaticResult,
        AdiabaticSnapshot, AdiabaticStats, AdiabaticUtils, GapMeasurement, GapTrackingConfig,
        ScheduleType,
    };
    pub use crate::advanced_variational_algorithms::{
        benchmark_advanced_vqa, AdvancedOptimizerType, AdvancedVQATrainer, AcquisitionFunction,
        BayesianModel, CompressionMethod, CostFunction, FiniteDifferenceGradient, GradientCalculator,
        GrowthCriterion, HamiltonianTerm as VQAHamiltonianTerm, IsingCostFunction, MixerHamiltonian, MixerType,
        NetworkConnectivity, OptimizerState as VQAOptimizerState, OptimizationProblemType, ParameterShiftGradient,
        ProblemHamiltonian, QuantumActivation, TensorTopology, VQAConfig, VQAResult, VQATrainerState,
        VQATrainingStats, VariationalAnsatz, WarmRestartConfig,
    };
    pub use crate::autodiff_vqe::{
        ansatze, AutoDiffContext, ConvergenceCriteria, GradientMethod, ParametricCircuit,
        ParametricGate, ParametricRX, ParametricRY, ParametricRZ, VQEIteration, VQEResult,
        VQEWithAutodiff,
    };
    pub use crate::cache_optimized_layouts::{
        CacheHierarchyConfig, CacheLayoutAdaptationResult, CacheOperationStats,
        CacheOptimizedGateManager, CacheOptimizedLayout, CacheOptimizedStateVector,
        CachePerformanceStats, CacheReplacementPolicy,
    };
    pub use crate::circuit_interfaces::{
        BackendCompiledData, CircuitExecutionResult, CircuitInterface, CircuitInterfaceConfig,
        CircuitInterfaceStats, CircuitInterfaceUtils, CircuitMetadata, CircuitOptimizationResult,
        CompilationMetadata, CompiledCircuit, InterfaceBenchmarkResults, InterfaceCircuit,
        InterfaceGate, InterfaceGateType, OptimizationStats, SimulationBackend, StabilizerOp,
    };
    pub use crate::circuit_optimization::{
        optimize_circuit, optimize_circuit_with_config, CircuitOptimizer, OptimizationConfig,
        OptimizationResult, OptimizationStatistics,
    };
    pub use crate::clifford_sparse::{CliffordGate, SparseCliffordSimulator};
    pub use crate::concatenated_error_correction::{
        benchmark_concatenated_error_correction, create_standard_concatenated_code, CodeParameters,
        ConcatenatedCodeConfig, ConcatenatedCorrectionResult, ConcatenatedErrorCorrection,
        ConcatenationLevel, ConcatenationStats, DecodingResult, ErrorCorrectionCode, ErrorType,
        HierarchicalDecodingMethod, LevelDecodingResult,
    };
    #[cfg(feature = "advanced_math")]
    pub use crate::cuda_kernels::{CudaContext, CudaDeviceProperties, CudaKernel};
    pub use crate::cuda_kernels::{
        CudaKernelConfig, CudaKernelStats, CudaQuantumKernels, GateType as CudaGateType,
        OptimizationLevel as CudaOptimizationLevel,
    };
    pub use crate::debugger::{
        BreakCondition, DebugConfig, DebugReport, PerformanceMetrics, QuantumDebugger, StepResult,
        WatchFrequency, WatchProperty, Watchpoint,
    };
    pub use crate::decision_diagram::{
        benchmark_dd_simulator, DDNode, DDOptimizer, DDSimulator, DDStats, DecisionDiagram, Edge,
    };
    pub use crate::device_noise_models::{
        CalibrationData, CoherenceParameters, DeviceNoiseConfig, DeviceNoiseModel,
        DeviceNoiseSimulator, DeviceNoiseUtils, DeviceTopology, DeviceType, FrequencyDrift,
        GateErrorRates, GateTimes, NoiseBenchmarkResults, NoiseSimulationStats,
        SuperconductingNoiseModel,
    };
    pub use crate::dynamic::*;
    pub use crate::enhanced_statevector::EnhancedStateVectorSimulator;
    pub use crate::error::{Result, SimulatorError};
    #[allow(unused_imports)]
    pub use crate::error_correction::*;
    pub use crate::fermionic_simulation::{
        benchmark_fermionic_simulation, FermionicHamiltonian, FermionicOperator,
        FermionicSimulator, FermionicStats, FermionicString, JordanWignerTransform,
    };
    pub use crate::fusion::{
        benchmark_fusion_strategies, FusedGate, FusionStats, FusionStrategy, GateFusion, GateGroup,
        OptimizedCircuit, OptimizedGate,
    };
    pub use crate::jit_compilation::{
        benchmark_jit_compilation, CompilationPriority, CompilationStatus, CompiledFunction,
        CompiledGateSequence, GateSequencePattern, JITBenchmarkResults, JITCompiler, JITConfig,
        JITOptimization, JITOptimizationLevel, JITPerformanceStats, JITQuantumSimulator,
        JITSimulatorStats, OptimizationSuggestion, PatternAnalysisResult, PatternComplexity,
        RuntimeProfiler, RuntimeProfilerStats,
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
    pub use crate::mps_basic::{BasicMPS, BasicMPSConfig, BasicMPSSimulator};
    #[cfg(feature = "mps")]
    pub use crate::mps_enhanced::{utils::*, EnhancedMPS, EnhancedMPSSimulator, MPSConfig};
    pub use crate::mps_simulator::{MPSSimulator, MPS};
    pub use crate::noise::*;
    pub use crate::noise::{NoiseChannel, NoiseModel};
    pub use crate::noise_advanced::*;
    pub use crate::noise_advanced::{AdvancedNoiseModel, RealisticNoiseModelBuilder};
    pub use crate::noise_extrapolation::{
        benchmark_noise_extrapolation, DistillationProtocol, ExtrapolationMethod, FitStatistics,
        NoiseScalingMethod, SymmetryOperation, SymmetryVerification, SymmetryVerificationResult,
        VirtualDistillation, VirtualDistillationResult, ZNEResult, ZeroNoiseExtrapolator,
    };
    pub use crate::open_quantum_systems::{
        quantum_fidelity, CompositeNoiseModel, EvolutionResult, IntegrationMethod, LindladOperator,
        LindladSimulator, NoiseModelBuilder, ProcessTomography, QuantumChannel,
    };
    pub use crate::opencl_amd_backend::{
        benchmark_amd_opencl_backend, AMDOpenCLSimulator, KernelArg, MemoryFlags, OpenCLBuffer,
        OpenCLConfig, OpenCLDevice, OpenCLDeviceType, OpenCLKernel, OpenCLPlatform, OpenCLStats,
        OptimizationLevel as OpenCLOptimizationLevel,
    };
    pub use crate::path_integral::{
        benchmark_path_integral_methods, ConvergenceStats, PathIntegralConfig, PathIntegralMethod,
        PathIntegralResult, PathIntegralSimulator, PathIntegralStats, PathIntegralUtils,
        QuantumPath,
    };
    pub use crate::pauli::{PauliOperator, PauliOperatorSum, PauliString, PauliUtils};
    pub use crate::performance_benchmark::{
        run_comprehensive_benchmark, run_quick_benchmark, BenchmarkComparison, BenchmarkConfig,
        BenchmarkResult, MemoryStats as BenchmarkMemoryStats, QuantumBenchmarkSuite,
        ScalabilityAnalysis, ThroughputStats, TimingStats,
    };
    pub use crate::photonic::{
        benchmark_photonic_methods, FockState, PhotonicConfig, PhotonicMethod, PhotonicOperator,
        PhotonicResult, PhotonicSimulator, PhotonicState, PhotonicStats, PhotonicUtils,
    };
    pub use crate::precision::{
        benchmark_precisions, AdaptivePrecisionConfig, AdaptiveStateVector, ComplexAmplitude,
        ComplexF16, Precision, PrecisionStats, PrecisionTracker,
    };
    pub use crate::qmc::{DMCResult, PIMCResult, VMCResult, Walker, WaveFunction, DMC, PIMC, VMC};
    pub use crate::qml_integration::{
        AdamOptimizer, LossFunction, OptimizerType, QMLBenchmarkResults, QMLFramework,
        QMLIntegration, QMLIntegrationConfig, QMLLayer, QMLLayerType, QMLOptimizer,
        QMLTrainingStats, QMLUtils, QuantumNeuralNetwork, SGDOptimizer, TrainingConfig,
        TrainingExample, TrainingResult,
    };
    pub use crate::qaoa_optimization::{
        benchmark_qaoa, LevelTransitionCriteria, MultiLevelQAOAConfig, QAOAConfig, QAOAConstraint,
        QAOAGraph, QAOAInitializationStrategy, QAOALevel, QAOAMixerType, QAOAOptimizationStrategy,
        QAOAOptimizer, QAOAProblemType, QAOAResult, QAOAStats, QuantumAdvantageMetrics as QAOAQuantumAdvantageMetrics,
        SolutionQuality,
    };
    pub use crate::quantum_advantage_demonstration::{
        benchmark_quantum_advantage, ClassicalAlgorithm, ClassicalAlgorithmType, 
        ClassicalHardwareSpecs, ClassicalResources, CostAnalysis, DetailedResult,
        FutureProjections, HardwareSpecs, MarketImpact, OperationalCosts, ProblemDomain,
        ProblemInstance, QuantumAdvantageConfig, QuantumAdvantageDemonstrator,
        QuantumAdvantageMetrics, QuantumAdvantageResult, QuantumAdvantageType, QuantumAlgorithm,
        QuantumHardwareSpecs, QuantumResources, ScalingAnalysis, StatisticalAnalysis,
        TechnologyProjection, TimelineProjection, VerificationResult,
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
    pub use crate::quantum_cellular_automata::{
        BoundaryConditions, MeasurementStrategy, NeighborhoodType, QCABenchmarkResults, QCAConfig,
        QCAEvolutionResult, QCARule, QCARuleType, QCASnapshot, QCAStats, QCAUtils,
        QuantumCellularAutomaton,
    };
    pub use crate::quantum_ldpc_codes::{
        benchmark_quantum_ldpc_codes, BPDecodingResult, BeliefPropagationAlgorithm, CheckNode,
        LDPCConfig, LDPCConstructionMethod, LDPCStats, QuantumLDPCCode, TannerGraph, VariableNode,
    };
    pub use crate::quantum_ml_algorithms::{
        benchmark_quantum_ml_algorithms, GradientMethod as QMLGradientMethod, HardwareArchitecture,
        HardwareAwareCompiler, HardwareMetrics, HardwareOptimizations, OptimizerState,
        OptimizerType as QMLOptimizerType, ParameterizedQuantumCircuit, QMLAlgorithmType,
        QMLConfig, QuantumMLTrainer, TrainingHistory, TrainingResult as QMLTrainingResult,
    };
    pub use crate::quantum_reservoir_computing::{
        benchmark_quantum_reservoir_computing, InputEncoding, OutputMeasurement,
        QuantumReservoirArchitecture, QuantumReservoirComputer, QuantumReservoirConfig,
        QuantumReservoirState, ReservoirDynamics, ReservoirMetrics, ReservoirTrainingData,
        TrainingResult as ReservoirTrainingResult,
    };
    pub use crate::quantum_supremacy::{
        benchmark_quantum_supremacy, verify_supremacy_claim, CircuitLayer, CostComparison,
        CrossEntropyResult, GateSet, HOGAnalysis, PorterThomasResult, QuantumGate,
        QuantumSupremacyVerifier, RandomCircuit, VerificationParams,
    };
    pub use crate::quantum_volume::{
        benchmark_quantum_volume, calculate_quantum_volume_with_params, QVCircuit, QVGate,
        QVParams, QVStats, QuantumVolumeCalculator, QuantumVolumeResult,
    };
    pub use crate::scirs2_eigensolvers::{
        benchmark_spectral_analysis, BandStructureResult, EntanglementSpectrumResult,
        PhaseTransitionResult, QuantumHamiltonianLibrary, SciRS2SpectralAnalyzer,
        SpectralAnalysisResult, SpectralConfig, SpectralDensityResult, SpectralStatistics,
    };
    pub use crate::scirs2_integration::{BackendStats, SciRS2Backend};
    pub use crate::scirs2_qft::{
        benchmark_qft_methods, compare_qft_accuracy, QFTConfig, QFTMethod, QFTStats, QFTUtils,
        SciRS2QFT,
    };
    pub use crate::scirs2_sparse::{
        benchmark_sparse_solvers, compare_sparse_solver_accuracy, Preconditioner,
        SciRS2SparseSolver, SparseEigenResult, SparseFormat, SparseMatrix, SparseMatrixUtils,
        SparseSolverConfig, SparseSolverMethod, SparseSolverStats,
    };
    pub use crate::shot_sampling::{
        analysis, BitString, ComparisonResult, ConvergenceResult, ExpectationResult,
        MeasurementStatistics, NoiseModel as SamplingNoiseModel, QuantumSampler, SamplingConfig,
        ShotResult, SimpleReadoutNoise,
    };
    #[allow(unused_imports)]
    pub use crate::simulator::*;
    pub use crate::simulator::{Simulator, SimulatorResult};
    pub use crate::sparse::{apply_sparse_gate, CSRMatrix, SparseGates, SparseMatrixBuilder};
    pub use crate::specialized_gates::{
        specialize_gate, CNOTSpecialized, CPhaseSpecialized, CZSpecialized, FredkinSpecialized,
        HadamardSpecialized, PauliXSpecialized, PauliYSpecialized, PauliZSpecialized,
        PhaseSpecialized, RXSpecialized, RYSpecialized, RZSpecialized, SGateSpecialized,
        SWAPSpecialized, SpecializedGate, TGateSpecialized, ToffoliSpecialized,
    };
    pub use crate::specialized_simulator::{
        benchmark_specialization, SpecializationStats, SpecializedSimulatorConfig,
        SpecializedStateVectorSimulator,
    };
    pub use crate::stabilizer::{is_clifford_circuit, StabilizerGate, StabilizerSimulator};
    pub use crate::statevector::StateVectorSimulator;
    pub use crate::tpu_acceleration::{
        benchmark_tpu_acceleration, CommunicationBackend, DistributedContext, MemoryOptimization,
        TPUConfig, TPUDataType, TPUDeviceInfo, TPUDeviceType, TPUMemoryManager,
        TPUQuantumSimulator, TPUStats, TPUTensorBuffer, TPUTopology, XLAComputation,
    };
    pub use crate::trotter::{
        Hamiltonian, HamiltonianLibrary, HamiltonianTerm, TrotterDecomposer, TrotterMethod,
    };
    pub use crate::visualization_hooks::{
        benchmark_visualization, VisualizationConfig, VisualizationData, VisualizationFramework,
        VisualizationHook, VisualizationManager, ASCIIVisualizationHook, JSONVisualizationHook,
        ColorScheme, GateVisualizationData,
    };
    pub use crate::telemetry::{
        benchmark_telemetry, TelemetryCollector, TelemetryConfig, TelemetryMetric,
        TelemetryExportFormat, PerformanceSnapshot, QuantumMetrics, Alert, AlertLevel,
        AlertThresholds, MetricsSummary, NetworkIOStats, DiskIOStats,
    };

    #[cfg(feature = "gpu")]
    pub use crate::gpu_linalg::{benchmark_gpu_linalg, GpuLinearAlgebra};
    #[allow(unused_imports)]
    pub use crate::statevector::*;
    pub use crate::tensor::*;
    pub use crate::utils::*;
    pub use num_complex::Complex64;
}

/// A placeholder for future error correction code implementations
#[derive(Debug, Clone)]
pub struct ErrorCorrection;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub mod gpu_linalg;

#[cfg(feature = "advanced_math")]
pub use crate::tensor_network::*;

// Temporarily disabled features
// pub use crate::optimized::*;
// pub use crate::optimized_simulator::*;
