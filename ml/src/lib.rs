#![recursion_limit = "8192"]
#![allow(warnings)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]
#![allow(clippy::restriction)]

//! # Quantum Machine Learning
//!
//! This crate provides quantum machine learning capabilities for the QuantRS2 framework.
//! It includes quantum neural networks, variational algorithms, and specialized tools for
//! high-energy physics data analysis, plus cutting-edge quantum ML algorithms.
//!
//! ## Core Features
//!
//! - Quantum Neural Networks
//! - Variational Quantum Algorithms
//! - High-Energy Physics Data Analysis
//! - Quantum Reinforcement Learning
//! - Quantum Generative Models
//! - Quantum Kernels for Classification
//! - Quantum-Enhanced Cryptographic Protocols
//! - Quantum Blockchain and Distributed Ledger Technology
//! - Quantum-Enhanced Natural Language Processing
//! - Quantum Anomaly Detection and Outlier Analysis
//!
//! ## Cutting-Edge Quantum ML Algorithms
//!
//! - **Quantum Neural ODEs**: Continuous-depth quantum neural networks using quantum circuits to parameterize derivative functions
//! - **Quantum Physics-Informed Neural Networks (QPINNs)**: Quantum neural networks that enforce physical laws and solve PDEs
//! - **Quantum Reservoir Computing**: Leverages quantum dynamics for temporal data processing with quantum advantages
//! - **Quantum Graph Attention Networks**: Combines graph neural networks with quantum attention mechanisms for complex graph analysis
//!
//! ## Recent Updates (v0.1.0-rc.1)
//!
//! - Refined SciRS2 v0.1.0-rc.1 integration with unified patterns
//! - Automatic differentiation leveraging SciRS2's linear algebra operations
//! - Parallel training with `scirs2_core::parallel_ops`
//! - SIMD-accelerated quantum kernel computations

use fastrand;
use std::error::Error;
use thiserror::Error;

pub mod barren_plateau;
pub mod blockchain;
pub mod classification;
pub mod crypto;
pub mod enhanced_gan;
pub mod gan;
pub mod hep;
pub mod kernels;
pub mod nlp;
pub mod optimization;
pub mod qcnn;
pub mod qnn;
pub mod qsvm;
pub mod reinforcement;
pub mod vae;
pub mod variational;

pub mod adversarial;
pub mod anneal_integration;
pub mod anomaly_detection;
pub mod attention;
pub mod autodiff;
pub mod automl;
pub mod benchmarking;
pub mod boltzmann;
pub mod circuit_integration;
pub mod classical_ml_integration;
pub mod clustering;
pub mod computer_vision;
pub mod continual_learning;
pub mod continuous_rl;
pub mod device_compilation;
pub mod diffusion;
pub mod dimensionality_reduction;
pub mod domain_templates;
pub mod error;
pub mod error_mitigation;
pub mod explainable_ai;
pub mod federated;
pub mod few_shot;
pub mod gnn;
#[cfg(feature = "gpu")]
pub mod gpu_backend_impl;
pub mod industry_examples;
pub mod keras_api;
pub mod lstm;
pub mod meta_learning;
pub mod model_zoo;
pub mod onnx_export;
pub mod performance_profiler;
pub mod pytorch_api;
pub mod quantum_advanced_diffusion;
pub mod quantum_advantage_validator;
pub mod quantum_continuous_flows;
pub mod quantum_graph_attention;
pub mod quantum_implicit_neural_representations;
pub mod quantum_in_context_learning;
pub mod quantum_llm;
pub mod quantum_memory_networks;
pub mod quantum_mixture_of_experts;
pub mod quantum_nas;
pub mod quantum_neural_odes;
pub mod quantum_neural_radiance_fields;
pub mod quantum_pinns;
pub mod quantum_reservoir_computing;
pub mod quantum_self_supervised_learning;
pub mod quantum_transformer;
pub mod recommender;
pub mod scirs2_integration;
pub mod simulator_backends;
pub mod sklearn_compatibility;
pub mod tensorflow_compatibility;
pub mod time_series;
pub mod transfer;
pub mod tutorials;

// Utilities module for calibration, metrics, preprocessing, etc.
pub mod utils;

// Advanced Quantum-Classical Hybrid AutoML Engine
pub mod hybrid_automl_engine;

/// Re-export error types for easier access
pub use error::MLError;
pub use error::Result;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::adversarial::{
        create_comprehensive_defense, create_default_adversarial_config, AdversarialTrainingConfig,
        QuantumAdversarialExample, QuantumAdversarialTrainer, QuantumAttackType,
        QuantumDefenseStrategy, RobustnessMetrics,
    };
    pub use crate::anneal_integration::{
        AnnealingClient, AnnealingParams, AnnealingResult,
        AnnealingSchedule as MLAnnealingSchedule, CircuitOptimizationProblem,
        FeatureSelectionProblem, HyperparameterProblem, IsingProblem, OptimizationResult,
        PortfolioOptimizationProblem, QuantumMLAnnealer, QuantumMLOptimizationProblem,
        QuantumMLQUBO,
    };
    pub use crate::anomaly_detection::{
        AnomalyDetectionMethod, AnomalyMetrics, AnomalyResult, PerformanceConfig,
        PreprocessingConfig as AnomalyPreprocessingConfig, QuantumAnomalyConfig,
        QuantumAnomalyDetector, QuantumAnomalyMetrics, QuantumAutoencoder,
        QuantumEnhancementConfig, QuantumIsolationForest, QuantumLOF, QuantumOneClassSVM,
        RealtimeConfig, SpecializedDetectorConfig,
    };
    pub use crate::automl::{
        create_comprehensive_automl_config, create_default_automl_config, AdvancedAutoMLFeatures,
        AlgorithmSearchSpace, EnsembleSearchSpace, EvaluationConfig, HyperparameterSearchSpace,
        MLTaskType, OptimizationObjective, QuantumAutoML, QuantumAutoMLConfig, QuantumConstraints,
        QuantumEncodingMethod, SearchBudgetConfig, SearchSpaceConfig,
    };
    pub use crate::benchmarking::{
        Benchmark, BenchmarkCategory, BenchmarkConfig, BenchmarkFramework, BenchmarkReport,
        BenchmarkResults, BenchmarkRunResult, BenchmarkSummary, ScalingType,
    };
    pub use crate::blockchain::{ConsensusType, QuantumBlockchain, QuantumToken, SmartContract};
    pub use crate::boltzmann::{
        AnnealingSchedule, DeepBoltzmannMachine, QuantumBoltzmannMachine, QuantumRBM,
    };
    pub use crate::circuit_integration::{
        BackendManager, DeviceTopology, ExpressionvityMetrics, HardwareAwareCompiler,
        MLCircuitAnalyzer, MLCircuitOptimizer, OptimizationPass, ParameterizedLayer, QuantumLayer,
        QuantumMLExecutor, QubitProperties, RotationAxis, TrainabilityMetrics,
    };
    pub use crate::classical_ml_integration::{
        utils as pipeline_utils, AutoOptimizationConfig, ClassicalModel, DataPreprocessor,
        DatasetInfo, EnsembleStrategy, HybridModel, HybridPipeline, HybridPipelineManager,
        MinMaxScaler, ModelRegistry, ModelType, OptimizedPipeline, PerformanceProfile,
        PipelineConfig, PipelineRecommendation, PipelineStage, PipelineTemplate,
        ResourceConstraints, StandardScaler, ValidationStrategy, WeightedVotingEnsemble,
    };
    pub use crate::classification::{ClassificationMetrics, Classifier};
    pub use crate::clustering::{
        ClusteringAlgorithm, CoreClusteringMetrics as QuantumClusteringMetrics, QuantumClusterer,
        QuantumClusteringConfig,
    };
    pub use crate::computer_vision::{
        AugmentationConfig, ColorSpace, ComputationalMetrics, ConvolutionalConfig,
        ImageEncodingMethod, ImagePreprocessor, PreprocessingConfig, QuantumConvolutionalNN,
        QuantumEnhancement, QuantumFeatureExtractor, QuantumImageEncoder, QuantumMetrics,
        QuantumSpatialAttention, QuantumVisionConfig, QuantumVisionPipeline, ResidualBlock,
        TaskOutput, TaskTarget, TrainingHistory, VisionBackbone, VisionMetrics, VisionTaskConfig,
    };
    pub use crate::continual_learning::{
        create_continual_task, generate_task_sequence, ContinualLearningStrategy, ContinualTask,
        Experience, ForgettingMetrics, MemoryBuffer, MemorySelectionStrategy,
        ParameterAllocationStrategy, QuantumContinualLearner, TaskMetrics, TaskType,
    };
    pub use crate::continuous_rl::{
        ContinuousEnvironment, Experience as RLExperience, PendulumEnvironment, QuantumActor,
        QuantumCritic, QuantumDDPG, QuantumSAC, ReplayBuffer,
    };
    pub use crate::crypto::{
        ProtocolType, QuantumAuthentication, QuantumKeyDistribution, QuantumSignature,
    };
    pub use crate::device_compilation::{
        CompilationMetrics, CompilationOptions, CompiledModel, DeviceCharacterization,
        DeviceCompiler, QuantumMLModel, QubitMapping, RoutingAlgorithm, SynthesisMethod,
    };
    pub use crate::diffusion::{
        NoiseSchedule, QuantumDiffusionModel, QuantumScoreDiffusion, QuantumVariationalDiffusion,
    };
    pub use crate::dimensionality_reduction::{
        AutoencoderArchitecture, DRTrainedState, DimensionalityReductionAlgorithm,
        DimensionalityReductionMetrics, ManifoldMetrics, QAutoencoderConfig, QCCAConfig,
        QFactorAnalysisConfig, QFeatureSelectionConfig, QICAConfig, QKernelPCAConfig, QLDAConfig,
        QManifoldConfig, QNMFConfig, QPCAConfig, QSpecializedConfig, QUMAPConfig, QtSNEConfig,
        QuantumDimensionalityReducer, QuantumDistanceMetric as DRQuantumDistanceMetric,
        QuantumEigensolver, QuantumEnhancementLevel as DRQuantumEnhancementLevel,
        QuantumFeatureMap, ReconstructionMetrics,
    };
    pub use crate::domain_templates::{
        utils as domain_utils, CreditRiskModel, Domain, DomainModel, DomainTemplateManager,
        DrugDiscoveryModel, FraudDetectionModel, MaterialPropertyModel, MedicalImageModel,
        ModelComplexity, MolecularPropertyModel, PortfolioOptimizationModel, ProblemType,
        SmartGridModel, TemplateConfig, TemplateMetadata, VehicleRoutingModel,
    };
    pub use crate::error::{MLError, Result};
    pub use crate::error_mitigation::{
        AdaptiveConfig, CalibrationData, CircuitFoldingMethod, CoherenceTimeModel, ErrorType,
        ExtrapolationMethod, GateErrorModel, MeasurementErrorModel, MitigatedInferenceData,
        MitigatedTrainingData, MitigationStrategy, NoiseModel, QuantumMLErrorMitigator,
        ReadoutCorrectionMethod,
    };
    pub use crate::explainable_ai::{
        create_default_xai_config, AggregationMethod, AttributionMethod, CircuitExplanation,
        ExplanationMethod, ExplanationResult, LRPRule, LocalModelType, PerturbationMethod,
        QuantumExplainableAI, QuantumStateProperties,
    };
    pub use crate::few_shot::{
        DistanceMetric, Episode, FewShotLearner, FewShotMethod, QuantumMAML,
        QuantumPrototypicalNetwork,
    };
    pub use crate::gan::{Discriminator, GANEvaluationMetrics, Generator, QuantumGAN};
    pub use crate::hep::{
        AnomalyDetector, EventReconstructor, HEPQuantumClassifier, ParticleCollisionClassifier,
    };
    pub use crate::industry_examples::{
        utils as industry_utils, BenchmarkResult, BusinessImpact, DataRequirements, ExampleResult,
        ImplementationComplexity, Industry, IndustryExampleManager, PerformanceMetrics,
        QuantumAdvantageMetrics, ROIEstimate, ROISummary, ResourceRequirements, UseCase,
    };
    pub use crate::keras_api::{
        utils as keras_utils, Activation, ActivationFunction, Callback, DataType, Dense,
        EarlyStopping, InitializerType, Input, KerasLayer, LayerInfo, LossFunction, MetricType,
        ModelSummary, OptimizerType, QuantumAnsatzType, QuantumDense, Sequential,
        TrainingHistory as KerasTrainingHistory,
    };
    pub use crate::kernels::{KernelMethod, QuantumKernel};
    pub use crate::meta_learning::{
        ContinualMetaLearner, MetaLearningAlgorithm, MetaLearningHistory, MetaTask,
        QuantumMetaLearner, TaskGenerator,
    };
    pub use crate::model_zoo::{
        utils as model_zoo_utils, IrisQuantumSVM, MNISTQuantumNN, ModelCategory, ModelMetadata,
        ModelRequirements, ModelZoo, PortfolioQAOA, QuantumModel, TrainingConfig, H2VQE,
    };
    pub use crate::nlp::{NLPTaskType, QuantumLanguageModel, SentimentAnalyzer, TextSummarizer};
    pub use crate::onnx_export::{
        utils as onnx_utils, ExportOptions, ImportOptions, ModelInfo, ONNXAttribute, ONNXDataType,
        ONNXExporter, ONNXGraph, ONNXImporter, ONNXNode, ONNXTensor, ONNXValueInfo,
        QuantumBackendTarget, TargetFramework, UnsupportedOpHandling, ValidationReport,
    };
    pub use crate::optimization::{ObjectiveFunction, OptimizationMethod, Optimizer};
    pub use crate::performance_profiler::{
        Bottleneck, BottleneckSeverity, CircuitMetrics, MemorySnapshot, MemoryStats,
        OperationStats, ProfilerConfig, ProfilingReport, QuantumMLProfiler,
    };
    pub use crate::pytorch_api::{
        ActivationType as PyTorchActivationType, DataLoader, InitType, MemoryDataLoader, Parameter,
        QuantumActivation, QuantumConv2d, QuantumCrossEntropyLoss, QuantumLinear, QuantumLoss,
        QuantumMSELoss, QuantumModule, QuantumSequential, QuantumTrainer,
        TrainingHistory as PyTorchTrainingHistory,
    };
    pub use crate::qnn::{QNNBuilder, QNNLayer, QuantumNeuralNetwork};
    pub use crate::qsvm::{
        FeatureMapType, QSVMParams, QuantumKernel as QSVMKernel, QuantumKernelRidge, QSVM,
    };
    pub use crate::quantum_llm::{
        GenerationConfig, GenerationStatistics, MemoryRetrievalType, ModelScale,
        QLLMTrainingConfig, QualityMetrics, QuantumAnalogyEngine, QuantumAssociativeMemory,
        QuantumLLM, QuantumLLMConfig, QuantumMemoryConfig, QuantumMemorySystem,
        QuantumParameterUpdate, QuantumReasoningConfig, QuantumReasoningModule, Vocabulary,
    };
    pub use crate::quantum_nas::{
        create_default_search_space, AcquisitionFunction, ArchitectureCandidate,
        ArchitectureMetrics, ArchitectureProperties, QuantumNAS, QuantumTopology, QubitConstraints,
        RLAgentType, SearchSpace, SearchStrategy,
    };
    pub use crate::quantum_transformer::{
        create_causal_mask, create_padding_mask, ActivationType, AttentionOutput,
        PositionEncodingType, QuantumAttentionInfo, QuantumAttentionType, QuantumFeedForward,
        QuantumMultiHeadAttention, QuantumPositionEncoding, QuantumTransformer,
        QuantumTransformerConfig, QuantumTransformerLayer,
    };
    pub use crate::recommender::{
        BusinessRules, FeatureExtractionMethod, ItemFeatures, ProfileLearningMethod,
        QuantumEnhancementLevel, QuantumRecommender, QuantumRecommenderConfig, Recommendation,
        RecommendationAlgorithm, RecommendationExplanation, RecommendationOptions,
        SimilarityMeasure, UserProfile,
    };
    pub use crate::reinforcement::{Environment, QuantumAgent, ReinforcementLearning};
    pub use crate::scirs2_integration::{
        SciRS2Array, SciRS2DistributedTrainer, SciRS2Optimizer, SciRS2Serializer, SciRS2Tensor,
    };
    pub use crate::simulator_backends::{
        BackendCapabilities, BackendSelectionStrategy, GradientMethod, MPSBackend, Observable,
        SimulationResult, SimulatorBackend, StatevectorBackend,
    };
    pub use crate::sklearn_compatibility::{
        model_selection, pipeline, QuantumKMeans, QuantumMLPClassifier, QuantumMLPRegressor,
        QuantumSVC, SklearnClassifier, SklearnClusterer, SklearnEstimator, SklearnRegressor,
    };
    pub use crate::tensorflow_compatibility::{
        tfq_utils, DataEncodingType, PQCLayer, PaddingType, ParameterInitStrategy,
        QuantumCircuitLayer, QuantumConvolutionalLayer, QuantumDataset, QuantumDatasetIterator,
        RegularizationType, TFQCircuitFormat, TFQGate, TFQLayer, TFQLossFunction, TFQModel,
        TFQOptimizer,
    };
    pub use crate::time_series::{
        generate_synthetic_time_series, AnomalyPoint, AnomalyType, DiversityStrategy,
        EnsembleConfig, EnsembleMethod, FeatureEngineeringConfig, ForecastMetrics, ForecastResult,
        QuantumEnhancementLevel as TSQuantumEnhancementLevel, QuantumTimeSeriesConfig,
        QuantumTimeSeriesForecaster, SeasonalityConfig, TimeSeriesModel,
    };
    pub use crate::transfer::{
        LayerConfig, PretrainedModel, QuantumModelZoo, QuantumTransferLearning, TransferStrategy,
    };
    pub use crate::tutorials::{
        utils as tutorial_utils, CodeExample, DifficultyLevel, Exercise, ExerciseResult,
        ExerciseType, ExperienceLevel, InteractiveElement, InteractiveType, TestCase, Tutorial,
        TutorialCategory, TutorialManager, TutorialProgress, TutorialSection, TutorialSession,
        UserBackground,
    };
    pub use crate::variational::{VariationalAlgorithm, VariationalCircuit};

    // New cutting-edge quantum ML algorithms
    pub use crate::quantum_graph_attention::{
        AttentionAnalysis, AttentionConfig as QGATAttentionConfig,
        BenchmarkResults as QGATBenchmarkResults, Graph, PoolingConfig, QGATConfig,
        QuantumAttentionType as QGATQuantumAttentionType, QuantumGraphAttentionNetwork,
        TrainingMetrics as QGATTrainingMetrics,
    };
    pub use crate::quantum_in_context_learning::{
        AdaptationResult, AdaptationStrategy, AdaptationTarget, ContextExample, ContextMetadata,
        ContextModality, ContextRetrievalMethod, EntanglementPattern, InContextLearningMetrics,
        InContextLearningOutput, InContextLearningStatistics, InterpolationMethod,
        MetaUpdateStrategy, QuantumAttentionMechanism, QuantumContextAttention,
        QuantumContextEncoder, QuantumContextEncoding, QuantumContextState, QuantumDistanceMetric,
        QuantumEpisodicMemory, QuantumInContextLearner, QuantumInContextLearningConfig,
        QuantumTaskAdapter, TransferLearningResults,
    };
    pub use crate::quantum_memory_networks::{
        AddressingConfig, AddressingType, BenchmarkResults as QMANBenchmarkResults,
        ControllerArchitecture, ControllerConfig as QMANControllerConfig, EpisodicMemory,
        HeadConfig, HeadType as QMANHeadType, MemoryInitialization, QMANConfig, QMANTrainingConfig,
        QuantumMemoryAugmentedNetwork, ReadParams, TrainingMetrics as QMANTrainingMetrics,
        WriteParams,
    };
    pub use crate::quantum_neural_odes::{
        AnsatzType as QNODEAnsatzType, BenchmarkResults as QNODEBenchmarkResults,
        IntegrationMethod, OptimizationStrategy as QNODEOptimizationStrategy, QNODEConfig,
        QuantumNeuralODE, TrainingMetrics as QNODETrainingMetrics,
    };
    pub use crate::quantum_pinns::{
        BoundaryCondition, DerivativeResults, InitialCondition, LossWeights, PhysicsEquationType,
        QPINNConfig, QuantumPINN, TrainingMetrics as QPINNTrainingMetrics,
    };
    pub use crate::quantum_reservoir_computing::{
        BenchmarkResults as QRCBenchmarkResults, DynamicsAnalysis, InputEncoding, QRCConfig,
        QuantumReservoirComputer, ReadoutConfig, ReservoirDynamics,
        TrainingMetrics as QRCTrainingMetrics,
    };

    pub use crate::quantum_advanced_diffusion::{
        DenoisingArchitecture, ErrorMitigationStrategy, GenerationMetrics,
        QuantumAdvancedDiffusionConfig, QuantumAdvancedDiffusionModel, QuantumGenerationOutput,
        QuantumNoiseSchedule, QuantumTrainingConfig,
    };

    pub use crate::quantum_advantage_validator::{
        ClassicalBaseline, ComparisonMetric, QuantumAdvantage, QuantumAdvantageValidator,
        QuantumResourceUsage, QuantumResult, ResourceUsage, StatisticalSignificance,
        ValidationConfig, ValidationReport as AdvantageValidationReport,
    };

    pub use crate::quantum_continuous_flows::{
        FlowArchitecture, FlowSamplingOutput, FlowTrainingConfig, QuantumContinuousFlow,
        QuantumContinuousFlowConfig, QuantumODEFunction,
    };

    pub use crate::quantum_neural_radiance_fields::{
        QuantumNeRF, QuantumNeRFConfig, QuantumRenderOutput, QuantumRenderingMetrics,
    };

    pub use crate::quantum_mixture_of_experts::{
        InterferencePattern, MoEOutput, MoEStatistics, MoETrainingConfig,
        QuantumCombinationMetrics, QuantumGatingMechanism, QuantumMixtureOfExperts,
        QuantumMixtureOfExpertsConfig, QuantumRoutingStrategy,
    };

    pub use crate::quantum_self_supervised_learning::{
        ContrastiveLossFunction, QuantumAugmentationStrategy, QuantumAugmenter, QuantumDecoder,
        QuantumEncoder, QuantumMaskingStrategy, QuantumProjector, QuantumSSLMethod,
        QuantumSSLMetrics, QuantumSelfSupervisedConfig, QuantumSelfSupervisedLearner,
        QuantumSimilarityMetric, RepresentationEvaluationResults, SSLLearningOutput,
        SSLTrainingConfig,
    };

    pub use crate::quantum_implicit_neural_representations::{
        AdaptationOutput, CompressedRepresentation, CompressionConfig, CompressionManager,
        EntanglementManager, INRQueryOutput, INRTrainingConfig, INRTrainingOutput,
        MetaLearningConfig, OptimizationConfig, QuantumActivationConfig, QuantumGradientEstimator,
        QuantumINRConfig, QuantumINRMetrics, QuantumImplicitNeuralRepresentation,
        QuantumLayerConfig, QuantumOptimizer, QuantumPositionalEncoding, QuantumStateManager,
        RepresentationMethod, SignalType,
    };
}
