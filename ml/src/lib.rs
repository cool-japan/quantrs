#![recursion_limit = "8192"]

//! # Quantum Machine Learning
//!
//! This crate provides quantum machine learning capabilities for the QuantRS2 framework.
//! It includes quantum neural networks, variational algorithms, and specialized tools for
//! high-energy physics data analysis.
//!
//! ## Features
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
pub mod industry_examples;
pub mod keras_api;
pub mod lstm;
pub mod meta_learning;
pub mod model_zoo;
pub mod onnx_export;
pub mod pytorch_api;
pub mod quantum_llm;
pub mod quantum_nas;
pub mod quantum_transformer;
pub mod recommender;
pub mod scirs2_integration;
pub mod simulator_backends;
pub mod sklearn_compatibility;
pub mod tensorflow_compatibility;
pub mod time_series;
pub mod transfer;
pub mod tutorials;

// Internal utilities module
mod utils;

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
}
