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

pub mod error;
pub mod autodiff;
pub mod lstm;
pub mod attention;
pub mod gnn;
pub mod federated;
pub mod transfer;
pub mod few_shot;
pub mod continuous_rl;
pub mod diffusion;
pub mod boltzmann;
pub mod meta_learning;
pub mod quantum_nas;
pub mod adversarial;
pub mod continual_learning;
pub mod explainable_ai;
pub mod quantum_transformer;
pub mod quantum_llm;
pub mod computer_vision;
pub mod recommender;
pub mod time_series;
pub mod anomaly_detection;
pub mod clustering;
pub mod dimensionality_reduction;
pub mod automl;
pub mod scirs2_integration;
pub mod circuit_integration;
pub mod simulator_backends;
pub mod device_compilation;
pub mod benchmarking;
pub mod anneal_integration;
pub mod pytorch_api;
pub mod tensorflow_compatibility;
pub mod sklearn_compatibility;
pub mod keras_api;
pub mod onnx_export;
pub mod model_zoo;
pub mod domain_templates;
pub mod industry_examples;
pub mod classical_ml_integration;

// Internal utilities module
mod utils;

/// Re-export error types for easier access
pub use error::MLError;
pub use error::Result;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::blockchain::{ConsensusType, QuantumBlockchain, QuantumToken, SmartContract};
    pub use crate::classification::{ClassificationMetrics, Classifier};
    pub use crate::crypto::{
        ProtocolType, QuantumAuthentication, QuantumKeyDistribution, QuantumSignature,
    };
    pub use crate::error::{MLError, Result};
    pub use crate::gan::{Discriminator, GANEvaluationMetrics, Generator, QuantumGAN};
    pub use crate::hep::{
        AnomalyDetector, EventReconstructor, HEPQuantumClassifier, ParticleCollisionClassifier,
    };
    pub use crate::kernels::{KernelMethod, QuantumKernel};
    pub use crate::nlp::{NLPTaskType, QuantumLanguageModel, SentimentAnalyzer, TextSummarizer};
    pub use crate::optimization::{ObjectiveFunction, OptimizationMethod, Optimizer};
    pub use crate::qnn::{QNNBuilder, QNNLayer, QuantumNeuralNetwork};
    pub use crate::qsvm::{
        FeatureMapType, QSVMParams, QuantumKernel as QSVMKernel, QuantumKernelRidge, QSVM,
    };
    pub use crate::reinforcement::{Environment, QuantumAgent, ReinforcementLearning};
    pub use crate::variational::{VariationalAlgorithm, VariationalCircuit};
    pub use crate::transfer::{
        QuantumTransferLearning, TransferStrategy, PretrainedModel, 
        LayerConfig, QuantumModelZoo
    };
    pub use crate::few_shot::{
        FewShotLearner, FewShotMethod, Episode, QuantumPrototypicalNetwork,
        QuantumMAML, DistanceMetric
    };
    pub use crate::continuous_rl::{
        ContinuousEnvironment, QuantumDDPG, QuantumSAC, QuantumActor, QuantumCritic,
        ReplayBuffer, Experience as RLExperience, PendulumEnvironment
    };
    pub use crate::diffusion::{
        QuantumDiffusionModel, NoiseSchedule, QuantumScoreDiffusion,
        QuantumVariationalDiffusion
    };
    pub use crate::boltzmann::{
        QuantumBoltzmannMachine, QuantumRBM, DeepBoltzmannMachine,
        AnnealingSchedule
    };
    pub use crate::meta_learning::{
        QuantumMetaLearner, MetaLearningAlgorithm, MetaTask, MetaLearningHistory,
        ContinualMetaLearner, TaskGenerator
    };
    pub use crate::quantum_nas::{
        QuantumNAS, SearchStrategy, SearchSpace, ArchitectureCandidate, 
        ArchitectureMetrics, ArchitectureProperties, QubitConstraints,
        QuantumTopology, RLAgentType, AcquisitionFunction, create_default_search_space
    };
    pub use crate::adversarial::{
        QuantumAdversarialTrainer, QuantumAttackType, QuantumDefenseStrategy,
        QuantumAdversarialExample, AdversarialTrainingConfig, RobustnessMetrics,
        create_default_adversarial_config, create_comprehensive_defense
    };
    pub use crate::continual_learning::{
        QuantumContinualLearner, ContinualLearningStrategy, ContinualTask,
        TaskType, MemoryBuffer, Experience, TaskMetrics, ForgettingMetrics,
        MemorySelectionStrategy, ParameterAllocationStrategy,
        create_continual_task, generate_task_sequence
    };
    pub use crate::explainable_ai::{
        QuantumExplainableAI, ExplanationMethod, ExplanationResult,
        CircuitExplanation, QuantumStateProperties, AttributionMethod,
        PerturbationMethod, AggregationMethod, LocalModelType, LRPRule,
        create_default_xai_config
    };
    pub use crate::quantum_transformer::{
        QuantumTransformer, QuantumTransformerConfig, QuantumMultiHeadAttention,
        QuantumPositionEncoding, QuantumFeedForward, QuantumTransformerLayer,
        QuantumAttentionType, PositionEncodingType, ActivationType,
        AttentionOutput, QuantumAttentionInfo, create_causal_mask, create_padding_mask
    };
    pub use crate::quantum_llm::{
        QuantumLLM, QuantumLLMConfig, QuantumMemorySystem, QuantumReasoningModule,
        QuantumAssociativeMemory, QuantumAnalogyEngine, ModelScale, QuantumReasoningConfig,
        QuantumMemoryConfig, QLLMTrainingConfig, MemoryRetrievalType, QuantumParameterUpdate,
        GenerationConfig, GenerationStatistics, QualityMetrics, Vocabulary
    };
    pub use crate::computer_vision::{
        QuantumVisionPipeline, QuantumVisionConfig, ImageEncodingMethod, VisionBackbone,
        VisionTaskConfig, PreprocessingConfig, AugmentationConfig, ColorSpace,
        QuantumEnhancement, ResidualBlock, QuantumImageEncoder, QuantumFeatureExtractor,
        QuantumSpatialAttention, ImagePreprocessor, TaskOutput, TaskTarget,
        VisionMetrics, QuantumMetrics, ComputationalMetrics, TrainingHistory,
        ConvolutionalConfig, QuantumConvolutionalNN
    };
    pub use crate::recommender::{
        QuantumRecommender, QuantumRecommenderConfig, RecommendationAlgorithm,
        FeatureExtractionMethod, ProfileLearningMethod, QuantumEnhancementLevel,
        SimilarityMeasure, Recommendation, RecommendationExplanation, UserProfile,
        ItemFeatures, RecommendationOptions, BusinessRules
    };
    pub use crate::time_series::{
        QuantumTimeSeriesForecaster, QuantumTimeSeriesConfig, TimeSeriesModel,
        FeatureEngineeringConfig, SeasonalityConfig, QuantumEnhancementLevel as TSQuantumEnhancementLevel,
        EnsembleConfig, EnsembleMethod, DiversityStrategy, ForecastResult,
        AnomalyPoint, AnomalyType, ForecastMetrics, generate_synthetic_time_series
    };
    pub use crate::anomaly_detection::{
        QuantumAnomalyDetector, QuantumAnomalyConfig, AnomalyDetectionMethod,
        AnomalyResult, AnomalyMetrics, QuantumAnomalyMetrics, SpecializedDetectorConfig,
        TimeSeriesAnomalyDetector, QuantumStateAnomalyDetector, QuantumIsolationForest,
        QuantumAutoencoder, QuantumOneClassSVM, QuantumLOF, 
        PreprocessingConfig as AnomalyPreprocessingConfig,
        QuantumEnhancementConfig, RealtimeConfig, PerformanceConfig,
        create_default_anomaly_config, create_comprehensive_anomaly_config
    };
    pub use crate::clustering::{
        QuantumClusterer, ClusteringAlgorithm, QuantumDistanceMetric, 
        QuantumEnhancementLevel as ClusteringQuantumEnhancementLevel,
        QuantumKMeansConfig, QuantumHierarchicalConfig, QuantumDBSCANConfig,
        QuantumSpectralConfig, QuantumFuzzyCMeansConfig, QuantumGMMConfig,
        SpecializedClusteringConfig, GraphClusteringConfig, TimeSeriesClusteringConfig,
        HighDimClusteringConfig, StreamingClusteringConfig, QuantumNativeConfig,
        EnsembleConfig as ClusteringEnsembleConfig, ClusteringResult, ClusteringMetrics, QuantumClusteringMetrics,
        LinkageCriterion, AffinityType, CovarianceType, GraphMethod, CommunityAlgorithm,
        TimeSeriesDistanceMetric, DimensionalityReduction, StatePreparationMethod,
        MeasurementStrategy, EntanglementStructure, EnsembleCombinationMethod,
        create_default_quantum_kmeans, create_default_quantum_dbscan, 
        create_default_quantum_spectral
    };
    pub use crate::dimensionality_reduction::{
        QuantumDimensionalityReducer, DimensionalityReductionAlgorithm,
        QuantumDistanceMetric as DRQuantumDistanceMetric, QuantumEnhancementLevel as DRQuantumEnhancementLevel,
        QuantumEigensolver, QuantumFeatureMap, AutoencoderArchitecture,
        QPCAConfig, QICAConfig, QtSNEConfig, QUMAPConfig, QLDAConfig,
        QFactorAnalysisConfig, QCCAConfig, QNMFConfig, QAutoencoderConfig,
        QManifoldConfig, QKernelPCAConfig, QFeatureSelectionConfig, QSpecializedConfig,
        QTimeSeriesConfig, QImageTensorConfig, QGraphConfig, QStreamingConfig,
        DimensionalityReductionResult, QuantumDRMetrics, DRMetrics, ComputationalMetrics as DRComputationalMetrics,
        DRTrainedState, create_default_qpca_config, create_default_qica_config,
        create_default_qtsne_config, create_default_qautoencoder_config,
        create_comprehensive_qpca, create_comprehensive_qtsne
    };
    pub use crate::automl::{
        QuantumAutoML, QuantumAutoMLConfig, MLTaskType, QuantumEncodingMethod,
        OptimizationObjective, AutoMLResults, SearchBudgetConfig, QuantumConstraints,
        EvaluationConfig, AdvancedAutoMLFeatures, SearchSpaceConfig,
        AlgorithmSearchSpace, HyperparameterSearchSpace, EnsembleSearchSpace,
        QuantumMLPipeline, PerformanceSummary, QuantumAdvantageAnalysis
    };
    pub use crate::scirs2_integration::{
        SciRS2Array, SciRS2Tensor, SciRS2Optimizer, SciRS2DistributedTrainer, SciRS2Serializer
    };
    pub use crate::circuit_integration::{
        QuantumMLExecutor, QuantumLayer, ParameterizedLayer, RotationAxis,
        HardwareAwareCompiler, DeviceTopology, QubitProperties, BackendManager,
        MLCircuitOptimizer, OptimizationPass, MLCircuitAnalyzer,
        ExpressionvityMetrics, TrainabilityMetrics
    };
    pub use crate::simulator_backends::{
        SimulatorBackend, SimulationResult, Observable, GradientMethod, BackendCapabilities,
        StatevectorBackend, MPSBackend, BackendSelectionStrategy
    };
    pub use crate::device_compilation::{
        DeviceCompiler, CompilationOptions, RoutingAlgorithm, SynthesisMethod,
        DeviceCharacterization, QuantumMLModel, CompiledModel, QubitMapping, CompilationMetrics
    };
    pub use crate::benchmarking::{
        BenchmarkFramework, BenchmarkConfig, Benchmark, BenchmarkRunResult, ScalingType,
        BenchmarkCategory, BenchmarkResults, BenchmarkSummary, BenchmarkReport
    };
    pub use crate::anneal_integration::{
        QuantumMLQUBO, IsingProblem, QuantumMLAnnealer, AnnealingParams, AnnealingSchedule as MLAnnealingSchedule,
        AnnealingClient, AnnealingResult, QuantumMLOptimizationProblem, OptimizationResult,
        FeatureSelectionProblem, HyperparameterProblem, CircuitOptimizationProblem,
        PortfolioOptimizationProblem
    };
    pub use crate::pytorch_api::{
        QuantumModule, Parameter, QuantumLinear, QuantumConv2d, QuantumActivation, ActivationType as PyTorchActivationType,
        QuantumSequential, QuantumLoss, QuantumMSELoss, QuantumCrossEntropyLoss, QuantumTrainer,
        TrainingHistory as PyTorchTrainingHistory, DataLoader, MemoryDataLoader, InitType
    };
    pub use crate::tensorflow_compatibility::{
        QuantumCircuitLayer, PQCLayer, QuantumConvolutionalLayer, TFQModel, TFQLayer,
        ParameterInitStrategy, RegularizationType, PaddingType, TFQLossFunction, TFQOptimizer,
        QuantumDataset, QuantumDatasetIterator, TFQCircuitFormat, TFQGate, DataEncodingType,
        tfq_utils
    };
    pub use crate::sklearn_compatibility::{
        SklearnEstimator, SklearnClassifier, SklearnRegressor, SklearnClusterer,
        QuantumSVC, QuantumMLPClassifier, QuantumMLPRegressor, QuantumKMeans,
        model_selection, pipeline
    };
    pub use crate::keras_api::{
        KerasLayer, Dense, QuantumDense, Activation, Sequential, Input,
        ActivationFunction, QuantumAnsatzType, InitializerType, LossFunction,
        OptimizerType, MetricType, Callback, EarlyStopping, TrainingHistory as KerasTrainingHistory,
        ModelSummary, LayerInfo, DataType, utils as keras_utils
    };
    pub use crate::onnx_export::{
        ONNXGraph, ONNXNode, ONNXAttribute, ONNXValueInfo, ONNXDataType, ONNXTensor,
        ONNXExporter, ONNXImporter, ExportOptions, ImportOptions, QuantumBackendTarget,
        TargetFramework, UnsupportedOpHandling, ValidationReport, ModelInfo,
        utils as onnx_utils
    };
    pub use crate::model_zoo::{
        ModelZoo, ModelMetadata, ModelCategory, ModelRequirements, QuantumModel,
        TrainingConfig, MNISTQuantumNN, IrisQuantumSVM, H2VQE, PortfolioQAOA,
        QuantumAnomalyDetector, QuantumTimeSeriesForecaster, utils as model_zoo_utils
    };
    pub use crate::domain_templates::{
        DomainTemplateManager, Domain, TemplateMetadata, ProblemType, ModelComplexity,
        TemplateConfig, DomainModel, PortfolioOptimizationModel, CreditRiskModel,
        FraudDetectionModel, MolecularPropertyModel, DrugDiscoveryModel, MedicalImageModel,
        VehicleRoutingModel, SmartGridModel, MaterialPropertyModel, utils as domain_utils
    };
    pub use crate::industry_examples::{
        IndustryExampleManager, Industry, UseCase, DataRequirements, ImplementationComplexity,
        ROIEstimate, BenchmarkResult, PerformanceMetrics, QuantumAdvantageMetrics,
        ResourceRequirements, ExampleResult, BusinessImpact, ROISummary, utils as industry_utils
    };
    pub use crate::classical_ml_integration::{
        HybridPipelineManager, PipelineTemplate, PipelineStage, ModelType, PerformanceProfile,
        DataPreprocessor, ModelRegistry, ClassicalModel, HybridModel, EnsembleStrategy,
        HybridPipeline, PipelineConfig, ResourceConstraints, ValidationStrategy,
        DatasetInfo, PipelineRecommendation, AutoOptimizationConfig, OptimizedPipeline,
        StandardScaler, MinMaxScaler, WeightedVotingEnsemble, utils as pipeline_utils
    };
}
