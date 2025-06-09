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
        QuantumAutoML, AutoMLConfig, AutoMLResult, AutoMLTaskType, AutoMLDataType, QuantumEncodingMethod,
        ModelSelectionStrategy, AutoMLEnsembleMethod, OptimizationObjective, QuantumSearchSpace,
        QuantumAlgorithm, PreprocessingMethod, ParameterRange, ArchitectureConstraints,
        ResourceConstraints, NoiseToleranceConfig, BudgetConfig, EvaluationConfig,
        QuantumMetric, QuantumAutoMLConfig, QuantumHardwareConstraints, ErrorMitigationConfig,
        QuantumAdvantageConfig, StatePreparationConfig, QuantumModel, QuantumEnsemble,
        PerformanceMetrics, SearchIteration, ModelConfiguration, ArchitectureConfiguration,
        LayerConfiguration, EnsembleResults, DiversityMetrics, QuantumAdvantageAnalysis,
        ResourceEfficiencyAnalysis, ScalingAnalysis, TheoreticalAdvantage, ResourceUsageSummary,
        EfficiencyMetrics, ModelExplanation, QuantumCircuitAnalysis, ParameterValue,
        create_default_automl_config, create_comprehensive_automl_config
    };
}
