//! Configuration types for SciRS2 mapping

use super::*;

/// Advanced mapping configuration using SciRS2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SciRS2MappingConfig {
    /// Graph algorithm to use for initial mapping
    pub initial_mapping_algorithm: InitialMappingAlgorithm,
    /// Routing algorithm for dynamic remapping
    pub routing_algorithm: SciRS2RoutingAlgorithm,
    /// Optimization objective
    pub optimization_objective: OptimizationObjective,
    /// Community detection method for clustering
    pub community_method: CommunityMethod,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Enable spectral analysis
    pub enable_spectral_analysis: bool,
    /// Enable centrality-based optimization
    pub enable_centrality_optimization: bool,
    /// Enable machine learning predictions
    pub enable_ml_predictions: bool,
    /// Parallel processing options
    pub parallel_config: ParallelConfig,
    /// Real-time adaptive mapping configuration
    pub adaptive_config: AdaptiveMappingConfig,
    /// Machine learning configuration
    pub ml_config: MLMappingConfig,
    /// Performance analytics configuration
    pub analytics_config: MappingAnalyticsConfig,
    /// Advanced optimization configuration
    pub advanced_optimization: AdvancedOptimizationConfig,
}

/// Parallel processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Number of threads to use
    pub num_threads: usize,
    /// Enable parallel graph analysis
    pub enable_parallel_analysis: bool,
    /// Enable parallel optimization
    pub enable_parallel_optimization: bool,
}

/// Adaptive mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveMappingConfig {
    /// Enable real-time adaptation
    pub enable_adaptation: bool,
    /// Adaptation trigger threshold
    pub adaptation_threshold: f64,
    /// Learning rate for adaptation
    pub learning_rate: f64,
    /// Memory size for online learning
    pub memory_size: usize,
    /// Feedback method
    pub feedback_method: FeedbackMethod,
    /// Online learning configuration
    pub online_learning: OnlineLearningConfig,
}

/// Feedback methods for adaptation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeedbackMethod {
    PerformanceBased,
    ErrorRateBased,
    LatencyBased,
    QualityBased,
    Hybrid,
}

/// Online learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineLearningConfig {
    /// Enable online learning
    pub enabled: bool,
    /// Learning algorithm
    pub algorithm: OnlineLearningAlgorithm,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size for mini-batch learning
    pub batch_size: usize,
    /// Experience replay buffer size
    pub replay_buffer_size: usize,
}

/// Online learning algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OnlineLearningAlgorithm {
    StochasticGradientDescent,
    AdaptiveGradient,
    RMSProp,
    Adam,
    OnlineRandomForest,
    IncrementalSVM,
    ReinforcementLearning,
}

impl Default for SciRS2MappingConfig {
    fn default() -> Self {
        Self {
            initial_mapping_algorithm: InitialMappingAlgorithm::SpectralEmbedding,
            routing_algorithm: SciRS2RoutingAlgorithm::AStarEnhanced,
            optimization_objective: OptimizationObjective::MinimizeSwaps,
            community_method: CommunityMethod::Louvain,
            max_iterations: 1000,
            tolerance: 1e-6,
            enable_spectral_analysis: true,
            enable_centrality_optimization: true,
            enable_ml_predictions: true,
            parallel_config: ParallelConfig {
                num_threads: num_cpus::get(),
                enable_parallel_analysis: true,
                enable_parallel_optimization: true,
            },
            adaptive_config: AdaptiveMappingConfig {
                enable_adaptation: true,
                adaptation_threshold: 0.1,
                learning_rate: 0.01,
                memory_size: 1000,
                feedback_method: FeedbackMethod::PerformanceBased,
                online_learning: OnlineLearningConfig {
                    enabled: true,
                    algorithm: OnlineLearningAlgorithm::Adam,
                    learning_rate: 0.001,
                    batch_size: 32,
                    replay_buffer_size: 10000,
                },
            },
            ml_config: MLMappingConfig {
                enable_ml: true,
                model_types: vec![], // Use empty for now
                feature_config: FeatureConfig {
                    enable_structural: true,
                    enable_temporal: true,
                    enable_hardware: true,
                    enable_circuit: true,
                    selection_method: FeatureSelectionMethod::VarianceThreshold { threshold: 0.01 },
                    max_features: 100,
                },
                training_config: TrainingConfig {
                    batch_size: 64,
                    epochs: 100,
                    learning_rate: 0.001,
                    validation_split: 0.2,
                    early_stopping_patience: 10,
                    regularization: RegularizationParams {
                        l1_lambda: 0.01,
                        l2_lambda: 0.01,
                        dropout: 0.5,
                        batch_norm: true,
                    },
                },
                prediction_config: PredictionConfig {
                    ensemble_size: 5,
                    confidence_threshold: 0.8,
                    use_uncertainty_estimation: true,
                    monte_carlo_samples: 100,
                    temperature_scaling: false,
                    calibration_method: CalibrationMethod::PlattScaling,
                },
                transfer_learning: TransferLearningConfig {
                    enable_transfer: true,
                    source_domains: vec!["general_circuits".to_string()],
                    adaptation_method: DomainAdaptationMethod::FineTuning,
                    fine_tuning: FineTuningConfig {
                        freeze_layers: vec![0, 1],
                        unfreeze_after_epochs: 20,
                        reduced_learning_rate: 0.0001,
                    },
                },
            },
            analytics_config: MappingAnalyticsConfig {
                enable_analytics: true,
                tracking_level: AnalysisDepth::Comprehensive,
                metrics_to_track: vec![
                    TrackingMetric::ExecutionTime,
                    TrackingMetric::MemoryUsage,
                    TrackingMetric::MappingQuality,
                    TrackingMetric::SwapCount,
                    TrackingMetric::FidelityLoss,
                ],
                anomaly_detection: AnomalyDetectionConfig {
                    enable_detection: true,
                    detection_method: AnomalyDetectionMethod::IsolationForest,
                    threshold: 0.1,
                    window_size: 100,
                },
                alerting: AlertConfig {
                    enable_alerts: true,
                    severity_threshold: 0.5,
                    notification_methods: vec![NotificationMethod::Log],
                    cooldown_period: Duration::from_secs(300),
                },
                reporting: ReportingConfig {
                    enable_reporting: true,
                    report_frequency: Duration::from_secs(3600),
                    report_format: ReportFormat::JSON,
                    content_config: ReportContentConfig {
                        include_performance_metrics: true,
                        include_trend_analysis: true,
                        include_recommendations: true,
                        include_visualizations: false,
                    },
                },
            },
            advanced_optimization: AdvancedOptimizationConfig {
                enable_advanced: true,
                multi_objective: MultiObjectiveConfig {
                    enable_multi_objective: true,
                    objectives: vec![
                        OptimizationObjective::MinimizeSwaps,
                        OptimizationObjective::MinimizeDepth,
                        OptimizationObjective::MaximizeFidelity,
                    ],
                    pareto_config: ParetoConfig {
                        population_size: 100,
                        generations: 50,
                        crossover_rate: 0.8,
                        mutation_rate: 0.1,
                        selection_method: SelectionMethod::Tournament { size: 3 },
                        scalarization: ScalarizationMethod::WeightedSum {
                            weights: vec![0.4, 0.3, 0.3],
                        },
                    },
                },
                constraint_handling: ConstraintHandlingConfig {
                    enable_constraints: true,
                    constraint_types: vec![
                        ConstraintType::ConnectivityConstraint,
                        ConstraintType::TimingConstraint,
                        ConstraintType::ResourceConstraint,
                    ],
                    penalty_method: PenaltyMethod::AdaptivePenalty {
                        initial_penalty: 1.0,
                    },
                    tolerance: 1e-6,
                },
                search_strategy: SearchStrategyConfig {
                    strategy: SearchStrategy::HybridSearch,
                    hybrid_config: HybridSearchConfig {
                        strategies: vec![
                            SearchStrategy::GeneticAlgorithm,
                            SearchStrategy::SimulatedAnnealing,
                            SearchStrategy::ParticleSwarm,
                        ],
                        switching_criteria: SwitchingCriteria {
                            performance_threshold: 0.05,
                            stagnation_limit: 50,
                            time_limit: Duration::from_secs(300),
                        },
                    },
                    budget: SearchBudgetConfig {
                        max_evaluations: 10000,
                        max_time: Duration::from_secs(600),
                        target_quality: 0.95,
                    },
                },
                parallel_optimization: ParallelOptimizationConfig {
                    enable_parallel: true,
                    num_workers: num_cpus::get(),
                    load_balancing: LoadBalancingStrategy::RoundRobin,
                    synchronization: SynchronizationMethod::Asynchronous,
                },
            },
        }
    }
}
