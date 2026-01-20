//! Search Space Configuration
//!
//! This module defines search spaces for algorithms, preprocessing, hyperparameters,
//! architectures, and ensembles.

use std::collections::HashMap;

/// Search space configuration
#[derive(Debug, Clone)]
pub struct SearchSpaceConfig {
    /// Algorithm search space
    pub algorithms: AlgorithmSearchSpace,

    /// Preprocessing search space
    pub preprocessing: PreprocessingSearchSpace,

    /// Hyperparameter search space
    pub hyperparameters: HyperparameterSearchSpace,

    /// Architecture search space
    pub architectures: ArchitectureSearchSpace,

    /// Ensemble search space
    pub ensembles: EnsembleSearchSpace,
}

/// Algorithm search space
#[derive(Debug, Clone)]
pub struct AlgorithmSearchSpace {
    /// Quantum neural networks
    pub quantum_neural_networks: bool,

    /// Quantum support vector machines
    pub quantum_svm: bool,

    /// Quantum clustering algorithms
    pub quantum_clustering: bool,

    /// Quantum dimensionality reduction
    pub quantum_dim_reduction: bool,

    /// Quantum time series models
    pub quantum_time_series: bool,

    /// Quantum anomaly detection
    pub quantum_anomaly_detection: bool,

    /// Classical algorithms for comparison
    pub classical_algorithms: bool,
}

/// Preprocessing search space
#[derive(Debug, Clone)]
pub struct PreprocessingSearchSpace {
    /// Feature scaling methods
    pub scaling_methods: Vec<ScalingMethod>,

    /// Feature selection methods
    pub feature_selection: Vec<FeatureSelectionMethod>,

    /// Quantum encoding methods
    pub quantum_encodings: Vec<QuantumEncodingMethod>,

    /// Data augmentation
    pub data_augmentation: bool,

    /// Missing value handling
    pub missing_value_handling: Vec<MissingValueMethod>,
}

/// Feature scaling methods
#[derive(Debug, Clone)]
pub enum ScalingMethod {
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileScaler,
    QuantumScaler,
    NoScaling,
}

/// Feature selection methods
#[derive(Debug, Clone)]
pub enum FeatureSelectionMethod {
    VarianceThreshold { threshold: f64 },
    UnivariateSelection { k: usize },
    RecursiveFeatureElimination { n_features: usize },
    QuantumFeatureSelection { method: String },
    PrincipalComponentAnalysis { n_components: usize },
    QuantumPCA { n_components: usize },
}

/// Quantum encoding methods
#[derive(Debug, Clone)]
pub enum QuantumEncodingMethod {
    AmplitudeEncoding,
    AngleEncoding,
    BasisEncoding,
    QuantumFeatureMap { map_type: String },
    VariationalEncoding { layers: usize },
    AutomaticEncoding,
}

/// Missing value handling methods
#[derive(Debug, Clone)]
pub enum MissingValueMethod {
    DropRows,
    DropColumns,
    MeanImputation,
    MedianImputation,
    ModeImputation,
    QuantumImputation,
    KNNImputation { k: usize },
}

/// Hyperparameter search space
#[derive(Debug, Clone)]
pub struct HyperparameterSearchSpace {
    /// Learning rates
    pub learning_rates: (f64, f64),

    /// Regularization strengths
    pub regularization: (f64, f64),

    /// Batch sizes
    pub batch_sizes: Vec<usize>,

    /// Number of epochs
    pub epochs: (usize, usize),

    /// Quantum-specific parameters
    pub quantum_params: QuantumHyperparameterSpace,
}

/// Quantum hyperparameter search space
#[derive(Debug, Clone)]
pub struct QuantumHyperparameterSpace {
    /// Number of qubits range
    pub num_qubits: (usize, usize),

    /// Circuit depth range
    pub circuit_depth: (usize, usize),

    /// Entanglement strengths
    pub entanglement_strength: (f64, f64),

    /// Variational parameters
    pub variational_params: (f64, f64),

    /// Measurement strategies
    pub measurement_strategies: Vec<String>,
}

/// Architecture search space
#[derive(Debug, Clone)]
pub struct ArchitectureSearchSpace {
    /// Network architectures
    pub network_architectures: Vec<NetworkArchitecture>,

    /// Quantum circuit architectures
    pub quantum_architectures: Vec<QuantumArchitecture>,

    /// Hybrid architectures
    pub hybrid_architectures: bool,

    /// Architecture generation strategy
    pub generation_strategy: ArchitectureGenerationStrategy,
}

/// Network architecture templates
#[derive(Debug, Clone)]
pub enum NetworkArchitecture {
    MLP {
        hidden_layers: Vec<usize>,
    },
    CNN {
        conv_layers: Vec<ConvLayer>,
        fc_layers: Vec<usize>,
    },
    RNN {
        rnn_type: String,
        hidden_size: usize,
        num_layers: usize,
    },
    Transformer {
        num_heads: usize,
        hidden_dim: usize,
        num_layers: usize,
    },
    Autoencoder {
        encoder_layers: Vec<usize>,
        decoder_layers: Vec<usize>,
    },
}

/// Convolutional layer configuration
#[derive(Debug, Clone)]
pub struct ConvLayer {
    pub filters: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

/// Quantum architecture templates
#[derive(Debug, Clone)]
pub enum QuantumArchitecture {
    VariationalCircuit {
        layers: Vec<String>,
        depth: usize,
    },
    QuantumConvolutional {
        pooling: String,
        layers: usize,
    },
    QuantumRNN {
        quantum_cells: usize,
        classical_layers: usize,
    },
    HardwareEfficient {
        connectivity: String,
        repetitions: usize,
    },
    ProblemInspired {
        problem_type: String,
        ansatz: String,
    },
}

/// Architecture generation strategy
#[derive(Debug, Clone)]
pub enum ArchitectureGenerationStrategy {
    Random,
    Evolutionary,
    GradientBased,
    BayesianOptimization,
    QuantumInspired,
    Reinforcement,
}

/// Ensemble search space
#[derive(Debug, Clone)]
pub struct EnsembleSearchSpace {
    /// Enable ensemble methods
    pub enabled: bool,

    /// Maximum ensemble size
    pub max_ensemble_size: usize,

    /// Ensemble combination methods
    pub combination_methods: Vec<EnsembleCombinationMethod>,

    /// Diversity strategies
    pub diversity_strategies: Vec<EnsembleDiversityStrategy>,
}

/// Ensemble combination methods
#[derive(Debug, Clone)]
pub enum EnsembleCombinationMethod {
    Voting,
    Averaging,
    WeightedAveraging,
    Stacking,
    Blending,
    QuantumSuperposition,
    BayesianModelAveraging,
}

/// Ensemble diversity strategies
#[derive(Debug, Clone)]
pub enum EnsembleDiversityStrategy {
    Bagging,
    Boosting,
    RandomSubspaces,
    QuantumDiversity,
    DifferentAlgorithms,
    DifferentHyperparameters,
}

// Default implementations
impl Default for SearchSpaceConfig {
    fn default() -> Self {
        Self {
            algorithms: AlgorithmSearchSpace::default(),
            preprocessing: PreprocessingSearchSpace::default(),
            hyperparameters: HyperparameterSearchSpace::default(),
            architectures: ArchitectureSearchSpace::default(),
            ensembles: EnsembleSearchSpace::default(),
        }
    }
}

impl Default for AlgorithmSearchSpace {
    fn default() -> Self {
        Self {
            quantum_neural_networks: true,
            quantum_svm: true,
            quantum_clustering: false,
            quantum_dim_reduction: false,
            quantum_time_series: false,
            quantum_anomaly_detection: false,
            classical_algorithms: true,
        }
    }
}

impl Default for PreprocessingSearchSpace {
    fn default() -> Self {
        Self {
            scaling_methods: vec![
                ScalingMethod::StandardScaler,
                ScalingMethod::MinMaxScaler,
                ScalingMethod::NoScaling,
            ],
            feature_selection: vec![
                FeatureSelectionMethod::VarianceThreshold { threshold: 0.01 },
                FeatureSelectionMethod::UnivariateSelection { k: 10 },
            ],
            quantum_encodings: vec![
                QuantumEncodingMethod::AngleEncoding,
                QuantumEncodingMethod::AmplitudeEncoding,
            ],
            data_augmentation: false,
            missing_value_handling: vec![
                MissingValueMethod::MeanImputation,
                MissingValueMethod::DropRows,
            ],
        }
    }
}

impl Default for HyperparameterSearchSpace {
    fn default() -> Self {
        Self {
            learning_rates: (0.001, 0.1),
            regularization: (0.0, 0.1),
            batch_sizes: vec![16, 32, 64, 128],
            epochs: (10, 100),
            quantum_params: QuantumHyperparameterSpace::default(),
        }
    }
}

impl Default for QuantumHyperparameterSpace {
    fn default() -> Self {
        Self {
            num_qubits: (4, 16),
            circuit_depth: (2, 10),
            entanglement_strength: (0.0, 1.0),
            variational_params: (-std::f64::consts::PI, std::f64::consts::PI),
            measurement_strategies: vec!["all_qubits".to_string(), "partial".to_string()],
        }
    }
}

impl Default for ArchitectureSearchSpace {
    fn default() -> Self {
        Self {
            network_architectures: vec![
                NetworkArchitecture::MLP {
                    hidden_layers: vec![64, 32],
                },
                NetworkArchitecture::MLP {
                    hidden_layers: vec![128, 64, 32],
                },
            ],
            quantum_architectures: vec![
                QuantumArchitecture::VariationalCircuit {
                    layers: vec!["RY".to_string(), "CNOT".to_string()],
                    depth: 3,
                },
                QuantumArchitecture::HardwareEfficient {
                    connectivity: "linear".to_string(),
                    repetitions: 2,
                },
            ],
            hybrid_architectures: true,
            generation_strategy: ArchitectureGenerationStrategy::Random,
        }
    }
}

impl Default for EnsembleSearchSpace {
    fn default() -> Self {
        Self {
            enabled: true,
            max_ensemble_size: 5,
            combination_methods: vec![
                EnsembleCombinationMethod::Voting,
                EnsembleCombinationMethod::Averaging,
            ],
            diversity_strategies: vec![
                EnsembleDiversityStrategy::DifferentAlgorithms,
                EnsembleDiversityStrategy::DifferentHyperparameters,
            ],
        }
    }
}

// Convenient configuration builders
impl SearchSpaceConfig {
    /// Comprehensive search space for maximum exploration
    pub fn comprehensive() -> Self {
        Self {
            algorithms: AlgorithmSearchSpace {
                quantum_neural_networks: true,
                quantum_svm: true,
                quantum_clustering: true,
                quantum_dim_reduction: true,
                quantum_time_series: true,
                quantum_anomaly_detection: true,
                classical_algorithms: true,
            },
            preprocessing: PreprocessingSearchSpace {
                scaling_methods: vec![
                    ScalingMethod::StandardScaler,
                    ScalingMethod::MinMaxScaler,
                    ScalingMethod::RobustScaler,
                    ScalingMethod::QuantileScaler,
                    ScalingMethod::QuantumScaler,
                    ScalingMethod::NoScaling,
                ],
                feature_selection: vec![
                    FeatureSelectionMethod::VarianceThreshold { threshold: 0.01 },
                    FeatureSelectionMethod::UnivariateSelection { k: 10 },
                    FeatureSelectionMethod::RecursiveFeatureElimination { n_features: 20 },
                    FeatureSelectionMethod::QuantumFeatureSelection {
                        method: "qpca".to_string(),
                    },
                    FeatureSelectionMethod::PrincipalComponentAnalysis { n_components: 10 },
                    FeatureSelectionMethod::QuantumPCA { n_components: 10 },
                ],
                quantum_encodings: vec![
                    QuantumEncodingMethod::AmplitudeEncoding,
                    QuantumEncodingMethod::AngleEncoding,
                    QuantumEncodingMethod::BasisEncoding,
                    QuantumEncodingMethod::VariationalEncoding { layers: 3 },
                    QuantumEncodingMethod::AutomaticEncoding,
                ],
                data_augmentation: true,
                missing_value_handling: vec![
                    MissingValueMethod::MeanImputation,
                    MissingValueMethod::MedianImputation,
                    MissingValueMethod::QuantumImputation,
                    MissingValueMethod::KNNImputation { k: 5 },
                ],
            },
            hyperparameters: HyperparameterSearchSpace {
                learning_rates: (0.0001, 0.5),
                regularization: (0.0, 0.5),
                batch_sizes: vec![8, 16, 32, 64, 128, 256],
                epochs: (5, 200),
                quantum_params: QuantumHyperparameterSpace {
                    num_qubits: (2, 32),
                    circuit_depth: (1, 20),
                    entanglement_strength: (0.0, 1.0),
                    variational_params: (-2.0 * std::f64::consts::PI, 2.0 * std::f64::consts::PI),
                    measurement_strategies: vec![
                        "all_qubits".to_string(),
                        "partial".to_string(),
                        "adaptive".to_string(),
                    ],
                },
            },
            architectures: ArchitectureSearchSpace {
                network_architectures: vec![
                    NetworkArchitecture::MLP {
                        hidden_layers: vec![32],
                    },
                    NetworkArchitecture::MLP {
                        hidden_layers: vec![64, 32],
                    },
                    NetworkArchitecture::MLP {
                        hidden_layers: vec![128, 64, 32],
                    },
                    NetworkArchitecture::MLP {
                        hidden_layers: vec![256, 128, 64],
                    },
                ],
                quantum_architectures: vec![
                    QuantumArchitecture::VariationalCircuit {
                        layers: vec!["RY".to_string(), "CNOT".to_string()],
                        depth: 2,
                    },
                    QuantumArchitecture::VariationalCircuit {
                        layers: vec!["RX".to_string(), "RZ".to_string(), "CNOT".to_string()],
                        depth: 4,
                    },
                    QuantumArchitecture::HardwareEfficient {
                        connectivity: "linear".to_string(),
                        repetitions: 3,
                    },
                    QuantumArchitecture::HardwareEfficient {
                        connectivity: "circular".to_string(),
                        repetitions: 2,
                    },
                ],
                hybrid_architectures: true,
                generation_strategy: ArchitectureGenerationStrategy::BayesianOptimization,
            },
            ensembles: EnsembleSearchSpace {
                enabled: true,
                max_ensemble_size: 10,
                combination_methods: vec![
                    EnsembleCombinationMethod::Voting,
                    EnsembleCombinationMethod::Averaging,
                    EnsembleCombinationMethod::WeightedAveraging,
                    EnsembleCombinationMethod::Stacking,
                    EnsembleCombinationMethod::QuantumSuperposition,
                ],
                diversity_strategies: vec![
                    EnsembleDiversityStrategy::Bagging,
                    EnsembleDiversityStrategy::DifferentAlgorithms,
                    EnsembleDiversityStrategy::DifferentHyperparameters,
                    EnsembleDiversityStrategy::QuantumDiversity,
                ],
            },
        }
    }

    /// Production search space with balanced exploration
    pub fn production() -> Self {
        Self {
            algorithms: AlgorithmSearchSpace {
                quantum_neural_networks: true,
                quantum_svm: true,
                quantum_clustering: false,
                quantum_dim_reduction: false,
                quantum_time_series: false,
                quantum_anomaly_detection: false,
                classical_algorithms: true,
            },
            preprocessing: PreprocessingSearchSpace {
                scaling_methods: vec![
                    ScalingMethod::StandardScaler,
                    ScalingMethod::MinMaxScaler,
                    ScalingMethod::RobustScaler,
                ],
                feature_selection: vec![
                    FeatureSelectionMethod::VarianceThreshold { threshold: 0.01 },
                    FeatureSelectionMethod::UnivariateSelection { k: 15 },
                ],
                quantum_encodings: vec![
                    QuantumEncodingMethod::AngleEncoding,
                    QuantumEncodingMethod::AmplitudeEncoding,
                ],
                data_augmentation: false,
                missing_value_handling: vec![
                    MissingValueMethod::MeanImputation,
                    MissingValueMethod::MedianImputation,
                ],
            },
            hyperparameters: HyperparameterSearchSpace {
                learning_rates: (0.001, 0.1),
                regularization: (0.0, 0.1),
                batch_sizes: vec![32, 64, 128],
                epochs: (20, 100),
                quantum_params: QuantumHyperparameterSpace {
                    num_qubits: (4, 20),
                    circuit_depth: (2, 8),
                    entanglement_strength: (0.1, 0.9),
                    variational_params: (-std::f64::consts::PI, std::f64::consts::PI),
                    measurement_strategies: vec!["all_qubits".to_string(), "partial".to_string()],
                },
            },
            architectures: ArchitectureSearchSpace {
                network_architectures: vec![
                    NetworkArchitecture::MLP {
                        hidden_layers: vec![64, 32],
                    },
                    NetworkArchitecture::MLP {
                        hidden_layers: vec![128, 64],
                    },
                ],
                quantum_architectures: vec![
                    QuantumArchitecture::VariationalCircuit {
                        layers: vec!["RY".to_string(), "CNOT".to_string()],
                        depth: 3,
                    },
                    QuantumArchitecture::HardwareEfficient {
                        connectivity: "linear".to_string(),
                        repetitions: 2,
                    },
                ],
                hybrid_architectures: true,
                generation_strategy: ArchitectureGenerationStrategy::BayesianOptimization,
            },
            ensembles: EnsembleSearchSpace {
                enabled: true,
                max_ensemble_size: 5,
                combination_methods: vec![
                    EnsembleCombinationMethod::Voting,
                    EnsembleCombinationMethod::WeightedAveraging,
                ],
                diversity_strategies: vec![
                    EnsembleDiversityStrategy::DifferentAlgorithms,
                    EnsembleDiversityStrategy::DifferentHyperparameters,
                ],
            },
        }
    }
}
