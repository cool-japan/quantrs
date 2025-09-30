//! Configuration types for quantum anomaly detection

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Configuration for quantum anomaly detection
#[derive(Debug, Clone)]
pub struct QuantumAnomalyConfig {
    /// Number of qubits for quantum processing
    pub num_qubits: usize,

    /// Primary detection method
    pub primary_method: AnomalyDetectionMethod,

    /// Ensemble methods for improved detection
    pub ensemble_methods: Vec<AnomalyDetectionMethod>,

    /// Contamination level (expected fraction of anomalies)
    pub contamination: f64,

    /// Detection threshold
    pub threshold: f64,

    /// Preprocessing configuration
    pub preprocessing: PreprocessingConfig,

    /// Quantum enhancement configuration
    pub quantum_enhancement: QuantumEnhancementConfig,

    /// Real-time processing configuration
    pub realtime_config: Option<RealtimeConfig>,

    /// Performance configuration
    pub performance_config: PerformanceConfig,

    /// Specialized detector configurations
    pub specialized_detectors: Vec<SpecializedDetectorConfig>,
}

/// Anomaly detection methods
#[derive(Debug, Clone)]
pub enum AnomalyDetectionMethod {
    /// Quantum Isolation Forest
    QuantumIsolationForest {
        n_estimators: usize,
        max_samples: usize,
        max_depth: Option<usize>,
        quantum_splitting: bool,
    },

    /// Quantum Autoencoder
    QuantumAutoencoder {
        encoder_layers: Vec<usize>,
        latent_dim: usize,
        decoder_layers: Vec<usize>,
        reconstruction_threshold: f64,
    },

    /// Quantum One-Class SVM
    QuantumOneClassSVM {
        kernel_type: QuantumKernelType,
        nu: f64,
        gamma: f64,
    },

    /// Quantum K-Means Based Detection
    QuantumKMeansDetection {
        n_clusters: usize,
        distance_metric: DistanceMetric,
        cluster_threshold: f64,
    },

    /// Quantum Local Outlier Factor
    QuantumLOF {
        n_neighbors: usize,
        contamination: f64,
        quantum_distance: bool,
    },

    /// Quantum DBSCAN
    QuantumDBSCAN {
        eps: f64,
        min_samples: usize,
        quantum_density: bool,
    },

    /// Quantum Novelty Detection
    QuantumNoveltyDetection {
        reference_dataset_size: usize,
        novelty_threshold: f64,
        adaptation_rate: f64,
    },

    /// Quantum Ensemble Method
    QuantumEnsemble {
        base_methods: Vec<AnomalyDetectionMethod>,
        voting_strategy: VotingStrategy,
        weight_adaptation: bool,
    },
}

/// Specialized detector configurations
#[derive(Debug, Clone)]
pub enum SpecializedDetectorConfig {
    /// Time series anomaly detection
    TimeSeries {
        window_size: usize,
        seasonal_period: Option<usize>,
        trend_detection: bool,
        quantum_temporal_encoding: bool,
    },

    /// Multivariate anomaly detection
    Multivariate {
        correlation_analysis: bool,
        causal_inference: bool,
        quantum_feature_entanglement: bool,
    },

    /// Network/Graph anomaly detection
    NetworkGraph {
        node_features: bool,
        edge_features: bool,
        structural_anomalies: bool,
        quantum_graph_embedding: bool,
    },

    /// Quantum state anomaly detection
    QuantumState {
        fidelity_threshold: f64,
        entanglement_entropy_analysis: bool,
        quantum_tomography: bool,
    },

    /// Quantum circuit anomaly detection
    QuantumCircuit {
        gate_sequence_analysis: bool,
        parameter_drift_detection: bool,
        noise_characterization: bool,
    },
}

/// Quantum enhancement configuration
#[derive(Debug, Clone)]
pub struct QuantumEnhancementConfig {
    /// Use quantum feature maps
    pub quantum_feature_maps: bool,

    /// Quantum entanglement for feature correlation
    pub entanglement_features: bool,

    /// Quantum superposition for ensemble methods
    pub superposition_ensemble: bool,

    /// Quantum interference for pattern detection
    pub interference_patterns: bool,

    /// Variational quantum eigensolvers for outlier scoring
    pub vqe_scoring: bool,

    /// Quantum approximate optimization for threshold learning
    pub qaoa_optimization: bool,
}

/// Preprocessing configuration
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    /// Normalization method
    pub normalization: NormalizationType,

    /// Dimensionality reduction
    pub dimensionality_reduction: Option<DimensionalityReduction>,

    /// Feature selection
    pub feature_selection: Option<FeatureSelection>,

    /// Noise filtering
    pub noise_filtering: Option<NoiseFiltering>,

    /// Missing value handling
    pub missing_value_strategy: MissingValueStrategy,
}

/// Real-time processing configuration
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Buffer size for streaming data
    pub buffer_size: usize,

    /// Update frequency
    pub update_frequency: usize,

    /// Drift detection
    pub drift_detection: bool,

    /// Online learning
    pub online_learning: bool,

    /// Latency requirements (milliseconds)
    pub max_latency_ms: usize,
}

/// Performance configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Parallel processing
    pub parallel_processing: bool,

    /// Batch size for processing
    pub batch_size: usize,

    /// Memory optimization
    pub memory_optimization: bool,

    /// GPU acceleration
    pub gpu_acceleration: bool,

    /// Quantum circuit optimization
    pub circuit_optimization: bool,
}

/// Supporting enums and types
#[derive(Debug, Clone)]
pub enum QuantumKernelType {
    RBF,
    Linear,
    Polynomial,
    QuantumFeatureMap,
    QuantumKernel,
}

#[derive(Debug, Clone)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Quantum,
    QuantumFidelity,
}

#[derive(Debug, Clone)]
pub enum VotingStrategy {
    Majority,
    Weighted,
    Quantum,
    Consensus,
}

#[derive(Debug, Clone)]
pub enum NormalizationType {
    MinMax,
    ZScore,
    Robust,
    Quantum,
}

#[derive(Debug, Clone)]
pub enum DimensionalityReduction {
    PCA,
    ICA,
    UMAP,
    QuantumPCA,
    QuantumManifold,
}

#[derive(Debug, Clone)]
pub enum FeatureSelection {
    Variance,
    Correlation,
    MutualInformation,
    QuantumInformation,
}

#[derive(Debug, Clone)]
pub enum NoiseFiltering {
    GaussianFilter,
    MedianFilter,
    WaveletDenoising,
    QuantumDenoising,
}

#[derive(Debug, Clone)]
pub enum MissingValueStrategy {
    Remove,
    Mean,
    Median,
    Interpolation,
    QuantumImputation,
}

/// Time series anomaly types
#[derive(Debug, Clone)]
pub enum TimeSeriesAnomalyType {
    /// Point anomaly (single outlier)
    Point,

    /// Contextual anomaly (normal value in wrong context)
    Contextual,

    /// Collective anomaly (sequence of points)
    Collective,

    /// Seasonal anomaly
    Seasonal,

    /// Trend anomaly
    Trend,

    /// Change point
    ChangePoint,
}

/// Seasonal context for time series anomalies
#[derive(Debug, Clone)]
pub struct SeasonalContext {
    /// Seasonal component value
    pub seasonal_value: f64,

    /// Expected seasonal pattern
    pub expected_pattern: Array1<f64>,

    /// Seasonal deviation
    pub seasonal_deviation: f64,
}

/// Trend context for time series anomalies
#[derive(Debug, Clone)]
pub struct TrendContext {
    /// Trend component value
    pub trend_value: f64,

    /// Trend direction
    pub trend_direction: i32, // -1: decreasing, 0: stable, 1: increasing

    /// Trend strength
    pub trend_strength: f64,
}

impl Default for QuantumAnomalyConfig {
    fn default() -> Self {
        Self {
            num_qubits: 4,
            primary_method: AnomalyDetectionMethod::QuantumIsolationForest {
                n_estimators: 100,
                max_samples: 256,
                max_depth: None,
                quantum_splitting: true,
            },
            ensemble_methods: Vec::new(),
            contamination: 0.1,
            threshold: 0.5,
            preprocessing: PreprocessingConfig {
                normalization: NormalizationType::ZScore,
                dimensionality_reduction: None,
                feature_selection: None,
                noise_filtering: None,
                missing_value_strategy: MissingValueStrategy::Mean,
            },
            quantum_enhancement: QuantumEnhancementConfig {
                quantum_feature_maps: true,
                entanglement_features: true,
                superposition_ensemble: false,
                interference_patterns: false,
                vqe_scoring: false,
                qaoa_optimization: false,
            },
            realtime_config: None,
            performance_config: PerformanceConfig {
                parallel_processing: true,
                batch_size: 32,
                memory_optimization: true,
                gpu_acceleration: false,
                circuit_optimization: true,
            },
            specialized_detectors: Vec::new(),
        }
    }
}
