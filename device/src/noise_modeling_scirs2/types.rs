//! Type definitions for SciRS2 noise modeling
//!
//! This module contains all the type definitions and data structures used
//! in the advanced SciRS2 noise modeling system.

use std::collections::HashMap;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;

use crate::{
    calibration::{DeviceCalibration, QubitCalibration},
    noise_model::{CalibrationNoiseModel, QubitNoiseParams, GateNoiseParams},
};

// Import configuration types from config module
use super::config::{
    SciRS2NoiseConfig, ValidationConfig, ValidationMetric, DistributionType,
    NoiseColor as NoiseColorConfig, DecompositionMethod as DecompositionMethodConfig,
    MLModelType,
};

// =============================================================================
// Main Model Types
// =============================================================================

/// Comprehensive noise modeling result
#[derive(Debug, Clone)]
pub struct SciRS2NoiseModel {
    /// Device identifier
    pub device_id: String,
    /// Configuration used
    pub config: SciRS2NoiseConfig,
    /// Statistical noise characterization
    pub statistical_model: StatisticalNoiseModel,
    /// Spectral noise analysis
    pub spectral_model: Option<SpectralNoiseModel>,
    /// Temporal correlation model
    pub temporal_model: Option<TemporalNoiseModel>,
    /// Spatial correlation model
    pub spatial_model: Option<SpatialNoiseModel>,
    /// Multi-level noise decomposition
    pub decomposition_model: Option<NoiseDecompositionModel>,
    /// Machine learning models
    pub ml_models: Option<MLNoiseModels>,
    /// Model validation results
    pub validation_results: ValidationResults,
    /// Noise prediction capabilities
    pub prediction_model: NoisePredictionModel,
}

// =============================================================================
// Statistical Model Types
// =============================================================================

/// Statistical noise characterization
#[derive(Debug, Clone)]
pub struct StatisticalNoiseModel {
    /// Distributional analysis for each noise source
    pub distributions: HashMap<String, NoiseDistribution>,
    /// Higher-order moment analysis
    pub moments: HashMap<String, MomentAnalysis>,
    /// Correlation structure between noise sources
    pub correlation_structure: CorrelationStructure,
    /// Outlier detection and analysis
    pub outlier_analysis: OutlierAnalysis,
    /// Non-parametric density estimates
    pub density_estimates: HashMap<String, DensityEstimate>,
}

/// Noise distribution analysis
#[derive(Debug, Clone)]
pub struct NoiseDistribution {
    pub distribution_type: DistributionType,
    pub parameters: Vec<f64>,
    pub goodness_of_fit: f64,
    pub confidence_intervals: Vec<(f64, f64)>,
    pub p_value: f64,
}

// DistributionType is imported from config module

/// Higher-order moment analysis
#[derive(Debug, Clone)]
pub struct MomentAnalysis {
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub higher_moments: Vec<f64>,
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}

/// Correlation structure analysis
#[derive(Debug, Clone)]
pub struct CorrelationStructure {
    pub correlationmatrix: Array2<f64>,
    pub partial_correlations: Array2<f64>,
    pub rank_correlations: Array2<f64>,
    pub time_varying_correlations: Option<Array3<f64>>,
    pub correlation_networks: CorrelationNetworks,
}

/// Correlation network analysis
#[derive(Debug, Clone)]
pub struct CorrelationNetworks {
    pub threshold_networks: HashMap<String, Array2<bool>>,
    pub community_structure: Vec<Vec<usize>>,
    pub centrality_measures: HashMap<usize, CentralityMeasures>,
}

/// Network centrality measures
#[derive(Debug, Clone)]
pub struct CentralityMeasures {
    pub betweenness: f64,
    pub closeness: f64,
    pub eigenvector: f64,
    pub pagerank: f64,
}

/// Outlier detection analysis
#[derive(Debug, Clone)]
pub struct OutlierAnalysis {
    pub outlier_indices: Vec<usize>,
    pub outlier_scores: Array1<f64>,
    pub outlier_method: OutlierMethod,
    pub contamination_rate: f64,
}

/// Outlier detection methods
#[derive(Debug, Clone, PartialEq)]
pub enum OutlierMethod {
    IsolationForest,
    LocalOutlierFactor,
    OneClassSVM,
    DBSCAN,
    StatisticalTests,
}

/// Non-parametric density estimation
#[derive(Debug, Clone)]
pub struct DensityEstimate {
    pub method: DensityMethod,
    pub bandwidth: f64,
    pub support: Array1<f64>,
    pub density: Array1<f64>,
    pub log_likelihood: f64,
}

/// Density estimation methods
#[derive(Debug, Clone, PartialEq)]
pub enum DensityMethod {
    KernelDensityEstimation,
    HistogramEstimation,
    GaussianMixture,
    Splines,
}

// =============================================================================
// Spectral Model Types
// =============================================================================

/// Spectral noise analysis
#[derive(Debug, Clone)]
pub struct SpectralNoiseModel {
    /// Power spectral density for each noise source
    pub power_spectra: HashMap<String, Array1<f64>>,
    /// Spectral peaks and their characteristics
    pub spectral_peaks: HashMap<String, Vec<SpectralPeak>>,
    /// Noise coloring analysis (1/f, white, etc.)
    pub noise_coloring: HashMap<String, NoiseColorAnalysis>,
    /// Cross-spectral analysis between noise sources
    pub cross_spectra: HashMap<(String, String), Array1<Complex64>>,
    /// Coherence analysis
    pub coherence_analysis: CoherenceAnalysis,
}

/// Spectral peak characteristics
#[derive(Debug, Clone)]
pub struct SpectralPeak {
    pub frequency: f64,
    pub amplitude: f64,
    pub width: f64,
    pub phase: f64,
    pub significance: f64,
    pub harmonic_number: Option<usize>,
}

/// Noise coloring analysis
#[derive(Debug, Clone)]
pub struct NoiseColorAnalysis {
    pub color_type: NoiseColorConfig,
    pub exponent: f64,
    pub confidence_interval: (f64, f64),
    pub fit_quality: f64,
}

// NoiseColorType is available as NoiseColor from config module

/// Coherence analysis between noise sources
#[derive(Debug, Clone)]
pub struct CoherenceAnalysis {
    pub coherence_matrix: Array2<f64>,
    pub significant_coherences: Vec<(usize, usize, f64)>,
    pub frequency_bands: Vec<FrequencyBand>,
    pub phase_relationships: Array2<f64>,
}

/// Frequency band analysis
#[derive(Debug, Clone)]
pub struct FrequencyBand {
    pub name: String,
    pub frequency_range: (f64, f64),
    pub coherence_statistics: HashMap<String, f64>,
}

// =============================================================================
// Temporal Model Types
// =============================================================================

/// Temporal correlation model
#[derive(Debug, Clone)]
pub struct TemporalNoiseModel {
    /// Autoregressive models for each noise source
    pub ar_models: HashMap<String, ARModel>,
    /// Long-term memory characteristics
    pub long_memory: HashMap<String, LongMemoryModel>,
    /// Non-stationary analysis
    pub nonstationarity: HashMap<String, NonstationarityAnalysis>,
    /// Change point detection
    pub change_points: HashMap<String, Vec<ChangePoint>>,
    /// Temporal clustering
    pub temporal_clusters: HashMap<String, TemporalClusters>,
}

/// Autoregressive model
#[derive(Debug, Clone)]
pub struct ARModel {
    pub order: usize,
    pub coefficients: Array1<f64>,
    pub noise_variance: f64,
    pub aic: f64,
    pub bic: f64,
    pub prediction_error: f64,
}

/// Long-term memory model
#[derive(Debug, Clone)]
pub struct LongMemoryModel {
    pub hurst_exponent: f64,
    pub fractal_dimension: f64,
    pub long_range_dependence: bool,
    pub memory_parameter: f64,
    pub confidence_interval: (f64, f64),
}

/// Non-stationarity analysis
#[derive(Debug, Clone)]
pub struct NonstationarityAnalysis {
    pub is_stationary: bool,
    pub test_statistics: HashMap<String, f64>,
    pub change_point_locations: Vec<usize>,
    pub trend_components: Array1<f64>,
    pub seasonal_components: Option<Array1<f64>>,
}

/// Change point detection
#[derive(Debug, Clone)]
pub struct ChangePoint {
    pub location: usize,
    pub timestamp: f64,
    pub change_magnitude: f64,
    pub change_type: ChangeType,
    pub confidence: f64,
}

/// Types of changes detected
#[derive(Debug, Clone, PartialEq)]
pub enum ChangeType {
    Mean,
    Variance,
    Distribution,
    Correlation,
}

/// Temporal clustering analysis
#[derive(Debug, Clone)]
pub struct TemporalClusters {
    pub cluster_labels: Array1<usize>,
    pub cluster_centers: Array2<f64>,
    pub cluster_statistics: Vec<ClusterStatistics>,
    pub temporal_transitions: Array2<f64>,
}

/// Cluster statistics
#[derive(Debug, Clone)]
pub struct ClusterStatistics {
    pub cluster_id: usize,
    pub size: usize,
    pub duration: f64,
    pub stability: f64,
    pub characteristics: HashMap<String, f64>,
}

// =============================================================================
// Spatial Model Types
// =============================================================================

/// Spatial correlation model
#[derive(Debug, Clone)]
pub struct SpatialNoiseModel {
    /// Spatial covariance structure
    pub covariance_structure: SpatialCovariance,
    /// Spatial basis functions
    pub basis_functions: SpatialBasisFunctions,
    /// Kriging models for spatial interpolation
    pub kriging_models: HashMap<String, KrigingModel>,
    /// Spatial clustering of noise patterns
    pub spatial_clusters: SpatialClusters,
    /// Anisotropy analysis
    pub anisotropy: AnisotropyAnalysis,
}

/// Spatial covariance function
#[derive(Debug, Clone)]
pub struct SpatialCovariance {
    pub covariance_function: CovarianceFunction,
    pub parameters: Vec<f64>,
    pub range_parameter: f64,
    pub nugget_effect: f64,
    pub anisotropy_parameters: Option<AnisotropyParameters>,
}

/// Covariance function types
#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceFunction {
    Exponential,
    Gaussian,
    Matern,
    Spherical,
    Linear,
    Custom,
}

/// Anisotropy parameters
#[derive(Debug, Clone)]
pub struct AnisotropyParameters {
    pub major_range: f64,
    pub minor_range: f64,
    pub rotation_angle: f64,
    pub anisotropy_ratio: f64,
}

/// Spatial basis functions
#[derive(Debug, Clone)]
pub struct SpatialBasisFunctions {
    pub basis_type: BasisType,
    pub num_functions: usize,
    pub coefficients: Array2<f64>,
    pub explained_variance: Array1<f64>,
}

/// Spatial basis function types
#[derive(Debug, Clone, PartialEq)]
pub enum BasisType {
    Fourier,
    Wavelets,
    Polynomials,
    RadialBasisFunctions,
    PrincipalComponents,
}

/// Kriging spatial interpolation model
#[derive(Debug, Clone)]
pub struct KrigingModel {
    pub kriging_type: KrigingType,
    pub covariance_model: SpatialCovariance,
    pub prediction_variance: Array2<f64>,
    pub cross_validation_score: f64,
}

/// Kriging model types
#[derive(Debug, Clone, PartialEq)]
pub enum KrigingType {
    Ordinary,
    Universal,
    Simple,
    Indicator,
}

/// Spatial clustering analysis
#[derive(Debug, Clone)]
pub struct SpatialClusters {
    pub cluster_labels: Array1<usize>,
    pub cluster_boundaries: Vec<SpatialBoundary>,
    pub cluster_characteristics: Vec<SpatialClusterStats>,
}

/// Spatial cluster boundary
#[derive(Debug, Clone)]
pub struct SpatialBoundary {
    pub cluster_id: usize,
    pub boundary_points: Array2<f64>,
    pub convex_hull: Array2<f64>,
    pub area: f64,
}

/// Spatial cluster statistics
#[derive(Debug, Clone)]
pub struct SpatialClusterStats {
    pub cluster_id: usize,
    pub centroid: Array1<f64>,
    pub variance: f64,
    pub size: usize,
    pub density: f64,
}

/// Anisotropy analysis
#[derive(Debug, Clone)]
pub struct AnisotropyAnalysis {
    pub anisotropy_detected: bool,
    pub primary_direction: f64,
    pub anisotropy_ratio: f64,
    pub confidence_ellipse: Array2<f64>,
}

// =============================================================================
// Decomposition Model Types
// =============================================================================

/// Multi-level noise decomposition
#[derive(Debug, Clone)]
pub struct NoiseDecompositionModel {
    /// High-frequency noise components
    pub high_frequency: HashMap<String, NoiseComponent>,
    /// Low-frequency drift
    pub low_frequency: HashMap<String, NoiseComponent>,
    /// Systematic noise patterns
    pub systematic: HashMap<String, NoiseComponent>,
    /// Random noise floor
    pub random: HashMap<String, NoiseComponent>,
    /// Decomposition method used
    pub decomposition_method: DecompositionMethodConfig,
}

/// Noise component description
#[derive(Debug, Clone)]
pub struct NoiseComponent {
    pub component_type: ComponentType,
    pub amplitude: Array1<f64>,
    pub frequency_content: Option<Array1<f64>>,
    pub spatial_pattern: Option<Array2<f64>>,
    pub explained_variance: f64,
}

/// Component types
#[derive(Debug, Clone, PartialEq)]
pub enum ComponentType {
    Systematic,
    Random,
    Periodic,
    Trend,
    Transient,
}

// DecompositionMethod is imported from config module as DecompositionMethodConfig

// =============================================================================
// Machine Learning Model Types
// =============================================================================

/// Machine learning noise models
#[derive(Debug, Clone)]
pub struct MLNoiseModels {
    /// Gaussian process models
    pub gaussian_process: HashMap<String, GaussianProcessModel>,
    /// Neural network models
    pub neural_networks: HashMap<String, NeuralNetworkModel>,
    /// Ensemble models
    pub ensemble_models: HashMap<String, EnsembleModel>,
    /// Feature importance analysis
    pub feature_importance: HashMap<String, Array1<f64>>,
    /// Model hyperparameters
    pub hyperparameters: HashMap<String, HashMap<String, f64>>,
}

/// Gaussian process model
#[derive(Debug, Clone)]
pub struct GaussianProcessModel {
    pub kernel_type: KernelType,
    pub hyperparameters: Vec<f64>,
    pub training_data: Array2<f64>,
    pub prediction_mean: Array1<f64>,
    pub prediction_variance: Array1<f64>,
    pub log_marginal_likelihood: f64,
}

/// Gaussian process kernel types
#[derive(Debug, Clone, PartialEq)]
pub enum KernelType {
    RBF,
    Matern32,
    Matern52,
    RationalQuadratic,
    PeriodicKernel,
    LinearKernel,
}

/// Neural network model
#[derive(Debug, Clone)]
pub struct NeuralNetworkModel {
    pub architecture: NetworkArchitecture,
    pub weights: Vec<Array2<f64>>,
    pub biases: Vec<Array1<f64>>,
    pub training_loss: f64,
    pub validation_loss: f64,
    pub feature_importance: Array1<f64>,
}

/// Neural network architecture
#[derive(Debug, Clone)]
pub struct NetworkArchitecture {
    pub input_size: usize,
    pub hidden_layers: Vec<usize>,
    pub output_size: usize,
    pub activation_functions: Vec<ActivationFunction>,
}

/// Activation function types
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Swish,
    GELU,
}

/// Ensemble model
#[derive(Debug, Clone)]
pub struct EnsembleModel {
    pub model_types: Vec<MLModelType>,
    pub model_weights: Array1<f64>,
    pub ensemble_predictions: Array1<f64>,
    pub individual_predictions: Array2<f64>,
    pub diversity_measures: Array1<f64>,
}

// ModelType is available as MLModelType from config module

// =============================================================================
// Validation Types
// =============================================================================

/// Model validation results
#[derive(Debug, Clone)]
pub struct ValidationResults {
    pub cross_validation: CrossValidationResults,
    pub bootstrap_results: Option<BootstrapResults>,
    pub model_comparison: ModelComparison,
    pub uncertainty_quantification: UncertaintyQuantification,
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResults {
    pub cv_scores: Array1<f64>,
    pub mean_score: f64,
    pub std_score: f64,
    pub best_fold: usize,
    pub worst_fold: usize,
}

/// Bootstrap validation results
#[derive(Debug, Clone)]
pub struct BootstrapResults {
    pub bootstrap_scores: Array1<f64>,
    pub bootstrap_mean: f64,
    pub bootstrap_std: f64,
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}

/// Model comparison results
#[derive(Debug, Clone)]
pub struct ModelComparison {
    pub model_rankings: Vec<(String, f64)>,
    pub statistical_tests: HashMap<String, f64>,
    pub effect_sizes: HashMap<String, f64>,
    pub model_selection_criteria: HashMap<String, f64>,
}

/// Uncertainty quantification
#[derive(Debug, Clone)]
pub struct UncertaintyQuantification {
    pub epistemic_uncertainty: Array1<f64>,
    pub aleatoric_uncertainty: Array1<f64>,
    pub total_uncertainty: Array1<f64>,
    pub uncertainty_decomposition: Array2<f64>,
}

// =============================================================================
// Prediction Model Types
// =============================================================================

/// Noise prediction model
#[derive(Debug, Clone)]
pub struct NoisePredictionModel {
    pub prediction_horizon: usize,
    pub prediction_accuracy: HashMap<String, f64>,
    pub feature_engineering: FeatureEngineering,
    pub adaptive_components: AdaptiveComponents,
}

/// Feature engineering for prediction
#[derive(Debug, Clone)]
pub struct FeatureEngineering {
    pub selected_features: Vec<String>,
    pub feature_transformations: Vec<FeatureTransform>,
    pub dimensionality_reduction: Option<DimensionalityReduction>,
}

/// Feature transformation
#[derive(Debug, Clone)]
pub struct FeatureTransform {
    pub transform_type: TransformType,
    pub parameters: Vec<f64>,
    pub importance_score: f64,
}

/// Feature transformation types
#[derive(Debug, Clone, PartialEq)]
pub enum TransformType {
    LogTransform,
    PowerTransform,
    BoxCoxTransform,
    StandardScaling,
    MinMaxScaling,
    PolynomialFeatures,
}

/// Dimensionality reduction
#[derive(Debug, Clone)]
pub struct DimensionalityReduction {
    pub method: DimReductionMethod,
    pub num_components: usize,
    pub explained_variance_ratio: Array1<f64>,
    pub transformation_matrix: Array2<f64>,
}

/// Dimensionality reduction methods
#[derive(Debug, Clone, PartialEq)]
pub enum DimReductionMethod {
    PCA,
    ICA,
    UMAP,
    tSNE,
    FactorAnalysis,
}

/// Adaptive prediction components
#[derive(Debug, Clone)]
pub struct AdaptiveComponents {
    pub adaptation_rate: f64,
    pub forgetting_factor: f64,
    pub change_detection: ChangeDetection,
    pub online_learning: OnlineLearning,
}

/// Change detection for adaptive modeling
#[derive(Debug, Clone)]
pub struct ChangeDetection {
    pub method: ChangeDetectionMethod,
    pub sensitivity: f64,
    pub false_alarm_rate: f64,
    pub detection_delay: f64,
}

/// Change detection methods
#[derive(Debug, Clone, PartialEq)]
pub enum ChangeDetectionMethod {
    CUSUM,
    PageHinkley,
    ADWIN,
    KSWIN,
    DDM,
}

/// Online learning configuration
#[derive(Debug, Clone)]
pub struct OnlineLearning {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub update_frequency: usize,
    pub performance_monitoring: PerformanceMonitoring,
}

/// Performance monitoring for online learning
#[derive(Debug, Clone)]
pub struct PerformanceMonitoring {
    pub drift_detection: bool,
    pub performance_threshold: f64,
    pub monitoring_window: usize,
    pub alert_mechanisms: Vec<AlertType>,
}

/// Alert types for performance monitoring
#[derive(Debug, Clone, PartialEq)]
pub enum AlertType {
    PerformanceDegradation,
    DistributionShift,
    ModelDrift,
    DataQualityIssue,
}