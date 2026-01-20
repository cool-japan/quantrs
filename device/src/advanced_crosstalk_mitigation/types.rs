//! Core types and data structures for advanced crosstalk mitigation

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::Complex64;

use super::config::*;
use crate::crosstalk::CrosstalkCharacterization;

#[cfg(feature = "scirs2")]
use scirs2_ml::StandardScaler;

#[cfg(not(feature = "scirs2"))]
use crate::ml_optimization::fallback_scirs2::StandardScaler;

/// Advanced crosstalk mitigation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedCrosstalkResult {
    /// Base crosstalk characterization
    pub base_characterization: CrosstalkCharacterization,
    /// ML analysis results
    pub ml_analysis: CrosstalkMLResult,
    /// Prediction results
    pub prediction_results: CrosstalkPredictionResult,
    /// Signal processing results
    pub signal_processing: SignalProcessingResult,
    /// Adaptive compensation results
    pub adaptive_compensation: AdaptiveCompensationResult,
    /// Real-time monitoring data
    pub realtime_monitoring: RealtimeMonitoringResult,
    /// Multi-level mitigation results
    pub multilevel_mitigation: MultilevelMitigationResult,
}

/// ML analysis results for crosstalk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrosstalkMLResult {
    /// Trained models
    pub models: HashMap<String, TrainedModel>,
    /// Feature analysis
    pub feature_analysis: FeatureAnalysisResult,
    /// Clustering results
    pub clustering_results: Option<ClusteringResult>,
    /// Anomaly detection results
    pub anomaly_detection: Option<AnomalyDetectionResult>,
    /// Model performance metrics
    pub performance_metrics: ModelPerformanceMetrics,
}

/// Trained model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainedModel {
    /// Model type
    pub model_type: CrosstalkMLModel,
    /// Training accuracy
    pub training_accuracy: f64,
    /// Validation accuracy
    pub validation_accuracy: f64,
    /// Cross-validation scores
    pub cv_scores: Vec<f64>,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Feature importance
    pub feature_importance: HashMap<String, f64>,
    /// Training time
    pub training_time: Duration,
    /// Model size (bytes)
    pub model_size: usize,
}

/// Feature analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureAnalysisResult {
    /// Selected features
    pub selected_features: Vec<String>,
    /// Feature importance scores
    pub importance_scores: HashMap<String, f64>,
    /// Feature correlations
    pub correlations: Array2<f64>,
    /// Mutual information scores
    pub mutual_information: HashMap<String, f64>,
    /// Statistical significance
    pub statistical_significance: HashMap<String, f64>,
}

/// Clustering results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResult {
    /// Cluster assignments
    pub cluster_labels: Vec<usize>,
    /// Cluster centers
    pub cluster_centers: Array2<f64>,
    /// Silhouette score
    pub silhouette_score: f64,
    /// Davies-Bouldin index
    pub davies_bouldin_index: f64,
    /// Calinski-Harabasz index
    pub calinski_harabasz_index: f64,
    /// Number of clusters
    pub n_clusters: usize,
}

/// Anomaly detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionResult {
    /// Anomaly scores
    pub anomaly_scores: Array1<f64>,
    /// Detected anomalies (indices)
    pub anomalies: Vec<usize>,
    /// Anomaly thresholds
    pub thresholds: HashMap<String, f64>,
    /// Anomaly types
    pub anomaly_types: HashMap<usize, String>,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMetrics {
    /// Overall accuracy
    pub accuracy: f64,
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
    /// ROC AUC
    pub roc_auc: f64,
    /// Mean squared error
    pub mse: f64,
    /// Mean absolute error
    pub mae: f64,
    /// R-squared
    pub r2_score: f64,
}

/// Crosstalk prediction results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrosstalkPredictionResult {
    /// Predicted crosstalk values
    pub predictions: Array2<f64>,
    /// Prediction timestamps
    pub timestamps: Vec<SystemTime>,
    /// Confidence intervals
    pub confidence_intervals: Array3<f64>,
    /// Uncertainty estimates
    pub uncertainty_estimates: Array2<f64>,
    /// Time series analysis
    pub time_series_analysis: TimeSeriesAnalysisResult,
}

/// Time series analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesAnalysisResult {
    /// Trend analysis
    pub trend_analysis: TrendAnalysisResult,
    /// Seasonality analysis
    pub seasonality_analysis: SeasonalityAnalysisResult,
    /// Changepoint analysis
    pub changepoint_analysis: ChangepointAnalysisResult,
    /// Forecast accuracy metrics
    pub forecast_metrics: ForecastMetrics,
}

/// Trend analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisResult {
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Trend strength
    pub trend_strength: f64,
    /// Trend significance
    pub trend_significance: f64,
    /// Trend change rate
    pub trend_rate: f64,
}

/// Trend directions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Cyclical,
    Irregular,
}

/// Seasonality analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityAnalysisResult {
    /// Seasonal periods detected
    pub periods: Vec<usize>,
    /// Seasonal strengths
    pub strengths: Vec<f64>,
    /// Seasonal patterns
    pub patterns: HashMap<usize, Array1<f64>>,
    /// Seasonal significance
    pub significance: HashMap<usize, f64>,
}

/// Changepoint analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangepointAnalysisResult {
    /// Detected changepoints
    pub changepoints: Vec<usize>,
    /// Changepoint scores
    pub scores: Vec<f64>,
    /// Changepoint types
    pub types: Vec<ChangepointType>,
    /// Confidence levels
    pub confidence_levels: Vec<f64>,
}

/// Changepoint types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChangepointType {
    MeanShift,
    VarianceChange,
    TrendChange,
    SeasonalityChange,
    StructuralBreak,
}

/// Forecast accuracy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastMetrics {
    /// Mean absolute error
    pub mae: f64,
    /// Mean squared error
    pub mse: f64,
    /// Root mean squared error
    pub rmse: f64,
    /// Mean absolute percentage error
    pub mape: f64,
    /// Symmetric mean absolute percentage error
    pub smape: f64,
    /// Mean absolute scaled error
    pub mase: f64,
}

/// Signal processing results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalProcessingResult {
    /// Filtered signals
    pub filtered_signals: HashMap<String, Array2<Complex64>>,
    /// Spectral analysis
    pub spectral_analysis: SpectralAnalysisResult,
    /// Time-frequency analysis
    pub timefreq_analysis: TimeFrequencyAnalysisResult,
    /// Wavelet analysis
    pub wavelet_analysis: WaveletAnalysisResult,
    /// Noise characteristics
    pub noise_characteristics: NoiseCharacteristicsResult,
}

/// Spectral analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalysisResult {
    /// Power spectral density
    pub power_spectral_density: HashMap<String, Array1<f64>>,
    /// Cross-spectral density
    pub cross_spectral_density: HashMap<(String, String), Array1<Complex64>>,
    /// Coherence
    pub coherence: HashMap<(String, String), Array1<f64>>,
    /// Transfer functions
    pub transfer_functions: HashMap<(String, String), Array1<Complex64>>,
    /// Spectral peaks
    pub spectral_peaks: HashMap<String, Vec<SpectralPeak>>,
}

/// Spectral peak information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralPeak {
    /// Peak frequency
    pub frequency: f64,
    /// Peak amplitude
    pub amplitude: f64,
    /// Peak width
    pub width: f64,
    /// Peak significance
    pub significance: f64,
    /// Peak quality factor
    pub q_factor: f64,
}

/// Time-frequency analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeFrequencyAnalysisResult {
    /// STFT results
    pub stft_results: HashMap<String, Array2<Complex64>>,
    /// CWT results
    pub cwt_results: HashMap<String, Array2<Complex64>>,
    /// HHT results
    pub hht_results: Option<HHTResult>,
    /// Instantaneous frequency
    pub instantaneous_frequency: HashMap<String, Array1<f64>>,
    /// Instantaneous amplitude
    pub instantaneous_amplitude: HashMap<String, Array1<f64>>,
}

/// Hilbert-Huang transform results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HHTResult {
    /// Intrinsic mode functions
    pub imfs: Vec<Array1<f64>>,
    /// Instantaneous frequencies
    pub instantaneous_frequencies: Vec<Array1<f64>>,
    /// Hilbert spectrum
    pub hilbert_spectrum: Array2<f64>,
    /// Marginal spectrum
    pub marginal_spectrum: Array1<f64>,
}

/// Wavelet analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveletAnalysisResult {
    /// Wavelet coefficients
    pub coefficients: HashMap<String, Vec<Array1<f64>>>,
    /// Reconstructed signals
    pub reconstructed_signals: HashMap<String, Array1<f64>>,
    /// Energy distribution
    pub energy_distribution: HashMap<String, Vec<f64>>,
    /// Denoising results
    pub denoising_results: HashMap<String, Array1<f64>>,
}

/// Noise characteristics results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseCharacteristicsResult {
    /// Noise power estimates
    pub noise_power: HashMap<String, f64>,
    /// Signal-to-noise ratio
    pub snr: HashMap<String, f64>,
    /// Noise color analysis
    pub noise_color: HashMap<String, NoiseColor>,
    /// Stationarity analysis
    pub stationarity: HashMap<String, StationarityResult>,
}

/// Noise color types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NoiseColor {
    White,
    Pink,
    Brown,
    Blue,
    Violet,
    Grey,
    Other(f64), // Spectral exponent
}

/// Stationarity analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationarityResult {
    /// Stationarity test statistic
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Is stationary
    pub is_stationary: bool,
    /// Stationarity confidence
    pub confidence: f64,
}

/// Adaptive compensation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCompensationResult {
    /// Compensation matrices
    pub compensation_matrices: HashMap<String, Array2<f64>>,
    /// Learning curves
    pub learning_curves: HashMap<String, Array1<f64>>,
    /// Convergence status
    pub convergence_status: HashMap<String, ConvergenceStatus>,
    /// Performance improvement
    pub performance_improvement: HashMap<String, f64>,
    /// Stability analysis
    pub stability_analysis: StabilityAnalysisResult,
}

/// Convergence status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    Converged,
    NotConverged,
    SlowConvergence,
    Oscillating,
    Diverging,
}

/// Stability analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityAnalysisResult {
    /// Stability margins
    pub stability_margins: StabilityMargins,
    /// Lyapunov exponents
    pub lyapunov_exponents: Array1<f64>,
    /// Stability regions
    pub stability_regions: Vec<StabilityRegion>,
    /// Robustness metrics
    pub robustness_metrics: RobustnessMetrics,
}

/// Stability region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityRegion {
    /// Region bounds
    pub bounds: Array2<f64>,
    /// Stability measure
    pub stability_measure: f64,
    /// Region type
    pub region_type: String,
}

/// Robustness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessMetrics {
    /// Sensitivity analysis
    pub sensitivity: HashMap<String, f64>,
    /// Worst-case performance
    pub worst_case_performance: f64,
    /// Robust stability margin
    pub robust_stability_margin: f64,
    /// Structured singular value
    pub structured_singular_value: f64,
}

/// Real-time monitoring results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeMonitoringResult {
    /// Current status
    pub current_status: SystemStatus,
    /// Performance history
    pub performance_history: VecDeque<PerformanceSnapshot>,
    /// Alert history
    pub alert_history: Vec<AlertEvent>,
    /// Control actions history
    pub control_actions: Vec<ControlAction>,
    /// System health metrics
    pub health_metrics: HealthMetrics,
}

/// System status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SystemStatus {
    Healthy,
    Warning,
    Critical,
    Failed,
    Maintenance,
    Unknown,
}

/// Performance snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Crosstalk matrix
    pub crosstalk_matrix: Array2<f64>,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    /// System state
    pub system_state: SystemState,
}

/// System state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SystemState {
    Idle,
    Active,
    Compensating,
    Learning,
    Optimizing,
    Error,
}

/// Alert event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Alert level
    pub level: AlertLevel,
    /// Alert type
    pub alert_type: String,
    /// Message
    pub message: String,
    /// Affected qubits
    pub affected_qubits: Vec<usize>,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Alert levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
    Critical,
    Emergency,
}

/// Control action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlAction {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Action type
    pub action_type: String,
    /// Parameters
    pub parameters: HashMap<String, f64>,
    /// Target qubits
    pub target_qubits: Vec<usize>,
    /// Expected effect
    pub expected_effect: f64,
    /// Actual effect (if measured)
    pub actual_effect: Option<f64>,
}

/// Health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// Overall health score
    pub overall_health: f64,
    /// Component health scores
    pub component_health: HashMap<String, f64>,
    /// Degradation rate
    pub degradation_rate: f64,
    /// Remaining useful life estimate
    pub remaining_life: Option<Duration>,
    /// Maintenance recommendations
    pub maintenance_recommendations: Vec<String>,
}

/// Multi-level mitigation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultilevelMitigationResult {
    /// Active levels
    pub active_levels: Vec<String>,
    /// Level performance
    pub level_performance: HashMap<String, LevelPerformance>,
    /// Coordination effectiveness
    pub coordination_effectiveness: f64,
    /// Resource utilization
    pub resource_utilization: ResourceUtilizationResult,
    /// Overall effectiveness
    pub overall_effectiveness: f64,
}

/// Level performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelPerformance {
    /// Effectiveness score
    pub effectiveness: f64,
    /// Resource usage
    pub resource_usage: f64,
    /// Response time
    pub response_time: Duration,
    /// Stability
    pub stability: f64,
}

/// Resource utilization results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationResult {
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// Computation time
    pub computation_time: Duration,
    /// Hardware utilization
    pub hardware_utilization: HashMap<String, f64>,
}

/// Feature extractor for ML analysis
pub struct FeatureExtractor {
    pub config: CrosstalkFeatureConfig,
    pub feature_cache: HashMap<String, Array2<f64>>,
    pub scaler: Option<StandardScaler>,
}

/// Crosstalk predictor
pub struct CrosstalkPredictor {
    pub models: HashMap<String, PredictionModel>,
    pub prediction_horizon: Duration,
    pub uncertainty_quantifier: UncertaintyQuantifier,
}

/// Prediction model
pub struct PredictionModel {
    pub model_type: TimeSeriesModel,
    pub model_data: Vec<u8>,
    pub accuracy_metrics: ForecastMetrics,
    pub last_updated: SystemTime,
}

/// Uncertainty quantifier
pub struct UncertaintyQuantifier {
    pub method: UncertaintyEstimationMethod,
    pub confidence_levels: Vec<f64>,
    pub uncertainty_history: VecDeque<Array1<f64>>,
}

/// Time series analyzer
pub struct TimeSeriesAnalyzer {
    pub config: TimeSeriesConfig,
    pub trend_detector: TrendDetector,
    pub seasonality_detector: SeasonalityDetector,
    pub changepoint_detector: ChangepointDetector,
}

/// Trend detector
pub struct TrendDetector {
    pub method: TrendDetectionMethod,
    pub significance_threshold: f64,
    pub trend_history: VecDeque<TrendAnalysisResult>,
}

/// Seasonality detector
pub struct SeasonalityDetector {
    pub periods: Vec<usize>,
    pub strength_threshold: f64,
    pub seasonal_patterns: HashMap<usize, Array1<f64>>,
}

/// Changepoint detector
pub struct ChangepointDetector {
    pub method: ChangepointDetectionMethod,
    pub min_segment_length: usize,
    pub detection_threshold: f64,
    pub changepoint_history: Vec<ChangepointAnalysisResult>,
}

/// Signal processor
pub struct SignalProcessor {
    pub config: SignalProcessingConfig,
    pub filter_bank: FilterBank,
    pub spectral_analyzer: SpectralAnalyzer,
    pub timefreq_analyzer: TimeFrequencyAnalyzer,
    pub wavelet_analyzer: WaveletAnalyzer,
}

/// Filter bank
pub struct FilterBank {
    pub filters: HashMap<String, DigitalFilter>,
    pub adaptive_filters: HashMap<String, AdaptiveFilter>,
    pub noise_reducer: NoiseReducer,
}

/// Digital filter
pub struct DigitalFilter {
    pub filter_type: FilterType,
    pub coefficients: Array1<f64>,
    pub state: Array1<f64>,
    pub parameters: FilterParameters,
}

/// Adaptive filter
pub struct AdaptiveFilter {
    pub algorithm: LearningAlgorithm,
    pub filter_length: usize,
    pub weights: Array1<f64>,
    pub learning_curve: VecDeque<f64>,
}

/// Noise reducer
pub struct NoiseReducer {
    pub method: NoiseReductionMethod,
    pub noise_estimator: NoiseEstimator,
    pub reduction_history: VecDeque<f64>,
}

/// Noise estimator
pub struct NoiseEstimator {
    pub method: NoiseEstimationMethod,
    pub noise_estimate: f64,
    pub adaptation_rate: f64,
}

/// Spectral analyzer
pub struct SpectralAnalyzer {
    pub config: SpectralAnalysisConfig,
    pub window_function: WindowFunction,
    pub spectral_cache: HashMap<String, Array1<f64>>,
}

/// Time-frequency analyzer
pub struct TimeFrequencyAnalyzer {
    pub stft_config: STFTConfig,
    pub cwt_config: CWTConfig,
    pub hht_config: HHTConfig,
    pub analysis_cache: HashMap<String, Array2<Complex64>>,
}

/// Wavelet analyzer
pub struct WaveletAnalyzer {
    pub config: WaveletConfig,
    pub wavelet_bank: HashMap<WaveletType, WaveletBasis>,
    pub decomposition_cache: HashMap<String, Vec<Array1<f64>>>,
}

/// Wavelet basis
pub struct WaveletBasis {
    pub wavelet_type: WaveletType,
    pub scaling_function: Array1<f64>,
    pub wavelet_function: Array1<f64>,
    pub filter_coefficients: (Array1<f64>, Array1<f64>),
}

/// Adaptive compensator
pub struct AdaptiveCompensator {
    pub config: AdaptiveCompensationConfig,
    pub compensation_matrix: Array2<f64>,
    pub learning_state: LearningState,
    pub performance_history: VecDeque<f64>,
    pub optimization_engine: OptimizationEngine,
}

/// Learning state for adaptive systems
pub struct LearningState {
    pub current_parameters: Array1<f64>,
    pub gradient_estimate: Array1<f64>,
    pub momentum: Array1<f64>,
    pub iteration_count: usize,
    pub convergence_history: VecDeque<f64>,
}

/// Optimization engine
pub struct OptimizationEngine {
    pub algorithm: OptimizationAlgorithm,
    pub objective_function: String,
    pub constraints: Vec<OptimizationConstraint>,
    pub optimization_history: VecDeque<f64>,
}

/// Feedback controller
pub struct FeedbackController {
    pub controller_type: ControllerType,
    pub control_state: ControlState,
    pub stability_analyzer: StabilityAnalyzer,
    pub setpoint_history: VecDeque<f64>,
    pub output_history: VecDeque<f64>,
}

/// Control state
pub struct ControlState {
    pub error_history: VecDeque<f64>,
    pub integral_sum: f64,
    pub derivative_estimate: f64,
    pub output_limits: (f64, f64),
    pub controller_parameters: HashMap<String, f64>,
}

/// Stability analyzer
pub struct StabilityAnalyzer {
    pub config: StabilityAnalysisConfig,
    pub stability_history: VecDeque<StabilityMargins>,
    pub robustness_analyzer: RobustnessAnalyzer,
}

/// Robustness analyzer
pub struct RobustnessAnalyzer {
    pub uncertainty_models: Vec<String>,
    pub robustness_metrics: RobustnessMetrics,
    pub sensitivity_analysis: HashMap<String, f64>,
}

/// Real-time monitor
pub struct RealtimeMonitor {
    pub config: RealtimeMitigationConfig,
    pub current_status: SystemStatus,
    pub performance_buffer: VecDeque<PerformanceSnapshot>,
    pub alert_generator: AlertGenerator,
}

/// Alert generator
pub struct AlertGenerator {
    pub thresholds: AlertThresholds,
    pub alert_history: VecDeque<AlertEvent>,
    pub escalation_manager: EscalationManager,
}

/// Alert system
pub struct AlertSystem {
    pub notification_channels: Vec<NotificationChannel>,
    pub alert_queue: VecDeque<AlertEvent>,
    pub notification_history: Vec<String>,
}

/// Escalation manager
pub struct EscalationManager {
    pub escalation_levels: Vec<EscalationLevel>,
    pub current_level: usize,
    pub escalation_timer: Option<SystemTime>,
}

/// Mitigation coordinator
pub struct MitigationCoordinator {
    pub config: MultilevelMitigationConfig,
    pub active_levels: HashMap<String, bool>,
    pub coordination_strategy: CoordinationStrategy,
    pub resource_manager: ResourceManager,
}

/// Resource manager
pub struct ResourceManager {
    pub available_resources: ResourceRequirements,
    pub allocated_resources: HashMap<String, f64>,
    pub optimization_targets: PerformanceTargets,
}