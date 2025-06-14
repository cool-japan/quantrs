//! Result and data structures for process tomography

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::Complex64;
use super::config::SciRS2ProcessTomographyConfig;

// Type placeholders for missing complex types
pub type DistributionType = String;

/// Comprehensive process tomography result with SciRS2 analysis
#[derive(Debug, Clone)]
pub struct SciRS2ProcessTomographyResult {
    /// Device identifier
    pub device_id: String,
    /// Configuration used
    pub config: SciRS2ProcessTomographyConfig,
    /// Reconstructed process matrix (Chi representation)
    pub process_matrix: Array4<Complex64>,
    /// Process matrix in Pauli transfer representation
    pub pauli_transfer_matrix: Array2<f64>,
    /// Statistical analysis of the reconstruction
    pub statistical_analysis: ProcessStatisticalAnalysis,
    /// Process characterization metrics
    pub process_metrics: ProcessMetrics,
    /// Validation results
    pub validation_results: ProcessValidationResults,
    /// Structure analysis
    pub structure_analysis: Option<ProcessStructureAnalysis>,
    /// Uncertainty quantification
    pub uncertainty_quantification: ProcessUncertaintyQuantification,
    /// Comparison with known processes
    pub process_comparisons: ProcessComparisons,
}

/// Statistical analysis of process reconstruction
#[derive(Debug, Clone)]
pub struct ProcessStatisticalAnalysis {
    /// Reconstruction quality metrics
    pub reconstruction_quality: ReconstructionQuality,
    /// Statistical tests on the process
    pub statistical_tests: HashMap<String, StatisticalTest>,
    /// Distribution analysis of process elements
    pub distribution_analysis: DistributionAnalysis,
    /// Correlation analysis
    pub correlation_analysis: CorrelationAnalysis,
}

/// Process characterization metrics
#[derive(Debug, Clone)]
pub struct ProcessMetrics {
    /// Process fidelity with ideal process
    pub process_fidelity: f64,
    /// Average gate fidelity
    pub average_gate_fidelity: f64,
    /// Unitarity measure
    pub unitarity: f64,
    /// Entangling power
    pub entangling_power: f64,
    /// Non-unitality measure
    pub non_unitality: f64,
    /// Channel capacity
    pub channel_capacity: f64,
    /// Coherent information
    pub coherent_information: f64,
    /// Diamond norm distance
    pub diamond_norm_distance: f64,
    /// Process spectrum (eigenvalues of the process)
    pub process_spectrum: Array1<f64>,
}

/// Process validation results
#[derive(Debug, Clone)]
pub struct ProcessValidationResults {
    /// Cross-validation results
    pub cross_validation: Option<CrossValidationResults>,
    /// Bootstrap validation results
    pub bootstrap_results: Option<BootstrapResults>,
    /// Benchmark comparison results
    pub benchmark_results: Option<BenchmarkResults>,
    /// Model selection results
    pub model_selection: ModelSelectionResults,
}

/// Process structure analysis
#[derive(Debug, Clone)]
pub struct ProcessStructureAnalysis {
    /// Kraus decomposition
    pub kraus_decomposition: KrausDecomposition,
    /// Noise decomposition
    pub noise_decomposition: NoiseDecomposition,
    /// Coherence analysis
    pub coherence_analysis: CoherenceAnalysis,
    /// Symmetry analysis
    pub symmetry_analysis: SymmetryAnalysis,
    /// Process graph representation
    pub process_graph: ProcessGraph,
}

/// Uncertainty quantification for process estimates
#[derive(Debug, Clone)]
pub struct ProcessUncertaintyQuantification {
    /// Confidence intervals for process elements
    pub confidence_intervals: Array4<(f64, f64)>,
    /// Bootstrap uncertainty estimates
    pub bootstrap_uncertainty: Array4<f64>,
    /// Fisher information matrix
    pub fisher_information: Array2<f64>,
}

/// Comparison with known process models
#[derive(Debug, Clone)]
pub struct ProcessComparisons {
    /// Fidelities with standard processes
    pub standard_process_fidelities: HashMap<String, f64>,
    /// Distance measures to known processes
    pub process_distances: HashMap<String, f64>,
    /// Model selection criteria
    pub model_selection_scores: HashMap<String, f64>,
}

/// Reconstruction quality metrics
#[derive(Debug, Clone)]
pub struct ReconstructionQuality {
    /// Log-likelihood of the reconstruction
    pub log_likelihood: f64,
    /// Physical validity metrics
    pub physical_validity: PhysicalValidityMetrics,
    /// Numerical stability indicators
    pub condition_number: f64,
}

/// Physical validity metrics for reconstructed processes
#[derive(Debug, Clone)]
pub struct PhysicalValidityMetrics {
    /// Is the process completely positive?
    pub is_completely_positive: bool,
    /// Is the process trace preserving?
    pub is_trace_preserving: bool,
    /// Positivity measure (0-1, 1 = perfectly positive)
    pub positivity_measure: f64,
    /// Trace preservation measure (0-1, 1 = perfectly trace preserving)
    pub trace_preservation_measure: f64,
}

/// Statistical test result
#[derive(Debug, Clone)]
pub struct StatisticalTest {
    /// Test statistic value
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical value at specified confidence level
    pub critical_value: f64,
    /// Whether the test is significant
    pub is_significant: bool,
    /// Effect size (if applicable)
    pub effect_size: Option<f64>,
}

/// Distribution analysis of process matrix elements
#[derive(Debug, Clone)]
pub struct DistributionAnalysis {
    /// Distribution fits for real and imaginary parts
    pub element_distributions: HashMap<String, ElementDistribution>,
    /// Global distribution properties
    pub global_properties: GlobalDistributionProperties,
}

/// Distribution fit for a specific element or set of elements
#[derive(Debug, Clone)]
pub struct ElementDistribution {
    /// Type of distribution (normal, gamma, etc.)
    pub distribution_type: DistributionType,
    /// Distribution parameters
    pub parameters: Vec<f64>,
    /// Goodness of fit measure
    pub goodness_of_fit: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

/// Global properties of the distribution of process elements
#[derive(Debug, Clone)]
pub struct GlobalDistributionProperties {
    /// Overall skewness
    pub skewness: f64,
    /// Overall kurtosis
    pub kurtosis: f64,
    /// Entropy of the distribution
    pub entropy: f64,
}

/// Correlation analysis between process elements
#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    /// Pairwise correlations between elements
    pub element_correlations: HashMap<String, f64>,
    /// Principal component analysis
    pub principal_components: Array2<f64>,
    /// Correlation network structure
    pub correlation_network: CorrelationNetwork,
}

/// Correlation network structure
#[derive(Debug, Clone)]
pub struct CorrelationNetwork {
    /// Adjacency matrix of significant correlations
    pub adjacency_matrix: Array2<f64>,
    /// Network centrality measures
    pub centrality_measures: HashMap<String, f64>,
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResults {
    /// CV scores for different folds
    pub fold_scores: Vec<f64>,
    /// Mean CV score
    pub mean_score: f64,
    /// Standard deviation of CV scores
    pub std_score: f64,
    /// Confidence interval for the mean score
    pub confidence_interval: (f64, f64),
}

/// Bootstrap validation results
#[derive(Debug, Clone)]
pub struct BootstrapResults {
    /// Bootstrap samples of process metrics
    pub bootstrap_samples: Vec<ProcessMetrics>,
    /// Bootstrap confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// Bootstrap bias estimates
    pub bias_estimates: HashMap<String, f64>,
}

/// Benchmark comparison results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Scores against benchmark processes
    pub benchmark_scores: HashMap<String, f64>,
    /// Rankings among benchmark processes
    pub rankings: HashMap<String, usize>,
}

/// Model selection results
#[derive(Debug, Clone)]
pub struct ModelSelectionResults {
    /// AIC scores for different models
    pub aic_scores: HashMap<String, f64>,
    /// BIC scores for different models
    pub bic_scores: HashMap<String, f64>,
    /// Cross-validation scores for different models
    pub cross_validation_scores: HashMap<String, f64>,
    /// Best model according to selection criteria
    pub best_model: String,
    /// Model weights for ensemble methods
    pub model_weights: HashMap<String, f64>,
}

/// Kraus decomposition of the quantum process
#[derive(Debug, Clone)]
pub struct KrausDecomposition {
    /// Kraus operators
    pub kraus_operators: Vec<Array2<Complex64>>,
    /// Decomposition fidelity
    pub decomposition_fidelity: f64,
    /// Number of significant Kraus operators
    pub rank: usize,
}

/// Noise decomposition analysis
#[derive(Debug, Clone)]
pub struct NoiseDecomposition {
    /// Coherent error component
    pub coherent_error: Array2<Complex64>,
    /// Incoherent error components
    pub incoherent_errors: HashMap<String, f64>,
    /// Overall error strength
    pub total_error_strength: f64,
}

/// Coherence analysis of the process
#[derive(Debug, Clone)]
pub struct CoherenceAnalysis {
    /// Coherence measures
    pub coherence_measures: HashMap<String, f64>,
    /// Decoherence time estimates
    pub decoherence_times: HashMap<String, f64>,
    /// Coherence matrix
    pub coherence_matrix: Array2<f64>,
}

/// Symmetry analysis of the process
#[derive(Debug, Clone)]
pub struct SymmetryAnalysis {
    /// Detected symmetries
    pub symmetries: Vec<String>,
    /// Symmetry violation measures
    pub symmetry_violations: HashMap<String, f64>,
    /// Symmetry preservation scores
    pub preservation_scores: HashMap<String, f64>,
}

/// Graph representation of the process
#[derive(Debug, Clone)]
pub struct ProcessGraph {
    /// Adjacency matrix of the process graph
    pub adjacency_matrix: Array2<f64>,
    /// Node properties
    pub node_properties: Vec<NodeProperties>,
    /// Graph metrics
    pub graph_metrics: GraphMetrics,
}

/// Properties of nodes in the process graph
#[derive(Debug, Clone)]
pub struct NodeProperties {
    /// Node index
    pub index: usize,
    /// Node strength (sum of connections)
    pub strength: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Betweenness centrality
    pub betweenness_centrality: f64,
}

/// Graph-level metrics
#[derive(Debug, Clone)]
pub struct GraphMetrics {
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Graph density
    pub density: f64,
    /// Average clustering coefficient
    pub average_clustering: f64,
    /// Average path length
    pub average_path_length: f64,
}

/// Experimental data collected for process tomography
#[derive(Debug, Clone)]
pub struct ExperimentalData {
    /// Input quantum states
    pub input_states: Vec<Array2<Complex64>>,
    /// Measurement operators
    pub measurement_operators: Vec<Array2<Complex64>>,
    /// Measurement outcomes (probabilities or counts)
    pub measurement_results: Vec<f64>,
    /// Measurement uncertainties
    pub measurement_uncertainties: Vec<f64>,
}

/// Anomaly detection for process monitoring
#[derive(Debug, Clone)]
pub struct ProcessAnomalyDetector {
    /// Historical process data
    pub historical_data: Vec<ProcessMetrics>,
    /// Anomaly detection threshold
    pub threshold: f64,
    /// Detection algorithm
    pub algorithm: AnomalyDetectionAlgorithm,
}

/// Anomaly detection algorithms
#[derive(Debug, Clone)]
pub enum AnomalyDetectionAlgorithm {
    StatisticalThreshold,
    IsolationForest,
    OneClassSVM,
    LocalOutlierFactor,
}

/// Drift detection for process stability monitoring
#[derive(Debug, Clone)]
pub struct ProcessDriftDetector {
    /// Reference process metrics
    pub reference_metrics: ProcessMetrics,
    /// Drift detection sensitivity
    pub sensitivity: f64,
    /// Drift detection method
    pub method: DriftDetectionMethod,
}

/// Drift detection methods
#[derive(Debug, Clone)]
pub enum DriftDetectionMethod {
    StatisticalTest,
    ChangePointDetection,
    KLDivergence,
    WassersteinDistance,
}

/// Process monitoring result
#[derive(Debug, Clone)]
pub struct ProcessMonitoringResult {
    /// Current process metrics
    pub current_metrics: ProcessMetrics,
    /// Experimental conditions
    pub experimental_conditions: ExperimentalConditions,
    /// Anomaly score (0-1, higher = more anomalous)
    pub anomaly_score: f64,
    /// Drift indicator (0-1, higher = more drift)
    pub drift_indicator: f64,
    /// Alert level
    pub alert_level: AlertLevel,
}

/// Experimental conditions during measurement
#[derive(Debug, Clone)]
pub struct ExperimentalConditions {
    /// Temperature (if available)
    pub temperature: Option<f64>,
    /// Estimated noise level
    pub noise_level: f64,
    /// Time since last calibration
    pub calibration_age: Duration,
    /// Number of gates in the process
    pub gate_count: usize,
    /// Circuit depth
    pub circuit_depth: usize,
}

/// Alert levels for process monitoring
#[derive(Debug, Clone, PartialEq)]
pub enum AlertLevel {
    Normal,
    Warning,
    Critical,
}

impl Default for ProcessMetrics {
    fn default() -> Self {
        Self {
            process_fidelity: 1.0,
            average_gate_fidelity: 1.0,
            unitarity: 1.0,
            entangling_power: 0.0,
            non_unitality: 0.0,
            channel_capacity: 1.0,
            coherent_information: 1.0,
            diamond_norm_distance: 0.0,
            process_spectrum: Array1::ones(2),
        }
    }
}

impl Default for StatisticalTest {
    fn default() -> Self {
        Self {
            statistic: 0.0,
            p_value: 1.0,
            critical_value: 0.0,
            is_significant: false,
            effect_size: None,
        }
    }
}

impl Default for ElementDistribution {
    fn default() -> Self {
        Self {
            distribution_type: "normal".to_string(),
            parameters: vec![0.0, 1.0],
            goodness_of_fit: 1.0,
            confidence_interval: (0.0, 1.0),
        }
    }
}

impl Default for GlobalDistributionProperties {
    fn default() -> Self {
        Self {
            skewness: 0.0,
            kurtosis: 0.0,
            entropy: 1.0,
        }
    }
}

impl Default for CorrelationNetwork {
    fn default() -> Self {
        Self {
            adjacency_matrix: Array2::zeros((2, 2)),
            centrality_measures: HashMap::new(),
        }
    }
}

impl Default for GraphMetrics {
    fn default() -> Self {
        Self {
            num_nodes: 0,
            num_edges: 0,
            density: 0.0,
            average_clustering: 0.0,
            average_path_length: 0.0,
        }
    }
}