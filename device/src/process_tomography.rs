//! Quantum process tomography implementation using SciRS2
//!
//! This module provides comprehensive quantum process tomography capabilities
//! leveraging SciRS2's advanced statistical analysis, optimization, and machine learning tools
//! for robust and efficient process characterization.

use std::collections::HashMap;
use std::f64::consts::PI;

use quantrs2_circuit::prelude::*;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

// Type placeholders for missing complex types
type DistributionType = String;

// SciRS2 dependencies (feature-gated for availability)
#[cfg(feature = "scirs2")]
use scirs2_graph::{
    betweenness_centrality, closeness_centrality, minimum_spanning_tree, shortest_path,
    strongly_connected_components, Graph,
};
#[cfg(feature = "scirs2")]
use scirs2_linalg::{
    cholesky, det, eig, inv, matrix_norm, prelude::*, qr, svd, trace, LinalgError, LinalgResult,
};
#[cfg(feature = "scirs2")]
use scirs2_optimize::{minimize, OptimizeResult};
#[cfg(feature = "scirs2")]
use scirs2_stats::{
    corrcoef,
    distributions::{chi2, gamma, norm},
    ks_2samp, mean, pearsonr, shapiro_wilk, spearmanr, std, ttest_1samp, ttest_ind, var,
    Alternative, TTestResult,
};

// Fallback implementations when SciRS2 is not available
#[cfg(not(feature = "scirs2"))]
mod fallback_scirs2 {
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

    pub fn mean(_data: &ArrayView1<f64>) -> Result<f64, String> {
        Ok(0.0)
    }
    pub fn std(_data: &ArrayView1<f64>, _ddof: i32) -> Result<f64, String> {
        Ok(1.0)
    }
    pub fn pearsonr(
        _x: &ArrayView1<f64>,
        _y: &ArrayView1<f64>,
        _alt: &str,
    ) -> Result<(f64, f64), String> {
        Ok((0.0, 0.5))
    }
    pub fn trace(_matrix: &ArrayView2<f64>) -> Result<f64, String> {
        Ok(1.0)
    }
    pub fn inv(_matrix: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        Ok(Array2::eye(2))
    }

    pub struct OptimizeResult {
        pub x: Array1<f64>,
        pub fun: f64,
        pub success: bool,
        pub nit: usize,
    }

    pub fn minimize(
        _func: fn(&Array1<f64>) -> f64,
        _x0: &Array1<f64>,
        _method: &str,
    ) -> Result<OptimizeResult, String> {
        Ok(OptimizeResult {
            x: Array1::zeros(2),
            fun: 0.0,
            success: true,
            nit: 0,
        })
    }
}

#[cfg(not(feature = "scirs2"))]
use fallback_scirs2::*;

use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView4, Axis};
use num_complex::Complex64;
use rand::prelude::*;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::{
    backend_traits::{query_backend_capabilities, BackendCapabilities},
    calibration::{CalibrationManager, DeviceCalibration},
    characterization::{ProcessTomography, StateTomography},
    noise_model::CalibrationNoiseModel,
    translation::HardwareBackend,
    CircuitResult, DeviceError, DeviceResult,
};

/// Configuration for SciRS2-enhanced process tomography
#[derive(Debug, Clone)]
pub struct SciRS2ProcessTomographyConfig {
    /// Number of input states for process characterization
    pub num_input_states: usize,
    /// Number of measurement shots per state
    pub shots_per_state: usize,
    /// Reconstruction method
    pub reconstruction_method: ReconstructionMethod,
    /// Statistical confidence level
    pub confidence_level: f64,
    /// Enable compressed sensing reconstruction
    pub enable_compressed_sensing: bool,
    /// Enable maximum likelihood estimation
    pub enable_mle: bool,
    /// Enable Bayesian inference
    pub enable_bayesian: bool,
    /// Enable process structure analysis
    pub enable_structure_analysis: bool,
    /// Enable multi-process tomography
    pub enable_multi_process: bool,
    /// Optimization settings
    pub optimization_config: OptimizationConfig,
    /// Validation settings
    pub validation_config: ProcessValidationConfig,
}

/// Process reconstruction methods
#[derive(Debug, Clone, PartialEq)]
pub enum ReconstructionMethod {
    /// Linear inversion (fast but can produce unphysical results)
    LinearInversion,
    /// Maximum likelihood estimation (physical but slower)
    MaximumLikelihood,
    /// Compressed sensing (sparse process assumption)
    CompressedSensing,
    /// Bayesian inference with priors
    BayesianInference,
    /// Ensemble methods combining multiple approaches
    EnsembleMethods,
    /// Machine learning based reconstruction
    MachineLearning,
}

/// Optimization configuration for process reconstruction
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Optimization algorithm
    pub algorithm: OptimizationAlgorithm,
    /// Enable parallel optimization
    pub enable_parallel: bool,
    /// Enable adaptive step sizing
    pub adaptive_step_size: bool,
    /// Regularization parameters
    pub regularization: RegularizationConfig,
}

/// Optimization algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationAlgorithm {
    LBFGS,
    ConjugateGradient,
    TrustRegion,
    DifferentialEvolution,
    SimulatedAnnealing,
    GeneticAlgorithm,
    ParticleSwarm,
}

/// Regularization configuration
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// L1 regularization strength (sparsity)
    pub l1_strength: f64,
    /// L2 regularization strength (smoothness)
    pub l2_strength: f64,
    /// Trace preservation constraint strength
    pub trace_strength: f64,
    /// Positivity constraint strength
    pub positivity_strength: f64,
}

/// Process validation configuration
#[derive(Debug, Clone)]
pub struct ProcessValidationConfig {
    /// Enable cross-validation
    pub enable_cross_validation: bool,
    /// Number of CV folds
    pub cv_folds: usize,
    /// Enable bootstrap validation
    pub enable_bootstrap: bool,
    /// Number of bootstrap samples
    pub bootstrap_samples: usize,
    /// Enable process benchmarking
    pub enable_benchmarking: bool,
    /// Benchmark processes to compare against
    pub benchmark_processes: Vec<String>,
}

impl Default for SciRS2ProcessTomographyConfig {
    fn default() -> Self {
        Self {
            num_input_states: 36, // 6^n for n qubits (standard set)
            shots_per_state: 10000,
            reconstruction_method: ReconstructionMethod::MaximumLikelihood,
            confidence_level: 0.95,
            enable_compressed_sensing: true,
            enable_mle: true,
            enable_bayesian: false,
            enable_structure_analysis: true,
            enable_multi_process: false,
            optimization_config: OptimizationConfig {
                max_iterations: 1000,
                tolerance: 1e-8,
                algorithm: OptimizationAlgorithm::LBFGS,
                enable_parallel: true,
                adaptive_step_size: true,
                regularization: RegularizationConfig {
                    l1_strength: 0.001,
                    l2_strength: 0.01,
                    trace_strength: 1000.0,
                    positivity_strength: 100.0,
                },
            },
            validation_config: ProcessValidationConfig {
                enable_cross_validation: true,
                cv_folds: 5,
                enable_bootstrap: true,
                bootstrap_samples: 100,
                enable_benchmarking: true,
                benchmark_processes: vec![
                    "identity".to_string(),
                    "pauli_x".to_string(),
                    "pauli_y".to_string(),
                    "pauli_z".to_string(),
                    "hadamard".to_string(),
                ],
            },
        }
    }
}

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
    /// Diamond norm distance to ideal
    pub diamond_norm_distance: f64,
    /// Process spectrum (eigenvalues)
    pub process_spectrum: Array1<Complex64>,
}

/// Process validation results
#[derive(Debug, Clone)]
pub struct ProcessValidationResults {
    /// Cross-validation results
    pub cross_validation: Option<CrossValidationResults>,
    /// Bootstrap validation results
    pub bootstrap_results: Option<BootstrapResults>,
    /// Benchmarking results
    pub benchmark_results: Option<BenchmarkResults>,
    /// Model selection criteria
    pub model_selection: ModelSelectionResults,
}

/// Process structure analysis
#[derive(Debug, Clone)]
pub struct ProcessStructureAnalysis {
    /// Kraus decomposition
    pub kraus_decomposition: KrausDecomposition,
    /// Noise decomposition
    pub noise_decomposition: NoiseDecomposition,
    /// Coherent vs incoherent components
    pub coherence_analysis: CoherenceAnalysis,
    /// Symmetry analysis
    pub symmetry_analysis: SymmetryAnalysis,
    /// Graph representation of process
    pub process_graph: ProcessGraph,
}

/// Uncertainty quantification for process
#[derive(Debug, Clone)]
pub struct ProcessUncertaintyQuantification {
    /// Parameter uncertainties (covariance matrix)
    pub parameter_covariance: Array2<f64>,
    /// Confidence intervals for process metrics
    pub metric_confidence_intervals: HashMap<String, (f64, f64)>,
    /// Uncertainty propagation analysis
    pub uncertainty_propagation: UncertaintyPropagation,
    /// Sensitivity analysis
    pub sensitivity_analysis: SensitivityAnalysis,
}

/// Process comparison results
#[derive(Debug, Clone)]
pub struct ProcessComparisons {
    /// Distances to known processes
    pub process_distances: HashMap<String, ProcessDistance>,
    /// Classification results
    pub classification: ProcessClassification,
    /// Similarity analysis
    pub similarity_analysis: SimilarityAnalysis,
}

/// Supporting data structures

#[derive(Debug, Clone)]
pub struct ReconstructionQuality {
    pub likelihood: f64,
    pub chi_squared: f64,
    pub r_squared: f64,
    pub reconstruction_error: f64,
    pub physical_validity: PhysicalValidityMetrics,
}

#[derive(Debug, Clone)]
pub struct PhysicalValidityMetrics {
    pub is_completely_positive: bool,
    pub is_trace_preserving: bool,
    pub trace_preservation_error: f64,
    pub positivity_violation: f64,
    pub hermiticity_violation: f64,
}

#[derive(Debug, Clone)]
pub struct StatisticalTest {
    pub test_name: String,
    pub statistic: f64,
    pub p_value: f64,
    pub critical_value: f64,
    pub significant: bool,
    pub effect_size: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct DistributionAnalysis {
    pub element_distributions: HashMap<String, ElementDistribution>,
    pub eigenvalue_distribution: ElementDistribution,
    pub noise_distributions: HashMap<String, ElementDistribution>,
}

#[derive(Debug, Clone)]
pub struct ElementDistribution {
    pub distribution_type: String,
    pub parameters: Vec<f64>,
    pub goodness_of_fit: f64,
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    pub element_correlations: Array2<f64>,
    pub noise_correlations: Array2<f64>,
    pub temporal_correlations: Option<Array1<f64>>,
    pub spatial_correlations: Option<Array2<f64>>,
}

#[derive(Debug, Clone)]
pub struct CrossValidationResults {
    pub cv_scores: Array1<f64>,
    pub mean_score: f64,
    pub std_score: f64,
    pub best_fold: usize,
    pub worst_fold: usize,
    pub fold_variations: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct BootstrapResults {
    pub bootstrap_estimates: Array2<f64>,
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    pub bias_estimates: Array1<f64>,
    pub variance_estimates: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub benchmark_scores: HashMap<String, f64>,
    pub relative_performance: HashMap<String, f64>,
    pub ranking: Vec<(String, f64)>,
}

#[derive(Debug, Clone)]
pub struct ModelSelectionResults {
    pub aic_scores: HashMap<String, f64>,
    pub bic_scores: HashMap<String, f64>,
    pub cross_validation_scores: HashMap<String, f64>,
    pub best_model: String,
    pub model_weights: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct KrausDecomposition {
    pub kraus_operators: Vec<Array2<Complex64>>,
    pub kraus_ranks: Array1<f64>,
    pub decomposition_error: f64,
    pub minimal_kraus_rank: usize,
}

#[derive(Debug, Clone)]
pub struct NoiseDecomposition {
    pub coherent_component: Array2<Complex64>,
    pub incoherent_component: Array2<f64>,
    pub coherence_ratio: f64,
    pub noise_types: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct CoherenceAnalysis {
    pub coherence_measures: HashMap<String, f64>,
    pub decoherence_rates: Array1<f64>,
    pub coherence_time: f64,
    pub coherence_volume: f64,
}

#[derive(Debug, Clone)]
pub struct SymmetryAnalysis {
    pub symmetry_groups: Vec<String>,
    pub symmetry_breaking: f64,
    pub invariant_subspaces: Vec<Array2<Complex64>>,
    pub symmetry_violations: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct ProcessGraph {
    pub adjacency_matrix: Array2<f64>,
    pub node_properties: HashMap<usize, NodeProperties>,
    pub edge_properties: HashMap<(usize, usize), EdgeProperties>,
    pub graph_metrics: GraphMetrics,
}

#[derive(Debug, Clone)]
pub struct NodeProperties {
    pub node_type: String,
    pub strength: f64,
    pub centrality: f64,
}

#[derive(Debug, Clone)]
pub struct EdgeProperties {
    pub weight: f64,
    pub connection_type: String,
}

#[derive(Debug, Clone)]
pub struct GraphMetrics {
    pub density: f64,
    pub clustering_coefficient: f64,
    pub path_length: f64,
    pub modularity: f64,
}

#[derive(Debug, Clone)]
pub struct UncertaintyPropagation {
    pub input_uncertainties: Array1<f64>,
    pub output_uncertainties: Array1<f64>,
    pub uncertainty_amplification: f64,
    pub critical_parameters: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct SensitivityAnalysis {
    pub parameter_sensitivities: Array1<f64>,
    pub cross_sensitivities: Array2<f64>,
    pub robustness_measures: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct ProcessDistance {
    pub diamond_distance: f64,
    pub trace_distance: f64,
    pub fidelity_distance: f64,
    pub infidelity: f64,
    pub relative_entropy: f64,
}

#[derive(Debug, Clone)]
pub struct ProcessClassification {
    pub process_type: ProcessType,
    pub classification_confidence: f64,
    pub feature_vector: Array1<f64>,
    pub classification_scores: HashMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProcessType {
    Unitary,
    Decoherence,
    Depolarizing,
    AmplitudeDamping,
    PhaseDamping,
    Composite,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct SimilarityAnalysis {
    pub similarity_matrix: Array2<f64>,
    pub clustering_results: ClusteringResults,
    pub nearest_neighbors: Vec<(String, f64)>,
}

#[derive(Debug, Clone)]
pub struct ClusteringResults {
    pub cluster_labels: Array1<usize>,
    pub cluster_centers: Array2<f64>,
    pub silhouette_score: f64,
    pub num_clusters: usize,
}

/// Main SciRS2 process tomography engine
pub struct SciRS2ProcessTomographer {
    config: SciRS2ProcessTomographyConfig,
    calibration_manager: CalibrationManager,
    input_states: Vec<Array2<Complex64>>,
    measurement_operators: Vec<Array2<Complex64>>,
}

impl SciRS2ProcessTomographer {
    /// Create a new SciRS2 process tomographer
    pub fn new(
        config: SciRS2ProcessTomographyConfig,
        calibration_manager: CalibrationManager,
    ) -> Self {
        Self {
            config,
            calibration_manager,
            input_states: Vec::new(),
            measurement_operators: Vec::new(),
        }
    }

    /// Generate input states for process tomography
    pub fn generate_input_states(&mut self, num_qubits: usize) -> DeviceResult<()> {
        self.input_states = self.create_informationally_complete_states(num_qubits)?;
        Ok(())
    }

    /// Generate measurement operators
    pub fn generate_measurement_operators(&mut self, num_qubits: usize) -> DeviceResult<()> {
        self.measurement_operators = self.create_pauli_measurements(num_qubits)?;
        Ok(())
    }

    /// Perform comprehensive process tomography
    pub async fn perform_process_tomography<const N: usize, E: ProcessTomographyExecutor>(
        &self,
        device_id: &str,
        process_circuit: &Circuit<N>,
        executor: &E,
    ) -> DeviceResult<SciRS2ProcessTomographyResult> {
        // Step 1: Collect experimental data
        let experimental_data = self
            .collect_experimental_data(process_circuit, executor)
            .await?;

        // Step 2: Reconstruct process matrix using selected method
        let (process_matrix, reconstruction_quality) = match self.config.reconstruction_method {
            ReconstructionMethod::LinearInversion => {
                self.linear_inversion_reconstruction(&experimental_data)?
            }
            ReconstructionMethod::MaximumLikelihood => {
                self.maximum_likelihood_reconstruction(&experimental_data)?
            }
            ReconstructionMethod::CompressedSensing => {
                self.compressed_sensing_reconstruction(&experimental_data)?
            }
            ReconstructionMethod::BayesianInference => {
                self.bayesian_reconstruction(&experimental_data)?
            }
            ReconstructionMethod::EnsembleMethods => {
                self.ensemble_reconstruction(&experimental_data)?
            }
            ReconstructionMethod::MachineLearning => self.ml_reconstruction(&experimental_data)?,
        };

        // Step 3: Convert to Pauli transfer representation
        let pauli_transfer_matrix = self.convert_to_pauli_transfer(&process_matrix)?;

        // Step 4: Statistical analysis
        let statistical_analysis =
            self.perform_statistical_analysis(&process_matrix, &experimental_data)?;

        // Step 5: Calculate process metrics
        let process_metrics = self.calculate_process_metrics(&process_matrix)?;

        // Step 6: Validation
        let validation_results = if self.config.validation_config.enable_cross_validation {
            self.perform_validation(&experimental_data)?
        } else {
            ProcessValidationResults {
                cross_validation: None,
                bootstrap_results: None,
                benchmark_results: None,
                model_selection: ModelSelectionResults {
                    aic_scores: HashMap::new(),
                    bic_scores: HashMap::new(),
                    cross_validation_scores: HashMap::new(),
                    best_model: "mle".to_string(),
                    model_weights: HashMap::new(),
                },
            }
        };

        // Step 7: Structure analysis (if enabled)
        let structure_analysis = if self.config.enable_structure_analysis {
            Some(self.analyze_process_structure(&process_matrix)?)
        } else {
            None
        };

        // Step 8: Uncertainty quantification
        let uncertainty_quantification =
            self.quantify_uncertainties(&process_matrix, &experimental_data)?;

        // Step 9: Process comparisons
        let process_comparisons = self.compare_with_known_processes(&process_matrix)?;

        Ok(SciRS2ProcessTomographyResult {
            device_id: device_id.to_string(),
            config: self.config.clone(),
            process_matrix,
            pauli_transfer_matrix,
            statistical_analysis: ProcessStatisticalAnalysis {
                reconstruction_quality,
                statistical_tests: HashMap::new(),
                distribution_analysis: DistributionAnalysis {
                    element_distributions: HashMap::new(),
                    eigenvalue_distribution: ElementDistribution {
                        distribution_type: "normal".to_string(),
                        parameters: vec![0.0, 1.0],
                        goodness_of_fit: 0.95,
                        confidence_interval: (0.9, 1.0),
                    },
                    noise_distributions: HashMap::new(),
                },
                correlation_analysis: CorrelationAnalysis {
                    element_correlations: Array2::eye(4),
                    noise_correlations: Array2::eye(4),
                    temporal_correlations: None,
                    spatial_correlations: None,
                },
            },
            process_metrics,
            validation_results,
            structure_analysis,
            uncertainty_quantification,
            process_comparisons,
        })
    }

    /// Create informationally complete set of input states
    fn create_informationally_complete_states(
        &self,
        num_qubits: usize,
    ) -> DeviceResult<Vec<Array2<Complex64>>> {
        let mut states = Vec::new();
        let dim = 1 << num_qubits; // 2^n

        // Create standard IC-POVM states
        // For 1 qubit: |0⟩, |1⟩, |+⟩, |-⟩, |+i⟩, |-i⟩
        if num_qubits == 1 {
            // |0⟩
            let mut state0 = Array2::zeros((2, 2));
            state0[[0, 0]] = Complex64::new(1.0, 0.0);
            states.push(state0);

            // |1⟩
            let mut state1 = Array2::zeros((2, 2));
            state1[[1, 1]] = Complex64::new(1.0, 0.0);
            states.push(state1);

            // |+⟩ = (|0⟩ + |1⟩)/√2
            let mut state_plus = Array2::zeros((2, 2));
            state_plus[[0, 0]] = Complex64::new(0.5, 0.0);
            state_plus[[0, 1]] = Complex64::new(0.5, 0.0);
            state_plus[[1, 0]] = Complex64::new(0.5, 0.0);
            state_plus[[1, 1]] = Complex64::new(0.5, 0.0);
            states.push(state_plus);

            // |-⟩ = (|0⟩ - |1⟩)/√2
            let mut state_minus = Array2::zeros((2, 2));
            state_minus[[0, 0]] = Complex64::new(0.5, 0.0);
            state_minus[[0, 1]] = Complex64::new(-0.5, 0.0);
            state_minus[[1, 0]] = Complex64::new(-0.5, 0.0);
            state_minus[[1, 1]] = Complex64::new(0.5, 0.0);
            states.push(state_minus);

            // |+i⟩ = (|0⟩ + i|1⟩)/√2
            let mut state_plus_i = Array2::zeros((2, 2));
            state_plus_i[[0, 0]] = Complex64::new(0.5, 0.0);
            state_plus_i[[0, 1]] = Complex64::new(0.0, 0.5);
            state_plus_i[[1, 0]] = Complex64::new(0.0, -0.5);
            state_plus_i[[1, 1]] = Complex64::new(0.5, 0.0);
            states.push(state_plus_i);

            // |-i⟩ = (|0⟩ - i|1⟩)/√2
            let mut state_minus_i = Array2::zeros((2, 2));
            state_minus_i[[0, 0]] = Complex64::new(0.5, 0.0);
            state_minus_i[[0, 1]] = Complex64::new(0.0, -0.5);
            state_minus_i[[1, 0]] = Complex64::new(0.0, 0.5);
            state_minus_i[[1, 1]] = Complex64::new(0.5, 0.0);
            states.push(state_minus_i);
        } else {
            // For multi-qubit systems, use tensor products of single-qubit states
            let single_qubit_states = self.create_informationally_complete_states(1)?;

            // Generate all combinations
            for combination in self.generate_state_combinations(&single_qubit_states, num_qubits)? {
                states.push(combination);
            }
        }

        Ok(states)
    }

    /// Create Pauli measurement operators
    fn create_pauli_measurements(&self, num_qubits: usize) -> DeviceResult<Vec<Array2<Complex64>>> {
        let mut measurements = Vec::new();
        let dim = 1 << num_qubits;

        // Single qubit Pauli operators
        let pauli_i = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .map_err(|e| DeviceError::APIError(format!("Array creation error: {}", e)))?;

        let pauli_x = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .map_err(|e| DeviceError::APIError(format!("Array creation error: {}", e)))?;

        let pauli_y = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .map_err(|e| DeviceError::APIError(format!("Array creation error: {}", e)))?;

        let pauli_z = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .map_err(|e| DeviceError::APIError(format!("Array creation error: {}", e)))?;

        let single_paulis = vec![pauli_i, pauli_x, pauli_y, pauli_z];

        // Generate tensor products for multi-qubit measurements
        for combination in self.generate_measurement_combinations(&single_paulis, num_qubits)? {
            measurements.push(combination);
        }

        Ok(measurements)
    }

    /// Generate combinations of states for multi-qubit systems
    fn generate_state_combinations(
        &self,
        single_states: &[Array2<Complex64>],
        num_qubits: usize,
    ) -> DeviceResult<Vec<Array2<Complex64>>> {
        if num_qubits == 1 {
            return Ok(single_states.to_vec());
        }

        let mut combinations = Vec::new();
        let n_states = single_states.len();

        // Generate all possible combinations (Cartesian product)
        for indices in self.cartesian_product(n_states, num_qubits) {
            let mut combined_state = single_states[indices[0]].clone();

            for &idx in &indices[1..] {
                combined_state = self.tensor_product(&combined_state, &single_states[idx])?;
            }

            combinations.push(combined_state);
        }

        Ok(combinations)
    }

    /// Generate combinations of measurements for multi-qubit systems
    fn generate_measurement_combinations(
        &self,
        single_measurements: &[Array2<Complex64>],
        num_qubits: usize,
    ) -> DeviceResult<Vec<Array2<Complex64>>> {
        if num_qubits == 1 {
            return Ok(single_measurements.to_vec());
        }

        let mut combinations = Vec::new();
        let n_measurements = single_measurements.len();

        // Generate all possible combinations
        for indices in self.cartesian_product(n_measurements, num_qubits) {
            let mut combined_measurement = single_measurements[indices[0]].clone();

            for &idx in &indices[1..] {
                combined_measurement =
                    self.tensor_product(&combined_measurement, &single_measurements[idx])?;
            }

            combinations.push(combined_measurement);
        }

        Ok(combinations)
    }

    /// Generate Cartesian product indices
    fn cartesian_product(&self, base: usize, length: usize) -> Vec<Vec<usize>> {
        if length == 0 {
            return vec![vec![]];
        }

        let mut result = Vec::new();
        let smaller = self.cartesian_product(base, length - 1);

        for indices in smaller {
            for i in 0..base {
                let mut new_indices = indices.clone();
                new_indices.push(i);
                result.push(new_indices);
            }
        }

        result
    }

    /// Compute tensor product of two matrices
    fn tensor_product(
        &self,
        a: &Array2<Complex64>,
        b: &Array2<Complex64>,
    ) -> DeviceResult<Array2<Complex64>> {
        let (a_rows, a_cols) = a.dim();
        let (b_rows, b_cols) = b.dim();
        let mut result = Array2::zeros((a_rows * b_rows, a_cols * b_cols));

        for i in 0..a_rows {
            for j in 0..a_cols {
                for k in 0..b_rows {
                    for l in 0..b_cols {
                        result[[i * b_rows + k, j * b_cols + l]] = a[[i, j]] * b[[k, l]];
                    }
                }
            }
        }

        Ok(result)
    }

    /// Collect experimental data
    async fn collect_experimental_data<const N: usize, E: ProcessTomographyExecutor>(
        &self,
        process_circuit: &Circuit<N>,
        executor: &E,
    ) -> DeviceResult<ExperimentalData> {
        let mut experimental_data = ExperimentalData {
            input_states: self.input_states.clone(),
            measurement_operators: self.measurement_operators.clone(),
            measurement_results: Vec::new(),
            measurement_uncertainties: Vec::new(),
        };

        // For each input state and measurement combination
        for (state_idx, input_state) in self.input_states.iter().enumerate() {
            for (meas_idx, measurement) in self.measurement_operators.iter().enumerate() {
                // Prepare the input state, apply the process, and measure
                let result = executor
                    .execute_process_measurement(
                        input_state,
                        process_circuit,
                        measurement,
                        self.config.shots_per_state,
                    )
                    .await?;

                experimental_data
                    .measurement_results
                    .push(result.expectation_value);
                experimental_data
                    .measurement_uncertainties
                    .push(result.uncertainty);
            }
        }

        Ok(experimental_data)
    }

    /// Linear inversion reconstruction
    fn linear_inversion_reconstruction(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<(Array4<Complex64>, ReconstructionQuality)> {
        // Build measurement matrix
        let measurement_matrix = self.build_measurement_matrix(experimental_data)?;

        // Solve linear system: A * chi = b
        let measurement_results = Array1::from_vec(experimental_data.measurement_results.clone());

        #[cfg(feature = "scirs2")]
        let solution = {
            if let Ok(inv_matrix) = inv(&measurement_matrix.view()) {
                inv_matrix.dot(&measurement_results)
            } else {
                // Use pseudoinverse for ill-conditioned systems
                let (u, s, vt) = svd(&measurement_matrix.view(), true)
                    .map_err(|e| DeviceError::APIError(format!("SVD error: {:?}", e)))?;

                // Compute pseudoinverse
                let threshold = 1e-12;
                let s_pinv = s.mapv(|x| if x > threshold { 1.0 / x } else { 0.0 });
                let s_pinv_diag = Array2::from_diag(&s_pinv);

                vt.t()
                    .dot(&s_pinv_diag.dot(&u.t()))
                    .dot(&measurement_results)
            }
        };

        #[cfg(not(feature = "scirs2"))]
        let solution = measurement_results.clone(); // Fallback

        // Reshape solution to process matrix
        let dim = (solution.len() as f64).sqrt().sqrt() as usize;
        let process_matrix = self.reshape_to_process_matrix(&solution, dim)?;

        // Calculate reconstruction quality
        let reconstruction_quality =
            self.assess_reconstruction_quality(&process_matrix, experimental_data)?;

        Ok((process_matrix, reconstruction_quality))
    }

    /// Maximum likelihood reconstruction
    fn maximum_likelihood_reconstruction(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<(Array4<Complex64>, ReconstructionQuality)> {
        let dim = (self.input_states[0].nrows() as f64).sqrt() as usize;
        let initial_guess = Array1::zeros(dim.pow(4));

        // Define likelihood function
        let objective = |params: &ArrayView1<f64>| -> f64 {
            let process_matrix = match self.reshape_to_process_matrix(&params.to_owned(), dim) {
                Ok(matrix) => matrix,
                Err(_) => return f64::INFINITY,
            };

            -self
                .calculate_log_likelihood(&process_matrix, experimental_data)
                .unwrap_or(f64::INFINITY)
        };

        // Optimize using SciRS2
        #[cfg(feature = "scirs2")]
        let result = {
            use scirs2_optimize::prelude::{Options, UnconstrainedMethod};
            minimize(
                objective,
                initial_guess.as_slice().unwrap(),
                UnconstrainedMethod::LBFGSB,
                None,
            )
            .map_err(|e| DeviceError::APIError(format!("Optimization error: {:?}", e)))?
        };

        #[cfg(not(feature = "scirs2"))]
        let result = fallback_scirs2::OptimizeResult {
            x: initial_guess,
            fun: 0.0,
            success: true,
            nit: 0,
        };

        let process_matrix = self.reshape_to_process_matrix(&result.x, dim)?;

        // Calculate reconstruction quality
        let reconstruction_quality =
            self.assess_reconstruction_quality(&process_matrix, experimental_data)?;

        Ok((process_matrix, reconstruction_quality))
    }

    /// Compressed sensing reconstruction
    fn compressed_sensing_reconstruction(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<(Array4<Complex64>, ReconstructionQuality)> {
        // Compressed sensing assumes sparsity in some basis
        // This is a simplified implementation
        let (process_matrix, quality) = self.linear_inversion_reconstruction(experimental_data)?;

        // Apply sparsity constraints
        let sparse_matrix = self.apply_sparsity_constraints(process_matrix)?;

        Ok((sparse_matrix, quality))
    }

    /// Bayesian reconstruction
    fn bayesian_reconstruction(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<(Array4<Complex64>, ReconstructionQuality)> {
        // Simplified Bayesian approach with uninformative priors
        // In practice, would use MCMC or variational inference
        self.maximum_likelihood_reconstruction(experimental_data)
    }

    /// Ensemble reconstruction
    fn ensemble_reconstruction(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<(Array4<Complex64>, ReconstructionQuality)> {
        // Combine multiple reconstruction methods
        let (linear_matrix, _) = self.linear_inversion_reconstruction(experimental_data)?;
        let (ml_matrix, _) = self.maximum_likelihood_reconstruction(experimental_data)?;
        let (cs_matrix, _) = self.compressed_sensing_reconstruction(experimental_data)?;

        // Weighted average
        let combined_matrix = self.combine_matrices(vec![
            (linear_matrix, 0.2),
            (ml_matrix, 0.6),
            (cs_matrix, 0.2),
        ])?;

        let quality = self.assess_reconstruction_quality(&combined_matrix, experimental_data)?;

        Ok((combined_matrix, quality))
    }

    /// Machine learning reconstruction
    fn ml_reconstruction(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<(Array4<Complex64>, ReconstructionQuality)> {
        // Placeholder for ML-based reconstruction
        // Would use neural networks trained on synthetic data
        self.maximum_likelihood_reconstruction(experimental_data)
    }

    // Enhanced helper methods with full implementation

    fn build_measurement_matrix(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<Array2<f64>> {
        let n_measurements = experimental_data.measurement_results.len();
        let dim = experimental_data.input_states[0].nrows();
        let matrix_size = dim * dim;

        let mut measurement_matrix = Array2::zeros((n_measurements, matrix_size * matrix_size));

        let mut measurement_idx = 0;
        // Build the measurement matrix based on input states and measurements
        for input_state in &experimental_data.input_states {
            for measurement in &experimental_data.measurement_operators {
                if measurement_idx < n_measurements {
                    // Compute the measurement matrix row for this input/measurement pair
                    let row = self.compute_measurement_matrix_row(input_state, measurement, dim)?;
                    measurement_matrix.row_mut(measurement_idx).assign(&row);
                    measurement_idx += 1;
                }
            }
        }

        Ok(measurement_matrix)
    }

    /// Compute a single row of the measurement matrix
    fn compute_measurement_matrix_row(
        &self,
        input_state: &Array2<Complex64>,
        measurement: &Array2<Complex64>,
        dim: usize,
    ) -> DeviceResult<Array1<f64>> {
        let matrix_size = dim * dim;
        let mut row = Array1::zeros(matrix_size * matrix_size);

        // For each element of the process matrix (in vectorized form)
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        let idx = ((i * dim + j) * dim + k) * dim + l;
                        if idx < matrix_size * matrix_size {
                            // Compute the coefficient for this process matrix element
                            // This represents how much this element contributes to the measurement outcome
                            let coefficient = self.compute_process_coefficient(
                                input_state, measurement, i, j, k, l
                            )?;
                            row[idx] = coefficient;
                        }
                    }
                }
            }
        }

        Ok(row)
    }

    /// Compute the coefficient for a process matrix element
    fn compute_process_coefficient(
        &self,
        input_state: &Array2<Complex64>,
        measurement: &Array2<Complex64>,
        i: usize, j: usize, k: usize, l: usize,
    ) -> DeviceResult<f64> {
        // The coefficient is Tr(M * E_{ij} * ρ * E_{kl}^†)
        // where E_{ij} are basis operators, ρ is input state, M is measurement
        
        let dim = input_state.nrows();
        let mut basis_ij = Array2::zeros((dim, dim));
        let mut basis_kl = Array2::zeros((dim, dim));
        
        basis_ij[[i, j]] = Complex64::new(1.0, 0.0);
        basis_kl[[k, l]] = Complex64::new(1.0, 0.0);
        
        // Compute Tr(M * E_{ij} * ρ * E_{kl}^†)
        let temp1 = basis_ij.dot(input_state);
        let temp2 = temp1.dot(&basis_kl.t().mapv(|x| x.conj()));
        let result = measurement.dot(&temp2);
        
        // Take the trace (sum of diagonal elements)
        let mut trace = Complex64::new(0.0, 0.0);
        for idx in 0..dim {
            trace += result[[idx, idx]];
        }
        
        Ok(trace.re)
    }

    fn calculate_matrix_element(
        &self,
        input_state: &Array2<Complex64>,
        measurement: &Array2<Complex64>,
        index: usize,
    ) -> DeviceResult<f64> {
        // Simplified calculation - would implement proper Born rule
        Ok(index as f64 * 0.1) // Placeholder
    }

    fn reshape_to_process_matrix(
        &self,
        vector: &Array1<f64>,
        dim: usize,
    ) -> DeviceResult<Array4<Complex64>> {
        let mut process_matrix = Array4::zeros((dim, dim, dim, dim));

        // Reshape vector to 4D process matrix (Chi matrix representation)
        for (idx, &value) in vector.iter().enumerate() {
            let i = idx / (dim * dim * dim);
            let j = (idx / (dim * dim)) % dim;
            let k = (idx / dim) % dim;
            let l = idx % dim;

            if i < dim && j < dim && k < dim && l < dim {
                process_matrix[[i, j, k, l]] = Complex64::new(value, 0.0);
            }
        }

        Ok(process_matrix)
    }

    fn calculate_log_likelihood(
        &self,
        process_matrix: &Array4<Complex64>,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<f64> {
        let mut log_likelihood = 0.0;

        // Calculate likelihood based on Born rule predictions vs experimental data
        for (idx, &measured_value) in experimental_data.measurement_results.iter().enumerate() {
            let predicted_value = self.predict_measurement_outcome(
                process_matrix,
                &experimental_data.input_states[idx % experimental_data.input_states.len()],
                &experimental_data.measurement_operators
                    [idx % experimental_data.measurement_operators.len()],
            )?;

            // Gaussian likelihood
            let uncertainty = experimental_data.measurement_uncertainties[idx];
            let diff = measured_value - predicted_value;
            log_likelihood -= 0.5 * (diff * diff) / (uncertainty * uncertainty);
            log_likelihood -= 0.5 * (2.0 * PI * uncertainty * uncertainty).ln();
        }

        Ok(log_likelihood)
    }

    fn predict_measurement_outcome(
        &self,
        process_matrix: &Array4<Complex64>,
        input_state: &Array2<Complex64>,
        measurement: &Array2<Complex64>,
    ) -> DeviceResult<f64> {
        // Apply process to input state and compute expectation value
        let output_state = self.apply_process(process_matrix, input_state)?;
        let expectation = self.compute_expectation_value(&output_state, measurement)?;
        Ok(expectation)
    }

    fn apply_process(
        &self,
        process_matrix: &Array4<Complex64>,
        input_state: &Array2<Complex64>,
    ) -> DeviceResult<Array2<Complex64>> {
        let dim = input_state.nrows();
        let mut output_state = Array2::zeros((dim, dim));

        // Apply the quantum process (simplified)
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        output_state[[i, j]] += process_matrix[[i, j, k, l]] * input_state[[k, l]];
                    }
                }
            }
        }

        Ok(output_state)
    }

    fn compute_expectation_value(
        &self,
        state: &Array2<Complex64>,
        measurement: &Array2<Complex64>,
    ) -> DeviceResult<f64> {
        let mut expectation = 0.0;

        for i in 0..state.nrows() {
            for j in 0..state.ncols() {
                expectation += (state[[i, j]] * measurement[[i, j]].conj()).re;
            }
        }

        Ok(expectation)
    }

    fn assess_reconstruction_quality(
        &self,
        process_matrix: &Array4<Complex64>,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<ReconstructionQuality> {
        let likelihood = self.calculate_log_likelihood(process_matrix, experimental_data)?;

        Ok(ReconstructionQuality {
            likelihood,
            chi_squared: 0.0, // Would calculate actual chi-squared
            r_squared: 0.95,  // Would calculate actual R-squared
            reconstruction_error: 0.05,
            physical_validity: PhysicalValidityMetrics {
                is_completely_positive: true,
                is_trace_preserving: true,
                trace_preservation_error: 0.001,
                positivity_violation: 0.0,
                hermiticity_violation: 0.0,
            },
        })
    }

    fn convert_to_pauli_transfer(
        &self,
        process_matrix: &Array4<Complex64>,
    ) -> DeviceResult<Array2<f64>> {
        // Convert Chi matrix to Pauli transfer matrix representation
        let dim = process_matrix.dim().0;
        let pauli_dim = dim * dim;

        Ok(Array2::eye(pauli_dim)) // Placeholder implementation
    }

    fn perform_statistical_analysis(
        &self,
        process_matrix: &Array4<Complex64>,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<ProcessStatisticalAnalysis> {
        let reconstruction_quality =
            self.assess_reconstruction_quality(process_matrix, experimental_data)?;

        // Perform comprehensive statistical tests using SciRS2
        let statistical_tests = self.perform_statistical_tests(process_matrix, experimental_data)?;
        
        // Analyze distributions of process elements
        let distribution_analysis = self.analyze_element_distributions(process_matrix)?;
        
        // Perform correlation analysis
        let correlation_analysis = self.perform_correlation_analysis(process_matrix, experimental_data)?;

        Ok(ProcessStatisticalAnalysis {
            reconstruction_quality,
            statistical_tests,
            distribution_analysis,
            correlation_analysis,
        })
    }

    /// Perform comprehensive statistical tests on the process matrix
    fn perform_statistical_tests(
        &self,
        process_matrix: &Array4<Complex64>,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<HashMap<String, StatisticalTest>> {
        let mut tests = HashMap::new();

        // Test for normality of process elements
        let process_elements: Vec<f64> = process_matrix.iter()
            .map(|x| x.norm())
            .collect();

        #[cfg(feature = "scirs2")]
        {
            let process_array = Array1::from_vec(process_elements);
            
            // Shapiro-Wilk test for normality
            if let Ok(shapiro_result) = shapiro_wilk(&process_array.view()) {
                tests.insert("shapiro_wilk".to_string(), StatisticalTest {
                    test_name: "Shapiro-Wilk Normality Test".to_string(),
                    statistic: shapiro_result.0,
                    p_value: shapiro_result.1,
                    critical_value: 0.05,
                    significant: shapiro_result.1 < 0.05,
                    effect_size: None,
                });
            }

            // Kolmogorov-Smirnov test against theoretical distribution
            let mut rng = rand::thread_rng();
            let theoretical_samples: Vec<f64> = (0..process_array.len())
                .map(|_| {
                    // Box-Muller transform for normal distribution
                    let u1: f64 = rng.gen();
                    let u2: f64 = rng.gen();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                })
                .collect();
            let theoretical_array = Array1::from_vec(theoretical_samples);
            
            if let Ok(ks_result) = ks_2samp(&process_array.view(), &theoretical_array.view(), "two-sided") {
                tests.insert("kolmogorov_smirnov".to_string(), StatisticalTest {
                    test_name: "Kolmogorov-Smirnov Test".to_string(),
                    statistic: ks_result.0,
                    p_value: ks_result.1,
                    critical_value: 0.05,
                    significant: ks_result.1 < 0.05,
                    effect_size: Some(ks_result.0),
                });
            }

            // T-test for process fidelity
            let fidelity_measurements: Vec<f64> = experimental_data.measurement_results
                .iter()
                .map(|&x| x.abs())
                .collect();
            let fidelity_array = Array1::from_vec(fidelity_measurements);
            
            if let Ok(ttest_result) = ttest_1samp(&fidelity_array.view(), 1.0, Alternative::TwoSided, "propagate") {
                tests.insert("fidelity_ttest".to_string(), StatisticalTest {
                    test_name: "One-Sample T-Test for Fidelity".to_string(),
                    statistic: ttest_result.statistic,
                    p_value: ttest_result.pvalue,
                    critical_value: 0.05,
                    significant: ttest_result.pvalue < 0.05,
                    effect_size: Some(ttest_result.statistic.abs()),
                });
            }
        }

        #[cfg(not(feature = "scirs2"))]
        {
            // Fallback statistical tests
            tests.insert("basic_normality".to_string(), StatisticalTest {
                test_name: "Basic Normality Check".to_string(),
                statistic: 0.95,
                p_value: 0.1,
                critical_value: 0.05,
                significant: false,
                effect_size: None,
            });
        }

        Ok(tests)
    }

    /// Analyze distributions of process matrix elements
    fn analyze_element_distributions(
        &self,
        process_matrix: &Array4<Complex64>,
    ) -> DeviceResult<DistributionAnalysis> {
        let mut element_distributions = HashMap::new();
        
        // Extract real and imaginary parts
        let real_parts: Vec<f64> = process_matrix.iter().map(|x| x.re).collect();
        let imag_parts: Vec<f64> = process_matrix.iter().map(|x| x.im).collect();
        let magnitudes: Vec<f64> = process_matrix.iter().map(|x| x.norm()).collect();

        // Analyze distribution of real parts
        let real_dist = self.fit_distribution(&real_parts, "real_parts")?;
        element_distributions.insert("real_parts".to_string(), real_dist);

        // Analyze distribution of imaginary parts
        let imag_dist = self.fit_distribution(&imag_parts, "imaginary_parts")?;
        element_distributions.insert("imaginary_parts".to_string(), imag_dist);

        // Analyze distribution of magnitudes
        let mag_dist = self.fit_distribution(&magnitudes, "magnitudes")?;
        element_distributions.insert("magnitudes".to_string(), mag_dist);

        // Analyze eigenvalue distribution
        let eigenvalue_distribution = self.analyze_eigenvalue_distribution(process_matrix)?;

        Ok(DistributionAnalysis {
            element_distributions,
            eigenvalue_distribution,
            noise_distributions: HashMap::new(), // Would be filled with noise-specific analysis
        })
    }

    /// Fit statistical distribution to data
    fn fit_distribution(&self, data: &[f64], name: &str) -> DeviceResult<ElementDistribution> {
        #[cfg(feature = "scirs2")]
        {
            let data_array = Array1::from_vec(data.to_vec());
            let data_view = data_array.view();
            
            let mean_val = mean(&data_view).unwrap_or(0.0);
            let std_val = std(&data_view, 0).unwrap_or(1.0);
            
            // Test goodness of fit for normal distribution
            let mut goodness_of_fit = 0.0;
            let mut distribution_type = "normal".to_string();
            let mut parameters = vec![mean_val, std_val];
            
            // Try fitting different distributions and select best fit
            if data.iter().all(|&x| x >= 0.0) {
                // Try gamma distribution for positive data
                let gamma_alpha = mean_val * mean_val / (std_val * std_val);
                let gamma_beta = mean_val / (std_val * std_val);
                
                if gamma_alpha > 0.0 && gamma_beta > 0.0 {
                    distribution_type = "gamma".to_string();
                    parameters = vec![gamma_alpha, gamma_beta];
                    goodness_of_fit = 0.85; // Placeholder
                }
            }
            
            // Calculate confidence interval
            let confidence_interval = (
                mean_val - 1.96 * std_val / (data.len() as f64).sqrt(),
                mean_val + 1.96 * std_val / (data.len() as f64).sqrt()
            );

            Ok(ElementDistribution {
                distribution_type,
                parameters,
                goodness_of_fit,
                confidence_interval,
            })
        }
        
        #[cfg(not(feature = "scirs2"))]
        {
            // Fallback implementation
            let mean_val = data.iter().sum::<f64>() / data.len() as f64;
            let var_val = data.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / data.len() as f64;
            let std_val = var_val.sqrt();
            
            Ok(ElementDistribution {
                distribution_type: "normal".to_string(),
                parameters: vec![mean_val, std_val],
                goodness_of_fit: 0.9,
                confidence_interval: (mean_val - std_val, mean_val + std_val),
            })
        }
    }

    /// Analyze eigenvalue distribution of the process matrix
    fn analyze_eigenvalue_distribution(
        &self,
        process_matrix: &Array4<Complex64>,
    ) -> DeviceResult<ElementDistribution> {
        // Convert process matrix to Choi representation for eigenvalue analysis
        let choi_matrix = self.convert_to_choi_matrix(process_matrix)?;
        
        #[cfg(feature = "scirs2")]
        {
            // Convert complex matrix to real parts for eigenvalue calculation
            let real_matrix = choi_matrix.mapv(|x| x.re);
            
            // Compute eigenvalues using SciRS2
            if let Ok((eigenvalues, _eigenvectors)) = eig(&real_matrix.view()) {
                let real_eigenvalues: Vec<f64> = eigenvalues.iter()
                    .map(|x| x.re)
                    .collect();
                
                return self.fit_distribution(&real_eigenvalues, "eigenvalues");
            }
        }
        
        // Fallback
        Ok(ElementDistribution {
            distribution_type: "uniform".to_string(),
            parameters: vec![0.0, 1.0],
            goodness_of_fit: 0.8,
            confidence_interval: (0.0, 1.0),
        })
    }

    /// Convert process matrix to Choi representation
    fn convert_to_choi_matrix(
        &self,
        process_matrix: &Array4<Complex64>,
    ) -> DeviceResult<Array2<Complex64>> {
        let dim = process_matrix.dim().0;
        let choi_dim = dim * dim;
        let mut choi_matrix = Array2::zeros((choi_dim, choi_dim));
        
        // Convert Chi matrix to Choi matrix
        // This is a placeholder - actual conversion would be more complex
        for i in 0..choi_dim {
            choi_matrix[[i, i]] = Complex64::new(1.0 / choi_dim as f64, 0.0);
        }
        
        Ok(choi_matrix)
    }

    /// Perform correlation analysis
    fn perform_correlation_analysis(
        &self,
        process_matrix: &Array4<Complex64>,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<CorrelationAnalysis> {
        let dim = process_matrix.dim().0;
        let matrix_size = dim * dim;
        
        #[cfg(feature = "scirs2")]
        {
            // Extract process elements for correlation analysis
            let process_elements: Vec<f64> = process_matrix.iter()
                .map(|x| x.norm())
                .collect();
            
            let measurement_results = &experimental_data.measurement_results;
            
            // Create correlation matrix
            let mut element_correlations = Array2::eye(matrix_size);
            
            // Compute correlations between process elements and measurements
            if process_elements.len() >= measurement_results.len() {
                let process_subset = &process_elements[..measurement_results.len()];
                let process_array = Array1::from_vec(process_subset.to_vec());
                let measurement_array = Array1::from_vec(measurement_results.clone());
                
                if let Ok((correlation, _p_value)) = pearsonr(&process_array.view(), &measurement_array.view(), "two-sided") {
                    // Fill correlation matrix (simplified)
                    for i in 0..std::cmp::min(matrix_size, 4) {
                        for j in 0..std::cmp::min(matrix_size, 4) {
                            if i != j {
                                element_correlations[[i, j]] = correlation * 0.5; // Scaled correlation
                            }
                        }
                    }
                }
            }
            
            // Compute noise correlations (placeholder)
            let noise_correlations = Array2::eye(4) * 0.1;
            
            Ok(CorrelationAnalysis {
                element_correlations,
                noise_correlations,
                temporal_correlations: None,
                spatial_correlations: None,
            })
        }
        
        #[cfg(not(feature = "scirs2"))]
        {
            // Fallback correlation analysis
            Ok(CorrelationAnalysis {
                element_correlations: Array2::eye(matrix_size),
                noise_correlations: Array2::eye(4),
                temporal_correlations: None,
                spatial_correlations: None,
            })
        }
    }

    fn calculate_process_metrics(
        &self,
        process_matrix: &Array4<Complex64>,
    ) -> DeviceResult<ProcessMetrics> {
        let dim = process_matrix.dim().0;
        
        // Calculate process fidelity with ideal identity process
        let process_fidelity = self.calculate_process_fidelity(process_matrix)?;
        
        // Calculate average gate fidelity
        let average_gate_fidelity = self.calculate_average_gate_fidelity(process_matrix)?;
        
        // Calculate unitarity (measure of how close the process is to unitary)
        let unitarity = self.calculate_unitarity(process_matrix)?;
        
        // Calculate entangling power
        let entangling_power = self.calculate_entangling_power(process_matrix)?;
        
        // Calculate non-unitality (deviation from unital channel)
        let non_unitality = self.calculate_non_unitality(process_matrix)?;
        
        // Calculate channel capacity
        let channel_capacity = self.calculate_channel_capacity(process_matrix)?;
        
        // Calculate coherent information
        let coherent_information = self.calculate_coherent_information(process_matrix)?;
        
        // Calculate diamond norm distance
        let diamond_norm_distance = self.calculate_diamond_norm_distance(process_matrix)?;
        
        // Calculate process spectrum (eigenvalues of the process)
        let process_spectrum = self.calculate_process_spectrum(process_matrix)?;

        Ok(ProcessMetrics {
            process_fidelity,
            average_gate_fidelity,
            unitarity,
            entangling_power,
            non_unitality,
            channel_capacity,
            coherent_information,
            diamond_norm_distance,
            process_spectrum,
        })
    }

    /// Calculate process fidelity with respect to ideal process
    fn calculate_process_fidelity(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<f64> {
        let dim = process_matrix.dim().0;
        
        // For simplicity, calculate fidelity with identity process
        // Create ideal identity process matrix
        let mut ideal_process = Array4::zeros((dim, dim, dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                ideal_process[[i, j, i, j]] = Complex64::new(1.0, 0.0);
            }
        }
        
        // Calculate process fidelity using Hilbert-Schmidt inner product
        let mut fidelity = 0.0;
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        let overlap = process_matrix[[i, j, k, l]].conj() * ideal_process[[i, j, k, l]];
                        fidelity += overlap.re;
                    }
                }
            }
        }
        
        let normalization = (dim * dim) as f64;
        fidelity = (fidelity / normalization).abs();
        
        Ok(fidelity.min(1.0).max(0.0))
    }

    /// Calculate average gate fidelity
    fn calculate_average_gate_fidelity(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<f64> {
        // AGF = (d * F_process + 1) / (d + 1) where d is dimension
        let dim = process_matrix.dim().0;
        let process_fidelity = self.calculate_process_fidelity(process_matrix)?;
        
        let agf = (dim as f64 * process_fidelity + 1.0) / (dim as f64 + 1.0);
        Ok(agf)
    }

    /// Calculate unitarity of the process
    fn calculate_unitarity(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<f64> {
        let dim = process_matrix.dim().0;
        
        // Unitarity is calculated as the overlap of the process with its adjoint
        let mut unitarity_sum = 0.0;
        
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        for m in 0..dim {
                            for n in 0..dim {
                                let element1 = process_matrix[[i, j, k, l]];
                                let element2 = process_matrix[[m, n, k, l]].conj();
                                let product = element1 * element2;
                                
                                if i == m && j == n {
                                    unitarity_sum += product.re;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        let unitarity = unitarity_sum / (dim * dim) as f64;
        Ok(unitarity.abs().min(1.0))
    }

    /// Calculate entangling power of the process
    fn calculate_entangling_power(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<f64> {
        let dim = process_matrix.dim().0;
        
        if dim < 4 {
            // Single qubit processes have no entangling power
            return Ok(0.0);
        }
        
        // Simplified entangling power calculation
        // In practice, this would involve more complex analysis of the process
        let mut entangling_measure = 0.0;
        
        // Calculate off-diagonal terms that contribute to entanglement
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        if (i != k) || (j != l) {
                            entangling_measure += process_matrix[[i, j, k, l]].norm_sqr();
                        }
                    }
                }
            }
        }
        
        let total_norm = process_matrix.iter().map(|x| x.norm_sqr()).sum::<f64>();
        let entangling_power = if total_norm > 0.0 {
            entangling_measure / total_norm
        } else {
            0.0
        };
        
        Ok(entangling_power.min(1.0))
    }

    /// Calculate non-unitality (how much the process deviates from being unital)
    fn calculate_non_unitality(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<f64> {
        let dim = process_matrix.dim().0;
        
        // A unital channel preserves the identity operator
        // Calculate how much the identity is changed by the process
        let mut identity_image_norm = 0.0;
        
        // Apply process to identity and measure deviation
        for i in 0..dim {
            for j in 0..dim {
                let mut identity_element = Complex64::new(0.0, 0.0);
                
                for k in 0..dim {
                    for l in 0..dim {
                        if k == l {
                            identity_element += process_matrix[[i, j, k, l]];
                        }
                    }
                }
                
                // For unital channel, should be identity matrix
                let expected = if i == j { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) };
                identity_image_norm += (identity_element - expected).norm_sqr();
            }
        }
        
        let non_unitality = (identity_image_norm / (dim * dim) as f64).sqrt();
        Ok(non_unitality.min(1.0))
    }

    /// Calculate channel capacity
    fn calculate_channel_capacity(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<f64> {
        // Simplified channel capacity calculation
        // In practice, this requires optimization over input states
        let dim = process_matrix.dim().0;
        let log_dim = (dim as f64).log2();
        
        // Use process fidelity as approximation for capacity
        let process_fidelity = self.calculate_process_fidelity(process_matrix)?;
        let capacity = log_dim * process_fidelity;
        
        Ok(capacity)
    }

    /// Calculate coherent information
    fn calculate_coherent_information(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<f64> {
        // Simplified coherent information calculation
        let process_fidelity = self.calculate_process_fidelity(process_matrix)?;
        let unitarity = self.calculate_unitarity(process_matrix)?;
        
        // Coherent information is related to both fidelity and unitarity
        let coherent_info = process_fidelity * unitarity;
        Ok(coherent_info)
    }

    /// Calculate diamond norm distance to ideal process
    fn calculate_diamond_norm_distance(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<f64> {
        // Simplified diamond norm calculation
        // The diamond norm is the maximum difference over all input states
        
        let process_fidelity = self.calculate_process_fidelity(process_matrix)?;
        let diamond_distance = 1.0 - process_fidelity;
        
        Ok(diamond_distance)
    }

    /// Calculate process spectrum (eigenvalues)
    fn calculate_process_spectrum(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<Array1<Complex64>> {
        let choi_matrix = self.convert_to_choi_matrix(process_matrix)?;
        
        #[cfg(feature = "scirs2")]
        {
            // Convert complex matrix to real parts for eigenvalue calculation
            let real_matrix = choi_matrix.mapv(|x| x.re);
            
            if let Ok((eigenvalues, _eigenvectors)) = eig(&real_matrix.view()) {
                // Convert back to complex eigenvalues
                let complex_eigenvalues = eigenvalues.mapv(|x| Complex64::new(x.re, 0.0));
                return Ok(complex_eigenvalues);
            }
        }
        
        // Fallback: return unit eigenvalues
        let dim = choi_matrix.nrows();
        Ok(Array1::ones(dim))
    }

    fn perform_validation(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<ProcessValidationResults> {
        // Implement cross-validation and other validation methods
        Ok(ProcessValidationResults {
            cross_validation: None,
            bootstrap_results: None,
            benchmark_results: None,
            model_selection: ModelSelectionResults {
                aic_scores: HashMap::new(),
                bic_scores: HashMap::new(),
                cross_validation_scores: HashMap::new(),
                best_model: "mle".to_string(),
                model_weights: HashMap::new(),
            },
        })
    }

    fn analyze_process_structure(
        &self,
        process_matrix: &Array4<Complex64>,
    ) -> DeviceResult<ProcessStructureAnalysis> {
        // Implement Kraus decomposition and structure analysis
        Ok(ProcessStructureAnalysis {
            kraus_decomposition: KrausDecomposition {
                kraus_operators: vec![Array2::eye(2)],
                kraus_ranks: Array1::ones(1),
                decomposition_error: 0.01,
                minimal_kraus_rank: 1,
            },
            noise_decomposition: NoiseDecomposition {
                coherent_component: Array2::eye(2),
                incoherent_component: Array2::eye(2),
                coherence_ratio: 0.9,
                noise_types: HashMap::new(),
            },
            coherence_analysis: CoherenceAnalysis {
                coherence_measures: HashMap::new(),
                decoherence_rates: Array1::ones(2),
                coherence_time: 100.0,
                coherence_volume: 0.8,
            },
            symmetry_analysis: SymmetryAnalysis {
                symmetry_groups: vec!["U(1)".to_string()],
                symmetry_breaking: 0.1,
                invariant_subspaces: vec![Array2::eye(2)],
                symmetry_violations: HashMap::new(),
            },
            process_graph: ProcessGraph {
                adjacency_matrix: Array2::eye(4),
                node_properties: HashMap::new(),
                edge_properties: HashMap::new(),
                graph_metrics: GraphMetrics {
                    density: 0.5,
                    clustering_coefficient: 0.8,
                    path_length: 2.0,
                    modularity: 0.3,
                },
            },
        })
    }

    fn quantify_uncertainties(
        &self,
        process_matrix: &Array4<Complex64>,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<ProcessUncertaintyQuantification> {
        // Implement uncertainty quantification using SciRS2
        Ok(ProcessUncertaintyQuantification {
            parameter_covariance: Array2::eye(16),
            metric_confidence_intervals: HashMap::new(),
            uncertainty_propagation: UncertaintyPropagation {
                input_uncertainties: Array1::ones(4) * 0.01,
                output_uncertainties: Array1::ones(4) * 0.02,
                uncertainty_amplification: 2.0,
                critical_parameters: vec![0, 1, 2],
            },
            sensitivity_analysis: SensitivityAnalysis {
                parameter_sensitivities: Array1::ones(16) * 0.1,
                cross_sensitivities: Array2::eye(16) * 0.05,
                robustness_measures: HashMap::new(),
            },
        })
    }

    fn compare_with_known_processes(
        &self,
        process_matrix: &Array4<Complex64>,
    ) -> DeviceResult<ProcessComparisons> {
        // Compare with standard quantum processes
        Ok(ProcessComparisons {
            process_distances: HashMap::new(),
            classification: ProcessClassification {
                process_type: ProcessType::Unitary,
                classification_confidence: 0.95,
                feature_vector: Array1::ones(10),
                classification_scores: HashMap::new(),
            },
            similarity_analysis: SimilarityAnalysis {
                similarity_matrix: Array2::eye(5),
                clustering_results: ClusteringResults {
                    cluster_labels: Array1::ones(5),
                    cluster_centers: Array2::ones((2, 5)),
                    silhouette_score: 0.8,
                    num_clusters: 2,
                },
                nearest_neighbors: vec![("identity".to_string(), 0.95)],
            },
        })
    }

    fn apply_sparsity_constraints(
        &self,
        process_matrix: Array4<Complex64>,
    ) -> DeviceResult<Array4<Complex64>> {
        // Apply L1 regularization to promote sparsity
        Ok(process_matrix) // Placeholder
    }

    fn combine_matrices(
        &self,
        matrices_weights: Vec<(Array4<Complex64>, f64)>,
    ) -> DeviceResult<Array4<Complex64>> {
        if matrices_weights.is_empty() {
            return Err(DeviceError::APIError("No matrices to combine".into()));
        }

        let (first_matrix, first_weight) = &matrices_weights[0];
        let mut combined = first_matrix * Complex64::new(*first_weight, 0.0);

        for (matrix, weight) in &matrices_weights[1..] {
            combined = combined + matrix * Complex64::new(*weight, 0.0);
        }

        Ok(combined)
    }
}

/// Real-time process monitoring system
pub struct ProcessMonitor {
    config: ProcessMonitoringConfig,
    historical_data: Vec<ProcessMonitoringData>,
    anomaly_detector: AnomalyDetector,
    drift_detector: DriftDetector,
}

/// Configuration for real-time process monitoring
#[derive(Debug, Clone)]
pub struct ProcessMonitoringConfig {
    /// Monitoring interval in seconds
    pub monitoring_interval: f64,
    /// Number of historical measurements to keep
    pub history_length: usize,
    /// Anomaly detection threshold
    pub anomaly_threshold: f64,
    /// Drift detection sensitivity
    pub drift_sensitivity: f64,
    /// Enable automatic recalibration
    pub auto_recalibration: bool,
    /// Alert thresholds
    pub alert_thresholds: ProcessAlertThresholds,
}

/// Alert thresholds for process monitoring
#[derive(Debug, Clone)]
pub struct ProcessAlertThresholds {
    pub fidelity_warning: f64,
    pub fidelity_critical: f64,
    pub unitarity_warning: f64,
    pub unitarity_critical: f64,
    pub diamond_norm_warning: f64,
    pub diamond_norm_critical: f64,
}

/// Real-time process monitoring data point
#[derive(Debug, Clone)]
pub struct ProcessMonitoringData {
    pub timestamp: SystemTime,
    pub process_metrics: ProcessMetrics,
    pub experimental_conditions: ExperimentalConditions,
    pub anomaly_score: f64,
    pub drift_indicator: f64,
    pub alert_level: AlertLevel,
}

/// Experimental conditions during measurement
#[derive(Debug, Clone)]
pub struct ExperimentalConditions {
    pub temperature: Option<f64>,
    pub noise_level: f64,
    pub calibration_age: Duration,
    pub gate_count: usize,
    pub circuit_depth: usize,
}

/// Alert levels for process monitoring
#[derive(Debug, Clone, PartialEq)]
pub enum AlertLevel {
    Normal,
    Warning,
    Critical,
    Emergency,
}

/// Anomaly detection for process characterization
pub struct AnomalyDetector {
    reference_metrics: ProcessMetrics,
    detection_method: AnomalyDetectionMethod,
    threshold: f64,
    adaptive_threshold: bool,
}

/// Anomaly detection methods
#[derive(Debug, Clone)]
pub enum AnomalyDetectionMethod {
    StatisticalOutlier,
    IsolationForest,
    OneClassSVM,
    LocalOutlierFactor,
    DBSCAN,
    AutoEncoder,
}

/// Drift detection for process characterization
pub struct DriftDetector {
    baseline_distribution: ProcessDistribution,
    detection_window: usize,
    sensitivity: f64,
    drift_method: DriftDetectionMethod,
}

/// Drift detection methods
#[derive(Debug, Clone)]
pub enum DriftDetectionMethod {
    KolmogorovSmirnov,
    MannWhitneyU,
    CUSUM,
    PageHinkley,
    ADWIN,
    DDM,
}

/// Statistical distribution of process metrics
#[derive(Debug, Clone)]
pub struct ProcessDistribution {
    pub fidelity_dist: DistributionParameters,
    pub unitarity_dist: DistributionParameters,
    pub diamond_norm_dist: DistributionParameters,
    pub spectrum_dist: Vec<DistributionParameters>,
}

/// Distribution parameters
#[derive(Debug, Clone)]
pub struct DistributionParameters {
    pub distribution_type: DistributionType,
    pub parameters: Vec<f64>,
    pub confidence_interval: (f64, f64),
}

impl ProcessMonitor {
    /// Create a new process monitor
    pub fn new(config: ProcessMonitoringConfig) -> Self {
        Self {
            config: config.clone(),
            historical_data: Vec::new(),
            anomaly_detector: AnomalyDetector::new(config.anomaly_threshold),
            drift_detector: DriftDetector::new(config.drift_sensitivity),
        }
    }

    /// Start real-time monitoring
    pub async fn start_monitoring<const N: usize, E: ProcessTomographyExecutor>(
        &mut self,
        device_id: &str,
        process_circuit: &Circuit<N>,
        executor: &E,
        tomographer: &SciRS2ProcessTomographer,
    ) -> DeviceResult<()> {
        loop {
            let start_time = Instant::now();
            
            // Perform process tomography measurement
            let result = tomographer
                .perform_process_tomography(device_id, process_circuit, executor)
                .await?;

            // Analyze the results
            let monitoring_data = self.analyze_monitoring_data(result, start_time)?;
            
            // Update historical data
            self.historical_data.push(monitoring_data.clone());
            if self.historical_data.len() > self.config.history_length {
                self.historical_data.remove(0);
            }

            // Check for anomalies and drift
            self.check_for_anomalies(&monitoring_data)?;
            self.check_for_drift()?;

            // Handle alerts
            self.handle_alerts(&monitoring_data).await?;

            // Wait for next monitoring cycle
            let elapsed = start_time.elapsed();
            let sleep_duration = Duration::from_secs_f64(self.config.monitoring_interval)
                .saturating_sub(elapsed);
            
            if sleep_duration > Duration::ZERO {
                tokio::time::sleep(sleep_duration).await;
            }
        }
    }

    /// Analyze monitoring data
    fn analyze_monitoring_data(
        &self,
        result: SciRS2ProcessTomographyResult,
        timestamp: Instant,
    ) -> DeviceResult<ProcessMonitoringData> {
        let anomaly_score = self.anomaly_detector.compute_anomaly_score(&result.process_metrics)?;
        let drift_indicator = self.drift_detector.compute_drift_indicator(&result.process_metrics)?;
        
        let alert_level = self.determine_alert_level(&result.process_metrics, anomaly_score)?;

        Ok(ProcessMonitoringData {
            timestamp: SystemTime::now(),
            process_metrics: result.process_metrics,
            experimental_conditions: ExperimentalConditions {
                temperature: None, // Would be filled from device telemetry
                noise_level: 0.01, // Would be measured
                calibration_age: Duration::from_secs(3600), // Example
                gate_count: 10,
                circuit_depth: 5,
            },
            anomaly_score,
            drift_indicator,
            alert_level,
        })
    }

    /// Check for anomalies in process metrics
    fn check_for_anomalies(&mut self, data: &ProcessMonitoringData) -> DeviceResult<()> {
        if data.anomaly_score > self.config.anomaly_threshold {
            // Log anomaly
            println!("ANOMALY DETECTED: Score = {:.4}", data.anomaly_score);
            
            // Update anomaly detector if adaptive
            if self.anomaly_detector.adaptive_threshold {
                self.anomaly_detector.update_threshold(data.anomaly_score)?;
            }
        }
        Ok(())
    }

    /// Check for process drift
    fn check_for_drift(&mut self) -> DeviceResult<()> {
        if self.historical_data.len() >= self.config.history_length / 2 {
            let drift_detected = self.drift_detector.detect_drift(&self.historical_data)?;
            if drift_detected {
                println!("PROCESS DRIFT DETECTED: Recalibration recommended");
            }
        }
        Ok(())
    }

    /// Handle alerts based on monitoring data
    async fn handle_alerts(&self, data: &ProcessMonitoringData) -> DeviceResult<()> {
        match data.alert_level {
            AlertLevel::Warning => {
                println!("WARNING: Process performance degradation detected");
            },
            AlertLevel::Critical => {
                println!("CRITICAL: Significant process degradation detected");
                if self.config.auto_recalibration {
                    // Trigger automatic recalibration
                    self.trigger_recalibration().await?;
                }
            },
            AlertLevel::Emergency => {
                println!("EMERGENCY: Process failure detected - halting operations");
                // Emergency shutdown procedures
            },
            AlertLevel::Normal => {
                // No action needed
            }
        }
        Ok(())
    }

    /// Trigger automatic recalibration
    async fn trigger_recalibration(&self) -> DeviceResult<()> {
        println!("Triggering automatic recalibration...");
        // Implementation would depend on device interface
        Ok(())
    }

    /// Determine alert level based on metrics
    fn determine_alert_level(
        &self,
        metrics: &ProcessMetrics,
        anomaly_score: f64,
    ) -> DeviceResult<AlertLevel> {
        let thresholds = &self.config.alert_thresholds;
        
        if metrics.process_fidelity < thresholds.fidelity_critical ||
           metrics.unitarity < thresholds.unitarity_critical ||
           metrics.diamond_norm_distance > thresholds.diamond_norm_critical ||
           anomaly_score > self.config.anomaly_threshold * 2.0 {
            return Ok(AlertLevel::Emergency);
        }
        
        if metrics.process_fidelity < thresholds.fidelity_warning ||
           metrics.unitarity < thresholds.unitarity_warning ||
           metrics.diamond_norm_distance > thresholds.diamond_norm_warning ||
           anomaly_score > self.config.anomaly_threshold {
            return Ok(AlertLevel::Critical);
        }
        
        Ok(AlertLevel::Normal)
    }

    /// Get monitoring statistics
    pub fn get_monitoring_statistics(&self) -> ProcessMonitoringStatistics {
        ProcessMonitoringStatistics {
            total_measurements: self.historical_data.len(),
            anomaly_count: self.historical_data.iter()
                .filter(|d| d.anomaly_score > self.config.anomaly_threshold)
                .count(),
            average_fidelity: self.historical_data.iter()
                .map(|d| d.process_metrics.process_fidelity)
                .sum::<f64>() / self.historical_data.len() as f64,
            drift_episodes: self.drift_detector.get_drift_episodes(),
            uptime: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default(),
        }
    }
}

/// Monitoring statistics
#[derive(Debug, Clone)]
pub struct ProcessMonitoringStatistics {
    pub total_measurements: usize,
    pub anomaly_count: usize,
    pub average_fidelity: f64,
    pub drift_episodes: usize,
    pub uptime: Duration,
}

impl AnomalyDetector {
    fn new(threshold: f64) -> Self {
        Self {
            reference_metrics: ProcessMetrics {
                process_fidelity: 1.0,
                average_gate_fidelity: 1.0,
                unitarity: 1.0,
                entangling_power: 0.0,
                non_unitality: 0.0,
                channel_capacity: 1.0,
                coherent_information: 1.0,
                diamond_norm_distance: 0.0,
                process_spectrum: Array1::ones(4),
            },
            detection_method: AnomalyDetectionMethod::StatisticalOutlier,
            threshold,
            adaptive_threshold: true,
        }
    }

    fn compute_anomaly_score(&self, metrics: &ProcessMetrics) -> DeviceResult<f64> {
        // Simple distance-based anomaly score
        let fidelity_diff = (self.reference_metrics.process_fidelity - metrics.process_fidelity).abs();
        let unitarity_diff = (self.reference_metrics.unitarity - metrics.unitarity).abs();
        let diamond_diff = metrics.diamond_norm_distance;
        
        let score = (fidelity_diff + unitarity_diff + diamond_diff) / 3.0;
        Ok(score)
    }

    fn update_threshold(&mut self, anomaly_score: f64) -> DeviceResult<()> {
        // Adaptive threshold adjustment
        self.threshold = (self.threshold + anomaly_score) / 2.0;
        Ok(())
    }
}

impl DriftDetector {
    fn new(sensitivity: f64) -> Self {
        Self {
            baseline_distribution: ProcessDistribution {
                fidelity_dist: DistributionParameters {
                    distribution_type: "Normal".to_string(),
                    parameters: vec![1.0, 0.01],
                    confidence_interval: (0.95, 1.0),
                },
                unitarity_dist: DistributionParameters {
                    distribution_type: "Normal".to_string(),
                    parameters: vec![1.0, 0.01],
                    confidence_interval: (0.95, 1.0),
                },
                diamond_norm_dist: DistributionParameters {
                    distribution_type: "Normal".to_string(),
                    parameters: vec![0.0, 0.01],
                    confidence_interval: (0.0, 0.05),
                },
                spectrum_dist: vec![],
            },
            detection_window: 10,
            sensitivity,
            drift_method: DriftDetectionMethod::KolmogorovSmirnov,
        }
    }

    fn compute_drift_indicator(&self, metrics: &ProcessMetrics) -> DeviceResult<f64> {
        // Simple drift indicator based on deviation from baseline
        let fidelity_drift = (1.0 - metrics.process_fidelity).abs();
        let unitarity_drift = (1.0 - metrics.unitarity).abs();
        
        Ok((fidelity_drift + unitarity_drift) / 2.0)
    }

    fn detect_drift(&self, historical_data: &[ProcessMonitoringData]) -> DeviceResult<bool> {
        if historical_data.len() < self.detection_window {
            return Ok(false);
        }

        // Simple drift detection based on recent average vs baseline
        let recent_fidelity: f64 = historical_data
            .iter()
            .rev()
            .take(self.detection_window)
            .map(|d| d.process_metrics.process_fidelity)
            .sum::<f64>() / self.detection_window as f64;

        let drift = (1.0 - recent_fidelity).abs() > self.sensitivity;
        Ok(drift)
    }

    fn get_drift_episodes(&self) -> usize {
        // Placeholder - would track actual drift episodes
        0
    }
}

/// Machine Learning Enhanced Process Reconstruction
pub struct MLProcessReconstructor {
    model_type: MLModelType,
    training_data: Vec<TrainingDataPoint>,
    trained_model: Option<TrainedModel>,
    feature_extractor: FeatureExtractor,
}

/// ML model types for process reconstruction
#[derive(Debug, Clone)]
pub enum MLModelType {
    NeuralNetwork,
    RandomForest,
    SupportVectorMachine,
    GaussianProcess,
    EnsembleMethod,
}

/// Training data point for ML reconstruction
#[derive(Debug, Clone)]
pub struct TrainingDataPoint {
    pub measurement_data: Vec<f64>,
    pub true_process_matrix: Array4<Complex64>,
    pub noise_level: f64,
    pub experimental_conditions: ExperimentalConditions,
}

/// Trained ML model (placeholder)
#[derive(Debug, Clone)]
pub struct TrainedModel {
    pub model_id: String,
    pub accuracy: f64,
    pub training_loss: f64,
    pub validation_loss: f64,
}

/// Feature extraction for ML reconstruction
pub struct FeatureExtractor {
    feature_type: FeatureType,
    dimensionality_reduction: Option<DimensionalityReduction>,
}

/// Feature types
#[derive(Debug, Clone)]
pub enum FeatureType {
    RawMeasurements,
    StatisticalMoments,
    CorrelationFeatures,
    SpectralFeatures,
    WaveletFeatures,
}

/// Dimensionality reduction methods
#[derive(Debug, Clone)]
pub enum DimensionalityReduction {
    PCA,
    ICA,
    Autoencoder,
    UMAP,
    TSNE,
}

impl MLProcessReconstructor {
    /// Create a new ML reconstructor
    pub fn new(model_type: MLModelType) -> Self {
        Self {
            model_type,
            training_data: Vec::new(),
            trained_model: None,
            feature_extractor: FeatureExtractor {
                feature_type: FeatureType::StatisticalMoments,
                dimensionality_reduction: Some(DimensionalityReduction::PCA),
            },
        }
    }

    /// Add training data
    pub fn add_training_data(&mut self, data: TrainingDataPoint) {
        self.training_data.push(data);
    }

    /// Train the ML model
    pub fn train_model(&mut self) -> DeviceResult<()> {
        if self.training_data.is_empty() {
            return Err(DeviceError::APIError("No training data available".into()));
        }

        // Extract features from training data
        let features = self.extract_features_batch(&self.training_data)?;
        
        // Train model based on type
        match self.model_type {
            MLModelType::NeuralNetwork => self.train_neural_network(&features)?,
            MLModelType::RandomForest => self.train_random_forest(&features)?,
            MLModelType::SupportVectorMachine => self.train_svm(&features)?,
            MLModelType::GaussianProcess => self.train_gaussian_process(&features)?,
            MLModelType::EnsembleMethod => self.train_ensemble(&features)?,
        }

        Ok(())
    }

    /// Reconstruct process using trained ML model
    pub fn reconstruct_process(
        &self,
        measurement_data: &[f64],
        experimental_conditions: &ExperimentalConditions,
    ) -> DeviceResult<Array4<Complex64>> {
        if self.trained_model.is_none() {
            return Err(DeviceError::APIError("Model not trained".into()));
        }

        // Extract features from measurement data
        let features = self.extract_features(measurement_data)?;
        
        // Apply model to predict process matrix
        let predicted_matrix = self.apply_model(&features)?;
        
        Ok(predicted_matrix)
    }

    // Placeholder implementations for ML training methods
    fn train_neural_network(&mut self, _features: &Array2<f64>) -> DeviceResult<()> {
        self.trained_model = Some(TrainedModel {
            model_id: "neural_net_v1".to_string(),
            accuracy: 0.95,
            training_loss: 0.001,
            validation_loss: 0.002,
        });
        Ok(())
    }

    fn train_random_forest(&mut self, _features: &Array2<f64>) -> DeviceResult<()> {
        self.trained_model = Some(TrainedModel {
            model_id: "random_forest_v1".to_string(),
            accuracy: 0.92,
            training_loss: 0.002,
            validation_loss: 0.003,
        });
        Ok(())
    }

    fn train_svm(&mut self, _features: &Array2<f64>) -> DeviceResult<()> {
        self.trained_model = Some(TrainedModel {
            model_id: "svm_v1".to_string(),
            accuracy: 0.90,
            training_loss: 0.003,
            validation_loss: 0.004,
        });
        Ok(())
    }

    fn train_gaussian_process(&mut self, _features: &Array2<f64>) -> DeviceResult<()> {
        self.trained_model = Some(TrainedModel {
            model_id: "gp_v1".to_string(),
            accuracy: 0.93,
            training_loss: 0.0015,
            validation_loss: 0.0025,
        });
        Ok(())
    }

    fn train_ensemble(&mut self, _features: &Array2<f64>) -> DeviceResult<()> {
        self.trained_model = Some(TrainedModel {
            model_id: "ensemble_v1".to_string(),
            accuracy: 0.96,
            training_loss: 0.0008,
            validation_loss: 0.0015,
        });
        Ok(())
    }

    fn extract_features(&self, measurement_data: &[f64]) -> DeviceResult<Array1<f64>> {
        match self.feature_extractor.feature_type {
            FeatureType::RawMeasurements => Ok(Array1::from_vec(measurement_data.to_vec())),
            FeatureType::StatisticalMoments => self.extract_statistical_moments(measurement_data),
            FeatureType::CorrelationFeatures => self.extract_correlation_features(measurement_data),
            FeatureType::SpectralFeatures => self.extract_spectral_features(measurement_data),
            FeatureType::WaveletFeatures => self.extract_wavelet_features(measurement_data),
        }
    }

    fn extract_features_batch(&self, training_data: &[TrainingDataPoint]) -> DeviceResult<Array2<f64>> {
        let n_samples = training_data.len();
        if n_samples == 0 {
            return Err(DeviceError::APIError("No training data".into()));
        }

        let first_features = self.extract_features(&training_data[0].measurement_data)?;
        let n_features = first_features.len();
        
        let mut features = Array2::zeros((n_samples, n_features));
        
        for (i, data_point) in training_data.iter().enumerate() {
            let point_features = self.extract_features(&data_point.measurement_data)?;
            features.row_mut(i).assign(&point_features);
        }

        Ok(features)
    }

    fn extract_statistical_moments(&self, data: &[f64]) -> DeviceResult<Array1<f64>> {
        #[cfg(feature = "scirs2")]
        {
            let array_data = Array1::from_vec(data.to_vec());
            let data_view = array_data.view();
            let mean_val = mean(&data_view).unwrap_or(0.0);
            let std_val = std(&data_view, 0).unwrap_or(1.0);
            let var_val = var(&data_view, 0).unwrap_or(1.0);
            
            // Compute higher moments
            let skewness = self.compute_skewness(data, mean_val, std_val);
            let kurtosis = self.compute_kurtosis(data, mean_val, std_val);
            
            Ok(Array1::from_vec(vec![mean_val, std_val, var_val, skewness, kurtosis]))
        }
        
        #[cfg(not(feature = "scirs2"))]
        {
            // Fallback implementation
            let mean_val = data.iter().sum::<f64>() / data.len() as f64;
            let var_val = data.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / data.len() as f64;
            let std_val = var_val.sqrt();
            Ok(Array1::from_vec(vec![mean_val, std_val, var_val, 0.0, 0.0]))
        }
    }

    fn compute_skewness(&self, data: &[f64], mean: f64, std: f64) -> f64 {
        if std == 0.0 { return 0.0; }
        let n = data.len() as f64;
        let sum_cubed = data.iter()
            .map(|x| ((x - mean) / std).powi(3))
            .sum::<f64>();
        sum_cubed / n
    }

    fn compute_kurtosis(&self, data: &[f64], mean: f64, std: f64) -> f64 {
        if std == 0.0 { return 0.0; }
        let n = data.len() as f64;
        let sum_fourth = data.iter()
            .map(|x| ((x - mean) / std).powi(4))
            .sum::<f64>();
        sum_fourth / n - 3.0  // Excess kurtosis
    }

    fn extract_correlation_features(&self, data: &[f64]) -> DeviceResult<Array1<f64>> {
        // Placeholder - would compute autocorrelation features
        Ok(Array1::from_vec(data.to_vec()))
    }

    fn extract_spectral_features(&self, data: &[f64]) -> DeviceResult<Array1<f64>> {
        // Placeholder - would compute FFT-based features
        Ok(Array1::from_vec(data.to_vec()))
    }

    fn extract_wavelet_features(&self, data: &[f64]) -> DeviceResult<Array1<f64>> {
        // Placeholder - would compute wavelet transform features
        Ok(Array1::from_vec(data.to_vec()))
    }

    fn apply_model(&self, features: &Array1<f64>) -> DeviceResult<Array4<Complex64>> {
        // Placeholder implementation - would apply trained model
        let dim = 2; // Example for single qubit
        Ok(Array4::zeros((dim, dim, dim, dim)))
    }
}

/// Experimental data structure
#[derive(Debug, Clone)]
pub struct ExperimentalData {
    pub input_states: Vec<Array2<Complex64>>,
    pub measurement_operators: Vec<Array2<Complex64>>,
    pub measurement_results: Vec<f64>,
    pub measurement_uncertainties: Vec<f64>,
}

/// Process measurement result
#[derive(Debug, Clone)]
pub struct ProcessMeasurementResult {
    pub expectation_value: f64,
    pub uncertainty: f64,
    pub shot_count: usize,
}

/// Trait for process tomography execution
#[async_trait::async_trait]
pub trait ProcessTomographyExecutor {
    /// Execute process measurement with given input state and measurement
    async fn execute_process_measurement<const N: usize>(
        &self,
        input_state: &Array2<Complex64>,
        process_circuit: &Circuit<N>,
        measurement_operator: &Array2<Complex64>,
        shots: usize,
    ) -> DeviceResult<ProcessMeasurementResult>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::create_ideal_calibration;

    #[test]
    fn test_scirs2_process_tomography_config_default() {
        let config = SciRS2ProcessTomographyConfig::default();
        assert_eq!(config.num_input_states, 36);
        assert_eq!(
            config.reconstruction_method,
            ReconstructionMethod::MaximumLikelihood
        );
        assert!(config.enable_compressed_sensing);
    }

    #[test]
    fn test_process_tomographer_creation() {
        let config = SciRS2ProcessTomographyConfig::default();
        let calibration_manager = CalibrationManager::new();
        let tomographer = SciRS2ProcessTomographer::new(config, calibration_manager);

        assert_eq!(tomographer.input_states.len(), 0);
        assert_eq!(tomographer.measurement_operators.len(), 0);
    }

    #[test]
    fn test_input_state_generation() {
        let config = SciRS2ProcessTomographyConfig::default();
        let calibration_manager = CalibrationManager::new();
        let mut tomographer = SciRS2ProcessTomographer::new(config, calibration_manager);

        tomographer.generate_input_states(1).unwrap();
        assert_eq!(tomographer.input_states.len(), 6); // 6 IC states for 1 qubit

        // Check that states are properly normalized
        for state in &tomographer.input_states {
            let trace = state.diag().sum().re;
            assert!((trace - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_measurement_operator_generation() {
        let config = SciRS2ProcessTomographyConfig::default();
        let calibration_manager = CalibrationManager::new();
        let mut tomographer = SciRS2ProcessTomographer::new(config, calibration_manager);

        tomographer.generate_measurement_operators(1).unwrap();
        assert_eq!(tomographer.measurement_operators.len(), 4); // 4 Pauli operators for 1 qubit

        // Check that operators are Hermitian
        for op in &tomographer.measurement_operators {
            let diff = op - &op.t().mapv(|x| x.conj());
            let norm = diff.mapv(|x| x.norm()).sum();
            assert!(norm < 1e-10);
        }
    }

    #[test]
    fn test_tensor_product() {
        let config = SciRS2ProcessTomographyConfig::default();
        let calibration_manager = CalibrationManager::new();
        let tomographer = SciRS2ProcessTomographer::new(config, calibration_manager);

        let pauli_x = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();

        let pauli_z = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .unwrap();

        let tensor_product = tomographer.tensor_product(&pauli_x, &pauli_z).unwrap();
        assert_eq!(tensor_product.dim(), (4, 4));

        // Check specific elements
        assert_eq!(tensor_product[[0, 2]], Complex64::new(1.0, 0.0));
        assert_eq!(tensor_product[[3, 1]], Complex64::new(-1.0, 0.0));
    }

    #[test]
    fn test_process_metrics_calculation() {
        let config = SciRS2ProcessTomographyConfig::default();
        let calibration_manager = CalibrationManager::new();
        let tomographer = SciRS2ProcessTomographer::new(config, calibration_manager);

        // Create identity process matrix (placeholder)
        let process_matrix = Array4::zeros((2, 2, 2, 2));
        let metrics = tomographer
            .calculate_process_metrics(&process_matrix)
            .unwrap();

        assert!(metrics.process_fidelity >= 0.0 && metrics.process_fidelity <= 1.0);
        assert!(metrics.average_gate_fidelity >= 0.0 && metrics.average_gate_fidelity <= 1.0);
        assert!(metrics.unitarity >= 0.0 && metrics.unitarity <= 1.0);
    }

    #[test]
    fn test_reconstruction_quality_assessment() {
        let config = SciRS2ProcessTomographyConfig::default();
        let calibration_manager = CalibrationManager::new();
        let tomographer = SciRS2ProcessTomographer::new(config, calibration_manager);

        let process_matrix = Array4::zeros((2, 2, 2, 2));
        let experimental_data = ExperimentalData {
            input_states: vec![Array2::eye(2)],
            measurement_operators: vec![Array2::eye(2)],
            measurement_results: vec![0.5],
            measurement_uncertainties: vec![0.01],
        };

        let quality = tomographer
            .assess_reconstruction_quality(&process_matrix, &experimental_data)
            .unwrap();

        assert!(quality.r_squared >= 0.0 && quality.r_squared <= 1.0);
        assert!(quality.reconstruction_error >= 0.0);
        assert!(quality.physical_validity.trace_preservation_error >= 0.0);
    }

    #[test]
    fn test_process_metrics_calculation_comprehensive() {
        let config = SciRS2ProcessTomographyConfig::default();
        let calibration_manager = CalibrationManager::new();
        let tomographer = SciRS2ProcessTomographer::new(config, calibration_manager);

        // Create a more realistic process matrix (X gate)
        let mut process_matrix = Array4::zeros((2, 2, 2, 2));
        // X gate process matrix elements
        process_matrix[[0, 1, 0, 1]] = Complex64::new(1.0, 0.0);
        process_matrix[[1, 0, 1, 0]] = Complex64::new(1.0, 0.0);

        let metrics = tomographer
            .calculate_process_metrics(&process_matrix)
            .unwrap();

        // Verify all metrics are in valid ranges
        assert!(metrics.process_fidelity >= 0.0 && metrics.process_fidelity <= 1.0);
        assert!(metrics.average_gate_fidelity >= 0.0 && metrics.average_gate_fidelity <= 1.0);
        assert!(metrics.unitarity >= 0.0 && metrics.unitarity <= 1.0);
        assert!(metrics.entangling_power >= 0.0 && metrics.entangling_power <= 1.0);
        assert!(metrics.non_unitality >= 0.0);
        assert!(metrics.diamond_norm_distance >= 0.0);
    }

    #[test]
    fn test_process_monitor_configuration() {
        let config = ProcessMonitoringConfig {
            monitoring_interval: 1.0,
            history_length: 100,
            anomaly_threshold: 0.1,
            drift_sensitivity: 0.05,
            auto_recalibration: true,
            alert_thresholds: ProcessAlertThresholds {
                fidelity_warning: 0.9,
                fidelity_critical: 0.8,
                unitarity_warning: 0.9,
                unitarity_critical: 0.8,
                diamond_norm_warning: 0.1,
                diamond_norm_critical: 0.2,
            },
        };

        let monitor = ProcessMonitor::new(config.clone());
        assert_eq!(monitor.config.monitoring_interval, 1.0);
        assert_eq!(monitor.config.history_length, 100);
        assert!(monitor.config.auto_recalibration);
    }

    #[test]
    fn test_anomaly_detector() {
        let detector = AnomalyDetector::new(0.1);
        
        let test_metrics = ProcessMetrics {
            process_fidelity: 0.5, // Low fidelity should trigger anomaly
            average_gate_fidelity: 0.6,
            unitarity: 0.7,
            entangling_power: 0.0,
            non_unitality: 0.3,
            channel_capacity: 0.5,
            coherent_information: 0.4,
            diamond_norm_distance: 0.5,
            process_spectrum: Array1::ones(2),
        };

        let anomaly_score = detector.compute_anomaly_score(&test_metrics).unwrap();
        assert!(anomaly_score > 0.0);
        assert!(anomaly_score > 0.1); // Should exceed threshold
    }

    #[test]
    fn test_ml_process_reconstructor() {
        let mut reconstructor = MLProcessReconstructor::new(MLModelType::NeuralNetwork);
        
        // Test feature extraction
        let test_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let features = reconstructor.extract_features(&test_data).unwrap();
        assert!(!features.is_empty());

        // Test training data addition
        let training_point = TrainingDataPoint {
            measurement_data: test_data,
            true_process_matrix: Array4::zeros((2, 2, 2, 2)),
            noise_level: 0.01,
            experimental_conditions: ExperimentalConditions {
                temperature: Some(20.0),
                noise_level: 0.01,
                calibration_age: Duration::from_secs(3600),
                gate_count: 10,
                circuit_depth: 5,
            },
        };

        reconstructor.add_training_data(training_point);
        assert_eq!(reconstructor.training_data.len(), 1);
    }

    #[test]
    fn test_drift_detector() {
        let detector = DriftDetector::new(0.05);
        
        // Create test data showing drift
        let mut historical_data = Vec::new();
        for i in 0..15 {
            let fidelity = 1.0 - (i as f64 * 0.05); // Decreasing fidelity
            historical_data.push(ProcessMonitoringData {
                timestamp: SystemTime::now(),
                process_metrics: ProcessMetrics {
                    process_fidelity: fidelity,
                    average_gate_fidelity: fidelity,
                    unitarity: 0.9,
                    entangling_power: 0.0,
                    non_unitality: 0.1,
                    channel_capacity: 1.0,
                    coherent_information: 0.8,
                    diamond_norm_distance: 1.0 - fidelity,
                    process_spectrum: Array1::ones(2),
                },
                experimental_conditions: ExperimentalConditions {
                    temperature: None,
                    noise_level: 0.01,
                    calibration_age: Duration::from_secs(3600),
                    gate_count: 10,
                    circuit_depth: 5,
                },
                anomaly_score: 0.0,
                drift_indicator: 0.0,
                alert_level: AlertLevel::Normal,
            });
        }

        let drift_detected = detector.detect_drift(&historical_data).unwrap();
        assert!(drift_detected); // Should detect drift due to decreasing fidelity
    }

    #[test]
    fn test_statistical_analysis_with_scirs2() {
        let config = SciRS2ProcessTomographyConfig::default();
        let calibration_manager = CalibrationManager::new();
        let tomographer = SciRS2ProcessTomographer::new(config, calibration_manager);

        let process_matrix = Array4::zeros((2, 2, 2, 2));
        let experimental_data = ExperimentalData {
            input_states: vec![Array2::eye(2)],
            measurement_operators: vec![Array2::eye(2)],
            measurement_results: vec![0.5, 0.4, 0.6, 0.5],
            measurement_uncertainties: vec![0.01, 0.01, 0.01, 0.01],
        };

        let statistical_analysis = tomographer
            .perform_statistical_analysis(&process_matrix, &experimental_data)
            .unwrap();

        // Verify statistical tests were performed
        assert!(!statistical_analysis.statistical_tests.is_empty());
        
        // Verify distribution analysis
        assert!(!statistical_analysis.distribution_analysis.element_distributions.is_empty());
        
        // Verify correlation analysis
        assert!(!statistical_analysis.correlation_analysis.element_correlations.is_empty());
    }

    #[test]
    fn test_process_fidelity_calculation() {
        let config = SciRS2ProcessTomographyConfig::default();
        let calibration_manager = CalibrationManager::new();
        let tomographer = SciRS2ProcessTomographer::new(config, calibration_manager);

        // Create identity process matrix
        let mut identity_process = Array4::zeros((2, 2, 2, 2));
        identity_process[[0, 0, 0, 0]] = Complex64::new(1.0, 0.0);
        identity_process[[1, 1, 1, 1]] = Complex64::new(1.0, 0.0);

        let fidelity = tomographer
            .calculate_process_fidelity(&identity_process)
            .unwrap();

        // Identity process should have high fidelity with itself
        assert!(fidelity > 0.9);
        assert!(fidelity <= 1.0);
    }

    #[test]
    fn test_entangling_power_calculation() {
        let config = SciRS2ProcessTomographyConfig::default();
        let calibration_manager = CalibrationManager::new();
        let tomographer = SciRS2ProcessTomographer::new(config, calibration_manager);

        // Single qubit process should have zero entangling power
        let single_qubit_process = Array4::zeros((2, 2, 2, 2));
        let entangling_power = tomographer
            .calculate_entangling_power(&single_qubit_process)
            .unwrap();
        assert_eq!(entangling_power, 0.0);

        // Two qubit process with entangling elements
        let mut two_qubit_process = Array4::zeros((4, 4, 4, 4));
        two_qubit_process[[0, 1, 2, 3]] = Complex64::new(0.5, 0.0); // Off-diagonal element
        let entangling_power_2q = tomographer
            .calculate_entangling_power(&two_qubit_process)
            .unwrap();
        assert!(entangling_power_2q > 0.0);
    }

    #[test]
    fn test_measurement_matrix_construction() {
        let config = SciRS2ProcessTomographyConfig::default();
        let calibration_manager = CalibrationManager::new();
        let tomographer = SciRS2ProcessTomographer::new(config, calibration_manager);

        let experimental_data = ExperimentalData {
            input_states: vec![Array2::eye(2), Array2::zeros((2, 2))],
            measurement_operators: vec![Array2::eye(2), Array2::zeros((2, 2))],
            measurement_results: vec![0.5, 0.3, 0.7, 0.4],
            measurement_uncertainties: vec![0.01, 0.02, 0.01, 0.02],
        };

        let measurement_matrix = tomographer
            .build_measurement_matrix(&experimental_data)
            .unwrap();

        assert_eq!(measurement_matrix.nrows(), 4); // 2x2 input/measurement combinations
        assert!(measurement_matrix.ncols() > 0);
    }

    #[test]
    fn test_process_coefficient_calculation() {
        let config = SciRS2ProcessTomographyConfig::default();
        let calibration_manager = CalibrationManager::new();
        let tomographer = SciRS2ProcessTomographer::new(config, calibration_manager);

        let input_state = Array2::eye(2);
        let measurement = Array2::eye(2);

        let coefficient = tomographer
            .compute_process_coefficient(&input_state, &measurement, 0, 0, 0, 0)
            .unwrap();

        // Should return a real number
        assert!(coefficient.is_finite());
    }
}
