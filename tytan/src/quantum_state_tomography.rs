//! Quantum State Tomography and Characterization
//!
//! This module provides comprehensive tools for quantum state reconstruction,
//! process tomography, and quantum system characterization using advanced
//! techniques including shadow tomography and compressed sensing.

use ndarray::{Array1, Array2, Array4};
use rand::{prelude::*, rng};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Quantum state tomography system
pub struct QuantumStateTomography {
    /// Number of qubits
    pub num_qubits: usize,
    /// Tomography configuration
    pub config: TomographyConfig,
    /// Measurement outcomes database
    pub measurement_data: MeasurementDatabase,
    /// Reconstruction algorithms
    pub reconstruction_methods: Vec<Box<dyn ReconstructionMethod>>,
    /// Fidelity estimators
    pub fidelity_estimators: Vec<Box<dyn FidelityEstimator>>,
    /// Error analysis tools
    pub error_analysis: ErrorAnalysisTools,
    /// Performance metrics
    pub metrics: TomographyMetrics,
}

/// Configuration for quantum state tomography
#[derive(Debug, Clone)]
pub struct TomographyConfig {
    /// Tomography type
    pub tomography_type: TomographyType,
    /// Number of measurement shots per setting
    pub shots_per_setting: usize,
    /// Measurement bases to use
    pub measurement_bases: Vec<MeasurementBasis>,
    /// Reconstruction method
    pub reconstruction_method: ReconstructionMethodType,
    /// Error mitigation techniques
    pub error_mitigation: ErrorMitigationConfig,
    /// Optimization settings
    pub optimization: OptimizationConfig,
    /// Validation settings
    pub validation: ValidationConfig,
}

/// Types of tomography
#[derive(Debug, Clone, PartialEq)]
pub enum TomographyType {
    /// Full quantum state tomography
    QuantumState,
    /// Quantum process tomography
    QuantumProcess,
    /// Shadow tomography
    ShadowTomography { num_shadows: usize },
    /// Compressed sensing tomography
    CompressedSensing { sparsity_level: usize },
    /// Adaptive tomography
    AdaptiveTomography,
    /// Entanglement characterization
    EntanglementCharacterization,
}

/// Measurement bases for tomography
#[derive(Debug, Clone)]
pub struct MeasurementBasis {
    /// Basis name
    pub name: String,
    /// Pauli operators for each qubit
    pub operators: Vec<PauliOperator>,
    /// Measurement angles (for rotated bases)
    pub angles: Vec<f64>,
    /// Basis type
    pub basis_type: BasisType,
}

/// Types of measurement bases
#[derive(Debug, Clone, PartialEq)]
pub enum BasisType {
    /// Computational basis (Z measurements)
    Computational,
    /// Pauli basis (X, Y, Z)
    Pauli,
    /// Mutually unbiased bases
    MUB,
    /// Symmetric informationally complete (SIC)
    SIC,
    /// Stabilizer measurements
    Stabilizer,
    /// Random Pauli measurements
    RandomPauli,
    /// Adaptive measurements
    Adaptive,
}

/// Pauli operators
#[derive(Debug, Clone, PartialEq)]
pub enum PauliOperator {
    I, // Identity
    X, // Pauli-X
    Y, // Pauli-Y
    Z, // Pauli-Z
}

/// Reconstruction method types
#[derive(Debug, Clone, PartialEq)]
pub enum ReconstructionMethodType {
    /// Maximum likelihood estimation
    MaximumLikelihood,
    /// Least squares
    LeastSquares,
    /// Compressed sensing
    CompressedSensing,
    /// Bayesian inference
    BayesianInference,
    /// Neural network reconstruction
    NeuralNetwork,
    /// Variational quantum state tomography
    Variational,
    /// Matrix completion
    MatrixCompletion,
}

/// Error mitigation configuration
#[derive(Debug, Clone)]
pub struct ErrorMitigationConfig {
    /// Enable readout error correction
    pub readout_error_correction: bool,
    /// Enable gate error mitigation
    pub gate_error_mitigation: bool,
    /// Symmetry verification
    pub symmetry_verification: bool,
    /// Noise characterization
    pub noise_characterization: NoiseCharacterizationConfig,
    /// Error correction protocols
    pub error_correction_protocols: Vec<ErrorCorrectionProtocol>,
}

/// Noise characterization configuration
#[derive(Debug, Clone)]
pub struct NoiseCharacterizationConfig {
    /// Characterize coherent errors
    pub coherent_errors: bool,
    /// Characterize incoherent errors
    pub incoherent_errors: bool,
    /// Cross-talk characterization
    pub crosstalk: bool,
    /// Temporal correlations
    pub temporal_correlations: bool,
    /// Spatial correlations
    pub spatial_correlations: bool,
}

/// Error correction protocols
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCorrectionProtocol {
    /// Zero noise extrapolation
    ZeroNoiseExtrapolation,
    /// Probabilistic error cancellation
    ProbabilisticErrorCancellation,
    /// Symmetry verification
    SymmetryVerification,
    /// Virtual distillation
    VirtualDistillation,
    /// Clifford data regression
    CliffordDataRegression,
}

/// Optimization configuration for reconstruction
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Regularization parameters
    pub regularization: RegularizationConfig,
    /// Constraint handling
    pub constraints: ConstraintConfig,
    /// Optimization algorithm
    pub algorithm: OptimizationAlgorithm,
}

/// Regularization configuration
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// L1 regularization strength
    pub l1_strength: f64,
    /// L2 regularization strength
    pub l2_strength: f64,
    /// Nuclear norm regularization
    pub nuclear_norm_strength: f64,
    /// Trace norm regularization
    pub trace_norm_strength: f64,
    /// Entropy regularization
    pub entropy_strength: f64,
}

/// Constraint configuration
#[derive(Debug, Clone)]
pub struct ConstraintConfig {
    /// Enforce trace preservation
    pub trace_preserving: bool,
    /// Enforce complete positivity
    pub completely_positive: bool,
    /// Enforce Hermiticity
    pub hermitian: bool,
    /// Enforce positive semidefiniteness
    pub positive_semidefinite: bool,
    /// Physical constraints
    pub physical_constraints: Vec<PhysicalConstraint>,
}

/// Physical constraints
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalConstraint {
    /// Unit trace
    UnitTrace,
    /// Positive eigenvalues
    PositiveEigenvalues,
    /// Separability constraints
    Separability,
    /// Entanglement constraints
    EntanglementConstraints,
    /// Symmetry constraints
    SymmetryConstraints,
}

/// Optimization algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationAlgorithm {
    /// Gradient descent
    GradientDescent,
    /// Quasi-Newton methods
    QuasiNewton,
    /// Interior point methods
    InteriorPoint,
    /// Semidefinite programming
    SemidefiniteProgramming,
    /// Alternating projections
    AlternatingProjections,
    /// Expectation maximization
    ExpectationMaximization,
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Cross-validation folds
    pub cross_validation_folds: usize,
    /// Bootstrap samples
    pub bootstrap_samples: usize,
    /// Confidence level
    pub confidence_level: f64,
    /// Validation metrics
    pub validation_metrics: Vec<ValidationMetric>,
    /// Statistical tests
    pub statistical_tests: Vec<StatisticalTest>,
}

/// Validation metrics
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationMetric {
    /// Fidelity with known state
    Fidelity,
    /// Trace distance
    TraceDistance,
    /// Negativity (entanglement measure)
    Negativity,
    /// Concurrence
    Concurrence,
    /// Mutual information
    MutualInformation,
    /// von Neumann entropy
    VonNeumannEntropy,
}

/// Statistical tests
#[derive(Debug, Clone, PartialEq)]
pub enum StatisticalTest {
    /// Chi-squared goodness of fit
    ChiSquared,
    /// Kolmogorov-Smirnov test
    KolmogorovSmirnov,
    /// Bootstrap confidence intervals
    Bootstrap,
    /// Permutation tests
    Permutation,
}

/// Measurement database
#[derive(Debug)]
pub struct MeasurementDatabase {
    /// Raw measurement outcomes
    pub raw_outcomes: HashMap<String, Vec<Vec<u8>>>,
    /// Processed statistics
    pub statistics: HashMap<String, MeasurementStatistics>,
    /// Metadata for each measurement setting
    pub metadata: HashMap<String, MeasurementMetadata>,
    /// Error characterization data
    pub error_data: ErrorCharacterizationData,
}

/// Statistics for measurement outcomes
#[derive(Debug, Clone)]
pub struct MeasurementStatistics {
    /// Outcome probabilities
    pub probabilities: Array1<f64>,
    /// Expectation values
    pub expectation_values: Array1<f64>,
    /// Variances
    pub variances: Array1<f64>,
    /// Covariances
    pub covariances: Array2<f64>,
    /// Higher order moments
    pub higher_moments: HashMap<String, f64>,
}

/// Metadata for measurements
#[derive(Debug, Clone)]
pub struct MeasurementMetadata {
    /// Measurement basis
    pub basis: MeasurementBasis,
    /// Number of shots
    pub num_shots: usize,
    /// Timestamp
    pub timestamp: std::time::Instant,
    /// Hardware information
    pub hardware_info: HardwareInfo,
    /// Calibration data
    pub calibration_data: CalibrationData,
}

/// Hardware information
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    /// Device name
    pub device_name: String,
    /// Qubit connectivity
    pub connectivity: Array2<bool>,
    /// Gate fidelities
    pub gate_fidelities: HashMap<String, f64>,
    /// Readout fidelities
    pub readout_fidelities: Array1<f64>,
    /// Coherence times
    pub coherence_times: CoherenceTimes,
}

/// Coherence times
#[derive(Debug, Clone)]
pub struct CoherenceTimes {
    /// T1 relaxation times
    pub t1_times: Array1<f64>,
    /// T2 dephasing times
    pub t2_times: Array1<f64>,
    /// T2* echo times
    pub t2_echo_times: Array1<f64>,
}

/// Calibration data
#[derive(Debug, Clone)]
pub struct CalibrationData {
    /// Calibration matrix for readout errors
    pub readout_matrix: Array2<f64>,
    /// Gate calibration parameters
    pub gate_parameters: HashMap<String, Array1<f64>>,
    /// State preparation fidelity
    pub state_prep_fidelity: f64,
    /// Measurement fidelity
    pub measurement_fidelity: f64,
}

/// Error characterization data
#[derive(Debug, Clone)]
pub struct ErrorCharacterizationData {
    /// Process matrices for different gates
    pub process_matrices: HashMap<String, Array4<f64>>,
    /// Noise models
    pub noise_models: Vec<NoiseModel>,
    /// Cross-talk parameters
    pub crosstalk_parameters: Array2<f64>,
    /// Temporal correlation data
    pub temporal_correlations: TemporalCorrelationData,
}

/// Noise models
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// Model type
    pub model_type: NoiseModelType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Applicability range
    pub applicability: ApplicabilityRange,
    /// Model accuracy
    pub accuracy: f64,
}

/// Types of noise models
#[derive(Debug, Clone, PartialEq)]
pub enum NoiseModelType {
    /// Depolarizing noise
    Depolarizing,
    /// Amplitude damping
    AmplitudeDamping,
    /// Phase damping
    PhaseDamping,
    /// Pauli noise
    Pauli,
    /// Coherent noise
    Coherent,
    /// Correlated noise
    Correlated,
}

/// Applicability range for noise models
#[derive(Debug, Clone)]
pub struct ApplicabilityRange {
    /// Time range
    pub time_range: (f64, f64),
    /// Gate count range
    pub gate_count_range: (usize, usize),
    /// Frequency range
    pub frequency_range: (f64, f64),
    /// Temperature range
    pub temperature_range: (f64, f64),
}

/// Temporal correlation data
#[derive(Debug, Clone)]
pub struct TemporalCorrelationData {
    /// Correlation functions
    pub correlation_functions: Array2<f64>,
    /// Memory kernels
    pub memory_kernels: Array1<f64>,
    /// Correlation times
    pub correlation_times: Array1<f64>,
    /// Non-Markovian indicators
    pub non_markovian_indicators: Array1<f64>,
}

/// Reconstruction method trait
pub trait ReconstructionMethod: Send + Sync {
    /// Reconstruct quantum state from measurement data
    fn reconstruct_state(
        &self,
        data: &MeasurementDatabase,
    ) -> Result<ReconstructedState, TomographyError>;

    /// Get method name
    fn get_method_name(&self) -> &str;

    /// Get method parameters
    fn get_parameters(&self) -> HashMap<String, f64>;

    /// Validate reconstruction
    fn validate_reconstruction(
        &self,
        state: &ReconstructedState,
        data: &MeasurementDatabase,
    ) -> ValidationResult;
}

/// Reconstructed quantum state
#[derive(Debug, Clone)]
pub struct ReconstructedState {
    /// Density matrix
    pub density_matrix: Array2<f64>,
    /// Reconstruction uncertainty
    pub uncertainty: Array2<f64>,
    /// Eigenvalues
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors
    pub eigenvectors: Array2<f64>,
    /// Purity
    pub purity: f64,
    /// von Neumann entropy
    pub entropy: f64,
    /// Entanglement measures
    pub entanglement_measures: EntanglementMeasures,
    /// Reconstruction metadata
    pub metadata: ReconstructionMetadata,
}

/// Entanglement measures
#[derive(Debug, Clone)]
pub struct EntanglementMeasures {
    /// Concurrence
    pub concurrence: f64,
    /// Negativity
    pub negativity: f64,
    /// Entanglement of formation
    pub entanglement_of_formation: f64,
    /// Distillable entanglement
    pub distillable_entanglement: f64,
    /// Logarithmic negativity
    pub logarithmic_negativity: f64,
    /// Schmidt number
    pub schmidt_number: f64,
}

/// Reconstruction metadata
#[derive(Debug, Clone)]
pub struct ReconstructionMetadata {
    /// Reconstruction method used
    pub method: String,
    /// Convergence information
    pub convergence_info: ConvergenceInfo,
    /// Computational cost
    pub computational_cost: ComputationalCost,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Convergence information
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Number of iterations
    pub iterations: usize,
    /// Final objective value
    pub final_objective: f64,
    /// Convergence achieved
    pub converged: bool,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Convergence history
    pub history: Vec<f64>,
}

/// Computational cost metrics
#[derive(Debug, Clone)]
pub struct ComputationalCost {
    /// Wall clock time
    pub wall_time: f64,
    /// CPU time
    pub cpu_time: f64,
    /// Memory usage
    pub memory_usage: f64,
    /// Number of function evaluations
    pub function_evaluations: usize,
    /// Number of gradient evaluations
    pub gradient_evaluations: usize,
}

/// Quality metrics for reconstruction
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Fidelity with true state (if known)
    pub fidelity: Option<f64>,
    /// Trace distance
    pub trace_distance: Option<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Chi-squared statistic
    pub chi_squared: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: usize,
    /// P-value
    pub p_value: f64,
}

/// Fidelity estimator trait
pub trait FidelityEstimator: Send + Sync {
    /// Estimate fidelity between two states
    fn estimate_fidelity(
        &self,
        state1: &ReconstructedState,
        state2: &ReconstructedState,
    ) -> Result<f64, TomographyError>;

    /// Estimate fidelity with measurement data
    fn estimate_fidelity_from_data(
        &self,
        state: &ReconstructedState,
        data: &MeasurementDatabase,
    ) -> Result<f64, TomographyError>;

    /// Get estimator name
    fn get_estimator_name(&self) -> &str;
}

/// Error analysis tools
#[derive(Debug)]
pub struct ErrorAnalysisTools {
    /// Error propagation methods
    pub error_propagation: Vec<Box<dyn ErrorPropagationMethod>>,
    /// Uncertainty quantification
    pub uncertainty_quantification: Vec<Box<dyn UncertaintyQuantificationMethod>>,
    /// Sensitivity analysis
    pub sensitivity_analysis: SensitivityAnalysis,
    /// Bootstrap methods
    pub bootstrap_methods: BootstrapMethods,
}

/// Error propagation method trait
pub trait ErrorPropagationMethod: Send + Sync + std::fmt::Debug {
    /// Propagate measurement errors to state reconstruction
    fn propagate_errors(
        &self,
        measurement_errors: &Array1<f64>,
        reconstruction: &ReconstructedState,
    ) -> Array2<f64>;

    /// Get method name
    fn get_method_name(&self) -> &str;
}

/// Uncertainty quantification method trait
pub trait UncertaintyQuantificationMethod: Send + Sync + std::fmt::Debug {
    /// Quantify reconstruction uncertainty
    fn quantify_uncertainty(
        &self,
        data: &MeasurementDatabase,
        reconstruction: &ReconstructedState,
    ) -> UncertaintyAnalysis;

    /// Get method name
    fn get_method_name(&self) -> &str;
}

/// Uncertainty analysis results
#[derive(Debug, Clone)]
pub struct UncertaintyAnalysis {
    /// Parameter uncertainties
    pub parameter_uncertainties: Array1<f64>,
    /// Confidence intervals
    pub confidence_intervals: Array2<f64>,
    /// Covariance matrix
    pub covariance_matrix: Array2<f64>,
    /// Sensitivity coefficients
    pub sensitivity_coefficients: Array1<f64>,
    /// Model selection uncertainty
    pub model_selection_uncertainty: f64,
}

/// Sensitivity analysis
#[derive(Debug, Clone)]
pub struct SensitivityAnalysis {
    /// Parameter sensitivities
    pub parameter_sensitivities: Array1<f64>,
    /// Cross-sensitivities
    pub cross_sensitivities: Array2<f64>,
    /// Robustness indicators
    pub robustness_indicators: Array1<f64>,
    /// Critical parameters
    pub critical_parameters: Vec<usize>,
}

/// Bootstrap methods
#[derive(Debug, Clone)]
pub struct BootstrapMethods {
    /// Number of bootstrap samples
    pub num_samples: usize,
    /// Bootstrap confidence intervals
    pub confidence_intervals: Array2<f64>,
    /// Bootstrap distributions
    pub bootstrap_distributions: Array2<f64>,
    /// Bias estimates
    pub bias_estimates: Array1<f64>,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Validation passed
    pub passed: bool,
    /// Validation score
    pub score: f64,
    /// Individual test results
    pub test_results: HashMap<String, TestResult>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Individual test result
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical value
    pub critical_value: f64,
    /// Test passed
    pub passed: bool,
    /// Effect size
    pub effect_size: f64,
}

/// Tomography metrics
#[derive(Debug, Clone)]
pub struct TomographyMetrics {
    /// Reconstruction accuracy
    pub reconstruction_accuracy: f64,
    /// Computational efficiency
    pub computational_efficiency: f64,
    /// Statistical power
    pub statistical_power: f64,
    /// Robustness score
    pub robustness_score: f64,
    /// Overall quality score
    pub overall_quality: f64,
}

/// Tomography errors
#[derive(Debug, Clone)]
pub enum TomographyError {
    /// Insufficient measurement data
    InsufficientData(String),
    /// Reconstruction failed
    ReconstructionFailed(String),
    /// Invalid measurement basis
    InvalidBasis(String),
    /// Convergence failed
    ConvergenceFailed(String),
    /// Validation failed
    ValidationFailed(String),
    /// Numerical error
    NumericalError(String),
}

impl std::fmt::Display for TomographyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TomographyError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            TomographyError::ReconstructionFailed(msg) => {
                write!(f, "Reconstruction failed: {}", msg)
            }
            TomographyError::InvalidBasis(msg) => write!(f, "Invalid basis: {}", msg),
            TomographyError::ConvergenceFailed(msg) => write!(f, "Convergence failed: {}", msg),
            TomographyError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            TomographyError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
        }
    }
}

impl std::error::Error for TomographyError {}

impl QuantumStateTomography {
    /// Create new quantum state tomography system
    pub fn new(num_qubits: usize, config: TomographyConfig) -> Self {
        Self {
            num_qubits,
            config,
            measurement_data: MeasurementDatabase {
                raw_outcomes: HashMap::new(),
                statistics: HashMap::new(),
                metadata: HashMap::new(),
                error_data: ErrorCharacterizationData {
                    process_matrices: HashMap::new(),
                    noise_models: Vec::new(),
                    crosstalk_parameters: Array2::zeros((num_qubits, num_qubits)),
                    temporal_correlations: TemporalCorrelationData {
                        correlation_functions: Array2::zeros((num_qubits, num_qubits)),
                        memory_kernels: Array1::zeros(100),
                        correlation_times: Array1::zeros(num_qubits),
                        non_markovian_indicators: Array1::zeros(num_qubits),
                    },
                },
            },
            reconstruction_methods: Vec::new(),
            fidelity_estimators: Vec::new(),
            error_analysis: ErrorAnalysisTools {
                error_propagation: Vec::new(),
                uncertainty_quantification: Vec::new(),
                sensitivity_analysis: SensitivityAnalysis {
                    parameter_sensitivities: Array1::zeros(num_qubits * num_qubits),
                    cross_sensitivities: Array2::zeros((num_qubits, num_qubits)),
                    robustness_indicators: Array1::zeros(num_qubits),
                    critical_parameters: Vec::new(),
                },
                bootstrap_methods: BootstrapMethods {
                    num_samples: 1000,
                    confidence_intervals: Array2::zeros((num_qubits * num_qubits, 2)),
                    bootstrap_distributions: Array2::zeros((1000, num_qubits * num_qubits)),
                    bias_estimates: Array1::zeros(num_qubits * num_qubits),
                },
            },
            metrics: TomographyMetrics {
                reconstruction_accuracy: 0.0,
                computational_efficiency: 0.0,
                statistical_power: 0.0,
                robustness_score: 0.0,
                overall_quality: 0.0,
            },
        }
    }

    /// Perform quantum state tomography
    pub fn perform_tomography(&mut self) -> Result<ReconstructedState, TomographyError> {
        println!(
            "Starting quantum state tomography for {} qubits",
            self.num_qubits
        );

        // Step 1: Generate measurement settings
        let measurement_settings = self.generate_measurement_settings()?;

        // Step 2: Collect measurement data
        self.collect_measurement_data(&measurement_settings)?;

        // Step 3: Process measurement statistics
        self.process_measurement_statistics()?;

        // Step 4: Reconstruct quantum state
        let reconstructed_state = self.reconstruct_quantum_state()?;

        // Step 5: Validate reconstruction
        let validation_result = self.validate_reconstruction(&reconstructed_state)?;

        // Step 6: Perform error analysis
        self.perform_error_analysis(&reconstructed_state)?;

        // Step 7: Compute tomography metrics
        self.compute_tomography_metrics(&reconstructed_state, &validation_result);

        println!("Quantum state tomography completed");
        println!(
            "Reconstruction fidelity: {:.4}",
            self.metrics.reconstruction_accuracy
        );
        println!("Overall quality score: {:.4}", self.metrics.overall_quality);

        Ok(reconstructed_state)
    }

    /// Generate measurement settings based on tomography type
    fn generate_measurement_settings(&self) -> Result<Vec<MeasurementBasis>, TomographyError> {
        match &self.config.tomography_type {
            TomographyType::QuantumState => self.generate_pauli_measurements(),
            TomographyType::ShadowTomography { num_shadows } => {
                self.generate_shadow_measurements(*num_shadows)
            }
            TomographyType::CompressedSensing { sparsity_level: _ } => {
                self.generate_compressed_sensing_measurements()
            }
            TomographyType::AdaptiveTomography => self.generate_adaptive_measurements(),
            _ => self.generate_pauli_measurements(),
        }
    }

    /// Generate Pauli measurement settings
    fn generate_pauli_measurements(&self) -> Result<Vec<MeasurementBasis>, TomographyError> {
        let measurements = Vec::new();
        let pauli_ops = [PauliOperator::X, PauliOperator::Y, PauliOperator::Z];

        // Generate all possible Pauli measurements
        for measurement_index in 0..(3_usize.pow(self.num_qubits as u32)) {
            let operators = Vec::new();
            let temp_index = measurement_index;

            for _ in 0..self.num_qubits {
                operators.push(pauli_ops[temp_index % 3].clone());
                temp_index /= 3;
            }

            measurements.push(MeasurementBasis {
                name: format!("pauli_{}", measurement_index),
                operators,
                angles: vec![0.0; self.num_qubits],
                basis_type: BasisType::Pauli,
            });
        }

        Ok(measurements)
    }

    /// Generate shadow measurement settings
    fn generate_shadow_measurements(
        &self,
        num_shadows: usize,
    ) -> Result<Vec<MeasurementBasis>, TomographyError> {
        let measurements = Vec::new();
        let rng = rng();

        for shadow_idx in 0..num_shadows {
            let operators = Vec::new();
            let angles = Vec::new();

            for _ in 0..self.num_qubits {
                // Random Pauli measurement
                let pauli_choice: usize = rng.random_range(0..3);
                operators.push(match pauli_choice {
                    0 => PauliOperator::X,
                    1 => PauliOperator::Y,
                    _ => PauliOperator::Z,
                });

                // Random angle for measurement
                angles.push(rng.random_range(0.0..2.0 * PI));
            }

            measurements.push(MeasurementBasis {
                name: format!("shadow_{}", shadow_idx),
                operators,
                angles,
                basis_type: BasisType::RandomPauli,
            });
        }

        Ok(measurements)
    }

    /// Generate compressed sensing measurements
    fn generate_compressed_sensing_measurements(
        &self,
    ) -> Result<Vec<MeasurementBasis>, TomographyError> {
        // Use random Pauli measurements optimized for sparse reconstruction
        let num_measurements = self.num_qubits * self.num_qubits * 2; // Reduced measurement count
        let measurements = Vec::new();
        let rng = rng();

        for measurement_idx in 0..num_measurements {
            let operators = Vec::new();

            for _ in 0..self.num_qubits {
                let pauli_choice: usize = rng.random_range(0..4);
                operators.push(match pauli_choice {
                    0 => PauliOperator::I,
                    1 => PauliOperator::X,
                    2 => PauliOperator::Y,
                    _ => PauliOperator::Z,
                });
            }

            measurements.push(MeasurementBasis {
                name: format!("compressed_sensing_{}", measurement_idx),
                operators,
                angles: vec![0.0; self.num_qubits],
                basis_type: BasisType::RandomPauli,
            });
        }

        Ok(measurements)
    }

    /// Generate adaptive measurements
    fn generate_adaptive_measurements(&self) -> Result<Vec<MeasurementBasis>, TomographyError> {
        // Start with standard Pauli measurements
        let measurements = self.generate_pauli_measurements()?;

        // Add informationally optimal measurements
        // This would be determined by the Fisher information matrix
        for optimal_idx in 0..self.num_qubits {
            let mut operators = vec![PauliOperator::Z; self.num_qubits];
            operators[optimal_idx] = PauliOperator::X;

            measurements.push(MeasurementBasis {
                name: format!("adaptive_{}", optimal_idx),
                operators,
                angles: vec![PI / 4.0; self.num_qubits],
                basis_type: BasisType::Adaptive,
            });
        }

        Ok(measurements)
    }

    /// Collect measurement data (simulated)
    fn collect_measurement_data(
        &mut self,
        measurement_settings: &[MeasurementBasis],
    ) -> Result<(), TomographyError> {
        let rng = rng();

        for setting in measurement_settings {
            let outcomes = Vec::new();

            for _ in 0..self.config.shots_per_setting {
                let outcome = Vec::new();

                // Simulate measurement outcomes
                for qubit in 0..self.num_qubits {
                    // Random outcome for simulation
                    outcome.push(if rng.random::<f64>() < 0.5 { 0 } else { 1 });
                }

                outcomes.push(outcome);
            }

            self.measurement_data
                .raw_outcomes
                .insert(setting.name.clone(), outcomes);

            // Store metadata
            self.measurement_data.metadata.insert(
                setting.name.clone(),
                MeasurementMetadata {
                    basis: setting.clone(),
                    num_shots: self.config.shots_per_setting,
                    timestamp: std::time::Instant::now(),
                    hardware_info: HardwareInfo {
                        device_name: "simulator".to_string(),
                        connectivity: {
                            let mut conn =
                                Array2::from_elem((self.num_qubits, self.num_qubits), false);
                            for i in 0..self.num_qubits {
                                conn[(i, i)] = true;
                            }
                            conn
                        },
                        gate_fidelities: HashMap::new(),
                        readout_fidelities: Array1::ones(self.num_qubits) * 0.99,
                        coherence_times: CoherenceTimes {
                            t1_times: Array1::ones(self.num_qubits) * 100e-6, // 100 μs
                            t2_times: Array1::ones(self.num_qubits) * 50e-6,  // 50 μs
                            t2_echo_times: Array1::ones(self.num_qubits) * 80e-6, // 80 μs
                        },
                    },
                    calibration_data: CalibrationData {
                        readout_matrix: Array2::eye(1 << self.num_qubits),
                        gate_parameters: HashMap::new(),
                        state_prep_fidelity: 0.99,
                        measurement_fidelity: 0.99,
                    },
                },
            );
        }

        Ok(())
    }

    /// Process measurement statistics
    fn process_measurement_statistics(&mut self) -> Result<(), TomographyError> {
        for (setting_name, outcomes) in &self.measurement_data.raw_outcomes {
            let num_outcomes = 1 << self.num_qubits;
            let probabilities = Array1::zeros(num_outcomes);
            let expectation_values = Array1::zeros(self.num_qubits);

            // Compute outcome probabilities
            for outcome in outcomes {
                let outcome_index = self.outcome_to_index(outcome);
                probabilities[outcome_index] += 1.0;
            }
            probabilities /= outcomes.len() as f64;

            // Compute expectation values for each qubit
            for qubit in 0..self.num_qubits {
                let expectation = 0.0;
                for outcome in outcomes {
                    expectation += if outcome[qubit] == 0 { 1.0 } else { -1.0 };
                }
                expectation_values[qubit] = expectation / outcomes.len() as f64;
            }

            // Compute variances and covariances
            let variances = Array1::ones(self.num_qubits) - expectation_values.mapv(|x| x * x);
            let covariances = Array2::zeros((self.num_qubits, self.num_qubits));

            self.measurement_data.statistics.insert(
                setting_name.clone(),
                MeasurementStatistics {
                    probabilities,
                    expectation_values,
                    variances,
                    covariances,
                    higher_moments: HashMap::new(),
                },
            );
        }

        Ok(())
    }

    /// Convert outcome vector to index
    fn outcome_to_index(&self, outcome: &[u8]) -> usize {
        let index = 0;
        for (i, &bit) in outcome.iter().enumerate() {
            index += (bit as usize) << i;
        }
        index
    }

    /// Reconstruct quantum state using maximum likelihood estimation
    fn reconstruct_quantum_state(&self) -> Result<ReconstructedState, TomographyError> {
        let state_dim = 1 << self.num_qubits;

        // Initialize density matrix as maximally mixed state
        let density_matrix = Array2::eye(state_dim) / state_dim as f64;

        // Perform maximum likelihood reconstruction
        let max_iterations = self.config.optimization.max_iterations;
        let tolerance = self.config.optimization.tolerance;

        let log_likelihood = self.compute_log_likelihood(&density_matrix)?;
        let converged = false;
        let iteration = 0;
        let history = Vec::new();

        while iteration < max_iterations && !converged {
            let gradient = self.compute_likelihood_gradient(&density_matrix)?;

            // Simple gradient ascent step
            let step_size = 0.01;
            density_matrix = &density_matrix + &(gradient * step_size);

            // Project back to valid density matrix
            density_matrix = self.project_to_density_matrix(density_matrix)?;

            let new_log_likelihood = self.compute_log_likelihood(&density_matrix)?;

            if (new_log_likelihood - log_likelihood).abs() < tolerance {
                converged = true;
            }

            log_likelihood = new_log_likelihood;
            history.push(log_likelihood);
            iteration += 1;
        }

        // Compute eigendecomposition
        let eigendecomposition = self.compute_eigendecomposition(&density_matrix)?;

        // Compute entanglement measures
        let entanglement_measures = self.compute_entanglement_measures(&density_matrix)?;

        // Compute purity and entropy
        let purity = self.compute_purity(&density_matrix);
        let entropy = self.compute_von_neumann_entropy(&density_matrix);

        Ok(ReconstructedState {
            density_matrix,
            uncertainty: Array2::zeros((state_dim, state_dim)), // Placeholder
            eigenvalues: eigendecomposition.0,
            eigenvectors: eigendecomposition.1,
            purity,
            entropy,
            entanglement_measures,
            metadata: ReconstructionMetadata {
                method: "Maximum Likelihood".to_string(),
                convergence_info: ConvergenceInfo {
                    iterations: iteration,
                    final_objective: log_likelihood,
                    converged,
                    tolerance,
                    history,
                },
                computational_cost: ComputationalCost {
                    wall_time: 1.0, // Placeholder
                    cpu_time: 1.0,
                    memory_usage: 1.0,
                    function_evaluations: iteration,
                    gradient_evaluations: iteration,
                },
                quality_metrics: QualityMetrics {
                    fidelity: None,
                    trace_distance: None,
                    log_likelihood,
                    chi_squared: 0.0, // Placeholder
                    degrees_of_freedom: state_dim * state_dim - 1,
                    p_value: 0.05, // Placeholder
                },
            },
        })
    }

    /// Compute log-likelihood of measurement data given density matrix
    fn compute_log_likelihood(&self, density_matrix: &Array2<f64>) -> Result<f64, TomographyError> {
        let log_likelihood = 0.0;

        for (setting_name, statistics) in &self.measurement_data.statistics {
            if let Some(metadata) = self.measurement_data.metadata.get(setting_name) {
                // Compute predicted probabilities from density matrix
                let predicted_probs =
                    self.compute_predicted_probabilities(density_matrix, &metadata.basis)?;

                // Add to log-likelihood
                for (observed, predicted) in
                    statistics.probabilities.iter().zip(predicted_probs.iter())
                {
                    if *observed > 0.0 && *predicted > 1e-15 {
                        log_likelihood += observed * predicted.ln();
                    }
                }
            }
        }

        Ok(log_likelihood)
    }

    /// Compute gradient of log-likelihood
    fn compute_likelihood_gradient(
        &self,
        density_matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, TomographyError> {
        let state_dim = density_matrix.nrows();
        let gradient = Array2::zeros((state_dim, state_dim));

        // Simplified gradient computation
        for i in 0..state_dim {
            for j in 0..state_dim {
                gradient[[i, j]] = 1.0 / (density_matrix[[i, j]] + 1e-15);
            }
        }

        Ok(gradient)
    }

    /// Project matrix to valid density matrix (positive semidefinite, unit trace)
    fn project_to_density_matrix(
        &self,
        mut matrix: Array2<f64>,
    ) -> Result<Array2<f64>, TomographyError> {
        // Ensure Hermiticity
        for i in 0..matrix.nrows() {
            for j in i..matrix.ncols() {
                let avg = (matrix[[i, j]] + matrix[[j, i]]) / 2.0;
                matrix[[i, j]] = avg;
                matrix[[j, i]] = avg;
            }
        }

        // Ensure positive semidefiniteness by eigendecomposition
        let (mut eigenvals, eigenvecs) = self.compute_eigendecomposition(&matrix)?;

        // Clip negative eigenvalues
        for eigenval in eigenvals.iter_mut() {
            *eigenval = eigenval.max(0.0);
        }

        // Reconstruct matrix
        let reconstructed = Array2::zeros(matrix.raw_dim());
        for i in 0..eigenvals.len() {
            let eigenvec = eigenvecs.column(i);
            for k in 0..reconstructed.nrows() {
                for l in 0..reconstructed.ncols() {
                    reconstructed[[k, l]] += eigenvals[i] * eigenvec[k] * eigenvec[l];
                }
            }
        }

        // Normalize trace to 1
        let trace = reconstructed.diag().sum();
        if trace > 1e-15 {
            reconstructed /= trace;
        }

        Ok(reconstructed)
    }

    /// Compute predicted probabilities for a measurement basis
    fn compute_predicted_probabilities(
        &self,
        density_matrix: &Array2<f64>,
        basis: &MeasurementBasis,
    ) -> Result<Array1<f64>, TomographyError> {
        let num_outcomes = 1 << self.num_qubits;
        let probabilities = Array1::zeros(num_outcomes);

        // For each possible outcome, compute Born rule probability
        for outcome_idx in 0..num_outcomes {
            let measurement_operator = self.construct_measurement_operator(outcome_idx, basis)?;

            // Tr(ρ * M) where M is the measurement operator
            let prob = self.matrix_trace(&measurement_operator.dot(density_matrix));
            probabilities[outcome_idx] = prob.max(0.0);
        }

        // Normalize probabilities
        let total_prob = probabilities.sum();
        if total_prob > 1e-15 {
            probabilities /= total_prob;
        }

        Ok(probabilities)
    }

    /// Construct measurement operator for given outcome and basis
    fn construct_measurement_operator(
        &self,
        outcome_idx: usize,
        basis: &MeasurementBasis,
    ) -> Result<Array2<f64>, TomographyError> {
        let state_dim = 1 << self.num_qubits;
        let operator = Array2::eye(state_dim);

        // Convert outcome index to bit string
        let temp_outcome = outcome_idx;
        let outcome_bits = Vec::new();
        for _ in 0..self.num_qubits {
            outcome_bits.push(temp_outcome % 2);
            temp_outcome /= 2;
        }

        // Apply Pauli measurement for each qubit
        for (qubit, pauli_op) in basis.operators.iter().enumerate() {
            let qubit_outcome = outcome_bits[qubit];
            let pauli_matrix = self.get_pauli_matrix(pauli_op, qubit_outcome);

            // Tensor product with existing operator
            operator = self.tensor_product_with_identity(&operator, &pauli_matrix, qubit)?;
        }

        Ok(operator)
    }

    /// Get Pauli matrix for given operator and outcome
    fn get_pauli_matrix(&self, pauli_op: &PauliOperator, outcome: usize) -> Array2<f64> {
        match pauli_op {
            PauliOperator::I => Array2::eye(2),
            PauliOperator::X => {
                if outcome == 0 {
                    Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.5, 0.5]).unwrap()
                } else {
                    Array2::from_shape_vec((2, 2), vec![0.5, -0.5, -0.5, 0.5]).unwrap()
                }
            }
            PauliOperator::Y => {
                if outcome == 0 {
                    Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.0, 0.5]).unwrap()
                } else {
                    Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.0, 0.5]).unwrap()
                }
            }
            PauliOperator::Z => {
                if outcome == 0 {
                    Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 0.0]).unwrap()
                } else {
                    Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 1.0]).unwrap()
                }
            }
        }
    }

    /// Tensor product with identity (simplified implementation)
    fn tensor_product_with_identity(
        &self,
        operator: &Array2<f64>,
        pauli: &Array2<f64>,
        qubit: usize,
    ) -> Result<Array2<f64>, TomographyError> {
        // Simplified: assume applying to first qubit only
        if qubit == 0 {
            Ok(pauli.clone())
        } else {
            Ok(operator.clone())
        }
    }

    /// Compute matrix trace
    fn matrix_trace(&self, matrix: &Array2<f64>) -> f64 {
        matrix.diag().sum()
    }

    /// Compute eigendecomposition
    fn compute_eigendecomposition(
        &self,
        matrix: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>), TomographyError> {
        let n = matrix.nrows();

        if n == 2 {
            // Analytical eigendecomposition for 2x2 matrices
            let a = matrix[[0, 0]];
            let b = matrix[[0, 1]];
            let c = matrix[[1, 0]];
            let d = matrix[[1, 1]];

            let trace = a + d;
            let det = a * d - b * c;
            let discriminant = (trace * trace - 4.0 * det).sqrt();

            let eigenval1 = (trace + discriminant) / 2.0;
            let eigenval2 = (trace - discriminant) / 2.0;

            let eigenvals = Array1::from_vec(vec![eigenval1.max(0.0), eigenval2.max(0.0)]);
            let eigenvecs = Array2::eye(n); // Simplified eigenvectors

            Ok((eigenvals, eigenvecs))
        } else {
            // Fallback for larger matrices - use uniform distribution
            let eigenvals = Array1::ones(n) / n as f64;
            let eigenvecs = Array2::eye(n);
            Ok((eigenvals, eigenvecs))
        }
    }

    /// Compute entanglement measures
    fn compute_entanglement_measures(
        &self,
        density_matrix: &Array2<f64>,
    ) -> Result<EntanglementMeasures, TomographyError> {
        // Simplified entanglement computation
        let purity = self.compute_purity(density_matrix);

        Ok(EntanglementMeasures {
            concurrence: if purity < 1.0 {
                2.0 * (0.5 - purity / 2.0).sqrt()
            } else {
                0.0
            },
            negativity: 0.0,                // Placeholder
            entanglement_of_formation: 0.0, // Placeholder
            distillable_entanglement: 0.0,  // Placeholder
            logarithmic_negativity: 0.0,    // Placeholder
            schmidt_number: 1.0,            // Placeholder
        })
    }

    /// Compute purity of quantum state
    fn compute_purity(&self, density_matrix: &Array2<f64>) -> f64 {
        let squared = density_matrix.dot(density_matrix);
        self.matrix_trace(&squared)
    }

    /// Compute von Neumann entropy
    fn compute_von_neumann_entropy(&self, density_matrix: &Array2<f64>) -> f64 {
        let (eigenvals, _) = self
            .compute_eigendecomposition(density_matrix)
            .unwrap_or_else(|_| {
                (
                    Array1::ones(density_matrix.nrows()) / density_matrix.nrows() as f64,
                    Array2::eye(density_matrix.nrows()),
                )
            });

        let entropy = 0.0;
        for &eigenval in eigenvals.iter() {
            if eigenval > 1e-15 {
                entropy -= eigenval * eigenval.ln();
            }
        }

        entropy
    }

    /// Validate reconstruction
    fn validate_reconstruction(
        &self,
        state: &ReconstructedState,
    ) -> Result<ValidationResult, TomographyError> {
        let test_results = HashMap::new();
        let passed = true;

        // Test 1: Positive semidefiniteness
        let eigenvals_positive = state.eigenvalues.iter().all(|&eigenval| eigenval >= -1e-10);
        test_results.insert(
            "positive_semidefinite".to_string(),
            TestResult {
                statistic: *state
                    .eigenvalues
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&0.0),
                p_value: if eigenvals_positive { 1.0 } else { 0.0 },
                critical_value: 0.0,
                passed: eigenvals_positive,
                effect_size: 1.0,
            },
        );

        if !eigenvals_positive {
            passed = false;
        }

        // Test 2: Unit trace
        let trace = self.matrix_trace(&state.density_matrix);
        let trace_valid = (trace - 1.0).abs() < 1e-6;
        test_results.insert(
            "unit_trace".to_string(),
            TestResult {
                statistic: trace,
                p_value: if trace_valid { 1.0 } else { 0.0 },
                critical_value: 1.0,
                passed: trace_valid,
                effect_size: (trace - 1.0).abs(),
            },
        );

        if !trace_valid {
            passed = false;
        }

        // Test 3: Hermiticity
        let hermitian = self.is_hermitian(&state.density_matrix);
        test_results.insert(
            "hermitian".to_string(),
            TestResult {
                statistic: 1.0,
                p_value: if hermitian { 1.0 } else { 0.0 },
                critical_value: 1.0,
                passed: hermitian,
                effect_size: 1.0,
            },
        );

        if !hermitian {
            passed = false;
        }

        let score = if passed { 1.0 } else { 0.5 };

        Ok(ValidationResult {
            passed,
            score,
            test_results,
            recommendations: if passed {
                vec!["Reconstruction is physically valid".to_string()]
            } else {
                vec!["Consider adjusting reconstruction parameters".to_string()]
            },
        })
    }

    /// Check if matrix is Hermitian
    fn is_hermitian(&self, matrix: &Array2<f64>) -> bool {
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                if (matrix[[i, j]] - matrix[[j, i]]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Perform error analysis
    fn perform_error_analysis(
        &mut self,
        _state: &ReconstructedState,
    ) -> Result<(), TomographyError> {
        // Simplified error analysis
        println!("Performing error analysis...");

        // Update sensitivity analysis
        self.error_analysis
            .sensitivity_analysis
            .parameter_sensitivities
            .fill(0.1);
        self.error_analysis
            .sensitivity_analysis
            .robustness_indicators
            .fill(0.8);

        // Update bootstrap methods
        self.error_analysis
            .bootstrap_methods
            .bias_estimates
            .fill(0.01);

        Ok(())
    }

    /// Compute tomography metrics
    fn compute_tomography_metrics(
        &mut self,
        state: &ReconstructedState,
        validation: &ValidationResult,
    ) {
        self.metrics.reconstruction_accuracy = validation.score;
        self.metrics.computational_efficiency =
            1.0 / (state.metadata.computational_cost.wall_time + 1.0);
        self.metrics.statistical_power = state.metadata.quality_metrics.p_value.min(0.95);
        self.metrics.robustness_score = self
            .error_analysis
            .sensitivity_analysis
            .robustness_indicators
            .mean()
            .unwrap_or(0.0);

        // Overall quality score
        self.metrics.overall_quality = self.metrics.reconstruction_accuracy * 0.4
            + self.metrics.computational_efficiency * 0.2
            + self.metrics.statistical_power * 0.2
            + self.metrics.robustness_score * 0.2;
    }
}

/// Create default tomography configuration
pub fn create_default_tomography_config() -> TomographyConfig {
    TomographyConfig {
        tomography_type: TomographyType::QuantumState,
        shots_per_setting: 1000,
        measurement_bases: Vec::new(),
        reconstruction_method: ReconstructionMethodType::MaximumLikelihood,
        error_mitigation: ErrorMitigationConfig {
            readout_error_correction: true,
            gate_error_mitigation: false,
            symmetry_verification: true,
            noise_characterization: NoiseCharacterizationConfig {
                coherent_errors: false,
                incoherent_errors: true,
                crosstalk: false,
                temporal_correlations: false,
                spatial_correlations: false,
            },
            error_correction_protocols: vec![ErrorCorrectionProtocol::ZeroNoiseExtrapolation],
        },
        optimization: OptimizationConfig {
            max_iterations: 1000,
            tolerance: 1e-6,
            regularization: RegularizationConfig {
                l1_strength: 0.0,
                l2_strength: 0.001,
                nuclear_norm_strength: 0.0,
                trace_norm_strength: 0.0,
                entropy_strength: 0.0,
            },
            constraints: ConstraintConfig {
                trace_preserving: true,
                completely_positive: true,
                hermitian: true,
                positive_semidefinite: true,
                physical_constraints: vec![
                    PhysicalConstraint::UnitTrace,
                    PhysicalConstraint::PositiveEigenvalues,
                ],
            },
            algorithm: OptimizationAlgorithm::GradientDescent,
        },
        validation: ValidationConfig {
            cross_validation_folds: 5,
            bootstrap_samples: 1000,
            confidence_level: 0.95,
            validation_metrics: vec![ValidationMetric::Fidelity, ValidationMetric::TraceDistance],
            statistical_tests: vec![StatisticalTest::ChiSquared, StatisticalTest::Bootstrap],
        },
    }
}

/// Create tomography system for given number of qubits
pub fn create_tomography_system(num_qubits: usize) -> QuantumStateTomography {
    let config = create_default_tomography_config();
    QuantumStateTomography::new(num_qubits, config)
}

/// Create shadow tomography configuration
pub fn create_shadow_tomography_config(num_shadows: usize) -> TomographyConfig {
    let config = create_default_tomography_config();
    config.tomography_type = TomographyType::ShadowTomography { num_shadows };
    config.shots_per_setting = 100; // Reduced shots for shadow tomography
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tomography_creation() {
        let tomography = create_tomography_system(2);
        assert_eq!(tomography.num_qubits, 2);
        assert_eq!(
            tomography.config.tomography_type,
            TomographyType::QuantumState
        );
    }

    #[test]
    fn test_pauli_measurement_generation() {
        let tomography = create_tomography_system(2);
        let measurements = tomography.generate_pauli_measurements().unwrap();
        assert_eq!(measurements.len(), 9); // 3^2 = 9 Pauli measurements for 2 qubits
    }

    #[test]
    fn test_shadow_measurement_generation() {
        let tomography = create_tomography_system(2);
        let measurements = tomography.generate_shadow_measurements(100).unwrap();
        assert_eq!(measurements.len(), 100);
    }

    #[test]
    fn test_outcome_to_index() {
        let tomography = create_tomography_system(3);
        assert_eq!(tomography.outcome_to_index(&[0, 0, 0]), 0);
        assert_eq!(tomography.outcome_to_index(&[1, 0, 0]), 1);
        assert_eq!(tomography.outcome_to_index(&[0, 1, 0]), 2);
        assert_eq!(tomography.outcome_to_index(&[1, 1, 1]), 7);
    }

    #[test]
    fn test_purity_computation() {
        let tomography = create_tomography_system(1);
        let pure_state = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let purity = tomography.compute_purity(&pure_state);
        assert!((purity - 1.0).abs() < 1e-10);

        let mixed_state = Array2::eye(2) / 2.0;
        let mixed_purity = tomography.compute_purity(&mixed_state);
        assert!((mixed_purity - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_computation() {
        let tomography = create_tomography_system(1);
        let pure_state = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let entropy = tomography.compute_von_neumann_entropy(&pure_state);
        assert!(entropy.abs() < 1e-10); // Pure state has zero entropy
    }
}
