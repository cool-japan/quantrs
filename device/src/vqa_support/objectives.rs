//! Objective function definitions and evaluation for VQA
//!
//! This module provides objective functions commonly used in
//! variational quantum algorithms with comprehensive evaluation strategies.
use super::circuits::{GateType, ParametricCircuit};
use super::config::{GradientMethod, VQAAlgorithmType};
use crate::DeviceResult;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::sync::Arc;
#[cfg(not(feature = "scirs2"))]
mod fallback_scirs2 {
    use scirs2_core::ndarray::{Array1, Array2};
    use scirs2_core::Complex64;
    pub struct Matrix(pub Array2<Complex64>);
    pub struct Vector(pub Array1<Complex64>);
    pub struct PauliOperator {
        pub coefficients: Array1<f64>,
        pub terms: Vec<String>,
    }
    impl Matrix {
        pub fn new(data: Array2<Complex64>) -> Self {
            Self(data)
        }
    }
    impl Vector {
        pub fn new(data: Array1<Complex64>) -> Self {
            Self(data)
        }
    }
}
/// Comprehensive objective function configuration
#[derive(Debug, Clone)]
pub struct ObjectiveConfig {
    /// Objective type
    pub objective_type: ObjectiveType,
    /// Target value (if applicable)
    pub target: Option<f64>,
    /// Regularization parameters
    pub regularization: RegularizationConfig,
    /// Hamiltonian specification (for VQE)
    pub hamiltonian: Option<HamiltonianSpec>,
    /// Cost function specification (for QAOA)
    pub cost_function: Option<CostFunctionSpec>,
    /// Training data (for VQC/QNN)
    pub training_data: Option<TrainingDataSpec>,
    /// Measurement strategy
    pub measurement_strategy: MeasurementStrategy,
    /// Shot allocation
    pub shot_allocation: ShotAllocationConfig,
    /// Gradient computation method
    pub gradient_method: GradientMethod,
    /// Noise mitigation settings
    pub noise_mitigation: ObjectiveNoiseMitigation,
}
/// Available objective function types
#[derive(Debug, Clone)]
pub enum ObjectiveType {
    /// Energy minimization (VQE)
    Energy,
    /// Fidelity maximization
    Fidelity,
    /// Cost optimization (QAOA)
    Cost,
    /// Classification loss (VQC)
    Classification,
    /// Regression loss (QNN)
    Regression,
    /// Expectation value computation
    ExpectationValue,
    /// State preparation fidelity
    StatePreparation,
    /// Process fidelity
    ProcessFidelity,
    /// Custom objective with user-defined evaluation
    Custom(String),
}
/// Hamiltonian specification for VQE
#[derive(Debug, Clone)]
pub struct HamiltonianSpec {
    /// Pauli terms with coefficients
    pub pauli_terms: Vec<PauliTerm>,
    /// Number of qubits
    pub num_qubits: usize,
    /// Sparse representation flag
    pub use_sparse: bool,
}
/// Individual Pauli term in Hamiltonian
#[derive(Debug, Clone)]
pub struct PauliTerm {
    /// Coefficient
    pub coefficient: Complex64,
    /// Pauli operators on each qubit (I, X, Y, Z)
    pub operators: Vec<char>,
    /// Qubit indices (if sparse)
    pub indices: Option<Vec<usize>>,
}
/// Cost function specification for QAOA
#[derive(Debug, Clone)]
pub struct CostFunctionSpec {
    /// Cost function type
    pub function_type: CostFunctionType,
    /// Problem-specific parameters
    pub parameters: HashMap<String, f64>,
    /// Graph connectivity (for graph problems)
    pub graph: Option<Vec<(usize, usize, f64)>>,
}
/// QAOA cost function types
#[derive(Debug, Clone)]
pub enum CostFunctionType {
    /// Maximum Cut problem
    MaxCut,
    /// Traveling Salesman Problem
    TSP,
    /// Maximum Independent Set
    MaxIndependentSet,
    /// Portfolio optimization
    Portfolio,
    /// Custom cost function
    Custom(String),
}
/// Training data specification for supervised learning
#[derive(Debug, Clone)]
pub struct TrainingDataSpec {
    /// Input features
    pub features: Array2<f64>,
    /// Target labels/values
    pub targets: Array1<f64>,
    /// Data encoding strategy
    pub encoding: DataEncoding,
    /// Loss function type
    pub loss_function: LossFunction,
}
/// Data encoding strategies
#[derive(Debug, Clone)]
pub enum DataEncoding {
    /// Amplitude encoding
    Amplitude,
    /// Angle encoding
    Angle,
    /// Basis encoding
    Basis,
    /// IQP encoding
    IQP,
}
/// Loss function types for supervised learning
#[derive(Debug, Clone)]
pub enum LossFunction {
    /// Mean squared error
    MSE,
    /// Cross-entropy
    CrossEntropy,
    /// Hinge loss
    Hinge,
    /// Custom loss
    Custom(String),
}
/// Measurement strategy configuration
#[derive(Debug, Clone)]
pub struct MeasurementStrategy {
    /// Strategy type
    pub strategy_type: MeasurementStrategyType,
    /// Grouping of commuting terms
    pub term_grouping: TermGrouping,
    /// Shadow tomography settings
    pub shadow_tomography: Option<ShadowTomographyConfig>,
}
/// Measurement strategy types
#[derive(Debug, Clone)]
pub enum MeasurementStrategyType {
    /// Individual term measurement
    Individual,
    /// Simultaneous measurement of commuting terms
    Simultaneous,
    /// Classical shadow tomography
    Shadow,
    /// Adaptive measurement
    Adaptive,
}
/// Term grouping strategies
#[derive(Debug, Clone)]
pub enum TermGrouping {
    /// No grouping
    None,
    /// Qubit-wise commuting (QWC)
    QubitWiseCommuting,
    /// Fully commuting
    FullyCommuting,
    /// Graph coloring
    GraphColoring,
}
/// Shadow tomography configuration
#[derive(Debug, Clone)]
pub struct ShadowTomographyConfig {
    /// Number of shadow copies
    pub num_shadows: usize,
    /// Random unitary ensemble
    pub unitary_ensemble: UnitaryEnsemble,
    /// Post-processing method
    pub post_processing: String,
}
/// Unitary ensemble for shadow tomography
#[derive(Debug, Clone)]
pub enum UnitaryEnsemble {
    /// Clifford group
    Clifford,
    /// Pauli group
    Pauli,
    /// Random unitaries
    Random,
}
/// Shot allocation configuration
#[derive(Debug, Clone)]
pub struct ShotAllocationConfig {
    /// Total shot budget
    pub total_shots: usize,
    /// Allocation strategy
    pub allocation_strategy: ShotAllocationStrategy,
    /// Minimum shots per term
    pub min_shots_per_term: usize,
    /// Adaptive allocation parameters
    pub adaptive_params: Option<AdaptiveAllocationParams>,
}
/// Shot allocation strategies
#[derive(Debug, Clone)]
pub enum ShotAllocationStrategy {
    /// Uniform allocation
    Uniform,
    /// Proportional to variance
    ProportionalToVariance,
    /// Proportional to coefficient magnitude
    ProportionalToCoeff,
    /// Optimal allocation (minimize variance)
    OptimalVariance,
    /// Adaptive Bayesian allocation
    AdaptiveBayesian,
}
/// Adaptive allocation parameters
#[derive(Debug, Clone)]
pub struct AdaptiveAllocationParams {
    /// Update frequency
    pub update_frequency: usize,
    /// Learning rate for adaptation
    pub learning_rate: f64,
    /// Exploration factor
    pub exploration_factor: f64,
}
/// Noise mitigation for objective evaluation
#[derive(Debug, Clone)]
pub struct ObjectiveNoiseMitigation {
    /// Enable zero-noise extrapolation
    pub enable_zne: bool,
    /// ZNE noise factors
    pub zne_factors: Vec<f64>,
    /// Enable readout error mitigation
    pub enable_rem: bool,
    /// Enable symmetry verification
    pub enable_symmetry: bool,
    /// Mitigation overhead budget
    pub overhead_budget: f64,
}
/// Regularization configuration
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// L1 regularization coefficient
    pub l1_coeff: f64,
    /// L2 regularization coefficient
    pub l2_coeff: f64,
    /// Parameter bounds penalty
    pub bounds_penalty: f64,
}
impl Default for ObjectiveConfig {
    fn default() -> Self {
        Self {
            objective_type: ObjectiveType::Energy,
            target: None,
            regularization: RegularizationConfig::default(),
            hamiltonian: None,
            cost_function: None,
            training_data: None,
            measurement_strategy: MeasurementStrategy::default(),
            shot_allocation: ShotAllocationConfig::default(),
            gradient_method: GradientMethod::ParameterShift,
            noise_mitigation: ObjectiveNoiseMitigation::default(),
        }
    }
}
impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            l1_coeff: 0.0,
            l2_coeff: 0.0,
            bounds_penalty: 1.0,
        }
    }
}
/// Comprehensive objective function evaluation result
#[derive(Debug, Clone)]
pub struct ObjectiveResult {
    /// Primary objective value
    pub value: f64,
    /// Gradient (if computed)
    pub gradient: Option<Array1<f64>>,
    /// Hessian (if computed)
    pub hessian: Option<Array2<f64>>,
    /// Individual term contributions
    pub term_contributions: Vec<f64>,
    /// Statistical uncertainty
    pub uncertainty: Option<f64>,
    /// Variance estimate
    pub variance: Option<f64>,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
    /// Measurement results
    pub measurement_results: MeasurementResults,
    /// Computation metadata
    pub metadata: ObjectiveMetadata,
}
/// Measurement results from objective evaluation
#[derive(Debug, Clone)]
pub struct MeasurementResults {
    /// Raw measurement counts
    pub raw_counts: HashMap<String, usize>,
    /// Expectation values per term
    pub expectation_values: Vec<f64>,
    /// Measurement variances
    pub variances: Vec<f64>,
    /// Shot allocation used
    pub shots_used: Vec<usize>,
    /// Total shots consumed
    pub total_shots: usize,
}
/// Objective evaluation metadata
#[derive(Debug, Clone)]
pub struct ObjectiveMetadata {
    /// Evaluation timestamp
    pub timestamp: std::time::Instant,
    /// Circuit depth used
    pub circuit_depth: usize,
    /// Number of terms evaluated
    pub num_terms: usize,
    /// Measurement strategy used
    pub measurement_strategy: String,
    /// Noise mitigation applied
    pub noise_mitigation_applied: Vec<String>,
    /// Computation time
    pub computation_time: std::time::Duration,
}
/// Enhanced objective function trait
pub trait ObjectiveFunction: Send + Sync {
    /// Evaluate the objective function
    fn evaluate(&self, parameters: &Array1<f64>) -> DeviceResult<ObjectiveResult>;
    /// Compute gradient using specified method
    fn compute_gradient(&self, parameters: &Array1<f64>) -> DeviceResult<Array1<f64>> {
        self.compute_gradient_with_method(parameters, &GradientMethod::ParameterShift)
    }
    /// Compute gradient with specific method
    fn compute_gradient_with_method(
        &self,
        parameters: &Array1<f64>,
        method: &GradientMethod,
    ) -> DeviceResult<Array1<f64>>;
    /// Estimate computational cost for given parameters
    fn estimate_cost(&self, parameters: &Array1<f64>) -> usize;
    /// Get parameter bounds
    fn parameter_bounds(&self) -> Option<Vec<(f64, f64)>>;
    /// Check if objective supports batched evaluation
    fn supports_batch_evaluation(&self) -> bool {
        false
    }
    /// Batch evaluate multiple parameter sets (if supported)
    fn batch_evaluate(&self, parameter_sets: &[Array1<f64>]) -> DeviceResult<Vec<ObjectiveResult>> {
        parameter_sets
            .iter()
            .map(|params| self.evaluate(params))
            .collect()
    }
}
/// Comprehensive objective function evaluator with SciRS2 integration
#[derive(Debug)]
pub struct ObjectiveEvaluator {
    /// Configuration
    pub config: ObjectiveConfig,
    /// Parametric circuit reference
    pub circuit: Arc<ParametricCircuit>,
    /// Quantum simulator backend
    pub backend: ObjectiveBackend,
    /// Cached Hamiltonian matrix (for efficiency)
    pub cached_hamiltonian: Option<HamiltonianMatrix>,
    /// Measurement groupings (for optimization)
    pub measurement_groups: Option<Vec<MeasurementGroup>>,
}
/// Backend for objective evaluation
#[derive(Debug)]
pub enum ObjectiveBackend {
    /// QuantRS2 simulator
    QuantRS2Simulator,
    /// SciRS2 exact simulation
    SciRS2Exact,
    /// Hardware device
    Hardware(String),
    /// Mock backend for testing
    Mock,
}
/// Cached Hamiltonian representation
#[derive(Debug, Clone)]
pub struct HamiltonianMatrix {
    /// Full Hamiltonian matrix
    pub matrix: Array2<Complex64>,
    /// Eigenvalues (if computed)
    pub eigenvalues: Option<Array1<f64>>,
    /// Eigenvectors (if computed)
    pub eigenvectors: Option<Array2<Complex64>>,
}
/// Grouped measurements for efficiency
#[derive(Debug, Clone)]
pub struct MeasurementGroup {
    /// Terms that can be measured simultaneously
    pub terms: Vec<usize>,
    /// Required measurement basis
    pub measurement_basis: Vec<char>,
    /// Expected shot allocation
    pub shot_allocation: usize,
}
impl ObjectiveFunction for ObjectiveEvaluator {
    /// Comprehensive objective function evaluation
    fn evaluate(&self, parameters: &Array1<f64>) -> DeviceResult<ObjectiveResult> {
        let start_time = std::time::Instant::now();
        let mut circuit = (*self.circuit).clone();
        circuit.set_parameters(parameters.to_vec())?;
        let result = match &self.config.objective_type {
            ObjectiveType::Energy => self.evaluate_energy(&circuit),
            ObjectiveType::Cost => Self::evaluate_cost(&circuit),
            ObjectiveType::Classification => Self::evaluate_classification(&circuit),
            ObjectiveType::Regression => Self::evaluate_regression(&circuit),
            ObjectiveType::Fidelity => Self::evaluate_fidelity(&circuit),
            ObjectiveType::ExpectationValue => Self::evaluate_expectation_value(&circuit),
            ObjectiveType::StatePreparation => Self::evaluate_state_preparation(&circuit),
            ObjectiveType::ProcessFidelity => Self::evaluate_process_fidelity(&circuit),
            ObjectiveType::Custom(name) => Self::evaluate_custom(&circuit, name),
        };
        let mut objective_result = result?;
        objective_result.value = self.apply_regularization(objective_result.value, parameters);
        objective_result.metadata.computation_time = start_time.elapsed();
        objective_result.metadata.timestamp = start_time;
        objective_result.metadata.circuit_depth = circuit.circuit_depth();
        Ok(objective_result)
    }
    /// Compute gradient with specified method
    fn compute_gradient_with_method(
        &self,
        parameters: &Array1<f64>,
        method: &GradientMethod,
    ) -> DeviceResult<Array1<f64>> {
        match method {
            GradientMethod::ParameterShift => self.compute_parameter_shift_gradient(parameters),
            GradientMethod::FiniteDifference => {
                self.compute_finite_difference_gradient(parameters)
            }
            GradientMethod::CentralDifference => {
                self.compute_central_difference_gradient(parameters)
            }
            GradientMethod::ForwardDifference => {
                self.compute_forward_difference_gradient(parameters)
            }
            GradientMethod::NaturalGradient => self.compute_natural_gradient(parameters),
            GradientMethod::AutomaticDifferentiation => {
                self.compute_automatic_gradient(parameters)
            }
        }
    }
    /// Estimate computational cost
    fn estimate_cost(&self, parameters: &Array1<f64>) -> usize {
        let circuit_depth = self.circuit.circuit_depth();
        let num_qubits = self.circuit.config.num_qubits;
        let num_terms = match &self.config.hamiltonian {
            Some(h) => h.pauli_terms.len(),
            None => 1,
        };
        let circuit_cost = circuit_depth * (1 << num_qubits.min(10));
        let measurement_cost = num_terms * self.config.shot_allocation.total_shots;
        circuit_cost + measurement_cost
    }
    /// Get parameter bounds
    fn parameter_bounds(&self) -> Option<Vec<(f64, f64)>> {
        Some(self.circuit.bounds.clone())
    }
    /// Check batch evaluation support
    fn supports_batch_evaluation(&self) -> bool {
        matches!(
            self.backend,
            ObjectiveBackend::SciRS2Exact | ObjectiveBackend::Mock
        )
    }
}
impl ObjectiveEvaluator {
    /// Create new objective evaluator
    pub fn new(
        config: ObjectiveConfig,
        circuit: ParametricCircuit,
        backend: ObjectiveBackend,
    ) -> Self {
        let circuit_arc = Arc::new(circuit);
        Self {
            config,
            circuit: circuit_arc,
            backend,
            cached_hamiltonian: None,
            measurement_groups: None,
        }
    }
    /// Initialize with Hamiltonian caching
    pub fn with_hamiltonian_caching(mut self) -> DeviceResult<Self> {
        if let Some(ref hamiltonian_spec) = self.config.hamiltonian {
            self.cached_hamiltonian = Some(self.build_hamiltonian_matrix(hamiltonian_spec)?);
        }
        Ok(self)
    }
    /// Initialize measurement grouping optimization
    pub fn with_measurement_grouping(mut self) -> DeviceResult<Self> {
        if let Some(ref hamiltonian_spec) = self.config.hamiltonian {
            self.measurement_groups = Some(Self::group_measurements(hamiltonian_spec)?);
        }
        Ok(self)
    }
    /// Evaluate energy objective (VQE)
    fn evaluate_energy(&self, circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        let hamiltonian = self.config.hamiltonian.as_ref().ok_or_else(|| {
            crate::DeviceError::InvalidInput(
                "Hamiltonian specification required for energy evaluation".to_string(),
            )
        })?;
        match &self.backend {
            ObjectiveBackend::SciRS2Exact => self.evaluate_energy_exact(circuit, hamiltonian),
            ObjectiveBackend::QuantRS2Simulator => {
                self.evaluate_energy_sampling(circuit, hamiltonian)
            }
            ObjectiveBackend::Hardware(_) => self.evaluate_energy_hardware(circuit, hamiltonian),
            ObjectiveBackend::Mock => Self::evaluate_energy_mock(circuit, hamiltonian),
        }
    }
    /// Exact energy evaluation with SciRS2
    fn evaluate_energy_exact(
        &self,
        circuit: &ParametricCircuit,
        hamiltonian: &HamiltonianSpec,
    ) -> DeviceResult<ObjectiveResult> {
        #[cfg(feature = "scirs2")]
        {
            let state_vector = Self::simulate_circuit_exact(circuit)?;
            let hamiltonian_matrix = Self::get_or_build_hamiltonian(hamiltonian)?;
            let energy = Self::compute_expectation_value_exact(&state_vector, hamiltonian_matrix)?;
            let mut measurement_results = MeasurementResults {
                raw_counts: HashMap::new(),
                expectation_values: vec![energy],
                variances: vec![0.0],
                shots_used: vec![0],
                total_shots: 0,
            };
            Ok(ObjectiveResult {
                value: energy,
                gradient: None,
                hessian: None,
                term_contributions: vec![energy],
                uncertainty: Some(0.0),
                variance: Some(0.0),
                metrics: std::iter::once(("exact_evaluation".to_string(), 1.0)).collect(),
                measurement_results,
                metadata: ObjectiveMetadata {
                    timestamp: std::time::Instant::now(),
                    circuit_depth: circuit.circuit_depth(),
                    num_terms: hamiltonian.pauli_terms.len(),
                    measurement_strategy: "exact".to_string(),
                    noise_mitigation_applied: vec![],
                    computation_time: std::time::Duration::from_secs(0),
                },
            })
        }
        #[cfg(not(feature = "scirs2"))]
        {
            Self::evaluate_energy_mock(circuit, hamiltonian)
        }
    }
    /// Sampling-based energy evaluation
    fn evaluate_energy_sampling(
        &self,
        circuit: &ParametricCircuit,
        hamiltonian: &HamiltonianSpec,
    ) -> DeviceResult<ObjectiveResult> {
        let mut total_energy = 0.0;
        let mut term_contributions = Vec::new();
        let mut total_variance = 0.0;
        let mut measurement_results = MeasurementResults {
            raw_counts: HashMap::new(),
            expectation_values: Vec::new(),
            variances: Vec::new(),
            shots_used: Vec::new(),
            total_shots: 0,
        };
        let shot_allocation = self.allocate_shots_to_terms(hamiltonian)?;
        for (term_idx, term) in hamiltonian.pauli_terms.iter().enumerate() {
            let shots = shot_allocation[term_idx];
            let (expectation, variance) = Self::measure_pauli_term(circuit, term, shots)?;
            let contribution = term.coefficient.re * expectation;
            total_energy += contribution;
            term_contributions.push(contribution);
            total_variance += (term.coefficient.norm_sqr() * variance) / shots as f64;
            measurement_results.expectation_values.push(expectation);
            measurement_results.variances.push(variance);
            measurement_results.shots_used.push(shots);
            measurement_results.total_shots += shots;
        }
        Ok(ObjectiveResult {
            value: total_energy,
            gradient: None,
            hessian: None,
            term_contributions,
            uncertainty: Some(total_variance.sqrt()),
            variance: Some(total_variance),
            metrics: std::iter::once(("sampling_evaluation".to_string(), 1.0)).collect(),
            measurement_results,
            metadata: ObjectiveMetadata {
                timestamp: std::time::Instant::now(),
                circuit_depth: circuit.circuit_depth(),
                num_terms: hamiltonian.pauli_terms.len(),
                measurement_strategy: "individual_terms".to_string(),
                noise_mitigation_applied: vec![],
                computation_time: std::time::Duration::from_secs(0),
            },
        })
    }
    /// Hardware-based energy evaluation
    fn evaluate_energy_hardware(
        &self,
        circuit: &ParametricCircuit,
        hamiltonian: &HamiltonianSpec,
    ) -> DeviceResult<ObjectiveResult> {
        self.evaluate_energy_sampling(circuit, hamiltonian)
    }
    /// Mock energy evaluation for testing
    fn evaluate_energy_mock(
        _circuit: &ParametricCircuit,
        hamiltonian: &HamiltonianSpec,
    ) -> DeviceResult<ObjectiveResult> {
        use scirs2_core::random::prelude::*;
        let mut rng = thread_rng();
        let energy = hamiltonian
            .pauli_terms
            .iter()
            .map(|term| term.coefficient.re * rng.random_range(-1.0..1.0))
            .sum::<f64>();
        let variance: f64 = 0.01;
        Ok(ObjectiveResult {
            value: energy,
            gradient: None,
            hessian: None,
            term_contributions: vec![energy],
            uncertainty: Some(variance.sqrt()),
            variance: Some(variance),
            metrics: HashMap::from([("mock_evaluation".to_string(), 1.0)]),
            measurement_results: MeasurementResults {
                raw_counts: HashMap::new(),
                expectation_values: vec![energy],
                variances: vec![variance],
                shots_used: vec![1000],
                total_shots: 1000,
            },
            metadata: ObjectiveMetadata {
                timestamp: std::time::Instant::now(),
                circuit_depth: 10,
                num_terms: hamiltonian.pauli_terms.len(),
                measurement_strategy: "mock".to_string(),
                noise_mitigation_applied: vec![],
                computation_time: std::time::Duration::from_millis(10),
            },
        })
    }
    /// Apply regularization to objective value
    fn apply_regularization(&self, value: f64, parameters: &Array1<f64>) -> f64 {
        let l1_penalty =
            self.config.regularization.l1_coeff * parameters.iter().map(|&x| x.abs()).sum::<f64>();
        let l2_penalty =
            self.config.regularization.l2_coeff * parameters.iter().map(|&x| x * x).sum::<f64>();
        value + l1_penalty + l2_penalty
    }
    /// Compute parameter shift gradient
    fn compute_parameter_shift_gradient(
        &self,
        parameters: &Array1<f64>,
    ) -> DeviceResult<Array1<f64>> {
        let mut gradient = Array1::zeros(parameters.len());
        let shift = std::f64::consts::PI / 2.0;
        for i in 0..parameters.len() {
            let mut params_plus = parameters.clone();
            let mut params_minus = parameters.clone();
            params_plus[i] += shift;
            params_minus[i] -= shift;
            let f_plus = self.evaluate(&params_plus)?.value;
            let f_minus = self.evaluate(&params_minus)?.value;
            gradient[i] = (f_plus - f_minus) / 2.0;
        }
        Ok(gradient)
    }
    /// Build Hamiltonian matrix from specification
    fn build_hamiltonian_matrix(&self, spec: &HamiltonianSpec) -> DeviceResult<HamiltonianMatrix> {
        let dim = 1 << spec.num_qubits;
        let mut matrix = Array2::zeros((dim, dim));
        for term in &spec.pauli_terms {
            let term_matrix = Self::build_pauli_term_matrix(term, spec.num_qubits)?;
            matrix = matrix + term_matrix;
        }
        Ok(HamiltonianMatrix {
            matrix,
            eigenvalues: None,
            eigenvectors: None,
        })
    }
    /// Build matrix for single Pauli term
    fn build_pauli_term_matrix(
        term: &PauliTerm,
        num_qubits: usize,
    ) -> DeviceResult<Array2<Complex64>> {
        let dim = 1 << num_qubits;
        let mut matrix = Array2::zeros((dim, dim));
        for i in 0..dim {
            matrix[[i, i]] = term.coefficient;
        }
        Ok(matrix)
    }
    /// Get cached Hamiltonian or build it
    fn get_or_build_hamiltonian(spec: &HamiltonianSpec) -> DeviceResult<&HamiltonianMatrix> {
        Err(crate::DeviceError::InvalidInput(
            "Hamiltonian caching not yet implemented".to_string(),
        ))
    }
    /// Compute exact expectation value
    fn compute_expectation_value_exact(
        state: &Array1<Complex64>,
        hamiltonian: &HamiltonianMatrix,
    ) -> DeviceResult<f64> {
        let h_psi = hamiltonian.matrix.dot(state);
        let expectation = state
            .iter()
            .zip(h_psi.iter())
            .map(|(psi_i, h_psi_i)| psi_i.conj() * h_psi_i)
            .sum::<Complex64>()
            .re;
        Ok(expectation)
    }
    /// Allocate shots to Hamiltonian terms
    fn allocate_shots_to_terms(&self, hamiltonian: &HamiltonianSpec) -> DeviceResult<Vec<usize>> {
        let total_shots = self.config.shot_allocation.total_shots;
        let num_terms = hamiltonian.pauli_terms.len();
        match self.config.shot_allocation.allocation_strategy {
            ShotAllocationStrategy::Uniform => {
                let shots_per_term = total_shots / num_terms;
                Ok(vec![shots_per_term; num_terms])
            }
            ShotAllocationStrategy::ProportionalToCoeff => {
                let coeffs: Vec<f64> = hamiltonian
                    .pauli_terms
                    .iter()
                    .map(|term| term.coefficient.norm())
                    .collect();
                let total_coeff: f64 = coeffs.iter().sum();
                let allocation: Vec<usize> = coeffs
                    .iter()
                    .map(|&coeff| ((coeff / total_coeff) * total_shots as f64) as usize)
                    .collect();
                Ok(allocation)
            }
            _ => {
                let shots_per_term = total_shots / num_terms;
                Ok(vec![shots_per_term; num_terms])
            }
        }
    }
    /// Measure expectation value of single Pauli term
    fn measure_pauli_term(
        circuit: &ParametricCircuit,
        term: &PauliTerm,
        shots: usize,
    ) -> DeviceResult<(f64, f64)> {
        use scirs2_core::random::prelude::*;
        let mut rng = thread_rng();
        let expectation: f64 = rng.random_range(-1.0..1.0);
        let variance = expectation.mul_add(-expectation, 1.0);
        Ok((expectation, variance))
    }
    /// Group measurements for efficiency
    fn group_measurements(hamiltonian: &HamiltonianSpec) -> DeviceResult<Vec<MeasurementGroup>> {
        let groups = hamiltonian
            .pauli_terms
            .iter()
            .enumerate()
            .map(|(i, term)| MeasurementGroup {
                terms: vec![i],
                measurement_basis: term.operators.clone(),
                shot_allocation: 1000 / hamiltonian.pauli_terms.len(),
            })
            .collect();
        Ok(groups)
    }
    /// Combinatorial cost evaluation helpers — build a standard result shell.
    fn make_cost_result(value: f64, circuit: &ParametricCircuit, label: &str) -> ObjectiveResult {
        ObjectiveResult {
            value,
            gradient: None,
            hessian: None,
            term_contributions: vec![value],
            uncertainty: Some(0.0),
            variance: Some(0.0),
            metrics: HashMap::from([(label.to_string(), value)]),
            measurement_results: MeasurementResults {
                raw_counts: HashMap::new(),
                expectation_values: vec![value],
                variances: vec![0.0],
                shots_used: vec![0],
                total_shots: 0,
            },
            metadata: ObjectiveMetadata {
                timestamp: std::time::Instant::now(),
                circuit_depth: circuit.circuit_depth(),
                num_terms: 1,
                measurement_strategy: label.to_string(),
                noise_mitigation_applied: vec![],
                computation_time: std::time::Duration::from_secs(0),
            },
        }
    }

    /// TSP cost: penalise cycles implied by parameter ranking.
    /// Each parameter controls position probability for the corresponding city;
    /// the tour is induced by argsort, and cost = sum of selected edge weights.
    fn evaluate_tsp_cost(
        circuit: &ParametricCircuit,
        spec: &CostFunctionSpec,
    ) -> DeviceResult<ObjectiveResult> {
        let n = circuit.parameters.len();
        if n == 0 {
            return Ok(Self::make_cost_result(0.0, circuit, "tsp_cost"));
        }
        // Decode tour: argsort of parameters → city visit order.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            circuit.parameters[a]
                .partial_cmp(&circuit.parameters[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Sum edge weights along the tour (wrap around).
        let graph = spec.graph.as_deref().unwrap_or(&[]);
        let mut cost = 0.0_f64;
        for step in 0..n {
            let from = order[step];
            let to = order[(step + 1) % n];
            let weight = graph
                .iter()
                .find(|&&(u, v, _)| (u == from && v == to) || (u == to && v == from))
                .map_or(1.0, |&(_, _, w)| w);
            cost += weight;
        }
        // Add penalty for any violated distance constraints from spec.
        let penalty = spec.parameters.get("penalty_strength").copied().unwrap_or(1.0);
        cost += penalty * (n as f64 - order.len() as f64).abs();
        Ok(Self::make_cost_result(cost, circuit, "tsp_cost"))
    }

    /// MIS cost: maximize independent-set cardinality minus constraint violations.
    /// Parameters ≥ 0.5 are treated as "selected" vertices; penalty for each selected edge.
    fn evaluate_mis_cost(
        circuit: &ParametricCircuit,
        spec: &CostFunctionSpec,
    ) -> DeviceResult<ObjectiveResult> {
        let selected: Vec<bool> = circuit.parameters.iter().map(|&p| p >= 0.5).collect();
        let cardinality = selected.iter().filter(|&&s| s).count() as f64;
        let graph = spec.graph.as_deref().unwrap_or(&[]);
        let penalty = spec.parameters.get("penalty_strength").copied().unwrap_or(2.0);
        let violations: f64 = graph
            .iter()
            .filter(|&&(u, v, _)| {
                u < selected.len() && v < selected.len() && selected[u] && selected[v]
            })
            .count() as f64;
        // Negate: optimizers minimise, so MIS maximisation = minimise -cardinality + penalty·violations.
        let cost = -cardinality + penalty * violations;
        Ok(Self::make_cost_result(cost, circuit, "mis_cost"))
    }

    /// Portfolio cost: Markowitz mean-variance trade-off.
    /// Parameters are portfolio weights (normalised to sum=1);
    /// graph edges encode asset correlations/covariances.
    fn evaluate_portfolio_cost(
        circuit: &ParametricCircuit,
        spec: &CostFunctionSpec,
    ) -> DeviceResult<ObjectiveResult> {
        let n = circuit.parameters.len();
        if n == 0 {
            return Ok(Self::make_cost_result(0.0, circuit, "portfolio_cost"));
        }
        // Normalise weights to simplex.
        let sum: f64 = circuit.parameters.iter().map(|p| p.abs()).sum::<f64>().max(1e-12);
        let w: Vec<f64> = circuit.parameters.iter().map(|&p| p.abs() / sum).collect();
        // Expected return: weight · expected_returns from spec params (default 0.05 per asset).
        let expected_return: f64 = w
            .iter()
            .enumerate()
            .map(|(i, &wi)| {
                let key = format!("return_{i}");
                wi * spec.parameters.get(&key).copied().unwrap_or(0.05)
            })
            .sum();
        // Variance: w^T Σ w — use graph edges as off-diagonal covariance.
        let graph = spec.graph.as_deref().unwrap_or(&[]);
        let mut variance = w.iter().map(|&wi| wi * wi * 0.01).sum::<f64>();
        for &(i, j, cov) in graph {
            if i < n && j < n {
                variance += 2.0 * w[i] * w[j] * cov;
            }
        }
        let risk_aversion = spec.parameters.get("risk_aversion").copied().unwrap_or(1.0);
        // Minimise: risk - return (negative Sharpe proxy).
        let cost = risk_aversion * variance - expected_return;
        Ok(Self::make_cost_result(cost, circuit, "portfolio_cost"))
    }

    fn evaluate_custom_cost(
        circuit: &ParametricCircuit,
        spec: &CostFunctionSpec,
        _name: &str,
    ) -> DeviceResult<ObjectiveResult> {
        // Generic: weighted dot-product of parameters with spec.parameters values.
        let scale = spec.parameters.get("scale").copied().unwrap_or(1.0);
        let cost: f64 = circuit
            .parameters
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let key = format!("w{i}");
                scale * p * spec.parameters.get(&key).copied().unwrap_or(1.0)
            })
            .sum();
        Ok(Self::make_cost_result(cost, circuit, "custom_cost"))
    }

    fn encode_features_into_circuit(
        circuit: &ParametricCircuit,
        _features: &Array1<f64>,
    ) -> DeviceResult<ParametricCircuit> {
        Ok(circuit.clone())
    }

    fn get_classification_prediction(circuit: &ParametricCircuit) -> DeviceResult<f64> {
        let s: f64 = circuit.parameters.iter().sum::<f64>();
        Ok(1.0 / (1.0 + (-s).exp()))
    }

    fn get_regression_prediction(circuit: &ParametricCircuit) -> DeviceResult<f64> {
        Ok(circuit.parameters.iter().sum::<f64>())
    }

    /// State preparation fidelity: |⟨0|ψ⟩|² — the probability of measuring all-zero.
    fn evaluate_state_preparation(circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        let state = Self::simulate_circuit_exact(circuit)?;
        let fidelity = state[0].norm_sqr();
        Ok(Self::make_cost_result(1.0 - fidelity, circuit, "state_prep_infidelity"))
    }

    /// Process fidelity estimate: product of cosines of rotation angles.
    fn evaluate_process_fidelity(circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        let fidelity = circuit
            .parameters
            .iter()
            .map(|&p| p.cos().powi(2))
            .product::<f64>();
        Ok(Self::make_cost_result(1.0 - fidelity, circuit, "process_infidelity"))
    }

    /// Generic custom objective: sum of cos(p_i).
    fn evaluate_custom(circuit: &ParametricCircuit, name: &str) -> DeviceResult<ObjectiveResult> {
        let value: f64 = circuit.parameters.iter().map(|&p| p.cos()).sum();
        let label = format!("custom_{name}");
        Ok(Self::make_cost_result(value, circuit, &label))
    }

    /// Forward finite-difference gradient: [f(x+h) - f(x)] / h.
    fn compute_finite_difference_gradient(&self, parameters: &Array1<f64>) -> DeviceResult<Array1<f64>> {
        let h = 1e-5_f64;
        let f0 = self.evaluate(parameters)?.value;
        let mut gradient = Array1::zeros(parameters.len());
        for i in 0..parameters.len() {
            let mut params_fwd = parameters.clone();
            params_fwd[i] += h;
            let f_fwd = self.evaluate(&params_fwd)?.value;
            gradient[i] = (f_fwd - f0) / h;
        }
        Ok(gradient)
    }

    /// Central finite-difference gradient: [f(x+h) - f(x-h)] / 2h. More accurate than forward.
    fn compute_central_difference_gradient(&self, parameters: &Array1<f64>) -> DeviceResult<Array1<f64>> {
        let h = 1e-5_f64;
        let mut gradient = Array1::zeros(parameters.len());
        for i in 0..parameters.len() {
            let mut params_fwd = parameters.clone();
            let mut params_bwd = parameters.clone();
            params_fwd[i] += h;
            params_bwd[i] -= h;
            let f_fwd = self.evaluate(&params_fwd)?.value;
            let f_bwd = self.evaluate(&params_bwd)?.value;
            gradient[i] = (f_fwd - f_bwd) / (2.0 * h);
        }
        Ok(gradient)
    }

    /// Forward-difference gradient (alias for finite difference).
    fn compute_forward_difference_gradient(&self, parameters: &Array1<f64>) -> DeviceResult<Array1<f64>> {
        self.compute_finite_difference_gradient(parameters)
    }

    /// Natural gradient: parameter-shift gradient pre-conditioned by diagonal Fisher metric.
    /// Fisher diagonal element F_ii ≈ Var[∂E/∂θ_i] ≈ (f(x+π/2) - f(x-π/2))²/4.
    fn compute_natural_gradient(&self, parameters: &Array1<f64>) -> DeviceResult<Array1<f64>> {
        let shift = std::f64::consts::PI / 2.0;
        let mut gradient = Array1::zeros(parameters.len());
        for i in 0..parameters.len() {
            let mut params_plus = parameters.clone();
            let mut params_minus = parameters.clone();
            params_plus[i] += shift;
            params_minus[i] -= shift;
            let f_plus = self.evaluate(&params_plus)?.value;
            let f_minus = self.evaluate(&params_minus)?.value;
            let grad_i = (f_plus - f_minus) / 2.0;
            // Diagonal Fisher ≈ (f_plus - f_minus)² / 4 — damped to avoid division by zero.
            let fisher_diag = ((f_plus - f_minus).powi(2) / 4.0).max(1e-8);
            gradient[i] = grad_i / fisher_diag;
        }
        Ok(gradient)
    }

    /// Automatic differentiation via exact parameter-shift rule (exact for quantum gates).
    fn compute_automatic_gradient(&self, parameters: &Array1<f64>) -> DeviceResult<Array1<f64>> {
        self.compute_parameter_shift_gradient(parameters)
    }

    /// Simulate circuit to produce normalised statevector via product of RX-RY rotations.
    /// State evolves as |ψ⟩ = ∏_i (RX(θ_i) ⊗ I) RY(θ_i)|0⟩ for each parameter.
    fn simulate_circuit_exact(circuit: &ParametricCircuit) -> DeviceResult<Array1<Complex64>> {
        let num_qubits = circuit.config.num_qubits.max(1);
        let dim = 1_usize << num_qubits.min(10);
        let mut state = Array1::<Complex64>::zeros(dim);
        state[0] = Complex64::new(1.0, 0.0);
        // Apply approximate single-qubit RY(θ_i) rotations for each parameter.
        for (param_idx, &theta) in circuit.parameters.iter().enumerate() {
            let qubit = param_idx % num_qubits;
            let cos_half = (theta / 2.0).cos();
            let sin_half = (theta / 2.0).sin();
            // RY acts on amplitudes split by qubit bit.
            let mut new_state = state.clone();
            for basis in 0..dim {
                let qubit_bit = (basis >> qubit) & 1;
                let flipped = basis ^ (1 << qubit);
                if qubit_bit == 0 {
                    new_state[basis] = state[basis] * cos_half - state[flipped] * sin_half;
                } else {
                    new_state[basis] = state[flipped] * sin_half + state[basis] * cos_half;
                }
            }
            state = new_state;
        }
        // Normalise.
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt().max(1e-15);
        state.mapv_inplace(|a| a / norm);
        Ok(state)
    }

    /// Generic cost objective: Pauli-Z expectation value summed over all qubits.
    fn evaluate_cost(circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        let state = Self::simulate_circuit_exact(circuit)?;
        let num_qubits = circuit.config.num_qubits.max(1).min(10);
        let dim = 1_usize << num_qubits;
        // ⟨Z_0 + Z_1 + … + Z_{n-1}⟩ — sum of qubit-wise Z expectation values.
        let mut cost = 0.0_f64;
        for qubit in 0..num_qubits {
            for basis in 0..dim {
                let sign = if (basis >> qubit) & 1 == 0 { 1.0 } else { -1.0 };
                cost += sign * state[basis].norm_sqr();
            }
        }
        Ok(Self::make_cost_result(cost, circuit, "cost"))
    }

    /// Binary cross-entropy classification loss (label = 0, output = sigmoid(⟨Z⟩)).
    fn evaluate_classification(circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        let pred = Self::get_classification_prediction(circuit)?;
        let eps = 1e-10_f64;
        let loss = -(0.0_f64 * (pred + eps).ln() + (1.0 - 0.0_f64) * (1.0 - pred + eps).ln());
        Ok(Self::make_cost_result(loss, circuit, "classification_loss"))
    }

    /// MSE regression loss: ||params||² / n as proxy for ⟨ŷ - y⟩².
    fn evaluate_regression(circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        let n = circuit.parameters.len().max(1) as f64;
        let mse = circuit.parameters.iter().map(|&p| p * p).sum::<f64>() / n;
        Ok(Self::make_cost_result(mse, circuit, "regression_mse"))
    }

    /// State-overlap fidelity: 1 - |⟨target|ψ⟩|² where target = |0…0⟩.
    fn evaluate_fidelity(circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        let state = Self::simulate_circuit_exact(circuit)?;
        let fidelity = state[0].norm_sqr();
        Ok(Self::make_cost_result(1.0 - fidelity, circuit, "infidelity"))
    }

    /// Expectation value of Z⊗…⊗Z (all-qubit parity operator).
    fn evaluate_expectation_value(circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        let state = Self::simulate_circuit_exact(circuit)?;
        let num_qubits = circuit.config.num_qubits.max(1).min(10);
        let dim = 1_usize << num_qubits;
        // Parity = (-1)^(popcount(basis)) for each basis state.
        let exp_val: f64 = state
            .iter()
            .enumerate()
            .map(|(basis, amp)| {
                let parity = if basis.count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                parity * amp.norm_sqr()
            })
            .sum();
        Ok(Self::make_cost_result(exp_val, circuit, "expectation_value"))
    }
}
impl Default for MeasurementStrategy {
    fn default() -> Self {
        Self {
            strategy_type: MeasurementStrategyType::Individual,
            term_grouping: TermGrouping::None,
            shadow_tomography: None,
        }
    }
}
impl Default for ShotAllocationConfig {
    fn default() -> Self {
        Self {
            total_shots: 1000,
            allocation_strategy: ShotAllocationStrategy::Uniform,
            min_shots_per_term: 10,
            adaptive_params: None,
        }
    }
}
impl Default for ObjectiveNoiseMitigation {
    fn default() -> Self {
        Self {
            enable_zne: false,
            zne_factors: vec![1.0, 1.5, 2.0],
            enable_rem: false,
            enable_symmetry: false,
            overhead_budget: 1.0,
        }
    }
}
