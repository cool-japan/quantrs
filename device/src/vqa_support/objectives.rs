//! Objective function definitions and evaluation for VQA
//!
//! This module provides objective functions commonly used in
//! variational quantum algorithms with comprehensive evaluation strategies.

use crate::DeviceResult;
use super::circuits::{ParametricCircuit, GateType};
use super::config::{VQAAlgorithmType, GradientMethod};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::sync::Arc;
use num_complex::Complex64;

// SciRS2 imports for numerical computation
// TODO: scirs2_core crate not available yet
// #[cfg(feature = "scirs2")]
// use scirs2_core::{
//     linalg::{Matrix, Vector},
//     quantum::{PauliOperator, ExpectationValue},
//     statistics::sampling,
// };

#[cfg(not(feature = "scirs2"))]
mod fallback_scirs2 {
    use ndarray::{Array1, Array2};
    use num_complex::Complex64;
    
    pub struct Matrix(pub Array2<Complex64>);
    pub struct Vector(pub Array1<Complex64>);
    pub struct PauliOperator {
        pub coefficients: Array1<f64>,
        pub terms: Vec<String>,
    }
    
    impl Matrix {
        pub fn new(data: Array2<Complex64>) -> Self { Self(data) }
    }
    
    impl Vector {
        pub fn new(data: Array1<Complex64>) -> Self { Self(data) }
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
        method: &GradientMethod
    ) -> DeviceResult<Array1<f64>>;
    
    /// Estimate computational cost for given parameters
    fn estimate_cost(&self, parameters: &Array1<f64>) -> usize;
    
    /// Get parameter bounds
    fn parameter_bounds(&self) -> Option<Vec<(f64, f64)>>;
    
    /// Check if objective supports batched evaluation
    fn supports_batch_evaluation(&self) -> bool { false }
    
    /// Batch evaluate multiple parameter sets (if supported)
    fn batch_evaluate(&self, parameter_sets: &[Array1<f64>]) -> DeviceResult<Vec<ObjectiveResult>> {
        parameter_sets.iter()
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
        
        // Update circuit parameters
        let mut circuit = (*self.circuit).clone();
        circuit.set_parameters(parameters.to_vec())?;
        
        let result = match &self.config.objective_type {
            ObjectiveType::Energy => self.evaluate_energy(&circuit),
            ObjectiveType::Cost => self.evaluate_cost(&circuit),
            ObjectiveType::Classification => self.evaluate_classification(&circuit),
            ObjectiveType::Regression => self.evaluate_regression(&circuit),
            ObjectiveType::Fidelity => self.evaluate_fidelity(&circuit),
            ObjectiveType::ExpectationValue => self.evaluate_expectation_value(&circuit),
            ObjectiveType::StatePreparation => self.evaluate_state_preparation(&circuit),
            ObjectiveType::ProcessFidelity => self.evaluate_process_fidelity(&circuit),
            ObjectiveType::Custom(name) => self.evaluate_custom(&circuit, name),
        };
        
        let mut objective_result = result?;
        
        // Apply regularization
        objective_result.value = self.apply_regularization(objective_result.value, parameters);
        
        // Add metadata
        objective_result.metadata.computation_time = start_time.elapsed();
        objective_result.metadata.timestamp = start_time;
        objective_result.metadata.circuit_depth = circuit.circuit_depth();
        
        Ok(objective_result)
    }
    
    /// Compute gradient with specified method
    fn compute_gradient_with_method(
        &self, 
        parameters: &Array1<f64>, 
        method: &GradientMethod
    ) -> DeviceResult<Array1<f64>> {
        match method {
            GradientMethod::ParameterShift => self.compute_parameter_shift_gradient(parameters),
            GradientMethod::FiniteDifference => self.compute_finite_difference_gradient(parameters),
            GradientMethod::CentralDifference => self.compute_central_difference_gradient(parameters),
            GradientMethod::ForwardDifference => self.compute_forward_difference_gradient(parameters),
            GradientMethod::NaturalGradient => self.compute_natural_gradient(parameters),
            GradientMethod::AutomaticDifferentiation => self.compute_automatic_gradient(parameters),
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
        
        // Cost heuristic: circuit complexity * measurement complexity
        let circuit_cost = circuit_depth * (1 << num_qubits.min(10)); // Cap exponential growth
        let measurement_cost = num_terms * self.config.shot_allocation.total_shots;
        
        circuit_cost + measurement_cost
    }
    
    /// Get parameter bounds
    fn parameter_bounds(&self) -> Option<Vec<(f64, f64)>> {
        Some(self.circuit.bounds.clone())
    }
    
    /// Check batch evaluation support
    fn supports_batch_evaluation(&self) -> bool {
        matches!(self.backend, ObjectiveBackend::SciRS2Exact | ObjectiveBackend::Mock)
    }
}

impl ObjectiveEvaluator {
    /// Create new objective evaluator
    pub fn new(
        config: ObjectiveConfig, 
        circuit: ParametricCircuit,
        backend: ObjectiveBackend
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
            self.measurement_groups = Some(self.group_measurements(hamiltonian_spec)?);
        }
        Ok(self)
    }
    
    /// Evaluate energy objective (VQE)
    fn evaluate_energy(&self, circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        let hamiltonian = self.config.hamiltonian.as_ref()
            .ok_or_else(|| crate::DeviceError::InvalidInput(
                "Hamiltonian specification required for energy evaluation".to_string()
            ))?;
            
        match &self.backend {
            ObjectiveBackend::SciRS2Exact => self.evaluate_energy_exact(circuit, hamiltonian),
            ObjectiveBackend::QuantRS2Simulator => self.evaluate_energy_sampling(circuit, hamiltonian),
            ObjectiveBackend::Hardware(_) => self.evaluate_energy_hardware(circuit, hamiltonian),
            ObjectiveBackend::Mock => self.evaluate_energy_mock(circuit, hamiltonian),
        }
    }
    
    /// Exact energy evaluation with SciRS2
    fn evaluate_energy_exact(
        &self, 
        circuit: &ParametricCircuit, 
        hamiltonian: &HamiltonianSpec
    ) -> DeviceResult<ObjectiveResult> {
        #[cfg(feature = "scirs2")]
        {
            // Generate quantum state from circuit
            let state_vector = self.simulate_circuit_exact(circuit)?;
            
            // Compute expectation value H|ψ⟩ with SciRS2
            let hamiltonian_matrix = self.get_or_build_hamiltonian(hamiltonian)?;
            let energy = self.compute_expectation_value_exact(&state_vector, &hamiltonian_matrix)?;
            
            let mut measurement_results = MeasurementResults {
                raw_counts: HashMap::new(),
                expectation_values: vec![energy],
                variances: vec![0.0], // Exact computation has no variance
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
                metrics: [("exact_evaluation".to_string(), 1.0)].into_iter().collect(),
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
            // Fallback implementation
            self.evaluate_energy_mock(circuit, hamiltonian)
        }
    }
    
    /// Sampling-based energy evaluation
    fn evaluate_energy_sampling(
        &self, 
        circuit: &ParametricCircuit, 
        hamiltonian: &HamiltonianSpec
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
        
        // Allocate shots to terms
        let shot_allocation = self.allocate_shots_to_terms(hamiltonian)?;
        
        // Measure each Pauli term
        for (term_idx, term) in hamiltonian.pauli_terms.iter().enumerate() {
            let shots = shot_allocation[term_idx];
            let (expectation, variance) = self.measure_pauli_term(circuit, term, shots)?;
            
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
            metrics: [("sampling_evaluation".to_string(), 1.0)].into_iter().collect(),
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
        hamiltonian: &HamiltonianSpec
    ) -> DeviceResult<ObjectiveResult> {
        // For now, use sampling-based evaluation as placeholder
        // In practice, this would interface with actual quantum hardware
        self.evaluate_energy_sampling(circuit, hamiltonian)
    }
    
    /// Mock energy evaluation for testing
    fn evaluate_energy_mock(
        &self, 
        _circuit: &ParametricCircuit, 
        hamiltonian: &HamiltonianSpec
    ) -> DeviceResult<ObjectiveResult> {
        // Simple mock that returns sum of squares with noise
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
        
        let energy = hamiltonian.pauli_terms.iter()
            .map(|term| term.coefficient.re * rng.gen_range(-1.0..1.0))
            .sum::<f64>();
            
        let variance: f64 = 0.01; // Mock variance
        
        Ok(ObjectiveResult {
            value: energy,
            gradient: None,
            hessian: None,
            term_contributions: vec![energy],
            uncertainty: Some(variance.sqrt()),
            variance: Some(variance),
            metrics: [("mock_evaluation".to_string(), 1.0)].into_iter().collect(),
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
        let l1_penalty = self.config.regularization.l1_coeff * 
            parameters.iter().map(|&x| x.abs()).sum::<f64>();
        let l2_penalty = self.config.regularization.l2_coeff * 
            parameters.iter().map(|&x| x * x).sum::<f64>();
            
        value + l1_penalty + l2_penalty
    }
    
    /// Compute parameter shift gradient
    fn compute_parameter_shift_gradient(&self, parameters: &Array1<f64>) -> DeviceResult<Array1<f64>> {
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
            let term_matrix = self.build_pauli_term_matrix(term, spec.num_qubits)?;
            matrix = matrix + term_matrix;
        }
        
        Ok(HamiltonianMatrix {
            matrix,
            eigenvalues: None,
            eigenvectors: None,
        })
    }
    
    /// Build matrix for single Pauli term
    fn build_pauli_term_matrix(&self, term: &PauliTerm, num_qubits: usize) -> DeviceResult<Array2<Complex64>> {
        // Implementation would build the tensor product of Pauli matrices
        // For now, return identity as placeholder
        let dim = 1 << num_qubits;
        let mut matrix = Array2::zeros((dim, dim));
        
        // Add diagonal elements (simplified)
        for i in 0..dim {
            matrix[[i, i]] = term.coefficient;
        }
        
        Ok(matrix)
    }
    
    /// Get cached Hamiltonian or build it
    fn get_or_build_hamiltonian(&self, spec: &HamiltonianSpec) -> DeviceResult<&HamiltonianMatrix> {
        // In practice, would implement proper caching logic
        // For now, return error indicating need for caching
        Err(crate::DeviceError::InvalidInput(
            "Hamiltonian caching not yet implemented".to_string()
        ))
    }
    
    /// Compute exact expectation value
    fn compute_expectation_value_exact(
        &self, 
        state: &Array1<Complex64>, 
        hamiltonian: &HamiltonianMatrix
    ) -> DeviceResult<f64> {
        // ⟨ψ|H|ψ⟩ = ψ* H ψ
        let h_psi = hamiltonian.matrix.dot(state);
        let expectation = state.iter()
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
            },
            ShotAllocationStrategy::ProportionalToCoeff => {
                let coeffs: Vec<f64> = hamiltonian.pauli_terms.iter()
                    .map(|term| term.coefficient.norm())
                    .collect();
                let total_coeff: f64 = coeffs.iter().sum();
                
                let allocation: Vec<usize> = coeffs.iter()
                    .map(|&coeff| ((coeff / total_coeff) * total_shots as f64) as usize)
                    .collect();
                    
                Ok(allocation)
            },
            _ => {
                // Default to uniform for other strategies
                let shots_per_term = total_shots / num_terms;
                Ok(vec![shots_per_term; num_terms])
            }
        }
    }
    
    /// Measure expectation value of single Pauli term
    fn measure_pauli_term(
        &self, 
        circuit: &ParametricCircuit, 
        term: &PauliTerm, 
        shots: usize
    ) -> DeviceResult<(f64, f64)> {
        // Simulate measurement of Pauli term
        // For now, return random values as placeholder
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
        
        let expectation: f64 = rng.gen_range(-1.0..1.0);
        let variance = 1.0 - expectation.powi(2); // Var = 1 - ⟨P⟩² for Pauli operators
        
        Ok((expectation, variance))
    }
    
    /// Group measurements for efficiency
    fn group_measurements(&self, hamiltonian: &HamiltonianSpec) -> DeviceResult<Vec<MeasurementGroup>> {
        // Simplified grouping - each term in its own group
        let groups = hamiltonian.pauli_terms.iter().enumerate()
            .map(|(i, term)| MeasurementGroup {
                terms: vec![i],
                measurement_basis: term.operators.clone(),
                shot_allocation: 1000 / hamiltonian.pauli_terms.len(),
            })
            .collect();
            
        Ok(groups)
    }
    
    /// Additional placeholder methods for evaluation types
    fn evaluate_tsp_cost(&self, _circuit: &ParametricCircuit, _spec: &CostFunctionSpec) -> DeviceResult<ObjectiveResult> {
        Err(crate::DeviceError::NotImplemented("TSP cost evaluation not yet implemented".to_string()))
    }
    
    fn evaluate_mis_cost(&self, _circuit: &ParametricCircuit, _spec: &CostFunctionSpec) -> DeviceResult<ObjectiveResult> {
        Err(crate::DeviceError::NotImplemented("MIS cost evaluation not yet implemented".to_string()))
    }
    
    fn evaluate_portfolio_cost(&self, _circuit: &ParametricCircuit, _spec: &CostFunctionSpec) -> DeviceResult<ObjectiveResult> {
        Err(crate::DeviceError::NotImplemented("Portfolio cost evaluation not yet implemented".to_string()))
    }
    
    fn evaluate_custom_cost(&self, _circuit: &ParametricCircuit, _spec: &CostFunctionSpec, _name: &str) -> DeviceResult<ObjectiveResult> {
        Err(crate::DeviceError::NotImplemented("Custom cost evaluation not yet implemented".to_string()))
    }
    
    fn encode_features_into_circuit(
        &self, 
        circuit: &ParametricCircuit, 
        _features: &Array1<f64>
    ) -> DeviceResult<ParametricCircuit> {
        // Placeholder - return original circuit
        Ok(circuit.clone())
    }
    
    fn get_classification_prediction(&self, _circuit: &ParametricCircuit) -> DeviceResult<f64> {
        // Placeholder prediction
        use rand::prelude::*;
        Ok(rand::thread_rng().gen_range(0.0..1.0))
    }
    
    fn get_regression_prediction(&self, _circuit: &ParametricCircuit) -> DeviceResult<f64> {
        // Placeholder prediction
        use rand::prelude::*;
        Ok(rand::thread_rng().gen_range(-1.0..1.0))
    }
    
    /// Missing method implementations
    fn evaluate_state_preparation(&self, _circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        Err(crate::DeviceError::NotImplemented("State preparation evaluation not yet implemented".to_string()))
    }
    
    fn evaluate_process_fidelity(&self, _circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        Err(crate::DeviceError::NotImplemented("Process fidelity evaluation not yet implemented".to_string()))
    }
    
    fn evaluate_custom(&self, _circuit: &ParametricCircuit, _name: &str) -> DeviceResult<ObjectiveResult> {
        Err(crate::DeviceError::NotImplemented("Custom evaluation not yet implemented".to_string()))
    }
    
    fn compute_finite_difference_gradient(&self, _parameters: &Array1<f64>) -> DeviceResult<Array1<f64>> {
        Err(crate::DeviceError::NotImplemented("Finite difference gradient not yet implemented".to_string()))
    }
    
    fn compute_central_difference_gradient(&self, _parameters: &Array1<f64>) -> DeviceResult<Array1<f64>> {
        Err(crate::DeviceError::NotImplemented("Central difference gradient not yet implemented".to_string()))
    }
    
    fn compute_forward_difference_gradient(&self, _parameters: &Array1<f64>) -> DeviceResult<Array1<f64>> {
        Err(crate::DeviceError::NotImplemented("Forward difference gradient not yet implemented".to_string()))
    }
    
    fn compute_natural_gradient(&self, _parameters: &Array1<f64>) -> DeviceResult<Array1<f64>> {
        Err(crate::DeviceError::NotImplemented("Natural gradient not yet implemented".to_string()))
    }
    
    fn compute_automatic_gradient(&self, _parameters: &Array1<f64>) -> DeviceResult<Array1<f64>> {
        Err(crate::DeviceError::NotImplemented("Automatic gradient not yet implemented".to_string()))
    }
    
    fn simulate_circuit_exact(&self, _circuit: &ParametricCircuit) -> DeviceResult<Array1<Complex64>> {
        Err(crate::DeviceError::NotImplemented("Exact circuit simulation not yet implemented".to_string()))
    }
    
    fn evaluate_cost(&self, _circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        Err(crate::DeviceError::NotImplemented("Cost evaluation not yet implemented".to_string()))
    }
    
    fn evaluate_classification(&self, _circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        Err(crate::DeviceError::NotImplemented("Classification evaluation not yet implemented".to_string()))
    }
    
    fn evaluate_regression(&self, _circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        Err(crate::DeviceError::NotImplemented("Regression evaluation not yet implemented".to_string()))
    }
    
    fn evaluate_fidelity(&self, _circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        Err(crate::DeviceError::NotImplemented("Fidelity evaluation not yet implemented".to_string()))
    }
    
    fn evaluate_expectation_value(&self, _circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
        Err(crate::DeviceError::NotImplemented("Expectation value evaluation not yet implemented".to_string()))
    }
}

// Default implementations for missing types
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