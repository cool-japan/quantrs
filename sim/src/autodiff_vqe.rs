//! Automatic differentiation for Variational Quantum Eigensolver (VQE).
//!
//! This module implements automatic differentiation techniques specifically designed
//! for variational quantum algorithms, including parameter-shift rule, finite differences,
//! and optimization strategies for VQE.

use crate::error::{Result, SimulatorError};
use crate::pauli::{PauliOperatorSum, PauliString};
use quantrs2_core::gate::GateOp;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::f64::consts::PI;

#[cfg(feature = "optimize")]
use crate::optirs_integration::{OptiRSConfig, OptiRSQuantumOptimizer};

/// Gradient computation method
#[derive(Debug, Clone, Copy)]
pub enum GradientMethod {
    /// Parameter-shift rule (exact for quantum gates)
    ParameterShift,
    /// Finite differences
    FiniteDifference { step_size: f64 },
    /// Simultaneous perturbation stochastic approximation
    SPSA { step_size: f64 },
}

/// Automatic differentiation context for tracking gradients
#[derive(Debug, Clone)]
pub struct AutoDiffContext {
    /// Parameter values
    pub parameters: Vec<f64>,
    /// Parameter names/indices
    pub parameter_names: Vec<String>,
    /// Gradient computation method
    pub method: GradientMethod,
    /// Current gradients
    pub gradients: Vec<f64>,
    /// Gradient computation count
    pub grad_evaluations: usize,
    /// Function evaluation count
    pub func_evaluations: usize,
}

impl AutoDiffContext {
    /// Create new autodiff context
    #[must_use]
    pub fn new(parameters: Vec<f64>, method: GradientMethod) -> Self {
        let num_params = parameters.len();
        Self {
            parameters,
            parameter_names: (0..num_params).map(|i| format!("θ{i}")).collect(),
            method,
            gradients: vec![0.0; num_params],
            grad_evaluations: 0,
            func_evaluations: 0,
        }
    }

    /// Set parameter names
    #[must_use]
    pub fn with_parameter_names(mut self, names: Vec<String>) -> Self {
        assert_eq!(names.len(), self.parameters.len());
        self.parameter_names = names;
        self
    }

    /// Update parameters
    pub fn update_parameters(&mut self, new_params: Vec<f64>) {
        assert_eq!(new_params.len(), self.parameters.len());
        self.parameters = new_params;
    }

    /// Get parameter by name
    #[must_use]
    pub fn get_parameter(&self, name: &str) -> Option<f64> {
        self.parameter_names
            .iter()
            .position(|n| n == name)
            .map(|i| self.parameters[i])
    }

    /// Set parameter by name
    pub fn set_parameter(&mut self, name: &str, value: f64) -> Result<()> {
        if let Some(i) = self.parameter_names.iter().position(|n| n == name) {
            self.parameters[i] = value;
            Ok(())
        } else {
            Err(SimulatorError::InvalidInput(format!(
                "Parameter '{name}' not found"
            )))
        }
    }
}

/// Parametric quantum gate that supports automatic differentiation
pub trait ParametricGate: Send + Sync {
    /// Get gate name
    fn name(&self) -> &str;

    /// Get qubits this gate acts on
    fn qubits(&self) -> Vec<usize>;

    /// Get parameter indices this gate depends on
    fn parameter_indices(&self) -> Vec<usize>;

    /// Evaluate gate matrix given parameter values
    fn matrix(&self, params: &[f64]) -> Result<Array2<Complex64>>;

    /// Compute gradient of gate matrix with respect to each parameter
    fn gradient(&self, params: &[f64], param_idx: usize) -> Result<Array2<Complex64>>;

    /// Apply parameter-shift rule for this gate
    fn parameter_shift_gradient(
        &self,
        params: &[f64],
        param_idx: usize,
    ) -> Result<(Array2<Complex64>, Array2<Complex64>)> {
        let shift = PI / 2.0;
        let mut params_plus = params.to_vec();
        let mut params_minus = params.to_vec();

        if param_idx < params.len() {
            params_plus[param_idx] += shift;
            params_minus[param_idx] -= shift;
        }

        let matrix_plus = self.matrix(&params_plus)?;
        let matrix_minus = self.matrix(&params_minus)?;

        Ok((matrix_plus, matrix_minus))
    }
}

/// Parametric rotation gates
pub struct ParametricRX {
    pub qubit: usize,
    pub param_idx: usize,
}

impl ParametricGate for ParametricRX {
    fn name(&self) -> &'static str {
        "RX"
    }

    fn qubits(&self) -> Vec<usize> {
        vec![self.qubit]
    }

    fn parameter_indices(&self) -> Vec<usize> {
        vec![self.param_idx]
    }

    fn matrix(&self, params: &[f64]) -> Result<Array2<Complex64>> {
        let theta = params[self.param_idx];
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();

        Ok(scirs2_core::ndarray::array![
            [Complex64::new(cos_half, 0.), Complex64::new(0., -sin_half)],
            [Complex64::new(0., -sin_half), Complex64::new(cos_half, 0.)]
        ])
    }

    fn gradient(&self, params: &[f64], param_idx: usize) -> Result<Array2<Complex64>> {
        if param_idx != self.param_idx {
            return Ok(Array2::zeros((2, 2)));
        }

        let theta = params[self.param_idx];
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();

        // d/dθ RX(θ) = -i/2 * X * RX(θ)
        Ok(scirs2_core::ndarray::array![
            [
                Complex64::new(-sin_half / 2.0, 0.),
                Complex64::new(0., -cos_half / 2.0)
            ],
            [
                Complex64::new(0., -cos_half / 2.0),
                Complex64::new(-sin_half / 2.0, 0.)
            ]
        ])
    }
}

pub struct ParametricRY {
    pub qubit: usize,
    pub param_idx: usize,
}

impl ParametricGate for ParametricRY {
    fn name(&self) -> &'static str {
        "RY"
    }

    fn qubits(&self) -> Vec<usize> {
        vec![self.qubit]
    }

    fn parameter_indices(&self) -> Vec<usize> {
        vec![self.param_idx]
    }

    fn matrix(&self, params: &[f64]) -> Result<Array2<Complex64>> {
        let theta = params[self.param_idx];
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();

        Ok(scirs2_core::ndarray::array![
            [Complex64::new(cos_half, 0.), Complex64::new(-sin_half, 0.)],
            [Complex64::new(sin_half, 0.), Complex64::new(cos_half, 0.)]
        ])
    }

    fn gradient(&self, params: &[f64], param_idx: usize) -> Result<Array2<Complex64>> {
        if param_idx != self.param_idx {
            return Ok(Array2::zeros((2, 2)));
        }

        let theta = params[self.param_idx];
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();

        Ok(scirs2_core::ndarray::array![
            [
                Complex64::new(-sin_half / 2.0, 0.),
                Complex64::new(-cos_half / 2.0, 0.)
            ],
            [
                Complex64::new(cos_half / 2.0, 0.),
                Complex64::new(-sin_half / 2.0, 0.)
            ]
        ])
    }
}

/// Non-parametric CNOT gate, used for the entangling layers of ansätze such
/// as [`ansatze::hardware_efficient`]. It has no free parameters (an empty
/// [`ParametricGate::parameter_indices`]), so parameter-shift/finite-
/// difference/SPSA gradient computation simply never varies it, and its own
/// gradient with respect to any parameter is the zero matrix.
pub struct ParametricCNOT {
    pub control: usize,
    pub target: usize,
}

impl ParametricGate for ParametricCNOT {
    fn name(&self) -> &'static str {
        "CNOT"
    }

    fn qubits(&self) -> Vec<usize> {
        vec![self.control, self.target]
    }

    fn parameter_indices(&self) -> Vec<usize> {
        Vec::new()
    }

    fn matrix(&self, _params: &[f64]) -> Result<Array2<Complex64>> {
        // Basis ordering |control, target> with `control` the high local
        // bit, matching `apply_two_qubit_matrix`'s bit-indexing convention.
        Ok(scirs2_core::ndarray::array![
            [
                Complex64::new(1., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.)
            ],
            [
                Complex64::new(0., 0.),
                Complex64::new(1., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.)
            ],
            [
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(1., 0.)
            ],
            [
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(1., 0.),
                Complex64::new(0., 0.)
            ]
        ])
    }

    fn gradient(&self, _params: &[f64], _param_idx: usize) -> Result<Array2<Complex64>> {
        // CNOT has no parameters, so its gradient w.r.t. any parameter is
        // identically zero.
        Ok(Array2::zeros((4, 4)))
    }
}

pub struct ParametricRZ {
    pub qubit: usize,
    pub param_idx: usize,
}

impl ParametricGate for ParametricRZ {
    fn name(&self) -> &'static str {
        "RZ"
    }

    fn qubits(&self) -> Vec<usize> {
        vec![self.qubit]
    }

    fn parameter_indices(&self) -> Vec<usize> {
        vec![self.param_idx]
    }

    fn matrix(&self, params: &[f64]) -> Result<Array2<Complex64>> {
        let theta = params[self.param_idx];
        let exp_pos = Complex64::from_polar(1.0, theta / 2.0);
        let exp_neg = Complex64::from_polar(1.0, -theta / 2.0);

        Ok(scirs2_core::ndarray::array![
            [exp_neg, Complex64::new(0., 0.)],
            [Complex64::new(0., 0.), exp_pos]
        ])
    }

    fn gradient(&self, params: &[f64], param_idx: usize) -> Result<Array2<Complex64>> {
        if param_idx != self.param_idx {
            return Ok(Array2::zeros((2, 2)));
        }

        let theta = params[self.param_idx];
        let exp_pos = Complex64::from_polar(1.0, theta / 2.0);
        let exp_neg = Complex64::from_polar(1.0, -theta / 2.0);

        Ok(scirs2_core::ndarray::array![
            [exp_neg * Complex64::new(0., -0.5), Complex64::new(0., 0.)],
            [Complex64::new(0., 0.), exp_pos * Complex64::new(0., 0.5)]
        ])
    }
}

/// Parametric quantum circuit for VQE
pub struct ParametricCircuit {
    /// Sequence of parametric gates
    pub gates: Vec<Box<dyn ParametricGate>>,
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of parameters
    pub num_parameters: usize,
}

impl ParametricCircuit {
    /// Create new parametric circuit
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
            num_parameters: 0,
        }
    }

    /// Add a parametric gate
    pub fn add_gate(&mut self, gate: Box<dyn ParametricGate>) {
        // Update parameter count
        for &param_idx in &gate.parameter_indices() {
            self.num_parameters = self.num_parameters.max(param_idx + 1);
        }
        self.gates.push(gate);
    }

    /// Add RX gate
    pub fn rx(&mut self, qubit: usize, param_idx: usize) {
        self.add_gate(Box::new(ParametricRX { qubit, param_idx }));
    }

    /// Add RY gate
    pub fn ry(&mut self, qubit: usize, param_idx: usize) {
        self.add_gate(Box::new(ParametricRY { qubit, param_idx }));
    }

    /// Add RZ gate
    pub fn rz(&mut self, qubit: usize, param_idx: usize) {
        self.add_gate(Box::new(ParametricRZ { qubit, param_idx }));
    }

    /// Add a (non-parametric) CNOT gate, used to build entangling layers.
    pub fn cnot(&mut self, control: usize, target: usize) {
        self.add_gate(Box::new(ParametricCNOT { control, target }));
    }

    /// Evaluate the circuit for the given parameters and return the final state vector.
    ///
    /// Starts from `|0...0>` and applies each parametric gate's matrix to the dense state
    /// vector in sequence. Amplitudes use the little-endian convention (qubit `q` maps to
    /// bit `q` of the basis index), matching [`compute_pauli_expectation_from_state`] so
    /// expectation values and parameter-shift gradients are computed on the genuine state
    /// produced by the circuit rather than a fixed placeholder.
    pub fn evaluate(&self, params: &[f64]) -> Result<Array1<Complex64>> {
        if params.len() != self.num_parameters {
            return Err(SimulatorError::InvalidInput(format!(
                "Expected {} parameters, got {}",
                self.num_parameters,
                params.len()
            )));
        }

        let dim = 1usize << self.num_qubits;
        let mut state = Array1::zeros(dim);
        state[0] = Complex64::new(1.0, 0.0); // |0...0>

        for gate in &self.gates {
            let matrix = gate.matrix(params)?;
            let qubits = gate.qubits();

            match qubits.as_slice() {
                [target] => {
                    apply_single_qubit_matrix(&mut state, &matrix, *target, self.num_qubits)?;
                }
                [control, target] => {
                    apply_two_qubit_matrix(
                        &mut state,
                        &matrix,
                        *control,
                        *target,
                        self.num_qubits,
                    )?;
                }
                _ => {
                    return Err(SimulatorError::UnsupportedOperation(format!(
                        "Parametric circuit evaluation supports one- and two-qubit gates, but '{}' acts on {} qubits",
                        gate.name(),
                        qubits.len()
                    )));
                }
            }
        }

        Ok(state)
    }

    /// Compute gradient of expectation value using parameter-shift rule
    pub fn gradient_expectation(
        &self,
        observable: &PauliOperatorSum,
        params: &[f64],
        method: GradientMethod,
    ) -> Result<Vec<f64>> {
        match method {
            GradientMethod::ParameterShift => self.parameter_shift_gradient(observable, params),
            GradientMethod::FiniteDifference { step_size } => {
                self.finite_difference_gradient(observable, params, step_size)
            }
            GradientMethod::SPSA { step_size } => self.spsa_gradient(observable, params, step_size),
        }
    }

    /// Parameter-shift rule gradient computation
    fn parameter_shift_gradient(
        &self,
        observable: &PauliOperatorSum,
        params: &[f64],
    ) -> Result<Vec<f64>> {
        let mut gradients = vec![0.0; self.num_parameters];

        // Use parameter-shift rule: ∂⟨H⟩/∂θᵢ = (⟨H⟩₊ - ⟨H⟩₋) / 2
        // where ±π/2 shifts are applied to parameter θᵢ
        for param_idx in 0..self.num_parameters {
            let shift = PI / 2.0;

            // Forward shift
            let mut params_plus = params.to_vec();
            params_plus[param_idx] += shift;
            let state_plus = self.evaluate(&params_plus)?;
            let expectation_plus = compute_expectation_value(&state_plus, observable)?;

            // Backward shift
            let mut params_minus = params.to_vec();
            params_minus[param_idx] -= shift;
            let state_minus = self.evaluate(&params_minus)?;
            let expectation_minus = compute_expectation_value(&state_minus, observable)?;

            // Gradient
            gradients[param_idx] = (expectation_plus - expectation_minus) / 2.0;
        }

        Ok(gradients)
    }

    /// Finite difference gradient computation
    fn finite_difference_gradient(
        &self,
        observable: &PauliOperatorSum,
        params: &[f64],
        step_size: f64,
    ) -> Result<Vec<f64>> {
        let mut gradients = vec![0.0; self.num_parameters];

        for param_idx in 0..self.num_parameters {
            // Forward difference
            let mut params_plus = params.to_vec();
            params_plus[param_idx] += step_size;
            let state_plus = self.evaluate(&params_plus)?;
            let expectation_plus = compute_expectation_value(&state_plus, observable)?;

            // Current value
            let state = self.evaluate(params)?;
            let expectation = compute_expectation_value(&state, observable)?;

            gradients[param_idx] = (expectation_plus - expectation) / step_size;
        }

        Ok(gradients)
    }

    /// SPSA gradient estimation
    fn spsa_gradient(
        &self,
        observable: &PauliOperatorSum,
        params: &[f64],
        step_size: f64,
    ) -> Result<Vec<f64>> {
        let mut rng = thread_rng();

        // Generate random perturbation vector
        let mut perturbation = vec![0.0; self.num_parameters];
        for p in &mut perturbation {
            *p = if rng.random::<bool>() { 1.0 } else { -1.0 };
        }

        // Two evaluations with opposite perturbations
        let mut params_plus = params.to_vec();
        let mut params_minus = params.to_vec();
        for i in 0..self.num_parameters {
            params_plus[i] += step_size * perturbation[i];
            params_minus[i] -= step_size * perturbation[i];
        }

        let state_plus = self.evaluate(&params_plus)?;
        let state_minus = self.evaluate(&params_minus)?;
        let expectation_plus = compute_expectation_value(&state_plus, observable)?;
        let expectation_minus = compute_expectation_value(&state_minus, observable)?;

        // SPSA gradient estimate
        let diff = (expectation_plus - expectation_minus) / (2.0 * step_size);
        let gradients = perturbation.iter().map(|&p| diff / p).collect();

        Ok(gradients)
    }
}

/// VQE algorithm with automatic differentiation
pub struct VQEWithAutodiff {
    /// Parametric ansatz circuit
    pub ansatz: ParametricCircuit,
    /// Hamiltonian observable
    pub hamiltonian: PauliOperatorSum,
    /// Autodiff context
    pub context: AutoDiffContext,
    /// Optimization history
    pub history: Vec<VQEIteration>,
    /// Convergence criteria
    pub convergence: ConvergenceCriteria,
}

/// Single VQE iteration data
#[derive(Clone)]
pub struct VQEIteration {
    /// Iteration number
    pub iteration: usize,
    /// Parameters at this iteration
    pub parameters: Vec<f64>,
    /// Energy expectation value
    pub energy: f64,
    /// Gradient norm
    pub gradient_norm: f64,
    /// Function evaluations so far
    pub func_evals: usize,
    /// Gradient evaluations so far
    pub grad_evals: usize,
}

/// Convergence criteria for VQE
pub struct ConvergenceCriteria {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Energy tolerance
    pub energy_tolerance: f64,
    /// Gradient norm tolerance
    pub gradient_tolerance: f64,
    /// Maximum function evaluations
    pub max_func_evals: usize,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            energy_tolerance: 1e-6,
            gradient_tolerance: 1e-6,
            max_func_evals: 10_000,
        }
    }
}

impl VQEWithAutodiff {
    /// Create new VQE instance
    #[must_use]
    pub fn new(
        ansatz: ParametricCircuit,
        hamiltonian: PauliOperatorSum,
        initial_params: Vec<f64>,
        gradient_method: GradientMethod,
    ) -> Self {
        let context = AutoDiffContext::new(initial_params, gradient_method);
        Self {
            ansatz,
            hamiltonian,
            context,
            history: Vec::new(),
            convergence: ConvergenceCriteria::default(),
        }
    }

    /// Set convergence criteria
    #[must_use]
    pub const fn with_convergence(mut self, convergence: ConvergenceCriteria) -> Self {
        self.convergence = convergence;
        self
    }

    /// Evaluate energy for current parameters
    pub fn evaluate_energy(&mut self) -> Result<f64> {
        let state = self.ansatz.evaluate(&self.context.parameters)?;
        let energy = compute_expectation_value(&state, &self.hamiltonian)?;
        self.context.func_evaluations += 1;
        Ok(energy)
    }

    /// Compute gradient for current parameters
    pub fn compute_gradient(&mut self) -> Result<Vec<f64>> {
        let gradients = self.ansatz.gradient_expectation(
            &self.hamiltonian,
            &self.context.parameters,
            self.context.method,
        )?;
        self.context.gradients.clone_from(&gradients);
        self.context.grad_evaluations += 1;
        Ok(gradients)
    }

    /// Perform one VQE optimization step using gradient descent
    pub fn step(&mut self, learning_rate: f64) -> Result<VQEIteration> {
        let energy = self.evaluate_energy()?;
        let gradients = self.compute_gradient()?;

        // Gradient descent update
        for (i, &grad) in gradients.iter().enumerate() {
            self.context.parameters[i] -= learning_rate * grad;
        }

        let gradient_norm = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();

        let iteration = VQEIteration {
            iteration: self.history.len(),
            parameters: self.context.parameters.clone(),
            energy,
            gradient_norm,
            func_evals: self.context.func_evaluations,
            grad_evals: self.context.grad_evaluations,
        };

        self.history.push(iteration.clone());
        Ok(iteration)
    }

    /// Run VQE optimization until convergence
    pub fn optimize(&mut self, learning_rate: f64) -> Result<VQEResult> {
        while !self.is_converged()? {
            let iteration = self.step(learning_rate)?;

            if iteration.iteration >= self.convergence.max_iterations {
                break;
            }
            if iteration.func_evals >= self.convergence.max_func_evals {
                break;
            }
        }

        let final_iteration = self.history.last().ok_or_else(|| {
            SimulatorError::InvalidOperation("VQE optimization produced no iterations".to_string())
        })?;
        Ok(VQEResult {
            optimal_parameters: final_iteration.parameters.clone(),
            optimal_energy: final_iteration.energy,
            iterations: self.history.len(),
            converged: self.is_converged()?,
            history: self.history.clone(),
        })
    }

    /// Check convergence
    fn is_converged(&self) -> Result<bool> {
        if self.history.len() < 2 {
            return Ok(false);
        }

        let current = &self.history[self.history.len() - 1];
        let previous = &self.history[self.history.len() - 2];

        let energy_converged =
            (current.energy - previous.energy).abs() < self.convergence.energy_tolerance;
        let gradient_converged = current.gradient_norm < self.convergence.gradient_tolerance;

        Ok(energy_converged && gradient_converged)
    }

    /// Run VQE optimization using `OptiRS` optimizers (Adam, SGD, `RMSprop`, etc.)
    ///
    /// This method provides state-of-the-art optimization using `OptiRS`'s advanced
    /// machine learning optimizers, which typically converge faster and more robustly
    /// than basic gradient descent.
    ///
    /// # Arguments
    /// * `config` - `OptiRS` optimizer configuration
    ///
    /// # Returns
    /// * `VQEResult` - Optimization result with optimal parameters and energy
    ///
    /// # Example
    /// ```ignore
    /// use quantrs2_sim::autodiff_vqe::*;
    /// use quantrs2_sim::optirs_integration::*;
    ///
    /// let mut vqe = VQEWithAutodiff::new(...);
    /// let config = OptiRSConfig {
    ///     optimizer_type: OptiRSOptimizerType::Adam,
    ///     learning_rate: 0.01,
    ///     ..Default::default()
    /// };
    /// let result = vqe.optimize_with_optirs(config)?;
    /// ```
    #[cfg(feature = "optimize")]
    pub fn optimize_with_optirs(&mut self, config: OptiRSConfig) -> Result<VQEResult> {
        use std::time::Instant;

        let start_time = Instant::now();
        let mut optimizer = OptiRSQuantumOptimizer::new(config)?;

        while !self.is_converged()? && !optimizer.has_converged() {
            // Evaluate energy and gradients
            let energy = self.evaluate_energy()?;
            let gradients = self.compute_gradient()?;

            // OptiRS optimization step
            let new_params =
                optimizer.optimize_step(&self.context.parameters, &gradients, energy)?;

            // Update parameters
            self.context.parameters = new_params;

            // Record iteration
            let gradient_norm = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
            let iteration = VQEIteration {
                iteration: self.history.len(),
                parameters: self.context.parameters.clone(),
                energy,
                gradient_norm,
                func_evals: self.context.func_evaluations,
                grad_evals: self.context.grad_evaluations,
            };
            self.history.push(iteration);

            // Check maximum iterations (use VQE's convergence criteria)
            if self.history.len() >= self.convergence.max_iterations {
                break;
            }
            if self.context.func_evaluations >= self.convergence.max_func_evals {
                break;
            }
        }

        let _optimization_time = start_time.elapsed();
        let final_iteration = self.history.last().ok_or_else(|| {
            SimulatorError::InvalidOperation(
                "VQE optimization with OptiRS produced no iterations".to_string(),
            )
        })?;

        Ok(VQEResult {
            optimal_parameters: final_iteration.parameters.clone(),
            optimal_energy: final_iteration.energy,
            iterations: self.history.len(),
            converged: self.is_converged()?,
            history: self.history.clone(),
        })
    }
}

/// VQE optimization result
pub struct VQEResult {
    /// Optimal parameters found
    pub optimal_parameters: Vec<f64>,
    /// Optimal energy value
    pub optimal_energy: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Full optimization history
    pub history: Vec<VQEIteration>,
}

// Helper functions

/// Apply a single-qubit gate matrix to a dense state vector in place.
///
/// `state` is indexed in the little-endian convention where bit `qubit` selects the
/// physical value of that qubit. The 2x2 `gate` matrix is applied to every pair of
/// amplitudes that differ only in qubit `qubit`.
fn apply_single_qubit_matrix(
    state: &mut Array1<Complex64>,
    gate: &Array2<Complex64>,
    qubit: usize,
    num_qubits: usize,
) -> Result<()> {
    if qubit >= num_qubits {
        return Err(SimulatorError::InvalidInput(format!(
            "Gate targets qubit {qubit} but circuit has only {num_qubits} qubits"
        )));
    }
    if gate.dim() != (2, 2) {
        return Err(SimulatorError::InvalidInput(format!(
            "Single-qubit gate matrix must be 2x2, got {:?}",
            gate.dim()
        )));
    }

    let stride = 1usize << qubit;
    let dim = state.len();
    let mut index = 0;
    while index < dim {
        if index & stride == 0 {
            let i0 = index;
            let i1 = index | stride;
            let a0 = state[i0];
            let a1 = state[i1];
            state[i0] = gate[[0, 0]] * a0 + gate[[0, 1]] * a1;
            state[i1] = gate[[1, 0]] * a0 + gate[[1, 1]] * a1;
        }
        index += 1;
    }

    Ok(())
}

/// Apply a two-qubit gate matrix to a dense state vector in place.
///
/// The 4x4 `gate` matrix acts on the basis `|control target>` (control is the high bit,
/// target the low bit). Amplitudes are indexed little-endian by qubit position; for each
/// group of four amplitudes that differ only in the control/target bits the gate is applied
/// as a genuine matrix-vector product.
fn apply_two_qubit_matrix(
    state: &mut Array1<Complex64>,
    gate: &Array2<Complex64>,
    control: usize,
    target: usize,
    num_qubits: usize,
) -> Result<()> {
    if control >= num_qubits || target >= num_qubits {
        return Err(SimulatorError::InvalidInput(format!(
            "Two-qubit gate targets qubits {control} and {target} but circuit has only {num_qubits} qubits"
        )));
    }
    if control == target {
        return Err(SimulatorError::InvalidInput(
            "Two-qubit gate requires two distinct qubits".to_string(),
        ));
    }
    if gate.dim() != (4, 4) {
        return Err(SimulatorError::InvalidInput(format!(
            "Two-qubit gate matrix must be 4x4, got {:?}",
            gate.dim()
        )));
    }

    let control_mask = 1usize << control;
    let target_mask = 1usize << target;
    let dim = state.len();

    for index in 0..dim {
        // Process each block once, from its representative with both bits cleared.
        if index & control_mask == 0 && index & target_mask == 0 {
            let i00 = index;
            let i01 = index | target_mask;
            let i10 = index | control_mask;
            let i11 = index | control_mask | target_mask;

            let amplitudes = [state[i00], state[i01], state[i10], state[i11]];
            let mut updated = [Complex64::new(0.0, 0.0); 4];
            for (row, slot) in updated.iter_mut().enumerate() {
                let mut acc = Complex64::new(0.0, 0.0);
                for (col, &amplitude) in amplitudes.iter().enumerate() {
                    acc += gate[[row, col]] * amplitude;
                }
                *slot = acc;
            }

            state[i00] = updated[0];
            state[i01] = updated[1];
            state[i10] = updated[2];
            state[i11] = updated[3];
        }
    }

    Ok(())
}

/// Compute expectation value of observable for given state
fn compute_expectation_value(
    state: &Array1<Complex64>,
    observable: &PauliOperatorSum,
) -> Result<f64> {
    let mut expectation = 0.0;

    for term in &observable.terms {
        // Compute ⟨ψ|P|ψ⟩ for each Pauli string P
        let pauli_expectation = compute_pauli_expectation_from_state(state, term)?;
        expectation += term.coefficient.re * pauli_expectation.re;
    }

    Ok(expectation)
}

/// Compute expectation value of a single Pauli string
fn compute_pauli_expectation_from_state(
    state: &Array1<Complex64>,
    pauli_string: &PauliString,
) -> Result<Complex64> {
    let num_qubits = pauli_string.num_qubits;
    let dim = 1 << num_qubits;
    let mut result = Complex64::new(0.0, 0.0);

    for (i, &amplitude) in state.iter().enumerate() {
        if i >= dim {
            break;
        }

        // Apply Pauli string to basis state |i⟩
        let mut coeff = Complex64::new(1.0, 0.0);
        let mut target_state = i;

        for (qubit, &pauli_op) in pauli_string.operators.iter().enumerate() {
            let bit = (i >> qubit) & 1;
            use crate::pauli::PauliOperator;

            match pauli_op {
                PauliOperator::I => {} // Identity does nothing
                PauliOperator::X => {
                    // X flips the bit
                    target_state ^= 1 << qubit;
                }
                PauliOperator::Y => {
                    // Y flips the bit and adds phase
                    target_state ^= 1 << qubit;
                    coeff *= if bit == 0 {
                        Complex64::new(0.0, 1.0)
                    } else {
                        Complex64::new(0.0, -1.0)
                    };
                }
                PauliOperator::Z => {
                    // Z adds phase based on bit value
                    if bit == 1 {
                        coeff *= Complex64::new(-1.0, 0.0);
                    }
                }
            }
        }

        if target_state < dim {
            result += amplitude.conj() * coeff * state[target_state];
        }
    }

    Ok(result * pauli_string.coefficient)
}

/// Convenience functions for creating common ansätze
pub mod ansatze {
    use super::ParametricCircuit;

    /// Create a hardware-efficient ansatz.
    ///
    /// Each layer applies a single-qubit `RY`/`RZ` rotation to every qubit
    /// followed by a linear CNOT entangling ladder (`CNOT(q, q+1)` for
    /// `q` in `0..num_qubits-1`). Without the entangling layer this ansatz
    /// could only ever represent a product state; the CNOT ladder is what
    /// lets it represent genuinely entangled states, as any real
    /// hardware-efficient VQE ansatz requires.
    #[must_use]
    pub fn hardware_efficient(num_qubits: usize, num_layers: usize) -> ParametricCircuit {
        let mut circuit = ParametricCircuit::new(num_qubits);
        let mut param_idx = 0;

        for _layer in 0..num_layers {
            // Single-qubit rotations
            for qubit in 0..num_qubits {
                circuit.ry(qubit, param_idx);
                param_idx += 1;
                circuit.rz(qubit, param_idx);
                param_idx += 1;
            }

            // Linear entangling layer: CNOT(q, q+1) for every adjacent pair.
            // This is the standard hardware-efficient-ansatz entangling
            // structure (e.g. Qiskit's `EfficientSU2` with linear
            // entanglement) and is what makes the ansatz capable of
            // representing entangled states at all.
            for qubit in 0..num_qubits.saturating_sub(1) {
                circuit.cnot(qubit, qubit + 1);
            }
        }

        circuit
    }

    /// Create a QAOA ansatz for `MaxCut` problem
    #[must_use]
    pub fn qaoa_maxcut(
        num_qubits: usize,
        num_layers: usize,
        edges: &[(usize, usize)],
    ) -> ParametricCircuit {
        let mut circuit = ParametricCircuit::new(num_qubits);
        let mut param_idx = 0;

        // Initial superposition
        for qubit in 0..num_qubits {
            circuit.ry(qubit, param_idx); // RY(π/2) for H gate equivalent
        }

        for _layer in 0..num_layers {
            // Problem Hamiltonian evolution (ZZ terms)
            for &(i, j) in edges {
                // Would implement ZZ rotation here
                // For now, approximate with RZ gates
                circuit.rz(i, param_idx);
                circuit.rz(j, param_idx);
                param_idx += 1;
            }

            // Mixer Hamiltonian evolution (X terms)
            for qubit in 0..num_qubits {
                circuit.rx(qubit, param_idx);
                param_idx += 1;
            }
        }

        circuit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parametric_rx_matrix() {
        let rx_gate = ParametricRX {
            qubit: 0,
            param_idx: 0,
        };
        let params = vec![PI / 2.0];
        let matrix = rx_gate
            .matrix(&params)
            .expect("RX gate matrix computation should succeed");

        // RX(π/2) should be approximately [[1/√2, -i/√2], [-i/√2, 1/√2]]
        let expected_val = 1.0 / 2.0_f64.sqrt();
        assert!((matrix[[0, 0]].re - expected_val).abs() < 1e-10);
        assert!((matrix[[0, 1]].im + expected_val).abs() < 1e-10);
    }

    #[test]
    fn test_autodiff_context() {
        let params = vec![1.0, 2.0, 3.0];
        let mut context = AutoDiffContext::new(params.clone(), GradientMethod::ParameterShift);

        assert_eq!(context.parameters, params);
        assert_eq!(context.gradients.len(), 3);

        context.update_parameters(vec![4.0, 5.0, 6.0]);
        assert_eq!(context.parameters, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_parametric_circuit_creation() {
        let mut circuit = ParametricCircuit::new(2);
        circuit.rx(0, 0);
        circuit.ry(1, 1);

        assert_eq!(circuit.gates.len(), 2);
        assert_eq!(circuit.num_parameters, 2);
    }

    #[test]
    fn test_hardware_efficient_ansatz() {
        let ansatz = ansatze::hardware_efficient(3, 2);
        assert_eq!(ansatz.num_qubits, 3);
        assert!(ansatz.num_parameters > 0);
    }

    /// Regression test for the P1 finding: `hardware_efficient` used to
    /// omit the entangling layer entirely, so the ansatz could only ever
    /// represent a product state. It must now include a CNOT ladder
    /// (`num_qubits - 1` CNOTs per layer) and actually produce an entangled
    /// state for generic rotation angles.
    #[test]
    fn test_hardware_efficient_ansatz_includes_entangling_layer() {
        let num_qubits = 2;
        let num_layers = 1;
        let ansatz = ansatze::hardware_efficient(num_qubits, num_layers);

        // 1 layer * (2 rotations/qubit * 2 qubits + 1 CNOT) = 5 gates.
        assert_eq!(ansatz.gates.len(), 5);
        let cnot_count = ansatz.gates.iter().filter(|g| g.name() == "CNOT").count();
        assert_eq!(
            cnot_count,
            num_qubits - 1,
            "hardware_efficient must add a linear CNOT entangling ladder"
        );

        // RY(pi/2) on qubit 0 (creating (|0>+|1>)/sqrt(2)), identity on
        // qubit 1 (|0>), then CNOT(0, 1) produces the Bell state
        // (|00>+|11>)/sqrt(2). Its reduced single-qubit density matrices
        // must be maximally mixed (purity = 0.5), which is impossible for
        // any product state built purely from local single-qubit
        // rotations -- i.e. impossible without a real entangling layer.
        let params = vec![PI / 2.0, 0.0, 0.0, 0.0];
        let state = ansatz
            .evaluate(&params)
            .expect("ansatz evaluation should succeed");

        // Reduced density matrix of qubit 0: rho_00 = sum_{b1} |amp(b0=0,b1)|^2 etc.
        let mut rho00 = Complex64::new(0.0, 0.0);
        let mut rho01 = Complex64::new(0.0, 0.0);
        let mut rho11 = Complex64::new(0.0, 0.0);
        for other_bit in 0..2usize {
            let idx0 = other_bit << 1; // qubit0 = 0, qubit1 = other_bit
            let idx1 = (1) | (other_bit << 1); // qubit0 = 1, qubit1 = other_bit
            rho00 += state[idx0] * state[idx0].conj();
            rho11 += state[idx1] * state[idx1].conj();
            rho01 += state[idx0] * state[idx1].conj();
        }
        let purity = (rho00 * rho00 + rho11 * rho11 + rho01 * rho01.conj() * 2.0).re;
        assert!(
            purity < 0.999,
            "expected an entangled (mixed reduced state) result, got purity {purity}"
        );
    }

    /// Test-only parametric gate that returns a fixed (parameter-independent) matrix.
    /// Used to drive the forward evaluation with a known gate such as Hadamard.
    struct FixedGate {
        matrix: Array2<Complex64>,
        wires: Vec<usize>,
    }

    impl ParametricGate for FixedGate {
        fn name(&self) -> &str {
            "FIXED"
        }

        fn qubits(&self) -> Vec<usize> {
            self.wires.clone()
        }

        fn parameter_indices(&self) -> Vec<usize> {
            Vec::new()
        }

        fn matrix(&self, _params: &[f64]) -> Result<Array2<Complex64>> {
            Ok(self.matrix.clone())
        }

        fn gradient(&self, _params: &[f64], _param_idx: usize) -> Result<Array2<Complex64>> {
            Ok(Array2::zeros(self.matrix.dim()))
        }
    }

    #[test]
    fn test_evaluate_hadamard_forward_state() {
        // A circuit consisting of a single Hadamard on qubit 0 of a 1-qubit register.
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let h_matrix = scirs2_core::ndarray::array![
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0)
            ],
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0)
            ]
        ];

        let mut circuit = ParametricCircuit::new(1);
        circuit.add_gate(Box::new(FixedGate {
            matrix: h_matrix,
            wires: vec![0],
        }));

        let state = circuit
            .evaluate(&[])
            .expect("forward evaluation should succeed");

        // Expected: (|0> + |1>)/sqrt(2), NOT the |0> placeholder.
        assert!((state[0].re - inv_sqrt2).abs() < 1e-10, "amp(|0>) wrong");
        assert!((state[1].re - inv_sqrt2).abs() < 1e-10, "amp(|1>) wrong");
        assert!(
            state[1].norm() > 1e-3,
            "forward pass returned the |0> placeholder instead of the real state"
        );
    }

    #[test]
    fn test_evaluate_rx_pi_forward_state() {
        // RX(pi) maps |0> -> -i|1>; this is a genuine, parameter-dependent result.
        let mut circuit = ParametricCircuit::new(1);
        circuit.rx(0, 0);

        let state = circuit
            .evaluate(&[PI])
            .expect("forward evaluation should succeed");

        assert!(state[0].norm() < 1e-10, "amp(|0>) should vanish for RX(pi)");
        assert!((state[1].im + 1.0).abs() < 1e-10, "amp(|1>) should be -i");
    }

    #[test]
    fn test_evaluate_bell_forward_state() {
        // H on qubit 0 then CNOT(0, 1) prepares a Bell state through the real forward pass.
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let h_matrix = scirs2_core::ndarray::array![
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0)
            ],
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0)
            ]
        ];
        let cnot = scirs2_core::ndarray::array![
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0)
            ]
        ];

        let mut circuit = ParametricCircuit::new(2);
        circuit.add_gate(Box::new(FixedGate {
            matrix: h_matrix,
            wires: vec![0],
        }));
        circuit.add_gate(Box::new(FixedGate {
            matrix: cnot,
            wires: vec![0, 1],
        }));

        let state = circuit
            .evaluate(&[])
            .expect("forward evaluation should succeed");

        // Little-endian ordering: index = bit0 + 2*bit1.
        // Bell state (|00> + |11>)/sqrt(2): indices 0 and 3 populated.
        assert!((state[0].re - inv_sqrt2).abs() < 1e-10, "amp(|00>) wrong");
        assert!(state[1].norm() < 1e-10, "amp(|01>) should vanish");
        assert!(state[2].norm() < 1e-10, "amp(|10>) should vanish");
        assert!((state[3].re - inv_sqrt2).abs() < 1e-10, "amp(|11>) wrong");
    }
}
