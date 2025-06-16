//! Quantum Gradient Computation
//!
//! This module implements various methods for computing gradients of quantum circuits,
//! including parameter shift rules, finite differences, and quantum natural gradients.

use super::*;
use crate::continuous_variable::Complex;
use crate::{CircuitExecutor, CircuitResult, DeviceError, DeviceResult, QuantumDevice};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Quantum gradient calculator
pub struct QuantumGradientCalculator {
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    config: GradientConfig,
    method: GradientMethod,
}

/// Configuration for gradient computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientConfig {
    /// Method for computing gradients
    pub method: GradientMethod,
    /// Number of shots per evaluation
    pub shots: usize,
    /// Finite difference step size
    pub finite_diff_step: f64,
    /// Parameter shift rule shift amount
    pub shift_amount: f64,
    /// Use error mitigation
    pub use_error_mitigation: bool,
    /// Parallel gradient computation
    pub parallel_execution: bool,
    /// Gradient clipping threshold
    pub gradient_clipping: Option<f64>,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            method: GradientMethod::ParameterShift,
            shots: 1024,
            finite_diff_step: 1e-4,
            shift_amount: std::f64::consts::PI / 2.0,
            use_error_mitigation: true,
            parallel_execution: true,
            gradient_clipping: Some(1.0),
        }
    }
}

impl QuantumGradientCalculator {
    /// Create a new gradient calculator
    pub fn new(
        device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
        config: GradientConfig,
    ) -> DeviceResult<Self> {
        let method = config.method.clone();

        Ok(Self {
            device,
            config,
            method,
        })
    }

    /// Compute gradients for a parameterized quantum circuit
    pub async fn compute_gradients(
        &self,
        circuit: ParameterizedQuantumCircuit,
        parameters: Vec<f64>,
    ) -> DeviceResult<Vec<f64>> {
        match self.method {
            GradientMethod::ParameterShift => {
                self.parameter_shift_gradients(circuit, parameters).await
            }
            GradientMethod::FiniteDifference => {
                self.finite_difference_gradients(circuit, parameters).await
            }
            GradientMethod::LinearCombination => {
                self.linear_combination_gradients(circuit, parameters).await
            }
            GradientMethod::QuantumNaturalGradient => {
                self.quantum_natural_gradients(circuit, parameters).await
            }
            GradientMethod::Adjoint => self.adjoint_gradients(circuit, parameters).await,
        }
    }

    /// Compute gradients using parameter shift rule
    async fn parameter_shift_gradients(
        &self,
        circuit: ParameterizedQuantumCircuit,
        parameters: Vec<f64>,
    ) -> DeviceResult<Vec<f64>> {
        let mut gradients = vec![0.0; parameters.len()];
        let shift = self.config.shift_amount;

        if self.config.parallel_execution {
            // Parallel computation of all parameter shifts
            let mut tasks = Vec::new();

            for i in 0..parameters.len() {
                let mut params_plus = parameters.clone();
                let mut params_minus = parameters.clone();
                params_plus[i] += shift;
                params_minus[i] -= shift;

                let circuit_plus = circuit.clone();
                let circuit_minus = circuit.clone();
                let device_plus = self.device.clone();
                let device_minus = self.device.clone();
                let shots = self.config.shots;

                let task_plus = tokio::spawn(async move {
                    let circuit_eval =
                        Self::evaluate_circuit_with_params(&circuit_plus, &params_plus)?;
                    let device = device_plus.read().await;
                    Self::execute_circuit_helper(&*device, &circuit_eval, shots).await
                });

                let task_minus = tokio::spawn(async move {
                    let circuit_eval =
                        Self::evaluate_circuit_with_params(&circuit_minus, &params_minus)?;
                    let device = device_minus.read().await;
                    Self::execute_circuit_helper(&*device, &circuit_eval, shots).await
                });

                tasks.push((i, task_plus, task_minus));
            }

            // Collect results
            for (param_idx, task_plus, task_minus) in tasks {
                let result_plus = task_plus
                    .await
                    .map_err(|e| DeviceError::InvalidInput(format!("Task error: {}", e)))??;
                let result_minus = task_minus
                    .await
                    .map_err(|e| DeviceError::InvalidInput(format!("Task error: {}", e)))??;

                let expectation_plus = self.compute_expectation_value(&result_plus)?;
                let expectation_minus = self.compute_expectation_value(&result_minus)?;

                gradients[param_idx] = (expectation_plus - expectation_minus) / 2.0;
            }
        } else {
            // Sequential computation
            for i in 0..parameters.len() {
                let mut params_plus = parameters.clone();
                let mut params_minus = parameters.clone();
                params_plus[i] += shift;
                params_minus[i] -= shift;

                let circuit_plus = Self::evaluate_circuit_with_params(&circuit, &params_plus)?;
                let circuit_minus = Self::evaluate_circuit_with_params(&circuit, &params_minus)?;

                let device = self.device.read().await;
                let result_plus = Self::execute_circuit_helper(&*device, &circuit_plus, self.config.shots).await?;
                let result_minus = Self::execute_circuit_helper(&*device, &circuit_minus, self.config.shots).await?;

                let expectation_plus = self.compute_expectation_value(&result_plus)?;
                let expectation_minus = self.compute_expectation_value(&result_minus)?;

                gradients[i] = (expectation_plus - expectation_minus) / 2.0;
            }
        }

        // Apply gradient clipping if specified
        if let Some(clip_value) = self.config.gradient_clipping {
            for grad in &mut gradients {
                *grad = grad.clamp(-clip_value, clip_value);
            }
        }

        Ok(gradients)
    }

    /// Compute gradients using finite differences
    async fn finite_difference_gradients(
        &self,
        circuit: ParameterizedQuantumCircuit,
        parameters: Vec<f64>,
    ) -> DeviceResult<Vec<f64>> {
        let mut gradients = vec![0.0; parameters.len()];
        let step = self.config.finite_diff_step;

        for i in 0..parameters.len() {
            let mut params_plus = parameters.clone();
            let mut params_minus = parameters.clone();
            params_plus[i] += step;
            params_minus[i] -= step;

            let circuit_plus = Self::evaluate_circuit_with_params(&circuit, &params_plus)?;
            let circuit_minus = Self::evaluate_circuit_with_params(&circuit, &params_minus)?;

            let device = self.device.read().await;
            let result_plus = Self::execute_circuit_helper(&*device, &circuit_plus, self.config.shots).await?;
            let result_minus = Self::execute_circuit_helper(&*device, &circuit_minus, self.config.shots).await?;

            let expectation_plus = self.compute_expectation_value(&result_plus)?;
            let expectation_minus = self.compute_expectation_value(&result_minus)?;

            gradients[i] = (expectation_plus - expectation_minus) / (2.0 * step);
        }

        Ok(gradients)
    }

    /// Compute gradients using linear combination of unitaries (LCU)
    async fn linear_combination_gradients(
        &self,
        circuit: ParameterizedQuantumCircuit,
        parameters: Vec<f64>,
    ) -> DeviceResult<Vec<f64>> {
        // This is a simplified implementation of LCU gradients
        // In practice, this would decompose the gradient operator into a linear combination
        let mut gradients = vec![0.0; parameters.len()];

        for i in 0..parameters.len() {
            // Simplified: use a small finite difference as approximation
            let step = 1e-3;
            let mut params_plus = parameters.clone();
            params_plus[i] += step;

            let circuit_original = Self::evaluate_circuit_with_params(&circuit, &parameters)?;
            let circuit_plus = Self::evaluate_circuit_with_params(&circuit, &params_plus)?;

            let device = self.device.read().await;
            let result_original = Self::execute_circuit_helper(&*device, &circuit_original, self.config.shots).await?;
            let result_plus = Self::execute_circuit_helper(&*device, &circuit_plus, self.config.shots).await?;

            let expectation_original = self.compute_expectation_value(&result_original)?;
            let expectation_plus = self.compute_expectation_value(&result_plus)?;

            gradients[i] = (expectation_plus - expectation_original) / step;
        }

        Ok(gradients)
    }

    /// Compute quantum natural gradients
    async fn quantum_natural_gradients(
        &self,
        circuit: ParameterizedQuantumCircuit,
        parameters: Vec<f64>,
    ) -> DeviceResult<Vec<f64>> {
        // First compute regular gradients
        let regular_gradients = self
            .parameter_shift_gradients(circuit.clone(), parameters.clone())
            .await?;

        // Compute quantum Fisher information matrix (simplified)
        let fisher_matrix = self
            .compute_quantum_fisher_information(&circuit, &parameters)
            .await?;

        // Solve Fisher^{-1} * gradient
        let natural_gradients = self.solve_linear_system(&fisher_matrix, &regular_gradients)?;

        Ok(natural_gradients)
    }

    /// Compute gradients using adjoint method (simplified)
    async fn adjoint_gradients(
        &self,
        circuit: ParameterizedQuantumCircuit,
        parameters: Vec<f64>,
    ) -> DeviceResult<Vec<f64>> {
        // This is a placeholder for adjoint gradient computation
        // Real implementation would require access to quantum state amplitudes
        // For now, fall back to parameter shift rule
        self.parameter_shift_gradients(circuit, parameters).await
    }

    /// Compute quantum Fisher information matrix
    async fn compute_quantum_fisher_information(
        &self,
        circuit: &ParameterizedQuantumCircuit,
        parameters: &[f64],
    ) -> DeviceResult<Vec<Vec<f64>>> {
        let n_params = parameters.len();
        let mut fisher_matrix = vec![vec![0.0; n_params]; n_params];
        let shift = std::f64::consts::PI / 2.0;

        for i in 0..n_params {
            for j in i..n_params {
                if i == j {
                    // Diagonal elements: Var[∂ψ/∂θᵢ]
                    let mut params_plus = parameters.to_vec();
                    let mut params_minus = parameters.to_vec();
                    params_plus[i] += shift;
                    params_minus[i] -= shift;

                    let circuit_plus = Self::evaluate_circuit_with_params(circuit, &params_plus)?;
                    let circuit_minus = Self::evaluate_circuit_with_params(circuit, &params_minus)?;

                    let device = self.device.read().await;
                    let result_plus = Self::execute_circuit_helper(&*device, &circuit_plus, self.config.shots).await?;
                    let result_minus = Self::execute_circuit_helper(&*device, &circuit_minus, self.config.shots).await?;

                    let overlap = self.compute_state_overlap(&result_plus, &result_minus)?;
                    fisher_matrix[i][j] = (1.0 - overlap.real) / 2.0;
                } else {
                    // Off-diagonal elements: Re[⟨∂ψ/∂θᵢ|∂ψ/∂θⱼ⟩]
                    // Simplified computation
                    fisher_matrix[i][j] = 0.0;
                    fisher_matrix[j][i] = fisher_matrix[i][j];
                }
            }
        }

        // Add regularization to ensure invertibility
        for i in 0..n_params {
            fisher_matrix[i][i] += 1e-6;
        }

        Ok(fisher_matrix)
    }

    /// Compute overlap between quantum states (simplified)
    fn compute_state_overlap(
        &self,
        result1: &CircuitResult,
        result2: &CircuitResult,
    ) -> DeviceResult<Complex> {
        // This is a simplified overlap computation based on measurement statistics
        // Real implementation would require access to quantum state amplitudes

        let mut overlap_real = 0.0;
        let total_shots1 = result1.shots as f64;
        let total_shots2 = result2.shots as f64;

        for (bitstring, count1) in &result1.counts {
            if let Some(count2) = result2.counts.get(bitstring) {
                let prob1 = *count1 as f64 / total_shots1;
                let prob2 = *count2 as f64 / total_shots2;
                overlap_real += (prob1 * prob2).sqrt();
            }
        }

        Ok(Complex::new(overlap_real, 0.0))
    }

    /// Solve linear system Ax = b
    fn solve_linear_system(&self, matrix: &[Vec<f64>], vector: &[f64]) -> DeviceResult<Vec<f64>> {
        let n = matrix.len();
        if n != vector.len() {
            return Err(DeviceError::InvalidInput(
                "Matrix and vector dimensions don't match".to_string(),
            ));
        }

        // Simple Gaussian elimination (for small systems)
        let mut augmented = matrix
            .iter()
            .zip(vector.iter())
            .map(|(row, &b)| {
                let mut aug_row = row.clone();
                aug_row.push(b);
                aug_row
            })
            .collect::<Vec<_>>();

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if augmented[k][i].abs() > augmented[max_row][i].abs() {
                    max_row = k;
                }
            }
            augmented.swap(i, max_row);

            // Check for singularity
            if augmented[i][i].abs() < 1e-10 {
                return Err(DeviceError::InvalidInput(
                    "Singular matrix in linear system".to_string(),
                ));
            }

            // Eliminate
            for k in i + 1..n {
                let factor = augmented[k][i] / augmented[i][i];
                for j in i..=n {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }

        // Back substitution
        let mut solution = vec![0.0; n];
        for i in (0..n).rev() {
            solution[i] = augmented[i][n];
            for j in i + 1..n {
                solution[i] -= augmented[i][j] * solution[j];
            }
            solution[i] /= augmented[i][i];
        }

        Ok(solution)
    }

    /// Execute a circuit on the quantum device
    async fn execute_circuit_helper(
        device: &(dyn QuantumDevice + Send + Sync),
        circuit: &ParameterizedQuantumCircuit,
        shots: usize,
    ) -> DeviceResult<CircuitResult> {
        // For now, return a mock result since we can't execute circuits directly
        // In a real implementation, this would need proper circuit execution
        let mut counts = std::collections::HashMap::new();
        counts.insert("0".repeat(circuit.num_qubits()), shots / 2);
        counts.insert("1".repeat(circuit.num_qubits()), shots / 2);
        
        Ok(CircuitResult {
            counts,
            shots,
            metadata: std::collections::HashMap::new(),
        })
    }

    /// Evaluate a parameterized circuit with specific parameter values
    fn evaluate_circuit_with_params(
        circuit: &ParameterizedQuantumCircuit,
        parameters: &[f64],
    ) -> DeviceResult<ParameterizedQuantumCircuit> {
        // This would substitute parameters into the circuit
        // For now, return a copy (implementation would be more sophisticated)
        Ok(circuit.clone())
    }

    /// Compute expectation value from measurement results
    fn compute_expectation_value(&self, result: &CircuitResult) -> DeviceResult<f64> {
        // Simple expectation value: average number of 1s
        let mut expectation = 0.0;
        let total_shots = result.shots as f64;

        for (bitstring, count) in &result.counts {
            let ones_count = bitstring.chars().filter(|&c| c == '1').count();
            let probability = *count as f64 / total_shots;
            expectation += ones_count as f64 * probability;
        }

        Ok(expectation)
    }

    /// Compute gradients with respect to a specific observable
    pub async fn compute_observable_gradients(
        &self,
        circuit: ParameterizedQuantumCircuit,
        parameters: Vec<f64>,
        observable: Observable,
    ) -> DeviceResult<Vec<f64>> {
        match self.method {
            GradientMethod::ParameterShift => {
                self.parameter_shift_observable_gradients(circuit, parameters, observable)
                    .await
            }
            _ => {
                // For other methods, use default expectation value
                self.compute_gradients(circuit, parameters).await
            }
        }
    }

    /// Compute gradients with respect to a specific observable using parameter shift
    async fn parameter_shift_observable_gradients(
        &self,
        circuit: ParameterizedQuantumCircuit,
        parameters: Vec<f64>,
        observable: Observable,
    ) -> DeviceResult<Vec<f64>> {
        let mut gradients = vec![0.0; parameters.len()];
        let shift = self.config.shift_amount;

        for i in 0..parameters.len() {
            let mut params_plus = parameters.clone();
            let mut params_minus = parameters.clone();
            params_plus[i] += shift;
            params_minus[i] -= shift;

            let circuit_plus = Self::evaluate_circuit_with_params(&circuit, &params_plus)?;
            let circuit_minus = Self::evaluate_circuit_with_params(&circuit, &params_minus)?;

            let device = self.device.read().await;
            let result_plus = Self::execute_circuit_helper(&*device, &circuit_plus, self.config.shots).await?;
            let result_minus = Self::execute_circuit_helper(&*device, &circuit_minus, self.config.shots).await?;

            let expectation_plus =
                self.compute_observable_expectation(&result_plus, &observable)?;
            let expectation_minus =
                self.compute_observable_expectation(&result_minus, &observable)?;

            gradients[i] = (expectation_plus - expectation_minus) / 2.0;
        }

        Ok(gradients)
    }

    /// Compute expectation value of an observable
    fn compute_observable_expectation(
        &self,
        result: &CircuitResult,
        observable: &Observable,
    ) -> DeviceResult<f64> {
        let mut expectation = 0.0;
        let total_shots = result.shots as f64;

        for (bitstring, count) in &result.counts {
            let probability = *count as f64 / total_shots;
            let eigenvalue = observable.evaluate_bitstring(bitstring)?;
            expectation += probability * eigenvalue;
        }

        Ok(expectation)
    }
}

/// Observable for expectation value computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observable {
    pub terms: Vec<ObservableTerm>,
}

/// Single term in an observable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservableTerm {
    pub coefficient: f64,
    pub pauli_string: Vec<(usize, PauliOperator)>, // (qubit_index, pauli_operator)
}

impl Observable {
    /// Create a Z observable on a single qubit
    pub fn single_z(qubit: usize) -> Self {
        Self {
            terms: vec![ObservableTerm {
                coefficient: 1.0,
                pauli_string: vec![(qubit, PauliOperator::Z)],
            }],
        }
    }

    /// Create an all-Z observable (sum of Z on all qubits)
    pub fn all_z(num_qubits: usize) -> Self {
        let terms = (0..num_qubits)
            .map(|i| ObservableTerm {
                coefficient: 1.0,
                pauli_string: vec![(i, PauliOperator::Z)],
            })
            .collect();

        Self { terms }
    }

    /// Evaluate observable for a given bitstring
    pub fn evaluate_bitstring(&self, bitstring: &str) -> DeviceResult<f64> {
        let mut value = 0.0;

        for term in &self.terms {
            let mut term_value = term.coefficient;

            for (qubit_idx, pauli_op) in &term.pauli_string {
                if let Some(bit_char) = bitstring.chars().nth(*qubit_idx) {
                    let bit_value = if bit_char == '1' { -1.0 } else { 1.0 };

                    match pauli_op {
                        PauliOperator::Z => term_value *= bit_value,
                        PauliOperator::I => {} // Identity
                        PauliOperator::X | PauliOperator::Y => {
                            // Would need basis rotation for X/Y measurements
                            return Err(DeviceError::InvalidInput(
                                "X and Y Pauli measurements require basis rotation".to_string(),
                            ));
                        }
                    }
                }
            }

            value += term_value;
        }

        Ok(value)
    }
}

/// Gradient computation utilities
pub struct GradientUtils;

impl GradientUtils {
    /// Estimate gradients using central differences
    pub fn central_difference(
        f: impl Fn(&[f64]) -> f64,
        parameters: &[f64],
        step_size: f64,
    ) -> Vec<f64> {
        let mut gradients = vec![0.0; parameters.len()];

        for i in 0..parameters.len() {
            let mut params_plus = parameters.to_vec();
            let mut params_minus = parameters.to_vec();
            params_plus[i] += step_size;
            params_minus[i] -= step_size;

            let f_plus = f(&params_plus);
            let f_minus = f(&params_minus);

            gradients[i] = (f_plus - f_minus) / (2.0 * step_size);
        }

        gradients
    }

    /// Clip gradients to prevent exploding gradients
    pub fn clip_gradients(gradients: &mut [f64], max_norm: f64) {
        let norm = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
        if norm > max_norm {
            let scale = max_norm / norm;
            for grad in gradients {
                *grad *= scale;
            }
        }
    }

    /// Apply momentum to gradient updates
    pub fn apply_momentum(
        gradients: &[f64],
        momentum_buffer: &mut Vec<f64>,
        momentum: f64,
    ) -> Vec<f64> {
        if momentum_buffer.len() != gradients.len() {
            momentum_buffer.resize(gradients.len(), 0.0);
        }

        let mut updated_gradients = Vec::with_capacity(gradients.len());
        for i in 0..gradients.len() {
            momentum_buffer[i] = momentum * momentum_buffer[i] + gradients[i];
            updated_gradients.push(momentum_buffer[i]);
        }

        updated_gradients
    }
}

/// Create a parameter shift gradient calculator
pub fn create_parameter_shift_calculator(
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    shots: usize,
) -> DeviceResult<QuantumGradientCalculator> {
    let config = GradientConfig {
        method: GradientMethod::ParameterShift,
        shots,
        ..Default::default()
    };

    QuantumGradientCalculator::new(device, config)
}

/// Create a finite difference gradient calculator
pub fn create_finite_difference_calculator(
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    step_size: f64,
) -> DeviceResult<QuantumGradientCalculator> {
    let config = GradientConfig {
        method: GradientMethod::FiniteDifference,
        finite_diff_step: step_size,
        ..Default::default()
    };

    QuantumGradientCalculator::new(device, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::create_mock_quantum_device;

    #[tokio::test]
    async fn test_gradient_calculator_creation() {
        let device = create_mock_quantum_device();
        let calculator = QuantumGradientCalculator::new(device, GradientConfig::default()).unwrap();

        assert_eq!(calculator.config.method, GradientMethod::ParameterShift);
        assert_eq!(calculator.config.shots, 1024);
    }

    #[test]
    fn test_observable_creation() {
        let obs = Observable::single_z(0);
        assert_eq!(obs.terms.len(), 1);
        assert_eq!(obs.terms[0].coefficient, 1.0);

        let obs_all = Observable::all_z(4);
        assert_eq!(obs_all.terms.len(), 4);
    }

    #[test]
    fn test_observable_evaluation() {
        let obs = Observable::single_z(0);

        let value_0 = obs.evaluate_bitstring("0").unwrap();
        assert_eq!(value_0, 1.0);

        let value_1 = obs.evaluate_bitstring("1").unwrap();
        assert_eq!(value_1, -1.0);
    }

    #[test]
    fn test_gradient_utils() {
        let quadratic = |params: &[f64]| params[0] * params[0] + 2.0 * params[1] * params[1];
        let gradients = GradientUtils::central_difference(quadratic, &[1.0, 2.0], 1e-5);

        // Analytical gradients: [2x, 4y] = [2.0, 8.0]
        assert!((gradients[0] - 2.0).abs() < 1e-3);
        assert!((gradients[1] - 8.0).abs() < 1e-3);
    }

    #[test]
    fn test_gradient_clipping() {
        let mut gradients = vec![3.0, 4.0]; // Norm = 5.0
        GradientUtils::clip_gradients(&mut gradients, 2.0);

        let new_norm = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
        assert!((new_norm - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_momentum() {
        let gradients = vec![1.0, 2.0];
        let mut momentum_buffer = vec![0.5, -0.5];

        let updated = GradientUtils::apply_momentum(&gradients, &mut momentum_buffer, 0.9);

        // Expected: [0.9 * 0.5 + 1.0, 0.9 * (-0.5) + 2.0] = [1.45, 1.55]
        assert!((updated[0] - 1.45).abs() < 1e-10);
        assert!((updated[1] - 1.55).abs() < 1e-10);
    }
}
