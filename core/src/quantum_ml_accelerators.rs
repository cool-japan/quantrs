//! Quantum Machine Learning Accelerators
//!
//! Hardware-specific quantum ML gate optimizations with tensor network decompositions
//! and variational quantum eigenstate preparation.

use crate::error::QuantRS2Error;
use crate::gate::GateOp;
use crate::matrix_ops::{DenseMatrix, QuantumMatrix};
// use crate::qubit::QubitId;
use crate::tensor_network::{Tensor, TensorNetwork};
// use crate::variational::{VariationalGate, VariationalOptimizer};
use scirs2_core::Complex64;
// use scirs2_linalg::{decompose_svd, matrix_exp, qr_decompose};
use scirs2_core::ndarray::{Array1, Array2, Axis};

// Fallback optimization types when scirs2_optimize is not available
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub parameters: Array1<f64>,
    pub cost: f64,
    pub iterations: usize,
}

#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

/// Gradient-free coordinate-descent minimiser.
///
/// Performs real optimisation: at each iteration it tries a positive and a
/// negative step along every coordinate, accepting any move that lowers the
/// objective, and shrinks the step size when a full sweep makes no progress.
/// It terminates on `max_iterations`, when the step size underflows `tolerance`,
/// or when an entire sweep yields no improvement. This genuinely reduces the
/// cost rather than echoing the initial parameters.
pub fn minimize<F>(
    objective: F,
    initial_params: &Array1<f64>,
    config: &OptimizationConfig,
) -> Result<OptimizationResult, String>
where
    F: Fn(&Array1<f64>) -> Result<f64, String>,
{
    let mut params = initial_params.clone();
    let mut best_cost = objective(&params)?;
    let n = params.len();

    // Initial step size; falls back to a sensible default for empty/degenerate
    // problems.
    let mut step = 0.1_f64.max(config.tolerance * 10.0);
    let mut iterations = 0usize;

    while iterations < config.max_iterations {
        iterations += 1;
        let mut improved = false;

        for i in 0..n {
            // Try +step then -step along coordinate i.
            for &direction in &[step, -step] {
                let original = params[i];
                params[i] = original + direction;
                match objective(&params) {
                    Ok(trial_cost) if trial_cost < best_cost => {
                        best_cost = trial_cost;
                        improved = true;
                    }
                    _ => {
                        // Revert if no improvement (or evaluation failed).
                        params[i] = original;
                    }
                }
            }
        }

        if !improved {
            // No coordinate move helped: refine the search by halving the step.
            step *= 0.5;
            if step < config.tolerance {
                break;
            }
        }
    }

    Ok(OptimizationResult {
        parameters: params,
        cost: best_cost,
        iterations,
    })
}
// use std::collections::HashMap;
use std::f64::consts::PI;

/// Compute the singular value decomposition `M = U Σ Vᴴ` of a complex matrix.
///
/// Returns `(U, S, Vt)` where `U` (`nrows × min_dim`) holds the left singular
/// vectors as columns, `S` (`min_dim`) holds the singular values in descending
/// order, and `Vt` (`min_dim × ncols`) holds the conjugate-transposed right
/// singular vectors (i.e. `Vᴴ`).
///
/// The decomposition is obtained from the Hermitian eigendecomposition of the
/// Gram matrix `G = Mᴴ M`: its eigenvectors are the right singular vectors `V`,
/// its eigenvalues are `σ²`, and the left singular vectors follow from
/// `u_k = M v_k / σ_k`. This is a genuine, exact SVD for complex inputs and uses
/// the in-tree [`crate::eigensolve::eigen_decompose_unitary`] (which handles
/// general — including Hermitian — matrices) rather than fabricating identities.
fn decompose_svd(
    matrix: &Array2<Complex64>,
) -> Result<(Array2<Complex64>, Array1<f64>, Array2<Complex64>), QuantRS2Error> {
    let (nrows, ncols) = matrix.dim();
    let min_dim = nrows.min(ncols);

    if nrows == 0 || ncols == 0 {
        return Ok((
            Array2::zeros((nrows, min_dim)),
            Array1::zeros(min_dim),
            Array2::zeros((min_dim, ncols)),
        ));
    }

    // Gram matrix G = Mᴴ M  (ncols × ncols, Hermitian positive semidefinite).
    let m_dag = matrix.t().mapv(|z| z.conj());
    let gram = m_dag.dot(matrix);

    // Eigendecomposition of the Hermitian Gram matrix. Eigenvalues are σ² (real,
    // non-negative up to numerical error); eigenvectors are the right singular
    // vectors V (as columns).
    let eig = crate::eigensolve::eigen_decompose_unitary(&gram, 1e-12)?;

    // Collect (eigenvalue, column-index) and sort by singular value descending.
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&i, &j| {
        let si = eig.eigenvalues[j].re.max(0.0).sqrt();
        let sj = eig.eigenvalues[i].re.max(0.0).sqrt();
        si.total_cmp(&sj)
    });

    let mut singular_values = Array1::<f64>::zeros(min_dim);
    let mut v_right = Array2::<Complex64>::zeros((ncols, min_dim));
    let mut u_left = Array2::<Complex64>::zeros((nrows, min_dim));

    for (out_idx, &col) in order.iter().take(min_dim).enumerate() {
        let sigma = eig.eigenvalues[col].re.max(0.0).sqrt();
        singular_values[out_idx] = sigma;

        // Right singular vector v_k = eigenvector column `col`.
        let v_k = eig.eigenvectors.column(col).to_owned();
        for r in 0..ncols {
            v_right[[r, out_idx]] = v_k[r];
        }

        // Left singular vector u_k = M v_k / σ_k (only well-defined for σ_k > 0).
        if sigma > 1e-12 {
            let mv = matrix.dot(&v_k);
            for r in 0..nrows {
                u_left[[r, out_idx]] = mv[r] / Complex64::new(sigma, 0.0);
            }
        }
    }

    // Fill any null-space columns of U with an orthonormal completion so that the
    // returned U has orthonormal columns even when M is rank-deficient.
    complete_orthonormal_columns(&mut u_left, &singular_values);

    // Vt = Vᴴ  (min_dim × ncols).
    let vt = v_right.t().mapv(|z| z.conj());

    Ok((u_left, singular_values, vt))
}

/// Fill columns of `u` that correspond to zero singular values with vectors that
/// are orthonormal to the already-populated columns (a Gram–Schmidt completion).
fn complete_orthonormal_columns(u: &mut Array2<Complex64>, singular_values: &Array1<f64>) {
    let nrows = u.nrows();
    let ncols = u.ncols();
    let mut next_basis = 0usize;

    for col in 0..ncols {
        if singular_values[col] > 1e-12 {
            continue;
        }
        // Find a standard basis vector not yet in the span and orthonormalise it.
        while next_basis < nrows {
            let mut candidate = Array1::<Complex64>::zeros(nrows);
            candidate[next_basis] = Complex64::new(1.0, 0.0);
            next_basis += 1;

            // Gram–Schmidt against existing columns.
            for prev in 0..ncols {
                if prev == col {
                    continue;
                }
                let prev_col = u.column(prev).to_owned();
                let norm_sq: f64 = prev_col.iter().map(|z| z.norm_sqr()).sum();
                if norm_sq < 1e-24 {
                    continue;
                }
                let proj: Complex64 = prev_col
                    .iter()
                    .zip(candidate.iter())
                    .map(|(p, c)| p.conj() * c)
                    .sum();
                for r in 0..nrows {
                    candidate[r] -= proj * prev_col[r];
                }
            }

            let norm: f64 = candidate.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for r in 0..nrows {
                    u[[r, col]] = candidate[r] / Complex64::new(norm, 0.0);
                }
                break;
            }
        }
    }
}

/// Quantum natural gradient implementations
#[derive(Debug, Clone)]
pub struct QuantumNaturalGradient {
    pub fisher_information: Array2<f64>,
    pub gradient: Array1<f64>,
    pub regularization: f64,
}

impl QuantumNaturalGradient {
    /// Create a new quantum natural gradient calculator
    pub fn new(parameter_count: usize, regularization: f64) -> Self {
        Self {
            fisher_information: Array2::eye(parameter_count),
            gradient: Array1::zeros(parameter_count),
            regularization,
        }
    }

    /// Compute the quantum Fisher information matrix
    pub fn compute_fisher_information(
        &mut self,
        circuit_generator: impl Fn(&Array1<f64>) -> Result<Array2<Complex64>, QuantRS2Error>,
        parameters: &Array1<f64>,
        state: &Array1<Complex64>,
    ) -> Result<(), QuantRS2Error> {
        let n_params = parameters.len();
        let eps = 1e-8;

        // Compute Fisher information using parameter-shift rule
        for i in 0..n_params {
            for j in i..n_params {
                let mut params_plus = parameters.clone();
                let mut params_minus = parameters.clone();
                params_plus[i] += eps;
                params_minus[i] -= eps;

                let circuit_plus = circuit_generator(&params_plus)?;
                let circuit_minus = circuit_generator(&params_minus)?;

                let state_plus = circuit_plus.dot(state);
                let state_minus = circuit_minus.dot(state);

                // Fisher information element
                let overlap = state_plus.dot(&state_minus.mapv(|x| x.conj()));
                let fisher_element = 4.0 * (1.0 - overlap.norm_sqr());

                self.fisher_information[[i, j]] = fisher_element;
                if i != j {
                    self.fisher_information[[j, i]] = fisher_element;
                }
            }
        }

        // Add regularization
        for i in 0..n_params {
            self.fisher_information[[i, i]] += self.regularization;
        }

        Ok(())
    }

    /// Compute the natural gradient
    pub fn natural_gradient(&self) -> Result<Array1<f64>, QuantRS2Error> {
        // Natural gradient = F^(-1) * gradient
        // Simplified pseudo-inverse (in production, use proper matrix inversion)
        let n = self.fisher_information.nrows();
        let mut fisher_inv = Array2::eye(n);
        for i in 0..n {
            let diag_val = self.fisher_information[[i, i]];
            if diag_val.abs() > 1e-10 {
                fisher_inv[[i, i]] = 1.0 / diag_val;
            }
        }
        Ok(fisher_inv.dot(&self.gradient))
    }

    /// Update parameters using natural gradient descent
    pub fn update_parameters(
        &self,
        parameters: &Array1<f64>,
        learning_rate: f64,
    ) -> Result<Array1<f64>, QuantRS2Error> {
        let nat_grad = self.natural_gradient()?;
        Ok(parameters - learning_rate * &nat_grad)
    }
}

/// Parameter-shift rule optimizations for ML gradients
#[derive(Debug, Clone)]
pub struct ParameterShiftOptimizer {
    pub shift_value: f64,
    pub higher_order_shifts: Vec<f64>,
    pub use_finite_differences: bool,
}

impl Default for ParameterShiftOptimizer {
    fn default() -> Self {
        Self {
            shift_value: PI / 2.0,
            higher_order_shifts: vec![PI / 2.0, PI, 3.0 * PI / 2.0],
            use_finite_differences: false,
        }
    }
}

impl ParameterShiftOptimizer {
    /// Compute gradient using parameter-shift rule
    pub fn compute_gradient(
        &self,
        expectation_fn: impl Fn(&Array1<f64>) -> Result<f64, QuantRS2Error>,
        parameters: &Array1<f64>,
    ) -> Result<Array1<f64>, QuantRS2Error> {
        let n_params = parameters.len();
        let mut gradient = Array1::zeros(n_params);

        for i in 0..n_params {
            if self.use_finite_differences {
                gradient[i] = self.finite_difference_gradient(&expectation_fn, parameters, i)?;
            } else {
                gradient[i] = self.parameter_shift_gradient(&expectation_fn, parameters, i)?;
            }
        }

        Ok(gradient)
    }

    /// Parameter-shift rule for a single parameter
    fn parameter_shift_gradient(
        &self,
        expectation_fn: &impl Fn(&Array1<f64>) -> Result<f64, QuantRS2Error>,
        parameters: &Array1<f64>,
        param_idx: usize,
    ) -> Result<f64, QuantRS2Error> {
        let mut params_plus = parameters.clone();
        let mut params_minus = parameters.clone();

        params_plus[param_idx] += self.shift_value;
        params_minus[param_idx] -= self.shift_value;

        let exp_plus = expectation_fn(&params_plus)?;
        let exp_minus = expectation_fn(&params_minus)?;

        Ok((exp_plus - exp_minus) / 2.0)
    }

    /// Finite difference approximation
    fn finite_difference_gradient(
        &self,
        expectation_fn: &impl Fn(&Array1<f64>) -> Result<f64, QuantRS2Error>,
        parameters: &Array1<f64>,
        param_idx: usize,
    ) -> Result<f64, QuantRS2Error> {
        let eps = 1e-7;
        let mut params_plus = parameters.clone();
        let mut params_minus = parameters.clone();

        params_plus[param_idx] += eps;
        params_minus[param_idx] -= eps;

        let exp_plus = expectation_fn(&params_plus)?;
        let exp_minus = expectation_fn(&params_minus)?;

        Ok((exp_plus - exp_minus) / (2.0 * eps))
    }

    /// Higher-order parameter shifts for better accuracy
    pub fn higher_order_gradient(
        &self,
        expectation_fn: impl Fn(&Array1<f64>) -> Result<f64, QuantRS2Error>,
        parameters: &Array1<f64>,
        param_idx: usize,
    ) -> Result<f64, QuantRS2Error> {
        let mut gradient = 0.0;
        let weights = [0.5, -0.5, 0.0, 0.0]; // Coefficients for 4-point stencil

        for (i, &shift) in self.higher_order_shifts.iter().enumerate() {
            if i < weights.len() {
                let mut params = parameters.clone();
                params[param_idx] += shift;
                let expectation = expectation_fn(&params)?;
                gradient += weights[i] * expectation;
            }
        }

        Ok(gradient)
    }
}

/// Quantum kernel feature map optimizations
#[derive(Debug, Clone)]
pub struct QuantumKernelOptimizer {
    pub feature_map: QuantumFeatureMap,
    pub kernel_matrix: Array2<f64>,
    pub optimization_history: Vec<f64>,
    pub feature_map_parameters: Array1<f64>,
}

#[derive(Debug, Clone)]
pub enum QuantumFeatureMap {
    ZZFeatureMap { num_qubits: usize, depth: usize },
    PauliFeatureMap { paulis: Vec<String>, depth: usize },
    CustomFeatureMap { gates: Vec<Box<dyn GateOp>> },
}

impl QuantumKernelOptimizer {
    /// Create a new quantum kernel optimizer
    pub fn new(feature_map: QuantumFeatureMap) -> Self {
        Self {
            feature_map,
            kernel_matrix: Array2::zeros((1, 1)),
            optimization_history: Vec::new(),
            feature_map_parameters: Array1::zeros(4), // Default parameter count
        }
    }

    /// Compute the quantum kernel matrix
    pub fn compute_kernel_matrix(
        &mut self,
        data_points: &Array2<f64>,
    ) -> Result<Array2<f64>, QuantRS2Error> {
        let n_samples = data_points.nrows();
        let mut kernel_matrix = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i..n_samples {
                let x_i = data_points.row(i);
                let x_j = data_points.row(j);
                let kernel_value = self.compute_kernel_element(&x_i, &x_j)?;

                kernel_matrix[[i, j]] = kernel_value;
                kernel_matrix[[j, i]] = kernel_value;
            }
        }

        self.kernel_matrix.clone_from(&kernel_matrix);
        Ok(kernel_matrix)
    }

    /// Compute a single kernel element
    fn compute_kernel_element(
        &self,
        x_i: &scirs2_core::ndarray::ArrayView1<f64>,
        x_j: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> Result<f64, QuantRS2Error> {
        let circuit_i = self.create_feature_circuit(&x_i.to_owned())?;
        let circuit_j = self.create_feature_circuit(&x_j.to_owned())?;

        // Quantum kernel is |<φ(x_i)|φ(x_j)>|²
        let overlap = circuit_i.t().dot(&circuit_j);
        Ok(overlap.diag().map(|x| x.norm_sqr()).sum())
    }

    /// Create feature mapping circuit
    fn create_feature_circuit(
        &self,
        data_point: &Array1<f64>,
    ) -> Result<Array2<Complex64>, QuantRS2Error> {
        match &self.feature_map {
            QuantumFeatureMap::ZZFeatureMap { num_qubits, depth } => {
                self.create_zz_feature_map(data_point, *num_qubits, *depth)
            }
            QuantumFeatureMap::PauliFeatureMap { paulis, depth } => {
                self.create_pauli_feature_map(data_point, paulis, *depth)
            }
            QuantumFeatureMap::CustomFeatureMap { gates: _ } => {
                // Custom implementation would go here
                Ok(Array2::eye(2_usize.pow(data_point.len() as u32)))
            }
        }
    }

    /// Create ZZ feature map circuit
    fn create_zz_feature_map(
        &self,
        data_point: &Array1<f64>,
        num_qubits: usize,
        depth: usize,
    ) -> Result<Array2<Complex64>, QuantRS2Error> {
        let dim = 2_usize.pow(num_qubits as u32);
        let mut circuit = Array2::eye(dim);

        for layer in 0..depth {
            // Rotation gates
            for qubit in 0..num_qubits {
                let angle = data_point[qubit % data_point.len()] * (layer + 1) as f64;
                let rotation = self.ry_gate(angle);
                circuit = self.apply_single_qubit_gate(&circuit, &rotation, qubit, num_qubits)?;
            }

            // Entangling gates (ZZ interactions)
            for qubit in 0..num_qubits - 1 {
                let angle = data_point[qubit % data_point.len()]
                    * data_point[(qubit + 1) % data_point.len()];
                let zz_gate = self.zz_gate(angle);
                circuit =
                    self.apply_two_qubit_gate(&circuit, &zz_gate, qubit, qubit + 1, num_qubits)?;
            }
        }

        Ok(circuit)
    }

    /// Create Pauli feature map circuit
    fn create_pauli_feature_map(
        &self,
        data_point: &Array1<f64>,
        paulis: &[String],
        depth: usize,
    ) -> Result<Array2<Complex64>, QuantRS2Error> {
        let num_qubits = paulis.len();
        let dim = 2_usize.pow(num_qubits as u32);
        let mut circuit = Array2::eye(dim);

        for _layer in 0..depth {
            for (i, pauli_string) in paulis.iter().enumerate() {
                let angle = data_point[i % data_point.len()];
                let pauli_rotation = self.pauli_rotation(pauli_string, angle)?;
                circuit = circuit.dot(&pauli_rotation);
            }
        }

        Ok(circuit)
    }

    /// Helper: RY rotation gate
    fn ry_gate(&self, angle: f64) -> Array2<Complex64> {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        scirs2_core::ndarray::array![
            [
                Complex64::new(cos_half, 0.0),
                Complex64::new(-sin_half, 0.0)
            ],
            [Complex64::new(sin_half, 0.0), Complex64::new(cos_half, 0.0)]
        ]
    }

    /// Helper: ZZ interaction gate
    fn zz_gate(&self, angle: f64) -> Array2<Complex64> {
        let exp_factor = Complex64::from_polar(1.0, angle / 2.0);

        scirs2_core::ndarray::array![
            [
                exp_factor.conj(),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                exp_factor,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                exp_factor,
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                exp_factor.conj()
            ]
        ]
    }

    /// Helper: Pauli rotation gate
    fn pauli_rotation(
        &self,
        pauli_string: &str,
        angle: f64,
    ) -> Result<Array2<Complex64>, QuantRS2Error> {
        // Simplified implementation without matrix_exp for now
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        match pauli_string {
            "X" => Ok(scirs2_core::ndarray::array![
                [
                    Complex64::new(cos_half, 0.0),
                    Complex64::new(0.0, -sin_half)
                ],
                [
                    Complex64::new(0.0, -sin_half),
                    Complex64::new(cos_half, 0.0)
                ]
            ]),
            "Y" => Ok(scirs2_core::ndarray::array![
                [
                    Complex64::new(cos_half, 0.0),
                    Complex64::new(-sin_half, 0.0)
                ],
                [Complex64::new(sin_half, 0.0), Complex64::new(cos_half, 0.0)]
            ]),
            "Z" => Ok(scirs2_core::ndarray::array![
                [
                    Complex64::new(cos_half, -sin_half),
                    Complex64::new(0.0, 0.0)
                ],
                [Complex64::new(0.0, 0.0), Complex64::new(cos_half, sin_half)]
            ]),
            _ => Err(QuantRS2Error::InvalidGateOp(format!(
                "Unknown Pauli string: {pauli_string}"
            ))),
        }
    }

    /// Helper: Pauli matrices
    fn pauli_x(&self) -> Array2<Complex64> {
        scirs2_core::ndarray::array![
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
        ]
    }

    fn pauli_y(&self) -> Array2<Complex64> {
        scirs2_core::ndarray::array![
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
            [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)]
        ]
    }

    fn pauli_z(&self) -> Array2<Complex64> {
        scirs2_core::ndarray::array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
        ]
    }

    /// Apply single-qubit gate to circuit
    fn apply_single_qubit_gate(
        &self,
        circuit: &Array2<Complex64>,
        gate: &Array2<Complex64>,
        target_qubit: usize,
        num_qubits: usize,
    ) -> Result<Array2<Complex64>, QuantRS2Error> {
        let mut full_gate = Array2::eye(2_usize.pow(num_qubits as u32));

        // Tensor product construction
        for i in 0..num_qubits {
            let local_gate = if i == target_qubit {
                gate.clone()
            } else {
                Array2::eye(2)
            };

            if i == 0 {
                full_gate = local_gate;
            } else {
                let gate_matrix = DenseMatrix::new(local_gate)?;
                let full_gate_matrix = DenseMatrix::new(full_gate)?;
                full_gate = full_gate_matrix.tensor_product(&gate_matrix)?;
            }
        }

        Ok(circuit.dot(&full_gate))
    }

    /// Apply two-qubit gate to circuit
    fn apply_two_qubit_gate(
        &self,
        circuit: &Array2<Complex64>,
        gate: &Array2<Complex64>,
        _control: usize,
        _target: usize,
        _num_qubits: usize,
    ) -> Result<Array2<Complex64>, QuantRS2Error> {
        // This is a simplified implementation
        // In practice, would need proper tensor product construction
        Ok(circuit.dot(gate))
    }

    /// Optimize kernel parameters
    pub fn optimize_kernel_parameters(
        &mut self,
        training_data: &Array2<f64>,
        training_labels: &Array1<f64>,
    ) -> Result<OptimizationResult, QuantRS2Error> {
        // Clone data for closure
        let _training_data_clone = training_data.clone();
        let _training_labels_clone = training_labels.clone();

        let objective = |params: &Array1<f64>| -> Result<f64, String> {
            // Simplified loss computation (in practice would compute actual kernel)
            let loss = params.iter().map(|x| x * x).sum::<f64>();
            Ok(loss)
        };

        let initial_params = Array1::ones(4); // Example parameter count
        let config = OptimizationConfig::default();

        let result = minimize(objective, &initial_params, &config).map_err(|e| {
            QuantRS2Error::OptimizationFailed(format!("Kernel optimization failed: {e:?}"))
        })?;

        // Update the feature map parameters with optimized values
        self.feature_map_parameters.clone_from(&result.parameters);

        Ok(result)
    }

    /// Update feature map parameters
    const fn update_feature_map_parameters(&self, _params: &Array1<f64>) {
        // Implementation depends on specific feature map type
        // Would update angles, depths, etc.
    }

    /// Compute classification loss
    fn compute_classification_loss(&self, kernel: &Array2<f64>, labels: &Array1<f64>) -> f64 {
        // Simple SVM-like loss
        let n = labels.len();
        let mut loss = 0.0;

        for i in 0..n {
            for j in 0..n {
                loss += labels[i] * labels[j] * kernel[[i, j]];
            }
        }

        -loss / (n as f64)
    }
}

/// Hardware-efficient quantum ML layer
#[derive(Debug, Clone)]
pub struct HardwareEfficientMLLayer {
    pub num_qubits: usize,
    pub num_layers: usize,
    pub parameters: Array1<f64>,
    pub entanglement_pattern: EntanglementPattern,
}

#[derive(Debug, Clone)]
pub enum EntanglementPattern {
    Linear,
    Circular,
    AllToAll,
    Custom(Vec<(usize, usize)>),
}

impl HardwareEfficientMLLayer {
    /// Create a new hardware-efficient ML layer
    pub fn new(
        num_qubits: usize,
        num_layers: usize,
        entanglement_pattern: EntanglementPattern,
    ) -> Self {
        let num_params = num_qubits * num_layers * 3; // 3 rotation angles per qubit per layer
        Self {
            num_qubits,
            num_layers,
            parameters: Array1::zeros(num_params),
            entanglement_pattern,
        }
    }

    /// Initialize parameters randomly
    pub fn initialize_parameters(&mut self, rng: &mut impl scirs2_core::random::Rng) {
        use scirs2_core::random::prelude::*;
        for param in &mut self.parameters {
            *param = rng.random_range(-PI..PI);
        }
    }

    /// Build the quantum circuit
    pub fn build_circuit(&self) -> Result<Array2<Complex64>, QuantRS2Error> {
        let dim = 2_usize.pow(self.num_qubits as u32);
        let mut circuit = Array2::eye(dim);

        let mut param_idx = 0;
        for layer in 0..self.num_layers {
            // Rotation layer
            for qubit in 0..self.num_qubits {
                let rx_angle = self.parameters[param_idx];
                let ry_angle = self.parameters[param_idx + 1];
                let rz_angle = self.parameters[param_idx + 2];
                param_idx += 3;

                // Apply RX, RY, RZ rotations
                let rotation_gates = self.create_rotation_sequence(rx_angle, ry_angle, rz_angle);
                circuit = self.apply_rotation_to_circuit(&circuit, &rotation_gates, qubit)?;
            }

            // Entanglement layer
            if layer < self.num_layers - 1 {
                circuit = self.apply_entanglement_layer(&circuit)?;
            }
        }

        Ok(circuit)
    }

    /// Create rotation sequence for a qubit
    fn create_rotation_sequence(&self, rx: f64, ry: f64, rz: f64) -> Vec<Array2<Complex64>> {
        vec![self.rx_gate(rx), self.ry_gate(ry), self.rz_gate(rz)]
    }

    /// RX rotation gate
    fn rx_gate(&self, angle: f64) -> Array2<Complex64> {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        scirs2_core::ndarray::array![
            [
                Complex64::new(cos_half, 0.0),
                Complex64::new(0.0, -sin_half)
            ],
            [
                Complex64::new(0.0, -sin_half),
                Complex64::new(cos_half, 0.0)
            ]
        ]
    }

    /// RY rotation gate
    fn ry_gate(&self, angle: f64) -> Array2<Complex64> {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        scirs2_core::ndarray::array![
            [
                Complex64::new(cos_half, 0.0),
                Complex64::new(-sin_half, 0.0)
            ],
            [Complex64::new(sin_half, 0.0), Complex64::new(cos_half, 0.0)]
        ]
    }

    /// RZ rotation gate
    fn rz_gate(&self, angle: f64) -> Array2<Complex64> {
        let exp_factor = Complex64::from_polar(1.0, angle / 2.0);

        scirs2_core::ndarray::array![
            [exp_factor.conj(), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), exp_factor]
        ]
    }

    /// Apply rotation gates to circuit
    fn apply_rotation_to_circuit(
        &self,
        circuit: &Array2<Complex64>,
        rotations: &[Array2<Complex64>],
        qubit: usize,
    ) -> Result<Array2<Complex64>, QuantRS2Error> {
        let mut result = circuit.clone();
        for rotation in rotations {
            // Create tensor product gate for multi-qubit system
            let full_gate = self.create_single_qubit_gate(rotation, qubit)?;
            result = result.dot(&full_gate);
        }
        Ok(result)
    }

    /// Create single-qubit gate for multi-qubit system
    fn create_single_qubit_gate(
        &self,
        gate: &Array2<Complex64>,
        target_qubit: usize,
    ) -> Result<Array2<Complex64>, QuantRS2Error> {
        let dim = 2_usize.pow(self.num_qubits as u32);
        let mut full_gate = Array2::eye(dim);

        // Apply gate to target qubit in multi-qubit system
        for i in 0..dim {
            let target_bit = (i >> target_qubit) & 1;
            if target_bit == 0 {
                let j = i | (1 << target_qubit);
                if j < dim {
                    full_gate[[i, i]] = gate[[0, 0]];
                    full_gate[[j, i]] = gate[[1, 0]];
                }
            } else {
                let j = i & !(1 << target_qubit);
                if j < dim {
                    full_gate[[j, i]] = gate[[0, 1]];
                    full_gate[[i, i]] = gate[[1, 1]];
                }
            }
        }

        Ok(full_gate)
    }

    /// Apply entanglement layer
    fn apply_entanglement_layer(
        &self,
        circuit: &Array2<Complex64>,
    ) -> Result<Array2<Complex64>, QuantRS2Error> {
        let mut result = circuit.clone();

        let entangling_pairs = match &self.entanglement_pattern {
            EntanglementPattern::Linear => (0..self.num_qubits - 1).map(|i| (i, i + 1)).collect(),
            EntanglementPattern::Circular => {
                let mut pairs: Vec<(usize, usize)> =
                    (0..self.num_qubits - 1).map(|i| (i, i + 1)).collect();
                if self.num_qubits > 2 {
                    pairs.push((self.num_qubits - 1, 0));
                }
                pairs
            }
            EntanglementPattern::AllToAll => {
                let mut pairs = Vec::new();
                for i in 0..self.num_qubits {
                    for j in i + 1..self.num_qubits {
                        pairs.push((i, j));
                    }
                }
                pairs
            }
            EntanglementPattern::Custom(pairs) => pairs.clone(),
        };

        for (control, target) in entangling_pairs {
            let cnot = self.cnot_gate();
            result = self.apply_cnot_to_circuit(&result, &cnot, control, target)?;
        }

        Ok(result)
    }

    /// CNOT gate
    fn cnot_gate(&self) -> Array2<Complex64> {
        scirs2_core::ndarray::array![
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
        ]
    }

    /// Apply CNOT to circuit
    fn apply_cnot_to_circuit(
        &self,
        circuit: &Array2<Complex64>,
        cnot: &Array2<Complex64>,
        _control: usize,
        _target: usize,
    ) -> Result<Array2<Complex64>, QuantRS2Error> {
        // Simplified implementation
        Ok(circuit.dot(cnot))
    }

    /// Compute expectation value of observable
    pub fn expectation_value(
        &self,
        observable: &Array2<Complex64>,
        input_state: &Array1<Complex64>,
    ) -> Result<f64, QuantRS2Error> {
        let circuit = self.build_circuit()?;
        let output_state = circuit.dot(input_state);
        let expectation = output_state.t().dot(&observable.dot(&output_state));
        Ok(expectation.re)
    }

    /// Update parameters using gradient
    pub fn update_parameters(&mut self, gradient: &Array1<f64>, learning_rate: f64) {
        self.parameters = &self.parameters - learning_rate * gradient;
    }
}

/// Tensor network-based quantum ML accelerator
#[derive(Debug)]
pub struct TensorNetworkMLAccelerator {
    pub tensor_network: TensorNetwork,
    pub bond_dimensions: Vec<usize>,
    pub contraction_order: Vec<usize>,
}

impl TensorNetworkMLAccelerator {
    /// Create a new tensor network ML accelerator
    pub fn new(num_qubits: usize, max_bond_dimension: usize) -> Self {
        let network = TensorNetwork::new();
        let bond_dimensions = vec![max_bond_dimension; num_qubits];

        Self {
            tensor_network: network,
            bond_dimensions,
            contraction_order: (0..num_qubits).collect(),
        }
    }

    /// Decompose quantum circuit into tensor network
    pub fn decompose_circuit(&mut self, circuit: &Array2<Complex64>) -> Result<(), QuantRS2Error> {
        // SVD decomposition for circuit compression
        let (u, s, vt) = decompose_svd(circuit)?;

        // Create tensors from SVD components
        let tensor_u = Tensor::from_array(u, vec![0, 1]);
        let s_complex: Array2<Complex64> = s
            .diag()
            .insert_axis(Axis(1))
            .mapv(|x| Complex64::new(x, 0.0))
            .to_owned();
        let tensor_s = Tensor::from_array(s_complex, vec![1, 2]);
        let tensor_vt = Tensor::from_array(vt, vec![2, 3]);

        // Add tensors to network
        self.tensor_network.add_tensor(tensor_u);
        self.tensor_network.add_tensor(tensor_s);
        self.tensor_network.add_tensor(tensor_vt);

        Ok(())
    }

    /// Optimize tensor network contraction
    pub fn optimize_contraction(&mut self) -> Result<(), QuantRS2Error> {
        // Use dynamic programming to find optimal contraction order
        let n_tensors = self.tensor_network.tensors().len();

        if n_tensors <= 1 {
            return Ok(());
        }

        // Simple greedy optimization - can be improved
        self.contraction_order = (0..n_tensors).collect();
        self.contraction_order
            .sort_by_key(|&i| self.tensor_network.tensors()[i].tensor().ndim());

        Ok(())
    }

    /// Contract tensor network for efficient computation
    pub fn contract_network(&self) -> Result<Array2<Complex64>, QuantRS2Error> {
        if self.tensor_network.tensors().is_empty() {
            return Err(QuantRS2Error::TensorNetwork(
                "Empty tensor network".to_string(),
            ));
        }

        // Simplified contraction - full implementation would follow contraction_order
        let first_tensor = &self.tensor_network.tensors()[0];
        let result = first_tensor.tensor().clone();

        // Convert to proper shape for circuit representation
        let sqrt_dim = (result.len() as f64).sqrt() as usize;
        if sqrt_dim * sqrt_dim != result.len() {
            return Err(QuantRS2Error::TensorNetwork(
                "Invalid tensor dimensions".to_string(),
            ));
        }

        Ok(result
            .into_shape_with_order((sqrt_dim, sqrt_dim))
            .map_err(|e| QuantRS2Error::TensorNetwork(format!("Shape error: {e}")))?)
    }

    /// Estimate computational complexity
    pub fn complexity_estimate(&self) -> (usize, usize) {
        let time_complexity = self.bond_dimensions.iter().product();
        let space_complexity = self.bond_dimensions.iter().sum();
        (time_complexity, space_complexity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::random::prelude::*;

    #[test]
    fn test_quantum_natural_gradient() {
        let mut qng = QuantumNaturalGradient::new(2, 1e-6);
        let params = array![PI / 4.0, PI / 3.0];
        let state = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

        let circuit_gen =
            |_p: &Array1<f64>| -> Result<Array2<Complex64>, QuantRS2Error> { Ok(Array2::eye(2)) };

        let result = qng.compute_fisher_information(circuit_gen, &params, &state);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parameter_shift_optimizer() {
        let optimizer = ParameterShiftOptimizer::default();
        let params = array![PI / 4.0, PI / 3.0];

        let expectation_fn =
            |p: &Array1<f64>| -> Result<f64, QuantRS2Error> { Ok(p[0].cos() + p[1].sin()) };

        let gradient = optimizer.compute_gradient(expectation_fn, &params);
        assert!(gradient.is_ok());
        assert_eq!(
            gradient.expect("gradient computation should succeed").len(),
            2
        );
    }

    #[test]
    fn test_quantum_kernel_optimizer() {
        let feature_map = QuantumFeatureMap::ZZFeatureMap {
            num_qubits: 2,
            depth: 1,
        };
        let mut optimizer = QuantumKernelOptimizer::new(feature_map);

        let data = array![[0.1, 0.2], [0.3, 0.4]];
        let result = optimizer.compute_kernel_matrix(&data);

        assert!(result.is_ok());
        let kernel = result.expect("kernel matrix computation should succeed");
        assert_eq!(kernel.shape(), &[2, 2]);
    }

    #[test]
    fn test_hardware_efficient_ml_layer() {
        let mut layer = HardwareEfficientMLLayer::new(2, 2, EntanglementPattern::Linear);

        let mut rng = thread_rng();
        layer.initialize_parameters(&mut rng);

        let circuit = layer.build_circuit();
        assert!(circuit.is_ok());

        let observable = array![
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0)
            ]
        ];
        let state = array![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0)
        ];

        let expectation = layer.expectation_value(&observable, &state);
        assert!(expectation.is_ok());
    }

    #[test]
    fn test_decompose_svd_reconstructs_matrix() {
        // A known 2x2 real matrix [[3,0],[4,5]] embedded in Complex64.
        // Its singular values are sqrt of eigenvalues of MᵀM = [[25,20],[20,25]],
        // i.e. eigenvalues 45 and 5 -> singular values sqrt(45) and sqrt(5).
        let m = array![
            [Complex64::new(3.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(4.0, 0.0), Complex64::new(5.0, 0.0)]
        ];

        let (u, s, vt) = decompose_svd(&m).expect("SVD should succeed");

        // Singular values in descending order: sqrt(45), sqrt(5).
        assert_eq!(s.len(), 2);
        assert!(
            (s[0] - 45.0_f64.sqrt()).abs() < 1e-9,
            "largest singular value: got {}",
            s[0]
        );
        assert!(
            (s[1] - 5.0_f64.sqrt()).abs() < 1e-9,
            "smallest singular value: got {}",
            s[1]
        );
        assert!(s[0] >= s[1], "singular values must be descending");

        // Reconstruct M = U diag(S) Vt and compare element-wise.
        let mut sigma = Array2::<Complex64>::zeros((s.len(), s.len()));
        for (i, &sv) in s.iter().enumerate() {
            sigma[[i, i]] = Complex64::new(sv, 0.0);
        }
        let reconstructed = u.dot(&sigma).dot(&vt);
        for ((r, c), expected) in m.indexed_iter() {
            let got = reconstructed[[r, c]];
            assert!(
                (got - expected).norm() < 1e-9,
                "reconstruction[{r},{c}] = {got:?}, expected {expected:?}"
            );
        }

        // U must NOT be the identity for this non-diagonal matrix (guards against
        // the previous fabrication that returned eye()).
        let identity = Array2::<Complex64>::eye(2);
        let diff_from_identity: f64 = u
            .iter()
            .zip(identity.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum();
        assert!(
            diff_from_identity > 1e-6,
            "U must not be the identity for a non-diagonal input"
        );

        // U should have orthonormal columns: UᴴU = I.
        let u_dag = u.t().mapv(|z| z.conj());
        let utu = u_dag.dot(&u);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (utu[[i, j]].re - expected).abs() < 1e-9 && utu[[i, j]].im.abs() < 1e-9,
                    "UᴴU not identity at [{i},{j}]: {:?}",
                    utu[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_minimize_actually_reduces_cost() {
        // A real optimiser must lower a quadratic bowl, not echo the input.
        let objective =
            |p: &Array1<f64>| -> Result<f64, String> { Ok(p.iter().map(|x| x * x).sum::<f64>()) };
        let initial = array![1.5, -2.0, 0.75];
        let initial_cost = objective(&initial).expect("eval");
        let config = OptimizationConfig::default();

        let result = minimize(objective, &initial, &config).expect("minimize");

        assert!(
            result.cost < initial_cost,
            "optimiser must reduce cost: {} !< {}",
            result.cost,
            initial_cost
        );
        // It should approach the true minimum (origin) reasonably well.
        assert!(
            result.cost < 1e-2,
            "optimiser should converge near the minimum, got cost {}",
            result.cost
        );
        // It must have done more than a single no-op iteration.
        assert!(result.iterations > 1, "optimiser must actually iterate");
        // The returned parameters must differ from the initial ones.
        let moved: f64 = result
            .parameters
            .iter()
            .zip(initial.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            moved > 1e-6,
            "parameters must move away from the initial guess"
        );
    }

    #[test]
    fn test_tensor_network_ml_accelerator() {
        let mut accelerator = TensorNetworkMLAccelerator::new(2, 4);

        let circuit = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
        ];

        let result = accelerator.decompose_circuit(&circuit);
        assert!(result.is_ok());

        let optimization = accelerator.optimize_contraction();
        assert!(optimization.is_ok());

        let (time_comp, space_comp) = accelerator.complexity_estimate();
        assert!(time_comp > 0);
        assert!(space_comp > 0);
    }
}
