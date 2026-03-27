//! Quantum Support Vector Machine (QSVM) implementation
//!
//! This module implements quantum-enhanced support vector machines for
//! classification tasks using quantum feature maps and kernel methods.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Quantum feature map types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureMapType {
    /// Pauli-Z feature map: exp(i·φ(x)·Z)
    ZFeatureMap,
    /// Pauli-ZZ feature map: exp(i·φ(x)·ZZ)
    ZZFeatureMap,
    /// Pauli feature map (general)
    PauliFeatureMap,
    /// Custom angle encoding
    AngleEncoding,
    /// Amplitude encoding
    AmplitudeEncoding,
}

/// Parameters for QSVM
#[derive(Debug, Clone)]
pub struct QSVMParams {
    /// Type of quantum feature map
    pub feature_map: FeatureMapType,
    /// Number of repetitions of the feature map circuit
    pub reps: usize,
    /// Regularization parameter (also accessible as c_parameter for compatibility)
    pub c: f64,
    /// Tolerance for convergence
    pub tolerance: f64,
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub depth: usize,
    /// Gamma parameter for RBF-like kernels
    pub gamma: Option<f64>,
    /// Regularization parameter (alias for c)
    pub regularization: f64,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for QSVMParams {
    fn default() -> Self {
        Self {
            feature_map: FeatureMapType::ZZFeatureMap,
            reps: 2,
            c: 1.0,
            tolerance: 1e-3,
            num_qubits: 4,
            depth: 2,
            gamma: None,
            regularization: 1.0,
            max_iterations: 1000,
            seed: None,
        }
    }
}

impl QSVMParams {
    /// Get c_parameter (alias for c)
    pub fn c_parameter(&self) -> f64 {
        self.c
    }

    /// Set c_parameter (updates both c and regularization)
    pub fn set_c_parameter(&mut self, value: f64) {
        self.c = value;
        self.regularization = value;
    }
}

/// Quantum kernel computation
pub struct QuantumKernel {
    feature_map: FeatureMapType,
    reps: usize,
}

impl QuantumKernel {
    /// Create a new quantum kernel
    pub fn new(feature_map: FeatureMapType, reps: usize) -> Self {
        Self { feature_map, reps }
    }

    /// Compute the quantum kernel between two data points
    pub fn compute(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
        match self.feature_map {
            FeatureMapType::ZFeatureMap => self.z_feature_map_kernel(x1, x2),
            FeatureMapType::ZZFeatureMap => self.zz_feature_map_kernel(x1, x2),
            FeatureMapType::PauliFeatureMap => self.zz_feature_map_kernel(x1, x2), // Use ZZ as fallback
            FeatureMapType::AngleEncoding => self.angle_encoding_kernel(x1, x2),
            FeatureMapType::AmplitudeEncoding => self.amplitude_encoding_kernel(x1, x2),
        }
    }

    /// Z feature map kernel
    fn z_feature_map_kernel(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
        let n = x1.len();
        let mut kernel_val = 1.0;

        for _ in 0..self.reps {
            for i in 0..n {
                let phase_diff = (x1[i] - x2[i]) * PI;
                kernel_val *= phase_diff.cos();
            }
        }

        kernel_val
    }

    /// ZZ feature map kernel (includes entanglement)
    fn zz_feature_map_kernel(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
        let n = x1.len();
        let mut kernel_val = 1.0;

        for rep in 0..self.reps {
            // Single-qubit rotations
            for i in 0..n {
                let phase_diff = (x1[i] - x2[i]) * PI * (rep + 1) as f64;
                kernel_val *= phase_diff.cos();
            }

            // Two-qubit interactions
            for i in 0..n - 1 {
                let interaction = (x1[i] - x2[i]) * (x1[i + 1] - x2[i + 1]) * PI;
                kernel_val *= interaction.cos();
            }
        }

        kernel_val
    }

    /// Angle encoding kernel
    fn angle_encoding_kernel(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x1.len() {
            sum += (x1[i] - x2[i]).powi(2);
        }
        (-sum / 2.0).exp()
    }

    /// Amplitude encoding kernel
    fn amplitude_encoding_kernel(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
        // Normalize vectors
        let norm1 = x1.dot(x1).sqrt();
        let norm2 = x2.dot(x2).sqrt();

        if norm1 < 1e-10 || norm2 < 1e-10 {
            return 0.0;
        }

        let x1_norm = x1 / norm1;
        let x2_norm = x2 / norm2;

        // Inner product gives fidelity
        x1_norm.dot(&x2_norm).powi(2)
    }

    /// Compute the full kernel matrix for a dataset
    pub fn compute_kernel_matrix(&self, data: &Array2<f64>) -> Array2<f64> {
        let n_samples = data.nrows();
        let mut kernel_matrix = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i..n_samples {
                let kernel_val = self.compute(&data.row(i).to_owned(), &data.row(j).to_owned());
                kernel_matrix[[i, j]] = kernel_val;
                kernel_matrix[[j, i]] = kernel_val; // Symmetric
            }
        }

        kernel_matrix
    }
}

/// Quantum Support Vector Machine classifier
pub struct QSVM {
    params: QSVMParams,
    kernel: QuantumKernel,
    support_vectors: Option<Array2<f64>>,
    support_labels: Option<Array1<i32>>,
    alphas: Option<Array1<f64>>,
    bias: f64,
    kernel_matrix_cache: HashMap<(usize, usize), f64>,
}

impl QSVM {
    /// Create a new QSVM classifier
    pub fn new(params: QSVMParams) -> Self {
        let kernel = QuantumKernel::new(params.feature_map, params.reps);
        Self {
            params,
            kernel,
            support_vectors: None,
            support_labels: None,
            alphas: None,
            bias: 0.0,
            kernel_matrix_cache: HashMap::new(),
        }
    }

    /// Train the QSVM using SMO (Sequential Minimal Optimization)
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> Result<(), String> {
        let n_samples = x.nrows();

        // Validate labels are binary (-1 or 1)
        for &label in y.iter() {
            if label != -1 && label != 1 {
                return Err("QSVM requires binary labels: -1 or 1".to_string());
            }
        }

        // Initialize alphas
        let mut alphas = Array1::zeros(n_samples);

        // Precompute kernel matrix
        let kernel_matrix = self.kernel.compute_kernel_matrix(x);

        // SMO optimization
        let mut converged = false;
        let mut iter = 0;

        while !converged && iter < self.params.max_iterations {
            let old_alphas = alphas.clone();

            // Select working set (simplified SMO)
            for i in 0..n_samples {
                // Compute error for i
                let ei = self.compute_error(&kernel_matrix, &alphas, y, i);

                // Check KKT conditions
                if !self.check_kkt(alphas[i], y[i], ei) {
                    // Select j != i
                    let j = self.select_second_alpha(i, n_samples, &kernel_matrix, &alphas, y);
                    if i == j {
                        continue;
                    }

                    // Compute error for j
                    let ej = self.compute_error(&kernel_matrix, &alphas, y, j);

                    // Save old alphas
                    let alpha_i_old = alphas[i];
                    let alpha_j_old = alphas[j];

                    // Compute bounds
                    let (l, h) = self.compute_bounds(y[i], y[j], alphas[i], alphas[j]);

                    if (h - l).abs() < 1e-10 {
                        continue;
                    }

                    // Compute eta
                    let eta =
                        kernel_matrix[[i, i]] + kernel_matrix[[j, j]] - 2.0 * kernel_matrix[[i, j]];

                    if eta <= 0.0 {
                        continue;
                    }

                    // Update alpha_j
                    alphas[j] += y[j] as f64 * (ei - ej) / eta;
                    alphas[j] = alphas[j].clamp(l, h);

                    if (alphas[j] - alpha_j_old).abs() < 1e-5 {
                        continue;
                    }

                    // Update alpha_i
                    alphas[i] += y[i] as f64 * y[j] as f64 * (alpha_j_old - alphas[j]);
                }
            }

            // Check convergence
            let alpha_change: f64 = (&alphas - &old_alphas).mapv(|a| a.abs()).sum();
            converged = alpha_change < self.params.tolerance;
            iter += 1;
        }

        // Extract support vectors
        let mut support_indices = Vec::new();
        let mut support_alphas = Vec::new();

        for i in 0..n_samples {
            if alphas[i] > 1e-5 {
                support_indices.push(i);
                support_alphas.push(alphas[i]);
            }
        }

        if support_indices.is_empty() {
            return Err("No support vectors found".to_string());
        }

        // Store support vectors
        let n_support = support_indices.len();
        let n_features = x.ncols();
        let mut support_vectors = Array2::zeros((n_support, n_features));
        let mut support_labels = Array1::zeros(n_support);

        for (idx, &i) in support_indices.iter().enumerate() {
            support_vectors.row_mut(idx).assign(&x.row(i));
            support_labels[idx] = y[i];
        }

        self.support_vectors = Some(support_vectors);
        self.support_labels = Some(support_labels);
        self.alphas = Some(Array1::from_vec(support_alphas));

        // Compute bias
        self.compute_bias(&kernel_matrix, &alphas, y, &support_indices);

        Ok(())
    }

    /// Compute error for a given sample
    fn compute_error(
        &self,
        kernel_matrix: &Array2<f64>,
        alphas: &Array1<f64>,
        y: &Array1<i32>,
        i: usize,
    ) -> f64 {
        let mut sum = self.bias;
        for j in 0..alphas.len() {
            if alphas[j] > 0.0 {
                sum += alphas[j] * y[j] as f64 * kernel_matrix[[i, j]];
            }
        }
        sum - y[i] as f64
    }

    /// Check KKT conditions
    fn check_kkt(&self, alpha: f64, y: i32, error: f64) -> bool {
        let y_error = y as f64 * error;

        if alpha < 1e-5 {
            y_error >= -self.params.tolerance
        } else if alpha > self.params.c - 1e-5 {
            y_error <= self.params.tolerance
        } else {
            (y_error).abs() <= self.params.tolerance
        }
    }

    /// Select second alpha using maximum step heuristic
    fn select_second_alpha(
        &self,
        i: usize,
        n_samples: usize,
        kernel_matrix: &Array2<f64>,
        alphas: &Array1<f64>,
        y: &Array1<i32>,
    ) -> usize {
        let ei = self.compute_error(kernel_matrix, alphas, y, i);
        let mut max_step = 0.0;
        let mut best_j = i;

        for j in 0..n_samples {
            if i == j {
                continue;
            }

            let ej = self.compute_error(kernel_matrix, alphas, y, j);
            let step = (ei - ej).abs();

            if step > max_step {
                max_step = step;
                best_j = j;
            }
        }

        best_j
    }

    /// Compute bounds for alpha updates
    fn compute_bounds(&self, yi: i32, yj: i32, alpha_i: f64, alpha_j: f64) -> (f64, f64) {
        if yi != yj {
            let l = (alpha_j - alpha_i).max(0.0);
            let h = (self.params.c + alpha_j - alpha_i).min(self.params.c);
            (l, h)
        } else {
            let l = (alpha_i + alpha_j - self.params.c).max(0.0);
            let h = (alpha_i + alpha_j).min(self.params.c);
            (l, h)
        }
    }

    /// Compute bias term
    fn compute_bias(
        &mut self,
        kernel_matrix: &Array2<f64>,
        alphas: &Array1<f64>,
        y: &Array1<i32>,
        support_indices: &[usize],
    ) {
        let mut bias_sum = 0.0;
        let mut count = 0;

        for &i in support_indices {
            if alphas[i] > 1e-5 && alphas[i] < self.params.c - 1e-5 {
                let mut sum = 0.0;
                for j in 0..alphas.len() {
                    if alphas[j] > 1e-5 {
                        sum += alphas[j] * y[j] as f64 * kernel_matrix[[i, j]];
                    }
                }
                bias_sum += y[i] as f64 - sum;
                count += 1;
            }
        }

        self.bias = if count > 0 {
            bias_sum / count as f64
        } else {
            0.0
        };
    }

    /// Predict labels for new data
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>, String> {
        let support_vectors = self.support_vectors.as_ref().ok_or("Model not trained")?;
        let support_labels = self.support_labels.as_ref().ok_or("Model not trained")?;
        let alphas = self.alphas.as_ref().ok_or("Model not trained")?;

        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut score = self.bias;

            for (j, sv) in support_vectors.rows().into_iter().enumerate() {
                let kernel_val = self.kernel.compute(&x.row(i).to_owned(), &sv.to_owned());
                score += alphas[j] * support_labels[j] as f64 * kernel_val;
            }

            predictions[i] = if score >= 0.0 { 1 } else { -1 };
        }

        Ok(predictions)
    }

    /// Get decision function values
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>, String> {
        let support_vectors = self.support_vectors.as_ref().ok_or("Model not trained")?;
        let support_labels = self.support_labels.as_ref().ok_or("Model not trained")?;
        let alphas = self.alphas.as_ref().ok_or("Model not trained")?;

        let n_samples = x.nrows();
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut score = self.bias;

            for (j, sv) in support_vectors.rows().into_iter().enumerate() {
                let kernel_val = self.kernel.compute(&x.row(i).to_owned(), &sv.to_owned());
                score += alphas[j] * support_labels[j] as f64 * kernel_val;
            }

            scores[i] = score;
        }

        Ok(scores)
    }

    /// Get the number of support vectors
    pub fn n_support_vectors(&self) -> usize {
        self.support_vectors
            .as_ref()
            .map(|sv| sv.nrows())
            .unwrap_or(0)
    }
}

/// Quantum kernel ridge regression for comparison
pub struct QuantumKernelRidge {
    kernel: QuantumKernel,
    alpha: f64,
    training_data: Option<Array2<f64>>,
    coefficients: Option<Array1<f64>>,
}

impl QuantumKernelRidge {
    /// Create new quantum kernel ridge regression
    pub fn new(feature_map: FeatureMapType, reps: usize, alpha: f64) -> Self {
        Self {
            kernel: QuantumKernel::new(feature_map, reps),
            alpha,
            training_data: None,
            coefficients: None,
        }
    }

    /// Fit the model
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        // Compute kernel matrix
        let mut k = self.kernel.compute_kernel_matrix(x);

        // Add regularization to diagonal
        let n = k.nrows();
        for i in 0..n {
            k[[i, i]] += self.alpha;
        }

        // Solve K * coefficients = y
        // Using simple matrix inversion (in practice, use Cholesky decomposition)
        match Self::solve_linear_system(&k, y) {
            Ok(coeffs) => {
                self.training_data = Some(x.clone());
                self.coefficients = Some(coeffs);
                Ok(())
            }
            Err(e) => Err(format!("Failed to solve linear system: {}", e)),
        }
    }

    /// Solve the linear system `A x = b` using Cholesky decomposition (LL^T factorisation).
    ///
    /// `A` is assumed to be symmetric positive-definite (the kernel matrix + regularisation
    /// diagonal satisfies this by construction).
    ///
    /// Algorithm:
    /// 1. Compute the lower-triangular Cholesky factor `L` such that `A = L Lᵀ`.
    /// 2. Forward-substitute: solve `L y = b`.
    /// 3. Backward-substitute: solve `Lᵀ x = y`.
    fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
        let n = a.nrows();
        if n == 0 {
            return Err("Empty matrix".to_string());
        }
        if n != a.ncols() {
            return Err(format!("Matrix is not square: {}×{}", n, a.ncols()));
        }
        if n != b.len() {
            return Err(format!(
                "Dimension mismatch: matrix is {}×{} but rhs has {} elements",
                n,
                n,
                b.len()
            ));
        }

        // ── Step 1: Cholesky factorisation A = L Lᵀ ─────────────────────────
        // L is stored as a flat row-major lower triangular matrix.
        let mut l = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            // Diagonal element
            let mut sum_sq = a[[i, i]];
            for k in 0..i {
                sum_sq -= l[[i, k]] * l[[i, k]];
            }
            if sum_sq <= 0.0 {
                // Matrix is not positive-definite; fall back to Gaussian elimination.
                return Self::solve_gaussian_elimination(a, b);
            }
            l[[i, i]] = sum_sq.sqrt();

            // Sub-diagonal elements for column i
            for j in (i + 1)..n {
                let mut sum_prod = a[[j, i]];
                for k in 0..i {
                    sum_prod -= l[[j, k]] * l[[i, k]];
                }
                l[[j, i]] = sum_prod / l[[i, i]];
            }
        }

        // ── Step 2: Forward substitution — solve L y = b ─────────────────────
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = b[i];
            for k in 0..i {
                s -= l[[i, k]] * y[k];
            }
            let diag = l[[i, i]];
            if diag.abs() < 1e-14 {
                return Err(format!("Singular Cholesky factor at row {i}"));
            }
            y[i] = s / diag;
        }

        // ── Step 3: Backward substitution — solve Lᵀ x = y ──────────────────
        let mut x = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in (i + 1)..n {
                s -= l[[k, i]] * x[k]; // l[[k,i]] = Lᵀ[[i,k]]
            }
            let diag = l[[i, i]];
            if diag.abs() < 1e-14 {
                return Err(format!("Singular Cholesky factor at column {i}"));
            }
            x[i] = s / diag;
        }

        Ok(x)
    }

    /// Fallback linear solver using Gaussian elimination with partial pivoting.
    ///
    /// Used when the matrix is not numerically positive-definite.
    fn solve_gaussian_elimination(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
        let n = a.nrows();

        // Build augmented matrix [A | b]
        let mut aug: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row: Vec<f64> = (0..n).map(|j| a[[i, j]]).collect();
                row.push(b[i]);
                row
            })
            .collect();

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let pivot_row = (col..n)
                .max_by(|&r1, &r2| {
                    aug[r1][col]
                        .abs()
                        .partial_cmp(&aug[r2][col].abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .ok_or("Empty column range during elimination")?;

            aug.swap(col, pivot_row);

            let pivot = aug[col][col];
            if pivot.abs() < 1e-14 {
                return Err(format!("Singular or near-singular matrix at column {col}"));
            }

            // Eliminate column entries below the pivot
            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for j in col..=n {
                    let val = aug[col][j];
                    aug[row][j] -= factor * val;
                }
            }
        }

        // Back-substitution
        let mut x = vec![0.0_f64; n];
        for i in (0..n).rev() {
            let mut s = aug[i][n];
            for j in (i + 1)..n {
                s -= aug[i][j] * x[j];
            }
            let diag = aug[i][i];
            if diag.abs() < 1e-14 {
                return Err(format!(
                    "Singular matrix during back-substitution at row {i}"
                ));
            }
            x[i] = s / diag;
        }

        Ok(Array1::from_vec(x))
    }

    /// Predict values for new data
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, String> {
        let training_data = self.training_data.as_ref().ok_or("Model not trained")?;
        let coefficients = self.coefficients.as_ref().ok_or("Model not trained")?;

        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut sum = 0.0;
            for (j, coeff) in coefficients.iter().enumerate() {
                let kernel_val = self
                    .kernel
                    .compute(&x.row(i).to_owned(), &training_data.row(j).to_owned());
                sum += coeff * kernel_val;
            }
            predictions[i] = sum;
        }

        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_quantum_kernel_computation() {
        let kernel = QuantumKernel::new(FeatureMapType::ZFeatureMap, 1);

        let x1 = array![0.5, 0.5];
        let x2 = array![0.5, 0.5];

        let kernel_val = kernel.compute(&x1, &x2);
        assert!((kernel_val - 1.0).abs() < 1e-6); // Same vectors should give 1

        let x3 = array![0.0, 1.0];
        let kernel_val2 = kernel.compute(&x1, &x3);
        assert!(kernel_val2 < 1.0); // Different vectors should give < 1
    }

    #[test]
    fn test_qsvm_basic() {
        // Create simple linearly separable dataset
        let x = array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [0.9, 0.9],];

        let y = array![-1, -1, 1, 1];

        let params = QSVMParams::default();
        let mut qsvm = QSVM::new(params);

        // Train
        qsvm.fit(&x, &y).expect("fit should succeed");

        // Check that we have support vectors
        assert!(qsvm.n_support_vectors() > 0);

        // Predict on training data
        let predictions = qsvm.predict(&x).expect("predict should succeed");

        // Should classify training data correctly
        for i in 0..y.len() {
            assert_eq!(predictions[i], y[i]);
        }
    }
}
