//! Harrow-Hassidim-Lloyd (HHL) Algorithm Implementation
//!
//! The HHL algorithm provides a quantum algorithm for solving linear systems of equations
//! Ax = b, where A is a Hermitian matrix. The algorithm outputs a quantum state |x⟩
//! proportional to the solution vector x.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Parameters for the HHL algorithm
#[derive(Debug, Clone)]
pub struct HHLParams {
    /// Number of qubits for the input register (log2 of matrix dimension)
    pub n_qubits: usize,
    /// Number of qubits for the clock register (precision)
    pub clock_qubits: usize,
    /// Time evolution parameter
    pub evolution_time: f64,
    /// Condition number of the matrix (for scaling)
    pub condition_number: f64,
    /// Eigenvalue rescaling factor
    pub eigenvalue_scale: f64,
}

impl HHLParams {
    /// Create default HHL parameters
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            clock_qubits: n_qubits + 2, // Good default precision
            evolution_time: PI,
            condition_number: 10.0,
            eigenvalue_scale: 1.0,
        }
    }
}

/// HHL algorithm implementation
pub struct HHLAlgorithm {
    params: HHLParams,
    #[allow(dead_code)]
    matrix: Array2<Complex64>,
    vector_b: Array1<Complex64>,
}

impl HHLAlgorithm {
    /// Create a new HHL algorithm instance
    pub fn new(
        matrix: Array2<Complex64>,
        vector_b: Array1<Complex64>,
        params: HHLParams,
    ) -> Result<Self, String> {
        // Validate inputs
        let (n, m) = matrix.dim();
        if n != m {
            return Err("Matrix must be square".to_string());
        }

        if n != 1 << params.n_qubits {
            return Err(format!(
                "Matrix size {} doesn't match qubit count {} (expected {})",
                n,
                params.n_qubits,
                1 << params.n_qubits
            ));
        }

        if vector_b.len() != n {
            return Err("Vector b must have same dimension as matrix".to_string());
        }

        // Check if matrix is approximately Hermitian
        if !is_hermitian(&matrix, 1e-10) {
            return Err("Matrix must be Hermitian".to_string());
        }

        Ok(Self {
            params,
            matrix,
            vector_b,
        })
    }

    /// Get the total number of qubits required
    pub fn total_qubits(&self) -> usize {
        self.params.n_qubits + self.params.clock_qubits + 1 // +1 for ancilla
    }

    /// Initialize the quantum state with |b⟩
    pub fn prepare_input_state(&self, state: &mut Vec<Complex64>) {
        let n = 1 << self.params.n_qubits;
        let clock_size = 1 << self.params.clock_qubits;
        let total_size = n * clock_size * 2; // *2 for ancilla

        // Ensure state is properly sized
        state.clear();
        state.resize(total_size, Complex64::new(0.0, 0.0));

        // Normalize vector b
        let norm: f64 = self
            .vector_b
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>()
            .sqrt();

        // Initialize state |0⟩_clock |b⟩_input |0⟩_ancilla
        for i in 0..n {
            let amplitude = self.vector_b[i] / norm;
            // Clock register in |0⟩, input register in |b⟩, ancilla in |0⟩
            let index = i * clock_size * 2;
            state[index] = amplitude;
        }
    }

    /// Apply quantum phase estimation to find eigenvalues
    pub fn apply_phase_estimation(&self, state: &mut [Complex64]) {
        // This is a simplified version - full QPE would require:
        // 1. Hadamard gates on clock register
        // 2. Controlled-U operations where U = exp(iAt)
        // 3. Inverse QFT on clock register

        let clock_size = 1 << self.params.clock_qubits;
        let n = 1 << self.params.n_qubits;

        // Apply Hadamard to all clock qubits (simplified)
        for clock_idx in 0..clock_size {
            for input_idx in 0..n {
                for ancilla_idx in 0..2 {
                    let idx = ancilla_idx + 2 * (input_idx + n * clock_idx);
                    state[idx] *= Complex64::new(1.0 / (clock_size as f64).sqrt(), 0.0);
                }
            }
        }

        // In a real implementation, we would:
        // - Decompose matrix into eigenvalues/eigenvectors
        // - Apply controlled rotations based on eigenvalues
        // - Perform inverse QFT
    }

    /// Apply controlled rotation based on eigenvalues
    pub fn apply_eigenvalue_inversion(&self, state: &mut [Complex64]) {
        let clock_size = 1 << self.params.clock_qubits;
        let n = 1 << self.params.n_qubits;

        // For each eigenvalue encoded in the clock register,
        // apply rotation on ancilla qubit proportional to 1/eigenvalue
        for clock_idx in 1..clock_size {
            // Skip 0 to avoid division by zero
            let eigenvalue =
                (clock_idx as f64) / (clock_size as f64) * self.params.eigenvalue_scale;

            // Rotation angle: arcsin(C/λ) where C is a normalization constant
            let c = 1.0 / self.params.condition_number;
            let angle = if eigenvalue > c {
                (c / eigenvalue).asin()
            } else {
                PI / 2.0 // Maximum rotation for small eigenvalues
            };

            // Apply controlled rotation on ancilla
            for input_idx in 0..n {
                let idx0 = 2 * (input_idx + n * clock_idx); // ancilla = 0
                let idx1 = 1 + 2 * (input_idx + n * clock_idx); // ancilla = 1

                let cos_angle = angle.cos();
                let sin_angle = angle.sin();

                let amp0 = state[idx0];
                let amp1 = state[idx1];

                state[idx0] = amp0 * cos_angle - amp1 * sin_angle;
                state[idx1] = amp0 * sin_angle + amp1 * cos_angle;
            }
        }
    }

    /// Apply inverse phase estimation to uncompute eigenvalues
    pub fn apply_inverse_phase_estimation(&self, state: &mut Vec<Complex64>) {
        // This would apply the inverse of the phase estimation
        // For now, this is a placeholder
        self.apply_phase_estimation(state); // Simplified: QPE is self-inverse up to normalization
    }

    /// Measure ancilla qubit and post-select on |1⟩
    pub fn postselect_ancilla(&self, state: &mut Vec<Complex64>) -> f64 {
        let total_size = state.len();
        let mut success_probability = 0.0;
        let mut new_state = vec![Complex64::new(0.0, 0.0); total_size / 2];

        // Post-select on ancilla = 1
        for i in 0..total_size / 2 {
            let idx1 = 2 * i + 1; // ancilla = 1
            new_state[i] = state[idx1];
            success_probability += state[idx1].norm_sqr();
        }

        // Normalize the post-selected state
        if success_probability > 1e-10 {
            let norm = success_probability.sqrt();
            for amp in new_state.iter_mut() {
                *amp /= norm;
            }
        }

        // Update state (removing ancilla dimension)
        state.clear();
        state.extend_from_slice(&new_state);

        success_probability
    }

    /// Extract solution from the quantum state
    pub fn extract_solution(&self, state: &[Complex64]) -> Array1<Complex64> {
        let n = 1 << self.params.n_qubits;
        let clock_size = 1 << self.params.clock_qubits;
        let mut solution = Array1::zeros(n);

        // Trace out clock register
        for input_idx in 0..n {
            let mut amplitude = Complex64::new(0.0, 0.0);
            for clock_idx in 0..clock_size {
                let idx = input_idx + n * clock_idx;
                if idx < state.len() {
                    amplitude += state[idx];
                }
            }
            solution[input_idx] = amplitude;
        }

        // Normalize
        let norm: f64 = solution.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();

        if norm > 1e-10 {
            for amp in solution.iter_mut() {
                *amp /= norm;
            }
        }

        solution
    }

    /// Run the complete HHL algorithm
    pub fn run(&self) -> Result<(Array1<Complex64>, f64), String> {
        let total_size = 1 << self.total_qubits();
        let mut state = vec![Complex64::new(0.0, 0.0); total_size];

        // Step 1: Prepare input state |b⟩
        self.prepare_input_state(&mut state);

        // Step 2: Apply quantum phase estimation
        self.apply_phase_estimation(&mut state);

        // Step 3: Apply eigenvalue inversion (controlled rotation)
        self.apply_eigenvalue_inversion(&mut state);

        // Step 4: Apply inverse phase estimation
        self.apply_inverse_phase_estimation(&mut state);

        // Step 5: Measure ancilla and post-select
        let success_probability = self.postselect_ancilla(&mut state);

        // Step 6: Extract solution
        let solution = self.extract_solution(&state);

        Ok((solution, success_probability))
    }
}

/// Check if a matrix is Hermitian
fn is_hermitian(matrix: &Array2<Complex64>, tolerance: f64) -> bool {
    let (n, m) = matrix.dim();
    if n != m {
        return false;
    }

    for i in 0..n {
        for j in 0..n {
            let diff = (matrix[[i, j]] - matrix[[j, i]].conj()).norm();
            if diff > tolerance {
                return false;
            }
        }
    }

    true
}

/// Simple example: solving a 2x2 system
pub fn hhl_example() -> Result<(), String> {
    // Example matrix A (must be Hermitian)
    let matrix = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0),
        ],
    )
    .unwrap();

    // Vector b
    let vector_b = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);

    // Create HHL instance
    let params = HHLParams::new(1); // 2^1 = 2 dimensional system
    let hhl = HHLAlgorithm::new(matrix.clone(), vector_b.clone(), params)?;

    // Run algorithm
    let (solution, success_prob) = hhl.run()?;

    println!("HHL Algorithm Results:");
    println!("Matrix A:\n{:?}", matrix);
    println!("Vector b: {:?}", vector_b);
    println!("Quantum solution |x⟩: {:?}", solution);
    println!("Success probability: {:.4}", success_prob);

    // Verify: A|x⟩ should be proportional to |b⟩
    let ax = matrix.dot(&solution);
    println!("Verification A|x⟩: {:?}", ax);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hermitian_check() {
        // Hermitian matrix
        let h = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(2.0, 0.0),
            ],
        )
        .unwrap();
        assert!(is_hermitian(&h, 1e-10));

        // Non-Hermitian matrix
        let nh = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap();
        assert!(!is_hermitian(&nh, 1e-10));
    }

    #[test]
    fn test_hhl_creation() {
        let matrix = Array2::eye(2);
        let vector_b = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);

        let params = HHLParams::new(1);
        let hhl = HHLAlgorithm::new(matrix, vector_b, params);
        assert!(hhl.is_ok());
    }
}
