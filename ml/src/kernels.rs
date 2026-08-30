//! Quantum kernel methods for support vector machines and kernel PCA.
//!
//! Computes quantum kernel matrices by estimating the overlap
//! ⟨φ(x)|φ(x′)⟩ between feature-map states encoded by parameterised
//! quantum circuits, enabling quantum-enhanced SVMs and kernel regression.

use crate::error::{MLError, Result};
use quantrs2_circuit::builder::Simulator;
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::ndarray::{Array1, Array2};

/// Kernel method for quantum machine learning
#[derive(Debug, Clone, Copy)]
pub enum KernelMethod {
    /// Linear kernel
    Linear,

    /// Polynomial kernel
    Polynomial,

    /// Radial basis function (RBF) kernel
    RBF,

    /// Quantum kernel
    QuantumKernel,

    /// Hybrid classical-quantum kernel
    HybridKernel,
}

/// Kernel function for machine learning
pub trait KernelFunction {
    /// Computes the kernel value for two vectors
    fn compute(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> Result<f64>;

    /// Computes the kernel matrix for a dataset
    fn compute_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let mut kernel_matrix = Array2::zeros((n, n));

        for i in 0..n {
            let x_i = x.row(i).to_owned();

            for j in 0..=i {
                let x_j = x.row(j).to_owned();

                let k_ij = self.compute(&x_i, &x_j)?;
                kernel_matrix[[i, j]] = k_ij;

                if i != j {
                    kernel_matrix[[j, i]] = k_ij; // Symmetric
                }
            }
        }

        Ok(kernel_matrix)
    }
}

/// Linear kernel for classical machine learning
#[derive(Debug, Clone)]
pub struct LinearKernel;

impl KernelFunction for LinearKernel {
    fn compute(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> Result<f64> {
        if x1.len() != x2.len() {
            return Err(MLError::InvalidParameter(format!(
                "Vector dimensions mismatch: {} != {}",
                x1.len(),
                x2.len()
            )));
        }

        let dot_product = x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum();

        Ok(dot_product)
    }
}

/// Polynomial kernel for classical machine learning
#[derive(Debug, Clone)]
pub struct PolynomialKernel {
    /// Degree of the polynomial
    pub degree: usize,

    /// Coefficient
    pub coef: f64,
}

impl PolynomialKernel {
    /// Creates a new polynomial kernel
    pub fn new(degree: usize, coef: f64) -> Self {
        PolynomialKernel { degree, coef }
    }
}

impl KernelFunction for PolynomialKernel {
    fn compute(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> Result<f64> {
        if x1.len() != x2.len() {
            return Err(MLError::InvalidParameter(format!(
                "Vector dimensions mismatch: {} != {}",
                x1.len(),
                x2.len()
            )));
        }

        let dot_product = x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum::<f64>();
        let value = (dot_product + self.coef).powi(self.degree as i32);

        Ok(value)
    }
}

/// Radial basis function (RBF) kernel for classical machine learning
#[derive(Debug, Clone)]
pub struct RBFKernel {
    /// Gamma parameter
    pub gamma: f64,
}

impl RBFKernel {
    /// Creates a new RBF kernel
    pub fn new(gamma: f64) -> Self {
        RBFKernel { gamma }
    }
}

impl KernelFunction for RBFKernel {
    fn compute(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> Result<f64> {
        if x1.len() != x2.len() {
            return Err(MLError::InvalidParameter(format!(
                "Vector dimensions mismatch: {} != {}",
                x1.len(),
                x2.len()
            )));
        }

        let squared_distance = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>();

        let value = (-self.gamma * squared_distance).exp();

        Ok(value)
    }
}

/// Quantum kernel for quantum machine learning
#[derive(Debug, Clone)]
pub struct QuantumKernel {
    /// Number of qubits
    pub num_qubits: usize,

    /// Feature dimension
    pub feature_dim: usize,

    /// Number of measurements to estimate the kernel
    pub num_measurements: usize,
}

impl QuantumKernel {
    /// Creates a new quantum kernel
    pub fn new(num_qubits: usize, feature_dim: usize) -> Self {
        QuantumKernel {
            num_qubits,
            feature_dim,
            num_measurements: 1000,
        }
    }

    /// Sets the number of measurements to estimate the kernel
    pub fn with_measurements(mut self, num_measurements: usize) -> Self {
        self.num_measurements = num_measurements;
        self
    }

    /// Encodes a feature vector into a quantum circuit
    fn encode_features<const N: usize>(
        &self,
        features: &Array1<f64>,
        circuit: &mut Circuit<N>,
    ) -> Result<()> {
        // This is a simplified implementation
        // In a real system, this would use more sophisticated feature encoding

        for i in 0..N.min(features.len()) {
            let angle = features[i] * std::f64::consts::PI;
            circuit.ry(i, angle)?;
        }

        Ok(())
    }

    /// Prepares a quantum circuit for kernel estimation
    fn prepare_kernel_circuit<const N: usize>(
        &self,
        x1: &Array1<f64>,
        x2: &Array1<f64>,
    ) -> Result<Circuit<N>> {
        let mut circuit = Circuit::<N>::new();

        // Apply Hadamard to all qubits
        for i in 0..N.min(self.num_qubits) {
            circuit.h(i)?;
        }

        // Encode the first feature vector
        self.encode_features(x1, &mut circuit)?;

        // Apply X gates as separators
        for i in 0..N.min(self.num_qubits) {
            circuit.x(i)?;
        }

        // Encode the second feature vector
        self.encode_features(x2, &mut circuit)?;

        // Apply Hadamard gates again
        for i in 0..N.min(self.num_qubits) {
            circuit.h(i)?;
        }

        Ok(circuit)
    }
}

impl QuantumKernel {
    /// Builds the kernel-estimation circuit on an `N`-qubit register, runs it
    /// on the real state-vector simulator, and returns the fidelity kernel
    /// value `|⟨0|U|0⟩|^2` where `U` is `H ⋅ encode(x2) ⋅ X ⋅ encode(x1) ⋅ H`
    /// (the standard SWAP-test-free construction: since `H` is self-inverse
    /// and the `X` gates only relabel which computational-basis amplitude
    /// carries the overlap, the probability of observing all zeros encodes
    /// the squared overlap between the two feature-map states).
    fn compute_sized<const N: usize>(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> Result<f64> {
        let circuit = self.prepare_kernel_circuit::<N>(x1, x2)?;
        let simulator = StateVectorSimulator::new();
        let register = simulator.run(&circuit)?;

        // Probability amplitude of the all-zeros computational basis state.
        let amplitude_zero = register.amplitudes()[0];
        Ok(amplitude_zero.norm_sqr().clamp(0.0, 1.0))
    }
}

impl KernelFunction for QuantumKernel {
    fn compute(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> Result<f64> {
        if x1.len() != x2.len() {
            return Err(MLError::InvalidParameter(format!(
                "Vector dimensions mismatch: {} != {}",
                x1.len(),
                x2.len()
            )));
        }

        if x1.len() != self.feature_dim {
            return Err(MLError::InvalidParameter(format!(
                "Feature dimension mismatch: {} != {}",
                x1.len(),
                self.feature_dim
            )));
        }

        // Estimate the kernel value by actually simulating the quantum
        // feature-map circuit (prepare_kernel_circuit/encode_features),
        // dispatching to the smallest supported register that can hold
        // `num_qubits`, matching the pattern used by QuantumNeuralNetwork.
        match self.num_qubits {
            0 => Err(MLError::InvalidConfiguration(
                "QuantumKernel requires at least one qubit".to_string(),
            )),
            1..=2 => self.compute_sized::<2>(x1, x2),
            3..=4 => self.compute_sized::<4>(x1, x2),
            5..=8 => self.compute_sized::<8>(x1, x2),
            9..=16 => self.compute_sized::<16>(x1, x2),
            n => Err(MLError::NotSupported(format!(
                "Quantum kernel estimation supports at most 16 qubits on the \
                 state-vector backend, got {n}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_quantum_kernel_exploration() {
        let kernel = QuantumKernel::new(2, 2);
        let x = array![0.3, 0.5];
        let y = array![0.3, 0.5];
        let z = array![0.9, 0.1];

        let k_xx = kernel.compute(&x, &x).expect("compute self");
        let k_xy = kernel.compute(&x, &y).expect("compute identical");
        let k_xz = kernel.compute(&x, &z).expect("compute distinct");
        let k_yx = kernel.compute(&y, &x).expect("compute swapped");
        eprintln!("k_xx={k_xx} k_xy={k_xy} k_xz={k_xz} k_yx={k_yx}");
    }
}
