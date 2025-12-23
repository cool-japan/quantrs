//! Quantum Variational Autoencoders (QVAE)
//!
//! This module implements quantum variational autoencoders for
//! quantum data compression and feature extraction.

use crate::error::MLError;
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64 as Complex;
use std::f64::consts::PI;

/// Quantum Variational Autoencoder
pub struct QVAE {
    /// Number of data qubits
    pub num_data_qubits: usize,
    /// Number of latent qubits (compressed representation)
    pub num_latent_qubits: usize,
    /// Number of ancilla qubits for encoding
    pub num_ancilla_qubits: usize,
    /// Encoder parameters
    pub encoder_params: Vec<f64>,
    /// Decoder parameters
    pub decoder_params: Vec<f64>,
}

impl QVAE {
    /// Create a new quantum variational autoencoder
    pub fn new(
        num_data_qubits: usize,
        num_latent_qubits: usize,
        num_ancilla_qubits: usize,
    ) -> Result<Self, MLError> {
        if num_latent_qubits >= num_data_qubits {
            return Err(MLError::InvalidParameter(
                "Latent space must be smaller than data space".to_string(),
            ));
        }

        // Initialize parameters for encoder and decoder
        let encoder_depth = 3;
        let decoder_depth = 3;

        let encoder_params = vec![0.1; num_data_qubits * encoder_depth * 3];
        let decoder_params = vec![0.1; num_data_qubits * decoder_depth * 3];

        Ok(Self {
            num_data_qubits,
            num_latent_qubits,
            num_ancilla_qubits,
            encoder_params,
            decoder_params,
        })
    }

    /// Get total number of qubits required
    pub fn total_qubits(&self) -> usize {
        self.num_data_qubits + self.num_latent_qubits + self.num_ancilla_qubits
    }

    /// Apply encoding circuit
    pub fn encode<const N: usize>(
        &self,
        circuit: &mut Circuit<N>,
        data_start: usize,
        latent_start: usize,
    ) -> Result<(), MLError> {
        // Check bounds
        if data_start + self.num_data_qubits > N {
            return Err(MLError::InvalidParameter(
                "Data qubits exceed circuit size".to_string(),
            ));
        }
        if latent_start + self.num_latent_qubits > N {
            return Err(MLError::InvalidParameter(
                "Latent qubits exceed circuit size".to_string(),
            ));
        }

        // Apply parameterized encoding layers
        let mut param_idx = 0;
        let depth = self.encoder_params.len() / (self.num_data_qubits * 3);

        for layer in 0..depth {
            // Single-qubit rotations
            for i in 0..self.num_data_qubits {
                let q = data_start + i;
                if param_idx < self.encoder_params.len() {
                    circuit.rx(q, self.encoder_params[param_idx])?;
                    param_idx += 1;
                }
                if param_idx < self.encoder_params.len() {
                    circuit.ry(q, self.encoder_params[param_idx])?;
                    param_idx += 1;
                }
                if param_idx < self.encoder_params.len() {
                    circuit.rz(q, self.encoder_params[param_idx])?;
                    param_idx += 1;
                }
            }

            // Entangling layer
            for i in 0..self.num_data_qubits - 1 {
                circuit.cnot(data_start + i, data_start + i + 1)?;
            }

            // Compression: entangle with latent qubits
            if layer == depth - 1 {
                for i in 0..self.num_latent_qubits {
                    let data_q = data_start + (i % self.num_data_qubits);
                    let latent_q = latent_start + i;
                    circuit.cnot(data_q, latent_q)?;
                }
            }
        }

        Ok(())
    }

    /// Apply decoding circuit
    pub fn decode<const N: usize>(
        &self,
        circuit: &mut Circuit<N>,
        latent_start: usize,
        output_start: usize,
    ) -> Result<(), MLError> {
        // Check bounds
        if latent_start + self.num_latent_qubits > N {
            return Err(MLError::InvalidParameter(
                "Latent qubits exceed circuit size".to_string(),
            ));
        }
        if output_start + self.num_data_qubits > N {
            return Err(MLError::InvalidParameter(
                "Output qubits exceed circuit size".to_string(),
            ));
        }

        // Apply parameterized decoding layers
        let mut param_idx = 0;
        let depth = self.decoder_params.len() / (self.num_data_qubits * 3);

        for layer in 0..depth {
            // Decompression: entangle latent with output qubits
            if layer == 0 {
                for i in 0..self.num_latent_qubits {
                    let latent_q = latent_start + i;
                    let output_q = output_start + (i % self.num_data_qubits);
                    circuit.cnot(latent_q, output_q)?;
                }
            }

            // Single-qubit rotations on output qubits
            for i in 0..self.num_data_qubits {
                let q = output_start + i;
                if param_idx < self.decoder_params.len() {
                    circuit.rx(q, self.decoder_params[param_idx])?;
                    param_idx += 1;
                }
                if param_idx < self.decoder_params.len() {
                    circuit.ry(q, self.decoder_params[param_idx])?;
                    param_idx += 1;
                }
                if param_idx < self.decoder_params.len() {
                    circuit.rz(q, self.decoder_params[param_idx])?;
                    param_idx += 1;
                }
            }

            // Entangling layer
            for i in 0..self.num_data_qubits - 1 {
                circuit.cnot(output_start + i, output_start + i + 1)?;
            }
        }

        Ok(())
    }

    /// Build full autoencoder circuit
    pub fn build_circuit<const N: usize>(&self) -> Result<Circuit<N>, MLError> {
        if N < self.total_qubits() {
            return Err(MLError::InvalidParameter(format!(
                "Circuit needs at least {} qubits",
                self.total_qubits()
            )));
        }

        let mut circuit = Circuit::<N>::new();

        // Qubit allocation
        let data_start = 0;
        let latent_start = self.num_data_qubits;
        let output_start = self.num_data_qubits + self.num_latent_qubits;

        // Encode data into latent space
        self.encode(&mut circuit, data_start, latent_start)?;

        // Decode from latent space to output
        self.decode(&mut circuit, latent_start, output_start)?;

        Ok(circuit)
    }

    /// Compute reconstruction fidelity
    pub fn reconstruction_fidelity(
        &self,
        input_state: &[Complex],
        output_state: &[Complex],
    ) -> Result<f64, MLError> {
        if input_state.len() != output_state.len() {
            return Err(MLError::InvalidParameter(
                "State dimensions mismatch".to_string(),
            ));
        }

        // Compute inner product
        let inner_product: Complex = input_state
            .iter()
            .zip(output_state.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();

        // Fidelity is |<ψ|φ>|²
        Ok(inner_product.norm_sqr())
    }

    /// Get all trainable parameters
    pub fn get_parameters(&self) -> Vec<f64> {
        let mut params = self.encoder_params.clone();
        params.extend(&self.decoder_params);
        params
    }

    /// Set parameters from a flat vector
    pub fn set_parameters(&mut self, params: &[f64]) -> Result<(), MLError> {
        let encoder_size = self.encoder_params.len();
        let decoder_size = self.decoder_params.len();

        if params.len() != encoder_size + decoder_size {
            return Err(MLError::InvalidParameter(format!(
                "Expected {} parameters, got {}",
                encoder_size + decoder_size,
                params.len()
            )));
        }

        self.encoder_params.copy_from_slice(&params[..encoder_size]);
        self.decoder_params.copy_from_slice(&params[encoder_size..]);

        Ok(())
    }

    /// Compute loss function (negative fidelity + regularization)
    pub fn compute_loss(&self, input_states: &[Vec<Complex>], lambda: f64) -> Result<f64, MLError> {
        // For simplicity, compute average negative fidelity
        // In practice, would simulate the circuit for each input
        let mut total_loss = 0.0;

        for _input in input_states {
            // Simplified: assume perfect reconstruction for demo
            // In real implementation, would run circuit simulation
            total_loss += 1.0; // Placeholder
        }

        // Add L2 regularization
        let reg_term: f64 = self.get_parameters().iter().map(|p| p * p).sum::<f64>() * lambda;

        Ok(total_loss / input_states.len() as f64 + reg_term)
    }
}

/// Classical Autoencoder for comparison
pub struct ClassicalAutoencoder {
    /// Input dimension
    pub input_dim: usize,
    /// Latent dimension
    pub latent_dim: usize,
    /// Encoder weights
    pub encoder_weights: Vec<Vec<f64>>,
    /// Decoder weights
    pub decoder_weights: Vec<Vec<f64>>,
}

impl ClassicalAutoencoder {
    /// Create a new classical autoencoder
    pub fn new(input_dim: usize, latent_dim: usize) -> Self {
        let mut rng = scirs2_core::random::ChaCha8Rng::seed_from_u64(42);

        // Initialize weights with small random values
        let encoder_weights = (0..latent_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| rng.gen::<f64>() * 0.1 - 0.05)
                    .collect()
            })
            .collect();

        let decoder_weights = (0..input_dim)
            .map(|_| {
                (0..latent_dim)
                    .map(|_| rng.gen::<f64>() * 0.1 - 0.05)
                    .collect()
            })
            .collect();

        Self {
            input_dim,
            latent_dim,
            encoder_weights,
            decoder_weights,
        }
    }

    /// Encode data to latent space
    pub fn encode(&self, input: &[f64]) -> Vec<f64> {
        let mut latent = vec![0.0; self.latent_dim];

        for i in 0..self.latent_dim {
            for j in 0..self.input_dim {
                latent[i] += self.encoder_weights[i][j] * input[j];
            }
            // Apply activation (tanh)
            latent[i] = latent[i].tanh();
        }

        latent
    }

    /// Decode from latent space
    pub fn decode(&self, latent: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.input_dim];

        for i in 0..self.input_dim {
            for j in 0..self.latent_dim {
                output[i] += self.decoder_weights[i][j] * latent[j];
            }
            // Apply activation (sigmoid for normalized output)
            output[i] = 1.0 / (1.0 + (-output[i]).exp());
        }

        output
    }

    /// Full forward pass
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let latent = self.encode(input);
        self.decode(&latent)
    }
}

/// Quantum-Classical Hybrid Autoencoder
pub struct HybridAutoencoder {
    /// Quantum encoder
    pub quantum_encoder: QVAE,
    /// Classical decoder
    pub classical_decoder: ClassicalAutoencoder,
}

impl HybridAutoencoder {
    /// Create a new hybrid autoencoder
    pub fn new(
        num_data_qubits: usize,
        num_latent_qubits: usize,
        classical_latent_dim: usize,
    ) -> Result<Self, MLError> {
        let quantum_encoder = QVAE::new(num_data_qubits, num_latent_qubits, 0)?;

        // Classical decoder takes quantum latent space measurements
        let quantum_latent_dim = 1 << num_latent_qubits;
        let classical_decoder = ClassicalAutoencoder::new(quantum_latent_dim, classical_latent_dim);

        Ok(Self {
            quantum_encoder,
            classical_decoder,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qvae_creation() {
        let qvae = QVAE::new(4, 2, 0).unwrap();
        assert_eq!(qvae.num_data_qubits, 4);
        assert_eq!(qvae.num_latent_qubits, 2);
        assert_eq!(qvae.total_qubits(), 6);
    }

    #[test]
    fn test_qvae_invalid_params() {
        // Latent space must be smaller than data space
        let result = QVAE::new(4, 5, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_classical_autoencoder() {
        let ae = ClassicalAutoencoder::new(10, 3);
        let input = vec![0.5; 10];
        let output = ae.forward(&input);

        assert_eq!(output.len(), 10);
        // Check output is normalized (between 0 and 1)
        for &val in &output {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_parameter_management() {
        let mut qvae = QVAE::new(4, 2, 0).unwrap();
        let params = qvae.get_parameters();
        let new_params = vec![0.2; params.len()];

        qvae.set_parameters(&new_params).unwrap();
        let retrieved = qvae.get_parameters();

        assert_eq!(retrieved, new_params);
    }

    #[test]
    fn test_reconstruction_fidelity() {
        let qvae = QVAE::new(2, 1, 0).unwrap();
        let state = vec![
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
        ];

        let fidelity = qvae.reconstruction_fidelity(&state, &state).unwrap();
        assert!((fidelity - 1.0).abs() < 1e-10);
    }
}
