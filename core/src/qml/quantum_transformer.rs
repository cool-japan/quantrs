//! Quantum Transformer with Attention Mechanisms
//!
//! This module implements quantum transformers with attention mechanisms for
//! processing sequential quantum data. It includes:
//! - Multi-head quantum attention
//! - Quantum positional encoding
//! - Quantum feed-forward networks
//! - Layer normalization for quantum states
//!
//! # Theoretical Background
//!
//! Quantum transformers extend classical transformer architectures to the quantum domain,
//! leveraging quantum superposition and entanglement for enhanced representation learning.
//! The attention mechanism is implemented using quantum circuits that compute attention
//! scores via quantum interference patterns.
//!
//! # References
//!
//! - "Quantum Attention Networks"
//! - "Self-Attention in Quantum Computing"
//! - "Quantum Transformers for Natural Language Processing"

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::f64::consts::PI;

/// Configuration for quantum transformer
#[derive(Debug, Clone)]
pub struct QuantumTransformerConfig {
    /// Number of qubits for data representation
    pub num_qubits: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Dimension of feed-forward network
    pub ffn_dim: usize,
    /// Dropout rate for regularization
    pub dropout_rate: f64,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Whether to use layer normalization
    pub use_layer_norm: bool,
}

impl Default for QuantumTransformerConfig {
    fn default() -> Self {
        Self {
            num_qubits: 4,
            num_heads: 2,
            head_dim: 2,
            num_layers: 2,
            ffn_dim: 8,
            dropout_rate: 0.1,
            max_seq_length: 64,
            use_layer_norm: true,
        }
    }
}

/// Quantum attention mechanism using quantum circuits
#[derive(Debug, Clone)]
pub struct QuantumAttention {
    /// Number of qubits
    num_qubits: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Query parameters
    query_params: Array2<f64>,
    /// Key parameters
    key_params: Array2<f64>,
    /// Value parameters
    value_params: Array2<f64>,
    /// Output projection parameters
    output_params: Array2<f64>,
}

impl QuantumAttention {
    /// Create a new quantum attention mechanism
    pub fn new(num_qubits: usize, num_heads: usize, head_dim: usize) -> QuantRS2Result<Self> {
        if num_qubits < 2 {
            return Err(QuantRS2Error::InvalidInput(
                "Quantum attention requires at least 2 qubits".to_string(),
            ));
        }

        if num_heads == 0 || head_dim == 0 {
            return Err(QuantRS2Error::InvalidInput(
                "Number of heads and head dimension must be positive".to_string(),
            ));
        }

        let total_dim = num_heads * head_dim;
        let mut rng = thread_rng();

        // Xavier initialization for quantum parameters
        let scale = (2.0 / (num_qubits as f64)).sqrt();

        let query_params =
            Array2::from_shape_fn((total_dim, num_qubits), |_| rng.gen_range(-scale..scale));

        let key_params =
            Array2::from_shape_fn((total_dim, num_qubits), |_| rng.gen_range(-scale..scale));

        let value_params =
            Array2::from_shape_fn((total_dim, num_qubits), |_| rng.gen_range(-scale..scale));

        let output_params =
            Array2::from_shape_fn((num_qubits, total_dim), |_| rng.gen_range(-scale..scale));

        Ok(Self {
            num_qubits,
            num_heads,
            head_dim,
            query_params,
            key_params,
            value_params,
            output_params,
        })
    }

    /// Compute quantum attention scores using quantum interference
    pub fn attention_scores(
        &self,
        query: &Array2<Complex64>,
        key: &Array2<Complex64>,
    ) -> QuantRS2Result<Array2<f64>> {
        let seq_len = query.shape()[0];
        let mut scores = Array2::zeros((seq_len, seq_len));

        // Compute attention scores via quantum state overlap
        for i in 0..seq_len {
            for j in 0..seq_len {
                let q = query.row(i);
                let k = key.row(j);

                // Quantum inner product (fidelity)
                let mut score = Complex64::new(0.0, 0.0);
                for (qi, ki) in q.iter().zip(k.iter()) {
                    score += qi.conj() * ki;
                }

                // Scale by sqrt(head_dim) as in classical transformers
                let scaled_score = score.norm() / (self.head_dim as f64).sqrt();
                scores[[i, j]] = scaled_score;
            }
        }

        Ok(scores)
    }

    /// Apply softmax to attention scores
    pub fn softmax(&self, scores: &Array2<f64>) -> Array2<f64> {
        let seq_len = scores.shape()[0];
        let mut softmax_scores = Array2::zeros((seq_len, seq_len));

        for i in 0..seq_len {
            let row = scores.row(i);
            let max_score = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            // Compute exp(score - max) for numerical stability
            let mut exp_scores = Array1::zeros(seq_len);
            let mut sum_exp = 0.0;

            for (j, &score) in row.iter().enumerate() {
                let exp_val = (score - max_score).exp();
                exp_scores[j] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize
            for j in 0..seq_len {
                softmax_scores[[i, j]] = exp_scores[j] / sum_exp;
            }
        }

        softmax_scores
    }

    /// Apply quantum attention to input
    pub fn forward(&self, input: &Array2<Complex64>) -> QuantRS2Result<Array2<Complex64>> {
        let seq_len = input.shape()[0];

        // Project to query, key, value using quantum rotations
        let query = self.project_qkv(input, &self.query_params)?;
        let key = self.project_qkv(input, &self.key_params)?;
        let value = self.project_qkv(input, &self.value_params)?;

        // Compute attention scores
        let scores = self.attention_scores(&query, &key)?;
        let attention_weights = self.softmax(&scores);

        // Apply attention to values
        let total_dim = self.num_heads * self.head_dim;
        let mut output = Array2::zeros((seq_len, total_dim));

        for i in 0..seq_len {
            for j in 0..seq_len {
                let weight = attention_weights[[i, j]];
                for k in 0..total_dim {
                    output[[i, k]] = output[[i, k]] + value[[j, k]] * weight;
                }
            }
        }

        // Project back to original dimension
        self.project_output(&output)
    }

    /// Project input to query/key/value space
    fn project_qkv(
        &self,
        input: &Array2<Complex64>,
        params: &Array2<f64>,
    ) -> QuantRS2Result<Array2<Complex64>> {
        let seq_len = input.shape()[0];
        let out_dim = params.shape()[0];
        let mut output = Array2::zeros((seq_len, out_dim));

        for i in 0..seq_len {
            for j in 0..out_dim {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..self.num_qubits {
                    // Quantum rotation based projection
                    let angle = params[[j, k]];
                    let rotation = Complex64::new(angle.cos(), angle.sin());
                    sum += input[[i, k]] * rotation;
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }

    /// Project output back to original dimension
    fn project_output(
        &self,
        attention_out: &Array2<Complex64>,
    ) -> QuantRS2Result<Array2<Complex64>> {
        let seq_len = attention_out.shape()[0];
        let mut output = Array2::zeros((seq_len, self.num_qubits));

        for i in 0..seq_len {
            for j in 0..self.num_qubits {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..(self.num_heads * self.head_dim) {
                    let angle = self.output_params[[j, k]];
                    let rotation = Complex64::new(angle.cos(), angle.sin());
                    sum += attention_out[[i, k]] * rotation;
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }
}

/// Quantum positional encoding for sequence information
#[derive(Debug, Clone)]
pub struct QuantumPositionalEncoding {
    /// Maximum sequence length
    max_seq_length: usize,
    /// Number of qubits
    num_qubits: usize,
    /// Encoding parameters
    encoding: Array2<f64>,
}

impl QuantumPositionalEncoding {
    /// Create new quantum positional encoding
    pub fn new(max_seq_length: usize, num_qubits: usize) -> Self {
        let mut encoding = Array2::zeros((max_seq_length, num_qubits));

        // Quantum sinusoidal positional encoding
        for pos in 0..max_seq_length {
            for i in 0..num_qubits {
                if i % 2 == 0 {
                    let freq = 1.0 / 10000_f64.powf(i as f64 / num_qubits as f64);
                    encoding[[pos, i]] = (pos as f64 * freq).sin();
                } else {
                    let freq = 1.0 / 10000_f64.powf((i - 1) as f64 / num_qubits as f64);
                    encoding[[pos, i]] = (pos as f64 * freq).cos();
                }
            }
        }

        Self {
            max_seq_length,
            num_qubits,
            encoding,
        }
    }

    /// Add positional encoding to input quantum states
    pub fn encode(&self, input: &Array2<Complex64>) -> QuantRS2Result<Array2<Complex64>> {
        let seq_len = input.shape()[0];

        if seq_len > self.max_seq_length {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.max_seq_length
            )));
        }

        let mut output = input.clone();

        // Add positional encoding using quantum phase shifts
        for i in 0..seq_len {
            for j in 0..self.num_qubits {
                let phase = self.encoding[[i, j]];
                let phase_shift = Complex64::new(phase.cos(), phase.sin());
                output[[i, j]] = output[[i, j]] * phase_shift;
            }
        }

        Ok(output)
    }
}

/// Quantum feed-forward network
#[derive(Debug, Clone)]
pub struct QuantumFeedForward {
    /// Input dimension
    input_dim: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// First layer parameters
    w1: Array2<f64>,
    /// Second layer parameters
    w2: Array2<f64>,
}

impl QuantumFeedForward {
    /// Create new quantum feed-forward network
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = thread_rng();
        let scale1 = (2.0 / input_dim as f64).sqrt();
        let scale2 = (2.0 / hidden_dim as f64).sqrt();

        let w1 = Array2::from_shape_fn((hidden_dim, input_dim), |_| rng.gen_range(-scale1..scale1));

        let w2 = Array2::from_shape_fn((input_dim, hidden_dim), |_| rng.gen_range(-scale2..scale2));

        Self {
            input_dim,
            hidden_dim,
            w1,
            w2,
        }
    }

    /// Forward pass through quantum FFN
    pub fn forward(&self, input: &Array2<Complex64>) -> QuantRS2Result<Array2<Complex64>> {
        let seq_len = input.shape()[0];

        // First layer with quantum activation
        let mut hidden = Array2::zeros((seq_len, self.hidden_dim));
        for i in 0..seq_len {
            for j in 0..self.hidden_dim {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..self.input_dim {
                    let angle = self.w1[[j, k]];
                    let rotation = Complex64::new(angle.cos(), angle.sin());
                    sum += input[[i, k]] * rotation;
                }
                // Quantum ReLU-like activation
                hidden[[i, j]] = self.quantum_activation(sum);
            }
        }

        // Second layer
        let mut output = Array2::zeros((seq_len, self.input_dim));
        for i in 0..seq_len {
            for j in 0..self.input_dim {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..self.hidden_dim {
                    let angle = self.w2[[j, k]];
                    let rotation = Complex64::new(angle.cos(), angle.sin());
                    sum += hidden[[i, k]] * rotation;
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }

    /// Quantum activation function
    fn quantum_activation(&self, z: Complex64) -> Complex64 {
        // Quantum version of ReLU using amplitude amplification
        let amplitude = z.norm();
        let phase = z.arg();

        if amplitude > 0.0 {
            // Amplify based on magnitude
            let amplified = amplitude.tanh();
            Complex64::new(amplified * phase.cos(), amplified * phase.sin())
        } else {
            Complex64::new(0.0, 0.0)
        }
    }
}

/// Complete quantum transformer layer
#[derive(Debug, Clone)]
pub struct QuantumTransformerLayer {
    /// Multi-head attention
    attention: QuantumAttention,
    /// Feed-forward network
    ffn: QuantumFeedForward,
    /// Configuration
    config: QuantumTransformerConfig,
}

impl QuantumTransformerLayer {
    /// Create new quantum transformer layer
    pub fn new(config: QuantumTransformerConfig) -> QuantRS2Result<Self> {
        let attention =
            QuantumAttention::new(config.num_qubits, config.num_heads, config.head_dim)?;

        let ffn = QuantumFeedForward::new(config.num_qubits, config.ffn_dim);

        Ok(Self {
            attention,
            ffn,
            config,
        })
    }

    /// Forward pass through transformer layer
    pub fn forward(&self, input: &Array2<Complex64>) -> QuantRS2Result<Array2<Complex64>> {
        // Multi-head attention with residual connection
        let attention_out = self.attention.forward(input)?;
        let after_attention = self.add_residual(input, &attention_out);

        // Layer normalization (if enabled)
        let normalized = if self.config.use_layer_norm {
            self.layer_norm(&after_attention)?
        } else {
            after_attention
        };

        // Feed-forward with residual connection
        let ffn_out = self.ffn.forward(&normalized)?;
        let output = self.add_residual(&normalized, &ffn_out);

        // Final layer normalization
        if self.config.use_layer_norm {
            self.layer_norm(&output)
        } else {
            Ok(output)
        }
    }

    /// Add residual connection
    fn add_residual(
        &self,
        input: &Array2<Complex64>,
        residual: &Array2<Complex64>,
    ) -> Array2<Complex64> {
        input + residual
    }

    /// Quantum layer normalization
    fn layer_norm(&self, input: &Array2<Complex64>) -> QuantRS2Result<Array2<Complex64>> {
        let seq_len = input.shape()[0];
        let num_features = input.shape()[1];
        let mut output = Array2::zeros((seq_len, num_features));

        for i in 0..seq_len {
            let row = input.row(i);

            // Compute mean and variance of quantum state amplitudes
            let mut mean_real = 0.0;
            let mut mean_imag = 0.0;
            for val in row {
                mean_real += val.re;
                mean_imag += val.im;
            }
            mean_real /= num_features as f64;
            mean_imag /= num_features as f64;
            let mean = Complex64::new(mean_real, mean_imag);

            let mut variance = 0.0;
            for val in row {
                let diff = val - mean;
                variance += diff.norm_sqr();
            }
            variance /= num_features as f64;

            let std = (variance + 1e-5).sqrt();

            // Normalize
            for j in 0..num_features {
                output[[i, j]] = (input[[i, j]] - mean) / std;
            }
        }

        Ok(output)
    }
}

/// Complete quantum transformer model
#[derive(Debug, Clone)]
pub struct QuantumTransformer {
    /// Configuration
    config: QuantumTransformerConfig,
    /// Positional encoding
    pos_encoding: QuantumPositionalEncoding,
    /// Transformer layers
    layers: Vec<QuantumTransformerLayer>,
}

impl QuantumTransformer {
    /// Create new quantum transformer
    pub fn new(config: QuantumTransformerConfig) -> QuantRS2Result<Self> {
        let pos_encoding = QuantumPositionalEncoding::new(config.max_seq_length, config.num_qubits);

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(QuantumTransformerLayer::new(config.clone())?);
        }

        Ok(Self {
            config,
            pos_encoding,
            layers,
        })
    }

    /// Forward pass through transformer
    pub fn forward(&self, input: &Array2<Complex64>) -> QuantRS2Result<Array2<Complex64>> {
        // Add positional encoding
        let mut x = self.pos_encoding.encode(input)?;

        // Pass through transformer layers
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        Ok(x)
    }

    /// Get configuration
    pub const fn config(&self) -> &QuantumTransformerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_attention() {
        let attention = QuantumAttention::new(4, 2, 2).expect("Failed to create QuantumAttention");

        // Create test input (sequence of 3 quantum states)
        let mut input = Array2::zeros((3, 4));
        for i in 0..3 {
            for j in 0..4 {
                input[[i, j]] = Complex64::new((i + j) as f64 * 0.1, 0.0);
            }
        }

        let output = attention
            .forward(&input)
            .expect("Attention forward pass should succeed");
        assert_eq!(output.shape(), &[3, 4]);
    }

    #[test]
    fn test_positional_encoding() {
        let pos_enc = QuantumPositionalEncoding::new(64, 4);

        let mut input = Array2::zeros((3, 4));
        for i in 0..3 {
            for j in 0..4 {
                input[[i, j]] = Complex64::new(1.0, 0.0);
            }
        }

        let encoded = pos_enc
            .encode(&input)
            .expect("Positional encoding should succeed");
        assert_eq!(encoded.shape(), &[3, 4]);
    }

    #[test]
    fn test_quantum_transformer() {
        let config = QuantumTransformerConfig {
            num_qubits: 4,
            num_heads: 2,
            head_dim: 2,
            num_layers: 2,
            ffn_dim: 8,
            dropout_rate: 0.1,
            max_seq_length: 64,
            use_layer_norm: true,
        };

        let transformer =
            QuantumTransformer::new(config).expect("Failed to create QuantumTransformer");

        // Create test input
        let mut input = Array2::zeros((3, 4));
        for i in 0..3 {
            for j in 0..4 {
                input[[i, j]] = Complex64::new((i + j) as f64 * 0.1, 0.0);
            }
        }

        let output = transformer
            .forward(&input)
            .expect("Transformer forward pass should succeed");
        assert_eq!(output.shape(), &[3, 4]);
    }
}
