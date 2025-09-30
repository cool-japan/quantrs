//! Quantum attention mechanisms for transformer architectures.
//!
//! This module implements quantum versions of attention mechanisms including
//! multi-head attention, cross-attention, and quantum transformer blocks.

use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::autodiff::DifferentiableParam;
use crate::error::{MLError, Result};
use crate::utils::VariationalCircuit;
use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::{multi::*, single::*, GateOp};

/// Quantum self-attention mechanism
#[derive(Debug, Clone)]
pub struct QuantumSelfAttention {
    /// Embedding dimension
    embed_dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Number of qubits per head
    qubits_per_head: usize,
    /// Query projection circuit
    query_circuit: QuantumProjection,
    /// Key projection circuit
    key_circuit: QuantumProjection,
    /// Value projection circuit
    value_circuit: QuantumProjection,
    /// Output projection circuit
    output_circuit: QuantumProjection,
    /// Dropout rate
    dropout_rate: f64,
    /// Temperature for attention scaling
    temperature: f64,
}

/// Quantum projection layer
#[derive(Debug, Clone)]
struct QuantumProjection {
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Number of qubits
    num_qubits: usize,
    /// Variational circuit
    circuit: VariationalCircuit,
    /// Parameters
    parameters: HashMap<String, f64>,
}

impl QuantumProjection {
    /// Create a new projection layer
    fn new(input_dim: usize, output_dim: usize) -> Self {
        let num_qubits = ((input_dim.max(output_dim)) as f64).log2().ceil() as usize;
        let circuit = Self::build_projection_circuit(num_qubits);

        Self {
            input_dim,
            output_dim,
            num_qubits,
            circuit,
            parameters: HashMap::new(),
        }
    }

    /// Build the projection circuit
    fn build_projection_circuit(num_qubits: usize) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(num_qubits);

        // Layer 1: Feature encoding
        for q in 0..num_qubits {
            circuit.add_gate("RY", vec![q], vec![format!("encode_{}", q)]);
        }

        // Layer 2: Entangling layer
        for q in 0..num_qubits - 1 {
            circuit.add_gate("CNOT", vec![q, q + 1], vec![]);
        }
        if num_qubits > 2 {
            circuit.add_gate("CNOT", vec![num_qubits - 1, 0], vec![]);
        }

        // Layer 3: Parameterized rotations
        for q in 0..num_qubits {
            circuit.add_gate("RX", vec![q], vec![format!("rx_{}", q)]);
            circuit.add_gate("RZ", vec![q], vec![format!("rz_{}", q)]);
        }

        // Layer 4: Second entangling layer
        for q in (0..num_qubits - 1).step_by(2) {
            circuit.add_gate("CZ", vec![q, q + 1], vec![]);
        }
        for q in (1..num_qubits - 1).step_by(2) {
            circuit.add_gate("CZ", vec![q, q + 1], vec![]);
        }

        // Layer 5: Final rotations
        for q in 0..num_qubits {
            circuit.add_gate("RY", vec![q], vec![format!("final_{}", q)]);
        }

        circuit
    }

    /// Project input through the quantum circuit
    fn forward(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        // Encode input
        let encoded = self.encode_input(input)?;

        // Execute circuit (simplified)
        let output_state = self.execute_circuit(&encoded)?;

        // Decode output
        self.decode_output(&output_state)
    }

    /// Encode classical input to quantum state
    fn encode_input(&self, input: &Array1<f64>) -> Result<Vec<Complex64>> {
        let state_dim = 2_usize.pow(self.num_qubits as u32);
        let mut quantum_state = vec![Complex64::new(0.0, 0.0); state_dim];

        // Amplitude encoding
        let norm: f64 = input.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Err(MLError::InvalidInput("Zero norm input".to_string()));
        }

        for (i, &val) in input.iter().enumerate() {
            if i < state_dim {
                quantum_state[i] = Complex64::new(val / norm, 0.0);
            }
        }

        Ok(quantum_state)
    }

    /// Execute the quantum circuit
    fn execute_circuit(&self, input_state: &[Complex64]) -> Result<Vec<Complex64>> {
        // Simplified circuit execution
        // In practice, would use actual quantum simulation
        let state_dim = input_state.len();
        let mut output_state = input_state.to_vec();

        // Apply some transformation
        for i in 0..state_dim {
            let phase = (i as f64) * 0.1;
            output_state[i] *= Complex64::new(phase.cos(), phase.sin());
        }

        Ok(output_state)
    }

    /// Decode quantum state to classical output
    fn decode_output(&self, quantum_state: &[Complex64]) -> Result<Array1<f64>> {
        let mut output = Array1::zeros(self.output_dim);

        // Extract amplitudes
        for i in 0..self.output_dim.min(quantum_state.len()) {
            output[i] = quantum_state[i].norm();
        }

        Ok(output)
    }
}

impl QuantumSelfAttention {
    /// Create a new quantum self-attention layer
    pub fn new(embed_dim: usize, num_heads: usize, dropout_rate: f64) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim must be divisible by num_heads"
        );

        let head_dim = embed_dim / num_heads;
        let qubits_per_head = (head_dim as f64).log2().ceil() as usize;

        Self {
            embed_dim,
            num_heads,
            head_dim,
            qubits_per_head,
            query_circuit: QuantumProjection::new(embed_dim, embed_dim),
            key_circuit: QuantumProjection::new(embed_dim, embed_dim),
            value_circuit: QuantumProjection::new(embed_dim, embed_dim),
            output_circuit: QuantumProjection::new(embed_dim, embed_dim),
            dropout_rate,
            temperature: (head_dim as f64).sqrt(),
        }
    }

    /// Forward pass through attention layer
    pub fn forward(
        &self,
        query: &Array2<f64>,
        key: &Array2<f64>,
        value: &Array2<f64>,
        mask: Option<&Array2<bool>>,
    ) -> Result<Array2<f64>> {
        let batch_size = query.nrows();
        let seq_len = query.ncols() / self.embed_dim;

        // Project Q, K, V
        let q = self.project_to_heads(query, &self.query_circuit)?;
        let k = self.project_to_heads(key, &self.key_circuit)?;
        let v = self.project_to_heads(value, &self.value_circuit)?;

        // Compute attention scores
        let attention_scores = self.compute_attention_scores(&q, &k)?;

        // Apply mask if provided
        let masked_scores = if let Some(mask) = mask {
            self.apply_mask(&attention_scores, mask)?
        } else {
            attention_scores
        };

        // Apply softmax
        let attention_weights = self.quantum_softmax(&masked_scores)?;

        // Apply attention to values
        let attended_values = self.apply_attention(&attention_weights, &v)?;

        // Concatenate heads and project output
        self.project_output(&attended_values)
    }

    /// Project input to multi-head format
    fn project_to_heads(
        &self,
        input: &Array2<f64>,
        projection: &QuantumProjection,
    ) -> Result<Array3<f64>> {
        let batch_size = input.nrows();
        let seq_len = input.ncols() / self.embed_dim;

        let mut output = Array3::zeros((batch_size, self.num_heads, seq_len * self.head_dim));

        for b in 0..batch_size {
            for s in 0..seq_len {
                let start = s * self.embed_dim;
                let end = start + self.embed_dim;
                let input_vec = input.row(b).slice(s![start..end]).to_owned();

                let projected = projection.forward(&input_vec)?;

                // Split into heads
                for h in 0..self.num_heads {
                    let head_start = h * self.head_dim;
                    let head_end = head_start + self.head_dim;

                    for i in 0..self.head_dim {
                        if head_start + i < projected.len() {
                            output[[b, h, s * self.head_dim + i]] = projected[head_start + i];
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    /// Compute quantum attention scores
    fn compute_attention_scores(
        &self,
        query: &Array3<f64>,
        key: &Array3<f64>,
    ) -> Result<Array3<f64>> {
        let batch_size = query.shape()[0];
        let seq_len = query.shape()[2] / self.head_dim;

        let mut scores = Array3::zeros((batch_size, self.num_heads, seq_len * seq_len));

        // Quantum dot product attention
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let q_start = i * self.head_dim;
                        let q_end = q_start + self.head_dim;
                        let k_start = j * self.head_dim;
                        let k_end = k_start + self.head_dim;

                        let q_vec = query.slice(s![b, h, q_start..q_end]);
                        let k_vec = key.slice(s![b, h, k_start..k_end]);

                        // Quantum inner product
                        let score =
                            self.quantum_inner_product(&q_vec.to_owned(), &k_vec.to_owned())?;
                        scores[[b, h, i * seq_len + j]] = score / self.temperature;
                    }
                }
            }
        }

        Ok(scores)
    }

    /// Compute quantum inner product
    fn quantum_inner_product(&self, vec1: &Array1<f64>, vec2: &Array1<f64>) -> Result<f64> {
        // Build quantum circuit for inner product
        let num_qubits = self.qubits_per_head * 2 + 1; // Extra qubit for measurement
        let mut circuit = VariationalCircuit::new(num_qubits);

        // Encode vectors
        for i in 0..self.qubits_per_head {
            if i < vec1.len() {
                let angle1 = vec1[i] * PI;
                circuit.add_gate("RY", vec![i], vec![angle1.to_string()]);
            }
            if i < vec2.len() {
                let angle2 = vec2[i] * PI;
                circuit.add_gate(
                    "RY",
                    vec![i + self.qubits_per_head],
                    vec![angle2.to_string()],
                );
            }
        }

        // Hadamard on ancilla
        circuit.add_gate("H", vec![num_qubits - 1], vec![]);

        // Controlled swap test
        for i in 0..self.qubits_per_head {
            circuit.add_gate(
                "CSWAP",
                vec![num_qubits - 1, i, i + self.qubits_per_head],
                vec![],
            );
        }

        // Hadamard on ancilla
        circuit.add_gate("H", vec![num_qubits - 1], vec![]);

        // Measurement probability gives inner product
        // Simplified: return dot product
        Ok(vec1.dot(vec2))
    }

    /// Quantum softmax implementation
    fn quantum_softmax(&self, scores: &Array3<f64>) -> Result<Array3<f64>> {
        let mut output = scores.clone();

        // Apply quantum softmax per attention head
        for b in 0..scores.shape()[0] {
            for h in 0..scores.shape()[1] {
                let head_scores = scores.slice(s![b, h, ..]);
                let seq_len = (head_scores.len() as f64).sqrt() as usize;

                for i in 0..seq_len {
                    let start = i * seq_len;
                    let end = start + seq_len;
                    let row_scores = head_scores.slice(s![start..end]);

                    // Quantum softmax circuit
                    let softmax_vals = self.quantum_softmax_circuit(&row_scores.to_owned())?;

                    for j in 0..seq_len {
                        output[[b, h, start + j]] = softmax_vals[j];
                    }
                }
            }
        }

        Ok(output)
    }

    /// Quantum circuit for softmax
    fn quantum_softmax_circuit(&self, logits: &Array1<f64>) -> Result<Vec<f64>> {
        // Classical softmax for now
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_logits: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f64 = exp_logits.iter().sum();

        Ok(exp_logits.into_iter().map(|x| x / sum_exp).collect())
    }

    /// Apply attention weights to values
    fn apply_attention(&self, weights: &Array3<f64>, values: &Array3<f64>) -> Result<Array3<f64>> {
        let batch_size = weights.shape()[0];
        let num_heads = weights.shape()[1];
        let seq_len = (weights.shape()[2] as f64).sqrt() as usize;

        let mut output = Array3::zeros((batch_size, num_heads, seq_len * self.head_dim));

        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let weight = weights[[b, h, i * seq_len + j]];

                        for d in 0..self.head_dim {
                            output[[b, h, i * self.head_dim + d]] +=
                                weight * values[[b, h, j * self.head_dim + d]];
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    /// Apply attention mask
    fn apply_mask(&self, scores: &Array3<f64>, mask: &Array2<bool>) -> Result<Array3<f64>> {
        let mut masked_scores = scores.clone();

        for b in 0..scores.shape()[0] {
            for h in 0..scores.shape()[1] {
                for (idx, &is_masked) in mask.iter().enumerate() {
                    if is_masked && idx < scores.shape()[2] {
                        masked_scores[[b, h, idx]] = -1e9; // Large negative value
                    }
                }
            }
        }

        Ok(masked_scores)
    }

    /// Project concatenated heads to output
    fn project_output(&self, attended: &Array3<f64>) -> Result<Array2<f64>> {
        let batch_size = attended.shape()[0];
        let seq_len = attended.shape()[2] / self.head_dim;

        let mut output = Array2::zeros((batch_size, seq_len * self.embed_dim));

        for b in 0..batch_size {
            for s in 0..seq_len {
                // Concatenate heads
                let mut concat = Array1::zeros(self.embed_dim);
                for h in 0..self.num_heads {
                    for d in 0..self.head_dim {
                        concat[h * self.head_dim + d] = attended[[b, h, s * self.head_dim + d]];
                    }
                }

                // Project through output circuit
                let projected = self.output_circuit.forward(&concat)?;

                for d in 0..self.embed_dim {
                    output[[b, s * self.embed_dim + d]] = projected[d];
                }
            }
        }

        Ok(output)
    }
}

/// Quantum transformer block
#[derive(Debug)]
pub struct QuantumTransformerBlock {
    /// Self-attention layer
    self_attention: QuantumSelfAttention,
    /// Feed-forward dimension
    ff_dim: usize,
    /// First feed-forward layer
    ff1: QuantumFeedForward,
    /// Second feed-forward layer
    ff2: QuantumFeedForward,
    /// Layer normalization (classical)
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    /// Dropout rate
    dropout_rate: f64,
}

/// Quantum feed-forward layer
#[derive(Debug)]
struct QuantumFeedForward {
    input_dim: usize,
    output_dim: usize,
    circuit: VariationalCircuit,
}

impl QuantumFeedForward {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        let num_qubits = ((input_dim.max(output_dim)) as f64).log2().ceil() as usize;
        let circuit = Self::build_ff_circuit(num_qubits);

        Self {
            input_dim,
            output_dim,
            circuit,
        }
    }

    fn build_ff_circuit(num_qubits: usize) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(num_qubits);

        // Dense connectivity pattern
        for layer in 0..3 {
            // Rotation layer
            for q in 0..num_qubits {
                circuit.add_gate("RY", vec![q], vec![format!("ff_ry_{}_{}", layer, q)]);
                circuit.add_gate("RZ", vec![q], vec![format!("ff_rz_{}_{}", layer, q)]);
            }

            // All-to-all entangling
            for i in 0..num_qubits {
                for j in i + 1..num_qubits {
                    circuit.add_gate("CZ", vec![i, j], vec![]);
                }
            }
        }

        circuit
    }

    fn forward(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        // Simplified forward pass
        let mut output = Array1::zeros(self.output_dim);

        // Apply non-linear transformation
        for i in 0..self.output_dim {
            if i < input.len() {
                output[i] = (input[i] * 2.0 * PI).sin() * 0.5 + 0.5;
            }
        }

        Ok(output)
    }
}

/// Classical layer normalization
#[derive(Debug)]
struct LayerNorm {
    normalized_shape: usize,
    epsilon: f64,
}

impl LayerNorm {
    fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            epsilon: 1e-5,
        }
    }

    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let mean = input.mean_axis(Axis(1)).unwrap();
        let variance = input.var_axis(Axis(1), 0.0);

        let mut output = input.clone();
        for i in 0..input.nrows() {
            let std = (variance[i] + self.epsilon).sqrt();
            for j in 0..input.ncols() {
                output[[i, j]] = (input[[i, j]] - mean[i]) / std;
            }
        }

        output
    }
}

impl QuantumTransformerBlock {
    /// Create a new transformer block
    pub fn new(embed_dim: usize, num_heads: usize, ff_dim: usize, dropout_rate: f64) -> Self {
        Self {
            self_attention: QuantumSelfAttention::new(embed_dim, num_heads, dropout_rate),
            ff_dim,
            ff1: QuantumFeedForward::new(embed_dim, ff_dim),
            ff2: QuantumFeedForward::new(ff_dim, embed_dim),
            layer_norm1: LayerNorm::new(embed_dim),
            layer_norm2: LayerNorm::new(embed_dim),
            dropout_rate,
        }
    }

    /// Forward pass through transformer block
    pub fn forward(&self, input: &Array2<f64>, mask: Option<&Array2<bool>>) -> Result<Array2<f64>> {
        // Self-attention with residual connection
        let attended = self.self_attention.forward(input, input, input, mask)?;
        let residual1 = &attended + input;
        let norm1 = self.layer_norm1.forward(&residual1);

        // Feed-forward with residual connection
        let batch_size = norm1.nrows();
        let seq_dim = norm1.ncols();
        let seq_len = seq_dim / self.self_attention.embed_dim;

        let mut ff_output = Array2::zeros((batch_size, seq_dim));

        for b in 0..batch_size {
            for s in 0..seq_len {
                let start = s * self.self_attention.embed_dim;
                let end = start + self.self_attention.embed_dim;

                let input_slice = norm1.slice(s![b, start..end]).to_owned();
                let hidden = self.ff1.forward(&input_slice)?;
                let output = self.ff2.forward(&hidden)?;

                for i in 0..self.self_attention.embed_dim {
                    ff_output[[b, start + i]] = output[i];
                }
            }
        }

        let residual2 = &ff_output + &norm1;
        let output = self.layer_norm2.forward(&residual2);

        Ok(output)
    }
}

/// Quantum transformer model
#[derive(Debug)]
pub struct QuantumTransformer {
    /// Embedding dimension
    embed_dim: usize,
    /// Number of transformer blocks
    num_layers: usize,
    /// Transformer blocks
    blocks: Vec<QuantumTransformerBlock>,
    /// Positional encoding
    positional_encoding: PositionalEncoding,
}

/// Quantum positional encoding
#[derive(Debug)]
struct PositionalEncoding {
    max_length: usize,
    embed_dim: usize,
}

impl PositionalEncoding {
    fn new(max_length: usize, embed_dim: usize) -> Self {
        Self {
            max_length,
            embed_dim,
        }
    }

    fn encode(&self, seq_len: usize) -> Array2<f64> {
        let mut encoding = Array2::zeros((seq_len, self.embed_dim));

        for pos in 0..seq_len {
            for i in 0..self.embed_dim {
                let angle = if i % 2 == 0 {
                    (pos as f64) / 10000_f64.powf((i as f64) / (self.embed_dim as f64))
                } else {
                    (pos as f64) / 10000_f64.powf(((i - 1) as f64) / (self.embed_dim as f64))
                };

                encoding[[pos, i]] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        encoding
    }
}

impl QuantumTransformer {
    /// Create a new quantum transformer
    pub fn new(
        embed_dim: usize,
        num_layers: usize,
        num_heads: usize,
        ff_dim: usize,
        max_length: usize,
        dropout_rate: f64,
    ) -> Self {
        let blocks = (0..num_layers)
            .map(|_| QuantumTransformerBlock::new(embed_dim, num_heads, ff_dim, dropout_rate))
            .collect();

        Self {
            embed_dim,
            num_layers,
            blocks,
            positional_encoding: PositionalEncoding::new(max_length, embed_dim),
        }
    }

    /// Forward pass through transformer
    pub fn forward(&self, input: &Array2<f64>, mask: Option<&Array2<bool>>) -> Result<Array2<f64>> {
        let seq_len = input.ncols() / self.embed_dim;

        // Add positional encoding
        let pos_encoding = self.positional_encoding.encode(seq_len);
        let mut encoded = input.clone();

        for i in 0..input.nrows() {
            for s in 0..seq_len {
                for d in 0..self.embed_dim {
                    encoded[[i, s * self.embed_dim + d]] += pos_encoding[[s, d]];
                }
            }
        }

        // Pass through transformer blocks
        let mut output = encoded;
        for block in &self.blocks {
            output = block.forward(&output, mask)?;
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_quantum_projection() {
        let proj = QuantumProjection::new(8, 8);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);

        let output = proj.forward(&input).unwrap();
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_quantum_self_attention() {
        let attention = QuantumSelfAttention::new(16, 4, 0.1);

        let batch_size = 2;
        let seq_len = 3;
        let embed_dim = 16;

        // Initialize input with non-zero values to avoid "Zero norm input" error
        let mut input = Array2::zeros((batch_size, seq_len * embed_dim));
        for i in 0..batch_size {
            for j in 0..seq_len * embed_dim {
                input[[i, j]] = 0.1 + (i * seq_len * embed_dim + j) as f64 * 0.01;
            }
        }

        let output = attention.forward(&input, &input, &input, None).unwrap();

        assert_eq!(output.shape(), &[batch_size, seq_len * embed_dim]);
    }

    #[test]
    fn test_quantum_transformer_block() {
        let block = QuantumTransformerBlock::new(8, 2, 16, 0.1);

        let batch_size = 1;
        let seq_len = 2;
        let embed_dim = 8;

        let input = Array2::ones((batch_size, seq_len * embed_dim));
        let output = block.forward(&input, None).unwrap();

        assert_eq!(output.shape(), &[batch_size, seq_len * embed_dim]);
    }

    #[test]
    fn test_positional_encoding() {
        let pos_enc = PositionalEncoding::new(100, 16);
        let encoding = pos_enc.encode(10);

        assert_eq!(encoding.shape(), &[10, 16]);

        // Check that different positions have different encodings
        let pos0 = encoding.row(0);
        let pos1 = encoding.row(1);
        let diff: f64 = (&pos1 - &pos0).iter().map(|x| x.abs()).sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_quantum_transformer() {
        let transformer = QuantumTransformer::new(8, 2, 2, 16, 100, 0.1);

        let batch_size = 1;
        let seq_len = 3;
        let embed_dim = 8;

        let input = Array2::zeros((batch_size, seq_len * embed_dim));
        let output = transformer.forward(&input, None).unwrap();

        assert_eq!(output.shape(), &[batch_size, seq_len * embed_dim]);
    }
}
