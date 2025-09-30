//! Quantum Transformer Architectures
//!
//! This module implements quantum transformer models with quantum attention mechanisms,
//! position encoding, and multi-head attention for processing quantum and classical data
//! in transformer-style architectures.

use crate::error::{MLError, Result};
use crate::optimization::OptimizationMethod;
use crate::qnn::{QNNLayerType, QuantumNeuralNetwork};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, Axis};
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::gate::{multi::*, single::*, GateOp};
use quantrs2_sim::statevector::StateVectorSimulator;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Quantum transformer model configuration
#[derive(Debug, Clone)]
pub struct QuantumTransformerConfig {
    /// Model dimension (d_model)
    pub model_dim: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Feedforward dimension
    pub ff_dim: usize,

    /// Number of transformer layers
    pub num_layers: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Number of qubits for quantum computation
    pub num_qubits: usize,

    /// Dropout rate
    pub dropout_rate: f64,

    /// Attention mechanism type
    pub attention_type: QuantumAttentionType,

    /// Position encoding type
    pub position_encoding: PositionEncodingType,
}

/// Types of quantum attention mechanisms
#[derive(Debug, Clone)]
pub enum QuantumAttentionType {
    /// Full quantum attention with entanglement
    FullQuantum,

    /// Hybrid quantum-classical attention
    HybridQuantumClassical,

    /// Variational quantum attention
    VariationalQuantum,

    /// Quantum-enhanced multi-head attention
    QuantumEnhancedMultiHead,

    /// Quantum self-attention with superposition
    QuantumSelfAttention,
}

/// Position encoding types for quantum transformers
#[derive(Debug, Clone)]
pub enum PositionEncodingType {
    /// Sinusoidal position encoding
    Sinusoidal,

    /// Quantum position encoding with phase rotation
    QuantumPhase,

    /// Learnable quantum position encoding
    LearnableQuantum,

    /// Relative position encoding
    Relative,

    /// Rotary position embedding (RoPE)
    Rotary,
}

/// Quantum multi-head attention module
#[derive(Debug, Clone)]
pub struct QuantumMultiHeadAttention {
    /// Number of attention heads
    num_heads: usize,

    /// Model dimension
    model_dim: usize,

    /// Head dimension
    head_dim: usize,

    /// Query projection layers
    query_layers: Vec<QuantumNeuralNetwork>,

    /// Key projection layers
    key_layers: Vec<QuantumNeuralNetwork>,

    /// Value projection layers
    value_layers: Vec<QuantumNeuralNetwork>,

    /// Output projection
    output_projection: QuantumNeuralNetwork,

    /// Attention type
    attention_type: QuantumAttentionType,

    /// Quantum circuit for attention computation
    attention_circuit: Circuit<16>,
}

/// Quantum position encoding module
#[derive(Debug, Clone)]
pub struct QuantumPositionEncoding {
    /// Encoding type
    encoding_type: PositionEncodingType,

    /// Model dimension
    model_dim: usize,

    /// Maximum sequence length
    max_seq_len: usize,

    /// Learnable parameters (for learnable encodings)
    learnable_params: Option<Array2<f64>>,

    /// Quantum circuits for position encoding
    encoding_circuits: Vec<Circuit<16>>,
}

/// Quantum feedforward network
#[derive(Debug, Clone)]
pub struct QuantumFeedForward {
    /// Input dimension
    input_dim: usize,

    /// Hidden dimension
    hidden_dim: usize,

    /// Output dimension
    output_dim: usize,

    /// First layer
    layer1: QuantumNeuralNetwork,

    /// Second layer
    layer2: QuantumNeuralNetwork,

    /// Activation function type
    activation: ActivationType,

    /// Dropout rate
    dropout_rate: f64,
}

/// Activation function types for quantum networks
#[derive(Debug, Clone)]
pub enum ActivationType {
    /// Quantum ReLU using amplitude encoding
    QuantumReLU,

    /// Quantum GELU approximation
    QuantumGELU,

    /// Quantum Swish activation
    QuantumSwish,

    /// Parameterized quantum activation
    ParameterizedQuantum,

    /// Classical activation applied to measurement outcomes
    ClassicalHybrid,
}

/// Single quantum transformer layer
#[derive(Debug, Clone)]
pub struct QuantumTransformerLayer {
    /// Multi-head attention module
    attention: QuantumMultiHeadAttention,

    /// Feedforward network
    feedforward: QuantumFeedForward,

    /// Layer normalization parameters
    norm1_scale: Array1<f64>,
    norm1_bias: Array1<f64>,
    norm2_scale: Array1<f64>,
    norm2_bias: Array1<f64>,

    /// Model dimension
    model_dim: usize,

    /// Dropout rate
    dropout_rate: f64,
}

/// Main quantum transformer model
#[derive(Debug, Clone)]
pub struct QuantumTransformer {
    /// Model configuration
    config: QuantumTransformerConfig,

    /// Position encoding module
    position_encoding: QuantumPositionEncoding,

    /// Transformer layers
    layers: Vec<QuantumTransformerLayer>,

    /// Input embedding layer
    input_embedding: QuantumNeuralNetwork,

    /// Output projection layer
    output_projection: QuantumNeuralNetwork,

    /// Layer normalization at output
    final_norm_scale: Array1<f64>,
    final_norm_bias: Array1<f64>,
}

/// Attention computation result
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// Attention weights
    pub attention_weights: Array3<f64>,

    /// Output values
    pub output: Array3<f64>,

    /// Quantum state information
    pub quantum_info: QuantumAttentionInfo,
}

/// Quantum attention information
#[derive(Debug, Clone)]
pub struct QuantumAttentionInfo {
    /// Entanglement measures between positions
    pub entanglement_matrix: Array2<f64>,

    /// Quantum coherence scores
    pub coherence_scores: Array1<f64>,

    /// Superposition amplitudes
    pub superposition_amplitudes: Array2<f64>,

    /// Measurement probabilities
    pub measurement_probs: Array3<f64>,
}

impl QuantumTransformerConfig {
    /// Create default transformer configuration
    pub fn default() -> Self {
        Self {
            model_dim: 512,
            num_heads: 8,
            ff_dim: 2048,
            num_layers: 6,
            max_seq_len: 512,
            num_qubits: 10,
            dropout_rate: 0.1,
            attention_type: QuantumAttentionType::HybridQuantumClassical,
            position_encoding: PositionEncodingType::QuantumPhase,
        }
    }

    /// Create configuration for large model
    pub fn large() -> Self {
        Self {
            model_dim: 1024,
            num_heads: 16,
            ff_dim: 4096,
            num_layers: 12,
            max_seq_len: 1024,
            num_qubits: 16,
            dropout_rate: 0.1,
            attention_type: QuantumAttentionType::FullQuantum,
            position_encoding: PositionEncodingType::LearnableQuantum,
        }
    }

    /// Create configuration for small/efficient model
    pub fn small() -> Self {
        Self {
            model_dim: 256,
            num_heads: 4,
            ff_dim: 1024,
            num_layers: 4,
            max_seq_len: 256,
            num_qubits: 8,
            dropout_rate: 0.1,
            attention_type: QuantumAttentionType::VariationalQuantum,
            position_encoding: PositionEncodingType::Sinusoidal,
        }
    }
}

impl QuantumMultiHeadAttention {
    /// Create new quantum multi-head attention module
    pub fn new(
        num_heads: usize,
        model_dim: usize,
        attention_type: QuantumAttentionType,
        num_qubits: usize,
    ) -> Result<Self> {
        if model_dim % num_heads != 0 {
            return Err(MLError::ConfigurationError(
                "Model dimension must be divisible by number of heads".to_string(),
            ));
        }

        let head_dim = model_dim / num_heads;

        // Create projection layers for each head
        let mut query_layers = Vec::new();
        let mut key_layers = Vec::new();
        let mut value_layers = Vec::new();

        for _ in 0..num_heads {
            let q_layers = vec![
                QNNLayerType::EncodingLayer {
                    num_features: model_dim,
                },
                QNNLayerType::VariationalLayer {
                    num_params: head_dim * 2,
                },
                QNNLayerType::MeasurementLayer {
                    measurement_basis: "computational".to_string(),
                },
            ];
            query_layers.push(QuantumNeuralNetwork::new(
                q_layers, num_qubits, model_dim, head_dim,
            )?);

            let k_layers = vec![
                QNNLayerType::EncodingLayer {
                    num_features: model_dim,
                },
                QNNLayerType::VariationalLayer {
                    num_params: head_dim * 2,
                },
                QNNLayerType::MeasurementLayer {
                    measurement_basis: "computational".to_string(),
                },
            ];
            key_layers.push(QuantumNeuralNetwork::new(
                k_layers, num_qubits, model_dim, head_dim,
            )?);

            let v_layers = vec![
                QNNLayerType::EncodingLayer {
                    num_features: model_dim,
                },
                QNNLayerType::VariationalLayer {
                    num_params: head_dim * 2,
                },
                QNNLayerType::MeasurementLayer {
                    measurement_basis: "computational".to_string(),
                },
            ];
            value_layers.push(QuantumNeuralNetwork::new(
                v_layers, num_qubits, model_dim, head_dim,
            )?);
        }

        // Output projection layer
        let out_layers = vec![
            QNNLayerType::EncodingLayer {
                num_features: model_dim,
            },
            QNNLayerType::VariationalLayer {
                num_params: model_dim,
            },
            QNNLayerType::MeasurementLayer {
                measurement_basis: "computational".to_string(),
            },
        ];
        let output_projection =
            QuantumNeuralNetwork::new(out_layers, num_qubits, model_dim, model_dim)?;

        // Create attention computation circuit
        let attention_circuit = Self::create_attention_circuit(num_qubits, &attention_type)?;

        Ok(Self {
            num_heads,
            model_dim,
            head_dim,
            query_layers,
            key_layers,
            value_layers,
            output_projection,
            attention_type,
            attention_circuit,
        })
    }

    /// Create quantum circuit for attention computation
    fn create_attention_circuit(
        num_qubits: usize,
        attention_type: &QuantumAttentionType,
    ) -> Result<Circuit<16>> {
        let mut circuit = Circuit::<16>::new();

        match attention_type {
            QuantumAttentionType::FullQuantum => {
                // Create fully quantum attention circuit
                // Initialize superposition
                for i in 0..num_qubits.min(16) {
                    circuit.h(i);
                }

                // Add entangling gates
                for i in 0..num_qubits.min(15) {
                    circuit.cnot(i, i + 1);
                }

                // Add parameterized rotations
                for i in 0..num_qubits.min(16) {
                    circuit.ry(i, 0.0); // Will be parameterized
                }
            }

            QuantumAttentionType::HybridQuantumClassical => {
                // Hybrid approach circuit
                let half_qubits = (num_qubits / 2).min(8);
                for i in 0..half_qubits {
                    circuit.h(i);
                }

                for i in 0..half_qubits - 1 {
                    circuit.cnot(i, i + 1);
                }

                for i in 0..num_qubits.min(16) {
                    circuit.rx(i, 0.0); // Will be parameterized
                }
            }

            QuantumAttentionType::VariationalQuantum => {
                // Variational quantum attention circuit
                for layer in 0..3 {
                    for i in 0..num_qubits.min(16) {
                        circuit.ry(i, 0.0); // Will be parameterized
                        circuit.rz(i, 0.0); // Will be parameterized
                    }

                    for i in 0..num_qubits.min(15) {
                        circuit.cnot(i, i + 1);
                    }
                }
            }

            _ => {
                // Default attention circuit
                for i in 0..num_qubits.min(16) {
                    circuit.h(i);
                    circuit.ry(i, 0.0); // Will be parameterized
                }
            }
        }

        Ok(circuit)
    }

    /// Forward pass through quantum multi-head attention
    pub fn forward(
        &self,
        query: &Array3<f64>, // [batch_size, seq_len, model_dim]
        key: &Array3<f64>,
        value: &Array3<f64>,
        attention_mask: Option<&Array3<bool>>,
    ) -> Result<AttentionOutput> {
        let (batch_size, seq_len, model_dim) = query.dim();

        if model_dim != self.model_dim {
            return Err(MLError::DimensionMismatch(format!(
                "Expected model_dim {}, got {}",
                self.model_dim, model_dim
            )));
        }

        let mut all_head_outputs = Vec::new();
        let mut attention_weights_all = Array3::zeros((batch_size, seq_len, seq_len));
        let mut quantum_info = QuantumAttentionInfo {
            entanglement_matrix: Array2::zeros((seq_len, seq_len)),
            coherence_scores: Array1::zeros(seq_len),
            superposition_amplitudes: Array2::zeros((seq_len, self.head_dim)),
            measurement_probs: Array3::zeros((batch_size, seq_len, self.head_dim)),
        };

        // Process each attention head
        for head_idx in 0..self.num_heads {
            let head_output = self.compute_head_attention(
                query,
                key,
                value,
                head_idx,
                attention_mask,
                &mut quantum_info,
            )?;
            all_head_outputs.push(head_output.0);

            // Accumulate attention weights
            attention_weights_all = attention_weights_all + &head_output.1;
        }

        // Average attention weights across heads
        attention_weights_all = attention_weights_all / self.num_heads as f64;

        // Concatenate all head outputs
        let concatenated = self.concatenate_heads(&all_head_outputs)?;

        // Apply output projection
        let mut final_output = Array3::zeros((batch_size, seq_len, self.model_dim));
        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                let input = concatenated.slice(s![batch_idx, seq_idx, ..]).to_owned();
                let output = self.output_projection.forward(&input)?;
                final_output
                    .slice_mut(s![batch_idx, seq_idx, ..])
                    .assign(&output);
            }
        }

        Ok(AttentionOutput {
            attention_weights: attention_weights_all,
            output: final_output,
            quantum_info,
        })
    }

    /// Compute attention for a single head
    fn compute_head_attention(
        &self,
        query: &Array3<f64>,
        key: &Array3<f64>,
        value: &Array3<f64>,
        head_idx: usize,
        attention_mask: Option<&Array3<bool>>,
        quantum_info: &mut QuantumAttentionInfo,
    ) -> Result<(Array3<f64>, Array3<f64>)> {
        let (batch_size, seq_len, _) = query.dim();

        // Project query, key, value through quantum networks
        let mut q_proj = Array3::zeros((batch_size, seq_len, self.head_dim));
        let mut k_proj = Array3::zeros((batch_size, seq_len, self.head_dim));
        let mut v_proj = Array3::zeros((batch_size, seq_len, self.head_dim));

        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                let q_input = query.slice(s![batch_idx, seq_idx, ..]).to_owned();
                let k_input = key.slice(s![batch_idx, seq_idx, ..]).to_owned();
                let v_input = value.slice(s![batch_idx, seq_idx, ..]).to_owned();

                let q_out = self.query_layers[head_idx].forward(&q_input)?;
                let k_out = self.key_layers[head_idx].forward(&k_input)?;
                let v_out = self.value_layers[head_idx].forward(&v_input)?;

                q_proj.slice_mut(s![batch_idx, seq_idx, ..]).assign(&q_out);
                k_proj.slice_mut(s![batch_idx, seq_idx, ..]).assign(&k_out);
                v_proj.slice_mut(s![batch_idx, seq_idx, ..]).assign(&v_out);
            }
        }

        // Compute quantum attention scores
        let attention_scores =
            self.compute_quantum_attention_scores(&q_proj, &k_proj, quantum_info)?;

        // Apply attention mask if provided
        let masked_scores = if let Some(mask) = attention_mask {
            self.apply_attention_mask(&attention_scores, mask)?
        } else {
            attention_scores
        };

        // Apply softmax to get attention weights
        let attention_weights = self.quantum_softmax(&masked_scores)?;

        // Apply attention to values
        let output = self.apply_attention_to_values(&attention_weights, &v_proj)?;

        Ok((output, attention_weights))
    }

    /// Compute quantum attention scores using quantum circuits
    fn compute_quantum_attention_scores(
        &self,
        query: &Array3<f64>,
        key: &Array3<f64>,
        quantum_info: &mut QuantumAttentionInfo,
    ) -> Result<Array3<f64>> {
        let (batch_size, seq_len, head_dim) = query.dim();
        let mut attention_scores = Array3::zeros((batch_size, seq_len, seq_len));

        match self.attention_type {
            QuantumAttentionType::FullQuantum => {
                // Use quantum interference for attention computation
                for batch_idx in 0..batch_size {
                    for i in 0..seq_len {
                        for j in 0..seq_len {
                            let score = self.quantum_dot_product(
                                &query.slice(s![batch_idx, i, ..]).to_owned(),
                                &key.slice(s![batch_idx, j, ..]).to_owned(),
                                quantum_info,
                                i,
                                j,
                            )?;
                            attention_scores[[batch_idx, i, j]] = score;
                        }
                    }
                }
            }

            QuantumAttentionType::HybridQuantumClassical => {
                // Hybrid computation
                for batch_idx in 0..batch_size {
                    let q_batch = query.slice(s![batch_idx, .., ..]);
                    let k_batch = key.slice(s![batch_idx, .., ..]);

                    // Classical dot product with quantum enhancement
                    let classical_scores = q_batch.dot(&k_batch.t()) / (head_dim as f64).sqrt();

                    // Apply quantum enhancement
                    for i in 0..seq_len {
                        for j in 0..seq_len {
                            let quantum_enhancement = self.compute_quantum_enhancement(
                                &query.slice(s![batch_idx, i, ..]).to_owned(),
                                &key.slice(s![batch_idx, j, ..]).to_owned(),
                            )?;
                            attention_scores[[batch_idx, i, j]] =
                                classical_scores[[i, j]] * (1.0 + quantum_enhancement);
                        }
                    }
                }
            }

            _ => {
                // Default classical computation with quantum post-processing
                for batch_idx in 0..batch_size {
                    let q_batch = query.slice(s![batch_idx, .., ..]);
                    let k_batch = key.slice(s![batch_idx, .., ..]);
                    let scores = q_batch.dot(&k_batch.t()) / (head_dim as f64).sqrt();
                    attention_scores
                        .slice_mut(s![batch_idx, .., ..])
                        .assign(&scores);
                }
            }
        }

        Ok(attention_scores)
    }

    /// Compute quantum dot product with entanglement tracking
    fn quantum_dot_product(
        &self,
        vec1: &Array1<f64>,
        vec2: &Array1<f64>,
        quantum_info: &mut QuantumAttentionInfo,
        pos1: usize,
        pos2: usize,
    ) -> Result<f64> {
        let dim = vec1.len();
        let num_qubits = self.attention_circuit.num_qubits();

        // Encode vectors into quantum states
        let mut circuit = self.attention_circuit.clone();

        // Encode first vector
        for i in 0..dim.min(num_qubits / 2) {
            let angle = vec1[i] * PI;
            circuit.ry(i, angle);
        }

        // Encode second vector
        for i in 0..dim.min(num_qubits / 2) {
            let angle = vec2[i] * PI;
            let qubit_idx = i + num_qubits / 2;
            if qubit_idx < num_qubits {
                circuit.ry(qubit_idx, angle);
            }
        }

        // Add entangling operations for interference
        for i in 0..num_qubits / 2 {
            let target = i + num_qubits / 2;
            if target < num_qubits {
                circuit.cnot(i, target);
            }
        }

        // Simulate and extract dot product
        let simulator = StateVectorSimulator::new();
        let register = simulator.run(&circuit)?;
        let state_probs = register.probabilities();

        // Compute expectation value as dot product
        let dot_product = self.extract_dot_product_from_state(&state_probs)?;

        // Update quantum information
        let entanglement = self.compute_entanglement(&state_probs)?;
        quantum_info.entanglement_matrix[[pos1, pos2]] = entanglement;

        if pos1 < quantum_info.coherence_scores.len() {
            quantum_info.coherence_scores[pos1] = self.compute_coherence(&state_probs)?;
        }

        Ok(dot_product)
    }

    /// Extract dot product value from quantum state
    fn extract_dot_product_from_state(&self, state: &[f64]) -> Result<f64> {
        // Simplified extraction - compute overlap amplitude
        let mut dot_product = 0.0;

        for (i, &amplitude) in state.iter().enumerate() {
            // Weight by computational basis state
            let weight = if i % 2 == 0 { 1.0 } else { -1.0 };
            dot_product += weight * amplitude * amplitude;
        }

        Ok(dot_product)
    }

    /// Compute quantum enhancement factor
    fn compute_quantum_enhancement(&self, vec1: &Array1<f64>, vec2: &Array1<f64>) -> Result<f64> {
        // Quantum enhancement based on vector properties
        let norm1 = vec1.mapv(|x| x * x).sum().sqrt();
        let norm2 = vec2.mapv(|x| x * x).sum().sqrt();

        if norm1 < 1e-10 || norm2 < 1e-10 {
            return Ok(0.0);
        }

        // Compute quantum coherence enhancement
        let classical_dot = vec1.dot(vec2);
        let quantum_interference = (norm1 * norm2 - classical_dot.abs()).max(0.0);

        Ok(quantum_interference / (norm1 * norm2 + 1e-10))
    }

    /// Apply quantum softmax with temperature scaling
    fn quantum_softmax(&self, scores: &Array3<f64>) -> Result<Array3<f64>> {
        let mut weights = Array3::zeros(scores.dim());
        let temperature = 1.0; // Could be learnable parameter

        for batch_idx in 0..scores.dim().0 {
            for seq_idx in 0..scores.dim().1 {
                let row = scores.slice(s![batch_idx, seq_idx, ..]);
                let max_score = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                // Apply quantum-enhanced softmax
                let mut exp_scores = Array1::zeros(row.len());
                let mut sum_exp = 0.0;

                for (i, &score) in row.iter().enumerate() {
                    let enhanced_score = score / temperature;
                    let quantum_factor = self.compute_quantum_softmax_factor(score)?;
                    let exp_val = (enhanced_score - max_score + quantum_factor).exp();
                    exp_scores[i] = exp_val;
                    sum_exp += exp_val;
                }

                // Normalize
                if sum_exp > 1e-10 {
                    exp_scores = exp_scores / sum_exp;
                }

                weights
                    .slice_mut(s![batch_idx, seq_idx, ..])
                    .assign(&exp_scores);
            }
        }

        Ok(weights)
    }

    /// Compute quantum factor for softmax enhancement
    fn compute_quantum_softmax_factor(&self, score: f64) -> Result<f64> {
        // Apply quantum superposition effect to softmax
        let quantum_phase = (score * PI).sin().abs();
        Ok(0.1 * quantum_phase) // Small quantum enhancement
    }

    /// Apply attention weights to values
    fn apply_attention_to_values(
        &self,
        attention_weights: &Array3<f64>,
        values: &Array3<f64>,
    ) -> Result<Array3<f64>> {
        let (batch_size, seq_len, head_dim) = values.dim();
        let mut output = Array3::zeros((batch_size, seq_len, head_dim));

        for batch_idx in 0..batch_size {
            let weights = attention_weights.slice(s![batch_idx, .., ..]);
            let vals = values.slice(s![batch_idx, .., ..]);

            let attended_values = weights.dot(&vals);
            output
                .slice_mut(s![batch_idx, .., ..])
                .assign(&attended_values);
        }

        Ok(output)
    }

    /// Apply attention mask
    fn apply_attention_mask(
        &self,
        scores: &Array3<f64>,
        mask: &Array3<bool>,
    ) -> Result<Array3<f64>> {
        let mut masked_scores = scores.clone();

        for ((i, j, k), &should_mask) in mask.indexed_iter() {
            if should_mask {
                masked_scores[[i, j, k]] = f64::NEG_INFINITY;
            }
        }

        Ok(masked_scores)
    }

    /// Concatenate outputs from all attention heads
    fn concatenate_heads(&self, head_outputs: &[Array3<f64>]) -> Result<Array3<f64>> {
        let (batch_size, seq_len, head_dim) = head_outputs[0].dim();
        let mut concatenated = Array3::zeros((batch_size, seq_len, self.model_dim));

        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                let mut concat_vec = Array1::zeros(self.model_dim);

                for (head_idx, head_output) in head_outputs.iter().enumerate() {
                    let start_idx = head_idx * head_dim;
                    let end_idx = start_idx + head_dim;

                    concat_vec
                        .slice_mut(s![start_idx..end_idx])
                        .assign(&head_output.slice(s![batch_idx, seq_idx, ..]));
                }

                concatenated
                    .slice_mut(s![batch_idx, seq_idx, ..])
                    .assign(&concat_vec);
            }
        }

        Ok(concatenated)
    }

    /// Compute entanglement measure from quantum state
    fn compute_entanglement(&self, state: &[f64]) -> Result<f64> {
        // Simplified entanglement computation
        let num_qubits = (state.len() as f64).log2() as usize;
        if num_qubits < 2 {
            return Ok(0.0);
        }

        // Compute entropy-based entanglement measure
        let mut entanglement = 0.0;
        for &amplitude in state {
            let prob = amplitude * amplitude;
            if prob > 1e-10 {
                entanglement -= prob * prob.ln();
            }
        }

        Ok(entanglement / (num_qubits as f64))
    }

    /// Compute quantum coherence measure
    fn compute_coherence(&self, state: &[f64]) -> Result<f64> {
        // L1 norm coherence measure
        let mut coherence = 0.0;

        for (i, &amplitude) in state.iter().enumerate() {
            if i > 0 {
                // Exclude diagonal elements (|0><0|, |1><1|, etc.)
                coherence += amplitude.abs();
            }
        }

        Ok(coherence)
    }
}

impl QuantumPositionEncoding {
    /// Create new quantum position encoding
    pub fn new(
        encoding_type: PositionEncodingType,
        model_dim: usize,
        max_seq_len: usize,
        num_qubits: usize,
    ) -> Result<Self> {
        let mut encoding_circuits = Vec::new();
        let mut learnable_params = None;

        match encoding_type {
            PositionEncodingType::LearnableQuantum => {
                learnable_params = Some(Array2::zeros((max_seq_len, model_dim)));

                // Create quantum circuits for learnable position encoding
                for _ in 0..max_seq_len {
                    let mut circuit = Circuit::<16>::new();
                    for i in 0..num_qubits.min(16) {
                        circuit.ry(i, 0.0); // Will be parameterized
                        circuit.rz(i, 0.0); // Will be parameterized
                    }
                    encoding_circuits.push(circuit);
                }
            }

            PositionEncodingType::QuantumPhase => {
                // Create quantum phase encoding circuits
                for pos in 0..max_seq_len {
                    let mut circuit = Circuit::<16>::new();
                    for i in 0..num_qubits.min(16) {
                        let phase = 2.0 * PI * pos as f64
                            / (10000_f64.powf(2.0 * i as f64 / model_dim as f64));
                        circuit.h(i);
                        circuit.rz(i, phase);
                    }
                    encoding_circuits.push(circuit);
                }
            }

            _ => {
                // Default circuits for other encoding types
                for _ in 0..max_seq_len {
                    let mut circuit = Circuit::<16>::new();
                    for i in 0..num_qubits.min(16) {
                        circuit.h(i);
                        circuit.ry(i, 0.0); // Will be parameterized
                    }
                    encoding_circuits.push(circuit);
                }
            }
        }

        Ok(Self {
            encoding_type,
            model_dim,
            max_seq_len,
            learnable_params,
            encoding_circuits,
        })
    }

    /// Generate position encodings for input sequence
    pub fn forward(&self, seq_len: usize, batch_size: usize) -> Result<Array3<f64>> {
        let mut encodings = Array3::zeros((batch_size, seq_len, self.model_dim));

        match self.encoding_type {
            PositionEncodingType::Sinusoidal => {
                self.generate_sinusoidal_encoding(&mut encodings, seq_len)?;
            }

            PositionEncodingType::QuantumPhase => {
                self.generate_quantum_phase_encoding(&mut encodings, seq_len)?;
            }

            PositionEncodingType::LearnableQuantum => {
                self.generate_learnable_quantum_encoding(&mut encodings, seq_len)?;
            }

            PositionEncodingType::Relative => {
                self.generate_relative_encoding(&mut encodings, seq_len)?;
            }

            PositionEncodingType::Rotary => {
                self.generate_rotary_encoding(&mut encodings, seq_len)?;
            }
        }

        Ok(encodings)
    }

    /// Generate sinusoidal position encoding
    fn generate_sinusoidal_encoding(
        &self,
        encodings: &mut Array3<f64>,
        seq_len: usize,
    ) -> Result<()> {
        for pos in 0..seq_len {
            for i in 0..self.model_dim {
                let angle =
                    pos as f64 / 10000_f64.powf(2.0 * (i / 2) as f64 / self.model_dim as f64);

                let encoding_value = if i % 2 == 0 { angle.sin() } else { angle.cos() };

                // Apply to all batches
                for batch in 0..encodings.dim().0 {
                    encodings[[batch, pos, i]] = encoding_value;
                }
            }
        }

        Ok(())
    }

    /// Generate quantum phase position encoding
    fn generate_quantum_phase_encoding(
        &self,
        encodings: &mut Array3<f64>,
        seq_len: usize,
    ) -> Result<()> {
        let simulator = StateVectorSimulator::new();

        for pos in 0..seq_len {
            if pos < self.encoding_circuits.len() {
                let register = simulator.run(&self.encoding_circuits[pos])?;
                let state = register.probabilities();

                // Extract encoding from quantum state
                for i in 0..self.model_dim.min(state.len()) {
                    let encoding_value = state[i % state.len()];

                    for batch in 0..encodings.dim().0 {
                        encodings[[batch, pos, i]] = encoding_value;
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate learnable quantum position encoding
    fn generate_learnable_quantum_encoding(
        &self,
        encodings: &mut Array3<f64>,
        seq_len: usize,
    ) -> Result<()> {
        if let Some(ref params) = self.learnable_params {
            for pos in 0..seq_len.min(params.nrows()) {
                for i in 0..self.model_dim.min(params.ncols()) {
                    let encoding_value = params[[pos, i]];

                    for batch in 0..encodings.dim().0 {
                        encodings[[batch, pos, i]] = encoding_value;
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate relative position encoding
    fn generate_relative_encoding(
        &self,
        encodings: &mut Array3<f64>,
        seq_len: usize,
    ) -> Result<()> {
        // Simplified relative encoding
        for pos in 0..seq_len {
            for i in 0..self.model_dim {
                let relative_pos = (pos as f64 - seq_len as f64 / 2.0) / seq_len as f64;
                let encoding_value = relative_pos * (i as f64 / self.model_dim as f64);

                for batch in 0..encodings.dim().0 {
                    encodings[[batch, pos, i]] = encoding_value.tanh();
                }
            }
        }

        Ok(())
    }

    /// Generate rotary position embedding (RoPE)
    fn generate_rotary_encoding(&self, encodings: &mut Array3<f64>, seq_len: usize) -> Result<()> {
        for pos in 0..seq_len {
            for i in 0..(self.model_dim / 2) {
                let theta = pos as f64 / 10000_f64.powf(2.0 * i as f64 / self.model_dim as f64);

                let cos_val = theta.cos();
                let sin_val = theta.sin();

                for batch in 0..encodings.dim().0 {
                    encodings[[batch, pos, 2 * i]] = cos_val;
                    encodings[[batch, pos, 2 * i + 1]] = sin_val;
                }
            }
        }

        Ok(())
    }
}

impl QuantumFeedForward {
    /// Create new quantum feedforward network
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_qubits: usize,
        activation: ActivationType,
        dropout_rate: f64,
    ) -> Result<Self> {
        // First layer: input_dim -> hidden_dim
        let layer1_structure = vec![
            QNNLayerType::EncodingLayer {
                num_features: input_dim,
            },
            QNNLayerType::VariationalLayer {
                num_params: hidden_dim * 2,
            },
            QNNLayerType::EntanglementLayer {
                connectivity: "circular".to_string(),
            },
            QNNLayerType::VariationalLayer {
                num_params: hidden_dim,
            },
            QNNLayerType::MeasurementLayer {
                measurement_basis: "computational".to_string(),
            },
        ];
        let layer1 =
            QuantumNeuralNetwork::new(layer1_structure, num_qubits, input_dim, hidden_dim)?;

        // Second layer: hidden_dim -> output_dim
        let layer2_structure = vec![
            QNNLayerType::EncodingLayer {
                num_features: hidden_dim,
            },
            QNNLayerType::VariationalLayer {
                num_params: output_dim * 2,
            },
            QNNLayerType::MeasurementLayer {
                measurement_basis: "computational".to_string(),
            },
        ];
        let layer2 =
            QuantumNeuralNetwork::new(layer2_structure, num_qubits, hidden_dim, output_dim)?;

        Ok(Self {
            input_dim,
            hidden_dim,
            output_dim,
            layer1,
            layer2,
            activation,
            dropout_rate,
        })
    }

    /// Forward pass through feedforward network
    pub fn forward(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        let (batch_size, seq_len) = (input.nrows(), input.ncols() / self.input_dim);
        let mut output = Array2::zeros((batch_size, seq_len * self.output_dim));

        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                let start_idx = seq_idx * self.input_dim;
                let end_idx = start_idx + self.input_dim;

                let input_slice = input.slice(s![batch_idx, start_idx..end_idx]).to_owned();

                // First layer
                let hidden = self.layer1.forward(&input_slice)?;

                // Apply quantum activation
                let activated = self.apply_quantum_activation(&hidden)?;

                // Apply dropout (simplified)
                let dropped = self.apply_dropout(&activated)?;

                // Second layer
                let output_slice = self.layer2.forward(&dropped)?;

                let out_start = seq_idx * self.output_dim;
                let out_end = out_start + self.output_dim;
                output
                    .slice_mut(s![batch_idx, out_start..out_end])
                    .assign(&output_slice);
            }
        }

        Ok(output)
    }

    /// Apply quantum activation function
    fn apply_quantum_activation(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        match self.activation {
            ActivationType::QuantumReLU => {
                // Quantum ReLU using amplitude encoding
                Ok(input.mapv(|x| if x > 0.0 { x } else { 0.0 }))
            }

            ActivationType::QuantumGELU => {
                // Quantum GELU approximation
                Ok(input.mapv(|x| {
                    let gelu =
                        0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh());
                    gelu
                }))
            }

            ActivationType::QuantumSwish => {
                // Quantum Swish activation
                Ok(input.mapv(|x| x / (1.0 + (-x).exp())))
            }

            ActivationType::ParameterizedQuantum => {
                // Parameterized quantum activation
                Ok(input.mapv(|x| (x * PI / 2.0).sin()))
            }

            ActivationType::ClassicalHybrid => {
                // Classical activation applied to quantum measurements
                Ok(input.mapv(|x| x.tanh()))
            }
        }
    }

    /// Apply dropout
    fn apply_dropout(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        // Simplified dropout - in practice would be training-dependent
        if self.dropout_rate > 0.0 {
            let scale = 1.0 / (1.0 - self.dropout_rate);
            Ok(input.mapv(|x| {
                if fastrand::f64() < self.dropout_rate {
                    0.0
                } else {
                    x * scale
                }
            }))
        } else {
            Ok(input.clone())
        }
    }
}

impl QuantumTransformerLayer {
    /// Create new quantum transformer layer
    pub fn new(
        model_dim: usize,
        num_heads: usize,
        ff_dim: usize,
        num_qubits: usize,
        attention_type: QuantumAttentionType,
        dropout_rate: f64,
    ) -> Result<Self> {
        let attention =
            QuantumMultiHeadAttention::new(num_heads, model_dim, attention_type, num_qubits)?;

        let feedforward = QuantumFeedForward::new(
            model_dim,
            ff_dim,
            model_dim,
            num_qubits,
            ActivationType::QuantumGELU,
            dropout_rate,
        )?;

        let norm1_scale = Array1::ones(model_dim);
        let norm1_bias = Array1::zeros(model_dim);
        let norm2_scale = Array1::ones(model_dim);
        let norm2_bias = Array1::zeros(model_dim);

        Ok(Self {
            attention,
            feedforward,
            norm1_scale,
            norm1_bias,
            norm2_scale,
            norm2_bias,
            model_dim,
            dropout_rate,
        })
    }

    /// Forward pass through transformer layer
    pub fn forward(
        &self,
        input: &Array3<f64>,
        attention_mask: Option<&Array3<bool>>,
    ) -> Result<Array3<f64>> {
        // Self-attention with residual connection and layer norm
        let attention_output = self
            .attention
            .forward(input, input, input, attention_mask)?;
        let attended = input + &attention_output.output;
        let normed1 = self.layer_norm(&attended, &self.norm1_scale, &self.norm1_bias)?;

        // Feedforward with residual connection and layer norm
        let ff_input = self.reshape_for_feedforward(&normed1)?;
        let ff_output = self.feedforward.forward(&ff_input)?;
        let ff_reshaped = self.reshape_from_feedforward(&ff_output, normed1.dim())?;
        let ff_residual = &normed1 + &ff_reshaped;
        let normed2 = self.layer_norm(&ff_residual, &self.norm2_scale, &self.norm2_bias)?;

        Ok(normed2)
    }

    /// Apply layer normalization
    fn layer_norm(
        &self,
        input: &Array3<f64>,
        scale: &Array1<f64>,
        bias: &Array1<f64>,
    ) -> Result<Array3<f64>> {
        let (batch_size, seq_len, model_dim) = input.dim();
        let mut output = Array3::zeros((batch_size, seq_len, model_dim));

        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                let input_slice = input.slice(s![batch_idx, seq_idx, ..]);

                // Compute mean and variance
                let mean = input_slice.mean().unwrap_or(0.0);
                let variance = input_slice
                    .mapv(|x| (x - mean).powi(2))
                    .mean()
                    .unwrap_or(1.0);
                let std = (variance + 1e-6).sqrt();

                // Normalize
                let normalized = input_slice.mapv(|x| (x - mean) / std);

                // Scale and shift
                let scaled = &normalized * scale + bias;
                output.slice_mut(s![batch_idx, seq_idx, ..]).assign(&scaled);
            }
        }

        Ok(output)
    }

    /// Reshape for feedforward processing
    fn reshape_for_feedforward(&self, input: &Array3<f64>) -> Result<Array2<f64>> {
        let (batch_size, seq_len, model_dim) = input.dim();
        let mut output = Array2::zeros((batch_size, seq_len * model_dim));

        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                let start_idx = seq_idx * model_dim;
                let end_idx = start_idx + model_dim;

                output
                    .slice_mut(s![batch_idx, start_idx..end_idx])
                    .assign(&input.slice(s![batch_idx, seq_idx, ..]));
            }
        }

        Ok(output)
    }

    /// Reshape from feedforward processing
    fn reshape_from_feedforward(
        &self,
        input: &Array2<f64>,
        target_shape: (usize, usize, usize),
    ) -> Result<Array3<f64>> {
        let (batch_size, seq_len, model_dim) = target_shape;
        let mut output = Array3::zeros((batch_size, seq_len, model_dim));

        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                let start_idx = seq_idx * model_dim;
                let end_idx = start_idx + model_dim;

                output
                    .slice_mut(s![batch_idx, seq_idx, ..])
                    .assign(&input.slice(s![batch_idx, start_idx..end_idx]));
            }
        }

        Ok(output)
    }
}

impl QuantumTransformer {
    /// Create new quantum transformer model
    pub fn new(config: QuantumTransformerConfig) -> Result<Self> {
        // Position encoding
        let position_encoding = QuantumPositionEncoding::new(
            config.position_encoding.clone(),
            config.model_dim,
            config.max_seq_len,
            config.num_qubits,
        )?;

        // Transformer layers
        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            let layer = QuantumTransformerLayer::new(
                config.model_dim,
                config.num_heads,
                config.ff_dim,
                config.num_qubits,
                config.attention_type.clone(),
                config.dropout_rate,
            )?;
            layers.push(layer);
        }

        // Input embedding layer
        let embedding_layers = vec![
            QNNLayerType::EncodingLayer {
                num_features: config.model_dim,
            },
            QNNLayerType::VariationalLayer {
                num_params: config.model_dim,
            },
            QNNLayerType::MeasurementLayer {
                measurement_basis: "computational".to_string(),
            },
        ];
        let input_embedding = QuantumNeuralNetwork::new(
            embedding_layers,
            config.num_qubits,
            config.model_dim,
            config.model_dim,
        )?;

        // Output projection layer
        let output_layers = vec![
            QNNLayerType::EncodingLayer {
                num_features: config.model_dim,
            },
            QNNLayerType::VariationalLayer {
                num_params: config.model_dim,
            },
            QNNLayerType::MeasurementLayer {
                measurement_basis: "computational".to_string(),
            },
        ];
        let output_projection = QuantumNeuralNetwork::new(
            output_layers,
            config.num_qubits,
            config.model_dim,
            config.model_dim,
        )?;

        // Final layer normalization
        let final_norm_scale = Array1::ones(config.model_dim);
        let final_norm_bias = Array1::zeros(config.model_dim);

        Ok(Self {
            config,
            position_encoding,
            layers,
            input_embedding,
            output_projection,
            final_norm_scale,
            final_norm_bias,
        })
    }

    /// Forward pass through quantum transformer
    pub fn forward(
        &self,
        input: &Array3<f64>, // [batch_size, seq_len, input_dim]
        attention_mask: Option<&Array3<bool>>,
    ) -> Result<Array3<f64>> {
        let (batch_size, seq_len, input_dim) = input.dim();

        if seq_len > self.config.max_seq_len {
            return Err(MLError::ConfigurationError(format!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.config.max_seq_len
            )));
        }

        // Input embedding
        let mut embedded = Array3::zeros((batch_size, seq_len, self.config.model_dim));
        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                let input_vec = input.slice(s![batch_idx, seq_idx, ..]).to_owned();

                // Pad or truncate to model_dim
                let mut padded_input = Array1::zeros(self.config.model_dim);
                let copy_len = input_dim.min(self.config.model_dim);
                padded_input
                    .slice_mut(s![..copy_len])
                    .assign(&input_vec.slice(s![..copy_len]));

                let embedding_output = self.input_embedding.forward(&padded_input)?;
                embedded
                    .slice_mut(s![batch_idx, seq_idx, ..])
                    .assign(&embedding_output);
            }
        }

        // Add position encoding
        let position_encodings = self.position_encoding.forward(seq_len, batch_size)?;
        let mut x = embedded + position_encodings;

        // Pass through transformer layers
        for layer in &self.layers {
            x = layer.forward(&x, attention_mask)?;
        }

        // Apply final layer normalization
        x = self.apply_final_layer_norm(&x)?;

        // Output projection
        let mut output = Array3::zeros((batch_size, seq_len, self.config.model_dim));
        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                let input_vec = x.slice(s![batch_idx, seq_idx, ..]).to_owned();
                let projected_output = self.output_projection.forward(&input_vec)?;
                output
                    .slice_mut(s![batch_idx, seq_idx, ..])
                    .assign(&projected_output);
            }
        }

        Ok(output)
    }

    /// Apply final layer normalization
    fn apply_final_layer_norm(&self, input: &Array3<f64>) -> Result<Array3<f64>> {
        let (batch_size, seq_len, model_dim) = input.dim();
        let mut output = Array3::zeros((batch_size, seq_len, model_dim));

        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                let input_slice = input.slice(s![batch_idx, seq_idx, ..]);

                let mean = input_slice.mean().unwrap_or(0.0);
                let variance = input_slice
                    .mapv(|x| (x - mean).powi(2))
                    .mean()
                    .unwrap_or(1.0);
                let std = (variance + 1e-6).sqrt();

                let normalized = input_slice.mapv(|x| (x - mean) / std);
                let scaled = &normalized * &self.final_norm_scale + &self.final_norm_bias;

                output.slice_mut(s![batch_idx, seq_idx, ..]).assign(&scaled);
            }
        }

        Ok(output)
    }

    /// Get model configuration
    pub fn config(&self) -> &QuantumTransformerConfig {
        &self.config
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Count parameters in all components
        total += self.input_embedding.parameters.len();
        total += self.output_projection.parameters.len();

        for layer in &self.layers {
            total += layer
                .attention
                .query_layers
                .iter()
                .map(|l| l.parameters.len())
                .sum::<usize>();
            total += layer
                .attention
                .key_layers
                .iter()
                .map(|l| l.parameters.len())
                .sum::<usize>();
            total += layer
                .attention
                .value_layers
                .iter()
                .map(|l| l.parameters.len())
                .sum::<usize>();
            total += layer.attention.output_projection.parameters.len();
            total += layer.feedforward.layer1.parameters.len();
            total += layer.feedforward.layer2.parameters.len();
            total += layer.norm1_scale.len() + layer.norm1_bias.len();
            total += layer.norm2_scale.len() + layer.norm2_bias.len();
        }

        if let Some(ref params) = self.position_encoding.learnable_params {
            total += params.len();
        }

        total += self.final_norm_scale.len() + self.final_norm_bias.len();

        total
    }
}

/// Helper function to create causal attention mask
pub fn create_causal_mask(batch_size: usize, seq_len: usize) -> Array3<bool> {
    let mut mask = Array3::from_elem((batch_size, seq_len, seq_len), false);

    for batch_idx in 0..batch_size {
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask[[batch_idx, i, j]] = true; // Mask future positions
            }
        }
    }

    mask
}

/// Helper function to create padding mask
pub fn create_padding_mask(
    batch_size: usize,
    seq_len: usize,
    actual_lengths: &[usize],
) -> Array3<bool> {
    let mut mask = Array3::from_elem((batch_size, seq_len, seq_len), false);

    for (batch_idx, &actual_len) in actual_lengths.iter().enumerate() {
        if batch_idx < batch_size {
            for i in 0..seq_len {
                for j in actual_len..seq_len {
                    mask[[batch_idx, i, j]] = true; // Mask padding positions
                }
            }
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_transformer_config() {
        let config = QuantumTransformerConfig::default();
        assert_eq!(config.model_dim, 512);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_layers, 6);

        let large_config = QuantumTransformerConfig::large();
        assert_eq!(large_config.model_dim, 1024);
        assert_eq!(large_config.num_heads, 16);
    }

    #[test]
    fn test_quantum_multi_head_attention_creation() {
        let attention = QuantumMultiHeadAttention::new(
            8,
            512,
            QuantumAttentionType::HybridQuantumClassical,
            10,
        );

        assert!(attention.is_ok());
        let attn = attention.unwrap();
        assert_eq!(attn.num_heads, 8);
        assert_eq!(attn.model_dim, 512);
        assert_eq!(attn.head_dim, 64);
    }

    #[test]
    fn test_quantum_position_encoding() {
        let pos_enc = QuantumPositionEncoding::new(PositionEncodingType::Sinusoidal, 256, 512, 8);

        assert!(pos_enc.is_ok());
        let pe = pos_enc.unwrap();
        assert_eq!(pe.model_dim, 256);
        assert_eq!(pe.max_seq_len, 512);
    }

    #[test]
    fn test_quantum_feedforward() {
        let ff = QuantumFeedForward::new(256, 1024, 256, 8, ActivationType::QuantumGELU, 0.1);

        assert!(ff.is_ok());
        let feedforward = ff.unwrap();
        assert_eq!(feedforward.input_dim, 256);
        assert_eq!(feedforward.hidden_dim, 1024);
        assert_eq!(feedforward.output_dim, 256);
    }

    #[test]
    fn test_causal_mask_creation() {
        let mask = create_causal_mask(2, 4);
        assert_eq!(mask.dim(), (2, 4, 4));

        // Check that lower triangle is false (not masked)
        assert!(!mask[[0, 0, 0]]);
        assert!(!mask[[0, 1, 0]]);
        assert!(!mask[[0, 1, 1]]);

        // Check that upper triangle is true (masked)
        assert!(mask[[0, 0, 1]]);
        assert!(mask[[0, 0, 2]]);
        assert!(mask[[0, 1, 2]]);
    }

    #[test]
    fn test_padding_mask_creation() {
        let actual_lengths = vec![3, 2];
        let mask = create_padding_mask(2, 4, &actual_lengths);

        // Check padding is masked for first batch (length 3)
        assert!(!mask[[0, 0, 2]]); // Not masked (within length)
        assert!(mask[[0, 0, 3]]); // Masked (padding)

        // Check padding is masked for second batch (length 2)
        assert!(!mask[[1, 0, 1]]); // Not masked (within length)
        assert!(mask[[1, 0, 2]]); // Masked (padding)
        assert!(mask[[1, 0, 3]]); // Masked (padding)
    }
}
