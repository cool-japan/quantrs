//! Quantum Long Short-Term Memory (QLSTM) and recurrent architectures.
//!
//! This module implements quantum versions of LSTM and other recurrent neural networks
//! for processing sequential data with quantum advantages.

use ndarray::{s, Array1, Array2, Array3};
use std::collections::HashMap;

use crate::error::{MLError, Result};
use crate::qnn::QNNLayer;
use crate::utils::VariationalCircuit;
use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::{multi::*, single::*, GateOp};

/// Quantum LSTM cell
#[derive(Debug, Clone)]
pub struct QLSTMCell {
    /// Number of qubits for hidden state
    hidden_qubits: usize,
    /// Number of qubits for cell state
    cell_qubits: usize,
    /// Input encoding qubits
    input_qubits: usize,
    /// Forget gate circuit
    forget_gate: VariationalCircuit,
    /// Input gate circuit
    input_gate: VariationalCircuit,
    /// Output gate circuit
    output_gate: VariationalCircuit,
    /// Candidate state circuit
    candidate_circuit: VariationalCircuit,
    /// Parameters
    parameters: HashMap<String, f64>,
}

impl QLSTMCell {
    /// Create a new QLSTM cell
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let input_qubits = (input_dim as f64).log2().ceil() as usize;
        let hidden_qubits = (hidden_dim as f64).log2().ceil() as usize;
        let cell_qubits = hidden_qubits;

        // Initialize gate circuits
        let total_qubits = input_qubits + hidden_qubits;

        let forget_gate = Self::create_gate_circuit(total_qubits, "forget");
        let input_gate = Self::create_gate_circuit(total_qubits, "input");
        let output_gate = Self::create_gate_circuit(total_qubits, "output");
        let candidate_circuit = Self::create_gate_circuit(total_qubits, "candidate");

        Self {
            hidden_qubits,
            cell_qubits,
            input_qubits,
            forget_gate,
            input_gate,
            output_gate,
            candidate_circuit,
            parameters: HashMap::new(),
        }
    }

    /// Create a gate circuit for LSTM
    fn create_gate_circuit(num_qubits: usize, gate_name: &str) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(num_qubits);

        // Layer 1: Hadamard initialization
        for q in 0..num_qubits {
            circuit.add_gate("H", vec![q], vec![]);
        }

        // Layer 2: Parameterized rotations
        for q in 0..num_qubits {
            circuit.add_gate("RY", vec![q], vec![format!("{}_{}_ry1", gate_name, q)]);
            circuit.add_gate("RZ", vec![q], vec![format!("{}_{}_rz1", gate_name, q)]);
        }

        // Layer 3: Entangling gates
        for q in 0..num_qubits - 1 {
            circuit.add_gate("CNOT", vec![q, q + 1], vec![]);
        }

        // Layer 4: Final rotations
        for q in 0..num_qubits {
            circuit.add_gate("RY", vec![q], vec![format!("{}_{}_ry2", gate_name, q)]);
        }

        circuit
    }

    /// Forward pass through LSTM cell
    pub fn forward(
        &self,
        input_state: &Array1<f64>,
        hidden_state: &Array1<f64>,
        cell_state: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        // Encode classical states to quantum
        let input_encoded = self.encode_classical_data(input_state)?;
        let hidden_encoded = self.encode_classical_data(hidden_state)?;

        // Compute forget gate
        let forget_output =
            self.compute_gate_output(&self.forget_gate, &input_encoded, &hidden_encoded)?;

        // Compute input gate
        let input_output =
            self.compute_gate_output(&self.input_gate, &input_encoded, &hidden_encoded)?;

        // Compute candidate values
        let candidate_output =
            self.compute_gate_output(&self.candidate_circuit, &input_encoded, &hidden_encoded)?;

        // Update cell state: C_t = f_t * C_{t-1} + i_t * C_tilde
        let mut new_cell_state = Array1::zeros(cell_state.len());
        for i in 0..cell_state.len() {
            new_cell_state[i] =
                forget_output[i] * cell_state[i] + input_output[i] * candidate_output[i];
        }

        // Compute output gate
        let output_gate_values =
            self.compute_gate_output(&self.output_gate, &input_encoded, &hidden_encoded)?;

        // Compute hidden state: h_t = o_t * tanh(C_t)
        let mut new_hidden_state = Array1::zeros(hidden_state.len());
        for i in 0..hidden_state.len() {
            new_hidden_state[i] = output_gate_values[i] * new_cell_state[i].tanh();
        }

        Ok((new_hidden_state, new_cell_state))
    }

    /// Encode classical data to quantum state
    fn encode_classical_data(&self, data: &Array1<f64>) -> Result<Vec<f64>> {
        // Amplitude encoding
        let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Err(MLError::InvalidInput("Zero norm input".to_string()));
        }

        Ok(data.iter().map(|x| x / norm).collect())
    }

    /// Compute gate output (simplified)
    fn compute_gate_output(
        &self,
        gate_circuit: &VariationalCircuit,
        input_encoded: &[f64],
        hidden_encoded: &[f64],
    ) -> Result<Array1<f64>> {
        // Simplified - would execute quantum circuit
        let output_dim = 2_usize.pow(self.hidden_qubits as u32);
        let mut output = Array1::zeros(output_dim);

        // Placeholder computation
        for i in 0..output_dim {
            output[i] = 0.5 + 0.5 * ((i as f64) * 0.1).sin();
        }

        Ok(output)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.forget_gate.num_parameters()
            + self.input_gate.num_parameters()
            + self.output_gate.num_parameters()
            + self.candidate_circuit.num_parameters()
    }
}

/// Quantum LSTM network
#[derive(Debug)]
pub struct QLSTM {
    /// LSTM cells for each layer
    cells: Vec<QLSTMCell>,
    /// Hidden dimensions
    hidden_dims: Vec<usize>,
    /// Whether to return sequences
    return_sequences: bool,
    /// Dropout rate
    dropout_rate: f64,
}

impl QLSTM {
    /// Create a new QLSTM network
    pub fn new(
        input_dim: usize,
        hidden_dims: Vec<usize>,
        return_sequences: bool,
        dropout_rate: f64,
    ) -> Self {
        let mut cells = Vec::new();

        // Create cells for each layer
        let mut prev_dim = input_dim;
        for &hidden_dim in &hidden_dims {
            cells.push(QLSTMCell::new(prev_dim, hidden_dim));
            prev_dim = hidden_dim;
        }

        Self {
            cells,
            hidden_dims,
            return_sequences,
            dropout_rate,
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, input_sequence: &Array2<f64>) -> Result<Array2<f64>> {
        let seq_len = input_sequence.nrows();
        let batch_size = 1; // Simplified for single sequence

        // Initialize hidden and cell states with small non-zero values
        let mut hidden_states: Vec<Array1<f64>> = self
            .hidden_dims
            .iter()
            .map(|&dim| Array1::from_elem(dim, 0.01))
            .collect();

        let mut cell_states: Vec<Array1<f64>> = self
            .hidden_dims
            .iter()
            .map(|&dim| Array1::from_elem(dim, 0.01))
            .collect();

        let mut outputs = Vec::new();

        // Process each time step
        for t in 0..seq_len {
            let input_t = input_sequence.row(t).to_owned();
            let mut layer_input = input_t;

            // Pass through each layer
            for (layer_idx, cell) in self.cells.iter().enumerate() {
                let (new_hidden, new_cell) = cell.forward(
                    &layer_input,
                    &hidden_states[layer_idx],
                    &cell_states[layer_idx],
                )?;

                hidden_states[layer_idx] = new_hidden.clone();
                cell_states[layer_idx] = new_cell;
                layer_input = new_hidden;
            }

            // Store output
            if self.return_sequences || t == seq_len - 1 {
                outputs.push(layer_input);
            }
        }

        // Convert outputs to Array2
        let output_dim = outputs[0].len();
        let mut output_array = Array2::zeros((outputs.len(), output_dim));
        for (i, output) in outputs.iter().enumerate() {
            output_array.row_mut(i).assign(output);
        }

        Ok(output_array)
    }

    /// Bidirectional QLSTM forward pass
    pub fn bidirectional_forward(&self, input_sequence: &Array2<f64>) -> Result<Array2<f64>> {
        // Forward pass
        let forward_output = self.forward(input_sequence)?;

        // Backward pass (reverse sequence)
        let mut reversed_input = input_sequence.clone();
        for i in 0..input_sequence.nrows() / 2 {
            let j = input_sequence.nrows() - 1 - i;
            for k in 0..input_sequence.ncols() {
                let tmp = reversed_input[[i, k]];
                reversed_input[[i, k]] = reversed_input[[j, k]];
                reversed_input[[j, k]] = tmp;
            }
        }
        let backward_output = self.forward(&reversed_input)?;

        // Concatenate outputs
        let output_dim = forward_output.ncols() + backward_output.ncols();
        let mut combined_output = Array2::zeros((forward_output.nrows(), output_dim));

        for i in 0..forward_output.nrows() {
            for j in 0..forward_output.ncols() {
                combined_output[[i, j]] = forward_output[[i, j]];
            }
            for j in 0..backward_output.ncols() {
                combined_output[[i, forward_output.ncols() + j]] =
                    backward_output[[backward_output.nrows() - 1 - i, j]];
            }
        }

        Ok(combined_output)
    }
}

/// Quantum Gated Recurrent Unit (QGRU)
#[derive(Debug)]
pub struct QGRUCell {
    /// Hidden dimension qubits
    hidden_qubits: usize,
    /// Update gate circuit
    update_gate: VariationalCircuit,
    /// Reset gate circuit
    reset_gate: VariationalCircuit,
    /// Candidate circuit
    candidate_circuit: VariationalCircuit,
}

impl QGRUCell {
    /// Create a new QGRU cell
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let input_qubits = (input_dim as f64).log2().ceil() as usize;
        let hidden_qubits = (hidden_dim as f64).log2().ceil() as usize;
        let total_qubits = input_qubits + hidden_qubits;

        Self {
            hidden_qubits,
            update_gate: QLSTMCell::create_gate_circuit(total_qubits, "update"),
            reset_gate: QLSTMCell::create_gate_circuit(total_qubits, "reset"),
            candidate_circuit: QLSTMCell::create_gate_circuit(total_qubits, "candidate"),
        }
    }

    /// Forward pass through GRU cell
    pub fn forward(
        &self,
        input_state: &Array1<f64>,
        hidden_state: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        // Simplified GRU computation
        // z_t = σ(W_z · [h_{t-1}, x_t])
        // r_t = σ(W_r · [h_{t-1}, x_t])
        // h_tilde = tanh(W · [r_t * h_{t-1}, x_t])
        // h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde

        let output_dim = hidden_state.len();
        let mut new_hidden = Array1::zeros(output_dim);

        // Placeholder computation
        for i in 0..output_dim {
            new_hidden[i] = 0.9 * hidden_state[i] + 0.1 * input_state[i % input_state.len()];
        }

        Ok(new_hidden)
    }
}

/// Quantum attention mechanism for sequence-to-sequence models
#[derive(Debug)]
pub struct QuantumAttention {
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Query circuit
    query_circuit: VariationalCircuit,
    /// Key circuit
    key_circuit: VariationalCircuit,
    /// Value circuit
    value_circuit: VariationalCircuit,
}

impl QuantumAttention {
    /// Create quantum attention layer
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let head_dim = embed_dim / num_heads;
        let num_qubits = (embed_dim as f64).log2().ceil() as usize;

        Self {
            num_heads,
            head_dim,
            query_circuit: Self::create_projection_circuit(num_qubits, "query"),
            key_circuit: Self::create_projection_circuit(num_qubits, "key"),
            value_circuit: Self::create_projection_circuit(num_qubits, "value"),
        }
    }

    /// Create projection circuit for Q, K, V
    fn create_projection_circuit(num_qubits: usize, name: &str) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(num_qubits);

        // Parameterized layer
        for q in 0..num_qubits {
            circuit.add_gate("RY", vec![q], vec![format!("{}_{}_theta", name, q)]);
            circuit.add_gate("RZ", vec![q], vec![format!("{}_{}_phi", name, q)]);
        }

        // Entangling layer
        for q in 0..num_qubits - 1 {
            circuit.add_gate("CZ", vec![q, q + 1], vec![]);
        }

        circuit
    }

    /// Compute attention scores
    pub fn forward(
        &self,
        query: &Array2<f64>,
        key: &Array2<f64>,
        value: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let seq_len = query.nrows();
        let embed_dim = query.ncols();

        // Simplified attention computation
        // Would compute Q, K, V projections using quantum circuits
        // Then compute attention scores as softmax(QK^T/√d_k)V

        let mut output = Array2::zeros((seq_len, embed_dim));

        // Placeholder
        for i in 0..seq_len {
            for j in 0..embed_dim {
                output[[i, j]] = 0.5 * query[[i, j]] + 0.3 * value[[i, j]];
            }
        }

        Ok(output)
    }
}

/// Sequence-to-sequence model with quantum components
#[derive(Debug)]
pub struct QuantumSeq2Seq {
    /// Encoder LSTM
    encoder: QLSTM,
    /// Decoder LSTM
    decoder: QLSTM,
    /// Attention mechanism
    attention: Option<QuantumAttention>,
    /// Output projection
    output_projection: QNNLayer,
}

impl QuantumSeq2Seq {
    /// Create a new seq2seq model
    pub fn new(
        input_vocab_size: usize,
        output_vocab_size: usize,
        embed_dim: usize,
        hidden_dims: Vec<usize>,
        use_attention: bool,
    ) -> Self {
        let encoder = QLSTM::new(embed_dim, hidden_dims.clone(), false, 0.1);
        let decoder = QLSTM::new(embed_dim, hidden_dims.clone(), true, 0.1);

        let attention = if use_attention {
            Some(QuantumAttention::new(
                hidden_dims.last().copied().unwrap_or(embed_dim),
                4,
            ))
        } else {
            None
        };

        let output_projection = QNNLayer::new(
            hidden_dims.last().copied().unwrap_or(embed_dim),
            output_vocab_size,
            crate::qnn::ActivationType::Linear,
        );

        Self {
            encoder,
            decoder,
            attention,
            output_projection,
        }
    }

    /// Encode input sequence
    pub fn encode(&self, input_sequence: &Array2<f64>) -> Result<Array2<f64>> {
        self.encoder.forward(input_sequence)
    }

    /// Decode with optional attention
    pub fn decode(
        &self,
        encoder_outputs: &Array2<f64>,
        decoder_input: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let decoder_outputs = self.decoder.forward(decoder_input)?;

        if let Some(attention) = &self.attention {
            // Apply attention
            attention.forward(&decoder_outputs, encoder_outputs, encoder_outputs)
        } else {
            Ok(decoder_outputs)
        }
    }
}

/// Training utilities for recurrent models
pub mod training {
    use super::*;
    use crate::autodiff::{optimizers::Adam, QuantumAutoDiff};

    /// Truncated backpropagation through time
    pub struct TBPTT {
        /// Truncation length
        truncation_length: usize,
        /// Gradient clipping value
        gradient_clip: f64,
    }

    impl TBPTT {
        pub fn new(truncation_length: usize, gradient_clip: f64) -> Self {
            Self {
                truncation_length,
                gradient_clip,
            }
        }

        /// Train QLSTM with TBPTT
        pub fn train_step(
            &self,
            model: &mut QLSTM,
            sequence: &Array2<f64>,
            targets: &Array2<f64>,
            optimizer: &mut Adam,
        ) -> Result<f64> {
            let seq_len = sequence.nrows();
            let mut total_loss = 0.0;

            // Process sequence in chunks
            for start in (0..seq_len).step_by(self.truncation_length) {
                let end = (start + self.truncation_length).min(seq_len);
                let chunk = sequence.slice(s![start..end, ..]).to_owned();
                let chunk_targets = targets.slice(s![start..end, ..]).to_owned();

                // Forward pass
                let outputs = model.forward(&chunk)?;

                // Compute loss (simplified)
                let loss = self.compute_loss(&outputs, &chunk_targets)?;
                total_loss += loss;

                // Backward pass would compute gradients
                // Clip gradients
                // Update parameters
            }

            Ok(total_loss / (seq_len as f64))
        }

        fn compute_loss(&self, outputs: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
            // MSE loss
            let diff = outputs - targets;
            Ok(diff.iter().map(|x| x * x).sum::<f64>() / diff.len() as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_qlstm_cell() {
        let cell = QLSTMCell::new(4, 4);

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let hidden = Array1::from_vec(vec![0.05, 0.05, 0.05, 0.05]);
        let cell_state = Array1::from_vec(vec![0.05, 0.05, 0.05, 0.05]);

        let (new_hidden, new_cell) = cell.forward(&input, &hidden, &cell_state).unwrap();

        assert_eq!(new_hidden.len(), 4);
        assert_eq!(new_cell.len(), 4);
    }

    #[test]
    fn test_qlstm_network() {
        let lstm = QLSTM::new(4, vec![8, 4], true, 0.1);

        let sequence = array![
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6]
        ];

        let output = lstm.forward(&sequence).unwrap();
        assert_eq!(output.nrows(), 3); // return_sequences=true
        assert_eq!(output.ncols(), 4); // Last hidden dim
    }

    #[test]
    fn test_bidirectional_lstm() {
        let lstm = QLSTM::new(4, vec![4], true, 0.0);

        let sequence = array![[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]];

        let output = lstm.bidirectional_forward(&sequence).unwrap();
        assert_eq!(output.nrows(), 2);
        assert_eq!(output.ncols(), 8); // Concatenated forward + backward
    }

    #[test]
    fn test_qgru_cell() {
        let gru = QGRUCell::new(4, 4);

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let hidden = Array1::zeros(4);

        let new_hidden = gru.forward(&input, &hidden).unwrap();
        assert_eq!(new_hidden.len(), 4);
    }

    #[test]
    fn test_quantum_attention() {
        let attention = QuantumAttention::new(16, 4);

        let seq_len = 3;
        let embed_dim = 16;
        let query = Array2::zeros((seq_len, embed_dim));
        let key = Array2::zeros((seq_len, embed_dim));
        let value = Array2::ones((seq_len, embed_dim));

        let output = attention.forward(&query, &key, &value).unwrap();
        assert_eq!(output.shape(), &[seq_len, embed_dim]);
    }
}
