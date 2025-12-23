//! Quantum Graph Neural Networks (GNNs) implementation.
//!
//! This module provides quantum versions of graph neural networks including
//! graph convolutional networks, graph attention networks, and message passing.

use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::autodiff::DifferentiableParam;
use crate::error::{MLError, Result};
use crate::utils::VariationalCircuit;
use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::{multi::*, single::*, GateOp};

/// Activation function types for quantum layers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    /// Linear activation (identity)
    Linear,
    /// ReLU activation
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
}

/// Graph structure for quantum processing
#[derive(Debug, Clone)]
pub struct QuantumGraph {
    /// Number of nodes
    num_nodes: usize,
    /// Adjacency matrix
    adjacency: Array2<f64>,
    /// Node features
    node_features: Array2<f64>,
    /// Edge features (optional)
    edge_features: Option<HashMap<(usize, usize), Array1<f64>>>,
    /// Graph-level features (optional)
    graph_features: Option<Array1<f64>>,
}

impl QuantumGraph {
    /// Create a new quantum graph
    pub fn new(num_nodes: usize, edges: Vec<(usize, usize)>, node_features: Array2<f64>) -> Self {
        let mut adjacency = Array2::zeros((num_nodes, num_nodes));

        // Build adjacency matrix
        for (src, dst) in edges {
            adjacency[[src, dst]] = 1.0;
            adjacency[[dst, src]] = 1.0; // Undirected graph
        }

        Self {
            num_nodes,
            adjacency,
            node_features,
            edge_features: None,
            graph_features: None,
        }
    }

    /// Add edge features
    pub fn with_edge_features(
        mut self,
        edge_features: HashMap<(usize, usize), Array1<f64>>,
    ) -> Self {
        self.edge_features = Some(edge_features);
        self
    }

    /// Add graph-level features
    pub fn with_graph_features(mut self, graph_features: Array1<f64>) -> Self {
        self.graph_features = Some(graph_features);
        self
    }

    /// Get node degree
    pub fn degree(&self, node: usize) -> usize {
        self.adjacency
            .row(node)
            .iter()
            .filter(|&&x| x > 0.0)
            .count()
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        self.adjacency
            .row(node)
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > 0.0)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Compute Laplacian matrix
    pub fn laplacian(&self) -> Array2<f64> {
        let mut degree_matrix = Array2::zeros((self.num_nodes, self.num_nodes));
        for i in 0..self.num_nodes {
            degree_matrix[[i, i]] = self.degree(i) as f64;
        }
        &degree_matrix - &self.adjacency
    }

    /// Compute normalized Laplacian
    pub fn normalized_laplacian(&self) -> Array2<f64> {
        let mut degree_matrix = Array2::zeros((self.num_nodes, self.num_nodes));
        let mut degree_sqrt_inv = Array1::zeros(self.num_nodes);

        for i in 0..self.num_nodes {
            let degree = self.degree(i) as f64;
            degree_matrix[[i, i]] = degree;
            if degree > 0.0 {
                degree_sqrt_inv[i] = 1.0 / degree.sqrt();
            }
        }

        let mut norm_laplacian = Array2::eye(self.num_nodes);
        for i in 0..self.num_nodes {
            for j in 0..self.num_nodes {
                if self.adjacency[[i, j]] > 0.0 {
                    norm_laplacian[[i, j]] -=
                        degree_sqrt_inv[i] * self.adjacency[[i, j]] * degree_sqrt_inv[j];
                }
            }
        }

        norm_laplacian
    }
}

/// Quantum Graph Convolutional Layer
#[derive(Debug)]
pub struct QuantumGCNLayer {
    /// Input feature dimension
    input_dim: usize,
    /// Output feature dimension
    output_dim: usize,
    /// Number of qubits
    num_qubits: usize,
    /// Variational circuit for node transformation
    node_circuit: VariationalCircuit,
    /// Variational circuit for aggregation
    aggregation_circuit: VariationalCircuit,
    /// Parameters
    parameters: HashMap<String, f64>,
    /// Activation type
    activation: ActivationType,
}

impl QuantumGCNLayer {
    /// Create a new quantum GCN layer
    pub fn new(input_dim: usize, output_dim: usize, activation: ActivationType) -> Self {
        let num_qubits = ((input_dim.max(output_dim)) as f64).log2().ceil() as usize;
        let node_circuit = Self::build_node_circuit(num_qubits);
        let aggregation_circuit = Self::build_aggregation_circuit(num_qubits);

        Self {
            input_dim,
            output_dim,
            num_qubits,
            node_circuit,
            aggregation_circuit,
            parameters: HashMap::new(),
            activation,
        }
    }

    /// Build node transformation circuit
    fn build_node_circuit(num_qubits: usize) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(num_qubits);

        // Layer 1: Feature encoding
        for q in 0..num_qubits {
            circuit.add_gate("RY", vec![q], vec![format!("node_encode_{}", q)]);
        }

        // Layer 2: Entangling
        for layer in 0..2 {
            for q in 0..num_qubits - 1 {
                circuit.add_gate("CNOT", vec![q, q + 1], vec![]);
            }
            if num_qubits > 2 {
                circuit.add_gate("CNOT", vec![num_qubits - 1, 0], vec![]);
            }

            // Parameterized rotations
            for q in 0..num_qubits {
                circuit.add_gate("RX", vec![q], vec![format!("node_rx_{}_{}", layer, q)]);
                circuit.add_gate("RZ", vec![q], vec![format!("node_rz_{}_{}", layer, q)]);
            }
        }

        circuit
    }

    /// Build aggregation circuit
    fn build_aggregation_circuit(num_qubits: usize) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(num_qubits * 2); // For neighbor aggregation

        // Combine node and neighbor features
        for q in 0..num_qubits {
            circuit.add_gate("CZ", vec![q, q + num_qubits], vec![]);
        }

        // Mixing layer
        for q in 0..num_qubits * 2 {
            circuit.add_gate("RY", vec![q], vec![format!("agg_ry_{}", q)]);
        }

        // Entangling
        for q in 0..num_qubits * 2 - 1 {
            circuit.add_gate("CNOT", vec![q, q + 1], vec![]);
        }

        // Final rotation
        for q in 0..num_qubits {
            circuit.add_gate("RX", vec![q], vec![format!("agg_final_{}", q)]);
        }

        circuit
    }

    /// Forward pass through GCN layer
    pub fn forward(&self, graph: &QuantumGraph) -> Result<Array2<f64>> {
        let mut output_features = Array2::zeros((graph.num_nodes, self.output_dim));

        // Process each node
        for node in 0..graph.num_nodes {
            // Get node features
            let node_feat = graph.node_features.row(node);

            // Get neighbor features
            let neighbors = graph.neighbors(node);
            let mut aggregated = Array1::zeros(self.input_dim);

            // Aggregate neighbor features
            for &neighbor in &neighbors {
                let neighbor_feat = graph.node_features.row(neighbor);
                aggregated = &aggregated + &neighbor_feat.to_owned();
            }

            // Normalize by degree
            let degree = neighbors.len().max(1) as f64;
            aggregated = aggregated / degree;

            // Apply quantum transformation
            let transformed = self.quantum_transform(&node_feat.to_owned(), &aggregated)?;

            // Store output
            for i in 0..self.output_dim {
                output_features[[node, i]] = transformed[i];
            }
        }

        Ok(output_features)
    }

    /// Apply quantum transformation
    fn quantum_transform(
        &self,
        node_features: &Array1<f64>,
        aggregated_features: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        // Encode features into quantum state
        let node_encoded = self.encode_features(node_features)?;
        let agg_encoded = self.encode_features(aggregated_features)?;

        // Apply quantum circuits (simplified)
        let mut output = Array1::zeros(self.output_dim);

        // Placeholder computation
        for i in 0..self.output_dim {
            let idx_node = i % node_features.len();
            let idx_agg = i % aggregated_features.len();

            output[i] = match self.activation {
                ActivationType::ReLU => {
                    (0.5 * node_features[idx_node] + 0.5 * aggregated_features[idx_agg]).max(0.0)
                }
                ActivationType::Tanh => {
                    (0.5 * node_features[idx_node] + 0.5 * aggregated_features[idx_agg]).tanh()
                }
                ActivationType::Sigmoid => {
                    let x = 0.5 * node_features[idx_node] + 0.5 * aggregated_features[idx_agg];
                    1.0 / (1.0 + (-x).exp())
                }
                ActivationType::Linear => {
                    0.5 * node_features[idx_node] + 0.5 * aggregated_features[idx_agg]
                }
            };
        }

        Ok(output)
    }

    /// Encode classical features to quantum state
    fn encode_features(&self, features: &Array1<f64>) -> Result<Vec<Complex64>> {
        let state_dim = 2_usize.pow(self.num_qubits as u32);
        let mut quantum_state = vec![Complex64::new(0.0, 0.0); state_dim];

        // Amplitude encoding
        let norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            quantum_state[0] = Complex64::new(1.0, 0.0);
        } else {
            for (i, &val) in features.iter().enumerate() {
                if i < state_dim {
                    quantum_state[i] = Complex64::new(val / norm, 0.0);
                }
            }
        }

        Ok(quantum_state)
    }
}

/// Quantum Graph Attention Layer
#[derive(Debug)]
pub struct QuantumGATLayer {
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Attention circuits for each head
    attention_circuits: Vec<VariationalCircuit>,
    /// Feature transformation circuits
    transform_circuits: Vec<VariationalCircuit>,
    /// Dropout rate
    dropout_rate: f64,
}

impl QuantumGATLayer {
    /// Create a new quantum GAT layer
    pub fn new(input_dim: usize, output_dim: usize, num_heads: usize, dropout_rate: f64) -> Self {
        let mut attention_circuits = Vec::new();
        let mut transform_circuits = Vec::new();

        let qubits_per_head = ((output_dim / num_heads) as f64).log2().ceil() as usize;

        for _ in 0..num_heads {
            attention_circuits.push(Self::build_attention_circuit(qubits_per_head));
            transform_circuits.push(Self::build_transform_circuit(qubits_per_head));
        }

        Self {
            input_dim,
            output_dim,
            num_heads,
            attention_circuits,
            transform_circuits,
            dropout_rate,
        }
    }

    /// Build attention circuit
    fn build_attention_circuit(num_qubits: usize) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(num_qubits * 2);

        // Attention computation between node pairs
        for q in 0..num_qubits {
            circuit.add_gate("RY", vec![q], vec![format!("att_src_{}", q)]);
            circuit.add_gate("RY", vec![q + num_qubits], vec![format!("att_dst_{}", q)]);
        }

        // Interaction layer
        for q in 0..num_qubits {
            circuit.add_gate("CZ", vec![q, q + num_qubits], vec![]);
        }

        // Attention score computation
        circuit.add_gate("H", vec![0], vec![]);
        for q in 1..num_qubits * 2 {
            circuit.add_gate("CNOT", vec![0, q], vec![]);
        }

        circuit
    }

    /// Build feature transformation circuit
    fn build_transform_circuit(num_qubits: usize) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(num_qubits);

        // Feature transformation
        for layer in 0..2 {
            for q in 0..num_qubits {
                circuit.add_gate("RY", vec![q], vec![format!("trans_ry_{}_{}", layer, q)]);
                circuit.add_gate("RZ", vec![q], vec![format!("trans_rz_{}_{}", layer, q)]);
            }

            // Entangling
            for q in 0..num_qubits - 1 {
                circuit.add_gate("CX", vec![q, q + 1], vec![]);
            }
        }

        circuit
    }

    /// Forward pass
    pub fn forward(&self, graph: &QuantumGraph) -> Result<Array2<f64>> {
        let head_dim = self.output_dim / self.num_heads;
        let mut all_head_outputs = Vec::new();

        // Process each attention head
        for head in 0..self.num_heads {
            let head_output = self.process_attention_head(graph, head)?;
            all_head_outputs.push(head_output);
        }

        // Concatenate heads
        let mut output = Array2::zeros((graph.num_nodes, self.output_dim));
        for (h, head_output) in all_head_outputs.iter().enumerate() {
            for node in 0..graph.num_nodes {
                for d in 0..head_dim {
                    output[[node, h * head_dim + d]] = head_output[[node, d]];
                }
            }
        }

        Ok(output)
    }

    /// Process single attention head
    fn process_attention_head(&self, graph: &QuantumGraph, head: usize) -> Result<Array2<f64>> {
        let head_dim = self.output_dim / self.num_heads;
        let mut output = Array2::zeros((graph.num_nodes, head_dim));

        // Compute attention scores
        let attention_scores = self.compute_attention_scores(graph, head)?;

        // Apply attention to features
        for node in 0..graph.num_nodes {
            let neighbors = graph.neighbors(node);
            let feature_dim = graph.node_features.ncols();
            let mut weighted_features = Array1::zeros(feature_dim);

            // Self-attention
            let self_score = attention_scores[[node, node]];
            weighted_features =
                &weighted_features + &(&graph.node_features.row(node).to_owned() * self_score);

            // Neighbor attention
            for &neighbor in &neighbors {
                let score = attention_scores[[node, neighbor]];
                weighted_features =
                    &weighted_features + &(&graph.node_features.row(neighbor).to_owned() * score);
            }

            // Transform features
            let transformed = self.transform_features(&weighted_features, head)?;

            for d in 0..head_dim {
                output[[node, d]] = transformed[d];
            }
        }

        Ok(output)
    }

    /// Compute attention scores
    fn compute_attention_scores(&self, graph: &QuantumGraph, head: usize) -> Result<Array2<f64>> {
        let mut scores = Array2::zeros((graph.num_nodes, graph.num_nodes));

        // Compute pairwise attention scores
        for i in 0..graph.num_nodes {
            for j in 0..graph.num_nodes {
                if i == j || graph.adjacency[[i, j]] > 0.0 {
                    // Quantum attention computation (simplified)
                    let score = self.quantum_attention_score(
                        &graph.node_features.row(i).to_owned(),
                        &graph.node_features.row(j).to_owned(),
                        head,
                    )?;
                    scores[[i, j]] = score;
                }
            }

            // Softmax normalization
            let neighbors = graph.neighbors(i);
            if !neighbors.is_empty() {
                let mut sum_exp = (scores[[i, i]]).exp();
                for &j in &neighbors {
                    sum_exp += scores[[i, j]].exp();
                }

                scores[[i, i]] = scores[[i, i]].exp() / sum_exp;
                for &j in &neighbors {
                    scores[[i, j]] = scores[[i, j]].exp() / sum_exp;
                }
            } else {
                scores[[i, i]] = 1.0;
            }
        }

        Ok(scores)
    }

    /// Compute quantum attention score
    fn quantum_attention_score(
        &self,
        feat_i: &Array1<f64>,
        feat_j: &Array1<f64>,
        head: usize,
    ) -> Result<f64> {
        // Simplified attention score computation
        let dot_product: f64 = feat_i.iter().zip(feat_j.iter()).map(|(a, b)| a * b).sum();

        Ok((dot_product / (self.input_dim as f64).sqrt()).tanh())
    }

    /// Transform features using quantum circuit
    fn transform_features(&self, features: &Array1<f64>, head: usize) -> Result<Array1<f64>> {
        let head_dim = self.output_dim / self.num_heads;
        let mut output = Array1::zeros(head_dim);

        // Apply transformation (simplified)
        for i in 0..head_dim {
            if i < features.len() {
                output[i] = features[i] * (1.0 + 0.1 * (i as f64).sin());
            }
        }

        Ok(output)
    }
}

/// Quantum Message Passing Neural Network
#[derive(Debug)]
pub struct QuantumMPNN {
    /// Message function circuit
    message_circuit: VariationalCircuit,
    /// Update function circuit
    update_circuit: VariationalCircuit,
    /// Readout function circuit
    readout_circuit: VariationalCircuit,
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of message passing steps
    num_steps: usize,
}

impl QuantumMPNN {
    /// Create a new quantum MPNN
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, num_steps: usize) -> Self {
        let num_qubits = (hidden_dim as f64).log2().ceil() as usize;

        Self {
            message_circuit: Self::build_message_circuit(num_qubits),
            update_circuit: Self::build_update_circuit(num_qubits),
            readout_circuit: Self::build_readout_circuit(num_qubits),
            hidden_dim,
            num_steps,
        }
    }

    /// Build message function circuit
    fn build_message_circuit(num_qubits: usize) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(num_qubits * 3); // Source, dest, edge

        // Encode node and edge features
        for q in 0..num_qubits * 3 {
            circuit.add_gate("RY", vec![q], vec![format!("msg_encode_{}", q)]);
        }

        // Interaction layers
        for layer in 0..2 {
            // Source-edge interaction
            for q in 0..num_qubits {
                circuit.add_gate("CZ", vec![q, q + num_qubits * 2], vec![]);
            }

            // Dest-edge interaction
            for q in 0..num_qubits {
                circuit.add_gate("CZ", vec![q + num_qubits, q + num_qubits * 2], vec![]);
            }

            // Parameterized rotations
            for q in 0..num_qubits * 3 {
                circuit.add_gate("RX", vec![q], vec![format!("msg_rx_{}_{}", layer, q)]);
            }
        }

        circuit
    }

    /// Build update function circuit
    fn build_update_circuit(num_qubits: usize) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(num_qubits * 2); // Hidden state + messages

        // Combine hidden state and messages
        for q in 0..num_qubits {
            circuit.add_gate("CNOT", vec![q, q + num_qubits], vec![]);
        }

        // Update layers
        for layer in 0..2 {
            for q in 0..num_qubits * 2 {
                circuit.add_gate("RY", vec![q], vec![format!("upd_ry_{}_{}", layer, q)]);
                circuit.add_gate("RZ", vec![q], vec![format!("upd_rz_{}_{}", layer, q)]);
            }

            // Entangling
            for q in 0..num_qubits * 2 - 1 {
                circuit.add_gate("CX", vec![q, q + 1], vec![]);
            }
        }

        circuit
    }

    /// Build readout function circuit
    fn build_readout_circuit(num_qubits: usize) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(num_qubits);

        // Global pooling layers
        for layer in 0..3 {
            for q in 0..num_qubits {
                circuit.add_gate("RY", vec![q], vec![format!("read_ry_{}_{}", layer, q)]);
            }

            // All-to-all connectivity
            for i in 0..num_qubits {
                for j in i + 1..num_qubits {
                    circuit.add_gate("CZ", vec![i, j], vec![]);
                }
            }
        }

        circuit
    }

    /// Forward pass
    pub fn forward(&self, graph: &QuantumGraph) -> Result<Array1<f64>> {
        // Initialize hidden states
        let mut hidden_states = Array2::zeros((graph.num_nodes, self.hidden_dim));

        // Initialize with node features
        for node in 0..graph.num_nodes {
            for d in 0..self.hidden_dim.min(graph.node_features.ncols()) {
                hidden_states[[node, d]] = graph.node_features[[node, d]];
            }
        }

        // Message passing steps
        for _ in 0..self.num_steps {
            hidden_states = self.message_passing_step(graph, &hidden_states)?;
        }

        // Global readout
        self.readout(graph, &hidden_states)
    }

    /// Single message passing step
    fn message_passing_step(
        &self,
        graph: &QuantumGraph,
        hidden_states: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let mut new_hidden = Array2::zeros((graph.num_nodes, self.hidden_dim));

        for node in 0..graph.num_nodes {
            let neighbors = graph.neighbors(node);
            let mut messages = Array1::zeros(self.hidden_dim);

            // Aggregate messages from neighbors
            for &neighbor in &neighbors {
                let message = self.compute_message(
                    &hidden_states.row(neighbor).to_owned(),
                    &hidden_states.row(node).to_owned(),
                    graph
                        .edge_features
                        .as_ref()
                        .and_then(|ef| ef.get(&(neighbor, node))),
                )?;
                messages = &messages + &message;
            }

            // Update hidden state
            let updated = self.update_node(&hidden_states.row(node).to_owned(), &messages)?;

            new_hidden.row_mut(node).assign(&updated);
        }

        Ok(new_hidden)
    }

    /// Compute message between nodes
    fn compute_message(
        &self,
        source_hidden: &Array1<f64>,
        dest_hidden: &Array1<f64>,
        edge_features: Option<&Array1<f64>>,
    ) -> Result<Array1<f64>> {
        // Simplified message computation
        let mut message = Array1::zeros(self.hidden_dim);

        for i in 0..self.hidden_dim {
            let src_val = if i < source_hidden.len() {
                source_hidden[i]
            } else {
                0.0
            };
            let dst_val = if i < dest_hidden.len() {
                dest_hidden[i]
            } else {
                0.0
            };
            let edge_val = edge_features
                .and_then(|ef| ef.get(i))
                .copied()
                .unwrap_or(1.0);

            message[i] = (src_val + dst_val) * edge_val * 0.5;
        }

        Ok(message)
    }

    /// Update node hidden state
    fn update_node(&self, hidden: &Array1<f64>, messages: &Array1<f64>) -> Result<Array1<f64>> {
        // GRU-like update
        let mut new_hidden = Array1::zeros(self.hidden_dim);

        for i in 0..self.hidden_dim {
            let h = if i < hidden.len() { hidden[i] } else { 0.0 };
            let m = if i < messages.len() { messages[i] } else { 0.0 };

            // Simplified GRU update
            let z = (h + m).tanh(); // Update gate
            let r = 1.0 / (1.0 + (-(h * m)).exp()); // Reset gate (sigmoid)
            let h_tilde = ((r * h) + m).tanh(); // Candidate

            new_hidden[i] = (1.0 - z) * h + z * h_tilde;
        }

        Ok(new_hidden)
    }

    /// Global graph readout
    fn readout(&self, graph: &QuantumGraph, hidden_states: &Array2<f64>) -> Result<Array1<f64>> {
        // Mean pooling
        let mut global_state: Array1<f64> = Array1::zeros(self.hidden_dim);

        for node in 0..graph.num_nodes {
            global_state = &global_state + &hidden_states.row(node).to_owned();
        }
        global_state = global_state / (graph.num_nodes as f64);

        // Apply readout transformation (simplified)
        let mut output = Array1::zeros(self.hidden_dim);
        for i in 0..self.hidden_dim {
            output[i] = global_state[i].tanh();
        }

        Ok(output)
    }
}

/// Quantum Graph Pooling Layer
#[derive(Debug)]
pub struct QuantumGraphPool {
    /// Pooling ratio
    pool_ratio: f64,
    /// Pooling method
    method: PoolingMethod,
    /// Score computation circuit
    score_circuit: VariationalCircuit,
}

#[derive(Debug, Clone)]
pub enum PoolingMethod {
    /// Top-K pooling
    TopK,
    /// Self-attention pooling
    SelfAttention,
    /// Differential pooling
    DiffPool,
}

impl QuantumGraphPool {
    /// Create a new quantum graph pooling layer
    pub fn new(pool_ratio: f64, method: PoolingMethod, feature_dim: usize) -> Self {
        let num_qubits = (feature_dim as f64).log2().ceil() as usize;

        Self {
            pool_ratio,
            method,
            score_circuit: Self::build_score_circuit(num_qubits),
        }
    }

    /// Build score computation circuit
    fn build_score_circuit(num_qubits: usize) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(num_qubits);

        // Score computation layers
        for layer in 0..2 {
            for q in 0..num_qubits {
                circuit.add_gate("RY", vec![q], vec![format!("pool_ry_{}_{}", layer, q)]);
            }

            // Entangling
            for q in 0..num_qubits - 1 {
                circuit.add_gate("CZ", vec![q, q + 1], vec![]);
            }
        }

        // Measurement preparation
        for q in 0..num_qubits {
            circuit.add_gate("RX", vec![q], vec![format!("pool_measure_{}", q)]);
        }

        circuit
    }

    /// Pool graph nodes
    pub fn pool(
        &self,
        graph: &QuantumGraph,
        node_features: &Array2<f64>,
    ) -> Result<(Vec<usize>, Array2<f64>)> {
        match self.method {
            PoolingMethod::TopK => self.topk_pool(graph, node_features),
            PoolingMethod::SelfAttention => self.attention_pool(graph, node_features),
            PoolingMethod::DiffPool => self.diff_pool(graph, node_features),
        }
    }

    /// Top-K pooling
    fn topk_pool(
        &self,
        graph: &QuantumGraph,
        node_features: &Array2<f64>,
    ) -> Result<(Vec<usize>, Array2<f64>)> {
        // Compute node scores
        let mut scores = Vec::new();
        for node in 0..graph.num_nodes {
            let score = self.compute_node_score(&node_features.row(node).to_owned())?;
            scores.push((node, score));
        }

        // Sort by score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select top-k nodes
        let k = ((graph.num_nodes as f64) * self.pool_ratio).ceil() as usize;
        let selected_nodes: Vec<usize> = scores.iter().take(k).map(|(idx, _)| *idx).collect();

        // Extract pooled features
        let mut pooled_features = Array2::zeros((k, node_features.ncols()));
        for (i, &node) in selected_nodes.iter().enumerate() {
            pooled_features.row_mut(i).assign(&node_features.row(node));
        }

        Ok((selected_nodes, pooled_features))
    }

    /// Self-attention pooling
    fn attention_pool(
        &self,
        graph: &QuantumGraph,
        node_features: &Array2<f64>,
    ) -> Result<(Vec<usize>, Array2<f64>)> {
        // Compute attention scores
        let mut attention_scores = Array1::zeros(graph.num_nodes);
        for node in 0..graph.num_nodes {
            attention_scores[node] =
                self.compute_node_score(&node_features.row(node).to_owned())?;
        }

        // Softmax normalization
        let max_score = attention_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Array1<f64> = attention_scores.mapv(|x| (x - max_score).exp());
        let sum_exp = exp_scores.sum();
        let normalized_scores = exp_scores / sum_exp;

        // Sample nodes based on attention
        let k = ((graph.num_nodes as f64) * self.pool_ratio).ceil() as usize;
        let mut selected_nodes = Vec::new();
        let mut remaining_scores = normalized_scores.clone();

        for _ in 0..k {
            let node = self.sample_node(&remaining_scores);
            selected_nodes.push(node);
            remaining_scores[node] = 0.0;
        }

        // Weight features by attention
        let mut pooled_features = Array2::zeros((k, node_features.ncols()));
        for (i, &node) in selected_nodes.iter().enumerate() {
            let weighted_feature = &node_features.row(node).to_owned() * normalized_scores[node];
            pooled_features.row_mut(i).assign(&weighted_feature);
        }

        Ok((selected_nodes, pooled_features))
    }

    /// Differentiable pooling
    fn diff_pool(
        &self,
        graph: &QuantumGraph,
        node_features: &Array2<f64>,
    ) -> Result<(Vec<usize>, Array2<f64>)> {
        // Compute soft cluster assignments
        let k = ((graph.num_nodes as f64) * self.pool_ratio).ceil() as usize;
        let mut assignments = Array2::zeros((graph.num_nodes, k));

        // Initialize with quantum circuit outputs
        for node in 0..graph.num_nodes {
            for cluster in 0..k {
                let score =
                    self.compute_cluster_assignment(&node_features.row(node).to_owned(), cluster)?;
                assignments[[node, cluster]] = score;
            }
        }

        // Normalize assignments (soft clustering)
        for node in 0..graph.num_nodes {
            let row_sum: f64 = assignments.row(node).sum();
            if row_sum > 0.0 {
                for cluster in 0..k {
                    assignments[[node, cluster]] /= row_sum;
                }
            }
        }

        // Compute pooled features
        let pooled_features = assignments.t().dot(node_features);

        // Select representative nodes (hard assignment)
        let mut selected_nodes = Vec::new();
        for cluster in 0..k {
            let mut best_node = 0;
            let mut best_score = 0.0;

            for node in 0..graph.num_nodes {
                if assignments[[node, cluster]] > best_score {
                    best_score = assignments[[node, cluster]];
                    best_node = node;
                }
            }

            selected_nodes.push(best_node);
        }

        Ok((selected_nodes, pooled_features))
    }

    /// Compute node score using quantum circuit
    fn compute_node_score(&self, features: &Array1<f64>) -> Result<f64> {
        // Simplified score computation
        let norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt();
        Ok(norm * (1.0 + 0.1 * fastrand::f64()))
    }

    /// Compute cluster assignment score
    fn compute_cluster_assignment(&self, features: &Array1<f64>, cluster: usize) -> Result<f64> {
        // Simplified cluster assignment
        let base_score = features.iter().sum::<f64>() / features.len() as f64;
        let cluster_bias = (cluster as f64) * 0.1;
        Ok((base_score + cluster_bias).exp() / (1.0 + (base_score + cluster_bias).exp()))
    }

    /// Sample node based on scores
    fn sample_node(&self, scores: &Array1<f64>) -> usize {
        let cumsum: Vec<f64> = scores
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        let r = fastrand::f64() * cumsum.last().unwrap();

        for (i, &cs) in cumsum.iter().enumerate() {
            if r <= cs {
                return i;
            }
        }

        scores.len() - 1
    }
}

/// Complete Quantum GNN model
#[derive(Debug)]
pub struct QuantumGNN {
    /// GNN layers
    layers: Vec<GNNLayer>,
    /// Pooling layers
    pooling: Vec<Option<QuantumGraphPool>>,
    /// Final readout
    readout: ReadoutType,
    /// Output dimension
    output_dim: usize,
}

#[derive(Debug)]
enum GNNLayer {
    GCN(QuantumGCNLayer),
    GAT(QuantumGATLayer),
    MPNN(QuantumMPNN),
}

#[derive(Debug, Clone)]
pub enum ReadoutType {
    Mean,
    Max,
    Sum,
    Attention,
}

impl QuantumGNN {
    /// Create a new quantum GNN
    pub fn new(
        layer_configs: Vec<(String, usize, usize)>, // (type, input_dim, output_dim)
        pooling_configs: Vec<Option<(f64, PoolingMethod)>>,
        readout: ReadoutType,
        output_dim: usize,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        let mut pooling = Vec::new();

        for (layer_type, input_dim, output_dim) in layer_configs {
            let layer = match layer_type.as_str() {
                "gcn" => GNNLayer::GCN(QuantumGCNLayer::new(
                    input_dim,
                    output_dim,
                    ActivationType::ReLU,
                )),
                "gat" => GNNLayer::GAT(QuantumGATLayer::new(
                    input_dim, output_dim, 4,   // num_heads
                    0.1, // dropout
                )),
                "mpnn" => GNNLayer::MPNN(QuantumMPNN::new(
                    input_dim, output_dim, output_dim, 3, // num_steps
                )),
                _ => {
                    return Err(MLError::InvalidConfiguration(format!(
                        "Unknown layer type: {}",
                        layer_type
                    )))
                }
            };
            layers.push(layer);
        }

        for pool_config in pooling_configs {
            let pool_layer = pool_config.map(|(ratio, method)| {
                QuantumGraphPool::new(ratio, method, 64) // feature_dim placeholder
            });
            pooling.push(pool_layer);
        }

        Ok(Self {
            layers,
            pooling,
            readout,
            output_dim,
        })
    }

    /// Forward pass through the GNN
    pub fn forward(&self, graph: &QuantumGraph) -> Result<Array1<f64>> {
        let mut current_graph = graph.clone();
        let mut current_features = graph.node_features.clone();
        let mut selected_nodes: Vec<usize> = (0..graph.num_nodes).collect();

        // Pass through layers with optional pooling
        for (i, layer) in self.layers.iter().enumerate() {
            // Apply GNN layer
            current_features = match layer {
                GNNLayer::GCN(gcn) => gcn.forward(&current_graph)?,
                GNNLayer::GAT(gat) => gat.forward(&current_graph)?,
                GNNLayer::MPNN(mpnn) => {
                    // MPNN returns graph-level features
                    let graph_features = mpnn.forward(&current_graph)?;
                    // Broadcast to all nodes for consistency
                    let mut node_features =
                        Array2::zeros((current_graph.num_nodes, graph_features.len()));
                    for node in 0..current_graph.num_nodes {
                        node_features.row_mut(node).assign(&graph_features);
                    }
                    node_features
                }
            };

            // Apply pooling if configured
            if let Some(Some(pool)) = self.pooling.get(i) {
                let (new_selected, pooled_features) =
                    pool.pool(&current_graph, &current_features)?;

                // Create subgraph with updated features
                current_graph =
                    self.create_subgraph(&current_graph, &new_selected, &pooled_features);
                current_features = pooled_features;
                selected_nodes = new_selected;
            }
        }

        // Global readout
        self.apply_readout(&current_features)
    }

    /// Create subgraph from selected nodes
    fn create_subgraph(
        &self,
        graph: &QuantumGraph,
        selected_nodes: &[usize],
        pooled_features: &Array2<f64>,
    ) -> QuantumGraph {
        let num_nodes = selected_nodes.len();
        let mut new_adjacency = Array2::zeros((num_nodes, num_nodes));

        // Map old indices to new indices
        let index_map: HashMap<usize, usize> = selected_nodes
            .iter()
            .enumerate()
            .map(|(new_idx, &old_idx)| (old_idx, new_idx))
            .collect();

        // Build new adjacency matrix
        for (i, &old_i) in selected_nodes.iter().enumerate() {
            for (j, &old_j) in selected_nodes.iter().enumerate() {
                new_adjacency[[i, j]] = graph.adjacency[[old_i, old_j]];
            }
        }

        // Build edge list
        let mut edges = Vec::new();
        for i in 0..num_nodes {
            for j in i + 1..num_nodes {
                if new_adjacency[[i, j]] > 0.0 {
                    edges.push((i, j));
                }
            }
        }

        // Use the pooled features instead of extracting from old graph
        QuantumGraph::new(num_nodes, edges, pooled_features.clone())
    }

    /// Apply readout operation
    fn apply_readout(&self, node_features: &Array2<f64>) -> Result<Array1<f64>> {
        let readout_features = match self.readout {
            ReadoutType::Mean => node_features
                .mean_axis(scirs2_core::ndarray::Axis(0))
                .unwrap(),
            ReadoutType::Max => {
                let mut max_features = Array1::from_elem(node_features.ncols(), f64::NEG_INFINITY);
                for row in node_features.rows() {
                    for (i, &val) in row.iter().enumerate() {
                        max_features[i] = max_features[i].max(val);
                    }
                }
                max_features
            }
            ReadoutType::Sum => node_features.sum_axis(scirs2_core::ndarray::Axis(0)),
            ReadoutType::Attention => {
                // Compute attention weights
                let mut weights = Array1::zeros(node_features.nrows());
                for (i, row) in node_features.rows().into_iter().enumerate() {
                    weights[i] = row.sum(); // Simple attention
                }

                // Softmax
                let max_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_weights = weights.mapv(|x| (x - max_weight).exp());
                let weights_norm = exp_weights.clone() / exp_weights.sum();

                // Weighted sum
                let mut result = Array1::zeros(node_features.ncols());
                for (i, row) in node_features.rows().into_iter().enumerate() {
                    result = &result + &(&row.to_owned() * weights_norm[i]);
                }
                result
            }
        };

        // Final projection to output dimension
        let mut output = Array1::zeros(self.output_dim);
        for i in 0..self.output_dim {
            if i < readout_features.len() {
                output[i] = readout_features[i];
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_graph() {
        let nodes = 5;
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)];
        let features = Array2::ones((nodes, 4));

        let graph = QuantumGraph::new(nodes, edges, features);

        assert_eq!(graph.num_nodes, 5);
        assert_eq!(graph.degree(0), 2);
        assert_eq!(graph.neighbors(0), vec![1, 4]);
    }

    #[test]
    fn test_quantum_gcn_layer() {
        let graph = QuantumGraph::new(
            3,
            vec![(0, 1), (1, 2)],
            Array2::from_shape_vec(
                (3, 4),
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
            .unwrap(),
        );

        let gcn = QuantumGCNLayer::new(4, 8, ActivationType::ReLU);
        let output = gcn.forward(&graph).unwrap();

        assert_eq!(output.shape(), &[3, 8]);
    }

    #[test]
    fn test_quantum_gat_layer() {
        let graph = QuantumGraph::new(
            4,
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
            Array2::ones((4, 8)),
        );

        let gat = QuantumGATLayer::new(8, 16, 4, 0.1);
        let output = gat.forward(&graph).unwrap();

        assert_eq!(output.shape(), &[4, 16]);
    }

    #[test]
    fn test_quantum_mpnn() {
        let graph = QuantumGraph::new(3, vec![(0, 1), (1, 2)], Array2::zeros((3, 4)));

        let mpnn = QuantumMPNN::new(4, 8, 16, 2);
        let output = mpnn.forward(&graph).unwrap();

        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_graph_pooling() {
        let graph = QuantumGraph::new(
            6,
            vec![(0, 1), (1, 2), (3, 4), (4, 5)],
            Array2::ones((6, 4)),
        );

        let pool = QuantumGraphPool::new(0.5, PoolingMethod::TopK, 4);
        let (selected, pooled) = pool.pool(&graph, &graph.node_features).unwrap();

        assert_eq!(selected.len(), 3);
        assert_eq!(pooled.shape(), &[3, 4]);
    }

    #[test]
    fn test_complete_gnn() {
        let layer_configs = vec![("gcn".to_string(), 4, 8), ("gat".to_string(), 8, 16)];
        let pooling_configs = vec![None, Some((0.5, PoolingMethod::TopK))];

        let gnn = QuantumGNN::new(layer_configs, pooling_configs, ReadoutType::Mean, 10).unwrap();

        let graph = QuantumGraph::new(
            5,
            vec![(0, 1), (1, 2), (2, 3), (3, 4)],
            Array2::ones((5, 4)),
        );

        let output = gnn.forward(&graph).unwrap();
        assert_eq!(output.len(), 10);
    }
}
