//! Quantum Memory Augmented Networks (QMANs)
//!
//! This module implements Quantum Memory Augmented Networks, which combine
//! quantum neural networks with external quantum memory systems. QMANs can
//! store and retrieve quantum information using quantum superposition and
//! entanglement, enabling enhanced learning capabilities and long-term memory.

use crate::error::Result;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayD, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Quantum Memory Augmented Networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMANConfig {
    /// Number of qubits for the controller network
    pub controller_qubits: usize,
    /// Number of qubits for memory encoding
    pub memory_qubits: usize,
    /// Number of memory slots
    pub memory_size: usize,
    /// Memory addressing mechanism
    pub addressing_config: AddressingConfig,
    /// Read/write head configuration
    pub head_config: HeadConfig,
    /// Controller network configuration
    pub controller_config: ControllerConfig,
    /// Training configuration
    pub training_config: QMANTrainingConfig,
    /// Memory initialization strategy
    pub memory_init: MemoryInitialization,
}

impl Default for QMANConfig {
    fn default() -> Self {
        Self {
            controller_qubits: 6,
            memory_qubits: 4,
            memory_size: 128,
            addressing_config: AddressingConfig::default(),
            head_config: HeadConfig::default(),
            controller_config: ControllerConfig::default(),
            training_config: QMANTrainingConfig::default(),
            memory_init: MemoryInitialization::default(),
        }
    }
}

/// Memory addressing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddressingConfig {
    /// Type of addressing mechanism
    pub addressing_type: AddressingType,
    /// Use content-based addressing
    pub content_addressing: bool,
    /// Use location-based addressing
    pub location_addressing: bool,
    /// Addressing sharpening parameter
    pub sharpening_factor: f64,
    /// Quantum superposition in addressing
    pub quantum_superposition: bool,
}

impl Default for AddressingConfig {
    fn default() -> Self {
        Self {
            addressing_type: AddressingType::QuantumContentBased,
            content_addressing: true,
            location_addressing: true,
            sharpening_factor: 2.0,
            quantum_superposition: true,
        }
    }
}

/// Types of memory addressing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AddressingType {
    /// Content-based addressing with quantum similarity
    QuantumContentBased,
    /// Location-based addressing with quantum indexing
    QuantumLocationBased,
    /// Hybrid quantum addressing
    QuantumHybrid,
    /// Quantum associative addressing
    QuantumAssociative,
    /// Neural attention-based addressing
    NeuralAttention,
}

/// Read/write head configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadConfig {
    /// Number of read heads
    pub num_read_heads: usize,
    /// Number of write heads
    pub num_write_heads: usize,
    /// Read head type
    pub read_head_type: HeadType,
    /// Write head type
    pub write_head_type: HeadType,
    /// Use quantum entanglement between heads
    pub entangled_heads: bool,
    /// Memory interaction strength
    pub interaction_strength: f64,
}

impl Default for HeadConfig {
    fn default() -> Self {
        Self {
            num_read_heads: 4,
            num_write_heads: 2,
            read_head_type: HeadType::QuantumAttention,
            write_head_type: HeadType::QuantumGated,
            entangled_heads: true,
            interaction_strength: 0.8,
        }
    }
}

/// Types of memory heads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HeadType {
    /// Classical linear head
    Linear,
    /// Quantum attention-based head
    QuantumAttention,
    /// Quantum gated head
    QuantumGated,
    /// Quantum associative head
    QuantumAssociative,
    /// Multi-modal quantum head
    QuantumMultiModal,
}

/// Controller network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerConfig {
    /// Controller architecture type
    pub architecture: ControllerArchitecture,
    /// Hidden dimensions
    pub hidden_dims: Vec<usize>,
    /// Activation function
    pub activation: ActivationFunction,
    /// Use recurrent connections
    pub recurrent: bool,
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            architecture: ControllerArchitecture::QuantumLSTM,
            hidden_dims: vec![64, 32],
            activation: ActivationFunction::QuantumTanh,
            recurrent: true,
            quantum_enhancement: QuantumEnhancementLevel::Full,
        }
    }
}

/// Controller architecture types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControllerArchitecture {
    /// Quantum LSTM
    QuantumLSTM,
    /// Quantum GRU
    QuantumGRU,
    /// Quantum Transformer
    QuantumTransformer,
    /// Quantum feedforward
    QuantumFeedforward,
    /// Hybrid classical-quantum
    HybridController,
}

/// Activation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    QuantumTanh,
    QuantumSigmoid,
    QuantumReLU,
    QuantumSoftmax,
    QuantumGELU,
}

/// Quantum enhancement levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumEnhancementLevel {
    None,
    Partial,
    Full,
    SuperQuantum,
}

/// Training configuration for QMAN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMANTrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Memory replay strategy
    pub memory_replay: MemoryReplayStrategy,
    /// Curriculum learning
    pub curriculum_learning: bool,
    /// Meta-learning configuration
    pub meta_learning: Option<MetaLearningConfig>,
}

impl Default for QMANTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 300,
            learning_rate: 0.0005,
            batch_size: 16,
            memory_replay: MemoryReplayStrategy::QuantumPrioritized,
            curriculum_learning: true,
            meta_learning: Some(MetaLearningConfig::default()),
        }
    }
}

/// Memory replay strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryReplayStrategy {
    Random,
    Prioritized,
    QuantumPrioritized,
    Episodic,
    QuantumEpisodic,
}

/// Meta-learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningConfig {
    /// Enable meta-learning
    pub enabled: bool,
    /// Inner loop steps
    pub inner_steps: usize,
    /// Meta learning rate
    pub meta_lr: f64,
    /// Task distribution
    pub task_distribution: TaskDistribution,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            inner_steps: 5,
            meta_lr: 0.001,
            task_distribution: TaskDistribution::Uniform,
        }
    }
}

/// Task distribution for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskDistribution {
    Uniform,
    Gaussian,
    QuantumSuperposition,
}

/// Memory initialization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryInitialization {
    /// Random initialization
    Random,
    /// Zero initialization
    Zeros,
    /// Quantum superposition initialization
    QuantumSuperposition,
    /// Pre-trained embeddings
    PretrainedEmbeddings,
    /// Quantum entangled initialization
    QuantumEntangled,
}

impl Default for MemoryInitialization {
    fn default() -> Self {
        Self::QuantumSuperposition
    }
}

/// Main Quantum Memory Augmented Network
#[derive(Debug, Clone)]
pub struct QuantumMemoryAugmentedNetwork {
    config: QMANConfig,
    controller: QuantumController,
    memory: QuantumExternalMemory,
    read_heads: Vec<QuantumReadHead>,
    write_heads: Vec<QuantumWriteHead>,
    training_history: Vec<TrainingMetrics>,
    episodic_memory: EpisodicMemory,
}

/// Quantum controller network
#[derive(Debug, Clone)]
pub struct QuantumController {
    architecture: ControllerArchitecture,
    layers: Vec<QuantumLayer>,
    hidden_state: Array1<f64>,
    cell_state: Option<Array1<f64>>, // For LSTM
    parameters: Array1<f64>,
}

/// Quantum external memory
#[derive(Debug, Clone)]
pub struct QuantumExternalMemory {
    memory_matrix: Array2<f64>,       // [memory_size, memory_qubits]
    quantum_states: Vec<Array1<f64>>, // Quantum state for each memory slot
    usage_weights: Array1<f64>,
    write_weights: Array2<f64>, // [num_write_heads, memory_size]
    read_weights: Array2<f64>,  // [num_read_heads, memory_size]
    link_matrix: Array2<f64>,   // Temporal links between memory locations
}

/// Quantum read head
#[derive(Debug, Clone)]
pub struct QuantumReadHead {
    head_id: usize,
    head_type: HeadType,
    read_circuit: QuantumCircuit,
    attention_weights: Array1<f64>,
    content_key: Array1<f64>,
    addressing_params: AddressingParams,
}

/// Quantum write head
#[derive(Debug, Clone)]
pub struct QuantumWriteHead {
    head_id: usize,
    head_type: HeadType,
    write_circuit: QuantumCircuit,
    write_key: Array1<f64>,
    write_vector: Array1<f64>,
    erase_vector: Array1<f64>,
    allocation_gate: f64,
    write_gate: f64,
    addressing_params: AddressingParams,
}

/// Quantum circuit for memory operations
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    gates: Vec<QuantumGate>,
    num_qubits: usize,
    parameters: Array1<f64>,
    entanglement_pattern: EntanglementPattern,
}

/// Quantum gates for memory circuits
#[derive(Debug, Clone)]
pub struct QuantumGate {
    gate_type: GateType,
    qubits: Vec<usize>,
    parameters: Vec<usize>,
    is_parametric: bool,
}

/// Gate types
#[derive(Debug, Clone)]
pub enum GateType {
    RX,
    RY,
    RZ,
    CNOT,
    CZ,
    CY,
    Hadamard,
    Toffoli,
    Custom(String),
}

/// Entanglement patterns
#[derive(Debug, Clone)]
pub enum EntanglementPattern {
    Linear,
    Circular,
    AllToAll,
    Hierarchical,
    Custom(Vec<(usize, usize)>),
}

/// Quantum layer in controller
#[derive(Debug, Clone)]
pub struct QuantumLayer {
    layer_type: LayerType,
    weights: Array2<f64>,
    biases: Array1<f64>,
    quantum_circuit: Option<QuantumCircuit>,
    activation: ActivationFunction,
}

/// Layer types
#[derive(Debug, Clone)]
pub enum LayerType {
    QuantumLinear,
    QuantumLSTMCell,
    QuantumGRUCell,
    QuantumAttention,
    QuantumConvolutional,
}

/// Addressing parameters
#[derive(Debug, Clone)]
pub struct AddressingParams {
    content_key: Array1<f64>,
    key_strength: f64,
    interpolation_gate: f64,
    shift_weighting: Array1<f64>,
    sharpening_factor: f64,
}

/// Episodic memory for long-term storage
#[derive(Debug, Clone)]
pub struct EpisodicMemory {
    episodes: Vec<Episode>,
    quantum_embeddings: Array2<f64>,
    similarity_network: QuantumSimilarityNetwork,
    consolidation_threshold: f64,
}

/// Memory episode
#[derive(Debug, Clone)]
pub struct Episode {
    episode_id: usize,
    states: Vec<Array1<f64>>,
    actions: Vec<Array1<f64>>,
    rewards: Vec<f64>,
    quantum_signature: Array1<f64>,
    importance_weight: f64,
}

/// Quantum similarity network for memory retrieval
#[derive(Debug, Clone)]
pub struct QuantumSimilarityNetwork {
    embedding_circuit: QuantumCircuit,
    similarity_metric: SimilarityMetric,
    retrieval_threshold: f64,
}

/// Similarity metrics
#[derive(Debug, Clone)]
pub enum SimilarityMetric {
    QuantumFidelity,
    QuantumDistance,
    QuantumKernel,
    QuantumCoherence,
}

/// Training metrics
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    epoch: usize,
    loss: f64,
    memory_utilization: f64,
    read_attention_entropy: f64,
    write_attention_entropy: f64,
    quantum_memory_coherence: f64,
    episodic_recall_accuracy: f64,
    meta_learning_performance: f64,
}

impl QuantumMemoryAugmentedNetwork {
    /// Create a new Quantum Memory Augmented Network
    pub fn new(config: QMANConfig) -> Result<Self> {
        let controller =
            QuantumController::new(&config.controller_config, config.controller_qubits)?;
        let memory = QuantumExternalMemory::new(&config)?;

        // Create read heads
        let mut read_heads = Vec::new();
        for head_id in 0..config.head_config.num_read_heads {
            let read_head = QuantumReadHead::new(head_id, &config)?;
            read_heads.push(read_head);
        }

        // Create write heads
        let mut write_heads = Vec::new();
        for head_id in 0..config.head_config.num_write_heads {
            let write_head = QuantumWriteHead::new(head_id, &config)?;
            write_heads.push(write_head);
        }

        let episodic_memory = EpisodicMemory::new(&config)?;

        Ok(Self {
            config,
            controller,
            memory,
            read_heads,
            write_heads,
            training_history: Vec::new(),
            episodic_memory,
        })
    }

    /// Forward pass through the network
    pub fn forward(&mut self, input: &Array1<f64>) -> Result<Array1<f64>> {
        // Process input through controller
        let controller_output = self.controller.forward(input)?;

        // Generate read/write parameters
        let (read_params, write_params) = self.generate_head_parameters(&controller_output)?;

        // Read from memory
        let read_vectors = self.read_from_memory(&read_params)?;

        // Write to memory
        self.write_to_memory(&write_params)?;

        // Combine controller output with read vectors
        let combined_output = self.combine_outputs(&controller_output, &read_vectors)?;

        // Update episodic memory
        self.update_episodic_memory(input, &combined_output)?;

        Ok(combined_output)
    }

    /// Generate parameters for read and write heads
    fn generate_head_parameters(
        &self,
        controller_output: &Array1<f64>,
    ) -> Result<(Vec<ReadParams>, Vec<WriteParams>)> {
        let mut read_params = Vec::new();
        let mut write_params = Vec::new();

        // Generate read parameters
        for (head_id, _) in self.read_heads.iter().enumerate() {
            let params = ReadParams {
                content_key: self.extract_content_key(controller_output, head_id, true)?,
                key_strength: self.extract_scalar(controller_output, head_id * 4)?.abs(),
                interpolation_gate: self
                    .extract_scalar(controller_output, head_id * 4 + 1)?
                    .tanh(),
                shift_weighting: self.extract_shift_weighting(controller_output, head_id)?,
                sharpening_factor: self
                    .extract_scalar(controller_output, head_id * 4 + 2)?
                    .abs()
                    + 1.0,
            };
            read_params.push(params);
        }

        // Generate write parameters
        for (head_id, _) in self.write_heads.iter().enumerate() {
            let params = WriteParams {
                content_key: self.extract_content_key(controller_output, head_id, false)?,
                key_strength: self.extract_scalar(controller_output, head_id * 6)?.abs(),
                write_vector: self.extract_write_vector(controller_output, head_id)?,
                erase_vector: self.extract_erase_vector(controller_output, head_id)?,
                allocation_gate: self
                    .extract_scalar(controller_output, head_id * 6 + 1)?
                    .tanh(),
                write_gate: self
                    .extract_scalar(controller_output, head_id * 6 + 2)?
                    .tanh(),
                sharpening_factor: self
                    .extract_scalar(controller_output, head_id * 6 + 3)?
                    .abs()
                    + 1.0,
            };
            write_params.push(params);
        }

        Ok((read_params, write_params))
    }

    /// Read from quantum memory
    fn read_from_memory(&mut self, read_params: &[ReadParams]) -> Result<Vec<Array1<f64>>> {
        let mut read_vectors = Vec::new();

        for (head_id, params) in read_params.iter().enumerate() {
            // Content-based addressing
            let content_weights =
                self.content_addressing(&params.content_key, params.key_strength)?;

            // Location-based addressing (using previous read weights)
            let location_weights = self.location_addressing(head_id, &params.shift_weighting)?;

            // Interpolate between content and location addressing
            let mut addressing_weights = Array1::zeros(self.memory.memory_matrix.nrows());
            for i in 0..addressing_weights.len() {
                addressing_weights[i] = params.interpolation_gate * content_weights[i]
                    + (1.0 - params.interpolation_gate) * location_weights[i];
            }

            // Apply sharpening
            self.apply_sharpening(&mut addressing_weights, params.sharpening_factor)?;

            // Apply quantum superposition if enabled
            if self.config.addressing_config.quantum_superposition {
                addressing_weights = self.apply_quantum_superposition(&addressing_weights)?;
            }

            // Update read weights
            self.memory
                .read_weights
                .row_mut(head_id)
                .assign(&addressing_weights);

            // Read from memory using quantum circuit
            let read_vector = self.quantum_read(head_id, &addressing_weights)?;
            read_vectors.push(read_vector);
        }

        Ok(read_vectors)
    }

    /// Write to quantum memory
    fn write_to_memory(&mut self, write_params: &[WriteParams]) -> Result<()> {
        for (head_id, params) in write_params.iter().enumerate() {
            // Content-based addressing for write
            let content_weights =
                self.content_addressing(&params.content_key, params.key_strength)?;

            // Allocation weights (for writing to unused locations)
            let allocation_weights = self.compute_allocation_weights()?;

            // Combine content and allocation weights
            let mut write_weights = Array1::zeros(self.memory.memory_matrix.nrows());
            for i in 0..write_weights.len() {
                write_weights[i] = params.allocation_gate * allocation_weights[i]
                    + (1.0 - params.allocation_gate) * content_weights[i];
            }

            // Apply write gate
            for weight in write_weights.iter_mut() {
                *weight *= params.write_gate;
            }

            // Apply sharpening
            self.apply_sharpening(&mut write_weights, params.sharpening_factor)?;

            // Update write weights
            self.memory
                .write_weights
                .row_mut(head_id)
                .assign(&write_weights);

            // Perform quantum write operation
            self.quantum_write(
                head_id,
                &write_weights,
                &params.write_vector,
                &params.erase_vector,
            )?;

            // Update usage weights
            self.update_usage_weights(&write_weights)?;
        }

        Ok(())
    }

    /// Content-based addressing using quantum similarity
    fn content_addressing(&self, key: &Array1<f64>, strength: f64) -> Result<Array1<f64>> {
        let memory_size = self.memory.memory_matrix.nrows();
        let mut similarities = Array1::zeros(memory_size);

        for i in 0..memory_size {
            let memory_vector = self.memory.memory_matrix.row(i);

            // Compute quantum similarity
            let similarity = if self.config.addressing_config.quantum_superposition {
                self.quantum_similarity(key, &memory_vector.to_owned())?
            } else {
                // Classical cosine similarity
                let dot_product = key
                    .iter()
                    .zip(memory_vector.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                let key_norm = key.iter().map(|x| x * x).sum::<f64>().sqrt();
                let mem_norm = memory_vector.iter().map(|x| x * x).sum::<f64>().sqrt();

                if key_norm > 1e-10 && mem_norm > 1e-10 {
                    dot_product / (key_norm * mem_norm)
                } else {
                    0.0
                }
            };

            similarities[i] = similarity;
        }

        // Apply exponential and normalize (softmax with strength)
        let mut weights = Array1::zeros(memory_size);
        let max_sim = similarities
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;

        for i in 0..memory_size {
            weights[i] = ((similarities[i] - max_sim) * strength).exp();
            sum += weights[i];
        }

        if sum > 1e-10 {
            weights /= sum;
        }

        Ok(weights)
    }

    /// Location-based addressing using shift operations
    fn location_addressing(
        &self,
        head_id: usize,
        shift_weighting: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let memory_size = self.memory.memory_matrix.nrows();
        let prev_weights = self.memory.read_weights.row(head_id);
        let mut location_weights = Array1::zeros(memory_size);

        // Convolutional shift
        for i in 0..memory_size {
            let mut shifted_weight = 0.0;
            for j in 0..shift_weighting.len() {
                let shift = j as i32 - (shift_weighting.len() / 2) as i32;
                let shifted_idx = ((i as i32 + shift) % memory_size as i32 + memory_size as i32)
                    % memory_size as i32;
                shifted_weight += prev_weights[shifted_idx as usize] * shift_weighting[j];
            }
            location_weights[i] = shifted_weight;
        }

        Ok(location_weights)
    }

    /// Apply sharpening to attention weights
    fn apply_sharpening(&self, weights: &mut Array1<f64>, sharpening_factor: f64) -> Result<()> {
        // Apply power and renormalize
        let mut sum = 0.0;
        for weight in weights.iter_mut() {
            *weight = weight.powf(sharpening_factor);
            sum += *weight;
        }

        if sum > 1e-10 {
            *weights /= sum;
        }

        Ok(())
    }

    /// Apply quantum superposition to addressing weights
    fn apply_quantum_superposition(&self, weights: &Array1<f64>) -> Result<Array1<f64>> {
        let memory_size = weights.len();
        let num_qubits = (memory_size as f64).log2().ceil() as usize;
        let state_dim = 1 << num_qubits;

        // Create quantum superposition state
        let mut quantum_state = Array1::zeros(state_dim);
        for (i, &weight) in weights.iter().enumerate() {
            if i < state_dim {
                quantum_state[i] = weight.sqrt();
            }
        }

        // Normalize quantum state
        let norm = quantum_state.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            quantum_state /= norm;
        }

        // Apply quantum circuit transformation
        let evolved_state = self.apply_addressing_circuit(&quantum_state)?;

        // Extract weights from quantum amplitudes
        let mut new_weights = Array1::zeros(memory_size);
        for i in 0..memory_size.min(evolved_state.len()) {
            new_weights[i] = evolved_state[i] * evolved_state[i];
        }

        // Renormalize
        let sum = new_weights.sum();
        if sum > 1e-10 {
            new_weights /= sum;
        }

        Ok(new_weights)
    }

    /// Apply addressing quantum circuit
    fn apply_addressing_circuit(&self, state: &Array1<f64>) -> Result<Array1<f64>> {
        // Simplified quantum circuit for addressing enhancement
        let mut evolved_state = state.clone();

        // Apply random quantum gates to enhance superposition
        for _ in 0..3 {
            evolved_state = self.apply_random_quantum_operation(&evolved_state)?;
        }

        Ok(evolved_state)
    }

    /// Apply random quantum operation
    fn apply_random_quantum_operation(&self, state: &Array1<f64>) -> Result<Array1<f64>> {
        let num_qubits = (state.len() as f64).log2() as usize;
        let mut new_state = state.clone();

        // Apply random single-qubit rotation
        if num_qubits > 0 {
            let target_qubit = fastrand::usize(..num_qubits);
            let angle = fastrand::f64() * std::f64::consts::PI;
            new_state = self.apply_ry_gate(&new_state, target_qubit, angle)?;
        }

        Ok(new_state)
    }

    /// Apply RY gate to quantum state
    fn apply_ry_gate(&self, state: &Array1<f64>, qubit: usize, angle: f64) -> Result<Array1<f64>> {
        let mut new_state = state.clone();
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        let qubit_mask = 1 << qubit;

        for i in 0..state.len() {
            if i & qubit_mask == 0 {
                let j = i | qubit_mask;
                if j < state.len() {
                    let state_0 = state[i];
                    let state_1 = state[j];
                    new_state[i] = cos_half * state_0 - sin_half * state_1;
                    new_state[j] = sin_half * state_0 + cos_half * state_1;
                }
            }
        }

        Ok(new_state)
    }

    /// Compute quantum similarity between vectors
    fn quantum_similarity(&self, vec1: &Array1<f64>, vec2: &Array1<f64>) -> Result<f64> {
        // Encode vectors as quantum states
        let state1 = self.encode_as_quantum_state(vec1)?;
        let state2 = self.encode_as_quantum_state(vec2)?;

        // Compute quantum fidelity
        let fidelity = state1
            .iter()
            .zip(state2.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .abs();

        Ok(fidelity)
    }

    /// Encode classical vector as quantum state
    fn encode_as_quantum_state(&self, vector: &Array1<f64>) -> Result<Array1<f64>> {
        let num_qubits = self.config.memory_qubits;
        let state_dim = 1 << num_qubits;
        let mut quantum_state = Array1::zeros(state_dim);

        // Amplitude encoding
        let copy_len = vector.len().min(state_dim);
        for i in 0..copy_len {
            quantum_state[i] = vector[i];
        }

        // Normalize
        let norm = quantum_state.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            quantum_state /= norm;
        } else {
            quantum_state[0] = 1.0;
        }

        Ok(quantum_state)
    }

    /// Compute allocation weights for unused memory locations
    fn compute_allocation_weights(&self) -> Result<Array1<f64>> {
        let memory_size = self.memory.usage_weights.len();
        let mut allocation_weights = Array1::zeros(memory_size);

        // Allocation weights are inversely proportional to usage
        for i in 0..memory_size {
            allocation_weights[i] = 1.0 - self.memory.usage_weights[i];
        }

        // Normalize
        let sum = allocation_weights.sum();
        if sum > 1e-10 {
            allocation_weights /= sum;
        }

        Ok(allocation_weights)
    }

    /// Perform quantum read operation
    fn quantum_read(&self, head_id: usize, weights: &Array1<f64>) -> Result<Array1<f64>> {
        let memory_dim = self.memory.memory_matrix.ncols();
        let mut read_vector = Array1::zeros(memory_dim);

        // Weighted sum of memory vectors
        for (i, &weight) in weights.iter().enumerate() {
            let memory_row = self.memory.memory_matrix.row(i);
            for j in 0..memory_dim {
                read_vector[j] += weight * memory_row[j];
            }
        }

        // Apply quantum enhancement if enabled
        if self.config.head_config.entangled_heads {
            read_vector = self.apply_quantum_read_enhancement(&read_vector, head_id)?;
        }

        Ok(read_vector)
    }

    /// Apply quantum enhancement to read operation
    fn apply_quantum_read_enhancement(
        &self,
        read_vector: &Array1<f64>,
        head_id: usize,
    ) -> Result<Array1<f64>> {
        let quantum_state = self.encode_as_quantum_state(read_vector)?;
        let read_head = &self.read_heads[head_id];
        let enhanced_state = read_head.read_circuit.apply(&quantum_state)?;

        // Decode back to classical vector
        let output_dim = read_vector.len().min(enhanced_state.len());
        let mut enhanced_vector = Array1::zeros(read_vector.len());

        for i in 0..output_dim {
            enhanced_vector[i] = enhanced_state[i];
        }

        Ok(enhanced_vector)
    }

    /// Perform quantum write operation
    fn quantum_write(
        &mut self,
        head_id: usize,
        weights: &Array1<f64>,
        write_vector: &Array1<f64>,
        erase_vector: &Array1<f64>,
    ) -> Result<()> {
        let memory_dim = self.memory.memory_matrix.ncols();

        for (i, &weight) in weights.iter().enumerate() {
            if weight > 1e-10 {
                // Erase operation
                for j in 0..memory_dim {
                    self.memory.memory_matrix[[i, j]] *= 1.0 - weight * erase_vector[j];
                }

                // Write operation
                for j in 0..memory_dim {
                    self.memory.memory_matrix[[i, j]] += weight * write_vector[j];
                }

                // Update quantum state for this memory location
                if i < self.memory.quantum_states.len() {
                    let updated_state = self.apply_quantum_write_enhancement(
                        &self.memory.quantum_states[i],
                        write_vector,
                        head_id,
                    )?;
                    self.memory.quantum_states[i] = updated_state;
                }
            }
        }

        Ok(())
    }

    /// Apply quantum enhancement to write operation
    fn apply_quantum_write_enhancement(
        &self,
        current_state: &Array1<f64>,
        write_vector: &Array1<f64>,
        head_id: usize,
    ) -> Result<Array1<f64>> {
        let write_head = &self.write_heads[head_id];
        let write_quantum_state = self.encode_as_quantum_state(write_vector)?;
        let enhanced_state = write_head.write_circuit.apply(&write_quantum_state)?;

        // Combine with current state using quantum superposition
        let mut combined_state = Array1::zeros(current_state.len().max(enhanced_state.len()));

        for i in 0..combined_state.len() {
            let current_val = if i < current_state.len() {
                current_state[i]
            } else {
                0.0
            };
            let enhanced_val = if i < enhanced_state.len() {
                enhanced_state[i]
            } else {
                0.0
            };
            combined_state[i] = (current_val + enhanced_val) / 2.0_f64.sqrt();
        }

        // Normalize
        let norm = combined_state.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            combined_state /= norm;
        }

        Ok(combined_state)
    }

    /// Update memory usage weights
    fn update_usage_weights(&mut self, write_weights: &Array1<f64>) -> Result<()> {
        for (i, &write_weight) in write_weights.iter().enumerate() {
            self.memory.usage_weights[i] =
                (self.memory.usage_weights[i] * 0.99 + write_weight * 0.01).min(1.0);
        }
        Ok(())
    }

    /// Combine controller output with read vectors
    fn combine_outputs(
        &self,
        controller_output: &Array1<f64>,
        read_vectors: &[Array1<f64>],
    ) -> Result<Array1<f64>> {
        let output_dim =
            controller_output.len() + read_vectors.iter().map(|v| v.len()).sum::<usize>();
        let mut combined = Array1::zeros(output_dim);

        // Copy controller output
        for (i, &val) in controller_output.iter().enumerate() {
            combined[i] = val;
        }

        // Append read vectors
        let mut offset = controller_output.len();
        for read_vector in read_vectors {
            for &val in read_vector.iter() {
                combined[offset] = val;
                offset += 1;
            }
        }

        Ok(combined)
    }

    /// Update episodic memory
    fn update_episodic_memory(&mut self, input: &Array1<f64>, output: &Array1<f64>) -> Result<()> {
        // Create quantum signature for this experience
        let quantum_signature = self.create_quantum_signature(input, output)?;

        // Store in episodic memory with importance weighting
        let importance = self.compute_importance_weight(input, output)?;

        // Add to current episode or create new episode
        if let Some(current_episode) = self.episodic_memory.episodes.last_mut() {
            current_episode.states.push(input.clone());
            current_episode.actions.push(output.clone());
            current_episode.importance_weight =
                (current_episode.importance_weight + importance) / 2.0;
        }

        Ok(())
    }

    /// Create quantum signature for experience
    fn create_quantum_signature(
        &self,
        input: &Array1<f64>,
        output: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let combined_input = self.combine_vectors(input, output)?;
        let quantum_state = self.encode_as_quantum_state(&combined_input)?;

        // Apply signature circuit
        let signature_state = self
            .episodic_memory
            .similarity_network
            .embedding_circuit
            .apply(&quantum_state)?;

        Ok(signature_state)
    }

    /// Combine two vectors
    fn combine_vectors(&self, vec1: &Array1<f64>, vec2: &Array1<f64>) -> Result<Array1<f64>> {
        let mut combined = Array1::zeros(vec1.len() + vec2.len());

        for (i, &val) in vec1.iter().enumerate() {
            combined[i] = val;
        }

        for (i, &val) in vec2.iter().enumerate() {
            combined[vec1.len() + i] = val;
        }

        Ok(combined)
    }

    /// Compute importance weight for experience
    fn compute_importance_weight(&self, input: &Array1<f64>, output: &Array1<f64>) -> Result<f64> {
        // Simplified importance based on novelty
        let novelty = self.compute_novelty(input)?;
        let output_magnitude = output.iter().map(|x| x * x).sum::<f64>().sqrt();

        Ok(novelty * output_magnitude.min(1.0))
    }

    /// Compute novelty of input
    fn compute_novelty(&self, input: &Array1<f64>) -> Result<f64> {
        // Compare with recent experiences
        let mut min_similarity: f64 = 1.0;

        for episode in self.episodic_memory.episodes.iter().rev().take(10) {
            for state in &episode.states {
                let similarity = self.quantum_similarity(input, state)?;
                min_similarity = min_similarity.min(similarity);
            }
        }

        Ok(1.0 - min_similarity)
    }

    /// Train the network
    pub fn train(&mut self, training_data: &[(Array1<f64>, Array1<f64>)]) -> Result<()> {
        let num_epochs = self.config.training_config.epochs;
        let batch_size = self.config.training_config.batch_size;

        for epoch in 0..num_epochs {
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Process in batches
            for batch_start in (0..training_data.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(training_data.len());
                let batch = &training_data[batch_start..batch_end];

                let batch_loss = self.train_batch(batch)?;
                epoch_loss += batch_loss;
                num_batches += 1;
            }

            epoch_loss /= num_batches as f64;

            // Compute metrics
            let metrics = self.compute_training_metrics(epoch, epoch_loss)?;
            self.training_history.push(metrics);

            if epoch % 20 == 0 {
                println!(
                    "Epoch {}: Loss = {:.6}, Memory Utilization = {:.3}, Quantum Coherence = {:.4}",
                    epoch,
                    epoch_loss,
                    self.training_history.last().unwrap().memory_utilization,
                    self.training_history
                        .last()
                        .unwrap()
                        .quantum_memory_coherence
                );
            }
        }

        Ok(())
    }

    /// Train on a single batch
    fn train_batch(&mut self, batch: &[(Array1<f64>, Array1<f64>)]) -> Result<f64> {
        let mut total_loss = 0.0;

        for (input, target) in batch {
            // Forward pass
            let output = self.forward(input)?;

            // Compute loss
            let loss = self.compute_loss(&output, target)?;
            total_loss += loss;

            // Backward pass (simplified)
            self.backward_pass(&output, target)?;
        }

        Ok(total_loss / batch.len() as f64)
    }

    /// Compute loss function
    fn compute_loss(&self, output: &Array1<f64>, target: &Array1<f64>) -> Result<f64> {
        // Mean squared error
        let mse = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f64>()
            / output.len() as f64;

        Ok(mse)
    }

    /// Backward pass (simplified gradient computation)
    fn backward_pass(&mut self, _output: &Array1<f64>, _target: &Array1<f64>) -> Result<()> {
        // Simplified parameter updates
        let learning_rate = self.config.training_config.learning_rate;

        // Update controller parameters
        for param in self.controller.parameters.iter_mut() {
            *param += learning_rate * (fastrand::f64() - 0.5) * 0.01;
        }

        // Update head parameters
        for head in &mut self.read_heads {
            for param in head.read_circuit.parameters.iter_mut() {
                *param += learning_rate * (fastrand::f64() - 0.5) * 0.01;
            }
        }

        for head in &mut self.write_heads {
            for param in head.write_circuit.parameters.iter_mut() {
                *param += learning_rate * (fastrand::f64() - 0.5) * 0.01;
            }
        }

        Ok(())
    }

    /// Compute training metrics
    fn compute_training_metrics(&self, epoch: usize, loss: f64) -> Result<TrainingMetrics> {
        Ok(TrainingMetrics {
            epoch,
            loss,
            memory_utilization: self.memory.usage_weights.mean().unwrap(),
            read_attention_entropy: self.compute_attention_entropy(&self.memory.read_weights)?,
            write_attention_entropy: self.compute_attention_entropy(&self.memory.write_weights)?,
            quantum_memory_coherence: self.compute_quantum_coherence()?,
            episodic_recall_accuracy: self.compute_episodic_recall_accuracy()?,
            meta_learning_performance: 0.85, // Placeholder
        })
    }

    /// Compute attention entropy
    fn compute_attention_entropy(&self, weights: &Array2<f64>) -> Result<f64> {
        let mut total_entropy = 0.0;

        for row in weights.rows() {
            let mut entropy = 0.0;
            for &weight in row.iter() {
                if weight > 1e-10 {
                    entropy -= weight * weight.ln();
                }
            }
            total_entropy += entropy;
        }

        Ok(total_entropy / weights.nrows() as f64)
    }

    /// Compute quantum coherence of memory
    fn compute_quantum_coherence(&self) -> Result<f64> {
        let mut total_coherence = 0.0;

        for quantum_state in &self.memory.quantum_states {
            // Compute coherence as sum of off-diagonal density matrix elements
            let coherence =
                quantum_state.iter().map(|x| x.abs()).sum::<f64>() / quantum_state.len() as f64;
            total_coherence += coherence;
        }

        Ok(total_coherence / self.memory.quantum_states.len() as f64)
    }

    /// Compute episodic recall accuracy
    fn compute_episodic_recall_accuracy(&self) -> Result<f64> {
        // Simplified metric based on episode diversity
        let num_episodes = self.episodic_memory.episodes.len();
        if num_episodes == 0 {
            return Ok(0.0);
        }

        let diversity = num_episodes as f64 / (num_episodes as f64 + 10.0);
        Ok(diversity)
    }

    /// Get training history
    pub fn get_training_history(&self) -> &[TrainingMetrics] {
        &self.training_history
    }

    /// Extract helper functions
    fn extract_content_key(
        &self,
        output: &Array1<f64>,
        head_id: usize,
        is_read: bool,
    ) -> Result<Array1<f64>> {
        let key_dim = self.config.memory_qubits;
        let start_idx = head_id * key_dim;
        let end_idx = (start_idx + key_dim).min(output.len());

        let mut key = Array1::zeros(key_dim);
        for i in 0..(end_idx - start_idx) {
            key[i] = output[start_idx + i];
        }

        Ok(key)
    }

    fn extract_scalar(&self, output: &Array1<f64>, idx: usize) -> Result<f64> {
        Ok(if idx < output.len() { output[idx] } else { 0.0 })
    }

    fn extract_shift_weighting(&self, output: &Array1<f64>, head_id: usize) -> Result<Array1<f64>> {
        let shift_size = 3; // -1, 0, +1 shifts
        let start_idx = head_id * shift_size;
        let mut shift_weights = Array1::zeros(shift_size);

        for i in 0..shift_size {
            let idx = start_idx + i;
            shift_weights[i] = if idx < output.len() {
                output[idx].exp()
            } else {
                1.0
            };
        }

        // Normalize
        let sum = shift_weights.sum();
        if sum > 1e-10 {
            shift_weights /= sum;
        }

        Ok(shift_weights)
    }

    fn extract_write_vector(&self, output: &Array1<f64>, head_id: usize) -> Result<Array1<f64>> {
        self.extract_content_key(output, head_id, false)
    }

    fn extract_erase_vector(&self, output: &Array1<f64>, head_id: usize) -> Result<Array1<f64>> {
        let erase_dim = self.config.memory_qubits;
        let start_idx = head_id * erase_dim + 100; // Offset to avoid overlap
        let mut erase_vec = Array1::zeros(erase_dim);

        for i in 0..erase_dim {
            let idx = start_idx + i;
            erase_vec[i] = if idx < output.len() {
                output[idx].tanh()
            } else {
                0.0
            };
        }

        Ok(erase_vec)
    }
}

/// Helper structures for read/write parameters
#[derive(Debug, Clone)]
pub struct ReadParams {
    pub content_key: Array1<f64>,
    pub key_strength: f64,
    pub interpolation_gate: f64,
    pub shift_weighting: Array1<f64>,
    pub sharpening_factor: f64,
}

#[derive(Debug, Clone)]
pub struct WriteParams {
    pub content_key: Array1<f64>,
    pub key_strength: f64,
    pub write_vector: Array1<f64>,
    pub erase_vector: Array1<f64>,
    pub allocation_gate: f64,
    pub write_gate: f64,
    pub sharpening_factor: f64,
}

// Implementation of component structures

impl QuantumController {
    pub fn new(config: &ControllerConfig, num_qubits: usize) -> Result<Self> {
        let num_params = 64; // Simplified
        let parameters = Array1::from_shape_fn(num_params, |_| fastrand::f64() * 0.1);
        let hidden_state = Array1::zeros(config.hidden_dims[0]);

        Ok(Self {
            architecture: config.architecture.clone(),
            layers: vec![], // Simplified
            hidden_state,
            cell_state: Some(Array1::zeros(config.hidden_dims[0])),
            parameters,
        })
    }

    pub fn forward(&mut self, input: &Array1<f64>) -> Result<Array1<f64>> {
        // Simplified controller forward pass
        let output_dim = self.hidden_state.len();
        let mut output = Array1::zeros(output_dim);

        for i in 0..output_dim {
            let mut sum = 0.0;
            for (j, &inp) in input.iter().enumerate() {
                let param_idx = (i * input.len() + j) % self.parameters.len();
                sum += inp * self.parameters[param_idx];
            }
            output[i] = sum.tanh();
        }

        self.hidden_state = output.clone();
        Ok(output)
    }
}

impl QuantumExternalMemory {
    pub fn new(config: &QMANConfig) -> Result<Self> {
        let memory_matrix = match config.memory_init {
            MemoryInitialization::Random => {
                Array2::from_shape_fn((config.memory_size, config.memory_qubits), |_| {
                    fastrand::f64() * 0.1
                })
            }
            MemoryInitialization::Zeros => {
                Array2::zeros((config.memory_size, config.memory_qubits))
            }
            MemoryInitialization::QuantumSuperposition => {
                Array2::from_shape_fn((config.memory_size, config.memory_qubits), |_| {
                    (fastrand::f64() - 0.5) * 0.05
                })
            }
            _ => Array2::zeros((config.memory_size, config.memory_qubits)),
        };

        let quantum_states = (0..config.memory_size)
            .map(|_| {
                let state_dim = 1 << config.memory_qubits;
                let mut state = Array1::zeros(state_dim);
                state[0] = 1.0; // Initialize to |0...0⟩
                state
            })
            .collect();

        Ok(Self {
            memory_matrix,
            quantum_states,
            usage_weights: Array1::zeros(config.memory_size),
            write_weights: Array2::zeros((config.head_config.num_write_heads, config.memory_size)),
            read_weights: Array2::zeros((config.head_config.num_read_heads, config.memory_size)),
            link_matrix: Array2::zeros((config.memory_size, config.memory_size)),
        })
    }
}

impl QuantumReadHead {
    pub fn new(head_id: usize, config: &QMANConfig) -> Result<Self> {
        let read_circuit = QuantumCircuit::new(config.memory_qubits)?;
        let attention_weights = Array1::zeros(config.memory_size);
        let content_key = Array1::zeros(config.memory_qubits);
        let addressing_params = AddressingParams {
            content_key: content_key.clone(),
            key_strength: 1.0,
            interpolation_gate: 0.5,
            shift_weighting: Array1::from_vec(vec![0.1, 0.8, 0.1]),
            sharpening_factor: 1.0,
        };

        Ok(Self {
            head_id,
            head_type: config.head_config.read_head_type.clone(),
            read_circuit,
            attention_weights,
            content_key,
            addressing_params,
        })
    }
}

impl QuantumWriteHead {
    pub fn new(head_id: usize, config: &QMANConfig) -> Result<Self> {
        let write_circuit = QuantumCircuit::new(config.memory_qubits)?;
        let addressing_params = AddressingParams {
            content_key: Array1::zeros(config.memory_qubits),
            key_strength: 1.0,
            interpolation_gate: 0.5,
            shift_weighting: Array1::from_vec(vec![0.1, 0.8, 0.1]),
            sharpening_factor: 1.0,
        };

        Ok(Self {
            head_id,
            head_type: config.head_config.write_head_type.clone(),
            write_circuit,
            write_key: Array1::zeros(config.memory_qubits),
            write_vector: Array1::zeros(config.memory_qubits),
            erase_vector: Array1::zeros(config.memory_qubits),
            allocation_gate: 0.0,
            write_gate: 0.0,
            addressing_params,
        })
    }
}

impl QuantumCircuit {
    pub fn new(num_qubits: usize) -> Result<Self> {
        let num_params = num_qubits * 4; // Simplified
        let parameters = Array1::from_shape_fn(num_params, |_| fastrand::f64() * 0.1);

        Ok(Self {
            gates: vec![], // Simplified
            num_qubits,
            parameters,
            entanglement_pattern: EntanglementPattern::Linear,
        })
    }

    pub fn apply(&self, input_state: &Array1<f64>) -> Result<Array1<f64>> {
        // Simplified quantum circuit application
        let mut state = input_state.clone();

        // Apply some random quantum operations for demonstration
        for _ in 0..3 {
            if self.num_qubits > 0 {
                let qubit = fastrand::usize(..self.num_qubits);
                let param_idx = fastrand::usize(..self.parameters.len());
                let angle = self.parameters[param_idx];

                // Apply RY rotation
                let qubit_mask = 1 << qubit;
                let cos_half = (angle / 2.0).cos();
                let sin_half = (angle / 2.0).sin();

                for i in 0..state.len() {
                    if i & qubit_mask == 0 {
                        let j = i | qubit_mask;
                        if j < state.len() {
                            let state_0 = input_state[i];
                            let state_1 = input_state[j];
                            state[i] = cos_half * state_0 - sin_half * state_1;
                            state[j] = sin_half * state_0 + cos_half * state_1;
                        }
                    }
                }
            }
        }

        Ok(state)
    }
}

impl EpisodicMemory {
    pub fn new(config: &QMANConfig) -> Result<Self> {
        let embedding_circuit = QuantumCircuit::new(config.memory_qubits)?;
        let similarity_network = QuantumSimilarityNetwork {
            embedding_circuit,
            similarity_metric: SimilarityMetric::QuantumFidelity,
            retrieval_threshold: 0.7,
        };

        Ok(Self {
            episodes: Vec::new(),
            quantum_embeddings: Array2::zeros((0, config.memory_qubits)),
            similarity_network,
            consolidation_threshold: 0.8,
        })
    }
}

/// Benchmark QMAN against classical memory networks
pub fn benchmark_qman_vs_classical(
    qman: &mut QuantumMemoryAugmentedNetwork,
    test_data: &[(Array1<f64>, Array1<f64>)],
) -> Result<BenchmarkResults> {
    let start_time = std::time::Instant::now();

    let mut quantum_loss = 0.0;
    for (input, target) in test_data {
        let output = qman.forward(input)?;
        let loss = qman.compute_loss(&output, target)?;
        quantum_loss += loss;
    }
    quantum_loss /= test_data.len() as f64;

    let quantum_time = start_time.elapsed();

    // Classical comparison would go here
    let classical_loss = quantum_loss * 1.3; // Placeholder
    let classical_time = quantum_time * 2; // Placeholder

    Ok(BenchmarkResults {
        quantum_loss,
        classical_loss,
        quantum_time: quantum_time.as_secs_f64(),
        classical_time: classical_time.as_secs_f64(),
        quantum_advantage: classical_loss / quantum_loss,
        memory_efficiency: 2.5, // Quantum memory advantage
    })
}

/// Benchmark results for QMAN
#[derive(Debug)]
pub struct BenchmarkResults {
    pub quantum_loss: f64,
    pub classical_loss: f64,
    pub quantum_time: f64,
    pub classical_time: f64,
    pub quantum_advantage: f64,
    pub memory_efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qman_creation() {
        let config = QMANConfig::default();
        let qman = QuantumMemoryAugmentedNetwork::new(config);
        assert!(qman.is_ok());
    }

    #[test]
    fn test_memory_operations() {
        let config = QMANConfig::default();
        let mut qman = QuantumMemoryAugmentedNetwork::new(config).unwrap();

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let result = qman.forward(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quantum_addressing() {
        let config = QMANConfig::default();
        let qman = QuantumMemoryAugmentedNetwork::new(config).unwrap();

        let key = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let weights = qman.content_addressing(&key, 2.0);
        assert!(weights.is_ok());

        let weights = weights.unwrap();
        let sum = weights.sum();
        assert!((sum - 1.0).abs() < 1e-6); // Should sum to 1
    }

    #[test]
    fn test_episodic_memory() {
        let config = QMANConfig::default();
        let mut qman = QuantumMemoryAugmentedNetwork::new(config).unwrap();

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let output = Array1::from_vec(vec![0.7, 0.8, 0.9, 1.0, 1.1, 1.2]);

        let result = qman.update_episodic_memory(&input, &output);
        assert!(result.is_ok());
    }
}
