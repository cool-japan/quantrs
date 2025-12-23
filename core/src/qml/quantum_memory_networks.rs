//! Quantum Memory Networks
//!
//! This module implements memory-augmented quantum neural networks that can
//! store and retrieve quantum information for enhanced learning capabilities.
//!
//! # Theoretical Background
//!
//! Quantum Memory Networks extend neural networks with external quantum memory,
//! allowing them to store and recall quantum states. This is particularly useful
//! for tasks requiring long-term dependencies and complex reasoning.
//!
//! # Key Components
//!
//! - **Quantum Memory Bank**: Stores quantum states with addressable slots
//! - **Attention-Based Addressing**: Uses quantum attention to read/write memory
//! - **Memory Controller**: Manages memory operations and updates
//! - **Neural Turing Machine-like Architecture**: Quantum version of differentiable memory
//!
//! # References
//!
//! - "Quantum Neural Turing Machines" (2023)
//! - "Memory-Augmented Quantum Neural Networks" (2024)
//! - "Differentiable Quantum Memory Systems" (2024)

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::f64::consts::PI;

/// Configuration for quantum memory network
#[derive(Debug, Clone)]
pub struct QuantumMemoryConfig {
    /// Number of memory slots
    pub memory_slots: usize,
    /// Number of qubits per memory slot
    pub qubits_per_slot: usize,
    /// Controller network size
    pub controller_size: usize,
    /// Number of read heads
    pub num_read_heads: usize,
    /// Number of write heads
    pub num_write_heads: usize,
    /// Memory initialization strategy
    pub init_strategy: MemoryInitStrategy,
}

impl Default for QuantumMemoryConfig {
    fn default() -> Self {
        Self {
            memory_slots: 128,
            qubits_per_slot: 4,
            controller_size: 64,
            num_read_heads: 1,
            num_write_heads: 1,
            init_strategy: MemoryInitStrategy::Zero,
        }
    }
}

/// Memory initialization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryInitStrategy {
    /// Initialize to |0⟩ states
    Zero,
    /// Random product states
    RandomProduct,
    /// Maximally entangled states
    MaximallyEntangled,
}

/// Quantum memory bank
#[derive(Debug, Clone)]
pub struct QuantumMemory {
    /// Memory slots (each is a quantum state)
    slots: Array2<Complex64>,
    /// Number of slots
    num_slots: usize,
    /// Qubits per slot
    qubits_per_slot: usize,
    /// Memory usage tracking
    usage_weights: Array1<f64>,
}

impl QuantumMemory {
    /// Create new quantum memory
    pub fn new(
        num_slots: usize,
        qubits_per_slot: usize,
        init_strategy: MemoryInitStrategy,
    ) -> Self {
        let state_dim = 1 << qubits_per_slot;
        let mut slots = Array2::zeros((num_slots, state_dim));

        // Initialize memory based on strategy
        match init_strategy {
            MemoryInitStrategy::Zero => {
                // All slots in |0...0⟩ state
                for i in 0..num_slots {
                    slots[[i, 0]] = Complex64::new(1.0, 0.0);
                }
            }
            MemoryInitStrategy::RandomProduct => {
                let mut rng = thread_rng();
                for i in 0..num_slots {
                    // Random product state
                    for j in 0..state_dim {
                        slots[[i, j]] =
                            Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
                    }
                    // Normalize
                    let norm: f64 = slots
                        .row(i)
                        .iter()
                        .map(|x| x.norm_sqr())
                        .sum::<f64>()
                        .sqrt();
                    for j in 0..state_dim {
                        slots[[i, j]] = slots[[i, j]] / norm;
                    }
                }
            }
            MemoryInitStrategy::MaximallyEntangled => {
                // Bell states for pairs of qubits
                for i in 0..num_slots {
                    let sqrt_half = 1.0 / (2.0_f64).sqrt();
                    slots[[i, 0]] = Complex64::new(sqrt_half, 0.0);
                    slots[[i, state_dim - 1]] = Complex64::new(sqrt_half, 0.0);
                }
            }
        }

        let usage_weights = Array1::zeros(num_slots);

        Self {
            slots,
            num_slots,
            qubits_per_slot,
            usage_weights,
        }
    }

    /// Read from memory using attention weights
    pub fn read(&self, attention_weights: &Array1<f64>) -> QuantRS2Result<Array1<Complex64>> {
        if attention_weights.len() != self.num_slots {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Attention weights size {} does not match memory slots {}",
                attention_weights.len(),
                self.num_slots
            )));
        }

        let state_dim = self.slots.shape()[1];
        let mut read_state = Array1::zeros(state_dim);

        // Weighted sum of memory slots
        for i in 0..self.num_slots {
            let weight = attention_weights[i];
            for j in 0..state_dim {
                read_state[j] = read_state[j] + self.slots[[i, j]] * weight;
            }
        }

        // Normalize
        let norm: f64 = read_state
            .iter()
            .map(|x: &Complex64| x.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm > 1e-10 {
            for i in 0..state_dim {
                read_state[i] = read_state[i] / norm;
            }
        }

        Ok(read_state)
    }

    /// Write to memory using attention weights
    pub fn write(
        &mut self,
        attention_weights: &Array1<f64>,
        write_vector: &Array1<Complex64>,
        erase_vector: &Array1<f64>,
    ) -> QuantRS2Result<()> {
        if attention_weights.len() != self.num_slots {
            return Err(QuantRS2Error::InvalidInput(
                "Attention weights size mismatch".to_string(),
            ));
        }

        let state_dim = self.slots.shape()[1];

        if write_vector.len() != state_dim || erase_vector.len() != state_dim {
            return Err(QuantRS2Error::InvalidInput(
                "Write/erase vector size mismatch".to_string(),
            ));
        }

        // Update each memory slot
        for i in 0..self.num_slots {
            let weight = attention_weights[i];

            // Erase step: M[i] = M[i] * (1 - w[i] * e)
            for j in 0..state_dim {
                let erase_amount = weight * erase_vector[j];
                self.slots[[i, j]] = self.slots[[i, j]] * (1.0 - erase_amount);
            }

            // Add step: M[i] = M[i] + w[i] * a
            for j in 0..state_dim {
                self.slots[[i, j]] = self.slots[[i, j]] + write_vector[j] * weight;
            }

            // Renormalize to maintain quantum state
            let norm: f64 = self
                .slots
                .row(i)
                .iter()
                .map(|x| x.norm_sqr())
                .sum::<f64>()
                .sqrt();
            if norm > 1e-10 {
                for j in 0..state_dim {
                    self.slots[[i, j]] = self.slots[[i, j]] / norm;
                }
            }
        }

        Ok(())
    }

    /// Get memory state
    pub fn get_state(&self, slot: usize) -> QuantRS2Result<Array1<Complex64>> {
        if slot >= self.num_slots {
            return Err(QuantRS2Error::InvalidInput(
                "Invalid slot index".to_string(),
            ));
        }

        Ok(self.slots.row(slot).to_owned())
    }

    /// Update usage weights
    pub fn update_usage(&mut self, read_weights: &Array1<f64>, write_weights: &Array1<f64>) {
        for i in 0..self.num_slots {
            let usage = read_weights[i] + write_weights[i];
            self.usage_weights[i] = (self.usage_weights[i] + usage).min(1.0);
        }
    }

    /// Get least used slot (for allocation)
    pub fn get_least_used_slot(&self) -> usize {
        let mut min_usage = f64::INFINITY;
        let mut min_idx = 0;

        for i in 0..self.num_slots {
            if self.usage_weights[i] < min_usage {
                min_usage = self.usage_weights[i];
                min_idx = i;
            }
        }

        min_idx
    }
}

/// Memory controller that manages read/write operations
#[derive(Debug, Clone)]
pub struct QuantumMemoryController {
    /// Input size
    input_size: usize,
    /// Controller hidden size
    hidden_size: usize,
    /// Memory configuration
    memory_config: QuantumMemoryConfig,
    /// Controller parameters (LSTM-like)
    w_input: Array2<f64>,
    w_hidden: Array2<f64>,
    b_hidden: Array1<f64>,
    /// Read head parameters
    w_read_key: Vec<Array2<f64>>,
    /// Write head parameters
    w_write_key: Vec<Array2<f64>>,
    w_write_add: Vec<Array2<Complex64>>,
    w_write_erase: Vec<Array2<f64>>,
    /// Controller state
    hidden_state: Array1<f64>,
}

impl QuantumMemoryController {
    /// Create new memory controller
    pub fn new(input_size: usize, memory_config: QuantumMemoryConfig) -> Self {
        let hidden_size = memory_config.controller_size;
        let mut rng = thread_rng();

        let scale_input = (2.0 / input_size as f64).sqrt();
        let scale_hidden = (2.0 / hidden_size as f64).sqrt();

        let w_input = Array2::from_shape_fn((hidden_size, input_size), |_| {
            rng.gen_range(-scale_input..scale_input)
        });

        let w_hidden = Array2::from_shape_fn((hidden_size, hidden_size), |_| {
            rng.gen_range(-scale_hidden..scale_hidden)
        });

        let b_hidden = Array1::zeros(hidden_size);

        // Initialize read head parameters
        let mut w_read_key = Vec::with_capacity(memory_config.num_read_heads);
        for _ in 0..memory_config.num_read_heads {
            let state_dim = 1 << memory_config.qubits_per_slot;
            w_read_key.push(Array2::from_shape_fn((state_dim, hidden_size), |_| {
                rng.gen_range(-scale_hidden..scale_hidden)
            }));
        }

        // Initialize write head parameters
        let mut w_write_key = Vec::with_capacity(memory_config.num_write_heads);
        let mut w_write_add = Vec::with_capacity(memory_config.num_write_heads);
        let mut w_write_erase = Vec::with_capacity(memory_config.num_write_heads);

        for _ in 0..memory_config.num_write_heads {
            let state_dim = 1 << memory_config.qubits_per_slot;

            w_write_key.push(Array2::from_shape_fn((state_dim, hidden_size), |_| {
                rng.gen_range(-scale_hidden..scale_hidden)
            }));

            w_write_add.push(Array2::from_shape_fn((state_dim, hidden_size), |_| {
                Complex64::new(
                    rng.gen_range(-scale_hidden..scale_hidden),
                    rng.gen_range(-scale_hidden..scale_hidden),
                )
            }));

            w_write_erase.push(Array2::from_shape_fn((state_dim, hidden_size), |_| {
                rng.gen_range(-scale_hidden..scale_hidden)
            }));
        }

        let hidden_state = Array1::zeros(hidden_size);

        Self {
            input_size,
            hidden_size,
            memory_config,
            w_input,
            w_hidden,
            b_hidden,
            w_read_key,
            w_write_key,
            w_write_add,
            w_write_erase,
            hidden_state,
        }
    }

    /// Update controller state
    pub fn update_state(&mut self, input: &Array1<f64>) -> QuantRS2Result<()> {
        if input.len() != self.input_size {
            return Err(QuantRS2Error::InvalidInput(
                "Input size mismatch".to_string(),
            ));
        }

        // Simple feedforward update (can be extended to LSTM)
        let mut new_hidden = self.b_hidden.clone();

        // Add input contribution
        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                new_hidden[i] += self.w_input[[i, j]] * input[j];
            }
        }

        // Add hidden state contribution
        for i in 0..self.hidden_size {
            for j in 0..self.hidden_size {
                new_hidden[i] += self.w_hidden[[i, j]] * self.hidden_state[j];
            }
        }

        // Activation (tanh)
        for i in 0..self.hidden_size {
            new_hidden[i] = new_hidden[i].tanh();
        }

        self.hidden_state = new_hidden;
        Ok(())
    }

    /// Generate read attention weights
    pub fn generate_read_weights(
        &self,
        head_idx: usize,
        memory: &QuantumMemory,
    ) -> QuantRS2Result<Array1<f64>> {
        if head_idx >= self.memory_config.num_read_heads {
            return Err(QuantRS2Error::InvalidInput(
                "Invalid read head index".to_string(),
            ));
        }

        // Generate read key from controller state
        let state_dim = 1 << self.memory_config.qubits_per_slot;
        let mut read_key = Array1::zeros(state_dim);

        for i in 0..state_dim {
            for j in 0..self.hidden_size {
                read_key[i] += self.w_read_key[head_idx][[i, j]] * self.hidden_state[j];
            }
        }

        // Compute similarity with memory slots (quantum fidelity)
        let mut similarities = Array1::zeros(memory.num_slots);
        for i in 0..memory.num_slots {
            let mem_state = memory.get_state(i)?;
            let mut fidelity = 0.0;

            for j in 0..state_dim {
                let key_complex = Complex64::new(read_key[j], 0.0);
                fidelity += (key_complex.conj() * mem_state[j]).norm_sqr();
            }

            similarities[i] = fidelity;
        }

        // Apply softmax
        let max_sim = similarities
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut weights = Array1::zeros(memory.num_slots);
        let mut sum = 0.0;

        for i in 0..memory.num_slots {
            weights[i] = (similarities[i] - max_sim).exp();
            sum += weights[i];
        }

        // Normalize
        for i in 0..memory.num_slots {
            weights[i] /= sum;
        }

        Ok(weights)
    }

    /// Generate write parameters
    pub fn generate_write_params(
        &self,
        head_idx: usize,
        memory: &QuantumMemory,
    ) -> QuantRS2Result<(Array1<f64>, Array1<Complex64>, Array1<f64>)> {
        if head_idx >= self.memory_config.num_write_heads {
            return Err(QuantRS2Error::InvalidInput(
                "Invalid write head index".to_string(),
            ));
        }

        let state_dim = 1 << self.memory_config.qubits_per_slot;

        // Generate write key
        let mut write_key = Array1::zeros(state_dim);
        for i in 0..state_dim {
            for j in 0..self.hidden_size {
                write_key[i] += self.w_write_key[head_idx][[i, j]] * self.hidden_state[j];
            }
        }

        // Compute attention weights (same as read)
        let mut similarities = Array1::zeros(memory.num_slots);
        for i in 0..memory.num_slots {
            let mem_state = memory.get_state(i)?;
            let mut fidelity = 0.0;

            for j in 0..state_dim {
                let key_complex = Complex64::new(write_key[j], 0.0);
                fidelity += (key_complex.conj() * mem_state[j]).norm_sqr();
            }

            similarities[i] = fidelity;
        }

        let max_sim = similarities
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut write_weights = Array1::zeros(memory.num_slots);
        let mut sum = 0.0;

        for i in 0..memory.num_slots {
            write_weights[i] = (similarities[i] - max_sim).exp();
            sum += write_weights[i];
        }

        for i in 0..memory.num_slots {
            write_weights[i] /= sum;
        }

        // Generate add vector (quantum state to add)
        let mut add_vector = Array1::zeros(state_dim);
        for i in 0..state_dim {
            for j in 0..self.hidden_size {
                add_vector[i] =
                    add_vector[i] + self.w_write_add[head_idx][[i, j]] * self.hidden_state[j];
            }
        }

        // Normalize add vector
        let norm: f64 = add_vector
            .iter()
            .map(|x: &Complex64| x.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm > 1e-10 {
            for i in 0..state_dim {
                add_vector[i] = add_vector[i] / norm;
            }
        }

        // Generate erase vector
        let mut erase_vector: Array1<f64> = Array1::zeros(state_dim);
        for i in 0..state_dim {
            for j in 0..self.hidden_size {
                erase_vector[i] += self.w_write_erase[head_idx][[i, j]] * self.hidden_state[j];
            }
            // Sigmoid to [0, 1]
            erase_vector[i] = 1.0 / (1.0 + (-erase_vector[i]).exp());
        }

        Ok((write_weights, add_vector, erase_vector))
    }

    /// Reset controller state
    pub fn reset(&mut self) {
        self.hidden_state = Array1::zeros(self.hidden_size);
    }
}

/// Complete quantum memory network
#[derive(Debug, Clone)]
pub struct QuantumMemoryNetwork {
    /// Memory bank
    memory: QuantumMemory,
    /// Memory controller
    controller: QuantumMemoryController,
    /// Configuration
    config: QuantumMemoryConfig,
}

impl QuantumMemoryNetwork {
    /// Create new quantum memory network
    pub fn new(input_size: usize, config: QuantumMemoryConfig) -> Self {
        let memory = QuantumMemory::new(
            config.memory_slots,
            config.qubits_per_slot,
            config.init_strategy,
        );

        let controller = QuantumMemoryController::new(input_size, config.clone());

        Self {
            memory,
            controller,
            config,
        }
    }

    /// Process one step
    pub fn step(&mut self, input: &Array1<f64>) -> QuantRS2Result<Vec<Array1<Complex64>>> {
        // Update controller with input
        self.controller.update_state(input)?;

        // Read from memory
        let mut read_outputs = Vec::with_capacity(self.config.num_read_heads);
        let mut all_read_weights = Vec::new();

        for i in 0..self.config.num_read_heads {
            let read_weights = self.controller.generate_read_weights(i, &self.memory)?;
            let read_output = self.memory.read(&read_weights)?;
            read_outputs.push(read_output);
            all_read_weights.push(read_weights);
        }

        // Write to memory
        let mut all_write_weights = Vec::new();

        for i in 0..self.config.num_write_heads {
            let (write_weights, add_vector, erase_vector) =
                self.controller.generate_write_params(i, &self.memory)?;

            self.memory
                .write(&write_weights, &add_vector, &erase_vector)?;
            all_write_weights.push(write_weights);
        }

        // Update memory usage
        for (read_w, write_w) in all_read_weights.iter().zip(all_write_weights.iter()) {
            self.memory.update_usage(read_w, write_w);
        }

        Ok(read_outputs)
    }

    /// Reset network state
    pub fn reset(&mut self) {
        self.controller.reset();
        self.memory = QuantumMemory::new(
            self.config.memory_slots,
            self.config.qubits_per_slot,
            self.config.init_strategy,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_memory() {
        let memory = QuantumMemory::new(10, 4, MemoryInitStrategy::Zero);

        let attention = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let read_state = memory.read(&attention).unwrap();

        assert_eq!(read_state.len(), 16); // 2^4 states
    }

    #[test]
    fn test_quantum_memory_network() {
        let config = QuantumMemoryConfig {
            memory_slots: 16,
            qubits_per_slot: 3,
            controller_size: 32,
            num_read_heads: 1,
            num_write_heads: 1,
            init_strategy: MemoryInitStrategy::Zero,
        };

        let mut network = QuantumMemoryNetwork::new(10, config);

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);
        let outputs = network.step(&input).unwrap();

        assert_eq!(outputs.len(), 1); // One read head
        assert_eq!(outputs[0].len(), 8); // 2^3 states per slot
    }
}
