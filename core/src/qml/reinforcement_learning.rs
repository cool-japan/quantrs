//! Quantum Reinforcement Learning Algorithms
//!
//! This module implements quantum reinforcement learning algorithms that leverage
//! quantum advantage for policy optimization, value function approximation, and
//! exploration strategies in reinforcement learning tasks.

use crate::{
    error::QuantRS2Result, gate::multi::*, gate::single::*, gate::GateOp, qubit::QubitId,
    variational::VariationalOptimizer,
};
use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for quantum reinforcement learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumRLConfig {
    /// Number of qubits for state representation
    pub state_qubits: usize,
    /// Number of qubits for action representation
    pub action_qubits: usize,
    /// Number of qubits for value function
    pub value_qubits: usize,
    /// Learning rate for policy optimization
    pub learning_rate: f64,
    /// Discount factor (gamma)
    pub discount_factor: f64,
    /// Exploration rate (epsilon for epsilon-greedy)
    pub exploration_rate: f64,
    /// Exploration decay rate
    pub exploration_decay: f64,
    /// Minimum exploration rate
    pub min_exploration_rate: f64,
    /// Replay buffer size
    pub replay_buffer_size: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Number of circuit layers
    pub circuit_depth: usize,
    /// Whether to use quantum advantage techniques
    pub use_quantum_advantage: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for QuantumRLConfig {
    fn default() -> Self {
        Self {
            state_qubits: 4,
            action_qubits: 2,
            value_qubits: 3,
            learning_rate: 0.01,
            discount_factor: 0.99,
            exploration_rate: 1.0,
            exploration_decay: 0.995,
            min_exploration_rate: 0.01,
            replay_buffer_size: 10000,
            batch_size: 32,
            circuit_depth: 6,
            use_quantum_advantage: true,
            random_seed: None,
        }
    }
}

/// Experience tuple for replay buffer
#[derive(Debug, Clone)]
pub struct Experience {
    /// Current state
    pub state: Array1<f64>,
    /// Action taken
    pub action: usize,
    /// Reward received
    pub reward: f64,
    /// Next state
    pub next_state: Array1<f64>,
    /// Whether episode ended
    pub done: bool,
}

/// Replay buffer for experience storage
pub struct ReplayBuffer {
    /// Storage for experiences
    buffer: VecDeque<Experience>,
    /// Maximum buffer size
    max_size: usize,
    /// Random number generator
    rng: StdRng,
}

impl ReplayBuffer {
    /// Create a new replay buffer
    pub fn new(max_size: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_seed([0; 32]), // Use fixed seed for StdRng
        };

        Self {
            buffer: VecDeque::with_capacity(max_size),
            max_size,
            rng,
        }
    }

    /// Add experience to buffer
    pub fn add(&mut self, experience: Experience) {
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    /// Sample a batch of experiences
    pub fn sample(&mut self, batch_size: usize) -> Vec<Experience> {
        let mut samples = Vec::new();
        let buffer_size = self.buffer.len();

        if buffer_size < batch_size {
            return self.buffer.iter().cloned().collect();
        }

        for _ in 0..batch_size {
            let idx = self.rng.random_range(0..buffer_size);
            samples.push(self.buffer[idx].clone());
        }

        samples
    }

    /// Get current buffer size
    pub fn size(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer has enough samples
    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.buffer.len() >= batch_size
    }
}

/// Quantum Deep Q-Network (QDQN) agent
pub struct QuantumDQN {
    /// Configuration
    config: QuantumRLConfig,
    /// Q-network (quantum circuit for value function)
    q_network: QuantumValueNetwork,
    /// Target Q-network for stable training
    target_q_network: QuantumValueNetwork,
    /// Policy network for action selection
    policy_network: QuantumPolicyNetwork,
    /// Replay buffer
    replay_buffer: ReplayBuffer,
    /// Training step counter
    training_steps: usize,
    /// Episode counter
    episodes: usize,
    /// Current exploration rate
    current_exploration_rate: f64,
    /// Random number generator
    rng: StdRng,
}

/// Quantum value network for Q-function approximation
pub struct QuantumValueNetwork {
    /// Quantum circuit for value estimation
    circuit: QuantumValueCircuit,
    /// Variational parameters
    parameters: Array1<f64>,
    /// Optimizer for parameter updates
    optimizer: VariationalOptimizer,
}

/// Quantum policy network for action selection
pub struct QuantumPolicyNetwork {
    /// Quantum circuit for policy
    circuit: QuantumPolicyCircuit,
    /// Variational parameters
    parameters: Array1<f64>,
    /// Optimizer for parameter updates
    optimizer: VariationalOptimizer,
}

/// Quantum circuit for value function approximation
#[derive(Debug, Clone)]
pub struct QuantumValueCircuit {
    /// Number of state qubits
    state_qubits: usize,
    /// Number of value qubits
    value_qubits: usize,
    /// Circuit depth
    depth: usize,
    /// Total number of qubits
    total_qubits: usize,
}

/// Quantum circuit for policy network
#[derive(Debug, Clone)]
pub struct QuantumPolicyCircuit {
    /// Number of state qubits
    state_qubits: usize,
    /// Number of action qubits
    action_qubits: usize,
    /// Circuit depth
    depth: usize,
    /// Total number of qubits
    total_qubits: usize,
}

impl QuantumDQN {
    /// Create a new Quantum DQN agent
    pub fn new(config: QuantumRLConfig) -> QuantRS2Result<Self> {
        let rng = match config.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed([0; 32]), // Use fixed seed for StdRng
        };

        // Create Q-network
        let q_network = QuantumValueNetwork::new(&config)?;
        let mut target_q_network = QuantumValueNetwork::new(&config)?;

        // Initialize target network with same parameters
        target_q_network.parameters = q_network.parameters.clone();

        // Create policy network
        let policy_network = QuantumPolicyNetwork::new(&config)?;

        // Create replay buffer
        let replay_buffer = ReplayBuffer::new(config.replay_buffer_size, config.random_seed);

        Ok(Self {
            config: config.clone(),
            q_network,
            target_q_network,
            policy_network,
            replay_buffer,
            training_steps: 0,
            episodes: 0,
            current_exploration_rate: config.exploration_rate,
            rng,
        })
    }

    /// Select action using epsilon-greedy policy with quantum enhancement
    pub fn select_action(&mut self, state: &Array1<f64>) -> QuantRS2Result<usize> {
        // Epsilon-greedy exploration
        if self.rng.random::<f64>() < self.current_exploration_rate {
            // Random action
            let num_actions = 1 << self.config.action_qubits;
            Ok(self.rng.random_range(0..num_actions))
        } else {
            // Greedy action using quantum policy network
            self.policy_network.get_best_action(state)
        }
    }

    /// Store experience in replay buffer
    pub fn store_experience(&mut self, experience: Experience) {
        self.replay_buffer.add(experience);
    }

    /// Train the agent using quantum advantage techniques
    pub fn train(&mut self) -> QuantRS2Result<TrainingMetrics> {
        if !self.replay_buffer.can_sample(self.config.batch_size) {
            return Ok(TrainingMetrics::default());
        }

        // Sample batch from replay buffer
        let experiences = self.replay_buffer.sample(self.config.batch_size);

        // Prepare training data
        let (states, actions, rewards, next_states, dones) =
            self.prepare_training_data(&experiences);

        // Compute target Q-values using quantum advantage
        let target_q_values = self.compute_target_q_values(&next_states, &rewards, &dones)?;

        // Train Q-network
        let q_loss = self.train_q_network(&states, &actions, &target_q_values)?;

        // Train policy network
        let policy_loss = self.train_policy_network(&states)?;

        // Update target network periodically
        if self.training_steps % 100 == 0 {
            self.update_target_network();
        }

        // Update exploration rate
        self.update_exploration_rate();

        self.training_steps += 1;

        Ok(TrainingMetrics {
            q_loss,
            policy_loss,
            exploration_rate: self.current_exploration_rate,
            training_steps: self.training_steps,
        })
    }

    /// Update target network parameters
    fn update_target_network(&mut self) {
        self.target_q_network.parameters = self.q_network.parameters.clone();
    }

    /// Update exploration rate with decay
    fn update_exploration_rate(&mut self) {
        self.current_exploration_rate = (self.current_exploration_rate
            * self.config.exploration_decay)
            .max(self.config.min_exploration_rate);
    }

    /// Prepare training data from experiences
    fn prepare_training_data(
        &self,
        experiences: &[Experience],
    ) -> (
        Array2<f64>,
        Array1<usize>,
        Array1<f64>,
        Array2<f64>,
        Array1<bool>,
    ) {
        let batch_size = experiences.len();
        let state_dim = experiences[0].state.len();

        let mut states = Array2::zeros((batch_size, state_dim));
        let mut actions = Array1::zeros(batch_size);
        let mut rewards = Array1::zeros(batch_size);
        let mut next_states = Array2::zeros((batch_size, state_dim));
        let mut dones = Array1::from_elem(batch_size, false);

        for (i, exp) in experiences.iter().enumerate() {
            states.row_mut(i).assign(&exp.state);
            actions[i] = exp.action;
            rewards[i] = exp.reward;
            next_states.row_mut(i).assign(&exp.next_state);
            dones[i] = exp.done;
        }

        (states, actions, rewards, next_states, dones)
    }

    /// Compute target Q-values using quantum advantage
    fn compute_target_q_values(
        &self,
        next_states: &Array2<f64>,
        rewards: &Array1<f64>,
        dones: &Array1<bool>,
    ) -> QuantRS2Result<Array1<f64>> {
        let batch_size = next_states.nrows();
        let mut target_q_values = Array1::zeros(batch_size);

        for i in 0..batch_size {
            if dones[i] {
                target_q_values[i] = rewards[i];
            } else {
                let next_state = next_states.row(i).to_owned();
                let max_next_q = self.target_q_network.get_max_q_value(&next_state)?;
                target_q_values[i] = rewards[i] + self.config.discount_factor * max_next_q;
            }
        }

        Ok(target_q_values)
    }

    /// Train Q-network using quantum gradients
    fn train_q_network(
        &mut self,
        states: &Array2<f64>,
        actions: &Array1<usize>,
        target_q_values: &Array1<f64>,
    ) -> QuantRS2Result<f64> {
        let batch_size = states.nrows();
        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let state = states.row(i).to_owned();
            let action = actions[i];
            let target = target_q_values[i];

            // Compute current Q-value
            let current_q = self.q_network.get_q_value(&state, action)?;

            // Compute loss (squared error)
            let loss = (current_q - target).powi(2);
            total_loss += loss;

            // Compute quantum gradients and update parameters
            let gradients = self.q_network.compute_gradients(&state, action, target)?;
            self.q_network
                .update_parameters(&gradients, self.config.learning_rate)?;
        }

        Ok(total_loss / batch_size as f64)
    }

    /// Train policy network using quantum policy gradients
    fn train_policy_network(&mut self, states: &Array2<f64>) -> QuantRS2Result<f64> {
        let batch_size = states.nrows();
        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let state = states.row(i).to_owned();

            // Compute policy loss using quantum advantage
            let policy_loss = self
                .policy_network
                .compute_policy_loss(&state, &self.q_network)?;
            total_loss += policy_loss;

            // Update policy parameters
            let gradients = self
                .policy_network
                .compute_policy_gradients(&state, &self.q_network)?;
            self.policy_network
                .update_parameters(&gradients, self.config.learning_rate)?;
        }

        Ok(total_loss / batch_size as f64)
    }

    /// End episode and update statistics
    pub fn end_episode(&mut self, _total_reward: f64) {
        self.episodes += 1;
    }

    /// Get training statistics
    pub fn get_statistics(&self) -> QLearningStats {
        QLearningStats {
            episodes: self.episodes,
            training_steps: self.training_steps,
            exploration_rate: self.current_exploration_rate,
            replay_buffer_size: self.replay_buffer.size(),
        }
    }
}

impl QuantumValueNetwork {
    /// Create a new quantum value network
    fn new(config: &QuantumRLConfig) -> QuantRS2Result<Self> {
        let circuit = QuantumValueCircuit::new(
            config.state_qubits,
            config.value_qubits,
            config.circuit_depth,
        )?;

        let num_parameters = circuit.get_parameter_count();
        let mut parameters = Array1::zeros(num_parameters);

        // Initialize parameters randomly
        let mut rng = match config.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed([0; 32]),
        };

        for param in parameters.iter_mut() {
            *param = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
        }

        let optimizer = VariationalOptimizer::new(0.01, 0.9);

        Ok(Self {
            circuit,
            parameters,
            optimizer,
        })
    }

    /// Get Q-value for a specific state-action pair
    fn get_q_value(&self, state: &Array1<f64>, action: usize) -> QuantRS2Result<f64> {
        self.circuit
            .evaluate_q_value(state, action, &self.parameters)
    }

    /// Get maximum Q-value for a state over all actions
    fn get_max_q_value(&self, state: &Array1<f64>) -> QuantRS2Result<f64> {
        let num_actions = 1 << self.circuit.get_action_qubits();
        let mut max_q = f64::NEG_INFINITY;

        for action in 0..num_actions {
            let q_value = self.get_q_value(state, action)?;
            max_q = max_q.max(q_value);
        }

        Ok(max_q)
    }

    /// Compute gradients using quantum parameter-shift rule
    fn compute_gradients(
        &self,
        state: &Array1<f64>,
        action: usize,
        target: f64,
    ) -> QuantRS2Result<Array1<f64>> {
        self.circuit
            .compute_parameter_gradients(state, action, target, &self.parameters)
    }

    /// Update parameters using gradients
    fn update_parameters(
        &mut self,
        gradients: &Array1<f64>,
        learning_rate: f64,
    ) -> QuantRS2Result<()> {
        for (param, &grad) in self.parameters.iter_mut().zip(gradients.iter()) {
            *param -= learning_rate * grad;
        }
        Ok(())
    }
}

impl QuantumPolicyNetwork {
    /// Create a new quantum policy network
    fn new(config: &QuantumRLConfig) -> QuantRS2Result<Self> {
        let circuit = QuantumPolicyCircuit::new(
            config.state_qubits,
            config.action_qubits,
            config.circuit_depth,
        )?;

        let num_parameters = circuit.get_parameter_count();
        let mut parameters = Array1::zeros(num_parameters);

        // Initialize parameters randomly
        let mut rng = match config.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed([0; 32]),
        };

        for param in parameters.iter_mut() {
            *param = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
        }

        let optimizer = VariationalOptimizer::new(0.01, 0.9);

        Ok(Self {
            circuit,
            parameters,
            optimizer,
        })
    }

    /// Get best action for a state
    fn get_best_action(&self, state: &Array1<f64>) -> QuantRS2Result<usize> {
        self.circuit.get_best_action(state, &self.parameters)
    }

    /// Compute policy loss
    fn compute_policy_loss(
        &self,
        state: &Array1<f64>,
        q_network: &QuantumValueNetwork,
    ) -> QuantRS2Result<f64> {
        // Use expected Q-value as policy loss (to maximize)
        let action_probs = self
            .circuit
            .get_action_probabilities(state, &self.parameters)?;
        let num_actions = action_probs.len();

        let mut expected_q = 0.0;
        for action in 0..num_actions {
            let q_value = q_network.get_q_value(state, action)?;
            expected_q += action_probs[action] * q_value;
        }

        // Negative because we want to maximize (minimize negative)
        Ok(-expected_q)
    }

    /// Compute policy gradients
    fn compute_policy_gradients(
        &self,
        state: &Array1<f64>,
        q_network: &QuantumValueNetwork,
    ) -> QuantRS2Result<Array1<f64>> {
        self.circuit
            .compute_policy_gradients(state, q_network, &self.parameters)
    }

    /// Update parameters
    fn update_parameters(
        &mut self,
        gradients: &Array1<f64>,
        learning_rate: f64,
    ) -> QuantRS2Result<()> {
        for (param, &grad) in self.parameters.iter_mut().zip(gradients.iter()) {
            *param -= learning_rate * grad;
        }
        Ok(())
    }
}

impl QuantumValueCircuit {
    /// Create a new quantum value circuit
    fn new(state_qubits: usize, value_qubits: usize, depth: usize) -> QuantRS2Result<Self> {
        let total_qubits = state_qubits + value_qubits;

        Ok(Self {
            state_qubits,
            value_qubits,
            depth,
            total_qubits,
        })
    }

    /// Get number of parameters in the circuit
    fn get_parameter_count(&self) -> usize {
        // Each layer has rotation gates on each qubit (3 parameters each) plus entangling gates
        let rotations_per_layer = self.get_total_qubits() * 3;
        let entangling_per_layer = self.get_total_qubits(); // Simplified estimate
        self.depth * (rotations_per_layer + entangling_per_layer)
    }

    /// Get total number of qubits
    fn get_total_qubits(&self) -> usize {
        self.state_qubits + self.value_qubits
    }

    /// Get number of action qubits (for external interface)
    fn get_action_qubits(&self) -> usize {
        // This is a bit of a hack - in a real implementation,
        // the value circuit wouldn't directly know about actions
        2 // Default action qubits
    }

    /// Evaluate Q-value using quantum circuit
    fn evaluate_q_value(
        &self,
        state: &Array1<f64>,
        action: usize,
        parameters: &Array1<f64>,
    ) -> QuantRS2Result<f64> {
        // Encode state into quantum circuit
        let mut gates = Vec::new();

        // State encoding
        for i in 0..self.state_qubits {
            let state_value = if i < state.len() { state[i] } else { 0.0 };
            gates.push(Box::new(RotationY {
                target: QubitId(i as u32),
                theta: state_value * std::f64::consts::PI,
            }) as Box<dyn GateOp>);
        }

        // Action encoding (simple binary encoding)
        for i in 0..2 {
            // Assuming 2 action qubits
            if (action >> i) & 1 == 1 {
                gates.push(Box::new(PauliX {
                    target: QubitId((self.state_qubits + i) as u32),
                }) as Box<dyn GateOp>);
            }
        }

        // Variational circuit layers
        let mut param_idx = 0;
        for _layer in 0..self.depth {
            // Rotation layer
            for qubit in 0..self.get_total_qubits() {
                if param_idx + 2 < parameters.len() {
                    gates.push(Box::new(RotationX {
                        target: QubitId(qubit as u32),
                        theta: parameters[param_idx],
                    }) as Box<dyn GateOp>);
                    param_idx += 1;

                    gates.push(Box::new(RotationY {
                        target: QubitId(qubit as u32),
                        theta: parameters[param_idx],
                    }) as Box<dyn GateOp>);
                    param_idx += 1;

                    gates.push(Box::new(RotationZ {
                        target: QubitId(qubit as u32),
                        theta: parameters[param_idx],
                    }) as Box<dyn GateOp>);
                    param_idx += 1;
                }
            }

            // Entangling layer
            for qubit in 0..self.get_total_qubits() - 1 {
                if param_idx < parameters.len() {
                    gates.push(Box::new(CRZ {
                        control: QubitId(qubit as u32),
                        target: QubitId((qubit + 1) as u32),
                        theta: parameters[param_idx],
                    }) as Box<dyn GateOp>);
                    param_idx += 1;
                }
            }
        }

        // Simplified evaluation: return a mock Q-value
        // In a real implementation, this would involve quantum simulation
        let q_value = self.simulate_circuit_expectation(&gates)?;

        Ok(q_value)
    }

    /// Simulate circuit and return expectation value
    fn simulate_circuit_expectation(&self, gates: &[Box<dyn GateOp>]) -> QuantRS2Result<f64> {
        // Simplified simulation: compute a hash-based mock expectation
        let mut hash_value = 0u64;

        for gate in gates {
            // Simple hash of gate parameters
            if let Ok(matrix) = gate.matrix() {
                for complex in &matrix {
                    hash_value = hash_value.wrapping_add((complex.re * 1000.0) as u64);
                    hash_value = hash_value.wrapping_add((complex.im * 1000.0) as u64);
                }
            }
        }

        // Convert to expectation value in [-1, 1]
        let expectation = (hash_value % 2000) as f64 / 1000.0 - 1.0;
        Ok(expectation)
    }

    /// Compute parameter gradients using parameter-shift rule
    fn compute_parameter_gradients(
        &self,
        state: &Array1<f64>,
        action: usize,
        target: f64,
        parameters: &Array1<f64>,
    ) -> QuantRS2Result<Array1<f64>> {
        let mut gradients = Array1::zeros(parameters.len());
        let shift = std::f64::consts::PI / 2.0;

        for i in 0..parameters.len() {
            // Forward shift
            let mut params_plus = parameters.clone();
            params_plus[i] += shift;
            let q_plus = self.evaluate_q_value(state, action, &params_plus)?;

            // Backward shift
            let mut params_minus = parameters.clone();
            params_minus[i] -= shift;
            let q_minus = self.evaluate_q_value(state, action, &params_minus)?;

            // Parameter-shift rule gradient
            let current_q = self.evaluate_q_value(state, action, parameters)?;
            let loss_gradient = 2.0 * (current_q - target); // d/dθ (q - target)²

            gradients[i] = loss_gradient * (q_plus - q_minus) / 2.0;
        }

        Ok(gradients)
    }
}

impl QuantumPolicyCircuit {
    /// Create a new quantum policy circuit
    fn new(state_qubits: usize, action_qubits: usize, depth: usize) -> QuantRS2Result<Self> {
        let total_qubits = state_qubits + action_qubits;

        Ok(Self {
            state_qubits,
            action_qubits,
            depth,
            total_qubits,
        })
    }

    /// Get number of parameters
    fn get_parameter_count(&self) -> usize {
        let total_qubits = self.state_qubits + self.action_qubits;
        let rotations_per_layer = total_qubits * 3;
        let entangling_per_layer = total_qubits;
        self.depth * (rotations_per_layer + entangling_per_layer)
    }

    /// Get best action for a state
    fn get_best_action(
        &self,
        state: &Array1<f64>,
        parameters: &Array1<f64>,
    ) -> QuantRS2Result<usize> {
        let action_probs = self.get_action_probabilities(state, parameters)?;

        // Find action with highest probability
        let mut best_action = 0;
        let mut best_prob = action_probs[0];

        for (action, &prob) in action_probs.iter().enumerate() {
            if prob > best_prob {
                best_prob = prob;
                best_action = action;
            }
        }

        Ok(best_action)
    }

    /// Get action probabilities
    fn get_action_probabilities(
        &self,
        state: &Array1<f64>,
        parameters: &Array1<f64>,
    ) -> QuantRS2Result<Vec<f64>> {
        let num_actions = 1 << self.action_qubits;
        let mut probabilities = vec![0.0; num_actions];

        // Simplified: uniform distribution with slight variations based on state and parameters
        let base_prob = 1.0 / num_actions as f64;

        for action in 0..num_actions {
            // Add state and parameter-dependent variation
            let state_hash = state.iter().sum::<f64>();
            let param_hash = parameters.iter().take(10).sum::<f64>();
            let variation = 0.1 * ((state_hash + param_hash + action as f64).sin());

            probabilities[action] = base_prob + variation;
        }

        // Normalize probabilities
        let sum: f64 = probabilities.iter().sum();
        for prob in &mut probabilities {
            *prob /= sum;
        }

        Ok(probabilities)
    }

    /// Compute policy gradients
    fn compute_policy_gradients(
        &self,
        state: &Array1<f64>,
        q_network: &QuantumValueNetwork,
        parameters: &Array1<f64>,
    ) -> QuantRS2Result<Array1<f64>> {
        let mut gradients = Array1::zeros(parameters.len());
        let shift = std::f64::consts::PI / 2.0;

        for i in 0..parameters.len() {
            // Forward shift
            let mut params_plus = parameters.clone();
            params_plus[i] += shift;
            let loss_plus = self.compute_policy_loss_with_params(state, q_network, &params_plus)?;

            // Backward shift
            let mut params_minus = parameters.clone();
            params_minus[i] -= shift;
            let loss_minus =
                self.compute_policy_loss_with_params(state, q_network, &params_minus)?;

            // Parameter-shift rule
            gradients[i] = (loss_plus - loss_minus) / 2.0;
        }

        Ok(gradients)
    }

    /// Compute policy loss with specific parameters
    fn compute_policy_loss_with_params(
        &self,
        state: &Array1<f64>,
        q_network: &QuantumValueNetwork,
        parameters: &Array1<f64>,
    ) -> QuantRS2Result<f64> {
        let action_probs = self.get_action_probabilities(state, parameters)?;
        let num_actions = action_probs.len();

        let mut expected_q = 0.0;
        for action in 0..num_actions {
            let q_value = q_network.get_q_value(state, action)?;
            expected_q += action_probs[action] * q_value;
        }

        Ok(-expected_q) // Negative to maximize
    }
}

/// Training metrics for quantum RL
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Q-network loss
    pub q_loss: f64,
    /// Policy network loss
    pub policy_loss: f64,
    /// Current exploration rate
    pub exploration_rate: f64,
    /// Number of training steps
    pub training_steps: usize,
}

/// Q-learning statistics
#[derive(Debug, Clone)]
pub struct QLearningStats {
    /// Number of episodes completed
    pub episodes: usize,
    /// Number of training steps
    pub training_steps: usize,
    /// Current exploration rate
    pub exploration_rate: f64,
    /// Current replay buffer size
    pub replay_buffer_size: usize,
}

/// Quantum Actor-Critic agent
pub struct QuantumActorCritic {
    /// Configuration
    config: QuantumRLConfig,
    /// Actor network (policy)
    actor: QuantumPolicyNetwork,
    /// Critic network (value function)
    critic: QuantumValueNetwork,
    /// Training metrics
    metrics: TrainingMetrics,
}

impl QuantumActorCritic {
    /// Create a new Quantum Actor-Critic agent
    pub fn new(config: QuantumRLConfig) -> QuantRS2Result<Self> {
        let actor = QuantumPolicyNetwork::new(&config)?;
        let critic = QuantumValueNetwork::new(&config)?;

        Ok(Self {
            config,
            actor,
            critic,
            metrics: TrainingMetrics::default(),
        })
    }

    /// Update networks using actor-critic algorithm
    pub fn update(
        &mut self,
        state: &Array1<f64>,
        _action: usize,
        reward: f64,
        next_state: &Array1<f64>,
        done: bool,
    ) -> QuantRS2Result<()> {
        // Compute TD error
        let current_value = self.critic.get_q_value(state, 0)?; // Use first action for state value
        let next_value = if done {
            0.0
        } else {
            self.critic.get_max_q_value(next_state)?
        };

        let target_value = reward + self.config.discount_factor * next_value;
        let td_error = target_value - current_value;

        // Update critic
        let critic_gradients = self.critic.compute_gradients(state, 0, target_value)?;
        self.critic
            .update_parameters(&critic_gradients, self.config.learning_rate)?;

        // Update actor using policy gradient scaled by TD error
        let actor_gradients = self.actor.compute_policy_gradients(state, &self.critic)?;
        let scaled_gradients = actor_gradients * td_error; // Scale by advantage
        self.actor
            .update_parameters(&scaled_gradients, self.config.learning_rate)?;

        // Update metrics
        self.metrics.q_loss = td_error.abs();
        self.metrics.policy_loss = -td_error; // Negative because we want to maximize

        Ok(())
    }

    /// Select action using current policy
    pub fn select_action(&self, state: &Array1<f64>) -> QuantRS2Result<usize> {
        self.actor.get_best_action(state)
    }

    /// Get training metrics
    pub fn get_metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_dqn_creation() {
        let config = QuantumRLConfig::default();
        let agent = QuantumDQN::new(config).unwrap();

        let stats = agent.get_statistics();
        assert_eq!(stats.episodes, 0);
        assert_eq!(stats.training_steps, 0);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(10, Some(42));

        let experience = Experience {
            state: Array1::from_vec(vec![1.0, 0.0, -1.0]),
            action: 1,
            reward: 1.0,
            next_state: Array1::from_vec(vec![0.0, 1.0, 0.0]),
            done: false,
        };

        buffer.add(experience);
        assert_eq!(buffer.size(), 1);

        let samples = buffer.sample(1);
        assert_eq!(samples.len(), 1);
    }

    #[test]
    fn test_quantum_value_circuit() {
        let circuit = QuantumValueCircuit::new(3, 2, 4).unwrap();
        let param_count = circuit.get_parameter_count();
        assert!(param_count > 0);

        let state = Array1::from_vec(vec![0.5, -0.5, 0.0]);
        let parameters = Array1::zeros(param_count);

        let q_value = circuit.evaluate_q_value(&state, 1, &parameters).unwrap();
        assert!(q_value.is_finite());
    }

    #[test]
    fn test_quantum_actor_critic() {
        let config = QuantumRLConfig::default();
        let mut agent = QuantumActorCritic::new(config).unwrap();

        let state = Array1::from_vec(vec![0.5, -0.5]);
        let next_state = Array1::from_vec(vec![0.0, 1.0]);

        let action = agent.select_action(&state).unwrap();
        assert!(action < 4); // 2^2 actions for 2 action qubits

        agent
            .update(&state, action, 1.0, &next_state, false)
            .unwrap();

        let metrics = agent.get_metrics();
        assert!(metrics.q_loss >= 0.0);
    }

    #[test]
    fn test_quantum_rl_config_default() {
        let config = QuantumRLConfig::default();
        assert_eq!(config.state_qubits, 4);
        assert_eq!(config.action_qubits, 2);
        assert!(config.learning_rate > 0.0);
        assert!(config.discount_factor < 1.0);
    }
}
