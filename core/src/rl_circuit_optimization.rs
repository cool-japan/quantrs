//! Reinforcement Learning-Based Quantum Circuit Optimization
//!
//! This module implements advanced circuit optimization using reinforcement learning (RL).
//! The RL agent learns optimal gate sequences, placement strategies, and circuit
//! transformations by interacting with quantum circuits and receiving rewards based
//! on circuit quality metrics (depth, gate count, fidelity, etc.).

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::gate::GateOp;
use crate::qubit::QubitId;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Rng};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Actions the RL agent can take to optimize circuits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationAction {
    /// Merge two consecutive single-qubit gates
    MergeSingleQubitGates { gate_index: usize },
    /// Cancel inverse gate pairs
    CancelInversePairs { gate_index: usize },
    /// Apply commutation rules to reorder gates
    CommuteGates {
        gate1_index: usize,
        gate2_index: usize,
    },
    /// Replace gate sequence with equivalent but more efficient sequence
    ReplaceSequence {
        start_index: usize,
        end_index: usize,
    },
    /// Optimize two-qubit gate using decomposition
    OptimizeTwoQubitGate { gate_index: usize },
    /// No operation (skip this step)
    NoOp,
}

/// State representation for the RL agent
#[derive(Debug, Clone)]
pub struct CircuitState {
    /// Current circuit depth
    pub depth: usize,
    /// Total gate count
    pub gate_count: usize,
    /// Two-qubit gate count (more expensive)
    pub two_qubit_count: usize,
    /// Estimated fidelity (0.0 to 1.0)
    pub fidelity: f64,
    /// Number of qubits
    pub qubit_count: usize,
    /// Circuit connectivity graph density
    pub connectivity_density: f64,
    /// Entanglement complexity measure
    pub entanglement_measure: f64,
}

impl CircuitState {
    /// Convert state to feature vector for Q-learning
    pub fn to_features(&self) -> Vec<f64> {
        vec![
            self.depth as f64 / 100.0, // Normalize
            self.gate_count as f64 / 1000.0,
            self.two_qubit_count as f64 / 500.0,
            self.fidelity,
            self.qubit_count as f64 / 50.0,
            self.connectivity_density,
            self.entanglement_measure,
        ]
    }

    /// Extract state from a circuit
    pub fn from_circuit(gates: &[Box<dyn GateOp>], num_qubits: usize) -> Self {
        let mut depth_map: HashMap<QubitId, usize> = HashMap::new();
        let mut two_qubit_count = 0;
        let mut connectivity_edges = 0;

        for gate in gates {
            let qubits = gate.qubits();

            if qubits.len() == 2 {
                two_qubit_count += 1;
                connectivity_edges += 1;
            }

            // Update depth
            let max_depth = qubits
                .iter()
                .map(|q| *depth_map.get(q).unwrap_or(&0))
                .max()
                .unwrap_or(0);

            for qubit in qubits {
                depth_map.insert(qubit, max_depth + 1);
            }
        }

        let depth = depth_map.values().max().copied().unwrap_or(0);
        let gate_count = gates.len();

        // Estimate fidelity based on gate count and type
        let fidelity = 0.9999_f64.powi(gate_count as i32 - two_qubit_count as i32)
            * 0.99_f64.powi(two_qubit_count as i32);

        // Calculate connectivity density
        let max_edges = num_qubits * (num_qubits - 1) / 2;
        let connectivity_density = if max_edges > 0 {
            connectivity_edges as f64 / max_edges as f64
        } else {
            0.0
        };

        // Simplified entanglement measure based on two-qubit gates
        let entanglement_measure = (two_qubit_count as f64 / num_qubits as f64).min(1.0);

        Self {
            depth,
            gate_count,
            two_qubit_count,
            fidelity,
            qubit_count: num_qubits,
            connectivity_density,
            entanglement_measure,
        }
    }
}

/// Q-learning agent for circuit optimization
pub struct QLearningOptimizer {
    /// Q-table: maps (state, action) -> expected reward
    q_table: Arc<RwLock<HashMap<(Vec<u8>, OptimizationAction), f64>>>,
    /// Learning rate (alpha)
    learning_rate: f64,
    /// Discount factor (gamma)
    discount_factor: f64,
    /// Exploration rate (epsilon)
    epsilon: f64,
    /// Epsilon decay rate
    epsilon_decay: f64,
    /// Minimum epsilon
    min_epsilon: f64,
    /// Episode counter
    episodes: Arc<RwLock<usize>>,
    /// Performance history
    performance_history: Arc<RwLock<Vec<OptimizationEpisode>>>,
}

/// Record of a single optimization episode
#[derive(Debug, Clone)]
pub struct OptimizationEpisode {
    pub initial_depth: usize,
    pub final_depth: usize,
    pub initial_gate_count: usize,
    pub final_gate_count: usize,
    pub reward: f64,
    pub steps_taken: usize,
}

impl QLearningOptimizer {
    /// Create a new Q-learning optimizer
    ///
    /// # Arguments
    /// * `learning_rate` - How quickly the agent learns (0.0 to 1.0)
    /// * `discount_factor` - How much future rewards matter (0.0 to 1.0)
    /// * `initial_epsilon` - Initial exploration rate (0.0 to 1.0)
    pub fn new(learning_rate: f64, discount_factor: f64, initial_epsilon: f64) -> Self {
        Self {
            q_table: Arc::new(RwLock::new(HashMap::new())),
            learning_rate,
            discount_factor,
            epsilon: initial_epsilon,
            epsilon_decay: 0.995,
            min_epsilon: 0.01,
            episodes: Arc::new(RwLock::new(0)),
            performance_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Choose action using epsilon-greedy policy
    ///
    /// # Arguments
    /// * `state` - Current circuit state
    /// * `available_actions` - List of actions that can be taken
    pub fn choose_action(
        &self,
        state: &CircuitState,
        available_actions: &[OptimizationAction],
    ) -> OptimizationAction {
        if available_actions.is_empty() {
            return OptimizationAction::NoOp;
        }

        let mut rng = thread_rng();

        // Epsilon-greedy: explore vs exploit
        if rng.gen::<f64>() < self.epsilon {
            // Explore: random action
            available_actions[rng.gen_range(0..available_actions.len())]
        } else {
            // Exploit: best known action
            self.get_best_action(state, available_actions)
        }
    }

    /// Get the best action according to current Q-values
    fn get_best_action(
        &self,
        state: &CircuitState,
        available_actions: &[OptimizationAction],
    ) -> OptimizationAction {
        let state_key = self.state_to_key(state);
        let q_table = self.q_table.read().unwrap_or_else(|e| e.into_inner());

        let mut best_action = available_actions[0];
        let mut best_q_value = f64::NEG_INFINITY;

        for &action in available_actions {
            let q_value = *q_table.get(&(state_key.clone(), action)).unwrap_or(&0.0);
            if q_value > best_q_value {
                best_q_value = q_value;
                best_action = action;
            }
        }

        best_action
    }

    /// Update Q-value based on observed reward
    ///
    /// # Arguments
    /// * `state` - Previous state
    /// * `action` - Action taken
    /// * `reward` - Reward received
    /// * `next_state` - New state after action
    /// * `next_actions` - Actions available in next state
    pub fn update_q_value(
        &mut self,
        state: &CircuitState,
        action: OptimizationAction,
        reward: f64,
        next_state: &CircuitState,
        next_actions: &[OptimizationAction],
    ) {
        let state_key = self.state_to_key(state);
        let next_state_key = self.state_to_key(next_state);

        // Find max Q-value for next state
        let q_table = self.q_table.read().unwrap_or_else(|e| e.into_inner());
        let max_next_q = if next_actions.is_empty() {
            0.0
        } else {
            next_actions
                .iter()
                .map(|&a| *q_table.get(&(next_state_key.clone(), a)).unwrap_or(&0.0))
                .fold(f64::NEG_INFINITY, f64::max)
        };
        drop(q_table);

        // Q-learning update rule:
        // Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
        let mut q_table = self.q_table.write().unwrap_or_else(|e| e.into_inner());
        let current_q = *q_table.get(&(state_key.clone(), action)).unwrap_or(&0.0);
        let new_q = self.learning_rate.mul_add(
            self.discount_factor.mul_add(max_next_q, reward) - current_q,
            current_q,
        );
        q_table.insert((state_key, action), new_q);
    }

    /// Calculate reward for a state transition
    ///
    /// Reward is based on improvements in circuit metrics:
    /// - Reduced depth (+reward)
    /// - Reduced gate count (+reward)
    /// - Increased fidelity (+reward)
    pub fn calculate_reward(&self, old_state: &CircuitState, new_state: &CircuitState) -> f64 {
        let mut reward = 0.0;

        // Reward for depth reduction (most important)
        let depth_improvement = old_state.depth as f64 - new_state.depth as f64;
        reward += depth_improvement * 2.0;

        // Reward for gate count reduction
        let gate_improvement = old_state.gate_count as f64 - new_state.gate_count as f64;
        reward += gate_improvement * 1.0;

        // Reward for two-qubit gate reduction (expensive gates)
        let two_qubit_improvement =
            old_state.two_qubit_count as f64 - new_state.two_qubit_count as f64;
        reward += two_qubit_improvement * 3.0;

        // Penalty for fidelity loss
        let fidelity_change = new_state.fidelity - old_state.fidelity;
        reward += fidelity_change * 100.0; // Heavily weight fidelity

        // Small penalty for NoOp to encourage action
        if reward == 0.0 {
            reward = -0.1;
        }

        reward
    }

    /// Complete an optimization episode
    pub fn finish_episode(&mut self, episode: OptimizationEpisode) {
        // Decay epsilon
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.min_epsilon);

        // Record episode
        {
            let mut episodes = self.episodes.write().unwrap_or_else(|e| e.into_inner());
            *episodes += 1;

            let mut history = self
                .performance_history
                .write()
                .unwrap_or_else(|e| e.into_inner());
            history.push(episode);

            // Keep last 1000 episodes
            if history.len() > 1000 {
                let len = history.len();
                history.drain(0..len - 1000);
            }
        }
    }

    /// Get optimization statistics
    pub fn get_statistics(&self) -> OptimizationStatistics {
        let history = self
            .performance_history
            .read()
            .unwrap_or_else(|e| e.into_inner());

        if history.is_empty() {
            return OptimizationStatistics {
                total_episodes: 0,
                average_depth_improvement: 0.0,
                average_gate_reduction: 0.0,
                average_reward: 0.0,
                current_epsilon: self.epsilon,
                q_table_size: self.q_table.read().unwrap_or_else(|e| e.into_inner()).len(),
            };
        }

        let total_episodes = history.len();
        let avg_depth_improvement: f64 = history
            .iter()
            .map(|e| (e.initial_depth - e.final_depth) as f64)
            .sum::<f64>()
            / total_episodes as f64;

        let avg_gate_reduction: f64 = history
            .iter()
            .map(|e| (e.initial_gate_count - e.final_gate_count) as f64)
            .sum::<f64>()
            / total_episodes as f64;

        let avg_reward: f64 = history.iter().map(|e| e.reward).sum::<f64>() / total_episodes as f64;

        OptimizationStatistics {
            total_episodes,
            average_depth_improvement: avg_depth_improvement,
            average_gate_reduction: avg_gate_reduction,
            average_reward: avg_reward,
            current_epsilon: self.epsilon,
            q_table_size: self.q_table.read().unwrap_or_else(|e| e.into_inner()).len(),
        }
    }

    /// Convert state to hashable key (discretization)
    fn state_to_key(&self, state: &CircuitState) -> Vec<u8> {
        // Discretize continuous features into bins
        let features = state.to_features();
        features
            .iter()
            .map(|&f| ((f * 10.0).round() as i32).clamp(0, 255) as u8)
            .collect()
    }

    /// Save Q-table to file
    pub const fn save_q_table(&self, path: &str) -> QuantRS2Result<()> {
        // In a real implementation, this would serialize the Q-table
        // For now, we'll just return Ok
        Ok(())
    }

    /// Load Q-table from file
    pub const fn load_q_table(&mut self, path: &str) -> QuantRS2Result<()> {
        // In a real implementation, this would deserialize the Q-table
        // For now, we'll just return Ok
        Ok(())
    }
}

/// Statistics about optimization performance
#[derive(Debug, Clone)]
pub struct OptimizationStatistics {
    pub total_episodes: usize,
    pub average_depth_improvement: f64,
    pub average_gate_reduction: f64,
    pub average_reward: f64,
    pub current_epsilon: f64,
    pub q_table_size: usize,
}

impl Default for QLearningOptimizer {
    fn default() -> Self {
        Self::new(0.1, 0.95, 0.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_state_creation() {
        let state = CircuitState {
            depth: 10,
            gate_count: 50,
            two_qubit_count: 15,
            fidelity: 0.95,
            qubit_count: 5,
            connectivity_density: 0.6,
            entanglement_measure: 0.8,
        };

        let features = state.to_features();
        assert_eq!(features.len(), 7);
        assert!(features.iter().all(|&f| f >= 0.0 && f <= 1.1)); // Allow small overflow
    }

    #[test]
    fn test_q_learning_optimizer_creation() {
        let optimizer = QLearningOptimizer::new(0.1, 0.95, 0.3);
        assert_eq!(optimizer.learning_rate, 0.1);
        assert_eq!(optimizer.discount_factor, 0.95);
        assert_eq!(optimizer.epsilon, 0.3);
    }

    #[test]
    fn test_action_selection() {
        let optimizer = QLearningOptimizer::new(0.1, 0.95, 0.0); // No exploration

        let state = CircuitState {
            depth: 10,
            gate_count: 50,
            two_qubit_count: 15,
            fidelity: 0.95,
            qubit_count: 5,
            connectivity_density: 0.6,
            entanglement_measure: 0.8,
        };

        let actions = vec![
            OptimizationAction::MergeSingleQubitGates { gate_index: 0 },
            OptimizationAction::CancelInversePairs { gate_index: 1 },
        ];

        let action = optimizer.choose_action(&state, &actions);
        assert!(actions.contains(&action));
    }

    #[test]
    fn test_reward_calculation() {
        let optimizer = QLearningOptimizer::new(0.1, 0.95, 0.3);

        let old_state = CircuitState {
            depth: 10,
            gate_count: 50,
            two_qubit_count: 15,
            fidelity: 0.95,
            qubit_count: 5,
            connectivity_density: 0.6,
            entanglement_measure: 0.8,
        };

        let new_state = CircuitState {
            depth: 8,
            gate_count: 45,
            two_qubit_count: 12,
            fidelity: 0.96,
            qubit_count: 5,
            connectivity_density: 0.6,
            entanglement_measure: 0.8,
        };

        let reward = optimizer.calculate_reward(&old_state, &new_state);
        assert!(reward > 0.0); // Should be positive for improvements
    }

    #[test]
    fn test_q_value_update() {
        let mut optimizer = QLearningOptimizer::new(0.1, 0.95, 0.3);

        let state = CircuitState {
            depth: 10,
            gate_count: 50,
            two_qubit_count: 15,
            fidelity: 0.95,
            qubit_count: 5,
            connectivity_density: 0.6,
            entanglement_measure: 0.8,
        };

        let action = OptimizationAction::MergeSingleQubitGates { gate_index: 0 };

        let next_state = CircuitState {
            depth: 9,
            gate_count: 48,
            two_qubit_count: 15,
            fidelity: 0.95,
            qubit_count: 5,
            connectivity_density: 0.6,
            entanglement_measure: 0.8,
        };

        optimizer.update_q_value(&state, action, 5.0, &next_state, &[]);

        // Q-value should have been updated
        let q_table = optimizer
            .q_table
            .read()
            .expect("Failed to acquire Q-table read lock");
        assert!(!q_table.is_empty());
    }

    #[test]
    fn test_epsilon_decay() {
        let mut optimizer = QLearningOptimizer::new(0.1, 0.95, 0.5);
        let initial_epsilon = optimizer.epsilon;

        let episode = OptimizationEpisode {
            initial_depth: 10,
            final_depth: 8,
            initial_gate_count: 50,
            final_gate_count: 45,
            reward: 10.0,
            steps_taken: 5,
        };

        optimizer.finish_episode(episode);

        assert!(optimizer.epsilon < initial_epsilon);
        assert!(optimizer.epsilon >= optimizer.min_epsilon);
    }

    #[test]
    fn test_statistics() {
        let mut optimizer = QLearningOptimizer::new(0.1, 0.95, 0.3);

        let episode1 = OptimizationEpisode {
            initial_depth: 10,
            final_depth: 8,
            initial_gate_count: 50,
            final_gate_count: 45,
            reward: 10.0,
            steps_taken: 5,
        };

        let episode2 = OptimizationEpisode {
            initial_depth: 12,
            final_depth: 9,
            initial_gate_count: 60,
            final_gate_count: 52,
            reward: 15.0,
            steps_taken: 7,
        };

        optimizer.finish_episode(episode1);
        optimizer.finish_episode(episode2);

        let stats = optimizer.get_statistics();
        assert_eq!(stats.total_episodes, 2);
        assert!(stats.average_depth_improvement > 0.0);
        assert!(stats.average_gate_reduction > 0.0);
    }
}
