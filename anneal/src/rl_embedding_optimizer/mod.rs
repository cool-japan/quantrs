//! Reinforcement Learning for Embedding Optimization
//!
//! This module implements advanced reinforcement learning techniques for optimizing
//! graph embeddings in quantum annealing hardware. It uses deep Q-learning and
//! policy gradient methods to learn optimal embedding strategies across different
//! problem types and hardware topologies.
//!
//! Key features:
//! - Deep Q-Network (DQN) for embedding decision making
//! - Policy gradient methods for continuous optimization
//! - Multi-objective embedding optimization
//! - Transfer learning across problem domains
//! - Hardware topology awareness
//! - Real-time adaptation and learning

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::embedding::{Embedding, HardwareTopology};
use crate::ising::IsingModel;

pub mod cache;
pub mod embedding;
pub mod error;
pub mod networks;
pub mod state_action;
pub mod training;
pub mod types;
pub mod utils;

pub use cache::{CacheManager, PerformanceTracker};
pub use embedding::EmbeddingOptimizer;
pub use error::{RLEmbeddingError, RLEmbeddingResult};
pub use networks::*;
pub use state_action::StateActionProcessor;
pub use training::TrainingManager;
pub use types::*;
pub use utils::*;

/// Reinforcement learning embedding optimizer
#[derive(Debug, Clone)]
pub struct RLEmbeddingOptimizer {
    /// Deep Q-Network for embedding decisions
    pub dqn: EmbeddingDQN,
    /// Policy network for continuous optimization
    pub policy_network: EmbeddingPolicyNetwork,
    /// Optimizer configuration
    pub config: RLEmbeddingConfig,
    /// Experience replay buffer
    pub experience_buffer: Vec<EmbeddingExperience>,
    /// Training statistics
    pub training_stats: RLTrainingStats,
    /// Hardware topology knowledge
    pub hardware_topologies: HashMap<String, HardwareTopology>,
    /// Problem embeddings cache
    pub embedding_cache: HashMap<String, CachedEmbedding>,
    /// Performance metrics
    pub performance_metrics: RLPerformanceMetrics,
}

impl RLEmbeddingOptimizer {
    /// Create a new RL embedding optimizer
    pub fn new(config: RLEmbeddingConfig) -> RLEmbeddingResult<Self> {
        // Validate configuration
        utils::validate_config(&config)?;

        let dqn = EmbeddingDQN::new(&config.dqn_layers, config.seed)?;
        let policy_network = EmbeddingPolicyNetwork::new(&config.policy_layers, config.seed)?;

        Ok(Self {
            dqn,
            policy_network,
            config,
            experience_buffer: Vec::new(),
            training_stats: RLTrainingStats::new(),
            hardware_topologies: HashMap::new(),
            embedding_cache: HashMap::new(),
            performance_metrics: RLPerformanceMetrics::new(),
        })
    }

    /// Optimize embedding for a given problem
    pub fn optimize_embedding(
        &mut self,
        problem: &IsingModel,
        hardware: &HardwareTopology,
    ) -> RLEmbeddingResult<Embedding> {
        let start_time = Instant::now();

        // Extract state features
        let state = StateActionProcessor::extract_state_features(problem, hardware)?;

        // Check cache for similar problems
        if let Some(cached_embedding) = CacheManager::check_cache(&self.embedding_cache, &state) {
            let cached_result = cached_embedding.embedding.clone();
            self.performance_metrics.problems_solved += 1;
            return Ok(cached_result);
        }

        // Generate initial embedding using baseline method
        let mut current_embedding =
            EmbeddingOptimizer::generate_initial_embedding(problem, hardware)?;
        let mut current_state = state;

        // Iterative improvement using RL
        for step in 0..100 {
            // Maximum steps
            // Select action using current policy
            let epsilon = TrainingManager::get_current_epsilon(
                &self.training_stats,
                &self.config.exploration_config,
            );
            let action = TrainingManager::select_action(&self.dqn, &current_state, epsilon)?;

            // Apply action to improve embedding
            let new_embedding = StateActionProcessor::apply_action(&current_embedding, &action)?;

            // Evaluate new embedding
            let reward = EmbeddingOptimizer::calculate_reward(
                &current_embedding,
                &new_embedding,
                hardware,
                &self.config.objective_weights,
            )?;

            // Update state
            let new_state = EmbeddingOptimizer::update_state(
                &current_state,
                &action,
                &new_embedding,
                hardware,
            )?;

            // Store experience
            let experience = EmbeddingExperience {
                state: current_state.clone(),
                action: action.clone(),
                reward,
                next_state: new_state.clone(),
                done: EmbeddingOptimizer::is_terminal_state(&new_state),
                context: utils::create_experience_context(
                    EmbeddingOptimizer::classify_problem_type(problem),
                    utils::hardware_topology_to_string(hardware),
                    self.performance_metrics.problems_solved,
                    step,
                ),
            };

            TrainingManager::store_experience(
                &mut self.experience_buffer,
                experience,
                self.config.buffer_size,
            );

            // Update current state and embedding
            if reward > 0.0 {
                current_embedding = new_embedding;
            }
            current_state = new_state;

            // Check for termination
            if EmbeddingOptimizer::is_terminal_state(&current_state) {
                break;
            }
        }

        // Cache the result
        CacheManager::cache_embedding(
            &mut self.embedding_cache,
            problem,
            &current_embedding,
            hardware,
            start_time.elapsed(),
        )?;

        // Update performance metrics
        self.performance_metrics.problems_solved += 1;

        Ok(current_embedding)
    }

    /// Train the RL networks
    pub fn train(&mut self, num_epochs: usize) -> RLEmbeddingResult<()> {
        TrainingManager::train_networks(
            &mut self.dqn,
            &mut self.policy_network,
            &self.experience_buffer,
            &mut self.training_stats,
            &self.config,
            num_epochs,
        )
    }

    /// Get performance report
    #[must_use]
    pub fn get_performance_report(&self) -> String {
        PerformanceTracker::generate_performance_report(&self.performance_metrics)
    }

    /// Get configuration summary
    #[must_use]
    pub fn get_config_summary(&self) -> String {
        utils::config_summary(&self.config)
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.embedding_cache.clear();
    }

    /// Cleanup old cache entries
    pub fn cleanup_cache(&mut self, max_age: Duration) {
        CacheManager::cleanup_cache(&mut self.embedding_cache, max_age);
    }

    /// Get cache statistics
    #[must_use]
    pub fn get_cache_statistics(&self) -> f64 {
        CacheManager::calculate_cache_hit_rate(&self.embedding_cache)
    }

    /// Add hardware topology knowledge
    pub fn add_hardware_topology(&mut self, name: String, topology: HardwareTopology) {
        self.hardware_topologies.insert(name, topology);
    }

    /// Get training statistics
    #[must_use]
    pub const fn get_training_stats(&self) -> &RLTrainingStats {
        &self.training_stats
    }

    /// Update performance metrics
    pub fn update_performance_metrics(&mut self, embedding_quality: f64, baseline_quality: f64) {
        PerformanceTracker::update_performance_metrics(
            &mut self.performance_metrics,
            embedding_quality,
            baseline_quality,
        );
    }
}

/// Create default RL embedding optimizer
pub fn create_rl_embedding_optimizer() -> RLEmbeddingResult<RLEmbeddingOptimizer> {
    RLEmbeddingOptimizer::new(RLEmbeddingConfig::default())
}

/// Create RL optimizer with custom configuration
pub fn create_custom_rl_embedding_optimizer(
    dqn_layers: Vec<usize>,
    policy_layers: Vec<usize>,
    learning_rate: f64,
) -> RLEmbeddingResult<RLEmbeddingOptimizer> {
    let config = utils::create_custom_config(dqn_layers, policy_layers, learning_rate);
    RLEmbeddingOptimizer::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rl_optimizer_creation() {
        let optimizer = create_rl_embedding_optimizer().expect("Failed to create RL optimizer");
        assert_eq!(optimizer.config.dqn_layers, vec![128, 256, 128, 64]);
        assert_eq!(optimizer.config.learning_rate, 0.0001);
    }

    #[test]
    fn test_embedding_network_creation() {
        let network =
            EmbeddingNetwork::new(&[10, 20, 5], Some(42)).expect("Failed to create network");
        assert_eq!(network.layers.len(), 2);
        assert_eq!(network.layers[0].weights.len(), 20);
        assert_eq!(network.layers[0].weights[0].len(), 10);
    }

    #[test]
    fn test_network_forward_pass() {
        let network =
            EmbeddingNetwork::new(&[3, 5, 2], Some(42)).expect("Failed to create network");
        let input = vec![1.0, 0.5, -0.5];
        let output = network.forward(&input).expect("Failed forward pass");
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_state_feature_extraction() {
        let _optimizer = create_rl_embedding_optimizer().expect("Failed to create RL optimizer");
        let mut ising = IsingModel::new(4);
        ising.set_bias(0, 1.0).expect("Failed to set bias");
        ising
            .set_coupling(0, 1, -0.5)
            .expect("Failed to set coupling");

        let hardware = HardwareTopology::Chimera(2, 2, 4);
        let state = StateActionProcessor::extract_state_features(&ising, &hardware)
            .expect("Failed to extract state features");

        assert_eq!(state.problem_features.num_vertices, 4);
        // Chimera(2, 2, 4) = 2 * 2 * 2 * 4 = 32 physical qubits
        assert_eq!(state.hardware_features.num_physical_qubits, 32);
    }

    #[test]
    fn test_action_sampling() {
        let _optimizer = create_rl_embedding_optimizer().expect("Failed to create RL optimizer");
        let ising = IsingModel::new(4);
        let hardware = HardwareTopology::Chimera(2, 2, 4);
        let state = StateActionProcessor::extract_state_features(&ising, &hardware)
            .expect("Failed to extract state features");

        let action =
            StateActionProcessor::sample_random_action(&state).expect("Failed to sample action");
        // Should not panic and return a valid action
        match action {
            DiscreteEmbeddingAction::AddToChain {
                logical_qubit,
                physical_qubit,
            } => {
                assert!(logical_qubit < 4);
                assert!(physical_qubit < 32);
            }
            _ => {} // Other actions are also valid
        }
    }

    #[test]
    fn test_embedding_quality_evaluation() {
        let _optimizer = create_rl_embedding_optimizer().expect("Failed to create RL optimizer");
        let hardware = HardwareTopology::Chimera(2, 2, 4);

        let mut embedding = Embedding {
            chains: HashMap::new(),
            qubit_to_variable: HashMap::new(),
        };

        embedding.chains.insert(0, vec![0, 1]);
        embedding.chains.insert(1, vec![2]);

        let quality = EmbeddingOptimizer::evaluate_embedding_quality(&embedding, &hardware)
            .expect("Failed to evaluate embedding quality");
        assert!(quality.is_finite());
    }

    #[test]
    fn test_config_validation() {
        let mut config = RLEmbeddingConfig::default();

        // Valid config should pass
        assert!(utils::validate_config(&config).is_ok());

        // Invalid learning rate should fail
        config.learning_rate = -0.1;
        assert!(utils::validate_config(&config).is_err());

        // Reset and test invalid buffer size
        config.learning_rate = 0.001;
        config.buffer_size = 0;
        assert!(utils::validate_config(&config).is_err());
    }

    #[test]
    fn test_memory_estimation() {
        let config = RLEmbeddingConfig::default();
        let memory = utils::estimate_memory_usage(&config);
        assert!(memory > 0);
    }
}
