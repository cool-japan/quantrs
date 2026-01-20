//! RL training logic and experience replay

use scirs2_core::random::prelude::*;
use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::{Rng, SeedableRng};
use std::time::Instant;

use super::error::{RLEmbeddingError, RLEmbeddingResult};
use super::networks::{EmbeddingDQN, EmbeddingPolicyNetwork};
use super::state_action::StateActionProcessor;
use super::types::{
    EmbeddingAction, EmbeddingExperience, EmbeddingState, ExplorationConfig, RLEmbeddingConfig,
    RLTrainingStats,
};

/// RL training manager
pub struct TrainingManager;

impl TrainingManager {
    /// Sample batch of experiences for training
    pub fn sample_experience_batch(
        experience_buffer: &[EmbeddingExperience],
        batch_size: usize,
    ) -> RLEmbeddingResult<Vec<EmbeddingExperience>> {
        let mut rng = ChaCha8Rng::seed_from_u64(thread_rng().gen());
        let mut batch = Vec::new();

        for _ in 0..batch_size {
            let idx = rng.gen_range(0..experience_buffer.len());
            batch.push(experience_buffer[idx].clone());
        }

        Ok(batch)
    }

    /// Train DQN on batch
    pub fn train_dqn_batch(
        dqn: &mut EmbeddingDQN,
        batch: &[EmbeddingExperience],
        discount_factor: f64,
    ) -> RLEmbeddingResult<f64> {
        let mut total_loss = 0.0;

        for experience in batch {
            let state_vector = StateActionProcessor::state_to_vector(&experience.state)?;
            let next_state_vector = StateActionProcessor::state_to_vector(&experience.next_state)?;

            // Q-learning update
            let current_q_values = dqn.q_network.forward(&state_vector)?;
            let next_q_values = dqn.target_network.forward(&next_state_vector)?;

            let max_next_q = next_q_values
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let target_q = (discount_factor * max_next_q)
                .mul_add(if experience.done { 0.0 } else { 1.0 }, experience.reward);

            // For simplicity, assume action maps to index 0
            let prediction_error = target_q - current_q_values.get(0).unwrap_or(&0.0);
            total_loss += prediction_error.powi(2);

            // Simplified backpropagation (in practice, would implement proper gradients)
            // dqn.q_network.backward(...)?;
        }

        Ok(total_loss / batch.len() as f64)
    }

    /// Train policy network on batch
    pub fn train_policy_batch(
        policy_network: &mut EmbeddingPolicyNetwork,
        batch: &[EmbeddingExperience],
    ) -> RLEmbeddingResult<f64> {
        let mut total_loss = 0.0;

        for experience in batch {
            let state_vector = StateActionProcessor::state_to_vector(&experience.state)?;

            // Actor-critic update
            let state_value = policy_network.critic_network.forward(&state_vector)?;
            let advantage = experience.reward - state_value.get(0).unwrap_or(&0.0);

            // Policy gradient loss
            let policy_loss = -advantage.ln().abs(); // Simplified
            total_loss += policy_loss;

            // Simplified backpropagation
            // policy_network.actor_network.backward(...)?;
            // policy_network.critic_network.backward(...)?;
        }

        Ok(total_loss / batch.len() as f64)
    }

    /// Update target networks
    pub fn update_target_networks(dqn: &mut EmbeddingDQN) -> RLEmbeddingResult<()> {
        // Soft update: target = tau * main + (1 - tau) * target
        let tau = 0.005; // Soft update coefficient

        // For simplicity, just copy the networks (in practice, would do soft update)
        dqn.target_network = dqn.q_network.clone();

        Ok(())
    }

    /// Get current exploration epsilon
    #[must_use]
    pub fn get_current_epsilon(
        training_stats: &RLTrainingStats,
        exploration_config: &ExplorationConfig,
    ) -> f64 {
        let steps_done = training_stats.episode_rewards.len();
        let epsilon_decay_steps = exploration_config.epsilon_decay_steps;

        if steps_done >= epsilon_decay_steps {
            exploration_config.final_epsilon
        } else {
            let decay_ratio = steps_done as f64 / epsilon_decay_steps as f64;
            exploration_config.initial_epsilon.mul_add(
                1.0 - decay_ratio,
                exploration_config.final_epsilon * decay_ratio,
            )
        }
    }

    /// Store experience in replay buffer
    pub fn store_experience(
        experience_buffer: &mut Vec<EmbeddingExperience>,
        experience: EmbeddingExperience,
        buffer_size: usize,
    ) {
        experience_buffer.push(experience);

        // Maintain buffer size
        if experience_buffer.len() > buffer_size {
            experience_buffer.remove(0);
        }
    }

    /// Select action using current policy
    pub fn select_action(
        dqn: &EmbeddingDQN,
        state: &EmbeddingState,
        epsilon: f64,
    ) -> RLEmbeddingResult<EmbeddingAction> {
        // Use epsilon-greedy for DQN
        let mut rng = ChaCha8Rng::seed_from_u64(thread_rng().gen());

        if rng.gen::<f64>() < epsilon {
            // Random exploration
            Ok(EmbeddingAction::Discrete(
                StateActionProcessor::sample_random_action(state)?,
            ))
        } else {
            // Greedy action from DQN
            let state_vector = StateActionProcessor::state_to_vector(state)?;
            let q_values = dqn.q_network.forward(&state_vector)?;
            let best_action_idx = q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx);

            Ok(EmbeddingAction::Discrete(
                StateActionProcessor::action_index_to_action(best_action_idx, state)?,
            ))
        }
    }

    /// Train the RL networks
    pub fn train_networks(
        dqn: &mut EmbeddingDQN,
        policy_network: &mut EmbeddingPolicyNetwork,
        experience_buffer: &[EmbeddingExperience],
        training_stats: &mut RLTrainingStats,
        config: &RLEmbeddingConfig,
        num_epochs: usize,
    ) -> RLEmbeddingResult<()> {
        println!("Training RL embedding optimizer for {num_epochs} epochs");

        for epoch in 0..num_epochs {
            if experience_buffer.len() < config.batch_size {
                continue; // Not enough experiences yet
            }

            let start_time = Instant::now();

            // Sample batch from experience buffer
            let batch = Self::sample_experience_batch(experience_buffer, config.batch_size)?;

            // Train DQN
            let dqn_loss = Self::train_dqn_batch(dqn, &batch, config.discount_factor)?;

            // Train policy network
            let policy_loss = Self::train_policy_batch(policy_network, &batch)?;

            // Update target networks periodically
            if epoch % config.target_update_frequency == 0 {
                Self::update_target_networks(dqn)?;
            }

            // Update training statistics
            training_stats.loss_history.push(dqn_loss + policy_loss);
            training_stats
                .exploration_history
                .push(Self::get_current_epsilon(
                    training_stats,
                    &config.exploration_config,
                ));

            let epoch_time = start_time.elapsed();

            if epoch % 100 == 0 {
                println!(
                    "Epoch {epoch}: DQN Loss = {dqn_loss:.6}, Policy Loss = {policy_loss:.6}, Time = {epoch_time:?}"
                );
            }
        }

        Ok(())
    }
}
