//! Core embedding optimization logic for RL embedding optimizer

use std::collections::HashMap;
use std::time::Instant;

use super::error::{RLEmbeddingError, RLEmbeddingResult};
use super::state_action::StateActionProcessor;
use super::types::*;
use crate::embedding::{Embedding, HardwareTopology};
use crate::ising::IsingModel;

/// Embedding evaluation and optimization utilities
pub struct EmbeddingOptimizer;

impl EmbeddingOptimizer {
    /// Generate initial embedding using baseline method
    pub fn generate_initial_embedding(
        problem: &IsingModel,
        hardware: &HardwareTopology,
    ) -> RLEmbeddingResult<Embedding> {
        // Use a simple greedy embedding as baseline
        let mut embedding = Embedding {
            chains: HashMap::new(),
            qubit_to_variable: HashMap::new(),
        };

        // Simple mapping: logical qubit i -> physical qubit i % num_physical
        for logical in 0..problem.num_qubits {
            let physical = logical % StateActionProcessor::get_num_qubits(hardware);
            embedding.chains.insert(logical, vec![physical]);
            // Chain strength would be stored elsewhere if needed
        }

        Ok(embedding)
    }

    /// Calculate reward for action
    pub fn calculate_reward(
        old_embedding: &Embedding,
        new_embedding: &Embedding,
        hardware: &HardwareTopology,
        objective_weights: &ObjectiveWeights,
    ) -> RLEmbeddingResult<f64> {
        let old_quality = Self::evaluate_embedding_quality(old_embedding, hardware)?;
        let new_quality = Self::evaluate_embedding_quality(new_embedding, hardware)?;

        let improvement = new_quality - old_quality;

        // Multi-objective reward
        let mut reward = 0.0;

        // Reward for quality improvement
        reward += improvement * objective_weights.efficiency_weight;

        // Penalty for increased chain lengths
        let old_avg_chain_length = Self::calculate_average_chain_length(old_embedding);
        let new_avg_chain_length = Self::calculate_average_chain_length(new_embedding);
        let chain_penalty =
            (new_avg_chain_length - old_avg_chain_length) * objective_weights.chain_length_weight;
        reward -= chain_penalty;

        // Reward for better hardware utilization
        let old_utilization = Self::calculate_hardware_utilization(old_embedding, hardware);
        let new_utilization = Self::calculate_hardware_utilization(new_embedding, hardware);
        let utilization_reward =
            (new_utilization - old_utilization) * objective_weights.utilization_weight;
        reward += utilization_reward;

        Ok(reward)
    }

    /// Evaluate embedding quality
    pub fn evaluate_embedding_quality(
        embedding: &Embedding,
        hardware: &HardwareTopology,
    ) -> RLEmbeddingResult<f64> {
        let mut quality = 0.0;

        // Factor 1: Chain length penalty
        let avg_chain_length = Self::calculate_average_chain_length(embedding);
        quality -= avg_chain_length * 0.1;

        // Factor 2: Hardware utilization
        let utilization = Self::calculate_hardware_utilization(embedding, hardware);
        quality += utilization * 0.5;

        // Factor 3: Connectivity preservation
        let connectivity = Self::calculate_connectivity_preservation(embedding, hardware);
        quality += connectivity * 0.3;

        // Factor 4: Compactness
        let compactness = Self::calculate_embedding_compactness(embedding, hardware);
        quality += compactness * 0.1;

        Ok(quality)
    }

    /// Calculate average chain length
    pub fn calculate_average_chain_length(embedding: &Embedding) -> f64 {
        if embedding.chains.is_empty() {
            return 0.0;
        }

        let total_length: usize = embedding.chains.values().map(|chain| chain.len()).sum();
        total_length as f64 / embedding.chains.len() as f64
    }

    /// Calculate hardware utilization
    pub fn calculate_hardware_utilization(
        embedding: &Embedding,
        hardware: &HardwareTopology,
    ) -> f64 {
        let used_qubits: std::collections::HashSet<usize> =
            embedding.chains.values().flatten().cloned().collect();

        used_qubits.len() as f64 / StateActionProcessor::get_num_qubits(hardware) as f64
    }

    /// Calculate connectivity preservation
    fn calculate_connectivity_preservation(
        embedding: &Embedding,
        hardware: &HardwareTopology,
    ) -> f64 {
        // Simplified: assume good connectivity preservation for valid embeddings
        if embedding.chains.is_empty() {
            0.0
        } else {
            0.8 // Placeholder
        }
    }

    /// Calculate embedding compactness
    fn calculate_embedding_compactness(embedding: &Embedding, hardware: &HardwareTopology) -> f64 {
        // Simplified compactness measure
        let avg_chain_length = Self::calculate_average_chain_length(embedding);
        1.0 / (1.0 + avg_chain_length)
    }

    /// Update state after action
    pub fn update_state(
        old_state: &EmbeddingState,
        action: &EmbeddingAction,
        new_embedding: &Embedding,
        hardware: &HardwareTopology,
    ) -> RLEmbeddingResult<EmbeddingState> {
        let mut new_state = old_state.clone();

        // Update embedding state
        new_state.embedding_state.logical_to_physical = new_embedding.chains.clone();
        new_state.embedding_state.chain_lengths = new_embedding
            .chains
            .values()
            .map(|chain| chain.len())
            .collect();

        // Update efficiency metrics
        new_state
            .embedding_state
            .efficiency_metrics
            .avg_chain_length = Self::calculate_average_chain_length(new_embedding);
        new_state
            .embedding_state
            .efficiency_metrics
            .max_chain_length = new_embedding
            .chains
            .values()
            .map(|chain| chain.len())
            .max()
            .unwrap_or(0);
        new_state
            .embedding_state
            .efficiency_metrics
            .utilization_ratio = Self::calculate_hardware_utilization(new_embedding, hardware);

        // Update quality score
        new_state.embedding_state.quality_score =
            Self::evaluate_embedding_quality(new_embedding, hardware)?;

        // Update performance history
        new_state
            .performance_history
            .push(new_state.embedding_state.quality_score);
        if new_state.performance_history.len() > 10 {
            new_state.performance_history.remove(0);
        }

        Ok(new_state)
    }

    /// Check if state is terminal
    pub fn is_terminal_state(state: &EmbeddingState) -> bool {
        // Terminal if quality score is very high or improvement has plateaued
        state.embedding_state.quality_score > 0.95
            || (state.performance_history.len() >= 5
                && state
                    .performance_history
                    .windows(2)
                    .all(|w| (w[1] - w[0]).abs() < 0.001))
    }

    /// Calculate problem density
    pub fn calculate_problem_density(problem: &IsingModel) -> f64 {
        let mut num_edges = 0;
        for i in 0..problem.num_qubits {
            for j in (i + 1)..problem.num_qubits {
                if let Ok(coupling) = problem.get_coupling(i, j) {
                    if coupling.abs() > 1e-10 {
                        num_edges += 1;
                    }
                }
            }
        }

        2.0 * num_edges as f64 / (problem.num_qubits * (problem.num_qubits - 1)) as f64
    }

    /// Classify problem type for transfer learning
    pub fn classify_problem_type(problem: &IsingModel) -> String {
        let density = Self::calculate_problem_density(problem);

        if density > 0.8 {
            "dense_random".to_string()
        } else if density < 0.1 {
            "sparse_structured".to_string()
        } else if problem.num_qubits < 50 {
            "small_optimization".to_string()
        } else {
            "large_optimization".to_string()
        }
    }
}
