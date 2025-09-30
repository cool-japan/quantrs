//! State and action representations and feature extraction

use scirs2_core::random::prelude::*;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::random::ChaCha8Rng;
use std::collections::HashMap;

use super::error::{RLEmbeddingError, RLEmbeddingResult};
use super::types::*;
use crate::embedding::{Embedding, HardwareTopology};
use crate::hardware_compilation::HardwareType;
use crate::ising::IsingModel;

/// State and action processing utilities
pub struct StateActionProcessor;

impl StateActionProcessor {
    /// Extract state features from problem and hardware
    pub fn extract_state_features(
        problem: &IsingModel,
        hardware: &HardwareTopology,
    ) -> RLEmbeddingResult<EmbeddingState> {
        // Extract problem graph features
        let problem_features = Self::extract_problem_features(problem)?;

        // Extract hardware features
        let hardware_features = Self::extract_hardware_features(hardware)?;

        // Initialize embedding state
        let embedding_state = CurrentEmbeddingState {
            logical_to_physical: HashMap::new(),
            chain_lengths: Vec::new(),
            efficiency_metrics: EmbeddingEfficiencyMetrics {
                utilization_ratio: 0.0,
                avg_chain_length: 0.0,
                max_chain_length: 0,
                connectivity_preservation: 0.0,
                compactness: 0.0,
            },
            unused_resources: (0..Self::get_num_qubits(hardware)).collect(),
            quality_score: 0.0,
        };

        // Initialize resource utilization
        let resource_utilization = ResourceUtilization {
            qubit_usage: vec![false; Self::get_num_qubits(hardware)],
            coupling_usage: vec![false; Self::get_couplings(hardware).len()],
            memory_usage: 0.0,
            computational_overhead: 0.0,
            energy_consumption: 0.0,
        };

        Ok(EmbeddingState {
            problem_features,
            hardware_features,
            embedding_state,
            performance_history: Vec::new(),
            resource_utilization,
        })
    }

    /// Extract problem-specific features
    fn extract_problem_features(problem: &IsingModel) -> RLEmbeddingResult<ProblemGraphFeatures> {
        let num_vertices = problem.num_qubits;
        let mut num_edges = 0;
        let mut degrees = vec![0; num_vertices];

        // Count edges and compute degrees
        for i in 0..num_vertices {
            for j in (i + 1)..num_vertices {
                if let Ok(coupling) = problem.get_coupling(i, j) {
                    if coupling.abs() > 1e-10 {
                        num_edges += 1;
                        degrees[i] += 1;
                        degrees[j] += 1;
                    }
                }
            }
        }

        let density = 2.0 * num_edges as f64 / (num_vertices * (num_vertices - 1)) as f64;
        let average_degree = degrees.iter().sum::<usize>() as f64 / num_vertices as f64;

        // Compute degree moments
        let degree_mean = average_degree;
        let degree_variance = degrees
            .iter()
            .map(|&d| (d as f64 - degree_mean).powi(2))
            .sum::<f64>()
            / num_vertices as f64;
        let degree_moments = vec![degree_mean, degree_variance, 0.0, 0.0]; // Simplified

        // Simplified clustering coefficient
        let clustering_coefficient = if num_edges > 0 { 0.3 } else { 0.0 }; // Placeholder

        // Simplified diameter
        let diameter = if num_vertices > 1 {
            num_vertices / 2
        } else {
            0
        };

        // Simplified modularity
        let modularity = 0.5; // Placeholder

        // Simplified spectral features
        let spectral_features = SpectralFeatures {
            largest_eigenvalue: average_degree * 1.2,
            second_largest_eigenvalue: average_degree * 0.8,
            spectral_gap: average_degree * 0.4,
            algebraic_connectivity: 0.1,
            eigenvalue_moments: vec![average_degree, degree_variance, 0.0, 0.0],
        };

        Ok(ProblemGraphFeatures {
            num_vertices,
            num_edges,
            density,
            clustering_coefficient,
            average_degree,
            degree_moments,
            diameter,
            modularity,
            spectral_features,
        })
    }

    /// Extract hardware-specific features
    fn extract_hardware_features(
        hardware: &HardwareTopology,
    ) -> RLEmbeddingResult<HardwareFeatures> {
        let connectivity_features = ConnectivityFeatures {
            degree_distribution: vec![4; Self::get_num_qubits(hardware)], // Simplified
            average_connectivity: 4.0, // Placeholder for chimera-like topology
            max_connectivity: 6,
            connectivity_variance: 1.0,
            regularity_measure: 0.8,
        };

        let noise_profile = NoiseProfile {
            coherence_times: vec![100.0; Self::get_num_qubits(hardware)], // microseconds
            error_rates: vec![0.01; Self::get_num_qubits(hardware)],
            crosstalk_matrix: vec![
                vec![0.001; Self::get_num_qubits(hardware)];
                Self::get_num_qubits(hardware)
            ],
            temperature_stability: 0.95,
        };

        let timing_constraints = TimingConstraints {
            min_anneal_time: 1.0, // microseconds
            max_anneal_time: 2000.0,
            readout_time: 5.0,
            programming_time: 10.0,
        };

        let constraints = HardwareConstraints {
            max_chain_length: 20,
            coupling_constraints: vec![1.0; Self::get_couplings(hardware).len()],
            noise_profile,
            timing_constraints,
        };

        let calibration_quality = CalibrationQuality {
            bias_accuracy: 0.99,
            coupling_accuracy: 0.98,
            frequency_drift: 0.001,
            last_calibration: std::time::Duration::from_secs(3600),
        };

        let performance_chars = HardwarePerformanceChars {
            success_probabilities: vec![0.8; Self::get_num_qubits(hardware)],
            energy_scales: vec![1.0; Self::get_num_qubits(hardware)],
            bandwidth_limits: vec![1000.0; Self::get_num_qubits(hardware)],
            calibration_quality,
        };

        Ok(HardwareFeatures {
            hardware_type: HardwareType::DWaveChimera {
                unit_cells: (4, 4),
                cell_size: 4,
            }, // Default
            num_physical_qubits: Self::get_num_qubits(hardware),
            connectivity_features,
            constraints,
            performance_chars,
        })
    }

    /// Convert state to feature vector
    pub fn state_to_vector(state: &EmbeddingState) -> RLEmbeddingResult<Vec<f64>> {
        let mut features = Vec::new();

        // Problem features
        features.push(state.problem_features.num_vertices as f64 / 1000.0);
        features.push(state.problem_features.num_edges as f64 / 1000.0);
        features.push(state.problem_features.density);
        features.push(state.problem_features.clustering_coefficient);
        features.push(state.problem_features.average_degree / 10.0);
        features.extend(
            state
                .problem_features
                .degree_moments
                .iter()
                .map(|&x| x / 10.0),
        );

        // Hardware features
        features.push(state.hardware_features.num_physical_qubits as f64 / 1000.0);
        features.push(
            state
                .hardware_features
                .connectivity_features
                .average_connectivity
                / 10.0,
        );
        features.push(
            state
                .hardware_features
                .connectivity_features
                .regularity_measure,
        );

        // Embedding state features
        features.push(state.embedding_state.efficiency_metrics.utilization_ratio);
        features.push(state.embedding_state.efficiency_metrics.avg_chain_length / 20.0);
        features.push(
            state
                .embedding_state
                .efficiency_metrics
                .connectivity_preservation,
        );
        features.push(state.embedding_state.quality_score);

        // Resource utilization
        features.push(state.resource_utilization.memory_usage);
        features.push(state.resource_utilization.computational_overhead);

        // Performance history (last few values)
        let history_len = state.performance_history.len();
        if history_len > 0 {
            features.push(state.performance_history[history_len - 1]);
            if history_len > 1 {
                features.push(state.performance_history[history_len - 2]);
            } else {
                features.push(0.0);
            }
        } else {
            features.extend(vec![0.0, 0.0]);
        }

        Ok(features)
    }

    /// Sample random action for exploration
    pub fn sample_random_action(
        state: &EmbeddingState,
    ) -> RLEmbeddingResult<DiscreteEmbeddingAction> {
        let mut rng = ChaCha8Rng::seed_from_u64(thread_rng().gen());

        // Randomly select action type
        match rng.gen_range(0..8) {
            0 => {
                let logical_qubit = rng.gen_range(0..state.problem_features.num_vertices);
                let physical_qubit = rng.gen_range(0..state.hardware_features.num_physical_qubits);
                Ok(DiscreteEmbeddingAction::AddToChain {
                    logical_qubit,
                    physical_qubit,
                })
            }
            1 => {
                let logical_qubit = rng.gen_range(0..state.problem_features.num_vertices);
                let physical_qubit = rng.gen_range(0..state.hardware_features.num_physical_qubits);
                Ok(DiscreteEmbeddingAction::RemoveFromChain {
                    logical_qubit,
                    physical_qubit,
                })
            }
            2 => {
                let chain1 = rng.gen_range(0..state.problem_features.num_vertices);
                let chain2 = rng.gen_range(0..state.problem_features.num_vertices);
                Ok(DiscreteEmbeddingAction::SwapChains { chain1, chain2 })
            }
            3 => {
                let logical_qubit = rng.gen_range(0..state.problem_features.num_vertices);
                let new_location =
                    vec![rng.gen_range(0..state.hardware_features.num_physical_qubits)];
                Ok(DiscreteEmbeddingAction::RelocateChain {
                    logical_qubit,
                    new_location,
                })
            }
            4 => {
                let chain1 = rng.gen_range(0..state.problem_features.num_vertices);
                let chain2 = rng.gen_range(0..state.problem_features.num_vertices);
                Ok(DiscreteEmbeddingAction::MergeChains { chain1, chain2 })
            }
            5 => {
                let chain = rng.gen_range(0..state.problem_features.num_vertices);
                let split_point = rng.gen_range(1..5); // Reasonable split point
                Ok(DiscreteEmbeddingAction::SplitChain { chain, split_point })
            }
            6 => {
                let chain = rng.gen_range(0..state.problem_features.num_vertices);
                Ok(DiscreteEmbeddingAction::OptimizeOrdering { chain })
            }
            _ => Ok(DiscreteEmbeddingAction::NoOp),
        }
    }

    /// Convert action index to actual action
    pub fn action_index_to_action(
        action_idx: usize,
        state: &EmbeddingState,
    ) -> RLEmbeddingResult<DiscreteEmbeddingAction> {
        // Map action index to specific action
        match action_idx % 8 {
            0 => Ok(DiscreteEmbeddingAction::AddToChain {
                logical_qubit: action_idx / 8 % state.problem_features.num_vertices,
                physical_qubit: (action_idx / 8 / state.problem_features.num_vertices)
                    % state.hardware_features.num_physical_qubits,
            }),
            1 => Ok(DiscreteEmbeddingAction::RemoveFromChain {
                logical_qubit: action_idx / 8 % state.problem_features.num_vertices,
                physical_qubit: (action_idx / 8 / state.problem_features.num_vertices)
                    % state.hardware_features.num_physical_qubits,
            }),
            2 => Ok(DiscreteEmbeddingAction::SwapChains {
                chain1: action_idx / 8 % state.problem_features.num_vertices,
                chain2: (action_idx / 8 / state.problem_features.num_vertices)
                    % state.problem_features.num_vertices,
            }),
            3 => Ok(DiscreteEmbeddingAction::RelocateChain {
                logical_qubit: action_idx / 8 % state.problem_features.num_vertices,
                new_location: vec![
                    (action_idx / 8 / state.problem_features.num_vertices)
                        % state.hardware_features.num_physical_qubits,
                ],
            }),
            4 => Ok(DiscreteEmbeddingAction::MergeChains {
                chain1: action_idx / 8 % state.problem_features.num_vertices,
                chain2: (action_idx / 8 / state.problem_features.num_vertices)
                    % state.problem_features.num_vertices,
            }),
            5 => Ok(DiscreteEmbeddingAction::SplitChain {
                chain: action_idx / 8 % state.problem_features.num_vertices,
                split_point: (action_idx / 8 / state.problem_features.num_vertices) % 5 + 1,
            }),
            6 => Ok(DiscreteEmbeddingAction::OptimizeOrdering {
                chain: action_idx / 8 % state.problem_features.num_vertices,
            }),
            _ => Ok(DiscreteEmbeddingAction::NoOp),
        }
    }

    /// Apply action to embedding
    pub fn apply_action(
        embedding: &Embedding,
        action: &EmbeddingAction,
    ) -> RLEmbeddingResult<Embedding> {
        let mut new_embedding = embedding.clone();

        match action {
            EmbeddingAction::Discrete(discrete_action) => {
                Self::apply_discrete_action(&mut new_embedding, discrete_action)?;
            }
            EmbeddingAction::Continuous(continuous_action) => {
                Self::apply_continuous_action(&mut new_embedding, continuous_action)?;
            }
            EmbeddingAction::Hybrid {
                discrete,
                continuous,
            } => {
                Self::apply_discrete_action(&mut new_embedding, discrete)?;
                Self::apply_continuous_action(&mut new_embedding, continuous)?;
            }
        }

        Ok(new_embedding)
    }

    /// Apply discrete action to embedding
    fn apply_discrete_action(
        embedding: &mut Embedding,
        action: &DiscreteEmbeddingAction,
    ) -> RLEmbeddingResult<()> {
        match action {
            DiscreteEmbeddingAction::AddToChain {
                logical_qubit,
                physical_qubit,
            } => {
                if let Some(chain) = embedding.chains.get_mut(logical_qubit) {
                    if !chain.contains(physical_qubit) {
                        chain.push(*physical_qubit);
                    }
                } else {
                    embedding
                        .chains
                        .insert(*logical_qubit, vec![*physical_qubit]);
                }
            }
            DiscreteEmbeddingAction::RemoveFromChain {
                logical_qubit,
                physical_qubit,
            } => {
                if let Some(chain) = embedding.chains.get_mut(logical_qubit) {
                    chain.retain(|&x| x != *physical_qubit);
                }
            }
            DiscreteEmbeddingAction::SwapChains { chain1, chain2 } => {
                if let (Some(ch1), Some(ch2)) = (
                    embedding.chains.get(chain1).cloned(),
                    embedding.chains.get(chain2).cloned(),
                ) {
                    embedding.chains.insert(*chain1, ch2);
                    embedding.chains.insert(*chain2, ch1);
                }
            }
            DiscreteEmbeddingAction::RelocateChain {
                logical_qubit,
                new_location,
            } => {
                embedding
                    .chains
                    .insert(*logical_qubit, new_location.clone());
            }
            DiscreteEmbeddingAction::MergeChains { chain1, chain2 } => {
                if let (Some(ch1), Some(ch2)) = (
                    embedding.chains.get(chain1).cloned(),
                    embedding.chains.get(chain2).cloned(),
                ) {
                    let mut merged = ch1;
                    merged.extend(ch2);
                    embedding.chains.insert(*chain1, merged);
                    embedding.chains.remove(chain2);
                }
            }
            DiscreteEmbeddingAction::SplitChain { chain, split_point } => {
                if let Some(original_chain) = embedding.chains.get(chain).cloned() {
                    if original_chain.len() > *split_point {
                        let (part1, part2) = original_chain.split_at(*split_point);
                        embedding.chains.insert(*chain, part1.to_vec());
                        // Find next available logical qubit index for second part
                        let next_logical = embedding.chains.keys().max().unwrap_or(&0) + 1;
                        embedding.chains.insert(next_logical, part2.to_vec());
                    }
                }
            }
            DiscreteEmbeddingAction::OptimizeOrdering { chain } => {
                // Simple optimization: sort physical qubits in chain
                if let Some(ch) = embedding.chains.get_mut(chain) {
                    ch.sort();
                }
            }
            DiscreteEmbeddingAction::NoOp => {
                // Do nothing
            }
        }

        Ok(())
    }

    /// Apply continuous action to embedding
    fn apply_continuous_action(
        embedding: &mut Embedding,
        action: &ContinuousEmbeddingAction,
    ) -> RLEmbeddingResult<()> {
        // Apply chain strength adjustments
        for (logical_qubit, adjustment) in action.chain_strength_adjustments.iter().enumerate() {
            // Chain strength adjustment would be done elsewhere
            if false {
                // Disable this code path
                let mut current_strength = 1.0;
                current_strength = (current_strength + adjustment).max(0.1).min(10.0);
            }
        }

        // Other continuous adjustments would be applied here
        // (coupling modifications, bias adjustments, etc.)

        Ok(())
    }

    /// Calculate the number of qubits from hardware topology
    pub fn get_num_qubits(hardware: &HardwareTopology) -> usize {
        match hardware {
            HardwareTopology::Chimera(m, n, t) => m * n * 2 * t,
            HardwareTopology::Pegasus(n) => 24 * n * (n - 1), // Approximate for Pegasus
            HardwareTopology::Zephyr(n) => 8 * n * n,         // Approximate for Zephyr
            HardwareTopology::Custom => 1000,                 // Default for custom topology
        }
    }

    /// Generate couplings from hardware topology
    pub fn get_couplings(hardware: &HardwareTopology) -> Vec<(usize, usize)> {
        let num_qubits = Self::get_num_qubits(hardware);
        let mut couplings = Vec::new();

        match hardware {
            HardwareTopology::Chimera(m, n, t) => {
                // Simplified Chimera connectivity - just create a sparse graph
                for i in 0..(num_qubits - 1) {
                    if i % 4 < 2 {
                        // Internal shore connections
                        couplings.push((i, i + 1));
                    }
                    if i + 4 < num_qubits {
                        // Inter-cell connections
                        couplings.push((i, i + 4));
                    }
                }
            }
            HardwareTopology::Pegasus(_)
            | HardwareTopology::Zephyr(_)
            | HardwareTopology::Custom => {
                // Simplified: create a lattice-like structure
                for i in 0..(num_qubits - 1) {
                    if i % 10 < 9 {
                        // Row connections
                        couplings.push((i, i + 1));
                    }
                    if i + 10 < num_qubits {
                        // Column connections
                        couplings.push((i, i + 10));
                    }
                }
            }
        }

        couplings
    }

    /// Generate adjacency map from hardware topology
    pub fn get_adjacency(hardware: &HardwareTopology) -> HashMap<usize, Vec<usize>> {
        let couplings = Self::get_couplings(hardware);
        let mut adjacency = HashMap::new();

        for (u, v) in couplings {
            adjacency.entry(u).or_insert_with(Vec::new).push(v);
            adjacency.entry(v).or_insert_with(Vec::new).push(u);
        }

        adjacency
    }
}
