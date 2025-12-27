//! Quantum Neural Architecture Search (NAS)
//!
//! This module implements automated search algorithms for finding optimal quantum
//! circuit architectures, including evolutionary algorithms, reinforcement learning-based
//! search, and gradient-based methods adapted for quantum circuits.

use crate::autodiff::optimizers::Optimizer;
use crate::error::{MLError, Result};
use crate::optimization::OptimizationMethod;
use crate::qnn::{QNNLayerType, QuantumNeuralNetwork};
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::gate::{
    single::{RotationX, RotationY, RotationZ},
    GateOp,
};
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};
use scirs2_core::random::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Search strategy for quantum architecture search
#[derive(Debug, Clone, Copy)]
pub enum SearchStrategy {
    /// Evolutionary algorithm search
    Evolutionary {
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        elitism_ratio: f64,
    },

    /// Reinforcement learning-based search
    ReinforcementLearning {
        agent_type: RLAgentType,
        exploration_rate: f64,
        learning_rate: f64,
    },

    /// Random search baseline
    Random { num_samples: usize },

    /// Bayesian optimization
    BayesianOptimization {
        acquisition_function: AcquisitionFunction,
        num_initial_points: usize,
    },

    /// Differentiable architecture search
    DARTS {
        learning_rate: f64,
        weight_decay: f64,
    },
}

/// RL agent types for NAS
#[derive(Debug, Clone, Copy)]
pub enum RLAgentType {
    /// Deep Q-Network
    DQN,
    /// Policy Gradient
    PolicyGradient,
    /// Actor-Critic
    ActorCritic,
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    ExpectedImprovement,
    /// Upper Confidence Bound
    UpperConfidenceBound,
    /// Probability of Improvement
    ProbabilityOfImprovement,
}

/// Architecture search space definition
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Available layer types
    pub layer_types: Vec<QNNLayerType>,

    /// Minimum/maximum depth
    pub depth_range: (usize, usize),

    /// Qubit count constraints
    pub qubit_constraints: QubitConstraints,

    /// Parameter ranges for variational layers
    pub param_ranges: HashMap<String, (usize, usize)>,

    /// Connectivity patterns
    pub connectivity_patterns: Vec<String>,

    /// Measurement basis options
    pub measurement_bases: Vec<String>,
}

/// Qubit constraints for architecture search
#[derive(Debug, Clone)]
pub struct QubitConstraints {
    /// Minimum number of qubits
    pub min_qubits: usize,

    /// Maximum number of qubits
    pub max_qubits: usize,

    /// Hardware topology constraints
    pub topology: Option<QuantumTopology>,
}

/// Quantum hardware topology
#[derive(Debug, Clone)]
pub enum QuantumTopology {
    /// Linear chain topology
    Linear,
    /// Ring/circular topology
    Ring,
    /// 2D grid topology
    Grid { width: usize, height: usize },
    /// Complete graph (all-to-all)
    Complete,
    /// Custom connectivity
    Custom { edges: Vec<(usize, usize)> },
}

/// Architecture candidate with evaluation metrics
#[derive(Debug, Clone)]
pub struct ArchitectureCandidate {
    /// Unique identifier
    pub id: String,

    /// Layer configuration
    pub layers: Vec<QNNLayerType>,

    /// Number of qubits
    pub num_qubits: usize,

    /// Performance metrics
    pub metrics: ArchitectureMetrics,

    /// Architecture properties
    pub properties: ArchitectureProperties,
}

/// Performance metrics for architecture evaluation
#[derive(Debug, Clone)]
pub struct ArchitectureMetrics {
    /// Validation accuracy
    pub accuracy: Option<f64>,

    /// Training loss
    pub loss: Option<f64>,

    /// Circuit depth
    pub circuit_depth: usize,

    /// Parameter count
    pub parameter_count: usize,

    /// Training time
    pub training_time: Option<f64>,

    /// Memory usage
    pub memory_usage: Option<usize>,

    /// Hardware efficiency score
    pub hardware_efficiency: Option<f64>,
}

/// Architecture properties for analysis
#[derive(Debug, Clone)]
pub struct ArchitectureProperties {
    /// Expressivity measure
    pub expressivity: Option<f64>,

    /// Entanglement capability
    pub entanglement_capability: Option<f64>,

    /// Gradient variance
    pub gradient_variance: Option<f64>,

    /// Barren plateau susceptibility
    pub barren_plateau_score: Option<f64>,

    /// Noise resilience
    pub noise_resilience: Option<f64>,
}

/// Main quantum neural architecture search engine
pub struct QuantumNAS {
    /// Search strategy
    strategy: SearchStrategy,

    /// Search space definition
    search_space: SearchSpace,

    /// Evaluation dataset
    eval_data: Option<(Array2<f64>, Array1<usize>)>,

    /// Best architectures found
    best_architectures: Vec<ArchitectureCandidate>,

    /// Search history
    search_history: Vec<ArchitectureCandidate>,

    /// Current generation (for evolutionary)
    current_generation: usize,

    /// RL agent state (for RL-based search)
    rl_state: Option<RLSearchState>,

    /// Pareto front for multi-objective optimization
    pareto_front: Vec<ArchitectureCandidate>,
}

/// RL agent state for reinforcement learning NAS
#[derive(Debug, Clone)]
pub struct RLSearchState {
    /// Q-values for actions
    q_values: HashMap<String, f64>,

    /// Policy parameters
    policy_params: Array1<f64>,

    /// Experience replay buffer
    replay_buffer: Vec<RLExperience>,

    /// Current state representation
    current_state: Array1<f64>,
}

/// Experience for RL training
#[derive(Debug, Clone)]
pub struct RLExperience {
    /// State before action
    pub state: Array1<f64>,

    /// Action taken
    pub action: ArchitectureAction,

    /// Reward received
    pub reward: f64,

    /// Next state
    pub next_state: Array1<f64>,

    /// Whether episode ended
    pub done: bool,
}

/// Actions for RL-based architecture search
#[derive(Debug, Clone)]
pub enum ArchitectureAction {
    /// Add a layer
    AddLayer(QNNLayerType),

    /// Remove a layer
    RemoveLayer(usize),

    /// Modify layer parameters
    ModifyLayer(usize, HashMap<String, f64>),

    /// Change connectivity
    ChangeConnectivity(String),

    /// Finish architecture
    Finish,
}

impl QuantumNAS {
    /// Create a new quantum NAS instance
    pub fn new(strategy: SearchStrategy, search_space: SearchSpace) -> Self {
        Self {
            strategy,
            search_space,
            eval_data: None,
            best_architectures: Vec::new(),
            search_history: Vec::new(),
            current_generation: 0,
            rl_state: None,
            pareto_front: Vec::new(),
        }
    }

    /// Set evaluation dataset
    pub fn set_evaluation_data(&mut self, data: Array2<f64>, labels: Array1<usize>) {
        self.eval_data = Some((data, labels));
    }

    /// Search for optimal architectures
    pub fn search(&mut self, max_iterations: usize) -> Result<Vec<ArchitectureCandidate>> {
        println!("Starting quantum neural architecture search...");

        match self.strategy {
            SearchStrategy::Evolutionary { .. } => self.evolutionary_search(max_iterations),
            SearchStrategy::ReinforcementLearning { .. } => self.rl_search(max_iterations),
            SearchStrategy::Random { .. } => self.random_search(max_iterations),
            SearchStrategy::BayesianOptimization { .. } => self.bayesian_search(max_iterations),
            SearchStrategy::DARTS { .. } => self.darts_search(max_iterations),
        }
    }

    /// Evolutionary algorithm search
    fn evolutionary_search(
        &mut self,
        max_generations: usize,
    ) -> Result<Vec<ArchitectureCandidate>> {
        let (population_size, mutation_rate, crossover_rate, elitism_ratio) = match self.strategy {
            SearchStrategy::Evolutionary {
                population_size,
                mutation_rate,
                crossover_rate,
                elitism_ratio,
            } => (
                population_size,
                mutation_rate,
                crossover_rate,
                elitism_ratio,
            ),
            _ => unreachable!(),
        };

        // Initialize population
        let mut population = self.initialize_population(population_size)?;

        for generation in 0..max_generations {
            self.current_generation = generation;

            // Evaluate population
            for candidate in &mut population {
                if candidate.metrics.accuracy.is_none() {
                    self.evaluate_architecture(candidate)?;
                }
            }

            // Sort by fitness
            population.sort_by(|a, b| {
                let fitness_a = self.compute_fitness(a);
                let fitness_b = self.compute_fitness(b);
                fitness_b
                    .partial_cmp(&fitness_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Update best architectures
            self.update_best_architectures(&population);

            // Update Pareto front
            self.update_pareto_front(&population);

            println!(
                "Generation {}: Best fitness = {:.4}",
                generation,
                self.compute_fitness(&population[0])
            );

            // Create next generation
            let elite_count = (population_size as f64 * elitism_ratio) as usize;
            let mut next_generation = population[..elite_count].to_vec();

            while next_generation.len() < population_size {
                // Tournament selection
                let parent1 = self.tournament_selection(&population, 3)?;
                let parent2 = self.tournament_selection(&population, 3)?;

                // Crossover
                let mut offspring = if thread_rng().gen::<f64>() < crossover_rate {
                    self.crossover(&parent1, &parent2)?
                } else {
                    parent1.clone()
                };

                // Mutation
                if thread_rng().gen::<f64>() < mutation_rate {
                    self.mutate(&mut offspring)?;
                }

                next_generation.push(offspring);
            }

            population = next_generation;

            // Add to search history
            self.search_history.extend(population.clone());
        }

        Ok(self.best_architectures.clone())
    }

    /// Reinforcement learning-based search
    fn rl_search(&mut self, max_episodes: usize) -> Result<Vec<ArchitectureCandidate>> {
        let (agent_type, exploration_rate, learning_rate) = match self.strategy {
            SearchStrategy::ReinforcementLearning {
                agent_type,
                exploration_rate,
                learning_rate,
            } => (agent_type, exploration_rate, learning_rate),
            _ => unreachable!(),
        };

        // Initialize RL agent
        self.initialize_rl_agent(agent_type, learning_rate)?;

        for episode in 0..max_episodes {
            let mut current_architecture = self.create_empty_architecture();
            let mut episode_reward = 0.0;
            let mut step = 0;

            loop {
                // Get current state
                let state = self.architecture_to_state(&current_architecture)?;

                // Choose action (epsilon-greedy)
                let action = if thread_rng().gen::<f64>() < exploration_rate {
                    self.sample_random_action(&current_architecture)?
                } else {
                    self.choose_best_action(&state)?
                };

                // Apply action
                let (next_architecture, reward, done) =
                    self.apply_action(&current_architecture, &action)?;

                // Update experience
                let next_state = self.architecture_to_state(&next_architecture)?;
                let experience = RLExperience {
                    state: state.clone(),
                    action: action.clone(),
                    reward,
                    next_state: next_state.clone(),
                    done,
                };

                if let Some(ref mut rl_state) = self.rl_state {
                    rl_state.replay_buffer.push(experience);
                }

                // Train agent
                if step % 10 == 0 {
                    self.train_rl_agent()?;
                }

                episode_reward += reward;
                current_architecture = next_architecture;
                step += 1;

                if done || step > 20 {
                    break;
                }
            }

            // Evaluate final architecture
            let mut final_candidate = current_architecture;
            self.evaluate_architecture(&mut final_candidate)?;
            self.search_history.push(final_candidate.clone());
            self.update_best_architectures(&[final_candidate]);

            if episode % 100 == 0 {
                println!("Episode {}: Reward = {:.4}", episode, episode_reward);
            }
        }

        Ok(self.best_architectures.clone())
    }

    /// Random search baseline
    fn random_search(&mut self, num_samples: usize) -> Result<Vec<ArchitectureCandidate>> {
        for i in 0..num_samples {
            let mut candidate = self.sample_random_architecture()?;
            self.evaluate_architecture(&mut candidate)?;

            self.search_history.push(candidate.clone());
            self.update_best_architectures(&[candidate]);

            if i % 100 == 0 {
                println!("Evaluated {} random architectures", i + 1);
            }
        }

        Ok(self.best_architectures.clone())
    }

    /// Bayesian optimization search
    fn bayesian_search(&mut self, max_iterations: usize) -> Result<Vec<ArchitectureCandidate>> {
        let (acquisition_fn, num_initial) = match self.strategy {
            SearchStrategy::BayesianOptimization {
                acquisition_function,
                num_initial_points,
            } => (acquisition_function, num_initial_points),
            _ => unreachable!(),
        };

        // Initial random sampling
        let mut candidates = Vec::new();
        for _ in 0..num_initial {
            let mut candidate = self.sample_random_architecture()?;
            self.evaluate_architecture(&mut candidate)?;
            candidates.push(candidate);
        }

        // Bayesian optimization loop
        for iteration in num_initial..max_iterations {
            // Fit surrogate model (simplified)
            let surrogate = self.fit_surrogate_model(&candidates)?;

            // Optimize acquisition function
            let next_candidate = self.optimize_acquisition(&surrogate, acquisition_fn)?;

            // Evaluate candidate
            let mut evaluated_candidate = next_candidate;
            self.evaluate_architecture(&mut evaluated_candidate)?;

            candidates.push(evaluated_candidate.clone());
            self.search_history.push(evaluated_candidate.clone());
            self.update_best_architectures(&[evaluated_candidate]);

            if iteration % 50 == 0 {
                let best_acc = self.best_architectures[0].metrics.accuracy.unwrap_or(0.0);
                println!("Iteration {}: Best accuracy = {:.4}", iteration, best_acc);
            }
        }

        Ok(self.best_architectures.clone())
    }

    /// DARTS (Differentiable Architecture Search)
    fn darts_search(&mut self, max_epochs: usize) -> Result<Vec<ArchitectureCandidate>> {
        let (learning_rate, weight_decay) = match self.strategy {
            SearchStrategy::DARTS {
                learning_rate,
                weight_decay,
            } => (learning_rate, weight_decay),
            _ => unreachable!(),
        };

        // Initialize architecture weights (alpha parameters)
        let num_layers = 8; // Fixed depth for DARTS
        let num_ops = self.search_space.layer_types.len();
        let mut alpha = Array2::zeros((num_layers, num_ops));

        // Initialize with uniform distribution
        for i in 0..num_layers {
            for j in 0..num_ops {
                alpha[[i, j]] = 1.0 / num_ops as f64;
            }
        }

        for epoch in 0..max_epochs {
            // Update alpha parameters using gradient descent
            let alpha_grad = self.compute_architecture_gradients(&alpha)?;
            alpha = alpha - learning_rate * &alpha_grad;

            // Apply softmax to alpha
            for i in 0..num_layers {
                let row_sum: f64 = alpha.row(i).iter().map(|x| x.exp()).sum();
                for j in 0..num_ops {
                    alpha[[i, j]] = alpha[[i, j]].exp() / row_sum;
                }
            }

            if epoch % 100 == 0 {
                println!("DARTS epoch {}: Architecture weights updated", epoch);
            }
        }

        // Derive final architecture from learned weights
        let final_architecture = self.derive_architecture_from_weights(&alpha)?;
        let mut candidate = final_architecture;
        self.evaluate_architecture(&mut candidate)?;

        self.search_history.push(candidate.clone());
        self.update_best_architectures(&[candidate]);

        Ok(self.best_architectures.clone())
    }

    /// Initialize random population for evolutionary search
    fn initialize_population(&self, size: usize) -> Result<Vec<ArchitectureCandidate>> {
        let mut population = Vec::new();
        for i in 0..size {
            let candidate = self.sample_random_architecture()?;
            population.push(candidate);
        }
        Ok(population)
    }

    /// Sample a random architecture from search space
    fn sample_random_architecture(&self) -> Result<ArchitectureCandidate> {
        let depth =
            fastrand::usize(self.search_space.depth_range.0..=self.search_space.depth_range.1);
        let num_qubits = fastrand::usize(
            self.search_space.qubit_constraints.min_qubits
                ..=self.search_space.qubit_constraints.max_qubits,
        );

        let mut layers = Vec::new();

        // Add encoding layer
        layers.push(QNNLayerType::EncodingLayer {
            num_features: fastrand::usize(2..8),
        });

        // Add random layers
        for _ in 0..depth {
            let layer_type_idx = fastrand::usize(0..self.search_space.layer_types.len());
            let layer_type = self.search_space.layer_types[layer_type_idx].clone();
            layers.push(layer_type);
        }

        // Add measurement layer
        let basis_idx = fastrand::usize(0..self.search_space.measurement_bases.len());
        layers.push(QNNLayerType::MeasurementLayer {
            measurement_basis: self.search_space.measurement_bases[basis_idx].clone(),
        });

        Ok(ArchitectureCandidate {
            id: format!("arch_{}", fastrand::u64(..)),
            layers,
            num_qubits,
            metrics: ArchitectureMetrics {
                accuracy: None,
                loss: None,
                circuit_depth: 0,
                parameter_count: 0,
                training_time: None,
                memory_usage: None,
                hardware_efficiency: None,
            },
            properties: ArchitectureProperties {
                expressivity: None,
                entanglement_capability: None,
                gradient_variance: None,
                barren_plateau_score: None,
                noise_resilience: None,
            },
        })
    }

    /// Evaluate architecture performance
    fn evaluate_architecture(&self, candidate: &mut ArchitectureCandidate) -> Result<()> {
        // Create QNN from architecture
        let qnn = QuantumNeuralNetwork::new(
            candidate.layers.clone(),
            candidate.num_qubits,
            4, // input_dim
            2, // output_dim
        )?;

        // Compute metrics
        candidate.metrics.parameter_count = qnn.parameters.len();
        candidate.metrics.circuit_depth = self.estimate_circuit_depth(&candidate.layers);

        // Evaluate on dataset if available
        if let Some((data, labels)) = &self.eval_data {
            let (accuracy, loss) = self.evaluate_on_dataset(&qnn, data, labels)?;
            candidate.metrics.accuracy = Some(accuracy);
            candidate.metrics.loss = Some(loss);
        } else {
            // Synthetic evaluation
            candidate.metrics.accuracy = Some(0.5 + 0.4 * thread_rng().gen::<f64>());
            candidate.metrics.loss = Some(0.5 + 0.5 * thread_rng().gen::<f64>());
        }

        // Compute architecture properties
        self.compute_architecture_properties(candidate)?;

        Ok(())
    }

    /// Compute fitness score for evolutionary algorithm
    fn compute_fitness(&self, candidate: &ArchitectureCandidate) -> f64 {
        let accuracy = candidate.metrics.accuracy.unwrap_or(0.0);
        let param_penalty = candidate.metrics.parameter_count as f64 / 1000.0;
        let depth_penalty = candidate.metrics.circuit_depth as f64 / 100.0;

        // Multi-objective fitness
        accuracy - 0.1 * param_penalty - 0.05 * depth_penalty
    }

    /// Tournament selection for evolutionary algorithm
    fn tournament_selection(
        &self,
        population: &[ArchitectureCandidate],
        tournament_size: usize,
    ) -> Result<ArchitectureCandidate> {
        let mut best = None;
        let mut best_fitness = f64::NEG_INFINITY;

        for _ in 0..tournament_size {
            let idx = fastrand::usize(0..population.len());
            let candidate = &population[idx];
            let fitness = self.compute_fitness(candidate);

            if fitness > best_fitness {
                best_fitness = fitness;
                best = Some(candidate.clone());
            }
        }

        best.ok_or_else(|| {
            MLError::MLOperationError("Tournament selection failed: no candidates".to_string())
        })
    }

    /// Crossover operation for evolutionary algorithm
    fn crossover(
        &self,
        parent1: &ArchitectureCandidate,
        parent2: &ArchitectureCandidate,
    ) -> Result<ArchitectureCandidate> {
        // Simple layer-wise crossover
        let mut child_layers = Vec::new();
        let max_len = parent1.layers.len().max(parent2.layers.len());

        for i in 0..max_len {
            if thread_rng().gen::<bool>() {
                if i < parent1.layers.len() {
                    child_layers.push(parent1.layers[i].clone());
                }
            } else {
                if i < parent2.layers.len() {
                    child_layers.push(parent2.layers[i].clone());
                }
            }
        }

        let num_qubits = if thread_rng().gen::<bool>() {
            parent1.num_qubits
        } else {
            parent2.num_qubits
        };

        Ok(ArchitectureCandidate {
            id: format!("crossover_{}", fastrand::u64(..)),
            layers: child_layers,
            num_qubits,
            metrics: ArchitectureMetrics {
                accuracy: None,
                loss: None,
                circuit_depth: 0,
                parameter_count: 0,
                training_time: None,
                memory_usage: None,
                hardware_efficiency: None,
            },
            properties: ArchitectureProperties {
                expressivity: None,
                entanglement_capability: None,
                gradient_variance: None,
                barren_plateau_score: None,
                noise_resilience: None,
            },
        })
    }

    /// Mutation operation for evolutionary algorithm
    fn mutate(&self, candidate: &mut ArchitectureCandidate) -> Result<()> {
        let mutation_type = fastrand::usize(0..4);

        match mutation_type {
            0 => {
                // Add layer
                if candidate.layers.len() < self.search_space.depth_range.1 + 2 {
                    let layer_idx = fastrand::usize(0..self.search_space.layer_types.len());
                    let new_layer = self.search_space.layer_types[layer_idx].clone();
                    let insert_pos = fastrand::usize(1..candidate.layers.len()); // Skip encoding layer
                    candidate.layers.insert(insert_pos, new_layer);
                }
            }
            1 => {
                // Remove layer
                if candidate.layers.len() > 3 {
                    // Keep encoding and measurement
                    let remove_pos = fastrand::usize(1..candidate.layers.len() - 1);
                    candidate.layers.remove(remove_pos);
                }
            }
            2 => {
                // Modify layer
                if candidate.layers.len() > 2 {
                    let layer_idx = fastrand::usize(1..candidate.layers.len() - 1);
                    let new_layer_idx = fastrand::usize(0..self.search_space.layer_types.len());
                    candidate.layers[layer_idx] =
                        self.search_space.layer_types[new_layer_idx].clone();
                }
            }
            3 => {
                // Change qubit count
                candidate.num_qubits = fastrand::usize(
                    self.search_space.qubit_constraints.min_qubits
                        ..=self.search_space.qubit_constraints.max_qubits,
                );
            }
            _ => {}
        }

        // Reset metrics
        candidate.metrics.accuracy = None;
        candidate.metrics.loss = None;

        Ok(())
    }

    /// Update best architectures list
    fn update_best_architectures(&mut self, candidates: &[ArchitectureCandidate]) {
        for candidate in candidates {
            if candidate.metrics.accuracy.is_some() {
                self.best_architectures.push(candidate.clone());
            }
        }

        // Compute fitness scores for sorting
        let mut fitness_scores: Vec<(usize, f64)> = self
            .best_architectures
            .iter()
            .enumerate()
            .map(|(i, arch)| (i, self.compute_fitness(arch)))
            .collect();

        // Sort by fitness (descending)
        fitness_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Reorder architectures based on fitness
        let sorted_architectures: Vec<_> = fitness_scores
            .into_iter()
            .take(10)
            .map(|(i, _)| self.best_architectures[i].clone())
            .collect();

        self.best_architectures = sorted_architectures;
    }

    /// Update Pareto front for multi-objective optimization
    fn update_pareto_front(&mut self, candidates: &[ArchitectureCandidate]) {
        for candidate in candidates {
            let is_dominated = self
                .pareto_front
                .iter()
                .any(|other| self.dominates(other, candidate));

            if !is_dominated {
                // Find indices of dominated solutions
                let mut to_remove = Vec::new();
                for (i, other) in self.pareto_front.iter().enumerate() {
                    if self.dominates(candidate, other) {
                        to_remove.push(i);
                    }
                }

                // Remove dominated solutions (in reverse order to maintain indices)
                for &i in to_remove.iter().rev() {
                    self.pareto_front.remove(i);
                }

                // Add new candidate
                self.pareto_front.push(candidate.clone());
            }
        }
    }

    /// Check if one candidate dominates another (Pareto dominance)
    fn dominates(&self, a: &ArchitectureCandidate, b: &ArchitectureCandidate) -> bool {
        let acc_a = a.metrics.accuracy.unwrap_or(0.0);
        let acc_b = b.metrics.accuracy.unwrap_or(0.0);
        let params_a = a.metrics.parameter_count as f64;
        let params_b = b.metrics.parameter_count as f64;

        (acc_a >= acc_b && params_a <= params_b) && (acc_a > acc_b || params_a < params_b)
    }

    /// Estimate circuit depth from layers
    fn estimate_circuit_depth(&self, layers: &[QNNLayerType]) -> usize {
        layers
            .iter()
            .map(|layer| match layer {
                QNNLayerType::EncodingLayer { .. } => 1,
                QNNLayerType::VariationalLayer { num_params } => num_params / 3, // Roughly gates per qubit
                QNNLayerType::EntanglementLayer { .. } => 1,
                QNNLayerType::MeasurementLayer { .. } => 1,
            })
            .sum()
    }

    /// Evaluate QNN on dataset
    fn evaluate_on_dataset(
        &self,
        qnn: &QuantumNeuralNetwork,
        data: &Array2<f64>,
        labels: &Array1<usize>,
    ) -> Result<(f64, f64)> {
        // Simplified evaluation - would use actual training/validation
        let accuracy = 0.6 + 0.3 * thread_rng().gen::<f64>();
        let loss = 0.2 + 0.8 * thread_rng().gen::<f64>();
        Ok((accuracy, loss))
    }

    /// Compute architecture properties
    fn compute_architecture_properties(&self, candidate: &mut ArchitectureCandidate) -> Result<()> {
        // Estimate expressivity based on parameter count and depth
        let expressivity = (candidate.metrics.parameter_count as f64).ln()
            * (candidate.metrics.circuit_depth as f64).sqrt()
            / 100.0;
        candidate.properties.expressivity = Some(expressivity.min(1.0));

        // Estimate entanglement capability
        let entanglement_layers = candidate
            .layers
            .iter()
            .filter(|layer| matches!(layer, QNNLayerType::EntanglementLayer { .. }))
            .count();
        candidate.properties.entanglement_capability =
            Some((entanglement_layers as f64 / candidate.layers.len() as f64).min(1.0));

        // Placeholder values for other properties
        candidate.properties.gradient_variance = Some(0.1 + 0.3 * thread_rng().gen::<f64>());
        candidate.properties.barren_plateau_score = Some(0.2 + 0.6 * thread_rng().gen::<f64>());
        candidate.properties.noise_resilience = Some(0.3 + 0.4 * thread_rng().gen::<f64>());

        Ok(())
    }

    /// Initialize RL agent
    fn initialize_rl_agent(&mut self, agent_type: RLAgentType, learning_rate: f64) -> Result<()> {
        let state_dim = 64; // State representation dimension

        self.rl_state = Some(RLSearchState {
            q_values: HashMap::new(),
            policy_params: Array1::zeros(state_dim),
            replay_buffer: Vec::new(),
            current_state: Array1::zeros(state_dim),
        });

        Ok(())
    }

    /// Convert architecture to state representation
    fn architecture_to_state(&self, arch: &ArchitectureCandidate) -> Result<Array1<f64>> {
        let mut state = Array1::zeros(64);

        // Encode architecture features
        state[0] = arch.layers.len() as f64 / 20.0; // Normalized depth
        state[1] = arch.num_qubits as f64 / 16.0; // Normalized qubit count

        // Encode layer types
        for (i, layer) in arch.layers.iter().enumerate().take(30) {
            let layer_code = match layer {
                QNNLayerType::EncodingLayer { .. } => 0.1,
                QNNLayerType::VariationalLayer { .. } => 0.3,
                QNNLayerType::EntanglementLayer { .. } => 0.5,
                QNNLayerType::MeasurementLayer { .. } => 0.7,
            };
            state[2 + i] = layer_code;
        }

        Ok(state)
    }

    /// Sample random action for RL
    fn sample_random_action(&self, arch: &ArchitectureCandidate) -> Result<ArchitectureAction> {
        let action_type = fastrand::usize(0..5);

        match action_type {
            0 => {
                let layer_idx = fastrand::usize(0..self.search_space.layer_types.len());
                Ok(ArchitectureAction::AddLayer(
                    self.search_space.layer_types[layer_idx].clone(),
                ))
            }
            1 => {
                if arch.layers.len() > 3 {
                    let layer_idx = fastrand::usize(1..arch.layers.len() - 1);
                    Ok(ArchitectureAction::RemoveLayer(layer_idx))
                } else {
                    self.sample_random_action(arch)
                }
            }
            2 => {
                let layer_idx = fastrand::usize(0..arch.layers.len());
                Ok(ArchitectureAction::ModifyLayer(layer_idx, HashMap::new()))
            }
            3 => {
                let conn_idx = fastrand::usize(0..self.search_space.connectivity_patterns.len());
                Ok(ArchitectureAction::ChangeConnectivity(
                    self.search_space.connectivity_patterns[conn_idx].clone(),
                ))
            }
            _ => Ok(ArchitectureAction::Finish),
        }
    }

    /// Choose best action using RL policy
    fn choose_best_action(&self, state: &Array1<f64>) -> Result<ArchitectureAction> {
        // Placeholder - would use trained policy
        Ok(ArchitectureAction::Finish)
    }

    /// Apply action to architecture
    fn apply_action(
        &self,
        arch: &ArchitectureCandidate,
        action: &ArchitectureAction,
    ) -> Result<(ArchitectureCandidate, f64, bool)> {
        let mut new_arch = arch.clone();
        let mut reward = 0.0;
        let mut done = false;

        match action {
            ArchitectureAction::AddLayer(layer) => {
                if new_arch.layers.len() < self.search_space.depth_range.1 + 2 {
                    let insert_pos = fastrand::usize(1..new_arch.layers.len());
                    new_arch.layers.insert(insert_pos, layer.clone());
                    reward = 0.1;
                } else {
                    reward = -0.1;
                }
            }
            ArchitectureAction::RemoveLayer(idx) => {
                if new_arch.layers.len() > 3 && *idx < new_arch.layers.len() {
                    new_arch.layers.remove(*idx);
                    reward = 0.05;
                } else {
                    reward = -0.1;
                }
            }
            ArchitectureAction::Finish => {
                done = true;
                reward = 1.0; // Will be replaced by actual evaluation
            }
            _ => {
                reward = 0.0;
            }
        }

        new_arch.id = format!("rl_{}", fastrand::u64(..));
        Ok((new_arch, reward, done))
    }

    /// Train RL agent
    fn train_rl_agent(&mut self) -> Result<()> {
        // Placeholder for RL training
        Ok(())
    }

    /// Create empty architecture
    fn create_empty_architecture(&self) -> ArchitectureCandidate {
        ArchitectureCandidate {
            id: format!("empty_{}", fastrand::u64(..)),
            layers: vec![
                QNNLayerType::EncodingLayer { num_features: 4 },
                QNNLayerType::MeasurementLayer {
                    measurement_basis: "computational".to_string(),
                },
            ],
            num_qubits: 4,
            metrics: ArchitectureMetrics {
                accuracy: None,
                loss: None,
                circuit_depth: 0,
                parameter_count: 0,
                training_time: None,
                memory_usage: None,
                hardware_efficiency: None,
            },
            properties: ArchitectureProperties {
                expressivity: None,
                entanglement_capability: None,
                gradient_variance: None,
                barren_plateau_score: None,
                noise_resilience: None,
            },
        }
    }

    /// Fit surrogate model for Bayesian optimization
    fn fit_surrogate_model(&self, candidates: &[ArchitectureCandidate]) -> Result<SurrogateModel> {
        // Placeholder for surrogate model
        Ok(SurrogateModel {
            mean_prediction: 0.7,
            uncertainty: 0.1,
        })
    }

    /// Optimize acquisition function
    fn optimize_acquisition(
        &self,
        surrogate: &SurrogateModel,
        acquisition_fn: AcquisitionFunction,
    ) -> Result<ArchitectureCandidate> {
        // Placeholder - would optimize acquisition function
        self.sample_random_architecture()
    }

    /// Compute architecture gradients for DARTS
    fn compute_architecture_gradients(&self, alpha: &Array2<f64>) -> Result<Array2<f64>> {
        // Placeholder for architecture gradient computation
        Ok(Array2::zeros(alpha.raw_dim()))
    }

    /// Derive final architecture from DARTS weights
    fn derive_architecture_from_weights(
        &self,
        alpha: &Array2<f64>,
    ) -> Result<ArchitectureCandidate> {
        let mut layers = vec![QNNLayerType::EncodingLayer { num_features: 4 }];

        for i in 0..alpha.nrows() {
            // Choose layer with highest weight
            let mut best_op = 0;
            let mut best_weight = alpha[[i, 0]];

            for j in 1..alpha.ncols() {
                if alpha[[i, j]] > best_weight {
                    best_weight = alpha[[i, j]];
                    best_op = j;
                }
            }

            if best_op < self.search_space.layer_types.len() {
                layers.push(self.search_space.layer_types[best_op].clone());
            }
        }

        layers.push(QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        });

        Ok(ArchitectureCandidate {
            id: format!("darts_{}", fastrand::u64(..)),
            layers,
            num_qubits: 4,
            metrics: ArchitectureMetrics {
                accuracy: None,
                loss: None,
                circuit_depth: 0,
                parameter_count: 0,
                training_time: None,
                memory_usage: None,
                hardware_efficiency: None,
            },
            properties: ArchitectureProperties {
                expressivity: None,
                entanglement_capability: None,
                gradient_variance: None,
                barren_plateau_score: None,
                noise_resilience: None,
            },
        })
    }

    /// Get search results summary
    pub fn get_search_summary(&self) -> SearchSummary {
        SearchSummary {
            total_architectures_evaluated: self.search_history.len(),
            best_architecture: self.best_architectures.first().cloned(),
            pareto_front_size: self.pareto_front.len(),
            search_generations: self.current_generation,
        }
    }

    /// Get Pareto front
    pub fn get_pareto_front(&self) -> &[ArchitectureCandidate] {
        &self.pareto_front
    }
}

/// Surrogate model for Bayesian optimization
#[derive(Debug, Clone)]
pub struct SurrogateModel {
    pub mean_prediction: f64,
    pub uncertainty: f64,
}

/// Search results summary
#[derive(Debug, Clone)]
pub struct SearchSummary {
    pub total_architectures_evaluated: usize,
    pub best_architecture: Option<ArchitectureCandidate>,
    pub pareto_front_size: usize,
    pub search_generations: usize,
}

impl fmt::Display for ArchitectureCandidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Architecture {} (layers: {}, qubits: {}, accuracy: {:.3})",
            self.id,
            self.layers.len(),
            self.num_qubits,
            self.metrics.accuracy.unwrap_or(0.0)
        )
    }
}

/// Helper function to create default search space
pub fn create_default_search_space() -> SearchSpace {
    SearchSpace {
        layer_types: vec![
            QNNLayerType::VariationalLayer { num_params: 6 },
            QNNLayerType::VariationalLayer { num_params: 9 },
            QNNLayerType::VariationalLayer { num_params: 12 },
            QNNLayerType::EntanglementLayer {
                connectivity: "circular".to_string(),
            },
            QNNLayerType::EntanglementLayer {
                connectivity: "full".to_string(),
            },
        ],
        depth_range: (2, 8),
        qubit_constraints: QubitConstraints {
            min_qubits: 3,
            max_qubits: 8,
            topology: Some(QuantumTopology::Complete),
        },
        param_ranges: vec![
            ("variational_params".to_string(), (3, 15)),
            ("encoding_features".to_string(), (2, 8)),
        ]
        .into_iter()
        .collect(),
        connectivity_patterns: vec![
            "linear".to_string(),
            "circular".to_string(),
            "full".to_string(),
        ],
        measurement_bases: vec![
            "computational".to_string(),
            "Pauli-Z".to_string(),
            "Pauli-X".to_string(),
            "Pauli-Y".to_string(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_space_creation() {
        let search_space = create_default_search_space();
        assert!(search_space.layer_types.len() > 0);
        assert!(search_space.depth_range.0 < search_space.depth_range.1);
        assert!(
            search_space.qubit_constraints.min_qubits <= search_space.qubit_constraints.max_qubits
        );
    }

    #[test]
    fn test_nas_initialization() {
        let search_space = create_default_search_space();
        let strategy = SearchStrategy::Random { num_samples: 10 };
        let nas = QuantumNAS::new(strategy, search_space);

        assert_eq!(nas.current_generation, 0);
        assert_eq!(nas.best_architectures.len(), 0);
    }

    #[test]
    fn test_random_architecture_sampling() {
        let search_space = create_default_search_space();
        let strategy = SearchStrategy::Random { num_samples: 10 };
        let nas = QuantumNAS::new(strategy, search_space);

        let arch = nas
            .sample_random_architecture()
            .expect("Random architecture sampling should succeed");
        assert!(arch.layers.len() >= 2); // At least encoding and measurement
        assert!(arch.num_qubits >= nas.search_space.qubit_constraints.min_qubits);
        assert!(arch.num_qubits <= nas.search_space.qubit_constraints.max_qubits);
    }

    #[test]
    fn test_fitness_computation() {
        let search_space = create_default_search_space();
        let strategy = SearchStrategy::Random { num_samples: 10 };
        let nas = QuantumNAS::new(strategy, search_space);

        let mut arch = nas
            .sample_random_architecture()
            .expect("Random architecture sampling should succeed");
        arch.metrics.accuracy = Some(0.8);
        arch.metrics.parameter_count = 50;
        arch.metrics.circuit_depth = 10;

        let fitness = nas.compute_fitness(&arch);
        assert!(fitness > 0.0);
    }

    #[test]
    fn test_architecture_mutation() {
        let search_space = create_default_search_space();
        let strategy = SearchStrategy::Evolutionary {
            population_size: 10,
            mutation_rate: 0.1,
            crossover_rate: 0.7,
            elitism_ratio: 0.1,
        };
        let nas = QuantumNAS::new(strategy, search_space);

        let mut arch = nas
            .sample_random_architecture()
            .expect("Random architecture sampling should succeed");
        let original_layers = arch.layers.len();

        nas.mutate(&mut arch).expect("Mutation should succeed");

        // Architecture should still be valid
        assert!(arch.layers.len() >= 2);
        assert!(arch.num_qubits >= nas.search_space.qubit_constraints.min_qubits);
    }
}
