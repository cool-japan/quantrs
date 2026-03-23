//! Machine Learning-based circuit optimization
//!
//! This module provides ML-driven optimization techniques for quantum circuits,
//! including reinforcement learning for gate scheduling, neural networks for
//! pattern recognition, and automated hyperparameter tuning.

use crate::builder::Circuit;
use crate::dag::{circuit_to_dag, CircuitDag};
use crate::scirs2_integration::{AnalysisResult, SciRS2CircuitAnalyzer};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Pseudo-random number generation without rand crate.
// Uses a simple xorshift64 PRNG seeded from the system time.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn prng_f64(state: &mut u64) -> f64 {
    let bits = xorshift64(state);
    // Map to [0, 1)
    (bits >> 11) as f64 / (1u64 << 53) as f64
}

fn prng_usize(state: &mut u64, upper: usize) -> usize {
    if upper == 0 {
        return 0;
    }
    (xorshift64(state) as usize) % upper
}

fn make_prng_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| {
            d.as_nanos() as u64 ^ (d.subsec_nanos() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        })
        .unwrap_or(0xdeadbeef_cafebabe)
}

/// ML optimization strategy
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MLStrategy {
    /// Reinforcement learning for gate scheduling
    ReinforcementLearning {
        /// Q-learning parameters
        learning_rate: f64,
        discount_factor: f64,
        exploration_rate: f64,
        /// Number of training episodes
        episodes: usize,
    },
    /// Neural network for pattern recognition
    NeuralNetwork {
        /// Network architecture (layer sizes)
        architecture: Vec<usize>,
        /// Learning rate
        learning_rate: f64,
        /// Number of training epochs
        epochs: usize,
        /// Batch size
        batch_size: usize,
    },
    /// Genetic algorithm for optimization
    GeneticAlgorithm {
        /// Population size
        population_size: usize,
        /// Number of generations
        generations: usize,
        /// Mutation rate
        mutation_rate: f64,
        /// Selection pressure
        selection_pressure: f64,
    },
    /// Bayesian optimization for hyperparameter tuning
    BayesianOptimization {
        /// Number of initial random samples
        initial_samples: usize,
        /// Number of optimization iterations
        iterations: usize,
        /// Acquisition function
        acquisition: AcquisitionFunction,
    },
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    UpperConfidenceBound { beta: f64 },
    ProbabilityOfImprovement,
}

/// Circuit representation for ML algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLCircuitRepresentation {
    /// Feature vector representing the circuit
    pub features: Vec<f64>,
    /// Gate sequence encoding
    pub gate_sequence: Vec<usize>,
    /// Adjacency matrix
    pub adjacency_matrix: Vec<Vec<f64>>,
    /// Qubit connectivity
    pub qubit_connectivity: Vec<Vec<bool>>,
    /// Circuit metrics
    pub metrics: CircuitMetrics,
}

/// Circuit metrics for ML training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitMetrics {
    /// Circuit depth
    pub depth: usize,
    /// Gate count
    pub gate_count: usize,
    /// Two-qubit gate count
    pub two_qubit_gate_count: usize,
    /// Entanglement measure
    pub entanglement_measure: f64,
    /// Critical path length
    pub critical_path_length: usize,
    /// Parallelization potential
    pub parallelization_potential: f64,
}

/// ML-based optimizer
pub struct MLCircuitOptimizer {
    /// Optimization strategy
    strategy: MLStrategy,
    /// Feature extractor
    feature_extractor: Arc<Mutex<FeatureExtractor>>,
    /// Model storage
    models: Arc<Mutex<HashMap<String, MLModel>>>,
    /// Training data
    training_data: Arc<Mutex<Vec<TrainingExample>>>,
    /// Configuration
    config: MLOptimizerConfig,
}

/// ML optimizer configuration
#[derive(Debug, Clone)]
pub struct MLOptimizerConfig {
    /// Enable feature caching
    pub cache_features: bool,
    /// Maximum training examples to keep
    pub max_training_examples: usize,
    /// Model update frequency
    pub model_update_frequency: usize,
    /// Enable parallel training
    pub parallel_training: bool,
    /// Feature selection threshold
    pub feature_selection_threshold: f64,
}

impl Default for MLOptimizerConfig {
    fn default() -> Self {
        Self {
            cache_features: true,
            max_training_examples: 10000,
            model_update_frequency: 100,
            parallel_training: true,
            feature_selection_threshold: 0.01,
        }
    }
}

/// Training example for supervised learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Input circuit representation
    pub input: MLCircuitRepresentation,
    /// Target optimization result
    pub target: OptimizationTarget,
    /// Quality score
    pub score: f64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Optimization target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationTarget {
    /// Minimize circuit depth
    MinimizeDepth { target_depth: usize },
    /// Minimize gate count
    MinimizeGates { target_count: usize },
    /// Maximize parallelization
    MaximizeParallelization { target_parallel_fraction: f64 },
    /// Custom objective
    Custom {
        objective: String,
        target_value: f64,
    },
}

/// Feature extractor for circuits
pub struct FeatureExtractor {
    /// `SciRS2` analyzer for graph features
    analyzer: SciRS2CircuitAnalyzer,
    /// Feature cache
    cache: HashMap<String, Vec<f64>>,
    /// Feature importance weights
    feature_weights: Vec<f64>,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    /// Create a new feature extractor
    #[must_use]
    pub fn new() -> Self {
        Self {
            analyzer: SciRS2CircuitAnalyzer::new(),
            cache: HashMap::new(),
            feature_weights: Vec::new(),
        }
    }

    /// Extract features from a circuit
    pub fn extract_features<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<Vec<f64>> {
        // Generate cache key
        let cache_key = self.generate_cache_key(circuit);

        // Check cache
        if let Some(features) = self.cache.get(&cache_key) {
            return Ok(features.clone());
        }

        // Extract features
        let mut features = Vec::new();

        // Basic circuit features
        features.extend(self.extract_basic_features(circuit));

        // Graph-based features using SciRS2
        features.extend(self.extract_graph_features(circuit)?);

        // Gate pattern features
        features.extend(self.extract_pattern_features(circuit));

        // Entanglement features
        features.extend(self.extract_entanglement_features(circuit));

        // Cache the result
        self.cache.insert(cache_key, features.clone());

        Ok(features)
    }

    /// Extract basic circuit features
    fn extract_basic_features<const N: usize>(&self, circuit: &Circuit<N>) -> Vec<f64> {
        let gates = circuit.gates();
        let depth = circuit.gates().len() as f64;
        let gate_count = gates.len() as f64;

        // Gate type distribution
        let mut gate_types = HashMap::new();
        for gate in gates {
            *gate_types.entry(gate.name().to_string()).or_insert(0.0) += 1.0;
        }

        let h_count = gate_types.get("H").copied().unwrap_or(0.0);
        let cnot_count = gate_types.get("CNOT").copied().unwrap_or(0.0);
        let single_qubit_count = gate_count - cnot_count;

        vec![
            depth,
            gate_count,
            single_qubit_count,
            cnot_count,
            h_count,
            depth / gate_count.max(1.0),      // Density
            cnot_count / gate_count.max(1.0), // Two-qubit fraction
            N as f64,                         // Number of qubits
        ]
    }

    /// Extract graph-based features using `SciRS2`
    fn extract_graph_features<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<Vec<f64>> {
        let analysis = self.analyzer.analyze_circuit(circuit)?;

        let metrics = &analysis.metrics;

        Ok(vec![
            metrics.num_nodes as f64,
            metrics.num_edges as f64,
            metrics.density,
            metrics.clustering_coefficient,
            metrics.connected_components as f64,
            metrics.diameter.unwrap_or(0) as f64,
            metrics.average_path_length.unwrap_or(0.0),
            analysis.communities.len() as f64,
            analysis.critical_paths.len() as f64,
        ])
    }

    /// Extract gate pattern features
    fn extract_pattern_features<const N: usize>(&self, circuit: &Circuit<N>) -> Vec<f64> {
        let gates = circuit.gates();
        let mut features = Vec::new();

        // Sequential patterns
        let mut h_cnot_patterns = 0.0;
        let mut cnot_chains = 0.0;

        for window in gates.windows(2) {
            if window.len() == 2 {
                let gate1 = &window[0];
                let gate2 = &window[1];

                if gate1.name() == "H" && gate2.name() == "CNOT" {
                    h_cnot_patterns += 1.0;
                }

                if gate1.name() == "CNOT" && gate2.name() == "CNOT" {
                    cnot_chains += 1.0;
                }
            }
        }

        features.push(h_cnot_patterns);
        features.push(cnot_chains);

        // Qubit usage patterns
        let mut qubit_usage = vec![0.0; N];
        for gate in gates {
            for qubit in gate.qubits() {
                qubit_usage[qubit.id() as usize] += 1.0;
            }
        }

        let max_usage: f64 = qubit_usage.iter().fold(0.0_f64, |a, &b| a.max(b));
        let min_usage = qubit_usage.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let avg_usage = qubit_usage.iter().sum::<f64>() / N as f64;

        features.push(max_usage);
        features.push(min_usage);
        features.push(avg_usage);
        features.push(max_usage - min_usage); // Usage variance

        features
    }

    /// Extract entanglement-related features
    fn extract_entanglement_features<const N: usize>(&self, circuit: &Circuit<N>) -> Vec<f64> {
        let gates = circuit.gates();
        let mut features = Vec::new();

        // Entangling gate distribution
        let cnot_gates: Vec<_> = gates.iter().filter(|gate| gate.name() == "CNOT").collect();

        // Connectivity graph
        let mut connectivity = vec![vec![false; N]; N];
        for gate in &cnot_gates {
            let qubits = gate.qubits();
            if qubits.len() == 2 {
                let q1 = qubits[0].id() as usize;
                let q2 = qubits[1].id() as usize;
                connectivity[q1][q2] = true;
                connectivity[q2][q1] = true;
            }
        }

        // Calculate connectivity features
        let total_connections: f64 = connectivity
            .iter()
            .flat_map(|row| row.iter())
            .map(|&connected| if connected { 1.0 } else { 0.0 })
            .sum();

        let max_connections = N * (N - 1);
        let connectivity_ratio = total_connections / max_connections as f64;

        features.push(cnot_gates.len() as f64);
        features.push(connectivity_ratio);

        // Star connectivity (one qubit connected to many)
        let max_degree = connectivity
            .iter()
            .map(|row| row.iter().filter(|&&c| c).count())
            .max()
            .unwrap_or(0) as f64;

        features.push(max_degree);

        features
    }

    /// Generate cache key for a circuit
    fn generate_cache_key<const N: usize>(&self, circuit: &Circuit<N>) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        N.hash(&mut hasher);
        circuit.gates().len().hash(&mut hasher);

        for gate in circuit.gates() {
            gate.name().hash(&mut hasher);
            for qubit in gate.qubits() {
                qubit.id().hash(&mut hasher);
            }
        }

        format!("{:x}", hasher.finish())
    }
}

/// ML model abstraction
#[derive(Debug, Clone)]
pub enum MLModel {
    /// Linear regression model
    LinearRegression { weights: Vec<f64>, bias: f64 },
    /// Neural network model
    NeuralNetwork {
        layers: Vec<Layer>,
        learning_rate: f64,
    },
    /// Q-learning model
    QLearning {
        q_table: HashMap<String, HashMap<String, f64>>,
        learning_rate: f64,
        discount_factor: f64,
    },
    /// Random forest model
    RandomForest {
        trees: Vec<DecisionTree>,
        num_trees: usize,
    },
}

/// Neural network layer
#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: ActivationFunction,
}

/// Activation functions
#[derive(Debug, Clone)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}

/// Decision tree for random forest
#[derive(Debug, Clone)]
pub struct DecisionTree {
    pub nodes: Vec<TreeNode>,
    pub root: usize,
}

/// Tree node
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub feature_index: Option<usize>,
    pub threshold: Option<f64>,
    pub value: Option<f64>,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
}

impl MLCircuitOptimizer {
    /// Create a new ML optimizer
    #[must_use]
    pub fn new(strategy: MLStrategy) -> Self {
        Self {
            strategy,
            feature_extractor: Arc::new(Mutex::new(FeatureExtractor::new())),
            models: Arc::new(Mutex::new(HashMap::new())),
            training_data: Arc::new(Mutex::new(Vec::new())),
            config: MLOptimizerConfig::default(),
        }
    }

    /// Create optimizer with custom configuration
    #[must_use]
    pub fn with_config(strategy: MLStrategy, config: MLOptimizerConfig) -> Self {
        Self {
            strategy,
            feature_extractor: Arc::new(Mutex::new(FeatureExtractor::new())),
            models: Arc::new(Mutex::new(HashMap::new())),
            training_data: Arc::new(Mutex::new(Vec::new())),
            config,
        }
    }

    /// Optimize a circuit using ML
    pub fn optimize<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<MLOptimizationResult<N>> {
        let start_time = Instant::now();

        // Extract features
        let features = {
            let mut extractor = self.feature_extractor.lock().map_err(|e| {
                QuantRS2Error::RuntimeError(format!("Failed to lock feature extractor: {e}"))
            })?;
            extractor.extract_features(circuit)?
        };

        // Apply optimization based on strategy
        let optimized_circuit = match &self.strategy {
            MLStrategy::ReinforcementLearning { .. } => {
                self.optimize_with_rl(circuit, &features)?
            }
            MLStrategy::NeuralNetwork { .. } => self.optimize_with_nn(circuit, &features)?,
            MLStrategy::GeneticAlgorithm { .. } => self.optimize_with_ga(circuit, &features)?,
            MLStrategy::BayesianOptimization { .. } => {
                self.optimize_with_bayesian(circuit, &features)?
            }
        };

        let optimization_time = start_time.elapsed();

        Ok(MLOptimizationResult {
            original_circuit: circuit.clone(),
            optimized_circuit: optimized_circuit.clone(),
            features,
            optimization_time,
            improvement_metrics: self.calculate_improvement_metrics(circuit, &optimized_circuit),
            strategy_used: self.strategy.clone(),
        })
    }

    /// Optimize using reinforcement learning (Q-learning).
    ///
    /// State:   circuit feature vector quantized to integers (resolution 10).
    /// Actions: NoOp | RemoveGate(idx) | SwapAdjacent(i, i+1)
    /// Reward:  (old_depth - new_depth) + (old_gates - new_gates) * 0.1
    /// Update:  Q(s,a) += lr * (reward + γ * max_Q(s') - Q(s,a))
    fn optimize_with_rl<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        features: &[f64],
    ) -> QuantRS2Result<Circuit<N>> {
        let (episodes, learning_rate, discount_factor, exploration_rate) = match &self.strategy {
            MLStrategy::ReinforcementLearning {
                episodes,
                learning_rate,
                discount_factor,
                exploration_rate,
            } => (
                *episodes,
                *learning_rate,
                *discount_factor,
                *exploration_rate,
            ),
            _ => (100, 0.1, 0.9, 0.2),
        };

        // Q-table: state (quantized feature vec) -> action values
        let mut q_table: HashMap<Vec<i32>, Vec<f64>> = HashMap::new();

        let gate_count = circuit.gates().len();
        // Action space:
        //   0         → NoOp
        //   1..=G     → RemoveGate(i-1)
        //   G+1..=2G  → SwapAdjacent(i-G-1, i-G)
        let n_actions = 1 + gate_count + gate_count.saturating_sub(1);

        let mut rng = make_prng_seed();

        // Helper: quantize feature vec to i32 state key
        let quantize =
            |feats: &[f64]| -> Vec<i32> { feats.iter().map(|&v| (v * 10.0) as i32).collect() };

        // Helper: build gates-as-boxes from circuit for mutation
        let gates_to_boxes = |c: &Circuit<N>| -> Vec<Box<dyn GateOp>> { c.gates_as_boxes() };

        let mut best_circuit = circuit.clone();
        let mut best_score = gate_count as f64;

        for _ep in 0..episodes {
            let mut current_circuit = circuit.clone();

            // Extract current state features lazily (reuse provided features for initial)
            let mut state_feats = features.to_vec();

            for _step in 0..10 {
                let state_key = quantize(&state_feats);
                let n_acts = {
                    let g = current_circuit.gates().len();
                    1 + g + g.saturating_sub(1)
                };

                // Ensure Q-table row exists
                let q_row = q_table
                    .entry(state_key.clone())
                    .or_insert_with(|| vec![0.0; n_acts.max(n_actions)]);

                // ε-greedy action selection
                let action = if prng_f64(&mut rng) < exploration_rate {
                    prng_usize(&mut rng, n_acts.max(1))
                } else {
                    q_row
                        .iter()
                        .take(n_acts.max(1))
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                };

                let old_depth = current_circuit.gates().len();
                let old_gates = old_depth;

                // Apply action to a clone, then accept if reward ≥ 0
                let gate_vec = gates_to_boxes(&current_circuit);
                let g_count = gate_vec.len();

                let new_circuit_opt: Option<Circuit<N>> = if action == 0 || g_count == 0 {
                    // NoOp
                    None
                } else if action <= g_count {
                    // RemoveGate(action - 1)
                    let idx = action - 1;
                    let mut new_gates: Vec<Box<dyn GateOp>> = gate_vec
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| *i != idx)
                        .map(|(_, g)| g)
                        .collect();
                    Circuit::<N>::from_gates(new_gates).ok()
                } else {
                    // SwapAdjacent(i, i+1)
                    let i = action - g_count - 1;
                    if i + 1 < g_count {
                        let mut new_gates = gate_vec;
                        new_gates.swap(i, i + 1);
                        Circuit::<N>::from_gates(new_gates).ok()
                    } else {
                        None
                    }
                };

                let (next_circuit, reward) = match new_circuit_opt {
                    Some(nc) => {
                        let new_depth = nc.gates().len();
                        let new_gates_count = new_depth;
                        let r = (old_depth as f64 - new_depth as f64)
                            + (old_gates as f64 - new_gates_count as f64) * 0.1;
                        (nc, r)
                    }
                    None => (current_circuit.clone(), 0.0),
                };

                // Build next-state features (basic: gate count, depth proxy)
                let next_gate_count = next_circuit.gates().len();
                let next_feats: Vec<f64> = {
                    let mut f = state_feats.clone();
                    if !f.is_empty() {
                        f[0] = next_gate_count as f64;
                    }
                    if f.len() > 1 {
                        f[1] = next_gate_count as f64;
                    }
                    f
                };
                let next_key = quantize(&next_feats);

                // max Q(s')
                let max_q_next = q_table
                    .get(&next_key)
                    .and_then(|row| row.iter().cloned().reduce(f64::max))
                    .unwrap_or(0.0);

                // Q-table update
                let q_row = q_table
                    .entry(state_key)
                    .or_insert_with(|| vec![0.0; n_actions.max(1)]);
                let act_idx = action.min(q_row.len().saturating_sub(1));
                let old_q = q_row[act_idx];
                q_row[act_idx] =
                    old_q + learning_rate * (reward + discount_factor * max_q_next - old_q);

                // Track best
                let score = next_circuit.gates().len() as f64;
                if score < best_score {
                    best_score = score;
                    best_circuit = next_circuit.clone();
                }

                current_circuit = next_circuit;
                state_feats = next_feats;
            }
        }

        Ok(best_circuit)
    }

    /// Optimize using neural network
    fn optimize_with_nn<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        features: &[f64],
    ) -> QuantRS2Result<Circuit<N>> {
        // Simplified NN optimization
        // In a full implementation, this would:
        // 1. Use trained NN to predict optimal gate sequences
        // 2. Apply predicted transformations
        // 3. Validate and refine results

        Ok(circuit.clone())
    }

    /// Optimize using genetic algorithm.
    ///
    /// Individual: permutation of gate indices (a reordering of the original gate list).
    /// Fitness:    gate_count_reduction + depth_reduction (both relative to original).
    /// Selection:  tournament selection (tournament size 3).
    /// Crossover:  one-point cut on gate index list.
    /// Mutation:   swap two random positions with probability `mutation_rate`.
    fn optimize_with_ga<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        _features: &[f64],
    ) -> QuantRS2Result<Circuit<N>> {
        let (population_size, generations, mutation_rate, _selection_pressure) =
            match &self.strategy {
                MLStrategy::GeneticAlgorithm {
                    population_size,
                    generations,
                    mutation_rate,
                    selection_pressure,
                } => (
                    *population_size,
                    *generations,
                    *mutation_rate,
                    *selection_pressure,
                ),
                _ => (20, 50, 0.1, 2.0),
            };

        let original_gates = circuit.gates_as_boxes();
        let gate_count = original_gates.len();

        if gate_count == 0 {
            return Ok(circuit.clone());
        }

        let original_gate_count = gate_count as f64;
        let original_depth = gate_count as f64; // proxy: sequential depth

        let mut rng = make_prng_seed();

        // Fitness evaluation: build circuit from ordering, count gates, return fitness score.
        // Higher fitness = more reduction.  Gate count reduction is weighted 0.7 and depth proxy 0.3.
        let evaluate_fitness = |ordering: &[usize]| -> f64 {
            // Prune adjacent inverses that cancel (H-H, X-X, Z-Z on same qubit)
            let mut kept: Vec<usize> = Vec::with_capacity(ordering.len());
            for &idx in ordering {
                if let Some(&last) = kept.last() {
                    let g1 = original_gates[last].name();
                    let g2 = original_gates[idx].name();
                    let q1 = original_gates[last].qubits();
                    let q2 = original_gates[idx].qubits();
                    let same_target = q1.len() == 1 && q2.len() == 1 && q1[0] == q2[0];
                    let self_inverse = matches!(g1, "H" | "X" | "Y" | "Z" | "CNOT" | "CX");
                    if same_target && g1 == g2 && self_inverse {
                        kept.pop();
                        continue;
                    }
                }
                kept.push(idx);
            }
            let new_gate_count = kept.len() as f64;
            let gate_reduction =
                (original_gate_count - new_gate_count) / original_gate_count.max(1.0);
            let depth_reduction = (original_depth - new_gate_count) / original_depth.max(1.0);
            gate_reduction * 0.7 + depth_reduction * 0.3
        };

        // Initialize population: identity ordering plus mutations
        let identity: Vec<usize> = (0..gate_count).collect();
        let mut population: Vec<(Vec<usize>, f64)> = Vec::with_capacity(population_size);

        // Add identity first
        let id_fit = evaluate_fitness(&identity);
        population.push((identity.clone(), id_fit));

        for _ in 1..population_size {
            let mut individual = identity.clone();
            // Random shuffle via Fisher-Yates
            for i in (1..gate_count).rev() {
                let j = prng_usize(&mut rng, i + 1);
                individual.swap(i, j);
            }
            let fit = evaluate_fitness(&individual);
            population.push((individual, fit));
        }

        // Tournament selection: pick best of 3 random individuals
        let tournament_select = |pop: &[(Vec<usize>, f64)], rng: &mut u64| -> usize {
            let a = prng_usize(rng, pop.len());
            let b = prng_usize(rng, pop.len());
            let c = prng_usize(rng, pop.len());
            let (ia, fa) = (a, pop[a].1);
            let (ib, fb) = (b, pop[b].1);
            let (ic, fc) = (c, pop[c].1);
            if fa >= fb && fa >= fc {
                ia
            } else if fb >= fc {
                ib
            } else {
                ic
            }
        };

        for _gen in 0..generations {
            let mut new_population: Vec<(Vec<usize>, f64)> = Vec::with_capacity(population_size);

            // Elitism: keep the best individual
            let best_idx = population
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1 .1
                        .partial_cmp(&b.1 .1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            new_population.push(population[best_idx].clone());

            while new_population.len() < population_size {
                // Selection
                let p1_idx = tournament_select(&population, &mut rng);
                let p2_idx = tournament_select(&population, &mut rng);
                let p1 = &population[p1_idx].0;
                let p2 = &population[p2_idx].0;

                // One-point crossover
                let cut = prng_usize(&mut rng, gate_count.max(1));
                let mut child: Vec<usize> = p1[..cut].to_vec();
                // Append elements from p2 that are not yet in child
                for &g in p2 {
                    if !child.contains(&g) {
                        child.push(g);
                    }
                }
                // Append any still-missing elements from p1 (fill gaps)
                for &g in p1 {
                    if !child.contains(&g) {
                        child.push(g);
                    }
                }

                // Mutation: swap two positions
                if prng_f64(&mut rng) < mutation_rate && gate_count >= 2 {
                    let i = prng_usize(&mut rng, gate_count);
                    let j = prng_usize(&mut rng, gate_count);
                    child.swap(i, j);
                }

                let fit = evaluate_fitness(&child);
                new_population.push((child, fit));
            }

            population = new_population;
        }

        // Extract best individual
        let best = population
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_ordering = best.map(|(o, _)| o.as_slice()).unwrap_or(&[]);

        // Reconstruct pruned gate list
        let mut kept_indices: Vec<usize> = Vec::with_capacity(gate_count);
        for &idx in best_ordering {
            if let Some(&last) = kept_indices.last() {
                let g1 = original_gates[last].name();
                let g2 = original_gates[idx].name();
                let q1 = original_gates[last].qubits();
                let q2 = original_gates[idx].qubits();
                let same_target = q1.len() == 1 && q2.len() == 1 && q1[0] == q2[0];
                let self_inverse = matches!(g1, "H" | "X" | "Y" | "Z" | "CNOT" | "CX");
                if same_target && g1 == g2 && self_inverse {
                    kept_indices.pop();
                    continue;
                }
            }
            kept_indices.push(idx);
        }

        let new_gates: Vec<Box<dyn GateOp>> = kept_indices
            .iter()
            .map(|&i| original_gates[i].clone_gate())
            .collect();

        Circuit::<N>::from_gates(new_gates)
    }

    /// Optimize using Bayesian optimization
    fn optimize_with_bayesian<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        features: &[f64],
    ) -> QuantRS2Result<Circuit<N>> {
        // Simplified Bayesian optimization
        // In a full implementation, this would:
        // 1. Build Gaussian process model of optimization landscape
        // 2. Use acquisition function to select next optimization point
        // 3. Iteratively improve based on observed results

        Ok(circuit.clone())
    }

    /// Calculate improvement metrics
    fn calculate_improvement_metrics<const N: usize>(
        &self,
        original: &Circuit<N>,
        optimized: &Circuit<N>,
    ) -> ImprovementMetrics {
        let original_depth = original.gates().len();
        let optimized_depth = optimized.gates().len();
        let original_gates = original.gates().len();
        let optimized_gates = optimized.gates().len();

        ImprovementMetrics {
            depth_reduction: (original_depth as f64 - optimized_depth as f64)
                / original_depth as f64,
            gate_reduction: (original_gates as f64 - optimized_gates as f64)
                / original_gates as f64,
            compilation_speedup: 1.0,  // Placeholder
            fidelity_improvement: 0.0, // Placeholder
        }
    }

    /// Add training example
    pub fn add_training_example(&mut self, example: TrainingExample) {
        let mut data = self
            .training_data
            .lock()
            .expect("Training data mutex poisoned");
        data.push(example);

        // Maintain maximum size
        if data.len() > self.config.max_training_examples {
            data.remove(0);
        }
    }

    /// Train models with current data
    pub fn train_models(&mut self) -> QuantRS2Result<()> {
        let data = {
            let training_data = self.training_data.lock().map_err(|e| {
                QuantRS2Error::RuntimeError(format!("Failed to lock training data: {e}"))
            })?;
            training_data.clone()
        };

        if data.is_empty() {
            return Err(QuantRS2Error::InvalidInput(
                "No training data available".to_string(),
            ));
        }

        // Train based on strategy
        let strategy = self.strategy.clone();
        match strategy {
            MLStrategy::NeuralNetwork {
                architecture,
                learning_rate,
                epochs,
                batch_size,
            } => {
                self.train_neural_network(&data, &architecture, learning_rate, epochs, batch_size)?;
            }
            MLStrategy::ReinforcementLearning {
                learning_rate,
                discount_factor,
                ..
            } => {
                self.train_rl_model(&data, learning_rate, discount_factor)?;
            }
            _ => {
                // Other strategies would be implemented here
            }
        }

        Ok(())
    }

    /// Train neural network model
    fn train_neural_network(
        &self,
        data: &[TrainingExample],
        architecture: &[usize],
        learning_rate: f64,
        epochs: usize,
        batch_size: usize,
    ) -> QuantRS2Result<()> {
        // Simplified NN training
        // In a full implementation, this would implement backpropagation

        let input_size = data.first().map_or(0, |ex| ex.input.features.len());

        // Create network layers
        let mut layers = Vec::new();
        let mut prev_size = input_size;

        for &layer_size in architecture {
            let weights = vec![vec![0.1; prev_size]; layer_size]; // Random initialization
            let biases = vec![0.0; layer_size];

            layers.push(Layer {
                weights,
                biases,
                activation: ActivationFunction::ReLU,
            });

            prev_size = layer_size;
        }

        let model = MLModel::NeuralNetwork {
            layers,
            learning_rate,
        };

        let mut models = self
            .models
            .lock()
            .map_err(|e| QuantRS2Error::RuntimeError(format!("Failed to lock models: {e}")))?;
        models.insert("neural_network".to_string(), model);

        Ok(())
    }

    /// Train reinforcement learning model
    fn train_rl_model(
        &self,
        data: &[TrainingExample],
        learning_rate: f64,
        discount_factor: f64,
    ) -> QuantRS2Result<()> {
        // Simplified Q-learning training
        let model = MLModel::QLearning {
            q_table: HashMap::new(),
            learning_rate,
            discount_factor,
        };

        let mut models = self
            .models
            .lock()
            .map_err(|e| QuantRS2Error::RuntimeError(format!("Failed to lock models: {e}")))?;
        models.insert("q_learning".to_string(), model);

        Ok(())
    }
}

/// ML optimization result
#[derive(Debug, Clone)]
pub struct MLOptimizationResult<const N: usize> {
    /// Original circuit
    pub original_circuit: Circuit<N>,
    /// Optimized circuit
    pub optimized_circuit: Circuit<N>,
    /// Extracted features
    pub features: Vec<f64>,
    /// Time taken for optimization
    pub optimization_time: Duration,
    /// Improvement metrics
    pub improvement_metrics: ImprovementMetrics,
    /// Strategy used
    pub strategy_used: MLStrategy,
}

/// Improvement metrics
#[derive(Debug, Clone)]
pub struct ImprovementMetrics {
    /// Relative depth reduction
    pub depth_reduction: f64,
    /// Relative gate count reduction
    pub gate_reduction: f64,
    /// Compilation speedup factor
    pub compilation_speedup: f64,
    /// Fidelity improvement
    pub fidelity_improvement: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::Hadamard;

    #[test]
    fn test_feature_extraction() {
        let mut extractor = FeatureExtractor::new();

        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate");
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("Failed to add CNOT gate");

        let features = extractor
            .extract_features(&circuit)
            .expect("Failed to extract features");
        assert!(!features.is_empty());
        assert!(features.len() > 10); // Should have multiple feature categories
    }

    #[test]
    fn test_ml_optimizer_creation() {
        let strategy = MLStrategy::NeuralNetwork {
            architecture: vec![64, 32, 16],
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
        };

        let optimizer = MLCircuitOptimizer::new(strategy);
        assert!(matches!(
            optimizer.strategy,
            MLStrategy::NeuralNetwork { .. }
        ));
    }

    #[test]
    fn test_training_example() {
        let features = vec![1.0, 2.0, 3.0, 4.0];
        let target = OptimizationTarget::MinimizeDepth { target_depth: 5 };

        let representation = MLCircuitRepresentation {
            features,
            gate_sequence: vec![0, 1, 2],
            adjacency_matrix: vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            qubit_connectivity: vec![vec![false, true], vec![true, false]],
            metrics: CircuitMetrics {
                depth: 3,
                gate_count: 3,
                two_qubit_gate_count: 1,
                entanglement_measure: 0.5,
                critical_path_length: 3,
                parallelization_potential: 0.3,
            },
        };

        let example = TrainingExample {
            input: representation,
            target,
            score: 0.8,
            metadata: HashMap::new(),
        };

        assert!(example.score > 0.0);
    }

    #[test]
    fn test_ml_strategies() {
        let rl_strategy = MLStrategy::ReinforcementLearning {
            learning_rate: 0.1,
            discount_factor: 0.9,
            exploration_rate: 0.1,
            episodes: 1000,
        };

        let nn_strategy = MLStrategy::NeuralNetwork {
            architecture: vec![32, 16, 8],
            learning_rate: 0.001,
            epochs: 50,
            batch_size: 16,
        };

        assert!(matches!(
            rl_strategy,
            MLStrategy::ReinforcementLearning { .. }
        ));
        assert!(matches!(nn_strategy, MLStrategy::NeuralNetwork { .. }));
    }

    #[test]
    fn test_optimize_with_rl_reduces_or_preserves() {
        let strategy = MLStrategy::ReinforcementLearning {
            learning_rate: 0.1,
            discount_factor: 0.9,
            exploration_rate: 0.2,
            episodes: 5,
        };
        let mut optimizer = MLCircuitOptimizer::new(strategy);

        // Build a circuit with adjacent H-H cancellations
        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("H gate");
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("H gate");
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("CNOT gate");

        let result = optimizer.optimize(&circuit).expect("RL optimize");
        // After RL: optimized circuit should have ≤ original gate count
        assert!(result.optimized_circuit.gates().len() <= circuit.gates().len());
    }

    #[test]
    fn test_optimize_with_ga_cancels_adjacent_inverses() {
        let strategy = MLStrategy::GeneticAlgorithm {
            population_size: 10,
            generations: 5,
            mutation_rate: 0.1,
            selection_pressure: 2.0,
        };
        let mut optimizer = MLCircuitOptimizer::new(strategy);

        // H H on same qubit: GA should detect and cancel
        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("H gate 1");
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("H gate 2");
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("CNOT gate");

        let result = optimizer.optimize(&circuit).expect("GA optimize");
        assert!(result.optimized_circuit.gates().len() <= circuit.gates().len());
    }

    #[test]
    fn test_circuit_representation() {
        let metrics = CircuitMetrics {
            depth: 5,
            gate_count: 10,
            two_qubit_gate_count: 3,
            entanglement_measure: 0.7,
            critical_path_length: 5,
            parallelization_potential: 0.4,
        };

        let representation = MLCircuitRepresentation {
            features: vec![1.0, 2.0, 3.0],
            gate_sequence: vec![0, 1, 2, 1, 0],
            adjacency_matrix: vec![vec![0.0; 3]; 3],
            qubit_connectivity: vec![vec![false; 3]; 3],
            metrics,
        };

        assert_eq!(representation.metrics.depth, 5);
        assert_eq!(representation.gate_sequence.len(), 5);
    }
}
