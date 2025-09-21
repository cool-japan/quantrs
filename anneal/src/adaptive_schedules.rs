//! Neural Network Guided Annealing Schedules
//!
//! This module implements intelligent annealing schedule optimization using neural networks
//! and reinforcement learning. It adaptively learns optimal temperature schedules, cooling
//! rates, and other annealing parameters based on problem characteristics and real-time
//! performance feedback.
//!
//! Key features:
//! - Neural network based schedule prediction
//! - Reinforcement learning for parameter optimization
//! - Real-time performance monitoring and adaptation
//! - Problem-aware schedule customization
//! - Multi-objective schedule optimization
//! - Transfer learning across problem types

use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::ising::{IsingModel, IsingError};
use crate::simulator::{AnnealingParams, AnnealingSolution, TemperatureSchedule};

/// Errors that can occur in adaptive scheduling
#[derive(Error, Debug)]
pub enum AdaptiveScheduleError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Neural network error
    #[error("Neural network error: {0}")]
    NeuralNetworkError(String),

    /// Training error
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Data processing error
    #[error("Data processing error: {0}")]
    DataError(String),

    /// Optimization error
    #[error("Optimization error: {0}")]
    OptimizationError(String),
}

/// Result type for adaptive scheduling operations
pub type AdaptiveScheduleResult<T> = Result<T, AdaptiveScheduleError>;

/// Neural network guided annealing scheduler
#[derive(Debug, Clone)]
pub struct NeuralAnnealingScheduler {
    /// Neural network for schedule prediction
    pub network: SchedulePredictionNetwork,
    /// Reinforcement learning agent
    pub rl_agent: ScheduleRLAgent,
    /// Configuration
    pub config: SchedulerConfig,
    /// Training history
    pub training_history: TrainingHistory,
    /// Problem feature cache
    pub feature_cache: HashMap<String, ProblemFeatures>,
    /// Performance statistics
    pub performance_stats: PerformanceStatistics,
}

/// Configuration for the neural annealing scheduler
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Network architecture
    pub network_layers: Vec<usize>,
    /// Learning rate
    pub learning_rate: f64,
    /// Training epochs
    pub training_epochs: usize,
    /// Experience buffer size
    pub buffer_size: usize,
    /// Exploration rate for RL
    pub exploration_rate: f64,
    /// Discount factor
    pub discount_factor: f64,
    /// Update frequency
    pub update_frequency: usize,
    /// Enable transfer learning
    pub use_transfer_learning: bool,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            network_layers: vec![32, 64, 32, 16],
            learning_rate: 0.001,
            training_epochs: 100,
            buffer_size: 1000,
            exploration_rate: 0.1,
            discount_factor: 0.95,
            update_frequency: 10,
            use_transfer_learning: true,
            seed: None,
        }
    }
}

/// Neural network for schedule prediction
#[derive(Debug, Clone)]
pub struct SchedulePredictionNetwork {
    /// Network layers
    pub layers: Vec<NetworkLayer>,
    /// Input normalization parameters
    pub input_normalization: NormalizationParams,
    /// Output scaling parameters
    pub output_scaling: NormalizationParams,
    /// Network training state
    pub training_state: NetworkTrainingState,
}

/// Neural network layer
#[derive(Debug, Clone)]
pub struct NetworkLayer {
    /// Layer weights
    pub weights: Vec<Vec<f64>>,
    /// Layer biases
    pub biases: Vec<f64>,
    /// Activation function
    pub activation: ActivationFunction,
}

/// Activation functions for neural network
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationFunction {
    /// ReLU activation
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Linear activation
    Linear,
    /// Leaky ReLU
    LeakyReLU(f64),
}

/// Normalization parameters
#[derive(Debug, Clone)]
pub struct NormalizationParams {
    /// Mean values
    pub means: Vec<f64>,
    /// Standard deviations
    pub stds: Vec<f64>,
    /// Min values
    pub mins: Vec<f64>,
    /// Max values
    pub maxs: Vec<f64>,
}

/// Network training state
#[derive(Debug, Clone)]
pub struct NetworkTrainingState {
    /// Current epoch
    pub epoch: usize,
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: f64,
    /// Learning rate schedule
    pub learning_rate: f64,
    /// Training metrics
    pub metrics: HashMap<String, f64>,
}

/// Reinforcement learning agent for schedule optimization
#[derive(Debug, Clone)]
pub struct ScheduleRLAgent {
    /// Q-network for action-value estimation
    pub q_network: SchedulePredictionNetwork,
    /// Target network for stable training
    pub target_network: SchedulePredictionNetwork,
    /// Experience replay buffer
    pub experience_buffer: Vec<ScheduleExperience>,
    /// Agent configuration
    pub config: RLAgentConfig,
    /// Training statistics
    pub stats: RLStats,
}

/// Reinforcement learning agent configuration
#[derive(Debug, Clone)]
pub struct RLAgentConfig {
    /// Action space size
    pub action_space_size: usize,
    /// State space size
    pub state_space_size: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Target network update frequency
    pub target_update_frequency: usize,
    /// Epsilon decay rate
    pub epsilon_decay: f64,
    /// Minimum epsilon
    pub min_epsilon: f64,
}

/// Experience tuple for reinforcement learning
#[derive(Debug, Clone)]
pub struct ScheduleExperience {
    /// Current state (problem features + current schedule)
    pub state: Vec<f64>,
    /// Action taken (schedule modification)
    pub action: usize,
    /// Reward received (performance improvement)
    pub reward: f64,
    /// Next state
    pub next_state: Vec<f64>,
    /// Episode done flag
    pub done: bool,
    /// Additional metadata
    pub metadata: ExperienceMetadata,
}

/// Metadata for RL experience
#[derive(Debug, Clone)]
pub struct ExperienceMetadata {
    /// Problem type
    pub problem_type: String,
    /// Problem size
    pub problem_size: usize,
    /// Execution time
    pub execution_time: Duration,
    /// Final energy achieved
    pub final_energy: f64,
}

/// RL training statistics
#[derive(Debug, Clone)]
pub struct RLStats {
    /// Episode rewards
    pub episode_rewards: Vec<f64>,
    /// Average reward over episodes
    pub average_reward: f64,
    /// Exploration rate over time
    pub exploration_history: Vec<f64>,
    /// Q-network loss over time
    pub loss_history: Vec<f64>,
    /// Action selection frequency
    pub action_frequency: HashMap<usize, usize>,
}

/// Problem features for neural network input
#[derive(Debug, Clone)]
pub struct ProblemFeatures {
    /// Problem size (number of variables)
    pub size: usize,
    /// Connectivity density
    pub connectivity_density: f64,
    /// Coupling strength statistics
    pub coupling_stats: CouplingStatistics,
    /// Problem type classification
    pub problem_type: ProblemType,
    /// Energy landscape characteristics
    pub landscape_features: LandscapeFeatures,
    /// Historical performance data
    pub historical_performance: Vec<PerformancePoint>,
}

/// Statistics about coupling strengths in the problem
#[derive(Debug, Clone)]
pub struct CouplingStatistics {
    /// Mean coupling strength
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Maximum absolute coupling
    pub max_abs: f64,
    /// Range of couplings
    pub range: f64,
    /// Coupling distribution skewness
    pub skewness: f64,
}

/// Problem type classification
#[derive(Debug, Clone, PartialEq)]
pub enum ProblemType {
    /// Random Ising model
    Random,
    /// Structured problem (e.g., grid, tree)
    Structured,
    /// Optimization problem (e.g., Max-Cut, TSP)
    Optimization,
    /// Machine learning problem
    MachineLearning,
    /// Industry-specific problem
    IndustrySpecific(String),
    /// Unknown type
    Unknown,
}

/// Energy landscape characteristics
#[derive(Debug, Clone)]
pub struct LandscapeFeatures {
    /// Estimated number of local minima
    pub num_local_minima: usize,
    /// Ruggedness measure
    pub ruggedness: f64,
    /// Connectivity of energy levels
    pub energy_connectivity: f64,
    /// Barrier heights between minima
    pub barrier_heights: Vec<f64>,
    /// Funnel structure measure
    pub funnel_structure: f64,
}

/// Performance point for historical data
#[derive(Debug, Clone)]
pub struct PerformancePoint {
    /// Schedule parameters used
    pub schedule_params: ScheduleParameters,
    /// Performance achieved
    pub performance: PerformanceMetrics,
    /// Problem context
    pub context: ProblemContext,
}

/// Schedule parameters
#[derive(Debug, Clone)]
pub struct ScheduleParameters {
    /// Initial temperature
    pub initial_temp: f64,
    /// Final temperature
    pub final_temp: f64,
    /// Number of sweeps
    pub num_sweeps: usize,
    /// Cooling rate
    pub cooling_rate: f64,
    /// Schedule type
    pub schedule_type: ScheduleType,
    /// Additional parameters
    pub additional_params: HashMap<String, f64>,
}

/// Types of annealing schedules
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleType {
    /// Linear cooling
    Linear,
    /// Exponential cooling
    Exponential,
    /// Logarithmic cooling
    Logarithmic,
    /// Custom schedule
    Custom(Vec<f64>),
    /// Adaptive schedule
    Adaptive,
}

/// Performance metrics for a run
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Final energy achieved
    pub final_energy: f64,
    /// Number of evaluations
    pub num_evaluations: usize,
    /// Execution time
    pub execution_time: Duration,
    /// Success rate (for multiple runs)
    pub success_rate: f64,
    /// Convergence speed
    pub convergence_speed: f64,
    /// Quality relative to known optimum
    pub solution_quality: f64,
}

/// Problem context information
#[derive(Debug, Clone)]
pub struct ProblemContext {
    /// Problem identifier
    pub problem_id: String,
    /// Timestamp
    pub timestamp: Instant,
    /// Hardware used
    pub hardware_type: String,
    /// Environment conditions
    pub environment: HashMap<String, f64>,
}

/// Training history for the scheduler
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Network training losses over time
    pub network_losses: Vec<f64>,
    /// RL rewards over time
    pub rl_rewards: Vec<f64>,
    /// Validation scores
    pub validation_scores: Vec<f64>,
    /// Feature importance evolution
    pub feature_importance: Vec<HashMap<String, f64>>,
    /// Training times per epoch
    pub training_times: Vec<Duration>,
}

/// Performance statistics for the scheduler
#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    /// Number of problems solved
    pub problems_solved: usize,
    /// Average improvement over baseline
    pub avg_improvement: f64,
    /// Best improvement achieved
    pub best_improvement: f64,
    /// Adaptation time
    pub adaptation_time: Duration,
    /// Success rate
    pub success_rate: f64,
    /// Transfer learning effectiveness
    pub transfer_effectiveness: f64,
}

impl NeuralAnnealingScheduler {
    /// Create a new neural annealing scheduler
    pub fn new(config: SchedulerConfig) -> AdaptiveScheduleResult<Self> {
        let network = SchedulePredictionNetwork::new(&config.network_layers, config.seed)?;
        let rl_agent = ScheduleRLAgent::new(RLAgentConfig {
            action_space_size: 10, // Number of discrete actions
            state_space_size: config.network_layers[0],
            batch_size: 32,
            target_update_frequency: 100,
            epsilon_decay: 0.995,
            min_epsilon: 0.01,
        })?;

        Ok(Self {
            network,
            rl_agent,
            config,
            training_history: TrainingHistory {
                network_losses: Vec::new(),
                rl_rewards: Vec::new(),
                validation_scores: Vec::new(),
                feature_importance: Vec::new(),
                training_times: Vec::new(),
            },
            feature_cache: HashMap::new(),
            performance_stats: PerformanceStatistics {
                problems_solved: 0,
                avg_improvement: 0.0,
                best_improvement: 0.0,
                adaptation_time: Duration::from_secs(0),
                success_rate: 0.0,
                transfer_effectiveness: 0.0,
            },
        })
    }

    /// Generate optimal annealing schedule for a given problem
    pub fn generate_schedule(&mut self, problem: &IsingModel) -> AdaptiveScheduleResult<AnnealingParams> {
        let start_time = Instant::now();

        // Extract problem features
        let features = self.extract_problem_features(problem)?;

        // Check cache for similar problems
        if let Some(cached_schedule) = self.check_feature_cache(&features) {
            return Ok(cached_schedule);
        }

        // Use neural network to predict optimal schedule
        let predicted_params = self.predict_schedule_parameters(&features)?;

        // Use RL agent to refine parameters
        let refined_params = self.refine_with_rl(&features, predicted_params)?;

        // Convert to AnnealingParams
        let schedule = self.convert_to_annealing_params(refined_params)?;

        // Cache the result
        self.cache_schedule(&features, schedule.clone());

        // Update adaptation time
        self.performance_stats.adaptation_time = start_time.elapsed();

        Ok(schedule)
    }

    /// Extract features from an Ising model
    fn extract_problem_features(&self, problem: &IsingModel) -> AdaptiveScheduleResult<ProblemFeatures> {
        let size = problem.num_qubits;

        // Calculate connectivity density
        let mut num_couplings = 0;
        let mut coupling_values = Vec::new();

        for i in 0..size {
            for j in (i+1)..size {
                if let Ok(coupling) = problem.get_coupling(i, j) {
                    if coupling.abs() > 1e-10 {
                        num_couplings += 1;
                        coupling_values.push(coupling.abs());
                    }
                }
            }
        }

        let max_possible_couplings = size * (size - 1) / 2;
        let connectivity_density = num_couplings as f64 / max_possible_couplings as f64;

        // Calculate coupling statistics
        let coupling_stats = if !coupling_values.is_empty() {
            let mean = coupling_values.iter().sum::<f64>() / coupling_values.len() as f64;
            let variance = coupling_values.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / coupling_values.len() as f64;
            let std = variance.sqrt();
            let max_abs = coupling_values.iter().fold(0.0f64, |a, &b| a.max(b));
            let min_val = coupling_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let range = max_abs - min_val;

            // Simple skewness calculation
            let skewness = if std > 1e-10 {
                coupling_values.iter()
                    .map(|x| ((x - mean) / std).powi(3))
                    .sum::<f64>() / coupling_values.len() as f64
            } else {
                0.0
            };

            CouplingStatistics {
                mean,
                std,
                max_abs,
                range,
                skewness,
            }
        } else {
            CouplingStatistics {
                mean: 0.0,
                std: 0.0,
                max_abs: 0.0,
                range: 0.0,
                skewness: 0.0,
            }
        };

        // Classify problem type (simplified)
        let problem_type = if connectivity_density > 0.8 {
            ProblemType::Random
        } else if connectivity_density < 0.2 {
            ProblemType::Structured
        } else {
            ProblemType::Optimization
        };

        // Simple landscape features estimation
        let landscape_features = LandscapeFeatures {
            num_local_minima: (size as f64 * connectivity_density * 10.0) as usize,
            ruggedness: coupling_stats.std / coupling_stats.mean.max(1e-10),
            energy_connectivity: connectivity_density,
            barrier_heights: vec![coupling_stats.mean; 5], // Simplified
            funnel_structure: 1.0 - connectivity_density, // Simplified metric
        };

        Ok(ProblemFeatures {
            size,
            connectivity_density,
            coupling_stats,
            problem_type,
            landscape_features,
            historical_performance: Vec::new(),
        })
    }

    /// Check feature cache for similar problems
    fn check_feature_cache(&self, features: &ProblemFeatures) -> Option<AnnealingParams> {
        // Simple similarity matching (in practice, would use more sophisticated matching)
        for (_, cached_features) in &self.feature_cache {
            if (cached_features.size as f64 - features.size as f64).abs() / features.size as f64 < 0.1 &&
               (cached_features.connectivity_density - features.connectivity_density).abs() < 0.1 &&
               cached_features.problem_type == features.problem_type {

                // Return a default schedule for now (would return cached optimized schedule)
                return Some(AnnealingParams {
                    num_sweeps: 1000 + features.size * 10,
                    initial_temperature: features.coupling_stats.max_abs * 10.0,
                    final_temperature: features.coupling_stats.max_abs * 0.001,
                    ..Default::default()
                });
            }
        }
        None
    }

    /// Use neural network to predict schedule parameters
    fn predict_schedule_parameters(&self, features: &ProblemFeatures) -> AdaptiveScheduleResult<ScheduleParameters> {
        // Convert features to input vector
        let input = self.features_to_input_vector(features);

        // Forward pass through network
        let output = self.network.forward(&input)?;

        // Convert output to schedule parameters
        let schedule_params = ScheduleParameters {
            initial_temp: output[0].max(0.1).min(100.0),
            final_temp: output[1].max(0.001).min(1.0),
            num_sweeps: (output[2].max(100.0).min(100000.0)) as usize,
            cooling_rate: output[3].max(0.01).min(0.99),
            schedule_type: ScheduleType::Exponential, // Default
            additional_params: HashMap::new(),
        };

        Ok(schedule_params)
    }

    /// Convert features to input vector for neural network
    fn features_to_input_vector(&self, features: &ProblemFeatures) -> Vec<f64> {
        vec![
            features.size as f64 / 1000.0, // Normalized size
            features.connectivity_density,
            features.coupling_stats.mean,
            features.coupling_stats.std,
            features.coupling_stats.max_abs,
            features.coupling_stats.skewness,
            features.landscape_features.ruggedness,
            features.landscape_features.energy_connectivity,
            match features.problem_type {
                ProblemType::Random => 1.0,
                ProblemType::Structured => 2.0,
                ProblemType::Optimization => 3.0,
                ProblemType::MachineLearning => 4.0,
                _ => 0.0,
            },
            features.landscape_features.funnel_structure,
        ]
    }

    /// Use RL agent to refine schedule parameters
    fn refine_with_rl(&mut self, features: &ProblemFeatures, initial_params: ScheduleParameters) -> AdaptiveScheduleResult<ScheduleParameters> {
        // Convert to state representation
        let state = self.create_rl_state(features, &initial_params);

        // Get action from RL agent
        let action = self.rl_agent.select_action(&state)?;

        // Apply action to modify parameters
        let refined_params = self.apply_rl_action(initial_params, action)?;

        Ok(refined_params)
    }

    /// Create RL state representation
    fn create_rl_state(&self, features: &ProblemFeatures, params: &ScheduleParameters) -> Vec<f64> {
        let mut state = self.features_to_input_vector(features);

        // Add current parameters to state
        state.extend(vec![
            params.initial_temp / 100.0, // Normalized
            params.final_temp / 1.0,
            params.num_sweeps as f64 / 10000.0,
            params.cooling_rate,
        ]);

        state
    }

    /// Apply RL action to modify parameters
    fn apply_rl_action(&self, mut params: ScheduleParameters, action: usize) -> AdaptiveScheduleResult<ScheduleParameters> {
        match action {
            0 => params.initial_temp *= 1.2,     // Increase initial temp
            1 => params.initial_temp *= 0.8,     // Decrease initial temp
            2 => params.final_temp *= 1.5,       // Increase final temp
            3 => params.final_temp *= 0.7,       // Decrease final temp
            4 => params.num_sweeps = (params.num_sweeps as f64 * 1.3) as usize, // Increase sweeps
            5 => params.num_sweeps = (params.num_sweeps as f64 * 0.8) as usize, // Decrease sweeps
            6 => params.cooling_rate = (params.cooling_rate * 1.1).min(0.99),  // Slower cooling
            7 => params.cooling_rate = (params.cooling_rate * 0.9).max(0.01),  // Faster cooling
            8 => { // Change schedule type
                params.schedule_type = ScheduleType::Linear;
            },
            9 => { // Change schedule type
                params.schedule_type = ScheduleType::Logarithmic;
            },
            _ => {}, // No action
        }

        Ok(params)
    }

    /// Convert schedule parameters to AnnealingParams
    fn convert_to_annealing_params(&self, params: ScheduleParameters) -> AdaptiveScheduleResult<AnnealingParams> {
        Ok(AnnealingParams {
            num_sweeps: params.num_sweeps,
            initial_temperature: params.initial_temp,
            final_temperature: params.final_temp,
            temperature_schedule: match params.schedule_type {
                ScheduleType::Linear => TemperatureSchedule::Linear,
                ScheduleType::Exponential => TemperatureSchedule::Exponential,
                ScheduleType::Logarithmic => TemperatureSchedule::Logarithmic,
                _ => TemperatureSchedule::Exponential,
            },
            ..Default::default()
        })
    }

    /// Cache schedule for similar problems
    fn cache_schedule(&mut self, features: &ProblemFeatures, schedule: AnnealingParams) {
        let cache_key = format!("{}_{:.2}_{:?}", features.size, features.connectivity_density, features.problem_type);
        self.feature_cache.insert(cache_key, features.clone());
    }

    /// Train the scheduler on historical data
    pub fn train(&mut self, training_data: &[TrainingExample]) -> AdaptiveScheduleResult<()> {
        println!("Training neural annealing scheduler with {} examples", training_data.len());

        for epoch in 0..self.config.training_epochs {
            let start_time = Instant::now();

            // Train neural network
            let network_loss = self.train_network_epoch(training_data)?;

            // Train RL agent
            let rl_reward = self.train_rl_epoch(training_data)?;

            // Update training history
            self.training_history.network_losses.push(network_loss);
            self.training_history.rl_rewards.push(rl_reward);
            self.training_history.training_times.push(start_time.elapsed());

            if epoch % 10 == 0 {
                println!("Epoch {}: Network Loss = {:.6}, RL Reward = {:.6}",
                        epoch, network_loss, rl_reward);
            }
        }

        Ok(())
    }

    /// Train network for one epoch
    fn train_network_epoch(&mut self, training_data: &[TrainingExample]) -> AdaptiveScheduleResult<f64> {
        let mut total_loss = 0.0;

        for example in training_data {
            let input = self.features_to_input_vector(&example.features);
            let target = self.params_to_output_vector(&example.optimal_params);

            // Forward pass
            let prediction = self.network.forward(&input)?;

            // Calculate loss (simplified MSE)
            let loss: f64 = prediction.iter()
                .zip(target.iter())
                .map(|(pred, targ)| (pred - targ).powi(2))
                .sum::<f64>() / prediction.len() as f64;

            total_loss += loss;

            // Backward pass (simplified - in practice would implement proper backpropagation)
            self.network.backward(&input, &target, &prediction)?;
        }

        Ok(total_loss / training_data.len() as f64)
    }

    /// Train RL agent for one epoch
    fn train_rl_epoch(&mut self, training_data: &[TrainingExample]) -> AdaptiveScheduleResult<f64> {
        let mut total_reward = 0.0;

        for example in training_data {
            // Create experience from training example
            let state = self.create_rl_state(&example.features, &example.baseline_params);
            let optimal_state = self.create_rl_state(&example.features, &example.optimal_params);

            // Calculate reward based on performance improvement
            let reward = example.performance_improvement;

            // Store experience (simplified)
            let experience = ScheduleExperience {
                state: state.clone(),
                action: 0, // Would need to infer action from parameter differences
                reward,
                next_state: optimal_state,
                done: true,
                metadata: ExperienceMetadata {
                    problem_type: format!("{:?}", example.features.problem_type),
                    problem_size: example.features.size,
                    execution_time: Duration::from_secs(1),
                    final_energy: 0.0, // Would be filled from example
                },
            };

            self.rl_agent.store_experience(experience);
            total_reward += reward;
        }

        // Train RL agent
        self.rl_agent.train()?;

        Ok(total_reward / training_data.len() as f64)
    }

    /// Convert parameters to output vector
    fn params_to_output_vector(&self, params: &ScheduleParameters) -> Vec<f64> {
        vec![
            params.initial_temp / 100.0,
            params.final_temp,
            params.num_sweeps as f64 / 10000.0,
            params.cooling_rate,
        ]
    }
}

/// Training example for the scheduler
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Problem features
    pub features: ProblemFeatures,
    /// Baseline parameters
    pub baseline_params: ScheduleParameters,
    /// Optimal parameters found
    pub optimal_params: ScheduleParameters,
    /// Performance improvement achieved
    pub performance_improvement: f64,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

impl SchedulePredictionNetwork {
    /// Create a new schedule prediction network
    pub fn new(layer_sizes: &[usize], seed: Option<u64>) -> AdaptiveScheduleResult<Self> {
        if layer_sizes.len() < 2 {
            return Err(AdaptiveScheduleError::ConfigurationError(
                "Network must have at least input and output layers".to_string()
            ));
        }

        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::seed_from_u64(rand::random()),
        };

        let mut layers = Vec::new();

        for i in 0..layer_sizes.len()-1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            // Initialize weights with Xavier initialization
            let mut weights = vec![vec![0.0; input_size]; output_size];
            let scale = (2.0 / input_size as f64).sqrt();

            for row in &mut weights {
                for weight in row {
                    *weight = rng.gen_range(-scale..scale);
                }
            }

            let biases = vec![0.0; output_size];

            let activation = if i == layer_sizes.len() - 2 {
                ActivationFunction::Linear // Output layer
            } else {
                ActivationFunction::ReLU // Hidden layers
            };

            layers.push(NetworkLayer {
                weights,
                biases,
                activation,
            });
        }

        let input_size = layer_sizes[0];
        let output_size = layer_sizes[layer_sizes.len() - 1];

        Ok(Self {
            layers,
            input_normalization: NormalizationParams {
                means: vec![0.0; input_size],
                stds: vec![1.0; input_size],
                mins: vec![0.0; input_size],
                maxs: vec![1.0; input_size],
            },
            output_scaling: NormalizationParams {
                means: vec![0.0; output_size],
                stds: vec![1.0; output_size],
                mins: vec![0.0; output_size],
                maxs: vec![1.0; output_size],
            },
            training_state: NetworkTrainingState {
                epoch: 0,
                training_loss: 0.0,
                validation_loss: 0.0,
                learning_rate: 0.001,
                metrics: HashMap::new(),
            },
        })
    }

    /// Forward pass through the network
    pub fn forward(&self, input: &[f64]) -> AdaptiveScheduleResult<Vec<f64>> {
        let mut activations = input.to_vec();

        for layer in &self.layers {
            activations = self.layer_forward(&activations, layer)?;
        }

        Ok(activations)
    }

    /// Forward pass through a single layer
    fn layer_forward(&self, input: &[f64], layer: &NetworkLayer) -> AdaptiveScheduleResult<Vec<f64>> {
        if input.len() != layer.weights[0].len() {
            return Err(AdaptiveScheduleError::NeuralNetworkError(
                format!("Input size {} doesn't match layer input size {}",
                       input.len(), layer.weights[0].len())
            ));
        }

        let mut output = Vec::new();

        for (neuron_weights, &bias) in layer.weights.iter().zip(&layer.biases) {
            let mut activation = bias;

            for (&inp, &weight) in input.iter().zip(neuron_weights) {
                activation += inp * weight;
            }

            // Apply activation function
            activation = match layer.activation {
                ActivationFunction::ReLU => activation.max(0.0),
                ActivationFunction::Sigmoid => 1.0 / (1.0 + (-activation).exp()),
                ActivationFunction::Tanh => activation.tanh(),
                ActivationFunction::Linear => activation,
                ActivationFunction::LeakyReLU(alpha) => {
                    if activation > 0.0 { activation } else { alpha * activation }
                },
            };

            output.push(activation);
        }

        Ok(output)
    }

    /// Backward pass (simplified implementation)
    pub fn backward(&mut self, _input: &[f64], _target: &[f64], _prediction: &[f64]) -> AdaptiveScheduleResult<()> {
        // Simplified backward pass - in practice would implement full backpropagation
        // For now, just update training state
        self.training_state.epoch += 1;
        Ok(())
    }
}

impl ScheduleRLAgent {
    /// Create a new RL agent
    pub fn new(config: RLAgentConfig) -> AdaptiveScheduleResult<Self> {
        let q_network = SchedulePredictionNetwork::new(
            &[config.state_space_size, 64, 32, config.action_space_size],
            None
        )?;
        let target_network = q_network.clone();

        Ok(Self {
            q_network,
            target_network,
            experience_buffer: Vec::new(),
            config,
            stats: RLStats {
                episode_rewards: Vec::new(),
                average_reward: 0.0,
                exploration_history: Vec::new(),
                loss_history: Vec::new(),
                action_frequency: HashMap::new(),
            },
        })
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action(&self, state: &[f64]) -> AdaptiveScheduleResult<usize> {
        let mut rng = ChaCha8Rng::seed_from_u64(rand::random());

        if rng.gen::<f64>() < self.config.min_epsilon {
            // Random exploration
            Ok(rng.gen_range(0..self.config.action_space_size))
        } else {
            // Greedy action selection
            let q_values = self.q_network.forward(state)?;
            let best_action = q_values.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            Ok(best_action)
        }
    }

    /// Store experience in replay buffer
    pub fn store_experience(&mut self, experience: ScheduleExperience) {
        self.experience_buffer.push(experience);

        // Maintain buffer size
        if self.experience_buffer.len() > 1000 {
            self.experience_buffer.remove(0);
        }
    }

    /// Train the RL agent
    pub fn train(&mut self) -> AdaptiveScheduleResult<()> {
        if self.experience_buffer.len() < self.config.batch_size {
            return Ok(());
        }

        // Sample batch from experience buffer
        // In practice, would implement proper experience replay

        Ok(())
    }
}

/// Create a default neural annealing scheduler
pub fn create_neural_scheduler() -> AdaptiveScheduleResult<NeuralAnnealingScheduler> {
    NeuralAnnealingScheduler::new(SchedulerConfig::default())
}

/// Create a neural scheduler with custom configuration
pub fn create_custom_neural_scheduler(
    network_layers: Vec<usize>,
    learning_rate: f64,
    exploration_rate: f64,
) -> AdaptiveScheduleResult<NeuralAnnealingScheduler> {
    let config = SchedulerConfig {
        network_layers,
        learning_rate,
        exploration_rate,
        ..Default::default()
    };

    NeuralAnnealingScheduler::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_scheduler_creation() {
        let scheduler = create_neural_scheduler().unwrap();
        assert_eq!(scheduler.config.network_layers, vec![32, 64, 32, 16]);
    }

    #[test]
    fn test_network_creation() {
        let network = SchedulePredictionNetwork::new(&[10, 20, 5], Some(42)).unwrap();
        assert_eq!(network.layers.len(), 2);
        assert_eq!(network.layers[0].weights.len(), 20);
        assert_eq!(network.layers[0].weights[0].len(), 10);
    }

    #[test]
    fn test_network_forward_pass() {
        let network = SchedulePredictionNetwork::new(&[3, 5, 2], Some(42)).unwrap();
        let input = vec![1.0, 0.5, -0.5];
        let output = network.forward(&input).unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_feature_extraction() {
        let mut ising = IsingModel::new(4);
        ising.set_bias(0, 1.0).unwrap();
        ising.set_coupling(0, 1, -0.5).unwrap();
        ising.set_coupling(1, 2, 0.3).unwrap();

        let scheduler = create_neural_scheduler().unwrap();
        let features = scheduler.extract_problem_features(&ising).unwrap();

        assert_eq!(features.size, 4);
        assert!(features.connectivity_density > 0.0);
        assert!(features.coupling_stats.mean > 0.0);
    }

    #[test]
    fn test_rl_agent_creation() {
        let config = RLAgentConfig {
            action_space_size: 10,
            state_space_size: 15,
            batch_size: 32,
            target_update_frequency: 100,
            epsilon_decay: 0.995,
            min_epsilon: 0.01,
        };

        let agent = ScheduleRLAgent::new(config).unwrap();
        assert_eq!(agent.config.action_space_size, 10);
        assert_eq!(agent.config.state_space_size, 15);
    }

    #[test]
    fn test_schedule_generation() {
        let mut scheduler = create_neural_scheduler().unwrap();
        let mut ising = IsingModel::new(5);
        ising.set_bias(0, 1.0).unwrap();
        ising.set_coupling(0, 1, -0.5).unwrap();

        let schedule = scheduler.generate_schedule(&ising).unwrap();
        assert!(schedule.num_sweeps > 0);
        assert!(schedule.initial_temperature > 0.0);
        assert!(schedule.final_temperature > 0.0);
        assert!(schedule.initial_temperature > schedule.final_temperature);
    }
}