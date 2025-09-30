//! Neural network implementations for RL embedding optimization

use scirs2_core::random::prelude::*;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::random::ChaCha8Rng;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::error::{RLEmbeddingError, RLEmbeddingResult};

/// Deep Q-Network for embedding decisions
#[derive(Debug, Clone)]
pub struct EmbeddingDQN {
    /// Main Q-network
    pub q_network: EmbeddingNetwork,
    /// Target Q-network for stable training
    pub target_network: EmbeddingNetwork,
    /// Network configuration
    pub config: NetworkConfig,
    /// Training state
    pub training_state: NetworkTrainingState,
}

/// Policy network for continuous embedding optimization
#[derive(Debug, Clone)]
pub struct EmbeddingPolicyNetwork {
    /// Actor network (policy)
    pub actor_network: EmbeddingNetwork,
    /// Critic network (value function)
    pub critic_network: EmbeddingNetwork,
    /// Network configuration
    pub config: NetworkConfig,
    /// Training state
    pub training_state: NetworkTrainingState,
}

/// Neural network for embedding optimization
#[derive(Debug, Clone)]
pub struct EmbeddingNetwork {
    /// Network layers
    pub layers: Vec<NetworkLayer>,
    /// Input normalization
    pub input_norm: NormalizationLayer,
    /// Output scaling
    pub output_scaling: NormalizationLayer,
    /// Network metadata
    pub metadata: NetworkMetadata,
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
    /// Dropout rate
    pub dropout_rate: f64,
    /// Batch normalization parameters
    pub batch_norm: Option<BatchNormalization>,
}

/// Activation functions
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationFunction {
    /// ReLU activation
    ReLU,
    /// Leaky ReLU
    LeakyReLU(f64),
    /// Tanh activation
    Tanh,
    /// Sigmoid activation
    Sigmoid,
    /// Swish activation
    Swish,
    /// Linear activation
    Linear,
}

/// Batch normalization parameters
#[derive(Debug, Clone)]
pub struct BatchNormalization {
    /// Running mean
    pub running_mean: Vec<f64>,
    /// Running variance
    pub running_var: Vec<f64>,
    /// Learnable scale parameter
    pub gamma: Vec<f64>,
    /// Learnable shift parameter
    pub beta: Vec<f64>,
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Momentum for running statistics
    pub momentum: f64,
}

/// Normalization layer
#[derive(Debug, Clone)]
pub struct NormalizationLayer {
    /// Mean values
    pub means: Vec<f64>,
    /// Standard deviations
    pub stds: Vec<f64>,
    /// Min values
    pub mins: Vec<f64>,
    /// Max values
    pub maxs: Vec<f64>,
    /// Normalization type
    pub norm_type: NormalizationType,
}

/// Types of normalization
#[derive(Debug, Clone, PartialEq)]
pub enum NormalizationType {
    /// Z-score normalization
    StandardScore,
    /// Min-max normalization
    MinMax,
    /// Robust normalization (median, IQR)
    Robust,
    /// No normalization
    None,
}

/// Network configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Layer sizes
    pub layer_sizes: Vec<usize>,
    /// Learning rate
    pub learning_rate: f64,
    /// Regularization parameters
    pub regularization: RegularizationConfig,
    /// Optimization method
    pub optimizer: OptimizerType,
    /// Loss function
    pub loss_function: LossFunction,
}

/// Regularization configuration
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// L1 regularization strength
    pub l1_strength: f64,
    /// L2 regularization strength
    pub l2_strength: f64,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Early stopping patience
    pub early_stopping_patience: usize,
}

/// Optimizer types
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD,
    /// Adam optimizer
    Adam { beta1: f64, beta2: f64 },
    /// RMSprop optimizer
    RMSprop { decay_rate: f64 },
    /// AdaGrad optimizer
    AdaGrad,
}

/// Loss functions
#[derive(Debug, Clone, PartialEq)]
pub enum LossFunction {
    /// Mean Squared Error
    MSE,
    /// Huber loss
    Huber { delta: f64 },
    /// Cross-entropy loss
    CrossEntropy,
    /// Custom multi-objective loss
    MultiObjective,
}

/// Network metadata
#[derive(Debug, Clone)]
pub struct NetworkMetadata {
    /// Creation timestamp
    pub created_at: Instant,
    /// Training history
    pub training_history: Vec<TrainingEpoch>,
    /// Performance metrics
    pub performance_metrics: NetworkPerformanceMetrics,
    /// Model version
    pub version: String,
}

/// Training epoch information
#[derive(Debug, Clone)]
pub struct TrainingEpoch {
    /// Epoch number
    pub epoch: usize,
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Training time
    pub duration: Duration,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Network performance metrics
#[derive(Debug, Clone)]
pub struct NetworkPerformanceMetrics {
    /// Best validation loss achieved
    pub best_validation_loss: f64,
    /// Training convergence rate
    pub convergence_rate: f64,
    /// Generalization gap
    pub generalization_gap: f64,
    /// Parameter efficiency
    pub parameter_efficiency: f64,
}

/// Network training state
#[derive(Debug, Clone)]
pub struct NetworkTrainingState {
    /// Current epoch
    pub current_epoch: usize,
    /// Current learning rate
    pub current_lr: f64,
    /// Optimizer state
    pub optimizer_state: OptimizerState,
    /// Best model weights
    pub best_weights: Option<Vec<Vec<Vec<f64>>>>,
    /// Early stopping counter
    pub early_stopping_counter: usize,
}

/// Optimizer state
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// Momentum buffers (for SGD with momentum)
    pub momentum_buffers: Vec<Vec<Vec<f64>>>,
    /// First moment estimates (for Adam)
    pub first_moments: Vec<Vec<Vec<f64>>>,
    /// Second moment estimates (for Adam)
    pub second_moments: Vec<Vec<Vec<f64>>>,
    /// Iteration counter
    pub iteration: usize,
}

impl EmbeddingDQN {
    /// Create new DQN
    pub fn new(layer_sizes: &[usize], seed: Option<u64>) -> RLEmbeddingResult<Self> {
        let q_network = EmbeddingNetwork::new(layer_sizes, seed)?;
        let target_network = q_network.clone();

        let config = NetworkConfig {
            layer_sizes: layer_sizes.to_vec(),
            learning_rate: 0.001,
            regularization: RegularizationConfig {
                l1_strength: 0.0001,
                l2_strength: 0.001,
                dropout_rate: 0.1,
                early_stopping_patience: 100,
            },
            optimizer: OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
            },
            loss_function: LossFunction::MSE,
        };

        let training_state = NetworkTrainingState {
            current_epoch: 0,
            current_lr: 0.001,
            optimizer_state: OptimizerState {
                momentum_buffers: Vec::new(),
                first_moments: Vec::new(),
                second_moments: Vec::new(),
                iteration: 0,
            },
            best_weights: None,
            early_stopping_counter: 0,
        };

        Ok(Self {
            q_network,
            target_network,
            config,
            training_state,
        })
    }
}

impl EmbeddingPolicyNetwork {
    /// Create new policy network
    pub fn new(layer_sizes: &[usize], seed: Option<u64>) -> RLEmbeddingResult<Self> {
        let actor_network = EmbeddingNetwork::new(layer_sizes, seed)?;
        let critic_network = EmbeddingNetwork::new(layer_sizes, seed)?;

        let config = NetworkConfig {
            layer_sizes: layer_sizes.to_vec(),
            learning_rate: 0.0001,
            regularization: RegularizationConfig {
                l1_strength: 0.0001,
                l2_strength: 0.001,
                dropout_rate: 0.1,
                early_stopping_patience: 100,
            },
            optimizer: OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
            },
            loss_function: LossFunction::MultiObjective,
        };

        let training_state = NetworkTrainingState {
            current_epoch: 0,
            current_lr: 0.0001,
            optimizer_state: OptimizerState {
                momentum_buffers: Vec::new(),
                first_moments: Vec::new(),
                second_moments: Vec::new(),
                iteration: 0,
            },
            best_weights: None,
            early_stopping_counter: 0,
        };

        Ok(Self {
            actor_network,
            critic_network,
            config,
            training_state,
        })
    }
}

impl EmbeddingNetwork {
    /// Create new neural network
    pub fn new(layer_sizes: &[usize], seed: Option<u64>) -> RLEmbeddingResult<Self> {
        if layer_sizes.len() < 2 {
            return Err(RLEmbeddingError::ConfigurationError(
                "Network must have at least input and output layers".to_string(),
            ));
        }

        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::seed_from_u64(thread_rng().gen()),
        };

        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            // Xavier initialization
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
                dropout_rate: 0.1,
                batch_norm: None,
            });
        }

        let input_size = layer_sizes[0];
        let output_size = layer_sizes[layer_sizes.len() - 1];

        let input_norm = NormalizationLayer {
            means: vec![0.0; input_size],
            stds: vec![1.0; input_size],
            mins: vec![0.0; input_size],
            maxs: vec![1.0; input_size],
            norm_type: NormalizationType::StandardScore,
        };

        let output_scaling = NormalizationLayer {
            means: vec![0.0; output_size],
            stds: vec![1.0; output_size],
            mins: vec![0.0; output_size],
            maxs: vec![1.0; output_size],
            norm_type: NormalizationType::None,
        };

        let metadata = NetworkMetadata {
            created_at: Instant::now(),
            training_history: Vec::new(),
            performance_metrics: NetworkPerformanceMetrics {
                best_validation_loss: f64::INFINITY,
                convergence_rate: 0.0,
                generalization_gap: 0.0,
                parameter_efficiency: 0.0,
            },
            version: "1.0.0".to_string(),
        };

        Ok(Self {
            layers,
            input_norm,
            output_scaling,
            metadata,
        })
    }

    /// Forward pass through network
    pub fn forward(&self, input: &[f64]) -> RLEmbeddingResult<Vec<f64>> {
        let mut activations = input.to_vec();

        for layer in &self.layers {
            activations = self.layer_forward(&activations, layer)?;
        }

        Ok(activations)
    }

    /// Forward pass through single layer
    fn layer_forward(&self, input: &[f64], layer: &NetworkLayer) -> RLEmbeddingResult<Vec<f64>> {
        if input.len() != layer.weights[0].len() {
            return Err(RLEmbeddingError::NeuralNetworkError(format!(
                "Input size {} doesn't match layer input size {}",
                input.len(),
                layer.weights[0].len()
            )));
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
                ActivationFunction::LeakyReLU(alpha) => {
                    if activation > 0.0 {
                        activation
                    } else {
                        alpha * activation
                    }
                }
                ActivationFunction::Tanh => activation.tanh(),
                ActivationFunction::Sigmoid => 1.0 / (1.0 + (-activation).exp()),
                ActivationFunction::Swish => activation / (1.0 + (-activation).exp()),
                ActivationFunction::Linear => activation,
            };

            output.push(activation);
        }

        Ok(output)
    }
}
