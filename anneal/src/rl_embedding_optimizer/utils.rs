//! Utility functions and helper methods for RL embedding optimization

use super::error::{RLEmbeddingError, RLEmbeddingResult};
use super::types::{
    ExperienceContext, ExplorationConfig, ObjectiveWeights, RLEmbeddingConfig,
    TransferLearningConfig,
};

/// Create default RL embedding optimizer configuration
#[must_use]
pub fn create_default_config() -> RLEmbeddingConfig {
    RLEmbeddingConfig::default()
}

/// Create custom RL embedding optimizer configuration
#[must_use]
pub fn create_custom_config(
    dqn_layers: Vec<usize>,
    policy_layers: Vec<usize>,
    learning_rate: f64,
) -> RLEmbeddingConfig {
    RLEmbeddingConfig {
        dqn_layers,
        policy_layers,
        learning_rate,
        ..Default::default()
    }
}

/// Create exploration configuration for different strategies
#[must_use]
pub fn create_exploration_config(strategy: ExplorationStrategy) -> ExplorationConfig {
    match strategy {
        ExplorationStrategy::Conservative => ExplorationConfig {
            initial_epsilon: 0.5,
            final_epsilon: 0.05,
            epsilon_decay_steps: 5000,
            policy_noise: 0.05,
            curiosity_weight: 0.05,
        },
        ExplorationStrategy::Moderate => ExplorationConfig::default(),
        ExplorationStrategy::Aggressive => ExplorationConfig {
            initial_epsilon: 1.0,
            final_epsilon: 0.01,
            epsilon_decay_steps: 20_000,
            policy_noise: 0.2,
            curiosity_weight: 0.2,
        },
    }
}

/// Exploration strategies
pub enum ExplorationStrategy {
    Conservative,
    Moderate,
    Aggressive,
}

/// Create objective weights for different optimization goals
#[must_use]
pub fn create_objective_weights(goal: OptimizationGoal) -> ObjectiveWeights {
    match goal {
        OptimizationGoal::MinimizeChainLength => ObjectiveWeights {
            chain_length_weight: 0.6,
            efficiency_weight: 0.2,
            utilization_weight: 0.1,
            connectivity_weight: 0.05,
            performance_weight: 0.05,
        },
        OptimizationGoal::MaximizeEfficiency => ObjectiveWeights {
            chain_length_weight: 0.1,
            efficiency_weight: 0.5,
            utilization_weight: 0.2,
            connectivity_weight: 0.1,
            performance_weight: 0.1,
        },
        OptimizationGoal::MaximizeUtilization => ObjectiveWeights {
            chain_length_weight: 0.1,
            efficiency_weight: 0.2,
            utilization_weight: 0.5,
            connectivity_weight: 0.1,
            performance_weight: 0.1,
        },
        OptimizationGoal::Balanced => ObjectiveWeights::default(),
    }
}

/// Optimization goals
pub enum OptimizationGoal {
    MinimizeChainLength,
    MaximizeEfficiency,
    MaximizeUtilization,
    Balanced,
}

/// Create transfer learning configuration for different scenarios
#[must_use]
pub const fn create_transfer_learning_config(scenario: TransferScenario) -> TransferLearningConfig {
    match scenario {
        TransferScenario::SimilarProblems => TransferLearningConfig {
            enabled: true,
            source_weight_decay: 0.95,
            adaptation_lr: 0.00_005,
            fine_tuning_epochs: 50,
            similarity_threshold: 0.8,
        },
        TransferScenario::DifferentProblems => TransferLearningConfig {
            enabled: true,
            source_weight_decay: 0.8,
            adaptation_lr: 0.0001,
            fine_tuning_epochs: 200,
            similarity_threshold: 0.5,
        },
        TransferScenario::NoTransfer => TransferLearningConfig {
            enabled: false,
            source_weight_decay: 0.0,
            adaptation_lr: 0.0,
            fine_tuning_epochs: 0,
            similarity_threshold: 0.0,
        },
    }
}

/// Transfer learning scenarios
pub enum TransferScenario {
    SimilarProblems,
    DifferentProblems,
    NoTransfer,
}

/// Validate configuration parameters
pub fn validate_config(config: &RLEmbeddingConfig) -> RLEmbeddingResult<()> {
    // Validate DQN layers
    if config.dqn_layers.len() < 2 {
        return Err(RLEmbeddingError::ConfigurationError(
            "DQN must have at least input and output layers".to_string(),
        ));
    }

    // Validate policy layers
    if config.policy_layers.len() < 2 {
        return Err(RLEmbeddingError::ConfigurationError(
            "Policy network must have at least input and output layers".to_string(),
        ));
    }

    // Validate learning rate
    if config.learning_rate <= 0.0 || config.learning_rate > 1.0 {
        return Err(RLEmbeddingError::ConfigurationError(
            "Learning rate must be between 0 and 1".to_string(),
        ));
    }

    // Validate buffer size
    if config.buffer_size == 0 {
        return Err(RLEmbeddingError::ConfigurationError(
            "Buffer size must be greater than 0".to_string(),
        ));
    }

    // Validate batch size
    if config.batch_size == 0 || config.batch_size > config.buffer_size {
        return Err(RLEmbeddingError::ConfigurationError(
            "Batch size must be between 1 and buffer size".to_string(),
        ));
    }

    // Validate discount factor
    if config.discount_factor < 0.0 || config.discount_factor > 1.0 {
        return Err(RLEmbeddingError::ConfigurationError(
            "Discount factor must be between 0 and 1".to_string(),
        ));
    }

    // Validate exploration config
    if config.exploration_config.initial_epsilon < 0.0
        || config.exploration_config.initial_epsilon > 1.0
    {
        return Err(RLEmbeddingError::ConfigurationError(
            "Initial epsilon must be between 0 and 1".to_string(),
        ));
    }

    if config.exploration_config.final_epsilon < 0.0
        || config.exploration_config.final_epsilon > 1.0
    {
        return Err(RLEmbeddingError::ConfigurationError(
            "Final epsilon must be between 0 and 1".to_string(),
        ));
    }

    if config.exploration_config.epsilon_decay_steps == 0 {
        return Err(RLEmbeddingError::ConfigurationError(
            "Epsilon decay steps must be greater than 0".to_string(),
        ));
    }

    // Validate objective weights (should sum to approximately 1.0)
    let weight_sum = config.objective_weights.chain_length_weight
        + config.objective_weights.efficiency_weight
        + config.objective_weights.utilization_weight
        + config.objective_weights.connectivity_weight
        + config.objective_weights.performance_weight;

    if (weight_sum - 1.0).abs() > 0.1 {
        return Err(RLEmbeddingError::ConfigurationError(
            "Objective weights should sum to approximately 1.0".to_string(),
        ));
    }

    Ok(())
}

/// Calculate memory usage estimate for configuration
#[must_use]
pub fn estimate_memory_usage(config: &RLEmbeddingConfig) -> usize {
    let mut memory = 0;

    // DQN memory
    for i in 0..config.dqn_layers.len() - 1 {
        let weights = config.dqn_layers[i] * config.dqn_layers[i + 1];
        let biases = config.dqn_layers[i + 1];
        memory += (weights + biases) * std::mem::size_of::<f64>();
    }

    // Policy network memory (actor + critic)
    for i in 0..config.policy_layers.len() - 1 {
        let weights = config.policy_layers[i] * config.policy_layers[i + 1];
        let biases = config.policy_layers[i + 1];
        memory += 2 * (weights + biases) * std::mem::size_of::<f64>(); // actor + critic
    }

    // Experience buffer memory (estimate)
    let experience_size = 1000; // Rough estimate per experience
    memory += config.buffer_size * experience_size;

    memory
}

/// Create configuration summary string
#[must_use]
pub fn config_summary(config: &RLEmbeddingConfig) -> String {
    format!(
        "RL Embedding Optimizer Configuration:\n\
         DQN Architecture: {:?}\n\
         Policy Architecture: {:?}\n\
         Learning Rate: {:.6}\n\
         Buffer Size: {}\n\
         Batch Size: {}\n\
         Discount Factor: {:.3}\n\
         Exploration: ε={:.3} → {:.3} over {} steps\n\
         Transfer Learning: {}\n\
         Estimated Memory: {:.1} MB",
        config.dqn_layers,
        config.policy_layers,
        config.learning_rate,
        config.buffer_size,
        config.batch_size,
        config.discount_factor,
        config.exploration_config.initial_epsilon,
        config.exploration_config.final_epsilon,
        config.exploration_config.epsilon_decay_steps,
        if config.transfer_learning.enabled {
            "Enabled"
        } else {
            "Disabled"
        },
        estimate_memory_usage(config) as f64 / (1024.0 * 1024.0)
    )
}

/// Helper function to create experience context
#[must_use]
pub fn create_experience_context(
    problem_type: String,
    hardware_id: String,
    episode_id: usize,
    step: usize,
) -> ExperienceContext {
    ExperienceContext {
        problem_type,
        hardware_id,
        timestamp: std::time::Instant::now(),
        episode_id,
        step,
        metadata: std::collections::HashMap::new(),
    }
}

/// Helper function to format hardware topology as string
#[must_use]
pub fn hardware_topology_to_string(hardware: &crate::embedding::HardwareTopology) -> String {
    match hardware {
        crate::embedding::HardwareTopology::Chimera(m, n, t) => {
            format!("Chimera_{m}x{n}x{t}")
        }
        crate::embedding::HardwareTopology::Pegasus(n) => {
            format!("Pegasus_{n}")
        }
        crate::embedding::HardwareTopology::Zephyr(n) => {
            format!("Zephyr_{n}")
        }
        crate::embedding::HardwareTopology::Custom => "Custom".to_string(),
    }
}
