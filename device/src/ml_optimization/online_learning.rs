//! Online Learning Configuration Types

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Online learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineLearningConfig {
    /// Enable online learning
    pub enable_online_learning: bool,
    /// Learning rate schedule
    pub learning_rate_schedule: LearningRateSchedule,
    /// Memory management
    pub memory_management: MemoryManagementConfig,
    /// Catastrophic forgetting prevention
    pub forgetting_prevention: ForgettingPreventionConfig,
    /// Incremental learning
    pub incremental_learning: IncrementalLearningConfig,
}

/// Learning rate schedule
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    Constant,
    ExponentialDecay,
    StepDecay,
    PolynomialDecay,
    CosineAnnealing,
    Adaptive,
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryManagementConfig {
    /// Maximum memory buffer size
    pub max_buffer_size: usize,
    /// Memory eviction strategy
    pub eviction_strategy: MemoryEvictionStrategy,
    /// Replay buffer management
    pub replay_buffer: bool,
    /// Experience prioritization
    pub experience_prioritization: bool,
}

/// Memory eviction strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryEvictionStrategy {
    FIFO,
    LRU,
    LFU,
    RandomEviction,
    ImportanceBased,
    RecentnessBased,
}

/// Forgetting prevention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgettingPreventionConfig {
    /// Enable elastic weight consolidation
    pub elastic_weight_consolidation: bool,
    /// Enable progressive neural networks
    pub progressive_networks: bool,
    /// Enable memory replay
    pub memory_replay: bool,
    /// Regularization strength
    pub regularization_strength: f64,
}

/// Incremental learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalLearningConfig {
    /// Batch size for incremental updates
    pub incremental_batch_size: usize,
    /// Update frequency
    pub update_frequency: Duration,
    /// Stability-plasticity balance
    pub stability_plasticity_balance: f64,
    /// Knowledge distillation
    pub knowledge_distillation: bool,
}
