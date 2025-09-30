//! Configuration types for quantum continual learning

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Continual learning strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContinualLearningStrategy {
    /// Elastic Weight Consolidation
    EWC,
    /// Synaptic Intelligence
    SI,
    /// Memory Aware Synapses
    MAS,
    /// Progressive Neural Networks
    Progressive,
    /// PackNet
    PackNet,
    /// Gradient Episodic Memory
    GEM,
    /// Averaged Gradient Episodic Memory
    AGEM,
    /// Experience Replay
    ExperienceReplay,
    /// Quantum Memory Consolidation
    QuantumMemoryConsolidation,
}

/// Memory types for continual learning
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryType {
    /// Episodic memory
    Episodic,
    /// Semantic memory
    Semantic,
    /// Working memory
    Working,
    /// Quantum memory
    Quantum,
}

/// Task types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskType {
    /// Classification task
    Classification,
    /// Regression task
    Regression,
    /// Reinforcement learning task
    ReinforcementLearning,
    /// Unsupervised learning task
    Unsupervised,
    /// Quantum task
    Quantum,
}

/// Configuration for quantum continual learning
#[derive(Debug, Clone)]
pub struct QuantumContinualLearningConfig {
    /// Primary learning strategy
    pub strategy: ContinualLearningStrategy,
    /// Memory types to use
    pub memory_types: Vec<MemoryType>,
    /// Maximum number of tasks
    pub max_tasks: usize,
    /// Memory capacity
    pub memory_capacity: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Regularization strength
    pub regularization_strength: f64,
    /// Number of qubits for quantum operations
    pub num_qubits: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

/// Task configuration
#[derive(Debug, Clone)]
pub struct ContinualTask {
    /// Task identifier
    pub task_id: usize,
    /// Task type
    pub task_type: TaskType,
    /// Task name
    pub name: String,
    /// Task description
    pub description: String,
    /// Task-specific parameters
    pub parameters: HashMap<String, f64>,
}

/// Memory configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Memory type
    pub memory_type: MemoryType,
    /// Capacity
    pub capacity: usize,
    /// Retention strategy
    pub retention_strategy: String,
    /// Quantum enhancement level
    pub quantum_enhancement: f64,
}

/// Evaluation configuration
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Metrics to compute
    pub metrics: Vec<String>,
    /// Evaluation frequency
    pub eval_frequency: usize,
    /// Use validation set
    pub use_validation: bool,
    /// Forgetting measurement
    pub measure_forgetting: bool,
}

impl Default for QuantumContinualLearningConfig {
    fn default() -> Self {
        Self {
            strategy: ContinualLearningStrategy::EWC,
            memory_types: vec![MemoryType::Episodic, MemoryType::Semantic],
            max_tasks: 10,
            memory_capacity: 1000,
            learning_rate: 0.001,
            regularization_strength: 0.1,
            num_qubits: 4,
            random_state: None,
        }
    }
}