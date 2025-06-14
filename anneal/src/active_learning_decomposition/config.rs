//! Configuration types for active learning decomposition

use std::time::Duration;

/// Configuration for active learning decomposition
#[derive(Debug, Clone)]
pub struct ActiveLearningConfig {
    /// Enable online learning
    pub enable_online_learning: bool,
    /// Maximum decomposition depth
    pub max_decomposition_depth: usize,
    /// Minimum subproblem size
    pub min_subproblem_size: usize,
    /// Maximum subproblem size
    pub max_subproblem_size: usize,
    /// Learning rate for strategy updates
    pub learning_rate: f64,
    /// Exploration rate for active learning
    pub exploration_rate: f64,
    /// Performance threshold for decomposition
    pub performance_threshold: f64,
    /// Enable transfer learning
    pub enable_transfer_learning: bool,
    /// Active learning query budget
    pub query_budget: usize,
    /// Decomposition overlap tolerance
    pub overlap_tolerance: f64,
}

impl Default for ActiveLearningConfig {
    fn default() -> Self {
        Self {
            enable_online_learning: true,
            max_decomposition_depth: 3,
            min_subproblem_size: 2,
            max_subproblem_size: 100,
            learning_rate: 0.01,
            exploration_rate: 0.1,
            performance_threshold: 0.7,
            enable_transfer_learning: true,
            query_budget: 100,
            overlap_tolerance: 0.1,
        }
    }
}

/// Metric computation configuration
#[derive(Debug, Clone)]
pub struct MetricComputationConfig {
    /// Enable expensive metrics
    pub enable_expensive_metrics: bool,
    /// Approximation algorithms enabled
    pub enable_approximation: bool,
    /// Sampling ratio for large graphs
    pub sampling_ratio: f64,
    /// Timeout for metric computation
    pub computation_timeout: Duration,
}

impl Default for MetricComputationConfig {
    fn default() -> Self {
        Self {
            enable_expensive_metrics: false,
            enable_approximation: true,
            sampling_ratio: 0.1,
            computation_timeout: Duration::from_secs(60),
        }
    }
}

/// Size constraints
#[derive(Debug, Clone)]
pub struct SizeConstraints {
    /// Minimum subproblem size
    pub min_size: usize,
    /// Maximum subproblem size
    pub max_size: usize,
    /// Target size
    pub target_size: usize,
    /// Size tolerance
    pub size_tolerance: f64,
}

impl Default for SizeConstraints {
    fn default() -> Self {
        Self {
            min_size: 2,
            max_size: 100,
            target_size: 20,
            size_tolerance: 0.2,
        }
    }
}

/// Resource constraints
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum computation time
    pub max_computation_time: Duration,
    /// Maximum memory usage
    pub max_memory_usage: usize,
    /// Number of available processors
    pub num_processors: usize,
    /// Communication bandwidth
    pub communication_bandwidth: f64,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_computation_time: Duration::from_secs(300),
            max_memory_usage: 1024 * 1024 * 1024, // 1 GB
            num_processors: 4,
            communication_bandwidth: 1000.0,
        }
    }
}