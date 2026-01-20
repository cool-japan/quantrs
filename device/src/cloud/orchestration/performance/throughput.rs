//! Throughput optimization configurations

use super::scaling::ResourceScalingConfig;
use serde::{Deserialize, Serialize};

/// Throughput optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputOptimizationConfig {
    /// Target throughput
    pub target_throughput: f64,
    /// Optimization techniques
    pub techniques: Vec<ThroughputOptimizationTechnique>,
    /// Parallelization configuration
    pub parallelization: ParallelizationConfig,
    /// Resource scaling
    pub scaling: ResourceScalingConfig,
}

/// Throughput optimization techniques
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThroughputOptimizationTechnique {
    ParallelProcessing,
    LoadBalancing,
    ResourceScaling,
    Caching,
    Pipelining,
    Custom(String),
}

/// Parallelization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelizationConfig {
    /// Max parallel jobs
    pub max_parallel_jobs: usize,
    /// Worker pool size
    pub worker_pool_size: usize,
    /// Work distribution strategy
    pub distribution_strategy: WorkDistributionStrategy,
}

/// Work distribution strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkDistributionStrategy {
    RoundRobin,
    LeastLoaded,
    WorkStealing,
    Random,
    Custom(String),
}
