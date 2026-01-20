//! Parallel Configuration Types

/// Parallel optimization configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Enable parallel evaluation
    pub enabled: bool,
    /// Number of parallel workers
    pub num_workers: usize,
    /// Batch size for parallel evaluation
    pub batch_size: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_workers: 4,
            batch_size: 4,
            load_balancing: LoadBalancingStrategy::RoundRobin,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Work-stealing approach
    WorkStealing,
    /// Dynamic load balancing
    Dynamic,
}
