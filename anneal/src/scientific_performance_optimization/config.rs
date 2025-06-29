//! Configuration types for scientific performance optimization
//!
//! This module contains all configuration structures and enums used throughout
//! the scientific performance optimization system.

use std::collections::HashMap;
use std::time::Duration;

/// Performance optimization configuration
#[derive(Debug, Clone)]
pub struct PerformanceOptimizationConfig {
    /// Memory management settings
    pub memory_config: MemoryOptimizationConfig,
    /// Parallel processing configuration
    pub parallel_config: ParallelProcessingConfig,
    /// Algorithm optimization settings
    pub algorithm_config: AlgorithmOptimizationConfig,
    /// Distributed computing configuration
    pub distributed_config: DistributedComputingConfig,
    /// Profiling and monitoring settings
    pub profiling_config: ProfilingConfig,
    /// GPU acceleration settings
    pub gpu_config: GPUAccelerationConfig,
}

impl Default for PerformanceOptimizationConfig {
    fn default() -> Self {
        Self {
            memory_config: MemoryOptimizationConfig::default(),
            parallel_config: ParallelProcessingConfig::default(),
            algorithm_config: AlgorithmOptimizationConfig::default(),
            distributed_config: DistributedComputingConfig::default(),
            profiling_config: ProfilingConfig::default(),
            gpu_config: GPUAccelerationConfig::default(),
        }
    }
}

/// Memory optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryOptimizationConfig {
    /// Enable hierarchical memory management
    pub enable_hierarchical_memory: bool,
    /// Cache size limits (in MB)
    pub cache_size_limit: usize,
    /// Memory pool configuration
    pub memory_pool_config: MemoryPoolConfig,
    /// Enable memory-mapped I/O
    pub enable_memory_mapping: bool,
    /// Compression settings
    pub compression_config: CompressionConfig,
    /// Garbage collection strategy
    pub gc_strategy: GarbageCollectionStrategy,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_hierarchical_memory: true,
            cache_size_limit: 8192, // 8GB
            memory_pool_config: MemoryPoolConfig::default(),
            enable_memory_mapping: true,
            compression_config: CompressionConfig::default(),
            gc_strategy: GarbageCollectionStrategy::Adaptive,
        }
    }
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Pool size in MB
    pub pool_size: usize,
    /// Block sizes for different allocations
    pub block_sizes: Vec<usize>,
    /// Enable pool preallocation
    pub enable_preallocation: bool,
    /// Pool growth strategy
    pub growth_strategy: PoolGrowthStrategy,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            pool_size: 4096,                               // 4GB
            block_sizes: vec![64, 256, 1024, 4096, 16384], // Various block sizes
            enable_preallocation: true,
            growth_strategy: PoolGrowthStrategy::Exponential,
        }
    }
}

/// Pool growth strategies
#[derive(Debug, Clone, PartialEq)]
pub enum PoolGrowthStrategy {
    /// Fixed size pools
    Fixed,
    /// Linear growth
    Linear(usize),
    /// Exponential growth
    Exponential,
    /// Adaptive growth based on usage patterns
    Adaptive,
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Enable data compression
    pub enable_compression: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (1-9)
    pub compression_level: u8,
    /// Threshold for compression (bytes)
    pub compression_threshold: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            algorithm: CompressionAlgorithm::LZ4,
            compression_level: 6,
            compression_threshold: 1024, // 1KB
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionAlgorithm {
    /// LZ4 - fast compression
    LZ4,
    /// ZSTD - balanced compression
    ZSTD,
    /// GZIP - high compression
    GZIP,
    /// Snappy - Google's compression
    Snappy,
}

/// Garbage collection strategies
#[derive(Debug, Clone, PartialEq)]
pub enum GarbageCollectionStrategy {
    /// Manual garbage collection
    Manual,
    /// Automatic GC based on memory pressure
    Automatic,
    /// Adaptive GC based on usage patterns
    Adaptive,
    /// Generational GC
    Generational,
}

/// Parallel processing configuration
#[derive(Debug, Clone)]
pub struct ParallelProcessingConfig {
    /// Number of worker threads
    pub num_threads: usize,
    /// Thread pool configuration
    pub thread_pool_config: ThreadPoolConfig,
    /// NUMA awareness settings
    pub numa_config: NUMAConfig,
    /// Task scheduling strategy
    pub scheduling_strategy: TaskSchedulingStrategy,
    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,
}

impl Default for ParallelProcessingConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            thread_pool_config: ThreadPoolConfig::default(),
            numa_config: NUMAConfig::default(),
            scheduling_strategy: TaskSchedulingStrategy::WorkStealing,
            load_balancing: LoadBalancingConfig::default(),
        }
    }
}

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Core pool size
    pub core_pool_size: usize,
    /// Maximum pool size
    pub max_pool_size: usize,
    /// Thread keep-alive time
    pub keep_alive_time: Duration,
    /// Task queue size
    pub queue_size: usize,
    /// Thread priority
    pub thread_priority: ThreadPriority,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            core_pool_size: num_cpus::get(),
            max_pool_size: num_cpus::get() * 2,
            keep_alive_time: Duration::from_secs(60),
            queue_size: 10000,
            thread_priority: ThreadPriority::Normal,
        }
    }
}

/// Thread priorities
#[derive(Debug, Clone, PartialEq)]
pub enum ThreadPriority {
    Low,
    Normal,
    High,
    RealTime,
}

/// NUMA configuration
#[derive(Debug, Clone)]
pub struct NUMAConfig {
    /// Enable NUMA awareness
    pub enable_numa_awareness: bool,
    /// Memory binding strategy
    pub memory_binding: NUMAMemoryBinding,
    /// Thread affinity settings
    pub thread_affinity: NUMAThreadAffinity,
}

impl Default for NUMAConfig {
    fn default() -> Self {
        Self {
            enable_numa_awareness: true,
            memory_binding: NUMAMemoryBinding::LocalPreferred,
            thread_affinity: NUMAThreadAffinity::Soft,
        }
    }
}

/// NUMA memory binding strategies
#[derive(Debug, Clone, PartialEq)]
pub enum NUMAMemoryBinding {
    /// No binding
    None,
    /// Prefer local node
    LocalPreferred,
    /// Strict local binding
    LocalStrict,
    /// Interleaved across nodes
    Interleaved,
}

/// NUMA thread affinity
#[derive(Debug, Clone, PartialEq)]
pub enum NUMAThreadAffinity {
    /// No affinity
    None,
    /// Soft affinity (hint)
    Soft,
    /// Hard affinity (strict)
    Hard,
}

/// Task scheduling strategies
#[derive(Debug, Clone, PartialEq)]
pub enum TaskSchedulingStrategy {
    /// First-In-First-Out
    FIFO,
    /// Priority-based scheduling
    Priority,
    /// Work-stealing
    WorkStealing,
    /// Adaptive scheduling
    Adaptive,
}

/// Load balancing configuration
#[derive(Debug, Clone)]
pub struct LoadBalancingConfig {
    /// Enable dynamic load balancing
    pub enable_dynamic_balancing: bool,
    /// Load measurement interval
    pub measurement_interval: Duration,
    /// Rebalancing threshold
    pub rebalancing_threshold: f64,
    /// Balancing strategy
    pub balancing_strategy: LoadBalancingStrategy,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            enable_dynamic_balancing: true,
            measurement_interval: Duration::from_secs(5),
            rebalancing_threshold: 0.2, // 20% imbalance
            balancing_strategy: LoadBalancingStrategy::RoundRobin,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Least-loaded assignment
    LeastLoaded,
    /// Weighted assignment
    Weighted,
    /// Adaptive assignment
    Adaptive,
}

/// Algorithm optimization configuration
#[derive(Debug, Clone)]
pub struct AlgorithmOptimizationConfig {
    /// Enable algorithmic improvements
    pub enable_algorithmic_improvements: bool,
    /// Problem decomposition settings
    pub decomposition_config: DecompositionConfig,
    /// Caching and memoization settings
    pub caching_config: CachingConfig,
    /// Approximation algorithms settings
    pub approximation_config: ApproximationConfig,
    /// Streaming algorithms settings
    pub streaming_config: StreamingConfig,
}

impl Default for AlgorithmOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_algorithmic_improvements: true,
            decomposition_config: DecompositionConfig::default(),
            caching_config: CachingConfig::default(),
            approximation_config: ApproximationConfig::default(),
            streaming_config: StreamingConfig::default(),
        }
    }
}

/// Problem decomposition configuration
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Enable hierarchical decomposition
    pub enable_hierarchical_decomposition: bool,
    /// Maximum subproblem size
    pub max_subproblem_size: usize,
    /// Decomposition strategy
    pub decomposition_strategy: DecompositionStrategy,
    /// Overlap strategy for subproblems
    pub overlap_strategy: OverlapStrategy,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            enable_hierarchical_decomposition: true,
            max_subproblem_size: 10000,
            decomposition_strategy: DecompositionStrategy::Adaptive,
            overlap_strategy: OverlapStrategy::MinimalOverlap,
        }
    }
}

/// Decomposition strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompositionStrategy {
    /// Uniform decomposition
    Uniform,
    /// Adaptive decomposition
    Adaptive,
    /// Graph-based decomposition
    GraphBased,
    /// Hierarchical decomposition
    Hierarchical,
}

/// Overlap strategies for subproblems
#[derive(Debug, Clone, PartialEq)]
pub enum OverlapStrategy {
    /// No overlap
    NoOverlap,
    /// Minimal overlap
    MinimalOverlap,
    /// Substantial overlap
    SubstantialOverlap,
    /// Adaptive overlap
    AdaptiveOverlap,
}

/// Caching and memoization configuration
#[derive(Debug, Clone)]
pub struct CachingConfig {
    /// Enable result caching
    pub enable_result_caching: bool,
    /// Cache size limit (in MB)
    pub cache_size_limit: usize,
    /// Cache eviction policy
    pub eviction_policy: CacheEvictionPolicy,
    /// Cache compression
    pub enable_cache_compression: bool,
    /// Cache persistence
    pub enable_cache_persistence: bool,
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            enable_result_caching: true,
            cache_size_limit: 2048, // 2GB
            eviction_policy: CacheEvictionPolicy::LRU,
            enable_cache_compression: true,
            enable_cache_persistence: false,
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, PartialEq)]
pub enum CacheEvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First-In-First-Out
    FIFO,
    /// Adaptive Replacement Cache
    ARC,
}

/// Approximation algorithms configuration
#[derive(Debug, Clone)]
pub struct ApproximationConfig {
    /// Enable approximation algorithms
    pub enable_approximations: bool,
    /// Approximation quality threshold
    pub quality_threshold: f64,
    /// Maximum approximation error
    pub max_approximation_error: f64,
    /// Approximation strategies
    pub approximation_strategies: Vec<ApproximationStrategy>,
}

impl Default for ApproximationConfig {
    fn default() -> Self {
        Self {
            enable_approximations: true,
            quality_threshold: 0.95,
            max_approximation_error: 0.05,
            approximation_strategies: vec![
                ApproximationStrategy::Sampling,
                ApproximationStrategy::Clustering,
                ApproximationStrategy::DimensionalityReduction,
            ],
        }
    }
}

/// Approximation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ApproximationStrategy {
    /// Monte Carlo sampling
    Sampling,
    /// Clustering-based approximation
    Clustering,
    /// Dimensionality reduction
    DimensionalityReduction,
    /// Hierarchical approximation
    Hierarchical,
    /// Machine learning approximation
    MachineLearning,
}

/// Streaming algorithms configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Enable streaming processing
    pub enable_streaming: bool,
    /// Buffer size for streaming
    pub buffer_size: usize,
    /// Streaming window size
    pub window_size: usize,
    /// Sliding window strategy
    pub sliding_strategy: SlidingWindowStrategy,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enable_streaming: true,
            buffer_size: 10000,
            window_size: 1000,
            sliding_strategy: SlidingWindowStrategy::Tumbling,
        }
    }
}

/// Sliding window strategies
#[derive(Debug, Clone, PartialEq)]
pub enum SlidingWindowStrategy {
    /// Tumbling windows
    Tumbling,
    /// Sliding windows
    Sliding,
    /// Session windows
    Session,
    /// Custom windows
    Custom,
}

/// Distributed computing configuration
#[derive(Debug, Clone)]
pub struct DistributedComputingConfig {
    /// Enable distributed processing
    pub enable_distributed: bool,
    /// Cluster configuration
    pub cluster_config: ClusterConfig,
    /// Communication protocol
    pub communication_protocol: CommunicationProtocol,
    /// Fault tolerance settings
    pub fault_tolerance: DistributedFaultTolerance,
}

impl Default for DistributedComputingConfig {
    fn default() -> Self {
        Self {
            enable_distributed: false,
            cluster_config: ClusterConfig::default(),
            communication_protocol: CommunicationProtocol::TCP,
            fault_tolerance: DistributedFaultTolerance::default(),
        }
    }
}

/// Cluster configuration
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Master node address
    pub master_address: String,
    /// Worker node addresses
    pub worker_addresses: Vec<String>,
    /// Node resources
    pub node_resources: HashMap<String, NodeResources>,
    /// Network topology
    pub network_topology: NetworkTopology,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            master_address: "localhost:8000".to_string(),
            worker_addresses: vec![],
            node_resources: HashMap::new(),
            network_topology: NetworkTopology::StarTopology,
        }
    }
}

/// Node resource specification
#[derive(Debug, Clone)]
pub struct NodeResources {
    /// CPU cores
    pub cpu_cores: usize,
    /// Memory in MB
    pub memory_mb: usize,
    /// GPU count
    pub gpu_count: usize,
    /// Network bandwidth (Mbps)
    pub network_bandwidth: f64,
}

/// Network topologies
#[derive(Debug, Clone, PartialEq)]
pub enum NetworkTopology {
    /// Star topology (master-worker)
    StarTopology,
    /// Ring topology
    RingTopology,
    /// Mesh topology
    MeshTopology,
    /// Tree topology
    TreeTopology,
}

/// Communication protocols
#[derive(Debug, Clone, PartialEq)]
pub enum CommunicationProtocol {
    /// TCP protocol
    TCP,
    /// UDP protocol
    UDP,
    /// MPI (Message Passing Interface)
    MPI,
    /// gRPC
    GRPC,
    /// Custom protocol
    Custom(String),
}

/// Distributed fault tolerance
#[derive(Debug, Clone)]
pub struct DistributedFaultTolerance {
    /// Enable automatic failover
    pub enable_failover: bool,
    /// Replication factor
    pub replication_factor: usize,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Recovery strategy
    pub recovery_strategy: RecoveryStrategy,
}

impl Default for DistributedFaultTolerance {
    fn default() -> Self {
        Self {
            enable_failover: true,
            replication_factor: 2,
            heartbeat_interval: Duration::from_secs(5),
            recovery_strategy: RecoveryStrategy::Restart,
        }
    }
}

/// Recovery strategies
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStrategy {
    /// Restart failed tasks
    Restart,
    /// Migrate to other nodes
    Migrate,
    /// Checkpoint and restore
    CheckpointRestore,
    /// Adaptive recovery
    Adaptive,
}

/// Profiling configuration
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Profiling granularity
    pub profiling_granularity: ProfilingGranularity,
    /// Metrics collection interval
    pub collection_interval: Duration,
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Enable CPU profiling
    pub enable_cpu_profiling: bool,
    /// Enable I/O profiling
    pub enable_io_profiling: bool,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enable_profiling: true,
            profiling_granularity: ProfilingGranularity::Function,
            collection_interval: Duration::from_millis(100),
            enable_memory_profiling: true,
            enable_cpu_profiling: true,
            enable_io_profiling: true,
        }
    }
}

/// Profiling granularity
#[derive(Debug, Clone, PartialEq)]
pub enum ProfilingGranularity {
    /// Line-by-line profiling
    Line,
    /// Function-level profiling
    Function,
    /// Module-level profiling
    Module,
    /// Application-level profiling
    Application,
}

/// GPU acceleration configuration
#[derive(Debug, Clone)]
pub struct GPUAccelerationConfig {
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// GPU device selection
    pub device_selection: GPUDeviceSelection,
    /// Memory management strategy
    pub memory_strategy: GPUMemoryStrategy,
    /// Kernel optimization settings
    pub kernel_config: GPUKernelConfig,
}

impl Default for GPUAccelerationConfig {
    fn default() -> Self {
        Self {
            enable_gpu: false, // Disabled by default
            device_selection: GPUDeviceSelection::Automatic,
            memory_strategy: GPUMemoryStrategy::Unified,
            kernel_config: GPUKernelConfig::default(),
        }
    }
}

/// GPU device selection strategies
#[derive(Debug, Clone, PartialEq)]
pub enum GPUDeviceSelection {
    /// Automatic device selection
    Automatic,
    /// Use specific device
    Specific(usize),
    /// Use multiple devices
    Multiple(Vec<usize>),
    /// Use all available devices
    All,
}

/// GPU memory management strategies
#[derive(Debug, Clone, PartialEq)]
pub enum GPUMemoryStrategy {
    /// Unified memory
    Unified,
    /// Explicit memory management
    Explicit,
    /// Streaming memory
    Streaming,
    /// Memory pooling
    Pooled,
}

/// GPU kernel configuration
#[derive(Debug, Clone)]
pub struct GPUKernelConfig {
    /// Block size for CUDA kernels
    pub block_size: usize,
    /// Grid size for CUDA kernels
    pub grid_size: usize,
    /// Enable kernel fusion
    pub enable_kernel_fusion: bool,
    /// Optimization level
    pub optimization_level: GPUOptimizationLevel,
}

impl Default for GPUKernelConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            grid_size: 1024,
            enable_kernel_fusion: true,
            optimization_level: GPUOptimizationLevel::Aggressive,
        }
    }
}

/// GPU optimization levels
#[derive(Debug, Clone, PartialEq)]
pub enum GPUOptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Aggressive optimization
    Aggressive,
    /// Maximum optimization
    Maximum,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_configs() {
        let config = PerformanceOptimizationConfig::default();
        assert!(config.memory_config.enable_hierarchical_memory);
        assert!(config.parallel_config.load_balancing.enable_dynamic_balancing);
        assert!(config.algorithm_config.enable_algorithmic_improvements);
        assert!(!config.distributed_config.enable_distributed);
        assert!(config.profiling_config.enable_profiling);
        assert!(!config.gpu_config.enable_gpu);
    }

    #[test]
    fn test_memory_pool_config() {
        let config = MemoryPoolConfig::default();
        assert_eq!(config.pool_size, 4096);
        assert!(!config.block_sizes.is_empty());
        assert_eq!(config.growth_strategy, PoolGrowthStrategy::Exponential);
    }

    #[test]
    fn test_parallel_config() {
        let config = ParallelProcessingConfig::default();
        assert!(config.num_threads > 0);
        assert_eq!(config.scheduling_strategy, TaskSchedulingStrategy::WorkStealing);
        assert!(config.numa_config.enable_numa_awareness);
    }

    #[test]
    fn test_gpu_config() {
        let config = GPUAccelerationConfig::default();
        assert!(!config.enable_gpu);
        assert_eq!(config.device_selection, GPUDeviceSelection::Automatic);
        assert_eq!(config.memory_strategy, GPUMemoryStrategy::Unified);
    }
}