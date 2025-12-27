//! Hardware-Aware Quantum Circuit Parallelization
//!
//! This module provides sophisticated parallelization capabilities that understand
//! and respect hardware constraints, topology, and resource limitations to maximize
//! throughput while maintaining circuit fidelity and correctness.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

use quantrs2_circuit::prelude::*;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    platform::PlatformCapabilities,
    qubit::QubitId,
};

// SciRS2 integration for advanced parallelization analysis
#[cfg(feature = "scirs2")]
use scirs2_graph::{
    betweenness_centrality, closeness_centrality, dijkstra_path, minimum_spanning_tree,
    strongly_connected_components, topological_sort, Graph,
};
#[cfg(feature = "scirs2")]
use scirs2_optimize::{differential_evolution, minimize, OptimizeResult};
#[cfg(feature = "scirs2")]
use scirs2_stats::{corrcoef, mean, pearsonr, std};

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex as AsyncMutex, RwLock as AsyncRwLock, Semaphore};

use crate::{
    backend_traits::{query_backend_capabilities, BackendCapabilities},
    calibration::{CalibrationManager, DeviceCalibration},
    integrated_device_manager::{DeviceInfo, IntegratedQuantumDeviceManager},
    routing_advanced::{AdvancedQubitRouter, AdvancedRoutingResult},
    topology::HardwareTopology,
    translation::HardwareBackend,
    DeviceError, DeviceResult,
};

/// Hardware-aware parallelization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelizationConfig {
    /// Parallelization strategy
    pub strategy: ParallelizationStrategy,
    /// Resource allocation settings
    pub resource_allocation: ResourceAllocationConfig,
    /// Scheduling configuration
    pub scheduling_config: ParallelSchedulingConfig,
    /// Hardware awareness settings
    pub hardware_awareness: HardwareAwarenessConfig,
    /// Performance optimization settings
    pub performance_config: PerformanceOptimizationConfig,
    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,
    /// Resource monitoring settings
    pub monitoring_config: ResourceMonitoringConfig,
}

/// Parallelization strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParallelizationStrategy {
    /// Circuit-level parallelization (multiple independent circuits)
    CircuitLevel,
    /// Gate-level parallelization (parallel gate execution)
    GateLevel,
    /// Hybrid approach combining both strategies
    Hybrid,
    /// Topology-aware parallelization
    TopologyAware,
    /// Resource-constrained parallelization
    ResourceConstrained,
    /// SciRS2-powered intelligent parallelization
    SciRS2Optimized,
    /// Custom strategy with specific parameters
    Custom {
        circuit_concurrency: usize,
        gate_concurrency: usize,
        resource_weights: HashMap<String, f64>,
    },
}

/// Resource allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocationConfig {
    /// Maximum concurrent circuits
    pub max_concurrent_circuits: usize,
    /// Maximum concurrent gates per circuit
    pub max_concurrent_gates: usize,
    /// CPU core allocation strategy
    pub cpu_allocation: CpuAllocationStrategy,
    /// Memory allocation limits
    pub memory_limits: MemoryLimits,
    /// QPU resource allocation
    pub qpu_allocation: QpuAllocationConfig,
    /// Network bandwidth allocation
    pub network_allocation: NetworkAllocationConfig,
}

/// CPU allocation strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CpuAllocationStrategy {
    /// Use all available cores
    AllCores,
    /// Use fixed number of cores
    FixedCores(usize),
    /// Use percentage of available cores
    PercentageCores(f64),
    /// Adaptive allocation based on load
    Adaptive,
    /// NUMA-aware allocation
    NumaAware,
}

/// Memory allocation limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Maximum total memory usage (MB)
    pub max_total_memory_mb: f64,
    /// Maximum memory per circuit (MB)
    pub max_per_circuit_mb: f64,
    /// Memory allocation strategy
    pub allocation_strategy: MemoryAllocationStrategy,
    /// Enable memory pooling
    pub enable_pooling: bool,
    /// Garbage collection threshold
    pub gc_threshold: f64,
}

/// Memory allocation strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryAllocationStrategy {
    /// Static allocation upfront
    Static,
    /// Dynamic allocation as needed
    Dynamic,
    /// Pooled allocation with reuse
    Pooled,
    /// Adaptive based on circuit complexity
    Adaptive,
}

/// QPU resource allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QpuAllocationConfig {
    /// Maximum QPU time per circuit
    pub max_qpu_time_per_circuit: Duration,
    /// QPU sharing strategy
    pub sharing_strategy: QpuSharingStrategy,
    /// Queue management
    pub queue_management: QueueManagementConfig,
    /// Fairness parameters
    pub fairness_config: FairnessConfig,
}

/// QPU sharing strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QpuSharingStrategy {
    /// Time slicing
    TimeSlicing,
    /// Space slicing (using different qubits)
    SpaceSlicing,
    /// Hybrid time/space slicing
    HybridSlicing,
    /// Exclusive access
    Exclusive,
    /// Best effort sharing
    BestEffort,
}

/// Queue management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueManagementConfig {
    /// Queue scheduling algorithm
    pub algorithm: QueueSchedulingAlgorithm,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Priority levels
    pub priority_levels: usize,
    /// Enable preemption
    pub enable_preemption: bool,
    /// Timeout settings
    pub timeout_config: TimeoutConfig,
}

/// Queue scheduling algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueueSchedulingAlgorithm {
    /// First-come, first-served
    FCFS,
    /// Shortest job first
    SJF,
    /// Priority-based scheduling
    Priority,
    /// Round-robin
    RoundRobin,
    /// Multilevel feedback queue
    MLFQ,
    /// SciRS2-optimized scheduling
    SciRS2Optimized,
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Circuit execution timeout
    pub execution_timeout: Duration,
    /// Queue wait timeout
    pub queue_timeout: Duration,
    /// Resource acquisition timeout
    pub resource_timeout: Duration,
    /// Enable adaptive timeouts
    pub adaptive_timeouts: bool,
}

/// Fairness configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessConfig {
    /// Fairness algorithm
    pub algorithm: FairnessAlgorithm,
    /// Resource quotas per user/circuit
    pub resource_quotas: ResourceQuotas,
    /// Aging factor for starvation prevention
    pub aging_factor: f64,
    /// Enable burst allowances
    pub enable_burst_allowances: bool,
}

/// Fairness algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FairnessAlgorithm {
    /// Proportional fair sharing
    ProportionalFair,
    /// Max-min fairness
    MaxMinFair,
    /// Weighted fair queuing
    WeightedFairQueuing,
    /// Lottery scheduling
    LotteryScheduling,
    /// Game-theoretic fair scheduling
    GameTheoretic,
}

/// Resource quotas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuotas {
    /// CPU time quota per user
    pub cpu_quota: Option<Duration>,
    /// QPU time quota per user
    pub qpu_quota: Option<Duration>,
    /// Memory quota per user (MB)
    pub memory_quota: Option<f64>,
    /// Circuit count quota per user
    pub circuit_quota: Option<usize>,
}

/// Network allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAllocationConfig {
    /// Maximum bandwidth per circuit (Mbps)
    pub max_bandwidth_per_circuit: f64,
    /// Network QoS class
    pub qos_class: NetworkQoSClass,
    /// Compression settings
    pub compression_config: CompressionConfig,
    /// Latency optimization
    pub latency_optimization: bool,
}

/// Network QoS classes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkQoSClass {
    /// Best effort
    BestEffort,
    /// Assured forwarding
    AssuredForwarding,
    /// Expedited forwarding
    ExpeditedForwarding,
    /// Real-time
    RealTime,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (0-9)
    pub level: u8,
    /// Minimum size threshold for compression
    pub size_threshold: usize,
}

/// Compression algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// Gzip compression
    Gzip,
    /// Zstd compression
    Zstd,
    /// LZ4 compression
    LZ4,
    /// Brotli compression
    Brotli,
}

/// Parallel scheduling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelSchedulingConfig {
    /// Scheduling algorithm
    pub algorithm: ParallelSchedulingAlgorithm,
    /// Work stealing configuration
    pub work_stealing: WorkStealingConfig,
    /// Load balancing parameters
    pub load_balancing_params: LoadBalancingParams,
    /// Thread pool configuration
    pub thread_pool_config: ThreadPoolConfig,
}

/// Parallel scheduling algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParallelSchedulingAlgorithm {
    /// Work stealing
    WorkStealing,
    /// Work sharing
    WorkSharing,
    /// Fork-join
    ForkJoin,
    /// Actor model
    ActorModel,
    /// Pipeline parallelism
    Pipeline,
    /// Data parallelism
    DataParallel,
    /// Task parallelism
    TaskParallel,
}

/// Work stealing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkStealingConfig {
    /// Enable work stealing
    pub enabled: bool,
    /// Stealing strategy
    pub strategy: WorkStealingStrategy,
    /// Queue size per worker
    pub queue_size: usize,
    /// Stealing threshold
    pub stealing_threshold: f64,
}

/// Work stealing strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkStealingStrategy {
    /// Random stealing
    Random,
    /// Round-robin stealing
    RoundRobin,
    /// Load-based stealing
    LoadBased,
    /// Locality-aware stealing
    LocalityAware,
}

/// Load balancing parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingParams {
    /// Rebalancing frequency
    pub rebalancing_frequency: Duration,
    /// Load threshold for rebalancing
    pub load_threshold: f64,
    /// Migration cost factor
    pub migration_cost_factor: f64,
    /// Enable adaptive load balancing
    pub adaptive_balancing: bool,
}

/// Thread pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolConfig {
    /// Core thread count
    pub core_threads: usize,
    /// Maximum thread count
    pub max_threads: usize,
    /// Keep-alive time for idle threads
    pub keep_alive_time: Duration,
    /// Thread priority
    pub thread_priority: ThreadPriority,
    /// Thread affinity settings
    pub affinity_config: ThreadAffinityConfig,
}

/// Thread priority levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreadPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Real-time priority
    RealTime,
}

/// Thread affinity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadAffinityConfig {
    /// Enable CPU affinity
    pub enabled: bool,
    /// CPU core assignment strategy
    pub assignment_strategy: CoreAssignmentStrategy,
    /// NUMA node preference
    pub numa_preference: NumaPreference,
}

/// Core assignment strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoreAssignmentStrategy {
    /// Automatic assignment
    Automatic,
    /// Fixed core assignment
    Fixed(Vec<usize>),
    /// Round-robin assignment
    RoundRobin,
    /// Load-based assignment
    LoadBased,
}

/// NUMA preferences
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumaPreference {
    /// No preference
    None,
    /// Local node preferred
    LocalNode,
    /// Specific node
    SpecificNode(usize),
    /// Interleaved across nodes
    Interleaved,
}

/// Hardware awareness configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareAwarenessConfig {
    /// Topology awareness level
    pub topology_awareness: TopologyAwarenessLevel,
    /// Calibration integration
    pub calibration_integration: CalibrationIntegrationConfig,
    /// Error rate consideration
    pub error_rate_config: ErrorRateConfig,
    /// Connectivity constraints
    pub connectivity_config: ConnectivityConfig,
    /// Resource usage tracking
    pub resource_tracking: ResourceTrackingConfig,
}

/// Topology awareness levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyAwarenessLevel {
    /// Basic awareness (qubit count only)
    Basic,
    /// Connectivity aware
    Connectivity,
    /// Calibration aware
    Calibration,
    /// Full topology optimization
    Full,
    /// SciRS2-powered topology analysis
    SciRS2Enhanced,
}

/// Calibration integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationIntegrationConfig {
    /// Use real-time calibration data
    pub use_realtime_calibration: bool,
    /// Calibration update frequency
    pub update_frequency: Duration,
    /// Quality threshold for gate selection
    pub quality_threshold: f64,
    /// Enable predictive calibration
    pub enable_predictive: bool,
}

/// Error rate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRateConfig {
    /// Consider error rates in scheduling
    pub consider_error_rates: bool,
    /// Error rate threshold
    pub error_threshold: f64,
    /// Error mitigation strategy
    pub mitigation_strategy: ErrorMitigationStrategy,
    /// Error prediction model
    pub prediction_model: ErrorPredictionModel,
}

/// Error mitigation strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorMitigationStrategy {
    /// No mitigation
    None,
    /// Retry on high error
    Retry,
    /// Dynamical decoupling
    DynamicalDecoupling,
    /// Zero-noise extrapolation
    ZeroNoiseExtrapolation,
    /// Composite mitigation
    Composite,
}

/// Error prediction models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorPredictionModel {
    /// Static error model
    Static,
    /// Time-dependent model
    TimeDependent,
    /// Machine learning model
    MachineLearning,
    /// Physics-based model
    PhysicsBased,
}

/// Connectivity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityConfig {
    /// Enforce connectivity constraints
    pub enforce_constraints: bool,
    /// SWAP insertion strategy
    pub swap_strategy: SwapInsertionStrategy,
    /// Routing algorithm preference
    pub routing_preference: RoutingPreference,
    /// Connectivity optimization
    pub optimization_config: ConnectivityOptimizationConfig,
}

/// SWAP insertion strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwapInsertionStrategy {
    /// Minimal SWAP insertion
    Minimal,
    /// Lookahead SWAP insertion
    Lookahead,
    /// Global optimization
    GlobalOptimal,
    /// Heuristic-based
    Heuristic,
}

/// Routing preferences
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingPreference {
    /// Shortest path
    ShortestPath,
    /// Minimum congestion
    MinimumCongestion,
    /// Load balancing
    LoadBalancing,
    /// Quality-aware routing
    QualityAware,
}

/// Connectivity optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityOptimizationConfig {
    /// Enable parallel routing
    pub enable_parallel_routing: bool,
    /// Routing optimization level
    pub optimization_level: OptimizationLevel,
    /// Use machine learning for routing
    pub use_ml_routing: bool,
    /// Precompute routing tables
    pub precompute_tables: bool,
}

/// Optimization levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Moderate optimization
    Moderate,
    /// Aggressive optimization
    Aggressive,
    /// Experimental optimization
    Experimental,
}

/// Resource tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceTrackingConfig {
    /// Enable CPU usage tracking
    pub track_cpu_usage: bool,
    /// Enable memory usage tracking
    pub track_memory_usage: bool,
    /// Enable QPU usage tracking
    pub track_qpu_usage: bool,
    /// Enable network usage tracking
    pub track_network_usage: bool,
    /// Tracking granularity
    pub tracking_granularity: TrackingGranularity,
    /// Reporting frequency
    pub reporting_frequency: Duration,
}

/// Tracking granularity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingGranularity {
    /// Coarse-grained (per circuit)
    Coarse,
    /// Medium-grained (per gate group)
    Medium,
    /// Fine-grained (per gate)
    Fine,
    /// Ultra-fine (per operation)
    UltraFine,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceOptimizationConfig {
    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,
    /// Caching configuration
    pub caching_config: CachingConfig,
    /// Prefetching settings
    pub prefetching_config: PrefetchingConfig,
    /// Batch processing settings
    pub batch_config: BatchProcessingConfig,
    /// Adaptive optimization
    pub adaptive_config: AdaptiveOptimizationConfig,
}

/// Optimization objectives
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Minimize execution time
    MinimizeTime,
    /// Maximize throughput
    MaximizeThroughput,
    /// Minimize resource usage
    MinimizeResources,
    /// Maximize quality
    MaximizeQuality,
    /// Minimize cost
    MinimizeCost,
    /// Minimize energy consumption
    MinimizeEnergy,
    /// Balanced optimization
    Balanced,
}

/// Caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Enable result caching
    pub enable_result_caching: bool,
    /// Enable compilation caching
    pub enable_compilation_caching: bool,
    /// Cache size limits
    pub size_limits: CacheSizeLimits,
    /// Cache eviction policy
    pub eviction_policy: CacheEvictionPolicy,
    /// Cache warming strategies
    pub warming_strategies: Vec<CacheWarmingStrategy>,
}

/// Cache size limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSizeLimits {
    /// Maximum cache entries
    pub max_entries: usize,
    /// Maximum memory usage (MB)
    pub max_memory_mb: f64,
    /// Maximum disk usage (MB)
    pub max_disk_mb: f64,
    /// Per-user cache limits
    pub per_user_limits: Option<Box<Self>>,
}

/// Cache eviction policies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheEvictionPolicy {
    /// Least recently used
    LRU,
    /// Least frequently used
    LFU,
    /// First in, first out
    FIFO,
    /// Random eviction
    Random,
    /// Time-based expiration
    TimeExpiration,
    /// Size-based eviction
    SizeBased,
}

/// Cache warming strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheWarmingStrategy {
    /// Preload common circuits
    PreloadCommon,
    /// Predictive preloading
    Predictive,
    /// User pattern based
    UserPatternBased,
    /// Background warming
    Background,
}

/// Prefetching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchingConfig {
    /// Enable prefetching
    pub enabled: bool,
    /// Prefetching strategy
    pub strategy: PrefetchingStrategy,
    /// Prefetch distance
    pub prefetch_distance: usize,
    /// Prefetch confidence threshold
    pub confidence_threshold: f64,
}

/// Prefetching strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrefetchingStrategy {
    /// Sequential prefetching
    Sequential,
    /// Pattern-based prefetching
    PatternBased,
    /// Machine learning prefetching
    MachineLearning,
    /// Adaptive prefetching
    Adaptive,
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingConfig {
    /// Enable batch processing
    pub enabled: bool,
    /// Batch size limits
    pub size_limits: BatchSizeLimits,
    /// Batching strategy
    pub strategy: BatchingStrategy,
    /// Batch timeout
    pub timeout: Duration,
}

/// Batch size limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSizeLimits {
    /// Minimum batch size
    pub min_size: usize,
    /// Maximum batch size
    pub max_size: usize,
    /// Optimal batch size
    pub optimal_size: usize,
    /// Dynamic sizing
    pub dynamic_sizing: bool,
}

/// Batching strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchingStrategy {
    /// Fixed size batching
    FixedSize,
    /// Time-based batching
    TimeBased,
    /// Adaptive batching
    Adaptive,
    /// Circuit similarity batching
    SimilarityBased,
    /// Resource-aware batching
    ResourceAware,
}

/// Adaptive optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveOptimizationConfig {
    /// Enable adaptive optimization
    pub enabled: bool,
    /// Adaptation frequency
    pub adaptation_frequency: Duration,
    /// Performance monitoring window
    pub monitoring_window: Duration,
    /// Adaptation sensitivity
    pub sensitivity: f64,
    /// Machine learning config
    pub ml_config: AdaptiveMLConfig,
}

/// Adaptive machine learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveMLConfig {
    /// Enable ML-based adaptation
    pub enabled: bool,
    /// ML model type
    pub model_type: MLModelType,
    /// Training frequency
    pub training_frequency: Duration,
    /// Feature engineering config
    pub feature_config: FeatureEngineeringConfig,
}

/// ML model types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MLModelType {
    /// Linear regression
    LinearRegression,
    /// Random forest
    RandomForest,
    /// Neural network
    NeuralNetwork,
    /// Reinforcement learning
    ReinforcementLearning,
    /// Ensemble methods
    Ensemble,
}

/// Feature engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    /// Circuit features to extract
    pub circuit_features: Vec<CircuitFeature>,
    /// Hardware features to extract
    pub hardware_features: Vec<HardwareFeature>,
    /// Performance features to extract
    pub performance_features: Vec<PerformanceFeature>,
    /// Feature normalization
    pub normalization: FeatureNormalization,
}

/// Circuit features
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitFeature {
    /// Number of qubits
    QubitCount,
    /// Circuit depth
    Depth,
    /// Gate count
    GateCount,
    /// Gate type distribution
    GateTypeDistribution,
    /// Connectivity requirements
    ConnectivityRequirements,
    /// Parallelism potential
    ParallelismPotential,
}

/// Hardware features
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareFeature {
    /// Available qubits
    AvailableQubits,
    /// Connectivity graph
    ConnectivityGraph,
    /// Error rates
    ErrorRates,
    /// Calibration quality
    CalibrationQuality,
    /// Queue status
    QueueStatus,
    /// Resource utilization
    ResourceUtilization,
}

/// Performance features
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceFeature {
    /// Execution time
    ExecutionTime,
    /// Throughput
    Throughput,
    /// Resource efficiency
    ResourceEfficiency,
    /// Quality metrics
    QualityMetrics,
    /// Cost metrics
    CostMetrics,
    /// Energy consumption
    EnergyConsumption,
}

/// Feature normalization methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureNormalization {
    /// No normalization
    None,
    /// Min-max normalization
    MinMax,
    /// Z-score normalization
    ZScore,
    /// Robust normalization
    Robust,
    /// Unit vector normalization
    UnitVector,
}

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// Load balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    /// Monitoring configuration
    pub monitoring: LoadMonitoringConfig,
    /// Rebalancing triggers
    pub rebalancing_triggers: RebalancingTriggers,
    /// Migration policies
    pub migration_policies: MigrationPolicies,
}

/// Load balancing algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    /// Round-robin
    RoundRobin,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// Least connections
    LeastConnections,
    /// Least response time
    LeastResponseTime,
    /// Resource-based balancing
    ResourceBased,
    /// Machine learning based
    MachineLearningBased,
}

/// Load monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadMonitoringConfig {
    /// Monitoring frequency
    pub frequency: Duration,
    /// Metrics to monitor
    pub metrics: Vec<LoadMetric>,
    /// Alerting thresholds
    pub thresholds: LoadThresholds,
    /// Historical data retention
    pub retention_period: Duration,
}

/// Load metrics
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadMetric {
    /// CPU utilization
    CpuUtilization,
    /// Memory utilization
    MemoryUtilization,
    /// QPU utilization
    QpuUtilization,
    /// Network utilization
    NetworkUtilization,
    /// Queue length
    QueueLength,
    /// Response time
    ResponseTime,
    /// Throughput
    Throughput,
    /// Error rate
    ErrorRate,
}

/// Load thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadThresholds {
    /// CPU utilization threshold
    pub cpu_threshold: f64,
    /// Memory utilization threshold
    pub memory_threshold: f64,
    /// QPU utilization threshold
    pub qpu_threshold: f64,
    /// Network utilization threshold
    pub network_threshold: f64,
    /// Queue length threshold
    pub queue_threshold: usize,
    /// Response time threshold
    pub response_time_threshold: Duration,
}

/// Rebalancing triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancingTriggers {
    /// CPU imbalance threshold
    pub cpu_imbalance_threshold: f64,
    /// Memory imbalance threshold
    pub memory_imbalance_threshold: f64,
    /// Queue imbalance threshold
    pub queue_imbalance_threshold: f64,
    /// Time-based rebalancing interval
    pub time_interval: Option<Duration>,
    /// Event-based triggers
    pub event_triggers: Vec<RebalancingEvent>,
}

/// Rebalancing events
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RebalancingEvent {
    /// Node failure
    NodeFailure,
    /// Node recovery
    NodeRecovery,
    /// Capacity change
    CapacityChange,
    /// Load spike
    LoadSpike,
    /// Performance degradation
    PerformanceDegradation,
}

/// Migration policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPolicies {
    /// Migration cost threshold
    pub cost_threshold: f64,
    /// Maximum migrations per period
    pub max_migrations_per_period: usize,
    /// Migration period
    pub migration_period: Duration,
    /// Circuit migration strategy
    pub circuit_migration_strategy: CircuitMigrationStrategy,
    /// Data migration strategy
    pub data_migration_strategy: DataMigrationStrategy,
}

/// Circuit migration strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitMigrationStrategy {
    /// No migration
    None,
    /// Checkpoint and restart
    CheckpointRestart,
    /// Live migration
    LiveMigration,
    /// Incremental migration
    Incremental,
}

/// Data migration strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataMigrationStrategy {
    /// No data migration
    None,
    /// Copy-based migration
    Copy,
    /// Move-based migration
    Move,
    /// Distributed caching
    DistributedCaching,
}

/// Resource monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitoringConfig {
    /// Enable real-time monitoring
    pub real_time_monitoring: bool,
    /// Monitoring granularity
    pub granularity: MonitoringGranularity,
    /// Metrics collection
    pub metrics_collection: MetricsCollectionConfig,
    /// Alerting configuration
    pub alerting: AlertingConfig,
    /// Reporting configuration
    pub reporting: ReportingConfig,
}

/// Monitoring granularity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonitoringGranularity {
    /// System-level monitoring
    System,
    /// Device-level monitoring
    Device,
    /// Circuit-level monitoring
    Circuit,
    /// Gate-level monitoring
    Gate,
    /// Operation-level monitoring
    Operation,
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsCollectionConfig {
    /// Collection frequency
    pub frequency: Duration,
    /// Metrics to collect
    pub metrics: Vec<MonitoringMetric>,
    /// Data retention policy
    pub retention_policy: RetentionPolicy,
    /// Storage configuration
    pub storage_config: StorageConfig,
}

/// Monitoring metrics
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonitoringMetric {
    /// Resource utilization
    ResourceUtilization,
    /// Performance metrics
    Performance,
    /// Quality metrics
    Quality,
    /// Cost metrics
    Cost,
    /// Energy metrics
    Energy,
    /// Availability metrics
    Availability,
    /// Security metrics
    Security,
}

/// Data retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Raw data retention period
    pub raw_data_retention: Duration,
    /// Aggregated data retention period
    pub aggregated_data_retention: Duration,
    /// Archive policy
    pub archive_policy: ArchivePolicy,
    /// Compression settings
    pub compression: CompressionConfig,
}

/// Archive policy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchivePolicy {
    /// No archiving
    None,
    /// Time-based archiving
    TimeBased(Duration),
    /// Size-based archiving
    SizeBased(usize),
    /// Custom archiving rules
    Custom(String),
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Storage backend
    pub backend: StorageBackend,
    /// Storage location
    pub location: String,
    /// Encryption settings
    pub encryption: EncryptionConfig,
    /// Replication settings
    pub replication: ReplicationConfig,
}

/// Storage backends
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageBackend {
    /// Local filesystem
    LocalFilesystem,
    /// Distributed filesystem
    DistributedFilesystem,
    /// Cloud storage
    CloudStorage,
    /// Database storage
    Database,
    /// Time-series database
    TimeSeriesDatabase,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Enable encryption
    pub enabled: bool,
    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,
    /// Key management
    pub key_management: KeyManagementConfig,
}

/// Encryption algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    /// AES-256
    AES256,
    /// ChaCha20-Poly1305
    ChaCha20Poly1305,
    /// XChaCha20-Poly1305
    XChaCha20Poly1305,
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementConfig {
    /// Key rotation frequency
    pub rotation_frequency: Duration,
    /// Key derivation function
    pub key_derivation: KeyDerivationFunction,
    /// Key storage backend
    pub storage_backend: KeyStorageBackend,
}

/// Key derivation functions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyDerivationFunction {
    /// PBKDF2
    PBKDF2,
    /// Scrypt
    Scrypt,
    /// Argon2
    Argon2,
}

/// Key storage backends
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyStorageBackend {
    /// Local keystore
    Local,
    /// Hardware security module
    HSM,
    /// Cloud key management
    CloudKMS,
    /// Distributed keystore
    Distributed,
}

/// Replication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    /// Replication factor
    pub replication_factor: usize,
    /// Replication strategy
    pub strategy: ReplicationStrategy,
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
}

/// Replication strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationStrategy {
    /// Synchronous replication
    Synchronous,
    /// Asynchronous replication
    Asynchronous,
    /// Semi-synchronous replication
    SemiSynchronous,
    /// Multi-master replication
    MultiMaster,
}

/// Consistency levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Strong consistency
    Strong,
    /// Eventual consistency
    Eventual,
    /// Causal consistency
    Causal,
    /// Session consistency
    Session,
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Alert rules
    pub rules: Vec<AlertRule>,
    /// Notification channels
    pub channels: Vec<NotificationChannel>,
    /// Alert aggregation
    pub aggregation: AlertAggregationConfig,
}

/// Alert rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Metric to monitor
    pub metric: MonitoringMetric,
    /// Threshold condition
    pub condition: ThresholdCondition,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Evaluation frequency
    pub frequency: Duration,
}

/// Threshold conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdCondition {
    /// Greater than threshold
    GreaterThan(f64),
    /// Less than threshold
    LessThan(f64),
    /// Equal to threshold
    EqualTo(f64),
    /// Within range
    WithinRange(f64, f64),
    /// Outside range
    OutsideRange(f64, f64),
}

/// Alert severities
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    /// Email notification
    Email {
        recipients: Vec<String>,
        smtp_config: SmtpConfig,
    },
    /// Slack notification
    Slack {
        webhook_url: String,
        channel: String,
    },
    /// HTTP webhook
    Webhook {
        url: String,
        headers: HashMap<String, String>,
    },
    /// SMS notification
    SMS {
        phone_numbers: Vec<String>,
        provider_config: SmsProviderConfig,
    },
}

/// SMTP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtpConfig {
    /// SMTP server
    pub server: String,
    /// SMTP port
    pub port: u16,
    /// Username
    pub username: String,
    /// Use TLS
    pub use_tls: bool,
}

/// SMS provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmsProviderConfig {
    /// Provider name
    pub provider: String,
    /// API key
    pub api_key: String,
    /// API endpoint
    pub endpoint: String,
}

/// Alert aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertAggregationConfig {
    /// Enable aggregation
    pub enabled: bool,
    /// Aggregation window
    pub window: Duration,
    /// Aggregation strategy
    pub strategy: AlertAggregationStrategy,
    /// Maximum alerts per window
    pub max_alerts_per_window: usize,
}

/// Alert aggregation strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertAggregationStrategy {
    /// Count-based aggregation
    Count,
    /// Severity-based aggregation
    SeverityBased,
    /// Metric-based aggregation
    MetricBased,
    /// Time-based aggregation
    TimeBased,
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Enable automated reporting
    pub enabled: bool,
    /// Report types to generate
    pub report_types: Vec<ReportType>,
    /// Report frequency
    pub frequency: Duration,
    /// Report format
    pub format: ReportFormat,
    /// Report distribution
    pub distribution: ReportDistribution,
}

/// Report types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportType {
    /// Performance report
    Performance,
    /// Resource utilization report
    ResourceUtilization,
    /// Quality metrics report
    QualityMetrics,
    /// Cost analysis report
    CostAnalysis,
    /// Capacity planning report
    CapacityPlanning,
    /// SLA compliance report
    SLACompliance,
}

/// Report formats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    /// PDF format
    PDF,
    /// HTML format
    HTML,
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// Excel format
    Excel,
}

/// Report distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportDistribution {
    /// Email recipients
    pub email_recipients: Vec<String>,
    /// File system location
    pub file_location: Option<String>,
    /// Cloud storage location
    pub cloud_location: Option<String>,
    /// API endpoints
    pub api_endpoints: Vec<String>,
}
