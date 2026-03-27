//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{gate::GateOp, qubit::QubitId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Barrier, Mutex, RwLock};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Hardware-specific parallelization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareStrategy {
    /// Optimize for cache locality
    CacheOptimized,
    /// Optimize for SIMD vectorization
    SIMDOptimized,
    /// NUMA-aware task distribution
    NUMAAware,
    /// Offload to GPU
    GPUOffload,
    /// Hybrid approach combining multiple optimizations
    Hybrid,
}
/// Types of optimization recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationType {
    /// Gate reordering for better parallelism
    GateReordering,
    /// Circuit decomposition
    CircuitDecomposition,
    /// Resource allocation adjustment
    ResourceAllocation,
    /// Strategy change recommendation
    StrategyChange,
    /// Hardware configuration
    HardwareConfiguration,
}
/// ML features extracted from circuits for parallelization prediction
#[derive(Debug, Clone)]
pub struct MLFeatures {
    /// Number of gates in the circuit
    pub num_gates: usize,
    /// Number of qubits in the circuit
    pub num_qubits: usize,
    /// Circuit depth (critical path length)
    pub circuit_depth: usize,
    /// Average gate connectivity
    pub avg_connectivity: f64,
    /// Parallelism factor (ratio of independent gates)
    pub parallelism_factor: f64,
    /// Gate type distribution
    pub gate_distribution: HashMap<String, usize>,
    /// Entanglement complexity score
    pub entanglement_score: f64,
    /// Dependency density (edges per gate)
    pub dependency_density: f64,
}
/// Hardware characteristics for hardware-aware parallelization
#[derive(Debug, Clone)]
pub struct HardwareCharacteristics {
    /// Number of available CPU cores
    pub num_cores: usize,
    /// L1 cache size per core (bytes)
    pub l1_cache_size: usize,
    /// L2 cache size per core (bytes)
    pub l2_cache_size: usize,
    /// L3 cache size (bytes)
    pub l3_cache_size: usize,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
    /// NUMA nodes available
    pub num_numa_nodes: usize,
    /// GPU availability
    pub has_gpu: bool,
    /// SIMD width (e.g., 256 for AVX2, 512 for AVX-512)
    pub simd_width: usize,
}
/// Parallelization analysis results
#[derive(Debug, Clone)]
pub struct ParallelizationAnalysis {
    /// Parallel tasks generated
    pub tasks: Vec<ParallelTask>,
    /// Total number of layers
    pub num_layers: usize,
    /// Parallelization efficiency (0.0 to 1.0)
    pub efficiency: f64,
    /// Maximum parallelism achievable
    pub max_parallelism: usize,
    /// Critical path length
    pub critical_path_length: usize,
    /// Resource utilization predictions
    pub resource_utilization: ResourceUtilization,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
}
/// Node capacity information for distributed task scheduling
#[derive(Debug, Clone)]
pub struct NodeCapacity {
    /// Number of CPU cores available
    pub cpu_cores: usize,
    /// Available memory in GB
    pub memory_gb: f64,
    /// GPU availability
    pub gpu_available: bool,
    /// Network bandwidth in Gbps
    pub network_bandwidth_gbps: f64,
    /// Relative performance score (normalized)
    pub relative_performance: f64,
}
/// Task completion statistics
#[derive(Debug, Clone, Default)]
pub struct TaskCompletionStats {
    /// Total tasks completed
    pub total_tasks: usize,
    /// Average task duration
    pub average_duration: Duration,
    /// Task success rate
    pub success_rate: f64,
    /// Load balancing effectiveness
    pub load_balance_effectiveness: f64,
}
/// Load balancer for parallel task execution
pub struct LoadBalancer {
    /// Current thread loads
    thread_loads: Vec<f64>,
    /// Task queue per thread
    task_queues: Vec<VecDeque<ParallelTask>>,
    /// Work stealing statistics
    work_stealing_stats: WorkStealingStats,
}
impl LoadBalancer {
    /// Create a new load balancer
    #[must_use]
    pub fn new(num_threads: usize) -> Self {
        Self {
            thread_loads: vec![0.0; num_threads],
            task_queues: vec![VecDeque::new(); num_threads],
            work_stealing_stats: WorkStealingStats::default(),
        }
    }
    /// Balance load across threads
    pub fn balance_load(&mut self, tasks: Vec<ParallelTask>) -> Vec<Vec<ParallelTask>> {
        let mut balanced_tasks = vec![Vec::new(); self.thread_loads.len()];
        for (i, task) in tasks.into_iter().enumerate() {
            let thread_index = i % self.thread_loads.len();
            balanced_tasks[thread_index].push(task);
        }
        balanced_tasks
    }
}
/// Resource utilization predictions
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// Estimated CPU utilization per thread
    pub cpu_utilization: Vec<f64>,
    /// Estimated memory usage per thread
    pub memory_usage: Vec<usize>,
    /// Load balancing score (0.0 to 1.0)
    pub load_balance_score: f64,
    /// Communication overhead estimate
    pub communication_overhead: f64,
}
/// Work stealing statistics
#[derive(Debug, Clone, Default)]
pub struct WorkStealingStats {
    /// Total steal attempts
    pub steal_attempts: usize,
    /// Successful steals
    pub successful_steals: usize,
    /// Failed steals
    pub failed_steals: usize,
    /// Average steal latency
    pub average_steal_latency: Duration,
}
/// Parallelization results for a single circuit
#[derive(Debug, Clone)]
pub struct CircuitParallelResult {
    /// Circuit size (number of gates)
    pub circuit_size: usize,
    /// Number of qubits
    pub num_qubits: usize,
    /// Time to analyze parallelization
    pub analysis_time: Duration,
    /// Parallelization efficiency
    pub efficiency: f64,
    /// Maximum parallelism achieved
    pub max_parallelism: usize,
    /// Number of parallel tasks generated
    pub num_tasks: usize,
}
/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    /// Low priority task
    Low = 1,
    /// Normal priority task
    Normal = 2,
    /// High priority task
    High = 3,
    /// Critical priority task
    Critical = 4,
}
/// Gate node in the dependency graph
#[derive(Debug, Clone)]
pub struct GateNode {
    /// Gate index in original circuit
    pub gate_index: usize,
    /// Gate operation
    pub gate: Arc<dyn GateOp + Send + Sync>,
    /// Qubits this gate operates on
    pub qubits: HashSet<QubitId>,
    /// Layer index in topological ordering
    pub layer: usize,
    /// Estimated execution cost
    pub cost: f64,
}
/// Load balancing configuration for parallel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// Enable dynamic load balancing
    pub enable_dynamic_balancing: bool,
    /// Work stealing strategy
    pub work_stealing_strategy: WorkStealingStrategy,
    /// Load monitoring interval
    pub monitoring_interval: Duration,
    /// Rebalancing threshold
    pub rebalancing_threshold: f64,
}
/// Circuit dependency graph for parallelization analysis
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Gate nodes in the dependency graph
    pub nodes: Vec<GateNode>,
    /// Adjacency list representation
    pub edges: HashMap<usize, Vec<usize>>,
    /// Reverse adjacency list
    pub reverse_edges: HashMap<usize, Vec<usize>>,
    /// Topological layers
    pub layers: Vec<Vec<usize>>,
}
/// Configuration for automatic parallelization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoParallelConfig {
    /// Maximum number of parallel execution threads
    pub max_threads: usize,
    /// Minimum gate count to enable parallelization
    pub min_gates_for_parallel: usize,
    /// Parallelization strategy
    pub strategy: ParallelizationStrategy,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Enable inter-layer parallelization
    pub enable_inter_layer_parallel: bool,
    /// Enable gate fusion optimization
    pub enable_gate_fusion: bool,
    /// `SciRS2` optimization level
    pub scirs2_optimization_level: OptimizationLevel,
    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,
    /// Enable circuit analysis caching
    pub enable_analysis_caching: bool,
    /// Memory budget for parallel execution
    pub memory_budget: usize,
}
/// Parallel execution task representing a group of independent gates
#[derive(Debug, Clone)]
pub struct ParallelTask {
    /// Unique task identifier
    pub id: Uuid,
    /// Gates to execute in this task
    pub gates: Vec<Arc<dyn GateOp + Send + Sync>>,
    /// Qubits involved in this task
    pub qubits: HashSet<QubitId>,
    /// Estimated execution cost
    pub cost: f64,
    /// Memory requirement estimate
    pub memory_requirement: usize,
    /// Dependencies (task IDs that must complete before this task)
    pub dependencies: HashSet<Uuid>,
    /// Priority level
    pub priority: TaskPriority,
}
/// Complexity levels for recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationComplexity {
    /// Low complexity, easy to implement
    Low,
    /// Medium complexity
    Medium,
    /// High complexity, significant changes required
    High,
}
/// Performance statistics for parallel execution
#[derive(Debug, Clone, Default)]
pub struct ParallelPerformanceStats {
    /// Total circuits processed
    pub circuits_processed: usize,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average parallelization efficiency
    pub average_efficiency: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Task completion statistics
    pub task_stats: TaskCompletionStats,
    /// Resource utilization history
    pub resource_history: Vec<ResourceSnapshot>,
}
/// Optimization recommendations for better parallelization
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Description of the recommendation
    pub description: String,
    /// Expected improvement (0.0 to 1.0)
    pub expected_improvement: f64,
    /// Implementation complexity
    pub complexity: RecommendationComplexity,
}
/// Work stealing strategies for load balancing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkStealingStrategy {
    /// Random work stealing
    Random,
    /// Cost-aware work stealing
    CostAware,
    /// Locality-aware work stealing
    LocalityAware,
    /// Adaptive strategy selection
    Adaptive,
}
/// `SciRS2` optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimizations
    Basic,
    /// Advanced optimizations
    Advanced,
    /// Aggressive optimizations
    Aggressive,
    /// Custom optimization profile
    Custom,
}
/// Resource constraints for parallel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum memory per thread (bytes)
    pub max_memory_per_thread: usize,
    /// Maximum CPU utilization (0.0 to 1.0)
    pub max_cpu_utilization: f64,
    /// Maximum gate operations per thread
    pub max_gates_per_thread: usize,
    /// Preferred NUMA node
    pub preferred_numa_node: Option<usize>,
}
/// Resource utilization snapshot
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// Timestamp
    pub timestamp: Instant,
    /// CPU utilization per core
    pub cpu_utilization: Vec<f64>,
    /// Memory usage
    pub memory_usage: usize,
    /// Active tasks
    pub active_tasks: usize,
}
/// Results from automatic parallelization benchmark
#[derive(Debug, Clone)]
pub struct AutoParallelBenchmarkResults {
    /// Total benchmark time
    pub total_time: Duration,
    /// Results for individual circuits
    pub circuit_results: Vec<CircuitParallelResult>,
    /// Average parallelization efficiency
    pub average_efficiency: f64,
    /// Average maximum parallelism
    pub average_parallelism: usize,
}
/// Parallelization strategies for circuit execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParallelizationStrategy {
    /// Analyze gate dependencies and parallelize independent operations
    DependencyAnalysis,
    /// Layer-based parallelization with depth analysis
    LayerBased,
    /// Qubit partitioning for independent subsystems
    QubitPartitioning,
    /// Hybrid approach combining multiple strategies
    Hybrid,
    /// Machine learning guided parallelization
    MLGuided,
    /// Hardware-aware parallelization
    HardwareAware,
}
/// ML-predicted parallelization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MLPredictedStrategy {
    /// High parallelism - aggressive parallel execution
    HighParallelism,
    /// Balanced parallelism - mixed approach
    BalancedParallelism,
    /// Conservative parallelism - careful dependency management
    ConservativeParallelism,
    /// Layer-optimized execution
    LayerOptimized,
}
