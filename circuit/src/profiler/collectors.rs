//! Profiling collectors for gates, memory, and resources
//!
//! This module provides specialized profilers for different aspects of
//! quantum circuit execution including gate-level profiling, memory profiling,
//! and resource utilization profiling.

use quantrs2_core::qubit::QubitId;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

// Import types from sibling modules
use super::metrics::*;

pub struct GateProfiler {
    /// Gate execution profiles
    pub gate_profiles: HashMap<String, GateProfile>,
    /// Gate timing statistics
    pub timing_stats: HashMap<String, TimingStatistics>,
    /// Gate resource usage
    pub resource_usage: HashMap<String, ResourceUsage>,
    /// Gate error analysis
    pub error_analysis: HashMap<String, ErrorAnalysis>,
}

/// Individual gate performance profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateProfile {
    /// Gate name
    pub gate_name: String,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Execution time variance
    pub execution_variance: f64,
    /// Memory usage pattern
    pub memory_pattern: MemoryPattern,
    /// Resource utilization
    pub resource_utilization: f64,
    /// Error characteristics
    pub error_characteristics: ErrorCharacteristics,
    /// Optimization potential
    pub optimization_potential: f64,
    /// Performance ranking
    pub performance_rank: u32,
}

/// Memory usage pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPattern {
    /// Peak memory usage
    pub peak_usage: usize,
    /// Average memory usage
    pub average_usage: f64,
    /// Memory allocation pattern
    pub allocation_pattern: AllocationPattern,
    /// Memory access pattern
    pub access_pattern: AccessPattern,
    /// Cache efficiency
    pub cache_efficiency: f64,
}

/// Memory allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationPattern {
    /// Constant allocation
    Constant,
    /// Linear growth
    Linear,
    /// Exponential growth
    Exponential,
    /// Periodic allocation
    Periodic,
    /// Irregular allocation
    Irregular,
}

/// Memory access patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessPattern {
    /// Sequential access
    Sequential,
    /// Random access
    Random,
    /// Stride access
    Stride { stride: usize },
    /// Cached access
    Cached,
    /// Mixed access
    Mixed,
}

/// Error characteristics for gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCharacteristics {
    /// Error rate
    pub error_rate: f64,
    /// Error distribution
    pub error_distribution: ErrorDistribution,
    /// Error correlation
    pub error_correlation: f64,
    /// Error propagation factor
    pub propagation_factor: f64,
    /// Mitigation effectiveness
    pub mitigation_effectiveness: f64,
}

/// Error distribution types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorDistribution {
    /// Normal distribution
    Normal { mean: f64, std_dev: f64 },
    /// Exponential distribution
    Exponential { lambda: f64 },
    /// Uniform distribution
    Uniform { min: f64, max: f64 },
    /// Custom distribution
    Custom { parameters: HashMap<String, f64> },
}

/// Timing statistics for gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStatistics {
    /// Minimum execution time
    pub min_time: Duration,
    /// Maximum execution time
    pub max_time: Duration,
    /// Average execution time
    pub avg_time: Duration,
    /// Median execution time
    pub median_time: Duration,
    /// Standard deviation
    pub std_deviation: Duration,
    /// Percentile distribution
    pub percentiles: HashMap<u8, Duration>,
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// GPU utilization (if applicable)
    pub gpu_utilization: Option<f64>,
    /// I/O utilization
    pub io_utilization: f64,
    /// Network utilization
    pub network_utilization: f64,
    /// Custom resource metrics
    pub custom_resources: HashMap<String, f64>,
}

/// Error analysis for gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    /// Error frequency
    pub error_frequency: f64,
    /// Error severity distribution
    pub severity_distribution: HashMap<ErrorSeverity, usize>,
    /// Common error patterns
    pub error_patterns: Vec<ErrorPattern>,
    /// Error recovery statistics
    pub recovery_stats: RecoveryStatistics,
}

/// Error severity levels
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Low severity error
    Low,
    /// Medium severity error
    Medium,
    /// High severity error
    High,
    /// Critical severity error
    Critical,
}

/// Error pattern identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// Pattern description
    pub description: String,
    /// Pattern frequency
    pub frequency: f64,
    /// Pattern confidence
    pub confidence: f64,
    /// Associated gates
    pub associated_gates: Vec<String>,
}

/// Error recovery statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStatistics {
    /// Recovery success rate
    pub success_rate: f64,
    /// Average recovery time
    pub avg_recovery_time: Duration,
    /// Recovery strategies used
    pub recovery_strategies: HashMap<String, usize>,
}

/// Memory profiler for quantum circuits
#[derive(Debug, Clone)]
pub struct MemoryProfiler {
    /// Memory usage snapshots
    pub snapshots: VecDeque<MemorySnapshot>,
    /// Memory leak detection
    pub leak_detector: LeakDetector,
    /// Memory optimization suggestions
    pub optimization_suggestions: Vec<MemoryOptimization>,
    /// Memory allocation tracking
    pub allocation_tracker: AllocationTracker,
}

/// Memory usage snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Snapshot timestamp
    pub timestamp: SystemTime,
    /// Total memory usage
    pub total_usage: usize,
    /// Memory breakdown by category
    pub breakdown: HashMap<String, usize>,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Memory efficiency score
    pub efficiency_score: f64,
    /// Fragmentation level
    pub fragmentation_level: f64,
}

/// Memory leak detection system
#[derive(Debug, Clone)]
pub struct LeakDetector {
    /// Detected leaks
    pub detected_leaks: Vec<MemoryLeak>,
    /// Leak detection threshold
    pub detection_threshold: f64,
    /// Leak analysis results
    pub analysis_results: LeakAnalysisResults,
}

/// Memory leak information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeak {
    /// Leak location
    pub location: String,
    /// Leak size
    pub size: usize,
    /// Leak growth rate
    pub growth_rate: f64,
    /// Leak confidence
    pub confidence: f64,
    /// Suggested fixes
    pub suggested_fixes: Vec<String>,
}

/// Leak analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakAnalysisResults {
    /// Total leaked memory
    pub total_leaked: usize,
    /// Leak sources
    pub leak_sources: HashMap<String, usize>,
    /// Leak severity assessment
    pub severity_assessment: LeakSeverity,
    /// Performance impact
    pub performance_impact: f64,
}

/// Leak severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeakSeverity {
    /// Minor leak
    Minor,
    /// Moderate leak
    Moderate,
    /// Major leak
    Major,
    /// Critical leak
    Critical,
}

/// Memory optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimization {
    /// Optimization type
    pub optimization_type: MemoryOptimizationType,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Implementation difficulty
    pub implementation_difficulty: OptimizationDifficulty,
    /// Description
    pub description: String,
    /// Implementation steps
    pub implementation_steps: Vec<String>,
}

/// Types of memory optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOptimizationType {
    /// Memory pool optimization
    PoolOptimization,
    /// Cache optimization
    CacheOptimization,
    /// Allocation strategy optimization
    AllocationOptimization,
    /// Memory compression
    Compression,
    /// Memory layout optimization
    LayoutOptimization,
}

/// Implementation difficulty levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationDifficulty {
    /// Easy to implement
    Easy,
    /// Medium difficulty
    Medium,
    /// Hard to implement
    Hard,
    /// Very hard to implement
    VeryHard,
}

/// Memory allocation tracking
#[derive(Debug, Clone)]
pub struct AllocationTracker {
    /// Active allocations
    pub active_allocations: HashMap<usize, AllocationInfo>,
    /// Allocation history
    pub allocation_history: VecDeque<AllocationEvent>,
    /// Allocation statistics
    pub allocation_stats: AllocationStatistics,
}

/// Individual allocation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationInfo {
    /// Allocation size
    pub size: usize,
    /// Allocation timestamp
    pub timestamp: SystemTime,
    /// Allocation source
    pub source: String,
    /// Allocation type
    pub allocation_type: AllocationType,
}

/// Types of memory allocations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationType {
    /// State vector allocation
    StateVector,
    /// Gate matrix allocation
    GateMatrix,
    /// Temporary buffer allocation
    TempBuffer,
    /// Cache allocation
    Cache,
    /// Workspace allocation
    Workspace,
}

/// Memory allocation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEvent {
    /// Event type
    pub event_type: AllocationEventType,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Allocation size
    pub size: usize,
    /// Source location
    pub source: String,
}

/// Types of allocation events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationEventType {
    /// Memory allocated
    Allocated,
    /// Memory deallocated
    Deallocated,
    /// Memory reallocated
    Reallocated,
    /// Memory moved
    Moved,
}

/// Allocation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationStatistics {
    /// Total allocations
    pub total_allocations: usize,
    /// Total deallocations
    pub total_deallocations: usize,
    /// Peak concurrent allocations
    pub peak_concurrent: usize,
    /// Average allocation size
    pub avg_allocation_size: f64,
    /// Allocation efficiency
    pub allocation_efficiency: f64,
}

/// Resource profiler for quantum circuits
#[derive(Debug, Clone)]
pub struct ResourceProfiler {
    /// CPU profiling data
    pub cpu_profiling: CpuProfilingData,
    /// GPU profiling data (if applicable)
    pub gpu_profiling: Option<GpuProfilingData>,
    /// I/O profiling data
    pub io_profiling: IoProfilingData,
    /// Network profiling data
    pub network_profiling: NetworkProfilingData,
    /// Resource bottleneck analysis
    pub bottleneck_analysis: BottleneckAnalysis,
}

/// CPU profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProfilingData {
    /// CPU utilization over time
    pub utilization_history: VecDeque<f64>,
    /// CPU core usage distribution
    pub core_usage: HashMap<u32, f64>,
    /// Cache miss rates
    pub cache_miss_rates: CacheMissRates,
    /// Instruction throughput
    pub instruction_throughput: f64,
    /// CPU-specific optimizations
    pub optimization_opportunities: Vec<CpuOptimization>,
}

/// Cache miss rate statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMissRates {
    /// L1 cache miss rate
    pub l1_miss_rate: f64,
    /// L2 cache miss rate
    pub l2_miss_rate: f64,
    /// L3 cache miss rate
    pub l3_miss_rate: f64,
    /// TLB miss rate
    pub tlb_miss_rate: f64,
}

/// CPU optimization opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuOptimization {
    /// Optimization type
    pub optimization_type: CpuOptimizationType,
    /// Potential speedup
    pub potential_speedup: f64,
    /// Implementation complexity
    pub complexity: OptimizationDifficulty,
    /// Description
    pub description: String,
}

/// Types of CPU optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CpuOptimizationType {
    /// Vectorization optimization
    Vectorization,
    /// Cache optimization
    CacheOptimization,
    /// Branch prediction optimization
    BranchPrediction,
    /// Instruction reordering
    InstructionReordering,
    /// Parallelization
    Parallelization,
}

/// GPU profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfilingData {
    /// GPU utilization
    pub gpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// Kernel execution times
    pub kernel_times: HashMap<String, Duration>,
    /// Memory transfer times
    pub transfer_times: MemoryTransferTimes,
    /// GPU-specific optimizations
    pub optimization_opportunities: Vec<GpuOptimization>,
}

/// Memory transfer timing data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTransferTimes {
    /// Host to device transfer time
    pub host_to_device: Duration,
    /// Device to host transfer time
    pub device_to_host: Duration,
    /// Device to device transfer time
    pub device_to_device: Duration,
}

/// GPU optimization opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOptimization {
    /// Optimization type
    pub optimization_type: GpuOptimizationType,
    /// Potential speedup
    pub potential_speedup: f64,
    /// Implementation complexity
    pub complexity: OptimizationDifficulty,
    /// Description
    pub description: String,
}

/// Types of GPU optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuOptimizationType {
    /// Memory coalescing
    MemoryCoalescing,
    /// Occupancy optimization
    OccupancyOptimization,
    /// Shared memory optimization
    SharedMemoryOptimization,
    /// Kernel fusion
    KernelFusion,
    /// Memory hierarchy optimization
    MemoryHierarchyOptimization,
}

/// I/O profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoProfilingData {
    /// Read throughput
    pub read_throughput: f64,
    /// Write throughput
    pub write_throughput: f64,
    /// I/O latency distribution
    pub latency_distribution: LatencyDistribution,
    /// I/O queue depth
    pub queue_depth: f64,
    /// I/O optimization opportunities
    pub optimization_opportunities: Vec<IoOptimization>,
}

/// Latency distribution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyDistribution {
    /// Minimum latency
    pub min_latency: Duration,
    /// Maximum latency
    pub max_latency: Duration,
    /// Average latency
    pub avg_latency: Duration,
    /// Latency percentiles
    pub percentiles: HashMap<u8, Duration>,
}

/// I/O optimization opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoOptimization {
    /// Optimization type
    pub optimization_type: IoOptimizationType,
    /// Potential improvement
    pub potential_improvement: f64,
    /// Implementation complexity
    pub complexity: OptimizationDifficulty,
    /// Description
    pub description: String,
}

/// Types of I/O optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IoOptimizationType {
    /// Buffer size optimization
    BufferSizeOptimization,
    /// Prefetching optimization
    PrefetchingOptimization,
    /// Batching optimization
    BatchingOptimization,
    /// Compression optimization
    CompressionOptimization,
    /// Caching optimization
    CachingOptimization,
}

/// Network profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkProfilingData {
    /// Network bandwidth utilization
    pub bandwidth_utilization: f64,
    /// Network latency
    pub network_latency: Duration,
    /// Packet loss rate
    pub packet_loss_rate: f64,
    /// Connection statistics
    pub connection_stats: ConnectionStatistics,
    /// Network optimization opportunities
    pub optimization_opportunities: Vec<NetworkOptimization>,
}

/// Network connection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStatistics {
    /// Active connections
    pub active_connections: usize,
    /// Connection establishment time
    pub connection_time: Duration,
    /// Connection reliability
    pub reliability: f64,
    /// Throughput statistics
    pub throughput_stats: ThroughputStatistics,
}

/// Throughput statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputStatistics {
    /// Average throughput
    pub avg_throughput: f64,
    /// Peak throughput
    pub peak_throughput: f64,
    /// Throughput variance
    pub throughput_variance: f64,
}

/// Network optimization opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOptimization {
    /// Optimization type
    pub optimization_type: NetworkOptimizationType,
    /// Potential improvement
    pub potential_improvement: f64,
    /// Implementation complexity
    pub complexity: OptimizationDifficulty,
    /// Description
    pub description: String,
}

/// Types of network optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkOptimizationType {
    /// Protocol optimization
    ProtocolOptimization,
    /// Connection pooling
    ConnectionPooling,
    /// Data compression
    DataCompression,
    /// Request batching
    RequestBatching,
    /// Load balancing
    LoadBalancing,
}

/// Resource bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    /// Identified bottlenecks
    pub bottlenecks: Vec<ResourceBottleneck>,
    /// Bottleneck severity ranking
    pub severity_ranking: Vec<BottleneckSeverity>,
    /// Impact analysis
    pub impact_analysis: BottleneckImpactAnalysis,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<MitigationStrategy>,
}

/// Resource bottleneck information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBottleneck {
    /// Bottleneck type
    pub bottleneck_type: ResourceBottleneckType,
    /// Severity score
    pub severity: f64,
    /// Impact on performance
    pub performance_impact: f64,
    /// Affected operations
    pub affected_operations: Vec<String>,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Types of resource bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceBottleneckType {
    /// CPU bottleneck
    Cpu,
    /// Memory bottleneck
    Memory,
    /// GPU bottleneck
    Gpu,
    /// I/O bottleneck
    Io,
    /// Network bottleneck
    Network,
    /// Mixed bottleneck
    Mixed { types: Vec<String> },
}

/// Bottleneck severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckSeverity {
    /// Bottleneck identifier
    pub bottleneck_id: String,
    /// Severity level
    pub severity: SeverityLevel,
    /// Confidence score
    pub confidence: f64,
}

/// Severity levels for bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeverityLevel {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Bottleneck impact analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckImpactAnalysis {
    /// Overall performance impact
    pub overall_impact: f64,
    /// Impact on specific metrics
    pub metric_impacts: HashMap<String, f64>,
    /// Cascading effects
    pub cascading_effects: Vec<CascadingEffect>,
    /// Cost-benefit analysis
    pub cost_benefit: CostBenefitAnalysis,
}

/// Cascading effect from bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadingEffect {
    /// Effect description
    pub description: String,
    /// Effect magnitude
    pub magnitude: f64,
    /// Affected components
    pub affected_components: Vec<String>,
}

/// Cost-benefit analysis for optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBenefitAnalysis {
    /// Implementation cost estimate
    pub implementation_cost: f64,
    /// Expected benefit
    pub expected_benefit: f64,
    /// ROI estimate
    pub roi_estimate: f64,
    /// Risk assessment
    pub risk_assessment: f64,
}

/// Mitigation strategy for bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationStrategy {
    /// Strategy name
    pub name: String,
    /// Strategy type
    pub strategy_type: MitigationStrategyType,
    /// Expected effectiveness
    pub effectiveness: f64,
    /// Implementation timeline
    pub timeline: Duration,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Types of mitigation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MitigationStrategyType {
    /// Hardware upgrade
    HardwareUpgrade,
    /// Software optimization
    SoftwareOptimization,
    /// Algorithm improvement
    AlgorithmImprovement,
    /// Resource reallocation
    ResourceReallocation,
    /// Workload distribution
    WorkloadDistribution,
}

/// Resource requirements for strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU requirements
    pub cpu_requirements: f64,
    /// Memory requirements
    pub memory_requirements: usize,
    /// Storage requirements
    pub storage_requirements: usize,
    /// Network requirements
    pub network_requirements: f64,
    /// Human resources
    pub human_resources: usize,
}
