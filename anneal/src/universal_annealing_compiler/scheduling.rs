//! Scheduling and resource allocation types.
//!
//! This module contains types for job scheduling, resource allocation,
//! and performance tracking.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::config::{ResourceAllocationStrategy, SchedulingPriority};
use super::platform::QuantumPlatform;

/// Universal resource scheduler
pub struct UniversalResourceScheduler {
    /// Scheduler configuration
    pub config: SchedulerConfig,
    /// Scheduling queue
    pub queue: SchedulingQueue,
    /// Resource allocator
    pub allocator: ResourceAllocator,
    /// Performance tracker
    pub performance_tracker: PerformanceTracker,
}

impl UniversalResourceScheduler {
    /// Create a new scheduler
    pub fn new() -> Self {
        Self {
            config: SchedulerConfig {
                algorithm: SchedulingAlgorithm::Priority,
                allocation_strategy: ResourceAllocationStrategy::CostEffective,
                fairness_policy: FairnessPolicy::ProportionalShare,
                load_balancing: LoadBalancingConfig {
                    enabled: true,
                    threshold: 0.8,
                    frequency: Duration::from_secs(60),
                    strategy: LoadBalancingStrategy::PerformanceBased,
                },
            },
            queue: SchedulingQueue {
                pending_jobs: VecDeque::new(),
                running_jobs: HashMap::new(),
                completed_jobs: VecDeque::new(),
                statistics: QueueStatistics {
                    total_jobs: 0,
                    average_wait_time: Duration::from_secs(0),
                    average_execution_time: Duration::from_secs(0),
                    throughput: 0.0,
                    utilization: 0.0,
                },
            },
            allocator: ResourceAllocator {
                config: AllocatorConfig {
                    strategy: AllocationStrategy::Optimized,
                    constraints: AllocationConstraints {
                        max_utilization: 0.9,
                        reservations: vec![],
                        affinity_constraints: vec![],
                    },
                    objectives: AllocationObjectives {
                        primary: AllocationObjective::MaximizePerformance,
                        secondary: vec![(AllocationObjective::MinimizeCost, 0.3)],
                    },
                },
                available_resources: HashMap::new(),
                allocation_history: VecDeque::new(),
            },
            performance_tracker: PerformanceTracker {
                config: TrackerConfig {
                    collection_interval: Duration::from_secs(10),
                    retention_period: Duration::from_secs(86_400),
                    alerting: AlertingConfig {
                        enabled: true,
                        thresholds: HashMap::new(),
                        channels: vec![],
                    },
                },
                metrics: HashMap::new(),
                historical_data: VecDeque::new(),
            },
        }
    }
}

impl Default for UniversalResourceScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Scheduling algorithm
    pub algorithm: SchedulingAlgorithm,
    /// Allocation strategy
    pub allocation_strategy: ResourceAllocationStrategy,
    /// Fairness policy
    pub fairness_policy: FairnessPolicy,
    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,
}

/// Scheduling algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulingAlgorithm {
    /// First-come-first-served
    FCFS,
    /// Shortest job first
    ShortestJobFirst,
    /// Priority-based
    Priority,
    /// Round-robin
    RoundRobin,
    /// Multi-level feedback queue
    MultilevelFeedback,
}

/// Fairness policies
#[derive(Debug, Clone, PartialEq)]
pub enum FairnessPolicy {
    /// Equal share
    EqualShare,
    /// Proportional share
    ProportionalShare,
    /// Weighted fair sharing
    WeightedFairSharing,
    /// Priority-based
    PriorityBased,
}

/// Load balancing configuration
#[derive(Debug, Clone)]
pub struct LoadBalancingConfig {
    /// Enable load balancing
    pub enabled: bool,
    /// Threshold for rebalancing
    pub threshold: f64,
    /// Rebalancing frequency
    pub frequency: Duration,
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
}

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancingStrategy {
    /// Round-robin
    RoundRobin,
    /// Least loaded
    LeastLoaded,
    /// Performance-based
    PerformanceBased,
    /// Cost-based
    CostBased,
}

/// Scheduling queue
#[derive(Debug)]
pub struct SchedulingQueue {
    /// Pending jobs
    pub pending_jobs: VecDeque<ScheduledJob>,
    /// Running jobs
    pub running_jobs: HashMap<String, RunningJob>,
    /// Completed jobs
    pub completed_jobs: VecDeque<CompletedJob>,
    /// Queue statistics
    pub statistics: QueueStatistics,
}

/// Scheduled job
#[derive(Debug, Clone)]
pub struct ScheduledJob {
    /// Job identifier
    pub job_id: String,
    /// Priority
    pub priority: SchedulingPriority,
    /// Estimated execution time
    pub estimated_execution_time: Duration,
    /// Resource requirements
    pub resource_requirements: JobResourceRequirements,
    /// Submission timestamp
    pub submitted_at: Instant,
}

/// Running job
#[derive(Debug, Clone)]
pub struct RunningJob {
    /// Job identifier
    pub job_id: String,
    /// Start time
    pub started_at: Instant,
    /// Allocated resources
    pub allocated_resources: AllocatedResources,
    /// Platform
    pub platform: QuantumPlatform,
}

/// Completed job
#[derive(Debug, Clone)]
pub struct CompletedJob {
    /// Job identifier
    pub job_id: String,
    /// Completion timestamp
    pub completed_at: Instant,
    /// Status
    pub status: JobStatus,
    /// Execution time
    pub execution_time: Duration,
    /// Wait time
    pub wait_time: Duration,
}

/// Job resource requirements
#[derive(Debug, Clone)]
pub struct JobResourceRequirements {
    /// Minimum qubits
    pub min_qubits: usize,
    /// Preferred qubits
    pub preferred_qubits: Option<usize>,
    /// Memory requirements
    pub memory_mb: usize,
    /// Expected duration
    pub expected_duration: Duration,
}

/// Allocated resources
#[derive(Debug, Clone)]
pub struct AllocatedResources {
    /// Allocated qubits
    pub qubits: Vec<usize>,
    /// Memory allocated
    pub memory_mb: usize,
    /// Time slot
    pub time_slot: TimeSlot,
}

/// Job status
#[derive(Debug, Clone, PartialEq)]
pub enum JobStatus {
    /// Pending
    Pending,
    /// Queued
    Queued,
    /// Running
    Running,
    /// Completed successfully
    Completed,
    /// Failed
    Failed,
    /// Cancelled
    Cancelled,
    /// Timed out
    TimedOut,
}

/// Queue statistics
#[derive(Debug, Clone)]
pub struct QueueStatistics {
    /// Total jobs
    pub total_jobs: u64,
    /// Average wait time
    pub average_wait_time: Duration,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Throughput (jobs per hour)
    pub throughput: f64,
    /// Utilization
    pub utilization: f64,
}

/// Resource allocator
#[derive(Debug)]
pub struct ResourceAllocator {
    /// Allocator configuration
    pub config: AllocatorConfig,
    /// Available resources per platform
    pub available_resources: HashMap<QuantumPlatform, AvailableResources>,
    /// Allocation history
    pub allocation_history: VecDeque<AllocationRecord>,
}

/// Allocator configuration
#[derive(Debug, Clone)]
pub struct AllocatorConfig {
    /// Allocation strategy
    pub strategy: AllocationStrategy,
    /// Constraints
    pub constraints: AllocationConstraints,
    /// Allocation objectives
    pub objectives: AllocationObjectives,
}

/// Allocation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationStrategy {
    /// First fit
    FirstFit,
    /// Best fit
    BestFit,
    /// Worst fit
    WorstFit,
    /// Optimized
    Optimized,
}

/// Allocation constraints
#[derive(Debug, Clone)]
pub struct AllocationConstraints {
    /// Maximum utilization
    pub max_utilization: f64,
    /// Reservations
    pub reservations: Vec<ResourceReservation>,
    /// Affinity constraints
    pub affinity_constraints: Vec<AffinityConstraint>,
}

/// Resource reservation
#[derive(Debug, Clone)]
pub struct ResourceReservation {
    /// Reservation identifier
    pub reservation_id: String,
    /// Reserved resources
    pub resources: ReservedResources,
    /// Start time
    pub start_time: Instant,
    /// Duration
    pub duration: Duration,
}

/// Reserved resources
#[derive(Debug, Clone)]
pub struct ReservedResources {
    /// Reserved qubits
    pub qubits: Vec<usize>,
    /// Reserved memory
    pub memory_mb: usize,
}

/// Time slot
#[derive(Debug, Clone)]
pub struct TimeSlot {
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Instant,
}

/// Affinity constraint
#[derive(Debug, Clone)]
pub struct AffinityConstraint {
    /// Target platform
    pub target: QuantumPlatform,
    /// Affinity type
    pub affinity_type: AffinityType,
    /// Strength
    pub strength: AffinityStrength,
}

/// Affinity types
#[derive(Debug, Clone, PartialEq)]
pub enum AffinityType {
    /// Must use this platform
    Required,
    /// Prefer this platform
    Preferred,
    /// Avoid this platform
    Avoid,
}

/// Affinity strength
#[derive(Debug, Clone, PartialEq)]
pub enum AffinityStrength {
    /// Weak
    Weak,
    /// Medium
    Medium,
    /// Strong
    Strong,
}

/// Allocation objectives
#[derive(Debug, Clone)]
pub struct AllocationObjectives {
    /// Primary objective
    pub primary: AllocationObjective,
    /// Secondary objectives with weights
    pub secondary: Vec<(AllocationObjective, f64)>,
}

/// Allocation objectives
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationObjective {
    /// Maximize performance
    MaximizePerformance,
    /// Minimize cost
    MinimizeCost,
    /// Minimize wait time
    MinimizeWaitTime,
    /// Maximize utilization
    MaximizeUtilization,
    /// Balance load
    BalanceLoad,
}

/// Available resources
#[derive(Debug, Clone)]
pub struct AvailableResources {
    /// Platform
    pub platform: QuantumPlatform,
    /// Capacity
    pub capacity: ResourceCapacity,
    /// Current load
    pub current_load: ResourceLoad,
}

/// Resource capacity
#[derive(Debug, Clone)]
pub struct ResourceCapacity {
    /// Total qubits
    pub total_qubits: usize,
    /// Total memory
    pub total_memory_mb: usize,
    /// Maximum concurrent jobs
    pub max_concurrent_jobs: usize,
}

/// Resource load
#[derive(Debug, Clone)]
pub struct ResourceLoad {
    /// Used qubits
    pub used_qubits: usize,
    /// Used memory
    pub used_memory_mb: usize,
    /// Active jobs
    pub active_jobs: usize,
}

/// Allocation record
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Job identifier
    pub job_id: String,
    /// Allocated platform
    pub platform: QuantumPlatform,
    /// Allocated resources
    pub resources: AllocatedResources,
    /// Allocation timestamp
    pub allocated_at: Instant,
}

/// Performance tracker
#[derive(Debug)]
pub struct PerformanceTracker {
    /// Tracker configuration
    pub config: TrackerConfig,
    /// Current metrics
    pub metrics: HashMap<String, MetricValue>,
    /// Historical data
    pub historical_data: VecDeque<PerformanceSnapshot>,
}

/// Tracker configuration
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Collection interval
    pub collection_interval: Duration,
    /// Retention period
    pub retention_period: Duration,
    /// Alerting configuration
    pub alerting: AlertingConfig,
}

/// Alerting configuration
#[derive(Debug, Clone)]
pub struct AlertingConfig {
    /// Alerting enabled
    pub enabled: bool,
    /// Thresholds
    pub thresholds: HashMap<String, f64>,
    /// Alert channels
    pub channels: Vec<AlertChannel>,
}

/// Alert channel
#[derive(Debug, Clone)]
pub struct AlertChannel {
    /// Channel name
    pub name: String,
    /// Channel type
    pub channel_type: AlertChannelType,
}

/// Alert channel types
#[derive(Debug, Clone, PartialEq)]
pub enum AlertChannelType {
    /// Email
    Email,
    /// Slack
    Slack,
    /// PagerDuty
    PagerDuty,
    /// Webhook
    Webhook,
    /// Log
    Log,
}

/// Metric value
#[derive(Debug, Clone)]
pub enum MetricValue {
    /// Counter
    Counter(u64),
    /// Gauge
    Gauge(f64),
    /// Histogram
    Histogram(Vec<f64>),
    /// Summary
    Summary { count: u64, sum: f64 },
}

/// Performance snapshot
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: Instant,
    /// Platform metrics
    pub platform_metrics: HashMap<QuantumPlatform, PlatformMetrics>,
}

/// Platform metrics
#[derive(Debug, Clone)]
pub struct PlatformMetrics {
    /// Success rate
    pub success_rate: f64,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Queue length
    pub queue_length: usize,
    /// Utilization
    pub utilization: f64,
}

/// System state
#[derive(Debug, Clone)]
pub struct SystemState {
    /// Queue lengths
    pub queue_lengths: HashMap<QuantumPlatform, usize>,
    /// Resource utilization
    pub resource_utilization: HashMap<QuantumPlatform, f64>,
}
