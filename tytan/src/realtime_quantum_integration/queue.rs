//! Queue management types for Real-time Quantum Computing Integration
//!
//! This module provides job queue management and load balancing types.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

use super::config::RealtimeConfig;
use super::resource::ResourceCapacity;
use super::types::{
    ConnectivityType, DeviceType, FairnessPolicy, HealthCheckStatus, HealthCheckType, JobPriority,
    JobStatus, LoadBalancingStrategy, PreemptionPolicy, SchedulingAlgorithm,
};

/// Queue management system
pub struct QueueManager {
    /// Job queues
    pub(crate) job_queues: HashMap<JobPriority, VecDeque<QueuedJob>>,
    /// Queue statistics
    pub(crate) queue_stats: QueueStatistics,
    /// Scheduling algorithm
    pub(crate) scheduling_algorithm: SchedulingAlgorithm,
    /// Queue policies
    pub(crate) queue_policies: QueuePolicies,
    /// Load balancer
    pub(crate) load_balancer: LoadBalancer,
}

impl QueueManager {
    pub fn new(_config: &RealtimeConfig) -> Self {
        let mut job_queues = HashMap::new();
        job_queues.insert(JobPriority::Critical, VecDeque::new());
        job_queues.insert(JobPriority::High, VecDeque::new());
        job_queues.insert(JobPriority::Normal, VecDeque::new());
        job_queues.insert(JobPriority::Low, VecDeque::new());
        job_queues.insert(JobPriority::Background, VecDeque::new());

        Self {
            job_queues,
            queue_stats: QueueStatistics::default(),
            scheduling_algorithm: SchedulingAlgorithm::PriorityBased,
            queue_policies: QueuePolicies::default(),
            load_balancer: LoadBalancer::new(),
        }
    }

    pub fn submit_job(&mut self, job: QueuedJob) -> Result<String, String> {
        let job_id = job.job_id.clone();
        let priority = job.priority.clone();

        if let Some(queue) = self.job_queues.get_mut(&priority) {
            queue.push_back(job);
            self.queue_stats.total_jobs_processed += 1;
            Ok(job_id)
        } else {
            Err("Invalid job priority".to_string())
        }
    }
}

/// Job in the queue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedJob {
    /// Job ID
    pub job_id: String,
    /// Job type
    pub job_type: JobType,
    /// Priority
    pub priority: JobPriority,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Submission time
    pub submission_time: SystemTime,
    /// Deadline
    pub deadline: Option<SystemTime>,
    /// Dependencies
    pub dependencies: Vec<String>,
    /// Job metadata
    pub metadata: JobMetadata,
    /// Current status
    pub status: JobStatus,
}

/// Types of jobs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobType {
    QuantumCircuit,
    Optimization,
    Simulation,
    Calibration,
    Maintenance,
    Hybrid,
}

/// Resource requirements for a job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Required qubits
    pub qubits_required: Option<usize>,
    /// Compute requirements
    pub compute_requirements: ComputeRequirements,
    /// Memory requirements
    pub memory_requirements: MemoryRequirements,
    /// Network requirements
    pub network_requirements: Option<NetworkRequirements>,
    /// Hardware constraints
    pub hardware_constraints: Vec<HardwareConstraint>,
}

/// Compute requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeRequirements {
    /// CPU cores
    pub cpu_cores: usize,
    /// GPU units
    pub gpu_units: Option<usize>,
    /// Quantum processing units
    pub qpu_units: Option<usize>,
    /// Estimated runtime
    pub estimated_runtime: Duration,
}

/// Memory requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRequirements {
    /// RAM (in GB)
    pub ram_gb: f64,
    /// Storage (in GB)
    pub storage_gb: f64,
    /// Temporary storage (in GB)
    pub temp_storage_gb: Option<f64>,
}

/// Network requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRequirements {
    /// Bandwidth (in Mbps)
    pub bandwidth_mbps: f64,
    /// Latency tolerance
    pub latency_tolerance: Duration,
    /// Location preferences
    pub location_preferences: Vec<String>,
}

/// Hardware constraints for job execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardwareConstraint {
    SpecificDevice(String),
    DeviceType(DeviceType),
    MinimumFidelity(f64),
    MaximumErrorRate(f64),
    Connectivity(ConnectivityRequirement),
}

/// Connectivity requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityRequirement {
    AllToAll,
    Linear,
    Grid,
    Custom(Vec<(usize, usize)>),
}

/// Job metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobMetadata {
    /// User ID
    pub user_id: String,
    /// Project ID
    pub project_id: String,
    /// Billing information
    pub billing_info: BillingInfo,
    /// Tags
    pub tags: Vec<String>,
    /// Experiment name
    pub experiment_name: Option<String>,
    /// Description
    pub description: Option<String>,
}

/// Billing information for job tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingInfo {
    /// Account ID
    pub account_id: String,
    /// Cost center
    pub cost_center: Option<String>,
    /// Budget limit
    pub budget_limit: Option<f64>,
    /// Cost estimate
    pub cost_estimate: Option<f64>,
}

/// Queue statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStatistics {
    /// Total jobs processed
    pub total_jobs_processed: usize,
    /// Average wait time
    pub average_wait_time: Duration,
    /// Queue lengths
    pub queue_lengths: HashMap<JobPriority, usize>,
    /// Throughput metrics
    pub throughput_metrics: ThroughputMetrics,
    /// Resource utilization
    pub resource_utilization: HashMap<String, f64>,
}

impl Default for QueueStatistics {
    fn default() -> Self {
        Self {
            total_jobs_processed: 0,
            average_wait_time: Duration::ZERO,
            queue_lengths: HashMap::new(),
            throughput_metrics: ThroughputMetrics {
                jobs_per_hour: 0.0,
                success_rate: 0.99,
                average_execution_time: Duration::from_secs(300),
                resource_efficiency: 0.85,
            },
            resource_utilization: HashMap::new(),
        }
    }
}

/// Throughput metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Jobs per hour
    pub jobs_per_hour: f64,
    /// Success rate
    pub success_rate: f64,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Resource efficiency
    pub resource_efficiency: f64,
}

/// Queue policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuePolicies {
    /// Maximum queue length
    pub max_queue_length: usize,
    /// Job timeout
    pub job_timeout: Duration,
    /// Preemption policy
    pub preemption_policy: PreemptionPolicy,
    /// Fairness policy
    pub fairness_policy: FairnessPolicy,
    /// Resource limits
    pub resource_limits: ResourceLimits,
}

impl Default for QueuePolicies {
    fn default() -> Self {
        Self {
            max_queue_length: 1000,
            job_timeout: Duration::from_secs(3600),
            preemption_policy: PreemptionPolicy::PriorityBased,
            fairness_policy: FairnessPolicy::WeightedFair,
            resource_limits: ResourceLimits {
                per_user_limits: HashMap::new(),
                per_project_limits: HashMap::new(),
                system_limits: ResourceCapacity {
                    compute_units: 1000.0,
                    memory_gb: 1024.0,
                    storage_gb: 10000.0,
                    network_mbps: 10000.0,
                    custom_metrics: HashMap::new(),
                },
                time_based_limits: vec![],
            },
        }
    }
}

/// Resource limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Per-user limits
    pub per_user_limits: HashMap<String, ResourceCapacity>,
    /// Per-project limits
    pub per_project_limits: HashMap<String, ResourceCapacity>,
    /// System-wide limits
    pub system_limits: ResourceCapacity,
    /// Time-based limits
    pub time_based_limits: Vec<TimeBoundLimit>,
}

/// Time-bound resource limit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeBoundLimit {
    /// Time window
    pub time_window: (SystemTime, SystemTime),
    /// Resource limits during window
    pub limits: ResourceCapacity,
    /// Priority override
    pub priority_override: Option<JobPriority>,
}

/// Load balancer
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    /// Load balancing strategy
    pub(crate) strategy: LoadBalancingStrategy,
    /// Server weights
    pub(crate) server_weights: HashMap<String, f64>,
    /// Health checks
    pub(crate) health_checks: HashMap<String, HealthCheck>,
    /// Load metrics
    pub(crate) load_metrics: HashMap<String, LoadMetrics>,
}

impl Default for LoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancer {
    pub fn new() -> Self {
        Self {
            strategy: LoadBalancingStrategy::RoundRobin,
            server_weights: HashMap::new(),
            health_checks: HashMap::new(),
            load_metrics: HashMap::new(),
        }
    }
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Check type
    pub check_type: HealthCheckType,
    /// Interval
    pub interval: Duration,
    /// Timeout
    pub timeout: Duration,
    /// Last check time
    pub last_check: SystemTime,
    /// Status
    pub status: HealthCheckStatus,
}

/// Load metrics for a server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadMetrics {
    /// Current load
    pub current_load: f64,
    /// Response time
    pub response_time: Duration,
    /// Error rate
    pub error_rate: f64,
    /// Throughput
    pub throughput: f64,
    /// Capacity utilization
    pub capacity_utilization: f64,
}
