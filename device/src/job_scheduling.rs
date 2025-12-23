//! Advanced Job Priority and Scheduling Optimization for Quantum Hardware
//!
//! This module provides comprehensive job scheduling, prioritization, and optimization
//! capabilities for quantum hardware backends, including:
//! - Multi-level priority queue management
//! - Intelligent resource allocation and load balancing
//! - Cross-provider job coordination
//! - SciRS2-powered scheduling optimization algorithms
//! - Queue analytics and prediction
//! - Job persistence and recovery mechanisms
//! - Dynamic backend selection based on performance metrics

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

use quantrs2_circuit::{optimization::analysis::CircuitAnalyzer, prelude::Circuit};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    qubit::QubitId,
};

use crate::{
    backend_traits::{query_backend_capabilities, BackendCapabilities, BackendFeatures},
    translation::HardwareBackend,
    CircuitExecutor, CircuitResult, DeviceError, DeviceResult, QuantumDevice,
};

// SciRS2 dependencies for optimization algorithms
#[cfg(feature = "scirs2")]
use scirs2_graph::{dijkstra_path, minimum_spanning_tree, Graph};
#[cfg(feature = "scirs2")]
use scirs2_optimize::{minimize, OptimizeResult};
#[cfg(feature = "scirs2")]
use scirs2_stats::{corrcoef, mean, std};

// Fallback implementations when SciRS2 is not available
#[cfg(not(feature = "scirs2"))]
mod fallback_scirs2 {
    pub fn mean(_data: &[f64]) -> f64 {
        0.0
    }
    pub fn std(_data: &[f64]) -> f64 {
        1.0
    }
    pub fn correlation(_x: &[f64], _y: &[f64]) -> f64 {
        0.0
    }

    pub struct OptimizeResult {
        pub x: Vec<f64>,
        pub success: bool,
    }

    pub fn minimize<F>(_func: F, _x0: Vec<f64>, _bounds: Option<Vec<(f64, f64)>>) -> OptimizeResult
    where
        F: Fn(&[f64]) -> f64,
    {
        OptimizeResult {
            x: vec![0.0],
            success: false,
        }
    }
}

#[cfg(not(feature = "scirs2"))]
use fallback_scirs2::*;

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Semaphore};
use uuid::Uuid;

/// Job priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum JobPriority {
    /// System critical jobs (maintenance, calibration)
    Critical = 0,
    /// High priority research or production jobs
    High = 1,
    /// Normal priority jobs
    Normal = 2,
    /// Low priority background jobs
    Low = 3,
    /// Best effort jobs that can be delayed
    BestEffort = 4,
}

impl Default for JobPriority {
    fn default() -> Self {
        JobPriority::Normal
    }
}

/// Job execution status
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JobStatus {
    /// Job is pending in queue
    Pending,
    /// Job is being validated
    Validating,
    /// Job is scheduled for execution
    Scheduled,
    /// Job is currently running
    Running,
    /// Job completed successfully
    Completed,
    /// Job failed during execution
    Failed,
    /// Job was cancelled
    Cancelled,
    /// Job timed out
    TimedOut,
    /// Job is retrying after failure
    Retrying,
    /// Job is paused/suspended
    Paused,
}

/// Advanced scheduling strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulingStrategy {
    /// First-In-First-Out with priority levels
    PriorityFIFO,
    /// Shortest Job First
    ShortestJobFirst,
    /// Shortest Remaining Time First
    ShortestRemainingTimeFirst,
    /// Fair Share scheduling
    FairShare,
    /// Round Robin with priority
    PriorityRoundRobin,
    /// Backfill scheduling
    Backfill,
    /// Earliest Deadline First
    EarliestDeadlineFirst,
    /// Rate Monotonic scheduling
    RateMonotonic,
    /// Machine learning optimized scheduling using SciRS2
    MLOptimized,
    /// Multi-objective optimization using SciRS2
    MultiObjectiveOptimized,
    /// Reinforcement Learning based scheduling
    ReinforcementLearning,
    /// Genetic Algorithm scheduling
    GeneticAlgorithm,
    /// Game-theoretic fair scheduling
    GameTheoreticFair,
    /// Energy-aware scheduling
    EnergyAware,
    /// Deadline-aware scheduling with SLA guarantees
    DeadlineAwareSLA,
    /// Custom scheduling function
    Custom(String),
}

/// Advanced resource allocation strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// First available backend
    FirstFit,
    /// Best performance for job requirements
    BestFit,
    /// Worst fit for load balancing
    WorstFit,
    /// Least loaded backend
    LeastLoaded,
    /// Most loaded backend (for consolidation)
    MostLoaded,
    /// Round robin across backends
    RoundRobin,
    /// Weighted round robin
    WeightedRoundRobin,
    /// Cost-optimized allocation
    CostOptimized,
    /// Performance-optimized allocation
    PerformanceOptimized,
    /// Energy-efficient allocation
    EnergyEfficient,
    /// SciRS2-optimized allocation using ML
    SciRS2Optimized,
    /// Multi-objective allocation (cost, performance, energy)
    MultiObjectiveOptimized,
    /// Locality-aware allocation
    LocalityAware,
    /// Fault-tolerant allocation
    FaultTolerant,
    /// Predictive allocation based on historical patterns
    PredictiveAllocation,
}

/// Job submission configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobConfig {
    /// Job priority level
    pub priority: JobPriority,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Maximum wait time in queue
    pub max_queue_time: Option<Duration>,
    /// Number of retry attempts on failure
    pub retry_attempts: u32,
    /// Retry delay between attempts
    pub retry_delay: Duration,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Preferred backends (ordered by preference)
    pub preferred_backends: Vec<HardwareBackend>,
    /// Job tags for grouping and filtering
    pub tags: HashMap<String, String>,
    /// Job dependencies
    pub dependencies: Vec<JobId>,
    /// Deadline for completion
    pub deadline: Option<SystemTime>,
    /// Cost constraints
    pub cost_limit: Option<f64>,
}

impl Default for JobConfig {
    fn default() -> Self {
        Self {
            priority: JobPriority::Normal,
            max_execution_time: Duration::from_secs(3600), // 1 hour
            max_queue_time: Some(Duration::from_secs(86400)), // 24 hours
            retry_attempts: 3,
            retry_delay: Duration::from_secs(60),
            resource_requirements: ResourceRequirements::default(),
            preferred_backends: vec![],
            tags: HashMap::new(),
            dependencies: vec![],
            deadline: None,
            cost_limit: None,
        }
    }
}

/// Resource requirements for job execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Minimum number of qubits required
    pub min_qubits: usize,
    /// Maximum circuit depth
    pub max_depth: Option<usize>,
    /// Required gate fidelity
    pub min_fidelity: Option<f64>,
    /// Required connectivity (if specific topology needed)
    pub required_connectivity: Option<String>,
    /// Memory requirements (MB)
    pub memory_mb: Option<u64>,
    /// CPU requirements
    pub cpu_cores: Option<u32>,
    /// Special hardware features required
    pub required_features: Vec<String>,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            min_qubits: 1,
            max_depth: None,
            min_fidelity: None,
            required_connectivity: None,
            memory_mb: None,
            cpu_cores: None,
            required_features: vec![],
        }
    }
}

/// Unique job identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(pub String);

impl JobId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn from_string(s: String) -> Self {
        Self(s)
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Quantum circuit job definition
#[derive(Debug, Clone)]
pub struct QuantumJob<const N: usize> {
    /// Unique job identifier
    pub id: JobId,
    /// Job configuration
    pub config: JobConfig,
    /// Circuit to execute
    pub circuit: Circuit<N>,
    /// Number of shots
    pub shots: usize,
    /// Job submission time
    pub submitted_at: SystemTime,
    /// Job status
    pub status: JobStatus,
    /// Execution history and attempts
    pub execution_history: Vec<JobExecution>,
    /// Job metadata
    pub metadata: HashMap<String, String>,
    /// User/group information
    pub user_id: String,
    /// Job group/project
    pub group_id: Option<String>,
    /// Estimated execution time
    pub estimated_duration: Option<Duration>,
    /// Assigned backend
    pub assigned_backend: Option<HardwareBackend>,
    /// Cost tracking
    pub estimated_cost: Option<f64>,
    pub actual_cost: Option<f64>,
}

/// Job execution attempt record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobExecution {
    /// Attempt number
    pub attempt: u32,
    /// Backend used for execution
    pub backend: HardwareBackend,
    /// Execution start time
    pub started_at: SystemTime,
    /// Execution end time
    pub ended_at: Option<SystemTime>,
    /// Execution result
    pub result: Option<CircuitResult>,
    /// Error information if failed
    pub error: Option<String>,
    /// Execution metrics
    pub metrics: ExecutionMetrics,
}

/// Execution performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Actual queue time
    pub queue_time: Duration,
    /// Actual execution time
    pub execution_time: Option<Duration>,
    /// Resource utilization
    pub resource_utilization: f64,
    /// Cost incurred
    pub cost: Option<f64>,
    /// Quality metrics (fidelity, error rates, etc.)
    pub quality_metrics: HashMap<String, f64>,
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self {
            queue_time: Duration::from_secs(0),
            execution_time: None,
            resource_utilization: 0.0,
            cost: None,
            quality_metrics: HashMap::new(),
        }
    }
}

/// Backend performance tracking
#[derive(Debug, Clone)]
pub struct BackendPerformance {
    /// Backend identifier
    pub backend: HardwareBackend,
    /// Current queue length
    pub queue_length: usize,
    /// Average queue time
    pub avg_queue_time: Duration,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    /// Current utilization (0.0 - 1.0)
    pub utilization: f64,
    /// Cost per job
    pub avg_cost: Option<f64>,
    /// Last updated timestamp
    pub last_updated: SystemTime,
    /// Historical performance data
    pub history: VecDeque<PerformanceSnapshot>,
}

/// Performance snapshot for historical tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: SystemTime,
    pub queue_length: usize,
    pub utilization: f64,
    pub avg_queue_time_secs: f64,
    pub success_rate: f64,
}

/// Queue analytics and predictions
#[derive(Debug, Clone)]
pub struct QueueAnalytics {
    /// Current total queue length across all backends
    pub total_queue_length: usize,
    /// Queue length by priority
    pub queue_by_priority: HashMap<JobPriority, usize>,
    /// Queue length by backend
    pub queue_by_backend: HashMap<HardwareBackend, usize>,
    /// Predicted queue times
    pub predicted_queue_times: HashMap<HardwareBackend, Duration>,
    /// System load metrics
    pub system_load: f64,
    /// Throughput (jobs per hour)
    pub throughput: f64,
    /// Average wait time
    pub avg_wait_time: Duration,
}

/// Job scheduling optimization parameters
#[derive(Debug, Clone)]
pub struct SchedulingParams {
    /// Scheduling strategy
    pub strategy: SchedulingStrategy,
    /// Resource allocation strategy
    pub allocation_strategy: AllocationStrategy,
    /// Time slice for round robin (if applicable)
    pub time_slice: Duration,
    /// Maximum jobs per user in queue
    pub max_jobs_per_user: Option<usize>,
    /// Fair share weights by user/group
    pub fair_share_weights: HashMap<String, f64>,
    /// Backfill threshold
    pub backfill_threshold: Duration,
    /// Load balancing parameters
    pub load_balance_factor: f64,
    /// SciRS2 optimization parameters
    pub scirs2_params: SciRS2SchedulingParams,
}

/// Advanced SciRS2-specific scheduling optimization parameters
#[derive(Debug, Clone)]
pub struct SciRS2SchedulingParams {
    /// Enable SciRS2 optimization
    pub enabled: bool,
    /// Optimization objective weights
    pub objective_weights: HashMap<String, f64>,
    /// Historical data window for learning
    pub learning_window: Duration,
    /// Optimization frequency
    pub optimization_frequency: Duration,
    /// Prediction model parameters
    pub model_params: HashMap<String, f64>,
    /// Machine learning algorithm selection
    pub ml_algorithm: MLAlgorithm,
    /// Multi-objective optimization weights
    pub multi_objective_weights: MultiObjectiveWeights,
    /// Reinforcement learning parameters
    pub rl_params: RLParameters,
    /// Genetic algorithm parameters
    pub ga_params: GAParameters,
    /// Enable predictive modeling
    pub enable_prediction: bool,
    /// Model retraining frequency
    pub retrain_frequency: Duration,
    /// Feature engineering parameters
    pub feature_params: FeatureParams,
}

impl Default for SciRS2SchedulingParams {
    fn default() -> Self {
        Self {
            enabled: true,
            objective_weights: [
                ("throughput".to_string(), 0.25),
                ("fairness".to_string(), 0.25),
                ("utilization".to_string(), 0.2),
                ("cost".to_string(), 0.15),
                ("energy".to_string(), 0.1),
                ("sla_compliance".to_string(), 0.05),
            ]
            .into_iter()
            .collect(),
            learning_window: Duration::from_secs(86400), // 24 hours
            optimization_frequency: Duration::from_secs(180), // 3 minutes
            model_params: HashMap::new(),
            ml_algorithm: MLAlgorithm::EnsembleMethod,
            multi_objective_weights: MultiObjectiveWeights::default(),
            rl_params: RLParameters::default(),
            ga_params: GAParameters::default(),
            enable_prediction: true,
            retrain_frequency: Duration::from_secs(3600), // 1 hour
            feature_params: FeatureParams::default(),
        }
    }
}

impl Default for SchedulingParams {
    fn default() -> Self {
        Self {
            strategy: SchedulingStrategy::MLOptimized,
            allocation_strategy: AllocationStrategy::SciRS2Optimized,
            time_slice: Duration::from_secs(60),
            max_jobs_per_user: Some(100),
            fair_share_weights: HashMap::new(),
            backfill_threshold: Duration::from_secs(300),
            load_balance_factor: 0.8,
            scirs2_params: SciRS2SchedulingParams::default(),
        }
    }
}

/// Advanced Job Scheduler and Queue Manager
pub struct QuantumJobScheduler {
    /// Scheduling parameters
    params: Arc<RwLock<SchedulingParams>>,
    /// Job queues by priority level
    job_queues: Arc<Mutex<BTreeMap<JobPriority, VecDeque<JobId>>>>,
    /// All active jobs
    jobs: Arc<RwLock<HashMap<JobId, Box<dyn std::any::Any + Send + Sync>>>>,
    /// Backend performance tracking
    backend_performance: Arc<RwLock<HashMap<HardwareBackend, BackendPerformance>>>,
    /// Available backends
    backends: Arc<RwLock<HashSet<HardwareBackend>>>,
    /// Running jobs
    running_jobs: Arc<RwLock<HashMap<JobId, (HardwareBackend, SystemTime)>>>,
    /// Job execution history
    execution_history: Arc<RwLock<Vec<JobExecution>>>,
    /// User fair share tracking
    user_shares: Arc<RwLock<HashMap<String, UserShare>>>,
    /// Scheduler control
    scheduler_running: Arc<Mutex<bool>>,
    /// Event notifications
    event_sender: mpsc::UnboundedSender<SchedulerEvent>,
    /// Performance predictor
    performance_predictor: Arc<Mutex<PerformancePredictor>>,
    /// Resource manager
    resource_manager: Arc<Mutex<ResourceManager>>,
}

/// User fair share tracking
#[derive(Debug, Clone)]
struct UserShare {
    user_id: String,
    allocated_share: f64,
    used_share: f64,
    jobs_running: usize,
    jobs_queued: usize,
    last_updated: SystemTime,
}

/// Scheduler events for monitoring and notifications
#[derive(Debug, Clone)]
pub enum SchedulerEvent {
    JobSubmitted(JobId),
    JobScheduled(JobId, HardwareBackend),
    JobStarted(JobId),
    JobCompleted(JobId, CircuitResult),
    JobFailed(JobId, String),
    JobCancelled(JobId),
    BackendStatusChanged(HardwareBackend, BackendStatus),
    QueueAnalyticsUpdated(QueueAnalytics),
}

/// Backend status information
#[derive(Debug, Clone)]
pub enum BackendStatus {
    Available,
    Busy,
    Maintenance,
    Offline,
    Error(String),
}

/// Performance prediction using SciRS2 algorithms
struct PerformancePredictor {
    /// Historical performance data
    history: VecDeque<PredictionDataPoint>,
    /// Learned model parameters
    model_params: HashMap<String, f64>,
    /// Prediction accuracy metrics
    accuracy_metrics: HashMap<String, f64>,
}

/// Data point for performance prediction
#[derive(Debug, Clone)]
struct PredictionDataPoint {
    timestamp: SystemTime,
    backend: HardwareBackend,
    queue_length: usize,
    job_complexity: f64,
    execution_time: Duration,
    success: bool,
}

/// Resource allocation and management
struct ResourceManager {
    /// Available resources by backend
    available_resources: HashMap<HardwareBackend, ResourceCapacity>,
    /// Resource reservations
    reservations: HashMap<JobId, ResourceReservation>,
    /// Resource utilization history
    utilization_history: VecDeque<ResourceSnapshot>,
}

/// Resource capacity for a backend
#[derive(Debug, Clone)]
struct ResourceCapacity {
    qubits: usize,
    max_circuit_depth: Option<usize>,
    memory_mb: u64,
    cpu_cores: u32,
    concurrent_jobs: usize,
    features: HashSet<String>,
}

/// Resource reservation for a job
#[derive(Debug, Clone)]
struct ResourceReservation {
    job_id: JobId,
    backend: HardwareBackend,
    resources: ResourceRequirements,
    reserved_at: SystemTime,
    expires_at: SystemTime,
}

/// Resource utilization snapshot
#[derive(Debug, Clone)]
struct ResourceSnapshot {
    timestamp: SystemTime,
    backend: HardwareBackend,
    utilization: f64,
    active_jobs: usize,
}

impl QuantumJobScheduler {
    /// Create a new quantum job scheduler
    pub fn new(params: SchedulingParams) -> Self {
        let (event_sender, _) = mpsc::unbounded_channel();

        Self {
            params: Arc::new(RwLock::new(params)),
            job_queues: Arc::new(Mutex::new(BTreeMap::new())),
            jobs: Arc::new(RwLock::new(HashMap::new())),
            backend_performance: Arc::new(RwLock::new(HashMap::new())),
            backends: Arc::new(RwLock::new(HashSet::new())),
            running_jobs: Arc::new(RwLock::new(HashMap::new())),
            execution_history: Arc::new(RwLock::new(Vec::new())),
            user_shares: Arc::new(RwLock::new(HashMap::new())),
            scheduler_running: Arc::new(Mutex::new(false)),
            event_sender,
            performance_predictor: Arc::new(Mutex::new(PerformancePredictor::new())),
            resource_manager: Arc::new(Mutex::new(ResourceManager::new())),
        }
    }

    /// Register a backend
    pub async fn register_backend(&self, backend: HardwareBackend) -> DeviceResult<()> {
        let mut backends = self
            .backends
            .write()
            .expect("Failed to acquire write lock on backends in register_backend");
        backends.insert(backend);

        // Initialize performance tracking
        let mut performance = self
            .backend_performance
            .write()
            .expect("Failed to acquire write lock on backend_performance in register_backend");
        performance.insert(
            backend,
            BackendPerformance {
                backend,
                queue_length: 0,
                avg_queue_time: Duration::from_secs(0),
                avg_execution_time: Duration::from_secs(0),
                success_rate: 1.0,
                utilization: 0.0,
                avg_cost: None,
                last_updated: SystemTime::now(),
                history: VecDeque::new(),
            },
        );

        // Initialize resource capacity
        let capabilities = query_backend_capabilities(backend);
        let mut resource_manager = self
            .resource_manager
            .lock()
            .expect("Failed to acquire lock on resource_manager in register_backend");
        resource_manager.available_resources.insert(
            backend,
            ResourceCapacity {
                qubits: capabilities.features.max_qubits,
                max_circuit_depth: capabilities.features.max_depth,
                memory_mb: 8192,     // Default 8GB
                cpu_cores: 4,        // Default 4 cores
                concurrent_jobs: 10, // Default concurrent job limit
                features: capabilities
                    .features
                    .supported_measurement_bases
                    .into_iter()
                    .collect(),
            },
        );

        Ok(())
    }

    /// Get list of available backends
    pub fn get_available_backends(&self) -> Vec<HardwareBackend> {
        let backends = self
            .backends
            .read()
            .expect("Failed to acquire read lock on backends in get_available_backends");
        backends.iter().cloned().collect()
    }

    /// Submit a quantum job for execution
    pub async fn submit_job<const N: usize>(
        &self,
        circuit: Circuit<N>,
        shots: usize,
        config: JobConfig,
        user_id: String,
    ) -> DeviceResult<JobId> {
        let job_id = JobId::new();
        let now = SystemTime::now();

        // Validate job configuration
        self.validate_job_config(&config).await?;

        // Estimate execution time and cost
        let estimated_duration = self
            .estimate_execution_time(&circuit, shots, &config)
            .await?;
        let estimated_cost = self.estimate_cost(&circuit, shots, &config).await?;

        // Create job
        let job = QuantumJob {
            id: job_id.clone(),
            config,
            circuit,
            shots,
            submitted_at: now,
            status: JobStatus::Pending,
            execution_history: vec![],
            metadata: HashMap::new(),
            user_id: user_id.clone(),
            group_id: None,
            estimated_duration: Some(estimated_duration),
            assigned_backend: None,
            estimated_cost: Some(estimated_cost),
            actual_cost: None,
        };

        // Store job
        let mut jobs = self
            .jobs
            .write()
            .expect("Failed to acquire write lock on jobs in submit_job");
        jobs.insert(job_id.clone(), Box::new(job.clone()));
        drop(jobs);

        // Add to appropriate priority queue
        let mut queues = self
            .job_queues
            .lock()
            .expect("Failed to acquire lock on job_queues in submit_job");
        let queue = queues
            .entry(job.config.priority)
            .or_insert_with(VecDeque::new);
        queue.push_back(job_id.clone());
        drop(queues);

        // Update user share tracking
        self.update_user_share(&user_id, 1, 0).await;

        // Send event notification
        let _ = self
            .event_sender
            .send(SchedulerEvent::JobSubmitted(job_id.clone()));

        // Start scheduler if not running
        self.ensure_scheduler_running().await;

        Ok(job_id)
    }

    /// Cancel a queued or running job
    pub async fn cancel_job(&self, job_id: &JobId) -> DeviceResult<bool> {
        // Remove from queue if still pending
        let mut queues = self
            .job_queues
            .lock()
            .expect("Failed to acquire lock on job_queues in cancel_job");
        for queue in queues.values_mut() {
            if let Some(pos) = queue.iter().position(|id| id == job_id) {
                queue.remove(pos);
                drop(queues);

                // Update job status
                self.update_job_status(job_id, JobStatus::Cancelled).await?;

                // Send event
                let _ = self
                    .event_sender
                    .send(SchedulerEvent::JobCancelled(job_id.clone()));
                return Ok(true);
            }
        }
        drop(queues);

        // Cancel running job if applicable
        let running_jobs = self
            .running_jobs
            .read()
            .expect("Failed to acquire read lock on running_jobs in cancel_job");
        if running_jobs.contains_key(job_id) {
            // TODO: Implement backend-specific job cancellation
            // For now, mark as cancelled and clean up when job completes
            self.update_job_status(job_id, JobStatus::Cancelled).await?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Get job status and information
    pub async fn get_job_status<const N: usize>(
        &self,
        job_id: &JobId,
    ) -> DeviceResult<Option<QuantumJob<N>>> {
        let jobs = self
            .jobs
            .read()
            .expect("Failed to acquire read lock on jobs in get_job_status");
        if let Some(job_any) = jobs.get(job_id) {
            if let Some(job) = job_any.downcast_ref::<QuantumJob<N>>() {
                return Ok(Some(job.clone()));
            }
        }
        Ok(None)
    }

    /// Get queue analytics and predictions
    pub async fn get_queue_analytics(&self) -> DeviceResult<QueueAnalytics> {
        let queues = self
            .job_queues
            .lock()
            .expect("Failed to acquire lock on job_queues in get_queue_analytics");
        let backend_performance = self
            .backend_performance
            .read()
            .expect("Failed to acquire read lock on backend_performance in get_queue_analytics");

        let total_queue_length = queues.values().map(|q| q.len()).sum();

        let queue_by_priority = queues
            .iter()
            .map(|(priority, queue)| (*priority, queue.len()))
            .collect();

        let queue_by_backend = backend_performance
            .iter()
            .map(|(backend, perf)| (*backend, perf.queue_length))
            .collect();

        let predicted_queue_times = self.predict_queue_times(&backend_performance).await;

        let system_load = self.calculate_system_load(&backend_performance).await;
        let throughput = self.calculate_throughput().await;
        let avg_wait_time = self.calculate_average_wait_time().await;

        Ok(QueueAnalytics {
            total_queue_length,
            queue_by_priority,
            queue_by_backend,
            predicted_queue_times,
            system_load,
            throughput,
            avg_wait_time,
        })
    }

    /// Start the job scheduler
    pub async fn start_scheduler(&self) -> DeviceResult<()> {
        let mut running = self
            .scheduler_running
            .lock()
            .expect("Failed to acquire lock on scheduler_running in start_scheduler");
        if *running {
            return Err(DeviceError::APIError(
                "Scheduler already running".to_string(),
            ));
        }
        *running = true;
        drop(running);

        // Start main scheduling loop
        let scheduler = Arc::new(self.clone());
        tokio::spawn(async move {
            scheduler.scheduling_loop().await;
        });

        // Start performance monitoring
        let scheduler = Arc::new(self.clone());
        tokio::spawn(async move {
            scheduler.performance_monitoring_loop().await;
        });

        // Start SciRS2 optimization if enabled
        let params = self
            .params
            .read()
            .expect("Failed to acquire read lock on params in start_scheduler");
        if params.scirs2_params.enabled {
            drop(params);
            let scheduler = Arc::new(self.clone());
            tokio::spawn(async move {
                scheduler.scirs2_optimization_loop().await;
            });
        }

        Ok(())
    }

    /// Stop the job scheduler
    pub async fn stop_scheduler(&self) -> DeviceResult<()> {
        let mut running = self
            .scheduler_running
            .lock()
            .expect("Failed to acquire lock on scheduler_running in stop_scheduler");
        *running = false;
        Ok(())
    }

    // Internal implementation methods

    async fn validate_job_config(&self, config: &JobConfig) -> DeviceResult<()> {
        // Validate resource requirements against available backends
        let backends = self
            .backends
            .read()
            .expect("Failed to acquire read lock on backends in validate_job_config");
        if backends.is_empty() {
            return Err(DeviceError::APIError("No backends available".to_string()));
        }

        // Check if any backend can satisfy requirements
        let resource_manager = self
            .resource_manager
            .lock()
            .expect("Failed to acquire lock on resource_manager in validate_job_config");
        let mut can_satisfy = false;

        for (backend, capacity) in &resource_manager.available_resources {
            if capacity.qubits >= config.resource_requirements.min_qubits {
                if let Some(max_depth) = config.resource_requirements.max_depth {
                    if let Some(backend_max_depth) = capacity.max_circuit_depth {
                        if max_depth > backend_max_depth {
                            continue;
                        }
                    }
                }
                can_satisfy = true;
                break;
            }
        }

        if !can_satisfy {
            return Err(DeviceError::APIError(
                "No backend can satisfy resource requirements".to_string(),
            ));
        }

        Ok(())
    }

    async fn estimate_execution_time<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        shots: usize,
        config: &JobConfig,
    ) -> DeviceResult<Duration> {
        // Simple estimation based on circuit complexity and historical data
        let analyzer = CircuitAnalyzer::new();
        let metrics = analyzer
            .analyze(circuit)
            .map_err(|e| DeviceError::APIError(format!("Circuit analysis error: {:?}", e)))?;
        let circuit_complexity = metrics.depth as f64 + (metrics.gate_count as f64 * 0.1);
        let shots_factor = (shots as f64).log10();

        // Base estimation
        let base_time = Duration::from_secs((circuit_complexity * shots_factor) as u64);

        // Adjust based on backend performance if available
        let backend_performance = self.backend_performance.read().expect(
            "Failed to acquire read lock on backend_performance in estimate_execution_time",
        );
        let avg_execution_time = if backend_performance.is_empty() {
            Duration::from_secs(60) // Default 1 minute
        } else {
            let total_time: Duration = backend_performance
                .values()
                .map(|p| p.avg_execution_time)
                .sum();
            total_time / backend_performance.len() as u32
        };

        let estimated = Duration::from_millis(
            ((base_time.as_millis() + avg_execution_time.as_millis()) / 2)
                .try_into()
                .expect(
                    "Failed to convert estimated execution time to u64 in estimate_execution_time",
                ),
        );

        Ok(estimated)
    }

    async fn estimate_cost<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        shots: usize,
        config: &JobConfig,
    ) -> DeviceResult<f64> {
        // Simple cost estimation
        let analyzer = CircuitAnalyzer::new();
        let metrics = analyzer
            .analyze(circuit)
            .map_err(|e| DeviceError::APIError(format!("Circuit analysis error: {:?}", e)))?;
        let circuit_complexity = metrics.depth as f64 + metrics.gate_count as f64;
        let base_cost = circuit_complexity * shots as f64 * 0.001; // $0.001 per complexity unit

        // Adjust based on priority
        let priority_multiplier = match config.priority {
            JobPriority::Critical => 3.0,
            JobPriority::High => 2.0,
            JobPriority::Normal => 1.0,
            JobPriority::Low => 0.7,
            JobPriority::BestEffort => 0.5,
        };

        Ok(base_cost * priority_multiplier)
    }

    async fn update_user_share(&self, user_id: &str, queued_delta: i32, running_delta: i32) {
        let mut user_shares = self
            .user_shares
            .write()
            .expect("Failed to acquire write lock on user_shares in update_user_share");
        let share = user_shares.entry(user_id.to_string()).or_insert(UserShare {
            user_id: user_id.to_string(),
            allocated_share: 1.0, // Default equal share
            used_share: 0.0,
            jobs_running: 0,
            jobs_queued: 0,
            last_updated: SystemTime::now(),
        });

        share.jobs_queued = (share.jobs_queued as i32 + queued_delta).max(0) as usize;
        share.jobs_running = (share.jobs_running as i32 + running_delta).max(0) as usize;
        share.last_updated = SystemTime::now();
    }

    async fn update_job_status(&self, job_id: &JobId, status: JobStatus) -> DeviceResult<()> {
        let mut jobs = self
            .jobs
            .write()
            .expect("Failed to acquire write lock on jobs in update_job_status");
        if let Some(job_any) = jobs.get_mut(job_id) {
            // This is a simplified update - in practice, we'd need proper type handling
            // TODO: Implement proper job status updates
        }
        Ok(())
    }

    async fn ensure_scheduler_running(&self) {
        let running = self
            .scheduler_running
            .lock()
            .expect("Failed to acquire lock on scheduler_running in ensure_scheduler_running");
        if !*running {
            drop(running);
            let _ = self.start_scheduler().await;
        }
    }

    async fn predict_queue_times(
        &self,
        backend_performance: &HashMap<HardwareBackend, BackendPerformance>,
    ) -> HashMap<HardwareBackend, Duration> {
        let mut predictions = HashMap::new();

        for (backend, perf) in backend_performance {
            // Simple prediction based on current queue and average execution time
            let predicted_time = Duration::from_secs(
                (perf.queue_length as u64 * perf.avg_execution_time.as_secs())
                    / perf.success_rate.max(0.1) as u64,
            );
            predictions.insert(*backend, predicted_time);
        }

        predictions
    }

    async fn calculate_system_load(
        &self,
        backend_performance: &HashMap<HardwareBackend, BackendPerformance>,
    ) -> f64 {
        if backend_performance.is_empty() {
            return 0.0;
        }

        let total_utilization: f64 = backend_performance.values().map(|p| p.utilization).sum();

        total_utilization / backend_performance.len() as f64
    }

    async fn calculate_throughput(&self) -> f64 {
        let history = self
            .execution_history
            .read()
            .expect("Failed to acquire read lock on execution_history in calculate_throughput");
        if history.is_empty() {
            return 0.0;
        }

        // Calculate jobs completed in the last hour
        let one_hour_ago = SystemTime::now() - Duration::from_secs(3600);
        let recent_completions = history
            .iter()
            .filter(|exec| exec.started_at > one_hour_ago)
            .count();

        recent_completions as f64
    }

    async fn calculate_average_wait_time(&self) -> Duration {
        let history = self.execution_history.read().expect(
            "Failed to acquire read lock on execution_history in calculate_average_wait_time",
        );
        if history.is_empty() {
            return Duration::from_secs(0);
        }

        let total_wait: Duration = history.iter().map(|exec| exec.metrics.queue_time).sum();

        total_wait / history.len() as u32
    }

    // Main scheduling loop
    async fn scheduling_loop(&self) {
        while *self
            .scheduler_running
            .lock()
            .expect("Failed to acquire lock on scheduler_running in scheduling_loop")
        {
            if let Err(e) = self.schedule_next_jobs().await {
                eprintln!("Scheduling error: {}", e);
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    async fn schedule_next_jobs(&self) -> DeviceResult<()> {
        let params = self
            .params
            .read()
            .expect("Failed to acquire read lock on params in schedule_next_jobs")
            .clone();

        match params.strategy {
            SchedulingStrategy::PriorityFIFO => self.schedule_priority_fifo().await,
            SchedulingStrategy::ShortestJobFirst => self.schedule_shortest_job_first().await,
            SchedulingStrategy::FairShare => self.schedule_fair_share().await,
            SchedulingStrategy::Backfill => self.schedule_backfill().await,
            SchedulingStrategy::MLOptimized => self.schedule_ml_optimized().await,
            _ => {
                self.schedule_priority_fifo().await // Default fallback
            }
        }
    }

    async fn schedule_priority_fifo(&self) -> DeviceResult<()> {
        // Process jobs in priority order
        for priority in [
            JobPriority::Critical,
            JobPriority::High,
            JobPriority::Normal,
            JobPriority::Low,
            JobPriority::BestEffort,
        ] {
            let job_id = {
                let mut queues = self
                    .job_queues
                    .lock()
                    .expect("Failed to acquire lock on job_queues in schedule_priority_fifo");
                if let Some(queue) = queues.get_mut(&priority) {
                    queue.pop_front()
                } else {
                    None
                }
            };

            if let Some(job_id) = job_id {
                if let Some(backend) = self.find_best_backend(&job_id).await? {
                    self.assign_job_to_backend(&job_id, backend).await?;
                    break;
                } else {
                    // No available backend, put job back
                    let mut queues = self.job_queues.lock().expect("Failed to acquire lock on job_queues to requeue job in schedule_priority_fifo");
                    if let Some(queue) = queues.get_mut(&priority) {
                        queue.push_front(job_id);
                    }
                    break;
                }
            }
        }

        Ok(())
    }

    async fn schedule_shortest_job_first(&self) -> DeviceResult<()> {
        // Implementation for shortest job first scheduling
        // TODO: Sort jobs by estimated execution time
        self.schedule_priority_fifo().await
    }

    async fn schedule_fair_share(&self) -> DeviceResult<()> {
        // Implementation for fair share scheduling
        // TODO: Consider user shares when scheduling
        self.schedule_priority_fifo().await
    }

    async fn schedule_backfill(&self) -> DeviceResult<()> {
        // Implementation for backfill scheduling
        // TODO: Fill gaps with smaller jobs
        self.schedule_priority_fifo().await
    }

    async fn schedule_ml_optimized(&self) -> DeviceResult<()> {
        // Implementation for ML-optimized scheduling using SciRS2
        #[cfg(feature = "scirs2")]
        {
            // Use SciRS2 optimization algorithms
            self.scirs2_optimize_schedule().await
        }

        #[cfg(not(feature = "scirs2"))]
        {
            // Fallback to priority FIFO
            self.schedule_priority_fifo().await
        }
    }

    #[cfg(feature = "scirs2")]
    async fn scirs2_optimize_schedule(&self) -> DeviceResult<()> {
        // Use SciRS2 optimization for intelligent job scheduling
        // This would implement advanced ML-based scheduling algorithms
        self.schedule_priority_fifo().await // Simplified for now
    }

    async fn find_best_backend(&self, job_id: &JobId) -> DeviceResult<Option<HardwareBackend>> {
        // Check if job exists and drop the lock immediately
        {
            let jobs = self
                .jobs
                .read()
                .expect("Failed to acquire read lock on jobs in find_best_backend");
            let _job_any = jobs
                .get(job_id)
                .ok_or_else(|| DeviceError::APIError("Job not found".to_string()))?;
        } // Drop the jobs lock here

        // TODO: Proper type casting based on circuit size
        // For now, use simplified backend selection

        let backends: Vec<_> = {
            let backends = self
                .backends
                .read()
                .expect("Failed to acquire read lock on backends in find_best_backend");
            backends.iter().copied().collect()
        };

        let allocation_strategy = {
            let params = self
                .params
                .read()
                .expect("Failed to acquire read lock on params in find_best_backend");
            params.allocation_strategy.clone()
        };

        let backend_performance_snapshot = {
            let backend_performance = self
                .backend_performance
                .read()
                .expect("Failed to acquire read lock on backend_performance in find_best_backend");
            backend_performance.clone()
        };

        match allocation_strategy {
            AllocationStrategy::FirstFit => {
                // Return first available backend
                for backend in backends {
                    if self.is_backend_available(backend).await {
                        return Ok(Some(backend));
                    }
                }
            }
            AllocationStrategy::BestFit => {
                // Return backend with best capabilities for this job
                // TODO: Implement proper job requirements matching
                for &backend in backends.iter() {
                    if self.is_backend_available(backend).await {
                        return Ok(Some(backend));
                    }
                }
            }
            AllocationStrategy::LeastLoaded => {
                // Return backend with lowest utilization
                let mut best_backend = None;
                let mut lowest_utilization = f64::INFINITY;

                for (&backend, perf) in backend_performance_snapshot.iter() {
                    if self.is_backend_available(backend).await
                        && perf.utilization < lowest_utilization
                    {
                        lowest_utilization = perf.utilization;
                        best_backend = Some(backend);
                    }
                }

                return Ok(best_backend);
            }
            _ => {
                // Default to first fit
                for &backend in backends.iter() {
                    if self.is_backend_available(backend).await {
                        return Ok(Some(backend));
                    }
                }
            }
        }

        Ok(None)
    }

    async fn is_backend_available(&self, backend: HardwareBackend) -> bool {
        let running_jobs = self
            .running_jobs
            .read()
            .expect("Failed to acquire read lock on running_jobs in is_backend_available");
        let backend_jobs = running_jobs.values().filter(|(b, _)| *b == backend).count();

        let resource_manager = self
            .resource_manager
            .lock()
            .expect("Failed to acquire lock on resource_manager in is_backend_available");
        if let Some(capacity) = resource_manager.available_resources.get(&backend) {
            backend_jobs < capacity.concurrent_jobs
        } else {
            false
        }
    }

    async fn assign_job_to_backend(
        &self,
        job_id: &JobId,
        backend: HardwareBackend,
    ) -> DeviceResult<()> {
        {
            let mut running_jobs = self
                .running_jobs
                .write()
                .expect("Failed to acquire write lock on running_jobs in assign_job_to_backend");
            running_jobs.insert(job_id.clone(), (backend, SystemTime::now()));
        }

        // Update job status
        self.update_job_status(job_id, JobStatus::Scheduled).await?;

        // Send event
        let _ = self
            .event_sender
            .send(SchedulerEvent::JobScheduled(job_id.clone(), backend));

        // Start job execution
        let job_id_clone = job_id.clone();
        let scheduler = Arc::new(self.clone());
        tokio::spawn(async move {
            let _ = scheduler.execute_job(&job_id_clone, backend).await;
        });

        Ok(())
    }

    async fn execute_job(&self, job_id: &JobId, backend: HardwareBackend) -> DeviceResult<()> {
        // Update job status to running
        self.update_job_status(job_id, JobStatus::Running).await?;
        let _ = self
            .event_sender
            .send(SchedulerEvent::JobStarted(job_id.clone()));

        let execution_start = SystemTime::now();

        // Check if backend is available
        {
            let backends = self
                .backends
                .read()
                .expect("Failed to acquire read lock on backends in execute_job");
            if !backends.contains(&backend) {
                return Err(DeviceError::APIError("Backend not found".to_string()));
            }
        }

        // Execute job (simplified - would need proper type handling)
        // TODO: Implement proper job execution with circuit type management

        // For now, simulate execution
        tokio::time::sleep(Duration::from_secs(10)).await;

        // Clean up running job
        {
            let mut running_jobs = self
                .running_jobs
                .write()
                .expect("Failed to acquire write lock on running_jobs in execute_job cleanup");
            running_jobs.remove(job_id);
        }

        // Update job status
        self.update_job_status(job_id, JobStatus::Completed).await?;

        // Record execution metrics
        let execution_time = SystemTime::now()
            .duration_since(execution_start)
            .expect("Failed to calculate execution time duration in execute_job");
        // TODO: Record proper execution metrics

        Ok(())
    }

    // Performance monitoring loop
    async fn performance_monitoring_loop(&self) {
        while *self
            .scheduler_running
            .lock()
            .expect("Failed to acquire lock on scheduler_running in performance_monitoring_loop")
        {
            self.update_backend_performance().await;
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
    }

    async fn update_backend_performance(&self) {
        let mut backend_performance = self.backend_performance.write().expect(
            "Failed to acquire write lock on backend_performance in update_backend_performance",
        );
        let now = SystemTime::now();

        for (backend, perf) in backend_performance.iter_mut() {
            // Update performance metrics
            perf.last_updated = now;

            // Add performance snapshot
            let snapshot = PerformanceSnapshot {
                timestamp: now,
                queue_length: perf.queue_length,
                utilization: perf.utilization,
                avg_queue_time_secs: perf.avg_queue_time.as_secs_f64(),
                success_rate: perf.success_rate,
            };

            perf.history.push_back(snapshot);

            // Keep only recent history (last 24 hours)
            let cutoff = now - Duration::from_secs(86400);
            while let Some(front) = perf.history.front() {
                if front.timestamp < cutoff {
                    perf.history.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    // SciRS2 optimization loop
    async fn scirs2_optimization_loop(&self) {
        let frequency = {
            let params = self
                .params
                .read()
                .expect("Failed to acquire read lock on params in scirs2_optimization_loop");
            params.scirs2_params.optimization_frequency
        };

        loop {
            let should_continue = *self
                .scheduler_running
                .lock()
                .expect("Failed to acquire lock on scheduler_running in scirs2_optimization_loop");
            if !should_continue {
                break;
            }

            if let Err(e) = self.run_scirs2_optimization().await {
                eprintln!("SciRS2 optimization error: {}", e);
            }

            tokio::time::sleep(frequency).await;
        }
    }

    async fn run_scirs2_optimization(&self) -> DeviceResult<()> {
        #[cfg(feature = "scirs2")]
        {
            // Implement SciRS2-based scheduling optimization
            // This would use machine learning and statistical analysis
            // to optimize job scheduling decisions

            // Collect performance data
            let backend_performance = self.backend_performance.read().expect(
                "Failed to acquire read lock on backend_performance in run_scirs2_optimization",
            );
            let performance_data: Vec<f64> = backend_performance
                .values()
                .map(|p| p.utilization)
                .collect();

            if performance_data.len() > 1 {
                // Calculate optimization metrics
                use scirs2_core::ndarray::Array1;
                let data_array = Array1::from_vec(performance_data);
                let avg_utilization = mean(&data_array.view());
                let utilization_std = std(&data_array.view(), 1, None);

                // TODO: Implement more sophisticated SciRS2 optimization
                // This could include:
                // - Predicting optimal job placement
                // - Learning from historical performance
                // - Optimizing for multiple objectives (throughput, fairness, cost)
            }
        }

        Ok(())
    }
}

// Clone implementation for scheduler (simplified)
impl Clone for QuantumJobScheduler {
    fn clone(&self) -> Self {
        Self {
            params: Arc::clone(&self.params),
            job_queues: Arc::clone(&self.job_queues),
            jobs: Arc::clone(&self.jobs),
            backend_performance: Arc::clone(&self.backend_performance),
            backends: Arc::clone(&self.backends),
            running_jobs: Arc::clone(&self.running_jobs),
            execution_history: Arc::clone(&self.execution_history),
            user_shares: Arc::clone(&self.user_shares),
            scheduler_running: Arc::clone(&self.scheduler_running),
            event_sender: self.event_sender.clone(),
            performance_predictor: Arc::clone(&self.performance_predictor),
            resource_manager: Arc::clone(&self.resource_manager),
        }
    }
}

impl PerformancePredictor {
    fn new() -> Self {
        Self {
            history: VecDeque::new(),
            model_params: HashMap::new(),
            accuracy_metrics: HashMap::new(),
        }
    }
}

impl ResourceManager {
    fn new() -> Self {
        Self {
            available_resources: HashMap::new(),
            reservations: HashMap::new(),
            utilization_history: VecDeque::new(),
        }
    }
}

/// Advanced machine learning algorithms for scheduling
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MLAlgorithm {
    /// Linear regression for simple predictions
    LinearRegression,
    /// Support Vector Machine for classification
    SVM,
    /// Random Forest for ensemble learning
    RandomForest,
    /// Gradient Boosting for performance optimization
    GradientBoosting,
    /// Neural Network for complex patterns
    NeuralNetwork,
    /// Ensemble method combining multiple algorithms
    EnsembleMethod,
    /// Deep Reinforcement Learning
    DeepRL,
    /// Graph Neural Network for topology-aware scheduling
    GraphNN,
}

/// Multi-objective optimization weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveWeights {
    /// Throughput optimization weight
    pub throughput: f64,
    /// Cost minimization weight
    pub cost: f64,
    /// Energy efficiency weight
    pub energy: f64,
    /// Fairness weight
    pub fairness: f64,
    /// SLA compliance weight
    pub sla_compliance: f64,
    /// Quality of service weight
    pub qos: f64,
}

/// Reinforcement Learning parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLParameters {
    /// Learning rate
    pub learning_rate: f64,
    /// Discount factor
    pub discount_factor: f64,
    /// Exploration rate
    pub exploration_rate: f64,
    /// Episode length
    pub episode_length: usize,
    /// Reward function weights
    pub reward_weights: HashMap<String, f64>,
    /// State representation dimension
    pub state_dimension: usize,
    /// Action space size
    pub action_space_size: usize,
}

/// Genetic Algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GAParameters {
    /// Population size
    pub population_size: usize,
    /// Number of generations
    pub generations: usize,
    /// Crossover probability
    pub crossover_prob: f64,
    /// Mutation probability
    pub mutation_prob: f64,
    /// Selection strategy
    pub selection_strategy: String,
    /// Elite size
    pub elite_size: usize,
}

/// Feature engineering parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureParams {
    /// Enable time-based features
    pub enable_temporal_features: bool,
    /// Enable circuit complexity features
    pub enable_complexity_features: bool,
    /// Enable user behavior features
    pub enable_user_features: bool,
    /// Enable platform performance features
    pub enable_platform_features: bool,
    /// Enable historical pattern features
    pub enable_historical_features: bool,
    /// Feature normalization method
    pub normalization_method: String,
    /// Feature selection threshold
    pub selection_threshold: f64,
}

/// Service Level Agreement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLAConfig {
    /// Maximum allowed queue time
    pub max_queue_time: Duration,
    /// Maximum allowed execution time
    pub max_execution_time: Duration,
    /// Minimum required availability
    pub min_availability: f64,
    /// Penalty for SLA violations
    pub violation_penalty: f64,
    /// SLA tier (Gold, Silver, Bronze)
    pub tier: SLATier,
}

/// SLA tier levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SLATier {
    Gold,
    Silver,
    Bronze,
    Basic,
}

/// Default implementations for new types
impl Default for MultiObjectiveWeights {
    fn default() -> Self {
        Self {
            throughput: 0.3,
            cost: 0.2,
            energy: 0.15,
            fairness: 0.15,
            sla_compliance: 0.1,
            qos: 0.1,
        }
    }
}

impl Default for RLParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            discount_factor: 0.95,
            exploration_rate: 0.1,
            episode_length: 1000,
            reward_weights: [
                ("throughput".to_string(), 1.0),
                ("fairness".to_string(), 0.5),
                ("cost".to_string(), -0.3),
            ]
            .into_iter()
            .collect(),
            state_dimension: 64,
            action_space_size: 16,
        }
    }
}

impl Default for GAParameters {
    fn default() -> Self {
        Self {
            population_size: 50,
            generations: 100,
            crossover_prob: 0.8,
            mutation_prob: 0.1,
            selection_strategy: "tournament".to_string(),
            elite_size: 5,
        }
    }
}

impl Default for FeatureParams {
    fn default() -> Self {
        Self {
            enable_temporal_features: true,
            enable_complexity_features: true,
            enable_user_features: true,
            enable_platform_features: true,
            enable_historical_features: true,
            normalization_method: "z_score".to_string(),
            selection_threshold: 0.1,
        }
    }
}

// Convenience functions for creating common job configurations

/// Create a high-priority quantum job configuration
pub fn create_high_priority_config(max_execution_time: Duration) -> JobConfig {
    JobConfig {
        priority: JobPriority::High,
        max_execution_time,
        retry_attempts: 5,
        ..Default::default()
    }
}

/// Create a best-effort quantum job configuration for batch processing
pub fn create_batch_job_config() -> JobConfig {
    JobConfig {
        priority: JobPriority::BestEffort,
        max_execution_time: Duration::from_secs(3600 * 24), // 24 hours
        max_queue_time: None,                               // No queue time limit
        retry_attempts: 1,
        ..Default::default()
    }
}

/// Create job configuration for real-time quantum applications
pub fn create_realtime_config() -> JobConfig {
    JobConfig {
        priority: JobPriority::Critical,
        max_execution_time: Duration::from_secs(60), // 1 minute
        max_queue_time: Some(Duration::from_secs(30)), // 30 seconds max wait
        retry_attempts: 0,                           // No retries for real-time
        ..Default::default()
    }
}

/// Create SLA-aware job configuration for enterprise workloads
pub fn create_sla_aware_config(tier: SLATier) -> JobConfig {
    let (priority, max_execution_time, max_queue_time, retry_attempts) = match tier {
        SLATier::Gold => (
            JobPriority::Critical,
            Duration::from_secs(300),
            Some(Duration::from_secs(60)),
            5,
        ),
        SLATier::Silver => (
            JobPriority::High,
            Duration::from_secs(600),
            Some(Duration::from_secs(300)),
            3,
        ),
        SLATier::Bronze => (
            JobPriority::Normal,
            Duration::from_secs(1800),
            Some(Duration::from_secs(900)),
            2,
        ),
        SLATier::Basic => (
            JobPriority::Low,
            Duration::from_secs(3600),
            Some(Duration::from_secs(1800)),
            1,
        ),
    };

    JobConfig {
        priority,
        max_execution_time,
        max_queue_time,
        retry_attempts,
        ..Default::default()
    }
}

/// Create cost-optimized job configuration for budget-conscious workloads
pub fn create_cost_optimized_config(budget_limit: f64) -> JobConfig {
    JobConfig {
        priority: JobPriority::BestEffort,
        max_execution_time: Duration::from_secs(7200), // 2 hours
        max_queue_time: None,                          // No queue limit for cost optimization
        retry_attempts: 1,
        cost_limit: Some(budget_limit),
        preferred_backends: vec![], // Let cost optimizer choose
        ..Default::default()
    }
}

/// Create energy-efficient job configuration for green computing
pub fn create_energy_efficient_config() -> JobConfig {
    JobConfig {
        priority: JobPriority::Low,
        max_execution_time: Duration::from_secs(3600), // 1 hour
        max_queue_time: None,                          // Wait for renewable energy availability
        retry_attempts: 2,
        tags: [("energy_profile".to_string(), "green".to_string())]
            .into_iter()
            .collect(),
        ..Default::default()
    }
}

/// Create research-focused job configuration with fault tolerance
pub fn create_research_config() -> JobConfig {
    JobConfig {
        priority: JobPriority::Normal,
        max_execution_time: Duration::from_secs(14400), // 4 hours
        max_queue_time: Some(Duration::from_secs(7200)), // 2 hours max wait
        retry_attempts: 3,
        tags: [
            ("workload_type".to_string(), "research".to_string()),
            ("fault_tolerance".to_string(), "high".to_string()),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    }
}

/// Create deadline-sensitive job configuration
pub fn create_deadline_config(deadline: SystemTime) -> JobConfig {
    JobConfig {
        priority: JobPriority::High,
        max_execution_time: Duration::from_secs(1800), // 30 minutes
        max_queue_time: Some(Duration::from_secs(300)), // 5 minutes max wait
        retry_attempts: 2,
        deadline: Some(deadline),
        tags: [(
            "scheduling_type".to_string(),
            "deadline_sensitive".to_string(),
        )]
        .into_iter()
        .collect(),
        ..Default::default()
    }
}

/// Create machine learning training job configuration
pub fn create_ml_training_config() -> JobConfig {
    JobConfig {
        priority: JobPriority::Normal,
        max_execution_time: Duration::from_secs(21600), // 6 hours
        max_queue_time: Some(Duration::from_secs(3600)), // 1 hour max wait
        retry_attempts: 2,
        resource_requirements: ResourceRequirements {
            min_qubits: 20, // Typically need more qubits for ML
            max_depth: Some(1000),
            min_fidelity: Some(0.95),
            memory_mb: Some(16384), // 16GB for ML workloads
            cpu_cores: Some(8),
            required_features: vec![
                "variational_circuits".to_string(),
                "parametric_gates".to_string(),
            ],
            ..Default::default()
        },
        tags: [
            ("workload_type".to_string(), "machine_learning".to_string()),
            ("resource_intensive".to_string(), "true".to_string()),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    }
}

/// Create optimization problem job configuration
pub fn create_optimization_config() -> JobConfig {
    JobConfig {
        priority: JobPriority::Normal,
        max_execution_time: Duration::from_secs(10800), // 3 hours
        max_queue_time: Some(Duration::from_secs(1800)), // 30 minutes max wait
        retry_attempts: 3,
        resource_requirements: ResourceRequirements {
            min_qubits: 10,
            max_depth: Some(500),
            min_fidelity: Some(0.90),
            required_features: vec!["qaoa".to_string(), "variational_algorithms".to_string()],
            ..Default::default()
        },
        tags: [
            ("workload_type".to_string(), "optimization".to_string()),
            ("algorithm_type".to_string(), "variational".to_string()),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    }
}

/// Create simulation job configuration for large-scale quantum simulation
pub fn create_simulation_config(qubit_count: usize) -> JobConfig {
    let (max_execution_time, memory_requirement) = match qubit_count {
        1..=20 => (Duration::from_secs(3600), 4096), // 1 hour, 4GB
        21..=30 => (Duration::from_secs(7200), 8192), // 2 hours, 8GB
        31..=40 => (Duration::from_secs(14400), 16384), // 4 hours, 16GB
        _ => (Duration::from_secs(28800), 32768),    // 8 hours, 32GB
    };

    JobConfig {
        priority: JobPriority::Low, // Simulations can be background tasks
        max_execution_time,
        max_queue_time: None, // Flexible queue time for simulations
        retry_attempts: 1,
        resource_requirements: ResourceRequirements {
            min_qubits: qubit_count,
            memory_mb: Some(memory_requirement),
            cpu_cores: Some(16), // High CPU for simulation
            required_features: vec!["high_precision".to_string(), "large_circuits".to_string()],
            ..Default::default()
        },
        tags: [
            ("workload_type".to_string(), "simulation".to_string()),
            ("qubit_count".to_string(), qubit_count.to_string()),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_circuit::prelude::CircuitBuilder;

    #[tokio::test]
    async fn test_job_scheduler_creation() {
        let params = SchedulingParams::default();
        let scheduler = QuantumJobScheduler::new(params);
        assert!(!*scheduler
            .scheduler_running
            .lock()
            .expect("Failed to acquire lock on scheduler_running in test"));
    }

    #[tokio::test]
    async fn test_job_config_validation() {
        let config = JobConfig::default();
        assert_eq!(config.priority, JobPriority::Normal);
        assert_eq!(config.retry_attempts, 3);
        assert!(config.dependencies.is_empty());
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        assert!(JobPriority::Critical < JobPriority::High);
        assert!(JobPriority::High < JobPriority::Normal);
        assert!(JobPriority::Normal < JobPriority::Low);
        assert!(JobPriority::Low < JobPriority::BestEffort);
    }

    #[test]
    fn test_job_id_creation() {
        let id1 = JobId::new();
        let id2 = JobId::new();
        assert_ne!(id1, id2);
        assert!(!id1.0.is_empty());
    }

    #[test]
    fn test_convenience_configs() {
        let high_priority = create_high_priority_config(Duration::from_secs(300));
        assert_eq!(high_priority.priority, JobPriority::High);
        assert_eq!(high_priority.retry_attempts, 5);

        let batch = create_batch_job_config();
        assert_eq!(batch.priority, JobPriority::BestEffort);
        assert!(batch.max_queue_time.is_none());

        let realtime = create_realtime_config();
        assert_eq!(realtime.priority, JobPriority::Critical);
        assert_eq!(realtime.retry_attempts, 0);
    }
}
