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
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};

use crate::{translation::HardwareBackend, CircuitResult};
use quantrs2_circuit::prelude::Circuit;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use uuid::Uuid;

/// Job priority levels
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub enum JobPriority {
    /// System critical jobs (maintenance, calibration)
    Critical = 0,
    /// High priority research or production jobs
    High = 1,
    /// Normal priority jobs
    #[default]
    Normal = 2,
    /// Low priority background jobs
    Low = 3,
    /// Best effort jobs that can be delayed
    BestEffort = 4,
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

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl JobId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub const fn from_string(s: String) -> Self {
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
    pub(super) params: Arc<RwLock<SchedulingParams>>,
    /// Job queues by priority level
    pub(super) job_queues: Arc<Mutex<BTreeMap<JobPriority, VecDeque<JobId>>>>,
    /// All active jobs
    pub(super) jobs: Arc<RwLock<HashMap<JobId, Box<dyn std::any::Any + Send + Sync>>>>,
    /// Backend performance tracking
    pub(super) backend_performance: Arc<RwLock<HashMap<HardwareBackend, BackendPerformance>>>,
    /// Available backends
    pub(super) backends: Arc<RwLock<HashSet<HardwareBackend>>>,
    /// Running jobs
    pub(super) running_jobs: Arc<RwLock<HashMap<JobId, (HardwareBackend, SystemTime)>>>,
    /// Job execution history
    pub(super) execution_history: Arc<RwLock<Vec<JobExecution>>>,
    /// User fair share tracking
    pub(super) user_shares: Arc<RwLock<HashMap<String, UserShare>>>,
    /// Scheduler control
    pub(super) scheduler_running: Arc<Mutex<bool>>,
    /// Event notifications
    pub(super) event_sender: mpsc::UnboundedSender<SchedulerEvent>,
    /// Performance predictor
    pub(super) performance_predictor: Arc<Mutex<PerformancePredictor>>,
    /// Resource manager
    pub(super) resource_manager: Arc<Mutex<ResourceManager>>,
    /// Lightweight job-status side-channel (type-erased jobs cannot carry mutable status)
    pub(super) job_status_map: Arc<RwLock<HashMap<JobId, JobStatus>>>,
    /// Job config side-channel for resource-aware backend selection (avoids downcasting Any)
    pub(super) job_config_map: Arc<RwLock<HashMap<JobId, JobConfig>>>,
    /// Execution metrics side-channel for recording start/end times without downcasting
    pub(super) job_metrics_map: Arc<RwLock<HashMap<JobId, ExecutionMetrics>>>,
}

/// User fair share tracking
#[derive(Debug, Clone)]
pub(super) struct UserShare {
    pub(super) user_id: String,
    pub(super) allocated_share: f64,
    pub(super) used_share: f64,
    pub(super) jobs_running: usize,
    pub(super) jobs_queued: usize,
    pub(super) last_updated: SystemTime,
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
pub(super) struct PerformancePredictor {
    /// Historical performance data
    pub(super) history: VecDeque<PredictionDataPoint>,
    /// Learned model parameters
    pub(super) model_params: HashMap<String, f64>,
    /// Prediction accuracy metrics
    pub(super) accuracy_metrics: HashMap<String, f64>,
}

/// Data point for performance prediction
#[derive(Debug, Clone)]
pub(super) struct PredictionDataPoint {
    pub(super) timestamp: SystemTime,
    pub(super) backend: HardwareBackend,
    pub(super) queue_length: usize,
    pub(super) job_complexity: f64,
    pub(super) execution_time: Duration,
    pub(super) success: bool,
}

/// Resource allocation and management
pub(super) struct ResourceManager {
    /// Available resources by backend
    pub(super) available_resources: HashMap<HardwareBackend, ResourceCapacity>,
    /// Resource reservations
    pub(super) reservations: HashMap<JobId, ResourceReservation>,
    /// Resource utilization history
    pub(super) utilization_history: VecDeque<ResourceSnapshot>,
}

/// Resource capacity for a backend
#[derive(Debug, Clone)]
pub(super) struct ResourceCapacity {
    pub(super) qubits: usize,
    pub(super) max_circuit_depth: Option<usize>,
    pub(super) memory_mb: u64,
    pub(super) cpu_cores: u32,
    pub(super) concurrent_jobs: usize,
    pub(super) features: HashSet<String>,
}

/// Resource reservation for a job
#[derive(Debug, Clone)]
pub(super) struct ResourceReservation {
    pub(super) job_id: JobId,
    pub(super) backend: HardwareBackend,
    pub(super) resources: ResourceRequirements,
    pub(super) reserved_at: SystemTime,
    pub(super) expires_at: SystemTime,
}

/// Resource utilization snapshot
#[derive(Debug, Clone)]
pub(super) struct ResourceSnapshot {
    pub(super) timestamp: SystemTime,
    pub(super) backend: HardwareBackend,
    pub(super) utilization: f64,
    pub(super) active_jobs: usize,
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
