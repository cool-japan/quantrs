//! Supporting types for hardware parallelization

use super::config::*;
use super::monitor::{
    ExecutionQualityMetrics, OptimizationSuggestion, PerformanceMetrics, ResourceUsage,
};
use crate::translation::HardwareBackend;
use quantrs2_core::qubit::QubitId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Parallel circuit execution task
#[derive(Debug)]
pub struct ParallelCircuitTask {
    pub id: String,
    pub circuit: Box<dyn std::any::Any + Send + Sync>,
    pub target_backend: HardwareBackend,
    pub priority: TaskPriority,
    pub resource_requirements: ParallelResourceRequirements,
    pub constraints: ExecutionConstraints,
    pub submitted_at: SystemTime,
    pub deadline: Option<SystemTime>,
}

/// Parallel gate execution task
#[derive(Debug, Clone)]
pub struct ParallelGateTask {
    pub id: String,
    pub gate_operations: Vec<ParallelGateOperation>,
    pub target_qubits: Vec<QubitId>,
    pub dependency_graph: HashMap<String, Vec<String>>,
    pub priority: TaskPriority,
    pub submitted_at: SystemTime,
}

/// Task priorities
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    /// Low priority (best effort)
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority (real-time)
    Critical,
    /// System priority (internal operations)
    System,
}

/// Parallel resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelResourceRequirements {
    /// Required CPU cores
    pub required_cpu_cores: usize,
    /// Required memory (MB)
    pub required_memory_mb: f64,
    /// Required QPU time
    pub required_qpu_time: Duration,
    /// Required network bandwidth (Mbps)
    pub required_bandwidth_mbps: f64,
    /// Required storage (MB)
    pub required_storage_mb: f64,
}

/// Execution constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConstraints {
    /// Allowed backends
    pub allowed_backends: Vec<HardwareBackend>,
    /// Forbidden backends
    pub forbidden_backends: Vec<HardwareBackend>,
    /// Quality requirements
    pub quality_requirements: QualityRequirements,
    /// Timing constraints
    pub timing_constraints: TimingConstraints,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}

/// Quality requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirements {
    /// Minimum fidelity
    pub min_fidelity: Option<f64>,
    /// Maximum error rate
    pub max_error_rate: Option<f64>,
    /// Required calibration recency
    pub calibration_recency: Option<Duration>,
    /// Quality assessment method
    pub assessment_method: QualityAssessmentMethod,
}

/// Quality assessment methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityAssessmentMethod {
    /// Static quality metrics
    Static,
    /// Dynamic quality monitoring
    Dynamic,
    /// Predictive quality modeling
    Predictive,
    /// Benchmarking-based assessment
    BenchmarkBased,
}

/// Timing constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConstraints {
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
    /// Maximum queue wait time
    pub max_queue_time: Option<Duration>,
    /// Preferred execution window
    pub preferred_window: Option<(SystemTime, SystemTime)>,
    /// Scheduling flexibility
    pub scheduling_flexibility: SchedulingFlexibility,
}

/// Scheduling flexibility levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulingFlexibility {
    /// Rigid scheduling (exact timing required)
    Rigid,
    /// Flexible scheduling (best effort)
    Flexible,
    /// Adaptive scheduling (can adjust based on conditions)
    Adaptive,
}

/// Resource constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum cost
    pub max_cost: Option<f64>,
    /// Maximum energy consumption
    pub max_energy: Option<f64>,
    /// Resource usage limits
    pub usage_limits: HashMap<String, f64>,
    /// Sharing preferences
    pub sharing_preferences: SharingPreferences,
}

/// Sharing preferences
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SharingPreferences {
    /// Exclusive resource access
    Exclusive,
    /// Shared resource access
    Shared,
    /// Best effort sharing
    BestEffort,
    /// Conditional sharing
    Conditional(Vec<SharingCondition>),
}

/// Sharing conditions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SharingCondition {
    /// Share only with specific users
    UserWhitelist(Vec<String>),
    /// Share only with specific circuit types
    CircuitTypeWhitelist(Vec<String>),
    /// Share only during specific time windows
    TimeWindow(SystemTime, SystemTime),
    /// Share only below resource threshold
    ResourceThreshold(String, f64),
}

/// Parallel gate operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelGateOperation {
    /// Operation ID
    pub id: String,
    /// Gate type
    pub gate_type: String,
    /// Target qubits
    pub qubits: Vec<QubitId>,
    /// Gate parameters
    pub parameters: Vec<f64>,
    /// Dependencies on other operations
    pub dependencies: Vec<String>,
    /// Parallelization hints
    pub parallelization_hints: ParallelizationHints,
}

/// Parallelization hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelizationHints {
    /// Can be executed in parallel with others
    pub parallel_safe: bool,
    /// Preferred execution order
    pub execution_order: Option<usize>,
    /// Resource affinity
    pub resource_affinity: ResourceAffinity,
    /// Scheduling hints
    pub scheduling_hints: SchedulingHints,
}

/// Resource affinity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceAffinity {
    /// No preference
    None,
    /// Prefer specific backend
    Backend(HardwareBackend),
    /// Prefer specific qubits
    Qubits(Vec<QubitId>),
    /// Prefer co-location with other operations
    CoLocation(Vec<String>),
}

/// Scheduling hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingHints {
    /// Preferred execution time
    pub preferred_time: Option<SystemTime>,
    /// Execution priority
    pub priority: TaskPriority,
    /// Deadline
    pub deadline: Option<SystemTime>,
    /// Batch compatibility
    pub batch_compatible: bool,
}

/// Resource monitor for tracking system resources

/// Load metrics for a backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadMetrics {
    pub cpu_load: f64,
    pub memory_load: f64,
    pub qpu_load: f64,
    pub network_load: f64,
    pub queue_length: usize,
    pub response_time: Duration,
    pub throughput: f64,
    pub error_rate: f64,
    pub last_updated: SystemTime,
}

/// Load snapshot for historical tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadSnapshot {
    pub timestamp: SystemTime,
    pub backend_loads: HashMap<HardwareBackend, LoadMetrics>,
    pub system_metrics: SystemMetrics,
    pub predictions: LoadPredictions,
}

/// System-wide metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub total_throughput: f64,
    pub average_latency: Duration,
    pub total_resource_utilization: f64,
    pub overall_quality_score: f64,
    pub cost_per_operation: f64,
    pub energy_per_operation: f64,
}

/// Load predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadPredictions {
    pub predicted_loads: HashMap<HardwareBackend, f64>,
    pub confidence_levels: HashMap<HardwareBackend, f64>,
    pub prediction_horizon: Duration,
    pub model_accuracy: f64,
}

/// Rebalancing strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RebalancingStrategy {
    /// Reactive rebalancing
    Reactive,
    /// Proactive rebalancing
    Proactive,
    /// Predictive rebalancing
    Predictive,
    /// Hybrid approach
    Hybrid,
}

/// Migration tracker
pub struct MigrationTracker {
    pub active_migrations: HashMap<String, MigrationStatus>,
    pub migration_history: VecDeque<MigrationRecord>,
    pub migration_costs: HashMap<(HardwareBackend, HardwareBackend), f64>,
}

/// Migration status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStatus {
    pub task_id: String,
    pub source_backend: HardwareBackend,
    pub target_backend: HardwareBackend,
    pub progress: f64, // 0.0 to 1.0
    pub started_at: SystemTime,
    pub estimated_completion: SystemTime,
    pub migration_type: MigrationType,
}

/// Migration types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationType {
    /// Circuit migration
    Circuit,
    /// Data migration
    Data,
    /// State migration
    State,
    /// Full migration
    Full,
}

/// Migration record for tracking history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationRecord {
    pub task_id: String,
    pub source_backend: HardwareBackend,
    pub target_backend: HardwareBackend,
    pub migration_time: Duration,
    pub success: bool,
    pub cost: f64,
    pub quality_impact: f64,
    pub timestamp: SystemTime,
}

/// Parallel execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelExecutionResult {
    pub task_id: String,
    pub success: bool,
    pub execution_time: Duration,
    pub resource_usage: ResourceUsage,
    pub quality_metrics: ExecutionQualityMetrics,
    pub results: Option<Vec<u8>>, // Serialized results
    pub error_message: Option<String>,
}

/// Load balancing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingResult {
    pub rebalancing_performed: bool,
    pub migrations_performed: usize,
    pub load_improvement: f64,
    pub estimated_performance_gain: f64,
    pub rebalancing_cost: f64,
}
