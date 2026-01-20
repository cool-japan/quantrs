//! Network optimization type definitions
//!
//! Auto-generated module split from network_optimization.rs
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use async_trait::async_trait;
use chrono::{DateTime, Datelike, Duration as ChronoDuration, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{mpsc, Semaphore};
use uuid::Uuid;

use crate::quantum_network::distributed_protocols::{
    NodeId, NodeInfo, PerformanceHistory, PerformanceMetrics, TrainingDataPoint,
};

/// Network optimization error types
#[derive(Error, Debug)]
pub enum NetworkOptimizationError {
    #[error("ML model training failed: {0}")]
    ModelTrainingFailed(String),
    #[error("Traffic shaping configuration error: {0}")]
    TrafficShapingError(String),
    #[error("QoS enforcement failed: {0}")]
    QoSEnforcementFailed(String),
    #[error("Topology optimization failed: {0}")]
    TopologyOptimizationFailed(String),
    #[error("Bandwidth allocation error: {0}")]
    BandwidthAllocationError(String),
}

pub type Result<T> = std::result::Result<T, NetworkOptimizationError>;

/// Generic ML model trait
#[async_trait]
pub trait MLModel: std::fmt::Debug {
    async fn predict(&self, features: &FeatureVector) -> Result<PredictionResult>;
    async fn train(&mut self, training_data: &[TrainingDataPoint]) -> Result<TrainingResult>;
    async fn update_weights(&mut self, feedback: &FeedbackData) -> Result<()>;
    fn get_model_metrics(&self) -> ModelMetrics;
}

/// Base trait for load balancing
#[async_trait]
pub trait LoadBalancer: std::fmt::Debug {
    async fn select_node(
        &self,
        available_nodes: &[NodeInfo],
        requirements: &ResourceRequirements,
    ) -> Result<NodeId>;
    async fn update_node_metrics(
        &self,
        node_id: &NodeId,
        metrics: &PerformanceMetrics,
    ) -> Result<()>;
    fn get_balancer_metrics(&self) -> LoadBalancerMetrics;
}

/// Traffic shaping parameters

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapingParameters {
    pub token_bucket_size: HashMap<Priority, u32>,
    pub token_generation_rate: HashMap<Priority, f64>,
    pub max_burst_duration: HashMap<Priority, Duration>,
}
/// Bandwidth allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthAllocation {
    pub guaranteed_bandwidth_mbps: f64,
    pub max_burst_bandwidth_mbps: f64,
    pub latency_budget_ms: f64,
    pub jitter_tolerance_ms: f64,
    pub packet_loss_tolerance: f64,
    pub priority_weight: f64,
}
/// Current performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentPerformanceMetrics {
    pub throughput_mbps: f64,
    pub latency_ms: f64,
    pub jitter_ms: f64,
    pub packet_loss_rate: f64,
    pub quantum_fidelity: f64,
    pub error_correction_overhead: f64,
}
/// Quantum channel optimizer
#[derive(Debug)]
pub struct QuantumChannelOptimizer {
    pub channel_configs: Vec<String>,
}
/// Congestion predictor
#[derive(Debug)]
pub struct CongestionPredictor {
    pub prediction_model: String,
}
/// Topology change recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TopologyChange {
    AddEdge {
        from: NodeId,
        to: NodeId,
        weight: f64,
    },
    RemoveEdge {
        from: NodeId,
        to: NodeId,
    },
    UpdateEdgeWeight {
        from: NodeId,
        to: NodeId,
        new_weight: f64,
    },
    AddNode {
        node_id: NodeId,
        connections: Vec<NodeId>,
    },
    RemoveNode {
        node_id: NodeId,
    },
}
/// RTT measurement data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RTTMeasurement {
    pub timestamp: DateTime<Utc>,
    pub rtt: Duration,
    pub node_pair: (NodeId, NodeId),
    pub packet_size: u32,
    pub quantum_payload: bool,
}
/// Entanglement swapping optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementSwappingOptimizations {
    pub optimal_swapping_tree: SwappingTree,
    pub fidelity_preservation_strategy: FidelityPreservationStrategy,
    pub timing_coordination: TimingCoordination,
}
/// Quantum resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResourceRequirements {
    pub qubits_needed: u32,
    pub gate_count_estimate: u32,
    pub circuit_depth: u32,
    pub fidelity_requirement: f64,
    pub coherence_time_needed: Duration,
    pub entanglement_pairs: u32,
}
/// Cooling system optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingOptimization {
    pub cooling_power_optimization: bool,
    pub temperature_gradient_minimization: bool,
    pub cooling_cycle_optimization: bool,
}
/// Adaptive rate control
#[derive(Debug, Clone)]
pub struct AdaptiveRateControl {
    pub initial_rate: f64,
    pub max_rate: f64,
    pub adjustment_factor: f64,
}
/// Queue-based latency optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueOptimizations {
    pub queue_discipline_updates: HashMap<NodeId, QueueDiscipline>,
    pub buffer_size_optimizations: HashMap<NodeId, BufferSizeConfiguration>,
    pub priority_scheduling_updates: HashMap<NodeId, PrioritySchedulingConfiguration>,
}
/// Network feature extractor for ML models
#[derive(Debug)]
pub struct NetworkFeatureExtractor {
    pub static_features: Arc<StaticFeatureExtractor>,
    pub dynamic_features: Arc<DynamicFeatureExtractor>,
    pub quantum_features: Arc<QuantumFeatureExtractor>,
    pub temporal_features: Arc<TemporalFeatureExtractor>,
}
/// Error syndrome analyzer for quantum error correction
#[derive(Debug)]
pub struct ErrorSyndromeAnalyzer {
    pub syndrome_patterns: Vec<String>,
    pub error_threshold: f64,
    pub correction_strategies: Vec<String>,
    pub analysis_depth: usize,
}
/// Latency optimization results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyOptimizationResult {
    pub routing_optimizations: RoutingOptimizations,
    pub queue_optimizations: QueueOptimizations,
    pub protocol_optimizations: ProtocolOptimizations,
    pub hardware_optimizations: HardwareOptimizations,
}
/// Routing-based latency optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingOptimizations {
    pub shortest_path_updates: HashMap<(NodeId, NodeId), Vec<NodeId>>,
    pub load_balanced_paths: HashMap<(NodeId, NodeId), Vec<Vec<NodeId>>>,
    pub quantum_aware_routes: HashMap<(NodeId, NodeId), QuantumRoute>,
}
/// Traffic rejection policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RejectionPolicy {
    HardReject,
    Defer { max_defer_time: Duration },
    Downgrade { fallback_priority: Priority },
    QuantumAware { coherence_consideration: bool },
}
/// Topology performance analyzer
#[derive(Debug, Clone)]
pub struct TopologyPerformanceAnalyzer {
    pub analysis_metrics: Vec<String>,
    pub analysis_window: Duration,
}
/// Bandwidth allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BandwidthAllocationStrategy {
    /// Fair sharing among all flows
    FairShare,
    /// Priority-based allocation
    PriorityBased {
        priority_weights: HashMap<Priority, f64>,
    },
    /// Proportional fair allocation
    ProportionalFair { fairness_parameter: f64 },
    /// Quantum-aware allocation considering coherence times
    QuantumAware {
        coherence_weight: f64,
        fidelity_weight: f64,
    },
    /// ML-optimized allocation
    MLOptimized {
        model_path: String,
        optimization_objective: String,
    },
}
/// Bandwidth optimizer with advanced algorithms
#[derive(Debug)]
pub struct BandwidthOptimizer {
    pub allocation_strategy: BandwidthAllocationStrategy,
    pub dynamic_adjustment: Arc<DynamicBandwidthAdjuster>,
    pub priority_enforcement: Arc<PriorityEnforcer>,
    pub quantum_channel_optimizer: Arc<QuantumChannelOptimizer>,
}
/// Timing optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingOptimization {
    pub gate_time_minimization: bool,
    pub idle_time_minimization: bool,
    pub synchronization_optimization: bool,
}
/// Advanced ML-based network optimizer
#[derive(Debug)]
pub struct MLNetworkOptimizer {
    pub traffic_shaper: Arc<QuantumTrafficShaper>,
    pub topology_optimizer: Arc<TopologyOptimizer>,
    pub bandwidth_optimizer: Arc<BandwidthOptimizer>,
    pub latency_optimizer: Arc<LatencyOptimizer>,
    pub ml_load_balancer: Arc<MLEnhancedLoadBalancer>,
    pub performance_predictor: Arc<NetworkPerformancePredictor>,
    pub congestion_controller: Arc<CongestionController>,
    pub qos_enforcer: Arc<QoSEnforcer>,
    pub metrics_collector: Arc<NetworkMetricsCollector>,
}
/// Admission controller for QoS enforcement
#[derive(Debug, Clone)]
pub struct AdmissionController {
    pub max_concurrent_jobs: usize,
    pub admission_criteria: Vec<String>,
}
/// Urgency evaluator for task prioritization
#[derive(Debug)]
pub struct UrgencyEvaluator {
    pub urgency_metrics: Vec<String>,
    pub weight_factors: HashMap<String, f64>,
    pub threshold_levels: Vec<f64>,
    pub evaluation_interval: Duration,
}
/// Traffic pattern analyzer
#[derive(Debug)]
pub struct TrafficPatternAnalyzer {
    pub pattern_types: Vec<String>,
    pub analysis_window: Duration,
    pub correlation_threshold: f64,
    pub seasonal_detection: bool,
}
/// Quantum-aware backoff strategy
#[derive(Debug)]
pub struct QuantumAwareBackoff {
    pub decoherence_factor: f64,
    pub coherence_time_map: Arc<RwLock<HashMap<NodeId, Duration>>>,
    pub urgency_scheduler: Arc<UrgencyScheduler>,
    pub backoff_multiplier: f64,
}
/// Swapping tree structure for entanglement distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwappingTree {
    pub nodes: Vec<SwappingNode>,
    pub edges: Vec<SwappingEdge>,
    pub root: NodeId,
    pub leaves: Vec<NodeId>,
}
/// Priority scheduling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritySchedulingConfiguration {
    pub strict_priority: bool,
    pub weighted_round_robin: Option<HashMap<Priority, f64>>,
    pub quantum_time_slices: HashMap<Priority, Duration>,
}
/// Measurement scheduling optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementSchedulingOptimizations {
    pub optimal_measurement_order: Vec<MeasurementOperation>,
    pub parallelization_strategy: ParallelizationStrategy,
    pub readout_optimization: ReadoutOptimization,
}
#[derive(Debug)]
pub struct DummyMLModel;
/// Dynamic topology optimizer
#[derive(Debug)]
pub struct TopologyOptimizer {
    pub real_time_optimization: bool,
    pub ml_based_prediction: Arc<ModelPredictor>,
    pub adaptive_routing: Arc<AdaptiveRouting>,
    pub topology_reconfiguration: Arc<TopologyReconfiguration>,
    pub performance_analyzer: Arc<TopologyPerformanceAnalyzer>,
    pub cost_optimizer: Arc<CostOptimizer>,
}
/// Topology-based features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyFeatures {
    pub clustering_coefficient: f64,
    pub average_path_length: f64,
    pub network_diameter: u32,
    pub node_degree_distribution: Vec<u32>,
    pub centrality_measures: HashMap<NodeId, CentralityMeasures>,
}
/// Round-robin balancer
#[derive(Debug)]
pub struct RoundRobinBalancer {
    pub current_index: std::sync::atomic::AtomicUsize,
}
/// Header compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderCompressionConfiguration {
    pub enabled: bool,
    pub compression_algorithm: String,
    pub compression_ratio_target: f64,
}
/// QoS policy updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSPolicyUpdates {
    pub service_class_updates: HashMap<Priority, ServiceClass>,
    pub admission_control_updates: AdmissionControlUpdates,
    pub monitoring_configuration: MonitoringConfiguration,
}
/// Quantum measurement operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementOperation {
    pub qubits: Vec<u32>,
    pub measurement_basis: String,
    pub timing_constraint: Option<Duration>,
    pub priority: u8,
}
/// Coherence metrics for quantum systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceMetrics {
    pub t1_times: HashMap<u32, Duration>,
    pub t2_times: HashMap<u32, Duration>,
    pub gate_times: HashMap<String, Duration>,
    pub readout_times: HashMap<u32, Duration>,
}
/// Routing table for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTable {
    pub routes: HashMap<NodeId, Route>,
    pub default_route: Option<NodeId>,
}
/// Enhanced ML load balancer with quantum awareness
#[derive(Debug)]
pub struct MLEnhancedLoadBalancer {
    pub base_balancer:
        Arc<dyn crate::quantum_network::distributed_protocols::LoadBalancer + Send + Sync>,
    pub ml_predictor: Arc<LoadPredictionModel>,
    pub quantum_scheduler: Arc<QuantumAwareScheduler>,
    pub performance_learner: Arc<PerformanceLearner>,
    pub adaptive_weights: Arc<Mutex<HashMap<String, f64>>>,
}
/// Quantum-aware routing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumRoute {
    pub path: Vec<NodeId>,
    pub expected_fidelity: f64,
    pub coherence_preservation: f64,
    pub entanglement_overhead: u32,
}
/// Network metrics collector
#[derive(Debug)]
pub struct NetworkMetricsCollector {
    pub collection_interval: Duration,
    pub metrics_buffer: Vec<String>,
}
/// Buffer size configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferSizeConfiguration {
    pub total_buffer_size: u32,
    pub per_priority_allocation: HashMap<Priority, u32>,
    pub overflow_handling: OverflowHandling,
}
/// Quantum protocol optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumProtocolOptimizations {
    pub entanglement_swapping_optimizations: EntanglementSwappingOptimizations,
    pub quantum_error_correction_optimizations: QECOptimizations,
    pub measurement_scheduling_optimizations: MeasurementSchedulingOptimizations,
}
/// Quantum-aware traffic shaping system
#[derive(Debug)]
pub struct QuantumTrafficShaper {
    pub bandwidth_allocation: Arc<RwLock<HashMap<Priority, BandwidthAllocation>>>,
    pub congestion_control: Arc<CongestionControl>,
    pub qos_enforcement: Arc<QoSEnforcement>,
    pub quantum_priority_scheduler: Arc<QuantumPriorityScheduler>,
    pub entanglement_aware_routing: Arc<EntanglementAwareRouting>,
    pub coherence_preserving_protocols: Arc<CoherencePreservingProtocols>,
}
/// Network topology representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopology {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<(NodeId, NodeId)>,
    pub edge_weights: HashMap<(NodeId, NodeId), f64>,
    pub clustering_coefficient: f64,
    pub diameter: u32,
}
/// Parallelization strategy for measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParallelizationStrategy {
    Sequential,
    MaximalParallel,
    ConstrainedParallel { max_simultaneous: u32 },
    QuantumAware { interference_avoidance: bool },
}
/// Target hardware specific optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetSpecificOptimizations {
    pub gate_set_optimization: bool,
    pub connectivity_aware_routing: bool,
    pub calibration_aware_compilation: bool,
}
/// Feature vector for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub features: HashMap<String, f64>,
    pub timestamp: DateTime<Utc>,
    pub context: ContextInfo,
}
/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub mae: f64,
    pub rmse: f64,
}
/// Node centrality measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityMeasures {
    pub betweenness_centrality: f64,
    pub closeness_centrality: f64,
    pub eigenvector_centrality: f64,
    pub page_rank: f64,
}
/// Circuit compilation optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitCompilationOptimizations {
    pub compilation_passes: Vec<CompilationPass>,
    pub optimization_level: OptimizationLevel,
    pub target_specific_optimizations: TargetSpecificOptimizations,
}
/// Computational resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationalRequirements {
    pub cpu_cores: u32,
    pub memory_gb: f64,
    pub storage_gb: f64,
    pub execution_time_estimate: Duration,
}
/// Congestion control parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionControlParameters {
    pub initial_window_size: f64,
    pub max_window_size: f64,
    pub backoff_factor: f64,
    pub rtt_smoothing_factor: f64,
}
/// Throughput predictor
#[derive(Debug)]
pub struct ThroughputPredictor {
    pub prediction_model: String,
}
/// Selection criteria for hybrid congestion control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    pub network_conditions: Vec<String>,
    pub quantum_metrics: Vec<String>,
    pub performance_thresholds: HashMap<String, f64>,
}
/// Quantum hardware features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumHardwareFeatures {
    pub qubit_count: u32,
    pub gate_fidelities: HashMap<String, f64>,
    pub coherence_times: HashMap<String, Duration>,
    pub connectivity_graph: Vec<(u32, u32)>,
    pub readout_fidelity: f64,
    pub error_rates: HashMap<String, f64>,
}
/// Monitoring configuration for QoS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfiguration {
    pub metrics_collection_interval: Duration,
    pub violation_detection_thresholds: HashMap<String, f64>,
    pub alert_escalation_policies: Vec<AlertPolicy>,
}
/// Failure predictor
#[derive(Debug)]
pub struct FailurePredictor {
    pub prediction_model: String,
}
/// Syndrome sharing optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyndromeSharingOptimization {
    pub sharing_protocol: String,
    pub compression_enabled: bool,
    pub aggregation_strategy: String,
}
/// Strategy change recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyChange {
    pub node_id: NodeId,
    pub old_strategy: String,
    pub new_strategy: String,
    pub expected_improvement: f64,
}
/// Protocol optimizer
#[derive(Debug)]
pub struct ProtocolOptimizer {
    pub protocol_configs: Vec<String>,
}
/// QoS monitoring system
#[derive(Debug, Clone)]
pub struct QoSMonitoringSystem {
    pub monitoring_interval: Duration,
    pub metrics_types: Vec<String>,
}
/// Recovery operation scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryOperationScheduling {
    pub scheduling_algorithm: String,
    pub priority_assignment: HashMap<String, u8>,
    pub batch_processing: bool,
}
/// Error correction code selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSelection {
    pub optimal_codes: HashMap<String, String>,
    pub adaptive_code_switching: bool,
    pub overhead_minimization: bool,
}
/// Traffic optimization results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficOptimizationResult {
    pub new_priority_weights: HashMap<Priority, f64>,
    pub queue_configurations: HashMap<NodeId, QueueConfiguration>,
    pub congestion_control_parameters: CongestionControlParameters,
}
/// QoS enforcement system
#[derive(Debug)]
pub struct QoSEnforcement {
    pub service_classes: HashMap<Priority, ServiceClass>,
    pub admission_controller: Arc<AdmissionController>,
    pub resource_allocator: Arc<QoSResourceAllocator>,
    pub monitoring_system: Arc<QoSMonitoringSystem>,
    pub violation_handler: Arc<ViolationHandler>,
}
/// Congestion controller
#[derive(Debug)]
pub struct CongestionController {
    pub congestion_threshold: f64,
    pub backoff_algorithm: String,
}
/// Frequency optimization for quantum hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyOptimization {
    pub optimal_frequencies: HashMap<u32, f64>,
    pub crosstalk_minimization: bool,
    pub frequency_collision_avoidance: bool,
}
/// Quantum-specific QoS requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumQoSRequirements {
    pub fidelity_preservation: f64,
    pub coherence_time_preservation: f64,
    pub entanglement_quality_threshold: f64,
    pub error_correction_overhead_limit: f64,
}
/// Readout optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadoutOptimization {
    pub readout_duration_optimization: bool,
    pub error_mitigation_integration: bool,
    pub classical_processing_optimization: bool,
}
/// Buffer overflow handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverflowHandling {
    DropTail,
    DropRandom,
    DropLowestPriority,
    QuantumAwareDropping { coherence_threshold: Duration },
}
/// Quantum priority scheduler
#[derive(Debug)]
pub struct QuantumPriorityScheduler {
    pub priority_queue: Vec<String>,
    pub scheduling_algorithm: String,
}
/// Current load metrics for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadMetrics {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub network_utilization: f64,
    pub queue_lengths: HashMap<Priority, u32>,
    pub active_connections: u32,
    pub quantum_circuit_count: u32,
}
/// Node in entanglement swapping tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwappingNode {
    pub node_id: NodeId,
    pub level: u32,
    pub children: Vec<NodeId>,
    pub parent: Option<NodeId>,
}
/// Deadline scheduler for task management
#[derive(Debug)]
pub struct DeadlineScheduler {
    pub deadline_window: Duration,
    pub urgency_factors: HashMap<String, f64>,
    pub preemption_enabled: bool,
    pub slack_time_threshold: Duration,
}
/// Hardware configuration optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfigurationOptimizations {
    pub frequency_optimization: FrequencyOptimization,
    pub power_optimization: PowerOptimization,
    pub thermal_optimization: ThermalOptimization,
}
/// Adaptive routing system
#[derive(Debug, Clone)]
pub struct AdaptiveRouting {
    pub routing_strategy: String,
    pub adaptation_interval: Duration,
}
/// QoS resource allocator
#[derive(Debug, Clone)]
pub struct QoSResourceAllocator {
    pub allocation_strategy: String,
    pub resource_pools: Vec<String>,
}
/// Dynamic network features (current load, performance metrics)
#[derive(Debug)]
pub struct DynamicFeatureExtractor {
    pub load_metrics: Arc<RwLock<HashMap<NodeId, LoadMetrics>>>,
    pub performance_metrics: Arc<RwLock<HashMap<NodeId, CurrentPerformanceMetrics>>>,
    pub traffic_patterns: Arc<TrafficPatternAnalyzer>,
}
/// Training scheduler for ML models
#[derive(Debug)]
pub struct TrainingScheduler {
    pub schedule_interval: Duration,
    pub max_training_duration: Duration,
    pub resource_threshold: f64,
    pub priority_level: u32,
}
/// Error correction scheduler
#[derive(Debug)]
pub struct ErrorCorrectionScheduler {
    pub correction_interval: Duration,
    pub max_correction_time: Duration,
    pub priority_levels: Vec<u32>,
    pub resource_allocation: HashMap<String, f64>,
}
/// Accuracy tracker for model performance monitoring
#[derive(Debug)]
pub struct AccuracyTracker {
    pub accuracy_history: Vec<f64>,
    pub tracking_window: Duration,
    pub threshold_accuracy: f64,
    pub performance_metrics: ModelMetrics,
}
/// Topology reconfiguration system
#[derive(Debug, Clone)]
pub struct TopologyReconfiguration {
    pub reconfiguration_strategies: Vec<String>,
    pub reconfiguration_threshold: f64,
}
/// Traffic preemption policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreemptionPolicy {
    NoPreemption,
    PreemptLowerPriority,
    QuantumContextAware { preserve_entanglement: bool },
}
/// Quantum-aware scheduling system
#[derive(Debug)]
pub struct QuantumAwareScheduler {
    pub entanglement_aware_scheduling: bool,
    pub coherence_time_optimization: bool,
    pub fidelity_preservation_priority: bool,
    pub error_correction_scheduling: Arc<ErrorCorrectionScheduler>,
    pub deadline_scheduler: Arc<DeadlineScheduler>,
    pub urgency_evaluator: Arc<UrgencyEvaluator>,
}
/// Topology optimization results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyOptimizationResult {
    pub recommended_topology_changes: Vec<TopologyChange>,
    pub routing_table_updates: HashMap<NodeId, RoutingTable>,
    pub load_balancing_updates: LoadBalancingUpdates,
}
/// Route information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    pub next_hop: NodeId,
    pub cost: f64,
    pub hop_count: u32,
    pub expected_latency: Duration,
    pub quantum_fidelity_estimate: f64,
}
/// Queue discipline algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueueDiscipline {
    FIFO,
    PriorityQueue,
    WeightedFairQueuing {
        weights: HashMap<Priority, f64>,
    },
    DeficitRoundRobin {
        quantum_sizes: HashMap<Priority, u32>,
    },
    QuantumAware {
        coherence_weights: HashMap<Priority, f64>,
    },
}
/// Load balancer performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerMetrics {
    pub total_decisions: u64,
    pub average_decision_time: Duration,
    pub prediction_accuracy: f64,
    pub load_distribution_variance: f64,
}
/// Timing coordination for quantum operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingCoordination {
    pub synchronization_protocol: String,
    pub clock_precision_requirement: Duration,
    pub coordination_overhead: Duration,
}
/// Coherence preserving protocols
#[derive(Debug, Clone)]
pub struct CoherencePreservingProtocols {
    pub protocol_types: Vec<String>,
    pub coherence_time_threshold: Duration,
}
/// Implementation step for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationStep {
    pub step_id: u32,
    pub step_name: String,
    pub step_description: String,
    pub estimated_implementation_time: Duration,
    pub expected_impact: f64,
    pub dependencies: Vec<u32>,
    pub risk_level: RiskLevel,
}
/// Training result for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub training_accuracy: f64,
    pub validation_accuracy: f64,
    pub loss_value: f64,
    pub training_duration: Duration,
    pub model_size_bytes: u64,
}
/// Static network features (topology, hardware capabilities)
#[derive(Debug)]
pub struct StaticFeatureExtractor {
    pub topology_features: TopologyFeatures,
    pub hardware_features: HashMap<NodeId, HardwareFeatures>,
    pub connectivity_matrix: Vec<Vec<f64>>,
}
/// Congestion control algorithms
#[derive(Debug)]
pub struct CongestionControl {
    pub algorithm: CongestionAlgorithm,
    pub window_size: Arc<Mutex<f64>>,
    pub rtt_estimator: Arc<RTTEstimator>,
    pub quantum_aware_backoff: Arc<QuantumAwareBackoff>,
    pub adaptive_rate_control: Arc<AdaptiveRateControl>,
}
/// Service class definition for QoS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceClass {
    pub class_name: String,
    pub guaranteed_bandwidth: f64,
    pub max_latency: Duration,
    pub max_jitter: Duration,
    pub max_packet_loss: f64,
    pub priority_level: u8,
    pub quantum_requirements: QuantumQoSRequirements,
}
/// Network performance predictor
#[derive(Debug)]
pub struct NetworkPerformancePredictor {
    pub throughput_predictor: Arc<ThroughputPredictor>,
    pub latency_predictor: Arc<LatencyPredictor>,
    pub congestion_predictor: Arc<CongestionPredictor>,
    pub failure_predictor: Arc<FailurePredictor>,
    pub quantum_performance_predictor: Arc<QuantumPerformancePredictor>,
}
/// ML model prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub predicted_values: HashMap<String, f64>,
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    pub uncertainty_estimate: f64,
    pub prediction_timestamp: DateTime<Utc>,
}
/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}
/// Risk levels for implementation steps
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}
/// Quantum volume calculator
#[derive(Debug)]
pub struct QuantumVolumeCalculator {
    pub circuit_depths: Vec<usize>,
    pub qubit_counts: Vec<usize>,
    pub fidelity_threshold: f64,
    pub trial_count: usize,
}
/// Admission control configuration updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionControlUpdates {
    pub acceptance_thresholds: HashMap<Priority, f64>,
    pub rejection_policies: HashMap<Priority, RejectionPolicy>,
    pub preemption_policies: HashMap<Priority, PreemptionPolicy>,
}
/// Quality of Service enforcer
#[derive(Debug)]
pub struct QoSEnforcer {
    pub qos_policies: Vec<String>,
    pub enforcement_mode: String,
}
/// Protocol-level latency optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolOptimizations {
    pub header_compression: HeaderCompressionConfiguration,
    pub connection_multiplexing: MultiplexingConfiguration,
    pub quantum_protocol_optimizations: QuantumProtocolOptimizations,
}
/// Hardware-level latency optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOptimizations {
    pub gate_scheduling_optimizations: GateSchedulingOptimizations,
    pub circuit_compilation_optimizations: CircuitCompilationOptimizations,
    pub hardware_configuration_optimizations: HardwareConfigurationOptimizations,
}
/// Gate parallelization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateParallelizationStrategy {
    GreedyParallel,
    OptimalParallel,
    ResourceAware { resource_constraints: Vec<String> },
    LatencyMinimizing,
}
/// Optimization predictions from ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPredictions {
    pub performance_improvement: f64,
    pub implementation_steps: Vec<ImplementationStep>,
    pub target_nodes: Vec<NodeId>,
    pub critical_weight: f64,
    pub entanglement_weight: f64,
    pub operations_weight: f64,
    pub error_correction_weight: f64,
    pub classical_weight: f64,
    pub background_weight: f64,
    pub best_effort_weight: f64,
    pub critical_queue_size_ratio: f64,
    pub entanglement_queue_size_ratio: f64,
    pub operations_queue_size_ratio: f64,
    pub error_correction_queue_size_ratio: f64,
    pub classical_queue_size_ratio: f64,
    pub background_queue_size_ratio: f64,
    pub best_effort_queue_size_ratio: f64,
    pub critical_service_rate: f64,
    pub entanglement_service_rate: f64,
    pub operations_service_rate: f64,
    pub error_correction_service_rate: f64,
    pub classical_service_rate: f64,
    pub background_service_rate: f64,
    pub best_effort_service_rate: f64,
    pub critical_coherence_threshold: f64,
    pub entanglement_red_min: f64,
    pub entanglement_red_max: f64,
    pub optimal_initial_window: f64,
    pub optimal_max_window: f64,
    pub optimal_backoff_factor: f64,
    pub optimal_rtt_smoothing: f64,
}
/// Hardware capability features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareFeatures {
    pub computational_capacity: f64,
    pub memory_capacity: f64,
    pub network_interface_speed: f64,
    pub quantum_specific_features: QuantumHardwareFeatures,
}
/// Current network state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkState {
    pub nodes: HashMap<NodeId, NodeInfo>,
    pub topology: NetworkTopology,
    pub performance_metrics: HashMap<NodeId, CurrentPerformanceMetrics>,
    pub load_metrics: HashMap<NodeId, LoadMetrics>,
    pub entanglement_quality: HashMap<(NodeId, NodeId), f64>,
    pub centrality_measures: HashMap<NodeId, CentralityMeasures>,
}
/// Cost optimizer for topology
#[derive(Debug, Clone)]
pub struct CostOptimizer {
    pub optimization_algorithm: String,
    pub cost_factors: Vec<String>,
}
/// Temporal feature extractor for time-series analysis
#[derive(Debug)]
pub struct TemporalFeatureExtractor {
    pub window_size: usize,
    pub feature_count: usize,
    pub sampling_rate: f64,
    pub feature_types: Vec<String>,
}
/// Dynamic bandwidth adjuster
#[derive(Debug, Clone)]
pub struct DynamicBandwidthAdjuster {
    pub adjustment_algorithm: String,
    pub min_bandwidth: f64,
    pub max_bandwidth: f64,
}
/// Thermal optimization for quantum hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalOptimization {
    pub cooling_optimization: CoolingOptimization,
    pub thermal_isolation_optimization: bool,
    pub temperature_stabilization: bool,
}
/// Load prediction model using ML
#[derive(Debug)]
pub struct LoadPredictionModel {
    pub model: Arc<Mutex<Box<dyn MLModel + Send + Sync>>>,
    pub feature_history: Arc<RwLock<VecDeque<FeatureVector>>>,
    pub prediction_horizon: Duration,
    pub accuracy_tracker: Arc<AccuracyTracker>,
}
/// Fidelity preservation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FidelityPreservationStrategy {
    MinimalHops,
    MaximalFidelity,
    BalancedHopsFidelity {
        hop_weight: f64,
        fidelity_weight: f64,
    },
    AdaptiveStrategy {
        context_dependent: bool,
    },
}
/// Optimization objectives for network performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    MinimizeLatency { weight: f64 },
    MaximizeThroughput { weight: f64 },
    MaximizeFidelity { weight: f64 },
    MinimizeResourceUsage { weight: f64 },
    BalanceLoad { weight: f64 },
    MinimizeJitter { weight: f64 },
    MaximizeReliability { weight: f64 },
}
/// Flow control configuration updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowControlUpdates {
    pub rate_limits: HashMap<Priority, f64>,
    pub burst_allowances: HashMap<Priority, f64>,
    pub shaping_parameters: ShapingParameters,
}
/// Connection multiplexing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiplexingConfiguration {
    pub max_concurrent_streams: u32,
    pub stream_priority_weights: HashMap<Priority, f64>,
    pub flow_control_window_size: u32,
}
/// Gate scheduling optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateSchedulingOptimizations {
    pub parallelization_strategy: GateParallelizationStrategy,
    pub resource_conflict_resolution: ResourceConflictResolution,
    pub timing_optimization: TimingOptimization,
}
/// Network resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRequirements {
    pub bandwidth_mbps: f64,
    pub latency_budget_ms: f64,
    pub packet_loss_tolerance: f64,
    pub priority_level: Priority,
}
/// Load balancing configuration updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingUpdates {
    pub weight_updates: HashMap<NodeId, f64>,
    pub capacity_updates: HashMap<NodeId, f64>,
    pub strategy_changes: Vec<StrategyChange>,
}
/// Quantum error correction optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QECOptimizations {
    pub code_selection: CodeSelection,
    pub syndrome_sharing_optimization: SyndromeSharingOptimization,
    pub recovery_operation_scheduling: RecoveryOperationScheduling,
}
/// Latency optimizer for quantum networks
#[derive(Debug)]
pub struct LatencyOptimizer {
    pub routing_optimizer: Arc<RoutingOptimizer>,
    pub queue_optimizer: Arc<QueueOptimizer>,
    pub protocol_optimizer: Arc<ProtocolOptimizer>,
    pub hardware_optimizer: Arc<HardwareLatencyOptimizer>,
}
/// Urgency scheduler
#[derive(Debug, Clone)]
pub struct UrgencyScheduler {
    pub urgency_levels: Vec<String>,
    pub scheduling_algorithm: String,
}
/// Bandwidth optimization results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthOptimizationResult {
    pub allocation_updates: HashMap<Priority, BandwidthAllocation>,
    pub flow_control_updates: FlowControlUpdates,
    pub qos_policy_updates: QoSPolicyUpdates,
}
/// Feedback data for model improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackData {
    pub actual_values: HashMap<String, f64>,
    pub prediction_quality: f64,
    pub context_feedback: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}
/// Compilation optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    Adaptive { context_aware: bool },
}
/// Power optimization for quantum hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerOptimization {
    pub idle_power_reduction: bool,
    pub dynamic_power_scaling: bool,
    pub thermal_power_management: bool,
}
/// Priority enforcer
#[derive(Debug)]
pub struct PriorityEnforcer {
    pub enforcement_rules: Vec<String>,
}
/// Performance learner
#[derive(Debug)]
pub struct PerformanceLearner {
    pub learning_rate: f64,
}
/// Packet drop policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DropPolicy {
    TailDrop,
    RandomEarlyDetection {
        min_threshold: u32,
        max_threshold: u32,
    },
    QuantumAware {
        coherence_threshold: Duration,
    },
}
/// Comprehensive optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub traffic_optimization: TrafficOptimizationResult,
    pub topology_optimization: TopologyOptimizationResult,
    pub bandwidth_optimization: BandwidthOptimizationResult,
    pub latency_optimization: LatencyOptimizationResult,
    pub overall_improvement_estimate: f64,
    pub implementation_steps: Vec<ImplementationStep>,
}
/// Latency predictor
#[derive(Debug)]
pub struct LatencyPredictor {
    pub prediction_model: String,
}
/// Hardware latency optimizer
#[derive(Debug)]
pub struct HardwareLatencyOptimizer {
    pub latency_configs: Vec<String>,
}
/// Routing optimizer
#[derive(Debug)]
pub struct RoutingOptimizer {
    pub routing_table: HashMap<String, String>,
}
/// Congestion control algorithm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CongestionAlgorithm {
    /// Traditional TCP-like congestion control
    TCP,
    /// Quantum-aware congestion control considering decoherence
    QuantumAware {
        decoherence_sensitivity: f64,
        coherence_time_factor: f64,
    },
    /// ML-based adaptive congestion control
    MLAdaptive {
        model_path: String,
        learning_rate: f64,
        prediction_horizon_ms: u64,
    },
    /// Hybrid approach combining multiple algorithms
    Hybrid {
        algorithms: Vec<String>,
        selection_criteria: SelectionCriteria,
    },
}
/// Queue configuration for traffic shaping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfiguration {
    pub queue_sizes: HashMap<Priority, u32>,
    pub service_rates: HashMap<Priority, f64>,
    pub drop_policies: HashMap<Priority, DropPolicy>,
}
/// Types of ML models for network optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MLModelType {
    /// Neural network for complex pattern recognition
    NeuralNetwork {
        layers: Vec<u32>,
        activation_function: String,
        learning_rate: f64,
    },
    /// Random forest for robust predictions
    RandomForest {
        n_estimators: u32,
        max_depth: Option<u32>,
        feature_sampling: f64,
    },
    /// Gradient boosting for high accuracy
    GradientBoosting {
        n_estimators: u32,
        learning_rate: f64,
        max_depth: u32,
    },
    /// Quantum ML model for quantum-specific optimizations
    QuantumML {
        ansatz_type: String,
        n_qubits: u32,
        optimization_method: String,
    },
}
/// Violation handler for QoS enforcement
#[derive(Debug, Clone)]
pub struct ViolationHandler {
    pub response_strategies: Vec<String>,
    pub escalation_threshold: u8,
}
/// Traffic priority levels for quantum communications
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Priority {
    /// Critical quantum state transfers requiring immediate transmission
    CriticalQuantumState,
    /// Real-time entanglement distribution
    EntanglementDistribution,
    /// Time-sensitive quantum gates and operations
    QuantumOperations,
    /// Quantum error correction communications
    ErrorCorrection,
    /// Classical control signals for quantum operations
    ClassicalControl,
    /// Background data synchronization
    BackgroundSync,
    /// Best-effort traffic
    BestEffort,
}
/// Model updater for continuous learning
#[derive(Debug)]
pub struct ModelUpdater {
    pub update_frequency: Duration,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub last_update: DateTime<Utc>,
}
/// Resource conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceConflictResolution {
    FirstComeFirstServed,
    PriorityBased { priority_function: String },
    OptimalReordering,
    AdaptiveReordering { learning_enabled: bool },
}
/// Context information for feature vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextInfo {
    pub network_state: String,
    pub time_of_day: u8,
    pub day_of_week: u8,
    pub quantum_experiment_type: Option<String>,
    pub user_priority: Option<String>,
}
/// Circuit compilation pass
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationPass {
    pub pass_name: String,
    pub enabled: bool,
    pub parameters: HashMap<String, f64>,
}
/// Entanglement-aware routing system
#[derive(Debug, Clone)]
pub struct EntanglementAwareRouting {
    pub routing_algorithm: String,
    pub entanglement_threshold: f64,
}
/// Resource requirements for load balancing decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub computational_requirements: ComputationalRequirements,
    pub quantum_requirements: QuantumResourceRequirements,
    pub network_requirements: NetworkRequirements,
    pub deadline_requirements: Option<DateTime<Utc>>,
}
/// Queue optimizer
#[derive(Debug)]
pub struct QueueOptimizer {
    pub queue_configs: Vec<String>,
}
/// Edge in entanglement swapping tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwappingEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub entanglement_quality: f64,
    pub swapping_time: Duration,
}
/// Quantum-specific features for ML
#[derive(Debug)]
pub struct QuantumFeatureExtractor {
    pub entanglement_quality: Arc<RwLock<HashMap<(NodeId, NodeId), f64>>>,
    pub coherence_metrics: Arc<RwLock<HashMap<NodeId, CoherenceMetrics>>>,
    pub error_syndrome_patterns: Arc<ErrorSyndromeAnalyzer>,
    pub quantum_volume_metrics: Arc<QuantumVolumeCalculator>,
}
/// Round-trip time estimator
#[derive(Debug)]
pub struct RTTEstimator {
    pub smoothed_rtt: Arc<Mutex<Duration>>,
    pub rtt_variance: Arc<Mutex<Duration>>,
    pub alpha: f64,
    pub beta: f64,
    pub measurements: Arc<Mutex<VecDeque<RTTMeasurement>>>,
}
/// Quantum performance predictor
#[derive(Debug)]
pub struct QuantumPerformancePredictor {
    pub prediction_model: String,
}
/// ML model predictor for network optimization
#[derive(Debug)]
pub struct ModelPredictor {
    pub model_type: MLModelType,
    pub feature_extractor: Arc<NetworkFeatureExtractor>,
    pub prediction_cache: Arc<Mutex<HashMap<String, PredictionResult>>>,
    pub model_updater: Arc<ModelUpdater>,
    pub training_scheduler: Arc<TrainingScheduler>,
}
/// Alert policy for QoS violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertPolicy {
    pub condition: String,
    pub severity: AlertSeverity,
    pub notification_channels: Vec<String>,
    pub escalation_delay: Duration,
}
