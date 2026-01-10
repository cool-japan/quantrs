//! Provider capabilities for the discovery system.
//!
//! This module contains comprehensive provider capability structures
//! covering hardware, software, performance, cost, security, and support.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime};

use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::config::ComplianceStandard;
use super::types::{
    AuthenticationMethod, AuthorizationModel, ConnectivityGraph, ImpactLevel, MaintenanceFrequency,
    MeasurementType, QuantumFramework, SupportChannel,
};

/// Comprehensive provider capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Basic capabilities
    pub basic: BasicCapabilities,
    /// Hardware capabilities
    pub hardware: HardwareCapabilities,
    /// Software capabilities
    pub software: SoftwareCapabilities,
    /// Performance characteristics
    pub performance: PerformanceCapabilities,
    /// Cost characteristics
    pub cost: CostCapabilities,
    /// Security capabilities
    pub security: SecurityCapabilities,
    /// Support capabilities
    pub support: SupportCapabilities,
    /// Advanced features
    pub advanced_features: AdvancedFeatures,
}

/// Basic provider capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicCapabilities {
    /// Number of qubits
    pub qubit_count: usize,
    /// Supported gate set
    pub gate_set: HashSet<String>,
    /// Connectivity graph
    pub connectivity: ConnectivityGraph,
    /// Supported measurements
    pub measurement_types: Vec<MeasurementType>,
    /// Classical register size
    pub classical_register_size: usize,
    /// Maximum circuit depth
    pub max_circuit_depth: Option<usize>,
    /// Maximum shots per execution
    pub max_shots: Option<u64>,
}

/// Hardware capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    /// Quantum volume
    pub quantum_volume: Option<u32>,
    /// Error rates
    pub error_rates: ErrorRates,
    /// Coherence times
    pub coherence_times: CoherenceTimes,
    /// Gate times
    pub gate_times: HashMap<String, Duration>,
    /// Crosstalk characteristics
    pub crosstalk: CrosstalkCharacteristics,
    /// Calibration information
    pub calibration: CalibrationInfo,
    /// Temperature information
    pub temperature: Option<f64>,
    /// Noise characteristics
    pub noise_characteristics: NoiseCharacteristics,
}

/// Error rate information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRates {
    /// Single-qubit gate error rates
    pub single_qubit_gates: HashMap<String, f64>,
    /// Two-qubit gate error rates
    pub two_qubit_gates: HashMap<String, f64>,
    /// Readout error rates
    pub readout_errors: HashMap<usize, f64>,
    /// Average error rate
    pub average_error_rate: f64,
    /// Error rate variance
    pub error_rate_variance: f64,
}

/// Coherence time information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceTimes {
    /// T1 relaxation times per qubit
    pub t1_times: HashMap<usize, Duration>,
    /// T2 dephasing times per qubit
    pub t2_times: HashMap<usize, Duration>,
    /// Average T1 time
    pub average_t1: Duration,
    /// Average T2 time
    pub average_t2: Duration,
}

/// Crosstalk characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrosstalkCharacteristics {
    /// Crosstalk matrix
    pub crosstalk_matrix: Array2<f64>,
    /// Spectral crosstalk
    pub spectral_crosstalk: HashMap<String, f64>,
    /// Temporal crosstalk
    pub temporal_crosstalk: HashMap<String, f64>,
    /// Mitigation strategies available
    pub mitigation_strategies: Vec<String>,
}

/// Calibration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationInfo {
    /// Last calibration time
    pub last_calibration: SystemTime,
    /// Calibration frequency
    pub calibration_frequency: Duration,
    /// Calibration quality score
    pub quality_score: f64,
    /// Drift rate
    pub drift_rate: f64,
    /// Calibration method
    pub calibration_method: String,
}

/// Noise characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseCharacteristics {
    /// Noise model type
    pub noise_model_type: String,
    /// Noise parameters
    pub noise_parameters: HashMap<String, f64>,
    /// Noise correlations
    pub noise_correlations: Array2<f64>,
    /// Environmental noise factors
    pub environmental_factors: HashMap<String, f64>,
}

/// Software capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftwareCapabilities {
    /// Supported frameworks
    pub supported_frameworks: Vec<QuantumFramework>,
    /// Programming languages
    pub programming_languages: Vec<String>,
    /// Compilation features
    pub compilation_features: CompilationFeatures,
    /// Optimization features
    pub optimization_features: OptimizationFeatures,
    /// Simulation capabilities
    pub simulation_capabilities: SimulationCapabilities,
    /// Integration capabilities
    pub integration_capabilities: IntegrationCapabilities,
}

/// Compilation features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationFeatures {
    /// Circuit optimization
    pub circuit_optimization: bool,
    /// Gate synthesis
    pub gate_synthesis: bool,
    /// Routing algorithms
    pub routing_algorithms: Vec<String>,
    /// Transpilation passes
    pub transpilation_passes: Vec<String>,
    /// Custom compilation
    pub custom_compilation: bool,
}

/// Optimization features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationFeatures {
    /// Parameter optimization
    pub parameter_optimization: bool,
    /// Circuit depth optimization
    pub depth_optimization: bool,
    /// Gate count optimization
    pub gate_count_optimization: bool,
    /// Noise-aware optimization
    pub noise_aware_optimization: bool,
    /// Variational algorithms
    pub variational_algorithms: Vec<String>,
}

/// Simulation capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationCapabilities {
    /// Classical simulation
    pub classical_simulation: bool,
    /// Noise simulation
    pub noise_simulation: bool,
    /// Error simulation
    pub error_simulation: bool,
    /// Maximum simulated qubits
    pub max_simulated_qubits: Option<usize>,
    /// Simulation backends
    pub simulation_backends: Vec<String>,
}

/// Integration capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationCapabilities {
    /// REST API
    pub rest_api: bool,
    /// GraphQL API
    pub graphql_api: bool,
    /// WebSocket support
    pub websocket_support: bool,
    /// SDK availability
    pub sdk_languages: Vec<String>,
    /// Third-party integrations
    pub third_party_integrations: Vec<String>,
}

/// Performance capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCapabilities {
    /// Throughput metrics
    pub throughput: ThroughputMetrics,
    /// Latency metrics
    pub latency: LatencyMetrics,
    /// Availability metrics
    pub availability: AvailabilityMetrics,
    /// Scalability characteristics
    pub scalability: ScalabilityCharacteristics,
    /// Resource utilization
    pub resource_utilization: ResourceUtilizationMetrics,
}

/// Throughput metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Circuits per hour
    pub circuits_per_hour: f64,
    /// Shots per second
    pub shots_per_second: f64,
    /// Jobs per day
    pub jobs_per_day: f64,
    /// Peak throughput
    pub peak_throughput: f64,
    /// Sustained throughput
    pub sustained_throughput: f64,
}

/// Latency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Job submission latency
    pub submission_latency: Duration,
    /// Queue wait time
    pub queue_wait_time: Duration,
    /// Execution time
    pub execution_time: Duration,
    /// Result retrieval time
    pub result_retrieval_time: Duration,
    /// Total turnaround time
    pub total_turnaround_time: Duration,
}

/// Availability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailabilityMetrics {
    /// Uptime percentage
    pub uptime_percentage: f64,
    /// Mean time between failures
    pub mtbf: Duration,
    /// Mean time to recovery
    pub mttr: Duration,
    /// Maintenance windows
    pub maintenance_windows: Vec<MaintenanceWindow>,
    /// Service level agreement
    pub sla: Option<ServiceLevelAgreement>,
}

/// Maintenance window information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceWindow {
    /// Start time
    pub start_time: SystemTime,
    /// Duration
    pub duration: Duration,
    /// Frequency
    pub frequency: MaintenanceFrequency,
    /// Impact level
    pub impact_level: ImpactLevel,
    /// Description
    pub description: String,
}

/// Service level agreement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceLevelAgreement {
    /// Guaranteed uptime
    pub guaranteed_uptime: f64,
    /// Maximum response time
    pub max_response_time: Duration,
    /// Support response time
    pub support_response_time: Duration,
    /// Resolution time
    pub resolution_time: Duration,
    /// Penalty clauses
    pub penalty_clauses: Vec<String>,
}

/// Scalability characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityCharacteristics {
    /// Horizontal scalability
    pub horizontal_scalability: bool,
    /// Vertical scalability
    pub vertical_scalability: bool,
    /// Auto-scaling support
    pub auto_scaling: bool,
    /// Maximum concurrent jobs
    pub max_concurrent_jobs: Option<u32>,
    /// Load balancing
    pub load_balancing: bool,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationMetrics {
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// Network utilization
    pub network_utilization: f64,
    /// Storage utilization
    pub storage_utilization: f64,
    /// Quantum resource utilization
    pub quantum_utilization: f64,
}

/// Cost capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostCapabilities {
    /// Cost model
    pub cost_model: CostModel,
    /// Cost optimization features
    pub cost_optimization: CostOptimizationFeatures,
    /// Budget management
    pub budget_management: BudgetManagementFeatures,
    /// Cost transparency
    pub cost_transparency: CostTransparencyFeatures,
}

/// Cost model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    /// Pricing structure
    pub pricing_structure: PricingStructure,
    /// Cost factors
    pub cost_factors: Vec<CostFactor>,
    /// Volume discounts
    pub volume_discounts: Vec<VolumeDiscount>,
    /// Regional pricing
    pub regional_pricing: HashMap<String, f64>,
    /// Currency support
    pub supported_currencies: Vec<String>,
}

/// Pricing structure
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PricingStructure {
    Fixed,
    Variable,
    Tiered,
    Usage,
    Hybrid,
    Negotiated,
}

/// Cost factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostFactor {
    /// Factor name
    pub name: String,
    /// Factor type
    pub factor_type: CostFactorType,
    /// Unit cost
    pub unit_cost: f64,
    /// Minimum charge
    pub minimum_charge: Option<f64>,
    /// Maximum charge
    pub maximum_charge: Option<f64>,
}

/// Cost factor types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CostFactorType {
    PerShot,
    PerCircuit,
    PerMinute,
    PerHour,
    PerQubit,
    PerGate,
    PerJob,
    Fixed,
}

/// Volume discount information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeDiscount {
    /// Minimum volume
    pub min_volume: u64,
    /// Discount percentage
    pub discount_percentage: f64,
    /// Discount type
    pub discount_type: DiscountType,
}

/// Discount types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscountType {
    Percentage,
    Fixed,
    Tiered,
    Progressive,
}

/// Cost optimization features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizationFeatures {
    /// Cost estimation
    pub cost_estimation: bool,
    /// Cost tracking
    pub cost_tracking: bool,
    /// Budget alerts
    pub budget_alerts: bool,
    /// Cost optimization recommendations
    pub optimization_recommendations: bool,
    /// Spot pricing
    pub spot_pricing: bool,
}

/// Budget management features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetManagementFeatures {
    /// Budget setting
    pub budget_setting: bool,
    /// Budget monitoring
    pub budget_monitoring: bool,
    /// Spending limits
    pub spending_limits: bool,
    /// Cost allocation
    pub cost_allocation: bool,
    /// Invoice management
    pub invoice_management: bool,
}

/// Cost transparency features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostTransparencyFeatures {
    /// Real-time cost display
    pub realtime_cost_display: bool,
    /// Detailed cost breakdown
    pub detailed_breakdown: bool,
    /// Historical cost analysis
    pub historical_analysis: bool,
    /// Cost comparison tools
    pub comparison_tools: bool,
    /// Cost reporting
    pub cost_reporting: bool,
}

/// Security capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityCapabilities {
    /// Authentication methods
    pub authentication: Vec<AuthenticationMethod>,
    /// Authorization models
    pub authorization: Vec<AuthorizationModel>,
    /// Encryption capabilities
    pub encryption: EncryptionCapabilities,
    /// Compliance certifications
    pub compliance: Vec<ComplianceStandard>,
    /// Security monitoring
    pub security_monitoring: SecurityMonitoringCapabilities,
}

/// Encryption capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionCapabilities {
    /// Data at rest encryption
    pub data_at_rest: bool,
    /// Data in transit encryption
    pub data_in_transit: bool,
    /// End-to-end encryption
    pub end_to_end: bool,
    /// Encryption algorithms
    pub algorithms: Vec<String>,
    /// Key management
    pub key_management: KeyManagementCapabilities,
}

/// Key management capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementCapabilities {
    /// Customer-managed keys
    pub customer_managed_keys: bool,
    /// Hardware security modules
    pub hsm_support: bool,
    /// Key rotation
    pub key_rotation: bool,
    /// Key escrow
    pub key_escrow: bool,
    /// Multi-party computation
    pub mpc_support: bool,
}

/// Security monitoring capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMonitoringCapabilities {
    /// Audit logging
    pub audit_logging: bool,
    /// Intrusion detection
    pub intrusion_detection: bool,
    /// Anomaly detection
    pub anomaly_detection: bool,
    /// Security alerts
    pub security_alerts: bool,
    /// Threat intelligence
    pub threat_intelligence: bool,
}

/// Support capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportCapabilities {
    /// Support channels
    pub support_channels: Vec<SupportChannel>,
    /// Support hours
    pub support_hours: SupportHours,
    /// Response times
    pub response_times: ResponseTimeGuarantees,
    /// Documentation quality
    pub documentation_quality: DocumentationQuality,
    /// Training and education
    pub training_education: TrainingEducationCapabilities,
}

/// Support hours
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportHours {
    /// Business hours support
    pub business_hours: bool,
    /// 24/7 support
    pub twenty_four_seven: bool,
    /// Weekend support
    pub weekend_support: bool,
    /// Holiday support
    pub holiday_support: bool,
    /// Timezone coverage
    pub timezone_coverage: Vec<String>,
}

/// Response time guarantees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeGuarantees {
    /// Critical issues
    pub critical_response_time: Duration,
    /// High priority issues
    pub high_priority_response_time: Duration,
    /// Medium priority issues
    pub medium_priority_response_time: Duration,
    /// Low priority issues
    pub low_priority_response_time: Duration,
    /// First response time
    pub first_response_time: Duration,
}

/// Documentation quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationQuality {
    /// Completeness score
    pub completeness_score: f64,
    /// Accuracy score
    pub accuracy_score: f64,
    /// Clarity score
    pub clarity_score: f64,
    /// Up-to-date score
    pub up_to_date_score: f64,
    /// Example quality
    pub example_quality: f64,
}

/// Training and education capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingEducationCapabilities {
    /// Online courses
    pub online_courses: bool,
    /// Workshops
    pub workshops: bool,
    /// Certification programs
    pub certification_programs: bool,
    /// Consulting services
    pub consulting_services: bool,
    /// Community forums
    pub community_forums: bool,
}

/// Advanced features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedFeatures {
    /// Machine learning integration
    pub ml_integration: MLIntegrationFeatures,
    /// Hybrid computing
    pub hybrid_computing: HybridComputingFeatures,
    /// Quantum networking
    pub quantum_networking: QuantumNetworkingFeatures,
    /// Research capabilities
    pub research_capabilities: ResearchCapabilities,
    /// Experimental features
    pub experimental_features: Vec<String>,
}

/// ML integration features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLIntegrationFeatures {
    /// Quantum machine learning
    pub quantum_ml: bool,
    /// Classical ML integration
    pub classical_ml_integration: bool,
    /// AutoML support
    pub automl_support: bool,
    /// Supported ML frameworks
    pub ml_frameworks: Vec<String>,
    /// GPU acceleration
    pub gpu_acceleration: bool,
}

/// Hybrid computing features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridComputingFeatures {
    /// Classical-quantum integration
    pub classical_quantum_integration: bool,
    /// Real-time feedback
    pub realtime_feedback: bool,
    /// Iterative algorithms
    pub iterative_algorithms: bool,
    /// HPC integration
    pub hpc_integration: bool,
    /// Edge computing
    pub edge_computing: bool,
}

/// Quantum networking features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumNetworkingFeatures {
    /// Quantum internet support
    pub quantum_internet: bool,
    /// Quantum key distribution
    pub qkd_support: bool,
    /// Distributed quantum computing
    pub distributed_computing: bool,
    /// Quantum teleportation
    pub quantum_teleportation: bool,
    /// Network protocols
    pub network_protocols: Vec<String>,
}

/// Research capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchCapabilities {
    /// Research partnerships
    pub research_partnerships: bool,
    /// Academic pricing
    pub academic_pricing: bool,
    /// Research tools
    pub research_tools: bool,
    /// Data sharing capabilities
    pub data_sharing: bool,
    /// Publication support
    pub publication_support: bool,
}
