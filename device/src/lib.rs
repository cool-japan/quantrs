//! Quantum device connectors for the quantrs framework.
//!
//! This crate provides connectivity to quantum hardware providers like IBM Quantum,
//! Azure Quantum, and AWS Braket. It enables users to run quantum circuits on real
//! quantum hardware or cloud-based simulators.

use quantrs2_circuit::prelude::Circuit;
use std::collections::HashMap;
use thiserror::Error;

pub mod adaptive_compilation;
pub mod advanced_benchmarking_suite;
pub mod advanced_scheduling;
/// Public exports for commonly used types
// Forward declaration - implemented below
// pub mod prelude;
pub mod aws;
pub mod aws_device;
pub mod azure;
pub mod azure_device;
pub mod backend_traits;
pub mod benchmarking;
pub mod calibration;
pub mod characterization;
pub mod cloud;
pub mod compiler_passes;
pub mod cross_platform_benchmarking;
pub mod crosstalk;
pub mod distributed;
pub mod dynamical_decoupling;
pub mod ibm;
pub mod ibm_device;
pub mod integrated_device_manager;
pub mod job_scheduling;
pub mod mapping_scirc2;
pub mod mid_circuit_measurements;
pub mod ml_optimization;
pub mod noise_model;
pub mod noise_modeling_scirs2;
pub mod optimization;
pub mod optimization_old;
pub mod parametric;
pub mod performance_analytics_dashboard;
pub mod performance_dashboard;
pub mod process_tomography;
pub mod pulse;
pub mod qec;
pub mod quantum_algorithm_marketplace;
pub mod quantum_network;
pub mod quantum_system_security;
pub mod routing;
pub mod routing_advanced;
pub mod security;
pub mod topology;
pub mod topology_analysis;
pub mod translation;
pub mod transpiler;
pub mod unified_benchmarking;
pub mod vqa_support;
pub mod zero_noise_extrapolation;

// AWS authentication module
#[cfg(feature = "aws")]
pub mod aws_auth;

// AWS circuit conversion module
#[cfg(feature = "aws")]
pub mod aws_conversion;

/// Result type for device operations
pub type DeviceResult<T> = Result<T, DeviceError>;

/// Errors that can occur during device operations
#[derive(Error, Debug)]
pub enum DeviceError {
    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("API error: {0}")]
    APIError(String),

    #[error("Job submission error: {0}")]
    JobSubmission(String),

    #[error("Job execution error: {0}")]
    JobExecution(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Device not supported: {0}")]
    UnsupportedDevice(String),

    #[error("Circuit conversion error: {0}")]
    CircuitConversion(String),

    #[error("Insufficient qubits: required {required}, available {available}")]
    InsufficientQubits { required: usize, available: usize },

    #[error("Routing error: {0}")]
    RoutingError(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Optimization error: {0}")]
    OptimizationError(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

/// Convert QuantRS2Error to DeviceError
impl From<quantrs2_core::error::QuantRS2Error> for DeviceError {
    fn from(err: quantrs2_core::error::QuantRS2Error) -> Self {
        DeviceError::APIError(err.to_string())
    }
}

/// General representation of quantum hardware
#[cfg(feature = "ibm")]
#[async_trait::async_trait]
pub trait QuantumDevice {
    /// Check if the device is available for use
    async fn is_available(&self) -> DeviceResult<bool>;

    /// Get the number of qubits on the device
    async fn qubit_count(&self) -> DeviceResult<usize>;

    /// Get device properties such as error rates, connectivity, etc.
    async fn properties(&self) -> DeviceResult<HashMap<String, String>>;

    /// Check if the device is a simulator
    async fn is_simulator(&self) -> DeviceResult<bool>;
}

#[cfg(not(feature = "ibm"))]
pub trait QuantumDevice {
    /// Check if the device is available for use
    fn is_available(&self) -> DeviceResult<bool>;

    /// Get the number of qubits on the device
    fn qubit_count(&self) -> DeviceResult<usize>;

    /// Get device properties such as error rates, connectivity, etc.
    fn properties(&self) -> DeviceResult<HashMap<String, String>>;

    /// Check if the device is a simulator
    fn is_simulator(&self) -> DeviceResult<bool>;
}

/// Trait for devices that can execute quantum circuits
#[cfg(feature = "ibm")]
#[async_trait::async_trait]
pub trait CircuitExecutor: QuantumDevice {
    /// Execute a quantum circuit on the device
    async fn execute_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        shots: usize,
    ) -> DeviceResult<CircuitResult>;

    /// Execute multiple circuits in parallel
    async fn execute_circuits<const N: usize>(
        &self,
        circuits: Vec<&Circuit<N>>,
        shots: usize,
    ) -> DeviceResult<Vec<CircuitResult>>;

    /// Check if a circuit can be executed on the device
    async fn can_execute_circuit<const N: usize>(&self, circuit: &Circuit<N>)
        -> DeviceResult<bool>;

    /// Get estimated queue time for a circuit execution
    async fn estimated_queue_time<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> DeviceResult<std::time::Duration>;
}

#[cfg(not(feature = "ibm"))]
pub trait CircuitExecutor: QuantumDevice {
    /// Execute a quantum circuit on the device
    fn execute_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        shots: usize,
    ) -> DeviceResult<CircuitResult>;

    /// Execute multiple circuits in parallel
    fn execute_circuits<const N: usize>(
        &self,
        circuits: Vec<&Circuit<N>>,
        shots: usize,
    ) -> DeviceResult<Vec<CircuitResult>>;

    /// Check if a circuit can be executed on the device
    fn can_execute_circuit<const N: usize>(&self, circuit: &Circuit<N>) -> DeviceResult<bool>;

    /// Get estimated queue time for a circuit execution
    fn estimated_queue_time<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> DeviceResult<std::time::Duration>;
}

/// Result of a circuit execution on hardware
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CircuitResult {
    /// Counts of each basis state
    pub counts: HashMap<String, usize>,

    /// Total number of shots executed
    pub shots: usize,

    /// Additional metadata about the execution
    pub metadata: HashMap<String, String>,
}

/// Check if device integration is available and properly set up
pub fn is_available() -> bool {
    #[cfg(any(feature = "ibm", feature = "azure", feature = "aws"))]
    {
        return true;
    }

    #[cfg(not(any(feature = "ibm", feature = "azure", feature = "aws")))]
    {
        false
    }
}

/// Create an IBM Quantum client
///
/// Requires the "ibm" feature to be enabled
#[cfg(feature = "ibm")]
pub fn create_ibm_client(token: &str) -> DeviceResult<ibm::IBMQuantumClient> {
    ibm::IBMQuantumClient::new(token)
}

/// Create an IBM Quantum client
///
/// This function is available as a stub when the "ibm" feature is not enabled
#[cfg(not(feature = "ibm"))]
pub fn create_ibm_client(_token: &str) -> DeviceResult<()> {
    Err(DeviceError::UnsupportedDevice(
        "IBM Quantum support not enabled. Recompile with the 'ibm' feature.".to_string(),
    ))
}

/// Create an IBM Quantum device instance
#[cfg(feature = "ibm")]
pub async fn create_ibm_device(
    token: &str,
    backend_name: &str,
    config: Option<ibm_device::IBMDeviceConfig>,
) -> DeviceResult<impl QuantumDevice + CircuitExecutor> {
    let client = create_ibm_client(token)?;
    ibm_device::IBMQuantumDevice::new(client, backend_name, config).await
}

/// Create an IBM Quantum device instance
///
/// This function is available as a stub when the "ibm" feature is not enabled
#[cfg(not(feature = "ibm"))]
pub async fn create_ibm_device(
    _token: &str,
    _backend_name: &str,
    _config: Option<()>,
) -> DeviceResult<()> {
    Err(DeviceError::UnsupportedDevice(
        "IBM Quantum support not enabled. Recompile with the 'ibm' feature.".to_string(),
    ))
}

/// Create an Azure Quantum client
///
/// Requires the "azure" feature to be enabled
#[cfg(feature = "azure")]
pub fn create_azure_client(
    token: &str,
    subscription_id: &str,
    resource_group: &str,
    workspace: &str,
    region: Option<&str>,
) -> DeviceResult<azure::AzureQuantumClient> {
    azure::AzureQuantumClient::new(token, subscription_id, resource_group, workspace, region)
}

/// Create an Azure Quantum client
///
/// This function is available as a stub when the "azure" feature is not enabled
#[cfg(not(feature = "azure"))]
pub fn create_azure_client(
    _token: &str,
    _subscription_id: &str,
    _resource_group: &str,
    _workspace: &str,
    _region: Option<&str>,
) -> DeviceResult<()> {
    Err(DeviceError::UnsupportedDevice(
        "Azure Quantum support not enabled. Recompile with the 'azure' feature.".to_string(),
    ))
}

/// Create an Azure Quantum device instance
#[cfg(feature = "azure")]
pub async fn create_azure_device(
    client: azure::AzureQuantumClient,
    target_id: &str,
    provider_id: Option<&str>,
    config: Option<azure_device::AzureDeviceConfig>,
) -> DeviceResult<impl QuantumDevice + CircuitExecutor> {
    azure_device::AzureQuantumDevice::new(client, target_id, provider_id, config).await
}

/// Create an Azure Quantum device instance
///
/// This function is available as a stub when the "azure" feature is not enabled
#[cfg(not(feature = "azure"))]
pub async fn create_azure_device(
    _client: (),
    _target_id: &str,
    _provider_id: Option<&str>,
    _config: Option<()>,
) -> DeviceResult<()> {
    Err(DeviceError::UnsupportedDevice(
        "Azure Quantum support not enabled. Recompile with the 'azure' feature.".to_string(),
    ))
}

/// Create an AWS Braket client
///
/// Requires the "aws" feature to be enabled
#[cfg(feature = "aws")]
pub fn create_aws_client(
    access_key: &str,
    secret_key: &str,
    region: Option<&str>,
    s3_bucket: &str,
    s3_key_prefix: Option<&str>,
) -> DeviceResult<aws::AWSBraketClient> {
    aws::AWSBraketClient::new(access_key, secret_key, region, s3_bucket, s3_key_prefix)
}

/// Create an AWS Braket client
///
/// This function is available as a stub when the "aws" feature is not enabled
#[cfg(not(feature = "aws"))]
pub fn create_aws_client(
    _access_key: &str,
    _secret_key: &str,
    _region: Option<&str>,
    _s3_bucket: &str,
    _s3_key_prefix: Option<&str>,
) -> DeviceResult<()> {
    Err(DeviceError::UnsupportedDevice(
        "AWS Braket support not enabled. Recompile with the 'aws' feature.".to_string(),
    ))
}

/// Create an AWS Braket device instance
#[cfg(feature = "aws")]
pub async fn create_aws_device(
    client: aws::AWSBraketClient,
    device_arn: &str,
    config: Option<aws_device::AWSDeviceConfig>,
) -> DeviceResult<impl QuantumDevice + CircuitExecutor> {
    aws_device::AWSBraketDevice::new(client, device_arn, config).await
}

/// Create an AWS Braket device instance
///
/// This function is available as a stub when the "aws" feature is not enabled
#[cfg(not(feature = "aws"))]
pub async fn create_aws_device(
    _client: (),
    _device_arn: &str,
    _config: Option<()>,
) -> DeviceResult<()> {
    Err(DeviceError::UnsupportedDevice(
        "AWS Braket support not enabled. Recompile with the 'aws' feature.".to_string(),
    ))
}

/// Re-exports of commonly used types and traits
pub mod prelude {
    pub use crate::advanced_benchmarking_suite::{
        AdvancedBenchmarkConfig, AdvancedBenchmarkResult, AdvancedHardwareBenchmarkSuite,
        MLAnalysisResult, PredictionResult, AnomalyDetectionResult, AdvancedStatisticalResult,
        MLBenchmarkConfig, RealtimeBenchmarkConfig, PredictiveModelingConfig,
        AnomalyDetectionConfig, AdvancedStatsConfig, BenchmarkOptimizationConfig,
    };
    pub use crate::backend_traits::{
        google_gates, honeywell_gates, ibm_gates, ionq_gates, query_backend_capabilities,
        rigetti_gates, BackendCapabilities, BackendFeatures, BackendPerformance, HardwareGate,
    };
    pub use crate::benchmarking::{
        BenchmarkConfig, BenchmarkResult, BenchmarkSuite, GraphAnalysis, HardwareBenchmarkSuite,
        NoiseAnalysis, PerformanceMetrics as BenchmarkingMetrics, StatisticalAnalysis,
    };
    pub use crate::calibration::{
        create_ideal_calibration, CalibrationBuilder, CalibrationManager, CrosstalkMatrix,
        DeviceCalibration, DeviceTopology, QubitCalibration, ReadoutCalibration,
        SingleQubitGateCalibration, TwoQubitGateCalibration,
    };
    pub use crate::characterization::{
        CrosstalkCharacterization as CharacterizationCrosstalk, DriftTracker, ProcessTomography,
        RandomizedBenchmarking, StateTomography,
    };
    pub use crate::cloud::{
        allocation::{AllocationAlgorithm, ResourceOptimizationObjective},
        cost_management::CostOptimizationStrategy,
        monitoring::CloudMonitoringConfig,
        orchestration::{LoadBalancingStrategy, PerformanceOptimizationStrategy},
        providers::{CloudProvider, MultiProviderConfig, ProviderSelectionStrategy},
    };
    pub use crate::compiler_passes::{
        CompilationResult, CompilerConfig, HardwareAllocation, HardwareCompiler,
        HardwareConstraints, OptimizationObjective, OptimizationStats, PassInfo,
        PerformancePrediction,
    };
    pub use crate::crosstalk::{
        CrosstalkAnalyzer, CrosstalkCharacterization, CrosstalkConfig, CrosstalkMechanism,
        MitigationStrategy, SpatialCrosstalkAnalysis, SpectralCrosstalkAnalysis,
        TemporalCrosstalkAnalysis,
    };
    pub use crate::distributed::{
        AuthenticationMethod as DistributedAuthenticationMethod, CircuitDecompositionResult,
        CommunicationProtocol, DistributedCommand, DistributedComputingConfig,
        DistributedCostAnalysis, DistributedEvent, DistributedExecutionResult,
        DistributedExecutionStatus, DistributedMonitoringConfig, DistributedOptimizationConfig,
        DistributedOrchestratorConfig, DistributedPerformanceAnalytics,
        DistributedQuantumOrchestrator, DistributedResourceConfig, DistributedResourceUtilization,
        DistributedWorkflow, DistributedWorkflowType,
        EncryptionAlgorithm as DistributedEncryptionAlgorithm, FaultToleranceConfig,
        FaultToleranceMetrics, LoadBalancingAlgorithm, LoadBalancingConfig, NetworkConfig,
        NetworkPerformanceMetrics, NetworkTopology, NodeCapabilities, NodeInfo, NodeStatus,
        OptimizationObjective as DistributedOptimizationObjective, ReplicationStrategy,
        SecurityAuditTrail, SecurityConfig as DistributedSecurityConfig,
        WorkloadDistributionStrategy,
    };
    pub use crate::ibm::IBMCircuitConfig;
    pub use crate::integrated_device_manager::{
        DeviceInfo,
        ExecutionStatus, // ExecutionStrategy, DeviceSelectionCriteria, ExecutionMode, DeviceCapabilityInfo,
        // OptimizationMode, IntegratedAnalyticsConfig, HardwareCompatibilityInfo, DeviceHealthInfo,
        IntegratedDeviceConfig,
        IntegratedExecutionResult,
        IntegratedQuantumDeviceManager,
    };
    pub use crate::job_scheduling::{
        create_batch_job_config, create_high_priority_config, create_realtime_config,
        AllocationStrategy as JobAllocationStrategy, BackendPerformance as JobBackendPerformance,
        BackendStatus, ExecutionMetrics, JobConfig, JobExecution, JobId, JobPriority, JobStatus,
        QuantumJob, QuantumJobScheduler, QueueAnalytics, ResourceRequirements, SchedulerEvent,
        SchedulingParams, SchedulingStrategy,
    };
    pub use crate::mapping_scirc2::{
        InitialMappingAlgorithm, OptimizationObjective as MappingObjective, SciRS2MappingConfig,
        SciRS2MappingResult, SciRS2QubitMapper, SciRS2RoutingAlgorithm,
    };
    pub use crate::mid_circuit_measurements::{
        ExecutionStats, HardwareOptimizations, MeasurementEvent, MidCircuitCapabilities,
        MidCircuitConfig, MidCircuitDeviceExecutor, MidCircuitExecutionResult, MidCircuitExecutor,
        PerformanceMetrics, ValidationConfig, ValidationResult,
    };
    pub use crate::noise_model::{
        CalibrationNoiseModel, GateNoiseParams, NoiseModelBuilder, QubitNoiseParams,
        ReadoutNoiseParams,
    };
    pub use crate::noise_modeling_scirs2::{SciRS2NoiseConfig, SciRS2NoiseModeler};
    pub use crate::optimization::{
        CalibrationOptimizer, FidelityEstimator, OptimizationConfig, OptimizationResult,
        PulseOptimizer,
    };
    pub use crate::parametric::{
        BatchExecutionRequest, BatchExecutionResult, Parameter, ParameterExpression,
        ParameterOptimizer, ParametricCircuit, ParametricCircuitBuilder, ParametricExecutor,
        ParametricGate, ParametricTemplates,
    };
    pub use crate::pulse::{
        ChannelType, MeasLevel, MeasurementData, PulseBackend, PulseBuilder, PulseCalibration,
        PulseInstruction, PulseLibrary, PulseResult, PulseSchedule, PulseShape, PulseTemplates,
    };
    pub use crate::qec::{
        AdaptiveQECConfig, ErrorMitigationConfig, QECCodeType, QECConfig, QECMLConfig,
        QECMonitoringConfig, QECOptimizationConfig, QECStrategy, SyndromeDetectionConfig,
    };
    pub use crate::quantum_algorithm_marketplace::{
        AlgorithmOptimizationStrategy, DiscoveryAlgorithm, IncentiveMechanism,
        MLRecommendationModel, QuantumAlgorithmMarketplaceConfig,
    };
    pub use crate::quantum_system_security::{
        AuthenticationMethod as SecurityAuthenticationMethod, AuthorizationModel,
        ComplianceStandard, EncryptionProtocol, PostQuantumAlgorithm, QuantumSecurityConfig,
        QuantumSecurityExecutionResult, QuantumSecurityExecutionStatus,
        QuantumSystemSecurityFramework, RegulatoryFramework,
        SecurityAnalyticsEngine as SecurityAnalyticsEngineType, SecurityClassification,
        SecurityMLModel, SecurityObjective, SecurityOperation, SecurityOperationType,
        SecurityStandard, ThreatDetectionAlgorithm,
    };
    pub use crate::routing_advanced::{
        AdvancedQubitRouter, AdvancedRoutingResult, AdvancedRoutingStrategy, RoutingMetrics,
        SwapOperation,
    };
    pub use crate::topology_analysis::{
        create_standard_topology, AllocationStrategy, HardwareMetrics, TopologyAnalysis,
        TopologyAnalyzer,
    };
    pub use crate::translation::{
        validate_native_circuit, DecomposedGate, GateTranslator, HardwareBackend, NativeGateSet,
        OptimizationStrategy, TranslationMethod, TranslationOptimizer, TranslationRule,
        TranslationStats,
    };
    pub use crate::unified_benchmarking::{
        UnifiedQuantumBenchmarkSystem, UnifiedBenchmarkConfig, UnifiedBenchmarkResult,
        QuantumPlatform as UnifiedQuantumPlatform, BenchmarkEvent,
        PerformanceBaseline, BaselineMetric, BaselineMetricValue,
    };
    pub use crate::vqa_support::{
        analysis::ConvergenceAnalysis,
        circuits::ParametricCircuit as VQAParametricCircuit,
        config::{
            AdaptiveShotConfig, ConvergenceCriterion, GradientMethod, MultiStartConfig,
            OptimizationTrajectory, ResourceUtilization, VQAAlgorithmType, VQAConfig,
            VQAHardwareAnalysis, VQAHardwareConfig, VQANoiseMitigation, VQAOptimizationConfig,
            VQAOptimizer, VQAStatisticalAnalysis, VQAStatisticalConfig, VQAValidationConfig,
            VQAValidationResults, WarmRestartConfig,
        },
        executor::{VQAExecutor, VQAResult},
        objectives::ObjectiveFunction,
    };
    pub use crate::zero_noise_extrapolation::{
        CircuitFolder, ExtrapolationFitter, ExtrapolationMethod, NoiseScalingMethod, Observable,
        ZNECapable, ZNEConfig, ZNEExecutor, ZNEResult,
    };
}
