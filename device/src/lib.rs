//! Quantum device connectors for the QuantRS2 framework.
//!
//! This crate provides connectivity to quantum hardware providers like IBM Quantum,
//! Azure Quantum, and AWS Braket. It enables users to run quantum circuits on real
//! quantum hardware or cloud-based simulators.
//!
//! ## Recent Updates (v0.1.0-beta.3)
//!
//! - ✅ Re-enabled enhanced SciRS2 modules with full API compliance
//! - ✅ `scirs2_hardware_benchmarks_enhanced`: ML-driven performance prediction and analysis
//! - ✅ `scirs2_noise_characterization_enhanced`: Advanced noise modeling with SciRS2 stats
//! - ✅ Enhanced transpilation using SciRS2 v0.1.0-rc.2's graph algorithms
//! - ✅ Stable APIs for IBM Quantum, Azure Quantum, and AWS Braket
//! - ✅ All 406 tests passing with zero compilation warnings
//! - ✅ Full SciRS2 Policy compliance for scientific computing operations

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
pub mod circuit_integration;
pub mod circuit_migration;
pub mod cloud;
pub mod continuous_variable;
pub mod scirs2_calibration_enhanced;
// pub mod cost_optimization;
pub mod compiler_passes;
pub mod cross_platform_benchmarking;
pub mod crosstalk;
pub mod distributed;
pub mod dynamical_decoupling;
pub mod hardware_parallelization;
pub mod hybrid_quantum_classical;
pub mod ibm;
pub mod ibm_device;
pub mod integrated_device_manager;
pub mod job_scheduling;
// pub mod mapping_scirc2; // Temporarily disabled due to scirs2-graph API changes
pub mod algorithm_marketplace;
pub mod mid_circuit_measurements;
pub mod ml_optimization;
pub mod neutral_atom;
pub mod noise_model;
pub mod noise_modeling_scirs2;
pub mod optimization;
pub mod optimization_old;
pub mod parametric;
pub mod performance_analytics_dashboard;
pub mod performance_dashboard;
pub mod photonic;
pub mod process_tomography;
pub mod provider_capability_discovery;
pub mod pulse;
pub mod qec;
pub mod quantum_ml;
pub mod quantum_ml_integration;
pub mod quantum_network;
pub mod quantum_system_security;
pub mod routing;
pub mod routing_advanced;
// Beta.3: Enhanced modules successfully re-enabled with full SciRS2 v0.1.0-rc.2 compliance
pub mod scirs2_hardware_benchmarks_enhanced;
pub mod scirs2_noise_characterization_enhanced;
pub mod security;
pub mod telemetry;
pub mod topological;
pub mod topology;
pub mod topology_analysis;
pub mod translation;
pub mod transpiler;
pub mod unified_benchmarking;
pub mod unified_error_handling;
pub mod vqa_support;
pub mod zero_noise_extrapolation;

// Test utilities
#[cfg(test)]
pub mod test_utils;

// AWS authentication module
#[cfg(feature = "aws")]
pub mod aws_auth;

// AWS circuit conversion module
#[cfg(feature = "aws")]
pub mod aws_conversion;

/// Result type for device operations
pub type DeviceResult<T> = Result<T, DeviceError>;

/// Errors that can occur during device operations
#[derive(Error, Debug, Clone)]
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

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

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

    #[error("Invalid mapping: {0}")]
    InvalidMapping(String),

    #[error("Graph analysis error: {0}")]
    GraphAnalysisError(String),

    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    #[error("Device not initialized: {0}")]
    DeviceNotInitialized(String),

    #[error("Job execution failed: {0}")]
    JobExecutionFailed(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Unknown job status: {0}")]
    UnknownJobStatus(String),

    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),
}

/// Convert QuantRS2Error to DeviceError
impl From<quantrs2_core::error::QuantRS2Error> for DeviceError {
    fn from(err: quantrs2_core::error::QuantRS2Error) -> Self {
        DeviceError::APIError(err.to_string())
    }
}

/// Convert String to DeviceError
impl From<String> for DeviceError {
    fn from(err: String) -> Self {
        DeviceError::APIError(err)
    }
}

/// Convert OptimizeError to DeviceError
impl From<crate::ml_optimization::OptimizeError> for DeviceError {
    fn from(err: crate::ml_optimization::OptimizeError) -> Self {
        DeviceError::OptimizationError(err.to_string())
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
        AdvancedStatisticalResult, AdvancedStatsConfig, AnomalyDetectionConfig,
        AnomalyDetectionResult, BenchmarkOptimizationConfig, MLAnalysisResult, MLBenchmarkConfig,
        PredictionResult, PredictiveModelingConfig, RealtimeBenchmarkConfig,
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
    pub use crate::circuit_integration::{
        create_high_performance_config, create_universal_interface, AnalyticsConfig, CacheConfig,
        CircuitVariant, CostInfo, ExecutionAnalytics, ExecutionMetadata, ExecutionResult,
        IntegrationConfig, OptimizationSettings, OptimizedCircuit,
        PerformanceMetrics as CircuitPerformanceMetrics, PlatformAdapter, PlatformConfig,
        PlatformMetrics, SelectionCriteria, UniversalCircuitInterface,
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
    pub use crate::continuous_variable::{
        cluster_states::{
            ClusterStateConfig, ClusterStateGenerator, ClusterStateType, ClusterStateValidation,
            MBQCMeasurement, MBQCResult,
        },
        create_cluster_state_cv_device, create_gaussian_cv_device,
        cv_gates::{CVGateLibrary, CVGateParams, CVGateSequence, CVGateType},
        error_correction::{
            CVErrorCorrectionCode, CVErrorCorrectionConfig, CVErrorCorrector, CVLogicalState,
            CorrectionResult,
        },
        gaussian_states::GaussianState,
        heterodyne::{
            HeterodyneDetector, HeterodyneDetectorConfig, HeterodyneResult, HeterodyneStatistics,
        },
        homodyne::{HomodyneDetector, HomodyneDetectorConfig, HomodyneResult, HomodyneStatistics},
        measurements::{
            CVMeasurementConfig, CVMeasurementEngine, CVMeasurementScheme, MeasurementStatistics,
        },
        CVDeviceConfig, CVDeviceDiagnostics, CVEntanglementMeasures, CVMeasurementOutcome,
        CVMeasurementResult, CVMeasurementType, CVModeState, CVQuantumDevice, CVSystemType,
        Complex,
    };
    pub use crate::scirs2_calibration_enhanced::{
        AnalysisOptions, CNOTData, CalibrationConfig, CalibrationFeedback, CalibrationInput,
        CalibrationModel, CalibrationObjective, CalibrationPrediction, CalibrationProtocols,
        CalibrationRecommendation, CalibrationReport, CalibrationState, CalibrationSummary,
        CalibrationVisualizations, ChevronData, CoherenceTimes, CoherentError, CorrelatedError,
        CrosstalkCharacterization as EnhancedCrosstalkCharacterization, CrosstalkParameters,
        CrosstalkProtocols, DetailedResults, DiscriminationData, DiscriminationParameters,
        DragData, DriftAnalysis, DriftDirection, DriftMeasurement, EnhancedCalibrationConfig,
        EnhancedCalibrationSystem, ErrorAnalysis, ErrorCharacterization, ErrorData, ErrorModel,
        ErrorModelTrait, GSTData, GateSet, HardwareSpec, IQData, IQParameters,
        IdentificationMethod, IncoherentError, IncoherentErrorType, MLSystemParameters,
        PerformanceMetrics, PerformanceThresholds, Priority, ProcessTomographyData, QualityMetrics,
        QuantumOperation, QubitParameters, RBData, RabiData, RamseyData,
        ReadoutCalibration as EnhancedReadoutCalibration, ReadoutParameters, ReadoutProtocols,
        RecommendationCategory, SingleQubitCalibration, SingleQubitProtocols, SystemAnalysis,
        SystemCalibrationResult, SystemModel, TwoQubitCalibration, TwoQubitParameters,
        TwoQubitProtocols, ZZData,
    };
    // pub use crate::cost_optimization::{
    //     CostOptimizationEngine, CostOptimizationConfig, CostOptimizationStrategy as CostStrategy, CostEstimate,
    //     CostBreakdown, CostEstimationMetadata, BudgetConfig, BudgetStatus, BudgetRolloverPolicy,
    //     CostModel, CostModelType, ProviderComparisonResult, ProviderMetrics, ComparisonMetric,
    //     PredictiveModelingConfig as CostPredictiveConfig, PredictiveModelType, PredictionResult as CostPredictionResult,
    //     ResourceRequirements as CostResourceRequirements, OptimizationResult as CostOptimizationResult,
    //     OptimizationStatus, BudgetConstraint, TimeConstraint, QualityRequirement,
    //     OptimizationRecommendation, RecommendationType, CostTrends, TrendDirection, CostAnomaly,
    //     MonitoringMetric, CostAlertConfig, CostAlertRule, AlertCondition, AlertSeverity,
    //     NotificationChannel, DashboardConfig, DashboardWidget, MLCostModel, PredictiveModel,
    // };
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
    pub use crate::hardware_parallelization::{
        ExecutionConstraints, ExecutionQualityMetrics, HardwareAwarenessConfig,
        HardwareParallelizationEngine, LoadBalancingConfig as ParallelLoadBalancingConfig,
        LoadBalancingResult, OptimizationSuggestion, ParallelCircuitTask, ParallelExecutionResult,
        ParallelGateTask, ParallelResourceRequirements, ParallelSchedulingConfig,
        ParallelizationConfig, ParallelizationStrategy,
        PerformanceMetrics as ParallelPerformanceMetrics, PerformanceOptimizationConfig,
        QualityRequirements, ResourceAllocationConfig, ResourceConstraints, ResourceUsage,
        TaskPriority, TimingConstraints,
    };
    pub use crate::hybrid_quantum_classical::{
        AdaptationAlgorithm, AdaptiveControlConfig, BackendSelectionConfig, BackoffStrategy,
        CircuitOptimizationConfig, ClassicalComputationConfig, ClassicalComputationResult,
        ConvergenceConfig, ConvergenceCriterion, ConvergenceReason, ConvergenceStatus,
        ErrorHandlingConfig, ErrorRecoveryStrategy, FeedbackAlgorithm, FeedbackControlConfig,
        HybridLoopConfig, HybridLoopResult, HybridLoopState, HybridLoopStrategy,
        HybridOptimizationConfig, HybridOptimizer, HybridPerformanceConfig,
        HybridQuantumClassicalExecutor, IterationResult, NoiseModelingConfig, OptimizationLevel,
        OptimizationPass, OptimizationSummary, PerformanceMetrics as HybridPerformanceMetrics,
        QualityMetrics as HybridQualityMetrics, QuantumExecutionConfig, QuantumExecutionResult,
        RetryConfig, SelectionCriterion, StateEstimationMethod,
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
    // Temporarily disabled due to scirs2-graph API changes
    // pub use crate::mapping_scirc2::{
    //     InitialMappingAlgorithm, OptimizationObjective as MappingObjective, SciRS2MappingConfig,
    //     SciRS2MappingResult, SciRS2QubitMapper, SciRS2RoutingAlgorithm,
    // };
    pub use crate::algorithm_marketplace::{
        APIConfig, ActiveDeployment, AlgorithmDeploymentManager, AlgorithmDiscoveryEngine,
        AlgorithmInfo, AlgorithmOptimizationEngine, AlgorithmRegistration, AlgorithmRegistry,
        AlgorithmValidationService, AlgorithmVersioningSystem, DeploymentRequest, DeploymentStatus,
        DiscoveryCriteria, MarketplaceAPI, MarketplaceConfig, MonetizationSystem,
        OptimizationConfig as MarketplaceOptimizationConfig, PaymentMethod, Permission,
        PricingStrategy, QuantumAlgorithmMarketplace, SubscriptionModel, UserSession, UserType,
        ValidationConfig as MarketplaceValidationConfig, VersioningConfig,
    };
    pub use crate::mid_circuit_measurements::{
        ExecutionStats, HardwareOptimizations, MeasurementEvent, MidCircuitCapabilities,
        MidCircuitConfig, MidCircuitDeviceExecutor, MidCircuitExecutionResult, MidCircuitExecutor,
        PerformanceMetrics as MidCircuitPerformanceMetrics, ValidationConfig, ValidationResult,
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
    pub use crate::photonic::{
        create_photonic_device,
        gate_based::{
            OpticalElement, PhotonicCircuitCompiler, PhotonicCircuitImplementation,
            PhotonicGateImpl, PhotonicGates, PhotonicHardwareConstraints, PhotonicQubitEncoding,
            PhotonicQubitState, PhotonicResourceRequirements,
        },
        gates::{
            BeamsplitterGate, CrossKerrGate, DisplacementGate, KerrGate, PhaseRotationGate,
            SqueezingGate, TwoModeSqueezingGate,
        },
        validate_photonic_config, PhotonicCircuitResult, PhotonicClient, PhotonicConfig,
        PhotonicDeviceConfig, PhotonicExecutionMetadata, PhotonicMeasurementData, PhotonicMode,
        PhotonicQuantumDevice, PhotonicSystemType,
    };
    pub use crate::provider_capability_discovery::{
        create_high_performance_discovery_config, create_provider_discovery_system,
        CachedCapability, CapabilityRequirements, ComparisonResults, ConnectivityRequirement,
        DiscoveryCommand, DiscoveryConfig, DiscoveryEvent, DiscoveryStrategy, FilteringConfig,
        ProviderCapabilities, ProviderCapabilityDiscoverySystem, ProviderFeature, ProviderInfo,
        ProviderRanking, ProviderType, ReportType as DiscoveryReportType, TopologyType,
        VerificationConfig, VerificationStatus,
    };
    pub use crate::pulse::{
        ChannelType, MeasLevel, MeasurementData, PulseBackend, PulseBuilder, PulseCalibration,
        PulseInstruction, PulseLibrary, PulseResult, PulseSchedule, PulseShape, PulseTemplates,
    };
    pub use crate::qec::{
        adaptive::AdaptiveQECConfig, codes::QECCodeType, detection::SyndromeDetectionConfig,
        mitigation::ErrorMitigationConfig, QECConfig, QECMLConfig, QECMonitoringConfig,
        QECOptimizationConfig, QECStrategy,
    };
    pub use crate::quantum_ml::{
        create_qaoa_accelerator, create_vqc_accelerator,
        gradients::{
            create_finite_difference_calculator, create_parameter_shift_calculator, GradientConfig,
            GradientUtils, Observable as QMLObservable, ObservableTerm, QuantumGradientCalculator,
        },
        optimization::{
            create_gradient_free_optimizer, create_gradient_optimizer, GradientBasedOptimizer,
            GradientFreeOptimizer, ObjectiveFunction as QMLObjectiveFunction,
            OptimizationResult as QMLOptResult, OptimizationStep, OptimizerConfig,
            QuantumOptimizer, VQEObjectiveFunction,
        },
        quantum_neural_networks::{
            create_pqc_classifier, create_qcnn_classifier, ClassificationResult,
            EntanglingStrategy, InputEncoding, OutputDecoding, PQCNetwork, QConvLayer,
            QNNArchitecture, QNNType, QPoolingLayer, QPoolingType, QuantumNeuralNetwork, QCNN, VQC,
        },
        training::{
            create_supervised_trainer, create_training_data, BatchObjectiveFunction,
            CrossEntropyLoss, LossFunction as QMLLossFunction, MSELoss, QuantumTrainer,
            TrainingData as QMLTrainingData, TrainingMetrics, TrainingResult as QMLTrainingResult,
        },
        variational_algorithms::{
            create_molecular_vqe, AdamOptimizer, EntanglingGateType, Hamiltonian,
            HardwareEfficientAnsatz, MolecularHamiltonian, ParameterizedQuantumCircuit,
            PauliOperator, PauliTerm, QAOAConfig, QAOAResult, QAOASolution, QuantumGate,
            QuantumState, VQEConfig, VQEResult, VariationalAnsatz, VariationalOptimizer, QAOA, VQE,
        },
        CircuitStructure, GradientMethod as QMLGradientMethod, InferenceData, InferenceResult,
        ModelExportFormat, ModelRegistry, NoiseResilienceLevel,
        OptimizationResult as QMLOptimizationResult, OptimizerType, QMLAccelerator, QMLConfig,
        QMLDiagnostics, QMLModel, QMLModelType, TrainingData, TrainingEpoch, TrainingResult,
        TrainingStatistics,
    };
    pub use crate::quantum_ml_integration::{
        create_high_performance_qml_config, create_qml_integration_hub,
        AnomalyType as QMLAnomalyType, FrameworkBridge, HybridMLOptimizer, LossFunction,
        MLFramework, MLPerformanceAnalytics, QMLArchitecture, QMLDataBatch, QMLDataPipeline,
        QMLDataset, QMLInferenceResult, QMLIntegrationConfig, QMLMonitoringConfig,
        QMLOptimizationConfig, QMLResourceConfig, QMLResourceRequirements, QMLTrainingConfig,
        QMLTrainingOrchestrator, QuantumEncodingType, QuantumMLIntegrationHub,
        QuantumNeuralNetworkExecutor, TrainingPriority,
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
    pub use crate::scirs2_hardware_benchmarks_enhanced::{
        AnalysisMethod, AnomalyType, BenchmarkConfig as EnhancedBenchmarkConfig2,
        BenchmarkRecommendation, BenchmarkReport, BenchmarkSuite as EnhancedBenchmarkSuite,
        BenchmarkSuiteResult, BenchmarkVisualizations, ChartType, ComparativeAnalysis,
        ComparativeSummary, ComparisonChart, ComparisonDataSet, ComprehensiveBenchmarkResult,
        ConfidenceInterval, CorrelationMatrix, DataSeries, DegradationEvent, DegradationThreshold,
        DegradationTimeline, DegradationType, DeviceComparison, DeviceInfo as BenchmarkDeviceInfo,
        EffortLevel, EnhancedBenchmarkConfig, EnhancedHardwareBenchmark, ExecutiveSummary,
        ExportFormat as BenchmarkExportFormat, HeatmapVisualization, HistoricalAnomaly,
        HistoricalComparison, ImpactLevel, IndustryPosition, IndustryTier,
        MaintenanceRecommendation, MaintenanceType, MetricReport, MetricTrend, PerformanceMetric,
        PerformancePredictions, PerformanceTrend, PlotType as BenchmarkPlotType,
        PredictedPerformance, PredictionSummary, Priority as BenchmarkPriority, RadarChart,
        RadarDataSet, RecommendationCategory as BenchmarkRecommendationCategory,
        ReportingOptions as BenchmarkReportingOptions, Severity as BenchmarkSeverity,
        SignificanceTest, StatisticalAnalysis as BenchmarkStatisticalAnalysis, StatisticalSummary,
        SuiteReport, SuiteStatistics, TrendPlot,
    };
    pub use crate::scirs2_noise_characterization_enhanced::{
        AlertType, AnalysisParameters, CorrelationAnalysis as NoiseCorrelationAnalysis,
        EnhancedNoiseCharacterizer, EnhancedNoiseConfig, ExportFormat as NoiseExportFormat,
        HeatmapData, Landscape3D, MLNoiseInsights, ModelAnalysis, NoiseAlert,
        NoiseCharacterizationConfig, NoiseCharacterizationResult, NoiseClassification, NoiseModel,
        NoisePredictions, NoiseReport, NoiseSummary, NoiseTrend, NoiseVisualizations, PlotData,
        PlotMetadata, PlotType, PredictedNoisePoint, Priority as NoisePriority,
        Recommendation as NoiseRecommendation, RecommendationType,
        ReportingOptions as NoiseReportingOptions, Severity, SpectralAnalysis, StatisticalMethod,
        SurfaceType, TemporalAnalysis, Visualization3DParams,
    };
    pub use crate::telemetry::{
        create_high_performance_telemetry_config, create_telemetry_system, Alert, AlertConfig,
        AlertManager, AlertSeverity, AlertState, AnalyticsConfig as TelemetryAnalyticsConfig,
        AnomalyDetector, AnomalyResult, AnomalyType as TelemetryAnomalyType, ExportConfig,
        HealthStatus, Metric, MetricCollector, MetricConfig, MetricType, MonitoringConfig,
        QuantumTelemetrySystem, RealTimeMonitor, ReportType, RetentionConfig, SystemHealth,
        SystemStatus, TelemetryAnalytics, TelemetryCommand, TelemetryConfig, TelemetryEvent,
        TelemetryReport, TelemetryStorage, TrendDirection,
    };
    pub use crate::topological::{
        anyons::{AnyonFactory, AnyonTracker, ChargeAlgebra},
        braiding::{BraidGroupElement, BraidingMatrixCalculator, BraidingOperationManager},
        device::{
            create_fibonacci_device, create_ising_device, EnhancedTopologicalDevice,
            TopologicalDeviceConfig, TopologicalDeviceDiagnostics,
        },
        error_correction::{
            ErrorCorrectionConfig, RealTimeErrorMonitor, TopologicalErrorCorrector,
        },
        fusion::{FSymbolCalculator, FusionOperationExecutor, FusionTree},
        topological_codes::{
            ColorCode, ErrorCorrection, SurfaceCode, SyndromeMeasurement, TopologicalCodeType,
            TopologicalDecoder,
        },
        Anyon, BraidingDirection, BraidingOperation, BraidingResult, FusionRuleSet,
        NonAbelianAnyonType, TopologicalCapabilities, TopologicalCharge, TopologicalDevice,
        TopologicalError, TopologicalOperation, TopologicalQubit, TopologicalQubitState,
        TopologicalResult, TopologicalSystemStatus, TopologicalSystemType,
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
        BaselineMetric, BaselineMetricValue, BenchmarkEvent, PerformanceBaseline,
        QuantumPlatform as UnifiedQuantumPlatform, UnifiedBenchmarkConfig, UnifiedBenchmarkResult,
        UnifiedQuantumBenchmarkSystem,
    };
    pub use crate::unified_error_handling::{
        ErrorCategory, ErrorSeverity, RecoveryStrategy, UnifiedDeviceError, UnifiedErrorContext,
        UnifiedErrorHandler, UnifiedRetryConfig,
    };
    pub use crate::vqa_support::{
        analysis::ConvergenceAnalysis,
        circuits::ParametricCircuit as VQAParametricCircuit,
        config::{
            AdaptiveShotConfig, ConvergenceCriterion as VQAConvergenceCriterion, GradientMethod,
            MultiStartConfig, OptimizationTrajectory, ResourceUtilization, VQAAlgorithmType,
            VQAConfig, VQAHardwareAnalysis, VQAHardwareConfig, VQANoiseMitigation,
            VQAOptimizationConfig, VQAOptimizer, VQAStatisticalAnalysis, VQAStatisticalConfig,
            VQAValidationConfig, VQAValidationResults, WarmRestartConfig,
        },
        executor::{VQAExecutor, VQAResult},
        objectives::ObjectiveFunction,
    };
    pub use crate::zero_noise_extrapolation::{
        CircuitFolder, ExtrapolationFitter, ExtrapolationMethod, NoiseScalingMethod, Observable,
        ZNECapable, ZNEConfig, ZNEExecutor, ZNEResult,
    };
}
