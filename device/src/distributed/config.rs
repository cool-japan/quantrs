//! Configuration structures for distributed quantum orchestration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedOrchestratorConfig {
    pub network_config: NetworkConfig,
    pub computing_config: DistributedComputingConfig,
    pub load_balancing_config: LoadBalancingConfig,
    pub fault_tolerance_config: FaultToleranceConfig,
    pub security_config: SecurityConfig,
    pub optimization_config: DistributedOptimizationConfig,
    pub monitoring_config: DistributedMonitoringConfig,
    pub resource_config: DistributedResourceConfig,
}

impl Default for DistributedOrchestratorConfig {
    fn default() -> Self {
        Self {
            network_config: NetworkConfig::default(),
            computing_config: DistributedComputingConfig::default(),
            load_balancing_config: LoadBalancingConfig::default(),
            fault_tolerance_config: FaultToleranceConfig::default(),
            security_config: SecurityConfig::default(),
            optimization_config: DistributedOptimizationConfig::default(),
            monitoring_config: DistributedMonitoringConfig::default(),
            resource_config: DistributedResourceConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub topology: NetworkTopology,
    pub communication_protocol: CommunicationProtocol,
    pub max_connections: u32,
    pub connection_timeout: Duration,
    pub heartbeat_interval: Duration,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            topology: NetworkTopology::Mesh,
            communication_protocol: CommunicationProtocol::TCP,
            max_connections: 100,
            connection_timeout: Duration::from_secs(30),
            heartbeat_interval: Duration::from_secs(10),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkTopology {
    Star,
    Mesh,
    Ring,
    Tree,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationProtocol {
    TCP,
    UDP,
    QUIC,
    WebSocket,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedComputingConfig {
    pub workload_distribution_strategy: WorkloadDistributionStrategy,
    pub circuit_decomposition: bool,
    pub parallel_execution: bool,
    pub resource_estimation: bool,
}

impl Default for DistributedComputingConfig {
    fn default() -> Self {
        Self {
            workload_distribution_strategy: WorkloadDistributionStrategy::LoadBased,
            circuit_decomposition: true,
            parallel_execution: true,
            resource_estimation: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkloadDistributionStrategy {
    RoundRobin,
    LoadBased,
    CapabilityBased,
    Geographic,
    CostOptimized,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    pub algorithm: LoadBalancingAlgorithm,
    pub rebalancing_threshold: f64,
    pub monitoring_interval: Duration,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            algorithm: LoadBalancingAlgorithm::WeightedRoundRobin,
            rebalancing_threshold: 0.8,
            monitoring_interval: Duration::from_secs(5),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    ResourceBased,
    AdaptiveWeighted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    pub failure_detection: FailureDetectionConfig,
    pub replication_strategy: ReplicationStrategy,
    pub checkpoint_config: CheckpointConfig,
    pub recovery_timeout: Duration,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            failure_detection: FailureDetectionConfig::default(),
            replication_strategy: ReplicationStrategy::ActivePassive,
            checkpoint_config: CheckpointConfig::default(),
            recovery_timeout: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureDetectionConfig {
    pub heartbeat_timeout: Duration,
    pub max_retries: u32,
    pub detection_interval: Duration,
}

impl Default for FailureDetectionConfig {
    fn default() -> Self {
        Self {
            heartbeat_timeout: Duration::from_secs(30),
            max_retries: 3,
            detection_interval: Duration::from_secs(5),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationStrategy {
    NoReplication,
    ActivePassive,
    ActiveActive,
    ChainReplication,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub checkpoint_interval: Duration,
    pub checkpoint_storage: String,
    pub max_checkpoints: u32,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval: Duration::from_secs(300),
            checkpoint_storage: "local".to_string(),
            max_checkpoints: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub authentication_method: AuthenticationMethod,
    pub authorization_model: AuthorizationModel,
    pub encryption_algorithm: EncryptionAlgorithm,
    pub key_rotation_interval: Duration,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            authentication_method: AuthenticationMethod::TokenBased,
            authorization_model: AuthorizationModel::RBAC,
            encryption_algorithm: EncryptionAlgorithm::AES256,
            key_rotation_interval: Duration::from_secs(3600),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthenticationMethod {
    TokenBased,
    CertificateBased,
    Biometric,
    MultiFactorAuthentication,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthorizationModel {
    RBAC,
    ABAC,
    DAC,
    MAC,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    AES256,
    ChaCha20,
    PostQuantumKyber,
    HybridClassicalQuantum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedOptimizationConfig {
    pub objective: OptimizationObjective,
    pub algorithm: String,
    pub convergence_threshold: f64,
    pub max_iterations: u32,
}

impl Default for DistributedOptimizationConfig {
    fn default() -> Self {
        Self {
            objective: OptimizationObjective::Latency,
            algorithm: "genetic".to_string(),
            convergence_threshold: 0.01,
            max_iterations: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    Latency,
    Throughput,
    Cost,
    EnergyEfficiency,
    MultiObjective,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedMonitoringConfig {
    pub performance_monitoring: bool,
    pub resource_monitoring: bool,
    pub security_monitoring: bool,
    pub monitoring_interval: Duration,
}

impl Default for DistributedMonitoringConfig {
    fn default() -> Self {
        Self {
            performance_monitoring: true,
            resource_monitoring: true,
            security_monitoring: true,
            monitoring_interval: Duration::from_secs(10),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedResourceConfig {
    pub cpu_allocation: f64,
    pub memory_allocation: f64,
    pub network_bandwidth: f64,
    pub quantum_resource_allocation: f64,
}

impl Default for DistributedResourceConfig {
    fn default() -> Self {
        Self {
            cpu_allocation: 0.8,
            memory_allocation: 0.8,
            network_bandwidth: 0.8,
            quantum_resource_allocation: 0.9,
        }
    }
}
