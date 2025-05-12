//! Quantum device connectors for the quantrs framework.
//!
//! This crate provides connectivity to quantum hardware providers like IBM Quantum,
//! Azure Quantum, and AWS Braket. It enables users to run quantum circuits on real
//! quantum hardware or cloud-based simulators.

use quantrs_circuit::prelude::Circuit;
use std::collections::HashMap;
use thiserror::Error;

pub mod aws;
pub mod aws_device;
pub mod azure;
pub mod azure_device;
pub mod ibm;
pub mod ibm_device;
pub mod transpiler;

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
#[derive(Debug, Clone)]
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
    pub use crate::CircuitExecutor;
    pub use crate::CircuitResult;
    pub use crate::DeviceError;
    pub use crate::DeviceResult;
    pub use crate::QuantumDevice;

    #[cfg(feature = "ibm")]
    pub use crate::ibm::*;
    #[cfg(feature = "ibm")]
    pub use crate::ibm_device::*;

    #[cfg(feature = "azure")]
    pub use crate::azure::*;
    #[cfg(feature = "azure")]
    pub use crate::azure_device::*;

    #[cfg(feature = "aws")]
    pub use crate::aws::*;
    #[cfg(feature = "aws")]
    pub use crate::aws_device::*;

    pub use crate::transpiler::*;
}
