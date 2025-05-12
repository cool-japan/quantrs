#[cfg(feature = "aws")]
use serde_json;
#[cfg(feature = "aws")]
use std::collections::HashMap;
#[cfg(feature = "aws")]
use std::sync::Arc;
#[cfg(feature = "aws")]
use std::time::Duration;

#[cfg(feature = "aws")]
use async_trait::async_trait;
#[cfg(feature = "aws")]
use tokio::sync::RwLock;

#[cfg(feature = "aws")]
use crate::aws::{AWSBraketClient, AWSCircuitConfig, AWSDevice, AWSTaskStatus};
use crate::DeviceError;
use crate::DeviceResult;
#[cfg(feature = "aws")]
use crate::{CircuitExecutor, CircuitResult, QuantumDevice};
#[cfg(feature = "aws")]
use quantrs_circuit::prelude::Circuit;

/// Configuration for an AWS Braket device
#[derive(Debug, Clone)]
pub struct AWSDeviceConfig {
    /// Number of shots to run for each circuit
    pub default_shots: usize,
    /// IR type to use (OPENQASM or BRAKET)
    pub ir_type: String,
    /// Device-specific parameters
    #[cfg(feature = "aws")]
    pub device_parameters: Option<serde_json::Value>,
    #[cfg(not(feature = "aws"))]
    pub device_parameters: Option<()>,
    /// Timeout for task completion (in seconds)
    pub timeout_secs: Option<u64>,
}

impl Default for AWSDeviceConfig {
    fn default() -> Self {
        Self {
            default_shots: 1000,
            ir_type: "BRAKET".to_string(),
            device_parameters: None,
            timeout_secs: None,
        }
    }
}

/// AWS Braket device implementation
#[cfg(feature = "aws")]
pub struct AWSBraketDevice {
    /// AWS Braket client
    client: Arc<AWSBraketClient>,
    /// Device ARN
    device_arn: String,
    /// Configuration
    config: AWSDeviceConfig,
    /// Cached device information
    device_cache: Arc<RwLock<Option<AWSDevice>>>,
}

#[cfg(feature = "aws")]
impl AWSBraketDevice {
    /// Create a new AWS Braket device instance
    pub async fn new(
        client: AWSBraketClient,
        device_arn: &str,
        config: Option<AWSDeviceConfig>,
    ) -> DeviceResult<Self> {
        let client = Arc::new(client);
        let device_cache = Arc::new(RwLock::new(None));

        // Get device details to validate
        let device = client.get_device(device_arn).await?;

        // Create and cache the device
        let mut cache = device_cache.write().await;
        *cache = Some(device);

        let config = config.unwrap_or_default();

        Ok(Self {
            client,
            device_arn: device_arn.to_string(),
            config,
            device_cache,
        })
    }

    /// Get cached device information, fetching if necessary
    async fn get_device(&self) -> DeviceResult<AWSDevice> {
        let cache = self.device_cache.read().await;

        if let Some(device) = cache.clone() {
            return Ok(device);
        }

        // Cache miss, need to fetch
        drop(cache);
        let device = self.client.get_device(&self.device_arn).await?;

        let mut cache = self.device_cache.write().await;
        *cache = Some(device.clone());

        Ok(device)
    }
}

#[cfg(feature = "aws")]
#[async_trait]
impl QuantumDevice for AWSBraketDevice {
    async fn is_available(&self) -> DeviceResult<bool> {
        let device = self.get_device().await?;
        Ok(device.status == "ONLINE")
    }

    async fn qubit_count(&self) -> DeviceResult<usize> {
        let device = self.get_device().await?;
        Ok(device.num_qubits)
    }

    async fn properties(&self) -> DeviceResult<HashMap<String, String>> {
        let device = self.get_device().await?;

        // Convert complex JSON capabilities to string representation
        let mut properties = HashMap::new();

        #[cfg(feature = "aws")]
        {
            if let serde_json::Value::Object(caps) = &device.device_capabilities {
                for (key, value) in caps {
                    properties.insert(key.clone(), value.to_string());
                }
            }
        }

        Ok(properties)
    }

    async fn is_simulator(&self) -> DeviceResult<bool> {
        let device = self.get_device().await?;
        Ok(device.device_type == "SIMULATOR")
    }
}

#[cfg(feature = "aws")]
#[async_trait]
impl CircuitExecutor for AWSBraketDevice {
    async fn execute_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        shots: usize,
    ) -> DeviceResult<CircuitResult> {
        // Check if circuit can be executed
        if !self.can_execute_circuit(circuit).await? {
            return Err(DeviceError::CircuitConversion(
                "Circuit cannot be executed on this device".to_string(),
            ));
        }

        // Convert circuit to the appropriate IR format
        let circuit_str = match self.config.ir_type.as_str() {
            "OPENQASM" => AWSBraketClient::circuit_to_qasm(circuit)?,
            "BRAKET" => AWSBraketClient::circuit_to_braket_ir(circuit)?,
            _ => {
                return Err(DeviceError::CircuitConversion(format!(
                    "Unsupported IR type: {}",
                    self.config.ir_type
                )))
            }
        };

        // Create task config
        let job_name = format!("quantrs_task_{}", chrono::Utc::now().timestamp());

        let s3_bucket = "amazon-braket-examples"; // This would be the client's S3 bucket in reality
        let s3_key_prefix = format!("quantrs-tasks/{}", job_name);

        let config = AWSCircuitConfig {
            name: job_name,
            ir: circuit_str,
            ir_type: self.config.ir_type.clone(),
            shots: shots.max(1), // Ensure at least 1 shot
            s3_bucket: s3_bucket.to_string(),
            s3_key_prefix,
            device_parameters: self.config.device_parameters.clone(),
        };

        // Submit task
        let task_arn = self.client.submit_circuit(&self.device_arn, config).await?;

        // Wait for completion
        let result = self
            .client
            .wait_for_task(&task_arn, self.config.timeout_secs)
            .await?;

        // Convert result to CircuitResult
        let mut counts = HashMap::new();
        for (bitstring, count) in result.measurements {
            counts.insert(bitstring, count);
        }

        let mut metadata = HashMap::new();
        metadata.insert("task_arn".to_string(), task_arn);
        metadata.insert("device_arn".to_string(), self.device_arn.clone());

        Ok(CircuitResult {
            counts,
            shots,
            metadata,
        })
    }

    async fn execute_circuits<const N: usize>(
        &self,
        circuits: Vec<&Circuit<N>>,
        shots: usize,
    ) -> DeviceResult<Vec<CircuitResult>> {
        let mut configs = Vec::with_capacity(circuits.len());

        // Prepare all circuit configs
        for (idx, circuit) in circuits.iter().enumerate() {
            // Convert circuit to the appropriate IR format
            let circuit_str = match self.config.ir_type.as_str() {
                "OPENQASM" => AWSBraketClient::circuit_to_qasm(circuit)?,
                "BRAKET" => AWSBraketClient::circuit_to_braket_ir(circuit)?,
                _ => {
                    return Err(DeviceError::CircuitConversion(format!(
                        "Unsupported IR type: {}",
                        self.config.ir_type
                    )))
                }
            };

            let job_name = format!(
                "quantrs_batch_{}_task_{}",
                chrono::Utc::now().timestamp(),
                idx
            );
            let s3_bucket = "amazon-braket-examples"; // This would be the client's S3 bucket in reality
            let s3_key_prefix = format!("quantrs-tasks/{}", job_name);

            let config = AWSCircuitConfig {
                name: job_name,
                ir: circuit_str,
                ir_type: self.config.ir_type.clone(),
                shots: shots.max(1), // Ensure at least 1 shot
                s3_bucket: s3_bucket.to_string(),
                s3_key_prefix,
                device_parameters: self.config.device_parameters.clone(),
            };

            configs.push(config);
        }

        // Submit all circuits in parallel
        let task_arns = self
            .client
            .submit_circuits_parallel(&self.device_arn, configs)
            .await?;

        // Wait for all tasks to complete and collect results
        let mut results = Vec::with_capacity(task_arns.len());
        for task_arn in task_arns {
            let result = self
                .client
                .wait_for_task(&task_arn, self.config.timeout_secs)
                .await?;

            let mut counts = HashMap::new();
            for (bitstring, count) in result.measurements {
                counts.insert(bitstring, count);
            }

            let mut metadata = HashMap::new();
            metadata.insert("task_arn".to_string(), task_arn);
            metadata.insert("device_arn".to_string(), self.device_arn.clone());

            results.push(CircuitResult {
                counts,
                shots,
                metadata,
            });
        }

        Ok(results)
    }

    async fn can_execute_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> DeviceResult<bool> {
        // Get device qubit count
        let device_qubits = self.qubit_count().await?;

        // Check if circuit qubit count exceeds device qubit count
        if N > device_qubits {
            return Ok(false);
        }

        // Check if the circuit can be converted to the specified IR format
        match self.config.ir_type.as_str() {
            "OPENQASM" => match AWSBraketClient::circuit_to_qasm(circuit) {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            },
            "BRAKET" => match AWSBraketClient::circuit_to_braket_ir(circuit) {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            },
            _ => Ok(false),
        }
    }

    async fn estimated_queue_time<const N: usize>(
        &self,
        _circuit: &Circuit<N>,
    ) -> DeviceResult<Duration> {
        // AWS Braket doesn't provide queue time estimates in the API
        // Return a conservative estimate based on device type
        let is_sim = self.is_simulator().await?;

        if is_sim {
            // Simulators tend to have shorter queue times
            Ok(Duration::from_secs(30)) // 30 seconds
        } else {
            // Hardware devices tend to have longer queue times
            Ok(Duration::from_secs(600)) // 10 minutes
        }
    }
}

#[cfg(not(feature = "aws"))]
pub struct AWSBraketDevice;

#[cfg(not(feature = "aws"))]
impl AWSBraketDevice {
    pub async fn new(
        _client: crate::aws::AWSBraketClient,
        _device_arn: &str,
        _config: Option<AWSDeviceConfig>,
    ) -> DeviceResult<Self> {
        Err(DeviceError::UnsupportedDevice(
            "AWS Braket support not enabled. Recompile with the 'aws' feature.".to_string(),
        ))
    }
}
