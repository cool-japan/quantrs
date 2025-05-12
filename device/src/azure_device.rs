#[cfg(feature = "azure")]
use serde_json;
use std::collections::HashMap;
#[cfg(feature = "azure")]
use std::sync::Arc;
#[cfg(feature = "azure")]
use std::time::Duration;

#[cfg(feature = "azure")]
use async_trait::async_trait;
#[cfg(feature = "azure")]
use quantrs_circuit::prelude::Circuit;
#[cfg(feature = "azure")]
use tokio::sync::RwLock;

#[cfg(feature = "azure")]
use crate::azure::{AzureCircuitConfig, AzureQuantumClient, AzureTarget};
use crate::DeviceError;
use crate::DeviceResult;
#[cfg(feature = "azure")]
use crate::{CircuitExecutor, CircuitResult, QuantumDevice};

/// Configuration for an Azure Quantum device
#[derive(Debug, Clone)]
pub struct AzureDeviceConfig {
    /// Provider ID (e.g., "ionq", "microsoft", "quantinuum")
    pub provider_id: String,
    /// Number of shots to run for each circuit
    pub default_shots: usize,
    /// Provider-specific parameters
    #[cfg(feature = "azure")]
    pub provider_parameters: HashMap<String, serde_json::Value>,
    #[cfg(not(feature = "azure"))]
    pub provider_parameters: HashMap<String, String>,
    /// Timeout for job completion (in seconds)
    pub timeout_secs: Option<u64>,
}

impl Default for AzureDeviceConfig {
    fn default() -> Self {
        Self {
            provider_id: "microsoft".to_string(),
            default_shots: 1000,
            #[cfg(feature = "azure")]
            provider_parameters: HashMap::new(),
            #[cfg(not(feature = "azure"))]
            provider_parameters: HashMap::new(),
            timeout_secs: None,
        }
    }
}

/// Azure Quantum device implementation
#[cfg(feature = "azure")]
pub struct AzureQuantumDevice {
    /// Azure Quantum client
    client: Arc<AzureQuantumClient>,
    /// Target device ID
    target_id: String,
    /// Provider ID
    provider_id: String,
    /// Configuration
    config: AzureDeviceConfig,
    /// Cached target information
    target_cache: Arc<RwLock<Option<AzureTarget>>>,
}

#[cfg(feature = "azure")]
impl AzureQuantumDevice {
    /// Create a new Azure Quantum device instance
    pub async fn new(
        client: AzureQuantumClient,
        target_id: &str,
        provider_id: Option<&str>,
        config: Option<AzureDeviceConfig>,
    ) -> DeviceResult<Self> {
        let client = Arc::new(client);
        let target_cache = Arc::new(RwLock::new(None));

        // Get target details to validate and get provider if not specified
        let target = client.get_target(target_id).await?;
        let provider_id = match provider_id {
            Some(id) => id.to_string(),
            None => target.provider_id.clone(),
        };

        // Create and cache the target
        let mut cache = target_cache.write().await;
        *cache = Some(target);

        let config = config.unwrap_or_default();

        Ok(Self {
            client,
            target_id: target_id.to_string(),
            provider_id,
            config,
            target_cache,
        })
    }

    /// Get cached target information, fetching if necessary
    async fn get_target(&self) -> DeviceResult<AzureTarget> {
        let cache = self.target_cache.read().await;

        if let Some(target) = cache.clone() {
            return Ok(target);
        }

        // Cache miss, need to fetch
        drop(cache);
        let target = self.client.get_target(&self.target_id).await?;

        let mut cache = self.target_cache.write().await;
        *cache = Some(target.clone());

        Ok(target)
    }
}

#[cfg(feature = "azure")]
#[async_trait]
impl QuantumDevice for AzureQuantumDevice {
    async fn is_available(&self) -> DeviceResult<bool> {
        let target = self.get_target().await?;
        Ok(target.status == "Available")
    }

    async fn qubit_count(&self) -> DeviceResult<usize> {
        let target = self.get_target().await?;
        Ok(target.num_qubits)
    }

    async fn properties(&self) -> DeviceResult<HashMap<String, String>> {
        let target = self.get_target().await?;

        // Convert complex JSON properties to string representation
        let mut properties = HashMap::new();
        for (key, value) in target.properties {
            properties.insert(key, value.to_string());
        }

        Ok(properties)
    }

    async fn is_simulator(&self) -> DeviceResult<bool> {
        let target = self.get_target().await?;
        Ok(target.is_simulator)
    }
}

#[cfg(feature = "azure")]
#[async_trait]
impl CircuitExecutor for AzureQuantumDevice {
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

        // Convert circuit to provider-specific format
        let circuit_str =
            AzureQuantumClient::circuit_to_provider_format(circuit, &self.provider_id)?;

        // Create config
        let job_name = format!("quantrs_job_{}", chrono::Utc::now().timestamp());
        let config = AzureCircuitConfig {
            name: job_name,
            circuit: circuit_str,
            shots: shots.max(1), // Ensure at least 1 shot
            provider_parameters: self.config.provider_parameters.clone(),
        };

        // Submit job
        let job_id = self
            .client
            .submit_circuit(&self.target_id, &self.provider_id, config)
            .await?;

        // Wait for completion
        let result = self
            .client
            .wait_for_job(&job_id, self.config.timeout_secs)
            .await?;

        // Convert result to CircuitResult
        let mut counts = HashMap::new();
        for (bitstring, probability) in result.histogram {
            // Convert probabilities to counts based on shots
            let count = (probability * shots as f64).round() as usize;
            counts.insert(bitstring, count);
        }

        let mut metadata = HashMap::new();
        metadata.insert("job_id".to_string(), job_id);
        metadata.insert("provider".to_string(), self.provider_id.clone());
        metadata.insert("target".to_string(), self.target_id.clone());

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
            let circuit_str =
                AzureQuantumClient::circuit_to_provider_format(circuit, &self.provider_id)?;
            let job_name = format!(
                "quantrs_batch_{}_job_{}",
                chrono::Utc::now().timestamp(),
                idx
            );

            let config = AzureCircuitConfig {
                name: job_name,
                circuit: circuit_str,
                shots: shots.max(1), // Ensure at least 1 shot
                provider_parameters: self.config.provider_parameters.clone(),
            };

            configs.push(config);
        }

        // Submit all circuits in parallel
        let job_ids = self
            .client
            .submit_circuits_parallel(&self.target_id, &self.provider_id, configs)
            .await?;

        // Wait for all jobs to complete and collect results
        let mut results = Vec::with_capacity(job_ids.len());
        for job_id in job_ids {
            let result = self
                .client
                .wait_for_job(&job_id, self.config.timeout_secs)
                .await?;

            let mut counts = HashMap::new();
            for (bitstring, probability) in result.histogram {
                // Convert probabilities to counts based on shots
                let count = (probability * shots as f64).round() as usize;
                counts.insert(bitstring, count);
            }

            let mut metadata = HashMap::new();
            metadata.insert("job_id".to_string(), job_id);
            metadata.insert("provider".to_string(), self.provider_id.clone());
            metadata.insert("target".to_string(), self.target_id.clone());

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

        // Check if the circuit can be converted to the provider's format
        // This is just a basic check, more sophisticated validation would be done
        // when actually converting the circuit
        match AzureQuantumClient::circuit_to_provider_format(circuit, &self.provider_id) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn estimated_queue_time<const N: usize>(
        &self,
        _circuit: &Circuit<N>,
    ) -> DeviceResult<Duration> {
        // Azure Quantum doesn't provide queue time estimates in the API
        // Return a conservative estimate based on device type
        let is_sim = self.is_simulator().await?;

        if is_sim {
            // Simulators tend to have shorter queue times
            Ok(Duration::from_secs(60)) // 1 minute
        } else {
            // Hardware devices tend to have longer queue times
            Ok(Duration::from_secs(300)) // 5 minutes
        }
    }
}

#[cfg(not(feature = "azure"))]
pub struct AzureQuantumDevice;

#[cfg(not(feature = "azure"))]
impl AzureQuantumDevice {
    pub async fn new(
        _client: crate::azure::AzureQuantumClient,
        _target_id: &str,
        _provider_id: Option<&str>,
        _config: Option<AzureDeviceConfig>,
    ) -> DeviceResult<Self> {
        Err(DeviceError::UnsupportedDevice(
            "Azure Quantum support not enabled. Recompile with the 'azure' feature.".to_string(),
        ))
    }
}
