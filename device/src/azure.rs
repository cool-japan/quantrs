//! Azure Quantum cloud backend connector.
//!
//! Implements job submission and result retrieval against the Azure Quantum
//! REST API.  Requires the `azure` Cargo feature and valid Azure credentials.

use quantrs2_circuit::prelude::Circuit;
use std::collections::HashMap;
#[cfg(feature = "azure")]
use std::sync::Arc;
#[cfg(feature = "azure")]
use std::thread::sleep;
#[cfg(feature = "azure")]
use std::time::Duration;

#[cfg(feature = "azure")]
use reqwest::{header, Client};
#[cfg(feature = "azure")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "azure")]
use serde_json;
use thiserror::Error;

use crate::DeviceError;
use crate::DeviceResult;

#[cfg(feature = "azure")]
const AZURE_QUANTUM_API_URL: &str = "https://eastus.quantum.azure.com";
#[cfg(feature = "azure")]
const DEFAULT_TIMEOUT_SECS: u64 = 90;

/// Represents the available providers on Azure Quantum
#[derive(Debug, Clone)]
#[cfg_attr(feature = "azure", derive(serde::Deserialize))]
pub struct AzureProvider {
    /// Unique identifier for the provider
    pub id: String,
    /// Name of the provider (e.g., "ionq", "microsoft", "quantinuum")
    pub name: String,
    /// Provider-specific capabilities and settings
    pub capabilities: HashMap<String, String>,
}

/// Represents the available target devices on Azure Quantum
#[derive(Debug, Clone)]
#[cfg_attr(feature = "azure", derive(serde::Deserialize))]
pub struct AzureTarget {
    /// Target ID
    pub id: String,
    /// Display name of the target
    pub name: String,
    /// Provider ID
    pub provider_id: String,
    /// Whether the target is a simulator or real quantum hardware
    pub is_simulator: bool,
    /// Number of qubits on the target
    pub num_qubits: usize,
    /// Status of the target (e.g., "Available", "Offline")
    pub status: String,
    /// Target-specific capabilities and properties
    #[cfg(feature = "azure")]
    pub properties: HashMap<String, serde_json::Value>,
    #[cfg(not(feature = "azure"))]
    pub properties: HashMap<String, String>,
}

/// Configuration for a quantum circuit to be submitted to Azure Quantum
#[derive(Debug, Clone)]
#[cfg_attr(feature = "azure", derive(Serialize))]
pub struct AzureCircuitConfig {
    /// Name of the job
    pub name: String,
    /// Circuit representation (varies by provider)
    pub circuit: String,
    /// Number of shots to run
    pub shots: usize,
    /// Provider-specific parameters
    #[cfg(feature = "azure")]
    pub provider_parameters: HashMap<String, serde_json::Value>,
    #[cfg(not(feature = "azure"))]
    pub provider_parameters: HashMap<String, String>,
}

/// Status of a job in Azure Quantum
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "azure", derive(Deserialize))]
pub enum AzureJobStatus {
    #[cfg_attr(feature = "azure", serde(rename = "Waiting"))]
    Waiting,
    #[cfg_attr(feature = "azure", serde(rename = "Executing"))]
    Executing,
    #[cfg_attr(feature = "azure", serde(rename = "Succeeded"))]
    Succeeded,
    #[cfg_attr(feature = "azure", serde(rename = "Failed"))]
    Failed,
    #[cfg_attr(feature = "azure", serde(rename = "Cancelled"))]
    Cancelled,
}

/// Response from submitting a job to Azure Quantum
#[cfg(feature = "azure")]
#[derive(Debug, Deserialize)]
pub struct AzureJobResponse {
    /// Job ID
    pub id: String,
    /// Name of the job
    pub name: String,
    /// Job status
    pub status: AzureJobStatus,
    /// Provider ID
    pub provider: String,
    /// Target ID
    pub target: String,
    /// Creation timestamp
    pub creation_time: String,
    /// Execution time (if completed)
    pub execution_time: Option<String>,
}

#[cfg(not(feature = "azure"))]
#[derive(Debug)]
pub struct AzureJobResponse {
    /// Job ID
    pub id: String,
    /// Name of the job
    pub name: String,
    /// Job status
    pub status: AzureJobStatus,
}

/// Results from a completed job
#[cfg(feature = "azure")]
#[derive(Debug, Deserialize)]
pub struct AzureJobResult {
    /// Counts of each basis state
    pub histogram: HashMap<String, f64>,
    /// Total number of shots executed
    pub shots: usize,
    /// Job status
    pub status: AzureJobStatus,
    /// Error message, if any
    pub error: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

#[cfg(not(feature = "azure"))]
#[derive(Debug)]
pub struct AzureJobResult {
    /// Counts of each basis state (as probabilities)
    pub histogram: HashMap<String, f64>,
    /// Total number of shots executed
    pub shots: usize,
    /// Job status
    pub status: AzureJobStatus,
    /// Error message, if any
    pub error: Option<String>,
}

/// Errors specific to Azure Quantum
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum AzureQuantumError {
    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("API error: {0}")]
    API(String),

    #[error("Target not available: {0}")]
    TargetUnavailable(String),

    #[error("Circuit conversion error: {0}")]
    CircuitConversion(String),

    #[error("Job submission error: {0}")]
    JobSubmission(String),

    #[error("Timeout waiting for job completion")]
    Timeout,
}

/// Client for interacting with Azure Quantum
#[cfg(feature = "azure")]
#[derive(Clone)]
pub struct AzureQuantumClient {
    /// HTTP client for making API requests
    client: Client,
    /// Base URL for the Azure Quantum API
    api_url: String,
    /// Workspace name
    workspace: String,
    /// Subscription ID
    subscription_id: String,
    /// Resource group
    resource_group: String,
    /// Authentication token
    token: String,
}

#[cfg(not(feature = "azure"))]
#[derive(Clone)]
pub struct AzureQuantumClient;

#[cfg(feature = "azure")]
impl AzureQuantumClient {
    /// Create a new Azure Quantum client with the given token and workspace details
    pub fn new(
        token: &str,
        subscription_id: &str,
        resource_group: &str,
        workspace: &str,
        region: Option<&str>,
    ) -> DeviceResult<Self> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );

        let client = Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| DeviceError::Connection(e.to_string()))?;

        let api_url = match region {
            Some(region) => format!("https://{}.quantum.azure.com", region),
            None => AZURE_QUANTUM_API_URL.to_string(),
        };

        Ok(Self {
            client,
            api_url,
            workspace: workspace.to_string(),
            subscription_id: subscription_id.to_string(),
            resource_group: resource_group.to_string(),
            token: token.to_string(),
        })
    }

    /// Get API base path for this workspace
    fn get_api_base_path(&self) -> String {
        format!(
            "/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Quantum/Workspaces/{}",
            self.subscription_id, self.resource_group, self.workspace
        )
    }

    /// List all available providers
    pub async fn list_providers(&self) -> DeviceResult<Vec<AzureProvider>> {
        let base_path = self.get_api_base_path();
        let url = format!("{}{}/providers", self.api_url, base_path);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .send()
            .await
            .map_err(|e| DeviceError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(DeviceError::APIError(error_msg));
        }

        let providers: Vec<AzureProvider> = response
            .json()
            .await
            .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

        Ok(providers)
    }

    /// List all available targets (devices and simulators)
    pub async fn list_targets(&self) -> DeviceResult<Vec<AzureTarget>> {
        let base_path = self.get_api_base_path();
        let url = format!("{}{}/targets", self.api_url, base_path);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .send()
            .await
            .map_err(|e| DeviceError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(DeviceError::APIError(error_msg));
        }

        let targets: Vec<AzureTarget> = response
            .json()
            .await
            .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

        Ok(targets)
    }

    /// Get details about a specific target
    pub async fn get_target(&self, target_id: &str) -> DeviceResult<AzureTarget> {
        let base_path = self.get_api_base_path();
        let url = format!("{}{}/targets/{}", self.api_url, base_path, target_id);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .send()
            .await
            .map_err(|e| DeviceError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(DeviceError::APIError(error_msg));
        }

        let target: AzureTarget = response
            .json()
            .await
            .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

        Ok(target)
    }

    /// Submit a circuit to be executed on an Azure Quantum target
    pub async fn submit_circuit(
        &self,
        target_id: &str,
        provider_id: &str,
        config: AzureCircuitConfig,
    ) -> DeviceResult<String> {
        let base_path = self.get_api_base_path();
        let url = format!("{}{}/jobs", self.api_url, base_path);

        use serde_json::json;

        let payload = json!({
            "name": config.name,
            "providerId": provider_id,
            "target": target_id,
            "input": config.circuit,
            "inputDataFormat": "qir", // Default to QIR, change based on provider
            "outputDataFormat": "microsoft.quantum-results.v1",
            "metadata": {
                "shots": config.shots
            },
            "params": config.provider_parameters
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .json(&payload)
            .send()
            .await
            .map_err(|e| DeviceError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(DeviceError::JobSubmission(error_msg));
        }

        let job_response: AzureJobResponse = response
            .json()
            .await
            .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

        Ok(job_response.id)
    }

    /// Get the status of a job
    pub async fn get_job_status(&self, job_id: &str) -> DeviceResult<AzureJobStatus> {
        let base_path = self.get_api_base_path();
        let url = format!("{}{}/jobs/{}", self.api_url, base_path, job_id);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .send()
            .await
            .map_err(|e| DeviceError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(DeviceError::APIError(error_msg));
        }

        let job: AzureJobResponse = response
            .json()
            .await
            .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

        Ok(job.status)
    }

    /// Get the results of a completed job
    pub async fn get_job_result(&self, job_id: &str) -> DeviceResult<AzureJobResult> {
        let base_path = self.get_api_base_path();
        let url = format!("{}{}/jobs/{}/results", self.api_url, base_path, job_id);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .send()
            .await
            .map_err(|e| DeviceError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(DeviceError::APIError(error_msg));
        }

        let result: AzureJobResult = response
            .json()
            .await
            .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

        Ok(result)
    }

    /// Wait for a job to complete with timeout
    pub async fn wait_for_job(
        &self,
        job_id: &str,
        timeout_secs: Option<u64>,
    ) -> DeviceResult<AzureJobResult> {
        let timeout = timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS);
        let mut elapsed = 0;
        let interval = 5; // Check status every 5 seconds

        while elapsed < timeout {
            let status = self.get_job_status(job_id).await?;

            match status {
                AzureJobStatus::Succeeded => {
                    return self.get_job_result(job_id).await;
                }
                AzureJobStatus::Failed => {
                    return Err(DeviceError::JobExecution(format!(
                        "Job {} encountered an error",
                        job_id
                    )));
                }
                AzureJobStatus::Cancelled => {
                    return Err(DeviceError::JobExecution(format!(
                        "Job {} was cancelled",
                        job_id
                    )));
                }
                _ => {
                    // Still in progress, wait and check again
                    sleep(Duration::from_secs(interval));
                    elapsed += interval;
                }
            }
        }

        Err(DeviceError::Timeout(format!(
            "Timed out waiting for job {} to complete",
            job_id
        )))
    }

    /// Submit multiple circuits in parallel
    pub async fn submit_circuits_parallel(
        &self,
        target_id: &str,
        provider_id: &str,
        configs: Vec<AzureCircuitConfig>,
    ) -> DeviceResult<Vec<String>> {
        let client = Arc::new(self.clone());

        let mut handles = vec![];

        for config in configs {
            let client_clone = client.clone();
            let target_id = target_id.to_string();
            let provider_id = provider_id.to_string();

            let handle = tokio::task::spawn(async move {
                client_clone
                    .submit_circuit(&target_id, &provider_id, config)
                    .await
            });

            handles.push(handle);
        }

        let mut job_ids = vec![];

        for handle in handles {
            match handle.await {
                Ok(result) => match result {
                    Ok(job_id) => job_ids.push(job_id),
                    Err(e) => return Err(e),
                },
                Err(e) => {
                    return Err(DeviceError::JobSubmission(format!(
                        "Failed to join task: {}",
                        e
                    )));
                }
            }
        }

        Ok(job_ids)
    }

    /// Convert a Quantrs circuit to a provider-specific format
    pub fn circuit_to_provider_format<const N: usize>(
        circuit: &Circuit<N>,
        provider_id: &str,
    ) -> DeviceResult<String> {
        // Different format conversions based on provider
        match provider_id {
            "ionq" => Self::circuit_to_ionq_format(circuit),
            "microsoft" => Self::circuit_to_qir_format(circuit),
            "quantinuum" => Self::circuit_to_qasm_format(circuit),
            _ => Err(DeviceError::CircuitConversion(format!(
                "Unsupported provider: {}",
                provider_id
            ))),
        }
    }

    // IonQ specific circuit format conversion.
    //
    // IonQ accepts a JSON circuit description where each gate is an object such
    // as `{"gate": "h", "target": 0}` or `{"gate": "rx", "target": 0,
    // "rotation": 1.57}` and two-qubit gates carry `control`/`target` (or two
    // `targets`). This iterates the circuit gates and emits one JSON object per
    // gate, returning an honest error for any gate IonQ cannot express directly.
    fn circuit_to_ionq_format<const N: usize>(circuit: &Circuit<N>) -> DeviceResult<String> {
        use quantrs2_core::gate::{multi::*, single::*, GateOp};
        use serde_json::json;

        let mut gates: Vec<serde_json::Value> = Vec::new();

        for gate in circuit.gates() {
            let any = gate.as_any();
            let qubits = gate.qubits();
            let q = |i: usize| qubits[i].id() as usize;

            // Parameterized single-qubit rotations (IonQ uses `rotation`).
            if let Some(g) = any.downcast_ref::<RotationX>() {
                gates.push(json!({ "gate": "rx", "target": q(0), "rotation": g.theta }));
                continue;
            }
            if let Some(g) = any.downcast_ref::<RotationY>() {
                gates.push(json!({ "gate": "ry", "target": q(0), "rotation": g.theta }));
                continue;
            }
            if let Some(g) = any.downcast_ref::<RotationZ>() {
                gates.push(json!({ "gate": "rz", "target": q(0), "rotation": g.theta }));
                continue;
            }

            let value = match gate.name() {
                "H" => json!({ "gate": "h", "target": q(0) }),
                "X" => json!({ "gate": "x", "target": q(0) }),
                "Y" => json!({ "gate": "y", "target": q(0) }),
                "Z" => json!({ "gate": "z", "target": q(0) }),
                "S" => json!({ "gate": "s", "target": q(0) }),
                "S†" => json!({ "gate": "si", "target": q(0) }),
                "T" => json!({ "gate": "t", "target": q(0) }),
                "T†" => json!({ "gate": "ti", "target": q(0) }),
                "√X" => json!({ "gate": "v", "target": q(0) }),
                "√X†" => json!({ "gate": "vi", "target": q(0) }),
                "CNOT" => json!({ "gate": "cnot", "control": q(0), "target": q(1) }),
                "SWAP" => json!({ "gate": "swap", "targets": [q(0), q(1)] }),
                other => {
                    return Err(DeviceError::CircuitConversion(format!(
                        "Gate '{other}' has no direct IonQ JSON representation; \
                         transpile to IonQ's native gate set first"
                    )));
                }
            };
            gates.push(value);
        }

        let ionq_circuit = json!({
            "qubits": N,
            "circuit": gates,
        });

        Ok(ionq_circuit.to_string())
    }

    // Microsoft QIR format conversion
    fn circuit_to_qir_format<const N: usize>(_circuit: &Circuit<N>) -> DeviceResult<String> {
        // QIR is a LLVM IR based format
        // For now, this is just a placeholder
        Err(DeviceError::CircuitConversion(
            "QIR conversion not yet implemented".to_string(),
        ))
    }

    // QASM format conversion for Quantinuum
    //
    // Quantinuum targets accept OpenQASM 2.0. The gate-by-gate translation is
    // delegated to the circuit crate's validated OpenQASM 2.0 exporter, which
    // emits the `OPENQASM 2.0` header, the `qelib1.inc` include, the quantum
    // register, and one line per gate in the circuit.
    fn circuit_to_qasm_format<const N: usize>(circuit: &Circuit<N>) -> DeviceResult<String> {
        quantrs2_circuit::qasm::circuit_to_qasm(circuit).map_err(|e| {
            DeviceError::CircuitConversion(format!(
                "Failed to convert circuit to OpenQASM 2.0 for Quantinuum: {e}"
            ))
        })
    }
}

#[cfg(not(feature = "azure"))]
impl AzureQuantumClient {
    pub fn new(
        _token: &str,
        _subscription_id: &str,
        _resource_group: &str,
        _workspace: &str,
        _region: Option<&str>,
    ) -> DeviceResult<Self> {
        Err(DeviceError::UnsupportedDevice(
            "Azure Quantum support not enabled. Recompile with the 'azure' feature.".to_string(),
        ))
    }

    pub async fn list_providers(&self) -> DeviceResult<Vec<AzureProvider>> {
        Err(DeviceError::UnsupportedDevice(
            "Azure Quantum support not enabled".to_string(),
        ))
    }

    pub async fn list_targets(&self) -> DeviceResult<Vec<AzureTarget>> {
        Err(DeviceError::UnsupportedDevice(
            "Azure Quantum support not enabled".to_string(),
        ))
    }

    pub async fn get_target(&self, _target_id: &str) -> DeviceResult<AzureTarget> {
        Err(DeviceError::UnsupportedDevice(
            "Azure Quantum support not enabled".to_string(),
        ))
    }

    pub async fn submit_circuit(
        &self,
        _target_id: &str,
        _provider_id: &str,
        _config: AzureCircuitConfig,
    ) -> DeviceResult<String> {
        Err(DeviceError::UnsupportedDevice(
            "Azure Quantum support not enabled".to_string(),
        ))
    }

    pub async fn get_job_status(&self, _job_id: &str) -> DeviceResult<AzureJobStatus> {
        Err(DeviceError::UnsupportedDevice(
            "Azure Quantum support not enabled".to_string(),
        ))
    }

    pub async fn get_job_result(&self, _job_id: &str) -> DeviceResult<AzureJobResult> {
        Err(DeviceError::UnsupportedDevice(
            "Azure Quantum support not enabled".to_string(),
        ))
    }

    pub async fn wait_for_job(
        &self,
        _job_id: &str,
        _timeout_secs: Option<u64>,
    ) -> DeviceResult<AzureJobResult> {
        Err(DeviceError::UnsupportedDevice(
            "Azure Quantum support not enabled".to_string(),
        ))
    }

    pub async fn submit_circuits_parallel(
        &self,
        _target_id: &str,
        _provider_id: &str,
        _configs: Vec<AzureCircuitConfig>,
    ) -> DeviceResult<Vec<String>> {
        Err(DeviceError::UnsupportedDevice(
            "Azure Quantum support not enabled".to_string(),
        ))
    }

    pub fn circuit_to_provider_format<const N: usize>(
        _circuit: &Circuit<N>,
        _provider_id: &str,
    ) -> DeviceResult<String> {
        Err(DeviceError::UnsupportedDevice(
            "Azure Quantum support not enabled".to_string(),
        ))
    }
}

#[cfg(all(test, feature = "azure"))]
mod tests {
    use super::*;

    #[test]
    fn test_quantinuum_qasm_emits_real_gates() {
        // Bell-state circuit.
        let mut circuit = Circuit::<2>::new();
        circuit.h(0).expect("add H");
        circuit.cnot(0, 1).expect("add CNOT");

        let qasm = AzureQuantumClient::circuit_to_provider_format(&circuit, "quantinuum")
            .expect("Quantinuum QASM conversion should succeed");

        assert!(qasm.contains("OPENQASM 2.0"), "missing header: {qasm}");
        // REAL gates must be present (not just register declarations).
        assert!(qasm.contains("h q[0]"), "missing H gate: {qasm}");
        assert!(qasm.contains("cx q[0], q[1]"), "missing CNOT gate: {qasm}");
    }

    #[test]
    fn test_ionq_format_emits_real_gates() {
        let mut circuit = Circuit::<2>::new();
        circuit.h(0).expect("add H");
        circuit.cnot(0, 1).expect("add CNOT");

        let json = AzureQuantumClient::circuit_to_provider_format(&circuit, "ionq")
            .expect("IonQ JSON conversion should succeed");

        // The emitted JSON must contain the actual gates, not an empty circuit.
        assert!(json.contains("\"qubits\":2"), "missing qubit count: {json}");
        assert!(json.contains("\"gate\":\"h\""), "missing H gate: {json}");
        assert!(
            json.contains("\"gate\":\"cnot\""),
            "missing CNOT gate: {json}"
        );
        assert!(
            json.contains("\"control\":0"),
            "missing CNOT control: {json}"
        );
        assert!(json.contains("\"target\":1"), "missing CNOT target: {json}");
    }

    #[test]
    fn test_ionq_unsupported_gate_is_honest_error() {
        // A gate with no direct IonQ JSON form must error, never silently drop.
        let mut circuit = Circuit::<3>::new();
        circuit.toffoli(0, 1, 2).expect("add Toffoli");

        let result = AzureQuantumClient::circuit_to_provider_format(&circuit, "ionq");
        assert!(
            matches!(result, Err(DeviceError::CircuitConversion(_))),
            "expected honest CircuitConversion error, got {result:?}"
        );
    }
}
