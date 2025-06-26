//! Rigetti Quantum Cloud Services client implementation
//!
//! This module provides integration with Rigetti's quantum computing platform,
//! including their superconducting qubit processors and the Quil programming language.

use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use tokio::time::sleep;
use async_trait::async_trait;

use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::gate::GateOp;

use crate::{DeviceError, DeviceResult, CircuitResult, QuantumDevice, CircuitExecutor};
use crate::translation::HardwareBackend;

/// Configuration for Rigetti Quantum Cloud Services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigettiConfig {
    /// Quantum Cloud Services API endpoint
    pub qcs_url: String,
    /// API token for authentication
    pub api_token: String,
    /// User ID
    pub user_id: String,
    /// Default quantum processor unit (QPU) to use
    pub default_qpu: Option<String>,
    /// Request timeout in seconds
    pub timeout: u64,
    /// Maximum number of retries for failed requests
    pub max_retries: u32,
}

impl Default for RigettiConfig {
    fn default() -> Self {
        Self {
            qcs_url: "https://qcs.rigetti.com".to_string(),
            api_token: String::new(),
            user_id: String::new(),
            default_qpu: None,
            timeout: 300, // 5 minutes
            max_retries: 3,
        }
    }
}

/// Rigetti Quantum Cloud Services client
pub struct RigettiClient {
    config: RigettiConfig,
    client: Client,
}

impl RigettiClient {
    /// Create a new Rigetti client
    pub fn new(config: RigettiConfig) -> DeviceResult<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout))
            .build()
            .map_err(|e| DeviceError::Connection(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self { config, client })
    }

    /// Get available quantum processing units
    pub async fn get_available_qpus(&self) -> DeviceResult<Vec<RigettiQPU>> {
        let url = format!("{}/api/v1/qpus", self.config.qcs_url);
        
        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_token))
            .send()
            .await
            .map_err(|e| DeviceError::Connection(format!("Failed to get QPUs: {}", e)))?;

        if !response.status().is_success() {
            return Err(DeviceError::Connection(
                format!("QPU listing failed with status: {}", response.status())
            ));
        }

        let qpus: Vec<RigettiQPU> = response
            .json()
            .await
            .map_err(|e| DeviceError::Parsing(format!("Failed to parse QPU response: {}", e)))?;

        Ok(qpus)
    }

    /// Submit a quantum program for execution
    pub async fn submit_program(&self, program: &RigettiProgram) -> DeviceResult<String> {
        let url = format!("{}/api/v1/jobs", self.config.qcs_url);
        
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_token))
            .header("Content-Type", "application/json")
            .json(program)
            .send()
            .await
            .map_err(|e| DeviceError::Connection(format!("Failed to submit program: {}", e)))?;

        if !response.status().is_success() {
            return Err(DeviceError::Execution(
                format!("Program submission failed with status: {}", response.status())
            ));
        }

        let job_response: RigettiJobResponse = response
            .json()
            .await
            .map_err(|e| DeviceError::Parsing(format!("Failed to parse job response: {}", e)))?;

        Ok(job_response.job_id)
    }

    /// Get job results
    pub async fn get_job_results(&self, job_id: &str) -> DeviceResult<RigettiJobResults> {
        let url = format!("{}/api/v1/jobs/{}/results", self.config.qcs_url, job_id);
        
        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_token))
            .send()
            .await
            .map_err(|e| DeviceError::Connection(format!("Failed to get job results: {}", e)))?;

        if !response.status().is_success() {
            return Err(DeviceError::Execution(
                format!("Failed to get results with status: {}", response.status())
            ));
        }

        let results: RigettiJobResults = response
            .json()
            .await
            .map_err(|e| DeviceError::Parsing(format!("Failed to parse job results: {}", e)))?;

        Ok(results)
    }

    /// Convert QuantRS circuit to Quil program
    pub fn circuit_to_quil<const N: usize>(&self, circuit: &Circuit<N>) -> DeviceResult<String> {
        let mut quil_lines = Vec::new();
        
        // Add DECLARE statement for memory
        quil_lines.push(format!("DECLARE memory BIT[{}]", N));
        
        // Convert gates to Quil instructions
        for gate_info in circuit.iter_gates() {
            let quil_instruction = self.gate_to_quil(&gate_info.gate, &gate_info.qubits)?;
            quil_lines.push(quil_instruction);
        }
        
        // Add measurements
        for i in 0..N {
            quil_lines.push(format!("MEASURE {} memory[{}]", i, i));
        }
        
        Ok(quil_lines.join("\n"))
    }

    /// Convert individual gate to Quil instruction
    fn gate_to_quil(&self, gate: &dyn GateOp, qubits: &[usize]) -> DeviceResult<String> {
        let gate_name = gate.name();
        match gate_name.as_str() {
            "I" => Ok(format!("I {}", qubits[0])),
            "X" => Ok(format!("X {}", qubits[0])),
            "Y" => Ok(format!("Y {}", qubits[0])),
            "Z" => Ok(format!("Z {}", qubits[0])),
            "H" => Ok(format!("H {}", qubits[0])),
            "S" => Ok(format!("S {}", qubits[0])),
            "T" => Ok(format!("T {}", qubits[0])),
            "RX" => {
                if let Some(params) = gate.parameters() {
                    Ok(format!("RX({}) {}", params[0], qubits[0]))
                } else {
                    Err(DeviceError::CircuitConversion("RX gate missing angle parameter".to_string()))
                }
            },
            "RY" => {
                if let Some(params) = gate.parameters() {
                    Ok(format!("RY({}) {}", params[0], qubits[0]))
                } else {
                    Err(DeviceError::CircuitConversion("RY gate missing angle parameter".to_string()))
                }
            },
            "RZ" => {
                if let Some(params) = gate.parameters() {
                    Ok(format!("RZ({}) {}", params[0], qubits[0]))
                } else {
                    Err(DeviceError::CircuitConversion("RZ gate missing angle parameter".to_string()))
                }
            },
            "CNOT" | "CX" => Ok(format!("CNOT {} {}", qubits[0], qubits[1])),
            "CZ" => Ok(format!("CZ {} {}", qubits[0], qubits[1])),
            "SWAP" => Ok(format!("SWAP {} {}", qubits[0], qubits[1])),
            "CCX" | "CCNOT" => Ok(format!("CCNOT {} {} {}", qubits[0], qubits[1], qubits[2])),
            _ => Err(DeviceError::UnsupportedGate(format!("Gate not supported by Rigetti backend: {}", gate_name))),
        }
    }
}

/// Rigetti QPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigettiQPU {
    pub name: String,
    pub id: String,
    pub status: String,
    pub num_qubits: usize,
    pub connectivity: Vec<(usize, usize)>,
    pub gate_times: HashMap<String, f64>,
    pub fidelities: HashMap<String, f64>,
    pub topology: String,
}

/// Rigetti quantum program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigettiProgram {
    pub quil: String,
    pub qpu: String,
    pub shots: u32,
    pub timeout: Option<u64>,
    pub priority: Option<String>,
}

/// Rigetti job response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigettiJobResponse {
    pub job_id: String,
    pub status: String,
    pub created_at: String,
}

/// Rigetti job results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigettiJobResults {
    pub job_id: String,
    pub measurements: Vec<Vec<i32>>,
    pub execution_time: f64,
    pub qpu_time: f64,
    pub readout_errors: Option<HashMap<String, f64>>,
}

/// Rigetti device implementation
pub struct RigettiDevice {
    client: RigettiClient,
    qpu_info: RigettiQPU,
}

impl RigettiDevice {
    /// Create a new Rigetti device
    pub async fn new(config: RigettiConfig) -> DeviceResult<Self> {
        let client = RigettiClient::new(config.clone())?;
        
        // Get default QPU info
        let qpu_name = config.default_qpu.as_ref()
            .ok_or_else(|| DeviceError::Configuration("No default QPU specified".to_string()))?;
        
        let qpus = client.get_available_qpus().await?;
        let qpu_info = qpus.into_iter()
            .find(|qpu| &qpu.name == qpu_name)
            .ok_or_else(|| DeviceError::Configuration(format!("QPU '{}' not found", qpu_name)))?;
        
        Ok(Self { client, qpu_info })
    }
}

#[async_trait]
impl QuantumDevice for RigettiDevice {
    async fn is_available(&self) -> DeviceResult<bool> {
        Ok(self.qpu_info.status == "online")
    }

    async fn qubit_count(&self) -> DeviceResult<usize> {
        Ok(self.qpu_info.num_qubits)
    }

    async fn properties(&self) -> DeviceResult<HashMap<String, String>> {
        let mut props = HashMap::new();
        props.insert("backend".to_string(), "rigetti".to_string());
        props.insert("qpu_name".to_string(), self.qpu_info.name.clone());
        props.insert("num_qubits".to_string(), self.qpu_info.num_qubits.to_string());
        props.insert("topology".to_string(), self.qpu_info.topology.clone());
        props.insert("status".to_string(), self.qpu_info.status.clone());
        Ok(props)
    }

    fn is_simulator(&self) -> bool {
        false
    }

    async fn estimated_queue_time<const N: usize>(&self, _circuit: &Circuit<N>) -> DeviceResult<Duration> {
        // Default estimate - in practice this would query actual queue status
        Ok(Duration::from_secs(60))
    }

    async fn can_execute_circuit<const N: usize>(&self, circuit: &Circuit<N>) -> DeviceResult<bool> {
        // Check if circuit fits on device
        if N > self.qpu_info.num_qubits {
            return Ok(false);
        }

        // Check gate support (simplified - could be more sophisticated)
        for gate_info in circuit.iter_gates() {
            let gate_name = gate_info.gate.name();
            match gate_name.as_str() {
                "I" | "X" | "Y" | "Z" | "H" | "S" | "T" | 
                "RX" | "RY" | "RZ" | "CNOT" | "CX" | "CZ" | "SWAP" | "CCX" | "CCNOT" => {},
                _ => return Ok(false),
            }
        }

        Ok(true)
    }
}

#[async_trait]
impl CircuitExecutor for RigettiDevice {
    async fn execute_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        shots: usize,
    ) -> DeviceResult<CircuitResult> {
        // Convert circuit to Quil
        let quil_program = self.client.circuit_to_quil(circuit)?;
        
        // Create Rigetti program
        let program = RigettiProgram {
            quil: quil_program,
            qpu: self.qpu_info.name.clone(),
            shots: shots as u32,
            timeout: Some(self.client.config.timeout),
            priority: None,
        };
        
        // Submit and wait for results
        let job_id = self.client.submit_program(&program).await?;
        
        // Poll for completion (simplified - real implementation would be more sophisticated)
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        let results = self.client.get_job_results(&job_id).await?;
        
        // Convert measurements to counts
        let mut counts = HashMap::new();
        for measurement in &results.measurements {
            let bitstring: String = measurement.iter().map(|&bit| bit.to_string()).collect();
            *counts.entry(bitstring).or_insert(0) += 1;
        }
        
        Ok(CircuitResult { counts, shots })
    }

    async fn execute_circuits<const N: usize>(
        &self,
        circuits: Vec<&Circuit<N>>,
        shots: usize,
    ) -> DeviceResult<Vec<CircuitResult>> {
        let mut results = Vec::new();
        
        // Execute circuits sequentially (could be parallelized)
        for circuit in circuits {
            let result = self.execute_circuit(circuit, shots).await?;
            results.push(result);
        }
        
        Ok(results)
    }
}

/// Create a Rigetti client
pub fn create_rigetti_client(config: RigettiConfig) -> DeviceResult<RigettiClient> {
    RigettiClient::new(config)
}

/// Create a Rigetti device
pub async fn create_rigetti_device(config: RigettiConfig) -> DeviceResult<RigettiDevice> {
    RigettiDevice::new(config).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rigetti_client_creation() {
        let config = RigettiConfig {
            api_token: "test_token".to_string(),
            user_id: "test_user".to_string(),
            ..RigettiConfig::default()
        };

        let client = RigettiClient::new(config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_circuit_to_quil_conversion() {
        let config = RigettiConfig::default();
        let client = RigettiClient::new(config).unwrap();
        
        let mut circuit = Circuit::<2>::new();
        // Note: This is a simplified test - actual circuit building would use proper QuantRS gates
        
        // For now just test the basic structure
        let quil = client.circuit_to_quil(&circuit).unwrap();
        
        assert!(quil.contains("DECLARE memory BIT[2]"));
        assert!(quil.contains("MEASURE 0 memory[0]"));
        assert!(quil.contains("MEASURE 1 memory[1]"));
    }
}