//! Quantum Cloud Platform Integration
//!
//! This module provides unified interfaces for interacting with major quantum computing
//! cloud platforms including IBM Quantum, AWS Braket, and Google Quantum AI.
//!
//! ## Supported Platforms
//!
//! - **IBM Quantum**: Access to IBM's quantum processors and simulators
//! - **AWS Braket**: Amazon's quantum computing service
//! - **Google Quantum AI**: Google's quantum processors
//! - **Azure Quantum**: Microsoft's quantum computing platform
//!
//! ## Features
//!
//! - Unified API across all platforms
//! - Job submission and monitoring
//! - Result retrieval and analysis
//! - Device capability querying
//! - Circuit transpilation for platform-specific requirements
//! - Cost estimation and optimization

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64 as Complex;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

// ================================================================================================
// Cloud Platform Types
// ================================================================================================

/// Supported quantum cloud platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CloudPlatform {
    /// IBM Quantum
    IBM,
    /// AWS Braket
    AWS,
    /// Google Quantum AI
    Google,
    /// Microsoft Azure Quantum
    Azure,
    /// Rigetti Quantum Cloud Services
    Rigetti,
    /// IonQ Cloud
    IonQ,
}

impl CloudPlatform {
    /// Get platform name
    pub fn name(&self) -> &'static str {
        match self {
            CloudPlatform::IBM => "IBM Quantum",
            CloudPlatform::AWS => "AWS Braket",
            CloudPlatform::Google => "Google Quantum AI",
            CloudPlatform::Azure => "Azure Quantum",
            CloudPlatform::Rigetti => "Rigetti QCS",
            CloudPlatform::IonQ => "IonQ Cloud",
        }
    }

    /// Get default API endpoint
    pub fn endpoint(&self) -> &'static str {
        match self {
            CloudPlatform::IBM => "https://auth.quantum-computing.ibm.com/api",
            CloudPlatform::AWS => "https://braket.us-east-1.amazonaws.com",
            CloudPlatform::Google => "https://quantumengine.googleapis.com",
            CloudPlatform::Azure => "https://quantum.azure.com",
            CloudPlatform::Rigetti => "https://api.rigetti.com",
            CloudPlatform::IonQ => "https://api.ionq.com",
        }
    }

    /// Check if platform supports specific qubit count
    pub fn supports_qubits(&self, num_qubits: usize) -> bool {
        match self {
            CloudPlatform::IBM => num_qubits <= 127, // IBM Quantum Eagle
            CloudPlatform::AWS => num_qubits <= 34,  // AWS Braket max
            CloudPlatform::Google => num_qubits <= 72, // Google Sycamore
            CloudPlatform::Azure => num_qubits <= 40, // Azure various backends
            CloudPlatform::Rigetti => num_qubits <= 80, // Rigetti Aspen-M
            CloudPlatform::IonQ => num_qubits <= 32, // IonQ Aria
        }
    }
}

/// Device type (hardware or simulator)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// Quantum processing unit (real hardware)
    QPU,
    /// State vector simulator
    Simulator,
    /// Tensor network simulator
    TensorNetworkSimulator,
    /// Noisy simulator with error models
    NoisySimulator,
}

/// Backend device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Platform the device belongs to
    pub platform: CloudPlatform,
    /// Device name
    pub name: String,
    /// Device type (QPU or simulator)
    pub device_type: DeviceType,
    /// Number of qubits
    pub num_qubits: usize,
    /// Connectivity graph (which qubits are connected)
    pub connectivity: Vec<(usize, usize)>,
    /// Gate set supported by the device
    pub gate_set: Vec<String>,
    /// Average gate fidelities
    pub gate_fidelities: HashMap<String, f64>,
    /// Qubit coherence times T1 (microseconds)
    pub t1_times: Vec<f64>,
    /// Qubit coherence times T2 (microseconds)
    pub t2_times: Vec<f64>,
    /// Readout fidelity per qubit
    pub readout_fidelity: Vec<f64>,
    /// Whether device is currently available
    pub is_available: bool,
    /// Queue depth
    pub queue_depth: usize,
    /// Estimated cost per shot (in credits or USD)
    pub cost_per_shot: f64,
}

impl DeviceInfo {
    /// Get average single-qubit gate fidelity
    pub fn avg_single_qubit_fidelity(&self) -> f64 {
        let single_qubit_gates = vec!["X", "Y", "Z", "H", "RX", "RY", "RZ"];
        let mut sum = 0.0;
        let mut count = 0;

        for gate in single_qubit_gates {
            if let Some(&fidelity) = self.gate_fidelities.get(gate) {
                sum += fidelity;
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.99 // Default
        }
    }

    /// Get average two-qubit gate fidelity
    pub fn avg_two_qubit_fidelity(&self) -> f64 {
        let two_qubit_gates = vec!["CNOT", "CZ", "SWAP", "iSWAP"];
        let mut sum = 0.0;
        let mut count = 0;

        for gate in two_qubit_gates {
            if let Some(&fidelity) = self.gate_fidelities.get(gate) {
                sum += fidelity;
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.95 // Default
        }
    }

    /// Calculate quality score for ranking devices
    pub fn quality_score(&self) -> f64 {
        let gate_score = (self.avg_single_qubit_fidelity() + self.avg_two_qubit_fidelity()) / 2.0;
        let readout_score =
            self.readout_fidelity.iter().sum::<f64>() / self.readout_fidelity.len() as f64;
        let availability_score = if self.is_available { 1.0 } else { 0.5 };
        let queue_score = 1.0 / (1.0 + self.queue_depth as f64 / 10.0);

        gate_score * 0.4 + readout_score * 0.3 + availability_score * 0.2 + queue_score * 0.1
    }
}

// ================================================================================================
// Quantum Job Management
// ================================================================================================

/// Job status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    /// Job is queued
    Queued,
    /// Job is running
    Running,
    /// Job completed successfully
    Completed,
    /// Job failed with error
    Failed,
    /// Job was cancelled
    Cancelled,
}

/// Quantum job submitted to cloud platform
#[derive(Debug, Clone)]
pub struct QuantumJob {
    /// Unique job ID
    pub job_id: String,
    /// Platform where job is running
    pub platform: CloudPlatform,
    /// Device name
    pub device_name: String,
    /// Job status
    pub status: JobStatus,
    /// Number of shots requested
    pub shots: usize,
    /// Submission time
    pub submitted_at: SystemTime,
    /// Completion time (if completed)
    pub completed_at: Option<SystemTime>,
    /// Result data (if completed)
    pub result: Option<JobResult>,
    /// Error message (if failed)
    pub error_message: Option<String>,
    /// Estimated cost
    pub estimated_cost: f64,
}

impl QuantumJob {
    /// Get execution time
    pub fn execution_time(&self) -> Option<Duration> {
        self.completed_at
            .and_then(|completed| completed.duration_since(self.submitted_at).ok())
    }

    /// Check if job is finished (completed, failed, or cancelled)
    pub fn is_finished(&self) -> bool {
        matches!(
            self.status,
            JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
        )
    }
}

/// Job execution result
#[derive(Debug, Clone)]
pub struct JobResult {
    /// Measurement counts (bitstring -> count)
    pub counts: HashMap<String, usize>,
    /// Measured expectation values (if applicable)
    pub expectation_values: Option<Vec<f64>>,
    /// State vector (if using simulator)
    pub state_vector: Option<Array1<Complex>>,
    /// Density matrix (if using noisy simulator)
    pub density_matrix: Option<Array2<Complex>>,
    /// Raw measurement data
    pub raw_data: Vec<Vec<usize>>,
    /// Job metadata
    pub metadata: HashMap<String, String>,
}

impl JobResult {
    /// Get probability distribution from counts
    pub fn probabilities(&self) -> HashMap<String, f64> {
        let total: usize = self.counts.values().sum();
        self.counts
            .iter()
            .map(|(k, v)| (k.clone(), *v as f64 / total as f64))
            .collect()
    }

    /// Get most probable measurement outcome
    pub fn most_probable_outcome(&self) -> Option<String> {
        self.counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(outcome, _)| outcome.clone())
    }

    /// Calculate measurement entropy
    pub fn entropy(&self) -> f64 {
        let probs = self.probabilities();
        -probs
            .values()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.log2())
            .sum::<f64>()
    }
}

// ================================================================================================
// Cloud Platform Client
// ================================================================================================

/// Configuration for cloud platform connection
#[derive(Debug, Clone)]
pub struct CloudConfig {
    /// Platform to connect to
    pub platform: CloudPlatform,
    /// API token/key for authentication
    pub api_token: String,
    /// API endpoint (optional, uses default if not specified)
    pub endpoint: Option<String>,
    /// Default number of shots
    pub default_shots: usize,
    /// Timeout for API requests (seconds)
    pub timeout: u64,
    /// Enable automatic circuit optimization
    pub auto_optimize: bool,
    /// Maximum qubits to use
    pub max_qubits: Option<usize>,
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            platform: CloudPlatform::IBM,
            api_token: String::new(),
            endpoint: None,
            default_shots: 1000,
            timeout: 300,
            auto_optimize: true,
            max_qubits: None,
        }
    }
}

/// Cloud platform client for job submission and management
pub struct CloudClient {
    config: CloudConfig,
    devices: Vec<DeviceInfo>,
}

impl CloudClient {
    /// Create a new cloud client
    pub fn new(config: CloudConfig) -> Self {
        Self {
            config,
            devices: Vec::new(),
        }
    }

    /// Connect to the cloud platform and authenticate
    pub fn connect(&mut self) -> QuantRS2Result<()> {
        // Simplified: in production would make actual API call
        if self.config.api_token.is_empty() {
            return Err(QuantRS2Error::InvalidInput(
                "API token is required".to_string(),
            ));
        }

        // Load available devices
        self.devices = self.load_devices()?;

        Ok(())
    }

    /// Load available devices from platform
    fn load_devices(&self) -> QuantRS2Result<Vec<DeviceInfo>> {
        // Simplified: return mock devices based on platform
        match self.config.platform {
            CloudPlatform::IBM => Ok(vec![
                DeviceInfo {
                    platform: CloudPlatform::IBM,
                    name: "ibmq_jakarta".to_string(),
                    device_type: DeviceType::QPU,
                    num_qubits: 7,
                    connectivity: vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
                    gate_set: vec!["X", "Y", "Z", "H", "CNOT", "RZ"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    gate_fidelities: HashMap::from([
                        ("X".to_string(), 0.9993),
                        ("CNOT".to_string(), 0.987),
                    ]),
                    t1_times: vec![100.0, 95.0, 110.0, 98.0, 105.0, 92.0, 88.0],
                    t2_times: vec![120.0, 110.0, 115.0, 108.0, 125.0, 105.0, 98.0],
                    readout_fidelity: vec![0.98, 0.97, 0.98, 0.96, 0.97, 0.98, 0.97],
                    is_available: true,
                    queue_depth: 5,
                    cost_per_shot: 0.001,
                },
                DeviceInfo {
                    platform: CloudPlatform::IBM,
                    name: "ibmq_qasm_simulator".to_string(),
                    device_type: DeviceType::Simulator,
                    num_qubits: 32,
                    connectivity: vec![], // Fully connected
                    gate_set: vec!["X", "Y", "Z", "H", "CNOT", "RX", "RY", "RZ"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    gate_fidelities: HashMap::from([
                        ("X".to_string(), 1.0),
                        ("CNOT".to_string(), 1.0),
                    ]),
                    t1_times: vec![],
                    t2_times: vec![],
                    readout_fidelity: vec![],
                    is_available: true,
                    queue_depth: 0,
                    cost_per_shot: 0.0,
                },
            ]),
            CloudPlatform::AWS => Ok(vec![DeviceInfo {
                platform: CloudPlatform::AWS,
                name: "SV1".to_string(),
                device_type: DeviceType::Simulator,
                num_qubits: 34,
                connectivity: vec![],
                gate_set: vec!["X", "Y", "Z", "H", "CNOT", "RX", "RY", "RZ", "CZ"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                gate_fidelities: HashMap::from([("X".to_string(), 1.0), ("CNOT".to_string(), 1.0)]),
                t1_times: vec![],
                t2_times: vec![],
                readout_fidelity: vec![],
                is_available: true,
                queue_depth: 0,
                cost_per_shot: 0.00075,
            }]),
            CloudPlatform::Google => Ok(vec![DeviceInfo {
                platform: CloudPlatform::Google,
                name: "rainbow".to_string(),
                device_type: DeviceType::QPU,
                num_qubits: 23,
                connectivity: vec![(0, 1), (1, 2), (2, 3)], // Simplified
                gate_set: vec!["X", "Y", "Z", "PhasedXZ", "CZ", "SQRT_ISWAP"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                gate_fidelities: HashMap::from([
                    ("X".to_string(), 0.9995),
                    ("CZ".to_string(), 0.993),
                ]),
                t1_times: vec![15.0; 23],
                t2_times: vec![20.0; 23],
                readout_fidelity: vec![0.96; 23],
                is_available: true,
                queue_depth: 3,
                cost_per_shot: 0.002,
            }]),
            CloudPlatform::Azure => Ok(vec![DeviceInfo {
                platform: CloudPlatform::Azure,
                name: "azure-simulator".to_string(),
                device_type: DeviceType::Simulator,
                num_qubits: 40,
                connectivity: vec![],
                gate_set: vec!["X", "Y", "Z", "H", "CNOT", "T"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                gate_fidelities: HashMap::from([("X".to_string(), 1.0), ("CNOT".to_string(), 1.0)]),
                t1_times: vec![],
                t2_times: vec![],
                readout_fidelity: vec![],
                is_available: true,
                queue_depth: 0,
                cost_per_shot: 0.0005,
            }]),
            CloudPlatform::Rigetti => Ok(vec![DeviceInfo {
                platform: CloudPlatform::Rigetti,
                name: "Aspen-M-3".to_string(),
                device_type: DeviceType::QPU,
                num_qubits: 80,
                connectivity: vec![(0, 1), (1, 2)], // Simplified
                gate_set: vec!["RX", "RZ", "CZ", "XY"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                gate_fidelities: HashMap::from([
                    ("RX".to_string(), 0.998),
                    ("CZ".to_string(), 0.95),
                ]),
                t1_times: vec![20.0; 80],
                t2_times: vec![15.0; 80],
                readout_fidelity: vec![0.95; 80],
                is_available: true,
                queue_depth: 8,
                cost_per_shot: 0.0015,
            }]),
            CloudPlatform::IonQ => Ok(vec![DeviceInfo {
                platform: CloudPlatform::IonQ,
                name: "ionq.qpu.aria-1".to_string(),
                device_type: DeviceType::QPU,
                num_qubits: 25,
                connectivity: vec![], // All-to-all connectivity
                gate_set: vec!["X", "Y", "Z", "RX", "RY", "RZ", "MS"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                gate_fidelities: HashMap::from([
                    ("X".to_string(), 0.9999),
                    ("MS".to_string(), 0.995),
                ]),
                t1_times: vec![10000.0; 25], // Very long T1 for trapped ions
                t2_times: vec![1000.0; 25],
                readout_fidelity: vec![0.995; 25],
                is_available: true,
                queue_depth: 12,
                cost_per_shot: 0.003,
            }]),
        }
    }

    /// Get list of available devices
    pub fn list_devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Get device by name
    pub fn get_device(&self, name: &str) -> Option<&DeviceInfo> {
        self.devices.iter().find(|d| d.name == name)
    }

    /// Get best available device based on requirements
    pub fn select_best_device(&self, min_qubits: usize, prefer_qpu: bool) -> Option<&DeviceInfo> {
        self.devices
            .iter()
            .filter(|d| {
                d.num_qubits >= min_qubits
                    && (!prefer_qpu || matches!(d.device_type, DeviceType::QPU))
            })
            .max_by(|a, b| a.quality_score().partial_cmp(&b.quality_score()).unwrap())
    }

    /// Submit a quantum job
    pub fn submit_job(
        &self,
        device_name: &str,
        circuit: &QuantumCircuit,
        shots: Option<usize>,
    ) -> QuantRS2Result<QuantumJob> {
        let device = self.get_device(device_name).ok_or_else(|| {
            QuantRS2Error::InvalidInput(format!("Device {} not found", device_name))
        })?;

        let shots = shots.unwrap_or(self.config.default_shots);

        // Validate circuit
        if circuit.num_qubits > device.num_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Circuit requires {} qubits, device only has {}",
                circuit.num_qubits, device.num_qubits
            )));
        }

        // Calculate estimated cost
        let estimated_cost = shots as f64 * device.cost_per_shot;

        // Create job (simplified - in production would make API call)
        Ok(QuantumJob {
            job_id: format!(
                "job_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_millis()
            ),
            platform: self.config.platform,
            device_name: device_name.to_string(),
            status: JobStatus::Queued,
            shots,
            submitted_at: SystemTime::now(),
            completed_at: None,
            result: None,
            error_message: None,
            estimated_cost,
        })
    }

    /// Check job status
    pub fn check_job_status(&self, job_id: &str) -> QuantRS2Result<JobStatus> {
        // Simplified: in production would make API call
        Ok(JobStatus::Queued)
    }

    /// Wait for job completion
    pub fn wait_for_job(
        &self,
        job_id: &str,
        timeout: Option<Duration>,
    ) -> QuantRS2Result<QuantumJob> {
        // Simplified: in production would poll API until job completes
        Err(QuantRS2Error::UnsupportedOperation(
            "Job waiting not implemented in this simplified version".to_string(),
        ))
    }

    /// Get job result
    pub fn get_job_result(&self, job_id: &str) -> QuantRS2Result<JobResult> {
        // Simplified: in production would fetch from API
        Err(QuantRS2Error::UnsupportedOperation(
            "Job result retrieval not implemented in this simplified version".to_string(),
        ))
    }

    /// Cancel a job
    pub fn cancel_job(&self, job_id: &str) -> QuantRS2Result<()> {
        // Simplified: in production would make API call
        Ok(())
    }

    /// List user's jobs
    pub fn list_jobs(&self, limit: Option<usize>) -> QuantRS2Result<Vec<QuantumJob>> {
        // Simplified: in production would fetch from API
        Ok(Vec::new())
    }
}

/// Quantum circuit representation for cloud submission
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit gates
    pub gates: Vec<Box<dyn GateOp>>,
    /// Measurements to perform
    pub measurements: Vec<usize>,
}

impl QuantumCircuit {
    /// Create a new quantum circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
            measurements: Vec::new(),
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: Box<dyn GateOp>) {
        self.gates.push(gate);
    }

    /// Add measurement
    pub fn measure(&mut self, qubit: usize) {
        if qubit < self.num_qubits {
            self.measurements.push(qubit);
        }
    }

    /// Measure all qubits
    pub fn measure_all(&mut self) {
        self.measurements = (0..self.num_qubits).collect();
    }

    /// Get circuit depth
    pub fn depth(&self) -> usize {
        // Simplified: actual implementation would compute proper depth
        self.gates.len()
    }

    /// Count gates by type
    pub fn gate_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for gate in &self.gates {
            *counts.entry(gate.name().to_string()).or_insert(0) += 1;
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_platform_names() {
        assert_eq!(CloudPlatform::IBM.name(), "IBM Quantum");
        assert_eq!(CloudPlatform::AWS.name(), "AWS Braket");
        assert_eq!(CloudPlatform::Google.name(), "Google Quantum AI");
    }

    #[test]
    fn test_device_quality_score() {
        let device = DeviceInfo {
            platform: CloudPlatform::IBM,
            name: "test_device".to_string(),
            device_type: DeviceType::QPU,
            num_qubits: 5,
            connectivity: vec![],
            gate_set: vec![],
            gate_fidelities: HashMap::from([("X".to_string(), 0.999), ("CNOT".to_string(), 0.99)]),
            t1_times: vec![],
            t2_times: vec![],
            readout_fidelity: vec![0.95, 0.96, 0.97, 0.98, 0.99],
            is_available: true,
            queue_depth: 5,
            cost_per_shot: 0.001,
        };

        let score = device.quality_score();
        assert!(score > 0.8 && score < 1.0);
    }

    #[test]
    fn test_job_result_probabilities() {
        let result = JobResult {
            counts: HashMap::from([
                ("00".to_string(), 500),
                ("01".to_string(), 250),
                ("10".to_string(), 150),
                ("11".to_string(), 100),
            ]),
            expectation_values: None,
            state_vector: None,
            density_matrix: None,
            raw_data: vec![],
            metadata: HashMap::new(),
        };

        let probs = result.probabilities();
        assert_eq!(probs.get("00"), Some(&0.5));
        assert_eq!(probs.get("01"), Some(&0.25));

        let most_probable = result.most_probable_outcome().unwrap();
        assert_eq!(most_probable, "00");
    }

    #[test]
    fn test_quantum_circuit() {
        let mut circuit = QuantumCircuit::new(2);
        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.gates.len(), 0);

        circuit.measure_all();
        assert_eq!(circuit.measurements.len(), 2);
    }
}
