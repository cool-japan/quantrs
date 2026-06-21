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
#[non_exhaustive]
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
    pub const fn name(&self) -> &'static str {
        match self {
            Self::IBM => "IBM Quantum",
            Self::AWS => "AWS Braket",
            Self::Google => "Google Quantum AI",
            Self::Azure => "Azure Quantum",
            Self::Rigetti => "Rigetti QCS",
            Self::IonQ => "IonQ Cloud",
        }
    }

    /// Get default API endpoint
    pub const fn endpoint(&self) -> &'static str {
        match self {
            Self::IBM => "https://auth.quantum-computing.ibm.com/api",
            Self::AWS => "https://braket.us-east-1.amazonaws.com",
            Self::Google => "https://quantumengine.googleapis.com",
            Self::Azure => "https://quantum.azure.com",
            Self::Rigetti => "https://api.rigetti.com",
            Self::IonQ => "https://api.ionq.com",
        }
    }

    /// Check if platform supports specific qubit count
    pub const fn supports_qubits(&self, num_qubits: usize) -> bool {
        match self {
            Self::IBM => num_qubits <= 127,    // IBM Quantum Eagle
            Self::AWS => num_qubits <= 34,     // AWS Braket max
            Self::Google => num_qubits <= 72,  // Google Sycamore
            Self::Azure => num_qubits <= 40,   // Azure various backends
            Self::Rigetti => num_qubits <= 80, // Rigetti Aspen-M
            Self::IonQ => num_qubits <= 32,    // IonQ Aria
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
        let gate_score = f64::midpoint(
            self.avg_single_qubit_fidelity(),
            self.avg_two_qubit_fidelity(),
        );
        let readout_score =
            self.readout_fidelity.iter().sum::<f64>() / self.readout_fidelity.len() as f64;
        let availability_score = if self.is_available { 1.0 } else { 0.5 };
        let queue_score = 1.0 / (1.0 + self.queue_depth as f64 / 10.0);

        gate_score.mul_add(0.4, readout_score * 0.3) + availability_score * 0.2 + queue_score * 0.1
    }
}

// ================================================================================================
// Quantum Job Management
// ================================================================================================

/// Job status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
    pub const fn is_finished(&self) -> bool {
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
    pub const fn new(config: CloudConfig) -> Self {
        Self {
            config,
            devices: Vec::new(),
        }
    }

    /// Connect to the cloud platform and authenticate.
    ///
    /// Validates that an API token is present, then queries the platform for its available
    /// devices. Live device enumeration requires a network backend which is not linked into
    /// this crate, so [`Self::load_devices`] currently returns an honest error rather than
    /// fabricating a hardcoded device list.
    pub fn connect(&mut self) -> QuantRS2Result<()> {
        if self.config.api_token.is_empty() {
            return Err(QuantRS2Error::InvalidInput(
                "API token is required".to_string(),
            ));
        }

        // Load available devices from the live platform.
        self.devices = self.load_devices()?;

        Ok(())
    }

    /// Load available devices from the platform.
    ///
    /// This performs live device enumeration against the configured cloud provider, which
    /// requires an HTTP/network backend. No such backend is linked into `quantrs2-core`, so
    /// rather than fabricating a hardcoded list of devices that would be presented as real
    /// hardware, this returns an honest [`QuantRS2Error::UnsupportedOperation`]. A
    /// network-enabled device crate must provide the real implementation.
    fn load_devices(&self) -> QuantRS2Result<Vec<DeviceInfo>> {
        Err(QuantRS2Error::UnsupportedOperation(format!(
            "live device enumeration requires a network backend which is not linked; \
             cannot query real {} devices",
            self.config.platform.name()
        )))
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
            .max_by(|a, b| {
                a.quality_score()
                    .partial_cmp(&b.quality_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Submit a quantum job
    pub fn submit_job(
        &self,
        device_name: &str,
        circuit: &QuantumCircuit,
        shots: Option<usize>,
    ) -> QuantRS2Result<QuantumJob> {
        let device = self.get_device(device_name).ok_or_else(|| {
            QuantRS2Error::InvalidInput(format!("Device {device_name} not found"))
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
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis();
        Ok(QuantumJob {
            job_id: format!("job_{}", timestamp),
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
    pub const fn check_job_status(&self, job_id: &str) -> QuantRS2Result<JobStatus> {
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
    pub const fn cancel_job(&self, job_id: &str) -> QuantRS2Result<()> {
        // Simplified: in production would make API call
        Ok(())
    }

    /// List user's jobs
    pub const fn list_jobs(&self, limit: Option<usize>) -> QuantRS2Result<Vec<QuantumJob>> {
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

        let most_probable = result
            .most_probable_outcome()
            .expect("should have most probable outcome");
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

    #[test]
    fn test_connect_requires_api_token() {
        let mut client = CloudClient::new(CloudConfig::default());
        let result = client.connect();
        // Empty token: must fail on the token check before reaching device loading.
        assert!(matches!(result, Err(QuantRS2Error::InvalidInput(_))));
    }

    #[test]
    fn test_connect_returns_honest_error_no_network_backend() {
        // With a token present, `connect` reaches `load_devices`, which must return an
        // honest "no network backend" error instead of fabricating mock devices.
        let config = CloudConfig {
            platform: CloudPlatform::IBM,
            api_token: "dummy-token".to_string(),
            ..CloudConfig::default()
        };
        let mut client = CloudClient::new(config);
        let result = client.connect();
        match result {
            Err(QuantRS2Error::UnsupportedOperation(msg)) => {
                assert!(
                    msg.contains("network backend"),
                    "error should explain the missing network backend, got: {msg}"
                );
                assert!(
                    msg.contains("IBM Quantum"),
                    "error should name the platform, got: {msg}"
                );
            }
            other => panic!("expected honest UnsupportedOperation error, got {other:?}"),
        }
        // No devices must have been fabricated into the client state.
        assert!(client.list_devices().is_empty());
    }
}
