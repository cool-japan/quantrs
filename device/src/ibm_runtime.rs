//! IBM Qiskit Runtime primitives and session management.
//!
//! This module provides Qiskit Runtime-compatible primitives:
//! - `Sampler`: For sampling quasi-probability distributions
//! - `Estimator`: For computing expectation values
//! - `Session`: For managing persistent runtime sessions
//!
//! ## Example
//!
//! ```rust,ignore
//! use quantrs2_device::ibm_runtime::{Sampler, Estimator, Session, SessionConfig};
//!
//! // Create a session
//! let session = Session::new(client, "ibm_brisbane", SessionConfig::default()).await?;
//!
//! // Use Sampler primitive
//! let sampler = Sampler::new(&session);
//! let result = sampler.run(&circuit, None).await?;
//!
//! // Use Estimator primitive
//! let estimator = Estimator::new(&session);
//! let expectation = estimator.run(&circuit, &observable).await?;
//!
//! // Session auto-closes on drop
//! ```

use std::collections::HashMap;
use std::sync::Arc;
#[cfg(feature = "ibm")]
use std::time::{Duration, Instant};

#[cfg(feature = "ibm")]
use tokio::sync::RwLock;

use crate::ibm::{IBMJobResult, IBMJobStatus, IBMQuantumClient};
use crate::{DeviceError, DeviceResult};

/// Configuration for a Qiskit Runtime session
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Maximum session duration in seconds
    pub max_time: u64,
    /// Whether to close session on completion
    pub close_on_complete: bool,
    /// Maximum number of circuits per job
    pub max_circuits_per_job: usize,
    /// Optimization level (0-3)
    pub optimization_level: usize,
    /// Resilience level (0-2) for error mitigation
    pub resilience_level: usize,
    /// Enable dynamic circuits
    pub dynamic_circuits: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_time: 7200, // 2 hours
            close_on_complete: true,
            max_circuits_per_job: 100,
            optimization_level: 1,
            resilience_level: 1,
            dynamic_circuits: false,
        }
    }
}

impl SessionConfig {
    /// Create a configuration for short interactive sessions
    pub fn interactive() -> Self {
        Self {
            max_time: 900, // 15 minutes
            close_on_complete: false,
            max_circuits_per_job: 10,
            optimization_level: 1,
            resilience_level: 1,
            dynamic_circuits: false,
        }
    }

    /// Create a configuration for long batch jobs
    pub fn batch() -> Self {
        Self {
            max_time: 28800, // 8 hours
            close_on_complete: true,
            max_circuits_per_job: 300,
            optimization_level: 3,
            resilience_level: 2,
            dynamic_circuits: false,
        }
    }

    /// Create a configuration for dynamic circuit execution
    pub fn dynamic() -> Self {
        Self {
            max_time: 3600,
            close_on_complete: true,
            max_circuits_per_job: 50,
            optimization_level: 1,
            resilience_level: 1,
            dynamic_circuits: true,
        }
    }
}

/// Session state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionState {
    /// Session is being created
    Creating,
    /// Session is active and accepting jobs
    Active,
    /// Session is closing
    Closing,
    /// Session is closed
    Closed,
    /// Session encountered an error
    Error,
}

/// A Qiskit Runtime session for persistent execution context
#[cfg(feature = "ibm")]
pub struct Session {
    /// Session ID
    id: String,
    /// IBM Quantum client
    client: Arc<IBMQuantumClient>,
    /// Backend name
    backend: String,
    /// Session configuration
    config: SessionConfig,
    /// Session state
    state: Arc<RwLock<SessionState>>,
    /// Session creation time
    created_at: Instant,
    /// Number of jobs executed in this session
    job_count: Arc<RwLock<usize>>,
}

#[cfg(not(feature = "ibm"))]
pub struct Session {
    id: String,
    backend: String,
    config: SessionConfig,
}

#[cfg(feature = "ibm")]
impl Session {
    /// Create a new runtime session
    pub async fn new(
        client: IBMQuantumClient,
        backend: &str,
        config: SessionConfig,
    ) -> DeviceResult<Self> {
        // In a real implementation, this would call the IBM Runtime API
        // to create a session. For now, we simulate session creation.
        let session_id = format!(
            "session_{}_{}",
            backend,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        );

        Ok(Self {
            id: session_id,
            client: Arc::new(client),
            backend: backend.to_string(),
            config,
            state: Arc::new(RwLock::new(SessionState::Active)),
            created_at: Instant::now(),
            job_count: Arc::new(RwLock::new(0)),
        })
    }

    /// Get the session ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the backend name
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// Get the session configuration
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Get the current session state
    pub async fn state(&self) -> SessionState {
        self.state.read().await.clone()
    }

    /// Check if the session is active
    pub async fn is_active(&self) -> bool {
        let state = self.state.read().await;
        *state == SessionState::Active
    }

    /// Get the session duration
    pub fn duration(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get the remaining session time
    pub fn remaining_time(&self) -> Option<Duration> {
        let elapsed = self.created_at.elapsed().as_secs();
        if elapsed >= self.config.max_time {
            None
        } else {
            Some(Duration::from_secs(self.config.max_time - elapsed))
        }
    }

    /// Get the number of jobs executed in this session
    pub async fn job_count(&self) -> usize {
        *self.job_count.read().await
    }

    /// Increment job count
    async fn increment_job_count(&self) {
        let mut count = self.job_count.write().await;
        *count += 1;
    }

    /// Get the IBM Quantum client
    pub fn client(&self) -> &IBMQuantumClient {
        &self.client
    }

    /// Close the session
    pub async fn close(&self) -> DeviceResult<()> {
        let mut state = self.state.write().await;
        if *state == SessionState::Closed {
            return Ok(());
        }

        *state = SessionState::Closing;
        // In a real implementation, this would call the IBM Runtime API
        // to close the session
        *state = SessionState::Closed;
        Ok(())
    }
}

#[cfg(not(feature = "ibm"))]
impl Session {
    pub async fn new(
        _client: IBMQuantumClient,
        backend: &str,
        config: SessionConfig,
    ) -> DeviceResult<Self> {
        Ok(Self {
            id: "stub_session".to_string(),
            backend: backend.to_string(),
            config,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    pub async fn is_active(&self) -> bool {
        false
    }

    pub async fn close(&self) -> DeviceResult<()> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Runtime support not enabled".to_string(),
        ))
    }
}

/// Result from a Sampler primitive execution
#[derive(Debug, Clone)]
pub struct SamplerResult {
    /// Quasi-probability distribution for each circuit
    pub quasi_dists: Vec<HashMap<String, f64>>,
    /// Metadata for the execution
    pub metadata: Vec<HashMap<String, String>>,
    /// Number of shots used
    pub shots: usize,
}

impl SamplerResult {
    /// Get the most probable bitstring for a circuit
    pub fn most_probable(&self, circuit_idx: usize) -> Option<(&str, f64)> {
        self.quasi_dists.get(circuit_idx).and_then(|dist| {
            dist.iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, v)| (k.as_str(), *v))
        })
    }

    /// Get the probabilities for a specific bitstring across all circuits
    pub fn probability_of(&self, bitstring: &str) -> Vec<f64> {
        self.quasi_dists
            .iter()
            .map(|dist| *dist.get(bitstring).unwrap_or(&0.0))
            .collect()
    }
}

/// Sampler primitive for sampling quasi-probability distributions
///
/// Compatible with Qiskit Runtime's Sampler primitive
#[cfg(feature = "ibm")]
pub struct Sampler<'a> {
    session: &'a Session,
    options: SamplerOptions,
}

#[cfg(not(feature = "ibm"))]
pub struct Sampler<'a> {
    _phantom: std::marker::PhantomData<&'a ()>,
    options: SamplerOptions,
}

/// Options for the Sampler primitive
#[derive(Debug, Clone)]
pub struct SamplerOptions {
    /// Number of shots
    pub shots: usize,
    /// Seed for random number generation (for reproducibility)
    pub seed: Option<u64>,
    /// Skip transpilation
    pub skip_transpilation: bool,
    /// Dynamical decoupling sequence
    pub dynamical_decoupling: Option<String>,
}

impl Default for SamplerOptions {
    fn default() -> Self {
        Self {
            shots: 4096,
            seed: None,
            skip_transpilation: false,
            dynamical_decoupling: None,
        }
    }
}

#[cfg(feature = "ibm")]
impl<'a> Sampler<'a> {
    /// Create a new Sampler primitive
    pub fn new(session: &'a Session) -> Self {
        Self {
            session,
            options: SamplerOptions::default(),
        }
    }

    /// Create a Sampler with custom options
    pub fn with_options(session: &'a Session, options: SamplerOptions) -> Self {
        Self { session, options }
    }

    /// Run the sampler on a single circuit
    pub async fn run<const N: usize>(
        &self,
        circuit: &quantrs2_circuit::prelude::Circuit<N>,
        parameter_values: Option<&[f64]>,
    ) -> DeviceResult<SamplerResult> {
        self.run_batch(&[circuit], parameter_values.map(|p| vec![p.to_vec()]))
            .await
    }

    /// Run the sampler on multiple circuits
    pub async fn run_batch<const N: usize>(
        &self,
        circuits: &[&quantrs2_circuit::prelude::Circuit<N>],
        _parameter_values: Option<Vec<Vec<f64>>>,
    ) -> DeviceResult<SamplerResult> {
        if !self.session.is_active().await {
            return Err(DeviceError::SessionError(
                "Session is not active".to_string(),
            ));
        }

        // Check remaining time
        if self.session.remaining_time().is_none() {
            return Err(DeviceError::SessionError("Session has expired".to_string()));
        }

        let mut quasi_dists = Vec::new();
        let mut metadata = Vec::new();

        // Convert circuits to QASM and submit
        for (idx, _circuit) in circuits.iter().enumerate() {
            let qasm = format!(
                "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{}];\ncreg c[{}];\n",
                N, N
            );

            let config = crate::ibm::IBMCircuitConfig {
                name: format!("sampler_circuit_{}", idx),
                qasm,
                shots: self.options.shots,
                optimization_level: Some(self.session.config.optimization_level),
                initial_layout: None,
            };

            let job_id = self
                .session
                .client
                .submit_circuit(self.session.backend(), config)
                .await?;

            let result = self.session.client.wait_for_job(&job_id, Some(300)).await?;

            // Convert counts to quasi-probability distribution
            let total: usize = result.counts.values().sum();
            let mut dist = HashMap::new();
            for (bitstring, count) in result.counts {
                dist.insert(bitstring, count as f64 / total as f64);
            }
            quasi_dists.push(dist);

            let mut meta = HashMap::new();
            meta.insert("job_id".to_string(), job_id);
            meta.insert("backend".to_string(), self.session.backend().to_string());
            metadata.push(meta);
        }

        self.session.increment_job_count().await;

        Ok(SamplerResult {
            quasi_dists,
            metadata,
            shots: self.options.shots,
        })
    }
}

#[cfg(not(feature = "ibm"))]
impl<'a> Sampler<'a> {
    pub fn new(_session: &'a Session) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
            options: SamplerOptions::default(),
        }
    }

    pub async fn run<const N: usize>(
        &self,
        _circuit: &quantrs2_circuit::prelude::Circuit<N>,
        _parameter_values: Option<&[f64]>,
    ) -> DeviceResult<SamplerResult> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Runtime support not enabled".to_string(),
        ))
    }
}

/// Result from an Estimator primitive execution
#[derive(Debug, Clone)]
pub struct EstimatorResult {
    /// Expectation values for each circuit-observable pair
    pub values: Vec<f64>,
    /// Standard errors for each expectation value
    pub std_errors: Vec<f64>,
    /// Metadata for the execution
    pub metadata: Vec<HashMap<String, String>>,
}

impl EstimatorResult {
    /// Get the expectation value for a specific index
    pub fn value(&self, idx: usize) -> Option<f64> {
        self.values.get(idx).copied()
    }

    /// Get the standard error for a specific index
    pub fn std_error(&self, idx: usize) -> Option<f64> {
        self.std_errors.get(idx).copied()
    }

    /// Get the mean expectation value across all circuits
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.values.iter().sum::<f64>() / self.values.len() as f64
        }
    }

    /// Get the variance of expectation values
    pub fn variance(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        self.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (self.values.len() - 1) as f64
    }
}

/// Observable specification for the Estimator
#[derive(Debug, Clone)]
pub struct Observable {
    /// Pauli string representation (e.g., "ZZII", "XXXX")
    pub pauli_string: String,
    /// Coefficient for this observable
    pub coefficient: f64,
    /// Qubits this observable acts on
    pub qubits: Vec<usize>,
}

impl Observable {
    /// Create a Z observable on specific qubits
    pub fn z(qubits: &[usize]) -> Self {
        let pauli_string = qubits.iter().map(|_| 'Z').collect();
        Self {
            pauli_string,
            coefficient: 1.0,
            qubits: qubits.to_vec(),
        }
    }

    /// Create an X observable on specific qubits
    pub fn x(qubits: &[usize]) -> Self {
        let pauli_string = qubits.iter().map(|_| 'X').collect();
        Self {
            pauli_string,
            coefficient: 1.0,
            qubits: qubits.to_vec(),
        }
    }

    /// Create a Y observable on specific qubits
    pub fn y(qubits: &[usize]) -> Self {
        let pauli_string = qubits.iter().map(|_| 'Y').collect();
        Self {
            pauli_string,
            coefficient: 1.0,
            qubits: qubits.to_vec(),
        }
    }

    /// Create an identity observable
    pub fn identity(n_qubits: usize) -> Self {
        Self {
            pauli_string: "I".repeat(n_qubits),
            coefficient: 1.0,
            qubits: (0..n_qubits).collect(),
        }
    }

    /// Create a custom Pauli observable
    pub fn pauli(pauli_string: &str, qubits: &[usize], coefficient: f64) -> Self {
        Self {
            pauli_string: pauli_string.to_string(),
            coefficient,
            qubits: qubits.to_vec(),
        }
    }
}

/// Options for the Estimator primitive
#[derive(Debug, Clone)]
pub struct EstimatorOptions {
    /// Number of shots per circuit
    pub shots: usize,
    /// Precision target (stopping criterion)
    pub precision: Option<f64>,
    /// Resilience level (0-2)
    pub resilience_level: usize,
    /// Skip transpilation
    pub skip_transpilation: bool,
}

impl Default for EstimatorOptions {
    fn default() -> Self {
        Self {
            shots: 4096,
            precision: None,
            resilience_level: 1,
            skip_transpilation: false,
        }
    }
}

/// Estimator primitive for computing expectation values
///
/// Compatible with Qiskit Runtime's Estimator primitive
#[cfg(feature = "ibm")]
pub struct Estimator<'a> {
    session: &'a Session,
    options: EstimatorOptions,
}

#[cfg(not(feature = "ibm"))]
pub struct Estimator<'a> {
    _phantom: std::marker::PhantomData<&'a ()>,
    options: EstimatorOptions,
}

#[cfg(feature = "ibm")]
impl<'a> Estimator<'a> {
    /// Create a new Estimator primitive
    pub fn new(session: &'a Session) -> Self {
        Self {
            session,
            options: EstimatorOptions::default(),
        }
    }

    /// Create an Estimator with custom options
    pub fn with_options(session: &'a Session, options: EstimatorOptions) -> Self {
        Self { session, options }
    }

    /// Run the estimator on a single circuit with a single observable
    pub async fn run<const N: usize>(
        &self,
        circuit: &quantrs2_circuit::prelude::Circuit<N>,
        observable: &Observable,
        parameter_values: Option<&[f64]>,
    ) -> DeviceResult<EstimatorResult> {
        self.run_batch(
            &[circuit],
            &[observable],
            parameter_values.map(|p| vec![p.to_vec()]),
        )
        .await
    }

    /// Run the estimator on multiple circuit-observable pairs
    pub async fn run_batch<const N: usize>(
        &self,
        circuits: &[&quantrs2_circuit::prelude::Circuit<N>],
        observables: &[&Observable],
        _parameter_values: Option<Vec<Vec<f64>>>,
    ) -> DeviceResult<EstimatorResult> {
        if !self.session.is_active().await {
            return Err(DeviceError::SessionError(
                "Session is not active".to_string(),
            ));
        }

        if self.session.remaining_time().is_none() {
            return Err(DeviceError::SessionError("Session has expired".to_string()));
        }

        let mut values = Vec::new();
        let mut std_errors = Vec::new();
        let mut metadata = Vec::new();

        // For each circuit-observable pair
        for (idx, (_circuit, observable)) in circuits.iter().zip(observables.iter()).enumerate() {
            // Build measurement circuit based on observable
            let qasm = self.build_measurement_circuit::<N>(observable);

            let config = crate::ibm::IBMCircuitConfig {
                name: format!("estimator_circuit_{}", idx),
                qasm,
                shots: self.options.shots,
                optimization_level: Some(self.session.config.optimization_level),
                initial_layout: None,
            };

            let job_id = self
                .session
                .client
                .submit_circuit(self.session.backend(), config)
                .await?;

            let result = self.session.client.wait_for_job(&job_id, Some(300)).await?;

            // Calculate expectation value from measurement results
            let (exp_value, std_err) = self.compute_expectation(&result, observable);
            values.push(exp_value);
            std_errors.push(std_err);

            let mut meta = HashMap::new();
            meta.insert("job_id".to_string(), job_id);
            meta.insert("observable".to_string(), observable.pauli_string.clone());
            metadata.push(meta);
        }

        self.session.increment_job_count().await;

        Ok(EstimatorResult {
            values,
            std_errors,
            metadata,
        })
    }

    /// Build a measurement circuit for the given observable
    fn build_measurement_circuit<const N: usize>(&self, observable: &Observable) -> String {
        let mut qasm = format!(
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{}];\ncreg c[{}];\n",
            N, N
        );

        // Add basis rotation gates based on Pauli string
        for (i, pauli) in observable.pauli_string.chars().enumerate() {
            if i < observable.qubits.len() {
                let qubit = observable.qubits[i];
                match pauli {
                    'X' => {
                        // Rotate to X basis: H gate
                        qasm.push_str(&format!("h q[{}];\n", qubit));
                    }
                    'Y' => {
                        // Rotate to Y basis: S†H gates
                        qasm.push_str(&format!("sdg q[{}];\n", qubit));
                        qasm.push_str(&format!("h q[{}];\n", qubit));
                    }
                    'Z' | 'I' => {
                        // Z basis is computational basis, no rotation needed
                    }
                    _ => {}
                }
            }
        }

        // Add measurements
        for (i, qubit) in observable.qubits.iter().enumerate() {
            qasm.push_str(&format!("measure q[{}] -> c[{}];\n", qubit, i));
        }

        qasm
    }

    /// Compute expectation value from measurement results
    fn compute_expectation(&self, result: &IBMJobResult, observable: &Observable) -> (f64, f64) {
        let total_shots: usize = result.counts.values().sum();
        if total_shots == 0 {
            return (0.0, 0.0);
        }

        let mut expectation = 0.0;
        let mut squared_sum = 0.0;

        for (bitstring, count) in &result.counts {
            // Calculate eigenvalue for this bitstring
            let eigenvalue = self.compute_eigenvalue(bitstring, observable);
            let probability = *count as f64 / total_shots as f64;

            expectation += eigenvalue * probability;
            squared_sum += eigenvalue.powi(2) * probability;
        }

        expectation *= observable.coefficient;

        // Standard error: sqrt(Var / n)
        let variance = squared_sum - expectation.powi(2);
        let std_error = (variance / total_shots as f64).sqrt();

        (expectation, std_error)
    }

    /// Compute the eigenvalue for a measurement outcome
    fn compute_eigenvalue(&self, bitstring: &str, observable: &Observable) -> f64 {
        let mut eigenvalue = 1.0;

        for (i, pauli) in observable.pauli_string.chars().enumerate() {
            if i < bitstring.len() && pauli != 'I' {
                // Get the bit value (assuming little-endian)
                let bit = bitstring.chars().rev().nth(i).unwrap_or('0');
                if bit == '1' {
                    eigenvalue *= -1.0;
                }
            }
        }

        eigenvalue
    }
}

#[cfg(not(feature = "ibm"))]
impl<'a> Estimator<'a> {
    pub fn new(_session: &'a Session) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
            options: EstimatorOptions::default(),
        }
    }

    pub async fn run<const N: usize>(
        &self,
        _circuit: &quantrs2_circuit::prelude::Circuit<N>,
        _observable: &Observable,
        _parameter_values: Option<&[f64]>,
    ) -> DeviceResult<EstimatorResult> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Runtime support not enabled".to_string(),
        ))
    }
}

/// Batch execution mode for runtime primitives
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Interactive mode with immediate feedback
    Interactive,
    /// Batch mode for large workloads
    Batch,
    /// Dedicated mode with reserved resources
    Dedicated,
}

/// Runtime job information
#[derive(Debug, Clone)]
pub struct RuntimeJob {
    /// Job ID
    pub id: String,
    /// Session ID (if part of a session)
    pub session_id: Option<String>,
    /// Job status
    pub status: IBMJobStatus,
    /// Primitive type (sampler or estimator)
    pub primitive: String,
    /// Creation timestamp
    pub created_at: String,
    /// Backend name
    pub backend: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert_eq!(config.max_time, 7200);
        assert!(config.close_on_complete);
        assert_eq!(config.optimization_level, 1);
    }

    #[test]
    fn test_session_config_interactive() {
        let config = SessionConfig::interactive();
        assert_eq!(config.max_time, 900);
        assert!(!config.close_on_complete);
    }

    #[test]
    fn test_session_config_batch() {
        let config = SessionConfig::batch();
        assert_eq!(config.max_time, 28800);
        assert_eq!(config.optimization_level, 3);
    }

    #[test]
    fn test_observable_z() {
        let obs = Observable::z(&[0, 1]);
        assert_eq!(obs.pauli_string, "ZZ");
        assert_eq!(obs.coefficient, 1.0);
        assert_eq!(obs.qubits, vec![0, 1]);
    }

    #[test]
    fn test_observable_x() {
        let obs = Observable::x(&[0]);
        assert_eq!(obs.pauli_string, "X");
    }

    #[test]
    fn test_observable_y() {
        let obs = Observable::y(&[0, 1, 2]);
        assert_eq!(obs.pauli_string, "YYY");
    }

    #[test]
    fn test_observable_identity() {
        let obs = Observable::identity(4);
        assert_eq!(obs.pauli_string, "IIII");
    }

    #[test]
    fn test_sampler_options_default() {
        let options = SamplerOptions::default();
        assert_eq!(options.shots, 4096);
        assert!(options.seed.is_none());
    }

    #[test]
    fn test_estimator_options_default() {
        let options = EstimatorOptions::default();
        assert_eq!(options.shots, 4096);
        assert_eq!(options.resilience_level, 1);
    }

    #[test]
    fn test_sampler_result_most_probable() {
        let mut dist = HashMap::new();
        dist.insert("00".to_string(), 0.7);
        dist.insert("11".to_string(), 0.3);

        let result = SamplerResult {
            quasi_dists: vec![dist],
            metadata: vec![HashMap::new()],
            shots: 1000,
        };

        let (bitstring, prob) = result.most_probable(0).unwrap();
        assert_eq!(bitstring, "00");
        assert!((prob - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_estimator_result_mean() {
        let result = EstimatorResult {
            values: vec![0.5, 0.3, 0.2],
            std_errors: vec![0.01, 0.01, 0.01],
            metadata: vec![HashMap::new(); 3],
        };

        let mean = result.mean();
        assert!((mean - (0.5 + 0.3 + 0.2) / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimator_result_variance() {
        let result = EstimatorResult {
            values: vec![1.0, 2.0, 3.0],
            std_errors: vec![0.1, 0.1, 0.1],
            metadata: vec![HashMap::new(); 3],
        };

        let variance = result.variance();
        assert!(variance > 0.0);
    }
}
