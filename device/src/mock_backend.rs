//! Mock quantum backend for integration testing without real cloud credentials.
//!
//! Provides a fully configurable fake quantum backend that generates plausible
//! measurement results, simulates job latency, and records all submitted jobs.
//!
//! # Examples
//!
//! ```rust
//! use quantrs2_device::mock_backend::{MockBackendConfig, MockQuantumBackend};
//!
//! let backend = MockQuantumBackend::new(MockBackendConfig::perfect(5));
//! let qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n";
//! let counts = backend.run(qasm, 1024).expect("run should succeed");
//! assert!(!counts.is_empty());
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for mock backend behavior.
#[derive(Debug, Clone)]
pub struct MockBackendConfig {
    /// Name shown in backend queries
    pub name: String,
    /// Number of available qubits
    pub max_qubits: usize,
    /// Maximum shots per job
    pub max_shots: usize,
    /// Simulated job queue latency in milliseconds
    pub latency_ms: u64,
    /// Depolarizing error rate per gate (0.0 = perfect)
    pub error_rate: f64,
    /// Probability that a job fails (for testing error handling)
    pub fail_rate: f64,
    /// Supported gate names (empty = all gates supported)
    pub gate_set: Vec<String>,
    /// Allowed qubit pairs for 2-qubit gates (empty = all pairs allowed)
    pub connectivity: Vec<(usize, usize)>,
    /// Random seed for reproducible results
    pub rng_seed: u64,
}

impl Default for MockBackendConfig {
    fn default() -> Self {
        Self {
            name: "mock_backend".to_string(),
            max_qubits: 32,
            max_shots: 8192,
            latency_ms: 0,
            error_rate: 0.0,
            fail_rate: 0.0,
            gate_set: vec![],
            connectivity: vec![],
            rng_seed: 42,
        }
    }
}

impl MockBackendConfig {
    /// Create a "perfect" noiseless backend with the given qubit count.
    pub fn perfect(n_qubits: usize) -> Self {
        Self {
            max_qubits: n_qubits,
            ..Default::default()
        }
    }

    /// Create a backend mimicking IBM Nairobi (7 qubits, T-topology).
    pub fn ibm_nairobi_like() -> Self {
        Self {
            name: "mock_ibm_nairobi".to_string(),
            max_qubits: 7,
            max_shots: 4096,
            latency_ms: 100,
            error_rate: 0.001,
            fail_rate: 0.0,
            gate_set: vec![
                "cx".to_string(),
                "rz".to_string(),
                "sx".to_string(),
                "x".to_string(),
            ],
            connectivity: vec![(0, 1), (1, 2), (1, 3), (3, 5), (4, 5), (5, 6)],
            rng_seed: 0,
        }
    }

    /// Create a backend that always fails — useful for testing error-handling paths.
    pub fn always_fails() -> Self {
        Self {
            name: "failing_backend".to_string(),
            fail_rate: 1.0,
            ..Default::default()
        }
    }

    /// Create a noisy backend with the given error rate.
    pub fn with_noise(n_qubits: usize, error_rate: f64) -> Self {
        Self {
            name: "noisy_mock_backend".to_string(),
            max_qubits: n_qubits,
            error_rate,
            ..Default::default()
        }
    }
}

// ─── Job records ─────────────────────────────────────────────────────────────

/// Record of a submitted job, persisted in the backend's job log.
#[derive(Debug, Clone)]
pub struct MockJobRecord {
    /// Unique job identifier assigned by the mock backend
    pub job_id: String,
    /// Raw QASM string that was submitted
    pub circuit_qasm: String,
    /// Number of shots requested
    pub shots: usize,
    /// Wall-clock time at submission
    pub submitted_at: std::time::SystemTime,
    /// Measurement counts (bitstring → count), `None` on failure
    pub result: Option<HashMap<String, usize>>,
    /// Whether the backend simulated a failure for this job
    pub failed: bool,
    /// Human-readable error description when `failed` is true
    pub error_message: Option<String>,
}

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors that can be returned by [`MockQuantumBackend::run`].
#[derive(Debug)]
pub enum MockBackendError {
    /// Simulated backend failure (configured via `fail_rate`)
    JobFailed(String),
    /// Circuit requires more qubits than the backend supports
    TooManyQubits {
        /// Qubits requested by the circuit
        requested: usize,
        /// Maximum the backend allows
        max: usize,
    },
    /// Shot count exceeds the backend's limit
    TooManyShots {
        /// Shots requested by the caller
        requested: usize,
        /// Maximum the backend allows
        max: usize,
    },
    /// QASM string could not be parsed or is structurally invalid
    InvalidCircuit(String),
}

impl std::fmt::Display for MockBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MockBackendError::JobFailed(msg) => write!(f, "mock job failed: {msg}"),
            MockBackendError::TooManyQubits { requested, max } => {
                write!(
                    f,
                    "requested {requested} qubits, backend max is {max}"
                )
            }
            MockBackendError::TooManyShots { requested, max } => {
                write!(
                    f,
                    "requested {requested} shots, backend max is {max}"
                )
            }
            MockBackendError::InvalidCircuit(msg) => write!(f, "invalid circuit: {msg}"),
        }
    }
}

impl std::error::Error for MockBackendError {}

// ─── Backend ─────────────────────────────────────────────────────────────────

/// A configurable mock quantum backend for use in integration tests.
///
/// All submitted jobs are recorded and can be retrieved via [`MockQuantumBackend::all_jobs`].
/// Measurement results are generated deterministically from the configured seed so that
/// tests are reproducible.
pub struct MockQuantumBackend {
    /// Public backend configuration
    pub config: MockBackendConfig,
    job_records: Arc<Mutex<Vec<MockJobRecord>>>,
}

impl std::fmt::Debug for MockQuantumBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockQuantumBackend")
            .field("config", &self.config)
            .finish()
    }
}

impl MockQuantumBackend {
    /// Create a new mock backend with the given configuration.
    pub fn new(config: MockBackendConfig) -> Self {
        Self {
            config,
            job_records: Arc::new(Mutex::new(vec![])),
        }
    }

    /// Return the number of jobs submitted so far.
    pub fn job_count(&self) -> usize {
        self.job_records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Return a snapshot of all job records accumulated so far.
    pub fn all_jobs(&self) -> Vec<MockJobRecord> {
        self.job_records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Return the most recently submitted job, or `None` if no jobs have run.
    pub fn last_job(&self) -> Option<MockJobRecord> {
        self.job_records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last()
            .cloned()
    }

    /// Submit a QASM 2.0 circuit and return measurement counts.
    ///
    /// Validates qubit and shot limits, optionally simulates latency, optionally
    /// injects a simulated failure, then generates and returns measurement counts.
    /// Every invocation appends a [`MockJobRecord`] to the backend's log.
    pub fn run(
        &self,
        circuit_qasm: &str,
        shots: usize,
    ) -> Result<HashMap<String, usize>, MockBackendError> {
        // Validate shots
        if shots > self.config.max_shots {
            return Err(MockBackendError::TooManyShots {
                requested: shots,
                max: self.config.max_shots,
            });
        }

        // Parse qubit count from QASM; fall back to 1 qubit for well-formed but
        // qubit-free strings — malformed input that still results in 0 shots is
        // handled gracefully by generating an empty count map.
        let n_qubits = self.parse_qubit_count(circuit_qasm).unwrap_or(1);
        if n_qubits > self.config.max_qubits {
            return Err(MockBackendError::TooManyQubits {
                requested: n_qubits,
                max: self.config.max_qubits,
            });
        }

        // Simulate latency
        if self.config.latency_ms > 0 {
            std::thread::sleep(Duration::from_millis(self.config.latency_ms));
        }

        // Generate job ID using fastrand (already a workspace dep of quantrs2-device)
        let raw_id = fastrand::u64(..);
        let job_id = format!("mock-job-{raw_id}");

        // Simulate random failure
        let failed = self.config.fail_rate > 0.0 && fastrand::f64() < self.config.fail_rate;

        if failed {
            let record = MockJobRecord {
                job_id,
                circuit_qasm: circuit_qasm.to_string(),
                shots,
                submitted_at: std::time::SystemTime::now(),
                result: None,
                failed: true,
                error_message: Some("simulated backend failure".to_string()),
            };
            self.job_records
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(record);
            return Err(MockBackendError::JobFailed(
                "simulated backend failure".to_string(),
            ));
        }

        // Generate measurement counts
        let counts = self.generate_counts(n_qubits, shots);

        let record = MockJobRecord {
            job_id,
            circuit_qasm: circuit_qasm.to_string(),
            shots,
            submitted_at: std::time::SystemTime::now(),
            result: Some(counts.clone()),
            failed: false,
            error_message: None,
        };
        self.job_records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(record);

        Ok(counts)
    }

    /// Parse `qreg q[N];` from a QASM 2.0 string and return N.
    ///
    /// Returns `None` when no `qreg` line is present.
    fn parse_qubit_count(&self, qasm: &str) -> Option<usize> {
        for line in qasm.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("qreg") {
                if let Some(start) = trimmed.find('[') {
                    if let Some(end) = trimmed.find(']') {
                        if start < end {
                            return trimmed[start + 1..end].parse().ok();
                        }
                    }
                }
            }
        }
        None
    }

    /// Generate `shots` measurement results for an `n_qubits`-qubit register.
    ///
    /// Uses a seeded PRNG for reproducibility. When `error_rate > 0.0`, each
    /// bit in each outcome is independently flipped with that probability.
    fn generate_counts(&self, n_qubits: usize, shots: usize) -> HashMap<String, usize> {
        // Guard: a 0-qubit circuit or 0 shots produces an empty map.
        if n_qubits == 0 || shots == 0 {
            return HashMap::new();
        }

        let mut rng = fastrand::Rng::with_seed(
            self.config
                .rng_seed
                .wrapping_add(n_qubits as u64)
                .wrapping_add(shots as u64),
        );

        let n_states = 1usize << n_qubits.min(63); // guard against overflow
        let mut counts: HashMap<String, usize> = HashMap::new();

        for _ in 0..shots {
            let state = rng.usize(..n_states);

            // Apply independent per-bit depolarizing noise
            let noisy_state = if self.config.error_rate > 0.0 {
                let mut s = state;
                for bit in 0..n_qubits {
                    if rng.f64() < self.config.error_rate {
                        s ^= 1 << bit;
                    }
                }
                s
            } else {
                state
            };

            let bitstring = format!("{noisy_state:0>n_qubits$b}");
            *counts.entry(bitstring).or_insert(0) += 1;
        }

        counts
    }

    /// Return a map of backend capabilities (suitable for display / comparison).
    pub fn capabilities(&self) -> HashMap<String, String> {
        let mut caps = HashMap::new();
        caps.insert("name".to_string(), self.config.name.clone());
        caps.insert("n_qubits".to_string(), self.config.max_qubits.to_string());
        caps.insert("max_shots".to_string(), self.config.max_shots.to_string());
        caps.insert("simulator".to_string(), "true".to_string());
        caps.insert(
            "error_rate".to_string(),
            format!("{:.6}", self.config.error_rate),
        );
        caps.insert(
            "supported_gates".to_string(),
            if self.config.gate_set.is_empty() {
                "all".to_string()
            } else {
                self.config.gate_set.join(",")
            },
        );
        caps.insert(
            "connectivity".to_string(),
            if self.config.connectivity.is_empty() {
                "all-to-all".to_string()
            } else {
                self.config
                    .connectivity
                    .iter()
                    .map(|(a, b)| format!("{a}-{b}"))
                    .collect::<Vec<_>>()
                    .join(",")
            },
        );
        caps
    }

    /// Return `true` when the given gate name is supported by this backend.
    ///
    /// An empty gate set (the default) means every gate is accepted.
    pub fn supports_gate(&self, gate_name: &str) -> bool {
        self.config.gate_set.is_empty()
            || self.config.gate_set.iter().any(|g| g == gate_name)
    }

    /// Return `true` when the pair `(q0, q1)` is a valid 2-qubit connection.
    ///
    /// An empty connectivity list (the default) means all pairs are allowed.
    pub fn is_connected(&self, q0: usize, q1: usize) -> bool {
        self.config.connectivity.is_empty()
            || self
                .config
                .connectivity
                .iter()
                .any(|&(a, b)| (a == q0 && b == q1) || (a == q1 && b == q0))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const BELL_QASM: &str = "OPENQASM 2.0;\n\
        include \"qelib1.inc\";\n\
        qreg q[2];\n\
        creg c[2];\n\
        h q[0];\n\
        cx q[0],q[1];\n\
        measure q[0] -> c[0];\n\
        measure q[1] -> c[1];\n";

    #[test]
    fn test_default_config() {
        let cfg = MockBackendConfig::default();
        assert_eq!(cfg.name, "mock_backend");
        assert_eq!(cfg.max_qubits, 32);
        assert_eq!(cfg.max_shots, 8192);
        assert_eq!(cfg.error_rate, 0.0);
        assert_eq!(cfg.fail_rate, 0.0);
    }

    #[test]
    fn test_parse_qubit_count() {
        let backend = MockQuantumBackend::new(MockBackendConfig::default());
        assert_eq!(backend.parse_qubit_count(BELL_QASM), Some(2));
        assert_eq!(backend.parse_qubit_count("no qreg here"), None);
        assert_eq!(backend.parse_qubit_count("qreg q[5];"), Some(5));
    }

    #[test]
    fn test_counts_sum_to_shots() {
        let backend = MockQuantumBackend::new(MockBackendConfig::perfect(3));
        let shots = 1024;
        let counts = backend.run(BELL_QASM, shots).expect("run succeeded");
        let total: usize = counts.values().sum();
        assert_eq!(total, shots);
    }

    #[test]
    fn test_capabilities_contains_name() {
        let backend = MockQuantumBackend::new(MockBackendConfig::ibm_nairobi_like());
        let caps = backend.capabilities();
        assert_eq!(caps.get("name").map(String::as_str), Some("mock_ibm_nairobi"));
        assert_eq!(caps.get("n_qubits").map(String::as_str), Some("7"));
    }

    #[test]
    fn test_supports_gate_empty_means_all() {
        let backend = MockQuantumBackend::new(MockBackendConfig::default());
        assert!(backend.supports_gate("any_gate"));
        assert!(backend.supports_gate("cx"));
    }

    #[test]
    fn test_supports_gate_restricted() {
        let backend = MockQuantumBackend::new(MockBackendConfig::ibm_nairobi_like());
        assert!(backend.supports_gate("cx"));
        assert!(!backend.supports_gate("ccx"));
    }
}
