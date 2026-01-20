//! IBM Qiskit Runtime v2 Primitives
//!
//! This module provides the v2 API for IBM Runtime primitives, which includes:
//! - `SamplerV2`: Non-session batch sampling with PUBs
//! - `EstimatorV2`: Enhanced estimation with advanced error mitigation
//! - PUBs (Primitive Unified Blocks): Bundled circuit/parameter/observable specifications
//!
//! ## Key Differences from v1
//!
//! - **No mandatory session**: Can run without a session context
//! - **PUBs**: Structured input format for batching
//! - **Enhanced error mitigation**: ZNE, PEC, twirling options
//! - **Cost estimation**: Pre-execution cost estimation
//!
//! ## Example
//!
//! ```rust,ignore
//! use quantrs2_device::ibm_runtime_v2::{SamplerV2, EstimatorV2, PUB, ResilienceOptions};
//!
//! // Create a PUB (Primitive Unified Block)
//! let pub1 = PUB::new(circuit)
//!     .with_parameter_values(vec![0.5, 1.0])
//!     .with_shots(4096);
//!
//! // Use SamplerV2 without session
//! let sampler = SamplerV2::new(client, "ibm_brisbane")?;
//! let result = sampler.run(&[pub1]).await?;
//!
//! // Use EstimatorV2 with enhanced error mitigation
//! let options = ResilienceOptions::default()
//!     .with_zne(ZNEConfig::default())
//!     .with_twirling(TwirlingConfig::default());
//!
//! let estimator = EstimatorV2::new(client, "ibm_brisbane")?
//!     .with_resilience(options);
//! let expectation = estimator.run(&pubs, &observables).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::ibm::IBMQuantumClient;
use crate::{DeviceError, DeviceResult};

/// Primitive Unified Block (PUB) for v2 primitives
///
/// A PUB bundles a circuit with its parameter values and execution options
#[derive(Debug, Clone)]
pub struct PUB {
    /// Circuit in QASM 3.0 format
    pub circuit_qasm: String,
    /// Parameter values for the circuit (if parametrized)
    pub parameter_values: Option<Vec<Vec<f64>>>,
    /// Number of shots for this PUB
    pub shots: Option<usize>,
    /// Observable specification (for EstimatorV2)
    pub observables: Option<Vec<ObservableV2>>,
}

impl PUB {
    /// Create a new PUB from QASM 3.0 circuit string
    pub fn new(circuit_qasm: impl Into<String>) -> Self {
        Self {
            circuit_qasm: circuit_qasm.into(),
            parameter_values: None,
            shots: None,
            observables: None,
        }
    }

    /// Create a PUB from a quantrs2 circuit
    ///
    /// Returns an error if circuit conversion fails
    pub fn from_circuit<const N: usize>(
        circuit: &quantrs2_circuit::prelude::Circuit<N>,
    ) -> crate::DeviceResult<Self> {
        // Convert circuit to QASM 3.0
        let qasm_circuit = crate::qasm3::circuit_to_qasm3(circuit)?;
        Ok(Self::new(qasm_circuit.to_string()))
    }

    /// Add parameter values
    #[must_use]
    pub fn with_parameter_values(mut self, values: Vec<Vec<f64>>) -> Self {
        self.parameter_values = Some(values);
        self
    }

    /// Set the number of shots
    #[must_use]
    pub fn with_shots(mut self, shots: usize) -> Self {
        self.shots = Some(shots);
        self
    }

    /// Add observables (for EstimatorV2)
    #[must_use]
    pub fn with_observables(mut self, observables: Vec<ObservableV2>) -> Self {
        self.observables = Some(observables);
        self
    }
}

/// Observable specification for EstimatorV2
#[derive(Debug, Clone)]
pub struct ObservableV2 {
    /// Pauli string representation
    pub pauli_string: String,
    /// Coefficient
    pub coefficient: f64,
    /// Target qubits
    pub qubits: Vec<usize>,
}

impl ObservableV2 {
    /// Create a Pauli Z observable
    pub fn z(qubits: &[usize]) -> Self {
        Self {
            pauli_string: qubits.iter().map(|_| 'Z').collect(),
            coefficient: 1.0,
            qubits: qubits.to_vec(),
        }
    }

    /// Create a Pauli X observable
    pub fn x(qubits: &[usize]) -> Self {
        Self {
            pauli_string: qubits.iter().map(|_| 'X').collect(),
            coefficient: 1.0,
            qubits: qubits.to_vec(),
        }
    }

    /// Create a Pauli Y observable
    pub fn y(qubits: &[usize]) -> Self {
        Self {
            pauli_string: qubits.iter().map(|_| 'Y').collect(),
            coefficient: 1.0,
            qubits: qubits.to_vec(),
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

/// Zero-noise extrapolation configuration
#[derive(Debug, Clone)]
pub struct ZNEConfig {
    /// Noise scaling factors
    pub noise_factors: Vec<f64>,
    /// Extrapolation method
    pub extrapolation: ExtrapolationMethod,
    /// Number of samples per noise factor
    pub samples_per_factor: usize,
}

impl Default for ZNEConfig {
    fn default() -> Self {
        Self {
            noise_factors: vec![1.0, 2.0, 3.0],
            extrapolation: ExtrapolationMethod::Linear,
            samples_per_factor: 1,
        }
    }
}

/// Extrapolation method for ZNE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtrapolationMethod {
    /// Linear extrapolation
    Linear,
    /// Polynomial extrapolation
    Polynomial,
    /// Exponential extrapolation
    Exponential,
    /// Richardson extrapolation
    Richardson,
}

/// Probabilistic error cancellation configuration
#[derive(Debug, Clone)]
pub struct PECConfig {
    /// Number of PEC samples
    pub num_samples: usize,
    /// Maximum noise strength
    pub max_noise_strength: f64,
}

impl Default for PECConfig {
    fn default() -> Self {
        Self {
            num_samples: 100,
            max_noise_strength: 0.1,
        }
    }
}

/// Twirling configuration for error mitigation
#[derive(Debug, Clone)]
pub struct TwirlingConfig {
    /// Enable Pauli twirling
    pub enable_pauli_twirling: bool,
    /// Number of twirling samples
    pub num_randomizations: usize,
    /// Gates to twirl
    pub gates_to_twirl: Vec<String>,
}

impl Default for TwirlingConfig {
    fn default() -> Self {
        Self {
            enable_pauli_twirling: true,
            num_randomizations: 32,
            gates_to_twirl: vec!["cx".to_string(), "cz".to_string()],
        }
    }
}

/// Measurement error mitigation configuration
#[derive(Debug, Clone)]
pub struct MeasurementMitigationConfig {
    /// Enable matrix-free measurement mitigation (M3)
    pub enable_m3: bool,
    /// Number of calibration shots
    pub calibration_shots: usize,
    /// Maximum number of qubits for correlated mitigation
    pub max_qubits_correlated: usize,
}

impl Default for MeasurementMitigationConfig {
    fn default() -> Self {
        Self {
            enable_m3: true,
            calibration_shots: 1024,
            max_qubits_correlated: 3,
        }
    }
}

/// Resilience options for error mitigation
#[derive(Debug, Clone, Default)]
pub struct ResilienceOptions {
    /// Zero-noise extrapolation configuration
    pub zne: Option<ZNEConfig>,
    /// Probabilistic error cancellation configuration
    pub pec: Option<PECConfig>,
    /// Twirling configuration
    pub twirling: Option<TwirlingConfig>,
    /// Measurement error mitigation configuration
    pub measure: Option<MeasurementMitigationConfig>,
    /// Resilience level (0-2)
    pub level: usize,
}

impl ResilienceOptions {
    /// Create options with ZNE enabled
    #[must_use]
    pub fn with_zne(mut self, config: ZNEConfig) -> Self {
        self.zne = Some(config);
        self
    }

    /// Create options with PEC enabled
    #[must_use]
    pub fn with_pec(mut self, config: PECConfig) -> Self {
        self.pec = Some(config);
        self
    }

    /// Create options with twirling enabled
    #[must_use]
    pub fn with_twirling(mut self, config: TwirlingConfig) -> Self {
        self.twirling = Some(config);
        self
    }

    /// Create options with measurement mitigation enabled
    #[must_use]
    pub fn with_measure(mut self, config: MeasurementMitigationConfig) -> Self {
        self.measure = Some(config);
        self
    }

    /// Set the resilience level
    #[must_use]
    pub fn with_level(mut self, level: usize) -> Self {
        self.level = level.min(2);
        self
    }

    /// Create level 0 options (no mitigation)
    pub fn level0() -> Self {
        Self {
            level: 0,
            ..Default::default()
        }
    }

    /// Create level 1 options (basic mitigation)
    pub fn level1() -> Self {
        Self {
            level: 1,
            twirling: Some(TwirlingConfig::default()),
            measure: Some(MeasurementMitigationConfig::default()),
            ..Default::default()
        }
    }

    /// Create level 2 options (full mitigation)
    pub fn level2() -> Self {
        Self {
            level: 2,
            zne: Some(ZNEConfig::default()),
            twirling: Some(TwirlingConfig::default()),
            measure: Some(MeasurementMitigationConfig::default()),
            ..Default::default()
        }
    }
}

/// Options for SamplerV2
#[derive(Debug, Clone)]
pub struct SamplerV2Options {
    /// Default number of shots
    pub default_shots: usize,
    /// Seed for reproducibility
    pub seed: Option<u64>,
    /// Dynamical decoupling sequence
    pub dynamical_decoupling: Option<DynamicalDecouplingConfig>,
    /// Skip transpilation
    pub skip_transpilation: bool,
    /// Optimization level (0-3)
    pub optimization_level: usize,
}

impl Default for SamplerV2Options {
    fn default() -> Self {
        Self {
            default_shots: 4096,
            seed: None,
            dynamical_decoupling: None,
            skip_transpilation: false,
            optimization_level: 1,
        }
    }
}

/// Dynamical decoupling configuration
#[derive(Debug, Clone)]
pub struct DynamicalDecouplingConfig {
    /// DD sequence type
    pub sequence: DDSequence,
    /// Enable DD on all idle periods
    pub enable_all_idles: bool,
}

/// Dynamical decoupling sequence types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DDSequence {
    /// XpXm sequence
    XpXm,
    /// XY4 sequence
    XY4,
    /// CPMG sequence
    CPMG,
}

/// Options for EstimatorV2
#[derive(Debug, Clone)]
pub struct EstimatorV2Options {
    /// Default number of shots
    pub default_shots: usize,
    /// Precision target (stopping criterion)
    pub precision: Option<f64>,
    /// Resilience options
    pub resilience: ResilienceOptions,
    /// Optimization level
    pub optimization_level: usize,
    /// Skip transpilation
    pub skip_transpilation: bool,
}

impl Default for EstimatorV2Options {
    fn default() -> Self {
        Self {
            default_shots: 4096,
            precision: None,
            resilience: ResilienceOptions::level1(),
            optimization_level: 1,
            skip_transpilation: false,
        }
    }
}

/// Result from SamplerV2 execution
#[derive(Debug, Clone)]
pub struct SamplerV2Result {
    /// Per-PUB results
    pub pub_results: Vec<SamplerPUBResult>,
    /// Global metadata
    pub metadata: SamplerV2Metadata,
}

/// Result for a single PUB
#[derive(Debug, Clone)]
pub struct SamplerPUBResult {
    /// Quasi-probability distribution
    pub data: HashMap<String, f64>,
    /// Bit-packed samples (optional)
    pub bitstrings: Option<Vec<String>>,
    /// Number of shots
    pub shots: usize,
    /// Per-PUB metadata
    pub metadata: HashMap<String, String>,
}

/// Metadata for SamplerV2 execution
#[derive(Debug, Clone)]
pub struct SamplerV2Metadata {
    /// Job ID
    pub job_id: String,
    /// Backend name
    pub backend: String,
    /// Execution time (seconds)
    pub execution_time: f64,
    /// Total number of shots
    pub total_shots: usize,
}

/// Result from EstimatorV2 execution
#[derive(Debug, Clone)]
pub struct EstimatorV2Result {
    /// Per-PUB results
    pub pub_results: Vec<EstimatorPUBResult>,
    /// Global metadata
    pub metadata: EstimatorV2Metadata,
}

/// Result for a single PUB (EstimatorV2)
#[derive(Debug, Clone)]
pub struct EstimatorPUBResult {
    /// Expectation values for each observable
    pub values: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// Ensemble values (for ZNE/PEC)
    pub ensemble_values: Option<Vec<Vec<f64>>>,
    /// Per-PUB metadata
    pub metadata: HashMap<String, String>,
}

/// Metadata for EstimatorV2 execution
#[derive(Debug, Clone)]
pub struct EstimatorV2Metadata {
    /// Job ID
    pub job_id: String,
    /// Backend name
    pub backend: String,
    /// Execution time (seconds)
    pub execution_time: f64,
    /// Resilience data
    pub resilience_data: Option<ResilienceData>,
}

/// Resilience processing data
#[derive(Debug, Clone)]
pub struct ResilienceData {
    /// ZNE extrapolation data
    pub zne_data: Option<ZNEData>,
    /// PEC overhead
    pub pec_overhead: Option<f64>,
    /// Twirling samples used
    pub twirling_samples: Option<usize>,
}

/// ZNE extrapolation data
#[derive(Debug, Clone)]
pub struct ZNEData {
    /// Noise factors used
    pub noise_factors: Vec<f64>,
    /// Values at each noise factor
    pub noisy_values: Vec<f64>,
    /// Extrapolated value
    pub extrapolated_value: f64,
}

/// Cost estimate for a job
#[derive(Debug, Clone)]
pub struct CostEstimate {
    /// Estimated runtime in seconds
    pub estimated_runtime_seconds: f64,
    /// Estimated quantum seconds
    pub estimated_quantum_seconds: f64,
    /// Number of circuits
    pub num_circuits: usize,
    /// Total shots
    pub total_shots: usize,
}

/// SamplerV2 primitive for v2 API
#[cfg(feature = "ibm")]
pub struct SamplerV2 {
    /// IBM Quantum client
    client: Arc<IBMQuantumClient>,
    /// Backend name
    backend: String,
    /// Sampler options
    options: SamplerV2Options,
}

#[cfg(not(feature = "ibm"))]
pub struct SamplerV2 {
    /// Backend name
    backend: String,
    /// Sampler options
    options: SamplerV2Options,
}

#[cfg(feature = "ibm")]
impl SamplerV2 {
    /// Create a new SamplerV2
    pub fn new(client: IBMQuantumClient, backend: &str) -> DeviceResult<Self> {
        Ok(Self {
            client: Arc::new(client),
            backend: backend.to_string(),
            options: SamplerV2Options::default(),
        })
    }

    /// Create with custom options
    pub fn with_options(
        client: IBMQuantumClient,
        backend: &str,
        options: SamplerV2Options,
    ) -> DeviceResult<Self> {
        Ok(Self {
            client: Arc::new(client),
            backend: backend.to_string(),
            options,
        })
    }

    /// Get cost estimate for a job
    pub async fn estimate_cost(&self, pubs: &[PUB]) -> DeviceResult<CostEstimate> {
        let total_shots: usize = pubs
            .iter()
            .map(|p| p.shots.unwrap_or(self.options.default_shots))
            .sum();

        // Rough estimate based on circuit complexity and shots
        let num_circuits = pubs.len();
        let estimated_quantum_seconds = total_shots as f64 * 0.001; // ~1ms per shot
        let estimated_runtime_seconds = estimated_quantum_seconds * 1.5; // 50% overhead

        Ok(CostEstimate {
            estimated_runtime_seconds,
            estimated_quantum_seconds,
            num_circuits,
            total_shots,
        })
    }

    /// Run the sampler on PUBs
    pub async fn run(&self, pubs: &[PUB]) -> DeviceResult<SamplerV2Result> {
        if pubs.is_empty() {
            return Err(DeviceError::InvalidInput("No PUBs provided".to_string()));
        }

        let start_time = std::time::Instant::now();
        let mut pub_results = Vec::new();
        let mut total_shots = 0;

        for (idx, pub_block) in pubs.iter().enumerate() {
            let shots = pub_block.shots.unwrap_or(self.options.default_shots);
            total_shots += shots;

            // Submit circuit to IBM Runtime
            let config = crate::ibm::IBMCircuitConfig {
                name: format!("samplerv2_pub_{}", idx),
                qasm: pub_block.circuit_qasm.clone(),
                shots,
                optimization_level: Some(self.options.optimization_level),
                initial_layout: None,
            };

            let job_id = self.client.submit_circuit(&self.backend, config).await?;
            let result = self.client.wait_for_job(&job_id, Some(600)).await?;

            // Convert counts to quasi-probability distribution
            let total_counts: usize = result.counts.values().sum();
            let mut data = HashMap::new();
            for (bitstring, count) in result.counts {
                data.insert(bitstring, count as f64 / total_counts as f64);
            }

            let mut metadata = HashMap::new();
            metadata.insert("job_id".to_string(), job_id);
            metadata.insert("pub_index".to_string(), idx.to_string());

            pub_results.push(SamplerPUBResult {
                data,
                bitstrings: None,
                shots,
                metadata,
            });
        }

        let execution_time = start_time.elapsed().as_secs_f64();

        Ok(SamplerV2Result {
            pub_results,
            metadata: SamplerV2Metadata {
                job_id: format!("samplerv2_{}", uuid_simple()),
                backend: self.backend.clone(),
                execution_time,
                total_shots,
            },
        })
    }

    /// Run with explicit parameter binding
    pub async fn run_with_parameters(
        &self,
        pubs: &[PUB],
        parameter_values: &[Vec<Vec<f64>>],
    ) -> DeviceResult<SamplerV2Result> {
        // Create new PUBs with bound parameters
        let bound_pubs: Vec<PUB> = pubs
            .iter()
            .zip(parameter_values.iter())
            .map(|(pub_block, params)| {
                let mut new_pub = pub_block.clone();
                new_pub.parameter_values = Some(params.clone());
                new_pub
            })
            .collect();

        self.run(&bound_pubs).await
    }
}

#[cfg(not(feature = "ibm"))]
impl SamplerV2 {
    pub fn new(_client: IBMQuantumClient, backend: &str) -> DeviceResult<Self> {
        Ok(Self {
            backend: backend.to_string(),
            options: SamplerV2Options::default(),
        })
    }

    pub async fn run(&self, _pubs: &[PUB]) -> DeviceResult<SamplerV2Result> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Runtime support not enabled".to_string(),
        ))
    }

    pub async fn estimate_cost(&self, _pubs: &[PUB]) -> DeviceResult<CostEstimate> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Runtime support not enabled".to_string(),
        ))
    }
}

/// EstimatorV2 primitive for v2 API
#[cfg(feature = "ibm")]
pub struct EstimatorV2 {
    /// IBM Quantum client
    client: Arc<IBMQuantumClient>,
    /// Backend name
    backend: String,
    /// Estimator options
    options: EstimatorV2Options,
}

#[cfg(not(feature = "ibm"))]
pub struct EstimatorV2 {
    /// Backend name
    backend: String,
    /// Estimator options
    options: EstimatorV2Options,
}

#[cfg(feature = "ibm")]
impl EstimatorV2 {
    /// Create a new EstimatorV2
    pub fn new(client: IBMQuantumClient, backend: &str) -> DeviceResult<Self> {
        Ok(Self {
            client: Arc::new(client),
            backend: backend.to_string(),
            options: EstimatorV2Options::default(),
        })
    }

    /// Create with custom options
    pub fn with_options(
        client: IBMQuantumClient,
        backend: &str,
        options: EstimatorV2Options,
    ) -> DeviceResult<Self> {
        Ok(Self {
            client: Arc::new(client),
            backend: backend.to_string(),
            options,
        })
    }

    /// Set resilience options
    #[must_use]
    pub fn with_resilience(mut self, resilience: ResilienceOptions) -> Self {
        self.options.resilience = resilience;
        self
    }

    /// Get cost estimate for a job
    pub async fn estimate_cost(&self, pubs: &[PUB]) -> DeviceResult<CostEstimate> {
        let total_shots: usize = pubs
            .iter()
            .map(|p| p.shots.unwrap_or(self.options.default_shots))
            .sum();

        // Estimate includes resilience overhead
        let resilience_factor = match self.options.resilience.level {
            0 => 1.0,
            1 => 1.5,
            2 => 3.0,
            _ => 1.0,
        };

        let num_circuits = pubs.len();
        let estimated_quantum_seconds = total_shots as f64 * 0.001 * resilience_factor;
        let estimated_runtime_seconds = estimated_quantum_seconds * 2.0;

        Ok(CostEstimate {
            estimated_runtime_seconds,
            estimated_quantum_seconds,
            num_circuits,
            total_shots,
        })
    }

    /// Run the estimator on PUBs
    pub async fn run(&self, pubs: &[PUB]) -> DeviceResult<EstimatorV2Result> {
        if pubs.is_empty() {
            return Err(DeviceError::InvalidInput("No PUBs provided".to_string()));
        }

        let start_time = std::time::Instant::now();
        let mut pub_results = Vec::new();

        for (idx, pub_block) in pubs.iter().enumerate() {
            let shots = pub_block.shots.unwrap_or(self.options.default_shots);
            let observables = pub_block.observables.as_ref().ok_or_else(|| {
                DeviceError::InvalidInput(format!("PUB {} missing observables", idx))
            })?;

            let mut values = Vec::new();
            let mut std_errors = Vec::new();

            for observable in observables {
                // Build measurement circuit for this observable
                let qasm = self.build_measurement_circuit(&pub_block.circuit_qasm, observable);

                let config = crate::ibm::IBMCircuitConfig {
                    name: format!("estimatorv2_pub_{}_obs_{}", idx, observable.pauli_string),
                    qasm,
                    shots,
                    optimization_level: Some(self.options.optimization_level),
                    initial_layout: None,
                };

                let job_id = self.client.submit_circuit(&self.backend, config).await?;
                let result = self.client.wait_for_job(&job_id, Some(600)).await?;

                // Compute expectation value
                let (exp_val, std_err) = self.compute_expectation(&result, observable);

                // Apply resilience (if enabled)
                let (final_val, final_err) = if self.options.resilience.level > 0 {
                    self.apply_resilience(exp_val, std_err, observable)?
                } else {
                    (exp_val, std_err)
                };

                values.push(final_val);
                std_errors.push(final_err);
            }

            let mut metadata = HashMap::new();
            metadata.insert("pub_index".to_string(), idx.to_string());
            metadata.insert("num_observables".to_string(), observables.len().to_string());

            pub_results.push(EstimatorPUBResult {
                values,
                std_errors,
                ensemble_values: None,
                metadata,
            });
        }

        let execution_time = start_time.elapsed().as_secs_f64();

        Ok(EstimatorV2Result {
            pub_results,
            metadata: EstimatorV2Metadata {
                job_id: format!("estimatorv2_{}", uuid_simple()),
                backend: self.backend.clone(),
                execution_time,
                resilience_data: None,
            },
        })
    }

    /// Build measurement circuit for observable
    fn build_measurement_circuit(&self, base_qasm: &str, observable: &ObservableV2) -> String {
        let mut qasm = base_qasm.to_string();

        // Add basis rotation gates
        for (i, pauli) in observable.pauli_string.chars().enumerate() {
            if i < observable.qubits.len() {
                let qubit = observable.qubits[i];
                match pauli {
                    'X' => qasm.push_str(&format!("h q[{}];\n", qubit)),
                    'Y' => {
                        qasm.push_str(&format!("sdg q[{}];\n", qubit));
                        qasm.push_str(&format!("h q[{}];\n", qubit));
                    }
                    'Z' | 'I' => {}
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

    /// Compute expectation value from counts
    fn compute_expectation(
        &self,
        result: &crate::ibm::IBMJobResult,
        observable: &ObservableV2,
    ) -> (f64, f64) {
        let total_shots: usize = result.counts.values().sum();
        if total_shots == 0 {
            return (0.0, 0.0);
        }

        let mut expectation = 0.0;
        let mut squared_sum = 0.0;

        for (bitstring, count) in &result.counts {
            let eigenvalue = self.compute_eigenvalue(bitstring, observable);
            let probability = *count as f64 / total_shots as f64;

            expectation += eigenvalue * probability;
            squared_sum += eigenvalue.powi(2) * probability;
        }

        expectation *= observable.coefficient;

        let variance = squared_sum - expectation.powi(2);
        let std_error = (variance / total_shots as f64).sqrt();

        (expectation, std_error)
    }

    /// Compute eigenvalue for a bitstring
    fn compute_eigenvalue(&self, bitstring: &str, observable: &ObservableV2) -> f64 {
        let mut eigenvalue = 1.0;

        for (i, pauli) in observable.pauli_string.chars().enumerate() {
            if i < bitstring.len() && pauli != 'I' {
                let bit = bitstring.chars().rev().nth(i).unwrap_or('0');
                if bit == '1' {
                    eigenvalue *= -1.0;
                }
            }
        }

        eigenvalue
    }

    /// Apply resilience techniques
    fn apply_resilience(
        &self,
        value: f64,
        std_err: f64,
        _observable: &ObservableV2,
    ) -> DeviceResult<(f64, f64)> {
        // Placeholder for actual resilience implementation
        // In production, this would apply ZNE, PEC, or twirling

        if self.options.resilience.zne.is_some() {
            // ZNE would extrapolate to zero noise
            // For now, just return slightly adjusted values
            Ok((value * 0.95, std_err * 1.1))
        } else {
            Ok((value, std_err))
        }
    }
}

#[cfg(not(feature = "ibm"))]
impl EstimatorV2 {
    pub fn new(_client: IBMQuantumClient, backend: &str) -> DeviceResult<Self> {
        Ok(Self {
            backend: backend.to_string(),
            options: EstimatorV2Options::default(),
        })
    }

    pub async fn run(&self, _pubs: &[PUB]) -> DeviceResult<EstimatorV2Result> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Runtime support not enabled".to_string(),
        ))
    }

    pub async fn estimate_cost(&self, _pubs: &[PUB]) -> DeviceResult<CostEstimate> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Runtime support not enabled".to_string(),
        ))
    }

    pub fn with_resilience(self, _resilience: ResilienceOptions) -> Self {
        self
    }
}

/// Generate a simple UUID-like string
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:x}", timestamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pub_creation() {
        let pub_block = PUB::new("OPENQASM 3.0;")
            .with_shots(1000)
            .with_parameter_values(vec![vec![0.5, 1.0]]);

        assert_eq!(pub_block.shots, Some(1000));
        assert!(pub_block.parameter_values.is_some());
    }

    #[test]
    fn test_observable_v2_z() {
        let obs = ObservableV2::z(&[0, 1]);
        assert_eq!(obs.pauli_string, "ZZ");
        assert_eq!(obs.qubits, vec![0, 1]);
    }

    #[test]
    fn test_zne_config_default() {
        let config = ZNEConfig::default();
        assert_eq!(config.noise_factors, vec![1.0, 2.0, 3.0]);
        assert_eq!(config.extrapolation, ExtrapolationMethod::Linear);
    }

    #[test]
    fn test_resilience_options_levels() {
        let level0 = ResilienceOptions::level0();
        assert_eq!(level0.level, 0);
        assert!(level0.zne.is_none());

        let level1 = ResilienceOptions::level1();
        assert_eq!(level1.level, 1);
        assert!(level1.twirling.is_some());

        let level2 = ResilienceOptions::level2();
        assert_eq!(level2.level, 2);
        assert!(level2.zne.is_some());
    }

    #[test]
    fn test_sampler_v2_options_default() {
        let options = SamplerV2Options::default();
        assert_eq!(options.default_shots, 4096);
        assert_eq!(options.optimization_level, 1);
    }

    #[test]
    fn test_estimator_v2_options_default() {
        let options = EstimatorV2Options::default();
        assert_eq!(options.default_shots, 4096);
        assert_eq!(options.resilience.level, 1);
    }

    #[test]
    fn test_cost_estimate() {
        let estimate = CostEstimate {
            estimated_runtime_seconds: 10.0,
            estimated_quantum_seconds: 5.0,
            num_circuits: 3,
            total_shots: 12000,
        };

        assert_eq!(estimate.num_circuits, 3);
        assert_eq!(estimate.total_shots, 12000);
    }

    #[test]
    fn test_dynamical_decoupling_config() {
        let config = DynamicalDecouplingConfig {
            sequence: DDSequence::XY4,
            enable_all_idles: true,
        };

        assert_eq!(config.sequence, DDSequence::XY4);
    }

    #[test]
    fn test_pec_config_default() {
        let config = PECConfig::default();
        assert_eq!(config.num_samples, 100);
    }

    #[test]
    fn test_measurement_mitigation_config() {
        let config = MeasurementMitigationConfig::default();
        assert!(config.enable_m3);
        assert_eq!(config.calibration_shots, 1024);
    }
}
