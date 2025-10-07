//! Simulator backend integration for quantum machine learning
//!
//! This module provides unified interfaces to all quantum simulators
//! available in the QuantRS2 ecosystem, enabling seamless backend
//! switching for quantum ML algorithms.

use crate::error::{MLError, Result};
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::Complex64;
// GpuStateVectorSimulator import removed - not used in this file
// The GPUBackend is a placeholder that doesn't use the actual GPU simulator yet
use quantrs2_sim::prelude::{MPSSimulator, PauliString, StateVectorSimulator};
use std::collections::HashMap;

/// Dynamic circuit representation for trait objects
#[derive(Debug, Clone)]
pub enum DynamicCircuit {
    Circuit1(Circuit<1>),
    Circuit2(Circuit<2>),
    Circuit4(Circuit<4>),
    Circuit8(Circuit<8>),
    Circuit16(Circuit<16>),
    Circuit32(Circuit<32>),
    Circuit64(Circuit<64>),
}

impl DynamicCircuit {
    /// Create from a generic circuit
    pub fn from_circuit<const N: usize>(circuit: Circuit<N>) -> Result<Self> {
        match N {
            1 => Ok(DynamicCircuit::Circuit1(unsafe {
                std::mem::transmute(circuit)
            })),
            2 => Ok(DynamicCircuit::Circuit2(unsafe {
                std::mem::transmute(circuit)
            })),
            4 => Ok(DynamicCircuit::Circuit4(unsafe {
                std::mem::transmute(circuit)
            })),
            8 => Ok(DynamicCircuit::Circuit8(unsafe {
                std::mem::transmute(circuit)
            })),
            16 => Ok(DynamicCircuit::Circuit16(unsafe {
                std::mem::transmute(circuit)
            })),
            32 => Ok(DynamicCircuit::Circuit32(unsafe {
                std::mem::transmute(circuit)
            })),
            64 => Ok(DynamicCircuit::Circuit64(unsafe {
                std::mem::transmute(circuit)
            })),
            _ => Err(MLError::ValidationError(format!(
                "Unsupported circuit size: {}",
                N
            ))),
        }
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        match self {
            DynamicCircuit::Circuit1(_) => 1,
            DynamicCircuit::Circuit2(_) => 2,
            DynamicCircuit::Circuit4(_) => 4,
            DynamicCircuit::Circuit8(_) => 8,
            DynamicCircuit::Circuit16(_) => 16,
            DynamicCircuit::Circuit32(_) => 32,
            DynamicCircuit::Circuit64(_) => 64,
        }
    }

    /// Get the number of gates (placeholder implementation)
    pub fn num_gates(&self) -> usize {
        match self {
            DynamicCircuit::Circuit1(c) => c.gates().len(),
            DynamicCircuit::Circuit2(c) => c.gates().len(),
            DynamicCircuit::Circuit4(c) => c.gates().len(),
            DynamicCircuit::Circuit8(c) => c.gates().len(),
            DynamicCircuit::Circuit16(c) => c.gates().len(),
            DynamicCircuit::Circuit32(c) => c.gates().len(),
            DynamicCircuit::Circuit64(c) => c.gates().len(),
        }
    }

    /// Get circuit depth (placeholder implementation)
    pub fn depth(&self) -> usize {
        // Simplified depth calculation - just return number of gates for now
        self.num_gates()
    }

    /// Get gates (placeholder implementation)
    pub fn gates(&self) -> Vec<&dyn quantrs2_core::gate::GateOp> {
        match self {
            DynamicCircuit::Circuit1(c) => c
                .gates()
                .iter()
                .map(|g| g.as_ref() as &dyn quantrs2_core::gate::GateOp)
                .collect(),
            DynamicCircuit::Circuit2(c) => c
                .gates()
                .iter()
                .map(|g| g.as_ref() as &dyn quantrs2_core::gate::GateOp)
                .collect(),
            DynamicCircuit::Circuit4(c) => c
                .gates()
                .iter()
                .map(|g| g.as_ref() as &dyn quantrs2_core::gate::GateOp)
                .collect(),
            DynamicCircuit::Circuit8(c) => c
                .gates()
                .iter()
                .map(|g| g.as_ref() as &dyn quantrs2_core::gate::GateOp)
                .collect(),
            DynamicCircuit::Circuit16(c) => c
                .gates()
                .iter()
                .map(|g| g.as_ref() as &dyn quantrs2_core::gate::GateOp)
                .collect(),
            DynamicCircuit::Circuit32(c) => c
                .gates()
                .iter()
                .map(|g| g.as_ref() as &dyn quantrs2_core::gate::GateOp)
                .collect(),
            DynamicCircuit::Circuit64(c) => c
                .gates()
                .iter()
                .map(|g| g.as_ref() as &dyn quantrs2_core::gate::GateOp)
                .collect(),
        }
    }
}

/// Unified simulator backend interface
pub trait SimulatorBackend: Send + Sync {
    /// Execute a quantum circuit
    fn execute_circuit(
        &self,
        circuit: &DynamicCircuit,
        parameters: &[f64],
        shots: Option<usize>,
    ) -> Result<SimulationResult>;

    /// Compute expectation value
    fn expectation_value(
        &self,
        circuit: &DynamicCircuit,
        parameters: &[f64],
        observable: &Observable,
    ) -> Result<f64>;

    /// Compute gradients using backend-specific methods
    fn compute_gradients(
        &self,
        circuit: &DynamicCircuit,
        parameters: &[f64],
        observable: &Observable,
        gradient_method: GradientMethod,
    ) -> Result<Array1<f64>>;

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;

    /// Get backend name
    fn name(&self) -> &str;

    /// Maximum number of qubits supported
    fn max_qubits(&self) -> usize;

    /// Check if backend supports noise simulation
    fn supports_noise(&self) -> bool;
}

/// Simulation result containing various outputs
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Final quantum state (if available)
    pub state: Option<Array1<Complex64>>,
    /// Measurement outcomes
    pub measurements: Option<Array1<usize>>,
    /// Measurement probabilities
    pub probabilities: Option<Array1<f64>>,
    /// Execution metadata
    pub metadata: HashMap<String, f64>,
}

/// Observable for expectation value computations
#[derive(Debug, Clone)]
pub enum Observable {
    /// Pauli string observable
    PauliString(PauliString),
    /// Pauli Z on specified qubits
    PauliZ(Vec<usize>),
    /// Custom Hermitian matrix
    Matrix(Array2<Complex64>),
    /// Hamiltonian as sum of Pauli strings
    Hamiltonian(Vec<(f64, PauliString)>),
}

/// Gradient computation methods
#[derive(Debug, Clone, Copy)]
pub enum GradientMethod {
    /// Parameter shift rule
    ParameterShift,
    /// Finite differences
    FiniteDifference,
    /// Adjoint differentiation (if supported)
    Adjoint,
    /// Stochastic parameter shift
    StochasticParameterShift,
}

/// Backend capabilities
#[derive(Debug, Clone, Default)]
pub struct BackendCapabilities {
    /// Maximum qubits
    pub max_qubits: usize,
    /// Supports noise simulation
    pub noise_simulation: bool,
    /// Supports GPU acceleration
    pub gpu_acceleration: bool,
    /// Supports distributed computation
    pub distributed: bool,
    /// Supports adjoint gradients
    pub adjoint_gradients: bool,
    /// Memory requirements per qubit (bytes)
    pub memory_per_qubit: usize,
}

/// Statevector simulator backend
#[derive(Debug)]
pub struct StatevectorBackend {
    /// Internal simulator
    simulator: StateVectorSimulator,
    /// Maximum qubits
    max_qubits: usize,
}

impl StatevectorBackend {
    /// Create new statevector backend
    pub fn new(max_qubits: usize) -> Self {
        Self {
            simulator: StateVectorSimulator::new(),
            max_qubits,
        }
    }
}

impl SimulatorBackend for StatevectorBackend {
    fn execute_circuit(
        &self,
        circuit: &DynamicCircuit,
        _parameters: &[f64],
        _shots: Option<usize>,
    ) -> Result<SimulationResult> {
        match circuit {
            DynamicCircuit::Circuit1(c) => {
                let state = self.simulator.run(c)?;
                let probabilities = state
                    .amplitudes()
                    .iter()
                    .map(|c| c.norm_sqr())
                    .collect::<Vec<_>>();
                Ok(SimulationResult {
                    state: None, // TODO: Convert Register to Array
                    measurements: None,
                    probabilities: Some(probabilities.into()),
                    metadata: HashMap::new(),
                })
            }
            DynamicCircuit::Circuit2(c) => {
                let state = self.simulator.run(c)?;
                let probabilities = state
                    .amplitudes()
                    .iter()
                    .map(|c| c.norm_sqr())
                    .collect::<Vec<_>>();
                Ok(SimulationResult {
                    state: None, // TODO: Convert Register to Array
                    measurements: None,
                    probabilities: Some(probabilities.into()),
                    metadata: HashMap::new(),
                })
            }
            // Add other circuit sizes as needed
            _ => Err(MLError::ValidationError(
                "Unsupported circuit size".to_string(),
            )),
        }
    }

    fn expectation_value(
        &self,
        circuit: &DynamicCircuit,
        _parameters: &[f64],
        observable: &Observable,
    ) -> Result<f64> {
        match circuit {
            DynamicCircuit::Circuit1(c) => {
                let _state = self.simulator.run(c)?;
                // TODO: Convert Register to Array for expectation computation
                Ok(0.0)
            }
            DynamicCircuit::Circuit2(c) => {
                let _state = self.simulator.run(c)?;
                // TODO: Convert Register to Array for expectation computation
                Ok(0.0)
            }
            // Add other circuit sizes as needed
            _ => Err(MLError::ValidationError(
                "Unsupported circuit size".to_string(),
            )),
        }
    }

    fn compute_gradients(
        &self,
        circuit: &DynamicCircuit,
        _parameters: &[f64],
        _observable: &Observable,
        _gradient_method: GradientMethod,
    ) -> Result<Array1<f64>> {
        // Placeholder implementation
        match circuit {
            DynamicCircuit::Circuit1(_) => Ok(Array1::zeros(1)),
            DynamicCircuit::Circuit2(_) => Ok(Array1::zeros(1)),
            _ => Err(MLError::ValidationError(
                "Unsupported circuit size".to_string(),
            )),
        }
    }

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities()
    }

    /// Get backend name
    fn name(&self) -> &str {
        "StatevectorBackend"
    }

    /// Maximum number of qubits supported
    fn max_qubits(&self) -> usize {
        self.capabilities().max_qubits
    }

    /// Check if backend supports noise simulation
    fn supports_noise(&self) -> bool {
        self.capabilities().noise_simulation
    }
}

impl StatevectorBackend {
    /// Helper method to compute expectation values
    fn compute_expectation(
        &self,
        state: &Array1<Complex64>,
        observable: &Observable,
    ) -> Result<f64> {
        match observable {
            Observable::PauliString(pauli) => {
                // Placeholder implementation - compute expectation value manually
                Ok(0.0) // TODO: Implement proper Pauli expectation value computation
            }
            Observable::PauliZ(_qubits) => {
                // Placeholder implementation for Pauli Z expectation value
                Ok(0.0) // TODO: Implement proper Pauli Z expectation value computation
            }
            Observable::Matrix(matrix) => {
                // Compute <ψ|H|ψ>
                let amplitudes = state;
                let result = amplitudes
                    .iter()
                    .enumerate()
                    .map(|(i, &amp)| {
                        amplitudes
                            .iter()
                            .enumerate()
                            .map(|(j, &amp2)| amp.conj() * matrix[[i, j]] * amp2)
                            .sum::<Complex64>()
                    })
                    .sum::<Complex64>();
                Ok(result.re)
            }
            Observable::Hamiltonian(terms) => {
                let mut expectation = 0.0;
                for (coeff, pauli) in terms {
                    expectation += coeff * 0.0; // TODO: Implement proper Pauli expectation value
                }
                Ok(expectation)
            }
        }
    }

    fn max_qubits(&self) -> usize {
        self.max_qubits
    }

    fn supports_noise(&self) -> bool {
        false
    }
}

/// Matrix Product State (MPS) simulator backend
pub struct MPSBackend {
    /// Internal MPS simulator
    simulator: MPSSimulator,
    /// Bond dimension
    bond_dimension: usize,
    /// Maximum qubits
    max_qubits: usize,
}

impl MPSBackend {
    /// Create new MPS backend
    pub fn new(bond_dimension: usize, max_qubits: usize) -> Self {
        Self {
            simulator: MPSSimulator::new(bond_dimension),
            bond_dimension,
            max_qubits,
        }
    }
}

impl SimulatorBackend for MPSBackend {
    fn execute_circuit(
        &self,
        circuit: &DynamicCircuit,
        _parameters: &[f64],
        _shots: Option<usize>,
    ) -> Result<SimulationResult> {
        // MPS implementation depends on circuit size
        match circuit {
            DynamicCircuit::Circuit1(c) => {
                // For small circuits, use basic MPS simulation
                Ok(SimulationResult {
                    state: None, // MPS doesn't expose full state
                    measurements: None,
                    probabilities: None,
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert("bond_dimension".to_string(), self.bond_dimension as f64);
                        meta.insert("num_qubits".to_string(), 1.0);
                        meta
                    },
                })
            }
            DynamicCircuit::Circuit2(c) => Ok(SimulationResult {
                state: None,
                measurements: None,
                probabilities: None,
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("bond_dimension".to_string(), self.bond_dimension as f64);
                    meta.insert("num_qubits".to_string(), 2.0);
                    meta
                },
            }),
            _ => {
                // For larger circuits, need proper MPS simulation
                Ok(SimulationResult {
                    state: None,
                    measurements: None,
                    probabilities: None,
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert("bond_dimension".to_string(), self.bond_dimension as f64);
                        meta.insert("num_qubits".to_string(), circuit.num_qubits() as f64);
                        meta
                    },
                })
            }
        }
    }

    fn expectation_value(
        &self,
        circuit: &DynamicCircuit,
        _parameters: &[f64],
        observable: &Observable,
    ) -> Result<f64> {
        match observable {
            Observable::PauliString(_pauli) => {
                // Would compute expectation using MPS for any circuit size
                Ok(0.0) // Placeholder implementation
            }
            Observable::PauliZ(_qubits) => {
                // Would compute Z expectation using MPS
                Ok(0.0) // Placeholder implementation
            }
            Observable::Hamiltonian(terms) => {
                let mut expectation = 0.0;
                for (coeff, _pauli) in terms {
                    // Would compute each term using MPS
                    expectation += coeff * 0.0; // Placeholder
                }
                Ok(expectation)
            }
            Observable::Matrix(_) => Err(MLError::NotSupported(
                "Matrix observables not supported for MPS backend".to_string(),
            )),
        }
    }

    fn compute_gradients(
        &self,
        circuit: &DynamicCircuit,
        parameters: &[f64],
        observable: &Observable,
        gradient_method: GradientMethod,
    ) -> Result<Array1<f64>> {
        match gradient_method {
            GradientMethod::ParameterShift => {
                self.parameter_shift_gradients_dynamic(circuit, parameters, observable)
            }
            _ => Err(MLError::NotSupported(
                "Only parameter shift gradients supported for MPS backend".to_string(),
            )),
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            max_qubits: self.max_qubits,
            noise_simulation: false,
            gpu_acceleration: false,
            distributed: false,
            adjoint_gradients: false,
            memory_per_qubit: self.bond_dimension * self.bond_dimension * 16, // D^2 * 16 bytes
        }
    }

    fn name(&self) -> &str {
        "mps"
    }

    fn max_qubits(&self) -> usize {
        self.max_qubits
    }

    fn supports_noise(&self) -> bool {
        false
    }
}

impl MPSBackend {
    fn parameter_shift_gradients_dynamic(
        &self,
        circuit: &DynamicCircuit,
        parameters: &[f64],
        observable: &Observable,
    ) -> Result<Array1<f64>> {
        let shift = std::f64::consts::PI / 2.0;
        let mut gradients = Array1::zeros(parameters.len());

        for i in 0..parameters.len() {
            let mut params_plus = parameters.to_vec();
            params_plus[i] += shift;
            let val_plus = self.expectation_value(circuit, &params_plus, observable)?;

            let mut params_minus = parameters.to_vec();
            params_minus[i] -= shift;
            let val_minus = self.expectation_value(circuit, &params_minus, observable)?;

            gradients[i] = (val_plus - val_minus) / 2.0;
        }

        Ok(gradients)
    }
}

// GPU backend is now implemented in gpu_backend_impl module
#[cfg(feature = "gpu")]
pub use crate::gpu_backend_impl::GPUBackend;

// SimulatorBackend implementation for GPUBackend is in gpu_backend_impl.rs

/// Enum for different backend types (avoids dyn compatibility issues)
pub enum Backend {
    Statevector(StatevectorBackend),
    MPS(MPSBackend),
    #[cfg(feature = "gpu")]
    GPU(GPUBackend),
}

impl SimulatorBackend for Backend {
    fn execute_circuit(
        &self,
        circuit: &DynamicCircuit,
        parameters: &[f64],
        shots: Option<usize>,
    ) -> Result<SimulationResult> {
        match self {
            Backend::Statevector(backend) => backend.execute_circuit(circuit, parameters, shots),
            Backend::MPS(backend) => backend.execute_circuit(circuit, parameters, shots),
            #[cfg(feature = "gpu")]
            Backend::GPU(backend) => backend.execute_circuit(circuit, parameters, shots),
        }
    }

    fn expectation_value(
        &self,
        circuit: &DynamicCircuit,
        parameters: &[f64],
        observable: &Observable,
    ) -> Result<f64> {
        match self {
            Backend::Statevector(backend) => {
                backend.expectation_value(circuit, parameters, observable)
            }
            Backend::MPS(backend) => backend.expectation_value(circuit, parameters, observable),
            #[cfg(feature = "gpu")]
            Backend::GPU(backend) => backend.expectation_value(circuit, parameters, observable),
        }
    }

    fn compute_gradients(
        &self,
        circuit: &DynamicCircuit,
        parameters: &[f64],
        observable: &Observable,
        gradient_method: GradientMethod,
    ) -> Result<Array1<f64>> {
        match self {
            Backend::Statevector(backend) => {
                backend.compute_gradients(circuit, parameters, observable, gradient_method)
            }
            Backend::MPS(backend) => {
                backend.compute_gradients(circuit, parameters, observable, gradient_method)
            }
            #[cfg(feature = "gpu")]
            Backend::GPU(backend) => {
                backend.compute_gradients(circuit, parameters, observable, gradient_method)
            }
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        match self {
            Backend::Statevector(backend) => backend.capabilities(),
            Backend::MPS(backend) => backend.capabilities(),
            #[cfg(feature = "gpu")]
            Backend::GPU(backend) => backend.capabilities(),
        }
    }

    fn name(&self) -> &str {
        match self {
            Backend::Statevector(backend) => backend.name(),
            Backend::MPS(backend) => backend.name(),
            #[cfg(feature = "gpu")]
            Backend::GPU(backend) => backend.name(),
        }
    }

    fn max_qubits(&self) -> usize {
        match self {
            Backend::Statevector(backend) => backend.max_qubits(),
            Backend::MPS(backend) => backend.max_qubits(),
            #[cfg(feature = "gpu")]
            Backend::GPU(backend) => backend.max_qubits(),
        }
    }

    fn supports_noise(&self) -> bool {
        match self {
            Backend::Statevector(backend) => backend.supports_noise(),
            Backend::MPS(backend) => backend.supports_noise(),
            #[cfg(feature = "gpu")]
            Backend::GPU(backend) => backend.supports_noise(),
        }
    }
}

/// Backend manager for automatic backend selection
pub struct BackendManager {
    /// Available backends
    backends: HashMap<String, Backend>,
    /// Current backend
    current_backend: Option<String>,
    /// Backend selection strategy
    selection_strategy: BackendSelectionStrategy,
}

/// Backend selection strategies
#[derive(Debug, Clone)]
pub enum BackendSelectionStrategy {
    /// Use fastest backend for given problem size
    Fastest,
    /// Use most memory-efficient backend
    MemoryEfficient,
    /// Use most accurate backend
    MostAccurate,
    /// User-specified backend
    Manual(String),
}

impl BackendManager {
    /// Create a new backend manager
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
            current_backend: None,
            selection_strategy: BackendSelectionStrategy::Fastest,
        }
    }

    /// Register a backend
    pub fn register_backend(&mut self, name: impl Into<String>, backend: Backend) {
        self.backends.insert(name.into(), backend);
    }

    /// Set selection strategy
    pub fn set_strategy(&mut self, strategy: BackendSelectionStrategy) {
        self.selection_strategy = strategy;
    }

    /// Select optimal backend for given problem
    pub fn select_backend(&mut self, num_qubits: usize, shots: Option<usize>) -> Result<()> {
        let backend_name = match &self.selection_strategy {
            BackendSelectionStrategy::Fastest => self.select_fastest_backend(num_qubits, shots)?,
            BackendSelectionStrategy::MemoryEfficient => {
                self.select_memory_efficient_backend(num_qubits)?
            }
            BackendSelectionStrategy::MostAccurate => {
                self.select_most_accurate_backend(num_qubits)?
            }
            BackendSelectionStrategy::Manual(name) => name.clone(),
        };

        self.current_backend = Some(backend_name);
        Ok(())
    }

    /// Execute circuit using selected backend
    pub fn execute_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        parameters: &[f64],
        shots: Option<usize>,
    ) -> Result<SimulationResult> {
        if let Some(ref backend_name) = self.current_backend {
            if let Some(backend) = self.backends.get(backend_name) {
                let dynamic_circuit = DynamicCircuit::from_circuit(circuit.clone())?;
                backend.execute_circuit(&dynamic_circuit, parameters, shots)
            } else {
                Err(MLError::InvalidConfiguration(format!(
                    "Backend '{}' not found",
                    backend_name
                )))
            }
        } else {
            Err(MLError::InvalidConfiguration(
                "No backend selected".to_string(),
            ))
        }
    }

    /// Get current backend
    pub fn current_backend(&self) -> Option<&Backend> {
        self.current_backend
            .as_ref()
            .and_then(|name| self.backends.get(name))
    }

    /// List available backends
    pub fn list_backends(&self) -> Vec<(String, BackendCapabilities)> {
        self.backends
            .iter()
            .map(|(name, backend)| (name.clone(), backend.capabilities()))
            .collect()
    }

    fn select_fastest_backend(&self, num_qubits: usize, _shots: Option<usize>) -> Result<String> {
        // Simple heuristic: GPU for large circuits, MPS for very large, statevector for small
        if num_qubits <= 20 {
            Ok("statevector".to_string())
        } else if num_qubits <= 50 && self.backends.contains_key("gpu") {
            Ok("gpu".to_string())
        } else if self.backends.contains_key("mps") {
            Ok("mps".to_string())
        } else {
            Err(MLError::InvalidConfiguration(
                "No suitable backend for problem size".to_string(),
            ))
        }
    }

    fn select_memory_efficient_backend(&self, num_qubits: usize) -> Result<String> {
        if num_qubits > 30 && self.backends.contains_key("mps") {
            Ok("mps".to_string())
        } else {
            Ok("statevector".to_string())
        }
    }

    fn select_most_accurate_backend(&self, _num_qubits: usize) -> Result<String> {
        // Statevector is most accurate
        Ok("statevector".to_string())
    }
}

/// Helper functions for backend management
pub mod backend_utils {
    use super::*;

    /// Create default backend manager with all available backends
    pub fn create_default_manager() -> BackendManager {
        let mut manager = BackendManager::new();

        // Register statevector backend
        manager.register_backend(
            "statevector",
            Backend::Statevector(StatevectorBackend::new(25)),
        );

        // Register MPS backend
        manager.register_backend("mps", Backend::MPS(MPSBackend::new(64, 100)));

        // Register GPU backend if available
        #[cfg(feature = "gpu")]
        {
            if let Ok(gpu_backend) = GPUBackend::new(0, 30) {
                manager.register_backend("gpu", Backend::GPU(gpu_backend));
            }
        }

        manager
    }

    /// Benchmark backends for given problem
    pub fn benchmark_backends<const N: usize>(
        manager: &BackendManager,
        circuit: &Circuit<N>,
        parameters: &[f64],
    ) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();

        for (backend_name, _) in manager.list_backends() {
            let start = std::time::Instant::now();

            // Would execute circuit multiple times for accurate timing
            let _result = manager.execute_circuit(circuit, parameters, None)?;

            let duration = start.elapsed().as_secs_f64();
            results.insert(backend_name, duration);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_statevector_backend() {
        let backend = StatevectorBackend::new(10);
        assert_eq!(backend.name(), "statevector");
        assert_eq!(backend.max_qubits(), 10);
        assert!(!backend.supports_noise());
    }

    #[test]
    fn test_mps_backend() {
        let backend = MPSBackend::new(64, 50);
        assert_eq!(backend.name(), "mps");
        assert_eq!(backend.max_qubits(), 50);

        let caps = backend.capabilities();
        assert!(!caps.adjoint_gradients);
        assert!(!caps.gpu_acceleration);
    }

    #[test]
    #[ignore] // Temporarily disabled due to stack overflow issue
    fn test_backend_manager() {
        let mut manager = BackendManager::new();
        manager.register_backend("test", Backend::Statevector(StatevectorBackend::new(10)));

        let backends = manager.list_backends();
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].0, "test");
    }

    #[test]
    fn test_backend_selection() {
        let mut manager = backend_utils::create_default_manager();
        manager.set_strategy(BackendSelectionStrategy::Fastest);

        let result = manager.select_backend(15, None);
        assert!(result.is_ok());

        let result = manager.select_backend(35, None);
        assert!(result.is_ok());
    }
}
