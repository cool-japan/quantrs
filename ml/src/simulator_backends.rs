//! Simulator backend integration for quantum machine learning
//!
//! This module provides unified interfaces to all quantum simulators
//! available in the QuantRS2 ecosystem, enabling seamless backend
//! switching for quantum ML algorithms.

use crate::error::{MLError, Result};
use quantrs2_core::prelude::*;
use quantrs2_sim::prelude::{StateVectorSimulator, PauliString, MPSSimulator};
use quantrs2_circuit::prelude::*;
use ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex64;
use std::collections::HashMap;

/// Unified simulator backend interface
pub trait SimulatorBackend: Send + Sync {
    /// Execute a quantum circuit
    fn execute_circuit(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        shots: Option<usize>,
    ) -> Result<SimulationResult>;
    
    /// Compute expectation value
    fn expectation_value(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        observable: &Observable,
    ) -> Result<f64>;
    
    /// Compute gradients using backend-specific methods
    fn compute_gradients(
        &self,
        circuit: &Circuit,
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
#[derive(Debug, Clone)]
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
            simulator: StatevectorSimulator::new(max_qubits),
            max_qubits,
        }
    }
}

impl SimulatorBackend for StatevectorBackend {
    fn execute_circuit(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        _shots: Option<usize>,
    ) -> Result<SimulationResult> {
        let state = self.simulator.run_circuit(circuit, parameters)?;
        let probabilities = state.probabilities();
        
        Ok(SimulationResult {
            state: Some(state),
            measurements: None,
            probabilities: Some(probabilities),
            metadata: HashMap::new(),
        })
    }
    
    fn expectation_value(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        observable: &Observable,
    ) -> Result<f64> {
        let state = self.simulator.run_circuit(circuit, parameters)?;
        
        match observable {
            Observable::PauliString(pauli) => {
                Ok(self.simulator.expectation_value(&state, pauli)?)
            }
            Observable::Matrix(matrix) => {
                // Compute <ψ|H|ψ>
                let amplitudes = state.amplitudes();
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
                    expectation += coeff * self.simulator.expectation_value(&state, pauli)?;
                }
                Ok(expectation)
            }
        }
    }
    
    fn compute_gradients(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        observable: &Observable,
        gradient_method: GradientMethod,
    ) -> Result<Array1<f64>> {
        match gradient_method {
            GradientMethod::ParameterShift => {
                self.parameter_shift_gradients(circuit, parameters, observable)
            }
            GradientMethod::FiniteDifference => {
                self.finite_difference_gradients(circuit, parameters, observable)
            }
            GradientMethod::Adjoint => {
                // Not implemented for statevector
                Err(MLError::NotSupported(
                    "Adjoint gradients not supported for statevector backend".to_string(),
                ))
            }
            GradientMethod::StochasticParameterShift => {
                self.stochastic_parameter_shift_gradients(circuit, parameters, observable)
            }
        }
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            max_qubits: self.max_qubits,
            noise_simulation: false,
            gpu_acceleration: false,
            distributed: false,
            adjoint_gradients: false,
            memory_per_qubit: 16, // 2^n * 16 bytes for complex128
        }
    }
    
    fn name(&self) -> &str {
        "statevector"
    }
    
    fn max_qubits(&self) -> usize {
        self.max_qubits
    }
    
    fn supports_noise(&self) -> bool {
        false
    }
}

impl StatevectorBackend {
    fn parameter_shift_gradients(
        &self,
        circuit: &Circuit,
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
    
    fn finite_difference_gradients(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        observable: &Observable,
    ) -> Result<Array1<f64>> {
        let epsilon = 1e-6;
        let mut gradients = Array1::zeros(parameters.len());
        
        let f0 = self.expectation_value(circuit, parameters, observable)?;
        
        for i in 0..parameters.len() {
            let mut params_plus = parameters.to_vec();
            params_plus[i] += epsilon;
            let f_plus = self.expectation_value(circuit, &params_plus, observable)?;
            
            gradients[i] = (f_plus - f0) / epsilon;
        }
        
        Ok(gradients)
    }
    
    fn stochastic_parameter_shift_gradients(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        observable: &Observable,
    ) -> Result<Array1<f64>> {
        // Simplified stochastic parameter shift - would use random sampling
        self.parameter_shift_gradients(circuit, parameters, observable)
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
        circuit: &Circuit,
        parameters: &[f64],
        _shots: Option<usize>,
    ) -> Result<SimulationResult> {
        let state = self.simulator.run_circuit(circuit, parameters)?;
        
        Ok(SimulationResult {
            state: None, // MPS doesn't expose full state
            measurements: None,
            probabilities: None,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("bond_dimension".to_string(), self.bond_dimension as f64);
                meta
            },
        })
    }
    
    fn expectation_value(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        observable: &Observable,
    ) -> Result<f64> {
        match observable {
            Observable::PauliString(pauli) => {
                // Would compute expectation using MPS
                Ok(0.0) // Placeholder
            }
            Observable::Hamiltonian(terms) => {
                let mut expectation = 0.0;
                for (coeff, pauli) in terms {
                    // Would compute each term using MPS
                    expectation += coeff * 0.0; // Placeholder
                }
                Ok(expectation)
            }
            Observable::Matrix(_) => {
                Err(MLError::NotSupported(
                    "Matrix observables not supported for MPS backend".to_string(),
                ))
            }
        }
    }
    
    fn compute_gradients(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        observable: &Observable,
        gradient_method: GradientMethod,
    ) -> Result<Array1<f64>> {
        match gradient_method {
            GradientMethod::ParameterShift => {
                self.parameter_shift_gradients(circuit, parameters, observable)
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
    fn parameter_shift_gradients(
        &self,
        circuit: &Circuit,
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

/// GPU-accelerated simulator backend
#[cfg(feature = "gpu")]
pub struct GPUBackend {
    /// GPU simulator
    simulator: GPUSimulator,
    /// Device ID
    device_id: usize,
    /// Maximum qubits
    max_qubits: usize,
}

#[cfg(feature = "gpu")]
impl GPUBackend {
    /// Create new GPU backend
    pub fn new(device_id: usize, max_qubits: usize) -> Result<Self> {
        let simulator = GPUSimulator::new(device_id)?;
        Ok(Self {
            simulator,
            device_id,
            max_qubits,
        })
    }
}

#[cfg(feature = "gpu")]
impl SimulatorBackend for GPUBackend {
    fn execute_circuit(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        _shots: Option<usize>,
    ) -> Result<SimulationResult> {
        let state = self.simulator.run_circuit(circuit, parameters)?;
        let probabilities = state.probabilities();
        
        Ok(SimulationResult {
            state: Some(state),
            measurements: None,
            probabilities: Some(probabilities),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("device_id".to_string(), self.device_id as f64);
                meta
            },
        })
    }
    
    fn expectation_value(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        observable: &Observable,
    ) -> Result<f64> {
        match observable {
            Observable::PauliString(pauli) => {
                let state = self.simulator.run_circuit(circuit, parameters)?;
                Ok(self.simulator.expectation_value(&state, pauli)?)
            }
            Observable::Hamiltonian(terms) => {
                let mut expectation = 0.0;
                for (coeff, pauli) in terms {
                    let state = self.simulator.run_circuit(circuit, parameters)?;
                    expectation += coeff * self.simulator.expectation_value(&state, pauli)?;
                }
                Ok(expectation)
            }
            Observable::Matrix(_) => {
                Err(MLError::NotSupported(
                    "Matrix observables not yet supported for GPU backend".to_string(),
                ))
            }
        }
    }
    
    fn compute_gradients(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        observable: &Observable,
        gradient_method: GradientMethod,
    ) -> Result<Array1<f64>> {
        match gradient_method {
            GradientMethod::ParameterShift => {
                self.parameter_shift_gradients(circuit, parameters, observable)
            }
            GradientMethod::Adjoint => {
                self.adjoint_gradients(circuit, parameters, observable)
            }
            _ => Err(MLError::NotSupported(
                "Gradient method not supported for GPU backend".to_string(),
            )),
        }
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            max_qubits: self.max_qubits,
            noise_simulation: false,
            gpu_acceleration: true,
            distributed: false,
            adjoint_gradients: true,
            memory_per_qubit: 16, // GPU memory
        }
    }
    
    fn name(&self) -> &str {
        "gpu"
    }
    
    fn max_qubits(&self) -> usize {
        self.max_qubits
    }
    
    fn supports_noise(&self) -> bool {
        false
    }
}

#[cfg(feature = "gpu")]
impl GPUBackend {
    fn parameter_shift_gradients(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        observable: &Observable,
    ) -> Result<Array1<f64>> {
        // Use GPU for parallel parameter shift
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
    
    fn adjoint_gradients(
        &self,
        _circuit: &Circuit,
        _parameters: &[f64],
        _observable: &Observable,
    ) -> Result<Array1<f64>> {
        // Placeholder for adjoint method implementation
        Err(MLError::NotSupported(
            "Adjoint gradients not yet implemented".to_string(),
        ))
    }
}

/// Backend manager for automatic backend selection
pub struct BackendManager {
    /// Available backends
    backends: HashMap<String, Box<dyn SimulatorBackend>>,
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
    pub fn register_backend(
        &mut self,
        name: impl Into<String>,
        backend: Box<dyn SimulatorBackend>,
    ) {
        self.backends.insert(name.into(), backend);
    }
    
    /// Set selection strategy
    pub fn set_strategy(&mut self, strategy: BackendSelectionStrategy) {
        self.selection_strategy = strategy;
    }
    
    /// Select optimal backend for given problem
    pub fn select_backend(&mut self, num_qubits: usize, shots: Option<usize>) -> Result<()> {
        let backend_name = match &self.selection_strategy {
            BackendSelectionStrategy::Fastest => {
                self.select_fastest_backend(num_qubits, shots)?
            }
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
    pub fn execute_circuit(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        shots: Option<usize>,
    ) -> Result<SimulationResult> {
        if let Some(ref backend_name) = self.current_backend {
            if let Some(backend) = self.backends.get(backend_name) {
                backend.execute_circuit(circuit, parameters, shots)
            } else {
                Err(MLError::InvalidConfiguration(
                    format!("Backend '{}' not found", backend_name),
                ))
            }
        } else {
            Err(MLError::InvalidConfiguration(
                "No backend selected".to_string(),
            ))
        }
    }
    
    /// Get current backend
    pub fn current_backend(&self) -> Option<&dyn SimulatorBackend> {
        self.current_backend
            .as_ref()
            .and_then(|name| self.backends.get(name))
            .map(|b| b.as_ref())
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
            Box::new(StatevectorBackend::new(25)),
        );
        
        // Register MPS backend
        manager.register_backend(
            "mps",
            Box::new(MPSBackend::new(64, 100)),
        );
        
        // Register GPU backend if available
        #[cfg(feature = "gpu")]
        {
            if let Ok(gpu_backend) = GPUBackend::new(0, 30) {
                manager.register_backend("gpu", Box::new(gpu_backend));
            }
        }
        
        manager
    }
    
    /// Benchmark backends for given problem
    pub fn benchmark_backends(
        manager: &BackendManager,
        circuit: &Circuit,
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
    fn test_backend_manager() {
        let mut manager = BackendManager::new();
        manager.register_backend(
            "test",
            Box::new(StatevectorBackend::new(10)),
        );
        
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