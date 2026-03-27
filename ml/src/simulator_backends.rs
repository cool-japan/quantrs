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
        /// Convert a Register's amplitude slice into an `Array1<Complex64>`.
        fn register_to_array(amplitudes: &[Complex64]) -> Array1<Complex64> {
            Array1::from_vec(amplitudes.to_vec())
        }

        macro_rules! run_circuit {
            ($c:expr) => {{
                let state = self.simulator.run($c)?;
                let amps = register_to_array(state.amplitudes());
                let probabilities: Vec<f64> = amps.iter().map(|c| c.norm_sqr()).collect();
                Ok(SimulationResult {
                    state: Some(amps),
                    measurements: None,
                    probabilities: Some(Array1::from_vec(probabilities)),
                    metadata: HashMap::new(),
                })
            }};
        }

        match circuit {
            DynamicCircuit::Circuit1(c) => run_circuit!(c),
            DynamicCircuit::Circuit2(c) => run_circuit!(c),
            DynamicCircuit::Circuit4(c) => run_circuit!(c),
            DynamicCircuit::Circuit8(c) => run_circuit!(c),
            DynamicCircuit::Circuit16(c) => run_circuit!(c),
            DynamicCircuit::Circuit32(c) => run_circuit!(c),
            DynamicCircuit::Circuit64(c) => run_circuit!(c),
        }
    }

    fn expectation_value(
        &self,
        circuit: &DynamicCircuit,
        _parameters: &[f64],
        observable: &Observable,
    ) -> Result<f64> {
        macro_rules! run_and_expect {
            ($c:expr) => {{
                let state = self.simulator.run($c)?;
                let amps = Array1::from_vec(state.amplitudes().to_vec());
                self.compute_expectation(&amps, observable)
            }};
        }

        match circuit {
            DynamicCircuit::Circuit1(c) => run_and_expect!(c),
            DynamicCircuit::Circuit2(c) => run_and_expect!(c),
            DynamicCircuit::Circuit4(c) => run_and_expect!(c),
            DynamicCircuit::Circuit8(c) => run_and_expect!(c),
            DynamicCircuit::Circuit16(c) => run_and_expect!(c),
            DynamicCircuit::Circuit32(c) => run_and_expect!(c),
            DynamicCircuit::Circuit64(c) => run_and_expect!(c),
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
        BackendCapabilities {
            max_qubits: self.max_qubits,
            noise_simulation: false,
            gpu_acceleration: false,
            distributed: false,
            adjoint_gradients: false,
            memory_per_qubit: 16, // 16 bytes per amplitude (Complex64)
        }
    }

    /// Get backend name
    fn name(&self) -> &str {
        "statevector"
    }

    /// Maximum number of qubits supported
    fn max_qubits(&self) -> usize {
        self.max_qubits
    }

    /// Check if backend supports noise simulation
    fn supports_noise(&self) -> bool {
        false
    }
}

impl StatevectorBackend {
    /// Compute `<ψ|P|ψ>` for a single-qubit Pauli on qubit index `qubit_idx`.
    ///
    /// The statevector has `2^n` entries.  The basis states are indexed by integers where
    /// bit `qubit_idx` (LSB = qubit 0) selects the computational basis state of that qubit.
    ///
    /// - `Z_i`: `<Z_i>` = Σ_{j: bit i is 0} |ψ_j|² - Σ_{j: bit i is 1} |ψ_j|²
    /// - `X_i`: `<X_i>` = 2 · Re[ Σ_{j: bit i is 0} ψ_j* · ψ_{j ⊕ (1 << i)} ]
    /// - `Y_i`: `<Y_i>` = 2 · Im[ Σ_{j: bit i is 0} ψ_j* · ψ_{j ⊕ (1 << i)} ]  (sign convention: Y = [[0,-i],[i,0]])
    /// - `I_i`:  1.0
    fn pauli_expectation_single(
        &self,
        state: &Array1<Complex64>,
        pauli: char,
        qubit_idx: usize,
    ) -> Result<f64> {
        let dim = state.len();
        if dim == 0 {
            return Err(MLError::ValidationError("Empty statevector".to_string()));
        }
        // Check dim is a power of 2.
        if dim & (dim - 1) != 0 {
            return Err(MLError::ValidationError(format!(
                "Statevector dimension {dim} is not a power of 2"
            )));
        }
        let n = dim.trailing_zeros() as usize; // number of qubits
        if qubit_idx >= n {
            return Err(MLError::ValidationError(format!(
                "Qubit index {qubit_idx} out of range for {n}-qubit state"
            )));
        }

        let bit = 1usize << qubit_idx;

        match pauli {
            'I' => Ok(1.0),
            'Z' => {
                let mut expectation = 0.0_f64;
                for (j, amp) in state.iter().enumerate() {
                    let prob = amp.norm_sqr();
                    if j & bit == 0 {
                        expectation += prob; // eigenvalue +1
                    } else {
                        expectation -= prob; // eigenvalue -1
                    }
                }
                Ok(expectation)
            }
            'X' => {
                // <X_i> = 2 Re[ Σ_{j: bit i=0} ψ_j* · ψ_{j ^ bit} ]
                let mut sum = Complex64::new(0.0, 0.0);
                for (j, amp) in state.iter().enumerate() {
                    if j & bit == 0 {
                        let partner = j ^ bit;
                        if partner < dim {
                            sum += amp.conj() * state[partner];
                        }
                    }
                }
                Ok(2.0 * sum.re)
            }
            'Y' => {
                // Y = [[0,-i],[i,0]]; <Y_i> = 2 Im[ Σ_{j: bit i=0} ψ_j* · ψ_{j ^ bit} ]
                // Because Y = -i·σ^+ + i·σ^-, the expectation is purely real:
                // <Y> = Im[2 · Σ_{j: bit=0} conj(ψ_j) * ψ_{j^bit}]
                let mut sum = Complex64::new(0.0, 0.0);
                for (j, amp) in state.iter().enumerate() {
                    if j & bit == 0 {
                        let partner = j ^ bit;
                        if partner < dim {
                            sum += amp.conj() * state[partner];
                        }
                    }
                }
                Ok(2.0 * sum.im)
            }
            _ => Err(MLError::ValidationError(format!(
                "Unknown Pauli operator '{pauli}'"
            ))),
        }
    }

    /// Helper method to compute expectation values `<ψ|O|ψ>`.
    fn compute_expectation(
        &self,
        state: &Array1<Complex64>,
        observable: &Observable,
    ) -> Result<f64> {
        match observable {
            Observable::PauliString(pauli_string) => {
                use quantrs2_sim::prelude::PauliOperator;

                let dim = state.len();
                // Apply the Pauli string to |ψ⟩: result_vec = P|ψ⟩
                // Then <ψ|P|ψ> = Re[<ψ|result_vec>]  (expectation is real for Hermitian P)
                let mut result_vec = state.clone();

                for (qubit_idx, pauli_op) in pauli_string.operators.iter().enumerate() {
                    let bit = 1usize << qubit_idx;
                    match pauli_op {
                        PauliOperator::I => {} // identity — no change
                        PauliOperator::Z => {
                            // Z flips sign for basis states where qubit i = 1
                            for j in 0..dim {
                                if j & bit != 0 {
                                    result_vec[j] = -result_vec[j];
                                }
                            }
                        }
                        PauliOperator::X => {
                            // X bit-flips qubit i: swap amplitude pairs (j, j^bit)
                            for j in 0..dim {
                                if j & bit == 0 {
                                    let partner = j ^ bit;
                                    if partner < dim {
                                        result_vec.swap(j, partner);
                                    }
                                }
                            }
                        }
                        PauliOperator::Y => {
                            // Y = [[0,-i],[i,0]]:  |0⟩ → i|1⟩,  |1⟩ → -i|0⟩
                            let mut new_vec = result_vec.clone();
                            for j in 0..dim {
                                if j & bit == 0 {
                                    let partner = j ^ bit;
                                    if partner < dim {
                                        let orig_0 = result_vec[j];
                                        let orig_1 = result_vec[partner];
                                        new_vec[j] = Complex64::new(0.0, -1.0) * orig_1;
                                        new_vec[partner] = Complex64::new(0.0, 1.0) * orig_0;
                                    }
                                }
                            }
                            result_vec = new_vec;
                        }
                    }
                }

                // Apply the overall coefficient from the PauliString.
                // Then <ψ|P|ψ> = Re[ coeff · Σ_j conj(ψ_j) * (P|ψ>)_j ]
                let inner: Complex64 = state
                    .iter()
                    .zip(result_vec.iter())
                    .map(|(&a, &b)| a.conj() * b)
                    .sum();
                Ok((pauli_string.coefficient * inner).re)
            }
            Observable::PauliZ(qubits) => {
                // Product of Z expectations on the given qubits.
                // For a single-qubit problem: <Z_i> as defined above.
                // For multiple qubits: < ⊗_i Z_i > using the combined bit-flip parity.
                let dim = state.len();
                let mut expectation = 0.0_f64;
                for (j, amp) in state.iter().enumerate() {
                    // Parity of bits at qubit positions listed in `qubits`.
                    let parity: u32 = qubits
                        .iter()
                        .map(|&q| if j & (1 << q) != 0 { 1u32 } else { 0u32 })
                        .sum::<u32>()
                        % 2;
                    let eigenvalue = if parity == 0 { 1.0 } else { -1.0 };
                    expectation += eigenvalue * amp.norm_sqr();
                }
                Ok(expectation)
            }
            Observable::Matrix(matrix) => {
                // Compute <ψ|H|ψ> = Σ_{i,j} ψ_i* H_{ij} ψ_j
                let result: Complex64 = state
                    .iter()
                    .enumerate()
                    .map(|(i, &amp_i)| {
                        state
                            .iter()
                            .enumerate()
                            .map(|(j, &amp_j)| amp_i.conj() * matrix[[i, j]] * amp_j)
                            .sum::<Complex64>()
                    })
                    .sum();
                Ok(result.re)
            }
            Observable::Hamiltonian(terms) => {
                // H = Σ_k c_k P_k  →  <H> = Σ_k c_k <P_k>
                let mut expectation = 0.0_f64;
                for (coeff, pauli_string) in terms {
                    let term_exp = self.compute_expectation(
                        state,
                        &Observable::PauliString(pauli_string.clone()),
                    )?;
                    expectation += coeff * term_exp;
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
