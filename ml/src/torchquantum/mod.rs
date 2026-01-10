//! TorchQuantum-compatible API for quantum machine learning
//!
//! This module provides a Pure Rust implementation compatible with TorchQuantum's API,
//! enabling seamless migration from PyTorch-based quantum ML workflows.
//!
//! ## Key Features
//!
//! - **QuantumModule**: Base trait for quantum modules (similar to PyTorch's nn.Module)
//! - **QuantumDevice**: Quantum state vector container with batch support
//! - **Operators**: Parameterized quantum gates with automatic differentiation support
//! - **Encoders**: Various encoding schemes (angle, amplitude, phase)
//! - **Measurements**: Expectation values, sampling, and observable measurements
//! - **Layers**: Pre-built quantum layer templates (Barren, Farhi, Maxwell, etc.)
//!
//! ## TorchQuantum Compatibility
//!
//! This module mirrors TorchQuantum's API patterns:
//! - `tq.QuantumModule` → `TQModule`
//! - `tq.QuantumDevice` → `TQDevice`
//! - `tq.Operator` → `TQOperator`
//! - `tq.encoding.*` → `encoding::*`
//! - `tq.measurement.*` → `measurement::*`

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayD, IxDyn};
use scirs2_core::Complex64;
use std::f64::consts::PI;

// Sub-modules
pub mod ansatz;
pub mod autograd;
pub mod conv;
pub mod encoding;
pub mod functional;
pub mod gates;
pub mod layer;
pub mod measurement;
pub mod noise;
pub mod pooling;
pub mod tensor_network;

// ============================================================================
// Core Types and Constants
// ============================================================================

/// Complex data type for quantum states (matches TorchQuantum's C_DTYPE)
pub type CType = Complex64;

/// Float data type for parameters (matches TorchQuantum's F_DTYPE)
pub type FType = f64;

/// Wire enumeration for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WiresEnum {
    /// Operation applies to any wires
    AnyWires,
    /// Operation applies to all wires
    AllWires,
    /// Operation applies to specific number of wires
    Fixed(usize),
}

/// Number of parameters enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NParamsEnum {
    /// Any number of parameters
    AnyNParams,
    /// Fixed number of parameters
    Fixed(usize),
}

// ============================================================================
// TQModule Trait - Core abstraction for quantum modules
// ============================================================================

/// Base trait for all TorchQuantum-compatible quantum modules
///
/// This trait mirrors TorchQuantum's `QuantumModule` class, providing:
/// - Forward pass execution
/// - Parameter management
/// - Static/dynamic mode switching
/// - Noise model support
pub trait TQModule: Send + Sync {
    /// Execute the forward pass on the quantum device
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()>;

    /// Execute forward pass with optional input data (for encoders)
    fn forward_with_input(&mut self, qdev: &mut TQDevice, _x: Option<&Array2<f64>>) -> Result<()> {
        self.forward(qdev)
    }

    /// Get all trainable parameters
    fn parameters(&self) -> Vec<TQParameter>;

    /// Get number of wires this module operates on
    fn n_wires(&self) -> Option<usize>;

    /// Set number of wires
    fn set_n_wires(&mut self, n_wires: usize);

    /// Check if module is in static mode
    fn is_static_mode(&self) -> bool;

    /// Enable static mode for graph optimization
    fn static_on(&mut self);

    /// Disable static mode
    fn static_off(&mut self);

    /// Get the unitary matrix representation (if available)
    fn get_unitary(&self) -> Option<Array2<CType>> {
        None
    }

    /// Module name for debugging
    fn name(&self) -> &str;

    /// Zero gradients of all parameters
    fn zero_grad(&mut self) {
        // Default implementation - override for modules with parameters
    }

    /// Set training mode
    fn train(&mut self, _mode: bool) {
        // Default implementation
    }

    /// Check if in training mode
    fn training(&self) -> bool {
        true
    }
}

// ============================================================================
// TQParameter - Trainable parameter wrapper
// ============================================================================

/// Quantum parameter wrapper (similar to TorchQuantum's parameter handling)
#[derive(Debug, Clone)]
pub struct TQParameter {
    /// Parameter values
    pub data: ArrayD<f64>,
    /// Parameter name
    pub name: String,
    /// Whether parameter requires gradient
    pub requires_grad: bool,
    /// Gradient values (if computed)
    pub grad: Option<ArrayD<f64>>,
}

impl TQParameter {
    /// Create new trainable parameter
    pub fn new(data: ArrayD<f64>, name: impl Into<String>) -> Self {
        Self {
            data,
            name: name.into(),
            requires_grad: true,
            grad: None,
        }
    }

    /// Create parameter without gradients
    pub fn no_grad(data: ArrayD<f64>, name: impl Into<String>) -> Self {
        Self {
            data,
            name: name.into(),
            requires_grad: false,
            grad: None,
        }
    }

    /// Get parameter shape
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Zero the gradient
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Initialize with uniform random values in [-pi, pi]
    pub fn init_uniform_pi(&mut self) {
        for elem in self.data.iter_mut() {
            *elem = (fastrand::f64() * 2.0 - 1.0) * PI;
        }
    }

    /// Initialize with constant value
    pub fn init_constant(&mut self, value: f64) {
        for elem in self.data.iter_mut() {
            *elem = value;
        }
    }
}

// ============================================================================
// TQDevice - Quantum device with state vector
// ============================================================================

/// Quantum device containing the quantum state vector
///
/// This struct mirrors TorchQuantum's `QuantumDevice` class, providing:
/// - Multi-dimensional state tensor representation
/// - Batch support for parallel execution
/// - State reset and cloning operations
#[derive(Debug, Clone)]
pub struct TQDevice {
    /// Number of qubits
    pub n_wires: usize,
    /// Device name
    pub device_name: String,
    /// Batch size
    pub bsz: usize,
    /// Quantum state vector (batched, multi-dimensional)
    pub states: ArrayD<CType>,
    /// Whether to record operations
    pub record_op: bool,
    /// Operation history
    pub op_history: Vec<OpHistoryEntry>,
}

/// Operation history entry
#[derive(Debug, Clone)]
pub struct OpHistoryEntry {
    /// Gate name
    pub name: String,
    /// Wires the operation acts on
    pub wires: Vec<usize>,
    /// Parameters (if any)
    pub params: Option<Vec<f64>>,
    /// Whether operation is inverse
    pub inverse: bool,
    /// Whether parameters are trainable
    pub trainable: bool,
}

impl TQDevice {
    /// Create new quantum device
    pub fn new(n_wires: usize) -> Self {
        Self::with_batch_size(n_wires, 1)
    }

    /// Create quantum device with batch size
    pub fn with_batch_size(n_wires: usize, bsz: usize) -> Self {
        // Initialize state vector |0...0>
        let state_size = 1 << n_wires; // 2^n_wires
        let mut state_data = vec![CType::new(0.0, 0.0); state_size * bsz];
        // Set |0...0> amplitude to 1 for each batch
        for b in 0..bsz {
            state_data[b * state_size] = CType::new(1.0, 0.0);
        }

        // Shape: [bsz, 2, 2, ..., 2] (n_wires times)
        let mut shape = vec![bsz];
        shape.extend(vec![2; n_wires]);

        let states = ArrayD::from_shape_vec(IxDyn(&shape), state_data)
            .unwrap_or_else(|_| ArrayD::zeros(IxDyn(&shape)));

        Self {
            n_wires,
            device_name: "default".to_string(),
            bsz,
            states,
            record_op: false,
            op_history: Vec::new(),
        }
    }

    /// Reset to |0...0> state
    pub fn reset_states(&mut self, bsz: usize) {
        self.bsz = bsz;
        let state_size = 1 << self.n_wires;
        let mut state_data = vec![CType::new(0.0, 0.0); state_size * bsz];
        for b in 0..bsz {
            state_data[b * state_size] = CType::new(1.0, 0.0);
        }

        let mut shape = vec![bsz];
        shape.extend(vec![2; self.n_wires]);
        self.states = ArrayD::from_shape_vec(IxDyn(&shape), state_data)
            .unwrap_or_else(|_| ArrayD::zeros(IxDyn(&shape)));
    }

    /// Reset to identity matrix (useful for computing unitaries)
    pub fn reset_identity_states(&mut self) {
        let state_size = 1 << self.n_wires;
        self.bsz = state_size;

        let mut state_data = vec![CType::new(0.0, 0.0); state_size * state_size];
        // Set diagonal elements to 1
        for i in 0..state_size {
            state_data[i * state_size + i] = CType::new(1.0, 0.0);
        }

        let mut shape = vec![state_size];
        shape.extend(vec![2; self.n_wires]);
        self.states = ArrayD::from_shape_vec(IxDyn(&shape), state_data)
            .unwrap_or_else(|_| ArrayD::zeros(IxDyn(&shape)));
    }

    /// Reset to equal superposition state
    pub fn reset_all_eq_states(&mut self, bsz: usize) {
        self.bsz = bsz;
        let state_size = 1 << self.n_wires;
        let amplitude = 1.0 / (state_size as f64).sqrt();
        let state_data = vec![CType::new(amplitude, 0.0); state_size * bsz];

        let mut shape = vec![bsz];
        shape.extend(vec![2; self.n_wires]);
        self.states = ArrayD::from_shape_vec(IxDyn(&shape), state_data)
            .unwrap_or_else(|_| ArrayD::zeros(IxDyn(&shape)));
    }

    /// Clone states from another device
    pub fn clone_states(&mut self, other: &TQDevice) {
        self.states = other.states.clone();
        self.bsz = other.bsz;
    }

    /// Set states directly
    pub fn set_states(&mut self, states: ArrayD<CType>) {
        self.bsz = states.shape()[0];
        self.states = states;
    }

    /// Get states as 1D vectors (shape: [bsz, 2^n_wires])
    pub fn get_states_1d(&self) -> Array2<CType> {
        let state_size = 1 << self.n_wires;
        let flat: Vec<CType> = self.states.iter().cloned().collect();
        Array2::from_shape_vec((self.bsz, state_size), flat)
            .unwrap_or_else(|_| Array2::zeros((self.bsz, state_size)))
    }

    /// Get probabilities (|amplitude|^2) as 1D vectors
    pub fn get_probs_1d(&self) -> Array2<f64> {
        let states_1d = self.get_states_1d();
        states_1d.mapv(|c| c.norm_sqr())
    }

    /// Record an operation in history
    pub fn record_operation(&mut self, entry: OpHistoryEntry) {
        if self.record_op {
            self.op_history.push(entry);
        }
    }

    /// Clear operation history
    pub fn reset_op_history(&mut self) {
        self.op_history.clear();
    }

    /// Apply a single-qubit gate matrix to specified wire
    pub fn apply_single_qubit_gate(&mut self, wire: usize, matrix: &Array2<CType>) -> Result<()> {
        if wire >= self.n_wires {
            return Err(MLError::InvalidConfiguration(format!(
                "Wire {} out of range for {} qubits",
                wire, self.n_wires
            )));
        }

        let state_size = 1 << self.n_wires;
        let states_1d = self.get_states_1d();
        let mut new_states = states_1d.clone();

        for batch in 0..self.bsz {
            for i in 0..state_size {
                // Find the pair of indices that differ only at position `wire`
                let bit = (i >> (self.n_wires - 1 - wire)) & 1;
                if bit == 0 {
                    let j = i | (1 << (self.n_wires - 1 - wire));
                    let amp0 = states_1d[[batch, i]];
                    let amp1 = states_1d[[batch, j]];
                    new_states[[batch, i]] = matrix[[0, 0]] * amp0 + matrix[[0, 1]] * amp1;
                    new_states[[batch, j]] = matrix[[1, 0]] * amp0 + matrix[[1, 1]] * amp1;
                }
            }
        }

        // Reshape back to multi-dimensional
        let flat: Vec<CType> = new_states.iter().cloned().collect();
        let mut shape = vec![self.bsz];
        shape.extend(vec![2; self.n_wires]);
        self.states = ArrayD::from_shape_vec(IxDyn(&shape), flat)
            .map_err(|e| MLError::InvalidConfiguration(e.to_string()))?;

        Ok(())
    }

    /// Apply a two-qubit gate matrix to specified wires
    pub fn apply_two_qubit_gate(
        &mut self,
        wire0: usize,
        wire1: usize,
        matrix: &Array2<CType>,
    ) -> Result<()> {
        if wire0 >= self.n_wires || wire1 >= self.n_wires {
            return Err(MLError::InvalidConfiguration(format!(
                "Wires ({}, {}) out of range for {} qubits",
                wire0, wire1, self.n_wires
            )));
        }

        let state_size = 1 << self.n_wires;
        let states_1d = self.get_states_1d();
        let mut new_states = states_1d.clone();

        let pos0 = self.n_wires - 1 - wire0;
        let pos1 = self.n_wires - 1 - wire1;

        for batch in 0..self.bsz {
            let mut visited = vec![false; state_size];

            for i in 0..state_size {
                if visited[i] {
                    continue;
                }

                // Get the 4 indices for the 2-qubit subspace
                // Base index (both bits = 0)
                let base = i & !(1 << pos0) & !(1 << pos1);

                let indices = [
                    base,                             // 00
                    base | (1 << pos1),               // 01
                    base | (1 << pos0),               // 10
                    base | (1 << pos0) | (1 << pos1), // 11
                ];

                let amps: Vec<CType> = indices.iter().map(|&idx| states_1d[[batch, idx]]).collect();

                for (row, &idx) in indices.iter().enumerate() {
                    let mut new_amp = CType::new(0.0, 0.0);
                    for (col, &amp) in amps.iter().enumerate() {
                        new_amp += matrix[[row, col]] * amp;
                    }
                    new_states[[batch, idx]] = new_amp;
                    visited[idx] = true;
                }
            }
        }

        // Reshape back
        let flat: Vec<CType> = new_states.iter().cloned().collect();
        let mut shape = vec![self.bsz];
        shape.extend(vec![2; self.n_wires]);
        self.states = ArrayD::from_shape_vec(IxDyn(&shape), flat)
            .map_err(|e| MLError::InvalidConfiguration(e.to_string()))?;

        Ok(())
    }

    /// Apply a multi-qubit gate matrix to specified wires (n-qubit gate)
    pub fn apply_multi_qubit_gate(
        &mut self,
        wires: &[usize],
        matrix: &Array2<CType>,
    ) -> Result<()> {
        let n_qubits = wires.len();

        // Validate wires
        for &wire in wires {
            if wire >= self.n_wires {
                return Err(MLError::InvalidConfiguration(format!(
                    "Wire {} out of range for {} qubits",
                    wire, self.n_wires
                )));
            }
        }

        // Expected matrix dimension: 2^n_qubits x 2^n_qubits
        let gate_dim = 1 << n_qubits;
        if matrix.nrows() != gate_dim || matrix.ncols() != gate_dim {
            return Err(MLError::InvalidConfiguration(format!(
                "Gate matrix must be {}x{} for {}-qubit gate",
                gate_dim, gate_dim, n_qubits
            )));
        }

        let state_size = 1 << self.n_wires;
        let states_1d = self.get_states_1d();
        let mut new_states = states_1d.clone();

        // Pre-compute bit positions for the wires (in reversed order for state indexing)
        let positions: Vec<usize> = wires.iter().map(|&w| self.n_wires - 1 - w).collect();

        // Create mask to identify which bits correspond to the gate qubits
        let mut wire_mask: usize = 0;
        for &pos in &positions {
            wire_mask |= 1 << pos;
        }

        for batch in 0..self.bsz {
            let mut visited = vec![false; state_size];

            for base_idx in 0..state_size {
                if visited[base_idx] {
                    continue;
                }

                // Get base index with all gate qubit bits cleared
                let base = base_idx & !wire_mask;

                // Generate all 2^n indices for the gate subspace
                let mut indices = Vec::with_capacity(gate_dim);
                for gate_idx in 0..gate_dim {
                    let mut idx = base;
                    // Set bits according to gate_idx
                    for (bit_pos, &pos) in positions.iter().enumerate() {
                        if (gate_idx >> (n_qubits - 1 - bit_pos)) & 1 == 1 {
                            idx |= 1 << pos;
                        }
                    }
                    indices.push(idx);
                }

                // Get current amplitudes
                let amps: Vec<CType> = indices.iter().map(|&idx| states_1d[[batch, idx]]).collect();

                // Apply matrix
                for (row, &idx) in indices.iter().enumerate() {
                    let mut new_amp = CType::new(0.0, 0.0);
                    for (col, &amp) in amps.iter().enumerate() {
                        new_amp += matrix[[row, col]] * amp;
                    }
                    new_states[[batch, idx]] = new_amp;
                    visited[idx] = true;
                }
            }
        }

        // Reshape back
        let flat: Vec<CType> = new_states.iter().cloned().collect();
        let mut shape = vec![self.bsz];
        shape.extend(vec![2; self.n_wires]);
        self.states = ArrayD::from_shape_vec(IxDyn(&shape), flat)
            .map_err(|e| MLError::InvalidConfiguration(e.to_string()))?;

        Ok(())
    }
}

// ============================================================================
// TQOperator - Base quantum operator
// ============================================================================

/// Base quantum operator trait
pub trait TQOperator: TQModule {
    /// Number of wires this operator acts on
    fn num_wires(&self) -> WiresEnum;

    /// Number of parameters
    fn num_params(&self) -> NParamsEnum;

    /// Get the unitary matrix for given parameters
    fn get_matrix(&self, params: Option<&[f64]>) -> Array2<CType>;

    /// Get eigenvalues (if applicable)
    fn get_eigvals(&self, _params: Option<&[f64]>) -> Option<Array1<CType>> {
        None
    }

    /// Apply the operator to a quantum device
    fn apply(&mut self, qdev: &mut TQDevice, wires: &[usize]) -> Result<()>;

    /// Apply with specific parameters
    fn apply_with_params(
        &mut self,
        qdev: &mut TQDevice,
        wires: &[usize],
        params: Option<&[f64]>,
    ) -> Result<()>;

    /// Whether this operator has trainable parameters
    fn has_params(&self) -> bool;

    /// Whether parameters are trainable
    fn trainable(&self) -> bool;

    /// Get/set inverse flag
    fn inverse(&self) -> bool;
    fn set_inverse(&mut self, inverse: bool);
}

// ============================================================================
// TQModuleList - Container for modules
// ============================================================================

/// Container for a list of TQModules (similar to PyTorch's ModuleList)
pub struct TQModuleList {
    modules: Vec<Box<dyn TQModule>>,
    static_mode: bool,
}

impl TQModuleList {
    /// Create empty module list
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            static_mode: false,
        }
    }

    /// Add a module to the list
    pub fn append(&mut self, module: Box<dyn TQModule>) {
        self.modules.push(module);
    }

    /// Get number of modules
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Get module at index
    pub fn get(&self, index: usize) -> Option<&Box<dyn TQModule>> {
        self.modules.get(index)
    }

    /// Get mutable module at index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Box<dyn TQModule>> {
        self.modules.get_mut(index)
    }

    /// Iterate over modules
    pub fn iter(&self) -> impl Iterator<Item = &Box<dyn TQModule>> {
        self.modules.iter()
    }

    /// Iterate mutably over modules
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn TQModule>> {
        self.modules.iter_mut()
    }
}

impl Default for TQModuleList {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQModuleList {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for module in &mut self.modules {
            module.forward(qdev)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        self.modules.first().and_then(|m| m.n_wires())
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        for module in &mut self.modules {
            module.set_n_wires(n_wires);
        }
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for module in &mut self.modules {
            module.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for module in &mut self.modules {
            module.static_off();
        }
    }

    fn name(&self) -> &str {
        "ModuleList"
    }

    fn zero_grad(&mut self) {
        for module in &mut self.modules {
            module.zero_grad();
        }
    }
}

// ============================================================================
// Prelude - Convenient re-exports
// ============================================================================

pub mod prelude {
    //! Convenient re-exports for TorchQuantum-compatible API

    pub use super::{
        CType, FType, NParamsEnum, OpHistoryEntry, TQDevice, TQModule, TQModuleList, TQOperator,
        TQParameter, WiresEnum,
    };

    // Gates
    pub use super::gates::{
        // Single-qubit gates
        TQHadamard,
        TQPauliX,
        TQPauliY,
        TQPauliZ,
        TQRx,
        TQRy,
        TQRz,
        // Two-qubit gates
        TQCNOT,
        // Controlled rotation gates
        TQCRX,
        TQCRY,
        TQCRZ,
        TQCZ,
        // Parameterized two-qubit gates
        TQRXX,
        TQRYY,
        TQRZX,
        TQRZZ,
        TQS,
        TQSWAP,
        TQSX,
        TQT,
        TQU1,
        TQU2,
        TQU3,
    };

    // Encoding
    pub use super::encoding::{
        EncodingOp, TQAmplitudeEncoder, TQEncoder, TQGeneralEncoder, TQPhaseEncoder, TQStateEncoder,
    };

    // Measurement
    pub use super::measurement::{
        expval_joint_analytical, expval_joint_sampling, gen_bitstrings, measure, TQMeasureAll,
    };

    // Layers
    pub use super::layer::{
        TQBarrenLayer, TQFarhiLayer, TQLayerConfig, TQMaxwellLayer, TQOp1QAllLayer, TQOp2QAllLayer,
        TQRXYZCXLayer, TQSethLayer, TQStrongEntanglingLayer,
    };

    // Autograd
    pub use super::autograd::{
        gradient_norm, gradient_statistics, ClippingStatistics, ClippingStrategy,
        GradientAccumulator, GradientCheckResult, GradientChecker, GradientClipper,
        GradientStatistics, ParameterGroup, ParameterGroupManager, ParameterRegistry,
        ParameterStatistics,
    };

    // Ansatz templates
    pub use super::ansatz::{
        EfficientSU2Layer, EntanglementPattern, RealAmplitudesLayer, TwoLocalLayer,
    };

    // Convolutional layers
    pub use super::conv::{QConv1D, QConv2D};

    // Pooling layers
    pub use super::pooling::{QAvgPool, QMaxPool};

    // Tensor network backend
    pub use super::tensor_network::{
        CompressionMethod, MPSTensor, MatrixProductState, TQTensorNetworkBackend,
        TensorNetworkConfig,
    };

    // Noise-aware training
    pub use super::noise::{
        GateTimes, MitigatedExpectation, MitigatedExpectationConfig, MitigationMethod,
        NoiseAwareGradient, NoiseAwareGradientConfig, NoiseAwareTrainer, NoiseModel,
        SingleQubitNoiseType, TrainingHistory, TrainingStatistics, TwoQubitNoiseType,
        VarianceReduction, ZNEExtrapolation,
    };
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use std::f64::consts::PI;

    #[test]
    fn test_tq_device_creation() {
        let qdev = TQDevice::new(4);
        assert_eq!(qdev.n_wires, 4);
        assert_eq!(qdev.bsz, 1);

        // Check initial state is |0000>
        let probs = qdev.get_probs_1d();
        assert!((probs[[0, 0]] - 1.0).abs() < 1e-10);
        for i in 1..(1 << 4) {
            assert!(probs[[0, i]].abs() < 1e-10);
        }
    }

    #[test]
    fn test_tq_device_reset() {
        let mut qdev = TQDevice::new(2);
        qdev.reset_all_eq_states(1);

        let probs = qdev.get_probs_1d();
        let expected = 0.25; // 1/4 for 2 qubits
        for i in 0..4 {
            assert!((probs[[0, i]] - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_tq_parameter() {
        use scirs2_core::ndarray::ArrayD;

        let mut param =
            TQParameter::new(ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[2, 3])), "test");
        assert_eq!(param.shape(), &[2, 3]);
        assert_eq!(param.numel(), 6);

        param.init_constant(1.5);
        for elem in param.data.iter() {
            assert!((elem - 1.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_hadamard_gate() {
        let mut qdev = TQDevice::new(1);
        let mut h = TQHadamard::new();

        h.apply(&mut qdev, &[0]).expect("Hadamard should succeed");

        let probs = qdev.get_probs_1d();
        assert!((probs[[0, 0]] - 0.5).abs() < 1e-10);
        assert!((probs[[0, 1]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_x_gate() {
        let mut qdev = TQDevice::new(1);
        let mut x = TQPauliX::new();

        x.apply(&mut qdev, &[0]).expect("PauliX should succeed");

        let probs = qdev.get_probs_1d();
        assert!(probs[[0, 0]].abs() < 1e-10);
        assert!((probs[[0, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rx_gate() {
        let mut qdev = TQDevice::new(1);
        let mut rx = TQRx::new(true, false);

        // RX(π) should be equivalent to X (up to global phase)
        rx.apply_with_params(&mut qdev, &[0], Some(&[PI]))
            .expect("RX should succeed");

        let probs = qdev.get_probs_1d();
        assert!(probs[[0, 0]].abs() < 1e-10);
        assert!((probs[[0, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_gate() {
        let mut qdev = TQDevice::new(2);
        let mut x = TQPauliX::new();
        let mut cnot = TQCNOT::new();

        // Apply X to first qubit, then CNOT
        x.apply(&mut qdev, &[0]).expect("X should succeed");
        cnot.apply(&mut qdev, &[0, 1]).expect("CNOT should succeed");

        let probs = qdev.get_probs_1d();
        // Should be in |11> state
        assert!(probs[[0, 0]].abs() < 1e-10); // |00>
        assert!(probs[[0, 1]].abs() < 1e-10); // |01>
        assert!(probs[[0, 2]].abs() < 1e-10); // |10>
        assert!((probs[[0, 3]] - 1.0).abs() < 1e-10); // |11>
    }

    #[test]
    fn test_bell_state() {
        let mut qdev = TQDevice::new(2);
        let mut h = TQHadamard::new();
        let mut cnot = TQCNOT::new();

        h.apply(&mut qdev, &[0]).expect("H should succeed");
        cnot.apply(&mut qdev, &[0, 1]).expect("CNOT should succeed");

        let probs = qdev.get_probs_1d();
        // Bell state: (|00> + |11>)/sqrt(2)
        assert!((probs[[0, 0]] - 0.5).abs() < 1e-10); // |00>
        assert!(probs[[0, 1]].abs() < 1e-10); // |01>
        assert!(probs[[0, 2]].abs() < 1e-10); // |10>
        assert!((probs[[0, 3]] - 0.5).abs() < 1e-10); // |11>
    }

    #[test]
    fn test_module_list() {
        let mut qdev = TQDevice::new(2);
        let mut module_list = TQModuleList::new();

        module_list.append(Box::new(TQHadamard::new()));
        module_list.append(Box::new(TQPauliX::new()));

        assert_eq!(module_list.len(), 2);
        assert!(!module_list.is_empty());
    }
}
