//! Quantum state encoders (TorchQuantum-compatible)
//!
//! This module provides encoding schemes for classical data into quantum states:
//! - GeneralEncoder: Configurable encoder with custom gate sequences
//! - PhaseEncoder: Phase encoding using RZ gates
//! - StateEncoder: Direct state vector encoding
//! - AmplitudeEncoder: Amplitude encoding

use super::{
    gates::{TQHadamard, TQPauliX, TQPauliY, TQPauliZ, TQRx, TQRy, TQRz, TQSX},
    CType, NParamsEnum, OpHistoryEntry, TQDevice, TQModule, TQOperator, TQParameter, WiresEnum,
};
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array2, ArrayD, IxDyn};
use scirs2_core::Complex64;
use std::f64::consts::PI;

/// Base encoder trait
pub trait TQEncoder: TQModule {
    /// Encode classical data into quantum state
    fn encode(&mut self, qdev: &mut TQDevice, x: &Array2<f64>) -> Result<()>;
}

/// General encoder with configurable gate sequence
#[derive(Debug, Clone)]
pub struct TQGeneralEncoder {
    /// Encoding function list
    pub func_list: Vec<EncodingOp>,
    n_wires: Option<usize>,
    static_mode: bool,
}

/// Encoding operation specification
#[derive(Debug, Clone)]
pub struct EncodingOp {
    /// Input indices from the data
    pub input_idx: Vec<usize>,
    /// Gate function name
    pub func: String,
    /// Wires to apply gate
    pub wires: Vec<usize>,
}

impl TQGeneralEncoder {
    pub fn new(func_list: Vec<EncodingOp>) -> Self {
        Self {
            func_list,
            n_wires: None,
            static_mode: false,
        }
    }

    /// Create encoder from predefined pattern
    pub fn from_pattern(pattern: &str, n_wires: usize) -> Self {
        let func_list = match pattern {
            "ry" => (0..n_wires)
                .map(|i| EncodingOp {
                    input_idx: vec![i],
                    func: "ry".to_string(),
                    wires: vec![i],
                })
                .collect(),
            "rx" => (0..n_wires)
                .map(|i| EncodingOp {
                    input_idx: vec![i],
                    func: "rx".to_string(),
                    wires: vec![i],
                })
                .collect(),
            "rz" => (0..n_wires)
                .map(|i| EncodingOp {
                    input_idx: vec![i],
                    func: "rz".to_string(),
                    wires: vec![i],
                })
                .collect(),
            "rxyz" => {
                let mut ops = Vec::new();
                for (gate_idx, gate) in ["rx", "ry", "rz"].iter().enumerate() {
                    for i in 0..n_wires {
                        ops.push(EncodingOp {
                            input_idx: vec![gate_idx * n_wires + i],
                            func: gate.to_string(),
                            wires: vec![i],
                        });
                    }
                }
                ops
            }
            _ => {
                // Default: RY encoding
                (0..n_wires)
                    .map(|i| EncodingOp {
                        input_idx: vec![i],
                        func: "ry".to_string(),
                        wires: vec![i],
                    })
                    .collect()
            }
        };

        Self {
            func_list,
            n_wires: Some(n_wires),
            static_mode: false,
        }
    }
}

impl TQModule for TQGeneralEncoder {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use encode() instead of forward() for encoders".to_string(),
        ))
    }

    fn forward_with_input(&mut self, qdev: &mut TQDevice, x: Option<&Array2<f64>>) -> Result<()> {
        if let Some(data) = x {
            self.encode(qdev, data)
        } else {
            Err(MLError::InvalidConfiguration(
                "Input data required for encoder".to_string(),
            ))
        }
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }

    fn n_wires(&self) -> Option<usize> {
        self.n_wires
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = Some(n_wires);
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
    }

    fn static_off(&mut self) {
        self.static_mode = false;
    }

    fn name(&self) -> &str {
        "GeneralEncoder"
    }
}

impl TQEncoder for TQGeneralEncoder {
    fn encode(&mut self, qdev: &mut TQDevice, x: &Array2<f64>) -> Result<()> {
        let bsz = x.nrows();

        // Ensure device batch size matches input
        if qdev.bsz != bsz {
            qdev.reset_states(bsz);
        }

        for op in &self.func_list {
            // Get parameters from input data
            let params: Vec<f64> = op
                .input_idx
                .iter()
                .filter_map(|&idx| {
                    if idx < x.ncols() {
                        Some(x[[0, idx]]) // Use first batch element for parameter
                    } else {
                        None
                    }
                })
                .collect();

            // Apply gate based on function name
            match op.func.as_str() {
                "rx" => {
                    let mut gate = TQRx::new(true, false);
                    gate.apply_with_params(qdev, &op.wires, Some(&params))?;
                }
                "ry" => {
                    let mut gate = TQRy::new(true, false);
                    gate.apply_with_params(qdev, &op.wires, Some(&params))?;
                }
                "rz" => {
                    let mut gate = TQRz::new(true, false);
                    gate.apply_with_params(qdev, &op.wires, Some(&params))?;
                }
                "h" | "hadamard" => {
                    let mut gate = TQHadamard::new();
                    gate.apply(qdev, &op.wires)?;
                }
                "x" | "paulix" => {
                    let mut gate = TQPauliX::new();
                    gate.apply(qdev, &op.wires)?;
                }
                "y" | "pauliy" => {
                    let mut gate = TQPauliY::new();
                    gate.apply(qdev, &op.wires)?;
                }
                "z" | "pauliz" => {
                    let mut gate = TQPauliZ::new();
                    gate.apply(qdev, &op.wires)?;
                }
                "sx" => {
                    let mut gate = TQSX::new();
                    gate.apply(qdev, &op.wires)?;
                }
                _ => {
                    return Err(MLError::InvalidConfiguration(format!(
                        "Unknown gate: {}",
                        op.func
                    )));
                }
            }
        }

        Ok(())
    }
}

/// Phase encoder (applies same rotation type to all qubits)
#[derive(Debug, Clone)]
pub struct TQPhaseEncoder {
    /// Rotation type (rx, ry, rz)
    pub func: String,
    n_wires: Option<usize>,
    static_mode: bool,
}

impl TQPhaseEncoder {
    pub fn new(func: impl Into<String>) -> Self {
        Self {
            func: func.into(),
            n_wires: None,
            static_mode: false,
        }
    }

    /// Create RY phase encoder
    pub fn ry() -> Self {
        Self::new("ry")
    }

    /// Create RX phase encoder
    pub fn rx() -> Self {
        Self::new("rx")
    }

    /// Create RZ phase encoder
    pub fn rz() -> Self {
        Self::new("rz")
    }
}

impl TQModule for TQPhaseEncoder {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use encode() instead of forward() for encoders".to_string(),
        ))
    }

    fn forward_with_input(&mut self, qdev: &mut TQDevice, x: Option<&Array2<f64>>) -> Result<()> {
        if let Some(data) = x {
            self.encode(qdev, data)
        } else {
            Err(MLError::InvalidConfiguration(
                "Input data required for encoder".to_string(),
            ))
        }
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }

    fn n_wires(&self) -> Option<usize> {
        self.n_wires
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = Some(n_wires);
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
    }

    fn static_off(&mut self) {
        self.static_mode = false;
    }

    fn name(&self) -> &str {
        "PhaseEncoder"
    }
}

impl TQEncoder for TQPhaseEncoder {
    fn encode(&mut self, qdev: &mut TQDevice, x: &Array2<f64>) -> Result<()> {
        let n_wires = qdev.n_wires;

        for wire in 0..n_wires {
            let param = if wire < x.ncols() { x[[0, wire]] } else { 0.0 };

            match self.func.as_str() {
                "rx" => {
                    let mut gate = TQRx::new(true, false);
                    gate.apply_with_params(qdev, &[wire], Some(&[param]))?;
                }
                "ry" => {
                    let mut gate = TQRy::new(true, false);
                    gate.apply_with_params(qdev, &[wire], Some(&[param]))?;
                }
                "rz" => {
                    let mut gate = TQRz::new(true, false);
                    gate.apply_with_params(qdev, &[wire], Some(&[param]))?;
                }
                _ => {
                    return Err(MLError::InvalidConfiguration(format!(
                        "Unknown rotation gate: {}",
                        self.func
                    )));
                }
            }
        }

        Ok(())
    }
}

/// Amplitude/State encoder (direct state preparation)
#[derive(Debug, Clone)]
pub struct TQStateEncoder {
    n_wires: Option<usize>,
    static_mode: bool,
}

impl TQStateEncoder {
    pub fn new() -> Self {
        Self {
            n_wires: None,
            static_mode: false,
        }
    }
}

impl Default for TQStateEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQStateEncoder {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use encode() instead of forward() for encoders".to_string(),
        ))
    }

    fn forward_with_input(&mut self, qdev: &mut TQDevice, x: Option<&Array2<f64>>) -> Result<()> {
        if let Some(data) = x {
            self.encode(qdev, data)
        } else {
            Err(MLError::InvalidConfiguration(
                "Input data required for encoder".to_string(),
            ))
        }
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }

    fn n_wires(&self) -> Option<usize> {
        self.n_wires
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = Some(n_wires);
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
    }

    fn static_off(&mut self) {
        self.static_mode = false;
    }

    fn name(&self) -> &str {
        "StateEncoder"
    }
}

impl TQEncoder for TQStateEncoder {
    fn encode(&mut self, qdev: &mut TQDevice, x: &Array2<f64>) -> Result<()> {
        let bsz = x.nrows();
        let state_size = 1 << qdev.n_wires;

        // Normalize input and prepare state
        let mut state_data = Vec::with_capacity(state_size * bsz);

        for batch in 0..bsz {
            // Get amplitude values
            let mut amplitudes: Vec<f64> = (0..state_size)
                .map(|i| if i < x.ncols() { x[[batch, i]] } else { 0.0 })
                .collect();

            // Normalize
            let norm: f64 = amplitudes.iter().map(|a| a * a).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for a in &mut amplitudes {
                    *a /= norm;
                }
            }

            // Convert to complex
            for &a in &amplitudes {
                state_data.push(CType::new(a, 0.0));
            }
        }

        // Reshape and set states
        let mut shape = vec![bsz];
        shape.extend(vec![2; qdev.n_wires]);
        let states = ArrayD::from_shape_vec(IxDyn(&shape), state_data)
            .map_err(|e| MLError::InvalidConfiguration(e.to_string()))?;

        qdev.set_states(states);

        Ok(())
    }
}

/// Alias for amplitude encoding
pub type TQAmplitudeEncoder = TQStateEncoder;

/// Multi-phase encoder - applies multiple rotation gates to each qubit
/// Each feature is encoded with a sequence of gates (e.g., RX, RY, RZ)
#[derive(Debug, Clone)]
pub struct TQMultiPhaseEncoder {
    /// Gate functions to apply (e.g., ["rx", "ry", "rz"])
    pub funcs: Vec<String>,
    /// Wire mapping (if None, uses wires 0, 1, 2, ...)
    pub wires: Option<Vec<usize>>,
    n_wires: Option<usize>,
    static_mode: bool,
}

impl TQMultiPhaseEncoder {
    pub fn new(funcs: Vec<&str>) -> Self {
        Self {
            funcs: funcs.iter().map(|s| s.to_string()).collect(),
            wires: None,
            n_wires: None,
            static_mode: false,
        }
    }

    /// Create with specific wire mapping
    pub fn with_wires(funcs: Vec<&str>, wires: Vec<usize>) -> Self {
        let n_wires = wires.len();
        Self {
            funcs: funcs.iter().map(|s| s.to_string()).collect(),
            wires: Some(wires),
            n_wires: Some(n_wires),
            static_mode: false,
        }
    }

    /// Create RX, RY, RZ encoder
    pub fn rxyz() -> Self {
        Self::new(vec!["rx", "ry", "rz"])
    }

    /// Create RY, RZ encoder (common for VQE)
    pub fn ryrz() -> Self {
        Self::new(vec!["ry", "rz"])
    }
}

impl TQModule for TQMultiPhaseEncoder {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use encode() instead of forward() for encoders".to_string(),
        ))
    }

    fn forward_with_input(&mut self, qdev: &mut TQDevice, x: Option<&Array2<f64>>) -> Result<()> {
        if let Some(data) = x {
            self.encode(qdev, data)
        } else {
            Err(MLError::InvalidConfiguration(
                "Input data required for encoder".to_string(),
            ))
        }
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }

    fn n_wires(&self) -> Option<usize> {
        self.n_wires
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = Some(n_wires);
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
    }

    fn static_off(&mut self) {
        self.static_mode = false;
    }

    fn name(&self) -> &str {
        "MultiPhaseEncoder"
    }
}

impl TQEncoder for TQMultiPhaseEncoder {
    fn encode(&mut self, qdev: &mut TQDevice, x: &Array2<f64>) -> Result<()> {
        let wires: Vec<usize> = self
            .wires
            .clone()
            .unwrap_or_else(|| (0..qdev.n_wires).collect());

        let mut x_idx = 0;

        for (func_idx, func) in self.funcs.iter().enumerate() {
            for (wire_idx, &wire) in wires.iter().enumerate() {
                // Calculate parameter index
                let param_idx = func_idx * wires.len() + wire_idx;
                let param = if param_idx < x.ncols() {
                    x[[0, param_idx]]
                } else {
                    0.0
                };

                match func.as_str() {
                    "rx" => {
                        let mut gate = TQRx::new(true, false);
                        gate.apply_with_params(qdev, &[wire], Some(&[param]))?;
                    }
                    "ry" => {
                        let mut gate = TQRy::new(true, false);
                        gate.apply_with_params(qdev, &[wire], Some(&[param]))?;
                    }
                    "rz" => {
                        let mut gate = TQRz::new(true, false);
                        gate.apply_with_params(qdev, &[wire], Some(&[param]))?;
                    }
                    "u1" | "phaseshift" => {
                        let mut gate = TQRz::new(true, false); // U1 ≈ RZ
                        gate.apply_with_params(qdev, &[wire], Some(&[param]))?;
                    }
                    _ => {
                        return Err(MLError::InvalidConfiguration(format!(
                            "Unknown gate in MultiPhaseEncoder: {}",
                            func
                        )));
                    }
                }
                x_idx += 1;
            }
        }

        Ok(())
    }
}

/// Magnitude encoder - encodes data in the magnitude of amplitudes
/// Each classical value is mapped to the magnitude of a computational basis state
#[derive(Debug, Clone)]
pub struct TQMagnitudeEncoder {
    n_wires: Option<usize>,
    static_mode: bool,
}

impl TQMagnitudeEncoder {
    pub fn new() -> Self {
        Self {
            n_wires: None,
            static_mode: false,
        }
    }
}

impl Default for TQMagnitudeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TQModule for TQMagnitudeEncoder {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use encode() instead of forward() for encoders".to_string(),
        ))
    }

    fn forward_with_input(&mut self, qdev: &mut TQDevice, x: Option<&Array2<f64>>) -> Result<()> {
        if let Some(data) = x {
            self.encode(qdev, data)
        } else {
            Err(MLError::InvalidConfiguration(
                "Input data required for encoder".to_string(),
            ))
        }
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }

    fn n_wires(&self) -> Option<usize> {
        self.n_wires
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = Some(n_wires);
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
    }

    fn static_off(&mut self) {
        self.static_mode = false;
    }

    fn name(&self) -> &str {
        "MagnitudeEncoder"
    }
}

impl TQEncoder for TQMagnitudeEncoder {
    fn encode(&mut self, qdev: &mut TQDevice, x: &Array2<f64>) -> Result<()> {
        let bsz = x.nrows();
        let state_size = 1 << qdev.n_wires;

        // Normalize input values to use as magnitudes
        let mut state_data = Vec::with_capacity(state_size * bsz);

        for batch in 0..bsz {
            // Get magnitude values (must be non-negative)
            let mut magnitudes: Vec<f64> = (0..state_size)
                .map(|i| {
                    if i < x.ncols() {
                        x[[batch, i]].abs()
                    } else {
                        0.0
                    }
                })
                .collect();

            // Normalize to ensure sum of squared magnitudes = 1
            let norm_sq: f64 = magnitudes.iter().map(|m| m * m).sum();
            let norm = norm_sq.sqrt();
            if norm > 1e-10 {
                for m in &mut magnitudes {
                    *m /= norm;
                }
            }

            // Convert to complex amplitudes (real-valued)
            for &m in &magnitudes {
                state_data.push(CType::new(m, 0.0));
            }
        }

        // Reshape and set states
        let mut shape = vec![bsz];
        shape.extend(vec![2; qdev.n_wires]);
        let states = ArrayD::from_shape_vec(IxDyn(&shape), state_data)
            .map_err(|e| MLError::InvalidConfiguration(e.to_string()))?;

        qdev.set_states(states);

        Ok(())
    }
}

/// Angle encoder - encodes data as angles in rotation gates
/// More flexible than PhaseEncoder with configurable scaling
#[derive(Debug, Clone)]
pub struct TQAngleEncoder {
    /// Rotation type (rx, ry, rz)
    pub func: String,
    /// Scaling factor for input values
    pub scaling: f64,
    n_wires: Option<usize>,
    static_mode: bool,
}

impl TQAngleEncoder {
    pub fn new(func: impl Into<String>, scaling: f64) -> Self {
        Self {
            func: func.into(),
            scaling,
            n_wires: None,
            static_mode: false,
        }
    }

    /// Create with default PI scaling (maps [0,1] to [0, PI])
    pub fn with_pi_scaling(func: impl Into<String>) -> Self {
        Self::new(func, PI)
    }

    /// Create with 2*PI scaling (maps [0,1] to [0, 2*PI])
    pub fn with_2pi_scaling(func: impl Into<String>) -> Self {
        Self::new(func, 2.0 * PI)
    }

    /// Create RY encoder with arcsin scaling (for probability amplitude encoding)
    pub fn arcsin() -> Self {
        Self {
            func: "ry".to_string(),
            scaling: 1.0, // Will use arcsin transformation
            n_wires: None,
            static_mode: false,
        }
    }
}

impl TQModule for TQAngleEncoder {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use encode() instead of forward() for encoders".to_string(),
        ))
    }

    fn forward_with_input(&mut self, qdev: &mut TQDevice, x: Option<&Array2<f64>>) -> Result<()> {
        if let Some(data) = x {
            self.encode(qdev, data)
        } else {
            Err(MLError::InvalidConfiguration(
                "Input data required for encoder".to_string(),
            ))
        }
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }

    fn n_wires(&self) -> Option<usize> {
        self.n_wires
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = Some(n_wires);
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
    }

    fn static_off(&mut self) {
        self.static_mode = false;
    }

    fn name(&self) -> &str {
        "AngleEncoder"
    }
}

impl TQEncoder for TQAngleEncoder {
    fn encode(&mut self, qdev: &mut TQDevice, x: &Array2<f64>) -> Result<()> {
        let n_wires = qdev.n_wires;

        for wire in 0..n_wires {
            let raw_value = if wire < x.ncols() { x[[0, wire]] } else { 0.0 };

            // Apply scaling
            let param = if self.func == "arcsin" {
                // Arcsin encoding: map value to angle via arcsin
                // Clamp to [-1, 1] for valid arcsin input
                let clamped = raw_value.clamp(-1.0, 1.0);
                2.0 * clamped.asin()
            } else {
                raw_value * self.scaling
            };

            match self.func.as_str() {
                "rx" => {
                    let mut gate = TQRx::new(true, false);
                    gate.apply_with_params(qdev, &[wire], Some(&[param]))?;
                }
                "ry" | "arcsin" => {
                    let mut gate = TQRy::new(true, false);
                    gate.apply_with_params(qdev, &[wire], Some(&[param]))?;
                }
                "rz" => {
                    let mut gate = TQRz::new(true, false);
                    gate.apply_with_params(qdev, &[wire], Some(&[param]))?;
                }
                _ => {
                    return Err(MLError::InvalidConfiguration(format!(
                        "Unknown rotation gate: {}",
                        self.func
                    )));
                }
            }
        }

        Ok(())
    }
}

/// IQP (Instantaneous Quantum Polynomial) encoder
/// Encodes data using IQP-style circuit with entangling ZZ interactions
#[derive(Debug, Clone)]
pub struct TQIQPEncoder {
    /// Number of repetitions
    pub reps: usize,
    n_wires: Option<usize>,
    static_mode: bool,
}

impl TQIQPEncoder {
    pub fn new(reps: usize) -> Self {
        Self {
            reps,
            n_wires: None,
            static_mode: false,
        }
    }
}

impl Default for TQIQPEncoder {
    fn default() -> Self {
        Self::new(1)
    }
}

impl TQModule for TQIQPEncoder {
    fn forward(&mut self, _qdev: &mut TQDevice) -> Result<()> {
        Err(MLError::InvalidConfiguration(
            "Use encode() instead of forward() for encoders".to_string(),
        ))
    }

    fn forward_with_input(&mut self, qdev: &mut TQDevice, x: Option<&Array2<f64>>) -> Result<()> {
        if let Some(data) = x {
            self.encode(qdev, data)
        } else {
            Err(MLError::InvalidConfiguration(
                "Input data required for encoder".to_string(),
            ))
        }
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }

    fn n_wires(&self) -> Option<usize> {
        self.n_wires
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = Some(n_wires);
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
    }

    fn static_off(&mut self) {
        self.static_mode = false;
    }

    fn name(&self) -> &str {
        "IQPEncoder"
    }
}

impl TQEncoder for TQIQPEncoder {
    fn encode(&mut self, qdev: &mut TQDevice, x: &Array2<f64>) -> Result<()> {
        use super::gates::TQRZZ;

        let n_wires = qdev.n_wires;

        for _ in 0..self.reps {
            // First: Hadamard on all qubits
            for wire in 0..n_wires {
                let mut h = TQHadamard::new();
                h.apply(qdev, &[wire])?;
            }

            // Second: RZ encoding
            for wire in 0..n_wires {
                let param = if wire < x.ncols() { x[[0, wire]] } else { 0.0 };
                let mut rz = TQRz::new(true, false);
                rz.apply_with_params(qdev, &[wire], Some(&[param]))?;
            }

            // Third: ZZ interactions (product encoding)
            let mut pair_idx = 0;
            for i in 0..n_wires {
                for j in (i + 1)..n_wires {
                    // Product of features
                    let xi = if i < x.ncols() { x[[0, i]] } else { 0.0 };
                    let xj = if j < x.ncols() { x[[0, j]] } else { 0.0 };
                    let param = xi * xj;

                    let mut rzz = TQRZZ::new(true, false);
                    rzz.apply_with_params(qdev, &[i, j], Some(&[param]))?;
                    pair_idx += 1;
                }
            }
        }

        Ok(())
    }
}
