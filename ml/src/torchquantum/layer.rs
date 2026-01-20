//! Pre-built quantum layer templates (TorchQuantum-compatible)
//!
//! This module provides various quantum layer templates:
//! - TQOp1QAllLayer: Apply single-qubit operation to all wires
//! - TQOp2QAllLayer: Apply two-qubit operation to pairs of wires
//! - TQBarrenLayer: Barren plateau layer
//! - TQRXYZCXLayer: RX, RY, RZ, CNOT layer
//! - TQFarhiLayer: QAOA-style mixer layer
//! - TQMaxwellLayer: Hardware-efficient ansatz
//! - TQSethLayer: Simple efficient ansatz
//! - TQStrongEntanglingLayer: Strong entanglement with varying patterns

use super::{
    gates::{
        TQHadamard, TQPauliX, TQPauliY, TQPauliZ, TQRx, TQRy, TQRz, TQCNOT, TQCRX, TQCRY, TQCRZ,
        TQCZ, TQRXX, TQRYY, TQRZX, TQRZZ, TQS, TQSWAP, TQSX, TQT,
    },
    CType, TQDevice, TQModule, TQOperator, TQParameter,
};
use crate::error::{MLError, Result};
use scirs2_core::ndarray::Array2;

/// Apply single-qubit operation to all wires
pub struct TQOp1QAllLayer {
    /// Number of wires
    pub n_wires: usize,
    /// Gate type name
    pub op_name: String,
    /// Whether gates have parameters
    pub has_params: bool,
    /// Whether parameters are trainable
    pub trainable: bool,
    /// Gate instances for each wire
    gates: Vec<Box<dyn TQOperator>>,
    static_mode: bool,
}

impl TQOp1QAllLayer {
    pub fn new(
        op_name: impl Into<String>,
        n_wires: usize,
        has_params: bool,
        trainable: bool,
    ) -> Self {
        let op_name = op_name.into();
        let gates: Vec<Box<dyn TQOperator>> = (0..n_wires)
            .map(|_| create_single_qubit_gate(&op_name, has_params, trainable))
            .collect();

        Self {
            n_wires,
            op_name,
            has_params,
            trainable,
            gates,
            static_mode: false,
        }
    }

    /// Create RX layer
    pub fn rx(n_wires: usize, trainable: bool) -> Self {
        Self::new("rx", n_wires, true, trainable)
    }

    /// Create RY layer
    pub fn ry(n_wires: usize, trainable: bool) -> Self {
        Self::new("ry", n_wires, true, trainable)
    }

    /// Create RZ layer
    pub fn rz(n_wires: usize, trainable: bool) -> Self {
        Self::new("rz", n_wires, true, trainable)
    }

    /// Create Hadamard layer
    pub fn hadamard(n_wires: usize) -> Self {
        Self::new("hadamard", n_wires, false, false)
    }
}

impl TQModule for TQOp1QAllLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for (wire, gate) in self.gates.iter_mut().enumerate() {
            gate.apply(qdev, &[wire])?;
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.gates.iter().flat_map(|g| g.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for gate in &mut self.gates {
            gate.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for gate in &mut self.gates {
            gate.static_off();
        }
    }

    fn name(&self) -> &str {
        "Op1QAllLayer"
    }

    fn zero_grad(&mut self) {
        for gate in &mut self.gates {
            gate.zero_grad();
        }
    }
}

/// Apply two-qubit operation to pairs of wires
pub struct TQOp2QAllLayer {
    /// Number of wires
    pub n_wires: usize,
    /// Gate type name
    pub op_name: String,
    /// Whether gates have parameters
    pub has_params: bool,
    /// Whether parameters are trainable
    pub trainable: bool,
    /// Jump between wire pairs (1 = nearest neighbor)
    pub jump: usize,
    /// Whether to connect last qubit to first (circular)
    pub circular: bool,
    /// Gate instances
    gates: Vec<Box<dyn TQOperator>>,
    static_mode: bool,
}

impl TQOp2QAllLayer {
    pub fn new(
        op_name: impl Into<String>,
        n_wires: usize,
        has_params: bool,
        trainable: bool,
        jump: usize,
        circular: bool,
    ) -> Self {
        let op_name = op_name.into();

        // Calculate number of gate pairs
        let n_pairs = if circular {
            n_wires
        } else {
            n_wires.saturating_sub(jump)
        };

        let gates: Vec<Box<dyn TQOperator>> = (0..n_pairs)
            .map(|_| create_two_qubit_gate(&op_name, has_params, trainable))
            .collect();

        Self {
            n_wires,
            op_name,
            has_params,
            trainable,
            jump,
            circular,
            gates,
            static_mode: false,
        }
    }

    /// Create CNOT layer
    pub fn cnot(n_wires: usize, circular: bool) -> Self {
        Self::new("cnot", n_wires, false, false, 1, circular)
    }

    /// Create CZ layer
    pub fn cz(n_wires: usize, circular: bool) -> Self {
        Self::new("cz", n_wires, false, false, 1, circular)
    }

    /// Create SWAP layer
    pub fn swap(n_wires: usize, circular: bool) -> Self {
        Self::new("swap", n_wires, false, false, 1, circular)
    }
}

impl TQModule for TQOp2QAllLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        let n_pairs = if self.circular {
            self.n_wires
        } else {
            self.n_wires.saturating_sub(self.jump)
        };

        for i in 0..n_pairs {
            let wire0 = i;
            let wire1 = (i + self.jump) % self.n_wires;

            if i < self.gates.len() {
                self.gates[i].apply(qdev, &[wire0, wire1])?;
            }
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.gates.iter().flat_map(|g| g.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for gate in &mut self.gates {
            gate.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for gate in &mut self.gates {
            gate.static_off();
        }
    }

    fn name(&self) -> &str {
        "Op2QAllLayer"
    }

    fn zero_grad(&mut self) {
        for gate in &mut self.gates {
            gate.zero_grad();
        }
    }
}

/// Layer configuration
#[derive(Debug, Clone)]
pub struct TQLayerConfig {
    pub n_wires: usize,
    pub n_blocks: usize,
    pub n_layers_per_block: Option<usize>,
}

impl TQLayerConfig {
    pub fn new(n_wires: usize, n_blocks: usize) -> Self {
        Self {
            n_wires,
            n_blocks,
            n_layers_per_block: None,
        }
    }

    pub fn with_layers_per_block(mut self, n: usize) -> Self {
        self.n_layers_per_block = Some(n);
        self
    }
}

/// Barren plateau layer (from TorchQuantum)
/// Pattern: H -> (RX -> RY -> RZ -> CZ) * n_blocks
pub struct TQBarrenLayer {
    config: TQLayerConfig,
    layers: Vec<Box<dyn TQModule>>,
    static_mode: bool,
}

impl TQBarrenLayer {
    pub fn new(config: TQLayerConfig) -> Self {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();

        // Initial Hadamard layer
        layers.push(Box::new(TQOp1QAllLayer::hadamard(config.n_wires)));

        // Blocks
        for _ in 0..config.n_blocks {
            layers.push(Box::new(TQOp1QAllLayer::rx(config.n_wires, true)));
            layers.push(Box::new(TQOp1QAllLayer::ry(config.n_wires, true)));
            layers.push(Box::new(TQOp1QAllLayer::rz(config.n_wires, true)));
            layers.push(Box::new(TQOp2QAllLayer::cz(config.n_wires, false)));
        }

        Self {
            config,
            layers,
            static_mode: false,
        }
    }
}

impl TQModule for TQBarrenLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for layer in &mut self.layers {
            layer.forward(qdev)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.config.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.config.n_wires = n_wires;
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for layer in &mut self.layers {
            layer.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for layer in &mut self.layers {
            layer.static_off();
        }
    }

    fn name(&self) -> &str {
        "BarrenLayer"
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }
}

/// RXYZCX layer (from TorchQuantum)
/// Pattern: (RX -> RY -> RZ -> CNOT) * n_blocks
pub struct TQRXYZCXLayer {
    config: TQLayerConfig,
    layers: Vec<Box<dyn TQModule>>,
    static_mode: bool,
}

impl TQRXYZCXLayer {
    pub fn new(config: TQLayerConfig) -> Self {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();

        for _ in 0..config.n_blocks {
            layers.push(Box::new(TQOp1QAllLayer::rx(config.n_wires, true)));
            layers.push(Box::new(TQOp1QAllLayer::ry(config.n_wires, true)));
            layers.push(Box::new(TQOp1QAllLayer::rz(config.n_wires, true)));
            layers.push(Box::new(TQOp2QAllLayer::cnot(config.n_wires, true)));
        }

        Self {
            config,
            layers,
            static_mode: false,
        }
    }
}

impl TQModule for TQRXYZCXLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for layer in &mut self.layers {
            layer.forward(qdev)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.config.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.config.n_wires = n_wires;
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for layer in &mut self.layers {
            layer.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for layer in &mut self.layers {
            layer.static_off();
        }
    }

    fn name(&self) -> &str {
        "RXYZCXLayer"
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }
}

/// Helper to create single-qubit gates
fn create_single_qubit_gate(name: &str, has_params: bool, trainable: bool) -> Box<dyn TQOperator> {
    match name.to_lowercase().as_str() {
        "rx" => Box::new(TQRx::new(has_params, trainable)),
        "ry" => Box::new(TQRy::new(has_params, trainable)),
        "rz" => Box::new(TQRz::new(has_params, trainable)),
        "h" | "hadamard" => Box::new(TQHadamard::new()),
        "x" | "paulix" => Box::new(TQPauliX::new()),
        "y" | "pauliy" => Box::new(TQPauliY::new()),
        "z" | "pauliz" => Box::new(TQPauliZ::new()),
        "s" => Box::new(TQS::new()),
        "t" => Box::new(TQT::new()),
        "sx" => Box::new(TQSX::new()),
        _ => Box::new(TQRy::new(has_params, trainable)), // Default
    }
}

/// Helper to create two-qubit gates
fn create_two_qubit_gate(name: &str, has_params: bool, trainable: bool) -> Box<dyn TQOperator> {
    match name.to_lowercase().as_str() {
        "cnot" | "cx" => Box::new(TQCNOT::new()),
        "cz" => Box::new(TQCZ::new()),
        "swap" => Box::new(TQSWAP::new()),
        // Parameterized two-qubit gates
        "rxx" => Box::new(TQRXX::new(has_params, trainable)),
        "ryy" => Box::new(TQRYY::new(has_params, trainable)),
        "rzz" => Box::new(TQRZZ::new(has_params, trainable)),
        "rzx" => Box::new(TQRZX::new(has_params, trainable)),
        // Controlled rotation gates
        "crx" => Box::new(TQCRX::new(has_params, trainable)),
        "cry" => Box::new(TQCRY::new(has_params, trainable)),
        "crz" => Box::new(TQCRZ::new(has_params, trainable)),
        _ => Box::new(TQCNOT::new()), // Default
    }
}

/// Farhi layer (from TorchQuantum)
/// Pattern: (RZX -> RXX) * n_blocks with circular connectivity
/// Implements the QAOA-style mixer for variational quantum circuits
pub struct TQFarhiLayer {
    config: TQLayerConfig,
    layers: Vec<Box<dyn TQModule>>,
    static_mode: bool,
}

impl TQFarhiLayer {
    pub fn new(config: TQLayerConfig) -> Self {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();

        for _ in 0..config.n_blocks {
            // RZX layer with circular connectivity
            layers.push(Box::new(TQOp2QAllLayer::new(
                "rzx",
                config.n_wires,
                true,
                true,
                1,
                true, // circular
            )));
            // RXX layer with circular connectivity
            layers.push(Box::new(TQOp2QAllLayer::new(
                "rxx",
                config.n_wires,
                true,
                true,
                1,
                true, // circular
            )));
        }

        Self {
            config,
            layers,
            static_mode: false,
        }
    }
}

impl TQModule for TQFarhiLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for layer in &mut self.layers {
            layer.forward(qdev)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.config.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.config.n_wires = n_wires;
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for layer in &mut self.layers {
            layer.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for layer in &mut self.layers {
            layer.static_off();
        }
    }

    fn name(&self) -> &str {
        "FarhiLayer"
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }
}

/// Maxwell layer (from TorchQuantum)
/// Pattern: (RX -> S -> CNOT -> RY -> T -> SWAP -> RZ -> H -> CNOT) * n_blocks
/// A hardware-efficient ansatz with diverse gate types
pub struct TQMaxwellLayer {
    config: TQLayerConfig,
    layers: Vec<Box<dyn TQModule>>,
    static_mode: bool,
}

impl TQMaxwellLayer {
    pub fn new(config: TQLayerConfig) -> Self {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();

        for _ in 0..config.n_blocks {
            // First block: RX -> S -> CNOT
            layers.push(Box::new(TQOp1QAllLayer::rx(config.n_wires, true)));
            layers.push(Box::new(TQOp1QAllLayer::new(
                "s",
                config.n_wires,
                false,
                false,
            )));
            layers.push(Box::new(TQOp2QAllLayer::new(
                "cnot",
                config.n_wires,
                false,
                false,
                1,
                true,
            )));

            // Second block: RY -> T -> SWAP
            layers.push(Box::new(TQOp1QAllLayer::ry(config.n_wires, true)));
            layers.push(Box::new(TQOp1QAllLayer::new(
                "t",
                config.n_wires,
                false,
                false,
            )));
            layers.push(Box::new(TQOp2QAllLayer::new(
                "swap",
                config.n_wires,
                false,
                false,
                1,
                true,
            )));

            // Third block: RZ -> H -> CNOT
            layers.push(Box::new(TQOp1QAllLayer::rz(config.n_wires, true)));
            layers.push(Box::new(TQOp1QAllLayer::hadamard(config.n_wires)));
            layers.push(Box::new(TQOp2QAllLayer::new(
                "cnot",
                config.n_wires,
                false,
                false,
                1,
                true,
            )));
        }

        Self {
            config,
            layers,
            static_mode: false,
        }
    }
}

impl TQModule for TQMaxwellLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for layer in &mut self.layers {
            layer.forward(qdev)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.config.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.config.n_wires = n_wires;
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for layer in &mut self.layers {
            layer.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for layer in &mut self.layers {
            layer.static_off();
        }
    }

    fn name(&self) -> &str {
        "MaxwellLayer"
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }
}

/// Seth layer (from TorchQuantum)
/// Pattern: (RY -> RZ -> CZ) * n_blocks
/// Simple efficient ansatz similar to EfficientSU2
pub struct TQSethLayer {
    config: TQLayerConfig,
    layers: Vec<Box<dyn TQModule>>,
    static_mode: bool,
}

impl TQSethLayer {
    pub fn new(config: TQLayerConfig) -> Self {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();

        for _ in 0..config.n_blocks {
            layers.push(Box::new(TQOp1QAllLayer::ry(config.n_wires, true)));
            layers.push(Box::new(TQOp1QAllLayer::rz(config.n_wires, true)));
            layers.push(Box::new(TQOp2QAllLayer::cz(config.n_wires, true)));
        }

        Self {
            config,
            layers,
            static_mode: false,
        }
    }
}

impl TQModule for TQSethLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for layer in &mut self.layers {
            layer.forward(qdev)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.config.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.config.n_wires = n_wires;
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for layer in &mut self.layers {
            layer.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for layer in &mut self.layers {
            layer.static_off();
        }
    }

    fn name(&self) -> &str {
        "SethLayer"
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }
}

/// Strong entangling layer (from TorchQuantum)
/// Pattern: (RX -> RY -> RZ -> CNOT) * n_blocks with varying entanglement patterns
/// Each block has different CNOT ranges for stronger entanglement
pub struct TQStrongEntanglingLayer {
    config: TQLayerConfig,
    layers: Vec<Box<dyn TQModule>>,
    static_mode: bool,
}

impl TQStrongEntanglingLayer {
    pub fn new(config: TQLayerConfig) -> Self {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();

        for block_idx in 0..config.n_blocks {
            // Rotation layers
            layers.push(Box::new(TQOp1QAllLayer::rx(config.n_wires, true)));
            layers.push(Box::new(TQOp1QAllLayer::ry(config.n_wires, true)));
            layers.push(Box::new(TQOp1QAllLayer::rz(config.n_wires, true)));

            // CNOT entanglement with varying range
            let jump = (block_idx % config.n_wires) + 1;
            layers.push(Box::new(TQOp2QAllLayer::new(
                "cnot",
                config.n_wires,
                false,
                false,
                jump,
                true, // circular
            )));
        }

        Self {
            config,
            layers,
            static_mode: false,
        }
    }
}

impl TQModule for TQStrongEntanglingLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for layer in &mut self.layers {
            layer.forward(qdev)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.config.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.config.n_wires = n_wires;
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for layer in &mut self.layers {
            layer.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for layer in &mut self.layers {
            layer.static_off();
        }
    }

    fn name(&self) -> &str {
        "StrongEntanglingLayer"
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }
}

/// Quantum Fourier Transform (QFT) layer
/// Implements the quantum Fourier transform on n qubits
pub struct TQQFTLayer {
    n_wires: usize,
    wires: Vec<usize>,
    do_swaps: bool,
    inverse: bool,
    static_mode: bool,
}

impl TQQFTLayer {
    pub fn new(n_wires: usize, do_swaps: bool, inverse: bool) -> Self {
        Self {
            n_wires,
            wires: (0..n_wires).collect(),
            do_swaps,
            inverse,
            static_mode: false,
        }
    }

    /// Create a standard QFT layer
    pub fn standard(n_wires: usize) -> Self {
        Self::new(n_wires, true, false)
    }

    /// Create an inverse QFT layer
    pub fn inverse(n_wires: usize) -> Self {
        Self::new(n_wires, true, true)
    }

    /// Create a QFT layer without final swaps
    pub fn no_swaps(n_wires: usize) -> Self {
        Self::new(n_wires, false, false)
    }

    /// Create a QFT layer with custom wires
    pub fn with_wires(wires: Vec<usize>, do_swaps: bool, inverse: bool) -> Self {
        let n_wires = wires.len();
        Self {
            n_wires,
            wires,
            do_swaps,
            inverse,
            static_mode: false,
        }
    }
}

impl TQModule for TQQFTLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        use super::gates::{TQHadamard, TQCU1, TQSWAP};
        use std::f64::consts::PI;

        if self.inverse {
            // Inverse QFT: swaps first, then reversed CU1 and H operations
            if self.do_swaps {
                for wire in 0..(self.n_wires / 2) {
                    let mut swap_gate = TQSWAP::new();
                    swap_gate.apply(
                        qdev,
                        &[self.wires[wire], self.wires[self.n_wires - wire - 1]],
                    )?;
                }
            }

            for top_wire in (0..self.n_wires).rev() {
                for wire in ((top_wire + 1)..self.n_wires).rev() {
                    let lam = -PI / (1 << (wire - top_wire)) as f64;
                    let mut cu1_gate = TQCU1::new(true, false);
                    cu1_gate.apply_with_params(
                        qdev,
                        &[self.wires[wire], self.wires[top_wire]],
                        Some(&[lam]),
                    )?;
                }
                let mut h_gate = TQHadamard::new();
                h_gate.apply(qdev, &[self.wires[top_wire]])?;
            }
        } else {
            // Standard QFT: H and CU1 operations, then swaps
            for top_wire in 0..self.n_wires {
                let mut h_gate = TQHadamard::new();
                h_gate.apply(qdev, &[self.wires[top_wire]])?;

                for wire in (top_wire + 1)..self.n_wires {
                    let lam = PI / (1 << (wire - top_wire)) as f64;
                    let mut cu1_gate = TQCU1::new(true, false);
                    cu1_gate.apply_with_params(
                        qdev,
                        &[self.wires[wire], self.wires[top_wire]],
                        Some(&[lam]),
                    )?;
                }
            }

            if self.do_swaps {
                for wire in 0..(self.n_wires / 2) {
                    let mut swap_gate = TQSWAP::new();
                    swap_gate.apply(
                        qdev,
                        &[self.wires[wire], self.wires[self.n_wires - wire - 1]],
                    )?;
                }
            }
        }

        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new() // QFT has no trainable parameters
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
        self.wires = (0..n_wires).collect();
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
        if self.inverse {
            "InverseQFTLayer"
        } else {
            "QFTLayer"
        }
    }
}

/// Entanglement pattern type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntanglementPattern {
    /// Linear: (0,1), (1,2), (2,3), ...
    Linear,
    /// Reverse linear: (n-1,n-2), (n-2,n-3), ...
    ReverseLinear,
    /// Circular: Linear + (n-1, 0)
    Circular,
    /// Full: All-to-all connectivity
    Full,
}

/// TwoLocal layer (from TorchQuantum)
/// Generic hardware-efficient ansatz with configurable rotation and entanglement gates
pub struct TQTwoLocalLayer {
    n_wires: usize,
    rotation_ops: Vec<String>,
    entanglement_ops: Vec<String>,
    entanglement_pattern: EntanglementPattern,
    reps: usize,
    skip_final_rotation: bool,
    layers: Vec<Box<dyn TQModule>>,
    static_mode: bool,
}

impl TQTwoLocalLayer {
    pub fn new(
        n_wires: usize,
        rotation_ops: Vec<&str>,
        entanglement_ops: Vec<&str>,
        entanglement_pattern: EntanglementPattern,
        reps: usize,
        skip_final_rotation: bool,
    ) -> Self {
        let rotation_ops: Vec<String> = rotation_ops.iter().map(|s| s.to_string()).collect();
        let entanglement_ops: Vec<String> =
            entanglement_ops.iter().map(|s| s.to_string()).collect();

        let layers = Self::build_layers(
            n_wires,
            &rotation_ops,
            &entanglement_ops,
            entanglement_pattern,
            reps,
            skip_final_rotation,
        );

        Self {
            n_wires,
            rotation_ops,
            entanglement_ops,
            entanglement_pattern,
            reps,
            skip_final_rotation,
            layers,
            static_mode: false,
        }
    }

    fn build_layers(
        n_wires: usize,
        rotation_ops: &[String],
        entanglement_ops: &[String],
        entanglement_pattern: EntanglementPattern,
        reps: usize,
        skip_final_rotation: bool,
    ) -> Vec<Box<dyn TQModule>> {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();

        let circular = matches!(entanglement_pattern, EntanglementPattern::Circular);

        for _ in 0..reps {
            // Rotation layer
            for op in rotation_ops {
                layers.push(Box::new(TQOp1QAllLayer::new(op, n_wires, true, true)));
            }

            // Entanglement layer
            if entanglement_pattern == EntanglementPattern::Full {
                // Full connectivity - all pairs
                for op in entanglement_ops {
                    layers.push(Box::new(TQOp2QDenseLayer::new(op, n_wires)));
                }
            } else {
                for op in entanglement_ops {
                    layers.push(Box::new(TQOp2QAllLayer::new(
                        op, n_wires, false, false, 1, circular,
                    )));
                }
            }
        }

        // Final rotation layer
        if !skip_final_rotation {
            for op in rotation_ops {
                layers.push(Box::new(TQOp1QAllLayer::new(op, n_wires, true, true)));
            }
        }

        layers
    }
}

impl TQModule for TQTwoLocalLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for layer in &mut self.layers {
            layer.forward(qdev)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
        self.layers = Self::build_layers(
            n_wires,
            &self.rotation_ops,
            &self.entanglement_ops,
            self.entanglement_pattern,
            self.reps,
            self.skip_final_rotation,
        );
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for layer in &mut self.layers {
            layer.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for layer in &mut self.layers {
            layer.static_off();
        }
    }

    fn name(&self) -> &str {
        "TwoLocalLayer"
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }
}

/// EfficientSU2 layer (from TorchQuantum/Qiskit)
/// Hardware-efficient ansatz with RY-RZ rotations and CX entanglement
pub struct TQEfficientSU2Layer {
    inner: TQTwoLocalLayer,
}

impl TQEfficientSU2Layer {
    pub fn new(n_wires: usize, reps: usize, entanglement: EntanglementPattern) -> Self {
        Self {
            inner: TQTwoLocalLayer::new(
                n_wires,
                vec!["ry", "rz"],
                vec!["cnot"],
                entanglement,
                reps,
                false,
            ),
        }
    }

    /// Create with default reverse linear entanglement
    pub fn default_entanglement(n_wires: usize, reps: usize) -> Self {
        Self::new(n_wires, reps, EntanglementPattern::ReverseLinear)
    }
}

impl TQModule for TQEfficientSU2Layer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        self.inner.forward(qdev)
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.inner.parameters()
    }

    fn n_wires(&self) -> Option<usize> {
        self.inner.n_wires()
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.inner.set_n_wires(n_wires);
    }

    fn is_static_mode(&self) -> bool {
        self.inner.is_static_mode()
    }

    fn static_on(&mut self) {
        self.inner.static_on();
    }

    fn static_off(&mut self) {
        self.inner.static_off();
    }

    fn name(&self) -> &str {
        "EfficientSU2Layer"
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }
}

/// RealAmplitudes layer (from TorchQuantum/Qiskit)
/// Hardware-efficient ansatz with RY rotations and CX entanglement
pub struct TQRealAmplitudesLayer {
    inner: TQTwoLocalLayer,
}

impl TQRealAmplitudesLayer {
    pub fn new(n_wires: usize, reps: usize, entanglement: EntanglementPattern) -> Self {
        Self {
            inner: TQTwoLocalLayer::new(
                n_wires,
                vec!["ry"],
                vec!["cnot"],
                entanglement,
                reps,
                false,
            ),
        }
    }

    /// Create with default reverse linear entanglement
    pub fn default_entanglement(n_wires: usize, reps: usize) -> Self {
        Self::new(n_wires, reps, EntanglementPattern::ReverseLinear)
    }
}

impl TQModule for TQRealAmplitudesLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        self.inner.forward(qdev)
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.inner.parameters()
    }

    fn n_wires(&self) -> Option<usize> {
        self.inner.n_wires()
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.inner.set_n_wires(n_wires);
    }

    fn is_static_mode(&self) -> bool {
        self.inner.is_static_mode()
    }

    fn static_on(&mut self) {
        self.inner.static_on();
    }

    fn static_off(&mut self) {
        self.inner.static_off();
    }

    fn name(&self) -> &str {
        "RealAmplitudesLayer"
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }
}

/// Dense two-qubit operation layer (full connectivity)
/// Applies gates to all pairs of qubits
pub struct TQOp2QDenseLayer {
    n_wires: usize,
    op_name: String,
    gates: Vec<Box<dyn TQOperator>>,
    static_mode: bool,
}

impl TQOp2QDenseLayer {
    pub fn new(op_name: impl Into<String>, n_wires: usize) -> Self {
        let op_name = op_name.into();
        let n_pairs = n_wires * (n_wires - 1) / 2;
        let gates: Vec<Box<dyn TQOperator>> = (0..n_pairs)
            .map(|_| create_two_qubit_gate(&op_name, false, false))
            .collect();

        Self {
            n_wires,
            op_name,
            gates,
            static_mode: false,
        }
    }

    /// Create CNOT dense layer
    pub fn cnot(n_wires: usize) -> Self {
        Self::new("cnot", n_wires)
    }

    /// Create CZ dense layer
    pub fn cz(n_wires: usize) -> Self {
        Self::new("cz", n_wires)
    }
}

impl TQModule for TQOp2QDenseLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        let mut gate_idx = 0;
        for i in 0..self.n_wires {
            for j in (i + 1)..self.n_wires {
                if gate_idx < self.gates.len() {
                    self.gates[gate_idx].apply(qdev, &[i, j])?;
                    gate_idx += 1;
                }
            }
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.gates.iter().flat_map(|g| g.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for gate in &mut self.gates {
            gate.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for gate in &mut self.gates {
            gate.static_off();
        }
    }

    fn name(&self) -> &str {
        "Op2QDenseLayer"
    }

    fn zero_grad(&mut self) {
        for gate in &mut self.gates {
            gate.zero_grad();
        }
    }
}

/// Random layer - applies random gates from a set
pub struct TQRandomLayer {
    n_wires: usize,
    n_ops: usize,
    rotation_ops: Vec<String>,
    entanglement_ops: Vec<String>,
    seed: Option<u64>,
    static_mode: bool,
    /// Cached gate sequence
    gate_sequence: Vec<(String, Vec<usize>)>,
}

impl TQRandomLayer {
    pub fn new(
        n_wires: usize,
        n_ops: usize,
        rotation_ops: Vec<&str>,
        entanglement_ops: Vec<&str>,
        seed: Option<u64>,
    ) -> Self {
        let rotation_ops: Vec<String> = rotation_ops.iter().map(|s| s.to_string()).collect();
        let entanglement_ops: Vec<String> =
            entanglement_ops.iter().map(|s| s.to_string()).collect();

        let gate_sequence =
            Self::generate_sequence(n_wires, n_ops, &rotation_ops, &entanglement_ops, seed);

        Self {
            n_wires,
            n_ops,
            rotation_ops,
            entanglement_ops,
            seed,
            static_mode: false,
            gate_sequence,
        }
    }

    fn generate_sequence(
        n_wires: usize,
        n_ops: usize,
        rotation_ops: &[String],
        entanglement_ops: &[String],
        seed: Option<u64>,
    ) -> Vec<(String, Vec<usize>)> {
        let mut sequence = Vec::with_capacity(n_ops);

        if let Some(s) = seed {
            fastrand::seed(s);
        }

        let all_ops: Vec<&String> = rotation_ops.iter().chain(entanglement_ops.iter()).collect();

        for _ in 0..n_ops {
            let op_idx = fastrand::usize(0..all_ops.len());
            let op_name = all_ops[op_idx].clone();

            let wires = if rotation_ops.contains(&op_name) {
                // Single-qubit gate
                vec![fastrand::usize(0..n_wires)]
            } else {
                // Two-qubit gate
                let w0 = fastrand::usize(0..n_wires);
                let mut w1 = fastrand::usize(0..n_wires);
                while w1 == w0 {
                    w1 = fastrand::usize(0..n_wires);
                }
                vec![w0, w1]
            };

            sequence.push((op_name, wires));
        }

        sequence
    }

    /// Regenerate the random gate sequence
    pub fn regenerate(&mut self) {
        self.gate_sequence = Self::generate_sequence(
            self.n_wires,
            self.n_ops,
            &self.rotation_ops,
            &self.entanglement_ops,
            self.seed,
        );
    }

    /// Create with default ops (RX, RY, RZ, CNOT)
    pub fn default_ops(n_wires: usize, n_ops: usize, seed: Option<u64>) -> Self {
        Self::new(n_wires, n_ops, vec!["rx", "ry", "rz"], vec!["cnot"], seed)
    }
}

impl TQModule for TQRandomLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for (op_name, wires) in &self.gate_sequence {
            if wires.len() == 1 {
                let mut gate = create_single_qubit_gate(op_name, true, false);
                gate.apply(qdev, wires)?;
            } else {
                let mut gate = create_two_qubit_gate(op_name, false, false);
                gate.apply(qdev, wires)?;
            }
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new() // Random layer doesn't expose trainable parameters
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
        self.regenerate();
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
        "RandomLayer"
    }
}

/// CX layer - applies CNOT gates in sequence
pub struct TQCXLayer {
    n_wires: usize,
    circular: bool,
    static_mode: bool,
}

impl TQCXLayer {
    pub fn new(n_wires: usize, circular: bool) -> Self {
        Self {
            n_wires,
            circular,
            static_mode: false,
        }
    }
}

impl TQModule for TQCXLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        let n_pairs = if self.circular {
            self.n_wires
        } else {
            self.n_wires.saturating_sub(1)
        };

        for i in 0..n_pairs {
            let wire0 = i;
            let wire1 = (i + 1) % self.n_wires;
            let mut gate = TQCNOT::new();
            gate.apply(qdev, &[wire0, wire1])?;
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
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
        "CXLayer"
    }
}

/// Triple CX layer - applies CNOT gates three times (for error correction patterns)
pub struct TQCXCXCXLayer {
    n_wires: usize,
    static_mode: bool,
}

impl TQCXCXCXLayer {
    pub fn new(n_wires: usize) -> Self {
        Self {
            n_wires,
            static_mode: false,
        }
    }
}

impl TQModule for TQCXCXCXLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        // Three rounds of CX gates
        for _ in 0..3 {
            for i in 0..(self.n_wires - 1) {
                let mut gate = TQCNOT::new();
                gate.apply(qdev, &[i, i + 1])?;
            }
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
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
        "CXCXCXLayer"
    }
}

// =============================================================================
// Chemistry/Fermionic Ansatz Layers
// =============================================================================

use super::gates::{TQFSimGate, TQGivensRotation};

/// Excitation preserving layer for quantum chemistry
///
/// This layer preserves the total number of excitations (particles) in the
/// quantum state, making it suitable for variational quantum eigensolver (VQE)
/// and other chemistry applications.
///
/// The layer uses fSim gates which naturally preserve particle number
/// while providing entanglement.
///
/// Pattern: For each pair of adjacent qubits, apply fSim gate with trainable
/// theta and phi parameters.
pub struct TQExcitationPreservingLayer {
    n_wires: usize,
    n_blocks: usize,
    circular: bool,
    gates: Vec<TQFSimGate>,
    static_mode: bool,
}

impl TQExcitationPreservingLayer {
    /// Create a new excitation preserving layer
    pub fn new(n_wires: usize, n_blocks: usize, circular: bool) -> Self {
        let n_pairs = if circular {
            n_wires
        } else {
            n_wires.saturating_sub(1)
        };
        let total_gates = n_pairs * n_blocks;

        let gates: Vec<TQFSimGate> = (0..total_gates)
            .map(|_| TQFSimGate::new(true, true))
            .collect();

        Self {
            n_wires,
            n_blocks,
            circular,
            gates,
            static_mode: false,
        }
    }

    /// Create with linear (non-circular) connectivity
    pub fn linear(n_wires: usize, n_blocks: usize) -> Self {
        Self::new(n_wires, n_blocks, false)
    }

    /// Create with circular connectivity
    pub fn circular(n_wires: usize, n_blocks: usize) -> Self {
        Self::new(n_wires, n_blocks, true)
    }
}

impl TQModule for TQExcitationPreservingLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        let n_pairs = if self.circular {
            self.n_wires
        } else {
            self.n_wires.saturating_sub(1)
        };

        let mut gate_idx = 0;
        for _ in 0..self.n_blocks {
            for pair in 0..n_pairs {
                let w0 = pair;
                let w1 = (pair + 1) % self.n_wires;

                if gate_idx < self.gates.len() {
                    self.gates[gate_idx].apply(qdev, &[w0, w1])?;
                    gate_idx += 1;
                }
            }
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.gates.iter().flat_map(|g| g.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
        // Recreate gates for new wire count
        let n_pairs = if self.circular {
            n_wires
        } else {
            n_wires.saturating_sub(1)
        };
        let total_gates = n_pairs * self.n_blocks;
        self.gates = (0..total_gates)
            .map(|_| TQFSimGate::new(true, true))
            .collect();
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for gate in &mut self.gates {
            gate.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for gate in &mut self.gates {
            gate.static_off();
        }
    }

    fn name(&self) -> &str {
        "ExcitationPreservingLayer"
    }

    fn zero_grad(&mut self) {
        for gate in &mut self.gates {
            gate.zero_grad();
        }
    }
}

/// Particle conserving layer using Givens rotations
///
/// This layer is specifically designed for quantum chemistry applications
/// where particle number conservation is critical. It uses Givens rotations
/// which naturally preserve the number of particles.
///
/// The layer can be used as an excitation preserving ansatz for molecular
/// simulations in variational quantum algorithms.
pub struct TQParticleConservingLayer {
    n_wires: usize,
    n_blocks: usize,
    pattern: GivensPattern,
    gates: Vec<TQGivensRotation>,
    static_mode: bool,
}

/// Pattern for Givens rotation application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GivensPattern {
    /// Adjacent pairs only
    Adjacent,
    /// Staircase pattern (used in chemistry)
    Staircase,
    /// Bricklayer pattern (alternating)
    Bricklayer,
}

impl TQParticleConservingLayer {
    /// Create a new particle conserving layer
    pub fn new(n_wires: usize, n_blocks: usize, pattern: GivensPattern) -> Self {
        let n_gates = Self::count_gates(n_wires, n_blocks, pattern);

        let gates: Vec<TQGivensRotation> = (0..n_gates)
            .map(|_| TQGivensRotation::new(true, true))
            .collect();

        Self {
            n_wires,
            n_blocks,
            pattern,
            gates,
            static_mode: false,
        }
    }

    /// Create with adjacent pattern (default)
    pub fn adjacent(n_wires: usize, n_blocks: usize) -> Self {
        Self::new(n_wires, n_blocks, GivensPattern::Adjacent)
    }

    /// Create with staircase pattern
    pub fn staircase(n_wires: usize, n_blocks: usize) -> Self {
        Self::new(n_wires, n_blocks, GivensPattern::Staircase)
    }

    /// Create with bricklayer pattern
    pub fn bricklayer(n_wires: usize, n_blocks: usize) -> Self {
        Self::new(n_wires, n_blocks, GivensPattern::Bricklayer)
    }

    fn count_gates(n_wires: usize, n_blocks: usize, pattern: GivensPattern) -> usize {
        match pattern {
            GivensPattern::Adjacent => (n_wires - 1) * n_blocks,
            GivensPattern::Staircase => {
                // Staircase: (0,1), (1,2), ..., (n-2,n-1), (n-2,n-1), ..., (1,2), (0,1)
                // This creates a "up and down" pattern
                (n_wires - 1) * 2 * n_blocks
            }
            GivensPattern::Bricklayer => {
                // Even pairs + odd pairs
                let even = n_wires / 2;
                let odd = (n_wires - 1) / 2;
                (even + odd) * n_blocks
            }
        }
    }

    fn get_wire_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();

        match self.pattern {
            GivensPattern::Adjacent => {
                for _ in 0..self.n_blocks {
                    for i in 0..(self.n_wires - 1) {
                        pairs.push((i, i + 1));
                    }
                }
            }
            GivensPattern::Staircase => {
                for _ in 0..self.n_blocks {
                    // Up staircase
                    for i in 0..(self.n_wires - 1) {
                        pairs.push((i, i + 1));
                    }
                    // Down staircase
                    for i in (0..(self.n_wires - 1)).rev() {
                        pairs.push((i, i + 1));
                    }
                }
            }
            GivensPattern::Bricklayer => {
                for _ in 0..self.n_blocks {
                    // Even pairs: (0,1), (2,3), ...
                    for i in (0..self.n_wires - 1).step_by(2) {
                        pairs.push((i, i + 1));
                    }
                    // Odd pairs: (1,2), (3,4), ...
                    for i in (1..self.n_wires - 1).step_by(2) {
                        pairs.push((i, i + 1));
                    }
                }
            }
        }

        pairs
    }
}

impl TQModule for TQParticleConservingLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        let pairs = self.get_wire_pairs();

        for (gate_idx, (w0, w1)) in pairs.iter().enumerate() {
            if gate_idx < self.gates.len() {
                self.gates[gate_idx].apply(qdev, &[*w0, *w1])?;
            }
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.gates.iter().flat_map(|g| g.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
        let n_gates = Self::count_gates(n_wires, self.n_blocks, self.pattern);
        self.gates = (0..n_gates)
            .map(|_| TQGivensRotation::new(true, true))
            .collect();
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for gate in &mut self.gates {
            gate.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for gate in &mut self.gates {
            gate.static_off();
        }
    }

    fn name(&self) -> &str {
        "ParticleConservingLayer"
    }

    fn zero_grad(&mut self) {
        for gate in &mut self.gates {
            gate.zero_grad();
        }
    }
}

/// Hardware Efficient 2 Layer - Alternative hardware efficient ansatz
///
/// This is an alternative to the standard hardware efficient ansatz with
/// different rotation and entanglement patterns. It uses:
/// - Initial layer of RY rotations
/// - Alternating CZ and CNOT entanglement
/// - RX and RZ rotations between entanglement layers
///
/// Pattern: RY -> (CZ -> RX -> CNOT -> RZ) * n_blocks
pub struct TQHardwareEfficient2Layer {
    config: TQLayerConfig,
    layers: Vec<Box<dyn TQModule>>,
    static_mode: bool,
}

impl TQHardwareEfficient2Layer {
    pub fn new(config: TQLayerConfig) -> Self {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();

        // Initial RY layer
        layers.push(Box::new(TQOp1QAllLayer::ry(config.n_wires, true)));

        for _ in 0..config.n_blocks {
            // CZ entanglement
            layers.push(Box::new(TQOp2QAllLayer::cz(config.n_wires, true)));

            // RX rotation
            layers.push(Box::new(TQOp1QAllLayer::rx(config.n_wires, true)));

            // CNOT entanglement (shifted by 1)
            layers.push(Box::new(TQOp2QAllLayer::new(
                "cnot",
                config.n_wires,
                false,
                false,
                1,
                true,
            )));

            // RZ rotation
            layers.push(Box::new(TQOp1QAllLayer::rz(config.n_wires, true)));
        }

        Self {
            config,
            layers,
            static_mode: false,
        }
    }
}

impl TQModule for TQHardwareEfficient2Layer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for layer in &mut self.layers {
            layer.forward(qdev)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.config.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.config.n_wires = n_wires;
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for layer in &mut self.layers {
            layer.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for layer in &mut self.layers {
            layer.static_off();
        }
    }

    fn name(&self) -> &str {
        "HardwareEfficient2Layer"
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }
}

/// UCCSD-inspired layer for molecular simulations
///
/// This layer provides a simplified version of the Unitary Coupled Cluster
/// Singles and Doubles (UCCSD) ansatz commonly used in VQE for chemistry.
///
/// It uses Givens rotations to implement fermionic excitations.
pub struct TQUCCSDLayer {
    n_wires: usize,
    n_electrons: usize,
    gates: Vec<TQGivensRotation>,
    static_mode: bool,
}

impl TQUCCSDLayer {
    /// Create a new UCCSD-inspired layer
    ///
    /// # Arguments
    /// * `n_wires` - Number of qubits (spin-orbitals)
    /// * `n_electrons` - Number of electrons
    pub fn new(n_wires: usize, n_electrons: usize) -> Self {
        // Calculate number of single and double excitations
        let n_virtual = n_wires - n_electrons;
        let n_singles = n_electrons * n_virtual;
        let n_doubles = n_singles * (n_singles - 1) / 2;

        // Simplified: just use n_singles Givens rotations
        // A full UCCSD would need many more gates
        let n_gates = n_singles.min(n_wires * 2);

        let gates: Vec<TQGivensRotation> = (0..n_gates)
            .map(|_| TQGivensRotation::new(true, true))
            .collect();

        Self {
            n_wires,
            n_electrons,
            gates,
            static_mode: false,
        }
    }
}

impl TQModule for TQUCCSDLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        // Apply Givens rotations in a pattern that mimics UCCSD excitations
        let mut gate_idx = 0;

        // Single excitations: connect occupied to virtual orbitals
        for occ in 0..self.n_electrons.min(self.n_wires) {
            for virt in self.n_electrons..self.n_wires {
                if gate_idx < self.gates.len() {
                    // Apply Givens rotation between occupied and virtual
                    if virt > occ {
                        // Chain Givens rotations to connect distant orbitals
                        self.gates[gate_idx].apply(qdev, &[occ, occ + 1])?;
                        gate_idx += 1;
                    }
                }
            }
        }

        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.gates.iter().flat_map(|g| g.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for gate in &mut self.gates {
            gate.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for gate in &mut self.gates {
            gate.static_off();
        }
    }

    fn name(&self) -> &str {
        "UCCSDLayer"
    }

    fn zero_grad(&mut self) {
        for gate in &mut self.gates {
            gate.zero_grad();
        }
    }
}

/// Symmetry preserving layer
///
/// This layer preserves certain symmetries of the quantum state,
/// useful for applications where conservation laws must be respected.
pub struct TQSymmetryPreservingLayer {
    n_wires: usize,
    n_blocks: usize,
    symmetry_type: SymmetryType,
    layers: Vec<Box<dyn TQModule>>,
    static_mode: bool,
}

/// Type of symmetry to preserve
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryType {
    /// Particle number conservation
    ParticleNumber,
    /// Spin conservation
    SpinConservation,
    /// Time reversal symmetry
    TimeReversal,
}

impl TQSymmetryPreservingLayer {
    pub fn new(n_wires: usize, n_blocks: usize, symmetry_type: SymmetryType) -> Self {
        let layers = Self::build_layers(n_wires, n_blocks, symmetry_type);

        Self {
            n_wires,
            n_blocks,
            symmetry_type,
            layers,
            static_mode: false,
        }
    }

    fn build_layers(
        n_wires: usize,
        n_blocks: usize,
        symmetry_type: SymmetryType,
    ) -> Vec<Box<dyn TQModule>> {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();

        match symmetry_type {
            SymmetryType::ParticleNumber => {
                // Use fSim gates which preserve particle number
                for _ in 0..n_blocks {
                    layers.push(Box::new(TQExcitationPreservingLayer::new(
                        n_wires, 1, false,
                    )));
                }
            }
            SymmetryType::SpinConservation => {
                // Use RZ rotations (diagonal) and XX+YY interactions
                for _ in 0..n_blocks {
                    layers.push(Box::new(TQOp1QAllLayer::rz(n_wires, true)));
                    layers.push(Box::new(TQOp2QAllLayer::new(
                        "rxx", n_wires, true, true, 1, false,
                    )));
                    layers.push(Box::new(TQOp2QAllLayer::new(
                        "ryy", n_wires, true, true, 1, false,
                    )));
                }
            }
            SymmetryType::TimeReversal => {
                // Use real operations only (RY rotations and real entanglement)
                for _ in 0..n_blocks {
                    layers.push(Box::new(TQOp1QAllLayer::ry(n_wires, true)));
                    layers.push(Box::new(TQOp2QAllLayer::cnot(n_wires, true)));
                }
            }
        }

        layers
    }

    pub fn particle_number(n_wires: usize, n_blocks: usize) -> Self {
        Self::new(n_wires, n_blocks, SymmetryType::ParticleNumber)
    }

    pub fn spin_conserving(n_wires: usize, n_blocks: usize) -> Self {
        Self::new(n_wires, n_blocks, SymmetryType::SpinConservation)
    }

    pub fn time_reversal(n_wires: usize, n_blocks: usize) -> Self {
        Self::new(n_wires, n_blocks, SymmetryType::TimeReversal)
    }
}

impl TQModule for TQSymmetryPreservingLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for layer in &mut self.layers {
            layer.forward(qdev)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
        self.layers = Self::build_layers(n_wires, self.n_blocks, self.symmetry_type);
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
        for layer in &mut self.layers {
            layer.static_on();
        }
    }

    fn static_off(&mut self) {
        self.static_mode = false;
        for layer in &mut self.layers {
            layer.static_off();
        }
    }

    fn name(&self) -> &str {
        "SymmetryPreservingLayer"
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }
}
