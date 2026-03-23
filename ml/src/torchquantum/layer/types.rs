//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::{create_single_qubit_gate, create_two_qubit_gate};
use crate::torchquantum::gates::{TQFSimGate, TQGivensRotation};
use crate::torchquantum::{TQModule, TQOperator};

/// TwoLocal layer (from TorchQuantum)
/// Generic hardware-efficient ansatz with configurable rotation and entanglement gates
pub struct TQTwoLocalLayer {
    pub(super) n_wires: usize,
    pub(super) rotation_ops: Vec<String>,
    pub(super) entanglement_ops: Vec<String>,
    pub(super) entanglement_pattern: EntanglementPattern,
    pub(super) reps: usize,
    pub(super) skip_final_rotation: bool,
    pub(super) layers: Vec<Box<dyn TQModule>>,
    pub(super) static_mode: bool,
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
    pub(super) fn build_layers(
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
            for op in rotation_ops {
                layers.push(Box::new(TQOp1QAllLayer::new(op, n_wires, true, true)));
            }
            if entanglement_pattern == EntanglementPattern::Full {
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
        if !skip_final_rotation {
            for op in rotation_ops {
                layers.push(Box::new(TQOp1QAllLayer::new(op, n_wires, true, true)));
            }
        }
        layers
    }
}
/// Dense two-qubit operation layer (full connectivity)
/// Applies gates to all pairs of qubits
pub struct TQOp2QDenseLayer {
    pub(super) n_wires: usize,
    op_name: String,
    pub(super) gates: Vec<Box<dyn TQOperator>>,
    pub(super) static_mode: bool,
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
/// EfficientSU2 layer (from TorchQuantum/Qiskit)
/// Hardware-efficient ansatz with RY-RZ rotations and CX entanglement
pub struct TQEfficientSU2Layer {
    pub(super) inner: TQTwoLocalLayer,
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
/// RXYZCX layer (from TorchQuantum)
/// Pattern: (RX -> RY -> RZ -> CNOT) * n_blocks
pub struct TQRXYZCXLayer {
    pub(super) config: TQLayerConfig,
    pub(super) layers: Vec<Box<dyn TQModule>>,
    pub(super) static_mode: bool,
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
/// CX layer - applies CNOT gates in sequence
pub struct TQCXLayer {
    pub(super) n_wires: usize,
    pub(super) circular: bool,
    pub(super) static_mode: bool,
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
/// Barren plateau layer (from TorchQuantum)
/// Pattern: H -> (RX -> RY -> RZ -> CZ) * n_blocks
pub struct TQBarrenLayer {
    pub(super) config: TQLayerConfig,
    pub(super) layers: Vec<Box<dyn TQModule>>,
    pub(super) static_mode: bool,
}
impl TQBarrenLayer {
    pub fn new(config: TQLayerConfig) -> Self {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();
        layers.push(Box::new(TQOp1QAllLayer::hadamard(config.n_wires)));
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
/// Symmetry preserving layer
///
/// This layer preserves certain symmetries of the quantum state,
/// useful for applications where conservation laws must be respected.
pub struct TQSymmetryPreservingLayer {
    pub(super) n_wires: usize,
    pub(super) n_blocks: usize,
    pub(super) symmetry_type: SymmetryType,
    pub(super) layers: Vec<Box<dyn TQModule>>,
    pub(super) static_mode: bool,
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
    pub(super) fn build_layers(
        n_wires: usize,
        n_blocks: usize,
        symmetry_type: SymmetryType,
    ) -> Vec<Box<dyn TQModule>> {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();
        match symmetry_type {
            SymmetryType::ParticleNumber => {
                for _ in 0..n_blocks {
                    layers.push(Box::new(TQExcitationPreservingLayer::new(
                        n_wires, 1, false,
                    )));
                }
            }
            SymmetryType::SpinConservation => {
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
/// Seth layer (from TorchQuantum)
/// Pattern: (RY -> RZ -> CZ) * n_blocks
/// Simple efficient ansatz similar to EfficientSU2
pub struct TQSethLayer {
    pub(super) config: TQLayerConfig,
    pub(super) layers: Vec<Box<dyn TQModule>>,
    pub(super) static_mode: bool,
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
/// Maxwell layer (from TorchQuantum)
/// Pattern: (RX -> S -> CNOT -> RY -> T -> SWAP -> RZ -> H -> CNOT) * n_blocks
/// A hardware-efficient ansatz with diverse gate types
pub struct TQMaxwellLayer {
    pub(super) config: TQLayerConfig,
    pub(super) layers: Vec<Box<dyn TQModule>>,
    pub(super) static_mode: bool,
}
impl TQMaxwellLayer {
    pub fn new(config: TQLayerConfig) -> Self {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();
        for _ in 0..config.n_blocks {
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
/// Triple CX layer - applies CNOT gates three times (for error correction patterns)
pub struct TQCXCXCXLayer {
    pub(super) n_wires: usize,
    pub(super) static_mode: bool,
}
impl TQCXCXCXLayer {
    pub fn new(n_wires: usize) -> Self {
        Self {
            n_wires,
            static_mode: false,
        }
    }
}
/// Random layer - applies random gates from a set
pub struct TQRandomLayer {
    pub(super) n_wires: usize,
    n_ops: usize,
    rotation_ops: Vec<String>,
    entanglement_ops: Vec<String>,
    seed: Option<u64>,
    pub(super) static_mode: bool,
    /// Cached gate sequence
    pub(super) gate_sequence: Vec<(String, Vec<usize>)>,
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
                vec![fastrand::usize(0..n_wires)]
            } else {
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
/// Particle conserving layer using Givens rotations
///
/// This layer is specifically designed for quantum chemistry applications
/// where particle number conservation is critical. It uses Givens rotations
/// which naturally preserve the number of particles.
///
/// The layer can be used as an excitation preserving ansatz for molecular
/// simulations in variational quantum algorithms.
pub struct TQParticleConservingLayer {
    pub(super) n_wires: usize,
    pub(super) n_blocks: usize,
    pub(super) pattern: GivensPattern,
    pub(super) gates: Vec<TQGivensRotation>,
    pub(super) static_mode: bool,
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
    pub(super) fn count_gates(n_wires: usize, n_blocks: usize, pattern: GivensPattern) -> usize {
        match pattern {
            GivensPattern::Adjacent => (n_wires - 1) * n_blocks,
            GivensPattern::Staircase => (n_wires - 1) * 2 * n_blocks,
            GivensPattern::Bricklayer => {
                let even = n_wires / 2;
                let odd = (n_wires - 1) / 2;
                (even + odd) * n_blocks
            }
        }
    }
    pub(super) fn get_wire_pairs(&self) -> Vec<(usize, usize)> {
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
                    for i in 0..(self.n_wires - 1) {
                        pairs.push((i, i + 1));
                    }
                    for i in (0..(self.n_wires - 1)).rev() {
                        pairs.push((i, i + 1));
                    }
                }
            }
            GivensPattern::Bricklayer => {
                for _ in 0..self.n_blocks {
                    for i in (0..self.n_wires - 1).step_by(2) {
                        pairs.push((i, i + 1));
                    }
                    for i in (1..self.n_wires - 1).step_by(2) {
                        pairs.push((i, i + 1));
                    }
                }
            }
        }
        pairs
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
    pub(super) gates: Vec<Box<dyn TQOperator>>,
    pub(super) static_mode: bool,
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
    pub(super) gates: Vec<Box<dyn TQOperator>>,
    pub(super) static_mode: bool,
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
    pub(super) config: TQLayerConfig,
    pub(super) layers: Vec<Box<dyn TQModule>>,
    pub(super) static_mode: bool,
}
impl TQHardwareEfficient2Layer {
    pub fn new(config: TQLayerConfig) -> Self {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();
        layers.push(Box::new(TQOp1QAllLayer::ry(config.n_wires, true)));
        for _ in 0..config.n_blocks {
            layers.push(Box::new(TQOp2QAllLayer::cz(config.n_wires, true)));
            layers.push(Box::new(TQOp1QAllLayer::rx(config.n_wires, true)));
            layers.push(Box::new(TQOp2QAllLayer::new(
                "cnot",
                config.n_wires,
                false,
                false,
                1,
                true,
            )));
            layers.push(Box::new(TQOp1QAllLayer::rz(config.n_wires, true)));
        }
        Self {
            config,
            layers,
            static_mode: false,
        }
    }
}
/// Farhi layer (from TorchQuantum)
/// Pattern: (RZX -> RXX) * n_blocks with circular connectivity
/// Implements the QAOA-style mixer for variational quantum circuits
pub struct TQFarhiLayer {
    pub(super) config: TQLayerConfig,
    pub(super) layers: Vec<Box<dyn TQModule>>,
    pub(super) static_mode: bool,
}
impl TQFarhiLayer {
    pub fn new(config: TQLayerConfig) -> Self {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();
        for _ in 0..config.n_blocks {
            layers.push(Box::new(TQOp2QAllLayer::new(
                "rzx",
                config.n_wires,
                true,
                true,
                1,
                true,
            )));
            layers.push(Box::new(TQOp2QAllLayer::new(
                "rxx",
                config.n_wires,
                true,
                true,
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
/// UCCSD-inspired layer for molecular simulations
///
/// This layer provides a simplified version of the Unitary Coupled Cluster
/// Singles and Doubles (UCCSD) ansatz commonly used in VQE for chemistry.
///
/// It uses Givens rotations to implement fermionic excitations.
pub struct TQUCCSDLayer {
    pub(super) n_wires: usize,
    pub(super) n_electrons: usize,
    pub(super) gates: Vec<TQGivensRotation>,
    pub(super) static_mode: bool,
}
impl TQUCCSDLayer {
    /// Create a new UCCSD-inspired layer
    ///
    /// # Arguments
    /// * `n_wires` - Number of qubits (spin-orbitals)
    /// * `n_electrons` - Number of electrons
    pub fn new(n_wires: usize, n_electrons: usize) -> Self {
        let n_virtual = n_wires - n_electrons;
        let n_singles = n_electrons * n_virtual;
        let n_doubles = n_singles * (n_singles - 1) / 2;
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
/// RealAmplitudes layer (from TorchQuantum/Qiskit)
/// Hardware-efficient ansatz with RY rotations and CX entanglement
pub struct TQRealAmplitudesLayer {
    pub(super) inner: TQTwoLocalLayer,
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
/// Strong entangling layer (from TorchQuantum)
/// Pattern: (RX -> RY -> RZ -> CNOT) * n_blocks with varying entanglement patterns
/// Each block has different CNOT ranges for stronger entanglement
pub struct TQStrongEntanglingLayer {
    pub(super) config: TQLayerConfig,
    pub(super) layers: Vec<Box<dyn TQModule>>,
    pub(super) static_mode: bool,
}
impl TQStrongEntanglingLayer {
    pub fn new(config: TQLayerConfig) -> Self {
        let mut layers: Vec<Box<dyn TQModule>> = Vec::new();
        for block_idx in 0..config.n_blocks {
            layers.push(Box::new(TQOp1QAllLayer::rx(config.n_wires, true)));
            layers.push(Box::new(TQOp1QAllLayer::ry(config.n_wires, true)));
            layers.push(Box::new(TQOp1QAllLayer::rz(config.n_wires, true)));
            let jump = (block_idx % config.n_wires) + 1;
            layers.push(Box::new(TQOp2QAllLayer::new(
                "cnot",
                config.n_wires,
                false,
                false,
                jump,
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
    pub(super) n_wires: usize,
    pub(super) n_blocks: usize,
    pub(super) circular: bool,
    pub(super) gates: Vec<TQFSimGate>,
    pub(super) static_mode: bool,
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
/// Quantum Fourier Transform (QFT) layer
/// Implements the quantum Fourier transform on n qubits
pub struct TQQFTLayer {
    pub(super) n_wires: usize,
    pub(super) wires: Vec<usize>,
    pub(super) do_swaps: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
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
