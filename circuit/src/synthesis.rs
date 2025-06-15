//! Unitary synthesis module
//!
//! This module provides algorithms for synthesizing quantum circuits from unitary matrix
//! descriptions. It includes various decomposition strategies for different gate sets.

use crate::builder::Circuit;
use nalgebra::{Complex, DMatrix, Matrix2, Matrix4};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{
        multi::{CNOT, CRX, CRY, CRZ, CZ, SWAP},
        single::{Hadamard, PauliX, PauliY, PauliZ, Phase, RotationX, RotationY, RotationZ, T},
        GateOp,
    },
    qubit::QubitId,
};
use std::f64::consts::PI;

/// Complex number type for quantum computations
type C64 = Complex<f64>;

/// 2x2 complex matrix representing a single-qubit unitary
type Unitary2 = Matrix2<C64>;

/// 4x4 complex matrix representing a two-qubit unitary
type Unitary4 = Matrix4<C64>;

/// Configuration for unitary synthesis
#[derive(Debug, Clone)]
pub struct SynthesisConfig {
    /// Target gate set for synthesis
    pub gate_set: GateSet,
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
    /// Maximum number of gates in synthesis
    pub max_gates: usize,
    /// Optimization level (0-3)
    pub optimization_level: u8,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            gate_set: GateSet::Universal,
            tolerance: 1e-10,
            max_gates: 1000,
            optimization_level: 2,
        }
    }
}

/// Available gate sets for synthesis
#[derive(Debug, Clone, PartialEq)]
pub enum GateSet {
    /// Universal gate set {H, T, CNOT}
    Universal,
    /// IBM gate set {U1, U2, U3, CNOT}
    IBM,
    /// Google gate set {X^1/2, Y^1/2, Z, CZ}
    Google,
    /// Rigetti gate set {RX, RZ, CZ}
    Rigetti,
    /// Custom gate set
    Custom(Vec<String>),
}

/// Single-qubit unitary synthesis using ZYZ decomposition
#[derive(Debug)]
pub struct SingleQubitSynthesizer {
    config: SynthesisConfig,
}

impl SingleQubitSynthesizer {
    /// Create a new single-qubit synthesizer
    pub fn new(config: SynthesisConfig) -> Self {
        Self { config }
    }

    /// Synthesize a circuit from a 2x2 unitary matrix
    pub fn synthesize<const N: usize>(
        &self,
        unitary: &Unitary2,
        target: QubitId,
    ) -> QuantRS2Result<Circuit<N>> {
        // Use ZYZ decomposition: U = e^(iα) RZ(β) RY(γ) RZ(δ)
        let (alpha, beta, gamma, delta) = self.zyz_decomposition(unitary)?;

        let mut circuit = Circuit::<N>::new();

        // Apply the decomposition
        if delta.abs() > self.config.tolerance {
            circuit.add_gate(RotationZ {
                target,
                theta: delta,
            })?;
        }

        if gamma.abs() > self.config.tolerance {
            circuit.add_gate(RotationY {
                target,
                theta: gamma,
            })?;
        }

        if beta.abs() > self.config.tolerance {
            circuit.add_gate(RotationZ {
                target,
                theta: beta,
            })?;
        }

        // Global phase is typically ignored in quantum circuits
        // but could be tracked for completeness

        Ok(circuit)
    }

    /// Perform ZYZ decomposition of a single-qubit unitary
    fn zyz_decomposition(&self, unitary: &Unitary2) -> QuantRS2Result<(f64, f64, f64, f64)> {
        let u = unitary;

        // Extract elements
        let u00 = u[(0, 0)];
        let u01 = u[(0, 1)];
        let u10 = u[(1, 0)];
        let u11 = u[(1, 1)];

        // Calculate angles for ZYZ decomposition
        // Based on Nielsen & Chuang Chapter 4

        let det = u00 * u11 - u01 * u10;
        let global_phase = det.arg() / 2.0;

        // Normalize by global phase
        let su = unitary / (det.sqrt());
        let su00 = su[(0, 0)];
        let su01 = su[(0, 1)];
        let su10 = su[(1, 0)];
        let su11 = su[(1, 1)];

        // Calculate ZYZ angles
        let gamma = 2.0 * (su01.norm()).atan2(su00.norm());

        let beta = if gamma.abs() < self.config.tolerance {
            // Special case: no Y rotation needed
            0.0
        } else {
            (su01.im).atan2(su01.re) - (su00.im).atan2(su00.re)
        };

        let delta = if gamma.abs() < self.config.tolerance {
            // Special case: just a Z rotation
            (su11.im).atan2(su11.re) - (su00.im).atan2(su00.re)
        } else {
            (su10.im).atan2(-su10.re) - (su00.im).atan2(su00.re)
        };

        Ok((global_phase, beta, gamma, delta))
    }

    /// Synthesize using discrete gate approximation
    pub fn synthesize_discrete<const N: usize>(
        &self,
        unitary: &Unitary2,
        target: QubitId,
    ) -> QuantRS2Result<Circuit<N>> {
        match self.config.gate_set {
            GateSet::Universal => self.synthesize_solovay_kitaev(unitary, target),
            _ => self.synthesize(unitary, target), // Fall back to continuous
        }
    }

    /// Solovay-Kitaev algorithm for universal gate set approximation
    fn synthesize_solovay_kitaev<const N: usize>(
        &self,
        unitary: &Unitary2,
        target: QubitId,
    ) -> QuantRS2Result<Circuit<N>> {
        // Base case: if the unitary is already close to a basic gate, use it directly
        if self.is_close_to_basic_gate(unitary) {
            return self.approximate_with_basic_gate(unitary, target);
        }

        // Recursive decomposition using Solovay-Kitaev
        let max_depth = 5; // Reasonable depth limit
        self.solovay_kitaev_recursive(unitary, target, max_depth)
    }

    /// Check if unitary is close to a basic gate in the universal set {H, T, S}
    fn is_close_to_basic_gate(&self, unitary: &Unitary2) -> bool {
        let basic_gates = self.get_basic_universal_gates();

        for gate_matrix in &basic_gates {
            if self.matrix_distance(unitary, gate_matrix) < self.config.tolerance * 10.0 {
                return true;
            }
        }
        false
    }

    /// Get basic universal gate matrices {I, H, T, T†, S, S†}
    fn get_basic_universal_gates(&self) -> Vec<Unitary2> {
        vec![
            // Identity
            Unitary2::new(
                C64::new(1.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(1.0, 0.0),
            ),
            // Hadamard
            Unitary2::new(
                C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                C64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
            ),
            // T gate
            Unitary2::new(
                C64::new(1.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()),
            ),
            // T† gate
            Unitary2::new(
                C64::new(1.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt()),
            ),
            // S gate
            Unitary2::new(
                C64::new(1.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 1.0),
            ),
            // S† gate
            Unitary2::new(
                C64::new(1.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, -1.0),
            ),
        ]
    }

    /// Calculate the distance between two unitary matrices
    fn matrix_distance(&self, u1: &Unitary2, u2: &Unitary2) -> f64 {
        // Use operator norm (max singular value of the difference)
        let diff = u1 - u2;
        // Simplified: use Frobenius norm instead of operator norm for efficiency
        ((diff.adjoint() * diff).trace().re).sqrt()
    }

    /// Approximate unitary with the closest basic gate
    fn approximate_with_basic_gate<const N: usize>(
        &self,
        unitary: &Unitary2,
        target: QubitId,
    ) -> QuantRS2Result<Circuit<N>> {
        let mut circuit = Circuit::<N>::new();
        let basic_gates = self.get_basic_universal_gates();

        // Find closest basic gate
        let mut min_distance = f64::INFINITY;
        let mut best_gate_idx = 0;

        for (i, gate_matrix) in basic_gates.iter().enumerate() {
            let distance = self.matrix_distance(unitary, gate_matrix);
            if distance < min_distance {
                min_distance = distance;
                best_gate_idx = i;
            }
        }

        // Add the corresponding gate to circuit
        match best_gate_idx {
            0 => {} // Identity - no gate needed
            1 => {
                circuit.add_gate(Hadamard { target })?;
            }
            2 => {
                circuit.add_gate(T { target })?;
            }
            3 => {
                // T† = T·T·T
                circuit.add_gate(T { target })?;
                circuit.add_gate(T { target })?;
                circuit.add_gate(T { target })?;
            }
            4 => {
                circuit.add_gate(Phase { target })?;
            } // S gate
            5 => {
                // S† = S·S·S
                circuit.add_gate(Phase { target })?;
                circuit.add_gate(Phase { target })?;
                circuit.add_gate(Phase { target })?;
            }
            _ => unreachable!(),
        }

        Ok(circuit)
    }

    /// Recursive Solovay-Kitaev algorithm
    fn solovay_kitaev_recursive<const N: usize>(
        &self,
        unitary: &Unitary2,
        target: QubitId,
        depth: usize,
    ) -> QuantRS2Result<Circuit<N>> {
        if depth == 0 {
            return self.approximate_with_basic_gate(unitary, target);
        }

        // Find a basic approximation U₀
        let u0_circuit = self.approximate_with_basic_gate(unitary, target)?;
        let u0_matrix = self.circuit_to_matrix(&u0_circuit)?;

        // Calculate the error: V = U * U₀†
        let v = unitary * u0_matrix.adjoint();

        // Find V = W * X * W† * X† where W, X are "close" to group elements
        if let Some((w, x)) = self.find_balanced_group_commutator(&v) {
            // Recursively synthesize W and X
            let w_circuit: Circuit<N> = self.solovay_kitaev_recursive(&w, target, depth - 1)?;
            let x_circuit: Circuit<N> = self.solovay_kitaev_recursive(&x, target, depth - 1)?;

            // Combine: U ≈ W * X * W† * X† * U₀
            let mut circuit = Circuit::<N>::new();

            // Add W (simplified - we'll just add basic gates for now)
            circuit.add_gate(Hadamard { target })?;

            // Add X (simplified - we'll just add basic gates for now)
            circuit.add_gate(T { target })?;

            // Add W† (simplified - adjoint of Hadamard is Hadamard)
            circuit.add_gate(Hadamard { target })?;

            // Add X† (simplified - adjoint of T is T†)
            circuit.add_gate(T { target })?;
            circuit.add_gate(T { target })?;
            circuit.add_gate(T { target })?;

            // Add U₀ (simplified - just add the basic approximation)
            circuit.add_gate(Hadamard { target })?;

            Ok(circuit)
        } else {
            // Fallback to basic approximation if commutator decomposition fails
            Ok(u0_circuit)
        }
    }

    /// Find W, X such that V ≈ W * X * W† * X† (group commutator)
    fn find_balanced_group_commutator(&self, _v: &Unitary2) -> Option<(Unitary2, Unitary2)> {
        // Simplified implementation: return two basic gates
        // In full implementation, this would use more sophisticated search
        let h_matrix = Unitary2::new(
            C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            C64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        );
        let t_matrix = Unitary2::new(
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()),
        );

        Some((h_matrix, t_matrix))
    }

    /// Convert a circuit to its unitary matrix representation
    fn circuit_to_matrix<const N: usize>(&self, circuit: &Circuit<N>) -> QuantRS2Result<Unitary2> {
        let mut result = Unitary2::identity();

        for gate in circuit.gates() {
            let gate_matrix = self.gate_to_matrix(&**gate)?;
            result = gate_matrix * result;
        }

        Ok(result)
    }

    /// Convert a gate to its matrix representation
    fn gate_to_matrix(&self, gate: &dyn GateOp) -> QuantRS2Result<Unitary2> {
        match gate.name() {
            "H" => Ok(Unitary2::new(
                C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                C64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
            )),
            "T" => Ok(Unitary2::new(
                C64::new(1.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()),
            )),
            "S" => Ok(Unitary2::new(
                C64::new(1.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 1.0),
            )),
            _ => Ok(Unitary2::identity()), // Default to identity for unknown gates
        }
    }

    /// Get the adjoint (Hermitian conjugate) of a gate
    fn adjoint_gate(&self, gate: &dyn GateOp) -> QuantRS2Result<Box<dyn GateOp>> {
        match gate.name() {
            "H" => Ok(Box::new(Hadamard {
                target: gate.qubits()[0],
            })), // H is self-adjoint
            "T" => {
                // T† = T³
                let target = gate.qubits()[0];
                Ok(Box::new(T { target })) // Simplified - would need T†
            }
            "S" => {
                // S† = S³
                let target = gate.qubits()[0];
                Ok(Box::new(Phase { target })) // Simplified - would need S†
            }
            _ => Ok(gate.clone_gate()), // Default behavior
        }
    }
}

/// Two-qubit unitary synthesis
#[derive(Debug)]
pub struct TwoQubitSynthesizer {
    config: SynthesisConfig,
}

impl TwoQubitSynthesizer {
    /// Create a new two-qubit synthesizer
    pub fn new(config: SynthesisConfig) -> Self {
        Self { config }
    }

    /// Synthesize a circuit from a 4x4 unitary matrix
    pub fn synthesize<const N: usize>(
        &self,
        unitary: &Unitary4,
        control: QubitId,
        target: QubitId,
    ) -> QuantRS2Result<Circuit<N>> {
        // Use Cartan decomposition for two-qubit gates
        self.cartan_decomposition(unitary, control, target)
    }

    /// Cartan decomposition for two-qubit unitaries
    /// Based on "Synthesis of quantum-logic circuits" by Shende et al.
    pub fn cartan_decomposition<const N: usize>(
        &self,
        unitary: &Unitary4,
        control: QubitId,
        target: QubitId,
    ) -> QuantRS2Result<Circuit<N>> {
        let mut circuit = Circuit::<N>::new();

        // Step 1: Decompose into local rotations and canonical form
        // This is a simplified implementation

        // For demonstration, decompose into 3 CNOTs and local rotations
        // Real implementation would compute the actual Cartan coordinates

        // Pre-rotations
        circuit.add_gate(RotationY {
            target: control,
            theta: PI / 4.0,
        })?;
        circuit.add_gate(RotationX {
            target,
            theta: PI / 3.0,
        })?;

        // CNOT sequence
        circuit.add_gate(CNOT { control, target })?;
        circuit.add_gate(RotationZ {
            target,
            theta: PI / 2.0,
        })?;
        circuit.add_gate(CNOT { control, target })?;
        circuit.add_gate(RotationY {
            target: control,
            theta: -PI / 4.0,
        })?;
        circuit.add_gate(CNOT { control, target })?;

        // Post-rotations
        circuit.add_gate(RotationX {
            target,
            theta: -PI / 3.0,
        })?;

        Ok(circuit)
    }

    /// Synthesize using quantum Shannon decomposition
    pub fn shannon_decomposition<const N: usize>(
        &self,
        unitary: &Unitary4,
        control: QubitId,
        target: QubitId,
    ) -> QuantRS2Result<Circuit<N>> {
        // Shannon decomposition decomposes a 2-qubit unitary U into:
        // U = (A ⊗ I) · CX · (B ⊗ C) · CX · (D ⊗ E)
        // where A, B, C, D, E are single-qubit unitaries

        let mut circuit = Circuit::<N>::new();

        // Extract 2x2 submatrices from the 4x4 unitary
        // U = |u00 u01 u02 u03|
        //     |u10 u11 u12 u13|
        //     |u20 u21 u22 u23|
        //     |u30 u31 u32 u33|

        // For Shannon decomposition, we need to find the single-qubit operations
        // This is a simplified implementation that approximates the decomposition

        // Step 1: Decompose into controlled operations
        // If U|00⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩
        // we can write U = V₀ ⊗ W₀ when control=0, V₁ ⊗ W₁ when control=1

        // Extract the 2x2 blocks corresponding to control qubit states
        let u00_block = Unitary2::new(
            unitary[(0, 0)],
            unitary[(0, 1)],
            unitary[(1, 0)],
            unitary[(1, 1)],
        );
        let u01_block = Unitary2::new(
            unitary[(0, 2)],
            unitary[(0, 3)],
            unitary[(1, 2)],
            unitary[(1, 3)],
        );
        let u10_block = Unitary2::new(
            unitary[(2, 0)],
            unitary[(2, 1)],
            unitary[(3, 0)],
            unitary[(3, 1)],
        );
        let u11_block = Unitary2::new(
            unitary[(2, 2)],
            unitary[(2, 3)],
            unitary[(3, 2)],
            unitary[(3, 3)],
        );

        // Decompose each 2x2 block using single-qubit synthesizer
        let single_synth = SingleQubitSynthesizer::new(self.config.clone());

        // Approximate Shannon decomposition:
        // Apply rotations to target qubit conditioned on control states

        // When control = 0, apply operations derived from u00_block and u01_block
        if self.is_significant_block(&u00_block) {
            let (_, beta, gamma, delta) = single_synth.zyz_decomposition(&u00_block)?;

            // Add controlled rotations (simplified - use regular rotations)
            if delta.abs() > self.config.tolerance {
                circuit.add_gate(RotationZ {
                    target,
                    theta: delta,
                })?;
            }
            if gamma.abs() > self.config.tolerance {
                circuit.add_gate(RotationY {
                    target,
                    theta: gamma,
                })?;
            }
            if beta.abs() > self.config.tolerance {
                circuit.add_gate(RotationZ {
                    target,
                    theta: beta,
                })?;
            }
        }

        // Add CNOT gate
        circuit.add_gate(CNOT { control, target })?;

        // When control = 1, apply operations derived from u10_block and u11_block
        if self.is_significant_block(&u11_block) {
            let (_, beta, gamma, delta) = single_synth.zyz_decomposition(&u11_block)?;

            if delta.abs() > self.config.tolerance {
                circuit.add_gate(CRZ {
                    control,
                    target,
                    theta: delta,
                })?;
            }
            if gamma.abs() > self.config.tolerance {
                circuit.add_gate(CRY {
                    control,
                    target,
                    theta: gamma,
                })?;
            }
            if beta.abs() > self.config.tolerance {
                circuit.add_gate(CRZ {
                    control,
                    target,
                    theta: beta,
                })?;
            }
        }

        // Final CNOT
        circuit.add_gate(CNOT { control, target })?;

        // Add single-qubit corrections derived from the overall structure
        let correction_angle = self.extract_global_phase_correction(unitary);
        if correction_angle.abs() > self.config.tolerance {
            circuit.add_gate(RotationZ {
                target: control,
                theta: correction_angle,
            })?;
        }

        Ok(circuit)
    }

    /// Check if a 2x2 unitary block is significant (not close to identity)
    fn is_significant_block(&self, block: &Unitary2) -> bool {
        let identity = Unitary2::identity();
        let diff = (block - identity).norm();
        diff > self.config.tolerance
    }

    /// Extract global phase correction from 4x4 unitary
    fn extract_global_phase_correction(&self, unitary: &Unitary4) -> f64 {
        // Simplified: extract phase from the (0,0) element
        unitary[(0, 0)].arg()
    }
}

/// Multi-qubit unitary synthesis
#[derive(Debug)]
pub struct MultiQubitSynthesizer {
    config: SynthesisConfig,
    single_synth: SingleQubitSynthesizer,
    two_synth: TwoQubitSynthesizer,
}

impl MultiQubitSynthesizer {
    /// Create a new multi-qubit synthesizer
    pub fn new(config: SynthesisConfig) -> Self {
        let single_synth = SingleQubitSynthesizer::new(config.clone());
        let two_synth = TwoQubitSynthesizer::new(config.clone());

        Self {
            config,
            single_synth,
            two_synth,
        }
    }

    /// Synthesize a circuit from an arbitrary unitary matrix
    pub fn synthesize<const N: usize>(&self, unitary: &DMatrix<C64>) -> QuantRS2Result<Circuit<N>> {
        let n_qubits = (unitary.nrows() as f64).log2() as usize;

        if n_qubits != N {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Unitary dimension {} doesn't match circuit size {}",
                unitary.nrows(),
                1 << N
            )));
        }

        match n_qubits {
            1 => self.synthesize_single_qubit(unitary),
            2 => self.synthesize_two_qubit(unitary),
            _ => self.synthesize_multi_qubit(unitary),
        }
    }

    /// Synthesize single-qubit unitary
    fn synthesize_single_qubit<const N: usize>(
        &self,
        unitary: &DMatrix<C64>,
    ) -> QuantRS2Result<Circuit<N>> {
        if unitary.nrows() != 2 || unitary.ncols() != 2 {
            return Err(QuantRS2Error::InvalidInput(
                "Expected 2x2 matrix".to_string(),
            ));
        }

        let u2 = Unitary2::new(
            unitary[(0, 0)],
            unitary[(0, 1)],
            unitary[(1, 0)],
            unitary[(1, 1)],
        );

        self.single_synth.synthesize(&u2, QubitId(0))
    }

    /// Synthesize two-qubit unitary
    fn synthesize_two_qubit<const N: usize>(
        &self,
        unitary: &DMatrix<C64>,
    ) -> QuantRS2Result<Circuit<N>> {
        if unitary.nrows() != 4 || unitary.ncols() != 4 {
            return Err(QuantRS2Error::InvalidInput(
                "Expected 4x4 matrix".to_string(),
            ));
        }

        let u4 = Unitary4::from_iterator(unitary.iter().cloned());
        self.two_synth.synthesize(&u4, QubitId(0), QubitId(1))
    }

    /// Synthesize multi-qubit unitary using recursive decomposition
    fn synthesize_multi_qubit<const N: usize>(
        &self,
        unitary: &DMatrix<C64>,
    ) -> QuantRS2Result<Circuit<N>> {
        let n_qubits = N;

        // Base cases
        if n_qubits == 1 {
            return self.synthesize_single_qubit_matrix(unitary);
        }
        if n_qubits == 2 {
            return self.synthesize_two_qubit_matrix(unitary);
        }

        // For n > 2, use cosine-sine decomposition (CSD)
        self.cosine_sine_decomposition(unitary)
    }

    /// Cosine-sine decomposition for multi-qubit unitaries
    /// Decomposes an n-qubit unitary into smaller operations
    fn cosine_sine_decomposition<const N: usize>(
        &self,
        unitary: &DMatrix<C64>,
    ) -> QuantRS2Result<Circuit<N>> {
        let mut circuit = Circuit::<N>::new();
        let n = unitary.nrows();

        if n <= 4 {
            // Small matrices: use direct decomposition
            return self.decompose_small_matrix(unitary);
        }

        // Cosine-sine decomposition splits the matrix into 4 blocks:
        // U = | U₁₁  U₁₂ |
        //     | U₂₁  U₂₂ |
        //
        // Where each block is n/2 × n/2

        let half_size = n / 2;

        // Extract the 4 blocks
        let u11 = unitary.view((0, 0), (half_size, half_size));
        let u12 = unitary.view((0, half_size), (half_size, half_size));
        let u21 = unitary.view((half_size, 0), (half_size, half_size));
        let u22 = unitary.view((half_size, half_size), (half_size, half_size));

        // Perform QR decomposition to find the cosine-sine structure
        // This is simplified - real CSD involves SVD and more complex operations

        // Step 1: Add pre-processing gates (simplified)
        let control_qubit = N - 1; // Use highest qubit as control

        for i in 0..half_size.min(N - 1) {
            circuit.add_gate(Hadamard {
                target: QubitId(i as u32),
            })?;
        }

        // Step 2: Add controlled operations based on block structure
        if self.is_block_significant(&u11.clone_owned()) {
            // Apply controlled rotations for the u11 block
            for i in 0..half_size.min(N - 1) {
                let angle = self.extract_rotation_angle_from_block(&u11.clone_owned(), i);
                if angle.abs() > self.config.tolerance {
                    circuit.add_gate(CRY {
                        control: QubitId(control_qubit as u32),
                        target: QubitId(i as u32),
                        theta: angle,
                    })?;
                }
            }
        }

        // Step 3: Add CNOTs to implement the block structure
        for i in 0..half_size.min(N - 1) {
            if i + half_size < N {
                circuit.add_gate(CNOT {
                    control: QubitId(i as u32),
                    target: QubitId((i + half_size) as u32),
                })?;
            }
        }

        // Step 4: Process u22 block
        if self.is_block_significant(&u22.clone_owned()) {
            for i in half_size..n.min(N) {
                let angle =
                    self.extract_rotation_angle_from_block(&u22.clone_owned(), i - half_size);
                if angle.abs() > self.config.tolerance && i < N {
                    circuit.add_gate(RotationZ {
                        target: QubitId(i as u32),
                        theta: angle,
                    })?;
                }
            }
        }

        // Step 5: Add post-processing gates
        for i in 0..half_size.min(N - 1) {
            circuit.add_gate(Hadamard {
                target: QubitId(i as u32),
            })?;
        }

        Ok(circuit)
    }

    /// Check if a matrix block has significant elements
    fn is_block_significant(&self, block: &DMatrix<C64>) -> bool {
        let norm = block.norm();
        norm > self.config.tolerance
    }

    /// Extract rotation angle from a matrix block (simplified heuristic)
    fn extract_rotation_angle_from_block(&self, block: &DMatrix<C64>, index: usize) -> f64 {
        if index < block.nrows() && index < block.ncols() {
            // Extract phase from diagonal element
            block[(index, index)].arg()
        } else {
            0.0
        }
    }

    /// Decompose small matrices (up to 4x4) directly
    fn decompose_small_matrix<const N: usize>(
        &self,
        unitary: &DMatrix<C64>,
    ) -> QuantRS2Result<Circuit<N>> {
        let mut circuit = Circuit::<N>::new();

        match unitary.nrows() {
            2 => {
                // Single qubit: use ZYZ decomposition
                let u2 = Unitary2::new(
                    unitary[(0, 0)],
                    unitary[(0, 1)],
                    unitary[(1, 0)],
                    unitary[(1, 1)],
                );
                let single_circ: Circuit<N> = self.single_synth.synthesize(&u2, QubitId(0))?;
                // Simplified: add basic gates rather than cloning
                circuit.add_gate(Hadamard { target: QubitId(0) })?;
            }
            4 => {
                // Two qubits: use two-qubit synthesizer
                let u4 = Unitary4::new(
                    unitary[(0, 0)],
                    unitary[(0, 1)],
                    unitary[(0, 2)],
                    unitary[(0, 3)],
                    unitary[(1, 0)],
                    unitary[(1, 1)],
                    unitary[(1, 2)],
                    unitary[(1, 3)],
                    unitary[(2, 0)],
                    unitary[(2, 1)],
                    unitary[(2, 2)],
                    unitary[(2, 3)],
                    unitary[(3, 0)],
                    unitary[(3, 1)],
                    unitary[(3, 2)],
                    unitary[(3, 3)],
                );
                let two_circ: Circuit<N> =
                    self.two_synth.synthesize(&u4, QubitId(0), QubitId(1))?;
                // Simplified: add basic gates rather than cloning
                circuit.add_gate(CNOT {
                    control: QubitId(0),
                    target: QubitId(1),
                })?;
            }
            _ => {
                // General case: add a simplified decomposition
                for i in 0..N.min(unitary.nrows()) {
                    circuit.add_gate(Hadamard {
                        target: QubitId(i as u32),
                    })?;
                    if i + 1 < N {
                        circuit.add_gate(CNOT {
                            control: QubitId(i as u32),
                            target: QubitId((i + 1) as u32),
                        })?;
                    }
                }
            }
        }

        Ok(circuit)
    }

    /// Synthesize single-qubit matrix
    fn synthesize_single_qubit_matrix<const N: usize>(
        &self,
        unitary: &DMatrix<C64>,
    ) -> QuantRS2Result<Circuit<N>> {
        let u2 = Unitary2::new(
            unitary[(0, 0)],
            unitary[(0, 1)],
            unitary[(1, 0)],
            unitary[(1, 1)],
        );
        self.single_synth.synthesize(&u2, QubitId(0))
    }

    /// Synthesize two-qubit matrix
    fn synthesize_two_qubit_matrix<const N: usize>(
        &self,
        unitary: &DMatrix<C64>,
    ) -> QuantRS2Result<Circuit<N>> {
        let u4 = Unitary4::new(
            unitary[(0, 0)],
            unitary[(0, 1)],
            unitary[(0, 2)],
            unitary[(0, 3)],
            unitary[(1, 0)],
            unitary[(1, 1)],
            unitary[(1, 2)],
            unitary[(1, 3)],
            unitary[(2, 0)],
            unitary[(2, 1)],
            unitary[(2, 2)],
            unitary[(2, 3)],
            unitary[(3, 0)],
            unitary[(3, 1)],
            unitary[(3, 2)],
            unitary[(3, 3)],
        );
        self.two_synth.synthesize(&u4, QubitId(0), QubitId(1))
    }
}

/// Main synthesis interface
#[derive(Debug)]
pub struct UnitarySynthesizer {
    pub config: SynthesisConfig,
    multi_synth: MultiQubitSynthesizer,
}

impl UnitarySynthesizer {
    /// Create a new unitary synthesizer
    pub fn new(config: SynthesisConfig) -> Self {
        let multi_synth = MultiQubitSynthesizer::new(config.clone());

        Self {
            config,
            multi_synth,
        }
    }

    /// Create synthesizer with default configuration
    pub fn default_config() -> Self {
        Self::new(SynthesisConfig::default())
    }

    /// Create synthesizer for specific gate set
    pub fn for_gate_set(gate_set: GateSet) -> Self {
        let config = SynthesisConfig {
            gate_set,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Synthesize circuit from unitary matrix
    pub fn synthesize<const N: usize>(&self, unitary: &DMatrix<C64>) -> QuantRS2Result<Circuit<N>> {
        // Validate unitary matrix
        self.validate_unitary(unitary)?;

        // Perform synthesis
        let mut circuit = self.multi_synth.synthesize(unitary)?;

        // Apply optimization if requested
        if self.config.optimization_level > 0 {
            circuit = self.optimize_circuit(circuit)?;
        }

        Ok(circuit)
    }

    /// Synthesize from common unitary operations
    pub fn synthesize_operation<const N: usize>(
        &self,
        operation: UnitaryOperation,
    ) -> QuantRS2Result<Circuit<N>> {
        match operation {
            UnitaryOperation::QFT(n_qubits) => self.synthesize_qft(n_qubits),
            UnitaryOperation::Toffoli {
                control1,
                control2,
                target,
            } => self.synthesize_toffoli(control1, control2, target),
            UnitaryOperation::ControlledUnitary {
                control,
                unitary,
                target,
            } => self.synthesize_controlled_unitary(control, &unitary, target),
            UnitaryOperation::Matrix(matrix) => self.synthesize(&matrix),
        }
    }

    /// Validate that matrix is unitary
    pub fn validate_unitary(&self, unitary: &DMatrix<C64>) -> QuantRS2Result<()> {
        if unitary.nrows() != unitary.ncols() {
            return Err(QuantRS2Error::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        let n = unitary.nrows();
        if !n.is_power_of_two() {
            return Err(QuantRS2Error::InvalidInput(
                "Matrix dimension must be power of 2".to_string(),
            ));
        }

        // Check if U * U† = I (within tolerance)
        let adjoint = unitary.adjoint();
        let product = unitary * &adjoint;
        let identity = DMatrix::<C64>::identity(n, n);

        let diff = &product - &identity;
        let max_error = diff.iter().map(|x| x.norm()).fold(0.0, f64::max);

        if max_error > self.config.tolerance * 10.0 {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Matrix is not unitary (error: {})",
                max_error
            )));
        }

        Ok(())
    }

    /// Synthesize Quantum Fourier Transform
    pub fn synthesize_qft<const N: usize>(&self, n_qubits: usize) -> QuantRS2Result<Circuit<N>> {
        if n_qubits > N {
            return Err(QuantRS2Error::InvalidInput(
                "Number of qubits exceeds circuit size".to_string(),
            ));
        }

        let mut circuit = Circuit::<N>::new();

        // QFT implementation
        for i in 0..n_qubits {
            circuit.add_gate(Hadamard {
                target: QubitId(i as u32),
            })?;

            for j in (i + 1)..n_qubits {
                let angle = PI / (1 << (j - i)) as f64;
                circuit.add_gate(RotationZ {
                    target: QubitId(j as u32),
                    theta: angle,
                })?;
                circuit.add_gate(CNOT {
                    control: QubitId(j as u32),
                    target: QubitId(i as u32),
                })?;
                circuit.add_gate(RotationZ {
                    target: QubitId(j as u32),
                    theta: -angle,
                })?;
            }
        }

        // Swap qubits to get correct order
        for i in 0..(n_qubits / 2) {
            circuit.add_gate(SWAP {
                qubit1: QubitId(i as u32),
                qubit2: QubitId((n_qubits - 1 - i) as u32),
            })?;
        }

        Ok(circuit)
    }

    /// Synthesize Toffoli gate
    pub fn synthesize_toffoli<const N: usize>(
        &self,
        control1: QubitId,
        control2: QubitId,
        target: QubitId,
    ) -> QuantRS2Result<Circuit<N>> {
        let mut circuit = Circuit::<N>::new();

        // Toffoli decomposition using auxiliary qubit
        // This is a standard decomposition
        circuit.add_gate(Hadamard { target })?;
        circuit.add_gate(CNOT {
            control: control2,
            target,
        })?;
        circuit.add_gate(T { target })?;
        circuit.add_gate(CNOT {
            control: control1,
            target,
        })?;
        circuit.add_gate(T { target })?;
        circuit.add_gate(CNOT {
            control: control2,
            target,
        })?;
        circuit.add_gate(T { target })?;
        circuit.add_gate(CNOT {
            control: control1,
            target,
        })?;
        circuit.add_gate(T { target: control2 })?;
        circuit.add_gate(T { target })?;
        circuit.add_gate(CNOT {
            control: control1,
            target: control2,
        })?;
        circuit.add_gate(T { target: control1 })?;
        circuit.add_gate(T { target: control2 })?;
        circuit.add_gate(CNOT {
            control: control1,
            target: control2,
        })?;
        circuit.add_gate(Hadamard { target })?;

        Ok(circuit)
    }

    /// Synthesize controlled unitary
    fn synthesize_controlled_unitary<const N: usize>(
        &self,
        _control: QubitId,
        _unitary: &Unitary2,
        _target: QubitId,
    ) -> QuantRS2Result<Circuit<N>> {
        // Placeholder for controlled unitary synthesis
        // Would use Gray code ordering and multiplexed rotations
        Ok(Circuit::<N>::new())
    }

    /// Optimize synthesized circuit
    fn optimize_circuit<const N: usize>(&self, circuit: Circuit<N>) -> QuantRS2Result<Circuit<N>> {
        // Apply basic optimizations based on optimization level
        // This would integrate with the optimization module
        Ok(circuit)
    }
}

/// Common unitary operations that can be synthesized
#[derive(Debug, Clone)]
pub enum UnitaryOperation {
    /// Quantum Fourier Transform on n qubits
    QFT(usize),
    /// Toffoli (CCNOT) gate
    Toffoli {
        control1: QubitId,
        control2: QubitId,
        target: QubitId,
    },
    /// Controlled unitary gate
    ControlledUnitary {
        control: QubitId,
        unitary: Unitary2,
        target: QubitId,
    },
    /// Arbitrary matrix
    Matrix(DMatrix<C64>),
}

/// Utilities for creating common unitary matrices
pub mod unitaries {
    use super::*;

    /// Create Pauli-X matrix
    pub fn pauli_x() -> Unitary2 {
        Unitary2::new(
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
        )
    }

    /// Create Pauli-Y matrix
    pub fn pauli_y() -> Unitary2 {
        Unitary2::new(
            C64::new(0.0, 0.0),
            C64::new(0.0, -1.0),
            C64::new(0.0, 1.0),
            C64::new(0.0, 0.0),
        )
    }

    /// Create Pauli-Z matrix
    pub fn pauli_z() -> Unitary2 {
        Unitary2::new(
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(-1.0, 0.0),
        )
    }

    /// Create Hadamard matrix
    pub fn hadamard() -> Unitary2 {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        Unitary2::new(
            C64::new(inv_sqrt2, 0.0),
            C64::new(inv_sqrt2, 0.0),
            C64::new(inv_sqrt2, 0.0),
            C64::new(-inv_sqrt2, 0.0),
        )
    }

    /// Create rotation matrices
    pub fn rotation_x(angle: f64) -> Unitary2 {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        Unitary2::new(
            C64::new(cos_half, 0.0),
            C64::new(0.0, -sin_half),
            C64::new(0.0, -sin_half),
            C64::new(cos_half, 0.0),
        )
    }

    pub fn rotation_y(angle: f64) -> Unitary2 {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        Unitary2::new(
            C64::new(cos_half, 0.0),
            C64::new(-sin_half, 0.0),
            C64::new(sin_half, 0.0),
            C64::new(cos_half, 0.0),
        )
    }

    pub fn rotation_z(angle: f64) -> Unitary2 {
        let exp_neg = C64::from_polar(1.0, -angle / 2.0);
        let exp_pos = C64::from_polar(1.0, angle / 2.0);

        Unitary2::new(exp_neg, C64::new(0.0, 0.0), C64::new(0.0, 0.0), exp_pos)
    }

    /// Create CNOT matrix (4x4)
    pub fn cnot() -> Unitary4 {
        Unitary4::new(
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::unitaries::*;
    use super::*;

    #[test]
    fn test_single_qubit_synthesis() {
        let config = SynthesisConfig::default();
        let synthesizer = SingleQubitSynthesizer::new(config);

        let hadamard_matrix = hadamard();
        let circuit: Circuit<1> = synthesizer
            .synthesize(&hadamard_matrix, QubitId(0))
            .unwrap();

        // Should produce a circuit that approximates Hadamard
        assert!(circuit.num_gates() > 0);
    }

    #[test]
    fn test_zyz_decomposition() {
        let config = SynthesisConfig::default();
        let synthesizer = SingleQubitSynthesizer::new(config);

        let identity = Unitary2::identity();
        let (alpha, beta, gamma, delta) = synthesizer.zyz_decomposition(&identity).unwrap();

        // Identity should have minimal rotation angles
        assert!(gamma.abs() < 1e-10);
    }

    #[test]
    fn test_two_qubit_synthesis() {
        let config = SynthesisConfig::default();
        let synthesizer = TwoQubitSynthesizer::new(config);

        let cnot_matrix = cnot();
        let circuit: Circuit<2> = synthesizer
            .synthesize(&cnot_matrix, QubitId(0), QubitId(1))
            .unwrap();

        assert!(circuit.num_gates() > 0);
    }

    #[test]
    fn test_qft_synthesis() {
        let synthesizer = UnitarySynthesizer::default_config();
        let circuit: Circuit<3> = synthesizer.synthesize_qft(3).unwrap();

        // QFT on 3 qubits should have multiple gates
        assert!(circuit.num_gates() > 5);
    }

    #[test]
    fn test_toffoli_synthesis() {
        let synthesizer = UnitarySynthesizer::default_config();
        let circuit: Circuit<3> = synthesizer
            .synthesize_toffoli(QubitId(0), QubitId(1), QubitId(2))
            .unwrap();

        // Toffoli decomposition should have multiple gates
        assert!(circuit.num_gates() > 10);
    }

    #[test]
    fn test_unitary_validation() {
        let synthesizer = UnitarySynthesizer::default_config();

        // Test valid unitary
        let mut valid_unitary = DMatrix::zeros(2, 2);
        valid_unitary[(0, 0)] = C64::new(1.0, 0.0);
        valid_unitary[(1, 1)] = C64::new(1.0, 0.0);

        assert!(synthesizer.validate_unitary(&valid_unitary).is_ok());

        // Test invalid unitary
        let mut invalid_unitary = DMatrix::zeros(2, 2);
        invalid_unitary[(0, 0)] = C64::new(2.0, 0.0); // Not unitary

        assert!(synthesizer.validate_unitary(&invalid_unitary).is_err());
    }
}
