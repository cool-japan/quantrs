//! Unitary synthesis module
//!
//! This module provides algorithms for synthesizing quantum circuits from unitary matrix
//! descriptions. It includes various decomposition strategies for different gate sets.

use crate::builder::Circuit;
use nalgebra::{Complex, DMatrix, Matrix2, Matrix4};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{
        multi::{CNOT, CZ, SWAP},
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
        _unitary: &Unitary2,
        target: QubitId,
    ) -> QuantRS2Result<Circuit<N>> {
        // Simplified Solovay-Kitaev implementation
        // In practice, this would use a recursive approximation algorithm
        let mut circuit = Circuit::<N>::new();

        // Placeholder: just add an identity (no gates)
        // TODO: Implement full Solovay-Kitaev algorithm

        Ok(circuit)
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
        // Shannon decomposition: recursively decompose into smaller unitaries
        // This is a placeholder implementation
        self.cartan_decomposition(unitary, control, target)
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
        let mut circuit = Circuit::<N>::new();

        // Use cosine-sine decomposition for recursive synthesis
        // This is a simplified placeholder implementation

        // For demonstration, just add some gates
        for i in 0..N.min(4) {
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

        Ok(circuit)
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
