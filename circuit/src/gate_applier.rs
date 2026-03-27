//! Single-qubit, two-qubit, and three-qubit gate application methods for [`Circuit`].
//!
//! This module provides all the gate-application convenience methods on `Circuit<N>`,
//! including individual gates (h, x, y, z, s, t, rx, ry, rz, cnot, …) as well as
//! bulk helpers (`h_all`, `x_all`, `cnot_all`, …) and structural helpers
//! (`measure`, `reset`, `barrier`, `measure_all`).

use std::sync::Arc;

use quantrs2_core::{
    error::QuantRS2Result,
    gate::{
        multi::{
            Fredkin, ISwap, Toffoli, CH, CNOT, CRX, CRY, CRZ, CS, CY, CZ, DCX, ECR, RXX, RYY, RZX,
            RZZ, SWAP,
        },
        single::{
            Hadamard, Identity, PGate, PauliX, PauliY, PauliZ, Phase, PhaseDagger, RotationX,
            RotationY, RotationZ, SqrtX, SqrtXDagger, TDagger, UGate, T,
        },
        GateOp,
    },
    qubit::QubitId,
};

use crate::builder::{BarrierInfo, Circuit, Measure};

impl<const N: usize> Circuit<N> {
    // ============ Single-Qubit Gates ============

    /// Apply a Hadamard gate to a qubit
    pub fn h(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(Hadamard {
            target: target.into(),
        })
    }

    /// Apply a Pauli-X gate to a qubit
    pub fn x(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(PauliX {
            target: target.into(),
        })
    }

    /// Apply a Pauli-Y gate to a qubit
    pub fn y(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(PauliY {
            target: target.into(),
        })
    }

    /// Apply a Pauli-Z gate to a qubit
    pub fn z(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(PauliZ {
            target: target.into(),
        })
    }

    /// Apply a rotation around X-axis
    pub fn rx(&mut self, target: impl Into<QubitId>, theta: f64) -> QuantRS2Result<&mut Self> {
        self.add_gate(RotationX {
            target: target.into(),
            theta,
        })
    }

    /// Apply a rotation around Y-axis
    pub fn ry(&mut self, target: impl Into<QubitId>, theta: f64) -> QuantRS2Result<&mut Self> {
        self.add_gate(RotationY {
            target: target.into(),
            theta,
        })
    }

    /// Apply a rotation around Z-axis
    pub fn rz(&mut self, target: impl Into<QubitId>, theta: f64) -> QuantRS2Result<&mut Self> {
        self.add_gate(RotationZ {
            target: target.into(),
            theta,
        })
    }

    /// Apply a Phase gate (S gate)
    pub fn s(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(Phase {
            target: target.into(),
        })
    }

    /// Apply a Phase-dagger gate (S† gate)
    pub fn sdg(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(PhaseDagger {
            target: target.into(),
        })
    }

    /// Apply a T gate
    pub fn t(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(T {
            target: target.into(),
        })
    }

    /// Apply a T-dagger gate (T† gate)
    pub fn tdg(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(TDagger {
            target: target.into(),
        })
    }

    /// Apply a Square Root of X gate (√X)
    pub fn sx(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(SqrtX {
            target: target.into(),
        })
    }

    /// Apply a Square Root of X Dagger gate (√X†)
    pub fn sxdg(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(SqrtXDagger {
            target: target.into(),
        })
    }

    // ============ Qiskit-Compatible Single-Qubit Gates ============

    /// Apply a U gate (general single-qubit rotation)
    ///
    /// U(θ, φ, λ) = [[cos(θ/2), -e^(iλ)·sin(θ/2)],
    ///              [e^(iφ)·sin(θ/2), e^(i(φ+λ))·cos(θ/2)]]
    pub fn u(
        &mut self,
        target: impl Into<QubitId>,
        theta: f64,
        phi: f64,
        lambda: f64,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(UGate {
            target: target.into(),
            theta,
            phi,
            lambda,
        })
    }

    /// Apply a P gate (phase gate with parameter)
    ///
    /// P(λ) = [[1, 0], [0, e^(iλ)]]
    pub fn p(&mut self, target: impl Into<QubitId>, lambda: f64) -> QuantRS2Result<&mut Self> {
        self.add_gate(PGate {
            target: target.into(),
            lambda,
        })
    }

    /// Apply an Identity gate
    pub fn id(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(Identity {
            target: target.into(),
        })
    }

    // ============ Two-Qubit Gates ============

    /// Apply a CNOT gate
    pub fn cnot(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CNOT {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a CNOT gate (alias for cnot)
    pub fn cx(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.cnot(control, target)
    }

    /// Apply a CY gate (Controlled-Y)
    pub fn cy(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CY {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a CZ gate (Controlled-Z)
    pub fn cz(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CZ {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a CH gate (Controlled-Hadamard)
    pub fn ch(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CH {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a CS gate (Controlled-Phase/S)
    pub fn cs(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CS {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a controlled rotation around X-axis (CRX)
    pub fn crx(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
        theta: f64,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CRX {
            control: control.into(),
            target: target.into(),
            theta,
        })
    }

    /// Apply a controlled rotation around Y-axis (CRY)
    pub fn cry(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
        theta: f64,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CRY {
            control: control.into(),
            target: target.into(),
            theta,
        })
    }

    /// Apply a controlled rotation around Z-axis (CRZ)
    pub fn crz(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
        theta: f64,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CRZ {
            control: control.into(),
            target: target.into(),
            theta,
        })
    }

    /// Apply a controlled phase gate
    pub fn cp(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
        lambda: f64,
    ) -> QuantRS2Result<&mut Self> {
        // CRZ(lambda) is equivalent to CP(lambda) up to a global phase
        self.crz(control, target, lambda)
    }

    /// Apply a SWAP gate
    pub fn swap(
        &mut self,
        qubit1: impl Into<QubitId>,
        qubit2: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(SWAP {
            qubit1: qubit1.into(),
            qubit2: qubit2.into(),
        })
    }

    /// Apply an iSWAP gate
    ///
    /// iSWAP swaps two qubits and phases |01⟩ and |10⟩ by i
    pub fn iswap(
        &mut self,
        qubit1: impl Into<QubitId>,
        qubit2: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(ISwap {
            qubit1: qubit1.into(),
            qubit2: qubit2.into(),
        })
    }

    /// Apply an ECR gate (IBM native echoed cross-resonance gate)
    pub fn ecr(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(ECR {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply an RXX gate (two-qubit XX rotation)
    ///
    /// RXX(θ) = exp(-i * θ/2 * X⊗X)
    pub fn rxx(
        &mut self,
        qubit1: impl Into<QubitId>,
        qubit2: impl Into<QubitId>,
        theta: f64,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(RXX {
            qubit1: qubit1.into(),
            qubit2: qubit2.into(),
            theta,
        })
    }

    /// Apply an RYY gate (two-qubit YY rotation)
    ///
    /// RYY(θ) = exp(-i * θ/2 * Y⊗Y)
    pub fn ryy(
        &mut self,
        qubit1: impl Into<QubitId>,
        qubit2: impl Into<QubitId>,
        theta: f64,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(RYY {
            qubit1: qubit1.into(),
            qubit2: qubit2.into(),
            theta,
        })
    }

    /// Apply an RZZ gate (two-qubit ZZ rotation)
    ///
    /// RZZ(θ) = exp(-i * θ/2 * Z⊗Z)
    pub fn rzz(
        &mut self,
        qubit1: impl Into<QubitId>,
        qubit2: impl Into<QubitId>,
        theta: f64,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(RZZ {
            qubit1: qubit1.into(),
            qubit2: qubit2.into(),
            theta,
        })
    }

    /// Apply an RZX gate (two-qubit ZX rotation / cross-resonance)
    ///
    /// RZX(θ) = exp(-i * θ/2 * Z⊗X)
    pub fn rzx(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
        theta: f64,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(RZX {
            control: control.into(),
            target: target.into(),
            theta,
        })
    }

    /// Apply a DCX gate (double CNOT gate)
    ///
    /// DCX = CNOT(0,1) @ CNOT(1,0)
    pub fn dcx(
        &mut self,
        qubit1: impl Into<QubitId>,
        qubit2: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(DCX {
            qubit1: qubit1.into(),
            qubit2: qubit2.into(),
        })
    }

    // ============ Three-Qubit Gates ============

    /// Apply a Toffoli (CCNOT) gate
    pub fn toffoli(
        &mut self,
        control1: impl Into<QubitId>,
        control2: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(Toffoli {
            control1: control1.into(),
            control2: control2.into(),
            target: target.into(),
        })
    }

    /// Apply a Fredkin (CSWAP) gate
    pub fn cswap(
        &mut self,
        control: impl Into<QubitId>,
        target1: impl Into<QubitId>,
        target2: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(Fredkin {
            control: control.into(),
            target1: target1.into(),
            target2: target2.into(),
        })
    }

    /// Apply a CCX gate (alias for Toffoli)
    pub fn ccx(
        &mut self,
        control1: impl Into<QubitId>,
        control2: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.toffoli(control1, control2, target)
    }

    /// Apply a Fredkin gate (alias for cswap)
    pub fn fredkin(
        &mut self,
        control: impl Into<QubitId>,
        target1: impl Into<QubitId>,
        target2: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.cswap(control, target1, target2)
    }

    // ============ Measurement and Control ============

    /// Measure a qubit (currently adds a placeholder measure gate)
    ///
    /// Note: This is currently a placeholder implementation for QASM export compatibility.
    /// For actual quantum measurements, use the measurement module functionality.
    pub fn measure(&mut self, qubit: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        let qubit_id = qubit.into();
        self.add_gate(Measure { target: qubit_id })?;
        Ok(self)
    }

    /// Reset a qubit to |0⟩ state
    ///
    /// Note: This operation is not yet fully implemented.
    /// Reset operations are complex and require special handling in quantum circuits.
    pub fn reset(&mut self, _qubit: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        Err(quantrs2_core::error::QuantRS2Error::UnsupportedOperation(
            "Reset operation is not yet implemented. Reset requires special quantum state manipulation.".to_string()
        ))
    }

    /// Add a barrier to prevent optimization across this point
    ///
    /// Barriers are used to prevent gate optimization algorithms from reordering gates
    /// across specific points in the circuit. This is useful for maintaining timing
    /// constraints or preserving specific circuit structure.
    pub fn barrier(&mut self, qubits: &[QubitId]) -> QuantRS2Result<&mut Self> {
        // Validate all qubits are within range
        for &qubit in qubits {
            if qubit.id() as usize >= N {
                return Err(quantrs2_core::error::QuantRS2Error::InvalidQubitId(
                    qubit.id(),
                ));
            }
        }

        // Record the barrier so that optimization passes (e.g. gate commutation
        // reordering, peephole fusion) know they must not move any gate in
        // `qubits` across this boundary.
        //
        // `after_gate_index` is the current gate-list length: the barrier is
        // logically placed *between* `gates[len-1]` and the next gate appended
        // after this call.
        self.barriers.push(BarrierInfo {
            after_gate_index: self.gates().len(),
            qubits: qubits.to_vec(),
        });
        Ok(self)
    }

    /// Measure all qubits in the circuit
    pub fn measure_all(&mut self) -> QuantRS2Result<&mut Self> {
        for i in 0..N {
            self.measure(QubitId(i as u32))?;
        }
        Ok(self)
    }

    // ============ Batch / Bulk Gate Helpers ============

    /// Apply Hadamard gates to multiple qubits at once
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<5>::new();
    /// circuit.h_all(&[0, 1, 2])?; // Apply H to qubits 0, 1, and 2
    /// ```
    pub fn h_all(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        for &qubit in qubits {
            self.h(QubitId::new(qubit))?;
        }
        Ok(self)
    }

    /// Apply Pauli-X gates to multiple qubits at once
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<5>::new();
    /// circuit.x_all(&[0, 2, 4])?; // Apply X to qubits 0, 2, and 4
    /// ```
    pub fn x_all(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        for &qubit in qubits {
            self.x(QubitId::new(qubit))?;
        }
        Ok(self)
    }

    /// Apply Pauli-Y gates to multiple qubits at once
    pub fn y_all(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        for &qubit in qubits {
            self.y(QubitId::new(qubit))?;
        }
        Ok(self)
    }

    /// Apply Pauli-Z gates to multiple qubits at once
    pub fn z_all(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        for &qubit in qubits {
            self.z(QubitId::new(qubit))?;
        }
        Ok(self)
    }

    /// Apply Hadamard gates to a range of qubits
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<5>::new();
    /// circuit.h_range(0..3)?; // Apply H to qubits 0, 1, and 2
    /// ```
    pub fn h_range(&mut self, range: std::ops::Range<u32>) -> QuantRS2Result<&mut Self> {
        for qubit in range {
            self.h(QubitId::new(qubit))?;
        }
        Ok(self)
    }

    /// Apply Pauli-X gates to a range of qubits
    pub fn x_range(&mut self, range: std::ops::Range<u32>) -> QuantRS2Result<&mut Self> {
        for qubit in range {
            self.x(QubitId::new(qubit))?;
        }
        Ok(self)
    }

    /// Apply a rotation gate to multiple qubits with the same angle
    pub fn rx_all(&mut self, qubits: &[u32], theta: f64) -> QuantRS2Result<&mut Self> {
        for &qubit in qubits {
            self.rx(QubitId::new(qubit), theta)?;
        }
        Ok(self)
    }

    /// Apply RY rotation to multiple qubits
    pub fn ry_all(&mut self, qubits: &[u32], theta: f64) -> QuantRS2Result<&mut Self> {
        for &qubit in qubits {
            self.ry(QubitId::new(qubit), theta)?;
        }
        Ok(self)
    }

    /// Apply RZ rotation to multiple qubits
    pub fn rz_all(&mut self, qubits: &[u32], theta: f64) -> QuantRS2Result<&mut Self> {
        for &qubit in qubits {
            self.rz(QubitId::new(qubit), theta)?;
        }
        Ok(self)
    }

    /// Apply SWAP gates to multiple qubit pairs
    ///
    /// # Arguments
    /// * `pairs` - Slice of (control, target) qubit pairs
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<6>::new();
    /// circuit.swap_all(&[(0, 1), (2, 3), (4, 5)])?; // Swap three pairs simultaneously
    /// ```
    pub fn swap_all(&mut self, pairs: &[(u32, u32)]) -> QuantRS2Result<&mut Self> {
        for &(q1, q2) in pairs {
            self.swap(QubitId::new(q1), QubitId::new(q2))?;
        }
        Ok(self)
    }

    /// Apply CZ gates to multiple qubit pairs
    ///
    /// # Arguments
    /// * `pairs` - Slice of (control, target) qubit pairs
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<6>::new();
    /// circuit.cz_all(&[(0, 1), (2, 3), (4, 5)])?; // Apply CZ to three pairs
    /// ```
    pub fn cz_all(&mut self, pairs: &[(u32, u32)]) -> QuantRS2Result<&mut Self> {
        for &(q1, q2) in pairs {
            self.cz(QubitId::new(q1), QubitId::new(q2))?;
        }
        Ok(self)
    }

    /// Apply CNOT gates to multiple qubit pairs
    ///
    /// # Arguments
    /// * `pairs` - Slice of (control, target) qubit pairs
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<6>::new();
    /// circuit.cnot_all(&[(0, 1), (2, 3), (4, 5)])?; // Apply CNOT to three pairs
    /// ```
    pub fn cnot_all(&mut self, pairs: &[(u32, u32)]) -> QuantRS2Result<&mut Self> {
        for &(control, target) in pairs {
            self.cnot(QubitId::new(control), QubitId::new(target))?;
        }
        Ok(self)
    }

    /// Add barriers to multiple qubits
    ///
    /// Barriers prevent optimization across them and can be used to
    /// visualize circuit structure.
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<5>::new();
    /// circuit.h_all(&[0, 1, 2])?;
    /// circuit.barrier_all(&[0, 1, 2])?; // Prevent optimization across this point
    /// circuit.cnot_ladder(&[0, 1, 2])?;
    /// ```
    pub fn barrier_all(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        let qubit_ids: Vec<QubitId> = qubits.iter().map(|&q| QubitId::new(q)).collect();
        self.barrier(&qubit_ids)?;
        Ok(self)
    }

    // ============ Python Feature-Gated Helpers ============

    /// Get a qubit for a specific single-qubit gate by gate type and index
    #[cfg(feature = "python")]
    pub fn get_single_qubit_for_gate(&self, gate_type: &str, index: usize) -> pyo3::PyResult<u32> {
        self.find_gate_by_type_and_index(gate_type, index)
            .and_then(|gate| {
                if gate.qubits().len() == 1 {
                    Some(gate.qubits()[0].id())
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Gate {gate_type} at index {index} not found or is not a single-qubit gate"
                ))
            })
    }

    /// Get rotation parameters (qubit, angle) for a specific gate by gate type and index
    #[cfg(feature = "python")]
    pub fn get_rotation_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> pyo3::PyResult<(u32, f64)> {
        self.find_gate_by_type_and_index(gate_type, index)
            .and_then(|gate| {
                if gate.qubits().len() == 1 {
                    Some((gate.qubits()[0].id(), 0.0))
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Gate {gate_type} at index {index} not found or is not a rotation gate"
                ))
            })
    }

    /// Get two-qubit parameters (control, target) for a specific gate by gate type and index
    #[cfg(feature = "python")]
    pub fn get_two_qubit_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> pyo3::PyResult<(u32, u32)> {
        self.find_gate_by_type_and_index(gate_type, index)
            .and_then(|gate| {
                if gate.qubits().len() == 2 {
                    Some((gate.qubits()[0].id(), gate.qubits()[1].id()))
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Gate {gate_type} at index {index} not found or is not a two-qubit gate"
                ))
            })
    }

    /// Get controlled rotation parameters (control, target, angle) for a specific gate
    #[cfg(feature = "python")]
    pub fn get_controlled_rotation_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> pyo3::PyResult<(u32, u32, f64)> {
        self.find_gate_by_type_and_index(gate_type, index)
            .and_then(|gate| {
                if gate.qubits().len() == 2 {
                    Some((gate.qubits()[0].id(), gate.qubits()[1].id(), 0.0))
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Gate {gate_type} at index {index} not found or is not a controlled rotation gate"
                ))
            })
    }

    /// Get three-qubit parameters for gates like Toffoli or Fredkin
    #[cfg(feature = "python")]
    pub fn get_three_qubit_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> pyo3::PyResult<(u32, u32, u32)> {
        self.find_gate_by_type_and_index(gate_type, index)
            .and_then(|gate| {
                if gate.qubits().len() == 3 {
                    Some((
                        gate.qubits()[0].id(),
                        gate.qubits()[1].id(),
                        gate.qubits()[2].id(),
                    ))
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Gate {gate_type} at index {index} not found or is not a three-qubit gate"
                ))
            })
    }
}
