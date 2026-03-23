//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::functions::{phase, StabilizerPhase};

/// Builder for creating circuits that can be simulated with the stabilizer formalism
pub struct CliffordCircuitBuilder {
    gates: Vec<StabilizerGate>,
    num_qubits: usize,
}
impl CliffordCircuitBuilder {
    /// Create a new Clifford circuit builder
    #[must_use]
    pub const fn new(num_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
        }
    }
    /// Add a Hadamard gate
    #[must_use]
    pub fn h(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::H(qubit));
        self
    }
    /// Add an S gate
    #[must_use]
    pub fn s(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::S(qubit));
        self
    }
    /// Add an S† (S-dagger) gate
    #[must_use]
    pub fn s_dag(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::SDag(qubit));
        self
    }
    /// Add a √X gate
    #[must_use]
    pub fn sqrt_x(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::SqrtX(qubit));
        self
    }
    /// Add a √X† gate
    #[must_use]
    pub fn sqrt_x_dag(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::SqrtXDag(qubit));
        self
    }
    /// Add a √Y gate
    #[must_use]
    pub fn sqrt_y(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::SqrtY(qubit));
        self
    }
    /// Add a √Y† gate
    #[must_use]
    pub fn sqrt_y_dag(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::SqrtYDag(qubit));
        self
    }
    /// Add a Pauli-X gate
    #[must_use]
    pub fn x(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::X(qubit));
        self
    }
    /// Add a Pauli-Y gate
    #[must_use]
    pub fn y(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::Y(qubit));
        self
    }
    /// Add a Pauli-Z gate
    #[must_use]
    pub fn z(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::Z(qubit));
        self
    }
    /// Add a CNOT gate
    #[must_use]
    pub fn cnot(mut self, control: usize, target: usize) -> Self {
        self.gates.push(StabilizerGate::CNOT(control, target));
        self
    }
    /// Add a CZ (Controlled-Z) gate
    #[must_use]
    pub fn cz(mut self, control: usize, target: usize) -> Self {
        self.gates.push(StabilizerGate::CZ(control, target));
        self
    }
    /// Add a CY (Controlled-Y) gate
    #[must_use]
    pub fn cy(mut self, control: usize, target: usize) -> Self {
        self.gates.push(StabilizerGate::CY(control, target));
        self
    }
    /// Add a SWAP gate
    #[must_use]
    pub fn swap(mut self, qubit1: usize, qubit2: usize) -> Self {
        self.gates.push(StabilizerGate::SWAP(qubit1, qubit2));
        self
    }
    /// Build and run the circuit
    pub fn run(self) -> Result<StabilizerSimulator, QuantRS2Error> {
        let mut sim = StabilizerSimulator::new(self.num_qubits);
        for gate in self.gates {
            sim.apply_gate(gate)?;
        }
        Ok(sim)
    }
}
/// Stabilizer tableau representation
///
/// The tableau stores generators of the stabilizer group as rows.
/// Each row represents a Pauli string with phase.
///
/// Phase encoding (Stim-compatible):
/// - 0 = +1
/// - 1 = +i
/// - 2 = -1
/// - 3 = -i
#[derive(Debug, Clone)]
pub struct StabilizerTableau {
    /// Number of qubits
    num_qubits: usize,
    /// X part of stabilizers (n x n matrix)
    x_matrix: Array2<bool>,
    /// Z part of stabilizers (n x n matrix)
    z_matrix: Array2<bool>,
    /// Phase vector (n elements, encoded as 0=+1, 1=+i, 2=-1, 3=-i)
    phase: Vec<StabilizerPhase>,
    /// Destabilizers X part (n x n matrix)
    destab_x: Array2<bool>,
    /// Destabilizers Z part (n x n matrix)
    destab_z: Array2<bool>,
    /// Destabilizer phases (same encoding as phase)
    destab_phase: Vec<StabilizerPhase>,
    /// Pauli string format: true for Stim-style (`_` for identity), false for standard (`I`)
    stim_format: bool,
}
impl StabilizerTableau {
    /// Create a new tableau in the |0...0⟩ state
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self::with_format(num_qubits, false)
    }
    /// Create a new tableau with specified Pauli string format
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    /// * `stim_format` - Use Stim format (`_` for identity) if true, standard format (`I`) if false
    #[must_use]
    pub fn with_format(num_qubits: usize, stim_format: bool) -> Self {
        let mut x_matrix = Array2::from_elem((num_qubits, num_qubits), false);
        let mut z_matrix = Array2::from_elem((num_qubits, num_qubits), false);
        let mut destab_x = Array2::from_elem((num_qubits, num_qubits), false);
        let mut destab_z = Array2::from_elem((num_qubits, num_qubits), false);
        for i in 0..num_qubits {
            z_matrix[[i, i]] = true;
            destab_x[[i, i]] = true;
        }
        Self {
            num_qubits,
            x_matrix,
            z_matrix,
            phase: vec![phase::PLUS_ONE; num_qubits],
            destab_x,
            destab_z,
            destab_phase: vec![phase::PLUS_ONE; num_qubits],
            stim_format,
        }
    }
    /// Set the Pauli string format
    pub fn set_stim_format(&mut self, stim_format: bool) {
        self.stim_format = stim_format;
    }
    /// Get the Pauli string format
    #[must_use]
    pub const fn is_stim_format(&self) -> bool {
        self.stim_format
    }
    /// Multiply phase by -1 (add 2 mod 4)
    #[inline]
    pub(crate) fn negate_phase(p: StabilizerPhase) -> StabilizerPhase {
        (p + 2) & 3
    }
    /// Multiply phase by i (add 1 mod 4)
    #[inline]
    pub(crate) fn multiply_by_i(p: StabilizerPhase) -> StabilizerPhase {
        (p + 1) & 3
    }
    /// Multiply phase by -i (add 3 mod 4)
    #[inline]
    pub(crate) fn multiply_by_minus_i(p: StabilizerPhase) -> StabilizerPhase {
        (p + 3) & 3
    }
    /// Add two phases (mod 4)
    #[inline]
    fn add_phases(p1: StabilizerPhase, p2: StabilizerPhase) -> StabilizerPhase {
        (p1 + p2) & 3
    }
    /// Compute the phase contribution from row multiplication
    /// When multiplying Pauli strings P1 and P2, the phase depends on the anticommutation
    /// XZ = iY, ZX = -iY, etc.
    #[inline]
    fn rowsum_phase(x1: bool, z1: bool, x2: bool, z2: bool) -> StabilizerPhase {
        match (x1, z1, x2, z2) {
            (false, false, _, _) | (_, _, false, false) => 0,
            (true, false, false, true) => 1,
            (false, true, true, false) => 3,
            (true, false, true, true) => 1,
            (true, true, true, false) => 3,
            (true, true, false, true) => 1,
            (false, true, true, true) => 3,
            (true, false, true, false) => 0,
            (false, true, false, true) => 0,
            (true, true, true, true) => 0,
        }
    }
    /// Compute the phase contribution when multiplying a Pauli string (given as vectors)
    /// with row `row_idx` from the tableau
    fn compute_multiplication_phase(
        &self,
        result_x: &[bool],
        result_z: &[bool],
        row_idx: usize,
    ) -> StabilizerPhase {
        let mut total_phase: StabilizerPhase = 0;
        for j in 0..self.num_qubits {
            let phase_contrib = Self::rowsum_phase(
                result_x[j],
                result_z[j],
                self.x_matrix[[row_idx, j]],
                self.z_matrix[[row_idx, j]],
            );
            total_phase = Self::add_phases(total_phase, phase_contrib);
        }
        total_phase
    }
    /// Apply a Hadamard gate
    ///
    /// H: X → Z, Z → X, Y → -Y
    /// Phase tracking: HYH = -Y, so Y component contributes i^2 = -1
    pub fn apply_h(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        for i in 0..self.num_qubits {
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];
            if x_val && z_val {
                self.phase[i] = Self::negate_phase(self.phase[i]);
            }
            self.x_matrix[[i, qubit]] = z_val;
            self.z_matrix[[i, qubit]] = x_val;
            let dx_val = self.destab_x[[i, qubit]];
            let dz_val = self.destab_z[[i, qubit]];
            if dx_val && dz_val {
                self.destab_phase[i] = Self::negate_phase(self.destab_phase[i]);
            }
            self.destab_x[[i, qubit]] = dz_val;
            self.destab_z[[i, qubit]] = dx_val;
        }
        Ok(())
    }
    /// Apply an S gate (phase gate)
    ///
    /// S conjugation rules (SPS†):
    /// - S: X → Y (no phase change, Pauli relabeling)
    /// - S: Y → -X (phase negation due to SYS† = -X)
    /// - S: Z → Z (no change)
    ///
    /// Note: The `i` in Y = iXZ is a matrix identity, not relevant to stabilizer
    /// conjugation. In stabilizer formalism, X, Y, Z are atomic Pauli labels.
    pub fn apply_s(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        for i in 0..self.num_qubits {
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];
            if x_val {
                if !z_val {
                    self.z_matrix[[i, qubit]] = true;
                } else {
                    self.z_matrix[[i, qubit]] = false;
                    self.phase[i] = Self::negate_phase(self.phase[i]);
                }
            }
            let dx_val = self.destab_x[[i, qubit]];
            let dz_val = self.destab_z[[i, qubit]];
            if dx_val {
                if !dz_val {
                    self.destab_z[[i, qubit]] = true;
                } else {
                    self.destab_z[[i, qubit]] = false;
                    self.destab_phase[i] = Self::negate_phase(self.destab_phase[i]);
                }
            }
        }
        Ok(())
    }
    /// Apply a CNOT gate
    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), QuantRS2Error> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(control.max(target) as u32));
        }
        if control == target {
            return Err(QuantRS2Error::InvalidInput(
                "CNOT control and target must be different".to_string(),
            ));
        }
        for i in 0..self.num_qubits {
            if self.x_matrix[[i, control]] {
                self.x_matrix[[i, target]] ^= true;
            }
            if self.z_matrix[[i, target]] {
                self.z_matrix[[i, control]] ^= true;
            }
            if self.destab_x[[i, control]] {
                self.destab_x[[i, target]] ^= true;
            }
            if self.destab_z[[i, target]] {
                self.destab_z[[i, control]] ^= true;
            }
        }
        Ok(())
    }
    /// Apply a Pauli X gate
    ///
    /// X anticommutes with Z and Y, commutes with X
    /// Phase: adds -1 when Z or Y is present on the qubit
    pub fn apply_x(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        for i in 0..self.num_qubits {
            if self.z_matrix[[i, qubit]] {
                self.phase[i] = Self::negate_phase(self.phase[i]);
            }
            if self.destab_z[[i, qubit]] {
                self.destab_phase[i] = Self::negate_phase(self.destab_phase[i]);
            }
        }
        Ok(())
    }
    /// Apply a Pauli Y gate
    ///
    /// Y = iXZ, anticommutes with X and Z (separately), commutes with Y
    /// Phase: adds -1 when X XOR Z is present (pure X or pure Z, not Y)
    pub fn apply_y(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        for i in 0..self.num_qubits {
            let has_x = self.x_matrix[[i, qubit]];
            let has_z = self.z_matrix[[i, qubit]];
            if has_x != has_z {
                self.phase[i] = Self::negate_phase(self.phase[i]);
            }
            let has_dx = self.destab_x[[i, qubit]];
            let has_dz = self.destab_z[[i, qubit]];
            if has_dx != has_dz {
                self.destab_phase[i] = Self::negate_phase(self.destab_phase[i]);
            }
        }
        Ok(())
    }
    /// Apply a Pauli Z gate
    ///
    /// Z anticommutes with X and Y, commutes with Z
    /// Phase: adds -1 when X or Y is present on the qubit
    pub fn apply_z(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        for i in 0..self.num_qubits {
            if self.x_matrix[[i, qubit]] {
                self.phase[i] = Self::negate_phase(self.phase[i]);
            }
            if self.destab_x[[i, qubit]] {
                self.destab_phase[i] = Self::negate_phase(self.destab_phase[i]);
            }
        }
        Ok(())
    }
    /// Apply S† (S-dagger) gate
    ///
    /// S† conjugation rules (S†PS):
    /// - S†: X → -Y (phase becomes -1)
    /// - S†: Y → X (no phase change)
    /// - S†: Z → Z (no change)
    pub fn apply_s_dag(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        for i in 0..self.num_qubits {
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];
            if x_val {
                if !z_val {
                    self.z_matrix[[i, qubit]] = true;
                    self.phase[i] = Self::negate_phase(self.phase[i]);
                } else {
                    self.z_matrix[[i, qubit]] = false;
                }
            }
            let dx_val = self.destab_x[[i, qubit]];
            let dz_val = self.destab_z[[i, qubit]];
            if dx_val {
                if !dz_val {
                    self.destab_z[[i, qubit]] = true;
                    self.destab_phase[i] = Self::negate_phase(self.destab_phase[i]);
                } else {
                    self.destab_z[[i, qubit]] = false;
                }
            }
        }
        Ok(())
    }
    /// Apply √X gate (SQRT_X, also called SX or V gate)
    ///
    /// Conjugation rules:
    /// - √X: X → X (no change)
    /// - √X: Y → -Z (phase becomes -1)
    /// - √X: Z → Y (no phase change)
    pub fn apply_sqrt_x(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        for i in 0..self.num_qubits {
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];
            match (x_val, z_val) {
                (false, false) => {}
                (true, false) => {}
                (false, true) => {
                    self.x_matrix[[i, qubit]] = true;
                }
                (true, true) => {
                    self.x_matrix[[i, qubit]] = false;
                    self.phase[i] = Self::negate_phase(self.phase[i]);
                }
            }
            let dx_val = self.destab_x[[i, qubit]];
            let dz_val = self.destab_z[[i, qubit]];
            match (dx_val, dz_val) {
                (false, false) => {}
                (true, false) => {}
                (false, true) => {
                    self.destab_x[[i, qubit]] = true;
                }
                (true, true) => {
                    self.destab_x[[i, qubit]] = false;
                    self.destab_phase[i] = Self::negate_phase(self.destab_phase[i]);
                }
            }
        }
        Ok(())
    }
    /// Apply √X† gate (SQRT_X_DAG)
    ///
    /// Conjugation rules:
    /// - √X†: X → X (no change)
    /// - √X†: Y → Z (no phase change)
    /// - √X†: Z → -Y (phase becomes -1)
    pub fn apply_sqrt_x_dag(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        for i in 0..self.num_qubits {
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];
            match (x_val, z_val) {
                (false, false) => {}
                (true, false) => {}
                (false, true) => {
                    self.x_matrix[[i, qubit]] = true;
                    self.phase[i] = Self::negate_phase(self.phase[i]);
                }
                (true, true) => {
                    self.x_matrix[[i, qubit]] = false;
                }
            }
            let dx_val = self.destab_x[[i, qubit]];
            let dz_val = self.destab_z[[i, qubit]];
            match (dx_val, dz_val) {
                (false, false) => {}
                (true, false) => {}
                (false, true) => {
                    self.destab_x[[i, qubit]] = true;
                    self.destab_phase[i] = Self::negate_phase(self.destab_phase[i]);
                }
                (true, true) => {
                    self.destab_x[[i, qubit]] = false;
                }
            }
        }
        Ok(())
    }
    /// Apply √Y gate (SQRT_Y)
    ///
    /// Conjugation rules:
    /// - √Y: X → Z (no phase change)
    /// - √Y: Y → Y (no change)
    /// - √Y: Z → -X (phase becomes -1)
    pub fn apply_sqrt_y(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        for i in 0..self.num_qubits {
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];
            match (x_val, z_val) {
                (false, false) => {}
                (true, false) => {
                    self.x_matrix[[i, qubit]] = false;
                    self.z_matrix[[i, qubit]] = true;
                }
                (false, true) => {
                    self.x_matrix[[i, qubit]] = true;
                    self.z_matrix[[i, qubit]] = false;
                    self.phase[i] = Self::negate_phase(self.phase[i]);
                }
                (true, true) => {}
            }
            let dx_val = self.destab_x[[i, qubit]];
            let dz_val = self.destab_z[[i, qubit]];
            match (dx_val, dz_val) {
                (false, false) => {}
                (true, false) => {
                    self.destab_x[[i, qubit]] = false;
                    self.destab_z[[i, qubit]] = true;
                }
                (false, true) => {
                    self.destab_x[[i, qubit]] = true;
                    self.destab_z[[i, qubit]] = false;
                    self.destab_phase[i] = Self::negate_phase(self.destab_phase[i]);
                }
                (true, true) => {}
            }
        }
        Ok(())
    }
    /// Apply √Y† gate (SQRT_Y_DAG)
    ///
    /// Conjugation rules:
    /// - √Y†: X → -Z (phase becomes -1)
    /// - √Y†: Y → Y (no change)
    /// - √Y†: Z → X (no phase change)
    pub fn apply_sqrt_y_dag(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        for i in 0..self.num_qubits {
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];
            match (x_val, z_val) {
                (false, false) => {}
                (true, false) => {
                    self.x_matrix[[i, qubit]] = false;
                    self.z_matrix[[i, qubit]] = true;
                    self.phase[i] = Self::negate_phase(self.phase[i]);
                }
                (false, true) => {
                    self.x_matrix[[i, qubit]] = true;
                    self.z_matrix[[i, qubit]] = false;
                }
                (true, true) => {}
            }
            let dx_val = self.destab_x[[i, qubit]];
            let dz_val = self.destab_z[[i, qubit]];
            match (dx_val, dz_val) {
                (false, false) => {}
                (true, false) => {
                    self.destab_x[[i, qubit]] = false;
                    self.destab_z[[i, qubit]] = true;
                    self.destab_phase[i] = Self::negate_phase(self.destab_phase[i]);
                }
                (false, true) => {
                    self.destab_x[[i, qubit]] = true;
                    self.destab_z[[i, qubit]] = false;
                }
                (true, true) => {}
            }
        }
        Ok(())
    }
    /// Apply CZ (Controlled-Z) gate
    ///
    /// CZ: X_c → X_c Z_t, X_t → Z_c X_t, Z_c → Z_c, Z_t → Z_t
    /// When both qubits have X component (product of X or Y), phase picks up -1
    pub fn apply_cz(&mut self, control: usize, target: usize) -> Result<(), QuantRS2Error> {
        if control >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(control as u32));
        }
        if target >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(target as u32));
        }
        for i in 0..self.num_qubits {
            if self.x_matrix[[i, control]] && self.x_matrix[[i, target]] {
                self.phase[i] = Self::negate_phase(self.phase[i]);
            }
            if self.x_matrix[[i, control]] {
                self.z_matrix[[i, target]] = !self.z_matrix[[i, target]];
            }
            if self.x_matrix[[i, target]] {
                self.z_matrix[[i, control]] = !self.z_matrix[[i, control]];
            }
            if self.destab_x[[i, control]] && self.destab_x[[i, target]] {
                self.destab_phase[i] = Self::negate_phase(self.destab_phase[i]);
            }
            if self.destab_x[[i, control]] {
                self.destab_z[[i, target]] = !self.destab_z[[i, target]];
            }
            if self.destab_x[[i, target]] {
                self.destab_z[[i, control]] = !self.destab_z[[i, control]];
            }
        }
        Ok(())
    }
    /// Apply CY (Controlled-Y) gate
    pub fn apply_cy(&mut self, control: usize, target: usize) -> Result<(), QuantRS2Error> {
        if control >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(control as u32));
        }
        if target >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(target as u32));
        }
        self.apply_s_dag(target)?;
        self.apply_cnot(control, target)?;
        self.apply_s(target)?;
        Ok(())
    }
    /// Apply SWAP gate
    pub fn apply_swap(&mut self, qubit1: usize, qubit2: usize) -> Result<(), QuantRS2Error> {
        if qubit1 >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit1 as u32));
        }
        if qubit2 >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit2 as u32));
        }
        self.apply_cnot(qubit1, qubit2)?;
        self.apply_cnot(qubit2, qubit1)?;
        self.apply_cnot(qubit1, qubit2)?;
        Ok(())
    }
    /// Measure a qubit in the computational (Z) basis
    /// Returns the measurement outcome (0 or 1)
    ///
    /// For phases with imaginary components, we project onto real eigenvalues:
    /// - Phase 0 (+1) or 1 (+i) → eigenvalue +1 → outcome 0
    /// - Phase 2 (-1) or 3 (-i) → eigenvalue -1 → outcome 1
    pub fn measure(&mut self, qubit: usize) -> Result<bool, QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        let mut anticommuting_row = None;
        for i in 0..self.num_qubits {
            if self.x_matrix[[i, qubit]] {
                anticommuting_row = Some(i);
                break;
            }
        }
        if let Some(p) = anticommuting_row {
            for j in 0..self.num_qubits {
                self.x_matrix[[p, j]] = false;
                self.z_matrix[[p, j]] = j == qubit;
            }
            let mut rng = thread_rng();
            let outcome = rng.gen_bool(0.5);
            self.phase[p] = if outcome {
                phase::MINUS_ONE
            } else {
                phase::PLUS_ONE
            };
            for i in 0..self.num_qubits {
                if i != p && self.x_matrix[[i, qubit]] {
                    let mut total_phase_contrib: StabilizerPhase = 0;
                    for j in 0..self.num_qubits {
                        let phase_contrib = Self::rowsum_phase(
                            self.x_matrix[[i, j]],
                            self.z_matrix[[i, j]],
                            self.x_matrix[[p, j]],
                            self.z_matrix[[p, j]],
                        );
                        total_phase_contrib = Self::add_phases(total_phase_contrib, phase_contrib);
                        self.x_matrix[[i, j]] ^= self.x_matrix[[p, j]];
                        self.z_matrix[[i, j]] ^= self.z_matrix[[p, j]];
                    }
                    self.phase[i] = Self::add_phases(
                        Self::add_phases(self.phase[i], self.phase[p]),
                        total_phase_contrib,
                    );
                }
            }
            Ok(outcome)
        } else {
            let mut pivot_row = None;
            for i in 0..self.num_qubits {
                if self.z_matrix[[i, qubit]] && !self.x_matrix[[i, qubit]] {
                    pivot_row = Some(i);
                    break;
                }
            }
            let Some(pivot) = pivot_row else {
                return Ok(false);
            };
            let mut result_x = vec![false; self.num_qubits];
            let mut result_z = vec![false; self.num_qubits];
            let mut result_phase = self.phase[pivot];
            for j in 0..self.num_qubits {
                result_x[j] = self.x_matrix[[pivot, j]];
                result_z[j] = self.z_matrix[[pivot, j]];
            }
            for other_qubit in 0..self.num_qubits {
                if other_qubit == qubit {
                    continue;
                }
                if result_z[other_qubit] && !result_x[other_qubit] {
                    for i in 0..self.num_qubits {
                        if i == pivot {
                            continue;
                        }
                        if self.z_matrix[[i, other_qubit]] && !self.x_matrix[[i, other_qubit]] {
                            let phase_contrib =
                                self.compute_multiplication_phase(&result_x, &result_z, i);
                            result_phase = Self::add_phases(result_phase, self.phase[i]);
                            result_phase = Self::add_phases(result_phase, phase_contrib);
                            for j in 0..self.num_qubits {
                                result_x[j] ^= self.x_matrix[[i, j]];
                                result_z[j] ^= self.z_matrix[[i, j]];
                            }
                            break;
                        }
                    }
                }
            }
            let outcome = result_phase >= phase::MINUS_ONE;
            Ok(outcome)
        }
    }
    /// Measure a qubit in the X basis (Stim MX instruction)
    ///
    /// Equivalent to: H · measure_z · H
    pub fn measure_x(&mut self, qubit: usize) -> Result<bool, QuantRS2Error> {
        self.apply_h(qubit)?;
        let outcome = self.measure(qubit)?;
        self.apply_h(qubit)?;
        Ok(outcome)
    }
    /// Measure a qubit in the Y basis (Stim MY instruction)
    ///
    /// Equivalent to: S† · H · measure_z · H · S
    pub fn measure_y(&mut self, qubit: usize) -> Result<bool, QuantRS2Error> {
        self.apply_s_dag(qubit)?;
        self.apply_h(qubit)?;
        let outcome = self.measure(qubit)?;
        self.apply_h(qubit)?;
        self.apply_s(qubit)?;
        Ok(outcome)
    }
    /// Reset a qubit to |0⟩ state (Stim R instruction)
    ///
    /// Performs measurement and applies X if outcome is |1⟩
    pub fn reset(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        let outcome = self.measure(qubit)?;
        if outcome {
            self.apply_x(qubit)?;
        }
        Ok(())
    }
    /// Get the current stabilizer generators as strings
    ///
    /// Phase encoding in output:
    /// - `+` for phase 0 (+1)
    /// - `+i` for phase 1 (+i)
    /// - `-` for phase 2 (-1)
    /// - `-i` for phase 3 (-i)
    ///
    /// Identity representation depends on `stim_format`:
    /// - Standard format: `I` for identity
    /// - Stim format: `_` for identity
    #[must_use]
    pub fn get_stabilizers(&self) -> Vec<String> {
        let mut stabilizers = Vec::new();
        let identity_char = if self.stim_format { '_' } else { 'I' };
        for i in 0..self.num_qubits {
            let mut stab = String::new();
            match self.phase[i] & 3 {
                phase::PLUS_ONE => stab.push('+'),
                phase::PLUS_I => stab.push_str("+i"),
                phase::MINUS_ONE => stab.push('-'),
                phase::MINUS_I => stab.push_str("-i"),
                _ => unreachable!(),
            }
            for j in 0..self.num_qubits {
                let has_x = self.x_matrix[[i, j]];
                let has_z = self.z_matrix[[i, j]];
                match (has_x, has_z) {
                    (false, false) => stab.push(identity_char),
                    (true, false) => stab.push('X'),
                    (false, true) => stab.push('Z'),
                    (true, true) => stab.push('Y'),
                }
            }
            stabilizers.push(stab);
        }
        stabilizers
    }
    /// Get the current destabilizer generators as strings
    ///
    /// Same format as `get_stabilizers()` but for destabilizers
    #[must_use]
    pub fn get_destabilizers(&self) -> Vec<String> {
        let mut destabilizers = Vec::new();
        let identity_char = if self.stim_format { '_' } else { 'I' };
        for i in 0..self.num_qubits {
            let mut destab = String::new();
            match self.destab_phase[i] & 3 {
                phase::PLUS_ONE => destab.push('+'),
                phase::PLUS_I => destab.push_str("+i"),
                phase::MINUS_ONE => destab.push('-'),
                phase::MINUS_I => destab.push_str("-i"),
                _ => unreachable!(),
            }
            for j in 0..self.num_qubits {
                let has_x = self.destab_x[[i, j]];
                let has_z = self.destab_z[[i, j]];
                match (has_x, has_z) {
                    (false, false) => destab.push(identity_char),
                    (true, false) => destab.push('X'),
                    (false, true) => destab.push('Z'),
                    (true, true) => destab.push('Y'),
                }
            }
            destabilizers.push(destab);
        }
        destabilizers
    }
}
/// Gates supported by the stabilizer simulator
#[derive(Debug, Clone, Copy)]
pub enum StabilizerGate {
    H(usize),
    S(usize),
    SDag(usize),
    SqrtX(usize),
    SqrtXDag(usize),
    SqrtY(usize),
    SqrtYDag(usize),
    X(usize),
    Y(usize),
    Z(usize),
    CNOT(usize, usize),
    CZ(usize, usize),
    CY(usize, usize),
    SWAP(usize, usize),
}
/// Stabilizer simulator that efficiently simulates Clifford circuits
#[derive(Debug, Clone)]
pub struct StabilizerSimulator {
    /// The stabilizer tableau
    pub tableau: StabilizerTableau,
    measurement_record: Vec<(usize, bool)>,
}
impl StabilizerSimulator {
    /// Create a new stabilizer simulator
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            tableau: StabilizerTableau::new(num_qubits),
            measurement_record: Vec::new(),
        }
    }
    /// Apply a gate to the simulator
    pub fn apply_gate(&mut self, gate: StabilizerGate) -> Result<(), QuantRS2Error> {
        match gate {
            StabilizerGate::H(q) => self.tableau.apply_h(q),
            StabilizerGate::S(q) => self.tableau.apply_s(q),
            StabilizerGate::SDag(q) => self.tableau.apply_s_dag(q),
            StabilizerGate::SqrtX(q) => self.tableau.apply_sqrt_x(q),
            StabilizerGate::SqrtXDag(q) => self.tableau.apply_sqrt_x_dag(q),
            StabilizerGate::SqrtY(q) => self.tableau.apply_sqrt_y(q),
            StabilizerGate::SqrtYDag(q) => self.tableau.apply_sqrt_y_dag(q),
            StabilizerGate::X(q) => self.tableau.apply_x(q),
            StabilizerGate::Y(q) => self.tableau.apply_y(q),
            StabilizerGate::Z(q) => self.tableau.apply_z(q),
            StabilizerGate::CNOT(c, t) => self.tableau.apply_cnot(c, t),
            StabilizerGate::CZ(c, t) => self.tableau.apply_cz(c, t),
            StabilizerGate::CY(c, t) => self.tableau.apply_cy(c, t),
            StabilizerGate::SWAP(q1, q2) => self.tableau.apply_swap(q1, q2),
        }
    }
    /// Measure a qubit
    pub fn measure(&mut self, qubit: usize) -> Result<bool, QuantRS2Error> {
        let outcome = self.tableau.measure(qubit)?;
        self.measurement_record.push((qubit, outcome));
        Ok(outcome)
    }
    /// Get the current stabilizers
    #[must_use]
    pub fn get_stabilizers(&self) -> Vec<String> {
        self.tableau.get_stabilizers()
    }
    /// Get measurement record
    #[must_use]
    pub fn get_measurements(&self) -> &[(usize, bool)] {
        &self.measurement_record
    }
    /// Reset the simulator
    pub fn reset(&mut self) {
        let num_qubits = self.tableau.num_qubits;
        self.tableau = StabilizerTableau::new(num_qubits);
        self.measurement_record.clear();
    }
    /// Get the number of qubits
    #[must_use]
    pub const fn num_qubits(&self) -> usize {
        self.tableau.num_qubits
    }
    /// Get the state vector (for compatibility with other simulators)
    /// Note: This is expensive for stabilizer states and returns a sparse representation
    #[must_use]
    pub fn get_statevector(&self) -> Vec<Complex64> {
        let n = self.tableau.num_qubits;
        let dim = 1 << n;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);
        state
    }
}
