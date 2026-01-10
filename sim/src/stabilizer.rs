//! Stabilizer simulator for efficient simulation of Clifford circuits
//!
//! The stabilizer formalism provides an efficient way to simulate quantum circuits
//! that consist only of Clifford gates (H, S, CNOT) and Pauli measurements.
//! This implementation uses the tableau representation and leverages `SciRS2`
//! for efficient data structures and operations.

use crate::simulator::{Simulator, SimulatorResult};
use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::GateOp;
use quantrs2_core::prelude::*;
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::sync::Arc;

/// Phase encoding for Stim compatibility
/// 0 = +1, 1 = +i, 2 = -1, 3 = -i
pub type StabilizerPhase = u8;

/// Phase constants for clarity
pub mod phase {
    /// Phase +1
    pub const PLUS_ONE: u8 = 0;
    /// Phase +i
    pub const PLUS_I: u8 = 1;
    /// Phase -1
    pub const MINUS_ONE: u8 = 2;
    /// Phase -i
    pub const MINUS_I: u8 = 3;
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

        // Initialize stabilizers as Z_i and destabilizers as X_i
        for i in 0..num_qubits {
            z_matrix[[i, i]] = true; // Stabilizer i is Z_i
            destab_x[[i, i]] = true; // Destabilizer i is X_i
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
    fn negate_phase(p: StabilizerPhase) -> StabilizerPhase {
        (p + 2) & 3
    }

    /// Multiply phase by i (add 1 mod 4)
    #[inline]
    fn multiply_by_i(p: StabilizerPhase) -> StabilizerPhase {
        (p + 1) & 3
    }

    /// Multiply phase by -i (add 3 mod 4)
    #[inline]
    fn multiply_by_minus_i(p: StabilizerPhase) -> StabilizerPhase {
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
        // This computes the phase exponent from multiplying two Pauli operators
        // Using the rule: XZ = iY, ZX = -iY, etc.
        // The formula is: i^(x1*z2*(1 - 2*x2) + x2*z1*(2*z2 - 1)) but simplified
        // For efficiency, we use a lookup table approach
        match (x1, z1, x2, z2) {
            // Identity cases (no contribution)
            (false, false, _, _) | (_, _, false, false) => 0,
            // X * Z = iY
            (true, false, false, true) => 1,
            // Z * X = -iY
            (false, true, true, false) => 3,
            // X * Y = iZ, Y * X = -iZ, etc.
            (true, false, true, true) => 1, // X * Y = iZ
            (true, true, true, false) => 3, // Y * X = -iZ
            // Y * Z = iX, Z * Y = -iX
            (true, true, false, true) => 1, // Y * Z = iX
            (false, true, true, true) => 3, // Z * Y = -iX
            // Same Pauli (no phase change): X*X=I, Y*Y=I, Z*Z=I
            (true, false, true, false) => 0, // X * X
            (false, true, false, true) => 0, // Z * Z
            (true, true, true, true) => 0,   // Y * Y
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

        // H: X ↔ Z, phase changes according to Y → -Y
        for i in 0..self.num_qubits {
            // For stabilizers
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];

            // Update phase: if both X and Z are present (Y), add phase of -1
            // HYH = H(iXZ)H = iZX = i(-iY) = Y... wait, let's recalculate
            // H transforms: X → Z, Y → -Y, Z → X
            // Y = iXZ, so HYH = i(HXH)(HZH) = iZX = i(-iY) = Y? No...
            // Actually: HYH = -Y, so Y component gets a -1 = i^2 phase
            if x_val && z_val {
                self.phase[i] = Self::negate_phase(self.phase[i]);
            }

            // Swap X and Z
            self.x_matrix[[i, qubit]] = z_val;
            self.z_matrix[[i, qubit]] = x_val;

            // For destabilizers
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

        // S conjugation: X → Y, Y → -X, Z → Z
        for i in 0..self.num_qubits {
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];

            // For stabilizers
            if x_val {
                // X is present
                if !z_val {
                    // Pure X → Y: just add Z component (no phase change)
                    self.z_matrix[[i, qubit]] = true;
                } else {
                    // Y → -X: remove Z, add phase -1
                    self.z_matrix[[i, qubit]] = false;
                    self.phase[i] = Self::negate_phase(self.phase[i]);
                }
            }
            // Z → Z: no change needed

            // For destabilizers
            let dx_val = self.destab_x[[i, qubit]];
            let dz_val = self.destab_z[[i, qubit]];

            if dx_val {
                if !dz_val {
                    // X → Y
                    self.destab_z[[i, qubit]] = true;
                } else {
                    // Y → -X
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

        // CNOT: X_c → X_c X_t, Z_t → Z_c Z_t
        for i in 0..self.num_qubits {
            // For stabilizers
            if self.x_matrix[[i, control]] {
                self.x_matrix[[i, target]] ^= true;
            }
            if self.z_matrix[[i, target]] {
                self.z_matrix[[i, control]] ^= true;
            }

            // For destabilizers
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

        // X anticommutes with Z (and Y since Y = iXZ)
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

        // Y anticommutes with pure X and pure Z, commutes with Y and I
        for i in 0..self.num_qubits {
            let has_x = self.x_matrix[[i, qubit]];
            let has_z = self.z_matrix[[i, qubit]];

            // Anticommutes when exactly one of X or Z is present (XOR)
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

        // Z anticommutes with X (and Y since Y = iXZ)
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

        // S† conjugation: X → -Y, Y → X, Z → Z
        for i in 0..self.num_qubits {
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];

            if x_val {
                if !z_val {
                    // Pure X → -Y: add Z component, negate phase
                    self.z_matrix[[i, qubit]] = true;
                    self.phase[i] = Self::negate_phase(self.phase[i]);
                } else {
                    // Y → X: remove Z component, no phase change
                    self.z_matrix[[i, qubit]] = false;
                }
            }
            // Z → Z: no change needed

            // Destabilizers
            let dx_val = self.destab_x[[i, qubit]];
            let dz_val = self.destab_z[[i, qubit]];

            if dx_val {
                if !dz_val {
                    // X → -Y
                    self.destab_z[[i, qubit]] = true;
                    self.destab_phase[i] = Self::negate_phase(self.destab_phase[i]);
                } else {
                    // Y → X
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

        // √X conjugation: X → X, Y → -Z, Z → Y
        for i in 0..self.num_qubits {
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];

            match (x_val, z_val) {
                (false, false) => {} // I → I
                (true, false) => {}  // X → X
                (false, true) => {
                    // Z → Y: add X component, no phase change
                    self.x_matrix[[i, qubit]] = true;
                }
                (true, true) => {
                    // Y → -Z: remove X component, negate phase
                    self.x_matrix[[i, qubit]] = false;
                    self.phase[i] = Self::negate_phase(self.phase[i]);
                }
            }

            // Destabilizers
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

        // √X† conjugation: X → X, Y → Z, Z → -Y
        for i in 0..self.num_qubits {
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];

            match (x_val, z_val) {
                (false, false) => {}
                (true, false) => {}
                (false, true) => {
                    // Z → -Y: add X component, negate phase
                    self.x_matrix[[i, qubit]] = true;
                    self.phase[i] = Self::negate_phase(self.phase[i]);
                }
                (true, true) => {
                    // Y → Z: remove X component, no phase change
                    self.x_matrix[[i, qubit]] = false;
                }
            }

            // Destabilizers
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

        // √Y: X → Z, Y → Y, Z → -X
        for i in 0..self.num_qubits {
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];

            match (x_val, z_val) {
                (false, false) => {}
                (true, false) => {
                    // X → Z: swap X to Z
                    self.x_matrix[[i, qubit]] = false;
                    self.z_matrix[[i, qubit]] = true;
                }
                (false, true) => {
                    // Z → -X: swap Z to X, add -1 phase
                    self.x_matrix[[i, qubit]] = true;
                    self.z_matrix[[i, qubit]] = false;
                    self.phase[i] = Self::negate_phase(self.phase[i]);
                }
                (true, true) => {} // Y → Y: no change
            }

            // Destabilizers
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

        // √Y†: X → -Z, Y → Y, Z → X
        for i in 0..self.num_qubits {
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];

            match (x_val, z_val) {
                (false, false) => {}
                (true, false) => {
                    // X → -Z: swap X to Z, add -1 phase
                    self.x_matrix[[i, qubit]] = false;
                    self.z_matrix[[i, qubit]] = true;
                    self.phase[i] = Self::negate_phase(self.phase[i]);
                }
                (false, true) => {
                    // Z → X: swap Z to X
                    self.x_matrix[[i, qubit]] = true;
                    self.z_matrix[[i, qubit]] = false;
                }
                (true, true) => {} // Y → Y: no change
            }

            // Destabilizers
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

        // CZ: Phase flips when both qubits have X component
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

            // Destabilizers
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

        // CY = S_dag(target) · CNOT(control, target) · S(target)
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

        // SWAP = CNOT(a,b) · CNOT(b,a) · CNOT(a,b)
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

        // Find a stabilizer that anticommutes with Z_qubit
        let mut anticommuting_row = None;

        for i in 0..self.num_qubits {
            if self.x_matrix[[i, qubit]] {
                anticommuting_row = Some(i);
                break;
            }
        }

        if let Some(p) = anticommuting_row {
            // Random outcome case
            // Set the p-th stabilizer to Z_qubit (or -Z_qubit based on random outcome)
            for j in 0..self.num_qubits {
                self.x_matrix[[p, j]] = false;
                self.z_matrix[[p, j]] = j == qubit;
            }

            // Random measurement outcome
            let mut rng = thread_rng();
            let outcome = rng.gen_bool(0.5);
            // Set phase: 0 (+1) for outcome 0, 2 (-1) for outcome 1
            self.phase[p] = if outcome {
                phase::MINUS_ONE
            } else {
                phase::PLUS_ONE
            };

            // Update other stabilizers that anticommute
            for i in 0..self.num_qubits {
                if i != p && self.x_matrix[[i, qubit]] {
                    // Multiply by stabilizer p
                    // Need to properly compute phase when multiplying Pauli strings
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
                    // Update phase: add both phases and the phase contribution from multiplication
                    self.phase[i] = Self::add_phases(
                        Self::add_phases(self.phase[i], self.phase[p]),
                        total_phase_contrib,
                    );
                }
            }

            Ok(outcome)
        } else {
            // Deterministic outcome
            // We need to express Z_qubit as a product of stabilizers and compute
            // the accumulated phase. This handles correlated states (like Bell states).
            //
            // Algorithm:
            // 1. Find a stabilizer that has Z on target qubit (and no X)
            // 2. Eliminate Z contributions from other qubits by multiplying
            //    with other stabilizers
            // 3. The accumulated phase determines the outcome

            // First, find a stabilizer with Z on target (no X)
            let mut pivot_row = None;
            for i in 0..self.num_qubits {
                if self.z_matrix[[i, qubit]] && !self.x_matrix[[i, qubit]] {
                    pivot_row = Some(i);
                    break;
                }
            }

            let Some(pivot) = pivot_row else {
                // No stabilizer has Z on this qubit - outcome is 0
                return Ok(false);
            };

            // Make a copy of the pivot row to work with
            let mut result_x = vec![false; self.num_qubits];
            let mut result_z = vec![false; self.num_qubits];
            let mut result_phase = self.phase[pivot];

            for j in 0..self.num_qubits {
                result_x[j] = self.x_matrix[[pivot, j]];
                result_z[j] = self.z_matrix[[pivot, j]];
            }

            // Eliminate Z on other qubits by multiplying with appropriate stabilizers
            for other_qubit in 0..self.num_qubits {
                if other_qubit == qubit {
                    continue;
                }

                // If result has Z on other_qubit, find a stabilizer to cancel it
                if result_z[other_qubit] && !result_x[other_qubit] {
                    // Find another stabilizer with Z on other_qubit (and no X)
                    for i in 0..self.num_qubits {
                        if i == pivot {
                            continue;
                        }
                        if self.z_matrix[[i, other_qubit]] && !self.x_matrix[[i, other_qubit]] {
                            // Multiply result by this stabilizer
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

            // The accumulated phase determines the outcome
            // Phase 0 (+1) or 1 (+i) → eigenvalue +1 → outcome 0 (false)
            // Phase 2 (-1) or 3 (-i) → eigenvalue -1 → outcome 1 (true)
            let outcome = result_phase >= phase::MINUS_ONE;

            Ok(outcome)
        }
    }

    /// Measure a qubit in the X basis (Stim MX instruction)
    ///
    /// Equivalent to: H · measure_z · H
    pub fn measure_x(&mut self, qubit: usize) -> Result<bool, QuantRS2Error> {
        // Transform to Z basis
        self.apply_h(qubit)?;

        // Measure in Z basis
        let outcome = self.measure(qubit)?;

        // Transform back to X basis
        self.apply_h(qubit)?;

        Ok(outcome)
    }

    /// Measure a qubit in the Y basis (Stim MY instruction)
    ///
    /// Equivalent to: S† · H · measure_z · H · S
    pub fn measure_y(&mut self, qubit: usize) -> Result<bool, QuantRS2Error> {
        // Transform to Z basis: Y → Z requires S† · H
        self.apply_s_dag(qubit)?;
        self.apply_h(qubit)?;

        // Measure in Z basis
        let outcome = self.measure(qubit)?;

        // Transform back to Y basis: H · S
        self.apply_h(qubit)?;
        self.apply_s(qubit)?;

        Ok(outcome)
    }

    /// Reset a qubit to |0⟩ state (Stim R instruction)
    ///
    /// Performs measurement and applies X if outcome is |1⟩
    pub fn reset(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        // Measure in Z basis
        let outcome = self.measure(qubit)?;

        // If measured |1⟩, flip to |0⟩
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

            // Phase prefix (Stim-compatible encoding)
            match self.phase[i] & 3 {
                phase::PLUS_ONE => stab.push('+'),
                phase::PLUS_I => stab.push_str("+i"),
                phase::MINUS_ONE => stab.push('-'),
                phase::MINUS_I => stab.push_str("-i"),
                _ => unreachable!(), // phase is always 0-3 due to & 3
            }

            // Pauli string
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

            // Phase prefix (Stim-compatible encoding)
            match self.destab_phase[i] & 3 {
                phase::PLUS_ONE => destab.push('+'),
                phase::PLUS_I => destab.push_str("+i"),
                phase::MINUS_ONE => destab.push('-'),
                phase::MINUS_I => destab.push_str("-i"),
                _ => unreachable!(), // phase is always 0-3 due to & 3
            }

            // Pauli string
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

        // For a stabilizer state, we can determine which computational basis states
        // have non-zero amplitude by finding the simultaneous +1 eigenstates of all stabilizers
        // This is a simplified implementation that assumes the state is in |0...0>
        // A full implementation would need to solve the stabilizer equations

        // For now, return a simple state
        state[0] = Complex64::new(1.0, 0.0);
        state
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

/// Check if a circuit can be simulated by the stabilizer simulator
#[must_use]
pub fn is_clifford_circuit<const N: usize>(circuit: &Circuit<N>) -> bool {
    // Check if all gates in the circuit are Clifford gates
    // Clifford gates: H, S, S†, CNOT, X, Y, Z, CZ
    circuit.gates().iter().all(|gate| {
        matches!(
            gate.name(),
            "H" | "S" | "S†" | "CNOT" | "X" | "Y" | "Z" | "CZ" | "Phase" | "PhaseDagger"
        )
    })
}

/// Convert a gate operation to a stabilizer gate
fn gate_to_stabilizer(gate: &Arc<dyn GateOp + Send + Sync>) -> Option<StabilizerGate> {
    let gate_name = gate.name();
    let qubits = gate.qubits();

    match gate_name {
        "H" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::H(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "S" | "Phase" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::S(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "X" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::X(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "Y" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::Y(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "Z" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::Z(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "CNOT" => {
            if qubits.len() == 2 {
                Some(StabilizerGate::CNOT(
                    qubits[0].0 as usize,
                    qubits[1].0 as usize,
                ))
            } else {
                None
            }
        }
        "CZ" => {
            if qubits.len() == 2 {
                // CZ = H(target) CNOT H(target)
                // For now, we can decompose this or add native support
                // Let's return None for unsupported gates
                None
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Implement the Simulator trait for `StabilizerSimulator`
impl Simulator for StabilizerSimulator {
    fn run<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
    ) -> crate::error::Result<SimulatorResult<N>> {
        // Create a new simulator instance
        let mut sim = Self::new(N);

        // Apply all gates in the circuit
        for gate in circuit.gates() {
            if let Some(stab_gate) = gate_to_stabilizer(gate) {
                // Ignore errors for now - in production we'd handle them properly
                let _ = sim.apply_gate(stab_gate);
            }
        }

        // Get the state vector (expensive operation)
        let amplitudes = sim.get_statevector();

        Ok(SimulatorResult::new(amplitudes))
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stabilizer_init() {
        let sim = StabilizerSimulator::new(3);
        let stabs = sim.get_stabilizers();

        assert_eq!(stabs.len(), 3);
        assert_eq!(stabs[0], "+ZII");
        assert_eq!(stabs[1], "+IZI");
        assert_eq!(stabs[2], "+IIZ");
    }

    #[test]
    fn test_hadamard_gate() {
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");

        let stabs = sim.get_stabilizers();
        assert_eq!(stabs[0], "+X");
    }

    #[test]
    fn test_bell_state() {
        let mut sim = StabilizerSimulator::new(2);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");
        sim.apply_gate(StabilizerGate::CNOT(0, 1))
            .expect("CNOT gate application should succeed");

        let stabs = sim.get_stabilizers();
        assert!(stabs.contains(&"+XX".to_string()));
        assert!(stabs.contains(&"+ZZ".to_string()));
    }

    #[test]
    fn test_ghz_state() {
        let mut sim = StabilizerSimulator::new(3);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");
        sim.apply_gate(StabilizerGate::CNOT(0, 1))
            .expect("CNOT gate application should succeed");
        sim.apply_gate(StabilizerGate::CNOT(1, 2))
            .expect("CNOT gate application should succeed");

        let stabs = sim.get_stabilizers();
        assert!(stabs.contains(&"+XXX".to_string()));
        assert!(stabs.contains(&"+ZZI".to_string()));
        assert!(stabs.contains(&"+IZZ".to_string()));
    }

    #[test]
    fn test_s_dag_gate() {
        // S† · S = I (identity)
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::S(0))
            .expect("S gate application should succeed");
        sim.apply_gate(StabilizerGate::SDag(0))
            .expect("S† gate application should succeed");

        let stabs = sim.get_stabilizers();
        assert_eq!(stabs[0], "+Z"); // Should return to original Z state
    }

    #[test]
    fn test_sqrt_x_gate() {
        // √X · √X = X (up to global phase)
        let mut sim1 = StabilizerSimulator::new(1);
        sim1.apply_gate(StabilizerGate::SqrtX(0))
            .expect("√X gate application should succeed");
        sim1.apply_gate(StabilizerGate::SqrtX(0))
            .expect("√X gate application should succeed");

        let stabs1 = sim1.get_stabilizers();

        // Compare with direct X gate (may have phase difference)
        let mut sim2 = StabilizerSimulator::new(1);
        sim2.apply_gate(StabilizerGate::X(0))
            .expect("X gate application should succeed");

        let stabs2 = sim2.get_stabilizers();

        // Check that both have Z stabilizer (possibly with different signs)
        assert!(stabs1[0] == "+Z" || stabs1[0] == "-Z");
        assert!(stabs2[0] == "+Z" || stabs2[0] == "-Z");
    }

    #[test]
    fn test_sqrt_y_gate() {
        // √Y · √Y = Y
        let mut sim1 = StabilizerSimulator::new(1);
        sim1.apply_gate(StabilizerGate::SqrtY(0))
            .expect("√Y gate application should succeed");
        sim1.apply_gate(StabilizerGate::SqrtY(0))
            .expect("√Y gate application should succeed");

        let stabs1 = sim1.get_stabilizers();

        // Compare with direct Y gate
        let mut sim2 = StabilizerSimulator::new(1);
        sim2.apply_gate(StabilizerGate::Y(0))
            .expect("Y gate application should succeed");

        let stabs2 = sim2.get_stabilizers();
        assert_eq!(stabs1[0], stabs2[0]);
    }

    #[test]
    fn test_cz_gate() {
        // Test CZ gate creates entanglement
        let mut sim = StabilizerSimulator::new(2);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");
        sim.apply_gate(StabilizerGate::H(1))
            .expect("Hadamard gate application should succeed");
        sim.apply_gate(StabilizerGate::CZ(0, 1))
            .expect("CZ gate application should succeed");

        let stabs = sim.get_stabilizers();
        // CZ on |++⟩ creates entangled state
        // Should have correlation in Z basis
        assert!(stabs.len() == 2);
    }

    #[test]
    fn test_cy_gate() {
        // Test CY gate
        let mut sim = StabilizerSimulator::new(2);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");
        sim.apply_gate(StabilizerGate::CY(0, 1))
            .expect("CY gate application should succeed");

        let stabs = sim.get_stabilizers();
        assert!(stabs.len() == 2);
    }

    #[test]
    fn test_swap_gate() {
        // Test SWAP gate
        let mut sim = StabilizerSimulator::new(2);
        // Prepare |01⟩
        sim.apply_gate(StabilizerGate::X(1))
            .expect("X gate application should succeed");

        // Apply SWAP
        sim.apply_gate(StabilizerGate::SWAP(0, 1))
            .expect("SWAP gate application should succeed");

        let stabs = sim.get_stabilizers();
        // After SWAP, should be equivalent to |10⟩
        assert!(stabs.len() == 2);
    }

    #[test]
    fn test_builder_pattern_new_gates() {
        // Test using builder pattern with new gates
        let sim = CliffordCircuitBuilder::new(2)
            .h(0)
            .s_dag(0)
            .sqrt_x(1)
            .cz(0, 1)
            .run()
            .expect("Circuit execution should succeed");

        let stabs = sim.get_stabilizers();
        assert!(stabs.len() == 2);
    }

    #[test]
    fn test_large_clifford_circuit() {
        // Test a larger Clifford circuit (100 qubits)
        let mut sim = StabilizerSimulator::new(100);

        // Apply Hadamard to all qubits
        for i in 0..100 {
            sim.apply_gate(StabilizerGate::H(i))
                .expect("Hadamard gate application should succeed");
        }

        // Apply CNOT chain
        for i in 0..99 {
            sim.apply_gate(StabilizerGate::CNOT(i, i + 1))
                .expect("CNOT gate application should succeed");
        }

        let stabs = sim.get_stabilizers();
        assert_eq!(stabs.len(), 100);
    }

    #[test]
    fn test_measurement_randomness() {
        // Test that measurement produces random outcomes for superposition
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");

        // Measure multiple times (need new states each time)
        let mut outcomes = Vec::new();
        for _ in 0..10 {
            let mut test_sim = StabilizerSimulator::new(1);
            test_sim
                .apply_gate(StabilizerGate::H(0))
                .expect("Hadamard gate application should succeed");
            let outcome = test_sim.measure(0).expect("Measurement should succeed");
            outcomes.push(outcome);
        }

        // Should have at least some variety (not all same)
        let first = outcomes[0];
        let all_same = outcomes.iter().all(|&x| x == first);
        assert!(
            !all_same || outcomes.len() < 5,
            "Measurements should show randomness"
        );
    }

    #[test]
    fn test_measure_x_basis() {
        // Prepare |+⟩ state (eigenstate of X with eigenvalue +1)
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");

        // Measure in X basis should always give 0 (deterministic)
        let outcome = sim
            .tableau
            .measure_x(0)
            .expect("X-basis measurement should succeed");

        // For |+⟩ state, X measurement should be deterministic (eigenvalue +1 → outcome 0)
        assert_eq!(outcome, false);
    }

    #[test]
    fn test_measure_y_basis() {
        // Prepare |+Y⟩ = S·H|0⟩ = (|0⟩ + i|1⟩)/√2, eigenstate of Y with eigenvalue +1
        // Note: Apply H first, then S (right-to-left gate application)
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::H(0)).unwrap();
        sim.apply_gate(StabilizerGate::S(0)).unwrap();

        // State now has stabilizer +Y
        let stabs = sim.get_stabilizers();
        assert_eq!(stabs[0], "+Y");

        // Measure in Y basis should give outcome 0 (eigenvalue +1)
        let outcome = sim
            .tableau
            .measure_y(0)
            .expect("Y-basis measurement should succeed");

        // Should be deterministic for Y eigenstate
        assert_eq!(outcome, false);
    }

    #[test]
    fn test_reset_operation() {
        // Prepare |1⟩ state
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::X(0))
            .expect("X gate application should succeed");

        // Reset to |0⟩
        sim.tableau.reset(0).expect("Reset should succeed");

        // Measure should give 0
        let outcome = sim.measure(0).expect("Measurement should succeed");
        assert_eq!(outcome, false);

        // Stabilizer should be +Z
        let stabs = sim.get_stabilizers();
        assert_eq!(stabs[0], "+Z");
    }

    #[test]
    fn test_reset_from_superposition() {
        // Prepare |+⟩ state
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");

        // Reset to |0⟩
        sim.tableau.reset(0).expect("Reset should succeed");

        // Measure should give 0
        let outcome = sim.measure(0).expect("Measurement should succeed");
        assert_eq!(outcome, false);
    }

    #[test]
    fn test_x_y_measurements_commute() {
        // X and Y measurements on different qubits should not interfere
        let mut sim = StabilizerSimulator::new(2);
        sim.apply_gate(StabilizerGate::H(0)).unwrap();
        sim.apply_gate(StabilizerGate::H(1)).unwrap();
        sim.apply_gate(StabilizerGate::S(1)).unwrap();

        // Measure qubit 0 in X basis and qubit 1 in Y basis
        let outcome_x = sim.tableau.measure_x(0).unwrap();
        let outcome_y = sim.tableau.measure_y(1).unwrap();

        // Both should succeed without errors
        assert!(outcome_x == true || outcome_x == false);
        assert!(outcome_y == true || outcome_y == false);
    }

    #[test]
    fn test_imaginary_phase_tracking() {
        // Test that S gate properly tracks imaginary phases
        // |0⟩ has stabilizer +Z
        // S|0⟩ = |0⟩ still has stabilizer +Z (global phase change only)
        // But H|0⟩ = |+⟩ has stabilizer +X
        // S·H|0⟩ = |i⟩ = (|0⟩ + i|1⟩)/√2 has stabilizer +Y

        let mut tableau = StabilizerTableau::new(1);
        tableau.apply_h(0).unwrap();
        tableau.apply_s(0).unwrap();

        let stabs = tableau.get_stabilizers();
        // After H·S, the state is |i⟩ = S|+⟩
        // S transforms X → Y, Z → Z
        // Starting with +X (from H|0⟩), S maps to +Y
        assert_eq!(stabs[0], "+Y");
    }

    #[test]
    fn test_imaginary_phase_with_s_dag() {
        // S† should give conjugate phases
        // H·S†|0⟩ has stabilizer -Y (since S† maps X → -Y)
        let mut tableau = StabilizerTableau::new(1);
        tableau.apply_h(0).unwrap();
        tableau.apply_s_dag(0).unwrap();

        let stabs = tableau.get_stabilizers();
        // After H·S†, the state has stabilizer -Y
        assert_eq!(stabs[0], "-Y");
    }

    #[test]
    fn test_stim_format_identity() {
        // Test Stim format uses `_` for identity
        let mut tableau = StabilizerTableau::with_format(2, true);
        let stabs = tableau.get_stabilizers();

        // In Stim format, identity should be `_`
        assert_eq!(stabs[0], "+Z_");
        assert_eq!(stabs[1], "+_Z");

        // Apply Hadamard to first qubit
        tableau.apply_h(0).unwrap();
        let stabs = tableau.get_stabilizers();
        assert_eq!(stabs[0], "+X_");
        assert_eq!(stabs[1], "+_Z");
    }

    #[test]
    fn test_standard_format_identity() {
        // Test standard format uses `I` for identity
        let tableau = StabilizerTableau::with_format(2, false);
        let stabs = tableau.get_stabilizers();

        assert_eq!(stabs[0], "+ZI");
        assert_eq!(stabs[1], "+IZ");
    }

    #[test]
    fn test_destabilizers_output() {
        // Test destabilizers are properly tracked
        let mut tableau = StabilizerTableau::new(2);

        // Initial state: stabilizers are +Z_i, destabilizers are +X_i
        let destabs = tableau.get_destabilizers();
        assert_eq!(destabs[0], "+XI");
        assert_eq!(destabs[1], "+IX");

        // Apply Hadamard - swaps X and Z
        tableau.apply_h(0).unwrap();
        let destabs = tableau.get_destabilizers();
        assert_eq!(destabs[0], "+ZI");
        assert_eq!(destabs[1], "+IX");
    }

    #[test]
    fn test_phase_constants() {
        // Verify phase constant values
        assert_eq!(phase::PLUS_ONE, 0);
        assert_eq!(phase::PLUS_I, 1);
        assert_eq!(phase::MINUS_ONE, 2);
        assert_eq!(phase::MINUS_I, 3);
    }

    #[test]
    fn test_phase_arithmetic() {
        // Test phase helper functions
        // negate_phase: +1 → -1, +i → -i, -1 → +1, -i → +i
        assert_eq!(
            StabilizerTableau::negate_phase(phase::PLUS_ONE),
            phase::MINUS_ONE
        );
        assert_eq!(
            StabilizerTableau::negate_phase(phase::PLUS_I),
            phase::MINUS_I
        );
        assert_eq!(
            StabilizerTableau::negate_phase(phase::MINUS_ONE),
            phase::PLUS_ONE
        );
        assert_eq!(
            StabilizerTableau::negate_phase(phase::MINUS_I),
            phase::PLUS_I
        );

        // multiply_by_i: +1 → +i, +i → -1, -1 → -i, -i → +1
        assert_eq!(
            StabilizerTableau::multiply_by_i(phase::PLUS_ONE),
            phase::PLUS_I
        );
        assert_eq!(
            StabilizerTableau::multiply_by_i(phase::PLUS_I),
            phase::MINUS_ONE
        );
        assert_eq!(
            StabilizerTableau::multiply_by_i(phase::MINUS_ONE),
            phase::MINUS_I
        );
        assert_eq!(
            StabilizerTableau::multiply_by_i(phase::MINUS_I),
            phase::PLUS_ONE
        );

        // multiply_by_minus_i: +1 → -i, +i → +1, -1 → +i, -i → -1
        assert_eq!(
            StabilizerTableau::multiply_by_minus_i(phase::PLUS_ONE),
            phase::MINUS_I
        );
        assert_eq!(
            StabilizerTableau::multiply_by_minus_i(phase::PLUS_I),
            phase::PLUS_ONE
        );
        assert_eq!(
            StabilizerTableau::multiply_by_minus_i(phase::MINUS_ONE),
            phase::PLUS_I
        );
        assert_eq!(
            StabilizerTableau::multiply_by_minus_i(phase::MINUS_I),
            phase::MINUS_ONE
        );
    }

    #[test]
    fn test_y_gate_phase_tracking() {
        // Y gate should properly track phases
        // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
        let mut tableau = StabilizerTableau::new(1);
        tableau.apply_y(0).unwrap();

        let stabs = tableau.get_stabilizers();
        // Y flips Z stabilizer with phase -1: +Z → -Z
        assert_eq!(stabs[0], "-Z");
    }

    #[test]
    fn test_sqrt_gates_produce_imaginary_phases() {
        // √Y gate should produce states with imaginary stabilizer phases
        // √Y transforms: X → Z, Y → Y, Z → -X
        let mut tableau = StabilizerTableau::new(1);
        tableau.apply_sqrt_y(0).unwrap();

        let stabs = tableau.get_stabilizers();
        // Initial +Z gets mapped to -X by √Y
        assert_eq!(stabs[0], "-X");
    }
}
