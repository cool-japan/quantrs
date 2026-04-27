//! Pauli frame tracking for fault-tolerant quantum error correction.
//!
//! A Pauli frame represents a virtual Pauli correction that accumulates during
//! stabilizer-based error correction. Rather than physically applying corrections
//! to the quantum state, we track the corrections in classical registers and
//! propagate them through gate operations.

use crate::error_correction::pauli::{Pauli, PauliString};

/// Classical Pauli frame tracker for fault-tolerant circuits.
///
/// Tracks accumulated X and Z corrections that would be needed to map the
/// current (potentially corrupted) state back to the ideal state.
/// Gates commuted through are tracked symbolically rather than applied.
#[derive(Debug, Clone, Default)]
pub struct PauliFrame {
    /// X component of the accumulated frame (true = X or Y on that qubit)
    pub x_frame: Vec<bool>,
    /// Z component of the accumulated frame (true = Z or Y on that qubit)
    pub z_frame: Vec<bool>,
}

impl PauliFrame {
    /// Create a new all-identity Pauli frame for `n_qubits` qubits.
    pub fn new(n_qubits: usize) -> Self {
        Self {
            x_frame: vec![false; n_qubits],
            z_frame: vec![false; n_qubits],
        }
    }

    /// Number of qubits tracked in this frame.
    pub fn n_qubits(&self) -> usize {
        self.x_frame.len()
    }

    /// Apply a `PauliString` correction to the frame (XOR into the frame).
    ///
    /// I → no change
    /// X → flip x_frame
    /// Z → flip z_frame
    /// Y → flip both x_frame and z_frame
    pub fn apply_pauli_string(&mut self, ps: &PauliString) {
        for (i, &p) in ps.paulis.iter().enumerate() {
            if i >= self.x_frame.len() {
                break;
            }
            match p {
                Pauli::I => {}
                Pauli::X => {
                    self.x_frame[i] ^= true;
                }
                Pauli::Z => {
                    self.z_frame[i] ^= true;
                }
                Pauli::Y => {
                    self.x_frame[i] ^= true;
                    self.z_frame[i] ^= true;
                }
            }
        }
    }

    /// Propagate the Pauli frame through a Hadamard gate on `qubit`.
    ///
    /// H conjugation rule: `H X H† = Z`, `H Z H† = X`.
    /// Therefore X and Z components are swapped.
    pub fn commute_through_h(&mut self, qubit: usize) {
        if qubit < self.x_frame.len() {
            // H swaps X and Z
            std::mem::swap(&mut self.x_frame[qubit], &mut self.z_frame[qubit]);
        }
    }

    /// Propagate the Pauli frame through an S gate on `qubit`.
    ///
    /// S conjugation rule: `S X S† = Y = iXZ`, `S Z S† = Z`.
    /// Effect on frame: X → X (x stays), Z component gains x_frame (Z gets +x).
    ///   If x was true: Z also becomes true (X → Y, so Z component added).
    ///   If x was false: nothing changes for Z.
    pub fn commute_through_s(&mut self, qubit: usize) {
        if qubit < self.x_frame.len() {
            // S: X → Y (X stays, Z appears); Z → Z (unchanged)
            // So z_frame[qubit] ^= x_frame[qubit]
            self.z_frame[qubit] ^= self.x_frame[qubit];
        }
    }

    /// Propagate the Pauli frame through a CNOT gate.
    ///
    /// CNOT conjugation rules:
    /// - `CNOT (X_c ⊗ I) CNOT† = X_c ⊗ X_t`   (X propagates forward from ctrl to tgt)
    /// - `CNOT (I ⊗ X_t) CNOT† = I ⊗ X_t`      (X on target unchanged)
    /// - `CNOT (Z_c ⊗ I) CNOT† = Z_c ⊗ I`      (Z on ctrl unchanged)
    /// - `CNOT (I ⊗ Z_t) CNOT† = Z_c ⊗ Z_t`    (Z propagates backward from tgt to ctrl)
    pub fn commute_through_cnot(&mut self, ctrl: usize, tgt: usize) {
        let n = self.x_frame.len();
        if ctrl >= n || tgt >= n {
            return;
        }
        // X propagates ctrl → tgt
        let x_ctrl = self.x_frame[ctrl];
        self.x_frame[tgt] ^= x_ctrl;
        // Z propagates tgt → ctrl
        let z_tgt = self.z_frame[tgt];
        self.z_frame[ctrl] ^= z_tgt;
    }

    /// Measure the logical X operator.
    ///
    /// Returns `true` if the accumulated frame has a logical-Z error (i.e., the
    /// Z-frame parity is odd over the logical-X support qubits).
    ///
    /// `logical_x_qubits`: the data qubit indices forming the logical X string.
    pub fn measure_logical_x(&self, logical_x_qubits: &[usize]) -> bool {
        logical_x_qubits
            .iter()
            .filter(|&&q| q < self.z_frame.len())
            .filter(|&&q| self.z_frame[q])
            .count()
            % 2
            == 1
    }

    /// Measure the logical Z operator.
    ///
    /// Returns `true` if the accumulated frame has a logical-X error (i.e., the
    /// X-frame parity is odd over the logical-Z support qubits).
    ///
    /// `logical_z_qubits`: the data qubit indices forming the logical Z string.
    pub fn measure_logical_z(&self, logical_z_qubits: &[usize]) -> bool {
        logical_z_qubits
            .iter()
            .filter(|&&q| q < self.x_frame.len())
            .filter(|&&q| self.x_frame[q])
            .count()
            % 2
            == 1
    }

    /// Merge another frame into this one (XOR both x and z components).
    pub fn merge(&mut self, other: &PauliFrame) {
        for (i, (&ox, &oz)) in other.x_frame.iter().zip(&other.z_frame).enumerate() {
            if i >= self.x_frame.len() {
                break;
            }
            self.x_frame[i] ^= ox;
            self.z_frame[i] ^= oz;
        }
    }

    /// Returns `true` if this frame represents the identity (no corrections needed).
    pub fn is_identity(&self) -> bool {
        self.x_frame.iter().all(|&x| !x) && self.z_frame.iter().all(|&z| !z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_identity() {
        let frame = PauliFrame::new(5);
        assert!(frame.is_identity());
    }

    #[test]
    fn test_apply_pauli_string_x() {
        let mut frame = PauliFrame::new(3);
        let ps = PauliString::new(vec![Pauli::X, Pauli::I, Pauli::I]);
        frame.apply_pauli_string(&ps);
        assert!(frame.x_frame[0]);
        assert!(!frame.z_frame[0]);
    }

    #[test]
    fn test_apply_pauli_string_y() {
        let mut frame = PauliFrame::new(3);
        let ps = PauliString::new(vec![Pauli::Y, Pauli::I, Pauli::I]);
        frame.apply_pauli_string(&ps);
        assert!(frame.x_frame[0]);
        assert!(frame.z_frame[0]);
    }

    #[test]
    fn test_h_gate_commutation() {
        let mut frame = PauliFrame::new(2);
        frame.x_frame[0] = true;
        frame.z_frame[0] = false;
        frame.commute_through_h(0);
        // H: X → Z, so x_frame should be false and z_frame true
        assert!(!frame.x_frame[0], "After H: x should be false (was true→Z)");
        assert!(frame.z_frame[0], "After H: z should be true (X→Z)");
    }

    #[test]
    fn test_h_gate_z_to_x() {
        let mut frame = PauliFrame::new(2);
        frame.x_frame[0] = false;
        frame.z_frame[0] = true;
        frame.commute_through_h(0);
        assert!(frame.x_frame[0], "After H: x should be true (Z→X)");
        assert!(!frame.z_frame[0], "After H: z should be false");
    }

    #[test]
    fn test_s_gate_commutation() {
        // S: X → Y (x stays true, z becomes true); Z → Z (unchanged)
        let mut frame = PauliFrame::new(2);
        frame.x_frame[0] = true;
        frame.z_frame[0] = false;
        frame.commute_through_s(0);
        assert!(frame.x_frame[0], "X component should remain after S");
        assert!(frame.z_frame[0], "Z component should appear after S on X");
    }

    #[test]
    fn test_s_gate_z_unchanged() {
        let mut frame = PauliFrame::new(2);
        frame.x_frame[0] = false;
        frame.z_frame[0] = true;
        frame.commute_through_s(0);
        assert!(!frame.x_frame[0], "X should not change for Z under S");
        assert!(frame.z_frame[0], "Z should remain unchanged under S");
    }

    #[test]
    fn test_cnot_x_propagation() {
        // CNOT ctrl=0, tgt=1: X_ctrl → X_ctrl X_tgt
        let mut frame = PauliFrame::new(2);
        frame.x_frame[0] = true;
        frame.commute_through_cnot(0, 1);
        assert!(frame.x_frame[0], "X on ctrl should remain");
        assert!(frame.x_frame[1], "X should propagate to tgt");
    }

    #[test]
    fn test_cnot_z_propagation() {
        // CNOT ctrl=0, tgt=1: Z_tgt → Z_ctrl Z_tgt
        let mut frame = PauliFrame::new(2);
        frame.z_frame[1] = true;
        frame.commute_through_cnot(0, 1);
        assert!(frame.z_frame[0], "Z should propagate back to ctrl");
        assert!(frame.z_frame[1], "Z on tgt should remain");
    }

    #[test]
    fn test_merge() {
        let mut frame1 = PauliFrame::new(3);
        frame1.x_frame[0] = true;
        frame1.z_frame[1] = true;

        let mut frame2 = PauliFrame::new(3);
        frame2.x_frame[0] = true; // same → cancel
        frame2.z_frame[2] = true;

        frame1.merge(&frame2);
        assert!(!frame1.x_frame[0], "Same X bits should cancel on merge");
        assert!(frame1.z_frame[1], "Z bit 1 should persist");
        assert!(frame1.z_frame[2], "Z bit 2 from frame2 should appear");
    }

    #[test]
    fn test_measure_logical_x() {
        let mut frame = PauliFrame::new(3);
        frame.z_frame[0] = true; // Z error on logical X support qubit
        // logical X support = [0] → odd parity of z_frame → logical Z error detected
        assert!(frame.measure_logical_x(&[0]));
        assert!(!frame.measure_logical_x(&[1]));
    }

    #[test]
    fn test_measure_logical_z() {
        let mut frame = PauliFrame::new(3);
        frame.x_frame[0] = true;
        assert!(frame.measure_logical_z(&[0]));
    }
}
