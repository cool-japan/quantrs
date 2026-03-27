//! Enhanced Clifford/Stabilizer simulator using sparse representations
//!
//! This implementation uses `scirs2_sparse::CsrMatrix<u8>` for efficient
//! simulation of large Clifford circuits.  The tableau stores Pauli X and Z
//! components as 1×N sparse row matrices (N = number of qubits).  This gives
//! O(nnz) time for gate updates, where nnz is the number of non-identity
//! Pauli factors in each generator — typically much smaller than N for early
//! circuit stages.
//!
//! ## Sparse Pauli Tableau Format
//!
//! Each stabilizer (or destabilizer) generator is stored as a `SparsePauli`:
//! - `x_sparse`: 1×N `CsrMatrix<u8>`, entry `(0, j) = 1` iff qubit j has an
//!   X or Y component.
//! - `z_sparse`: 1×N `CsrMatrix<u8>`, entry `(0, j) = 1` iff qubit j has a
//!   Z or Y component.
//! - `phase`: 0=+1, 1=+i, 2=-1, 3=-i.
//!
//! ## Gate Update Rules
//!
//! | Gate | Control (c) / Target (t) | X_c | Z_c | X_t | Z_t | Phase |
//! |------|--------------------------|-----|-----|-----|-----|-------|
//! | H(t) | —                        | ↔Z | ↔X | —  | —   | XZ→-XZ |
//! | S(t) | —                        | X→Y| —  | —  | —   | +i per Y prod |
//! | CNOT | c→t                      | X_c→X_cX_t | — | — | Z_t→Z_cZ_t | — |

use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use scirs2_sparse::CsrMatrix;

/// Build a 1×N CSR row matrix with entries at the specified column positions.
///
/// All entries are set to 1u8.  If `cols` is empty, returns an all-zero row.
fn build_row(num_qubits: usize, cols: &[usize]) -> Result<CsrMatrix<u8>, QuantRS2Error> {
    if cols.is_empty() {
        CsrMatrix::from_triplets(1, num_qubits, vec![], vec![], vec![])
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Failed to build empty CSR row: {e}")))
    } else {
        let rows: Vec<usize> = vec![0; cols.len()];
        let values: Vec<u8> = vec![1u8; cols.len()];
        CsrMatrix::from_triplets(1, num_qubits, rows, cols.to_vec(), values)
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Failed to build CSR row: {e}")))
    }
}

/// Collect non-zero column indices from a 1-row `CsrMatrix<u8>`.
fn nonzero_cols(mat: &CsrMatrix<u8>) -> Vec<usize> {
    let (_, cols, vals) = mat.get_triplets();
    cols.into_iter()
        .zip(vals)
        .filter(|(_, v)| *v > 0)
        .map(|(c, _)| c)
        .collect()
}

/// Sparse representation of a single Pauli operator over N qubits.
///
/// Both `x_sparse` and `z_sparse` are 1×N row matrices.
#[derive(Debug, Clone)]
pub struct SparsePauli {
    /// Sparse X component (1 where X or Y is present)
    x_sparse: CsrMatrix<u8>,
    /// Sparse Z component (1 where Z or Y is present)
    z_sparse: CsrMatrix<u8>,
    /// Global phase: 0 → +1, 1 → +i, 2 → −1, 3 → −i
    pub phase: u8,
    /// Number of qubits (= number of columns in each row matrix)
    num_qubits: usize,
}

impl SparsePauli {
    /// Create an identity Pauli operator (no X, no Z, phase = +1).
    pub fn identity(num_qubits: usize) -> QuantRS2Result<Self> {
        let x_sparse = build_row(num_qubits, &[])?;
        let z_sparse = build_row(num_qubits, &[])?;
        Ok(Self {
            x_sparse,
            z_sparse,
            phase: 0,
            num_qubits,
        })
    }

    /// Create a single-qubit Pauli: `pauli` ∈ {'I', 'X', 'Y', 'Z'}.
    pub fn single_qubit(num_qubits: usize, qubit: usize, pauli: char) -> QuantRS2Result<Self> {
        let x_cols: Vec<usize> = match pauli {
            'X' | 'Y' => vec![qubit],
            _ => vec![],
        };
        let z_cols: Vec<usize> = match pauli {
            'Z' | 'Y' => vec![qubit],
            _ => vec![],
        };

        Ok(Self {
            x_sparse: build_row(num_qubits, &x_cols)?,
            z_sparse: build_row(num_qubits, &z_cols)?,
            phase: 0,
            num_qubits,
        })
    }

    /// Get the X bit for a given qubit (0 or 1).
    #[inline]
    pub fn x_bit(&self, qubit: usize) -> u8 {
        self.x_sparse.get(0, qubit)
    }

    /// Get the Z bit for a given qubit (0 or 1).
    #[inline]
    pub fn z_bit(&self, qubit: usize) -> u8 {
        self.z_sparse.get(0, qubit)
    }

    /// Set the X bit for a given qubit; rebuilds the row matrix.
    pub fn set_x_bit(&mut self, qubit: usize, value: u8) -> QuantRS2Result<()> {
        let mut cols = nonzero_cols(&self.x_sparse);
        cols.retain(|&c| c != qubit);
        if value > 0 {
            cols.push(qubit);
            cols.sort_unstable();
        }
        self.x_sparse = build_row(self.num_qubits, &cols)?;
        Ok(())
    }

    /// Set the Z bit for a given qubit; rebuilds the row matrix.
    pub fn set_z_bit(&mut self, qubit: usize, value: u8) -> QuantRS2Result<()> {
        let mut cols = nonzero_cols(&self.z_sparse);
        cols.retain(|&c| c != qubit);
        if value > 0 {
            cols.push(qubit);
            cols.sort_unstable();
        }
        self.z_sparse = build_row(self.num_qubits, &cols)?;
        Ok(())
    }

    /// Number of qubits this Pauli acts on.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of non-identity factors (X + Y + Z positions).
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.x_sparse.nnz() + self.z_sparse.nnz()
    }
}

/// Enhanced stabilizer tableau backed by `SparsePauli` rows.
///
/// For an N-qubit system, the tableau holds:
/// - N stabilizer generators (initially Z_0, Z_1, …, Z_{N-1})
/// - N destabilizer generators (initially X_0, X_1, …, X_{N-1})
pub struct SparseStabilizerTableau {
    num_qubits: usize,
    stabilizers: Vec<SparsePauli>,
    destabilizers: Vec<SparsePauli>,
}

impl SparseStabilizerTableau {
    /// Initialise the tableau to |0…0⟩ (all-zero computational basis state).
    pub fn new(num_qubits: usize) -> QuantRS2Result<Self> {
        let mut stabilizers = Vec::with_capacity(num_qubits);
        let mut destabilizers = Vec::with_capacity(num_qubits);

        for i in 0..num_qubits {
            stabilizers.push(SparsePauli::single_qubit(num_qubits, i, 'Z')?);
            destabilizers.push(SparsePauli::single_qubit(num_qubits, i, 'X')?);
        }

        Ok(Self {
            num_qubits,
            stabilizers,
            destabilizers,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Apply the Hadamard update rule to one `SparsePauli` at qubit `target`.
    ///
    /// H: X_t ↔ Z_t; phase += 2 (i.e. multiply by −1) if both X and Z present.
    fn apply_h_to_pauli(pauli: &mut SparsePauli, target: usize) -> QuantRS2Result<()> {
        let x_val = pauli.x_bit(target);
        let z_val = pauli.z_bit(target);

        // Phase correction for Y (both X and Z present before swap)
        if x_val > 0 && z_val > 0 {
            pauli.phase = (pauli.phase + 2) % 4;
        }

        // Swap X ↔ Z at target
        pauli.set_x_bit(target, z_val)?;
        pauli.set_z_bit(target, x_val)?;
        Ok(())
    }

    /// Apply the S gate update rule to one `SparsePauli` at qubit `target`.
    ///
    /// S: X_t → Y_t (i.e. add Z_t), Z_t → Z_t.  If X was set but Z was not,
    /// add Z and multiply phase by i.
    fn apply_s_to_pauli(pauli: &mut SparsePauli, target: usize) -> QuantRS2Result<()> {
        let x_val = pauli.x_bit(target);
        let z_val = pauli.z_bit(target);

        if x_val > 0 && z_val == 0 {
            // X → Y: add Z component and accumulate phase i
            pauli.set_z_bit(target, 1)?;
            pauli.phase = (pauli.phase + 1) % 4;
        } else if x_val > 0 && z_val > 0 {
            // Y → -X: remove Z component and accumulate phase i
            // Y*S = iX, so phase gains +1 (i) and Z is removed
            pauli.set_z_bit(target, 0)?;
            pauli.phase = (pauli.phase + 1) % 4;
        }
        // Z → Z: no change; I → I: no change
        Ok(())
    }

    /// Apply the CNOT update rule to one `SparsePauli`.
    ///
    /// CNOT: X_c → X_c X_t (toggle X_t if X_c set)
    ///       Z_t → Z_c Z_t (toggle Z_c if Z_t set)
    fn apply_cnot_to_pauli(
        pauli: &mut SparsePauli,
        control: usize,
        target: usize,
    ) -> QuantRS2Result<()> {
        let control_x = pauli.x_bit(control);
        let target_z = pauli.z_bit(target);

        if control_x > 0 {
            let current_x_t = pauli.x_bit(target);
            pauli.set_x_bit(target, current_x_t ^ 1)?;
        }

        if target_z > 0 {
            let current_z_c = pauli.z_bit(control);
            pauli.set_z_bit(control, current_z_c ^ 1)?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Public gate interfaces
    // -----------------------------------------------------------------------

    /// Apply a Hadamard gate on `qubit`.
    pub fn apply_h(&mut self, qubit: usize) -> QuantRS2Result<()> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        for i in 0..self.num_qubits {
            Self::apply_h_to_pauli(&mut self.stabilizers[i], qubit)?;
            Self::apply_h_to_pauli(&mut self.destabilizers[i], qubit)?;
        }
        Ok(())
    }

    /// Apply an S gate on `qubit`.
    pub fn apply_s(&mut self, qubit: usize) -> QuantRS2Result<()> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        for i in 0..self.num_qubits {
            Self::apply_s_to_pauli(&mut self.stabilizers[i], qubit)?;
            Self::apply_s_to_pauli(&mut self.destabilizers[i], qubit)?;
        }
        Ok(())
    }

    /// Apply a CNOT gate with `control` and `target` qubits.
    pub fn apply_cnot(&mut self, control: usize, target: usize) -> QuantRS2Result<()> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(
                control.max(target) as u32,
            ));
        }
        for i in 0..self.num_qubits {
            Self::apply_cnot_to_pauli(&mut self.stabilizers[i], control, target)?;
            Self::apply_cnot_to_pauli(&mut self.destabilizers[i], control, target)?;
        }
        Ok(())
    }

    /// Return the stabilizer generators as human-readable Pauli strings.
    ///
    /// Each string has the form `+XIIZIY…` (phase prefix, then one Pauli
    /// letter per qubit).
    pub fn get_stabilizers(&self) -> Vec<String> {
        self.stabilizers
            .iter()
            .map(|stab| pauli_to_string(stab, self.num_qubits))
            .collect()
    }

    /// Return `(stabilizer_sparsity, destabilizer_sparsity)` where sparsity is
    /// the fraction of _zero_ entries (1 = fully sparse).
    #[must_use]
    pub fn get_sparsity_info(&self) -> (f64, f64) {
        let total = self.num_qubits * self.num_qubits;

        let stab_nnz: usize = self.stabilizers.iter().map(|p| p.nnz()).sum();
        let destab_nnz: usize = self.destabilizers.iter().map(|p| p.nnz()).sum();

        let stab_sparsity = if total == 0 {
            1.0
        } else {
            1.0 - (stab_nnz as f64 / total as f64)
        };
        let destab_sparsity = if total == 0 {
            1.0
        } else {
            1.0 - (destab_nnz as f64 / total as f64)
        };

        (stab_sparsity, destab_sparsity)
    }

    /// Number of qubits in the tableau.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

/// Format a `SparsePauli` as a printable string (`+X`, `-Z`, etc.).
fn pauli_to_string(pauli: &SparsePauli, num_qubits: usize) -> String {
    let phase_char = match pauli.phase % 4 {
        0 => '+',
        1 => 'i',
        2 => '-',
        3 => 'j', // -i represented as 'j' for brevity
        _ => '+',
    };

    let mut s = String::with_capacity(1 + num_qubits);
    s.push(phase_char);

    for j in 0..num_qubits {
        let has_x = pauli.x_bit(j) > 0;
        let has_z = pauli.z_bit(j) > 0;
        s.push(match (has_x, has_z) {
            (false, false) => 'I',
            (true, false) => 'X',
            (false, true) => 'Z',
            (true, true) => 'Y',
        });
    }

    s
}

// ---------------------------------------------------------------------------
// High-level simulator facade
// ---------------------------------------------------------------------------

/// Clifford gates supported by the sparse simulator.
#[derive(Debug, Clone, Copy)]
pub enum CliffordGate {
    H(usize),
    S(usize),
    X(usize),
    Y(usize),
    Z(usize),
    CNOT(usize, usize),
}

/// Enhanced Clifford simulator backed by a sparse stabilizer tableau.
pub struct SparseCliffordSimulator {
    tableau: SparseStabilizerTableau,
    measurement_record: Vec<(usize, bool)>,
}

impl SparseCliffordSimulator {
    /// Create a new simulator for `num_qubits` qubits, initialised to |0…0⟩.
    pub fn new(num_qubits: usize) -> QuantRS2Result<Self> {
        Ok(Self {
            tableau: SparseStabilizerTableau::new(num_qubits)?,
            measurement_record: Vec::new(),
        })
    }

    /// Apply a Clifford gate.
    pub fn apply_gate(&mut self, gate: CliffordGate) -> QuantRS2Result<()> {
        match gate {
            CliffordGate::H(q) => self.tableau.apply_h(q),
            CliffordGate::S(q) => self.tableau.apply_s(q),
            CliffordGate::CNOT(c, t) => self.tableau.apply_cnot(c, t),
            // X = HZH = S²; implement as S²·H·S²·H to keep within Clifford set.
            // For simplicity, apply the tableau update for X directly:
            // X on qubit q: Z_q → −Z_q (phase flip on Z stabilisers containing Z_q).
            // This is equivalent to H·Z·H.
            CliffordGate::X(q) => {
                self.tableau.apply_h(q)?;
                self.tableau.apply_s(q)?;
                self.tableau.apply_s(q)?;
                self.tableau.apply_h(q)
            }
            // Y = iXZ.  Apply as X then Z.
            CliffordGate::Y(q) => {
                self.apply_gate(CliffordGate::X(q))?;
                self.apply_gate(CliffordGate::Z(q))
            }
            // Z = S²
            CliffordGate::Z(q) => {
                self.tableau.apply_s(q)?;
                self.tableau.apply_s(q)
            }
        }
    }

    /// Get the current stabilizer strings.
    pub fn get_stabilizers(&self) -> Vec<String> {
        self.tableau.get_stabilizers()
    }

    /// Get sparsity information `(stab_sparsity, destab_sparsity)`.
    pub fn get_sparsity_info(&self) -> (f64, f64) {
        self.tableau.get_sparsity_info()
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.tableau.num_qubits()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_init() {
        let sim = SparseCliffordSimulator::new(100).expect("Failed to create simulator");
        let (stab_sparsity, destab_sparsity) = sim.get_sparsity_info();

        // Initial state should be very sparse (only diagonal elements)
        assert!(
            stab_sparsity > 0.98,
            "stab_sparsity too low: {}",
            stab_sparsity
        );
        assert!(
            destab_sparsity > 0.98,
            "destab_sparsity too low: {}",
            destab_sparsity
        );
    }

    #[test]
    fn test_sparse_hadamard() {
        let mut sim = SparseCliffordSimulator::new(5).expect("Failed to create simulator");
        sim.apply_gate(CliffordGate::H(0))
            .expect("Failed to apply Hadamard gate");

        let stabs = sim.get_stabilizers();
        // After H(0), stabiliser 0 was Z_0 → X_0
        assert_eq!(stabs[0], "+XIIII");
    }

    #[test]
    fn test_sparse_cnot_chain() {
        let mut sim = SparseCliffordSimulator::new(10).expect("Failed to create simulator");

        // Create a chain of CNOTs
        for i in 0..9 {
            sim.apply_gate(CliffordGate::CNOT(i, i + 1))
                .expect("Failed to apply CNOT gate");
        }

        let (stab_sparsity, _) = sim.get_sparsity_info();
        // Should still be relatively sparse
        assert!(
            stab_sparsity > 0.8,
            "CNOT chain made tableau too dense: {}",
            stab_sparsity
        );
    }

    #[test]
    fn test_s_gate() {
        let mut sim = SparseCliffordSimulator::new(3).expect("Failed to create simulator");
        // H(0) turns Z_0 stabiliser into X_0
        sim.apply_gate(CliffordGate::H(0))
            .expect("Failed to apply H");
        // S(0): X → Y
        sim.apply_gate(CliffordGate::S(0))
            .expect("Failed to apply S");

        let stabs = sim.get_stabilizers();
        // Stabiliser 0 should now be +Y_0 I I
        assert_eq!(stabs[0], "+YIII");
    }
}
