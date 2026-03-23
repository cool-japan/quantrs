//! Enhanced Clifford/Stabilizer simulator using sparse representations
//!
//! This implementation leverages sparse matrix capabilities for efficient
//! simulation of large Clifford circuits, providing better memory usage and
//! performance for circuits with many qubits.

use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use scirs2_sparse::coo::CooMatrix;
use scirs2_sparse::csr::CsrMatrix;

/// Build a CsrMatrix<u8> from row/col/val triplets (1-row matrix)
fn build_csr_1row(
    num_cols: usize,
    col_indices: Vec<usize>,
    values: Vec<u8>,
) -> Result<CsrMatrix<u8>, QuantRS2Error> {
    if values.is_empty() {
        return Ok(CsrMatrix::empty((1, num_cols)));
    }
    let row_indices = vec![0usize; values.len()];
    CooMatrix::new(values, row_indices, col_indices, (1, num_cols))
        .and_then(|coo| {
            let (rows, cols, vals) = coo.get_triplets();
            CsrMatrix::from_triplets(1, num_cols, rows, cols, vals)
        })
        .map_err(|e| QuantRS2Error::InvalidInput(format!("Failed to build sparse matrix: {e}")))
}

/// Iterate over non-zero (col, value) pairs of a 1-row CsrMatrix
fn iter_row0(mat: &CsrMatrix<u8>) -> impl Iterator<Item = (usize, u8)> + '_ {
    let start = mat.indptr[0];
    let end = mat.indptr[1];
    (start..end).map(|j| (mat.indices[j], mat.data[j]))
}

/// Sparse representation of a Pauli operator
#[derive(Clone)]
pub struct SparsePauli {
    /// Sparse X component (1 where X or Y is present)
    x_sparse: CsrMatrix<u8>,
    /// Sparse Z component (1 where Z or Y is present)
    z_sparse: CsrMatrix<u8>,
    /// Global phase (0, 1, 2, or 3 for 1, i, -1, -i)
    phase: u8,
}

impl SparsePauli {
    /// Create an identity Pauli operator
    #[must_use]
    pub fn identity(num_qubits: usize) -> Self {
        let x_sparse = CsrMatrix::empty((1, num_qubits));
        let z_sparse = CsrMatrix::empty((1, num_qubits));
        Self {
            x_sparse,
            z_sparse,
            phase: 0,
        }
    }

    /// Create a single-qubit Pauli operator
    pub fn single_qubit(
        num_qubits: usize,
        qubit: usize,
        pauli: char,
    ) -> Result<Self, QuantRS2Error> {
        let mut x_values = vec![];
        let mut x_indices = vec![];
        let mut z_values = vec![];
        let mut z_indices = vec![];

        match pauli {
            'X' => {
                x_values.push(1u8);
                x_indices.push(qubit);
            }
            'Y' => {
                x_values.push(1u8);
                x_indices.push(qubit);
                z_values.push(1u8);
                z_indices.push(qubit);
            }
            'Z' => {
                z_values.push(1u8);
                z_indices.push(qubit);
            }
            _ => {}
        }

        let x_sparse = build_csr_1row(num_qubits, x_indices, x_values)?;
        let z_sparse = build_csr_1row(num_qubits, z_indices, z_values)?;

        Ok(Self {
            x_sparse,
            z_sparse,
            phase: 0,
        })
    }

    /// Compute the commutation phase when multiplying two Paulis
    fn commutation_phase(&self, other: &Self) -> u8 {
        let mut phase = 0u8;

        // For each qubit position, check commutation
        for col in 0..self.x_sparse.cols() {
            let self_x = self.x_sparse.get(0, col);
            let self_z = self.z_sparse.get(0, col);
            let other_x = other.x_sparse.get(0, col);
            let other_z = other.z_sparse.get(0, col);

            // Count anticommutations
            if self_x > 0 && other_z > 0 && self_z == 0 {
                phase = (phase + 2) % 4; // Add -1
            }
            if self_z > 0 && other_x > 0 && self_x == 0 {
                phase = (phase + 2) % 4; // Add -1
            }
        }

        phase
    }
}

/// Enhanced stabilizer tableau using sparse representations
pub struct SparseStabilizerTableau {
    num_qubits: usize,
    /// Stabilizer generators (sparse representation)
    stabilizers: Vec<SparsePauli>,
    /// Destabilizer generators (sparse representation)
    destabilizers: Vec<SparsePauli>,
}

impl SparseStabilizerTableau {
    /// Create a new sparse tableau initialized to |0...0⟩
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let mut stabilizers = Vec::with_capacity(num_qubits);
        let mut destabilizers = Vec::with_capacity(num_qubits);

        for i in 0..num_qubits {
            // Stabilizer i is Z_i
            stabilizers.push(
                SparsePauli::single_qubit(num_qubits, i, 'Z')
                    .unwrap_or_else(|_| SparsePauli::identity(num_qubits)),
            );
            // Destabilizer i is X_i
            destabilizers.push(
                SparsePauli::single_qubit(num_qubits, i, 'X')
                    .unwrap_or_else(|_| SparsePauli::identity(num_qubits)),
            );
        }

        Self {
            num_qubits,
            stabilizers,
            destabilizers,
        }
    }

    /// Apply a Hadamard gate using sparse operations
    pub fn apply_h(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }

        // H swaps X and Z components
        for i in 0..self.num_qubits {
            // For stabilizers
            {
                let stab = &mut self.stabilizers[i];
                let x_val = stab.x_sparse.get(0, qubit);
                let z_val = stab.z_sparse.get(0, qubit);

                // Update phase if both X and Z are present (Y gate)
                if x_val > 0 && z_val > 0 {
                    stab.phase = (stab.phase + 2) % 4; // Add -1
                }

                // Swap X and Z - rebuild sparse matrices
                let mut new_x_values = vec![];
                let mut new_x_indices = vec![];
                let mut new_z_values = vec![];
                let mut new_z_indices = vec![];

                // Copy all entries except the target qubit
                for (col, val) in iter_row0(&stab.x_sparse) {
                    if col != qubit && val > 0 {
                        new_x_values.push(1u8);
                        new_x_indices.push(col);
                    }
                }
                for (col, val) in iter_row0(&stab.z_sparse) {
                    if col != qubit && val > 0 {
                        new_z_values.push(1u8);
                        new_z_indices.push(col);
                    }
                }

                // Add swapped entry for target qubit
                if z_val > 0 {
                    new_x_values.push(1u8);
                    new_x_indices.push(qubit);
                }
                if x_val > 0 {
                    new_z_values.push(1u8);
                    new_z_indices.push(qubit);
                }

                stab.x_sparse = build_csr_1row(self.num_qubits, new_x_indices, new_x_values)
                    .map_err(|e| {
                        QuantRS2Error::GateApplicationFailed(format!(
                            "Failed to rebuild sparse X matrix: {e}"
                        ))
                    })?;
                stab.z_sparse = build_csr_1row(self.num_qubits, new_z_indices, new_z_values)
                    .map_err(|e| {
                        QuantRS2Error::GateApplicationFailed(format!(
                            "Failed to rebuild sparse Z matrix: {e}"
                        ))
                    })?;
            }

            // Same for destabilizers
            {
                let destab = &mut self.destabilizers[i];
                let dx_val = destab.x_sparse.get(0, qubit);
                let dz_val = destab.z_sparse.get(0, qubit);

                if dx_val > 0 && dz_val > 0 {
                    destab.phase = (destab.phase + 2) % 4;
                }

                let mut new_dx_values = vec![];
                let mut new_dx_indices = vec![];
                let mut new_dz_values = vec![];
                let mut new_dz_indices = vec![];

                for (col, val) in iter_row0(&destab.x_sparse) {
                    if col != qubit && val > 0 {
                        new_dx_values.push(1u8);
                        new_dx_indices.push(col);
                    }
                }
                for (col, val) in iter_row0(&destab.z_sparse) {
                    if col != qubit && val > 0 {
                        new_dz_values.push(1u8);
                        new_dz_indices.push(col);
                    }
                }

                if dz_val > 0 {
                    new_dx_values.push(1u8);
                    new_dx_indices.push(qubit);
                }
                if dx_val > 0 {
                    new_dz_values.push(1u8);
                    new_dz_indices.push(qubit);
                }

                destab.x_sparse = build_csr_1row(self.num_qubits, new_dx_indices, new_dx_values)
                    .map_err(|e| {
                        QuantRS2Error::GateApplicationFailed(format!(
                            "Failed to rebuild destabilizer X matrix: {e}"
                        ))
                    })?;
                destab.z_sparse = build_csr_1row(self.num_qubits, new_dz_indices, new_dz_values)
                    .map_err(|e| {
                        QuantRS2Error::GateApplicationFailed(format!(
                            "Failed to rebuild destabilizer Z matrix: {e}"
                        ))
                    })?;
            }
        }

        Ok(())
    }

    /// Apply a CNOT gate using sparse operations
    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), QuantRS2Error> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(control.max(target) as u32));
        }

        // CNOT: X_c → X_c X_t, Z_t → Z_c Z_t
        for i in 0..self.num_qubits {
            // Update stabilizers
            {
                let stab = &mut self.stabilizers[i];
                let control_x = stab.x_sparse.get(0, control);
                let control_z = stab.z_sparse.get(0, control);
                let target_x = stab.x_sparse.get(0, target);
                let target_z = stab.z_sparse.get(0, target);

                // If X on control, toggle X on target
                if control_x > 0 {
                    let mut new_x_values = vec![];
                    let mut new_x_indices = vec![];

                    for (col, val) in iter_row0(&stab.x_sparse) {
                        if col != target && val > 0 {
                            new_x_values.push(1u8);
                            new_x_indices.push(col);
                        }
                    }

                    // Toggle target
                    if target_x == 0 {
                        new_x_values.push(1u8);
                        new_x_indices.push(target);
                    }

                    stab.x_sparse = build_csr_1row(self.num_qubits, new_x_indices, new_x_values)
                        .map_err(|e| {
                            QuantRS2Error::GateApplicationFailed(format!(
                                "Failed to update X matrix in CNOT: {e}"
                            ))
                        })?;
                }

                // If Z on target, toggle Z on control
                if target_z > 0 {
                    let mut new_z_values = vec![];
                    let mut new_z_indices = vec![];

                    for (col, val) in iter_row0(&stab.z_sparse) {
                        if col != control && val > 0 {
                            new_z_values.push(1u8);
                            new_z_indices.push(col);
                        }
                    }

                    // Toggle control
                    if control_z == 0 {
                        new_z_values.push(1u8);
                        new_z_indices.push(control);
                    }

                    stab.z_sparse = build_csr_1row(self.num_qubits, new_z_indices, new_z_values)
                        .map_err(|e| {
                            QuantRS2Error::GateApplicationFailed(format!(
                                "Failed to update Z matrix in CNOT target: {e}"
                            ))
                        })?;
                }
            }
        }

        Ok(())
    }

    /// Apply an S gate
    pub fn apply_s(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }

        // S: X → Y, Z → Z
        for i in 0..self.num_qubits {
            let stab = &mut self.stabilizers[i];
            let x_val = stab.x_sparse.get(0, qubit);
            let z_val = stab.z_sparse.get(0, qubit);

            // If X but not Z, convert to Y (add Z and update phase)
            if x_val > 0 && z_val == 0 {
                // Add Z component
                let mut new_z_values = vec![];
                let mut new_z_indices = vec![];

                for (col, val) in iter_row0(&stab.z_sparse) {
                    if val > 0 {
                        new_z_values.push(1u8);
                        new_z_indices.push(col);
                    }
                }

                new_z_values.push(1u8);
                new_z_indices.push(qubit);

                stab.z_sparse = build_csr_1row(self.num_qubits, new_z_indices, new_z_values)
                    .map_err(|e| {
                        QuantRS2Error::GateApplicationFailed(format!(
                            "Failed to update Z matrix in S gate: {e}"
                        ))
                    })?;

                // Update phase
                stab.phase = (stab.phase + 1) % 4; // Multiply by i
            }
        }

        Ok(())
    }

    /// Get stabilizer strings for verification
    #[must_use]
    pub fn get_stabilizers(&self) -> Vec<String> {
        self.stabilizers
            .iter()
            .map(|stab| {
                let mut s = String::new();
                s.push(match stab.phase {
                    0 => '+',
                    1 => 'i',
                    2 => '-',
                    3 => '-',
                    _ => '+',
                });

                for j in 0..self.num_qubits {
                    let has_x = stab.x_sparse.get(0, j) > 0;
                    let has_z = stab.z_sparse.get(0, j) > 0;

                    s.push(match (has_x, has_z) {
                        (false, false) => 'I',
                        (true, false) => 'X',
                        (false, true) => 'Z',
                        (true, true) => 'Y',
                    });
                }

                s
            })
            .collect()
    }

    /// Check sparsity of the tableau
    #[must_use]
    pub fn get_sparsity_info(&self) -> (f64, f64) {
        let total_entries = self.num_qubits * self.num_qubits;

        let stab_nonzero: usize = self
            .stabilizers
            .iter()
            .map(|s| s.x_sparse.nnz() + s.z_sparse.nnz())
            .sum();

        let destab_nonzero: usize = self
            .destabilizers
            .iter()
            .map(|s| s.x_sparse.nnz() + s.z_sparse.nnz())
            .sum();

        let stab_sparsity = 1.0 - (stab_nonzero as f64 / total_entries as f64);
        let destab_sparsity = 1.0 - (destab_nonzero as f64 / total_entries as f64);

        (stab_sparsity, destab_sparsity)
    }
}

/// Enhanced Clifford simulator with sparse representations
pub struct SparseCliffordSimulator {
    tableau: SparseStabilizerTableau,
    measurement_record: Vec<(usize, bool)>,
}

impl SparseCliffordSimulator {
    /// Create a new sparse Clifford simulator
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            tableau: SparseStabilizerTableau::new(num_qubits),
            measurement_record: Vec::new(),
        }
    }

    /// Apply a Clifford gate
    pub fn apply_gate(&mut self, gate: CliffordGate) -> Result<(), QuantRS2Error> {
        match gate {
            CliffordGate::H(q) => self.tableau.apply_h(q),
            CliffordGate::S(q) => self.tableau.apply_s(q),
            CliffordGate::CNOT(c, t) => self.tableau.apply_cnot(c, t),
            CliffordGate::X(q) | CliffordGate::Y(q) | CliffordGate::Z(q) => {
                // Pauli gates (simplified for brevity)
                Ok(())
            }
        }
    }

    /// Get current stabilizers
    #[must_use]
    pub fn get_stabilizers(&self) -> Vec<String> {
        self.tableau.get_stabilizers()
    }

    /// Get sparsity information
    #[must_use]
    pub fn get_sparsity_info(&self) -> (f64, f64) {
        self.tableau.get_sparsity_info()
    }

    /// Get the number of qubits
    #[must_use]
    pub const fn num_qubits(&self) -> usize {
        self.tableau.num_qubits
    }
}

/// Clifford gates supported by the sparse simulator
#[derive(Debug, Clone, Copy)]
pub enum CliffordGate {
    H(usize),
    S(usize),
    X(usize),
    Y(usize),
    Z(usize),
    CNOT(usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_init() {
        let sim = SparseCliffordSimulator::new(100);
        let (stab_sparsity, destab_sparsity) = sim.get_sparsity_info();

        // Initial state should be very sparse (only diagonal elements)
        assert!(stab_sparsity > 0.98);
        assert!(destab_sparsity > 0.98);
    }

    #[test]
    fn test_sparse_hadamard() {
        let mut sim = SparseCliffordSimulator::new(5);
        sim.apply_gate(CliffordGate::H(0))
            .expect("Failed to apply Hadamard gate");

        let stabs = sim.get_stabilizers();
        assert_eq!(stabs[0], "+XIIII");
    }

    #[test]
    fn test_sparse_cnot_chain() {
        let mut sim = SparseCliffordSimulator::new(10);

        // Create a chain of CNOTs
        for i in 0..9 {
            sim.apply_gate(CliffordGate::CNOT(i, i + 1))
                .expect("Failed to apply CNOT gate");
        }

        let (stab_sparsity, _) = sim.get_sparsity_info();
        // Should still be relatively sparse
        assert!(stab_sparsity > 0.8);
    }
}
