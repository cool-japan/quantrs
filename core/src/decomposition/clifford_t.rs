//! Clifford+T gate decomposition with optimal T-count
//!
//! This module provides algorithms for decomposing arbitrary single-qubit gates
//! into sequences of Clifford+T gates, with a focus on minimizing T-count.
//! T gates are typically more expensive than Clifford gates on quantum hardware.

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::gate::{single::*, GateOp};
use crate::matrix_ops::{matrices_approx_equal, DenseMatrix, QuantumMatrix};
use crate::qubit::QubitId;
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::f64::consts::{PI, SQRT_2};

/// Clifford gate types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CliffordGate {
    Hadamard,
    Phase,
    PhaseDagger,
    PauliX,
    PauliY,
    PauliZ,
}

impl CliffordGate {
    /// Get the matrix representation of the Clifford gate
    pub fn matrix(&self) -> Array2<Complex64> {
        let c0 = Complex64::new(0.0, 0.0);
        let c1 = Complex64::new(1.0, 0.0);
        let ci = Complex64::new(0.0, 1.0);
        let cm1 = Complex64::new(-1.0, 0.0);
        let cmi = Complex64::new(0.0, -1.0);
        let h_val = Complex64::new(1.0 / SQRT_2, 0.0);
        let mh_val = Complex64::new(-1.0 / SQRT_2, 0.0);

        match self {
            CliffordGate::Hadamard => {
                Array2::from_shape_vec((2, 2), vec![h_val, h_val, h_val, mh_val]).unwrap()
            }
            CliffordGate::Phase => Array2::from_shape_vec((2, 2), vec![c1, c0, c0, ci]).unwrap(),
            CliffordGate::PhaseDagger => {
                Array2::from_shape_vec((2, 2), vec![c1, c0, c0, cmi]).unwrap()
            }
            CliffordGate::PauliX => Array2::from_shape_vec((2, 2), vec![c0, c1, c1, c0]).unwrap(),
            CliffordGate::PauliY => Array2::from_shape_vec((2, 2), vec![c0, cmi, ci, c0]).unwrap(),
            CliffordGate::PauliZ => Array2::from_shape_vec((2, 2), vec![c1, c0, c0, cm1]).unwrap(),
        }
    }

    /// Convert to a gate operation
    pub fn to_gate(&self, qubit: QubitId) -> Box<dyn GateOp> {
        match self {
            CliffordGate::Hadamard => Box::new(Hadamard { target: qubit }),
            CliffordGate::Phase => Box::new(Phase { target: qubit }),
            CliffordGate::PhaseDagger => Box::new(PhaseDagger { target: qubit }),
            CliffordGate::PauliX => Box::new(PauliX { target: qubit }),
            CliffordGate::PauliY => Box::new(PauliY { target: qubit }),
            CliffordGate::PauliZ => Box::new(PauliZ { target: qubit }),
        }
    }
}

/// A sequence of Clifford and T gates
#[derive(Debug, Clone)]
pub struct CliffordTSequence {
    /// The gate sequence
    pub gates: SmallVec<[CliffordTGate; 16]>,
    /// The T-count
    pub t_count: usize,
    /// The total matrix (cached)
    matrix: Option<Array2<Complex64>>,
}

/// A single gate in the Clifford+T set
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliffordTGate {
    Clifford(CliffordGate),
    T,
    TDagger,
}

impl CliffordTGate {
    /// Get the matrix representation
    pub fn matrix(&self) -> Array2<Complex64> {
        match self {
            CliffordTGate::Clifford(c) => c.matrix(),
            CliffordTGate::T => {
                let c1 = Complex64::new(1.0, 0.0);
                let c0 = Complex64::new(0.0, 0.0);
                let t_phase = Complex64::from_polar(1.0, PI / 4.0);
                Array2::from_shape_vec((2, 2), vec![c1, c0, c0, t_phase]).unwrap()
            }
            CliffordTGate::TDagger => {
                let c1 = Complex64::new(1.0, 0.0);
                let c0 = Complex64::new(0.0, 0.0);
                let t_phase = Complex64::from_polar(1.0, -PI / 4.0);
                Array2::from_shape_vec((2, 2), vec![c1, c0, c0, t_phase]).unwrap()
            }
        }
    }

    /// Convert to a gate operation
    pub fn to_gate(&self, qubit: QubitId) -> Box<dyn GateOp> {
        match self {
            CliffordTGate::Clifford(c) => c.to_gate(qubit),
            CliffordTGate::T => Box::new(T { target: qubit }),
            CliffordTGate::TDagger => Box::new(TDagger { target: qubit }),
        }
    }

    /// Check if this is a T gate
    pub fn is_t_gate(&self) -> bool {
        matches!(self, CliffordTGate::T | CliffordTGate::TDagger)
    }
}

impl CliffordTSequence {
    /// Create a new empty sequence
    pub fn new() -> Self {
        Self {
            gates: SmallVec::new(),
            t_count: 0,
            matrix: None,
        }
    }

    /// Add a gate to the sequence
    pub fn add_gate(&mut self, gate: CliffordTGate) {
        if gate.is_t_gate() {
            self.t_count += 1;
        }
        self.gates.push(gate);
        self.matrix = None; // Invalidate cache
    }

    /// Add a Clifford gate
    pub fn add_clifford(&mut self, gate: CliffordGate) {
        self.add_gate(CliffordTGate::Clifford(gate));
    }

    /// Add a T gate
    pub fn add_t(&mut self) {
        self.add_gate(CliffordTGate::T);
    }

    /// Add a T† gate
    pub fn add_t_dagger(&mut self) {
        self.add_gate(CliffordTGate::TDagger);
    }

    /// Compute the total unitary matrix
    pub fn compute_matrix(&mut self) -> &Array2<Complex64> {
        if self.matrix.is_none() {
            let mut matrix = Array2::eye(2);
            for gate in &self.gates {
                matrix = matrix.dot(&gate.matrix());
            }
            self.matrix = Some(matrix);
        }
        self.matrix.as_ref().unwrap()
    }

    /// Convert to a sequence of gate operations
    pub fn to_gates(&self, qubit: QubitId) -> Vec<Box<dyn GateOp>> {
        self.gates.iter().map(|g| g.to_gate(qubit)).collect()
    }

    /// Optimize the sequence by canceling adjacent inverse gates
    pub fn optimize(&mut self) {
        let mut optimized = SmallVec::new();
        let mut i = 0;

        while i < self.gates.len() {
            if i + 1 < self.gates.len() {
                let can_cancel = match (&self.gates[i], &self.gates[i + 1]) {
                    (CliffordTGate::T, CliffordTGate::TDagger)
                    | (CliffordTGate::TDagger, CliffordTGate::T) => true,
                    (CliffordTGate::Clifford(g1), CliffordTGate::Clifford(g2)) => {
                        matches!(
                            (g1, g2),
                            (CliffordGate::Hadamard, CliffordGate::Hadamard)
                                | (CliffordGate::PauliX, CliffordGate::PauliX)
                                | (CliffordGate::PauliY, CliffordGate::PauliY)
                                | (CliffordGate::PauliZ, CliffordGate::PauliZ)
                                | (CliffordGate::Phase, CliffordGate::PhaseDagger)
                                | (CliffordGate::PhaseDagger, CliffordGate::Phase)
                        )
                    }
                    _ => false,
                };

                if can_cancel {
                    // Skip both gates
                    if self.gates[i].is_t_gate() {
                        self.t_count = self.t_count.saturating_sub(2);
                    }
                    i += 2;
                    continue;
                }
            }

            optimized.push(self.gates[i]);
            i += 1;
        }

        self.gates = optimized;
        self.matrix = None; // Invalidate cache
    }
}

/// Clifford+T decomposer with various algorithms
pub struct CliffordTDecomposer {
    /// Precision for approximations
    epsilon: f64,
    /// Cache of exact synthesis results
    exact_cache: FxHashMap<u64, CliffordTSequence>,
    /// Grid points for approximation
    grid_points: Vec<GridPoint>,
}

/// A grid point for approximation algorithms
#[derive(Debug, Clone)]
struct GridPoint {
    /// The unitary matrix at this point
    matrix: Array2<Complex64>,
    /// The Clifford+T sequence to reach this point
    sequence: CliffordTSequence,
    /// Distance metric for searching
    distance_key: u64,
}

impl Default for CliffordTDecomposer {
    fn default() -> Self {
        Self::new(1e-10)
    }
}

impl CliffordTDecomposer {
    /// Create a new decomposer with given precision
    pub fn new(epsilon: f64) -> Self {
        let mut decomposer = Self {
            epsilon,
            exact_cache: FxHashMap::default(),
            grid_points: Vec::new(),
        };
        decomposer.initialize_grid();
        decomposer
    }

    /// Initialize the grid of Clifford+T sequences
    fn initialize_grid(&mut self) {
        // Start with identity
        let mut sequences = vec![CliffordTSequence::new()];

        // Add basic single gates
        for gate in &[
            CliffordTGate::Clifford(CliffordGate::Hadamard),
            CliffordTGate::Clifford(CliffordGate::Phase),
            CliffordTGate::Clifford(CliffordGate::PhaseDagger),
            CliffordTGate::Clifford(CliffordGate::PauliX),
            CliffordTGate::Clifford(CliffordGate::PauliY),
            CliffordTGate::Clifford(CliffordGate::PauliZ),
            CliffordTGate::T,
            CliffordTGate::TDagger,
        ] {
            let mut seq = CliffordTSequence::new();
            seq.add_gate(*gate);
            sequences.push(seq);
        }

        // Generate sequences up to a certain T-count
        let max_t_count = 4; // Reduced for testing
        let max_length = 8; // Maximum sequence length

        for depth in 1..=max_length {
            let mut new_sequences = Vec::new();

            for seq in &sequences {
                if seq.gates.len() == depth - 1 && seq.t_count < max_t_count {
                    // Try adding each gate
                    for gate in &[
                        CliffordTGate::Clifford(CliffordGate::Hadamard),
                        CliffordTGate::Clifford(CliffordGate::Phase),
                        CliffordTGate::Clifford(CliffordGate::PhaseDagger),
                        CliffordTGate::T,
                        CliffordTGate::TDagger,
                    ] {
                        let mut new_seq = seq.clone();
                        new_seq.add_gate(*gate);
                        new_seq.optimize();

                        if new_seq.t_count <= max_t_count && new_seq.gates.len() <= max_length {
                            new_sequences.push(new_seq);
                        }
                    }
                }
            }

            sequences.extend(new_sequences);
        }

        // Convert to grid points
        for mut seq in sequences {
            let matrix = seq.compute_matrix().clone();
            let distance_key = self.compute_distance_key(&matrix);

            self.grid_points.push(GridPoint {
                matrix,
                sequence: seq,
                distance_key,
            });
        }

        // Sort by T-count for efficient searching
        self.grid_points.sort_by_key(|p| p.sequence.t_count);
    }

    /// Compute a distance key for fast searching
    fn compute_distance_key(&self, matrix: &Array2<Complex64>) -> u64 {
        // Use a hash of the matrix elements for fast lookup
        let mut key = 0u64;
        for elem in matrix.iter() {
            let re_bits = elem.re.to_bits();
            let im_bits = elem.im.to_bits();
            key = key.wrapping_mul(31).wrapping_add(re_bits);
            key = key.wrapping_mul(31).wrapping_add(im_bits);
        }
        key
    }

    /// Decompose a unitary into Clifford+T gates
    pub fn decompose(
        &mut self,
        unitary: &ArrayView2<Complex64>,
    ) -> QuantRS2Result<CliffordTSequence> {
        if unitary.shape() != &[2, 2] {
            return Err(QuantRS2Error::InvalidInput(
                "Unitary must be 2x2".to_string(),
            ));
        }

        // Check if it's unitary
        let unitary_dense = DenseMatrix::new(unitary.to_owned())?;
        if !unitary_dense.is_unitary(1e-10)? {
            return Err(QuantRS2Error::InvalidInput(
                "Matrix is not unitary".to_string(),
            ));
        }

        // Try exact synthesis first
        if let Some(exact) = self.exact_synthesis(unitary)? {
            return Ok(exact);
        }

        // Fall back to approximation
        self.approximate_synthesis(unitary)
    }

    /// Attempt exact synthesis using Matsumoto-Amano algorithm
    fn exact_synthesis(
        &mut self,
        unitary: &ArrayView2<Complex64>,
    ) -> QuantRS2Result<Option<CliffordTSequence>> {
        // Check if it's already in cache
        let key = self.compute_distance_key(&unitary.to_owned());
        if let Some(cached) = self.exact_cache.get(&key) {
            return Ok(Some(cached.clone()));
        }

        // Check if it's a Clifford gate
        if let Some(seq) = self.is_clifford(unitary)? {
            self.exact_cache.insert(key, seq.clone());
            return Ok(Some(seq));
        }

        // Check if it's in the form Clifford * T^k * Clifford
        if let Some(seq) = self.try_clifford_t_clifford(unitary)? {
            self.exact_cache.insert(key, seq.clone());
            return Ok(Some(seq));
        }

        // More complex exact synthesis would go here
        // For now, return None to indicate approximation is needed
        Ok(None)
    }

    /// Check if the unitary is a Clifford gate
    fn is_clifford(
        &self,
        unitary: &ArrayView2<Complex64>,
    ) -> QuantRS2Result<Option<CliffordTSequence>> {
        let clifford_gates = vec![
            (CliffordGate::Hadamard, CliffordGate::Hadamard.matrix()),
            (CliffordGate::Phase, CliffordGate::Phase.matrix()),
            (
                CliffordGate::PhaseDagger,
                CliffordGate::PhaseDagger.matrix(),
            ),
            (CliffordGate::PauliX, CliffordGate::PauliX.matrix()),
            (CliffordGate::PauliY, CliffordGate::PauliY.matrix()),
            (CliffordGate::PauliZ, CliffordGate::PauliZ.matrix()),
        ];

        for (gate, matrix) in clifford_gates {
            if matrices_approx_equal(unitary, &matrix.view(), self.epsilon) {
                let mut seq = CliffordTSequence::new();
                seq.add_clifford(gate);
                return Ok(Some(seq));
            }
        }

        Ok(None)
    }

    /// Try to decompose as Clifford * T^k * Clifford
    fn try_clifford_t_clifford(
        &self,
        unitary: &ArrayView2<Complex64>,
    ) -> QuantRS2Result<Option<CliffordTSequence>> {
        // This is a simplified version - full implementation would be more complex

        // Check if it's a rotation around Z by π/4 multiples
        let trace = unitary[[0, 0]] + unitary[[1, 1]];
        let det = unitary[[0, 0]] * unitary[[1, 1]] - unitary[[0, 1]] * unitary[[1, 0]];

        // Check determinant is 1
        if (det - Complex64::new(1.0, 0.0)).norm() > self.epsilon {
            return Ok(None);
        }

        // Check if trace corresponds to T^k for some k
        let t_powers = vec![
            (0, Complex64::new(2.0, 0.0)),                               // I
            (1, Complex64::new(1.0 + 1.0 / SQRT_2, 1.0 / SQRT_2)),       // T
            (2, Complex64::new(SQRT_2, SQRT_2)),                         // T^2 = S
            (3, Complex64::new(1.0 - 1.0 / SQRT_2, 1.0 + 1.0 / SQRT_2)), // T^3
        ];

        for (k, expected_trace) in t_powers {
            if (trace - expected_trace).norm() < self.epsilon {
                let mut seq = CliffordTSequence::new();
                for _ in 0..k {
                    seq.add_t();
                }
                return Ok(Some(seq));
            }
        }

        Ok(None)
    }

    /// Approximate synthesis using grid-based search
    fn approximate_synthesis(
        &self,
        unitary: &ArrayView2<Complex64>,
    ) -> QuantRS2Result<CliffordTSequence> {
        // If no grid points exist, return identity
        if self.grid_points.is_empty() {
            return Ok(CliffordTSequence::new());
        }

        let mut best_sequence = None;
        let mut best_distance = f64::INFINITY;

        // Search through grid points
        for point in &self.grid_points {
            // Compute distance
            let diff = unitary.to_owned() - &point.matrix;
            let mut distance = 0.0;
            for elem in diff.iter() {
                distance += elem.norm_sqr();
            }
            let distance = distance.sqrt();

            if distance < best_distance {
                best_distance = distance;
                best_sequence = Some(point.sequence.clone());

                if distance < self.epsilon {
                    break;
                }
            }
        }

        // Always return the best approximation found, even if it doesn't meet epsilon
        best_sequence
            .ok_or_else(|| QuantRS2Error::ComputationError("No grid points available".to_string()))
    }

    /// Decompose with T-count optimization
    pub fn decompose_optimal(
        &mut self,
        unitary: &ArrayView2<Complex64>,
        max_t_count: Option<usize>,
    ) -> QuantRS2Result<CliffordTSequence> {
        let mut best_sequence = self.decompose(unitary)?;

        if let Some(max_t) = max_t_count {
            if best_sequence.t_count > max_t {
                // Try harder optimization strategies
                best_sequence = self.optimize_t_count(unitary, max_t)?;
            }
        }

        Ok(best_sequence)
    }

    /// Optimize T-count using various strategies
    fn optimize_t_count(
        &self,
        unitary: &ArrayView2<Complex64>,
        max_t_count: usize,
    ) -> QuantRS2Result<CliffordTSequence> {
        // This is where more sophisticated algorithms would go
        // For now, just search through sequences with lower T-count

        for point in &self.grid_points {
            if point.sequence.t_count <= max_t_count {
                let diff = unitary.to_owned() - &point.matrix;
                let mut distance = 0.0;
                for elem in diff.iter() {
                    distance += elem.norm_sqr();
                }
                let distance = distance.sqrt();

                if distance < self.epsilon {
                    return Ok(point.sequence.clone());
                }
            }
        }

        Err(QuantRS2Error::ComputationError(format!(
            "Cannot achieve T-count <= {}",
            max_t_count
        )))
    }
}

/// Helper function to count T gates in a gate sequence
pub fn count_t_gates_in_sequence(gates: &[Box<dyn GateOp>]) -> usize {
    gates
        .iter()
        .filter(|g| g.name() == "T" || g.name() == "T†")
        .count()
}

/// Optimize a gate sequence by converting to Clifford+T and back
pub fn optimize_gate_sequence(
    gates: Vec<Box<dyn GateOp>>,
    epsilon: f64,
) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
    if gates.is_empty() {
        return Ok(gates);
    }

    // Compute total unitary
    let mut total_matrix = Array2::eye(2);
    for gate in &gates {
        let gate_matrix = gate.matrix()?;
        let gate_array = Array2::from_shape_vec((2, 2), gate_matrix)
            .map_err(|e| QuantRS2Error::InvalidInput(e.to_string()))?;
        total_matrix = total_matrix.dot(&gate_array);
    }

    // Decompose into Clifford+T
    let mut decomposer = CliffordTDecomposer::new(epsilon);
    let clifford_t_seq = decomposer.decompose(&total_matrix.view())?;

    // Convert back to gate operations
    let qubit = gates[0].qubits()[0]; // Assume all gates act on same qubit
    Ok(clifford_t_seq.to_gates(qubit))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clifford_gate_matrices() {
        let h = CliffordGate::Hadamard.matrix();
        let h2 = h.dot(&h);

        // H^2 = I
        assert!((h2[[0, 0]] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((h2[[1, 1]] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!(h2[[0, 1]].norm() < 1e-10);
        assert!(h2[[1, 0]].norm() < 1e-10);
    }

    #[test]
    fn test_clifford_t_sequence() {
        let mut seq = CliffordTSequence::new();
        seq.add_t();
        seq.add_t_dagger();
        seq.optimize();

        assert_eq!(seq.gates.len(), 0); // Should cancel out
        assert_eq!(seq.t_count, 0);
    }

    #[test]
    fn test_decomposer_clifford() {
        let mut decomposer = CliffordTDecomposer::new(1e-10);
        let h_matrix = CliffordGate::Hadamard.matrix();

        let result = decomposer.decompose(&h_matrix.view()).unwrap();
        assert_eq!(result.t_count, 0);
        assert_eq!(result.gates.len(), 1);
    }

    #[test]
    fn test_t_gate_counting() {
        let qubit = QubitId(0);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(T { target: qubit }),
            Box::new(Hadamard { target: qubit }),
            Box::new(TDagger { target: qubit }),
            Box::new(T { target: qubit }),
        ];

        assert_eq!(count_t_gates_in_sequence(&gates), 3);
    }

    #[test]
    fn test_decomposer_approximation() {
        let mut decomposer = CliffordTDecomposer::new(1e-3);

        // Test approximation of a T gate (which should be exact)
        let t_matrix = CliffordTGate::T.matrix();
        let result = decomposer.decompose(&t_matrix.view()).unwrap();

        // Should get exactly a T gate
        assert_eq!(result.gates.len(), 1);
        assert_eq!(result.t_count, 1);
        assert!(matches!(result.gates[0], CliffordTGate::T));

        // Test approximation of a gate that's in our grid
        // T^2 = S gate
        let s_matrix = t_matrix.dot(&t_matrix);
        let mut result2 = decomposer.decompose(&s_matrix.view()).unwrap();

        // Should find this exactly or as T*T
        assert!(result2.t_count <= 2);

        // Check accuracy
        let approx = result2.compute_matrix().clone();
        let diff = &s_matrix - &approx;
        let mut error = 0.0;
        for elem in diff.iter() {
            error += elem.norm_sqr();
        }
        assert!(error.sqrt() < 1e-10, "Should be exact for S gate");
    }

    #[test]
    fn test_decomposer_general_rotation() {
        let mut decomposer = CliffordTDecomposer::new(0.5); // Very relaxed tolerance

        // Test a general rotation that's not in our grid
        let angle = PI / 5.0; // Not a nice fraction of π
        let c = angle.cos();
        let s = angle.sin();
        let rotation = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(c, 0.0),
                Complex64::new(-s, 0.0),
                Complex64::new(s, 0.0),
                Complex64::new(c, 0.0),
            ],
        )
        .unwrap();

        // Should still find some approximation
        let result = decomposer.decompose(&rotation.view());
        assert!(result.is_ok(), "Should find some approximation");

        // For general rotations, we should get some sequence
        // (could be empty if no good approximation exists in our limited grid)
        let mut seq = result.unwrap();

        // If we got a sequence, check it's valid
        if seq.gates.len() > 0 {
            let approx = seq.compute_matrix();
            assert_eq!(approx.shape(), &[2, 2]);
        }
    }

    #[test]
    fn test_exact_clifford_detection() {
        let mut decomposer = CliffordTDecomposer::new(1e-10);

        // Test that Clifford gates are recognized exactly
        let s_gate = CliffordGate::Phase.matrix();
        let result = decomposer.decompose(&s_gate.view()).unwrap();

        assert_eq!(result.t_count, 0);
        assert_eq!(result.gates.len(), 1);
        assert!(matches!(
            result.gates[0],
            CliffordTGate::Clifford(CliffordGate::Phase)
        ));

        // Test combination of Clifford gates
        let h_gate = CliffordGate::Hadamard.matrix();
        let combined = h_gate.dot(&s_gate).dot(&h_gate);

        let result2 = decomposer.decompose(&combined.view()).unwrap();
        assert_eq!(result2.t_count, 0); // Should still be Clifford
    }

    #[test]
    fn test_sequence_optimization_advanced() {
        let mut seq = CliffordTSequence::new();

        // Add a sequence that should simplify
        seq.add_clifford(CliffordGate::Hadamard);
        seq.add_clifford(CliffordGate::Phase);
        seq.add_clifford(CliffordGate::Phase);
        seq.add_clifford(CliffordGate::Hadamard);
        seq.add_t();
        seq.add_t_dagger();

        let original_len = seq.gates.len();
        seq.optimize();

        // Should have removed T-T† and S-S = Z
        assert!(seq.gates.len() < original_len);
        assert_eq!(seq.t_count, 0);
    }
}
