//! Quantum Volume and Process Tomography
//!
//! This module implements quantum benchmarking and characterization protocols
//! for evaluating quantum computer performance.
//!
//! ## Quantum Volume
//! Quantum Volume (QV) is a holistic metric that captures the overall performance
//! of a quantum computer, taking into account:
//! - Number of qubits
//! - Gate fidelity
//! - Qubit connectivity
//! - Error rates
//! - Measurement quality
//!
//! ## Quantum Process Tomography
//! QPT completely characterizes a quantum operation by reconstructing its
//! process matrix (chi matrix) or Choi representation.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::ndarray::{Array1, Array2, Array3, Array4};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::collections::HashMap;

/// Quantum Volume Protocol
///
/// Measures the largest random square circuit (n×n) that can be executed
/// reliably on a quantum computer.
pub struct QuantumVolume {
    /// Maximum number of qubits to test
    pub max_qubits: usize,
    /// Number of random circuits per qubit count
    pub num_circuits: usize,
    /// Number of shots per circuit
    pub num_shots: usize,
    /// Success threshold (heavy output probability)
    pub success_threshold: f64,
    /// Random number generator
    rng: ThreadRng,
}

impl QuantumVolume {
    /// Create a new quantum volume protocol
    pub fn new(max_qubits: usize, num_circuits: usize, num_shots: usize) -> Self {
        Self {
            max_qubits,
            num_circuits,
            num_shots,
            success_threshold: 2.0 / 3.0, // Standard QV threshold
            rng: thread_rng(),
        }
    }

    /// Run quantum volume protocol
    ///
    /// Returns the achieved quantum volume (largest successful n)
    pub fn run<F>(&mut self, mut circuit_executor: F) -> QuantRS2Result<QuantumVolumeResult>
    where
        F: FnMut(&[Box<dyn GateOp>], usize) -> Vec<usize>, // Returns measured bitstrings
    {
        let mut results = HashMap::new();
        let mut quantum_volume = 1;

        for n_qubits in 1..=self.max_qubits {
            let success_rate = self.test_quantum_volume(n_qubits, &mut circuit_executor)?;

            results.insert(n_qubits, success_rate);

            // Check if QV is achieved for this qubit count
            if success_rate >= self.success_threshold {
                quantum_volume = 1 << n_qubits; // 2^n
            } else {
                break; // Stop at first failure
            }
        }

        Ok(QuantumVolumeResult {
            quantum_volume,
            success_rates: results,
            max_qubits_tested: self.max_qubits,
        })
    }

    /// Test quantum volume for a specific number of qubits
    fn test_quantum_volume<F>(
        &mut self,
        n_qubits: usize,
        circuit_executor: &mut F,
    ) -> QuantRS2Result<f64>
    where
        F: FnMut(&[Box<dyn GateOp>], usize) -> Vec<usize>,
    {
        let mut successful_circuits = 0;

        for _ in 0..self.num_circuits {
            // Generate random model circuit
            let (circuit, heavy_outputs) = self.generate_random_circuit(n_qubits)?;

            // Execute circuit and collect measurements
            let measurements = circuit_executor(&circuit, self.num_shots);

            // Calculate heavy output probability
            let hop = self.calculate_heavy_output_probability(&measurements, &heavy_outputs);

            // Check if circuit passed (HOP > 2/3)
            if hop > 2.0 / 3.0 {
                successful_circuits += 1;
            }
        }

        let success_rate = successful_circuits as f64 / self.num_circuits as f64;
        Ok(success_rate)
    }

    /// Generate a random model circuit for quantum volume.
    ///
    /// Builds the standard quantum-volume "square" circuit: `depth = n_qubits`
    /// layers, where each layer randomly permutes the qubits (Fisher-Yates),
    /// pairs them up, and applies a Haar-random 2-qubit unitary (an SU(4) element)
    /// to each pair. The circuit is then classically simulated to determine the
    /// heavy outputs.
    ///
    /// Returns the circuit (a non-empty list of gates) and the set of heavy
    /// outputs (computational-basis indices with strictly-above-median ideal
    /// probability).
    fn generate_random_circuit(
        &mut self,
        n_qubits: usize,
    ) -> QuantRS2Result<(Vec<Box<dyn GateOp>>, Vec<usize>)> {
        // For quantum volume, the model circuit depth equals the qubit count.
        let depth = n_qubits;
        let mut circuit: Vec<Box<dyn GateOp>> = Vec::new();

        for _layer in 0..depth {
            // Random permutation of the qubits, then pair adjacent entries.
            let mut order: Vec<usize> = (0..n_qubits).collect();
            self.shuffle(&mut order);

            let num_pairs = n_qubits / 2;
            for pair in 0..num_pairs {
                let q1 = order[2 * pair];
                let q2 = order[2 * pair + 1];
                let unitary = self.random_su4()?;
                circuit.push(Box::new(TwoQubitUnitaryGate::new(unitary, q1, q2)));
            }
        }

        // Classically simulate the ideal circuit to find heavy outputs.
        let heavy_outputs = self.find_heavy_outputs(n_qubits, &circuit)?;

        Ok((circuit, heavy_outputs))
    }

    /// Find heavy outputs: computational-basis states whose ideal probability is
    /// strictly above the median probability.
    ///
    /// The circuit is simulated to a full state vector starting from `|0...0>`,
    /// every `|amplitude|^2` probability is computed, the median probability is
    /// taken, and the indices with probability strictly greater than the median
    /// are returned. This is the genuine quantum-volume heavy-output definition.
    fn find_heavy_outputs(
        &self,
        n_qubits: usize,
        circuit: &[Box<dyn GateOp>],
    ) -> QuantRS2Result<Vec<usize>> {
        let num_states = 1usize << n_qubits;

        // Simulate the circuit to obtain the ideal probability distribution.
        let state = simulate_circuit(circuit, n_qubits)?;
        let probabilities: Vec<f64> = state.iter().map(|amp| amp.norm_sqr()).collect();

        // Median of the probability list.
        let mut sorted = probabilities.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = if num_states % 2 == 0 {
            0.5 * (sorted[num_states / 2 - 1] + sorted[num_states / 2])
        } else {
            sorted[num_states / 2]
        };

        // Indices strictly above the median probability.
        let heavy_outputs: Vec<usize> = probabilities
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > median)
            .map(|(idx, _)| idx)
            .collect();

        Ok(heavy_outputs)
    }

    /// Fisher-Yates shuffle using the protocol's RNG.
    fn shuffle(&mut self, slice: &mut [usize]) {
        let n = slice.len();
        if n < 2 {
            return;
        }
        for i in 0..n - 1 {
            let j = self.rng.random_range(i..n);
            slice.swap(i, j);
        }
    }

    /// Generate a Haar-random 4x4 unitary (an element of U(4), which contains the
    /// SU(4) gates used by the quantum-volume protocol).
    ///
    /// A matrix with i.i.d. complex-Gaussian entries is orthonormalised via the
    /// Gram-Schmidt process; the resulting unitary is Haar-distributed (up to the
    /// usual phase convention), giving a genuine random 2-qubit gate rather than a
    /// fixed or parameterised placeholder.
    fn random_su4(&mut self) -> QuantRS2Result<Array2<Complex64>> {
        let dim = 4;
        let mut matrix = Array2::<Complex64>::zeros((dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                let (re, im) = self.standard_normal_pair();
                matrix[[i, j]] = Complex64::new(re, im);
            }
        }
        gram_schmidt_unitary(&matrix)
    }

    /// Draw a pair of independent standard-normal samples via the Box-Muller
    /// transform, sourcing uniforms from the protocol's RNG.
    fn standard_normal_pair(&mut self) -> (f64, f64) {
        // Guard against log(0) by clamping u1 away from zero.
        let u1: f64 = self.rng.random_range(f64::EPSILON..1.0);
        let u2: f64 = self.rng.random_range(0.0..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }

    /// Calculate heavy output probability
    fn calculate_heavy_output_probability(
        &self,
        measurements: &[usize],
        heavy_outputs: &[usize],
    ) -> f64 {
        let heavy_count = measurements
            .iter()
            .filter(|&&bitstring| heavy_outputs.contains(&bitstring))
            .count();

        heavy_count as f64 / measurements.len() as f64
    }
}

/// A general 2-qubit unitary gate wrapping an arbitrary 4x4 unitary matrix.
///
/// The matrix is stored in row-major order over the local 2-qubit basis
/// `{|q1 q2>}` with `q1` the high-order local bit, consistent with the row-major
/// `matrix()` convention used by the gates in [`crate::gate::functions`].
#[derive(Debug, Clone)]
struct TwoQubitUnitaryGate {
    matrix: Array2<Complex64>,
    qubit1: QubitId,
    qubit2: QubitId,
}

impl TwoQubitUnitaryGate {
    fn new(matrix: Array2<Complex64>, qubit1: usize, qubit2: usize) -> Self {
        Self {
            matrix,
            qubit1: QubitId::new(qubit1 as u32),
            qubit2: QubitId::new(qubit2 as u32),
        }
    }
}

impl GateOp for TwoQubitUnitaryGate {
    fn name(&self) -> &'static str {
        "QV_SU4"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.qubit1, self.qubit2]
    }

    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        // Row-major flatten.
        let (rows, cols) = self.matrix.dim();
        let mut flat = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                flat.push(self.matrix[[i, j]]);
            }
        }
        Ok(flat)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(self.clone())
    }
}

/// Gram-Schmidt orthonormalisation of the columns of `matrix`, yielding a unitary.
fn gram_schmidt_unitary(matrix: &Array2<Complex64>) -> QuantRS2Result<Array2<Complex64>> {
    let dim = matrix.nrows();
    let mut result = Array2::<Complex64>::zeros((dim, dim));

    for j in 0..dim {
        let mut col = matrix.column(j).to_owned();

        // Subtract projections onto previously-computed orthonormal columns.
        for k in 0..j {
            let prev = result.column(k);
            let proj: Complex64 = col.iter().zip(prev.iter()).map(|(a, b)| b.conj() * a).sum();
            for i in 0..dim {
                col[i] -= proj * prev[i];
            }
        }

        let norm = col.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if norm < 1e-12 {
            return Err(QuantRS2Error::ComputationError(
                "Gram-Schmidt failed: degenerate random matrix".to_string(),
            ));
        }
        for i in 0..dim {
            result[[i, j]] = col[i] / Complex64::new(norm, 0.0);
        }
    }

    Ok(result)
}

/// Classically simulate a gate circuit on `n_qubits` qubits starting from
/// `|0...0>`, returning the final state vector.
///
/// Each gate's `matrix()` (row-major over its local basis, with `qubits()[0]` the
/// high-order local bit) is applied to the relevant amplitude tuples via direct
/// bit-mask indexing — the standard state-vector update.
fn simulate_circuit(
    circuit: &[Box<dyn GateOp>],
    n_qubits: usize,
) -> QuantRS2Result<Array1<Complex64>> {
    let dim = 1usize << n_qubits;
    let mut state = Array1::<Complex64>::zeros(dim);
    state[0] = Complex64::new(1.0, 0.0);

    for gate in circuit {
        apply_gate(&mut state, gate.as_ref(), n_qubits)?;
    }

    Ok(state)
}

/// Apply a single (1- or multi-qubit) gate to the state vector in place.
fn apply_gate(
    state: &mut Array1<Complex64>,
    gate: &dyn GateOp,
    n_qubits: usize,
) -> QuantRS2Result<()> {
    let qubits = gate.qubits();
    let k = qubits.len();
    let gate_dim = 1usize << k;

    let flat = gate.matrix()?;
    if flat.len() != gate_dim * gate_dim {
        return Err(QuantRS2Error::InvalidInput(format!(
            "Gate matrix has {} entries, expected {} for {}-qubit gate",
            flat.len(),
            gate_dim * gate_dim,
            k
        )));
    }
    // Reshape row-major flat matrix into a 2D view.
    let gate_matrix = Array2::from_shape_vec((gate_dim, gate_dim), flat)
        .map_err(|e| QuantRS2Error::ComputationError(format!("Gate reshape failed: {e}")))?;

    // Global bit positions for the gate's local bits. Local bit 0 (most
    // significant in the gate basis) corresponds to qubits[0].
    let qubit_bits: Vec<usize> = qubits.iter().map(|q| q.id() as usize).collect();
    for &b in &qubit_bits {
        if b >= n_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Gate acts on qubit {b} but circuit has only {n_qubits} qubits"
            )));
        }
    }

    let dim = 1usize << n_qubits;
    // Iterate over all "base" indices where the gate's qubits are 0, then update
    // the 2^k amplitudes for each combination of the gate's local bits.
    let mut visited = vec![false; dim];
    for base in 0..dim {
        // Skip indices that set any of the gate qubits (we enumerate those via
        // the local-combination loop below) and any already-processed group.
        if visited[base] {
            continue;
        }
        let mut anchor = base;
        for &b in &qubit_bits {
            anchor &= !(1 << b);
        }
        if anchor != base {
            continue;
        }

        // Gather the 2^k amplitudes of this group.
        let mut indices = vec![0usize; gate_dim];
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); gate_dim];
        for local in 0..gate_dim {
            let mut idx = anchor;
            for (pos, &b) in qubit_bits.iter().enumerate() {
                // Local bit `pos` is bit `(k - 1 - pos)` of `local` so that
                // qubits[0] is the most-significant local bit.
                let bit = (local >> (k - 1 - pos)) & 1;
                if bit == 1 {
                    idx |= 1 << b;
                }
            }
            indices[local] = idx;
            amplitudes[local] = state[idx];
            visited[idx] = true;
        }

        // Apply the gate matrix: new[r] = Σ_c M[r,c] * old[c].
        for r in 0..gate_dim {
            let mut acc = Complex64::new(0.0, 0.0);
            for c in 0..gate_dim {
                acc += gate_matrix[[r, c]] * amplitudes[c];
            }
            state[indices[r]] = acc;
        }
    }

    Ok(())
}

/// Result of quantum volume protocol
#[derive(Debug, Clone)]
pub struct QuantumVolumeResult {
    /// Achieved quantum volume (2^n)
    pub quantum_volume: usize,
    /// Success rates for each qubit count tested
    pub success_rates: HashMap<usize, f64>,
    /// Maximum number of qubits tested
    pub max_qubits_tested: usize,
}

impl QuantumVolumeResult {
    /// Get the number of qubits achieved
    pub fn num_qubits_achieved(&self) -> usize {
        (self.quantum_volume as f64).log2() as usize
    }

    /// Check if quantum volume was achieved for n qubits
    pub fn is_qv_achieved(&self, n_qubits: usize) -> bool {
        self.success_rates
            .get(&n_qubits)
            .is_some_and(|&rate| rate >= 2.0 / 3.0)
    }
}

/// Quantum Process Tomography Protocol
///
/// Completely characterizes a quantum operation by measuring its action
/// on a complete set of input states.
pub struct QuantumProcessTomography {
    /// Number of qubits in the process
    pub num_qubits: usize,
    /// Basis for state preparation (typically Pauli basis)
    pub preparation_basis: Vec<String>,
    /// Basis for measurement (typically Pauli basis)
    pub measurement_basis: Vec<String>,
}

impl QuantumProcessTomography {
    /// Create a new QPT protocol
    pub fn new(num_qubits: usize) -> Self {
        // Generate Pauli basis for preparation and measurement
        let basis = Self::generate_pauli_basis(num_qubits);

        Self {
            num_qubits,
            preparation_basis: basis.clone(),
            measurement_basis: basis,
        }
    }

    /// Generate Pauli basis strings for n qubits
    fn generate_pauli_basis(n_qubits: usize) -> Vec<String> {
        let paulis = ['I', 'X', 'Y', 'Z'];
        let basis_size = 4_usize.pow(n_qubits as u32);

        let mut basis = Vec::with_capacity(basis_size);

        for i in 0..basis_size {
            let mut pauli_string = String::with_capacity(n_qubits);
            let mut idx = i;

            for _ in 0..n_qubits {
                pauli_string.push(paulis[idx % 4]);
                idx /= 4;
            }

            basis.push(pauli_string);
        }

        basis
    }

    /// Run quantum process tomography
    ///
    /// Returns the reconstructed process matrix (chi matrix)
    pub fn run<F>(&self, mut apply_process: F) -> QuantRS2Result<ProcessMatrix>
    where
        F: FnMut(&str, &str) -> Complex64, // (prep_basis, meas_basis) -> expectation value
    {
        let dim = 1 << self.num_qubits;
        let basis_size = self.preparation_basis.len();

        // Allocate chi matrix
        let mut chi_matrix = Array2::zeros((basis_size, basis_size));

        // Perform tomography: measure E[P_out | P_in] for all Pauli pairs
        for (i, prep) in self.preparation_basis.iter().enumerate() {
            for (j, meas) in self.measurement_basis.iter().enumerate() {
                let expectation = apply_process(prep, meas);
                chi_matrix[[i, j]] = expectation;
            }
        }

        // Post-process to enforce physicality (positive semidefinite, trace-preserving)
        let chi_matrix = self.enforce_physicality(chi_matrix)?;

        Ok(ProcessMatrix {
            chi_matrix,
            num_qubits: self.num_qubits,
            basis_labels: self.preparation_basis.clone(),
        })
    }

    /// Enforce physicality constraints on the process matrix
    fn enforce_physicality(&self, chi: Array2<Complex64>) -> QuantRS2Result<Array2<Complex64>> {
        // Simplified physicality enforcement
        // In practice, this would use:
        // 1. Maximum likelihood estimation
        // 2. Projection onto physical process matrices
        // 3. Constrained optimization

        // For now, just normalize
        let trace: Complex64 = chi.diag().iter().sum();
        let normalized = if trace.norm() > 1e-10 {
            &chi / trace
        } else {
            chi
        };

        Ok(normalized)
    }

    /// Compute process fidelity between two process matrices
    pub fn process_fidelity(chi1: &Array2<Complex64>, chi2: &Array2<Complex64>) -> f64 {
        // F_proc = Tr(chi1^† chi2)
        let product = chi1.t().mapv(|x| x.conj()).dot(chi2);
        let trace: Complex64 = product.diag().iter().sum();
        trace.norm()
    }

    /// Compute average gate fidelity from process matrix
    pub fn average_gate_fidelity(
        &self,
        chi: &Array2<Complex64>,
        ideal_chi: &Array2<Complex64>,
    ) -> f64 {
        let dim = 1 << self.num_qubits;
        let d = dim as f64;

        // F_avg = (d * F_proc + 1) / (d + 1)
        let f_proc = Self::process_fidelity(chi, ideal_chi);
        (d * f_proc + 1.0) / (d + 1.0)
    }
}

/// Reconstructed process matrix from QPT
#[derive(Debug, Clone)]
pub struct ProcessMatrix {
    /// Chi matrix in Pauli basis
    pub chi_matrix: Array2<Complex64>,
    /// Number of qubits
    pub num_qubits: usize,
    /// Basis labels
    pub basis_labels: Vec<String>,
}

impl ProcessMatrix {
    /// Get the process matrix element for specific Pauli operators
    pub fn get_element(&self, prep_pauli: &str, meas_pauli: &str) -> Option<Complex64> {
        let i = self.basis_labels.iter().position(|s| s == prep_pauli)?;
        let j = self.basis_labels.iter().position(|s| s == meas_pauli)?;
        Some(self.chi_matrix[[i, j]])
    }

    /// Check if the process is trace-preserving
    pub fn is_trace_preserving(&self, tolerance: f64) -> bool {
        let trace: Complex64 = self.chi_matrix.diag().iter().sum();
        (trace - Complex64::new(1.0, 0.0)).norm() < tolerance
    }

    /// Check if the process is completely positive
    pub fn is_completely_positive(&self, tolerance: f64) -> bool {
        // Simplified check: chi should be positive semidefinite
        // In practice, would compute eigenvalues

        // For now, check diagonal elements are non-negative
        self.chi_matrix.diag().iter().all(|&x| x.re >= -tolerance)
    }

    /// Compute the diamond norm distance to another process
    pub fn diamond_distance(&self, other: &Self) -> QuantRS2Result<f64> {
        if self.num_qubits != other.num_qubits {
            return Err(QuantRS2Error::InvalidInput(
                "Process matrices must have same dimension".to_string(),
            ));
        }

        // Simplified diamond distance computation
        // Full implementation requires semidefinite programming

        // Approximate using Frobenius norm
        let diff = &self.chi_matrix - &other.chi_matrix;
        let frobenius_norm = diff.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();

        Ok(frobenius_norm)
    }
}

/// Gate Set Tomography (GST)
///
/// More comprehensive than QPT, GST characterizes an entire gate set
/// including state preparation and measurement errors.
pub struct GateSetTomography {
    /// Number of qubits
    pub num_qubits: usize,
    /// Gate set to characterize
    pub gate_set: Vec<String>,
    /// Maximum sequence length
    pub max_length: usize,
}

impl GateSetTomography {
    /// Create a new GST protocol
    pub const fn new(num_qubits: usize, gate_set: Vec<String>, max_length: usize) -> Self {
        Self {
            num_qubits,
            gate_set,
            max_length,
        }
    }

    /// Run gate set tomography
    ///
    /// This is a placeholder for the full GST algorithm
    pub fn run<F>(&self, mut execute_sequence: F) -> QuantRS2Result<GateSetModel>
    where
        F: FnMut(&[&str]) -> f64, // Gate sequence -> measurement probability
    {
        // GST consists of three types of sequences:
        // 1. Germ sequences (repeated short sequences)
        // 2. Fiducial sequences (state prep and measurement)
        // 3. Amplification sequences (repeated germs)

        let germs = self.generate_germs();
        let fiducials = self.generate_fiducials();

        // Collect data from all sequences
        let mut data = HashMap::new();

        for prep_fiducial in &fiducials {
            for germ in &germs {
                for meas_fiducial in &fiducials {
                    // Build amplified sequence
                    for power in 1..=self.max_length {
                        let mut sequence = Vec::new();

                        // Prep fiducial
                        sequence.extend_from_slice(prep_fiducial);

                        // Repeated germ
                        for _ in 0..power {
                            sequence.extend_from_slice(germ);
                        }

                        // Measurement fiducial
                        sequence.extend_from_slice(meas_fiducial);

                        // Execute and collect data
                        let probability = execute_sequence(&sequence);
                        data.insert(sequence.clone(), probability);
                    }
                }
            }
        }

        // Fit model to data using maximum likelihood estimation
        let model = self.fit_model(&data)?;

        Ok(model)
    }

    /// Generate germ sequences
    fn generate_germs(&self) -> Vec<Vec<&str>> {
        // Standard germs for single qubit: I, X, Y, XY, XYX
        // This is a simplified set
        vec![vec!["I"], vec!["X"], vec!["Y"], vec!["X", "Y"]]
    }

    /// Generate fiducial sequences
    fn generate_fiducials(&self) -> Vec<Vec<&str>> {
        // Standard fiducials for single qubit
        vec![
            vec!["I"],
            vec!["X"],
            vec!["Y"],
            vec!["X", "X"], // -I
        ]
    }

    /// Fit GST model to data
    fn fit_model(&self, _data: &HashMap<Vec<&str>, f64>) -> QuantRS2Result<GateSetModel> {
        // Placeholder: maximum likelihood estimation
        // Real implementation would use iterative optimization

        Ok(GateSetModel {
            num_qubits: self.num_qubits,
            gate_errors: HashMap::new(),
            spam_errors: vec![],
        })
    }
}

/// GST model describing errors in gates and measurements
#[derive(Debug, Clone)]
pub struct GateSetModel {
    /// Number of qubits
    pub num_qubits: usize,
    /// Error models for each gate
    pub gate_errors: HashMap<String, Array2<Complex64>>,
    /// State preparation and measurement (SPAM) errors
    pub spam_errors: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_volume_result() {
        let mut result = QuantumVolumeResult {
            quantum_volume: 16,
            success_rates: HashMap::new(),
            max_qubits_tested: 5,
        };

        result.success_rates.insert(1, 0.95);
        result.success_rates.insert(2, 0.85);
        result.success_rates.insert(3, 0.75);
        result.success_rates.insert(4, 0.70);

        assert_eq!(result.num_qubits_achieved(), 4);
        assert!(result.is_qv_achieved(1));
        assert!(result.is_qv_achieved(2));
        assert!(result.is_qv_achieved(3));
        assert!(result.is_qv_achieved(4));

        println!("Quantum Volume: {}", result.quantum_volume);
    }

    #[test]
    fn test_pauli_basis_generation() {
        let basis = QuantumProcessTomography::generate_pauli_basis(1);
        assert_eq!(basis.len(), 4);
        assert!(basis.contains(&"I".to_string()));
        assert!(basis.contains(&"X".to_string()));
        assert!(basis.contains(&"Y".to_string()));
        assert!(basis.contains(&"Z".to_string()));

        let basis_2q = QuantumProcessTomography::generate_pauli_basis(2);
        assert_eq!(basis_2q.len(), 16);
    }

    #[test]
    fn test_process_matrix() {
        let qpt = QuantumProcessTomography::new(1);

        // Mock process: identity
        let mock_process = |_prep: &str, meas: &str| {
            if meas == "I" {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        };

        let result = qpt
            .run(mock_process)
            .expect("QPT run should succeed with mock process");

        assert_eq!(result.num_qubits, 1);
        assert!(result.is_trace_preserving(1e-6));
        println!("Process matrix shape: {:?}", result.chi_matrix.dim());
    }

    #[test]
    fn test_process_fidelity() {
        let dim = 4;
        let identity = Array2::eye(dim);
        let noisy = &identity * Complex64::new(0.95, 0.0);

        let fidelity = QuantumProcessTomography::process_fidelity(&identity, &noisy);

        // Fidelity is the trace of the product, which for scaled identity is just the scaling factor times dim
        // So for 0.95 * I with dim=4, we expect fidelity = 0.95 * 4 = 3.8
        println!("Process fidelity: {}", fidelity);

        // The fidelity should be proportional to the scaling
        assert!(fidelity > 0.0 && fidelity <= dim as f64);
    }

    #[test]
    fn test_gst_initialization() {
        let gate_set = vec!["I".to_string(), "X".to_string(), "H".to_string()];
        let gst = GateSetTomography::new(1, gate_set, 10);

        assert_eq!(gst.num_qubits, 1);
        assert_eq!(gst.max_length, 10);

        let germs = gst.generate_germs();
        assert!(!germs.is_empty());

        let fiducials = gst.generate_fiducials();
        assert!(!fiducials.is_empty());
    }

    /// Build a 2-qubit gate whose action on |00> yields a chosen amplitude vector.
    /// The supplied amplitudes form the first column of the unitary; the remaining
    /// columns are completed by Gram-Schmidt from the standard basis.
    fn gate_from_first_column(amps: [Complex64; 4], q1: usize, q2: usize) -> TwoQubitUnitaryGate {
        let mut m = Array2::<Complex64>::zeros((4, 4));
        for i in 0..4 {
            m[[i, 0]] = amps[i];
        }
        // Seed the other columns with distinct standard basis vectors.
        m[[1, 1]] = Complex64::new(1.0, 0.0);
        m[[2, 2]] = Complex64::new(1.0, 0.0);
        m[[3, 3]] = Complex64::new(1.0, 0.0);
        let u = gram_schmidt_unitary(&m).expect("gram-schmidt");
        TwoQubitUnitaryGate::new(u, q1, q2)
    }

    #[test]
    fn test_apply_gate_bit_ordering_cnot() {
        // A CNOT (control = qubit 0, target = qubit 1) in row-major form, with
        // qubit 0 as the most-significant local bit. Applied to |10> (qubit 0
        // set) it must produce |11>.
        let cnot = scirs2_core::ndarray::array![
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0)
            ]
        ];
        let gate: Box<dyn GateOp> = Box::new(TwoQubitUnitaryGate::new(cnot, 0, 1));

        // simulate_circuit starts at |00>; CNOT leaves it unchanged.
        let state = simulate_circuit(std::slice::from_ref(&gate), 2).expect("simulate");
        assert!((state[0].norm() - 1.0).abs() < 1e-12);

        // Apply to a custom |10> state to check the controlled flip.
        let mut custom = Array1::<Complex64>::zeros(4);
        custom[1] = Complex64::new(1.0, 0.0); // qubit0 = 1, qubit1 = 0
        apply_gate(&mut custom, gate.as_ref(), 2).expect("apply");
        // Expect |11>: qubit0=1, qubit1=1 -> bits 0 and 1 set -> index 3.
        assert!((custom[3].norm() - 1.0).abs() < 1e-12, "got {custom:?}");
        assert!(custom[1].norm() < 1e-12);
    }

    #[test]
    fn test_find_heavy_outputs_above_median_not_first_half() {
        // Construct a circuit with a deliberately non-uniform output distribution
        // (the four basis probabilities are all distinct), so the median is well
        // defined and the heavy-output set is unambiguous.
        let amps = [
            Complex64::new(0.1_f64.sqrt(), 0.0),
            Complex64::new(0.4_f64.sqrt(), 0.0),
            Complex64::new(0.2_f64.sqrt(), 0.0),
            Complex64::new(0.3_f64.sqrt(), 0.0),
        ];
        let gate = gate_from_first_column(amps, 0, 1);
        let circuit: Vec<Box<dyn GateOp>> = vec![Box::new(gate)];

        let qv = QuantumVolume::new(2, 1, 100);
        let heavy = qv.find_heavy_outputs(2, &circuit).expect("heavy outputs");

        // Independently recompute the expected heavy set directly from the
        // simulated state vector (the source of truth).
        let state = simulate_circuit(&circuit, 2).expect("simulate");
        let probs: Vec<f64> = state.iter().map(|a| a.norm_sqr()).collect();

        // The distribution must be genuinely non-uniform and normalised.
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
        let max_p = probs.iter().cloned().fold(0.0_f64, f64::max);
        let min_p = probs.iter().cloned().fold(1.0_f64, f64::min);
        assert!(max_p - min_p > 1e-3, "distribution should be non-uniform");

        let mut sorted = probs.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = 0.5 * (sorted[1] + sorted[2]);
        let mut expected: Vec<usize> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > median)
            .map(|(i, _)| i)
            .collect();
        expected.sort_unstable();

        let mut heavy_sorted = heavy.clone();
        heavy_sorted.sort_unstable();
        assert_eq!(
            heavy_sorted, expected,
            "heavy outputs must be exactly the strictly-above-median indices"
        );
        // For four distinct probabilities, exactly two are above the median.
        assert_eq!(heavy_sorted.len(), 2);

        // It must NOT be the old fabricated "first half" (0..num_states/2).
        let first_half: Vec<usize> = (0..(1usize << 2) / 2).collect();
        assert_ne!(
            heavy_sorted, first_half,
            "heavy outputs must be computed from probabilities, not the first half"
        );
    }

    #[test]
    fn test_generate_random_circuit_is_non_empty() {
        // A real QV circuit must contain depth * (n/2) two-qubit gates, never an
        // empty placeholder.
        let mut qv = QuantumVolume::new(4, 1, 10);
        let n = 4;
        let (circuit, heavy) = qv.generate_random_circuit(n).expect("circuit");

        // depth = n layers, each with n/2 = 2 gates -> 8 gates.
        assert_eq!(circuit.len(), n * (n / 2));
        assert!(!circuit.is_empty(), "QV circuit must not be empty");
        for gate in &circuit {
            assert_eq!(gate.qubits().len(), 2, "each QV gate acts on 2 qubits");
        }

        // The state must be normalised and heavy outputs computed from it.
        let state = simulate_circuit(&circuit, n).expect("simulate");
        let total: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "state must stay normalised: {total}"
        );

        // Heavy outputs are a strict subset of all 2^n states and (for a generic
        // random circuit) non-empty.
        assert!(heavy.iter().all(|&i| i < (1usize << n)));
    }
}
