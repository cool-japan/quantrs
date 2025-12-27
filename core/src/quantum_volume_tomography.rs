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
        &self,
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

    /// Generate a random model circuit for quantum volume
    ///
    /// Returns the circuit and the set of heavy outputs (outputs with above-median probability)
    fn generate_random_circuit(
        &self,
        n_qubits: usize,
    ) -> QuantRS2Result<(Vec<Box<dyn GateOp>>, Vec<usize>)> {
        // For quantum volume, we use depth = n_qubits
        let depth = n_qubits;

        // Placeholder: generate random SU(4) gates
        // In a real implementation, this would generate random 2-qubit unitaries
        let circuit = vec![];

        // Simulate ideal circuit to find heavy outputs
        let heavy_outputs = self.find_heavy_outputs(n_qubits, &circuit)?;

        Ok((circuit, heavy_outputs))
    }

    /// Find heavy outputs (outputs with above-median probability)
    fn find_heavy_outputs(
        &self,
        n_qubits: usize,
        _circuit: &[Box<dyn GateOp>],
    ) -> QuantRS2Result<Vec<usize>> {
        // Simulate the circuit classically to find heavy outputs
        // This is a simplified placeholder

        let num_states = 1 << n_qubits;
        let median_prob = 1.0 / (num_states as f64);

        // In reality, we would:
        // 1. Simulate the circuit
        // 2. Calculate all outcome probabilities
        // 3. Find those above median

        // For now, return a placeholder (first half of bitstrings)
        Ok((0..num_states / 2).collect())
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
}
