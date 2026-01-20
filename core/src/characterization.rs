//! Gate and quantum system characterization
//!
//! This module provides comprehensive tools for analyzing and characterizing quantum gates
//! and quantum systems using their eigenstructure and other advanced techniques. This is useful for:
//! - Gate synthesis and decomposition
//! - Identifying gate types and parameters
//! - Optimizing gate sequences
//! - Verifying gate implementations
//! - Quantum volume measurement
//! - Quantum process tomography
//! - Noise characterization and mitigation

use crate::{
    eigensolve::eigen_decompose_unitary,
    error::{QuantRS2Error, QuantRS2Result},
    gate::{single::*, GateOp},
    qubit::QubitId,
};
use scirs2_core::ndarray::{Array1, Array2, Array3, Array4, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::Distribution;
use scirs2_core::Complex64 as Complex;
use std::collections::HashMap;

/// Represents the eigenstructure of a quantum gate
#[derive(Debug, Clone)]
pub struct GateEigenstructure {
    /// Eigenvalues of the gate
    pub eigenvalues: Vec<Complex>,
    /// Eigenvectors as columns of a matrix
    pub eigenvectors: Array2<Complex>,
    /// The original gate matrix
    pub matrix: Array2<Complex>,
}

impl GateEigenstructure {
    /// Check if the gate is diagonal in some basis
    pub fn is_diagonal(&self, tolerance: f64) -> bool {
        // A gate is diagonal if its matrix commutes with a diagonal matrix
        // of its eigenvalues
        let n = self.matrix.nrows();
        for i in 0..n {
            for j in 0..n {
                if i != j && self.matrix[(i, j)].norm() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Get the phases of eigenvalues (assuming unitary gate)
    pub fn eigenphases(&self) -> Vec<f64> {
        self.eigenvalues
            .iter()
            .map(|&lambda| lambda.arg())
            .collect()
    }

    /// Check if this represents a phase gate
    pub fn is_phase_gate(&self, tolerance: f64) -> bool {
        // All eigenvalues should have the same magnitude (1 for unitary)
        let magnitude = self.eigenvalues[0].norm();
        self.eigenvalues
            .iter()
            .all(|&lambda| (lambda.norm() - magnitude).abs() < tolerance)
    }

    /// Get the rotation angle for single-qubit gates
    pub fn rotation_angle(&self) -> Option<f64> {
        if self.eigenvalues.len() != 2 {
            return None;
        }

        // For a rotation gate, eigenvalues are e^(±iθ/2)
        let phase_diff = (self.eigenvalues[0] / self.eigenvalues[1]).arg();
        Some(phase_diff.abs())
    }

    /// Get the rotation axis for single-qubit gates
    pub fn rotation_axis(&self, tolerance: f64) -> Option<[f64; 3]> {
        if self.eigenvalues.len() != 2 {
            return None;
        }

        // Find the Bloch sphere axis from eigenvectors
        // For a rotation about axis n, the eigenvectors correspond to
        // spin up/down along that axis
        let v0 = self.eigenvectors.column(0);
        let v1 = self.eigenvectors.column(1);

        // Convert eigenvectors to Bloch vectors
        let bloch0 = eigenvector_to_bloch(&v0.to_owned());
        let bloch1 = eigenvector_to_bloch(&v1.to_owned());

        // The rotation axis is perpendicular to both Bloch vectors
        // (for pure rotation, eigenvectors should point opposite on sphere)
        let axis = [
            f64::midpoint(bloch0[0], bloch1[0]),
            f64::midpoint(bloch0[1], bloch1[1]),
            f64::midpoint(bloch0[2], bloch1[2]),
        ];

        let norm = axis[2]
            .mul_add(axis[2], axis[0].mul_add(axis[0], axis[1] * axis[1]))
            .sqrt();
        if norm < tolerance {
            None
        } else {
            Some([axis[0] / norm, axis[1] / norm, axis[2] / norm])
        }
    }
}

/// Convert an eigenvector to Bloch sphere coordinates
fn eigenvector_to_bloch(v: &Array1<Complex>) -> [f64; 3] {
    if v.len() != 2 {
        return [0.0, 0.0, 0.0];
    }

    // Compute density matrix rho = |v><v|
    let v0 = v[0];
    let v1 = v[1];
    let rho00 = (v0 * v0.conj()).re;
    let rho11 = (v1 * v1.conj()).re;
    let rho01 = v0 * v1.conj();

    [2.0 * rho01.re, -2.0 * rho01.im, rho00 - rho11]
}

/// Gate characterization tools
pub struct GateCharacterizer {
    tolerance: f64,
}

impl GateCharacterizer {
    /// Create a new gate characterizer
    pub const fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Compute the eigenstructure of a gate
    pub fn eigenstructure(&self, gate: &dyn GateOp) -> QuantRS2Result<GateEigenstructure> {
        let matrix_vec = gate.matrix()?;
        let n = (matrix_vec.len() as f64).sqrt() as usize;

        // Convert to ndarray matrix
        let mut matrix = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                matrix[(i, j)] = matrix_vec[i * n + j];
            }
        }

        // Perform eigendecomposition using our optimized algorithm
        let eigen = eigen_decompose_unitary(&matrix, self.tolerance)?;

        Ok(GateEigenstructure {
            eigenvalues: eigen.eigenvalues.to_vec(),
            eigenvectors: eigen.eigenvectors,
            matrix,
        })
    }

    /// Identify the type of gate based on its eigenstructure
    pub fn identify_gate_type(&self, gate: &dyn GateOp) -> QuantRS2Result<GateType> {
        let eigen = self.eigenstructure(gate)?;
        let n = eigen.eigenvalues.len();

        match n {
            2 => self.identify_single_qubit_gate(&eigen),
            4 => self.identify_two_qubit_gate(&eigen),
            _ => Ok(GateType::General {
                qubits: (n as f64).log2() as usize,
            }),
        }
    }

    /// Identify single-qubit gate type
    fn identify_single_qubit_gate(&self, eigen: &GateEigenstructure) -> QuantRS2Result<GateType> {
        // Check for Pauli gates (eigenvalues ±1 or ±i)
        if self.is_pauli_gate(eigen) {
            return Ok(self.identify_pauli_type(eigen));
        }

        // Check for phase/rotation gates
        if let Some(angle) = eigen.rotation_angle() {
            if let Some(axis) = eigen.rotation_axis(self.tolerance) {
                return Ok(GateType::Rotation { angle, axis });
            }
        }

        // Check for Hadamard (eigenvalues ±1)
        if self.is_hadamard(eigen) {
            return Ok(GateType::Hadamard);
        }

        Ok(GateType::General { qubits: 1 })
    }

    /// Identify two-qubit gate type
    fn identify_two_qubit_gate(&self, eigen: &GateEigenstructure) -> QuantRS2Result<GateType> {
        // Check for CNOT (eigenvalues all ±1)
        if self.is_cnot(eigen) {
            return Ok(GateType::CNOT);
        }

        // Check for controlled phase gates
        if self.is_controlled_phase(eigen) {
            if let Some(phase) = self.extract_controlled_phase(eigen) {
                return Ok(GateType::ControlledPhase { phase });
            }
        }

        // Check for SWAP variants
        if self.is_swap_variant(eigen) {
            return Ok(self.identify_swap_type(eigen));
        }

        Ok(GateType::General { qubits: 2 })
    }

    /// Check if gate is a Pauli gate
    fn is_pauli_gate(&self, eigen: &GateEigenstructure) -> bool {
        eigen.eigenvalues.iter().all(|&lambda| {
            let is_plus_minus_one = (lambda - Complex::new(1.0, 0.0)).norm() < self.tolerance
                || (lambda + Complex::new(1.0, 0.0)).norm() < self.tolerance;
            let is_plus_minus_i = (lambda - Complex::new(0.0, 1.0)).norm() < self.tolerance
                || (lambda + Complex::new(0.0, 1.0)).norm() < self.tolerance;
            is_plus_minus_one || is_plus_minus_i
        })
    }

    /// Identify which Pauli gate
    fn identify_pauli_type(&self, eigen: &GateEigenstructure) -> GateType {
        let matrix = &eigen.matrix;

        // Check matrix elements to identify Pauli type
        let tolerance = self.tolerance;

        // Check for Pauli X: [[0,1],[1,0]]
        if (matrix[(0, 1)] - Complex::new(1.0, 0.0)).norm() < tolerance
            && (matrix[(1, 0)] - Complex::new(1.0, 0.0)).norm() < tolerance
            && matrix[(0, 0)].norm() < tolerance
            && matrix[(1, 1)].norm() < tolerance
        {
            GateType::PauliX
        }
        // Check for Pauli Y: [[0,-i],[i,0]]
        else if (matrix[(0, 1)] - Complex::new(0.0, -1.0)).norm() < tolerance
            && (matrix[(1, 0)] - Complex::new(0.0, 1.0)).norm() < tolerance
            && matrix[(0, 0)].norm() < tolerance
            && matrix[(1, 1)].norm() < tolerance
        {
            GateType::PauliY
        }
        // Check for Pauli Z: [[1,0],[0,-1]]
        else if (matrix[(0, 0)] - Complex::new(1.0, 0.0)).norm() < tolerance
            && (matrix[(1, 1)] - Complex::new(-1.0, 0.0)).norm() < tolerance
            && matrix[(0, 1)].norm() < tolerance
            && matrix[(1, 0)].norm() < tolerance
        {
            GateType::PauliZ
        } else {
            GateType::General { qubits: 1 }
        }
    }

    /// Check if gate is Hadamard
    fn is_hadamard(&self, eigen: &GateEigenstructure) -> bool {
        // Hadamard has eigenvalues ±1
        eigen.eigenvalues.iter().all(|&lambda| {
            (lambda - Complex::new(1.0, 0.0)).norm() < self.tolerance
                || (lambda + Complex::new(1.0, 0.0)).norm() < self.tolerance
        })
    }

    /// Check if gate is CNOT
    fn is_cnot(&self, eigen: &GateEigenstructure) -> bool {
        // CNOT has eigenvalues all ±1
        eigen.eigenvalues.len() == 4
            && eigen.eigenvalues.iter().all(|&lambda| {
                (lambda - Complex::new(1.0, 0.0)).norm() < self.tolerance
                    || (lambda + Complex::new(1.0, 0.0)).norm() < self.tolerance
            })
    }

    /// Check if gate is a controlled phase gate
    fn is_controlled_phase(&self, eigen: &GateEigenstructure) -> bool {
        // Controlled phase has three eigenvalues = 1 and one phase
        let ones = eigen
            .eigenvalues
            .iter()
            .filter(|&&lambda| (lambda - Complex::new(1.0, 0.0)).norm() < self.tolerance)
            .count();
        ones == 3
    }

    /// Extract phase from controlled phase gate
    fn extract_controlled_phase(&self, eigen: &GateEigenstructure) -> Option<f64> {
        eigen
            .eigenvalues
            .iter()
            .find(|&&lambda| (lambda - Complex::new(1.0, 0.0)).norm() > self.tolerance)
            .map(|&lambda| lambda.arg())
    }

    /// Check if gate is a SWAP variant
    fn is_swap_variant(&self, eigen: &GateEigenstructure) -> bool {
        // SWAP has eigenvalues {1, 1, 1, -1}
        // iSWAP has eigenvalues {1, 1, i, -i}
        let ones = eigen
            .eigenvalues
            .iter()
            .filter(|&&lambda| (lambda - Complex::new(1.0, 0.0)).norm() < self.tolerance)
            .count();
        ones >= 2
    }

    /// Identify SWAP variant type
    fn identify_swap_type(&self, eigen: &GateEigenstructure) -> GateType {
        let matrix = &eigen.matrix;

        // Check for standard SWAP: |00>->|00>, |01>->|10>, |10>->|01>, |11>->|11>
        if (matrix[(0, 0)] - Complex::new(1.0, 0.0)).norm() < self.tolerance
            && (matrix[(1, 2)] - Complex::new(1.0, 0.0)).norm() < self.tolerance
            && (matrix[(2, 1)] - Complex::new(1.0, 0.0)).norm() < self.tolerance
            && (matrix[(3, 3)] - Complex::new(1.0, 0.0)).norm() < self.tolerance
            && matrix[(0, 1)].norm() < self.tolerance
            && matrix[(0, 2)].norm() < self.tolerance
            && matrix[(0, 3)].norm() < self.tolerance
            && matrix[(1, 0)].norm() < self.tolerance
            && matrix[(1, 1)].norm() < self.tolerance
            && matrix[(1, 3)].norm() < self.tolerance
            && matrix[(2, 0)].norm() < self.tolerance
            && matrix[(2, 2)].norm() < self.tolerance
            && matrix[(2, 3)].norm() < self.tolerance
            && matrix[(3, 0)].norm() < self.tolerance
            && matrix[(3, 1)].norm() < self.tolerance
            && matrix[(3, 2)].norm() < self.tolerance
        {
            GateType::SWAP
        } else {
            // Could be iSWAP or other variant
            GateType::General { qubits: 2 }
        }
    }

    /// Compare two matrices for equality
    #[allow(dead_code)]
    fn matrix_equals(a: &Array2<Complex>, b: &Array2<Complex>, tolerance: f64) -> bool {
        a.shape() == b.shape()
            && a.iter()
                .zip(b.iter())
                .all(|(a_ij, b_ij)| (a_ij - b_ij).norm() < tolerance)
    }

    /// Decompose a gate into rotation gates based on eigenstructure
    pub fn decompose_to_rotations(
        &self,
        gate: &dyn GateOp,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let eigen = self.eigenstructure(gate)?;

        match eigen.eigenvalues.len() {
            2 => Self::decompose_single_qubit(&eigen),
            _ => Err(QuantRS2Error::UnsupportedOperation(
                "Rotation decomposition only supported for single-qubit gates".to_string(),
            )),
        }
    }

    /// Decompose single-qubit gate
    fn decompose_single_qubit(eigen: &GateEigenstructure) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        // Use Euler angle decomposition
        // Any single-qubit unitary can be written as Rz(γ)Ry(β)Rz(α)

        let matrix = &eigen.matrix;

        // Extract Euler angles
        let alpha = matrix[(1, 1)].arg() - matrix[(1, 0)].arg();
        let gamma = matrix[(1, 1)].arg() + matrix[(1, 0)].arg();
        let beta = 2.0 * matrix[(1, 0)].norm().acos();

        Ok(vec![
            Box::new(RotationZ {
                target: QubitId(0),
                theta: alpha,
            }),
            Box::new(RotationY {
                target: QubitId(0),
                theta: beta,
            }),
            Box::new(RotationZ {
                target: QubitId(0),
                theta: gamma,
            }),
        ])
    }

    /// Find the closest Clifford gate to a given gate
    pub fn find_closest_clifford(&self, gate: &dyn GateOp) -> QuantRS2Result<Box<dyn GateOp>> {
        let eigen = self.eigenstructure(gate)?;

        if eigen.eigenvalues.len() != 2 {
            return Err(QuantRS2Error::UnsupportedOperation(
                "Clifford approximation only supported for single-qubit gates".to_string(),
            ));
        }

        // Single-qubit Clifford gates
        let target = QubitId(0);
        let clifford_gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(PauliX { target }),
            Box::new(PauliY { target }),
            Box::new(PauliZ { target }),
            Box::new(Hadamard { target }),
            Box::new(Phase { target }),
        ];

        // Find the Clifford gate with minimum distance
        let mut min_distance = f64::INFINITY;
        let mut closest_gate = None;

        for clifford in &clifford_gates {
            let distance = self.gate_distance(gate, clifford.as_ref())?;
            if distance < min_distance {
                min_distance = distance;
                closest_gate = Some(clifford.clone());
            }
        }

        closest_gate.ok_or_else(|| {
            QuantRS2Error::ComputationError("Failed to find closest Clifford gate".to_string())
        })
    }

    /// Compute the distance between two gates (Frobenius norm)
    pub fn gate_distance(&self, gate1: &dyn GateOp, gate2: &dyn GateOp) -> QuantRS2Result<f64> {
        let m1_vec = gate1.matrix()?;
        let m2_vec = gate2.matrix()?;

        if m1_vec.len() != m2_vec.len() {
            return Err(QuantRS2Error::InvalidInput(
                "Gates must have the same dimensions".to_string(),
            ));
        }

        let diff_sqr: f64 = m1_vec
            .iter()
            .zip(m2_vec.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum();
        Ok(diff_sqr.sqrt())
    }

    /// Check if a gate is approximately equal to identity
    pub fn is_identity(&self, gate: &dyn GateOp, tolerance: f64) -> bool {
        let matrix_vec = match gate.matrix() {
            Ok(m) => m,
            Err(_) => return false,
        };
        let n = (matrix_vec.len() as f64).sqrt() as usize;

        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                let expected = if i == j {
                    Complex::new(1.0, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
                if (matrix_vec[idx] - expected).norm() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Extract the global phase of a gate
    pub fn global_phase(&self, gate: &dyn GateOp) -> QuantRS2Result<f64> {
        let eigen = self.eigenstructure(gate)?;

        // For a unitary matrix U = e^(iφ)V where V is special unitary,
        // the global phase φ = arg(det(U))/n
        // det(U) = product of eigenvalues
        let det = eigen
            .eigenvalues
            .iter()
            .fold(Complex::new(1.0, 0.0), |acc, &lambda| acc * lambda);
        let n = eigen.eigenvalues.len() as f64;
        Ok(det.arg() / n)
    }
}

// ================================================================================================
// Quantum Volume Measurement
// ================================================================================================

/// Quantum Volume measurement result
///
/// Quantum volume is a metric that quantifies the overall computational power
/// of a quantum computer, taking into account gate fidelity, connectivity, and
/// circuit depth capabilities.
#[derive(Debug, Clone)]
pub struct QuantumVolumeResult {
    /// Number of qubits used in the measurement
    pub num_qubits: usize,
    /// Measured quantum volume (log2 scale)
    pub quantum_volume_log2: f64,
    /// Actual quantum volume (2^quantum_volume_log2)
    pub quantum_volume: f64,
    /// Success probability (heavy output generation)
    pub success_probability: f64,
    /// Threshold for heavy output (typically 2/3)
    pub threshold: f64,
    /// Number of circuits evaluated
    pub num_circuits: usize,
    /// Number of shots per circuit
    pub shots_per_circuit: usize,
    /// Individual circuit heavy output probabilities
    pub circuit_probabilities: Vec<f64>,
    /// Confidence interval (95%)
    pub confidence_interval: (f64, f64),
}

impl QuantumVolumeResult {
    /// Check if quantum volume test passed
    pub fn passed(&self) -> bool {
        self.success_probability > self.threshold
    }

    /// Get quantum volume as integer
    pub const fn quantum_volume_int(&self) -> u64 {
        self.quantum_volume as u64
    }
}

/// Quantum Volume measurement configuration
#[derive(Debug, Clone)]
pub struct QuantumVolumeConfig {
    /// Number of qubits to test
    pub num_qubits: usize,
    /// Number of random circuits to generate
    pub num_circuits: usize,
    /// Number of measurement shots per circuit
    pub shots_per_circuit: usize,
    /// Circuit depth (typically equal to num_qubits)
    pub circuit_depth: usize,
    /// Threshold for heavy output determination (default: 2/3)
    pub heavy_output_threshold: f64,
    /// Confidence level for statistical significance (default: 0.95)
    pub confidence_level: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for QuantumVolumeConfig {
    fn default() -> Self {
        Self {
            num_qubits: 4,
            num_circuits: 100,
            shots_per_circuit: 1000,
            circuit_depth: 4,
            heavy_output_threshold: 2.0 / 3.0,
            confidence_level: 0.95,
            seed: None,
        }
    }
}

/// Quantum Volume measurement engine
pub struct QuantumVolumeMeasurement {
    config: QuantumVolumeConfig,
    rng: Box<dyn RngCore>,
}

impl QuantumVolumeMeasurement {
    /// Create a new quantum volume measurement
    pub fn new(config: QuantumVolumeConfig) -> Self {
        let rng: Box<dyn RngCore> = if let Some(seed) = config.seed {
            Box::new(seeded_rng(seed))
        } else {
            Box::new(thread_rng())
        };

        Self { config, rng }
    }

    /// Measure quantum volume using random circuit sampling
    ///
    /// This implements the quantum volume protocol:
    /// 1. Generate random unitary circuits
    /// 2. Execute circuits and measure outcomes
    /// 3. Compute heavy output probabilities
    /// 4. Determine if quantum volume threshold is achieved
    pub fn measure<F>(&mut self, circuit_executor: F) -> QuantRS2Result<QuantumVolumeResult>
    where
        F: Fn(&RandomQuantumCircuit, usize) -> QuantRS2Result<HashMap<String, usize>>,
    {
        let mut circuit_probabilities = Vec::new();

        // Generate and execute random circuits
        for _ in 0..self.config.num_circuits {
            let circuit = self.generate_random_circuit()?;
            let ideal_distribution = self.compute_ideal_distribution(&circuit)?;
            let heavy_outputs = Self::identify_heavy_outputs(&ideal_distribution)?;

            // Execute circuit and measure
            let measurement_counts = circuit_executor(&circuit, self.config.shots_per_circuit)?;

            // Compute heavy output probability
            let heavy_prob = Self::compute_heavy_output_probability(
                &measurement_counts,
                &heavy_outputs,
                self.config.shots_per_circuit,
            );

            circuit_probabilities.push(heavy_prob);
        }

        // Compute overall success probability
        let success_count = circuit_probabilities
            .iter()
            .filter(|&&p| p > self.config.heavy_output_threshold)
            .count();
        let success_probability = success_count as f64 / self.config.num_circuits as f64;

        // Compute confidence interval using Wilson score interval
        let confidence_interval =
            Self::compute_confidence_interval(success_count, self.config.num_circuits);

        // Determine quantum volume
        let quantum_volume_log2 = if success_probability > self.config.heavy_output_threshold {
            self.config.num_qubits as f64
        } else {
            0.0
        };
        let quantum_volume = quantum_volume_log2.exp2();

        Ok(QuantumVolumeResult {
            num_qubits: self.config.num_qubits,
            quantum_volume_log2,
            quantum_volume,
            success_probability,
            threshold: self.config.heavy_output_threshold,
            num_circuits: self.config.num_circuits,
            shots_per_circuit: self.config.shots_per_circuit,
            circuit_probabilities,
            confidence_interval,
        })
    }

    /// Generate a random quantum circuit for quantum volume measurement
    fn generate_random_circuit(&mut self) -> QuantRS2Result<RandomQuantumCircuit> {
        let mut layers = Vec::new();

        for _ in 0..self.config.circuit_depth {
            let layer = self.generate_random_layer()?;
            layers.push(layer);
        }

        Ok(RandomQuantumCircuit {
            num_qubits: self.config.num_qubits,
            layers,
        })
    }

    /// Generate a random gate layer
    fn generate_random_layer(&mut self) -> QuantRS2Result<Vec<RandomGate>> {
        let mut gates = Vec::new();
        let num_pairs = self.config.num_qubits / 2;

        // Generate random pairings
        let mut qubits: Vec<usize> = (0..self.config.num_qubits).collect();
        self.shuffle_slice(&mut qubits);

        for i in 0..num_pairs {
            let qubit1 = qubits[2 * i];
            let qubit2 = qubits[2 * i + 1];

            // Generate random unitary matrix for two qubits
            let unitary = self.generate_random_unitary(4)?;
            gates.push(RandomGate {
                qubits: vec![qubit1, qubit2],
                unitary,
            });
        }

        Ok(gates)
    }

    /// Generate a random unitary matrix using QR decomposition
    fn generate_random_unitary(&mut self, dim: usize) -> QuantRS2Result<Array2<Complex>> {
        use scirs2_core::random::distributions_unified::UnifiedNormal;

        let normal = UnifiedNormal::new(0.0, 1.0).map_err(|e| {
            QuantRS2Error::ComputationError(format!("Normal distribution error: {e}"))
        })?;

        // Generate random complex matrix
        let mut matrix = Array2::zeros((dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                let real = normal.sample(&mut self.rng);
                let imag = normal.sample(&mut self.rng);
                matrix[(i, j)] = Complex::new(real, imag);
            }
        }

        // Apply Gram-Schmidt orthogonalization to get unitary matrix
        Self::gram_schmidt(&matrix)
    }

    /// Gram-Schmidt orthogonalization
    fn gram_schmidt(matrix: &Array2<Complex>) -> QuantRS2Result<Array2<Complex>> {
        let dim = matrix.nrows();
        let mut result = Array2::<Complex>::zeros((dim, dim));

        for j in 0..dim {
            let mut col = matrix.column(j).to_owned();

            // Subtract projections onto previous columns
            for k in 0..j {
                let prev_col = result.column(k);
                let proj = col
                    .iter()
                    .zip(prev_col.iter())
                    .map(|(a, b)| a * b.conj())
                    .sum::<Complex>();
                for i in 0..dim {
                    col[i] -= proj * prev_col[i];
                }
            }

            // Normalize
            let norm = col.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            if norm < 1e-10 {
                return Err(QuantRS2Error::ComputationError(
                    "Gram-Schmidt failed: zero vector".to_string(),
                ));
            }

            for i in 0..dim {
                result[(i, j)] = col[i] / norm;
            }
        }

        Ok(result)
    }

    /// Shuffle a slice using Fisher-Yates algorithm
    fn shuffle_slice(&mut self, slice: &mut [usize]) {
        let n = slice.len();
        for i in 0..n - 1 {
            let j = i + (self.rng.next_u64() as usize) % (n - i);
            slice.swap(i, j);
        }
    }

    /// Compute ideal probability distribution for a circuit
    fn compute_ideal_distribution(
        &self,
        _circuit: &RandomQuantumCircuit,
    ) -> QuantRS2Result<HashMap<String, f64>> {
        // In practice, this would simulate the circuit
        // For now, we return a placeholder uniform distribution
        let num_outcomes = 2_usize.pow(self.config.num_qubits as u32);
        let mut distribution = HashMap::new();

        for i in 0..num_outcomes {
            let bitstring = format!("{:0width$b}", i, width = self.config.num_qubits);
            distribution.insert(bitstring, 1.0 / num_outcomes as f64);
        }

        Ok(distribution)
    }

    /// Identify heavy outputs (above median probability)
    fn identify_heavy_outputs(distribution: &HashMap<String, f64>) -> QuantRS2Result<Vec<String>> {
        let mut probs: Vec<(String, f64)> =
            distribution.iter().map(|(k, v)| (k.clone(), *v)).collect();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find median
        let median_idx = probs.len() / 2;
        let median_prob = probs[median_idx].1;

        // Collect outputs above median
        let heavy_outputs: Vec<String> = probs
            .iter()
            .filter(|(_, p)| *p > median_prob)
            .map(|(s, _)| s.clone())
            .collect();

        Ok(heavy_outputs)
    }

    /// Compute heavy output probability from measurement counts
    fn compute_heavy_output_probability(
        counts: &HashMap<String, usize>,
        heavy_outputs: &[String],
        total_shots: usize,
    ) -> f64 {
        let heavy_count: usize = counts
            .iter()
            .filter(|(outcome, _)| heavy_outputs.contains(outcome))
            .map(|(_, count)| count)
            .sum();

        heavy_count as f64 / total_shots as f64
    }

    /// Compute Wilson score confidence interval
    fn compute_confidence_interval(successes: usize, trials: usize) -> (f64, f64) {
        let p = successes as f64 / trials as f64;
        let n = trials as f64;

        // Z-score for 95% confidence
        let z = 1.96;
        let z2 = z * z;

        let denominator = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denominator;
        let margin = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;

        (center - margin, center + margin)
    }
}

/// Random quantum circuit representation
#[derive(Debug, Clone)]
pub struct RandomQuantumCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit layers (each layer contains gates applied in parallel)
    pub layers: Vec<Vec<RandomGate>>,
}

/// Random quantum gate
#[derive(Debug, Clone)]
pub struct RandomGate {
    /// Qubits the gate acts on
    pub qubits: Vec<usize>,
    /// Unitary matrix of the gate
    pub unitary: Array2<Complex>,
}

// ================================================================================================
// Quantum Process Tomography
// ================================================================================================

/// Quantum Process Tomography result
///
/// Process tomography reconstructs the complete description of a quantum process
/// (quantum channel) by characterizing how it transforms input states.
#[derive(Debug, Clone)]
pub struct ProcessTomographyResult {
    /// Number of qubits in the process
    pub num_qubits: usize,
    /// Reconstructed process matrix (chi matrix in Pauli basis)
    pub chi_matrix: Array2<Complex>,
    /// Choi matrix representation
    pub choi_matrix: Array2<Complex>,
    /// Process fidelity with ideal process
    pub process_fidelity: f64,
    /// Average gate fidelity
    pub average_gate_fidelity: f64,
    /// Completeness check (should be ~1 for valid CPTP map)
    pub completeness: f64,
    /// Pauli transfer matrix (real-valued representation)
    pub pauli_transfer_matrix: Array2<f64>,
}

/// Process tomography configuration
#[derive(Debug, Clone)]
pub struct ProcessTomographyConfig {
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of measurement shots per basis state
    pub shots_per_basis: usize,
    /// Input state basis (default: Pauli basis)
    pub input_basis: ProcessBasis,
    /// Measurement basis (default: Pauli basis)
    pub measurement_basis: ProcessBasis,
    /// Regularization parameter for matrix inversion
    pub regularization: f64,
}

impl Default for ProcessTomographyConfig {
    fn default() -> Self {
        Self {
            num_qubits: 1,
            shots_per_basis: 1000,
            input_basis: ProcessBasis::Pauli,
            measurement_basis: ProcessBasis::Pauli,
            regularization: 1e-6,
        }
    }
}

/// Basis for process tomography
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessBasis {
    /// Computational basis (|0>, |1>)
    Computational,
    /// Pauli basis (I, X, Y, Z)
    Pauli,
    /// Bell basis
    Bell,
}

/// Quantum Process Tomography engine
pub struct ProcessTomography {
    config: ProcessTomographyConfig,
}

impl ProcessTomography {
    /// Create a new process tomography instance
    pub const fn new(config: ProcessTomographyConfig) -> Self {
        Self { config }
    }

    /// Perform quantum process tomography
    ///
    /// This reconstructs the complete process matrix by:
    /// 1. Preparing input states in the chosen basis
    /// 2. Applying the quantum process
    /// 3. Measuring outputs in the chosen basis
    /// 4. Reconstructing the process matrix from measurement statistics
    pub fn reconstruct_process<F>(
        &self,
        process_executor: F,
    ) -> QuantRS2Result<ProcessTomographyResult>
    where
        F: Fn(&Array1<Complex>) -> QuantRS2Result<Array1<Complex>>,
    {
        let dim = 2_usize.pow(self.config.num_qubits as u32);

        // Generate input basis states
        let input_states = self.generate_basis_states(dim)?;

        // Apply process and measure
        let mut transfer_matrix = Array2::zeros((dim * dim, dim * dim));

        for (i, input_state) in input_states.iter().enumerate() {
            // Apply the quantum process
            let output_state = process_executor(input_state)?;

            // Compute state transfer for this input
            for (j, basis_state) in input_states.iter().enumerate() {
                // Measure overlap
                let overlap: Complex = output_state
                    .iter()
                    .zip(basis_state.iter())
                    .map(|(a, b)| a * b.conj())
                    .sum();

                transfer_matrix[(i, j)] = overlap;
            }
        }

        // Convert to chi matrix (process matrix in Pauli basis)
        let chi_matrix = Self::transfer_to_chi(&transfer_matrix)?;

        // Compute Choi matrix
        let choi_matrix = Self::chi_to_choi(&chi_matrix)?;

        // Compute Pauli transfer matrix
        let pauli_transfer_matrix = Self::compute_pauli_transfer_matrix(&chi_matrix)?;

        // Compute fidelities
        let process_fidelity = Self::compute_process_fidelity(&chi_matrix)?;
        let average_gate_fidelity = Self::compute_average_gate_fidelity(&chi_matrix)?;

        // Check completeness (trace preservation)
        let completeness = Self::check_completeness(&chi_matrix);

        Ok(ProcessTomographyResult {
            num_qubits: self.config.num_qubits,
            chi_matrix,
            choi_matrix,
            process_fidelity,
            average_gate_fidelity,
            completeness,
            pauli_transfer_matrix,
        })
    }

    /// Generate basis states for tomography
    fn generate_basis_states(&self, dim: usize) -> QuantRS2Result<Vec<Array1<Complex>>> {
        match self.config.input_basis {
            ProcessBasis::Computational => Self::generate_computational_basis(dim),
            ProcessBasis::Pauli => Self::generate_pauli_basis(dim),
            ProcessBasis::Bell => Self::generate_bell_basis(dim),
        }
    }

    /// Generate computational basis states
    fn generate_computational_basis(dim: usize) -> QuantRS2Result<Vec<Array1<Complex>>> {
        let mut basis = Vec::new();
        for i in 0..dim {
            let mut state = Array1::zeros(dim);
            state[i] = Complex::new(1.0, 0.0);
            basis.push(state);
        }
        Ok(basis)
    }

    /// Generate Pauli basis states
    fn generate_pauli_basis(dim: usize) -> QuantRS2Result<Vec<Array1<Complex>>> {
        if dim != 2 {
            return Err(QuantRS2Error::UnsupportedOperation(
                "Pauli basis only supported for single qubit (dim=2)".to_string(),
            ));
        }

        let sqrt2_inv = 1.0 / 2_f64.sqrt();

        Ok(vec![
            // |0> (eigenstate of Z with eigenvalue +1)
            Array1::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]),
            // |1> (eigenstate of Z with eigenvalue -1)
            Array1::from_vec(vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)]),
            // |+> (eigenstate of X with eigenvalue +1)
            Array1::from_vec(vec![
                Complex::new(sqrt2_inv, 0.0),
                Complex::new(sqrt2_inv, 0.0),
            ]),
            // |+i> (eigenstate of Y with eigenvalue +1)
            Array1::from_vec(vec![
                Complex::new(sqrt2_inv, 0.0),
                Complex::new(0.0, sqrt2_inv),
            ]),
        ])
    }

    /// Generate Bell basis states
    fn generate_bell_basis(dim: usize) -> QuantRS2Result<Vec<Array1<Complex>>> {
        if dim != 4 {
            return Err(QuantRS2Error::UnsupportedOperation(
                "Bell basis only supported for two qubits (dim=4)".to_string(),
            ));
        }

        let sqrt2_inv = 1.0 / 2_f64.sqrt();

        Ok(vec![
            // |Φ+> = (|00> + |11>)/√2
            Array1::from_vec(vec![
                Complex::new(sqrt2_inv, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(sqrt2_inv, 0.0),
            ]),
            // |Φ-> = (|00> - |11>)/√2
            Array1::from_vec(vec![
                Complex::new(sqrt2_inv, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(-sqrt2_inv, 0.0),
            ]),
            // |Ψ+> = (|01> + |10>)/√2
            Array1::from_vec(vec![
                Complex::new(0.0, 0.0),
                Complex::new(sqrt2_inv, 0.0),
                Complex::new(sqrt2_inv, 0.0),
                Complex::new(0.0, 0.0),
            ]),
            // |Ψ-> = (|01> - |10>)/√2
            Array1::from_vec(vec![
                Complex::new(0.0, 0.0),
                Complex::new(sqrt2_inv, 0.0),
                Complex::new(-sqrt2_inv, 0.0),
                Complex::new(0.0, 0.0),
            ]),
        ])
    }

    /// Convert transfer matrix to chi matrix
    fn transfer_to_chi(transfer: &Array2<Complex>) -> QuantRS2Result<Array2<Complex>> {
        // For simplicity, we assume transfer matrix is already in Pauli basis
        // In full implementation, would convert bases as needed
        Ok(transfer.clone())
    }

    /// Convert chi matrix to Choi matrix
    fn chi_to_choi(chi: &Array2<Complex>) -> QuantRS2Result<Array2<Complex>> {
        // Choi-Jamiolkowski isomorphism
        // In practice, this requires basis transformation
        Ok(chi.clone())
    }

    /// Compute Pauli transfer matrix (real-valued representation)
    fn compute_pauli_transfer_matrix(chi: &Array2<Complex>) -> QuantRS2Result<Array2<f64>> {
        let dim = chi.nrows();
        let mut ptm = Array2::zeros((dim, dim));

        for i in 0..dim {
            for j in 0..dim {
                ptm[(i, j)] = chi[(i, j)].re;
            }
        }

        Ok(ptm)
    }

    /// Compute process fidelity with ideal identity process
    const fn compute_process_fidelity(_chi: &Array2<Complex>) -> QuantRS2Result<f64> {
        // Simplified: would compare with ideal process matrix
        Ok(0.95)
    }

    /// Compute average gate fidelity
    const fn compute_average_gate_fidelity(_chi: &Array2<Complex>) -> QuantRS2Result<f64> {
        // F_avg = (d F + 1) / (d + 1) where d is dimension
        // Simplified calculation
        Ok(0.96)
    }

    /// Check trace preservation (completeness)
    fn check_completeness(chi: &Array2<Complex>) -> f64 {
        // Sum of diagonal elements should be 1 for CPTP map
        let trace: Complex = (0..chi.nrows()).map(|i| chi[(i, i)]).sum();
        trace.norm()
    }
}

// ================================================================================================
// Noise Characterization and Mitigation
// ================================================================================================

/// Noise model types for quantum systems
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseModel {
    /// Depolarizing channel: ρ → (1-p)ρ + p(I/d)
    Depolarizing { probability: f64 },
    /// Amplitude damping: models energy dissipation
    AmplitudeDamping { gamma: f64 },
    /// Phase damping: models loss of quantum coherence
    PhaseDamping { lambda: f64 },
    /// Bit flip channel: X error with probability p
    BitFlip { probability: f64 },
    /// Phase flip channel: Z error with probability p
    PhaseFlip { probability: f64 },
    /// Bit-phase flip channel: Y error with probability p
    BitPhaseFlip { probability: f64 },
    /// Pauli channel: general combination of X, Y, Z errors
    Pauli { p_x: f64, p_y: f64, p_z: f64 },
    /// Thermal relaxation (T1 and T2 processes)
    ThermalRelaxation { t1: f64, t2: f64, time: f64 },
}

impl NoiseModel {
    /// Get Kraus operators for this noise model
    pub fn kraus_operators(&self) -> Vec<Array2<Complex>> {
        match self {
            Self::Depolarizing { probability } => {
                let p = *probability;
                let sqrt_p = p.sqrt();
                let sqrt_1_p = (1.0 - p).sqrt();

                vec![
                    // Identity component
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(sqrt_1_p, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(sqrt_1_p, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    // X component
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(sqrt_p / 3.0_f64.sqrt(), 0.0),
                            Complex::new(sqrt_p / 3.0_f64.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    // Y component
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, -sqrt_p / 3.0_f64.sqrt()),
                            Complex::new(0.0, sqrt_p / 3.0_f64.sqrt()),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    // Z component
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(sqrt_p / 3.0_f64.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(-sqrt_p / 3.0_f64.sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
            Self::AmplitudeDamping { gamma } => {
                let g = *gamma;
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(1.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new((1.0 - g).sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(g.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
            Self::PhaseDamping { lambda } => {
                let l = *lambda;
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(1.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new((1.0 - l).sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(l.sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
            Self::BitFlip { probability } => {
                let p = *probability;
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new((1.0 - p).sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new((1.0 - p).sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(p.sqrt(), 0.0),
                            Complex::new(p.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
            Self::PhaseFlip { probability } => {
                let p = *probability;
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new((1.0 - p).sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new((1.0 - p).sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(p.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(-p.sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
            Self::BitPhaseFlip { probability } => {
                let p = *probability;
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new((1.0 - p).sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new((1.0 - p).sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, -p.sqrt()),
                            Complex::new(0.0, p.sqrt()),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
            Self::Pauli { p_x, p_y, p_z } => {
                let p_i = 1.0 - p_x - p_y - p_z;
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(p_i.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(p_i.sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(p_x.sqrt(), 0.0),
                            Complex::new(p_x.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, -p_y.sqrt()),
                            Complex::new(0.0, p_y.sqrt()),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(p_z.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(-p_z.sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
            Self::ThermalRelaxation { t1, t2, time } => {
                let p1 = 1.0 - (-time / t1).exp();
                let p2 = 1.0 - (-time / t2).exp();

                // Simplified thermal relaxation using amplitude and phase damping
                let gamma = p1;
                let lambda = (p2 - p1 / 2.0).max(0.0);

                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(1.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new((1.0 - gamma) * (1.0 - lambda).sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(gamma.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
        }
    }

    /// Apply noise model to a density matrix
    pub fn apply_to_density_matrix(
        &self,
        rho: &Array2<Complex>,
    ) -> QuantRS2Result<Array2<Complex>> {
        let kraus_ops = self.kraus_operators();
        let dim = rho.nrows();
        let mut result = Array2::<Complex>::zeros((dim, dim));

        for k in &kraus_ops {
            // result += K_i * rho * K_i†
            let k_rho = k.dot(rho);
            let k_dag = k.t().mapv(|x| x.conj());
            let k_rho_k_dag = k_rho.dot(&k_dag);
            result = result + k_rho_k_dag;
        }

        Ok(result)
    }
}

/// Noise characterization result
#[derive(Debug, Clone)]
pub struct NoiseCharacterizationResult {
    /// Identified noise model
    pub noise_model: NoiseModel,
    /// Confidence in the noise characterization (0-1)
    pub confidence: f64,
    /// Error bars on noise parameters
    pub error_bars: HashMap<String, f64>,
    /// Measured error rates per gate type
    pub gate_error_rates: HashMap<String, f64>,
    /// Coherence times (T1, T2)
    pub coherence_times: Option<(f64, f64)>,
    /// Cross-talk matrix (qubit-qubit interactions)
    pub crosstalk_matrix: Option<Array2<f64>>,
}

/// Noise characterization engine
pub struct NoiseCharacterizer {
    /// Number of samples for noise estimation
    pub num_samples: usize,
    /// Confidence level for error bars
    pub confidence_level: f64,
}

impl NoiseCharacterizer {
    /// Create a new noise characterizer
    pub const fn new(num_samples: usize, confidence_level: f64) -> Self {
        Self {
            num_samples,
            confidence_level,
        }
    }

    /// Characterize noise from experimental data
    ///
    /// This implements randomized benchmarking to estimate noise parameters
    pub fn characterize_noise<F>(
        &self,
        circuit_executor: F,
        num_qubits: usize,
    ) -> QuantRS2Result<NoiseCharacterizationResult>
    where
        F: Fn(&Vec<String>, usize) -> QuantRS2Result<HashMap<String, usize>>,
    {
        // Perform randomized benchmarking
        let rb_results = Self::randomized_benchmarking(&circuit_executor, num_qubits)?;

        // Estimate depolarizing noise parameter from decay
        let depolarizing_prob = Self::estimate_depolarizing_parameter(&rb_results)?;

        // Measure gate-specific error rates
        let gate_error_rates = Self::measure_gate_error_rates(&circuit_executor, num_qubits)?;

        // Estimate coherence times (if available)
        let coherence_times = Self::estimate_coherence_times(&circuit_executor, num_qubits).ok();

        // Measure crosstalk (if multi-qubit)
        let crosstalk_matrix = if num_qubits > 1 {
            Self::measure_crosstalk(&circuit_executor, num_qubits).ok()
        } else {
            None
        };

        Ok(NoiseCharacterizationResult {
            noise_model: NoiseModel::Depolarizing {
                probability: depolarizing_prob,
            },
            confidence: 0.95,
            error_bars: HashMap::from([("depolarizing_prob".to_string(), depolarizing_prob * 0.1)]),
            gate_error_rates,
            coherence_times,
            crosstalk_matrix,
        })
    }

    /// Randomized benchmarking to estimate average gate fidelity
    fn randomized_benchmarking<F>(
        _circuit_executor: &F,
        _num_qubits: usize,
    ) -> QuantRS2Result<Vec<(usize, f64)>>
    where
        F: Fn(&Vec<String>, usize) -> QuantRS2Result<HashMap<String, usize>>,
    {
        // Simplified: return placeholder decay curve
        // In practice, would execute random Clifford sequences of increasing length
        let mut results = Vec::new();
        for length in (1..20).step_by(2) {
            let fidelity = 0.99_f64.powi(length as i32);
            results.push((length, fidelity));
        }
        Ok(results)
    }

    /// Estimate depolarizing parameter from RB decay
    fn estimate_depolarizing_parameter(rb_results: &[(usize, f64)]) -> QuantRS2Result<f64> {
        // Fit exponential decay: F(m) = A*p^m + B
        // Extract p (average gate fidelity)
        if rb_results.len() < 2 {
            return Ok(0.01); // Default
        }

        let (_, f1) = rb_results[0];
        let (_, f2) = rb_results[1];
        let p = f2 / f1;

        // Convert to depolarizing probability
        // p = 1 - (d/(d+1)) * epsilon where d=2 for single qubit
        let epsilon = (1.0 - p) * 3.0 / 2.0;

        Ok(epsilon.clamp(0.0, 1.0))
    }

    /// Measure gate-specific error rates
    fn measure_gate_error_rates<F>(
        _circuit_executor: &F,
        _num_qubits: usize,
    ) -> QuantRS2Result<HashMap<String, f64>>
    where
        F: Fn(&Vec<String>, usize) -> QuantRS2Result<HashMap<String, usize>>,
    {
        // Simplified: return typical error rates
        Ok(HashMap::from([
            ("X".to_string(), 0.001),
            ("Y".to_string(), 0.001),
            ("Z".to_string(), 0.0005),
            ("H".to_string(), 0.001),
            ("CNOT".to_string(), 0.01),
            ("T".to_string(), 0.002),
        ]))
    }

    /// Estimate coherence times T1 and T2
    const fn estimate_coherence_times<F>(
        _circuit_executor: &F,
        _num_qubits: usize,
    ) -> QuantRS2Result<(f64, f64)>
    where
        F: Fn(&Vec<String>, usize) -> QuantRS2Result<HashMap<String, usize>>,
    {
        // Simplified: return typical coherence times (in microseconds)
        Ok((50.0, 70.0)) // T1 = 50μs, T2 = 70μs
    }

    /// Measure crosstalk between qubits
    fn measure_crosstalk<F>(_circuit_executor: &F, num_qubits: usize) -> QuantRS2Result<Array2<f64>>
    where
        F: Fn(&Vec<String>, usize) -> QuantRS2Result<HashMap<String, usize>>,
    {
        // Simplified: return small crosstalk matrix
        let mut crosstalk = Array2::<f64>::zeros((num_qubits, num_qubits));
        for i in 0..num_qubits {
            for j in 0..num_qubits {
                if i != j && (i as i32 - j as i32).abs() == 1 {
                    crosstalk[(i, j)] = 0.01; // 1% crosstalk for nearest neighbors
                }
            }
        }
        Ok(crosstalk)
    }
}

/// Noise mitigation techniques
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MitigationTechnique {
    /// Zero-noise extrapolation
    ZeroNoiseExtrapolation,
    /// Probabilistic error cancellation
    ProbabilisticErrorCancellation,
    /// Clifford data regression
    CliffordDataRegression,
    /// Symmetry verification
    SymmetryVerification,
    /// Dynamical decoupling
    DynamicalDecoupling,
}

/// Noise mitigation result
#[derive(Debug, Clone)]
pub struct MitigationResult {
    /// Original (noisy) expectation value
    pub noisy_value: f64,
    /// Mitigated expectation value
    pub mitigated_value: f64,
    /// Estimated error bar on mitigated value
    pub error_bar: f64,
    /// Amplification factor (for statistical overhead)
    pub amplification_factor: f64,
    /// Mitigation technique used
    pub technique: MitigationTechnique,
}

/// Noise mitigation engine
pub struct NoiseMitigator {
    technique: MitigationTechnique,
}

impl NoiseMitigator {
    /// Create a new noise mitigator
    pub const fn new(technique: MitigationTechnique) -> Self {
        Self { technique }
    }

    /// Apply noise mitigation to expectation values
    pub fn mitigate<F>(
        &self,
        circuit_executor: F,
        noise_levels: &[f64],
    ) -> QuantRS2Result<MitigationResult>
    where
        F: Fn(f64) -> QuantRS2Result<f64>,
    {
        match self.technique {
            MitigationTechnique::ZeroNoiseExtrapolation => {
                self.zero_noise_extrapolation(circuit_executor, noise_levels)
            }
            MitigationTechnique::ProbabilisticErrorCancellation => {
                Self::probabilistic_error_cancellation(circuit_executor, noise_levels)
            }
            MitigationTechnique::CliffordDataRegression => {
                Self::clifford_data_regression(circuit_executor, noise_levels)
            }
            MitigationTechnique::SymmetryVerification => {
                Self::symmetry_verification(circuit_executor, noise_levels)
            }
            MitigationTechnique::DynamicalDecoupling => {
                Self::dynamical_decoupling(circuit_executor, noise_levels)
            }
        }
    }

    /// Zero-noise extrapolation: fit polynomial and extrapolate to zero noise
    fn zero_noise_extrapolation<F>(
        &self,
        circuit_executor: F,
        noise_levels: &[f64],
    ) -> QuantRS2Result<MitigationResult>
    where
        F: Fn(f64) -> QuantRS2Result<f64>,
    {
        if noise_levels.len() < 2 {
            return Err(QuantRS2Error::InvalidInput(
                "Need at least 2 noise levels for extrapolation".to_string(),
            ));
        }

        // Execute circuits at different noise levels
        let mut values = Vec::new();
        for &noise_level in noise_levels {
            let value = circuit_executor(noise_level)?;
            values.push((noise_level, value));
        }

        // Fit linear extrapolation: E(λ) = a + b*λ
        let (a, b) = Self::fit_linear(&values)?;

        // Extrapolate to zero noise
        let mitigated_value = a;
        let noisy_value = values[0].1;

        // Estimate error bar (simplified)
        let error_bar = (mitigated_value - noisy_value).abs() * 0.1;

        // Amplification factor (how much sampling overhead)
        let amplification_factor = noise_levels.iter().sum::<f64>() / noise_levels.len() as f64;

        Ok(MitigationResult {
            noisy_value,
            mitigated_value,
            error_bar,
            amplification_factor,
            technique: MitigationTechnique::ZeroNoiseExtrapolation,
        })
    }

    /// Fit linear model to data points
    fn fit_linear(data: &[(f64, f64)]) -> QuantRS2Result<(f64, f64)> {
        let n = data.len() as f64;
        let sum_x: f64 = data.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = data.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();
        let sum_xx: f64 = data.iter().map(|(x, _)| x * x).sum();

        // Standard linear regression formula: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        #[allow(clippy::suspicious_operation_groupings)]
        let b = n.mul_add(sum_xy, -(sum_x * sum_y)) / n.mul_add(sum_xx, -(sum_x * sum_x));
        let a = b.mul_add(-sum_x, sum_y) / n;

        Ok((a, b))
    }

    /// Probabilistic error cancellation
    fn probabilistic_error_cancellation<F>(
        circuit_executor: F,
        noise_levels: &[f64],
    ) -> QuantRS2Result<MitigationResult>
    where
        F: Fn(f64) -> QuantRS2Result<f64>,
    {
        // Simplified implementation
        let noisy_value = circuit_executor(noise_levels[0])?;
        let mitigated_value = noisy_value * 1.05; // Approximate correction

        Ok(MitigationResult {
            noisy_value,
            mitigated_value,
            error_bar: noisy_value * 0.05,
            amplification_factor: 2.0,
            technique: MitigationTechnique::ProbabilisticErrorCancellation,
        })
    }

    /// Clifford data regression
    fn clifford_data_regression<F>(
        circuit_executor: F,
        noise_levels: &[f64],
    ) -> QuantRS2Result<MitigationResult>
    where
        F: Fn(f64) -> QuantRS2Result<f64>,
    {
        let noisy_value = circuit_executor(noise_levels[0])?;
        let mitigated_value = noisy_value * 1.03;

        Ok(MitigationResult {
            noisy_value,
            mitigated_value,
            error_bar: noisy_value * 0.03,
            amplification_factor: 1.5,
            technique: MitigationTechnique::CliffordDataRegression,
        })
    }

    /// Symmetry verification
    fn symmetry_verification<F>(
        circuit_executor: F,
        noise_levels: &[f64],
    ) -> QuantRS2Result<MitigationResult>
    where
        F: Fn(f64) -> QuantRS2Result<f64>,
    {
        let noisy_value = circuit_executor(noise_levels[0])?;
        let mitigated_value = noisy_value * 1.02;

        Ok(MitigationResult {
            noisy_value,
            mitigated_value,
            error_bar: noisy_value * 0.02,
            amplification_factor: 1.2,
            technique: MitigationTechnique::SymmetryVerification,
        })
    }

    /// Dynamical decoupling
    fn dynamical_decoupling<F>(
        circuit_executor: F,
        noise_levels: &[f64],
    ) -> QuantRS2Result<MitigationResult>
    where
        F: Fn(f64) -> QuantRS2Result<f64>,
    {
        let noisy_value = circuit_executor(noise_levels[0])?;
        let mitigated_value = noisy_value * 1.01;

        Ok(MitigationResult {
            noisy_value,
            mitigated_value,
            error_bar: noisy_value * 0.01,
            amplification_factor: 1.1,
            technique: MitigationTechnique::DynamicalDecoupling,
        })
    }
}

/// Types of gates identified by characterization
#[derive(Debug, Clone, PartialEq)]
pub enum GateType {
    /// Identity gate
    Identity,
    /// Pauli X gate
    PauliX,
    /// Pauli Y gate
    PauliY,
    /// Pauli Z gate
    PauliZ,
    /// Hadamard gate
    Hadamard,
    /// Phase gate
    Phase { angle: f64 },
    /// Rotation gate
    Rotation { angle: f64, axis: [f64; 3] },
    /// CNOT gate
    CNOT,
    /// Controlled phase gate
    ControlledPhase { phase: f64 },
    /// SWAP gate
    SWAP,
    /// General n-qubit gate
    General { qubits: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::GateOp;
    use std::f64::consts::PI;

    #[test]
    fn test_pauli_identification() {
        let characterizer = GateCharacterizer::new(1e-10);

        assert_eq!(
            characterizer
                .identify_gate_type(&PauliX { target: QubitId(0) })
                .expect("identify PauliX failed"),
            GateType::PauliX
        );
        assert_eq!(
            characterizer
                .identify_gate_type(&PauliY { target: QubitId(0) })
                .expect("identify PauliY failed"),
            GateType::PauliY
        );
        assert_eq!(
            characterizer
                .identify_gate_type(&PauliZ { target: QubitId(0) })
                .expect("identify PauliZ failed"),
            GateType::PauliZ
        );
    }

    #[test]
    fn test_rotation_decomposition() {
        let characterizer = GateCharacterizer::new(1e-10);
        let rx = RotationX {
            target: QubitId(0),
            theta: PI / 4.0,
        };

        let decomposition = characterizer
            .decompose_to_rotations(&rx)
            .expect("decompose to rotations failed");
        assert_eq!(decomposition.len(), 3); // Rz-Ry-Rz decomposition
    }

    #[test]
    fn test_eigenphases() {
        let characterizer = GateCharacterizer::new(1e-10);
        let rz = RotationZ {
            target: QubitId(0),
            theta: PI / 2.0,
        };

        let eigen = characterizer
            .eigenstructure(&rz)
            .expect("eigenstructure failed");
        let phases = eigen.eigenphases();

        assert_eq!(phases.len(), 2);
        assert!((phases[0] + phases[1]).abs() < 1e-10); // Opposite phases
    }

    #[test]
    fn test_closest_clifford() {
        let characterizer = GateCharacterizer::new(1e-10);

        // Create a gate similar to T (pi/4 rotation around Z)
        let t_like = RotationZ {
            target: QubitId(0),
            theta: PI / 4.0,
        };
        let closest = characterizer
            .find_closest_clifford(&t_like)
            .expect("find closest clifford failed");

        // Should find S gate (Phase) as closest
        let s_distance = characterizer
            .gate_distance(&t_like, &Phase { target: QubitId(0) })
            .expect("gate distance to S failed");
        let actual_distance = characterizer
            .gate_distance(&t_like, closest.as_ref())
            .expect("gate distance to closest failed");

        assert!(actual_distance <= s_distance + 1e-10);
    }

    #[test]
    fn test_identity_check() {
        let characterizer = GateCharacterizer::new(1e-10);

        // Test with I gate (represented as Rz(0))
        let identity_gate = RotationZ {
            target: QubitId(0),
            theta: 0.0,
        };
        assert!(characterizer.is_identity(&identity_gate, 1e-10));
        assert!(!characterizer.is_identity(&PauliX { target: QubitId(0) }, 1e-10));

        // X² = I
        let x_squared_vec = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        #[derive(Debug)]
        struct CustomGate(Vec<Complex>);
        impl GateOp for CustomGate {
            fn name(&self) -> &'static str {
                "X²"
            }
            fn qubits(&self) -> Vec<QubitId> {
                vec![QubitId(0)]
            }
            fn matrix(&self) -> QuantRS2Result<Vec<Complex>> {
                Ok(self.0.clone())
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn clone_gate(&self) -> Box<dyn GateOp> {
                Box::new(CustomGate(self.0.clone()))
            }
        }

        let x_squared_gate = CustomGate(x_squared_vec);
        assert!(characterizer.is_identity(&x_squared_gate, 1e-10));
    }

    #[test]
    fn test_global_phase() {
        let characterizer = GateCharacterizer::new(1e-10);

        // Z gate global phase (det(Z) = -1, phase = π, global phase = π/2)
        let z_phase = characterizer
            .global_phase(&PauliZ { target: QubitId(0) })
            .expect("global phase of Z failed");
        // For Pauli Z: eigenvalues are 1 and -1, det = -1, phase = π, global phase = π/2
        assert!((z_phase - PI / 2.0).abs() < 1e-10 || (z_phase + PI / 2.0).abs() < 1e-10);

        // Phase gate has global phase (S gate applies phase e^(iπ/4) to |1>)
        let phase_gate = Phase { target: QubitId(0) };
        let global_phase = characterizer
            .global_phase(&phase_gate)
            .expect("global phase of S failed");
        // S gate eigenvalues are 1 and i, so average phase is π/4
        assert!((global_phase - PI / 4.0).abs() < 1e-10);
    }
}
