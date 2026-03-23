//! Gate eigenstructure and characterization tools

use crate::{
    eigensolve::eigen_decompose_unitary,
    error::{QuantRS2Error, QuantRS2Result},
    gate::{single::*, GateOp},
    qubit::QubitId,
};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64 as Complex;

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
        let phase_diff = (self.eigenvalues[0] / self.eigenvalues[1]).arg();
        Some(phase_diff.abs())
    }

    /// Get the rotation axis for single-qubit gates
    pub fn rotation_axis(&self, tolerance: f64) -> Option<[f64; 3]> {
        if self.eigenvalues.len() != 2 {
            return None;
        }

        let v0 = self.eigenvectors.column(0);
        let v1 = self.eigenvectors.column(1);

        let bloch0 = eigenvector_to_bloch(&v0.to_owned());
        let bloch1 = eigenvector_to_bloch(&v1.to_owned());

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

    let v0 = v[0];
    let v1 = v[1];
    let rho00 = (v0 * v0.conj()).re;
    let rho11 = (v1 * v1.conj()).re;
    let rho01 = v0 * v1.conj();

    [2.0 * rho01.re, -2.0 * rho01.im, rho00 - rho11]
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

        let mut matrix = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                matrix[(i, j)] = matrix_vec[i * n + j];
            }
        }

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
        if self.is_pauli_gate(eigen) {
            return Ok(self.identify_pauli_type(eigen));
        }

        if let Some(angle) = eigen.rotation_angle() {
            if let Some(axis) = eigen.rotation_axis(self.tolerance) {
                return Ok(GateType::Rotation { angle, axis });
            }
        }

        if self.is_hadamard(eigen) {
            return Ok(GateType::Hadamard);
        }

        Ok(GateType::General { qubits: 1 })
    }

    /// Identify two-qubit gate type
    fn identify_two_qubit_gate(&self, eigen: &GateEigenstructure) -> QuantRS2Result<GateType> {
        if self.is_cnot(eigen) {
            return Ok(GateType::CNOT);
        }

        if self.is_controlled_phase(eigen) {
            if let Some(phase) = self.extract_controlled_phase(eigen) {
                return Ok(GateType::ControlledPhase { phase });
            }
        }

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
        let tolerance = self.tolerance;

        if (matrix[(0, 1)] - Complex::new(1.0, 0.0)).norm() < tolerance
            && (matrix[(1, 0)] - Complex::new(1.0, 0.0)).norm() < tolerance
            && matrix[(0, 0)].norm() < tolerance
            && matrix[(1, 1)].norm() < tolerance
        {
            GateType::PauliX
        } else if (matrix[(0, 1)] - Complex::new(0.0, -1.0)).norm() < tolerance
            && (matrix[(1, 0)] - Complex::new(0.0, 1.0)).norm() < tolerance
            && matrix[(0, 0)].norm() < tolerance
            && matrix[(1, 1)].norm() < tolerance
        {
            GateType::PauliY
        } else if (matrix[(0, 0)] - Complex::new(1.0, 0.0)).norm() < tolerance
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
        eigen.eigenvalues.iter().all(|&lambda| {
            (lambda - Complex::new(1.0, 0.0)).norm() < self.tolerance
                || (lambda + Complex::new(1.0, 0.0)).norm() < self.tolerance
        })
    }

    /// Check if gate is CNOT
    fn is_cnot(&self, eigen: &GateEigenstructure) -> bool {
        eigen.eigenvalues.len() == 4
            && eigen.eigenvalues.iter().all(|&lambda| {
                (lambda - Complex::new(1.0, 0.0)).norm() < self.tolerance
                    || (lambda + Complex::new(1.0, 0.0)).norm() < self.tolerance
            })
    }

    /// Check if gate is a controlled phase gate
    fn is_controlled_phase(&self, eigen: &GateEigenstructure) -> bool {
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
        let matrix = &eigen.matrix;

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

        let target = QubitId(0);
        let clifford_gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(PauliX { target }),
            Box::new(PauliY { target }),
            Box::new(PauliZ { target }),
            Box::new(Hadamard { target }),
            Box::new(Phase { target }),
        ];

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

        let det = eigen
            .eigenvalues
            .iter()
            .fold(Complex::new(1.0, 0.0), |acc, &lambda| acc * lambda);
        let n = eigen.eigenvalues.len() as f64;
        Ok(det.arg() / n)
    }
}
