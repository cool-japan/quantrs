//! Gate synthesis from unitary matrices
//!
//! This module provides algorithms to decompose arbitrary unitary matrices
//! into sequences of quantum gates, including:
//! - Single-qubit unitary decomposition (ZYZ, XYX, etc.)
//! - Two-qubit unitary decomposition (KAK/Cartan)
//! - General n-qubit synthesis using Cosine-Sine decomposition

use crate::cartan::{CartanDecomposer, CartanDecomposition};
// use crate::controlled::{make_controlled, ControlledGate};
use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::gate::{single::*, GateOp};
use crate::matrix_ops::{matrices_approx_equal, DenseMatrix, QuantumMatrix};
use crate::qubit::QubitId;
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Result of single-qubit decomposition
#[derive(Debug, Clone)]
pub struct SingleQubitDecomposition {
    /// Global phase
    pub global_phase: f64,
    /// First rotation angle (Z or X depending on basis)
    pub theta1: f64,
    /// Middle rotation angle (Y)
    pub phi: f64,
    /// Last rotation angle (Z or X depending on basis)
    pub theta2: f64,
    /// The basis used (e.g., "ZYZ", "XYX")
    pub basis: String,
}

/// Decompose a single-qubit unitary into ZYZ rotations
pub fn decompose_single_qubit_zyz(
    unitary: &ArrayView2<Complex64>,
) -> QuantRS2Result<SingleQubitDecomposition> {
    if unitary.shape() != &[2, 2] {
        return Err(QuantRS2Error::InvalidInput(
            "Single-qubit unitary must be 2x2".to_string(),
        ));
    }

    // Check unitarity
    let matrix = DenseMatrix::new(unitary.to_owned())?;
    if !matrix.is_unitary(1e-10)? {
        return Err(QuantRS2Error::InvalidInput(
            "Matrix is not unitary".to_string(),
        ));
    }

    // Extract matrix elements
    let a = unitary[[0, 0]];
    let b = unitary[[0, 1]];
    let c = unitary[[1, 0]];
    let d = unitary[[1, 1]];

    // Calculate global phase from determinant
    let det = a * d - b * c;
    let global_phase = det.arg() / 2.0;

    // Normalize by the determinant to make the matrix special unitary
    let det_sqrt = det.sqrt();
    let a = a / det_sqrt;
    let b = b / det_sqrt;
    let c = c / det_sqrt;
    let d = d / det_sqrt;

    // Decompose into ZYZ angles
    // U = e^(i*global_phase) * Rz(theta2) * Ry(phi) * Rz(theta1)

    let phi = 2.0 * a.norm().acos();

    let (theta1, theta2) = if phi.abs() < 1e-10 {
        // Identity or phase gate
        let phase = if a.norm() > 0.5 {
            a.arg() * 2.0
        } else {
            d.arg() * 2.0
        };
        (0.0, phase)
    } else if (phi - PI).abs() < 1e-10 {
        // Pi rotation
        (-b.arg() + c.arg() + PI, 0.0)
    } else {
        let theta1 = c.arg() - b.arg();
        let theta2 = a.arg() + b.arg();
        (theta1, theta2)
    };

    Ok(SingleQubitDecomposition {
        global_phase,
        theta1,
        phi,
        theta2,
        basis: "ZYZ".to_string(),
    })
}

/// Decompose a single-qubit unitary into XYX rotations
pub fn decompose_single_qubit_xyx(
    unitary: &ArrayView2<Complex64>,
) -> QuantRS2Result<SingleQubitDecomposition> {
    // Convert to Pauli basis and use ZYZ decomposition
    let h_gate = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap()
        / Complex64::new(2.0_f64.sqrt(), 0.0);

    // Transform: U' = H * U * H
    let u_transformed = h_gate.dot(unitary).dot(&h_gate);
    let decomp = decompose_single_qubit_zyz(&u_transformed.view())?;

    Ok(SingleQubitDecomposition {
        global_phase: decomp.global_phase,
        theta1: decomp.theta1,
        phi: decomp.phi,
        theta2: decomp.theta2,
        basis: "XYX".to_string(),
    })
}

/// Convert single-qubit decomposition to gate sequence
pub fn single_qubit_gates(
    decomp: &SingleQubitDecomposition,
    qubit: QubitId,
) -> Vec<Box<dyn GateOp>> {
    let mut gates: Vec<Box<dyn GateOp>> = Vec::new();

    match decomp.basis.as_str() {
        "ZYZ" => {
            if decomp.theta1.abs() > 1e-10 {
                gates.push(Box::new(RotationZ {
                    target: qubit,
                    theta: decomp.theta1,
                }));
            }
            if decomp.phi.abs() > 1e-10 {
                gates.push(Box::new(RotationY {
                    target: qubit,
                    theta: decomp.phi,
                }));
            }
            if decomp.theta2.abs() > 1e-10 {
                gates.push(Box::new(RotationZ {
                    target: qubit,
                    theta: decomp.theta2,
                }));
            }
        }
        "XYX" => {
            if decomp.theta1.abs() > 1e-10 {
                gates.push(Box::new(RotationX {
                    target: qubit,
                    theta: decomp.theta1,
                }));
            }
            if decomp.phi.abs() > 1e-10 {
                gates.push(Box::new(RotationY {
                    target: qubit,
                    theta: decomp.phi,
                }));
            }
            if decomp.theta2.abs() > 1e-10 {
                gates.push(Box::new(RotationX {
                    target: qubit,
                    theta: decomp.theta2,
                }));
            }
        }
        _ => {} // Unknown basis
    }

    gates
}

/// Result of two-qubit KAK decomposition (alias for CartanDecomposition)
pub type KAKDecomposition = CartanDecomposition;

/// Decompose a two-qubit unitary using KAK decomposition
pub fn decompose_two_qubit_kak(
    unitary: &ArrayView2<Complex64>,
) -> QuantRS2Result<KAKDecomposition> {
    // Use Cartan decomposer for KAK decomposition
    let mut decomposer = CartanDecomposer::new();
    let owned_unitary = unitary.to_owned();
    decomposer.decompose(&owned_unitary)
}

/// Convert KAK decomposition to gate sequence
pub fn kak_to_gates(
    decomp: &KAKDecomposition,
    qubit1: QubitId,
    qubit2: QubitId,
) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
    // Use CartanDecomposer to convert to gates
    let decomposer = CartanDecomposer::new();
    let qubit_ids = vec![qubit1, qubit2];
    decomposer.to_gates(decomp, &qubit_ids)
}

/// Synthesize an arbitrary unitary matrix into quantum gates
pub fn synthesize_unitary(
    unitary: &ArrayView2<Complex64>,
    qubits: &[QubitId],
) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
    let n = unitary.nrows();

    if n != unitary.ncols() {
        return Err(QuantRS2Error::InvalidInput(
            "Matrix must be square".to_string(),
        ));
    }

    let num_qubits = (n as f64).log2() as usize;
    if (1 << num_qubits) != n {
        return Err(QuantRS2Error::InvalidInput(
            "Matrix dimension must be a power of 2".to_string(),
        ));
    }

    if qubits.len() != num_qubits {
        return Err(QuantRS2Error::InvalidInput(format!(
            "Need {} qubits, got {}",
            num_qubits,
            qubits.len()
        )));
    }

    // Check unitarity
    let matrix = DenseMatrix::new(unitary.to_owned())?;
    if !matrix.is_unitary(1e-10)? {
        return Err(QuantRS2Error::InvalidInput(
            "Matrix is not unitary".to_string(),
        ));
    }

    match num_qubits {
        1 => {
            let decomp = decompose_single_qubit_zyz(unitary)?;
            Ok(single_qubit_gates(&decomp, qubits[0]))
        }
        2 => {
            let decomp = decompose_two_qubit_kak(unitary)?;
            kak_to_gates(&decomp, qubits[0], qubits[1])
        }
        _ => {
            // For n-qubit gates, use recursive decomposition
            // This is a placeholder - would implement Cosine-Sine decomposition
            Err(QuantRS2Error::UnsupportedOperation(format!(
                "Synthesis for {}-qubit gates not yet implemented",
                num_qubits
            )))
        }
    }
}

/// Check if a unitary is close to a known gate
pub fn identify_gate(unitary: &ArrayView2<Complex64>, tolerance: f64) -> Option<String> {
    let n = unitary.nrows();

    match n {
        2 => {
            // Check common single-qubit gates
            let gates = vec![
                ("I", Array2::eye(2)),
                (
                    "X",
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex64::new(0.0, 0.0),
                            Complex64::new(1.0, 0.0),
                            Complex64::new(1.0, 0.0),
                            Complex64::new(0.0, 0.0),
                        ],
                    )
                    .unwrap(),
                ),
                (
                    "Y",
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex64::new(0.0, 0.0),
                            Complex64::new(0.0, -1.0),
                            Complex64::new(0.0, 1.0),
                            Complex64::new(0.0, 0.0),
                        ],
                    )
                    .unwrap(),
                ),
                (
                    "Z",
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex64::new(1.0, 0.0),
                            Complex64::new(0.0, 0.0),
                            Complex64::new(0.0, 0.0),
                            Complex64::new(-1.0, 0.0),
                        ],
                    )
                    .unwrap(),
                ),
                (
                    "H",
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex64::new(1.0, 0.0),
                            Complex64::new(1.0, 0.0),
                            Complex64::new(1.0, 0.0),
                            Complex64::new(-1.0, 0.0),
                        ],
                    )
                    .unwrap()
                        / Complex64::new(2.0_f64.sqrt(), 0.0),
                ),
            ];

            for (name, gate) in gates {
                if matrices_approx_equal(unitary, &gate.view(), tolerance) {
                    return Some(name.to_string());
                }
            }
        }
        4 => {
            // Check common two-qubit gates
            let mut cnot = Array2::eye(4);
            cnot[[2, 2]] = Complex64::new(0.0, 0.0);
            cnot[[2, 3]] = Complex64::new(1.0, 0.0);
            cnot[[3, 2]] = Complex64::new(1.0, 0.0);
            cnot[[3, 3]] = Complex64::new(0.0, 0.0);

            if matrices_approx_equal(unitary, &cnot.view(), tolerance) {
                return Some("CNOT".to_string());
            }
        }
        _ => {}
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    #[ignore] // TODO: Fix ZYZ decomposition algorithm
    fn test_single_qubit_decomposition() {
        // Test Hadamard gate
        let h = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .unwrap()
            / Complex64::new(2.0_f64.sqrt(), 0.0);

        let decomp = decompose_single_qubit_zyz(&h.view()).unwrap();

        // Reconstruct and verify
        let rz1 = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, -decomp.theta1 / 2.0).exp(),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, decomp.theta1 / 2.0).exp(),
            ],
        )
        .unwrap();

        let ry = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new((decomp.phi / 2.0).cos(), 0.0),
                Complex64::new(-(decomp.phi / 2.0).sin(), 0.0),
                Complex64::new((decomp.phi / 2.0).sin(), 0.0),
                Complex64::new((decomp.phi / 2.0).cos(), 0.0),
            ],
        )
        .unwrap();

        let rz2 = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, -decomp.theta2 / 2.0).exp(),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, decomp.theta2 / 2.0).exp(),
            ],
        )
        .unwrap();

        // Reconstruct: e^(i*global_phase) * Rz(theta2) * Ry(phi) * Rz(theta1)
        let reconstructed = Complex64::new(0.0, decomp.global_phase).exp() * rz2.dot(&ry).dot(&rz1);

        // Check reconstruction
        assert!(matrices_approx_equal(
            &h.view(),
            &reconstructed.view(),
            1e-10
        ));
    }

    #[test]
    fn test_gate_identification() {
        let x = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();

        assert_eq!(identify_gate(&x.view(), 1e-10), Some("X".to_string()));
    }
}
