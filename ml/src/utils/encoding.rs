//! Quantum encoding schemes for feature mapping

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;

use super::*;
/// Amplitude encoding: encode classical data into quantum amplitudes
pub fn amplitude_encode(data: &Array1<f64>) -> Result<Array1<Complex64>> {
    let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-10 {
        return Err(MLError::ComputationError(
            "Cannot amplitude encode zero vector".to_string(),
        ));
    }
    let encoded = data.mapv(|x| Complex64::new(x / norm, 0.0));
    Ok(encoded)
}
/// Angle encoding: encode data as rotation angles
pub fn angle_encode(data: &Array1<f64>, scale: f64) -> Result<Array1<f64>> {
    let pi = std::f64::consts::PI;
    let encoded = data.mapv(|x| (x * scale).rem_euclid(2.0 * pi));
    Ok(encoded)
}
/// Basis encoding: encode integer data as computational basis states
/// Each integer is encoded as a binary string representing a basis state
pub fn basis_encode(data: &Array1<usize>, num_qubits: usize) -> Result<Array2<u8>> {
    let max_val = 1 << num_qubits;
    for &val in data.iter() {
        if val >= max_val {
            return Err(MLError::InvalidInput(format!(
                "Value {} exceeds maximum {} for {} qubits",
                val,
                max_val - 1,
                num_qubits
            )));
        }
    }
    let mut encoded = Array2::zeros((data.len(), num_qubits));
    for (i, &val) in data.iter().enumerate() {
        for j in 0..num_qubits {
            encoded[(i, num_qubits - 1 - j)] = ((val >> j) & 1) as u8;
        }
    }
    Ok(encoded)
}
/// Product encoding: encode data using tensor product of single-qubit rotations
/// Each feature is encoded using RY rotation on a separate qubit
pub fn product_encode(data: &Array1<f64>) -> Result<Array1<f64>> {
    let pi = std::f64::consts::PI;
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    if range < 1e-10 {
        return Ok(Array1::from_elem(data.len(), pi / 2.0));
    }
    let encoded = data.mapv(|x| ((x - min_val) / range) * pi);
    Ok(encoded)
}
/// Dense angle encoding: encode multiple features per qubit using multiple rotations
/// Uses RY and RZ rotations to encode 2 features per qubit
pub fn dense_angle_encode(data: &Array1<f64>) -> Result<Array1<f64>> {
    let pi = std::f64::consts::PI;
    let encoded = data.mapv(|x| {
        let normalized = (x.atan() / (pi / 2.0) + 1.0) * pi;
        normalized
    });
    Ok(encoded)
}
/// IQP (Instantaneous Quantum Polynomial) encoding
/// Encodes data using diagonal unitary gates with polynomial feature map
pub fn iqp_encode(data: &Array1<f64>, degree: usize) -> Result<Array1<f64>> {
    let n = data.len();
    let mut encoded = Vec::new();
    for &x in data.iter() {
        encoded.push(x);
    }
    if degree >= 2 {
        for i in 0..n {
            for j in i..n {
                encoded.push(data[i] * data[j]);
            }
        }
    }
    if degree >= 3 {
        for i in 0..n {
            for j in i..n {
                for k in j..n {
                    encoded.push(data[i] * data[j] * data[k]);
                }
            }
        }
    }
    Ok(Array1::from_vec(encoded))
}
/// Pauli feature map encoding
/// Encodes data using Pauli rotation gates with entanglement
pub fn pauli_feature_map_encode(data: &Array1<f64>, reps: usize) -> Result<Array1<f64>> {
    let pi = std::f64::consts::PI;
    let n = data.len();
    let mut encoded = Vec::new();
    for _ in 0..reps {
        for &x in data.iter() {
            encoded.push(2.0 * x);
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let angle = (pi - data[i]) * (pi - data[j]);
                encoded.push(2.0 * angle);
            }
        }
    }
    Ok(Array1::from_vec(encoded))
}
