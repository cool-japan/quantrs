//! Discrete Wigner function visualization for n=1 and n=2 qubit states.
//!
//! Implements Wootters' discrete phase-space Wigner function using
//! the displacement-operator basis {I, X, Z, Y} for n=1 and its
//! tensor-product extension for n=2.
//!
//! # Scope
//! Strictly limited to n=1 (4 phase-space points) and n=2 (16 points).
//! Returns an error for n ≥ 3 because the GF(2^n) construction is
//! research-grade and has multiple inequivalent definitions.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use serde_json::{json, Value};

use crate::error::{QuantRS2Error, QuantRS2Result};

/// Pauli matrices as 2×2 complex arrays.
fn pauli_i() -> Array2<Complex64> {
    let mut m = Array2::zeros((2, 2));
    m[[0, 0]] = Complex64::new(1.0, 0.0);
    m[[1, 1]] = Complex64::new(1.0, 0.0);
    m
}

fn pauli_x() -> Array2<Complex64> {
    let mut m = Array2::zeros((2, 2));
    m[[0, 1]] = Complex64::new(1.0, 0.0);
    m[[1, 0]] = Complex64::new(1.0, 0.0);
    m
}

fn pauli_y() -> Array2<Complex64> {
    let mut m = Array2::zeros((2, 2));
    m[[0, 1]] = Complex64::new(0.0, -1.0);
    m[[1, 0]] = Complex64::new(0.0, 1.0);
    m
}

fn pauli_z() -> Array2<Complex64> {
    let mut m = Array2::zeros((2, 2));
    m[[0, 0]] = Complex64::new(1.0, 0.0);
    m[[1, 1]] = Complex64::new(-1.0, 0.0);
    m
}

/// Phase-space point operator A(q, p) for a single qubit.
///
/// Uses the Wootters (1987) definition where the 4 operators form a
/// complete orthogonal set satisfying:
///   Σ_{q,p} A(q,p) = 2·I   (enabling Wigner normalization Σ W = 1)
///   Tr(A(q,p)) = 1
///
/// The operators are:
///   A(q,p) = ½(I + (-1)^p X + (-1)^{q+p} Y + (-1)^q Z)
///
/// Explicitly:
///   A(0,0) = ½(I + X + Y + Z)
///   A(1,0) = ½(I + X - Y - Z)
///   A(0,1) = ½(I - X - Y + Z)
///   A(1,1) = ½(I - X + Y - Z)
fn displacement_op_1(q: usize, p: usize) -> Array2<Complex64> {
    let i = pauli_i();
    let x = pauli_x();
    let y = pauli_y();
    let z = pauli_z();

    let sx = if p % 2 == 0 { 1.0f64 } else { -1.0f64 };
    let sy = if (q + p) % 2 == 0 { 1.0f64 } else { -1.0f64 };
    let sz = if q % 2 == 0 { 1.0f64 } else { -1.0f64 };

    let half = 0.5;
    let mut result = Array2::zeros((2, 2));
    for row in 0..2 {
        for col in 0..2 {
            result[[row, col]] = Complex64::new(half, 0.0)
                * (i[[row, col]]
                    + Complex64::new(sx, 0.0) * x[[row, col]]
                    + Complex64::new(sy, 0.0) * y[[row, col]]
                    + Complex64::new(sz, 0.0) * z[[row, col]]);
        }
    }
    result
}

/// Tensor product of two 2×2 matrices → 4×4 matrix.
fn tensor_product_2x2(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let mut out = Array2::zeros((4, 4));
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    out[[2 * i + k, 2 * j + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }
    out
}

/// Trace of a square matrix.
fn matrix_trace(m: &Array2<Complex64>) -> Complex64 {
    let n = m.nrows().min(m.ncols());
    (0..n).map(|i| m[[i, i]]).sum()
}

/// Compute the density matrix ρ = |ψ⟩⟨ψ| from a state vector.
fn density_matrix(state: &Array1<Complex64>) -> Array2<Complex64> {
    let d = state.len();
    let mut rho = Array2::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            rho[[i, j]] = state[i] * state[j].conj();
        }
    }
    rho
}

/// Matrix–matrix multiply for square complex matrices.
fn mat_mul(a: &Array2<Complex64>, b: &Array2<Complex64>) -> QuantRS2Result<Array2<Complex64>> {
    let n = a.nrows();
    if a.ncols() != b.nrows() || b.ncols() != n {
        return Err(QuantRS2Error::InvalidInput(
            "Incompatible matrix dimensions for multiplication".to_string(),
        ));
    }
    let mut out = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut s = Complex64::new(0.0, 0.0);
            for k in 0..n {
                s += a[[i, k]] * b[[k, j]];
            }
            out[[i, j]] = s;
        }
    }
    Ok(out)
}

/// Compute the discrete Wigner function for n=1.
///
/// W(q,p) = (1/d) Tr(A(q,p) ρ)  where d=2.
///
/// Returns a 2×2 array indexed by (q, p) ∈ {0,1}².
fn wigner_n1(state: &Array1<Complex64>) -> QuantRS2Result<[[f64; 2]; 2]> {
    let rho = density_matrix(state);
    let mut w = [[0.0f64; 2]; 2];
    for q in 0..2usize {
        for p in 0..2usize {
            let a = displacement_op_1(q, p);
            let ap = mat_mul(&a, &rho)?;
            let tr = matrix_trace(&ap);
            w[q][p] = tr.re / 2.0; // d = 2
        }
    }
    Ok(w)
}

/// Compute the discrete Wigner function for n=2.
///
/// Uses the tensor-product displacement operator:
/// A⊗(q₁q₂, p₁p₂) = A₁(q₁,p₁) ⊗ A₂(q₂,p₂)
///
/// W(q₁,q₂; p₁,p₂) = (1/4) Tr(A⊗ ρ)
///
/// Returns a 4×4 array indexed by (q, p) ∈ {0..3}×{0..3},
/// where q = 2·q₁ + q₂ and p = 2·p₁ + p₂.
fn wigner_n2(state: &Array1<Complex64>) -> QuantRS2Result<[[f64; 4]; 4]> {
    let rho = density_matrix(state);
    let mut w = [[0.0f64; 4]; 4];

    for q in 0..4usize {
        let q1 = q >> 1;
        let q2 = q & 1;
        for p in 0..4usize {
            let p1 = p >> 1;
            let p2 = p & 1;

            let a1 = displacement_op_1(q1, p1);
            let a2 = displacement_op_1(q2, p2);
            let a_tensor = tensor_product_2x2(&a1, &a2);
            let ap = mat_mul(&a_tensor, &rho)?;
            let tr = matrix_trace(&ap);
            w[q][p] = tr.re / 4.0; // d = 4
        }
    }
    Ok(w)
}

/// Discrete Wigner function for n=1 (4-point) or n=2 (16-point) states.
///
/// Returns an `Err` for n ≥ 3 — the GF(2^n) construction is out of
/// scope for this version.
pub fn wigner_plotly_json(state: &Array1<Complex64>, n_qubits: usize) -> QuantRS2Result<String> {
    match n_qubits {
        0 => Err(QuantRS2Error::InvalidInput(
            "n_qubits must be ≥ 1".to_string(),
        )),
        1 => {
            if state.len() != 2 {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "State length {} does not match 2^1 = 2",
                    state.len()
                )));
            }
            let w = wigner_n1(state)?;
            build_wigner_heatmap_n1(&w)
        }
        2 => {
            if state.len() != 4 {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "State length {} does not match 2^2 = 4",
                    state.len()
                )));
            }
            let w = wigner_n2(state)?;
            build_wigner_heatmap_n2(&w)
        }
        _ => Err(QuantRS2Error::UnsupportedOperation(format!(
            "Discrete Wigner for n={} requires GF(2^n) phase space — \
             only n=1 and n=2 are supported in this version",
            n_qubits
        ))),
    }
}

/// Build a Plotly heatmap for the n=1 Wigner function (2×2 grid).
fn build_wigner_heatmap_n1(w: &[[f64; 2]; 2]) -> QuantRS2Result<String> {
    let labels = ["(0,0)", "(1,0)", "(0,1)", "(1,1)"];

    // Arrange as a 2×2 grid: rows = q, columns = p
    let z: Vec<Vec<f64>> = (0..2).map(|q| (0..2).map(|p| w[q][p]).collect()).collect();

    let x_labels: Vec<&str> = vec!["p=0", "p=1"];
    let y_labels: Vec<&str> = vec!["q=0", "q=1"];

    let hovertext: Vec<Vec<String>> = (0..2)
        .map(|q| {
            (0..2)
                .map(|p| format!("{} W={:.4}", labels[2 * q + p], w[q][p]))
                .collect()
        })
        .collect();

    let figure = json!({
        "data": [{
            "type": "heatmap",
            "z": z,
            "x": x_labels,
            "y": y_labels,
            "colorscale": "RdBu",
            "zmid": 0.0,
            "text": hovertext,
            "hoverinfo": "text",
            "colorbar": {"title": "W(q,p)"}
        }],
        "layout": {
            "title": "Discrete Wigner Function (n=1)",
            "xaxis": {"title": "p"},
            "yaxis": {"title": "q"},
            "height": 450
        }
    });

    serde_json::to_string(&figure).map_err(QuantRS2Error::from)
}

/// Build a Plotly heatmap for the n=2 Wigner function (4×4 grid).
fn build_wigner_heatmap_n2(w: &[[f64; 4]; 4]) -> QuantRS2Result<String> {
    let coord_labels = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"];

    let z: Vec<Vec<f64>> = (0..4).map(|q| (0..4).map(|p| w[q][p]).collect()).collect();

    let x_labels: Vec<String> = (0..4usize)
        .map(|p| format!("p={}", coord_labels[p]))
        .collect();
    let y_labels: Vec<String> = (0..4usize)
        .map(|q| format!("q={}", coord_labels[q]))
        .collect();

    let hovertext: Vec<Vec<String>> = (0..4)
        .map(|q| {
            (0..4)
                .map(|p| {
                    format!(
                        "q={} p={} W={:.4}",
                        coord_labels[q], coord_labels[p], w[q][p]
                    )
                })
                .collect()
        })
        .collect();

    let figure = json!({
        "data": [{
            "type": "heatmap",
            "z": z,
            "x": x_labels,
            "y": y_labels,
            "colorscale": "RdBu",
            "zmid": 0.0,
            "text": hovertext,
            "hoverinfo": "text",
            "colorbar": {"title": "W(q,p)"}
        }],
        "layout": {
            "title": "Discrete Wigner Function (n=2)",
            "xaxis": {"title": "p (phase-space momentum)"},
            "yaxis": {"title": "q (phase-space position)"},
            "height": 550
        }
    });

    serde_json::to_string(&figure).map_err(QuantRS2Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::Complex64;

    fn state_zero_1q() -> Array1<Complex64> {
        Array1::from(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)])
    }

    fn state_bell_2q() -> Array1<Complex64> {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        Array1::from(vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ])
    }

    #[test]
    fn test_wigner_n1_zero_state() {
        let state = state_zero_1q();
        let w = wigner_n1(&state).expect("wigner_n1 failed");

        // |0⟩: ρ = [[1,0],[0,0]]
        // W(0,0) = (1/2) Tr(I ρ) = 1/2
        assert!(
            (w[0][0] - 0.5).abs() < 1e-10,
            "W(0,0) should be 0.5, got {}",
            w[0][0]
        );

        // Normalization: Σ W = 1
        let sum: f64 = w.iter().flat_map(|row| row.iter()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Wigner normalization should be 1, got {}",
            sum
        );
    }

    #[test]
    fn test_wigner_n2_normalization() {
        let state = state_bell_2q();
        let w = wigner_n2(&state).expect("wigner_n2 failed");

        let sum: f64 = w.iter().flat_map(|row| row.iter()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "n=2 Wigner normalization should be 1, got {}",
            sum
        );
    }

    #[test]
    fn test_wigner_n3_returns_err() {
        // Build a valid 3-qubit state (|000⟩)
        let mut state = Array1::zeros(8);
        state[0] = Complex64::new(1.0, 0.0);
        let result = wigner_plotly_json(&state, 3);
        assert!(result.is_err(), "n=3 should return Err");
        if let Err(e) = result {
            assert!(
                matches!(e, QuantRS2Error::UnsupportedOperation(_)),
                "Error should be UnsupportedOperation, got {:?}",
                e
            );
        }
    }

    #[test]
    fn test_wigner_json_valid() {
        let state = state_zero_1q();
        let json_str = wigner_plotly_json(&state, 1).expect("Wigner JSON failed");
        let _parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("Output should be valid JSON");
    }
}
