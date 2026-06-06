//! Bloch sphere visualization for multi-qubit states.
//!
//! Provides single-qubit Bloch vector computation via partial trace
//! and a Plotly JSON generator for an N-sphere grid.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use serde_json::{json, Value};

use crate::error::{QuantRS2Error, QuantRS2Result};

/// Compute the Bloch vector for a single qubit by partial-tracing the full state.
///
/// Returns `(x, y, z)` where:
/// - `x = 2 · Re(ρ₀₁)`
/// - `y = 2 · Im(ρ₁₀)` (= −2 · Im(ρ₀₁) due to Hermiticity)
/// - `z = ρ₀₀ − ρ₁₁`
pub fn bloch_vector_for_qubit(
    state: &Array1<Complex64>,
    qubit_idx: usize,
    n_qubits: usize,
) -> QuantRS2Result<(f64, f64, f64)> {
    let dim = 1usize << n_qubits;
    if state.len() != dim {
        return Err(QuantRS2Error::InvalidInput(format!(
            "State length {} does not match 2^{} = {}",
            state.len(),
            n_qubits,
            dim
        )));
    }
    if qubit_idx >= n_qubits {
        return Err(QuantRS2Error::InvalidInput(format!(
            "qubit_idx {} out of range for {} qubits",
            qubit_idx, n_qubits
        )));
    }

    // Build density matrix ρ = |ψ⟩⟨ψ|
    let mut rho = Array2::zeros((dim, dim));
    for i in 0..dim {
        for j in 0..dim {
            rho[[i, j]] = state[i] * state[j].conj();
        }
    }

    // Partial trace: keep only qubit_idx
    let reduced = crate::matrix_ops::partial_trace(&rho, &[qubit_idx], n_qubits)?;

    // Bloch vector components from the 2×2 reduced density matrix
    let x = 2.0 * reduced[[0, 1]].re;
    let y = 2.0 * reduced[[1, 0]].im; // Im(ρ₁₀) = −Im(ρ₀₁)
    let z = reduced[[0, 0]].re - reduced[[1, 1]].re;

    Ok((x, y, z))
}

/// Build a unit-sphere surface mesh as a Plotly `surface` trace.
///
/// Returns a `serde_json::Value` ready to be embedded in a `data` array.
fn sphere_surface_trace(x_axis: &str) -> Value {
    // Parametric sphere: 20×20 mesh is sufficient for background rendering
    let n = 20usize;
    let mut x_vals: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    let mut y_vals: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    let mut z_vals: Vec<Vec<f64>> = Vec::with_capacity(n + 1);

    for i in 0..=n {
        let theta = std::f64::consts::PI * (i as f64) / (n as f64);
        let mut row_x = Vec::with_capacity(n + 1);
        let mut row_y = Vec::with_capacity(n + 1);
        let mut row_z = Vec::with_capacity(n + 1);
        for j in 0..=n {
            let phi = 2.0 * std::f64::consts::PI * (j as f64) / (n as f64);
            row_x.push(theta.sin() * phi.cos());
            row_y.push(theta.sin() * phi.sin());
            row_z.push(theta.cos());
        }
        x_vals.push(row_x);
        y_vals.push(row_y);
        z_vals.push(row_z);
    }

    json!({
        "type": "surface",
        "x": x_vals,
        "y": y_vals,
        "z": z_vals,
        "opacity": 0.25,
        "colorscale": [[0, "lightblue"], [1, "lightblue"]],
        "showscale": false,
        "scene": format!("scene{}", x_axis.trim_start_matches("xaxis")),
        "hoverinfo": "none"
    })
}

/// Build a cone trace representing the Bloch vector.
fn bloch_vector_trace(bx: f64, by: f64, bz: f64, scene: &str) -> Value {
    json!({
        "type": "cone",
        "x": [0.0],
        "y": [0.0],
        "z": [0.0],
        "u": [bx],
        "v": [by],
        "w": [bz],
        "colorscale": [[0, "red"], [1, "darkred"]],
        "showscale": false,
        "sizemode": "absolute",
        "sizeref": 0.5,
        "anchor": "tail",
        "scene": scene,
        "hoverinfo": "text",
        "text": [format!("({:.3}, {:.3}, {:.3})", bx, by, bz)]
    })
}

/// Returns a Plotly-JSON string for a grid of Bloch spheres, one per qubit.
///
/// The grid is arranged in a `ceil(sqrt(N)) × ceil(sqrt(N))` layout.
/// Each subplot shows the Bloch sphere surface and the qubit's Bloch vector.
pub fn bloch_array_plotly_json(
    state: &Array1<Complex64>,
    n_qubits: usize,
) -> QuantRS2Result<String> {
    if n_qubits == 0 {
        return Err(QuantRS2Error::InvalidInput(
            "n_qubits must be > 0".to_string(),
        ));
    }

    // Compute Bloch vectors for all qubits
    let mut vectors: Vec<(f64, f64, f64)> = Vec::with_capacity(n_qubits);
    for i in 0..n_qubits {
        vectors.push(bloch_vector_for_qubit(state, i, n_qubits)?);
    }

    let cols = (n_qubits as f64).sqrt().ceil() as usize;
    let cols = cols.max(1);
    let rows = (n_qubits + cols - 1) / cols;

    let mut data: Vec<Value> = Vec::new();
    let mut layout = json!({});

    // Build scene layout and traces for each qubit
    for (idx, &(bx, by, bz)) in vectors.iter().enumerate() {
        let scene_name = if idx == 0 {
            "scene".to_string()
        } else {
            format!("scene{}", idx + 1)
        };

        let row = idx / cols;
        let col = idx % cols;

        // Each subplot occupies a fraction of the total plotting area
        let w = 1.0 / (cols as f64);
        let h = 1.0 / (rows as f64);
        let x_start = (col as f64) * w;
        let y_start = 1.0 - ((row + 1) as f64) * h;

        // Sphere surface trace
        let mut sphere = sphere_surface_trace("x");
        // Overwrite scene key to correct scene
        if let Value::Object(ref mut map) = sphere {
            map.insert("scene".to_string(), json!(scene_name));
        }
        data.push(sphere);

        // Bloch vector cone trace
        let mut cone = bloch_vector_trace(bx, by, bz, &scene_name);
        // scene is already set in the function above; override if needed
        if let Value::Object(ref mut map) = cone {
            map.insert("scene".to_string(), json!(scene_name));
        }
        data.push(cone);

        // Scene layout entry
        let scene_def = json!({
            "xaxis": {"title": "x", "range": [-1.2, 1.2]},
            "yaxis": {"title": "y", "range": [-1.2, 1.2]},
            "zaxis": {"title": "z", "range": [-1.2, 1.2]},
            "aspectmode": "cube",
            "annotations": [{
                "text": format!("Qubit {}", idx),
                "x": 0.5,
                "y": 1.05,
                "z": 1.0,
                "showarrow": false,
                "font": {"size": 12}
            }],
            "domain": {
                "x": [x_start, x_start + w],
                "y": [y_start, y_start + h]
            }
        });

        if let Value::Object(ref mut layout_map) = layout {
            layout_map.insert(scene_name, scene_def);
        }
    }

    if let Value::Object(ref mut layout_map) = layout {
        layout_map.insert("title".to_string(), json!("Bloch Sphere Array"));
        layout_map.insert("showlegend".to_string(), json!(false));
        layout_map.insert("height".to_string(), json!(400 * rows));
    }

    let figure = json!({
        "data": data,
        "layout": layout
    });

    serde_json::to_string(&figure).map_err(QuantRS2Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::Complex64;

    fn state_zero() -> Array1<Complex64> {
        Array1::from(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)])
    }

    fn state_one() -> Array1<Complex64> {
        Array1::from(vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)])
    }

    fn state_plus() -> Array1<Complex64> {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        Array1::from(vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ])
    }

    fn state_plus_i() -> Array1<Complex64> {
        // |+i⟩ = (|0⟩ + i|1⟩)/√2
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        Array1::from(vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(0.0, inv_sqrt2),
        ])
    }

    fn state_bell() -> Array1<Complex64> {
        // |Φ+⟩ = (|00⟩ + |11⟩)/√2
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        Array1::from(vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ])
    }

    #[test]
    fn test_bloch_zero_state() {
        let v = bloch_vector_for_qubit(&state_zero(), 0, 1).expect("Bloch vector failed");
        assert!((v.0).abs() < 1e-10, "x should be 0, got {}", v.0);
        assert!((v.1).abs() < 1e-10, "y should be 0, got {}", v.1);
        assert!((v.2 - 1.0).abs() < 1e-10, "z should be 1, got {}", v.2);
    }

    #[test]
    fn test_bloch_one_state() {
        let v = bloch_vector_for_qubit(&state_one(), 0, 1).expect("Bloch vector failed");
        assert!((v.0).abs() < 1e-10, "x should be 0, got {}", v.0);
        assert!((v.1).abs() < 1e-10, "y should be 0, got {}", v.1);
        assert!((v.2 + 1.0).abs() < 1e-10, "z should be -1, got {}", v.2);
    }

    #[test]
    fn test_bloch_plus_state() {
        let v = bloch_vector_for_qubit(&state_plus(), 0, 1).expect("Bloch vector failed");
        assert!((v.0 - 1.0).abs() < 1e-10, "x should be 1, got {}", v.0);
        assert!((v.1).abs() < 1e-10, "y should be 0, got {}", v.1);
        assert!((v.2).abs() < 1e-10, "z should be 0, got {}", v.2);
    }

    #[test]
    fn test_bloch_plus_i_state() {
        // |+i⟩ = (|0⟩+i|1⟩)/√2 → Bloch vector = (0, 1, 0)
        let v = bloch_vector_for_qubit(&state_plus_i(), 0, 1).expect("Bloch vector failed");
        assert!((v.0).abs() < 1e-10, "x should be 0, got {}", v.0);
        assert!((v.1 - 1.0).abs() < 1e-10, "y should be 1, got {}", v.1);
        assert!((v.2).abs() < 1e-10, "z should be 0, got {}", v.2);
    }

    #[test]
    fn test_bloch_bell_state() {
        // Both qubits of |Φ+⟩ should be maximally mixed → (0, 0, 0)
        let bell = state_bell();
        for q in 0..2 {
            let v = bloch_vector_for_qubit(&bell, q, 2).expect("Bloch vector failed");
            assert!(
                v.0.abs() < 1e-10 && v.1.abs() < 1e-10 && v.2.abs() < 1e-10,
                "Bell qubit {} Bloch vector should be (0,0,0), got {:?}",
                q,
                v
            );
        }
    }

    #[test]
    fn test_bloch_json_valid() {
        let state = state_bell();
        let json_str = bloch_array_plotly_json(&state, 2).expect("JSON generation failed");
        let _parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("Output should be valid JSON");
    }
}
