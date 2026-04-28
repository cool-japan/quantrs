//! Density matrix 3D bar-plot visualization.
//!
//! Generates two side-by-side 3D bar charts showing Re(ρ) and Im(ρ)
//! for a quantum state's density matrix ρ = |ψ⟩⟨ψ|.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use serde_json::{json, Value};

use crate::error::{QuantRS2Error, QuantRS2Result};

/// Build ρ = |ψ⟩⟨ψ| from a state vector.
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

/// Generate a basis label string for a given index with `n_qubits`.
///
/// For example, index 3 with n_qubits=2 → "|11⟩".
fn basis_label(idx: usize, n_qubits: usize) -> String {
    let bits: String = (0..n_qubits)
        .rev()
        .map(|b| if (idx >> b) & 1 == 1 { '1' } else { '0' })
        .collect();
    format!("|{}⟩", bits)
}

/// Map a float value to an RGB colour string.
///
/// Positive values → red end; negative values → blue end.
fn value_to_color(v: f64, vmax: f64) -> String {
    let t = if vmax < 1e-12 {
        0.5
    } else {
        // Map [-vmax, +vmax] → [0, 1]
        0.5 + 0.5 * (v / vmax).clamp(-1.0, 1.0)
    };
    // Interpolate: blue (0,0,255) → white (255,255,255) → red (255,0,0)
    let r = (t * 255.0) as u8;
    let b = ((1.0 - t) * 255.0) as u8;
    let g = ((1.0 - (2.0 * t - 1.0).abs()) * 180.0) as u8;
    format!("rgb({},{},{})", r, g, b)
}

/// Build Plotly mesh3d vertices for a single rectangular bar.
///
/// The bar runs from (x0, y0, 0) to (x0+w, y0+h, z_height).
/// Returns (vertices x, y, z, face indices i, j, k).
fn bar_mesh(
    xi: f64,
    yi: f64,
    width: f64,
    depth: f64,
    height: f64,
) -> (
    [f64; 8],
    [f64; 8],
    [f64; 8],
    [usize; 12],
    [usize; 12],
    [usize; 12],
) {
    let x = [
        xi,
        xi + width,
        xi + width,
        xi,
        xi,
        xi + width,
        xi + width,
        xi,
    ];
    let y = [
        yi,
        yi,
        yi + depth,
        yi + depth,
        yi,
        yi,
        yi + depth,
        yi + depth,
    ];
    // Bottom face z=0, top face z=height; ensure non-degenerate for tiny heights
    let z_top = if height.abs() < 1e-15 { 1e-15 } else { height };
    let z = [0.0, 0.0, 0.0, 0.0, z_top, z_top, z_top, z_top];

    // 12 triangles (6 faces × 2 triangles each)
    let i = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6];
    let j = [1, 3, 2, 5, 3, 6, 0, 7, 5, 7, 6, 7];
    let k = [2, 7, 6, 4, 7, 5, 4, 4, 6, 6, 7, 5];
    (x, y, z, i, j, k)
}

/// Build all mesh3d data for a d×d matrix as accumulated bar mesh.
///
/// Returns a Plotly `mesh3d` trace value.
fn matrix_to_mesh3d(values: &Array2<f64>, scene: &str, title: &str) -> Value {
    let d = values.nrows();
    let bar_size = 0.7f64; // width/depth of each bar (leaves gap between bars)
    let gap = 1.0f64; // spacing between bar centres

    // Find max absolute value for colour scaling
    let vmax = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    // Accumulate all bars into a single mesh3d trace
    let mut all_x: Vec<f64> = Vec::new();
    let mut all_y: Vec<f64> = Vec::new();
    let mut all_z: Vec<f64> = Vec::new();
    let mut all_i: Vec<usize> = Vec::new();
    let mut all_j: Vec<usize> = Vec::new();
    let mut all_k: Vec<usize> = Vec::new();
    let mut all_colors: Vec<String> = Vec::new();
    let mut offset = 0usize;

    for i in 0..d {
        for j in 0..d {
            let v = values[[i, j]];
            let xi = (j as f64) * gap;
            let yi = (i as f64) * gap;

            let (bx, by, bz, bi, bj, bk) = bar_mesh(xi, yi, bar_size, bar_size, v);

            all_x.extend_from_slice(&bx);
            all_y.extend_from_slice(&by);
            all_z.extend_from_slice(&bz);

            for &fi in &bi {
                all_i.push(fi + offset);
            }
            for &fi in &bj {
                all_j.push(fi + offset);
            }
            for &fi in &bk {
                all_k.push(fi + offset);
            }

            let color = value_to_color(v, vmax);
            for _ in 0..8 {
                all_colors.push(color.clone());
            }
            offset += 8;
        }
    }

    json!({
        "type": "mesh3d",
        "x": all_x,
        "y": all_y,
        "z": all_z,
        "i": all_i,
        "j": all_j,
        "k": all_k,
        "vertexcolor": all_colors,
        "opacity": 0.9,
        "scene": scene,
        "name": title,
        "hoverinfo": "none"
    })
}

/// Two side-by-side 3D bar plots of Re(ρ) and Im(ρ).
///
/// Builds ρ = |ψ⟩⟨ψ| as a d×d matrix (d = 2^n_qubits) and
/// generates a Plotly figure with two 3D subplots.
pub fn density_matrix_bars_plotly_json(
    state: &Array1<Complex64>,
    n_qubits: usize,
) -> QuantRS2Result<String> {
    let dim = 1usize << n_qubits;
    if state.len() != dim {
        return Err(QuantRS2Error::InvalidInput(format!(
            "State length {} does not match 2^{} = {}",
            state.len(),
            n_qubits,
            dim
        )));
    }
    if n_qubits == 0 {
        return Err(QuantRS2Error::InvalidInput(
            "n_qubits must be > 0".to_string(),
        ));
    }

    let rho = density_matrix(state);
    let labels: Vec<String> = (0..dim).map(|i| basis_label(i, n_qubits)).collect();

    // Extract real and imaginary parts
    let re_matrix: Array2<f64> = rho.mapv(|c| c.re);
    let im_matrix: Array2<f64> = rho.mapv(|c| c.im);

    let re_trace = matrix_to_mesh3d(&re_matrix, "scene", "Re(ρ)");
    let im_trace = matrix_to_mesh3d(&im_matrix, "scene2", "Im(ρ)");

    // Tick configuration for axes
    let tick_vals: Vec<f64> = (0..dim).map(|k| (k as f64) * 1.0 + 0.35).collect();
    let tick_text: Vec<String> = labels.clone();

    let axis_def = json!({
        "tickvals": tick_vals,
        "ticktext": tick_text
    });

    let layout = json!({
        "title": "Density Matrix 3D Bar Plot",
        "scene": {
            "xaxis": axis_def,
            "yaxis": axis_def,
            "zaxis": {"title": "Re(ρ)"},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
            "domain": {"x": [0.0, 0.48], "y": [0.0, 1.0]},
            "annotations": [{
                "text": "Re(ρ)",
                "x": 0.5, "y": 1.0, "z": 0.0,
                "showarrow": false,
                "font": {"size": 14}
            }]
        },
        "scene2": {
            "xaxis": axis_def,
            "yaxis": axis_def,
            "zaxis": {"title": "Im(ρ)"},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
            "domain": {"x": [0.52, 1.0], "y": [0.0, 1.0]},
            "annotations": [{
                "text": "Im(ρ)",
                "x": 0.5, "y": 1.0, "z": 0.0,
                "showarrow": false,
                "font": {"size": 14}
            }]
        },
        "height": 600,
        "showlegend": false
    });

    let figure = json!({
        "data": [re_trace, im_trace],
        "layout": layout
    });

    serde_json::to_string(&figure).map_err(QuantRS2Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::Complex64;

    /// Build a maximally mixed 2-qubit state (sqrt of I/4).
    /// As a pure state |ψ⟩, the density matrix of |00⟩ is not I/4;
    /// but we can check diagonal of a computational basis state.
    fn state_zero_2q() -> Array1<Complex64> {
        // |00⟩: ρ has Re diagonal = (1,0,0,0), Im = 0
        let mut s = Array1::zeros(4);
        s[0] = Complex64::new(1.0, 0.0);
        s
    }

    fn state_plus_plus() -> Array1<Complex64> {
        // |++⟩ = (|00⟩+|01⟩+|10⟩+|11⟩)/2
        let half = 0.5;
        Array1::from(vec![
            Complex64::new(half, 0.0),
            Complex64::new(half, 0.0),
            Complex64::new(half, 0.0),
            Complex64::new(half, 0.0),
        ])
    }

    #[test]
    fn test_density_bars_identity_2qubit() {
        // For |++⟩ = (|00⟩+|01⟩+|10⟩+|11⟩)/2,
        // ρ = |++⟩⟨++| = outer product → all entries = 1/4.
        // Re(ρ) diagonal = 0.25, off-diagonal Re = 0.25, Im = 0.
        let state = state_plus_plus();
        let rho = density_matrix(&state);

        for i in 0..4 {
            for j in 0..4 {
                let re = rho[[i, j]].re;
                let im = rho[[i, j]].im;
                assert!(
                    (re - 0.25).abs() < 1e-10,
                    "Re(ρ[{},{}]) should be 0.25, got {}",
                    i,
                    j,
                    re
                );
                assert!(
                    im.abs() < 1e-10,
                    "Im(ρ[{},{}]) should be 0, got {}",
                    i,
                    j,
                    im
                );
            }
        }
    }

    #[test]
    fn test_density_bars_zero_state() {
        // |00⟩: ρ[0,0] = 1, all others = 0
        let state = state_zero_2q();
        let rho = density_matrix(&state);
        assert!((rho[[0, 0]].re - 1.0).abs() < 1e-10);
        for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 0)] {
            assert!(rho[[i, j]].norm_sqr() < 1e-20);
        }
    }

    #[test]
    fn test_density_bars_json_valid() {
        let state = state_zero_2q();
        let json_str =
            density_matrix_bars_plotly_json(&state, 2).expect("Density bars JSON failed");
        let _parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("Output should be valid JSON");
    }
}
