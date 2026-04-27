//! Husimi Q-distribution visualization for quantum states.
//!
//! Evaluates Q(θ, φ) = |⟨α(θ,φ)|^⊗n |ψ⟩|² on a 64×64 grid over
//! the sphere (θ ∈ [0,π], φ ∈ [0,2π]), where |α(θ,φ)⟩ is the
//! single-qubit SU(2) coherent state and the n-qubit coherent state
//! is the tensor product |α⟩^⊗n.

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use serde_json::{json, Value};

use crate::error::{QuantRS2Error, QuantRS2Result};

/// Resolution of the Husimi grid (64 points per axis).
const GRID_SIZE: usize = 64;

/// Compute the n-qubit coherent state |α(θ,φ)⟩^⊗n.
///
/// The single-qubit coherent state is:
///   |α⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
///
/// For n qubits the coherent state is the tensor product |α⟩⊗n.
fn coherent_state_tensor(theta: f64, phi: f64, n_qubits: usize) -> Array1<Complex64> {
    let dim = 1usize << n_qubits;
    // Single-qubit coherent-state amplitudes
    let c0 = Complex64::new((theta / 2.0).cos(), 0.0);
    let c1 =
        Complex64::from_polar((theta / 2.0).sin(), phi);

    let mut state = Array1::zeros(dim);
    // Each basis index encodes n bits; the k-th qubit contributes c0 or c1
    // according to whether bit k of the index is 0 or 1.
    // Qubit 0 is the most significant bit (MSB convention).
    for idx in 0..dim {
        let mut amp = Complex64::new(1.0, 0.0);
        for q in 0..n_qubits {
            let bit = (idx >> (n_qubits - 1 - q)) & 1;
            amp *= if bit == 0 { c0 } else { c1 };
        }
        state[idx] = amp;
    }
    state
}

/// Compute Q(θ, φ) = |⟨α(θ,φ)|ψ⟩|²  (no extra 1/π factor needed here
/// since we only use Q for visualisation, not for phase-space integrals).
fn husimi_q(
    state: &Array1<Complex64>,
    theta: f64,
    phi: f64,
    n_qubits: usize,
) -> f64 {
    let coherent = coherent_state_tensor(theta, phi, n_qubits);
    let inner: Complex64 = coherent
        .iter()
        .zip(state.iter())
        .map(|(c, s)| c.conj() * s)
        .sum();
    inner.norm_sqr()
}

/// Returns a Plotly-JSON string for a Husimi Q-distribution surface plot.
///
/// The Q-function is evaluated on a 64×64 grid of (θ, φ) angles
/// and rendered as a 3D surface.
pub fn husimi_plotly_json(
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

    let pi = std::f64::consts::PI;
    let n = GRID_SIZE;

    let theta_vals: Vec<f64> = (0..n)
        .map(|i| pi * (i as f64) / ((n - 1) as f64))
        .collect();
    let phi_vals: Vec<f64> = (0..n)
        .map(|j| 2.0 * pi * (j as f64) / ((n - 1) as f64))
        .collect();

    // Compute Q on the grid; store as (n_theta × n_phi) outer → inner
    let mut z_matrix: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut x_sph: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y_sph: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut z_sph: Vec<Vec<f64>> = Vec::with_capacity(n);

    for &theta in &theta_vals {
        let mut row_z = Vec::with_capacity(n);
        let mut row_x = Vec::with_capacity(n);
        let mut row_y = Vec::with_capacity(n);
        let mut row_zs = Vec::with_capacity(n);
        for &phi in &phi_vals {
            let q_val = husimi_q(state, theta, phi, n_qubits);
            row_z.push(q_val);
            row_x.push(theta.sin() * phi.cos());
            row_y.push(theta.sin() * phi.sin());
            row_zs.push(theta.cos());
        }
        z_matrix.push(row_z);
        x_sph.push(row_x);
        y_sph.push(row_y);
        z_sph.push(row_zs);
    }

    // Sparse labels for heatmap axes (every 8th gridpoint)
    let phi_axis_labels: Vec<String> = phi_vals
        .iter()
        .step_by(8)
        .map(|&p| format!("{:.2}π", p / std::f64::consts::PI))
        .collect();
    let phi_axis_vals: Vec<usize> = (0..n).step_by(8).collect();
    let theta_axis_labels: Vec<String> = theta_vals
        .iter()
        .step_by(8)
        .map(|&t| format!("{:.2}π", t / std::f64::consts::PI))
        .collect();
    let theta_axis_vals: Vec<usize> = (0..n).step_by(8).collect();

    // Two traces: heatmap (θ,φ) and 3D surface on sphere
    let heatmap = json!({
        "type": "heatmap",
        "z": z_matrix,
        "colorscale": "Viridis",
        "colorbar": {"title": "Q(θ,φ)"},
        "xaxis": "x",
        "yaxis": "y"
    });

    let surface3d = json!({
        "type": "surface",
        "x": x_sph,
        "y": y_sph,
        "z": z_sph,
        "surfacecolor": z_matrix,
        "colorscale": "Viridis",
        "showscale": true,
        "colorbar": {"title": "Q(θ,φ)"},
        "scene": "scene"
    });

    let figure = json!({
        "data": [surface3d, heatmap],
        "layout": {
            "title": "Husimi Q-Distribution",
            "scene": {
                "xaxis": {"title": "x"},
                "yaxis": {"title": "y"},
                "zaxis": {"title": "z"},
                "aspectmode": "cube",
                "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
                "domain": {"x": [0.0, 0.6], "y": [0.0, 1.0]}
            },
            "xaxis": {
                "title": "φ (radians/π)",
                "domain": [0.65, 1.0],
                "tickvals": phi_axis_vals,
                "ticktext": phi_axis_labels
            },
            "yaxis": {
                "title": "θ (radians/π)",
                "tickvals": theta_axis_vals,
                "ticktext": theta_axis_labels
            },
            "height": 600,
            "showlegend": false
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

    fn state_plus_1q() -> Array1<Complex64> {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        Array1::from(vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ])
    }

    #[test]
    fn test_husimi_zero_state_peak() {
        let state = state_zero_1q();
        let pi = std::f64::consts::PI;
        let n = GRID_SIZE;

        // For |0⟩, Q(θ,φ) = cos²(θ/2), which is maximum at θ=0
        let q_at_north = husimi_q(&state, 0.0, 0.0, 1);
        let q_at_equator = husimi_q(&state, pi / 2.0, 0.0, 1);
        let q_at_south = husimi_q(&state, pi, 0.0, 1);

        assert!(
            q_at_north > q_at_equator,
            "Q peak at θ=0 should be > equator: {} vs {}",
            q_at_north,
            q_at_equator
        );
        assert!(
            q_at_north > q_at_south,
            "Q peak at θ=0 should be > south: {} vs {}",
            q_at_north,
            q_at_south
        );
        assert!(
            (q_at_north - 1.0).abs() < 1e-10,
            "Q(0,0) for |0⟩ should be 1.0, got {}",
            q_at_north
        );

        // Check that the grid maximum is at the first theta point (closest to 0)
        let theta_step = pi / ((n - 1) as f64);
        let q_first_row_max = (0..GRID_SIZE)
            .map(|j| {
                let phi = 2.0 * pi * (j as f64) / ((n - 1) as f64);
                husimi_q(&state, theta_step * 0.0, phi, 1)
            })
            .fold(f64::NEG_INFINITY, f64::max);
        let q_last_row_max = (0..GRID_SIZE)
            .map(|j| {
                let phi = 2.0 * pi * (j as f64) / ((n - 1) as f64);
                husimi_q(&state, pi, phi, 1)
            })
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            q_first_row_max > q_last_row_max,
            "Q peak should be at north (θ≈0)"
        );
    }

    #[test]
    fn test_husimi_plus_state_peak() {
        // |+⟩ = (|0⟩+|1⟩)/√2 → coherent state at θ=π/2, φ=0
        // Q(π/2, 0) should be 1.0 for |+⟩
        let state = state_plus_1q();
        let pi = std::f64::consts::PI;

        let q_at_equator_0 = husimi_q(&state, pi / 2.0, 0.0, 1);
        let q_at_equator_pi = husimi_q(&state, pi / 2.0, pi, 1);
        let q_at_north = husimi_q(&state, 0.0, 0.0, 1);

        assert!(
            q_at_equator_0 > q_at_north,
            "Q(π/2, 0) should exceed Q(0, 0) for |+⟩: {} vs {}",
            q_at_equator_0,
            q_at_north
        );
        assert!(
            q_at_equator_0 > q_at_equator_pi,
            "Q(π/2, 0) should exceed Q(π/2, π) for |+⟩: {} vs {}",
            q_at_equator_0,
            q_at_equator_pi
        );
        // Q(π/2, 0) = |cos(π/4)·1 + e^{0}·sin(π/4)·1/√2|² ... let's compute:
        // |α(π/2, 0)⟩ = (|0⟩+|1⟩)/√2 = |+⟩, so Q(π/2, 0) = |⟨+|+⟩|² = 1
        assert!(
            (q_at_equator_0 - 1.0).abs() < 1e-10,
            "Q(π/2, 0) for |+⟩ should be 1.0, got {}",
            q_at_equator_0
        );
    }

    #[test]
    fn test_husimi_json_valid() {
        let state = state_zero_1q();
        let json_str = husimi_plotly_json(&state, 1).expect("Husimi JSON failed");
        let _parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("Output should be valid JSON");
    }
}
