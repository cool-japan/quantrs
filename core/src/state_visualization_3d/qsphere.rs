//! Q-sphere visualization (Qiskit-style) for quantum states.
//!
//! Maps each computational basis state to a point on a unit sphere
//! with latitude proportional to Hamming weight and longitude
//! determined by rank within the Hamming shell.

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use serde_json::{json, Value};

use crate::error::{QuantRS2Error, QuantRS2Result};

/// Count set bits (Hamming weight / popcount).
fn popcount(mut x: usize) -> usize {
    let mut count = 0usize;
    while x != 0 {
        count += x & 1;
        x >>= 1;
    }
    count
}

/// Map a phase angle in [0, 2π) to an RGB hex string using a cyclic colormap.
///
/// Uses a simple HSV→RGB conversion with hue = phase / (2π).
fn phase_to_color(phase: f64) -> String {
    let two_pi = 2.0 * std::f64::consts::PI;
    // Normalize phase to [0, 1)
    let mut h = phase / two_pi;
    h -= h.floor();

    // HSV with S=1, V=1
    let hi = (h * 6.0).floor() as u32 % 6;
    let f = h * 6.0 - h.floor() * 6.0;
    // Note: this re-computes h floor so let's compute f properly
    let f = h * 6.0 - (h * 6.0).floor();
    let (r, g, b): (f64, f64, f64) = match hi {
        0 => (1.0, f, 0.0),
        1 => (1.0 - f, 1.0, 0.0),
        2 => (0.0, 1.0, f),
        3 => (0.0, 1.0 - f, 1.0),
        4 => (f, 0.0, 1.0),
        _ => (1.0, 0.0, 1.0 - f),
    };
    format!(
        "#{:02x}{:02x}{:02x}",
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8
    )
}

/// Returns a Plotly-JSON string for a Q-sphere visualization of the state.
///
/// The sphere surface is rendered as a background mesh; each non-zero
/// amplitude is plotted as a 3D scatter marker whose size encodes
/// `|a|²` and colour encodes `arg(a)`.
pub fn qsphere_plotly_json(
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

    // Precompute Hamming shells: for each Hamming weight w, list indices with that weight
    let max_w = n_qubits;
    let mut shells: Vec<Vec<usize>> = vec![Vec::new(); max_w + 1];
    for x in 0..dim {
        shells[popcount(x)].push(x);
    }

    // Compute spherical coordinates for every non-negligible basis state
    let pi = std::f64::consts::PI;

    let mut scatter_x: Vec<f64> = Vec::new();
    let mut scatter_y: Vec<f64> = Vec::new();
    let mut scatter_z: Vec<f64> = Vec::new();
    let mut marker_sizes: Vec<f64> = Vec::new();
    let mut marker_colors: Vec<String> = Vec::new();
    let mut hover_texts: Vec<String> = Vec::new();

    for x in 0..dim {
        let amp = state[x];
        let prob = amp.norm_sqr();
        if prob < 1e-12 {
            continue;
        }

        let w = popcount(x);
        let theta = if n_qubits == 1 {
            // n=1: |0⟩ at north (0), |1⟩ at south (π)
            pi * (w as f64)
        } else {
            pi * (w as f64) / (n_qubits as f64)
        };

        // Rank of x within its Hamming shell
        let shell = &shells[w];
        let rank = shell
            .iter()
            .position(|&v| v == x)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Shell rank not found".to_string()))?;

        let phi = if shell.len() == 1 {
            0.0
        } else {
            2.0 * pi * (rank as f64) / (shell.len() as f64)
        };

        let sx = theta.sin() * phi.cos();
        let sy = theta.sin() * phi.sin();
        let sz = theta.cos();

        scatter_x.push(sx);
        scatter_y.push(sy);
        scatter_z.push(sz);

        // Marker size: scale probability to 5–30 range
        let size = 5.0 + 25.0 * prob;
        marker_sizes.push(size);

        // Marker colour encodes phase
        let phase = amp.im.atan2(amp.re);
        let phase = if phase < 0.0 {
            phase + 2.0 * pi
        } else {
            phase
        };
        marker_colors.push(phase_to_color(phase));

        // Binary string label, e.g. "|101⟩"
        let label: String = (0..n_qubits)
            .rev()
            .map(|bit| if (x >> bit) & 1 == 1 { '1' } else { '0' })
            .collect();
        hover_texts.push(format!(
            "|{}⟩  p={:.4}  arg={:.3}rad",
            label, prob, phase
        ));
    }

    // Background sphere surface
    let sphere = build_sphere_mesh3d();

    // Scatter trace for basis states
    let scatter = json!({
        "type": "scatter3d",
        "x": scatter_x,
        "y": scatter_y,
        "z": scatter_z,
        "mode": "markers",
        "marker": {
            "size": marker_sizes,
            "color": marker_colors,
            "opacity": 0.9,
            "line": {"width": 1, "color": "black"}
        },
        "text": hover_texts,
        "hoverinfo": "text",
        "name": "Basis states"
    });

    let layout = json!({
        "title": "Q-Sphere",
        "scene": {
            "xaxis": {"title": "x", "range": [-1.3, 1.3]},
            "yaxis": {"title": "y", "range": [-1.3, 1.3]},
            "zaxis": {"title": "z", "range": [-1.3, 1.3]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.4, "y": 1.4, "z": 1.0}}
        },
        "showlegend": false,
        "height": 600
    });

    let figure = json!({
        "data": [sphere, scatter],
        "layout": layout
    });

    serde_json::to_string(&figure).map_err(QuantRS2Error::from)
}

/// Build a unit-sphere as a `mesh3d` background trace.
fn build_sphere_mesh3d() -> Value {
    let n = 18usize; // 18 latitude × 18 longitude segments
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let mut zs: Vec<f64> = Vec::new();
    let mut is: Vec<usize> = Vec::new();
    let mut js: Vec<usize> = Vec::new();
    let mut ks: Vec<usize> = Vec::new();

    let pi = std::f64::consts::PI;

    // Vertices
    for i in 0..=n {
        let theta = pi * (i as f64) / (n as f64);
        for j in 0..=n {
            let phi = 2.0 * pi * (j as f64) / (n as f64);
            xs.push(theta.sin() * phi.cos());
            ys.push(theta.sin() * phi.sin());
            zs.push(theta.cos());
        }
    }

    let stride = n + 1;
    // Triangles (two triangles per quad)
    for i in 0..n {
        for j in 0..n {
            let a = i * stride + j;
            let b = i * stride + j + 1;
            let c = (i + 1) * stride + j;
            let d = (i + 1) * stride + j + 1;
            // Triangle 1: a, b, c
            is.push(a);
            js.push(b);
            ks.push(c);
            // Triangle 2: b, d, c
            is.push(b);
            js.push(d);
            ks.push(c);
        }
    }

    json!({
        "type": "mesh3d",
        "x": xs,
        "y": ys,
        "z": zs,
        "i": is,
        "j": js,
        "k": ks,
        "opacity": 0.15,
        "color": "lightblue",
        "hoverinfo": "none",
        "name": "Sphere"
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::Complex64;

    fn state_ghz() -> Array1<Complex64> {
        // (|000⟩ + |111⟩)/√2  (3-qubit GHZ)
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let mut s = Array1::zeros(8);
        s[0] = Complex64::new(inv_sqrt2, 0.0); // |000⟩
        s[7] = Complex64::new(inv_sqrt2, 0.0); // |111⟩
        s
    }

    #[test]
    fn test_qsphere_ghz() {
        // GHZ has exactly 2 non-zero amplitudes: |000⟩ (w=0) and |111⟩ (w=3)
        let state = state_ghz();
        let json_str = qsphere_plotly_json(&state, 3).expect("Q-sphere failed");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("Should be valid JSON");

        // Find the scatter trace (index 1)
        let data = parsed["data"].as_array().expect("data array missing");
        let scatter = data
            .iter()
            .find(|t| t["type"] == "scatter3d")
            .expect("No scatter3d trace found");
        let x = scatter["x"].as_array().expect("scatter x missing");
        assert_eq!(x.len(), 2, "GHZ should have exactly 2 markers");
    }

    #[test]
    fn test_qsphere_json_valid() {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let state = Array1::from(vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ]);
        let json_str = qsphere_plotly_json(&state, 2).expect("Q-sphere failed");
        let _parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("Output should be valid JSON");
    }
}
