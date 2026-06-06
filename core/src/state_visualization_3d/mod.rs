//! Advanced 3D quantum state visualization.
//!
//! Provides five visualization types for quantum state vectors:
//!
//! - **Bloch sphere array**: per-qubit Bloch vectors on a sphere grid
//! - **Q-sphere**: Qiskit-style global phase-weighted amplitude map
//! - **Discrete Wigner**: Wootters phase-space function (n=1, 2 only)
//! - **Husimi Q**: SU(2) coherent-state projection on the sphere
//! - **Density matrix bars**: 3D bar plots of Re(ρ) and Im(ρ)
//!
//! All renderers return Plotly-JSON strings that can be wrapped in
//! a self-contained HTML page or consumed by the Plotly.js library.

#![allow(clippy::missing_const_for_fn)]

pub mod bloch;
pub mod density_bars;
pub mod husimi;
pub mod qsphere;
pub mod wigner;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use scirs2_core::ndarray::Array1;
#[cfg(feature = "python")]
use scirs2_core::Complex64;

/// Wraps a Plotly-JSON object in a self-contained HTML page.
///
/// The resulting HTML pulls Plotly.js from CDN and renders the
/// figure in a full-page `<div>`.
pub fn make_plotly_html(plotly_json: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QuantRS2 Quantum State Visualization</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ margin: 0; padding: 10px; font-family: Arial, sans-serif; }}
  #plot {{ width: 100%; height: calc(100vh - 20px); }}
</style>
</head>
<body>
<div id="plot"></div>
<script>
var figure = {json};
Plotly.newPlot('plot', figure.data, figure.layout, {{responsive: true}});
</script>
</body>
</html>"#,
        json = plotly_json
    )
}

/// Python class for 3D quantum state visualization.
///
/// Accepts state amplitudes as a list of `(re, im)` tuples and
/// exposes methods for each visualization type returning HTML strings.
#[cfg(feature = "python")]
#[pyclass(name = "QuantumState3DVisualizer")]
pub struct PyQuantumState3DVisualizer {
    state: Array1<Complex64>,
    n_qubits: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQuantumState3DVisualizer {
    /// Create a new visualizer from state amplitudes.
    ///
    /// Parameters
    /// ----------
    /// state_amplitudes : list[tuple[float, float]]
    ///     List of `(re, im)` pairs for each computational basis state.
    ///     Length must equal `2**n_qubits`.
    /// n_qubits : int
    ///     Number of qubits.
    #[new]
    pub fn new(state_amplitudes: Vec<(f64, f64)>, n_qubits: usize) -> PyResult<Self> {
        let expected = 1usize << n_qubits;
        if state_amplitudes.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "state_amplitudes length {} does not match 2^{} = {}",
                state_amplitudes.len(),
                n_qubits,
                expected
            )));
        }
        let state: Array1<Complex64> = state_amplitudes
            .into_iter()
            .map(|(re, im)| Complex64::new(re, im))
            .collect();
        Ok(Self { state, n_qubits })
    }

    /// Generate a multi-qubit Bloch sphere array visualization.
    ///
    /// Returns an HTML string with Plotly 3D scenes, one per qubit.
    pub fn bloch_array_html(&self) -> PyResult<String> {
        let json = bloch::bloch_array_plotly_json(&self.state, self.n_qubits)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(make_plotly_html(&json))
    }

    /// Generate a Q-sphere visualization.
    ///
    /// Returns an HTML string with a Plotly 3D scatter on a sphere,
    /// each marker encoding amplitude magnitude and phase.
    pub fn qsphere_html(&self) -> PyResult<String> {
        let json = qsphere::qsphere_plotly_json(&self.state, self.n_qubits)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(make_plotly_html(&json))
    }

    /// Generate a discrete Wigner function visualization.
    ///
    /// Returns an HTML string with a Plotly heatmap.
    /// Only n=1 and n=2 are supported; raises ValueError for n ≥ 3.
    pub fn wigner_html(&self) -> PyResult<String> {
        let json = wigner::wigner_plotly_json(&self.state, self.n_qubits)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(make_plotly_html(&json))
    }

    /// Generate a Husimi Q-distribution visualization.
    ///
    /// Returns an HTML string with a Plotly 3D surface over the sphere.
    pub fn husimi_html(&self) -> PyResult<String> {
        let json = husimi::husimi_plotly_json(&self.state, self.n_qubits)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(make_plotly_html(&json))
    }

    /// Generate density matrix 3D bar plots.
    ///
    /// Returns an HTML string with two side-by-side 3D bar charts
    /// showing Re(ρ) and Im(ρ).
    pub fn density_bars_html(&self) -> PyResult<String> {
        let json = density_bars::density_matrix_bars_plotly_json(&self.state, self.n_qubits)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(make_plotly_html(&json))
    }

    /// String representation showing qubit count and state dimension.
    pub fn __repr__(&self) -> String {
        format!(
            "QuantumState3DVisualizer(n_qubits={}, dim={})",
            self.n_qubits,
            self.state.len()
        )
    }
}
