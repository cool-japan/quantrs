//! Jupyter Notebook Visualization Tools for QuantRS2-Core
//!
//! This module provides comprehensive visualization capabilities for Jupyter notebooks,
//! enabling interactive quantum circuit visualization, quantum state plotting, and
//! real-time quantum computation monitoring within Jupyter environments.

use pyo3::prelude::*;
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use scirs2_numpy::{PyArray1, PyArrayMethods};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantum circuit visualization for Jupyter notebooks
#[pyclass(name = "QuantumCircuitVisualizer")]
pub struct PyQuantumCircuitVisualizer {
    gates: Vec<CircuitGateInfo>,
    num_qubits: usize,
    circuit_name: String,
}

/// Information about a gate for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CircuitGateInfo {
    gate_type: String,
    qubits: Vec<u32>,
    parameters: Vec<f64>,
    time_step: usize,
    fidelity: Option<f64>,
}

#[pymethods]
impl PyQuantumCircuitVisualizer {
    #[new]
    fn new(num_qubits: usize, circuit_name: Option<String>) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
            circuit_name: circuit_name.unwrap_or_else(|| "Quantum Circuit".to_string()),
        }
    }

    /// Add a gate to the circuit visualization
    fn add_gate(
        &mut self,
        gate_type: String,
        qubits: Vec<u32>,
        parameters: Option<Vec<f64>>,
        fidelity: Option<f64>,
    ) {
        let gate_info = CircuitGateInfo {
            gate_type,
            qubits,
            parameters: parameters.unwrap_or_default(),
            time_step: self.gates.len(),
            fidelity,
        };
        self.gates.push(gate_info);
    }

    /// Generate HTML visualization for Jupyter notebooks
    fn to_html(&self) -> String {
        let mut html = format!(
            r#"
<div style="font-family: Arial, sans-serif; border: 1px solid #ddd; padding: 20px; margin: 10px; border-radius: 8px;">
    <h3 style="color: #2E86C1; margin-top: 0;">{}</h3>
    <div style="background: #f8f9fa; padding: 15px; border-radius: 4px;">
        <div style="display: flex; flex-direction: column; gap: 8px;">
"#,
            self.circuit_name
        );

        // Draw qubit lines
        for qubit in 0..self.num_qubits {
            html.push_str(&format!(
                r#"
            <div style="display: flex; align-items: center; height: 40px;">
                <span style="width: 60px; font-weight: bold; color: #34495e;">|q{}⟩</span>
                <div style="flex: 1; height: 2px; background: #34495e; position: relative;">
"#,
                qubit
            ));

            // Add gates on this qubit line
            for (step, gate) in self.gates.iter().enumerate() {
                if gate.qubits.contains(&(qubit as u32)) {
                    let position_percent =
                        (step as f64 / (self.gates.len().max(1) as f64 - 1.0)) * 100.0;
                    let color = match gate.gate_type.as_str() {
                        "H" => "#e74c3c",
                        "X" | "Y" | "Z" => "#3498db",
                        "RX" | "RY" | "RZ" => "#f39c12",
                        "CNOT" => "#9b59b6",
                        _ => "#95a5a6",
                    };
                    let fidelity_info = gate
                        .fidelity
                        .map(|f| format!(" (F: {:.3})", f))
                        .unwrap_or_default();
                    html.push_str(&format!(
                        r#"
                    <div style="position: absolute; left: {}%; transform: translateX(-50%);
                                width: 30px; height: 30px; background: {}; color: white;
                                border-radius: 4px; display: flex; align-items: center;
                                justify-content: center; font-size: 12px; font-weight: bold;
                                cursor: pointer; top: -14px;"
                         title="{}{} at step {}">
                        {}
                    </div>
"#,
                        position_percent,
                        color,
                        gate.gate_type,
                        fidelity_info,
                        step,
                        gate.gate_type
                    ));
                }
            }

            html.push_str("                </div>\n            </div>");
        }

        html.push_str(&format!(
            r#"
        </div>
    </div>
    <div style="margin-top: 15px; font-size: 12px; color: #7f8c8d;">
        <strong>Circuit Statistics:</strong> {} gates, {} qubits, {} time steps
    </div>
</div>
"#,
            self.gates.len(),
            self.num_qubits,
            self.gates.len()
        ));

        html
    }

    /// Generate SVG visualization for high-quality output
    fn to_svg(&self) -> String {
        let width = 800;
        let height = self.num_qubits * 80 + 100;
        let gate_spacing = if self.gates.is_empty() {
            80
        } else {
            (width - 120) / self.gates.len().max(1)
        };

        let mut svg = format!("<svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">
    <defs>
        <style>
            .qubit-line {{ stroke: #34495e; stroke-width: 2; }}
            .gate-h {{ fill: #e74c3c; }}
            .gate-pauli {{ fill: #3498db; }}
            .gate-rotation {{ fill: #f39c12; }}
            .gate-cnot {{ fill: #9b59b6; }}
            .gate-text {{ font-family: Arial, sans-serif; font-size: 14px; fill: white; text-anchor: middle; }}
            .qubit-label {{ font-family: Arial, sans-serif; font-size: 14px; fill: #34495e; }}
        </style>
    </defs>
    <rect width=\"100%\" height=\"100%\" fill=\"#f8f9fa\"/>
    <text x=\"{}\" y=\"25\" text-anchor=\"middle\" style=\"font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; fill: #2E86C1;\">{}</text>", width, height, width/2, self.circuit_name);

        // Draw qubit lines and labels
        for qubit in 0..self.num_qubits {
            let y = 60 + qubit * 80;
            svg.push_str(&format!(
                "
    <text x=\"50\" y=\"{}\" class=\"qubit-label\" dy=\"5\">|q{}⟩</text>
    <line x1=\"80\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"qubit-line\"/>",
                y,
                qubit,
                y,
                width - 40,
                y
            ));
        }

        // Draw gates
        for (step, gate) in self.gates.iter().enumerate() {
            let x = 100 + step * gate_spacing;

            for &qubit_id in &gate.qubits {
                let y = 60 + (qubit_id as usize) * 80;
                let (gate_class, _gate_color) = match gate.gate_type.as_str() {
                    "H" => ("gate-h", "#e74c3c"),
                    "X" | "Y" | "Z" => ("gate-pauli", "#3498db"),
                    "RX" | "RY" | "RZ" => ("gate-rotation", "#f39c12"),
                    "CNOT" => ("gate-cnot", "#9b59b6"),
                    _ => ("gate-pauli", "#95a5a6"),
                };

                svg.push_str(&format!("
    <rect x=\"{}\" y=\"{}\" width=\"30\" height=\"30\" rx=\"4\" class=\"{}\" stroke=\"#2c3e50\" stroke-width=\"1\"/>
    <text x=\"{}\" y=\"{}\" class=\"gate-text\">{}</text>", x - 15, y - 15, gate_class, x, y + 5, gate.gate_type));
            }

            // Draw control lines for CNOT gates
            if gate.gate_type == "CNOT" && gate.qubits.len() == 2 {
                let control_y = 60 + (gate.qubits[0] as usize) * 80;
                let target_y = 60 + (gate.qubits[1] as usize) * 80;
                svg.push_str(&format!(
                    "
    <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#9b59b6\" stroke-width=\"3\"/>
    <circle cx=\"{}\" cy=\"{}\" r=\"4\" fill=\"#9b59b6\"/>",
                    x, control_y, x, target_y, x, control_y
                ));
            }
        }

        svg.push_str("</svg>");
        svg
    }

    /// Get circuit statistics
    fn get_statistics(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total_gates".to_string(), self.gates.len());
        stats.insert("num_qubits".to_string(), self.num_qubits);

        // Count gate types
        for gate in &self.gates {
            let count = stats.entry(format!("gate_{}", gate.gate_type)).or_insert(0);
            *count += 1;
        }

        stats
    }

    /// Export circuit data as JSON for external visualization tools
    fn to_json(&self) -> String {
        serde_json::to_string_pretty(&CircuitData {
            name: self.circuit_name.clone(),
            num_qubits: self.num_qubits,
            gates: self.gates.clone(),
        })
        .unwrap_or_else(|_| "{}".to_string())
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantumCircuitVisualizer(name='{}', qubits={}, gates={})",
            self.circuit_name,
            self.num_qubits,
            self.gates.len()
        )
    }
}

#[derive(Serialize, Deserialize)]
struct CircuitData {
    name: String,
    num_qubits: usize,
    gates: Vec<CircuitGateInfo>,
}

/// Quantum state visualization for Jupyter notebooks
#[pyclass(name = "QuantumStateVisualizer")]
pub struct PyQuantumStateVisualizer {
    state_vector: Array1<Complex64>,
    num_qubits: usize,
}

#[pymethods]
impl PyQuantumStateVisualizer {
    #[new]
    fn new(state_vector: &Bound<'_, PyArray1<Complex64>>) -> PyResult<Self> {
        let state_array = state_vector.readonly().as_array().to_owned();
        let num_qubits = (state_array.len() as f64).log2() as usize;

        if 2usize.pow(num_qubits as u32) != state_array.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "State vector length must be a power of 2",
            ));
        }

        Ok(Self {
            state_vector: state_array,
            num_qubits,
        })
    }

    /// Generate amplitude bar chart visualization
    fn amplitude_plot_html(&self) -> String {
        let max_amplitude = self
            .state_vector
            .iter()
            .map(|c| c.norm())
            .fold(0.0f64, f64::max);

        let mut html = format!(
            r#"
<div style="font-family: Arial, sans-serif; border: 1px solid #ddd; padding: 20px; margin: 10px; border-radius: 8px;">
    <h3 style="color: #2E86C1; margin-top: 0;">Quantum State Amplitudes ({} qubits)</h3>
    <div style="display: flex; flex-wrap: wrap; gap: 4px; margin: 20px 0;">
"#,
            self.num_qubits
        );

        for (i, amplitude) in self.state_vector.iter().enumerate() {
            let prob = amplitude.norm_sqr();
            let height_percent = if max_amplitude > 0.0 {
                (amplitude.norm() / max_amplitude * 100.0).min(100.0)
            } else {
                0.0
            };
            let color = if prob > 0.001 { "#3498db" } else { "#ecf0f1" };

            let binary_state = format!("{:0width$b}", i, width = self.num_qubits);
            html.push_str(&format!(r#"
        <div style="display: flex; flex-direction: column; align-items: center; margin: 2px;">
            <div style="width: 20px; height: 100px; background: #f8f9fa; border: 1px solid #ddd;
                        display: flex; align-items: flex-end; position: relative;">
                <div style="width: 100%; height: {}%; background: {};
                            transition: height 0.3s ease;" title="State |{}⟩: {:.3}"></div>
            </div>
            <span style="font-size: 10px; margin-top: 4px; writing-mode: vertical-rl; text-orientation: mixed;">|{}⟩</span>
        </div>
"#, height_percent, color, binary_state, prob, binary_state));
        }

        html.push_str(r#"
    </div>
    <div style="margin-top: 15px; font-size: 12px; color: #7f8c8d;">
        <strong>Hover over bars to see probabilities.</strong> Height represents amplitude magnitude.
    </div>
</div>
"#);

        html
    }

    /// Generate Bloch sphere visualization for single qubits
    fn bloch_sphere_html(&self) -> PyResult<String> {
        if self.num_qubits != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Bloch sphere visualization only available for single qubits",
            ));
        }

        let alpha = self.state_vector[0];
        let beta = self.state_vector[1];

        // Calculate Bloch sphere coordinates
        let theta = 2.0 * (beta.norm() / alpha.norm()).atan();
        let phi = (beta / alpha).arg() - alpha.arg();

        let x = theta.sin() * phi.cos();
        let y = theta.sin() * phi.sin();
        let z = theta.cos();

        Ok(format!(
            r#"
<div style="font-family: Arial, sans-serif; border: 1px solid #ddd; padding: 20px; margin: 10px; border-radius: 8px;">
    <h3 style="color: #2E86C1; margin-top: 0;">Bloch Sphere Representation</h3>
    <div style="display: flex; align-items: center; gap: 20px;">
        <div style="width: 300px; height: 300px; border: 2px solid #34495e; border-radius: 50%;
                    position: relative; background: radial-gradient(circle, #ecf0f1, #bdc3c7);">
            <!-- Sphere guidelines -->
            <div style="position: absolute; top: 50%; left: 0; width: 100%; height: 2px; background: #7f8c8d; transform: translateY(-50%);"></div>
            <div style="position: absolute; left: 50%; top: 0; height: 100%; width: 2px; background: #7f8c8d; transform: translateX(-50%);"></div>

            <!-- State vector -->
            <div style="position: absolute; top: 50%; left: 50%; width: 8px; height: 8px;
                        background: #e74c3c; border-radius: 50%; transform: translate(-50%, -50%)
                        translate({}px, {}px);" title="State vector: ({:.3}, {:.3}, {:.3})">
            </div>

            <!-- Axis labels -->
            <div style="position: absolute; top: -25px; left: 50%; transform: translateX(-50%); font-weight: bold;">|0⟩</div>
            <div style="position: absolute; bottom: -25px; left: 50%; transform: translateX(-50%); font-weight: bold;">|1⟩</div>
            <div style="position: absolute; top: 50%; left: -20px; transform: translateY(-50%); font-weight: bold;">Y</div>
            <div style="position: absolute; top: 50%; right: -20px; transform: translateY(-50%); font-weight: bold;">X</div>
        </div>

        <div style="flex: 1;">
            <h4>State Information:</h4>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 4px; font-family: monospace;">
                <div>α = {:.3} + {:.3}i</div>
                <div>β = {:.3} + {:.3}i</div>
                <div style="margin-top: 10px;">
                    <div>θ = {:.3} rad</div>
                    <div>φ = {:.3} rad</div>
                </div>
                <div style="margin-top: 10px;">
                    <div>|α|² = {:.3}</div>
                    <div>|β|² = {:.3}</div>
                </div>
            </div>
        </div>
    </div>
</div>
"#,
            x * 140.0,
            -y * 140.0,
            x,
            y,
            z,
            alpha.re,
            alpha.im,
            beta.re,
            beta.im,
            theta,
            phi,
            alpha.norm_sqr(),
            beta.norm_sqr()
        ))
    }

    /// Get measurement probabilities
    fn measurement_probabilities(&self) -> HashMap<String, f64> {
        let mut probs = HashMap::new();
        for (i, amplitude) in self.state_vector.iter().enumerate() {
            let binary_repr = format!("{:0width$b}", i, width = self.num_qubits);
            let basis_state = format!("|{}⟩", binary_repr);
            probs.insert(basis_state, amplitude.norm_sqr());
        }
        probs
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantumStateVisualizer(qubits={}, dim={})",
            self.num_qubits,
            self.state_vector.len()
        )
    }
}

/// Performance visualization for quantum algorithms
#[pyclass(name = "QuantumPerformanceMonitor")]
pub struct PyQuantumPerformanceMonitor {
    measurements: Vec<PerformanceMeasurement>,
    algorithm_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceMeasurement {
    timestamp: f64,
    operation: String,
    duration_ms: f64,
    fidelity: Option<f64>,
    gate_count: usize,
    qubit_count: usize,
}

#[pymethods]
impl PyQuantumPerformanceMonitor {
    #[new]
    fn new(algorithm_name: String) -> Self {
        Self {
            measurements: Vec::new(),
            algorithm_name,
        }
    }

    /// Add a performance measurement
    fn add_measurement(
        &mut self,
        operation: String,
        duration_ms: f64,
        fidelity: Option<f64>,
        gate_count: usize,
        qubit_count: usize,
    ) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        self.measurements.push(PerformanceMeasurement {
            timestamp,
            operation,
            duration_ms,
            fidelity,
            gate_count,
            qubit_count,
        });
    }

    /// Generate performance timeline visualization
    fn timeline_html(&self) -> String {
        if self.measurements.is_empty() {
            return "<div>No performance data available</div>".to_string();
        }

        let max_duration = self
            .measurements
            .iter()
            .map(|m| m.duration_ms)
            .fold(0.0f64, f64::max);

        let mut html = format!(
            r#"
<div style="font-family: Arial, sans-serif; border: 1px solid #ddd; padding: 20px; margin: 10px; border-radius: 8px;">
    <h3 style="color: #2E86C1; margin-top: 0;">Performance Timeline: {}</h3>
    <div style="margin: 20px 0;">
"#,
            self.algorithm_name
        );

        for (_i, measurement) in self.measurements.iter().enumerate() {
            let width_percent = if max_duration > 0.0 {
                (measurement.duration_ms / max_duration * 80.0).max(2.0)
            } else {
                20.0
            };

            let color = match measurement.operation.as_str() {
                op if op.contains("gate") => "#3498db",
                op if op.contains("measure") => "#e74c3c",
                op if op.contains("compile") => "#f39c12",
                _ => "#95a5a6",
            };

            let fidelity_display = measurement
                .fidelity
                .map(|f| format!(" (F: {:.3})", f))
                .unwrap_or_default();

            html.push_str(&format!(r#"
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <span style="width: 150px; font-size: 12px; color: #34495e;">{}</span>
            <div style="flex: 1; height: 25px; background: #f8f9fa; border: 1px solid #ddd; position: relative; border-radius: 4px;">
                <div style="height: 100%; width: {}%; background: {}; border-radius: 3px;
                            display: flex; align-items: center; padding: 0 8px; color: white; font-size: 11px;"
                     title="{:.2}ms, {} gates, {} qubits{}">
                    {:.1}ms
                </div>
            </div>
        </div>
"#, measurement.operation, width_percent, color, measurement.duration_ms,
    measurement.gate_count, measurement.qubit_count, fidelity_display, measurement.duration_ms));
        }

        let avg_duration = self.measurements.iter().map(|m| m.duration_ms).sum::<f64>()
            / self.measurements.len() as f64;
        html.push_str(&format!(
            r#"
    </div>
    <div style="margin-top: 15px; font-size: 12px; color: #7f8c8d;">
        <strong>Total measurements:</strong> {} | <strong>Average duration:</strong> {:.2}ms
    </div>
</div>
"#,
            self.measurements.len(),
            avg_duration
        ));

        html
    }

    /// Get performance statistics
    fn get_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        if !self.measurements.is_empty() {
            let durations: Vec<f64> = self.measurements.iter().map(|m| m.duration_ms).collect();
            let total_duration: f64 = durations.iter().sum();

            stats.insert("total_duration_ms".to_string(), total_duration);
            stats.insert(
                "average_duration_ms".to_string(),
                total_duration / durations.len() as f64,
            );
            stats.insert(
                "max_duration_ms".to_string(),
                durations.iter().fold(0.0f64, |a, &b| a.max(b)),
            );
            stats.insert(
                "min_duration_ms".to_string(),
                durations.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            );
            stats.insert(
                "measurement_count".to_string(),
                self.measurements.len() as f64,
            );

            let total_gates: usize = self.measurements.iter().map(|m| m.gate_count).sum();
            stats.insert("total_gates".to_string(), total_gates as f64);

            if let Some(avg_fidelity) = self.average_fidelity() {
                stats.insert("average_fidelity".to_string(), avg_fidelity);
            }
        }

        stats
    }

    fn average_fidelity(&self) -> Option<f64> {
        let fidelities: Vec<f64> = self
            .measurements
            .iter()
            .filter_map(|m| m.fidelity)
            .collect();

        if fidelities.is_empty() {
            None
        } else {
            Some(fidelities.iter().sum::<f64>() / fidelities.len() as f64)
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantumPerformanceMonitor(algorithm='{}', measurements={})",
            self.algorithm_name,
            self.measurements.len()
        )
    }
}

/// Module initialization function for Python bindings
pub fn init_jupyter_visualization() {
    // Initialization code for Jupyter visualization tools
}
