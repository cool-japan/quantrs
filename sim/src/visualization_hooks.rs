//! Visualization hooks for quantum simulation debugging and analysis.
//!
//! This module provides comprehensive visualization capabilities for quantum
//! simulations, including real-time state visualization, circuit diagrams,
//! performance plots, and debugging interfaces. It integrates with various
//! visualization frameworks and provides export capabilities for analysis.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use crate::debugger::{ExecutionSnapshot, PerformanceMetrics};
use crate::error::{Result, SimulatorError};

/// Visualization framework types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VisualizationFramework {
    /// Matplotlib-compatible output
    Matplotlib,
    /// Plotly interactive plots
    Plotly,
    /// D3.js web visualization
    D3JS,
    /// Custom SVG output
    SVG,
    /// ASCII art for terminal
    ASCII,
    /// LaTeX/TikZ for publications
    LaTeX,
    /// JSON data export
    JSON,
}

/// Visualization configuration
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Target visualization framework
    pub framework: VisualizationFramework,
    /// Enable real-time visualization
    pub real_time: bool,
    /// Maximum data points to keep in memory
    pub max_data_points: usize,
    /// Export directory for plots
    pub export_directory: String,
    /// Color scheme for plots
    pub color_scheme: ColorScheme,
    /// Enable interactive features
    pub interactive: bool,
    /// Plot dimensions (width, height)
    pub plot_dimensions: (usize, usize),
    /// Enable animation for time series
    pub enable_animation: bool,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            framework: VisualizationFramework::JSON,
            real_time: false,
            max_data_points: 10000,
            export_directory: "./visualization_output".to_string(),
            color_scheme: ColorScheme::Default,
            interactive: false,
            plot_dimensions: (800, 600),
            enable_animation: false,
        }
    }
}

/// Color schemes for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorScheme {
    Default,
    Dark,
    Light,
    Scientific,
    Quantum,
    Accessibility,
}

/// Visualization data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationData {
    /// Quantum state vector amplitudes
    StateVector {
        amplitudes: Vec<Complex64>,
        basis_labels: Vec<String>,
        timestamp: f64,
    },
    /// Circuit diagram data
    CircuitDiagram {
        gates: Vec<GateVisualizationData>,
        num_qubits: usize,
        circuit_depth: usize,
    },
    /// Performance metrics over time
    PerformanceTimeSeries {
        timestamps: Vec<f64>,
        execution_times: Vec<f64>,
        memory_usage: Vec<f64>,
        gate_counts: Vec<HashMap<String, usize>>,
    },
    /// Entanglement structure
    EntanglementVisualization {
        entanglement_matrix: Array2<f64>,
        qubit_labels: Vec<String>,
        bipartite_entropies: Vec<f64>,
    },
    /// Error correction syndrome patterns
    SyndromePattern {
        syndrome_data: Vec<Vec<bool>>,
        error_locations: Vec<usize>,
        correction_success: bool,
        timestamp: f64,
    },
    /// Optimization landscape
    OptimizationLandscape {
        parameter_values: Vec<Vec<f64>>,
        cost_values: Vec<f64>,
        gradient_norms: Vec<f64>,
        parameter_names: Vec<String>,
    },
}

/// Gate visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateVisualizationData {
    pub gate_type: String,
    pub qubits: Vec<usize>,
    pub parameters: Vec<f64>,
    pub position: usize,
    pub execution_time: Option<f64>,
    pub label: Option<String>,
}

/// Visualization hook interface
pub trait VisualizationHook: Send + Sync {
    /// Process visualization data
    fn process_data(&mut self, data: VisualizationData) -> Result<()>;

    /// Export accumulated data
    fn export(&self, path: &str) -> Result<()>;

    /// Clear accumulated data
    fn clear(&mut self);

    /// Get visualization framework name
    fn framework(&self) -> VisualizationFramework;
}

/// Main visualization manager
pub struct VisualizationManager {
    /// Configuration
    config: VisualizationConfig,
    /// Registered visualization hooks
    hooks: Vec<Box<dyn VisualizationHook>>,
    /// Data buffer for real-time visualization
    data_buffer: Arc<Mutex<VecDeque<VisualizationData>>>,
    /// Performance metrics history
    performance_history: Arc<Mutex<Vec<PerformanceMetrics>>>,
    /// Active visualizations
    active_visualizations: HashMap<String, Box<dyn VisualizationHook>>,
}

impl VisualizationManager {
    /// Create new visualization manager
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            config,
            hooks: Vec::new(),
            data_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            performance_history: Arc::new(Mutex::new(Vec::new())),
            active_visualizations: HashMap::new(),
        }
    }

    /// Register a visualization hook
    pub fn register_hook(&mut self, hook: Box<dyn VisualizationHook>) {
        self.hooks.push(hook);
    }

    /// Visualize quantum state
    pub fn visualize_state(
        &mut self,
        state: &Array1<Complex64>,
        label: Option<String>,
    ) -> Result<()> {
        let amplitudes = state.to_vec();
        let basis_labels = self.generate_basis_labels(state.len());
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let data = VisualizationData::StateVector {
            amplitudes,
            basis_labels,
            timestamp,
        };

        self.process_visualization_data(data)?;
        Ok(())
    }

    /// Visualize circuit structure
    pub fn visualize_circuit(&mut self, circuit: &InterfaceCircuit) -> Result<()> {
        let gates = circuit
            .gates
            .iter()
            .enumerate()
            .map(|(pos, gate)| GateVisualizationData {
                gate_type: format!("{:?}", gate.gate_type),
                qubits: gate.qubits.clone(),
                parameters: self.extract_gate_parameters(&gate.gate_type),
                position: pos,
                execution_time: None,
                label: gate.label.clone(),
            })
            .collect();

        let data = VisualizationData::CircuitDiagram {
            gates,
            num_qubits: circuit.num_qubits,
            circuit_depth: circuit.gates.len(),
        };

        self.process_visualization_data(data)?;
        Ok(())
    }

    /// Visualize performance metrics
    pub fn visualize_performance(&mut self, metrics: &PerformanceMetrics) -> Result<()> {
        {
            let mut history = self.performance_history.lock().unwrap();
            history.push(metrics.clone());

            // Keep only recent data
            if history.len() > self.config.max_data_points {
                history.remove(0);
            }
        }

        // Create time series data
        let data = {
            let history = self.performance_history.lock().unwrap();
            let timestamps: Vec<f64> = (0..history.len()).map(|i| i as f64).collect();
            let execution_times: Vec<f64> =
                history.iter().map(|m| m.total_time.as_secs_f64()).collect();
            let memory_usage: Vec<f64> = history
                .iter()
                .map(|m| m.memory_usage.peak_statevector_memory as f64)
                .collect();
            let gate_counts: Vec<HashMap<String, usize>> =
                history.iter().map(|m| m.gate_counts.clone()).collect();

            VisualizationData::PerformanceTimeSeries {
                timestamps,
                execution_times,
                memory_usage,
                gate_counts,
            }
        };

        self.process_visualization_data(data)?;
        Ok(())
    }

    /// Visualize entanglement structure
    pub fn visualize_entanglement(
        &mut self,
        state: &Array1<Complex64>,
        qubit_labels: Option<Vec<String>>,
    ) -> Result<()> {
        let num_qubits = (state.len() as f64).log2().round() as usize;
        let labels =
            qubit_labels.unwrap_or_else(|| (0..num_qubits).map(|i| format!("q{i}")).collect());

        // Calculate entanglement matrix (simplified)
        let entanglement_matrix = self.calculate_entanglement_matrix(state, num_qubits)?;

        // Calculate bipartite entropies
        let bipartite_entropies = self.calculate_bipartite_entropies(state, num_qubits)?;

        let data = VisualizationData::EntanglementVisualization {
            entanglement_matrix,
            qubit_labels: labels,
            bipartite_entropies,
        };

        self.process_visualization_data(data)?;
        Ok(())
    }

    /// Visualize syndrome patterns (for error correction)
    pub fn visualize_syndrome_pattern(
        &mut self,
        syndrome_data: Vec<Vec<bool>>,
        error_locations: Vec<usize>,
        correction_success: bool,
    ) -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let data = VisualizationData::SyndromePattern {
            syndrome_data,
            error_locations,
            correction_success,
            timestamp,
        };

        self.process_visualization_data(data)?;
        Ok(())
    }

    /// Visualize optimization landscape
    pub fn visualize_optimization_landscape(
        &mut self,
        parameter_values: Vec<Vec<f64>>,
        cost_values: Vec<f64>,
        gradient_norms: Vec<f64>,
        parameter_names: Vec<String>,
    ) -> Result<()> {
        let data = VisualizationData::OptimizationLandscape {
            parameter_values,
            cost_values,
            gradient_norms,
            parameter_names,
        };

        self.process_visualization_data(data)?;
        Ok(())
    }

    /// Process visualization data through all hooks
    fn process_visualization_data(&mut self, data: VisualizationData) -> Result<()> {
        // Add to buffer for real-time processing
        if self.config.real_time {
            let mut buffer = self.data_buffer.lock().unwrap();
            buffer.push_back(data.clone());
            if buffer.len() > self.config.max_data_points {
                buffer.pop_front();
            }
        }

        // Process through all hooks
        for hook in &mut self.hooks {
            hook.process_data(data.clone())?;
        }

        Ok(())
    }

    /// Export all visualization data
    pub fn export_all(&self, base_path: &str) -> Result<()> {
        std::fs::create_dir_all(base_path).map_err(|e| {
            SimulatorError::InvalidInput(format!("Failed to create export directory: {e}"))
        })?;

        for (i, hook) in self.hooks.iter().enumerate() {
            let export_path = format!(
                "{}/visualization_{}.{}",
                base_path,
                i,
                self.get_file_extension(hook.framework())
            );
            hook.export(&export_path)?;
        }

        Ok(())
    }

    /// Clear all visualization data
    pub fn clear_all(&mut self) {
        for hook in &mut self.hooks {
            hook.clear();
        }

        self.data_buffer.lock().unwrap().clear();
        self.performance_history.lock().unwrap().clear();
    }

    /// Generate basis state labels
    fn generate_basis_labels(&self, state_size: usize) -> Vec<String> {
        let num_qubits = (state_size as f64).log2().round() as usize;
        (0..state_size)
            .map(|i| format!("|{i:0num_qubits$b}⟩"))
            .collect()
    }

    /// Extract parameters from gate type
    fn extract_gate_parameters(&self, gate_type: &InterfaceGateType) -> Vec<f64> {
        match gate_type {
            InterfaceGateType::Phase(angle) => vec![*angle],
            InterfaceGateType::RX(angle) => vec![*angle],
            InterfaceGateType::RY(angle) => vec![*angle],
            InterfaceGateType::RZ(angle) => vec![*angle],
            InterfaceGateType::U1(angle) => vec![*angle],
            InterfaceGateType::U2(theta, phi) => vec![*theta, *phi],
            InterfaceGateType::U3(theta, phi, lambda) => vec![*theta, *phi, *lambda],
            InterfaceGateType::CRX(angle) => vec![*angle],
            InterfaceGateType::CRY(angle) => vec![*angle],
            InterfaceGateType::CRZ(angle) => vec![*angle],
            InterfaceGateType::CPhase(angle) => vec![*angle],
            _ => Vec::new(),
        }
    }

    /// Calculate entanglement matrix between qubits
    fn calculate_entanglement_matrix(
        &self,
        state: &Array1<Complex64>,
        num_qubits: usize,
    ) -> Result<Array2<f64>> {
        let mut entanglement_matrix = Array2::zeros((num_qubits, num_qubits));

        // Simplified entanglement measure (mutual information approximation)
        for i in 0..num_qubits {
            for j in i..num_qubits {
                if i == j {
                    entanglement_matrix[[i, j]] = 1.0;
                } else {
                    // Calculate mutual information between qubits i and j
                    let mutual_info = self.calculate_mutual_information(state, i, j, num_qubits)?;
                    entanglement_matrix[[i, j]] = mutual_info;
                    entanglement_matrix[[j, i]] = mutual_info;
                }
            }
        }

        Ok(entanglement_matrix)
    }

    /// Calculate bipartite entropies for different cuts
    fn calculate_bipartite_entropies(
        &self,
        state: &Array1<Complex64>,
        num_qubits: usize,
    ) -> Result<Vec<f64>> {
        let mut entropies = Vec::new();

        for cut in 1..num_qubits {
            let entropy = self.calculate_bipartite_entropy(state, cut, num_qubits)?;
            entropies.push(entropy);
        }

        Ok(entropies)
    }

    /// Calculate mutual information between two qubits (simplified)
    const fn calculate_mutual_information(
        &self,
        _state: &Array1<Complex64>,
        _qubit_i: usize,
        _qubit_j: usize,
        _num_qubits: usize,
    ) -> Result<f64> {
        // Simplified placeholder - in practice would compute reduced density matrices
        Ok(0.5)
    }

    /// Calculate bipartite entropy for a given cut
    fn calculate_bipartite_entropy(
        &self,
        state: &Array1<Complex64>,
        cut: usize,
        num_qubits: usize,
    ) -> Result<f64> {
        // Simplified von Neumann entropy calculation
        let left_size = 1 << cut;
        let right_size = 1 << (num_qubits - cut);

        // Calculate reduced density matrix for left subsystem (simplified)
        let mut reduced_dm = Array2::zeros((left_size, left_size));

        for i in 0..left_size {
            for j in 0..left_size {
                let mut trace_val = Complex64::new(0.0, 0.0);
                for k in 0..right_size {
                    let idx1 = i * right_size + k;
                    let idx2 = j * right_size + k;
                    if idx1 < state.len() && idx2 < state.len() {
                        trace_val += state[idx1] * state[idx2].conj();
                    }
                }
                reduced_dm[[i, j]] = trace_val;
            }
        }

        // Calculate von Neumann entropy (simplified)
        let mut entropy = 0.0;
        for i in 0..left_size {
            let eigenval = reduced_dm[[i, i]].norm();
            if eigenval > 1e-10 {
                entropy -= eigenval * eigenval.ln();
            }
        }

        Ok(entropy)
    }

    /// Get file extension for visualization framework
    const fn get_file_extension(&self, framework: VisualizationFramework) -> &str {
        match framework {
            VisualizationFramework::Matplotlib => "py",
            VisualizationFramework::Plotly => "html",
            VisualizationFramework::D3JS => "html",
            VisualizationFramework::SVG => "svg",
            VisualizationFramework::ASCII => "txt",
            VisualizationFramework::LaTeX => "tex",
            VisualizationFramework::JSON => "json",
        }
    }
}

/// JSON export visualization hook
pub struct JSONVisualizationHook {
    /// Accumulated visualization data
    data: Vec<VisualizationData>,
    /// Export configuration
    config: VisualizationConfig,
}

impl JSONVisualizationHook {
    pub const fn new(config: VisualizationConfig) -> Self {
        Self {
            data: Vec::new(),
            config,
        }
    }
}

impl VisualizationHook for JSONVisualizationHook {
    fn process_data(&mut self, data: VisualizationData) -> Result<()> {
        self.data.push(data);

        // Keep only recent data
        if self.data.len() > self.config.max_data_points {
            self.data.remove(0);
        }

        Ok(())
    }

    fn export(&self, path: &str) -> Result<()> {
        let json_data = serde_json::to_string_pretty(&self.data)
            .map_err(|e| SimulatorError::InvalidInput(format!("Failed to serialize data: {e}")))?;

        let mut file = File::create(path)
            .map_err(|e| SimulatorError::InvalidInput(format!("Failed to create file: {e}")))?;

        file.write_all(json_data.as_bytes())
            .map_err(|e| SimulatorError::InvalidInput(format!("Failed to write file: {e}")))?;

        Ok(())
    }

    fn clear(&mut self) {
        self.data.clear();
    }

    fn framework(&self) -> VisualizationFramework {
        VisualizationFramework::JSON
    }
}

/// ASCII visualization hook for terminal output
pub struct ASCIIVisualizationHook {
    /// Recent state vectors for display
    recent_states: VecDeque<Array1<Complex64>>,
    /// Configuration
    config: VisualizationConfig,
}

impl ASCIIVisualizationHook {
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            recent_states: VecDeque::with_capacity(100),
            config,
        }
    }

    /// Generate ASCII representation of quantum state
    fn state_to_ascii(&self, state: &Array1<Complex64>) -> String {
        let mut output = String::new();
        output.push_str("Quantum State Visualization:\n");
        output.push_str("==========================\n");

        for (i, amplitude) in state.iter().enumerate().take(16) {
            let magnitude = amplitude.norm();
            let phase = amplitude.arg();

            let bar_length = (magnitude * 20.0) as usize;
            let bar = "█".repeat(bar_length) + &"░".repeat(20 - bar_length);

            output.push_str(&format!(
                "|{:02}⟩: {} {:.4} ∠{:.2}π\n",
                i,
                bar,
                magnitude,
                phase / std::f64::consts::PI
            ));
        }

        if state.len() > 16 {
            output.push_str(&format!("... ({} more states)\n", state.len() - 16));
        }

        output
    }
}

impl VisualizationHook for ASCIIVisualizationHook {
    fn process_data(&mut self, data: VisualizationData) -> Result<()> {
        match data {
            VisualizationData::StateVector { amplitudes, .. } => {
                let state = Array1::from_vec(amplitudes);

                if self.config.real_time {
                    println!("{}", self.state_to_ascii(&state));
                }

                self.recent_states.push_back(state);
                if self.recent_states.len() > 100 {
                    self.recent_states.pop_front();
                }
            }
            VisualizationData::CircuitDiagram {
                gates, num_qubits, ..
            } => {
                if self.config.real_time {
                    println!(
                        "Circuit Diagram ({} qubits, {} gates):",
                        num_qubits,
                        gates.len()
                    );
                    for gate in gates.iter().take(10) {
                        println!("  {} on qubits {:?}", gate.gate_type, gate.qubits);
                    }
                    if gates.len() > 10 {
                        println!("  ... ({} more gates)", gates.len() - 10);
                    }
                }
            }
            _ => {
                // Handle other data types
            }
        }

        Ok(())
    }

    fn export(&self, path: &str) -> Result<()> {
        let mut output = String::new();
        output.push_str("ASCII Visualization Export\n");
        output.push_str("==========================\n\n");

        for (i, state) in self.recent_states.iter().enumerate() {
            output.push_str(&format!("State {i}:\n"));
            output.push_str(&self.state_to_ascii(state));
            output.push('\n');
        }

        let mut file = File::create(path)
            .map_err(|e| SimulatorError::InvalidInput(format!("Failed to create file: {e}")))?;

        file.write_all(output.as_bytes())
            .map_err(|e| SimulatorError::InvalidInput(format!("Failed to write file: {e}")))?;

        Ok(())
    }

    fn clear(&mut self) {
        self.recent_states.clear();
    }

    fn framework(&self) -> VisualizationFramework {
        VisualizationFramework::ASCII
    }
}

/// Benchmark visualization performance
pub fn benchmark_visualization() -> Result<HashMap<String, f64>> {
    let mut results = HashMap::new();

    // Test JSON hook performance
    let start = std::time::Instant::now();
    let mut json_hook = JSONVisualizationHook::new(VisualizationConfig::default());

    for i in 0..1000 {
        let test_state = Array1::from_vec(vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ]);

        let data = VisualizationData::StateVector {
            amplitudes: test_state.to_vec(),
            basis_labels: vec![
                "00".to_string(),
                "01".to_string(),
                "10".to_string(),
                "11".to_string(),
            ],
            timestamp: i as f64,
        };

        json_hook.process_data(data)?;
    }

    let json_time = start.elapsed().as_millis() as f64;
    results.insert("json_hook_1000_states".to_string(), json_time);

    // Test ASCII hook performance
    let start = std::time::Instant::now();
    let mut ascii_hook = ASCIIVisualizationHook::new(VisualizationConfig {
        real_time: false,
        ..Default::default()
    });

    for i in 0..100 {
        let test_state = Array1::from_vec(vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ]);

        let data = VisualizationData::StateVector {
            amplitudes: test_state.to_vec(),
            basis_labels: vec![
                "00".to_string(),
                "01".to_string(),
                "10".to_string(),
                "11".to_string(),
            ],
            timestamp: i as f64,
        };

        ascii_hook.process_data(data)?;
    }

    let ascii_time = start.elapsed().as_millis() as f64;
    results.insert("ascii_hook_100_states".to_string(), ascii_time);

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_visualization_manager_creation() {
        let config = VisualizationConfig::default();
        let manager = VisualizationManager::new(config);
        assert_eq!(manager.hooks.len(), 0);
    }

    #[test]
    fn test_json_hook() {
        let mut hook = JSONVisualizationHook::new(VisualizationConfig::default());

        let test_data = VisualizationData::StateVector {
            amplitudes: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            basis_labels: vec!["0".to_string(), "1".to_string()],
            timestamp: 0.0,
        };

        assert!(hook.process_data(test_data).is_ok());
        assert_eq!(hook.data.len(), 1);
    }

    #[test]
    fn test_ascii_hook() {
        let mut hook = ASCIIVisualizationHook::new(VisualizationConfig {
            real_time: false,
            ..Default::default()
        });

        let test_data = VisualizationData::StateVector {
            amplitudes: vec![Complex64::new(0.707, 0.0), Complex64::new(0.707, 0.0)],
            basis_labels: vec!["0".to_string(), "1".to_string()],
            timestamp: 0.0,
        };

        assert!(hook.process_data(test_data).is_ok());
        assert_eq!(hook.recent_states.len(), 1);
    }

    #[test]
    fn test_state_visualization() {
        let config = VisualizationConfig::default();
        let mut manager = VisualizationManager::new(config);

        let test_state = Array1::from_vec(vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ]);

        assert!(manager.visualize_state(&test_state, None).is_ok());
    }

    #[test]
    fn test_circuit_visualization() {
        let config = VisualizationConfig::default();
        let mut manager = VisualizationManager::new(config);

        let mut circuit = InterfaceCircuit::new(2, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]));

        assert!(manager.visualize_circuit(&circuit).is_ok());
    }

    #[test]
    fn test_parameter_extraction() {
        let config = VisualizationConfig::default();
        let manager = VisualizationManager::new(config);

        let params = manager.extract_gate_parameters(&InterfaceGateType::RX(1.5));
        assert_eq!(params, vec![1.5]);

        let params = manager.extract_gate_parameters(&InterfaceGateType::U3(1.0, 2.0, 3.0));
        assert_eq!(params, vec![1.0, 2.0, 3.0]);

        let params = manager.extract_gate_parameters(&InterfaceGateType::Hadamard);
        assert_eq!(params.len(), 0);
    }

    #[test]
    fn test_basis_label_generation() {
        let config = VisualizationConfig::default();
        let manager = VisualizationManager::new(config);

        let labels = manager.generate_basis_labels(4);
        assert_eq!(labels, vec!["|00⟩", "|01⟩", "|10⟩", "|11⟩"]);

        let labels = manager.generate_basis_labels(8);
        assert_eq!(labels.len(), 8);
        assert_eq!(labels[0], "|000⟩");
        assert_eq!(labels[7], "|111⟩");
    }

    #[test]
    fn test_entanglement_calculation() {
        let config = VisualizationConfig::default();
        let manager = VisualizationManager::new(config);

        let bell_state = Array1::from_vec(vec![
            Complex64::new(0.707, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.707, 0.0),
        ]);

        let entanglement_matrix = manager
            .calculate_entanglement_matrix(&bell_state, 2)
            .unwrap();
        assert_eq!(entanglement_matrix.shape(), [2, 2]);

        let entropies = manager
            .calculate_bipartite_entropies(&bell_state, 2)
            .unwrap();
        assert_eq!(entropies.len(), 1);
    }
}
