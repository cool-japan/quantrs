//! Hardware noise model integration for quantum circuits
//!
//! This module provides comprehensive noise modeling capabilities for various quantum
//! hardware platforms, including gate errors, decoherence, crosstalk, and readout errors.

use crate::builder::Circuit;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Comprehensive noise model for quantum devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseModel {
    /// Single-qubit gate errors
    pub single_qubit_errors: HashMap<String, SingleQubitError>,
    /// Two-qubit gate errors
    pub two_qubit_errors: HashMap<String, TwoQubitError>,
    /// Qubit decoherence parameters
    pub decoherence: HashMap<usize, DecoherenceParams>,
    /// Readout errors
    pub readout_errors: HashMap<usize, ReadoutError>,
    /// Crosstalk parameters
    pub crosstalk: Option<CrosstalkModel>,
    /// Thermal noise
    pub thermal_noise: Option<ThermalNoise>,
    /// Leakage errors
    pub leakage_errors: HashMap<usize, LeakageError>,
    /// Calibration timestamp
    pub calibration_time: std::time::SystemTime,
}

/// Single-qubit gate error model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleQubitError {
    /// Depolarizing error probability
    pub depolarizing: f64,
    /// Pauli X error probability
    pub pauli_x: f64,
    /// Pauli Y error probability
    pub pauli_y: f64,
    /// Pauli Z error probability
    pub pauli_z: f64,
    /// Amplitude damping probability
    pub amplitude_damping: f64,
    /// Phase damping probability
    pub phase_damping: f64,
    /// Gate duration in nanoseconds
    pub duration: f64,
}

/// Two-qubit gate error model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoQubitError {
    /// Depolarizing error probability
    pub depolarizing: f64,
    /// Individual Pauli error probabilities (16 combinations)
    pub pauli_errors: [[f64; 4]; 4], // [I,X,Y,Z] x [I,X,Y,Z]
    /// Amplitude damping for both qubits
    pub amplitude_damping: [f64; 2],
    /// Phase damping for both qubits
    pub phase_damping: [f64; 2],
    /// Gate duration in nanoseconds
    pub duration: f64,
    /// Crosstalk coupling strength
    pub crosstalk_strength: f64,
}

/// Decoherence parameters for a qubit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoherenceParams {
    /// T1 relaxation time in microseconds
    pub t1: f64,
    /// T2 dephasing time in microseconds
    pub t2: f64,
    /// T2* pure dephasing time in microseconds
    pub t2_star: f64,
    /// Effective temperature in mK
    pub temperature: f64,
}

/// Readout error model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadoutError {
    /// Probability of reading |1⟩ when state is |0⟩
    pub prob_0_to_1: f64,
    /// Probability of reading |0⟩ when state is |1⟩
    pub prob_1_to_0: f64,
    /// Readout fidelity
    pub fidelity: f64,
    /// Measurement duration in nanoseconds
    pub duration: f64,
}

/// Crosstalk model between qubits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrosstalkModel {
    /// Coupling matrix (symmetric)
    pub coupling_matrix: Vec<Vec<f64>>,
    /// ZZ coupling strengths
    pub zz_coupling: HashMap<(usize, usize), f64>,
    /// XY coupling strengths
    pub xy_coupling: HashMap<(usize, usize), f64>,
    /// Frequency shifts due to neighboring qubits
    pub frequency_shifts: HashMap<usize, Vec<(usize, f64)>>,
}

/// Thermal noise parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalNoise {
    /// Ambient temperature in mK
    pub temperature: f64,
    /// Thermal photon population
    pub thermal_photons: f64,
    /// Heating rate per second
    pub heating_rate: f64,
}

/// Leakage error model (transitions to non-computational states)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakageError {
    /// Probability of leaking to |2⟩ state
    pub leakage_to_2: f64,
    /// Probability of leaking to higher states
    pub leakage_to_higher: f64,
    /// Seepage probability (returning from leakage states)
    pub seepage: f64,
}

/// Noise-aware circuit analysis and optimization
pub struct NoiseAnalyzer {
    noise_models: HashMap<String, NoiseModel>,
}

impl NoiseAnalyzer {
    /// Create a new noise analyzer
    pub fn new() -> Self {
        let mut analyzer = Self {
            noise_models: HashMap::new(),
        };

        // Load common device noise models
        analyzer.load_device_noise_models();
        analyzer
    }

    /// Add a custom noise model
    pub fn add_noise_model(&mut self, device: String, model: NoiseModel) {
        self.noise_models.insert(device, model);
    }

    /// Get available noise models
    pub fn available_models(&self) -> Vec<String> {
        self.noise_models.keys().cloned().collect()
    }

    /// Analyze circuit noise properties
    pub fn analyze_circuit_noise<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        device: &str,
    ) -> QuantRS2Result<NoiseAnalysisResult> {
        let noise_model = self
            .noise_models
            .get(device)
            .ok_or_else(|| QuantRS2Error::InvalidInput(format!("Unknown device: {}", device)))?;

        let mut total_error = 0.0;
        let mut gate_errors = Vec::new();
        let mut decoherence_error = 0.0;
        let mut readout_error = 0.0;
        let mut crosstalk_error = 0.0;

        // Analyze each gate
        for (i, gate) in circuit.gates().iter().enumerate() {
            let gate_error = self.calculate_gate_error(gate.as_ref(), noise_model)?;
            gate_errors.push((i, gate.name().to_string(), gate_error));
            total_error += gate_error;
        }

        // Calculate decoherence errors
        decoherence_error = self.calculate_decoherence_error(circuit, noise_model)?;

        // Calculate readout errors
        readout_error = self.calculate_readout_error(N, noise_model);

        // Calculate crosstalk errors
        if let Some(crosstalk) = &noise_model.crosstalk {
            crosstalk_error = self.calculate_crosstalk_error(circuit, crosstalk)?;
        }

        let total_fidelity =
            1.0 - (total_error + decoherence_error + readout_error + crosstalk_error);

        Ok(NoiseAnalysisResult {
            total_error: total_error + decoherence_error + readout_error + crosstalk_error,
            total_fidelity,
            gate_errors,
            decoherence_error,
            readout_error,
            crosstalk_error,
            dominant_error_source: self.identify_dominant_error_source(
                total_error,
                decoherence_error,
                readout_error,
                crosstalk_error,
            ),
        })
    }

    /// Calculate error for a single gate
    fn calculate_gate_error(
        &self,
        gate: &dyn GateOp,
        noise_model: &NoiseModel,
    ) -> QuantRS2Result<f64> {
        let gate_name = gate.name();
        let qubits = gate.qubits();

        match qubits.len() {
            1 => {
                if let Some(error) = noise_model.single_qubit_errors.get(gate_name) {
                    Ok(error.depolarizing + error.amplitude_damping + error.phase_damping)
                } else {
                    // Use average single-qubit error if specific gate not found
                    let avg_error = noise_model
                        .single_qubit_errors
                        .values()
                        .map(|e| e.depolarizing + e.amplitude_damping + e.phase_damping)
                        .sum::<f64>()
                        / noise_model.single_qubit_errors.len() as f64;
                    Ok(avg_error)
                }
            }
            2 => {
                if let Some(error) = noise_model.two_qubit_errors.get(gate_name) {
                    Ok(error.depolarizing
                        + error.amplitude_damping.iter().sum::<f64>()
                        + error.phase_damping.iter().sum::<f64>())
                } else {
                    // Use average two-qubit error if specific gate not found
                    let avg_error = noise_model
                        .two_qubit_errors
                        .values()
                        .map(|e| {
                            e.depolarizing
                                + e.amplitude_damping.iter().sum::<f64>()
                                + e.phase_damping.iter().sum::<f64>()
                        })
                        .sum::<f64>()
                        / noise_model.two_qubit_errors.len() as f64;
                    Ok(avg_error)
                }
            }
            _ => {
                // Multi-qubit gates - estimate based on constituent gates
                Ok(0.01 * qubits.len() as f64) // Rough estimate
            }
        }
    }

    /// Calculate decoherence error over circuit execution
    fn calculate_decoherence_error<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        noise_model: &NoiseModel,
    ) -> QuantRS2Result<f64> {
        let total_time = self.estimate_circuit_time(circuit, noise_model);
        let mut total_decoherence = 0.0;

        for qubit_id in 0..N {
            if let Some(decoherence) = noise_model.decoherence.get(&qubit_id) {
                // T1 relaxation error
                let t1_error = 1.0 - (-total_time / decoherence.t1).exp();

                // T2 dephasing error
                let t2_error = 1.0 - (-total_time / decoherence.t2).exp();

                total_decoherence += t1_error + t2_error;
            }
        }

        Ok(total_decoherence / N as f64)
    }

    /// Estimate total circuit execution time
    fn estimate_circuit_time<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        noise_model: &NoiseModel,
    ) -> f64 {
        let mut total_time = 0.0;

        for gate in circuit.gates() {
            let gate_name = gate.name();
            let qubits = gate.qubits();

            let duration = match qubits.len() {
                1 => noise_model
                    .single_qubit_errors
                    .get(gate_name)
                    .map(|e| e.duration)
                    .unwrap_or(10.0), // Default 10ns
                2 => noise_model
                    .two_qubit_errors
                    .get(gate_name)
                    .map(|e| e.duration)
                    .unwrap_or(200.0), // Default 200ns
                _ => 500.0, // Multi-qubit gates take longer
            };

            total_time += duration;
        }

        total_time / 1000.0 // Convert to microseconds
    }

    /// Calculate readout error
    fn calculate_readout_error(&self, num_qubits: usize, noise_model: &NoiseModel) -> f64 {
        let mut total_readout_error = 0.0;

        for qubit_id in 0..num_qubits {
            if let Some(readout) = noise_model.readout_errors.get(&qubit_id) {
                total_readout_error += 1.0 - readout.fidelity;
            }
        }

        total_readout_error / num_qubits as f64
    }

    /// Calculate crosstalk error
    fn calculate_crosstalk_error<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        crosstalk: &CrosstalkModel,
    ) -> QuantRS2Result<f64> {
        let mut total_crosstalk = 0.0;

        for gate in circuit.gates() {
            if gate.qubits().len() == 2 {
                let qubits: Vec<_> = gate.qubits().iter().map(|q| q.id() as usize).collect();
                let q1 = qubits[0];
                let q2 = qubits[1];

                // ZZ crosstalk
                if let Some(zz_strength) = crosstalk.zz_coupling.get(&(q1, q2)) {
                    total_crosstalk += zz_strength.abs();
                }

                // XY crosstalk
                if let Some(xy_strength) = crosstalk.xy_coupling.get(&(q1, q2)) {
                    total_crosstalk += xy_strength.abs();
                }
            }
        }

        Ok(total_crosstalk / circuit.gates().len() as f64)
    }

    /// Identify the dominant source of error
    fn identify_dominant_error_source(
        &self,
        gate_error: f64,
        decoherence_error: f64,
        readout_error: f64,
        crosstalk_error: f64,
    ) -> ErrorSource {
        let max_error = gate_error
            .max(decoherence_error)
            .max(readout_error)
            .max(crosstalk_error);

        if max_error == gate_error {
            ErrorSource::GateErrors
        } else if max_error == decoherence_error {
            ErrorSource::Decoherence
        } else if max_error == readout_error {
            ErrorSource::Readout
        } else {
            ErrorSource::Crosstalk
        }
    }

    /// Load device-specific noise models
    fn load_device_noise_models(&mut self) {
        // IBM Quantum noise model
        self.add_noise_model("ibm_quantum".to_string(), NoiseModel::ibm_quantum());

        // Google Quantum AI noise model
        self.add_noise_model("google_quantum".to_string(), NoiseModel::google_quantum());

        // AWS Braket noise model
        self.add_noise_model("aws_braket".to_string(), NoiseModel::aws_braket());
    }
}

/// Result of noise analysis
#[derive(Debug, Clone)]
pub struct NoiseAnalysisResult {
    /// Total error probability
    pub total_error: f64,
    /// Overall circuit fidelity
    pub total_fidelity: f64,
    /// Individual gate errors (index, gate_name, error)
    pub gate_errors: Vec<(usize, String, f64)>,
    /// Decoherence contribution
    pub decoherence_error: f64,
    /// Readout error contribution
    pub readout_error: f64,
    /// Crosstalk error contribution
    pub crosstalk_error: f64,
    /// Dominant error source
    pub dominant_error_source: ErrorSource,
}

/// Primary sources of quantum errors
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorSource {
    GateErrors,
    Decoherence,
    Readout,
    Crosstalk,
}

impl NoiseModel {
    /// Create IBM Quantum noise model based on typical device characteristics
    pub fn ibm_quantum() -> Self {
        let mut single_qubit_errors = HashMap::new();

        // Typical IBM single-qubit gate errors
        single_qubit_errors.insert(
            "X".to_string(),
            SingleQubitError {
                depolarizing: 0.0001,
                pauli_x: 0.00005,
                pauli_y: 0.00005,
                pauli_z: 0.0001,
                amplitude_damping: 0.0002,
                phase_damping: 0.0003,
                duration: 35.0, // nanoseconds
            },
        );

        single_qubit_errors.insert(
            "RZ".to_string(),
            SingleQubitError {
                depolarizing: 0.0,
                pauli_x: 0.0,
                pauli_y: 0.0,
                pauli_z: 0.00001,
                amplitude_damping: 0.0,
                phase_damping: 0.00002,
                duration: 0.0, // Virtual gate
            },
        );

        let mut two_qubit_errors = HashMap::new();

        // CNOT gate error
        two_qubit_errors.insert(
            "CNOT".to_string(),
            TwoQubitError {
                depolarizing: 0.01,
                pauli_errors: [[0.0; 4]; 4], // Simplified for now
                amplitude_damping: [0.002, 0.002],
                phase_damping: [0.003, 0.003],
                duration: 300.0, // nanoseconds
                crosstalk_strength: 0.001,
            },
        );

        let mut decoherence = HashMap::new();
        for i in 0..127 {
            decoherence.insert(
                i,
                DecoherenceParams {
                    t1: 100.0, // microseconds
                    t2: 80.0,  // microseconds
                    t2_star: 70.0,
                    temperature: 15.0, // mK
                },
            );
        }

        let mut readout_errors = HashMap::new();
        for i in 0..127 {
            readout_errors.insert(
                i,
                ReadoutError {
                    prob_0_to_1: 0.01,
                    prob_1_to_0: 0.02,
                    fidelity: 0.985,
                    duration: 1000.0, // nanoseconds
                },
            );
        }

        Self {
            single_qubit_errors,
            two_qubit_errors,
            decoherence,
            readout_errors,
            crosstalk: Some(CrosstalkModel::default()),
            thermal_noise: Some(ThermalNoise {
                temperature: 15.0,
                thermal_photons: 0.001,
                heating_rate: 0.0001,
            }),
            leakage_errors: HashMap::new(),
            calibration_time: std::time::SystemTime::now(),
        }
    }

    /// Create Google Quantum AI noise model
    pub fn google_quantum() -> Self {
        let mut single_qubit_errors = HashMap::new();

        single_qubit_errors.insert(
            "RZ".to_string(),
            SingleQubitError {
                depolarizing: 0.0,
                pauli_x: 0.0,
                pauli_y: 0.0,
                pauli_z: 0.00001,
                amplitude_damping: 0.0,
                phase_damping: 0.00001,
                duration: 0.0,
            },
        );

        single_qubit_errors.insert(
            "SQRT_X".to_string(),
            SingleQubitError {
                depolarizing: 0.0005,
                pauli_x: 0.0002,
                pauli_y: 0.0002,
                pauli_z: 0.0001,
                amplitude_damping: 0.0001,
                phase_damping: 0.0002,
                duration: 25.0,
            },
        );

        let mut two_qubit_errors = HashMap::new();

        two_qubit_errors.insert(
            "CZ".to_string(),
            TwoQubitError {
                depolarizing: 0.005,
                pauli_errors: [[0.0; 4]; 4],
                amplitude_damping: [0.001, 0.001],
                phase_damping: [0.002, 0.002],
                duration: 20.0,
                crosstalk_strength: 0.0005,
            },
        );

        let mut decoherence = HashMap::new();
        for i in 0..70 {
            decoherence.insert(
                i,
                DecoherenceParams {
                    t1: 50.0,
                    t2: 40.0,
                    t2_star: 35.0,
                    temperature: 10.0,
                },
            );
        }

        let mut readout_errors = HashMap::new();
        for i in 0..70 {
            readout_errors.insert(
                i,
                ReadoutError {
                    prob_0_to_1: 0.005,
                    prob_1_to_0: 0.008,
                    fidelity: 0.99,
                    duration: 500.0,
                },
            );
        }

        Self {
            single_qubit_errors,
            two_qubit_errors,
            decoherence,
            readout_errors,
            crosstalk: Some(CrosstalkModel::default()),
            thermal_noise: Some(ThermalNoise {
                temperature: 10.0,
                thermal_photons: 0.0005,
                heating_rate: 0.00005,
            }),
            leakage_errors: HashMap::new(),
            calibration_time: std::time::SystemTime::now(),
        }
    }

    /// Create AWS Braket noise model
    pub fn aws_braket() -> Self {
        // Simplified model that varies by backend
        let mut single_qubit_errors = HashMap::new();

        single_qubit_errors.insert(
            "RZ".to_string(),
            SingleQubitError {
                depolarizing: 0.0001,
                pauli_x: 0.00005,
                pauli_y: 0.00005,
                pauli_z: 0.00002,
                amplitude_damping: 0.0001,
                phase_damping: 0.0002,
                duration: 0.0,
            },
        );

        let mut two_qubit_errors = HashMap::new();

        two_qubit_errors.insert(
            "CNOT".to_string(),
            TwoQubitError {
                depolarizing: 0.008,
                pauli_errors: [[0.0; 4]; 4],
                amplitude_damping: [0.0015, 0.0015],
                phase_damping: [0.0025, 0.0025],
                duration: 200.0,
                crosstalk_strength: 0.0008,
            },
        );

        let mut decoherence = HashMap::new();
        for i in 0..100 {
            decoherence.insert(
                i,
                DecoherenceParams {
                    t1: 80.0,
                    t2: 60.0,
                    t2_star: 50.0,
                    temperature: 12.0,
                },
            );
        }

        let mut readout_errors = HashMap::new();
        for i in 0..100 {
            readout_errors.insert(
                i,
                ReadoutError {
                    prob_0_to_1: 0.008,
                    prob_1_to_0: 0.012,
                    fidelity: 0.988,
                    duration: 800.0,
                },
            );
        }

        Self {
            single_qubit_errors,
            two_qubit_errors,
            decoherence,
            readout_errors,
            crosstalk: Some(CrosstalkModel::default()),
            thermal_noise: Some(ThermalNoise {
                temperature: 12.0,
                thermal_photons: 0.0008,
                heating_rate: 0.00008,
            }),
            leakage_errors: HashMap::new(),
            calibration_time: std::time::SystemTime::now(),
        }
    }
}

impl Default for CrosstalkModel {
    fn default() -> Self {
        Self {
            coupling_matrix: Vec::new(),
            zz_coupling: HashMap::new(),
            xy_coupling: HashMap::new(),
            frequency_shifts: HashMap::new(),
        }
    }
}

impl Default for NoiseAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::Hadamard;

    #[test]
    fn test_noise_analyzer_creation() {
        let analyzer = NoiseAnalyzer::new();
        assert!(!analyzer.available_models().is_empty());
    }

    #[test]
    fn test_noise_model_creation() {
        let model = NoiseModel::ibm_quantum();
        assert!(!model.single_qubit_errors.is_empty());
        assert!(!model.two_qubit_errors.is_empty());
        assert!(!model.decoherence.is_empty());
    }

    #[test]
    fn test_circuit_noise_analysis() {
        let analyzer = NoiseAnalyzer::new();
        let mut circuit = Circuit::<2>::new();
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();

        let result = analyzer
            .analyze_circuit_noise(&circuit, "ibm_quantum")
            .unwrap();
        assert!(result.total_fidelity > 0.0 && result.total_fidelity < 1.0);
        assert!(!result.gate_errors.is_empty());
    }

    #[test]
    fn test_single_qubit_error_calculation() {
        let analyzer = NoiseAnalyzer::new();
        let model = NoiseModel::ibm_quantum();
        let h_gate = Hadamard { target: QubitId(0) };

        let error = analyzer.calculate_gate_error(&h_gate, &model).unwrap();
        assert!(error > 0.0);
    }

    #[test]
    fn test_decoherence_params() {
        let params = DecoherenceParams {
            t1: 100.0,
            t2: 80.0,
            t2_star: 70.0,
            temperature: 15.0,
        };

        assert!(params.t1 > params.t2);
        assert!(params.t2 > params.t2_star);
    }

    #[test]
    fn test_readout_error() {
        let error = ReadoutError {
            prob_0_to_1: 0.01,
            prob_1_to_0: 0.02,
            fidelity: 0.985,
            duration: 1000.0,
        };

        assert!(error.fidelity < 1.0);
        assert!(error.prob_0_to_1 + error.prob_1_to_0 < 1.0);
    }
}
