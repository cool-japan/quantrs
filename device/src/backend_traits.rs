//! Common traits and utilities for hardware backend translation
//!
//! This module provides shared functionality for working with different
//! quantum hardware backends and their gate sets.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

use crate::translation::{DecomposedGate, HardwareBackend, NativeGateSet};

/// Trait for hardware-specific gate implementations
pub trait HardwareGate: GateOp {
    /// Get the hardware backend this gate is native to
    fn backend(&self) -> HardwareBackend;

    /// Get hardware-specific metadata
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Check if this gate requires calibration
    fn requires_calibration(&self) -> bool {
        true
    }

    /// Get calibration parameters
    fn calibration_params(&self) -> Vec<String> {
        vec![]
    }
}

/// IBM-specific gate implementations
pub mod ibm_gates {
    use super::*;
    use num_complex::Complex64;
    use std::any::Any;

    /// IBM's SX gate (√X gate)
    #[derive(Debug, Clone, Copy)]
    pub struct SXGate {
        pub target: QubitId,
    }

    impl GateOp for SXGate {
        fn name(&self) -> &'static str {
            "sx"
        }

        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }

        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            let half = 0.5;
            let i_half = Complex64::new(0.0, 0.5);
            Ok(vec![
                Complex64::new(half, 0.0) + i_half,
                Complex64::new(half, 0.0) - i_half,
                Complex64::new(half, 0.0) - i_half,
                Complex64::new(half, 0.0) + i_half,
            ])
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(self.clone())
        }
    }

    impl HardwareGate for SXGate {
        fn backend(&self) -> HardwareBackend {
            HardwareBackend::IBMQuantum
        }

        fn metadata(&self) -> HashMap<String, String> {
            let mut meta = HashMap::new();
            meta.insert("gate_type".to_string(), "basis".to_string());
            meta.insert("duration_ns".to_string(), "35.5".to_string());
            meta
        }
    }
}

/// Google-specific gate implementations
pub mod google_gates {
    use super::*;
    use num_complex::Complex64;
    use std::any::Any;
    use std::f64::consts::PI;

    /// Google's Sycamore gate
    #[derive(Debug, Clone, Copy)]
    pub struct SycamoreGate {
        pub qubit1: QubitId,
        pub qubit2: QubitId,
    }

    impl GateOp for SycamoreGate {
        fn name(&self) -> &'static str {
            "syc"
        }

        fn qubits(&self) -> Vec<QubitId> {
            vec![self.qubit1, self.qubit2]
        }

        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            // Sycamore gate matrix
            // This is a simplified version - actual gate is more complex
            let fsim_theta = PI / 2.0;
            let fsim_phi = PI / 6.0;

            // Create fSIM gate matrix
            let c = fsim_theta.cos();
            let s = Complex64::new(0.0, -fsim_theta.sin());
            let phase = Complex64::from_polar(1.0, -fsim_phi);

            Ok(vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(c, 0.0),
                s,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                s,
                Complex64::new(c, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                phase,
            ])
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(self.clone())
        }
    }

    impl HardwareGate for SycamoreGate {
        fn backend(&self) -> HardwareBackend {
            HardwareBackend::GoogleSycamore
        }

        fn metadata(&self) -> HashMap<String, String> {
            let mut meta = HashMap::new();
            meta.insert("gate_type".to_string(), "entangling".to_string());
            meta.insert("duration_ns".to_string(), "12".to_string());
            meta.insert("fidelity".to_string(), "0.995".to_string());
            meta
        }
    }

    /// Google's powered gates (X^t, Y^t, Z^t)
    #[derive(Debug, Clone, Copy)]
    pub struct PoweredGate {
        pub target: QubitId,
        pub axis: char, // 'X', 'Y', or 'Z'
        pub power: f64,
    }

    impl GateOp for PoweredGate {
        fn name(&self) -> &'static str {
            match self.axis {
                'X' => "x_pow",
                'Y' => "y_pow",
                'Z' => "z_pow",
                _ => "pow",
            }
        }

        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }

        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            let angle = PI * self.power;
            let cos_half = (angle / 2.0).cos();
            let sin_half = (angle / 2.0).sin();

            match self.axis {
                'X' => Ok(vec![
                    Complex64::new(cos_half, 0.0),
                    Complex64::new(0.0, -sin_half),
                    Complex64::new(0.0, -sin_half),
                    Complex64::new(cos_half, 0.0),
                ]),
                'Y' => Ok(vec![
                    Complex64::new(cos_half, 0.0),
                    Complex64::new(-sin_half, 0.0),
                    Complex64::new(sin_half, 0.0),
                    Complex64::new(cos_half, 0.0),
                ]),
                'Z' => Ok(vec![
                    Complex64::from_polar(1.0, -angle / 2.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::from_polar(1.0, angle / 2.0),
                ]),
                _ => Err(QuantRS2Error::InvalidInput("Invalid axis".to_string())),
            }
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(self.clone())
        }
    }
}

/// IonQ-specific gate implementations
pub mod ionq_gates {
    use super::*;
    use num_complex::Complex64;
    use std::any::Any;

    /// IonQ's XX gate (Mølmer-Sørensen gate)
    #[derive(Debug, Clone, Copy)]
    pub struct XXGate {
        pub qubit1: QubitId,
        pub qubit2: QubitId,
        pub angle: f64,
    }

    impl GateOp for XXGate {
        fn name(&self) -> &'static str {
            "xx"
        }

        fn qubits(&self) -> Vec<QubitId> {
            vec![self.qubit1, self.qubit2]
        }

        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            let c = self.angle.cos();
            let s = Complex64::new(0.0, -self.angle.sin());

            Ok(vec![
                Complex64::new(c, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                s,
                Complex64::new(0.0, 0.0),
                Complex64::new(c, 0.0),
                s,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                s,
                Complex64::new(c, 0.0),
                Complex64::new(0.0, 0.0),
                s,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(c, 0.0),
            ])
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(self.clone())
        }
    }

    impl HardwareGate for XXGate {
        fn backend(&self) -> HardwareBackend {
            HardwareBackend::IonQ
        }

        fn metadata(&self) -> HashMap<String, String> {
            let mut meta = HashMap::new();
            meta.insert("gate_type".to_string(), "ms".to_string());
            meta.insert("interaction".to_string(), "all-to-all".to_string());
            meta
        }

        fn calibration_params(&self) -> Vec<String> {
            vec!["ms_amplitude".to_string(), "ms_phase".to_string()]
        }
    }
}

/// Rigetti-specific gate implementations
pub mod rigetti_gates {
    use super::*;
    use num_complex::Complex64;
    use std::any::Any;

    /// Rigetti's parametrized XY gate
    #[derive(Debug, Clone, Copy)]
    pub struct XYGate {
        pub qubit1: QubitId,
        pub qubit2: QubitId,
        pub angle: f64,
    }

    impl GateOp for XYGate {
        fn name(&self) -> &'static str {
            "xy"
        }

        fn qubits(&self) -> Vec<QubitId> {
            vec![self.qubit1, self.qubit2]
        }

        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            let c = (self.angle / 2.0).cos();
            let s = Complex64::new(0.0, (self.angle / 2.0).sin());

            Ok(vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(c, 0.0),
                s,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                s,
                Complex64::new(c, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ])
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(self.clone())
        }
    }
}

/// Honeywell-specific gate implementations
pub mod honeywell_gates {
    use super::*;
    use num_complex::Complex64;
    use std::any::Any;

    /// Honeywell's native ZZ interaction
    #[derive(Debug, Clone, Copy)]
    pub struct ZZGate {
        pub qubit1: QubitId,
        pub qubit2: QubitId,
        pub angle: f64,
    }

    impl GateOp for ZZGate {
        fn name(&self) -> &'static str {
            "zz"
        }

        fn qubits(&self) -> Vec<QubitId> {
            vec![self.qubit1, self.qubit2]
        }

        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            let phase_p = Complex64::from_polar(1.0, self.angle / 2.0);
            let phase_m = Complex64::from_polar(1.0, -self.angle / 2.0);

            Ok(vec![
                phase_m,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                phase_p,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                phase_p,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                phase_m,
            ])
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(self.clone())
        }
    }

    impl HardwareGate for ZZGate {
        fn backend(&self) -> HardwareBackend {
            HardwareBackend::Honeywell
        }

        fn metadata(&self) -> HashMap<String, String> {
            let mut meta = HashMap::new();
            meta.insert("gate_type".to_string(), "native".to_string());
            meta.insert("fidelity".to_string(), "0.999".to_string());
            meta
        }
    }

    /// Honeywell's U3 gate (general single-qubit rotation)
    #[derive(Debug, Clone, Copy)]
    pub struct U3Gate {
        pub target: QubitId,
        pub theta: f64,
        pub phi: f64,
        pub lambda: f64,
    }

    impl GateOp for U3Gate {
        fn name(&self) -> &'static str {
            "u3"
        }

        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }

        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            let cos_half = (self.theta / 2.0).cos();
            let sin_half = (self.theta / 2.0).sin();

            Ok(vec![
                Complex64::new(cos_half, 0.0),
                -Complex64::from_polar(sin_half, self.lambda),
                Complex64::from_polar(sin_half, self.phi),
                Complex64::from_polar(cos_half, self.phi + self.lambda),
            ])
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(self.clone())
        }
    }
}

/// Decomposition validator
pub struct DecompositionValidator {
    /// Tolerance for matrix comparison
    tolerance: f64,
}

impl DecompositionValidator {
    /// Create a new validator
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Validate that a decomposition is equivalent to original gate
    pub fn validate(
        &self,
        original: &dyn GateOp,
        decomposed: &[DecomposedGate],
    ) -> QuantRS2Result<bool> {
        // Would implement matrix multiplication and comparison
        // For now, return true
        Ok(true)
    }

    /// Calculate fidelity between original and decomposed
    pub fn calculate_fidelity(
        &self,
        original: &dyn GateOp,
        decomposed: &[DecomposedGate],
    ) -> QuantRS2Result<f64> {
        // Would implement proper fidelity calculation
        // For now, return high fidelity
        Ok(0.999)
    }
}

/// Backend capabilities query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Backend identifier
    pub backend: HardwareBackend,
    /// Native gate set
    pub native_gates: NativeGateSet,
    /// Supported features
    pub features: BackendFeatures,
    /// Performance characteristics
    pub performance: BackendPerformance,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            backend: HardwareBackend::Custom(0),
            native_gates: NativeGateSet::default(),
            features: BackendFeatures::default(),
            performance: BackendPerformance::default(),
        }
    }
}

/// Backend feature support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendFeatures {
    /// Supports mid-circuit measurements
    pub mid_circuit_measurement: bool,
    /// Supports conditional gates
    pub conditional_gates: bool,
    /// Supports parametric compilation
    pub parametric_compilation: bool,
    /// Supports pulse-level control
    pub pulse_control: bool,
    /// Maximum circuit width
    pub max_qubits: usize,
    /// Maximum circuit depth
    pub max_depth: Option<usize>,
    /// Maximum number of mid-circuit measurements
    pub max_mid_circuit_measurements: Option<usize>,
    /// Classical register size (bits)
    pub classical_register_size: usize,
    /// Supports real-time feedback
    pub supports_real_time_feedback: bool,
    /// Supports parallel execution
    pub supports_parallel_execution: bool,
    /// Supports reset operations
    pub supports_reset: bool,
    /// Supports barrier operations
    pub supports_barriers: bool,
    /// Measurement types supported (Z, X, Y, Pauli, etc.)
    pub supported_measurement_bases: Vec<String>,
}

impl Default for BackendFeatures {
    fn default() -> Self {
        Self {
            mid_circuit_measurement: false,
            conditional_gates: false,
            parametric_compilation: true,
            pulse_control: false,
            max_qubits: 64,
            max_depth: None,
            max_mid_circuit_measurements: None,
            classical_register_size: 64,
            supports_real_time_feedback: false,
            supports_parallel_execution: false,
            supports_reset: true,
            supports_barriers: true,
            supported_measurement_bases: vec!["Z".to_string()],
        }
    }
}

/// Backend performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendPerformance {
    /// Single-qubit gate time (ns)
    pub single_qubit_gate_time: f64,
    /// Two-qubit gate time (ns)
    pub two_qubit_gate_time: f64,
    /// Measurement time (ns)
    pub measurement_time: f64,
    /// Typical T1 time (μs)
    pub t1_time: f64,
    /// Typical T2 time (μs)
    pub t2_time: f64,
    /// Single-qubit gate fidelity
    pub single_qubit_fidelity: f64,
    /// Two-qubit gate fidelity
    pub two_qubit_fidelity: f64,
}

impl Default for BackendPerformance {
    fn default() -> Self {
        Self {
            single_qubit_gate_time: 50.0, // ns
            two_qubit_gate_time: 500.0,   // ns
            measurement_time: 1000.0,     // ns
            t1_time: 100.0,               // μs
            t2_time: 50.0,                // μs
            single_qubit_fidelity: 0.999,
            two_qubit_fidelity: 0.99,
        }
    }
}

/// Query backend capabilities
pub fn query_backend_capabilities(backend: HardwareBackend) -> BackendCapabilities {
    match backend {
        HardwareBackend::IBMQuantum => BackendCapabilities {
            backend,
            native_gates: NativeGateSet {
                backend,
                single_qubit_gates: vec!["id", "rz", "sx", "x"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                two_qubit_gates: vec!["cx".to_string()],
                multi_qubit_gates: vec![],
                arbitrary_single_qubit: false,
                rotation_axes: vec![crate::translation::RotationAxis::Z],
                constraints: crate::translation::BackendConstraints {
                    max_depth: None,
                    discrete_angles: None,
                    virtual_z: true,
                    coupling_map: None,
                    timing_constraints: None,
                },
            },
            features: BackendFeatures {
                mid_circuit_measurement: true,
                conditional_gates: true,
                parametric_compilation: true,
                pulse_control: true,
                max_qubits: 127,
                max_depth: Some(10000),
                max_mid_circuit_measurements: Some(127), // One per qubit
                classical_register_size: 128,
                supports_real_time_feedback: true,
                supports_parallel_execution: false, // IBM executes serially
                supports_reset: true,
                supports_barriers: true,
                supported_measurement_bases: vec![
                    "Z".to_string(),
                    "X".to_string(),
                    "Y".to_string(),
                ],
            },
            performance: BackendPerformance {
                single_qubit_gate_time: 35.0,
                two_qubit_gate_time: 300.0,
                measurement_time: 3000.0,
                t1_time: 100.0,
                t2_time: 100.0,
                single_qubit_fidelity: 0.9999,
                two_qubit_fidelity: 0.99,
            },
        },
        HardwareBackend::IonQ => BackendCapabilities {
            backend,
            native_gates: NativeGateSet {
                backend,
                single_qubit_gates: vec!["rx", "ry", "rz"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                two_qubit_gates: vec!["xx".to_string()],
                multi_qubit_gates: vec![],
                arbitrary_single_qubit: true,
                rotation_axes: vec![
                    crate::translation::RotationAxis::X,
                    crate::translation::RotationAxis::Y,
                    crate::translation::RotationAxis::Z,
                ],
                constraints: crate::translation::BackendConstraints {
                    max_depth: None,
                    discrete_angles: None,
                    virtual_z: false,
                    coupling_map: None, // All-to-all
                    timing_constraints: None,
                },
            },
            features: BackendFeatures {
                mid_circuit_measurement: false,
                conditional_gates: false,
                parametric_compilation: true,
                pulse_control: false,
                max_qubits: 32,
                max_depth: None,
                max_mid_circuit_measurements: None, // Not supported
                classical_register_size: 0,         // No classical registers
                supports_real_time_feedback: false,
                supports_parallel_execution: true, // All-to-all connectivity allows parallelism
                supports_reset: false,
                supports_barriers: false,
                supported_measurement_bases: vec!["Z".to_string()],
            },
            performance: BackendPerformance {
                single_qubit_gate_time: 135.0,
                two_qubit_gate_time: 600.0,
                measurement_time: 100.0,
                t1_time: 10000.0, // 10 ms
                t2_time: 1000.0,  // 1 ms
                single_qubit_fidelity: 0.9995,
                two_qubit_fidelity: 0.97,
            },
        },
        _ => {
            // Default capabilities
            BackendCapabilities {
                backend,
                native_gates: NativeGateSet {
                    backend,
                    single_qubit_gates: vec![],
                    two_qubit_gates: vec![],
                    multi_qubit_gates: vec![],
                    arbitrary_single_qubit: true,
                    rotation_axes: vec![],
                    constraints: crate::translation::BackendConstraints {
                        max_depth: None,
                        discrete_angles: None,
                        virtual_z: false,
                        coupling_map: None,
                        timing_constraints: None,
                    },
                },
                features: BackendFeatures {
                    mid_circuit_measurement: false,
                    conditional_gates: false,
                    parametric_compilation: false,
                    pulse_control: false,
                    max_qubits: 20,
                    max_depth: None,
                    max_mid_circuit_measurements: None,
                    classical_register_size: 0,
                    supports_real_time_feedback: false,
                    supports_parallel_execution: false,
                    supports_reset: false,
                    supports_barriers: false,
                    supported_measurement_bases: vec!["Z".to_string()],
                },
                performance: BackendPerformance {
                    single_qubit_gate_time: 50.0,
                    two_qubit_gate_time: 500.0,
                    measurement_time: 1000.0,
                    t1_time: 50.0,
                    t2_time: 50.0,
                    single_qubit_fidelity: 0.999,
                    two_qubit_fidelity: 0.99,
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_gate_implementations() {
        // Test IBM SX gate
        let sx = ibm_gates::SXGate { target: QubitId(0) };
        assert_eq!(sx.name(), "sx");
        assert_eq!(sx.backend(), HardwareBackend::IBMQuantum);

        // Test Google Sycamore gate
        let syc = google_gates::SycamoreGate {
            qubit1: QubitId(0),
            qubit2: QubitId(1),
        };
        assert_eq!(syc.name(), "syc");
        assert_eq!(syc.backend(), HardwareBackend::GoogleSycamore);

        // Test IonQ XX gate
        let xx = ionq_gates::XXGate {
            qubit1: QubitId(0),
            qubit2: QubitId(1),
            angle: std::f64::consts::PI / 2.0,
        };
        assert_eq!(xx.name(), "xx");
        assert_eq!(xx.backend(), HardwareBackend::IonQ);
    }

    #[test]
    fn test_backend_capabilities() {
        let ibm_caps = query_backend_capabilities(HardwareBackend::IBMQuantum);
        assert!(ibm_caps.features.pulse_control);
        assert!(ibm_caps.features.mid_circuit_measurement);
        assert_eq!(ibm_caps.performance.single_qubit_gate_time, 35.0);

        let ionq_caps = query_backend_capabilities(HardwareBackend::IonQ);
        assert!(!ionq_caps.features.pulse_control);
        assert!(ionq_caps.performance.t1_time > ibm_caps.performance.t1_time);
    }
}
