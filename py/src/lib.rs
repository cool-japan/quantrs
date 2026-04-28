//! Python bindings for the `QuantRS2` framework.
//!
//! This crate provides Python bindings using `PyO3`,
//! allowing `QuantRS2` to be used from Python.
//!
//! ## Recent Updates (v0.1.3)
//!
//! - Refined `SciRS2` v0.1.3 integration with unified patterns
//! - Enhanced cross-platform support (macOS, Linux, Windows)
//! - Improved GPU acceleration with CUDA support
//! - Advanced quantum ML capabilities with autograd support
//! - Comprehensive policy documentation for Python quantum computing

use pyo3::prelude::*;

// Include the QEC module
mod qec;

// Include the 3D state visualization module
mod state_viz_3d;

// Include the visualization module
mod visualization;
use visualization::PyCircuitVisualizer;

// Include the gates module
mod gates;

// Include the SciRS2 bindings module
mod scirs2_bindings;

// Include the parametric circuits module
mod parametric;

// Include the optimization passes module
mod optimization_passes;

// Include the Pythonic API module
mod pythonic_api;

// Include the custom gates module
mod custom_gates;

// Include the measurement and tomography module
mod measurement;

// Include the quantum algorithms module
mod algorithms;

// Include the pulse control module (requires device feature)
#[cfg(feature = "device")]
mod pulse;

// Include the error mitigation module (requires device feature)
#[cfg(feature = "device")]
mod mitigation;

// Include the ML transfer learning module
#[cfg(feature = "ml")]
mod ml_transfer;

// Include the anneal module
#[cfg(feature = "anneal")]
mod anneal;

// Include the tytan module
#[cfg(feature = "tytan")]
mod tytan;

// Include the multi-GPU module
mod multi_gpu;

// Include the simulators module
mod simulators;

// Include the circuit helpers module (get_visualizer, apply_gate for PyCircuit)
mod circuit_helpers;

// Noise model: PyRealisticNoiseModel
pub(crate) mod noise_model;
pub(crate) use noise_model::PyRealisticNoiseModel;

// Circuit core types: PyCircuit, CircuitOp, PyDynamicCircuit
pub(crate) mod circuit_core;
pub(crate) use circuit_core::{CircuitOp, PyCircuit, PyDynamicCircuit};

// Simulation result type
pub(crate) mod simulation_result;
pub(crate) use simulation_result::PySimulationResult;

/// Python module for `QuantRS2`
#[pymodule]
fn quantrs2(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add classes to the module
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyDynamicCircuit>()?;
    m.add_class::<PySimulationResult>()?;
    m.add_class::<PyRealisticNoiseModel>()?;
    m.add_class::<PyCircuitVisualizer>()?;

    // Register the gates submodule
    gates::register_module(m)?;

    // Register the SciRS2 submodule
    scirs2_bindings::create_scirs2_module(m)?;
    m.add_class::<scirs2_bindings::PyQuantumNumerics>()?;

    // Register the parametric module
    parametric::register_parametric_module(m)?;

    // Register the optimization module
    optimization_passes::register_optimization_module(m)?;

    // Register the Pythonic API module
    pythonic_api::register_pythonic_module(m)?;

    // Register the custom gates module
    custom_gates::register_custom_gates_module(m)?;

    // Register the measurement module
    measurement::register_measurement_module(m)?;

    // Register the algorithms module
    algorithms::register_algorithms_module(m)?;

    // Register the pulse module (requires device feature)
    #[cfg(feature = "device")]
    pulse::register_pulse_module(m)?;

    // Register the mitigation module (requires device feature)
    #[cfg(feature = "device")]
    mitigation::register_mitigation_module(m)?;

    // Register the ML transfer learning module
    #[cfg(feature = "ml")]
    ml_transfer::register_ml_transfer_module(m)?;

    // Register the anneal module
    #[cfg(feature = "anneal")]
    anneal::register_anneal_module(m)?;

    // Register the tytan module
    #[cfg(feature = "tytan")]
    tytan::register_tytan_module(m)?;

    // Register the multi-GPU module
    multi_gpu::register_multi_gpu_module(m)?;

    // Register the simulators module
    simulators::register_simulators_module(m)?;

    // Register the QEC submodule
    qec::register_qec_module(m)?;

    // Register the 3D state visualization submodule
    state_viz_3d::register_state_viz_3d_module(m)?;

    // Add metadata
    m.setattr(
        "__doc__",
        "QuantRS2 Quantum Computing Framework Python Bindings",
    )?;

    // Add constants
    m.add("MAX_QUBITS", 32)?;
    m.add(
        "SUPPORTED_QUBITS",
        vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 32],
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::qubit::QubitId;

    #[test]
    fn test_circuit_op_inverse_hermitian() {
        // Test self-inverse (Hermitian) gates
        let h = CircuitOp::Hadamard(QubitId::new(0));
        assert!(matches!(h.inverse(), CircuitOp::Hadamard(_)));

        let x = CircuitOp::PauliX(QubitId::new(0));
        assert!(matches!(x.inverse(), CircuitOp::PauliX(_)));

        let y = CircuitOp::PauliY(QubitId::new(0));
        assert!(matches!(y.inverse(), CircuitOp::PauliY(_)));

        let z = CircuitOp::PauliZ(QubitId::new(0));
        assert!(matches!(z.inverse(), CircuitOp::PauliZ(_)));

        let cnot = CircuitOp::Cnot(QubitId::new(0), QubitId::new(1));
        assert!(matches!(cnot.inverse(), CircuitOp::Cnot(_, _)));

        let swap = CircuitOp::Swap(QubitId::new(0), QubitId::new(1));
        assert!(matches!(swap.inverse(), CircuitOp::Swap(_, _)));
    }

    #[test]
    fn test_circuit_op_inverse_paired() {
        // Test S/Sdg pair
        let s = CircuitOp::S(QubitId::new(0));
        assert!(matches!(s.inverse(), CircuitOp::SDagger(_)));

        let sdg = CircuitOp::SDagger(QubitId::new(0));
        assert!(matches!(sdg.inverse(), CircuitOp::S(_)));

        // Test T/Tdg pair
        let t = CircuitOp::T(QubitId::new(0));
        assert!(matches!(t.inverse(), CircuitOp::TDagger(_)));

        let tdg = CircuitOp::TDagger(QubitId::new(0));
        assert!(matches!(tdg.inverse(), CircuitOp::T(_)));

        // Test SX/SXdg pair
        let sx = CircuitOp::SX(QubitId::new(0));
        assert!(matches!(sx.inverse(), CircuitOp::SXDagger(_)));

        let sxdg = CircuitOp::SXDagger(QubitId::new(0));
        assert!(matches!(sxdg.inverse(), CircuitOp::SX(_)));
    }

    #[test]
    fn test_circuit_op_inverse_rotation() {
        // Test rotation gates with negated angles
        let pi = std::f64::consts::PI;

        let rx = CircuitOp::Rx(QubitId::new(0), pi);
        if let CircuitOp::Rx(_, angle) = rx.inverse() {
            assert!((angle + pi).abs() < 1e-10);
        } else {
            panic!("Expected Rx inverse");
        }

        let ry = CircuitOp::Ry(QubitId::new(0), pi / 2.0);
        if let CircuitOp::Ry(_, angle) = ry.inverse() {
            assert!((angle + pi / 2.0).abs() < 1e-10);
        } else {
            panic!("Expected Ry inverse");
        }

        let rz = CircuitOp::Rz(QubitId::new(0), pi / 4.0);
        if let CircuitOp::Rz(_, angle) = rz.inverse() {
            assert!((angle + pi / 4.0).abs() < 1e-10);
        } else {
            panic!("Expected Rz inverse");
        }

        // Test controlled rotations
        let crx = CircuitOp::CRX(QubitId::new(0), QubitId::new(1), pi);
        if let CircuitOp::CRX(_, _, angle) = crx.inverse() {
            assert!((angle + pi).abs() < 1e-10);
        } else {
            panic!("Expected CRX inverse");
        }
    }

    #[test]
    fn test_circuit_op_inverse_u_gate() {
        // U(θ, φ, λ)† = U(-θ, -λ, -φ)
        let u = CircuitOp::U(QubitId::new(0), 1.0, 2.0, 3.0);
        if let CircuitOp::U(_, theta, phi, lambda) = u.inverse() {
            assert!((theta + 1.0).abs() < 1e-10);
            assert!((phi + 3.0).abs() < 1e-10); // φ and λ are swapped
            assert!((lambda + 2.0).abs() < 1e-10);
        } else {
            panic!("Expected U inverse");
        }
    }

    #[test]
    fn test_circuit_op_affected_qubits_single() {
        let h = CircuitOp::Hadamard(QubitId::new(0));
        let (q1, q2, q3) = h.affected_qubits();
        assert!(q1.is_some());
        assert!(q2.is_none());
        assert!(q3.is_none());
    }

    #[test]
    fn test_circuit_op_affected_qubits_two() {
        let cnot = CircuitOp::Cnot(QubitId::new(0), QubitId::new(1));
        let (q1, q2, q3) = cnot.affected_qubits();
        assert!(q1.is_some());
        assert!(q2.is_some());
        assert!(q3.is_none());
    }

    #[test]
    fn test_circuit_op_affected_qubits_three() {
        let toffoli = CircuitOp::Toffoli(QubitId::new(0), QubitId::new(1), QubitId::new(2));
        let (q1, q2, q3) = toffoli.affected_qubits();
        assert!(q1.is_some());
        assert!(q2.is_some());
        assert!(q3.is_some());
    }

    #[test]
    fn test_circuit_op_clone_copy() {
        // Test that CircuitOp is Copy (can be used multiple times)
        let h = CircuitOp::Hadamard(QubitId::new(0));
        let h2 = h; // Copy
        let h3 = h; // Another copy

        assert!(matches!(h, CircuitOp::Hadamard(_)));
        assert!(matches!(h2, CircuitOp::Hadamard(_)));
        assert!(matches!(h3, CircuitOp::Hadamard(_)));
    }
}
