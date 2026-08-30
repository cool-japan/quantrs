#![allow(clippy::pedantic)]
//! Regression test for GitHub issue #32.
//!
//! `RotationZ` (and `CRZ`, `ParametricRotationZ`) previously used the reversed
//! convention `diag(e^{+iθ/2}, e^{-iθ/2})`. quantrs2 now follows the
//! IBM/Qiskit/OpenQASM-3 standard `Rz(θ) = diag(e^{-iθ/2}, e^{+iθ/2})`.
//!
//! This runs the reporter's exact circuit on the real `StateVectorSimulator`
//! and checks the output statevector against the standard-convention reference.

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_sim::quantum_info::{state_fidelity, QuantumState};
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use std::f64::consts::PI;

/// The reporter's circuit:
///   `h(0); h(1); cx(0,1); rz(1, -3π/2); cx(0,1); rx(0, -π/4); rx(1, -π/4)`
///
/// Under the IBM/Qiskit/OpenQASM-3 convention the exact final statevector is
/// `[0, -0.5-0.5i, -0.5-0.5i, 0]` — all amplitude on |01⟩ and |10⟩. Under the
/// old reversed Rz convention the simulator instead produced
/// `[-0.5-0.5i, 0, 0, -0.5-0.5i]` (amplitude on |00⟩ and |11⟩), i.e. fidelity 0
/// against the correct result.
#[test]
fn test_issue_32_reporter_circuit_statevector() {
    let mut circuit = Circuit::<2>::new();
    circuit.h(0).expect("h(0)");
    circuit.h(1).expect("h(1)");
    circuit.cnot(0, 1).expect("cx(0,1)");
    circuit.rz(1, -3.0 * PI / 2.0).expect("rz(1, -3π/2)");
    circuit.cnot(0, 1).expect("cx(0,1)");
    circuit.rx(0, -PI / 4.0).expect("rx(0, -π/4)");
    circuit.rx(1, -PI / 4.0).expect("rx(1, -π/4)");

    let sim = StateVectorSimulator::new();
    let register = sim.run(&circuit).expect("state-vector simulation");
    let amps = register.amplitudes();

    // Standard-convention reference statevector.
    let reference = vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(-0.5, -0.5),
        Complex64::new(-0.5, -0.5),
        Complex64::new(0.0, 0.0),
    ];

    // Direct amplitude check (crisp regression guard).
    for (i, (got, want)) in amps.iter().zip(reference.iter()).enumerate() {
        assert!(
            (got - want).norm() < 1e-9,
            "amplitude {i} mismatch: got {got:?}, want {want:?}"
        );
    }

    // Global-phase-insensitive fidelity check against the standard reference.
    let sim_state = QuantumState::Pure(Array1::from_vec(amps.to_vec()));
    let ref_state = QuantumState::Pure(Array1::from_vec(reference));
    let fidelity = state_fidelity(&sim_state, &ref_state).expect("state fidelity");
    assert!(
        (fidelity - 1.0).abs() < 1e-10,
        "fidelity against standard reference must be ~1.0, got {fidelity}"
    );
}
