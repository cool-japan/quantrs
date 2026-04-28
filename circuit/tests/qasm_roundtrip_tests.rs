//! Integration tests for OpenQASM 2.0 round-trip functionality.

use quantrs2_circuit::builder::Circuit;
use quantrs2_circuit::qasm::{circuit_to_qasm, qasm_to_gates};
use quantrs2_core::gate::multi::{CRX, CRY, CRZ, CNOT, CZ, SWAP, Toffoli};
use quantrs2_core::gate::single::{
    Hadamard, Phase, PauliX, PauliY, PauliZ, PhaseDagger, RotationX, RotationY, RotationZ,
    SqrtX, T, TDagger, UGate,
};
use quantrs2_core::qubit::QubitId;

// ─── helper: make QubitId from usize ────────────────────────────────────────

const fn q(n: u32) -> QubitId {
    QubitId(n)
}

// ─── export tests ────────────────────────────────────────────────────────────

#[test]
fn test_qasm2_export_bell_state() {
    let mut circuit = Circuit::<2>::new();
    circuit.add_gate(Hadamard { target: q(0) }).expect("H");
    circuit
        .add_gate(CNOT {
            control: q(0),
            target: q(1),
        })
        .expect("CNOT");

    let qasm = circuit_to_qasm(&circuit).expect("export bell state");

    assert!(qasm.starts_with("OPENQASM 2.0;"), "wrong header");
    assert!(qasm.contains(r#"include "qelib1.inc""#), "missing include");
    assert!(qasm.contains("qreg q[2]"), "missing qreg");
    assert!(qasm.contains("h q[0];"), "missing H gate");
    assert!(qasm.contains("cx q[0], q[1];"), "missing cx gate");
}

#[test]
fn test_qasm2_export_rotation_gates() {
    let pi = std::f64::consts::PI;
    let mut circuit = Circuit::<2>::new();
    circuit
        .add_gate(RotationX {
            target: q(0),
            theta: pi / 2.0,
        })
        .expect("RX");
    circuit
        .add_gate(RotationY {
            target: q(1),
            theta: pi / 4.0,
        })
        .expect("RY");
    circuit
        .add_gate(RotationZ {
            target: q(0),
            theta: pi / 8.0,
        })
        .expect("RZ");

    let qasm = circuit_to_qasm(&circuit).expect("export rotations");
    assert!(qasm.contains("rx("), "missing rx");
    assert!(qasm.contains("ry("), "missing ry");
    assert!(qasm.contains("rz("), "missing rz");
}

#[test]
fn test_qasm2_export_u_gate() {
    let mut circuit = Circuit::<2>::new();
    circuit
        .add_gate(UGate {
            target: q(0),
            theta: 1.0,
            phi: 2.0,
            lambda: 3.0,
        })
        .expect("U");

    let qasm = circuit_to_qasm(&circuit).expect("export U gate");
    assert!(qasm.contains("u3("), "missing u3 gate");
}

#[test]
fn test_qasm2_export_s_t_gates() {
    let mut circuit = Circuit::<2>::new();
    circuit.add_gate(Phase { target: q(0) }).expect("S");
    circuit.add_gate(PhaseDagger { target: q(1) }).expect("Sdg");
    circuit.add_gate(T { target: q(0) }).expect("T");
    circuit.add_gate(TDagger { target: q(1) }).expect("Tdg");
    circuit.add_gate(SqrtX { target: q(0) }).expect("SX");

    let qasm = circuit_to_qasm(&circuit).expect("export s/t gates");
    assert!(qasm.contains("s q[0];"), "missing s");
    assert!(qasm.contains("sdg q[1];"), "missing sdg");
    assert!(qasm.contains("t q[0];"), "missing t");
    assert!(qasm.contains("tdg q[1];"), "missing tdg");
    assert!(qasm.contains("sx q[0];"), "missing sx");
}

#[test]
fn test_qasm2_export_toffoli() {
    let mut circuit = Circuit::<3>::new();
    circuit
        .add_gate(Toffoli {
            control1: q(0),
            control2: q(1),
            target: q(2),
        })
        .expect("Toffoli");

    let qasm = circuit_to_qasm(&circuit).expect("export toffoli");
    assert!(qasm.contains("qreg q[3]"), "missing 3-qubit qreg");
    assert!(qasm.contains("ccx q[0], q[1], q[2];"), "missing ccx");
}

// ─── import tests ────────────────────────────────────────────────────────────

#[test]
fn test_qasm2_import_bell_state() {
    let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"#;

    let (gates, num_qubits) = qasm_to_gates(qasm).expect("parse bell state");
    assert_eq!(num_qubits, 2);
    assert_eq!(gates.len(), 2);
    assert_eq!(gates[0].name(), "H");
    assert_eq!(gates[1].name(), "CNOT");
    assert_eq!(gates[0].qubits()[0].0, 0);
    assert_eq!(gates[1].qubits()[0].0, 0); // control
    assert_eq!(gates[1].qubits()[1].0, 1); // target
}

#[test]
fn test_qasm2_import_rotation_pi_expression() {
    let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rx(pi/2) q[0];
ry(pi/4) q[1];
rz(2*pi) q[0];
"#;
    let (gates, _) = qasm_to_gates(qasm).expect("parse rotations");
    assert_eq!(gates.len(), 3);
    assert_eq!(gates[0].name(), "RX");
    assert_eq!(gates[1].name(), "RY");
    assert_eq!(gates[2].name(), "RZ");
}

#[test]
fn test_qasm2_import_u3_gate() {
    let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
u3(1.0,2.0,3.0) q[0];
"#;
    let (gates, _) = qasm_to_gates(qasm).expect("parse u3");
    assert_eq!(gates.len(), 1);
    assert_eq!(gates[0].name(), "U");
}

#[test]
fn test_qasm2_import_toffoli() {
    let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
ccx q[0], q[1], q[2];
"#;
    let (gates, n) = qasm_to_gates(qasm).expect("parse toffoli");
    assert_eq!(n, 3);
    assert_eq!(gates.len(), 1);
    assert_eq!(gates[0].name(), "Toffoli");
}

#[test]
fn test_qasm2_import_skips_measure_and_barrier() {
    let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
barrier q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#;
    // Measures and barriers are currently skipped in gate list
    let (gates, _) = qasm_to_gates(qasm).expect("parse with measure");
    assert_eq!(gates.len(), 2, "should have only H and CNOT, not measure/barrier");
}

// ─── round-trip tests ────────────────────────────────────────────────────────

#[test]
fn test_qasm2_roundtrip_hadamard_cnot() {
    // Build → export → import → compare gate names
    let mut circuit = Circuit::<2>::new();
    circuit.add_gate(Hadamard { target: q(0) }).expect("H");
    circuit
        .add_gate(CNOT {
            control: q(0),
            target: q(1),
        })
        .expect("CNOT");

    let qasm = circuit_to_qasm(&circuit).expect("export");
    let (gates, num_qubits) = qasm_to_gates(&qasm).expect("import");

    assert_eq!(num_qubits, 2);
    assert_eq!(gates.len(), 2);
    assert_eq!(gates[0].name(), "H");
    assert_eq!(gates[1].name(), "CNOT");
}

#[test]
fn test_qasm2_roundtrip_rotation_preserves_angle() {
    let pi = std::f64::consts::PI;
    let angle = pi / 3.0;

    let mut circuit = Circuit::<2>::new();
    circuit
        .add_gate(RotationX {
            target: q(0),
            theta: angle,
        })
        .expect("RX");

    let qasm = circuit_to_qasm(&circuit).expect("export RX");
    let (gates, _) = qasm_to_gates(&qasm).expect("import RX");

    assert_eq!(gates.len(), 1);
    assert_eq!(gates[0].name(), "RX");

    // Extract the angle via downcast
    use quantrs2_core::gate::single::RotationX as RX;
    if let Some(rx) = gates[0].as_any().downcast_ref::<RX>() {
        assert!(
            (rx.theta - angle).abs() < 1e-7,
            "angle mismatch: {} vs {}",
            rx.theta,
            angle
        );
    } else {
        panic!("Expected RotationX gate");
    }
}

#[test]
fn test_qasm2_roundtrip_multi_qubit_circuit() {
    let mut circuit = Circuit::<4>::new();
    circuit.add_gate(Hadamard { target: q(0) }).expect("H");
    circuit.add_gate(PauliX { target: q(1) }).expect("X");
    circuit.add_gate(PauliY { target: q(2) }).expect("Y");
    circuit.add_gate(PauliZ { target: q(3) }).expect("Z");
    circuit
        .add_gate(CNOT {
            control: q(0),
            target: q(1),
        })
        .expect("CNOT");
    circuit
        .add_gate(CZ {
            control: q(2),
            target: q(3),
        })
        .expect("CZ");
    circuit
        .add_gate(SWAP {
            qubit1: q(0),
            qubit2: q(3),
        })
        .expect("SWAP");

    let qasm = circuit_to_qasm(&circuit).expect("export 4-qubit circuit");
    let (gates, num_qubits) = qasm_to_gates(&qasm).expect("import 4-qubit circuit");

    assert_eq!(num_qubits, 4);
    assert_eq!(gates.len(), 7);

    let names: Vec<&str> = gates.iter().map(|g| g.name()).collect();
    assert_eq!(names, ["H", "X", "Y", "Z", "CNOT", "CZ", "SWAP"]);
}

#[test]
fn test_qasm2_roundtrip_controlled_rotations() {
    let pi = std::f64::consts::PI;
    let mut circuit = Circuit::<2>::new();
    circuit
        .add_gate(CRX {
            control: q(0),
            target: q(1),
            theta: pi / 2.0,
        })
        .expect("CRX");
    circuit
        .add_gate(CRY {
            control: q(0),
            target: q(1),
            theta: pi / 4.0,
        })
        .expect("CRY");
    circuit
        .add_gate(CRZ {
            control: q(0),
            target: q(1),
            theta: pi / 8.0,
        })
        .expect("CRZ");

    let qasm = circuit_to_qasm(&circuit).expect("export CRX/CRY/CRZ");
    let (gates, _) = qasm_to_gates(&qasm).expect("import CRX/CRY/CRZ");

    assert_eq!(gates.len(), 3);
    assert_eq!(gates[0].name(), "CRX");
    assert_eq!(gates[1].name(), "CRY");
    assert_eq!(gates[2].name(), "CRZ");
}
