//! Tests for OpenQASM 3.0 import/export functionality

use quantrs2_circuit::builder::{Circuit, CircuitBuilder};
use quantrs2_circuit::prelude::*;
use quantrs2_core::qubit::QubitId;

#[test]
fn test_qasm3_export_simple() {
    // Create a simple Bell state circuit
    let mut builder = CircuitBuilder::<2>::new();
    builder.h(Qubit::new(0));
    builder.cx(Qubit::new(0), Qubit::new(1));

    let circuit = builder.build();

    // Export to QASM 3.0
    let qasm = export_qasm3(&circuit).expect("Failed to export circuit");

    // Check that the QASM contains expected elements
    assert!(qasm.contains("OPENQASM 3.0"));
    assert!(qasm.contains("qubit[2] q"));
    assert!(qasm.contains("h q[0]"));
    assert!(qasm.contains("cx q[0], q[1]"));

    println!("Generated QASM:\n{}", qasm);
}

#[test]
fn test_qasm3_export_with_measurements() {
    let mut builder = CircuitBuilder::<3>::new();

    // Create GHZ state
    builder.h(Qubit::new(0));
    builder.cx(Qubit::new(0), Qubit::new(1));
    builder.cx(Qubit::new(1), Qubit::new(2));

    // Add measurements
    builder.measure(Qubit::new(0));
    builder.measure(Qubit::new(1));
    builder.measure(Qubit::new(2));

    let circuit = builder.build();

    // Export with custom options
    let mut exporter = QasmExporter::new(ExportOptions {
        include_stdgates: true,
        decompose_custom: false,
        include_gate_comments: false,
        optimize: false,
        pretty_print: true,
    });

    let qasm = exporter.export(&circuit).expect("Failed to export circuit");

    // Check for measurements
    assert!(qasm.contains("bit[3] c"));
    assert!(qasm.contains("measure"));

    println!("GHZ circuit QASM:\n{}", qasm);
}

#[test]
fn test_qasm3_parse_simple() {
    let qasm_code = r#"
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] q;
bit[3] c;

// Create superposition
h q[0];

// Bell pair on first two qubits
cx q[0], q[1];

// Entangle third qubit
cx q[1], q[2];

// Measure all
measure q -> c;
"#;

    // Parse the QASM code
    let program = parse_qasm3(qasm_code).expect("Failed to parse QASM");

    // Validate the program
    validate_qasm3(&program).expect("Validation failed");

    // Check program structure
    assert_eq!(program.version, "3.0");
    assert_eq!(program.includes.len(), 1);
    assert_eq!(program.declarations.len(), 2); // qubit and bit registers
    assert_eq!(program.statements.len(), 4); // h, cx, cx, measure
}

#[test]
fn test_qasm3_parse_parametric_gates() {
    let qasm_code = r#"
OPENQASM 3.0;

qubit[2] q;

// Parametric rotations
rx(pi/2) q[0];
ry(pi/4) q[1];
rz(pi/8) q[0];

// Controlled rotation
crx(pi/3) q[0], q[1];
"#;

    let program = parse_qasm3(qasm_code).expect("Failed to parse QASM");
    validate_qasm3(&program).expect("Validation failed");

    // Check that we have the right number of statements
    assert_eq!(program.statements.len(), 4);
}

#[test]
fn test_qasm3_parse_custom_gate() {
    let qasm_code = r#"
OPENQASM 3.0;

// Define custom bell gate
gate bell a, b {
    h a;
    cx a, b;
}

qubit[2] q;

// Use custom gate
bell q[0], q[1];
"#;

    let program = parse_qasm3(qasm_code).expect("Failed to parse QASM");
    validate_qasm3(&program).expect("Validation failed");

    // Check for custom gate definition
    assert_eq!(program.declarations.len(), 2); // gate def and qubit register

    // Find the gate definition
    let mut found_gate = false;
    for decl in &program.declarations {
        if let quantrs2_circuit::qasm::ast::Declaration::GateDefinition(def) = decl {
            assert_eq!(def.name, "bell");
            assert_eq!(def.qubits.len(), 2);
            assert_eq!(def.body.len(), 2); // h and cx
            found_gate = true;
        }
    }
    assert!(found_gate, "Custom gate definition not found");
}

#[test]
fn test_qasm3_parse_control_flow() {
    let qasm_code = r#"
OPENQASM 3.0;

qubit[4] q;
bit[4] c;

// Basic gates
h q[0];
cx q[0], q[1];

// For loop with fixed indices
for i in [0:2] {
    h q[0];
    h q[1];
}
"#;

    let program = parse_qasm3(qasm_code).expect("Failed to parse QASM");
    validate_qasm3(&program).expect("Validation failed");

    // Check for control flow statements
    assert_eq!(program.statements.len(), 3); // measure, if, for
}

#[test]
fn test_qasm3_validation_errors() {
    // Test undefined register
    let qasm_code = r#"
OPENQASM 3.0;
qubit[2] q;
h r[0];  // r is undefined
"#;

    let program = parse_qasm3(qasm_code).expect("Failed to parse");
    let result = validate_qasm3(&program);
    assert!(result.is_err());

    // Test index out of bounds
    let qasm_code = r#"
OPENQASM 3.0;
qubit[2] q;
h q[5];  // index 5 is out of bounds
"#;

    let program = parse_qasm3(qasm_code).expect("Failed to parse");
    let result = validate_qasm3(&program);
    assert!(result.is_err());

    // Test parameter count mismatch
    let qasm_code = r#"
OPENQASM 3.0;
qubit q;
rx q;  // rx requires 1 parameter
"#;

    let program = parse_qasm3(qasm_code).expect("Failed to parse");
    let result = validate_qasm3(&program);
    assert!(result.is_err());
}

#[test]
fn test_qasm3_round_trip() {
    // Create a circuit
    let mut builder = CircuitBuilder::<3>::new();
    builder.h(Qubit::new(0));
    builder.rx(Qubit::new(1), std::f64::consts::PI / 4.0);
    builder.cx(Qubit::new(0), Qubit::new(2));
    builder.measure(Qubit::new(0));
    builder.measure(Qubit::new(1));
    builder.measure(Qubit::new(2));

    let original_circuit = builder.build();

    // Export to QASM
    let qasm = export_qasm3(&original_circuit).expect("Failed to export");
    println!("Exported QASM:\n{}", qasm);

    // Parse back
    let program = parse_qasm3(&qasm).expect("Failed to parse exported QASM");

    // Validate
    validate_qasm3(&program).expect("Validation failed");

    // Check that we have the expected operations
    assert_eq!(program.statements.len(), 6); // h, rx, cx, 3 measures
}

#[test]
fn test_qasm3_advanced_features() {
    let qasm_code = r#"
OPENQASM 3.0;
include "stdgates.inc";

// Constants
const n = 4;
const angle = pi/8;

// Registers
qubit[n] q;
bit[n] c;

// Basic operations
h q[0];
rx(angle) q[1];
cx q[0], q[1];

// Reset
reset q[0];

// Barrier
barrier q;
"#;

    let program = parse_qasm3(qasm_code).expect("Failed to parse QASM");
    validate_qasm3(&program).expect("Validation failed");

    // Check for constants and registers
    assert_eq!(program.declarations.len(), 4); // 2 constants + 2 registers

    // Check that we have basic statements (h, rx, cx, reset, barrier)
    assert!(
        program.statements.len() >= 5,
        "Expected at least 5 statements"
    );
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_qasm3_compatibility() {
        // Test that our QASM output is valid QASM 3.0
        let mut builder = CircuitBuilder::<4>::new();

        // Build a quantum Fourier transform circuit
        let n = 4;
        for j in 0..n {
            builder.h(Qubit::new(j));
            for k in (j + 1)..n {
                let angle = std::f64::consts::PI / (1 << (k - j)) as f64;
                builder.cp(Qubit::new(j), Qubit::new(k), angle);
            }
        }

        // Add SWAP gates to reverse qubit order
        for i in 0..n / 2 {
            builder.swap(Qubit::new(i), Qubit::new(n - 1 - i));
        }

        let circuit = builder.build();
        let qasm = export_qasm3(&circuit).expect("Failed to export QFT circuit");

        println!("QFT Circuit QASM:\n{}", qasm);

        // Verify the QASM can be parsed and validated
        let program = parse_qasm3(&qasm).expect("Failed to parse QFT QASM");
        validate_qasm3(&program).expect("QFT validation failed");
    }
}
