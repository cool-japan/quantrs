//! Example demonstrating OpenQASM 3.0 import/export functionality

use quantrs2_circuit::prelude::*;
use quantrs2_circuit::builder::CircuitBuilder;
use quantrs2_core::qubit::QubitId;

fn main() {
    println!("=== OpenQASM 3.0 Import/Export Example ===\n");
    
    // Example 1: Export a circuit to QASM
    export_example();
    
    // Example 2: Parse QASM code
    parse_example();
    
    // Example 3: Validate QASM programs
    validation_example();
    
    // Example 4: Round-trip conversion
    round_trip_example();
}

fn export_example() {
    println!("1. Exporting a circuit to OpenQASM 3.0");
    println!("--------------------------------------");
    
    // Create a quantum teleportation circuit
    let mut builder = CircuitBuilder::<3>::new();
    
    // Create Bell pair between Alice and Bob
    builder.h(Qubit::new(1));
    builder.cx(Qubit::new(1), Qubit::new(2));
    
    // Alice's operations
    builder.cx(Qubit::new(0), Qubit::new(1));
    builder.h(Qubit::new(0));
    
    // Measurements (in real teleportation, these would be mid-circuit)
    builder.measure(Qubit::new(0));
    builder.measure(Qubit::new(1));
    
    // Bob's corrections (would be conditional in real circuit)
    builder.cx(Qubit::new(1), Qubit::new(2));
    builder.cz(Qubit::new(0), Qubit::new(2));
    
    let circuit = builder.build();
    
    // Export with default options
    match export_qasm3(&circuit) {
        Ok(qasm) => {
            println!("Teleportation circuit in OpenQASM 3.0:");
            println!("{}", qasm);
        }
        Err(e) => println!("Export error: {}", e),
    }
    
    // Export with custom options
    let options = ExportOptions {
        include_stdgates: true,
        decompose_custom: true,
        include_gate_comments: true,
        optimize: true,
        pretty_print: true,
    };
    
    let mut exporter = QasmExporter::new(options);
    match exporter.export(&circuit) {
        Ok(qasm) => {
            println!("\nWith custom options:");
            println!("{}", qasm);
        }
        Err(e) => println!("Export error: {}", e),
    }
}

fn parse_example() {
    println!("\n2. Parsing OpenQASM 3.0 code");
    println!("----------------------------");
    
    let qasm_code = r#"
OPENQASM 3.0;
include "stdgates.inc";

// Quantum registers
qubit[5] q;
bit[5] c;

// Create W state
reset q;
ry(1.91063) q[0];  // arccos(1/sqrt(5))
cx q[0], q[1];

// Controlled rotations to distribute amplitude
cry(1.10715) q[1], q[2];  // arccos(1/2)
cx q[1], q[2];

cry(0.95532) q[2], q[3];  // arccos(1/sqrt(3))
cx q[2], q[3];

cry(pi/4) q[3], q[4];
cx q[3], q[4];

// Measure all qubits
measure q -> c;
"#;
    
    match parse_qasm3(qasm_code) {
        Ok(program) => {
            println!("Successfully parsed QASM program!");
            println!("Version: {}", program.version);
            println!("Includes: {:?}", program.includes);
            println!("Declarations: {} items", program.declarations.len());
            println!("Statements: {} operations", program.statements.len());
            
            // Pretty print the parsed program
            println!("\nReconstructed QASM:");
            println!("{}", program);
        }
        Err(e) => println!("Parse error: {}", e),
    }
}

fn validation_example() {
    println!("\n3. Validating QASM programs");
    println!("---------------------------");
    
    // Valid program
    let valid_qasm = r#"
OPENQASM 3.0;

gate mybell a, b {
    h a;
    cx a, b;
}

qubit[4] q;
bit[2] c;

mybell q[0], q[1];
mybell q[2], q[3];

measure q[0] -> c[0];
measure q[2] -> c[1];
"#;
    
    println!("Validating correct program...");
    match parse_qasm3(valid_qasm) {
        Ok(program) => {
            match validate_qasm3(&program) {
                Ok(()) => println!("✓ Program is valid!"),
                Err(e) => println!("✗ Validation error: {}", e),
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }
    
    // Program with errors
    let invalid_qasm = r#"
OPENQASM 3.0;

qubit[2] q;
bit[2] c;

// Error: using undefined register
h r[0];

// Error: index out of bounds
cx q[0], q[5];

// Error: wrong number of parameters
rx q[0];  // Missing angle parameter
"#;
    
    println!("\nValidating program with errors...");
    match parse_qasm3(invalid_qasm) {
        Ok(program) => {
            match validate_qasm3(&program) {
                Ok(()) => println!("Program is valid (unexpected!)"),
                Err(e) => println!("✓ Caught validation error: {}", e),
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }
}

fn round_trip_example() {
    println!("\n4. Round-trip conversion");
    println!("------------------------");
    
    // Create a variational circuit
    let mut builder = CircuitBuilder::<4>::new();
    
    // Layer 1: Single-qubit rotations
    for i in 0..4 {
        builder.ry(Qubit::new(i), 0.5);
        builder.rz(Qubit::new(i), 0.3);
    }
    
    // Layer 2: Entangling gates
    for i in 0..3 {
        builder.cx(Qubit::new(i), Qubit::new(i + 1));
    }
    builder.cx(Qubit::new(3), Qubit::new(0)); // Circular connectivity
    
    // Layer 3: More rotations
    for i in 0..4 {
        builder.rx(Qubit::new(i), -0.2);
    }
    
    // Measurements
    for i in 0..4 {
        builder.measure(Qubit::new(i));
    }
    
    let original = builder.build();
    
    println!("Original circuit created with {} gates", original.gates().len());
    
    // Export to QASM
    match export_qasm3(&original) {
        Ok(qasm) => {
            println!("\nExported QASM:");
            println!("{}", qasm);
            
            // Parse it back
            match parse_qasm3(&qasm) {
                Ok(program) => {
                    println!("\n✓ Successfully parsed the exported QASM!");
                    
                    // Validate it
                    match validate_qasm3(&program) {
                        Ok(()) => println!("✓ Validation passed!"),
                        Err(e) => println!("✗ Validation error: {}", e),
                    }
                    
                    // Count operations
                    let gate_count = program.statements.iter()
                        .filter(|s| matches!(s, quantrs2_circuit::qasm::ast::QasmStatement::Gate(_)))
                        .count();
                    let measure_count = program.statements.iter()
                        .filter(|s| matches!(s, quantrs2_circuit::qasm::ast::QasmStatement::Measure(_)))
                        .count();
                    
                    println!("\nParsed circuit has:");
                    println!("  - {} gate operations", gate_count);
                    println!("  - {} measurements", measure_count);
                }
                Err(e) => println!("Parse error: {}", e),
            }
        }
        Err(e) => println!("Export error: {}", e),
    }
}

// Additional example showing advanced QASM 3.0 features
fn advanced_features_example() {
    println!("\n5. Advanced QASM 3.0 Features");
    println!("-----------------------------");
    
    let advanced_qasm = r#"
OPENQASM 3.0;
include "stdgates.inc";

// Constants and expressions
const n_layers = 3;
const rotation_angle = pi / 4;

// Quantum and classical registers
qubit[4] q;
bit[4] c;
bit[1] syndrome;

// Gate with modifiers
ctrl(2) x q[0], q[1], q[2];  // Toffoli gate
inv s q[3];                   // Inverse S gate

// Parameterized gate with expression
rx(2 * rotation_angle) q[0];

// For loop
for layer in [0:n_layers] {
    for i in [0:3] {
        ry(rotation_angle * (layer + 1)) q[i];
    }
    barrier q;
}

// Conditional operation
measure q[0] -> c[0];
if (c[0] == 1) {
    x q[1];
    measure q[1] -> syndrome[0];
}

// Reset and final measurements
reset q[0];
measure q -> c;
"#;
    
    match parse_qasm3(advanced_qasm) {
        Ok(program) => {
            println!("Successfully parsed advanced QASM features!");
            
            // Analyze the program
            let mut constants = 0;
            let mut gates_with_modifiers = 0;
            let mut control_flow = 0;
            
            for decl in &program.declarations {
                if matches!(decl, quantrs2_circuit::qasm::ast::Declaration::Constant(_, _)) {
                    constants += 1;
                }
            }
            
            for stmt in &program.statements {
                match stmt {
                    quantrs2_circuit::qasm::ast::QasmStatement::Gate(gate) => {
                        if gate.control.is_some() || gate.inverse || gate.power.is_some() {
                            gates_with_modifiers += 1;
                        }
                    }
                    quantrs2_circuit::qasm::ast::QasmStatement::If(_, _) |
                    quantrs2_circuit::qasm::ast::QasmStatement::For(_) |
                    quantrs2_circuit::qasm::ast::QasmStatement::While(_, _) => {
                        control_flow += 1;
                    }
                    _ => {}
                }
            }
            
            println!("\nProgram analysis:");
            println!("  - {} constants defined", constants);
            println!("  - {} gates with modifiers", gates_with_modifiers);
            println!("  - {} control flow statements", control_flow);
            
            // Validate
            match validate_qasm3(&program) {
                Ok(()) => println!("\n✓ All advanced features validated successfully!"),
                Err(e) => println!("\n✗ Validation error: {}", e),
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }
}