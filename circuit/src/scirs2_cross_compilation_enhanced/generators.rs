//! Target code generators for different quantum platforms
//!
//! This module contains code generators for IBM Quantum, Google Sycamore,
//! IonQ, Rigetti, and generic platforms.

use super::config::{EnhancedCrossCompilationConfig, TargetPlatform};
use super::types::{CodeFormat, IRGate, IROperationType, QuantumIR, TargetCode};
use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use std::sync::Arc;

use std::fmt::Write;
/// Target code generator trait
pub trait TargetCodeGenerator: Send + Sync {
    fn generate(&self, ir: &QuantumIR) -> QuantRS2Result<TargetCode>;
}

/// Create target generator for platform
pub fn create_target_generator(
    platform: TargetPlatform,
    config: EnhancedCrossCompilationConfig,
) -> Arc<dyn TargetCodeGenerator> {
    match platform {
        TargetPlatform::IBMQuantum => Arc::new(IBMQuantumGenerator::new(config)),
        TargetPlatform::GoogleSycamore => Arc::new(GoogleSycamoreGenerator::new(config)),
        TargetPlatform::IonQ => Arc::new(IonQGenerator::new(config)),
        TargetPlatform::Rigetti => Arc::new(RigettiGenerator::new(config)),
        _ => Arc::new(GenericGenerator::new(config)),
    }
}

/// IBM Quantum code generator
pub struct IBMQuantumGenerator {
    config: EnhancedCrossCompilationConfig,
}

impl IBMQuantumGenerator {
    pub const fn new(config: EnhancedCrossCompilationConfig) -> Self {
        Self { config }
    }

    fn generate_qasm(ir: &QuantumIR) -> QuantRS2Result<String> {
        let mut qasm = String::new();

        // Header
        qasm.push_str("OPENQASM 2.0;\n");
        qasm.push_str("include \"qelib1.inc\";\n\n");

        // Quantum registers
        writeln!(qasm, "qreg q[{}];", ir.num_qubits).expect("Writing to String is infallible");

        // Classical registers
        if ir.num_classical_bits > 0 {
            writeln!(qasm, "creg c[{}];", ir.num_classical_bits)
                .expect("Writing to String is infallible");
        }

        qasm.push('\n');

        // Operations
        for op in &ir.operations {
            let gate_str = Self::ir_op_to_qasm(op)?;
            writeln!(qasm, "{gate_str}").expect("Writing to String is infallible");
        }

        Ok(qasm)
    }

    fn ir_op_to_qasm(op: &super::types::IROperation) -> QuantRS2Result<String> {
        match &op.operation_type {
            IROperationType::Gate(gate) => Self::gate_to_qasm(gate, &op.qubits),
            IROperationType::Measurement(qubits, bits) => {
                Ok(format!("measure q[{}] -> c[{}];", qubits[0], bits[0]))
            }
            _ => Err(QuantRS2Error::UnsupportedOperation(format!(
                "Operation {:?} not supported in QASM",
                op.operation_type
            ))),
        }
    }

    fn gate_to_qasm(gate: &IRGate, qubits: &[usize]) -> QuantRS2Result<String> {
        match gate {
            IRGate::H => Ok(format!("h q[{}];", qubits[0])),
            IRGate::X => Ok(format!("x q[{}];", qubits[0])),
            IRGate::Y => Ok(format!("y q[{}];", qubits[0])),
            IRGate::Z => Ok(format!("z q[{}];", qubits[0])),
            IRGate::CNOT => Ok(format!("cx q[{}], q[{}];", qubits[0], qubits[1])),
            IRGate::RX(angle) => Ok(format!("rx({}) q[{}];", angle, qubits[0])),
            IRGate::RY(angle) => Ok(format!("ry({}) q[{}];", angle, qubits[0])),
            IRGate::RZ(angle) => Ok(format!("rz({}) q[{}];", angle, qubits[0])),
            _ => Err(QuantRS2Error::UnsupportedOperation(format!(
                "Gate {gate:?} not supported in QASM"
            ))),
        }
    }
}

impl TargetCodeGenerator for IBMQuantumGenerator {
    fn generate(&self, ir: &QuantumIR) -> QuantRS2Result<TargetCode> {
        let mut code = TargetCode::new(TargetPlatform::IBMQuantum);

        // Generate QASM code for IBM Quantum
        let qasm = Self::generate_qasm(ir)?;
        code.code = qasm;
        code.format = CodeFormat::QASM;

        // Add IBM-specific metadata
        code.metadata
            .insert("backend".to_string(), "ibmq_qasm_simulator".to_string());

        Ok(code)
    }
}

/// Google Sycamore code generator
pub struct GoogleSycamoreGenerator {
    config: EnhancedCrossCompilationConfig,
}

impl GoogleSycamoreGenerator {
    pub const fn new(config: EnhancedCrossCompilationConfig) -> Self {
        Self { config }
    }

    fn generate_cirq(ir: &QuantumIR) -> QuantRS2Result<String> {
        let mut code = String::new();

        // Imports
        code.push_str("import cirq\n");
        code.push_str("import numpy as np\n\n");

        // Create qubits
        writeln!(code, "qubits = cirq.LineQubit.range({})", ir.num_qubits)
            .expect("Writing to String is infallible");
        code.push_str("circuit = cirq.Circuit()\n\n");

        // Add operations
        for op in &ir.operations {
            let op_str = Self::ir_op_to_cirq(op)?;
            writeln!(code, "circuit.append({op_str})").expect("Writing to String is infallible");
        }

        Ok(code)
    }

    fn ir_op_to_cirq(op: &super::types::IROperation) -> QuantRS2Result<String> {
        match &op.operation_type {
            IROperationType::Gate(gate) => Self::gate_to_cirq(gate, &op.qubits),
            IROperationType::Measurement(qubits, _) => Ok(format!(
                "cirq.measure(qubits[{}], key='m{}')",
                qubits[0], qubits[0]
            )),
            _ => Err(QuantRS2Error::UnsupportedOperation(format!(
                "Operation {:?} not supported in Cirq",
                op.operation_type
            ))),
        }
    }

    fn gate_to_cirq(gate: &IRGate, qubits: &[usize]) -> QuantRS2Result<String> {
        match gate {
            IRGate::H => Ok(format!("cirq.H(qubits[{}])", qubits[0])),
            IRGate::X => Ok(format!("cirq.X(qubits[{}])", qubits[0])),
            IRGate::Y => Ok(format!("cirq.Y(qubits[{}])", qubits[0])),
            IRGate::Z => Ok(format!("cirq.Z(qubits[{}])", qubits[0])),
            IRGate::CNOT => Ok(format!(
                "cirq.CNOT(qubits[{}], qubits[{}])",
                qubits[0], qubits[1]
            )),
            IRGate::RX(angle) => Ok(format!("cirq.rx({}).on(qubits[{}])", angle, qubits[0])),
            IRGate::RY(angle) => Ok(format!("cirq.ry({}).on(qubits[{}])", angle, qubits[0])),
            IRGate::RZ(angle) => Ok(format!("cirq.rz({}).on(qubits[{}])", angle, qubits[0])),
            IRGate::SqrtISWAP => Ok(format!(
                "cirq.SQRT_ISWAP(qubits[{}], qubits[{}])",
                qubits[0], qubits[1]
            )),
            _ => Err(QuantRS2Error::UnsupportedOperation(format!(
                "Gate {gate:?} not supported in Cirq"
            ))),
        }
    }
}

impl TargetCodeGenerator for GoogleSycamoreGenerator {
    fn generate(&self, ir: &QuantumIR) -> QuantRS2Result<TargetCode> {
        let mut code = TargetCode::new(TargetPlatform::GoogleSycamore);

        // Generate Cirq code for Google Sycamore
        let cirq_code = Self::generate_cirq(ir)?;
        code.code = cirq_code;
        code.format = CodeFormat::Cirq;

        Ok(code)
    }
}

/// `IonQ` code generator
pub struct IonQGenerator {
    config: EnhancedCrossCompilationConfig,
}

impl IonQGenerator {
    pub const fn new(config: EnhancedCrossCompilationConfig) -> Self {
        Self { config }
    }

    fn generate_ionq_json(ir: &QuantumIR) -> QuantRS2Result<String> {
        let mut circuit = serde_json::json!({
            "format": "ionq.circuit.v0",
            "qubits": ir.num_qubits,
            "circuit": []
        });

        let circuit_ops = circuit["circuit"].as_array_mut().ok_or_else(|| {
            QuantRS2Error::RuntimeError("Failed to get circuit array".to_string())
        })?;

        for op in &ir.operations {
            if let IROperationType::Gate(gate) = &op.operation_type {
                let ionq_op = Self::ir_gate_to_ionq(gate, &op.qubits)?;
                circuit_ops.push(ionq_op);
            }
        }

        Ok(serde_json::to_string_pretty(&circuit)?)
    }

    fn ir_gate_to_ionq(gate: &IRGate, qubits: &[usize]) -> QuantRS2Result<serde_json::Value> {
        match gate {
            IRGate::H => Ok(serde_json::json!({
                "gate": "h",
                "target": qubits[0]
            })),
            IRGate::X => Ok(serde_json::json!({
                "gate": "x",
                "target": qubits[0]
            })),
            IRGate::Y => Ok(serde_json::json!({
                "gate": "y",
                "target": qubits[0]
            })),
            IRGate::Z => Ok(serde_json::json!({
                "gate": "z",
                "target": qubits[0]
            })),
            IRGate::CNOT => Ok(serde_json::json!({
                "gate": "cnot",
                "control": qubits[0],
                "target": qubits[1]
            })),
            _ => Err(QuantRS2Error::UnsupportedOperation(format!(
                "Gate {gate:?} not supported on IonQ"
            ))),
        }
    }
}

impl TargetCodeGenerator for IonQGenerator {
    fn generate(&self, ir: &QuantumIR) -> QuantRS2Result<TargetCode> {
        let mut code = TargetCode::new(TargetPlatform::IonQ);

        // Generate IonQ JSON format
        let ionq_json = Self::generate_ionq_json(ir)?;
        code.code = ionq_json;
        code.format = CodeFormat::IonQJSON;

        Ok(code)
    }
}

/// Rigetti code generator
pub struct RigettiGenerator {
    config: EnhancedCrossCompilationConfig,
}

impl RigettiGenerator {
    pub const fn new(config: EnhancedCrossCompilationConfig) -> Self {
        Self { config }
    }

    fn generate_quil(ir: &QuantumIR) -> QuantRS2Result<String> {
        let mut quil = String::new();

        // Declare qubits (implicit in Quil)

        // Generate gates
        for op in &ir.operations {
            if let IROperationType::Gate(gate) = &op.operation_type {
                let gate_str = Self::ir_gate_to_quil(gate, &op.qubits)?;
                writeln!(quil, "{gate_str}").expect("Writing to String is infallible");
            } else if let IROperationType::Measurement(qubits, bits) = &op.operation_type {
                writeln!(quil, "MEASURE {} ro[{}]", qubits[0], bits[0])
                    .expect("Writing to String is infallible");
            }
        }

        Ok(quil)
    }

    fn ir_gate_to_quil(gate: &IRGate, qubits: &[usize]) -> QuantRS2Result<String> {
        match gate {
            IRGate::H => Ok(format!("H {}", qubits[0])),
            IRGate::X => Ok(format!("X {}", qubits[0])),
            IRGate::Y => Ok(format!("Y {}", qubits[0])),
            IRGate::Z => Ok(format!("Z {}", qubits[0])),
            IRGate::CNOT => Ok(format!("CNOT {} {}", qubits[0], qubits[1])),
            IRGate::RX(angle) => Ok(format!("RX({}) {}", angle, qubits[0])),
            IRGate::RY(angle) => Ok(format!("RY({}) {}", angle, qubits[0])),
            IRGate::RZ(angle) => Ok(format!("RZ({}) {}", angle, qubits[0])),
            _ => Err(QuantRS2Error::UnsupportedOperation(format!(
                "Gate {gate:?} not supported in Quil"
            ))),
        }
    }
}

impl TargetCodeGenerator for RigettiGenerator {
    fn generate(&self, ir: &QuantumIR) -> QuantRS2Result<TargetCode> {
        let mut code = TargetCode::new(TargetPlatform::Rigetti);

        // Generate Quil code
        let quil = Self::generate_quil(ir)?;
        code.code = quil;
        code.format = CodeFormat::Quil;

        Ok(code)
    }
}

/// Generic code generator
pub struct GenericGenerator {
    config: EnhancedCrossCompilationConfig,
}

impl GenericGenerator {
    pub const fn new(config: EnhancedCrossCompilationConfig) -> Self {
        Self { config }
    }
}

impl TargetCodeGenerator for GenericGenerator {
    fn generate(&self, ir: &QuantumIR) -> QuantRS2Result<TargetCode> {
        // Generate generic quantum assembly
        Ok(TargetCode::new(TargetPlatform::Simulator))
    }
}
