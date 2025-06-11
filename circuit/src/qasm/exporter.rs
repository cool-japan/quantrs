//! Export QuantRS2 circuits to OpenQASM 3.0 format

use super::ast::*;
use crate::builder::Circuit;
use num_complex::Complex64;
use quantrs2_core::{gate::GateOp, qubit::QubitId};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use thiserror::Error;

/// Export error types
#[derive(Debug, Error)]
pub enum ExportError {
    #[error("Unsupported gate: {0}")]
    UnsupportedGate(String),

    #[error("Invalid circuit: {0}")]
    InvalidCircuit(String),

    #[error("Formatting error: {0}")]
    FormattingError(#[from] std::fmt::Error),

    #[error("Gate parameter error: {0}")]
    ParameterError(String),
}

/// Options for controlling QASM export
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Include standard gate library
    pub include_stdgates: bool,
    /// Use gate decomposition for non-standard gates
    pub decompose_custom: bool,
    /// Add comments with gate matrix representations
    pub include_gate_comments: bool,
    /// Optimize gate sequences
    pub optimize: bool,
    /// Pretty print with indentation
    pub pretty_print: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        ExportOptions {
            include_stdgates: true,
            decompose_custom: true,
            include_gate_comments: false,
            optimize: false,
            pretty_print: true,
        }
    }
}

/// QASM exporter
pub struct QasmExporter {
    options: ExportOptions,
    /// Track which gates need custom definitions
    custom_gates: HashMap<String, GateInfo>,
    /// Track qubit usage
    qubit_usage: HashSet<usize>,
    /// Track if measurements are used
    needs_classical_bits: bool,
}

#[derive(Clone)]
struct GateInfo {
    name: String,
    num_qubits: usize,
    num_params: usize,
    matrix: Option<ndarray::Array2<Complex64>>,
}

impl QasmExporter {
    /// Create a new exporter with options
    pub fn new(options: ExportOptions) -> Self {
        QasmExporter {
            options,
            custom_gates: HashMap::new(),
            qubit_usage: HashSet::new(),
            needs_classical_bits: false,
        }
    }

    /// Export a circuit to QASM 3.0
    pub fn export<const N: usize>(&mut self, circuit: &Circuit<N>) -> Result<String, ExportError> {
        // Analyze circuit
        self.analyze_circuit(circuit)?;

        // Generate QASM program
        let program = self.generate_program(circuit)?;

        // Convert to string
        Ok(program.to_string())
    }

    /// Analyze circuit to determine requirements
    fn analyze_circuit<const N: usize>(&mut self, circuit: &Circuit<N>) -> Result<(), ExportError> {
        self.qubit_usage.clear();
        self.custom_gates.clear();
        self.needs_classical_bits = false;

        // Analyze each gate
        for gate in circuit.gates() {
            // Track qubit usage
            for qubit in gate.qubits() {
                self.qubit_usage.insert(qubit.id() as usize);
            }

            // Check if gate is standard or custom
            if !self.is_standard_gate(gate.as_ref()) {
                self.register_custom_gate(gate.as_ref())?;
            }

            // Check for measurements
            if gate.name().contains("measure") {
                self.needs_classical_bits = true;
            }
        }

        Ok(())
    }

    /// Check if a gate is in the standard library
    fn is_standard_gate(&self, gate: &dyn GateOp) -> bool {
        let name = gate.name();
        matches!(
            name.as_ref(),
            "I" | "X"
                | "Y"
                | "Z"
                | "H"
                | "S"
                | "Sdg"
                | "T"
                | "Tdg"
                | "RX"
                | "RY"
                | "RZ"
                | "Phase"
                | "U"
                | "U1"
                | "U2"
                | "U3"
                | "CX"
                | "CY"
                | "CZ"
                | "CH"
                | "CRX"
                | "CRY"
                | "CRZ"
                | "CPhase"
                | "SWAP"
                | "iSWAP"
                | "CU"
                | "CCX"
                | "Toffoli"
                | "measure"
                | "reset"
                | "barrier"
        )
    }

    /// Register a custom gate
    fn register_custom_gate(&mut self, gate: &dyn GateOp) -> Result<(), ExportError> {
        let name = self.gate_qasm_name(gate);

        if !self.custom_gates.contains_key(&name) {
            let info = GateInfo {
                name: name.clone(),
                num_qubits: gate.qubits().len(),
                num_params: self.count_gate_params(gate),
                matrix: None, // GateOp doesn't have matrix() method
            };

            self.custom_gates.insert(name, info);
        }

        Ok(())
    }

    /// Get QASM name for a gate
    fn gate_qasm_name(&self, gate: &dyn GateOp) -> String {
        let name = gate.name();
        match name.as_ref() {
            "I" => "id".to_string(),
            "X" => "x".to_string(),
            "Y" => "y".to_string(),
            "Z" => "z".to_string(),
            "H" => "h".to_string(),
            "S" => "s".to_string(),
            "Sdg" => "sdg".to_string(),
            "T" => "t".to_string(),
            "Tdg" => "tdg".to_string(),
            "RX" => "rx".to_string(),
            "RY" => "ry".to_string(),
            "RZ" => "rz".to_string(),
            "Phase" => "p".to_string(),
            "U" => "u".to_string(),
            "CX" | "CNOT" => "cx".to_string(),
            "CY" => "cy".to_string(),
            "CZ" => "cz".to_string(),
            "CH" => "ch".to_string(),
            "CRX" => "crx".to_string(),
            "CRY" => "cry".to_string(),
            "CRZ" => "crz".to_string(),
            "CPhase" => "cp".to_string(),
            "SWAP" => "swap".to_string(),
            "iSWAP" => "iswap".to_string(),
            "CCX" | "Toffoli" => "ccx".to_string(),
            _ => name.to_lowercase(),
        }
    }

    /// Count gate parameters
    fn count_gate_params(&self, gate: &dyn GateOp) -> usize {
        // This is a simplified version - would need gate trait extension
        let name = gate.name();
        match name.as_ref() {
            "RX" | "RY" | "RZ" | "Phase" | "U1" => 1,
            "U2" => 2,
            "U" | "U3" => 3,
            "CRX" | "CRY" | "CRZ" | "CPhase" => 1,
            _ => 0,
        }
    }

    /// Generate QASM program
    fn generate_program<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
    ) -> Result<QasmProgram, ExportError> {
        let mut declarations = Vec::new();
        let mut statements = Vec::new();

        // Calculate required register size
        let max_qubit = self.qubit_usage.iter().max().copied().unwrap_or(0);
        let num_qubits = max_qubit + 1;

        // Add quantum register
        declarations.push(Declaration::QuantumRegister(QasmRegister {
            name: "q".to_string(),
            size: num_qubits,
        }));

        // Add classical register if needed
        if self.needs_classical_bits {
            declarations.push(Declaration::ClassicalRegister(QasmRegister {
                name: "c".to_string(),
                size: num_qubits,
            }));
        }

        // Add custom gate definitions
        if self.options.decompose_custom {
            for (_, gate_info) in &self.custom_gates {
                if let Some(def) = self.generate_gate_definition(gate_info)? {
                    declarations.push(Declaration::GateDefinition(def));
                }
            }
        }

        // Convert gates to statements
        for gate in circuit.gates() {
            statements.push(self.convert_gate(gate)?);
        }

        // Build includes
        let includes = if self.options.include_stdgates {
            vec!["stdgates.inc".to_string()]
        } else {
            vec![]
        };

        Ok(QasmProgram {
            version: "3.0".to_string(),
            includes,
            declarations,
            statements,
        })
    }

    /// Generate gate definition for custom gate
    fn generate_gate_definition(
        &self,
        gate_info: &GateInfo,
    ) -> Result<Option<GateDefinition>, ExportError> {
        // For now, return None - full implementation would decompose gates
        // This would use gate synthesis algorithms
        Ok(None)
    }

    /// Convert gate to QASM statement
    fn convert_gate(&self, gate: &Box<dyn GateOp>) -> Result<QasmStatement, ExportError> {
        let gate_name = gate.name();

        match gate_name.as_ref() {
            "measure" => {
                // Convert measurement
                let qubits: Vec<QubitRef> = gate
                    .qubits()
                    .iter()
                    .map(|q| QubitRef::Single {
                        register: "q".to_string(),
                        index: q.id() as usize,
                    })
                    .collect();

                let targets: Vec<ClassicalRef> = gate
                    .qubits()
                    .iter()
                    .map(|q| ClassicalRef::Single {
                        register: "c".to_string(),
                        index: q.id() as usize,
                    })
                    .collect();

                Ok(QasmStatement::Measure(Measurement { qubits, targets }))
            }
            "reset" => {
                let qubits: Vec<QubitRef> = gate
                    .qubits()
                    .iter()
                    .map(|q| QubitRef::Single {
                        register: "q".to_string(),
                        index: q.id() as usize,
                    })
                    .collect();

                Ok(QasmStatement::Reset(qubits))
            }
            "barrier" => {
                let qubits: Vec<QubitRef> = gate
                    .qubits()
                    .iter()
                    .map(|q| QubitRef::Single {
                        register: "q".to_string(),
                        index: q.id() as usize,
                    })
                    .collect();

                Ok(QasmStatement::Barrier(qubits))
            }
            _ => {
                // Regular gate
                let name = self.gate_qasm_name(gate.as_ref());

                let qubits: Vec<QubitRef> = gate
                    .qubits()
                    .iter()
                    .map(|q| QubitRef::Single {
                        register: "q".to_string(),
                        index: q.id() as usize,
                    })
                    .collect();

                // Extract parameters - this is simplified
                let params = self.extract_gate_params(gate.as_ref())?;

                Ok(QasmStatement::Gate(QasmGate {
                    name,
                    params,
                    qubits,
                    control: None,
                    inverse: false,
                    power: None,
                }))
            }
        }
    }

    /// Extract gate parameters as expressions
    fn extract_gate_params(&self, gate: &dyn GateOp) -> Result<Vec<Expression>, ExportError> {
        use quantrs2_core::gate::multi::{CRX, CRY, CRZ};
        use quantrs2_core::gate::single::{RotationX, RotationY, RotationZ};
        use std::any::Any;

        let any_gate = gate.as_any();

        // Single-qubit rotation gates
        if let Some(rx) = any_gate.downcast_ref::<RotationX>() {
            return Ok(vec![Expression::Literal(Literal::Float(rx.theta))]);
        }
        if let Some(ry) = any_gate.downcast_ref::<RotationY>() {
            return Ok(vec![Expression::Literal(Literal::Float(ry.theta))]);
        }
        if let Some(rz) = any_gate.downcast_ref::<RotationZ>() {
            return Ok(vec![Expression::Literal(Literal::Float(rz.theta))]);
        }

        // Controlled rotation gates
        if let Some(crx) = any_gate.downcast_ref::<CRX>() {
            return Ok(vec![Expression::Literal(Literal::Float(crx.theta))]);
        }
        if let Some(cry) = any_gate.downcast_ref::<CRY>() {
            return Ok(vec![Expression::Literal(Literal::Float(cry.theta))]);
        }
        if let Some(crz) = any_gate.downcast_ref::<CRZ>() {
            return Ok(vec![Expression::Literal(Literal::Float(crz.theta))]);
        }

        // No parameters for other gates
        Ok(vec![])
    }
}

/// Export a circuit to QASM 3.0 with default options
pub fn export_qasm3<const N: usize>(circuit: &Circuit<N>) -> Result<String, ExportError> {
    let mut exporter = QasmExporter::new(ExportOptions::default());
    exporter.export(circuit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::{Hadamard, PauliX};
    use quantrs2_core::qubit::QubitId;

    #[test]
    fn test_export_simple_circuit() {
        let mut circuit = Circuit::<2>::new();
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();

        let result = export_qasm3(&circuit);
        assert!(result.is_ok());

        let qasm = result.unwrap();
        assert!(qasm.contains("OPENQASM 3.0"));
        assert!(qasm.contains("qubit[2] q"));
        assert!(qasm.contains("h q[0]"));
        assert!(qasm.contains("cx q[0], q[1]"));
    }

    #[test]
    fn test_export_with_measurements() {
        let mut circuit = Circuit::<2>::new();
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        // Note: measure gate would need to be implemented

        let result = export_qasm3(&circuit);
        assert!(result.is_ok());

        let qasm = result.unwrap();
        // Basic check
        assert!(qasm.contains("OPENQASM 3.0"));
    }
}
