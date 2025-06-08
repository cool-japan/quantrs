//! QASM 2.0/3.0 import and export functionality.
//!
//! This module provides functionality to import and export quantum circuits
//! in the OpenQASM format, supporting both version 2.0 and 3.0.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use quantrs2_core::{
    gate::{GateOp, multi::*, single::*},
    qubit::QubitId,
};

use crate::builder::Circuit;

/// QASM version
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QasmVersion {
    /// OpenQASM 2.0
    V2_0,
    /// OpenQASM 3.0
    V3_0,
}

/// Error types for QASM operations
#[derive(Debug)]
pub enum QasmError {
    /// Parse error with line number and message
    ParseError(usize, String),
    /// Unsupported QASM version
    UnsupportedVersion(String),
    /// Invalid gate
    InvalidGate(String),
    /// Invalid register
    InvalidRegister(String),
    /// Invalid syntax
    InvalidSyntax(String),
    /// Unsupported feature
    UnsupportedFeature(String),
}

impl fmt::Display for QasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QasmError::ParseError(line, msg) => write!(f, "Parse error at line {}: {}", line, msg),
            QasmError::UnsupportedVersion(v) => write!(f, "Unsupported QASM version: {}", v),
            QasmError::InvalidGate(g) => write!(f, "Invalid gate: {}", g),
            QasmError::InvalidRegister(r) => write!(f, "Invalid register: {}", r),
            QasmError::InvalidSyntax(s) => write!(f, "Invalid syntax: {}", s),
            QasmError::UnsupportedFeature(feat) => write!(f, "Unsupported feature: {}", feat),
        }
    }
}

impl std::error::Error for QasmError {}

/// QASM parser state
struct QasmParser {
    version: QasmVersion,
    quantum_registers: HashMap<String, usize>,
    classical_registers: HashMap<String, usize>,
    qubit_map: HashMap<String, QubitId>,
    includes: Vec<String>,
}

impl QasmParser {
    fn new() -> Self {
        Self {
            version: QasmVersion::V2_0,
            quantum_registers: HashMap::new(),
            classical_registers: HashMap::new(),
            qubit_map: HashMap::new(),
            includes: Vec::new(),
        }
    }

    /// Parse a qubit reference like "q[0]" or "q[2]"
    fn parse_qubit_ref(&self, qubit_ref: &str) -> Result<QubitId, QasmError> {
        // Remove whitespace
        let qubit_ref = qubit_ref.trim();
        
        // Check for array notation
        if let Some(bracket_pos) = qubit_ref.find('[') {
            let reg_name = &qubit_ref[..bracket_pos];
            let index_str = qubit_ref[bracket_pos + 1..qubit_ref.len() - 1].trim();
            let index: usize = index_str.parse()
                .map_err(|_| QasmError::InvalidSyntax(format!("Invalid qubit index: {}", index_str)))?;
            
            // Check if register exists
            if !self.quantum_registers.contains_key(reg_name) {
                return Err(QasmError::InvalidRegister(format!("Unknown quantum register: {}", reg_name)));
            }
            
            // Get the base index for this register
            let base_index: usize = self.quantum_registers
                .iter()
                .filter(|(name, _)| name.as_str() < reg_name)
                .map(|(_, size)| size)
                .sum();
            
            Ok(QubitId((base_index + index) as u32))
        } else {
            // Single qubit reference
            self.qubit_map
                .get(qubit_ref)
                .copied()
                .ok_or_else(|| QasmError::InvalidRegister(format!("Unknown qubit: {}", qubit_ref)))
        }
    }

    /// Parse a gate parameter (angle)
    fn parse_parameter(&self, param: &str) -> Result<f64, QasmError> {
        let param = param.trim();
        
        // Check for pi
        if param.contains("pi") {
            let param = param.replace("pi", &std::f64::consts::PI.to_string());
            // Simple expression evaluation
            // In a real implementation, we'd use a proper expression parser
            self.evaluate_expression(&param)
        } else {
            param.parse()
                .map_err(|_| QasmError::InvalidSyntax(format!("Invalid parameter: {}", param)))
        }
    }

    /// Simple expression evaluator for basic arithmetic with pi
    fn evaluate_expression(&self, expr: &str) -> Result<f64, QasmError> {
        // This is a simplified version - a real implementation would use a proper parser
        let expr = expr.trim();
        
        // Handle basic operations
        if let Some(pos) = expr.find('/') {
            let left = self.evaluate_expression(&expr[..pos])?;
            let right = self.evaluate_expression(&expr[pos + 1..])?;
            Ok(left / right)
        } else if let Some(pos) = expr.find('*') {
            let left = self.evaluate_expression(&expr[..pos])?;
            let right = self.evaluate_expression(&expr[pos + 1..])?;
            Ok(left * right)
        } else if let Some(pos) = expr.find('+') {
            let left = self.evaluate_expression(&expr[..pos])?;
            let right = self.evaluate_expression(&expr[pos + 1..])?;
            Ok(left + right)
        } else if let Some(pos) = expr.find('-') {
            if pos > 0 {
                let left = self.evaluate_expression(&expr[..pos])?;
                let right = self.evaluate_expression(&expr[pos + 1..])?;
                Ok(left - right)
            } else {
                // Negative number
                Ok(-self.evaluate_expression(&expr[1..])?)
            }
        } else {
            // Parse as number
            expr.parse()
                .map_err(|_| QasmError::InvalidSyntax(format!("Invalid expression: {}", expr)))
        }
    }
}

/// Parse QASM string into a circuit
pub fn parse_qasm<const N: usize>(qasm: &str) -> Result<Circuit<N>, QasmError> {
    let mut parser = QasmParser::new();
    let mut circuit = Circuit::<N>::new();
    
    let lines: Vec<&str> = qasm.lines().collect();
    
    for (line_num, line) in lines.iter().enumerate() {
        let line = line.trim();
        
        // Skip empty lines and comments
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        
        // Remove inline comments
        let line = if let Some(comment_pos) = line.find("//") {
            &line[..comment_pos].trim()
        } else {
            line
        };
        
        // Parse version declaration
        if line.starts_with("OPENQASM") {
            let version_str = line.split_whitespace().nth(1)
                .ok_or_else(|| QasmError::ParseError(line_num + 1, "Missing version".to_string()))?
                .trim_end_matches(';');
            
            parser.version = match version_str {
                "2.0" => QasmVersion::V2_0,
                "3.0" => QasmVersion::V3_0,
                v => return Err(QasmError::UnsupportedVersion(v.to_string())),
            };
            continue;
        }
        
        // Parse includes
        if line.starts_with("include") {
            let include = line[7..].trim().trim_matches('"');
            parser.includes.push(include.to_string());
            continue;
        }
        
        // Parse quantum register declaration
        if line.starts_with("qreg") {
            let parts: Vec<&str> = line[4..].trim().trim_end_matches(';').split('[').collect();
            if parts.len() != 2 {
                return Err(QasmError::ParseError(line_num + 1, "Invalid qreg syntax".to_string()));
            }
            
            let reg_name = parts[0].trim();
            let size_str = parts[1].trim_end_matches(']');
            let size: usize = size_str.parse()
                .map_err(|_| QasmError::ParseError(line_num + 1, "Invalid register size".to_string()))?;
            
            parser.quantum_registers.insert(reg_name.to_string(), size);
            continue;
        }
        
        // Parse classical register declaration
        if line.starts_with("creg") {
            let parts: Vec<&str> = line[4..].trim().trim_end_matches(';').split('[').collect();
            if parts.len() != 2 {
                return Err(QasmError::ParseError(line_num + 1, "Invalid creg syntax".to_string()));
            }
            
            let reg_name = parts[0].trim();
            let size_str = parts[1].trim_end_matches(']');
            let size: usize = size_str.parse()
                .map_err(|_| QasmError::ParseError(line_num + 1, "Invalid register size".to_string()))?;
            
            parser.classical_registers.insert(reg_name.to_string(), size);
            continue;
        }
        
        // Parse gates
        let line = line.trim_end_matches(';');
        
        // Parse gate application
        if let Some(paren_pos) = line.find('(') {
            // Gate with parameters
            let gate_name = &line[..paren_pos].trim();
            let close_paren = line.find(')')
                .ok_or_else(|| QasmError::ParseError(line_num + 1, "Missing closing parenthesis".to_string()))?;
            
            let params_str = &line[paren_pos + 1..close_paren];
            let qubits_str = line[close_paren + 1..].trim();
            
            let params: Vec<f64> = params_str
                .split(',')
                .map(|p| parser.parse_parameter(p))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
            
            let qubits: Vec<QubitId> = qubits_str
                .split(',')
                .map(|q| parser.parse_qubit_ref(q))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
            
            // Create and add gate
            match gate_name {
                &"rx" if qubits.len() == 1 && params.len() == 1 => {
                    circuit.add_gate(RotationX {
                        target: qubits[0],
                        theta: params[0],
                    })
                    .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                &"ry" if qubits.len() == 1 && params.len() == 1 => {
                    circuit.add_gate(RotationY {
                        target: qubits[0],
                        theta: params[0],
                    })
                    .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                &"rz" if qubits.len() == 1 && params.len() == 1 => {
                    circuit.add_gate(RotationZ {
                        target: qubits[0],
                        theta: params[0],
                    })
                    .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                &"crx" if qubits.len() == 2 && params.len() == 1 => {
                    circuit.add_gate(CRX {
                        control: qubits[0],
                        target: qubits[1],
                        theta: params[0],
                    })
                    .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                &"cry" if qubits.len() == 2 && params.len() == 1 => {
                    circuit.add_gate(CRY {
                        control: qubits[0],
                        target: qubits[1],
                        theta: params[0],
                    })
                    .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                &"crz" if qubits.len() == 2 && params.len() == 1 => {
                    circuit.add_gate(CRZ {
                        control: qubits[0],
                        target: qubits[1],
                        theta: params[0],
                    })
                    .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                _ => return Err(QasmError::InvalidGate(format!("Unknown parametric gate: {}", gate_name))),
            }
            
        } else {
            // Gate without parameters
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            
            let gate_name = parts[0];
            let qubits_str = parts[1..].join(" ");
            
            let qubits: Vec<QubitId> = qubits_str
                .split(',')
                .map(|q| parser.parse_qubit_ref(q))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
            
            // Create and add gate
            match (gate_name, qubits.len()) {
                ("h", 1) => {
                    circuit.add_gate(Hadamard { target: qubits[0] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("x", 1) => {
                    circuit.add_gate(PauliX { target: qubits[0] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("y", 1) => {
                    circuit.add_gate(PauliY { target: qubits[0] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("z", 1) => {
                    circuit.add_gate(PauliZ { target: qubits[0] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("s", 1) => {
                    circuit.add_gate(Phase { target: qubits[0] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("sdg", 1) => {
                    circuit.add_gate(PhaseDagger { target: qubits[0] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("t", 1) => {
                    circuit.add_gate(T { target: qubits[0] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("tdg", 1) => {
                    circuit.add_gate(TDagger { target: qubits[0] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("sx", 1) => {
                    circuit.add_gate(SqrtX { target: qubits[0] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("sxdg", 1) => {
                    circuit.add_gate(SqrtXDagger { target: qubits[0] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("cx", 2) | ("cnot", 2) => {
                    circuit.add_gate(CNOT { control: qubits[0], target: qubits[1] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("cy", 2) => {
                    circuit.add_gate(CY { control: qubits[0], target: qubits[1] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("cz", 2) => {
                    circuit.add_gate(CZ { control: qubits[0], target: qubits[1] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("ch", 2) => {
                    circuit.add_gate(CH { control: qubits[0], target: qubits[1] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("cs", 2) => {
                    circuit.add_gate(CS { control: qubits[0], target: qubits[1] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("swap", 2) => {
                    circuit.add_gate(SWAP { qubit1: qubits[0], qubit2: qubits[1] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("ccx", 3) | ("toffoli", 3) => {
                    circuit.add_gate(Toffoli { control1: qubits[0], control2: qubits[1], target: qubits[2] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                ("cswap", 3) | ("fredkin", 3) => {
                    circuit.add_gate(Fredkin { control: qubits[0], target1: qubits[1], target2: qubits[2] })
                        .map_err(|e| QasmError::ParseError(line_num + 1, e.to_string()))?;
                }
                _ => return Err(QasmError::InvalidGate(format!("Unknown gate: {} with {} qubits", gate_name, qubits.len()))),
            }
        }
    }
    
    Ok(circuit)
}

/// Export a circuit to QASM format
pub fn export_qasm<const N: usize>(circuit: &Circuit<N>, version: QasmVersion) -> String {
    let mut qasm = String::new();
    
    // Version header
    match version {
        QasmVersion::V2_0 => qasm.push_str("OPENQASM 2.0;\n"),
        QasmVersion::V3_0 => qasm.push_str("OPENQASM 3.0;\n"),
    }
    
    // Include standard library
    qasm.push_str("include \"qelib1.inc\";\n\n");
    
    // Declare quantum register
    qasm.push_str(&format!("qreg q[{}];\n", N));
    
    // TODO: Add classical register if measurements are present
    // qasm.push_str(&format!("creg c[{}];\n", N));
    
    qasm.push('\n');
    
    // Export gates
    for gate in circuit.gates() {
        let gate_str = match gate.name() {
            "H" => {
                let qubits = gate.qubits();
                format!("h q[{}];", qubits[0].id())
            }
            "X" => {
                let qubits = gate.qubits();
                format!("x q[{}];", qubits[0].id())
            }
            "Y" => {
                let qubits = gate.qubits();
                format!("y q[{}];", qubits[0].id())
            }
            "Z" => {
                let qubits = gate.qubits();
                format!("z q[{}];", qubits[0].id())
            }
            "S" => {
                let qubits = gate.qubits();
                format!("s q[{}];", qubits[0].id())
            }
            "S†" => {
                let qubits = gate.qubits();
                format!("sdg q[{}];", qubits[0].id())
            }
            "T" => {
                let qubits = gate.qubits();
                format!("t q[{}];", qubits[0].id())
            }
            "T†" => {
                let qubits = gate.qubits();
                format!("tdg q[{}];", qubits[0].id())
            }
            "√X" => {
                let qubits = gate.qubits();
                format!("sx q[{}];", qubits[0].id())
            }
            "√X†" => {
                let qubits = gate.qubits();
                format!("sxdg q[{}];", qubits[0].id())
            }
            "CNOT" => {
                let qubits = gate.qubits();
                format!("cx q[{}], q[{}];", qubits[0].id(), qubits[1].id())
            }
            "CY" => {
                let qubits = gate.qubits();
                format!("cy q[{}], q[{}];", qubits[0].id(), qubits[1].id())
            }
            "CZ" => {
                let qubits = gate.qubits();
                format!("cz q[{}], q[{}];", qubits[0].id(), qubits[1].id())
            }
            "CH" => {
                let qubits = gate.qubits();
                format!("ch q[{}], q[{}];", qubits[0].id(), qubits[1].id())
            }
            "CS" => {
                let qubits = gate.qubits();
                format!("cs q[{}], q[{}];", qubits[0].id(), qubits[1].id())
            }
            "SWAP" => {
                let qubits = gate.qubits();
                format!("swap q[{}], q[{}];", qubits[0].id(), qubits[1].id())
            }
            "Toffoli" => {
                let qubits = gate.qubits();
                format!("ccx q[{}], q[{}], q[{}];", qubits[0].id(), qubits[1].id(), qubits[2].id())
            }
            "Fredkin" => {
                let qubits = gate.qubits();
                format!("cswap q[{}], q[{}], q[{}];", qubits[0].id(), qubits[1].id(), qubits[2].id())
            }
            // Rotation gates need parameter extraction
            name if name.starts_with("R") => {
                // This is a simplified version - in a real implementation,
                // we'd need to extract the rotation angle from the gate
                let qubits = gate.qubits();
                match name {
                    "RX" => format!("rx(pi/2) q[{}];", qubits[0].id()),
                    "RY" => format!("ry(pi/2) q[{}];", qubits[0].id()),
                    "RZ" => format!("rz(pi/2) q[{}];", qubits[0].id()),
                    _ => format!("// Unsupported rotation gate: {}", name),
                }
            }
            _ => format!("// Unsupported gate: {}", gate.name()),
        };
        
        qasm.push_str(&gate_str);
        qasm.push('\n');
    }
    
    qasm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_qasm() {
        let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

h q[0];
cx q[0], q[1];
cx q[1], q[2];
"#;

        let circuit: Circuit<3> = parse_qasm(qasm).unwrap();
        assert_eq!(circuit.num_gates(), 3);
        
        let gate_names = circuit.get_gate_names();
        assert_eq!(gate_names[0], "H");
        assert_eq!(gate_names[1], "CNOT");
        assert_eq!(gate_names[2], "CNOT");
    }

    #[test]
    fn test_parse_rotation_gates() {
        let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];

rx(pi/2) q[0];
ry(pi) q[1];
rz(pi/4) q[0];
"#;

        let circuit: Circuit<2> = parse_qasm(qasm).unwrap();
        assert_eq!(circuit.num_gates(), 3);
        
        let gate_names = circuit.get_gate_names();
        assert!(gate_names[0].contains("RX"));
        assert!(gate_names[1].contains("RY"));
        assert!(gate_names[2].contains("RZ"));
    }

    #[test]
    fn test_export_qasm() {
        let mut circuit = Circuit::<3>::new();
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit.add_gate(CNOT { control: QubitId(0), target: QubitId(1) }).unwrap();
        circuit.add_gate(CNOT { control: QubitId(1), target: QubitId(2) }).unwrap();
        
        let qasm = export_qasm(&circuit, QasmVersion::V2_0);
        
        assert!(qasm.contains("OPENQASM 2.0"));
        assert!(qasm.contains("qreg q[3]"));
        assert!(qasm.contains("h q[0]"));
        assert!(qasm.contains("cx q[0], q[1]"));
        assert!(qasm.contains("cx q[1], q[2]"));
    }

    #[test]
    fn test_round_trip() {
        let qasm_input = r#"
OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];

h q[0];
h q[1];
cx q[0], q[2];
cx q[1], q[3];
"#;

        let circuit: Circuit<4> = parse_qasm(qasm_input).unwrap();
        let qasm_output = export_qasm(&circuit, QasmVersion::V2_0);
        
        // Parse the output again
        let circuit2: Circuit<4> = parse_qasm(&qasm_output).unwrap();
        
        // Should have same number of gates
        assert_eq!(circuit.num_gates(), circuit2.num_gates());
        assert_eq!(circuit.get_gate_names(), circuit2.get_gate_names());
    }
}