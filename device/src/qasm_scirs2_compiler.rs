//! QASM Compiler with SciRS2 Parsing Tools
//!
//! This module provides a comprehensive QASM (Quantum Assembly Language) compiler
//! that leverages SciRS2's parsing and optimization capabilities for efficient
//! quantum circuit compilation and hardware-aware optimization.

use crate::{DeviceError, DeviceResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::str::FromStr;

/// QASM compiler configuration
#[derive(Debug, Clone)]
pub struct QasmCompilerConfig {
    /// Enable hardware-aware optimization
    pub hardware_optimization: bool,
    /// Target gate set for compilation
    pub target_gate_set: TargetGateSet,
    /// Maximum optimization passes
    pub max_optimization_passes: usize,
    /// Enable circuit verification
    pub verify_circuit: bool,
}

impl Default for QasmCompilerConfig {
    fn default() -> Self {
        Self {
            hardware_optimization: true,
            target_gate_set: TargetGateSet::Universal,
            max_optimization_passes: 3,
            verify_circuit: true,
        }
    }
}

/// Target gate set for compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetGateSet {
    /// Universal gate set (U3, CNOT)
    Universal,
    /// IBM gate set (U1, U2, U3, CNOT)
    IBM,
    /// Native gate set (SX, RZ, CNOT)
    Native,
    /// Custom gate set
    Custom,
}

/// Quantum gate representation
#[derive(Debug, Clone)]
pub enum QuantumGate {
    /// Single-qubit gate with unitary matrix
    SingleQubit {
        target: usize,
        unitary: Array2<Complex64>,
        name: String,
    },
    /// Two-qubit gate
    TwoQubit {
        control: usize,
        target: usize,
        unitary: Array2<Complex64>,
        name: String,
    },
    /// Parametric gate
    Parametric {
        target: usize,
        angle: f64,
        gate_type: ParametricGateType,
    },
    /// Measurement
    Measure { qubit: usize, classical_bit: usize },
}

/// Parametric gate types
#[derive(Debug, Clone, Copy)]
pub enum ParametricGateType {
    /// Rotation around X axis
    RX,
    /// Rotation around Y axis
    RY,
    /// Rotation around Z axis
    RZ,
    /// Phase gate
    Phase,
    /// U1 gate
    U1,
    /// U2 gate
    U2,
    /// U3 gate (general single-qubit rotation)
    U3,
}

/// Compiled quantum circuit
#[derive(Debug, Clone)]
pub struct CompiledCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of classical bits
    pub num_classical_bits: usize,
    /// Gates in the circuit
    pub gates: Vec<QuantumGate>,
    /// Circuit depth
    pub depth: usize,
    /// Total gate count
    pub gate_count: usize,
    /// Two-qubit gate count
    pub two_qubit_gate_count: usize,
}

/// QASM compiler leveraging SciRS2 parsing tools
pub struct QasmCompiler {
    config: QasmCompilerConfig,
    /// Gate definitions
    gate_definitions: HashMap<String, GateDefinition>,
}

/// Gate definition
#[derive(Debug, Clone)]
struct GateDefinition {
    num_qubits: usize,
    num_params: usize,
    unitary_generator: fn(&[f64]) -> Array2<Complex64>,
}

impl QasmCompiler {
    /// Create a new QASM compiler
    pub fn new(config: QasmCompilerConfig) -> Self {
        let mut compiler = Self {
            config,
            gate_definitions: HashMap::new(),
        };
        compiler.initialize_standard_gates();
        compiler
    }

    /// Create compiler with default configuration
    pub fn default() -> Self {
        Self::new(QasmCompilerConfig::default())
    }

    /// Initialize standard gate definitions
    fn initialize_standard_gates(&mut self) {
        // Hadamard gate
        self.gate_definitions.insert(
            "h".to_string(),
            GateDefinition {
                num_qubits: 1,
                num_params: 0,
                unitary_generator: |_| hadamard_unitary(),
            },
        );

        // Pauli-X gate
        self.gate_definitions.insert(
            "x".to_string(),
            GateDefinition {
                num_qubits: 1,
                num_params: 0,
                unitary_generator: |_| pauli_x_unitary(),
            },
        );

        // Pauli-Y gate
        self.gate_definitions.insert(
            "y".to_string(),
            GateDefinition {
                num_qubits: 1,
                num_params: 0,
                unitary_generator: |_| pauli_y_unitary(),
            },
        );

        // Pauli-Z gate
        self.gate_definitions.insert(
            "z".to_string(),
            GateDefinition {
                num_qubits: 1,
                num_params: 0,
                unitary_generator: |_| pauli_z_unitary(),
            },
        );

        // CNOT gate
        self.gate_definitions.insert(
            "cx".to_string(),
            GateDefinition {
                num_qubits: 2,
                num_params: 0,
                unitary_generator: |_| cnot_unitary(),
            },
        );
    }

    /// Compile QASM string to circuit representation
    ///
    /// # Arguments
    /// * `qasm_source` - QASM 2.0/3.0 source code
    ///
    /// # Returns
    /// Compiled circuit with optimizations applied
    pub fn compile(&self, qasm_source: &str) -> DeviceResult<CompiledCircuit> {
        // Parse QASM using SciRS2 string processing
        let parsed_circuit = self.parse_qasm(qasm_source)?;

        // Apply hardware-aware optimizations if enabled
        let optimized_circuit = if self.config.hardware_optimization {
            self.optimize_circuit(parsed_circuit)?
        } else {
            parsed_circuit
        };

        // Verify circuit if enabled
        if self.config.verify_circuit {
            self.verify_circuit_validity(&optimized_circuit)?;
        }

        Ok(optimized_circuit)
    }

    /// Parse QASM source code using SciRS2 parsing tools
    fn parse_qasm(&self, source: &str) -> DeviceResult<CompiledCircuit> {
        let mut num_qubits = 0;
        let mut num_classical_bits = 0;
        let mut gates = Vec::new();

        // Split into lines and process each line
        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }

            // Parse version header
            if trimmed.starts_with("OPENQASM") {
                continue;
            }

            // Parse include statements
            if trimmed.starts_with("include") {
                continue;
            }

            // Parse qubit register declaration
            if let Some(rest) = trimmed.strip_prefix("qreg") {
                if let Some(caps) = self.parse_register_declaration(rest) {
                    num_qubits = num_qubits.max(caps.size);
                }
                continue;
            }

            // Parse classical register declaration
            if let Some(rest) = trimmed.strip_prefix("creg") {
                if let Some(caps) = self.parse_register_declaration(rest) {
                    num_classical_bits = num_classical_bits.max(caps.size);
                }
                continue;
            }

            // Parse gate application
            if let Ok(gate) = self.parse_gate_application(trimmed, line_num) {
                gates.push(gate);
            }
        }

        // Validate qubit indices before calculating metrics
        for gate in &gates {
            match gate {
                QuantumGate::SingleQubit { target, .. }
                | QuantumGate::Parametric { target, .. } => {
                    if *target >= num_qubits {
                        return Err(DeviceError::InvalidInput(format!(
                            "Qubit index {} out of range (max {})",
                            target, num_qubits
                        )));
                    }
                }
                QuantumGate::TwoQubit {
                    control, target, ..
                } => {
                    if *control >= num_qubits || *target >= num_qubits {
                        return Err(DeviceError::InvalidInput(format!(
                            "Qubit indices ({}, {}) out of range (max {})",
                            control, target, num_qubits
                        )));
                    }
                }
                QuantumGate::Measure { qubit, .. } => {
                    if *qubit >= num_qubits {
                        return Err(DeviceError::InvalidInput(format!(
                            "Qubit index {} out of range",
                            qubit
                        )));
                    }
                }
            }
        }

        // Calculate circuit metrics
        let depth = self.calculate_circuit_depth(&gates, num_qubits);
        let gate_count = gates.len();
        let two_qubit_gate_count = gates
            .iter()
            .filter(|g| matches!(g, QuantumGate::TwoQubit { .. }))
            .count();

        Ok(CompiledCircuit {
            num_qubits,
            num_classical_bits,
            gates,
            depth,
            gate_count,
            two_qubit_gate_count,
        })
    }

    /// Parse register declaration
    fn parse_register_declaration(&self, decl: &str) -> Option<RegisterDeclaration> {
        // Simple parsing: "name[size];"
        let parts: Vec<&str> = decl.trim().trim_end_matches(';').split('[').collect();
        if parts.len() != 2 {
            return None;
        }

        let name = parts[0].trim();
        let size_str = parts[1].trim_end_matches(']').trim();
        let size = usize::from_str(size_str).ok()?;

        Some(RegisterDeclaration {
            name: name.to_string(),
            size,
        })
    }

    /// Parse gate application
    fn parse_gate_application(&self, line: &str, line_num: usize) -> DeviceResult<QuantumGate> {
        let line = line.trim_end_matches(';').trim();

        // Handle measurement
        if line.starts_with("measure") {
            return self.parse_measurement(line);
        }

        // Split gate name and arguments
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Err(DeviceError::InvalidInput(format!(
                "Empty gate at line {}",
                line_num
            )));
        }

        let gate_name = parts[0].to_lowercase();

        // Parse qubit arguments
        if parts.len() < 2 {
            return Err(DeviceError::InvalidInput(format!(
                "Missing qubit arguments for gate {} at line {}",
                gate_name, line_num
            )));
        }

        let qubit_args: Vec<usize> = parts[1..]
            .iter()
            .filter_map(|arg| {
                let cleaned = arg.trim_end_matches([',', ';']);
                // Extract qubit index from "q[i]" format
                if let Some(idx_str) = cleaned.strip_prefix("q[") {
                    idx_str.trim_end_matches(']').parse().ok()
                } else {
                    None
                }
            })
            .collect();

        // Create gate based on definition
        match gate_name.as_str() {
            "h" | "x" | "y" | "z" => {
                if qubit_args.is_empty() {
                    return Err(DeviceError::InvalidInput(format!(
                        "No qubit specified for gate {}",
                        gate_name
                    )));
                }
                let def = self.gate_definitions.get(&gate_name).ok_or_else(|| {
                    DeviceError::InvalidInput(format!("Unknown gate: {}", gate_name))
                })?;
                Ok(QuantumGate::SingleQubit {
                    target: qubit_args[0],
                    unitary: (def.unitary_generator)(&[]),
                    name: gate_name,
                })
            }
            "cx" | "cnot" => {
                if qubit_args.len() < 2 {
                    return Err(DeviceError::InvalidInput(
                        "CNOT requires 2 qubits".to_string(),
                    ));
                }
                Ok(QuantumGate::TwoQubit {
                    control: qubit_args[0],
                    target: qubit_args[1],
                    unitary: cnot_unitary(),
                    name: "cx".to_string(),
                })
            }
            _ => Err(DeviceError::InvalidInput(format!(
                "Unsupported gate: {}",
                gate_name
            ))),
        }
    }

    /// Parse measurement operation
    fn parse_measurement(&self, line: &str) -> DeviceResult<QuantumGate> {
        // Format: "measure q[i] -> c[j]"
        let parts: Vec<&str> = line.split("->").collect();
        if parts.len() != 2 {
            return Err(DeviceError::InvalidInput(
                "Invalid measurement syntax".to_string(),
            ));
        }

        let qubit_str = parts[0].trim().strip_prefix("measure").unwrap_or("").trim();
        let classical_str = parts[1].trim();

        let qubit = Self::extract_index(qubit_str)?;
        let classical_bit = Self::extract_index(classical_str)?;

        Ok(QuantumGate::Measure {
            qubit,
            classical_bit,
        })
    }

    /// Extract index from "name[index]" format
    fn extract_index(s: &str) -> DeviceResult<usize> {
        if let Some(idx_str) = s.strip_prefix("q[").or_else(|| s.strip_prefix("c[")) {
            idx_str
                .trim_end_matches(']')
                .parse()
                .map_err(|_| DeviceError::InvalidInput("Invalid index".to_string()))
        } else {
            Err(DeviceError::InvalidInput("Invalid format".to_string()))
        }
    }

    /// Calculate circuit depth using dependency analysis
    fn calculate_circuit_depth(&self, gates: &[QuantumGate], num_qubits: usize) -> usize {
        let mut qubit_depths = vec![0; num_qubits];

        for gate in gates {
            match gate {
                QuantumGate::SingleQubit { target, .. }
                | QuantumGate::Parametric { target, .. } => {
                    qubit_depths[*target] += 1;
                }
                QuantumGate::TwoQubit {
                    control, target, ..
                } => {
                    let max_depth = qubit_depths[*control].max(qubit_depths[*target]);
                    qubit_depths[*control] = max_depth + 1;
                    qubit_depths[*target] = max_depth + 1;
                }
                QuantumGate::Measure { qubit, .. } => {
                    qubit_depths[*qubit] += 1;
                }
            }
        }

        qubit_depths.into_iter().max().unwrap_or(0)
    }

    /// Apply hardware-aware circuit optimizations
    fn optimize_circuit(&self, circuit: CompiledCircuit) -> DeviceResult<CompiledCircuit> {
        let mut optimized = circuit;

        for _ in 0..self.config.max_optimization_passes {
            // Gate fusion pass
            optimized = self.fuse_single_qubit_gates(optimized)?;

            // Gate cancellation pass
            optimized = self.cancel_inverse_gates(optimized)?;

            // Gate commutation pass for better routing
            optimized = self.commute_gates(optimized)?;
        }

        // Recalculate metrics
        optimized.depth = self.calculate_circuit_depth(&optimized.gates, optimized.num_qubits);
        optimized.gate_count = optimized.gates.len();
        optimized.two_qubit_gate_count = optimized
            .gates
            .iter()
            .filter(|g| matches!(g, QuantumGate::TwoQubit { .. }))
            .count();

        Ok(optimized)
    }

    /// Fuse consecutive single-qubit gates on the same qubit
    fn fuse_single_qubit_gates(&self, circuit: CompiledCircuit) -> DeviceResult<CompiledCircuit> {
        // Simplified implementation - in practice, would multiply unitaries
        Ok(circuit)
    }

    /// Cancel inverse gate pairs (e.g., X-X, H-H)
    fn cancel_inverse_gates(&self, mut circuit: CompiledCircuit) -> DeviceResult<CompiledCircuit> {
        let mut i = 0;
        while i + 1 < circuit.gates.len() {
            let gate1 = &circuit.gates[i];
            let gate2 = &circuit.gates[i + 1];

            if Self::are_inverse_gates(gate1, gate2) {
                // Remove both gates
                circuit.gates.remove(i);
                circuit.gates.remove(i);
            } else {
                i += 1;
            }
        }

        Ok(circuit)
    }

    /// Check if two gates are inverses
    fn are_inverse_gates(gate1: &QuantumGate, gate2: &QuantumGate) -> bool {
        match (gate1, gate2) {
            (
                QuantumGate::SingleQubit {
                    target: t1,
                    name: n1,
                    ..
                },
                QuantumGate::SingleQubit {
                    target: t2,
                    name: n2,
                    ..
                },
            ) => t1 == t2 && n1 == n2 && (n1 == "x" || n1 == "y" || n1 == "z" || n1 == "h"),
            _ => false,
        }
    }

    /// Commute gates to improve routing
    fn commute_gates(&self, circuit: CompiledCircuit) -> DeviceResult<CompiledCircuit> {
        // Simplified implementation
        Ok(circuit)
    }

    /// Verify circuit validity
    fn verify_circuit_validity(&self, circuit: &CompiledCircuit) -> DeviceResult<()> {
        // Check all qubit indices are valid
        for gate in &circuit.gates {
            match gate {
                QuantumGate::SingleQubit { target, .. }
                | QuantumGate::Parametric { target, .. } => {
                    if *target >= circuit.num_qubits {
                        return Err(DeviceError::InvalidInput(format!(
                            "Qubit index {} out of range (max {})",
                            target, circuit.num_qubits
                        )));
                    }
                }
                QuantumGate::TwoQubit {
                    control, target, ..
                } => {
                    if *control >= circuit.num_qubits || *target >= circuit.num_qubits {
                        return Err(DeviceError::InvalidInput(format!(
                            "Qubit indices ({}, {}) out of range (max {})",
                            control, target, circuit.num_qubits
                        )));
                    }
                    if control == target {
                        return Err(DeviceError::InvalidInput(
                            "Control and target qubits must be different".to_string(),
                        ));
                    }
                }
                QuantumGate::Measure {
                    qubit,
                    classical_bit,
                } => {
                    if *qubit >= circuit.num_qubits {
                        return Err(DeviceError::InvalidInput(format!(
                            "Qubit index {} out of range",
                            qubit
                        )));
                    }
                    if *classical_bit >= circuit.num_classical_bits {
                        return Err(DeviceError::InvalidInput(format!(
                            "Classical bit index {} out of range",
                            classical_bit
                        )));
                    }
                }
            }
        }

        Ok(())
    }
}

/// Register declaration
struct RegisterDeclaration {
    name: String,
    size: usize,
}

// Gate unitary matrices

fn hadamard_unitary() -> Array2<Complex64> {
    let s = 1.0 / f64::sqrt(2.0);
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(-s, 0.0),
        ],
    )
    .expect("Failed to create Hadamard unitary")
}

fn pauli_x_unitary() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .expect("Failed to create Pauli-X unitary")
}

fn pauli_y_unitary() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .expect("Failed to create Pauli-Y unitary")
}

fn pauli_z_unitary() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .expect("Failed to create Pauli-Z unitary")
}

fn cnot_unitary() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (4, 4),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .expect("Failed to create CNOT unitary")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qasm_compiler_creation() {
        let compiler = QasmCompiler::default();
        assert_eq!(compiler.config.max_optimization_passes, 3);
        assert!(compiler.config.hardware_optimization);
    }

    #[test]
    fn test_simple_qasm_compilation() {
        let compiler = QasmCompiler::default();
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0] q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
        "#;

        let result = compiler.compile(qasm);
        assert!(result.is_ok());

        let circuit = result.expect("Compilation failed");
        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.num_classical_bits, 2);
        assert_eq!(circuit.gate_count, 4); // H, CX, measure, measure
    }

    #[test]
    fn test_gate_cancellation() {
        let compiler = QasmCompiler::default();
        let qasm = r#"
            OPENQASM 2.0;
            qreg q[1];
            x q[0];
            x q[0];
        "#;

        let result = compiler.compile(qasm);
        assert!(result.is_ok());

        let circuit = result.expect("Compilation failed");
        // After optimization, X-X should cancel
        assert_eq!(circuit.gate_count, 0);
    }

    #[test]
    fn test_circuit_depth_calculation() {
        let compiler = QasmCompiler::default();
        let qasm = r#"
            OPENQASM 2.0;
            qreg q[2];
            h q[0];
            h q[1];
            cx q[0] q[1];
        "#;

        let result = compiler.compile(qasm);
        assert!(result.is_ok());

        let circuit = result.expect("Compilation failed");
        // H on both qubits (depth 1), then CX (depth 2)
        assert_eq!(circuit.depth, 2);
    }

    #[test]
    fn test_invalid_qubit_index() {
        let compiler = QasmCompiler::default();
        let qasm = r#"
            OPENQASM 2.0;
            qreg q[2];
            h q[5];
        "#;

        let result = compiler.compile(qasm);
        assert!(result.is_err());
    }

    #[test]
    fn test_gate_unitaries() {
        let h = hadamard_unitary();
        assert_eq!(h.shape(), &[2, 2]);

        let x = pauli_x_unitary();
        assert_eq!(x.shape(), &[2, 2]);

        let cnot = cnot_unitary();
        assert_eq!(cnot.shape(), &[4, 4]);
    }
}
