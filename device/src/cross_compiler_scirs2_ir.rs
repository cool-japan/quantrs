//! Cross-Compilation with SciRS2 IR Tools
//!
//! This module provides quantum circuit cross-compilation using SciRS2's intermediate
//! representation (IR) tools for multi-backend optimization and code generation.

use crate::{DeviceError, DeviceResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::collections::HashMap;

/// Target backend for cross-compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetBackend {
    /// IBM Quantum (OpenQASM 2.0)
    IBMQuantum,
    /// AWS Braket
    AWSBraket,
    /// Azure Quantum
    AzureQuantum,
    /// Generic QASM
    GenericQASM,
    /// Custom backend
    Custom,
}

/// Optimization level for IR compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization
    O0,
    /// Basic optimization
    O1,
    /// Moderate optimization
    O2,
    /// Aggressive optimization
    O3,
}

/// Cross-compiler configuration
#[derive(Debug, Clone)]
pub struct CrossCompilerConfig {
    /// Target backend
    pub target: TargetBackend,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Enable gate fusion
    pub enable_gate_fusion: bool,
    /// Enable commutation-based optimization
    pub enable_commutation_opt: bool,
    /// Enable backend-specific optimizations
    pub enable_backend_opt: bool,
    /// Maximum circuit depth (None for unlimited)
    pub max_circuit_depth: Option<usize>,
}

impl Default for CrossCompilerConfig {
    fn default() -> Self {
        Self {
            target: TargetBackend::GenericQASM,
            optimization_level: OptimizationLevel::O2,
            enable_gate_fusion: true,
            enable_commutation_opt: true,
            enable_backend_opt: true,
            max_circuit_depth: None,
        }
    }
}

/// Quantum circuit intermediate representation
#[derive(Debug, Clone)]
pub struct QuantumIR {
    /// Number of qubits
    pub num_qubits: usize,
    /// IR instructions
    pub instructions: Vec<IRInstruction>,
    /// Metadata
    pub metadata: IRMetadata,
}

/// IR instruction representing quantum operations
#[derive(Debug, Clone)]
pub enum IRInstruction {
    /// Single-qubit gate
    SingleQubitGate {
        gate_type: SingleQubitGateType,
        target: usize,
        parameters: Vec<f64>,
    },
    /// Two-qubit gate
    TwoQubitGate {
        gate_type: TwoQubitGateType,
        control: usize,
        target: usize,
        parameters: Vec<f64>,
    },
    /// Measurement
    Measure { qubit: usize, classical_bit: usize },
    /// Barrier (prevents optimization across it)
    Barrier { qubits: Vec<usize> },
    /// Custom gate
    Custom {
        name: String,
        qubits: Vec<usize>,
        parameters: Vec<f64>,
    },
}

/// Single-qubit gate types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingleQubitGateType {
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    Phase,
    T,
    TDagger,
    S,
    SDagger,
    Rx,
    Ry,
    Rz,
    U3,
}

/// Two-qubit gate types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoQubitGateType {
    CNOT,
    CZ,
    SWAP,
    Controlled,
}

/// IR metadata
#[derive(Debug, Clone)]
pub struct IRMetadata {
    /// Circuit name
    pub name: String,
    /// Creation timestamp
    pub created: std::time::SystemTime,
    /// Source backend (if converted)
    pub source_backend: Option<TargetBackend>,
    /// Optimization passes applied
    pub optimization_passes: Vec<String>,
}

impl Default for IRMetadata {
    fn default() -> Self {
        Self {
            name: "quantum_circuit".to_string(),
            created: std::time::SystemTime::now(),
            source_backend: None,
            optimization_passes: Vec::new(),
        }
    }
}

/// Compilation result
#[derive(Debug, Clone)]
pub struct CompilationResult {
    /// Compiled code for target backend
    pub code: String,
    /// Optimized IR
    pub optimized_ir: QuantumIR,
    /// Compilation statistics
    pub statistics: CompilationStatistics,
}

/// Compilation statistics
#[derive(Debug, Clone)]
pub struct CompilationStatistics {
    /// Original instruction count
    pub original_instruction_count: usize,
    /// Optimized instruction count
    pub optimized_instruction_count: usize,
    /// Circuit depth before optimization
    pub original_depth: usize,
    /// Circuit depth after optimization
    pub optimized_depth: usize,
    /// Number of optimization passes applied
    pub num_optimization_passes: usize,
    /// Compilation time (milliseconds)
    pub compilation_time_ms: u128,
}

/// Cross-compiler using SciRS2 IR tools
pub struct CrossCompiler {
    config: CrossCompilerConfig,
}

impl CrossCompiler {
    /// Create a new cross-compiler
    pub fn new(config: CrossCompilerConfig) -> Self {
        Self { config }
    }

    /// Create compiler with default configuration
    pub fn default() -> Self {
        Self::new(CrossCompilerConfig::default())
    }

    /// Compile quantum IR to target backend
    ///
    /// # Arguments
    /// * `ir` - Quantum circuit in IR format
    ///
    /// # Returns
    /// Compilation result with backend-specific code
    pub fn compile(&self, ir: QuantumIR) -> DeviceResult<CompilationResult> {
        let start_time = std::time::Instant::now();

        let original_count = ir.instructions.len();
        let original_depth = self.compute_circuit_depth(&ir);

        // Apply optimization passes based on configuration
        let mut optimized_ir = ir.clone();
        let mut num_passes = 0;

        if self.config.optimization_level != OptimizationLevel::O0 {
            // Gate fusion pass
            if self.config.enable_gate_fusion {
                optimized_ir = self.apply_gate_fusion(optimized_ir)?;
                num_passes += 1;
            }

            // Commutation-based optimization
            if self.config.enable_commutation_opt {
                optimized_ir = self.apply_commutation_optimization(optimized_ir)?;
                num_passes += 1;
            }

            // Backend-specific optimizations
            if self.config.enable_backend_opt {
                optimized_ir = self.apply_backend_optimization(optimized_ir)?;
                num_passes += 1;
            }

            // Remove identity gates
            optimized_ir = self.remove_identity_gates(optimized_ir)?;
            num_passes += 1;
        }

        let optimized_count = optimized_ir.instructions.len();
        let optimized_depth = self.compute_circuit_depth(&optimized_ir);

        // Generate code for target backend
        let code = self.generate_backend_code(&optimized_ir)?;

        let compilation_time = start_time.elapsed().as_millis();

        Ok(CompilationResult {
            code,
            optimized_ir,
            statistics: CompilationStatistics {
                original_instruction_count: original_count,
                optimized_instruction_count: optimized_count,
                original_depth,
                optimized_depth,
                num_optimization_passes: num_passes,
                compilation_time_ms: compilation_time,
            },
        })
    }

    /// Compute circuit depth
    fn compute_circuit_depth(&self, ir: &QuantumIR) -> usize {
        let mut qubit_depths = vec![0; ir.num_qubits];

        for instr in &ir.instructions {
            match instr {
                IRInstruction::SingleQubitGate { target, .. } => {
                    qubit_depths[*target] += 1;
                }
                IRInstruction::TwoQubitGate {
                    control, target, ..
                } => {
                    let max_depth = qubit_depths[*control].max(qubit_depths[*target]);
                    qubit_depths[*control] = max_depth + 1;
                    qubit_depths[*target] = max_depth + 1;
                }
                IRInstruction::Measure { qubit, .. } => {
                    qubit_depths[*qubit] += 1;
                }
                IRInstruction::Barrier { qubits } => {
                    if !qubits.is_empty() {
                        let max_depth = qubits.iter().map(|&q| qubit_depths[q]).max().unwrap_or(0);
                        for &q in qubits {
                            qubit_depths[q] = max_depth + 1;
                        }
                    }
                }
                IRInstruction::Custom { qubits, .. } => {
                    if !qubits.is_empty() {
                        let max_depth = qubits.iter().map(|&q| qubit_depths[q]).max().unwrap_or(0);
                        for &q in qubits {
                            qubit_depths[q] = max_depth + 1;
                        }
                    }
                }
            }
        }

        qubit_depths.into_iter().max().unwrap_or(0)
    }

    /// Apply gate fusion optimization
    fn apply_gate_fusion(&self, mut ir: QuantumIR) -> DeviceResult<QuantumIR> {
        let mut optimized_instructions = Vec::new();
        let mut i = 0;

        while i < ir.instructions.len() {
            // Look for consecutive rotation gates on the same qubit
            if i + 1 < ir.instructions.len() {
                match (&ir.instructions[i], &ir.instructions[i + 1]) {
                    (
                        IRInstruction::SingleQubitGate {
                            gate_type: SingleQubitGateType::Rz,
                            target: t1,
                            parameters: p1,
                        },
                        IRInstruction::SingleQubitGate {
                            gate_type: SingleQubitGateType::Rz,
                            target: t2,
                            parameters: p2,
                        },
                    ) if t1 == t2 && p1.len() == 1 && p2.len() == 1 => {
                        // Fuse consecutive Rz gates
                        optimized_instructions.push(IRInstruction::SingleQubitGate {
                            gate_type: SingleQubitGateType::Rz,
                            target: *t1,
                            parameters: vec![p1[0] + p2[0]],
                        });
                        i += 2;
                        continue;
                    }
                    _ => {}
                }
            }

            optimized_instructions.push(ir.instructions[i].clone());
            i += 1;
        }

        ir.instructions = optimized_instructions;
        ir.metadata
            .optimization_passes
            .push("gate_fusion".to_string());
        Ok(ir)
    }

    /// Apply commutation-based optimization
    fn apply_commutation_optimization(&self, mut ir: QuantumIR) -> DeviceResult<QuantumIR> {
        // Simplified commutation optimization: move measurements to the end
        let mut measurements = Vec::new();
        let mut other_instructions = Vec::new();

        for instr in ir.instructions {
            match instr {
                IRInstruction::Measure { .. } => measurements.push(instr),
                _ => other_instructions.push(instr),
            }
        }

        other_instructions.extend(measurements);
        ir.instructions = other_instructions;
        ir.metadata
            .optimization_passes
            .push("commutation_opt".to_string());
        Ok(ir)
    }

    /// Apply backend-specific optimizations
    fn apply_backend_optimization(&self, mut ir: QuantumIR) -> DeviceResult<QuantumIR> {
        match self.config.target {
            TargetBackend::IBMQuantum => {
                // IBM prefers U3 gates and CX
                ir = self.decompose_to_ibm_basis(ir)?;
            }
            TargetBackend::AWSBraket => {
                // AWS Braket supports various gate sets
                // Keep standard gates
            }
            TargetBackend::AzureQuantum => {
                // Azure supports standard gates
                // Keep standard gates
            }
            _ => {}
        }

        ir.metadata
            .optimization_passes
            .push("backend_specific".to_string());
        Ok(ir)
    }

    /// Decompose gates to IBM basis (U3, CX)
    fn decompose_to_ibm_basis(&self, mut ir: QuantumIR) -> DeviceResult<QuantumIR> {
        let mut decomposed = Vec::new();

        for instr in ir.instructions {
            match instr {
                IRInstruction::SingleQubitGate {
                    gate_type: SingleQubitGateType::Hadamard,
                    target,
                    ..
                } => {
                    // H = U3(π/2, 0, π)
                    decomposed.push(IRInstruction::SingleQubitGate {
                        gate_type: SingleQubitGateType::U3,
                        target,
                        parameters: vec![std::f64::consts::PI / 2.0, 0.0, std::f64::consts::PI],
                    });
                }
                IRInstruction::TwoQubitGate {
                    gate_type: TwoQubitGateType::CNOT,
                    control,
                    target,
                    ..
                } => {
                    // CNOT is native, keep as is
                    decomposed.push(IRInstruction::TwoQubitGate {
                        gate_type: TwoQubitGateType::CNOT,
                        control,
                        target,
                        parameters: vec![],
                    });
                }
                other => decomposed.push(other),
            }
        }

        ir.instructions = decomposed;
        Ok(ir)
    }

    /// Remove identity gates and no-ops
    fn remove_identity_gates(&self, mut ir: QuantumIR) -> DeviceResult<QuantumIR> {
        ir.instructions.retain(|instr| {
            !matches!(
                instr,
                IRInstruction::SingleQubitGate {
                    gate_type: SingleQubitGateType::Rz,
                    parameters,
                    ..
                } if parameters.len() == 1 && parameters[0].abs() < 1e-10
            )
        });

        ir.metadata
            .optimization_passes
            .push("identity_removal".to_string());
        Ok(ir)
    }

    /// Generate backend-specific code
    fn generate_backend_code(&self, ir: &QuantumIR) -> DeviceResult<String> {
        match self.config.target {
            TargetBackend::IBMQuantum | TargetBackend::GenericQASM => self.generate_qasm_code(ir),
            TargetBackend::AWSBraket => self.generate_braket_code(ir),
            TargetBackend::AzureQuantum => self.generate_azure_code(ir),
            TargetBackend::Custom => {
                Ok("// Custom backend code generation not implemented\n".to_string())
            }
        }
    }

    /// Generate OpenQASM 2.0 code
    fn generate_qasm_code(&self, ir: &QuantumIR) -> DeviceResult<String> {
        let mut code = String::new();
        code.push_str("OPENQASM 2.0;\n");
        code.push_str("include \"qelib1.inc\";\n");
        code.push_str(&format!("qreg q[{}];\n", ir.num_qubits));
        code.push_str(&format!("creg c[{}];\n", ir.num_qubits));

        for instr in &ir.instructions {
            match instr {
                IRInstruction::SingleQubitGate {
                    gate_type,
                    target,
                    parameters,
                } => {
                    let gate_str = match gate_type {
                        SingleQubitGateType::Hadamard => format!("h q[{}];", target),
                        SingleQubitGateType::PauliX => format!("x q[{}];", target),
                        SingleQubitGateType::PauliY => format!("y q[{}];", target),
                        SingleQubitGateType::PauliZ => format!("z q[{}];", target),
                        SingleQubitGateType::T => format!("t q[{}];", target),
                        SingleQubitGateType::TDagger => format!("tdg q[{}];", target),
                        SingleQubitGateType::S => format!("s q[{}];", target),
                        SingleQubitGateType::SDagger => format!("sdg q[{}];", target),
                        SingleQubitGateType::Rx if parameters.len() == 1 => {
                            format!("rx({}) q[{}];", parameters[0], target)
                        }
                        SingleQubitGateType::Ry if parameters.len() == 1 => {
                            format!("ry({}) q[{}];", parameters[0], target)
                        }
                        SingleQubitGateType::Rz if parameters.len() == 1 => {
                            format!("rz({}) q[{}];", parameters[0], target)
                        }
                        SingleQubitGateType::U3 if parameters.len() == 3 => {
                            format!(
                                "u3({},{},{}) q[{}];",
                                parameters[0], parameters[1], parameters[2], target
                            )
                        }
                        _ => format!("// Unknown gate on q[{}]", target),
                    };
                    code.push_str(&gate_str);
                    code.push('\n');
                }
                IRInstruction::TwoQubitGate {
                    gate_type,
                    control,
                    target,
                    ..
                } => {
                    let gate_str = match gate_type {
                        TwoQubitGateType::CNOT => format!("cx q[{}],q[{}];", control, target),
                        TwoQubitGateType::CZ => format!("cz q[{}],q[{}];", control, target),
                        TwoQubitGateType::SWAP => format!("swap q[{}],q[{}];", control, target),
                        _ => "// Unknown two-qubit gate".to_string(),
                    };
                    code.push_str(&gate_str);
                    code.push('\n');
                }
                IRInstruction::Measure {
                    qubit,
                    classical_bit,
                } => {
                    code.push_str(&format!("measure q[{}] -> c[{}];\n", qubit, classical_bit));
                }
                IRInstruction::Barrier { qubits } => {
                    if qubits.is_empty() {
                        code.push_str("barrier q;\n");
                    } else {
                        let qubit_list = qubits
                            .iter()
                            .map(|q| format!("q[{}]", q))
                            .collect::<Vec<_>>()
                            .join(",");
                        code.push_str(&format!("barrier {};\n", qubit_list));
                    }
                }
                IRInstruction::Custom {
                    name,
                    qubits,
                    parameters,
                } => {
                    let qubit_list = qubits
                        .iter()
                        .map(|q| format!("q[{}]", q))
                        .collect::<Vec<_>>()
                        .join(",");
                    if parameters.is_empty() {
                        code.push_str(&format!("{} {};\n", name, qubit_list));
                    } else {
                        let param_list = parameters
                            .iter()
                            .map(|p| p.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        code.push_str(&format!("{}({}) {};\n", name, param_list, qubit_list));
                    }
                }
            }
        }

        Ok(code)
    }

    /// Generate AWS Braket code (OpenQASM-like)
    fn generate_braket_code(&self, ir: &QuantumIR) -> DeviceResult<String> {
        // AWS Braket uses OpenQASM with some extensions
        self.generate_qasm_code(ir)
    }

    /// Generate Azure Quantum code
    fn generate_azure_code(&self, ir: &QuantumIR) -> DeviceResult<String> {
        // Azure Quantum supports OpenQASM
        self.generate_qasm_code(ir)
    }

    /// Create IR from gate sequence
    pub fn create_ir_from_gates(&self, num_qubits: usize, gates: Vec<IRInstruction>) -> QuantumIR {
        QuantumIR {
            num_qubits,
            instructions: gates,
            metadata: IRMetadata::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_creation() {
        let config = CrossCompilerConfig::default();
        let compiler = CrossCompiler::new(config);
        assert_eq!(compiler.config.target, TargetBackend::GenericQASM);
    }

    #[test]
    fn test_ir_creation() {
        let compiler = CrossCompiler::default();
        let gates = vec![
            IRInstruction::SingleQubitGate {
                gate_type: SingleQubitGateType::Hadamard,
                target: 0,
                parameters: vec![],
            },
            IRInstruction::TwoQubitGate {
                gate_type: TwoQubitGateType::CNOT,
                control: 0,
                target: 1,
                parameters: vec![],
            },
        ];

        let ir = compiler.create_ir_from_gates(2, gates);
        assert_eq!(ir.num_qubits, 2);
        assert_eq!(ir.instructions.len(), 2);
    }

    #[test]
    fn test_circuit_depth_calculation() {
        let compiler = CrossCompiler::default();
        let gates = vec![
            IRInstruction::SingleQubitGate {
                gate_type: SingleQubitGateType::Hadamard,
                target: 0,
                parameters: vec![],
            },
            IRInstruction::SingleQubitGate {
                gate_type: SingleQubitGateType::Hadamard,
                target: 1,
                parameters: vec![],
            },
            IRInstruction::TwoQubitGate {
                gate_type: TwoQubitGateType::CNOT,
                control: 0,
                target: 1,
                parameters: vec![],
            },
        ];

        let ir = compiler.create_ir_from_gates(2, gates);
        let depth = compiler.compute_circuit_depth(&ir);
        assert_eq!(depth, 2); // H on q0, H on q1 (parallel), then CNOT
    }

    #[test]
    fn test_qasm_code_generation() {
        let config = CrossCompilerConfig {
            target: TargetBackend::GenericQASM,
            optimization_level: OptimizationLevel::O0,
            ..Default::default()
        };
        let compiler = CrossCompiler::new(config);

        let gates = vec![
            IRInstruction::SingleQubitGate {
                gate_type: SingleQubitGateType::Hadamard,
                target: 0,
                parameters: vec![],
            },
            IRInstruction::TwoQubitGate {
                gate_type: TwoQubitGateType::CNOT,
                control: 0,
                target: 1,
                parameters: vec![],
            },
            IRInstruction::Measure {
                qubit: 0,
                classical_bit: 0,
            },
        ];

        let ir = compiler.create_ir_from_gates(2, gates);
        let result = compiler.compile(ir);

        assert!(result.is_ok());
        let result = result.expect("Compilation failed");
        assert!(result.code.contains("OPENQASM 2.0"));
        assert!(result.code.contains("h q[0]"));
        assert!(result.code.contains("cx q[0],q[1]"));
        assert!(result.code.contains("measure q[0] -> c[0]"));
    }

    #[test]
    fn test_gate_fusion_optimization() {
        let config = CrossCompilerConfig {
            optimization_level: OptimizationLevel::O2,
            enable_gate_fusion: true,
            ..Default::default()
        };
        let compiler = CrossCompiler::new(config);

        let gates = vec![
            IRInstruction::SingleQubitGate {
                gate_type: SingleQubitGateType::Rz,
                target: 0,
                parameters: vec![0.5],
            },
            IRInstruction::SingleQubitGate {
                gate_type: SingleQubitGateType::Rz,
                target: 0,
                parameters: vec![0.3],
            },
        ];

        let ir = compiler.create_ir_from_gates(1, gates);
        let result = compiler.compile(ir);

        assert!(result.is_ok());
        let result = result.expect("Compilation failed");

        // Should have fused the two Rz gates into one
        assert_eq!(result.statistics.optimized_instruction_count, 1);
    }

    #[test]
    fn test_identity_gate_removal() {
        let config = CrossCompilerConfig {
            optimization_level: OptimizationLevel::O2,
            ..Default::default()
        };
        let compiler = CrossCompiler::new(config);

        let gates = vec![
            IRInstruction::SingleQubitGate {
                gate_type: SingleQubitGateType::Hadamard,
                target: 0,
                parameters: vec![],
            },
            IRInstruction::SingleQubitGate {
                gate_type: SingleQubitGateType::Rz,
                target: 0,
                parameters: vec![0.0], // Identity (no rotation)
            },
            IRInstruction::Measure {
                qubit: 0,
                classical_bit: 0,
            },
        ];

        let ir = compiler.create_ir_from_gates(1, gates);
        let result = compiler.compile(ir);

        assert!(result.is_ok());
        let result = result.expect("Compilation failed");

        // Should have removed the identity Rz gate
        assert_eq!(result.statistics.optimized_instruction_count, 2);
    }

    #[test]
    fn test_compilation_statistics() {
        let config = CrossCompilerConfig {
            optimization_level: OptimizationLevel::O2,
            ..Default::default()
        };
        let compiler = CrossCompiler::new(config);

        let gates = vec![
            IRInstruction::SingleQubitGate {
                gate_type: SingleQubitGateType::Hadamard,
                target: 0,
                parameters: vec![],
            },
            IRInstruction::SingleQubitGate {
                gate_type: SingleQubitGateType::Rz,
                target: 0,
                parameters: vec![0.0],
            },
        ];

        let ir = compiler.create_ir_from_gates(1, gates);
        let result = compiler.compile(ir);

        assert!(result.is_ok());
        let result = result.expect("Compilation failed");

        assert_eq!(result.statistics.original_instruction_count, 2);
        assert!(result.statistics.num_optimization_passes > 0);
        // Compilation time is u64, always non-negative
    }

    #[test]
    fn test_different_backends() {
        let backends = vec![
            TargetBackend::IBMQuantum,
            TargetBackend::AWSBraket,
            TargetBackend::AzureQuantum,
        ];

        for backend in backends {
            let config = CrossCompilerConfig {
                target: backend,
                optimization_level: OptimizationLevel::O0,
                ..Default::default()
            };
            let compiler = CrossCompiler::new(config);

            let gates = vec![IRInstruction::SingleQubitGate {
                gate_type: SingleQubitGateType::Hadamard,
                target: 0,
                parameters: vec![],
            }];

            let ir = compiler.create_ir_from_gates(1, gates);
            let result = compiler.compile(ir);

            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_barrier_instruction() {
        let compiler = CrossCompiler::default();

        let gates = vec![
            IRInstruction::SingleQubitGate {
                gate_type: SingleQubitGateType::Hadamard,
                target: 0,
                parameters: vec![],
            },
            IRInstruction::Barrier { qubits: vec![0, 1] },
            IRInstruction::TwoQubitGate {
                gate_type: TwoQubitGateType::CNOT,
                control: 0,
                target: 1,
                parameters: vec![],
            },
        ];

        let ir = compiler.create_ir_from_gates(2, gates);
        let result = compiler.compile(ir);

        assert!(result.is_ok());
        let result = result.expect("Compilation failed");
        assert!(result.code.contains("barrier"));
    }

    #[test]
    fn test_optimization_levels() {
        let levels = vec![
            OptimizationLevel::O0,
            OptimizationLevel::O1,
            OptimizationLevel::O2,
            OptimizationLevel::O3,
        ];

        for level in levels {
            let config = CrossCompilerConfig {
                optimization_level: level,
                ..Default::default()
            };
            let compiler = CrossCompiler::new(config);

            let gates = vec![IRInstruction::SingleQubitGate {
                gate_type: SingleQubitGateType::Hadamard,
                target: 0,
                parameters: vec![],
            }];

            let ir = compiler.create_ir_from_gates(1, gates);
            let result = compiler.compile(ir);

            assert!(result.is_ok());
        }
    }
}
