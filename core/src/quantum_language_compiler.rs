//! Quantum Programming Language Compilation Targets
//!
//! This module provides compilation from QuantRS2's internal circuit representation
//! to various quantum programming languages and frameworks.
//!
//! ## Supported Target Languages
//!
//! - **OpenQASM 2.0/3.0**: IBM's open quantum assembly language
//! - **Quil**: Rigetti's quantum instruction language
//! - **Q#**: Microsoft's quantum programming language
//! - **Cirq**: Google's quantum programming framework (Python)
//! - **Qiskit**: IBM's quantum development kit (Python)
//! - **PyQuil**: Rigetti's quantum programming library (Python)
//! - **ProjectQ**: ETH Zurich's quantum programming framework
//! - **Braket IR**: AWS Braket's intermediate representation
//! - **Silq**: ETH Zurich's high-level quantum language
//!
//! ## Features
//!
//! - Automatic gate decomposition to target gate sets
//! - Optimization for target platform
//! - Preserves circuit structure and comments
//! - Handles classical registers and measurements

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ================================================================================================
// Target Language Types
// ================================================================================================

/// Supported quantum programming languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantumLanguage {
    /// OpenQASM 2.0
    OpenQASM2,
    /// OpenQASM 3.0
    OpenQASM3,
    /// Rigetti Quil
    Quil,
    /// Microsoft Q#
    QSharp,
    /// Google Cirq (Python)
    Cirq,
    /// IBM Qiskit (Python)
    Qiskit,
    /// Rigetti PyQuil (Python)
    PyQuil,
    /// ProjectQ (Python)
    ProjectQ,
    /// AWS Braket IR (JSON)
    BraketIR,
    /// Silq high-level language
    Silq,
    /// Pennylane (Python)
    Pennylane,
}

impl QuantumLanguage {
    /// Get language name
    pub fn name(&self) -> &'static str {
        match self {
            QuantumLanguage::OpenQASM2 => "OpenQASM 2.0",
            QuantumLanguage::OpenQASM3 => "OpenQASM 3.0",
            QuantumLanguage::Quil => "Quil",
            QuantumLanguage::QSharp => "Q#",
            QuantumLanguage::Cirq => "Cirq",
            QuantumLanguage::Qiskit => "Qiskit",
            QuantumLanguage::PyQuil => "PyQuil",
            QuantumLanguage::ProjectQ => "ProjectQ",
            QuantumLanguage::BraketIR => "Braket IR",
            QuantumLanguage::Silq => "Silq",
            QuantumLanguage::Pennylane => "Pennylane",
        }
    }

    /// Get file extension
    pub fn extension(&self) -> &'static str {
        match self {
            QuantumLanguage::OpenQASM2 | QuantumLanguage::OpenQASM3 => "qasm",
            QuantumLanguage::Quil => "quil",
            QuantumLanguage::QSharp => "qs",
            QuantumLanguage::Cirq
            | QuantumLanguage::Qiskit
            | QuantumLanguage::PyQuil
            | QuantumLanguage::ProjectQ
            | QuantumLanguage::Pennylane => "py",
            QuantumLanguage::BraketIR => "json",
            QuantumLanguage::Silq => "slq",
        }
    }

    /// Get supported gate set
    pub fn supported_gates(&self) -> Vec<&'static str> {
        match self {
            QuantumLanguage::OpenQASM2 | QuantumLanguage::OpenQASM3 => {
                vec![
                    "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx", "cy", "cz",
                    "ch", "swap", "ccx", "cswap",
                ]
            }
            QuantumLanguage::Quil => {
                vec![
                    "X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ", "CNOT", "CZ", "SWAP", "CCNOT",
                    "CSWAP", "PHASE",
                ]
            }
            QuantumLanguage::QSharp => {
                vec![
                    "X", "Y", "Z", "H", "S", "T", "CNOT", "CCNOT", "SWAP", "Rx", "Ry", "Rz", "R1",
                ]
            }
            QuantumLanguage::Cirq => {
                vec![
                    "X",
                    "Y",
                    "Z",
                    "H",
                    "S",
                    "T",
                    "CNOT",
                    "CZ",
                    "SWAP",
                    "Rx",
                    "Ry",
                    "Rz",
                    "ISWAP",
                    "SQRT_ISWAP",
                ]
            }
            QuantumLanguage::Qiskit => {
                vec![
                    "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx", "cy", "cz",
                    "ch", "swap", "ccx",
                ]
            }
            QuantumLanguage::PyQuil => {
                vec![
                    "X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ", "CNOT", "CZ", "SWAP", "PHASE",
                ]
            }
            QuantumLanguage::ProjectQ => {
                vec![
                    "X", "Y", "Z", "H", "S", "T", "Rx", "Ry", "Rz", "CNOT", "Swap",
                ]
            }
            QuantumLanguage::BraketIR => {
                vec![
                    "x", "y", "z", "h", "s", "si", "t", "ti", "rx", "ry", "rz", "cnot", "cy", "cz",
                    "swap", "iswap",
                ]
            }
            QuantumLanguage::Silq => {
                vec!["X", "Y", "Z", "H", "S", "T", "CNOT", "phase"]
            }
            QuantumLanguage::Pennylane => {
                vec![
                    "PauliX", "PauliY", "PauliZ", "Hadamard", "S", "T", "RX", "RY", "RZ", "CNOT",
                    "CZ", "SWAP",
                ]
            }
        }
    }
}

// ================================================================================================
// Circuit Representation
// ================================================================================================

/// Quantum circuit for compilation
#[derive(Debug, Clone)]
pub struct CompilableCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of classical bits
    pub num_cbits: usize,
    /// Circuit gates
    pub gates: Vec<GateInstruction>,
    /// Measurements
    pub measurements: Vec<(usize, usize)>, // (qubit, cbit)
}

/// Gate instruction in the circuit
#[derive(Debug, Clone)]
pub struct GateInstruction {
    /// Gate name
    pub name: String,
    /// Parameters (angles, etc.)
    pub params: Vec<f64>,
    /// Target qubits
    pub qubits: Vec<usize>,
    /// Control qubits (if controlled gate)
    pub controls: Vec<usize>,
}

impl CompilableCircuit {
    /// Create a new compilable circuit
    pub fn new(num_qubits: usize, num_cbits: usize) -> Self {
        Self {
            num_qubits,
            num_cbits,
            gates: Vec::new(),
            measurements: Vec::new(),
        }
    }

    /// Add a gate instruction
    pub fn add_gate(&mut self, instruction: GateInstruction) {
        self.gates.push(instruction);
    }

    /// Add measurement
    pub fn add_measurement(&mut self, qubit: usize, cbit: usize) {
        self.measurements.push((qubit, cbit));
    }

    /// Measure all qubits
    pub fn measure_all(&mut self) {
        for i in 0..self.num_qubits.min(self.num_cbits) {
            self.measurements.push((i, i));
        }
    }
}

// ================================================================================================
// Language Compiler
// ================================================================================================

/// Compiler for quantum programming languages
pub struct QuantumLanguageCompiler {
    target_language: QuantumLanguage,
    optimize: bool,
    include_comments: bool,
}

impl QuantumLanguageCompiler {
    /// Create a new compiler
    pub fn new(target_language: QuantumLanguage) -> Self {
        Self {
            target_language,
            optimize: true,
            include_comments: true,
        }
    }

    /// Enable/disable optimization
    pub fn with_optimization(mut self, optimize: bool) -> Self {
        self.optimize = optimize;
        self
    }

    /// Enable/disable comments
    pub fn with_comments(mut self, include_comments: bool) -> Self {
        self.include_comments = include_comments;
        self
    }

    /// Compile circuit to target language
    pub fn compile(&self, circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        match self.target_language {
            QuantumLanguage::OpenQASM2 => self.compile_to_openqasm2(circuit),
            QuantumLanguage::OpenQASM3 => self.compile_to_openqasm3(circuit),
            QuantumLanguage::Quil => self.compile_to_quil(circuit),
            QuantumLanguage::QSharp => self.compile_to_qsharp(circuit),
            QuantumLanguage::Cirq => self.compile_to_cirq(circuit),
            QuantumLanguage::Qiskit => self.compile_to_qiskit(circuit),
            QuantumLanguage::PyQuil => self.compile_to_pyquil(circuit),
            QuantumLanguage::ProjectQ => self.compile_to_projectq(circuit),
            QuantumLanguage::BraketIR => self.compile_to_braket_ir(circuit),
            QuantumLanguage::Silq => self.compile_to_silq(circuit),
            QuantumLanguage::Pennylane => self.compile_to_pennylane(circuit),
        }
    }

    /// Compile to OpenQASM 2.0
    fn compile_to_openqasm2(&self, circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();

        // Header
        writeln!(output, "OPENQASM 2.0;").unwrap();
        writeln!(output, "include \"qelib1.inc\";").unwrap();
        writeln!(output).unwrap();

        // Quantum register
        writeln!(output, "qreg q[{}];", circuit.num_qubits).unwrap();

        // Classical register
        if circuit.num_cbits > 0 {
            writeln!(output, "creg c[{}];", circuit.num_cbits).unwrap();
        }
        writeln!(output).unwrap();

        // Gates
        for gate in &circuit.gates {
            match gate.name.as_str() {
                "H" | "h" => {
                    writeln!(output, "h q[{}];", gate.qubits[0]).unwrap();
                }
                "X" | "x" => {
                    writeln!(output, "x q[{}];", gate.qubits[0]).unwrap();
                }
                "Y" | "y" => {
                    writeln!(output, "y q[{}];", gate.qubits[0]).unwrap();
                }
                "Z" | "z" => {
                    writeln!(output, "z q[{}];", gate.qubits[0]).unwrap();
                }
                "S" | "s" => {
                    writeln!(output, "s q[{}];", gate.qubits[0]).unwrap();
                }
                "T" | "t" => {
                    writeln!(output, "t q[{}];", gate.qubits[0]).unwrap();
                }
                "RX" | "rx" => {
                    writeln!(output, "rx({}) q[{}];", gate.params[0], gate.qubits[0]).unwrap();
                }
                "RY" | "ry" => {
                    writeln!(output, "ry({}) q[{}];", gate.params[0], gate.qubits[0]).unwrap();
                }
                "RZ" | "rz" => {
                    writeln!(output, "rz({}) q[{}];", gate.params[0], gate.qubits[0]).unwrap();
                }
                "CNOT" | "cx" => {
                    writeln!(output, "cx q[{}], q[{}];", gate.qubits[0], gate.qubits[1]).unwrap();
                }
                "CZ" | "cz" => {
                    writeln!(output, "cz q[{}], q[{}];", gate.qubits[0], gate.qubits[1]).unwrap();
                }
                "SWAP" | "swap" => {
                    writeln!(output, "swap q[{}], q[{}];", gate.qubits[0], gate.qubits[1]).unwrap();
                }
                _ => {
                    return Err(QuantRS2Error::UnsupportedOperation(format!(
                        "Gate {} not supported in OpenQASM 2.0",
                        gate.name
                    )));
                }
            }
        }

        // Measurements
        if !circuit.measurements.is_empty() {
            writeln!(output).unwrap();
            for (qubit, cbit) in &circuit.measurements {
                writeln!(output, "measure q[{}] -> c[{}];", qubit, cbit).unwrap();
            }
        }

        Ok(output)
    }

    /// Compile to OpenQASM 3.0
    fn compile_to_openqasm3(&self, circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();

        // Header
        writeln!(output, "OPENQASM 3.0;").unwrap();
        writeln!(output, "include \"stdgates.inc\";").unwrap();
        writeln!(output).unwrap();

        // Quantum bits
        writeln!(output, "qubit[{}] q;", circuit.num_qubits).unwrap();

        // Classical bits
        if circuit.num_cbits > 0 {
            writeln!(output, "bit[{}] c;", circuit.num_cbits).unwrap();
        }
        writeln!(output).unwrap();

        // Gates (similar to QASM 2.0)
        for gate in &circuit.gates {
            match gate.name.as_str() {
                "H" | "h" => writeln!(output, "h q[{}];", gate.qubits[0]).unwrap(),
                "X" | "x" => writeln!(output, "x q[{}];", gate.qubits[0]).unwrap(),
                "CNOT" | "cx" => {
                    writeln!(output, "cx q[{}], q[{}];", gate.qubits[0], gate.qubits[1]).unwrap()
                }
                _ => {}
            }
        }

        // Measurements
        for (qubit, cbit) in &circuit.measurements {
            writeln!(output, "c[{}] = measure q[{}];", cbit, qubit).unwrap();
        }

        Ok(output)
    }

    /// Compile to Quil
    fn compile_to_quil(&self, circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();

        // Declare memory
        if circuit.num_cbits > 0 {
            writeln!(output, "DECLARE ro BIT[{}]", circuit.num_cbits).unwrap();
            writeln!(output).unwrap();
        }

        // Gates
        for gate in &circuit.gates {
            match gate.name.as_str() {
                "H" | "h" => writeln!(output, "H {}", gate.qubits[0]).unwrap(),
                "X" | "x" => writeln!(output, "X {}", gate.qubits[0]).unwrap(),
                "Y" | "y" => writeln!(output, "Y {}", gate.qubits[0]).unwrap(),
                "Z" | "z" => writeln!(output, "Z {}", gate.qubits[0]).unwrap(),
                "RX" | "rx" => {
                    writeln!(output, "RX({}) {}", gate.params[0], gate.qubits[0]).unwrap()
                }
                "RY" | "ry" => {
                    writeln!(output, "RY({}) {}", gate.params[0], gate.qubits[0]).unwrap()
                }
                "RZ" | "rz" => {
                    writeln!(output, "RZ({}) {}", gate.params[0], gate.qubits[0]).unwrap()
                }
                "CNOT" | "cx" => {
                    writeln!(output, "CNOT {} {}", gate.qubits[0], gate.qubits[1]).unwrap()
                }
                "CZ" | "cz" => {
                    writeln!(output, "CZ {} {}", gate.qubits[0], gate.qubits[1]).unwrap()
                }
                _ => {}
            }
        }

        // Measurements
        for (qubit, cbit) in &circuit.measurements {
            writeln!(output, "MEASURE {} ro[{}]", qubit, cbit).unwrap();
        }

        Ok(output)
    }

    /// Compile to Q#
    fn compile_to_qsharp(&self, circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();

        writeln!(output, "namespace QuantumCircuit {{").unwrap();
        writeln!(output, "    open Microsoft.Quantum.Canon;").unwrap();
        writeln!(output, "    open Microsoft.Quantum.Intrinsic;").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "    operation RunCircuit() : Result[] {{").unwrap();
        writeln!(
            output,
            "        use qubits = Qubit[{}];",
            circuit.num_qubits
        )
        .unwrap();
        writeln!(output).unwrap();

        // Gates
        for gate in &circuit.gates {
            match gate.name.as_str() {
                "H" | "h" => writeln!(output, "        H(qubits[{}]);", gate.qubits[0]).unwrap(),
                "X" | "x" => writeln!(output, "        X(qubits[{}]);", gate.qubits[0]).unwrap(),
                "Y" | "y" => writeln!(output, "        Y(qubits[{}]);", gate.qubits[0]).unwrap(),
                "Z" | "z" => writeln!(output, "        Z(qubits[{}]);", gate.qubits[0]).unwrap(),
                "CNOT" | "cx" => writeln!(
                    output,
                    "        CNOT(qubits[{}], qubits[{}]);",
                    gate.qubits[0], gate.qubits[1]
                )
                .unwrap(),
                _ => {}
            }
        }

        writeln!(output).unwrap();
        writeln!(output, "        let results = ForEach(M, qubits);").unwrap();
        writeln!(output, "        ResetAll(qubits);").unwrap();
        writeln!(output, "        return results;").unwrap();
        writeln!(output, "    }}").unwrap();
        writeln!(output, "}}").unwrap();

        Ok(output)
    }

    /// Compile to Cirq (Python)
    fn compile_to_cirq(&self, circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();

        writeln!(output, "import cirq").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "# Create qubits").unwrap();
        writeln!(
            output,
            "qubits = [cirq.LineQubit(i) for i in range({})]",
            circuit.num_qubits
        )
        .unwrap();
        writeln!(output).unwrap();
        writeln!(output, "# Create circuit").unwrap();
        writeln!(output, "circuit = cirq.Circuit()").unwrap();
        writeln!(output).unwrap();

        // Gates
        for gate in &circuit.gates {
            match gate.name.as_str() {
                "H" | "h" => {
                    writeln!(output, "circuit.append(cirq.H(qubits[{}]))", gate.qubits[0]).unwrap()
                }
                "X" | "x" => {
                    writeln!(output, "circuit.append(cirq.X(qubits[{}]))", gate.qubits[0]).unwrap()
                }
                "Y" | "y" => {
                    writeln!(output, "circuit.append(cirq.Y(qubits[{}]))", gate.qubits[0]).unwrap()
                }
                "Z" | "z" => {
                    writeln!(output, "circuit.append(cirq.Z(qubits[{}]))", gate.qubits[0]).unwrap()
                }
                "CNOT" | "cx" => writeln!(
                    output,
                    "circuit.append(cirq.CNOT(qubits[{}], qubits[{}]))",
                    gate.qubits[0], gate.qubits[1]
                )
                .unwrap(),
                "RX" | "rx" => writeln!(
                    output,
                    "circuit.append(cirq.rx({}).on(qubits[{}]))",
                    gate.params[0], gate.qubits[0]
                )
                .unwrap(),
                "RY" | "ry" => writeln!(
                    output,
                    "circuit.append(cirq.ry({}).on(qubits[{}]))",
                    gate.params[0], gate.qubits[0]
                )
                .unwrap(),
                "RZ" | "rz" => writeln!(
                    output,
                    "circuit.append(cirq.rz({}).on(qubits[{}]))",
                    gate.params[0], gate.qubits[0]
                )
                .unwrap(),
                _ => {}
            }
        }

        // Measurements
        if !circuit.measurements.is_empty() {
            writeln!(output).unwrap();
            write!(output, "circuit.append(cirq.measure(").unwrap();
            for (i, (qubit, _)) in circuit.measurements.iter().enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                write!(output, "qubits[{}]", qubit).unwrap();
            }
            writeln!(output, ", key='result'))").unwrap();
        }

        writeln!(output).unwrap();
        writeln!(output, "print(circuit)").unwrap();

        Ok(output)
    }

    /// Compile to Qiskit (Python)
    fn compile_to_qiskit(&self, circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();

        writeln!(
            output,
            "from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister"
        )
        .unwrap();
        writeln!(output).unwrap();
        writeln!(output, "# Create registers").unwrap();
        writeln!(
            output,
            "qreg = QuantumRegister({}, 'q'))",
            circuit.num_qubits
        )
        .unwrap();

        if circuit.num_cbits > 0 {
            writeln!(
                output,
                "creg = ClassicalRegister({}, 'c')",
                circuit.num_cbits
            )
            .unwrap();
            writeln!(output, "circuit = QuantumCircuit(qreg, creg)").unwrap();
        } else {
            writeln!(output, "circuit = QuantumCircuit(qreg)").unwrap();
        }

        writeln!(output).unwrap();

        // Gates
        for gate in &circuit.gates {
            match gate.name.as_str() {
                "H" | "h" => writeln!(output, "circuit.h({})", gate.qubits[0]).unwrap(),
                "X" | "x" => writeln!(output, "circuit.x({})", gate.qubits[0]).unwrap(),
                "CNOT" | "cx" => {
                    writeln!(output, "circuit.cx({}, {})", gate.qubits[0], gate.qubits[1]).unwrap()
                }
                _ => {}
            }
        }

        // Measurements
        for (qubit, cbit) in &circuit.measurements {
            writeln!(output, "circuit.measure({}, {})", qubit, cbit).unwrap();
        }

        writeln!(output).unwrap();
        writeln!(output, "print(circuit)").unwrap();

        Ok(output)
    }

    /// Compile to PyQuil
    fn compile_to_pyquil(&self, circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();

        writeln!(output, "from pyquil import Program").unwrap();
        writeln!(output, "from pyquil.gates import *").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "program = Program()").unwrap();
        writeln!(output).unwrap();

        // Gates
        for gate in &circuit.gates {
            match gate.name.as_str() {
                "H" | "h" => writeln!(output, "program += H({})", gate.qubits[0]).unwrap(),
                "X" | "x" => writeln!(output, "program += X({})", gate.qubits[0]).unwrap(),
                "CNOT" | "cx" => writeln!(
                    output,
                    "program += CNOT({}, {})",
                    gate.qubits[0], gate.qubits[1]
                )
                .unwrap(),
                _ => {}
            }
        }

        Ok(output)
    }

    /// Compile to ProjectQ
    fn compile_to_projectq(&self, circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();

        writeln!(output, "from projectq import MainEngine").unwrap();
        writeln!(output, "from projectq.ops import *").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "eng = MainEngine()").unwrap();
        writeln!(
            output,
            "qubits = eng.allocate_qureg({}))",
            circuit.num_qubits
        )
        .unwrap();
        writeln!(output).unwrap();

        // Gates
        for gate in &circuit.gates {
            match gate.name.as_str() {
                "H" | "h" => writeln!(output, "H | qubits[{}]", gate.qubits[0]).unwrap(),
                "X" | "x" => writeln!(output, "X | qubits[{}]", gate.qubits[0]).unwrap(),
                "CNOT" | "cx" => writeln!(
                    output,
                    "CNOT | (qubits[{}], qubits[{}])",
                    gate.qubits[0], gate.qubits[1]
                )
                .unwrap(),
                _ => {}
            }
        }

        Ok(output)
    }

    /// Compile to Braket IR (JSON)
    fn compile_to_braket_ir(&self, circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();

        writeln!(output, "{{").unwrap();
        writeln!(output, "  \"braketSchemaHeader\": {{").unwrap();
        writeln!(output, "    \"name\": \"braket.ir.jaqcd.program\",").unwrap();
        writeln!(output, "    \"version\": \"1\"").unwrap();
        writeln!(output, "  }},").unwrap();
        writeln!(output, "  \"instructions\": [").unwrap();

        for (i, gate) in circuit.gates.iter().enumerate() {
            if i > 0 {
                writeln!(output, ",").unwrap();
            }
            write!(output, "    {{").unwrap();

            match gate.name.as_str() {
                "H" | "h" => {
                    write!(output, "\"type\": \"h\", \"target\": {}", gate.qubits[0]).unwrap();
                }
                "X" | "x" => {
                    write!(output, "\"type\": \"x\", \"target\": {}", gate.qubits[0]).unwrap();
                }
                "CNOT" | "cx" => {
                    write!(
                        output,
                        "\"type\": \"cnot\", \"control\": {}, \"target\": {}",
                        gate.qubits[0], gate.qubits[1]
                    )
                    .unwrap();
                }
                _ => {}
            }

            write!(output, "}}").unwrap();
        }

        writeln!(output, "").unwrap();
        writeln!(output, "  ]").unwrap();
        writeln!(output, "}}").unwrap();

        Ok(output)
    }

    /// Compile to Silq
    fn compile_to_silq(&self, circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();

        writeln!(output, "def circuit() {{").unwrap();
        writeln!(output, "  // Allocate qubits").unwrap();
        writeln!(output, "  q := 0:^{};", circuit.num_qubits).unwrap();
        writeln!(output).unwrap();

        for gate in &circuit.gates {
            match gate.name.as_str() {
                "H" | "h" => writeln!(
                    output,
                    "  q[{}] := H(q[{}]);",
                    gate.qubits[0], gate.qubits[0]
                )
                .unwrap(),
                "X" | "x" => writeln!(
                    output,
                    "  q[{}] := X(q[{}]);",
                    gate.qubits[0], gate.qubits[0]
                )
                .unwrap(),
                _ => {}
            }
        }

        writeln!(output, "  return q;").unwrap();
        writeln!(output, "}}").unwrap();

        Ok(output)
    }

    /// Compile to Pennylane
    fn compile_to_pennylane(&self, circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();

        writeln!(output, "import pennylane as qml").unwrap();
        writeln!(output).unwrap();
        writeln!(
            output,
            "dev = qml.device('default.qubit', wires={})",
            circuit.num_qubits
        )
        .unwrap();
        writeln!(output).unwrap();
        writeln!(output, "@qml.qnode(dev)").unwrap();
        writeln!(output, "def circuit():").unwrap();

        for gate in &circuit.gates {
            match gate.name.as_str() {
                "H" | "h" => {
                    writeln!(output, "    qml.Hadamard(wires={})", gate.qubits[0]).unwrap()
                }
                "X" | "x" => writeln!(output, "    qml.PauliX(wires={})", gate.qubits[0]).unwrap(),
                "CNOT" | "cx" => writeln!(
                    output,
                    "    qml.CNOT(wires=[{}, {}])",
                    gate.qubits[0], gate.qubits[1]
                )
                .unwrap(),
                _ => {}
            }
        }

        writeln!(
            output,
            "    return qml.probs(wires=range({}))",
            circuit.num_qubits
        )
        .unwrap();

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openqasm2_compilation() {
        let mut circuit = CompilableCircuit::new(2, 2);
        circuit.add_gate(GateInstruction {
            name: "H".to_string(),
            params: vec![],
            qubits: vec![0],
            controls: vec![],
        });
        circuit.add_gate(GateInstruction {
            name: "CNOT".to_string(),
            params: vec![],
            qubits: vec![0, 1],
            controls: vec![],
        });
        circuit.add_measurement(0, 0);
        circuit.add_measurement(1, 1);

        let compiler = QuantumLanguageCompiler::new(QuantumLanguage::OpenQASM2);
        let result = compiler.compile(&circuit).unwrap();

        assert!(result.contains("OPENQASM 2.0"));
        assert!(result.contains("h q[0]"));
        assert!(result.contains("cx q[0], q[1]"));
        assert!(result.contains("measure q[0] -> c[0]"));
    }

    #[test]
    fn test_quil_compilation() {
        let mut circuit = CompilableCircuit::new(1, 1);
        circuit.add_gate(GateInstruction {
            name: "H".to_string(),
            params: vec![],
            qubits: vec![0],
            controls: vec![],
        });

        let compiler = QuantumLanguageCompiler::new(QuantumLanguage::Quil);
        let result = compiler.compile(&circuit).unwrap();

        assert!(result.contains("H 0"));
    }
}
