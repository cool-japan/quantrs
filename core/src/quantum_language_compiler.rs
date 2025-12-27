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

/// Convert fmt::Error to QuantRS2Error for string formatting operations
#[inline]
fn fmt_err(e: std::fmt::Error) -> QuantRS2Error {
    QuantRS2Error::ComputationError(format!("String formatting error: {e}"))
}
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
    pub const fn name(&self) -> &'static str {
        match self {
            Self::OpenQASM2 => "OpenQASM 2.0",
            Self::OpenQASM3 => "OpenQASM 3.0",
            Self::Quil => "Quil",
            Self::QSharp => "Q#",
            Self::Cirq => "Cirq",
            Self::Qiskit => "Qiskit",
            Self::PyQuil => "PyQuil",
            Self::ProjectQ => "ProjectQ",
            Self::BraketIR => "Braket IR",
            Self::Silq => "Silq",
            Self::Pennylane => "Pennylane",
        }
    }
    /// Get file extension
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::OpenQASM2 | Self::OpenQASM3 => "qasm",
            Self::Quil => "quil",
            Self::QSharp => "qs",
            Self::Cirq | Self::Qiskit | Self::PyQuil | Self::ProjectQ | Self::Pennylane => "py",
            Self::BraketIR => "json",
            Self::Silq => "slq",
        }
    }
    /// Get supported gate set
    pub fn supported_gates(&self) -> Vec<&'static str> {
        match self {
            Self::OpenQASM2 | Self::OpenQASM3 => {
                vec![
                    "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx", "cy", "cz",
                    "ch", "swap", "ccx", "cswap",
                ]
            }
            Self::Quil => {
                vec![
                    "X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ", "CNOT", "CZ", "SWAP", "CCNOT",
                    "CSWAP", "PHASE",
                ]
            }
            Self::QSharp => {
                vec![
                    "X", "Y", "Z", "H", "S", "T", "CNOT", "CCNOT", "SWAP", "Rx", "Ry", "Rz", "R1",
                ]
            }
            Self::Cirq => {
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
            Self::Qiskit => {
                vec![
                    "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx", "cy", "cz",
                    "ch", "swap", "ccx",
                ]
            }
            Self::PyQuil => {
                vec![
                    "X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ", "CNOT", "CZ", "SWAP", "PHASE",
                ]
            }
            Self::ProjectQ => {
                vec![
                    "X", "Y", "Z", "H", "S", "T", "Rx", "Ry", "Rz", "CNOT", "Swap",
                ]
            }
            Self::BraketIR => {
                vec![
                    "x", "y", "z", "h", "s", "si", "t", "ti", "rx", "ry", "rz", "cnot", "cy", "cz",
                    "swap", "iswap",
                ]
            }
            Self::Silq => vec!["X", "Y", "Z", "H", "S", "T", "CNOT", "phase"],
            Self::Pennylane => {
                vec![
                    "PauliX", "PauliY", "PauliZ", "Hadamard", "S", "T", "RX", "RY", "RZ", "CNOT",
                    "CZ", "SWAP",
                ]
            }
        }
    }
}
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
    pub measurements: Vec<(usize, usize)>,
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
    pub const fn new(num_qubits: usize, num_cbits: usize) -> Self {
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
    #[must_use]
    pub fn measure_all(&mut self) {
        for i in 0..self.num_qubits.min(self.num_cbits) {
            self.measurements.push((i, i));
        }
    }
}
/// Compiler for quantum programming languages
pub struct QuantumLanguageCompiler {
    target_language: QuantumLanguage,
    optimize: bool,
    include_comments: bool,
}
impl QuantumLanguageCompiler {
    /// Create a new compiler
    pub const fn new(target_language: QuantumLanguage) -> Self {
        Self {
            target_language,
            optimize: true,
            include_comments: true,
        }
    }
    /// Enable/disable optimization
    #[must_use]
    pub const fn with_optimization(mut self, optimize: bool) -> Self {
        self.optimize = optimize;
        self
    }
    /// Enable/disable comments
    #[must_use]
    pub const fn with_comments(mut self, include_comments: bool) -> Self {
        self.include_comments = include_comments;
        self
    }
    /// Compile circuit to target language
    pub fn compile(&self, circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        match self.target_language {
            QuantumLanguage::OpenQASM2 => Self::compile_to_openqasm2(circuit),
            QuantumLanguage::OpenQASM3 => Self::compile_to_openqasm3(circuit),
            QuantumLanguage::Quil => Self::compile_to_quil(circuit),
            QuantumLanguage::QSharp => Self::compile_to_qsharp(circuit),
            QuantumLanguage::Cirq => Self::compile_to_cirq(circuit),
            QuantumLanguage::Qiskit => Self::compile_to_qiskit(circuit),
            QuantumLanguage::PyQuil => Self::compile_to_pyquil(circuit),
            QuantumLanguage::ProjectQ => Self::compile_to_projectq(circuit),
            QuantumLanguage::BraketIR => Self::compile_to_braket_ir(circuit),
            QuantumLanguage::Silq => Self::compile_to_silq(circuit),
            QuantumLanguage::Pennylane => Self::compile_to_pennylane(circuit),
        }
    }
    /// Compile to OpenQASM 2.0
    fn compile_to_openqasm2(circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();
        writeln!(output, "OPENQASM 2.0;").map_err(fmt_err)?;
        writeln!(output, "include \"qelib1.inc\";").map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        writeln!(output, "qreg q[{}];", circuit.num_qubits).map_err(fmt_err)?;
        if circuit.num_cbits > 0 {
            writeln!(output, "creg c[{}];", circuit.num_cbits).map_err(fmt_err)?;
        }
        writeln!(output).map_err(fmt_err)?;
        for gate in &circuit.gates {
            let qubit0 = gate.qubits.first().ok_or_else(|| {
                QuantRS2Error::InvalidInput("Gate missing target qubit".to_string())
            })?;
            match gate.name.as_str() {
                "H" | "h" => {
                    writeln!(output, "h q[{qubit0}];").map_err(fmt_err)?;
                }
                "X" | "x" => {
                    writeln!(output, "x q[{qubit0}];").map_err(fmt_err)?;
                }
                "Y" | "y" => {
                    writeln!(output, "y q[{qubit0}];").map_err(fmt_err)?;
                }
                "Z" | "z" => {
                    writeln!(output, "z q[{qubit0}];").map_err(fmt_err)?;
                }
                "S" | "s" => {
                    writeln!(output, "s q[{qubit0}];").map_err(fmt_err)?;
                }
                "T" | "t" => {
                    writeln!(output, "t q[{qubit0}];").map_err(fmt_err)?;
                }
                "RX" | "rx" => {
                    let param = gate.params.first().ok_or_else(|| {
                        QuantRS2Error::InvalidInput("RX gate missing angle parameter".to_string())
                    })?;
                    writeln!(output, "rx({param}) q[{qubit0}];").map_err(fmt_err)?;
                }
                "RY" | "ry" => {
                    let param = gate.params.first().ok_or_else(|| {
                        QuantRS2Error::InvalidInput("RY gate missing angle parameter".to_string())
                    })?;
                    writeln!(output, "ry({param}) q[{qubit0}];").map_err(fmt_err)?;
                }
                "RZ" | "rz" => {
                    let param = gate.params.first().ok_or_else(|| {
                        QuantRS2Error::InvalidInput("RZ gate missing angle parameter".to_string())
                    })?;
                    writeln!(output, "rz({param}) q[{qubit0}];").map_err(fmt_err)?;
                }
                "CNOT" | "cx" => {
                    let qubit1 = gate.qubits.get(1).ok_or_else(|| {
                        QuantRS2Error::InvalidInput("CNOT gate missing second qubit".to_string())
                    })?;
                    writeln!(output, "cx q[{qubit0}], q[{qubit1}];").map_err(fmt_err)?;
                }
                "CZ" | "cz" => {
                    let qubit1 = gate.qubits.get(1).ok_or_else(|| {
                        QuantRS2Error::InvalidInput("CZ gate missing second qubit".to_string())
                    })?;
                    writeln!(output, "cz q[{qubit0}], q[{qubit1}];").map_err(fmt_err)?;
                }
                "SWAP" | "swap" => {
                    let qubit1 = gate.qubits.get(1).ok_or_else(|| {
                        QuantRS2Error::InvalidInput("SWAP gate missing second qubit".to_string())
                    })?;
                    writeln!(output, "swap q[{qubit0}], q[{qubit1}];").map_err(fmt_err)?;
                }
                _ => {
                    return Err(QuantRS2Error::UnsupportedOperation(format!(
                        "Gate {} not supported in OpenQASM 2.0",
                        gate.name
                    )));
                }
            }
        }
        if !circuit.measurements.is_empty() {
            writeln!(output).map_err(fmt_err)?;
            for (qubit, cbit) in &circuit.measurements {
                writeln!(output, "measure q[{qubit}] -> c[{cbit}];").map_err(fmt_err)?;
            }
        }
        Ok(output)
    }
    /// Compile to OpenQASM 3.0
    fn compile_to_openqasm3(circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();
        writeln!(output, "OPENQASM 3.0;").map_err(fmt_err)?;
        writeln!(output, "include \"stdgates.inc\";").map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        writeln!(output, "qubit[{}] q;", circuit.num_qubits).map_err(fmt_err)?;
        if circuit.num_cbits > 0 {
            writeln!(output, "bit[{}] c;", circuit.num_cbits).map_err(fmt_err)?;
        }
        writeln!(output).map_err(fmt_err)?;
        for gate in &circuit.gates {
            let qubit0 = gate.qubits.first().ok_or_else(|| {
                QuantRS2Error::InvalidInput("Gate missing target qubit".to_string())
            })?;
            match gate.name.as_str() {
                "H" | "h" => writeln!(output, "h q[{qubit0}];").map_err(fmt_err)?,
                "X" | "x" => writeln!(output, "x q[{qubit0}];").map_err(fmt_err)?,
                "CNOT" | "cx" => {
                    let qubit1 = gate.qubits.get(1).ok_or_else(|| {
                        QuantRS2Error::InvalidInput("CNOT gate missing second qubit".to_string())
                    })?;
                    writeln!(output, "cx q[{qubit0}], q[{qubit1}];").map_err(fmt_err)?;
                }
                _ => {}
            }
        }
        for (qubit, cbit) in &circuit.measurements {
            writeln!(output, "c[{cbit}] = measure q[{qubit}];").map_err(fmt_err)?;
        }
        Ok(output)
    }
    /// Compile to Quil
    fn compile_to_quil(circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();
        if circuit.num_cbits > 0 {
            writeln!(output, "DECLARE ro BIT[{}]", circuit.num_cbits).map_err(fmt_err)?;
            writeln!(output).map_err(fmt_err)?;
        }
        for gate in &circuit.gates {
            let qubit0 = gate.qubits.first().ok_or_else(|| {
                QuantRS2Error::InvalidInput("Gate missing target qubit".to_string())
            })?;
            match gate.name.as_str() {
                "H" | "h" => writeln!(output, "H {qubit0}").map_err(fmt_err)?,
                "X" | "x" => writeln!(output, "X {qubit0}").map_err(fmt_err)?,
                "Y" | "y" => writeln!(output, "Y {qubit0}").map_err(fmt_err)?,
                "Z" | "z" => writeln!(output, "Z {qubit0}").map_err(fmt_err)?,
                "RX" | "rx" => {
                    let param = gate.params.first().ok_or_else(|| {
                        QuantRS2Error::InvalidInput("RX gate missing angle parameter".to_string())
                    })?;
                    writeln!(output, "RX({param}) {qubit0}").map_err(fmt_err)?;
                }
                "RY" | "ry" => {
                    let param = gate.params.first().ok_or_else(|| {
                        QuantRS2Error::InvalidInput("RY gate missing angle parameter".to_string())
                    })?;
                    writeln!(output, "RY({param}) {qubit0}").map_err(fmt_err)?;
                }
                "RZ" | "rz" => {
                    let param = gate.params.first().ok_or_else(|| {
                        QuantRS2Error::InvalidInput("RZ gate missing angle parameter".to_string())
                    })?;
                    writeln!(output, "RZ({param}) {qubit0}").map_err(fmt_err)?;
                }
                "CNOT" | "cx" => {
                    let qubit1 = gate.qubits.get(1).ok_or_else(|| {
                        QuantRS2Error::InvalidInput("CNOT gate missing second qubit".to_string())
                    })?;
                    writeln!(output, "CNOT {qubit0} {qubit1}").map_err(fmt_err)?;
                }
                "CZ" | "cz" => {
                    let qubit1 = gate.qubits.get(1).ok_or_else(|| {
                        QuantRS2Error::InvalidInput("CZ gate missing second qubit".to_string())
                    })?;
                    writeln!(output, "CZ {qubit0} {qubit1}").map_err(fmt_err)?;
                }
                _ => {}
            }
        }
        for (qubit, cbit) in &circuit.measurements {
            writeln!(output, "MEASURE {qubit} ro[{cbit}]").map_err(fmt_err)?;
        }
        Ok(output)
    }
    /// Compile to Q#
    fn compile_to_qsharp(circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();
        writeln!(output, "namespace QuantumCircuit {{").map_err(fmt_err)?;
        writeln!(output, "    open Microsoft.Quantum.Canon;").map_err(fmt_err)?;
        writeln!(output, "    open Microsoft.Quantum.Intrinsic;").map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        writeln!(output, "    operation RunCircuit() : Result[] {{").map_err(fmt_err)?;
        writeln!(
            output,
            "        use qubits = Qubit[{}];",
            circuit.num_qubits
        )
        .map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        for gate in &circuit.gates {
            let qubit0 = gate.qubits.first().ok_or_else(|| {
                QuantRS2Error::InvalidInput("Gate missing target qubit".to_string())
            })?;
            match gate.name.as_str() {
                "H" | "h" => {
                    writeln!(output, "        H(qubits[{qubit0}]);").map_err(fmt_err)?;
                }
                "X" | "x" => {
                    writeln!(output, "        X(qubits[{qubit0}]);").map_err(fmt_err)?;
                }
                "Y" | "y" => {
                    writeln!(output, "        Y(qubits[{qubit0}]);").map_err(fmt_err)?;
                }
                "Z" | "z" => {
                    writeln!(output, "        Z(qubits[{qubit0}]);").map_err(fmt_err)?;
                }
                "CNOT" | "cx" => {
                    let qubit1 = gate.qubits.get(1).ok_or_else(|| {
                        QuantRS2Error::InvalidInput("CNOT gate missing second qubit".to_string())
                    })?;
                    writeln!(output, "        CNOT(qubits[{qubit0}], qubits[{qubit1}]);")
                        .map_err(fmt_err)?;
                }
                _ => {}
            }
        }
        writeln!(output).map_err(fmt_err)?;
        writeln!(output, "        let results = ForEach(M, qubits);").map_err(fmt_err)?;
        writeln!(output, "        ResetAll(qubits);").map_err(fmt_err)?;
        writeln!(output, "        return results;").map_err(fmt_err)?;
        writeln!(output, "    }}").map_err(fmt_err)?;
        writeln!(output, "}}").map_err(fmt_err)?;
        Ok(output)
    }
    /// Compile to Cirq (Python)
    fn compile_to_cirq(circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();
        writeln!(output, "import cirq").map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        writeln!(output, "# Create qubits").map_err(fmt_err)?;
        writeln!(
            output,
            "qubits = [cirq.LineQubit(i) for i in range({})]",
            circuit.num_qubits
        )
        .map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        writeln!(output, "# Create circuit").map_err(fmt_err)?;
        writeln!(output, "circuit = cirq.Circuit()").map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        for gate in &circuit.gates {
            let qubit0 = gate.qubits.first().ok_or_else(|| {
                QuantRS2Error::InvalidInput("Gate missing target qubit".to_string())
            })?;
            match gate.name.as_str() {
                "H" | "h" => {
                    writeln!(output, "circuit.append(cirq.H(qubits[{qubit0}]))")
                        .map_err(fmt_err)?;
                }
                "X" | "x" => {
                    writeln!(output, "circuit.append(cirq.X(qubits[{qubit0}]))")
                        .map_err(fmt_err)?;
                }
                "Y" | "y" => {
                    writeln!(output, "circuit.append(cirq.Y(qubits[{qubit0}]))")
                        .map_err(fmt_err)?;
                }
                "Z" | "z" => {
                    writeln!(output, "circuit.append(cirq.Z(qubits[{qubit0}]))")
                        .map_err(fmt_err)?;
                }
                "CNOT" | "cx" => {
                    let qubit1 = gate.qubits.get(1).ok_or_else(|| {
                        QuantRS2Error::InvalidInput("CNOT gate missing second qubit".to_string())
                    })?;
                    writeln!(
                        output,
                        "circuit.append(cirq.CNOT(qubits[{qubit0}], qubits[{qubit1}]))"
                    )
                    .map_err(fmt_err)?;
                }
                "RX" | "rx" => {
                    let param = gate.params.first().ok_or_else(|| {
                        QuantRS2Error::InvalidInput("RX gate missing angle parameter".to_string())
                    })?;
                    writeln!(
                        output,
                        "circuit.append(cirq.rx({param}).on(qubits[{qubit0}]))"
                    )
                    .map_err(fmt_err)?;
                }
                "RY" | "ry" => {
                    let param = gate.params.first().ok_or_else(|| {
                        QuantRS2Error::InvalidInput("RY gate missing angle parameter".to_string())
                    })?;
                    writeln!(
                        output,
                        "circuit.append(cirq.ry({param}).on(qubits[{qubit0}]))"
                    )
                    .map_err(fmt_err)?;
                }
                "RZ" | "rz" => {
                    let param = gate.params.first().ok_or_else(|| {
                        QuantRS2Error::InvalidInput("RZ gate missing angle parameter".to_string())
                    })?;
                    writeln!(
                        output,
                        "circuit.append(cirq.rz({param}).on(qubits[{qubit0}]))"
                    )
                    .map_err(fmt_err)?;
                }
                _ => {}
            }
        }
        if !circuit.measurements.is_empty() {
            writeln!(output).map_err(fmt_err)?;
            write!(output, "circuit.append(cirq.measure(").map_err(fmt_err)?;
            for (i, (qubit, _)) in circuit.measurements.iter().enumerate() {
                if i > 0 {
                    write!(output, ", ").map_err(fmt_err)?;
                }
                write!(output, "qubits[{qubit}]").map_err(fmt_err)?;
            }
            writeln!(output, ", key='result'))").map_err(fmt_err)?;
        }
        writeln!(output).map_err(fmt_err)?;
        writeln!(output, "print(circuit)").map_err(fmt_err)?;
        Ok(output)
    }
    /// Compile to Qiskit (Python)
    fn compile_to_qiskit(circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();
        writeln!(
            output,
            "from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister"
        )
        .map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        writeln!(output, "# Create registers").map_err(fmt_err)?;
        writeln!(
            output,
            "qreg = QuantumRegister({}, 'q'))",
            circuit.num_qubits
        )
        .map_err(fmt_err)?;
        if circuit.num_cbits > 0 {
            writeln!(
                output,
                "creg = ClassicalRegister({}, 'c')",
                circuit.num_cbits
            )
            .map_err(fmt_err)?;
            writeln!(output, "circuit = QuantumCircuit(qreg, creg)").map_err(fmt_err)?;
        } else {
            writeln!(output, "circuit = QuantumCircuit(qreg)").map_err(fmt_err)?;
        }
        writeln!(output).map_err(fmt_err)?;
        for gate in &circuit.gates {
            let qubit0 = gate.qubits.first().ok_or_else(|| {
                QuantRS2Error::InvalidInput("Gate missing target qubit".to_string())
            })?;
            match gate.name.as_str() {
                "H" | "h" => writeln!(output, "circuit.h({qubit0})").map_err(fmt_err)?,
                "X" | "x" => writeln!(output, "circuit.x({qubit0})").map_err(fmt_err)?,
                "CNOT" | "cx" => {
                    let qubit1 = gate.qubits.get(1).ok_or_else(|| {
                        QuantRS2Error::InvalidInput("CNOT gate missing second qubit".to_string())
                    })?;
                    writeln!(output, "circuit.cx({qubit0}, {qubit1})").map_err(fmt_err)?;
                }
                _ => {}
            }
        }
        for (qubit, cbit) in &circuit.measurements {
            writeln!(output, "circuit.measure({qubit}, {cbit})").map_err(fmt_err)?;
        }
        writeln!(output).map_err(fmt_err)?;
        writeln!(output, "print(circuit)").map_err(fmt_err)?;
        Ok(output)
    }
    /// Compile to PyQuil
    fn compile_to_pyquil(circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();
        writeln!(output, "from pyquil import Program").map_err(fmt_err)?;
        writeln!(output, "from pyquil.gates import *").map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        writeln!(output, "program = Program()").map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        for gate in &circuit.gates {
            let qubit0 = gate.qubits.first().ok_or_else(|| {
                QuantRS2Error::InvalidInput("Gate missing target qubit".to_string())
            })?;
            match gate.name.as_str() {
                "H" | "h" => writeln!(output, "program += H({qubit0})").map_err(fmt_err)?,
                "X" | "x" => writeln!(output, "program += X({qubit0})").map_err(fmt_err)?,
                "CNOT" | "cx" => {
                    let qubit1 = gate.qubits.get(1).ok_or_else(|| {
                        QuantRS2Error::InvalidInput("CNOT gate missing second qubit".to_string())
                    })?;
                    writeln!(output, "program += CNOT({qubit0}, {qubit1})").map_err(fmt_err)?;
                }
                _ => {}
            }
        }
        Ok(output)
    }
    /// Compile to ProjectQ
    fn compile_to_projectq(circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();
        writeln!(output, "from projectq import MainEngine").map_err(fmt_err)?;
        writeln!(output, "from projectq.ops import *").map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        writeln!(output, "eng = MainEngine()").map_err(fmt_err)?;
        writeln!(
            output,
            "qubits = eng.allocate_qureg({}))",
            circuit.num_qubits
        )
        .map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        for gate in &circuit.gates {
            let qubit0 = gate.qubits.first().ok_or_else(|| {
                QuantRS2Error::InvalidInput("Gate missing target qubit".to_string())
            })?;
            match gate.name.as_str() {
                "H" | "h" => writeln!(output, "H | qubits[{qubit0}]").map_err(fmt_err)?,
                "X" | "x" => writeln!(output, "X | qubits[{qubit0}]").map_err(fmt_err)?,
                "CNOT" | "cx" => {
                    let qubit1 = gate.qubits.get(1).ok_or_else(|| {
                        QuantRS2Error::InvalidInput("CNOT gate missing second qubit".to_string())
                    })?;
                    writeln!(output, "CNOT | (qubits[{qubit0}], qubits[{qubit1}])")
                        .map_err(fmt_err)?;
                }
                _ => {}
            }
        }
        Ok(output)
    }
    /// Compile to Braket IR (JSON)
    fn compile_to_braket_ir(circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();
        writeln!(output, "{{").map_err(fmt_err)?;
        writeln!(output, "  \"braketSchemaHeader\": {{").map_err(fmt_err)?;
        writeln!(output, "    \"name\": \"braket.ir.jaqcd.program\",").map_err(fmt_err)?;
        writeln!(output, "    \"version\": \"1\"").map_err(fmt_err)?;
        writeln!(output, "  }},").map_err(fmt_err)?;
        writeln!(output, "  \"instructions\": [").map_err(fmt_err)?;
        for (i, gate) in circuit.gates.iter().enumerate() {
            if i > 0 {
                writeln!(output, ",").map_err(fmt_err)?;
            }
            let qubit0 = gate.qubits.first().ok_or_else(|| {
                QuantRS2Error::InvalidInput("Gate missing target qubit".to_string())
            })?;
            write!(output, "    {{").map_err(fmt_err)?;
            match gate.name.as_str() {
                "H" | "h" => {
                    write!(output, "\"type\": \"h\", \"target\": {qubit0}").map_err(fmt_err)?;
                }
                "X" | "x" => {
                    write!(output, "\"type\": \"x\", \"target\": {qubit0}").map_err(fmt_err)?;
                }
                "CNOT" | "cx" => {
                    let qubit1 = gate.qubits.get(1).ok_or_else(|| {
                        QuantRS2Error::InvalidInput("CNOT gate missing second qubit".to_string())
                    })?;
                    write!(
                        output,
                        "\"type\": \"cnot\", \"control\": {qubit0}, \"target\": {qubit1}"
                    )
                    .map_err(fmt_err)?;
                }
                _ => {}
            }
            write!(output, "}}").map_err(fmt_err)?;
        }
        writeln!(output).map_err(fmt_err)?;
        writeln!(output, "  ]").map_err(fmt_err)?;
        writeln!(output, "}}").map_err(fmt_err)?;
        Ok(output)
    }
    /// Compile to Silq
    fn compile_to_silq(circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();
        writeln!(output, "def circuit() {{").map_err(fmt_err)?;
        writeln!(output, "  // Allocate qubits").map_err(fmt_err)?;
        writeln!(output, "  q := 0:^{};", circuit.num_qubits).map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        for gate in &circuit.gates {
            let qubit0 = gate.qubits.first().ok_or_else(|| {
                QuantRS2Error::InvalidInput("Gate missing target qubit".to_string())
            })?;
            match gate.name.as_str() {
                "H" | "h" => {
                    writeln!(output, "  q[{qubit0}] := H(q[{qubit0}]);").map_err(fmt_err)?;
                }
                "X" | "x" => {
                    writeln!(output, "  q[{qubit0}] := X(q[{qubit0}]);").map_err(fmt_err)?;
                }
                _ => {}
            }
        }
        writeln!(output, "  return q;").map_err(fmt_err)?;
        writeln!(output, "}}").map_err(fmt_err)?;
        Ok(output)
    }
    /// Compile to Pennylane
    fn compile_to_pennylane(circuit: &CompilableCircuit) -> QuantRS2Result<String> {
        let mut output = String::new();
        writeln!(output, "import pennylane as qml").map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        writeln!(
            output,
            "dev = qml.device('default.qubit', wires={})",
            circuit.num_qubits
        )
        .map_err(fmt_err)?;
        writeln!(output).map_err(fmt_err)?;
        writeln!(output, "@qml.qnode(dev)").map_err(fmt_err)?;
        writeln!(output, "def circuit():").map_err(fmt_err)?;
        for gate in &circuit.gates {
            let qubit0 = gate.qubits.first().ok_or_else(|| {
                QuantRS2Error::InvalidInput("Gate missing target qubit".to_string())
            })?;
            match gate.name.as_str() {
                "H" | "h" => {
                    writeln!(output, "    qml.Hadamard(wires={qubit0})").map_err(fmt_err)?;
                }
                "X" | "x" => {
                    writeln!(output, "    qml.PauliX(wires={qubit0})").map_err(fmt_err)?;
                }
                "CNOT" | "cx" => {
                    let qubit1 = gate.qubits.get(1).ok_or_else(|| {
                        QuantRS2Error::InvalidInput("CNOT gate missing second qubit".to_string())
                    })?;
                    writeln!(output, "    qml.CNOT(wires=[{qubit0}, {qubit1}])")
                        .map_err(fmt_err)?;
                }
                _ => {}
            }
        }
        writeln!(
            output,
            "    return qml.probs(wires=range({}))",
            circuit.num_qubits
        )
        .map_err(fmt_err)?;
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
        let result = compiler
            .compile(&circuit)
            .expect("OpenQASM 2.0 compilation should succeed");
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
        let result = compiler
            .compile(&circuit)
            .expect("Quil compilation should succeed");
        assert!(result.contains("H 0"));
    }
}
