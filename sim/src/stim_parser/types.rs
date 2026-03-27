//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use crate::stabilizer::StabilizerGate;

use std::collections::HashMap;

use super::functions::parse_instruction;

/// Pauli operator type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauliType {
    I,
    X,
    Y,
    Z,
}
/// Stim circuit representation
#[derive(Debug, Clone)]
pub struct StimCircuit {
    /// Instructions in the circuit
    pub instructions: Vec<StimInstruction>,
    /// Number of qubits (inferred from max qubit index)
    pub num_qubits: usize,
    /// Metadata/comments
    pub metadata: Vec<String>,
}
impl StimCircuit {
    /// Create new empty Stim circuit
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            num_qubits: 0,
            metadata: Vec::new(),
        }
    }
    /// Add an instruction
    pub fn add_instruction(&mut self, instruction: StimInstruction) {
        match &instruction {
            StimInstruction::SingleQubitGate { qubit, .. } => {
                self.num_qubits = self.num_qubits.max(*qubit + 1);
            }
            StimInstruction::TwoQubitGate {
                control, target, ..
            } => {
                self.num_qubits = self.num_qubits.max(*control + 1).max(*target + 1);
            }
            StimInstruction::Measure { qubits, .. } | StimInstruction::Reset { qubits } => {
                if let Some(&max_qubit) = qubits.iter().max() {
                    self.num_qubits = self.num_qubits.max(max_qubit + 1);
                }
            }
            _ => {}
        }
        self.instructions.push(instruction);
    }
    /// Parse from Stim format string
    pub fn from_str(s: &str) -> Result<Self> {
        let mut circuit = Self::new();
        for (line_num, line) in s.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(stripped) = line.strip_prefix('#') {
                circuit.metadata.push(stripped.trim().to_string());
                circuit.add_instruction(StimInstruction::Comment(stripped.trim().to_string()));
                continue;
            }
            match parse_instruction(line) {
                Ok(instruction) => circuit.add_instruction(instruction),
                Err(e) => {
                    return Err(SimulatorError::InvalidOperation(format!(
                        "Line {}: {}",
                        line_num + 1,
                        e
                    )));
                }
            }
        }
        Ok(circuit)
    }
    /// Get all gates (excluding measurements, resets, comments)
    pub fn gates(&self) -> Vec<StabilizerGate> {
        self.instructions
            .iter()
            .filter_map(|inst| match inst {
                StimInstruction::SingleQubitGate { gate_type, qubit } => {
                    Some(gate_type.to_stabilizer_gate(*qubit))
                }
                StimInstruction::TwoQubitGate {
                    gate_type,
                    control,
                    target,
                } => Some(gate_type.to_stabilizer_gate(*control, *target)),
                _ => None,
            })
            .collect()
    }
    /// Get measurement instructions
    pub fn measurements(&self) -> Vec<(MeasurementBasis, Vec<usize>)> {
        self.instructions
            .iter()
            .filter_map(|inst| match inst {
                StimInstruction::Measure { basis, qubits } => Some((*basis, qubits.clone())),
                _ => None,
            })
            .collect()
    }
    /// Get reset instructions
    pub fn resets(&self) -> Vec<Vec<usize>> {
        self.instructions
            .iter()
            .filter_map(|inst| match inst {
                StimInstruction::Reset { qubits } => Some(qubits.clone()),
                _ => None,
            })
            .collect()
    }
    /// Convert to Stim format string
    pub fn to_stim_string(&self) -> String {
        let mut output = String::new();
        for instruction in &self.instructions {
            match instruction {
                StimInstruction::SingleQubitGate { gate_type, qubit } => {
                    let gate_name = match gate_type {
                        SingleQubitGateType::H => "H",
                        SingleQubitGateType::S => "S",
                        SingleQubitGateType::SDag => "S_DAG",
                        SingleQubitGateType::SqrtX => "SQRT_X",
                        SingleQubitGateType::SqrtXDag => "SQRT_X_DAG",
                        SingleQubitGateType::SqrtY => "SQRT_Y",
                        SingleQubitGateType::SqrtYDag => "SQRT_Y_DAG",
                        SingleQubitGateType::X => "X",
                        SingleQubitGateType::Y => "Y",
                        SingleQubitGateType::Z => "Z",
                    };
                    output.push_str(&format!("{} {}\n", gate_name, qubit));
                }
                StimInstruction::TwoQubitGate {
                    gate_type,
                    control,
                    target,
                } => {
                    let gate_name = match gate_type {
                        TwoQubitGateType::CNOT => "CNOT",
                        TwoQubitGateType::CZ => "CZ",
                        TwoQubitGateType::CY => "CY",
                        TwoQubitGateType::SWAP => "SWAP",
                    };
                    output.push_str(&format!("{} {} {}\n", gate_name, control, target));
                }
                StimInstruction::Measure { basis, qubits } => {
                    let measure_name = match basis {
                        MeasurementBasis::Z => "M",
                        MeasurementBasis::X => "MX",
                        MeasurementBasis::Y => "MY",
                    };
                    output.push_str(&format!(
                        "{} {}\n",
                        measure_name,
                        qubits
                            .iter()
                            .map(|q| q.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    ));
                }
                StimInstruction::Reset { qubits } => {
                    output.push_str(&format!(
                        "R {}\n",
                        qubits
                            .iter()
                            .map(|q| q.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    ));
                }
                StimInstruction::Comment(comment) => {
                    output.push_str(&format!("# {}\n", comment));
                }
                StimInstruction::Tick => {
                    output.push_str("TICK\n");
                }
                StimInstruction::Detector {
                    coordinates,
                    record_targets,
                } => {
                    output.push_str("DETECTOR");
                    for coord in coordinates {
                        output.push_str(&format!(" {}", coord));
                    }
                    for target in record_targets {
                        output.push_str(&format!(" rec[{}]", target));
                    }
                    output.push('\n');
                }
                StimInstruction::ObservableInclude {
                    observable_index,
                    record_targets,
                } => {
                    output.push_str(&format!("OBSERVABLE_INCLUDE({})", observable_index));
                    for target in record_targets {
                        output.push_str(&format!(" rec[{}]", target));
                    }
                    output.push('\n');
                }
                StimInstruction::MeasureReset { basis, qubits } => {
                    let measure_name = match basis {
                        MeasurementBasis::Z => "MR",
                        MeasurementBasis::X => "MRX",
                        MeasurementBasis::Y => "MRY",
                    };
                    output.push_str(&format!(
                        "{} {}\n",
                        measure_name,
                        qubits
                            .iter()
                            .map(|q| q.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    ));
                }
                StimInstruction::Depolarize1 {
                    probability,
                    qubits,
                } => {
                    output.push_str(&format!(
                        "DEPOLARIZE1({}) {}\n",
                        probability,
                        qubits
                            .iter()
                            .map(|q| q.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    ));
                }
                StimInstruction::Depolarize2 {
                    probability,
                    qubit_pairs,
                } => {
                    output.push_str(&format!("DEPOLARIZE2({})", probability));
                    for (q1, q2) in qubit_pairs {
                        output.push_str(&format!(" {} {}", q1, q2));
                    }
                    output.push('\n');
                }
                StimInstruction::XError {
                    probability,
                    qubits,
                } => {
                    output.push_str(&format!(
                        "X_ERROR({}) {}\n",
                        probability,
                        qubits
                            .iter()
                            .map(|q| q.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    ));
                }
                StimInstruction::YError {
                    probability,
                    qubits,
                } => {
                    output.push_str(&format!(
                        "Y_ERROR({}) {}\n",
                        probability,
                        qubits
                            .iter()
                            .map(|q| q.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    ));
                }
                StimInstruction::ZError {
                    probability,
                    qubits,
                } => {
                    output.push_str(&format!(
                        "Z_ERROR({}) {}\n",
                        probability,
                        qubits
                            .iter()
                            .map(|q| q.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    ));
                }
                StimInstruction::PauliChannel1 { px, py, pz, qubits } => {
                    output.push_str(&format!(
                        "PAULI_CHANNEL_1({},{},{}) {}\n",
                        px,
                        py,
                        pz,
                        qubits
                            .iter()
                            .map(|q| q.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    ));
                }
                StimInstruction::PauliChannel2 {
                    probabilities,
                    qubit_pairs,
                } => {
                    output.push_str(&format!(
                        "PAULI_CHANNEL_2({})",
                        probabilities
                            .iter()
                            .map(|p| p.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    ));
                    for (q1, q2) in qubit_pairs {
                        output.push_str(&format!(" {} {}", q1, q2));
                    }
                    output.push('\n');
                }
                StimInstruction::CorrelatedError {
                    probability,
                    targets,
                } => {
                    output.push_str(&format!("E({})", probability));
                    for target in targets {
                        let pauli_char = match target.pauli {
                            PauliType::I => 'I',
                            PauliType::X => 'X',
                            PauliType::Y => 'Y',
                            PauliType::Z => 'Z',
                        };
                        output.push_str(&format!(" {}{}", pauli_char, target.qubit));
                    }
                    output.push('\n');
                }
                StimInstruction::ElseCorrelatedError {
                    probability,
                    targets,
                } => {
                    output.push_str(&format!("ELSE_CORRELATED_ERROR({})", probability));
                    for target in targets {
                        let pauli_char = match target.pauli {
                            PauliType::I => 'I',
                            PauliType::X => 'X',
                            PauliType::Y => 'Y',
                            PauliType::Z => 'Z',
                        };
                        output.push_str(&format!(" {}{}", pauli_char, target.qubit));
                    }
                    output.push('\n');
                }
                StimInstruction::ShiftCoords { shifts } => {
                    output.push_str(&format!(
                        "SHIFT_COORDS {}\n",
                        shifts
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    ));
                }
                StimInstruction::QubitCoords { qubit, coordinates } => {
                    output.push_str(&format!(
                        "QUBIT_COORDS({}) {}\n",
                        qubit,
                        coordinates
                            .iter()
                            .map(|c| c.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    ));
                }
                StimInstruction::Repeat {
                    count,
                    instructions,
                } => {
                    output.push_str(&format!("REPEAT {} {{\n", count));
                    for inst in instructions {
                        output.push_str("  # nested instruction\n");
                    }
                    output.push_str("}\n");
                }
            }
        }
        output
    }
    /// Get circuit statistics
    pub fn statistics(&self) -> CircuitStatistics {
        let mut num_gates = 0;
        let mut num_measurements = 0;
        let mut num_resets = 0;
        let mut gate_counts = std::collections::HashMap::new();
        for instruction in &self.instructions {
            match instruction {
                StimInstruction::SingleQubitGate { gate_type, .. } => {
                    num_gates += 1;
                    *gate_counts.entry(format!("{:?}", gate_type)).or_insert(0) += 1;
                }
                StimInstruction::TwoQubitGate { gate_type, .. } => {
                    num_gates += 1;
                    *gate_counts.entry(format!("{:?}", gate_type)).or_insert(0) += 1;
                }
                StimInstruction::Measure { qubits, .. } => {
                    num_measurements += qubits.len();
                }
                StimInstruction::Reset { qubits } => {
                    num_resets += qubits.len();
                }
                _ => {}
            }
        }
        CircuitStatistics {
            num_qubits: self.num_qubits,
            num_gates,
            num_measurements,
            num_resets,
            gate_counts,
        }
    }
}
/// Pauli target for correlated errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PauliTarget {
    pub pauli: PauliType,
    pub qubit: usize,
}
/// Stim instruction type
#[derive(Debug, Clone, PartialEq)]
pub enum StimInstruction {
    /// Single-qubit gate
    SingleQubitGate {
        gate_type: SingleQubitGateType,
        qubit: usize,
    },
    /// Two-qubit gate
    TwoQubitGate {
        gate_type: TwoQubitGateType,
        control: usize,
        target: usize,
    },
    /// Measurement
    Measure {
        basis: MeasurementBasis,
        qubits: Vec<usize>,
    },
    /// Reset operation
    Reset { qubits: Vec<usize> },
    /// Comment (ignored during execution)
    Comment(String),
    /// Barrier/tick (for timing information)
    Tick,
    /// Detector annotation for error correction
    /// DETECTOR [coords...] rec[-1] rec[-2] ...
    Detector {
        coordinates: Vec<f64>,
        record_targets: Vec<i32>,
    },
    /// Observable annotation for logical observables
    /// OBSERVABLE_INCLUDE(k) rec[-1] rec[-2] ...
    ObservableInclude {
        observable_index: usize,
        record_targets: Vec<i32>,
    },
    /// Measure and reset to |0⟩
    /// MR q1 q2 ...
    MeasureReset {
        basis: MeasurementBasis,
        qubits: Vec<usize>,
    },
    /// Single-qubit depolarizing noise
    /// DEPOLARIZE1(p) q1 q2 ...
    Depolarize1 {
        probability: f64,
        qubits: Vec<usize>,
    },
    /// Two-qubit depolarizing noise
    /// DEPOLARIZE2(p) q1 q2 q3 q4 ...
    Depolarize2 {
        probability: f64,
        qubit_pairs: Vec<(usize, usize)>,
    },
    /// X error (bit flip)
    /// X_ERROR(p) q1 q2 ...
    XError {
        probability: f64,
        qubits: Vec<usize>,
    },
    /// Y error (bit-phase flip)
    /// Y_ERROR(p) q1 q2 ...
    YError {
        probability: f64,
        qubits: Vec<usize>,
    },
    /// Z error (phase flip)
    /// Z_ERROR(p) q1 q2 ...
    ZError {
        probability: f64,
        qubits: Vec<usize>,
    },
    /// Pauli channel (single qubit)
    /// PAULI_CHANNEL_1(px, py, pz) q1 q2 ...
    PauliChannel1 {
        px: f64,
        py: f64,
        pz: f64,
        qubits: Vec<usize>,
    },
    /// Pauli channel (two qubits)
    /// PAULI_CHANNEL_2(p_IX, p_IY, ..., p_ZZ) q1 q2
    PauliChannel2 {
        probabilities: Vec<f64>,
        qubit_pairs: Vec<(usize, usize)>,
    },
    /// Correlated error
    /// CORRELATED_ERROR(p) X1 Y2 Z3 ... or shorthand: E(p) X1 Y2 Z3 ...
    CorrelatedError {
        probability: f64,
        targets: Vec<PauliTarget>,
    },
    /// Else correlated error (conditional on previous E not triggering)
    /// ELSE_CORRELATED_ERROR(p) X1 Y2 Z3 ...
    ElseCorrelatedError {
        probability: f64,
        targets: Vec<PauliTarget>,
    },
    /// Shift coordinate system
    /// SHIFT_COORDS [dx, dy, dz, ...]
    ShiftCoords { shifts: Vec<f64> },
    /// Qubit coordinates
    /// QUBIT_COORDS(q) x y z ...
    QubitCoords { qubit: usize, coordinates: Vec<f64> },
    /// Repeat block
    /// REPEAT N { ... }
    Repeat {
        count: usize,
        instructions: Vec<StimInstruction>,
    },
}
/// Single-qubit Clifford gate types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingleQubitGateType {
    H,
    S,
    SDag,
    SqrtX,
    SqrtXDag,
    SqrtY,
    SqrtYDag,
    X,
    Y,
    Z,
}
impl SingleQubitGateType {
    /// Convert to stabilizer gate
    pub fn to_stabilizer_gate(self, qubit: usize) -> StabilizerGate {
        match self {
            Self::H => StabilizerGate::H(qubit),
            Self::S => StabilizerGate::S(qubit),
            Self::SDag => StabilizerGate::SDag(qubit),
            Self::SqrtX => StabilizerGate::SqrtX(qubit),
            Self::SqrtXDag => StabilizerGate::SqrtXDag(qubit),
            Self::SqrtY => StabilizerGate::SqrtY(qubit),
            Self::SqrtYDag => StabilizerGate::SqrtYDag(qubit),
            Self::X => StabilizerGate::X(qubit),
            Self::Y => StabilizerGate::Y(qubit),
            Self::Z => StabilizerGate::Z(qubit),
        }
    }
}
/// Two-qubit Clifford gate types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoQubitGateType {
    CNOT,
    CZ,
    CY,
    SWAP,
}
impl TwoQubitGateType {
    /// Convert to stabilizer gate
    pub fn to_stabilizer_gate(self, control: usize, target: usize) -> StabilizerGate {
        match self {
            Self::CNOT => StabilizerGate::CNOT(control, target),
            Self::CZ => StabilizerGate::CZ(control, target),
            Self::CY => StabilizerGate::CY(control, target),
            Self::SWAP => StabilizerGate::SWAP(control, target),
        }
    }
}
/// Measurement basis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementBasis {
    Z,
    X,
    Y,
}
/// Circuit statistics
#[derive(Debug, Clone)]
pub struct CircuitStatistics {
    pub num_qubits: usize,
    pub num_gates: usize,
    pub num_measurements: usize,
    pub num_resets: usize,
    pub gate_counts: std::collections::HashMap<String, usize>,
}
/// Error type helper for parsing
#[derive(Debug, Clone, Copy)]
pub(super) enum ErrorType {
    X,
    Y,
    Z,
}
