//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};

use super::types::{
    ErrorType, MeasurementBasis, PauliTarget, PauliType, SingleQubitGateType, StimCircuit,
    StimInstruction, TwoQubitGateType,
};

/// Parse a single Stim instruction from a line
pub(super) fn parse_instruction(line: &str) -> Result<StimInstruction> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return Err(SimulatorError::InvalidOperation(
            "Empty instruction".to_string(),
        ));
    }
    let instruction_name = if let Some(paren_pos) = parts[0].find('(') {
        parts[0][..paren_pos].to_uppercase()
    } else {
        parts[0].to_uppercase()
    };
    match instruction_name.as_str() {
        "H" => parse_single_qubit_gate(&parts, SingleQubitGateType::H),
        "S" => parse_single_qubit_gate(&parts, SingleQubitGateType::S),
        "S_DAG" => parse_single_qubit_gate(&parts, SingleQubitGateType::SDag),
        "SQRT_X" => parse_single_qubit_gate(&parts, SingleQubitGateType::SqrtX),
        "SQRT_X_DAG" => parse_single_qubit_gate(&parts, SingleQubitGateType::SqrtXDag),
        "SQRT_Y" => parse_single_qubit_gate(&parts, SingleQubitGateType::SqrtY),
        "SQRT_Y_DAG" => parse_single_qubit_gate(&parts, SingleQubitGateType::SqrtYDag),
        "X" => parse_single_qubit_gate(&parts, SingleQubitGateType::X),
        "Y" => parse_single_qubit_gate(&parts, SingleQubitGateType::Y),
        "Z" => parse_single_qubit_gate(&parts, SingleQubitGateType::Z),
        "CNOT" | "CX" => parse_two_qubit_gate(&parts, TwoQubitGateType::CNOT),
        "CZ" => parse_two_qubit_gate(&parts, TwoQubitGateType::CZ),
        "CY" => parse_two_qubit_gate(&parts, TwoQubitGateType::CY),
        "SWAP" => parse_two_qubit_gate(&parts, TwoQubitGateType::SWAP),
        "M" => parse_measurement(&parts, MeasurementBasis::Z),
        "MX" => parse_measurement(&parts, MeasurementBasis::X),
        "MY" => parse_measurement(&parts, MeasurementBasis::Y),
        "R" => parse_reset(&parts),
        "TICK" => Ok(StimInstruction::Tick),
        "DETECTOR" => parse_detector(&parts),
        "OBSERVABLE_INCLUDE" => parse_observable_include(&parts),
        "MR" => parse_measure_reset(&parts, MeasurementBasis::Z),
        "MRX" => parse_measure_reset(&parts, MeasurementBasis::X),
        "MRY" => parse_measure_reset(&parts, MeasurementBasis::Y),
        "DEPOLARIZE1" => parse_depolarize1(&parts),
        "DEPOLARIZE2" => parse_depolarize2(&parts),
        "X_ERROR" => parse_error(&parts, ErrorType::X),
        "Y_ERROR" => parse_error(&parts, ErrorType::Y),
        "Z_ERROR" => parse_error(&parts, ErrorType::Z),
        "PAULI_CHANNEL_1" => parse_pauli_channel_1(&parts),
        "PAULI_CHANNEL_2" => parse_pauli_channel_2(&parts),
        "CORRELATED_ERROR" | "E" => parse_correlated_error(&parts),
        "ELSE_CORRELATED_ERROR" => parse_else_correlated_error(&parts),
        "SHIFT_COORDS" => parse_shift_coords(&parts),
        "QUBIT_COORDS" => parse_qubit_coords(&parts),
        "REPEAT" => parse_repeat(&parts),
        _ => Err(SimulatorError::InvalidOperation(format!(
            "Unknown instruction: {}",
            instruction_name
        ))),
    }
}
fn parse_single_qubit_gate(
    parts: &[&str],
    gate_type: SingleQubitGateType,
) -> Result<StimInstruction> {
    if parts.len() != 2 {
        return Err(SimulatorError::InvalidOperation(format!(
            "Single-qubit gate requires 1 qubit argument, got {}",
            parts.len() - 1
        )));
    }
    let qubit = parts[1].parse::<usize>().map_err(|_| {
        SimulatorError::InvalidOperation(format!("Invalid qubit index: {}", parts[1]))
    })?;
    Ok(StimInstruction::SingleQubitGate { gate_type, qubit })
}
fn parse_two_qubit_gate(parts: &[&str], gate_type: TwoQubitGateType) -> Result<StimInstruction> {
    if parts.len() != 3 {
        return Err(SimulatorError::InvalidOperation(format!(
            "Two-qubit gate requires 2 qubit arguments, got {}",
            parts.len() - 1
        )));
    }
    let control = parts[1].parse::<usize>().map_err(|_| {
        SimulatorError::InvalidOperation(format!("Invalid control qubit: {}", parts[1]))
    })?;
    let target = parts[2].parse::<usize>().map_err(|_| {
        SimulatorError::InvalidOperation(format!("Invalid target qubit: {}", parts[2]))
    })?;
    Ok(StimInstruction::TwoQubitGate {
        gate_type,
        control,
        target,
    })
}
fn parse_measurement(parts: &[&str], basis: MeasurementBasis) -> Result<StimInstruction> {
    if parts.len() < 2 {
        return Err(SimulatorError::InvalidOperation(
            "Measurement requires at least 1 qubit".to_string(),
        ));
    }
    let qubits = parts[1..]
        .iter()
        .map(|s| {
            s.parse::<usize>().map_err(|_| {
                SimulatorError::InvalidOperation(format!("Invalid qubit index: {}", s))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StimInstruction::Measure { basis, qubits })
}
fn parse_reset(parts: &[&str]) -> Result<StimInstruction> {
    if parts.len() < 2 {
        return Err(SimulatorError::InvalidOperation(
            "Reset requires at least 1 qubit".to_string(),
        ));
    }
    let qubits = parts[1..]
        .iter()
        .map(|s| {
            s.parse::<usize>().map_err(|_| {
                SimulatorError::InvalidOperation(format!("Invalid qubit index: {}", s))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StimInstruction::Reset { qubits })
}
/// Parse parameter from instruction name (e.g., "DEPOLARIZE1(0.01)")
fn parse_parameter(instruction: &str) -> Result<f64> {
    let start = instruction
        .find('(')
        .ok_or_else(|| SimulatorError::InvalidOperation("Missing parameter".to_string()))?;
    let end = instruction.find(')').ok_or_else(|| {
        SimulatorError::InvalidOperation("Missing closing parenthesis".to_string())
    })?;
    instruction[start + 1..end]
        .trim()
        .parse::<f64>()
        .map_err(|_| {
            SimulatorError::InvalidOperation(format!(
                "Invalid parameter: {}",
                &instruction[start + 1..end]
            ))
        })
}
/// Parse multiple parameters from instruction (e.g., "PAULI_CHANNEL_1(0.01,0.02,0.03)")
fn parse_parameters(instruction: &str) -> Result<Vec<f64>> {
    let start = instruction
        .find('(')
        .ok_or_else(|| SimulatorError::InvalidOperation("Missing parameters".to_string()))?;
    let end = instruction.find(')').ok_or_else(|| {
        SimulatorError::InvalidOperation("Missing closing parenthesis".to_string())
    })?;
    instruction[start + 1..end]
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|_| SimulatorError::InvalidOperation(format!("Invalid parameter: {}", s)))
        })
        .collect::<Result<Vec<_>>>()
}
fn parse_detector(parts: &[&str]) -> Result<StimInstruction> {
    let mut coordinates = Vec::new();
    let mut record_targets = Vec::new();
    for part in parts.iter().skip(1) {
        if part.starts_with("rec[") {
            let idx_str = part.trim_start_matches("rec[").trim_end_matches(']');
            let idx = idx_str.parse::<i32>().map_err(|_| {
                SimulatorError::InvalidOperation(format!("Invalid record target: {}", part))
            })?;
            record_targets.push(idx);
        } else {
            if let Ok(coord) = part.parse::<f64>() {
                coordinates.push(coord);
            }
        }
    }
    Ok(StimInstruction::Detector {
        coordinates,
        record_targets,
    })
}
fn parse_observable_include(parts: &[&str]) -> Result<StimInstruction> {
    if parts.is_empty() {
        return Err(SimulatorError::InvalidOperation(
            "OBSERVABLE_INCLUDE requires index".to_string(),
        ));
    }
    let observable_index = parse_parameter(parts[0])? as usize;
    let mut record_targets = Vec::new();
    for part in parts.iter().skip(1) {
        if part.starts_with("rec[") {
            let idx_str = part.trim_start_matches("rec[").trim_end_matches(']');
            let idx = idx_str.parse::<i32>().map_err(|_| {
                SimulatorError::InvalidOperation(format!("Invalid record target: {}", part))
            })?;
            record_targets.push(idx);
        }
    }
    Ok(StimInstruction::ObservableInclude {
        observable_index,
        record_targets,
    })
}
fn parse_measure_reset(parts: &[&str], basis: MeasurementBasis) -> Result<StimInstruction> {
    if parts.len() < 2 {
        return Err(SimulatorError::InvalidOperation(
            "Measure-reset requires at least 1 qubit".to_string(),
        ));
    }
    let qubits = parts[1..]
        .iter()
        .map(|s| {
            s.parse::<usize>().map_err(|_| {
                SimulatorError::InvalidOperation(format!("Invalid qubit index: {}", s))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StimInstruction::MeasureReset { basis, qubits })
}
fn parse_depolarize1(parts: &[&str]) -> Result<StimInstruction> {
    if parts.is_empty() {
        return Err(SimulatorError::InvalidOperation(
            "DEPOLARIZE1 requires probability".to_string(),
        ));
    }
    let probability = parse_parameter(parts[0])?;
    let qubits = parts[1..]
        .iter()
        .map(|s| {
            s.parse::<usize>().map_err(|_| {
                SimulatorError::InvalidOperation(format!("Invalid qubit index: {}", s))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StimInstruction::Depolarize1 {
        probability,
        qubits,
    })
}
fn parse_depolarize2(parts: &[&str]) -> Result<StimInstruction> {
    if parts.is_empty() {
        return Err(SimulatorError::InvalidOperation(
            "DEPOLARIZE2 requires probability".to_string(),
        ));
    }
    let probability = parse_parameter(parts[0])?;
    let qubits: Vec<usize> = parts[1..]
        .iter()
        .map(|s| {
            s.parse::<usize>().map_err(|_| {
                SimulatorError::InvalidOperation(format!("Invalid qubit index: {}", s))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut qubit_pairs = Vec::new();
    for chunk in qubits.chunks(2) {
        if chunk.len() == 2 {
            qubit_pairs.push((chunk[0], chunk[1]));
        }
    }
    Ok(StimInstruction::Depolarize2 {
        probability,
        qubit_pairs,
    })
}
fn parse_error(parts: &[&str], error_type: ErrorType) -> Result<StimInstruction> {
    if parts.is_empty() {
        return Err(SimulatorError::InvalidOperation(
            "Error instruction requires probability".to_string(),
        ));
    }
    let probability = parse_parameter(parts[0])?;
    let qubits = parts[1..]
        .iter()
        .map(|s| {
            s.parse::<usize>().map_err(|_| {
                SimulatorError::InvalidOperation(format!("Invalid qubit index: {}", s))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    match error_type {
        ErrorType::X => Ok(StimInstruction::XError {
            probability,
            qubits,
        }),
        ErrorType::Y => Ok(StimInstruction::YError {
            probability,
            qubits,
        }),
        ErrorType::Z => Ok(StimInstruction::ZError {
            probability,
            qubits,
        }),
    }
}
fn parse_pauli_channel_1(parts: &[&str]) -> Result<StimInstruction> {
    if parts.is_empty() {
        return Err(SimulatorError::InvalidOperation(
            "PAULI_CHANNEL_1 requires parameters".to_string(),
        ));
    }
    let params = parse_parameters(parts[0])?;
    if params.len() != 3 {
        return Err(SimulatorError::InvalidOperation(
            "PAULI_CHANNEL_1 requires 3 parameters (px, py, pz)".to_string(),
        ));
    }
    let qubits = parts[1..]
        .iter()
        .map(|s| {
            s.parse::<usize>().map_err(|_| {
                SimulatorError::InvalidOperation(format!("Invalid qubit index: {}", s))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StimInstruction::PauliChannel1 {
        px: params[0],
        py: params[1],
        pz: params[2],
        qubits,
    })
}
fn parse_pauli_channel_2(parts: &[&str]) -> Result<StimInstruction> {
    if parts.is_empty() {
        return Err(SimulatorError::InvalidOperation(
            "PAULI_CHANNEL_2 requires parameters".to_string(),
        ));
    }
    let probabilities = parse_parameters(parts[0])?;
    if probabilities.len() != 15 {
        return Err(SimulatorError::InvalidOperation(
            "PAULI_CHANNEL_2 requires 15 parameters".to_string(),
        ));
    }
    let qubits: Vec<usize> = parts[1..]
        .iter()
        .map(|s| {
            s.parse::<usize>().map_err(|_| {
                SimulatorError::InvalidOperation(format!("Invalid qubit index: {}", s))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut qubit_pairs = Vec::new();
    for chunk in qubits.chunks(2) {
        if chunk.len() == 2 {
            qubit_pairs.push((chunk[0], chunk[1]));
        }
    }
    Ok(StimInstruction::PauliChannel2 {
        probabilities,
        qubit_pairs,
    })
}
fn parse_correlated_error(parts: &[&str]) -> Result<StimInstruction> {
    if parts.is_empty() {
        return Err(SimulatorError::InvalidOperation(
            "CORRELATED_ERROR requires probability".to_string(),
        ));
    }
    let probability = parse_parameter(parts[0])?;
    let mut targets = Vec::new();
    for part in parts.iter().skip(1) {
        let pauli_char = part
            .chars()
            .next()
            .ok_or_else(|| SimulatorError::InvalidOperation("Empty Pauli target".to_string()))?;
        let pauli = match pauli_char.to_ascii_uppercase() {
            'I' => PauliType::I,
            'X' => PauliType::X,
            'Y' => PauliType::Y,
            'Z' => PauliType::Z,
            _ => {
                return Err(SimulatorError::InvalidOperation(format!(
                    "Invalid Pauli type: {}",
                    pauli_char
                )));
            }
        };
        let qubit = part[1..].parse::<usize>().map_err(|_| {
            SimulatorError::InvalidOperation(format!("Invalid qubit in target: {}", part))
        })?;
        targets.push(PauliTarget { pauli, qubit });
    }
    Ok(StimInstruction::CorrelatedError {
        probability,
        targets,
    })
}
fn parse_else_correlated_error(parts: &[&str]) -> Result<StimInstruction> {
    if parts.is_empty() {
        return Err(SimulatorError::InvalidOperation(
            "ELSE_CORRELATED_ERROR requires probability".to_string(),
        ));
    }
    let probability = parse_parameter(parts[0])?;
    let mut targets = Vec::new();
    for part in parts.iter().skip(1) {
        let pauli_char = part
            .chars()
            .next()
            .ok_or_else(|| SimulatorError::InvalidOperation("Empty Pauli target".to_string()))?;
        let pauli = match pauli_char.to_ascii_uppercase() {
            'I' => PauliType::I,
            'X' => PauliType::X,
            'Y' => PauliType::Y,
            'Z' => PauliType::Z,
            _ => {
                return Err(SimulatorError::InvalidOperation(format!(
                    "Invalid Pauli type: {}",
                    pauli_char
                )));
            }
        };
        let qubit = part[1..].parse::<usize>().map_err(|_| {
            SimulatorError::InvalidOperation(format!("Invalid qubit in target: {}", part))
        })?;
        targets.push(PauliTarget { pauli, qubit });
    }
    Ok(StimInstruction::ElseCorrelatedError {
        probability,
        targets,
    })
}
fn parse_shift_coords(parts: &[&str]) -> Result<StimInstruction> {
    let shifts = parts[1..]
        .iter()
        .map(|s| {
            s.parse::<f64>().map_err(|_| {
                SimulatorError::InvalidOperation(format!("Invalid coordinate shift: {}", s))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StimInstruction::ShiftCoords { shifts })
}
fn parse_qubit_coords(parts: &[&str]) -> Result<StimInstruction> {
    if parts.is_empty() {
        return Err(SimulatorError::InvalidOperation(
            "QUBIT_COORDS requires qubit index".to_string(),
        ));
    }
    let qubit = parse_parameter(parts[0])? as usize;
    let coordinates = parts[1..]
        .iter()
        .map(|s| {
            s.parse::<f64>()
                .map_err(|_| SimulatorError::InvalidOperation(format!("Invalid coordinate: {}", s)))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StimInstruction::QubitCoords { qubit, coordinates })
}
fn parse_repeat(_parts: &[&str]) -> Result<StimInstruction> {
    Err(SimulatorError::NotImplemented(
        "REPEAT blocks not yet implemented".to_string(),
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse_single_qubit_gates() {
        let circuit = StimCircuit::from_str("H 0\nS 1\nX 2").unwrap();
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.instructions.len(), 3);
        match &circuit.instructions[0] {
            StimInstruction::SingleQubitGate { gate_type, qubit } => {
                assert_eq!(*gate_type, SingleQubitGateType::H);
                assert_eq!(*qubit, 0);
            }
            _ => panic!("Expected SingleQubitGate"),
        }
    }
    #[test]
    fn test_parse_two_qubit_gates() {
        let circuit = StimCircuit::from_str("CNOT 0 1\nCZ 1 2").unwrap();
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.instructions.len(), 2);
        match &circuit.instructions[0] {
            StimInstruction::TwoQubitGate {
                gate_type,
                control,
                target,
            } => {
                assert_eq!(*gate_type, TwoQubitGateType::CNOT);
                assert_eq!(*control, 0);
                assert_eq!(*target, 1);
            }
            _ => panic!("Expected TwoQubitGate"),
        }
    }
    #[test]
    fn test_parse_measurements() {
        let circuit = StimCircuit::from_str("M 0 1\nMX 2\nMY 3").unwrap();
        assert_eq!(circuit.instructions.len(), 3);
        match &circuit.instructions[0] {
            StimInstruction::Measure { basis, qubits } => {
                assert_eq!(*basis, MeasurementBasis::Z);
                assert_eq!(*qubits, vec![0, 1]);
            }
            _ => panic!("Expected Measure"),
        }
        match &circuit.instructions[1] {
            StimInstruction::Measure { basis, qubits } => {
                assert_eq!(*basis, MeasurementBasis::X);
                assert_eq!(*qubits, vec![2]);
            }
            _ => panic!("Expected Measure"),
        }
    }
    #[test]
    fn test_parse_reset() {
        let circuit = StimCircuit::from_str("R 0 1 2").unwrap();
        assert_eq!(circuit.instructions.len(), 1);
        match &circuit.instructions[0] {
            StimInstruction::Reset { qubits } => {
                assert_eq!(*qubits, vec![0, 1, 2]);
            }
            _ => panic!("Expected Reset"),
        }
    }
    #[test]
    fn test_parse_comments() {
        let circuit = StimCircuit::from_str("# Bell state\nH 0\nCNOT 0 1\n# End").unwrap();
        assert_eq!(circuit.metadata.len(), 2);
        assert_eq!(circuit.gates().len(), 2);
    }
    #[test]
    fn test_bell_state_circuit() {
        let stim_code = r#"
# Bell state preparation
H 0
CNOT 0 1
M 0 1
        "#;
        let circuit = StimCircuit::from_str(stim_code).unwrap();
        assert_eq!(circuit.num_qubits, 2);
        let gates = circuit.gates();
        assert_eq!(gates.len(), 2);
        let measurements = circuit.measurements();
        assert_eq!(measurements.len(), 1);
        assert_eq!(measurements[0].1, vec![0, 1]);
    }
    #[test]
    fn test_to_stim_string() {
        let mut circuit = StimCircuit::new();
        circuit.add_instruction(StimInstruction::SingleQubitGate {
            gate_type: SingleQubitGateType::H,
            qubit: 0,
        });
        circuit.add_instruction(StimInstruction::TwoQubitGate {
            gate_type: TwoQubitGateType::CNOT,
            control: 0,
            target: 1,
        });
        circuit.add_instruction(StimInstruction::Measure {
            basis: MeasurementBasis::Z,
            qubits: vec![0, 1],
        });
        let stim_string = circuit.to_stim_string();
        let parsed = StimCircuit::from_str(&stim_string).unwrap();
        assert_eq!(parsed.num_qubits, circuit.num_qubits);
        assert_eq!(parsed.gates().len(), circuit.gates().len());
    }
    #[test]
    fn test_circuit_statistics() {
        let stim_code = r#"
H 0
H 1
CNOT 0 1
CNOT 1 2
M 0 1 2
R 0
        "#;
        let circuit = StimCircuit::from_str(stim_code).unwrap();
        let stats = circuit.statistics();
        assert_eq!(stats.num_qubits, 3);
        assert_eq!(stats.num_gates, 4);
        assert_eq!(stats.num_measurements, 3);
        assert_eq!(stats.num_resets, 1);
        assert_eq!(stats.gate_counts.get("H"), Some(&2));
        assert_eq!(stats.gate_counts.get("CNOT"), Some(&2));
    }
    #[test]
    fn test_error_invalid_instruction() {
        let result = StimCircuit::from_str("INVALID_GATE 0");
        assert!(result.is_err());
    }
    #[test]
    fn test_error_invalid_qubit() {
        let result = StimCircuit::from_str("H abc");
        assert!(result.is_err());
    }
    #[test]
    fn test_error_wrong_arity() {
        let result = StimCircuit::from_str("CNOT 0");
        assert!(result.is_err());
    }
    #[test]
    fn test_case_insensitive() {
        let circuit = StimCircuit::from_str("h 0\ncnot 0 1").unwrap();
        assert_eq!(circuit.gates().len(), 2);
    }
    #[test]
    fn test_detector_parsing() {
        let circuit = StimCircuit::from_str("DETECTOR rec[-1] rec[-2]").unwrap();
        assert_eq!(circuit.instructions.len(), 1);
        match &circuit.instructions[0] {
            StimInstruction::Detector { record_targets, .. } => {
                assert_eq!(record_targets.len(), 2);
                assert_eq!(record_targets[0], -1);
                assert_eq!(record_targets[1], -2);
            }
            _ => panic!("Expected Detector"),
        }
    }
    #[test]
    fn test_observable_include_parsing() {
        let circuit = StimCircuit::from_str("OBSERVABLE_INCLUDE(0) rec[-1]").unwrap();
        assert_eq!(circuit.instructions.len(), 1);
        match &circuit.instructions[0] {
            StimInstruction::ObservableInclude {
                observable_index,
                record_targets,
            } => {
                assert_eq!(*observable_index, 0);
                assert_eq!(record_targets.len(), 1);
                assert_eq!(record_targets[0], -1);
            }
            _ => panic!("Expected ObservableInclude"),
        }
    }
    #[test]
    fn test_measure_reset_parsing() {
        let circuit = StimCircuit::from_str("MR 0 1\nMRX 2").unwrap();
        assert_eq!(circuit.instructions.len(), 2);
        match &circuit.instructions[0] {
            StimInstruction::MeasureReset { basis, qubits } => {
                assert_eq!(*basis, MeasurementBasis::Z);
                assert_eq!(qubits, &vec![0, 1]);
            }
            _ => panic!("Expected MeasureReset"),
        }
    }
    #[test]
    fn test_depolarize1_parsing() {
        let circuit = StimCircuit::from_str("DEPOLARIZE1(0.01) 0 1 2").unwrap();
        assert_eq!(circuit.instructions.len(), 1);
        match &circuit.instructions[0] {
            StimInstruction::Depolarize1 {
                probability,
                qubits,
            } => {
                assert!((probability - 0.01).abs() < 1e-10);
                assert_eq!(qubits, &vec![0, 1, 2]);
            }
            _ => panic!("Expected Depolarize1"),
        }
    }
    #[test]
    fn test_x_error_parsing() {
        let circuit = StimCircuit::from_str("X_ERROR(0.05) 0 1").unwrap();
        assert_eq!(circuit.instructions.len(), 1);
        match &circuit.instructions[0] {
            StimInstruction::XError {
                probability,
                qubits,
            } => {
                assert!((probability - 0.05).abs() < 1e-10);
                assert_eq!(qubits, &vec![0, 1]);
            }
            _ => panic!("Expected XError"),
        }
    }
    #[test]
    fn test_pauli_channel_1_parsing() {
        let circuit = StimCircuit::from_str("PAULI_CHANNEL_1(0.01,0.02,0.03) 0 1").unwrap();
        assert_eq!(circuit.instructions.len(), 1);
        match &circuit.instructions[0] {
            StimInstruction::PauliChannel1 { px, py, pz, qubits } => {
                assert!((px - 0.01).abs() < 1e-10);
                assert!((py - 0.02).abs() < 1e-10);
                assert!((pz - 0.03).abs() < 1e-10);
                assert_eq!(qubits, &vec![0, 1]);
            }
            _ => panic!("Expected PauliChannel1"),
        }
    }
    #[test]
    fn test_correlated_error_parsing() {
        let circuit = StimCircuit::from_str("CORRELATED_ERROR(0.1) X0 Y1 Z2").unwrap();
        assert_eq!(circuit.instructions.len(), 1);
        match &circuit.instructions[0] {
            StimInstruction::CorrelatedError {
                probability,
                targets,
            } => {
                assert!((probability - 0.1).abs() < 1e-10);
                assert_eq!(targets.len(), 3);
                assert_eq!(targets[0].pauli, PauliType::X);
                assert_eq!(targets[0].qubit, 0);
                assert_eq!(targets[1].pauli, PauliType::Y);
                assert_eq!(targets[1].qubit, 1);
                assert_eq!(targets[2].pauli, PauliType::Z);
                assert_eq!(targets[2].qubit, 2);
            }
            _ => panic!("Expected CorrelatedError"),
        }
    }
    #[test]
    fn test_shift_coords_parsing() {
        let circuit = StimCircuit::from_str("SHIFT_COORDS 1.0 2.0 3.0").unwrap();
        assert_eq!(circuit.instructions.len(), 1);
        match &circuit.instructions[0] {
            StimInstruction::ShiftCoords { shifts } => {
                assert_eq!(shifts.len(), 3);
                assert!((shifts[0] - 1.0).abs() < 1e-10);
                assert!((shifts[1] - 2.0).abs() < 1e-10);
                assert!((shifts[2] - 3.0).abs() < 1e-10);
            }
            _ => panic!("Expected ShiftCoords"),
        }
    }
    #[test]
    fn test_full_error_correction_circuit() {
        let circuit_str = r#"
            # Surface code preparation
            H 0
            CNOT 0 1
            CNOT 0 2
            M 1 2
            DETECTOR rec[-1] rec[-2]
            OBSERVABLE_INCLUDE(0) rec[-1]
            X_ERROR(0.01) 0
            Z_ERROR(0.01) 1 2
        "#;
        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        assert!(circuit.instructions.len() >= 7);
        let has_detector = circuit
            .instructions
            .iter()
            .any(|inst| matches!(inst, StimInstruction::Detector { .. }));
        assert!(has_detector);
        let has_observable = circuit
            .instructions
            .iter()
            .any(|inst| matches!(inst, StimInstruction::ObservableInclude { .. }));
        assert!(has_observable);
    }
    #[test]
    fn test_e_shorthand_parsing() {
        let circuit = StimCircuit::from_str("E(0.1) X0 Y1 Z2").unwrap();
        assert_eq!(circuit.instructions.len(), 1);
        match &circuit.instructions[0] {
            StimInstruction::CorrelatedError {
                probability,
                targets,
            } => {
                assert!((probability - 0.1).abs() < 1e-10);
                assert_eq!(targets.len(), 3);
                assert_eq!(targets[0].pauli, PauliType::X);
                assert_eq!(targets[0].qubit, 0);
            }
            _ => panic!("Expected CorrelatedError"),
        }
    }
    #[test]
    fn test_else_correlated_error_parsing() {
        let circuit = StimCircuit::from_str("ELSE_CORRELATED_ERROR(0.2) X0 Z1").unwrap();
        assert_eq!(circuit.instructions.len(), 1);
        match &circuit.instructions[0] {
            StimInstruction::ElseCorrelatedError {
                probability,
                targets,
            } => {
                assert!((probability - 0.2).abs() < 1e-10);
                assert_eq!(targets.len(), 2);
                assert_eq!(targets[0].pauli, PauliType::X);
                assert_eq!(targets[0].qubit, 0);
                assert_eq!(targets[1].pauli, PauliType::Z);
                assert_eq!(targets[1].qubit, 1);
            }
            _ => panic!("Expected ElseCorrelatedError"),
        }
    }
    #[test]
    fn test_e_else_chain() {
        let circuit_str = r#"
            E(0.1) X0
            ELSE_CORRELATED_ERROR(0.2) Y0
        "#;
        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        assert_eq!(circuit.instructions.len(), 2);
        assert!(matches!(
            &circuit.instructions[0],
            StimInstruction::CorrelatedError { .. }
        ));
        assert!(matches!(
            &circuit.instructions[1],
            StimInstruction::ElseCorrelatedError { .. }
        ));
    }
}
