//! Export `QuantRS2` circuits to OpenQASM 2.0 format
//!
//! Produces valid `OPENQASM 2.0` programs with `qelib1.inc` and standard
//! gate names as per the OpenQASM 2.0 specification.

use super::error::QasmError;
use crate::builder::Circuit;
use quantrs2_core::gate::GateOp;
use quantrs2_core::qubit::QubitId;
use std::fmt::Write as FmtWrite;
use std::sync::Arc;

// ─── gate-name helpers (shared with import) ────────────────────────────────

/// Return the `qelib1.inc` gate mnemonic for a `GateOp` name.
/// Returns `None` for names that have no direct QASM 2.0 counterpart.
pub fn gate_name_to_qasm2(name: &str) -> Option<&'static str> {
    match name {
        "I" => Some("id"),
        "X" => Some("x"),
        "Y" => Some("y"),
        "Z" => Some("z"),
        "H" => Some("h"),
        "S" => Some("s"),
        "S†" => Some("sdg"),
        "T" => Some("t"),
        "T†" => Some("tdg"),
        "√X" => Some("sx"),
        "√X†" => Some("sxdg"),
        "RX" => Some("rx"),
        "RY" => Some("ry"),
        "RZ" => Some("rz"),
        // P(λ) == u1(λ) in qelib1
        "P" => Some("u1"),
        // U(θ,φ,λ) == u3(θ,φ,λ) in qelib1
        "U" => Some("u3"),
        "CNOT" | "CX" => Some("cx"),
        "CY" => Some("cy"),
        "CZ" => Some("cz"),
        "CH" => Some("ch"),
        "CRX" => Some("crx"),
        "CRY" => Some("cry"),
        "CRZ" => Some("crz"),
        "CS" => Some("cs"),
        "SWAP" => Some("swap"),
        "CCX" | "Toffoli" => Some("ccx"),
        "Fredkin" => Some("cswap"),
        _ => None,
    }
}

/// How many (angle) parameters does a qelib1 gate take?
pub fn qasm2_gate_param_count(qasm_name: &str) -> usize {
    match qasm_name {
        "rx" | "ry" | "rz" | "u1" | "p" | "crx" | "cry" | "crz" => 1,
        "u2" => 2,
        "u3" => 3,
        _ => 0,
    }
}

/// How many qubits does a qelib1 gate take?
pub fn qasm2_gate_qubit_count(qasm_name: &str) -> usize {
    match qasm_name {
        "id" | "x" | "y" | "z" | "h" | "s" | "sdg" | "t" | "tdg" | "sx" | "sxdg" | "rx"
        | "ry" | "rz" | "u1" | "u2" | "u3" | "p" => 1,
        "cx" | "cy" | "cz" | "ch" | "crx" | "cry" | "crz" | "cp" | "cs" | "swap" => 2,
        "ccx" | "cswap" => 3,
        _ => 1, // default guess
    }
}

// ─── parameter extraction ───────────────────────────────────────────────────

/// Extract angle parameters from a gate via downcast.
pub fn extract_params(gate: &dyn GateOp) -> Vec<f64> {
    use quantrs2_core::gate::multi::{CRX, CRY, CRZ};
    use quantrs2_core::gate::single::{PGate, RotationX, RotationY, RotationZ, UGate};

    let any = gate.as_any();

    if let Some(g) = any.downcast_ref::<RotationX>() {
        return vec![g.theta];
    }
    if let Some(g) = any.downcast_ref::<RotationY>() {
        return vec![g.theta];
    }
    if let Some(g) = any.downcast_ref::<RotationZ>() {
        return vec![g.theta];
    }
    if let Some(g) = any.downcast_ref::<UGate>() {
        return vec![g.theta, g.phi, g.lambda];
    }
    if let Some(g) = any.downcast_ref::<PGate>() {
        return vec![g.lambda];
    }
    if let Some(g) = any.downcast_ref::<CRX>() {
        return vec![g.theta];
    }
    if let Some(g) = any.downcast_ref::<CRY>() {
        return vec![g.theta];
    }
    if let Some(g) = any.downcast_ref::<CRZ>() {
        return vec![g.theta];
    }
    vec![]
}

// ─── exporter ───────────────────────────────────────────────────────────────

/// Convert a single gate arc to its QASM 2.0 statement string.
///
/// `reg_name` is the quantum register name (usually `"q"`).
fn gate_to_qasm2_line(
    gate: &Arc<dyn GateOp + Send + Sync>,
    reg_name: &str,
) -> Result<Option<String>, QasmError> {
    let name = gate.name();

    // measurements handled specially
    if name == "measure" {
        let qubits = gate.qubits();
        if qubits.is_empty() {
            return Ok(None);
        }
        // Emit one measure per qubit: measure q[i] -> c[i];
        let mut lines = String::new();
        for q in &qubits {
            writeln!(
                lines,
                "measure {}[{}] -> c[{}];",
                reg_name,
                q.id(),
                q.id()
            )?;
        }
        // Trim trailing newline so caller can add its own
        return Ok(Some(lines.trim_end_matches('\n').to_string()));
    }

    if name == "reset" {
        let qubits = gate.qubits();
        let mut lines = String::new();
        for q in &qubits {
            writeln!(lines, "reset {}[{}];", reg_name, q.id())?;
        }
        return Ok(Some(lines.trim_end_matches('\n').to_string()));
    }

    if name == "barrier" {
        let qubits = gate.qubits();
        if qubits.is_empty() {
            return Ok(None);
        }
        let args: Vec<String> = qubits
            .iter()
            .map(|q| format!("{}[{}]", reg_name, q.id()))
            .collect();
        return Ok(Some(format!("barrier {};", args.join(", "))));
    }

    let qasm_name = gate_name_to_qasm2(name).ok_or_else(|| QasmError::UnsupportedGate(name.to_string()))?;

    let params = extract_params(gate.as_ref());
    let qubits = gate.qubits();

    let mut out = String::new();
    if params.is_empty() {
        write!(out, "{}", qasm_name)?;
    } else {
        write!(out, "{}(", qasm_name)?;
        for (i, p) in params.iter().enumerate() {
            if i > 0 {
                write!(out, ",")?;
            }
            write!(out, "{:.10}", p)?;
        }
        write!(out, ")")?;
    }

    // qubit args
    for q in &qubits {
        write!(out, " {}[{}],", reg_name, q.id())?;
    }
    // remove trailing comma, add semicolon
    if out.ends_with(',') {
        out.pop();
    }
    out.push(';');

    Ok(Some(out))
}

/// Export a `Circuit<N>` to an OpenQASM 2.0 string.
///
/// # Errors
///
/// Returns `QasmError` if any gate in the circuit has no QASM 2.0 equivalent,
/// or if a formatting error occurs.
pub fn circuit_to_qasm<const N: usize>(circuit: &Circuit<N>) -> Result<String, QasmError> {
    let gates = circuit.gates();

    // Determine the highest qubit index used
    let max_qubit = gates
        .iter()
        .flat_map(|g| g.qubits())
        .map(|q| q.id() as usize)
        .max();

    let num_qubits = match max_qubit {
        Some(m) => m + 1,
        None => {
            // No gates at all — use N as the qubit count
            N
        }
    };
    let num_qubits = num_qubits.max(N);

    // Detect if any gate is a measurement (need creg)
    let has_measure = gates
        .iter()
        .any(|g| matches!(g.name(), "measure" | "Measure"));

    let reg = "q";

    let mut out = String::new();
    writeln!(out, "OPENQASM 2.0;")?;
    writeln!(out, "include \"qelib1.inc\";")?;
    writeln!(out, "qreg {}[{}];", reg, num_qubits)?;
    if has_measure {
        writeln!(out, "creg c[{}];", num_qubits)?;
    }

    for gate in gates {
        if let Some(line) = gate_to_qasm2_line(gate, reg)? {
            writeln!(out, "{}", line)?;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::{Hadamard, PauliX, Phase, RotationX, T};
    use quantrs2_core::qubit::QubitId;

    #[test]
    fn test_qasm2_header() {
        let circuit = Circuit::<2>::new();
        let qasm = circuit_to_qasm(&circuit).expect("export should succeed");
        assert!(qasm.starts_with("OPENQASM 2.0;"), "wrong header: {}", qasm);
        assert!(qasm.contains("qelib1.inc"), "missing include: {}", qasm);
    }

    #[test]
    fn test_qasm2_registers() {
        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("add H");
        let qasm = circuit_to_qasm(&circuit).expect("export");
        assert!(qasm.contains("qreg q[2]"), "qreg missing: {}", qasm);
    }

    #[test]
    fn test_qasm2_single_qubit_gates() {
        let mut circuit = Circuit::<2>::new();
        circuit.add_gate(Hadamard { target: QubitId(0) }).expect("H");
        circuit.add_gate(PauliX { target: QubitId(1) }).expect("X");
        circuit.add_gate(Phase { target: QubitId(0) }).expect("S");
        circuit.add_gate(T { target: QubitId(1) }).expect("T");
        let qasm = circuit_to_qasm(&circuit).expect("export");
        assert!(qasm.contains("h q[0];"), "H missing: {}", qasm);
        assert!(qasm.contains("x q[1];"), "X missing: {}", qasm);
        assert!(qasm.contains("s q[0];"), "S missing: {}", qasm);
        assert!(qasm.contains("t q[1];"), "T missing: {}", qasm);
    }

    #[test]
    fn test_qasm2_cnot() {
        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("CNOT");
        let qasm = circuit_to_qasm(&circuit).expect("export");
        assert!(qasm.contains("cx q[0], q[1];"), "cx missing: {}", qasm);
    }

    #[test]
    fn test_qasm2_rotation() {
        let mut circuit = Circuit::<1>::new();
        circuit
            .add_gate(RotationX {
                target: QubitId(0),
                theta: std::f64::consts::PI / 2.0,
            })
            .expect("RX");
        let qasm = circuit_to_qasm(&circuit).expect("export");
        assert!(qasm.contains("rx("), "rx missing: {}", qasm);
    }
}
