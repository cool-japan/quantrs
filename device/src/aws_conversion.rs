use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::gate::{multi, single, GateOp};
use serde_json::json;
use std::collections::HashMap;

use crate::DeviceError;
use crate::DeviceResult;

/// Convert a Quantrs circuit to Amazon Braket IR format
pub fn circuit_to_braket_ir<const N: usize>(circuit: &Circuit<N>) -> DeviceResult<String> {
    let mut instructions = Vec::new();

    // Iterate through gates and convert to Braket IR
    for gate in circuit.gates() {
        let instruction = gate_to_braket_ir(gate)?;
        instructions.push(instruction);
    }

    // Create the full Braket IR program
    let braket_ir = json!({
        "braketSchemaHeader": {
            "name": "braket.ir.jaqcd.Program",
            "version": "1"
        },
        "results": [
            {
                "type": {
                    "measurements": {
                        "type": "measurements"
                    }
                }
            }
        ],
        "basis_rotation_instructions": [],
        "instructions": instructions
    });

    Ok(braket_ir.to_string())
}

/// Convert a Quantrs gate to Amazon Braket IR instruction
fn gate_to_braket_ir(gate: &Box<dyn GateOp>) -> DeviceResult<serde_json::Value> {
    match gate.name() {
        // Single-qubit gates
        "H" => {
            let target = gate.qubits()[0].id() as usize;
            Ok(json!({
                "type": "h",
                "target": target
            }))
        }
        "X" => {
            let target = gate.qubits()[0].id() as usize;
            Ok(json!({
                "type": "x",
                "target": target
            }))
        }
        "Y" => {
            let target = gate.qubits()[0].id() as usize;
            Ok(json!({
                "type": "y",
                "target": target
            }))
        }
        "Z" => {
            let target = gate.qubits()[0].id() as usize;
            Ok(json!({
                "type": "z",
                "target": target
            }))
        }
        "S" => {
            let target = gate.qubits()[0].id() as usize;
            Ok(json!({
                "type": "s",
                "target": target
            }))
        }
        "S†" => {
            let target = gate.qubits()[0].id() as usize;
            Ok(json!({
                "type": "si",
                "target": target
            }))
        }
        "T" => {
            let target = gate.qubits()[0].id() as usize;
            Ok(json!({
                "type": "t",
                "target": target
            }))
        }
        "T†" => {
            let target = gate.qubits()[0].id() as usize;
            Ok(json!({
                "type": "ti",
                "target": target
            }))
        }
        "√X" => {
            let target = gate.qubits()[0].id() as usize;
            Ok(json!({
                "type": "v",
                "target": target
            }))
        }
        "√X†" => {
            let target = gate.qubits()[0].id() as usize;
            Ok(json!({
                "type": "vi",
                "target": target
            }))
        }

        // Rotation gates
        "RX" => {
            if let Some(rx) = gate.as_any().downcast_ref::<single::RotationX>() {
                let target = rx.target.id() as usize;
                let angle = rx.theta;
                Ok(json!({
                    "type": "rx",
                    "target": target,
                    "angle": angle
                }))
            } else {
                Err(DeviceError::CircuitConversion(
                    "Failed to downcast RX gate".to_string(),
                ))
            }
        }
        "RY" => {
            if let Some(ry) = gate.as_any().downcast_ref::<single::RotationY>() {
                let target = ry.target.id() as usize;
                let angle = ry.theta;
                Ok(json!({
                    "type": "ry",
                    "target": target,
                    "angle": angle
                }))
            } else {
                Err(DeviceError::CircuitConversion(
                    "Failed to downcast RY gate".to_string(),
                ))
            }
        }
        "RZ" => {
            if let Some(rz) = gate.as_any().downcast_ref::<single::RotationZ>() {
                let target = rz.target.id() as usize;
                let angle = rz.theta;
                Ok(json!({
                    "type": "rz",
                    "target": target,
                    "angle": angle
                }))
            } else {
                Err(DeviceError::CircuitConversion(
                    "Failed to downcast RZ gate".to_string(),
                ))
            }
        }

        // Two-qubit gates
        "CNOT" => {
            if let Some(cnot) = gate.as_any().downcast_ref::<multi::CNOT>() {
                let control = cnot.control.id() as usize;
                let target = cnot.target.id() as usize;
                Ok(json!({
                    "type": "cnot",
                    "control": control,
                    "target": target
                }))
            } else {
                Err(DeviceError::CircuitConversion(
                    "Failed to downcast CNOT gate".to_string(),
                ))
            }
        }
        "CY" => {
            if let Some(cy) = gate.as_any().downcast_ref::<multi::CY>() {
                let control = cy.control.id() as usize;
                let target = cy.target.id() as usize;
                Ok(json!({
                    "type": "cy",
                    "control": control,
                    "target": target
                }))
            } else {
                Err(DeviceError::CircuitConversion(
                    "Failed to downcast CY gate".to_string(),
                ))
            }
        }
        "CZ" => {
            if let Some(cz) = gate.as_any().downcast_ref::<multi::CZ>() {
                let control = cz.control.id() as usize;
                let target = cz.target.id() as usize;
                Ok(json!({
                    "type": "cz",
                    "control": control,
                    "target": target
                }))
            } else {
                Err(DeviceError::CircuitConversion(
                    "Failed to downcast CZ gate".to_string(),
                ))
            }
        }
        "SWAP" => {
            if let Some(swap) = gate.as_any().downcast_ref::<multi::SWAP>() {
                let qubit1 = swap.qubit1.id() as usize;
                let qubit2 = swap.qubit2.id() as usize;
                Ok(json!({
                    "type": "swap",
                    "targets": [qubit1, qubit2]
                }))
            } else {
                Err(DeviceError::CircuitConversion(
                    "Failed to downcast SWAP gate".to_string(),
                ))
            }
        }

        // Controlled rotation gates
        "CRX" => {
            if let Some(crx) = gate.as_any().downcast_ref::<multi::CRX>() {
                let control = crx.control.id() as usize;
                let target = crx.target.id() as usize;
                let angle = crx.theta;
                Ok(json!({
                    "type": "crx",
                    "control": control,
                    "target": target,
                    "angle": angle
                }))
            } else {
                Err(DeviceError::CircuitConversion(
                    "Failed to downcast CRX gate".to_string(),
                ))
            }
        }
        "CRY" => {
            if let Some(cry) = gate.as_any().downcast_ref::<multi::CRY>() {
                let control = cry.control.id() as usize;
                let target = cry.target.id() as usize;
                let angle = cry.theta;
                Ok(json!({
                    "type": "cry",
                    "control": control,
                    "target": target,
                    "angle": angle
                }))
            } else {
                Err(DeviceError::CircuitConversion(
                    "Failed to downcast CRY gate".to_string(),
                ))
            }
        }
        "CRZ" => {
            if let Some(crz) = gate.as_any().downcast_ref::<multi::CRZ>() {
                let control = crz.control.id() as usize;
                let target = crz.target.id() as usize;
                let angle = crz.theta;
                Ok(json!({
                    "type": "crz",
                    "control": control,
                    "target": target,
                    "angle": angle
                }))
            } else {
                Err(DeviceError::CircuitConversion(
                    "Failed to downcast CRZ gate".to_string(),
                ))
            }
        }

        // Multi-qubit gates
        "Toffoli" => {
            if let Some(toffoli) = gate.as_any().downcast_ref::<multi::Toffoli>() {
                let control1 = toffoli.control1.id() as usize;
                let control2 = toffoli.control2.id() as usize;
                let target = toffoli.target.id() as usize;
                Ok(json!({
                    "type": "ccnot",
                    "controls": [control1, control2],
                    "target": target
                }))
            } else {
                Err(DeviceError::CircuitConversion(
                    "Failed to downcast Toffoli gate".to_string(),
                ))
            }
        }

        // Unsupported gate
        _ => Err(DeviceError::CircuitConversion(format!(
            "Gate {} not supported in Braket IR",
            gate.name()
        ))),
    }
}

/// Convert a Quantrs circuit to OpenQASM 2.0 format
pub fn circuit_to_qasm<const N: usize>(circuit: &Circuit<N>) -> DeviceResult<String> {
    let mut qasm = String::from("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\n");

    // Define the quantum and classical registers
    qasm.push_str(&format!("qreg q[{}];\n", N));
    qasm.push_str(&format!("creg c[{}];\n\n", N));

    // Map of gate conversions
    let gate_conversions: HashMap<&str, &str> = [
        ("H", "h"),
        ("X", "x"),
        ("Y", "y"),
        ("Z", "z"),
        ("S", "s"),
        ("S†", "sdg"),
        ("T", "t"),
        ("T†", "tdg"),
        ("√X", "sx"),
        ("√X†", "sxdg"),
        ("CNOT", "cx"),
        ("CY", "cy"),
        ("CZ", "cz"),
        ("SWAP", "swap"),
    ]
    .iter()
    .cloned()
    .collect();

    // Iterate through gates and convert to OpenQASM
    for gate in circuit.gates() {
        let gate_name = gate.name();

        if let Some(&qasm_name) = gate_conversions.get(gate_name) {
            let qubits = gate.qubits();

            if qubits.len() == 1 {
                // Single qubit gate
                qasm.push_str(&format!("{} q[{}];\n", qasm_name, qubits[0].id()));
            } else if qubits.len() == 2 {
                // Two qubit gate
                match gate_name {
                    "CNOT" => {
                        if let Some(cnot) = gate.as_any().downcast_ref::<multi::CNOT>() {
                            qasm.push_str(&format!(
                                "{} q[{}], q[{}];\n",
                                qasm_name,
                                cnot.control.id(),
                                cnot.target.id()
                            ));
                        }
                    }
                    "CY" | "CZ" => {
                        if let Some(cg) = gate.as_any().downcast_ref::<multi::CZ>() {
                            qasm.push_str(&format!(
                                "{} q[{}], q[{}];\n",
                                qasm_name,
                                cg.control.id(),
                                cg.target.id()
                            ));
                        }
                    }
                    "SWAP" => {
                        if let Some(swap) = gate.as_any().downcast_ref::<multi::SWAP>() {
                            qasm.push_str(&format!(
                                "{} q[{}], q[{}];\n",
                                qasm_name,
                                swap.qubit1.id(),
                                swap.qubit2.id()
                            ));
                        }
                    }
                    _ => {}
                }
            }
        } else {
            // Handle rotation gates
            match gate_name {
                "RX" => {
                    if let Some(rx) = gate.as_any().downcast_ref::<single::RotationX>() {
                        qasm.push_str(&format!("rx({}) q[{}];\n", rx.theta, rx.target.id()));
                    }
                }
                "RY" => {
                    if let Some(ry) = gate.as_any().downcast_ref::<single::RotationY>() {
                        qasm.push_str(&format!("ry({}) q[{}];\n", ry.theta, ry.target.id()));
                    }
                }
                "RZ" => {
                    if let Some(rz) = gate.as_any().downcast_ref::<single::RotationZ>() {
                        qasm.push_str(&format!("rz({}) q[{}];\n", rz.theta, rz.target.id()));
                    }
                }
                "CRX" => {
                    if let Some(crx) = gate.as_any().downcast_ref::<multi::CRX>() {
                        qasm.push_str(&format!(
                            "crx({}) q[{}], q[{}];\n",
                            crx.theta,
                            crx.control.id(),
                            crx.target.id()
                        ));
                    }
                }
                "CRY" => {
                    if let Some(cry) = gate.as_any().downcast_ref::<multi::CRY>() {
                        qasm.push_str(&format!(
                            "cry({}) q[{}], q[{}];\n",
                            cry.theta,
                            cry.control.id(),
                            cry.target.id()
                        ));
                    }
                }
                "CRZ" => {
                    if let Some(crz) = gate.as_any().downcast_ref::<multi::CRZ>() {
                        qasm.push_str(&format!(
                            "crz({}) q[{}], q[{}];\n",
                            crz.theta,
                            crz.control.id(),
                            crz.target.id()
                        ));
                    }
                }
                "Toffoli" => {
                    if let Some(toffoli) = gate.as_any().downcast_ref::<multi::Toffoli>() {
                        qasm.push_str(&format!(
                            "ccx q[{}], q[{}], q[{}];\n",
                            toffoli.control1.id(),
                            toffoli.control2.id(),
                            toffoli.target.id()
                        ));
                    }
                }
                _ => {
                    return Err(DeviceError::CircuitConversion(format!(
                        "Gate {} not supported in OpenQASM",
                        gate_name
                    )));
                }
            }
        }
    }

    // Add measurements
    for i in 0..N {
        qasm.push_str(&format!("measure q[{}] -> c[{}];\n", i, i));
    }

    Ok(qasm)
}
