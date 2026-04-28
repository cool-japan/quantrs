//! Python bindings for QuantRS2 ecosystem integration.
//!
//! Provides Python access to:
//! - OpenQASM 2.0 export (`circuit_to_qasm`) and import (`qasm_to_circuit_ops`)
//! - PennyLane JSON device execution (`execute_pennylane_circuit`)

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Gate operation tuple: (gate_name, qubit_indices, parameters)
type GateOp = (String, Vec<u32>, Vec<f64>);

// ─── QASM 2.0 bindings ───────────────────────────────────────────────────────

/// Export a `PyCircuit` to an OpenQASM 2.0 string.
///
/// Parameters
/// ----------
/// circuit : PyCircuit
///     The circuit to export.
///
/// Returns
/// -------
/// str
///     A valid `OPENQASM 2.0` program string.
///
/// Raises
/// ------
/// ValueError
///     If any gate in the circuit has no QASM 2.0 equivalent.
#[pyfunction]
pub fn circuit_to_qasm(circuit: &crate::circuit_core::PyCircuit) -> PyResult<String> {
    use quantrs2_circuit::qasm::circuit_to_qasm as rust_circuit_to_qasm;
    use quantrs2_sim::dynamic::DynamicCircuit;

    let dynamic = circuit.circuit.as_ref().ok_or_else(|| {
        PyValueError::new_err("circuit is not initialized")
    })?;

    // Dispatch to the const-generic Circuit through DynamicCircuit arms
    macro_rules! export_arm {
        ($c:ident) => {{
            rust_circuit_to_qasm($c).map_err(|e| PyValueError::new_err(e.to_string()))
        }};
    }

    match dynamic {
        DynamicCircuit::Q2(c) => export_arm!(c),
        DynamicCircuit::Q3(c) => export_arm!(c),
        DynamicCircuit::Q4(c) => export_arm!(c),
        DynamicCircuit::Q5(c) => export_arm!(c),
        DynamicCircuit::Q6(c) => export_arm!(c),
        DynamicCircuit::Q7(c) => export_arm!(c),
        DynamicCircuit::Q8(c) => export_arm!(c),
        DynamicCircuit::Q9(c) => export_arm!(c),
        DynamicCircuit::Q10(c) => export_arm!(c),
        DynamicCircuit::Q12(c) => export_arm!(c),
        DynamicCircuit::Q16(c) => export_arm!(c),
        DynamicCircuit::Q20(c) => export_arm!(c),
        DynamicCircuit::Q24(c) => export_arm!(c),
        DynamicCircuit::Q32(c) => export_arm!(c),
    }
}

/// Parse an OpenQASM 2.0 string and return gate metadata.
///
/// This function parses a QASM 2.0 program and returns a list of
/// `(gate_name, qubit_indices, params)` tuples describing each gate.
///
/// Parameters
/// ----------
/// qasm : str
///     A valid `OPENQASM 2.0` program string.
///
/// Returns
/// -------
/// tuple[list[tuple[str, list[int], list[float]]], int]
///     A pair `(ops, num_qubits)` where:
///     - `ops` is a list of `(name, wires, params)` tuples
///     - `num_qubits` is the size of the quantum register
///
/// Raises
/// ------
/// ValueError
///     If the QASM string is invalid or contains unsupported gates.
#[pyfunction]
pub fn qasm_to_circuit_ops(
    qasm: &str,
) -> PyResult<(Vec<GateOp>, usize)> {
    use quantrs2_circuit::qasm::qasm_to_gates;
    use quantrs2_circuit::qasm::export::extract_params;

    let (gates, num_qubits) =
        qasm_to_gates(qasm).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let ops: Vec<GateOp> = gates
        .iter()
        .map(|g| {
            let name = g.name().to_string();
            let qubits: Vec<u32> = g.qubits().iter().map(|q| q.0).collect();
            let params: Vec<f64> = extract_params(g.as_ref());
            (name, qubits, params)
        })
        .collect();

    Ok((ops, num_qubits))
}

// ─── PennyLane device bindings ───────────────────────────────────────────────

/// Execute a PennyLane circuit via the QuantRS2 state-vector simulator.
///
/// Parameters
/// ----------
/// json_input : str
///     A JSON string describing the circuit in PennyLane's device protocol:
///
///     .. code-block:: json
///
///         {
///           "num_wires": 2,
///           "operations": [
///             {"name": "Hadamard", "wires": [0], "params": []},
///             {"name": "CNOT",     "wires": [0, 1], "params": []}
///           ],
///           "observables": [
///             {"name": "PauliZ", "wires": [0]}
///           ]
///         }
///
/// Returns
/// -------
/// str
///     A JSON string with fields:
///     - ``probabilities``: list of probabilities for each basis state
///     - ``state_re`` / ``state_im``: real and imaginary parts of the state vector
///     - ``expval``: list of expectation values (one per observable)
///
/// Raises
/// ------
/// ValueError
///     If the JSON is malformed or the circuit contains unsupported operations.
#[pyfunction]
pub fn execute_pennylane_circuit(json_input: &str) -> PyResult<String> {
    use quantrs2_sim::pennylane::QuantRS2Device;

    let device = QuantRS2Device::new();
    device
        .execute_json(json_input)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Execute a PennyLane circuit and return structured Python objects.
///
/// Parameters
/// ----------
/// json_input : str
///     PennyLane device JSON payload.
///
/// Returns
/// -------
/// dict
///     Dictionary with keys:
///     - ``probabilities`` (list[float])
///     - ``state_re`` (list[float])
///     - ``state_im`` (list[float])
///     - ``expval`` (list[float])
///
/// Raises
/// ------
/// ValueError
///     If the JSON is malformed or the circuit contains unsupported operations.
#[pyfunction]
pub fn execute_pennylane_circuit_dict(py: Python<'_>, json_input: &str) -> PyResult<Py<pyo3::types::PyDict>> {
    use pyo3::types::PyDict;
    use quantrs2_sim::pennylane::QuantRS2Device;

    let device = QuantRS2Device::new();
    let circuit: quantrs2_sim::pennylane::PennyLaneCircuit =
        serde_json::from_str(json_input)
            .map_err(|e| PyValueError::new_err(format!("JSON parse error: {e}")))?;

    let result = device
        .execute(&circuit)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("probabilities", result.probabilities.clone())?;
    d.set_item("state_re", result.state_re.clone())?;
    d.set_item("state_im", result.state_im.clone())?;
    d.set_item("expval", result.expval)?;

    Ok(d.into())
}

// ─── module registration ─────────────────────────────────────────────────────

/// Register the `ecosystem` submodule on the parent module `m`.
pub fn register_ecosystem_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let ecosystem = pyo3::types::PyModule::new(m.py(), "ecosystem")?;

    // QASM 2.0 functions
    ecosystem.add_function(wrap_pyfunction!(circuit_to_qasm, &ecosystem)?)?;
    ecosystem.add_function(wrap_pyfunction!(qasm_to_circuit_ops, &ecosystem)?)?;

    // PennyLane functions
    ecosystem.add_function(wrap_pyfunction!(execute_pennylane_circuit, &ecosystem)?)?;
    ecosystem.add_function(wrap_pyfunction!(execute_pennylane_circuit_dict, &ecosystem)?)?;

    m.add_submodule(&ecosystem)?;

    Ok(())
}
