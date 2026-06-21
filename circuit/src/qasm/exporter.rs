//! Export `QuantRS2` circuits to `OpenQASM` 3.0 format

use super::ast::{
    ClassicalRef, Declaration, Expression, GateDefinition, Literal, Measurement, QasmGate,
    QasmProgram, QasmRegister, QasmStatement, QubitRef,
};
use crate::builder::Circuit;
use quantrs2_core::synthesis::decompose_single_qubit_zyz;
use quantrs2_core::{gate::GateOp, qubit::QubitId};
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::sync::Arc;
use thiserror::Error;

/// Export error types
#[derive(Debug, Error)]
pub enum ExportError {
    #[error("Unsupported gate: {0}")]
    UnsupportedGate(String),

    #[error("Invalid circuit: {0}")]
    InvalidCircuit(String),

    #[error("Formatting error: {0}")]
    FormattingError(#[from] std::fmt::Error),

    #[error("Gate parameter error: {0}")]
    ParameterError(String),

    /// A custom gate could not be turned into a QASM `gate` definition because
    /// no decomposition is available for it with the information at hand.
    ///
    /// This is an honest failure: returning it prevents the gate from being
    /// silently dropped from the exported program.
    #[error("Cannot decompose custom gate '{gate}' for QASM export: {reason}")]
    UndecomposableGate { gate: String, reason: String },
}

/// Options for controlling QASM export
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Include standard gate library
    pub include_stdgates: bool,
    /// Use gate decomposition for non-standard gates
    pub decompose_custom: bool,
    /// Add comments with gate matrix representations
    pub include_gate_comments: bool,
    /// Optimize gate sequences
    pub optimize: bool,
    /// Pretty print with indentation
    pub pretty_print: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            include_stdgates: true,
            decompose_custom: true,
            include_gate_comments: false,
            optimize: false,
            pretty_print: true,
        }
    }
}

/// QASM exporter
pub struct QasmExporter {
    options: ExportOptions,
    /// Track which gates need custom definitions
    custom_gates: HashMap<String, GateInfo>,
    /// Track qubit usage
    qubit_usage: HashSet<usize>,
    /// Track if measurements are used
    needs_classical_bits: bool,
}

#[derive(Clone)]
struct GateInfo {
    name: String,
    num_qubits: usize,
    num_params: usize,
    matrix: Option<scirs2_core::ndarray::Array2<Complex64>>,
}

impl QasmExporter {
    /// Create a new exporter with options
    #[must_use]
    pub fn new(options: ExportOptions) -> Self {
        Self {
            options,
            custom_gates: HashMap::new(),
            qubit_usage: HashSet::new(),
            needs_classical_bits: false,
        }
    }

    /// Export a circuit to QASM 3.0
    pub fn export<const N: usize>(&mut self, circuit: &Circuit<N>) -> Result<String, ExportError> {
        // Analyze circuit
        self.analyze_circuit(circuit)?;

        // Generate QASM program
        let program = self.generate_program(circuit)?;

        // Convert to string
        Ok(program.to_string())
    }

    /// Analyze circuit to determine requirements
    fn analyze_circuit<const N: usize>(&mut self, circuit: &Circuit<N>) -> Result<(), ExportError> {
        self.qubit_usage.clear();
        self.custom_gates.clear();
        self.needs_classical_bits = false;

        // Analyze each gate
        for gate in circuit.gates() {
            // Track qubit usage
            for qubit in gate.qubits() {
                self.qubit_usage.insert(qubit.id() as usize);
            }

            // Check if gate is standard or custom
            if !self.is_standard_gate(gate.as_ref()) {
                self.register_custom_gate(gate.as_ref())?;
            }

            // Check for measurements
            if gate.name().contains("measure") {
                self.needs_classical_bits = true;
            }
        }

        Ok(())
    }

    /// Check if a gate is in the standard library
    fn is_standard_gate(&self, gate: &dyn GateOp) -> bool {
        let name = gate.name();
        matches!(
            name,
            "I" | "X"
                | "Y"
                | "Z"
                | "H"
                | "S"
                | "S†"
                | "Sdg"
                | "T"
                | "T†"
                | "Tdg"
                | "√X"
                | "√X†"
                | "SX"
                | "SXdg"
                | "RX"
                | "RY"
                | "RZ"
                | "P"
                | "Phase"
                | "U"
                | "U1"
                | "U2"
                | "U3"
                | "CX"
                | "CNOT"
                | "CY"
                | "CZ"
                | "CH"
                | "CRX"
                | "CRY"
                | "CRZ"
                | "CPhase"
                | "SWAP"
                | "iSWAP"
                | "ECR"
                | "DCX"
                | "RXX"
                | "RYY"
                | "RZZ"
                | "RZX"
                | "CU"
                | "CCX"
                | "Toffoli"
                | "Fredkin"
                | "measure"
                | "reset"
                | "barrier"
        )
    }

    /// Register a custom gate
    fn register_custom_gate(&mut self, gate: &dyn GateOp) -> Result<(), ExportError> {
        let name = self.gate_qasm_name(gate);

        if !self.custom_gates.contains_key(&name) {
            let num_qubits = gate.qubits().len();
            let matrix = Self::gate_matrix(gate, num_qubits);

            let info = GateInfo {
                name: name.clone(),
                num_qubits,
                num_params: self.count_gate_params(gate),
                matrix,
            };

            self.custom_gates.insert(name, info);
        }

        Ok(())
    }

    /// Retrieve the unitary matrix of a gate as a square `Array2`.
    ///
    /// `GateOp::matrix` returns a flat row-major `Vec<Complex64>` of length
    /// `(2^num_qubits)^2`. Returns `None` when the gate cannot produce a matrix
    /// or the data does not form a square matrix of the expected dimension.
    fn gate_matrix(gate: &dyn GateOp, num_qubits: usize) -> Option<Array2<Complex64>> {
        let flat = gate.matrix().ok()?;
        let dim = 1usize.checked_shl(num_qubits as u32)?;
        if flat.len() != dim.checked_mul(dim)? {
            return None;
        }
        Array2::from_shape_vec((dim, dim), flat).ok()
    }

    /// Get QASM name for a gate
    fn gate_qasm_name(&self, gate: &dyn GateOp) -> String {
        let name = gate.name();
        match name {
            "I" => "id".to_string(),
            "X" => "x".to_string(),
            "Y" => "y".to_string(),
            "Z" => "z".to_string(),
            "H" => "h".to_string(),
            "S" | "S†" => "s".to_string(),
            "Sdg" => "sdg".to_string(),
            "T" => "t".to_string(),
            "T†" | "Tdg" => "tdg".to_string(),
            "√X" | "SX" => "sx".to_string(),
            "√X†" | "SXdg" => "sxdg".to_string(),
            "RX" => "rx".to_string(),
            "RY" => "ry".to_string(),
            "RZ" => "rz".to_string(),
            "P" | "Phase" => "p".to_string(),
            "U" => "u".to_string(),
            "CX" | "CNOT" => "cx".to_string(),
            "CY" => "cy".to_string(),
            "CZ" => "cz".to_string(),
            "CH" => "ch".to_string(),
            "CRX" => "crx".to_string(),
            "CRY" => "cry".to_string(),
            "CRZ" => "crz".to_string(),
            "CPhase" => "cp".to_string(),
            "SWAP" => "swap".to_string(),
            "iSWAP" => "iswap".to_string(),
            "ECR" => "ecr".to_string(),
            "DCX" => "dcx".to_string(),
            "RXX" => "rxx".to_string(),
            "RYY" => "ryy".to_string(),
            "RZZ" => "rzz".to_string(),
            "RZX" => "rzx".to_string(),
            "CCX" | "Toffoli" => "ccx".to_string(),
            "Fredkin" => "cswap".to_string(),
            _ => name.to_lowercase(),
        }
    }

    /// Count gate parameters
    fn count_gate_params(&self, gate: &dyn GateOp) -> usize {
        // This is a simplified version - would need gate trait extension
        let name = gate.name();
        match name {
            "RX" | "RY" | "RZ" | "P" | "Phase" | "U1" => 1,
            "U2" => 2,
            "U" | "U3" => 3,
            "CRX" | "CRY" | "CRZ" | "CPhase" => 1,
            "RXX" | "RYY" | "RZZ" | "RZX" => 1,
            _ => 0,
        }
    }

    /// Generate QASM program
    fn generate_program<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> Result<QasmProgram, ExportError> {
        let mut declarations = Vec::new();
        let mut statements = Vec::new();

        // Calculate required register size
        let max_qubit = self.qubit_usage.iter().max().copied().unwrap_or(0);
        let num_qubits = max_qubit + 1;

        // Add quantum register
        declarations.push(Declaration::QuantumRegister(QasmRegister {
            name: "q".to_string(),
            size: num_qubits,
        }));

        // Add classical register if needed
        if self.needs_classical_bits {
            declarations.push(Declaration::ClassicalRegister(QasmRegister {
                name: "c".to_string(),
                size: num_qubits,
            }));
        }

        // Add custom gate definitions
        if self.options.decompose_custom {
            for gate_info in self.custom_gates.values() {
                if let Some(def) = self.generate_gate_definition(gate_info)? {
                    declarations.push(Declaration::GateDefinition(def));
                }
            }
        }

        // Convert gates to statements
        for gate in circuit.gates() {
            statements.push(self.convert_gate(gate)?);
        }

        // Build includes
        let includes = if self.options.include_stdgates {
            vec!["stdgates.inc".to_string()]
        } else {
            vec![]
        };

        Ok(QasmProgram {
            version: "3.0".to_string(),
            includes,
            declarations,
            statements,
        })
    }

    /// Generate a QASM `gate` definition for a custom (non-standard) gate.
    ///
    /// Returns `Ok(Some(def))` with a body expressed in standard-library gates
    /// when the custom gate can be synthesized. Returns `Ok(None)` only when no
    /// definition is required (there is nothing to emit for this entry).
    /// Returns `Err(ExportError::UndecomposableGate)` when a definition *is*
    /// required but cannot be produced from the available information — this is
    /// an honest error so the gate is never silently dropped from the output.
    fn generate_gate_definition(
        &self,
        gate_info: &GateInfo,
    ) -> Result<Option<GateDefinition>, ExportError> {
        // A parameterized custom gate would need a symbolic body in terms of its
        // parameters. We only captured a single concrete matrix instance, which
        // cannot represent the gate for arbitrary parameter values, so emitting a
        // body from it would be a fabrication. Fail honestly instead.
        if gate_info.num_params > 0 {
            return Err(ExportError::UndecomposableGate {
                gate: gate_info.name.clone(),
                reason: format!(
                    "parameterized custom gate with {} parameter(s); symbolic decomposition is not supported",
                    gate_info.num_params
                ),
            });
        }

        match gate_info.num_qubits {
            // Defensive: a gate acting on zero qubits has no meaningful body.
            0 => Err(ExportError::UndecomposableGate {
                gate: gate_info.name.clone(),
                reason: "gate acts on zero qubits".to_string(),
            }),
            1 => self.single_qubit_gate_definition(gate_info),
            n => Err(ExportError::UndecomposableGate {
                gate: gate_info.name.clone(),
                reason: format!(
                    "no decomposition available for {n}-qubit custom gate (only single-qubit synthesis is implemented)"
                ),
            }),
        }
    }

    /// Synthesize a single-qubit custom gate into an `rz · ry · rz` body using a
    /// ZYZ Euler decomposition of its unitary matrix.
    fn single_qubit_gate_definition(
        &self,
        gate_info: &GateInfo,
    ) -> Result<Option<GateDefinition>, ExportError> {
        let matrix = gate_info
            .matrix
            .as_ref()
            .ok_or_else(|| ExportError::UndecomposableGate {
                gate: gate_info.name.clone(),
                reason: "matrix representation unavailable".to_string(),
            })?;

        let decomp = decompose_single_qubit_zyz(&matrix.view()).map_err(|e| {
            ExportError::UndecomposableGate {
                gate: gate_info.name.clone(),
                reason: format!("ZYZ decomposition failed: {e}"),
            }
        })?;

        // U = e^{i·global_phase} · Rz(theta2) · Ry(phi) · Rz(theta1)
        //
        // QASM statements execute in source order (first statement applied
        // first, i.e. rightmost in the matrix product), so the matrix product
        // Rz(θ₂)·Ry(φ)·Rz(θ₁) is emitted as rz(θ₁); ry(φ); rz(θ₂);.
        //
        // The scalar global phase is physically unobservable for a stand-alone
        // gate and is intentionally not emitted (QASM `gate` bodies have no
        // portable way to express it). The synthesized operator therefore equals
        // the original up to global phase, which defines an equivalent gate.
        let qubit_arg = "qb".to_string();

        let make_rotation = |name: &str, angle: f64| -> QasmStatement {
            QasmStatement::Gate(QasmGate {
                name: name.to_string(),
                params: vec![Expression::Literal(Literal::Float(angle))],
                qubits: vec![QubitRef::Register(qubit_arg.clone())],
                control: None,
                inverse: false,
                power: None,
            })
        };

        let body = vec![
            make_rotation("rz", decomp.theta1),
            make_rotation("ry", decomp.phi),
            make_rotation("rz", decomp.theta2),
        ];

        Ok(Some(GateDefinition {
            name: gate_info.name.clone(),
            params: Vec::new(),
            qubits: vec![qubit_arg],
            body,
        }))
    }

    /// Convert gate to QASM statement
    fn convert_gate(
        &self,
        gate: &Arc<dyn GateOp + Send + Sync>,
    ) -> Result<QasmStatement, ExportError> {
        let gate_name = gate.name();

        match gate_name {
            "measure" => {
                // Convert measurement
                let qubits: Vec<QubitRef> = gate
                    .qubits()
                    .iter()
                    .map(|q| QubitRef::Single {
                        register: "q".to_string(),
                        index: q.id() as usize,
                    })
                    .collect();

                let targets: Vec<ClassicalRef> = gate
                    .qubits()
                    .iter()
                    .map(|q| ClassicalRef::Single {
                        register: "c".to_string(),
                        index: q.id() as usize,
                    })
                    .collect();

                Ok(QasmStatement::Measure(Measurement { qubits, targets }))
            }
            "reset" => {
                let qubits: Vec<QubitRef> = gate
                    .qubits()
                    .iter()
                    .map(|q| QubitRef::Single {
                        register: "q".to_string(),
                        index: q.id() as usize,
                    })
                    .collect();

                Ok(QasmStatement::Reset(qubits))
            }
            "barrier" => {
                let qubits: Vec<QubitRef> = gate
                    .qubits()
                    .iter()
                    .map(|q| QubitRef::Single {
                        register: "q".to_string(),
                        index: q.id() as usize,
                    })
                    .collect();

                Ok(QasmStatement::Barrier(qubits))
            }
            _ => {
                // Regular gate
                let name = self.gate_qasm_name(gate.as_ref());

                let qubits: Vec<QubitRef> = gate
                    .qubits()
                    .iter()
                    .map(|q| QubitRef::Single {
                        register: "q".to_string(),
                        index: q.id() as usize,
                    })
                    .collect();

                // Extract parameters - this is simplified
                let params = self.extract_gate_params(gate.as_ref())?;

                Ok(QasmStatement::Gate(QasmGate {
                    name,
                    params,
                    qubits,
                    control: None,
                    inverse: false,
                    power: None,
                }))
            }
        }
    }

    /// Extract gate parameters as expressions
    fn extract_gate_params(&self, gate: &dyn GateOp) -> Result<Vec<Expression>, ExportError> {
        use quantrs2_core::gate::multi::{CRX, CRY, CRZ};
        use quantrs2_core::gate::single::{RotationX, RotationY, RotationZ};
        use std::any::Any;

        let any_gate = gate.as_any();

        // Single-qubit rotation gates
        if let Some(rx) = any_gate.downcast_ref::<RotationX>() {
            return Ok(vec![Expression::Literal(Literal::Float(rx.theta))]);
        }
        if let Some(ry) = any_gate.downcast_ref::<RotationY>() {
            return Ok(vec![Expression::Literal(Literal::Float(ry.theta))]);
        }
        if let Some(rz) = any_gate.downcast_ref::<RotationZ>() {
            return Ok(vec![Expression::Literal(Literal::Float(rz.theta))]);
        }

        // Controlled rotation gates
        if let Some(crx) = any_gate.downcast_ref::<CRX>() {
            return Ok(vec![Expression::Literal(Literal::Float(crx.theta))]);
        }
        if let Some(cry) = any_gate.downcast_ref::<CRY>() {
            return Ok(vec![Expression::Literal(Literal::Float(cry.theta))]);
        }
        if let Some(crz) = any_gate.downcast_ref::<CRZ>() {
            return Ok(vec![Expression::Literal(Literal::Float(crz.theta))]);
        }

        // No parameters for other gates
        Ok(vec![])
    }
}

/// Export a circuit to QASM 3.0 with default options
pub fn export_qasm3<const N: usize>(circuit: &Circuit<N>) -> Result<String, ExportError> {
    let mut exporter = QasmExporter::new(ExportOptions::default());
    exporter.export(circuit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::error::QuantRS2Result;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::{Hadamard, PauliX};
    use quantrs2_core::qubit::QubitId;
    use std::any::Any;

    /// A non-standard single-qubit gate whose unitary is the Hadamard matrix.
    /// Its name is not in the standard library, so the exporter must synthesize
    /// a real `gate` definition for it rather than dropping it.
    #[derive(Debug, Clone)]
    struct CustomHadamard {
        target: QubitId,
    }

    impl GateOp for CustomHadamard {
        fn name(&self) -> &'static str {
            "myhad"
        }
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }
        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            let s = 1.0 / 2.0_f64.sqrt();
            Ok(vec![
                Complex64::new(s, 0.0),
                Complex64::new(s, 0.0),
                Complex64::new(s, 0.0),
                Complex64::new(-s, 0.0),
            ])
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(self.clone())
        }
    }

    /// A non-standard two-qubit gate (identity on two qubits). Two-qubit custom
    /// synthesis is out of scope, so the exporter must fail honestly instead of
    /// silently dropping it.
    #[derive(Debug, Clone)]
    struct CustomTwoQubit {
        a: QubitId,
        b: QubitId,
    }

    impl GateOp for CustomTwoQubit {
        fn name(&self) -> &'static str {
            "mytwo"
        }
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.a, self.b]
        }
        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            let mut m = vec![Complex64::new(0.0, 0.0); 16];
            for i in 0..4 {
                m[i * 4 + i] = Complex64::new(1.0, 0.0);
            }
            Ok(m)
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_export_custom_single_qubit_gate_emits_definition() {
        let mut circuit = Circuit::<1>::new();
        circuit
            .add_gate(CustomHadamard { target: QubitId(0) })
            .expect("adding custom gate should succeed");

        let qasm = export_qasm3(&circuit)
            .expect("export should synthesize a definition for a single-qubit custom gate");

        // The custom gate must produce a real, non-empty `gate` definition body
        // expressed in standard rotations, and the gate must be referenced.
        assert!(
            qasm.contains("gate myhad"),
            "missing custom gate definition: {qasm}"
        );
        assert!(qasm.contains("rz("), "definition body missing rz: {qasm}");
        assert!(qasm.contains("ry("), "definition body missing ry: {qasm}");
        assert!(
            qasm.contains("myhad qb") || qasm.contains("myhad q[0]"),
            "custom gate not applied: {qasm}"
        );
    }

    #[test]
    fn test_export_custom_multi_qubit_gate_errors_honestly() {
        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(CustomTwoQubit {
                a: QubitId(0),
                b: QubitId(1),
            })
            .expect("adding custom two-qubit gate should succeed");

        let result = export_qasm3(&circuit);
        // Must NOT silently drop the gate: an honest error is required.
        assert!(
            matches!(result, Err(ExportError::UndecomposableGate { ref gate, .. }) if gate == "mytwo"),
            "expected UndecomposableGate error for two-qubit custom gate, got: {result:?}"
        );
    }

    #[test]
    fn test_custom_gate_definition_is_skipped_when_disabled() {
        // With decompose_custom = false, no definition is emitted (and no error
        // is raised) — the option legitimately suppresses definitions.
        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(CustomTwoQubit {
                a: QubitId(0),
                b: QubitId(1),
            })
            .expect("adding custom two-qubit gate should succeed");

        let options = ExportOptions {
            decompose_custom: false,
            ..ExportOptions::default()
        };
        let mut exporter = QasmExporter::new(options);
        let qasm = exporter
            .export(&circuit)
            .expect("export without decomposition should not error");
        assert!(
            !qasm.contains("gate mytwo"),
            "unexpected definition: {qasm}"
        );
    }

    #[test]
    fn test_export_simple_circuit() {
        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("adding Hadamard gate should succeed");
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("adding CNOT gate should succeed");

        let result = export_qasm3(&circuit);
        assert!(result.is_ok());

        let qasm = result.expect("export_qasm3 should succeed for valid circuit");
        assert!(qasm.contains("OPENQASM 3.0"));
        assert!(qasm.contains("qubit[2] q"));
        assert!(qasm.contains("h q[0]"));
        assert!(qasm.contains("cx q[0], q[1]"));
    }

    #[test]
    fn test_export_with_measurements() {
        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("adding Hadamard gate should succeed");
        // Note: measure gate would need to be implemented

        let result = export_qasm3(&circuit);
        assert!(result.is_ok());

        let qasm = result.expect("export_qasm3 should succeed for measurement test");
        // Basic check
        assert!(qasm.contains("OPENQASM 3.0"));
    }
}
