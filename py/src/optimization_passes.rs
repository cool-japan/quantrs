//! Quantum circuit optimization passes for Python.
//!
//! This module provides various optimization passes for quantum circuits,
//! including gate cancellation, commutation, synthesis, and depth reduction.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;
use scirs2_numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use std::collections::{HashMap, HashSet, VecDeque};

// Gate representation for optimization
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationGate {
    // Single-qubit gates
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    SDag(usize),
    T(usize),
    TDag(usize),
    RX(usize, f64),
    RY(usize, f64),
    RZ(usize, f64),
    U1(usize, f64),
    U2(usize, f64, f64),
    U3(usize, f64, f64, f64),

    // Two-qubit gates
    CNOT(usize, usize),
    CY(usize, usize),
    CZ(usize, usize),
    SWAP(usize, usize),
    CRX(usize, usize, f64),
    CRY(usize, usize, f64),
    CRZ(usize, usize, f64),

    // Three-qubit gates
    Toffoli(usize, usize, usize),
    Fredkin(usize, usize, usize),
}

impl OptimizationGate {
    fn qubits(&self) -> Vec<usize> {
        match self {
            Self::H(q)
            | Self::X(q)
            | Self::Y(q)
            | Self::Z(q)
            | Self::S(q)
            | Self::SDag(q)
            | Self::T(q)
            | Self::TDag(q)
            | Self::RX(q, _)
            | Self::RY(q, _)
            | Self::RZ(q, _)
            | Self::U1(q, _)
            | Self::U2(q, _, _)
            | Self::U3(q, _, _, _) => vec![*q],

            Self::CNOT(c, t)
            | Self::CY(c, t)
            | Self::CZ(c, t)
            | Self::CRX(c, t, _)
            | Self::CRY(c, t, _)
            | Self::CRZ(c, t, _) => vec![*c, *t],

            Self::SWAP(q1, q2) => vec![*q1, *q2],

            Self::Toffoli(c1, c2, t) => vec![*c1, *c2, *t],
            Self::Fredkin(c, t1, t2) => vec![*c, *t1, *t2],
        }
    }

    fn is_inverse_of(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::H(q1), Self::H(q2)) => q1 == q2,
            (Self::X(q1), Self::X(q2)) => q1 == q2,
            (Self::Y(q1), Self::Y(q2)) => q1 == q2,
            (Self::Z(q1), Self::Z(q2)) => q1 == q2,
            (Self::S(q1), Self::SDag(q2)) => q1 == q2,
            (Self::SDag(q1), Self::S(q2)) => q1 == q2,
            (Self::T(q1), Self::TDag(q2)) => q1 == q2,
            (Self::TDag(q1), Self::T(q2)) => q1 == q2,
            (Self::CNOT(c1, t1), Self::CNOT(c2, t2)) => c1 == c2 && t1 == t2,
            (Self::CY(c1, t1), Self::CY(c2, t2)) => c1 == c2 && t1 == t2,
            (Self::CZ(c1, t1), Self::CZ(c2, t2)) => c1 == c2 && t1 == t2,
            (Self::SWAP(q1, q2), Self::SWAP(q3, q4)) => {
                (q1 == q3 && q2 == q4) || (q1 == q4 && q2 == q3)
            }
            _ => false,
        }
    }

    fn commutes_with(&self, other: &Self) -> bool {
        let self_qubits: HashSet<_> = self.qubits().into_iter().collect();
        let other_qubits: HashSet<_> = other.qubits().into_iter().collect();

        // Gates on disjoint qubits always commute
        if self_qubits.is_disjoint(&other_qubits) {
            return true;
        }

        // Check specific commutation rules
        match (self, other) {
            // Z-basis gates commute
            (Self::Z(q1), Self::Z(q2)) => q1 == q2,
            (Self::Z(q1), Self::S(q2)) => q1 == q2,
            (Self::Z(q1), Self::SDag(q2)) => q1 == q2,
            (Self::Z(q1), Self::T(q2)) => q1 == q2,
            (Self::Z(q1), Self::TDag(q2)) => q1 == q2,
            (Self::Z(q1), Self::RZ(q2, _)) => q1 == q2,

            // CNOT commutation rules
            (Self::CNOT(c1, t1), Self::CNOT(c2, t2)) => {
                // CNOT gates commute if they share control but different targets
                (c1 == c2 && t1 != t2) ||
                // Or if first target equals second control and vice versa
                (t1 == c2 && c1 == t2)
            }

            // CZ gates always commute with themselves
            (Self::CZ(_, _), Self::CZ(_, _)) => true,

            _ => false,
        }
    }
}

// Circuit optimization context
pub struct OptimizationContext {
    gates: Vec<OptimizationGate>,
    n_qubits: usize,
}

impl OptimizationContext {
    const fn new(n_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            n_qubits,
        }
    }

    fn add_gate(&mut self, gate: OptimizationGate) {
        self.gates.push(gate);
    }

    fn apply_pass(&mut self, pass: &dyn OptimizationPass) -> usize {
        pass.optimize(self)
    }
}

// Trait for optimization passes
trait OptimizationPass {
    fn optimize(&self, context: &mut OptimizationContext) -> usize;
    fn name(&self) -> &str;
}

// Gate cancellation pass
struct GateCancellationPass;

impl OptimizationPass for GateCancellationPass {
    fn optimize(&self, context: &mut OptimizationContext) -> usize {
        let mut new_gates = Vec::new();
        let mut removed = 0;
        let mut i = 0;

        while i < context.gates.len() {
            if i + 1 < context.gates.len() {
                let gate1 = &context.gates[i];
                let gate2 = &context.gates[i + 1];

                if gate1.is_inverse_of(gate2) {
                    // Skip both gates
                    i += 2;
                    removed += 2;
                    continue;
                }
            }

            new_gates.push(context.gates[i].clone());
            i += 1;
        }

        context.gates = new_gates;
        removed
    }

    fn name(&self) -> &'static str {
        "GateCancellation"
    }
}

// Commutation pass
struct CommutationPass;

impl OptimizationPass for CommutationPass {
    fn optimize(&self, context: &mut OptimizationContext) -> usize {
        let mut changed = 0;
        let mut made_change = true;

        while made_change {
            made_change = false;

            for i in 0..context.gates.len().saturating_sub(1) {
                let gate1 = &context.gates[i];
                let gate2 = &context.gates[i + 1];

                // Try to move single-qubit gates before two-qubit gates
                if gate1.qubits().len() > gate2.qubits().len() && gate1.commutes_with(gate2) {
                    context.gates.swap(i, i + 1);
                    changed += 1;
                    made_change = true;
                }
            }
        }

        changed
    }

    fn name(&self) -> &'static str {
        "Commutation"
    }
}

// Merge rotation gates pass
struct MergeRotationsPass;

impl OptimizationPass for MergeRotationsPass {
    fn optimize(&self, context: &mut OptimizationContext) -> usize {
        let mut new_gates = Vec::new();
        let mut merged = 0;
        let mut i = 0;

        while i < context.gates.len() {
            if i + 1 < context.gates.len() {
                let merged_gate = match (&context.gates[i], &context.gates[i + 1]) {
                    (OptimizationGate::RX(q1, a1), OptimizationGate::RX(q2, a2)) if q1 == q2 => {
                        merged += 1;
                        Some(OptimizationGate::RX(*q1, a1 + a2))
                    }
                    (OptimizationGate::RY(q1, a1), OptimizationGate::RY(q2, a2)) if q1 == q2 => {
                        merged += 1;
                        Some(OptimizationGate::RY(*q1, a1 + a2))
                    }
                    (OptimizationGate::RZ(q1, a1), OptimizationGate::RZ(q2, a2)) if q1 == q2 => {
                        merged += 1;
                        Some(OptimizationGate::RZ(*q1, a1 + a2))
                    }
                    _ => None,
                };

                if let Some(gate) = merged_gate {
                    new_gates.push(gate);
                    i += 2;
                    continue;
                }
            }

            new_gates.push(context.gates[i].clone());
            i += 1;
        }

        context.gates = new_gates;
        merged
    }

    fn name(&self) -> &'static str {
        "MergeRotations"
    }
}

// Decompose multi-qubit gates pass
struct DecomposePass {
    basis_gates: HashSet<String>,
}

impl OptimizationPass for DecomposePass {
    fn optimize(&self, context: &mut OptimizationContext) -> usize {
        let mut new_gates = Vec::new();
        let mut decomposed = 0;

        for gate in &context.gates {
            match gate {
                OptimizationGate::Toffoli(c1, c2, t) => {
                    if self.basis_gates.contains("toffoli") {
                        new_gates.push(gate.clone());
                    } else {
                        // Decompose Toffoli into CNOT and single-qubit gates
                        new_gates.push(OptimizationGate::H(*t));
                        new_gates.push(OptimizationGate::CNOT(*c2, *t));
                        new_gates.push(OptimizationGate::TDag(*t));
                        new_gates.push(OptimizationGate::CNOT(*c1, *t));
                        new_gates.push(OptimizationGate::T(*t));
                        new_gates.push(OptimizationGate::CNOT(*c2, *t));
                        new_gates.push(OptimizationGate::TDag(*t));
                        new_gates.push(OptimizationGate::CNOT(*c1, *t));
                        new_gates.push(OptimizationGate::T(*c2));
                        new_gates.push(OptimizationGate::T(*t));
                        new_gates.push(OptimizationGate::H(*t));
                        new_gates.push(OptimizationGate::CNOT(*c1, *c2));
                        new_gates.push(OptimizationGate::T(*c1));
                        new_gates.push(OptimizationGate::TDag(*c2));
                        new_gates.push(OptimizationGate::CNOT(*c1, *c2));
                        decomposed += 1;
                    }
                }
                OptimizationGate::SWAP(q1, q2) => {
                    if self.basis_gates.contains("swap") {
                        new_gates.push(gate.clone());
                    } else {
                        // Decompose SWAP into 3 CNOTs
                        new_gates.push(OptimizationGate::CNOT(*q1, *q2));
                        new_gates.push(OptimizationGate::CNOT(*q2, *q1));
                        new_gates.push(OptimizationGate::CNOT(*q1, *q2));
                        decomposed += 1;
                    }
                }
                _ => new_gates.push(gate.clone()),
            }
        }

        context.gates = new_gates;
        decomposed
    }

    fn name(&self) -> &'static str {
        "Decompose"
    }
}

/// Circuit optimizer for Python
#[pyclass(name = "CircuitTranspiler")]
pub struct PyCircuitTranspiler {
    optimization_level: u8,
    basis_gates: HashSet<String>,
}

#[pymethods]
impl PyCircuitTranspiler {
    #[new]
    #[pyo3(signature = (optimization_level=2, basis_gates=None))]
    fn new(optimization_level: u8, basis_gates: Option<Vec<String>>) -> Self {
        let default_basis = vec![
            "h".to_string(),
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
            "s".to_string(),
            "sdg".to_string(),
            "t".to_string(),
            "tdg".to_string(),
            "rx".to_string(),
            "ry".to_string(),
            "rz".to_string(),
            "cnot".to_string(),
            "cy".to_string(),
            "cz".to_string(),
        ];

        let basis_gates = basis_gates.unwrap_or(default_basis).into_iter().collect();

        Self {
            optimization_level,
            basis_gates,
        }
    }

    /// Optimize a quantum circuit
    fn optimize(&self, py: Python, circuit: &Bound<'_, PyList>) -> PyResult<(PyObject, PyObject)> {
        let mut context = OptimizationContext::new(10); // Default to 10 qubits

        // Parse input circuit
        for gate_info in circuit {
            let gate_tuple: (String, Vec<usize>, Option<Vec<f64>>) = gate_info.extract()?;
            let (gate_name, qubits, params) = gate_tuple;

            let gate = match gate_name.as_str() {
                "h" => OptimizationGate::H(qubits[0]),
                "x" => OptimizationGate::X(qubits[0]),
                "y" => OptimizationGate::Y(qubits[0]),
                "z" => OptimizationGate::Z(qubits[0]),
                "s" => OptimizationGate::S(qubits[0]),
                "sdg" => OptimizationGate::SDag(qubits[0]),
                "t" => OptimizationGate::T(qubits[0]),
                "tdg" => OptimizationGate::TDag(qubits[0]),
                "rx" => OptimizationGate::RX(qubits[0], params.unwrap()[0]),
                "ry" => OptimizationGate::RY(qubits[0], params.unwrap()[0]),
                "rz" => OptimizationGate::RZ(qubits[0], params.unwrap()[0]),
                "cnot" => OptimizationGate::CNOT(qubits[0], qubits[1]),
                "cy" => OptimizationGate::CY(qubits[0], qubits[1]),
                "cz" => OptimizationGate::CZ(qubits[0], qubits[1]),
                "swap" => OptimizationGate::SWAP(qubits[0], qubits[1]),
                "toffoli" => OptimizationGate::Toffoli(qubits[0], qubits[1], qubits[2]),
                _ => return Err(PyValueError::new_err(format!("Unknown gate: {gate_name}"))),
            };

            context.add_gate(gate);
        }

        // Apply optimization passes based on level
        let mut stats = PyDict::new(py);

        if self.optimization_level >= 1 {
            // Level 1: Basic optimizations
            let removed = context.apply_pass(&GateCancellationPass);
            stats.set_item("gates_cancelled", removed)?;

            let merged = context.apply_pass(&MergeRotationsPass);
            stats.set_item("rotations_merged", merged)?;
        }

        if self.optimization_level >= 2 {
            // Level 2: Commutation
            let commuted = context.apply_pass(&CommutationPass);
            stats.set_item("gates_commuted", commuted)?;
        }

        if self.optimization_level >= 3 {
            // Level 3: Decomposition
            let decompose_pass = DecomposePass {
                basis_gates: self.basis_gates.clone(),
            };
            let decomposed = context.apply_pass(&decompose_pass);
            stats.set_item("gates_decomposed", decomposed)?;
        }

        // Convert optimized circuit back to Python format
        let optimized_circuit = PyList::empty(py);
        for gate in context.gates {
            let (name, qubits, params) = match gate {
                OptimizationGate::H(q) => ("h", vec![q], None),
                OptimizationGate::X(q) => ("x", vec![q], None),
                OptimizationGate::Y(q) => ("y", vec![q], None),
                OptimizationGate::Z(q) => ("z", vec![q], None),
                OptimizationGate::S(q) => ("s", vec![q], None),
                OptimizationGate::SDag(q) => ("sdg", vec![q], None),
                OptimizationGate::T(q) => ("t", vec![q], None),
                OptimizationGate::TDag(q) => ("tdg", vec![q], None),
                OptimizationGate::RX(q, a) => ("rx", vec![q], Some(vec![a])),
                OptimizationGate::RY(q, a) => ("ry", vec![q], Some(vec![a])),
                OptimizationGate::RZ(q, a) => ("rz", vec![q], Some(vec![a])),
                OptimizationGate::CNOT(c, t) => ("cnot", vec![c, t], None),
                OptimizationGate::CY(c, t) => ("cy", vec![c, t], None),
                OptimizationGate::CZ(c, t) => ("cz", vec![c, t], None),
                OptimizationGate::SWAP(q1, q2) => ("swap", vec![q1, q2], None),
                OptimizationGate::Toffoli(c1, c2, t) => ("toffoli", vec![c1, c2, t], None),
                _ => continue,
            };

            let gate_tuple = (name, qubits, params);
            optimized_circuit.append(gate_tuple)?;
        }

        Ok((optimized_circuit.into(), stats.into()))
    }

    /// Analyze circuit depth
    fn analyze_depth(&self, circuit: &Bound<'_, PyList>) -> PyResult<usize> {
        let mut qubit_depths = HashMap::new();
        let mut max_depth = 0;

        for gate_info in circuit {
            let gate_tuple: (String, Vec<usize>) = gate_info.extract()?;
            let (_, qubits) = gate_tuple;

            // Find maximum depth among involved qubits
            let current_depth = qubits
                .iter()
                .map(|q| qubit_depths.get(q).unwrap_or(&0))
                .max()
                .unwrap_or(&0)
                + 1;

            // Update depth for all involved qubits
            for q in qubits {
                qubit_depths.insert(q, current_depth);
            }

            max_depth = max_depth.max(current_depth);
        }

        Ok(max_depth)
    }

    /// Count gates by type
    fn gate_counts(&self, py: Python, circuit: &Bound<'_, PyList>) -> PyResult<PyObject> {
        let mut counts = HashMap::new();

        for gate_info in circuit {
            let gate_tuple: (String, Vec<usize>) = gate_info.extract()?;
            let (gate_name, _) = gate_tuple;

            *counts.entry(gate_name).or_insert(0) += 1;
        }

        let py_dict = PyDict::new(py);
        for (gate, count) in counts {
            py_dict.set_item(gate, count)?;
        }

        Ok(py_dict.into())
    }
}

/// Routing optimization for mapping to device topology
#[pyclass(name = "DeviceRouter")]
pub struct PyDeviceRouter {
    coupling_map: Vec<(usize, usize)>,
    n_physical_qubits: usize,
}

#[pymethods]
impl PyDeviceRouter {
    #[new]
    fn new(coupling_map: Vec<(usize, usize)>) -> PyResult<Self> {
        let n_physical_qubits = coupling_map
            .iter()
            .flat_map(|(a, b)| vec![*a, *b])
            .max()
            .unwrap_or(0)
            + 1;

        Ok(Self {
            coupling_map,
            n_physical_qubits,
        })
    }

    /// Route a circuit to device topology
    fn route(
        &self,
        py: Python,
        circuit: &Bound<'_, PyList>,
        initial_layout: Option<Vec<usize>>,
    ) -> PyResult<(PyObject, Vec<usize>)> {
        // This is a simplified routing algorithm
        // In reality, would use more sophisticated algorithms like SABRE

        let n_logical_qubits = self.extract_n_qubits(circuit)?;

        // Create initial layout
        let layout = initial_layout.unwrap_or_else(|| (0..n_logical_qubits).collect());

        // For now, just return the circuit with SWAPs inserted where needed
        let routed_circuit = PyList::empty(py);

        for gate_info in circuit {
            let gate_tuple: (String, Vec<usize>, Option<Vec<f64>>) = gate_info.extract()?;
            let (gate_name, logical_qubits, params) = gate_tuple;

            // Map logical to physical qubits
            let physical_qubits: Vec<usize> = logical_qubits.iter().map(|&q| layout[q]).collect();

            // Check if two-qubit gate is allowed
            if physical_qubits.len() == 2 {
                let (p0, p1) = (physical_qubits[0], physical_qubits[1]);
                let connected =
                    self.coupling_map.contains(&(p0, p1)) || self.coupling_map.contains(&(p1, p0));

                if !connected {
                    // Need to insert SWAPs (simplified - just fail for now)
                    return Err(PyValueError::new_err(format!(
                        "No connection between physical qubits {p0} and {p1}"
                    )));
                }
            }

            routed_circuit.append((gate_name, physical_qubits, params))?;
        }

        Ok((routed_circuit.into(), layout))
    }

    /// Extract number of logical qubits from circuit
    fn extract_n_qubits(&self, circuit: &Bound<'_, PyList>) -> PyResult<usize> {
        let mut max_qubit = 0;

        for gate_info in circuit {
            let gate_tuple: (String, Vec<usize>) = gate_info.extract()?;
            let (_, qubits) = gate_tuple;

            if let Some(&max) = qubits.iter().max() {
                max_qubit = max_qubit.max(max);
            }
        }

        Ok(max_qubit + 1)
    }
}

/// Register the optimization module
pub fn register_optimization_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(m.py(), "optimization")?;

    submodule.add_class::<PyCircuitTranspiler>()?;
    submodule.add_class::<PyDeviceRouter>()?;

    m.add_submodule(&submodule)?;
    Ok(())
}
