//! Builder types for quantum circuits.
//!
//! This module contains the Circuit type for building and
//! executing quantum circuits.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Type alias for backwards compatibility
pub type CircuitBuilder<const N: usize> = Circuit<N>;

use quantrs2_core::{
    decomposition::{utils as decomp_utils, CompositeGate},
    error::QuantRS2Result,
    gate::{
        multi::{Fredkin, Toffoli, CH, CNOT, CRX, CRY, CRZ, CS, CY, CZ, SWAP},
        single::{
            Hadamard, PauliX, PauliY, PauliZ, Phase, PhaseDagger, RotationX, RotationY, RotationZ,
            SqrtX, SqrtXDagger, TDagger, T,
        },
        GateOp,
    },
    qubit::QubitId,
    register::Register,
};

use num_complex::Complex64;
use std::any::Any;
use std::collections::HashSet;

/// Circuit statistics for introspection and optimization
#[derive(Debug, Clone)]
pub struct CircuitStats {
    /// Total number of gates
    pub total_gates: usize,
    /// Gate counts by type
    pub gate_counts: HashMap<String, usize>,
    /// Circuit depth (sequential length)
    pub depth: usize,
    /// Number of two-qubit gates
    pub two_qubit_gates: usize,
    /// Number of multi-qubit gates (3+)
    pub multi_qubit_gates: usize,
    /// Gate density (gates per qubit)
    pub gate_density: f64,
    /// Number of qubits actually used
    pub used_qubits: usize,
    /// Total qubits available
    pub total_qubits: usize,
}

/// Gate pool for reusing common gates to reduce memory allocations
#[derive(Debug, Clone)]
pub struct GatePool {
    /// Common single-qubit gates that can be shared
    gates: HashMap<String, Arc<dyn GateOp + Send + Sync>>,
}

impl GatePool {
    /// Create a new gate pool with common gates pre-allocated
    pub fn new() -> Self {
        let mut gates = HashMap::with_capacity(16);

        // Pre-allocate common gates for different qubits
        for qubit_id in 0..32 {
            let qubit = QubitId::new(qubit_id);

            // Common single-qubit gates
            gates.insert(
                format!("H_{}", qubit_id),
                Arc::new(Hadamard { target: qubit }) as Arc<dyn GateOp + Send + Sync>,
            );
            gates.insert(
                format!("X_{}", qubit_id),
                Arc::new(PauliX { target: qubit }) as Arc<dyn GateOp + Send + Sync>,
            );
            gates.insert(
                format!("Y_{}", qubit_id),
                Arc::new(PauliY { target: qubit }) as Arc<dyn GateOp + Send + Sync>,
            );
            gates.insert(
                format!("Z_{}", qubit_id),
                Arc::new(PauliZ { target: qubit }) as Arc<dyn GateOp + Send + Sync>,
            );
            gates.insert(
                format!("S_{}", qubit_id),
                Arc::new(Phase { target: qubit }) as Arc<dyn GateOp + Send + Sync>,
            );
            gates.insert(
                format!("T_{}", qubit_id),
                Arc::new(T { target: qubit }) as Arc<dyn GateOp + Send + Sync>,
            );
        }

        Self { gates }
    }

    /// Get a gate from the pool if available, otherwise create new
    pub fn get_gate<G: GateOp + Clone + Send + Sync + 'static>(
        &mut self,
        gate: G,
    ) -> Arc<dyn GateOp + Send + Sync> {
        let key = format!("{}_{:?}", gate.name(), gate.qubits());

        if let Some(cached_gate) = self.gates.get(&key) {
            cached_gate.clone()
        } else {
            let arc_gate = Arc::new(gate) as Arc<dyn GateOp + Send + Sync>;
            self.gates.insert(key, arc_gate.clone());
            arc_gate
        }
    }
}

impl Default for GatePool {
    fn default() -> Self {
        Self::new()
    }
}

/// A placeholder measurement gate for QASM export
#[derive(Debug, Clone)]
pub struct Measure {
    pub target: QubitId,
}

impl GateOp for Measure {
    fn name(&self) -> &'static str {
        "measure"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.target]
    }

    fn is_parameterized(&self) -> bool {
        false
    }

    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        // Measurement doesn't have a unitary matrix representation
        Ok(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(self.clone())
    }
}

/// A quantum circuit with a fixed number of qubits
pub struct Circuit<const N: usize> {
    /// Vector of gates to be applied in sequence using Arc for shared ownership
    gates: Vec<Arc<dyn GateOp + Send + Sync>>,
    /// Gate pool for reusing common gates
    gate_pool: GatePool,
}

impl<const N: usize> Clone for Circuit<N> {
    fn clone(&self) -> Self {
        // With Arc, cloning is much more efficient - just clone the references
        Self {
            gates: self.gates.clone(),
            gate_pool: self.gate_pool.clone(),
        }
    }
}

impl<const N: usize> fmt::Debug for Circuit<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Circuit")
            .field("num_qubits", &N)
            .field("num_gates", &self.gates.len())
            .finish()
    }
}

impl<const N: usize> Circuit<N> {
    /// Create a new empty circuit with N qubits
    pub fn new() -> Self {
        Self {
            gates: Vec::with_capacity(64), // Pre-allocate capacity for better performance
            gate_pool: GatePool::new(),
        }
    }

    /// Create a new circuit with estimated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            gates: Vec::with_capacity(capacity),
            gate_pool: GatePool::new(),
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate<G: GateOp + Clone + Send + Sync + 'static>(
        &mut self,
        gate: G,
    ) -> QuantRS2Result<&mut Self> {
        // Validate that all qubits are within range
        for qubit in gate.qubits() {
            if qubit.id() as usize >= N {
                return Err(quantrs2_core::error::QuantRS2Error::InvalidInput(format!(
                    "Gate '{}' targets qubit {} which is out of range for {}-qubit circuit (valid range: 0-{})",
                    gate.name(),
                    qubit.id(),
                    N,
                    N - 1
                )));
            }
        }

        // Use gate pool for common gates to reduce memory allocations
        let gate_arc = self.gate_pool.get_gate(gate);
        self.gates.push(gate_arc);
        Ok(self)
    }

    /// Add a gate from an Arc (for copying gates between circuits)
    pub fn add_gate_arc(
        &mut self,
        gate: Arc<dyn GateOp + Send + Sync>,
    ) -> QuantRS2Result<&mut Self> {
        // Validate that all qubits are within range
        for qubit in gate.qubits() {
            if qubit.id() as usize >= N {
                return Err(quantrs2_core::error::QuantRS2Error::InvalidInput(format!(
                    "Gate '{}' targets qubit {} which is out of range for {}-qubit circuit (valid range: 0-{})",
                    gate.name(),
                    qubit.id(),
                    N,
                    N - 1
                )));
            }
        }

        self.gates.push(gate);
        Ok(self)
    }

    /// Get all gates in the circuit
    pub fn gates(&self) -> &[Arc<dyn GateOp + Send + Sync>] {
        &self.gates
    }

    /// Get gates as Vec for compatibility with existing optimization code
    pub fn gates_as_boxes(&self) -> Vec<Box<dyn GateOp>> {
        self.gates
            .iter()
            .map(|arc_gate| arc_gate.clone_gate())
            .collect()
    }

    /// Circuit introspection methods for optimization

    /// Count gates by type
    pub fn count_gates_by_type(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for gate in &self.gates {
            *counts.entry(gate.name().to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Calculate circuit depth (longest sequential path)
    pub fn calculate_depth(&self) -> usize {
        if self.gates.is_empty() {
            return 0;
        }

        // Track the last time each qubit was used
        let mut qubit_last_used = vec![0; N];
        let mut max_depth = 0;

        for (gate_idx, gate) in self.gates.iter().enumerate() {
            let gate_qubits = gate.qubits();

            // Find the maximum depth among all qubits this gate uses
            let gate_start_depth = gate_qubits
                .iter()
                .map(|q| qubit_last_used[q.id() as usize])
                .max()
                .unwrap_or(0);

            let gate_end_depth = gate_start_depth + 1;

            // Update the depth for all qubits this gate touches
            for qubit in gate_qubits {
                qubit_last_used[qubit.id() as usize] = gate_end_depth;
            }

            max_depth = max_depth.max(gate_end_depth);
        }

        max_depth
    }

    /// Count two-qubit gates
    pub fn count_two_qubit_gates(&self) -> usize {
        self.gates
            .iter()
            .filter(|gate| gate.qubits().len() == 2)
            .count()
    }

    /// Count multi-qubit gates (3 or more qubits)
    pub fn count_multi_qubit_gates(&self) -> usize {
        self.gates
            .iter()
            .filter(|gate| gate.qubits().len() >= 3)
            .count()
    }

    /// Calculate the critical path length (same as depth for now, but could be enhanced)
    pub fn calculate_critical_path(&self) -> usize {
        self.calculate_depth()
    }

    /// Calculate gate density (gates per qubit)
    pub fn calculate_gate_density(&self) -> f64 {
        if N == 0 {
            0.0
        } else {
            self.gates.len() as f64 / N as f64
        }
    }

    /// Get all unique qubits used in the circuit
    pub fn get_used_qubits(&self) -> HashSet<QubitId> {
        let mut used_qubits = HashSet::new();
        for gate in &self.gates {
            for qubit in gate.qubits() {
                used_qubits.insert(qubit);
            }
        }
        used_qubits
    }

    /// Check if the circuit uses all available qubits
    pub fn uses_all_qubits(&self) -> bool {
        self.get_used_qubits().len() == N
    }

    /// Get gates that operate on a specific qubit
    pub fn gates_on_qubit(&self, target_qubit: QubitId) -> Vec<&Arc<dyn GateOp + Send + Sync>> {
        self.gates
            .iter()
            .filter(|gate| gate.qubits().contains(&target_qubit))
            .collect()
    }

    /// Get gates between two indices (inclusive)
    pub fn gates_in_range(&self, start: usize, end: usize) -> &[Arc<dyn GateOp + Send + Sync>] {
        let end = end.min(self.gates.len().saturating_sub(1));
        let start = start.min(end);
        &self.gates[start..=end]
    }

    /// Check if circuit is empty
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// Get circuit statistics summary
    pub fn get_stats(&self) -> CircuitStats {
        let gate_counts = self.count_gates_by_type();
        let depth = self.calculate_depth();
        let two_qubit_gates = self.count_two_qubit_gates();
        let multi_qubit_gates = self.count_multi_qubit_gates();
        let gate_density = self.calculate_gate_density();
        let used_qubits = self.get_used_qubits().len();

        CircuitStats {
            total_gates: self.gates.len(),
            gate_counts,
            depth,
            two_qubit_gates,
            multi_qubit_gates,
            gate_density,
            used_qubits,
            total_qubits: N,
        }
    }

    /// Get the number of qubits in the circuit
    pub fn num_qubits(&self) -> usize {
        N
    }

    /// Get the number of gates in the circuit
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    /// Get the names of all gates in the circuit
    pub fn get_gate_names(&self) -> Vec<String> {
        self.gates
            .iter()
            .map(|gate| gate.name().to_string())
            .collect()
    }

    /// Get a qubit for a specific single-qubit gate by gate type and index
    pub fn get_single_qubit_for_gate(&self, gate_type: &str, index: usize) -> pyo3::PyResult<u32> {
        self.find_gate_by_type_and_index(gate_type, index)
            .and_then(|gate| {
                if gate.qubits().len() == 1 {
                    Some(gate.qubits()[0].id())
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Gate {} at index {} not found or is not a single-qubit gate",
                    gate_type, index
                ))
            })
    }

    /// Get rotation parameters (qubit, angle) for a specific gate by gate type and index
    pub fn get_rotation_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> pyo3::PyResult<(u32, f64)> {
        // Note: This is a simplified implementation, actual implementation would check
        // gate type and extract the rotation parameter
        self.find_gate_by_type_and_index(gate_type, index)
            .and_then(|gate| {
                if gate.qubits().len() == 1 {
                    // Default angle (in a real implementation, we would extract this from the gate)
                    Some((gate.qubits()[0].id(), 0.0))
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Gate {} at index {} not found or is not a rotation gate",
                    gate_type, index
                ))
            })
    }

    /// Get two-qubit parameters (control, target) for a specific gate by gate type and index
    pub fn get_two_qubit_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> pyo3::PyResult<(u32, u32)> {
        self.find_gate_by_type_and_index(gate_type, index)
            .and_then(|gate| {
                if gate.qubits().len() == 2 {
                    Some((gate.qubits()[0].id(), gate.qubits()[1].id()))
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Gate {} at index {} not found or is not a two-qubit gate",
                    gate_type, index
                ))
            })
    }

    /// Get controlled rotation parameters (control, target, angle) for a specific gate
    pub fn get_controlled_rotation_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> pyo3::PyResult<(u32, u32, f64)> {
        // Note: This is a simplified implementation, actual implementation would check
        // gate type and extract the rotation parameter
        self.find_gate_by_type_and_index(gate_type, index)
            .and_then(|gate| {
                if gate.qubits().len() == 2 {
                    // Default angle (in a real implementation, we would extract this from the gate)
                    Some((gate.qubits()[0].id(), gate.qubits()[1].id(), 0.0))
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Gate {} at index {} not found or is not a controlled rotation gate",
                    gate_type, index
                ))
            })
    }

    /// Get three-qubit parameters for gates like Toffoli or Fredkin
    pub fn get_three_qubit_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> pyo3::PyResult<(u32, u32, u32)> {
        self.find_gate_by_type_and_index(gate_type, index)
            .and_then(|gate| {
                if gate.qubits().len() == 3 {
                    Some((
                        gate.qubits()[0].id(),
                        gate.qubits()[1].id(),
                        gate.qubits()[2].id(),
                    ))
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Gate {} at index {} not found or is not a three-qubit gate",
                    gate_type, index
                ))
            })
    }

    /// Helper method to find a gate by type and index
    fn find_gate_by_type_and_index(&self, gate_type: &str, index: usize) -> Option<&dyn GateOp> {
        let mut count = 0;
        for gate in &self.gates {
            if gate.name() == gate_type {
                if count == index {
                    return Some(gate.as_ref());
                }
                count += 1;
            }
        }
        None
    }

    /// Apply a Hadamard gate to a qubit
    pub fn h(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(Hadamard {
            target: target.into(),
        })
    }

    /// Apply a Pauli-X gate to a qubit
    pub fn x(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(PauliX {
            target: target.into(),
        })
    }

    /// Apply a Pauli-Y gate to a qubit
    pub fn y(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(PauliY {
            target: target.into(),
        })
    }

    /// Apply a Pauli-Z gate to a qubit
    pub fn z(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(PauliZ {
            target: target.into(),
        })
    }

    /// Apply a rotation around X-axis
    pub fn rx(&mut self, target: impl Into<QubitId>, theta: f64) -> QuantRS2Result<&mut Self> {
        self.add_gate(RotationX {
            target: target.into(),
            theta,
        })
    }

    /// Apply a rotation around Y-axis
    pub fn ry(&mut self, target: impl Into<QubitId>, theta: f64) -> QuantRS2Result<&mut Self> {
        self.add_gate(RotationY {
            target: target.into(),
            theta,
        })
    }

    /// Apply a rotation around Z-axis
    pub fn rz(&mut self, target: impl Into<QubitId>, theta: f64) -> QuantRS2Result<&mut Self> {
        self.add_gate(RotationZ {
            target: target.into(),
            theta,
        })
    }

    /// Apply a Phase gate (S gate)
    pub fn s(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(Phase {
            target: target.into(),
        })
    }

    /// Apply a Phase-dagger gate (S† gate)
    pub fn sdg(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(PhaseDagger {
            target: target.into(),
        })
    }

    /// Apply a T gate
    pub fn t(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(T {
            target: target.into(),
        })
    }

    /// Apply a T-dagger gate (T† gate)
    pub fn tdg(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(TDagger {
            target: target.into(),
        })
    }

    /// Apply a Square Root of X gate (√X)
    pub fn sx(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(SqrtX {
            target: target.into(),
        })
    }

    /// Apply a Square Root of X Dagger gate (√X†)
    pub fn sxdg(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(SqrtXDagger {
            target: target.into(),
        })
    }

    /// Apply a CNOT gate
    pub fn cnot(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CNOT {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a CNOT gate (alias for cnot)
    pub fn cx(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.cnot(control, target)
    }

    /// Apply a CY gate (Controlled-Y)
    pub fn cy(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CY {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a CZ gate (Controlled-Z)
    pub fn cz(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CZ {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a CH gate (Controlled-Hadamard)
    pub fn ch(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CH {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a CS gate (Controlled-Phase/S)
    pub fn cs(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CS {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a controlled rotation around X-axis (CRX)
    pub fn crx(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
        theta: f64,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CRX {
            control: control.into(),
            target: target.into(),
            theta,
        })
    }

    /// Apply a controlled rotation around Y-axis (CRY)
    pub fn cry(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
        theta: f64,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CRY {
            control: control.into(),
            target: target.into(),
            theta,
        })
    }

    /// Apply a controlled rotation around Z-axis (CRZ)
    pub fn crz(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
        theta: f64,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CRZ {
            control: control.into(),
            target: target.into(),
            theta,
        })
    }

    /// Apply a controlled phase gate
    pub fn cp(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
        lambda: f64,
    ) -> QuantRS2Result<&mut Self> {
        // CRZ(lambda) is equivalent to CP(lambda) up to a global phase
        self.crz(control, target, lambda)
    }

    /// Apply a SWAP gate
    pub fn swap(
        &mut self,
        qubit1: impl Into<QubitId>,
        qubit2: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(SWAP {
            qubit1: qubit1.into(),
            qubit2: qubit2.into(),
        })
    }

    /// Apply a Toffoli (CCNOT) gate
    pub fn toffoli(
        &mut self,
        control1: impl Into<QubitId>,
        control2: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(Toffoli {
            control1: control1.into(),
            control2: control2.into(),
            target: target.into(),
        })
    }

    /// Apply a Fredkin (CSWAP) gate
    pub fn cswap(
        &mut self,
        control: impl Into<QubitId>,
        target1: impl Into<QubitId>,
        target2: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(Fredkin {
            control: control.into(),
            target1: target1.into(),
            target2: target2.into(),
        })
    }

    /// Measure a qubit (currently adds a placeholder measure gate)
    ///
    /// Note: This is currently a placeholder implementation for QASM export compatibility.
    /// For actual quantum measurements, use the measurement module functionality.
    pub fn measure(&mut self, qubit: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        let qubit_id = qubit.into();
        self.add_gate(Measure { target: qubit_id })?;
        Ok(self)
    }

    /// Reset a qubit to |0⟩ state
    ///
    /// Note: This operation is not yet fully implemented.
    /// Reset operations are complex and require special handling in quantum circuits.
    pub fn reset(&mut self, _qubit: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        Err(quantrs2_core::error::QuantRS2Error::UnsupportedOperation(
            "Reset operation is not yet implemented. Reset requires special quantum state manipulation.".to_string()
        ))
    }

    /// Add a barrier to prevent optimization across this point
    ///
    /// Barriers are used to prevent gate optimization algorithms from reordering gates
    /// across specific points in the circuit. This is useful for maintaining timing
    /// constraints or preserving specific circuit structure.
    pub fn barrier(&mut self, qubits: &[QubitId]) -> QuantRS2Result<&mut Self> {
        // Validate all qubits are within range
        for &qubit in qubits {
            if qubit.id() as usize >= N {
                return Err(quantrs2_core::error::QuantRS2Error::InvalidQubitId(
                    qubit.id(),
                ));
            }
        }

        // For now, barriers are implicit - they don't add gates but could be used
        // by optimization passes. In a full implementation, we'd store barrier information
        // for use by the optimization framework.

        // TODO: Implement barrier storage for optimization passes
        Ok(self)
    }

    /// Run the circuit on a simulator
    pub fn run<S: Simulator<N>>(&self, simulator: S) -> QuantRS2Result<Register<N>> {
        simulator.run(self)
    }

    /// Decompose the circuit into a sequence of standard gates
    ///
    /// This method will return a new circuit with complex gates decomposed
    /// into sequences of simpler gates.
    pub fn decompose(&self) -> QuantRS2Result<Self> {
        let mut decomposed = Self::new();

        // Convert Arc gates to Box gates for compatibility with decomposition utilities
        let boxed_gates = self.gates_as_boxes();

        // Decompose all gates
        let simple_gates = decomp_utils::decompose_circuit(&boxed_gates)?;

        // Add each decomposed gate to the new circuit
        for gate in simple_gates {
            decomposed.add_gate_box(gate)?;
        }

        Ok(decomposed)
    }

    /// Build the circuit (for compatibility - returns self)
    pub fn build(self) -> Self {
        self
    }

    /// Optimize the circuit by combining or removing gates
    ///
    /// This method will return a new circuit with simplified gates
    /// by removing unnecessary gates or combining adjacent gates.
    pub fn optimize(&self) -> QuantRS2Result<Self> {
        let mut optimized = Self::new();

        // Convert Arc gates to Box gates for compatibility with optimization utilities
        let boxed_gates = self.gates_as_boxes();

        // Optimize the gate sequence
        let simplified_gates_result = decomp_utils::optimize_gate_sequence(&boxed_gates);

        // Add each optimized gate to the new circuit
        if let Ok(simplified_gates) = simplified_gates_result {
            // We need to handle each gate individually
            for g in simplified_gates {
                optimized.add_gate_box(g)?;
            }
        }

        Ok(optimized)
    }

    /// Add a raw boxed gate to the circuit
    /// This is an internal utility and not part of the public API
    fn add_gate_box(&mut self, gate: Box<dyn GateOp>) -> QuantRS2Result<&mut Self> {
        // Validate that all qubits are within range
        for qubit in gate.qubits() {
            if qubit.id() as usize >= N {
                return Err(quantrs2_core::error::QuantRS2Error::InvalidInput(format!(
                    "Gate '{}' targets qubit {} which is out of range for {}-qubit circuit (valid range: 0-{})",
                    gate.name(),
                    qubit.id(),
                    N,
                    N - 1
                )));
            }
        }

        // For now, convert via cloning until we can update all callers to use Arc directly
        // This maintains safety but has some performance cost
        let cloned_gate = gate.clone_gate();

        // Convert the specific gate types to Arc using match
        if let Some(h_gate) = cloned_gate.as_any().downcast_ref::<Hadamard>() {
            self.gates
                .push(Arc::new(h_gate.clone()) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(x_gate) = cloned_gate.as_any().downcast_ref::<PauliX>() {
            self.gates
                .push(Arc::new(x_gate.clone()) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(y_gate) = cloned_gate.as_any().downcast_ref::<PauliY>() {
            self.gates
                .push(Arc::new(y_gate.clone()) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(z_gate) = cloned_gate.as_any().downcast_ref::<PauliZ>() {
            self.gates
                .push(Arc::new(z_gate.clone()) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(cnot_gate) = cloned_gate.as_any().downcast_ref::<CNOT>() {
            self.gates
                .push(Arc::new(cnot_gate.clone()) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(measure_gate) = cloned_gate.as_any().downcast_ref::<Measure>() {
            self.gates
                .push(Arc::new(measure_gate.clone()) as Arc<dyn GateOp + Send + Sync>);
        } else {
            // For unknown gate types, we'll use a less efficient fallback
            // TODO: Extend this to cover all gate types or implement a better conversion mechanism
            return Err(quantrs2_core::error::QuantRS2Error::UnsupportedOperation(
                format!(
                    "Gate type '{}' not yet supported in Arc conversion",
                    gate.name()
                ),
            ));
        }

        Ok(self)
    }

    /// Create a composite gate from a subsequence of this circuit
    ///
    /// This method allows creating a custom gate that combines several
    /// other gates, which can be applied as a single unit to a circuit.
    pub fn create_composite(
        &self,
        start_idx: usize,
        end_idx: usize,
        name: &str,
    ) -> QuantRS2Result<CompositeGate> {
        if start_idx >= self.gates.len() || end_idx > self.gates.len() || start_idx >= end_idx {
            return Err(quantrs2_core::error::QuantRS2Error::InvalidInput(format!(
                "Invalid start/end indices ({}/{}) for circuit with {} gates",
                start_idx,
                end_idx,
                self.gates.len()
            )));
        }

        // Get the gates in the specified range
        // We need to create box clones of each gate
        let mut gates: Vec<Box<dyn GateOp>> = Vec::new();
        for gate in &self.gates[start_idx..end_idx] {
            gates.push(decomp_utils::clone_gate(gate.as_ref())?);
        }

        // Collect all unique qubits these gates act on
        let mut qubits = Vec::new();
        for gate in &gates {
            for qubit in gate.qubits() {
                if !qubits.contains(&qubit) {
                    qubits.push(qubit);
                }
            }
        }

        Ok(CompositeGate {
            gates,
            qubits,
            name: name.to_string(),
        })
    }

    /// Add all gates from a composite gate to this circuit
    pub fn add_composite(&mut self, composite: &CompositeGate) -> QuantRS2Result<&mut Self> {
        // Clone each gate from the composite and add to this circuit
        for gate in &composite.gates {
            // We can't directly clone a Box<dyn GateOp>, so we need a different approach
            // We need to create a new gate by using the type information
            // This is a simplified version - in a real implementation,
            // we would have a more robust way to clone gates
            let gate_clone = decomp_utils::clone_gate(gate.as_ref())?;
            self.add_gate_box(gate_clone)?;
        }

        Ok(self)
    }

    // Classical control flow extensions

    /// Measure all qubits in the circuit
    pub fn measure_all(&mut self) -> QuantRS2Result<&mut Self> {
        for i in 0..N {
            self.measure(QubitId(i as u32))?;
        }
        Ok(self)
    }

    /// Convert this circuit to a ClassicalCircuit with classical control support
    pub fn with_classical_control(self) -> crate::classical::ClassicalCircuit<N> {
        let mut classical_circuit = crate::classical::ClassicalCircuit::new();

        // Add a default classical register for measurements
        let _ = classical_circuit.add_classical_register("c", N);

        // Transfer all gates, converting Arc to Box for compatibility
        for gate in self.gates {
            let boxed_gate = gate.clone_gate();
            classical_circuit
                .operations
                .push(crate::classical::CircuitOp::Quantum(boxed_gate));
        }

        classical_circuit
    }
}

impl<const N: usize> Default for Circuit<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for quantum circuit simulators
pub trait Simulator<const N: usize> {
    /// Run a quantum circuit and return the final register state
    fn run(&self, circuit: &Circuit<N>) -> QuantRS2Result<Register<N>>;
}
