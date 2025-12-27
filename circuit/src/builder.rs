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

use scirs2_core::Complex64;
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
    #[must_use]
    pub fn new() -> Self {
        let mut gates = HashMap::with_capacity(16);

        // Pre-allocate common gates for different qubits
        for qubit_id in 0..32 {
            let qubit = QubitId::new(qubit_id);

            // Common single-qubit gates
            gates.insert(
                format!("H_{qubit_id}"),
                Arc::new(Hadamard { target: qubit }) as Arc<dyn GateOp + Send + Sync>,
            );
            gates.insert(
                format!("X_{qubit_id}"),
                Arc::new(PauliX { target: qubit }) as Arc<dyn GateOp + Send + Sync>,
            );
            gates.insert(
                format!("Y_{qubit_id}"),
                Arc::new(PauliY { target: qubit }) as Arc<dyn GateOp + Send + Sync>,
            );
            gates.insert(
                format!("Z_{qubit_id}"),
                Arc::new(PauliZ { target: qubit }) as Arc<dyn GateOp + Send + Sync>,
            );
            gates.insert(
                format!("S_{qubit_id}"),
                Arc::new(Phase { target: qubit }) as Arc<dyn GateOp + Send + Sync>,
            );
            gates.insert(
                format!("T_{qubit_id}"),
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
    #[must_use]
    pub fn new() -> Self {
        Self {
            gates: Vec::with_capacity(64), // Pre-allocate capacity for better performance
            gate_pool: GatePool::new(),
        }
    }

    /// Create a new circuit with estimated capacity
    #[must_use]
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
    #[must_use]
    pub fn gates(&self) -> &[Arc<dyn GateOp + Send + Sync>] {
        &self.gates
    }

    /// Get gates as Vec for compatibility with existing optimization code
    #[must_use]
    pub fn gates_as_boxes(&self) -> Vec<Box<dyn GateOp>> {
        self.gates
            .iter()
            .map(|arc_gate| arc_gate.clone_gate())
            .collect()
    }

    /// Circuit introspection methods for optimization

    /// Count gates by type
    #[must_use]
    pub fn count_gates_by_type(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for gate in &self.gates {
            *counts.entry(gate.name().to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Calculate circuit depth (longest sequential path)
    #[must_use]
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
    #[must_use]
    pub fn count_two_qubit_gates(&self) -> usize {
        self.gates
            .iter()
            .filter(|gate| gate.qubits().len() == 2)
            .count()
    }

    /// Count multi-qubit gates (3 or more qubits)
    #[must_use]
    pub fn count_multi_qubit_gates(&self) -> usize {
        self.gates
            .iter()
            .filter(|gate| gate.qubits().len() >= 3)
            .count()
    }

    /// Calculate the critical path length (same as depth for now, but could be enhanced)
    #[must_use]
    pub fn calculate_critical_path(&self) -> usize {
        self.calculate_depth()
    }

    /// Calculate gate density (gates per qubit)
    #[must_use]
    pub fn calculate_gate_density(&self) -> f64 {
        if N == 0 {
            0.0
        } else {
            self.gates.len() as f64 / N as f64
        }
    }

    /// Get all unique qubits used in the circuit
    #[must_use]
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
    #[must_use]
    pub fn uses_all_qubits(&self) -> bool {
        self.get_used_qubits().len() == N
    }

    /// Get gates that operate on a specific qubit
    #[must_use]
    pub fn gates_on_qubit(&self, target_qubit: QubitId) -> Vec<&Arc<dyn GateOp + Send + Sync>> {
        self.gates
            .iter()
            .filter(|gate| gate.qubits().contains(&target_qubit))
            .collect()
    }

    /// Get gates between two indices (inclusive)
    #[must_use]
    pub fn gates_in_range(&self, start: usize, end: usize) -> &[Arc<dyn GateOp + Send + Sync>] {
        let end = end.min(self.gates.len().saturating_sub(1));
        let start = start.min(end);
        &self.gates[start..=end]
    }

    /// Check if circuit is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// Get circuit statistics summary
    #[must_use]
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
    #[must_use]
    pub const fn num_qubits(&self) -> usize {
        N
    }

    /// Get the number of gates in the circuit
    #[must_use]
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    /// Get the names of all gates in the circuit
    #[must_use]
    pub fn get_gate_names(&self) -> Vec<String> {
        self.gates
            .iter()
            .map(|gate| gate.name().to_string())
            .collect()
    }

    /// Get a qubit for a specific single-qubit gate by gate type and index
    #[cfg(feature = "python")]
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
                    "Gate {gate_type} at index {index} not found or is not a single-qubit gate"
                ))
            })
    }

    /// Get rotation parameters (qubit, angle) for a specific gate by gate type and index
    #[cfg(feature = "python")]
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
                    "Gate {gate_type} at index {index} not found or is not a rotation gate"
                ))
            })
    }

    /// Get two-qubit parameters (control, target) for a specific gate by gate type and index
    #[cfg(feature = "python")]
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
                    "Gate {gate_type} at index {index} not found or is not a two-qubit gate"
                ))
            })
    }

    /// Get controlled rotation parameters (control, target, angle) for a specific gate
    #[cfg(feature = "python")]
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
                    "Gate {gate_type} at index {index} not found or is not a controlled rotation gate"
                ))
            })
    }

    /// Get three-qubit parameters for gates like Toffoli or Fredkin
    #[cfg(feature = "python")]
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
                    "Gate {gate_type} at index {index} not found or is not a three-qubit gate"
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
    #[must_use]
    pub const fn build(self) -> Self {
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
                .push(Arc::new(*h_gate) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(x_gate) = cloned_gate.as_any().downcast_ref::<PauliX>() {
            self.gates
                .push(Arc::new(*x_gate) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(y_gate) = cloned_gate.as_any().downcast_ref::<PauliY>() {
            self.gates
                .push(Arc::new(*y_gate) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(z_gate) = cloned_gate.as_any().downcast_ref::<PauliZ>() {
            self.gates
                .push(Arc::new(*z_gate) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(cnot_gate) = cloned_gate.as_any().downcast_ref::<CNOT>() {
            self.gates
                .push(Arc::new(*cnot_gate) as Arc<dyn GateOp + Send + Sync>);
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

    /// Convert this circuit to a `ClassicalCircuit` with classical control support
    #[must_use]
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

    // Batch operations for improved ergonomics

    /// Apply Hadamard gates to multiple qubits at once
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<5>::new();
    /// circuit.h_all(&[0, 1, 2])?; // Apply H to qubits 0, 1, and 2
    /// ```
    pub fn h_all(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        for &qubit in qubits {
            self.h(QubitId::new(qubit))?;
        }
        Ok(self)
    }

    /// Apply Pauli-X gates to multiple qubits at once
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<5>::new();
    /// circuit.x_all(&[0, 2, 4])?; // Apply X to qubits 0, 2, and 4
    /// ```
    pub fn x_all(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        for &qubit in qubits {
            self.x(QubitId::new(qubit))?;
        }
        Ok(self)
    }

    /// Apply Pauli-Y gates to multiple qubits at once
    pub fn y_all(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        for &qubit in qubits {
            self.y(QubitId::new(qubit))?;
        }
        Ok(self)
    }

    /// Apply Pauli-Z gates to multiple qubits at once
    pub fn z_all(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        for &qubit in qubits {
            self.z(QubitId::new(qubit))?;
        }
        Ok(self)
    }

    /// Apply Hadamard gates to a range of qubits
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<5>::new();
    /// circuit.h_range(0..3)?; // Apply H to qubits 0, 1, and 2
    /// ```
    pub fn h_range(&mut self, range: std::ops::Range<u32>) -> QuantRS2Result<&mut Self> {
        for qubit in range {
            self.h(QubitId::new(qubit))?;
        }
        Ok(self)
    }

    /// Apply Pauli-X gates to a range of qubits
    pub fn x_range(&mut self, range: std::ops::Range<u32>) -> QuantRS2Result<&mut Self> {
        for qubit in range {
            self.x(QubitId::new(qubit))?;
        }
        Ok(self)
    }

    // Common quantum state preparation patterns

    /// Prepare a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 on two qubits
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<2>::new();
    /// circuit.bell_state(0, 1)?; // Prepare Bell state on qubits 0 and 1
    /// ```
    pub fn bell_state(&mut self, qubit1: u32, qubit2: u32) -> QuantRS2Result<&mut Self> {
        self.h(QubitId::new(qubit1))?;
        self.cnot(QubitId::new(qubit1), QubitId::new(qubit2))?;
        Ok(self)
    }

    /// Prepare a GHZ state (|000...⟩ + |111...⟩)/√2 on specified qubits
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<3>::new();
    /// circuit.ghz_state(&[0, 1, 2])?; // Prepare GHZ state on qubits 0, 1, and 2
    /// ```
    pub fn ghz_state(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        if qubits.is_empty() {
            return Ok(self);
        }

        // Apply Hadamard to first qubit
        self.h(QubitId::new(qubits[0]))?;

        // Apply CNOT gates to entangle all qubits
        for i in 1..qubits.len() {
            self.cnot(QubitId::new(qubits[0]), QubitId::new(qubits[i]))?;
        }

        Ok(self)
    }

    /// Prepare a W state on specified qubits
    ///
    /// W state: (|100...⟩ + |010...⟩ + |001...⟩ + ...)/√n
    ///
    /// This is an approximation using rotation gates.
    pub fn w_state(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        if qubits.is_empty() {
            return Ok(self);
        }

        let n = qubits.len() as f64;

        // For n qubits, prepare W state using controlled rotations
        // This is a simplified implementation
        self.ry(QubitId::new(qubits[0]), 2.0 * (1.0 / n.sqrt()).acos())?;

        for i in 1..qubits.len() {
            let angle = 2.0 * (1.0 / (n - i as f64).sqrt()).acos();
            self.cry(QubitId::new(qubits[i - 1]), QubitId::new(qubits[i]), angle)?;
        }

        // Apply X gates to ensure proper state preparation
        for i in 0..qubits.len() - 1 {
            self.cnot(QubitId::new(qubits[i + 1]), QubitId::new(qubits[i]))?;
        }

        Ok(self)
    }

    /// Prepare a product state |++++...⟩ by applying Hadamard to all qubits
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<4>::new();
    /// circuit.plus_state_all()?; // Prepare |+⟩ on all 4 qubits
    /// ```
    pub fn plus_state_all(&mut self) -> QuantRS2Result<&mut Self> {
        for i in 0..N {
            self.h(QubitId::new(i as u32))?;
        }
        Ok(self)
    }

    /// Apply a rotation gate to multiple qubits with the same angle
    pub fn rx_all(&mut self, qubits: &[u32], theta: f64) -> QuantRS2Result<&mut Self> {
        for &qubit in qubits {
            self.rx(QubitId::new(qubit), theta)?;
        }
        Ok(self)
    }

    /// Apply RY rotation to multiple qubits
    pub fn ry_all(&mut self, qubits: &[u32], theta: f64) -> QuantRS2Result<&mut Self> {
        for &qubit in qubits {
            self.ry(QubitId::new(qubit), theta)?;
        }
        Ok(self)
    }

    /// Apply RZ rotation to multiple qubits
    pub fn rz_all(&mut self, qubits: &[u32], theta: f64) -> QuantRS2Result<&mut Self> {
        for &qubit in qubits {
            self.rz(QubitId::new(qubit), theta)?;
        }
        Ok(self)
    }

    /// Create a ladder of CNOT gates connecting adjacent qubits
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<4>::new();
    /// circuit.cnot_ladder(&[0, 1, 2, 3])?; // Creates: CNOT(0,1), CNOT(1,2), CNOT(2,3)
    /// ```
    pub fn cnot_ladder(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        if qubits.len() < 2 {
            return Ok(self);
        }

        for i in 0..qubits.len() - 1 {
            self.cnot(QubitId::new(qubits[i]), QubitId::new(qubits[i + 1]))?;
        }

        Ok(self)
    }

    /// Create a ring of CNOT gates connecting qubits in a cycle
    ///
    /// Like CNOT ladder but also connects last to first qubit.
    pub fn cnot_ring(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        if qubits.len() < 2 {
            return Ok(self);
        }

        // Add ladder
        self.cnot_ladder(qubits)?;

        // Close the ring by connecting last to first
        let last_idx = qubits.len() - 1;
        self.cnot(QubitId::new(qubits[last_idx]), QubitId::new(qubits[0]))?;

        Ok(self)
    }

    /// Create a ladder of SWAP gates connecting adjacent qubits
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<4>::new();
    /// circuit.swap_ladder(&[0, 1, 2, 3])?; // Creates: SWAP(0,1), SWAP(1,2), SWAP(2,3)
    /// ```
    pub fn swap_ladder(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        if qubits.len() < 2 {
            return Ok(self);
        }

        for i in 0..qubits.len() - 1 {
            self.swap(QubitId::new(qubits[i]), QubitId::new(qubits[i + 1]))?;
        }

        Ok(self)
    }

    /// Create a ladder of CZ gates connecting adjacent qubits
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<4>::new();
    /// circuit.cz_ladder(&[0, 1, 2, 3])?; // Creates: CZ(0,1), CZ(1,2), CZ(2,3)
    /// ```
    pub fn cz_ladder(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        if qubits.len() < 2 {
            return Ok(self);
        }

        for i in 0..qubits.len() - 1 {
            self.cz(QubitId::new(qubits[i]), QubitId::new(qubits[i + 1]))?;
        }

        Ok(self)
    }

    /// Apply SWAP gates to multiple qubit pairs
    ///
    /// # Arguments
    /// * `pairs` - Slice of (control, target) qubit pairs
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<6>::new();
    /// circuit.swap_all(&[(0, 1), (2, 3), (4, 5)])?; // Swap three pairs simultaneously
    /// ```
    pub fn swap_all(&mut self, pairs: &[(u32, u32)]) -> QuantRS2Result<&mut Self> {
        for &(q1, q2) in pairs {
            self.swap(QubitId::new(q1), QubitId::new(q2))?;
        }
        Ok(self)
    }

    /// Apply CZ gates to multiple qubit pairs
    ///
    /// # Arguments
    /// * `pairs` - Slice of (control, target) qubit pairs
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<6>::new();
    /// circuit.cz_all(&[(0, 1), (2, 3), (4, 5)])?; // Apply CZ to three pairs
    /// ```
    pub fn cz_all(&mut self, pairs: &[(u32, u32)]) -> QuantRS2Result<&mut Self> {
        for &(q1, q2) in pairs {
            self.cz(QubitId::new(q1), QubitId::new(q2))?;
        }
        Ok(self)
    }

    /// Apply CNOT gates to multiple qubit pairs
    ///
    /// # Arguments
    /// * `pairs` - Slice of (control, target) qubit pairs
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<6>::new();
    /// circuit.cnot_all(&[(0, 1), (2, 3), (4, 5)])?; // Apply CNOT to three pairs
    /// ```
    pub fn cnot_all(&mut self, pairs: &[(u32, u32)]) -> QuantRS2Result<&mut Self> {
        for &(control, target) in pairs {
            self.cnot(QubitId::new(control), QubitId::new(target))?;
        }
        Ok(self)
    }

    /// Add barriers to multiple qubits
    ///
    /// Barriers prevent optimization across them and can be used to
    /// visualize circuit structure.
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<5>::new();
    /// circuit.h_all(&[0, 1, 2])?;
    /// circuit.barrier_all(&[0, 1, 2])?; // Prevent optimization across this point
    /// circuit.cnot_ladder(&[0, 1, 2])?;
    /// ```
    pub fn barrier_all(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        let qubit_ids: Vec<QubitId> = qubits.iter().map(|&q| QubitId::new(q)).collect();
        self.barrier(&qubit_ids)?;
        Ok(self)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h_all() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<5>::new();
        circuit.h_all(&[0, 1, 2])?;

        assert_eq!(circuit.gates().len(), 3);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "H");
        }
        Ok(())
    }

    #[test]
    fn test_x_all() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<5>::new();
        circuit.x_all(&[0, 2, 4])?;

        assert_eq!(circuit.gates().len(), 3);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "X");
        }
        Ok(())
    }

    #[test]
    fn test_y_all() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<3>::new();
        circuit.y_all(&[0, 1, 2])?;

        assert_eq!(circuit.gates().len(), 3);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "Y");
        }
        Ok(())
    }

    #[test]
    fn test_z_all() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<4>::new();
        circuit.z_all(&[1, 3])?;

        assert_eq!(circuit.gates().len(), 2);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "Z");
        }
        Ok(())
    }

    #[test]
    fn test_h_range() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<5>::new();
        circuit.h_range(0..3)?;

        assert_eq!(circuit.gates().len(), 3);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "H");
        }
        Ok(())
    }

    #[test]
    fn test_x_range() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<5>::new();
        circuit.x_range(1..4)?;

        assert_eq!(circuit.gates().len(), 3);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "X");
        }
        Ok(())
    }

    #[test]
    fn test_bell_state() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<2>::new();
        circuit.bell_state(0, 1)?;

        assert_eq!(circuit.gates().len(), 2);
        assert_eq!(circuit.gates()[0].name(), "H");
        assert_eq!(circuit.gates()[1].name(), "CNOT");
        Ok(())
    }

    #[test]
    fn test_ghz_state() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<4>::new();
        circuit.ghz_state(&[0, 1, 2, 3])?;

        // Should have 1 H + 3 CNOTs
        assert_eq!(circuit.gates().len(), 4);
        assert_eq!(circuit.gates()[0].name(), "H");
        for i in 1..4 {
            assert_eq!(circuit.gates()[i].name(), "CNOT");
        }
        Ok(())
    }

    #[test]
    fn test_ghz_state_empty() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<4>::new();
        circuit.ghz_state(&[])?;

        assert_eq!(circuit.gates().len(), 0);
        Ok(())
    }

    #[test]
    fn test_w_state() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<3>::new();
        circuit.w_state(&[0, 1, 2])?;

        // W state requires RY + CRY + CNOT gates
        assert!(!circuit.gates().is_empty());
        // At least one rotation gate
        assert!(circuit
            .gates()
            .iter()
            .any(|g| g.name() == "RY" || g.name() == "CRY"));
        Ok(())
    }

    #[test]
    fn test_w_state_empty() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<3>::new();
        circuit.w_state(&[])?;

        assert_eq!(circuit.gates().len(), 0);
        Ok(())
    }

    #[test]
    fn test_plus_state_all() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<4>::new();
        circuit.plus_state_all()?;

        assert_eq!(circuit.gates().len(), 4);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "H");
        }
        Ok(())
    }

    #[test]
    fn test_rx_all() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<4>::new();
        let theta = std::f64::consts::PI / 4.0;
        circuit.rx_all(&[0, 1, 2], theta)?;

        assert_eq!(circuit.gates().len(), 3);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "RX");
        }
        Ok(())
    }

    #[test]
    fn test_ry_all() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<4>::new();
        let theta = std::f64::consts::PI / 3.0;
        circuit.ry_all(&[0, 2], theta)?;

        assert_eq!(circuit.gates().len(), 2);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "RY");
        }
        Ok(())
    }

    #[test]
    fn test_rz_all() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<5>::new();
        let theta = std::f64::consts::PI / 2.0;
        circuit.rz_all(&[1, 2, 3], theta)?;

        assert_eq!(circuit.gates().len(), 3);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "RZ");
        }
        Ok(())
    }

    #[test]
    fn test_cnot_ladder() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<5>::new();
        circuit.cnot_ladder(&[0, 1, 2, 3])?;

        assert_eq!(circuit.gates().len(), 3);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "CNOT");
        }
        Ok(())
    }

    #[test]
    fn test_cnot_ladder_too_small() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<5>::new();
        circuit.cnot_ladder(&[0])?;

        assert_eq!(circuit.gates().len(), 0);
        Ok(())
    }

    #[test]
    fn test_cnot_ring() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<4>::new();
        circuit.cnot_ring(&[0, 1, 2, 3])?;

        // Should have 4 CNOTs (3 for ladder + 1 to close ring)
        assert_eq!(circuit.gates().len(), 4);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "CNOT");
        }
        Ok(())
    }

    #[test]
    fn test_cnot_ring_too_small() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<4>::new();
        circuit.cnot_ring(&[0])?;

        assert_eq!(circuit.gates().len(), 0);
        Ok(())
    }

    #[test]
    fn test_combined_patterns() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<5>::new();

        // Initialize all qubits to |+⟩
        circuit.plus_state_all()?;

        // Create entanglement with CNOT ladder
        circuit.cnot_ladder(&[0, 1, 2, 3, 4])?;

        // Apply phase to some qubits
        circuit.z_all(&[0, 2, 4])?;

        let stats = circuit.get_stats();
        assert_eq!(stats.total_gates, 5 + 4 + 3); // 5 H + 4 CNOT + 3 Z
        assert_eq!(stats.total_qubits, 5);
        Ok(())
    }

    #[test]
    fn test_swap_ladder() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<5>::new();
        circuit.swap_ladder(&[0, 1, 2, 3])?;

        assert_eq!(circuit.gates().len(), 3); // SWAP(0,1), SWAP(1,2), SWAP(2,3)
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "SWAP");
        }
        Ok(())
    }

    #[test]
    fn test_swap_ladder_empty() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<5>::new();
        circuit.swap_ladder(&[])?;

        assert_eq!(circuit.gates().len(), 0);
        Ok(())
    }

    #[test]
    fn test_swap_ladder_single() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<5>::new();
        circuit.swap_ladder(&[0])?;

        assert_eq!(circuit.gates().len(), 0);
        Ok(())
    }

    #[test]
    fn test_cz_ladder() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<4>::new();
        circuit.cz_ladder(&[0, 1, 2, 3])?;

        assert_eq!(circuit.gates().len(), 3); // CZ(0,1), CZ(1,2), CZ(2,3)
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "CZ");
        }
        Ok(())
    }

    #[test]
    fn test_cz_ladder_empty() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<4>::new();
        circuit.cz_ladder(&[])?;

        assert_eq!(circuit.gates().len(), 0);
        Ok(())
    }

    #[test]
    fn test_swap_all() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<6>::new();
        circuit.swap_all(&[(0, 1), (2, 3), (4, 5)])?;

        assert_eq!(circuit.gates().len(), 3);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "SWAP");
        }
        Ok(())
    }

    #[test]
    fn test_swap_all_empty() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<6>::new();
        circuit.swap_all(&[])?;

        assert_eq!(circuit.gates().len(), 0);
        Ok(())
    }

    #[test]
    fn test_cz_all() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<6>::new();
        circuit.cz_all(&[(0, 1), (2, 3), (4, 5)])?;

        assert_eq!(circuit.gates().len(), 3);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "CZ");
        }
        Ok(())
    }

    #[test]
    fn test_cz_all_empty() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<6>::new();
        circuit.cz_all(&[])?;

        assert_eq!(circuit.gates().len(), 0);
        Ok(())
    }

    #[test]
    fn test_cnot_all() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<6>::new();
        circuit.cnot_all(&[(0, 1), (2, 3), (4, 5)])?;

        assert_eq!(circuit.gates().len(), 3);
        for gate in circuit.gates() {
            assert_eq!(gate.name(), "CNOT");
        }
        Ok(())
    }

    #[test]
    fn test_cnot_all_empty() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<6>::new();
        circuit.cnot_all(&[])?;

        assert_eq!(circuit.gates().len(), 0);
        Ok(())
    }

    #[test]
    fn test_barrier_all() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<5>::new();
        circuit.h_all(&[0, 1, 2])?;
        circuit.barrier_all(&[0, 1, 2])?;
        circuit.cnot_ladder(&[0, 1, 2])?;

        // Barriers don't currently add gates (they're implicit in the optimization framework)
        // Should have 3 H + 2 CNOT
        assert_eq!(circuit.gates().len(), 5);
        Ok(())
    }

    #[test]
    fn test_barrier_all_empty() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<5>::new();
        circuit.barrier_all(&[])?;

        assert_eq!(circuit.gates().len(), 0);
        Ok(())
    }

    #[test]
    fn test_advanced_entanglement_patterns() -> QuantRS2Result<()> {
        let mut circuit = Circuit::<6>::new();

        // Create superposition
        circuit.h_all(&[0, 1, 2, 3, 4, 5])?;

        // Add barrier to prevent optimization (implicit, doesn't add gates)
        circuit.barrier_all(&[0, 1, 2, 3, 4, 5])?;

        // Create entanglement with CZ ladder
        circuit.cz_ladder(&[0, 1, 2, 3, 4, 5])?;

        // Add more entanglement with CNOT pairs
        circuit.cnot_all(&[(0, 3), (1, 4), (2, 5)])?;

        let stats = circuit.get_stats();
        // 6 H + 5 CZ + 3 CNOT = 14 gates (barriers are implicit)
        assert_eq!(stats.total_gates, 14);
        assert_eq!(stats.total_qubits, 6);
        Ok(())
    }
}
