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
        multi::{
            Fredkin,
            ISwap,
            Toffoli,
            CH,
            CNOT,
            CRX,
            CRY,
            CRZ,
            CS,
            CY,
            CZ,
            // Qiskit-compatible gates
            DCX,
            ECR,
            RXX,
            RYY,
            RZX,
            RZZ,
            SWAP,
        },
        single::{
            Hadamard,
            // Qiskit-compatible gates
            Identity,
            PGate,
            PauliX,
            PauliY,
            PauliZ,
            Phase,
            PhaseDagger,
            RotationX,
            RotationY,
            RotationZ,
            SqrtX,
            SqrtXDagger,
            TDagger,
            UGate,
            T,
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

    /// Get a gate from the pool if available, otherwise create new.
    ///
    /// Parameterized gates (RX, RY, RZ, rotation angles, etc.) are NEVER cached
    /// because two gates with the same name and target qubit can have different
    /// rotation angles.  Caching them by (name, qubits) alone would incorrectly
    /// return a stale angle on repeated calls — which breaks variational circuits.
    pub fn get_gate<G: GateOp + Clone + Send + Sync + 'static>(
        &mut self,
        gate: G,
    ) -> Arc<dyn GateOp + Send + Sync> {
        // Parameterized gates must not be pooled: always allocate fresh.
        if gate.is_parameterized() {
            return Arc::new(gate) as Arc<dyn GateOp + Send + Sync>;
        }

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

/// Wrapper that lets a `Box<dyn GateOp>` live inside an `Arc<dyn GateOp + Send + Sync>`.
///
/// `GateOp` is already `Send + Sync` (see the trait definition), so this is safe.
struct BoxGateWrapper(Box<dyn GateOp>);

impl std::fmt::Debug for BoxGateWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

// SAFETY: GateOp: Send + Sync, so Box<dyn GateOp> is Send + Sync.
unsafe impl Send for BoxGateWrapper {}
unsafe impl Sync for BoxGateWrapper {}

impl GateOp for BoxGateWrapper {
    fn name(&self) -> &'static str {
        self.0.name()
    }
    fn qubits(&self) -> Vec<QubitId> {
        self.0.qubits()
    }
    fn num_qubits(&self) -> usize {
        self.0.num_qubits()
    }
    fn is_parameterized(&self) -> bool {
        self.0.is_parameterized()
    }
    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        self.0.matrix()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self.0.as_any()
    }
    fn clone_gate(&self) -> Box<dyn GateOp> {
        self.0.clone_gate()
    }
}

/// Barrier metadata: records which gate-index a barrier was inserted after
/// and which qubits it spans.  Optimization passes inspect this list and
/// refuse to move any gate across a barrier boundary that includes its qubit.
#[derive(Debug, Clone)]
pub struct BarrierInfo {
    /// Index into `gates` *after* which this barrier is logically placed.
    /// A value of `0` means "before all gates".
    pub after_gate_index: usize,
    /// Qubits covered by this barrier.
    pub qubits: Vec<QubitId>,
}

/// A quantum circuit with a fixed number of qubits.
///
/// `Circuit<N>` stores a sequence of quantum gate operations over `N` qubits.
/// Gates can be appended with the builder methods (e.g. [`Circuit::h`],
/// [`Circuit::cnot`]) and the circuit can be simulated by passing it to any
/// type that implements [`Simulator`].
///
/// # Examples
///
/// ```rust
/// use quantrs2_circuit::builder::Circuit;
///
/// // Build a 2-qubit Bell state preparation circuit
/// let mut circ: Circuit<2> = Circuit::new();
/// circ.h(0).expect("h failed").cnot(0, 1).expect("cnot failed");
/// assert_eq!(circ.num_gates(), 2);
/// ```
pub struct Circuit<const N: usize> {
    /// Vector of gates to be applied in sequence using Arc for shared ownership
    gates: Vec<Arc<dyn GateOp + Send + Sync>>,
    /// Gate pool for reusing common gates
    gate_pool: GatePool,
    /// Barrier metadata stored for use by optimization passes.
    /// Barriers are *not* real gates — they carry no unitary — but they
    /// partition the gate list and must be respected by any reordering pass.
    pub barriers: Vec<BarrierInfo>,
}

impl<const N: usize> Clone for Circuit<N> {
    fn clone(&self) -> Self {
        // With Arc, cloning is much more efficient - just clone the references
        Self {
            gates: self.gates.clone(),
            gate_pool: self.gate_pool.clone(),
            barriers: self.barriers.clone(),
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
    /// Create a new empty circuit with N qubits.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use quantrs2_circuit::builder::Circuit;
    /// let circ: Circuit<3> = Circuit::new();
    /// assert_eq!(circ.num_gates(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            gates: Vec::with_capacity(64), // Pre-allocate capacity for better performance
            gate_pool: GatePool::new(),
            barriers: Vec::new(),
        }
    }

    /// Create a new circuit with estimated capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            gates: Vec::with_capacity(capacity),
            gate_pool: GatePool::new(),
            barriers: Vec::new(),
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

    /// Create a circuit from a list of boxed gate operations.
    ///
    /// Gates are added sequentially; any gate targeting a qubit ≥ N is silently
    /// dropped (rather than causing a hard error) so that optimization passes
    /// that may insert temporary placeholders still produce a valid circuit.
    pub fn from_gates(gates: Vec<Box<dyn GateOp>>) -> QuantRS2Result<Self> {
        let mut circuit = Self::with_capacity(gates.len());
        for gate in gates {
            // Validate qubit bounds; skip rather than abort on out-of-range gates.
            let in_range = gate.qubits().iter().all(|q| (q.id() as usize) < N);
            if in_range {
                // Clone the gate via the trait method to get a concrete Arc.
                // `clone_gate` returns `Box<dyn GateOp>` which is Send+Sync;
                // We use the BoxGateWrapper to convert to Arc<dyn GateOp+Send+Sync>.
                let arc: Arc<dyn GateOp + Send + Sync> = Arc::new(BoxGateWrapper(gate));
                circuit.gates.push(arc);
            }
        }
        Ok(circuit)
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

    /// Helper method to find a gate by type and index
    pub(crate) fn find_gate_by_type_and_index(
        &self,
        gate_type: &str,
        index: usize,
    ) -> Option<&dyn GateOp> {
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
    /// Barrier metadata is preserved: each barrier is re-anchored to the
    /// closest gate index in the optimized circuit.
    pub fn optimize(&self) -> QuantRS2Result<Self> {
        let mut optimized = Self::new();

        // Convert Arc gates to Box gates for compatibility with optimization utilities
        let boxed_gates = self.gates_as_boxes();

        // Optimize the gate sequence
        let simplified_gates_result = decomp_utils::optimize_gate_sequence(&boxed_gates);

        // Add each optimized gate to the new circuit
        if let Ok(simplified_gates) = simplified_gates_result {
            for g in simplified_gates {
                optimized.add_gate_box(g)?;
            }
        }

        // Re-anchor barriers: clamp after_gate_index to the new gate count so
        // that barriers are not lost even if the gate list shrinks.
        let new_gate_count = optimized.gates.len();
        optimized.barriers = self
            .barriers
            .iter()
            .map(|b| BarrierInfo {
                after_gate_index: b.after_gate_index.min(new_gate_count),
                qubits: b.qubits.clone(),
            })
            .collect();

        Ok(optimized)
    }

    /// Add a raw boxed gate to the circuit
    /// Exposed as `pub(crate)` so that routing and transpiler passes can use it.
    pub(crate) fn add_gate_box(&mut self, gate: Box<dyn GateOp>) -> QuantRS2Result<&mut Self> {
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

        // Attempt a zero-copy fast-path for every concrete gate type that is
        // Copy/Clone.  For any type not listed here the BoxGateWrapper fallback
        // is used instead, which avoids the UnsupportedOperation error that
        // previously blocked callers like `decompose()` and `add_composite()`.
        if let Some(g) = cloned_gate.as_any().downcast_ref::<Hadamard>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<PauliX>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<PauliY>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<PauliZ>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<CNOT>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<CZ>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<SWAP>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<CY>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<CH>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<CS>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<Toffoli>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<Fredkin>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<CRX>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<CRY>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<CRZ>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<ISwap>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<ECR>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<RXX>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<RYY>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<RZZ>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<RZX>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<DCX>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<RotationX>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<RotationY>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<RotationZ>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<Phase>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<PhaseDagger>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<T>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<TDagger>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<SqrtX>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<SqrtXDagger>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<UGate>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<PGate>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<Identity>() {
            self.gates
                .push(Arc::new(*g) as Arc<dyn GateOp + Send + Sync>);
        } else if let Some(g) = cloned_gate.as_any().downcast_ref::<Measure>() {
            self.gates
                .push(Arc::new(g.clone()) as Arc<dyn GateOp + Send + Sync>);
        } else {
            // Generic fallback for any gate type not listed above.
            // BoxGateWrapper is Send + Sync (enforced by SAFETY comment on the
            // struct), so we can safely wrap any Box<dyn GateOp> in an Arc.
            // This avoids an UnsupportedOperation error and keeps callers like
            // `decompose()`, `optimize()`, and `add_composite()` working for
            // third-party gate types defined outside this crate.
            self.gates
                .push(Arc::new(BoxGateWrapper(cloned_gate)) as Arc<dyn GateOp + Send + Sync>);
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
mod tests;
