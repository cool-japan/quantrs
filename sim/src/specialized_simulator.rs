//! Optimized state vector simulator using specialized gate implementations
//!
//! This simulator automatically detects and uses specialized gate implementations
//! for improved performance compared to general matrix multiplication.

use scirs2_core::parallel_ops::{
    IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use scirs2_core::Complex64;
use std::sync::Arc;

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{multi, single, GateOp},
    qubit::QubitId,
    register::Register,
};

use crate::specialized_gates::{specialize_gate, SpecializedGate};
use crate::statevector::StateVectorSimulator;
use crate::utils::flip_bit;

/// Configuration for specialized simulator
#[derive(Debug, Clone)]
pub struct SpecializedSimulatorConfig {
    /// Use parallel execution
    pub parallel: bool,
    /// Enable gate fusion optimization
    pub enable_fusion: bool,
    /// Enable gate reordering optimization
    pub enable_reordering: bool,
    /// Cache specialized gate conversions
    pub cache_conversions: bool,
    /// Minimum qubit count for parallel execution
    pub parallel_threshold: usize,
}

impl Default for SpecializedSimulatorConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            enable_fusion: true,
            enable_reordering: true,
            cache_conversions: true,
            parallel_threshold: 10,
        }
    }
}

/// Statistics about specialized gate usage
#[derive(Debug, Clone, Default)]
pub struct SpecializationStats {
    /// Total gates processed
    pub total_gates: usize,
    /// Gates using specialized implementation
    pub specialized_gates: usize,
    /// Gates using generic implementation
    pub generic_gates: usize,
    /// Gates that were fused
    pub fused_gates: usize,
    /// Time saved by specialization (estimated ms)
    pub time_saved_ms: f64,
}

/// Optimized state vector simulator with specialized gate implementations
pub struct SpecializedStateVectorSimulator {
    /// Configuration
    config: SpecializedSimulatorConfig,
    /// Base state vector simulator for fallback
    base_simulator: StateVectorSimulator,
    /// Statistics tracker
    stats: SpecializationStats,
    /// Cache for specialized gate conversions (simplified to avoid Clone issues)
    conversion_cache: Option<Arc<dashmap::DashMap<String, bool>>>,
    /// Reusable buffer for parallel gate application (avoids allocation per gate)
    work_buffer: Vec<Complex64>,
}

impl SpecializedStateVectorSimulator {
    /// Create a new specialized simulator
    #[must_use]
    pub fn new(config: SpecializedSimulatorConfig) -> Self {
        let base_simulator = if config.parallel {
            StateVectorSimulator::new()
        } else {
            StateVectorSimulator::sequential()
        };

        let conversion_cache = if config.cache_conversions {
            Some(Arc::new(dashmap::DashMap::new()))
        } else {
            None
        };

        Self {
            config,
            base_simulator,
            stats: SpecializationStats::default(),
            conversion_cache,
            work_buffer: Vec::new(),
        }
    }

    /// Get specialization statistics
    pub const fn get_stats(&self) -> &SpecializationStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SpecializationStats::default();
    }

    /// Run a quantum circuit
    pub fn run<const N: usize>(&mut self, circuit: &Circuit<N>) -> QuantRS2Result<Vec<Complex64>> {
        let n_qubits = N;
        let mut state = self.initialize_state(n_qubits);

        // Process gates with optimization
        let gates = if self.config.enable_reordering {
            self.reorder_gates(circuit.gates())?
        } else {
            circuit.gates().to_vec()
        };

        // Apply gates with fusion if enabled
        if self.config.enable_fusion {
            self.apply_gates_with_fusion(&mut state, &gates, n_qubits)?;
        } else {
            for gate in gates {
                self.apply_gate(&mut state, &gate, n_qubits)?;
            }
        }

        Ok(state)
    }

    /// Initialize quantum state
    fn initialize_state(&self, n_qubits: usize) -> Vec<Complex64> {
        let size = 1 << n_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); size];
        state[0] = Complex64::new(1.0, 0.0);
        state
    }

    /// Apply a single gate
    fn apply_gate(
        &mut self,
        state: &mut [Complex64],
        gate: &Arc<dyn GateOp + Send + Sync>,
        n_qubits: usize,
    ) -> QuantRS2Result<()> {
        self.stats.total_gates += 1;

        // Try to get specialized implementation
        if let Some(specialized) = self.get_specialized_gate(gate.as_ref()) {
            self.stats.specialized_gates += 1;
            self.stats.time_saved_ms += self.estimate_time_saved(gate.as_ref());

            let parallel = self.config.parallel && n_qubits >= self.config.parallel_threshold;
            specialized.apply_specialized(state, n_qubits, parallel)
        } else {
            self.stats.generic_gates += 1;

            // Fall back to generic implementation
            match gate.num_qubits() {
                1 => {
                    let qubits = gate.qubits();
                    let matrix = gate.matrix()?;
                    self.apply_single_qubit_generic(state, &matrix, qubits[0], n_qubits)
                }
                2 => {
                    let qubits = gate.qubits();
                    let matrix = gate.matrix()?;
                    self.apply_two_qubit_generic(state, &matrix, qubits[0], qubits[1], n_qubits)
                }
                _ => {
                    // For multi-qubit gates, use general matrix application
                    self.apply_multi_qubit_generic(state, gate.as_ref(), n_qubits)
                }
            }
        }
    }

    /// Get specialized gate implementation with caching
    fn get_specialized_gate(&self, gate: &dyn GateOp) -> Option<Box<dyn SpecializedGate>> {
        // Simplified: always create new specialized gate to avoid Clone constraints
        specialize_gate(gate)
    }

    /// Apply gates with fusion optimization
    fn apply_gates_with_fusion(
        &mut self,
        state: &mut [Complex64],
        gates: &[Arc<dyn GateOp + Send + Sync>],
        n_qubits: usize,
    ) -> QuantRS2Result<()> {
        let mut i = 0;

        while i < gates.len() {
            // Try to fuse with next gate
            if i + 1 < gates.len() {
                if let (Some(gate1), Some(gate2)) = (
                    self.get_specialized_gate(gates[i].as_ref()),
                    self.get_specialized_gate(gates[i + 1].as_ref()),
                ) {
                    if gate1.can_fuse_with(gate2.as_ref()) {
                        if let Some(fused) = gate1.fuse_with(gate2.as_ref()) {
                            self.stats.fused_gates += 2;
                            self.stats.total_gates += 1;

                            let parallel =
                                self.config.parallel && n_qubits >= self.config.parallel_threshold;
                            fused.apply_specialized(state, n_qubits, parallel)?;

                            i += 2;
                            continue;
                        }
                    }
                }
            }

            // Apply single gate
            self.apply_gate(state, &gates[i], n_qubits)?;
            i += 1;
        }

        Ok(())
    }

    /// Reorder gates for better cache locality *without* changing the
    /// circuit's semantics.
    ///
    /// A blind `sort_by_key` on the first qubit id (the previous
    /// implementation) can silently reorder gates whose relative order
    /// matters -- e.g. it would happily move `X(0)` before `CNOT(0, 1)` even
    /// though they act on a shared qubit and do not commute, corrupting the
    /// computed state. This implementation only ever moves a gate earlier in
    /// program order past gates it can *provably* commute with:
    ///
    /// * two gates acting on completely disjoint qubit sets always commute
    ///   (they act on independent tensor factors), and
    /// * two gates that are both diagonal in the computational basis (Z, S,
    ///   T, RZ, phase, controlled-diagonal, global phase, ...) always
    ///   commute with each other regardless of qubit overlap, since diagonal
    ///   matrices always commute.
    ///
    /// This is a greedy selection sort: for each output position, the
    /// earliest-by-first-qubit gate that can be proven to commute with every
    /// gate between its current position and the target position is chosen.
    /// Any gate that cannot be proven to commute stops the scan, so no
    /// dependency-violating move is ever made.
    fn reorder_gates(
        &self,
        gates: &[Arc<dyn GateOp + Send + Sync>],
    ) -> QuantRS2Result<Vec<Arc<dyn GateOp + Send + Sync>>> {
        let mut reordered: Vec<Arc<dyn GateOp + Send + Sync>> = gates.to_vec();
        let key =
            |gate: &Arc<dyn GateOp + Send + Sync>| gate.qubits().first().map_or(0, QubitId::id);

        for i in 0..reordered.len() {
            let mut best_j = i;
            let mut best_key = key(&reordered[i]);

            for j in (i + 1)..reordered.len() {
                // gates[j] can only be considered as a candidate for
                // position i if it provably commutes with every gate
                // currently occupying positions i..j (i.e. every gate it
                // would have to move past).
                if !Self::commutes_with_all(reordered[j].as_ref(), &reordered[i..j]) {
                    break;
                }
                let candidate_key = key(&reordered[j]);
                if candidate_key < best_key {
                    best_key = candidate_key;
                    best_j = j;
                }
            }

            if best_j != i {
                let gate = reordered.remove(best_j);
                reordered.insert(i, gate);
            }
        }

        Ok(reordered)
    }

    /// Whether `candidate` provably commutes with every gate in `others`,
    /// i.e. it is safe to move `candidate` past all of them without
    /// changing the circuit's semantics.
    fn commutes_with_all(candidate: &dyn GateOp, others: &[Arc<dyn GateOp + Send + Sync>]) -> bool {
        others
            .iter()
            .all(|other| Self::gates_commute(candidate, other.as_ref()))
    }

    /// Whether two gates provably commute.
    ///
    /// This is intentionally conservative: it only returns `true` when
    /// commutation is guaranteed by a structural property (disjoint qubits,
    /// or both gates diagonal in the computational basis), never by
    /// inspecting the numeric gate matrices. A `false` result may still
    /// correspond to gates that happen to commute (e.g. two different CNOTs
    /// sharing a qubit in specific configurations) -- that only forgoes an
    /// optimization opportunity, it never risks correctness.
    fn gates_commute(a: &dyn GateOp, b: &dyn GateOp) -> bool {
        let qubits_a = a.qubits();
        let qubits_b = b.qubits();
        let disjoint = qubits_a.iter().all(|q| !qubits_b.contains(q));
        if disjoint {
            return true;
        }
        Self::is_diagonal_gate(a) && Self::is_diagonal_gate(b)
    }

    /// Whether a gate's matrix is diagonal in the computational basis.
    ///
    /// Any two diagonal matrices commute regardless of which qubits they
    /// act on, so this is the basis for the only qubit-overlapping
    /// commutation case `gates_commute` recognizes.
    fn is_diagonal_gate(gate: &dyn GateOp) -> bool {
        matches!(
            gate.name(),
            "Z" | "S" | "S†" | "T" | "T†" | "RZ" | "P" | "I" | "CZ" | "CRZ" | "CS" | "GlobalPhase"
        )
    }

    /// Estimate time saved by using specialized implementation
    fn estimate_time_saved(&self, gate: &dyn GateOp) -> f64 {
        // Rough estimates based on gate type
        match gate.name() {
            "H" | "X" | "Y" | "Z" => 0.001, // Simple gates save ~1μs
            "RX" | "RY" | "RZ" => 0.002,    // Rotation gates save ~2μs
            "CNOT" | "CZ" => 0.005,         // Two-qubit gates save ~5μs
            "Toffoli" => 0.010,             // Three-qubit gates save ~10μs
            _ => 0.0,
        }
    }

    /// Apply single-qubit gate (generic fallback) - optimized with reusable buffer
    fn apply_single_qubit_generic(
        &mut self,
        state: &mut [Complex64],
        matrix: &[Complex64],
        target: QubitId,
        n_qubits: usize,
    ) -> QuantRS2Result<()> {
        let target_idx = target.id() as usize;

        if self.config.parallel && n_qubits >= self.config.parallel_threshold {
            // Reuse work_buffer to avoid allocation per gate
            if self.work_buffer.len() < state.len() {
                self.work_buffer
                    .resize(state.len(), Complex64::new(0.0, 0.0));
            }
            self.work_buffer[..state.len()].copy_from_slice(state);
            let state_copy = &self.work_buffer[..state.len()];

            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                let bit_val = (idx >> target_idx) & 1;
                let paired_idx = idx ^ (1 << target_idx);

                let idx0 = if bit_val == 0 { idx } else { paired_idx };
                let idx1 = if bit_val == 0 { paired_idx } else { idx };

                *amp = matrix[2 * bit_val] * state_copy[idx0]
                    + matrix[2 * bit_val + 1] * state_copy[idx1];
            });
        } else {
            // Sequential in-place update (already optimal - no allocation)
            for i in 0..(1 << n_qubits) {
                if (i >> target_idx) & 1 == 0 {
                    let j = i | (1 << target_idx);
                    let temp0 = state[i];
                    let temp1 = state[j];
                    state[i] = matrix[0] * temp0 + matrix[1] * temp1;
                    state[j] = matrix[2] * temp0 + matrix[3] * temp1;
                }
            }
        }

        Ok(())
    }

    /// Apply two-qubit gate (generic fallback) - optimized with reusable buffer
    fn apply_two_qubit_generic(
        &mut self,
        state: &mut [Complex64],
        matrix: &[Complex64],
        control: QubitId,
        target: QubitId,
        n_qubits: usize,
    ) -> QuantRS2Result<()> {
        let control_idx = control.id() as usize;
        let target_idx = target.id() as usize;

        if control_idx == target_idx {
            return Err(QuantRS2Error::CircuitValidationFailed(
                "Control and target must be different".into(),
            ));
        }

        // Ensure work_buffer is large enough (reused across calls)
        if self.work_buffer.len() < state.len() {
            self.work_buffer
                .resize(state.len(), Complex64::new(0.0, 0.0));
        }

        if self.config.parallel && n_qubits >= self.config.parallel_threshold {
            // Copy state to work buffer for reading
            self.work_buffer[..state.len()].copy_from_slice(state);
            let state_copy = &self.work_buffer[..state.len()];

            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                let ctrl_bit = (idx >> control_idx) & 1;
                let tgt_bit = (idx >> target_idx) & 1;
                let basis_idx = (ctrl_bit << 1) | tgt_bit;

                let idx00 = idx & !(1 << control_idx) & !(1 << target_idx);
                let idx01 = idx00 | (1 << target_idx);
                let idx10 = idx00 | (1 << control_idx);
                let idx11 = idx00 | (1 << control_idx) | (1 << target_idx);

                *amp = matrix[4 * basis_idx] * state_copy[idx00]
                    + matrix[4 * basis_idx + 1] * state_copy[idx01]
                    + matrix[4 * basis_idx + 2] * state_copy[idx10]
                    + matrix[4 * basis_idx + 3] * state_copy[idx11];
            });
        } else {
            // Use work_buffer as temporary storage to avoid separate allocation
            for i in 0..state.len() {
                let ctrl_bit = (i >> control_idx) & 1;
                let tgt_bit = (i >> target_idx) & 1;
                let basis_idx = (ctrl_bit << 1) | tgt_bit;

                let i00 = i & !(1 << control_idx) & !(1 << target_idx);
                let i01 = i00 | (1 << target_idx);
                let i10 = i00 | (1 << control_idx);
                let i11 = i10 | (1 << target_idx);

                self.work_buffer[i] = matrix[4 * basis_idx] * state[i00]
                    + matrix[4 * basis_idx + 1] * state[i01]
                    + matrix[4 * basis_idx + 2] * state[i10]
                    + matrix[4 * basis_idx + 3] * state[i11];
            }

            state.copy_from_slice(&self.work_buffer[..state.len()]);
        }

        Ok(())
    }

    /// Apply multi-qubit gate (generic fallback) - optimized with reusable buffer
    fn apply_multi_qubit_generic(
        &mut self,
        state: &mut [Complex64],
        gate: &dyn GateOp,
        _n_qubits: usize,
    ) -> QuantRS2Result<()> {
        // For now, convert to matrix and apply
        // This is a placeholder for more sophisticated multi-qubit handling
        let matrix = gate.matrix()?;
        let qubits = gate.qubits();
        let gate_qubits = qubits.len();
        let gate_dim = 1 << gate_qubits;

        if matrix.len() != gate_dim * gate_dim {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Invalid matrix size for {gate_qubits}-qubit gate"
            )));
        }

        // Ensure work_buffer is large enough (reused across calls)
        if self.work_buffer.len() < state.len() {
            self.work_buffer
                .resize(state.len(), Complex64::new(0.0, 0.0));
        }

        // Apply gate by iterating over all basis states
        for idx in 0..state.len() {
            let mut basis_idx = 0;
            for (i, &qubit) in qubits.iter().enumerate() {
                if (idx >> qubit.id()) & 1 == 1 {
                    basis_idx |= 1 << i;
                }
            }

            let mut new_amp = Complex64::new(0.0, 0.0);
            for j in 0..gate_dim {
                let mut target_idx = idx;
                for (i, &qubit) in qubits.iter().enumerate() {
                    if (j >> i) & 1 != (idx >> qubit.id()) & 1 {
                        target_idx ^= 1 << qubit.id();
                    }
                }

                new_amp += matrix[basis_idx * gate_dim + j] * state[target_idx];
            }

            self.work_buffer[idx] = new_amp;
        }

        state.copy_from_slice(&self.work_buffer[..state.len()]);
        Ok(())
    }
}

/// Benchmark comparison between specialized and generic implementations
#[must_use]
pub fn benchmark_specialization(
    n_qubits: usize,
    n_gates: usize,
) -> (f64, f64, SpecializationStats) {
    use quantrs2_circuit::builder::Circuit;
    use scirs2_core::random::prelude::*;
    use std::time::Instant;

    let mut rng = thread_rng();

    // For benchmark purposes, we'll use a fixed-size circuit
    // In practice, you'd want to handle different sizes more elegantly
    assert!(
        (n_qubits == 8),
        "Benchmark currently only supports 8 qubits"
    );

    let mut circuit = Circuit::<8>::new();

    for _ in 0..n_gates {
        let gate_type = rng.random_range(0..5);
        let qubit = QubitId(rng.random_range(0..n_qubits as u32));

        match gate_type {
            0 => {
                let _ = circuit.h(qubit);
            }
            1 => {
                let _ = circuit.x(qubit);
            }
            2 => {
                let _ = circuit.ry(qubit, rng.random_range(0.0..std::f64::consts::TAU));
            }
            3 => {
                if n_qubits > 1 {
                    let qubit2 = QubitId(rng.random_range(0..n_qubits as u32));
                    if qubit != qubit2 {
                        let _ = circuit.cnot(qubit, qubit2);
                    }
                }
            }
            _ => {
                let _ = circuit.z(qubit);
            }
        }
    }

    // Run with specialized simulator
    let mut specialized_sim = SpecializedStateVectorSimulator::new(Default::default());
    let start = Instant::now();
    let _ = specialized_sim
        .run(&circuit)
        .expect("Specialized simulator benchmark failed");
    let specialized_time = start.elapsed().as_secs_f64();

    // Run with base simulator
    let mut base_sim = StateVectorSimulator::new();
    let start = Instant::now();
    let _ = base_sim
        .run(&circuit)
        .expect("Base simulator benchmark failed");
    let base_time = start.elapsed().as_secs_f64();

    (specialized_time, base_time, specialized_sim.stats.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_circuit::builder::Circuit;
    use quantrs2_core::gate::single::{Hadamard, PauliX};

    #[test]
    fn test_specialized_simulator() {
        let mut circuit = Circuit::<2>::new();
        let _ = circuit.h(QubitId(0));
        let _ = circuit.cnot(QubitId(0), QubitId(1));

        let mut sim = SpecializedStateVectorSimulator::new(Default::default());
        let state = sim
            .run(&circuit)
            .expect("Failed to run specialized simulator test circuit");

        // Should create Bell state |00> + |11>
        let expected_amp = 1.0 / std::f64::consts::SQRT_2;
        assert!((state[0].norm() - expected_amp).abs() < 1e-10);
        assert!(state[1].norm() < 1e-10);
        assert!(state[2].norm() < 1e-10);
        assert!((state[3].norm() - expected_amp).abs() < 1e-10);

        // Check stats
        assert_eq!(sim.get_stats().total_gates, 2);
        assert_eq!(sim.get_stats().specialized_gates, 2);
        assert_eq!(sim.get_stats().generic_gates, 0);
    }

    /// Regression test for the P1 finding: `reorder_gates` used to sort
    /// gates purely by first-qubit-id, with no regard for whether the
    /// reordered gates actually commute. `X(1)` followed by
    /// `CNOT(control=0, target=1)` do *not* commute (they share qubit 1),
    /// so a naive sort (which would hoist the CNOT, whose first qubit is
    /// 0, ahead of the X, whose first qubit is 1) changes the computed
    /// state. Running the same circuit with reordering enabled and
    /// disabled must now produce identical results.
    #[test]
    fn test_reorder_gates_preserves_semantics_for_noncommuting_gates() {
        let mut circuit = Circuit::<2>::new();
        let _ = circuit.x(QubitId(1));
        let _ = circuit.cnot(QubitId(0), QubitId(1));

        let reordering_config = SpecializedSimulatorConfig {
            enable_reordering: true,
            ..Default::default()
        };
        let mut sim_reordered = SpecializedStateVectorSimulator::new(reordering_config);
        let state_reordered = sim_reordered.run(&circuit).expect("reordered run failed");

        let no_reorder_config = SpecializedSimulatorConfig {
            enable_reordering: false,
            ..Default::default()
        };
        let mut sim_baseline = SpecializedStateVectorSimulator::new(no_reorder_config);
        let state_baseline = sim_baseline
            .run(&circuit)
            .expect("baseline (unreordered) run failed");

        for (i, (reordered_amp, baseline_amp)) in state_reordered
            .iter()
            .zip(state_baseline.iter())
            .enumerate()
        {
            assert!(
                (reordered_amp - baseline_amp).norm() < 1e-10,
                "reordering changed circuit semantics at index {i}: {reordered_amp:?} vs {baseline_amp:?}"
            );
        }
    }

    /// Direct unit test on `reorder_gates`: a diagonal gate (`RZ`) may be
    /// hoisted past a non-commuting, qubit-disjoint gate boundary check --
    /// but a non-diagonal gate sharing a qubit with a preceding gate must
    /// never be moved past it.
    #[test]
    fn test_gates_commute_structural_checks() {
        use quantrs2_core::gate::single::RotationZ;

        let x0 = PauliX { target: QubitId(0) };
        let x0_again = PauliX { target: QubitId(0) };
        let rz0 = RotationZ {
            target: QubitId(0),
            theta: 0.5,
        };
        let rz0_b = RotationZ {
            target: QubitId(0),
            theta: 1.5,
        };
        let x1 = PauliX { target: QubitId(1) };

        // Two X gates on the same qubit do not commute in general (X does
        // not commute with itself under this conservative structural
        // check -- it's not on the recognized diagonal list), so this must
        // be false even though X*X happens to be trivial.
        assert!(!SpecializedStateVectorSimulator::gates_commute(
            &x0, &x0_again
        ));
        // Two diagonal RZ gates on the same qubit always commute.
        assert!(SpecializedStateVectorSimulator::gates_commute(&rz0, &rz0_b));
        // Disjoint qubits always commute regardless of gate type.
        assert!(SpecializedStateVectorSimulator::gates_commute(&x0, &x1));
    }

    #[test]
    fn test_benchmark() {
        let (spec_time, base_time, stats) = benchmark_specialization(8, 20);

        println!(
            "Specialized: {:.3}ms, Base: {:.3}ms",
            spec_time * 1000.0,
            base_time * 1000.0
        );
        println!("Stats: {stats:?}");

        // Specialized should generally be faster
        assert!(spec_time <= base_time * 1.1); // Allow 10% margin
    }
}
