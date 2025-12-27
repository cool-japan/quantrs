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

    /// Reorder gates for better performance
    fn reorder_gates(
        &self,
        gates: &[Arc<dyn GateOp + Send + Sync>],
    ) -> QuantRS2Result<Vec<Arc<dyn GateOp + Send + Sync>>> {
        // Simple reordering: group gates by qubit locality
        // This is a placeholder for more sophisticated reordering
        let mut reordered = gates.to_vec();

        // Sort by first qubit to improve cache locality
        reordered.sort_by_key(|gate| gate.qubits().first().map_or(0, quantrs2_core::QubitId::id));

        Ok(reordered)
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

    /// Apply single-qubit gate (generic fallback)
    fn apply_single_qubit_generic(
        &self,
        state: &mut [Complex64],
        matrix: &[Complex64],
        target: QubitId,
        n_qubits: usize,
    ) -> QuantRS2Result<()> {
        let target_idx = target.id() as usize;

        if self.config.parallel && n_qubits >= self.config.parallel_threshold {
            let state_copy = state.to_vec();
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                let bit_val = (idx >> target_idx) & 1;
                let paired_idx = idx ^ (1 << target_idx);

                let idx0 = if bit_val == 0 { idx } else { paired_idx };
                let idx1 = if bit_val == 0 { paired_idx } else { idx };

                *amp = matrix[2 * bit_val] * state_copy[idx0]
                    + matrix[2 * bit_val + 1] * state_copy[idx1];
            });
        } else {
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

    /// Apply two-qubit gate (generic fallback)
    fn apply_two_qubit_generic(
        &self,
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

        if self.config.parallel && n_qubits >= self.config.parallel_threshold {
            let state_copy = state.to_vec();

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
            let mut new_state = vec![Complex64::new(0.0, 0.0); state.len()];

            for i in 0..state.len() {
                let ctrl_bit = (i >> control_idx) & 1;
                let tgt_bit = (i >> target_idx) & 1;
                let basis_idx = (ctrl_bit << 1) | tgt_bit;

                let i00 = i & !(1 << control_idx) & !(1 << target_idx);
                let i01 = i00 | (1 << target_idx);
                let i10 = i00 | (1 << control_idx);
                let i11 = i10 | (1 << target_idx);

                new_state[i] = matrix[4 * basis_idx] * state[i00]
                    + matrix[4 * basis_idx + 1] * state[i01]
                    + matrix[4 * basis_idx + 2] * state[i10]
                    + matrix[4 * basis_idx + 3] * state[i11];
            }

            state.copy_from_slice(&new_state);
        }

        Ok(())
    }

    /// Apply multi-qubit gate (generic fallback)
    fn apply_multi_qubit_generic(
        &self,
        state: &mut [Complex64],
        gate: &dyn GateOp,
        n_qubits: usize,
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

        // Apply gate by iterating over all basis states
        let mut new_state = state.to_vec();

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

            new_state[idx] = new_amp;
        }

        state.copy_from_slice(&new_state);
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
        let gate_type = rng.gen_range(0..5);
        let qubit = QubitId(rng.gen_range(0..n_qubits as u32));

        match gate_type {
            0 => {
                let _ = circuit.h(qubit);
            }
            1 => {
                let _ = circuit.x(qubit);
            }
            2 => {
                let _ = circuit.ry(qubit, rng.gen_range(0.0..std::f64::consts::TAU));
            }
            3 => {
                if n_qubits > 1 {
                    let qubit2 = QubitId(rng.gen_range(0..n_qubits as u32));
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
