//! Enhanced state vector simulator using SciRS2 features
//!
//! This module provides an enhanced state vector simulator that leverages
//! SciRS2's advanced features for better performance and memory efficiency.

use num_complex::Complex64;

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    memory_efficient::EfficientStateVector,
    register::Register,
    simd_ops,
};

#[cfg(feature = "advanced_math")]
use crate::linalg_ops;

use crate::statevector::StateVectorSimulator;

/// An enhanced state vector simulator that uses SciRS2 features
///
/// This simulator provides better performance through:
/// - SIMD acceleration for vector operations
/// - Memory-efficient state storage for large circuits
/// - Optimized linear algebra operations when available
pub struct EnhancedStateVectorSimulator {
    /// Base simulator for fallback operations
    base_simulator: StateVectorSimulator,

    /// Whether to use SIMD operations
    use_simd: bool,

    /// Whether to use memory-efficient storage for large states
    use_memory_efficient: bool,

    /// Threshold for switching to memory-efficient storage (number of qubits)
    memory_efficient_threshold: usize,
}

impl EnhancedStateVectorSimulator {
    /// Create a new enhanced state vector simulator
    pub fn new() -> Self {
        Self {
            base_simulator: StateVectorSimulator::new(),
            use_simd: true,
            use_memory_efficient: true,
            memory_efficient_threshold: 20, // Use memory-efficient storage for >20 qubits
        }
    }

    /// Set whether to use SIMD operations
    pub fn set_use_simd(&mut self, use_simd: bool) -> &mut Self {
        self.use_simd = use_simd;
        self
    }

    /// Set whether to use memory-efficient storage
    pub fn set_use_memory_efficient(&mut self, use_memory_efficient: bool) -> &mut Self {
        self.use_memory_efficient = use_memory_efficient;
        self
    }

    /// Set the threshold for switching to memory-efficient storage
    pub fn set_memory_efficient_threshold(&mut self, threshold: usize) -> &mut Self {
        self.memory_efficient_threshold = threshold;
        self
    }

    /// Apply a gate using enhanced operations when possible
    fn apply_gate_enhanced<const N: usize>(
        &self,
        state: &mut [Complex64],
        gate: &dyn GateOp,
    ) -> QuantRS2Result<()> {
        // For specific gates, use optimized implementations
        match gate.name() {
            "RZ" | "RY" | "RX" => {
                // Use SIMD phase rotation for rotation gates
                if self.use_simd && gate.num_qubits() == 1 {
                    if let Some(rotation) = gate
                        .as_any()
                        .downcast_ref::<quantrs2_core::gate::single::RotationZ>()
                    {
                        simd_ops::apply_phase_simd(state, rotation.theta);
                        return Ok(());
                    }
                }
            }
            _ => {}
        }

        // For general gates, use optimized matrix multiplication if available
        #[cfg(feature = "advanced_math")]
        {
            let matrix = gate.matrix()?;
            if gate.num_qubits() == 1 {
                // Single-qubit gate: create a 2x2 matrix and apply
                use ndarray::arr2;
                let gate_matrix = arr2(&[[matrix[0], matrix[1]], [matrix[2], matrix[3]]]);

                // Apply to each affected amplitude pair
                let target = gate.qubits()[0];
                let n_qubits = (state.len() as f64).log2() as usize;
                let target_mask = 1 << target.id();

                for i in 0..(state.len() / 2) {
                    let idx0 = (i & !(target_mask - 1)) << 1 | (i & (target_mask - 1));
                    let idx1 = idx0 | target_mask;

                    let mut local_state = vec![state[idx0], state[idx1]];
                    linalg_ops::apply_unitary(&gate_matrix.view(), &mut local_state)
                        .map_err(|e| QuantRS2Error::InvalidInput(e))?;
                    state[idx0] = local_state[0];
                    state[idx1] = local_state[1];
                }

                return Ok(());
            }
        }

        // Fallback to base implementation
        Err(QuantRS2Error::InvalidInput(
            "Enhanced gate application not available for this gate".to_string(),
        ))
    }
}

impl<const N: usize> Simulator<N> for EnhancedStateVectorSimulator {
    fn run(&self, circuit: &Circuit<N>) -> QuantRS2Result<Register<N>> {
        // Decide whether to use memory-efficient storage
        if self.use_memory_efficient && N > self.memory_efficient_threshold {
            // Use memory-efficient storage for large circuits
            let mut efficient_state = EfficientStateVector::new(N)?;

            // Apply gates
            for gate in circuit.gates() {
                // Try enhanced application first
                if self
                    .apply_gate_enhanced::<N>(efficient_state.data_mut(), gate.as_ref())
                    .is_err()
                {
                    // Fall back to base simulator if enhanced application fails
                    return self.base_simulator.run(circuit);
                }
            }

            // Normalize if using SIMD
            if self.use_simd {
                simd_ops::normalize_simd(efficient_state.data_mut())?;
            } else {
                efficient_state.normalize()?;
            }

            // Convert back to Register
            Register::with_amplitudes(efficient_state.data().to_vec())
        } else {
            // For smaller circuits, use regular state vector
            let mut state = vec![Complex64::new(0.0, 0.0); 1 << N];
            state[0] = Complex64::new(1.0, 0.0);

            // Apply gates
            for gate in circuit.gates() {
                // Try enhanced application first
                if self
                    .apply_gate_enhanced::<N>(&mut state, gate.as_ref())
                    .is_err()
                {
                    // Fall back to base simulator if enhanced application fails
                    return self.base_simulator.run(circuit);
                }
            }

            // Normalize if using SIMD
            if self.use_simd {
                simd_ops::normalize_simd(&mut state)?;
            }

            Register::with_amplitudes(state)
        }
    }
}

impl Default for EnhancedStateVectorSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::qubit::QubitId;

    #[test]
    fn test_enhanced_simulator() {
        let mut circuit = Circuit::<2>::new();
        circuit.h(QubitId(0));
        circuit.cnot(QubitId(0), QubitId(1));

        let mut simulator = EnhancedStateVectorSimulator::new();
        let result = simulator.run(&circuit).unwrap();

        // Should produce Bell state |00⟩ + |11⟩
        let probs = result.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!(probs[1].abs() < 1e-10);
        assert!(probs[2].abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }
}
