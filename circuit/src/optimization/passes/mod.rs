//! Individual optimization passes
//!
//! This module implements various optimization passes that can be applied to quantum circuits.
//! It re-exports passes from the sub-modules:
//! - `basic_passes`: GateCancellation, GateCommutation, GateMerging, RotationMerging
//! - `advanced_passes`: DecompositionOptimization, CostBasedOptimization, TwoQubitOptimization
//! - `rewriting_passes`: PeepholeOptimization, TemplateMatching, CircuitRewriting, ParallelizationPass

pub mod advanced_passes;
pub mod basic_passes;
pub mod rewriting_passes;

use crate::builder::Circuit;
use crate::optimization::cost_model::CostModel;
use quantrs2_core::error::QuantRS2Result;
use quantrs2_core::gate::GateOp;

// Re-export all public pass types and functions
pub use advanced_passes::{
    CostBasedOptimization, CostTarget, DecompositionOptimization, TwoQubitOptimization,
};
pub use basic_passes::{GateCancellation, GateCommutation, GateMerging, RotationMerging};
pub use rewriting_passes::{
    all_same_single_qubit, extract_rx_angle, extract_ry_angle, extract_rz_angle, is_identity_angle,
    normalise_angle, parallelize_gates, single_qubit_of, utils, CircuitRewriting, CircuitTemplate,
    ParallelizationPass, PeepholeOptimization, PeepholePattern, RewriteRule, TemplateMatching,
};

/// Trait for optimization passes (object-safe version)
pub trait OptimizationPass: Send + Sync {
    /// Name of the optimization pass
    fn name(&self) -> &str;

    /// Apply the optimization pass to a gate list
    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>>;

    /// Check if this pass should be applied
    fn should_apply(&self) -> bool {
        true
    }
}

/// Extension trait for circuit operations
pub trait OptimizationPassExt<const N: usize> {
    fn apply(&self, circuit: &Circuit<N>, cost_model: &dyn CostModel)
        -> QuantRS2Result<Circuit<N>>;
    fn should_apply_to_circuit(&self, circuit: &Circuit<N>) -> bool;
}

impl<T: OptimizationPass + ?Sized, const N: usize> OptimizationPassExt<N> for T {
    fn apply(
        &self,
        circuit: &Circuit<N>,
        cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Circuit<N>> {
        // Extract gates from the circuit as owned boxes.
        let gates: Vec<Box<dyn GateOp>> = circuit.gates_as_boxes();

        // Run the optimisation pass on the gate list.
        let optimized_gates = self.apply_to_gates(gates, cost_model)?;

        // Reconstruct a new circuit from the optimised gate list.
        Circuit::<N>::from_gates(optimized_gates)
    }

    fn should_apply_to_circuit(&self, _circuit: &Circuit<N>) -> bool {
        self.should_apply()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::Circuit;
    use crate::optimization::cost_model::AbstractCostModel;
    use quantrs2_core::gate::single::{Hadamard, PauliX};
    use quantrs2_core::qubit::QubitId;

    /// `GateCancellation` is a pass-through if nothing cancels.
    #[test]
    fn test_gate_cancellation_no_op() {
        let pass = GateCancellation::new(false);
        let cost = AbstractCostModel::default();

        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard {
                target: QubitId::new(0),
            })
            .expect("add gate");

        let result = pass.apply(&circuit, &cost).expect("apply pass");
        assert_eq!(result.num_gates(), 1, "single H should not be removed");
    }

    /// Two consecutive X gates on the same qubit should cancel (X is self-inverse).
    #[test]
    fn test_xx_cancellation_reduces_gate_count() {
        let pass = GateCancellation::new(false);
        let cost = AbstractCostModel::default();

        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(PauliX {
                target: QubitId::new(0),
            })
            .expect("add gate 1");
        circuit
            .add_gate(PauliX {
                target: QubitId::new(0),
            })
            .expect("add gate 2");

        assert_eq!(
            circuit.num_gates(),
            2,
            "circuit should have 2 gates before optimization"
        );

        let result = pass.apply(&circuit, &cost).expect("apply pass");
        assert_eq!(
            result.num_gates(),
            0,
            "X-X on same qubit should cancel to empty"
        );
    }

    /// Two consecutive H gates on the same qubit should cancel (H is self-inverse).
    #[test]
    fn test_hh_cancellation_reduces_gate_count() {
        let pass = GateCancellation::new(false);
        let cost = AbstractCostModel::default();

        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard {
                target: QubitId::new(1),
            })
            .expect("add gate 1");
        circuit
            .add_gate(Hadamard {
                target: QubitId::new(1),
            })
            .expect("add gate 2");

        assert_eq!(
            circuit.num_gates(),
            2,
            "circuit should have 2 gates before optimization"
        );

        let result = pass.apply(&circuit, &cost).expect("apply pass");
        assert_eq!(
            result.num_gates(),
            0,
            "H-H on same qubit should cancel to empty"
        );
    }

    /// X-X on *different* qubits must NOT cancel.
    #[test]
    fn test_xx_different_qubits_no_cancellation() {
        let pass = GateCancellation::new(false);
        let cost = AbstractCostModel::default();

        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(PauliX {
                target: QubitId::new(0),
            })
            .expect("add gate 1");
        circuit
            .add_gate(PauliX {
                target: QubitId::new(1),
            })
            .expect("add gate 2");

        let result = pass.apply(&circuit, &cost).expect("apply pass");
        assert_eq!(
            result.num_gates(),
            2,
            "X on qubit 0 and X on qubit 1 should not cancel"
        );
    }

    /// Verify that `OptimizationPassExt::apply` actually reconstructs the circuit
    /// (gate count changes, not just a clone).
    #[test]
    fn test_apply_ext_returns_optimized_circuit() {
        let pass = GateCancellation::new(false);
        let cost = AbstractCostModel::default();

        // Three gates: X(0), X(0), H(1)  →  X(0) and X(0) cancel, leaving H(1).
        let mut circuit = Circuit::<2>::new();
        circuit.x(QubitId::new(0)).expect("x 0");
        circuit.x(QubitId::new(0)).expect("x 0 again");
        circuit.h(QubitId::new(1)).expect("h 1");

        let result = pass.apply(&circuit, &cost).expect("apply");
        assert_eq!(result.num_gates(), 1, "only H on qubit 1 should remain");
        assert_eq!(result.gates()[0].name(), "H");
    }

    /// CNOT(0,1) followed immediately by CNOT(0,1) must cancel to empty.
    #[test]
    fn test_two_qubit_cnot_cancellation() {
        use quantrs2_core::gate::multi::CNOT;
        let pass = TwoQubitOptimization::new(false, true);
        let cost = AbstractCostModel::default();
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(CNOT {
                control: q0,
                target: q1,
            }),
            Box::new(CNOT {
                control: q0,
                target: q1,
            }),
        ];
        let result = pass.apply_to_gates(gates, &cost).expect("apply");
        assert_eq!(result.len(), 0, "CNOT(0,1)+CNOT(0,1) must cancel");
    }

    /// CNOT(a,b), CNOT(b,a), CNOT(a,b) must become SWAP(a,b).
    #[test]
    fn test_two_qubit_swap_detection() {
        use quantrs2_core::gate::multi::CNOT;
        let pass = TwoQubitOptimization::new(false, true);
        let cost = AbstractCostModel::default();
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(CNOT {
                control: q0,
                target: q1,
            }),
            Box::new(CNOT {
                control: q1,
                target: q0,
            }),
            Box::new(CNOT {
                control: q0,
                target: q1,
            }),
        ];
        let result = pass.apply_to_gates(gates, &cost).expect("apply");
        assert_eq!(
            result.len(),
            1,
            "3-CNOT SWAP pattern must become 1 SWAP gate"
        );
        assert_eq!(result[0].name(), "SWAP");
    }

    /// H(q0) and H(q1) are independent → both appear in the first layer (depth 1).
    /// CNOT(q0,q1) depends on both → appears in the second layer (depth 2).
    #[test]
    fn test_parallelize_independent_then_cnot() {
        use quantrs2_core::gate::multi::CNOT;
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: q0 }),
            Box::new(Hadamard { target: q1 }),
            Box::new(CNOT {
                control: q0,
                target: q1,
            }),
        ];
        let result = parallelize_gates(gates);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].name(), "H");
        assert_eq!(result[0].qubits()[0], q0);
        assert_eq!(result[1].name(), "H");
        assert_eq!(result[1].qubits()[0], q1);
        assert_eq!(result[2].name(), "CNOT");
    }
}
