//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::adaptive_gate_fusion::QuantumGate;
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use scirs2_core::Complex64;

use super::types::{
    AdvancedContractionAlgorithms, ContractionStrategy, IndexType, Tensor, TensorIndex,
    TensorNetwork, TensorNetworkSimulator,
};

pub(super) fn pauli_x() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .expect("Pauli-X matrix has valid 2x2 shape")
}
pub(super) fn pauli_y() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .expect("Pauli-Y matrix has valid 2x2 shape")
}
pub(super) fn pauli_z() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .expect("Pauli-Z matrix has valid 2x2 shape")
}
pub(super) fn pauli_h() -> Array2<Complex64> {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(-inv_sqrt2, 0.0),
        ],
    )
    .expect("Hadamard matrix has valid 2x2 shape")
}
pub(super) fn cnot_matrix() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (4, 4),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .expect("CNOT matrix has valid 4x4 shape")
}
pub(super) fn rotation_x(theta: f64) -> Array2<Complex64> {
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(cos_half, 0.0),
            Complex64::new(0.0, -sin_half),
            Complex64::new(0.0, -sin_half),
            Complex64::new(cos_half, 0.0),
        ],
    )
    .expect("Rotation-X matrix has valid 2x2 shape")
}
pub(super) fn rotation_y(theta: f64) -> Array2<Complex64> {
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(cos_half, 0.0),
            Complex64::new(-sin_half, 0.0),
            Complex64::new(sin_half, 0.0),
            Complex64::new(cos_half, 0.0),
        ],
    )
    .expect("Rotation-Y matrix has valid 2x2 shape")
}
pub(super) fn rotation_z(theta: f64) -> Array2<Complex64> {
    let exp_neg = Complex64::from_polar(1.0, -theta / 2.0);
    let exp_pos = Complex64::from_polar(1.0, theta / 2.0);
    Array2::from_shape_vec(
        (2, 2),
        vec![
            exp_neg,
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            exp_pos,
        ],
    )
    .expect("Rotation-Z matrix has valid 2x2 shape")
}
/// S gate (phase gate)
pub(super) fn s_gate() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 1.0),
        ],
    )
    .expect("S gate matrix has valid 2x2 shape")
}
/// T gate (π/8 gate)
pub(super) fn t_gate() -> Array2<Complex64> {
    let phase = Complex64::from_polar(1.0, std::f64::consts::PI / 4.0);
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            phase,
        ],
    )
    .expect("T gate matrix has valid 2x2 shape")
}
/// CZ gate (controlled-Z)
pub(super) fn cz_gate() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (4, 4),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .expect("CZ gate matrix has valid 4x4 shape")
}
/// SWAP gate
pub(super) fn swap_gate() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (4, 4),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .expect("SWAP gate matrix has valid 4x4 shape")
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn test_tensor_creation() {
        let data = Array3::zeros((2, 2, 1));
        let indices = vec![
            TensorIndex {
                id: 0,
                dimension: 2,
                index_type: IndexType::Physical(0),
            },
            TensorIndex {
                id: 1,
                dimension: 2,
                index_type: IndexType::Physical(0),
            },
        ];
        let tensor = Tensor::new(data, indices, "test".to_string());
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.label, "test");
    }
    #[test]
    fn test_tensor_network_creation() {
        let network = TensorNetwork::new(3);
        assert_eq!(network.num_qubits, 3);
        assert_eq!(network.tensors.len(), 0);
    }
    #[test]
    fn test_simulator_initialization() {
        let mut sim = TensorNetworkSimulator::new(2);
        sim.initialize_zero_state()
            .expect("Failed to initialize zero state");
        assert_eq!(sim.network.tensors.len(), 2);
    }
    #[test]
    fn test_single_qubit_gate() {
        let mut sim = TensorNetworkSimulator::new(1);
        sim.initialize_zero_state()
            .expect("Failed to initialize zero state");
        let initial_tensors = sim.network.tensors.len();
        let h_gate = QuantumGate::new(
            crate::adaptive_gate_fusion::GateType::Hadamard,
            vec![0],
            vec![],
        );
        sim.apply_gate(h_gate)
            .expect("Failed to apply Hadamard gate");
        assert_eq!(sim.network.tensors.len(), initial_tensors + 1);
    }
    #[test]
    fn test_measurement() {
        let mut sim = TensorNetworkSimulator::new(1);
        sim.initialize_zero_state()
            .expect("Failed to initialize zero state");
        let result = sim.measure(0).expect("Failed to measure qubit");
        let _: bool = result;
    }
    #[test]
    fn test_contraction_strategies() {
        let _sim = TensorNetworkSimulator::new(2);
        let strat1 = ContractionStrategy::Sequential;
        let strat2 = ContractionStrategy::Greedy;
        let strat3 = ContractionStrategy::Custom(vec![0, 1]);
        assert_ne!(strat1, strat2);
        assert_ne!(strat2, strat3);
    }
    #[test]
    fn test_gate_matrices() {
        let h = pauli_h();
        assert_abs_diff_eq!(h[[0, 0]].re, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
        let x = pauli_x();
        assert_abs_diff_eq!(x[[0, 1]].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[[1, 0]].re, 1.0, epsilon = 1e-10);
    }
    #[test]
    fn test_enhanced_tensor_contraction() {
        let mut id_gen = 0;
        let tensor_a = Tensor::identity(0, &mut id_gen);
        let tensor_b = Tensor::identity(0, &mut id_gen);
        let result = tensor_a.contract(&tensor_b, 1, 0);
        assert!(result.is_ok());
        let contracted = result.expect("Failed to contract tensors");
        assert!(!contracted.data.is_empty());
    }
    #[test]
    fn test_contraction_cost_estimation() {
        let network = TensorNetwork::new(2);
        let mut id_gen = 0;
        let tensor_a = Tensor::identity(0, &mut id_gen);
        let tensor_b = Tensor::identity(1, &mut id_gen);
        let cost = network.estimate_contraction_cost(&tensor_a, &tensor_b);
        assert!(cost > 0.0);
        assert!(cost.is_finite());
    }
    #[test]
    fn test_optimal_contraction_order() {
        let mut network = TensorNetwork::new(3);
        let mut id_gen = 0;
        for i in 0..3 {
            let tensor = Tensor::identity(i, &mut id_gen);
            network.add_tensor(tensor);
        }
        let order = network.find_optimal_contraction_order();
        assert!(order.is_ok());
        let order_vec = order.expect("Failed to find optimal contraction order");
        assert!(!order_vec.is_empty());
    }
    #[test]
    fn test_greedy_contraction_strategy() {
        let mut simulator =
            TensorNetworkSimulator::new(2).with_strategy(ContractionStrategy::Greedy);
        let mut id_gen = 0;
        for i in 0..2 {
            let tensor = Tensor::identity(i, &mut id_gen);
            simulator.network.add_tensor(tensor);
        }
        let result = simulator.contract_greedy();
        assert!(result.is_ok());
        let amplitude = result.expect("Failed to contract network");
        assert!(amplitude.norm() >= 0.0);
    }
    #[test]
    fn test_basis_state_boundary_conditions() {
        let mut network = TensorNetwork::new(2);
        let mut id_gen = 0;
        for i in 0..2 {
            let tensor = Tensor::identity(i, &mut id_gen);
            network.add_tensor(tensor);
        }
        let result = network.set_basis_state_boundary(1);
        assert!(result.is_ok());
    }
    #[test]
    fn test_full_state_vector_contraction() {
        let simulator = TensorNetworkSimulator::new(2);
        let result = simulator.contract_network_to_state_vector();
        assert!(result.is_ok());
        let state_vector = result.expect("Failed to contract network to state vector");
        assert_eq!(state_vector.len(), 4);
        assert!((state_vector[0].norm() - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_advanced_contraction_algorithms() {
        let mut id_gen = 0;
        let tensor = Tensor::identity(0, &mut id_gen);
        let qr_result = AdvancedContractionAlgorithms::hotqr_decomposition(&tensor);
        assert!(qr_result.is_ok());
        let (q, r) = qr_result.expect("Failed to perform HOTQR decomposition");
        assert_eq!(q.label, "Q");
        assert_eq!(r.label, "R");
    }
    #[test]
    fn test_tree_contraction() {
        let mut id_gen = 0;
        let tensors = vec![
            Tensor::identity(0, &mut id_gen),
            Tensor::identity(1, &mut id_gen),
        ];
        let result = AdvancedContractionAlgorithms::tree_contraction(&tensors);
        assert!(result.is_ok());
        let amplitude = result.expect("Failed to perform tree contraction");
        assert!(amplitude.norm() >= 0.0);
    }
}
