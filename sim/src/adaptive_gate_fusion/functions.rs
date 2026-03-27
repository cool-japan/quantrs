//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "advanced_math")]
use quantrs2_circuit::prelude::*;

use super::types::{
    AdaptiveFusionConfig, AdaptiveGateFusion, FusedGateBlock, FusionStrategy, FusionUtils,
    GateType, QuantumGate,
};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn test_quantum_gate_creation() {
        let gate = QuantumGate::new(GateType::PauliX, vec![0], vec![]);
        assert_eq!(gate.gate_type, GateType::PauliX);
        assert_eq!(gate.qubits, vec![0]);
        assert!(gate.cost > 0.0);
    }
    #[test]
    fn test_gate_commutation() {
        let gate1 = QuantumGate::new(GateType::PauliX, vec![0], vec![]);
        let gate2 = QuantumGate::new(GateType::PauliY, vec![1], vec![]);
        let gate3 = QuantumGate::new(GateType::PauliX, vec![0], vec![]);
        assert!(gate1.commutes_with(&gate2));
        assert!(gate1.commutes_with(&gate3));
    }
    #[test]
    fn test_gate_fusion_compatibility() {
        let gate1 = QuantumGate::new(GateType::RotationX, vec![0], vec![0.5]);
        let gate2 = QuantumGate::new(GateType::RotationX, vec![0], vec![0.3]);
        let gate3 = QuantumGate::new(GateType::RotationY, vec![1], vec![0.2]);
        assert!(gate1.can_fuse_with(&gate2));
        assert!(!gate1.can_fuse_with(&gate3));
    }
    #[test]
    fn test_fused_gate_block() {
        let gates = vec![
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.5]),
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.3]),
        ];
        let block =
            FusedGateBlock::new(gates).expect("Fused gate block creation should succeed in test");
        assert_eq!(block.qubits, vec![0]);
        assert!(block.improvement_factor > 0.0);
    }
    #[test]
    fn test_adaptive_fusion_config() {
        let config = AdaptiveFusionConfig::default();
        assert_eq!(config.strategy, FusionStrategy::Adaptive);
        assert_eq!(config.max_fusion_size, 8);
        assert!(config.enable_cross_qubit_fusion);
    }
    #[test]
    fn test_circuit_analysis() {
        let gates = FusionUtils::create_test_sequence("rotation_chain", 2);
        let config = AdaptiveFusionConfig::default();
        let mut fusion_engine =
            AdaptiveGateFusion::new(config).expect("Fusion engine creation should succeed in test");
        let analysis = fusion_engine
            .analyze_circuit(&gates)
            .expect("Circuit analysis should succeed in test");
        assert_eq!(analysis.original_gate_count, gates.len());
        assert!(!analysis.fusion_opportunities.is_empty());
    }
    #[test]
    fn test_fusion_utils_test_sequences() {
        let rotation_chain = FusionUtils::create_test_sequence("rotation_chain", 2);
        assert_eq!(rotation_chain.len(), 6);
        let cnot_ladder = FusionUtils::create_test_sequence("cnot_ladder", 3);
        assert_eq!(cnot_ladder.len(), 2);
        let mixed_gates = FusionUtils::create_test_sequence("mixed_gates", 2);
        assert!(!mixed_gates.is_empty());
    }
    #[test]
    fn test_fusion_potential_estimation() {
        let gates = vec![
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.1]),
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.2]),
            QuantumGate::new(GateType::RotationY, vec![1], vec![0.3]),
        ];
        let potential = FusionUtils::estimate_fusion_potential(&gates);
        assert!(potential > 0.0);
        assert!(potential <= 1.0);
    }
    #[test]
    fn test_gate_matrix_generation() {
        let pauli_x = QuantumGate::new(GateType::PauliX, vec![0], vec![]);
        assert_eq!(pauli_x.matrix.shape(), &[2, 2]);
        assert_abs_diff_eq!(pauli_x.matrix[[0, 1]].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pauli_x.matrix[[1, 0]].re, 1.0, epsilon = 1e-10);
    }
    #[test]
    fn test_circuit_depth_calculation() {
        let gates = vec![
            QuantumGate::new(GateType::Hadamard, vec![0], vec![]),
            QuantumGate::new(GateType::CNOT, vec![0, 1], vec![]),
            QuantumGate::new(GateType::RotationZ, vec![1], vec![0.5]),
        ];
        let config = AdaptiveFusionConfig::default();
        let fusion_engine =
            AdaptiveGateFusion::new(config).expect("Fusion engine creation should succeed in test");
        let depth = fusion_engine.calculate_circuit_depth(&gates);
        assert!(depth > 0);
    }
}
