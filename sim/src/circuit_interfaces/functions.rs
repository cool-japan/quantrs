//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    CircuitInterface, CircuitInterfaceConfig, CircuitInterfaceUtils, InterfaceCircuit,
    InterfaceGate, InterfaceGateType,
};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn test_interface_gate_creation() {
        let gate = InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]);
        assert_eq!(gate.qubits, vec![0]);
        assert!(gate.is_unitary());
        assert!(!gate.is_measurement());
    }
    #[test]
    fn test_measurement_gate() {
        let gate = InterfaceGate::measurement(0, 0);
        assert!(gate.is_measurement());
        assert!(!gate.is_unitary());
        assert_eq!(gate.classical_targets, vec![0]);
    }
    #[test]
    fn test_circuit_creation() {
        let mut circuit = InterfaceCircuit::new(3, 3);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]));
        assert_eq!(circuit.gates.len(), 2);
        assert_eq!(circuit.calculate_depth(), 2);
    }
    #[test]
    fn test_circuit_optimization() {
        let mut circuit = InterfaceCircuit::new(2, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::PauliX, vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::PauliX, vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::Identity, vec![1]));
        let result = circuit.optimize();
        assert_eq!(result.gates_eliminated, 3);
    }
    #[test]
    fn test_gate_unitary_matrices() {
        let hadamard = InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]);
        let matrix = hadamard
            .unitary_matrix()
            .expect("should get Hadamard matrix");
        let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
        assert_abs_diff_eq!(matrix[[0, 0]].re, inv_sqrt2, epsilon = 1e-10);
        assert_abs_diff_eq!(matrix[[1, 1]].re, -inv_sqrt2, epsilon = 1e-10);
    }
    #[test]
    fn test_rotation_gate_merging() {
        let mut circuit = InterfaceCircuit::new(1, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::RX(0.5), vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::RX(0.3), vec![0]));
        let _ = circuit.optimize();
        assert_eq!(circuit.gates.len(), 1);
        if let InterfaceGateType::RX(angle) = &circuit.gates[0].gate_type {
            assert_abs_diff_eq!(*angle, 0.8, epsilon = 1e-10);
        } else {
            panic!("Expected merged RX gate");
        }
    }
    #[test]
    fn test_circuit_interface_creation() {
        let config = CircuitInterfaceConfig::default();
        let _interface =
            CircuitInterface::new(config.clone()).expect("should create circuit interface");
        assert!(config.auto_backend_selection);
        assert!(config.enable_optimization);
        assert_eq!(config.max_statevector_qubits, 25);
    }
    #[test]
    fn test_test_circuit_creation() {
        let ghz_circuit = CircuitInterfaceUtils::create_test_circuit("ghz", 3);
        assert_eq!(ghz_circuit.num_qubits, 3);
        assert_eq!(ghz_circuit.gates.len(), 3);
        let qft_circuit = CircuitInterfaceUtils::create_test_circuit("qft", 3);
        assert!(qft_circuit.gates.len() > 3);
    }
    #[test]
    fn test_circuit_metadata() {
        let circuit = CircuitInterfaceUtils::create_test_circuit("ghz", 4);
        assert_eq!(circuit.metadata.depth, 4);
        assert_eq!(circuit.metadata.two_qubit_gates, 3);
    }
    #[test]
    fn test_clifford_detection() {
        let mut circuit = InterfaceCircuit::new(2, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::S, vec![1]));
        let config = CircuitInterfaceConfig::default();
        let interface = CircuitInterface::new(config).expect("should create circuit interface");
        assert!(interface.is_clifford_circuit(&circuit));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::T, vec![0]));
        assert!(!interface.is_clifford_circuit(&circuit));
    }
}
