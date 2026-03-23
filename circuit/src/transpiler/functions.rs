//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::builder::Circuit;

use super::types::{DeviceTranspiler, HardwareSpec, TranspilationOptions, TranspilationStrategy};

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::Hadamard;
    use quantrs2_core::qubit::QubitId;
    #[test]
    #[ignore = "slow test: creates large coupling maps (1000+ qubits)"]
    fn test_transpiler_creation() {
        let transpiler = DeviceTranspiler::new();
        assert!(!transpiler.available_devices().is_empty());
    }
    #[test]
    fn test_hardware_spec_creation() {
        let spec = HardwareSpec::ibm_quantum();
        assert_eq!(spec.name, "ibm_quantum");
        assert!(spec.native_gates.single_qubit.contains("H"));
        assert!(spec.native_gates.two_qubit.contains("CNOT"));
    }
    #[test]
    #[ignore = "slow test: uses default options with large coupling maps"]
    fn test_transpilation_options() {
        let options = TranspilationOptions {
            strategy: TranspilationStrategy::MinimizeDepth,
            max_iterations: 5,
            ..Default::default()
        };
        assert_eq!(options.strategy, TranspilationStrategy::MinimizeDepth);
        assert_eq!(options.max_iterations, 5);
    }
    #[test]
    #[ignore = "slow test: loads multiple hardware specs with large coupling maps"]
    fn test_native_gate_checking() {
        let transpiler = DeviceTranspiler::new();
        let spec = HardwareSpec::ibm_quantum();
        let h_gate = Hadamard { target: QubitId(0) };
        assert!(transpiler.is_native_gate(&h_gate, &spec));
    }
    #[test]
    #[ignore = "slow test: creates transpiler with large coupling maps"]
    fn test_needs_decomposition() {
        let transpiler = DeviceTranspiler::new();
        let spec = HardwareSpec::ibm_quantum();
        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("add H gate to circuit");
        assert!(!transpiler.needs_decomposition(&circuit, &spec));
    }
}
