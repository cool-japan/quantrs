//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use crate::error::{Result, SimulatorError};
use std::time::{Duration, Instant};

use super::types::{HardwareArchitecture, HardwareAwareConfig, HardwareAwareQMLOptimizer};

/// Benchmark function for hardware-aware QML optimization
pub fn benchmark_hardware_aware_qml() -> Result<()> {
    println!("Benchmarking Hardware-Aware QML Optimization...");
    let config = HardwareAwareConfig {
        target_architecture: HardwareArchitecture::IBMQuantum,
        ..Default::default()
    };
    let mut optimizer = HardwareAwareQMLOptimizer::new(config)?;
    let mut circuit = InterfaceCircuit::new(4, 0);
    circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]));
    circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]));
    circuit.add_gate(InterfaceGate::new(InterfaceGateType::RY(0.5), vec![2]));
    circuit.add_gate(InterfaceGate::new(
        InterfaceGateType::Toffoli,
        vec![0, 1, 2],
    ));
    let start_time = Instant::now();
    let optimized_result = optimizer.optimize_qml_circuit(&circuit, None)?;
    let duration = start_time.elapsed();
    println!("✅ Hardware-Aware QML Optimization Results:");
    println!("   Original Gates: {}", circuit.gates.len());
    println!(
        "   Optimized Gates: {}",
        optimized_result.circuit.gates.len()
    );
    println!(
        "   Gate Count Optimization: {:?}",
        optimized_result.gate_count_optimization
    );
    println!(
        "   Depth Optimization: {:?}",
        optimized_result.depth_optimization
    );
    println!(
        "   Expected Error Rate: {:.6}",
        optimized_result.expected_error_rate
    );
    println!(
        "   Gates Eliminated: {}",
        optimized_result.optimization_stats.gates_eliminated
    );
    println!(
        "   SWAP Gates Added: {}",
        optimized_result.optimization_stats.swap_gates_inserted
    );
    println!(
        "   Compilation Time: {}ms",
        optimized_result.compilation_time_ms
    );
    println!("   Total Optimization Time: {:.2}ms", duration.as_millis());
    let ansatz_circuit = optimizer.generate_hardware_efficient_ansatz(4, 3, 0.8)?;
    println!("   Generated Ansatz Gates: {}", ansatz_circuit.gates.len());
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_hardware_aware_optimizer_creation() {
        let config = HardwareAwareConfig::default();
        let optimizer = HardwareAwareQMLOptimizer::new(config);
        assert!(optimizer.is_ok());
    }
    #[test]
    fn test_circuit_analysis() {
        let config = HardwareAwareConfig::default();
        let optimizer = HardwareAwareQMLOptimizer::new(config)
            .expect("Optimizer creation should succeed in test");
        let mut circuit = InterfaceCircuit::new(2, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]));
        let analysis = optimizer.analyze_circuit(&circuit);
        assert!(analysis.is_ok());
        let analysis = analysis.expect("Circuit analysis should succeed in test");
        assert_eq!(analysis.two_qubit_gates.len(), 1);
        assert!(analysis.gate_counts.contains_key("Hadamard"));
    }
    #[test]
    fn test_qubit_mapping_optimization() {
        let config = HardwareAwareConfig::default();
        let optimizer = HardwareAwareQMLOptimizer::new(config)
            .expect("Optimizer creation should succeed in test");
        let circuit = InterfaceCircuit::new(4, 0);
        let analysis = optimizer
            .analyze_circuit(&circuit)
            .expect("Circuit analysis should succeed in test");
        let mapping = optimizer.optimize_qubit_mapping(&circuit, &analysis);
        assert!(mapping.is_ok());
        let mapping = mapping.expect("Qubit mapping optimization should succeed in test");
        assert_eq!(mapping.len(), 4);
    }
    #[test]
    fn test_hardware_specific_optimizations() {
        let config = HardwareAwareConfig {
            target_architecture: HardwareArchitecture::IBMQuantum,
            ..Default::default()
        };
        let optimizer = HardwareAwareQMLOptimizer::new(config)
            .expect("Optimizer creation should succeed in test");
        let mut circuit = InterfaceCircuit::new(2, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::RZ(0.1), vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::RZ(0.2), vec![0]));
        let original_gates = circuit.gates.len();
        optimizer
            .apply_ibm_optimizations(&mut circuit)
            .expect("IBM optimizations should succeed in test");
        assert!(circuit.gates.len() <= original_gates);
    }
    #[test]
    fn test_gate_cancellation() {
        let config = HardwareAwareConfig::default();
        let optimizer = HardwareAwareQMLOptimizer::new(config)
            .expect("Optimizer creation should succeed in test");
        let gate1 = InterfaceGate::new(InterfaceGateType::PauliX, vec![0]);
        let gate2 = InterfaceGate::new(InterfaceGateType::PauliX, vec![0]);
        assert!(optimizer.gates_cancel(&gate1, &gate2));
        let gate3 = InterfaceGate::new(InterfaceGateType::PauliY, vec![0]);
        assert!(!optimizer.gates_cancel(&gate1, &gate3));
    }
    #[test]
    fn test_error_rate_estimation() {
        let config = HardwareAwareConfig::default();
        let optimizer = HardwareAwareQMLOptimizer::new(config)
            .expect("Optimizer creation should succeed in test");
        let mut circuit = InterfaceCircuit::new(2, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]));
        let error_rate = optimizer.estimate_error_rate(&circuit);
        assert!(error_rate.is_ok());
        assert!(error_rate.expect("Error rate estimation should succeed in test") > 0.0);
    }
    #[test]
    fn test_cross_device_compatibility() {
        let config = HardwareAwareConfig::default();
        let optimizer = HardwareAwareQMLOptimizer::new(config)
            .expect("Optimizer creation should succeed in test");
        let compatibility = optimizer.get_cross_device_compatibility(
            HardwareArchitecture::IBMQuantum,
            HardwareArchitecture::GoogleQuantumAI,
        );
        assert!((0.0..=1.0).contains(&compatibility));
    }
}
