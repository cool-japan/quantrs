//! Just-in-time compilation for frequently used gate sequences.
//!
//! This module provides advanced JIT compilation capabilities for quantum circuit
//! simulation, enabling compilation of frequently used gate sequences into optimized
//! machine code for dramatic performance improvements.

mod analyzer;
mod compiler;
mod profiler;
mod simulator;
mod types;

// Re-export all public types
pub use analyzer::*;
pub use compiler::*;
pub use profiler::*;
pub use simulator::*;
pub use types::*;

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::circuit_interfaces::{InterfaceGate, InterfaceGateType};
    use scirs2_core::Complex64;

    #[test]
    fn test_jit_compiler_creation() {
        let config = JITConfig::default();
        let compiler = JITCompiler::new(config);
        let stats = compiler.get_stats();
        assert_eq!(stats.total_compilations, 0);
    }

    #[test]
    fn test_pattern_extraction() {
        let config = JITConfig::default();
        let _compiler = JITCompiler::new(config);

        let gates = vec![
            InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]),
            InterfaceGate::new(InterfaceGateType::PauliX, vec![1]),
        ];

        let pattern =
            JITCompiler::extract_pattern(&gates).expect("Pattern extraction should succeed");
        assert_eq!(pattern.gate_types.len(), 2);
        assert_eq!(pattern.frequency, 1);
    }

    #[test]
    fn test_gate_matrix_generation() {
        let config = JITConfig::default();
        let _compiler = JITCompiler::new(config);

        let pauli_x = JITCompiler::get_gate_matrix(&InterfaceGateType::PauliX)
            .expect("PauliX matrix generation should succeed");
        assert_eq!(pauli_x.shape(), [2, 2]);
        assert_eq!(pauli_x[(0, 1)], Complex64::new(1.0, 0.0));
        assert_eq!(pauli_x[(1, 0)], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_pattern_analysis() {
        let mut analyzer = PatternAnalyzer::new();

        let gates = vec![
            InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]),
            InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]),
        ];

        let result = analyzer.analyze_pattern(&gates);
        assert_eq!(result.frequency, 1);
        assert!(result
            .optimization_suggestions
            .contains(&OptimizationSuggestion::GateFusion));
    }

    #[test]
    fn test_complexity_analysis() {
        let analyzer = ComplexityAnalyzer::new();

        let gates = vec![
            InterfaceGate::new(InterfaceGateType::PauliX, vec![0]),
            InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]),
        ];

        let complexity = analyzer.analyze_complexity(&gates);
        assert_eq!(complexity.gate_count, 2);
        assert!(complexity.computational_cost > 0.0);
    }

    #[test]
    fn test_jit_simulator_creation() {
        let config = JITConfig::default();
        let simulator = JITQuantumSimulator::new(2, config);

        assert_eq!(simulator.num_qubits, 2);
        assert_eq!(simulator.get_state().len(), 4);
        assert_eq!(simulator.get_state()[0], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_gate_application() {
        let config = JITConfig::default();
        let mut simulator = JITQuantumSimulator::new(1, config);

        let gate = InterfaceGate::new(InterfaceGateType::PauliX, vec![0]);

        simulator
            .apply_gate_interpreted(&gate)
            .expect("PauliX gate application should succeed");

        // After Pauli-X, state should be |1⟩
        assert_eq!(simulator.get_state()[0], Complex64::new(0.0, 0.0));
        assert_eq!(simulator.get_state()[1], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_hadamard_gate() {
        let config = JITConfig::default();
        let mut simulator = JITQuantumSimulator::new(1, config);

        let gate = InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]);

        simulator
            .apply_gate_interpreted(&gate)
            .expect("Hadamard gate application should succeed");

        // After Hadamard, state should be (|0⟩ + |1⟩)/√2
        let sqrt2_inv = 1.0 / (2.0_f64).sqrt();
        assert!((simulator.get_state()[0].re - sqrt2_inv).abs() < 1e-10);
        assert!((simulator.get_state()[1].re - sqrt2_inv).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_gate() {
        let config = JITConfig::default();
        let mut simulator = JITQuantumSimulator::new(2, config);

        // Prepare |10⟩ state by applying X gate to qubit 1
        // In little-endian convention: |10⟩ means qubit 1 is |1⟩, qubit 0 is |0⟩
        let x_gate = InterfaceGate::new(InterfaceGateType::PauliX, vec![1]);
        simulator
            .apply_gate_interpreted(&x_gate)
            .expect("PauliX gate application should succeed");

        // Verify we have |10⟩ state (index 2 in little-endian)
        assert!((simulator.get_state()[0].norm() - 0.0).abs() < 1e-10);
        assert!((simulator.get_state()[1].norm() - 0.0).abs() < 1e-10);
        assert!((simulator.get_state()[2].norm() - 1.0).abs() < 1e-10);
        assert!((simulator.get_state()[3].norm() - 0.0).abs() < 1e-10);

        let gate = InterfaceGate::new(InterfaceGateType::CNOT, vec![1, 0]);

        simulator
            .apply_gate_interpreted(&gate)
            .expect("CNOT gate application should succeed");

        // After CNOT with control=1, target=0: |10⟩ → |11⟩ (index 3)
        assert!((simulator.get_state()[0].norm() - 0.0).abs() < 1e-10);
        assert!((simulator.get_state()[1].norm() - 0.0).abs() < 1e-10);
        assert!((simulator.get_state()[2].norm() - 0.0).abs() < 1e-10);
        assert!((simulator.get_state()[3].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_gates() {
        let config = JITConfig::default();
        let mut simulator = JITQuantumSimulator::new(1, config);

        // Test RX gate
        let gate_rx = InterfaceGate::new(InterfaceGateType::RX(std::f64::consts::PI), vec![0]);

        simulator
            .apply_gate_interpreted(&gate_rx)
            .expect("RX gate application should succeed");

        // RX(π) should be equivalent to Pauli-X up to global phase
        assert!((simulator.get_state()[0].norm() - 0.0).abs() < 1e-10);
        assert!((simulator.get_state()[1].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gate_sequence_compilation() {
        let mut config = JITConfig::default();
        config.compilation_threshold = 1; // Compile after 1 usage

        let mut simulator = JITQuantumSimulator::new(2, config);

        let sequence = vec![
            InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]),
            InterfaceGate::new(InterfaceGateType::PauliX, vec![1]),
        ];

        // First execution should be interpreted
        let _time1 = simulator
            .apply_gate_sequence(&sequence)
            .expect("First gate sequence should succeed");
        assert_eq!(simulator.get_stats().interpreted_executions, 1);

        // Second execution might be compiled
        let _time2 = simulator
            .apply_gate_sequence(&sequence)
            .expect("Second gate sequence should succeed");
        assert!(simulator.get_compiler_stats().patterns_analyzed > 0);
    }

    #[test]
    fn test_optimization_suggestions() {
        let mut analyzer = PatternAnalyzer::new();

        let gates = vec![
            InterfaceGate::new(InterfaceGateType::RX(std::f64::consts::PI / 4.0), vec![0]),
            InterfaceGate::new(InterfaceGateType::RY(std::f64::consts::PI / 2.0), vec![0]),
        ];

        let result = analyzer.analyze_pattern(&gates);
        assert!(result
            .optimization_suggestions
            .contains(&OptimizationSuggestion::GateFusion));
    }

    #[test]
    fn test_runtime_profiler() {
        use std::time::Duration;

        let mut profiler = RuntimeProfiler::new();

        profiler.record_execution_time(Duration::from_millis(100));
        profiler.record_execution_time(Duration::from_millis(200));
        profiler.record_memory_usage(1024);
        profiler.record_memory_usage(2048);

        let stats = profiler.get_stats();
        assert_eq!(stats.sample_count, 2);
        assert_eq!(stats.average_memory_usage, 1536);
        assert_eq!(stats.peak_memory_usage, 2048);
    }

    #[test]
    fn test_constant_folding_optimization() {
        let config = JITConfig::default();
        let _compiler = JITCompiler::new(config);

        let mut instructions = vec![
            BytecodeInstruction::ApplySingleQubit {
                gate_type: InterfaceGateType::RX(0.0), // Zero rotation
                target: 0,
            },
            BytecodeInstruction::ApplySingleQubit {
                gate_type: InterfaceGateType::RY(std::f64::consts::PI),
                target: 0,
            },
        ];

        JITCompiler::apply_constant_folding(&mut instructions)
            .expect("Constant folding should succeed");

        // Check that zero rotation was folded to identity
        if let BytecodeInstruction::ApplySingleQubit { gate_type, .. } = &instructions[0] {
            assert_eq!(*gate_type, InterfaceGateType::Identity);
        }
    }

    #[test]
    fn test_dead_code_elimination() {
        let config = JITConfig::default();
        let _compiler = JITCompiler::new(config);

        let mut instructions = vec![
            BytecodeInstruction::ApplySingleQubit {
                gate_type: InterfaceGateType::Identity,
                target: 0,
            },
            BytecodeInstruction::ApplySingleQubit {
                gate_type: InterfaceGateType::RY(std::f64::consts::PI),
                target: 0,
            },
        ];

        let original_len = instructions.len();
        JITCompiler::apply_dead_code_elimination(&mut instructions)
            .expect("Dead code elimination should succeed");

        assert!(instructions.len() <= original_len);
    }

    #[test]
    fn test_benchmark_jit_compilation() {
        let results =
            benchmark_jit_compilation().expect("JIT benchmark should complete successfully");

        assert!(results.total_sequences > 0);
        assert!(results.compilation_success_rate >= 0.0);
        assert!(results.compilation_success_rate <= 1.0);
        assert!(results.speedup_factor >= 0.0);
    }
}
