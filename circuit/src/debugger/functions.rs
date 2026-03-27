//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::builder::Circuit;

use super::types::{DebuggerConfig, ExecutionStatus, QuantumDebugger, StepResult};

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::Hadamard;
    use quantrs2_core::qubit::QubitId;
    #[test]
    fn test_debugger_creation() {
        let circuit = Circuit::<2>::new();
        let debugger = QuantumDebugger::new(circuit);
        assert_eq!(debugger.get_execution_status(), ExecutionStatus::Ready);
    }
    #[test]
    fn test_breakpoint_management() {
        let circuit = Circuit::<2>::new();
        let mut debugger = QuantumDebugger::new(circuit);
        debugger
            .add_gate_breakpoint(0)
            .expect("add_gate_breakpoint should succeed");
        let breakpoints = debugger
            .breakpoints
            .read()
            .expect("breakpoints lock should not be poisoned");
        assert!(breakpoints.gate_breakpoints.contains(&0));
    }
    #[test]
    fn test_step_execution() {
        let mut circuit = Circuit::<1>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("add_gate should succeed");
        let mut debugger = QuantumDebugger::new(circuit);
        debugger
            .start_session()
            .expect("start_session should succeed");
        let result = debugger.step_next().expect("step_next should succeed");
        match result {
            StepResult::Success => {}
            _ => panic!("Expected successful step execution"),
        }
    }
    #[test]
    fn test_visualization_configuration() {
        let circuit = Circuit::<2>::new();
        let config = DebuggerConfig {
            enable_auto_visualization: true,
            ..Default::default()
        };
        let debugger = QuantumDebugger::with_config(circuit, config);
        let visualizer = debugger
            .visualizer
            .read()
            .expect("visualizer lock should not be poisoned");
        assert!(visualizer.config.enable_realtime);
    }
    #[test]
    fn test_performance_profiling() {
        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("add_gate Hadamard should succeed");
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("add_gate CNOT should succeed");
        let config = DebuggerConfig {
            enable_profiling: true,
            ..Default::default()
        };
        let mut debugger = QuantumDebugger::with_config(circuit, config);
        let _summary = debugger.run().expect("debugger run should succeed");
        let analysis = debugger
            .get_performance_analysis()
            .expect("get_performance_analysis should succeed");
        assert!(!analysis.suggestions.is_empty() || analysis.suggestions.is_empty());
    }
    #[test]
    fn test_error_detection() {
        let circuit = Circuit::<1>::new();
        let config = DebuggerConfig {
            enable_error_detection: true,
            ..Default::default()
        };
        let debugger = QuantumDebugger::with_config(circuit, config);
        let detector = debugger
            .error_detector
            .read()
            .expect("error_detector lock should not be poisoned");
        assert!(detector.config.enable_auto_detection);
    }
}
