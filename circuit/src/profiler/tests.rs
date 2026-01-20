//! Tests for the profiler module

use super::*;
use crate::builder::Circuit;

mod tests {
    use super::*;
    use quantrs2_core::gate::single::Hadamard;

    #[test]
    fn test_profiler_creation() {
        let circuit = Circuit::<1>::new();
        let profiler = QuantumProfiler::new(circuit);

        assert!(profiler.config.enable_gate_profiling);
        assert!(profiler.config.enable_memory_profiling);
        assert!(profiler.config.enable_resource_profiling);
    }

    #[test]
    fn test_profiler_configuration() {
        let circuit = Circuit::<1>::new();
        let config = ProfilerConfig {
            precision_level: PrecisionLevel::Ultra,
            enable_scirs2_analysis: true,
            ..Default::default()
        };

        let profiler = QuantumProfiler::with_config(circuit, config);

        match profiler.config.precision_level {
            PrecisionLevel::Ultra => (),
            _ => panic!("Expected Ultra precision level"),
        }
    }

    #[test]
    fn test_profiling_session() {
        let mut circuit = Circuit::<1>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to circuit");

        let mut profiler = QuantumProfiler::new(circuit);
        let session_id = profiler
            .start_profiling()
            .expect("Failed to start profiling session");

        // Simulate some profiling
        std::thread::sleep(Duration::from_millis(10));

        let report = profiler
            .stop_profiling(&session_id)
            .expect("Failed to stop profiling session");
        assert_eq!(report.session_id, session_id);
    }

    #[test]
    fn test_realtime_metrics() {
        let circuit = Circuit::<1>::new();
        let profiler = QuantumProfiler::new(circuit);

        let metrics = profiler
            .get_realtime_metrics()
            .expect("Failed to get realtime metrics");
        assert!(metrics.current_metrics.len() <= 10);
    }

    #[test]
    fn test_performance_analysis() {
        let mut circuit = Circuit::<1>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to circuit");

        let mut profiler = QuantumProfiler::new(circuit);
        let _analysis = profiler
            .analyze_performance()
            .expect("Failed to analyze performance");

        // Analysis should complete without errors
    }

    #[test]
    fn test_regression_detection() {
        let circuit = Circuit::<1>::new();
        let mut profiler = QuantumProfiler::new(circuit);

        let regressions = profiler
            .detect_regressions()
            .expect("Failed to detect regressions");
        // Should return empty list for new profiler
        assert!(regressions.is_empty());
    }
}
