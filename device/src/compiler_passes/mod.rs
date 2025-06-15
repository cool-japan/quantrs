//! Hardware-specific compiler passes for quantum circuit optimization
//!
//! This module provides advanced compiler passes that leverage hardware-specific
//! information including topology, calibration data, noise models, and backend
//! capabilities to optimize quantum circuits for specific hardware platforms.

pub mod compiler;
pub mod config;
pub mod optimization;
pub mod passes;
pub mod test_utils;
pub mod types;

// Re-export commonly used types
pub use config::{
    AnalysisDepth, CompilerConfig, HardwareConstraints, OptimizationObjective, ParallelConfig,
    PassConfig, PassPriority, SciRS2Config, SciRS2OptimizationMethod,
};

pub use types::{
    AdvancedMetrics, AllocationStrategy, AzureProvider, BraketProvider, CompilationResult,
    CompilationTarget, ComplexityMetrics, ConnectivityPattern, GoogleGateSet, GridTopology,
    HardwareAllocation, OptimizationStats, PassInfo, PerformancePrediction, PlatformConstraints,
    PlatformSpecificResults, RigettiLattice, VerificationResults,
};

pub use optimization::{
    AdvancedCrosstalkMitigation, CrosstalkAnalysisResult, CrosstalkModel, GraphOptimizationResult,
    MitigationStrategyType, SciRS2OptimizationEngine, StatisticalAnalysisResult, TrendDirection,
};

pub use passes::{
    CompilerPass, PassCoordinator, PassExecutionResult, PerformanceMetrics, PerformanceMonitor,
    PerformanceSummary,
};

pub use compiler::HardwareCompiler;

// Re-export test utilities when testing
#[cfg(test)]
pub use test_utils::*;

// Helper functions for creating common configurations
pub fn create_standard_topology(
    topology_type: &str,
    num_qubits: usize,
) -> crate::DeviceResult<crate::topology::HardwareTopology> {
    match topology_type {
        "linear" => Ok(crate::topology::HardwareTopology::linear_topology(
            num_qubits,
        )),
        "grid" => {
            let side = (num_qubits as f64).sqrt() as usize;
            Ok(crate::topology::HardwareTopology::grid_topology(side, side))
        }
        "complete" => {
            let mut topology = crate::topology::HardwareTopology::new(num_qubits);
            // Add all possible connections for complete graph
            for i in 0..num_qubits {
                for j in i + 1..num_qubits {
                    topology.add_connection(
                        i as u32,
                        j as u32,
                        crate::topology::GateProperties {
                            error_rate: 0.01,
                            duration: 100.0,
                            gate_type: "CZ".to_string(),
                        },
                    );
                }
            }
            Ok(topology)
        }
        _ => Err(crate::DeviceError::InvalidInput(format!(
            "Unknown topology type: {}",
            topology_type
        ))),
    }
}

pub fn create_ideal_calibration(
    device_name: String,
    _num_qubits: usize,
) -> crate::calibration::DeviceCalibration {
    use std::collections::HashMap;
    use std::time::{Duration, SystemTime};

    // Create default calibration with minimal required fields
    crate::calibration::DeviceCalibration {
        device_id: device_name,
        timestamp: SystemTime::now(),
        valid_duration: Duration::from_secs(3600),
        qubit_calibrations: HashMap::new(),
        single_qubit_gates: HashMap::new(),
        two_qubit_gates: HashMap::new(),
        multi_qubit_gates: HashMap::new(),
        readout_calibration: crate::calibration::ReadoutCalibration::default(),
        crosstalk_matrix: crate::calibration::CrosstalkMatrix::default(),
        topology: crate::calibration::DeviceTopology::default(),
        metadata: HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_traits::BackendCapabilities;
    use quantrs2_circuit::prelude::Circuit;
    use quantrs2_core::qubit::QubitId;

    #[test]
    fn test_compiler_config_default() {
        let config = CompilerConfig::default();
        assert!(config.enable_gate_synthesis);
        assert!(config.enable_error_optimization);
        assert_eq!(config.max_iterations, 50);
        assert_eq!(config.tolerance, 1e-6);
    }

    #[test]
    fn test_compilation_targets() {
        let ibm_target = CompilationTarget::IBMQuantum {
            backend_name: "ibmq_qasm_simulator".to_string(),
            coupling_map: vec![(0, 1), (1, 2)],
            native_gates: ["rz", "sx", "cx"].iter().map(|s| s.to_string()).collect(),
            basis_gates: vec!["rz".to_string(), "sx".to_string(), "cx".to_string()],
            max_shots: 8192,
            simulator: true,
        };

        match ibm_target {
            CompilationTarget::IBMQuantum { backend_name, .. } => {
                assert_eq!(backend_name, "ibmq_qasm_simulator");
            }
            _ => panic!("Expected IBM Quantum target"),
        }
    }

    #[test]
    fn test_parallel_config() {
        let parallel_config = ParallelConfig {
            enable_parallel_passes: true,
            num_threads: 4,
            chunk_size: 100,
            enable_simd: true,
        };

        assert!(parallel_config.enable_parallel_passes);
        assert_eq!(parallel_config.num_threads, 4);
        assert!(parallel_config.enable_simd);
    }

    #[tokio::test]
    async fn test_advanced_compilation() {
        let topology = create_standard_topology("linear", 4).unwrap();
        let calibration = create_ideal_calibration("test".to_string(), 4);
        let config = CompilerConfig::default();
        let backend_capabilities = BackendCapabilities::default();

        let compiler =
            HardwareCompiler::new(config, topology, calibration, None, backend_capabilities)
                .unwrap();

        let mut circuit = Circuit::<4>::new();
        circuit.h(QubitId(0));
        circuit.cnot(QubitId(0), QubitId(1));
        circuit.cnot(QubitId(1), QubitId(2));

        let result = compiler.compile_circuit(&circuit).await.unwrap();
        assert!(!result.applied_passes.is_empty());
        assert!(!result.optimization_history.is_empty());
        assert!(result.verification_results.equivalence_verified);
    }

    #[test]
    fn test_topology_creation() {
        let linear_topology = create_standard_topology("linear", 4).unwrap();
        assert!(linear_topology.num_qubits >= 4);

        let grid_topology = create_standard_topology("grid", 4).unwrap();
        assert!(grid_topology.num_qubits >= 4);
    }

    #[test]
    fn test_calibration_creation() {
        let calibration = create_ideal_calibration("test_device".to_string(), 3);
        assert_eq!(calibration.device_id, "test_device");
        assert_eq!(calibration.single_qubit_gates.len(), 0); // Empty for ideal calibration
    }
}
