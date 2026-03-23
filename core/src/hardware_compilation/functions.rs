//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    matrix_ops::DenseMatrix,
    pulse::PulseSequence,
    qubit::QubitId,
    synthesis::decompose_two_qubit_kak,
};
use scirs2_core::ndarray::Array2;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use super::types::{
    CompiledGate, HardwareCompilationConfig, HardwareCompiler, HardwarePlatform, HardwareTopology,
    NativeGateSet, NativeGateType, OptimizedSequence, PlatformConstraints,
    SuperconductingOptimizer,
};

/// Platform-specific optimization trait
pub trait PlatformOptimizer: std::fmt::Debug + Send + Sync {
    /// Optimize gate sequence for specific platform
    fn optimize_sequence(
        &self,
        gates: &[CompiledGate],
        config: &HardwareCompilationConfig,
    ) -> QuantRS2Result<OptimizedSequence>;
    /// Estimate sequence fidelity
    fn estimate_fidelity(&self, sequence: &[CompiledGate]) -> f64;
    /// Get platform-specific constraints
    fn get_constraints(&self) -> PlatformConstraints;
}
/// Helper functions for creating platform-specific gate sets
pub(super) fn create_superconducting_gate_set() -> NativeGateSet {
    let mut gate_fidelities = HashMap::new();
    gate_fidelities.insert(NativeGateType::VirtualZ, 1.0);
    gate_fidelities.insert(NativeGateType::Rx, 0.9995);
    gate_fidelities.insert(NativeGateType::Ry, 0.9995);
    gate_fidelities.insert(NativeGateType::CNOT, 0.995);
    let mut gate_durations = HashMap::new();
    gate_durations.insert(NativeGateType::VirtualZ, Duration::from_nanos(0));
    gate_durations.insert(NativeGateType::Rx, Duration::from_nanos(20));
    gate_durations.insert(NativeGateType::Ry, Duration::from_nanos(20));
    gate_durations.insert(NativeGateType::CNOT, Duration::from_nanos(300));
    NativeGateSet {
        single_qubit_gates: vec![
            NativeGateType::Rx,
            NativeGateType::Ry,
            NativeGateType::VirtualZ,
        ],
        two_qubit_gates: vec![NativeGateType::CNOT],
        multi_qubit_gates: vec![],
        parametric_constraints: HashMap::new(),
        gate_fidelities,
        gate_durations,
    }
}
pub(super) fn create_trapped_ion_gate_set() -> NativeGateSet {
    let mut gate_fidelities = HashMap::new();
    gate_fidelities.insert(NativeGateType::Rx, 0.9999);
    gate_fidelities.insert(NativeGateType::Ry, 0.9999);
    gate_fidelities.insert(NativeGateType::Rz, 0.9999);
    gate_fidelities.insert(NativeGateType::MS, 0.998);
    let mut gate_durations = HashMap::new();
    gate_durations.insert(NativeGateType::Rx, Duration::from_micros(10));
    gate_durations.insert(NativeGateType::Ry, Duration::from_micros(10));
    gate_durations.insert(NativeGateType::Rz, Duration::from_micros(1));
    gate_durations.insert(NativeGateType::MS, Duration::from_micros(100));
    NativeGateSet {
        single_qubit_gates: vec![NativeGateType::Rx, NativeGateType::Ry, NativeGateType::Rz],
        two_qubit_gates: vec![NativeGateType::MS],
        multi_qubit_gates: vec![NativeGateType::MS],
        parametric_constraints: HashMap::new(),
        gate_fidelities,
        gate_durations,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubit::QubitId;
    use scirs2_core::Complex64;
    use std::collections::{HashMap, HashSet};
    fn create_test_topology() -> HardwareTopology {
        let mut connectivity = HashMap::new();
        let mut qubit_positions = HashMap::new();
        for i in 0..4 {
            let qubit = QubitId::new(i);
            qubit_positions.insert(qubit, (i as f64, 0.0, 0.0));
            let mut neighbors = HashSet::new();
            if i > 0 {
                neighbors.insert(QubitId::new(i - 1));
            }
            if i < 3 {
                neighbors.insert(QubitId::new(i + 1));
            }
            connectivity.insert(qubit, neighbors);
        }
        HardwareTopology {
            connectivity,
            qubit_positions,
            coupling_strengths: HashMap::new(),
            crosstalk_matrix: Array2::zeros((4, 4)),
            max_parallel_ops: 2,
        }
    }
    #[test]
    fn test_superconducting_compiler_creation() {
        let topology = create_test_topology();
        let compiler = HardwareCompiler::for_superconducting(topology);
        assert!(compiler.is_ok());
        let compiler = compiler.expect("superconducting compiler creation failed");
        assert_eq!(compiler.config.platform, HardwarePlatform::Superconducting);
        assert!(compiler
            .config
            .native_gates
            .single_qubit_gates
            .contains(&NativeGateType::VirtualZ));
    }
    #[test]
    fn test_trapped_ion_compiler_creation() {
        let topology = create_test_topology();
        let compiler = HardwareCompiler::for_trapped_ion(topology);
        assert!(compiler.is_ok());
        let compiler = compiler.expect("trapped ion compiler creation failed");
        assert_eq!(compiler.config.platform, HardwarePlatform::TrappedIon);
        assert!(compiler
            .config
            .native_gates
            .two_qubit_gates
            .contains(&NativeGateType::MS));
    }
    #[test]
    fn test_connectivity_check() {
        let topology = create_test_topology();
        let compiler =
            HardwareCompiler::for_superconducting(topology).expect("compiler creation failed");
        assert!(compiler
            .check_connectivity(QubitId::new(0), QubitId::new(1))
            .expect("connectivity check failed"));
        assert!(compiler
            .check_connectivity(QubitId::new(1), QubitId::new(2))
            .expect("connectivity check failed"));
        assert!(!compiler
            .check_connectivity(QubitId::new(0), QubitId::new(2))
            .expect("connectivity check failed"));
        assert!(!compiler
            .check_connectivity(QubitId::new(0), QubitId::new(3))
            .expect("connectivity check failed"));
    }
    #[test]
    fn test_shortest_path_finding() {
        let topology = create_test_topology();
        let compiler =
            HardwareCompiler::for_superconducting(topology).expect("compiler creation failed");
        let path = compiler
            .find_shortest_path(QubitId::new(0), QubitId::new(1))
            .expect("path finding failed");
        assert_eq!(path, vec![QubitId::new(0), QubitId::new(1)]);
        let path = compiler
            .find_shortest_path(QubitId::new(0), QubitId::new(3))
            .expect("path finding failed");
        assert_eq!(
            path,
            vec![
                QubitId::new(0),
                QubitId::new(1),
                QubitId::new(2),
                QubitId::new(3)
            ]
        );
    }
    #[test]
    fn test_virtual_z_optimization() {
        let optimizer = SuperconductingOptimizer::new();
        let gates = vec![
            CompiledGate {
                gate_type: NativeGateType::VirtualZ,
                qubits: vec![QubitId::new(0)],
                parameters: vec![0.5],
                fidelity: 1.0,
                duration: Duration::from_nanos(0),
                pulse_sequence: None,
            },
            CompiledGate {
                gate_type: NativeGateType::VirtualZ,
                qubits: vec![QubitId::new(0)],
                parameters: vec![0.3],
                fidelity: 1.0,
                duration: Duration::from_nanos(0),
                pulse_sequence: None,
            },
            CompiledGate {
                gate_type: NativeGateType::Rx,
                qubits: vec![QubitId::new(0)],
                parameters: vec![1.0],
                fidelity: 0.999,
                duration: Duration::from_nanos(20),
                pulse_sequence: None,
            },
        ];
        let optimized = optimizer
            .fuse_virtual_z_gates(&gates)
            .expect("virtual z gate fusion failed");
        assert_eq!(optimized.len(), 2);
        assert_eq!(optimized[0].gate_type, NativeGateType::VirtualZ);
        assert!((optimized[0].parameters[0] - 0.8).abs() < 1e-10);
    }
    #[test]
    fn test_gate_fidelity_calculation() {
        let topology = create_test_topology();
        let compiler =
            HardwareCompiler::for_superconducting(topology).expect("compiler creation failed");
        assert_eq!(compiler.get_gate_fidelity(NativeGateType::VirtualZ), 1.0);
        assert!(compiler.get_gate_fidelity(NativeGateType::Rx) > 0.999);
        assert!(
            compiler.get_gate_fidelity(NativeGateType::CNOT)
                < compiler.get_gate_fidelity(NativeGateType::Rx)
        );
    }
    #[test]
    fn test_platform_constraints() {
        let superconducting_optimizer = SuperconductingOptimizer::new();
        let constraints = superconducting_optimizer.get_constraints();
        assert!(constraints.max_qubits >= 100);
        assert!(constraints.timing_constraints.min_gate_separation < Duration::from_micros(1));
    }
    #[test]
    fn test_compilation_performance_tracking() {
        let topology = create_test_topology();
        let compiler =
            HardwareCompiler::for_superconducting(topology).expect("compiler creation failed");
        compiler.record_compilation_time(Duration::from_millis(10));
        compiler.record_compilation_time(Duration::from_millis(15));
        compiler.record_compilation_time(Duration::from_millis(12));
        let stats = compiler.get_performance_stats();
        assert_eq!(stats.total_compilations, 3);
        assert!(stats.average_compilation_time > Duration::from_millis(10));
        assert!(stats.average_compilation_time < Duration::from_millis(15));
    }
    #[test]
    fn test_cache_functionality() {
        let topology = create_test_topology();
        let compiler =
            HardwareCompiler::for_superconducting(topology).expect("compiler creation failed");
        let test_gates = vec![CompiledGate {
            gate_type: NativeGateType::Rx,
            qubits: vec![QubitId::new(0)],
            parameters: vec![1.0],
            fidelity: 0.999,
            duration: Duration::from_nanos(20),
            pulse_sequence: None,
        }];
        let cache_key = "test_gate_0";
        compiler
            .cache_result(cache_key, &test_gates)
            .expect("cache result failed");
        let cached_result = compiler.check_cache(cache_key).expect("check cache failed");
        assert!(cached_result.is_some());
        let cached_gates = cached_result.expect("cached result should be Some");
        assert_eq!(cached_gates.len(), 1);
        assert_eq!(cached_gates[0].gate_type, NativeGateType::Rx);
    }
    #[test]
    fn test_z_rotation_detection() {
        let topology = create_test_topology();
        let compiler =
            HardwareCompiler::for_superconducting(topology).expect("compiler creation failed");
        let angle = std::f64::consts::PI / 4.0;
        let mut z_matrix = Array2::zeros((2, 2));
        z_matrix[(0, 0)] = Complex64::from_polar(1.0, -angle / 2.0);
        z_matrix[(1, 1)] = Complex64::from_polar(1.0, angle / 2.0);
        let dense_z_matrix = DenseMatrix::new(z_matrix).expect("matrix creation failed");
        assert!(compiler
            .is_z_rotation(&dense_z_matrix)
            .expect("z rotation check failed"));
        let extracted_angle = compiler
            .extract_z_rotation_angle(&dense_z_matrix)
            .expect("angle extraction failed");
        assert!((extracted_angle - angle).abs() < 1e-10);
    }
    #[test]
    fn test_euler_angle_extraction() {
        let topology = create_test_topology();
        let compiler =
            HardwareCompiler::for_superconducting(topology).expect("compiler creation failed");
        let mut identity = Array2::zeros((2, 2));
        identity[(0, 0)] = Complex64::new(1.0, 0.0);
        identity[(1, 1)] = Complex64::new(1.0, 0.0);
        let dense_identity = DenseMatrix::new(identity).expect("matrix creation failed");
        let (theta, _phi, _lambda) = compiler
            .extract_euler_angles(&dense_identity)
            .expect("euler angle extraction failed");
        assert!(theta.abs() < 1e-10);
    }
    #[test]
    fn test_optimization_metrics_calculation() {
        let optimizer = SuperconductingOptimizer::new();
        let original_gates = vec![
            CompiledGate {
                gate_type: NativeGateType::VirtualZ,
                qubits: vec![QubitId::new(0)],
                parameters: vec![0.5],
                fidelity: 1.0,
                duration: Duration::from_nanos(0),
                pulse_sequence: None,
            },
            CompiledGate {
                gate_type: NativeGateType::VirtualZ,
                qubits: vec![QubitId::new(0)],
                parameters: vec![0.3],
                fidelity: 1.0,
                duration: Duration::from_nanos(0),
                pulse_sequence: None,
            },
        ];
        let optimized_gates = vec![CompiledGate {
            gate_type: NativeGateType::VirtualZ,
            qubits: vec![QubitId::new(0)],
            parameters: vec![0.8],
            fidelity: 1.0,
            duration: Duration::from_nanos(0),
            pulse_sequence: None,
        }];
        let metrics = optimizer.calculate_metrics(&original_gates, &optimized_gates, 1.0);
        assert_eq!(metrics.original_gate_count, 2);
        assert_eq!(metrics.optimized_gate_count, 1);
        assert_eq!(metrics.gate_count_reduction, 50.0);
    }
}
