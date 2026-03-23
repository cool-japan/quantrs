//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::parallel_ops::*;
use scirs2_core::Complex64;

use super::types::{
    DistributedGpuConfig, DistributedGpuStateVector, DistributedGpuUtils, PartitionScheme,
    SyncStrategy,
};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn test_distributed_gpu_config_default() {
        let config = DistributedGpuConfig::default();
        assert_eq!(config.num_gpus, 0);
        assert_eq!(config.min_qubits_for_gpu, 15);
        assert_eq!(config.sync_strategy, SyncStrategy::AllReduce);
    }
    #[test]
    fn test_partition_scheme_selection() {
        let config = DistributedGpuConfig::default();
        let scheme = DistributedGpuStateVector::select_partition_scheme(20, 2, &config);
        assert_eq!(scheme, PartitionScheme::Block);
        let scheme = DistributedGpuStateVector::select_partition_scheme(30, 2, &config);
        assert_eq!(scheme, PartitionScheme::Adaptive);
    }
    #[test]
    fn test_memory_estimation() {
        let memory_1gpu = DistributedGpuUtils::estimate_memory_requirements(20, 1);
        let memory_4gpu = DistributedGpuUtils::estimate_memory_requirements(20, 4);
        assert!(memory_4gpu < memory_1gpu / 2);
    }
    #[test]
    fn test_optimal_gpu_count() {
        let memory_per_gpu = 8_000_000_000;
        let optimal = DistributedGpuUtils::optimal_gpu_count(25, 8, memory_per_gpu);
        assert!(optimal >= 1);
        assert!(optimal <= 8);
    }
    #[test]
    fn test_distributed_simulation_small() {
        if !DistributedGpuStateVector::is_gpu_available() {
            eprintln!("Skipping GPU test: GPU backend not available");
            return;
        }
        let config = DistributedGpuConfig {
            min_qubits_for_gpu: 5,
            num_gpus: 1,
            ..Default::default()
        };
        let mut simulator =
            DistributedGpuStateVector::new(5, config).expect("failed to create simulator");
        simulator
            .initialize_zero_state()
            .expect("failed to initialize state");
        let pauli_x = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("valid 2x2 matrix shape");
        simulator
            .apply_single_qubit_gate(0, &pauli_x)
            .expect("failed to apply gate");
        let prob = simulator
            .measure_probability(0)
            .expect("failed to measure probability");
        assert_abs_diff_eq!(prob, 1.0, epsilon = 1e-10);
    }
    #[test]
    fn test_synchronization_strategies() {
        if !DistributedGpuStateVector::is_gpu_available() {
            eprintln!("Skipping GPU test: GPU backend not available");
            return;
        }
        let strategies = vec![
            SyncStrategy::AllReduce,
            SyncStrategy::RingReduce,
            SyncStrategy::TreeReduce,
            SyncStrategy::PointToPoint,
        ];
        for strategy in strategies {
            let config = DistributedGpuConfig {
                min_qubits_for_gpu: 5,
                num_gpus: 2,
                sync_strategy: strategy,
                ..Default::default()
            };
            let mut simulator =
                DistributedGpuStateVector::new(5, config).expect("failed to create simulator");
            simulator
                .initialize_zero_state()
                .expect("failed to initialize state");
            let cnot = Array2::from_shape_vec(
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
            .expect("valid 4x4 matrix shape");
            let result = simulator.apply_two_qubit_gate(0, 1, &cnot);
            assert!(
                result.is_ok(),
                "Failed to apply two-qubit gate with {:?}",
                strategy
            );
            let stats = simulator.get_stats();
            assert!(stats.communication_time_ms >= 0.0);
        }
    }
    #[test]
    fn test_partition_schemes() {
        let schemes = vec![
            PartitionScheme::Block,
            PartitionScheme::Interleaved,
            PartitionScheme::Adaptive,
        ];
        for scheme in schemes {
            let config = DistributedGpuConfig {
                min_qubits_for_gpu: 5,
                num_gpus: 2,
                auto_load_balance: false,
                ..Default::default()
            };
            let selected = DistributedGpuStateVector::select_partition_scheme(5, 2, &config);
            assert_eq!(selected, PartitionScheme::Block);
            let config_auto = DistributedGpuConfig {
                auto_load_balance: true,
                ..config
            };
            let selected_auto =
                DistributedGpuStateVector::select_partition_scheme(10, 2, &config_auto);
            assert_ne!(selected_auto, PartitionScheme::HilbertCurve);
        }
    }
    #[test]
    fn test_gpu_utils_memory_estimation() {
        let memory_1gpu = DistributedGpuUtils::estimate_memory_requirements(20, 1);
        let memory_4gpu = DistributedGpuUtils::estimate_memory_requirements(20, 4);
        assert!(memory_4gpu <= memory_1gpu);
        assert!(memory_1gpu > 0);
        assert!(memory_4gpu > 0);
        let memory_per_gpu = 8_000_000_000;
        let optimal = DistributedGpuUtils::optimal_gpu_count(25, 8, memory_per_gpu);
        assert!(optimal >= 1 && optimal <= 8);
    }
    #[test]
    fn test_benchmark_partitioning_strategies() {
        let result = DistributedGpuUtils::benchmark_partitioning_strategies(16, 2);
        assert!(result.is_ok());
        let benchmarks = result.expect("benchmark should succeed");
        assert!(benchmarks.contains_key("Block"));
        assert!(benchmarks.contains_key("Interleaved"));
        assert!(benchmarks.contains_key("Adaptive"));
        for (strategy, time) in benchmarks {
            assert!(time >= 0.0, "Negative time for strategy {}", strategy);
        }
    }
    #[test]
    #[ignore = "Skipping distributed GPU test"]
    fn test_state_vector_retrieval() {
        if !DistributedGpuStateVector::is_gpu_available() {
            eprintln!("Skipping GPU test: GPU backend not available");
            return;
        }
        let config = DistributedGpuConfig {
            min_qubits_for_gpu: 5,
            num_gpus: 2,
            ..Default::default()
        };
        let mut simulator =
            DistributedGpuStateVector::new(5, config).expect("failed to create simulator");
        simulator
            .initialize_zero_state()
            .expect("failed to initialize state");
        let state = simulator
            .get_state_vector()
            .expect("failed to get state vector");
        assert_eq!(state.len(), 32);
        assert_abs_diff_eq!(state[0].norm_sqr(), 1.0, epsilon = 1e-10);
        for i in 1..state.len() {
            assert_abs_diff_eq!(state[i].norm_sqr(), 0.0, epsilon = 1e-10);
        }
    }
    #[test]
    fn test_multi_gpu_gate_application() {
        if !DistributedGpuStateVector::is_gpu_available() {
            eprintln!("Skipping GPU test: GPU backend not available");
            return;
        }
        let config = DistributedGpuConfig {
            min_qubits_for_gpu: 5,
            num_gpus: 2,
            ..Default::default()
        };
        let mut simulator =
            DistributedGpuStateVector::new(6, config).expect("failed to create simulator");
        simulator
            .initialize_zero_state()
            .expect("failed to initialize state");
        let hadamard = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
            ],
        )
        .expect("valid 2x2 matrix shape");
        let result = simulator.apply_single_qubit_gate(0, &hadamard);
        assert!(result.is_ok());
        let prob_0 = 1.0
            - simulator
                .measure_probability(0)
                .expect("failed to measure probability");
        let prob_1 = simulator
            .measure_probability(0)
            .expect("failed to measure probability");
        assert_abs_diff_eq!(prob_0, 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(prob_1, 0.5, epsilon = 1e-6);
    }
    #[test]
    fn test_partition_synchronization() {
        if !DistributedGpuStateVector::is_gpu_available() {
            eprintln!("Skipping GPU test: GPU backend not available");
            return;
        }
        let config = DistributedGpuConfig {
            min_qubits_for_gpu: 5,
            num_gpus: 3,
            ..Default::default()
        };
        let mut simulator =
            DistributedGpuStateVector::new(6, config).expect("failed to create simulator");
        assert!(simulator.partitions_require_sync(0, 1));
        let result = simulator.exchange_boundary_states(0, 1);
        assert!(result.is_ok());
        let sync_result = simulator.synchronize_all_reduce();
        assert!(sync_result.is_ok());
        let stats = simulator.get_stats();
        assert!(stats.communication_time_ms >= 0.0);
    }
    #[test]
    fn test_inter_gpu_communication_detection() {
        if !DistributedGpuStateVector::is_gpu_available() {
            eprintln!("Skipping GPU test: GPU backend not available");
            return;
        }
        let config = DistributedGpuConfig {
            min_qubits_for_gpu: 5,
            num_gpus: 2,
            ..Default::default()
        };
        let simulator =
            DistributedGpuStateVector::new(6, config).expect("failed to create simulator");
        let requires_comm = simulator.requires_inter_gpu_communication(0, 1);
        let same_qubit_comm = simulator.requires_inter_gpu_communication(0, 0);
        assert!(!same_qubit_comm);
    }
    #[test]
    fn test_performance_statistics() {
        if !DistributedGpuStateVector::is_gpu_available() {
            eprintln!("Skipping GPU test: GPU backend not available");
            return;
        }
        let config = DistributedGpuConfig {
            min_qubits_for_gpu: 5,
            num_gpus: 2,
            ..Default::default()
        };
        let mut simulator =
            DistributedGpuStateVector::new(6, config).expect("failed to create simulator");
        simulator
            .initialize_zero_state()
            .expect("failed to initialize state");
        let identity = Array2::eye(2);
        for i in 0..3 {
            simulator
                .apply_single_qubit_gate(i, &identity)
                .expect("failed to apply gate");
        }
        let stats = simulator.get_stats();
        assert!(stats.total_execution_time_ms >= 0.0);
        assert!(stats.memory_transfer_time_ms >= 0.0);
        assert_eq!(stats.gpu_computation_time_ms.len(), 2);
        assert_eq!(stats.gpu_utilization.len(), 2);
        assert_eq!(stats.memory_usage_bytes.len(), 2);
    }
}
