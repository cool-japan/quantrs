//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use quantrs2_circuit::prelude::Circuit;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::collections::HashMap;

use super::types::{
    CircuitComplexity, ComputePrecision, ContractionPath, CuQuantumConfig, CuQuantumError,
    CuQuantumResult, CuQuantumSimulator, CuStateVecSimulator, CuTensorNetSimulator, CudaDeviceInfo,
    GateFusionLevel, GpuResourcePlanner, Observable, PerformanceEstimator, RecommendedBackend,
    SimulationStats, TensorNetworkState,
};

impl From<CuQuantumError> for SimulatorError {
    fn from(err: CuQuantumError) -> Self {
        SimulatorError::GpuError(err.to_string())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_config_default() {
        let config = CuQuantumConfig::default();
        assert_eq!(config.device_id, -1);
        assert!(!config.multi_gpu);
        assert_eq!(config.precision, ComputePrecision::Double);
    }
    #[test]
    fn test_config_large_circuit() {
        let config = CuQuantumConfig::large_circuit();
        assert!(config.memory_optimization);
        assert_eq!(config.gate_fusion_level, GateFusionLevel::Aggressive);
    }
    #[test]
    fn test_config_variational() {
        let config = CuQuantumConfig::variational();
        assert!(config.async_execution);
        assert_eq!(config.gate_fusion_level, GateFusionLevel::Moderate);
    }
    #[test]
    fn test_config_multi_gpu() {
        let config = CuQuantumConfig::multi_gpu(4);
        assert!(config.multi_gpu);
        assert_eq!(config.num_gpus, 4);
    }
    #[test]
    fn test_device_info_mock() {
        let info = CuStateVecSimulator::get_device_info(-1).expect("Should get mock device info");
        assert!(info.total_memory > 0);
        assert!(info.compute_capability.0 >= 7);
    }
    #[test]
    fn test_device_max_qubits() {
        let info = CudaDeviceInfo {
            device_id: 0,
            name: "Test A100".to_string(),
            total_memory: 40 * 1024 * 1024 * 1024,
            free_memory: 32 * 1024 * 1024 * 1024,
            compute_capability: (8, 0),
            sm_count: 108,
            max_threads_per_block: 1024,
            warp_size: 32,
            has_tensor_cores: true,
        };
        let max_qubits = info.max_statevec_qubits();
        assert!(
            max_qubits >= 30,
            "Expected >= 30 qubits, got {}",
            max_qubits
        );
        assert!(
            max_qubits <= 35,
            "Expected <= 35 qubits, got {}",
            max_qubits
        );
        let info_small = CudaDeviceInfo {
            device_id: 0,
            name: "Test RTX 3080".to_string(),
            total_memory: 12 * 1024 * 1024 * 1024,
            free_memory: 10 * 1024 * 1024 * 1024,
            compute_capability: (8, 6),
            sm_count: 68,
            max_threads_per_block: 1024,
            warp_size: 32,
            has_tensor_cores: true,
        };
        let max_qubits_small = info_small.max_statevec_qubits();
        assert!(
            max_qubits_small >= 28,
            "Expected >= 28 qubits for 12GB GPU, got {}",
            max_qubits_small
        );
    }
    #[test]
    fn test_custatevec_simulator_creation() {
        let config = CuQuantumConfig::default();
        let sim = CuStateVecSimulator::new(config);
        assert!(sim.is_ok());
    }
    #[test]
    fn test_cutensornet_simulator_creation() {
        let config = CuQuantumConfig::default();
        let sim = CuTensorNetSimulator::new(config);
        assert!(sim.is_ok());
    }
    #[test]
    fn test_simulation_stats() {
        let mut stats = SimulationStats::default();
        stats.total_simulations = 10;
        stats.total_gates = 100;
        stats.total_time_ms = 500.0;
        stats.total_flops = 1e9;
        assert_eq!(stats.avg_gates_per_sim(), 10.0);
        assert_eq!(stats.avg_time_per_sim(), 50.0);
        assert!((stats.throughput_gflops() - 2.0).abs() < 0.01);
    }
    #[test]
    fn test_contraction_path() {
        let mut path = ContractionPath::new();
        path.add_contraction(0, 1);
        path.add_contraction(0, 2);
        assert_eq!(path.contractions.len(), 2);
        assert!(path.total_cost() > 0.0);
    }
    #[test]
    fn test_unified_simulator_creation() {
        let config = CuQuantumConfig::default();
        let sim = CuQuantumSimulator::new(config);
        assert!(sim.is_ok());
    }
    #[test]
    fn test_observable_creation() {
        let obs = Observable::PauliZ(vec![0, 1]);
        match obs {
            Observable::PauliZ(qubits) => assert_eq!(qubits.len(), 2),
            _ => panic!("Wrong observable type"),
        }
    }
    #[test]
    fn test_cuquantum_result_from_state_vector() {
        use scirs2_core::ndarray::Array1;
        use scirs2_core::Complex64;
        let mut state = Array1::zeros(4);
        state[0] = Complex64::new(1.0, 0.0);
        let result = CuQuantumResult::from_state_vector(state.clone(), 2);
        assert_eq!(result.num_qubits, 2);
        assert!(result.state_vector.is_some());
        assert!(result.counts.is_empty());
        let probs = result.probabilities().expect("Should have probabilities");
        assert_eq!(probs.len(), 4);
        assert!((probs[0] - 1.0).abs() < 1e-10);
        assert!(probs[1] < 1e-10);
        assert!(probs[2] < 1e-10);
        assert!(probs[3] < 1e-10);
    }
    #[test]
    fn test_cuquantum_result_from_counts() {
        let mut counts = HashMap::new();
        counts.insert("00".to_string(), 500);
        counts.insert("11".to_string(), 500);
        let result = CuQuantumResult::from_counts(counts.clone(), 2);
        assert_eq!(result.num_qubits, 2);
        assert!(result.state_vector.is_none());
        assert_eq!(result.counts.len(), 2);
        assert_eq!(*result.counts.get("00").unwrap_or(&0), 500);
        assert_eq!(*result.counts.get("11").unwrap_or(&0), 500);
    }
    #[test]
    fn test_cuquantum_result_expectation_z() {
        use scirs2_core::ndarray::Array1;
        use scirs2_core::Complex64;
        let mut state_zero = Array1::zeros(2);
        state_zero[0] = Complex64::new(1.0, 0.0);
        let result_zero = CuQuantumResult::from_state_vector(state_zero, 1);
        let exp_z = result_zero
            .expectation_z(0)
            .expect("Should compute expectation");
        assert!(
            (exp_z - 1.0).abs() < 1e-10,
            "Expected +1 for |0⟩, got {}",
            exp_z
        );
        let mut state_one = Array1::zeros(2);
        state_one[1] = Complex64::new(1.0, 0.0);
        let result_one = CuQuantumResult::from_state_vector(state_one, 1);
        let exp_z_one = result_one
            .expectation_z(0)
            .expect("Should compute expectation");
        assert!(
            (exp_z_one - (-1.0)).abs() < 1e-10,
            "Expected -1 for |1⟩, got {}",
            exp_z_one
        );
        let mut state_plus = Array1::zeros(2);
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        state_plus[0] = Complex64::new(inv_sqrt2, 0.0);
        state_plus[1] = Complex64::new(inv_sqrt2, 0.0);
        let result_plus = CuQuantumResult::from_state_vector(state_plus, 1);
        let exp_z_plus = result_plus
            .expectation_z(0)
            .expect("Should compute expectation");
        assert!(
            exp_z_plus.abs() < 1e-10,
            "Expected 0 for |+⟩, got {}",
            exp_z_plus
        );
    }
    #[test]
    fn test_custatevec_circuit_simulation() {
        use quantrs2_circuit::prelude::Circuit;
        let config = CuQuantumConfig::default();
        let mut sim = CuStateVecSimulator::new(config).expect("Should create simulator");
        let circuit: Circuit<2> = Circuit::new();
        let result = sim.simulate(&circuit);
        // On non-macOS with cuquantum feature, simulation is not yet implemented
        #[cfg(all(feature = "cuquantum", not(target_os = "macos")))]
        {
            assert!(
                result.is_err(),
                "Expected error for unimplemented cuStateVec"
            );
        }
        #[cfg(any(target_os = "macos", not(feature = "cuquantum")))]
        {
            let result = result.expect("Should simulate circuit");
            assert_eq!(result.num_qubits, 2);
            assert!(result.state_vector.is_some());
            let sv = result
                .state_vector
                .as_ref()
                .expect("Should have state vector");
            assert_eq!(sv.len(), 4);
            assert!((sv[0].norm() - 1.0).abs() < 1e-10);
        }
    }
    #[test]
    fn test_custatevec_statistics() {
        use quantrs2_circuit::prelude::Circuit;
        let config = CuQuantumConfig::default();
        let mut sim = CuStateVecSimulator::new(config).expect("Should create simulator");
        let stats = sim.stats();
        assert_eq!(stats.total_simulations, 0);
        let circuit: Circuit<2> = Circuit::new();
        let result = sim.simulate(&circuit);
        let stats_after = sim.stats();
        // On non-macOS with cuquantum feature, simulation fails so stats won't increment
        #[cfg(all(feature = "cuquantum", not(target_os = "macos")))]
        {
            assert!(result.is_err());
            assert_eq!(stats_after.total_simulations, 0);
        }
        #[cfg(any(target_os = "macos", not(feature = "cuquantum")))]
        {
            let _ = result;
            assert_eq!(stats_after.total_simulations, 1);
        }
        sim.reset_stats();
        let stats_reset = sim.stats();
        assert_eq!(stats_reset.total_simulations, 0);
    }
    #[test]
    fn test_unified_simulator_auto_selection() {
        use quantrs2_circuit::prelude::Circuit;
        let config_small = CuQuantumConfig::default();
        let mut sim_small = CuQuantumSimulator::new(config_small).expect("Should create simulator");
        let circuit: Circuit<2> = Circuit::new();
        let result = sim_small.simulate(&circuit);
        // On non-macOS with cuquantum feature, simulation is not yet implemented
        #[cfg(all(feature = "cuquantum", not(target_os = "macos")))]
        {
            assert!(
                result.is_err(),
                "Expected error for unimplemented cuQuantum"
            );
        }
        #[cfg(any(target_os = "macos", not(feature = "cuquantum")))]
        {
            let result = result.expect("Should simulate");
            assert_eq!(result.num_qubits, 2);
            let mut config_large = CuQuantumConfig::default();
            config_large.max_statevec_qubits = 10;
            let mut sim_large =
                CuQuantumSimulator::new(config_large).expect("Should create simulator");
            let result_large = sim_large.simulate(&circuit).expect("Should simulate");
            assert_eq!(result_large.num_qubits, 2);
        }
    }
    #[test]
    fn test_cutensornet_network_building() {
        use quantrs2_circuit::prelude::Circuit;
        let config = CuQuantumConfig::default();
        let mut sim = CuTensorNetSimulator::new(config).expect("Should create simulator");
        let circuit: Circuit<4> = Circuit::new();
        sim.build_network(&circuit)
            .expect("Should build tensor network");
        assert!(sim.tensor_network.is_some());
    }
    #[test]
    fn test_cutensornet_contraction() {
        use quantrs2_circuit::prelude::Circuit;
        let config = CuQuantumConfig::default();
        let mut sim = CuTensorNetSimulator::new(config).expect("Should create simulator");
        let circuit: Circuit<3> = Circuit::new();
        sim.build_network(&circuit)
            .expect("Should build tensor network");
        let result = sim.contract(&[0, 1, 2]);
        // On non-macOS with cuquantum feature, contraction is not yet implemented
        #[cfg(all(feature = "cuquantum", not(target_os = "macos")))]
        {
            assert!(
                result.is_err(),
                "Expected error for unimplemented cuTensorNet"
            );
        }
        #[cfg(any(target_os = "macos", not(feature = "cuquantum")))]
        {
            let amplitudes = result.expect("Should contract network");
            assert_eq!(amplitudes.len(), 8);
        }
    }
    #[test]
    fn test_cutensornet_expectation_value() {
        use quantrs2_circuit::prelude::Circuit;
        let config = CuQuantumConfig::default();
        let mut sim = CuTensorNetSimulator::new(config).expect("Should create simulator");
        let circuit: Circuit<2> = Circuit::new();
        sim.build_network(&circuit)
            .expect("Should build tensor network");
        let observable = Observable::PauliZ(vec![0]);
        let result = sim.expectation_value(&observable);
        // On non-macOS with cuquantum feature, expectation value is not yet implemented
        #[cfg(all(feature = "cuquantum", not(target_os = "macos")))]
        {
            assert!(
                result.is_err(),
                "Expected error for unimplemented cuTensorNet"
            );
        }
        #[cfg(any(target_os = "macos", not(feature = "cuquantum")))]
        {
            let exp_val = result.expect("Should compute expectation");
            assert!((exp_val - 0.5).abs() < 1e-10);
        }
    }
    #[test]
    fn test_tensor_network_state_creation() {
        use quantrs2_circuit::prelude::Circuit;
        let circuit: Circuit<3> = Circuit::new();
        let network = TensorNetworkState::from_circuit(&circuit).expect("Should create network");
        assert!(
            network.num_tensors() >= 3,
            "Should have at least 3 tensors (one per qubit)"
        );
    }
    #[test]
    fn test_contraction_algorithms() {
        use quantrs2_circuit::prelude::Circuit;
        let config = CuQuantumConfig::default();
        let mut sim = CuTensorNetSimulator::new(config).expect("Should create simulator");
        let circuit: Circuit<4> = Circuit::new();
        sim.build_network(&circuit)
            .expect("Should build tensor network");
        let path = sim
            .find_contraction_order()
            .expect("Should find contraction order");
        assert!(!path.contractions.is_empty() || path.total_cost() >= 0.0);
    }
    #[test]
    fn test_device_info_methods() {
        let info = CudaDeviceInfo {
            device_id: 0,
            name: "Test Device".to_string(),
            total_memory: 80 * 1024 * 1024 * 1024,
            free_memory: 70 * 1024 * 1024 * 1024,
            compute_capability: (8, 0),
            sm_count: 108,
            max_threads_per_block: 1024,
            warp_size: 32,
            has_tensor_cores: true,
        };
        let max_qubits = info.max_statevec_qubits();
        assert!(
            max_qubits >= 31,
            "Expected >= 31 qubits for A100, got {}",
            max_qubits
        );
    }
    #[test]
    fn test_is_available() {
        let available = CuStateVecSimulator::is_available();
        #[cfg(not(feature = "cuquantum"))]
        assert!(!available);
    }
    #[test]
    fn test_observable_variants() {
        let obs_z = Observable::PauliZ(vec![0, 1]);
        let obs_x = Observable::PauliX(vec![0]);
        let obs_y = Observable::PauliY(vec![1]);
        let mut hermitian = Array2::zeros((2, 2));
        hermitian[[0, 0]] = Complex64::new(1.0, 0.0);
        hermitian[[1, 1]] = Complex64::new(-1.0, 0.0);
        let obs_h = Observable::Hermitian(hermitian);
        let obs_sum = Observable::Sum(vec![
            Observable::PauliZ(vec![0]),
            Observable::PauliZ(vec![1]),
        ]);
        let obs_prod = Observable::Product(vec![
            Observable::PauliX(vec![0]),
            Observable::PauliY(vec![1]),
        ]);
        assert!(matches!(obs_z, Observable::PauliZ(_)));
        assert!(matches!(obs_x, Observable::PauliX(_)));
        assert!(matches!(obs_y, Observable::PauliY(_)));
        assert!(matches!(obs_h, Observable::Hermitian(_)));
        assert!(matches!(obs_sum, Observable::Sum(_)));
        assert!(matches!(obs_prod, Observable::Product(_)));
    }
    #[test]
    fn test_performance_estimator_creation() {
        let config = CuQuantumConfig::default();
        let estimator = PerformanceEstimator::with_default_device(config);
        assert!(estimator.is_ok());
    }
    #[test]
    fn test_performance_estimate_small_circuit() {
        use quantrs2_circuit::prelude::Circuit;
        let config = CuQuantumConfig::default();
        let estimator =
            PerformanceEstimator::with_default_device(config).expect("Should create estimator");
        let circuit: Circuit<5> = Circuit::new();
        let estimate = estimator.estimate(&circuit);
        assert!(estimate.fits_in_memory);
        assert_eq!(
            estimate.recommended_backend,
            RecommendedBackend::StateVector
        );
        assert!(estimate.estimated_memory_bytes > 0);
        assert!(estimate.estimated_gpu_utilization >= 0.0);
        assert!(estimate.estimated_gpu_utilization <= 1.0);
    }
    #[test]
    fn test_performance_estimate_memory_calculation() {
        let config = CuQuantumConfig::default();
        let estimator =
            PerformanceEstimator::with_default_device(config).expect("Should create estimator");
        let device_info = estimator.device_info();
        let _ = device_info;
        use quantrs2_circuit::prelude::Circuit;
        let circuit_10: Circuit<10> = Circuit::new();
        let estimate_10 = estimator.estimate(&circuit_10);
        assert_eq!(estimate_10.estimated_memory_bytes, 1024 * 16);
        let circuit_20: Circuit<20> = Circuit::new();
        let estimate_20 = estimator.estimate(&circuit_20);
        assert_eq!(estimate_20.estimated_memory_bytes, 1024 * 1024 * 16);
    }
    #[test]
    fn test_performance_estimate_flops_calculation() {
        use quantrs2_circuit::prelude::Circuit;
        let config = CuQuantumConfig::default();
        let estimator =
            PerformanceEstimator::with_default_device(config).expect("Should create estimator");
        let circuit_empty: Circuit<5> = Circuit::new();
        let estimate_empty = estimator.estimate(&circuit_empty);
        assert_eq!(estimate_empty.estimated_flops, 0.0);
    }
    #[test]
    fn test_circuit_complexity_analysis() {
        use quantrs2_circuit::prelude::Circuit;
        let circuit: Circuit<4> = Circuit::new();
        let complexity = CircuitComplexity::analyze(&circuit);
        assert_eq!(complexity.num_qubits, 4);
        assert_eq!(complexity.num_gates, 0);
        assert_eq!(complexity.depth, 0);
        assert_eq!(complexity.entanglement_degree, 0.0);
    }
    #[test]
    fn test_gpu_resource_planner() {
        use quantrs2_circuit::prelude::Circuit;
        let device = CudaDeviceInfo {
            device_id: 0,
            name: "Test GPU".to_string(),
            total_memory: 16 * 1024 * 1024 * 1024,
            free_memory: 12 * 1024 * 1024 * 1024,
            compute_capability: (8, 6),
            sm_count: 68,
            max_threads_per_block: 1024,
            warp_size: 32,
            has_tensor_cores: true,
        };
        let config = CuQuantumConfig::default();
        let planner = GpuResourcePlanner::new(vec![device], config);
        let circuits: Vec<Circuit<4>> = vec![Circuit::new(), Circuit::new(), Circuit::new()];
        let assignments = planner.plan_batch(&circuits);
        assert_eq!(assignments.len(), 3);
        for (device_id, _) in &assignments {
            assert_eq!(*device_id, 0);
        }
        let batch_memory = planner.estimate_batch_memory(&circuits);
        assert_eq!(batch_memory, 3 * 16 * 16);
    }
    #[test]
    fn test_recommended_backend_enum() {
        assert_ne!(
            RecommendedBackend::StateVector,
            RecommendedBackend::TensorNetwork
        );
        assert_ne!(RecommendedBackend::Hybrid, RecommendedBackend::NotFeasible);
        let sv = format!("{:?}", RecommendedBackend::StateVector);
        assert!(sv.contains("StateVector"));
    }
    #[test]
    fn test_performance_suggestions() {
        use quantrs2_circuit::prelude::Circuit;
        let mut config = CuQuantumConfig::default();
        config.gate_fusion_level = GateFusionLevel::None;
        let estimator =
            PerformanceEstimator::with_default_device(config).expect("Should create estimator");
        let circuit: Circuit<26> = Circuit::new();
        let estimate = estimator.estimate(&circuit);
        let has_fusion_suggestion = estimate
            .suggestions
            .iter()
            .any(|s| s.contains("gate fusion"));
        assert!(
            has_fusion_suggestion,
            "Should suggest gate fusion for 26 qubit circuit"
        );
    }
    #[test]
    fn test_multi_gpu_planner() {
        use quantrs2_circuit::prelude::Circuit;
        let devices = vec![
            CudaDeviceInfo {
                device_id: 0,
                name: "GPU 0".to_string(),
                total_memory: 16 * 1024 * 1024 * 1024,
                free_memory: 12 * 1024 * 1024 * 1024,
                compute_capability: (8, 6),
                sm_count: 68,
                max_threads_per_block: 1024,
                warp_size: 32,
                has_tensor_cores: true,
            },
            CudaDeviceInfo {
                device_id: 1,
                name: "GPU 1".to_string(),
                total_memory: 16 * 1024 * 1024 * 1024,
                free_memory: 12 * 1024 * 1024 * 1024,
                compute_capability: (8, 6),
                sm_count: 68,
                max_threads_per_block: 1024,
                warp_size: 32,
                has_tensor_cores: true,
            },
        ];
        let config = CuQuantumConfig::default();
        let planner = GpuResourcePlanner::new(devices, config);
        let circuits: Vec<Circuit<4>> = vec![
            Circuit::new(),
            Circuit::new(),
            Circuit::new(),
            Circuit::new(),
        ];
        let assignments = planner.plan_batch(&circuits);
        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0].0, 0);
        assert_eq!(assignments[1].0, 1);
        assert_eq!(assignments[2].0, 0);
        assert_eq!(assignments[3].0, 1);
    }
}
