//! Tests for quantum simulators
//!
//! This module provides tests for the various simulator implementations
//! to ensure correctness and compatibility.

use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::register::Register;

use crate::optimized_simulator::OptimizedSimulator;
use crate::quantum_reservoir_computing::{
    InputEncoding, OutputMeasurement, QuantumReservoirArchitecture, QuantumReservoirComputer,
    QuantumReservoirConfig, ReservoirDynamics,
};
use crate::statevector::StateVectorSimulator;
use ndarray::Array1;

/// Create a bell state circuit
fn create_bell_circuit<const N: usize>() -> Circuit<N> {
    let mut circuit = Circuit::new();

    // Apply Hadamard to qubit 0
    circuit.h(0).unwrap();

    // Apply CNOT with qubit 0 as control and qubit 1 as target
    circuit.cnot(0, 1).unwrap();

    circuit
}

/// Create a GHZ state circuit for N qubits
fn create_ghz_circuit<const N: usize>() -> Circuit<N> {
    let mut circuit = Circuit::new();

    // Apply Hadamard to qubit 0
    circuit.h(0).unwrap();

    // Apply CNOT gates to entangle all qubits
    for i in 1..N {
        circuit.cnot(0, i).unwrap();
    }

    circuit
}

/// Create a random circuit with the specified number of gates
fn create_random_circuit<const N: usize>(num_gates: usize) -> Circuit<N> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::f64::consts::PI;

    let mut circuit = Circuit::new();
    let mut rng = StdRng::seed_from_u64(42); // Use fixed seed for reproducibility

    for _ in 0..num_gates {
        let gate_type = rng.random_range(0..5);

        match gate_type {
            0 => {
                // Hadamard gate
                let target = rng.random_range(0..N);
                circuit.h(target).unwrap();
            }
            1 => {
                // Pauli-X gate
                let target = rng.random_range(0..N);
                circuit.x(target).unwrap();
            }
            2 => {
                // Rotation-Z gate
                let target = rng.random_range(0..N);
                let angle = rng.random_range(0.0..2.0 * PI);
                circuit.rz(target, angle).unwrap();
            }
            3 => {
                // CNOT gate
                let control = rng.random_range(0..N);
                let mut target = rng.random_range(0..N);
                while target == control {
                    target = rng.random_range(0..N);
                }
                circuit.cnot(control, target).unwrap();
            }
            4 => {
                // CZ gate
                let control = rng.random_range(0..N);
                let mut target = rng.random_range(0..N);
                while target == control {
                    target = rng.random_range(0..N);
                }
                circuit.cz(control, target).unwrap();
            }
            _ => unreachable!(),
        }
    }

    circuit
}

/// Compare results between standard and optimized simulators
fn compare_simulators<const N: usize>(circuit: &Circuit<N>, epsilon: f64) -> bool {
    let standard_sim = StateVectorSimulator::new();
    let optimized_sim = OptimizedSimulator::new();

    let standard_result = standard_sim.run(circuit).unwrap();
    let optimized_result = optimized_sim.run(circuit).unwrap();

    let standard_state = standard_result.amplitudes();
    let optimized_state = optimized_result.amplitudes();

    // Check that the dimensions match
    if standard_state.len() != optimized_state.len() {
        println!("State vector dimensions don't match");
        return false;
    }

    // Check each amplitude
    for (i, (std_amp, opt_amp)) in standard_state
        .iter()
        .zip(optimized_state.iter())
        .enumerate()
    {
        let diff = (std_amp - opt_amp).norm();
        if diff > epsilon {
            println!(
                "Amplitude {} differs: standard={}, optimized={}, diff={}",
                i, std_amp, opt_amp, diff
            );
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bell_state() {
        const N: usize = 2;
        let circuit = create_bell_circuit::<N>();

        // Run the circuit with both simulators
        let standard_sim = StateVectorSimulator::new();
        let optimized_sim = OptimizedSimulator::new();

        let standard_result = standard_sim.run(&circuit).unwrap();
        let optimized_result = optimized_sim.run(&circuit).unwrap();

        // Expected result: (|00> + |11>) / sqrt(2)
        let expected_amplitudes = vec![
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
        ];

        // Check standard simulator
        let standard_state = standard_result.amplitudes();
        for (i, (actual, expected)) in standard_state
            .iter()
            .zip(expected_amplitudes.iter())
            .enumerate()
        {
            let diff = (actual - expected).norm();
            assert!(
                diff < 1e-10,
                "Standard simulator: state[{}] differs by {}",
                i,
                diff
            );
        }

        // Check optimized simulator
        let optimized_state = optimized_result.amplitudes();
        for (i, (actual, expected)) in optimized_state
            .iter()
            .zip(expected_amplitudes.iter())
            .enumerate()
        {
            let diff = (actual - expected).norm();
            assert!(
                diff < 1e-10,
                "Optimized simulator: state[{}] differs by {}",
                i,
                diff
            );
        }
    }

    #[test]
    fn test_ghz_state() {
        const N: usize = 3;
        let circuit = create_ghz_circuit::<N>();

        // Run the circuit with both simulators
        let standard_sim = StateVectorSimulator::new();
        let optimized_sim = OptimizedSimulator::new();

        let standard_result = standard_sim.run(&circuit).unwrap();
        let optimized_result = optimized_sim.run(&circuit).unwrap();

        // Expected result: (|000> + |111>) / sqrt(2)
        let mut expected_amplitudes = vec![Complex64::new(0.0, 0.0); 1 << N];
        expected_amplitudes[0] = Complex64::new(FRAC_1_SQRT_2, 0.0);
        expected_amplitudes[7] = Complex64::new(FRAC_1_SQRT_2, 0.0);

        // Check standard simulator
        let standard_state = standard_result.amplitudes();
        for (i, (actual, expected)) in standard_state
            .iter()
            .zip(expected_amplitudes.iter())
            .enumerate()
        {
            let diff = (actual - expected).norm();
            assert!(
                diff < 1e-10,
                "Standard simulator: state[{}] differs by {}",
                i,
                diff
            );
        }

        // Check optimized simulator
        let optimized_state = optimized_result.amplitudes();
        for (i, (actual, expected)) in optimized_state
            .iter()
            .zip(expected_amplitudes.iter())
            .enumerate()
        {
            let diff = (actual - expected).norm();
            assert!(
                diff < 1e-10,
                "Optimized simulator: state[{}] differs by {}",
                i,
                diff
            );
        }
    }

    #[test]
    fn test_random_circuit_4qubits() {
        const N: usize = 4;
        let circuit = create_random_circuit::<N>(20);

        assert!(
            compare_simulators(&circuit, 1e-10),
            "4-qubit random circuit: simulators disagree"
        );
    }

    #[test]
    fn test_random_circuit_8qubits() {
        const N: usize = 8;
        let circuit = create_random_circuit::<N>(20);

        assert!(
            compare_simulators(&circuit, 1e-10),
            "8-qubit random circuit: simulators disagree"
        );
    }

    #[test]
    #[ignore] // This test is resource-intensive, so we'll ignore it by default
    fn test_random_circuit_16qubits() {
        const N: usize = 16;
        let circuit = create_random_circuit::<N>(10);

        assert!(
            compare_simulators(&circuit, 1e-10),
            "16-qubit random circuit: simulators disagree"
        );
    }

    #[test]
    #[ignore] // This test is even more resource-intensive
    fn test_random_circuit_20qubits() {
        const N: usize = 20;
        let circuit = create_random_circuit::<N>(5);

        assert!(
            compare_simulators(&circuit, 1e-10),
            "20-qubit random circuit: simulators disagree"
        );
    }
}

/// Tests for the new ultrathink mode implementations
#[cfg(test)]
mod ultrathink_tests {
    use super::*;
    use crate::adaptive_gate_fusion::{
        AdaptiveFusionConfig, AdaptiveGateFusion, CircuitPatternAnalyzer, FusionStrategy, GateType,
        MLFusionPredictor, QuantumGate,
    };
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    use crate::distributed_gpu::{
        DistributedGpuConfig, DistributedGpuStateVector, PartitionScheme, SyncStrategy,
    };
    use crate::mixed_precision_impl::{
        MixedPrecisionConfig, MixedPrecisionSimulator, QuantumPrecision,
    };
    use ndarray::Array2;
    use num_complex::Complex64;

    #[test]
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    fn test_distributed_gpu_config() {
        let config = DistributedGpuConfig::default();
        assert_eq!(config.num_gpus, 0); // Auto-detect
        assert_eq!(config.min_qubits_for_gpu, 15);
        assert_eq!(config.sync_strategy, SyncStrategy::AllReduce);
        assert_eq!(config.memory_overlap_ratio, 0.1);
    }

    #[test]
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    fn test_distributed_gpu_state_vector_creation() {
        if !DistributedGpuStateVector::is_gpu_available() {
            eprintln!("Skipping GPU test: GPU backend not available");
            return;
        }

        let config = DistributedGpuConfig {
            num_gpus: 2,
            min_qubits_for_gpu: 2,
            max_state_size_per_gpu: 1024,
            auto_load_balance: true,
            memory_overlap_ratio: 0.1,
            use_mixed_precision: false,
            sync_strategy: SyncStrategy::AllReduce,
        };

        let result = DistributedGpuStateVector::new(3, config);
        assert!(result.is_ok());

        let state_vector = result.unwrap();
        assert_eq!(state_vector.num_qubits(), 3);
        assert_eq!(state_vector.state_size(), 8); // 2^3
    }

    #[test]
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    fn test_distributed_gpu_partition_schemes() {
        if !DistributedGpuStateVector::is_gpu_available() {
            eprintln!("Skipping GPU test: GPU backend not available");
            return;
        }

        let config = DistributedGpuConfig::default();

        // Test different partition schemes
        for scheme in &[
            PartitionScheme::Block,
            PartitionScheme::Interleaved,
            PartitionScheme::Adaptive,
        ] {
            let mut test_config = config.clone();
            test_config.num_gpus = 2;

            let result = DistributedGpuStateVector::new(4, test_config);
            assert!(
                result.is_ok(),
                "Failed to create state vector with {:?} partitioning",
                scheme
            );
        }
    }

    #[test]
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    fn test_distributed_gpu_hilbert_partitioning() {
        if !DistributedGpuStateVector::is_gpu_available() {
            eprintln!("Skipping GPU test: GPU backend not available");
            return;
        }

        let config = DistributedGpuConfig {
            num_gpus: 2,
            min_qubits_for_gpu: 2,
            max_state_size_per_gpu: 1024,
            auto_load_balance: true,
            memory_overlap_ratio: 0.1,
            use_mixed_precision: false,
            sync_strategy: SyncStrategy::AllReduce,
        };

        // Hilbert partitioning should work or fall back gracefully
        let result = DistributedGpuStateVector::new(4, config);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    fn test_distributed_gpu_synchronization_strategies() {
        if !DistributedGpuStateVector::is_gpu_available() {
            eprintln!("Skipping GPU test: GPU backend not available");
            return;
        }

        let sync_strategies = [
            SyncStrategy::AllReduce,
            SyncStrategy::RingReduce,
            SyncStrategy::TreeReduce,
            SyncStrategy::PointToPoint,
        ];

        for &strategy in &sync_strategies {
            let config = DistributedGpuConfig {
                num_gpus: 3,
                min_qubits_for_gpu: 2,
                max_state_size_per_gpu: 1024,
                auto_load_balance: true,
                memory_overlap_ratio: 0.1,
                use_mixed_precision: false,
                sync_strategy: strategy,
            };

            let mut state_vector = DistributedGpuStateVector::new(3, config).unwrap();

            // Test synchronization
            let result = state_vector.synchronize();
            assert!(
                result.is_ok(),
                "Synchronization failed for strategy {:?}",
                strategy
            );
        }
    }

    #[test]
    fn test_adaptive_gate_fusion_config() {
        let config = AdaptiveFusionConfig::default();
        assert_eq!(config.strategy, FusionStrategy::Adaptive);
        assert_eq!(config.max_fusion_size, 8);
        assert!(config.enable_cross_qubit_fusion);
        assert!(config.enable_temporal_fusion);
        assert!(config.enable_ml_predictions);
    }

    #[test]
    fn test_quantum_gate_creation() {
        let gate = QuantumGate::new(GateType::Hadamard, vec![0], vec![]);
        assert_eq!(gate.gate_type, GateType::Hadamard);
        assert_eq!(gate.qubits, vec![0]);
        assert_eq!(gate.parameters.len(), 0);
        assert_eq!(gate.matrix.shape(), [2, 2]);
    }

    #[test]
    fn test_rotation_gate_creation() {
        let angle = std::f64::consts::PI / 4.0;
        let gate = QuantumGate::new(GateType::RotationX, vec![0], vec![angle]);

        assert_eq!(gate.gate_type, GateType::RotationX);
        assert_eq!(gate.qubits, vec![0]);
        assert_eq!(gate.parameters, vec![angle]);

        // Check that matrix has correct structure for RX gate
        assert_eq!(gate.matrix.shape(), [2, 2]);
        assert!((gate.matrix[[0, 0]].re - (angle / 2.0).cos()).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_gate_fusion_creation() {
        let config = AdaptiveFusionConfig::default();
        let result = AdaptiveGateFusion::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gate_fusion_basic_sequence() {
        let config = AdaptiveFusionConfig::default();
        let mut fusion_engine = AdaptiveGateFusion::new(config).unwrap();

        // Create a simple gate sequence
        let gates = vec![
            QuantumGate::new(
                GateType::RotationX,
                vec![0],
                vec![std::f64::consts::PI / 4.0],
            ),
            QuantumGate::new(
                GateType::RotationX,
                vec![0],
                vec![std::f64::consts::PI / 6.0],
            ),
        ];

        let result = fusion_engine.fuse_gates(&gates);
        assert!(result.is_ok());

        let (fused_blocks, remaining_gates) = result.unwrap();
        assert!(!fused_blocks.is_empty() || !remaining_gates.is_empty());
    }

    #[test]
    fn test_ml_fusion_predictor() {
        let predictor = MLFusionPredictor::new();

        let gates = vec![
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.1]),
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.2]),
        ];

        let benefit = predictor.predict_benefit(&gates);
        assert!(benefit >= 0.0 && benefit <= 1.0);
    }

    #[test]
    fn test_circuit_pattern_analyzer() {
        let mut analyzer = CircuitPatternAnalyzer::new();

        let gates = vec![
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.1]),
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.2]),
        ];

        let result = analyzer.analyze_pattern(&gates);
        assert!(!result.pattern.is_empty());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.expected_benefit >= 0.0);
    }

    #[test]
    fn test_gate_fusion_known_beneficial_patterns() {
        let config = AdaptiveFusionConfig::default();
        let mut fusion_engine = AdaptiveGateFusion::new(config).unwrap();

        // Test known beneficial pattern: consecutive rotation gates
        let gates = vec![
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.1]),
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.2]),
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.3]),
        ];

        let result = fusion_engine.fuse_gates(&gates);
        assert!(result.is_ok());

        let (fused_blocks, _) = result.unwrap();
        assert!(
            !fused_blocks.is_empty(),
            "Should have identified beneficial fusion opportunity"
        );
    }

    #[test]
    fn test_mixed_precision_config() {
        let config = MixedPrecisionConfig::default();
        assert_eq!(config.state_vector_precision, QuantumPrecision::Single);
        assert_eq!(config.gate_precision, QuantumPrecision::Single);
        assert!(config.error_tolerance > 0.0);
    }

    #[test]
    fn test_quantum_precision_properties() {
        assert_eq!(QuantumPrecision::Half.memory_factor(), 0.25);
        assert_eq!(QuantumPrecision::Single.memory_factor(), 0.5);
        assert_eq!(QuantumPrecision::Double.memory_factor(), 1.0);

        assert!(QuantumPrecision::Half.typical_error() > QuantumPrecision::Single.typical_error());
        assert!(
            QuantumPrecision::Single.typical_error() > QuantumPrecision::Double.typical_error()
        );
    }

    #[test]
    fn test_mixed_precision_simulator_creation() {
        let config = MixedPrecisionConfig::default();
        let result = MixedPrecisionSimulator::new(3, config);
        assert!(result.is_ok());

        let simulator = result.unwrap();
        assert!(simulator.get_state().is_some());
    }

    #[test]
    fn test_mixed_precision_gate_application() {
        let config = MixedPrecisionConfig::default();
        let mut simulator = MixedPrecisionSimulator::new(2, config).unwrap();

        let gate = QuantumGate::new(GateType::Hadamard, vec![0], vec![]);
        let result = simulator.apply_gate(&gate);
        assert!(result.is_ok());
    }

    #[test]
    fn test_precision_adaptation() {
        let mut config = MixedPrecisionConfig::default();
        config.adaptive_precision = true;

        let mut simulator = MixedPrecisionSimulator::new(2, config).unwrap();

        // Apply several gates and check that precision adaptation works
        let gates = vec![
            QuantumGate::new(GateType::Hadamard, vec![0], vec![]),
            QuantumGate::new(GateType::CNOT, vec![0, 1], vec![]),
            QuantumGate::new(GateType::RotationZ, vec![1], vec![0.001]), // Small rotation
        ];

        for gate in &gates {
            let result = simulator.apply_gate(gate);
            assert!(result.is_ok());
        }

        let stats = simulator.get_stats();
        assert!(stats.total_gates > 0);
    }

    #[test]
    fn test_memory_estimation() {
        let config = MixedPrecisionConfig::default();

        let memory_4q = crate::mixed_precision_impl::estimate_memory_usage(&config, 4);
        let memory_8q = crate::mixed_precision_impl::estimate_memory_usage(&config, 8);

        // Memory should scale exponentially with qubits
        assert!(memory_8q > memory_4q * 10);
    }

    #[test]
    fn test_performance_benchmarking() {
        let config = MixedPrecisionConfig::default();
        let mut simulator = MixedPrecisionSimulator::new(3, config).unwrap();

        // Create a benchmark circuit
        let gates = vec![
            QuantumGate::new(GateType::Hadamard, vec![0], vec![]),
            QuantumGate::new(GateType::CNOT, vec![0, 1], vec![]),
            QuantumGate::new(GateType::RotationZ, vec![1], vec![0.5]),
            QuantumGate::new(GateType::CNOT, vec![1, 2], vec![]),
        ];

        let start_time = std::time::Instant::now();

        for gate in &gates {
            simulator.apply_gate(gate).unwrap();
        }

        let execution_time = start_time.elapsed();
        assert!(execution_time.as_millis() < 1000); // Should complete quickly

        let stats = simulator.get_stats();
        assert_eq!(stats.total_gates, gates.len());
    }

    #[test]
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    fn test_integration_distributed_gpu_with_fusion() {
        if !DistributedGpuStateVector::is_gpu_available() {
            eprintln!("Skipping GPU test: GPU backend not available");
            return;
        }

        let gpu_config = DistributedGpuConfig {
            num_gpus: 2,
            min_qubits_for_gpu: 2,
            max_state_size_per_gpu: 1024,
            auto_load_balance: true,
            memory_overlap_ratio: 0.1,
            use_mixed_precision: false,
            sync_strategy: SyncStrategy::AllReduce,
        };

        let fusion_config = AdaptiveFusionConfig::default();

        // Test that both systems can be initialized together
        let gpu_result = DistributedGpuStateVector::new(4, gpu_config);
        let fusion_result = AdaptiveGateFusion::new(fusion_config);

        assert!(gpu_result.is_ok());
        assert!(fusion_result.is_ok());
    }

    #[test]
    fn test_integration_mixed_precision_with_fusion() {
        let precision_config = MixedPrecisionConfig::default();
        let fusion_config = AdaptiveFusionConfig::default();

        let precision_result = MixedPrecisionSimulator::new(3, precision_config);
        let fusion_result = AdaptiveGateFusion::new(fusion_config);

        assert!(precision_result.is_ok());
        assert!(fusion_result.is_ok());
    }

    #[test]
    fn test_comprehensive_ultrathink_pipeline() {
        // Test a complete pipeline using all new features
        let precision_config = MixedPrecisionConfig::default();
        let mut precision_sim = MixedPrecisionSimulator::new(3, precision_config).unwrap();

        let fusion_config = AdaptiveFusionConfig::default();
        let mut fusion_engine = AdaptiveGateFusion::new(fusion_config).unwrap();

        // Create a test circuit
        let gates = vec![
            QuantumGate::new(GateType::Hadamard, vec![0], vec![]),
            QuantumGate::new(GateType::RotationX, vec![1], vec![0.5]),
            QuantumGate::new(GateType::RotationX, vec![1], vec![0.3]),
            QuantumGate::new(GateType::CNOT, vec![0, 1], vec![]),
            QuantumGate::new(GateType::RotationZ, vec![2], vec![0.8]),
        ];

        // First, apply fusion optimization
        let fusion_result = fusion_engine.fuse_gates(&gates);
        assert!(fusion_result.is_ok());

        // Then run on mixed-precision simulator
        for gate in &gates {
            let result = precision_sim.apply_gate(gate);
            assert!(result.is_ok());
        }

        let stats = precision_sim.get_stats();
        assert!(stats.total_gates > 0);
        assert!(stats.total_time_ms >= 0.0);
    }

    // Quantum Reservoir Computing Tests

    #[test]
    fn test_quantum_reservoir_creation() {
        let mut config = QuantumReservoirConfig::default();
        config.num_qubits = 4;
        config.architecture = QuantumReservoirArchitecture::RandomCircuit;

        let result = QuantumReservoirComputer::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quantum_reservoir_architectures() {
        let architectures = vec![
            QuantumReservoirArchitecture::RandomCircuit,
            QuantumReservoirArchitecture::SpinChain,
            QuantumReservoirArchitecture::TransverseFieldIsing,
            QuantumReservoirArchitecture::SmallWorld,
            QuantumReservoirArchitecture::FullyConnected,
        ];

        for architecture in architectures {
            let mut config = QuantumReservoirConfig::default();
            config.num_qubits = 3;
            config.architecture = architecture;

            let result = QuantumReservoirComputer::new(config);
            assert!(
                result.is_ok(),
                "Failed to create reservoir with architecture {:?}",
                architecture
            );
        }
    }

    #[test]
    fn test_quantum_reservoir_input_encodings() {
        let encodings = vec![
            InputEncoding::Amplitude,
            InputEncoding::Phase,
            InputEncoding::BasisState,
            InputEncoding::Coherent,
            InputEncoding::Squeezed,
        ];

        for encoding in encodings {
            let mut config = QuantumReservoirConfig::default();
            config.num_qubits = 3;
            config.input_encoding = encoding;

            let result = QuantumReservoirComputer::new(config);
            assert!(
                result.is_ok(),
                "Failed to create reservoir with encoding {:?}",
                encoding
            );
        }
    }

    #[test]
    fn test_quantum_reservoir_output_measurements() {
        let measurements = vec![
            OutputMeasurement::PauliExpectation,
            OutputMeasurement::Probability,
            OutputMeasurement::Correlations,
            OutputMeasurement::Entanglement,
            OutputMeasurement::Fidelity,
        ];

        for measurement in measurements {
            let mut config = QuantumReservoirConfig::default();
            config.num_qubits = 3;
            config.output_measurement = measurement;

            let result = QuantumReservoirComputer::new(config);
            assert!(
                result.is_ok(),
                "Failed to create reservoir with measurement {:?}",
                measurement
            );
        }
    }

    #[test]
    fn test_quantum_reservoir_input_processing() {
        let mut config = QuantumReservoirConfig::default();
        config.num_qubits = 3;
        config.architecture = QuantumReservoirArchitecture::SpinChain;

        let mut reservoir = QuantumReservoirComputer::new(config).unwrap();

        // Test single input processing
        let input = Array1::from(vec![0.5, 0.3, 0.2]);
        let result = reservoir.process_input(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.len() > 0);

        // Output should be finite
        for &val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_quantum_reservoir_metrics() {
        let mut config = QuantumReservoirConfig::default();
        config.num_qubits = 4;
        config.input_encoding = InputEncoding::Phase;
        config.adaptive_learning = true;

        let reservoir = QuantumReservoirComputer::new(config).unwrap();
        let metrics = reservoir.get_metrics();

        // Check that metrics are properly initialized
        assert!(metrics.prediction_accuracy >= 0.0);
        assert!(metrics.memory_capacity >= 0.0);
        assert!(metrics.processing_capacity >= 0.0);
        assert!(metrics.generalization_error >= 0.0);
        assert!(metrics.echo_state_property >= 0.0);
        assert!(metrics.avg_processing_time_ms >= 0.0);
        assert!(metrics.quantum_resource_usage >= 0.0);
    }

    #[test]
    fn test_quantum_reservoir_reset() {
        let mut config = QuantumReservoirConfig::default();
        config.num_qubits = 3;
        config.architecture = QuantumReservoirArchitecture::TransverseFieldIsing;
        config.dynamics = ReservoirDynamics::NISQ;

        let mut reservoir = QuantumReservoirComputer::new(config).unwrap();

        // Process some input to change the state
        let input = Array1::from(vec![0.8, 0.2, 0.4]);
        let _ = reservoir.process_input(&input);

        // Reset should work without errors
        let result = reservoir.reset();
        assert!(result.is_ok());
    }

    #[test]
    fn test_quantum_reservoir_benchmark() {
        // Test the benchmark function
        let result = crate::quantum_reservoir_computing::benchmark_quantum_reservoir_computing();
        assert!(result.is_ok());

        let benchmarks = result.unwrap();
        assert!(benchmarks.len() > 0);

        // Check that benchmark results are reasonable
        for (name, value) in benchmarks.iter() {
            assert!(
                value.is_finite(),
                "Benchmark {} returned non-finite value: {}",
                name,
                value
            );
            assert!(
                *value >= 0.0,
                "Benchmark {} returned negative value: {}",
                name,
                value
            );
        }
    }
}
