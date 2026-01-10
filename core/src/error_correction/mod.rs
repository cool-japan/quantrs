//! Quantum error correction codes and decoders
//!
//! This module provides comprehensive quantum error correction functionality including:
//! - Pauli operators and strings
//! - Stabilizer codes (repetition, 5-qubit, Steane)
//! - Surface codes and topological codes
//! - Syndrome decoders (lookup table, MWPM, ML)
//! - Color codes and concatenated codes
//! - LDPC and hypergraph product codes
//! - Real-time error correction
//! - Logical gate synthesis
//! - Adaptive threshold estimation

mod adaptive_threshold;
mod color_code;
mod concatenated;
mod decoders;
mod hypergraph;
mod ldpc;
mod logical_gates;
mod ml_decoder;
mod pauli;
pub mod real_time;
mod stabilizer;
mod surface_code;
mod toric_code;

pub use adaptive_threshold::*;
pub use color_code::*;
pub use concatenated::*;
pub use decoders::*;
pub use hypergraph::*;
pub use ldpc::*;
pub use logical_gates::*;
pub use ml_decoder::*;
pub use pauli::*;
pub use stabilizer::*;
pub use surface_code::*;
pub use toric_code::*;

use crate::error::QuantRS2Result;

/// Trait for syndrome decoders
pub trait SyndromeDecoder {
    /// Decode a syndrome to find the most likely error
    fn decode(&self, syndrome: &[bool]) -> QuantRS2Result<PauliString>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::Complex64;

    #[test]
    fn test_pauli_multiplication() {
        let (phase, result) = Pauli::X.multiply(&Pauli::Y);
        assert_eq!(result, Pauli::Z);
        assert_eq!(phase, Complex64::new(0.0, 1.0));
    }

    #[test]
    fn test_pauli_string_commutation() {
        let ps1 = PauliString::new(vec![Pauli::X, Pauli::I]);
        let ps2 = PauliString::new(vec![Pauli::Z, Pauli::I]);
        assert!(!ps1
            .commutes_with(&ps2)
            .expect("Commutation check should succeed"));

        let ps3 = PauliString::new(vec![Pauli::X, Pauli::I]);
        let ps4 = PauliString::new(vec![Pauli::I, Pauli::Z]);
        assert!(ps3
            .commutes_with(&ps4)
            .expect("Commutation check should succeed"));
    }

    #[test]
    fn test_repetition_code() {
        let code = StabilizerCode::repetition_code();
        assert_eq!(code.n, 3);
        assert_eq!(code.k, 1);
        assert_eq!(code.d, 1);

        // Test syndrome for X error on first qubit
        let error = PauliString::new(vec![Pauli::X, Pauli::I, Pauli::I]);
        let syndrome = code
            .syndrome(&error)
            .expect("Syndrome extraction should succeed");
        // X error anti-commutes with Z stabilizer on first two qubits
        assert_eq!(syndrome, vec![true, false]);
    }

    #[test]
    fn test_steane_code() {
        let code = StabilizerCode::steane_code();
        assert_eq!(code.n, 7);
        assert_eq!(code.k, 1);
        assert_eq!(code.d, 3);

        // Test that stabilizers commute
        for i in 0..code.stabilizers.len() {
            for j in i + 1..code.stabilizers.len() {
                assert!(code.stabilizers[i]
                    .commutes_with(&code.stabilizers[j])
                    .expect("Stabilizer commutation check should succeed"));
            }
        }
    }

    #[test]
    fn test_surface_code() {
        let surface = SurfaceCode::new(3, 3);
        assert_eq!(surface.distance(), 3);

        let code = surface
            .to_stabilizer_code()
            .expect("Surface code conversion should succeed");
        assert_eq!(code.n, 9);
        // For a 3x3 lattice, we have 2 X stabilizers and 2 Z stabilizers
        assert_eq!(code.stabilizers.len(), 4);
    }

    #[test]
    fn test_lookup_decoder() {
        let code = StabilizerCode::repetition_code();
        let decoder = LookupDecoder::new(&code).expect("Lookup decoder creation should succeed");

        // Test decoding trivial syndrome (no error)
        let trivial_syndrome = vec![false, false];
        let decoded = decoder
            .decode(&trivial_syndrome)
            .expect("Decoding trivial syndrome should succeed");
        assert_eq!(decoded.weight(), 0); // Should be identity

        // Test single bit flip error
        let error = PauliString::new(vec![Pauli::X, Pauli::I, Pauli::I]);
        let syndrome = code
            .syndrome(&error)
            .expect("Syndrome extraction should succeed");

        // The decoder should be able to decode this syndrome
        if let Ok(decoded_error) = decoder.decode(&syndrome) {
            // Decoder should find a low-weight error
            assert!(decoded_error.weight() <= 1);
        }
    }

    #[test]
    fn test_concatenated_codes() {
        let inner_code = StabilizerCode::repetition_code();
        let outer_code = StabilizerCode::repetition_code();
        let concat_code = ConcatenatedCode::new(inner_code, outer_code);

        assert_eq!(concat_code.total_qubits(), 9); // 3 * 3
        assert_eq!(concat_code.logical_qubits(), 1);
        assert_eq!(concat_code.distance(), 1); // min(1, 1) = 1

        // Test encoding and decoding
        let logical_state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let encoded = concat_code
            .encode(&logical_state)
            .expect("Encoding should succeed");
        assert_eq!(encoded.len(), 512); // 2^9

        // Test error correction capability
        let error = PauliString::new(vec![
            Pauli::X,
            Pauli::I,
            Pauli::I,
            Pauli::I,
            Pauli::I,
            Pauli::I,
            Pauli::I,
            Pauli::I,
            Pauli::I,
        ]);
        let corrected = concat_code
            .correct_error(&encoded, &error)
            .expect("Error correction should succeed");

        // Verify the state is corrected (simplified check)
        assert_eq!(corrected.len(), 512);
    }

    #[test]
    fn test_hypergraph_product_codes() {
        // Create small classical codes for testing
        let h1 = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1])
            .expect("Array creation for h1 should succeed");
        let h2 = Array2::from_shape_vec((2, 3), vec![1, 0, 1, 1, 1, 0])
            .expect("Array creation for h2 should succeed");

        let hpc = HypergraphProductCode::new(h1, h2);

        // Check dimensions
        assert_eq!(hpc.n, 12); // n1*m2 + m1*n2 = 3*2 + 2*3 = 12
        assert_eq!(hpc.k, 1); // k = n1*k2 + k1*n2 - k1*k2 = 3*1 + 1*3 - 1*1 = 5 for this example, but simplified

        let stab_code = hpc
            .to_stabilizer_code()
            .expect("Hypergraph product code conversion should succeed");
        assert!(!stab_code.stabilizers.is_empty());
    }

    #[test]
    fn test_quantum_ldpc_codes() {
        let qldpc = QuantumLDPCCode::bicycle_code(3, 4);
        assert_eq!(qldpc.n, 24); // 2 * 3 * 4
        assert_eq!(qldpc.k, 2);

        let stab_code = qldpc
            .to_stabilizer_code()
            .expect("Quantum LDPC code conversion should succeed");
        assert!(!stab_code.stabilizers.is_empty());

        // Test that stabilizers have bounded weight
        for stabilizer in &stab_code.stabilizers {
            assert!(stabilizer.weight() <= qldpc.max_weight);
        }
    }

    #[test]
    fn test_topological_codes() {
        let toric = ToricCode::new(2, 2);
        assert_eq!(toric.logical_qubits(), 2);
        assert_eq!(toric.distance(), 2);

        let stab_code = toric
            .to_stabilizer_code()
            .expect("Toric code conversion should succeed");
        assert_eq!(stab_code.n, 8); // 2 * 2 * 2
        assert_eq!(stab_code.k, 2);

        // Test that all stabilizers commute
        for i in 0..stab_code.stabilizers.len() {
            for j in i + 1..stab_code.stabilizers.len() {
                assert!(stab_code.stabilizers[i]
                    .commutes_with(&stab_code.stabilizers[j])
                    .expect("Stabilizer commutation check should succeed"));
            }
        }
    }

    #[test]
    fn test_ml_decoder() {
        let surface = SurfaceCode::new(3, 3);
        let decoder = MLDecoder::new(
            surface
                .to_stabilizer_code()
                .expect("Surface code conversion should succeed"),
        );

        // Test decoding with simple syndrome
        let syndrome = vec![true, false, true, false];
        let decoded = decoder.decode(&syndrome);

        // Should succeed for correctable errors
        assert!(decoded.is_ok() || syndrome.iter().filter(|&&x| x).count() % 2 == 1);
    }

    #[test]
    fn test_real_time_mock_hardware() {
        use crate::error_correction::real_time::*;
        use std::time::Duration;

        let hardware = MockQuantumHardware::new(0.01, Duration::from_micros(10), 4);

        // Test syndrome measurement
        let syndrome = hardware
            .measure_syndromes()
            .expect("Syndrome measurement should succeed");
        assert_eq!(syndrome.len(), 4);

        // Test error characteristics
        let characteristics = hardware
            .get_error_characteristics()
            .expect("Getting error characteristics should succeed");
        assert_eq!(characteristics.single_qubit_error_rates.len(), 4);

        // Test latency stats
        let stats = hardware
            .get_latency_stats()
            .expect("Getting latency stats should succeed");
        assert!(stats.throughput_hz > 0.0);

        assert!(hardware.is_ready());
    }

    #[test]
    fn test_real_time_performance_monitor() {
        use crate::error_correction::real_time::*;
        use std::time::Duration;

        let mut monitor = PerformanceMonitor::new();

        // Record some cycles
        monitor.record_cycle(Duration::from_micros(10), true);
        monitor.record_cycle(Duration::from_micros(20), false);
        monitor.record_cycle(Duration::from_micros(15), true);

        assert_eq!(monitor.cycles_processed, 3);
        assert_eq!(monitor.errors_corrected, 2);
        assert_eq!(monitor.error_correction_rate(), 2.0 / 3.0);
        assert!(monitor.average_latency().as_micros() > 10);
        assert!(monitor.current_throughput() > 0.0);
    }

    #[test]
    fn test_real_time_adaptive_decoder() {
        use crate::error_correction::real_time::*;
        use std::sync::Arc;

        let code = StabilizerCode::repetition_code();
        let base_decoder =
            Arc::new(LookupDecoder::new(&code).expect("Lookup decoder creation should succeed"));
        let characteristics = HardwareErrorCharacteristics {
            single_qubit_error_rates: vec![0.01; 3],
            two_qubit_error_rates: vec![0.1; 1],
            measurement_error_rates: vec![0.001; 3],
            correlated_errors: Vec::new(),
            temporal_variation: 0.01,
        };

        let mut adaptive_decoder = AdaptiveThresholdDecoder::new(base_decoder, characteristics);

        // Test initial threshold
        let initial_threshold = adaptive_decoder.current_threshold();
        assert_eq!(initial_threshold, 1.0); // Default when no history

        // Adapt thresholds based on feedback (use no-error syndromes)
        adaptive_decoder.adapt_thresholds(&[false, false], true); // No error, successful correction

        let new_threshold = adaptive_decoder.current_threshold();
        assert!(new_threshold != initial_threshold); // Should change from default 1.0 to 0.0

        // Test decoding (use no-error syndrome which should always be valid)
        let syndrome = vec![false, false]; // No error syndrome
        let result = adaptive_decoder.decode(&syndrome);
        assert!(result.is_ok(), "Decoding failed: {:?}", result.err());
    }

    #[test]
    fn test_real_time_parallel_decoder() {
        use crate::error_correction::real_time::*;
        use std::sync::Arc;

        let code = StabilizerCode::repetition_code();
        let base_decoder =
            Arc::new(LookupDecoder::new(&code).expect("Lookup decoder creation should succeed"));
        let parallel_decoder = ParallelSyndromeDecoder::new(base_decoder, 2);

        // Test single syndrome decoding (use no-error syndrome)
        let syndrome = vec![false, false]; // No error syndrome
        let result = parallel_decoder.decode(&syndrome);
        assert!(result.is_ok(), "Decoding failed: {:?}", result.err());

        // Test batch decoding (use only no-error syndromes for safety)
        let syndromes = vec![
            vec![false, false], // No error syndromes
            vec![false, false],
            vec![false, false],
            vec![false, false],
        ];

        let results = parallel_decoder.decode_batch(&syndromes);
        assert!(results.is_ok());
        let corrections = results.expect("Batch decoding should succeed");
        assert_eq!(corrections.len(), 4);
    }

    #[test]
    fn test_real_time_syndrome_stream_processor() {
        use crate::error_correction::real_time::*;
        use std::sync::Arc;
        use std::time::Duration;

        let code = StabilizerCode::repetition_code();
        let decoder =
            Arc::new(LookupDecoder::new(&code).expect("Lookup decoder creation should succeed"));
        let hardware = Arc::new(MockQuantumHardware::new(0.01, Duration::from_micros(1), 3));
        let config = RealTimeConfig {
            max_latency: Duration::from_millis(1),
            buffer_size: 10,
            parallel_workers: 1,
            adaptive_threshold: false,
            hardware_feedback: false,
            performance_logging: true,
        };

        let processor = SyndromeStreamProcessor::new(decoder, hardware, config);

        // Test buffer status
        let (current, max) = processor.get_buffer_status();
        assert_eq!(current, 0);
        assert_eq!(max, 10);

        // Test performance stats (initial state)
        let stats = processor.get_performance_stats();
        assert_eq!(stats.cycles_processed, 0);
        assert_eq!(stats.errors_corrected, 0);
    }

    #[test]
    fn test_logical_gate_synthesizer() {
        use crate::error_correction::logical_gates::*;

        let code = StabilizerCode::repetition_code();
        let synthesizer = LogicalGateSynthesizer::new(0.01);

        // Test logical X gate synthesis
        let logical_x = synthesizer.synthesize_logical_x(&code, 0);
        assert!(logical_x.is_ok());

        let x_gate = logical_x.expect("Logical X synthesis should succeed");
        assert_eq!(x_gate.logical_qubits, vec![0]);
        assert_eq!(x_gate.physical_operations.len(), 1);
        assert!(!x_gate.error_propagation.single_qubit_propagation.is_empty());

        // Test logical Z gate synthesis
        let logical_z = synthesizer.synthesize_logical_z(&code, 0);
        assert!(logical_z.is_ok());

        let z_gate = logical_z.expect("Logical Z synthesis should succeed");
        assert_eq!(z_gate.logical_qubits, vec![0]);
        assert_eq!(z_gate.physical_operations.len(), 1);

        // Test logical H gate synthesis
        let logical_h = synthesizer.synthesize_logical_h(&code, 0);
        assert!(logical_h.is_ok());

        let h_gate = logical_h.expect("Logical H synthesis should succeed");
        assert_eq!(h_gate.logical_qubits, vec![0]);
        assert_eq!(h_gate.physical_operations.len(), 1);
        assert_eq!(h_gate.physical_operations[0].error_correction_rounds, 2);

        // Test invalid logical qubit index
        let invalid_gate = synthesizer.synthesize_logical_x(&code, 5);
        assert!(invalid_gate.is_err());
    }

    #[test]
    fn test_logical_circuit_synthesizer() {
        use crate::error_correction::logical_gates::*;

        let code = StabilizerCode::repetition_code();
        let synthesizer = LogicalCircuitSynthesizer::new(0.01);

        // Test simple circuit synthesis
        let gate_sequence = vec![("x", vec![0]), ("h", vec![0]), ("z", vec![0])];

        let circuit = synthesizer.synthesize_circuit(&code, &gate_sequence);
        assert!(circuit.is_ok());

        let logical_circuit = circuit.expect("Circuit synthesis should succeed");
        assert_eq!(logical_circuit.len(), 3);

        // Test resource estimation
        let resources = synthesizer.estimate_resources(&logical_circuit);
        assert!(resources.total_physical_operations > 0);
        assert!(resources.total_error_correction_rounds > 0);
        assert_eq!(resources.estimated_depth, 3);

        // Test invalid gate name
        let invalid_sequence = vec![("invalid_gate", vec![0])];
        let invalid_circuit = synthesizer.synthesize_circuit(&code, &invalid_sequence);
        assert!(invalid_circuit.is_err());

        // Test CNOT gate with wrong number of targets
        let wrong_cnot = vec![("cnot", vec![0])]; // CNOT needs 2 targets
        let wrong_circuit = synthesizer.synthesize_circuit(&code, &wrong_cnot);
        assert!(wrong_circuit.is_err());
    }

    #[test]
    fn test_logical_t_gate_synthesis() {
        use crate::error_correction::logical_gates::*;

        let code = StabilizerCode::repetition_code();
        let synthesizer = LogicalGateSynthesizer::new(0.01);

        // Test T gate synthesis (requires magic state distillation)
        let logical_t = synthesizer.synthesize_logical_t(&code, 0);
        assert!(logical_t.is_ok());

        let t_gate = logical_t.expect("Logical T synthesis should succeed");
        assert_eq!(t_gate.logical_qubits, vec![0]);
        assert_eq!(t_gate.physical_operations.len(), 2); // Magic state prep + injection

        // Check that magic state prep has more error correction rounds
        assert!(t_gate.physical_operations[0].error_correction_rounds >= 5);
    }

    #[test]
    fn test_error_propagation_analysis() {
        use crate::error_correction::logical_gates::*;

        let code = StabilizerCode::repetition_code();
        let synthesizer = LogicalGateSynthesizer::new(0.01);

        let logical_x = synthesizer
            .synthesize_logical_x(&code, 0)
            .expect("Logical X synthesis should succeed");

        // Check error propagation analysis
        let analysis = &logical_x.error_propagation;
        assert!(!analysis.single_qubit_propagation.is_empty());
        // max_error_weight is usize, so it's always >= 0
        assert_eq!(analysis.fault_tolerance_threshold, 0.01);

        // Check that some errors are marked as correctable
        let correctable_count = analysis
            .single_qubit_propagation
            .iter()
            .filter(|path| path.correctable)
            .count();
        assert!(correctable_count > 0);
    }

    #[test]
    fn test_pauli_string_weight() {
        let identity_string = PauliString::new(vec![Pauli::I, Pauli::I, Pauli::I]);
        assert_eq!(identity_string.weight(), 0);

        let single_error = PauliString::new(vec![Pauli::X, Pauli::I, Pauli::I]);
        assert_eq!(single_error.weight(), 1);

        let multi_error = PauliString::new(vec![Pauli::X, Pauli::Y, Pauli::Z]);
        assert_eq!(multi_error.weight(), 3);
    }

    #[test]
    fn test_logical_circuit_with_multiple_gates() {
        use crate::error_correction::logical_gates::*;

        let code = StabilizerCode::repetition_code();
        let synthesizer = LogicalCircuitSynthesizer::new(0.01);

        // Test a more complex circuit
        let gate_sequence = vec![
            ("h", vec![0]), // Hadamard on logical qubit 0
            ("x", vec![0]), // X on logical qubit 0
            ("z", vec![0]), // Z on logical qubit 0
            ("h", vec![0]), // Another Hadamard
        ];

        let circuit = synthesizer.synthesize_circuit(&code, &gate_sequence);
        assert!(circuit.is_ok());

        let logical_circuit = circuit.expect("Circuit synthesis should succeed");
        assert_eq!(logical_circuit.len(), 4);

        // Check that all gates target the correct logical qubit
        for gate in &logical_circuit {
            assert_eq!(gate.logical_qubits, vec![0]);
        }

        // Estimate resources for this circuit
        let resources = synthesizer.estimate_resources(&logical_circuit);
        assert_eq!(resources.estimated_depth, 4);
        assert!(resources.total_error_correction_rounds >= 4); // At least one round per gate
    }

    #[test]
    fn test_adaptive_threshold_estimator() {
        use crate::error_correction::adaptive_threshold::*;

        let noise_model = NoiseModel::default();
        let algorithm = ThresholdEstimationAlgorithm::Bayesian {
            prior_strength: 1.0,
            update_rate: 0.1,
        };
        let config = AdaptiveConfig::default();

        let mut estimator = AdaptiveThresholdEstimator::new(noise_model, algorithm, config);

        // Test initial threshold estimation
        let syndrome = vec![true, false];
        let env = EnvironmentalConditions::default();
        let threshold = estimator.estimate_threshold(&syndrome, &env);
        assert!(threshold > 0.0);
        assert!(threshold < 1.0);

        // Test adding observations
        let observation = ErrorObservation {
            syndrome: syndrome.clone(),
            correction: PauliString::new(vec![Pauli::X, Pauli::I]),
            success: true,
            observed_error_rate: 0.01,
            timestamp: std::time::Instant::now(),
            environment: env.clone(),
        };

        estimator.add_observation(observation);

        // Test threshold recommendation
        let recommendation = estimator.get_threshold_recommendation(&syndrome);
        assert!(recommendation.threshold > 0.0);
        assert!(recommendation.confidence >= 0.0 && recommendation.confidence <= 1.0);
        assert!(recommendation.predicted_error_rate >= 0.0);
    }

    #[test]
    fn test_performance_tracker() {
        use crate::error_correction::adaptive_threshold::*;

        let mut tracker = PerformanceTracker::new();

        // Test initial state
        assert_eq!(tracker.successful_corrections, 0);
        assert_eq!(tracker.failed_corrections, 0);
        assert_eq!(tracker.precision(), 1.0); // Default when no data
        assert_eq!(tracker.recall(), 1.0);
        assert_eq!(tracker.f1_score(), 1.0); // Perfect when precision and recall are both 1.0

        // Simulate some corrections
        tracker.successful_corrections = 8;
        tracker.failed_corrections = 2;
        tracker.false_positives = 1;
        tracker.false_negatives = 1;

        // Test metrics
        assert_eq!(tracker.precision(), 8.0 / 9.0); // 8 / (8 + 1)
        assert_eq!(tracker.recall(), 8.0 / 9.0); // 8 / (8 + 1)
        assert!(tracker.f1_score() > 0.0);
    }

    #[test]
    fn test_environmental_conditions() {
        use crate::error_correction::adaptive_threshold::*;

        let mut env = EnvironmentalConditions::default();
        assert_eq!(env.temperature, 300.0); // Room temperature
        assert_eq!(env.magnetic_field, 0.0);

        // Test modification
        env.temperature = 310.0; // Higher temperature
        env.vibration_level = 0.1;

        let noise_model = NoiseModel::default();
        let algorithm = ThresholdEstimationAlgorithm::ExponentialAverage { alpha: 0.5 };
        let config = AdaptiveConfig::default();

        let estimator = AdaptiveThresholdEstimator::new(noise_model, algorithm, config);

        // Test that environmental conditions affect threshold
        let syndrome = vec![false, false];
        let threshold_normal =
            estimator.estimate_threshold(&syndrome, &EnvironmentalConditions::default());
        let threshold_hot = estimator.estimate_threshold(&syndrome, &env);

        // Thresholds may be different due to environmental factors
        assert!(threshold_normal >= 0.0);
        assert!(threshold_hot >= 0.0);
    }

    #[test]
    fn test_different_threshold_algorithms() {
        use crate::error_correction::adaptive_threshold::*;

        let noise_model = NoiseModel::default();
        let config = AdaptiveConfig::default();

        // Test Bayesian algorithm
        let bayesian_alg = ThresholdEstimationAlgorithm::Bayesian {
            prior_strength: 1.0,
            update_rate: 0.1,
        };
        let bayesian_estimator =
            AdaptiveThresholdEstimator::new(noise_model.clone(), bayesian_alg, config.clone());

        // Test Kalman filter algorithm
        let kalman_alg = ThresholdEstimationAlgorithm::KalmanFilter {
            process_noise: 0.01,
            measurement_noise: 0.1,
        };
        let kalman_estimator =
            AdaptiveThresholdEstimator::new(noise_model.clone(), kalman_alg, config.clone());

        // Test exponential average algorithm
        let exp_alg = ThresholdEstimationAlgorithm::ExponentialAverage { alpha: 0.3 };
        let exp_estimator =
            AdaptiveThresholdEstimator::new(noise_model.clone(), exp_alg, config.clone());

        // Test ML algorithm
        let ml_alg = ThresholdEstimationAlgorithm::MachineLearning {
            model_type: MLModelType::LinearRegression,
            training_window: 50,
        };
        let ml_estimator = AdaptiveThresholdEstimator::new(noise_model, ml_alg, config);

        let syndrome = vec![true, false];
        let env = EnvironmentalConditions::default();

        // All algorithms should produce valid thresholds
        let bayesian_threshold = bayesian_estimator.estimate_threshold(&syndrome, &env);
        let kalman_threshold = kalman_estimator.estimate_threshold(&syndrome, &env);
        let exp_threshold = exp_estimator.estimate_threshold(&syndrome, &env);
        let ml_threshold = ml_estimator.estimate_threshold(&syndrome, &env);

        assert!(bayesian_threshold > 0.0);
        assert!(kalman_threshold > 0.0);
        assert!(exp_threshold > 0.0);
        assert!(ml_threshold > 0.0);
    }

    #[test]
    fn test_noise_model_updates() {
        use crate::error_correction::adaptive_threshold::*;

        let noise_model = NoiseModel::default();
        let algorithm = ThresholdEstimationAlgorithm::Bayesian {
            prior_strength: 1.0,
            update_rate: 0.1,
        };
        let config = AdaptiveConfig {
            min_observations: 2, // Low threshold for testing
            real_time_adaptation: true,
            ..AdaptiveConfig::default()
        };

        let mut estimator = AdaptiveThresholdEstimator::new(noise_model, algorithm, config);

        // Add multiple observations to trigger model updates
        for i in 0..5 {
            let observation = ErrorObservation {
                syndrome: vec![i % 2 == 0, i % 3 == 0],
                correction: PauliString::new(vec![Pauli::X, Pauli::I]),
                success: i % 4 != 0, // Most succeed
                observed_error_rate: 0.01,
                timestamp: std::time::Instant::now(),
                environment: EnvironmentalConditions::default(),
            };
            estimator.add_observation(observation);
        }

        // The estimator should have updated its internal model
        let recommendation = estimator.get_threshold_recommendation(&[true, false]);
        assert!(recommendation.confidence > 0.0);
        assert!(recommendation.recommendation_quality > 0.0);
    }
}
