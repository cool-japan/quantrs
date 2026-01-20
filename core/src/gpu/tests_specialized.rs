//! Comprehensive tests for enhanced GPU kernel optimization
//!
//! This module tests the specialized GPU kernels for holonomic gates,
//! post-quantum cryptography, quantum ML, and adaptive SIMD dispatch.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gate::{single::*, multi::*},
        gpu::{
            adaptive_simd::*,
            specialized_kernels::*,
            *,
        },
        qubit::QubitId,
    };
    use scirs2_core::Complex64;

    /// Test adaptive SIMD CPU feature detection
    #[test]
    fn test_adaptive_simd_feature_detection() {
        let result = initialize_adaptive_simd();
        assert!(result.is_ok(), "Failed to initialize adaptive SIMD: {:?}", result);

        let report = get_adaptive_performance_report();
        assert!(report.is_ok(), "Failed to get performance report: {:?}", report);

        let perf_report = report.expect("Failed to get performance report");
        println!("CPU Features: {:?}", perf_report.cpu_features);
        println!("Selected SIMD Variant: {:?}", perf_report.selected_variant);

        // Basic sanity checks
        assert!(perf_report.cpu_features.num_cores >= 1);
        assert!(perf_report.cpu_features.l1_cache_size > 0);
    }

    /// Test adaptive single-qubit gate application
    #[test]
    fn test_adaptive_single_qubit_gate() {
        let _ = initialize_adaptive_simd();

        let mut state = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        // Hadamard gate matrix
        let hadamard_matrix = [
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        ];

        let result = apply_single_qubit_adaptive(&mut state, 0, &hadamard_matrix);
        assert!(result.is_ok(), "Failed to apply adaptive single-qubit gate: {:?}", result);

        // Verify the state is in superposition
        let expected_amplitude = 1.0 / 2.0_f64.sqrt();
        assert!((state[0].re - expected_amplitude).abs() < 1e-10);
        assert!((state[1].re - expected_amplitude).abs() < 1e-10);

        println!("Adaptive single-qubit gate test passed");
    }

    /// Test specialized GPU kernels creation and configuration
    #[test]
    fn test_specialized_gpu_kernels_creation() {
        let config = OptimizationConfig {
            use_tensor_cores: true,
            optimize_memory_access: true,
            enable_gate_fusion: true,
            max_fusion_length: 8,
            coalescing_threshold: 32,
            use_mixed_precision: true,
        };

        let kernels = SpecializedGpuKernels::new(config);
        assert!(kernels.is_ok(), "Failed to create specialized GPU kernels: {:?}", kernels);

        println!("Specialized GPU kernels created successfully");
    }

    /// Test holonomic gate application (mock implementation)
    #[test]
    fn test_holonomic_gate_application() {
        let config = OptimizationConfig::default();
        let kernels = SpecializedGpuKernels::new(config).expect("Failed to create GPU kernels");

        let mut state = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        // Mock holonomy matrix (2x2 for single qubit)
        let holonomy_matrix = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        let target_qubits = vec![QubitId(0)];

        let result = kernels.apply_holonomic_gate(&mut state, &holonomy_matrix, &target_qubits);
        assert!(result.is_ok(), "Failed to apply holonomic gate: {:?}", result);

        println!("Holonomic gate application test passed");
    }

    /// Test post-quantum cryptographic hash gate
    #[test]
    fn test_post_quantum_hash_gate() {
        let config = OptimizationConfig::default();
        let kernels = SpecializedGpuKernels::new(config).expect("Failed to create GPU kernels for hash gate");

        let mut state = vec![Complex64::new(1.0, 0.0); 8];
        let hash_circuit = vec![Complex64::new(0.5, 0.5); 16];

        let compression_type = PostQuantumCompressionType::QuantumSponge {
            rate: 4,
            capacity: 4,
        };

        let result = kernels.apply_post_quantum_hash_gate(&mut state, &hash_circuit, compression_type);
        assert!(result.is_ok(), "Failed to apply post-quantum hash gate: {:?}", result);

        println!("Post-quantum hash gate test passed");
    }

    /// Test quantum ML attention mechanism
    #[test]
    fn test_quantum_ml_attention() {
        let config = OptimizationConfig::default();
        let kernels = SpecializedGpuKernels::new(config).expect("Failed to create GPU kernels for ML attention");

        let mut state = vec![Complex64::new(0.5, 0.5); 16];
        let query_params = vec![Complex64::new(1.0, 0.0); 8];
        let key_params = vec![Complex64::new(0.0, 1.0); 8];
        let value_params = vec![Complex64::new(0.5, 0.5); 8];
        let num_heads = 2;

        let result = kernels.apply_quantum_ml_attention(
            &mut state,
            &query_params,
            &key_params,
            &value_params,
            num_heads,
        );

        assert!(result.is_ok(), "Failed to apply quantum ML attention: {:?}", result);

        println!("Quantum ML attention test passed");
    }

    /// Test gate fusion optimization
    #[test]
    fn test_gate_fusion() {
        let config = OptimizationConfig {
            enable_gate_fusion: true,
            max_fusion_length: 4,
            ..Default::default()
        };
        let kernels = SpecializedGpuKernels::new(config).expect("Failed to create GPU kernels for gate fusion");

        let mut state = vec![Complex64::new(1.0, 0.0); 4];

        // Create a sequence of gates for fusion
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: QubitId(0) }),
            Box::new(PauliX { target: QubitId(1) }),
            Box::new(Hadamard { target: QubitId(0) }),
        ];

        let result = kernels.apply_fused_gate_sequence(&mut state, &gates);
        assert!(result.is_ok(), "Failed to apply fused gate sequence: {:?}", result);

        println!("Gate fusion test passed");
    }

    /// Test performance reporting
    #[test]
    fn test_performance_reporting() {
        let config = OptimizationConfig::default();
        let kernels = SpecializedGpuKernels::new(config).expect("Failed to create GPU kernels for performance reporting");

        let report = kernels.get_performance_report();

        // Verify report structure
        assert!(report.cache_hit_rate >= 0.0 && report.cache_hit_rate <= 1.0);
        assert!(report.tensor_core_utilization >= 0.0 && report.tensor_core_utilization <= 1.0);
        assert!(report.memory_bandwidth_utilization >= 0.0);

        println!("Performance report: {:?}", report);
        println!("Performance reporting test passed");
    }

    /// Test SIMD variant selection heuristics
    #[test]
    fn test_simd_variant_selection() {
        // Test different CPU feature combinations
        let test_cases = vec![
            (
                CpuFeatures {
                    has_avx512: true,
                    has_avx2: true,
                    has_fma: true,
                    has_avx512vl: true,
                    has_avx512dq: true,
                    has_avx512cd: true,
                    has_sse41: true,
                    has_sse42: true,
                    num_cores: 8,
                    l1_cache_size: 32768,
                    l2_cache_size: 262144,
                    l3_cache_size: 8388608,
                },
                SimdVariant::Avx512,
            ),
            (
                CpuFeatures {
                    has_avx512: false,
                    has_avx2: true,
                    has_fma: true,
                    has_avx512vl: false,
                    has_avx512dq: false,
                    has_avx512cd: false,
                    has_sse41: true,
                    has_sse42: true,
                    num_cores: 4,
                    l1_cache_size: 32768,
                    l2_cache_size: 262144,
                    l3_cache_size: 8388608,
                },
                SimdVariant::Avx2,
            ),
            (
                CpuFeatures {
                    has_avx512: false,
                    has_avx2: false,
                    has_fma: false,
                    has_avx512vl: false,
                    has_avx512dq: false,
                    has_avx512cd: false,
                    has_sse41: true,
                    has_sse42: true,
                    num_cores: 2,
                    l1_cache_size: 32768,
                    l2_cache_size: 262144,
                    l3_cache_size: 8388608,
                },
                SimdVariant::Sse4,
            ),
        ];

        for (features, expected_variant) in test_cases {
            let selected_variant = crate::gpu::adaptive_simd::AdaptiveSimdDispatcher::select_optimal_variant(&features);
            assert_eq!(selected_variant, expected_variant,
                      "Unexpected SIMD variant for features: {:?}", features);
        }

        println!("SIMD variant selection test passed");
    }

    /// Benchmark adaptive SIMD performance
    #[test]
    fn benchmark_adaptive_simd_performance() {
        let _ = initialize_adaptive_simd();

        let sizes = vec![64, 256, 1024, 4096];
        let num_trials = 10;

        for size in sizes {
            let mut state = vec![Complex64::new(1.0, 0.0); size];
            let matrix = [
                Complex64::new(0.7071, 0.0),
                Complex64::new(0.7071, 0.0),
                Complex64::new(0.7071, 0.0),
                Complex64::new(-0.7071, 0.0),
            ];

            let start_time = std::time::Instant::now();

            for _ in 0..num_trials {
                let _ = apply_single_qubit_adaptive(&mut state, 0, &matrix);
            }

            let avg_time = start_time.elapsed().as_nanos() as f64 / num_trials as f64;
            println!("Size {}: Average time = {:.2} ns", size, avg_time);
        }

        println!("Adaptive SIMD performance benchmark completed");
    }

    /// Test error handling in specialized kernels
    #[test]
    fn test_error_handling() {
        let config = OptimizationConfig::default();
        let kernels = SpecializedGpuKernels::new(config).expect("Failed to create GPU kernels for error handling test");

        // Test with empty state
        let mut empty_state = vec![];
        let matrix = [Complex64::new(1.0, 0.0); 4];
        let target_qubits = vec![QubitId(0)];

        let result = kernels.apply_holonomic_gate(&mut empty_state, &matrix, &target_qubits);
        // Should handle gracefully without panicking

        // Test with mismatched dimensions
        let mut small_state = vec![Complex64::new(1.0, 0.0); 2];
        let large_matrix = vec![Complex64::new(1.0, 0.0); 16];
        let multi_qubits = vec![QubitId(0), QubitId(1)];

        let result = kernels.apply_holonomic_gate(&mut small_state, &large_matrix, &multi_qubits);
        // Should handle gracefully

        println!("Error handling test completed");
    }

    /// Integration test with actual quantum circuit
    #[test]
    fn test_integration_quantum_circuit() {
        let _ = initialize_adaptive_simd();
        let config = OptimizationConfig::default();
        let kernels = SpecializedGpuKernels::new(config).expect("Failed to create GPU kernels for integration test");

        // Create a 3-qubit quantum circuit state
        let mut state = vec![Complex64::new(0.0, 0.0); 8];
        state[0] = Complex64::new(1.0, 0.0); // |000⟩

        // Apply a sequence of gates
        let h_matrix = [
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        ];

        // Apply Hadamard to each qubit
        for qubit in 0..3 {
            let result = apply_single_qubit_adaptive(&mut state, qubit, &h_matrix);
            assert!(result.is_ok(), "Failed to apply Hadamard to qubit {}: {:?}", qubit, result);
        }

        // Verify uniform superposition
        let expected_amplitude = 1.0 / (8.0_f64.sqrt());
        for (i, amplitude) in state.iter().enumerate() {
            assert!((amplitude.re - expected_amplitude).abs() < 1e-10,
                   "Incorrect amplitude at position {}: expected {}, got {}",
                   i, expected_amplitude, amplitude.re);
        }

        // Calculate total probability
        let total_prob: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert!((total_prob - 1.0).abs() < 1e-10, "Total probability not normalized: {}", total_prob);

        println!("Integration test with quantum circuit passed");
    }
}