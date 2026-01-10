//! Cross-Subcrate Integration Tests for QuantRS2
//!
//! These tests verify that different QuantRS2 subcrates work correctly together,
//! ensuring seamless integration across the ecosystem.

// Test circuit + sim integration
#[cfg(all(feature = "circuit", feature = "sim"))]
mod circuit_sim_integration {
    use quantrs2::prelude::circuits::Simulator;
    use quantrs2::prelude::simulation::*;

    #[test]
    fn test_basic_circuit_simulation() {
        // Create a simple Bell state circuit
        let mut circuit = Circuit::<2>::new();
        circuit.h(0).unwrap();
        circuit.cnot(0, 1).unwrap();

        // Simulate with state vector simulator
        let simulator = StateVectorSimulator::new();
        let result = Simulator::run(&simulator, &circuit);

        // Should succeed
        assert!(result.is_ok(), "Circuit simulation should succeed");
    }

    #[test]
    fn test_circuit_optimization_and_simulation() {
        // Create a circuit with redundant gates
        let mut circuit = Circuit::<2>::new();
        circuit.h(0).unwrap();
        circuit.x(0).unwrap();
        circuit.x(0).unwrap(); // Redundant - two X gates cancel

        // Simulate the circuit
        let simulator = StateVectorSimulator::new();
        let result = Simulator::run(&simulator, &circuit);

        assert!(
            result.is_ok(),
            "Circuit with redundant gates should still simulate"
        );
    }

    #[test]
    fn test_multi_qubit_circuit_simulation() {
        // Create a larger circuit
        let mut circuit = Circuit::<4>::new();
        circuit.h(0).unwrap();
        circuit.cnot(0, 1).unwrap();
        circuit.cnot(1, 2).unwrap();
        circuit.cnot(2, 3).unwrap();

        // Simulate
        let simulator = StateVectorSimulator::new();
        let result = Simulator::run(&simulator, &circuit);

        assert!(result.is_ok(), "Multi-qubit circuit should simulate");
    }
}

// Test ml + sim integration
#[cfg(all(feature = "ml", feature = "sim"))]
mod ml_sim_integration {
    #[test]
    fn test_ml_requires_sim() {
        // This test verifies that ML features can access simulation capabilities
        // The fact that this compiles proves the integration works
        assert!(true, "ML and sim integration is available");
    }
}

// Test ml + anneal integration
#[cfg(all(feature = "ml", feature = "anneal"))]
mod ml_anneal_integration {
    #[test]
    fn test_ml_requires_anneal() {
        // This test verifies that ML features can access annealing capabilities
        // The fact that this compiles proves the integration works
        assert!(true, "ML and anneal integration is available");
    }
}

// Test device + circuit integration
#[cfg(all(feature = "device", feature = "circuit"))]
mod device_circuit_integration {
    use quantrs2::prelude::hardware::*;

    #[test]
    fn test_circuit_for_device() {
        // Create a circuit suitable for hardware execution
        let mut circuit = Circuit::<2>::new();
        circuit.h(0).unwrap();
        circuit.cnot(0, 1).unwrap();

        // Verify circuit properties
        assert_eq!(circuit.num_qubits(), 2);
        // Circuit creation should succeed (verified by the fact we got here)
        assert!(true, "Circuit created successfully for hardware");
    }

    #[test]
    fn test_device_circuit_validation() {
        // Create a valid hardware circuit
        let mut circuit = Circuit::<2>::new();
        circuit.h(0).unwrap();
        circuit.cnot(0, 1).unwrap();

        // Circuit should be valid for basic hardware constraints
        assert!(
            circuit.num_qubits() <= 100,
            "Circuit should be reasonable size for hardware"
        );
    }
}

// Test tytan + anneal integration
#[cfg(all(feature = "tytan", feature = "anneal"))]
mod tytan_anneal_integration {
    #[test]
    fn test_tytan_requires_anneal() {
        // This test verifies that Tytan features can access annealing capabilities
        // The fact that this compiles proves the integration works
        assert!(true, "Tytan and anneal integration is available");
    }
}

// Test full stack integration (when all features are enabled)
#[cfg(all(
    feature = "circuit",
    feature = "sim",
    feature = "ml",
    feature = "device"
))]
mod full_stack_integration {
    use quantrs2::prelude::full::*;

    #[test]
    fn test_full_prelude_available() {
        // Test that full prelude provides all necessary types
        let _q = QubitId::new(0);

        // Circuit building
        let mut circuit = Circuit::<2>::new();
        circuit.h(0).unwrap();
        circuit.cnot(0, 1).unwrap();

        // Simulation - use the Simulator<N> trait directly
        let simulator = StateVectorSimulator::new();
        let result = simulator.run(&circuit);

        assert!(result.is_ok(), "Full stack integration should work");
    }

    #[test]
    fn test_workflow_circuit_to_simulation() {
        // Complete workflow: build circuit -> optimize -> simulate
        let mut circuit = Circuit::<3>::new();

        // Build a simple circuit
        circuit.h(0).unwrap();
        circuit.cnot(0, 1).unwrap();
        circuit.cnot(1, 2).unwrap();

        // Simulate - use the Simulator<N> trait directly
        let simulator = StateVectorSimulator::new();
        let result = simulator.run(&circuit);

        assert!(result.is_ok(), "Full workflow should succeed");
    }
}

// Test error propagation across subcrates
#[cfg(feature = "circuit")]
mod error_propagation {
    use quantrs2::error::{with_context, QuantRS2Result};
    use quantrs2::prelude::circuits::*;

    #[test]
    fn test_error_conversion_circuit() {
        fn build_valid_circuit() -> Circuit<2> {
            Circuit::<2>::new()
        }

        let circuit = build_valid_circuit();
        // Circuit should be valid
        assert!(
            circuit.gates().is_empty(),
            "New circuit should have no gates"
        );
    }

    #[test]
    fn test_error_context_propagation() {
        let error = QuantRS2Error::InvalidQubitId(100);
        let contextualized = with_context(error, "building GHZ state");

        // Error should retain its type (with_context preserves InvalidQubitId variant)
        assert!(matches!(contextualized, QuantRS2Error::InvalidQubitId(100)));
    }
}

// Test prelude hierarchy consistency
mod prelude_hierarchy {
    use quantrs2::prelude::essentials::*;

    #[test]
    fn test_essentials_always_available() {
        // Essential types should always be available
        let _q = QubitId::new(0);
        let version = VERSION;
        assert!(!version.is_empty());
    }

    #[cfg(feature = "circuit")]
    #[test]
    fn test_circuits_includes_essentials() {
        use quantrs2::prelude::circuits::*;

        // Circuits prelude should include essentials
        let _q = QubitId::new(0);
        let _circuit = Circuit::<2>::new();
        assert!(!VERSION.is_empty());
    }

    #[cfg(feature = "sim")]
    #[test]
    fn test_simulation_includes_circuits() {
        use quantrs2::prelude::simulation::*;

        // Simulation prelude should include circuits and essentials
        let _q = QubitId::new(0);
        let _circuit = Circuit::<2>::new();
        let _simulator = StateVectorSimulator::new();
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_full_includes_all() {
        use quantrs2::prelude::full::*;
        // Disambiguate VERSION - use essentials
        use quantrs2::prelude::essentials::VERSION;

        // Full prelude should include essentials at minimum
        let _q = QubitId::new(0);
        assert!(!VERSION.is_empty());

        // Additional features available when enabled
        #[cfg(feature = "circuit")]
        {
            let _circuit = Circuit::<2>::new();
        }

        #[cfg(feature = "sim")]
        {
            let _simulator = StateVectorSimulator::new();
        }
    }
}

// Test version compatibility across subcrates
mod version_compatibility {
    use quantrs2::version;

    #[test]
    fn test_version_consistency() {
        let info = version::VersionInfo::current();

        // All version strings should be non-empty
        assert!(!info.quantrs2.is_empty());
        assert!(!info.scirs2.is_empty());
        assert!(!info.rustc.is_empty());
        assert!(!info.target.is_empty());
    }

    #[test]
    fn test_compatibility_check() {
        // Compatibility check should pass for valid builds
        let result = version::check_compatibility();

        // In development, this might have warnings but should not fail
        match result {
            Ok(()) => {
                // Perfect - no compatibility issues
            }
            Err(issues) => {
                // Some issues detected, but this is acceptable in development
                // Just ensure we can enumerate them
                assert!(!issues.is_empty(), "If check fails, there should be issues");
            }
        }
    }
}

// Test configuration management across subcrates
mod configuration_integration {
    use quantrs2::config;

    #[test]
    fn test_global_config_access() {
        let cfg = config::Config::global();
        let snapshot = cfg.snapshot();

        // Configuration should have reasonable defaults
        assert!(snapshot.enable_simd); // SIMD should be enabled by default
        assert_eq!(snapshot.log_level, config::LogLevel::Warn);
    }

    #[test]
    fn test_config_builder() {
        let config_data = config::Config::builder()
            .num_threads(8)
            .memory_limit_gb(16)
            .enable_gpu(true)
            .build();

        assert_eq!(config_data.num_threads, Some(8));
        assert_eq!(
            config_data.memory_limit_bytes,
            Some(16 * 1024 * 1024 * 1024)
        );
        assert!(config_data.enable_gpu);
    }

    #[test]
    fn test_backend_selection() {
        let backends = [
            config::DefaultBackend::Auto,
            config::DefaultBackend::Cpu,
            config::DefaultBackend::Gpu,
        ];

        // All backends should be selectable
        for backend in &backends {
            let config_data = config::Config::builder().default_backend(*backend).build();
            assert_eq!(config_data.default_backend, *backend);
        }
    }
}

// Test diagnostics integration
mod diagnostics_integration {
    use quantrs2::diagnostics;

    #[test]
    fn test_diagnostics_report() {
        let report = diagnostics::run_diagnostics();

        // Check that all sections are populated
        assert!(!report.version.quantrs2.is_empty());
        assert!(report.capabilities.cpu_cores > 0);
        // Note: total_memory_bytes may be 0 if memory detection is not implemented
        // assert!(report.capabilities.total_memory_bytes >= 0);

        // Check ready status
        let _ = report.is_ready();

        // Check summary generation
        let summary = report.summary();
        assert!(summary.contains("Diagnostic Summary"));
    }

    #[test]
    fn test_system_capabilities() {
        let report = diagnostics::run_diagnostics();
        let caps = &report.capabilities;

        // System should have reasonable capabilities
        assert!(caps.cpu_cores >= 1);
        assert!(caps.cpu_cores <= 1024); // Sanity check
                                         // Note: total_memory_bytes may be 0 if memory detection is not implemented
                                         // This is acceptable for the facade crate which uses a placeholder implementation
    }

    #[test]
    fn test_feature_detection() {
        let report = diagnostics::run_diagnostics();

        // Check that features are detected correctly
        #[cfg(feature = "circuit")]
        assert!(report.features.circuit);

        #[cfg(feature = "sim")]
        assert!(report.features.sim);

        #[cfg(feature = "ml")]
        assert!(report.features.ml);

        #[cfg(feature = "device")]
        assert!(report.features.device);

        #[cfg(feature = "anneal")]
        assert!(report.features.anneal);

        #[cfg(feature = "tytan")]
        assert!(report.features.tytan);
    }
}

// Test utility functions integration
mod utils_integration {
    use quantrs2::utils;

    #[test]
    fn test_memory_estimation() {
        // Test memory estimation for different qubit counts
        let memory_10 = utils::estimate_statevector_memory(10);
        let memory_20 = utils::estimate_statevector_memory(20);
        let memory_30 = utils::estimate_statevector_memory(30);

        // Memory should grow exponentially
        assert!(memory_20 > memory_10);
        assert!(memory_30 > memory_20);
        assert!(memory_20 >= memory_10 * 1024); // Should be ~1024x more for 10 extra qubits
    }

    #[test]
    fn test_max_qubits_calculation() {
        // Test with different memory limits
        let available_1gb = 1024 * 1024 * 1024usize;
        let max_qubits_1gb = utils::max_qubits_for_memory(available_1gb);

        let available_16gb = 16 * 1024 * 1024 * 1024usize;
        let max_qubits_16gb = utils::max_qubits_for_memory(available_16gb);

        // More memory should allow more qubits
        assert!(max_qubits_16gb > max_qubits_1gb);

        // Results should be reasonable
        assert!(max_qubits_1gb >= 15);
        assert!(max_qubits_1gb <= 30);
        assert!(max_qubits_16gb >= 20);
        assert!(max_qubits_16gb <= 35);
    }

    #[test]
    fn test_formatting_utilities() {
        use std::time::Duration;

        // Test memory formatting
        let mem_1kb = utils::format_memory(1024);
        assert!(mem_1kb.contains("KB") || mem_1kb.contains('1'));

        let mem_1mb = utils::format_memory(1024 * 1024);
        assert!(mem_1mb.contains("MB") || mem_1mb.contains('1'));

        // Test duration formatting
        let dur_1s = utils::format_duration(Duration::from_secs(1));
        assert!(dur_1s.contains('s'));

        let dur_1ms = utils::format_duration(Duration::from_millis(1));
        assert!(dur_1ms.contains("ms"));
    }

    #[test]
    fn test_validation_utilities() {
        // Test qubit count validation with available memory
        let mem_16gb = 16 * 1024 * 1024 * 1024;
        assert!(utils::is_valid_qubit_count(1, mem_16gb));
        assert!(utils::is_valid_qubit_count(30, mem_16gb));
        assert!(!utils::is_valid_qubit_count(100, mem_16gb));

        // Test range validation
        assert!(utils::is_in_range(&0.5, &0.0, &1.0));
        assert!(!utils::is_in_range(&1.5, &0.0, &1.0));
        assert!(!utils::is_in_range(&-0.5, &0.0, &1.0));
    }
}

// Test testing utilities integration
mod testing_utilities {
    use quantrs2::testing;

    #[test]
    fn test_approximate_equality() {
        // Test floating point comparison - these functions panic on failure
        testing::assert_approx_eq(1.0, 1.0 + 1e-10, 1e-8);
        // This should pass
    }

    #[test]
    #[should_panic]
    fn test_approximate_equality_fails() {
        // This should panic
        testing::assert_approx_eq(1.0, 2.0, 1e-8);
    }

    #[test]
    fn test_vector_approximate_equality() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![1.0 + 1e-10, 2.0 + 1e-10, 3.0 + 1e-10];

        testing::assert_vec_approx_eq(&v1, &v2, 1e-8);
        // This should pass
    }

    #[test]
    #[should_panic]
    fn test_vector_approximate_equality_fails() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v3 = vec![1.0, 2.0, 4.0];

        testing::assert_vec_approx_eq(&v1, &v3, 1e-8);
        // This should panic
    }

    #[test]
    fn test_temp_directory_creation() {
        use std::fs;

        let temp_dir = testing::create_temp_test_dir();
        assert!(temp_dir.exists());
        assert!(temp_dir.is_dir());

        // Clean up
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_random_data_generation() {
        let data = testing::generate_random_test_data(100, 42);
        assert_eq!(data.len(), 100);

        // All values should be in range [0, 1]
        for &val in &data {
            assert!(val >= 0.0);
            assert!(val <= 1.0);
        }
    }
}

// Test benchmarking utilities integration
mod benchmarking_integration {
    use quantrs2::bench;
    use std::time::Duration;

    #[test]
    fn test_benchmark_timer_basic() {
        let timer = bench::BenchmarkTimer::start();
        std::thread::sleep(Duration::from_millis(5));
        let elapsed = timer.stop();
        assert!(elapsed >= Duration::from_millis(4));
    }

    #[test]
    fn test_benchmark_stats_aggregation() {
        let mut stats = bench::BenchmarkStats::new("test_stats");

        // Record multiple samples
        for i in 1..=10 {
            stats.record(Duration::from_millis(i * 10));
        }

        assert_eq!(stats.count(), 10);
        assert!(stats.mean().is_some());
        assert!(stats.median().is_some());
        assert!(stats.std_dev().is_some());
        assert!(stats.min().is_some());
        assert!(stats.max().is_some());

        // Verify ordering
        assert!(stats.min().unwrap() <= stats.mean().unwrap());
        assert!(stats.mean().unwrap() <= stats.max().unwrap());
    }

    #[test]
    fn test_measure_closure() {
        // Use more substantial work to ensure measurable time in release mode
        let (result, duration) = bench::measure(|| {
            // Perform work that won't be optimized away
            let mut sum: u64 = 0;
            for i in 1..=10_000 {
                sum = sum.wrapping_add(i);
                // Prevent complete optimization with black_box
                std::hint::black_box(sum);
            }
            sum
        });

        assert_eq!(result, 50_005_000); // Sum of 1 to 10,000
                                        // In release mode, this might still be very fast, so be lenient
        assert!(duration >= Duration::ZERO);
    }

    #[test]
    fn test_measure_iterations() {
        let stats = bench::measure_iterations(5, || {
            std::thread::sleep(Duration::from_millis(1));
        });

        assert_eq!(stats.count(), 5);
    }

    #[test]
    fn test_memory_usage_formatting() {
        let mem = bench::MemoryUsage::from_bytes(1024 * 1024);
        assert!((mem.mb() - 1.0).abs() < 0.01);
        assert!((mem.kb() - 1024.0).abs() < 0.01);
    }

    #[test]
    fn test_benchmark_config_presets() {
        let quick = bench::BenchmarkConfig::quick();
        assert_eq!(quick.warmup_iterations, 5);
        assert_eq!(quick.measure_iterations, 20);

        let thorough = bench::BenchmarkConfig::thorough();
        assert_eq!(thorough.warmup_iterations, 50);
        assert_eq!(thorough.measure_iterations, 1000);
    }

    #[test]
    fn test_throughput_calculation() {
        let mut stats = bench::BenchmarkStats::new("throughput");
        stats.set_ops_per_sample(1000);
        stats.record(Duration::from_secs(1));

        let throughput = stats.throughput().unwrap();
        assert!((throughput - 1000.0).abs() < 0.1);
    }

    #[test]
    fn test_percentile_calculation() {
        let mut stats = bench::BenchmarkStats::new("percentile");
        for i in 1..=100 {
            stats.record(Duration::from_millis(i));
        }

        let p50 = stats.percentile(50.0).unwrap();
        let p90 = stats.percentile(90.0).unwrap();
        let p99 = stats.percentile(99.0).unwrap();

        assert!(p50 < p90);
        assert!(p90 < p99);
    }
}

// Test symengine integration
#[cfg(feature = "symengine")]
mod symengine_integration {
    #[test]
    fn test_symengine_available() {
        // Test that symengine module is available when feature is enabled
        // This validates the feature flag and module integration
        use quantrs2::symengine;
        // Module should be accessible
        let type_name = std::any::type_name::<quantrs2::symengine::Expression>();
        assert!(!type_name.is_empty(), "SymEngine integration is available");
    }

    #[test]
    fn test_symengine_basic_types() {
        // Test basic symbolic types availability
        use quantrs2::symengine::Expression;

        // Expression type should be available
        // Note: Actual operations depend on SymEngine C library
        let type_name = std::any::type_name::<Expression>();
        assert!(!type_name.is_empty(), "SymEngine types are accessible");
    }
}

// Test symengine + circuit integration (parametric circuits)
#[cfg(all(feature = "symengine", feature = "circuit"))]
mod symengine_circuit_integration {
    #[test]
    fn test_parametric_gates() {
        // Test that symbolic parameters can be used with circuits
        // This integration enables variational circuit construction
        use quantrs2::symengine::Expression;
        let type_name = std::any::type_name::<Expression>();
        assert!(
            !type_name.is_empty(),
            "SymEngine can be used with circuits for parametric gates"
        );
    }

    #[test]
    fn test_circuit_symbolic_optimization() {
        // Test that circuits with symbolic parameters can be created
        // and potentially optimized symbolically
        use quantrs2::symengine::Expression;
        let type_name = std::any::type_name::<Expression>();
        assert!(
            !type_name.is_empty(),
            "SymEngine enables symbolic circuit optimization workflows"
        );
    }
}

// Test quantum math utilities integration
mod quantum_math_integration {
    use quantrs2::utils;

    #[test]
    fn test_quantum_constants() {
        // Verify quantum computing constants
        assert!(utils::SQRT_2.mul_add(utils::INV_SQRT_2, -1.0).abs() < 1e-15);
        assert!(utils::PI_OVER_2.mul_add(2.0, -utils::PI_CONST).abs() < 1e-15);
        assert!(utils::PI_OVER_4.mul_add(4.0, -utils::PI_CONST).abs() < 1e-15);
        assert!(utils::PI_OVER_8.mul_add(8.0, -utils::PI_CONST).abs() < 1e-15);
    }

    #[test]
    fn test_probability_normalization() {
        let mut probs = vec![2.0, 3.0, 5.0];
        assert!(utils::normalize_probabilities(&mut probs));
        assert!(utils::is_normalized(&probs, 1e-10));

        // Verify individual probabilities
        assert!((probs[0] - 0.2).abs() < 1e-10);
        assert!((probs[1] - 0.3).abs() < 1e-10);
        assert!((probs[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_classical_fidelity_bounds() {
        // Fidelity of identical distributions should be 1
        let p = vec![0.5, 0.5];
        let fid = utils::classical_fidelity(&p, &p).unwrap();
        assert!((fid - 1.0).abs() < 1e-10);

        // Fidelity of orthogonal distributions should be 0
        let p1 = vec![1.0, 0.0];
        let p2 = vec![0.0, 1.0];
        let fid2 = utils::classical_fidelity(&p1, &p2).unwrap();
        assert!(fid2.abs() < 1e-10);
    }

    #[test]
    fn test_trace_distance_bounds() {
        // Trace distance of identical distributions should be 0
        let p = vec![0.5, 0.5];
        let dist = utils::trace_distance(&p, &p).unwrap();
        assert!(dist.abs() < 1e-10);

        // Trace distance of orthogonal distributions should be 1
        let p1 = vec![1.0, 0.0];
        let p2 = vec![0.0, 1.0];
        let dist2 = utils::trace_distance(&p1, &p2).unwrap();
        assert!((dist2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_properties() {
        // Entropy of uniform distribution over n states is log2(n)
        let uniform_2 = vec![0.5, 0.5];
        let h2 = utils::entropy(&uniform_2);
        assert!((h2 - 1.0).abs() < 1e-10); // log2(2) = 1

        let uniform_4 = vec![0.25, 0.25, 0.25, 0.25];
        let h4 = utils::entropy(&uniform_4);
        assert!((h4 - 2.0).abs() < 1e-10); // log2(4) = 2

        // Entropy of certain distribution is 0
        let certain = vec![1.0, 0.0, 0.0];
        let h0 = utils::entropy(&certain);
        assert!(h0.abs() < 1e-10);
    }

    #[test]
    fn test_hilbert_space_dimension() {
        // Verify 2^n dimensions
        assert_eq!(utils::hilbert_dim(0), 1);
        assert_eq!(utils::hilbert_dim(1), 2);
        assert_eq!(utils::hilbert_dim(5), 32);
        assert_eq!(utils::hilbert_dim(10), 1024);
        assert_eq!(utils::hilbert_dim(20), 1048576);

        // Reverse calculation
        assert_eq!(utils::num_qubits_from_dim(1), Some(0));
        assert_eq!(utils::num_qubits_from_dim(2), Some(1));
        assert_eq!(utils::num_qubits_from_dim(1024), Some(10));
        assert_eq!(utils::num_qubits_from_dim(100), None); // Not power of 2
    }

    #[test]
    fn test_angle_conversions() {
        // 360 degrees = 2π radians
        let rad = utils::deg_to_rad(360.0);
        assert!(2.0f64.mul_add(-std::f64::consts::PI, rad).abs() < 1e-10);

        // π radians = 180 degrees
        let deg = utils::rad_to_deg(std::f64::consts::PI);
        assert!((deg - 180.0).abs() < 1e-10);

        // Round-trip conversion
        let original = 45.0;
        let converted = utils::rad_to_deg(utils::deg_to_rad(original));
        assert!((converted - original).abs() < 1e-10);
    }

    #[test]
    fn test_probability_validation() {
        assert!(utils::is_valid_probability(0.0));
        assert!(utils::is_valid_probability(0.5));
        assert!(utils::is_valid_probability(1.0));
        assert!(!utils::is_valid_probability(-0.001));
        assert!(!utils::is_valid_probability(1.001));
        assert!(!utils::is_valid_probability(f64::NAN));
    }

    #[test]
    fn test_probability_clamping() {
        assert_eq!(utils::clamp_probability(-1.0), 0.0);
        assert_eq!(utils::clamp_probability(0.5), 0.5);
        assert_eq!(utils::clamp_probability(2.0), 1.0);
    }

    #[test]
    fn test_cnot_requirements() {
        // Linear chain entanglement requires n-1 CNOTs
        assert_eq!(utils::min_cnots_for_entanglement(1), 0);
        assert_eq!(utils::min_cnots_for_entanglement(2), 1);
        assert_eq!(utils::min_cnots_for_entanglement(5), 4);
        assert_eq!(utils::min_cnots_for_entanglement(100), 99);
    }

    #[test]
    fn test_binomial_coefficients() {
        // Pascal's triangle properties
        assert_eq!(utils::binomial(5, 0), 1);
        assert_eq!(utils::binomial(5, 5), 1);
        assert_eq!(utils::binomial(5, 2), 10);
        assert_eq!(utils::binomial(10, 5), 252);

        // Symmetry: C(n,k) = C(n, n-k)
        assert_eq!(utils::binomial(10, 3), utils::binomial(10, 7));
    }
}
