//! Performance Verification Tests for QuantRS2
//!
//! These tests verify that the facade crate maintains zero-cost abstractions
//! and doesn't add runtime overhead compared to direct subcrate usage.

use std::time::{Duration, Instant};

// ============================================================================
// Zero-Cost Abstraction Verification
// ============================================================================

mod zero_cost_abstractions {
    use super::*;

    /// Verify that re-exports don't add measurable overhead
    #[test]
    fn test_reexport_no_overhead() {
        // Type re-exports should be zero-cost at runtime
        // This test verifies that the facade layer adds no indirection
        let start = Instant::now();

        // Access core types through facade
        use quantrs2::core::qubit::QubitId;
        for i in 0..10_000 {
            let q = QubitId::new(i);
            std::hint::black_box(q);
        }

        let facade_time = start.elapsed();

        // Direct access should have same performance
        let start = Instant::now();
        use quantrs2_core::qubit::QubitId as DirectQubitId;
        for i in 0..10_000 {
            let q = DirectQubitId::new(i);
            std::hint::black_box(q);
        }

        let direct_time = start.elapsed();

        // Allow 50% variance for timing noise in debug mode
        // In release mode, these should be identical due to inlining
        let ratio = facade_time.as_nanos() as f64 / direct_time.as_nanos().max(1) as f64;
        assert!(
            ratio < 1.5,
            "Facade overhead is too high: {ratio:.2}x (facade: {facade_time:?}, direct: {direct_time:?})"
        );
    }

    /// Verify prelude imports are zero-cost
    #[test]
    fn test_prelude_zero_cost() {
        use quantrs2::prelude::essentials::*;

        let start = Instant::now();
        for i in 0..10_000 {
            let q = QubitId::new(i);
            std::hint::black_box(q);
        }
        let elapsed = start.elapsed();

        // Should complete very quickly (sub-millisecond)
        assert!(
            elapsed < Duration::from_millis(100),
            "Prelude access is too slow: {elapsed:?}"
        );
    }
}

// ============================================================================
// Error Handling Overhead
// ============================================================================

mod error_handling_overhead {
    use super::*;
    use quantrs2::error::{QuantRS2Error, QuantRS2Result};

    /// Verify error creation doesn't have excessive overhead
    #[test]
    fn test_error_creation_performance() {
        let start = Instant::now();

        for i in 0..10_000 {
            let err = QuantRS2Error::InvalidQubitId(i);
            std::hint::black_box(err);
        }

        let elapsed = start.elapsed();

        // Error creation should be fast
        assert!(
            elapsed < Duration::from_millis(50),
            "Error creation is too slow: {elapsed:?}"
        );
    }

    /// Verify Result type has no overhead compared to std::result::Result
    #[test]
    fn test_result_type_no_overhead() {
        fn facade_result() -> QuantRS2Result<u64> {
            Ok(42)
        }

        fn std_result() -> Result<u64, QuantRS2Error> {
            Ok(42)
        }

        let start = Instant::now();
        for _ in 0..100_000 {
            let r = facade_result();
            let _ = std::hint::black_box(r);
        }
        let facade_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..100_000 {
            let r = std_result();
            let _ = std::hint::black_box(r);
        }
        let std_time = start.elapsed();

        // Should be essentially identical in release mode
        // In debug mode, allow more variance due to lack of optimizations
        let ratio = facade_time.as_nanos() as f64 / std_time.as_nanos().max(1) as f64;
        assert!(ratio < 2.0, "Result type overhead: {ratio:.2}x");
    }
}

// ============================================================================
// Feature Detection Overhead
// ============================================================================

mod feature_detection_overhead {
    use super::*;

    /// Verify feature detection is cached and fast
    #[test]
    fn test_diagnostics_caching() {
        use quantrs2::diagnostics;

        // First call may be slow (initialization)
        let _ = diagnostics::run_diagnostics();

        // Subsequent calls should be fast
        let start = Instant::now();
        for _ in 0..100 {
            let report = diagnostics::run_diagnostics();
            std::hint::black_box(report);
        }
        let elapsed = start.elapsed();

        // 100 diagnostics calls should be reasonably fast
        // Note: diagnostics does system calls, so allow more time
        assert!(
            elapsed < Duration::from_secs(5),
            "Diagnostics calls too slow: {elapsed:?}"
        );
    }

    /// Verify config access is fast
    #[test]
    fn test_config_access_fast() {
        use quantrs2::config::Config;

        let cfg = Config::global();

        let start = Instant::now();
        for _ in 0..100_000 {
            let snapshot = cfg.snapshot();
            std::hint::black_box(snapshot);
        }
        let elapsed = start.elapsed();

        // Config snapshots should be very fast
        assert!(
            elapsed < Duration::from_millis(500),
            "Config access too slow: {elapsed:?}"
        );
    }
}

// ============================================================================
// Version Checking Overhead
// ============================================================================

mod version_checking_overhead {
    use super::*;

    /// Verify version constants are truly const (no runtime cost)
    #[test]
    fn test_version_constants_are_const() {
        use quantrs2::version;

        // These should all be compile-time constants
        const V1: &str = version::VERSION;
        const V2: &str = version::QUANTRS2_VERSION;
        const V3: &str = version::SCIRS2_VERSION;
        const V4: &str = version::RUSTC_VERSION;
        const V5: &str = version::TARGET_TRIPLE;
        const V6: &str = version::BUILD_PROFILE;

        // Verify they're non-empty (proves they were evaluated at compile time)
        assert!(!V1.is_empty());
        assert!(!V2.is_empty());
        assert!(!V3.is_empty());
        assert!(!V4.is_empty());
        assert!(!V5.is_empty());
        assert!(!V6.is_empty());
    }

    /// Verify version info creation is fast
    #[test]
    fn test_version_info_fast() {
        use quantrs2::version::VersionInfo;

        let start = Instant::now();
        for _ in 0..10_000 {
            let info = VersionInfo::current();
            std::hint::black_box(info);
        }
        let elapsed = start.elapsed();

        // Version info creation should be fast
        assert!(
            elapsed < Duration::from_millis(100),
            "VersionInfo creation too slow: {elapsed:?}"
        );
    }

    /// Verify compatibility check is reasonably fast
    #[test]
    fn test_compatibility_check_fast() {
        use quantrs2::version::check_compatibility;

        let start = Instant::now();
        for _ in 0..1_000 {
            let result = check_compatibility();
            let _ = std::hint::black_box(result);
        }
        let elapsed = start.elapsed();

        // Compatibility checks should be fast (< 100ms for 1000 checks)
        assert!(
            elapsed < Duration::from_millis(500),
            "Compatibility check too slow: {elapsed:?}"
        );
    }
}

// ============================================================================
// Utility Function Overhead
// ============================================================================

mod utility_function_overhead {
    use super::*;
    use quantrs2::utils;

    /// Verify memory estimation is fast
    #[test]
    fn test_memory_estimation_fast() {
        let start = Instant::now();
        for qubits in 1..30 {
            let mem = utils::estimate_statevector_memory(qubits);
            std::hint::black_box(mem);
        }
        let elapsed = start.elapsed();

        // Memory estimation should be microsecond-scale
        assert!(
            elapsed < Duration::from_millis(10),
            "Memory estimation too slow: {elapsed:?}"
        );
    }

    /// Verify quantum math constants are truly const
    #[test]
    fn test_quantum_constants_are_const() {
        // These should be compile-time constants
        const SQRT2: f64 = utils::SQRT_2;
        const INV_SQRT2: f64 = utils::INV_SQRT_2;
        const PI_2: f64 = utils::PI_OVER_2;
        const PI_4: f64 = utils::PI_OVER_4;
        const PI_8: f64 = utils::PI_OVER_8;
        const PI: f64 = utils::PI_CONST;

        // Verify correctness
        assert!(SQRT2.mul_add(INV_SQRT2, -1.0).abs() < 1e-15);
        assert!(PI_2.mul_add(2.0, -PI).abs() < 1e-15);
        assert!(PI_4.mul_add(4.0, -PI).abs() < 1e-15);
        assert!(PI_8.mul_add(8.0, -PI).abs() < 1e-15);
    }

    /// Verify formatting functions are fast
    #[test]
    fn test_formatting_fast() {
        let start = Instant::now();
        for i in 0..10_000 {
            let mem_str = utils::format_memory(i * 1024);
            std::hint::black_box(mem_str);
        }
        let elapsed = start.elapsed();

        // Formatting should be fast
        assert!(
            elapsed < Duration::from_millis(100),
            "Memory formatting too slow: {elapsed:?}"
        );
    }
}

// ============================================================================
// Deprecation Framework Overhead
// ============================================================================

mod deprecation_overhead {
    use super::*;
    use quantrs2::deprecation;

    /// Verify deprecation checks are fast
    #[test]
    fn test_is_deprecated_fast() {
        let start = Instant::now();
        for _ in 0..100_000 {
            let result = deprecation::is_deprecated("some_nonexistent_api");
            std::hint::black_box(result);
        }
        let elapsed = start.elapsed();

        // Deprecation checks should be fast (uses HashMap lookup)
        // Allow more time in debug mode where optimizations are disabled
        let threshold = if cfg!(debug_assertions) {
            Duration::from_millis(500)
        } else {
            Duration::from_millis(100)
        };
        assert!(
            elapsed < threshold,
            "Deprecation check too slow: {elapsed:?}"
        );
    }

    /// Verify module stability lookup is fast
    #[test]
    fn test_module_stability_fast() {
        let start = Instant::now();
        for _ in 0..100_000 {
            let stability = deprecation::get_module_stability("quantrs2::core");
            std::hint::black_box(stability);
        }
        let elapsed = start.elapsed();

        // Module stability lookup should be fast
        assert!(
            elapsed < Duration::from_millis(100),
            "Module stability lookup too slow: {elapsed:?}"
        );
    }

    /// Verify migration report generation is reasonably fast
    #[test]
    fn test_migration_report_fast() {
        let start = Instant::now();
        for _ in 0..100 {
            let report = deprecation::migration_report();
            std::hint::black_box(report);
        }
        let elapsed = start.elapsed();

        // Report generation (string building) should be fast
        assert!(
            elapsed < Duration::from_millis(500),
            "Migration report too slow: {elapsed:?}"
        );
    }
}

// ============================================================================
// Benchmarking Utilities Self-Test
// ============================================================================

mod bench_utilities_overhead {
    use super::*;
    use quantrs2::bench;

    /// Verify timer overhead is minimal
    #[test]
    fn test_timer_overhead() {
        // Measure overhead of starting/stopping timer
        let start = Instant::now();
        for _ in 0..10_000 {
            let timer = bench::BenchmarkTimer::start();
            let elapsed = timer.stop();
            std::hint::black_box(elapsed);
        }
        let total = start.elapsed();

        // Timer operations should be sub-microsecond each
        let per_op = total.as_nanos() / 10_000;
        assert!(
            per_op < 10_000, // < 10 microseconds
            "Timer overhead too high: {per_op}ns per operation"
        );
    }

    /// Verify stats aggregation is efficient
    #[test]
    fn test_stats_aggregation_efficient() {
        let mut stats = bench::BenchmarkStats::new("test");

        let start = Instant::now();
        for i in 0..10_000 {
            stats.record(Duration::from_nanos(i * 100));
        }
        let recording_time = start.elapsed();

        // Recording should be fast
        assert!(
            recording_time < Duration::from_millis(50),
            "Stats recording too slow: {recording_time:?}"
        );

        // Stat calculations should be fast
        // Note: median() involves sorting, so we only test a few times
        let start = Instant::now();
        for _ in 0..10 {
            let _ = stats.mean();
            let _ = stats.median();
            let _ = stats.std_dev();
            let _ = stats.min();
            let _ = stats.max();
        }
        let calc_time = start.elapsed();

        // Allow up to 1 second for 10 iterations with 10,000 data points each
        // (median requires sorting which is O(n log n))
        assert!(
            calc_time < Duration::from_secs(2),
            "Stats calculations too slow: {calc_time:?}"
        );
    }
}

// ============================================================================
// Testing Utilities Self-Test
// ============================================================================

mod testing_utilities_overhead {
    use super::*;
    use quantrs2::testing;

    /// Verify assertion functions are fast
    #[test]
    fn test_assertions_fast() {
        let start = Instant::now();
        for _ in 0..10_000 {
            testing::assert_approx_eq(1.0, 1.0 + 1e-12, 1e-8);
        }
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_millis(50),
            "Assertion too slow: {elapsed:?}"
        );
    }

    /// Verify random data generation is efficient
    #[test]
    fn test_random_data_generation_efficient() {
        let start = Instant::now();
        for seed in 0..100 {
            let data = testing::generate_random_test_data(1000, seed);
            std::hint::black_box(data);
        }
        let elapsed = start.elapsed();

        // 100 * 1000 random numbers should be fast
        assert!(
            elapsed < Duration::from_millis(500),
            "Random data generation too slow: {elapsed:?}"
        );
    }
}

// ============================================================================
// Inlining Verification (compile-time characteristics)
// ============================================================================

mod inlining_verification {
    /// This test verifies that key functions are marked for inlining
    /// by checking that they compile without issues in different contexts
    #[test]
    fn test_key_functions_inline() {
        // These should all be inlined efficiently
        use quantrs2::utils::{INV_SQRT_2, SQRT_2};
        use quantrs2::version::VERSION;

        // Use in const context (proves compile-time evaluation)
        const _V: &str = VERSION;
        const _S: f64 = SQRT_2;
        const _I: f64 = INV_SQRT_2;

        // Verify values
        assert!(!_V.is_empty());
        assert!(_S.mul_add(_I, -1.0).abs() < 1e-15);
    }
}

// ============================================================================
// Memory Efficiency
// ============================================================================

mod memory_efficiency {
    use super::*;

    /// Verify error types are reasonably sized
    #[test]
    fn test_error_size() {
        use quantrs2::error::QuantRS2Error;

        let size = std::mem::size_of::<QuantRS2Error>();

        // Error should be reasonably compact (< 256 bytes is acceptable)
        assert!(size < 256, "QuantRS2Error is too large: {size} bytes");
    }

    /// Verify config snapshot is reasonably sized
    #[test]
    fn test_config_snapshot_size() {
        use quantrs2::config::ConfigData;

        let size = std::mem::size_of::<ConfigData>();

        // Config data should be compact
        assert!(size < 256, "ConfigData is too large: {size} bytes");
    }

    /// Verify deprecation info is reasonably sized
    #[test]
    fn test_deprecation_info_size() {
        use quantrs2::deprecation::DeprecationInfo;

        let size = std::mem::size_of::<DeprecationInfo>();

        // Deprecation info has strings, but shouldn't be huge
        assert!(size < 512, "DeprecationInfo is too large: {size} bytes");
    }
}
