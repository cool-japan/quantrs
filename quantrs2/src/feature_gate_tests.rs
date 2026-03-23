//! Unit tests for feature gates

use super::*;

/// Test that core module is always available
#[test]
fn test_core_always_available() {
    // Core should always be accessible
    let _qubit = core::QubitId::new(0);
    assert_eq!(VERSION, QUANTRS2_VERSION);
}

/// Test circuit feature availability
#[cfg(feature = "circuit")]
#[test]
fn test_circuit_feature_available() {
    #[allow(unused_imports)]
    use crate::circuit;
}

/// Test circuit feature unavailability
#[cfg(not(feature = "circuit"))]
#[test]
fn test_circuit_feature_unavailable() {
    // When circuit feature is disabled, circuit module should not be available
}

/// Test sim feature availability
#[cfg(feature = "sim")]
#[test]
fn test_sim_feature_available() {
    #[allow(unused_imports)]
    use crate::sim;
}

/// Test sim feature dependency on circuit
#[cfg(feature = "sim")]
#[test]
fn test_sim_requires_circuit() {
    #[allow(unused_imports)]
    use crate::circuit;
}

/// Test anneal feature availability
#[cfg(feature = "anneal")]
#[test]
fn test_anneal_feature_available() {
    #[allow(unused_imports)]
    use crate::anneal;
}

/// Test device feature availability
#[cfg(feature = "device")]
#[test]
fn test_device_feature_available() {
    #[allow(unused_imports)]
    use crate::device;
}

/// Test ml feature availability
#[cfg(feature = "ml")]
#[test]
fn test_ml_feature_available() {
    #[allow(unused_imports)]
    use crate::ml;
}

/// Test ml feature dependencies
#[cfg(feature = "ml")]
#[test]
fn test_ml_requires_sim_and_anneal() {
    #[allow(unused_imports)]
    use crate::{anneal, circuit, sim};
}

/// Test tytan feature availability
#[cfg(feature = "tytan")]
#[test]
fn test_tytan_feature_available() {
    #[allow(unused_imports)]
    use crate::tytan;
}

/// Test tytan feature dependency on anneal
#[cfg(feature = "tytan")]
#[test]
fn test_tytan_requires_anneal() {
    #[allow(unused_imports)]
    use crate::{anneal, circuit};
}

/// Test symengine feature availability
#[cfg(feature = "symengine")]
#[test]
fn test_symengine_feature_available() {
    #[allow(unused_imports)]
    use crate::symengine;
}

/// Test full feature set
#[cfg(feature = "full")]
#[test]
fn test_full_feature_enables_all() {
    #[allow(unused_imports)]
    use crate::{anneal, circuit, device, ml, sim, symengine, tytan};
}

/// Test prelude essentials always available
#[test]
fn test_prelude_essentials_available() {
    use crate::prelude::essentials::*;

    assert!(!VERSION.is_empty());
    assert_eq!(VERSION, QUANTRS2_VERSION);
}

/// Test prelude circuits availability
#[cfg(feature = "circuit")]
#[test]
fn test_prelude_circuits_available() {
    use crate::prelude::circuits::*;

    assert!(!VERSION.is_empty());
}

/// Test prelude simulation availability
#[cfg(feature = "sim")]
#[test]
fn test_prelude_simulation_available() {
    use crate::prelude::simulation::*;

    assert!(!VERSION.is_empty());
}

/// Test prelude algorithms availability
#[cfg(feature = "ml")]
#[test]
fn test_prelude_algorithms_available() {
    use crate::prelude::algorithms::*;

    assert!(!VERSION.is_empty());
}

/// Test prelude hardware availability
#[cfg(feature = "device")]
#[test]
fn test_prelude_hardware_available() {
    use crate::prelude::hardware::*;

    assert!(!VERSION.is_empty());
}

/// Test prelude quantum_annealing availability
#[cfg(feature = "anneal")]
#[test]
fn test_prelude_quantum_annealing_available() {
    use crate::prelude::quantum_annealing::*;

    assert!(!VERSION.is_empty());
}

/// Test prelude tytan availability
#[cfg(feature = "tytan")]
#[test]
fn test_prelude_tytan_available() {
    use crate::prelude::tytan::*;

    assert!(!VERSION.is_empty());
}

/// Test prelude full availability
#[cfg(feature = "full")]
#[test]
fn test_prelude_full_available() {
    #[allow(unused_imports)]
    use crate::prelude::full;
}

/// Test facade modules are always available
#[test]
fn test_facade_modules_always_available() {
    use crate::bench;
    use crate::config;
    use crate::deprecation;
    use crate::diagnostics;
    use crate::error;
    use crate::testing;
    use crate::utils;
    use crate::version;

    let _ = error::ErrorCategory::Core;
    let _ = version::VERSION;
    let _ = config::Config::global();
    let _ = diagnostics::is_ready();
    let _ = utils::estimate_statevector_memory(10);
    let _ = testing::test_seed();
    let _timer = bench::BenchmarkTimer::start();
    let _ = deprecation::is_deprecated("test");
}

/// Test default feature set (no features)
#[cfg(all(
    not(feature = "circuit"),
    not(feature = "sim"),
    not(feature = "anneal"),
    not(feature = "device"),
    not(feature = "ml"),
    not(feature = "tytan"),
    not(feature = "symengine"),
    not(feature = "full")
))]
#[test]
fn test_default_feature_set() {
    use crate::core;

    let _qubit = core::QubitId::new(0);
    assert!(!VERSION.is_empty());
}

/// Test version constants are consistent
#[test]
fn test_version_constants_consistent() {
    assert_eq!(VERSION, QUANTRS2_VERSION);
    assert_eq!(VERSION, version::VERSION);
    assert_eq!(VERSION, version::QUANTRS2_VERSION);
}

/// Test error module integration
#[test]
fn test_error_module_integration() {
    use crate::error::*;

    let err = QuantRS2Error::InvalidInput("test".into());
    assert!(err.is_invalid_input());

    let category = err.category();
    assert_eq!(category.name(), "Core");
}

/// Test config module integration
#[test]
fn test_config_module_integration() {
    use crate::config::*;

    let cfg = Config::global();
    let _ = cfg.num_threads();
}

/// Test diagnostics module integration
#[test]
fn test_diagnostics_module_integration() {
    use crate::diagnostics::*;

    let report = run_diagnostics();

    assert!(!report.summary().is_empty());
}

/// Test utils module integration
#[test]
fn test_utils_module_integration() {
    use crate::utils::*;

    let mem = estimate_statevector_memory(10);
    assert!(mem > 0);

    let formatted = format_memory(1024 * 1024);
    assert!(formatted.contains("MB") || formatted.contains("KiB"));
}

/// Test testing module integration
#[test]
fn test_testing_module_integration() {
    use crate::testing::*;

    assert_approx_eq(1.0, 1.0, 1e-10);

    let seed = test_seed();
    assert!(seed > 0);
}

/// Test bench module integration
#[test]
fn test_bench_module_integration() {
    use crate::bench::*;
    use std::time::Duration;

    let timer = BenchmarkTimer::start();
    std::thread::sleep(Duration::from_micros(1));
    let elapsed = timer.stop();
    assert!(elapsed > Duration::ZERO);

    let mut stats = BenchmarkStats::new("test");
    stats.record(Duration::from_millis(10));
    assert_eq!(stats.count(), 1);
}

/// Test deprecation module integration
#[test]
fn test_deprecation_module_integration() {
    use crate::deprecation::*;

    let is_deprecated = is_deprecated("nonexistent");
    assert!(!is_deprecated);

    let stability = get_module_stability("quantrs2::core");
    assert!(stability.is_some());
}

/// Test feature combination: circuit + sim
#[cfg(all(feature = "circuit", feature = "sim"))]
#[test]
fn test_feature_combination_circuit_sim() {
    #[allow(unused_imports)]
    use crate::{circuit, sim};
}

/// Test feature combination: sim + ml
#[cfg(all(feature = "sim", feature = "ml"))]
#[test]
fn test_feature_combination_sim_ml() {
    #[allow(unused_imports)]
    use crate::{anneal, ml, sim};
}

/// Test feature combination: anneal + tytan
#[cfg(all(feature = "anneal", feature = "tytan"))]
#[test]
fn test_feature_combination_anneal_tytan() {
    #[allow(unused_imports)]
    use crate::{anneal, tytan};
}

/// Test feature combination: device + circuit
#[cfg(all(feature = "device", feature = "circuit"))]
#[test]
fn test_feature_combination_device_circuit() {
    #[allow(unused_imports)]
    use crate::{circuit, device};
}
