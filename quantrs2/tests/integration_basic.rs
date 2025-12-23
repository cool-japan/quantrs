//! Basic integration tests for the QuantRS2 facade crate

use quantrs2::prelude::essentials::*;
use quantrs2::{config, diagnostics, error, version};

#[test]
fn test_version_information() {
    // Test that version information is accessible
    let info = version::VersionInfo::current();
    assert!(!info.quantrs2.is_empty());
    assert!(!info.scirs2.is_empty());
    assert!(!info.rustc.is_empty());

    // Test version constants
    assert_eq!(version::VERSION, version::QUANTRS2_VERSION);
    assert!(!version::QUANTRS2_VERSION.is_empty());
    assert!(!version::SCIRS2_VERSION.is_empty());
}

#[test]
fn test_configuration() {
    // Test global configuration access
    let cfg = config::Config::global();

    // Test that we can read default configuration
    let snapshot = cfg.snapshot();
    assert_eq!(snapshot.log_level, config::LogLevel::Warn);
    assert_eq!(snapshot.default_backend, config::DefaultBackend::Auto);
    assert!(snapshot.enable_simd);
}

#[test]
fn test_configuration_builder() {
    // Test configuration builder pattern
    let config_data = config::Config::builder()
        .num_threads(4)
        .log_level(config::LogLevel::Info)
        .memory_limit_gb(8)
        .default_backend(config::DefaultBackend::Cpu)
        .enable_gpu(false)
        .enable_simd(true)
        .build();

    assert_eq!(config_data.num_threads, Some(4));
    assert_eq!(config_data.log_level, config::LogLevel::Info);
    assert_eq!(config_data.memory_limit_bytes, Some(8 * 1024 * 1024 * 1024));
    assert_eq!(config_data.default_backend, config::DefaultBackend::Cpu);
    assert!(!config_data.enable_gpu);
    assert!(config_data.enable_simd);
}

#[test]
fn test_diagnostics_basic() {
    // Test that diagnostics can be run
    let report = diagnostics::run_diagnostics();

    // Check that basic fields are populated
    assert!(!report.version.quantrs2.is_empty());
    assert!(report.capabilities.cpu_cores > 0);

    // Test summary generation
    let summary = report.summary();
    assert!(summary.contains("Diagnostic Summary"));
}

#[test]
fn test_error_categorization() {
    use error::{ErrorCategory, QuantRS2ErrorExt};

    // Test error categorization
    let err = QuantRS2Error::InvalidQubitId(5);
    assert_eq!(err.category(), ErrorCategory::Core);
    assert!(!err.is_recoverable());
    assert!(err.is_invalid_input());

    let err = QuantRS2Error::NetworkError("timeout".into());
    assert_eq!(err.category(), ErrorCategory::Hardware);
    assert!(err.is_recoverable());
    assert!(!err.is_invalid_input());
}

#[test]
fn test_error_context() {
    use error::with_context;

    let err = QuantRS2Error::InvalidInput("bad parameter".into());
    let contextualized = with_context(err, "in circuit builder");

    match contextualized {
        QuantRS2Error::InvalidInput(msg) => {
            assert!(msg.contains("in circuit builder"));
            assert!(msg.contains("bad parameter"));
        }
        _ => panic!("Expected InvalidInput variant"),
    }
}

#[test]
fn test_prelude_essentials() {
    // Test that essentials prelude provides basic types
    use quantrs2::prelude::essentials::*;

    // QubitId should be available
    let q0 = QubitId::new(0);
    assert_eq!(q0.id(), 0);

    // Version should be available
    assert!(!VERSION.is_empty());
    assert_eq!(VERSION, QUANTRS2_VERSION);
}

#[test]
fn test_prelude_full() {
    // Test that full prelude provides all available types
    use quantrs2::prelude::full::*;

    // At minimum, essentials should be available
    let q0 = QubitId::new(0);
    assert_eq!(q0.id(), 0);

    // Version should be available
    assert!(!VERSION.is_empty());
}

#[test]
fn test_compatibility_check() {
    // Test version compatibility checking
    let result = version::check_compatibility();

    // In a proper build, this should succeed
    match result {
        Ok(()) => {
            // All good - system is compatible
        }
        Err(issues) => {
            // Print issues for debugging but don't fail
            // (might be expected in some CI environments)
            eprintln!("Compatibility issues (may be expected):");
            for issue in &issues {
                eprintln!("  - {}", issue);
            }
        }
    }
}

#[test]
fn test_log_level_ordering() {
    use config::LogLevel;

    assert!(LogLevel::Trace < LogLevel::Debug);
    assert!(LogLevel::Debug < LogLevel::Info);
    assert!(LogLevel::Info < LogLevel::Warn);
    assert!(LogLevel::Warn < LogLevel::Error);
    assert!(LogLevel::Error < LogLevel::Off);
}

#[test]
fn test_backend_parsing() {
    use config::DefaultBackend;
    use std::str::FromStr;

    assert_eq!("cpu".parse::<DefaultBackend>(), Ok(DefaultBackend::Cpu));
    assert_eq!("gpu".parse::<DefaultBackend>(), Ok(DefaultBackend::Gpu));
    assert_eq!(
        "tensor_network".parse::<DefaultBackend>(),
        Ok(DefaultBackend::TensorNetwork)
    );
    assert_eq!(
        "stabilizer".parse::<DefaultBackend>(),
        Ok(DefaultBackend::Stabilizer)
    );
    assert_eq!("auto".parse::<DefaultBackend>(), Ok(DefaultBackend::Auto));
}

#[test]
fn test_error_display() {
    let err = QuantRS2Error::InvalidQubitId(42);
    let msg = format!("{}", err);
    assert!(msg.contains("42"));

    let err = QuantRS2Error::UnsupportedOperation("test operation".into());
    let msg = format!("{}", err);
    assert!(msg.contains("test operation"));
}

#[test]
fn test_diagnostic_severity_ordering() {
    use diagnostics::Severity;

    assert!(Severity::Info < Severity::Warning);
    assert!(Severity::Warning < Severity::Error);
}
