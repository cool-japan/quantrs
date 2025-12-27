//! Integration tests for version compatibility and diagnostics

use quantrs2::{diagnostics, version};

#[test]
fn test_version_info_detailed() {
    let info = version::VersionInfo::current();

    // Test version string generation
    let version_str = info.version_string();
    assert!(version_str.contains("QuantRS2"));
    assert!(version_str.contains(&info.quantrs2));

    // Test detailed version string
    let detailed = info.detailed_version_string();
    assert!(detailed.contains("QuantRS2"));
    assert!(detailed.contains("SciRS2"));
    assert!(detailed.contains("rustc"));
}

#[test]
fn test_version_display() {
    let info = version::VersionInfo::current();
    let display_str = format!("{info}");
    assert!(!display_str.is_empty());
    assert!(display_str.contains("QuantRS2"));
}

#[test]
fn test_compatibility_issue_display() {
    use version::CompatibilityIssue;

    let issue = CompatibilityIssue::RustVersionTooOld {
        current: "1.60.0".to_string(),
        required: "1.86.0".to_string(),
    };
    let display_str = format!("{issue}");
    assert!(display_str.contains("1.60.0"));
    assert!(display_str.contains("1.86.0"));
    assert!(display_str.contains("too old"));

    let issue = CompatibilityIssue::UnsupportedFeatureCombination {
        description: "Test issue".to_string(),
    };
    let display_str = format!("{issue}");
    assert!(display_str.contains("Test issue"));

    let issue = CompatibilityIssue::DependencyVersionMismatch {
        dependency: "scirs2".to_string(),
        expected: "0.1.0-rc.2".to_string(),
        detected: Some("0.1.0-rc.1".to_string()),
    };
    let display_str = format!("{issue}");
    assert!(display_str.contains("scirs2"));
    assert!(display_str.contains("0.1.0-rc.2"));
    assert!(display_str.contains("0.1.0-rc.1"));
}

#[test]
fn test_diagnostic_report() {
    let report = diagnostics::run_diagnostics();

    // Test version information
    assert!(!report.version.quantrs2.is_empty());
    assert!(!report.version.scirs2.is_empty());

    // Test capabilities
    assert!(report.capabilities.cpu_cores > 0);

    // Test configuration
    assert_eq!(report.config.log_level, quantrs2::config::LogLevel::Warn);

    // Test issue filtering
    let errors = report.errors();
    let warnings = report.warnings();

    // Count issues by severity
    let error_count = errors.len();
    let warning_count = warnings.len();

    // Test summary
    let summary = report.summary();
    assert!(summary.contains(&error_count.to_string()));
    assert!(summary.contains(&warning_count.to_string()));
}

#[test]
fn test_diagnostic_report_display() {
    let report = diagnostics::run_diagnostics();
    let display_str = format!("{report}");

    // Check that key sections are present
    assert!(display_str.contains("QuantRS2 Diagnostic Report"));
    assert!(display_str.contains("Version:"));
    assert!(display_str.contains("System Capabilities:"));
    assert!(display_str.contains("Configuration:"));
    assert!(display_str.contains("Diagnostic Summary:"));
}

#[test]
fn test_diagnostic_issue_creation() {
    use diagnostics::{DiagnosticIssue, Severity};

    let issue = DiagnosticIssue::new(Severity::Warning, "Test", "Test message");
    assert_eq!(issue.severity, Severity::Warning);
    assert_eq!(issue.component, "Test");
    assert_eq!(issue.message, "Test message");
    assert!(issue.suggestion.is_none());

    let issue = issue.with_suggestion("Fix it");
    assert_eq!(issue.suggestion, Some("Fix it".to_string()));
}

#[test]
fn test_diagnostic_issue_display() {
    use diagnostics::{DiagnosticIssue, Severity};

    let issue = DiagnosticIssue::new(Severity::Error, "CPU", "Not enough cores")
        .with_suggestion("Use a multi-core CPU");

    let display_str = format!("{issue}");
    assert!(display_str.contains("ERROR"));
    assert!(display_str.contains("CPU"));
    assert!(display_str.contains("Not enough cores"));
    assert!(display_str.contains("Suggestion"));
    assert!(display_str.contains("multi-core"));
}

#[test]
fn test_system_capabilities_detection() {
    let caps = diagnostics::SystemCapabilities::detect();

    // Basic sanity checks
    assert!(caps.cpu_cores > 0);

    // SIMD capabilities depend on the target architecture
    #[cfg(target_feature = "avx2")]
    assert!(caps.has_avx2);

    #[cfg(target_arch = "aarch64")]
    assert!(caps.has_neon);
}

#[test]
fn test_diagnostic_ready_check() {
    let report = diagnostics::run_diagnostics();

    // Test is_ready vs has_errors
    if report.has_errors() {
        assert!(!report.is_ready());
    }

    // Test has_warnings
    if report.has_warnings() {
        assert!(!report.warnings().is_empty());
    }
}

#[test]
fn test_version_comparison() {
    use version::CompatibilityIssue;

    // Test that the current Rust version is acceptable
    let result = version::check_compatibility();

    match result {
        Ok(()) => {
            // All compatibility checks passed
        }
        Err(issues) => {
            // If there are issues, they should be specific
            for issue in &issues {
                match issue {
                    CompatibilityIssue::RustVersionTooOld { current, required } => {
                        eprintln!("Rust version issue: {current} < {required}");
                    }
                    CompatibilityIssue::UnsupportedFeatureCombination { description } => {
                        eprintln!("Feature issue: {description}");
                    }
                    CompatibilityIssue::DependencyVersionMismatch {
                        dependency,
                        expected,
                        detected,
                    } => {
                        eprintln!(
                            "Dependency issue: {dependency} (expected {expected}, detected {detected:?})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_is_ready_function() {
    // Test the standalone is_ready function
    let ready = diagnostics::is_ready();

    // Should match the report's is_ready
    let report = diagnostics::run_diagnostics();
    assert_eq!(ready, report.is_ready());
}

#[test]
fn test_config_in_diagnostics() {
    let report = diagnostics::run_diagnostics();

    // Test that configuration is captured
    assert!(report.config.enable_simd);
    assert!(report.config.enable_gpu);
    assert_eq!(
        report.config.default_backend,
        quantrs2::config::DefaultBackend::Auto
    );
}

#[test]
fn test_version_constants() {
    // Test that version constants are properly set
    assert!(!version::QUANTRS2_VERSION.is_empty());
    assert!(!version::SCIRS2_VERSION.is_empty());
    assert!(!version::RUSTC_VERSION.is_empty());
    assert!(!version::BUILD_TIMESTAMP.is_empty());
    assert!(!version::TARGET_TRIPLE.is_empty());
    assert!(!version::BUILD_PROFILE.is_empty());

    // Test that min versions are set
    assert!(!version::MIN_RUST_VERSION.is_empty());
    assert!(!version::SCIRS2_MIN_VERSION.is_empty());
}

#[test]
fn test_version_alias() {
    // Test that VERSION is an alias for QUANTRS2_VERSION
    assert_eq!(version::VERSION, version::QUANTRS2_VERSION);
}
