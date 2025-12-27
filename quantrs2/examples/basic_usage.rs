//! Basic usage example of the QuantRS2 facade crate
//!
//! This example demonstrates how to use the basic features of QuantRS2
//! including version information, configuration, and diagnostics.
//!
//! Run with: cargo run --example basic_usage

use quantrs2::prelude::essentials::*;
use quantrs2::{config, diagnostics, version};

fn main() {
    println!("=== QuantRS2 Basic Usage Example ===\n");

    // 1. Display version information
    println!("1. Version Information:");
    println!("   QuantRS2 version: {}", version::VERSION);
    println!("   SciRS2 version: {}", version::SCIRS2_VERSION);
    println!("   Rust compiler: {}", version::RUSTC_VERSION);
    println!("   Build timestamp: {}", version::BUILD_TIMESTAMP);
    println!();

    // 2. Get detailed version info
    println!("2. Detailed Version Info:");
    let version_info = version::VersionInfo::current();
    println!("   {version_info}");
    println!();

    // 3. Configuration management
    println!("3. Global Configuration:");
    let cfg = config::Config::global();
    println!("   CPU cores: {}", num_cpus::get());
    println!(
        "   Configured threads: {}",
        cfg.num_threads()
            .map_or_else(|| "auto".to_string(), |n| n.to_string())
    );
    println!("   Log level: {}", cfg.log_level().as_str());
    println!("   Default backend: {}", cfg.default_backend().as_str());
    println!("   GPU enabled: {}", cfg.is_gpu_enabled());
    println!("   SIMD enabled: {}", cfg.is_simd_enabled());
    println!();

    // 4. Modify configuration
    println!("4. Modifying Configuration:");
    cfg.set_num_threads(4);
    cfg.set_log_level(config::LogLevel::Info);
    println!("   Set threads to: 4");
    println!("   Set log level to: info");
    println!("   Current threads: {:?}", cfg.num_threads());
    println!();

    // 5. Run diagnostics
    println!("5. System Diagnostics:");
    let report = diagnostics::run_diagnostics();
    println!("   {}", report.summary());
    println!();

    if report.is_ready() {
        println!("   ✓ System is READY for quantum simulation!");
    } else {
        println!("   ✗ System has issues - review diagnostics");
        for error in report.errors() {
            println!("     ERROR: {error}");
        }
        for warning in report.warnings() {
            println!("     WARNING: {warning}");
        }
    }
    println!();

    // 6. Version compatibility check
    println!("6. Compatibility Check:");
    match version::check_compatibility() {
        Ok(()) => println!("   ✓ All compatibility checks passed!"),
        Err(issues) => {
            println!("   ✗ Compatibility issues found:");
            for issue in issues {
                println!("     - {issue}");
            }
        }
    }
    println!();

    // 7. Working with QubitId (from essentials prelude)
    println!("7. Basic Quantum Types:");
    let q0 = QubitId::new(0);
    let q1 = QubitId::new(1);
    let q2 = QubitId::new(2);
    println!(
        "   Created qubits: q0={}, q1={}, q2={}",
        q0.id(),
        q1.id(),
        q2.id()
    );
    println!();

    // 8. Error handling example
    println!("8. Error Handling:");
    use quantrs2::error::{ErrorCategory, QuantRS2ErrorExt};

    let err = QuantRS2Error::InvalidQubitId(42);
    println!("   Error: {err}");
    println!("   Category: {:?}", err.category());
    println!("   Is recoverable: {}", err.is_recoverable());
    println!("   User message: {}", err.user_message());
    println!();

    println!("=== Example Complete ===");
}
