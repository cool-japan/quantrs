//! System diagnostics example for QuantRS2
//!
//! This example demonstrates how to use the diagnostics module to check
//! system capabilities and compatibility.
//!
//! Run with: cargo run --example diagnostics

use quantrs2::{diagnostics, version};

fn main() {
    println!("=== QuantRS2 System Diagnostics Example ===\n");

    // 1. Run full diagnostics
    println!("1. Running Full System Diagnostics:");
    println!("   Please wait...\n");
    let report = diagnostics::run_diagnostics();

    // 2. Display the full report
    println!("{}", report);
    println!();

    // 3. Check system readiness
    println!("2. System Readiness Check:");
    if report.is_ready() {
        println!("   ✓ System is READY for quantum simulation!");
        println!("   All compatibility checks passed.");
    } else if report.has_errors() {
        println!("   ✗ System has CRITICAL ERRORS!");
        println!("   Please resolve the following issues:");
        for error in report.errors() {
            println!("     - {}", error);
        }
    } else if report.has_warnings() {
        println!("   ⚠ System is functional but has warnings:");
        for warning in report.warnings() {
            println!("     - {}", warning);
        }
        println!("   The system will work, but performance may be suboptimal.");
    }
    println!();

    // 4. System capabilities detail
    println!("3. Detailed System Capabilities:");
    let caps = &report.capabilities;
    println!("   CPU:");
    println!("     - Cores: {}", caps.cpu_cores);
    println!("     - AVX2 support: {}", caps.has_avx2);
    println!("     - AVX-512 support: {}", caps.has_avx512);
    println!("     - ARM NEON support: {}", caps.has_neon);
    println!();
    println!("   Memory:");
    if caps.total_memory_bytes > 0 {
        println!(
            "     - Total: {:.2} GB",
            caps.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!(
            "     - Available: {:.2} GB",
            caps.available_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else {
        println!("     - Memory detection not available");
    }
    println!();
    println!("   GPU:");
    println!("     - GPU available: {}", caps.has_gpu);
    println!();

    // 5. Version compatibility
    println!("4. Version Compatibility:");
    match version::check_compatibility() {
        Ok(()) => {
            println!("   ✓ All version requirements met!");
            println!(
                "   - Rust version: {} (required: {}+)",
                version::RUSTC_VERSION,
                version::MIN_RUST_VERSION
            );
            println!(
                "   - SciRS2 version: {} (required: {}+)",
                version::SCIRS2_VERSION,
                version::SCIRS2_MIN_VERSION
            );
        }
        Err(issues) => {
            println!("   ✗ Compatibility issues detected:");
            for (i, issue) in issues.iter().enumerate() {
                println!("     {}. {}", i + 1, issue);
            }
        }
    }
    println!();

    // 6. Configuration summary
    println!("5. Active Configuration:");
    let config = &report.config;
    println!("   Performance:");
    println!(
        "     - Threads: {}",
        config
            .num_threads
            .map_or_else(|| "auto".to_string(), |n| n.to_string())
    );
    println!("     - SIMD: {}", config.enable_simd);
    println!("     - GPU: {}", config.enable_gpu);
    println!();
    println!("   Backend:");
    println!("     - Default: {:?}", config.default_backend);
    println!();
    println!("   Logging:");
    println!("     - Level: {:?}", config.log_level);
    println!("     - Telemetry: {}", config.enable_telemetry);
    println!();
    if let Some(limit) = config.memory_limit_bytes {
        println!("   Memory:");
        println!(
            "     - Limit: {:.2} GB",
            limit as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!();
    }
    if let Some(ref dir) = config.cache_dir {
        println!("   Cache:");
        println!("     - Directory: {}", dir);
        if let Some(size) = config.max_cache_size_bytes {
            println!("     - Max size: {:.2} MB", size as f64 / (1024.0 * 1024.0));
        }
        println!();
    }

    // 7. Issue categorization
    if !report.issues.is_empty() {
        println!("6. Issue Analysis:");
        let errors: Vec<_> = report.errors();
        let warnings: Vec<_> = report.warnings();
        let info: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.severity == diagnostics::Severity::Info)
            .collect();

        if !errors.is_empty() {
            println!("   Errors ({}):", errors.len());
            for error in errors {
                println!("     - [{}] {}", error.component, error.message);
                if let Some(ref suggestion) = error.suggestion {
                    println!("       → Suggestion: {}", suggestion);
                }
            }
            println!();
        }

        if !warnings.is_empty() {
            println!("   Warnings ({}):", warnings.len());
            for warning in warnings {
                println!("     - [{}] {}", warning.component, warning.message);
                if let Some(ref suggestion) = warning.suggestion {
                    println!("       → Suggestion: {}", suggestion);
                }
            }
            println!();
        }

        if !info.is_empty() {
            println!("   Information ({}):", info.len());
            for info_item in info {
                println!("     - [{}] {}", info_item.component, info_item.message);
            }
            println!();
        }
    }

    // 8. Recommendations
    println!("7. Recommendations:");
    if caps.cpu_cores >= 4 {
        println!("   ✓ Multi-core CPU detected - parallel execution will be efficient");
    } else {
        println!("   ⚠ Limited CPU cores - consider cloud execution for large workloads");
    }

    if caps.has_avx2 || caps.has_neon {
        println!("   ✓ SIMD support available - vectorized operations enabled");
    } else {
        println!("   ⚠ No advanced SIMD - performance may be limited");
    }

    if caps.has_gpu && config.enable_gpu {
        println!("   ✓ GPU available and enabled - can use GPU acceleration");
    } else if caps.has_gpu && !config.enable_gpu {
        println!("   ℹ GPU available but disabled - enable for better performance");
    } else if !caps.has_gpu && config.enable_gpu {
        println!("   ⚠ GPU enabled but not detected - check GPU drivers");
    }
    println!();

    // 9. Quick readiness check
    println!("8. Quick Check Functions:");
    println!("   is_ready(): {}", diagnostics::is_ready());
    println!();

    println!("=== Diagnostics Complete ===");
}
