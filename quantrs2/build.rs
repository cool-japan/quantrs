use std::env;

fn main() {
    // Set build timestamp
    let timestamp = chrono::Utc::now().to_rfc3339();
    println!("cargo:rustc-env=VERGEN_BUILD_TIMESTAMP={timestamp}");

    // Get git information if available
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
    {
        if output.status.success() {
            let git_hash = String::from_utf8_lossy(&output.stdout);
            println!("cargo:rustc-env=VERGEN_GIT_SHA={}", git_hash.trim());
        }
    }

    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
    {
        if output.status.success() {
            let git_branch = String::from_utf8_lossy(&output.stdout);
            println!("cargo:rustc-env=VERGEN_GIT_BRANCH={}", git_branch.trim());
        }
    }

    // Set Rust compiler version
    let rustc_version = rustc_version::version().unwrap();
    println!("cargo:rustc-env=VERGEN_RUSTC_SEMVER={rustc_version}");

    // Set target triple
    let target = env::var("TARGET").unwrap();
    println!("cargo:rustc-env=VERGEN_CARGO_TARGET_TRIPLE={target}");

    // Set build profile
    let profile = env::var("PROFILE").unwrap();
    println!("cargo:rustc-env=VERGEN_CARGO_PROFILE={profile}");

    // Rerun if git HEAD changes
    println!("cargo:rerun-if-changed=.git/HEAD");

    // Build-time feature validation
    validate_features();

    // SciRS2 dependency version validation
    validate_scirs2_dependencies();
}

/// Validate SciRS2 dependency versions
fn validate_scirs2_dependencies() {
    // Expected SciRS2 version from workspace
    const EXPECTED_SCIRS2_VERSION: &str = "0.1.0-rc.4";

    // This is a compile-time check that the expected version is documented
    println!("cargo:rustc-env=EXPECTED_SCIRS2_VERSION={EXPECTED_SCIRS2_VERSION}");

    // Validate that we're not using deprecated dependencies
    let deprecated_deps = [
        "num-complex",
        "ndarray",
        "rand",
        "rand_distr",
        "num-traits",
        "num-integer",
        "rayon",
    ];

    // Note: This is a build-time reminder; actual enforcement is in Cargo.toml
    println!("cargo:warning=QuantRS2 SciRS2 Policy: Using scirs2-core v{EXPECTED_SCIRS2_VERSION}");
    println!(
        "cargo:warning=Deprecated dependencies to avoid: {}",
        deprecated_deps.join(", ")
    );
}

/// Validate feature flag combinations at build time
fn validate_features() {
    let mut warnings: Vec<&str> = Vec::new();
    let mut errors: Vec<&str> = Vec::new();
    let mut info: Vec<String> = Vec::new();

    // Count enabled features
    let mut feature_count = 0;
    #[cfg(feature = "circuit")]
    {
        feature_count += 1;
    }
    #[cfg(feature = "sim")]
    {
        feature_count += 1;
    }
    #[cfg(feature = "ml")]
    {
        feature_count += 1;
    }
    #[cfg(feature = "device")]
    {
        feature_count += 1;
    }
    #[cfg(feature = "anneal")]
    {
        feature_count += 1;
    }
    #[cfg(feature = "tytan")]
    {
        feature_count += 1;
    }
    #[cfg(feature = "symengine")]
    {
        feature_count += 1;
    }

    info.push(format!("Enabled features: {feature_count}/7"));

    // Check feature dependencies
    #[cfg(all(feature = "sim", not(feature = "circuit")))]
    {
        errors.push("Feature 'sim' requires 'circuit' to be enabled. Please enable both features or use 'sim' feature which includes 'circuit'.");
    }

    #[cfg(all(feature = "ml", not(feature = "sim")))]
    {
        errors.push("Feature 'ml' requires 'sim' to be enabled. Please enable both features.");
    }

    #[cfg(all(feature = "ml", not(feature = "anneal")))]
    {
        errors.push("Feature 'ml' requires 'anneal' to be enabled. Please enable both features.");
    }

    #[cfg(all(feature = "tytan", not(feature = "anneal")))]
    {
        errors
            .push("Feature 'tytan' requires 'anneal' to be enabled. Please enable both features.");
    }

    #[cfg(all(feature = "device", not(feature = "circuit")))]
    {
        warnings.push("Feature 'device' works best with 'circuit' feature enabled.");
    }

    // Check for optimal feature combinations
    #[cfg(all(feature = "ml", not(feature = "device")))]
    {
        warnings.push("You have enabled 'ml' but not 'device'. Consider enabling 'device' for hardware execution of ML algorithms.");
    }

    #[cfg(all(feature = "anneal", not(feature = "tytan")))]
    {
        info.push(
            "Consider enabling 'tytan' for high-level annealing DSL alongside 'anneal'".to_string(),
        );
    }

    #[cfg(feature = "full")]
    {
        info.push("Full feature set enabled - all QuantRS2 modules available".to_string());
        info.push("Note: This increases compilation time significantly".to_string());
    }

    // Suggest optimal combinations based on enabled features
    if feature_count == 1 {
        #[cfg(feature = "circuit")]
        {
            info.push(
                "Tip: Enable 'sim' feature to add quantum simulation capabilities".to_string(),
            );
        }
        #[cfg(feature = "anneal")]
        {
            info.push("Tip: Enable 'tytan' for high-level annealing interface".to_string());
        }
    }

    // Platform-specific recommendations
    #[cfg(target_os = "macos")]
    {
        info.push("Platform: macOS (Metal GPU support available)".to_string());
    }
    #[cfg(target_os = "linux")]
    {
        info.push("Platform: Linux (CUDA/OpenCL GPU support available)".to_string());
    }
    #[cfg(target_os = "windows")]
    {
        info.push("Platform: Windows (CUDA/DirectX GPU support available)".to_string());
    }

    // Architecture-specific info
    #[cfg(target_arch = "x86_64")]
    {
        info.push("Architecture: x86_64 (AVX2/AVX-512 SIMD available)".to_string());
    }
    #[cfg(target_arch = "aarch64")]
    {
        info.push("Architecture: aarch64 (NEON SIMD available)".to_string());
    }

    // Check for no features enabled
    if feature_count == 0 {
        warnings.push("No optional features enabled. Consider enabling 'circuit' for basic functionality or 'full' for all features.");
    }

    // Print info messages (only when building with verbose output)
    if std::env::var("CARGO_TERM_VERBOSE").is_ok() {
        for msg in &info {
            println!("cargo:warning=QuantRS2 Info: {msg}");
        }
    }

    // Silence unused variable warning when not in verbose mode
    let _ = &info;

    // Print warnings
    for warning in &warnings {
        println!("cargo:warning=QuantRS2 Feature Warning: {warning}");
    }

    // Print errors and panic if any critical issues
    if !errors.is_empty() {
        eprintln!("\n╔════════════════════════════════════════════════════════════╗");
        eprintln!("║  QuantRS2 Build Configuration Errors                       ║");
        eprintln!("╚════════════════════════════════════════════════════════════╝\n");
        for error in &errors {
            eprintln!("  ❌ {error}\n");
        }
        eprintln!("╔════════════════════════════════════════════════════════════╗");
        eprintln!("║  Recommended Feature Configurations:                       ║");
        eprintln!("╠════════════════════════════════════════════════════════════╣");
        eprintln!("║  • Basic quantum circuits:                                 ║");
        eprintln!("║    features = [\"circuit\"]                                   ║");
        eprintln!("║                                                            ║");
        eprintln!("║  • Quantum simulation:                                     ║");
        eprintln!("║    features = [\"sim\"] (automatically includes circuit)     ║");
        eprintln!("║                                                            ║");
        eprintln!("║  • Quantum machine learning:                               ║");
        eprintln!("║    features = [\"ml\"] (includes sim, anneal, circuit)       ║");
        eprintln!("║                                                            ║");
        eprintln!("║  • Full features:                                          ║");
        eprintln!("║    features = [\"full\"]                                      ║");
        eprintln!("╚════════════════════════════════════════════════════════════╝\n");
        panic!("Build failed due to invalid feature configuration");
    }

    // Print success message for valid configurations
    if warnings.is_empty() {
        println!("cargo:warning=✅ QuantRS2 feature configuration validated successfully");
    }
}
