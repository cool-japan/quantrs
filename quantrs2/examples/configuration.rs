//! Configuration example for QuantRS2
//!
//! This example demonstrates how to configure QuantRS2 for different use cases.
//!
//! Run with: cargo run --example configuration

use quantrs2::config::*;

fn main() {
    println!("=== QuantRS2 Configuration Example ===\n");

    // 1. Using the global configuration
    println!("1. Global Configuration (Default):");
    let config = Config::global();
    let snapshot = config.snapshot();
    println!("   Threads: {:?}", snapshot.num_threads);
    println!("   Log level: {:?}", snapshot.log_level);
    println!("   Memory limit: {:?}", snapshot.memory_limit_bytes);
    println!("   Default backend: {:?}", snapshot.default_backend);
    println!("   GPU enabled: {}", snapshot.enable_gpu);
    println!("   SIMD enabled: {}", snapshot.enable_simd);
    println!("   Telemetry: {}", snapshot.enable_telemetry);
    println!();

    // 2. Modifying global configuration
    println!("2. Modifying Global Configuration:");
    config.set_num_threads(8);
    config.set_log_level(LogLevel::Debug);
    config.set_memory_limit_gb(16);
    config.set_default_backend(DefaultBackend::Cpu);
    println!("   ✓ Set 8 threads");
    println!("   ✓ Set log level to Debug");
    println!("   ✓ Set memory limit to 16 GB");
    println!("   ✓ Set default backend to CPU");
    println!();

    let updated = config.snapshot();
    println!("   Updated configuration:");
    println!("   Threads: {:?}", updated.num_threads);
    println!("   Log level: {:?}", updated.log_level);
    println!(
        "   Memory limit: {} GB",
        updated.memory_limit_bytes.unwrap() / (1024 * 1024 * 1024)
    );
    println!("   Default backend: {:?}", updated.default_backend);
    println!();

    // 3. Using the configuration builder
    println!("3. Configuration Builder Pattern:");
    let custom_config = Config::builder()
        .num_threads(4)
        .log_level(LogLevel::Info)
        .memory_limit_gb(32)
        .default_backend(DefaultBackend::TensorNetwork)
        .enable_gpu(true)
        .enable_simd(true)
        .enable_telemetry(false)
        .cache_dir("/tmp/quantrs2_cache")
        .max_cache_size_mb(512)
        .build();

    println!("   Custom configuration created:");
    println!("   Threads: {:?}", custom_config.num_threads);
    println!("   Log level: {:?}", custom_config.log_level);
    println!(
        "   Memory limit: {} GB",
        custom_config.memory_limit_bytes.unwrap() / (1024 * 1024 * 1024)
    );
    println!("   Default backend: {:?}", custom_config.default_backend);
    println!("   GPU: {}", custom_config.enable_gpu);
    println!("   SIMD: {}", custom_config.enable_simd);
    println!("   Telemetry: {}", custom_config.enable_telemetry);
    println!("   Cache dir: {:?}", custom_config.cache_dir);
    println!(
        "   Max cache size: {} MB",
        custom_config.max_cache_size_bytes.unwrap() / (1024 * 1024)
    );
    println!();

    // 4. Parsing configuration from strings
    println!("4. Parsing from Strings:");

    // Parse log levels
    let log_levels = ["trace", "debug", "info", "warn", "error", "off"];
    println!("   Log levels:");
    for level_str in &log_levels {
        match level_str.parse::<LogLevel>() {
            Ok(level) => println!("     '{level_str}' → {level:?}"),
            Err(e) => println!("     '{level_str}' → Error: {e}"),
        }
    }
    println!();

    // Parse backends
    let backends = [
        "cpu",
        "gpu",
        "tensor_network",
        "tensor-network",
        "stabilizer",
        "auto",
    ];
    println!("   Backends:");
    for backend_str in &backends {
        match backend_str.parse::<DefaultBackend>() {
            Ok(backend) => println!("     '{backend_str}' → {backend:?}"),
            Err(e) => println!("     '{backend_str}' → Error: {e}"),
        }
    }
    println!();

    // 5. Configuration presets
    println!("5. Configuration Presets:");

    // Development preset
    println!("   Development Preset:");
    let dev_config = Config::builder()
        .num_threads(2)
        .log_level(LogLevel::Debug)
        .memory_limit_gb(4)
        .default_backend(DefaultBackend::Cpu)
        .enable_gpu(false)
        .enable_telemetry(false)
        .build();
    println!("     - 2 threads (fast compile, easy debug)");
    println!("     - Debug logging");
    println!("     - 4 GB memory limit");
    println!("     - CPU backend only");
    println!();

    // Production preset
    println!("   Production Preset:");
    let prod_config = Config::builder()
        .log_level(LogLevel::Warn)
        .default_backend(DefaultBackend::Auto)
        .enable_gpu(true)
        .enable_simd(true)
        .enable_telemetry(true)
        .max_cache_size_mb(2048)
        .build();
    println!("     - Auto thread count");
    println!("     - Warning-level logging");
    println!("     - Auto backend selection");
    println!("     - GPU & SIMD enabled");
    println!("     - Telemetry enabled");
    println!("     - 2 GB cache");
    println!();

    // Benchmark preset
    println!("   Benchmark Preset:");
    let bench_config = Config::builder()
        .num_threads(num_cpus::get())
        .log_level(LogLevel::Error)
        .default_backend(DefaultBackend::Cpu)
        .enable_simd(true)
        .enable_telemetry(false)
        .build();
    println!("     - All CPU cores ({})", num_cpus::get());
    println!("     - Error-only logging (minimal overhead)");
    println!("     - CPU backend (deterministic)");
    println!("     - SIMD enabled");
    println!();

    // 6. Environment variable configuration
    println!("6. Environment Variable Configuration:");
    println!("   Set these environment variables to configure QuantRS2:");
    println!("     QUANTRS2_NUM_THREADS=8");
    println!("     QUANTRS2_LOG_LEVEL=info");
    println!("     QUANTRS2_MEMORY_LIMIT_GB=16");
    println!("     QUANTRS2_BACKEND=gpu");
    println!("     QUANTRS2_ENABLE_GPU=true");
    println!("     QUANTRS2_ENABLE_SIMD=true");
    println!("     QUANTRS2_ENABLE_TELEMETRY=false");
    println!("     QUANTRS2_CACHE_DIR=/path/to/cache");
    println!("     QUANTRS2_MAX_CACHE_SIZE_MB=512");
    println!();

    // 7. Configuration best practices
    println!("7. Best Practices:");
    println!("   For CPU-bound workloads:");
    println!("     - Set num_threads to match CPU cores");
    println!("     - Enable SIMD for performance");
    println!("     - Use CPU or Auto backend");
    println!();
    println!("   For memory-limited systems:");
    println!("     - Set explicit memory limit");
    println!("     - Use TensorNetwork for larger circuits");
    println!("     - Monitor with diagnostics");
    println!();
    println!("   For GPU acceleration:");
    println!("     - Enable GPU in configuration");
    println!("     - Use GPU or Auto backend");
    println!("     - Verify GPU is detected via diagnostics");
    println!();

    // Reset to defaults
    config.reset();
    println!("8. Configuration Reset:");
    println!("   ✓ Global configuration reset to defaults");
    println!();

    println!("=== Example Complete ===");
}
