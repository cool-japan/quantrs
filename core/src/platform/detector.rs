//! Platform detection implementation

use super::capabilities::*;
use std::env;

/// Detect comprehensive platform capabilities
pub fn detect_platform_capabilities() -> PlatformCapabilities {
    // Try to use SciRS2's platform detection if available
    // TODO: Use SciRS2's platform detection when available

    // Fallback to our own detection
    PlatformCapabilities {
        cpu: detect_cpu_capabilities(),
        gpu: detect_gpu_capabilities(),
        memory: detect_memory_capabilities(),
        platform_type: detect_platform_type(),
        os: detect_operating_system(),
        architecture: detect_architecture(),
    }
}

/// Detect CPU capabilities
fn detect_cpu_capabilities() -> CpuCapabilities {
    let logical_cores = num_cpus::get();
    let physical_cores = num_cpus::get_physical();

    CpuCapabilities {
        physical_cores,
        logical_cores,
        simd: detect_simd_capabilities(),
        cache: detect_cache_info(),
        base_clock_mhz: detect_cpu_frequency(),
        vendor: detect_cpu_vendor(),
        model_name: detect_cpu_model(),
    }
}

/// Detect CPU frequency in MHz
fn detect_cpu_frequency() -> Option<f32> {
    use sysinfo::System;

    let mut sys = System::new();
    sys.refresh_cpu_all();

    // Get frequency from first CPU (all cores typically have same base frequency)
    sys.cpus().first().map(|cpu| cpu.frequency() as f32)
}

/// Detect SIMD capabilities
fn detect_simd_capabilities() -> SimdCapabilities {
    // Try to use SciRS2's SIMD detection if available
    // TODO: Use SciRS2's SIMD capability detection when available

    #[cfg(target_arch = "x86_64")]
    {
        SimdCapabilities {
            sse: is_x86_feature_detected!("sse"),
            sse2: is_x86_feature_detected!("sse2"),
            sse3: is_x86_feature_detected!("sse3"),
            ssse3: is_x86_feature_detected!("ssse3"),
            sse4_1: is_x86_feature_detected!("sse4.1"),
            sse4_2: is_x86_feature_detected!("sse4.2"),
            avx: is_x86_feature_detected!("avx"),
            avx2: is_x86_feature_detected!("avx2"),
            avx512: cfg!(target_feature = "avx512f"),
            fma: is_x86_feature_detected!("fma"),
            neon: false,
            sve: false,
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        SimdCapabilities {
            sse: false,
            sse2: false,
            sse3: false,
            ssse3: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: false,
            avx512: false,
            fma: false,
            neon: cfg!(target_feature = "neon"),
            sve: cfg!(target_feature = "sve"),
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SimdCapabilities {
            sse: false,
            sse2: false,
            sse3: false,
            ssse3: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: false,
            avx512: false,
            fma: false,
            neon: false,
            sve: false,
        }
    }
}

/// Detect cache information
const fn detect_cache_info() -> CacheInfo {
    // Basic implementation - can be enhanced with platform-specific detection
    CacheInfo {
        l1_data: Some(32 * 1024),        // 32KB default
        l1_instruction: Some(32 * 1024), // 32KB default
        l2: Some(256 * 1024),            // 256KB default
        l3: Some(8 * 1024 * 1024),       // 8MB default
        line_size: Some(64),             // 64 byte cache line default
    }
}

/// Detect CPU vendor
fn detect_cpu_vendor() -> String {
    use sysinfo::System;

    let mut sys = System::new();
    sys.refresh_cpu_all();

    // Extract vendor from CPU brand string
    if let Some(cpu) = sys.cpus().first() {
        let brand = cpu.brand();
        if brand.contains("Intel") {
            return "Intel".to_string();
        } else if brand.contains("AMD") {
            return "AMD".to_string();
        } else if brand.contains("Apple") {
            return "Apple".to_string();
        } else if brand.contains("ARM") {
            return "ARM".to_string();
        } else if brand.contains("Qualcomm") {
            return "Qualcomm".to_string();
        }
        // Return brand if no known vendor found
        brand.to_string()
    } else {
        "Unknown".to_string()
    }
}

/// Detect CPU model
fn detect_cpu_model() -> String {
    use sysinfo::System;

    let mut sys = System::new();
    sys.refresh_cpu_all();

    // Get CPU brand/model name
    sys.cpus()
        .first()
        .map(|cpu| cpu.brand().to_string())
        .unwrap_or_else(|| "Unknown".to_string())
}

/// Detect GPU capabilities
const fn detect_gpu_capabilities() -> GpuCapabilities {
    // Check for GPU availability
    let devices = Vec::new();

    // Try to detect WebGPU devices (cross-platform)
    // Note: This is a placeholder - actual implementation would use wgpu

    GpuCapabilities {
        available: false,
        devices,
        primary_device: None,
    }
}

/// Detect memory capabilities
fn detect_memory_capabilities() -> MemoryCapabilities {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_memory();

    MemoryCapabilities {
        total_memory: sys.total_memory() as usize,
        available_memory: sys.available_memory() as usize,
        bandwidth_gbps: detect_memory_bandwidth(),
        numa_nodes: detect_numa_nodes(),
        hugepage_support: detect_hugepage_support(),
    }
}

/// Detect memory bandwidth in GB/s
fn detect_memory_bandwidth() -> Option<f32> {
    #[cfg(target_os = "linux")]
    {
        // Try to read DMI information
        if let Ok(output) = std::process::Command::new("dmidecode")
            .args(["-t", "memory"])
            .output()
        {
            if output.status.success() {
                if let Ok(text) = String::from_utf8(output.stdout) {
                    // Look for "Speed:" lines in DMI output
                    for line in text.lines() {
                        if line.contains("Speed:") && line.contains("MT/s") {
                            // Extract speed value
                            if let Some(speed_str) = line.split_whitespace().nth(1) {
                                if let Ok(speed_mts) = speed_str.parse::<f32>() {
                                    // Estimate bandwidth: speed (MT/s) * bus width (8 bytes) / 1000
                                    // This is a rough estimate assuming DDR with 64-bit bus
                                    let bandwidth_gbps = (speed_mts * 8.0) / 1000.0;
                                    return Some(bandwidth_gbps);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Fallback: estimate based on total memory
        // Modern DDR4: ~20-40 GB/s, DDR5: ~40-80 GB/s
        Some(25.0) // Conservative estimate
    }

    #[cfg(target_os = "macos")]
    {
        // macOS: Use sysctl to get memory info
        if let Ok(output) = std::process::Command::new("sysctl")
            .arg("hw.memsize")
            .output()
        {
            if output.status.success() {
                // Estimate based on Apple Silicon vs Intel
                // M1/M2/M3: ~100-400 GB/s unified memory
                // Intel: ~20-40 GB/s
                if std::process::Command::new("sysctl")
                    .arg("machdep.cpu.brand_string")
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|s| s.contains("Apple"))
                    .unwrap_or(false)
                {
                    return Some(200.0); // Apple Silicon estimate
                }
                return Some(30.0); // Intel Mac estimate
            }
        }
        Some(30.0)
    }

    #[cfg(target_os = "windows")]
    {
        // Windows: Rough estimate based on typical RAM speeds
        // DDR4-3200: ~25 GB/s, DDR4-2666: ~21 GB/s
        Some(25.0)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

/// Detect number of NUMA nodes
fn detect_numa_nodes() -> usize {
    #[cfg(target_os = "linux")]
    {
        // Check /sys/devices/system/node/ for node directories
        if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") {
            let node_count = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name().to_string_lossy().starts_with("node") && e.file_name() != "node"
                })
                .count();

            if node_count > 0 {
                return node_count;
            }
        }

        // Fallback: try numactl
        if let Ok(output) = std::process::Command::new("numactl")
            .arg("--hardware")
            .output()
        {
            if output.status.success() {
                if let Ok(text) = String::from_utf8(output.stdout) {
                    // Look for "available: N nodes"
                    for line in text.lines() {
                        if line.contains("available:") && line.contains("nodes") {
                            if let Some(word) = line.split_whitespace().nth(1) {
                                if let Ok(n) = word.parse::<usize>() {
                                    return n;
                                }
                            }
                        }
                    }
                }
            }
        }

        1 // Default to 1 NUMA node
    }

    #[cfg(target_os = "macos")]
    {
        // macOS typically doesn't expose NUMA topology on consumer hardware
        // Server-grade Mac Pros might have NUMA, but it's not common
        1
    }

    #[cfg(target_os = "windows")]
    {
        // Windows: Could use GetNumaHighestNodeNumber, but requires unsafe FFI
        // For now, assume single NUMA node unless on server hardware
        // Most desktop/laptop systems have 1 NUMA node
        1
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        1
    }
}

/// Detect hugepage support
fn detect_hugepage_support() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new("/sys/kernel/mm/hugepages").exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Detect platform type
fn detect_platform_type() -> PlatformType {
    // Check for cloud/container environments
    if env::var("KUBERNETES_SERVICE_HOST").is_ok()
        || env::var("ECS_CONTAINER_METADATA_URI").is_ok()
        || env::var("AWS_EXECUTION_ENV").is_ok()
        || env::var("GOOGLE_CLOUD_PROJECT").is_ok()
        || env::var("AZURE_FUNCTIONS_ENVIRONMENT").is_ok()
    {
        return PlatformType::Cloud;
    }

    // Check for mobile platforms
    if cfg!(target_os = "android") || cfg!(target_os = "ios") {
        return PlatformType::Mobile;
    }

    // Detect server vs desktop based on hardware characteristics
    let logical_cores = num_cpus::get();
    let physical_cores = num_cpus::get_physical();

    use sysinfo::System;
    let mut sys = System::new_all();
    sys.refresh_memory();
    let total_memory_gb = sys.total_memory() / (1024 * 1024 * 1024);

    // Server heuristics:
    // - High core count (>16 logical cores)
    // - Large memory (>64 GB)
    // - NUMA nodes > 1
    // - Specific CPU model indicators
    let is_server = logical_cores > 16
        || total_memory_gb > 64
        || detect_numa_nodes() > 1
        || detect_cpu_model().contains("Xeon")
        || detect_cpu_model().contains("EPYC")
        || detect_cpu_model().contains("Threadripper");

    if is_server {
        PlatformType::Server
    } else if cfg!(any(target_arch = "arm", target_arch = "aarch64")) && !cfg!(target_os = "macos")
    {
        // ARM but not macOS might be embedded
        PlatformType::Embedded
    } else {
        PlatformType::Desktop
    }
}

/// Detect operating system
const fn detect_operating_system() -> OperatingSystem {
    #[cfg(target_os = "linux")]
    {
        OperatingSystem::Linux
    }
    #[cfg(target_os = "windows")]
    {
        OperatingSystem::Windows
    }
    #[cfg(target_os = "macos")]
    {
        OperatingSystem::MacOS
    }
    #[cfg(target_os = "freebsd")]
    {
        OperatingSystem::FreeBSD
    }
    #[cfg(target_os = "android")]
    {
        OperatingSystem::Android
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "windows",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "android"
    )))]
    {
        OperatingSystem::Unknown
    }
}

/// Detect architecture
const fn detect_architecture() -> Architecture {
    #[cfg(target_arch = "x86_64")]
    {
        Architecture::X86_64
    }
    #[cfg(target_arch = "aarch64")]
    {
        Architecture::Aarch64
    }
    #[cfg(target_arch = "riscv64")]
    {
        Architecture::Riscv64
    }
    #[cfg(target_arch = "wasm32")]
    {
        Architecture::Wasm32
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64",
        target_arch = "wasm32"
    )))]
    {
        Architecture::Unknown
    }
}
