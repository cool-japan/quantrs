//! Platform detection implementation

use super::capabilities::*;
use std::env;

/// Detect comprehensive platform capabilities
pub fn detect_platform_capabilities() -> PlatformCapabilities {
    // Consult SciRS2's platform detector. It exposes the acceleration-backend
    // view (GPU/CUDA/OpenCL/Metal flags) reflecting how the SciRS2 stack was
    // built, plus a compile-time SIMD summary (AVX2/AVX512/NEON). We fold those
    // SIMD signals into our own *runtime* probing below as a build-time fallback —
    // runtime `is_x86_feature_detected!` is strictly more precise, so it wins when
    // both are available.
    let scirs2_caps = scirs2_core::simd_ops::PlatformCapabilities::detect();

    PlatformCapabilities {
        cpu: detect_cpu_capabilities(&scirs2_caps),
        gpu: detect_gpu_capabilities(),
        memory: detect_memory_capabilities(),
        platform_type: detect_platform_type(),
        os: detect_operating_system(),
        architecture: detect_architecture(),
    }
}

/// Detect CPU capabilities
fn detect_cpu_capabilities(
    scirs2_caps: &scirs2_core::simd_ops::PlatformCapabilities,
) -> CpuCapabilities {
    let logical_cores = num_cpus::get();
    let physical_cores = num_cpus::get_physical();

    CpuCapabilities {
        physical_cores,
        logical_cores,
        simd: detect_simd_capabilities(scirs2_caps),
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

/// Detect SIMD capabilities.
///
/// CPU feature flags are probed at *runtime* via `is_x86_feature_detected!`
/// (x86_64) / target-feature cfgs (aarch64), which reflects the actual host the
/// binary is executing on. SciRS2's compile-time SIMD summary (`scirs2_caps`) is
/// OR-ed in as a fallback so features baked in at build time are never lost on
/// targets where runtime probing is unavailable.
fn detect_simd_capabilities(
    scirs2_caps: &scirs2_core::simd_ops::PlatformCapabilities,
) -> SimdCapabilities {
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
            avx2: is_x86_feature_detected!("avx2") || scirs2_caps.avx2_available,
            // Runtime AVX-512 probing (more precise than the previous compile-time
            // `cfg!(target_feature)`), reconciled with SciRS2's build-time view.
            avx512: is_x86_feature_detected!("avx512f") || scirs2_caps.avx512_available,
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
            neon: cfg!(target_feature = "neon") || scirs2_caps.neon_available,
            sve: cfg!(target_feature = "sve"),
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Unknown architecture: no runtime probing available, so fall back
        // entirely to SciRS2's compile-time SIMD summary.
        SimdCapabilities {
            sse: false,
            sse2: false,
            sse3: false,
            ssse3: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: scirs2_caps.avx2_available,
            avx512: scirs2_caps.avx512_available,
            fma: false,
            neon: scirs2_caps.neon_available,
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

/// Detect GPU capabilities.
///
/// With the `gpu` feature enabled, this performs a *real* probe via the OxiCUDA
/// driver (which loads `libcuda.so`/`nvcuda.dll` at runtime): each CUDA device
/// is enumerated and its genuine name, memory, SM count, max-threads, warp size,
/// and compute capability are reported. Without the `gpu` feature, or when no
/// GPU/driver is present, it honestly reports no GPU (`available: false`) — it
/// never fabricates a device.
fn detect_gpu_capabilities() -> GpuCapabilities {
    #[cfg(feature = "gpu")]
    {
        if let Some(devices) = detect_cuda_gpu_devices() {
            if !devices.is_empty() {
                return GpuCapabilities {
                    available: true,
                    devices,
                    primary_device: Some(0),
                };
            }
        }
    }

    // Honest fallback: no GPU detected (or GPU support not compiled in).
    GpuCapabilities {
        available: false,
        devices: Vec::new(),
        primary_device: None,
    }
}

/// Enumerate real CUDA devices via OxiCUDA and map them to [`GpuDevice`].
///
/// Returns `None` when the driver cannot be initialized (no GPU / no driver) and
/// `Some(vec)` otherwise. All fields are genuine driver queries; fields the
/// driver does not expose (e.g. exact CUDA-core count) are left as `None`.
#[cfg(feature = "gpu")]
fn detect_cuda_gpu_devices() -> Option<Vec<GpuDevice>> {
    oxicuda::init().ok()?;
    let count = oxicuda::Device::count().ok()?;
    if count <= 0 {
        return None;
    }

    let mut devices = Vec::with_capacity(count as usize);
    for ordinal in 0..count {
        let Ok(device) = oxicuda::Device::get(ordinal) else {
            continue;
        };
        let Ok(info) = device.info() else {
            continue;
        };
        let (cc_major, cc_minor) = info.compute_capability;
        devices.push(GpuDevice {
            name: info.name,
            vendor: "NVIDIA".to_string(),
            device_type: if device.is_integrated().unwrap_or(false) {
                GpuType::Integrated
            } else {
                GpuType::Discrete
            },
            memory_bytes: info.total_memory_bytes,
            compute_units: info.multiprocessor_count.max(0) as usize,
            max_workgroup_size: info.max_threads_per_block.max(0) as usize,
            // The driver does not directly report a CUDA-core count; leave None
            // rather than fabricating one from the SM count.
            cuda_cores: None,
            compute_capability: Some((cc_major.max(0) as u32, cc_minor.max(0) as u32)),
        });
    }

    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

/// Detect memory capabilities
fn detect_memory_capabilities() -> MemoryCapabilities {
    use sysinfo::System;

    // `System::new()`, not `new_all()`: only memory is read here, and `new_all` additionally
    // enumerates every process on the host.
    let mut sys = System::new();
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

        // Neither /sys nor numactl was readable: honest single-node fallback
        // (NOT MEASURED). The /sys path above is the real measurement.
        1
    }

    #[cfg(target_os = "macos")]
    {
        // macOS typically doesn't expose NUMA topology on consumer hardware
        // Server-grade Mac Pros might have NUMA, but it's not common
        1
    }

    #[cfg(target_os = "windows")]
    {
        // Windows NUMA topology is not measured here (it would require the
        // `GetNumaHighestNodeNumber` Win32 call via unsafe FFI). We return the
        // honest single-node default rather than an invented value; most
        // desktop/laptop systems do have exactly 1 NUMA node. NOT MEASURED.
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
    // `System::new()`, not `new_all()`: only memory is read here, and `new_all` additionally
    // enumerates every process on the host.
    let mut sys = System::new();
    sys.refresh_memory();
    let total_memory_gb = sys.total_memory() / (1024 * 1024 * 1024);

    // Server heuristics:
    // - High core count (>16 logical cores)
    // - Large memory (>64 GB)
    // - NUMA nodes > 1
    // - Specific CPU model indicators
    // The model string is read once; each `detect_cpu_model()` call refreshes every CPU.
    let cpu_model = detect_cpu_model();
    let is_server = logical_cores > 16
        || total_memory_gb > 64
        || detect_numa_nodes() > 1
        || cpu_model.contains("Xeon")
        || cpu_model.contains("EPYC")
        || cpu_model.contains("Threadripper");

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_detection_is_real_on_linux() {
        // On Linux the count comes from /sys; it must be at least 1 and match a
        // direct read of the node directories when available.
        let n = detect_numa_nodes();
        assert!(n >= 1, "NUMA node count must be >= 1");

        #[cfg(target_os = "linux")]
        {
            if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") {
                let direct = entries
                    .filter_map(Result::ok)
                    .filter(|e| {
                        let name = e.file_name();
                        let name = name.to_string_lossy();
                        name.starts_with("node")
                            && name["node".len()..].chars().all(|c| c.is_ascii_digit())
                            && name.len() > "node".len()
                    })
                    .count();
                if direct > 0 {
                    assert_eq!(n, direct, "NUMA count must equal the real /sys node count");
                }
            }
        }
    }

    #[test]
    fn test_gpu_detection_consistency() {
        // The detected GPU capabilities must be internally consistent and must
        // reflect a real probe (no fabricated devices).
        let caps = detect_gpu_capabilities();

        // `available` implies at least one real device with sane fields.
        assert_eq!(caps.available, !caps.devices.is_empty());
        if caps.available {
            assert!(caps.primary_device.is_some());
            for dev in &caps.devices {
                assert!(!dev.name.is_empty(), "real device must have a name");
                // Real compute capability is never the fabricated (7,5) constant
                // unless the hardware genuinely is 7.5 — but it must be a real
                // Some(..) probe, not a hardcoded None-vs-constant guess.
                if let Some((maj, _min)) = dev.compute_capability {
                    assert!(maj >= 1, "real CC major must be >= 1");
                }
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            // Without the gpu feature, detection must honestly report no GPU.
            assert!(!caps.available);
            assert!(caps.devices.is_empty());
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_detection_matches_oxicuda_probe() {
        // detect_gpu_capabilities() must agree with a direct OxiCUDA probe.
        let caps = detect_gpu_capabilities();
        let truth = oxicuda::init().is_ok() && oxicuda::Device::count().unwrap_or(0) > 0;
        assert_eq!(
            caps.available, truth,
            "platform GPU detection must match the real OxiCUDA probe"
        );
    }
}
