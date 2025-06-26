//! Platform capabilities structures

use serde::{Deserialize, Serialize};

/// Comprehensive platform capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCapabilities {
    /// CPU capabilities
    pub cpu: CpuCapabilities,
    /// GPU capabilities
    pub gpu: GpuCapabilities,
    /// Memory capabilities
    pub memory: MemoryCapabilities,
    /// Platform type
    pub platform_type: PlatformType,
    /// Operating system
    pub os: OperatingSystem,
    /// Architecture
    pub architecture: Architecture,
}

/// CPU capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuCapabilities {
    /// Number of physical cores
    pub physical_cores: usize,
    /// Number of logical cores
    pub logical_cores: usize,
    /// SIMD capabilities
    pub simd: SimdCapabilities,
    /// Cache sizes
    pub cache: CacheInfo,
    /// Clock speed in MHz
    pub base_clock_mhz: Option<f32>,
    /// CPU vendor
    pub vendor: String,
    /// CPU model name
    pub model_name: String,
}

/// SIMD capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdCapabilities {
    /// SSE support
    pub sse: bool,
    /// SSE2 support
    pub sse2: bool,
    /// SSE3 support
    pub sse3: bool,
    /// SSSE3 support
    pub ssse3: bool,
    /// SSE4.1 support
    pub sse4_1: bool,
    /// SSE4.2 support
    pub sse4_2: bool,
    /// AVX support
    pub avx: bool,
    /// AVX2 support
    pub avx2: bool,
    /// AVX512 support
    pub avx512: bool,
    /// FMA support
    pub fma: bool,
    /// ARM NEON support
    pub neon: bool,
    /// ARM SVE support
    pub sve: bool,
}

/// Cache information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheInfo {
    /// L1 data cache size in bytes
    pub l1_data: Option<usize>,
    /// L1 instruction cache size in bytes
    pub l1_instruction: Option<usize>,
    /// L2 cache size in bytes
    pub l2: Option<usize>,
    /// L3 cache size in bytes
    pub l3: Option<usize>,
    /// Cache line size in bytes
    pub line_size: Option<usize>,
}

/// GPU capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapabilities {
    /// GPU available
    pub available: bool,
    /// GPU devices
    pub devices: Vec<GpuDevice>,
    /// Primary GPU index
    pub primary_device: Option<usize>,
}

/// Individual GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    /// Device name
    pub name: String,
    /// Device vendor
    pub vendor: String,
    /// Device type
    pub device_type: GpuType,
    /// Memory in bytes
    pub memory_bytes: usize,
    /// Compute units
    pub compute_units: usize,
    /// Maximum workgroup size
    pub max_workgroup_size: usize,
    /// CUDA cores (if applicable)
    pub cuda_cores: Option<usize>,
    /// Compute capability (for NVIDIA)
    pub compute_capability: Option<(u32, u32)>,
}

/// GPU type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuType {
    Discrete,
    Integrated,
    Virtual,
    Unknown,
}

/// Memory capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCapabilities {
    /// Total physical memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Memory bandwidth estimate in GB/s
    pub bandwidth_gbps: Option<f32>,
    /// NUMA nodes
    pub numa_nodes: usize,
    /// Hugepage support
    pub hugepage_support: bool,
}

/// Platform type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlatformType {
    Desktop,
    Server,
    Mobile,
    Embedded,
    Cloud,
    Unknown,
}

/// Operating system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatingSystem {
    Linux,
    Windows,
    MacOS,
    FreeBSD,
    Android,
    Unknown,
}

/// Architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Architecture {
    X86_64,
    Aarch64,
    Riscv64,
    Wasm32,
    Unknown,
}

impl PlatformCapabilities {
    /// Detect platform capabilities
    /// 
    /// This is the main entry point for platform detection as required by SciRS2 policy.
    /// All modules should use this instead of direct platform detection.
    pub fn detect() -> Self {
        crate::platform::detector::detect_platform_capabilities()
    }

    /// Check if the platform supports SIMD operations
    pub fn has_simd(&self) -> bool {
        self.cpu.simd.sse2
            || self.cpu.simd.avx
            || self.cpu.simd.avx2
            || self.cpu.simd.neon
            || self.cpu.simd.sve
    }

    /// Get the optimal SIMD width for f64 operations
    pub fn optimal_simd_width_f64(&self) -> usize {
        if self.cpu.simd.avx512 {
            8
        } else if self.cpu.simd.avx || self.cpu.simd.avx2 {
            4
        } else if self.cpu.simd.sse2 || self.cpu.simd.neon {
            2
        } else {
            1
        }
    }

    /// Check if GPU acceleration is available
    pub fn has_gpu(&self) -> bool {
        self.gpu.available && !self.gpu.devices.is_empty()
    }

    /// Get the primary GPU device
    pub fn primary_gpu(&self) -> Option<&GpuDevice> {
        self.gpu
            .primary_device
            .and_then(|idx| self.gpu.devices.get(idx))
    }

    /// Check if the platform is suitable for large-scale quantum simulation
    pub fn is_suitable_for_large_quantum_sim(&self) -> bool {
        self.memory.total_memory >= 16 * 1024 * 1024 * 1024 // At least 16GB RAM
            && self.cpu.logical_cores >= 8
            && self.has_simd()
    }
}