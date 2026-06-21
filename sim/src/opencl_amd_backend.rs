//! `OpenCL` Backend for AMD GPU Acceleration
//!
//! This module defines the data model and kernel *source templates* for a
//! quantum circuit simulation backend targeting AMD GPUs via `OpenCL`.
//!
//! HONEST-AVAILABILITY NOTE
//! ------------------------
//! This build does **not** link any `OpenCL` / AMD `ROCm` runtime. There is no
//! way here to enumerate real AMD devices, allocate device memory, compile a
//! kernel, or dispatch GPU work. Consequently the construction / availability
//! entry points (`AMDOpenCLSimulator::new`, `benchmark_amd_opencl_backend`)
//! return an honest [`SimulatorError::UnsupportedOperation`] instead of
//! fabricating device discovery (e.g. a hardcoded "Radeon RX 7900 XTX" with
//! invented compute-unit / memory figures) or fabricating kernel timings.
//!
//! The public type definitions and the `OpenCL` kernel *source strings* are
//! retained: they are plain data / text and are useful to callers that have a
//! real `OpenCL` toolchain available outside this build. They never claim that
//! any silicon compiled or executed them.

use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, SimulatorError};

/// `OpenCL` platform information
#[derive(Debug, Clone)]
pub struct OpenCLPlatform {
    /// Platform ID
    pub platform_id: usize,
    /// Platform name
    pub name: String,
    /// Platform vendor
    pub vendor: String,
    /// Platform version
    pub version: String,
    /// Supported extensions
    pub extensions: Vec<String>,
}

/// `OpenCL` device information
#[derive(Debug, Clone)]
pub struct OpenCLDevice {
    /// Device ID
    pub device_id: usize,
    /// Device name
    pub name: String,
    /// Device vendor
    pub vendor: String,
    /// Device type (GPU, CPU, etc.)
    pub device_type: OpenCLDeviceType,
    /// Compute units
    pub compute_units: u32,
    /// Maximum work group size
    pub max_work_group_size: usize,
    /// Maximum work item dimensions
    pub max_work_item_dimensions: u32,
    /// Maximum work item sizes
    pub max_work_item_sizes: Vec<usize>,
    /// Global memory size
    pub global_memory_size: u64,
    /// Local memory size
    pub local_memory_size: u64,
    /// Maximum constant buffer size
    pub max_constant_buffer_size: u64,
    /// Supports double precision
    pub supports_double: bool,
    /// Device extensions
    pub extensions: Vec<String>,
}

/// `OpenCL` device types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenCLDeviceType {
    GPU,
    CPU,
    Accelerator,
    Custom,
    All,
}

/// `OpenCL` backend configuration
#[derive(Debug, Clone)]
pub struct OpenCLConfig {
    /// Preferred platform vendor
    pub preferred_vendor: Option<String>,
    /// Preferred device type
    pub preferred_device_type: OpenCLDeviceType,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Maximum memory allocation per buffer
    pub max_buffer_size: usize,
    /// Work group size for kernels
    pub work_group_size: usize,
    /// Enable kernel caching
    pub enable_kernel_cache: bool,
    /// `OpenCL` optimization level
    pub optimization_level: OptimizationLevel,
    /// Enable automatic fallback to CPU
    pub enable_cpu_fallback: bool,
}

/// `OpenCL` optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization (-O0)
    None,
    /// Basic optimization (-O1)
    Basic,
    /// Standard optimization (-O2)
    Standard,
    /// Aggressive optimization (-O3)
    Aggressive,
}

impl Default for OpenCLConfig {
    fn default() -> Self {
        Self {
            preferred_vendor: Some("Advanced Micro Devices".to_string()),
            preferred_device_type: OpenCLDeviceType::GPU,
            enable_profiling: true,
            max_buffer_size: 1 << 30, // 1GB
            work_group_size: 256,
            enable_kernel_cache: true,
            optimization_level: OptimizationLevel::Standard,
            enable_cpu_fallback: true,
        }
    }
}

/// `OpenCL` kernel information
#[derive(Debug, Clone)]
pub struct OpenCLKernel {
    /// Kernel name
    pub name: String,
    /// Kernel source code
    pub source: String,
    /// Compilation options
    pub build_options: String,
    /// Local memory usage
    pub local_memory_usage: usize,
    /// Work group size
    pub work_group_size: usize,
}

/// `OpenCL` memory buffer descriptor.
///
/// This is a host-side *description* of a buffer; in this build no device
/// allocation backs it (there is no `OpenCL` runtime).
#[derive(Debug, Clone)]
pub struct OpenCLBuffer {
    /// Buffer ID
    pub buffer_id: usize,
    /// Buffer size in bytes
    pub size: usize,
    /// Memory flags
    pub flags: MemoryFlags,
}

/// `OpenCL` memory flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryFlags {
    ReadWrite,
    ReadOnly,
    WriteOnly,
    UseHostPtr,
    AllocHostPtr,
    CopyHostPtr,
}

/// `OpenCL` performance statistics.
///
/// These counters are honest accounting helpers: every value is whatever the
/// caller measured/recorded. The backend itself does not populate them with
/// fabricated numbers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenCLStats {
    /// Total kernel executions
    pub total_kernel_executions: usize,
    /// Total execution time (ms)
    pub total_execution_time: f64,
    /// Average kernel execution time (ms)
    pub avg_kernel_time: f64,
    /// Memory transfer time (ms)
    pub memory_transfer_time: f64,
    /// Compilation time (ms)
    pub compilation_time: f64,
    /// GPU memory usage (bytes)
    pub gpu_memory_usage: u64,
    /// GPU utilization percentage
    pub gpu_utilization: f64,
    /// Number of state vector operations
    pub state_vector_operations: usize,
    /// Number of gate operations
    pub gate_operations: usize,
    /// Fallback to CPU count
    pub cpu_fallback_count: usize,
}

impl OpenCLStats {
    /// Update statistics after kernel execution
    pub fn update_kernel_execution(&mut self, execution_time: f64) {
        self.total_kernel_executions += 1;
        self.total_execution_time += execution_time;
        self.avg_kernel_time = self.total_execution_time / self.total_kernel_executions as f64;
    }

    /// Calculate performance metrics from recorded counters.
    ///
    /// `gpu_efficiency` is `gpu_utilization / 100.0`, i.e. it is derived
    /// directly from whatever utilization the caller recorded - it is not a
    /// fabricated constant.
    #[must_use]
    pub fn get_performance_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        if self.total_execution_time > 0.0 {
            metrics.insert(
                "kernel_executions_per_second".to_string(),
                self.total_kernel_executions as f64 / (self.total_execution_time / 1000.0),
            );
        }
        if self.memory_transfer_time > 0.0 {
            metrics.insert(
                "memory_bandwidth_gb_s".to_string(),
                self.gpu_memory_usage as f64 / (self.memory_transfer_time / 1000.0) / 1e9,
            );
        }
        metrics.insert("gpu_efficiency".to_string(), self.gpu_utilization / 100.0);
        metrics
    }
}

/// Kernel argument types
#[derive(Debug, Clone)]
pub enum KernelArg {
    Buffer(String),
    ConstantBuffer(String),
    Int(i32),
    Float(f32),
    Double(f64),
    LocalMemory(usize),
}

/// AMD GPU-optimized quantum simulator using `OpenCL`.
///
/// In this build there is no `OpenCL` runtime, so this type cannot be
/// constructed (see [`AMDOpenCLSimulator::new`]). The type and its associated
/// kernel-source templates are retained for callers that link a real `OpenCL`
/// toolchain elsewhere.
pub struct AMDOpenCLSimulator {
    /// Configuration
    config: OpenCLConfig,
    /// Selected device (only set when a real runtime is present)
    device: Option<OpenCLDevice>,
    /// Compiled kernel sources (text templates)
    kernels: HashMap<String, OpenCLKernel>,
    /// Performance statistics
    stats: OpenCLStats,
}

impl AMDOpenCLSimulator {
    /// Create a new AMD `OpenCL` simulator.
    ///
    /// HONEST AVAILABILITY GATE: this build links no `OpenCL`/AMD `ROCm`
    /// runtime, so AMD device discovery and GPU kernel dispatch are impossible.
    /// Constructing a working simulator here would require fabricating device
    /// discovery and kernel timings, so we fail loudly instead.
    pub fn new(_config: OpenCLConfig) -> Result<Self> {
        Err(SimulatorError::UnsupportedOperation(
            "AMD OpenCL backend: no OpenCL runtime available in this build \
             (no OpenCL/ROCm SDK linked); cannot enumerate AMD devices or \
             dispatch GPU kernels"
                .to_string(),
        ))
    }

    /// Get device information (only present with a real runtime).
    #[must_use]
    pub const fn get_device_info(&self) -> Option<&OpenCLDevice> {
        self.device.as_ref()
    }

    /// Get the compiled kernel-source templates.
    #[must_use]
    pub const fn get_kernels(&self) -> &HashMap<String, OpenCLKernel> {
        &self.kernels
    }

    /// Get performance statistics.
    #[must_use]
    pub const fn get_stats(&self) -> &OpenCLStats {
        &self.stats
    }

    /// Get the backend configuration.
    #[must_use]
    pub const fn config(&self) -> &OpenCLConfig {
        &self.config
    }

    /// Build the standard set of quantum `OpenCL` kernel *source templates*.
    ///
    /// Returns the kernels as text only; nothing here compiles or executes on a
    /// device. Exposed as an associated function so the templates remain
    /// reachable/inspectable even though [`Self::new`] cannot succeed in this
    /// build.
    #[must_use]
    pub fn kernel_source_templates(config: &OpenCLConfig) -> HashMap<String, OpenCLKernel> {
        let build_options = Self::build_options_for(config);
        let mut kernels = HashMap::new();
        kernels.insert(
            "single_qubit_gate".to_string(),
            OpenCLKernel {
                name: "single_qubit_gate".to_string(),
                source: SINGLE_QUBIT_KERNEL_SRC.to_string(),
                build_options: build_options.clone(),
                local_memory_usage: 0,
                work_group_size: config.work_group_size,
            },
        );
        kernels.insert(
            "two_qubit_gate".to_string(),
            OpenCLKernel {
                name: "two_qubit_gate".to_string(),
                source: TWO_QUBIT_KERNEL_SRC.to_string(),
                build_options: build_options.clone(),
                local_memory_usage: 128,
                work_group_size: config.work_group_size,
            },
        );
        kernels.insert(
            "state_vector_ops".to_string(),
            OpenCLKernel {
                name: "state_vector_ops".to_string(),
                source: STATE_VECTOR_KERNEL_SRC.to_string(),
                build_options: build_options.clone(),
                local_memory_usage: config.work_group_size * 16,
                work_group_size: config.work_group_size,
            },
        );
        kernels.insert(
            "measurement".to_string(),
            OpenCLKernel {
                name: "measurement".to_string(),
                source: MEASUREMENT_KERNEL_SRC.to_string(),
                build_options: build_options.clone(),
                local_memory_usage: config.work_group_size * 16,
                work_group_size: config.work_group_size,
            },
        );
        kernels.insert(
            "expectation_value".to_string(),
            OpenCLKernel {
                name: "expectation_value".to_string(),
                source: EXPECTATION_KERNEL_SRC.to_string(),
                build_options,
                local_memory_usage: config.work_group_size * 8,
                work_group_size: config.work_group_size,
            },
        );
        kernels
    }

    /// Build `OpenCL` kernel-compilation options for the given configuration.
    #[must_use]
    pub fn build_options_for(config: &OpenCLConfig) -> String {
        let mut options = Vec::new();
        match config.optimization_level {
            OptimizationLevel::None => options.push("-O0"),
            OptimizationLevel::Basic => options.push("-O1"),
            OptimizationLevel::Standard => options.push("-O2"),
            OptimizationLevel::Aggressive => options.push("-O3"),
        }
        options.push("-cl-mad-enable");
        options.push("-cl-fast-relaxed-math");
        options.join(" ")
    }
}

/// `OpenCL` single-qubit-gate kernel source (text template).
const SINGLE_QUBIT_KERNEL_SRC: &str = r"
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    typedef double2 complex_t;

    complex_t complex_mul(complex_t a, complex_t b) {
        return (complex_t)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }

    complex_t complex_add(complex_t a, complex_t b) {
        return (complex_t)(a.x + b.x, a.y + b.y);
    }

    __kernel void single_qubit_gate(
        __global complex_t* state,
        __global const double* gate_matrix,
        const int target_qubit,
        const int num_qubits
    ) {
        const int global_id = get_global_id(0);
        const int total_states = 1 << num_qubits;

        if (global_id >= total_states / 2) return;

        const int target_mask = 1 << target_qubit;
        const int i = global_id;
        const int j = i | target_mask;

        if ((i & target_mask) == 0) {
            complex_t gate_00 = (complex_t)(gate_matrix[0], gate_matrix[1]);
            complex_t gate_01 = (complex_t)(gate_matrix[2], gate_matrix[3]);
            complex_t gate_10 = (complex_t)(gate_matrix[4], gate_matrix[5]);
            complex_t gate_11 = (complex_t)(gate_matrix[6], gate_matrix[7]);

            complex_t state_i = state[i];
            complex_t state_j = state[j];

            state[i] = complex_add(complex_mul(gate_00, state_i), complex_mul(gate_01, state_j));
            state[j] = complex_add(complex_mul(gate_10, state_i), complex_mul(gate_11, state_j));
        }
    }
";

/// `OpenCL` two-qubit-gate kernel source (text template).
const TWO_QUBIT_KERNEL_SRC: &str = r"
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    typedef double2 complex_t;

    complex_t complex_mul(complex_t a, complex_t b) {
        return (complex_t)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }

    complex_t complex_add(complex_t a, complex_t b) {
        return (complex_t)(a.x + b.x, a.y + b.y);
    }

    __kernel void two_qubit_gate(
        __global complex_t* state,
        __global const double* gate_matrix,
        const int control_qubit,
        const int target_qubit,
        const int num_qubits
    ) {
        const int global_id = get_global_id(0);
        const int total_states = 1 << num_qubits;

        if (global_id >= total_states / 4) return;

        const int control_mask = 1 << control_qubit;
        const int target_mask = 1 << target_qubit;
        const int both_mask = control_mask | target_mask;

        int base = global_id;
        if (global_id & (target_mask - 1)) base = (base & ~(target_mask - 1)) << 1 | (base & (target_mask - 1));
        if (base & (control_mask - 1)) base = (base & ~(control_mask - 1)) << 1 | (base & (control_mask - 1));

        int state_00 = base;
        int state_01 = base | target_mask;
        int state_10 = base | control_mask;
        int state_11 = base | both_mask;

        complex_t gate[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                gate[i][j] = (complex_t)(gate_matrix[(i*4+j)*2], gate_matrix[(i*4+j)*2+1]);
            }
        }

        complex_t old_states[4];
        old_states[0] = state[state_00];
        old_states[1] = state[state_01];
        old_states[2] = state[state_10];
        old_states[3] = state[state_11];

        complex_t new_states[4] = {0};
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                new_states[i] = complex_add(new_states[i], complex_mul(gate[i][j], old_states[j]));
            }
        }

        state[state_00] = new_states[0];
        state[state_01] = new_states[1];
        state[state_10] = new_states[2];
        state[state_11] = new_states[3];
    }
";

/// `OpenCL` state-vector-operations kernel source (text template).
const STATE_VECTOR_KERNEL_SRC: &str = r"
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    typedef double2 complex_t;

    __kernel void normalize_state(
        __global complex_t* state,
        const int num_states,
        const double norm_factor
    ) {
        const int global_id = get_global_id(0);
        if (global_id >= num_states) return;
        state[global_id].x *= norm_factor;
        state[global_id].y *= norm_factor;
    }

    __kernel void compute_probabilities(
        __global const complex_t* state,
        __global double* probabilities,
        const int num_states
    ) {
        const int global_id = get_global_id(0);
        if (global_id >= num_states) return;
        complex_t amplitude = state[global_id];
        probabilities[global_id] = amplitude.x * amplitude.x + amplitude.y * amplitude.y;
    }
";

/// `OpenCL` measurement kernel source (text template).
const MEASUREMENT_KERNEL_SRC: &str = r"
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    typedef double2 complex_t;

    __kernel void measure_qubit(
        __global complex_t* state,
        const int target_qubit,
        const int num_qubits,
        const int measurement_result
    ) {
        const int global_id = get_global_id(0);
        const int total_states = 1 << num_qubits;
        if (global_id >= total_states) return;

        const int target_mask = 1 << target_qubit;
        const int qubit_value = (global_id & target_mask) ? 1 : 0;
        if (qubit_value != measurement_result) {
            state[global_id] = (complex_t)(0.0, 0.0);
        }
    }
";

/// `OpenCL` expectation-value kernel source (text template).
const EXPECTATION_KERNEL_SRC: &str = r"
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    typedef double2 complex_t;

    __kernel void expectation_value_pauli(
        __global const complex_t* state,
        __global double* partial_results,
        __local double* local_data,
        const int pauli_string,
        const int num_qubits
    ) {
        const int global_id = get_global_id(0);
        const int local_id = get_local_id(0);
        const int local_size = get_local_size(0);
        const int group_id = get_group_id(0);
        const int total_states = 1 << num_qubits;

        double local_expectation = 0.0;
        if (global_id < total_states) {
            complex_t amplitude = state[global_id];
            double sign = 1.0;
            for (int qubit = 0; qubit < num_qubits; qubit++) {
                int pauli_op = (pauli_string >> (2 * qubit)) & 3;
                int qubit_mask = 1 << qubit;
                if (pauli_op == 3 && (global_id & qubit_mask)) sign *= -1.0;
            }
            local_expectation = sign * (amplitude.x * amplitude.x + amplitude.y * amplitude.y);
        }

        local_data[local_id] = local_expectation;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = local_size / 2; stride > 0; stride /= 2) {
            if (local_id < stride) {
                local_data[local_id] += local_data[local_id + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (local_id == 0) {
            partial_results[group_id] = local_data[0];
        }
    }
";

/// Benchmark the AMD `OpenCL` backend.
///
/// HONEST GATE: with no `OpenCL` runtime there is nothing real to benchmark, so
/// this returns an honest [`SimulatorError::UnsupportedOperation`] rather than
/// fabricating throughput / utilization figures.
pub fn benchmark_amd_opencl_backend() -> Result<HashMap<String, f64>> {
    Err(SimulatorError::UnsupportedOperation(
        "AMD OpenCL backend: no OpenCL runtime available in this build; \
         refusing to report fabricated benchmark numbers"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opencl_simulator_unavailable() {
        // Honest behavior: no OpenCL runtime in this build, so construction fails.
        let config = OpenCLConfig::default();
        let result = AMDOpenCLSimulator::new(config);
        assert!(result.is_err());
        match result {
            Err(SimulatorError::UnsupportedOperation(msg)) => {
                assert!(msg.contains("OpenCL"));
            }
            other => panic!("expected UnsupportedOperation, got {other:?}"),
        }
    }

    #[test]
    fn test_benchmark_unavailable() {
        let result = benchmark_amd_opencl_backend();
        assert!(result.is_err());
    }

    #[test]
    fn test_kernel_source_templates_present() {
        // The kernel source templates are plain text and remain inspectable.
        let config = OpenCLConfig::default();
        let kernels = AMDOpenCLSimulator::kernel_source_templates(&config);
        assert!(kernels.contains_key("single_qubit_gate"));
        assert!(kernels.contains_key("two_qubit_gate"));
        assert!(kernels.contains_key("state_vector_ops"));
        assert!(kernels.contains_key("measurement"));
        assert!(kernels.contains_key("expectation_value"));
        assert!(!kernels["single_qubit_gate"].source.is_empty());
    }

    #[test]
    fn test_build_options() {
        let config = OpenCLConfig {
            optimization_level: OptimizationLevel::Aggressive,
            ..Default::default()
        };
        let build_options = AMDOpenCLSimulator::build_options_for(&config);
        assert!(build_options.contains("-O3"));
        assert!(build_options.contains("-cl-mad-enable"));
        assert!(build_options.contains("-cl-fast-relaxed-math"));
    }

    #[test]
    fn test_stats_update() {
        let mut stats = OpenCLStats::default();
        stats.update_kernel_execution(10.0);
        stats.update_kernel_execution(20.0);
        assert_eq!(stats.total_kernel_executions, 2);
        assert!((stats.total_execution_time - 30.0).abs() < 1e-10);
        assert!((stats.avg_kernel_time - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_performance_metrics_from_recorded_counters() {
        // gpu_efficiency is derived from recorded gpu_utilization, not fabricated.
        let mut stats = OpenCLStats {
            total_kernel_executions: 100,
            total_execution_time: 1000.0,
            gpu_memory_usage: 1_000_000_000,
            memory_transfer_time: 100.0,
            gpu_utilization: 85.0,
            ..Default::default()
        };
        let metrics = stats.get_performance_metrics();
        assert!(metrics.contains_key("kernel_executions_per_second"));
        assert!(metrics.contains_key("memory_bandwidth_gb_s"));
        assert!(metrics.contains_key("gpu_efficiency"));
        assert!((metrics["kernel_executions_per_second"] - 100.0).abs() < 1e-10);
        assert!((metrics["gpu_efficiency"] - 0.85).abs() < 1e-10);

        // With no recorded utilization, efficiency is honestly 0.
        stats.gpu_utilization = 0.0;
        let metrics0 = stats.get_performance_metrics();
        assert!((metrics0["gpu_efficiency"] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_memory_flags_distinct() {
        assert_ne!(MemoryFlags::ReadWrite, MemoryFlags::ReadOnly);
    }

    #[test]
    fn test_buffer_descriptor_fields() {
        let buffer = OpenCLBuffer {
            buffer_id: 0,
            size: 1024,
            flags: MemoryFlags::ReadWrite,
        };
        assert_eq!(buffer.size, 1024);
        assert_eq!(buffer.buffer_id, 0);
    }
}
