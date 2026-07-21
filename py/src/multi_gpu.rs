//! Multi-GPU Support for QuantRS2
//!
//! This module provides comprehensive multi-GPU capabilities for quantum circuit simulation,
//! including automatic device detection, load balancing, and distributed computation.
//!
//! ## Features
//!
//! - **Multi-GPU Detection**: Automatic detection of all available CUDA-capable GPUs
//! - **Load Balancing**: Intelligent distribution of quantum state vectors across GPUs
//! - **Parallel Execution**: Concurrent gate application on multiple GPUs
//! - **Memory Management**: Efficient memory allocation and transfer between host and devices
//! - **Performance Monitoring**: Real-time tracking of GPU utilization and performance metrics
//!
//! ## SciRS2 Policy Compliance
//!
//! All numerical operations use SciRS2-Core abstractions:
//! - Complex numbers: `scirs2_core::Complex64`
//! - Arrays: `scirs2_core::ndarray::*`
//! - Parallel operations: `scirs2_core::parallel_ops`
//! - GPU operations: `quantrs2_core::gpu` abstractions
//!
//! ## Note on GPU API (v0.1.2)
//!
//! This module currently uses CPU fallback implementations with stub GPU detection.
//! Full multi-GPU support will be implemented when the SciRS2 GPU API stabilizes.
//! The current implementation provides the API surface and performance monitoring
//! infrastructure, ready for GPU acceleration in future releases.

// Allow unused_self for PyO3 method bindings, unnecessary_wraps for future error handling,
// and missing_const_for_fn for GPU-related functions that need runtime checks
#![allow(clippy::unused_self)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::missing_const_for_fn)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Errors that can occur during multi-GPU operations
#[derive(Error, Debug)]
pub enum MultiGpuError {
    /// No GPUs available for computation
    #[error("No GPUs available for computation")]
    NoGpusAvailable,

    /// Insufficient GPU memory for the operation
    #[error("Insufficient GPU memory: required {required} bytes, available {available} bytes")]
    InsufficientMemory { required: usize, available: usize },

    /// GPU synchronization error
    #[error("GPU synchronization failed: {message}")]
    SyncError { message: String },

    /// Memory transfer error
    #[error("Memory transfer error: {message}")]
    TransferError { message: String },

    /// Load balancing error
    #[error("Load balancing error: {message}")]
    LoadBalancingError { message: String },

    /// General GPU error
    #[error("GPU error: {message}")]
    GpuError { message: String },
}

impl From<MultiGpuError> for PyErr {
    fn from(err: MultiGpuError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device ID
    pub device_id: i32,
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Compute capability (major.minor)
    pub compute_capability: (i32, i32),
    /// Number of multiprocessors
    pub multiprocessor_count: i32,
    /// Maximum threads per block
    pub max_threads_per_block: i32,
    /// Whether the device is currently available
    pub is_available: bool,
}

impl GpuDeviceInfo {
    /// Create a mock device info for testing when GPU is not available
    pub fn mock(device_id: i32) -> Self {
        Self {
            device_id,
            name: format!("Mock GPU {device_id}"),
            total_memory: 8 * 1024 * 1024 * 1024,     // 8 GB
            available_memory: 6 * 1024 * 1024 * 1024, // 6 GB available
            compute_capability: (7, 5),               // Typical modern GPU
            multiprocessor_count: 40,
            max_threads_per_block: 1024,
            is_available: true,
        }
    }
}

/// Multi-GPU manager for coordinating quantum circuit execution across multiple GPUs
pub struct MultiGpuManager {
    /// Available GPU devices
    devices: Vec<GpuDeviceInfo>,
    /// Current allocation strategy
    allocation_strategy: AllocationStrategy,
    /// Performance metrics
    metrics: Arc<Mutex<PerformanceMetrics>>,
    /// Whether GPU support is actually available
    #[cfg(feature = "gpu")]
    gpu_available: bool,
    #[cfg(not(feature = "gpu"))]
    gpu_available: bool,
}

/// Strategy for allocating work across GPUs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Distribute work evenly across all available GPUs
    RoundRobin,
    /// Allocate based on available memory
    MemoryBased,
    /// Allocate based on GPU performance characteristics
    PerformanceBased,
    /// Use a single GPU (best available)
    SingleGpu,
    /// Custom allocation based on heuristics
    Adaptive,
}

/// Performance metrics for multi-GPU execution
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Total execution time in milliseconds
    pub total_time_ms: f64,
    /// Time spent on each GPU
    pub per_gpu_time_ms: HashMap<i32, f64>,
    /// Memory transferred to GPUs in bytes
    pub memory_transferred_bytes: usize,
    /// Number of gates executed on each GPU
    pub gates_per_gpu: HashMap<i32, usize>,
    /// Average GPU utilization (0.0 to 1.0)
    pub avg_gpu_utilization: f64,
}

impl MultiGpuManager {
    /// Create a new multi-GPU manager. Always succeeds (an empty/no-GPU
    /// manager is a valid manager), but `gpu_available`/`devices` honestly
    /// reflect real detection via [`Self::detect_gpus`] -- this never
    /// fabricates a device that isn't actually there.
    #[allow(clippy::unnecessary_wraps)] // API design: may return errors in future
    pub fn new() -> Result<Self, MultiGpuError> {
        let devices = Self::detect_gpus();
        let gpu_available = !devices.is_empty();
        Ok(Self {
            devices,
            allocation_strategy: AllocationStrategy::SingleGpu,
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            gpu_available,
        })
    }

    /// Detect available GPU-capable backends via `scirs2_core::gpu`'s real
    /// backend-preference/availability probing.
    ///
    /// Without the `gpu` cargo feature (the default), no GPU backend is
    /// compiled into this crate at all, so detection is honestly empty --
    /// this is *not* a stub returning a fixed answer, it is the correct
    /// answer for a build with no GPU support compiled in. With the `gpu`
    /// feature enabled, this probes `scirs2_core::gpu::GpuBackend::preferred()`;
    /// on a machine with no real CUDA/ROCm/Metal/OpenCL/WGPU driver
    /// available, that honestly resolves to `GpuBackend::Cpu` /
    /// `is_available() == false` too, so the result stays an empty `Vec`
    /// until real hardware is detected. Per-device memory/topology fields
    /// are left at `0`/unknown pending `scirs2_core`'s multi-device
    /// enumeration API; only `device_id`, `name`, and `is_available` are
    /// populated for now.
    fn detect_gpus() -> Vec<GpuDeviceInfo> {
        #[cfg(feature = "gpu")]
        {
            let backend = scirs2_core::gpu::GpuBackend::preferred();
            if backend == scirs2_core::gpu::GpuBackend::Cpu || !backend.is_available() {
                return Vec::new();
            }
            return vec![GpuDeviceInfo {
                device_id: 0,
                name: format!("{backend:?}"),
                total_memory: 0,
                available_memory: 0,
                compute_capability: (0, 0),
                multiprocessor_count: 0,
                max_threads_per_block: 0,
                is_available: true,
            }];
        }

        #[cfg(not(feature = "gpu"))]
        {
            Vec::new()
        }
    }

    /// Get the number of available GPUs
    pub fn num_gpus(&self) -> usize {
        self.devices.len()
    }

    /// Get information about all available GPUs
    pub fn get_devices(&self) -> &[GpuDeviceInfo] {
        &self.devices
    }

    /// Set the allocation strategy
    pub const fn set_strategy(&mut self, strategy: AllocationStrategy) {
        self.allocation_strategy = strategy;
    }

    /// Get the current allocation strategy
    pub const fn get_strategy(&self) -> AllocationStrategy {
        self.allocation_strategy
    }

    /// Select GPUs for a quantum circuit simulation based on the allocation strategy
    pub fn select_gpus(&self, n_qubits: usize) -> Result<Vec<i32>, MultiGpuError> {
        if self.devices.is_empty() {
            return Err(MultiGpuError::NoGpusAvailable);
        }

        // Calculate required memory for the quantum state
        let state_size = (1 << n_qubits) * std::mem::size_of::<Complex64>();

        match self.allocation_strategy {
            AllocationStrategy::SingleGpu => {
                // Find the best single GPU
                let best_gpu = self
                    .devices
                    .iter()
                    .filter(|d| d.is_available && d.available_memory >= state_size)
                    .max_by_key(|d| d.available_memory)
                    .ok_or_else(|| MultiGpuError::InsufficientMemory {
                        required: state_size,
                        available: self
                            .devices
                            .iter()
                            .filter(|d| d.is_available)
                            .map(|d| d.available_memory)
                            .max()
                            .unwrap_or(0),
                    })?;

                Ok(vec![best_gpu.device_id])
            }

            AllocationStrategy::RoundRobin => {
                // Use all available GPUs in round-robin fashion
                Ok(self
                    .devices
                    .iter()
                    .filter(|d| d.is_available)
                    .map(|d| d.device_id)
                    .collect())
            }

            AllocationStrategy::MemoryBased => {
                // Select GPUs based on available memory
                let mut selected = Vec::new();
                let mut remaining_memory = state_size;

                for device in self
                    .devices
                    .iter()
                    .filter(|d| d.is_available)
                    .filter(|d| d.available_memory > 0)
                {
                    selected.push(device.device_id);
                    if device.available_memory >= remaining_memory {
                        break;
                    }
                    remaining_memory = remaining_memory.saturating_sub(device.available_memory);
                }

                if remaining_memory > 0 {
                    return Err(MultiGpuError::InsufficientMemory {
                        required: state_size,
                        available: state_size - remaining_memory,
                    });
                }

                Ok(selected)
            }

            AllocationStrategy::PerformanceBased => {
                // Select GPUs based on compute capability and multiprocessor count
                let mut devices: Vec<_> = self
                    .devices
                    .iter()
                    .filter(|d| d.is_available && d.available_memory >= state_size / 2)
                    .collect();

                devices.sort_by(|a, b| {
                    let a_score = a.compute_capability.0 * 100
                        + a.compute_capability.1 * 10
                        + a.multiprocessor_count / 10;
                    let b_score = b.compute_capability.0 * 100
                        + b.compute_capability.1 * 10
                        + b.multiprocessor_count / 10;
                    b_score.cmp(&a_score)
                });

                Ok(devices.iter().map(|d| d.device_id).collect())
            }

            AllocationStrategy::Adaptive => {
                // Use adaptive strategy based on problem size: single best
                // GPU for small problems, memory-based allocation for large ones.
                let mut temp = self.clone();
                let strategy = if n_qubits <= 20 {
                    AllocationStrategy::SingleGpu
                } else {
                    AllocationStrategy::MemoryBased
                };
                temp.set_strategy(strategy);
                temp.select_gpus(n_qubits)
            }
        }
    }

    /// Execute a quantum circuit on multiple GPUs
    pub fn execute_distributed(
        &self,
        state: &Array1<Complex64>,
        n_qubits: usize,
    ) -> Result<Array1<Complex64>, MultiGpuError> {
        #[cfg(feature = "gpu")]
        {
            let gpus = self.select_gpus(n_qubits)?;

            if gpus.is_empty() {
                return Err(MultiGpuError::NoGpusAvailable);
            }

            // For single GPU, just execute directly
            if gpus.len() == 1 {
                return Ok(Self::execute_single_gpu(state, gpus[0]));
            }

            // Multi-GPU execution: partition state across GPUs
            let chunk_size = state.len() / gpus.len();
            let mut results = Vec::new();

            for (i, &gpu_id) in gpus.iter().enumerate() {
                let start = i * chunk_size;
                let end = if i == gpus.len() - 1 {
                    state.len()
                } else {
                    (i + 1) * chunk_size
                };
                let chunk = state.slice(scirs2_core::ndarray::s![start..end]).to_owned();

                let result = Self::execute_single_gpu(&chunk, gpu_id);
                results.push(result);
            }

            // Combine results from all GPUs
            let mut combined = Array1::zeros(state.len());
            let mut offset = 0;
            for result in results {
                combined
                    .slice_mut(scirs2_core::ndarray::s![offset..offset + result.len()])
                    .assign(&result);
                offset += result.len();
            }

            Ok(combined)
        }

        #[cfg(not(feature = "gpu"))]
        {
            // Mock implementation - just return the input state
            Ok(state.clone())
        }
    }

    /// Execute on a single GPU
    ///
    /// v0.1.2 Note: Stub implementation with CPU fallback.
    /// Will use SciRS2 GPU API for actual GPU execution when stable.
    fn execute_single_gpu(state: &Array1<Complex64>, _gpu_id: i32) -> Array1<Complex64> {
        // v0.1.2: CPU fallback implementation
        // Future: Use scirs2_core::gpu for actual GPU execution
        state.clone()
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Reset performance metrics
    pub fn reset_metrics(&self) {
        *self.metrics.lock().unwrap_or_else(|e| e.into_inner()) = PerformanceMetrics::default();
    }

    /// Check if multi-GPU is actually available
    pub fn is_available(&self) -> bool {
        self.gpu_available && !self.devices.is_empty()
    }
}

impl Clone for MultiGpuManager {
    fn clone(&self) -> Self {
        Self {
            devices: self.devices.clone(),
            allocation_strategy: self.allocation_strategy,
            metrics: Arc::new(Mutex::new(
                self.metrics
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone(),
            )),
            gpu_available: self.gpu_available,
        }
    }
}

impl Default for MultiGpuManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback: create a mock manager
            Self {
                devices: vec![],
                allocation_strategy: AllocationStrategy::SingleGpu,
                metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
                gpu_available: false,
            }
        })
    }
}

/// Python wrapper for the multi-GPU manager
#[pyclass]
pub struct PyMultiGpuManager {
    manager: Arc<Mutex<MultiGpuManager>>,
}

#[pymethods]
impl PyMultiGpuManager {
    /// Create a new multi-GPU manager
    ///
    /// Returns:
    ///     PyMultiGpuManager: A new multi-GPU manager instance
    ///
    /// Raises:
    ///     ValueError: If no GPUs are available
    #[new]
    fn new() -> PyResult<Self> {
        let manager = MultiGpuManager::new().map_err(|e| {
            PyValueError::new_err(format!("Failed to create multi-GPU manager: {e}"))
        })?;

        Ok(Self {
            manager: Arc::new(Mutex::new(manager)),
        })
    }

    /// Get the number of available GPUs
    ///
    /// Returns:
    ///     int: Number of available GPUs
    fn num_gpus(&self) -> usize {
        self.manager
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .num_gpus()
    }

    /// Get information about all available GPUs
    ///
    /// Returns:
    ///     list: List of dictionaries containing GPU information
    fn get_devices(&self, py: Python) -> PyResult<Py<PyAny>> {
        let devices = {
            let manager = self.manager.lock().unwrap_or_else(|e| e.into_inner());
            manager.get_devices().to_vec()
        };

        let result = PyList::empty(py);
        for device in devices {
            let dict = PyDict::new(py);
            dict.set_item("device_id", device.device_id)?;
            dict.set_item("name", &device.name)?;
            dict.set_item("total_memory", device.total_memory)?;
            dict.set_item("available_memory", device.available_memory)?;
            dict.set_item(
                "compute_capability",
                format!(
                    "{}.{}",
                    device.compute_capability.0, device.compute_capability.1
                ),
            )?;
            dict.set_item("multiprocessor_count", device.multiprocessor_count)?;
            dict.set_item("max_threads_per_block", device.max_threads_per_block)?;
            dict.set_item("is_available", device.is_available)?;
            result.append(dict)?;
        }

        Ok(result.into())
    }

    /// Set the allocation strategy
    ///
    /// Args:
    ///     strategy (str): Allocation strategy ("round_robin", "memory_based",
    ///                     "performance_based", "single_gpu", "adaptive")
    ///
    /// Raises:
    ///     ValueError: If the strategy is invalid
    fn set_strategy(&self, strategy: &str) -> PyResult<()> {
        let strat = match strategy.to_lowercase().as_str() {
            "round_robin" => AllocationStrategy::RoundRobin,
            "memory_based" => AllocationStrategy::MemoryBased,
            "performance_based" => AllocationStrategy::PerformanceBased,
            "single_gpu" => AllocationStrategy::SingleGpu,
            "adaptive" => AllocationStrategy::Adaptive,
            _ => return Err(PyValueError::new_err(
                format!("Invalid strategy: {strategy}. Valid options: round_robin, memory_based, performance_based, single_gpu, adaptive")
            )),
        };

        self.manager
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_strategy(strat);
        Ok(())
    }

    /// Get the current allocation strategy
    ///
    /// Returns:
    ///     str: Current allocation strategy
    fn get_strategy(&self) -> String {
        let strategy = self
            .manager
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_strategy();
        match strategy {
            AllocationStrategy::RoundRobin => "round_robin",
            AllocationStrategy::MemoryBased => "memory_based",
            AllocationStrategy::PerformanceBased => "performance_based",
            AllocationStrategy::SingleGpu => "single_gpu",
            AllocationStrategy::Adaptive => "adaptive",
        }
        .to_string()
    }

    /// Select GPUs for a quantum circuit simulation
    ///
    /// Args:
    ///     n_qubits (int): Number of qubits in the circuit
    ///
    /// Returns:
    ///     list: List of GPU device IDs to use
    ///
    /// Raises:
    ///     ValueError: If no suitable GPUs are available
    fn select_gpus(&self, n_qubits: usize) -> PyResult<Vec<i32>> {
        self.manager
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .select_gpus(n_qubits)
            .map_err(|e| e.into())
    }

    /// Get performance metrics
    ///
    /// Returns:
    ///     dict: Performance metrics
    fn get_metrics(&self, py: Python) -> PyResult<Py<PyAny>> {
        let metrics = self
            .manager
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_metrics();
        let dict = PyDict::new(py);

        dict.set_item("total_time_ms", metrics.total_time_ms)?;
        dict.set_item("memory_transferred_bytes", metrics.memory_transferred_bytes)?;
        dict.set_item("avg_gpu_utilization", metrics.avg_gpu_utilization)?;

        // Per-GPU times
        let per_gpu_times = PyDict::new(py);
        for (gpu_id, time) in metrics.per_gpu_time_ms {
            per_gpu_times.set_item(gpu_id, time)?;
        }
        dict.set_item("per_gpu_time_ms", per_gpu_times)?;

        // Gates per GPU
        let gates_per_gpu = PyDict::new(py);
        for (gpu_id, gates) in metrics.gates_per_gpu {
            gates_per_gpu.set_item(gpu_id, gates)?;
        }
        dict.set_item("gates_per_gpu", gates_per_gpu)?;

        Ok(dict.into())
    }

    /// Reset performance metrics
    fn reset_metrics(&self) {
        self.manager
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .reset_metrics();
    }

    /// Check if multi-GPU support is available
    ///
    /// Delegates to the same real, instance-level detection check used
    /// everywhere else (`MultiGpuManager::is_available`), rather than
    /// `new().is_ok()` (which was always `true` since construction always
    /// succeeds, regardless of whether any GPU was actually found).
    ///
    /// Returns:
    ///     bool: True if multi-GPU is available, False otherwise
    #[staticmethod]
    fn is_available() -> bool {
        MultiGpuManager::new().is_ok_and(|manager| manager.is_available())
    }

    /// Get a string representation
    fn __repr__(&self) -> String {
        let (num_gpus, strategy, available) = {
            let manager = self.manager.lock().unwrap_or_else(|e| e.into_inner());
            (
                manager.num_gpus(),
                manager.get_strategy(),
                manager.is_available(),
            )
        };
        format!(
            "PyMultiGpuManager(num_gpus={num_gpus}, strategy={strategy:?}, available={available})"
        )
    }
}

/// Register the multi-GPU module with Python
pub fn register_multi_gpu_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiGpuManager>()?;

    // Add module-level functions
    m.add_function(wrap_pyfunction!(get_gpu_count, m)?)?;
    m.add_function(wrap_pyfunction!(is_multi_gpu_available, m)?)?;

    Ok(())
}

/// Get the number of available GPUs
///
/// Delegates to the same real detection as [`PyMultiGpuManager::is_available`]
/// (was previously hardcoded to always return `0`, inconsistent with
/// whatever `is_available()`/`is_multi_gpu_available()` reported).
///
/// Returns:
///     int: Number of available GPUs
#[pyfunction]
fn get_gpu_count() -> usize {
    MultiGpuManager::new().map_or(0, |m| m.num_gpus())
}

/// Check if multi-GPU support is available
///
/// Returns:
///     bool: True if multi-GPU is available, False otherwise
#[pyfunction]
fn is_multi_gpu_available() -> bool {
    PyMultiGpuManager::is_available()
}

// `MultiGpuManager` (unlike `PyMultiGpuManager`) is a plain Rust type -- its
// `new()`/`detect_gpus()` return `Result<_, MultiGpuError>`/`Vec<_>`, not
// `PyResult`/`PyErr`, so these are safe to call directly from a standalone
// test binary (see the note on this crate's test-linking constraint in
// `scirs2_bindings.rs`'s and `parametric.rs`'s `#[cfg(test)]` modules).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_gpus_and_is_available_agree_on_whether_any_device_was_found() {
        let manager = MultiGpuManager::new().expect("construction cannot fail");
        assert_eq!(
            manager.is_available(),
            !manager.get_devices().is_empty(),
            "is_available() must reflect real detection, not a hardcoded answer"
        );
    }

    #[test]
    #[cfg(not(feature = "gpu"))]
    fn without_the_gpu_feature_detection_is_honestly_empty() {
        // No GPU backend is compiled in at all without --features gpu, so
        // there must be no fabricated "Mock GPU" device (the old behavior).
        assert!(MultiGpuManager::detect_gpus().is_empty());
        let manager = MultiGpuManager::new().expect("construction cannot fail");
        assert!(!manager.is_available());
        assert_eq!(manager.num_gpus(), 0);
    }

    #[test]
    fn get_gpu_count_matches_manager_device_count() {
        let manager = MultiGpuManager::new().expect("construction cannot fail");
        assert_eq!(get_gpu_count(), manager.num_gpus());
    }
}
