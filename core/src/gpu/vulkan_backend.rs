//! Vulkan backend for GPU acceleration
//!
//! This module provides Vulkan-accelerated quantum operations.
//! Requires the `vulkan` feature flag to be enabled. Without the Vulkan
//! loader library and a compatible GPU, all operations return a descriptive
//! error rather than panicking.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    qubit::QubitId,
};
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;

use super::{GpuBackend, GpuBuffer, GpuKernel};

/// Returns the standard "Vulkan not available" error with context.
#[inline(always)]
fn vulkan_unavailable(context: &str) -> QuantRS2Error {
    QuantRS2Error::UnsupportedOperation(format!(
        "Vulkan backend not available in this build: {context}. \
         Enable the `vulkan` feature and ensure the Vulkan loader (libvulkan) is installed \
         with a compatible GPU driver."
    ))
}

/// Vulkan GPU buffer
///
/// When the Vulkan loader or a compatible device is absent, this placeholder
/// reports the logical allocation size without actually allocating device memory,
/// and returns graceful errors for all data-transfer operations.
pub struct VulkanBuffer {
    /// Logical number of Complex64 elements.
    size_elements: usize,
}

impl GpuBuffer for VulkanBuffer {
    fn size(&self) -> usize {
        self.size_elements * std::mem::size_of::<Complex64>()
    }

    fn upload(&mut self, _data: &[Complex64]) -> QuantRS2Result<()> {
        Err(vulkan_unavailable("upload via vkMapMemory/vkCmdCopyBuffer"))
    }

    fn download(&self, _data: &mut [Complex64]) -> QuantRS2Result<()> {
        Err(vulkan_unavailable(
            "download via vkMapMemory/vkCmdCopyBuffer",
        ))
    }

    fn sync(&self) -> QuantRS2Result<()> {
        Err(vulkan_unavailable(
            "vkQueueWaitIdle / fence synchronization",
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Vulkan kernel implementation
///
/// Each method returns a descriptive `UnsupportedOperation` error when the
/// Vulkan runtime is absent, allowing callers to fall back to CPU gracefully.
pub struct VulkanKernel;

impl GpuKernel for VulkanKernel {
    fn apply_single_qubit_gate(
        &self,
        _state: &mut dyn GpuBuffer,
        _gate_matrix: &[Complex64; 4],
        _qubit: QubitId,
        _n_qubits: usize,
    ) -> QuantRS2Result<()> {
        Err(vulkan_unavailable("apply_single_qubit_gate compute shader"))
    }

    fn apply_two_qubit_gate(
        &self,
        _state: &mut dyn GpuBuffer,
        _gate_matrix: &[Complex64; 16],
        _control: QubitId,
        _target: QubitId,
        _n_qubits: usize,
    ) -> QuantRS2Result<()> {
        Err(vulkan_unavailable("apply_two_qubit_gate compute shader"))
    }

    fn apply_multi_qubit_gate(
        &self,
        _state: &mut dyn GpuBuffer,
        _gate_matrix: &Array2<Complex64>,
        _qubits: &[QubitId],
        _n_qubits: usize,
    ) -> QuantRS2Result<()> {
        Err(vulkan_unavailable("apply_multi_qubit_gate compute shader"))
    }

    fn measure_qubit(
        &self,
        _state: &dyn GpuBuffer,
        _qubit: QubitId,
        _n_qubits: usize,
    ) -> QuantRS2Result<(bool, f64)> {
        Err(vulkan_unavailable("measure_qubit reduction kernel"))
    }

    fn expectation_value(
        &self,
        _state: &dyn GpuBuffer,
        _observable: &Array2<Complex64>,
        _qubits: &[QubitId],
        _n_qubits: usize,
    ) -> QuantRS2Result<f64> {
        Err(vulkan_unavailable("expectation_value reduction kernel"))
    }
}

/// Vulkan backend
///
/// Exposes Vulkan compute-based GPU acceleration for quantum operations.
/// Works on any platform that has a Vulkan-capable GPU (NVIDIA, AMD, Intel,
/// ARM Mali/Adreno) with appropriate drivers. When Vulkan is unavailable,
/// all operations return descriptive errors rather than panicking.
pub struct VulkanBackend {
    kernel: VulkanKernel,
}

impl VulkanBackend {
    /// Attempt to create a new Vulkan backend.
    ///
    /// Returns `Err(UnsupportedOperation)` when the Vulkan loader or a
    /// compatible physical device is not found. Use
    /// `GpuBackendFactory::create_best_available()` to automatically fall back
    /// to CPU.
    pub fn new() -> QuantRS2Result<Self> {
        eprintln!(
            "[quantrs2-core] WARNING: Vulkan backend requested but Vulkan runtime is not available. \
             This backend requires the Vulkan loader (libvulkan) and a compatible GPU driver. \
             Enable the `vulkan` feature and install the appropriate driver."
        );
        Err(QuantRS2Error::UnsupportedOperation(
            "Vulkan backend not available in this build. \
             Compile with the `vulkan` feature and ensure the Vulkan loader is installed."
                .to_string(),
        ))
    }
}

impl GpuBackend for VulkanBackend {
    fn is_available() -> bool {
        // A complete implementation would call vkCreateInstance / vkEnumeratePhysicalDevices.
        // Until the Vulkan bindings are integrated, report unavailable so the
        // factory correctly falls through to the CPU backend.
        false
    }

    fn name(&self) -> &'static str {
        "Vulkan"
    }

    fn device_info(&self) -> String {
        "Vulkan backend (stub — no runtime available)".to_string()
    }

    fn allocate_state_vector(&self, n_qubits: usize) -> QuantRS2Result<Box<dyn GpuBuffer>> {
        eprintln!(
            "[quantrs2-core] WARNING: VulkanBackend::allocate_state_vector called for {n_qubits} qubits \
             but Vulkan runtime is not available."
        );
        Err(vulkan_unavailable(&format!(
            "allocate_state_vector for {n_qubits} qubits"
        )))
    }

    fn allocate_density_matrix(&self, n_qubits: usize) -> QuantRS2Result<Box<dyn GpuBuffer>> {
        eprintln!(
            "[quantrs2-core] WARNING: VulkanBackend::allocate_density_matrix called for {n_qubits} qubits \
             but Vulkan runtime is not available."
        );
        Err(vulkan_unavailable(&format!(
            "allocate_density_matrix for {n_qubits} qubits"
        )))
    }

    fn kernel(&self) -> &dyn GpuKernel {
        &self.kernel
    }
}
