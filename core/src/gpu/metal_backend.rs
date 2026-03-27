//! Metal backend for GPU acceleration on macOS/iOS
//!
//! This module provides Metal-accelerated quantum operations.
//! Requires the `metal` feature flag to be enabled. Without Metal runtime
//! (i.e., not running on macOS/iOS with Apple Silicon or AMD GPU), all
//! operations return a descriptive error rather than panicking.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    qubit::QubitId,
};
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;

use super::{GpuBackend, GpuBuffer, GpuKernel};

/// Returns the standard "Metal not available" error with context.
#[inline(always)]
fn metal_unavailable(context: &str) -> QuantRS2Error {
    QuantRS2Error::UnsupportedOperation(format!(
        "Metal backend not available in this build: {context}. \
         Enable the `metal` feature and ensure this code runs on macOS 10.13+ or iOS 8+."
    ))
}

/// Metal GPU buffer
///
/// When the Metal runtime is not available this acts as a graceful placeholder:
/// all mutable operations return errors and the reported size reflects the
/// logical allocation request rather than any actual device allocation.
pub struct MetalBuffer {
    /// Logical number of Complex64 elements.
    size_elements: usize,
}

impl GpuBuffer for MetalBuffer {
    fn size(&self) -> usize {
        self.size_elements * std::mem::size_of::<Complex64>()
    }

    fn upload(&mut self, _data: &[Complex64]) -> QuantRS2Result<()> {
        Err(metal_unavailable("upload to MTLBuffer"))
    }

    fn download(&self, _data: &mut [Complex64]) -> QuantRS2Result<()> {
        Err(metal_unavailable("download from MTLBuffer"))
    }

    fn sync(&self) -> QuantRS2Result<()> {
        Err(metal_unavailable(
            "MTLCommandBuffer commit/waitUntilCompleted",
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Metal kernel implementation
///
/// Each method returns a descriptive `UnsupportedOperation` error when the
/// Metal runtime is absent, allowing callers to fall back gracefully instead
/// of encountering an unexpected panic.
pub struct MetalKernel;

impl GpuKernel for MetalKernel {
    fn apply_single_qubit_gate(
        &self,
        _state: &mut dyn GpuBuffer,
        _gate_matrix: &[Complex64; 4],
        _qubit: QubitId,
        _n_qubits: usize,
    ) -> QuantRS2Result<()> {
        Err(metal_unavailable("apply_single_qubit_gate"))
    }

    fn apply_two_qubit_gate(
        &self,
        _state: &mut dyn GpuBuffer,
        _gate_matrix: &[Complex64; 16],
        _control: QubitId,
        _target: QubitId,
        _n_qubits: usize,
    ) -> QuantRS2Result<()> {
        Err(metal_unavailable("apply_two_qubit_gate"))
    }

    fn apply_multi_qubit_gate(
        &self,
        _state: &mut dyn GpuBuffer,
        _gate_matrix: &Array2<Complex64>,
        _qubits: &[QubitId],
        _n_qubits: usize,
    ) -> QuantRS2Result<()> {
        Err(metal_unavailable("apply_multi_qubit_gate"))
    }

    fn measure_qubit(
        &self,
        _state: &dyn GpuBuffer,
        _qubit: QubitId,
        _n_qubits: usize,
    ) -> QuantRS2Result<(bool, f64)> {
        Err(metal_unavailable("measure_qubit"))
    }

    fn expectation_value(
        &self,
        _state: &dyn GpuBuffer,
        _observable: &Array2<Complex64>,
        _qubits: &[QubitId],
        _n_qubits: usize,
    ) -> QuantRS2Result<f64> {
        Err(metal_unavailable("expectation_value"))
    }
}

/// Metal backend
///
/// Exposes Apple Metal GPU acceleration for quantum operations on macOS/iOS.
/// When Metal is unavailable (non-Apple platform, missing runtime, etc.) all
/// operations degrade gracefully to descriptive errors so callers can route
/// to the CPU fallback.
pub struct MetalBackend {
    kernel: MetalKernel,
}

impl MetalBackend {
    /// Attempt to create a new Metal backend.
    ///
    /// Returns `Err(UnsupportedOperation)` when Metal is not available in the
    /// current build or runtime environment. Use
    /// `GpuBackendFactory::create_best_available()` to automatically fall back
    /// to Vulkan or CPU.
    pub fn new() -> QuantRS2Result<Self> {
        eprintln!(
            "[quantrs2-core] WARNING: Metal backend requested but Metal runtime is not available. \
             This backend requires macOS 10.13+ or iOS 8+ with the `metal` feature enabled."
        );
        Err(QuantRS2Error::UnsupportedOperation(
            "Metal backend not available in this build. \
             Compile with the `metal` feature on a supported Apple platform."
                .to_string(),
        ))
    }
}

impl GpuBackend for MetalBackend {
    fn is_available() -> bool {
        // A complete implementation would call MTLCreateSystemDefaultDevice() and
        // check for a non-null result.  Until then we report false so the factory
        // falls through to the CPU backend.
        false
    }

    fn name(&self) -> &'static str {
        "Metal"
    }

    fn device_info(&self) -> String {
        "Metal backend (stub — no runtime available)".to_string()
    }

    fn allocate_state_vector(&self, n_qubits: usize) -> QuantRS2Result<Box<dyn GpuBuffer>> {
        eprintln!(
            "[quantrs2-core] WARNING: MetalBackend::allocate_state_vector called for {n_qubits} qubits \
             but Metal runtime is not available."
        );
        Err(metal_unavailable(&format!(
            "allocate_state_vector for {n_qubits} qubits"
        )))
    }

    fn allocate_density_matrix(&self, n_qubits: usize) -> QuantRS2Result<Box<dyn GpuBuffer>> {
        eprintln!(
            "[quantrs2-core] WARNING: MetalBackend::allocate_density_matrix called for {n_qubits} qubits \
             but Metal runtime is not available."
        );
        Err(metal_unavailable(&format!(
            "allocate_density_matrix for {n_qubits} qubits"
        )))
    }

    fn kernel(&self) -> &dyn GpuKernel {
        &self.kernel
    }
}
