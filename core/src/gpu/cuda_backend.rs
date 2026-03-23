//! CUDA backend for GPU acceleration
//!
//! This module provides CUDA-accelerated quantum operations.
//! Requires the `cuda` feature flag to be enabled. Without the CUDA runtime
//! and compatible NVIDIA hardware, all operations return a descriptive error
//! rather than panicking.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    qubit::QubitId,
};
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;

use super::{GpuBackend, GpuBuffer, GpuKernel};

/// Returns the standard "CUDA not available" error with the given context.
#[inline(always)]
fn cuda_unavailable(context: &str) -> QuantRS2Error {
    QuantRS2Error::UnsupportedOperation(format!(
        "CUDA backend not available in this build: {context}. \
         Enable the `cuda` feature and ensure CUDA runtime is installed."
    ))
}

/// CUDA GPU buffer
///
/// When the CUDA runtime is not available, this acts as a placeholder that
/// returns errors for all operations rather than panicking.
pub struct CudaBuffer {
    /// Allocated size in number of Complex64 elements (0 when unavailable).
    size_elements: usize,
}

impl GpuBuffer for CudaBuffer {
    fn size(&self) -> usize {
        // Return the logical size; actual device allocation is zero when CUDA
        // is not present.
        self.size_elements * std::mem::size_of::<Complex64>()
    }

    fn upload(&mut self, _data: &[Complex64]) -> QuantRS2Result<()> {
        Err(cuda_unavailable("upload to device buffer"))
    }

    fn download(&self, _data: &mut [Complex64]) -> QuantRS2Result<()> {
        Err(cuda_unavailable("download from device buffer"))
    }

    fn sync(&self) -> QuantRS2Result<()> {
        Err(cuda_unavailable("device stream synchronization"))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// CUDA kernel implementation
///
/// All methods return `UnsupportedOperation` errors when CUDA hardware/runtime
/// is unavailable, enabling callers to fall back to the CPU backend gracefully.
pub struct CudaKernel;

impl GpuKernel for CudaKernel {
    fn apply_single_qubit_gate(
        &self,
        _state: &mut dyn GpuBuffer,
        _gate_matrix: &[Complex64; 4],
        _qubit: QubitId,
        _n_qubits: usize,
    ) -> QuantRS2Result<()> {
        Err(cuda_unavailable("apply_single_qubit_gate"))
    }

    fn apply_two_qubit_gate(
        &self,
        _state: &mut dyn GpuBuffer,
        _gate_matrix: &[Complex64; 16],
        _control: QubitId,
        _target: QubitId,
        _n_qubits: usize,
    ) -> QuantRS2Result<()> {
        Err(cuda_unavailable("apply_two_qubit_gate"))
    }

    fn apply_multi_qubit_gate(
        &self,
        _state: &mut dyn GpuBuffer,
        _gate_matrix: &Array2<Complex64>,
        _qubits: &[QubitId],
        _n_qubits: usize,
    ) -> QuantRS2Result<()> {
        Err(cuda_unavailable("apply_multi_qubit_gate"))
    }

    fn measure_qubit(
        &self,
        _state: &dyn GpuBuffer,
        _qubit: QubitId,
        _n_qubits: usize,
    ) -> QuantRS2Result<(bool, f64)> {
        Err(cuda_unavailable("measure_qubit"))
    }

    fn expectation_value(
        &self,
        _state: &dyn GpuBuffer,
        _observable: &Array2<Complex64>,
        _qubits: &[QubitId],
        _n_qubits: usize,
    ) -> QuantRS2Result<f64> {
        Err(cuda_unavailable("expectation_value"))
    }
}

/// CUDA backend
///
/// Exposes CUDA GPU acceleration for quantum operations. Constructing this
/// backend succeeds only when the CUDA feature flag is enabled **and** a
/// compatible device is detected at runtime. All operations degrade
/// gracefully to descriptive errors when CUDA is unavailable.
pub struct CudaBackend {
    kernel: CudaKernel,
}

impl CudaBackend {
    /// Attempt to create a new CUDA backend.
    ///
    /// Returns `Err(UnsupportedOperation)` when CUDA is not available in the
    /// current build or runtime environment. Callers should check
    /// `CudaBackend::is_available()` first, or use
    /// `GpuBackendFactory::create_best_available()` which falls back to CPU.
    pub fn new() -> QuantRS2Result<Self> {
        // Log a warning so users can diagnose missing CUDA support without
        // hunting through a panic stack trace.
        eprintln!(
            "[quantrs2-core] WARNING: CUDA backend requested but no CUDA runtime is available. \
             Falling back is recommended. Enable the `cuda` feature and install CUDA ≥ 11.x."
        );
        Err(QuantRS2Error::UnsupportedOperation(
            "CUDA backend not available in this build. \
             Compile with the `cuda` feature and ensure a CUDA-capable NVIDIA GPU is present."
                .to_string(),
        ))
    }
}

impl GpuBackend for CudaBackend {
    fn is_available() -> bool {
        // The CUDA feature is compiled in, but we have no runtime to query.
        // A full implementation would call `cuInit(0)` / `cudaGetDeviceCount`
        // here.  For now, report unavailable so the factory falls through to
        // Metal or CPU.
        false
    }

    fn name(&self) -> &'static str {
        "CUDA"
    }

    fn device_info(&self) -> String {
        "CUDA backend (stub — no runtime available)".to_string()
    }

    fn allocate_state_vector(&self, n_qubits: usize) -> QuantRS2Result<Box<dyn GpuBuffer>> {
        // Return a typed error instead of panicking so callers can recover.
        eprintln!(
            "[quantrs2-core] WARNING: CudaBackend::allocate_state_vector called for {n_qubits} qubits \
             but CUDA runtime is not available."
        );
        Err(cuda_unavailable(&format!(
            "allocate_state_vector for {n_qubits} qubits"
        )))
    }

    fn allocate_density_matrix(&self, n_qubits: usize) -> QuantRS2Result<Box<dyn GpuBuffer>> {
        eprintln!(
            "[quantrs2-core] WARNING: CudaBackend::allocate_density_matrix called for {n_qubits} qubits \
             but CUDA runtime is not available."
        );
        Err(cuda_unavailable(&format!(
            "allocate_density_matrix for {n_qubits} qubits"
        )))
    }

    fn kernel(&self) -> &dyn GpuKernel {
        &self.kernel
    }
}
