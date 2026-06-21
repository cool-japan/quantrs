//! Metal-based GPU acceleration for macOS
//!
//! This module provides GPU acceleration using Apple's Metal API
//! for quantum circuit simulation on macOS devices.
//!
//! **Status:** scaffolding only — the Metal compute backend is not yet
//! implemented. All entry points honestly return errors or report unavailability
//! (via SciRS2 platform detection) rather than fabricating results. A future
//! implementation would use Metal Performance Shaders (MPS) and Metal Compute
//! Shaders, covering:
//! - State vector allocation using Metal buffers
//! - Quantum gate kernels using Metal shaders
//! - Memory management for unified memory architecture
//! - Optimizations for Apple Silicon (M1/M2/M3)

use crate::error::{Result, SimulatorError};
use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::prelude::QubitId;
use std::sync::Arc;

/// Metal-based GPU simulator for macOS
pub struct MetalGpuSimulator {
    /// Number of qubits
    num_qubits: usize,
    /// Metal device handle (placeholder)
    _device: Arc<()>,
}

impl MetalGpuSimulator {
    /// Create a new Metal GPU simulator.
    ///
    /// The Metal compute backend is not yet implemented, so this honestly returns
    /// an error rather than fabricating a non-functional simulator. The error text
    /// reflects whether a Metal-capable platform was actually detected.
    pub fn new(num_qubits: usize) -> Result<Self> {
        let _ = num_qubits;
        if Self::is_available() {
            Err(SimulatorError::GpuError(
                "Metal-capable platform detected, but the QuantRS2 Metal backend is not yet \
                 implemented. Please use CPU simulation on macOS for now."
                    .to_string(),
            ))
        } else {
            Err(SimulatorError::GpuError(
                "Metal GPU is not available on this platform. Please use CPU simulation."
                    .to_string(),
            ))
        }
    }

    /// Simulate a quantum circuit
    pub fn simulate<const N: usize>(&mut self, _circuit: &Circuit<N>) -> Result<()> {
        Err(SimulatorError::GpuError(
            "Metal GPU simulation not yet implemented".to_string(),
        ))
    }

    /// Get available Metal devices.
    ///
    /// No Metal backend is implemented yet, so no Metal devices are enumerated
    /// even on macOS — reported honestly rather than fabricating device names.
    pub fn available_devices() -> Vec<String> {
        Vec::new()
    }

    /// Check if Metal acceleration is available on this system.
    ///
    /// Delegates to SciRS2's platform detection (`metal_available`), which is true
    /// only on macOS with the Metal backend compiled in. The QuantRS2 Metal
    /// *compute* path is still unimplemented, so [`Self::new`] refuses even when
    /// this returns `true` — it never fabricates results.
    pub fn is_available() -> bool {
        scirs2_core::simd_ops::PlatformCapabilities::detect().metal_available
    }
}

/// Metal GPU backend interface (placeholder for future implementation)
pub trait MetalBackend {
    /// Allocate state vector on GPU
    fn allocate_state_vector(&self, size: usize) -> Result<()>;

    /// Apply quantum gate
    fn apply_gate(&self, gate: &str, qubits: &[QubitId]) -> Result<()>;

    /// Transfer data between CPU and GPU
    fn sync(&self) -> Result<()>;
}
