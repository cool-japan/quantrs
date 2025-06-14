//! CUDA kernels for GPU-accelerated quantum simulations using SciRS2.
//!
//! This module has been refactored into a modular structure under `crate::cuda`.
//! This file now serves as a compatibility layer.

// Re-export the new CUDA module for backwards compatibility
pub use crate::cuda::*;