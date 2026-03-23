//! # GpuContextWrapper - Trait Implementations
//!
//! This module contains trait implementations for `GpuContextWrapper`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::parallel_ops::*;

use super::types::GpuContextWrapper;

impl std::fmt::Debug for GpuContextWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContextWrapper")
            .field("device_id", &self.device_id)
            .field("memory_available", &self.memory_available)
            .field("compute_capability", &self.compute_capability)
            .finish()
    }
}
