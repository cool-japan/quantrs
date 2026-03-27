//! # TopologicalQuantumSimulator - create_color_code_group Methods
//!
//! This module contains method implementations for `TopologicalQuantumSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};

use super::types::SurfaceCode;

use super::topologicalquantumsimulator_type::TopologicalQuantumSimulator;

impl TopologicalQuantumSimulator {
    /// Create color code
    pub(super) fn create_color_code(dimensions: &[usize]) -> Result<SurfaceCode> {
        Self::create_toric_surface_code(dimensions)
    }
}
