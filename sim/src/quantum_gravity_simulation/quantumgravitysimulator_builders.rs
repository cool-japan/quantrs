//! # QuantumGravitySimulator - builders Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::scirs2_integration::SciRS2Backend;
use scirs2_core::random::prelude::*;

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Initialize the simulator with `SciRS2` backend
    #[must_use]
    pub fn with_backend(mut self, backend: SciRS2Backend) -> Self {
        self.backend = Some(backend);
        self
    }
}
