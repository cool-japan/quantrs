//! # QuantumNeRF - quantum_view_encoding_group Methods
//!
//! This module contains method implementations for `QuantumNeRF`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::QuantumEncodingOutput;

use super::quantumnerf_type::QuantumNeRF;

impl QuantumNeRF {
    /// Quantum view encoding
    pub(super) fn quantum_view_encoding(
        &self,
        view_direction: &Array1<f64>,
    ) -> Result<QuantumEncodingOutput> {
        let normalized_view = view_direction / view_direction.dot(view_direction).sqrt();
        if self.quantum_view_encoder.quantum_spherical_harmonics {
            self.quantum_spherical_harmonics_encoding(&normalized_view)
        } else {
            self.standard_view_encoding(&normalized_view)
        }
    }
}
