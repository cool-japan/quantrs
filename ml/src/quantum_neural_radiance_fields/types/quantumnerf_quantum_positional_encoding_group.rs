//! # QuantumNeRF - quantum_positional_encoding_group Methods
//!
//! This module contains method implementations for `QuantumNeRF`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::{QuantumEncodingOutput, QuantumPositionalEncodingType};

use super::quantumnerf_type::QuantumNeRF;

impl QuantumNeRF {
    /// Quantum positional encoding
    pub(super) fn quantum_positional_encoding(
        &self,
        position: &Array1<f64>,
    ) -> Result<QuantumEncodingOutput> {
        match self.quantum_positional_encoder.encoding_type {
            QuantumPositionalEncodingType::QuantumFourierEncoding => {
                self.quantum_fourier_encoding(position)
            }
            QuantumPositionalEncodingType::EntanglementBasedEncoding => {
                self.entanglement_based_encoding(position)
            }
            _ => self.standard_quantum_encoding(position),
        }
    }
}
