//! # QuantumAttentionMechanism - Trait Implementations
//!
//! This module contains trait implementations for `QuantumAttentionMechanism`.
//!
//! ## Implemented Traits
//!
//! - `From`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex32, Complex64};
use std::f64::consts::PI;

use super::types::QuantumAttentionMechanism;

impl From<QuantumContextEncoding> for QuantumAttentionMechanism {
    fn from(encoding: QuantumContextEncoding) -> Self {
        match encoding {
            QuantumContextEncoding::AmplitudeEncoding => {
                QuantumAttentionMechanism::SingleHead { attention_dim: 64 }
            }
            QuantumContextEncoding::AngleEncoding { .. } => QuantumAttentionMechanism::MultiHead {
                num_heads: 4,
                head_dim: 16,
            },
            QuantumContextEncoding::EntanglementEncoding { .. } => {
                QuantumAttentionMechanism::EntanglementBased {
                    entanglement_strength: 0.5,
                }
            }
            QuantumContextEncoding::QuantumFourierEncoding { .. } => {
                QuantumAttentionMechanism::QuantumFourier { frequency_bins: 32 }
            }
            _ => QuantumAttentionMechanism::SingleHead { attention_dim: 64 },
        }
    }
}
