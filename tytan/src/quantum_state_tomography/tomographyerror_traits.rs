//! # TomographyError - Trait Implementations
//!
//! This module contains trait implementations for `TomographyError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::TomographyError;

impl std::fmt::Display for TomographyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientData(msg) => write!(f, "Insufficient data: {msg}"),
            Self::ReconstructionFailed(msg) => write!(f, "Reconstruction failed: {msg}"),
            Self::InvalidBasis(msg) => write!(f, "Invalid basis: {msg}"),
            Self::ConvergenceFailed(msg) => write!(f, "Convergence failed: {msg}"),
            Self::ValidationFailed(msg) => write!(f, "Validation failed: {msg}"),
            Self::NumericalError(msg) => write!(f, "Numerical error: {msg}"),
        }
    }
}

impl std::error::Error for TomographyError {}
