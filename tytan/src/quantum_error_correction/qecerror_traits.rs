//! # QECError - Trait Implementations
//!
//! This module contains trait implementations for `QECError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;
use scirs2_core::random::prelude::*;

use super::types::QECError;

impl std::fmt::Display for QECError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCodeParameters(msg) => {
                write!(f, "Invalid code parameters: {msg}")
            }
            Self::SyndromeExtractionFailed(msg) => {
                write!(f, "Syndrome extraction failed: {msg}")
            }
            Self::DecodingFailed(msg) => write!(f, "Decoding failed: {msg}"),
            Self::InsufficientCorrection(msg) => {
                write!(f, "Insufficient correction: {msg}")
            }
            Self::ThresholdExceeded(msg) => write!(f, "Threshold exceeded: {msg}"),
            Self::ResourceEstimationFailed(msg) => {
                write!(f, "Resource estimation failed: {msg}")
            }
            Self::NumericalError(msg) => write!(f, "Numerical error: {msg}"),
        }
    }
}

impl std::error::Error for QECError {}
