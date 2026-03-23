//! # TensorNetworkError - Trait Implementations
//!
//! This module contains trait implementations for `TensorNetworkError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::TensorNetworkError;

impl std::fmt::Display for TensorNetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {msg}"),
            Self::CompressionFailed(msg) => write!(f, "Compression failed: {msg}"),
            Self::OptimizationFailed(msg) => write!(f, "Optimization failed: {msg}"),
            Self::MemoryAllocationFailed(msg) => {
                write!(f, "Memory allocation failed: {msg}")
            }
            Self::SymmetryViolation(msg) => write!(f, "Symmetry violation: {msg}"),
            Self::ConvergenceFailed(msg) => write!(f, "Convergence failed: {msg}"),
            Self::NumericalError(msg) => write!(f, "Numerical error: {msg}"),
        }
    }
}

impl std::error::Error for TensorNetworkError {}
