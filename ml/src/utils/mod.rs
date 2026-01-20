//! Internal utilities for quantum ML modules
//!
//! This module provides both internal utilities and public helpers for common QML tasks including:
//! - Data preprocessing and normalization
//! - Quantum state encoding
//! - Feature extraction
//! - Model evaluation metrics
//! - Data splitting utilities
//! - Model calibration

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::collections::HashMap;

// Type definitions
pub mod types;

// Functional modules
pub mod calibration;
pub mod encoding;
pub mod metrics;
pub mod preprocessing;
pub mod split;

// Tests
#[cfg(test)]
mod tests;

// Re-export all public items
pub use calibration::*;
pub use encoding::*;
pub use metrics::*;
pub use preprocessing::*;
pub use split::*;
pub use types::*;
