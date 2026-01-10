//! SciRS2 integration module.
//!
//! This module provides seamless integration with the SciRS2 scientific
//! computing ecosystem, following the SciRS2 POLICY.

pub mod complex;
pub mod ndarray;

// Re-export for convenience
pub use complex::*;
pub use ndarray::*;
