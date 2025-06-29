//! Temporary parallel operations stubs to replace scirs2_core::parallel_ops
//! TODO: Replace with scirs2_core when regex dependency issue is fixed

// Re-export rayon traits and functions that were likely provided by scirs2_core::parallel_ops
pub use rayon::prelude::*;
