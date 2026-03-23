//! Simulator comparison framework for benchmarking and comparing quantum simulators.
//!
//! This module provides comprehensive tools for comparing different quantum simulators
//! across multiple dimensions including performance, accuracy, scalability, and features.

mod impls;
pub mod types;

// Re-export all public types for backward compatibility
pub use types::*;
