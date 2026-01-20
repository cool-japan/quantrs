//! Holographic Quantum Error Correction Framework
//!
//! This module provides a comprehensive implementation of holographic quantum error correction
//! using AdS/CFT correspondence, bulk-boundary duality, and emergent geometry from quantum
//! entanglement. This framework enables error correction through holographic principles,
//! where quantum information in a boundary theory is protected by geometry in the bulk.
//!
//! # Module Organization
//!
//! - `config`: Configuration types and enums
//! - `simulator`: Core simulator struct with initialization
//! - `encoding`: Holographic encoding matrix methods
//! - `error_correction`: Error correction and decoding methods
//! - `results`: Result types for operations
//! - `utils`: Utility functions
//! - `benchmark`: Benchmarking functions and tests

mod benchmark;
mod config;
mod encoding;
mod error_correction;
mod results;
mod simulator;
mod utils;

// Re-export public types
pub use benchmark::{benchmark_holographic_qec, HolographicQECBenchmarkResults};
pub use config::{BulkReconstructionMethod, HolographicCodeType, HolographicQECConfig};
pub use results::{BulkReconstructionResult, HolographicQECResult, HolographicQECStats};
pub use simulator::HolographicQECSimulator;
pub use utils::HolographicQECUtils;
