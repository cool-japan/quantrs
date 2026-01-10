//! Result types for holographic quantum error correction.
//!
//! This module contains result structures for error correction
//! and bulk reconstruction operations.

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};

use super::config::BulkReconstructionMethod;

/// Holographic quantum error correction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolographicQECResult {
    /// Whether the error correction was successful
    pub correction_successful: bool,
    /// Measured syndromes
    pub syndromes: Vec<f64>,
    /// Decoded error locations
    pub decoded_errors: Vec<usize>,
    /// Original error locations
    pub error_locations: Vec<usize>,
    /// Time taken for correction
    pub correction_time: std::time::Duration,
    /// Total entanglement entropy
    pub entanglement_entropy: f64,
    /// Holographic complexity
    pub holographic_complexity: f64,
}

/// Bulk reconstruction result
#[derive(Debug, Clone)]
pub struct BulkReconstructionResult {
    /// Reconstructed bulk state
    pub reconstructed_bulk: Array1<Complex64>,
    /// Reconstruction fidelity
    pub reconstruction_fidelity: f64,
    /// Time taken for reconstruction
    pub reconstruction_time: std::time::Duration,
    /// Reconstruction method used
    pub method_used: BulkReconstructionMethod,
}

/// Holographic QEC simulation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HolographicQECStats {
    /// Total number of error corrections performed
    pub total_corrections: u64,
    /// Number of successful corrections
    pub successful_corrections: u64,
    /// Total time spent on error correction
    pub correction_time: std::time::Duration,
    /// Average entanglement entropy
    pub average_entanglement_entropy: f64,
    /// Average holographic complexity
    pub average_holographic_complexity: f64,
    /// Total bulk reconstructions performed
    pub total_reconstructions: u64,
    /// Average reconstruction fidelity
    pub average_reconstruction_fidelity: f64,
}
