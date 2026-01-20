//! Utility functions for holographic quantum error correction.
//!
//! This module contains utility functions for calculating error thresholds,
//! estimating qubit requirements, and verifying code parameters.

use crate::error::{Result, SimulatorError};

use super::config::HolographicQECConfig;

/// Holographic QEC utilities
pub struct HolographicQECUtils;

impl HolographicQECUtils {
    /// Calculate holographic error correction threshold
    #[must_use]
    pub fn calculate_error_threshold(
        ads_radius: f64,
        central_charge: f64,
        boundary_qubits: usize,
    ) -> f64 {
        let holographic_factor = ads_radius / central_charge.sqrt();
        let qubit_factor = 1.0 / (boundary_qubits as f64).sqrt();

        holographic_factor * qubit_factor
    }

    /// Estimate bulk qubits needed for given boundary
    #[must_use]
    pub fn estimate_bulk_qubits(boundary_qubits: usize, encoding_ratio: f64) -> usize {
        ((boundary_qubits as f64) * encoding_ratio) as usize
    }

    /// Calculate optimal `AdS` radius for error correction
    #[must_use]
    pub fn calculate_optimal_ads_radius(
        boundary_qubits: usize,
        error_rate: f64,
        central_charge: f64,
    ) -> f64 {
        let boundary_factor = (boundary_qubits as f64).sqrt();
        let error_factor = 1.0 / error_rate.sqrt();
        let cft_factor = central_charge.sqrt();

        boundary_factor * error_factor / cft_factor
    }

    /// Verify holographic error correction code parameters
    pub fn verify_code_parameters(config: &HolographicQECConfig) -> Result<bool> {
        // Check AdS radius positivity
        if config.ads_radius <= 0.0 {
            return Err(SimulatorError::InvalidParameter(
                "AdS radius must be positive".to_string(),
            ));
        }

        // Check central charge positivity
        if config.central_charge <= 0.0 {
            return Err(SimulatorError::InvalidParameter(
                "Central charge must be positive".to_string(),
            ));
        }

        // Check qubit counts
        if config.boundary_qubits == 0 || config.bulk_qubits == 0 {
            return Err(SimulatorError::InvalidParameter(
                "Qubit counts must be positive".to_string(),
            ));
        }

        // Check bulk/boundary ratio
        if config.bulk_qubits < config.boundary_qubits {
            return Err(SimulatorError::InvalidParameter(
                "Bulk qubits should be at least as many as boundary qubits".to_string(),
            ));
        }

        Ok(true)
    }
}
